"""Cross-role policy: agents dynamically choose aligner or miner role based on team needs.

Instead of fixed role assignment, each agent asks "what does the team need now?"
and picks skills from both aligner and miner skill sets accordingly.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, replace
from typing import Callable

from cogames.policy.aligner_agent import (
    AlignerPolicyImpl,
    SharedMap,
    _FRIENDLY_TERRITORY_DISTANCE,
    _HP_RETREAT_THRESHOLD,
)
from cogames.policy.llm_miner_policy import LLMMinerPlannerClient, LLMMinerPolicyImpl, LLMMinerState
from cogames.policy.llm_skills import MinerSkillImpl, MinerSkillState
from mettagrid.policy.policy import MultiAgentPolicy, StatefulAgentPolicy, StatefulPolicyImpl
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action
from mettagrid.simulator.interface import AgentObservation

logger = logging.getLogger("cogames.policy.cross_role")

Coord = tuple[int, int]

CROSS_ROLE_SKILLS = {
    "gear_up_aligner": "Route to the aligner station and acquire aligner gear.",
    "gear_up_miner": "Route to the miner station and acquire miner gear.",
    "get_heart": "Route to the hub and obtain a heart (requires aligner gear).",
    "align_neutral": "Route to a neutral or enemy junction and align it (requires aligner gear + heart).",
    "mine_until_full": "Route to extractors and mine until cargo full (requires miner gear).",
    "deposit_to_hub": "Route to hub and deposit carried resources (requires miner gear + carried resources).",
    "explore": "Explore the map to discover new junctions, extractors, and routes.",
    "unstuck": "Execute a short escape pattern to break navigation deadlocks.",
}

_SKILL_ALIASES = {"unstick": "unstuck"}


def _parse_cross_role_skill(text: str) -> tuple[str | None, str]:
    text = text.strip()
    if not text:
        return None, "empty response"
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        skill = text.splitlines()[0].strip()
        skill = _SKILL_ALIASES.get(skill, skill)
        return (skill if skill in CROSS_ROLE_SKILLS else None, "non-json response")
    if not isinstance(payload, dict):
        return None, "response was not a JSON object"
    skill = payload.get("skill")
    reason = payload.get("reason", "")
    if not isinstance(skill, str):
        return None, "missing skill field"
    skill = _SKILL_ALIASES.get(skill, skill)
    return (skill if skill in CROSS_ROLE_SKILLS else None, str(reason))


_ALIGNER_SKILLS = {k: v for k, v in CROSS_ROLE_SKILLS.items() if k in {"get_heart", "align_neutral", "explore", "unstuck"}}
_MINER_SKILLS = {k: v for k, v in CROSS_ROLE_SKILLS.items() if k in {"mine_until_full", "deposit_to_hub", "explore", "unstuck"}}


def build_cross_role_prompt(
    *,
    current_gear: str,
    has_heart: bool,
    carried_resources: int,
    return_load: int,
    hub_visible: bool,
    known_hubs: int,
    known_neutral_junctions: int,
    known_friendly_junctions: int,
    known_enemy_junctions: int,
    known_extractors: int,
    current_skill: str | None,
    no_move_steps: int,
    recent_events: list[str],
    team_aligners: int = 0,
    team_miners: int = 0,
    team_size: int = 8,
    preferred_role: str = "",
) -> str:
    # Select skills relevant to current gear (no gear switching via LLM)
    if current_gear == "aligner":
        skill_set = _ALIGNER_SKILLS
        role_hint = "You are in ALIGNER mode. Your job: get hearts, then align junctions."
        preconditions = (
            "- get_heart: requires current_gear == 'aligner' AND hub accessible\n"
            "- align_neutral: requires current_gear == 'aligner' AND has_heart == true AND known_alignable_junctions > 0\n"
        )
    elif current_gear == "miner":
        skill_set = _MINER_SKILLS
        role_hint = "You are in MINER mode. Your job: mine resources, then deposit to hub."
        preconditions = (
            "- mine_until_full: requires current_gear == 'miner' AND known_extractors > 0\n"
            "- deposit_to_hub: requires current_gear == 'miner' AND carried_resources > 0\n"
        )
    else:
        # No valid gear (none, scrambler, scout) — include gear_up skills so LLM can recover.
        # Scrambler/scout gear is contamination from wrong station; agent needs to navigate to
        # a gear station to acquire the correct gear.
        skill_set = {
            "gear_up_aligner": CROSS_ROLE_SKILLS["gear_up_aligner"],
            "gear_up_miner": CROSS_ROLE_SKILLS["gear_up_miner"],
            "explore": CROSS_ROLE_SKILLS["explore"],
            "unstuck": CROSS_ROLE_SKILLS["unstuck"],
        }
        if current_gear in ("scrambler", "scout"):
            if preferred_role:
                role_hint = (
                    f"You have {current_gear} gear (wrong gear — contamination). "
                    f"Your PREFERRED ROLE is {preferred_role.upper()}. "
                    f"You MUST use gear_up_{preferred_role} to restore your role. "
                    f"Do NOT switch roles."
                )
            else:
                role_hint = (
                    f"You have {current_gear} gear (wrong gear — contamination). "
                    "Navigate to the aligner or miner station to acquire the correct gear."
                )
        else:
            if preferred_role:
                role_hint = (
                    f"You have no gear yet. Your PREFERRED ROLE is {preferred_role.upper()}. "
                    f"Use gear_up_{preferred_role} to acquire your role gear."
                )
            else:
                role_hint = "You have no gear yet — explore to find a gear station, then gear up."
        preconditions = ""

    skills_text = "\n".join(f"- {name}: {desc}" for name, desc in skill_set.items())
    events_text = "\n".join(f"- {e}" for e in recent_events[-6:]) or "- none"

    return (
        f"You are a cog agent in CoGames. {role_hint}\n"
        "Your team wins by aligning and holding junctions.\n\n"
        f"Preconditions:\n{preconditions}"
        "Respond as JSON: {\"skill\": \"<skill_name>\", \"reason\": \"...\"}\n\n"
        f"Available skills:\n{skills_text}\n\n"
        f"Team state:\n"
        f"- team_aligners: {team_aligners}\n"
        f"- team_miners: {team_miners}\n"
        f"- team_size: {team_size}\n\n"
        f"Agent state:\n"
        f"- current_gear: {current_gear}\n"
        f"- has_heart: {has_heart}\n"
        f"- carried_resources: {carried_resources}\n"
        f"- return_load: {return_load}\n"
        f"- hub_visible: {hub_visible}\n"
        f"- known_hubs: {known_hubs}\n"
        f"- known_neutral_junctions: {known_neutral_junctions}\n"
        f"- known_friendly_junctions: {known_friendly_junctions}\n"
        f"- known_enemy_junctions: {known_enemy_junctions}\n"
        f"- known_extractors: {known_extractors}\n"
        f"- known_alignable_junctions: {known_neutral_junctions + known_enemy_junctions}\n"
        f"- current_skill: {current_skill or 'none'}\n"
        f"- no_move_steps: {no_move_steps}\n"
        f"\nRecent events:\n{events_text}\n"
    )


@dataclass
class CrossRoleState:
    """Combined state for agents that can play both aligner and miner roles.

    Has all fields required by AlignerPolicyImpl AND MinerSkillImpl methods
    (duck typing approach — both can operate on this state).
    """

    # StarterCogState base fields
    wander_direction_index: int = 0
    wander_steps_remaining: int = 0
    last_mode: str = "bootstrap"

    # Map memory (shared via SharedMap)
    known_free_cells: set[Coord] = field(default_factory=set)
    blocked_cells: set[Coord] = field(default_factory=set)
    move_blocked_cells: set[Coord] = field(default_factory=set)
    known_hubs: set[Coord] = field(default_factory=set)

    # Aligner-specific structures
    known_aligner_stations: set[Coord] = field(default_factory=set)
    known_neutral_junctions: set[Coord] = field(default_factory=set)
    known_friendly_junctions: set[Coord] = field(default_factory=set)
    known_enemy_junctions: set[Coord] = field(default_factory=set)
    known_hazard_stations: set[Coord] = field(default_factory=set)
    blacklisted_junctions: set[Coord] = field(default_factory=set)

    # Miner-specific structures
    known_miner_stations: set[Coord] = field(default_factory=set)
    known_extractors: set[Coord] = field(default_factory=set)
    remembered_hub_row_from_spawn: int | None = None
    remembered_hub_col_from_spawn: int | None = None

    # Move-failure tracking
    last_pos: Coord | None = None
    last_move_target: Coord | None = None

    # LLM planning state
    current_skill: str | None = None
    current_reason: str = ""
    skill_steps: int = 0
    no_move_steps: int = 0
    no_progress_on_target_steps: int = 0
    recent_events: list[str] = field(default_factory=list)

    # Aligner LLM tracking
    last_has_heart: bool = False
    last_friendly_junctions: int = 0
    consecutive_unstuck: int = 0
    explore_start_junctions: int = 0
    align_neutral_timeouts: int = 0
    get_heart_timeouts: int = 0
    max_hp_seen: int = 0
    retreating: bool = False

    # Miner LLM tracking
    last_carried_total: int = 0
    explore_start_extractors: int = 0

    # Gear acquisition tracking (for retry + fallback logic)
    gear_up_failures: int = 0
    gear_up_completed: bool = False  # True once any gear_up succeeds; prevents retry after accidental gear change

    # Gear test harness (issue-12)
    episode_step: int = 0  # incremented once per game step
    phase: int = 1  # 1 = initial gear acquisition, 2 = switched gear acquisition
    phase_preferred_gear: str = ""  # overrides preferred_initial_gear in phase 2


class CrossRolePolicyImpl(StatefulPolicyImpl[CrossRoleState]):
    """Agent that can dynamically choose aligner or miner skills based on team needs."""

    _UNSTUCK_DIRECTIONS = ("north", "east", "south", "west")

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        agent_id: int,
        planner: LLMMinerPlannerClient,
        stuck_threshold: int,
        unstuck_horizon: int,
        return_load: int,
        shared_map: SharedMap | None = None,
        preferred_initial_gear: str = "",
        phase_switch_step: int = 0,
    ) -> None:
        # Aligner impl handles aligner-specific skills (gear_up aligner station, get_heart, align_neutral)
        self._aligner = AlignerPolicyImpl(policy_env_info, agent_id, shared_map=shared_map)
        # Miner impl handles miner-specific skills (gear_up miner station, mine_until_full, deposit_to_hub)
        self._miner = MinerSkillImpl(policy_env_info, agent_id, return_load=return_load, shared_map=shared_map)
        self._planner = planner
        self._stuck_threshold = stuck_threshold
        self._unstuck_horizon = unstuck_horizon
        self._return_load = return_load
        self._shared_map = shared_map
        self._preferred_initial_gear = preferred_initial_gear
        self._phase_switch_step = phase_switch_step
        # Store hub_tags for hub visibility check
        self._hub_tags = self._aligner._starter._resolve_tag_ids(["hub"])

    def initial_agent_state(self) -> CrossRoleState:
        state = CrossRoleState()
        self._bind_shared_map(state)
        return state

    def _bind_shared_map(self, state: CrossRoleState) -> None:
        sm = self._shared_map
        if sm is None:
            return
        state.known_free_cells = sm.known_free_cells
        state.blocked_cells = sm.blocked_cells
        state.move_blocked_cells = sm.move_blocked_cells
        state.known_hubs = sm.known_hubs
        state.known_aligner_stations = sm.known_aligner_stations
        state.known_miner_stations = sm.known_miner_stations
        state.known_hazard_stations = sm.known_hazard_stations
        state.known_extractors = sm.known_extractors
        state.known_neutral_junctions = sm.known_neutral_junctions
        state.known_friendly_junctions = sm.known_friendly_junctions
        state.known_enemy_junctions = sm.known_enemy_junctions

    def _copy_with_shared(self, state: CrossRoleState) -> CrossRoleState:
        """Return state with shared map fields re-bound (after delegate calls may have returned new state)."""
        sm = self._shared_map
        if sm is None:
            return state
        return replace(
            state,
            known_free_cells=sm.known_free_cells,
            blocked_cells=sm.blocked_cells,
            move_blocked_cells=sm.move_blocked_cells,
            known_hubs=sm.known_hubs,
            known_aligner_stations=sm.known_aligner_stations,
            known_miner_stations=sm.known_miner_stations,
            known_hazard_stations=sm.known_hazard_stations,
            known_extractors=sm.known_extractors,
            known_neutral_junctions=sm.known_neutral_junctions,
            known_friendly_junctions=sm.known_friendly_junctions,
            known_enemy_junctions=sm.known_enemy_junctions,
        )

    def _event(self, state: CrossRoleState, message: str) -> None:
        state.recent_events.append(message)
        del state.recent_events[:-10]

    def _current_gear(self, obs: AgentObservation) -> str:
        gear = self._aligner._current_gear(obs)
        return gear if gear else "none"

    def _inventory_count(self, obs: AgentObservation, item: str) -> int:
        return self._aligner._inventory_count(obs, item)

    def _carried_total(self, obs: AgentObservation) -> int:
        return self._miner._carried_total(obs)

    def _feature_value(self, obs: AgentObservation, feature_name: str) -> int | None:
        for token in obs.tokens:
            if token.feature.name == feature_name:
                return int(token.value)
        return None

    def _hub_visible(self, obs: AgentObservation) -> bool:
        return self._aligner._starter._closest_tag_location(obs, self._hub_tags) is not None

    def _current_abs(self, obs: AgentObservation) -> Coord:
        return self._aligner._spawn_offset(obs)

    def _known_alignable_junctions(self, state: CrossRoleState) -> set[Coord]:
        # v14: include both neutral AND enemy junctions so aligners re-align captured ones
        # without waiting until all neutral junctions are exhausted.
        return (state.known_neutral_junctions | state.known_enemy_junctions) - state.blacklisted_junctions

    def _update_map_memory(self, obs: AgentObservation, state: CrossRoleState) -> Coord:
        """Update map from both aligner and miner perspectives."""
        # Use aligner map memory (handles junctions, aligner stations, hazards)
        current_abs = self._aligner._update_map_memory(obs, state)
        # Additionally update miner-specific structures (extractors, miner stations)
        # We reuse the miner's _update_map_memory but pass our CrossRoleState (duck typing)
        self._miner._update_map_memory(obs, state)
        return current_abs

    def _update_progress(self, obs: AgentObservation, state: CrossRoleState) -> None:
        has_heart = self._inventory_count(obs, "heart") > 0
        friendly_count = len(state.known_friendly_junctions)
        carried_total = self._carried_total(obs)
        current_abs = self._current_abs(obs)

        if state.current_skill == "get_heart" and has_heart and not state.last_has_heart:
            self._event(state, "acquired a heart")
        if state.current_skill == "align_neutral" and friendly_count > state.last_friendly_junctions:
            self._event(state, f"friendly junction count increased {state.last_friendly_junctions}→{friendly_count}")
        if state.current_skill == "deposit_to_hub" and carried_total < state.last_carried_total:
            self._event(state, f"deposited cargo {state.last_carried_total}→{carried_total}")
        if state.current_skill == "mine_until_full" and carried_total > state.last_carried_total:
            self._event(state, f"cargo increased {state.last_carried_total}→{carried_total}")

        state.last_has_heart = has_heart
        state.last_friendly_junctions = friendly_count
        state.last_carried_total = carried_total

        last_action_move = self._feature_value(obs, "last_action_move")
        gear = self._current_gear(obs)

        near_hub = any(
            abs(current_abs[0] - h[0]) + abs(current_abs[1] - h[1]) <= 1
            for h in state.known_hubs
        )
        near_aligner_station = any(
            abs(current_abs[0] - s[0]) + abs(current_abs[1] - s[1]) <= 1
            for s in state.known_aligner_stations
        )
        near_miner_station = any(
            abs(current_abs[0] - s[0]) + abs(current_abs[1] - s[1]) <= 1
            for s in state.known_miner_stations
        )

        made_progress = (
            (state.current_skill == "get_heart" and has_heart and not state.last_has_heart)
            or (state.current_skill == "align_neutral" and friendly_count > state.last_friendly_junctions)
            or (state.current_skill == "gear_up_aligner" and gear == "aligner")
            or (state.current_skill == "gear_up_miner" and gear == "miner")
            or (state.current_skill == "deposit_to_hub" and carried_total < state.last_carried_total)
            or (state.current_skill == "mine_until_full" and carried_total > state.last_carried_total)
        )
        stationary_on_valid_target = (
            (state.current_skill == "get_heart" and near_hub)
            or (state.current_skill == "align_neutral" and current_abs in self._known_alignable_junctions(state))
            or (state.current_skill == "gear_up_aligner" and near_aligner_station)
            or (state.current_skill == "gear_up_miner" and near_miner_station)
            or (state.current_skill in {"mine_until_full", "deposit_to_hub"} and near_hub)
        )

        if made_progress:
            state.no_move_steps = 0
            state.no_progress_on_target_steps = 0
        elif stationary_on_valid_target and not made_progress:
            state.no_move_steps = 0
            state.no_progress_on_target_steps += 1
        elif state.current_skill is not None and last_action_move == 0:
            state.no_move_steps += 1
            state.no_progress_on_target_steps = 0
        else:
            state.no_move_steps = 0
            state.no_progress_on_target_steps = 0

    def _team_gear_counts(self) -> tuple[int, int]:
        """Return (num_aligners, num_miners) across the team from shared map."""
        sm = self._shared_map
        if sm is None or not hasattr(sm, "agent_gears"):
            return 0, 0
        gears = sm.agent_gears.values()
        return sum(1 for g in gears if g == "aligner"), sum(1 for g in gears if g == "miner")

    def _plan_skill(self, obs: AgentObservation, state: CrossRoleState) -> None:
        gear = self._current_gear(obs)
        has_heart = self._inventory_count(obs, "heart") > 0
        carried = self._carried_total(obs)
        known_alignable = self._known_alignable_junctions(state)

        # Update shared gear tracking
        if self._shared_map is not None and hasattr(self._shared_map, "agent_gears"):
            self._shared_map.agent_gears[obs.agent_id] = gear

        # Effective preferred gear: phase_preferred_gear (set at phase switch) or original preference
        effective_preferred = state.phase_preferred_gear or self._preferred_initial_gear

        # Bootstrap: acquire preferred gear.
        # Phase 1: fire when gear=="none" and gear_up_completed==False (initial acquisition)
        # Phase 2: fire when gear!=effective_preferred and gear_up_completed==False (gear switch)
        # v17: LLM prompt includes gear_up_aligner/miner for scrambler/scout so agents can
        # self-correct without bootstrap (but bootstrap is faster and more reliable).
        needs_gear_up = (
            effective_preferred
            and not state.gear_up_completed
            and (gear == "none" or (state.phase == 2 and gear != effective_preferred and gear in ("aligner", "miner")))
        )
        if needs_gear_up:
            failures = state.gear_up_failures
            if failures == 0:
                bootstrap_gear = effective_preferred
                reason = f"phase{state.phase} gear target: {bootstrap_gear} (attempt 1)"
            elif failures == 1:
                bootstrap_gear = "miner" if effective_preferred == "aligner" else "aligner"
                reason = f"phase{state.phase} fallback gear: {bootstrap_gear} (preferred {effective_preferred} failed)"
            else:
                bootstrap_gear = ""  # 2+ failures: let LLM choose

            if bootstrap_gear:
                skill = f"gear_up_{bootstrap_gear}"
                logger.info("agent=%s bootstrap_skill=%s failures=%d phase=%d", obs.agent_id, skill, failures, state.phase)
                if skill in CROSS_ROLE_SKILLS:
                    state.current_skill = skill
                    state.current_reason = reason
                    state.skill_steps = 0
                    state.no_move_steps = 0
                    state.no_progress_on_target_steps = 0
                    self._event(state, f"planner selected {skill}: {reason}")
                    return

        team_aligners, team_miners = self._team_gear_counts()
        team_size = max(1, len(self._shared_map.agent_gears) if self._shared_map and hasattr(self._shared_map, "agent_gears") else 8)

        prompt = build_cross_role_prompt(
            current_gear=gear,
            has_heart=has_heart,
            carried_resources=carried,
            return_load=self._return_load,
            hub_visible=self._hub_visible(obs),
            known_hubs=len(state.known_hubs),
            known_neutral_junctions=len(state.known_neutral_junctions),
            known_friendly_junctions=len(state.known_friendly_junctions),
            known_enemy_junctions=len(state.known_enemy_junctions),
            known_extractors=len(state.known_extractors),
            current_skill=state.current_skill,
            no_move_steps=state.no_move_steps,
            recent_events=state.recent_events,
            team_aligners=team_aligners,
            team_miners=team_miners,
            team_size=team_size,
            preferred_role=state.phase_preferred_gear or self._preferred_initial_gear,
        )
        logger.info("agent=%s cross_role_prompt=%s", obs.agent_id, prompt.replace("\n", " | "))
        started_at = time.perf_counter()
        text = ""
        for attempt in range(3):
            try:
                text = self._planner.complete(prompt)
                break
            except Exception as exc:
                wait_s = 3.0 * (attempt + 1)
                logger.warning("agent=%s LLM attempt %d failed (%s), retrying in %.0fs", obs.agent_id, attempt + 1, exc, wait_s)
                if attempt < 2:
                    time.sleep(wait_s)
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        logger.info(
            "agent=%s cross_role_response_ms=%.1f response=%s",
            obs.agent_id, latency_ms, text.replace("\n", " "),
        )
        skill, reason = _parse_cross_role_skill(text)
        was_stuck = bool(
            state.recent_events and (
                "exited as stuck" in state.recent_events[-1]
                or "exited as stale" in state.recent_events[-1]
                or "timed out after" in state.recent_events[-1]
            )
        )

        # Scripted fallback if LLM returns invalid skill
        if skill is None:
            if gear == "none":
                skill = "gear_up_aligner" if len(known_alignable) >= len(state.known_extractors) else "gear_up_miner"
            elif gear == "aligner":
                if not has_heart and state.known_hubs:
                    skill = "get_heart"
                elif has_heart and known_alignable:
                    skill = "align_neutral"
                else:
                    skill = "explore"
            elif gear == "miner":
                if carried >= self._return_load:
                    skill = "deposit_to_hub"
                elif state.known_extractors:
                    skill = "mine_until_full"
                else:
                    skill = "explore"
            else:
                skill = "explore"
            reason = f"scripted fallback ({reason})"

        # Precondition enforcement
        # Must have correct gear for role-specific skills
        if skill == "get_heart" and gear != "aligner":
            skill = "gear_up_aligner"
            reason = f"overrode to gear_up_aligner: need aligner gear for get_heart (current={gear})"
        if skill == "align_neutral" and gear != "aligner":
            skill = "gear_up_aligner"
            reason = f"overrode to gear_up_aligner: need aligner gear for align_neutral"
        if skill == "align_neutral" and gear == "aligner" and not has_heart:
            skill = "get_heart"
            reason = "overrode to get_heart: need heart for align_neutral"
        if skill == "align_neutral" and gear == "aligner" and has_heart and not known_alignable:
            skill = "explore"
            reason = "overrode to explore: no alignable junctions known"
        if skill == "mine_until_full" and gear != "miner":
            skill = "gear_up_miner"
            reason = f"overrode to gear_up_miner: need miner gear for mining (current={gear})"
        if skill == "deposit_to_hub" and gear != "miner":
            skill = "gear_up_miner"
            reason = f"overrode to gear_up_miner: need miner gear for deposit (current={gear})"
        if skill == "deposit_to_hub" and gear == "miner" and carried == 0:
            skill = "mine_until_full" if state.known_extractors else "explore"
            reason = "overrode to mine: no cargo to deposit"
        if skill == "gear_up_aligner" and gear == "aligner":
            if has_heart and known_alignable:
                skill = "align_neutral"
                reason = "overrode: already have aligner gear, have heart and target"
            elif not has_heart and state.known_hubs:
                skill = "get_heart"
                reason = "overrode: already have aligner gear, need heart"
            else:
                skill = "explore"
                reason = "overrode: already have aligner gear"
        if skill == "gear_up_miner" and gear == "miner":
            if carried >= self._return_load:
                skill = "deposit_to_hub"
                reason = "overrode: already have miner gear, cargo full"
            elif state.known_extractors:
                skill = "mine_until_full"
                reason = "overrode: already have miner gear"
            else:
                skill = "explore"
                reason = "overrode: already have miner gear, no extractors"

        # Prevent consecutive unstuck loops
        if skill == "unstuck":
            state.consecutive_unstuck += 1
        else:
            state.consecutive_unstuck = 0
        if state.consecutive_unstuck >= 2 and skill == "unstuck":
            skill = "explore"
            reason = f"overrode unstuck to explore after {state.consecutive_unstuck} consecutive unstuck"
            state.consecutive_unstuck = 0

        if skill in {"explore", "gear_up_aligner", "gear_up_miner"}:
            state.explore_start_junctions = len(state.known_neutral_junctions)
            state.explore_start_extractors = len(state.known_extractors)

        state.current_skill = skill
        state.current_reason = reason
        state.skill_steps = 0
        state.no_move_steps = 0
        state.no_progress_on_target_steps = 0
        self._event(state, f"planner selected {skill}: {reason}")

    def _maybe_finish_skill(self, obs: AgentObservation, state: CrossRoleState) -> None:
        gear = self._current_gear(obs)
        has_heart = self._inventory_count(obs, "heart") > 0
        carried = self._carried_total(obs)
        friendly_count = len(state.known_friendly_junctions)

        if state.current_skill == "gear_up_aligner" and gear == "aligner" and state.skill_steps > 0:
            self._event(state, "gear_up_aligner completed")
            state.gear_up_completed = True
            state.current_skill = None
        elif state.current_skill == "gear_up_miner" and gear == "miner" and state.skill_steps > 0:
            self._event(state, "gear_up_miner completed")
            state.gear_up_completed = True
            state.current_skill = None
        elif state.current_skill == "get_heart" and has_heart and state.skill_steps > 0:
            self._event(state, "get_heart completed: acquired heart")
            state.get_heart_timeouts = 0
            state.current_skill = None
        elif state.current_skill == "align_neutral" and not has_heart and state.skill_steps > 0:
            self._event(state, "align_neutral completed: heart spent")
            state.align_neutral_timeouts = 0
            state.current_skill = None
        elif state.current_skill == "mine_until_full" and carried >= self._return_load:
            self._event(state, f"mine_until_full completed: cargo={carried}")
            state.current_skill = None
        elif state.current_skill == "deposit_to_hub" and carried == 0:
            self._event(state, "deposit_to_hub completed")
            state.current_skill = None
        elif state.current_skill == "explore" and (
            len(state.known_neutral_junctions) > state.explore_start_junctions
            or len(state.known_extractors) > state.explore_start_extractors
        ):
            new_junctions = len(state.known_neutral_junctions) - state.explore_start_junctions
            new_extractors = len(state.known_extractors) - state.explore_start_extractors
            self._event(state, f"explore completed: +{new_junctions} junctions, +{new_extractors} extractors")
            state.current_skill = None
        elif state.current_skill == "unstuck" and state.skill_steps >= self._unstuck_horizon:
            self._event(state, "unstuck finished horizon")
            state.current_skill = None
        elif state.current_skill in {"gear_up_aligner", "gear_up_miner"} and state.skill_steps >= self._stuck_threshold * 10:
            state.gear_up_failures += 1
            self._event(state, f"{state.current_skill} timed out after {state.skill_steps} steps (failure #{state.gear_up_failures})")
            state.current_skill = None
        elif state.current_skill in {"mine_until_full", "deposit_to_hub"} and state.skill_steps >= self._stuck_threshold * 20:
            # Worker skills get a much longer timeout (400 steps with default threshold=20).
            # The baseline miner has NO timeout at all — skills run until completion.
            # Short timeouts waste 65% of step capacity in failed navigation to far hub/extractors.
            self._event(state, f"{state.current_skill} timed out after {state.skill_steps} steps (long-timeout)")
            state.current_skill = None
        elif state.current_skill in {"get_heart", "align_neutral"} and state.skill_steps >= self._stuck_threshold * 5:
            if state.current_skill == "align_neutral":
                state.align_neutral_timeouts += 1
                if state.align_neutral_timeouts >= 1:
                    current_abs = self._current_abs(obs)
                    non_bl_neutral = state.known_neutral_junctions - state.blacklisted_junctions
                    non_bl_enemy = state.known_enemy_junctions - state.blacklisted_junctions
                    targets = non_bl_neutral or non_bl_enemy
                    if targets:
                        stuck = self._aligner._nearest_known(current_abs, targets)
                        if stuck:
                            state.blacklisted_junctions.add(stuck)
                            state.known_neutral_junctions.discard(stuck)
                            state.known_enemy_junctions.discard(stuck)
                            state.align_neutral_timeouts = 0
            elif state.current_skill == "get_heart":
                state.get_heart_timeouts += 1
            self._event(state, f"{state.current_skill} timed out after {state.skill_steps} steps")
            state.current_skill = None
        elif state.current_skill is not None and state.no_move_steps >= self._stuck_threshold:
            if state.current_skill in {"gear_up_aligner", "gear_up_miner"}:
                state.gear_up_failures += 1
            self._event(state, f"{state.current_skill} exited as stuck after {state.no_move_steps} blocked steps")
            state.current_skill = None
        elif state.current_skill is not None and state.no_progress_on_target_steps >= self._stuck_threshold:
            # Remove depleted extractors
            if state.current_skill == "mine_until_full":
                current_abs = self._current_abs(obs)
                if current_abs in state.known_extractors:
                    state.known_extractors.discard(current_abs)
                    self._event(state, f"removed depleted extractor at {current_abs}")
            if state.current_skill in {"gear_up_aligner", "gear_up_miner"}:
                state.gear_up_failures += 1
            self._event(state, f"{state.current_skill} exited as stale after {state.no_progress_on_target_steps} steps")
            state.current_skill = None

    def _unstuck_move(self, state: CrossRoleState) -> tuple[Action, CrossRoleState]:
        state.last_mode = "unstuck"
        direction = self._UNSTUCK_DIRECTIONS[state.wander_direction_index % len(self._UNSTUCK_DIRECTIONS)]
        state.wander_direction_index = (state.wander_direction_index + 1) % len(self._UNSTUCK_DIRECTIONS)
        return self._aligner._starter._action(f"move_{direction}"), state

    def _gear_up_miner_safe(self, obs: AgentObservation, state: CrossRoleState, current_abs: Coord) -> tuple[Action, CrossRoleState]:
        """Gear up to miner using hazard-aware navigation (avoids other gear stations).

        Issue-12 fix: the original miner._gear_up uses _move_toward_target which falls back
        to optimistic BFS without hazard avoidance when the primary BFS fails. This causes
        agents to route through scout/scrambler/aligner stations, accidentally equipping
        wrong gear. This version uses the aligner's _navigate_to_station with avoid_hazards=True.
        """
        visible_target = self._miner._closest_visible_location(obs, self._miner._miner_station_tags)
        if visible_target is not None:
            target_abs = self._miner._visible_abs_cell(current_abs, visible_target)
            direction = self._aligner._navigate_to_station(state, current_abs, target_abs, avoid_hazards=True)
            if direction is not None:
                return self._aligner._starter._action(f"move_{direction}"), state
            action, next_state = self._aligner._greedy_move_toward_abs(state, current_abs, target_abs)
            return action, state
        target_abs = self._aligner._nearest_known(current_abs, state.known_miner_stations)
        if target_abs is None:
            if state.known_hubs:
                action, base_state = self._aligner._explore_near_hub(obs, state)
                return action, self._copy_with_shared(replace(state,
                    wander_direction_index=base_state.wander_direction_index,
                    wander_steps_remaining=base_state.wander_steps_remaining,
                    last_mode=base_state.last_mode,
                ))
            action, base_state = self._aligner._explore(obs, state)
            return action, self._copy_with_shared(replace(state,
                wander_direction_index=base_state.wander_direction_index,
                wander_steps_remaining=base_state.wander_steps_remaining,
                last_mode=base_state.last_mode,
            ))
        direction = self._aligner._navigate_to_station(state, current_abs, target_abs, avoid_hazards=True)
        if direction is not None:
            return self._aligner._starter._action(f"move_{direction}"), state
        action, next_state = self._aligner._greedy_move_toward_abs(state, current_abs, target_abs)
        return action, state

    def _handle_phase_switch(self, obs: AgentObservation, state: CrossRoleState) -> None:
        """At phase switch step, flip preferred gear and reset gear acquisition state."""
        gear = self._current_gear(obs)
        effective_preferred = state.phase_preferred_gear or self._preferred_initial_gear
        new_preferred = "miner" if effective_preferred == "aligner" else "aligner"
        logger.info(
            "PHASE_SWITCH agent=%d step=%d current_gear=%s old_preferred=%s new_preferred=%s",
            obs.agent_id, state.episode_step, gear, effective_preferred, new_preferred,
        )
        state.phase = 2
        state.phase_preferred_gear = new_preferred
        state.gear_up_completed = False
        state.gear_up_failures = 0
        state.current_skill = None

    def step_with_state(self, obs: AgentObservation, state: CrossRoleState) -> tuple[Action, CrossRoleState]:
        state.episode_step += 1

        # Phase switch: flip preferred gear at phase_switch_step
        if self._phase_switch_step > 0 and state.episode_step == self._phase_switch_step and state.phase == 1:
            self._handle_phase_switch(obs, state)

        # Log gear state at key steps for metric computation
        if self._phase_switch_step > 0 and state.episode_step in (self._phase_switch_step, self._phase_switch_step * 2):
            gear = self._current_gear(obs)
            effective_preferred = state.phase_preferred_gear or self._preferred_initial_gear
            logger.info(
                "GEAR_STATE agent=%d step=%d gear=%s phase=%d intended=%s",
                obs.agent_id, state.episode_step, gear, state.phase, effective_preferred,
            )

        current_abs = self._update_map_memory(obs, state)
        self._update_progress(obs, state)
        self._maybe_finish_skill(obs, state)

        if state.current_skill is None:
            self._plan_skill(obs, state)

        # Navigation shake to break stuck loops
        if state.current_skill not in {None, "unstuck"} and state.no_move_steps >= 5 and state.no_move_steps % 3 == 0:
            action, state = self._unstuck_move(state)
            state.skill_steps += 1
            return action, state

        skill = state.current_skill

        if skill == "gear_up_aligner":
            action, base_state = self._aligner._gear_up(obs, state, current_abs)
            state = self._copy_with_shared(replace(state,
                wander_direction_index=base_state.wander_direction_index,
                wander_steps_remaining=base_state.wander_steps_remaining,
                last_mode=base_state.last_mode,
                last_pos=base_state.last_pos,
                last_move_target=base_state.last_move_target,
            ))

        elif skill == "gear_up_miner":
            # Use safe version: hazard-aware navigation (avoids other gear stations)
            action, base_state = self._gear_up_miner_safe(obs, state, current_abs)
            state = self._copy_with_shared(replace(state,
                wander_direction_index=base_state.wander_direction_index,
                wander_steps_remaining=base_state.wander_steps_remaining,
                last_mode=base_state.last_mode,
                last_pos=base_state.last_pos,
                last_move_target=getattr(base_state, 'last_move_target', state.last_move_target),
            ))

        elif skill == "get_heart":
            action, base_state = self._aligner._get_heart(obs, state, current_abs)
            state = self._copy_with_shared(replace(state,
                wander_direction_index=base_state.wander_direction_index,
                wander_steps_remaining=base_state.wander_steps_remaining,
                last_mode=base_state.last_mode,
                last_pos=base_state.last_pos,
                last_move_target=base_state.last_move_target,
            ))

        elif skill == "align_neutral":
            # v14: merge enemy junctions into neutral pool so aligner routes to nearest
            # of (neutral ∪ enemy) instead of neutral-first-then-enemy-as-fallback.
            # This makes aligners defend recaptured junctions immediately rather than
            # continuing to chase new neutral ones, increasing average hold time.
            merged_neutral = state.known_neutral_junctions | state.known_enemy_junctions
            align_state = replace(state, known_neutral_junctions=merged_neutral)
            action, base_state = self._aligner._align_neutral(obs, align_state, current_abs)
            state = self._copy_with_shared(replace(state,
                wander_direction_index=base_state.wander_direction_index,
                wander_steps_remaining=base_state.wander_steps_remaining,
                last_mode=base_state.last_mode,
                last_pos=base_state.last_pos,
                last_move_target=base_state.last_move_target,
            ))

        elif skill == "mine_until_full":
            action, base_state = self._miner._mine_until_full(obs, state)
            state = self._copy_with_shared(replace(state,
                wander_direction_index=base_state.wander_direction_index,
                wander_steps_remaining=base_state.wander_steps_remaining,
                last_mode=base_state.last_mode,
                remembered_hub_row_from_spawn=base_state.remembered_hub_row_from_spawn,
                remembered_hub_col_from_spawn=base_state.remembered_hub_col_from_spawn,
                last_pos=base_state.last_pos,
                last_move_target=base_state.last_move_target,
            ))

        elif skill == "deposit_to_hub":
            action, base_state = self._miner._deposit_to_hub(obs, state)
            state = self._copy_with_shared(replace(state,
                wander_direction_index=base_state.wander_direction_index,
                wander_steps_remaining=base_state.wander_steps_remaining,
                last_mode=base_state.last_mode,
                remembered_hub_row_from_spawn=base_state.remembered_hub_row_from_spawn,
                remembered_hub_col_from_spawn=base_state.remembered_hub_col_from_spawn,
                last_pos=base_state.last_pos,
                last_move_target=base_state.last_move_target,
            ))

        elif skill == "explore":
            gear = self._current_gear(obs)
            has_heart = self._inventory_count(obs, "heart") > 0
            if gear == "aligner" and has_heart:
                action, base_state = self._aligner._explore_for_alignment(obs, state)
                state = self._copy_with_shared(replace(state,
                    wander_direction_index=base_state.wander_direction_index,
                    wander_steps_remaining=base_state.wander_steps_remaining,
                    last_mode=base_state.last_mode,
                    last_pos=getattr(base_state, 'last_pos', state.last_pos),
                    last_move_target=getattr(base_state, 'last_move_target', state.last_move_target),
                ))
            elif gear == "miner":
                # Use miner-specific explore to stay near hub/extractor areas (avoids junction/scout areas)
                if state.known_hubs:
                    action, base_state = self._miner._explore_near_hub(obs, state)
                else:
                    action, base_state = self._miner._explore(obs, state)
                state = self._copy_with_shared(replace(state,
                    wander_direction_index=base_state.wander_direction_index,
                    wander_steps_remaining=base_state.wander_steps_remaining,
                    last_mode=base_state.last_mode,
                    remembered_hub_row_from_spawn=base_state.remembered_hub_row_from_spawn,
                    remembered_hub_col_from_spawn=base_state.remembered_hub_col_from_spawn,
                    last_pos=base_state.last_pos,
                    last_move_target=base_state.last_move_target,
                ))
            elif state.known_hubs:
                action, base_state = self._aligner._explore_near_hub(obs, state)
                state = self._copy_with_shared(replace(state,
                    wander_direction_index=base_state.wander_direction_index,
                    wander_steps_remaining=base_state.wander_steps_remaining,
                    last_mode=base_state.last_mode,
                    last_pos=getattr(base_state, 'last_pos', state.last_pos),
                    last_move_target=getattr(base_state, 'last_move_target', state.last_move_target),
                ))
            else:
                action, base_state = self._aligner._explore(obs, state)
                state = self._copy_with_shared(replace(state,
                    wander_direction_index=base_state.wander_direction_index,
                    wander_steps_remaining=base_state.wander_steps_remaining,
                    last_mode=base_state.last_mode,
                    last_pos=getattr(base_state, 'last_pos', state.last_pos),
                    last_move_target=getattr(base_state, 'last_move_target', state.last_move_target),
                ))
            # Note: state already updated in each branch above — don't double-update below

        else:
            action, state = self._unstuck_move(state)

        state.skill_steps += 1
        action_name = action.name if hasattr(action, "name") else ""
        if action_name.startswith("move_"):
            direction = action_name[len("move_"):]
            delta_map = {"north": (-1, 0), "east": (0, 1), "south": (1, 0), "west": (0, -1)}
            dr, dc = delta_map.get(direction, (0, 0))
            state.last_move_target = (current_abs[0] + dr, current_abs[1] + dc)
        return action, state


class CrossRolePolicy(MultiAgentPolicy):
    """Policy where all agents can dynamically choose aligner or miner role."""

    short_names = ["cross_role", "machina_cross_role"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        device: str = "cpu",
        num_aligners: int | str = 3,
        return_load: int | str = 40,
        stuck_threshold: int | str = 20,
        unstuck_horizon: int | str = 4,
        llm_api_url: str | None = None,
        llm_model: str | None = "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        llm_api_key_env: str = "OPENROUTER_API_KEY",
        llm_site_url: str | None = None,
        llm_app_name: str = "cogames-cross-role",
        llm_timeout_s: float | str = 10.0,
        llm_responder: Callable[[str], str] | None = None,
        llm_local_model_path: str | None = None,
    ):
        super().__init__(policy_env_info, device=device)
        self._shared_map = SharedMap()
        self._planner = LLMMinerPlannerClient(
            api_url=llm_api_url,
            model=llm_model,
            api_key_env=llm_api_key_env,
            site_url=llm_site_url,
            app_name=llm_app_name,
            timeout_s=float(llm_timeout_s),
            responder=llm_responder,
            local_model_path=llm_local_model_path,
        )
        self._num_aligners = int(num_aligners)
        self._return_load = int(return_load)
        self._stuck_threshold = int(stuck_threshold)
        self._unstuck_horizon = int(unstuck_horizon)
        self._agent_policies: dict[int, StatefulAgentPolicy[CrossRoleState]] = {}

    def agent_policy(self, agent_id: int) -> StatefulAgentPolicy[CrossRoleState]:
        if agent_id not in self._agent_policies:
            preferred = "aligner" if agent_id < self._num_aligners else "miner"
            impl = CrossRolePolicyImpl(
                self._policy_env_info,
                agent_id,
                planner=self._planner,
                stuck_threshold=self._stuck_threshold,
                unstuck_horizon=self._unstuck_horizon,
                return_load=self._return_load,
                shared_map=self._shared_map,
                preferred_initial_gear=preferred,
            )
            self._agent_policies[agent_id] = StatefulAgentPolicy(
                impl,
                self._policy_env_info,
                agent_id=agent_id,
            )
        return self._agent_policies[agent_id]


class GearTestPolicy(MultiAgentPolicy):
    """Two-phase gear test harness for issue-12 gear reliability experiments.

    Phase 1 (steps 0-phase_switch_step): agents acquire initial preferred gear.
    Phase 2 (steps phase_switch_step-episode_end): agents switch to opposite gear.

    Metrics (from GEAR_STATE log lines):
    - initial_gear_success_rate: fraction holding correct gear at step phase_switch_step
    - gear_change_success_rate: fraction holding switched gear at end
    - gear_contamination_rate: fraction holding scout/scrambler gear at end
    """

    short_names = ["gear_test"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        device: str = "cpu",
        num_aligners: int | str = 3,
        return_load: int | str = 40,
        stuck_threshold: int | str = 20,
        unstuck_horizon: int | str = 4,
        phase_switch_step: int | str = 200,
        llm_api_url: str | None = None,
        llm_model: str | None = "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        llm_api_key_env: str = "OPENROUTER_API_KEY",
        llm_site_url: str | None = None,
        llm_app_name: str = "cogames-gear-test",
        llm_timeout_s: float | str = 10.0,
        llm_responder: Callable[[str], str] | None = None,
        llm_local_model_path: str | None = None,
    ):
        super().__init__(policy_env_info, device=device)
        self._shared_map = SharedMap()
        self._planner = LLMMinerPlannerClient(
            api_url=llm_api_url,
            model=llm_model,
            api_key_env=llm_api_key_env,
            site_url=llm_site_url,
            app_name=llm_app_name,
            timeout_s=float(llm_timeout_s),
            responder=llm_responder,
            local_model_path=llm_local_model_path,
        )
        self._num_aligners = int(num_aligners)
        self._return_load = int(return_load)
        self._stuck_threshold = int(stuck_threshold)
        self._unstuck_horizon = int(unstuck_horizon)
        self._phase_switch_step = int(phase_switch_step)
        self._agent_policies: dict[int, StatefulAgentPolicy[CrossRoleState]] = {}

    def agent_policy(self, agent_id: int) -> StatefulAgentPolicy[CrossRoleState]:
        if agent_id not in self._agent_policies:
            preferred = "aligner" if agent_id < self._num_aligners else "miner"
            impl = CrossRolePolicyImpl(
                self._policy_env_info,
                agent_id,
                planner=self._planner,
                stuck_threshold=self._stuck_threshold,
                unstuck_horizon=self._unstuck_horizon,
                return_load=self._return_load,
                shared_map=self._shared_map,
                preferred_initial_gear=preferred,
                phase_switch_step=self._phase_switch_step,
            )
            self._agent_policies[agent_id] = StatefulAgentPolicy(
                impl,
                self._policy_env_info,
                agent_id=agent_id,
            )
        return self._agent_policies[agent_id]
