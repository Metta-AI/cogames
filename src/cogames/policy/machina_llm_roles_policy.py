from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, replace
from typing import Callable

from cogames.policy.aligner_agent import AlignerPolicyImpl, AlignerState
from cogames.policy.llm_aligner_prompt import ALIGNER_SKILL_DESCRIPTIONS, build_llm_aligner_prompt
from cogames.policy.llm_miner_policy import LLMMinerPlannerClient, LLMMinerPolicyImpl, LLMMinerState
from mettagrid.policy.policy import MultiAgentPolicy, StatefulAgentPolicy, StatefulPolicyImpl
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action
from mettagrid.simulator.interface import AgentObservation

logger = logging.getLogger("cogames.policy.machina_llm_roles")


def _parse_role_skill_choice(text: str, valid_skills: set[str]) -> tuple[str | None, str]:
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
        return (skill if skill in valid_skills else None, "non-json response")
    if not isinstance(payload, dict):
        return None, "response was not a JSON object"
    skill = payload.get("skill")
    reason = payload.get("reason", "")
    if not isinstance(skill, str):
        return None, "missing skill field"
    normalized_skill = {"unstick": "unstuck"}.get(skill, skill)
    return (normalized_skill if normalized_skill in valid_skills else None, str(reason))


@dataclass
class LLMAlignerState(AlignerState):
    current_skill: str | None = None
    current_reason: str = ""
    skill_steps: int = 0
    no_move_steps: int = 0
    last_has_heart: bool = False
    last_friendly_junctions: int = 0
    recent_events: list[str] = field(default_factory=list)


class LLMAlignerPolicyImpl(AlignerPolicyImpl, StatefulPolicyImpl[LLMAlignerState]):
    _UNSTUCK_DIRECTIONS = ("north", "east", "south", "west")

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        agent_id: int,
        planner: LLMMinerPlannerClient,
        stuck_threshold: int,
        unstuck_horizon: int,
    ) -> None:
        super().__init__(policy_env_info, agent_id)
        self._planner = planner
        self._stuck_threshold = stuck_threshold
        self._unstuck_horizon = unstuck_horizon

    def initial_agent_state(self) -> LLMAlignerState:
        base = super().initial_agent_state()
        return LLMAlignerState(
            wander_direction_index=base.wander_direction_index,
            wander_steps_remaining=base.wander_steps_remaining,
            last_mode=base.last_mode,
        )

    def _copy_with(self, state: LLMAlignerState, base: AlignerState) -> LLMAlignerState:
        return replace(
            state,
            wander_direction_index=base.wander_direction_index,
            wander_steps_remaining=base.wander_steps_remaining,
            last_mode=base.last_mode,
            known_free_cells=set(base.known_free_cells),
            blocked_cells=set(base.blocked_cells),
            known_hubs=set(base.known_hubs),
            known_aligner_stations=set(base.known_aligner_stations),
            known_neutral_junctions=set(base.known_neutral_junctions),
            known_friendly_junctions=set(base.known_friendly_junctions),
            known_enemy_junctions=set(base.known_enemy_junctions),
            current_skill=state.current_skill,
            current_reason=state.current_reason,
            skill_steps=state.skill_steps,
            no_move_steps=state.no_move_steps,
            last_has_heart=state.last_has_heart,
            last_friendly_junctions=state.last_friendly_junctions,
            recent_events=list(state.recent_events),
        )

    def _event(self, state: LLMAlignerState, message: str) -> None:
        state.recent_events.append(message)
        del state.recent_events[:-10]

    def _feature_value(self, obs: AgentObservation, feature_name: str) -> int | None:
        for token in obs.tokens:
            if token.feature.name == feature_name:
                return int(token.value)
        return None

    def _hub_visible(self, obs: AgentObservation) -> bool:
        return self._starter._closest_tag_location(obs, self._hub_tags) is not None

    def _known_alignable_junctions(self, state: LLMAlignerState) -> set[tuple[int, int]]:
        return {junction for junction in state.known_neutral_junctions if self._is_alignable(junction, state)}

    def _update_progress(self, obs: AgentObservation, state: LLMAlignerState) -> None:
        has_heart = self._inventory_count(obs, "heart") > 0
        friendly_count = len(state.known_friendly_junctions)
        current_abs = self._spawn_offset(obs)
        if state.current_skill == "get_heart" and has_heart and not state.last_has_heart:
            self._event(state, "acquired a heart")
        if state.current_skill == "align_neutral" and friendly_count > state.last_friendly_junctions:
            self._event(state, f"friendly junction count increased from {state.last_friendly_junctions} to {friendly_count}")
        state.last_has_heart = has_heart
        state.last_friendly_junctions = friendly_count

        last_action_move = self._feature_value(obs, "last_action_move")
        stationary_on_valid_target = (
            (state.current_skill == "get_heart" and current_abs in state.known_hubs)
            or (state.current_skill == "align_neutral" and current_abs in self._known_alignable_junctions(state))
            or (state.current_skill == "gear_up" and current_abs in state.known_aligner_stations)
        )
        if stationary_on_valid_target:
            state.no_move_steps = 0
        elif state.current_skill is not None and last_action_move == 0:
            state.no_move_steps += 1
        else:
            state.no_move_steps = 0

    def _plan_skill(self, obs: AgentObservation, state: LLMAlignerState) -> None:
        has_aligner = self._current_gear(obs) == "aligner"
        has_heart = self._inventory_count(obs, "heart") > 0
        known_alignable_junctions = self._known_alignable_junctions(state)
        prompt = build_llm_aligner_prompt(
            has_aligner=has_aligner,
            has_heart=has_heart,
            hub_visible=self._hub_visible(obs),
            known_hubs=len(state.known_hubs),
            known_neutral_junctions=len(state.known_neutral_junctions),
            known_alignable_junctions=len(known_alignable_junctions),
            known_friendly_junctions=len(state.known_friendly_junctions),
            current_skill=state.current_skill,
            no_move_steps=state.no_move_steps,
            recent_events=state.recent_events,
        )
        logger.info("agent=%s role=aligner llm_prompt=%s", obs.agent_id, prompt.replace("\n", " | "))
        started_at = time.perf_counter()
        text = self._planner.complete(prompt)
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        logger.info(
            "agent=%s role=aligner llm_response_ms=%.1f llm_response=%s",
            obs.agent_id,
            latency_ms,
            text.replace("\n", " "),
        )
        skill, reason = _parse_role_skill_choice(text, set(ALIGNER_SKILL_DESCRIPTIONS))
        if skill is None:
            if not has_aligner:
                skill = "gear_up"
            elif not has_heart and state.known_hubs:
                skill = "get_heart"
            elif known_alignable_junctions:
                skill = "align_neutral"
            else:
                skill = "explore"
            reason = f"fallback after invalid planner response: {reason}"
        if not has_aligner and skill != "gear_up":
            reason = f"overrode {skill} to gear_up because aligner gear is missing"
            skill = "gear_up"
        if has_aligner and skill == "gear_up":
            if has_heart and state.known_neutral_junctions:
                reason = "overrode gear_up to align_neutral because aligner gear is already equipped and a target is known"
                skill = "align_neutral"
            elif not has_heart:
                reason = "overrode gear_up to get_heart because aligner gear is already equipped"
                skill = "get_heart"
            else:
                reason = "overrode gear_up to explore because aligner gear is already equipped"
                skill = "explore"
        if has_aligner and not has_heart and state.known_hubs and skill in {"explore", "unstuck"}:
            reason = f"overrode {skill} to get_heart because aligner gear is equipped and a hub is known"
            skill = "get_heart"
        if has_aligner and not has_heart and skill == "align_neutral":
            reason = "overrode align_neutral to get_heart because no heart is held"
            skill = "get_heart"
        if has_aligner and has_heart and known_alignable_junctions and skill in {"explore", "get_heart", "unstuck"}:
            reason = f"overrode {skill} to align_neutral because an alignable neutral junction is already known"
            skill = "align_neutral"
        state.current_skill = skill
        state.current_reason = reason
        state.skill_steps = 0
        self._event(state, f"planner selected {skill}: {reason}")

    def _maybe_finish_skill(self, obs: AgentObservation, state: LLMAlignerState) -> None:
        has_heart = self._inventory_count(obs, "heart") > 0
        has_aligner = self._current_gear(obs) == "aligner"
        if state.current_skill == "gear_up" and has_aligner:
            self._event(state, "gear_up completed after acquiring aligner gear")
            state.current_skill = None
        elif state.current_skill == "get_heart" and has_heart:
            self._event(state, "get_heart completed after acquiring heart")
            state.current_skill = None
        elif state.current_skill == "align_neutral" and not has_heart:
            self._event(state, "align_neutral completed after spending heart")
            state.current_skill = None
        elif state.current_skill == "explore" and state.known_neutral_junctions:
            self._event(state, f"explore completed after discovering {len(state.known_neutral_junctions)} neutral junction(s)")
            state.current_skill = None
        elif state.current_skill == "unstuck" and state.skill_steps >= self._unstuck_horizon:
            self._event(state, "unstuck finished its bounded horizon")
            state.current_skill = None
        elif state.current_skill is not None and state.no_move_steps >= self._stuck_threshold:
            self._event(state, f"{state.current_skill} exited as stuck after {state.no_move_steps} blocked steps")
            state.current_skill = None

    def _unstuck(self, state: LLMAlignerState) -> tuple[Action, LLMAlignerState]:
        state.last_mode = "unstuck"
        direction = self._UNSTUCK_DIRECTIONS[state.wander_direction_index % len(self._UNSTUCK_DIRECTIONS)]
        state.wander_direction_index = (state.wander_direction_index + 1) % len(self._UNSTUCK_DIRECTIONS)
        return self._starter._action(f"move_{direction}"), state

    def step_with_state(self, obs: AgentObservation, state: LLMAlignerState) -> tuple[Action, LLMAlignerState]:
        current_abs = self._update_map_memory(obs, state)
        self._update_progress(obs, state)

        self._maybe_finish_skill(obs, state)
        if state.current_skill is None:
            self._plan_skill(obs, state)

        if state.current_skill == "gear_up":
            action, base_state = self._gear_up(obs, state, current_abs)
            state = self._copy_with(state, base_state)
        elif state.current_skill == "get_heart":
            action, base_state = self._get_heart(obs, state, current_abs)
            state = self._copy_with(state, base_state)
        elif state.current_skill == "align_neutral":
            action, base_state = self._align_neutral(obs, state, current_abs)
            state = self._copy_with(state, base_state)
        elif state.current_skill == "explore":
            if self._inventory_count(obs, "heart") > 0:
                action, base_state = self._explore_for_alignment(obs, state)
            elif state.known_hubs:
                action, base_state = self._explore_near_hub(obs, state)
            else:
                action, base_state = self._explore(obs, state)
            state = self._copy_with(state, base_state)
        else:
            action, state = self._unstuck(state)

        state.skill_steps += 1
        return action, state


class MachinaLLMRolesPolicy(MultiAgentPolicy):
    short_names = ["machina_llm_roles", "llm_team3"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        device: str = "cpu",
        num_aligners: int | str = 1,
        aligner_ids: str = "",
        return_load: int | str = 40,
        stuck_threshold: int | str = 6,
        unstuck_horizon: int | str = 4,
        llm_api_url: str | None = None,
        llm_model: str | None = "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        llm_api_key_env: str = "OPENROUTER_API_KEY",
        llm_site_url: str | None = None,
        llm_app_name: str = "cogames-voyager",
        llm_timeout_s: float | str = 10.0,
        llm_responder: Callable[[str], str] | None = None,
    ):
        super().__init__(policy_env_info, device=device)
        parsed_aligner_ids = tuple(int(part.strip()) for part in aligner_ids.split(",") if part.strip())
        if parsed_aligner_ids:
            self._aligner_ids = frozenset(parsed_aligner_ids)
        else:
            self._aligner_ids = frozenset(range(min(int(num_aligners), policy_env_info.num_agents)))
        self._planner = LLMMinerPlannerClient(
            api_url=llm_api_url,
            model=llm_model,
            api_key_env=llm_api_key_env,
            site_url=llm_site_url,
            app_name=llm_app_name,
            timeout_s=float(llm_timeout_s),
            responder=llm_responder,
        )
        self._return_load = int(return_load)
        self._stuck_threshold = int(stuck_threshold)
        self._unstuck_horizon = int(unstuck_horizon)
        self._agent_policies: dict[int, StatefulAgentPolicy[LLMAlignerState | LLMMinerState]] = {}

    def agent_policy(self, agent_id: int) -> StatefulAgentPolicy[LLMAlignerState | LLMMinerState]:
        if agent_id not in self._agent_policies:
            if agent_id in self._aligner_ids:
                impl = LLMAlignerPolicyImpl(
                    self._policy_env_info,
                    agent_id,
                    planner=self._planner,
                    stuck_threshold=self._stuck_threshold,
                    unstuck_horizon=self._unstuck_horizon,
                )
            else:
                impl = LLMMinerPolicyImpl(
                    self._policy_env_info,
                    agent_id,
                    planner=self._planner,
                    return_load=self._return_load,
                    stuck_threshold=self._stuck_threshold,
                    unstuck_horizon=self._unstuck_horizon,
                )
            self._agent_policies[agent_id] = StatefulAgentPolicy(
                impl,
                self._policy_env_info,
                agent_id=agent_id,
            )
        return self._agent_policies[agent_id]
