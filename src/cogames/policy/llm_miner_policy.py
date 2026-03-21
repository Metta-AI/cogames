from __future__ import annotations

import logging
import json
import os
import re
import time
from dataclasses import dataclass, field, replace
from typing import Callable

import httpx

from cogames.policy.llm_miner_prompt import SKILL_DESCRIPTIONS, build_llm_miner_prompt
from cogames.policy.llm_skills import MinerSkillImpl, MinerSkillState
from mettagrid.policy.policy import MultiAgentPolicy, StatefulAgentPolicy, StatefulPolicyImpl
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action
from mettagrid.simulator.interface import AgentObservation

logger = logging.getLogger("cogames.policy.llm_miner")


@dataclass
class LLMMinerState(MinerSkillState):
    current_skill: str | None = None
    current_reason: str = ""
    skill_steps: int = 0
    no_move_steps: int = 0
    last_carried_total: int = 0
    recent_events: list[str] = field(default_factory=list)


class LLMMinerPlannerClient:
    def __init__(
        self,
        api_url: str | None = None,
        model: str | None = None,
        api_key_env: str = "OPENROUTER_API_KEY",
        site_url: str | None = None,
        app_name: str = "cogames-voyager",
        timeout_s: float = 5.0,
        responder: Callable[[str], str] | None = None,
    ) -> None:
        self._api_url = api_url
        self._model = model
        self._api_key_env = api_key_env
        self._site_url = site_url
        self._app_name = app_name
        self._timeout_s = timeout_s
        self._responder = responder

    def complete(self, prompt: str) -> str:
        if self._responder is not None:
            return self._responder(prompt)
        api_key = os.environ.get(self._api_key_env)
        if self._model:
            return self._complete_openrouter(prompt, api_key)
        if not self._api_url:
            raise RuntimeError("LLM planner API is not configured")
        with httpx.Client(timeout=self._timeout_s) as client:
            response = client.post(self._api_url, json={"prompt": prompt})
            response.raise_for_status()
            payload = response.json()
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("LLM planner response missing non-empty 'text'")
        return text

    def _complete_openrouter(self, prompt: str, api_key: str | None) -> str:
        if not api_key:
            raise RuntimeError(f"Missing API key in environment variable {self._api_key_env}")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": self._app_name,
        }
        if self._site_url:
            headers["HTTP-Referer"] = self._site_url
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "/no_think\n"
                        "You are a planner for one miner cog in CoGames. "
                        "Respond with a single JSON object and no extra text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 120,
        }
        with httpx.Client(timeout=self._timeout_s) as client:
            response = client.post(
                self._api_url or "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenRouter response missing choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise RuntimeError("OpenRouter response missing message")
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
            text = "".join(text_parts).strip()
            if text:
                return text
        raise RuntimeError("OpenRouter response missing non-empty assistant content")


def _parse_skill_choice(text: str) -> tuple[str | None, str]:
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
        return (skill if skill in SKILL_DESCRIPTIONS else None, "non-json response")
    if not isinstance(payload, dict):
        return None, "response was not a JSON object"
    skill = payload.get("skill")
    reason = payload.get("reason", "")
    if not isinstance(skill, str):
        return None, "missing skill field"
    normalized_skill = {"unstick": "unstuck"}.get(skill, skill)
    return (normalized_skill if normalized_skill in SKILL_DESCRIPTIONS else None, str(reason))


class LLMMinerPolicyImpl(MinerSkillImpl, StatefulPolicyImpl[LLMMinerState]):
    _UNSTUCK_DIRECTIONS = ("north", "east", "south", "west")

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        agent_id: int,
        planner: LLMMinerPlannerClient,
        return_load: int,
        stuck_threshold: int,
        unstuck_horizon: int,
    ) -> None:
        super().__init__(policy_env_info, agent_id, return_load=return_load)
        self._planner = planner
        self._stuck_threshold = stuck_threshold
        self._unstuck_horizon = unstuck_horizon

    def initial_agent_state(self) -> LLMMinerState:
        base = super().initial_agent_state()
        return LLMMinerState(
            wander_direction_index=base.wander_direction_index,
            wander_steps_remaining=base.wander_steps_remaining,
            last_mode=base.last_mode,
            remembered_hub_row_from_spawn=base.remembered_hub_row_from_spawn,
            remembered_hub_col_from_spawn=base.remembered_hub_col_from_spawn,
        )

    def _copy_with(self, state: LLMMinerState, base: MinerSkillState) -> LLMMinerState:
        return replace(
            state,
            wander_direction_index=base.wander_direction_index,
            wander_steps_remaining=base.wander_steps_remaining,
            last_mode=base.last_mode,
            remembered_hub_row_from_spawn=base.remembered_hub_row_from_spawn,
            remembered_hub_col_from_spawn=base.remembered_hub_col_from_spawn,
            known_free_cells=set(base.known_free_cells),
            blocked_cells=set(base.blocked_cells),
            known_hubs=set(base.known_hubs),
            known_miner_stations=set(base.known_miner_stations),
            known_extractors=set(base.known_extractors),
            current_skill=state.current_skill,
            current_reason=state.current_reason,
            skill_steps=state.skill_steps,
            no_move_steps=state.no_move_steps,
            last_carried_total=state.last_carried_total,
            recent_events=list(state.recent_events),
        )

    def _event(self, state: LLMMinerState, message: str) -> None:
        state.recent_events.append(message)
        del state.recent_events[:-10]

    def _feature_value(self, obs: AgentObservation, feature_name: str) -> int | None:
        for token in obs.tokens:
            if token.feature.name == feature_name:
                return int(token.value)
        return None

    def _hub_visible(self, obs: AgentObservation) -> bool:
        return self._starter._closest_tag_location(obs, self._hub_tags) is not None

    def _frontier_count(self, state: LLMMinerState) -> int:
        return len(self._frontier_cells(state))

    def _update_progress(self, obs: AgentObservation, state: LLMMinerState) -> None:
        carried_total = self._carried_total(obs)
        if state.current_skill == "deposit_to_hub" and carried_total < state.last_carried_total:
            self._event(state, f"deposited cargo from {state.last_carried_total} to {carried_total}")
        elif state.current_skill == "mine_until_full" and carried_total > state.last_carried_total:
            self._event(state, f"cargo increased from {state.last_carried_total} to {carried_total}")
        state.last_carried_total = carried_total

        last_action_move = self._feature_value(obs, "last_action_move")
        current_abs = self._current_abs(obs)
        stationary_on_valid_target = (
            (state.current_skill == "mine_until_full" and current_abs in state.known_extractors)
            or (state.current_skill == "deposit_to_hub" and current_abs in state.known_hubs)
            or (state.current_skill == "gear_up" and current_abs in state.known_miner_stations)
        )
        if stationary_on_valid_target:
            state.no_move_steps = 0
        elif state.current_skill is not None and last_action_move == 0:
            state.no_move_steps += 1
        else:
            state.no_move_steps = 0

    def _plan_skill(self, obs: AgentObservation, state: LLMMinerState) -> None:
        has_miner = self._starter._current_gear(self._starter._inventory_items(obs)) == "miner"
        prompt = build_llm_miner_prompt(
            carried_total=self._carried_total(obs),
            return_load=self._return_load,
            has_miner=has_miner,
            hub_visible=self._hub_visible(obs),
            remembered_hub=(state.remembered_hub_row_from_spawn, state.remembered_hub_col_from_spawn),
            known_extractors=len(state.known_extractors),
            frontier_count=self._frontier_count(state),
            current_skill=state.current_skill,
            no_move_steps=state.no_move_steps,
            recent_events=state.recent_events,
        )
        logger.info("agent=%s llm_prompt=%s", obs.agent_id, prompt.replace("\n", " | "))
        started_at = time.perf_counter()
        text = self._planner.complete(prompt)
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        logger.info("agent=%s llm_response_ms=%.1f llm_response=%s", obs.agent_id, latency_ms, text.replace("\n", " "))
        skill, reason = _parse_skill_choice(text)
        if skill is None:
            carried_total = self._carried_total(obs)
            if not has_miner:
                skill = "gear_up"
            elif carried_total >= self._return_load:
                skill = "deposit_to_hub"
            elif state.known_extractors:
                skill = "mine_until_full"
            else:
                skill = "explore"
            reason = f"fallback after invalid planner response: {reason}"
        if not has_miner and skill != "gear_up":
            reason = f"overrode {skill} to gear_up because miner gear is missing"
            skill = "gear_up"
        if has_miner and skill == "gear_up":
            if self._carried_total(obs) >= self._return_load:
                reason = "overrode gear_up to deposit_to_hub because miner gear is already equipped and cargo is full"
                skill = "deposit_to_hub"
            elif state.known_extractors:
                reason = "overrode gear_up to mine_until_full because miner gear is already equipped"
                skill = "mine_until_full"
            else:
                reason = "overrode gear_up to explore because miner gear is already equipped and no extractor is known"
                skill = "explore"
        if has_miner and self._carried_total(obs) >= self._return_load and skill == "mine_until_full":
            reason = "overrode mine_until_full to deposit_to_hub because cargo is full"
            skill = "deposit_to_hub"
        state.current_skill = skill
        state.current_reason = reason
        state.skill_steps = 0
        self._event(state, f"planner selected {skill}: {reason}")

    def _maybe_finish_skill(self, obs: AgentObservation, state: LLMMinerState) -> None:
        carried_total = self._carried_total(obs)
        has_miner = self._starter._current_gear(self._starter._inventory_items(obs)) == "miner"
        if state.current_skill == "gear_up" and has_miner:
            self._event(state, "gear_up completed after acquiring miner gear")
            state.current_skill = None
        elif state.current_skill == "mine_until_full" and carried_total >= self._return_load:
            self._event(state, f"mine_until_full completed at load={carried_total}")
            state.current_skill = None
        elif state.current_skill == "deposit_to_hub" and carried_total == 0:
            self._event(state, "deposit_to_hub completed after deposit")
            state.current_skill = None
        elif state.current_skill == "explore" and state.known_extractors:
            self._event(state, f"explore completed after discovering {len(state.known_extractors)} extractor(s)")
            state.current_skill = None
        elif state.current_skill == "unstuck" and state.skill_steps >= self._unstuck_horizon:
            self._event(state, "unstuck finished its bounded horizon")
            state.current_skill = None
        elif state.current_skill is not None and state.no_move_steps >= self._stuck_threshold:
            self._event(state, f"{state.current_skill} exited as stuck after {state.no_move_steps} blocked steps")
            state.current_skill = None

    def _unstuck(self, state: LLMMinerState) -> tuple[Action, LLMMinerState]:
        state.last_mode = "unstuck"
        direction = self._UNSTUCK_DIRECTIONS[state.wander_direction_index % len(self._UNSTUCK_DIRECTIONS)]
        state.wander_direction_index = (state.wander_direction_index + 1) % len(self._UNSTUCK_DIRECTIONS)
        return self._starter._action(f"move_{direction}"), state

    def step_with_state(self, obs: AgentObservation, state: LLMMinerState) -> tuple[Action, LLMMinerState]:
        self._update_map_memory(obs, state)
        self._update_progress(obs, state)

        self._maybe_finish_skill(obs, state)
        if state.current_skill is None:
            self._plan_skill(obs, state)

        if state.current_skill == "gear_up":
            action, base_state = self._gear_up(obs, state)
            state = self._copy_with(state, base_state)
        elif state.current_skill == "mine_until_full":
            action, base_state = self._mine_until_full(obs, state)
            state = self._copy_with(state, base_state)
        elif state.current_skill == "deposit_to_hub":
            action, base_state = self._deposit_to_hub(obs, state)
            state = self._copy_with(state, base_state)
        elif state.current_skill == "explore":
            action, base_state = self._explore(obs, state)
            state = self._copy_with(state, base_state)
        else:
            action, state = self._unstuck(state)

        state.skill_steps += 1
        return action, state


class LLMMinerPolicy(MultiAgentPolicy):
    short_names = ["llm_miner"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        device: str = "cpu",
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
        self._return_load = int(return_load)
        self._stuck_threshold = int(stuck_threshold)
        self._unstuck_horizon = int(unstuck_horizon)
        self._planner = LLMMinerPlannerClient(
            api_url=llm_api_url,
            model=llm_model,
            api_key_env=llm_api_key_env,
            site_url=llm_site_url,
            app_name=llm_app_name,
            timeout_s=float(llm_timeout_s),
            responder=llm_responder,
        )
        self._agent_policies: dict[int, StatefulAgentPolicy[LLMMinerState]] = {}

    def agent_policy(self, agent_id: int) -> StatefulAgentPolicy[LLMMinerState]:
        if agent_id not in self._agent_policies:
            self._agent_policies[agent_id] = StatefulAgentPolicy(
                LLMMinerPolicyImpl(
                    self._policy_env_info,
                    agent_id,
                    planner=self._planner,
                    return_load=self._return_load,
                    stuck_threshold=self._stuck_threshold,
                    unstuck_horizon=self._unstuck_horizon,
                ),
                self._policy_env_info,
                agent_id=agent_id,
            )
        return self._agent_policies[agent_id]
