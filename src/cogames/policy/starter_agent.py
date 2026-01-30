"""
Sample policy for the CoGames CogsGuard environment.

This starter policy uses simple heuristics:
- If the agent has no gear, head toward the nearest gear station.
- If the agent has aligner or scrambler gear, try to get hearts (and influence for aligner) then head to junctions.
- If the agent has miner gear, head to extractors.
- If the agent has scout gear, explore in a simple pattern.

Note to users of this policy:
We don't intend for scripted policies to be the final word on how policies are generated (e.g., we expect the
environment to be complicated enough that trained agents will be necessary). So we expect that scripting policies
is a good way to start, but don't want you to get stuck here. Feel free to prove us wrong!

Note to cogames developers:
This policy should be kept relatively minimalist, without dependencies on intricate algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from mettagrid.policy.policy import MultiAgentPolicy, StatefulAgentPolicy, StatefulPolicyImpl
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action
from mettagrid.simulator.interface import AgentObservation

GEAR = ("aligner", "scrambler", "miner", "scout")
ELEMENTS = ("carbon", "oxygen", "germanium", "silicon")
WANDER_DIRECTIONS = ("east", "south", "west", "north")
WANDER_STEPS = 8


@dataclass
class StarterCogState:
    wander_direction_index: int = 0
    wander_steps_remaining: int = WANDER_STEPS


class StarterCogPolicyImpl(StatefulPolicyImpl[StarterCogState]):
    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        agent_id: int,
    ):
        self._agent_id = agent_id
        self._policy_env_info = policy_env_info

        self._action_names = policy_env_info.action_names
        self._action_name_set = set(self._action_names)
        self._fallback_action_name = "noop" if "noop" in self._action_name_set else self._action_names[0]
        self._center = (policy_env_info.obs_height // 2, policy_env_info.obs_width // 2)
        self._tag_name_to_id = {name: idx for idx, name in enumerate(policy_env_info.tags)}
        self._gear_station_tags = self._resolve_tag_ids([f"{gear}_station" for gear in GEAR])
        self._extractor_tags = self._resolve_tag_ids([f"{element}_extractor" for element in ELEMENTS])
        self._junction_tags = self._resolve_tag_ids(["junction"])
        self._chest_tags = self._resolve_tag_ids(["chest"])
        self._hub_tags = self._resolve_tag_ids(["hub"])

    def _resolve_tag_ids(self, names: Iterable[str]) -> set[int]:
        tag_ids: set[int] = set()
        for name in names:
            if name in self._tag_name_to_id:
                tag_ids.add(self._tag_name_to_id[name])
            if name.startswith("type:"):
                continue
            type_name = f"type:{name}"
            if type_name in self._tag_name_to_id:
                tag_ids.add(self._tag_name_to_id[type_name])
        return tag_ids

    def _inventory_items(self, obs: AgentObservation) -> set[str]:
        items: set[str] = set()
        for token in obs.tokens:
            if token.location != self._center:
                continue
            name = token.feature.name
            if not name.startswith("inv:"):
                continue
            parts = name.split(":", 2)
            if len(parts) >= 2:
                items.add(parts[1])
        return items

    def _closest_tag_location(self, obs: AgentObservation, tag_ids: set[int]) -> Optional[tuple[int, int]]:
        if not tag_ids:
            return None
        best_location: Optional[tuple[int, int]] = None
        best_distance = 999
        for token in obs.tokens:
            if token.feature.name != "tag":
                continue
            if token.value not in tag_ids:
                continue
            distance = abs(token.location[0] - self._center[0]) + abs(token.location[1] - self._center[1])
            if distance < best_distance:
                best_distance = distance
                best_location = token.location
        return best_location

    def _action(self, name: str) -> Action:
        if name in self._action_name_set:
            return Action(name=name)
        return Action(name=self._fallback_action_name)

    def _wander(self, state: StarterCogState) -> tuple[Action, StarterCogState]:
        if state.wander_steps_remaining <= 0:
            state.wander_direction_index = (state.wander_direction_index + 1) % len(WANDER_DIRECTIONS)
            state.wander_steps_remaining = WANDER_STEPS
        direction = WANDER_DIRECTIONS[state.wander_direction_index]
        state.wander_steps_remaining -= 1
        return self._action(f"move_{direction}"), state

    def _move_toward(self, state: StarterCogState, target: Optional[tuple[int, int]]) -> tuple[Action, StarterCogState]:
        if target is None:
            return self._wander(state)
        delta_row = target[0] - self._center[0]
        delta_col = target[1] - self._center[1]
        if delta_row == 0 and delta_col == 0:
            return self._action(self._fallback_action_name), state
        if abs(delta_row) >= abs(delta_col):
            direction = "south" if delta_row > 0 else "north"
        else:
            direction = "east" if delta_col > 0 else "west"
        return self._action(f"move_{direction}"), state

    def _current_gear(self, items: set[str]) -> Optional[str]:
        for gear in GEAR:
            if gear in items:
                return gear
        return None

    def step_with_state(self, obs: AgentObservation, state: StarterCogState) -> tuple[Action, StarterCogState]:
        """Compute the action for this Cog."""
        items = self._inventory_items(obs)
        gear = self._current_gear(items)
        has_heart = "heart" in items
        has_influence = "influence" in items

        if gear is None:
            target_tags = self._gear_station_tags
        elif gear == "aligner":
            if has_heart and has_influence:
                target_tags = self._junction_tags
            elif not has_heart:
                target_tags = self._chest_tags
            else:
                target_tags = self._hub_tags
        elif gear == "scrambler":
            target_tags = self._junction_tags if has_heart else self._chest_tags
        elif gear == "miner":
            target_tags = self._extractor_tags
        else:
            target_tags = set()

        target_location = self._closest_tag_location(obs, target_tags) if target_tags else None
        return self._move_toward(state, target_location)

    def initial_agent_state(self) -> StarterCogState:
        """Get the initial state for a new agent."""
        return StarterCogState()


# ============================================================================
# Policy Wrapper Classes
# ============================================================================


class StarterPolicy(MultiAgentPolicy):
    short_names = ["starter"]

    def __init__(self, policy_env_info: PolicyEnvInterface, device: str = "cpu"):
        super().__init__(policy_env_info, device=device)
        self._agent_policies: dict[int, StatefulAgentPolicy[StarterCogState]] = {}

    def agent_policy(self, agent_id: int) -> StatefulAgentPolicy[StarterCogState]:
        if agent_id not in self._agent_policies:
            self._agent_policies[agent_id] = StatefulAgentPolicy(
                StarterCogPolicyImpl(self._policy_env_info, agent_id),
                self._policy_env_info,
                agent_id=agent_id,
            )
        return self._agent_policies[agent_id]
