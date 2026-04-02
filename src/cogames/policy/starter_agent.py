"""
Sample policy for the CoGames CvC environment.

This starter policy uses simple heuristics:
- If the agent has no gear, head toward the nearest gear station.
- If the agent has aligner or scrambler gear, try to get hearts then head to junctions.
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
TEAM_TAG_PREFIX = "team:"


@dataclass
class StarterCogState:
    wander_direction_index: int = 0
    wander_steps_remaining: int = WANDER_STEPS


class StarterCogPolicyImpl(StatefulPolicyImpl[StarterCogState]):
    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        agent_id: int,
        preferred_gear: Optional[str] = None,
    ):
        self._agent_id = agent_id
        self._policy_env_info = policy_env_info
        self._preferred_gear = preferred_gear

        self._action_names = policy_env_info.action_names
        self._action_name_set = set(self._action_names)
        self._fallback_action_name = "noop" if "noop" in self._action_name_set else self._action_names[0]
        self._center = (policy_env_info.obs_height // 2, policy_env_info.obs_width // 2)
        self._tag_name_to_id = {name: idx for idx, name in enumerate(policy_env_info.tags)}
        self._gear_station_tags_by_gear = {gear: self._resolve_tag_ids([gear, f"c:{gear}"]) for gear in GEAR}
        self._gear_station_tags = set().union(*self._gear_station_tags_by_gear.values())
        self._extractor_tags = self._resolve_tag_ids([f"{element}_extractor" for element in ELEMENTS])
        self._junction_tags = self._resolve_tag_ids(["junction"])
        self._heart_source_tags = self._resolve_tag_ids(["hub", "chest"])

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

    def _inventory_amounts(self, obs: AgentObservation) -> dict[str, int]:
        items: dict[str, int] = {}
        for token in obs.tokens:
            if token.location != self._center:
                continue
            name = token.feature.name
            if not name.startswith("inv:"):
                continue
            suffix = name[4:]
            if not suffix:
                continue
            item_name, sep, power_str = suffix.rpartition(":p")
            if not sep or not item_name or not power_str.isdigit():
                item_name = suffix
                power = 0
            else:
                power = int(power_str)
            value = int(token.value)
            if value <= 0:
                continue
            base = max(int(token.feature.normalization), 1)
            items[item_name] = items.get(item_name, 0) + value * (base**power)
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
            loc = token.location
            if loc is None:
                continue
            distance = abs(loc[0] - self._center[0]) + abs(loc[1] - self._center[1])
            if distance < best_distance:
                best_distance = distance
                best_location = token.location
        return best_location

    def _own_team_tag_ids(self, obs: AgentObservation) -> set[int]:
        team_tag_ids: set[int] = set()
        for token in obs.tokens:
            if token.feature.name != "tag" or token.location != self._center:
                continue
            tag_name = self._policy_env_info.tags[token.value]
            if tag_name.startswith(TEAM_TAG_PREFIX):
                team_tag_ids.add(token.value)
        return team_tag_ids

    def _closest_friendly_gear_station(
        self,
        obs: AgentObservation,
        gear_tag_ids: set[int],
    ) -> Optional[tuple[int, int]]:
        own_team_tag_ids = self._own_team_tag_ids(obs)
        if not own_team_tag_ids:
            return self._closest_tag_location(obs, gear_tag_ids)

        tags_by_location: dict[tuple[int, int], set[int]] = {}
        for token in obs.tokens:
            if token.feature.name != "tag" or token.location is None:
                continue
            tags_by_location.setdefault(token.location, set()).add(token.value)

        best_location: Optional[tuple[int, int]] = None
        best_distance = 999
        for location, location_tag_ids in tags_by_location.items():
            if not (location_tag_ids & gear_tag_ids):
                continue
            if not (location_tag_ids & own_team_tag_ids):
                continue
            distance = abs(location[0] - self._center[0]) + abs(location[1] - self._center[1])
            if distance < best_distance:
                best_distance = distance
                best_location = location
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

    def _current_gear(self, items: dict[str, int]) -> Optional[str]:
        if self._preferred_gear is not None and items.get(self._preferred_gear, 0) > 0:
            return self._preferred_gear
        for gear in GEAR:
            if items.get(gear, 0) > 0:
                return gear
        return None

    def step_with_state(self, obs: AgentObservation, state: StarterCogState) -> tuple[Action, StarterCogState]:
        """Compute the action for this Cog."""
        items = self._inventory_amounts(obs)
        gear = self._current_gear(items)
        has_heart = items.get("heart", 0) > 0

        if self._preferred_gear is not None and gear != self._preferred_gear:
            target_tags = self._gear_station_tags_by_gear.get(self._preferred_gear, set())
        elif gear is None:
            target_tags = self._gear_station_tags
        elif gear == "aligner":
            target_tags = self._junction_tags if has_heart else self._heart_source_tags
        elif gear == "scrambler":
            target_tags = self._junction_tags if has_heart else self._heart_source_tags
        elif gear == "miner":
            target_tags = self._extractor_tags
        else:
            target_tags = set()

        if target_tags & self._gear_station_tags:
            target_location = self._closest_friendly_gear_station(obs, target_tags)
        else:
            target_location = self._closest_tag_location(obs, target_tags) if target_tags else None
        return self._move_toward(state, target_location)

    def initial_agent_state(self) -> StarterCogState:
        """Get the initial state for a new agent."""
        return StarterCogState(wander_direction_index=self._agent_id % len(WANDER_DIRECTIONS))


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
