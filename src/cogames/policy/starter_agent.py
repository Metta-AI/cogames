"""
Sample policy for the CoGames CvC environment.

This starter policy keeps one fixed role per agent:
- miners gather ore and deposit it at friendly territory
- aligners get hearts and capture neutral junctions
- scramblers get hearts and neutralize enemy junctions
- scouts gear up, then explore

Note to users of this policy:
We don't intend for scripted policies to be the final word on how policies are generated (e.g., we expect the
environment to be complicated enough that trained agents will be necessary). So we expect that scripting policies
is a good way to start, but don't want you to get stuck here. Feel free to prove us wrong!

Note to cogames developers:
This policy should be kept relatively minimalist, without dependencies on intricate algorithms.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

from mettagrid.policy.policy import MultiAgentPolicy, StatefulAgentPolicy, StatefulPolicyImpl
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action
from mettagrid.simulator.interface import AgentObservation

STARTER_ROLE_CYCLE = ("miner", "aligner", "scrambler", "scout")
ELEMENTS = ("carbon", "oxygen", "germanium", "silicon")
WANDER_DIRECTIONS = ("east", "south", "west", "north")
TEAM_TAG_PREFIX = "team:"
MOVE_DELTAS = {
    "north": (-1, 0),
    "south": (1, 0),
    "west": (0, -1),
    "east": (0, 1),
}

type StarterCogState = None


class StarterCogPolicyImpl(StatefulPolicyImpl[StarterCogState]):
    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        agent_id: int,
        role: str | None = None,
    ):
        self._policy_env_info = policy_env_info
        self._role = role or STARTER_ROLE_CYCLE[agent_id % len(STARTER_ROLE_CYCLE)]
        self._explore_direction_start = (
            STARTER_ROLE_CYCLE.index(self._role) + agent_id // len(STARTER_ROLE_CYCLE)
        ) % len(WANDER_DIRECTIONS)

        action_names = policy_env_info.action_names
        self._action_name_set = set(action_names)
        self._fallback_action_name = "noop" if "noop" in self._action_name_set else action_names[0]
        self._center = (policy_env_info.obs_height // 2, policy_env_info.obs_width // 2)
        self._tag_name_to_id = {name: idx for idx, name in enumerate(policy_env_info.tags)}

        self._team_tag_ids = {idx for idx, name in enumerate(policy_env_info.tags) if name.startswith(TEAM_TAG_PREFIX)}
        self._agent_tags = self._resolve_tag_ids(["agent"])
        self._role_station_tags = {
            role_name: self._resolve_tag_ids([role_name, f"c:{role_name}"]) for role_name in STARTER_ROLE_CYCLE
        }
        self._extractor_tags = self._resolve_tag_ids([f"{element}_extractor" for element in ELEMENTS])
        self._junction_tags = self._resolve_tag_ids(["junction"])
        self._heart_source_tags = self._resolve_tag_ids(["hub", "chest"])
        self._deposit_tags = self._resolve_tag_ids(["hub", "junction"])

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
                scale = 1
            else:
                scale = max(int(token.feature.normalization), 1) ** int(power_str)
            value = int(token.value)
            if value <= 0:
                continue
            items[item_name] = items.get(item_name, 0) + value * scale
        return items

    def _closest_visible_location(
        self,
        tags_by_location: dict[tuple[int, int], set[int]],
        include_tag_ids: set[int],
        *,
        require_tag_ids: set[int] | None = None,
        exclude_tag_ids: set[int] | None = None,
    ) -> tuple[int, int] | None:
        return min(
            (
                location
                for location, location_tag_ids in tags_by_location.items()
                if location_tag_ids & include_tag_ids
                and (require_tag_ids is None or not require_tag_ids or location_tag_ids & require_tag_ids)
                and (exclude_tag_ids is None or not (location_tag_ids & exclude_tag_ids))
            ),
            key=lambda location: (
                abs(location[0] - self._center[0]) + abs(location[1] - self._center[1]),
                location[0],
                location[1],
            ),
            default=None,
        )

    def _action(self, name: str) -> Action:
        return Action(name=name if name in self._action_name_set else self._fallback_action_name)

    def _explore(self, tags_by_location: dict[tuple[int, int], set[int]]) -> Action:
        blocked_locations = set(tags_by_location)
        blocked_locations.discard(self._center)
        for direction_offset in range(len(WANDER_DIRECTIONS)):
            direction_index = (self._explore_direction_start + direction_offset) % len(WANDER_DIRECTIONS)
            direction = WANDER_DIRECTIONS[direction_index]
            move_delta = MOVE_DELTAS[direction]
            next_location = (self._center[0] + move_delta[0], self._center[1] + move_delta[1])
            if (
                next_location in blocked_locations
                or not (0 <= next_location[0] < self._policy_env_info.obs_height)
                or not (0 <= next_location[1] < self._policy_env_info.obs_width)
            ):
                continue
            return self._action(f"move_{direction}")
        return self._action(self._fallback_action_name)

    def step_with_state(self, obs: AgentObservation, state: StarterCogState) -> tuple[Action, StarterCogState]:
        """Compute the action for this Cog."""
        items = self._inventory_amounts(obs)

        # Bucket visible tags by map cell so role logic and pathing can query them cheaply.
        tags_by_location: dict[tuple[int, int], set[int]] = {}
        for token in obs.tokens:
            if token.feature.name != "tag" or token.location is None:
                continue
            tags_by_location.setdefault(token.location, set()).add(token.value)

        own_team_tag_ids = tags_by_location.get(self._center, set()) & self._team_tag_ids
        enemy_team_tag_ids = (self._team_tag_ids - own_team_tag_ids) or self._team_tag_ids

        has_role_gear = items.get(self._role, 0) > 0
        has_heart = items.get("heart", 0) > 0
        cargo_amount = sum(items.get(element, 0) for element in ELEMENTS)
        role_station_tags = self._role_station_tags[self._role]

        # Choose one visible target for the fixed role.
        target_tag_ids: set[int] | None = None
        require_tag_ids: set[int] | None = None
        exclude_tag_ids: set[int] | None = None
        if self._role == "miner":
            if cargo_amount > 0:
                target_tag_ids = self._deposit_tags
                require_tag_ids = own_team_tag_ids
            elif not has_role_gear:
                target_tag_ids = role_station_tags
                require_tag_ids = own_team_tag_ids
            else:
                target_tag_ids = self._extractor_tags
        elif not has_role_gear:
            target_tag_ids = role_station_tags
            require_tag_ids = own_team_tag_ids
        elif self._role != "scout":
            target_tag_ids = self._heart_source_tags
            require_tag_ids = own_team_tag_ids
            if has_heart:
                target_tag_ids = self._junction_tags
                require_tag_ids = enemy_team_tag_ids if self._role == "scrambler" else None
                exclude_tag_ids = self._team_tag_ids if self._role == "aligner" else None

        target_location = (
            None
            if target_tag_ids is None
            else self._closest_visible_location(
                tags_by_location,
                target_tag_ids,
                require_tag_ids=require_tag_ids,
                exclude_tag_ids=exclude_tag_ids,
            )
        )

        # If nothing useful is visible, fall back to deterministic exploration.
        if target_location is None:
            return self._explore(tags_by_location), state

        # Step directly onto adjacent targets, otherwise route to an open neighbor cell.
        delta_row = target_location[0] - self._center[0]
        delta_col = target_location[1] - self._center[1]
        if delta_row == 0 and delta_col == 0:
            return self._action(self._fallback_action_name), state

        if abs(delta_row) + abs(delta_col) == 1:
            if abs(delta_row) >= abs(delta_col):
                direction = "south" if delta_row > 0 else "north"
            else:
                direction = "east" if delta_col > 0 else "west"
            return self._action(f"move_{direction}"), state

        blocked_locations = set(tags_by_location)
        blocked_locations.discard(self._center)
        if not (tags_by_location.get(target_location, set()) & self._agent_tags):
            blocked_locations.discard(target_location)
        goal_locations = {
            (target_location[0] + move_delta[0], target_location[1] + move_delta[1])
            for move_delta in MOVE_DELTAS.values()
            if 0 <= target_location[0] + move_delta[0] < self._policy_env_info.obs_height
            and 0 <= target_location[1] + move_delta[1] < self._policy_env_info.obs_width
            and (target_location[0] + move_delta[0], target_location[1] + move_delta[1]) not in blocked_locations
        }
        if not goal_locations:
            return self._explore(tags_by_location), state

        queue = deque([self._center])
        visited = {self._center}
        first_directions: dict[tuple[int, int], str] = {}
        while queue:
            current = queue.popleft()
            delta_row = target_location[0] - current[0]
            delta_col = target_location[1] - current[1]
            direction_candidates: list[str] = []
            if abs(delta_row) >= abs(delta_col):
                if delta_row != 0:
                    direction_candidates.append("south" if delta_row > 0 else "north")
                if delta_col != 0:
                    direction_candidates.append("east" if delta_col > 0 else "west")
            else:
                if delta_col != 0:
                    direction_candidates.append("east" if delta_col > 0 else "west")
                if delta_row != 0:
                    direction_candidates.append("south" if delta_row > 0 else "north")
            for direction in direction_candidates:
                move_delta = MOVE_DELTAS[direction]
                next_location = (current[0] + move_delta[0], current[1] + move_delta[1])
                if (
                    next_location in visited
                    or next_location in blocked_locations
                    or not (0 <= next_location[0] < self._policy_env_info.obs_height)
                    or not (0 <= next_location[1] < self._policy_env_info.obs_width)
                ):
                    continue
                visited.add(next_location)
                first_directions[next_location] = first_directions.get(current, direction)
                if next_location in goal_locations:
                    return self._action(f"move_{first_directions[next_location]}"), state
                queue.append(next_location)

        return self._explore(tags_by_location), state

    def initial_agent_state(self) -> StarterCogState:
        """Get the initial state for a new agent."""
        return None


class BaseStarterPolicy(MultiAgentPolicy):
    short_names: list[str]
    _role: str | None = None

    def __init__(self, policy_env_info: PolicyEnvInterface, device: str = "cpu"):
        super().__init__(policy_env_info, device=device)
        self._agent_policies: dict[int, StatefulAgentPolicy[StarterCogState]] = {}

    def agent_policy(self, agent_id: int) -> StatefulAgentPolicy[StarterCogState]:
        if agent_id not in self._agent_policies:
            self._agent_policies[agent_id] = StatefulAgentPolicy(
                StarterCogPolicyImpl(self._policy_env_info, agent_id, role=self._role),
                self._policy_env_info,
                agent_id=agent_id,
            )
        return self._agent_policies[agent_id]


class StarterPolicy(BaseStarterPolicy):
    short_names = ["starter"]
