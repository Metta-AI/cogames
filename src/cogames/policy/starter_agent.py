"""
Sample policy for the CoGames CvC environment.

This starter policy keeps one fixed role per agent:
- miners gather ore and deposit it at friendly territory
- aligners get hearts and capture nearby neutral junctions

Note to users of this policy:
We don't intend for scripted policies to be the final word on how policies are generated (e.g., we expect the
environment to be complicated enough that trained agents will be necessary). So we expect that scripting policies
is a good way to start, but don't want you to get stuck here. Feel free to prove us wrong!

Note to cogames developers:
This policy should be kept relatively minimalist, without dependencies on intricate algorithms.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from mettagrid.policy.policy import MultiAgentPolicy, StatefulAgentPolicy, StatefulPolicyImpl
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action
from mettagrid.simulator.interface import AgentObservation

STARTER_ROLE_CYCLE = ("miner", "aligner")
ALL_ROLES = ("miner", "aligner", "scrambler", "scout")
ELEMENTS = ("carbon", "oxygen", "germanium", "silicon")
WANDER_DIRECTIONS = ("east", "south", "west", "north")
TEAM_TAG_PREFIX = "team:"
MAX_REMEMBERED_JUNCTION_DISTANCE = 24
MAX_ALIGNER_JUNCTION_FRONTIER_DISTANCE = 25
MAX_ALIGNER_RETURN_TO_FRONTIER_DISTANCE = 20
MOVE_DELTAS = {
    "north": (-1, 0),
    "south": (1, 0),
    "west": (0, -1),
    "east": (0, 1),
}
type Coordinate = tuple[int, int]


@dataclass
class StarterCogState:
    """Small amount of map memory so starter cogs can avoid tiny local loops."""

    explore_direction_index: int
    position: Coordinate = (0, 0)
    visited: set[Coordinate] = field(default_factory=lambda: {(0, 0)})
    blocked: set[Coordinate] = field(default_factory=set)
    seen_tags_by_position: dict[Coordinate, set[int]] = field(default_factory=dict)
    last_move_direction: str | None = None


class StarterCogPolicyImpl(StatefulPolicyImpl[StarterCogState]):
    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        agent_id: int,
        role: str | None = None,
    ):
        self._policy_env_info = policy_env_info
        self._role = role or STARTER_ROLE_CYCLE[agent_id % len(STARTER_ROLE_CYCLE)]
        self._explore_direction_start = (ALL_ROLES.index(self._role) + agent_id // len(STARTER_ROLE_CYCLE)) % len(
            WANDER_DIRECTIONS
        )

        self._center = (policy_env_info.obs_height // 2, policy_env_info.obs_width // 2)
        self._tag_name_to_id = {name: idx for idx, name in enumerate(policy_env_info.tags)}

        # Starter policies only target the current canonical CvC action and tag contract. If that contract changes, we
        # want the policy to fail loudly instead of carrying compat shims forever.
        required_action_names = {"noop"} | {f"move_{direction}" for direction in MOVE_DELTAS}
        missing_action_names = required_action_names - set(policy_env_info.action_names)
        assert not missing_action_names, f"Starter policy requires actions {sorted(missing_action_names)}"

        required_tag_names = {
            "type:agent",
            "type:wall",
            "type:hub",
            "type:junction",
            *(f"type:{role_name}" for role_name in ALL_ROLES),
            *(f"type:{element}_extractor" for element in ELEMENTS),
        }
        missing_tag_names = required_tag_names - self._tag_name_to_id.keys()
        assert not missing_tag_names, f"Starter policy requires tags {sorted(missing_tag_names)}"

        self._noop_action_name = "noop"
        self._move_action_names = {direction: f"move_{direction}" for direction in MOVE_DELTAS}
        self._team_tag_ids = {idx for idx, name in enumerate(policy_env_info.tags) if name.startswith(TEAM_TAG_PREFIX)}
        self._agent_tags = {self._tag_name_to_id["type:agent"]}
        self._wall_tags = {self._tag_name_to_id["type:wall"]}
        self._role_station_tags = {role_name: {self._tag_name_to_id[f"type:{role_name}"]} for role_name in ALL_ROLES}
        self._extractor_tags = {self._tag_name_to_id[f"type:{element}_extractor"] for element in ELEMENTS}
        hub_tag_id = self._tag_name_to_id["type:hub"]
        self._junction_tags = {self._tag_name_to_id["type:junction"]}
        self._heart_source_tags = {hub_tag_id}
        if "type:chest" in self._tag_name_to_id:
            self._heart_source_tags.add(self._tag_name_to_id["type:chest"])
        self._deposit_tags = self._junction_tags | {hub_tag_id}

    def _closest_matching_location(
        self,
        tags_by_location: dict[Coordinate, set[int]],
        origin: Coordinate,
        include_tag_ids: set[int],
        *,
        require_tag_ids: set[int] | None = None,
        exclude_tag_ids: set[int] | None = None,
        allow_origin: bool = True,
    ) -> Coordinate | None:
        # Deterministically pick the nearest matching cell so ties stay stable across runs.
        best_location: Coordinate | None = None
        best_key: tuple[int, int, int] | None = None
        for location, location_tag_ids in tags_by_location.items():
            if not allow_origin and location == origin:
                continue
            if not (location_tag_ids & include_tag_ids):
                continue
            if require_tag_ids and not (location_tag_ids & require_tag_ids):
                continue
            if exclude_tag_ids and location_tag_ids & exclude_tag_ids:
                continue

            distance_key = (
                abs(location[0] - origin[0]) + abs(location[1] - origin[1]),
                location[0],
                location[1],
            )
            if best_key is None or distance_key < best_key:
                best_location = location
                best_key = distance_key

        return best_location

    def _move(self, direction: str, state: StarterCogState) -> tuple[Action, StarterCogState]:
        state.last_move_direction = direction
        return Action(name=self._move_action_names[direction]), state

    def _toward_directions(self, delta_row: int, delta_col: int) -> list[str]:
        # Prefer the dominant axis first so direct lines win when both row and column progress are possible.
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
        return direction_candidates

    def _explore(
        self,
        tags_by_location: dict[tuple[int, int], set[int]],
        state: StarterCogState,
    ) -> tuple[Action, StarterCogState]:
        blocked_locations = set(tags_by_location)
        blocked_locations.discard(self._center)
        for prefer_visited in (False, True):
            # First keep pushing into new cells. Only fall back to revisits when the local frontier is boxed in.
            for direction_offset in range(len(WANDER_DIRECTIONS)):
                direction_index = (state.explore_direction_index + direction_offset) % len(WANDER_DIRECTIONS)
                direction = WANDER_DIRECTIONS[direction_index]
                move_delta = MOVE_DELTAS[direction]
                next_location = (self._center[0] + move_delta[0], self._center[1] + move_delta[1])
                next_position = (state.position[0] + move_delta[0], state.position[1] + move_delta[1])
                if (
                    next_location in blocked_locations
                    or next_position in state.blocked
                    or not (0 <= next_location[0] < self._policy_env_info.obs_height)
                    or not (0 <= next_location[1] < self._policy_env_info.obs_width)
                ):
                    continue
                if prefer_visited and next_position not in state.visited:
                    continue
                if not prefer_visited and next_position in state.visited:
                    continue
                state.explore_direction_index = direction_index
                return self._move(direction, state)
        return Action(name=self._noop_action_name), state

    def step_with_state(self, obs: AgentObservation, state: StarterCogState) -> tuple[Action, StarterCogState]:
        """Compute the action for this Cog."""
        # Bucket visible tags by map cell so role logic and pathing can query them cheaply.
        tags_by_location: dict[tuple[int, int], set[int]] = {}
        for token in obs.tokens:
            if token.feature.name != "tag" or token.location is None:
                continue
            tags_by_location.setdefault(token.location, set()).add(token.value)

        # Fold the previous move attempt into map memory, then remember the tags on every visible cell.
        if state.last_move_direction is not None:
            move_delta = MOVE_DELTAS[state.last_move_direction]
            attempted_location = (self._center[0] + move_delta[0], self._center[1] + move_delta[1])
            attempted_position = (state.position[0] + move_delta[0], state.position[1] + move_delta[1])
            if any(token.feature.name == "last_action_move" and bool(token.value) for token in obs.tokens):
                state.position = attempted_position
                state.visited.add(attempted_position)
            elif tags_by_location.get(attempted_location, set()) & self._wall_tags:
                # Only walls become permanent blockers. Another cog in the way is just traffic.
                state.blocked.add(attempted_position)
            state.last_move_direction = None

        for location, tag_ids in tags_by_location.items():
            absolute_location = (
                state.position[0] + location[0] - self._center[0],
                state.position[1] + location[1] - self._center[1],
            )
            state.seen_tags_by_position[absolute_location] = set(tag_ids)
            if location != self._center and tag_ids & self._wall_tags:
                state.blocked.add(absolute_location)

        # Parse center-cell inventory tokens into real item counts. Powered inventory features encode their scaling in
        # the feature name, e.g. `inv:carbon:p1`.
        items: dict[str, int] = {}
        for token in obs.tokens:
            if token.location != self._center or not token.feature.name.startswith("inv:"):
                continue
            suffix = token.feature.name[4:]
            if not suffix or token.value <= 0:
                continue
            item_name, sep, power_str = suffix.rpartition(":p")
            if sep and item_name and power_str.isdigit():
                scale = max(int(token.feature.normalization), 1) ** int(power_str)
            else:
                item_name = suffix
                scale = 1
            items[item_name] = items.get(item_name, 0) + int(token.value) * scale

        own_team_tag_ids = tags_by_location.get(self._center, set()) & self._team_tag_ids
        enemy_team_tag_ids = (self._team_tag_ids - own_team_tag_ids) or self._team_tag_ids
        has_role_gear = items.get(self._role, 0) > 0
        has_heart = items.get("heart", 0) > 0
        cargo_amount = sum(items.get(element, 0) for element in ELEMENTS)
        role_station_tags = self._role_station_tags[self._role]
        own_anchor_positions = [
            position
            for position, tag_ids in state.seen_tags_by_position.items()
            if tag_ids & self._deposit_tags and tag_ids & own_team_tag_ids
        ]
        aligner_frontier_play = self._role == "aligner" and has_heart and bool(own_anchor_positions)

        # Choose one target for the fixed role.
        # Miners: deposit, gear up, then mine. Aligner/scrambler: gear up, grab a heart, then go to junctions.
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
        elif self._role in {"aligner", "scrambler"}:
            target_tag_ids = self._heart_source_tags
            require_tag_ids = own_team_tag_ids
            if has_heart:
                target_tag_ids = self._junction_tags
                if self._role == "aligner":
                    exclude_tag_ids = self._team_tag_ids
                else:
                    require_tag_ids = enemy_team_tag_ids

        target_location = None
        if target_tag_ids is not None:
            if aligner_frontier_play:
                # Heart carriers should spend hearts near our existing territory instead of wandering into deep neutral
                # space.
                best_target_key: tuple[int, int, int, int] | None = None
                for location, location_tag_ids in tags_by_location.items():
                    if not (location_tag_ids & target_tag_ids) or location_tag_ids & self._team_tag_ids:
                        continue
                    absolute_location = (
                        state.position[0] + location[0] - self._center[0],
                        state.position[1] + location[1] - self._center[1],
                    )
                    frontier_distance = min(
                        abs(absolute_location[0] - anchor[0]) + abs(absolute_location[1] - anchor[1])
                        for anchor in own_anchor_positions
                    )
                    if frontier_distance > MAX_ALIGNER_JUNCTION_FRONTIER_DISTANCE:
                        continue
                    distance_to_agent = abs(location[0] - self._center[0]) + abs(location[1] - self._center[1])
                    target_key = (frontier_distance, distance_to_agent, location[0], location[1])
                    if best_target_key is None or target_key < best_target_key:
                        best_target_key = target_key
                        target_location = location
            else:
                target_location = self._closest_matching_location(
                    tags_by_location,
                    self._center,
                    target_tag_ids,
                    require_tag_ids=require_tag_ids,
                    exclude_tag_ids=exclude_tag_ids,
                )

        # If nothing useful is visible, fall back to deterministic exploration.
        if target_location is None:
            if target_tag_ids is not None:
                remembered_target = None
                if aligner_frontier_play:
                    # Keep heart carriers near the friendly frontier so they turn hearts into junction captures quickly.
                    best_target_key: tuple[int, int, int, int] | None = None
                    for position, tag_ids in state.seen_tags_by_position.items():
                        if not (tag_ids & target_tag_ids) or tag_ids & self._team_tag_ids or position == state.position:
                            continue
                        frontier_distance = min(
                            abs(position[0] - anchor[0]) + abs(position[1] - anchor[1])
                            for anchor in own_anchor_positions
                        )
                        if frontier_distance > MAX_ALIGNER_JUNCTION_FRONTIER_DISTANCE:
                            continue
                        distance_to_agent = abs(position[0] - state.position[0]) + abs(position[1] - state.position[1])
                        if distance_to_agent > MAX_REMEMBERED_JUNCTION_DISTANCE:
                            continue
                        target_key = (frontier_distance, distance_to_agent, position[0], position[1])
                        if best_target_key is None or target_key < best_target_key:
                            best_target_key = target_key
                            remembered_target = position

                    if remembered_target is None:
                        nearest_anchor = min(
                            own_anchor_positions,
                            key=lambda anchor: abs(anchor[0] - state.position[0]) + abs(anchor[1] - state.position[1]),
                        )
                        if (
                            abs(nearest_anchor[0] - state.position[0]) + abs(nearest_anchor[1] - state.position[1])
                            > MAX_ALIGNER_RETURN_TO_FRONTIER_DISTANCE
                        ):
                            remembered_target = nearest_anchor
                else:
                    # Remember stable objectives freely, but only chase nearby junction memories so mutable targets stay
                    # local.
                    remembered_target = self._closest_matching_location(
                        state.seen_tags_by_position,
                        state.position,
                        target_tag_ids,
                        require_tag_ids=require_tag_ids,
                        exclude_tag_ids=exclude_tag_ids,
                        allow_origin=False,
                    )
                if remembered_target is not None:
                    # Prefer moves that both close distance to the remembered target and keep expanding fresh cells.
                    delta_row = remembered_target[0] - state.position[0]
                    delta_col = remembered_target[1] - state.position[1]
                    current_distance = abs(delta_row) + abs(delta_col)
                    if target_tag_ids == self._junction_tags and current_distance > MAX_REMEMBERED_JUNCTION_DISTANCE:
                        return self._explore(tags_by_location, state)
                    direction_candidates = self._toward_directions(delta_row, delta_col)
                    direction_candidates.extend(
                        direction for direction in WANDER_DIRECTIONS if direction not in direction_candidates
                    )
                    blocked_locations = set(tags_by_location)
                    blocked_locations.discard(self._center)
                    for prefer_reducing in (True, False):
                        for prefer_unvisited in (True, False):
                            # Try the greedy fresh-cell move first, then relax into sideways or revisiting moves if the
                            # local frontier is crowded.
                            for direction in direction_candidates:
                                move_delta = MOVE_DELTAS[direction]
                                next_location = (
                                    self._center[0] + move_delta[0],
                                    self._center[1] + move_delta[1],
                                )
                                next_position = (
                                    state.position[0] + move_delta[0],
                                    state.position[1] + move_delta[1],
                                )
                                if next_location in blocked_locations or next_position in state.blocked:
                                    continue
                                next_distance = abs(remembered_target[0] - next_position[0]) + abs(
                                    remembered_target[1] - next_position[1]
                                )
                                if prefer_reducing and next_distance >= current_distance:
                                    continue
                                if not prefer_reducing and next_distance < current_distance:
                                    continue
                                if prefer_unvisited and next_position in state.visited:
                                    continue
                                if not prefer_unvisited and next_position not in state.visited:
                                    continue
                                return self._move(direction, state)
            return self._explore(tags_by_location, state)

        # Step directly onto adjacent targets, otherwise route to an open neighbor cell.
        delta_row = target_location[0] - self._center[0]
        delta_col = target_location[1] - self._center[1]
        if delta_row == 0 and delta_col == 0:
            return Action(name=self._noop_action_name), state

        if abs(delta_row) + abs(delta_col) == 1:
            return self._move(self._toward_directions(delta_row, delta_col)[0], state)

        blocked_locations = set(tags_by_location)
        blocked_locations.discard(self._center)
        if not (tags_by_location.get(target_location, set()) & self._agent_tags):
            blocked_locations.discard(target_location)
        # Route toward a free neighbor of the target cell; the target itself is often occupied or interactive.
        goal_locations = {
            (target_location[0] + move_delta[0], target_location[1] + move_delta[1])
            for move_delta in MOVE_DELTAS.values()
            if 0 <= target_location[0] + move_delta[0] < self._policy_env_info.obs_height
            and 0 <= target_location[1] + move_delta[1] < self._policy_env_info.obs_width
            and (target_location[0] + move_delta[0], target_location[1] + move_delta[1]) not in blocked_locations
        }
        if not goal_locations:
            return self._explore(tags_by_location, state)

        queue = deque([self._center])
        visited = {self._center}
        first_directions: dict[tuple[int, int], str] = {}
        while queue:
            current = queue.popleft()
            delta_row = target_location[0] - current[0]
            delta_col = target_location[1] - current[1]
            # Expand BFS in target-facing order so the first route we find is usually the cleanest one.
            for direction in self._toward_directions(delta_row, delta_col):
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
                    return self._move(first_directions[next_location], state)
                queue.append(next_location)

        return self._explore(tags_by_location, state)

    def initial_agent_state(self) -> StarterCogState:
        """Get the initial state for a new agent."""
        return StarterCogState(explore_direction_index=self._explore_direction_start)


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
