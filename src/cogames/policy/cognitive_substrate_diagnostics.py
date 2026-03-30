from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation

Coordinate = tuple[int, int]

_MOVE_DELTAS: dict[str, Coordinate] = {
    "move_north": (-1, 0),
    "move_south": (1, 0),
    "move_west": (0, -1),
    "move_east": (0, 1),
}
_MOVE_ORDER = ("move_north", "move_south", "move_west", "move_east")


def _add(pos: Coordinate, delta: Coordinate) -> Coordinate:
    return pos[0] + delta[0], pos[1] + delta[1]


def _direction_between(start: Coordinate, end: Coordinate) -> str:
    delta = end[0] - start[0], end[1] - start[1]
    for action_name, action_delta in _MOVE_DELTAS.items():
        if action_delta == delta:
            return action_name
    raise ValueError(f"Unsupported step from {start} to {end}")


@dataclass
class _NavigatorState:
    position: Coordinate = (0, 0)
    visited: set[Coordinate] = field(default_factory=lambda: {(0, 0)})
    open_cells: set[Coordinate] = field(default_factory=lambda: {(0, 0)})
    blocked: set[Coordinate] = field(default_factory=set)
    occupied: set[Coordinate] = field(default_factory=set)
    parents: dict[Coordinate, Coordinate] = field(default_factory=dict)
    last_move_action: str | None = None
    actor_tags: set[str] = field(default_factory=set)
    seen_types: dict[str, set[Coordinate]] = field(default_factory=lambda: defaultdict(set))


class _NavigatorPolicy(AgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)
        self._tag_names = policy_env_info.tags
        self._center = (policy_env_info.obs_height // 2, policy_env_info.obs_width // 2)
        self._noop_action = Action(name="noop")
        self._state = _NavigatorState()

    def reset(self, simulation=None) -> None:
        _ = simulation
        self._state = _NavigatorState()

    def step(self, obs: AgentObservation) -> Action:
        last_action_move = self._last_action_move(obs)
        self._apply_previous_move(last_action_move)
        self._update_from_observation(obs)
        action_name = self._choose_action()
        if action_name in _MOVE_DELTAS:
            self._state.last_move_action = action_name
        else:
            self._state.last_move_action = None
        return Action(name=action_name)

    def _last_action_move(self, obs: AgentObservation) -> bool:
        for token in obs.tokens:
            if token.feature.name == "last_action_move":
                return bool(token.value)
        return False

    def _apply_previous_move(self, moved: bool) -> None:
        previous_action = self._state.last_move_action
        if previous_action is None:
            return
        target = _add(self._state.position, _MOVE_DELTAS[previous_action])
        if moved:
            previous_position = self._state.position
            self._state.position = target
            if target not in self._state.visited:
                self._state.parents[target] = previous_position
            self._state.visited.add(target)
        else:
            self._state.blocked.add(target)
        self._state.last_move_action = None

    def _update_from_observation(self, obs: AgentObservation) -> None:
        self._state.actor_tags.clear()
        visible_types: defaultdict[str, set[Coordinate]] = defaultdict(set)
        wall_positions: set[Coordinate] = set()
        occupied_positions: set[Coordinate] = set()
        for token in obs.tokens:
            if token.feature.name != "tag" or token.location is None:
                continue
            tag_name = self._tag_names[token.value]
            relative = token.location.row - self._center[0], token.location.col - self._center[1]
            absolute = _add(self._state.position, relative)
            if absolute == self._state.position:
                self._state.actor_tags.add(tag_name)
                continue
            if tag_name == "type:wall":
                self._state.blocked.add(absolute)
                wall_positions.add(absolute)
                continue
            if tag_name.startswith("type:"):
                type_name = tag_name[5:]
                visible_types[type_name].add(absolute)
                if type_name != "agent":
                    occupied_positions.add(absolute)

        for type_name, positions in visible_types.items():
            self._state.seen_types[type_name].update(positions)
        self._state.occupied.update(occupied_positions)

        for row in range(self._center[0] * 2 + 1):
            for col in range(self._center[1] * 2 + 1):
                relative = row - self._center[0], col - self._center[1]
                absolute = _add(self._state.position, relative)
                if absolute in wall_positions or absolute in occupied_positions:
                    continue
                self._state.open_cells.add(absolute)

    def _adjacent_positions(self, type_name: str) -> list[Coordinate]:
        adjacent: list[Coordinate] = []
        for position in sorted(self._state.seen_types.get(type_name, set())):
            if abs(position[0] - self._state.position[0]) + abs(position[1] - self._state.position[1]) == 1:
                adjacent.append(position)
        return adjacent

    def _traversable_positions(self) -> set[Coordinate]:
        traversable = set(self._state.open_cells)
        traversable.add(self._state.position)
        traversable.difference_update(self._state.blocked)
        traversable.difference_update(self._state.occupied)
        return traversable

    def _route_to_positions(self, target_positions: set[Coordinate]) -> str | None:
        traversable = self._traversable_positions()
        targets = target_positions & traversable
        if not targets:
            return None

        frontier: deque[Coordinate] = deque([self._state.position])
        parents: dict[Coordinate, Coordinate | None] = {self._state.position: None}

        while frontier:
            position = frontier.popleft()
            if position in targets and position != self._state.position:
                break
            for action_name in _MOVE_ORDER:
                next_pos = _add(position, _MOVE_DELTAS[action_name])
                if next_pos in parents or next_pos not in traversable:
                    continue
                parents[next_pos] = position
                frontier.append(next_pos)
        else:
            return None

        while parents[position] != self._state.position:
            parent = parents[position]
            if parent is None:
                return None
            position = parent
        return _direction_between(self._state.position, position)

    def _route_to_known_neighbors(self, target_positions: set[Coordinate]) -> str | None:
        traversable = self._traversable_positions()
        target_neighbors = {
            _add(target_position, delta)
            for target_position in target_positions
            for delta in _MOVE_DELTAS.values()
            if _add(target_position, delta) in traversable
        }
        return self._route_to_positions(target_neighbors)

    def _expand_positions(self) -> set[Coordinate]:
        traversable = self._traversable_positions()
        expansion_positions: set[Coordinate] = set()
        for position in traversable:
            for delta in _MOVE_DELTAS.values():
                neighbor = _add(position, delta)
                if (
                    neighbor not in self._state.blocked
                    and neighbor not in self._state.occupied
                    and neighbor not in self._state.open_cells
                ):
                    expansion_positions.add(position)
                    break
        return expansion_positions

    def _explore_move(self) -> str:
        for action_name in _MOVE_ORDER:
            next_pos = _add(self._state.position, _MOVE_DELTAS[action_name])
            if next_pos in self._state.blocked or next_pos in self._state.occupied:
                continue
            if next_pos in self._state.open_cells and next_pos not in self._state.visited:
                return action_name
        frontier_route = self._route_to_positions(set(self._state.open_cells) - self._state.visited)
        if frontier_route is not None:
            return frontier_route
        expansion_route = self._route_to_positions(self._expand_positions())
        if expansion_route is not None:
            return expansion_route
        parent = self._state.parents.get(self._state.position)
        if parent is not None:
            return _direction_between(self._state.position, parent)
        return "noop"

    def _choose_action(self) -> str:
        raise NotImplementedError


class _MemoryAgentPolicy(_NavigatorPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)
        self._cue_index = 0
        self._pending_goal_attempt: Coordinate | None = None
        self._rejected_goal_positions: set[Coordinate] = set()

    def reset(self, simulation=None) -> None:
        super().reset(simulation)
        self._cue_index = 0
        self._pending_goal_attempt = None
        self._rejected_goal_positions = set()

    def step(self, obs: AgentObservation) -> Action:
        last_action_move = self._last_action_move(obs)
        self._apply_previous_move(last_action_move)
        self._update_from_observation(obs)
        if self._pending_goal_attempt is not None:
            if "state:solved" not in self._state.actor_tags:
                self._rejected_goal_positions.add(self._pending_goal_attempt)
            self._pending_goal_attempt = None
        action_name = self._choose_action()
        if action_name in _MOVE_DELTAS:
            self._state.last_move_action = action_name
        else:
            self._state.last_move_action = None
        return Action(name=action_name)

    def _choose_action(self) -> str:
        for type_name in self._state.seen_types:
            if type_name.startswith("cue_"):
                self._cue_index = int(type_name.rsplit("_", 1)[1]) - 1
                break

        goal_positions = sorted(self._state.seen_types.get("memory_goal", set()) - self._rejected_goal_positions)
        if len(goal_positions) > self._cue_index:
            target_position = goal_positions[self._cue_index]
            if target_position in self._adjacent_positions("memory_goal"):
                self._pending_goal_attempt = target_position
                return _direction_between(self._state.position, target_position)
            route = self._route_to_known_neighbors({target_position})
            if route is not None:
                return route

        return self._explore_move()


class _ExplorationAgentPolicy(_NavigatorPolicy):
    def _choose_action(self) -> str:
        goal_positions = set(self._state.seen_types.get("goal", set()))
        adjacent_goals = self._adjacent_positions("goal")
        if adjacent_goals:
            return _direction_between(self._state.position, adjacent_goals[0])
        if goal_positions:
            route = self._route_to_known_neighbors(goal_positions)
            if route is not None:
                return route
        return self._explore_move()


class _PlanningAgentPolicy(_NavigatorPolicy):
    def _target_type(self) -> str:
        if "state:key" not in self._state.actor_tags:
            return "key"
        if "state:switch" not in self._state.actor_tags:
            return "switch"
        if "state:door" not in self._state.actor_tags:
            return "door"
        return "goal"

    def _choose_action(self) -> str:
        target_type = self._target_type()
        target_positions = set(self._state.seen_types.get(target_type, set()))
        adjacent_targets = self._adjacent_positions(target_type)
        if adjacent_targets:
            return _direction_between(self._state.position, adjacent_targets[0])
        if target_positions:
            route = self._route_to_known_neighbors(target_positions)
            if route is not None:
                return route
        return self._explore_move()


class _SingleAgentDiagnosticsPolicy(MultiAgentPolicy):
    _policy_cls: type[AgentPolicy]

    def __init__(self, policy_env_info: PolicyEnvInterface, device: str = "cpu"):
        super().__init__(policy_env_info, device=device)
        self._agent_policies: dict[int, AgentPolicy] = {}

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        if agent_id not in self._agent_policies:
            self._agent_policies[agent_id] = self._policy_cls(self._policy_env_info)
        return self._agent_policies[agent_id]


class SubstrateMemoryPolicy(_SingleAgentDiagnosticsPolicy):
    short_names = ["substrate_memory"]
    _policy_cls = _MemoryAgentPolicy


class SubstrateExplorationPolicy(_SingleAgentDiagnosticsPolicy):
    short_names = ["substrate_exploration"]
    _policy_cls = _ExplorationAgentPolicy


class SubstratePlanningPolicy(_SingleAgentDiagnosticsPolicy):
    short_names = ["substrate_planning"]
    _policy_cls = _PlanningAgentPolicy
