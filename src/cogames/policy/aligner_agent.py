from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field, replace

from cogames.policy.starter_agent import StarterCogPolicyImpl, StarterCogState
from mettagrid.policy.policy import StatefulPolicyImpl
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action
from mettagrid.simulator.interface import AgentObservation

logger = logging.getLogger("cogames.policy.aligner_agent")

Coord = tuple[int, int]
_DIRECTION_DELTAS: tuple[tuple[str, Coord], ...] = (
    ("north", (-1, 0)),
    ("east", (0, 1)),
    ("south", (1, 0)),
    ("west", (0, -1)),
)
_HUB_ALIGN_DISTANCE = 25
_JUNCTION_ALIGN_DISTANCE = 15


@dataclass
class AlignerState(StarterCogState):
    last_mode: str = "bootstrap"
    known_free_cells: set[Coord] = field(default_factory=set)
    blocked_cells: set[Coord] = field(default_factory=set)
    known_hubs: set[Coord] = field(default_factory=set)
    known_aligner_stations: set[Coord] = field(default_factory=set)
    known_neutral_junctions: set[Coord] = field(default_factory=set)
    known_friendly_junctions: set[Coord] = field(default_factory=set)
    known_enemy_junctions: set[Coord] = field(default_factory=set)


class AlignerPolicyImpl(StatefulPolicyImpl[AlignerState]):
    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        self._starter = StarterCogPolicyImpl(policy_env_info, agent_id, preferred_gear="aligner")
        self._team_tag = self._tag_id("team:cogs")
        self._net_tag = self._tag_id("net:cogs")
        self._enemy_team_tag = self._tag_id("team:clips")
        self._enemy_net_tag = self._tag_id("net:clips")
        self._hub_tags = self._starter._resolve_tag_ids(["hub"])
        self._junction_tags = self._starter._resolve_tag_ids(["junction"])
        self._aligner_station_tags = self._starter._resolve_tag_ids(self._gear_station_names(policy_env_info.tags))
        self._wall_tags = self._starter._resolve_tag_ids(["wall"])
        self._obs_radius_row = self._starter._center[0]
        self._obs_radius_col = self._starter._center[1]

    def _tag_id(self, name: str) -> int | None:
        return self._starter._tag_name_to_id.get(name)

    def _gear_station_names(self, all_tags: list[str]) -> list[str]:
        names = {"aligner_station"}
        for tag_name in all_tags:
            if not tag_name.startswith("type:"):
                continue
            object_name = tag_name.removeprefix("type:")
            if object_name == "aligner" or object_name.endswith(":aligner"):
                names.add(object_name)
        return sorted(names)

    def initial_agent_state(self) -> AlignerState:
        starter_state = self._starter.initial_agent_state()
        return AlignerState(
            wander_direction_index=starter_state.wander_direction_index,
            wander_steps_remaining=starter_state.wander_steps_remaining,
        )

    def _spawn_offset(self, obs: AgentObservation) -> Coord:
        row = 0
        col = 0
        for token in obs.tokens:
            name = token.feature.name
            value = int(token.value)
            if name == "lp:north":
                row -= value
            elif name == "lp:south":
                row += value
            elif name == "lp:east":
                col += value
            elif name == "lp:west":
                col -= value
        return row, col

    def _visible_abs_cell(self, current_abs: Coord, location: Coord) -> Coord:
        return (
            current_abs[0] + (location[0] - self._starter._center[0]),
            current_abs[1] + (location[1] - self._starter._center[1]),
        )

    def _visible_abs_cells(self, current_abs: Coord) -> set[Coord]:
        cells: set[Coord] = set()
        for d_row in range(-self._obs_radius_row, self._obs_radius_row + 1):
            for d_col in range(-self._obs_radius_col, self._obs_radius_col + 1):
                cells.add((current_abs[0] + d_row, current_abs[1] + d_col))
        return cells

    def _neighbors(self, cell: Coord) -> list[tuple[str, Coord]]:
        return [(name, (cell[0] + delta[0], cell[1] + delta[1])) for name, delta in _DIRECTION_DELTAS]

    def _ordered_neighbors_toward(self, cell: Coord, goal: Coord) -> list[tuple[str, Coord]]:
        return sorted(
            self._neighbors(cell),
            key=lambda item: (
                abs(item[1][0] - goal[0]) + abs(item[1][1] - goal[1]),
                item[0] != "west",
                item[0] != "east",
                item[0] != "north",
                item[0] != "south",
            ),
        )

    def _nearest_known(self, current_abs: Coord, candidates: set[Coord]) -> Coord | None:
        if not candidates:
            return None
        return min(candidates, key=lambda coord: (abs(coord[0] - current_abs[0]) + abs(coord[1] - current_abs[1]), coord))

    def _bfs_first_direction(self, state: AlignerState, start: Coord, goal: Coord) -> str | None:
        if start == goal:
            return self._starter._fallback_action_name
        if goal not in state.known_free_cells:
            return None
        frontier: deque[Coord] = deque([start])
        parents: dict[Coord, tuple[Coord, str] | None] = {start: None}
        while frontier:
            cell = frontier.popleft()
            if cell == goal:
                break
            for direction, neighbor in self._ordered_neighbors_toward(cell, goal):
                if neighbor in parents or neighbor not in state.known_free_cells:
                    continue
                parents[neighbor] = (cell, direction)
                frontier.append(neighbor)
        if goal not in parents:
            return None
        step = goal
        while parents[step] is not None and parents[step][0] != start:
            step = parents[step][0]
        if parents[step] is None:
            return None
        return parents[step][1]

    def _move_to(self, state: AlignerState, current_abs: Coord, target_abs: Coord | None) -> tuple[Action, AlignerState]:
        if target_abs is None:
            return self._starter._wander(state)
        direction = self._bfs_first_direction(state, current_abs, target_abs)
        if direction is None:
            return self._starter._wander(state)
        return self._starter._action(f"move_{direction}"), state

    def _frontier_cells(self, state: AlignerState) -> set[Coord]:
        frontier: set[Coord] = set()
        for cell in state.known_free_cells:
            for _, neighbor in self._neighbors(cell):
                if neighbor not in state.known_free_cells and neighbor not in state.blocked_cells:
                    frontier.add(cell)
                    break
        return frontier

    def _inventory_count(self, obs: AgentObservation, item: str) -> int:
        for token in obs.tokens:
            if token.location != self._starter._center:
                continue
            if token.feature.name == f"inv:{item}":
                return int(token.value)
        return 0

    def _current_gear(self, obs: AgentObservation) -> str | None:
        return self._starter._current_gear(self._starter._inventory_items(obs))

    def _update_known_objects(self, visible_cells: set[Coord], target_set: set[Coord], current_values: set[Coord]) -> None:
        target_set.difference_update(visible_cells)
        target_set.update(current_values)

    def _update_map_memory(self, obs: AgentObservation, state: AlignerState) -> Coord:
        current_abs = self._spawn_offset(obs)
        visible_cells = self._visible_abs_cells(current_abs)
        visible_tag_ids_by_cell: dict[Coord, set[int]] = {}
        blocked_now: set[Coord] = set()
        hubs_now: set[Coord] = set()
        stations_now: set[Coord] = set()

        for token in obs.tokens:
            if token.feature.name != "tag" or token.location is None:
                continue
            abs_cell = self._visible_abs_cell(current_abs, token.location)
            visible_tag_ids_by_cell.setdefault(abs_cell, set()).add(int(token.value))
            if token.value in self._wall_tags:
                blocked_now.add(abs_cell)
            if token.value in self._hub_tags:
                hubs_now.add(abs_cell)
            if token.value in self._aligner_station_tags:
                stations_now.add(abs_cell)

        neutral_now: set[Coord] = set()
        friendly_now: set[Coord] = set()
        enemy_now: set[Coord] = set()
        for abs_cell, tag_ids in visible_tag_ids_by_cell.items():
            if not (tag_ids & self._junction_tags):
                continue
            if (self._team_tag in tag_ids) or (self._net_tag in tag_ids):
                friendly_now.add(abs_cell)
            elif (self._enemy_team_tag in tag_ids) or (self._enemy_net_tag in tag_ids):
                enemy_now.add(abs_cell)
            else:
                neutral_now.add(abs_cell)

        state.blocked_cells.difference_update(visible_cells)
        state.blocked_cells.update(blocked_now)
        state.known_free_cells.update(visible_cells - blocked_now)
        state.known_free_cells.difference_update(state.blocked_cells)
        state.known_free_cells.add(current_abs)

        self._update_known_objects(visible_cells, state.known_hubs, hubs_now)
        self._update_known_objects(visible_cells, state.known_aligner_stations, stations_now)
        self._update_known_objects(visible_cells, state.known_neutral_junctions, neutral_now)
        self._update_known_objects(visible_cells, state.known_friendly_junctions, friendly_now)
        self._update_known_objects(visible_cells, state.known_enemy_junctions, enemy_now)
        state.known_neutral_junctions.difference_update(state.known_friendly_junctions)
        state.known_neutral_junctions.difference_update(state.known_enemy_junctions)
        return current_abs

    def _log_mode(self, obs: AgentObservation, state: AlignerState, mode: str) -> None:
        if state.last_mode != mode:
            logger.info("agent=%s mode=%s", obs.agent_id, mode)
            state.last_mode = mode

    def _explore(self, obs: AgentObservation, state: AlignerState) -> tuple[Action, AlignerState]:
        self._log_mode(obs, state, "explore")
        action, next_state = self._starter._wander(state)
        return action, replace(next_state, last_mode=state.last_mode)

    def _gear_up(self, obs: AgentObservation, state: AlignerState, current_abs: Coord) -> tuple[Action, AlignerState]:
        self._log_mode(obs, state, "gear_up")
        target_abs = self._nearest_known(current_abs, state.known_aligner_stations)
        if target_abs is None:
            return self._explore(obs, state)
        action, next_state = self._move_to(state, current_abs, target_abs)
        return action, replace(next_state, last_mode=state.last_mode)

    def _get_heart(self, obs: AgentObservation, state: AlignerState, current_abs: Coord) -> tuple[Action, AlignerState]:
        self._log_mode(obs, state, "get_heart")
        target_abs = self._nearest_known(current_abs, state.known_hubs)
        if target_abs is None:
            return self._explore(obs, state)
        action, next_state = self._move_to(state, current_abs, target_abs)
        return action, replace(next_state, last_mode=state.last_mode)

    def _is_alignable(self, junction: Coord, state: AlignerState) -> bool:
        for hub in state.known_hubs:
            if abs(junction[0] - hub[0]) + abs(junction[1] - hub[1]) <= _HUB_ALIGN_DISTANCE:
                return True
        for friendly in state.known_friendly_junctions:
            if abs(junction[0] - friendly[0]) + abs(junction[1] - friendly[1]) <= _JUNCTION_ALIGN_DISTANCE:
                return True
        return False

    def _align_neutral(self, obs: AgentObservation, state: AlignerState, current_abs: Coord) -> tuple[Action, AlignerState]:
        alignable = {junction for junction in state.known_neutral_junctions if self._is_alignable(junction, state)}
        target_abs = self._nearest_known(current_abs, alignable)
        if target_abs is None:
            return self._explore(obs, state)
        self._log_mode(obs, state, "align_neutral")
        action, next_state = self._move_to(state, current_abs, target_abs)
        return action, replace(next_state, last_mode=state.last_mode)

    def step_with_state(self, obs: AgentObservation, state: AlignerState) -> tuple[Action, AlignerState]:
        current_abs = self._update_map_memory(obs, state)
        if self._current_gear(obs) != "aligner":
            return self._gear_up(obs, state, current_abs)
        if self._inventory_count(obs, "heart") <= 0:
            return self._get_heart(obs, state, current_abs)
        return self._align_neutral(obs, state, current_abs)
