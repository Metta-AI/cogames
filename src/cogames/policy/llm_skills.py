from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field, replace

from cogames.policy.starter_agent import ELEMENTS, StarterCogPolicyImpl, StarterCogState
from mettagrid.policy.policy import StatefulPolicyImpl
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action
from mettagrid.simulator.interface import AgentObservation

logger = logging.getLogger("cogames.policy.llm_skills")

Coord = tuple[int, int]
_HUB_SEARCH_DISTANCE = 20
_HUB_EXTRACTOR_OFFSETS: tuple[Coord, ...] = ((-8, -8), (-8, 8), (8, -8), (8, 8))
_DIRECTION_DELTAS: tuple[tuple[str, Coord], ...] = (
    ("north", (-1, 0)),
    ("east", (0, 1)),
    ("south", (1, 0)),
    ("west", (0, -1)),
)


@dataclass
class MinerSkillState(StarterCogState):
    last_mode: str = "bootstrap"
    remembered_hub_row_from_spawn: int | None = None
    remembered_hub_col_from_spawn: int | None = None
    known_free_cells: set[Coord] = field(default_factory=set)
    blocked_cells: set[Coord] = field(default_factory=set)
    known_hubs: set[Coord] = field(default_factory=set)
    known_miner_stations: set[Coord] = field(default_factory=set)
    known_extractors: set[Coord] = field(default_factory=set)
    known_hazard_stations: set[Coord] = field(default_factory=set)


class MinerSkillImpl(StatefulPolicyImpl[MinerSkillState]):
    """Bounded miner skills plus navigation primitives shared by scripted and LLM-controlled miners."""

    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int, return_load: int):
        self._starter = StarterCogPolicyImpl(policy_env_info, agent_id, preferred_gear="miner")
        self._policy_env_info = policy_env_info
        self._hub_tags = self._starter._resolve_tag_ids(["hub"])
        miner_station_names = self._miner_station_names(policy_env_info)
        self._miner_station_tags = self._starter._resolve_tag_ids(miner_station_names)
        self._hazard_station_tags = self._resolve_non_miner_station_tags(policy_env_info, miner_station_names)
        self._wall_tags = self._starter._resolve_tag_ids(["wall"])
        self._return_load = return_load
        self._obs_radius_row = self._starter._center[0]
        self._obs_radius_col = self._starter._center[1]

    def _miner_station_names(self, policy_env_info: PolicyEnvInterface) -> list[str]:
        names = {"miner_station"}
        for tag_name in policy_env_info.tags:
            if not tag_name.startswith("type:"):
                continue
            object_name = tag_name.removeprefix("type:")
            if object_name.endswith(":miner") or object_name == "miner":
                names.add(object_name)
        return sorted(names)

    def _resolve_non_miner_station_tags(self, policy_env_info: PolicyEnvInterface, miner_names: list[str]) -> set[int]:
        other_gear = ("aligner", "scrambler", "scout")
        names: set[str] = set()
        for gear in other_gear:
            names.add(f"{gear}_station")
            for tag_name in policy_env_info.tags:
                if not tag_name.startswith("type:"):
                    continue
                object_name = tag_name.removeprefix("type:")
                if object_name.endswith(f":{gear}") or object_name == gear:
                    names.add(object_name)
        return self._starter._resolve_tag_ids(sorted(names))

    def initial_agent_state(self) -> MinerSkillState:
        starter_state = self._starter.initial_agent_state()
        return MinerSkillState(
            wander_direction_index=starter_state.wander_direction_index,
            wander_steps_remaining=starter_state.wander_steps_remaining,
        )

    def _inventory_counts(self, obs: AgentObservation) -> dict[str, int]:
        counts: dict[str, int] = {}
        center = self._starter._center
        for token in obs.tokens:
            if token.location != center:
                continue
            name = token.feature.name
            if not name.startswith("inv:"):
                continue
            parts = name.split(":", 2)
            if len(parts) >= 2 and parts[1] in ELEMENTS:
                counts[parts[1]] = int(token.value)
        return counts

    def _carried_total(self, obs: AgentObservation) -> int:
        return sum(self._inventory_counts(obs).values())

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

    def _current_abs(self, obs: AgentObservation) -> Coord:
        return self._spawn_offset(obs)

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

    def _remember_static_objects(self, target_set: set[Coord], current_values: set[Coord]) -> None:
        target_set.update(current_values)

    def _remember_visible_hub(self, obs: AgentObservation, state: MinerSkillState) -> None:
        if not state.known_hubs:
            return
        hub_row, hub_col = min(state.known_hubs, key=lambda coord: abs(coord[0]) + abs(coord[1]))
        state.remembered_hub_row_from_spawn = hub_row
        state.remembered_hub_col_from_spawn = hub_col

    def _update_map_memory(self, obs: AgentObservation, state: MinerSkillState) -> None:
        current_abs = self._current_abs(obs)
        visible_cells = self._visible_abs_cells(current_abs)
        blocked_now: set[Coord] = set()
        hubs_now: set[Coord] = set()
        miner_stations_now: set[Coord] = set()
        extractors_now: set[Coord] = set()
        hazard_stations_now: set[Coord] = set()

        for token in obs.tokens:
            if token.feature.name != "tag" or token.location is None:
                continue
            abs_cell = self._visible_abs_cell(current_abs, token.location)
            if token.value in self._wall_tags:
                blocked_now.add(abs_cell)
            if token.value in self._hub_tags:
                hubs_now.add(abs_cell)
            if token.value in self._miner_station_tags:
                miner_stations_now.add(abs_cell)
            if token.value in self._starter._extractor_tags:
                extractors_now.add(abs_cell)
            if token.value in self._hazard_station_tags:
                hazard_stations_now.add(abs_cell)

        state.blocked_cells.difference_update(visible_cells)
        state.blocked_cells.update(blocked_now)
        state.known_free_cells.update(visible_cells - blocked_now)
        state.known_free_cells.difference_update(state.blocked_cells)
        state.known_free_cells.add(current_abs)

        self._remember_static_objects(state.known_hubs, hubs_now)
        self._remember_static_objects(state.known_miner_stations, miner_stations_now)
        self._remember_static_objects(state.known_extractors, extractors_now)
        self._remember_static_objects(state.known_hazard_stations, hazard_stations_now)
        self._remember_visible_hub(obs, state)

    def _neighbors(self, cell: Coord) -> list[tuple[str, Coord]]:
        return [(name, (cell[0] + delta[0], cell[1] + delta[1])) for name, delta in _DIRECTION_DELTAS]

    def _nearest_known(self, current_abs: Coord, candidates: set[Coord]) -> Coord | None:
        if not candidates:
            return None
        return min(candidates, key=lambda coord: (abs(coord[0] - current_abs[0]) + abs(coord[1] - current_abs[1]), coord))

    def _closest_visible_location(self, obs: AgentObservation, tag_ids: set[int]) -> Coord | None:
        return self._starter._closest_tag_location(obs, tag_ids)

    def _frontier_cells(self, state: MinerSkillState) -> set[Coord]:
        frontier: set[Coord] = set()
        for cell in state.known_free_cells:
            for _, neighbor in self._neighbors(cell):
                if neighbor not in state.known_free_cells and neighbor not in state.blocked_cells:
                    frontier.add(cell)
                    break
        return frontier

    def _frontier_near(self, state: MinerSkillState, anchors: set[Coord], max_anchor_distance: int) -> set[Coord]:
        frontier = self._frontier_cells(state)
        if not anchors:
            return frontier
        near_frontier: set[Coord] = set()
        for cell in frontier:
            if min(abs(cell[0] - anchor[0]) + abs(cell[1] - anchor[1]) for anchor in anchors) <= max_anchor_distance:
                near_frontier.add(cell)
        return near_frontier or frontier

    def _predicted_extractor_positions(self, state: MinerSkillState) -> set[Coord]:
        predicted: set[Coord] = set()
        for hub_row, hub_col in state.known_hubs:
            for d_row, d_col in _HUB_EXTRACTOR_OFFSETS:
                predicted.add((hub_row + d_row, hub_col + d_col))
        return predicted

    def _bfs_first_direction(self, state: MinerSkillState, start: Coord, goal: Coord) -> str | None:
        if start == goal:
            return self._starter._fallback_action_name
        if goal not in state.known_free_cells:
            return None
        avoid = state.known_hazard_stations - {goal}
        frontier: deque[Coord] = deque([start])
        parents: dict[Coord, tuple[Coord, str] | None] = {start: None}
        while frontier:
            cell = frontier.popleft()
            if cell == goal:
                break
            for direction, neighbor in self._neighbors(cell):
                if neighbor in parents or neighbor not in state.known_free_cells or neighbor in avoid:
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

    def _bfs_optimistic_direction(self, state: MinerSkillState, start: Coord, goal: Coord, max_cells: int = 20000) -> str | None:
        """Optimistic BFS: treat unknown cells as traversable, only avoid known walls."""
        if start == goal:
            return self._starter._fallback_action_name
        frontier: deque[Coord] = deque([start])
        parents: dict[Coord, tuple[Coord, str] | None] = {start: None}
        while frontier and len(parents) < max_cells:
            cell = frontier.popleft()
            if cell == goal:
                break
            for direction, neighbor in self._neighbors(cell):
                if neighbor in parents or neighbor in state.blocked_cells:
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

    def _move_to(self, state: MinerSkillState, current_abs: Coord, target_abs: Coord | None) -> tuple[Action, MinerSkillState]:
        if target_abs is None:
            return self._starter._wander(state)
        direction = self._bfs_first_direction(state, current_abs, target_abs)
        if direction is None:
            return self._starter._wander(state)
        return self._starter._action(f"move_{direction}"), state

    def _move_toward_target(
        self,
        state: MinerSkillState,
        current_abs: Coord,
        target_abs: Coord | None,
    ) -> tuple[Action, MinerSkillState]:
        if target_abs is None:
            return self._starter._wander(state)
        direction = self._bfs_first_direction(state, current_abs, target_abs)
        if direction is not None:
            return self._starter._action(f"move_{direction}"), state

        # BFS failed (path requires unexplored territory) - try optimistic BFS through unknown cells
        direction = self._bfs_optimistic_direction(state, current_abs, target_abs)
        if direction is not None:
            return self._starter._action(f"move_{direction}"), state

        frontier_cells = self._frontier_cells(state)
        if not frontier_cells:
            return self._starter._wander(state)

        best_frontier = min(
            frontier_cells,
            key=lambda cell: (
                abs(cell[0] - target_abs[0]) + abs(cell[1] - target_abs[1]),
                abs(cell[0] - current_abs[0]) + abs(cell[1] - current_abs[1]),
                cell,
            ),
        )
        if current_abs == best_frontier:
            for direction_name, neighbor in sorted(
                self._neighbors(current_abs),
                key=lambda item: (
                    item[1] in state.blocked_cells,
                    item[1] in state.known_free_cells,
                    abs(item[1][0] - target_abs[0]) + abs(item[1][1] - target_abs[1]),
                ),
            ):
                if neighbor in state.blocked_cells or neighbor in state.known_free_cells:
                    continue
                return self._starter._action(f"move_{direction_name}"), state
            return self._starter._wander(state)
        return self._move_to(state, current_abs, best_frontier)

    def _explore(self, obs: AgentObservation, state: MinerSkillState) -> tuple[Action, MinerSkillState]:
        if state.last_mode != "explore":
            logger.info("agent=%s mode=explore", obs.agent_id)
            state.last_mode = "explore"
        current_abs = self._current_abs(obs)
        frontier_cells = self._frontier_cells(state)
        if current_abs in frontier_cells:
            for direction, neighbor in self._neighbors(current_abs):
                if neighbor in state.blocked_cells:
                    continue
                if neighbor not in state.known_free_cells:
                    return self._starter._action(f"move_{direction}"), replace(state, last_mode=state.last_mode)
        target_abs = self._nearest_known(current_abs, frontier_cells)
        action, next_state = self._move_to(state, current_abs, target_abs)
        return action, replace(next_state, last_mode=state.last_mode)

    def _explore_near_hub(self, obs: AgentObservation, state: MinerSkillState) -> tuple[Action, MinerSkillState]:
        if state.last_mode != "explore":
            logger.info("agent=%s mode=explore", obs.agent_id)
            state.last_mode = "explore"
        current_abs = self._current_abs(obs)
        frontier_cells = self._frontier_near(state, state.known_hubs, max_anchor_distance=_HUB_SEARCH_DISTANCE)
        if current_abs in frontier_cells:
            ordered = sorted(
                self._neighbors(current_abs),
                key=lambda item: (
                    item[1] in state.blocked_cells,
                    item[1] in state.known_free_cells,
                    min(
                        (abs(item[1][0] - hub[0]) + abs(item[1][1] - hub[1]) for hub in state.known_hubs),
                        default=9999,
                    ),
                ),
            )
            for direction, neighbor in ordered:
                if neighbor in state.blocked_cells or neighbor in state.known_free_cells:
                    continue
                return self._starter._action(f"move_{direction}"), replace(state, last_mode=state.last_mode)
        target_abs = self._nearest_known(current_abs, frontier_cells)
        action, next_state = self._move_to(state, current_abs, target_abs)
        return action, replace(next_state, last_mode=state.last_mode)

    def _gear_up(self, obs: AgentObservation, state: MinerSkillState) -> tuple[Action, MinerSkillState]:
        if state.last_mode != "gear_up":
            logger.info("agent=%s mode=gear_up", obs.agent_id)
            state.last_mode = "gear_up"
        current_abs = self._current_abs(obs)
        visible_target = self._closest_visible_location(obs, self._miner_station_tags)
        if visible_target is not None:
            target_abs = self._visible_abs_cell(current_abs, visible_target)
            action, next_state = self._move_toward_target(state, current_abs, target_abs)
            return action, replace(next_state, last_mode=state.last_mode)
        target_abs = self._nearest_known(current_abs, state.known_miner_stations)
        if target_abs is None:
            if state.known_hubs:
                return self._explore_near_hub(obs, state)
            return self._explore(obs, state)
        action, next_state = self._move_toward_target(state, current_abs, target_abs)
        return action, replace(next_state, last_mode=state.last_mode)

    def _mine_until_full(self, obs: AgentObservation, state: MinerSkillState) -> tuple[Action, MinerSkillState]:
        if state.last_mode != "mine_until_full":
            logger.info("agent=%s mode=mine_until_full", obs.agent_id)
            state.last_mode = "mine_until_full"
        current_abs = self._current_abs(obs)
        visible_target = self._closest_visible_location(obs, self._starter._extractor_tags)
        if visible_target is not None:
            target_abs = self._visible_abs_cell(current_abs, visible_target)
            action, next_state = self._move_toward_target(state, current_abs, target_abs)
            return action, replace(next_state, last_mode=state.last_mode)
        target_abs = self._nearest_known(current_abs, state.known_extractors)
        if target_abs is None:
            if state.known_hubs:
                predicted = self._predicted_extractor_positions(state)
                predicted_target = self._nearest_known(current_abs, predicted)
                if predicted_target is not None:
                    action, next_state = self._move_toward_target(state, current_abs, predicted_target)
                    return action, replace(next_state, last_mode=state.last_mode)
                return self._explore_near_hub(obs, state)
            return self._explore(obs, state)
        action, next_state = self._move_toward_target(state, current_abs, target_abs)
        return action, replace(next_state, last_mode=state.last_mode)

    def _deposit_to_hub(self, obs: AgentObservation, state: MinerSkillState) -> tuple[Action, MinerSkillState]:
        if state.last_mode != "deposit_to_hub":
            logger.info("agent=%s mode=deposit_to_hub load=%s", obs.agent_id, self._carried_total(obs))
            state.last_mode = "deposit_to_hub"
        current_abs = self._current_abs(obs)
        visible_target = self._closest_visible_location(obs, self._hub_tags)
        if visible_target is not None:
            target_abs = self._visible_abs_cell(current_abs, visible_target)
            action, next_state = self._move_toward_target(state, current_abs, target_abs)
            return action, replace(next_state, last_mode=state.last_mode)
        target_abs = self._nearest_known(current_abs, state.known_hubs)
        if target_abs is None and state.remembered_hub_row_from_spawn is not None and state.remembered_hub_col_from_spawn is not None:
            target_abs = (state.remembered_hub_row_from_spawn, state.remembered_hub_col_from_spawn)
            state.known_free_cells.add(target_abs)
        if target_abs is None:
            return self._explore(obs, state)
        action, next_state = self._move_toward_target(state, current_abs, target_abs)
        return action, replace(next_state, last_mode=state.last_mode)

    def step_with_state(self, obs: AgentObservation, state: MinerSkillState) -> tuple[Action, MinerSkillState]:
        self._update_map_memory(obs, state)
        gear = self._starter._current_gear(self._starter._inventory_items(obs))
        if gear != "miner":
            return self._gear_up(obs, state)

        if self._carried_total(obs) >= self._return_load:
            return self._deposit_to_hub(obs, state)

        return self._mine_until_full(obs, state)
