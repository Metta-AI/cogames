from __future__ import annotations

import logging
from dataclasses import dataclass

from cogames.policy.starter_agent import ELEMENTS, StarterCogPolicyImpl, StarterCogState
from mettagrid.policy.policy import MultiAgentPolicy, StatefulAgentPolicy, StatefulPolicyImpl
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action
from mettagrid.simulator.interface import AgentObservation

logger = logging.getLogger("cogames.policy.mine_closest")


@dataclass
class MineClosestState(StarterCogState):
    last_mode: str = "bootstrap"
    remembered_hub_row_from_spawn: int | None = None
    remembered_hub_col_from_spawn: int | None = None


class MineClosestPolicyImpl(StatefulPolicyImpl[MineClosestState]):
    """A simple miner loop: get miner gear, mine the nearest extractor, return to hub when carrying resources."""

    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int, return_load: int):
        self._starter = StarterCogPolicyImpl(policy_env_info, agent_id, preferred_gear="miner")
        self._hub_tags = self._starter._resolve_tag_ids(["hub"])
        miner_station_names = self._miner_station_names(policy_env_info)
        self._miner_station_tags = self._starter._resolve_tag_ids(miner_station_names)
        self._return_load = return_load

    def _miner_station_names(self, policy_env_info: PolicyEnvInterface) -> list[str]:
        names = {"miner_station"}
        for tag_name in policy_env_info.tags:
            if not tag_name.startswith("type:"):
                continue
            object_name = tag_name.removeprefix("type:")
            if object_name.endswith(":miner") or object_name == "miner":
                names.add(object_name)
        return sorted(names)

    def initial_agent_state(self) -> MineClosestState:
        starter_state = self._starter.initial_agent_state()
        return MineClosestState(
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

    def _spawn_offset(self, obs: AgentObservation) -> tuple[int, int]:
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

    def _remember_visible_hub(self, obs: AgentObservation, state: MineClosestState) -> None:
        hub_loc = self._starter._closest_tag_location(obs, self._hub_tags)
        if hub_loc is None:
            return
        current_row, current_col = self._spawn_offset(obs)
        delta_row = hub_loc[0] - self._starter._center[0]
        delta_col = hub_loc[1] - self._starter._center[1]
        state.remembered_hub_row_from_spawn = current_row + delta_row
        state.remembered_hub_col_from_spawn = current_col + delta_col

    def _move_toward_remembered_hub(
        self, obs: AgentObservation, state: MineClosestState
    ) -> tuple[Action, MineClosestState] | None:
        if state.remembered_hub_row_from_spawn is None or state.remembered_hub_col_from_spawn is None:
            return None
        current_row, current_col = self._spawn_offset(obs)
        target = (
            self._starter._center[0] + (state.remembered_hub_row_from_spawn - current_row),
            self._starter._center[1] + (state.remembered_hub_col_from_spawn - current_col),
        )
        action, next_state = self._starter._move_toward(state, target)
        return (
            action,
            MineClosestState(
                wander_direction_index=next_state.wander_direction_index,
                wander_steps_remaining=next_state.wander_steps_remaining,
                last_mode=state.last_mode,
                remembered_hub_row_from_spawn=state.remembered_hub_row_from_spawn,
                remembered_hub_col_from_spawn=state.remembered_hub_col_from_spawn,
            ),
        )

    def _gear_up(self, obs: AgentObservation, state: MineClosestState) -> tuple[Action, MineClosestState]:
        self._remember_visible_hub(obs, state)
        if state.last_mode != "gear_up":
            logger.info("agent=%s mode=gear_up", obs.agent_id)
            state.last_mode = "gear_up"
        target = self._starter._closest_tag_location(obs, self._miner_station_tags)
        action, next_state = self._starter._move_toward(state, target)
        return action, MineClosestState(
            wander_direction_index=next_state.wander_direction_index,
            wander_steps_remaining=next_state.wander_steps_remaining,
            last_mode=state.last_mode,
            remembered_hub_row_from_spawn=state.remembered_hub_row_from_spawn,
            remembered_hub_col_from_spawn=state.remembered_hub_col_from_spawn,
        )

    def _mine_until_full(self, obs: AgentObservation, state: MineClosestState) -> tuple[Action, MineClosestState]:
        self._remember_visible_hub(obs, state)
        if state.last_mode != "mine_until_full":
            logger.info("agent=%s mode=mine_until_full", obs.agent_id)
            state.last_mode = "mine_until_full"
        action, next_state = self._starter.step_with_state(obs, state)
        return action, MineClosestState(
            wander_direction_index=next_state.wander_direction_index,
            wander_steps_remaining=next_state.wander_steps_remaining,
            last_mode=state.last_mode,
            remembered_hub_row_from_spawn=state.remembered_hub_row_from_spawn,
            remembered_hub_col_from_spawn=state.remembered_hub_col_from_spawn,
        )

    def _deposit_to_hub(self, obs: AgentObservation, state: MineClosestState) -> tuple[Action, MineClosestState]:
        self._remember_visible_hub(obs, state)
        if state.last_mode != "deposit_to_hub":
            logger.info("agent=%s mode=deposit_to_hub load=%s", obs.agent_id, self._carried_total(obs))
            state.last_mode = "deposit_to_hub"
        target = self._starter._closest_tag_location(obs, self._hub_tags)
        if target is not None:
            action, next_state = self._starter._move_toward(state, target)
            return action, MineClosestState(
                wander_direction_index=next_state.wander_direction_index,
                wander_steps_remaining=next_state.wander_steps_remaining,
                last_mode=state.last_mode,
                remembered_hub_row_from_spawn=state.remembered_hub_row_from_spawn,
                remembered_hub_col_from_spawn=state.remembered_hub_col_from_spawn,
            )

        remembered_move = self._move_toward_remembered_hub(obs, state)
        if remembered_move is not None:
            return remembered_move

        action, next_state = self._starter._wander(state)
        return action, MineClosestState(
            wander_direction_index=next_state.wander_direction_index,
            wander_steps_remaining=next_state.wander_steps_remaining,
            last_mode=state.last_mode,
            remembered_hub_row_from_spawn=state.remembered_hub_row_from_spawn,
            remembered_hub_col_from_spawn=state.remembered_hub_col_from_spawn,
        )

    def step_with_state(self, obs: AgentObservation, state: MineClosestState) -> tuple[Action, MineClosestState]:
        gear = self._starter._current_gear(self._starter._inventory_items(obs))
        if gear != "miner":
            return self._gear_up(obs, state)

        if self._carried_total(obs) >= self._return_load:
            return self._deposit_to_hub(obs, state)

        return self._mine_until_full(obs, state)


class MineClosestPolicy(MultiAgentPolicy):
    short_names = ["mine_closest"]

    def __init__(self, policy_env_info: PolicyEnvInterface, device: str = "cpu", return_load: int | str = 40):
        super().__init__(policy_env_info, device=device)
        self._return_load = int(return_load)
        self._agent_policies: dict[int, StatefulAgentPolicy[MineClosestState]] = {}

    def agent_policy(self, agent_id: int) -> StatefulAgentPolicy[MineClosestState]:
        if agent_id not in self._agent_policies:
            self._agent_policies[agent_id] = StatefulAgentPolicy(
                MineClosestPolicyImpl(self._policy_env_info, agent_id, self._return_load),
                self._policy_env_info,
                agent_id=agent_id,
            )
        return self._agent_policies[agent_id]
