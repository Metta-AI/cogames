from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from cogsguard.evals.cvc_evals import (
    CVC_EVAL_COGS,
    CVC_EVAL_MISSIONS,
)

from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
from mettagrid.simulator import Simulator


def test_cvc_eval_cog_counts() -> None:
    expected = {Path(map_name).stem: count for map_name, count in CVC_EVAL_COGS.items()}
    mission_by_name = {mission.name: mission for mission in CVC_EVAL_MISSIONS}

    assert set(expected) == set(mission_by_name)

    for name, count in expected.items():
        mission = mission_by_name[name]
        assert mission.num_cogs == count
        assert mission.min_cogs == count
        assert mission.max_cogs == count


@pytest.mark.parametrize("mission", CVC_EVAL_MISSIONS, ids=lambda m: m.full_name())
def test_cvc_eval_mission_smoke(mission) -> None:
    env_cfg = mission.make_env()
    env_cfg.game.max_steps = 5

    simulator = Simulator()
    env = MettaGridPufferEnv(simulator, env_cfg)
    try:
        observations, _ = env.reset(seed=123)
        assert observations.shape[0] == env_cfg.game.num_agents

        assert env._sim is not None
        noop_idx = env._sim.action_names.index("noop")
        actions = np.full(env_cfg.game.num_agents, noop_idx, dtype=np.int32)

        next_obs, rewards, terminals, truncations, _ = env.step(actions)
        assert next_obs.shape == observations.shape
        assert rewards.shape == (env_cfg.game.num_agents,)
        assert terminals.shape == (env_cfg.game.num_agents,)
        assert truncations.shape == (env_cfg.game.num_agents,)
    finally:
        env.close()
        simulator.close()
