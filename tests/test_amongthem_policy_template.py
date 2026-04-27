from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces as gym_spaces

from cogames.policy.amongthem_policy_template import AmongThemPolicy
from mettagrid.bitworld import BITWORLD_ACTION_COUNT, bitworld_action_names
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import AgentObservation


def _bitworld_env_info() -> PolicyEnvInterface:
    return PolicyEnvInterface.from_spaces(
        observation_space=gym_spaces.Box(low=0, high=15, shape=(4, 128, 128), dtype=np.uint8),
        action_space=gym_spaces.Discrete(BITWORLD_ACTION_COUNT),
        num_agents=5,
        action_names=bitworld_action_names(),
        observation_kind="pixels",
    )


def test_amongthem_policy_template_emits_valid_bitworld_actions() -> None:
    policy = AmongThemPolicy(_bitworld_env_info(), hold_ticks=2)
    raw_observations = np.zeros((5, 4, 128, 128), dtype=np.uint8)
    raw_observations[:, 0, 0, 0] = np.arange(5, dtype=np.uint8)
    raw_actions = np.full(5, -1, dtype=np.int32)

    policy.step_batch(raw_observations, raw_actions)

    assert np.all((0 <= raw_actions) & (raw_actions < BITWORLD_ACTION_COUNT))
    assert len(set(raw_actions.tolist())) > 1


def test_amongthem_policy_template_keeps_agent_and_batch_ticks_aligned() -> None:
    env_info = _bitworld_env_info()
    batch_policy = AmongThemPolicy(env_info, hold_ticks=1)
    agent_policy = AmongThemPolicy(env_info, hold_ticks=1)

    raw_observations = np.zeros((env_info.num_agents, 4, 128, 128), dtype=np.uint8)
    batch_actions = np.full(env_info.num_agents, -1, dtype=np.int32)
    batch_policy.step_batch(raw_observations, batch_actions)

    single_actions = [
        agent_policy.agent_policy(agent_id).step(AgentObservation(agent_id=agent_id, tokens=())).name
        for agent_id in range(env_info.num_agents)
    ]

    assert single_actions == [bitworld_action_names()[action] for action in batch_actions]


def test_amongthem_policy_template_requires_bitworld_action_space() -> None:
    env_info = PolicyEnvInterface.from_spaces(
        observation_space=gym_spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
        action_space=gym_spaces.Discrete(1),
        num_agents=1,
        action_names=["noop"],
    )

    with pytest.raises(ValueError, match="BitWorld AmongThem action space"):
        AmongThemPolicy(env_info)
