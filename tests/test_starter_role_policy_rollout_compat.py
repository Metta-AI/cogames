from __future__ import annotations

import pytest

from cogames.cogs_vs_clips.missions import get_core_missions
from cogames.cogs_vs_clips.tutorials.aligner_tutorial import AlignerTutorialMission
from cogames.cogs_vs_clips.tutorials.miner_tutorial import MinerTutorialMission
from cogames.cogs_vs_clips.tutorials.scout_tutorial import ScoutTutorialMission
from cogames.cogs_vs_clips.tutorials.scrambler_tutorial import ScramblerTutorialMission
from cogames.policy.starter_agent import StarterCogPolicyImpl
from mettagrid.policy.policy import PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.runner.rollout import run_episode_local

ROLES = ("miner", "aligner", "scrambler", "scout")
ROLE_POLICY_CLASS_PATH = {
    "miner": "cogames.policy.role_policies.MinerRolePolicy",
    "aligner": "cogames.policy.role_policies.AlignerRolePolicy",
    "scrambler": "cogames.policy.role_policies.ScramblerRolePolicy",
    "scout": "cogames.policy.role_policies.ScoutRolePolicy",
}


@pytest.mark.parametrize("mission", get_core_missions(), ids=lambda m: m.full_name())
def test_starter_role_policy_resolves_current_cogsguard_tags(mission) -> None:
    policy_env_info = PolicyEnvInterface.from_mg_cfg(mission.make_env())
    assert "type:hub" in policy_env_info.tags
    assert not any(tag.endswith("_station") for tag in policy_env_info.tags)

    hub_tag_id = policy_env_info.tags.index("type:hub")
    for role in ROLES:
        impl = StarterCogPolicyImpl(policy_env_info, agent_id=0, preferred_gear=role)
        expected_station_tag = policy_env_info.tags.index(f"type:c:{role}")
        assert impl._gear_station_tags_by_gear[role] == {expected_station_tag}
        assert hub_tag_id in impl._heart_source_tags


@pytest.mark.parametrize(
    ("role", "mission"),
    [
        ("miner", MinerTutorialMission),
        ("aligner", AlignerTutorialMission),
        ("scrambler", ScramblerTutorialMission),
        ("scout", ScoutTutorialMission),
    ],
)
def test_role_policies_complete_role_objective_on_tutorials(role: str, mission) -> None:
    env_cfg = mission.make_env()
    env_cfg.game.max_steps = 1000
    results, _ = run_episode_local(
        policy_specs=[PolicySpec(class_path=ROLE_POLICY_CLASS_PATH[role])],
        assignments=[0] * env_cfg.game.num_agents,
        env=env_cfg,
        seed=42,
        max_action_time_ms=10000,
        render_mode="none",
        device="cpu",
    )

    agent_stats = results.stats["agent"]
    assert any(agent.get(f"{role}.gained", 0) > 0 for agent in agent_stats)

    if role == "miner":
        mined_total = sum(
            agent.get("carbon.gained", 0)
            + agent.get("oxygen.gained", 0)
            + agent.get("germanium.gained", 0)
            + agent.get("silicon.gained", 0)
            for agent in agent_stats
        )
        assert mined_total > 0
        return

    if role == "aligner":
        assert any(agent.get("heart.gained", 0) > 0 for agent in agent_stats)
        assert sum(agent.get("junction.aligned_by_agent", 0) for agent in agent_stats) > 0
        return

    if role == "scrambler":
        assert any(agent.get("heart.gained", 0) > 0 for agent in agent_stats)
        assert sum(agent.get("junction.scrambled_by_agent", 0) for agent in agent_stats) > 0
        return

    assert sum(agent.get("cell.visited", 0) for agent in agent_stats) > 0
