"""Tests for the role_conditional reward variant."""

from cogsguard.game import ForcedRoleVibesVariant
from cogsguard.game.damage import DamageVariant
from cogsguard.game.teams import TeamConfig, TeamVariant
from cogsguard.game.vibes import VibesVariant
from cogsguard.missions.arena import make_arena_map_builder
from cogsguard.missions.mission import CvCMission
from cogsguard.train.reward_variants import apply_reward_variants


def test_role_conditional_applies_per_agent_shaping_using_forced_role_vibes() -> None:
    mission = CvCMission(
        name="basic",
        description="test",
        map_builder=make_arena_map_builder(num_agents=4),
        min_cogs=4,
        max_cogs=4,
        max_steps=100,
    ).with_variants(
        [
            TeamVariant(default_teams={"cogs": TeamConfig(num_agents=4)}),
            DamageVariant(),
            VibesVariant(),
            ForcedRoleVibesVariant(),
        ]
    )
    env = mission.make_env()
    apply_reward_variants(env, variants=["role_conditional"])

    rewards_by_agent = [agent.rewards for agent in env.game.agents]

    assert "gain_diversity" in rewards_by_agent[0]  # miner
    assert "loss_diversity" in rewards_by_agent[0]
    assert "junction_aligned_by_agent" in rewards_by_agent[1]  # aligner
    assert "miner_gained" in rewards_by_agent[1]
    assert "scout_gained" in rewards_by_agent[1]
    assert "scrambler_gained" in rewards_by_agent[1]
    assert "junction_scrambled_by_agent" in rewards_by_agent[2]  # scrambler
    assert "miner_gained" in rewards_by_agent[2]
    assert "scout_gained" in rewards_by_agent[2]
    assert "aligner_gained" in rewards_by_agent[2]
    assert "cell_visited" in rewards_by_agent[3]  # scout
    assert "miner_gained" in rewards_by_agent[3]
    assert "scrambler_gained" in rewards_by_agent[3]
    assert "aligner_gained" in rewards_by_agent[3]


def test_role_conditional_respects_custom_role_order_from_forced_vibes() -> None:
    mission = CvCMission(
        name="basic",
        description="test",
        map_builder=make_arena_map_builder(num_agents=4),
        min_cogs=4,
        max_cogs=4,
        max_steps=100,
    ).with_variants(
        [
            TeamVariant(default_teams={"cogs": TeamConfig(num_agents=4)}),
            DamageVariant(),
            VibesVariant(),
            ForcedRoleVibesVariant(role_order=["scout", "miner", "aligner", "scrambler"]),
        ]
    )
    env = mission.make_env()
    apply_reward_variants(env, variants=["role_conditional"])

    rewards_by_agent = [agent.rewards for agent in env.game.agents]

    assert "cell_visited" in rewards_by_agent[0]  # scout
    assert "gain_diversity" in rewards_by_agent[1]  # miner
    assert "loss_diversity" in rewards_by_agent[1]
    assert "junction_aligned_by_agent" in rewards_by_agent[2]  # aligner
    assert "miner_gained" in rewards_by_agent[2]
    assert "scout_gained" in rewards_by_agent[2]
    assert "scrambler_gained" in rewards_by_agent[2]
    assert "junction_scrambled_by_agent" in rewards_by_agent[3]  # scrambler
    assert "miner_gained" in rewards_by_agent[3]
    assert "scout_gained" in rewards_by_agent[3]
    assert "aligner_gained" in rewards_by_agent[3]


def test_role_conditional_falls_back_to_default_role_order_without_explicit_vibes() -> None:
    mission = CvCMission(
        name="basic",
        description="test",
        map_builder=make_arena_map_builder(num_agents=4),
        min_cogs=4,
        max_cogs=4,
        max_steps=100,
    ).with_variants(
        [
            TeamVariant(default_teams={"cogs": TeamConfig(num_agents=4)}),
            DamageVariant(),
            VibesVariant(),
        ]
    )
    env = mission.make_env()
    apply_reward_variants(env, variants=["role_conditional"])
    rewards_by_agent = [agent.rewards for agent in env.game.agents]

    assert "gain_diversity" in rewards_by_agent[0]
    assert "junction_aligned_by_agent" in rewards_by_agent[1]
    assert "junction_scrambled_by_agent" in rewards_by_agent[2]
    assert "cell_visited" in rewards_by_agent[3]
    assert "gain_diversity" in rewards_by_agent[4]
    assert "junction_aligned_by_agent" in rewards_by_agent[5]
    assert "junction_scrambled_by_agent" in rewards_by_agent[6]
    assert "cell_visited" in rewards_by_agent[7]
