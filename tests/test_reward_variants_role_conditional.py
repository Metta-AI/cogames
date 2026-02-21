from cogames.cogs_vs_clips.cog import CogTeam
from cogames.cogs_vs_clips.mission import CvCMission
from cogames.cogs_vs_clips.reward_variants import apply_reward_variants
from cogames.cogs_vs_clips.sites import make_cogsguard_arena_site
from cogames.cogs_vs_clips.variants import ForcedRoleVibesVariant


def test_role_conditional_applies_per_agent_shaping_using_role_id_when_present() -> None:
    mission = CvCMission(
        name="basic",
        description="test",
        site=make_cogsguard_arena_site(num_agents=4),
        teams={"cogs": CogTeam(num_agents=4)},
        max_steps=100,
        variants=[ForcedRoleVibesVariant()],
    )
    env = mission.make_env()
    apply_reward_variants(env, variants=["role_conditional"])

    rewards_by_agent = [agent.rewards for agent in env.game.agents]

    assert "gain_diversity" in rewards_by_agent[0]  # miner
    assert "junction_aligned_by_agent" in rewards_by_agent[1]  # aligner
    assert "junction_scrambled_by_agent" in rewards_by_agent[2]  # scrambler
    assert "cell_visited" in rewards_by_agent[3]  # scout


def test_role_conditional_respects_custom_role_order_from_forced_vibes() -> None:
    mission = CvCMission(
        name="basic",
        description="test",
        site=make_cogsguard_arena_site(num_agents=4),
        teams={"cogs": CogTeam(num_agents=4)},
        max_steps=100,
        variants=[ForcedRoleVibesVariant(role_order=["scout", "miner", "aligner", "scrambler"])],
    )
    env = mission.make_env()
    apply_reward_variants(env, variants=["role_conditional"])

    rewards_by_agent = [agent.rewards for agent in env.game.agents]

    assert "cell_visited" in rewards_by_agent[0]  # scout
    assert "gain_diversity" in rewards_by_agent[1]  # miner
    assert "junction_aligned_by_agent" in rewards_by_agent[2]  # aligner
    assert "junction_scrambled_by_agent" in rewards_by_agent[3]  # scrambler
