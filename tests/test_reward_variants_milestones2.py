import pytest

from cogames.cogs_vs_clips.cog import CogTeam
from cogames.cogs_vs_clips.mission import CvCMission
from cogames.cogs_vs_clips.reward_variants import apply_reward_variants
from cogames.cogs_vs_clips.sites import make_cogsguard_arena_site
from mettagrid.config.game_value import Scope, StatValue, SumGameValue


def _make_env():
    mission = CvCMission(
        name="basic",
        description="test",
        site=make_cogsguard_arena_site(num_agents=4),
        teams={"cogs": CogTeam(num_agents=4)},
        max_steps=100,
    )
    return mission.make_env()


@pytest.mark.parametrize("factor", ["nan", "inf", "-inf", "1e309", "-1e309"])
def test_milestones_2_rejects_non_finite_compounding_factor(factor: str) -> None:
    env = _make_env()

    with pytest.raises(ValueError, match="must be finite"):
        apply_reward_variants(env, variants=[f"milestones_2:{factor}"])


@pytest.mark.parametrize("factor", ["0", "-1"])
def test_milestones_2_rejects_non_positive_compounding_factor(factor: str) -> None:
    env = _make_env()

    with pytest.raises(ValueError, match="must be > 0"):
        apply_reward_variants(env, variants=[f"milestones_2:{factor}"])


def test_milestones_2_accepts_finite_positive_compounding_factor() -> None:
    env = _make_env()

    apply_reward_variants(env, variants=["milestones_2:1.5"])

    assert env.label.endswith(".milestones_2")


def test_milestones_2_wires_role_shaping_without_caps() -> None:
    env = _make_env()

    apply_reward_variants(env, variants=["milestones_2"])

    rewards = env.game.agents[0].rewards
    gained_reward = rewards["milestones2_elements_gained"].reward
    assert isinstance(gained_reward, SumGameValue)
    assert gained_reward.log is True

    deposited_reward = rewards["milestones2_elements_deposited"].reward
    assert isinstance(deposited_reward, SumGameValue)
    assert deposited_reward.log is True
    assert all(isinstance(value, StatValue) and value.scope == Scope.GAME for value in deposited_reward.values)

    aligned_reward = rewards["milestones2_junction_aligned_by_agent"].reward
    assert isinstance(aligned_reward, SumGameValue)
    assert aligned_reward.weights == pytest.approx([0.3])
