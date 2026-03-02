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
def test_objective_mine_rejects_non_finite_compounding_factor(factor: str) -> None:
    env = _make_env()

    with pytest.raises(ValueError, match="must be finite"):
        apply_reward_variants(env, variants=[f"objective_mine:{factor}"])


@pytest.mark.parametrize("factor", ["0", "-1"])
def test_objective_mine_rejects_non_positive_compounding_factor(factor: str) -> None:
    env = _make_env()

    with pytest.raises(ValueError, match="must be > 0"):
        apply_reward_variants(env, variants=[f"objective_mine:{factor}"])


def test_objective_mine_accepts_finite_positive_compounding_factor() -> None:
    env = _make_env()

    apply_reward_variants(env, variants=["objective_mine:1.5"])

    assert env.label.endswith(".objective_mine")


def test_objective_mine_wires_role_shaping_without_caps() -> None:
    env = _make_env()

    apply_reward_variants(env, variants=["objective_mine"])

    rewards = env.game.agents[0].rewards
    gained_reward = rewards["objective_mine_elements_gained"].reward
    assert isinstance(gained_reward, SumGameValue)
    assert gained_reward.log is True

    deposited_reward = rewards["objective_mine_elements_deposited"].reward
    assert isinstance(deposited_reward, SumGameValue)
    assert deposited_reward.log is True
    assert all(isinstance(value, StatValue) and value.scope == Scope.GAME for value in deposited_reward.values)

    aligned_reward = rewards["objective_mine_junction_aligned_by_agent"].reward
    assert isinstance(aligned_reward, SumGameValue)
    assert aligned_reward.weights == pytest.approx([0.3])
