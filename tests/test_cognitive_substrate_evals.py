from __future__ import annotations

import pytest
from typer.testing import CliRunner

import cogames.diagnose as diagnose_module
from cogames.games.cogs_vs_clips.evals.cognitive_substrate import CATEGORY_MISSIONS, EVAL_MISSIONS
from cogames.games.cogs_vs_clips.evals.cognitive_substrate.benchmark import (
    default_benchmark_policy_specs,
    run_local_benchmark,
    scripted_gap_closed,
)
from cogames.main import app
from mettagrid.map_builder.map_builder import HasSeed
from mettagrid.policy.policy import PolicySpec
from mettagrid.runner.rollout import resolve_env_for_seed, run_episode_local

runner = CliRunner()

_SCRIPTED_POLICY_CLASS_PATH = {
    "memory": "cogames.policy.cognitive_substrate_diagnostics.SubstrateMemoryPolicy",
    "exploration": "cogames.policy.cognitive_substrate_diagnostics.SubstrateExplorationPolicy",
    "planning": "cogames.policy.cognitive_substrate_diagnostics.SubstratePlanningPolicy",
}


def _category_for_mission_name(mission_name: str) -> str:
    return mission_name.split("_", 1)[0]


@pytest.mark.parametrize("mission", EVAL_MISSIONS, ids=lambda mission: mission.name)
def test_cognitive_substrate_missions_build_minimal_single_agent_envs(mission) -> None:
    env = mission.make_env()

    assert env.game.num_agents == 1
    assert [action.name for action in env.game.actions.actions()] == [
        "noop",
        "move_north",
        "move_south",
        "move_west",
        "move_east",
    ]
    assert env.game.obs.num_tokens == 64


@pytest.mark.parametrize("mission", EVAL_MISSIONS, ids=lambda mission: mission.name)
def test_cognitive_substrate_missions_defer_map_seed_to_rollout_seed(mission) -> None:
    env = mission.make_env()
    first = resolve_env_for_seed(env, seed=3)
    second = resolve_env_for_seed(env, seed=11)

    assert isinstance(env.game.map_builder, HasSeed)
    assert isinstance(first.game.map_builder, HasSeed)
    assert isinstance(second.game.map_builder, HasSeed)
    assert env.game.map_builder.seed is None
    assert first is not env
    assert second is not env
    assert first.game.map_builder.seed == 3
    assert second.game.map_builder.seed == 11


def test_cognitive_substrate_diagnose_sources_load_expected_categories() -> None:
    aggregate = diagnose_module._load_diagnose_missions("cognitive_substrate_evals")
    assert {mission.name for mission in aggregate} == {mission.name for mission in EVAL_MISSIONS}

    for category, expected_missions in CATEGORY_MISSIONS.items():
        loaded = diagnose_module._load_diagnose_missions(f"cognitive_substrate_{category}")
        assert {mission.name for mission in loaded} == {mission.name for mission in expected_missions}


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("mission", EVAL_MISSIONS, ids=lambda mission: mission.name)
def test_cognitive_substrate_scripted_policies_solve_each_tier(mission, seed: int) -> None:
    policy_class_path = _SCRIPTED_POLICY_CLASS_PATH[_category_for_mission_name(mission.name)]
    results, _ = run_episode_local(
        policy_specs=[PolicySpec(class_path=policy_class_path)],
        assignments=[0],
        env=mission.make_env(),
        seed=seed,
        render_mode="none",
    )

    assert results.rewards == [1.0]
    assert results.stats["agent"][0].get("goal.reached", 0.0) == 1.0


@pytest.mark.parametrize(
    ("mission_name", "policy_name"),
    [
        ("evals.memory_mystery_path_easy", "substrate_memory"),
        ("evals.exploration_sparse_search_easy", "substrate_exploration"),
        ("evals.planning_unlock_chain_easy", "substrate_planning"),
    ],
)
@pytest.mark.timeout(60)
def test_cognitive_substrate_eval_missions_are_playable_via_cli(mission_name: str, policy_name: str) -> None:
    result = runner.invoke(
        app,
        [
            "play",
            "-m",
            mission_name,
            "--policy",
            policy_name,
            "--steps",
            "10",
            "--render",
            "none",
        ],
    )

    assert result.exit_code == 0, result.output


def test_cognitive_substrate_local_benchmark_runner_reports_required_metrics() -> None:
    rows = run_local_benchmark(
        category="memory",
        mission_names=["memory_mystery_path_easy"],
        policy_specs=default_benchmark_policy_specs(
            "memory",
            default_policy="substrate_memory",
            cognitive_substrate="substrate_memory",
            no_intrinsic="substrate_memory",
            no_successor="substrate_memory",
            inner_steps_1="substrate_memory",
        ),
    )

    assert {row["comparator"] for row in rows} == {
        "scripted",
        "default_policy",
        "cognitive_substrate",
        "no_intrinsic",
        "no_successor",
        "inner_steps_1",
    }
    assert all("scripted_gap_closed" in row for row in rows)


def test_local_benchmark_runner_requires_scripted_and_default_policy_comparators() -> None:
    with pytest.raises(
        ValueError,
        match="policy_specs must include benchmark comparators: scripted, default_policy",
    ):
        run_local_benchmark(
            category="memory",
            mission_names=["memory_mystery_path_easy"],
            policy_specs={"cognitive_substrate": "substrate_memory"},
        )


def test_scripted_gap_closed_clips_to_unit_interval() -> None:
    assert scripted_gap_closed(0.0, default_policy_score=1.0, scripted_score=2.0) == 0.0
    assert scripted_gap_closed(2.0, default_policy_score=0.0, scripted_score=1.0) == 1.0
