from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Literal

from cogames.cli.policy import parse_policy_spec
from cogames.games.cogs_vs_clips.evals.cognitive_substrate import CATEGORY_MISSIONS, EVAL_MISSIONS
from mettagrid.runner.rollout import run_episode_local
from mettagrid.runner.types import PureSingleEpisodeResult

CategoryName = Literal["memory", "exploration", "planning"]

REQUIRED_LEARNED_COMPARATORS = (
    "default_policy",
    "cognitive_substrate",
    "no_intrinsic",
    "no_successor",
    "inner_steps_1",
)

SCRIPTED_POLICY_SPECS: dict[CategoryName, str] = {
    "memory": "substrate_memory",
    "exploration": "substrate_exploration",
    "planning": "substrate_planning",
}
REQUIRED_BENCHMARK_COMPARATORS = ("scripted", "default_policy")

MISSION_BY_NAME = {mission.name: mission for mission in EVAL_MISSIONS}


def scripted_gap_closed(score: float, default_policy_score: float, scripted_score: float) -> float:
    denominator = scripted_score - default_policy_score
    if denominator <= 0:
        return 1.0 if score >= scripted_score else 0.0
    return max(0.0, min(1.0, (score - default_policy_score) / denominator))


def scripted_policy_spec(category: CategoryName) -> str:
    return SCRIPTED_POLICY_SPECS[category]


def default_benchmark_policy_specs(
    category: CategoryName,
    *,
    default_policy: str,
    cognitive_substrate: str,
    no_intrinsic: str,
    no_successor: str,
    inner_steps_1: str,
) -> dict[str, str]:
    return {
        "scripted": scripted_policy_spec(category),
        "default_policy": default_policy,
        "cognitive_substrate": cognitive_substrate,
        "no_intrinsic": no_intrinsic,
        "no_successor": no_successor,
        "inner_steps_1": inner_steps_1,
    }


def _category_missions(category: CategoryName, mission_names: Sequence[str] | None) -> list:
    if mission_names is None:
        return list(CATEGORY_MISSIONS[category])
    return [MISSION_BY_NAME[mission_name] for mission_name in mission_names]


def _result_metrics(result: PureSingleEpisodeResult, *, max_steps: int) -> dict[str, float]:
    agent_stats = result.stats["agent"][0]
    success = float(agent_stats.get("goal.reached", 0.0) > 0)
    return {
        "reward": float(result.rewards[0]),
        "success": success,
        "steps": float(result.steps),
        "steps_to_goal": float(agent_stats.get("goal.steps_to_goal", result.steps if success else max_steps)),
        "timeout": float(success == 0.0 and result.steps >= max_steps),
        "cell_visited": float(agent_stats.get("cell.visited", 0.0)),
    }


def _validate_benchmark_inputs(policy_specs: Mapping[str, str], seeds: Sequence[int]) -> None:
    missing = [name for name in REQUIRED_BENCHMARK_COMPARATORS if name not in policy_specs]
    if missing:
        missing_names = ", ".join(missing)
        raise ValueError(f"policy_specs must include benchmark comparators: {missing_names}")
    if not seeds:
        raise ValueError("seeds must be non-empty")


def run_local_benchmark(
    *,
    category: CategoryName,
    policy_specs: Mapping[str, str],
    mission_names: Sequence[str] | None = None,
    seeds: Sequence[int] = (0,),
    device: str = "cpu",
) -> list[dict[str, float | str]]:
    _validate_benchmark_inputs(policy_specs, seeds)
    missions = _category_missions(category, mission_names)
    parsed_specs = {
        name: parse_policy_spec(spec, device=device).to_policy_spec() for name, spec in policy_specs.items()
    }

    by_mission: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for mission in missions:
        env = mission.make_env()
        for comparator_name, policy_spec in parsed_specs.items():
            for seed in seeds:
                result, _ = run_episode_local(
                    policy_specs=[policy_spec],
                    assignments=[0],
                    env=env,
                    seed=seed,
                    render_mode="none",
                    device=device,
                )
                by_mission[mission.name][comparator_name].append(_result_metrics(result, max_steps=mission.max_steps))

    rows: list[dict[str, float | str]] = []
    for mission in missions:
        mission_stats = by_mission[mission.name]
        default_policy_score = mean(item["success"] for item in mission_stats["default_policy"])
        scripted_score = mean(item["success"] for item in mission_stats["scripted"])
        for comparator_name, metrics in mission_stats.items():
            success_rate = mean(item["success"] for item in metrics)
            rows.append(
                {
                    "category": category,
                    "mission": mission.name,
                    "comparator": comparator_name,
                    "success_rate": success_rate,
                    "avg_reward": mean(item["reward"] for item in metrics),
                    "avg_steps": mean(item["steps"] for item in metrics),
                    "avg_steps_to_goal": mean(item["steps_to_goal"] for item in metrics),
                    "timeout_rate": mean(item["timeout"] for item in metrics),
                    "cell_visited": mean(item["cell_visited"] for item in metrics),
                    "scripted_gap_closed": scripted_gap_closed(
                        success_rate,
                        default_policy_score=default_policy_score,
                        scripted_score=scripted_score,
                    ),
                }
            )
    return rows


__all__ = [
    "REQUIRED_LEARNED_COMPARATORS",
    "SCRIPTED_POLICY_SPECS",
    "default_benchmark_policy_specs",
    "run_local_benchmark",
    "scripted_gap_closed",
    "scripted_policy_spec",
]
