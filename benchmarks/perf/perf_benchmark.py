#!/usr/bin/env -S uv run python
"""CogsGuard env-only performance benchmark.

Uses the shared mettagrid perf harness with the CogsGuard mission config.
For toy/arena benchmarks, use packages/mettagrid/benchmarks/perf/perf_benchmark.py.
"""

import argparse
import json
import os
import sys

from cogames.cogs_vs_clips.mission import CvCMission
from cogames.cogs_vs_clips.sites import make_cogsguard_machina1_site
from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
from mettagrid.perf.harness import (
    compare_multiple,
    print_comparison,
    print_scorecard_reminder,
    run_performance,
    save_results,
)
from mettagrid.simulator import Simulator

DEFAULT_NUM_AGENTS = 8
DEFAULT_MAX_STEPS = 10000


def create_cogsguard_env(num_agents: int = DEFAULT_NUM_AGENTS, max_steps: int = DEFAULT_MAX_STEPS):
    site = make_cogsguard_machina1_site(num_agents)
    mission = CvCMission(
        name="basic",
        description="CogsGuard env-only benchmark (machina_1 layout)",
        site=site,
        num_cogs=num_agents,
        max_steps=max_steps,
    )
    cfg = mission.make_env()
    simulator = Simulator()
    env = MettaGridPufferEnv(simulator, cfg)
    env.reset()
    return env


def main():
    parser = argparse.ArgumentParser(
        description="CogsGuard env-only performance benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python perf_benchmark.py --profile
  python perf_benchmark.py --phase baseline --output /tmp/cogsguard_baseline.json
  python perf_benchmark.py --phase branch --output /tmp/cogsguard_branch.json \\
      --baseline /tmp/cogsguard_baseline.json
        """,
    )
    parser.add_argument("--agents", type=int, default=DEFAULT_NUM_AGENTS, help="Number of agents")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Max steps per episode")
    parser.add_argument("--iterations", type=int, default=20000, help="Steps per round")
    parser.add_argument("--rounds", type=int, default=20, help="Number of measurement rounds")
    parser.add_argument("--warmup", type=int, default=150000, help="Warm-up steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--baseline", type=str, nargs="+", help="Compare against baseline JSON file(s)")
    parser.add_argument("--phase", type=str, default="", help="Label for this optimization phase")
    parser.add_argument("--profile", action="store_true", help="Print per-phase step timing breakdown")
    args = parser.parse_args()

    if args.profile:
        os.environ["METTAGRID_PROFILING"] = "1"

    print(f"Creating CogsGuard environment: {args.agents} agents, machina_1 layout")
    env = create_cogsguard_env(num_agents=args.agents, max_steps=args.max_steps)
    if args.phase:
        print(f"Phase: {args.phase}")

    stats = run_performance(
        env, iterations=args.iterations, rounds=args.rounds, warmup=args.warmup, profile=args.profile
    )

    if args.output:
        config = {
            "agents": args.agents,
            "max_steps": args.max_steps,
            "layout": "machina_1",
            "iterations": args.iterations,
            "rounds": args.rounds,
            "warmup": args.warmup,
            "seed": args.seed,
        }
        save_results(stats, config, args.phase, args.output)

    if args.baseline:
        print(f"\n{'=' * 60}")
        print("Comparisons")
        print(f"{'=' * 60}")
        comparisons = compare_multiple(args.baseline, stats, args.phase or "current")
        for comparison in comparisons:
            print_comparison(comparison)

        if args.output:
            with open(args.output) as f:
                result = json.load(f)
            result["comparisons"] = comparisons
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)

    print_scorecard_reminder(
        stats,
        config_label=f"env-only (cogsguard, {args.agents}a, machina_1)",
        runs_label=f"{args.rounds}x{args.iterations // 1000}K steps",
        num_rounds=args.rounds,
        phase=args.phase,
        baseline_paths=args.baseline,
        output_path=args.output,
    )

    if stats["cv"] > 0.20:
        print("\nPerformance measurement unstable!")
        sys.exit(1)

    if stats.get("validation_mismatches", 0) > 0:
        print("\nObservations validation FAILED!")
        sys.exit(2)


if __name__ == "__main__":
    main()
