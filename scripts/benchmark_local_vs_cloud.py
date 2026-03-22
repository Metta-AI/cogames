#!/usr/bin/env python3
"""Benchmark local Nemotron LLM vs cloud API for CoGames miner policy.

Runs two back-to-back episodes of the miner_tutorial mission (100 steps,
8 robots) and compares:

  1. **Local LLM** – nvidia/NVIDIA-Nemotron-Nano-9B-v2 running on-device
     (uses LOCAL_LLM_MODEL_PATH; falls back to an instant mock when unset).
  2. **Cloud API**  – mock responder that sleeps for a realistic cloud round-trip
     (~800 ms/call), approximating OpenRouter latency.

Usage:
    # Full run with real local model:
    export LOCAL_LLM_MODEL_PATH=/workspace/models/nemotron-nano-9b-v2
    python scripts/benchmark_local_vs_cloud.py

    # Dry-run (both modes mocked, no GPU needed):
    python scripts/benchmark_local_vs_cloud.py --mock-local

Options:
    --steps N            Game steps per episode (default: 100)
    --agents N           Robots per game (default: 8)
    --cloud-latency MS   Simulated cloud latency in ms (default: 800)
    --mock-local         Use instant mock instead of loading real model
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ── logging ────────────────────────────────────────────────────────────────────

_fmt = "%(asctime)s %(levelname)-7s %(name)s │ %(message)s"
logging.basicConfig(level=logging.WARNING, format=_fmt, datefmt="%H:%M:%S")
log = logging.getLogger("benchmark")
log.setLevel(logging.INFO)

# ── args ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Benchmark local vs cloud LLM in CoGames")
parser.add_argument("--steps", type=int, default=100, help="Max game steps per episode (default: 100)")
parser.add_argument("--agents", type=int, default=None, help="Override number of robots (default: use mission native)")
parser.add_argument("--mission", default="miner_tutorial", help="Mission name (default: miner_tutorial)")
parser.add_argument("--cloud-latency", type=float, default=800.0, help="Simulated cloud latency ms (default: 800)")
parser.add_argument("--mock-local", action="store_true", help="Use instant mock instead of real local model")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# ── imports ────────────────────────────────────────────────────────────────────

try:
    from cogames.cogs_vs_clips.missions import get_core_missions
    from cogames.play import play
    from mettagrid.policy.policy import PolicySpec
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint
except ImportError as exc:
    log.error("Import failed: %s", exc)
    log.error("Run from the cogames venv with cogames installed.")
    sys.exit(1)

# ── mission setup ──────────────────────────────────────────────────────────────

_all_missions = get_core_missions()
# Pick the first mission matching the requested name
_mission = next((m for m in _all_missions if m.name == args.mission), None)
if _mission is None:
    available = sorted({m.name for m in _all_missions})
    log.error("Mission %r not found. Available: %s", args.mission, available)
    sys.exit(1)

_base_env = _mission.make_env()
_game_overrides: dict = {"max_steps": args.steps}
if args.agents is not None:
    _game_overrides["num_agents"] = args.agents
env_cfg = _base_env.model_copy(
    update={"game": _base_env.game.model_copy(update=_game_overrides)}
)

log.info("Mission  : %s", _mission.description)
log.info("Agents   : %d", env_cfg.game.num_agents)
log.info("Steps    : %d", env_cfg.game.max_steps)

# ── timing collector ───────────────────────────────────────────────────────────


@dataclass
class BenchStats:
    label: str
    llm_latencies_ms: list[float] = field(default_factory=list)
    total_wall_s: float = 0.0

    @property
    def llm_calls(self) -> int:
        return len(self.llm_latencies_ms)

    @property
    def total_llm_ms(self) -> float:
        return sum(self.llm_latencies_ms)

    @property
    def avg_llm_ms(self) -> float:
        return self.total_llm_ms / self.llm_calls if self.llm_calls else 0.0

    @property
    def p95_llm_ms(self) -> float:
        if not self.llm_latencies_ms:
            return 0.0
        s = sorted(self.llm_latencies_ms)
        idx = max(0, int(len(s) * 0.95) - 1)
        return s[idx]

    @property
    def game_sim_ms(self) -> float:
        return self.total_wall_s * 1000.0 - self.total_llm_ms

    @property
    def steps_per_sec(self) -> float:
        return args.steps / self.total_wall_s if self.total_wall_s > 0 else 0.0


# ── responder factories ────────────────────────────────────────────────────────

_SKILL_CYCLE = ["explore", "gear_up", "mine_until_full", "deposit_to_hub", "mine_until_full", "deposit_to_hub"]


def _make_mock_instant_responder(stats: BenchStats):
    """Instant mock — simulates already-loaded local LLM with negligible latency."""
    counter = {"n": 0}

    def responder(prompt: str) -> str:
        t0 = time.perf_counter()
        skill = _SKILL_CYCLE[counter["n"] % len(_SKILL_CYCLE)]
        counter["n"] += 1
        result = json.dumps({"skill": skill, "reason": f"mock #{counter['n']}"})
        stats.llm_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        return result

    return responder


def _make_cloud_mock_responder(stats: BenchStats, latency_ms: float):
    """Mock that sleeps to simulate cloud API round-trip latency."""
    counter = {"n": 0}
    latency_s = latency_ms / 1000.0

    def responder(prompt: str) -> str:
        t0 = time.perf_counter()
        time.sleep(latency_s)
        skill = _SKILL_CYCLE[counter["n"] % len(_SKILL_CYCLE)]
        counter["n"] += 1
        result = json.dumps({"skill": skill, "reason": f"cloud #{counter['n']}"})
        stats.llm_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        return result

    return responder


def _make_local_llm_responder(stats: BenchStats, model_path: str):
    """Real local LLM responder using Nemotron — loads model on first call."""
    try:
        from cogames.policy.local_llm import LocalLLMInference
    except ImportError as exc:
        log.error("local_llm import failed: %s", exc)
        sys.exit(1)

    inference = LocalLLMInference(model_path)
    log.info("Local model path : %s", model_path)

    def responder(prompt: str) -> str:
        t0 = time.perf_counter()
        result = inference.complete(prompt)
        stats.llm_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        return result

    return responder


# ── run one episode ────────────────────────────────────────────────────────────


def run_episode(stats: BenchStats, responder) -> BenchStats:
    console = Console(no_color=True, quiet=True)
    policy_spec = PolicySpec(
        class_path="llm_miner",
        init_kwargs={"llm_model": None, "llm_api_url": None, "llm_responder": responder},
    )
    log.info("─" * 60)
    log.info("Running: %s", stats.label)
    t0 = time.perf_counter()
    play(
        console=console,
        env_cfg=env_cfg,
        policy_spec=policy_spec,
        game_name=args.mission,
        seed=args.seed,
        device="cpu",
        render_mode="none",
    )
    stats.total_wall_s = time.perf_counter() - t0
    log.info("Done: %s  wall=%.1fs  llm_calls=%d  avg_llm=%.1fms",
             stats.label, stats.total_wall_s, stats.llm_calls, stats.avg_llm_ms)
    return stats


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    results: list[BenchStats] = []

    # ── 1. Local LLM ──────────────────────────────────────────────────────────
    local_path = os.environ.get("LOCAL_LLM_MODEL_PATH", "")
    if args.mock_local or not local_path:
        if not args.mock_local and not local_path:
            log.warning("LOCAL_LLM_MODEL_PATH not set — using instant mock for local LLM.")
            log.warning("To run with the real model: export LOCAL_LLM_MODEL_PATH=<path>")
        local_stats = BenchStats(label="Local Nemotron (mock)" if not local_path else "Local Nemotron (mock, --mock-local)")
        responder = _make_mock_instant_responder(local_stats)
        # Warm the mock so the first call isn't cold
        responder("warmup")
        local_stats.llm_latencies_ms.clear()
    else:
        local_stats = BenchStats(label=f"Local Nemotron 9B ({Path(local_path).name})")
        log.info("Loading local Nemotron model (this may take ~30-60s on first run)…")
        # Pre-load by running a warm-up call
        from cogames.policy.local_llm import LocalLLMInference
        _warmup_inf = LocalLLMInference(local_path)
        _warmup_inf.complete("warmup")  # load into GPU memory
        responder = _make_local_llm_responder(local_stats, local_path)

    local_result = run_episode(local_stats, responder)
    results.append(local_result)

    # ── 2. Cloud API (simulated) ───────────────────────────────────────────────
    cloud_stats = BenchStats(label=f"Cloud API (mock {args.cloud_latency:.0f}ms/call)")
    cloud_responder = _make_cloud_mock_responder(cloud_stats, args.cloud_latency)
    cloud_result = run_episode(cloud_stats, cloud_responder)
    results.append(cloud_result)

    # ── results table ─────────────────────────────────────────────────────────
    print()
    n_agents = env_cfg.game.num_agents
    table = Table(title=f"⚡ Local vs Cloud LLM Benchmark  ({args.steps} steps, {n_agents} robots)")
    table.add_column("Metric", style="bold")
    for r in results:
        table.add_column(r.label, justify="right")

    rows = [
        ("Total wall time (s)", lambda r: f"{r.total_wall_s:.2f}"),
        ("Steps / second", lambda r: f"{r.steps_per_sec:.1f}"),
        ("LLM calls", lambda r: str(r.llm_calls)),
        ("Avg LLM latency (ms)", lambda r: f"{r.avg_llm_ms:.1f}"),
        ("p95 LLM latency (ms)", lambda r: f"{r.p95_llm_ms:.1f}"),
        ("Total LLM time (s)", lambda r: f"{r.total_llm_ms / 1000:.2f}"),
        ("Game sim time (s)", lambda r: f"{r.game_sim_ms / 1000:.2f}"),
        ("LLM % of wall time", lambda r: f"{100 * r.total_llm_ms / 1000 / r.total_wall_s:.1f}%"),
    ]
    for label, fmt in rows:
        table.add_row(label, *[fmt(r) for r in results])

    console = Console()
    console.print(table)

    # ── speedup summary ───────────────────────────────────────────────────────
    local_r, cloud_r = results[0], results[1]
    speedup = cloud_r.total_wall_s / local_r.total_wall_s if local_r.total_wall_s > 0 else 0
    latency_ratio = cloud_r.avg_llm_ms / local_r.avg_llm_ms if local_r.avg_llm_ms > 0 else float("inf")
    print()
    rprint(f"[bold green]Speedup (wall time):[/bold green]  {speedup:.1f}×  faster with local LLM")
    rprint(f"[bold green]LLM latency ratio:[/bold green]   {latency_ratio:.1f}×  lower per call with local LLM")
    print()


if __name__ == "__main__":
    main()
