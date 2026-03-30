"""
Frame capture script for the Autoresearch Director.

Runs an episode and saves emoji map snapshots at regular intervals to a text file.
No TTY, no interaction required — designed for autonomous director use.

Usage:
    python scripts/capture_frames.py [options]

Example:
    python scripts/capture_frames.py --steps 500 --every 50 --out docs/autoresearch_director/frames.txt
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mettagrid.renderer.miniscope.buffer import MapBuffer
from mettagrid.renderer.miniscope.symbol import DEFAULT_SYMBOL_MAP
from mettagrid.renderer.renderer import Renderer
from mettagrid.runner.rollout import run_episode_local
from mettagrid.simulator.interface import SimulatorEventHandler
from typing_extensions import override


class FrameCaptureRenderer(Renderer):
    """Non-interactive renderer that captures emoji map frames to a file."""

    def __init__(self, output_path: Path, capture_every: int = 100):
        super().__init__()
        self._output_path = output_path
        self._capture_every = capture_every
        self._map_buffer: Optional[MapBuffer] = None
        self._frames: list[tuple[int, str, str]] = []  # (step, reward_str, grid)

    @override
    def on_episode_start(self) -> None:
        symbol_map = DEFAULT_SYMBOL_MAP.copy()
        for obj in self._sim.config.game.objects.values():
            symbol_map[obj.render_name or obj.name] = obj.render_symbol
            if obj.render_name and obj.render_name != obj.name:
                symbol_map[obj.name] = obj.render_symbol
        self._map_buffer = MapBuffer(
            symbol_map=symbol_map,
            initial_height=self._sim.map_height,
            initial_width=self._sim.map_width,
        )
        self._frames = []
        self._capture(0)

    @override
    def on_step(self) -> None:
        step = self._sim.current_step
        if step % self._capture_every == 0:
            self._capture(step)

    @override
    def on_episode_end(self) -> None:
        self._capture(self._sim.current_step)
        self._write()

    def _capture(self, step: int) -> None:
        if self._map_buffer is None:
            return
        grid_objects = self._sim.grid_objects()
        grid = self._map_buffer.render_full_map(grid_objects)
        rewards = self._sim.episode_rewards
        reward_str = " ".join(f"{r:.4f}" for r in rewards) if rewards is not None else "n/a"
        self._frames.append((step, reward_str, grid))

    def _write(self) -> None:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._output_path, "w") as f:
            f.write(f"# Frame Capture — {datetime.now().isoformat()}\n")
            f.write(f"# map={self._sim.map_height}x{self._sim.map_width}  "
                    f"agents={self._sim.num_agents}  frames={len(self._frames)}\n")
            f.write("# Symbols: 🟦=agent0 🟧=agent1 🟩=agent2 🟨=agent3  "
                    "⬜=wall  · =empty\n\n")
            for step, reward_str, grid in self._frames:
                f.write(f"{'='*60}\n")
                f.write(f"Step {step:4d}  rewards=[{reward_str}]\n")
                f.write(f"{'='*60}\n")
                f.write(grid)
                f.write("\n\n")
        print(f"Saved {len(self._frames)} frames → {self._output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", default="cogsguard_machina_1.basic")
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--aligners", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="docs/autoresearch_director/frames.txt")
    parser.add_argument("--policy-class", default="machina_llm_roles",
                        help="Policy class name")
    parser.add_argument("--policy-kw", nargs="*", default=[],
                        help="Extra policy kwargs, e.g. llm_timeout_s=20 scripted_miners=true")
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.out)
    capturer = FrameCaptureRenderer(output_path=output_path, capture_every=args.every)

    from cogames.cli.policy import parse_policy_spec
    extra_kw = ",".join(f"kw.{kv}" for kv in args.policy_kw)
    spec_str = f"class={args.policy_class},kw.num_aligners={args.aligners}"
    if extra_kw:
        spec_str += "," + extra_kw
    policy_spec_with_prop = parse_policy_spec(spec_str)
    policy_spec = policy_spec_with_prop.to_policy_spec()

    from cogames.cli.mission import get_mission
    _name, env_cfg, _mission = get_mission(args.mission, cogs=args.agents)
    env_cfg.game.max_steps = args.steps

    from mettagrid.runner.rollout import resolve_env_for_seed, Rollout
    from mettagrid.policy.loader import initialize_or_load_policy
    from mettagrid.policy.policy_env_interface import PolicyEnvInterface
    from mettagrid.simulator.time_averaged_stats import TimeAveragedStatsHandler

    env_for_rollout = resolve_env_for_seed(env_cfg, args.seed)
    env_interface = PolicyEnvInterface.from_mg_cfg(env_for_rollout)
    multi_policy = initialize_or_load_policy(env_interface, policy_spec)

    num_agents = env_for_rollout.game.num_agents
    agent_policies = [multi_policy.agent_policy(i) for i in range(num_agents)]

    stats_handler = TimeAveragedStatsHandler()

    rollout = Rollout(
        env_for_rollout,
        agent_policies,
        render_mode="none",
        seed=args.seed,
        event_handlers=[stats_handler, capturer],
        autostart=True,
    )
    rollout.run_until_done()

    rewards = list(rollout._sim.episode_rewards)
    print(f"\nEpisode done. Steps: {rollout._sim.current_step}  "
          f"Avg reward/agent: {sum(rewards)/len(rewards):.4f}")


if __name__ == "__main__":
    main()
