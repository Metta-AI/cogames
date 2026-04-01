"""Game playing functionality for CoGames."""

import logging
import uuid
from pathlib import Path
from typing import Optional

from rich import box
from rich.console import Console
from rich.table import Table

from cogames.cli.policy import PolicySpecWithProportion
from mettagrid import MettaGridConfig
from mettagrid.renderer.renderer import RenderMode
from mettagrid.runner.rollout import run_episode_local
from mettagrid.runner.types import PureSingleEpisodeResult

logger = logging.getLogger("cogames.play")

# Resources and gear types for CvC
ELEMENTS = ["carbon", "oxygen", "germanium", "silicon"]
GEAR = ["miner", "aligner", "scrambler", "scout"]


def _print_episode_stats(console: Console, results: PureSingleEpisodeResult) -> None:
    """Print episode statistics in a formatted table."""
    stats = results.stats
    total_reward = sum(results.rewards)
    num_agents = len(results.rewards)
    avg_reward_per_agent = total_reward / num_agents if num_agents > 0 else 0.0

    # Aggregate agent stats
    agent_stats = stats.get("agent", [])
    totals: dict[str, float] = {}
    for agent in agent_stats:
        for key, value in agent.items():
            totals[key] = totals.get(key, 0) + value

    has_gear_stats = any(f"{gear}.gained" in totals for gear in GEAR)
    if has_gear_stats:
        _print_cvc_stats(console, totals, avg_reward_per_agent)
    else:
        # Standard mission - show basic stats
        _print_standard_stats(console, totals, avg_reward_per_agent)


def _print_cvc_stats(
    console: Console,
    agent_totals: dict[str, float],
    avg_reward_per_agent: float,
) -> None:
    """Print CvC-specific statistics."""
    table = Table(title="Episode Stats", box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Category", style="yellow")
    table.add_column("Stat", style="white")
    table.add_column("Gained", style="green", justify="right")
    table.add_column("Lost", style="red", justify="right")
    table.add_column("Final", style="cyan", justify="right")

    sections_added = 0

    # Gear: gained | lost | final (net = gained - lost)
    gear_added = False
    for gear in GEAR:
        gained = int(agent_totals.get(f"{gear}.gained", 0))
        lost = int(agent_totals.get(f"{gear}.lost", 0))
        final = gained - lost
        if gained > 0 or lost > 0:
            if not gear_added:
                if sections_added > 0:
                    table.add_section()
                table.add_row("[bold]Gear[/bold]", "", "", "", "")
                gear_added = True
                sections_added += 1
            table.add_row("", gear, str(gained), str(lost), str(final))

    # Hearts (in Gear section)
    hearts_gained = int(agent_totals.get("heart.gained", 0))
    hearts_lost = int(agent_totals.get("heart.lost", 0))
    if hearts_gained > 0 or hearts_lost > 0:
        if not gear_added:
            if sections_added > 0:
                table.add_section()
            table.add_row("[bold]Gear[/bold]", "", "", "", "")
            gear_added = True
            sections_added += 1
        table.add_row("", "hearts", str(hearts_gained), str(hearts_lost), "")

    # Score at bottom
    if sections_added > 0:
        table.add_section()
    table.add_row("[bold]Score[/bold]", "per cog", "", "", f"{avg_reward_per_agent:.2f}")

    console.print(table)


def _print_standard_stats(console: Console, agent_totals: dict[str, float], avg_reward_per_agent: float) -> None:
    """Print standard statistics for non-CvC missions."""
    # Filter for interesting stats
    interesting = {}
    for key, value in agent_totals.items():
        if value != 0 and any(pattern in key for pattern in [".gained", ".lost", ".deposited", ".withdrawn", "heart"]):
            interesting[key] = value

    table = Table(title="Episode Stats", box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Stat", style="white")
    table.add_column("Value", style="green", justify="right")

    for key in sorted(interesting.keys()):
        table.add_row(key, f"{int(interesting[key])}")

    # Score at bottom
    if interesting:
        table.add_section()
    table.add_row("[bold]Score (per cog)[/bold]", f"{avg_reward_per_agent:.2f}")

    console.print(table)


def _build_team_assignments(env_cfg: "MettaGridConfig", specs: list[PolicySpecWithProportion]) -> list[int]:
    """Map each agent to a policy index using per-team cycling.

    Builds a repeating pattern from policy counts (the proportion field) and
    applies it independently within each team.  For example, specs with
    proportions [1, 2] produce pattern [0, 1, 1] which cycles per team:
    a team of 4 agents gets [0, 1, 1, 0].
    """
    if len(specs) == 1:
        return [0] * env_cfg.game.num_agents

    pattern: list[int] = []
    for idx, spec in enumerate(specs):
        pattern.extend([idx] * int(spec.proportion))

    # Group agent indices by team_id, preserving first-appearance order.
    teams: dict[int, list[int]] = {}
    for agent_idx, agent in enumerate(env_cfg.game.agents):
        teams.setdefault(agent.team_id, []).append(agent_idx)

    assignments = [0] * env_cfg.game.num_agents
    for team_agents in teams.values():
        for i, agent_idx in enumerate(team_agents):
            assignments[agent_idx] = pattern[i % len(pattern)]

    return assignments


def play(
    console: Console,
    env_cfg: "MettaGridConfig",
    policy_specs: list[PolicySpecWithProportion],
    game_name: str,
    seed: int = 42,
    device: str = "cpu",
    render_mode: RenderMode = "gui",
    action_timeout_ms: int = 10000,
    save_replay: Optional[Path] = None,
    save_replay_file: Optional[Path] = None,
    autostart: bool = False,
) -> None:
    """Play a single game episode with one or more policies.

    Args:
        console: Rich console for output
        env_cfg: Game configuration
        policy_specs: Policy specifications. One spec applies to all agents;
            multiple specs assign one policy per team.
        game_name: Human-readable name of the game (used for logging/metadata)
        seed: Random seed
        render_mode: Render mode - "gui", "unicode", or "none"
        save_replay: Optional directory path to save replay. Directory will be created if it doesn't exist.
            Replay will be saved with a unique UUID-based filename.
        save_replay_file: Optional file path to save replay to. Parent directory will be created
            if needed, and existing file is overwritten.
    """

    logger.debug("Starting play session", extra={"game_name": game_name})

    replay_path = None
    if save_replay_file:
        save_replay_file.parent.mkdir(parents=True, exist_ok=True)
        replay_path = save_replay_file
    elif save_replay:
        save_replay.mkdir(parents=True, exist_ok=True)
        replay_path = save_replay / f"{uuid.uuid4()}.json.z"

    assignments = _build_team_assignments(env_cfg, policy_specs)

    results, _replay = run_episode_local(
        policy_specs=[s.to_policy_spec() for s in policy_specs],
        assignments=assignments,
        env=env_cfg,
        replay_path=replay_path,
        seed=seed,
        max_action_time_ms=action_timeout_ms,
        device=device,
        render_mode=render_mode,
        autostart=autostart,
    )

    # Print summary
    console.print("\n[bold green]Episode Complete![/bold green]")
    console.print(f"Steps: {results.steps}")

    # Print episode stats
    _print_episode_stats(console, results)

    # Print replay command if replay was saved
    if replay_path:
        console.print("\n[bold cyan]Replay saved![/bold cyan]")
        console.print("To watch the replay, run:")
        console.print(f"[bold green]cogames replay {replay_path}[/bold green]")
