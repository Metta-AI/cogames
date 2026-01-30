"""Game playing functionality for CoGames."""

import logging
import uuid
from pathlib import Path
from typing import Optional

from rich import box
from rich.console import Console
from rich.table import Table

from metta_alo.pure_single_episode_runner import PureSingleEpisodeResult
from metta_alo.rollout import run_single_episode
from mettagrid import MettaGridConfig
from mettagrid.policy.policy import PolicySpec
from mettagrid.renderer.renderer import RenderMode

logger = logging.getLogger("cogames.play")

# Resources and gear types for CogsGuard
ELEMENTS = ["carbon", "oxygen", "germanium", "silicon"]
GEAR = ["miner", "aligner", "scrambler", "scout"]


def _print_episode_stats(console: Console, results: PureSingleEpisodeResult) -> None:
    """Print episode statistics in a formatted table."""
    stats = results.stats
    total_reward = sum(results.rewards)

    # Aggregate agent stats
    agent_stats = stats.get("agent", [])
    totals: dict[str, float] = {}
    for agent in agent_stats:
        for key, value in agent.items():
            totals[key] = totals.get(key, 0) + value

    # Check if this is a CogsGuard mission (has collective stats)
    collective_stats = stats.get("collective", {})
    cogs_stats = collective_stats.get("cogs", {})
    clips_stats = collective_stats.get("clips", {})

    if cogs_stats or clips_stats:
        # CogsGuard mission - show relevant stats
        _print_cogsguard_stats(console, totals, cogs_stats, clips_stats, total_reward)
    else:
        # Standard mission - show basic stats
        _print_standard_stats(console, totals, total_reward)


def _print_cogsguard_stats(
    console: Console,
    agent_totals: dict[str, float],
    cogs_stats: dict[str, float],
    clips_stats: dict[str, float],
    total_reward: float,
) -> None:
    """Print CogsGuard-specific statistics."""
    table = Table(title="Episode Stats", box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Category", style="yellow")
    table.add_column("Stat", style="white")
    table.add_column("Gained", style="green", justify="right")
    table.add_column("Lost", style="red", justify="right")
    table.add_column("Final", style="cyan", justify="right")

    sections_added = 0

    # Junctions: gained (aligned) | lost | final (current count)
    junctions_added = False
    for team, team_stats, color in [("cogs", cogs_stats, "green"), ("clips", clips_stats, "red")]:
        gained = int(team_stats.get("junction.gained", 0))
        lost = int(team_stats.get("junction.lost", 0))
        final = int(team_stats.get("junction", 0))
        if gained > 0 or lost > 0 or final > 0:
            if not junctions_added:
                table.add_row("[bold]Junctions[/bold]", "", "", "", "")
                junctions_added = True
                sections_added += 1
            table.add_row(
                "",
                f"[{color}]{team}[/{color}]",
                f"[{color}]{gained}[/{color}]",
                f"[{color}]{lost}[/{color}]",
                f"[{color}]{final}[/{color}]",
            )

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

    # Resources: gained (deposited) | lost (withdrawn) | final (current amount)
    resources_added = False
    for resource in ELEMENTS:
        gained = int(cogs_stats.get(f"collective.{resource}.deposited", 0))
        lost = int(cogs_stats.get(f"collective.{resource}.withdrawn", 0))
        final = int(cogs_stats.get(f"collective.{resource}.amount", 0))
        if gained > 0 or lost > 0 or final > 0:
            if not resources_added:
                if sections_added > 0:
                    table.add_section()
                table.add_row("[bold]Resources[/bold]", "", "", "", "")
                resources_added = True
                sections_added += 1
            table.add_row("", resource, str(gained), str(lost), str(final))

    # Total reward at bottom
    if sections_added > 0:
        table.add_section()
    table.add_row("[bold]Reward[/bold]", "total", f"{total_reward:.2f}", "", "")

    console.print(table)


def _print_standard_stats(console: Console, agent_totals: dict[str, float], total_reward: float) -> None:
    """Print standard statistics for non-CogsGuard missions."""
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

    # Total reward at bottom
    if interesting:
        table.add_section()
    table.add_row("[bold]Reward (total)[/bold]", f"{total_reward:.2f}")

    console.print(table)


def play(
    console: Console,
    env_cfg: "MettaGridConfig",
    policy_spec: PolicySpec,
    game_name: str,
    seed: int = 42,
    render_mode: RenderMode = "gui",
    save_replay: Optional[Path] = None,
) -> None:
    """Play a single game episode with a policy.

    Args:
        console: Rich console for output
        env_cfg: Game configuration
        policy_spec: Policy specification (class path and optional data path)
        game_name: Human-readable name of the game (used for logging/metadata)
        seed: Random seed
        render_mode: Render mode - "gui", "vibescope", "unicode", or "none"
        save_replay: Optional directory path to save replay. Directory will be created if it doesn't exist.
            Replay will be saved with a unique UUID-based filename.
    """

    logger.debug("Starting play session", extra={"game_name": game_name})

    replay_path = None
    if save_replay:
        save_replay.mkdir(parents=True, exist_ok=True)
        replay_path = save_replay / f"{uuid.uuid4()}.json.z"

    try:
        results, _replay = run_single_episode(
            policy_specs=[policy_spec],
            assignments=[0] * env_cfg.game.num_agents,
            env=env_cfg,
            results_uri=None,
            replay_uri=str(replay_path) if replay_path else None,
            seed=seed,
            device="cpu",
            render_mode=render_mode,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted; ending episode early.")
        return

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
