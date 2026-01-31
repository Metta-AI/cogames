#!/usr/bin/env -S uv run
# need this to import and call suppress_noisy_logs first
# ruff: noqa: E402

"""CLI for CoGames - collection of environments for multi-agent cooperative and competitive games."""

from cogames.cli.utils import suppress_noisy_logs

suppress_noisy_logs()

import importlib.metadata
import importlib.util
import json
import logging
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Literal, Optional, TypeVar

import typer
import yaml  # type: ignore[import]
from click.core import ParameterSource
from packaging.version import Version
from rich import box
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

import cogames.policy.starter_agent as starter_agent
import cogames.policy.trainable_policy_template as trainable_policy_template
from cogames import evaluate as evaluate_module
from cogames import game, verbose
from cogames import pickup as pickup_module
from cogames import play as play_module
from cogames import train as train_module
from cogames.cli.base import console
from cogames.cli.client import SeasonInfo, TournamentServerClient, fetch_default_season, fetch_season_info
from cogames.cli.docsync import docsync
from cogames.cli.leaderboard import (
    leaderboard_cmd,
    parse_policy_identifier,
    seasons_cmd,
    submissions_cmd,
)
from cogames.cli.login import DEFAULT_COGAMES_SERVER, perform_login
from cogames.cli.mission import (
    describe_mission,
    get_mission_name_and_config,
    get_mission_names_and_configs,
    list_evals,
    list_missions,
    list_variants,
)
from cogames.cli.policy import (
    _translate_error,
    get_policy_spec,
    get_policy_specs_with_proportions,
    parse_policy_spec,
    policy_arg_example,
    policy_arg_w_proportion_example,
)
from cogames.cli.submit import DEFAULT_SUBMIT_SERVER, results_url_for_season, upload_policy, validate_policy_spec
from cogames.curricula import make_rotation
from cogames.device import resolve_training_device
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.policy.loader import discover_and_register_policies
from mettagrid.policy.policy_registry import get_policy_registry
from mettagrid.renderer.renderer import RenderMode
from mettagrid.simulator import Simulator

# Always add current directory to Python path so optional plugins in the repo are discoverable.
sys.path.insert(0, ".")

try:  # Optional plugin
    from tribal_village_env.cogames import register_cli as register_tribal_cli  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - plugin optional
    register_tribal_cli = None


logger = logging.getLogger("cogames.main")


T = TypeVar("T")


def _resolve_mettascope_script() -> Path:
    spec = importlib.util.find_spec("mettagrid")
    if spec is None or spec.origin is None:
        raise FileNotFoundError("mettagrid package is not available; cannot locate MettaScope.")

    package_dir = Path(spec.origin).resolve().parent
    search_roots = (package_dir, *package_dir.parents)

    for root in search_roots:
        candidate = root / "nim" / "mettascope" / "src" / "mettascope.nim"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"MettaScope sources not found relative to installed mettagrid package (searched from {package_dir})."
    )


def _register_policies() -> None:
    discover_and_register_policies()


app = typer.Typer(
    help="CoGames - Multi-agent cooperative and competitive games",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    callback=_register_policies,
)

tutorial_app = typer.Typer(
    help="Tutorial commands to help you get started with CoGames",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    rich_markup_mode="rich",
)

if register_tribal_cli is not None:
    register_tribal_cli(app)

app.add_typer(docsync.app, name="docsync", hidden=True)


@tutorial_app.command(
    name="play", help="Interactive tutorial - learn to play Cogs vs Clips", rich_help_panel="Tutorial"
)
def tutorial_cmd(
    ctx: typer.Context,
) -> None:
    """Run the CoGames tutorial."""
    # Suppress logs during tutorial to keep instructions visible
    logging.getLogger().setLevel(logging.ERROR)

    console.print(
        Panel.fit(
            "[bold cyan]MISSION BRIEFING: CogsGuard Training Sector[/bold cyan]\n\n"
            "Welcome, Cognitive. This simulation mirrors frontline CogsGuard ops.\n"
            "We will launch the Mettascope visual interface now.\n\n"
            "When you are ready to deploy, press Enter below and then return here to receive instructions.",
            title="Mission Briefing",
            border_style="green",
        )
    )

    Prompt.ask("[dim]Press Enter to launch simulation[/dim]", default="", show_default=False)
    console.print("[dim]Initializing Mettascope...[/dim]")

    # Load tutorial mission (CogsGuard)
    from cogames.cogs_vs_clips.missions import make_cogsguard_mission

    # Create environment config
    env_cfg = make_cogsguard_mission(num_agents=1, max_steps=1000).make_env()

    stop_event = threading.Event()

    def _wait_for_enter(prompt: str) -> bool:
        if stop_event.is_set():
            return False
        try:
            Prompt.ask(prompt, default="", show_default=False)
        except (KeyboardInterrupt, EOFError):
            stop_event.set()
            return False
        return True

    def run_tutorial_steps():
        # Wait a moment for the window to appear
        time.sleep(3)

        tutorial_steps = (
            {
                "title": "Step 1 — Interface & Controls",
                "lines": (
                    "Left Pane (Intel): Shows details for selected objects (Stations, Tiles, Cogs).",
                    "Right Pane (Vibe Deck): Select icons here to change your Cog's broadcast resonance.",
                    "Zoom/Pan: Scroll or pinch to zoom the arena; drag to pan.",
                    "Click various buildings to view their details in the Left Pane.",
                    "Look for the Hub (Hub), Junctions, Gear Stations, and Extractors.",
                    "Click your Cog to assume control.",
                ),
            },
            {
                "title": "Step 2 — Movement & Energy",
                "lines": (
                    "Use WASD or Arrow Keys to move your Cog.",
                    "Every move costs Energy, and aligned hubs/junctions recharge you.",
                    "Watch your battery bar on the Cog or in the HUD.",
                    "If low, rest (skip turn), lean against a wall (walk into it), or",
                    "stand near the Hub or an aligned Junction.",
                ),
            },
            {
                "title": "Step 3 — Gear Up",
                "lines": (
                    "Primary interaction mode is WALKING INTO things.",
                    "Locate a Gear Station and walk into it to equip a role:",
                    "  [yellow]⛏ Miner[/yellow], [yellow]🔭 Scout[/yellow],",
                    "  [yellow]🔗 Aligner[/yellow], [yellow]🌀 Scrambler[/yellow].",
                    "Gear costs are paid from the team commons.",
                ),
            },
            {
                "title": "Step 4 — Resources & Hearts",
                "lines": (
                    "Find an Extractor station to gather elements:",
                    "  [yellow]C[/yellow] (Carbon), [yellow]O[/yellow] (Oxygen),",
                    "  [yellow]G[/yellow] (Germanium), [yellow]S[/yellow] (Silicon).",
                    "Visit the Chest to assemble or withdraw Hearts from the commons.",
                ),
            },
            {
                "title": "Step 5 — Junction Control",
                "lines": (
                    "Junctions (junctions) can be aligned to your team.",
                    "As an Aligner: get Influence (stand near the Hub) + a Heart, then bump a neutral junction.",
                    "As a Scrambler: get a Heart, then bump an enemy-aligned junction to neutralize it.",
                    "Aligned junctions recharge energy for your team.",
                ),
            },
            {
                "title": "Step 6 — Objective Complete",
                "lines": (
                    "[bold green]🎉 Congratulations![/bold green] You have completed the tutorial.",
                    "You've mastered movement, gear, resources, and junction control.",
                    "[bold cyan]You're now ready to tackle the full CogsGuard arena![/bold cyan]",
                ),
            },
        )

        for idx, step in enumerate(tutorial_steps):
            if stop_event.is_set():
                return
            console.print()
            console.print(f"[bold cyan]{step['title']}[/bold cyan]")
            console.print()
            for line in step["lines"]:
                console.print(f"  • {line}")
            console.print()
            if idx < len(tutorial_steps) - 1:
                if not _wait_for_enter("[dim]Press Enter for next step[/dim]"):
                    return

        console.print(
            "[bold green]REFERENCE DOSSIERS[/bold green]\n"
            "- [link=packages/cogames/MISSION.md]MISSION.md[/link]: CogsGuard deployment orders.\n"
            "- [link=packages/cogames/README.md]README.md[/link]: System overview and CLI quick start.\n"
            "- [link=packages/cogames/TECHNICAL_MANUAL.md]TECHNICAL_MANUAL.md[/link]: FACE sensor/command schematics."
        )
        console.print()
        console.print("[dim]Tutorial briefing complete. Good luck, Cognitive.[/dim]")
        console.print("[dim]Close the Mettascope window to exit the tutorial.[/dim]")

    # Start tutorial interaction in a background thread
    tutorial_thread = threading.Thread(target=run_tutorial_steps, daemon=True)
    tutorial_thread.start()

    # Run play (blocks main thread)
    try:
        play_module.play(
            console,
            env_cfg=env_cfg,
            policy_spec=get_policy_spec(ctx, "class=noop"),  # Default to noop, assuming human control
            game_name="tutorial",
            render_mode="gui",
        )
    except KeyboardInterrupt:
        logger.info("Tutorial interrupted; exiting.")
    finally:
        stop_event.set()


@tutorial_app.command(
    name="cogsguard",
    help="Interactive CogsGuard tutorial - learn roles and territory control",
    rich_help_panel="Tutorial",
)
def cogsguard_tutorial_cmd(
    ctx: typer.Context,
) -> None:
    """Run the CogsGuard tutorial."""
    # Suppress logs during tutorial to keep instructions visible
    logging.getLogger().setLevel(logging.ERROR)

    console.print(
        Panel.fit(
            "[bold cyan]MISSION BRIEFING: CogsGuard Training[/bold cyan]\n\n"
            "Welcome, Cognitive. This simulation introduces you to CogsGuard operations.\n"
            "You will learn about specialized gear, resource management, and territory control.\n\n"
            "When you are ready to deploy, press Enter below and then return here to receive instructions.",
            title="CogsGuard Briefing",
            border_style="green",
        )
    )

    Prompt.ask("[dim]Press Enter to launch simulation[/dim]", default="", show_default=False)
    console.print("[dim]Initializing Mettascope...[/dim]")

    # Load CogsGuard tutorial mission
    from cogames.cogs_vs_clips.cogsguard_tutorial import CogsGuardTutorialMission

    # Create environment config
    env_cfg = CogsGuardTutorialMission.make_env()

    stop_event = threading.Event()

    def _wait_for_enter(prompt: str) -> bool:
        if stop_event.is_set():
            return False
        try:
            Prompt.ask(prompt, default="", show_default=False)
        except (KeyboardInterrupt, EOFError):
            stop_event.set()
            return False
        return True

    def run_cogsguard_tutorial_steps():
        # Wait a moment for the window to appear
        time.sleep(3)

        tutorial_steps = (
            {
                "title": "Step 1 — Objective & Scoring",
                "lines": (
                    "CogsGuard is a territory control game. Your team earns points by holding junctions.",
                    "[bold]Reward per tick[/bold] = junctions held / max_steps / num_cogs",
                    "Control more junctions, earn more points. You start in your Hub (center).",
                ),
                "task": "Click your Cog to select it, then explore your Hub and familiarize yourself with the area.",
            },
            {
                "title": "Step 2 — The Clips Threat",
                "lines": (
                    "[bold red]WARNING:[/bold red] Clips are automated enemies that expand territory!",
                    "Every ~300 steps, Clips [yellow]scramble[/yellow] nearby Cog junctions to neutral.",
                    "Every ~300 steps, Clips [yellow]capture[/yellow] nearby neutral junctions.",
                    "Clips expansion has a 25-cell radius. You must actively defend or be overrun!",
                ),
            },
            {
                "title": "Step 3 — Territory & Resources",
                "lines": (
                    "Junctions and Hubs project effects in a [bold]10-cell radius[/bold]:",
                    "[green]Friendly territory:[/green] Restores +100 HP, +100 energy, +10 influence per tick.",
                    "[red]Enemy territory:[/red] Drains -1 HP and -100 influence per tick.",
                    "[bold]HP:[/bold] Base 100. You lose -1 HP/tick outside friendly territory.",
                    "  At 0 HP, gear and hearts are [bold red]destroyed[/bold red].",
                    "[bold]Energy:[/bold] Base 20. Moving costs [yellow]3 energy[/yellow]. Regens +1/tick.",
                    "[yellow]Key insight:[/yellow] Aligners can't capture in enemy AOE (influence drains too fast).",
                ),
                "task": "Walk outside your Hub, watch your HP drain, then return to heal.",
            },
            {
                "title": "Step 4 — Gear Stations",
                "lines": (
                    "Equip gear at stations. Each costs 6 collective resources (different mixes):",
                    "[yellow]Miner[/yellow]: +40 cargo, 10x extraction. Cost: 1C/1O/[bold]3G[/bold]/1S",
                    "[yellow]Aligner[/yellow]: +20 influence cap, captures territory. Cost: [bold]3C[/bold]/1O/1G/1S",
                    "[yellow]Scrambler[/yellow]: +200 HP, disrupts enemy junctions. Cost: 1C/[bold]3O[/bold]/1G/1S",
                    "[yellow]Scout[/yellow]: +400 HP, +100 energy, mobile recon. Cost: 1C/1O/1G/[bold]3S[/bold]",
                    "Switching gear replaces your current gear (only hold one at a time).",
                ),
                "task": "Find a Gear Station in your base and equip Miner gear (walk into it).",
            },
            {
                "title": "Step 5 — Capturing & Scrambling",
                "lines": (
                    "[bold]To capture a neutral junction (Aligner only):[/bold]",
                    "  • Requires: Aligner gear + [yellow]1 heart[/yellow] + [yellow]1 influence[/yellow]",
                    "  • Must NOT be in enemy AOE (influence would be drained)",
                    "[bold]To scramble an enemy junction (Scrambler only):[/bold]",
                    "  • Requires: Scrambler gear + [yellow]1 heart[/yellow]",
                    "  • Converts enemy junction to neutral (then Aligners can capture it)",
                ),
            },
            {
                "title": "Step 6 — Resources & Hearts",
                "lines": (
                    "[bold]Extractors:[/bold] Walk into them to gather resources (1 per use, 10 with Miner gear).",
                    "[bold]Deposit:[/bold] Walk into the Hub (center of Hub) to deposit resources.",
                    "[bold]Hearts:[/bold] At the Chest, convert [yellow]1C + 1O + 1G + 1S[/yellow] into 1 heart.",
                    "  Hearts are spent to capture/scramble junctions.",
                    "[bold]Aligning:[/bold] Switch to Aligner gear, then walk into a neutral junction to capture it.",
                    "Team coordination: Miners gather → deposit → make hearts → Aligners/Scramblers use them.",
                ),
                "task": (
                    "Extract resources (C/O/G/S), deposit at the Hub, craft a heart, "
                    "then switch to Aligner and capture a junction."
                ),
            },
            {
                "title": "Step 7 — Tutorial Complete",
                "lines": (
                    "[bold green]Congratulations![/bold green] You've completed the CogsGuard tutorial.",
                    "",
                    "[bold]Remember the core loop:[/bold]",
                    "  1. Miners gather resources and deposit at the Hub",
                    "  2. Convert resources to hearts at the Chest",
                    "  3. Scramblers neutralize enemy junctions (1 heart each)",
                    "  4. Aligners capture neutral junctions (1 heart + 1 influence each)",
                    "  5. Defend against Clips expansion!",
                    "",
                    "[bold cyan]You're ready for full CogsGuard missions![/bold cyan]",
                ),
            },
        )

        for idx, step in enumerate(tutorial_steps):
            if stop_event.is_set():
                return
            console.print()
            console.print(f"[bold cyan]{step['title']}[/bold cyan]")
            console.print()
            for line in step["lines"]:
                console.print(f"  • {line}")
            # Display task if present
            if "task" in step:
                console.print()
                console.print(f"  [bold yellow]TASK:[/bold yellow] {step['task']}")
            console.print()
            if idx < len(tutorial_steps) - 1:
                console.print("[dim]Press Enter to continue...[/dim]")
                if not _wait_for_enter(""):
                    return

        console.print("[dim]CogsGuard tutorial briefing complete. Good luck, Cognitive.[/dim]")
        console.print("[dim]Close the Mettascope window to exit the tutorial.[/dim]")

    # Start tutorial interaction in a background thread
    tutorial_thread = threading.Thread(target=run_cogsguard_tutorial_steps, daemon=True)
    tutorial_thread.start()

    # Run play (blocks main thread)
    try:
        play_module.play(
            console,
            env_cfg=env_cfg,
            policy_spec=get_policy_spec(ctx, "class=noop"),
            game_name="cogsguard_tutorial",
            render_mode="gui",
        )
    except KeyboardInterrupt:
        logger.info("CogsGuard tutorial interrupted; exiting.")
    finally:
        stop_event.set()


app.add_typer(tutorial_app, name="tutorial", rich_help_panel="Tutorials")


def _help_callback(ctx: typer.Context, value: bool) -> None:
    """Callback for custom help option."""
    if value:
        console.print(ctx.get_help())
        raise typer.Exit()


@app.command(
    name="missions",
    help="""List available missions.

This command has three modes:

[bold]1. List sites:[/bold] Run with no arguments to see all available sites.

[bold]2. List missions at a site:[/bold] Pass a site name (e.g., 'cogsguard_machina_1') to see its missions.

[bold]3. Describe a mission:[/bold] Use -m to describe a specific mission. Only in this mode do \
--cogs, --variant, --format, and --save have any effect.""",
    rich_help_panel="Missions",
    epilog="""[dim]Examples:[/dim]

  [cyan]cogames missions[/cyan]                                    List all sites

  [cyan]cogames missions cogsguard_machina_1[/cyan]                     List missions at site

  [cyan]cogames missions -m cogsguard_machina_1.basic[/cyan]           Describe a mission

  [cyan]cogames missions -m arena --format json[/cyan]             Output as JSON""",
    add_help_option=False,
)
@app.command("games", hidden=True)
@app.command("mission", hidden=True)
def games_cmd(
    ctx: typer.Context,
    # --- List ---
    site: Optional[str] = typer.Argument(
        None,
        metavar="SITE",
        help="Filter by site (e.g., cogsguard_machina_1)",
    ),
    # --- Describe (requires -m) ---
    mission: Optional[str] = typer.Option(
        None,
        "--mission",
        "-m",
        metavar="MISSION",
        help="Mission to describe",
        rich_help_panel="Describe",
    ),
    cogs: Optional[int] = typer.Option(
        None,
        "--cogs",
        "-c",
        help="Override agent count (requires -m)",
        rich_help_panel="Describe",
    ),
    variant: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--variant",
        "-v",
        metavar="VARIANT",
        help="Apply variant (requires -m, repeatable)",
        rich_help_panel="Describe",
    ),
    format_: Optional[Literal["yaml", "json"]] = typer.Option(
        None,
        "--format",
        help="Output format (requires -m)",
        rich_help_panel="Describe",
    ),
    save: Optional[Path] = typer.Option(  # noqa: B008
        None,
        "--save",
        "-s",
        metavar="PATH",
        help="Save config to file (requires -m)",
        rich_help_panel="Describe",
    ),
    # --- Debug ---
    print_cvc_config: bool = typer.Option(
        False,
        "--print-cvc-config",
        help="Print CVC mission config (requires -m)",
        hidden=True,
    ),
    print_mg_config: bool = typer.Option(
        False,
        "--print-mg-config",
        help="Print MettaGrid config (requires -m)",
        hidden=True,
    ),
    # --- Help ---
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    if mission is None:
        list_missions(site)
        return

    try:
        resolved_mission, env_cfg, mission_cfg = get_mission_name_and_config(ctx, mission, variant, cogs)
    except typer.Exit as exc:
        if exc.exit_code != 1:
            raise
        return

    if print_cvc_config or print_mg_config:
        try:
            verbose.print_configs(console, env_cfg, mission_cfg, print_cvc_config, print_mg_config)
        except Exception as exc:
            console.print(f"[red]Error printing config: {exc}[/red]")
            raise typer.Exit(1) from exc

    if save is not None:
        try:
            game.save_mission_config(env_cfg, save)
            console.print(f"[green]Mission configuration saved to: {save}[/green]")
        except ValueError as exc:  # pragma: no cover - user input
            console.print(f"[red]Error saving configuration: {exc}[/red]")
            raise typer.Exit(1) from exc
        return

    if format_ is not None:
        try:
            data = env_cfg.model_dump(mode="json")
            if format_ == "json":
                console.print(json.dumps(data, indent=2))
            else:
                console.print(yaml.safe_dump(data, sort_keys=False))
        except Exception as exc:  # pragma: no cover - serialization errors
            console.print(f"[red]Error formatting configuration: {exc}[/red]")
            raise typer.Exit(1) from exc
        return

    try:
        describe_mission(resolved_mission, env_cfg, mission_cfg)
    except ValueError as exc:  # pragma: no cover - user input
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc


@app.command("evals", help="List all eval missions", rich_help_panel="Missions")
def evals_cmd() -> None:
    list_evals()


@app.command("variants", help="List all available mission variants", rich_help_panel="Missions")
def variants_cmd() -> None:
    list_variants()


@app.command(
    name="describe",
    help="Describe a mission and its configuration",
    rich_help_panel="Missions",
    epilog="""[dim]Examples:[/dim]

  [cyan]cogames describe hello_world.open_world[/cyan]                Describe mission

  [cyan]cogames describe arena -c 4 -v dark_side[/cyan]               With 4 cogs and variant""",
    add_help_option=False,
)
def describe_cmd(
    ctx: typer.Context,
    mission: str = typer.Argument(
        ...,
        metavar="MISSION",
        help="Mission name (e.g., hello_world.open_world)",
    ),
    cogs: Optional[int] = typer.Option(
        None,
        "--cogs",
        "-c",
        help="Number of cogs (agents)",
        rich_help_panel="Configuration",
    ),
    variant: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--variant",
        "-v",
        metavar="VARIANT",
        help="Apply variant (repeatable)",
        rich_help_panel="Configuration",
    ),
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    resolved_mission, env_cfg, mission_cfg = get_mission_name_and_config(ctx, mission, variant, cogs)
    describe_mission(resolved_mission, env_cfg, mission_cfg)


@app.command(
    name="play",
    rich_help_panel="Play",
    help="""Play a game interactively.

This runs a single episode of the game using the specified policy.

By default, the policy is 'noop', so agents won't move unless manually controlled.
To see agents move by themselves, use `--policy class=random` or `--policy class=baseline`.

You can manually control the actions of a specific cog by clicking on a cog
in GUI mode or pressing M in unicode mode and using your arrow or WASD keys.
Log mode is non-interactive and doesn't support manual control.
""",
    epilog="""[dim]Examples:[/dim]

[cyan]cogames play -m cogsguard_machina_1.basic[/cyan]                        Interactive

[cyan]cogames play -m cogsguard_machina_1.basic -p class=random[/cyan]        Random policy

[cyan]cogames play -m cogsguard_machina_1.basic -c 4 -p class=baseline[/cyan] Baseline, 4 cogs

[cyan]cogames play -m cogsguard_machina_1 -r unicode[/cyan]                   Terminal mode""",
    add_help_option=False,
)
def play_cmd(
    ctx: typer.Context,
    # --- Game Setup ---
    mission: Optional[str] = typer.Option(
        None,
        "--mission",
        "-m",
        metavar="MISSION",
        help="Mission to play (run [bold]cogames missions[/bold] to list)",
        rich_help_panel="Game Setup",
    ),
    variant: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--variant",
        "-v",
        metavar="VARIANT",
        help="Apply variant modifier (repeatable)",
        rich_help_panel="Game Setup",
    ),
    cogs: Optional[int] = typer.Option(
        None,
        "--cogs",
        "-c",
        metavar="N",
        help="Number of cogs/agents",
        show_default="from mission",
        rich_help_panel="Game Setup",
    ),
    # --- Policy ---
    policy: str = typer.Option(
        "class=noop",
        "--policy",
        "-p",
        metavar="POLICY",
        help="Policy controlling cogs ([bold]noop[/bold], [bold]random[/bold], [bold]lstm[/bold], or path)",
        rich_help_panel="Policy",
    ),
    # --- Simulation ---
    steps: int = typer.Option(
        1000,
        "--steps",
        "-s",
        metavar="N",
        help="Max steps per episode",
        rich_help_panel="Simulation",
    ),
    render: RenderMode = typer.Option(  # noqa: B008
        "gui",
        "--render",
        "-r",
        help=(
            "[bold]gui[/bold]=MettaScope, [bold]vibescope[/bold]=VibeScope, "
            "[bold]unicode[/bold]=terminal, [bold]log[/bold]=metrics only"
        ),
        rich_help_panel="Simulation",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="RNG seed for reproducibility",
        rich_help_panel="Simulation",
    ),
    map_seed: Optional[int] = typer.Option(
        None,
        "--map-seed",
        metavar="SEED",
        help="Separate seed for procedural map generation",
        show_default="same as --seed",
        rich_help_panel="Simulation",
    ),
    # --- Output ---
    save_replay_dir: Optional[Path] = typer.Option(  # noqa: B008
        None,
        "--save-replay-dir",
        metavar="DIR",
        help="Save replay file for later viewing with [bold]cogames replay[/bold]",
        rich_help_panel="Output",
    ),
    # --- Debug (hidden from casual users) ---
    print_cvc_config: bool = typer.Option(
        False,
        "--print-cvc-config",
        help="Print mission config and exit",
        rich_help_panel="Debug",
        hidden=True,
    ),
    print_mg_config: bool = typer.Option(
        False,
        "--print-mg-config",
        help="Print MettaGrid config and exit",
        rich_help_panel="Debug",
        hidden=True,
    ),
    # --- Help at end ---
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    resolved_mission, env_cfg, mission_cfg = get_mission_name_and_config(ctx, mission, variant, cogs)

    if print_cvc_config or print_mg_config:
        try:
            verbose.print_configs(console, env_cfg, mission_cfg, print_cvc_config, print_mg_config)
        except Exception as exc:
            console.print(f"[red]Error printing config: {exc}[/red]")
            raise typer.Exit(1) from exc

    # Optional MapGen seed override for procedural maps.
    if map_seed is not None:
        map_builder = getattr(env_cfg.game, "map_builder", None)
        if isinstance(map_builder, MapGen.Config):
            map_builder.seed = map_seed

    policy_spec = get_policy_spec(ctx, policy)
    console.print(f"[cyan]Playing {resolved_mission}[/cyan]")
    console.print(f"Max Steps: {steps}, Render: {render}")

    if ctx.get_parameter_source("steps") in (
        ParameterSource.COMMANDLINE,
        ParameterSource.ENVIRONMENT,
        ParameterSource.PROMPT,
    ):
        env_cfg.game.max_steps = steps

    play_module.play(
        console,
        env_cfg=env_cfg,
        policy_spec=policy_spec,
        seed=seed,
        render_mode=render,
        game_name=resolved_mission,
        save_replay=save_replay_dir,
    )


@app.command(
    name="replay",
    help="Replay a saved game episode from a file in the GUI",
    rich_help_panel="Play",
    epilog="""[dim]Examples:[/dim]

  [cyan]cogames replay ./replays/game.replay[/cyan]              Replay a saved game

  [cyan]cogames replay ./train_dir/my_run/replay.bin[/cyan]      Replay from training run""",
    add_help_option=False,
)
def replay_cmd(
    replay_path: Path = typer.Argument(  # noqa: B008
        ...,
        metavar="FILE",
        help="Path to the replay file (.replay or .bin)",
    ),
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit",
        is_eager=True,
        callback=_help_callback,
    ),
) -> None:
    if not replay_path.exists():
        console.print(f"[red]Error: Replay file not found: {replay_path}[/red]")
        raise typer.Exit(1)

    try:
        mettascope_path = _resolve_mettascope_script()
    except FileNotFoundError as exc:
        console.print(f"[red]Error locating MettaScope: {exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(f"[cyan]Launching MettaScope to replay: {replay_path}[/cyan]")

    try:
        # Run nim with mettascope and replay argument
        cmd = ["nim", "r", str(mettascope_path), f"--replay:{replay_path}"]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]Error running MettaScope: {exc}[/red]")
        raise typer.Exit(1) from exc
    except FileNotFoundError as exc:
        console.print("[red]Error: 'nim' command not found. Please ensure Nim is installed and in your PATH.[/red]")
        raise typer.Exit(1) from exc


@app.command(
    name="make-mission",
    help="Create a custom mission from a base template",
    rich_help_panel="Missions",
    epilog="""[dim]Examples:[/dim]

  [cyan]cogames make-mission -m hello_world -c 8 -o my_mission.yml[/cyan]             8 cogs

  [cyan]cogames make-mission -m arena --width 64 --height 64 -o big.yml[/cyan]        64x64 map

  [cyan]cogames play -m my_mission.yml[/cyan]                                         Use custom mission""",
    add_help_option=False,
)
@app.command("make-game", hidden=True)
def make_mission(
    ctx: typer.Context,
    # --- Mission ---
    base_mission: Optional[str] = typer.Option(
        None,
        "--mission",
        "-m",
        metavar="MISSION",
        help="Base mission to start from",
        rich_help_panel="Mission",
    ),
    # --- Customization ---
    cogs: Optional[int] = typer.Option(
        None,
        "--cogs",
        "-c",
        help="Number of cogs (agents)",
        min=1,
        rich_help_panel="Customization",
    ),
    width: Optional[int] = typer.Option(
        None,
        "--width",
        help="Map width",
        min=1,
        rich_help_panel="Customization",
    ),
    height: Optional[int] = typer.Option(
        None,
        "--height",
        help="Map height",
        min=1,
        rich_help_panel="Customization",
    ),
    # --- Output ---
    output: Optional[Path] = typer.Option(  # noqa: B008
        None,
        "--output",
        "-o",
        metavar="PATH",
        help="Output file path (.yml or .json)",
        rich_help_panel="Output",
    ),
    # --- Help ---
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    try:
        resolved_mission, env_cfg, _ = get_mission_name_and_config(ctx, base_mission)

        # Update map dimensions if explicitly provided and supported
        if width is not None:
            if not hasattr(env_cfg.game.map_builder, "width"):
                console.print("[yellow]Warning: Map builder does not support custom width. Ignoring --width.[/yellow]")
            else:
                env_cfg.game.map_builder.width = width  # type: ignore[attr-defined]

        if height is not None:
            if not hasattr(env_cfg.game.map_builder, "height"):
                console.print(
                    "[yellow]Warning: Map builder does not support custom height. Ignoring --height.[/yellow]"
                )
            else:
                env_cfg.game.map_builder.height = height  # type: ignore[attr-defined]

        if cogs is not None:
            env_cfg.game.num_agents = cogs

        # Validate the environment configuration

        _ = Simulator().new_simulation(env_cfg)

        if output:
            game.save_mission_config(env_cfg, output)
            console.print(f"[green]Modified {resolved_mission} configuration saved to: {output}[/green]")
        else:
            console.print("\n[yellow]To save this configuration, use the --output option.[/yellow]")

    except Exception as exc:  # pragma: no cover - user input
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc


# TODO (cogsguard migration): Verify make-policy templates work with CogsGuard game mechanics
@tutorial_app.command(
    name="make-policy",
    help="Create a new policy from a template. Requires --trainable or --scripted.",
    rich_help_panel="Tutorial",
    epilog="""[dim]Examples:[/dim]

[cyan]cogames tutorial make-policy -t -o my_nn_policy.py[/cyan]        Trainable (neural network)

[cyan]cogames tutorial make-policy -s -o my_scripted_policy.py[/cyan]  Scripted (rule-based)""",
    add_help_option=False,
)
def make_policy(
    # --- Policy Type ---
    trainable: bool = typer.Option(
        False,
        "--trainable",
        help="Create a trainable (neural network) policy",
        rich_help_panel="Policy Type",
    ),
    scripted: bool = typer.Option(
        False,
        "--scripted",
        help="Create a scripted (rule-based) policy",
        rich_help_panel="Policy Type",
    ),
    # --- Output ---
    output: Path = typer.Option(  # noqa: B008
        "my_policy.py",
        "--output",
        "-o",
        metavar="FILE",
        help="Output file path",
        rich_help_panel="Output",
    ),
    # --- Help ---
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    if trainable == scripted:
        console.print("[red]Error: Specify exactly one of --trainable or --scripted[/red]")
        console.print("[dim]Examples:[/dim]")
        console.print("[dim]  cogames make-policy --trainable -o my_nn_policy.py[/dim]")
        console.print("[dim]  cogames make-policy --scripted -o my_scripted_policy.py[/dim]")
        raise typer.Exit(1)

    try:
        if trainable:
            template_path = Path(trainable_policy_template.__file__)
            policy_class = "MyTrainablePolicy"
            policy_type = "Trainable"
        else:
            template_path = Path(starter_agent.__file__)
            policy_class = "StarterPolicy"
            policy_type = "Scripted"

        if not template_path.exists():
            console.print(f"[red]Error: {policy_type} policy template not found[/red]")
            raise typer.Exit(1)

        dest_path = Path.cwd() / output

        if dest_path.exists():
            console.print(f"[yellow]Warning: {dest_path} already exists. Overwriting...[/yellow]")

        shutil.copy2(template_path, dest_path)
        console.print(f"[green]{policy_type} policy template copied to: {dest_path}[/green]")

        if not trainable:
            content = dest_path.read_text()
            lines = content.splitlines()
            lines = [line for line in lines if not line.strip().startswith("short_names =")]
            dest_path.write_text("\n".join(lines) + "\n")

        if trainable:
            console.print(
                "[dim]Train with: cogames tutorial train -m cogsguard_machina_1.basic -p class="
                f"{dest_path.stem}.{policy_class}[/dim]"
            )
        else:
            console.print(
                "[dim]Play with: cogames play -m cogsguard_machina_1.basic -p class="
                f"{dest_path.stem}.{policy_class}[/dim]"
            )

    except Exception as exc:  # pragma: no cover - user input
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc


app.command(name="make-policy", hidden=True)(make_policy)


@tutorial_app.command(
    name="train",
    help="""Train a policy on one or more missions.

By default, our 'lstm' policy architecture is used. You can select a different architecture
(like 'stateless' or 'baseline'), or define your own implementing the MultiAgentPolicy
interface with a trainable network() method (see mettagrid/policy/policy.py).

Continue training from a checkpoint using URI format, or load weights into an explicit class
with class=...,data=... syntax.

Supply repeated -m flags to create a training curriculum that rotates through missions.
Use wildcards (*) in mission names to match multiple missions at once.""",
    rich_help_panel="Tutorial",
    epilog="""[dim]Examples:[/dim]

[cyan]cogames tutorial train -m cogsguard_machina_1.basic[/cyan]                   Basic training

[cyan]cogames tutorial train -m cogsguard_machina_1.basic -p class=baseline[/cyan]
                                                                 Train baseline policy

[cyan]cogames tutorial train -p ./train_dir/my_run:v5[/cyan]                  Continue from checkpoint

[cyan]cogames tutorial train -p class=lstm,data=./weights.safetensors[/cyan]  Load weights into class

[cyan]cogames tutorial train -m mission_1 -m mission_2[/cyan]                 Curriculum (rotates)

[dim]Wildcard patterns:[/dim]

[cyan]cogames tutorial train -m 'machina_2_bigger:*'[/cyan]                   All missions on machina_2_bigger

[cyan]cogames tutorial train -m '*:shaped'[/cyan]                             All "shaped" missions

[cyan]cogames tutorial train -m 'machina*:shaped'[/cyan]                      All "shaped" on machina maps""",
    add_help_option=False,
)
def train_cmd(
    ctx: typer.Context,
    # --- Mission Setup ---
    missions: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--mission",
        "-m",
        metavar="MISSION",
        help="Missions to train on (wildcards supported, repeatable for curriculum)",
        rich_help_panel="Mission Setup",
    ),
    cogs: Optional[int] = typer.Option(
        None,
        "--cogs",
        "-c",
        metavar="N",
        help="Number of cogs (agents)",
        show_default="from mission",
        rich_help_panel="Mission Setup",
    ),
    variant: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--variant",
        "-v",
        metavar="VARIANT",
        help="Mission variant (repeatable)",
        rich_help_panel="Mission Setup",
    ),
    # --- Policy ---
    policy: str = typer.Option(
        "class=lstm",
        "--policy",
        "-p",
        metavar="POLICY",
        help=f"Policy to train ({policy_arg_example})",
        rich_help_panel="Policy",
    ),
    # --- Training ---
    steps: int = typer.Option(
        10_000_000_000,
        "--steps",
        metavar="N",
        help="Number of training steps",
        min=1,
        rich_help_panel="Training",
    ),
    batch_size: int = typer.Option(
        4096,
        "--batch-size",
        metavar="N",
        help="Batch size for training",
        min=1,
        rich_help_panel="Training",
    ),
    minibatch_size: int = typer.Option(
        4096,
        "--minibatch-size",
        metavar="N",
        help="Minibatch size for training",
        min=1,
        rich_help_panel="Training",
    ),
    # --- Hardware ---
    device: str = typer.Option(
        "cpu",
        "--device",
        metavar="DEVICE",
        help="Device to train on (auto, cpu, cuda, mps)",
        rich_help_panel="Hardware",
    ),
    num_workers: Optional[int] = typer.Option(
        None,
        "--num-workers",
        metavar="N",
        help="Number of worker processes",
        show_default="CPU cores",
        min=1,
        rich_help_panel="Hardware",
    ),
    parallel_envs: Optional[int] = typer.Option(
        None,
        "--parallel-envs",
        metavar="N",
        help="Number of parallel environments",
        min=1,
        rich_help_panel="Hardware",
    ),
    vector_batch_size: Optional[int] = typer.Option(
        None,
        "--vector-batch-size",
        metavar="N",
        help="Vectorized environment batch size",
        min=1,
        rich_help_panel="Hardware",
    ),
    # --- Reproducibility ---
    seed: int = typer.Option(
        42,
        "--seed",
        metavar="N",
        help="Seed for training RNG",
        min=0,
        rich_help_panel="Reproducibility",
    ),
    map_seed: Optional[int] = typer.Option(
        None,
        "--map-seed",
        metavar="N",
        help="MapGen seed for procedural map layout",
        show_default="same as --seed",
        min=0,
        rich_help_panel="Reproducibility",
    ),
    # --- Output ---
    checkpoints_path: str = typer.Option(
        "./train_dir",
        "--checkpoints",
        metavar="DIR",
        help="Path to save training checkpoints",
        rich_help_panel="Output",
    ),
    log_outputs: bool = typer.Option(
        False,
        "--log-outputs",
        help="Log training outputs",
        rich_help_panel="Output",
    ),
    # --- Help ---
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    selected_missions = get_mission_names_and_configs(ctx, missions, variants_arg=variant, cogs=cogs)
    if len(selected_missions) == 1:
        mission_name, env_cfg = selected_missions[0]
        supplier = None
        console.print(f"Training on mission: {mission_name}\n")
    elif len(selected_missions) > 1:
        env_cfg = None
        supplier = make_rotation(selected_missions)
        console.print("Training on missions:\n" + "\n".join(f"- {m}" for m, _ in selected_missions) + "\n")
    else:
        # Should not get here
        raise ValueError("Please specify at least one mission")

    policy_spec = get_policy_spec(ctx, policy)
    torch_device = resolve_training_device(console, device)

    try:
        train_module.train(
            env_cfg=env_cfg,
            policy_class_path=policy_spec.class_path,
            initial_weights_path=policy_spec.data_path,
            device=torch_device,
            num_steps=steps,
            checkpoints_path=Path(checkpoints_path),
            seed=seed,
            map_seed=map_seed,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            vector_num_workers=num_workers,
            vector_num_envs=parallel_envs,
            vector_batch_size=vector_batch_size,
            env_cfg_supplier=supplier,
            missions_arg=missions,
            log_outputs=log_outputs,
        )

    except ValueError as exc:  # pragma: no cover - user input
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(f"[green]Training complete. Checkpoints saved to: {checkpoints_path}[/green]")


app.command(name="train", hidden=True)(train_cmd)


@app.command(
    name="run",
    help="""Evaluate one or more policies on missions.

With multiple policies (e.g., 2 policies, 4 agents), each policy always controls 2 agents,
but which agents swap between policies each episode.

With one policy, this command is equivalent to `cogames scrimmage`.
""",
    rich_help_panel="Evaluate",
    epilog="""[dim]Examples:[/dim]

[cyan]cogames run -m cogsguard_machina_1.basic -p lstm[/cyan]               Evaluate single policy

[cyan]cogames run -m cogsguard_machina_1 -p ./train_dir/my_run:v5[/cyan]     Evaluate a checkpoint bundle

[cyan]cogames run -S integrated_evals -p ./train_dir/my_run:v5[/cyan]    Evaluate on mission set

[cyan]cogames run -m 'arena.*' -p lstm -p random -e 20[/cyan]            Evaluate multiple policies together

[cyan]cogames run -m cogsguard_machina_1 -p ./train_dir/my_run:v5,proportion=3 -p class=random,proportion=5[/cyan]
                                                             Evaluate policies in 3:5 mix""",
    add_help_option=False,
)
@app.command(
    name="scrimmage",
    help="""Evaluate a single policy controlling all agents.

This command is equivalent to running `cogames run` with a single policy.
""",
    rich_help_panel="Evaluate",
    epilog="""[dim]Examples:[/dim]

[cyan]cogames scrimmage -m arena.battle -p lstm[/cyan]                   Single policy eval""",
    add_help_option=False,
)
@app.command("eval", hidden=True)
@app.command("evaluate", hidden=True)
def run_cmd(
    ctx: typer.Context,
    # --- Mission ---
    missions: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--mission",
        "-m",
        metavar="MISSION",
        help="Missions to evaluate (supports wildcards)",
        rich_help_panel="Mission",
    ),
    mission_set: Optional[str] = typer.Option(
        None,
        "--mission-set",
        "-S",
        metavar="SET",
        help="Predefined set: integrated_evals, spanning_evals, diagnostic_evals, all",
        rich_help_panel="Mission",
    ),
    cogs: Optional[int] = typer.Option(
        None,
        "--cogs",
        "-c",
        metavar="N",
        help="Number of cogs (agents)",
        rich_help_panel="Mission",
    ),
    variant: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--variant",
        "-v",
        metavar="VARIANT",
        help="Mission variant (repeatable)",
        rich_help_panel="Mission",
    ),
    # --- Policy ---
    policies: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--policy",
        "-p",
        metavar="POLICY",
        help=f"Policies to evaluate: ({policy_arg_w_proportion_example}...)",
        rich_help_panel="Policy",
    ),
    # --- Simulation ---
    episodes: int = typer.Option(
        10,
        "--episodes",
        "-e",
        metavar="N",
        help="Number of evaluation episodes",
        min=1,
        rich_help_panel="Simulation",
    ),
    steps: Optional[int] = typer.Option(
        1000,
        "--steps",
        "-s",
        metavar="N",
        help="Max steps per episode",
        min=1,
        rich_help_panel="Simulation",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        metavar="N",
        help="Seed for evaluation RNG",
        min=0,
        rich_help_panel="Simulation",
    ),
    map_seed: Optional[int] = typer.Option(
        None,
        "--map-seed",
        metavar="N",
        help="MapGen seed for procedural maps",
        min=0,
        show_default="same as --seed",
        rich_help_panel="Simulation",
    ),
    action_timeout_ms: int = typer.Option(
        250,
        "--action-timeout-ms",
        metavar="MS",
        help="Max ms per action before noop",
        min=1,
        rich_help_panel="Simulation",
    ),
    # --- Output ---
    format_: Optional[Literal["yaml", "json"]] = typer.Option(
        None,
        "--format",
        metavar="FMT",
        help="Output format: yaml or json",
        rich_help_panel="Output",
    ),
    save_replay_dir: Optional[Path] = typer.Option(  # noqa: B008
        None,
        "--save-replay-dir",
        metavar="DIR",
        help="Directory to save replays",
        rich_help_panel="Output",
    ),
    # --- Help ---
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    # Handle mission set expansion
    if mission_set and missions:
        console.print("[red]Error: Cannot use both --mission-set and --mission[/red]")
        raise typer.Exit(1)

    if mission_set:
        from cogames.cli.mission import load_mission_set

        try:
            mission_objs = load_mission_set(mission_set)
            missions = [m.full_name() for m in mission_objs]
            console.print(f"[cyan]Using mission set '{mission_set}' ({len(missions)} missions)[/cyan]")
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from e

        # Default to 4 cogs for mission sets unless explicitly specified
        if cogs is None:
            cogs = 4

    selected_missions = get_mission_names_and_configs(ctx, missions, variants_arg=variant, cogs=cogs, steps=steps)

    # Optional MapGen seed override for procedural maps.
    if map_seed is not None:
        for _, env_cfg in selected_missions:
            map_builder = getattr(env_cfg.game, "map_builder", None)
            if isinstance(map_builder, MapGen.Config):
                map_builder.seed = map_seed

    policy_specs = get_policy_specs_with_proportions(ctx, policies)

    if ctx.info_name == "scrimmage":
        if len(policy_specs) != 1:
            console.print("[red]Error: scrimmage accepts exactly one --policy / -p value.[/red]")
            raise typer.Exit(1)
        if policy_specs[0].proportion != 1.0:
            console.print("[red]Error: scrimmage does not support policy proportions.[/red]")
            raise typer.Exit(1)

    console.print(
        f"[cyan]Preparing evaluation for {len(policy_specs)} policies across {len(selected_missions)} mission(s)[/cyan]"
    )

    evaluate_module.evaluate(
        console,
        missions=selected_missions,
        policy_specs=[spec.to_policy_spec() for spec in policy_specs],
        proportions=[spec.proportion for spec in policy_specs],
        action_timeout_ms=action_timeout_ms,
        episodes=episodes,
        seed=seed,
        output_format=format_,
        save_replay=str(save_replay_dir) if save_replay_dir else None,
    )


@app.command(
    name="pickup",
    help="Evaluate a policy against a pool of other policies and compute VOR",
    rich_help_panel="Evaluate",
    epilog="""[dim]Examples:[/dim]

[cyan]cogames pickup -p greedy --pool random[/cyan]                      Test greedy against pool of random""",
    add_help_option=False,
)
def pickup_cmd(
    ctx: typer.Context,
    # --- Mission ---
    mission: str = typer.Option(
        "cogsguard_machina_1.basic",
        "--mission",
        "-m",
        metavar="MISSION",
        help="Mission to evaluate on",
        rich_help_panel="Mission",
    ),
    cogs: int = typer.Option(
        4,
        "--cogs",
        "-c",
        metavar="N",
        help="Number of cogs (agents)",
        min=1,
        rich_help_panel="Mission",
    ),
    variant: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--variant",
        "-v",
        metavar="VARIANT",
        help="Mission variant (repeatable)",
        rich_help_panel="Mission",
    ),
    # --- Policy ---
    policy: Optional[str] = typer.Option(
        None,
        "--policy",
        "-p",
        metavar="POLICY",
        help="Candidate policy to evaluate",
        rich_help_panel="Policy",
    ),
    pool: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--pool",
        metavar="POLICY",
        help="Pool policy (repeatable)",
        rich_help_panel="Policy",
    ),
    # --- Simulation ---
    episodes: int = typer.Option(
        1,
        "--episodes",
        "-e",
        metavar="N",
        help="Episodes per scenario",
        min=1,
        rich_help_panel="Simulation",
    ),
    steps: Optional[int] = typer.Option(
        1000,
        "--steps",
        "-s",
        metavar="N",
        help="Max steps per episode",
        min=1,
        rich_help_panel="Simulation",
    ),
    seed: int = typer.Option(
        50,
        "--seed",
        metavar="N",
        help="Base random seed",
        min=0,
        rich_help_panel="Simulation",
    ),
    map_seed: Optional[int] = typer.Option(
        None,
        "--map-seed",
        metavar="N",
        help="MapGen seed for procedural maps",
        min=0,
        show_default="same as --seed",
        rich_help_panel="Simulation",
    ),
    action_timeout_ms: int = typer.Option(
        250,
        "--action-timeout-ms",
        metavar="MS",
        help="Max ms per action before noop",
        min=1,
        rich_help_panel="Simulation",
    ),
    # --- Output ---
    save_replay_dir: Optional[Path] = typer.Option(  # noqa: B008
        None,
        "--save-replay-dir",
        metavar="DIR",
        help="Directory to save replays",
        rich_help_panel="Output",
    ),
    # --- Help ---
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    import httpx

    if policy is None:
        console.print(ctx.get_help())
        console.print("[yellow]Missing: --policy / -p[/yellow]\n")
        raise typer.Exit(1)

    if not pool:
        console.print(ctx.get_help())
        console.print("[yellow]Supply at least one: --pool[/yellow]\n")
        raise typer.Exit(1)

    # Resolve mission
    resolved_mission, env_cfg, _ = get_mission_name_and_config(ctx, mission, variants_arg=variant, cogs=cogs)
    if steps is not None:
        env_cfg.game.max_steps = steps

    candidate_label = policy
    pool_labels = pool
    candidate_spec = get_policy_spec(ctx, policy)
    try:
        pool_specs = [parse_policy_spec(spec).to_policy_spec() for spec in pool]
    except (ValueError, ModuleNotFoundError, httpx.HTTPError) as exc:
        translated = _translate_error(exc)
        console.print(f"[yellow]Error parsing pool policy: {translated}[/yellow]\n")
        raise typer.Exit(1) from exc

    pickup_module.pickup(
        console,
        candidate_spec,
        pool_specs,
        env_cfg=env_cfg,
        mission_name=resolved_mission,
        episodes=episodes,
        seed=seed,
        map_seed=map_seed,
        action_timeout_ms=action_timeout_ms,
        save_replay_dir=save_replay_dir,
        candidate_label=candidate_label,
        pool_labels=pool_labels,
    )


@app.command(
    name="version",
    help="Show version information for cogames and dependencies",
    rich_help_panel="Info",
)
def version_cmd() -> None:
    def public_version(dist_name: str) -> str:
        return str(Version(importlib.metadata.version(dist_name)).public)

    table = Table(show_header=False, box=None, show_lines=False, pad_edge=False)
    table.add_column("", justify="right", style="bold cyan")
    table.add_column("", justify="right")

    for dist_name in ["mettagrid", "pufferlib-core", "cogames"]:
        table.add_row(dist_name, public_version(dist_name))

    console.print(table)


@app.command(
    name="policies",
    help="Show available policy shorthand names",
    rich_help_panel="Policies",
    epilog="""[dim]Usage:[/dim]

  Use these shorthand names with [cyan]--policy[/cyan] or [cyan]-p[/cyan]:

  [cyan]cogames play -m arena -p class=random[/cyan]     Use random policy

  [cyan]cogames play -m arena -p class=baseline[/cyan]   Use baseline policy""",
)
def policies_cmd() -> None:
    policy_registry = get_policy_registry()
    table = Table(show_header=False, box=None, show_lines=False, pad_edge=False)
    table.add_column("", justify="left", style="bold cyan")
    table.add_column("", justify="right")

    for policy_name, policy_path in policy_registry.items():
        table.add_row(policy_name, policy_path)
    table.add_row("custom", "path.to.your.PolicyClass")

    console.print(table)


@app.command(
    name="login",
    help="Authenticate with CoGames server",
    rich_help_panel="Tournament",
    epilog="""[dim]Examples:[/dim]

[cyan]cogames login[/cyan]                       Authenticate with default server

[cyan]cogames login --force[/cyan]               Re-authenticate even if already logged in""",
    add_help_option=False,
)
def login_cmd(
    server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--login-server",
        metavar="URL",
        help="Authentication server URL",
        rich_help_panel="Server",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Re-authenticate even if already logged in",
        rich_help_panel="Options",
    ),
    timeout: int = typer.Option(
        300,
        "--timeout",
        "-t",
        metavar="SECS",
        help="Authentication timeout in seconds",
        rich_help_panel="Options",
    ),
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    from urllib.parse import urlparse

    # Check if we already have a token
    from cogames.auth import BaseCLIAuthenticator

    temp_auth = BaseCLIAuthenticator(
        token_file_name="cogames.yaml",
        token_storage_key="login_tokens",
    )

    if temp_auth.has_saved_token(server) and not force:
        console.print(f"[green]Already authenticated with {urlparse(server).hostname}[/green]")
        return

    # Perform authentication
    console.print(f"[cyan]Authenticating with {server}...[/cyan]")
    if perform_login(auth_server_url=server, force=force, timeout=timeout):
        console.print("[green]Authentication successful![/green]")
    else:
        console.print("[red]Authentication failed![/red]")
        raise typer.Exit(1)


app.command(
    name="submissions",
    help="Show your uploads and tournament submissions",
    rich_help_panel="Tournament",
    epilog="""[dim]Examples:[/dim]

[cyan]cogames submissions[/cyan]                         All your uploads

[cyan]cogames submissions --season beta-cogsguard[/cyan]           Submissions in a season

[cyan]cogames submissions -p my-policy[/cyan]            Info on a specific policy""",
    add_help_option=False,
)(submissions_cmd)

app.command(
    name="seasons",
    help="List currently running tournament seasons",
    rich_help_panel="Tournament",
    add_help_option=False,
)(seasons_cmd)

app.command(
    name="leaderboard",
    help="Show tournament leaderboard for a season",
    rich_help_panel="Tournament",
    epilog="""[dim]Examples:[/dim]

[cyan]cogames leaderboard --season beta-cogsguard[/cyan]           View rankings""",
    add_help_option=False,
)(leaderboard_cmd)


@app.command(
    name="diagnose",
    help="Run diagnostic evals for a policy checkpoint",
    rich_help_panel="Evaluate",
    epilog="""[dim]Examples:[/dim]

[cyan]cogames diagnose ./train_dir/my_run[/cyan]                         Default diagnostics

[cyan]cogames diagnose lstm -S tournament[/cyan]                         Tournament suite

[cyan]cogames diagnose lstm -c 4 -c 8 -e 5[/cyan]                        Custom cog counts""",
    add_help_option=False,
)
def diagnose_cmd(
    policy: str = typer.Argument(
        ...,
        metavar="POLICY",
        help=f"Policy specification: {policy_arg_example}",
    ),
    # --- Evaluation ---
    mission_set: Literal[
        "diagnostic_evals",
        "integrated_evals",
        "spanning_evals",
        "thinky_evals",
        "tournament",
        "all",
    ] = typer.Option(
        "diagnostic_evals",
        "--mission-set",
        "-S",
        metavar="SET",
        help="Eval suite to run",
        rich_help_panel="Evaluation",
    ),
    experiments: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--experiments",
        metavar="NAME",
        help="Specific experiments (subset of mission set)",
        rich_help_panel="Evaluation",
    ),
    cogs: Optional[list[int]] = typer.Option(  # noqa: B008
        None,
        "--cogs",
        "-c",
        metavar="N",
        help="Agent counts to test (repeatable)",
        rich_help_panel="Evaluation",
    ),
    # --- Simulation ---
    steps: int = typer.Option(
        1000,
        "--steps",
        "-s",
        metavar="N",
        help="Max steps per episode",
        rich_help_panel="Simulation",
    ),
    episodes: int = typer.Option(
        3,
        "--episodes",
        "-e",
        metavar="N",
        help="Episodes per case",
        rich_help_panel="Simulation",
    ),
    # --- Help ---
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_evaluation.py"

    cmd = [sys.executable, str(script_path)]
    cmd.extend(["--mission-set", mission_set])

    if experiments:
        cmd.append("--experiments")
        cmd.extend(experiments)

    if cogs:
        cmd.append("--cogs")
        cmd.extend(str(c) for c in cogs)

    cmd.extend(["--steps", str(steps)])
    cmd.extend(["--repeats", str(episodes)])
    cmd.append("--no-plots")

    cmd.extend(["--policy", policy])

    console.print("[cyan]Running diagnostic evaluation...[/cyan]")
    console.print(f"[dim]{' '.join(cmd)}[/dim]")
    subprocess.run(cmd, check=True)


def _resolve_season(server: str, season_name: str | None = None) -> SeasonInfo:
    try:
        if season_name is not None:
            info = fetch_season_info(server, season_name)
            console.print(f"[dim]Using season: {info.name}[/dim]")
        else:
            info = fetch_default_season(server)
            console.print(f"[dim]Using default season: {info.name}[/dim]")
        return info
    except Exception as e:
        console.print(f"[red]Could not fetch season from server:[/red] {e}")
        console.print("Specify a season explicitly with [cyan]--season[/cyan]")
        raise typer.Exit(1) from None


@app.command(
    name="validate-policy",
    help="Validate the policy loads and runs for at least a single step",
    rich_help_panel="Policies",
    add_help_option=False,
)
def validate_policy_cmd(
    ctx: typer.Context,
    policy: str = typer.Option(
        ...,
        "--policy",
        "-p",
        metavar="POLICY",
        help=f"Policy specification: {policy_arg_example}",
        rich_help_panel="Policy",
    ),
    setup_script: Optional[str] = typer.Option(
        None,
        "--setup-script",
        help="Path to a Python setup script to run before loading the policy",
        rich_help_panel="Policy",
    ),
    season: Optional[str] = typer.Option(
        None,
        "--season",
        metavar="SEASON",
        help="Tournament season (determines which game to validate against)",
        rich_help_panel="Tournament",
    ),
    server: str = typer.Option(
        DEFAULT_SUBMIT_SERVER,
        "--server",
        metavar="URL",
        help="Tournament server URL (used to resolve default season)",
        rich_help_panel="Server",
    ),
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    season_info = _resolve_season(server, season)
    entry_pool_info = next((p for p in season_info.pools if p.name == season_info.entry_pool), None)
    if not entry_pool_info or not entry_pool_info.config_id:
        console.print("[red]No entry config found for season[/red]")
        raise typer.Exit(1)

    with TournamentServerClient(server_url=server) as client:
        config_data = client.get_config(entry_pool_info.config_id)
    env_cfg = MettaGridConfig.model_validate(config_data)

    if setup_script:
        import subprocess
        import sys
        from pathlib import Path

        script_path = Path(setup_script)
        if not script_path.exists():
            console.print(f"[red]Setup script not found: {setup_script}[/red]")
            raise typer.Exit(1)
        console.print(f"[yellow]Running setup script: {setup_script}[/yellow]")
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            console.print(f"[red]Setup script failed:[/red]\n{result.stderr}")
            raise typer.Exit(1)
        console.print("[green]Setup script completed[/green]")

    policy_spec = get_policy_spec(ctx, policy)
    validate_policy_spec(policy_spec, env_cfg)
    console.print("[green]Policy validated successfully[/green]")
    raise typer.Exit(0)


def _parse_init_kwarg(value: str) -> tuple[str, str]:
    """Parse a key=value string into a tuple."""
    if "=" not in value:
        raise typer.BadParameter(f"Expected key=value format, got: {value}")
    key, _, val = value.partition("=")
    return key.replace("-", "_"), val


@app.command(
    name="upload",
    help="Upload a policy to CoGames",
    rich_help_panel="Tournament",
    epilog="""[dim]Examples:[/dim]

[cyan]cogames upload -p ./train_dir/my_run -n my-policy[/cyan]       Upload and submit to default season

[cyan]cogames upload -p ./run -n my-policy --season beta-cvc[/cyan]  Upload and submit to specific season

[cyan]cogames upload -p ./run -n my-policy --no-submit[/cyan]        Upload without submitting

[cyan]cogames upload -p lstm -n my-lstm --dry-run[/cyan]             Validate only""",
    add_help_option=False,
)
def upload_cmd(
    ctx: typer.Context,
    # --- Upload ---
    name: str = typer.Option(
        ...,
        "--name",
        "-n",
        metavar="NAME",
        help="Name for your uploaded policy",
        rich_help_panel="Upload",
    ),
    # --- Policy ---
    policy: str = typer.Option(
        ...,
        "--policy",
        "-p",
        metavar="POLICY",
        help=f"Policy specification: {policy_arg_example}",
        rich_help_panel="Policy",
    ),
    init_kwarg: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--init-kwarg",
        "-k",
        metavar="KEY=VAL",
        help="Policy init kwargs (can be repeated)",
        rich_help_panel="Policy",
    ),
    # --- Files ---
    include_files: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--include-files",
        "-f",
        metavar="PATH",
        help="Files or directories to include (can be repeated)",
        rich_help_panel="Files",
    ),
    setup_script: Optional[str] = typer.Option(
        None,
        "--setup-script",
        metavar="PATH",
        help="Python setup script to run before loading the policy",
        rich_help_panel="Files",
    ),
    # --- Tournament ---
    season: Optional[str] = typer.Option(
        None,
        "--season",
        metavar="SEASON",
        help="Tournament season (default: server's default season)",
        rich_help_panel="Tournament",
    ),
    no_submit: bool = typer.Option(
        False,
        "--no-submit",
        help="Upload without submitting to a season",
        rich_help_panel="Tournament",
    ),
    # --- Validation ---
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run validation only without uploading",
        rich_help_panel="Validation",
    ),
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help="Skip policy validation in isolated environment",
        rich_help_panel="Validation",
    ),
    # --- Server ---
    login_server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--login-server",
        metavar="URL",
        help="Authentication server URL",
        rich_help_panel="Server",
    ),
    server: str = typer.Option(
        DEFAULT_SUBMIT_SERVER,
        "--server",
        metavar="URL",
        help="Tournament server URL",
        rich_help_panel="Server",
    ),
    # --- Help ---
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    season_info = _resolve_season(server, season)

    has_entry_config = any(p.config_id for p in season_info.pools if p.name == season_info.entry_pool)
    if not has_entry_config and not skip_validation:
        console.print("[yellow]Warning: No entry config found for season. Skipping validation.[/yellow]")
        skip_validation = True

    init_kwargs: dict[str, str] = {}
    if init_kwarg:
        for kv in init_kwarg:
            key, val = _parse_init_kwarg(kv)
            init_kwargs[key] = val

    result = upload_policy(
        ctx=ctx,
        policy=policy,
        name=name,
        include_files=include_files,
        login_server=login_server,
        server=server,
        dry_run=dry_run,
        skip_validation=skip_validation,
        init_kwargs=init_kwargs if init_kwargs else None,
        setup_script=setup_script,
        validation_season=season_info.name,
        season=season_info.name if not no_submit else None,
    )

    if result:
        console.print(f"[green]Upload complete: {result.name}:v{result.version}[/green]")
        if result.pools:
            console.print(f"[dim]Added to pools: {', '.join(result.pools)}[/dim]")
            console.print(f"[dim]Results:[/dim] {results_url_for_season(server, season_info.name)}")
        elif no_submit:
            console.print(f"\nTo submit to a tournament: cogames submit {result.name}:v{result.version}")


@app.command(
    name="submit",
    help="Submit a policy to a tournament season",
    rich_help_panel="Tournament",
    epilog="""[dim]Examples:[/dim]

[cyan]cogames submit my-policy[/cyan]                                   Submit to default season

[cyan]cogames submit my-policy:v3 --season beta-cvc[/cyan]              Submit specific version to specific season""",
    add_help_option=False,
)
def submit_cmd(
    policy_name: str = typer.Argument(
        ...,
        metavar="POLICY",
        help="Policy name (e.g., 'my-policy' or 'my-policy:v3' for specific version)",
    ),
    season: Optional[str] = typer.Option(
        None,
        "--season",
        metavar="SEASON",
        help="Tournament season name",
        rich_help_panel="Tournament",
    ),
    login_server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--login-server",
        metavar="URL",
        help="Authentication server URL",
        rich_help_panel="Server",
    ),
    server: str = typer.Option(
        DEFAULT_SUBMIT_SERVER,
        "--server",
        "-s",
        metavar="URL",
        help="Tournament server URL",
        rich_help_panel="Server",
    ),
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    import httpx

    season_info = _resolve_season(server, season)
    season_name = season_info.name

    client = TournamentServerClient.from_login(server_url=server, login_server=login_server)
    if not client:
        raise typer.Exit(1)

    try:
        name, version = parse_policy_identifier(policy_name)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None

    version_str = f"[dim]:v{version}[/dim]" if version is not None else "[dim] (latest)[/dim]"
    console.print(f"[bold]Submitting {name}[/bold]{version_str} to season '{season_name}'\n")

    with client:
        pv = client.lookup_policy_version(name=name, version=version)
        if pv is None:
            version_hint = f" v{version}" if version is not None else ""
            console.print(f"[red]Policy '{name}'{version_hint} not found.[/red]")
            console.print("\nDid you upload it first? Use: [cyan]cogames upload[/cyan]")
            raise typer.Exit(1)

        try:
            result = client.submit_to_season(season_name, pv.id)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                console.print(f"[red]Season '{season_name}' not found[/red]")
            elif exc.response.status_code == 409:
                console.print(f"[red]Policy already submitted to season '{season_name}'[/red]")
            else:
                console.print(f"[red]Submit failed with status {exc.response.status_code}[/red]")
                console.print(f"[dim]{exc.response.text}[/dim]")
            raise typer.Exit(1) from exc
        except httpx.HTTPError as exc:
            console.print(f"[red]Submit failed:[/red] {exc}")
            raise typer.Exit(1) from exc

    console.print(f"\n[bold green]Submitted to season '{season_name}'[/bold green]")
    if result.pools:
        console.print(f"[dim]Added to pools: {', '.join(result.pools)}[/dim]")
    console.print(f"[dim]Results:[/dim] {results_url_for_season(server, season_name)}")
    console.print(f"[dim]CLI:[/dim] cogames leaderboard --season {season_name}")


@app.command(
    name="docs",
    help="Print documentation (run without arguments to see available docs)",
    rich_help_panel="Info",
    epilog="""[dim]Examples:[/dim]

  [cyan]cogames docs[/cyan]                             List available documents

  [cyan]cogames docs readme[/cyan]                      Print README

  [cyan]cogames docs mission[/cyan]                     Print mission briefing""",
    add_help_option=False,
)
def docs_cmd(
    doc_name: Optional[str] = typer.Argument(
        None,
        metavar="DOC",
        help="Document name (readme, mission, technical_manual, scripted_agent, evals, mapgen)",
    ),
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit",
        is_eager=True,
        callback=_help_callback,
    ),
) -> None:
    # Hardcoded mapping of document names to file paths and descriptions
    package_root = Path(__file__).parent.parent.parent
    docs_map: dict[str, tuple[Path, str]] = {
        "readme": (package_root / "README.md", "CoGames overview and documentation"),
        "mission": (package_root / "MISSION.md", "Mission briefing for CogsGuard Deployment"),
        "technical_manual": (package_root / "TECHNICAL_MANUAL.md", "Technical manual for Cogames"),
        "scripted_agent": (
            Path(__file__).parent / "docs" / "SCRIPTED_AGENT.md",
            "Scripted agent policy documentation",
        ),
        "evals": (
            Path(__file__).parent / "cogs_vs_clips" / "evals" / "README.md",
            "Evaluation missions documentation",
        ),
        "mapgen": (
            Path(__file__).parent / "cogs_vs_clips" / "cogs_vs_clips_mapgen.md",
            "Cogs vs Clips map generation documentation",
        ),
    }

    # If no argument provided, show available documents
    if doc_name is None:
        from rich.table import Table

        console.print("\n[bold cyan]Available Documents:[/bold cyan]\n")
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED, padding=(0, 1))
        table.add_column("Document", style="blue", no_wrap=True)
        table.add_column("Description", style="white")

        for name, (_, description) in sorted(docs_map.items()):
            table.add_row(name, description)

        console.print(table)
        console.print("\nUsage: [bold]cogames docs <document_name>[/bold]")
        console.print("Example: [bold]cogames docs mission[/bold]")
        return

    if doc_name not in docs_map:
        available = ", ".join(sorted(docs_map.keys()))
        console.print(f"[red]Error: Unknown document '{doc_name}'[/red]")
        console.print(f"\nAvailable documents: {available}")
        raise typer.Exit(1)

    doc_path, _ = docs_map[doc_name]

    if not doc_path.exists():
        console.print(f"[red]Error: Document file not found: {doc_path}[/red]")
        raise typer.Exit(1)

    try:
        content = doc_path.read_text()
        console.print(content)
    except Exception as exc:
        console.print(f"[red]Error reading document: {exc}[/red]")
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
