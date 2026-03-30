#!/usr/bin/env -S uv run
# need this to import and call suppress_noisy_logs first
# ruff: noqa: E402

"""CLI for CoGames - collection of environments for multi-agent cooperative and competitive games."""

from cogames.cli.utils import suppress_noisy_logs

suppress_noisy_logs()

import importlib
import importlib.metadata
import importlib.util
import logging
import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Literal, Optional

import httpx
import typer
import yaml  # type: ignore[import]
from click.core import ParameterSource
from packaging.version import Version
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

import cogames.policy.starter_agent as starter_agent
import cogames.policy.trainable_policy_template as trainable_policy_template
from cogames import diagnose as diagnose_module
from cogames import evaluate as evaluate_module
from cogames import pickup as pickup_module
from cogames import play as play_module
from cogames import train as train_module
from cogames import verbose
from cogames.auth import DEFAULT_COGAMES_SERVER, load_token
from cogames.cli.assay import assay_app
from cogames.cli.auth import auth_app
from cogames.cli.base import console, emit_json
from cogames.cli.client import SeasonDetail, TournamentServerClient
from cogames.cli.episode import episode_app
from cogames.cli.leaderboard import (
    leaderboard_cmd,
    parse_policy_identifier,
    submissions_cmd,
)
from cogames.cli.matches import match_artifacts_cmd, matches_cmd
from cogames.cli.mission import (
    describe_mission,
    get_mission_name_and_config,
    get_mission_names_and_configs,
    list_evals,
    list_missions,
    list_variants,
    print_mission_dependencies,
    print_variant_graph,
    save_mission_config,
)
from cogames.cli.policy import (
    _translate_error,
    get_policy_spec,
    get_policy_specs_with_proportions,
    parse_policy_spec,
    policy_arg_example,
    policy_arg_w_proportion_example,
)
from cogames.cli.season import season_app
from cogames.cli.submit import (
    DEFAULT_EPISODE_RUNNER_IMAGE,
    DEFAULT_SUBMIT_SERVER,
    RESULTS_URL,
    create_bundle,
    ensure_docker_daemon_access,
    observatory_profile_url,
    upload_policy,
    validate_bundle_docker,
)
from cogames.device import resolve_training_device
from cogames.display_detect import has_display
from cogames.games.cogs_vs_clips.train.curricula import make_rotation
from cogames.seed import seed_rollout_rng
from cogames.token_storage import TokenKind
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.policy.loader import discover_and_register_policies
from mettagrid.policy.policy_registry import get_policy_registry
from mettagrid.simulator import Simulator

# Always add current directory to Python path so optional plugins in the repo are discoverable.
sys.path.insert(0, ".")

try:  # Optional plugin
    from tribal_village_env.cogames import register_cli as register_tribal_cli  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - plugin optional
    register_tribal_cli = None


logger = logging.getLogger("cogames.main")


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
    help="CoGames - Multi-agent cooperative and competitive games.",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    callback=_register_policies,
)

tutorial_app = typer.Typer(
    help="Tutorial commands to help you get started with CoGames.",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    rich_markup_mode="rich",
)

if register_tribal_cli is not None:
    register_tribal_cli(app)


@app.command(
    name="docsync",
    hidden=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def docsync_cmd(ctx: typer.Context) -> None:
    """Sync cogames docs between .ipynb, .py, and .md formats (dev-only)."""
    from cogames.cli.docsync import docsync  # noqa: PLC0415

    docsync.app(prog_name="cogames docsync", standalone_mode=False, args=list(ctx.args))


@tutorial_app.command(
    name="play", help="Interactive tutorial - learn to play Cogs vs Clips.", rich_help_panel="Tutorial"
)
def tutorial_cmd(
    ctx: typer.Context,
) -> None:
    """Run the CoGames tutorial."""
    if not has_display():
        console.print("[red]Error: This command requires a GUI display.[/red]")
        raise typer.Exit(1)

    # Suppress logs during tutorial to keep output focused.
    logging.getLogger().setLevel(logging.ERROR)
    os.environ["METTASCOPE_SHOW_VALIDATION"] = "0"

    console.print(
        Panel.fit(
            "[bold cyan]CoGames Tutorial[/bold cyan]\n\n"
            "Mission: stabilize Machina by gathering resources, crafting hearts, and securing junctions.\n"
            "Guidance appears in-game. Press Enter or click Next to advance each tutorial phase.",
            title="Tutorial Briefing",
            border_style="green",
        )
    )

    Prompt.ask("[dim]Press Enter to launch tutorial[/dim]", default="", show_default=False)
    console.print("[dim]Initializing Mettascope...[/dim]")

    # Load tutorial mission (CvC)
    from cogames.games.cogs_vs_clips.missions.machina_1 import make_machina1_mission  # noqa: PLC0415

    # Create environment config
    env_cfg = make_machina1_mission(num_agents=1, max_steps=1000).make_env()
    console.print("[dim]Tutorial phases appear in-game. Press Enter or click Next to advance.[/dim]")
    console.print("[dim]Close the Mettascope window to exit.[/dim]")

    # Run play (blocks main thread)
    try:
        play_module.play(
            console,
            env_cfg=env_cfg,
            policy_spec=get_policy_spec(ctx, "class=tutorial_noop,kw.tutorial=play"),
            game_name="tutorial",
            render_mode="gui",
            autostart=False,
        )
    except KeyboardInterrupt:
        logger.info("Tutorial interrupted; exiting.")


@tutorial_app.command(
    name="cvc",
    help="Interactive CvC tutorial - learn roles and territory control.",
    rich_help_panel="Tutorial",
)
def cvc_tutorial_cmd(
    ctx: typer.Context,
) -> None:
    """Run the CvC tutorial."""
    if not has_display():
        console.print("[red]Error: This command requires a GUI display.[/red]")
        raise typer.Exit(1)

    # Suppress logs during tutorial to keep output focused.
    logging.getLogger().setLevel(logging.ERROR)
    os.environ["METTASCOPE_SHOW_VALIDATION"] = "0"

    console.print(
        Panel.fit(
            "[bold cyan]CvC Tutorial[/bold cyan]\n\n"
            "Mission: outscore Clips by sustaining junction control under pressure.\n"
            "Guidance appears in-game. Press Enter or click Next to advance each tutorial phase.",
            title="Tutorial Briefing",
            border_style="green",
        )
    )

    Prompt.ask("[dim]Press Enter to launch tutorial[/dim]", default="", show_default=False)
    console.print("[dim]Initializing Mettascope...[/dim]")

    # Load CvC tutorial mission
    from cogames.games.cogs_vs_clips.missions.tutorial import make_tutorial_mission  # noqa: PLC0415

    env_cfg = make_tutorial_mission().make_env()
    console.print("[dim]Tutorial phases appear in-game. Press Enter or click Next to advance.[/dim]")
    console.print("[dim]Close the Mettascope window to exit.[/dim]")

    # Run play (blocks main thread)
    try:
        play_module.play(
            console,
            env_cfg=env_cfg,
            policy_spec=get_policy_spec(ctx, "class=tutorial_noop,kw.tutorial=cvc"),
            game_name="tutorial",
            render_mode="gui",
            autostart=False,
        )
    except KeyboardInterrupt:
        logger.info("CvC tutorial interrupted; exiting.")


app.add_typer(tutorial_app, name="tutorial", rich_help_panel="Tutorials")
app.add_typer(auth_app, name="auth", rich_help_panel="Tournament")
app.add_typer(season_app, name="season", rich_help_panel="Tournament")
app.add_typer(episode_app, name="episode", rich_help_panel="Tournament")
app.add_typer(assay_app, name="assay", rich_help_panel="Tournament")


def _help_callback(ctx: typer.Context, value: bool) -> None:
    """Callback for custom help option."""
    if value:
        console.print(ctx.get_help())
        raise typer.Exit()


@app.command(
    name="missions",
    help="""List available missions.

This command has two modes:

[bold]1. List missions:[/bold] Run with no arguments to see all available missions.

[bold]2. Describe a mission:[/bold] Use -m to describe a specific mission. Only in this mode do \
--cogs, --variant, --format, and --save have any effect.""",
    rich_help_panel="Missions",
    epilog="""[dim]Examples:[/dim]

  [cyan]cogames missions[/cyan]                                    List all missions

  [cyan]cogames missions -m machina_1[/cyan]                       Describe a mission

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
        metavar="FILTER",
        help="Filter missions by name.",
    ),
    # --- Describe (requires -m) ---
    mission: Optional[str] = typer.Option(
        None,
        "--mission",
        "-m",
        metavar="MISSION",
        help="Mission to describe.",
        rich_help_panel="Describe",
    ),
    cogs: Optional[int] = typer.Option(
        None,
        "--cogs",
        "-c",
        help="Override agent count (requires -m).",
        rich_help_panel="Describe",
    ),
    variant: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--variant",
        "-v",
        metavar="VARIANT",
        help="Apply variant (requires -m, repeatable).",
        rich_help_panel="Describe",
    ),
    format_: Optional[Literal["yaml", "json"]] = typer.Option(
        None,
        "--format",
        help="Output format (requires -m).",
        rich_help_panel="Describe",
    ),
    save: Optional[Path] = typer.Option(  # noqa: B008
        None,
        "--save",
        "-s",
        metavar="PATH",
        help="Save config to file (requires -m).",
        rich_help_panel="Describe",
    ),
    # --- Dependencies ---
    dependencies: bool = typer.Option(
        False,
        "--dependencies",
        help="Show variant dependencies for each mission.",
    ),
    # --- Debug ---
    print_cvc_config: bool = typer.Option(
        False,
        "--print-cvc-config",
        help="Print CVC mission config (requires -m).",
        hidden=True,
    ),
    print_mg_config: bool = typer.Option(
        False,
        "--print-mg-config",
        help="Print MettaGrid config (requires -m).",
        hidden=True,
    ),
    # --- Help ---
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit.",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    if mission is None:
        if dependencies:
            from cogames.game import get_game  # noqa: PLC0415

            print_mission_dependencies(get_game("cogs_vs_clips"), console)
        else:
            list_missions(site)
        return

    try:
        resolved_mission, env_cfg, mission_cfg = get_mission_name_and_config(
            ctx,
            mission,
            variants_arg=variant,
            cogs=cogs,
        )
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
            save_mission_config(env_cfg, save)
            console.print(f"[green]Mission configuration saved to: {save}[/green]")
        except ValueError as exc:  # pragma: no cover - user input
            console.print(f"[red]Error saving configuration: {exc}[/red]")
            raise typer.Exit(1) from exc
        return

    if format_ is not None:
        try:
            data = env_cfg.model_dump(mode="json")
            if format_ == "json":
                emit_json(data)
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


@app.command("evals", help="List all eval missions.", rich_help_panel="Missions")
def evals_cmd() -> None:
    list_evals()


@app.command("variants", help="List all available mission variants.", rich_help_panel="Missions")
def variants_cmd(
    dependencies: bool = typer.Option(False, "--dependencies", help="Print the variant dependency graph."),
) -> None:
    if dependencies:
        from cogames.game import get_game  # noqa: PLC0415

        print_variant_graph(get_game("cogs_vs_clips"), console)
    else:
        list_variants()


@app.command(
    name="describe",
    help="Describe a mission and its configuration.",
    rich_help_panel="Missions",
    epilog="""[dim]Examples:[/dim]

  [cyan]cogames describe machina_1.basic[/cyan]             Describe mission

  [cyan]cogames describe arena -c 4 -v dark_side[/cyan]               With 4 cogs and variant""",
    add_help_option=False,
)
def describe_cmd(
    ctx: typer.Context,
    mission: str = typer.Argument(
        ...,
        metavar="MISSION",
        help="Mission name (e.g., machina_1.basic).",
    ),
    cogs: Optional[int] = typer.Option(
        None,
        "--cogs",
        "-c",
        help="Number of cogs (agents).",
        rich_help_panel="Configuration",
    ),
    variant: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--variant",
        "-v",
        metavar="VARIANT",
        help="Apply variant (repeatable).",
        rich_help_panel="Configuration",
    ),
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit.",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    resolved_mission, env_cfg, mission_cfg = get_mission_name_and_config(
        ctx,
        mission,
        variants_arg=variant,
        cogs=cogs,
    )
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

[cyan]cogames play -m machina_1.basic[/cyan]                        Interactive

[cyan]cogames play -m machina_1.basic -p class=random[/cyan]        Random policy

[cyan]cogames play -m machina_1.basic -c 4 -p class=baseline[/cyan] Baseline, 4 cogs

[cyan]cogames play -m machina_1.basic --save-replay-file ./latest.json.z[/cyan] Overwrite fixed replay file

[cyan]cogames play -m machina_1 -r unicode[/cyan]                   Terminal mode""",
    add_help_option=False,
)
def play_cmd(
    ctx: typer.Context,
    # --- Game Setup ---
    game: str = typer.Option(
        "cogs_vs_clips",
        "--game",
        metavar="GAME",
        help="Game to play (default: cogs_vs_clips).",
        rich_help_panel="Game Setup",
    ),
    mission: Optional[str] = typer.Option(
        None,
        "--mission",
        "-m",
        metavar="MISSION",
        help="Mission to play (run [bold]cogames missions[/bold] to list).",
        rich_help_panel="Game Setup",
    ),
    variant: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--variant",
        "-v",
        metavar="VARIANT",
        help="Apply variant modifier (repeatable).",
        rich_help_panel="Game Setup",
    ),
    cogs: Optional[int] = typer.Option(
        None,
        "--cogs",
        "-c",
        metavar="N",
        help="Number of cogs/agents.",
        show_default="from mission",
        rich_help_panel="Game Setup",
    ),
    # --- Policy ---
    policy: str = typer.Option(
        "class=noop",
        "--policy",
        "-p",
        metavar="POLICY",
        help="Policy controlling cogs ([bold]noop[/bold], [bold]random[/bold], [bold]lstm[/bold], or path).",
        rich_help_panel="Policy",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        metavar="DEVICE",
        help="Policy device (auto, cpu, cuda, cuda:0, etc.).",
        rich_help_panel="Policy",
    ),
    # --- Simulation ---
    steps: int = typer.Option(
        1000,
        "--steps",
        "-s",
        metavar="N",
        help="Max steps per episode (note: -s is steps, not seed).",
        rich_help_panel="Simulation",
    ),
    action_timeout_ms: int = typer.Option(
        10000,
        "--action-timeout-ms",
        metavar="MS",
        help="Max ms per action before noop.",
        min=1,
        rich_help_panel="Simulation",
    ),
    render: Literal["auto", "gui", "unicode", "log", "none"] = typer.Option(  # noqa: B008
        "auto",
        "--render",
        "-r",
        help=(
            "[bold]auto[/bold]=gui when display is available, otherwise unicode; "
            "[bold]gui[/bold]=MettaScope, "
            "[bold]unicode[/bold]=terminal, [bold]log[/bold]=metrics only."
        ),
        rich_help_panel="Simulation",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="RNG seed for reproducibility (use --seed, not -s).",
        rich_help_panel="Simulation",
    ),
    map_seed: Optional[int] = typer.Option(
        None,
        "--map-seed",
        metavar="SEED",
        help="Separate seed for procedural map generation.",
        show_default="same as --seed",
        rich_help_panel="Simulation",
    ),
    autostart: bool = typer.Option(
        False,
        "--autostart",
        help="Start simulation immediately without waiting for user input.",
        rich_help_panel="Simulation",
    ),
    # --- Output ---
    save_replay_dir: Optional[Path] = typer.Option(  # noqa: B008
        None,
        "--save-replay-dir",
        metavar="DIR",
        help="Save replay file for later viewing with [bold]cogames replay[/bold].",
        rich_help_panel="Output",
    ),
    save_replay_file: Optional[Path] = typer.Option(  # noqa: B008
        None,
        "--save-replay-file",
        metavar="FILE",
        help="Save replay to a fixed file path (overwrites existing file)",
        rich_help_panel="Output",
    ),
    # --- Debug (hidden from casual users) ---
    print_cvc_config: bool = typer.Option(
        False,
        "--print-cvc-config",
        help="Print mission config and exit.",
        rich_help_panel="Debug",
        hidden=True,
    ),
    print_mg_config: bool = typer.Option(
        False,
        "--print-mg-config",
        help="Print MettaGrid config and exit.",
        rich_help_panel="Debug",
        hidden=True,
    ),
    # --- Help at end ---
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit.",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    if save_replay_dir is not None and save_replay_file is not None:
        console.print("[red]Error: Use only one of --save-replay-dir or --save-replay-file.[/red]")
        raise typer.Exit(1)

    display_available = has_display()
    if render == "auto":
        render = "gui" if display_available else "unicode"
    if render == "gui" and not display_available:
        console.print("[red]Error: This render mode requires a GUI display.[/red]")
        raise typer.Exit(1)

    _explicit_steps = ctx.get_parameter_source("steps") in (
        ParameterSource.COMMANDLINE,
        ParameterSource.ENVIRONMENT,
        ParameterSource.PROMPT,
    )

    resolved_mission, env_cfg, mission_cfg = get_mission_name_and_config(
        ctx,
        mission,
        game_name=game,
        variants_arg=variant,
        cogs=cogs,
        steps=steps if _explicit_steps else None,
    )

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

    seed_rollout_rng(seed)
    resolved_device = resolve_training_device(console, device)
    policy_spec = get_policy_spec(ctx, policy, device=str(resolved_device))

    console.print(f"[cyan]Playing {resolved_mission}[/cyan]")
    console.print(f"Max Steps: {env_cfg.game.max_steps}, Render: {render}")

    play_module.play(
        console,
        env_cfg=env_cfg,
        policy_spec=policy_spec,
        seed=seed,
        device=str(resolved_device),
        render_mode=render,
        action_timeout_ms=action_timeout_ms,
        game_name=resolved_mission,
        save_replay=save_replay_dir,
        save_replay_file=save_replay_file,
        autostart=autostart,
    )


@app.command(
    name="replay",
    help="Replay a saved game episode from a file in the GUI.",
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
        help="Path to the replay file (.replay or .bin).",
    ),
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit.",
        is_eager=True,
        callback=_help_callback,
    ),
) -> None:
    if not replay_path.exists():
        console.print(f"[red]Error: Replay file not found: {replay_path}[/red]")
        raise typer.Exit(1)

    if not has_display():
        console.print("[red]Error: This command requires a GUI display.[/red]")
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
    help="Create a custom mission from a base template.",
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
        help="Base mission to start from.",
        rich_help_panel="Mission",
    ),
    # --- Customization ---
    cogs: Optional[int] = typer.Option(
        None,
        "--cogs",
        "-c",
        help="Number of cogs (agents).",
        min=1,
        rich_help_panel="Customization",
    ),
    width: Optional[int] = typer.Option(
        None,
        "--width",
        help="Map width.",
        min=1,
        rich_help_panel="Customization",
    ),
    height: Optional[int] = typer.Option(
        None,
        "--height",
        help="Map height.",
        min=1,
        rich_help_panel="Customization",
    ),
    # --- Output ---
    output: Optional[Path] = typer.Option(  # noqa: B008
        None,
        "--output",
        "-o",
        metavar="PATH",
        help="Output file path (.yml or .json).",
        rich_help_panel="Output",
    ),
    # --- Help ---
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit.",
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
            save_mission_config(env_cfg, output)
            console.print(f"[green]Modified {resolved_mission} configuration saved to: {output}[/green]")
        else:
            console.print("\n[yellow]To save this configuration, use the --output option.[/yellow]")

    except Exception as exc:  # pragma: no cover - user input
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc


# TODO: Verify make-policy templates work with CvC game mechanics
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
        help="Create a trainable (neural network) policy.",
        rich_help_panel="Policy Type",
    ),
    scripted: bool = typer.Option(
        False,
        "--scripted",
        help="Create a scripted (rule-based) policy.",
        rich_help_panel="Policy Type",
    ),
    # --- Output ---
    output: Path = typer.Option(  # noqa: B008
        "my_policy.py",
        "--output",
        "-o",
        metavar="FILE",
        help="Output file path.",
        rich_help_panel="Output",
    ),
    # --- Help ---
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit.",
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
                "[dim]Train with: cogames tutorial train -m machina_1.basic -p class="
                f"{dest_path.stem}.{policy_class}[/dim]"
            )
        else:
            console.print(
                f"[dim]Play with: cogames play -m machina_1.basic -p class={dest_path.stem}.{policy_class}[/dim]"
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

[cyan]cogames tutorial train -m machina_1.basic[/cyan]                   Basic training

[cyan]cogames tutorial train -m machina_1.basic -p class=baseline[/cyan]
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
        help="Missions to train on (wildcards supported, repeatable for curriculum).",
        rich_help_panel="Mission Setup",
    ),
    cogs: Optional[int] = typer.Option(
        None,
        "--cogs",
        "-c",
        metavar="N",
        help="Number of cogs (agents).",
        show_default="from mission",
        rich_help_panel="Mission Setup",
    ),
    variant: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--variant",
        "-v",
        metavar="VARIANT",
        help="Mission variant (repeatable).",
        rich_help_panel="Mission Setup",
    ),
    # --- Policy ---
    policy: str = typer.Option(
        "class=lstm",
        "--policy",
        "-p",
        metavar="POLICY",
        help=f"Policy to train ({policy_arg_example}).",
        rich_help_panel="Policy",
    ),
    # --- Training ---
    steps: int = typer.Option(
        10_000_000_000,
        "--steps",
        metavar="N",
        help="Number of training steps.",
        min=1,
        rich_help_panel="Training",
    ),
    minibatch_size: int = typer.Option(
        4096,
        "--minibatch-size",
        metavar="N",
        help="Minibatch size for training.",
        min=1,
        rich_help_panel="Training",
    ),
    # --- Hardware ---
    device: str = typer.Option(
        "auto",
        "--device",
        metavar="DEVICE",
        help="Device to train on (auto, cpu, cuda, mps).",
        rich_help_panel="Hardware",
    ),
    num_workers: Optional[int] = typer.Option(
        None,
        "--num-workers",
        metavar="N",
        help="Number of worker processes.",
        show_default="CPU cores",
        min=1,
        rich_help_panel="Hardware",
    ),
    parallel_envs: Optional[int] = typer.Option(
        None,
        "--parallel-envs",
        metavar="N",
        help="Number of parallel environments.",
        min=1,
        rich_help_panel="Hardware",
    ),
    vector_batch_size: Optional[int] = typer.Option(
        None,
        "--vector-batch-size",
        metavar="N",
        help="Vectorized environment batch size.",
        min=1,
        rich_help_panel="Hardware",
    ),
    # --- Reproducibility ---
    seed: int = typer.Option(
        42,
        "--seed",
        metavar="N",
        help="Seed for training RNG.",
        min=0,
        rich_help_panel="Reproducibility",
    ),
    map_seed: Optional[int] = typer.Option(
        None,
        "--map-seed",
        metavar="N",
        help="MapGen seed for procedural map layout.",
        show_default="same as --seed",
        min=0,
        rich_help_panel="Reproducibility",
    ),
    # --- Output ---
    checkpoints_path: str = typer.Option(
        "./train_dir",
        "--checkpoints",
        metavar="DIR",
        help="Path to save training checkpoints.",
        rich_help_panel="Output",
    ),
    log_outputs: bool = typer.Option(
        False,
        "--log-outputs",
        help="Log training outputs.",
        rich_help_panel="Output",
    ),
    # --- Help ---
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit.",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    selected_missions = get_mission_names_and_configs(
        ctx,
        missions,
        variants_arg=variant,
        cogs=cogs,
    )
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

[cyan]cogames run -m machina_1.basic -p lstm[/cyan]               Evaluate single policy

[cyan]cogames run -m machina_1 -p ./train_dir/my_run:v5[/cyan]     Evaluate a checkpoint bundle

[cyan]cogames run -m 'arena.*' -p lstm -p random -e 20[/cyan]            Evaluate multiple policies together

[cyan]cogames run -m machina_1 -p ./train_dir/my_run:v5,proportion=3 -p class=random,proportion=5[/cyan]
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
        help="Missions to evaluate (supports wildcards).",
        rich_help_panel="Mission",
    ),
    cogs: Optional[int] = typer.Option(
        None,
        "--cogs",
        "-c",
        metavar="N",
        help="Number of cogs (agents).",
        rich_help_panel="Mission",
    ),
    variant: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--variant",
        "-v",
        metavar="VARIANT",
        help="Mission variant (repeatable).",
        rich_help_panel="Mission",
    ),
    # --- Policy ---
    policies: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--policy",
        "-p",
        metavar="POLICY",
        help=f"Policies to evaluate: ({policy_arg_w_proportion_example}...).",
        rich_help_panel="Policy",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        metavar="DEVICE",
        help="Policy device (auto, cpu, cuda, cuda:0, etc.).",
        rich_help_panel="Policy",
    ),
    # --- Simulation ---
    episodes: int = typer.Option(
        10,
        "--episodes",
        "-e",
        metavar="N",
        help="Number of evaluation episodes.",
        min=1,
        rich_help_panel="Simulation",
    ),
    steps: Optional[int] = typer.Option(
        None,
        "--steps",
        "-s",
        metavar="N",
        help="Max steps per episode (note: -s is steps, not seed).",
        min=1,
        show_default="from mission",
        rich_help_panel="Simulation",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        metavar="N",
        help="Seed for evaluation RNG (use --seed, not -s).",
        min=0,
        rich_help_panel="Simulation",
    ),
    map_seed: Optional[int] = typer.Option(
        None,
        "--map-seed",
        metavar="N",
        help="MapGen seed for procedural maps.",
        min=0,
        show_default="same as --seed",
        rich_help_panel="Simulation",
    ),
    action_timeout_ms: int = typer.Option(
        250,
        "--action-timeout-ms",
        metavar="MS",
        help="Max ms per action before noop.",
        min=1,
        rich_help_panel="Simulation",
    ),
    # --- Output ---
    format_: Optional[Literal["yaml", "json"]] = typer.Option(
        None,
        "--format",
        metavar="FMT",
        help="Output format: yaml or json.",
        rich_help_panel="Output",
    ),
    save_replay_dir: Optional[Path] = typer.Option(  # noqa: B008
        None,
        "--save-replay-dir",
        metavar="DIR",
        help="Directory to save replays.",
        rich_help_panel="Output",
    ),
    # --- Help ---
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit.",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    # When structured output is requested, redirect all status messages to stderr
    # so only clean JSON/YAML appears on stdout.
    out = Console(stderr=True) if format_ else console

    selected_missions = get_mission_names_and_configs(
        ctx,
        missions,
        variants_arg=variant,
        cogs=cogs,
        steps=steps,
    )

    # Optional MapGen seed override for procedural maps.
    if map_seed is not None:
        for _, env_cfg in selected_missions:
            map_builder = getattr(env_cfg.game, "map_builder", None)
            if isinstance(map_builder, MapGen.Config):
                map_builder.seed = map_seed

    resolved_device = resolve_training_device(out, device)
    policy_specs = get_policy_specs_with_proportions(ctx, policies, device=str(resolved_device))

    if ctx.info_name == "scrimmage":
        if len(policy_specs) != 1:
            out.print("[red]Error: scrimmage accepts exactly one --policy / -p value.[/red]")
            raise typer.Exit(1)
        if policy_specs[0].proportion != 1.0:
            out.print("[red]Error: scrimmage does not support policy proportions.[/red]")
            raise typer.Exit(1)

    out.print(
        f"[cyan]Preparing evaluation for {len(policy_specs)} policies across {len(selected_missions)} mission(s)[/cyan]"
    )

    evaluate_module.evaluate(
        out,
        missions=selected_missions,
        policy_specs=[spec.to_policy_spec() for spec in policy_specs],
        proportions=[spec.proportion for spec in policy_specs],
        action_timeout_ms=action_timeout_ms,
        episodes=episodes,
        seed=seed,
        device=str(resolved_device),
        output_format=format_,
        save_replay=str(save_replay_dir) if save_replay_dir else None,
    )


@app.command(
    name="pickup",
    help="Evaluate a policy against a pool of other policies and compute VOR.",
    rich_help_panel="Evaluate",
    epilog="""[dim]Examples:[/dim]

[cyan]cogames pickup -p greedy --pool random[/cyan]                      Test greedy against pool of random""",
    add_help_option=False,
)
def pickup_cmd(
    ctx: typer.Context,
    # --- Mission ---
    mission: str = typer.Option(
        "machina_1.basic",
        "--mission",
        "-m",
        metavar="MISSION",
        help="Mission to evaluate on.",
        rich_help_panel="Mission",
    ),
    cogs: int = typer.Option(
        4,
        "--cogs",
        "-c",
        metavar="N",
        help="Number of cogs (agents).",
        min=1,
        rich_help_panel="Mission",
    ),
    variant: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--variant",
        "-v",
        metavar="VARIANT",
        help="Mission variant (repeatable).",
        rich_help_panel="Mission",
    ),
    # --- Policy ---
    policy: Optional[str] = typer.Option(
        None,
        "--policy",
        "-p",
        metavar="POLICY",
        help="Candidate policy to evaluate.",
        rich_help_panel="Policy",
    ),
    pool: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--pool",
        metavar="POLICY",
        help="Pool policy (repeatable).",
        rich_help_panel="Policy",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        metavar="DEVICE",
        help="Policy device (auto, cpu, cuda, cuda:0, etc.).",
        rich_help_panel="Policy",
    ),
    # --- Simulation ---
    episodes: int = typer.Option(
        1,
        "--episodes",
        "-e",
        metavar="N",
        help="Episodes per scenario.",
        min=1,
        rich_help_panel="Simulation",
    ),
    steps: Optional[int] = typer.Option(
        1000,
        "--steps",
        "-s",
        metavar="N",
        help="Max steps per episode (note: -s is steps, not seed).",
        min=1,
        rich_help_panel="Simulation",
    ),
    seed: int = typer.Option(
        50,
        "--seed",
        metavar="N",
        help="Base random seed (use --seed, not -s).",
        min=0,
        rich_help_panel="Simulation",
    ),
    map_seed: Optional[int] = typer.Option(
        None,
        "--map-seed",
        metavar="N",
        help="MapGen seed for procedural maps.",
        min=0,
        show_default="same as --seed",
        rich_help_panel="Simulation",
    ),
    action_timeout_ms: int = typer.Option(
        250,
        "--action-timeout-ms",
        metavar="MS",
        help="Max ms per action before noop.",
        min=1,
        rich_help_panel="Simulation",
    ),
    # --- Output ---
    save_replay_dir: Optional[Path] = typer.Option(  # noqa: B008
        None,
        "--save-replay-dir",
        metavar="DIR",
        help="Directory to save replays.",
        rich_help_panel="Output",
    ),
    # --- Help ---
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit.",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    if policy is None:
        console.print(ctx.get_help())
        console.print("[yellow]Missing: --policy / -p[/yellow]\n")
        raise typer.Exit(1)

    if not pool:
        console.print(ctx.get_help())
        console.print("[yellow]Supply at least one: --pool[/yellow]\n")
        raise typer.Exit(1)

    # Resolve mission
    resolved_mission, env_cfg, _ = get_mission_name_and_config(
        ctx,
        mission,
        variants_arg=variant,
        cogs=cogs,
        steps=steps,
    )

    candidate_label = policy
    pool_labels = pool
    resolved_device = resolve_training_device(console, device)
    candidate_spec = get_policy_spec(ctx, policy, device=str(resolved_device))
    try:
        pool_specs = [parse_policy_spec(spec, device=str(resolved_device)).to_policy_spec() for spec in pool]
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
        device=str(resolved_device),
        candidate_label=candidate_label,
        pool_labels=pool_labels,
    )


@app.command(
    name="version",
    help="Show version information for cogames and dependencies.",
    rich_help_panel="Info",
)
def version_cmd() -> None:
    def public_version(dist_name: str) -> str:
        return str(Version(importlib.metadata.version(dist_name)).public)

    table = Table(show_header=False, box=None, show_lines=False, pad_edge=False)
    table.add_column("", justify="right", style="bold cyan")
    table.add_column("", justify="right")

    for dist_name in ["mettagrid", "cogames"]:
        table.add_row(dist_name, public_version(dist_name))

    console.print(table)


@app.command(
    name="policies",
    help="Show available policy shorthand names.",
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
    help="Sign in to cogames interactively.",
    rich_help_panel="Tournament",
    add_help_option=False,
)
def login_cmd(
    login_server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--login-server",
        metavar="URL",
        help="Authentication server URL.",
        rich_help_panel="Server",
    ),
    no_browser: bool = typer.Option(
        False,
        "--no-browser",
        help="Skip opening browser automatically.",
        rich_help_panel="Options",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Re-authenticate even if already logged in.",
        rich_help_panel="Options",
    ),
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit.",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    from cogames.cli.auth import login_cmd as auth_login  # noqa: PLC0415

    auth_login(login_server=login_server, no_browser=no_browser, force=force)


app.command(
    name="submissions",
    help="Show your uploads and tournament submissions.",
    rich_help_panel="Tournament",
    epilog="""[dim]Examples:[/dim]

[cyan]cogames submissions[/cyan]                         All your uploads

[cyan]cogames submissions --season beta-cvc[/cyan]           Submissions in a season

[cyan]cogames submissions -p my-policy[/cyan]            Info on a specific policy""",
    add_help_option=False,
)(submissions_cmd)

app.command(
    name="leaderboard",
    help="Show tournament leaderboard for a season.",
    rich_help_panel="Tournament",
    epilog="""[dim]Examples:[/dim]

[cyan]cogames leaderboard beta-cvc[/cyan]                          View rankings (positional season)

[cyan]cogames leaderboard --season beta-cvc[/cyan]                 View rankings (option)

[cyan]cogames leaderboard beta-cvc --policy slanky[/cyan]          Filter by policy name

[cyan]cogames leaderboard beta-cvc --mine[/cyan]                   Show only your policies""",
    add_help_option=False,
)(leaderboard_cmd)

app.command(
    name="matches",
    help="Show your recent matches and policy logs.",
    rich_help_panel="Tournament",
    epilog="""[dim]Examples:[/dim]

[cyan]cogames matches[/cyan]                              List recent matches

[cyan]cogames matches --policy slanky[/cyan]               Filter by policy name

[cyan]cogames matches <match-id>[/cyan]                   Show match details

[cyan]cogames matches <match-id> --logs[/cyan]            Show available logs

[cyan]cogames matches <match-id> -d ./logs[/cyan]         Download logs""",
    add_help_option=False,
)(matches_cmd)

app.command(
    name="match-artifacts",
    help="Retrieve artifacts for a match (logs, etc.).",
    rich_help_panel="Tournament",
    epilog="""[dim]Examples:[/dim]

[cyan]cogames match-artifacts <match-id>[/cyan]                     Get match logs

[cyan]cogames match-artifacts <match-id> logs -o out.txt[/cyan]     Save to file""",
    add_help_option=False,
)(match_artifacts_cmd)


app.command(
    name="diagnose",
    help="Run diagnostic evals for a policy checkpoint.",
    rich_help_panel="Evaluate",
    epilog="""[dim]Examples:[/dim]

[cyan]cogames diagnose ./train_dir/my_run[/cyan]                         Default CvC evals

[cyan]cogames diagnose lstm --scripted-baseline-policy scripted.basic[/cyan]   Compare against scripted baseline

[cyan]cogames diagnose lstm --known-strong-policy my_best_policy[/cyan]         Normalize against known-strong policy

[cyan]cogames diagnose lstm --compare-run-dir outputs/cogames-diagnose/prev_run[/cyan]  Stability comparison""",
    add_help_option=False,
)(diagnose_module.diagnose_cmd)


def _resolve_season(server: str, login_server: str | None = None, season_name: str | None = None) -> SeasonDetail:
    auth_token = load_token(token_kind=TokenKind.COGAMES, server=login_server) if login_server else None
    try:
        with TournamentServerClient(server_url=server, token=auth_token, login_server=login_server) as client:
            if season_name is None:
                season_name = client.get_default_season().name
            info = client.get_season(season_name)
            console.print(f"[dim]Using season: {info.name}[/dim]")
            return info
    except Exception as e:
        console.print(f"[red]Could not fetch season from server:[/red] {e}")
        console.print("Specify a season explicitly with [cyan]--season[/cyan]")
        raise typer.Exit(1) from None


@app.command(
    name="create-bundle",
    help="Create a submission bundle zip from a policy.",
    rich_help_panel="Policies",
    add_help_option=False,
)
def create_bundle_cmd(
    ctx: typer.Context,
    policy: str = typer.Option(
        ...,
        "--policy",
        "-p",
        metavar="POLICY",
        help=f"Policy specification: {policy_arg_example}.",
        rich_help_panel="Policy",
    ),
    output: Path = typer.Option(  # noqa: B008
        Path("submission.zip"),
        "--output",
        "-o",
        metavar="PATH",
        help="Output path for the bundle zip.",
        rich_help_panel="Output",
    ),
    init_kwarg: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--init-kwarg",
        "-k",
        metavar="KEY=VAL",
        help="Policy init kwargs (can be repeated).",
        rich_help_panel="Policy",
    ),
    include_files: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--include-files",
        "-f",
        metavar="PATH",
        help="Files or directories to include (can be repeated).",
        rich_help_panel="Files",
    ),
    setup_script: Optional[str] = typer.Option(
        None,
        "--setup-script",
        metavar="PATH",
        help="Python setup script to include in the bundle.",
        rich_help_panel="Files",
    ),
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit.",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    init_kwargs: dict[str, str] = {}
    if init_kwarg:
        for kv in init_kwarg:
            key, val = _parse_init_kwarg(kv)
            init_kwargs[key] = val

    result_path = create_bundle(
        ctx=ctx,
        policy=policy,
        output=output.resolve(),
        include_files=include_files,
        init_kwargs=init_kwargs if init_kwargs else None,
        setup_script=setup_script,
    )
    console.print(f"[green]Bundle created:[/green] {result_path}")


@app.command(
    name="validate-bundle",
    help="Validate a policy bundle runs correctly in Docker.",
    rich_help_panel="Policies",
    add_help_option=False,
)
def validate_bundle_cmd(
    policy: str = typer.Option(
        ...,
        "--policy",
        "-p",
        metavar="URI",
        help="Bundle URI (file://, s3://, or local path to .zip or directory).",
    ),
    season: Optional[str] = typer.Option(
        None,
        "--season",
        metavar="SEASON",
        help="Tournament season (determines which game to validate against).",
        rich_help_panel="Tournament",
    ),
    server: str = typer.Option(
        DEFAULT_SUBMIT_SERVER,
        "--server",
        metavar="URL",
        help="Tournament server URL (used to resolve default season).",
        rich_help_panel="Server",
    ),
    image: str = typer.Option(
        DEFAULT_EPISODE_RUNNER_IMAGE,
        "--image",
        help="Docker image for container validation.",
        rich_help_panel="Validation",
    ),
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit.",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    ensure_docker_daemon_access()

    season_info = _resolve_season(server, season_name=season)
    entry_pool_info = next((p for p in season_info.pools if p.name == season_info.entry_pool), None)
    if not entry_pool_info or not entry_pool_info.config_id:
        console.print("[red]No entry config found for season[/red]")
        raise typer.Exit(1)

    if image == DEFAULT_EPISODE_RUNNER_IMAGE and season_info.compat_version is not None:
        image = f"ghcr.io/metta-ai/episode-runner:compat-v{season_info.compat_version}"

    with TournamentServerClient(server_url=server) as client:
        config_data = client.get_config(entry_pool_info.config_id)

    validate_bundle_docker(policy, config_data, image)

    console.print("[green]Policy validated successfully[/green]")
    raise typer.Exit(0)


def _parse_init_kwarg(value: str) -> tuple[str, str]:
    """Parse a key=value string into a tuple."""
    if "=" not in value:
        raise typer.BadParameter(f"Expected key=value format, got: {value}")
    key, _, val = value.partition("=")
    return key.replace("-", "_"), val


def _parse_secret_env(value: str) -> tuple[str, str]:
    """Parse KEY=VALUE into a tuple for secret environment variables."""
    if "=" not in value:
        raise typer.BadParameter(f"Expected KEY=VALUE format, got: {value}")
    key, _, val = value.partition("=")
    return key, val


@app.command(
    name="upload",
    help="Upload a policy to CoGames.",
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
        help="Name for your uploaded policy.",
        rich_help_panel="Upload",
    ),
    # --- Policy ---
    policy: str = typer.Option(
        ...,
        "--policy",
        "-p",
        metavar="POLICY",
        help=f"Policy specification: {policy_arg_example}.",
        rich_help_panel="Policy",
    ),
    init_kwarg: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--init-kwarg",
        "-k",
        metavar="KEY=VAL",
        help="Policy init kwargs (can be repeated).",
        rich_help_panel="Policy",
    ),
    # --- Files ---
    include_files: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--include-files",
        "-f",
        metavar="PATH",
        help="Files or directories to include (can be repeated).",
        rich_help_panel="Files",
    ),
    setup_script: Optional[str] = typer.Option(
        None,
        "--setup-script",
        metavar="PATH",
        help="Python setup script to run before loading the policy.",
        rich_help_panel="Files",
    ),
    # --- Secrets ---
    secret_env: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--secret-env",
        metavar="KEY=VALUE",
        help="Secret environment variable for policy execution (can be repeated). Stored in AWS Secrets Manager.",
        rich_help_panel="Secrets",
    ),
    use_bedrock: bool = typer.Option(
        False,
        "--use-bedrock",
        help="Enable AWS Bedrock access for this policy. Sets USE_BEDROCK=true in policy environment.",
        rich_help_panel="Secrets",
    ),
    # --- Tournament ---
    season: Optional[str] = typer.Option(
        None,
        "--season",
        metavar="SEASON",
        help="Tournament season (default: server's default season).",
        rich_help_panel="Tournament",
    ),
    no_submit: bool = typer.Option(
        False,
        "--no-submit",
        help="Upload without submitting to a season.",
        rich_help_panel="Tournament",
    ),
    # --- Validation ---
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run validation only without uploading.",
        rich_help_panel="Validation",
    ),
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help="Skip policy validation in Docker.",
        rich_help_panel="Validation",
    ),
    image: str = typer.Option(
        DEFAULT_EPISODE_RUNNER_IMAGE,
        "--image",
        help="Docker image for container validation.",
        rich_help_panel="Validation",
    ),
    # --- Server ---
    login_server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--login-server",
        metavar="URL",
        help="Authentication server URL.",
        rich_help_panel="Server",
    ),
    server: str = typer.Option(
        DEFAULT_SUBMIT_SERVER,
        "--server",
        metavar="URL",
        help="Tournament server URL.",
        rich_help_panel="Server",
    ),
    # --- Help ---
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit.",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    if ":" in name:
        console.print("[red]Policy name must not contain ':'[/red]")
        raise typer.Exit(1)
    if len(name) > 64:
        console.print("[red]Policy name must be at most 64 characters[/red]")
        raise typer.Exit(1)

    season_info = _resolve_season(server, login_server, season)

    init_kwargs: dict[str, str] = {}
    if init_kwarg:
        for kv in init_kwarg:
            key, val = _parse_init_kwarg(kv)
            init_kwargs[key] = val

    submitting = not no_submit
    parsed_secret_env: dict[str, str] = {}
    if use_bedrock:
        parsed_secret_env["USE_BEDROCK"] = "true"
    if secret_env:
        for kv in secret_env:
            key, val = _parse_secret_env(kv)
            parsed_secret_env[key] = val

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
        season=season_info.name if submitting else None,
        image=image,
        secret_env=parsed_secret_env if parsed_secret_env else None,
    )

    if result:
        console.print(f"[green]Upload complete: {result.name}:v{result.version}[/green]")
        if result.pools:
            console.print(f"[dim]Added to pools: {', '.join(result.pools)}[/dim]")
            console.print(f"[dim]Results:[/dim] {RESULTS_URL}")
        elif no_submit:
            console.print(f"\nTo submit to a tournament: cogames submit {result.name}:v{result.version}")


@app.command(
    name="submit",
    help="Submit a policy to a tournament season.",
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
        help="Policy name (e.g., 'my-policy' or 'my-policy:v3' for specific version).",
    ),
    season: Optional[str] = typer.Option(
        None,
        "--season",
        metavar="SEASON",
        help="Tournament season name.",
        rich_help_panel="Tournament",
    ),
    login_server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--login-server",
        metavar="URL",
        help="Authentication server URL.",
        rich_help_panel="Server",
    ),
    server: str = typer.Option(
        DEFAULT_SUBMIT_SERVER,
        "--server",
        "-s",
        metavar="URL",
        help="Tournament server URL.",
        rich_help_panel="Server",
    ),
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit.",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    import httpx  # noqa: PLC0415

    season_info = _resolve_season(server, login_server, season)
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
    profile_url = observatory_profile_url(pv.id, login_server_url=login_server)
    console.print(f"[dim]Profile:[/dim] {profile_url}")
    if has_display():
        webbrowser.open(profile_url)
    else:
        console.print("[dim]Browser launch skipped: no GUI display detected[/dim]")
    console.print(f"[dim]Results:[/dim] {RESULTS_URL}")
    console.print(f"[dim]CLI:[/dim] cogames leaderboard --season {season_name}")


@app.command(
    name="ship",
    help="Bundle, validate, upload, and submit a policy in one command.",
    rich_help_panel="Tournament",
    epilog="""[dim]Examples:[/dim]

[cyan]cogames ship -p ./train_dir/my_run -n my-policy[/cyan]            Ship and submit

[cyan]cogames ship -p ./run -n my-policy --dry-run[/cyan]               Validate only""",
    add_help_option=False,
)
def ship_cmd(
    ctx: typer.Context,
    name: str = typer.Option(
        ...,
        "--name",
        "-n",
        metavar="NAME",
        help="Name for your uploaded policy.",
        rich_help_panel="Upload",
    ),
    policy: str = typer.Option(
        ...,
        "--policy",
        "-p",
        metavar="POLICY",
        help=f"Policy specification: {policy_arg_example}.",
        rich_help_panel="Policy",
    ),
    init_kwarg: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--init-kwarg",
        "-k",
        metavar="KEY=VAL",
        help="Policy init kwargs (can be repeated).",
        rich_help_panel="Policy",
    ),
    include_files: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--include-files",
        "-f",
        metavar="PATH",
        help="Files or directories to include (can be repeated).",
        rich_help_panel="Files",
    ),
    setup_script: Optional[str] = typer.Option(
        None,
        "--setup-script",
        metavar="PATH",
        help="Python setup script to run before loading the policy.",
        rich_help_panel="Files",
    ),
    season: Optional[str] = typer.Option(
        None,
        "--season",
        metavar="SEASON",
        help="Tournament season (default: server's default season).",
        rich_help_panel="Tournament",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run validation only without uploading.",
        rich_help_panel="Validation",
    ),
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help="Skip policy validation in Docker.",
        rich_help_panel="Validation",
    ),
    image: str = typer.Option(
        DEFAULT_EPISODE_RUNNER_IMAGE,
        "--image",
        help="Docker image for container validation.",
        rich_help_panel="Validation",
    ),
    login_server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--login-server",
        metavar="URL",
        help="Authentication server URL.",
        rich_help_panel="Server",
    ),
    server: str = typer.Option(
        DEFAULT_SUBMIT_SERVER,
        "--server",
        metavar="URL",
        help="Tournament server URL.",
        rich_help_panel="Server",
    ),
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit.",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    if ":" in name:
        console.print("[red]Policy name must not contain ':'[/red]")
        raise typer.Exit(1)
    if len(name) > 64:
        console.print("[red]Policy name must be at most 64 characters[/red]")
        raise typer.Exit(1)

    season_info = _resolve_season(server, login_server, season)

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
        season=season_info.name,
        image=image,
    )

    if not result:
        return

    console.print(f"[green]Shipped: {result.name}:v{result.version}[/green]")
    if result.pools:
        console.print(f"[dim]Added to pools: {', '.join(result.pools)}[/dim]")

    if dry_run:
        console.print(f"[dim]Results:[/dim] {RESULTS_URL}")
        return

    console.print(f"[dim]Results:[/dim] {RESULTS_URL}")


@app.command(
    name="docs",
    help="Print documentation (run without arguments to see available docs).",
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
        help="Document name (readme, mission, technical_manual, scripted_agent, evals, mapgen).",
    ),
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit.",
        is_eager=True,
        callback=_help_callback,
    ),
) -> None:
    # Hardcoded mapping of document names to file paths and descriptions
    package_root = Path(__file__).parent.parent.parent
    docs_map: dict[str, tuple[Path, str]] = {
        "readme": (package_root / "README.md", "CoGames overview and documentation"),
        "mission": (package_root / "MISSION.md", "Mission briefing for CvC Deployment"),
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
        from rich.table import Table  # noqa: PLC0415

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
    app(prog_name="cogames")
