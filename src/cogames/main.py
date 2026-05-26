#!/usr/bin/env -S uv run
# need this to import and call suppress_noisy_logs first
# ruff: noqa: E402

"""CLI for CoGames - collection of environments for multi-agent cooperative and competitive games."""

from cogames.cli.utils import suppress_noisy_logs

suppress_noisy_logs()

import importlib
import importlib.metadata
import importlib.resources
import keyword
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, cast

if TYPE_CHECKING:
    import torch

import httpx
import typer
from click.core import ParameterSource
from packaging.version import Version
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from cogames import pickup as pickup_module
from cogames import play as play_module
from cogames import verbose
from cogames.cli.base import console
from cogames.cli.bitworld import bitworld_app
from cogames.cli.mission import (
    get_mission_name_and_config,
    get_mission_names_and_configs,
)
from cogames.cli.policy import (
    _translate_error,
    get_policy_spec,
    get_policy_specs_with_proportions,
    parse_policy_spec,
    policy_arg_example,
    policy_arg_w_proportion_example,
)
from cogames.cli.submit import (
    create_bundle,
)
from cogames.curricula import make_rotation
from cogames.device import resolve_training_device
from cogames.display_detect import has_display
from cogames.optional_deps import require_neural
from cogames.replays import ReplayPathRequest, launch_replay_path
from cogames.seed import seed_rollout_rng

# Always add current directory to Python path so optional plugins in the repo are discoverable.
sys.path.insert(0, ".")


logger = logging.getLogger("cogames.main")
_REPO_COGAMES_ROOT = Path(__file__).resolve().parents[2]
_DOC_DESCRIPTIONS: dict[str, str] = {
    "amongthem_policy": "AmongThem policy practice walkthrough",
    "readme": "CoGames overview and documentation",
    "mission": "Mission briefing for CvC Deployment",
    "technical_manual": "Technical manual for Cogames",
    "scripted_agent": "Scripted agent policy documentation",
}
_DOC_RESOURCE_PATHS: dict[str, tuple[str, ...]] = {
    "amongthem_policy": ("docs", "AMONGTHEM_POLICY.md"),
    "mission": ("docs", "MISSION.md"),
    "technical_manual": ("docs", "TECHNICAL_MANUAL.md"),
    "scripted_agent": ("docs", "SCRIPTED_AGENT.md"),
}

_POLICY_FREE_COMMANDS = {
    "bitworld",
    "docs",
    "docsync",
    "replay",
    "version",
}


def _read_docs_readme() -> str:
    readme_path = _REPO_COGAMES_ROOT / "README.md"
    return readme_path.read_text() if readme_path.exists() else importlib.metadata.metadata("cogames")["Description"]


def _read_packaged_doc(doc_name: str) -> str:
    resource_path = _DOC_RESOURCE_PATHS[doc_name]
    return importlib.resources.files("cogames").joinpath(*resource_path).read_text()


def _read_doc_text(doc_name: str) -> str:
    if doc_name == "readme":
        return _read_docs_readme()
    return _read_packaged_doc(doc_name)


def _register_policies(ctx: typer.Context) -> None:
    invoked_subcommand = ctx.invoked_subcommand
    if invoked_subcommand is None or invoked_subcommand in _POLICY_FREE_COMMANDS:
        return

    from mettagrid.policy.loader import discover_and_register_policies  # noqa: PLC0415

    discover_and_register_policies()


def _validate_policy_module_name_or_exit(path: Path) -> None:
    module_name = path.stem
    if not module_name.isidentifier() or keyword.iskeyword(module_name):
        console.print(
            f"[red]Output filename stem '{module_name}' is not importable as a Python module. "
            "Use letters, numbers, and underscores, and do not start with a number.[/red]"
        )
        raise typer.Exit(1)


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
    from cogsguard.missions.machina_1 import make_machina1_mission  # noqa: PLC0415

    # Create environment config
    env_cfg = make_machina1_mission(num_agents=1, max_steps=1000).make_env()
    console.print("[dim]Tutorial phases appear in-game. Press Enter or click Next to advance.[/dim]")
    console.print("[dim]Close the Mettascope window to exit.[/dim]")

    # Run play (blocks main thread)
    try:
        play_module.play(
            console,
            env_cfg=env_cfg,
            policy_specs=[parse_policy_spec("class=tutorial_noop,kw.tutorial=play")],
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
    from cogsguard.missions.tutorial import make_tutorial_mission  # noqa: PLC0415

    env_cfg = make_tutorial_mission().make_env()
    console.print("[dim]Tutorial phases appear in-game. Press Enter or click Next to advance.[/dim]")
    console.print("[dim]Close the Mettascope window to exit.[/dim]")

    # Run play (blocks main thread)
    try:
        play_module.play(
            console,
            env_cfg=env_cfg,
            policy_specs=[parse_policy_spec("class=tutorial_noop,kw.tutorial=cvc")],
            game_name="tutorial",
            render_mode="gui",
            autostart=False,
        )
    except KeyboardInterrupt:
        logger.info("CvC tutorial interrupted; exiting.")


app.add_typer(tutorial_app, name="tutorial", rich_help_panel="Tutorials")
app.add_typer(bitworld_app, name="bitworld", rich_help_panel="BitWorld")


def _help_callback(ctx: typer.Context, value: bool) -> None:
    """Callback for custom help option."""
    if value:
        console.print(ctx.get_help())
        raise typer.Exit()


@app.command(
    name="play",
    rich_help_panel="Play",
    help="""Play a game interactively.

This runs a single episode of the game using one or more policies.

By default, the policy is 'noop', so agents won't move unless manually controlled.
To see agents move by themselves, use `--policy starter` or `--policy class=random`.

Multiple -p flags assign one policy per team (in team order).

You can manually control the actions of a specific cog by clicking on a cog
in GUI mode or pressing M in unicode mode and using your arrow or WASD keys.
Log mode is non-interactive and doesn't support manual control.
""",
    epilog="""[dim]Examples:[/dim]

[cyan]cogames play -m arena[/cyan]                                  Interactive

[cyan]cogames play -m arena -p class=random[/cyan]                  Random policy

[cyan]cogames play -m arena -c 4 -p starter[/cyan]                  Starter policy, 4 cogs

[cyan]cogames play -m four_score -p nlanky -p starter -p random -p noop[/cyan]
                                                                 One policy per team

[cyan]cogames play -m four_score -p nlanky:1 -p random:2[/cyan]     Mixed teams (cycling pattern)

[cyan]cogames play -m arena --save-replay-file ./latest.json.z[/cyan] Overwrite fixed replay file

[cyan]cogames play -m machina_1 -v talk -r gui[/cyan]             Speech bubbles over cogs

[cyan]cogames play -m machina_1 -r unicode[/cyan]                   Terminal mode""",
    add_help_option=False,
)
def play_cmd(
    ctx: typer.Context,
    # --- Game Setup ---
    game: str = typer.Option(
        "cogsguard",
        "--game",
        metavar="GAME",
        help="Game to play (default: cogsguard).",
        rich_help_panel="Game Setup",
    ),
    mission: Optional[str] = typer.Option(
        None,
        "--mission",
        "-m",
        metavar="MISSION",
        help="Mission to play, or a path to a mission config file.",
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
    policies: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--policy",
        "-p",
        metavar="POLICY",
        help="Policy per team. One -p applies to all teams; multiple -p assigns one per team.",
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
    from mettagrid.mapgen.mapgen import MapGen  # noqa: PLC0415

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
    raw_policies = policies if policies else ["class=noop"]
    policy_specs = get_policy_specs_with_proportions(ctx, raw_policies, device=str(resolved_device))

    console.print(f"[cyan]Playing {resolved_mission}[/cyan]")
    console.print(f"Max Steps: {env_cfg.game.max_steps}, Render: {render}")

    play_module.play(
        console,
        env_cfg=env_cfg,
        policy_specs=policy_specs,
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

  [cyan]cogames replay ./replays/game.json.z[/cyan]              Replay Cogs vs Clips in MettaScope

  [cyan]cogames replay ./train_dir/my_run/replay.bin[/cyan]      Replay a legacy MettaGrid run

  [cyan]cogames replay ./among_them.bitreplay[/cyan]             Replay BitWorld in the global client""",
    add_help_option=False,
)
def replay_cmd(
    replay_path: Path = typer.Argument(  # noqa: B008
        ...,
        metavar="FILE",
        help="Path to a MettaGrid replay (.json.z, .replay, .bin) or a BitWorld replay (.bitreplay).",
    ),
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit.",
        is_eager=True,
        callback=_help_callback,
    ),
    duration: Optional[float] = typer.Option(
        None,
        "--duration",
        help="Seconds to keep a BitWorld replay server alive. MettaScope replays ignore this option.",
    ),
) -> None:
    if not replay_path.exists():
        console.print(f"[red]Error: Replay file not found: {replay_path}[/red]")
        raise typer.Exit(1)

    if not has_display():
        console.print("[red]Error: This command requires a GUI display.[/red]")
        raise typer.Exit(1)

    exit_code = launch_replay_path(ReplayPathRequest(replay_path=replay_path, duration=duration))
    if exit_code != 0:
        raise typer.Exit(exit_code)


# TODO: Verify make-policy templates work with CvC game mechanics
@tutorial_app.command(
    name="make-policy",
    help="Create a new policy from a template. Requires exactly one policy type.",
    rich_help_panel="Tutorial",
    epilog="""[dim]Examples:[/dim]

[cyan]cogames tutorial make-policy --trainable -o my_nn_policy.py[/cyan]        Trainable (neural network)

[cyan]cogames tutorial make-policy --scripted -o my_scripted_policy.py[/cyan]  Scripted (rule-based)

[cyan]cogames tutorial make-policy --amongthem -o amongthem_policy.py[/cyan]
                                                                  AmongThem scripted practice

[cyan]cogames tutorial make-policy --amongthem --print[/cyan]     Preview template source""",
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
    amongthem: bool = typer.Option(
        False,
        "--amongthem",
        help="Create an AmongThem BitWorld scripted practice policy.",
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
    print_template: bool = typer.Option(
        False,
        "--print",
        help="Print the policy template instead of writing a file.",
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
    if sum(int(selected) for selected in (trainable, scripted, amongthem)) != 1:
        console.print("[red]Error: Specify exactly one of --trainable, --scripted, or --amongthem[/red]")
        console.print("[dim]Examples:[/dim]")
        console.print("[dim]  cogames tutorial make-policy --trainable -o my_nn_policy.py[/dim]")
        console.print("[dim]  cogames tutorial make-policy --scripted -o my_scripted_policy.py[/dim]")
        console.print("[dim]  cogames tutorial make-policy --amongthem -o amongthem_policy.py[/dim]")
        raise typer.Exit(1)

    try:
        if trainable:
            require_neural("cogames tutorial make-policy --trainable")
            import cogames.policy.trainable_policy_template as trainable_policy_template  # noqa: PLC0415

            template_path = Path(trainable_policy_template.__file__)
            policy_class = "MyTrainablePolicy"
            policy_type = "Trainable"
        elif amongthem:
            import cogames.policy.amongthem_policy_template as amongthem_policy_template  # noqa: PLC0415

            template_path = Path(amongthem_policy_template.__file__)
            policy_class = "AmongThemPolicy"
            policy_type = "AmongThem"
        else:
            # Deferred: imported only to locate its source file as a template.
            import cogames.policy.starter_agent as starter_agent  # noqa: PLC0415

            template_path = Path(starter_agent.__file__)
            policy_class = "StarterPolicy"
            policy_type = "Scripted"

        if not template_path.exists():
            console.print(f"[red]Error: {policy_type} policy template not found[/red]")
            raise typer.Exit(1)

        template_text = template_path.read_text()
        if not trainable:
            lines = template_text.splitlines()
            lines = [line for line in lines if not line.strip().startswith("short_names =")]
            template_text = "\n".join(lines) + "\n"

        if print_template:
            sys.stdout.write(template_text)
            return

        dest_path = Path.cwd() / output
        _validate_policy_module_name_or_exit(dest_path)

        if dest_path.exists():
            console.print(f"[yellow]Warning: {dest_path} already exists. Overwriting...[/yellow]")

        dest_path.write_text(template_text)
        console.print(f"[green]{policy_type} policy template copied to: {dest_path}[/green]")

        if trainable:
            console.print(
                f"[dim]Train with: cogames tutorial train -m arena -p class={dest_path.stem}.{policy_class}[/dim]"
            )
        elif amongthem:
            console.print(f"[dim]Edit starter policy: {dest_path}[/dim]", soft_wrap=True)
            console.print("[dim]Start with: AmongThemPolicy._choose_actions()[/dim]")
            console.print("[dim]Preview template: cogames tutorial make-policy --amongthem --print[/dim]")
            console.print("[dim]Wrap it in a Docker player before submitting to Among Them Daily.[/dim]")
            console.print("[dim]Docker/Coworld submission guide: https://softmax.com/play_amongthem.md[/dim]")
        else:
            console.print(f"[dim]Play with: cogames play -m arena -p class={dest_path.stem}.{policy_class}[/dim]")

    except Exception as exc:  # pragma: no cover - user input
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc


@tutorial_app.command(
    name="train",
    help="""Train a policy on one or more missions.

Requires the ``neural`` extra (PyTorch + PufferLib).
Install with: ``pip install cogames\\[neural]``.

By default, our 'lstm' policy architecture is used. You can select a different architecture
(like 'stateless' or 'baseline'), or define your own implementing the MultiAgentPolicy
interface with a trainable network() method (see mettagrid/policy/policy.py).

Continue training from a checkpoint using URI format, or load weights into an explicit class
with class=...,data=... syntax.

Supply repeated -m flags to create a training curriculum that rotates through missions.
Use wildcards (*) in mission names to match multiple missions at once.""",
    rich_help_panel="Tutorial",
    epilog="""[dim]Examples:[/dim]

[cyan]cogames tutorial train -m arena[/cyan]                             Basic training

[cyan]cogames tutorial train -m arena -p class=baseline[/cyan]
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
    require_neural("cogames tutorial train")
    from cogames import train as train_module  # noqa: PLC0415

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
    # require_neural above guarantees torch is installed, so this always returns torch.device.
    torch_device = cast("torch.device", resolve_training_device(console, device))

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


@app.command(
    name="run",
    help="""Evaluate one or more policies on missions.

With multiple policies (e.g., 2 policies, 4 agents), each policy always controls 2 agents,
but which agents swap between policies each episode.

With one policy, this command is equivalent to `cogames scrimmage`.
""",
    rich_help_panel="Evaluate",
    epilog="""[dim]Examples:[/dim]

[cyan]cogames run -m arena -p lstm[/cyan]                         Evaluate single policy

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

[cyan]cogames scrimmage -m arena -p lstm[/cyan]                          Single policy eval""",
    add_help_option=False,
)
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
    from cogames import evaluate as evaluate_module  # noqa: PLC0415
    from cogames.device import resolve_training_device  # noqa: PLC0415
    from mettagrid.mapgen.mapgen import MapGen  # noqa: PLC0415

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
        "arena",
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
    name="diagnose",
    help="Run diagnostic evals for a policy checkpoint.",
    rich_help_panel="Evaluate",
    epilog="""[dim]Examples:[/dim]

[cyan]cogames diagnose ./train_dir/my_run[/cyan]                         Default CvC evals

[cyan]cogames diagnose lstm --scripted-baseline-policy scripted.basic[/cyan]   Compare against scripted baseline

[cyan]cogames diagnose lstm --known-strong-policy my_best_policy[/cyan]
    Normalize against known-strong policy

[cyan]cogames diagnose lstm --compare-run-dir outputs/cogames-diagnose/prev_run[/cyan]  Stability comparison""",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def diagnose_cmd(ctx: typer.Context) -> None:
    from cogames import diagnose as diagnose_module  # noqa: PLC0415

    diagnose_app = typer.Typer(add_help_option=False)
    diagnose_app.command()(diagnose_module.diagnose_cmd)
    diagnose_app(prog_name="cogames diagnose", args=list(ctx.args))


@app.command(
    name="create-bundle",
    help="Create a submission bundle zip from a policy.",
    rich_help_panel="Policies",
    epilog="""[dim]Examples:[/dim]

[cyan]cogames create-bundle -p <POLICY_OR_CHECKPOINT> -o submission.zip[/cyan]
  Create a submission bundle

[cyan]cogames create-bundle -p <POLICY_OR_CHECKPOINT> -o submission.zip
  -f <EXTRA_PATH> ... --setup-script <SETUP.py>[/cyan]
  Include extra runtime files or setup when needed""",
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


def _parse_init_kwarg(value: str) -> tuple[str, str]:
    """Parse a key=value string into a tuple."""
    if "=" not in value:
        raise typer.BadParameter(f"Expected key=value format, got: {value}")
    key, _, val = value.partition("=")
    return key.replace("-", "_"), val


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
        help=f"Document name ({', '.join(sorted(_DOC_DESCRIPTIONS))}).",
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
    # If no argument provided, show available documents
    if doc_name is None:
        from rich.table import Table  # noqa: PLC0415

        console.print("\n[bold cyan]Available Documents:[/bold cyan]\n")
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED, padding=(0, 1))
        table.add_column("Document", style="blue", no_wrap=True)
        table.add_column("Description", style="white")

        for name, description in sorted(_DOC_DESCRIPTIONS.items()):
            table.add_row(name, description)

        console.print(table)
        console.print("\nUsage: [bold]cogames docs <document_name>[/bold]")
        console.print("Example: [bold]cogames docs mission[/bold]")
        return

    if doc_name not in _DOC_DESCRIPTIONS:
        available = ", ".join(sorted(_DOC_DESCRIPTIONS.keys()))
        console.print(f"[red]Error: Unknown document '{doc_name}'[/red]")
        console.print(f"\nAvailable documents: {available}")
        raise typer.Exit(1)

    try:
        content = _read_doc_text(doc_name)
        console.print(content, markup=False)
    except Exception as exc:
        console.print(f"[red]Error reading document: {exc}[/red]")
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app(prog_name="cogames")
