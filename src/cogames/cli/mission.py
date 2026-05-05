"""CLI mission resolution and display."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich import box
from rich.console import Console
from rich.table import Table

from cogames.cli.base import console
from cogames.core import CoGameMission, CoGameMissionVariant
from cogames.game import CoGame, get_game
from cogames.variants import VariantRegistry
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.mapgen.mapgen import MapGenConfig

_SUPPORTED_MISSION_EXTENSIONS = [".yaml", ".yml", ".json"]


# ---------------------------------------------------------------------------
# Mission lookup
# ---------------------------------------------------------------------------


def find_mission(
    game: CoGame,
    mission_name: str,
    *,
    include_evals: bool = False,
) -> CoGameMission:
    parts = mission_name.split(".", 1)
    base_name = parts[0]
    sub_name = parts[1] if len(parts) > 1 else None

    # "evals.<name>" is a namespace shortcut — look up <name> directly among eval missions.
    if base_name == "evals" and sub_name is not None:
        for mission in game.eval_missions:
            if mission.name == sub_name:
                return mission
        available = [m.name for m in game.eval_missions]
        raise ValueError(f"Unknown eval mission '{sub_name}'. Available: {', '.join(available)}")

    search: list[CoGameMission] = list(game.missions)
    if include_evals:
        search = [*search, *game.eval_missions]
    for mission in search:
        if mission.name == base_name:
            if sub_name is not None:
                if sub_name not in mission.sub_missions:
                    raise ValueError(
                        f"Unknown sub-mission '{sub_name}' for '{base_name}'. "
                        f"Available: {', '.join(mission.sub_missions) if mission.sub_missions else 'none'}"
                    )
                variant = game.variant_registry.get(sub_name)
                assert variant is not None, f"Sub-mission variant '{sub_name}' not in game registry"
                return mission.with_variants([variant])
            return mission
    available = [m.name for m in search]
    raise ValueError(f"Unknown mission '{base_name}'. Available: {', '.join(available)}")


def apply_variants(
    mission: CoGameMission,
    variants: Sequence[str | CoGameMissionVariant],
    cogs: Optional[int],
) -> CoGameMission:
    """Apply variants and cogs selection to a mission."""
    if variants:
        mission = mission.with_variants(variants)
    if cogs is not None:
        mission = mission.with_cogs(cogs)
    return mission


def _resolve_mission_scoped_variant(mission: CoGameMission, name: str) -> CoGameMissionVariant | None:
    mission._ensure_variant_modules_loaded()
    for prefix in mission.variant_module_prefixes():
        for candidate in reversed(CoGameMissionVariant._type_candidates.get(name, [])):
            if candidate.__module__.startswith(prefix):
                return candidate()  # pyright: ignore[reportCallIssue]
    return None


def resolve_mission_variants(
    game: CoGame,
    mission: CoGameMission,
    variants_arg: Optional[list[str]],
) -> list[CoGameMissionVariant]:
    if not variants_arg:
        return []

    mission_variant_names = {
        name
        for name, candidates in CoGameMissionVariant._type_candidates.items()
        if any(
            candidate.__module__.startswith(prefix)
            for prefix in mission.variant_module_prefixes()
            for candidate in candidates
        )
    }
    available = ", ".join(sorted({variant.name for variant in game.variant_registry.all()} | mission_variant_names))

    resolved: list[CoGameMissionVariant] = []
    for name in variants_arg:
        mission_variant = _resolve_mission_scoped_variant(mission, name)
        if mission_variant is not None:
            resolved.append(mission_variant)
            continue

        game_variant = game.variant_registry.get(name)
        if game_variant is not None:
            resolved.append(game_variant)
            continue

        raise ValueError(f"Unknown variant '{name}'.\nAvailable variants: {available}")

    return resolved


def resolve_mission(
    game: CoGame,
    mission_arg: str,
    variants_arg: Optional[list[str]] = None,
    cogs: Optional[int] = None,
) -> tuple[str, MettaGridConfig, Optional[CoGameMission]]:
    """Resolve mission by name or file path. Returns (name, env_cfg, mission or None)."""
    if any(mission_arg.endswith(ext) for ext in [".yaml", ".yml", ".json", ".py"]):
        path = Path(mission_arg)
        if not path.exists():
            raise ValueError(f"File not found: {mission_arg}")
        if not path.is_file():
            raise ValueError(f"Not a file: {mission_arg}")
        if path.suffix == ".py":
            return mission_arg, load_mission_config_from_python(path), None
        if path.suffix in [".yaml", ".yml", ".json"]:
            return mission_arg, load_mission_config(path), None
        raise ValueError(f"Unsupported file format: {path.suffix}")

    mission = find_mission(game, mission_arg, include_evals=True)
    requested_variants = resolve_mission_variants(game, mission, variants_arg)
    mission = apply_variants(mission, requested_variants, cogs)
    return mission.full_name(), mission.make_env(), mission


def resolve_missions_by_wildcard(
    game: CoGame,
    mission_arg: str,
    variants_arg: Optional[list[str]],
    cogs: Optional[int],
) -> list[tuple[str, MettaGridConfig]]:
    if "*" not in mission_arg:
        name, env_cfg, _ = resolve_mission(game, mission_arg, variants_arg, cogs)
        return [(name, env_cfg)]
    regex = mission_arg.replace(".", "\\.").replace("*", ".*")
    all_names = get_all_mission_names(game) + get_all_eval_mission_names(game)
    matching = [n for n in all_names if re.search(regex, n)]
    return [(name, env_cfg) for name, env_cfg, _ in (resolve_mission(game, m, variants_arg, cogs) for m in matching)]


def get_all_mission_names(game: CoGame) -> list[str]:
    return [m.full_name() for m in game.missions]


def get_all_eval_mission_names(game: CoGame) -> list[str]:
    return [m.full_name() for m in game.eval_missions]


# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------


def load_mission_config_from_python(path: Path) -> MettaGridConfig:
    """Load a mission configuration from a Python file.

    The Python file should define a function called 'get_config()' that returns a MettaGridConfig.
    Alternatively, it can define a variable named 'config' that is a MettaGridConfig.
    """
    spec = importlib.util.spec_from_file_location("game_config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Failed to load Python module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["game_config"] = module
    spec.loader.exec_module(module)

    if hasattr(module, "get_config") and callable(module.get_config):
        config = module.get_config()
    elif hasattr(module, "config"):
        config = module.config
    else:
        raise ValueError(
            f"Python file {path} must define either a 'get_config()' function "
            "or a 'config' variable that returns/contains a MettaGridConfig"
        )

    if not isinstance(config, MettaGridConfig):
        raise ValueError(f"Python file {path} must return a MettaGridConfig instance")

    del sys.modules["game_config"]
    return config


def save_mission_config(config: MettaGridConfig, output_path: Path) -> None:
    """Save a mission configuration to file."""
    if output_path.suffix in [".yaml", ".yml"]:
        with open(output_path, "w") as f:
            yaml.dump(config.model_dump(mode="yaml"), f, default_flow_style=False, sort_keys=False)
    elif output_path.suffix == ".json":
        with open(output_path, "w") as f:
            json.dump(config.model_dump(mode="json"), f, indent=2)
    else:
        raise ValueError(
            f"Unsupported file format: {output_path.suffix}. Supported: {', '.join(_SUPPORTED_MISSION_EXTENSIONS)}"
        )


def load_mission_config(path: Path) -> MettaGridConfig:
    """Load a mission configuration from file."""
    if path.suffix in [".yaml", ".yml"]:
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
    elif path.suffix == ".json":
        with open(path, "r") as f:
            config_dict = json.load(f)
    else:
        raise ValueError(
            f"Unsupported file format: {path.suffix}. Supported: {', '.join(_SUPPORTED_MISSION_EXTENSIONS)}"
        )
    return MettaGridConfig(**config_dict)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def print_missions(game: CoGame, con: Console, mission_filter: Optional[str] = None) -> None:
    """List available missions."""
    missions = game.missions
    if not missions:
        con.print("No missions found")
        return
    if mission_filter:
        missions = [m for m in missions if mission_filter in m.name]
        if not missions:
            con.print(f"[red]No missions matching '{mission_filter}'[/red]")
            return
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED, padding=(0, 1))
    table.add_column("Mission", style="blue", no_wrap=True)
    table.add_column("Cogs", style="green", justify="center")
    table.add_column("Map Size", style="green", justify="center")
    table.add_column("Description", style="white")
    for m in missions:
        mb = m.map_builder
        map_size = f"{mb.width}x{mb.height}" if hasattr(mb, "width") and hasattr(mb, "height") else "N/A"  # type: ignore[attr-defined]
        agent_range = f"{m.min_cogs}-{m.max_cogs}"
        table.add_row(m.name, agent_range, map_size, m.description)
        for sub_name in m.sub_missions:
            sub_variant = game.variant_registry.get(sub_name)
            desc = sub_variant.description if sub_variant else ""
            table.add_row(
                f"  [dim]{m.name}.[/dim]{sub_name}",
                agent_range,
                map_size,
                desc or f"[dim]{sub_name}[/dim]",
            )
    con.print(table)
    con.print("\nTo set [bold blue]-m[/bold blue]:")
    con.print("  • Use [blue]<mission>[/blue] (e.g., machina_1)")
    con.print("  • Use [blue]<mission>.<sub>[/blue] (e.g., machina_1.clips)")
    con.print("  • Or pass a mission config file path")
    con.print("\nCogs:")
    con.print("  • [green]--cogs N[/green] or [green]-c N[/green]")
    con.print("\n[bold green]Examples:[/bold green]")
    con.print("  cogames play --mission [blue]machina_1[/blue]")
    con.print("  cogames play --mission [blue]machina_1.clips[/blue]")
    con.print("  cogames play --mission [blue]machina_1[/blue] --cogs [green]8[/green]")
    con.print("  cogames tutorial train --mission [blue]machina_1[/blue] --cogs [green]4[/green]")


def print_evals(game: CoGame, con: Console) -> None:
    """Print a table listing all available eval missions."""
    evals = game.eval_missions
    if not evals:
        con.print("No eval missions found")
        return
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED, padding=(0, 1))
    table.add_column("Mission", style="blue", no_wrap=True)
    table.add_column("Cogs", style="green", justify="center")
    table.add_column("Map Size", style="green", justify="center")
    table.add_column("Description", style="white")
    for m in evals:
        mb = m.map_builder
        map_size = "N/A"
        if hasattr(mb, "width") and hasattr(mb, "height"):
            map_size = f"{mb.width}x{mb.height}"  # type: ignore[attr-defined]
        agent_range = f"{m.min_cogs}-{m.max_cogs}"
        table.add_row(m.name, agent_range, map_size, m.description)
    con.print(table)
    con.print("\nTo play an eval mission:")
    con.print("  [bold]cogames play[/bold] --mission [blue]oxygen_bottleneck[/blue]")


def print_variants(game: CoGame, con: Console) -> None:
    """Print a table listing all available variants."""
    variants = game.variant_registry.all()
    if not variants:
        return
    con.print("\n")
    table = Table(
        title="Available Variants",
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
        padding=(0, 1),
    )
    table.add_column("Variant", style="yellow", no_wrap=True)
    table.add_column("Description", style="white")
    for v in variants:
        table.add_row(v.name, v.description)
    con.print(table)


def print_mission_dependencies(game: CoGame, con: Console) -> None:
    """Print the variant dependency tree for each mission."""
    for mission in game.missions:
        copy = mission.model_copy(deep=True)
        extra_names = [n for n in copy._variant_registry._variants if n != copy.default_variant]
        default = [copy.default_variant] if copy.default_variant else []
        copy._variant_registry.run_configure([*default, *extra_names])
        edges = copy._variant_registry._edges

        con.print(f"\n[bold cyan]{mission.name}[/bold cyan]", end="")
        if mission.description:
            con.print(f" [dim]— {mission.description}[/dim]")
        else:
            con.print()

        if not edges:
            con.print("  [dim](no variant dependencies)[/dim]")
            continue

        _print_dependency_tree(copy._variant_registry, edges, con, indent="  ")


def _print_dependency_tree(
    registry: VariantRegistry,
    edges: list[tuple[str, str, str]],
    con: Console,
    indent: str = "",
) -> None:
    """Render a dependency tree from edges."""
    # Build adjacency
    deps: dict[str, list[tuple[str, str]]] = {}
    all_nodes: set[str] = set()
    for src, dst, kind in edges:
        deps.setdefault(src, []).append((dst, kind))
        all_nodes.update((src, dst))

    # Roots: nodes not depended on by others
    depended_on = {dst for _, dst, _ in edges}
    roots = sorted(all_nodes - depended_on)

    printed: set[str] = set()

    def _walk(name: str, prefix: str, is_last: bool) -> None:
        connector = "└── " if is_last else "├── "
        if name in printed:
            con.print(f"{prefix}{connector}[dim]{name} (see above)[/dim]")
            return
        printed.add(name)
        variant = registry.get(name)
        desc = f" [dim]— {variant.description}[/dim]" if variant and variant.description else ""
        con.print(f"{prefix}{connector}[yellow]{name}[/yellow]{desc}")
        children = deps.get(name, [])
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, (child, kind) in enumerate(children):
            tag = "" if kind == "required" else " [dim](optional)[/dim]"
            if child in printed:
                child_connector = "└── " if i == len(children) - 1 else "├── "
                con.print(f"{child_prefix}{child_connector}[dim]{child} (see above)[/dim]{tag}")
            else:
                # Recurse for full depth
                printed.add(child)
                child_variant = registry.get(child)
                child_desc = (
                    f" [dim]— {child_variant.description}[/dim]" if child_variant and child_variant.description else ""
                )
                child_connector = "└── " if i == len(children) - 1 else "├── "
                con.print(f"{child_prefix}{child_connector}[yellow]{child}[/yellow]{child_desc}{tag}")
                gc = deps.get(child, [])
                gc_prefix = child_prefix + ("    " if i == len(children) - 1 else "│   ")
                for j, (grandchild, _gc_kind) in enumerate(gc):
                    _walk(grandchild, gc_prefix, j == len(gc) - 1)

    for i, root in enumerate(roots):
        _walk(root, indent, i == len(roots) - 1)


def print_variant_graph(game: CoGame, con: Console) -> None:
    """Print the variant dependency graph."""
    edges = game.variant_registry.build_dependency_graph()
    con.print("\n[bold]Variant Dependency Graph[/bold]\n")
    _print_dependency_tree(game.variant_registry, edges, con)


# ---------------------------------------------------------------------------
# CLI-level convenience wrappers (resolve game by name, use default console)
# ---------------------------------------------------------------------------


def get_mission_name_and_config(
    ctx: typer.Context,
    mission_arg: Optional[str],
    *,
    game_name: str = "cogsguard",
    variants_arg: Optional[list[str]] = None,
    cogs: Optional[int] = None,
    steps: Optional[int] = None,
) -> tuple[str, MettaGridConfig, Optional[CoGameMission]]:
    if not mission_arg:
        console.print(ctx.get_help())
        console.print("[yellow]Missing: --mission / -m[/yellow]\n")
    else:
        try:
            game = get_game(game_name)
            name, env_cfg, mission_cfg = resolve_mission(game, mission_arg, variants_arg, cogs)
            if steps is not None:
                env_cfg.game.max_steps = steps
            return name, env_cfg, mission_cfg
        except ValueError as e:
            error_msg = str(e)
            if "variant" in error_msg.lower():
                console.print(f"[red]{error_msg}[/red]")
            else:
                console.print(f"[red]Mission '{mission_arg}' not found.[/red]")
                console.print("[dim]Use a known mission name or pass a mission config file path.[/dim]\n")
            raise typer.Exit(1) from e
    print_missions(get_game(game_name), console)
    console.print("\n" + ctx.get_usage())
    console.print("\n")
    raise typer.Exit(1)


def get_mission_names_and_configs(
    ctx: typer.Context,
    missions_arg: Optional[list[str]],
    *,
    game_name: str = "cogsguard",
    variants_arg: Optional[list[str]] = None,
    cogs: Optional[int] = None,
    steps: Optional[int] = None,
) -> list[tuple[str, MettaGridConfig]]:
    if not missions_arg:
        console.print(ctx.get_help())
        console.print("[yellow]Supply at least one: --mission / -m[/yellow]\n")
    else:
        try:
            game = get_game(game_name)
            not_deduped = [
                (name, env_cfg)
                for mission in missions_arg
                for name, env_cfg in resolve_missions_by_wildcard(game, mission, variants_arg, cogs)
            ]
            seen: set[str] = set()
            deduped = []
            for name, env_cfg in not_deduped:
                if name not in seen:
                    seen.add(name)
                    deduped.append((name, env_cfg))
            if not deduped:
                raise ValueError(f"No missions found for {missions_arg}")
            if steps is not None:
                for _, env_cfg in deduped:
                    env_cfg.game.max_steps = steps
            return deduped
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            console.print("[dim]Use known mission names, wildcard patterns, or mission config file paths.[/dim]\n")
            raise typer.Exit(1) from e
    print_missions(get_game(game_name), console)
    console.print("\n" + ctx.get_usage())
    console.print("\n")
    raise typer.Exit(1)


def get_mission(
    mission_arg: str,
    variants_arg: Optional[list[str]] = None,
    cogs: Optional[int] = None,
    game_name: str = "cogsguard",
) -> tuple[str, MettaGridConfig, Optional[CoGameMission]]:
    """Get a specific mission configuration by name."""
    game = get_game(game_name)
    return resolve_mission(game, mission_arg, variants_arg, cogs)


def get_all_missions(game_name: str = "cogsguard") -> list[str]:
    """Get all core mission names (excludes evals)."""
    return get_all_mission_names(get_game(game_name))


def get_all_missions_list(game_name: str = "cogsguard") -> list[CoGameMission]:
    """Get all core mission objects (excludes evals)."""
    return list(get_game(game_name).missions)


def list_missions(mission_filter: Optional[str] = None, game_name: str = "cogsguard") -> None:
    print_missions(get_game(game_name), console, mission_filter)


def list_evals(game_name: str = "cogsguard") -> None:
    print_evals(get_game(game_name), console)


def list_variants(game_name: str = "cogsguard") -> None:
    print_variants(get_game(game_name), console)


def describe_mission(
    mission_name: str,
    game_config: MettaGridConfig,
    mission_cfg: Optional[CoGameMission] = None,
) -> None:
    """Print detailed information about a specific mission."""
    from cogsguard.missions.terrain import MachinaArena  # noqa: PLC0415

    console.print(f"\n[bold cyan]{mission_name}[/bold cyan]\n")

    if mission_cfg is not None:
        console.print("[bold]Description:[/bold]")
        console.print(f"  {mission_cfg.description}\n")
        console.print(f"[bold]Default Variant:[/bold] {mission_cfg.default_variant}")
        console.print("")

    console.print("[bold]Mission Configuration:[/bold]")
    console.print(f"  • Number of agents: {game_config.game.num_agents}")
    if isinstance(game_config.game.map_builder, MapGenConfig):
        console.print(f"  • Map size: {game_config.game.map_builder.width}x{game_config.game.map_builder.height}")
        instance = getattr(game_config.game.map_builder, "instance", None)
        if isinstance(instance, MachinaArena.Config):
            console.print("\n[bold]MapGen (MachinaArena):[/bold]")
            console.print(f"  • Base biome: {instance.base_biome}")
            if instance.biome_weights:
                console.print(f"  • Biome weights: {instance.biome_weights}")
            console.print(f"  • Building coverage: {instance.building_coverage}")
    console.print(f"  • Move energy cost: {game_config.game.actions.move.consumed_resources.get('energy', 0)}")

    console.print("\n[bold]Available Actions:[/bold]")
    for n, a in game_config.game.actions.model_dump().items():
        if a["enabled"]:
            console.print(f"  • {n}: {a['consumed_resources']}")

    console.print("\n[bold]Stations:[/bold]")
    for obj_name in game_config.game.objects:
        console.print(f"  • {obj_name}")

    console.print("\n[bold]Agent Configuration:[/bold]")
    console.print(f"  • Default resource limit: {game_config.game.agent.inventory.default_limit}")
    if game_config.game.agent.inventory.limits:
        console.print(f"  • Resource limits: {game_config.game.agent.inventory.limits}")
