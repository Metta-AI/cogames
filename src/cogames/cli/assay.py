"""CLI commands for assay runs."""

from __future__ import annotations

import time
from typing import Optional
from uuid import UUID

import httpx
import typer
from rich import box
from rich.table import Table

from cogames.cli.base import cli_http_errors, console, emit_json
from cogames.cli.client import TournamentServerClient
from cogames.cli.generated_models import AssayRunResponse, AssayStatus, MissionSpec
from cogames.cli.submit import DEFAULT_SUBMIT_SERVER
from softmax.auth import DEFAULT_COGAMES_SERVER

assay_app = typer.Typer(
    help="Assay run commands.",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    rich_markup_mode="rich",
)

_SERVER_OPTION = typer.Option(
    DEFAULT_SUBMIT_SERVER,
    "--server",
    "-s",
    metavar="URL",
    help="Tournament server URL.",
    rich_help_panel="Server",
)
_LOGIN_SERVER_OPTION = typer.Option(
    DEFAULT_COGAMES_SERVER,
    "--login-server",
    metavar="URL",
    help="Authentication server URL.",
    rich_help_panel="Server",
)
_JSON_OPTION = typer.Option(False, "--json", help="Print raw JSON.", rich_help_panel="Output")


def _get_authed_client(login_server: str, server: str) -> TournamentServerClient:
    client = TournamentServerClient.from_login(server_url=server, login_server=login_server)
    if client is None:
        raise typer.Exit(1)
    return client


def _resolve_policy_version_id(client: TournamentServerClient, policy_arg: str) -> UUID:
    """Resolve POLICY_NAME[:VERSION] or bare UUID to a policy_version_id."""
    try:
        return UUID(policy_arg)
    except ValueError:
        pass
    name, _, version_str = policy_arg.partition(":")
    version = int(version_str.lstrip("v")) if version_str else None
    pv = client.lookup_policy_version(name=name, version=version)
    if pv is None:
        console.print(f"[red]Policy not found:[/red] {policy_arg!r}")
        raise typer.Exit(1)
    return pv.id


def _build_mission_specs(
    mission_set_name: str,
    episodes: int,
    max_steps: int,
) -> list[MissionSpec]:
    from cogames.diagnose import _load_diagnose_missions  # noqa: PLC0415

    missions_objs = _load_diagnose_missions(mission_set_name)
    specs: list[MissionSpec] = []
    for mission in missions_objs:
        env_cfg = mission.make_env()
        env_cfg.game.max_steps = max_steps
        specs.append(
            MissionSpec(
                mission_name=mission.full_name(),
                env_config=env_cfg.model_dump(mode="json"),
                num_episodes=episodes,
                max_steps=max_steps,
            )
        )
    return specs


def _format_status(status: AssayStatus) -> str:
    colors = {
        AssayStatus.pending: "yellow",
        AssayStatus.completed: "green",
        AssayStatus.failed: "red",
    }
    color = colors.get(status, "white")
    return f"[{color}]{status.value}[/{color}]"


def _print_run(run: AssayRunResponse) -> None:
    console.print(f"\n[bold]Assay Run:[/bold] {run.id}")
    console.print(f"Status: {_format_status(run.status)}")
    console.print(f"Policy version: {run.policy_version_id}")
    if run.name:
        console.print(f"Name: {run.name}")
    if run.compat_version:
        console.print(f"Compat version: {run.compat_version}")
    console.print(f"Jobs: {run.completed_jobs}/{run.total_jobs} completed, {run.failed_jobs} failed")
    if run.error:
        console.print(f"[red]Error:[/red] {run.error}")


@assay_app.command(name="status", help="Show status of an assay run or the latest run for a policy.")
def assay_status(
    policy_or_run_id: str = typer.Argument(
        ..., metavar="POLICY_OR_RUN_ID", help="Assay run UUID, or policy name (name[:version])."
    ),
    login_server: str = _LOGIN_SERVER_OPTION,
    server: str = _SERVER_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    client = _get_authed_client(login_server, server)
    with cli_http_errors("Assay status"), client:
        try:
            run_id = UUID(policy_or_run_id)
            run = client.get_assay_run(run_id)
        except ValueError:
            policy_version_id = _resolve_policy_version_id(client, policy_or_run_id)
            runs = client.list_assay_runs(policy_version_id=policy_version_id)
            if not runs:
                console.print(f"[yellow]No assay runs found for policy {policy_or_run_id!r}.[/yellow]")
                return
            run = runs[0]

    if json_output:
        emit_json(run.model_dump(mode="json"))
        return

    _print_run(run)


@assay_app.command(name="list", help="List assay runs.")
def assay_list(
    policy: Optional[str] = typer.Option(None, "--policy", "-p", help="Filter by policy name[:version] or UUID."),
    login_server: str = _LOGIN_SERVER_OPTION,
    server: str = _SERVER_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    client = _get_authed_client(login_server, server)
    with cli_http_errors("Assay runs"), client:
        policy_version_id: UUID | None = None
        if policy is not None:
            policy_version_id = _resolve_policy_version_id(client, policy)
        runs = client.list_assay_runs(policy_version_id=policy_version_id)

    if json_output:
        emit_json([r.model_dump(mode="json") for r in runs])
        return

    if not runs:
        console.print("[yellow]No assay runs found.[/yellow]")
        return

    table = Table(title="Assay Runs", box=box.SIMPLE_HEAVY, show_lines=False, pad_edge=False)
    table.add_column("ID", style="dim", overflow="fold")
    table.add_column("Status")
    table.add_column("Name")
    table.add_column("Jobs", justify="right")
    table.add_column("Compat")
    table.add_column("Created", style="dim")

    for run in runs:
        table.add_row(
            str(run.id)[:8] + "...",
            _format_status(run.status),
            run.name or "—",
            f"{run.completed_jobs}/{run.total_jobs}",
            run.compat_version or "—",
            run.created_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)


@assay_app.command(name="results", help="Get scoring results for an assay run.")
def assay_results(
    policy_or_run_id: str = typer.Argument(
        ..., metavar="POLICY_OR_RUN_ID", help="Assay run UUID, or policy name (name[:version])."
    ),
    login_server: str = _LOGIN_SERVER_OPTION,
    server: str = _SERVER_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    client = _get_authed_client(login_server, server)
    with cli_http_errors("Assay results"), client:
        try:
            run_id = UUID(policy_or_run_id)
        except ValueError:
            policy_version_id = _resolve_policy_version_id(client, policy_or_run_id)
            runs = client.list_assay_runs(policy_version_id=policy_version_id)
            if not runs:
                console.print(f"[yellow]No assay runs found for policy {policy_or_run_id!r}.[/yellow]")
                return
            scorable = [r for r in runs if r.status != AssayStatus.pending]
            if not scorable:
                console.print(
                    f"[yellow]Latest run is still pending ({runs[0].id}). No completed runs available.[/yellow]"
                )
                return
            run_id = scorable[0].id

        results = client.get_assay_results(run_id)

    if json_output:
        emit_json(results.model_dump(mode="json"))
        return

    console.print(f"\n[bold]Assay Results[/bold] — {run_id}")
    console.print(f"Status: {_format_status(results.status)}\n")

    if results.axes:
        table = Table(title="Axis Scores", box=box.SIMPLE_HEAVY, pad_edge=False)
        table.add_column("Axis")
        table.add_column("Score", justify="right")
        for axis in results.axes:
            table.add_row(axis.axis.value, f"{axis.score:.2f}")
        console.print(table)
        console.print()

    if results.missions:
        table = Table(title="Per-Mission Metrics", box=box.SIMPLE_HEAVY, pad_edge=False)
        table.add_column("Mission")
        table.add_column("Reward Var", justify="right")
        table.add_column("Non-Zero %", justify="right")
        table.add_column("Timeout %", justify="right")
        table.add_column("Move Success", justify="right")
        table.add_column("Action Failed", justify="right")
        table.add_column("Stuck Steps", justify="right")
        for name, m in results.missions.items():
            table.add_row(
                name,
                f"{m.reward_variance:.3f}",
                f"{m.non_zero_episode_pct:.1f}",
                f"{m.timeout_rate:.1f}",
                f"{m.mean_move_success:.0f}",
                f"{m.mean_action_failed:.0f}",
                f"{m.mean_stuck_steps:.1f}",
            )
        console.print(table)


@assay_app.command(name="submit", help="Submit an assay run for a policy.")
def assay_submit(
    policy: str = typer.Argument(..., metavar="POLICY", help="Policy name[:version] or UUID."),
    source: str = typer.Option(
        "cvc_evals",
        "--source",
        help="cogames mission set name to run (cvc_evals, role_specific_evals, …).",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Optional label for this assay run (used for deduplication).",
    ),
    compat_version: Optional[str] = typer.Option(
        None,
        "--compat-version",
        "-c",
        help="Compat version override.",
    ),
    episodes: int = typer.Option(3, "--episodes", "-e", help="Episodes per mission."),
    max_steps: int = typer.Option(10000, "--max-steps", help="Max steps per episode."),
    watch: bool = typer.Option(False, "--watch", "-w", help="Poll until the run completes."),
    login_server: str = _LOGIN_SERVER_OPTION,
    server: str = _SERVER_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    if not json_output:
        console.print(f"[dim]Building missions from:[/dim] {source}")

    mission_specs = _build_mission_specs(source, episodes, max_steps)

    if not json_output:
        console.print(
            f"[dim]{len(mission_specs)} mission specs, name={name!r}, compat={compat_version or 'none'}[/dim]"
        )

    client = _get_authed_client(login_server, server)
    with client:
        with cli_http_errors("Policy lookup"):
            policy_version_id = _resolve_policy_version_id(client, policy)
        try:
            resp = client.create_assay_run(
                policy_version_id=policy_version_id,
                missions=mission_specs,
                name=name,
                compat_version=compat_version,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 409:
                detail = exc.response.json().get("detail", "Active assay run with this name already exists")
                console.print(f"[yellow]Skipped:[/yellow] {detail}")
                return
            raise

        if json_output:
            emit_json(resp.model_dump(mode="json"))
            return

        console.print(f"[bold]Assay run created:[/bold] {resp.id}")
        console.print(f"Status: {_format_status(resp.status)}")

        if watch:
            console.print("\n[dim]Watching for completion (Ctrl-C to stop)…[/dim]")
            run_id = resp.id
            while True:
                time.sleep(10)
                run = client.get_assay_run(run_id)
                console.print(f"  {_format_status(run.status)} — {run.completed_jobs}/{run.total_jobs} jobs completed")
                if run.status in (AssayStatus.completed, AssayStatus.failed):
                    # Already materialized (e.g. dispatch failure)
                    break
                if run.completed_jobs + run.failed_jobs >= run.total_jobs:
                    # All jobs finished — finalize to materialize terminal status
                    run = client.finalize_assay_run(run_id)
                    break
            if run.status == AssayStatus.completed:
                console.print(f"\n[green]Done![/green] Run ID: {run_id}")
            else:
                console.print(f"\n[red]Failed:[/red] {run.error or 'unknown error'}")
                raise typer.Exit(1)
