"""Policy submission command for CoGames."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx
import typer

from cogames.cli.base import console
from cogames.cli.client import TournamentServerClient
from cogames.cli.login import DEFAULT_COGAMES_SERVER
from cogames.cli.policy import PolicySpec, get_policy_spec
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.policy.submission import POLICY_SPEC_FILENAME, SubmissionPolicySpec
from mettagrid.runner.episode_runner import run_episode_isolated
from mettagrid.runner.types import EpisodeSpec
from mettagrid.util.uri_resolvers.schemes import localize_uri, parse_uri

DEFAULT_SUBMIT_SERVER = "https://api.observatory.softmax-research.net"
RESULTS_URL = "https://www.softmax.com/alignmentleague"


@dataclass
class UploadResult:
    policy_version_id: uuid.UUID
    name: str
    version: int
    pools: list[str] | None = None


def _resolve_path_within_cwd(path_str: str, cwd: Path) -> Path:
    """Resolve a path and return it relative to CWD. Raises if path escapes CWD."""
    raw_path = Path(path_str).expanduser()
    resolved = raw_path.resolve() if raw_path.is_absolute() else (cwd / raw_path).resolve()
    if not resolved.is_relative_to(cwd):
        console.print(f"[red]Error:[/red] Path must be within the current directory: {path_str}")
        raise ValueError(f"Path escapes CWD: {path_str}")
    return resolved.relative_to(cwd)


def validate_paths(paths: list[str]) -> list[Path]:
    """Validate paths exist and are within CWD, return them as relative paths."""
    cwd = Path.cwd().resolve()
    validated_paths = []
    for path_str in paths:
        relative = _resolve_path_within_cwd(path_str, cwd)
        resolved = cwd / relative
        if not resolved.exists():
            console.print(f"[red]Error:[/red] Path does not exist: {path_str}")
            raise FileNotFoundError(f"Path not found: {path_str}")
        validated_paths.append(relative)
    return validated_paths


def _zip_directory_to(src: Path, dest: Path) -> None:
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in src.rglob("*"):
            if file_path.is_file():
                zipf.write(file_path, arcname=file_path.relative_to(src))


def _collect_ancestor_init_files(include_files: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for path in include_files:
        parent = path.parent
        while parent != Path(".") and parent != parent.parent:
            init = parent / "__init__.py"
            if init.is_file():
                found.add(init)
            parent = parent.parent
    return sorted(found)


def create_submission_zip(
    include_files: list[Path],
    policy_spec: PolicySpec,
    setup_script: str | None = None,
) -> Path:
    """Create a zip file containing all include-files.

    Maintains directory structure exactly as provided.
    Returns path to created zip file.
    """
    zip_fd, zip_path = tempfile.mkstemp(suffix=".zip", prefix="cogames_submission_")
    os.close(zip_fd)

    submission_spec = SubmissionPolicySpec(
        class_path=policy_spec.class_path,
        data_path=policy_spec.data_path,
        init_kwargs=policy_spec.init_kwargs,
        setup_script=setup_script,
    )

    all_files: dict[str, Path] = {}
    for init_path in _collect_ancestor_init_files(include_files):
        all_files[str(init_path)] = init_path
    for file_path in include_files:
        if file_path.is_dir():
            for root, _, files in os.walk(file_path):
                for file in files:
                    full = Path(root) / file
                    all_files[str(full)] = full
        else:
            all_files[str(file_path)] = file_path

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(data=submission_spec.model_dump_json(), zinfo_or_arcname=POLICY_SPEC_FILENAME)
        for arcname, path in all_files.items():
            zipf.write(path, arcname=arcname)

    return Path(zip_path)


def create_bundle(
    ctx: typer.Context,
    policy: str,
    output: Path,
    include_files: list[str] | None = None,
    init_kwargs: dict[str, str] | None = None,
    setup_script: str | None = None,
) -> Path:
    # TODO: Unify the two paths below. For URI inputs, extract the PolicySpec from the
    # bundle so we can apply init_kwargs/include_files/setup_script, then re-zip.
    local = localize_uri(policy) if parse_uri(policy, allow_none=True, default_scheme=None) else None
    if local is not None:
        if init_kwargs or include_files or setup_script:
            console.print("[red]Error:[/red] Extra files/kwargs are not supported with bundle URIs.")
            raise typer.Exit(1)
        console.print(f"[dim]Packaging existing bundle: {local}[/dim]")
        if local.is_dir():
            _zip_directory_to(local, output)
        else:
            shutil.copy2(local, output)
        console.print(f"[dim]Bundle size: {output.stat().st_size / 1024:.0f} KB[/dim]")
        return output

    policy_spec = get_policy_spec(ctx, policy)
    console.print(f"[dim]Policy class: {policy_spec.class_path}[/dim]")

    if init_kwargs:
        merged_kwargs = {**policy_spec.init_kwargs, **init_kwargs}
        policy_spec = PolicySpec(
            class_path=policy_spec.class_path,
            data_path=policy_spec.data_path,
            init_kwargs=merged_kwargs,
        )

    if policy_spec.init_kwargs:
        console.print(f"[dim]Init kwargs: {policy_spec.init_kwargs}[/dim]")

    cwd = Path.cwd().resolve()
    if policy_spec.data_path:
        data_rel = str(_resolve_path_within_cwd(policy_spec.data_path, cwd))
        policy_spec = PolicySpec(
            class_path=policy_spec.class_path,
            data_path=data_rel,
            init_kwargs=policy_spec.init_kwargs,
        )
        console.print(f"[dim]Data path: {data_rel}[/dim]")

    setup_script_rel: str | None = None
    if setup_script:
        setup_script_rel = str(_resolve_path_within_cwd(setup_script, cwd))
        console.print(f"[dim]Setup script: {setup_script_rel}[/dim]")

    files_to_include = []
    if policy_spec.data_path:
        files_to_include.append(policy_spec.data_path)
    if setup_script_rel:
        files_to_include.append(setup_script_rel)
    if include_files:
        files_to_include.extend(include_files)

    validated_paths: list[Path] = []
    if files_to_include:
        validated_paths = validate_paths(files_to_include)
        console.print(f"[dim]Including {len(validated_paths)} file(s)[/dim]")

    tmp_zip = create_submission_zip(validated_paths, policy_spec, setup_script=setup_script_rel)
    shutil.move(str(tmp_zip), str(output))
    console.print(f"[dim]Bundle size: {output.stat().st_size / 1024:.0f} KB[/dim]")
    return output


def validate_bundle(policy_uri: str, env_cfg: MettaGridConfig) -> None:
    """Validate a policy bundle by running a short episode in process isolation."""
    env_cfg.game.max_steps = 10

    spec = EpisodeSpec(
        policy_uris=[policy_uri],
        assignments=[0] * env_cfg.game.num_agents,
        env=env_cfg,
        seed=42,
        max_action_time_ms=10000,
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as results_file:
        res = run_episode_isolated(spec, Path(results_file.name))
        console.print(f"[dim]Ran for {res.steps} steps[/dim]")

        non_noop_actions = sum(
            v for k, v in res.stats["agent"][0].items() if k.startswith("action.") and ".noop." not in k
        )
        if non_noop_actions == 0:
            console.print("[yellow]Warning: Policy took no actions (all no-ops)[/yellow]")
            raise typer.Exit(1)


def upload_submission(
    client: TournamentServerClient,
    zip_path: Path,
    submission_name: str,
    season: str | None = None,
) -> UploadResult | None:
    """Upload submission to CoGames backend using a presigned S3 URL."""
    console.print("[bold]Uploading[/bold]")

    presigned_data = client.get_presigned_upload_url()
    upload_url = presigned_data.get("upload_url")
    upload_id = presigned_data.get("upload_id")

    if not upload_url or not upload_id:
        raise ValueError("Upload URL missing from response")

    console.print("[dim]Uploading to storage...[/dim]")

    with open(zip_path, "rb") as f:
        upload_response = httpx.put(
            upload_url,
            content=f,
            headers={"Content-Type": "application/zip"},
            timeout=600.0,
        )
    upload_response.raise_for_status()

    if not season:
        console.print("[dim]Uploading policy...[/dim]")
    else:
        console.print(f"[dim]Uploading policy and submitting to season {season}...[/dim]")

    result = client.complete_policy_upload(upload_id, submission_name, season=season)
    submission_id = result.get("id")
    name = result.get("name")
    version = result.get("version")
    pools = result.get("pools")
    if submission_id is None or name is None or version is None:
        raise ValueError("Missing fields in response")
    try:
        return UploadResult(
            policy_version_id=uuid.UUID(str(submission_id)),
            name=name,
            version=version,
            pools=pools,
        )
    except ValueError as exc:
        raise ValueError(f"Invalid submission ID returned: {submission_id}") from exc


def upload_policy(
    ctx: typer.Context,
    policy: str,
    name: str,
    include_files: list[str] | None = None,
    login_server: str = DEFAULT_COGAMES_SERVER,
    server: str = DEFAULT_SUBMIT_SERVER,
    init_kwargs: dict[str, str] | None = None,
    dry_run: bool = False,
    skip_validation: bool = False,
    setup_script: str | None = None,
    season: str | None = None,
) -> UploadResult | None:
    if dry_run:
        console.print("[dim]Dry run mode - no upload[/dim]\n")

    client = TournamentServerClient.from_login(server_url=server, login_server=login_server)
    if not client:
        return None

    with tempfile.TemporaryDirectory(prefix="cogames_bundle_") as tmp_dir:
        zip_path = Path(tmp_dir) / "bundle.zip"

        create_bundle(
            ctx=ctx,
            policy=policy,
            output=zip_path,
            include_files=include_files,
            init_kwargs=init_kwargs,
            setup_script=setup_script,
        )

        if not skip_validation:
            cmd = [
                sys.executable,
                "-m",
                "cogames",
                "validate-bundle",
                "--policy",
                zip_path.as_uri(),
                "--server",
                server,
            ]
            if season:
                cmd.extend(["--season", season])
            result = subprocess.run(cmd, text=True, timeout=300)
            if result.returncode != 0:
                console.print("[red]Validation failed[/red]")
                return None
            console.print("[green]Validation passed[/green]")
        else:
            console.print("[dim]Skipping validation[/dim]")

        if dry_run:
            console.print("[green]Dry run complete[/green]")
            return None

        with client:
            result = upload_submission(client, zip_path, name, season=season)
        if not result:
            console.print("\n[red]Upload failed.[/red]")
            return None

        return result
