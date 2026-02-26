"""Policy submission command for CoGames."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import typer

from cogames.cli.base import console
from cogames.cli.client import TournamentServerClient
from cogames.cli.login import DEFAULT_COGAMES_SERVER
from cogames.cli.policy import PolicySpec, get_policy_spec
from mettagrid.policy.prepare_policy_spec import extract_submission_archive
from mettagrid.policy.submission import POLICY_SPEC_FILENAME, SubmissionPolicySpec, write_submission_policy_spec
from mettagrid.runner.types import PureSingleEpisodeResult
from mettagrid.util.uri_resolvers.schemes import localize_uri, parse_uri

DEFAULT_SUBMIT_SERVER = "https://api.observatory.softmax-research.net"
DEFAULT_EPISODE_RUNNER_IMAGE = "ghcr.io/metta-ai/episode-runner:latest"
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


def _copy_tree_to(src: Path, dest: Path) -> None:
    for file_path in src.rglob("*"):
        if not file_path.is_file():
            continue
        target = dest / file_path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, target)


def _materialize_bundle_from_local(local: Path, bundle_root: Path) -> SubmissionPolicySpec:
    if local.is_dir():
        _copy_tree_to(local, bundle_root)
    else:
        extract_submission_archive(local, bundle_root)
    return _load_submission_spec(bundle_root)


def _load_submission_spec(bundle_root: Path) -> SubmissionPolicySpec:
    spec_path = bundle_root / POLICY_SPEC_FILENAME
    if not spec_path.exists():
        raise FileNotFoundError(f"{POLICY_SPEC_FILENAME} not found in bundle: {bundle_root}")
    return SubmissionPolicySpec.model_validate_json(spec_path.read_text())


def _copy_include_paths_into_bundle(paths: list[Path], cwd: Path, bundle_root: Path) -> None:
    for path in paths:
        source = cwd / path
        target = bundle_root / path
        if source.is_dir():
            _copy_tree_to(source, target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _prepare_submission_spec_from_policy(
    ctx: typer.Context,
    policy: str,
    cwd: Path,
    init_kwargs: dict[str, str] | None,
) -> tuple[SubmissionPolicySpec, list[str]]:
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

    files_to_include: list[str] = []
    if policy_spec.data_path:
        data_rel = str(_resolve_path_within_cwd(policy_spec.data_path, cwd))
        policy_spec = PolicySpec(
            class_path=policy_spec.class_path,
            data_path=data_rel,
            init_kwargs=policy_spec.init_kwargs,
        )
        files_to_include.append(data_rel)
        console.print(f"[dim]Data path: {data_rel}[/dim]")

    submission_spec = SubmissionPolicySpec(
        class_path=policy_spec.class_path,
        data_path=policy_spec.data_path,
        init_kwargs=policy_spec.init_kwargs,
        setup_script=None,
    )
    return submission_spec, files_to_include


def _prepare_submission_spec_from_uri(
    policy: str,
    bundle_root: Path,
    init_kwargs: dict[str, str] | None,
) -> SubmissionPolicySpec:
    local = localize_uri(policy)
    if local is None:
        raise ValueError(f"Cannot localize policy URI: {policy}")
    console.print(f"[dim]Policy bundle: {local}[/dim]")
    submission_spec = _materialize_bundle_from_local(local, bundle_root)
    if init_kwargs:
        submission_spec.init_kwargs.update(init_kwargs)
        console.print(f"[dim]Init kwargs: {submission_spec.init_kwargs}[/dim]")
    return submission_spec


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
                    if not full.exists():
                        continue
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
    cwd = Path.cwd().resolve()
    is_uri = parse_uri(policy, allow_none=True, default_scheme=None) is not None
    files_to_include = list(include_files or [])

    with tempfile.TemporaryDirectory(prefix="cogames_bundle_build_") as tmp_dir:
        bundle_root = Path(tmp_dir) / "bundle"
        bundle_root.mkdir()

        if is_uri:
            submission_spec = _prepare_submission_spec_from_uri(policy, bundle_root, init_kwargs)
        else:
            submission_spec, policy_files = _prepare_submission_spec_from_policy(ctx, policy, cwd, init_kwargs)
            files_to_include.extend(policy_files)

        if setup_script:
            setup_script_rel = str(_resolve_path_within_cwd(setup_script, cwd))
            files_to_include.append(setup_script_rel)
            submission_spec.setup_script = setup_script_rel
            console.print(f"[dim]Setup script: {setup_script_rel}[/dim]")

        validated_paths: list[Path] = []
        if files_to_include:
            validated_paths = validate_paths(files_to_include)
            console.print(f"[dim]Including {len(validated_paths)} file(s)[/dim]")

        if validated_paths:
            include_with_ancestors = validated_paths + _collect_ancestor_init_files(validated_paths)
            _copy_include_paths_into_bundle(include_with_ancestors, cwd, bundle_root)

        write_submission_policy_spec(bundle_root / POLICY_SPEC_FILENAME, submission_spec)
        _zip_directory_to(bundle_root, output)

    console.print(f"[dim]Bundle size: {output.stat().st_size / 1024:.0f} KB[/dim]")
    return output


def _validation_job_spec(policy_uri: str, config_data: dict[str, Any]) -> dict[str, Any]:
    env_cfg = dict(config_data)
    env_cfg["game"] = dict(env_cfg["game"])
    env_cfg["game"]["max_steps"] = 10
    return {
        "policy_uris": [policy_uri],
        "assignments": [0] * env_cfg["game"]["num_agents"],
        "env": env_cfg,
        "seed": 42,
        "max_action_time_ms": 10000,
    }


def _check_results(res: PureSingleEpisodeResult) -> None:
    console.print(f"[dim]Ran for {res.steps} steps[/dim]")
    non_noop_actions = sum(v for k, v in res.stats["agent"][0].items() if k.startswith("action.") and ".noop." not in k)
    if non_noop_actions == 0:
        console.print("[yellow]Warning: Policy took no actions (all no-ops)[/yellow]")
        raise typer.Exit(1)


def ensure_docker_daemon_access() -> None:
    docker = shutil.which("docker")
    if docker is None:
        console.print("[red]Docker CLI not found on PATH.[/red]")
        console.print("[dim]Install Docker and ensure the `docker` command is available, then retry.[/dim]")
        raise typer.Exit(1)

    try:
        result = subprocess.run([docker, "info"], capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        console.print("[red]Timed out while checking Docker daemon access.[/red]")
        console.print(
            "[dim]Docker did not respond to `docker info` within 15 seconds. "
            "Check DOCKER_HOST and daemon health, then retry.[/dim]"
        )
        raise typer.Exit(1) from None

    if result.returncode != 0:
        console.print("[red]Cannot access the Docker daemon.[/red]")
        detail = (result.stderr or result.stdout).strip()
        if detail:
            console.print(f"[red]{detail}[/red]")
        console.print(
            "[dim]Start Docker Desktop (or your Docker service) and ensure your user can access "
            "the daemon socket.[/dim]"
        )
        raise typer.Exit(1)


def validate_bundle_docker(policy_uri: str, config_data: dict[str, Any], image: str) -> None:
    local_path = localize_uri(policy_uri)
    if local_path is None:
        raise ValueError(f"Cannot localize policy URI: {policy_uri}")

    if local_path.is_dir():
        container_policy_uri = "file:///workspace/policy"
        container_mount_target = "/workspace/policy"
    else:
        container_policy_uri = f"file:///workspace/policy/{local_path.name}"
        container_mount_target = f"/workspace/policy/{local_path.name}"

    job_spec = _validation_job_spec(container_policy_uri, config_data)

    with tempfile.TemporaryDirectory(prefix="cogames_docker_validate_") as workspace:
        spec_path = Path(workspace) / "spec.json"
        results_path = Path(workspace) / "results.json"
        spec_path.write_text(json.dumps(job_spec))

        cmd = [
            "docker",
            "run",
            "--rm",
            "--platform",
            "linux/amd64",
            "-e",
            "JOB_SPEC_URI=file:///workspace/io/spec.json",
            "-e",
            "RESULTS_URI=file:///workspace/io/results.json",
            "-v",
            f"{local_path.resolve()}:{container_mount_target}:ro",
            "-v",
            f"{workspace}:/workspace/io:rw",
            image,
        ]

        console.print(f"[dim]Pulling latest image ({image})...[/dim]")
        subprocess.run(["docker", "pull", "--platform", "linux/amd64", image], text=True, timeout=300)

        console.print(f"[dim]Running validation in Docker ({image})...[/dim]")
        result = subprocess.run(cmd, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"Docker validation failed (exit {result.returncode})")

        if not results_path.exists():
            raise RuntimeError("Docker validation produced no results file")

        res = PureSingleEpisodeResult.model_validate_json(results_path.read_text())
        _check_results(res)


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

    # Server checks actual S3 object size (catches old clients or misreported sizes). Only
    # the server knows the real size, so this try/except can't be replaced by a client-side check.
    try:
        result = client.complete_policy_upload(upload_id, submission_name, season=season)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 413:
            detail = e.response.json().get("detail", "Policy too large")
            console.print(f"[red]{detail}[/red]")
            raise typer.Exit(1) from None
        raise
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
    include_hidden: bool = False,
    image: str = DEFAULT_EPISODE_RUNNER_IMAGE,
) -> UploadResult | None:
    if dry_run:
        console.print("[dim]Dry run mode - no upload[/dim]\n")

    client = TournamentServerClient.from_login(server_url=server, login_server=login_server)
    if not client:
        raise typer.Exit(1)

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
            if include_hidden:
                cmd.append("--include-hidden")
            if image != DEFAULT_EPISODE_RUNNER_IMAGE:
                cmd.extend(["--image", image])
            result = subprocess.run(cmd, text=True, timeout=300)
            if result.returncode != 0:
                console.print("[red]Validation failed[/red]")
                raise typer.Exit(1)
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
            raise typer.Exit(1)

        return result
