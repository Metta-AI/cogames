"""Policy bundle creation for CoGames."""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import typer

from cogames.cli.base import console
from cogames.cli.policy import PolicySpec, get_policy_spec
from mettagrid.policy.prepare_policy_spec import extract_submission_archive, find_package_source_root
from mettagrid.policy.submission import POLICY_SPEC_FILENAME, SubmissionPolicySpec, write_submission_policy_spec
from mettagrid.util.uri_resolvers.schemes import localize_uri, parse_uri

_METTA_POLICY_CLASS_PREFIX = "metta.agent."


def _resolve_path_within_cwd(path_str: str, cwd: Path) -> Path:
    """Resolve a path and return it relative to CWD. Raises if path escapes CWD."""
    raw_path = Path(path_str).expanduser()
    resolved = raw_path.resolve() if raw_path.is_absolute() else (cwd / raw_path).resolve()
    if not resolved.is_relative_to(cwd):
        console.print(f"[red]Error:[/red] Path must be within the current directory: {path_str}")
        raise ValueError(f"Path escapes CWD: {path_str}")
    return resolved.relative_to(cwd)


def _existing_local_bundle_path(policy: str) -> Path | None:
    if "://" in policy:
        return None

    # Preserve NAME policy parsing unless the input is unambiguously a local path.
    if not policy.endswith(".zip") and not Path(policy).is_absolute() and not policy.startswith((".", "~")):
        return None

    candidate = Path(policy).expanduser()
    if not candidate.exists():
        return None
    resolved = candidate.resolve()
    if resolved.is_dir() or resolved.suffix == ".zip":
        return resolved
    return None


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


def _bundle_target_for_include(path: Path, bundle_root: Path, class_path: str) -> Path:
    module_path = class_path.rsplit(".", 1)[0]
    if "." not in module_path and path.name == f"{module_path}.py":
        return bundle_root / path.name
    return bundle_root / path


def _copy_include_paths_into_bundle(paths: list[Path], cwd: Path, bundle_root: Path, class_path: str) -> None:
    for path in paths:
        source = cwd / path
        target = _bundle_target_for_include(path, bundle_root, class_path)
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
        policy_spec = policy_spec.model_copy(update={"init_kwargs": merged_kwargs})

    if policy_spec.init_kwargs:
        console.print(f"[dim]Init kwargs: {policy_spec.init_kwargs}[/dim]")

    files_to_include: list[str] = []
    if policy_spec.data_path:
        data_rel = str(_resolve_path_within_cwd(policy_spec.data_path, cwd))
        policy_spec = policy_spec.model_copy(update={"data_path": data_rel})
        files_to_include.append(data_rel)
        console.print(f"[dim]Data path: {data_rel}[/dim]")

    submission_spec = SubmissionPolicySpec.model_validate(policy_spec.model_dump())
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

    submission_spec = SubmissionPolicySpec.model_validate(
        {
            **policy_spec.model_dump(),
            "setup_script": setup_script,
        }
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
    local_bundle_path = _existing_local_bundle_path(policy)
    bundle_source = local_bundle_path.as_uri() if local_bundle_path is not None else policy
    is_uri = parse_uri(bundle_source, allow_none=True, default_scheme=None) is not None
    files_to_include = list(include_files or [])

    with tempfile.TemporaryDirectory(prefix="cogames_bundle_build_") as tmp_dir:
        bundle_root = Path(tmp_dir) / "bundle"
        bundle_root.mkdir()

        if is_uri:
            submission_spec = _prepare_submission_spec_from_uri(bundle_source, bundle_root, init_kwargs)
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
            _copy_include_paths_into_bundle(include_with_ancestors, cwd, bundle_root, submission_spec.class_path)

        has_embedded_package_root = find_package_source_root(bundle_root, submission_spec.class_path) is not None
        if (
            submission_spec.class_path.startswith(_METTA_POLICY_CLASS_PREFIX)
            and submission_spec.setup_script is None
            and not has_embedded_package_root
        ):
            console.print(
                "[red]Error:[/red] Build a submission bundle that includes the runtime code your policy imports and "
                "a setup script before uploading this checkpoint."
            )
            console.print(
                "\n[dim]Generic pattern:[/dim]\n"
                "[cyan]cogames create-bundle -p <checkpoint-or-policy> -o submission.zip "
                "-f <runtime-path> ... --setup-script <setup.py>[/cyan]\n"
            )
            if (cwd / "agent/COGAMES_SUBMISSION.md").is_file():
                console.print("\n[dim]Metta repo guide:[/dim] agent/COGAMES_SUBMISSION.md")
            raise typer.Exit(1)

        write_submission_policy_spec(bundle_root / POLICY_SPEC_FILENAME, submission_spec)
        _zip_directory_to(bundle_root, output)

    console.print(f"[dim]Bundle size: {output.stat().st_size / 1024:.0f} KB[/dim]")
    return output
