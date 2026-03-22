"""Three-way sync between .py, .ipynb, and .md files."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import typer

from cogames.cli.docsync._nb_md_sync import convert_nb_to_md, convert_nb_to_md_in_memory
from cogames.cli.docsync._nb_py_sync import convert_nb_to_py, sync_nb_and_py_by_user_choice
from cogames.cli.docsync._utils import files_equal, lint_py_file


def get_stem_paths(*, cogames_root: Path) -> list[Path]:
    """Get all notebook stem paths (README + tutorials/*)."""
    stem_paths: list[Path] = []

    # README notebook at root
    readme_nb_path = cogames_root / "README.ipynb"
    if readme_nb_path.exists():
        stem_paths.append(readme_nb_path.with_suffix(""))

    # Notebooks in tutorials directory
    tutorials_dir = cogames_root / "tutorials"
    if tutorials_dir.exists():
        stem_paths.extend(sorted({nb_path.with_suffix("") for nb_path in tutorials_dir.glob("*.ipynb")}))

    return stem_paths


def sync_stem_three_way(*, stem_path: Path, should_rerun: bool, cogames_root: Path) -> None:
    """Sync .py and .ipynb with each other based on mtime, then export to .md."""

    py_path = stem_path.with_suffix(".py")
    nb_path = stem_path.with_suffix(".ipynb")

    did_rerun = sync_nb_and_py_by_user_choice(
        py_path=py_path,
        nb_path=nb_path,
        should_rerun=should_rerun,
    )

    typer.echo(f"  Exporting {nb_path.name} to markdown...")
    still_should_rerun = should_rerun and not did_rerun
    convert_nb_to_md(nb_path=nb_path, should_rerun=still_should_rerun, cogames_root=cogames_root)
    typer.echo(f"  Exported {nb_path.name} → {nb_path.stem}.md")


def is_stem_in_sync_three_way(*, stem_path: Path, should_rerun: bool, cogames_root: Path) -> list[str]:
    """Check if .py, .ipynb, .md for a stem are in sync. Returns list of errors."""
    errors: list[str] = []
    py_path = stem_path.with_suffix(".py")
    nb_path = stem_path.with_suffix(".ipynb")
    md_path = stem_path.with_suffix(".md")

    # All three files must exist
    if not py_path.exists():
        errors.append(f"{py_path.name} is missing")
    if not md_path.exists():
        errors.append(f"{md_path.name} is missing")
    if not nb_path.exists():
        errors.append(f"{nb_path.name} is missing")
        return errors  # Can't check further without .ipynb

    # Work with a temp copy of the notebook to avoid modifying original
    # Maintain directory structure so relative_to(cogames_root) works
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_relpath = nb_path.relative_to(cogames_root)
        tmp_nb_path = Path(tmpdir) / notebook_relpath
        tmp_nb_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(nb_path, tmp_nb_path)

        # Optionally re-run the notebook (in original working dir so paths work)
        if should_rerun:
            typer.echo(f"  Re-running {nb_path.name}...")
            result = subprocess.run(
                ["jupyter", "execute", str(tmp_nb_path), "--inplace"],
                cwd=nb_path.parent,  # Use original working dir for relative paths
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                errors.append(f"{nb_path.name} failed to execute: {result.stderr}")
                return errors

        # Convert temp .ipynb -> temp .py, compare with existing .py
        if py_path.exists():
            tmp_py_path = Path(tmpdir) / py_path.name
            convert_nb_to_py(nb_path=tmp_nb_path, py_path=tmp_py_path)
            # Copy and format existing .py for fair comparison
            tmp_existing_py = Path(tmpdir) / f"existing_{py_path.name}"
            shutil.copy(py_path, tmp_existing_py)
            lint_py_file(tmp_existing_py)
            if not files_equal(tmp_py_path, tmp_existing_py):
                errors.append(f"{py_path.name} doesn't match {nb_path.name}")

        # Convert temp .ipynb -> temp .md, compare with existing .md and resources
        if md_path.exists():
            md_content, resources = convert_nb_to_md_in_memory(nb_path=tmp_nb_path, cogames_root=Path(tmpdir))
            tmp_md_path = Path(tmpdir) / md_path.name
            tmp_md_path.write_text(md_content)
            if not files_equal(tmp_md_path, md_path):
                errors.append(f"{md_path.name} doesn't match {nb_path.name}")

            # Compare resources (images, etc.)
            output_files = resources["outputs"]
            for filename, generated_data in output_files.items():
                existing_path = nb_path.parent / filename
                if not existing_path.exists():
                    errors.append(f"{filename} is missing")
                elif existing_path.read_bytes() != generated_data:
                    errors.append(f"{filename} doesn't match {nb_path.name}")

    return errors
