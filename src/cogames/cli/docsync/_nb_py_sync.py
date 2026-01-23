"""Convert notebook to/from Python using jupytext."""

from datetime import datetime
from pathlib import Path
from typing import Literal

import jupytext
import questionary
import typer

from cogames.cli.docsync._utils import clean_notebook_metadata, lint_py_file, py_nb_content_equal, run_notebook


def convert_nb_to_py(*, nb_path: Path, py_path: Path) -> None:
    """Convert .ipynb to .py percent format using jupytext."""
    nb = jupytext.read(nb_path)
    jupytext.write(nb, py_path, fmt="py:percent")
    lint_py_file(py_path)


def convert_py_to_nb(*, py_path: Path, nb_path: Path, should_rerun: bool) -> None:
    """Convert .py to .ipynb using jupytext, optionally execute."""
    nb = jupytext.read(py_path)
    jupytext.write(nb, nb_path)

    if should_rerun:
        run_notebook(nb_path=nb_path)

    clean_notebook_metadata(nb_path=nb_path)


def _format_mtime(path: Path) -> str:
    """Format file mtime with millisecond precision."""
    mtime = path.stat().st_mtime
    dt = datetime.fromtimestamp(mtime)
    ms = int((mtime % 1) * 1000)
    return f"{dt.strftime('%Y-%m-%d %H:%M:%S')}.{ms:03d}"


def _prompt_sync_direction(*, py_path: Path, nb_path: Path) -> Literal["py", "nb", "skip"]:
    """Prompt user to choose sync direction. Returns 'py', 'nb', or 'skip'."""
    nb_mtime = nb_path.stat().st_mtime
    py_mtime = py_path.stat().st_mtime
    py_newer = py_mtime > nb_mtime

    # Format: ".ipynb → .py     (newer)  2026-01-21 10:30:15.456"
    #         "   .py → .ipynb           2026-01-22 15:45:32.123"
    newer_tag = "(newer)"
    blank_pad = " " * len(newer_tag)
    nb_label = f".ipynb → .py     {newer_tag if not py_newer else blank_pad}  {_format_mtime(nb_path)}"
    py_label = f"   .py → .ipynb  {newer_tag if py_newer else blank_pad}  {_format_mtime(py_path)}"

    choice = questionary.select(
        f"{py_path.stem}.py and {py_path.stem}.ipynb are out of sync. Which should be source of truth?",
        choices=[nb_label, py_label, "Skip"],
    ).ask()

    if choice is None:
        return "skip"
    elif ".py →" in choice:
        return "py"
    elif ".ipynb →" in choice:
        return "nb"
    else:
        return "skip"


def sync_nb_and_py_by_user_choice(*, py_path: Path, nb_path: Path, should_rerun: bool) -> bool:
    """Sync .py/.ipynb with each other based on user selection. Returns whether notebook was re-run.

    Args:
        py_path: Path to .py file
        nb_path: Path to .ipynb file
        should_rerun: Whether to re-run notebook after syncing py->nb
    """

    if py_path.exists() and nb_path.exists():
        # Check if content is already equal (no sync needed)
        if py_nb_content_equal(py_path=py_path, nb_path=nb_path):
            typer.echo(f"  {py_path.name} and {nb_path.name} are already in sync")
            return False

        # Prompt user to choose sync direction
        choice = _prompt_sync_direction(py_path=py_path, nb_path=nb_path)

        # Perform sync based on user choice
        match choice:
            case "py":
                typer.echo("  Syncing .py → .ipynb...")
                convert_py_to_nb(py_path=py_path, nb_path=nb_path, should_rerun=should_rerun)
                typer.echo(f"  Synced {py_path.name} → {nb_path.name}")
                py_path.touch()  # Touch source so it stays "newer"
                return should_rerun
            case "nb":
                typer.echo("  Syncing .ipynb → .py...")
                convert_nb_to_py(nb_path=nb_path, py_path=py_path)
                typer.echo(f"  Synced {nb_path.name} → {py_path.name}")
                nb_path.touch()  # Touch source so it stays "newer"
                return False
            case "skip":
                typer.echo(f"  Skipping {py_path.stem}")
                return False

    elif py_path.exists():
        typer.echo(f"  Only {py_path.name} exists, creating .ipynb...")
        convert_py_to_nb(py_path=py_path, nb_path=nb_path, should_rerun=should_rerun)
        typer.echo(f"  Created {nb_path.name}")
        # Touch the original file (not the created one)
        py_path.touch()
        return should_rerun

    elif nb_path.exists():
        typer.echo(f"  Only {nb_path.name} exists, creating .py...")
        convert_nb_to_py(nb_path=nb_path, py_path=py_path)
        typer.echo(f"  Created {py_path.name}")
        # Touch the original file (not the created one)
        nb_path.touch()
        return False

    else:
        typer.echo(f"  Error: neither {py_path.name} nor {nb_path.name} exist", err=True)
        raise typer.Exit(1)
