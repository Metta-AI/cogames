"""
`cogames docsync` is a dev-only tool to sync cogames docs between .ipynb, .py, and .md formats.

Tests for this module are in tests/docsync/
"""

from pathlib import Path
from typing import Annotated

import typer

from cogames.cli.docsync._nb_md_sync import convert_nb_to_md
from cogames.cli.docsync._nb_py_sync import convert_nb_to_py, convert_py_to_nb
from cogames.cli.docsync._three_way_sync import get_stem_paths, is_stem_in_sync_three_way, sync_stem_three_way
from cogames.cli.docsync._utils import get_cogames_root

app = typer.Typer(
    help="""Sync cogames docs between .ipynb, .py, and .md formats

Directives (first line of code cell):
  # <<hide>>            - skip entire cell (code + output)
  # <<hide-input>>      - skip code, show output
  # <<hide-output>>     - show code, skip output
  # <<collapse-input>>  - wrap code in <details>
  # <<collapse-output>> - wrap output in <details>

Placeholders (in markdown cells):
  <<colab-link>>        - replaced with Colab URL for this notebook
"""
)


@app.command()
def nb_to_md(
    nb_path: Annotated[Path, typer.Argument(help="Path to .ipynb file")],
    skip_rerun: Annotated[bool, typer.Option("--skip-rerun", help="Skip re-running notebook")] = False,
    cogames_root_raw: Annotated[
        Path | None, typer.Option("--cogames-root", hidden=True, help="Override root path (for testing)")
    ] = None,
):
    """Convert .ipynb to .md (reruns notebook by default to update outputs)."""

    if not nb_path.exists():
        typer.echo(f"Error: {nb_path} not found", err=True)
        raise typer.Exit(1)
    if nb_path.suffix != ".ipynb":
        typer.echo(f"Error: expected .ipynb file, got {nb_path.suffix}", err=True)
        raise typer.Exit(1)

    cogames_root = cogames_root_raw or get_cogames_root()
    typer.echo(f"Exporting {nb_path.name}...")
    convert_nb_to_md(nb_path=nb_path, should_rerun=not skip_rerun, cogames_root=cogames_root)


@app.command()
def nb_to_py(
    nb_path: Annotated[Path, typer.Argument(help="Path to .ipynb file")],
):
    """Convert notebook to .py percent format (ignores outputs, keeps code)."""

    if not nb_path.exists():
        typer.echo(f"Error: {nb_path} not found", err=True)
        raise typer.Exit(1)
    if nb_path.suffix != ".ipynb":
        typer.echo(f"Error: expected .ipynb file, got {nb_path.suffix}", err=True)
        raise typer.Exit(1)

    py_path = nb_path.with_suffix(".py")
    typer.echo(f"Converting {nb_path.name} to .py...")
    convert_nb_to_py(nb_path=nb_path, py_path=py_path)
    typer.echo(f"Created {py_path.name}")


@app.command()
def py_to_nb(
    py_path: Annotated[Path, typer.Argument(help="Path to .py file")],
    skip_rerun: Annotated[bool, typer.Option("--skip-rerun", help="Skip re-running notebook")] = False,
):
    """Convert .py percent format to .ipynb (runs notebook by default to populate outputs)."""

    if not py_path.exists():
        typer.echo(f"Error: {py_path} not found", err=True)
        raise typer.Exit(1)
    if py_path.suffix != ".py":
        typer.echo(f"Error: expected .py file, got {py_path.suffix}", err=True)
        raise typer.Exit(1)

    nb_path = py_path.with_suffix(".ipynb")
    typer.echo(f"Converting {py_path.name} to .ipynb...")
    convert_py_to_nb(py_path=py_path, nb_path=nb_path, should_rerun=not skip_rerun)
    typer.echo(f"Created {nb_path.name}")


@app.command()
def check(
    skip_rerun: Annotated[bool, typer.Option("--skip-rerun", help="Skip re-running notebooks")] = False,
    cogames_root_raw: Annotated[
        Path | None, typer.Option("--cogames-root", hidden=True, help="Override root path (for testing)")
    ] = None,
):
    """Verify that .py, .ipynb, and .md files are all in sync (without modifying files)."""
    cogames_root = cogames_root_raw or get_cogames_root()
    stem_paths = get_stem_paths(cogames_root=cogames_root)
    should_rerun = not skip_rerun

    typer.echo(f"Checking {len(stem_paths)} notebook(s)...")

    errors: list[str] = []
    for stem_path in stem_paths:
        stem_errors = is_stem_in_sync_three_way(
            stem_path=stem_path, should_rerun=should_rerun, cogames_root=cogames_root
        )
        if stem_errors:
            typer.echo(f"  ✗ {stem_path.name}")
        else:
            typer.echo(f"  ✓ {stem_path.name}")
        errors.extend(stem_errors)

    if errors:
        typer.echo("\nNotebook documentation is out of sync!", err=True)
        for error in errors:
            typer.echo(f"  {error}", err=True)
        typer.echo("\nTo fix: run 'cogames docsync all'", err=True)
        raise typer.Exit(1)

    typer.echo("All notebooks are in sync!")


@app.command()
def all(
    skip_rerun: Annotated[bool, typer.Option("--skip-rerun", help="Skip re-running notebooks")] = False,
    cogames_root_raw: Annotated[
        Path | None, typer.Option("--cogames-root", hidden=True, help="Override root path (for testing)")
    ] = None,
):
    """Resyncs README and tutorials/ .py and .ipynb files with each other and exports to .md files."""
    cogames_root = cogames_root_raw or get_cogames_root()
    stem_paths = get_stem_paths(cogames_root=cogames_root)
    should_rerun = not skip_rerun

    typer.echo(f"Found {len(stem_paths)} notebook(s) to process")

    for stem_path in stem_paths:
        stem_path_relative = stem_path.relative_to(cogames_root)
        typer.echo(f"Processing {stem_path_relative}...")
        sync_stem_three_way(
            stem_path=stem_path,
            should_rerun=should_rerun,
            cogames_root=cogames_root,
        )

    typer.echo(f"Done! Processed {len(stem_paths)} notebook(s)")


if __name__ == "__main__":
    app()
