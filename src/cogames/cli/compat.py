import importlib.metadata
from pathlib import Path

import typer
from packaging.version import InvalidVersion, Version
from rich.console import Console

from cogames.cli.client import SeasonSummary

console = Console(stderr=True)


def _find_repo_compat_version() -> str | None:
    current = Path.cwd().resolve()
    for parent in [current, *current.parents]:
        if (parent / ".repo-root").exists():
            compat_file = parent / "COMPAT_VERSION"
            if compat_file.exists():
                return compat_file.read_text().strip()
            return None
    return None


def _get_installed_compat() -> str | None:
    raw = importlib.metadata.version("cogames")
    try:
        v = Version(raw)
    except InvalidVersion:
        return None
    if v.dev is not None:
        return _find_repo_compat_version()
    return f"{v.major}.{v.minor}"


def check_compat_version(season: SeasonSummary) -> None:
    if season.compat_version is None:
        return

    installed = _get_installed_compat()
    if installed is None:
        return

    if installed != season.compat_version:
        console.print(
            f'[red]Error:[/red] Season "{season.name}" requires compat version '
            f"{season.compat_version}, but you have cogames {importlib.metadata.version('cogames')} "
            f"(compat {installed}).\n"
            f"Run: [cyan]pip install --upgrade cogames[/cyan]"
        )
        raise typer.Exit(1)
