from __future__ import annotations

import json
import sys
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import httpx
import typer
from rich.console import Console

console = Console()


def emit_json(payload: Any) -> None:
    """Write machine-parseable JSON directly to stdout, bypassing Rich console.

    Using sys.stdout.write instead of console.print ensures the output is clean
    JSON without Rich markup, terminal width wrapping, or ANSI escape codes that
    would corrupt the output when piped to other programs.
    """
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")


def _extract_detail(exc: httpx.HTTPStatusError) -> str | None:
    """Extract a human-readable detail string from an HTTP error response."""
    try:
        payload = exc.response.json()
    except ValueError:
        return None
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str):
            return detail
    return None


@contextmanager
def cli_http_errors(resource: str) -> Generator[None, None, None]:
    """Wrap HTTP calls with consistent CLI error reporting.

    Catches httpx errors and prints a user-friendly message before exiting.
    """
    try:
        yield
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            console.print(f"[red]{resource} not found[/red]")
        else:
            detail = _extract_detail(exc) or f"status {exc.response.status_code}"
            console.print(f"[red]Request failed: {detail}[/red]")
        raise typer.Exit(1) from exc
    except httpx.HTTPError as exc:
        console.print(f"[red]Failed to reach server:[/red] {exc}")
        raise typer.Exit(1) from exc
