from __future__ import annotations

import json

from _test_support import capture_output, render_output
from rich.console import Console

from cogames.cli.base import emit_json


def test_capture_output_collects_console_prints_and_stdout_json(monkeypatch) -> None:
    console = Console()
    printed = capture_output(monkeypatch, console)

    console.print("hello")
    emit_json({"status": "ok"})

    assert printed[0] == "hello"
    assert json.loads(str(printed[1])) == {"status": "ok"}


def test_render_output_renders_rich_objects() -> None:
    assert "hello" in render_output("[bold]hello[/bold]")
