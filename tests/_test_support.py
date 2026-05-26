from __future__ import annotations

import io

import pytest
from rich.console import Console


def capture_output(monkeypatch: pytest.MonkeyPatch, console: Console) -> list[object]:
    printed: list[object] = []
    monkeypatch.setattr(console, "print", lambda value, *args, **kwargs: printed.append(value))
    buffer: list[str] = []

    def _stdout_write(text: str) -> int:
        buffer.append(text)
        if "\n" in text:
            payload = "".join(buffer).rstrip("\n")
            buffer.clear()
            if payload:
                printed.append(payload)
        return len(text)

    monkeypatch.setattr("sys.stdout.write", _stdout_write)
    return printed


def render_output(*objects: object) -> str:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=200)
    for obj in objects:
        console.print(obj)
    return buffer.getvalue()
