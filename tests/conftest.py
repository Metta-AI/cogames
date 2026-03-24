from __future__ import annotations

from typing import NoReturn

import pytest

collect_ignore = ["test_episode_runner_compat.py"]


def _fail_browser_launch(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Tests must not launch a real browser.")


@pytest.fixture(autouse=True)
def _block_browser_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("webbrowser.open", _fail_browser_launch)
    monkeypatch.setattr("webbrowser.open_new", _fail_browser_launch)
    monkeypatch.setattr("webbrowser.open_new_tab", _fail_browser_launch)
