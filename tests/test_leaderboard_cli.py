from __future__ import annotations

import json
from typing import Any, cast

import pytest

from cogames.cli import leaderboard
from cogames.cli.client import TournamentServerClient


class _FakeClient:
    def __init__(self) -> None:
        self.called: bool = False

    def get_season_policies(
        self,
        season_name: str,
        mine: bool = False,
    ) -> list[Any]:
        _ = (season_name, mine)
        self.called = True
        return []


def test_show_season_submissions_calls_client(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient()
    printed: list[str] = []
    monkeypatch.setattr(leaderboard.console, "print", lambda value, *args, **kwargs: printed.append(str(value)))

    leaderboard._show_season_submissions(
        cast(TournamentServerClient, client),
        "test-season",
        policy_name=None,
        json_output=True,
    )

    assert client.called
    assert printed == [json.dumps([], indent=2)]
