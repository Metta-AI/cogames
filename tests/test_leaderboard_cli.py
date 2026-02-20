from __future__ import annotations

import json
from typing import Any, cast

import pytest

from cogames.cli import leaderboard
from cogames.cli.client import TournamentServerClient


class _FakeClient:
    def __init__(self) -> None:
        self.include_hidden_args: list[bool] = []

    def get_season_policies(
        self,
        season_name: str,
        mine: bool = False,
        include_hidden_seasons: bool = False,
    ) -> list[Any]:
        _ = (season_name, mine)
        self.include_hidden_args.append(include_hidden_seasons)
        return []


@pytest.mark.parametrize("include_hidden", [False, True])
def test_show_season_submissions_forwards_include_hidden(monkeypatch: pytest.MonkeyPatch, include_hidden: bool) -> None:
    client = _FakeClient()
    printed: list[str] = []
    monkeypatch.setattr(leaderboard.console, "print", lambda value, *args, **kwargs: printed.append(str(value)))

    leaderboard._show_season_submissions(
        cast(TournamentServerClient, client),
        "test-season",
        policy_name=None,
        json_output=True,
        include_hidden=include_hidden,
    )

    assert client.include_hidden_args == [include_hidden]
    assert printed == [json.dumps([], indent=2)]
