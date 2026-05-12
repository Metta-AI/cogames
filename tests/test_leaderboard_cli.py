from __future__ import annotations

import json
import uuid
from typing import Any, cast

import pytest
from _test_support import capture_output, season_summary

from cogames.cli import leaderboard
from cogames.cli.client import TournamentServerClient
from cogames.cli.generated_models import LeaderboardEntry, PolicySummary, PolicyVersionSummary

POLICY_ID_1 = uuid.UUID("00000000-0000-0000-0000-000000000001")
POLICY_ID_2 = uuid.UUID("00000000-0000-0000-0000-000000000002")


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
    printed = capture_output(monkeypatch, leaderboard.console)

    leaderboard._show_season_submissions(
        cast(TournamentServerClient, client),
        "test-season",
        policy_name=None,
        json_output=True,
    )

    assert client.called
    assert printed == [json.dumps([], indent=2)]


class _PublicLeaderboardClient:
    def __init__(self, entries: list[LeaderboardEntry]) -> None:
        self._entries = entries

    def __enter__(self) -> _PublicLeaderboardClient:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def get_default_season(self) -> Any:
        return season_summary(name="test-season")

    def get_leaderboard(self, season_name: str) -> list[LeaderboardEntry]:
        _ = season_name
        return self._entries


class _AuthLeaderboardClient:
    def __init__(self, mine_entries: list[PolicySummary]) -> None:
        self._mine_entries = mine_entries
        self.called_with: tuple[str, bool] | None = None

    def __enter__(self) -> _AuthLeaderboardClient:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def get_season_policies(self, season_name: str, mine: bool = False) -> list[PolicySummary]:
        self.called_with = (season_name, mine)
        return self._mine_entries


class _LeaderboardClientFactory:
    def __init__(
        self,
        *,
        public_client: _PublicLeaderboardClient,
        auth_client: _AuthLeaderboardClient,
    ) -> None:
        self._public_client = public_client
        self._auth_client = auth_client
        self.public_call: tuple[str, str | None] | None = None

    def __call__(
        self,
        server_url: str,
        token: str | None = None,
    ) -> _PublicLeaderboardClient:
        self.public_call = (server_url, token)
        return self._public_client

    def from_login(self, *, server_url: str) -> _AuthLeaderboardClient:
        return self._auth_client


def test_leaderboard_mine_uses_season_policies_for_filtering(monkeypatch: pytest.MonkeyPatch) -> None:
    printed = capture_output(monkeypatch, leaderboard.console)
    all_entries = [
        LeaderboardEntry(
            rank=1,
            policy=PolicyVersionSummary(id=POLICY_ID_1, name="mine-policy", version=3),
            score=100.0,
            matches=10,
        ),
        LeaderboardEntry(
            rank=2,
            policy=PolicyVersionSummary(id=POLICY_ID_2, name="other-policy", version=9),
            score=90.0,
            matches=8,
        ),
    ]
    mine_entries = [
        PolicySummary(
            policy=PolicyVersionSummary(id=POLICY_ID_1, name="mine-policy", version=3),
            pools=[],
            entered_at="2026-01-01T00:00:00Z",
        ),
    ]
    public_client = _PublicLeaderboardClient(entries=all_entries)
    auth_client = _AuthLeaderboardClient(mine_entries=mine_entries)

    client_factory = _LeaderboardClientFactory(public_client=public_client, auth_client=auth_client)
    monkeypatch.setattr(leaderboard, "TournamentServerClient", client_factory)
    monkeypatch.setattr(leaderboard, "load_current_cogames_token", lambda login_server: "test-token")
    monkeypatch.setattr(leaderboard, "get_login_server", lambda: "https://login.example")

    leaderboard.leaderboard_cmd(
        season_arg="test-season",
        season=None,
        policy_filter=None,
        mine=True,
        server="https://server.example",
        json_output=True,
    )

    assert auth_client.called_with == ("test-season", True)
    assert client_factory.public_call == ("https://server.example", "test-token")
    payload = json.loads(str(printed[0]))
    assert [entry["policy"]["id"] for entry in payload] == [str(POLICY_ID_1)]


def test_leaderboard_uses_login_token_without_mine_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    printed = capture_output(monkeypatch, leaderboard.console)
    public_client = _PublicLeaderboardClient(entries=[])
    auth_client = _AuthLeaderboardClient(mine_entries=[])
    client_factory = _LeaderboardClientFactory(public_client=public_client, auth_client=auth_client)

    monkeypatch.setattr(leaderboard, "TournamentServerClient", client_factory)
    monkeypatch.setattr(leaderboard, "load_current_cogames_token", lambda login_server: "test-token")
    monkeypatch.setattr(leaderboard, "get_login_server", lambda: "https://login.example")

    leaderboard.leaderboard_cmd(
        season_arg="private-season",
        season=None,
        policy_filter=None,
        mine=False,
        server="https://server.example",
        json_output=True,
    )

    assert client_factory.public_call == ("https://server.example", "test-token")
    assert auth_client.called_with is None
    assert printed == [json.dumps([], indent=2)]
