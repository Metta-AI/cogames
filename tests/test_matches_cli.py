from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime
from typing import Any, cast

import pytest

from cogames.cli import matches
from cogames.cli.client import TournamentServerClient
from cogames.cli.generated_models import MatchPlayerInfo, MatchResponse, PolicyVersionRow, PolicyVersionSummary

POLICY_VERSION_ID = uuid.UUID("00000000-0000-0000-0000-0000000000a1")
POLICY_ID = uuid.UUID("00000000-0000-0000-0000-0000000000b2")
MATCH_ID = uuid.UUID("00000000-0000-0000-0000-0000000000c3")


def _capture(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    printed: list[str] = []
    monkeypatch.setattr(matches.console, "print", lambda value, *args, **kwargs: printed.append(str(value)))
    buffer: list[str] = []

    def _stdout_write(text: str) -> int:
        buffer.append(text)
        if "\n" in text:
            payload = "".join(buffer).rstrip("\n")
            buffer.clear()
            if payload:
                printed.append(payload)
        return len(text)

    monkeypatch.setattr(sys.stdout, "write", _stdout_write)
    return printed


class _FakeClient:
    def __init__(self) -> None:
        self.policy_calls: list[tuple[str | None, int | None]] = []
        self.match_calls: list[tuple[str, list[uuid.UUID] | None, int | None]] = []

    def get_my_policy_versions(self, name: str | None = None, version: int | None = None) -> list[PolicyVersionRow]:
        self.policy_calls.append((name, version))
        if name == "mine-policy" and version == 7:
            return [
                PolicyVersionRow(
                    user_id="u1",
                    id=POLICY_VERSION_ID,
                    policy_id=POLICY_ID,
                    name="mine-policy",
                    version=7,
                    created_at=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
                    policy_created_at=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
                )
            ]
        return []

    def get_default_season(self) -> Any:
        return type("SeasonSummary", (), {"name": "default-season"})()

    def get_season_matches(
        self,
        season_name: str,
        policy_version_ids: list[uuid.UUID] | None = None,
        limit: int | None = None,
    ) -> list[MatchResponse]:
        self.match_calls.append((season_name, policy_version_ids, limit))
        return [
            MatchResponse(
                id=MATCH_ID,
                season_name=season_name,
                pool_name="entry",
                status="completed",
                assignments=[0],
                players=[
                    MatchPlayerInfo(
                        policy=PolicyVersionSummary(id=POLICY_VERSION_ID, name="mine-policy", version=7),
                        num_agents=1,
                        score=1.0,
                    )
                ],
                error=None,
                episode_id=None,
                created_at=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
            )
        ]


def test_list_matches_policy_filter_queries_server_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    printed = _capture(monkeypatch)
    client = _FakeClient()

    matches._list_matches(
        client=cast(TournamentServerClient, client),
        season="test-season",
        limit=5,
        json_output=True,
        policy_filter="mine-policy:v7",
    )

    assert client.policy_calls == [("mine-policy", 7)]
    assert client.match_calls == [("test-season", [POLICY_VERSION_ID], 5)]
    payload = json.loads(printed[0])
    assert payload[0]["id"] == str(MATCH_ID)
