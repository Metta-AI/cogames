from __future__ import annotations

import json
import uuid
from typing import Any, cast

import pytest
from _test_support import (
    capture_output,
    match_response,
    policy_version_row,
    policy_version_summary,
    season_summary,
)

from cogames.cli import matches
from cogames.cli.client import TournamentServerClient
from cogames.cli.generated_models import MatchPlayerInfo, MatchResponse, PolicyVersionRow

POLICY_VERSION_ID = uuid.UUID("00000000-0000-0000-0000-0000000000a1")
POLICY_ID = uuid.UUID("00000000-0000-0000-0000-0000000000b2")
MATCH_ID = uuid.UUID("00000000-0000-0000-0000-0000000000c3")


class _FakeClient:
    def __init__(self) -> None:
        self.policy_calls: list[tuple[str | None, int | None]] = []
        self.match_calls: list[tuple[str, list[uuid.UUID] | None, int | None]] = []

    def get_my_policy_versions(self, name: str | None = None, version: int | None = None) -> list[PolicyVersionRow]:
        self.policy_calls.append((name, version))
        if name == "mine-policy" and version == 7:
            return [
                policy_version_row(
                    id=POLICY_VERSION_ID,
                    policy_id=POLICY_ID,
                    name="mine-policy",
                    version=7,
                    user_id="u1",
                )
            ]
        return []

    def get_default_season(self) -> Any:
        return season_summary(name="default-season")

    def get_season_matches(
        self,
        season_name: str,
        policy_version_ids: list[uuid.UUID] | None = None,
        limit: int | None = None,
    ) -> list[MatchResponse]:
        self.match_calls.append((season_name, policy_version_ids, limit))
        return [
            match_response(
                id=MATCH_ID,
                season_name=season_name,
                pool_name="entry",
                players=[
                    MatchPlayerInfo(
                        policy=policy_version_summary(id=POLICY_VERSION_ID, name="mine-policy", version=7),
                        num_agents=1,
                        score=1.0,
                    )
                ],
                assignments=[0],
            )
        ]


def test_list_matches_policy_filter_queries_server_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    printed = capture_output(monkeypatch, matches.console)
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
    payload = json.loads(str(printed[0]))
    assert payload[0]["id"] == str(MATCH_ID)
