from __future__ import annotations

import io
import uuid
from datetime import datetime
from typing import Any

import pytest
from rich.console import Console

from cogames.cli.generated_models import (
    LeaderboardEntry,
    MatchPlayerInfo,
    MatchResponse,
    PolicyVersionRow,
    PolicyVersionSummary,
    PoolInfo,
    SeasonDetail,
    SeasonSummary,
)

TEST_UUID = uuid.UUID("00000000-0000-0000-0000-000000000001")
TEST_UUID_2 = uuid.UUID("00000000-0000-0000-0000-000000000002")
TEST_UUID_3 = uuid.UUID("00000000-0000-0000-0000-000000000003")
TEST_UUID_4 = uuid.UUID("00000000-0000-0000-0000-000000000004")


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


def season_summary(**overrides: Any) -> SeasonSummary:
    return SeasonSummary(
        **{
            "id": TEST_UUID,
            "name": "test-season",
            "display_name": "Test Season",
            "version": 1,
            "canonical": True,
            "summary": "A test season",
            "entry_pool": "entry",
            "leaderboard_pool": "leaderboard",
            "is_default": True,
            "status": "in_progress",
            "compat_version": "0.15",
            "created_at": "2026-01-01T00:00:00Z",
            "public": True,
            "tournament_type": "freeplay",
            "pools": [PoolInfo(name="entry", description="Entry pool")],
        }
        | overrides
    )


def season_detail(**overrides: Any) -> SeasonDetail:
    return SeasonDetail(
        **{
            **season_summary().model_dump(mode="json"),
            "status": "in_progress",
            "started_at": "2026-01-01T00:00:00Z",
            "entrant_count": 10,
            "active_entrant_count": 8,
            "match_count": 50,
            "stage_count": 3,
        }
        | overrides
    )


def policy_version_summary(*, name: str = "policy-a", version: int = 1, **overrides: Any) -> PolicyVersionSummary:
    return PolicyVersionSummary(**{"id": TEST_UUID, "name": name, "version": version} | overrides)


def policy_version_row(*, name: str = "my-policy", version: int = 1, **overrides: Any) -> PolicyVersionRow:
    return PolicyVersionRow(
        **{
            "user_id": "test-user",
            "id": TEST_UUID_2,
            "policy_id": TEST_UUID_3,
            "name": name,
            "version": version,
            "created_at": datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
            "policy_created_at": datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
        }
        | overrides
    )


def leaderboard_entry(
    *,
    rank: int = 1,
    policy: PolicyVersionSummary | None = None,
    **overrides: Any,
) -> LeaderboardEntry:
    return LeaderboardEntry(
        **{
            "rank": rank,
            "policy": policy or policy_version_summary(),
            "score": 95.5,
            "score_stddev": 2.1,
            "matches": 10,
        }
        | overrides
    )


def match_response(
    *,
    policy: PolicyVersionSummary | None = None,
    players: list[MatchPlayerInfo] | None = None,
    **overrides: Any,
) -> MatchResponse:
    return MatchResponse(
        **{
            "id": TEST_UUID_4,
            "season_name": "test-season",
            "pool_name": "pool-a",
            "status": "completed",
            "assignments": [0, 1],
            "players": players
            or [MatchPlayerInfo(policy=policy or policy_version_summary(), num_agents=1, score=10.0)],
            "error": None,
            "episode_id": None,
            "created_at": datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
        }
        | overrides
    )
