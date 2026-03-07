from __future__ import annotations

import io
import json
import sys
import uuid
from datetime import datetime
from typing import Any

import httpx
import pytest
import typer
from rich.console import Console

from cogames.cli import season
from cogames.cli.client import PoolConfigInfo
from cogames.cli.generated_models import (
    Kind,
    LeaderboardEntry,
    MatchPlayerInfo,
    MatchResponse,
    PolicyVersionSummary,
    PoolInfo,
    ProgressStage,
    ScorePoliciesLeaderboardEntry,
    SeasonDetail,
    SeasonSummary,
    SeasonVersionInfo,
    StageStats,
    Status,
    TeamCogSummary,
    TeamSummary,
    TeamTournamentProgress,
)

FAKE_UUID = uuid.UUID("00000000-0000-0000-0000-000000000001")
FAKE_UUID_2 = uuid.UUID("00000000-0000-0000-0000-000000000002")


def _make_season_summary(**overrides: Any) -> SeasonSummary:
    defaults: dict[str, Any] = {
        "id": FAKE_UUID,
        "name": "test-season",
        "display_name": "Test Season",
        "version": 1,
        "canonical": True,
        "summary": "A test season",
        "entry_pool": "entry",
        "leaderboard_pool": "leaderboard",
        "is_default": True,
        "compat_version": "0.15",
        "created_at": "2026-01-01T00:00:00Z",
        "public": True,
        "tournament_type": "freeplay",
        "pools": [PoolInfo(name="entry", description="Entry pool")],
    }
    defaults.update(overrides)
    return SeasonSummary(**defaults)


def _make_season_info(**overrides: Any) -> SeasonDetail:
    defaults: dict[str, Any] = {
        "id": FAKE_UUID,
        "name": "test-season",
        "display_name": "Test Season",
        "version": 1,
        "canonical": True,
        "summary": "A test season",
        "entry_pool": "entry",
        "leaderboard_pool": "leaderboard",
        "is_default": True,
        "compat_version": "0.15",
        "created_at": "2026-01-01T00:00:00Z",
        "public": True,
        "tournament_type": "freeplay",
        "pools": [],
        "status": "in_progress",
        "started_at": "2026-01-01T00:00:00Z",
        "entrant_count": 10,
        "active_entrant_count": 8,
        "match_count": 50,
        "stage_count": 3,
    }
    defaults.update(overrides)
    return SeasonDetail(**defaults)


def _policy_summary(name: str = "policy-a", version: int = 1) -> PolicyVersionSummary:
    return PolicyVersionSummary(id=FAKE_UUID, name=name, version=version)


def _render(*objects: Any) -> str:
    buf = io.StringIO()
    c = Console(file=buf, force_terminal=False, width=200)
    for obj in objects:
        c.print(obj)
    return buf.getvalue()


class _FakeClient:
    def __init__(self) -> None:
        self.seasons: list[SeasonSummary] = [_make_season_summary()]
        self.season_info: SeasonDetail = _make_season_info()
        self.versions: list[SeasonVersionInfo] = [
            SeasonVersionInfo(
                version=1,
                canonical=True,
                disabled_at=None,
                created_at="2026-01-01T00:00:00Z",
                compat_version="0.15",
            ),
        ]
        self.stages: list[StageStats] = [
            StageStats(
                name="stage-1",
                policy_count=12,
                match_count=24,
                completion_pct=100.0,
                team_count=6,
                eliminated_count=2,
            ),
            StageStats(
                name="stage-2",
                policy_count=8,
                match_count=16,
                completion_pct=62.5,
                team_count=4,
                eliminated_count=None,
            ),
        ]
        self.progress: TeamTournamentProgress = TeamTournamentProgress(
            phase="policy_eval",
            phase_detail={"pool": "stage-2", "stage_index": 1},
            stages=self.stages,
            stage_flow=[
                ProgressStage(
                    index=1,
                    name="Play-ins: one-player",
                    kind=Kind.policy_eval,
                    description="Stage one",
                    input_pool="stage-1",
                    output_pool="stage-2",
                    status=Status.complete,
                ),
                ProgressStage(
                    index=2,
                    name="Play-ins: two-player",
                    kind=Kind.policy_eval,
                    description="Stage two",
                    input_pool="stage-2",
                    output_pool="stage-3",
                    status=Status.active,
                ),
            ],
            started=True,
        )
        self.teams: list[TeamSummary] = [
            TeamSummary(
                id=FAKE_UUID_2,
                pool_name="pool-a",
                eliminated=False,
                score=100.0,
                matches=12,
                cogs=[TeamCogSummary(position=0, policy=_policy_summary())],
                created_at="2026-01-01T00:00:00Z",
            ),
        ]
        self.leaderboard: list[LeaderboardEntry] = [
            LeaderboardEntry(rank=1, policy=_policy_summary(), score=95.5, score_stddev=2.1, matches=10),
        ]
        self.score_policies_leaderboard: list[ScorePoliciesLeaderboardEntry] = [
            ScorePoliciesLeaderboardEntry(
                rank=1,
                policy=_policy_summary(),
                placement_score=7.0,
                team_appearances=3,
                team_ranks=[1, 2, 3],
            ),
        ]
        self.matches: list[MatchResponse] = [
            MatchResponse(
                id=FAKE_UUID_2,
                season_name="test-season",
                pool_name="pool-a",
                status="completed",
                assignments=[0, 1],
                players=[MatchPlayerInfo(policy=_policy_summary(), num_agents=1, score=10.0)],
                error=None,
                episode_id=None,
                created_at=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
            ),
        ]
        self.pool_config: PoolConfigInfo = PoolConfigInfo(pool_name="pool-a", config={"max_steps": 1000})
        self.last_leaderboard_season: str | None = None

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def get_seasons(self) -> list[SeasonSummary]:
        return self.seasons

    def get_season(self, season_name: str) -> SeasonDetail:
        return self.season_info

    def get_season_versions(self, season_name: str) -> list[SeasonVersionInfo]:
        return self.versions

    def get_default_season(self) -> SeasonSummary:
        return self.seasons[0]

    def get_stages(self, season_name: str) -> list[StageStats]:
        return self.stages

    def get_progress(self, season_name: str) -> TeamTournamentProgress:
        return self.progress

    def get_teams(self, season_name: str, **kwargs: Any) -> list[TeamSummary]:
        _ = kwargs
        return self.teams

    def get_leaderboard(self, season_name: str) -> list[LeaderboardEntry]:
        self.last_leaderboard_season = season_name
        return self.leaderboard

    def get_stage_leaderboard(
        self,
        season_name: str,
        leaderboard_type: str,
        pool_name: str,
    ) -> list[LeaderboardEntry] | list[ScorePoliciesLeaderboardEntry] | list[TeamSummary]:
        self.last_leaderboard_season = season_name
        if leaderboard_type == "team":
            return self.teams
        if leaderboard_type == "score-policies":
            return self.score_policies_leaderboard
        return self.leaderboard

    def get_score_policies_leaderboard(self, season_name: str) -> list[ScorePoliciesLeaderboardEntry]:
        self.last_leaderboard_season = season_name
        return self.score_policies_leaderboard

    def get_season_matches(
        self,
        season_name: str,
        policy_version_ids: list[uuid.UUID] | None = None,
        limit: int | None = None,
    ) -> list[MatchResponse]:
        return self.matches

    def get_pool_config(self, season_name: str, pool_name: str) -> PoolConfigInfo:
        return self.pool_config


@pytest.fixture()
def fake_client(monkeypatch: pytest.MonkeyPatch) -> _FakeClient:
    client = _FakeClient()
    monkeypatch.setattr(season, "_get_client", lambda *args, **kwargs: client)
    return client


def _capture(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    printed: list[Any] = []
    monkeypatch.setattr(season.console, "print", lambda value, *args, **kwargs: printed.append(value))
    buf: list[str] = []

    def _stdout_write(s: str) -> int:
        buf.append(s)
        if "\n" in s:
            text = "".join(buf).rstrip("\n")
            buf.clear()
            if text:
                printed.append(text)
        return len(s)

    monkeypatch.setattr(sys.stdout, "write", _stdout_write)
    return printed


def test_season_list_json(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = _capture(monkeypatch)
    season.season_list(json_output=True, login_server="x", server="y")
    output = json.loads(str(printed[0]))
    assert len(output) == 1
    assert output[0]["name"] == "test-season"


def test_season_list_table(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = _capture(monkeypatch)
    season.season_list(json_output=False, login_server="x", server="y")
    rendered = _render(*printed)
    assert "test-season" in rendered


def test_season_show_json(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = _capture(monkeypatch)
    season.season_show(season_name="test-season", json_output=True, login_server="x", server="y")
    output = json.loads(str(printed[0]))
    assert output["name"] == "test-season"
    assert output["status"] == "in_progress"


def test_season_show_table(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = _capture(monkeypatch)
    season.season_show(season_name="test-season", json_output=False, login_server="x", server="y")
    rendered = _render(*printed)
    assert "Test Season" in rendered
    assert "in_progress" in rendered


def test_season_versions_json(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = _capture(monkeypatch)
    season.season_versions(season_name="test-season", json_output=True, login_server="x", server="y")
    output = json.loads(str(printed[0]))
    assert len(output) == 1
    assert output[0]["version"] == 1


def test_season_stages_json(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = _capture(monkeypatch)
    season.season_stages(season_name="test-season", json_output=True, login_server="x", server="y")
    output = json.loads(str(printed[0]))
    assert len(output) == 2
    assert output[0]["name"] == "stage-1"


def test_season_stages_table(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = _capture(monkeypatch)
    season.season_stages(season_name="test-season", json_output=False, login_server="x", server="y")
    rendered = _render(*printed)
    assert "stage-1" in rendered
    assert "stage-2" in rendered


def test_season_progress_json(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = _capture(monkeypatch)
    season.season_progress(season_name="test-season", json_output=True, login_server="x", server="y")
    output = json.loads(str(printed[0]))
    assert output["phase"] == "policy_eval"
    assert output["started"] is True


def test_season_progress_400_shows_detail(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = _capture(monkeypatch)

    def _raise_400(season_name: str) -> TeamTournamentProgress:
        _ = season_name
        req = httpx.Request("GET", "https://example.test/tournament/seasons/test-season/progress")
        resp = httpx.Response(400, request=req, json={"detail": "Progress not available for this season type"})
        raise httpx.HTTPStatusError("400", request=req, response=resp)

    monkeypatch.setattr(fake_client, "get_progress", _raise_400)

    with pytest.raises(typer.Exit):
        season.season_progress(season_name="test-season", json_output=False, login_server="x", server="y")

    assert any("Progress not available for this season type" in str(p) for p in printed)


def test_season_leaderboard_json(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = _capture(monkeypatch)
    season.season_leaderboard(
        season_name="test-season",
        pool=None,
        leaderboard_type="policy",
        json_output=True,
        login_server="x",
        server="y",
    )
    output = json.loads(str(printed[0]))
    assert len(output) == 1
    assert output[0]["rank"] == 1


def test_season_leaderboard_with_pool_json(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = _capture(monkeypatch)
    season.season_leaderboard(
        season_name="test-season",
        pool="pool-a",
        leaderboard_type="policy",
        json_output=True,
        login_server="x",
        server="y",
    )
    output = json.loads(str(printed[0]))
    assert len(output) == 1


def test_season_leaderboard_table(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = _capture(monkeypatch)
    season.season_leaderboard(
        season_name="test-season",
        pool=None,
        leaderboard_type="policy",
        json_output=False,
        login_server="x",
        server="y",
    )
    rendered = _render(*printed)
    assert "policy-a" in rendered
    assert "95.50" in rendered


def test_season_leaderboard_team_table(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = _capture(monkeypatch)
    season.season_leaderboard(
        season_name="test-season",
        pool="pool-a",
        leaderboard_type="team",
        json_output=False,
        login_server="x",
        server="y",
    )
    rendered = _render(*printed)
    assert "Team Leaderboard" in rendered
    assert "pool-a" in rendered
    assert "policy-a" in rendered


def test_season_leaderboard_400_shows_detail(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = _capture(monkeypatch)

    def _raise_400(season_name: str, leaderboard_type: str, pool_name: str) -> list[LeaderboardEntry]:
        _ = season_name, leaderboard_type, pool_name
        req = httpx.Request(
            "GET", "https://example.test/tournament/seasons/test-season/leaderboard/score-policies/stage-1"
        )
        detail = "Score-policies leaderboard must use pool 'policy-scores-1'"
        resp = httpx.Response(400, request=req, json={"detail": detail})
        raise httpx.HTTPStatusError("400", request=req, response=resp)

    monkeypatch.setattr(fake_client, "get_stage_leaderboard", _raise_400)

    with pytest.raises(typer.Exit):
        season.season_leaderboard(
            season_name="test-season",
            pool="stage-1",
            leaderboard_type="score-policies",
            json_output=False,
            login_server="x",
            server="y",
        )

    assert any("Score-policies leaderboard must use pool 'policy-scores-1'" in str(p) for p in printed)


def test_season_leaderboard_uses_server_default_when_omitted(
    fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_client.seasons = [_make_season_summary(name="server-default", is_default=True)]
    printed = _capture(monkeypatch)
    season.season_leaderboard(
        season_name=None,
        pool=None,
        leaderboard_type="policy",
        json_output=True,
        login_server="x",
        server="y",
    )
    output = json.loads(str(printed[0]))
    assert len(output) == 1
    assert fake_client.last_leaderboard_season == "server-default"


def test_season_leaderboard_score_policies_without_pool(
    fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    printed = _capture(monkeypatch)

    def _fail_stage(*args: Any, **kwargs: Any) -> list[LeaderboardEntry]:
        _ = args, kwargs
        raise AssertionError("stage leaderboard should not be called without pool for score-policies")

    monkeypatch.setattr(fake_client, "get_stage_leaderboard", _fail_stage)
    season.season_leaderboard(
        season_name="test-season",
        pool=None,
        leaderboard_type="score-policies",
        json_output=True,
        login_server="x",
        server="y",
    )
    output = json.loads(str(printed[0]))
    assert len(output) == 1
    assert output[0]["placement_score"] == 7.0


def test_season_leaderboard_team_requires_pool(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = _capture(monkeypatch)
    with pytest.raises(typer.Exit):
        season.season_leaderboard(
            season_name="test-season",
            pool=None,
            leaderboard_type="team",
            json_output=False,
            login_server="x",
            server="y",
        )
    assert any("Team leaderboard requires --pool <POOL>." in str(p) for p in printed)


def test_season_teams_json(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = _capture(monkeypatch)
    season.season_teams(season_name="test-season", json_output=True, login_server="x", server="y")
    output = json.loads(str(printed[0]))
    assert len(output) == 1
    assert output[0]["pool_name"] == "pool-a"


def test_season_teams_table(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = _capture(monkeypatch)
    season.season_teams(season_name="test-season", json_output=False, login_server="x", server="y")
    rendered = _render(*printed)
    assert "pool-a" in rendered
    assert "policy-a" in rendered


def test_season_matches_json(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = _capture(monkeypatch)
    season.season_matches(season_name="test-season", limit=20, json_output=True, login_server="x", server="y")
    output = json.loads(str(printed[0]))
    assert len(output) == 1


def test_season_matches_table(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = _capture(monkeypatch)
    season.season_matches(season_name="test-season", limit=20, json_output=False, login_server="x", server="y")
    rendered = _render(*printed)
    assert "pool-a" in rendered


def test_season_pool_config_json(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = _capture(monkeypatch)
    season.season_pool_config(
        season_name="test-season",
        pool_name="pool-a",
        json_output=True,
        login_server="x",
        server="y",
    )
    output = json.loads(str(printed[0]))
    assert output["pool_name"] == "pool-a"
    assert output["config"]["max_steps"] == 1000


def test_season_pool_config_table(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = _capture(monkeypatch)
    season.season_pool_config(
        season_name="test-season",
        pool_name="pool-a",
        json_output=False,
        login_server="x",
        server="y",
    )
    rendered = _render(*printed)
    assert "pool-a" in rendered
    assert "max_steps" in rendered


def test_season_list_empty(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client.seasons = []
    printed = _capture(monkeypatch)
    season.season_list(json_output=False, login_server="x", server="y")
    assert any("No seasons found" in str(p) for p in printed)


def test_legacy_seasons_alias_removed() -> None:
    from cogames.main import app  # noqa: PLC0415

    command_names = [cmd.name for cmd in app.registered_commands]
    assert "seasons" not in command_names


def test_legacy_leaderboard_alias_exists() -> None:
    from cogames.main import app  # noqa: PLC0415

    command_names = [cmd.name for cmd in app.registered_commands]
    assert "leaderboard" in command_names


def test_season_subapp_registered() -> None:
    from cogames.main import app  # noqa: PLC0415

    group_names = [g.name for g in app.registered_groups]
    assert "season" in group_names
