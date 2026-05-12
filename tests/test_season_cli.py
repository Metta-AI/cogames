from __future__ import annotations

import json
import uuid
from typing import Any

import httpx
import pytest
import typer
from _test_support import (
    TEST_UUID_2,
    capture_output,
    leaderboard_entry,
    match_response,
    policy_version_summary,
    render_output,
    season_detail,
    season_summary,
)

from cogames.cli import season
from cogames.cli.client import PoolConfigInfo
from cogames.cli.generated_models import (
    Kind,
    LeaderboardEntry,
    MatchPlayerInfo,
    MatchResponse,
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


class _FakeClient:
    def __init__(self) -> None:
        self.seasons: list[SeasonSummary] = [
            season_summary(
                display_name="Test Season",
                entry_pool="entry",
                leaderboard_pool="leaderboard",
                pools=[PoolInfo(name="entry", description="Entry pool")],
            )
        ]
        self.season_info: SeasonDetail = season_detail(
            display_name="Test Season",
            entry_pool="entry",
            leaderboard_pool="leaderboard",
            pools=[],
        )
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
                id=TEST_UUID_2,
                pool_name="pool-a",
                eliminated=False,
                score=100.0,
                matches=12,
                cogs=[TeamCogSummary(position=0, policy=policy_version_summary())],
                created_at="2026-01-01T00:00:00Z",
            ),
        ]
        self.leaderboard: list[LeaderboardEntry] = [
            leaderboard_entry(policy=policy_version_summary()),
        ]
        self.score_policies_leaderboard: list[ScorePoliciesLeaderboardEntry] = [
            ScorePoliciesLeaderboardEntry(
                rank=1,
                policy=policy_version_summary(),
                placement_score=7.0,
                team_appearances=3,
                team_ranks=[1, 2, 3],
            ),
        ]
        self.matches: list[MatchResponse] = [
            match_response(
                id=TEST_UUID_2,
                players=[MatchPlayerInfo(policy=policy_version_summary(), num_agents=1, score=10.0)],
            ),
        ]
        self.pool_config: PoolConfigInfo = PoolConfigInfo(
            pool_name="pool-a",
            game_engine="mettagrid",
            config={"max_steps": 1000},
        )
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


def test_get_client_uses_login_token(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_load_current_cogames_token(login_server: str) -> str:
        captured["token_login_server"] = login_server
        return "test-token"

    def fake_tournament_server_client(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "client"

    monkeypatch.setattr(season, "load_current_cogames_token", fake_load_current_cogames_token)
    monkeypatch.setattr(season, "TournamentServerClient", fake_tournament_server_client)
    monkeypatch.setattr(season, "get_login_server", lambda: "http://login")

    assert season._get_client(server="http://server") == "client"
    assert captured == {
        "token_login_server": "http://login",
        "server_url": "http://server",
        "token": "test-token",
    }


def test_season_list_json(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = capture_output(monkeypatch, season.console)
    season.season_list(json_output=True, server="y")
    output = json.loads(str(printed[0]))
    assert len(output) == 1
    assert output[0]["name"] == "test-season"


def test_season_list_table(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = capture_output(monkeypatch, season.console)
    season.season_list(json_output=False, server="y")
    rendered = render_output(*printed)
    assert "test-season" in rendered


def test_season_show_json(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = capture_output(monkeypatch, season.console)
    season.season_show(season_name="test-season", json_output=True, server="y")
    output = json.loads(str(printed[0]))
    assert output["name"] == "test-season"
    assert output["status"] == "in_progress"


def test_season_show_table(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = capture_output(monkeypatch, season.console)
    season.season_show(season_name="test-season", json_output=False, server="y")
    rendered = render_output(*printed)
    assert "Test Season" in rendered
    assert "in_progress" in rendered


def test_season_versions_json(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = capture_output(monkeypatch, season.console)
    season.season_versions(season_name="test-season", json_output=True, server="y")
    output = json.loads(str(printed[0]))
    assert len(output) == 1
    assert output[0]["version"] == 1


def test_season_stages_json(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = capture_output(monkeypatch, season.console)
    season.season_stages(season_name="test-season", json_output=True, server="y")
    output = json.loads(str(printed[0]))
    assert len(output) == 2
    assert output[0]["name"] == "stage-1"


def test_season_stages_table(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = capture_output(monkeypatch, season.console)
    season.season_stages(season_name="test-season", json_output=False, server="y")
    rendered = render_output(*printed)
    assert "stage-1" in rendered
    assert "stage-2" in rendered


def test_season_progress_json(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = capture_output(monkeypatch, season.console)
    season.season_progress(season_name="test-season", json_output=True, server="y")
    output = json.loads(str(printed[0]))
    assert output["phase"] == "policy_eval"
    assert output["started"] is True


def test_season_progress_400_shows_detail(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = capture_output(monkeypatch, season.console)

    def _raise_400(season_name: str) -> TeamTournamentProgress:
        _ = season_name
        req = httpx.Request("GET", "https://example.test/tournament/seasons/test-season/progress")
        resp = httpx.Response(400, request=req, json={"detail": "Progress not available for this season type"})
        raise httpx.HTTPStatusError("400", request=req, response=resp)

    monkeypatch.setattr(fake_client, "get_progress", _raise_400)

    with pytest.raises(typer.Exit):
        season.season_progress(season_name="test-season", json_output=False, server="y")

    assert any("Progress not available for this season type" in str(p) for p in printed)


def test_season_leaderboard_json(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = capture_output(monkeypatch, season.console)
    season.season_leaderboard(
        season_name="test-season",
        pool=None,
        leaderboard_type="policy",
        json_output=True,
        server="y",
    )
    output = json.loads(str(printed[0]))
    assert len(output) == 1
    assert output[0]["rank"] == 1


def test_season_leaderboard_with_pool_json(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = capture_output(monkeypatch, season.console)
    season.season_leaderboard(
        season_name="test-season",
        pool="pool-a",
        leaderboard_type="policy",
        json_output=True,
        server="y",
    )
    output = json.loads(str(printed[0]))
    assert len(output) == 1


def test_season_leaderboard_table(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = capture_output(monkeypatch, season.console)
    season.season_leaderboard(
        season_name="test-season",
        pool=None,
        leaderboard_type="policy",
        json_output=False,
        server="y",
    )
    rendered = render_output(*printed)
    assert "policy-a" in rendered
    assert "95.50" in rendered


def test_season_leaderboard_team_table(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = capture_output(monkeypatch, season.console)
    season.season_leaderboard(
        season_name="test-season",
        pool="pool-a",
        leaderboard_type="team",
        json_output=False,
        server="y",
    )
    rendered = render_output(*printed)
    assert "Team Leaderboard" in rendered
    assert "pool-a" in rendered
    assert "policy-a" in rendered


def test_season_leaderboard_400_shows_detail(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = capture_output(monkeypatch, season.console)

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
            server="y",
        )

    assert any("Score-policies leaderboard must use pool 'policy-scores-1'" in str(p) for p in printed)


def test_season_leaderboard_uses_server_default_when_omitted(
    fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_client.seasons = [season_summary(name="server-default", is_default=True)]
    printed = capture_output(monkeypatch, season.console)
    season.season_leaderboard(
        season_name=None,
        pool=None,
        leaderboard_type="policy",
        json_output=True,
        server="y",
    )
    output = json.loads(str(printed[0]))
    assert len(output) == 1
    assert fake_client.last_leaderboard_season == "server-default"


def test_season_leaderboard_score_policies_without_pool(
    fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    printed = capture_output(monkeypatch, season.console)

    def _fail_stage(*args: Any, **kwargs: Any) -> list[LeaderboardEntry]:
        _ = args, kwargs
        raise AssertionError("stage leaderboard should not be called without pool for score-policies")

    monkeypatch.setattr(fake_client, "get_stage_leaderboard", _fail_stage)
    season.season_leaderboard(
        season_name="test-season",
        pool=None,
        leaderboard_type="score-policies",
        json_output=True,
        server="y",
    )
    output = json.loads(str(printed[0]))
    assert len(output) == 1
    assert output[0]["placement_score"] == 7.0


def test_season_leaderboard_team_requires_pool(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = capture_output(monkeypatch, season.console)
    with pytest.raises(typer.Exit):
        season.season_leaderboard(
            season_name="test-season",
            pool=None,
            leaderboard_type="team",
            json_output=False,
            server="y",
        )
    assert any("Team leaderboard requires --pool <POOL>." in str(p) for p in printed)


def test_season_teams_json(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = capture_output(monkeypatch, season.console)
    season.season_teams(season_name="test-season", json_output=True, server="y")
    output = json.loads(str(printed[0]))
    assert len(output) == 1
    assert output[0]["pool_name"] == "pool-a"


def test_season_teams_table(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = capture_output(monkeypatch, season.console)
    season.season_teams(season_name="test-season", json_output=False, server="y")
    rendered = render_output(*printed)
    assert "pool-a" in rendered
    assert "policy-a" in rendered


def test_season_matches_json(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = capture_output(monkeypatch, season.console)
    season.season_matches(season_name="test-season", limit=20, json_output=True, server="y")
    output = json.loads(str(printed[0]))
    assert len(output) == 1


def test_season_matches_table(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = capture_output(monkeypatch, season.console)
    season.season_matches(season_name="test-season", limit=20, json_output=False, server="y")
    rendered = render_output(*printed)
    assert "pool-a" in rendered


def test_season_pool_config_json(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = capture_output(monkeypatch, season.console)
    season.season_pool_config(
        season_name="test-season",
        pool_name="pool-a",
        json_output=True,
        server="y",
    )
    output = json.loads(str(printed[0]))
    assert output["pool_name"] == "pool-a"
    assert output["config"]["max_steps"] == 1000


def test_season_pool_config_table(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = capture_output(monkeypatch, season.console)
    season.season_pool_config(
        season_name="test-season",
        pool_name="pool-a",
        json_output=False,
        server="y",
    )
    rendered = render_output(*printed)
    assert "pool-a" in rendered
    assert "max_steps" in rendered


def test_season_list_empty(fake_client: _FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client.seasons = []
    printed = capture_output(monkeypatch, season.console)
    season.season_list(json_output=False, server="y")
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
