from __future__ import annotations

import json
import uuid

import pytest
from pytest_httpserver import HTTPServer

from cogames.cli.client import (
    EpisodeQueryResponse,
    EpisodeResponse,
    LeaderboardEntry,
    PlayerLoginResponse,
    PlayerResponse,
    PoliciesResponse,
    ScorePoliciesLeaderboardEntry,
    StageStats,
    TeamTournamentProgress,
    TournamentServerClient,
)
from cogames.cli.generated_models import (
    AgentResult,
    PolicyResult,
    PolicyRow,
    ProgressStage,
    TeamCogSummary,
    TeamSummary,
)

# ---------------------------------------------------------------------------
# Sample data factories
# ---------------------------------------------------------------------------

_PV_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
_PV_ID2 = "11111111-2222-3333-4444-555555555555"
_MATCH_ID = "99999999-8888-7777-6666-555555555555"
_EPISODE_ID = "abababab-cdcd-efef-1234-567890abcdef"
_TEAM_ID = "fefefefe-dcdc-baba-9876-543210fedcba"
_POLICY_ID = "deadbeef-dead-beef-dead-beefdeadbeef"


def _stage_stats_payload(name: str = "stage-1", policy_count: int = 10) -> dict:
    return {
        "name": name,
        "policy_count": policy_count,
        "match_count": 25,
        "completion_pct": 80.0,
        "team_count": 5,
        "eliminated_count": 2,
    }


def _progress_payload() -> dict:
    return {
        "phase": "sample_teams",
        "phase_detail": {"pool": "sample-1", "stage_index": 0},
        "stages": [_stage_stats_payload()],
        "stage_flow": [
            {
                "index": 1,
                "name": "Sampling",
                "kind": "sample_teams",
                "description": "Sample teams",
                "input_pool": "sample-1",
                "output_pool": "team-round-1",
                "status": "active",
            }
        ],
        "started": True,
    }


def _team_summary_payload() -> dict:
    return {
        "id": _TEAM_ID,
        "pool_name": "stage-1",
        "eliminated": False,
        "score": 1500.0,
        "matches": 12,
        "cogs": [
            {"position": 0, "policy": {"id": _PV_ID, "name": "alpha", "version": 3}},
            {"position": 1, "policy": {"id": _PV_ID2, "name": "beta", "version": 1}},
        ],
        "created_at": "2026-02-20T12:00:00Z",
    }


def _score_policies_entry_payload() -> dict:
    return {
        "rank": 1,
        "policy": {"id": _PV_ID, "name": "alpha", "version": 3},
        "placement_score": 42.5,
        "team_appearances": 3,
        "team_ranks": [1, 2, 5],
    }


def _episode_response_payload() -> dict:
    return {
        "id": _EPISODE_ID,
        "replay_url": "https://example.com/replay/1",
        "thumbnail_url": "https://example.com/thumb/1",
        "tags": {"game": "cogsguard", "match_type": "ranked"},
        "game_stats": {"objects.wall": 42.0, "tokens_written": 100.0},
        "policy_results": [
            {
                "position": 0,
                "policy": {"id": _PV_ID, "name": "alpha", "version": 3},
                "num_agents": 2,
                "avg_reward": 15.5,
                "avg_metrics": {"action.move": 0.8},
                "agents": [
                    {"agent_id": 0, "reward": 20.0, "metrics": {"action.move": 0.9}},
                    {"agent_id": 1, "reward": 11.0, "metrics": {"action.move": 0.7}},
                ],
            },
        ],
        "steps": 10000,
        "created_at": "2026-02-20T12:00:00Z",
    }


def _policy_row_payload() -> dict:
    return {
        "id": _POLICY_ID,
        "name": "alpha",
        "created_at": "2026-02-20T12:00:00Z",
        "user_id": _PV_ID,
        "attributes": {"tag": "v1"},
        "version_count": 5,
    }


def _player_response_payload() -> dict:
    return {
        "id": "ply_playeralpha",
        "user_id": "regular@example.com",
        "user": None,
        "name": "alpha",
        "created_at": "2026-02-20T12:00:00Z",
        "disabled_at": None,
    }


def _player_login_response_payload() -> dict:
    return {
        "player_id": "ply_playeralpha",
        "token": "ply_secret-token",
        "expires_at": "2026-02-21T12:00:00Z",
    }


# ---------------------------------------------------------------------------
# Model parsing tests
# ---------------------------------------------------------------------------


class TestStageStatsModel:
    def test_parse(self) -> None:
        data = _stage_stats_payload()
        stage = StageStats.model_validate(data)
        assert stage.name == "stage-1"
        assert stage.policy_count == 10
        assert stage.match_count == 25
        assert stage.completion_pct == 80.0
        assert stage.team_count == 5
        assert stage.eliminated_count == 2

    def test_nullable_team_fields(self) -> None:
        data = _stage_stats_payload()
        data["team_count"] = None
        data["eliminated_count"] = None
        stage = StageStats.model_validate(data)
        assert stage.team_count is None
        assert stage.eliminated_count is None


class TestProgressStageModel:
    def test_parse(self) -> None:
        data = {
            "index": 1,
            "name": "Sampling",
            "kind": "sample_teams",
            "description": "Sample teams",
            "input_pool": "sample-1",
            "output_pool": "team-round-1",
            "status": "active",
        }
        ps = ProgressStage.model_validate(data)
        assert ps.index == 1
        assert ps.name == "Sampling"
        assert ps.kind.value == "sample_teams"
        assert ps.status.value == "active"


class TestTeamTournamentProgressModel:
    def test_parse(self) -> None:
        data = _progress_payload()
        progress = TeamTournamentProgress.model_validate(data)
        assert progress.phase == "sample_teams"
        assert progress.started is True
        assert len(progress.stages) == 1
        assert isinstance(progress.stages[0], StageStats)
        assert progress.stage_flow is not None
        assert len(progress.stage_flow) == 1
        assert isinstance(progress.stage_flow[0], ProgressStage)


class TestTeamSummaryModel:
    def test_parse(self) -> None:
        data = _team_summary_payload()
        team = TeamSummary.model_validate(data)
        assert team.id == uuid.UUID(_TEAM_ID)
        assert team.pool_name == "stage-1"
        assert team.eliminated is False
        assert team.score == 1500.0
        assert team.matches == 12
        assert len(team.cogs) == 2
        assert isinstance(team.cogs[0], TeamCogSummary)
        assert team.cogs[0].position == 0
        assert team.cogs[0].policy.name == "alpha"


class TestScorePoliciesLeaderboardEntryModel:
    def test_parse(self) -> None:
        data = _score_policies_entry_payload()
        entry = ScorePoliciesLeaderboardEntry.model_validate(data)
        assert entry.rank == 1
        assert entry.placement_score == 42.5
        assert entry.team_appearances == 3
        assert entry.team_ranks == [1, 2, 5]
        assert entry.policy.name == "alpha"


class TestEpisodeResponseModel:
    def test_parse(self) -> None:
        data = _episode_response_payload()
        ep = EpisodeResponse.model_validate(data)
        assert ep.id == uuid.UUID(_EPISODE_ID)
        assert ep.replay_url == "https://example.com/replay/1"
        assert ep.steps == 10000
        assert len(ep.policy_results) == 1
        pr = ep.policy_results[0]
        assert isinstance(pr, PolicyResult)
        assert pr.num_agents == 2
        assert pr.avg_reward == 15.5
        assert len(pr.agents) == 2
        assert isinstance(pr.agents[0], AgentResult)
        assert pr.agents[0].reward == 20.0

    def test_nullable_fields(self) -> None:
        data = _episode_response_payload()
        data["replay_url"] = None
        data["thumbnail_url"] = None
        data["steps"] = None
        ep = EpisodeResponse.model_validate(data)
        assert ep.replay_url is None
        assert ep.thumbnail_url is None
        assert ep.steps is None


class TestPoliciesResponseModel:
    def test_parse(self) -> None:
        data = {"entries": [_policy_row_payload()], "total_count": 1}
        resp = PoliciesResponse.model_validate(data)
        assert resp.total_count == 1
        assert len(resp.entries) == 1
        row = resp.entries[0]
        assert isinstance(row, PolicyRow)
        assert row.name == "alpha"
        assert row.version_count == 5

    def test_parse_non_uuid_user_id(self) -> None:
        data = {"entries": [_policy_row_payload()], "total_count": 1}
        data["entries"][0]["user_id"] = "gh732qp1wp4svaaiakiywlnr"
        resp = PoliciesResponse.model_validate(data)
        assert resp.entries[0].user_id == "gh732qp1wp4svaaiakiywlnr"


class TestPlayerResponseModel:
    def test_parse(self) -> None:
        player = PlayerResponse.model_validate(_player_response_payload())
        assert player.id == "ply_playeralpha"
        assert player.user_id == "regular@example.com"
        assert player.name == "alpha"


class TestPlayerLoginResponseModel:
    def test_parse(self) -> None:
        login = PlayerLoginResponse.model_validate(_player_login_response_payload())
        assert login.player_id == "ply_playeralpha"
        assert login.token == "ply_secret-token"


# ---------------------------------------------------------------------------
# Client method tests (request shape + response parsing)
# ---------------------------------------------------------------------------


@pytest.fixture
def client(httpserver: HTTPServer) -> TournamentServerClient:
    return TournamentServerClient(server_url=httpserver.url_for(""), token="test-token")


class TestGetStages:
    def test_request_and_parse(self, httpserver: HTTPServer, client: TournamentServerClient) -> None:
        httpserver.expect_request("/tournament/seasons/s1/stages", method="GET").respond_with_json(
            [_stage_stats_payload()]
        )
        stages = client.get_stages("s1")
        assert len(stages) == 1
        assert isinstance(stages[0], StageStats)
        assert stages[0].name == "stage-1"


class TestPlayers:
    def test_list_players(self, httpserver: HTTPServer, client: TournamentServerClient) -> None:
        httpserver.expect_request("/players", method="GET").respond_with_json([_player_response_payload()])
        players = client.list_players()
        assert len(players) == 1
        assert isinstance(players[0], PlayerResponse)
        assert players[0].id == "ply_playeralpha"

    def test_login_player(self, httpserver: HTTPServer, client: TournamentServerClient) -> None:
        httpserver.expect_request("/players/ply_playeralpha/login", method="POST").respond_with_json(
            _player_login_response_payload()
        )
        login = client.login_player("ply_playeralpha")
        assert isinstance(login, PlayerLoginResponse)
        assert login.player_id == "ply_playeralpha"
        assert login.token == "ply_secret-token"


class TestGetProgress:
    def test_request_and_parse(self, httpserver: HTTPServer, client: TournamentServerClient) -> None:
        httpserver.expect_request("/tournament/seasons/s1/progress", method="GET").respond_with_json(
            _progress_payload()
        )
        progress = client.get_progress("s1")
        assert isinstance(progress, TeamTournamentProgress)
        assert progress.phase == "sample_teams"


class TestGetTeams:
    def test_request_and_parse(self, httpserver: HTTPServer, client: TournamentServerClient) -> None:
        httpserver.expect_request("/tournament/seasons/s1/teams", method="GET").respond_with_json(
            [_team_summary_payload()]
        )
        teams = client.get_teams("s1")
        assert len(teams) == 1
        assert isinstance(teams[0], TeamSummary)
        assert teams[0].pool_name == "stage-1"

    def test_filters(self, httpserver: HTTPServer, client: TournamentServerClient) -> None:
        httpserver.expect_request("/tournament/seasons/s1/teams", method="GET").respond_with_json([])
        result = client.get_teams("s1", limit=10, offset=5, pool_name="stage-2", eliminated=True)
        assert result == []
        req, _ = httpserver.log[-1]
        assert req.args["limit"] == "10"
        assert req.args["offset"] == "5"
        assert req.args["pool_name"] == "stage-2"
        assert req.args["eliminated"] == "true"

    def test_policy_version_id_filter(self, httpserver: HTTPServer, client: TournamentServerClient) -> None:
        httpserver.expect_request("/tournament/seasons/s1/teams", method="GET").respond_with_json([])
        pv_id = uuid.UUID(_PV_ID)
        result = client.get_teams("s1", policy_version_id=pv_id)
        assert result == []
        req, _ = httpserver.log[-1]
        assert req.args["policy_version_id"] == _PV_ID


class TestGetStageLeaderboard:
    def test_policy_leaderboard(self, httpserver: HTTPServer, client: TournamentServerClient) -> None:
        entry = {
            "rank": 1,
            "policy": {"id": _PV_ID, "name": "alpha", "version": 3},
            "score": 1600.0,
            "score_stddev": 50.0,
            "matches": 20,
        }
        httpserver.expect_request("/tournament/seasons/s1/leaderboard/policy/stage-1", method="GET").respond_with_json(
            [entry]
        )
        result = client.get_stage_leaderboard("s1", "policy", "stage-1")
        assert len(result) == 1
        assert isinstance(result[0], LeaderboardEntry)
        assert result[0].rank == 1
        assert result[0].score == 1600.0
        assert result[0].policy.name == "alpha"

    def test_score_policies_leaderboard(self, httpserver: HTTPServer, client: TournamentServerClient) -> None:
        httpserver.expect_request(
            "/tournament/seasons/s1/leaderboard/score-policies/stage-1", method="GET"
        ).respond_with_json([_score_policies_entry_payload()])
        result = client.get_stage_leaderboard("s1", "score-policies", "stage-1")
        assert len(result) == 1
        assert isinstance(result[0], ScorePoliciesLeaderboardEntry)
        assert result[0].placement_score == 42.5

    def test_team_leaderboard(self, httpserver: HTTPServer, client: TournamentServerClient) -> None:
        httpserver.expect_request("/tournament/seasons/s1/leaderboard/team/stage-1", method="GET").respond_with_json(
            [_team_summary_payload()]
        )
        result = client.get_stage_leaderboard("s1", "team", "stage-1")
        assert len(result) == 1
        assert isinstance(result[0], TeamSummary)
        assert result[0].pool_name == "stage-1"


class TestGetPoolConfig:
    def test_request_and_parse(self, httpserver: HTTPServer, client: TournamentServerClient) -> None:
        config = {"game": {"num_agents": 4}, "map": {"width": 32}}
        httpserver.expect_request("/tournament/seasons/s1/pools/stage-1/config", method="GET").respond_with_json(config)
        result = client.get_pool_config("s1", "stage-1")
        assert result == config


class TestGetMatchArtifacts:
    def test_request_and_parse(self, httpserver: HTTPServer, client: TournamentServerClient) -> None:
        match_id = uuid.UUID(_MATCH_ID)
        pv_id = uuid.UUID(_PV_ID)
        httpserver.expect_request(
            f"/tournament/matches/{_MATCH_ID}/{_PV_ID}/artifacts/logs", method="GET"
        ).respond_with_data("log line 1\nlog line 2\n", content_type="text/plain")
        result = client.get_match_artifacts(match_id, pv_id, "logs")
        assert "log line 1" in result


class TestListEpisodes:
    def test_request_and_parse(self, httpserver: HTTPServer, client: TournamentServerClient) -> None:
        httpserver.expect_request("/episodes", method="GET").respond_with_json([_episode_response_payload()])
        episodes = client.list_episodes()
        assert len(episodes) == 1
        assert isinstance(episodes[0], EpisodeResponse)
        assert episodes[0].id == uuid.UUID(_EPISODE_ID)

    def test_filters(self, httpserver: HTTPServer, client: TournamentServerClient) -> None:
        httpserver.expect_request("/episodes", method="GET").respond_with_json([])
        pv_id = uuid.UUID(_PV_ID)
        result = client.list_episodes(policy_version_id=pv_id, limit=10, offset=5, tags="game:cogs")
        assert result == []
        req, _ = httpserver.log[-1]
        assert req.args["policy_version_id"] == _PV_ID
        assert req.args["limit"] == "10"
        assert req.args["offset"] == "5"
        assert req.args["tags"] == "game:cogs"


class TestGetEpisode:
    def test_request_and_parse(self, httpserver: HTTPServer, client: TournamentServerClient) -> None:
        httpserver.expect_request(f"/episodes/{_EPISODE_ID}", method="GET").respond_with_json(
            _episode_response_payload()
        )
        ep = client.get_episode(uuid.UUID(_EPISODE_ID))
        assert isinstance(ep, EpisodeResponse)
        assert ep.steps == 10000


class TestBrowsePolicies:
    def test_request_and_parse(self, httpserver: HTTPServer, client: TournamentServerClient) -> None:
        httpserver.expect_request("/stats/policies", method="GET").respond_with_json(
            {"entries": [_policy_row_payload()], "total_count": 1}
        )
        result = client.browse_policies()
        assert isinstance(result, PoliciesResponse)
        assert result.total_count == 1

    def test_filters(self, httpserver: HTTPServer, client: TournamentServerClient) -> None:
        httpserver.expect_request("/stats/policies", method="GET").respond_with_json({"entries": [], "total_count": 0})
        result = client.browse_policies(name_exact="alpha", name_fuzzy="alp", limit=20, offset=10)
        assert result.total_count == 0
        req, _ = httpserver.log[-1]
        assert req.args["name_exact"] == "alpha"
        assert req.args["name_fuzzy"] == "alp"
        assert req.args["limit"] == "20"
        assert req.args["offset"] == "10"


# ---------------------------------------------------------------------------
# get_score_policies_leaderboard
# ---------------------------------------------------------------------------


def _episode_query_result_payload() -> dict:
    return {
        "id": _EPISODE_ID,
        "replay_url": "https://example.com/replay/1",
        "thumbnail_url": "https://example.com/thumb/1",
        "attributes": {"game": "cogsguard"},
        "eval_task_id": None,
        "created_at": "2026-02-20T12:00:00Z",
        "tags": {"game": "cogsguard"},
        "avg_rewards": {_PV_ID: 15.5},  # string UUID keys as returned by the server
        "job_id": None,
    }


class TestGetScorePoliciesLeaderboard:
    def test_request_and_parse(self, httpserver: HTTPServer, client: TournamentServerClient) -> None:
        httpserver.expect_request("/tournament/seasons/s1/score-policies-leaderboard", method="GET").respond_with_json(
            [_score_policies_entry_payload()]
        )
        result = client.get_score_policies_leaderboard("s1")
        assert len(result) == 1
        assert isinstance(result[0], ScorePoliciesLeaderboardEntry)
        assert result[0].rank == 1
        assert result[0].placement_score == 42.5
        assert result[0].policy.name == "alpha"


# ---------------------------------------------------------------------------
# query_episodes
# ---------------------------------------------------------------------------


class TestQueryEpisodes:
    def test_basic_request(self, httpserver: HTTPServer, client: TournamentServerClient) -> None:
        httpserver.expect_request("/stats/episodes/query", method="POST").respond_with_json(
            {"episodes": [_episode_query_result_payload()]}
        )
        result = client.query_episodes()
        assert isinstance(result, EpisodeQueryResponse)
        assert len(result.episodes) == 1

    def test_with_policy_version_ids(self, httpserver: HTTPServer, client: TournamentServerClient) -> None:
        httpserver.expect_request("/stats/episodes/query", method="POST").respond_with_json({"episodes": []})
        pv_id = uuid.UUID(_PV_ID)
        result = client.query_episodes(primary_policy_version_ids=[pv_id])
        assert result.episodes == []
        req, _ = httpserver.log[-1]
        body = json.loads(req.data)
        assert body["primary_policy_version_ids"] == [_PV_ID]

    def test_with_all_filters(self, httpserver: HTTPServer, client: TournamentServerClient) -> None:
        httpserver.expect_request("/stats/episodes/query", method="POST").respond_with_json({"episodes": []})
        pv_id = uuid.UUID(_PV_ID)
        ep_id = uuid.UUID(_EPISODE_ID)
        result = client.query_episodes(
            primary_policy_version_ids=[pv_id],
            episode_ids=[ep_id],
            tag_filters={"game": ["cogsguard"]},
            limit=50,
            offset=10,
        )
        assert result.episodes == []
        req, _ = httpserver.log[-1]
        body = json.loads(req.data)
        assert body["primary_policy_version_ids"] == [_PV_ID]
        assert body["episode_ids"] == [_EPISODE_ID]
        assert body["tag_filters"] == {"game": ["cogsguard"]}
        assert body["limit"] == 50
        assert body["offset"] == 10
