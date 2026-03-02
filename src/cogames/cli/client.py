from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal, TypeVar, overload

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from cogames.cli.base import console
from cogames.cli.login import CoGamesAuthenticator

T = TypeVar("T")


class CLIModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class PolicyVersionInfo(CLIModel):
    id: uuid.UUID
    policy_id: uuid.UUID
    created_at: datetime
    policy_created_at: datetime
    user_id: str
    name: str
    version: int


class PoolInfo(CLIModel):
    id: uuid.UUID | None = None
    name: str
    description: str
    config_id: uuid.UUID | None = None


class SeasonSummary(CLIModel):
    id: uuid.UUID = Field(description="Unique season identifier")
    name: str = Field(description="Short name of the season")
    display_name: str | None = Field(default=None, description="Human-readable season title for UI display")
    version: int = Field(description="Season version number")
    canonical: bool = Field(description="Whether this is the canonical (active) version")
    summary: str = Field(description="Human-readable description of the season")
    entry_pool: str | None = Field(default=None, description="Name of the pool where new policies are submitted")
    leaderboard_pool: str | None = Field(default=None, description="Name of the pool used for the leaderboard")
    is_default: bool = Field(description="Whether this is the default season")
    compat_version: str | None = Field(default=None, description="Compatibility version string (e.g. '0.4')")
    created_at: str | None = Field(default=None, description="ISO 8601 timestamp when this season version was created")
    public: bool | None = Field(default=None, description="Whether this season is visible to non-Softmax users")
    tournament_type: Literal["freeplay", "team"] | None = Field(default=None, description="Tournament format")
    pools: list[PoolInfo] = Field(description="Pools in this season")


class SeasonInfo(SeasonSummary):
    status: Literal["not_started", "in_progress", "complete"]
    display_name: str = Field(default=..., description="Human-readable season title for UI display")
    started_at: str | None = None
    tournament_type: Literal["freeplay", "team"] = Field(default=..., description="Tournament format")
    entrant_count: int
    active_entrant_count: int
    match_count: int
    stage_count: int


class MatchPlayerInfo(CLIModel):
    policy: PolicyVersionSummary
    num_agents: int
    score: float | None


class MatchResponse(CLIModel):
    id: uuid.UUID
    season_name: str
    pool_name: str
    status: str
    assignments: list[int]
    players: list[MatchPlayerInfo]
    error: str | None
    episode_id: uuid.UUID | None
    job_id: uuid.UUID | None = None
    episode: dict[str, Any] | None = None
    created_at: datetime


class SeasonVersionInfo(CLIModel):
    version: int
    canonical: bool
    disabled_at: str | None
    created_at: str
    compat_version: str | None = None


class LeaderboardEntry(CLIModel):
    rank: int
    policy: PolicyVersionSummary
    score: float
    score_stddev: float | None = None
    matches: int


class PolicyVersionSummary(CLIModel):
    id: uuid.UUID
    name: str | None
    version: int | None


class SubmitToSeasonResponse(CLIModel):
    pools: list[str]


class PoolMembership(CLIModel):
    pool_name: str
    active: bool
    completed: int
    failed: int
    pending: int


class SeasonPolicyEntry(CLIModel):
    policy: PolicyVersionSummary
    pools: list[PoolMembership]
    entered_at: str


class PoolConfigInfo(CLIModel):
    pool_name: str
    config: dict[str, Any]


class MembershipHistoryEntry(CLIModel):
    season_name: str
    season_version: int | None
    pool_name: str
    action: str
    notes: str | None
    created_at: str


class StageStats(CLIModel):
    name: str
    policy_count: int = 0
    match_count: int = 0
    completion_pct: float = 0.0
    team_count: int | None = None
    eliminated_count: int | None = None


class ProgressStage(CLIModel):
    index: int
    name: str
    kind: str
    description: str
    input_pool: str
    output_pool: str
    status: str


class TeamTournamentProgress(CLIModel):
    phase: str
    phase_detail: dict[str, str | int] = Field(default_factory=dict)
    stages: list[StageStats]
    stage_flow: list[ProgressStage] = Field(default_factory=list)
    started: bool = False


class TeamCogSummary(CLIModel):
    position: int
    policy: PolicyVersionSummary


class TeamSummary(CLIModel):
    id: uuid.UUID | None = None
    name: str | None = None
    pool_name: str
    eliminated: bool | None = None
    score: float | None = None
    matches: int | None = None
    rank: int | None = None
    members: list[PolicyVersionSummary] = Field(default_factory=list)
    cogs: list[TeamCogSummary] = Field(default_factory=list)
    created_at: str | None = None

    @model_validator(mode="after")
    def _populate_cogs_from_members(self) -> TeamSummary:
        """Normalize legacy team payloads that use `members` into the modern `cogs` shape."""
        if not self.cogs and self.members:
            self.cogs = [TeamCogSummary(position=index, policy=member) for index, member in enumerate(self.members)]
        return self


class ScorePoliciesLeaderboardEntry(CLIModel):
    rank: int
    policy: PolicyVersionSummary
    placement_score: float
    team_appearances: int
    team_ranks: list[int]


class AgentResult(CLIModel):
    agent_id: int
    reward: float
    metrics: dict[str, float]


class PolicyResult(CLIModel):
    position: int
    policy: PolicyVersionSummary
    num_agents: int
    avg_reward: float
    avg_metrics: dict[str, float]
    agents: list[AgentResult]


class EpisodeResponse(CLIModel):
    id: uuid.UUID
    replay_url: str | None = None
    thumbnail_url: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    game_stats: dict[str, float] = Field(default_factory=dict)
    policy_results: list[PolicyResult] = Field(default_factory=list)
    steps: int | None = None
    created_at: datetime


class EpisodeQueryResult(CLIModel):
    id: uuid.UUID
    replay_url: str | None = None
    thumbnail_url: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    eval_task_id: uuid.UUID | None = None
    created_at: datetime
    tags: dict[str, str] = Field(default_factory=dict)
    avg_rewards: dict[uuid.UUID, float] = Field(default_factory=dict)
    job_id: uuid.UUID | None = None


class EpisodeQueryResponse(CLIModel):
    episodes: list[EpisodeQueryResult]


class PolicyRow(CLIModel):
    id: uuid.UUID
    name: str
    created_at: datetime
    user_id: str
    attributes: dict[str, Any]
    version_count: int


class PoliciesResponse(CLIModel):
    entries: list[PolicyRow]
    total_count: int


class TournamentServerClient:
    def __init__(
        self,
        server_url: str,
        token: str | None = None,
        login_server: str | None = None,
    ):
        self._server_url = server_url
        self._token = token
        self._login_server = login_server
        self._http_client = httpx.Client(base_url=server_url, timeout=30.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        self.close()

    def close(self):
        self._http_client.close()

    @classmethod
    def from_login(cls, server_url: str, login_server: str) -> TournamentServerClient | None:
        authenticator = CoGamesAuthenticator()
        if not authenticator.has_saved_token(login_server):
            console.print("[red]Error:[/red] Not authenticated.")
            console.print("Please run: [cyan]cogames login[/cyan]")
            return None

        token = authenticator.load_token(login_server)
        if not token:
            console.print(f"[red]Error:[/red] Token not found for {login_server}")
            return None

        return cls(server_url=server_url, token=token, login_server=login_server)

    def _request(
        self,
        method: str,
        path: str,
        response_type: type[T] | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> T | dict[str, Any]:
        headers = kwargs.pop("headers", {})
        if self._token:
            headers["X-Auth-Token"] = self._token

        if timeout is not None:
            kwargs["timeout"] = timeout

        response = self._http_client.request(method, path, headers=headers, **kwargs)
        response.raise_for_status()

        if response_type is not None:
            return TypeAdapter(response_type).validate_python(response.json())
        return response.json()

    @overload
    def _get(self, path: str, response_type: type[T], **kwargs: Any) -> T: ...
    @overload
    def _get(self, path: str, response_type: None = None, **kwargs: Any) -> dict[str, Any]: ...
    def _get(self, path: str, response_type: type[T] | None = None, **kwargs: Any) -> T | dict[str, Any]:
        return self._request("GET", path, response_type, **kwargs)

    @overload
    def _post(self, path: str, response_type: type[T], **kwargs: Any) -> T: ...
    @overload
    def _post(self, path: str, response_type: None = None, **kwargs: Any) -> dict[str, Any]: ...
    def _post(self, path: str, response_type: type[T] | None = None, **kwargs: Any) -> T | dict[str, Any]:
        return self._request("POST", path, response_type, **kwargs)

    @overload
    def _put(self, path: str, response_type: type[T], **kwargs: Any) -> T: ...
    @overload
    def _put(self, path: str, response_type: None = None, **kwargs: Any) -> dict[str, Any]: ...
    def _put(self, path: str, response_type: type[T] | None = None, **kwargs: Any) -> T | dict[str, Any]:
        return self._request("PUT", path, response_type, **kwargs)

    def get_seasons(self) -> list[SeasonSummary]:
        return self._get("/tournament/seasons", list[SeasonSummary])

    def get_season(self, season_name: str) -> SeasonInfo:
        return self._get(f"/tournament/seasons/{season_name}", SeasonInfo)

    def get_default_season(self) -> SeasonSummary:
        seasons = self.get_seasons()
        for s in seasons:
            if s.is_default:
                return s
        if seasons:
            return seasons[0]
        raise RuntimeError("No seasons available from server")

    def get_season_matches(
        self,
        season_name: str,
        policy_version_ids: list[uuid.UUID] | None = None,
        limit: int | None = None,
    ) -> list[MatchResponse]:
        params: dict[str, str | list[str]] = {}

        if policy_version_ids:
            params["policy_version_ids"] = [str(pvid) for pvid in policy_version_ids]

        if limit is not None:
            params["limit"] = str(limit)

        return self._get(
            f"/tournament/seasons/{season_name}/matches",
            list[MatchResponse],
            params=params if params else None,
        )

    def get_config(self, config_id: uuid.UUID | str) -> dict[str, Any]:
        return self._get(f"/tournament/configs/{config_id}")

    def get_season_versions(self, season_name: str) -> list[SeasonVersionInfo]:
        return self._get(f"/tournament/seasons/{season_name}/versions", list[SeasonVersionInfo])

    def get_leaderboard(self, season_name: str) -> list[LeaderboardEntry]:
        return self._get(
            f"/tournament/seasons/{season_name}/leaderboard",
            list[LeaderboardEntry],
        )

    def get_my_policy_versions(
        self,
        name: str | None = None,
        version: int | None = None,
    ) -> list[PolicyVersionInfo]:
        params: dict[str, Any] = {"mine": "true", "limit": 100}
        if name is not None:
            params["name_exact"] = name
        if version is not None:
            params["version"] = version
        result = self._get("/stats/policy-versions", params=params)
        entries = result.get("entries", [])
        return [PolicyVersionInfo.model_validate(e) for e in entries]

    def lookup_policy_version(
        self,
        name: str,
        version: int | None = None,
    ) -> PolicyVersionInfo | None:
        versions = self.get_my_policy_versions(name=name, version=version)
        return versions[0] if versions else None

    def get_policy_version(self, policy_version_id: uuid.UUID) -> PolicyVersionInfo:
        return self._get(f"/stats/policy-versions/{policy_version_id}", PolicyVersionInfo)

    def submit_to_season(self, season_name: str, policy_version_id: uuid.UUID) -> SubmitToSeasonResponse:
        return self._post(
            f"/tournament/seasons/{season_name}/submissions",
            SubmitToSeasonResponse,
            json={"policy_version_id": str(policy_version_id)},
        )

    def get_season_policies(self, season_name: str, mine: bool = False) -> list[SeasonPolicyEntry]:
        params: dict[str, str] = {}
        if mine:
            params["mine"] = "true"

        return self._get(
            f"/tournament/seasons/{season_name}/policies",
            list[SeasonPolicyEntry],
            params=params if params else None,
        )

    def get_presigned_upload_url(self) -> dict[str, Any]:
        return self._post("/stats/policies/submit/presigned-url", timeout=60.0)

    def complete_policy_upload(self, upload_id: str, name: str, season: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"upload_id": upload_id, "name": name}
        if season is not None:
            payload["season"] = season
        return self._post(
            "/stats/policies/submit/complete",
            timeout=120.0,
            json=payload,
        )

    def get_policy_memberships(self, policy_version_id: uuid.UUID) -> list[MembershipHistoryEntry]:
        return self._get(f"/tournament/policies/{policy_version_id}/memberships", list[MembershipHistoryEntry])

    def get_my_memberships(self) -> dict[str, list[str]]:
        """Get all season memberships for the user's policy versions.

        Returns a mapping of policy_version_id -> list of season names.
        """
        return self._get("/tournament/my-memberships", dict[str, list[str]])

    def get_pool_config(self, season_name: str, pool_name: str) -> PoolConfigInfo | dict[str, Any]:
        result = self._get(f"/tournament/seasons/{season_name}/pools/{pool_name}/config")
        if isinstance(result, dict):
            response_pool_name = result.get("pool_name")
            response_config = result.get("config")
            if isinstance(response_pool_name, str) and isinstance(response_config, dict):
                return PoolConfigInfo(pool_name=response_pool_name, config=response_config)
        return result

    def get_match(self, match_id: uuid.UUID) -> MatchResponse:
        """Get details for a specific match."""
        return self._get(f"/tournament/matches/{match_id}", MatchResponse)

    def list_match_policy_logs(self, match_id: uuid.UUID, policy_version_id: uuid.UUID) -> list[str]:
        """List available policy log files for a policy in a match."""
        return self._get(
            f"/tournament/matches/{match_id}/{policy_version_id}/policy-logs",
            list[str],
        )

    def get_match_policy_log(self, match_id: uuid.UUID, policy_version_id: uuid.UUID, agent_idx: int) -> str:
        headers = {}
        if self._token:
            headers["X-Auth-Token"] = self._token
        response = self._http_client.get(
            f"/tournament/matches/{match_id}/{policy_version_id}/policy-logs/{agent_idx}",
            headers=headers,
        )
        response.raise_for_status()
        return response.text

    def get_stages(self, season_name: str) -> list[StageStats]:
        return self._get(
            f"/tournament/seasons/{season_name}/stages",
            list[StageStats],
        )

    def get_progress(self, season_name: str) -> TeamTournamentProgress:
        return self._get(
            f"/tournament/seasons/{season_name}/progress",
            TeamTournamentProgress,
        )

    def get_teams(
        self,
        season_name: str,
        *,
        limit: int | None = None,
        offset: int | None = None,
        pool_name: str | None = None,
        eliminated: bool | None = None,
        policy_version_id: uuid.UUID | None = None,
    ) -> list[TeamSummary]:
        params: dict[str, str] = {}
        if limit is not None:
            params["limit"] = str(limit)
        if offset is not None:
            params["offset"] = str(offset)
        if pool_name is not None:
            params["pool_name"] = pool_name
        if eliminated is not None:
            params["eliminated"] = str(eliminated).lower()
        if policy_version_id is not None:
            params["policy_version_id"] = str(policy_version_id)
        return self._get(
            f"/tournament/seasons/{season_name}/teams",
            list[TeamSummary],
            params=params if params else None,
        )

    @overload
    def get_stage_leaderboard(
        self,
        season_name: str,
        leaderboard_type: Literal["policy"],
        pool_name: str,
    ) -> list[LeaderboardEntry]: ...
    @overload
    def get_stage_leaderboard(
        self,
        season_name: str,
        leaderboard_type: Literal["score-policies"],
        pool_name: str,
    ) -> list[ScorePoliciesLeaderboardEntry]: ...
    @overload
    def get_stage_leaderboard(
        self,
        season_name: str,
        leaderboard_type: Literal["team"],
        pool_name: str,
    ) -> list[TeamSummary]: ...
    @overload
    def get_stage_leaderboard(
        self,
        season_name: str,
        leaderboard_type: str,
        pool_name: str,
    ) -> list[LeaderboardEntry] | list[ScorePoliciesLeaderboardEntry] | list[TeamSummary]: ...
    def get_stage_leaderboard(
        self,
        season_name: str,
        leaderboard_type: str,
        pool_name: str,
    ) -> list[LeaderboardEntry] | list[ScorePoliciesLeaderboardEntry] | list[TeamSummary]:
        if leaderboard_type == "score-policies":
            return self._get(
                f"/tournament/seasons/{season_name}/leaderboard/{leaderboard_type}/{pool_name}",
                list[ScorePoliciesLeaderboardEntry],
            )
        if leaderboard_type == "team":
            return self._get(
                f"/tournament/seasons/{season_name}/leaderboard/{leaderboard_type}/{pool_name}",
                list[TeamSummary],
            )
        return self._get(
            f"/tournament/seasons/{season_name}/leaderboard/{leaderboard_type}/{pool_name}",
            list[LeaderboardEntry],
        )

    def get_match_artifact(
        self, match_id: uuid.UUID, policy_version_id: uuid.UUID, artifact_type: str
    ) -> httpx.Response:
        headers = {}
        if self._token:
            headers["X-Auth-Token"] = self._token
        response = self._http_client.get(
            f"/tournament/matches/{match_id}/{policy_version_id}/artifacts/{artifact_type}",
            headers=headers,
        )
        response.raise_for_status()
        return response

    def get_match_artifacts(self, match_id: uuid.UUID, policy_version_id: uuid.UUID, artifact_type: str) -> str:
        return self.get_match_artifact(match_id, policy_version_id, artifact_type).text

    def list_episodes(
        self,
        policy_version_id: uuid.UUID | None = None,
        tags: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[EpisodeResponse]:
        params: dict[str, str] = {"limit": str(limit), "offset": str(offset)}
        if policy_version_id:
            params["policy_version_id"] = str(policy_version_id)
        if tags:
            params["tags"] = tags
        return self._get("/episodes", list[EpisodeResponse], params=params)

    def get_episode(self, episode_id: uuid.UUID) -> EpisodeResponse:
        return self._get(f"/episodes/{episode_id}", EpisodeResponse)

    def get_score_policies_leaderboard(
        self,
        season_name: str,
    ) -> list[ScorePoliciesLeaderboardEntry]:
        return self._get(
            f"/tournament/seasons/{season_name}/score-policies-leaderboard",
            list[ScorePoliciesLeaderboardEntry],
        )

    def query_episodes(
        self,
        *,
        primary_policy_version_ids: list[uuid.UUID] | None = None,
        policy_version_ids: list[uuid.UUID] | None = None,
        episode_ids: list[uuid.UUID] | None = None,
        tag_filters: dict[str, list[str] | None] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> EpisodeQueryResponse:
        body: dict[str, Any] = {}
        pvids = primary_policy_version_ids if primary_policy_version_ids is not None else policy_version_ids
        if pvids is not None:
            body["primary_policy_version_ids"] = [str(pvid) for pvid in pvids]
        if episode_ids is not None:
            body["episode_ids"] = [str(eid) for eid in episode_ids]
        if tag_filters is not None:
            body["tag_filters"] = tag_filters
        if limit is not None:
            body["limit"] = limit
        if offset is not None:
            body["offset"] = offset
        return self._post("/stats/episodes/query", EpisodeQueryResponse, json=body)

    def browse_policies(
        self,
        *,
        name_exact: str | None = None,
        name_fuzzy: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> PoliciesResponse:
        params: dict[str, str] = {}
        if name_exact is not None:
            params["name_exact"] = name_exact
        if name_fuzzy is not None:
            params["name_fuzzy"] = name_fuzzy
        if limit is not None:
            params["limit"] = str(limit)
        if offset is not None:
            params["offset"] = str(offset)
        return self._get(
            "/stats/policies",
            PoliciesResponse,
            params=params if params else None,
        )
