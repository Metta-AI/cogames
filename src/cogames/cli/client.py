from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal, TypeVar, overload

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

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
    version: int = Field(description="Season version number")
    canonical: bool = Field(description="Whether this is the canonical (active) version")
    summary: str = Field(description="Human-readable description of the season")
    entry_pool: str | None = Field(default=None, description="Name of the pool where new policies are submitted")
    leaderboard_pool: str | None = Field(default=None, description="Name of the pool used for the leaderboard")
    is_default: bool = Field(description="Whether this is the default season")
    compat_version: str | None = Field(default=None, description="Compatibility version string (e.g. '0.4')")
    pools: list[PoolInfo] = Field(description="Pools in this season")


class SeasonInfo(SeasonSummary):
    status: Literal["not_started", "in_progress", "complete"]
    display_name: str
    started_at: str | None = None
    tournament_type: Literal["freeplay", "team"]
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


class MembershipHistoryEntry(CLIModel):
    season_name: str
    season_version: int | None
    pool_name: str
    action: str
    notes: str | None
    created_at: str


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

    def get_season(self, season_name: str, include_hidden: bool = False) -> SeasonInfo:
        params = {"include_hidden": "true"} if include_hidden else None
        return self._get(f"/tournament/seasons/{season_name}", SeasonInfo, params=params)

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
        include_hidden_seasons: bool = False,
        policy_version_ids: list[uuid.UUID] | None = None,
        limit: int | None = None,
    ) -> list[MatchResponse]:
        params: dict[str, str | list[str]] = {}

        if include_hidden_seasons:
            params["include_hidden"] = "true"

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

    def get_leaderboard(self, season_name: str, include_hidden_seasons: bool = False) -> list[LeaderboardEntry]:
        return self._get(
            f"/tournament/seasons/{season_name}/leaderboard",
            list[LeaderboardEntry],
            params={"include_hidden": "true"} if include_hidden_seasons else None,
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

    def get_season_policies(
        self, season_name: str, mine: bool = False, include_hidden_seasons: bool = False
    ) -> list[SeasonPolicyEntry]:
        params: dict[str, str] = {}
        if mine:
            params["mine"] = "true"
        if include_hidden_seasons:
            params["include_hidden"] = "true"

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
        """Get the content of a specific agent's policy log in a match."""
        headers = {}
        if self._token:
            headers["X-Auth-Token"] = self._token
        response = self._http_client.get(
            f"/tournament/matches/{match_id}/{policy_version_id}/policy-logs/{agent_idx}",
            headers=headers,
        )
        response.raise_for_status()
        return response.text
