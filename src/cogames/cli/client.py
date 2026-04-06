from __future__ import annotations

import uuid
from typing import Any, Literal, TypeVar, overload

import httpx
from pydantic import TypeAdapter

from cogames.cli._model_base import CLIModel
from cogames.cli.base import console
from cogames.cli.generated_models import (
    AssayResultsResponse,
    AssayRunResponse,
    EpisodeQueryResponse,
    EpisodeResponse,
    LeaderboardEntry,
    MatchResponse,
    MembershipHistoryEntry,
    MissionSpec,
    PoliciesResponse,
    PolicySummary,
    PolicyVersionResponse,
    PolicyVersionRow,
    PolicyVersionsResponse,
    PresignedUploadUrlResponse,
    ScorePoliciesLeaderboardEntry,
    SeasonDetail,
    SeasonSummary,
    SeasonVersionInfo,
    StageStats,
    SubmitResponse,
    TeamSummary,
    TeamTournamentProgress,
)
from softmax.auth import has_saved_token, load_token
from softmax.token_storage import TokenKind

T = TypeVar("T")


class PoolConfigInfo(CLIModel):
    pool_name: str
    config: dict[str, Any]


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
        if not has_saved_token(token_kind=TokenKind.COGAMES, server=login_server):
            console.print("[red]Error:[/red] Not authenticated.")
            console.print("Please run: [cyan]softmax login[/cyan]")
            return None

        token = load_token(token_kind=TokenKind.COGAMES, server=login_server)
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

    def get_season(self, season_name: str) -> SeasonDetail:
        return self._get(f"/tournament/seasons/{season_name}", SeasonDetail)

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
    ) -> list[PolicyVersionRow]:
        params: dict[str, Any] = {"mine": "true", "limit": 100}
        if name is not None:
            params["name_exact"] = name
        if version is not None:
            params["version"] = version
        result = self._get("/stats/policy-versions", PolicyVersionsResponse, params=params)
        return result.entries

    def lookup_policy_version(
        self,
        name: str,
        version: int | None = None,
    ) -> PolicyVersionRow | None:
        versions = self.get_my_policy_versions(name=name, version=version)
        return versions[0] if versions else None

    def get_policy_version(self, policy_version_id: uuid.UUID) -> PolicyVersionRow:
        return self._get(f"/stats/policy-versions/{policy_version_id}", PolicyVersionRow)

    def submit_to_season(self, season_name: str, policy_version_id: uuid.UUID) -> SubmitResponse:
        return self._post(
            f"/tournament/seasons/{season_name}/submissions",
            SubmitResponse,
            json={"policy_version_id": str(policy_version_id)},
        )

    def get_season_policies(self, season_name: str, mine: bool = False) -> list[PolicySummary]:
        params: dict[str, str] = {}
        if mine:
            params["mine"] = "true"

        return self._get(
            f"/tournament/seasons/{season_name}/policies",
            list[PolicySummary],
            params=params if params else None,
        )

    def get_presigned_upload_url(self) -> PresignedUploadUrlResponse:
        return self._post("/stats/policies/submit/presigned-url", PresignedUploadUrlResponse, timeout=60.0)

    def complete_policy_upload(
        self,
        upload_id: str,
        name: str,
        season: str | None = None,
        secret_env: dict[str, str] | None = None,
    ) -> PolicyVersionResponse:
        payload: dict[str, Any] = {"upload_id": upload_id, "name": name}
        if season is not None:
            payload["season"] = season
        if secret_env:
            payload["policy_secret_env"] = secret_env
        return self._post(
            "/stats/policies/submit/complete",
            PolicyVersionResponse,
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
        return self._get(f"/tournament/matches/{match_id}", MatchResponse)

    def list_match_policy_logs(self, match_id: uuid.UUID, policy_version_id: uuid.UUID) -> list[str]:
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

    # ---------------------------------------------------------------------------
    # Assay endpoints
    # ---------------------------------------------------------------------------

    def create_assay_run(
        self,
        *,
        policy_version_id: uuid.UUID,
        missions: list[MissionSpec],
        name: str | None = None,
        compat_version: str | None = None,
    ) -> AssayRunResponse:
        body: dict = {
            "policy_version_id": str(policy_version_id),
            "missions": [m.model_dump() for m in missions],
        }
        if name is not None:
            body["name"] = name
        if compat_version is not None:
            body["compat_version"] = compat_version
        return self._post("/assay/runs", AssayRunResponse, json=body)

    def get_assay_run(self, run_id: uuid.UUID) -> AssayRunResponse:
        return self._get(f"/assay/runs/{run_id}", AssayRunResponse)

    def list_assay_runs(self, policy_version_id: uuid.UUID | None = None) -> list[AssayRunResponse]:
        params: dict[str, str] = {}
        if policy_version_id is not None:
            params["policy_version_id"] = str(policy_version_id)
        return self._get("/assay/runs", list[AssayRunResponse], params=params if params else None)

    def get_assay_results(self, run_id: uuid.UUID) -> AssayResultsResponse:
        return self._get(f"/assay/runs/{run_id}/results", AssayResultsResponse)

    def finalize_assay_run(self, run_id: uuid.UUID) -> AssayRunResponse:
        return self._post(f"/assay/runs/{run_id}/finalize", AssayRunResponse)

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
