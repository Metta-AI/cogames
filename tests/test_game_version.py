"""Regression tests for cogames game-version handling."""

import subprocess

import pytest

from metta.common.tool import game_version as game_version_module
from metta.common.util.fs import get_repo_root


def test_cogames_game_version_alias_env_override_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("METTA_GAME_VERSION_ARENA_BASIC_EASY_SHAPED_LAST_GOOD", "deadbeef")

    assert game_version_module.resolve_game_version("arena_basic_easy_shaped_last_good") == "deadbeef"


def test_cogames_game_version_short_hash_resolution_regression() -> None:
    repo_root = get_repo_root()
    short_result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    full_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )

    resolved = game_version_module.resolve_full_commit_hash(str(repo_root), short_result.stdout.strip())

    assert resolved == full_result.stdout.strip()
