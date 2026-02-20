from uuid import UUID

import pytest
from click.exceptions import Exit

from cogames.cli.client import PoolInfo, SeasonInfo
from cogames.cli.compat import _get_installed_compat, check_compat_version


def _season(compat_version: str | None) -> SeasonInfo:
    return SeasonInfo(
        id=UUID("00000000-0000-0000-0000-000000000001"),
        name="test",
        version=1,
        canonical=True,
        summary="test",
        is_default=False,
        pools=[PoolInfo(name="p", description="d")],
        compat_version=compat_version,
        status="not_started",
        display_name="Test Season",
        tournament_type="policy",
        entrant_count=0,
        active_entrant_count=0,
        match_count=0,
        stage_count=1,
    )


def test_compat_check_matching(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cogames.cli.compat._get_installed_compat", lambda: "0.4")
    check_compat_version(_season("0.4"))


def test_compat_check_null_season() -> None:
    check_compat_version(_season(None))


def test_compat_check_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cogames.cli.compat._get_installed_compat", lambda: "0.3")
    with pytest.raises(Exit):
        check_compat_version(_season("0.4"))


def test_compat_check_unparseable_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cogames.cli.compat._get_installed_compat", lambda: None)
    check_compat_version(_season("0.4"))


def test_get_installed_compat_dev_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("importlib.metadata.version", lambda _pkg: "0.4.2.post1.dev375")
    monkeypatch.setattr("cogames.cli.compat._find_repo_compat_version", lambda: "0.4")
    assert _get_installed_compat() == "0.4"


def test_get_installed_compat_dev_version_no_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("importlib.metadata.version", lambda _pkg: "0.4.2.post1.dev375")
    monkeypatch.setattr("cogames.cli.compat._find_repo_compat_version", lambda: None)
    assert _get_installed_compat() is None


def test_get_installed_compat_release_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("importlib.metadata.version", lambda _pkg: "0.4.2")
    assert _get_installed_compat() == "0.4"
