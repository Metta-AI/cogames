import json
from pathlib import Path

from pytest_httpserver import HTTPServer

from cogames.coworld.manifest_uri import materialized_manifest_path


def test_materialized_manifest_path_accepts_local_paths(tmp_path: Path) -> None:
    manifest_path = tmp_path / "coworld_manifest.json"
    manifest_path.write_text(json.dumps({"game": {}}))

    with materialized_manifest_path(str(manifest_path)) as resolved:
        assert resolved == manifest_path.resolve()


def test_materialized_manifest_path_downloads_raw_manifest(httpserver: HTTPServer) -> None:
    manifest = {"game": {"name": "downloaded"}}
    httpserver.expect_request("/manifest.json").respond_with_json(manifest)

    with materialized_manifest_path(httpserver.url_for("/manifest.json")) as resolved:
        assert json.loads(resolved.read_text()) == manifest


def test_materialized_manifest_path_downloads_coworld_response(httpserver: HTTPServer) -> None:
    manifest = {"game": {"name": "downloaded"}}
    httpserver.expect_request("/v2/coworlds/cow_test").respond_with_json({"manifest": manifest})

    with materialized_manifest_path(httpserver.url_for("/v2/coworlds/cow_test")) as resolved:
        assert json.loads(resolved.read_text()) == manifest
