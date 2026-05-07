import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path
from typing import BinaryIO, cast

import pytest
from pytest_httpserver import HTTPServer
from typer.testing import CliRunner

from cogames.coworld.schema_validation import JsonObject
from cogames.coworld.upload import (
    _docker_archive_client_hash,
    _local_image_client_hash,
    _manifest_doc_paths,
    _manifest_with_inlined_docs,
    upload_coworld,
)
from cogames.main import app


def test_upload_coworld_posts_standalone_manifest(
    tmp_path: Path,
    httpserver: HTTPServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    certification_calls: list[tuple[Path, float]] = []
    softmax_image_uri = "123456789012.dkr.ecr.us-east-1.amazonaws.com/cogames/user/unit-test-runtime@sha256:digest"
    pushed_images: list[tuple[str, str]] = []

    monkeypatch.setattr("cogames.coworld.upload.load_current_cogames_token", lambda *, login_server: "token")
    monkeypatch.setattr(
        "cogames.coworld.upload.certify_coworld",
        lambda manifest_path, *, timeout_seconds: certification_calls.append((manifest_path, timeout_seconds)),
    )
    monkeypatch.setattr("cogames.coworld.upload._local_image_client_hash", lambda image: "sha256:client-hash")
    monkeypatch.setattr(
        "cogames.coworld.upload._push_container_image",
        lambda source_image, push_info: pushed_images.append((source_image, push_info.image_uri)),
    )
    httpserver.expect_request(
        "/v2/container_images/upload",
        method="POST",
        headers={"X-Auth-Token": "token"},
        json={"name": "unit-test-runtime", "client_hash": "sha256:client-hash"},
    ).respond_with_json(
        {
            "image": {
                "id": "img_00000000-0000-0000-0000-000000000010",
                "name": "unit-test-runtime",
                "version": 1,
                "client_hash": "sha256:client-hash",
                "status": "pending",
            },
            "pre_signed_info": {
                "kind": "ecr",
                "region": "us-east-1",
                "registry": "123456789012.dkr.ecr.us-east-1.amazonaws.com",
                "repository": "cogames/user/unit-test-runtime",
                "tag": "v1",
                "image_uri": "123456789012.dkr.ecr.us-east-1.amazonaws.com/cogames/user/unit-test-runtime:v1",
                "expires_at": "2026-05-06T22:00:00Z",
                "credentials": {
                    "access_key_id": "access-key",
                    "secret_access_key": "secret-key",
                    "session_token": "session-token",
                },
            },
        }
    )
    httpserver.expect_request(
        "/v2/container_images/upload/complete",
        method="POST",
        headers={"X-Auth-Token": "token"},
        json={"id": "img_00000000-0000-0000-0000-000000000010"},
    ).respond_with_json(
        {
            "id": "img_00000000-0000-0000-0000-000000000010",
            "name": "unit-test-runtime",
            "version": 1,
            "client_hash": "sha256:client-hash",
            "status": "ready",
            "image_uri": softmax_image_uri,
            "image_digest": "sha256:digest",
        }
    )
    httpserver.expect_request(
        "/v2/coworlds/upload",
        method="POST",
        headers={"X-Auth-Token": "token"},
    ).respond_with_json(
        {
            "id": "cow_00000000-0000-0000-0000-000000000001",
            "name": "unit-test-game",
            "version": "0.1.0",
            "manifest": _manifest_with_image(softmax_image_uri),
            "manifest_hash": "sha256:manifest-hash",
            "size_bytes": 1234,
        }
    )

    result = upload_coworld(
        manifest_path,
        server=httpserver.url_for(""),
        login_server="https://softmax.test/api",
    )

    assert result.name == "unit-test-game"
    assert result.version == "0.1.0"
    assert result.id == "cow_00000000-0000-0000-0000-000000000001"
    assert result.manifest_hash == "sha256:manifest-hash"
    assert certification_calls == [(manifest_path.resolve(), 60.0)]
    assert pushed_images == [
        ("unit-test-runtime:latest", "123456789012.dkr.ecr.us-east-1.amazonaws.com/cogames/user/unit-test-runtime:v1")
    ]
    upload_req = next(req for req, _ in httpserver.log if req.path == "/v2/coworlds/upload")
    uploaded_manifest = upload_req.get_json()["manifest"]
    assert uploaded_manifest["game"]["runnable"]["image"] == softmax_image_uri
    assert uploaded_manifest["player"][0]["image"] == softmax_image_uri
    assert uploaded_manifest["game"]["protocols"]["player"] == "# Player Protocol\n"
    assert uploaded_manifest["game"]["protocols"]["global"] == "# Global Protocol\n"


def test_upload_coworld_command_certifies_before_uploading(
    tmp_path: Path,
    httpserver: HTTPServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    certification_calls: list[tuple[Path, float]] = []
    softmax_image_uri = "123456789012.dkr.ecr.us-east-1.amazonaws.com/cogames/user/unit-test-runtime@sha256:digest"

    monkeypatch.setattr("cogames.coworld.upload.load_current_cogames_token", lambda *, login_server: "token")
    monkeypatch.setattr(
        "cogames.coworld.upload.certify_coworld",
        lambda manifest_path, *, timeout_seconds: certification_calls.append((manifest_path, timeout_seconds)),
    )
    monkeypatch.setattr("cogames.coworld.upload._local_image_client_hash", lambda image: "sha256:client-hash")
    monkeypatch.setattr("cogames.coworld.upload._push_container_image", lambda source_image, push_info: None)
    httpserver.expect_request("/v2/container_images/upload", method="POST").respond_with_json(
        {
            "image": {
                "id": "img_00000000-0000-0000-0000-000000000020",
                "name": "unit-test-runtime",
                "version": 1,
                "client_hash": "sha256:client-hash",
                "status": "ready",
                "image_uri": softmax_image_uri,
                "image_digest": "sha256:digest",
            },
            "pre_signed_info": None,
        }
    )
    httpserver.expect_request("/v2/coworlds/upload", method="POST").respond_with_json(
        {
            "id": "cow_00000000-0000-0000-0000-000000000002",
            "name": "unit-test-game",
            "version": "0.1.0",
            "manifest": _manifest_with_image(softmax_image_uri),
            "manifest_hash": "sha256:manifest-hash",
            "size_bytes": 1234,
        }
    )

    result = CliRunner().invoke(
        app,
        [
            "coworld",
            "upload-coworld",
            str(manifest_path),
            "--server",
            httpserver.url_for(""),
            "--login-server",
            "https://softmax.test/api",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Upload complete: unit-test-game:0.1.0" in result.output
    assert "Coworld: cow_00000000-0000-0000-0000-000000000002" in result.output
    assert "Manifest hash: sha256:manifest-hash" in result.output
    assert certification_calls == [(manifest_path.resolve(), 60.0)]


def test_upload_policy_command_creates_docker_image_policy(
    httpserver: HTTPServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    softmax_image_uri = "123456789012.dkr.ecr.us-east-1.amazonaws.com/cogames/user/unit-test-policy@sha256:digest"

    monkeypatch.setattr("cogames.coworld.upload.load_current_cogames_token", lambda *, login_server: "token")
    monkeypatch.setattr("cogames.coworld.upload._local_image_client_hash", lambda image: "sha256:client-hash")
    monkeypatch.setattr("cogames.coworld.upload._push_container_image", lambda source_image, push_info: None)
    httpserver.expect_request(
        "/v2/container_images/upload",
        method="POST",
        headers={"X-Auth-Token": "token"},
        json={"name": "unit-test-policy", "client_hash": "sha256:client-hash"},
    ).respond_with_json(
        {
            "image": {
                "id": "img_00000000-0000-0000-0000-000000000030",
                "name": "unit-test-policy",
                "version": 1,
                "client_hash": "sha256:client-hash",
                "status": "ready",
                "image_uri": softmax_image_uri,
                "image_digest": "sha256:digest",
            },
            "pre_signed_info": None,
        }
    )
    httpserver.expect_request(
        "/stats/policies/docker-img/complete",
        method="POST",
        headers={"X-Auth-Token": "token"},
        json={
            "name": "paintbot",
            "container_image_id": "img_00000000-0000-0000-0000-000000000030",
        },
    ).respond_with_json(
        {
            "id": "00000000-0000-0000-0000-000000000031",
            "name": "paintbot",
            "version": 1,
        }
    )

    result = CliRunner().invoke(
        app,
        [
            "coworld",
            "upload-policy",
            "unit-test-policy:latest",
            "--name",
            "paintbot",
            "--server",
            httpserver.url_for(""),
            "--login-server",
            "https://softmax.test/api",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Upload complete: paintbot:v1" in result.output


def test_local_image_client_hash_uses_docker_archive_content(monkeypatch: pytest.MonkeyPatch) -> None:
    archive = _docker_archive(config=b'{"cmd":["python","game.py"]}', layers=[b"layer-one", b"layer-two"])

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command == ["docker", "image", "save", "unit-test-runtime:latest"]
        stdout = cast(BinaryIO, kwargs["stdout"])
        stdout.write(archive)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("cogames.coworld.upload.subprocess.run", fake_run)

    assert _local_image_client_hash("unit-test-runtime:latest") == _expected_archive_hash(
        b'{"cmd":["python","game.py"]}',
        [b"layer-one", b"layer-two"],
    )


def test_docker_archive_client_hash_matches_config_and_layer_digests() -> None:
    archive = io.BytesIO(_docker_archive(config=b'{"env":["A=B"]}', layers=[b"layer-one"]))

    assert _docker_archive_client_hash(archive) == _expected_archive_hash(b'{"env":["A=B"]}', [b"layer-one"])


def _docker_archive(*, config: bytes, layers: list[bytes]) -> bytes:
    archive = io.BytesIO()
    layer_names = [f"{index}/layer.tar" for index in range(len(layers))]
    with tarfile.open(fileobj=archive, mode="w") as tar:
        _add_tar_file(tar, "manifest.json", json.dumps([{"Config": "config.json", "Layers": layer_names}]).encode())
        _add_tar_file(tar, "config.json", config)
        for name, content in zip(layer_names, layers, strict=True):
            _add_tar_file(tar, name, content)
    return archive.getvalue()


def _add_tar_file(tar: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(content)
    tar.addfile(info, io.BytesIO(content))


def _expected_archive_hash(config: bytes, layers: list[bytes]) -> str:
    content = {
        "config": _sha256_digest(config),
        "layers": [_sha256_digest(layer) for layer in layers],
    }
    return _sha256_digest(json.dumps(content, sort_keys=True, separators=(",", ":")).encode())


def _sha256_digest(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def test_manifest_doc_paths_collects_local_paths() -> None:
    manifest: dict[str, object] = {
        "game": {
            "name": "test",
            "version": "1.0",
            "protocols": {
                "player": "docs/player.md",
                "global": "https://example.com/global.md",
            },
            "docs": {
                "readme": "README.md",
                "pages": [
                    {"id": "setup", "title": "Setup", "content": "docs/setup.md"},
                    {"id": "ext", "title": "External", "content": "https://example.com/ext.md"},
                ],
            },
        },
    }
    paths = _manifest_doc_paths(manifest)
    assert (["game", "protocols", "player"], "docs/player.md") in paths
    assert (["game", "docs", "readme"], "README.md") in paths
    assert (["game", "docs", "pages", 0, "content"], "docs/setup.md") in paths
    assert len(paths) == 3


def test_manifest_with_inlined_docs_reads_file_content(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "player.md").write_text("# Player Protocol\n\nConnect via websocket.\n")
    (docs_dir / "global.md").write_text("# Global Protocol\n")
    (tmp_path / "README.md").write_text("# My Game\n\n![arch](https://example.com/arch.png)\n")

    manifest: dict[str, object] = {
        "game": {
            "name": "test-game",
            "version": "2.0",
            "protocols": {
                "player": "docs/player.md",
                "global": "docs/global.md",
            },
            "docs": {
                "readme": "README.md",
            },
        },
    }

    result = _manifest_with_inlined_docs(manifest, tmp_path)
    game = cast(JsonObject, result["game"])
    protocols = cast(JsonObject, game["protocols"])
    docs = cast(JsonObject, game["docs"])

    assert protocols["player"] == "# Player Protocol\n\nConnect via websocket.\n"
    assert protocols["global"] == "# Global Protocol\n"
    assert docs["readme"] == "# My Game\n\n![arch](https://example.com/arch.png)\n"


def test_manifest_with_inlined_docs_reads_file_uris(tmp_path: Path) -> None:
    protocol_path = tmp_path / "player.md"
    protocol_path.write_text("# Player Protocol\n")
    manifest: dict[str, object] = {
        "game": {
            "name": "test-game",
            "version": "1.0",
            "protocols": {
                "player": protocol_path.as_uri(),
                "global": "https://example.com/global.md",
            },
        },
    }

    result = _manifest_with_inlined_docs(manifest, tmp_path / "unused")
    game = cast(JsonObject, result["game"])
    protocols = cast(JsonObject, game["protocols"])

    assert protocols["player"] == "# Player Protocol\n"
    assert protocols["global"] == "https://example.com/global.md"


def test_manifest_with_inlined_docs_skips_urls() -> None:
    manifest: dict[str, object] = {
        "game": {
            "name": "test-game",
            "version": "1.0",
            "protocols": {
                "player": "https://example.com/player.md",
                "global": "https://example.com/global.md",
            },
        },
    }

    result = _manifest_with_inlined_docs(manifest, Path("/nonexistent"))

    assert result is manifest


def test_manifest_with_inlined_docs_requires_local_files(tmp_path: Path) -> None:
    manifest: dict[str, object] = {
        "game": {
            "name": "test-game",
            "version": "1.0",
            "docs": {
                "readme": "README.md",
            },
        },
    }

    with pytest.raises(FileNotFoundError):
        _manifest_with_inlined_docs(manifest, tmp_path)


def _write_manifest(tmp_path: Path) -> Path:
    world_dir = tmp_path / "world"
    docs_dir = world_dir / "game" / "docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "player_protocol_spec.md").write_text("# Player Protocol\n")
    (docs_dir / "global_protocol_spec.md").write_text("# Global Protocol\n")
    manifest_path = world_dir / "coworld_manifest.json"
    manifest_path.write_text(json.dumps(_manifest()))
    return manifest_path


def _manifest() -> dict[str, object]:
    return {
        "game": {
            "name": "unit-test-game",
            "version": "0.1.0",
            "description": "Unit test Cogame manifest.",
            "owner": "cogames@softmax.com",
            "runnable": {
                "type": "game",
                "image": "unit-test-runtime:latest",
                "run": ["python", "-m", "unit_test.game"],
            },
            "config_schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "additionalProperties": False,
                "required": ["tokens"],
                "properties": {"tokens": {"type": "array", "items": {"type": "string"}}},
            },
            "results_schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "required": ["scores"],
                "properties": {"scores": {"type": "array", "items": {"type": "number"}}},
            },
            "protocols": {
                "player": "game/docs/player_protocol_spec.md",
                "global": "game/docs/global_protocol_spec.md",
            },
        },
        "player": [
            {
                "id": "unit-test-player",
                "name": "Unit Test Player",
                "description": "Unit test player.",
                "type": "player",
                "image": "unit-test-runtime:latest",
                "run": ["python", "-m", "unit_test.player"],
            }
        ],
        "variants": [
            {
                "id": "default",
                "name": "Default",
                "description": "Default unit test variant.",
                "game_config": {"tokens": []},
            }
        ],
        "certification": {"variant_id": "default", "players": [{"player_id": "unit-test-player"}]},
    }


def _manifest_with_image(image: str) -> dict[str, object]:
    manifest = _manifest()
    game = manifest["game"]
    assert isinstance(game, dict)
    runnable = game["runnable"]
    assert isinstance(runnable, dict)
    runnable["image"] = image
    players = manifest["player"]
    assert isinstance(players, list)
    player = players[0]
    assert isinstance(player, dict)
    player["image"] = image
    return manifest
