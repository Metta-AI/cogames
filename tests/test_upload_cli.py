"""Tests for cogames upload CLI command."""

import io
import json
import os
import subprocess
import uuid
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import typer
from pytest_httpserver import HTTPServer
from typer.testing import CliRunner
from werkzeug import Response

from cogames.auth import AuthConfigReaderWriter
from cogames.cli.submit import ensure_docker_daemon_access
from cogames.main import app
from mettagrid.config.mettagrid_config import MettaGridConfig

_SEASON_ID = "11111111-1111-1111-1111-111111111111"
_ENTRY_CONFIG_ID = "22222222-2222-2222-2222-222222222222"


def _season_info_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": summary["id"],
        "name": summary["name"],
        "version": summary["version"],
        "canonical": summary["canonical"],
        "is_default": summary["is_default"],
        "status": "in_progress",
        "display_name": summary["name"],
        "tournament_type": "freeplay",
        "entrant_count": 1,
        "active_entrant_count": 1,
        "match_count": 0,
        "stage_count": 1,
        "entry_pool": summary["entry_pool"],
        "leaderboard_pool": summary["leaderboard_pool"],
        "summary": summary["summary"],
        "pools": summary["pools"],
    }


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a fake HOME directory with a pre-configured auth token."""
    monkeypatch.setenv("HOME", str(tmp_path))

    writer = AuthConfigReaderWriter("cogames.yaml", "login_tokens")
    writer.save_token("test-token-12345", "http://fake-login-server")

    return tmp_path


def test_upload_command_sends_correct_requests(
    httpserver: HTTPServer,
    fake_home: Path,
    tmp_path: Path,
) -> None:
    """Test that 'cogames upload' sends the expected requests to the server."""
    upload_id = "test-upload-id-abc"
    _setup_mock_upload_server(httpserver, upload_id=upload_id)

    # Run the upload command
    result = subprocess.run(
        [
            "cogames",
            "upload",
            "--policy",
            "class=cogames.policy.starter_agent.StarterPolicy",
            "--name",
            "my-test-policy",
            "--server",
            httpserver.url_for(""),
            "--login-server",
            "http://fake-login-server",
            "--skip-validation",  # Skip isolated validation to speed up test
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            "HOME": str(fake_home),
            "PATH": os.environ.get("PATH", ""),
        },
    )

    assert result.returncode == 0, f"Upload failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert "my-test-policy:v1" in result.stdout

    # Verify the requests that were made
    # 0: /tournament/seasons, 1: /tournament/seasons/{name}, 2: presigned-url, 3: upload, 4: complete
    assert len(httpserver.log) == 5, f"Expected 5 requests, got {len(httpserver.log)}"

    presign_req, _ = httpserver.log[2]
    assert presign_req.headers.get("X-Auth-Token") == "test-token-12345"

    upload_req, _ = httpserver.log[3]
    with zipfile.ZipFile(io.BytesIO(upload_req.data)) as zf:
        assert "policy_spec.json" in zf.namelist()
        spec = json.loads(zf.read("policy_spec.json"))
        assert spec["class_path"] == "cogames.policy.starter_agent.StarterPolicy"

    complete_req, _ = httpserver.log[4]
    complete_body = complete_req.json
    assert complete_body["upload_id"] == upload_id
    assert complete_body["name"] == "my-test-policy"


def test_upload_command_fails_without_auth(
    httpserver: HTTPServer,
    tmp_path: Path,
) -> None:
    """Test that 'cogames upload' fails gracefully when not authenticated."""
    season_summary = {
        "id": _SEASON_ID,
        "name": "test-season",
        "version": 1,
        "canonical": True,
        "is_default": True,
        "entry_pool": None,
        "leaderboard_pool": None,
        "summary": "",
        "pools": [],
    }

    httpserver.expect_request(
        "/tournament/seasons",
        method="GET",
    ).respond_with_json([season_summary])
    httpserver.expect_request(
        "/tournament/seasons/test-season",
        method="GET",
    ).respond_with_json(_season_info_from_summary(season_summary))

    # Use tmp_path as HOME but don't create any token file
    result = subprocess.run(
        [
            "cogames",
            "upload",
            "--policy",
            "class=cogames.policy.starter_agent.StarterPolicy",
            "--name",
            "my-test-policy",
            "--server",
            httpserver.url_for(""),
            "--login-server",
            "http://fake-login-server",
            "--skip-validation",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            "HOME": str(tmp_path),
            "PATH": os.environ.get("PATH", ""),
        },
    )

    # Should show auth error message
    # Note: Currently returns exit code 0 even on auth failure - ideally this would be non-zero
    combined_output = (result.stdout + result.stderr).lower()
    assert "not authenticated" in combined_output or "cogames login" in combined_output


def _setup_mock_upload_server(
    httpserver: HTTPServer,
    upload_id: str = "test-upload-id",
) -> None:
    """Configure httpserver with the endpoints needed for upload."""
    season_summary = {
        "id": _SEASON_ID,
        "name": "test-season",
        "version": 1,
        "canonical": True,
        "is_default": True,
        "entry_pool": None,
        "leaderboard_pool": None,
        "summary": "",
        "pools": [],
    }

    httpserver.expect_request(
        "/tournament/seasons",
        method="GET",
    ).respond_with_json([season_summary])
    httpserver.expect_request(
        "/tournament/seasons/test-season",
        method="GET",
    ).respond_with_json(_season_info_from_summary(season_summary))

    httpserver.expect_request(
        "/stats/policies/submit/presigned-url",
        method="POST",
    ).respond_with_json(
        {
            "upload_url": httpserver.url_for("/fake-s3-upload"),
            "upload_id": upload_id,
        }
    )

    httpserver.expect_request(
        "/fake-s3-upload",
        method="PUT",
    ).respond_with_data("")

    def handle_complete(request):
        body = request.json
        return Response(
            json.dumps(
                {
                    "id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                    "name": body["name"],
                    "version": 1,
                }
            ),
            content_type="application/json",
        )

    httpserver.expect_request(
        "/stats/policies/submit/complete",
        method="POST",
    ).respond_with_handler(handle_complete)


def test_upload_directory_policy(
    httpserver: HTTPServer,
    fake_home: Path,
    tmp_path: Path,
) -> None:
    """Test uploading a policy from a local directory."""
    _setup_mock_upload_server(httpserver)

    # Create a policy directory with policy_spec.json
    policy_dir = tmp_path / "my_policy"
    policy_dir.mkdir()
    policy_spec = {
        "class_path": "my_policies.CustomAgent",
        "data_path": "weights.pt",
        "init_kwargs": {"hidden_size": 256},
    }
    (policy_dir / "policy_spec.json").write_text(json.dumps(policy_spec))
    (policy_dir / "weights.pt").write_bytes(b"fake weights data")
    (policy_dir / "config.yaml").write_text("learning_rate: 0.001\nbatch_size: 32\n")

    result = subprocess.run(
        [
            "cogames",
            "upload",
            "--policy",
            policy_dir.as_uri(),
            "--name",
            "test-policy",
            "--server",
            httpserver.url_for(""),
            "--login-server",
            "http://fake-login-server",
            "--skip-validation",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            "HOME": str(fake_home),
            "PATH": os.environ.get("PATH", ""),
        },
    )

    assert result.returncode == 0, f"Upload failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    # Verify the uploaded zip contains all files from the directory
    assert len(httpserver.log) == 5
    upload_req, _ = httpserver.log[3]
    with zipfile.ZipFile(io.BytesIO(upload_req.data)) as zf:
        assert "policy_spec.json" in zf.namelist()
        assert "weights.pt" in zf.namelist()
        assert "config.yaml" in zf.namelist()
        spec = json.loads(zf.read("policy_spec.json"))
        assert spec["class_path"] == "my_policies.CustomAgent"
        assert spec["data_path"] == "weights.pt"
        assert zf.read("config.yaml").decode() == "learning_rate: 0.001\nbatch_size: 32\n"


def test_upload_zip_policy(
    httpserver: HTTPServer,
    fake_home: Path,
    tmp_path: Path,
) -> None:
    """Test uploading a policy from a local zip file."""
    _setup_mock_upload_server(httpserver)

    # Create a policy zip file with policy_spec.json and model weights
    policy_zip = tmp_path / "my_policy.zip"
    policy_spec = {
        "class_path": "my_policies.TrainedAgent",
        "data_path": "model.safetensors",
        "init_kwargs": {"num_layers": 4},
    }
    with zipfile.ZipFile(policy_zip, "w") as zf:
        zf.writestr("policy_spec.json", json.dumps(policy_spec))
        zf.writestr("model.safetensors", b"fake model data")
        zf.writestr("hyperparams.json", json.dumps({"lr": 1e-4, "epochs": 100}))

    result = subprocess.run(
        [
            "cogames",
            "upload",
            "--policy",
            policy_zip.as_uri(),
            "--name",
            "test-policy",
            "--server",
            httpserver.url_for(""),
            "--login-server",
            "http://fake-login-server",
            "--skip-validation",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            "HOME": str(fake_home),
            "PATH": os.environ.get("PATH", ""),
        },
    )

    assert result.returncode == 0, f"Upload failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    # Verify the uploaded zip contains all files
    assert len(httpserver.log) == 5
    upload_req, _ = httpserver.log[3]
    with zipfile.ZipFile(io.BytesIO(upload_req.data)) as zf:
        assert "policy_spec.json" in zf.namelist()
        assert "model.safetensors" in zf.namelist()
        assert "hyperparams.json" in zf.namelist()
        spec = json.loads(zf.read("policy_spec.json"))
        assert spec["class_path"] == "my_policies.TrainedAgent"
        assert spec["data_path"] == "model.safetensors"
        hyperparams = json.loads(zf.read("hyperparams.json"))
        assert hyperparams == {"lr": 1e-4, "epochs": 100}


def test_upload_zip_policy_with_includes_and_setup_script(
    httpserver: HTTPServer,
    fake_home: Path,
    tmp_path: Path,
) -> None:
    """Test augmenting an existing zip URI with include-files, setup script, and init kwargs."""
    _setup_mock_upload_server(httpserver)

    policy_zip = tmp_path / "my_policy.zip"
    policy_spec = {
        "class_path": "my_policies.TrainedAgent",
        "data_path": "model.safetensors",
        "init_kwargs": {"num_layers": 4},
    }
    with zipfile.ZipFile(policy_zip, "w") as zf:
        zf.writestr("policy_spec.json", json.dumps(policy_spec))
        zf.writestr("model.safetensors", b"fake model data")

    extra_pkg = tmp_path / "extra" / "pkg"
    extra_pkg.mkdir(parents=True)
    (tmp_path / "extra" / "__init__.py").write_text("")
    (extra_pkg / "__init__.py").write_text("")
    (extra_pkg / "module.py").write_text("class Extra: pass\n")
    (tmp_path / "setup_script.py").write_text("print('setup')\n")

    result = subprocess.run(
        [
            "cogames",
            "upload",
            "--policy",
            policy_zip.as_uri(),
            "--name",
            "test-policy",
            "--server",
            httpserver.url_for(""),
            "--login-server",
            "http://fake-login-server",
            "--skip-validation",
            "--include-files",
            "extra/pkg",
            "--setup-script",
            "setup_script.py",
            "--init-kwarg",
            "dropout=0.1",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
        env={
            "HOME": str(fake_home),
            "PATH": os.environ.get("PATH", ""),
        },
    )

    assert result.returncode == 0, f"Upload failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    assert len(httpserver.log) == 5
    upload_req, _ = httpserver.log[3]
    with zipfile.ZipFile(io.BytesIO(upload_req.data)) as zf:
        names = set(zf.namelist())
        spec = json.loads(zf.read("policy_spec.json"))
        assert "model.safetensors" in names
        assert "extra/__init__.py" in names
        assert "extra/pkg/__init__.py" in names
        assert "extra/pkg/module.py" in names
        assert "setup_script.py" in names
        assert spec["setup_script"] == "setup_script.py"
        assert spec["init_kwargs"] == {"num_layers": 4, "dropout": "0.1"}


def test_upload_s3_policy(
    httpserver: HTTPServer,
    fake_home: Path,
    tmp_path: Path,
) -> None:
    """Test uploading a policy from an S3 URI."""
    _setup_mock_upload_server(httpserver)

    # Use a unique S3 key to avoid caching across test runs
    unique_key = f"policies/agent-{uuid.uuid4().hex[:8]}.zip"

    # Create a policy zip in memory
    policy_spec = {
        "class_path": "my_policies.S3Agent",
        "data_path": "checkpoint.pt",
        "init_kwargs": {"from_s3": True},
    }
    policy_bytes = io.BytesIO()
    with zipfile.ZipFile(policy_bytes, "w") as zf:
        zf.writestr("policy_spec.json", json.dumps(policy_spec))
        zf.writestr("checkpoint.pt", b"fake checkpoint data")
        zf.writestr("training_config.yaml", "epochs: 50\nlr: 0.0003\n")
    policy_bytes.seek(0)

    # Mock S3 GetObject endpoint (path-style: GET /bucket/key)
    httpserver.expect_request(
        f"/test-bucket/{unique_key}",
        method="GET",
    ).respond_with_data(policy_bytes.read(), content_type="application/zip")

    result = subprocess.run(
        [
            "cogames",
            "upload",
            "--policy",
            f"s3://test-bucket/{unique_key}",
            "--name",
            "test-s3-policy",
            "--server",
            httpserver.url_for(""),
            "--login-server",
            "http://fake-login-server",
            "--skip-validation",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            "HOME": str(fake_home),
            "PATH": os.environ.get("PATH", ""),
            "AWS_ENDPOINT_URL_S3": httpserver.url_for(""),
            "AWS_ACCESS_KEY_ID": "testing",
            "AWS_SECRET_ACCESS_KEY": "testing",
        },
    )

    assert result.returncode == 0, f"Upload failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    # Verify: seasons list + season detail + S3 download + 3 upload requests = 6 total
    assert len(httpserver.log) == 6, f"Expected 6 requests, got {len(httpserver.log)}"

    # Third request should be the S3 GetObject (after seasons list + season detail)
    s3_req, _ = httpserver.log[2]
    assert s3_req.path == f"/test-bucket/{unique_key}"

    # Verify the uploaded zip contains the files from S3
    upload_req, _ = httpserver.log[4]  # After seasons list/detail, S3 download, presigned URL, then upload
    with zipfile.ZipFile(io.BytesIO(upload_req.data)) as zf:
        assert "policy_spec.json" in zf.namelist()
        assert "checkpoint.pt" in zf.namelist()
        assert "training_config.yaml" in zf.namelist()
        spec = json.loads(zf.read("policy_spec.json"))
        assert spec["class_path"] == "my_policies.S3Agent"
        assert zf.read("training_config.yaml").decode() == "epochs: 50\nlr: 0.0003\n"


# ---------------------------------------------------------------------------
# CliRunner-based tests for upload validation flow
# ---------------------------------------------------------------------------

_SEASON_WITH_ENTRY_CONFIG: list[dict[str, Any]] = [
    {
        "id": _SEASON_ID,
        "name": "test-season",
        "version": 1,
        "canonical": True,
        "is_default": True,
        "entry_pool": "qualifying",
        "leaderboard_pool": "ranked",
        "summary": "",
        "pools": [
            {"name": "qualifying", "description": "entry pool", "config_id": _ENTRY_CONFIG_ID},
            {"name": "ranked", "description": "ranked pool", "config_id": None},
        ],
    }
]

_SEASON_NO_ENTRY_CONFIG: list[dict[str, Any]] = [
    {
        "id": _SEASON_ID,
        "name": "test-season",
        "version": 1,
        "canonical": True,
        "is_default": True,
        "entry_pool": None,
        "leaderboard_pool": "ranked",
        "summary": "",
        "pools": [
            {"name": "ranked", "description": "ranked pool", "config_id": None},
        ],
    }
]


def _setup_mock_upload_server_with_season(
    httpserver: HTTPServer,
    seasons: list[dict[str, Any]],
    upload_id: str = "test-upload-id",
) -> None:
    season_summary = seasons[0]

    httpserver.expect_request(
        "/tournament/seasons",
        method="GET",
    ).respond_with_json(seasons)
    httpserver.expect_request(
        f"/tournament/seasons/{season_summary['name']}",
        method="GET",
    ).respond_with_json(_season_info_from_summary(season_summary))

    httpserver.expect_request(
        f"/tournament/configs/{_ENTRY_CONFIG_ID}",
        method="GET",
    ).respond_with_json(MettaGridConfig().model_dump(mode="json"))

    httpserver.expect_request(
        "/stats/policies/submit/presigned-url",
        method="POST",
    ).respond_with_json(
        {
            "upload_url": httpserver.url_for("/fake-s3-upload"),
            "upload_id": upload_id,
        }
    )

    httpserver.expect_request(
        "/fake-s3-upload",
        method="PUT",
    ).respond_with_data("")

    def handle_complete(request):
        body = request.json
        return Response(
            json.dumps(
                {
                    "id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                    "name": body["name"],
                    "version": 1,
                }
            ),
            content_type="application/json",
        )

    httpserver.expect_request(
        "/stats/policies/submit/complete",
        method="POST",
    ).respond_with_handler(handle_complete)


def test_upload_resolves_season_and_validates(
    httpserver: HTTPServer,
    fake_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_mock_upload_server_with_season(httpserver, _SEASON_WITH_ENTRY_CONFIG)

    policy_dir = tmp_path / "my_policy"
    policy_dir.mkdir()
    (policy_dir / "policy_spec.json").write_text(json.dumps({"class_path": "my_policies.Agent", "init_kwargs": {}}))

    captured: dict[str, Any] = {}

    def fake_subprocess_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["called"] = True
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("cogames.cli.submit.subprocess.run", fake_subprocess_run)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "upload",
            "--policy",
            policy_dir.as_uri(),
            "--name",
            "test-policy",
            "--server",
            httpserver.url_for(""),
            "--login-server",
            "http://fake-login-server",
        ],
    )

    assert result.exit_code == 0, f"Upload failed:\n{result.output}"
    assert captured.get("called") is True
    assert "validate-bundle" in captured["cmd"]

    # seasons list + season detail + presigned-url + s3 upload + complete = 5 requests
    assert len(httpserver.log) == 5


def test_upload_returns_nonzero_when_validation_fails(
    httpserver: HTTPServer,
    fake_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_mock_upload_server_with_season(httpserver, _SEASON_WITH_ENTRY_CONFIG)

    policy_dir = tmp_path / "my_policy"
    policy_dir.mkdir()
    (policy_dir / "policy_spec.json").write_text(json.dumps({"class_path": "my_policies.Agent", "init_kwargs": {}}))

    def fake_subprocess_run(cmd, **kwargs):
        print("invalid season not found")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="invalid season not found")

    monkeypatch.setattr("cogames.cli.submit.subprocess.run", fake_subprocess_run)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "upload",
            "--policy",
            policy_dir.as_uri(),
            "--name",
            "test-policy",
            "--server",
            httpserver.url_for(""),
            "--login-server",
            "http://fake-login-server",
        ],
    )

    assert result.exit_code == 1
    assert "Validation failed" in result.output
    assert "invalid season not found" in result.output

    # Only season lookups should happen before validation fails.
    assert len(httpserver.log) == 2


def test_upload_skips_validation_when_no_entry_config(
    httpserver: HTTPServer,
    fake_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_mock_upload_server_with_season(httpserver, _SEASON_NO_ENTRY_CONFIG)

    policy_dir = tmp_path / "my_policy"
    policy_dir.mkdir()
    (policy_dir / "policy_spec.json").write_text(json.dumps({"class_path": "my_policies.Agent", "init_kwargs": {}}))

    validation_called = False

    def fake_subprocess_run(cmd, **kwargs):
        nonlocal validation_called
        validation_called = True
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("cogames.cli.submit.subprocess.run", fake_subprocess_run)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "upload",
            "--policy",
            policy_dir.as_uri(),
            "--name",
            "test-policy",
            "--server",
            httpserver.url_for(""),
            "--login-server",
            "http://fake-login-server",
            "--skip-validation",
        ],
    )

    assert result.exit_code == 0, f"Upload failed:\n{result.output}"
    assert not validation_called, "Validation should be skipped"


def test_validate_policy_fetches_config_and_runs(
    httpserver: HTTPServer,
) -> None:
    default_cfg = MettaGridConfig()

    httpserver.expect_request(
        "/tournament/seasons/test-season",
        method="GET",
    ).respond_with_json(
        {
            "id": _SEASON_ID,
            "name": "test-season",
            "version": 1,
            "canonical": True,
            "is_default": True,
            "status": "in_progress",
            "display_name": "Test Season",
            "tournament_type": "freeplay",
            "entrant_count": 1,
            "active_entrant_count": 1,
            "match_count": 0,
            "stage_count": 1,
            "entry_pool": "qualifying",
            "leaderboard_pool": "ranked",
            "summary": "",
            "pools": [
                {"name": "qualifying", "description": "entry pool", "config_id": _ENTRY_CONFIG_ID},
                {"name": "ranked", "description": "ranked pool", "config_id": None},
            ],
        }
    )

    httpserver.expect_request(
        f"/tournament/configs/{_ENTRY_CONFIG_ID}",
        method="GET",
    ).respond_with_json(default_cfg.model_dump(mode="json"))

    captured_args: dict[str, Any] = {}

    def fake_ensure_docker_daemon_access() -> None:
        captured_args["preflight_called"] = True

    def fake_validate_bundle_docker(policy_uri, config_data, image):
        captured_args["called"] = True
        captured_args["policy_uri"] = policy_uri
        captured_args["config_data"] = config_data
        captured_args["image"] = image

    runner = CliRunner()
    with (
        patch("cogames.main.ensure_docker_daemon_access", fake_ensure_docker_daemon_access),
        patch("cogames.main.validate_bundle_docker", fake_validate_bundle_docker),
    ):
        result = runner.invoke(
            app,
            [
                "validate-bundle",
                "--policy",
                "class=cogames.policy.starter_agent.StarterPolicy",
                "--season",
                "test-season",
                "--server",
                httpserver.url_for(""),
            ],
        )

    assert result.exit_code == 0, f"Validation failed:\n{result.output}"

    # Config endpoint was hit
    config_requests = [req for req, _ in httpserver.log if "/tournament/configs/" in req.path]
    assert len(config_requests) == 1

    assert captured_args.get("preflight_called") is True
    assert captured_args.get("called") is True
    assert captured_args["policy_uri"] == "class=cogames.policy.starter_agent.StarterPolicy"
    assert captured_args["config_data"]["game"]["num_agents"] == default_cfg.game.num_agents


def test_validate_policy_exits_early_when_docker_unavailable(httpserver: HTTPServer) -> None:
    def fake_ensure_docker_daemon_access() -> None:
        raise typer.Exit(1)

    runner = CliRunner()
    with patch("cogames.main.ensure_docker_daemon_access", fake_ensure_docker_daemon_access):
        result = runner.invoke(
            app,
            [
                "validate-bundle",
                "--policy",
                "class=cogames.policy.starter_agent.StarterPolicy",
                "--season",
                "test-season",
                "--server",
                httpserver.url_for(""),
            ],
        )

    assert result.exit_code == 1
    assert len(httpserver.log) == 0


def test_ensure_docker_daemon_access_exits_cleanly_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cogames.cli.submit.shutil.which", lambda _name: "/usr/bin/docker")

    def _fake_run(*_args: Any, **_kwargs: Any):
        raise subprocess.TimeoutExpired(cmd=["docker", "info"], timeout=15)

    monkeypatch.setattr("cogames.cli.submit.subprocess.run", _fake_run)

    with pytest.raises(typer.Exit) as exc:
        ensure_docker_daemon_access()

    assert exc.value.exit_code == 1


# --- Policy Size Rejection Tests ---


def test_upload_rejects_oversized_at_completion(
    httpserver: HTTPServer,
    fake_home: Path,
    tmp_path: Path,
) -> None:
    """Server returns 413 at completion stage → clean error after upload."""
    season_summary = {
        "id": _SEASON_ID,
        "name": "test-season",
        "version": 1,
        "canonical": True,
        "is_default": True,
        "entry_pool": None,
        "leaderboard_pool": None,
        "summary": "",
        "pools": [],
    }

    httpserver.expect_request("/tournament/seasons", method="GET").respond_with_json([season_summary])
    httpserver.expect_request("/tournament/seasons/test-season", method="GET").respond_with_json(
        _season_info_from_summary(season_summary)
    )
    httpserver.expect_request("/stats/policies/submit/presigned-url", method="POST").respond_with_json(
        {"upload_url": httpserver.url_for("/fake-s3-upload"), "upload_id": "test-upload-id"}
    )
    httpserver.expect_request("/fake-s3-upload", method="PUT").respond_with_data("")
    httpserver.expect_request("/stats/policies/submit/complete", method="POST").respond_with_json(
        {"detail": "Policy too large: 600 MB (max 500 MB)"},
        status=413,
    )

    policy_dir = tmp_path / "my_policy"
    policy_dir.mkdir()
    (policy_dir / "policy_spec.json").write_text(json.dumps({"class_path": "my_policies.Agent", "init_kwargs": {}}))

    result = subprocess.run(
        [
            "cogames",
            "upload",
            "--policy",
            policy_dir.as_uri(),
            "--name",
            "test-policy",
            "--server",
            httpserver.url_for(""),
            "--login-server",
            "http://fake-login-server",
            "--skip-validation",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            "HOME": str(fake_home),
            "PATH": os.environ.get("PATH", ""),
        },
    )

    assert result.returncode != 0
    assert "Policy too large" in result.stdout or "Policy too large" in result.stderr
