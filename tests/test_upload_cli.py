"""Tests for cogames upload CLI command."""

import io
import json
import os
import subprocess
import uuid
import zipfile
from pathlib import Path

import pytest
from pytest_httpserver import HTTPServer
from werkzeug import Response

from cogames.auth import AuthConfigReaderWriter


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
    assert len(httpserver.log) == 3, f"Expected 3 requests, got {len(httpserver.log)}"

    # 1. Presigned URL request should have auth token
    presign_req, _ = httpserver.log[0]
    assert presign_req.headers.get("X-Auth-Token") == "test-token-12345"

    # 2. Upload request should contain a valid zip with policy_spec.json
    upload_req, _ = httpserver.log[1]
    with zipfile.ZipFile(io.BytesIO(upload_req.data)) as zf:
        assert "policy_spec.json" in zf.namelist()
        spec = json.loads(zf.read("policy_spec.json"))
        assert spec["class_path"] == "cogames.policy.starter_agent.StarterPolicy"

    # 3. Complete request should have correct upload_id and name
    complete_req, _ = httpserver.log[2]
    complete_body = complete_req.json
    assert complete_body["upload_id"] == upload_id
    assert complete_body["name"] == "my-test-policy"


def test_upload_command_fails_without_auth(
    httpserver: HTTPServer,
    tmp_path: Path,
) -> None:
    """Test that 'cogames upload' fails gracefully when not authenticated."""
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


def test_upload_directory_bundle(
    httpserver: HTTPServer,
    fake_home: Path,
    tmp_path: Path,
) -> None:
    """Test uploading a policy from a local directory."""
    _setup_mock_upload_server(httpserver)

    # Create a bundle directory with policy_spec.json
    bundle_dir = tmp_path / "my_bundle"
    bundle_dir.mkdir()
    policy_spec = {
        "class_path": "my_policies.CustomAgent",
        "data_path": "weights.pt",
        "init_kwargs": {"hidden_size": 256},
    }
    (bundle_dir / "policy_spec.json").write_text(json.dumps(policy_spec))
    (bundle_dir / "weights.pt").write_bytes(b"fake weights data")
    (bundle_dir / "config.yaml").write_text("learning_rate: 0.001\nbatch_size: 32\n")

    result = subprocess.run(
        [
            "cogames",
            "upload",
            "--policy",
            bundle_dir.as_uri(),
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
    assert len(httpserver.log) == 3
    upload_req, _ = httpserver.log[1]
    with zipfile.ZipFile(io.BytesIO(upload_req.data)) as zf:
        assert "policy_spec.json" in zf.namelist()
        assert "weights.pt" in zf.namelist()
        assert "config.yaml" in zf.namelist()
        spec = json.loads(zf.read("policy_spec.json"))
        assert spec["class_path"] == "my_policies.CustomAgent"
        assert spec["data_path"] == "weights.pt"
        assert zf.read("config.yaml").decode() == "learning_rate: 0.001\nbatch_size: 32\n"


def test_upload_zip_bundle(
    httpserver: HTTPServer,
    fake_home: Path,
    tmp_path: Path,
) -> None:
    """Test uploading a policy from a local zip file."""
    _setup_mock_upload_server(httpserver)

    # Create a bundle zip file with policy_spec.json and model weights
    bundle_zip = tmp_path / "my_bundle.zip"
    policy_spec = {
        "class_path": "my_policies.TrainedAgent",
        "data_path": "model.safetensors",
        "init_kwargs": {"num_layers": 4},
    }
    with zipfile.ZipFile(bundle_zip, "w") as zf:
        zf.writestr("policy_spec.json", json.dumps(policy_spec))
        zf.writestr("model.safetensors", b"fake model data")
        zf.writestr("hyperparams.json", json.dumps({"lr": 1e-4, "epochs": 100}))

    result = subprocess.run(
        [
            "cogames",
            "upload",
            "--policy",
            bundle_zip.as_uri(),
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
    assert len(httpserver.log) == 3
    upload_req, _ = httpserver.log[1]
    with zipfile.ZipFile(io.BytesIO(upload_req.data)) as zf:
        assert "policy_spec.json" in zf.namelist()
        assert "model.safetensors" in zf.namelist()
        assert "hyperparams.json" in zf.namelist()
        spec = json.loads(zf.read("policy_spec.json"))
        assert spec["class_path"] == "my_policies.TrainedAgent"
        assert spec["data_path"] == "model.safetensors"
        hyperparams = json.loads(zf.read("hyperparams.json"))
        assert hyperparams == {"lr": 1e-4, "epochs": 100}


def test_upload_s3_bundle(
    httpserver: HTTPServer,
    fake_home: Path,
    tmp_path: Path,
) -> None:
    """Test uploading a policy from an S3 URI."""
    _setup_mock_upload_server(httpserver)

    # Use a unique S3 key to avoid caching across test runs
    unique_key = f"policies/agent-{uuid.uuid4().hex[:8]}.zip"

    # Create a bundle zip in memory
    policy_spec = {
        "class_path": "my_policies.S3Agent",
        "data_path": "checkpoint.pt",
        "init_kwargs": {"from_s3": True},
    }
    bundle_bytes = io.BytesIO()
    with zipfile.ZipFile(bundle_bytes, "w") as zf:
        zf.writestr("policy_spec.json", json.dumps(policy_spec))
        zf.writestr("checkpoint.pt", b"fake checkpoint data")
        zf.writestr("training_config.yaml", "epochs: 50\nlr: 0.0003\n")
    bundle_bytes.seek(0)

    # Mock S3 GetObject endpoint (path-style: GET /bucket/key)
    httpserver.expect_request(
        f"/test-bucket/{unique_key}",
        method="GET",
    ).respond_with_data(bundle_bytes.read(), content_type="application/zip")

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

    # Verify: S3 download + 3 upload requests = 4 total
    assert len(httpserver.log) == 4, f"Expected 4 requests, got {len(httpserver.log)}"

    # First request should be the S3 GetObject
    s3_req, _ = httpserver.log[0]
    assert s3_req.path == f"/test-bucket/{unique_key}"

    # Verify the uploaded zip contains the files from S3
    upload_req, _ = httpserver.log[2]  # After S3 download, presigned URL, then upload
    with zipfile.ZipFile(io.BytesIO(upload_req.data)) as zf:
        assert "policy_spec.json" in zf.namelist()
        assert "checkpoint.pt" in zf.namelist()
        assert "training_config.yaml" in zf.namelist()
        spec = json.loads(zf.read("policy_spec.json"))
        assert spec["class_path"] == "my_policies.S3Agent"
        assert zf.read("training_config.yaml").decode() == "epochs: 50\nlr: 0.0003\n"
