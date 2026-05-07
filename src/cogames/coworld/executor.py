from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from cogames.coworld.episode_runner import EpisodeArtifacts, compress_replay, run_coworld_episode
from cogames.coworld.types import CoworldEpisodeJobSpec
from mettagrid.runner.types import RunnerError, RuntimeInfo
from mettagrid.util.file import copy_data, read, write_data

WORKDIR = Path(os.environ.get("COWORLD_WORKDIR", "/coworld"))


def run_from_env() -> None:
    spec = CoworldEpisodeJobSpec.model_validate_json(read(os.environ["JOB_SPEC_URI"]))
    artifacts = EpisodeArtifacts.create(WORKDIR, prefix="coworld-job-")
    _ensure_docker_available(artifacts)
    _write_runtime_info()
    try:
        run_coworld_episode(
            spec,
            artifacts,
            timeout_seconds=float(os.environ.get("COWORLD_TIMEOUT_SECONDS", "3600")),
        )
    except Exception as exc:
        _write_error_info(exc)
        raise
    _upload_outputs(artifacts)


def _write_runtime_info() -> None:
    runtime_info_uri = os.environ.get("RUNTIME_INFO_URI")
    if runtime_info_uri is None:
        return
    runtime_info = RuntimeInfo(
        git_commit=os.environ.get("GIT_COMMIT"),
        cogames_version=os.environ.get("COGAMES_VERSION"),
    )
    payload = runtime_info.model_dump_json(exclude_none=True).encode()
    write_data(runtime_info_uri, payload, content_type="application/json")


def _write_error_info(exc: Exception) -> None:
    error_info_uri = os.environ.get("ERROR_INFO_URI")
    if error_info_uri is None:
        return
    runner_error = RunnerError(error_type="crash", message=str(exc)[:2000])
    write_data(error_info_uri, runner_error.model_dump_json().encode(), content_type="application/json")


def _upload_outputs(artifacts: EpisodeArtifacts) -> None:
    results_uri = os.environ.get("RESULTS_URI")
    if results_uri is not None:
        copy_data(artifacts.results_path.as_uri(), results_uri, content_type="application/json")

    replay_uri = os.environ.get("REPLAY_URI")
    if replay_uri is not None and artifacts.replay_path.exists():
        copy_data(compress_replay(artifacts).as_uri(), replay_uri, content_type="application/x-compress")

    debug_uri = os.environ.get("DEBUG_URI")
    if debug_uri is not None:
        write_data(debug_uri, _zip_logs(artifacts.logs_dir), content_type="application/zip")

    policy_log_urls = os.environ.get("POLICY_LOG_URLS")
    if policy_log_urls is not None:
        for slot, log_uri in json.loads(policy_log_urls).items():
            log_path = artifacts.policy_log_path(int(slot))
            if log_path.exists():
                write_data(log_uri, log_path.read_bytes(), content_type="text/plain")


def _zip_logs(logs_dir: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in logs_dir.iterdir():
            if path.is_file():
                zf.write(path, path.name)
    return buf.getvalue()


def _ensure_docker_available(artifacts: EpisodeArtifacts) -> None:
    docker_info = subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if docker_info.returncode == 0:
        return

    dockerd_log_path = artifacts.logs_dir / "dockerd.log"
    dockerd_log = dockerd_log_path.open("ab")
    subprocess.Popen(
        ["dockerd", "--host=unix:///var/run/docker.sock"],
        stdout=dockerd_log,
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        docker_info = subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if docker_info.returncode == 0:
            return
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for Docker daemon. See {dockerd_log_path}")


if __name__ == "__main__":
    sys.exit(run_from_env())
