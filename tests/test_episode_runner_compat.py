# This test serializes a job spec from the CURRENT code and runs it on an OLD episode-runner
# Docker image. If this test fails, it means you broke backwards compatibility.
#
# The WRONG fix is to modify the job/env/config in this test to strip out or scrub fields
# that the old image doesn't understand. That defeats the entire purpose — this test is
# supposed to catch exactly that situation. If the old runner can't deserialize what the
# current code produces, the schemas are not backwards compatible.
#
# The RIGHT fix is one of:
#   1. Make your schema change backwards compatible (e.g. optional fields with defaults).
#   2. Bump COMPAT_VERSION and publish a new compat image that includes your changes.
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from cogames.cogs_vs_clips.missions import make_game
from mettagrid.runner.types import SingleEpisodeJob


def _repo_compat_version() -> str:
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / ".repo-root").exists():
            compat_file = parent / "COMPAT_VERSION"
            if compat_file.exists():
                return compat_file.read_text().strip()
            break
    raise RuntimeError("Could not locate COMPAT_VERSION at repo root")


def _compat_image() -> str:
    return f"ghcr.io/metta-ai/episode-runner:compat-v{_repo_compat_version()}"


IMAGE = os.environ.get("EPISODE_RUNNER_IMAGE", _compat_image())


def _docker_available() -> bool:
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=10)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.mark.skipif(not _docker_available(), reason="Docker not available")
def test_job_runs_on_compat_episode_runner():
    env_cfg = make_game()
    env_cfg.game.max_steps = 10

    job = SingleEpisodeJob(
        policy_uris=["mock://random"],
        assignments=[0] * env_cfg.game.num_agents,
        env=env_cfg,
        seed=42,
    )

    with tempfile.TemporaryDirectory() as workspace:
        spec_path = Path(workspace) / "spec.json"
        spec_path.write_text(job.model_dump_json())

        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--platform",
                "linux/amd64",
                "-e",
                "JOB_SPEC_URI=file:///workspace/spec.json",
                "-e",
                "RESULTS_URI=file:///workspace/results.json",
                "-v",
                f"{workspace}:/workspace:rw",
                IMAGE,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode == 0, f"Episode runner failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
        assert (Path(workspace) / "results.json").exists(), "No results file produced"
