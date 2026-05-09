from types import SimpleNamespace

import pytest

from cogames.coworld.runner.kubernetes_runner import _wait_for_results
from cogames.coworld.runner.runner import EpisodeArtifacts


class _FakeCoreV1:
    def __init__(self, phases: dict[str, str]):
        self._phases = phases

    def read_namespaced_pod(self, *, name: str, namespace: str):
        return SimpleNamespace(
            status=SimpleNamespace(
                phase=self._phases.get(name, "Running"),
                container_statuses=[],
            )
        )


def test_wait_for_results_does_not_wait_for_player_pods_to_succeed(tmp_path):
    artifacts = EpisodeArtifacts.create(tmp_path)
    artifacts.results_path.write_text("{}", encoding="utf-8")
    core_v1 = _FakeCoreV1({"player-0": "Running"})

    _wait_for_results(
        artifacts,
        core_v1,
        "default",
        "game-pod",
        timeout_seconds=0.01,
        player_pod_names=["player-0"],
    )


def test_wait_for_results_raises_when_player_pod_fails(tmp_path):
    artifacts = EpisodeArtifacts.create(tmp_path)
    core_v1 = _FakeCoreV1({"player-0": "Failed"})

    with pytest.raises(RuntimeError, match="Player pod player-0 failed"):
        _wait_for_results(
            artifacts,
            core_v1,
            "default",
            "game-pod",
            timeout_seconds=0.01,
            player_pod_names=["player-0"],
        )
