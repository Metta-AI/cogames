from typing import Optional

import numpy as np
import pytest

from metta_alo.rollout import write_replay
from metta_alo.scoring import (
    VorScenarioSummary,
    VorTotals,
    allocate_counts,
    summarize_vor_scenario,
)
from mettagrid.simulator.multi_episode.rollout import EpisodeRolloutResult, MultiEpisodeRolloutResult
from mettagrid.types import EpisodeStats

EMPTY_STATS: EpisodeStats = {"game": {}, "agent": []}


def _episode(assignments: list[int], rewards: list[float]) -> EpisodeRolloutResult:
    return EpisodeRolloutResult(
        assignments=np.array(assignments, dtype=int),
        rewards=np.array(rewards, dtype=float),
        action_timeouts=np.zeros(len(rewards), dtype=float),
        stats=EMPTY_STATS,
        replay_path=None,
        steps=3,
        max_steps=10,
    )


def test_allocate_counts_even_split() -> None:
    counts = allocate_counts(10, [1.0, 1.0])

    assert counts == [5, 5]


def test_allocate_counts_zero_total_allow() -> None:
    counts = allocate_counts(0, [1.0, 2.0], allow_zero_total=True)

    assert counts == [0, 0]


def test_allocate_counts_rejects_zero_weights() -> None:
    with pytest.raises(ValueError):
        allocate_counts(3, [0.0, 0.0])


def test_summarize_vor_scenario_candidate() -> None:
    rollout = MultiEpisodeRolloutResult(
        episodes=[
            _episode([0, 1], [1.0, 3.0]),
            _episode([1, 1], [4.0, 5.0]),
        ]
    )

    summary = summarize_vor_scenario(rollout, candidate_policy_index=0, candidate_count=1)

    assert summary.candidate_mean == pytest.approx(1.0)
    assert summary.candidate_episode_count == 1
    assert summary.replacement_mean is None


def test_summarize_vor_scenario_replacement() -> None:
    rollout = MultiEpisodeRolloutResult(
        episodes=[
            _episode([0, 1], [1.0, 3.0]),
            _episode([0, 1], [2.0, 2.0]),
        ]
    )

    summary = summarize_vor_scenario(rollout, candidate_policy_index=0, candidate_count=0)

    assert summary.candidate_mean is None
    assert summary.replacement_mean == pytest.approx(2.0)


def test_vor_totals_update() -> None:
    totals = VorTotals()
    summary = VorScenarioSummary(candidate_mean=1.5, replacement_mean=None, candidate_episode_count=2)

    totals.update(2, summary)
    totals.update(0, VorScenarioSummary(candidate_mean=None, replacement_mean=2.0, candidate_episode_count=0))

    assert totals.total_candidate_weighted_sum == pytest.approx(6.0)
    assert totals.total_candidate_agents == 4
    assert totals.replacement_mean == pytest.approx(2.0)


class DummyReplay:
    def __init__(self) -> None:
        self.compression: Optional[str] = None
        self.path: Optional[str] = None

    def set_compression(self, compression: str) -> None:
        self.compression = compression

    def write_replay(self, path: str) -> None:
        self.path = path


def test_write_replay_sets_gzip() -> None:
    replay = DummyReplay()

    write_replay(replay, "replay.json.gz")

    assert replay.compression == "gzip"
    assert replay.path == "replay.json.gz"


def test_write_replay_sets_zlib() -> None:
    replay = DummyReplay()

    write_replay(replay, "replay.json.z")

    assert replay.compression == "zlib"
    assert replay.path == "replay.json.z"


def test_write_replay_no_compression() -> None:
    replay = DummyReplay()

    write_replay(replay, "replay.json")

    assert replay.compression is None
    assert replay.path == "replay.json"
