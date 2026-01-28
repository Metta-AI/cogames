import math
from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, Sequence
from uuid import UUID

import numpy as np

from mettagrid.simulator.multi_episode.rollout import MultiEpisodeRolloutResult


def allocate_counts(total: int, weights: Sequence[float], *, allow_zero_total: bool = False) -> list[int]:
    total_weight = sum(weights)
    if total == 0 and allow_zero_total:
        return [0] * len(weights)
    if total < 0 or not weights or total_weight <= 0:
        raise ValueError("Counts require a non-negative total and positive weights.")

    fractions = [weight / total_weight for weight in weights]
    ideals = [total * fraction for fraction in fractions]
    counts = [math.floor(value) for value in ideals]
    remaining = total - sum(counts)

    remainders = [ideal - count for ideal, count in zip(ideals, counts, strict=True)]
    for idx in sorted(range(len(remainders)), key=remainders.__getitem__, reverse=True)[:remaining]:
        counts[idx] += 1
    return counts


def validate_proportions(proportions: Sequence[float] | None, num_policies: int) -> None:
    if proportions is None:
        return
    if len(proportions) != num_policies or sum(proportions) <= 0:
        raise ValueError("Proportions must match policy count and sum to a positive value.")


class ScoredMatchLike(Protocol):
    assignments: list[int]
    policy_version_ids: list[UUID]
    policy_scores: dict[UUID, float]
    policy_agent_counts: Mapping[UUID, int]


class Scorer(Protocol):
    def compute_scores(
        self,
        policy_version_ids: Sequence[UUID],
        matches: Sequence[ScoredMatchLike],
    ) -> dict[UUID, float]: ...


def compute_weighted_scores(
    policy_version_ids: Sequence[UUID],
    matches: Sequence[ScoredMatchLike],
) -> dict[UUID, float]:
    """Average per-policy scores across matches, weighted by agent share per match.

    This expects match.policy_scores to be the per-match policy scores (not already
    weighted across matches). Weighting by agent share keeps matches comparable
    when agent counts differ.
    """
    weighted_sums: dict[UUID, float] = {pv: 0.0 for pv in policy_version_ids}
    weight_totals: dict[UUID, float] = {pv: 0.0 for pv in policy_version_ids}

    for match in matches:
        policy_agent_counts = match.policy_agent_counts
        total_agents = sum(policy_agent_counts.values())
        if total_agents == 0:
            continue

        for pv, score in match.policy_scores.items():
            if pv not in weighted_sums:
                continue
            agent_count = policy_agent_counts.get(pv, 0)
            weight = agent_count / total_agents
            weighted_sums[pv] += score * weight
            weight_totals[pv] += weight

    return {pv: weighted_sums[pv] / weight_totals[pv] if weight_totals[pv] > 0 else 0.0 for pv in policy_version_ids}


class WeightedScorer:
    def compute_scores(
        self,
        policy_version_ids: Sequence[UUID],
        matches: Sequence[ScoredMatchLike],
    ) -> dict[UUID, float]:
        return compute_weighted_scores(policy_version_ids, matches)


def compute_average_scores_per_agent(
    total_scores: Mapping[UUID, float],
    agent_counts: Mapping[UUID, int],
) -> dict[UUID, float]:
    return {pv: total_score / max(agent_counts.get(pv, 0), 1) for pv, total_score in total_scores.items()}


def value_over_replacement(candidate_score: float, replacement_score: float) -> float:
    return candidate_score - replacement_score


def overall_value_over_replacement(
    weighted_sum: float,
    total_agents: int,
    replacement_score: float,
) -> Optional[float]:
    if total_agents <= 0:
        return None
    return weighted_sum / total_agents - replacement_score


@dataclass(frozen=True)
class VorScenarioSummary:
    """Summary stats for a single VOR scenario."""

    candidate_mean: Optional[float]
    replacement_mean: Optional[float]
    candidate_episode_count: int


@dataclass
class VorTotals:
    """Accumulates candidate-weighted totals across scenarios."""

    replacement_mean: Optional[float] = None
    total_candidate_weighted_sum: float = 0.0
    total_candidate_agents: int = 0

    def update(self, candidate_count: int, summary: VorScenarioSummary) -> None:
        if candidate_count == 0:
            self.replacement_mean = summary.replacement_mean
            return
        if summary.candidate_mean is None or summary.candidate_episode_count == 0:
            return
        self.total_candidate_weighted_sum += summary.candidate_mean * candidate_count * summary.candidate_episode_count
        self.total_candidate_agents += candidate_count * summary.candidate_episode_count


def summarize_vor_scenario(
    rollout: MultiEpisodeRolloutResult,
    *,
    candidate_policy_index: int,
    candidate_count: int,
) -> VorScenarioSummary:
    candidate_sum = 0.0
    candidate_episode_count = 0
    replacement_sum = 0.0
    replacement_episode_count = 0

    for episode in rollout.episodes:
        if episode.rewards.size == 0:
            continue
        if candidate_count == 0:
            replacement_sum += float(episode.rewards.mean())
            replacement_episode_count += 1
        else:
            mask = episode.assignments == candidate_policy_index
            if np.any(mask):
                candidate_sum += float(episode.rewards[mask].mean())
                candidate_episode_count += 1

    candidate_mean = candidate_sum / candidate_episode_count if candidate_episode_count else None
    replacement_mean = replacement_sum / replacement_episode_count if replacement_episode_count else None

    return VorScenarioSummary(
        candidate_mean=candidate_mean,
        replacement_mean=replacement_mean,
        candidate_episode_count=candidate_episode_count,
    )
