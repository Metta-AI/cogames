"""Shared random seeding helpers for rollout/evaluation paths."""

import random

import numpy as np

from cogames.optional_deps import has_neural


def seed_rollout_rng(seed: int) -> None:
    """Seed process RNGs so stochastic policy sampling is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    if has_neural():
        import torch  # noqa: PLC0415

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
