"""Shared random seeding helpers for rollout/evaluation paths."""

import random

import numpy as np


def seed_rollout_rng(seed: int) -> None:
    """Seed process RNGs so stochastic policy sampling is reproducible."""
    import torch  # noqa: PLC0415  # deferred so importing this module doesn't pull in torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
