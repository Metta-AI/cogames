"""Shared random seeding helpers for rollout/evaluation paths."""

import random

import numpy as np
import torch


def seed_rollout_rng(seed: int) -> None:
    """Seed process RNGs so stochastic policy sampling is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
