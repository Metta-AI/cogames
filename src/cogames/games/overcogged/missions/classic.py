"""Classic Overcogged mission factory."""

from __future__ import annotations

from cogames.games.overcogged.classic.game import ClassicOvercoggedGame


def make_classic_mission():
    return ClassicOvercoggedGame.create(num_agents=2, max_steps=400)
