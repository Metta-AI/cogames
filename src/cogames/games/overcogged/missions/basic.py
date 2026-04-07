"""Canonical Overcogged mission factories for the future CoGames layout."""

from __future__ import annotations

import importlib


def make_basic_mission():
    overcogged_game = importlib.import_module("cogames.games.overcogged.game.game")
    return overcogged_game.OvercookedGame.create(num_agents=4, max_steps=300)
