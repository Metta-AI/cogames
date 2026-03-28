"""Canonical Overcooked mission factories for the future CoGames layout."""

from __future__ import annotations

import importlib


def make_basic_mission():
    overcooked_game = importlib.import_module("metta.games.overcooked.game")
    return overcooked_game.OvercookedGame.create(num_agents=4, max_steps=300)
