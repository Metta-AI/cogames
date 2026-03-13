"""Tests for the scout role variant: HP and energy modifiers."""

from cogames.games.cogs_vs_clips.game.roles.scout import ScoutVariant


def test_scout_constants():
    v = ScoutVariant()
    assert v.name == "scout_role"
    assert v.hp_modifier == 400
    assert v.energy_modifier == 100
