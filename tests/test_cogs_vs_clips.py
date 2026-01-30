from cogames.cogs_vs_clips.missions import (
    make_game,
)
from mettagrid.config.mettagrid_config import MettaGridConfig


def test_make_cogs_vs_clips_scenario():
    """Test that make_cogs_vs_clips_scenario creates a valid configuration."""
    # Create the scenario
    config = make_game()

    # Verify it returns a MettaGridConfig
    assert isinstance(config, MettaGridConfig)
