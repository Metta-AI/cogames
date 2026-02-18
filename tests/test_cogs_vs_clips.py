from cogames.cogs_vs_clips.missions import (
    CogsGuardMachina1Mission,
    make_game,
)
from mettagrid.config.mettagrid_config import MettaGridConfig


def test_make_cogs_vs_clips_scenario():
    """Test that make_cogs_vs_clips_scenario creates a valid configuration."""
    # Create the scenario
    config = make_game()

    # Verify it returns a MettaGridConfig
    assert isinstance(config, MettaGridConfig)


def test_cogsguard_enables_aoe_mask_observation() -> None:
    env = CogsGuardMachina1Mission.make_env()
    assert env.game.obs.aoe_mask is True
    # Ensure feature exists when enabled.
    env.game.id_map().feature_id("aoe_mask")
    assert env.game.obs.territory_map is True
    env.game.id_map().feature_id("territory")


def test_alignment_mutations_reference_valid_collectives():
    """Alignment mutations in events must only reference registered collectives."""
    config = make_game()
    collective_names = set(config.game.collectives.keys())

    for event_name, event in config.game.events.items():
        for mutation in event.mutations:
            collective = getattr(mutation, "collective", None)
            if collective is not None:
                assert collective in collective_names, (
                    f"Event '{event_name}' has alignment mutation referencing "
                    f"unregistered collective '{collective}'. "
                    f"Valid collectives: {collective_names}"
                )
