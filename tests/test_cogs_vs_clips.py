import pytest

from cogames.cogs_vs_clips.missions import (
    CogsGuardMachina1Mission,
    make_game,
)
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.simulator import Simulation


def test_make_cogs_vs_clips_scenario():
    """Test that make_cogs_vs_clips_scenario creates a valid configuration."""
    # Create the scenario
    config = make_game()

    # Verify it returns a MettaGridConfig
    assert isinstance(config, MettaGridConfig)


def test_cogsguard_enables_territory_observation() -> None:
    env = CogsGuardMachina1Mission.make_env()
    assert env.game.obs.width == 13
    assert env.game.obs.height == 13
    assert env.game.obs.territory is True
    # Ensure feature exists when enabled.
    env.game.id_map().feature_id("territory")


def test_cogsguard_emits_territory_tokens_runtime() -> None:
    env = CogsGuardMachina1Mission.make_env()
    id_map = env.game.id_map()
    territory_feature_id = id_map.feature_id("territory")
    with pytest.raises(KeyError):
        id_map.feature_id("aoe_mask")

    sim = Simulation(env)
    obs = sim._c_sim.observations()[0]
    territory_tokens = sum(1 for token in obs.tolist() if token[1] == territory_feature_id)
    assert territory_tokens > 0


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
