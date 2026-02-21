from cogames.cogs_vs_clips.cog import CogTeam
from cogames.cogs_vs_clips.missions import (
    CogsGuardMachina1Mission,
    make_game,
)
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.config.query import ClosureQuery, MaterializedQuery
from mettagrid.simulator import Simulation


def test_make_cogs_vs_clips_scenario():
    """Test that make_cogs_vs_clips_scenario creates a valid configuration."""
    config = make_game()
    assert isinstance(config, MettaGridConfig)


def test_cogsguard_enables_aoe_mask_observation() -> None:
    env = CogsGuardMachina1Mission.make_env()
    assert env.game.obs.width == 11
    assert env.game.obs.height == 11
    assert env.game.obs.aoe_mask is True
    env.game.id_map().feature_id("aoe_mask")


def test_cogsguard_emits_aoe_mask_tokens_runtime() -> None:
    env = CogsGuardMachina1Mission.make_env()
    id_map = env.game.id_map()
    aoe_mask_feature_id = id_map.feature_id("aoe_mask")

    sim = Simulation(env)
    obs = sim._c_sim.observations()[0]
    aoe_mask_tokens = sum(1 for token in obs.tolist() if token[1] == aoe_mask_feature_id)
    assert aoe_mask_tokens > 0


def test_tag_mutations_reference_valid_tags():
    """Tag mutations in events must only reference registered tags."""
    config = make_game()
    tag_names = set(config.game.id_map().tag_names())

    for event_name, event in config.game.events.items():
        for mutation in event.mutations:
            t = getattr(mutation, "tag", None)
            if t is not None:
                assert t in tag_names, (
                    f"Event '{event_name}' has tag mutation referencing unregistered tag '{t}'. Valid tags: {tag_names}"
                )


def test_team_net_tag_uses_type_hub_source():
    """CogTeam produces a MaterializedQuery for net:cogs with ClosureQuery source using type:hub."""
    team = CogTeam()
    assert team.net_tag() == "net:cogs"
    mqs = team.materialized_queries()
    assert len(mqs) == 1
    mq = mqs[0]
    assert isinstance(mq, MaterializedQuery)
    assert mq.tag == "net:cogs"
    assert isinstance(mq.query, ClosureQuery)
    assert "type:hub" in str(mq.query.source), "Source query should include type:hub"
