import pytest

from cogames.cogs_vs_clips.clips import ClipsConfig
from cogames.cogs_vs_clips.cog import CogTeam
from cogames.cogs_vs_clips.config import CvCConfig
from cogames.cogs_vs_clips.mission import CvCMission
from cogames.cogs_vs_clips.missions import (
    CogsGuardMachina1Mission,
    make_game,
)
from cogames.core import CoGameSite
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.config.query import ClosureQuery, MaterializedQuery
from mettagrid.simulator import Simulation
from mettagrid.test_support.map_builders import ObjectNameMapBuilder


def test_make_cogs_vs_clips_scenario():
    """Test that make_cogs_vs_clips_scenario creates a valid configuration."""
    config = make_game()
    assert isinstance(config, MettaGridConfig)


def test_cogsguard_enables_aoe_mask_observation() -> None:
    env = CogsGuardMachina1Mission.make_env()
    assert env.game.obs.aoe_mask is True
    env.game.id_map().feature_id("aoe_mask")


@pytest.mark.skip(reason="Requires territory-only AOEs and team tag setup; not wired up yet")
def test_cogsguard_emits_aoe_mask_tokens_runtime() -> None:
    env = CogsGuardMachina1Mission.make_env()
    id_map = env.game.id_map()
    aoe_mask_feature_id = id_map.feature_id("aoe_mask")

    sim = Simulation(env)
    sim.step()
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


def test_hub_global_obs_shows_own_team_only():
    """Each agent sees only their own team's hub resources in global obs, not the other team's."""
    alpha = CogTeam(name="alpha", short_name="a", num_agents=1, wealth=1, initial_hearts=0)
    beta = CogTeam(name="beta", short_name="b", num_agents=1, wealth=2, initial_hearts=0)

    mission = CvCMission(
        name="two_team_obs_test",
        description="Test per-team hub global obs",
        site=CoGameSite(
            name="test",
            description="minimal 2-team map",
            map_builder=ObjectNameMapBuilder.Config(
                map_data=[
                    ["wall", "wall", "wall", "wall", "wall"],
                    ["wall", "agent.red", "a:hub", "empty", "wall"],
                    ["wall", "agent.blue", "b:hub", "empty", "wall"],
                    ["wall", "empty", "empty", "empty", "wall"],
                    ["wall", "wall", "wall", "wall", "wall"],
                ]
            ),
        ),
        teams={"alpha": alpha, "beta": beta},
        clips=ClipsConfig(disabled=True),
        max_steps=100,
    )

    env = mission.make_env()
    sim = Simulation(env, seed=42)

    alpha_hub_inv = alpha._hub_inventory().initial
    beta_hub_inv = beta._hub_inventory().initial
    assert alpha_hub_inv != beta_hub_inv, "teams must have different hub inventories to distinguish"

    agent_alpha = sim.agent(0)
    agent_beta = sim.agent(1)

    alpha_obs = agent_alpha.global_observations
    beta_obs = agent_beta.global_observations

    for element in CvCConfig.ELEMENTS:
        key = f"team:{element}"
        assert key in alpha_obs, f"alpha agent missing global obs '{key}'"
        assert key in beta_obs, f"beta agent missing global obs '{key}'"

        assert alpha_obs[key] == alpha_hub_inv[element], (
            f"alpha agent should see alpha hub's {element}={alpha_hub_inv[element]}, got {alpha_obs[key]}"
        )
        assert beta_obs[key] == beta_hub_inv[element], (
            f"beta agent should see beta hub's {element}={beta_hub_inv[element]}, got {beta_obs[key]}"
        )
