import pytest

from cogames.cogs_vs_clips.clips import ClipsConfig
from cogames.cogs_vs_clips.cog import CogTeam
from cogames.cogs_vs_clips.config import CvCConfig
from cogames.cogs_vs_clips.mission import CvCMission
from cogames.cogs_vs_clips.missions import (
    CogsGuardMachina1Mission,
    make_game,
)
from cogames.cogs_vs_clips.scrambler_tutorial import ScramblerTutorialMission
from cogames.cogs_vs_clips.ships import count_clips_ships_in_map_config
from cogames.cogs_vs_clips.sites import COGSGUARD_MACHINA_1
from cogames.cogs_vs_clips.variants import MultiTeamVariant
from cogames.core import CoGameSite
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.config.query import ClosureQuery, MaterializedQuery
from mettagrid.map_builder.ascii import AsciiMapBuilder
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


def test_clips_event_targets_scale_with_default_clips_ship_count() -> None:
    mission = CogsGuardMachina1Mission.model_copy(deep=True)
    assert count_clips_ships_in_map_config(mission.site.map_builder) == 4

    env = mission.make_env()

    assert env.game.events["neutral_to_clips"].max_targets == 4
    assert env.game.events["cogs_to_neutral"].max_targets == 4


def test_clips_uses_ship_object_with_junction_territory_range() -> None:
    env = CogsGuardMachina1Mission.make_env()

    assert "clips:ship" in env.game.objects
    assert "clips:hub" not in env.game.objects

    ship = env.game.objects["clips:ship"]
    assert ship.name == "ship"
    assert ship.territory_controls
    assert ship.territory_controls[0].strength == CvCConfig.TERRITORY_CONTROL_RADIUS


def test_clips_alignment_range_uses_ship_and_junction_distance() -> None:
    env = CogsGuardMachina1Mission.make_env()

    net_clips_query = next(mq for mq in env.game.materialize_queries if mq.tag == "net:clips").model_dump(mode="python")
    assert net_clips_query["query"]["source"]["source"] == "type:ship"
    assert net_clips_query["query"]["edge_filters"][0]["radius"] == CvCConfig.JUNCTION_ALIGN_DISTANCE

    event_dump = env.game.events["neutral_to_clips"].model_dump(mode="python")
    inner_filters = event_dump["target_query"]["filters"][1]["inner"]
    max_distance_filters = [f for f in inner_filters if f["filter_type"] == "max_distance"]
    assert len(max_distance_filters) == 2
    assert {f["radius"] for f in max_distance_filters} == {CvCConfig.JUNCTION_ALIGN_DISTANCE}
    assert {f["query"]["source"] for f in max_distance_filters} == {"net:clips", "type:ship"}

    scramble_inner_filters = env.game.events["cogs_to_neutral"].model_dump(mode="python")["target_query"]["filters"]
    scramble_net_filter = next(
        f for f in scramble_inner_filters if f["filter_type"] == "max_distance" and f["query"]["source"] == "net:clips"
    )
    assert scramble_net_filter["radius"] == CvCConfig.JUNCTION_ALIGN_DISTANCE


def test_clips_event_targets_use_clips_ship_map_placements_for_ascii_builder() -> None:
    mission = CvCMission(
        name="clips_ship_map_config_scaling",
        description="Scale clips events by clips ship map placements",
        site=CoGameSite(
            name="test",
            description="map clips:ship placement drives events",
            map_builder=AsciiMapBuilder.Config(
                char_to_map_name={
                    "#": "wall",
                    ".": "empty",
                    "a": "agent.cogs",
                    "S": "clips:ship",
                    "j": "junction",
                },
                map_data=[
                    ["#", "#", "#", "#", "#"],
                    ["#", "a", "S", "j", "#"],
                    ["#", ".", "j", ".", "#"],
                    ["#", ".", "S", ".", "#"],
                    ["#", "#", "#", "#", "#"],
                ],
            ),
            min_cogs=1,
            max_cogs=1,
        ),
        teams={"cogs": CogTeam(name="cogs", short_name="c", num_agents=1, wealth=1)},
        max_steps=100,
    )

    env = mission.make_env()

    assert env.game.events["neutral_to_clips"].max_targets == 2
    assert env.game.events["cogs_to_neutral"].max_targets == 2


def test_clips_event_targets_scale_after_multi_team_map_rewrite() -> None:
    mission = CogsGuardMachina1Mission.with_variants([MultiTeamVariant(num_teams=2)])
    env = mission.make_env()

    assert env.game.events["neutral_to_clips"].max_targets == 8
    assert env.game.events["cogs_to_neutral"].max_targets == 8


def test_multiteam_variant_does_not_mutate_shared_site_constants() -> None:
    original_ship_count = count_clips_ships_in_map_config(COGSGUARD_MACHINA_1.map_builder)
    mission = CvCMission(
        name="basic",
        description="Constructor variant path should not mutate shared site state",
        site=COGSGUARD_MACHINA_1,
        num_cogs=8,
        max_steps=1000,
        variants=[MultiTeamVariant(num_teams=2)],
    )

    assert count_clips_ships_in_map_config(mission.site.map_builder) == original_ship_count * 2
    assert count_clips_ships_in_map_config(COGSGUARD_MACHINA_1.map_builder) == original_ship_count


def test_scrambler_tutorial_overrun_alignment_still_applies() -> None:
    env = ScramblerTutorialMission.make_env()
    assert env.game.events["neutral_to_clips"].max_targets is None
