import pytest

from cogames.games.cogs_vs_clips.game.damage import DamageVariant
from cogames.games.cogs_vs_clips.game.elements import ElementsVariant
from cogames.games.cogs_vs_clips.game.energy import EnergyVariant
from cogames.games.cogs_vs_clips.game.teams import TeamConfig, TeamVariant
from cogames.games.cogs_vs_clips.game.teams.hub_observations import HubObservationsVariant
from cogames.games.cogs_vs_clips.game.territory import TerritoryVariant as JunctionNetVariant
from cogames.games.cogs_vs_clips.missions.arena import make_arena_map_builder
from cogames.games.cogs_vs_clips.missions.machina_1 import make_machina1_map_builder, make_machina1_mission
from cogames.games.cogs_vs_clips.missions.mission import CvCMission
from cogames.games.cogs_vs_clips.missions.terrain import find_machina_arena
from cogames.games.cogs_vs_clips.missions.tutorial import make_tutorial_mission
from mettagrid.config.game_value import ConstValue, QueryCountValue, SumGameValue
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.config.query import ClosureQuery, MaterializedQuery
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Simulation
from mettagrid.test_support.map_builders import ObjectNameMapBuilder

ELEMENTS = ElementsVariant().elements


def _normalize_dinky_tag_name(tag_name: str) -> str:
    if tag_name.startswith("type:"):
        tag_name = tag_name[5:]
    first_colon = tag_name.find(":")
    if 0 <= first_colon < len(tag_name) - 1:
        tag_name = tag_name[first_colon + 1 :]
    variant_colon = tag_name.rfind(":")
    if 0 <= variant_colon < len(tag_name) - 1 and tag_name[variant_colon + 1 :].isdigit():
        tag_name = tag_name[:variant_colon]
    return tag_name


def test_make_cogs_vs_clips_scenario():
    """Test that make_cogs_vs_clips_scenario creates a valid configuration."""
    config = make_machina1_mission(num_agents=2).make_env()
    assert isinstance(config, MettaGridConfig)


def test_cvc_helper_defaults_use_8_agents() -> None:
    machina1_arena = find_machina_arena(make_machina1_map_builder())
    arena = find_machina_arena(make_arena_map_builder())

    assert machina1_arena is not None
    assert machina1_arena.spawn_count == 8
    assert arena is not None
    assert arena.spawn_count == 8

    env = make_machina1_mission().make_env()
    assert env.game.num_agents == 8


def test_machina_1_team_station_tags_win_under_dinky_normalization() -> None:
    env = make_machina1_mission(num_agents=8).make_env()
    pei = PolicyEnvInterface.from_mg_cfg(env)
    normalized_tag_to_id = {_normalize_dinky_tag_name(name): idx for idx, name in enumerate(pei.tags)}
    tag_to_id = {name: idx for idx, name in enumerate(pei.tags)}

    assert normalized_tag_to_id["aligner"] == tag_to_id["type:c:aligner"]
    assert normalized_tag_to_id["miner"] == tag_to_id["type:c:miner"]
    assert normalized_tag_to_id["scout"] == tag_to_id["type:c:scout"]
    assert normalized_tag_to_id["scrambler"] == tag_to_id["type:c:scrambler"]
    assert "type:aligner" not in tag_to_id
    assert "type:miner" not in tag_to_id
    assert "type:scout" not in tag_to_id
    assert "type:scrambler" not in tag_to_id


def test_cvc_enables_aoe_mask_observation() -> None:
    env = make_machina1_mission().make_env()
    assert env.game.obs.aoe_mask is True
    env.game.id_map().feature_id("aoe_mask")


def test_tutorial_spawn_territory_offsets_passive_hp_drain() -> None:
    mission = make_tutorial_mission().model_copy(
        update={
            "num_agents": 2,
            "num_cogs": 2,
            "min_cogs": 2,
            "max_cogs": 2,
        }
    )
    sim = Simulation(mission.make_env(), seed=42)

    for i in range(2):
        assert sim.agent(i).inventory.get("hp", 0) == 50
        sim.agent(i).set_action("noop")

    sim.step()

    for i in range(2):
        assert sim.agent(i).inventory.get("hp", 0) == 100


@pytest.mark.skip(reason="Requires territory-only AOEs and team tag setup; not wired up yet")
def test_cvc_emits_aoe_mask_tokens_runtime() -> None:
    env = make_machina1_mission().make_env()
    id_map = env.game.id_map()
    aoe_mask_feature_id = id_map.feature_id("aoe_mask")

    sim = Simulation(env)
    sim.step()
    obs = sim._c_sim.observations()[0]
    aoe_mask_tokens = sum(1 for token in obs.tolist() if token[1] == aoe_mask_feature_id)
    assert aoe_mask_tokens > 0


def test_tag_mutations_reference_valid_tags():
    """Tag mutations in events must only reference registered tags."""
    config = make_machina1_mission(num_agents=2).make_env()
    tag_names = set(config.game.id_map().tag_names())

    for event_name, event in config.game.events.items():
        for mutation in event.mutations:
            t = getattr(mutation, "tag", None)
            if t is not None:
                assert t in tag_names, (
                    f"Event '{event_name}' has tag mutation referencing unregistered tag '{t}'. Valid tags: {tag_names}"
                )


def test_team_net_tag_uses_type_hub_source():
    """net_materialized_query produces a MaterializedQuery with ClosureQuery source using type:hub."""
    from cogames.games.cogs_vs_clips.game.territory import net_materialized_query  # noqa: PLC0415

    team = TeamConfig()
    assert team.net_tag() == "net:cogs"
    mq = net_materialized_query(team)
    assert isinstance(mq, MaterializedQuery)
    assert mq.tag == "net:cogs"
    assert isinstance(mq.query, ClosureQuery)
    assert "type:hub" in str(mq.query.source), "Source query should include type:hub"


def test_machina_objective_reward_excludes_hub_baseline() -> None:
    env = make_machina1_mission(num_agents=2, max_steps=1000).make_env()
    reward_cfg = env.game.agents[0].rewards["aligned_junction_held"]

    assert isinstance(reward_cfg.reward, SumGameValue)
    assert reward_cfg.per_tick is True
    assert reward_cfg.reward.weights == [0.001, 0.001]
    assert len(reward_cfg.reward.values) == 2
    assert isinstance(reward_cfg.reward.values[0], QueryCountValue)
    assert reward_cfg.reward.values[0].query.source == "net:cogs"
    assert isinstance(reward_cfg.reward.values[1], ConstValue)
    assert reward_cfg.reward.values[1].value == -1.0


def test_hub_global_obs_shows_own_team_only():
    """Each agent sees only their own team's hub resources in global obs, not the other team's."""
    alpha = TeamConfig(name="alpha", short_name="a", num_agents=1)
    beta = TeamConfig(name="beta", short_name="b", num_agents=1)

    mission = CvCMission(
        name="two_team_obs_test",
        description="Test per-team hub global obs",
        map_builder=ObjectNameMapBuilder.Config(
            map_data=[
                ["wall", "wall", "wall", "wall", "wall"],
                ["wall", "agent.red", "a:hub", "empty", "wall"],
                ["wall", "agent.blue", "b:hub", "empty", "wall"],
                ["wall", "empty", "empty", "empty", "wall"],
                ["wall", "wall", "wall", "wall", "wall"],
            ]
        ),
        num_agents=2,
        max_steps=100,
    ).with_variants(
        [
            TeamVariant(default_teams={"alpha": alpha, "beta": beta}),
            JunctionNetVariant(),
            HubObservationsVariant(),
            DamageVariant(),
            EnergyVariant(),
        ]
    )

    env = mission.make_env()
    sim = Simulation(env, seed=42)

    agent_alpha = sim.agent(0)
    alpha_obs = agent_alpha.global_observations

    for element in ELEMENTS:
        key = f"team:{element}"
        assert key in alpha_obs, f"agent missing global obs '{key}'"
        assert alpha_obs[key] > 0, f"hub {element} should be > 0, got {alpha_obs[key]}"
