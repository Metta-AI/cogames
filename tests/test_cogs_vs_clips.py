import pytest
from cogsguard.game.damage import DamageVariant
from cogsguard.game.elements import ElementsVariant
from cogsguard.game.energy import EnergyVariant
from cogsguard.game.game import CvCGame
from cogsguard.game.teams import TeamConfig, TeamVariant
from cogsguard.game.teams.four_score import FourScoreVariant
from cogsguard.game.teams.hub_observations import HubObservationsVariant
from cogsguard.game.territory import TerritoryVariant as JunctionNetVariant
from cogsguard.missions.arena import make_arena_map_builder
from cogsguard.missions.four_score import FourScoreMission
from cogsguard.missions.machina_1 import make_machina1_map_builder, make_machina1_mission
from cogsguard.missions.mission import CvCMission
from cogsguard.missions.terrain import find_machina_arena
from cogsguard.missions.tutorial import make_tutorial_mission

from cogames.cli.mission import find_mission
from mettagrid.config.game_value import ConstValue, QueryCountValue, SumGameValue
from mettagrid.config.handler_config import AllOf, FirstMatch, Handler
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.config.query import ClosureQuery, MaterializedQuery
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Simulation
from mettagrid.test_support.map_builders import ObjectNameMapBuilder

ELEMENTS = ElementsVariant().elements


def _handler_names(handler) -> set[str]:
    """Collect all handler names from a handler tree."""
    if handler is None:
        return set()
    if isinstance(handler, Handler):
        return {handler.name} if handler.name else set()
    if isinstance(handler, (FirstMatch, AllOf)):
        names = set()
        for h in handler.handlers:
            names |= _handler_names(h)
        return names
    return set()


def test_make_cogs_vs_clips_scenario():
    """Test that make_cogs_vs_clips_scenario creates a valid configuration."""
    config = make_machina1_mission(num_agents=2).make_env()
    assert isinstance(config, MettaGridConfig)


def test_resolved_cli_missions_with_passive_hp_and_territory_install_friendly_hp_heal() -> None:
    game = CvCGame()
    audited: dict[str, list[str]] = {}
    mission_names = [mission.full_name() for mission in game.missions]
    mission_names.extend(f"{mission.name}.{sub_name}" for mission in game.missions for sub_name in mission.sub_missions)

    for name in mission_names:
        env = find_mission(game, name).make_env()
        territory = env.game.territories.get("team_territory")
        presence = sorted(territory.presence) if territory is not None else []
        audited[name] = presence

        has_passive_hp_drain = "hp" in env.game.resource_names and any(
            "hp_regen" in _handler_names(agent.on_tick) for agent in env.game.agents
        )
        if territory is not None and has_passive_hp_drain:
            assert "heal_hp" in territory.presence, f"{name} missing heal_hp; territory presence={presence}"

    assert {"tutorial", "tutorial.aligner", "tutorial.miner", "tutorial.scout", "tutorial.scrambler"} <= audited.keys()


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
    normalized_tag_to_id: dict[str, int] = {}
    for idx, name in enumerate(pei.tags):
        normalized = name[5:] if name.startswith("type:") else name
        first_colon = normalized.find(":")
        if 0 <= first_colon < len(normalized) - 1:
            normalized = normalized[first_colon + 1 :]
        variant_colon = normalized.rfind(":")
        if 0 <= variant_colon < len(normalized) - 1 and normalized[variant_colon + 1 :].isdigit():
            normalized = normalized[:variant_colon]
        normalized_tag_to_id[normalized] = idx
    tag_to_id = {name: idx for idx, name in enumerate(pei.tags)}

    assert normalized_tag_to_id["aligner"] == tag_to_id["type:aligner"]
    assert normalized_tag_to_id["miner"] == tag_to_id["type:miner"]
    assert normalized_tag_to_id["scout"] == tag_to_id["type:scout"]
    assert normalized_tag_to_id["scrambler"] == tag_to_id["type:scrambler"]
    assert "type:c:aligner" not in tag_to_id
    assert "type:c:miner" not in tag_to_id
    assert "type:c:scout" not in tag_to_id
    assert "type:c:scrambler" not in tag_to_id


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
    from cogsguard.game.territory import net_materialized_query  # noqa: PLC0415

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


def test_machina_1_emits_held_stat_per_tick_after_alignment() -> None:
    mission = CvCMission(
        name="held_metric_test",
        description="Minimal Machina setup for held stat checks.",
        map_builder=ObjectNameMapBuilder.Config(
            map_data=[
                ["wall", "wall", "wall", "wall", "wall"],
                ["wall", "empty", "empty", "empty", "wall"],
                ["wall", "agent.agent", "junction", "c:hub", "wall"],
                ["wall", "empty", "empty", "empty", "wall"],
                ["wall", "wall", "wall", "wall", "wall"],
            ]
        ),
        num_agents=1,
        num_cogs=1,
        min_cogs=1,
        max_cogs=1,
        max_steps=100,
    ).with_variants(["machina_1"])
    env = mission.make_env()
    env.game.agents[0].inventory.initial = {"aligner": 1, "heart": 1}

    sim = Simulation(env, seed=42)
    sim.agent(0).set_action("move_east")
    sim.step()

    assert sim.agent(0).last_action_success
    assert sim._c_sim.get_game_stat("cogs/aligned.junction.gained") == pytest.approx(1.0)
    assert sim._c_sim.get_game_stat("cogs/aligned.junction.held") == pytest.approx(1.0)
    assert float(sim.episode_rewards[0]) == pytest.approx(0.01)

    sim.agent(0).set_action("noop")
    sim.step()

    assert sim._c_sim.get_game_stat("cogs/aligned.junction.held") == pytest.approx(2.0)
    assert float(sim.episode_rewards[0]) == pytest.approx(0.02)


def test_machina_1_clips_held_stat_excludes_all_ship_roots() -> None:
    env = make_machina1_mission(num_agents=2, max_steps=100).make_env()
    sim = Simulation(env, seed=42)

    for agent_id in range(sim.num_agents):
        sim.agent(agent_id).set_action("noop")
    sim.step()

    assert sim._c_sim.get_game_stat("clips/aligned.junction.held") == pytest.approx(0.0)


def test_machina_1_emits_held_stat_handlers_for_clips_team() -> None:
    env = make_machina1_mission(num_agents=2, max_steps=100).make_env()

    assert {
        "aligned_junction_held_cogs",
        "aligned_junction_held_clips",
    } <= _handler_names(env.game.on_tick)


def test_four_score_emits_held_stat_handlers_for_each_team() -> None:
    env = FourScoreMission(
        num_agents=4,
        num_cogs=4,
        min_cogs=4,
        max_cogs=4,
        max_steps=100,
    ).make_env()

    assert {
        "aligned_junction_held_cogs_red",
        "aligned_junction_held_cogs_blue",
        "aligned_junction_held_cogs_green",
        "aligned_junction_held_cogs_yellow",
        "aligned_junction_held_four_score_avg",
    } <= _handler_names(env.game.on_tick)


def test_four_score_does_not_duplicate_team_held_stat_handlers() -> None:
    env = FourScoreMission(
        num_agents=4,
        num_cogs=4,
        min_cogs=4,
        max_cogs=4,
        max_steps=100,
    ).make_env()

    counts: dict[str, int] = {}
    stack = [env.game.on_tick]
    while stack:
        handler = stack.pop()
        if handler is None:
            continue
        if isinstance(handler, Handler):
            if handler.name:
                counts[handler.name] = counts.get(handler.name, 0) + 1
            continue
        if isinstance(handler, (FirstMatch, AllOf)):
            stack.extend(handler.handlers)

    assert counts["aligned_junction_held_cogs_red"] == 1
    assert counts["aligned_junction_held_cogs_blue"] == 1
    assert counts["aligned_junction_held_cogs_green"] == 1
    assert counts["aligned_junction_held_cogs_yellow"] == 1
    assert counts["aligned_junction_held_four_score_avg"] == 1


def test_four_score_without_default_variant_still_installs_machina_held_rewards() -> None:
    env = (
        CvCMission(
            name="four_score_defaultless",
            description="Four Score without a mission default variant.",
            map_builder=make_machina1_map_builder(4),
            num_agents=4,
            max_steps=100,
            default_variant=None,
        )
        .with_variants([FourScoreVariant()])
        .make_env()
    )

    assert {
        "aligned_junction_held_cogs_red",
        "aligned_junction_held_cogs_blue",
        "aligned_junction_held_cogs_green",
        "aligned_junction_held_cogs_yellow",
        "aligned_junction_held_four_score_avg",
    } <= _handler_names(env.game.on_tick)
    assert all("aligned_junction_held" in agent.rewards for agent in env.game.agents)


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
