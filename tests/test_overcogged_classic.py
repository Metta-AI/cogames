from typing import cast

from cogames.cli.mission import resolve_mission
from cogames.core import CoGameMission
from cogames.game import get_game
from cogames.games.overcogged.classic.game import (
    CARRY_RESOURCES,
    COOK_TIME,
    GEAR_RESOURCES,
    RESOURCE_NAMES,
    ClassicOvercoggedGame,
    _carbon_extractor,
    _chest,
    _game_events,
    _hub,
    _junction,
    _miner_station,
    _scrambler_station,
)
from cogames.games.overcogged.classic.variants import (
    BURN_TIME,
    EXTRA_CARRY,
    FAST_BURN_TIME,
    LONG_COOK_TIME,
    SHORT_COOK_TIME,
    BurnVariant,
    CrampedKitchenVariant,
    FastBurnVariant,
    LongCookVariant,
    RecipesVariant,
    ShortCookVariant,
    TipsVariant,
)
from mettagrid.config.action_config import ActionsConfig, MoveActionConfig, NoopActionConfig
from mettagrid.config.game_value import stat as game_stat
from mettagrid.config.mettagrid_config import (
    AgentConfig,
    GameConfig,
    InventoryConfig,
    MettaGridConfig,
    ResourceLimitsConfig,
    WallConfig,
)
from mettagrid.config.render_config import RenderConfig
from mettagrid.config.reward_config import reward
from mettagrid.map_builder.ascii import AsciiMapBuilder, AsciiMapBuilderConfig
from mettagrid.simulator.simulator import Simulation


def _game_cfg(
    num_agents: int = 1,
    max_steps: int = 200,
    map_data: list[list[str]] | None = None,
    char_to_map_name: dict[str, str] | None = None,
) -> MettaGridConfig:
    if map_data is None:
        map_data = [
            ["#", "#", "#", "#", "#"],
            ["#", ".", ".", ".", "#"],
            ["#", "@", ".", ".", "#"],
            ["#", ".", ".", ".", "#"],
            ["#", "#", "#", "#", "#"],
        ]
    if char_to_map_name is None:
        char_to_map_name = {"#": "wall", "@": "agent.agent", ".": "empty"}

    return MettaGridConfig(
        game=GameConfig(
            num_agents=num_agents,
            max_steps=max_steps,
            resource_names=list(RESOURCE_NAMES),
            actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
            agents=[
                AgentConfig(
                    inventory=InventoryConfig(
                        limits={
                            "carry": ResourceLimitsConfig(base=1, max=1, resources=list(CARRY_RESOURCES)),
                            "gear": ResourceLimitsConfig(base=1, max=1, resources=list(GEAR_RESOURCES)),
                            "item_count": ResourceLimitsConfig(base=1, max=1, resources=["num_items"]),
                        },
                        initial={"miner": 1},
                    ),
                    rewards={"deliveries": reward(game_stat("delivery", delta=True), weight=1.0)},
                )
                for _ in range(num_agents)
            ],
            objects={
                "wall": WallConfig(),
                "carbon_extractor": _carbon_extractor(),
                "hub": _hub(),
                "miner_station": _miner_station(),
                "scrambler_station": _scrambler_station(),
                "chest": _chest(),
                "junction": _junction(),
            },
            events=_game_events(),
            render=RenderConfig(object_status={"agent": {}}),
            map_builder=AsciiMapBuilder.Config(map_data=map_data, char_to_map_name=char_to_map_name),
        )
    )


def _sim(cfg: MettaGridConfig, seed: int = 42) -> Simulation:
    return Simulation(cfg, seed=seed)


def _classic_mission() -> CoGameMission:
    game = get_game("overcogged")
    return next(mission for mission in game.missions if mission.name == "classic")


def _step_noop(sim: Simulation, n: int = 1) -> None:
    for _ in range(n):
        for i in range(sim.num_agents):
            sim.agent(i).set_action("noop")
        sim.step()


def _find_object_by_type(sim: Simulation, type_name: str) -> dict | None:
    for obj in sim.grid_objects().values():
        if obj.get("type_name") == type_name:
            return obj
    return None


def _object_inventory(sim: Simulation, type_name: str) -> dict[str, int]:
    obj = _find_object_by_type(sim, type_name)
    assert obj is not None, f"No object of type {type_name} found"
    raw = obj.get("inventory", {})
    resource_names = sim.resource_names
    return {resource_names[k]: v for k, v in raw.items() if v > 0}


def test_overcogged_package_exposes_classic_mission() -> None:
    game = get_game("overcogged")
    assert [mission.name for mission in game.missions] == ["basic", "classic"]


def test_resolve_classic_mission_preserves_original_objects() -> None:
    game = get_game("overcogged")
    resolved_name, env, _ = resolve_mission(game, "classic")

    assert resolved_name == "classic"
    for resource in RESOURCE_NAMES:
        assert resource in env.game.resource_names
    for obj_name in ["carbon_extractor", "hub", "miner_station", "scrambler_station", "chest", "junction"]:
        assert obj_name in env.game.objects


def test_resolve_classic_full_uses_classic_variant_graph() -> None:
    game = get_game("overcogged")
    _, env, _ = resolve_mission(game, "classic", variants_arg=["full"])

    for resource in EXTRA_CARRY:
        assert resource in env.game.resource_names
    assert "decoder" in env.game.resource_names
    assert "burn_complete" in env.game.events


def test_resolve_basic_full_stays_kitchen_variant_graph() -> None:
    game = get_game("overcogged")
    _, env, _ = resolve_mission(game, "basic", variants_arg=["full"])

    assert "queue_soup" in env.game.resource_names
    assert "oxygen" not in env.game.resource_names
    assert "burn_complete" not in env.game.events


def test_classic_carbon_extractor_requires_miner() -> None:
    cfg = _game_cfg(
        map_data=[
            ["#", "#", "#", "#", "#"],
            ["#", ".", ".", ".", "#"],
            ["#", "@", "I", ".", "#"],
            ["#", ".", ".", ".", "#"],
            ["#", "#", "#", "#", "#"],
        ],
        char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty", "I": "carbon_extractor"},
    )
    sim = _sim(cfg)
    agent = sim.agent(0)

    agent.set_action("move_east")
    sim.step()
    assert agent.inventory.get("carbon", 0) == 1

    agent.set_inventory({"scrambler": 1})
    agent.set_action("move_east")
    sim.step()
    assert agent.inventory.get("carbon", 0) == 0
    sim.close()


def test_classic_full_pipeline() -> None:
    cfg = _game_cfg(
        max_steps=500,
        map_data=[
            ["#", "#", "#", "#", "#", "#", "#", "#"],
            ["#", ".", ".", ".", ".", ".", ".", "#"],
            ["#", "@", "P", ".", "M", "D", "S", "#"],
            ["#", ".", ".", ".", ".", ".", ".", "#"],
            ["#", "#", "#", "#", "#", "#", "#", "#"],
        ],
        char_to_map_name={
            "#": "wall",
            "@": "agent.agent",
            ".": "empty",
            "P": "hub",
            "M": "miner_station",
            "D": "scrambler_station",
            "S": "junction",
        },
    )
    sim = _sim(cfg)
    agent = sim.agent(0)

    for _ in range(3):
        agent.set_inventory({"miner": 1, "carbon": 1})
        agent.set_action("move_east")
        sim.step()

    _step_noop(sim, COOK_TIME + 1)
    assert _object_inventory(sim, "hub").get("heart", 0) == 1

    agent.set_inventory({"scrambler": 1})
    agent.set_action("move_east")
    sim.step()
    assert agent.inventory.get("heart", 0) == 1

    agent.set_action("move_south")
    sim.step()
    for _ in range(5):
        agent.set_action("move_east")
        sim.step()
    initial_reward = agent.episode_reward
    agent.set_action("move_north")
    sim.step()

    assert agent.inventory.get("heart", 0) == 0
    assert agent.episode_reward > initial_reward
    sim.close()


def test_classic_recipes_variant_creates_per_hub_objects() -> None:
    cfg = _game_cfg(
        map_data=[
            ["#", "#", "#", "#", "#"],
            ["#", ".", ".", ".", "#"],
            ["#", "@", "P", ".", "#"],
            ["#", ".", ".", ".", "#"],
            ["#", "#", "#", "#", "#"],
        ],
        char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty", "P": "hub"},
    )
    RecipesVariant().modify_env(_classic_mission(), cfg)

    assert "hub" not in cfg.game.objects
    assert "hub_0" in cfg.game.objects
    for resource in EXTRA_CARRY:
        assert resource in cfg.game.resource_names
        assert f"{resource}_extractor" in cfg.game.objects
    assert any(name.startswith("recipe_hub0_") for name in cfg.game.events)


def test_classic_burn_variant_burns_heart_after_threshold() -> None:
    cfg = _game_cfg(
        max_steps=500,
        map_data=[
            ["#", "#", "#", "#", "#"],
            ["#", ".", ".", ".", "#"],
            ["#", "@", "P", ".", "#"],
            ["#", ".", ".", ".", "#"],
            ["#", "#", "#", "#", "#"],
        ],
        char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty", "P": "hub"},
    )
    BurnVariant().modify_env(_classic_mission(), cfg)
    sim = _sim(cfg)
    agent = sim.agent(0)

    for _ in range(3):
        agent.set_inventory({"miner": 1, "carbon": 1})
        agent.set_action("move_east")
        sim.step()
    _step_noop(sim, COOK_TIME + 1)
    assert _object_inventory(sim, "hub").get("heart", 0) == 1

    _step_noop(sim, BURN_TIME + 1)
    hub_inv = _object_inventory(sim, "hub")
    assert hub_inv.get("heart", 0) == 0
    assert hub_inv.get("decoder", 0) == 1
    sim.close()


def test_classic_fast_burn_variant_reduces_threshold() -> None:
    cfg = _game_cfg(
        max_steps=500,
        map_data=[
            ["#", "#", "#", "#", "#"],
            ["#", ".", ".", ".", "#"],
            ["#", "@", "P", ".", "#"],
            ["#", ".", ".", ".", "#"],
            ["#", "#", "#", "#", "#"],
        ],
        char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty", "P": "hub"},
    )
    BurnVariant().modify_env(_classic_mission(), cfg)
    FastBurnVariant().modify_env(_classic_mission(), cfg)
    sim = _sim(cfg)
    agent = sim.agent(0)

    for _ in range(3):
        agent.set_inventory({"miner": 1, "carbon": 1})
        agent.set_action("move_east")
        sim.step()
    _step_noop(sim, COOK_TIME + 1)
    _step_noop(sim, FAST_BURN_TIME + 1)

    assert _object_inventory(sim, "hub").get("decoder", 0) == 1
    sim.close()


def test_classic_short_and_long_cook_variants_change_thresholds() -> None:
    short_cfg = _game_cfg(
        max_steps=500,
        map_data=[
            ["#", "#", "#", "#", "#"],
            ["#", ".", ".", ".", "#"],
            ["#", "@", "P", ".", "#"],
            ["#", ".", ".", ".", "#"],
            ["#", "#", "#", "#", "#"],
        ],
        char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty", "P": "hub"},
    )
    ShortCookVariant().modify_env(_classic_mission(), short_cfg)
    short_sim = _sim(short_cfg)
    short_agent = short_sim.agent(0)
    for _ in range(3):
        short_agent.set_inventory({"miner": 1, "carbon": 1})
        short_agent.set_action("move_east")
        short_sim.step()
    _step_noop(short_sim, SHORT_COOK_TIME + 1)
    assert _object_inventory(short_sim, "hub").get("heart", 0) == 1
    short_sim.close()

    long_cfg = _game_cfg(
        max_steps=500,
        map_data=[
            ["#", "#", "#", "#", "#"],
            ["#", ".", ".", ".", "#"],
            ["#", "@", "P", ".", "#"],
            ["#", ".", ".", ".", "#"],
            ["#", "#", "#", "#", "#"],
        ],
        char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty", "P": "hub"},
    )
    LongCookVariant().modify_env(_classic_mission(), long_cfg)
    long_sim = _sim(long_cfg)
    long_agent = long_sim.agent(0)
    for _ in range(3):
        long_agent.set_inventory({"miner": 1, "carbon": 1})
        long_agent.set_action("move_east")
        long_sim.step()
    _step_noop(long_sim, COOK_TIME + 1)
    assert _object_inventory(long_sim, "hub").get("heart", 0) == 0
    _step_noop(long_sim, LONG_COOK_TIME - COOK_TIME)
    assert _object_inventory(long_sim, "hub").get("heart", 0) == 1
    long_sim.close()


def test_classic_tips_variant_adds_reward() -> None:
    cfg = _game_cfg()
    TipsVariant().modify_env(_classic_mission(), cfg)
    for agent_cfg in cfg.game.agents:
        assert "tips" in agent_cfg.rewards


def test_classic_cramped_kitchen_variant_shrinks_map() -> None:
    env = ClassicOvercoggedGame.create(num_agents=2, max_steps=200).make_env()
    CrampedKitchenVariant().modify_env(_classic_mission(), env)
    map_data = cast(AsciiMapBuilderConfig, env.game.map_builder).map_data

    assert len(map_data) <= 10
    assert len(map_data[0]) <= 10
