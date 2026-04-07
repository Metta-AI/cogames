"""Classic Overcogged mission configuration.

This preserves the original cog-assembly game that used to live under
``metta.games.overcogged`` before the kitchen mission took over the canonical
Metta surface.
"""

from __future__ import annotations

from typing import cast

from pydantic import Field

from cogames.core import CoGameMission
from cogames.games.overcogged.classic.map import overcogged_map
from mettagrid.config.action_config import ActionsConfig, MoveActionConfig, NoopActionConfig
from mettagrid.config.event_config import EventConfig, periodic
from mettagrid.config.filter import (
    actorHas,
    actorHasAnyOf,
    isNot,
    query,
    targetHas,
    targetHasAnyOf,
)
from mettagrid.config.game_value import stat as game_stat
from mettagrid.config.handler_config import Handler
from mettagrid.config.mettagrid_config import (
    AgentConfig,
    GameConfig,
    GridObjectConfig,
    InventoryConfig,
    MettaGridConfig,
    ResourceLimitsConfig,
    WallConfig,
)
from mettagrid.config.mutation import (
    deposit,
    logActorAgentStat,
    updateActor,
    updateTarget,
    withdraw,
)
from mettagrid.config.obs_config import GlobalObsConfig, ObsConfig
from mettagrid.config.render_config import RenderAsset, RenderConfig, RenderHudConfig, RenderStatusBarConfig
from mettagrid.config.reward_config import reward
from mettagrid.config.tag import typeTag

COOK_TIME = 20
CARRY_RESOURCES = ["carbon", "heart"]
ALL_ELEMENTS = ["carbon"]
GEAR_RESOURCES = ["miner", "scrambler"]
INGREDIENTS_NEEDED = 3
RESOURCE_NAMES = ["carbon", "miner", "scrambler", "heart", "energy", "missing", "num_items"]


def _carbon_extractor() -> GridObjectConfig:
    return GridObjectConfig(
        name="carbon_extractor",
        on_use_handlers={
            "pick_up": Handler(
                filters=[
                    actorHas({"miner": 1}),
                    isNot(actorHasAnyOf(CARRY_RESOURCES)),
                ],
                mutations=[updateActor({"carbon": 1, "num_items": 1})],
            ),
        },
    )


def _hub() -> GridObjectConfig:
    return GridObjectConfig(
        name="hub",
        inventory=InventoryConfig(
            limits={
                "ingredients": ResourceLimitsConfig(
                    base=INGREDIENTS_NEEDED,
                    max=INGREDIENTS_NEEDED,
                    resources=["carbon"],
                ),
                "progress": ResourceLimitsConfig(
                    base=65535,
                    max=65535,
                    resources=["energy", "missing"],
                ),
                "output": ResourceLimitsConfig(base=1, max=1, resources=["heart"]),
            },
            initial={
                "carbon": INGREDIENTS_NEEDED,
                "missing": INGREDIENTS_NEEDED,
                "energy": COOK_TIME,
            },
        ),
        on_use_handlers={
            "pickup_heart": Handler(
                filters=[
                    actorHas({"scrambler": 1}),
                    isNot(actorHasAnyOf(CARRY_RESOURCES)),
                    targetHas({"heart": 1}),
                ],
                mutations=[
                    updateActor({"heart": 1, "num_items": 1}),
                    updateTarget({"heart": -1}),
                ],
            ),
            "deposit_carbon": Handler(
                filters=[
                    actorHas({"carbon": 1}),
                    targetHas({"carbon": 1}),
                    isNot(targetHas({"heart": 1})),
                ],
                mutations=[
                    updateActor({"carbon": -1, "num_items": -1}),
                    updateTarget({"carbon": -1, "missing": -1}),
                ],
            ),
        },
    )


def _miner_station() -> GridObjectConfig:
    return GridObjectConfig(
        name="miner_station",
        on_use_handlers={
            "swap_to_miner": Handler(
                filters=[isNot(actorHas({"miner": 1})), isNot(actorHas({"heart": 1}))],
                mutations=[updateActor({"scrambler": -1, "miner": 1})],
            ),
        },
    )


def _scrambler_station() -> GridObjectConfig:
    return GridObjectConfig(
        name="scrambler_station",
        on_use_handlers={
            "swap_to_scrambler": Handler(
                filters=[isNot(actorHas({"scrambler": 1}))],
                mutations=[updateActor({"miner": -1, "scrambler": 1})],
            ),
        },
    )


def _chest() -> GridObjectConfig:
    element_resources = [resource for resource in CARRY_RESOURCES if resource != "heart"]
    handlers: dict[str, Handler] = {}
    for resource in element_resources:
        handlers[f"deposit_{resource}"] = Handler(
            filters=[
                actorHas({"miner": 1}),
                actorHas({resource: 1}),
                isNot(targetHasAnyOf(element_resources)),
            ],
            mutations=[deposit({resource: 1}), updateActor({"num_items": -1})],
        )
    for resource in element_resources:
        handlers[f"withdraw_{resource}"] = Handler(
            filters=[
                actorHas({"miner": 1}),
                isNot(actorHasAnyOf(CARRY_RESOURCES)),
                targetHas({resource: 1}),
            ],
            mutations=[withdraw({resource: 1}), updateActor({"num_items": 1})],
        )

    return GridObjectConfig(
        name="chest",
        inventory=InventoryConfig(
            limits={
                "items": ResourceLimitsConfig(base=1, max=1, resources=element_resources),
            },
        ),
        on_use_handlers=handlers,
    )


def _junction() -> GridObjectConfig:
    return GridObjectConfig(
        name="junction",
        inventory=InventoryConfig(
            limits={
                "served": ResourceLimitsConfig(base=1, max=1, resources=["heart"]),
            },
        ),
        on_use_handlers={
            "serve_heart": Handler(
                filters=[actorHas({"heart": 1}), isNot(targetHas({"heart": 1}))],
                mutations=[
                    updateActor({"heart": -1, "num_items": -1}),
                    updateTarget({"heart": 1}),
                    logActorAgentStat("delivery"),
                ],
            ),
        },
    )


def _game_events() -> dict[str, EventConfig]:
    return {
        "cook_tick": EventConfig(
            name="cook_tick",
            target_query=query(typeTag("hub")),
            timesteps=periodic(start=0, period=1),
            filters=[
                isNot(targetHas({"missing": 1})),
                targetHas({"energy": 1}),
                isNot(targetHas({"heart": 1})),
            ],
            mutations=[updateTarget({"energy": -1})],
            max_targets=None,
        ),
        "cook_complete": EventConfig(
            name="cook_complete",
            target_query=query(typeTag("hub")),
            timesteps=periodic(start=0, period=1),
            filters=[
                isNot(targetHas({"missing": 1})),
                isNot(targetHas({"energy": 1})),
                isNot(targetHas({"heart": 1})),
            ],
            mutations=[
                updateTarget(
                    {
                        "carbon": INGREDIENTS_NEEDED,
                        "missing": INGREDIENTS_NEEDED,
                        "energy": COOK_TIME,
                        "heart": 1,
                    }
                ),
            ],
            max_targets=None,
        ),
        "junction_consume": EventConfig(
            name="junction_consume",
            target_query=query(typeTag("junction")),
            timesteps=periodic(start=0, period=1),
            filters=[targetHas({"heart": 1})],
            mutations=[updateTarget({"heart": -1})],
            max_targets=None,
        ),
    }


def _hub_render_assets() -> list[RenderAsset]:
    return [
        RenderAsset(asset="hub.green.4", resources={"heart": 1}),
        RenderAsset(asset="hub", resources={"carbon": INGREDIENTS_NEEDED}),
        RenderAsset(asset="hub.red.1", resources={"carbon": INGREDIENTS_NEEDED - 1}),
        RenderAsset(asset="hub.red.2", resources={"carbon": INGREDIENTS_NEEDED - 2}),
        RenderAsset(asset="hub.red.3.green.1", resources={"energy": COOK_TIME * 2 // 3}),
        RenderAsset(asset="hub.red.2.green.2", resources={"energy": COOK_TIME // 3}),
        RenderAsset(asset="hub.red.1.green.3", resources={"energy": 1}),
        RenderAsset(asset="hub.green.4"),
    ]


def _hub_render_assets_missing() -> list[RenderAsset]:
    return [
        RenderAsset(asset="hub.green.4", resources={"heart": 1}),
        RenderAsset(asset="hub", resources={"missing": INGREDIENTS_NEEDED}),
        RenderAsset(asset="hub.red.1", resources={"missing": INGREDIENTS_NEEDED - 1}),
        RenderAsset(asset="hub.red.2", resources={"missing": INGREDIENTS_NEEDED - 2}),
        RenderAsset(asset="hub.red.3.green.1", resources={"energy": COOK_TIME * 2 // 3}),
        RenderAsset(asset="hub.red.2.green.2", resources={"energy": COOK_TIME // 3}),
        RenderAsset(asset="hub.red.1.green.3", resources={"energy": 1}),
        RenderAsset(asset="hub.green.4"),
    ]


class ClassicOvercoggedGame(CoGameMission):
    max_steps: int = Field(default=400)

    @classmethod
    def create(cls, num_agents: int, max_steps: int) -> ClassicOvercoggedGame:
        return cls(
            name="classic",
            description="Original cooperative cog assembly mission.",
            map_builder=overcogged_map(num_agents),
            num_cogs=num_agents,
            min_cogs=1,
            max_cogs=num_agents,
            max_steps=max_steps,
        )

    @classmethod
    def variant_module_prefixes(cls) -> tuple[str, ...]:
        return ("cogames.games.overcogged.classic.",)

    def make_base_env(self) -> MettaGridConfig:
        num_cogs = cast(int, self.num_cogs)
        game = GameConfig(
            map_builder=self.map_builder,
            max_steps=self.max_steps,
            num_agents=num_cogs,
            resource_names=RESOURCE_NAMES,
            obs=ObsConfig(
                global_obs=GlobalObsConfig(
                    local_position=True,
                    last_action_move=True,
                ),
            ),
            actions=ActionsConfig(move=MoveActionConfig(), noop=NoopActionConfig()),
            agents=[
                AgentConfig(
                    inventory=InventoryConfig(
                        limits={
                            "carry": ResourceLimitsConfig(base=1, max=1, resources=CARRY_RESOURCES),
                            "gear": ResourceLimitsConfig(base=1, max=1, resources=GEAR_RESOURCES),
                            "item_count": ResourceLimitsConfig(base=1, max=1, resources=["num_items"]),
                        },
                        initial={"miner": 1},
                    ),
                    rewards={"deliveries": reward(game_stat("delivery", delta=True), weight=1.0)},
                )
                for _ in range(num_cogs)
            ],
            objects={
                "wall": WallConfig(name="wall"),
                "carbon_extractor": _carbon_extractor(),
                "hub": _hub(),
                "miner_station": _miner_station(),
                "scrambler_station": _scrambler_station(),
                "chest": _chest(),
                "junction": _junction(),
            },
            events=_game_events(),
            render=RenderConfig(
                hud1=RenderHudConfig(resource="num_items", short_name="I", max=1),
                hud2=RenderHudConfig(resource="num_items", short_name="I", max=1),
                assets={
                    "agent": [
                        RenderAsset(asset="aligner", resources={"heart": 1}),
                        RenderAsset(asset="scrambler", resources={"scrambler": 1}),
                        RenderAsset(asset="miner"),
                    ],
                    "hub": _hub_render_assets(),
                    "carbon_extractor": [RenderAsset(asset="carbon_extractor.working")],
                    "miner_station": [RenderAsset(asset="miner")],
                    "scrambler_station": [RenderAsset(asset="scrambler")],
                    "junction": [
                        RenderAsset(asset="junction.working", resources={"heart": 1}),
                        RenderAsset(asset="junction"),
                    ],
                    "chest": [
                        RenderAsset(asset="chest_carbon", resources={"carbon": 1}),
                        RenderAsset(asset="chest"),
                    ],
                },
                agent_huds={
                    "num_items": RenderHudConfig(resource="num_items", short_name="I", max=1, rank=0),
                },
                object_status={
                    "hub": {
                        "energy": RenderStatusBarConfig(
                            resource="energy",
                            short_name="E",
                            max=COOK_TIME,
                            divisions=COOK_TIME,
                            rank=0,
                        ),
                    },
                },
            ),
        )
        return MettaGridConfig(game=game)
