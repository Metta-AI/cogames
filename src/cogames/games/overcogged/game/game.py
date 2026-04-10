"""Game configuration for the Overcogged event-driven kitchen game."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from pydantic import Field
from typing_extensions import Self

from cogames.core import CoGameMission, CoGameMissionVariant
from cogames.game import CoGame, register_game
from cogames.games.overcogged.defaults import FRIES_BURN_TICKS, FRIES_COOK_TICKS, SOUP_BURN_TICKS, SOUP_COOK_TICKS
from cogames.games.overcogged.game import load_variants
from cogames.games.overcogged.missions import make_basic_mission, make_classic_mission
from cogames.games.overcogged.variants import (
    HIDDEN_VARIANT_NAMES,
    normalize_variant_names,
    resolve_variant_selection,
)
from cogames.variants import VariantRegistry
from mettagrid.config.action_config import (
    ActionsConfig,
    ChangeVibeActionConfig,
    MoveActionConfig,
    NoopActionConfig,
)
from mettagrid.config.event_config import EventConfig, periodic
from mettagrid.config.filter import GameValueFilter, HandlerTarget, actorHasAnyOf, isNot, targetHasAnyOf
from mettagrid.config.game_value import QueryInventoryValue, SumGameValue, stat
from mettagrid.config.handler_config import (
    Handler,
    actorHas,
    deposit,
    queryDelta,
    targetHas,
    updateActor,
    updateTarget,
    withdraw,
)
from mettagrid.config.mettagrid_config import (
    AgentConfig,
    GameConfig,
    GridObjectConfig,
    InventoryConfig,
    MettaGridConfig,
    ResourceLimitsConfig,
    TalkConfig,
    WallConfig,
)
from mettagrid.config.mutation.stats_mutation import logActorAgentStat, logStatToGame
from mettagrid.config.obs_config import GlobalObsConfig, ObsConfig
from mettagrid.config.query import Query, query
from mettagrid.config.render_config import RenderAsset, RenderConfig, RenderHudConfig, RenderStatusBarConfig
from mettagrid.config.reward_config import AgentReward, reward
from mettagrid.config.tag import typeTag
from mettagrid.mapgen.mapgen import MapGenConfig
from mettagrid.mapgen.scenes.compound import CompoundConfig

RecipeName = Literal["salad", "soup", "fries"]

VEG = "veg"
MEAT = "meat"
CHOPPED_VEG = "chopped_veg"
CHOPPED_MEAT = "chopped_meat"
CHOP_VEG_PROGRESS = "chop_veg_progress"
CHOP_MEAT_PROGRESS = "chop_meat_progress"

CLEAN_PLATE = "clean_plate"
DIRTY_PLATE = "dirty_plate"
WASH_PROGRESS = "wash_progress"

DISH_SALAD = "dish_salad"
DISH_SOUP = "dish_soup"
DISH_FRIES = "dish_fries"

QUEUE_SALAD = "queue_salad"
QUEUE_SOUP = "queue_soup"
QUEUE_FRIES = "queue_fries"

POT_SOUP_COOKING = "pot_soup_cooking"
POT_SOUP_READY = "pot_soup_ready"
POT_SOUP_BURNED = "pot_soup_burned"
POT_TIMER = "pot_timer"
POT_READY_AGE = "pot_ready_age"

FRYER_FRIES_COOKING = "fryer_fries_cooking"
FRYER_FRIES_READY = "fryer_fries_ready"
FRYER_FRIES_BURNED = "fryer_fries_burned"
FRYER_TIMER = "fryer_timer"
FRYER_READY_AGE = "fryer_ready_age"

BASE_AGENT_RESOURCES = [
    VEG,
    MEAT,
    CHOPPED_VEG,
    CHOPPED_MEAT,
    CLEAN_PLATE,
    DIRTY_PLATE,
    DISH_SALAD,
    DISH_SOUP,
    DISH_FRIES,
]
PREP_PROGRESS_RESOURCES = [CHOP_VEG_PROGRESS, CHOP_MEAT_PROGRESS, WASH_PROGRESS]
QUEUE_COUNTER_RESOURCES = [QUEUE_SALAD, QUEUE_SOUP, QUEUE_FRIES]
POT_RESOURCES = [POT_SOUP_COOKING, POT_SOUP_READY, POT_SOUP_BURNED, POT_TIMER, POT_READY_AGE]
FRYER_RESOURCES = [FRYER_FRIES_COOKING, FRYER_FRIES_READY, FRYER_FRIES_BURNED, FRYER_TIMER, FRYER_READY_AGE]

ORDER_QUEUE_MAX = 8
CHOP_TICKS = 3
WASH_TICKS = 3

TICKET_FIRST_ARRIVAL = 0
TICKET_INTERARRIVAL = 18
TICKET_DEADLINE = 50

ORDER_BOARD_QUERY: Query = query(typeTag("order_board"))
COOKING_STATION_QUERY: Query = query(typeTag("cooking_station"))
FRYER_STATION_QUERY: Query = query(typeTag("fryer_station"))

STATIONS = [
    "veg_station",
    "meat_station",
    "plate_station",
    "chopping_station",
    "cooking_station",
    "fryer_station",
    "serving_station",
    "wash_station",
    "order_board",
]
DEFAULT_KITCHEN_STATION_OFFSETS: list[tuple[int, int]] = [
    (-4, -2),
    (-2, -2),
    (4, 0),
    (0, -2),
    (2, -2),
    (4, -2),
    (2, 2),
    (4, 2),
    (0, 2),
]
RECIPE_CYCLE: tuple[RecipeName, ...] = ("soup", "salad", "soup", "fries", "salad")
QUEUE_RESOURCE_BY_RECIPE: dict[RecipeName, str] = {
    "salad": QUEUE_SALAD,
    "soup": QUEUE_SOUP,
    "fries": QUEUE_FRIES,
}
DISH_RESOURCE_BY_RECIPE: dict[RecipeName, str] = {
    "salad": DISH_SALAD,
    "soup": DISH_SOUP,
    "fries": DISH_FRIES,
}


@dataclass(frozen=True, slots=True)
class TicketSpec:
    index: int
    recipe: RecipeName
    arrival: int
    expiry: int
    resource: str

    @property
    def queue_resource(self) -> str:
        return queue_resource_for_recipe(self.recipe)


@dataclass(slots=True)
class OvercookedSettings:
    max_steps: int
    ticket_first_arrival: int
    ticket_interarrival: int
    ticket_deadline: int
    chop_ticks: int
    wash_ticks: int
    soup_cook_ticks: int
    soup_burn_ticks: int
    fries_cook_ticks: int
    fries_burn_ticks: int
    order_queue_max: int
    hub_layout: Literal["default", "tight", "cramped_room", "service_pass_room"]
    station_order: list[str]
    station_offsets: list[tuple[int, int]]
    hub_width: int
    hub_height: int
    randomize_spawn_positions: bool
    enable_queue_orders: bool
    enable_salad_recipe: bool
    enable_soup_recipe: bool
    enable_fries_recipe: bool
    enable_wash_cycle: bool
    enable_soup_burn: bool
    enable_fries_burn: bool

    @classmethod
    def from_mission(cls, mission: OvercookedGame) -> OvercookedSettings:
        return cls(
            max_steps=mission.max_steps,
            ticket_first_arrival=mission.ticket_first_arrival,
            ticket_interarrival=mission.ticket_interarrival,
            ticket_deadline=mission.ticket_deadline,
            chop_ticks=mission.chop_ticks,
            wash_ticks=mission.wash_ticks,
            soup_cook_ticks=mission.soup_cook_ticks,
            soup_burn_ticks=mission.soup_burn_ticks,
            fries_cook_ticks=mission.fries_cook_ticks,
            fries_burn_ticks=mission.fries_burn_ticks,
            order_queue_max=mission.order_queue_max,
            hub_layout=mission.hub_layout,
            station_order=list(mission.station_order),
            station_offsets=list(mission.station_offsets),
            hub_width=mission.hub_width,
            hub_height=mission.hub_height,
            randomize_spawn_positions=mission.randomize_spawn_positions,
            enable_queue_orders=mission.enable_queue_orders,
            enable_salad_recipe=mission.enable_salad_recipe,
            enable_soup_recipe=mission.enable_soup_recipe,
            enable_fries_recipe=mission.enable_fries_recipe,
            enable_wash_cycle=mission.enable_wash_cycle,
            enable_soup_burn=mission.enable_soup_burn,
            enable_fries_burn=mission.enable_fries_burn,
        )


class SupportsModifyMission(Protocol):
    def modify_mission(self, mission: OvercookedSettings) -> None: ...


def queue_resource_for_recipe(recipe: RecipeName) -> str:
    return QUEUE_RESOURCE_BY_RECIPE[recipe]


def dish_resource_for_recipe(recipe: RecipeName) -> str:
    return DISH_RESOURCE_BY_RECIPE[recipe]


def validate_station_order(station_order: list[str]) -> list[str]:
    expected = set(STATIONS)
    order_set = set(station_order)
    if order_set != expected or len(station_order) != len(STATIONS):
        missing = sorted(expected - order_set)
        extra = sorted(order_set - expected)
        raise ValueError(
            f"station_order must include each station exactly once. missing={missing} extra={extra} got={station_order}"
        )
    return list(station_order)


def validate_station_offsets(station_offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(station_offsets) != len(STATIONS):
        raise ValueError(
            f"station_offsets must include {len(STATIONS)} entries matching stations. "
            f"got={len(station_offsets)} offsets={station_offsets}"
        )
    normalized = [(int(dx), int(dy)) for dx, dy in station_offsets]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"station_offsets must be unique. got={station_offsets}")
    return normalized


def _ticket_resource_name(index: int, recipe: RecipeName) -> str:
    return f"ticket_{index:03d}_{recipe}"


def build_ticket_specs(
    max_steps: int,
    *,
    first_arrival: int = TICKET_FIRST_ARRIVAL,
    interarrival: int = TICKET_INTERARRIVAL,
    deadline: int = TICKET_DEADLINE,
) -> list[TicketSpec]:
    if interarrival <= 0:
        raise ValueError("interarrival must be positive")
    if deadline <= 0:
        raise ValueError("deadline must be positive")

    specs: list[TicketSpec] = []
    arrival = first_arrival
    idx = 0
    while arrival < max_steps:
        recipe = RECIPE_CYCLE[idx % len(RECIPE_CYCLE)]
        specs.append(
            TicketSpec(
                index=idx,
                recipe=recipe,
                arrival=arrival,
                expiry=min(max_steps, arrival + deadline),
                resource=_ticket_resource_name(idx, recipe),
            )
        )
        idx += 1
        arrival += interarrival
    return specs


def resource_names_for_tickets(ticket_specs: list[TicketSpec]) -> list[str]:
    return [
        *BASE_AGENT_RESOURCES,
        *PREP_PROGRESS_RESOURCES,
        *QUEUE_COUNTER_RESOURCES,
        *POT_RESOURCES,
        *FRYER_RESOURCES,
        *[ticket.resource for ticket in ticket_specs],
    ]


def overcooked_render_asset(
    asset_name: str,
    *,
    resources: dict[str, int] | None = None,
) -> RenderAsset:
    return RenderAsset(
        asset=asset_name,
        resources={} if resources is None else dict(resources),
    )


def overcooked_render_assets() -> dict[str, list[RenderAsset]]:
    return {
        "agent": [RenderAsset(asset="scrambler")],
        "veg_station": [overcooked_render_asset("overcooked_veg_station")],
        "meat_station": [overcooked_render_asset("overcooked_meat_station")],
        "plate_station": [overcooked_render_asset("overcooked_plate_station")],
        "chopping_station": [overcooked_render_asset("overcooked_chopping_station")],
        "cooking_station": [
            overcooked_render_asset(
                "overcooked_cooking_burned",
                resources={POT_SOUP_BURNED: 1},
            ),
            overcooked_render_asset(
                "overcooked_cooking_ready",
                resources={POT_SOUP_READY: 1},
            ),
            overcooked_render_asset(
                "overcooked_cooking_station",
                resources={POT_SOUP_COOKING: 1},
            ),
            overcooked_render_asset("overcooked_cooking_station"),
        ],
        "fryer_station": [
            overcooked_render_asset(
                "overcooked_fryer_burned",
                resources={FRYER_FRIES_BURNED: 1},
            ),
            overcooked_render_asset(
                "overcooked_fryer_ready",
                resources={FRYER_FRIES_READY: 1},
            ),
            overcooked_render_asset(
                "overcooked_fryer_station",
                resources={FRYER_FRIES_COOKING: 1},
            ),
            overcooked_render_asset("overcooked_fryer_station"),
        ],
        "serving_station": [overcooked_render_asset("overcooked_serving_station")],
        "wash_station": [overcooked_render_asset("overcooked_wash_station")],
        "order_board": [overcooked_render_asset("overcooked_order_board")],
    }


def overcooked_render_config(
    settings: OvercookedSettings,
    ticket_specs: list[TicketSpec],
) -> RenderConfig:
    return RenderConfig(
        agent_huds={
            DISH_SALAD: RenderHudConfig(resource=DISH_SALAD, short_name="SD", max=1, rank=0),
            DISH_SOUP: RenderHudConfig(resource=DISH_SOUP, short_name="SP", max=1, rank=1),
            DISH_FRIES: RenderHudConfig(resource=DISH_FRIES, short_name="FR", max=1, rank=2),
            CLEAN_PLATE: RenderHudConfig(resource=CLEAN_PLATE, short_name="PL", max=1, rank=3),
            DIRTY_PLATE: RenderHudConfig(resource=DIRTY_PLATE, short_name="DP", max=1, rank=4),
        },
        object_status={
            "agent": {
                CHOPPED_VEG: RenderStatusBarConfig(resource=CHOPPED_VEG, short_name="CV", max=1, rank=0),
                CHOPPED_MEAT: RenderStatusBarConfig(resource=CHOPPED_MEAT, short_name="CM", max=1, rank=1),
                DISH_SALAD: RenderStatusBarConfig(resource=DISH_SALAD, short_name="SD", max=1, rank=2),
                DISH_SOUP: RenderStatusBarConfig(resource=DISH_SOUP, short_name="SP", max=1, rank=3),
                DISH_FRIES: RenderStatusBarConfig(resource=DISH_FRIES, short_name="FR", max=1, rank=4),
            },
            "chopping_station": {
                CHOP_VEG_PROGRESS: RenderStatusBarConfig(
                    resource=CHOP_VEG_PROGRESS,
                    short_name="VG",
                    max=settings.chop_ticks,
                    rank=0,
                ),
                CHOP_MEAT_PROGRESS: RenderStatusBarConfig(
                    resource=CHOP_MEAT_PROGRESS,
                    short_name="MT",
                    max=settings.chop_ticks,
                    rank=1,
                ),
            },
            "order_board": {
                QUEUE_SALAD: RenderStatusBarConfig(
                    resource=QUEUE_SALAD,
                    short_name="QSD",
                    max=settings.order_queue_max,
                    rank=0,
                ),
                QUEUE_SOUP: RenderStatusBarConfig(
                    resource=QUEUE_SOUP,
                    short_name="QSP",
                    max=settings.order_queue_max,
                    rank=1,
                ),
                QUEUE_FRIES: RenderStatusBarConfig(
                    resource=QUEUE_FRIES,
                    short_name="QFR",
                    max=settings.order_queue_max,
                    rank=2,
                ),
            },
            "cooking_station": {
                POT_SOUP_COOKING: RenderStatusBarConfig(
                    resource=POT_SOUP_COOKING,
                    short_name="CK",
                    max=1,
                    rank=0,
                ),
                POT_SOUP_READY: RenderStatusBarConfig(
                    resource=POT_SOUP_READY,
                    short_name="RD",
                    max=1,
                    rank=1,
                ),
                POT_SOUP_BURNED: RenderStatusBarConfig(
                    resource=POT_SOUP_BURNED,
                    short_name="BR",
                    max=1,
                    rank=2,
                ),
            },
            "fryer_station": {
                FRYER_FRIES_COOKING: RenderStatusBarConfig(
                    resource=FRYER_FRIES_COOKING,
                    short_name="FC",
                    max=1,
                    rank=0,
                ),
                FRYER_FRIES_READY: RenderStatusBarConfig(
                    resource=FRYER_FRIES_READY,
                    short_name="FR",
                    max=1,
                    rank=1,
                ),
                FRYER_FRIES_BURNED: RenderStatusBarConfig(
                    resource=FRYER_FRIES_BURNED,
                    short_name="FB",
                    max=1,
                    rank=2,
                ),
            },
            "wash_station": {
                WASH_PROGRESS: RenderStatusBarConfig(
                    resource=WASH_PROGRESS,
                    short_name="WS",
                    max=settings.wash_ticks,
                    rank=0,
                ),
            },
        },
        assets=overcooked_render_assets(),
    )


def veg_station_config(*, enable_veg_pickup: bool = True) -> GridObjectConfig:
    handlers: dict[str, Handler] = {}
    if enable_veg_pickup:
        handlers["pickup_veg"] = Handler(
            filters=[isNot(actorHasAnyOf(BASE_AGENT_RESOURCES))],
            mutations=[updateActor({VEG: 1})],
        )
    return GridObjectConfig(
        name="veg_station",
        on_use_handlers=handlers,
    )


def meat_station_config(*, enable_meat_pickup: bool = True) -> GridObjectConfig:
    handlers: dict[str, Handler] = {}
    if enable_meat_pickup:
        handlers["pickup_meat"] = Handler(
            filters=[isNot(actorHasAnyOf(BASE_AGENT_RESOURCES))],
            mutations=[updateActor({MEAT: 1})],
        )
    return GridObjectConfig(
        name="meat_station",
        on_use_handlers=handlers,
    )


def plate_station_config(*, enable_plate_pickup: bool = True) -> GridObjectConfig:
    handlers: dict[str, Handler] = {}
    if enable_plate_pickup:
        handlers["pickup_clean_plate"] = Handler(
            filters=[isNot(actorHasAnyOf(BASE_AGENT_RESOURCES))],
            mutations=[updateActor({CLEAN_PLATE: 1})],
        )
    return GridObjectConfig(
        name="plate_station",
        on_use_handlers=handlers,
    )


def chopping_station_config(
    chop_ticks: int,
    *,
    enable_salad_recipe: bool = True,
    enable_soup_recipe: bool = True,
    enable_fries_recipe: bool = True,
) -> GridObjectConfig:
    needs_chopped_veg = enable_salad_recipe or enable_soup_recipe or enable_fries_recipe
    stored_ingredients = [CHOPPED_VEG, CHOPPED_MEAT]
    handlers: dict[str, Handler] = {}
    if needs_chopped_veg:
        handlers["finish_chop_veg"] = Handler(
            filters=[targetHas({CHOP_VEG_PROGRESS: chop_ticks - 1})],
            mutations=[
                updateActor({CHOPPED_VEG: 1}),
                updateTarget({CHOP_VEG_PROGRESS: -999}),
                logActorAgentStat("veg_chopped"),
                logStatToGame("veg_chopped"),
            ],
        )
        handlers["continue_chop_veg"] = Handler(
            filters=[targetHas({CHOP_VEG_PROGRESS: 1}), isNot(targetHas({CHOP_VEG_PROGRESS: chop_ticks - 1}))],
            mutations=[updateTarget({CHOP_VEG_PROGRESS: 1})],
        )
    if enable_soup_recipe:
        handlers["finish_chop_meat"] = Handler(
            filters=[targetHas({CHOP_MEAT_PROGRESS: chop_ticks - 1})],
            mutations=[
                updateActor({CHOPPED_MEAT: 1}),
                updateTarget({CHOP_MEAT_PROGRESS: -999}),
                logActorAgentStat("meat_chopped"),
                logStatToGame("meat_chopped"),
            ],
        )
        handlers["continue_chop_meat"] = Handler(
            filters=[targetHas({CHOP_MEAT_PROGRESS: 1}), isNot(targetHas({CHOP_MEAT_PROGRESS: chop_ticks - 1}))],
            mutations=[updateTarget({CHOP_MEAT_PROGRESS: 1})],
        )
    if enable_salad_recipe:
        handlers["plate_salad"] = Handler(
            filters=[
                actorHas({CLEAN_PLATE: 1}),
                targetHas({CHOPPED_VEG: 1}),
                isNot(targetHas({CHOP_VEG_PROGRESS: 1})),
                isNot(targetHas({CHOP_MEAT_PROGRESS: 1})),
            ],
            mutations=[updateActor({CLEAN_PLATE: -1, DISH_SALAD: 1}), updateTarget({CHOPPED_VEG: -1})],
        )
    if needs_chopped_veg:
        handlers["store_chopped_veg"] = Handler(
            filters=[
                actorHas({CHOPPED_VEG: 1}),
                isNot(targetHasAnyOf(stored_ingredients)),
                isNot(targetHas({CHOP_VEG_PROGRESS: 1})),
                isNot(targetHas({CHOP_MEAT_PROGRESS: 1})),
            ],
            mutations=[updateActor({CHOPPED_VEG: -1}), updateTarget({CHOPPED_VEG: 1})],
        )
        handlers["pickup_chopped_veg"] = Handler(
            filters=[
                isNot(actorHasAnyOf(BASE_AGENT_RESOURCES)),
                targetHas({CHOPPED_VEG: 1}),
                isNot(targetHas({CHOP_VEG_PROGRESS: 1})),
                isNot(targetHas({CHOP_MEAT_PROGRESS: 1})),
            ],
            mutations=[updateActor({CHOPPED_VEG: 1}), updateTarget({CHOPPED_VEG: -1})],
        )
        handlers["start_chop_veg"] = Handler(
            filters=[
                actorHas({VEG: 1}),
                isNot(targetHasAnyOf(stored_ingredients)),
                isNot(targetHas({CHOP_VEG_PROGRESS: 1})),
                isNot(targetHas({CHOP_MEAT_PROGRESS: 1})),
            ],
            mutations=[updateActor({VEG: -1}), updateTarget({CHOP_VEG_PROGRESS: 1})],
        )
    if enable_soup_recipe:
        handlers["store_chopped_meat"] = Handler(
            filters=[
                actorHas({CHOPPED_MEAT: 1}),
                isNot(targetHasAnyOf(stored_ingredients)),
                isNot(targetHas({CHOP_VEG_PROGRESS: 1})),
                isNot(targetHas({CHOP_MEAT_PROGRESS: 1})),
            ],
            mutations=[updateActor({CHOPPED_MEAT: -1}), updateTarget({CHOPPED_MEAT: 1})],
        )
        handlers["pickup_chopped_meat"] = Handler(
            filters=[
                isNot(actorHasAnyOf(BASE_AGENT_RESOURCES)),
                targetHas({CHOPPED_MEAT: 1}),
                isNot(targetHas({CHOP_VEG_PROGRESS: 1})),
                isNot(targetHas({CHOP_MEAT_PROGRESS: 1})),
            ],
            mutations=[updateActor({CHOPPED_MEAT: 1}), updateTarget({CHOPPED_MEAT: -1})],
        )
        handlers["start_chop_meat"] = Handler(
            filters=[
                actorHas({MEAT: 1}),
                isNot(targetHasAnyOf(stored_ingredients)),
                isNot(targetHas({CHOP_VEG_PROGRESS: 1})),
                isNot(targetHas({CHOP_MEAT_PROGRESS: 1})),
            ],
            mutations=[updateActor({MEAT: -1}), updateTarget({CHOP_MEAT_PROGRESS: 1})],
        )

    return GridObjectConfig(
        name="chopping_station",
        inventory=InventoryConfig(initial={CHOP_VEG_PROGRESS: 0, CHOP_MEAT_PROGRESS: 0}),
        on_use_handlers=handlers,
    )


def cooking_station_config(
    soup_cook_ticks: int,
    *,
    enable_soup_recipe: bool = True,
    enable_soup_burn: bool = True,
) -> GridObjectConfig:
    handlers: dict[str, Handler] = {}
    if enable_soup_recipe:
        handlers["collect_ready_soup"] = Handler(
            filters=[actorHas({CLEAN_PLATE: 1}), targetHas({POT_SOUP_READY: 1})],
            mutations=[
                updateActor({CLEAN_PLATE: -1, DISH_SOUP: 1}),
                updateTarget({POT_SOUP_READY: -1, POT_READY_AGE: -999}),
            ],
        )
        handlers["load_soup_veg_and_start"] = Handler(
            filters=[
                actorHas({CHOPPED_VEG: 1}),
                targetHas({CHOPPED_MEAT: 1}),
                isNot(targetHas({POT_SOUP_COOKING: 1})),
                isNot(targetHas({POT_SOUP_READY: 1})),
                isNot(targetHas({POT_SOUP_BURNED: 1})),
            ],
            mutations=[
                updateActor({CHOPPED_VEG: -1}),
                updateTarget({CHOPPED_MEAT: -1, POT_SOUP_COOKING: 1, POT_TIMER: soup_cook_ticks, POT_READY_AGE: -999}),
                logActorAgentStat("soups_started"),
                logStatToGame("soups_started"),
            ],
        )
        handlers["load_soup_meat_and_start"] = Handler(
            filters=[
                actorHas({CHOPPED_MEAT: 1}),
                targetHas({CHOPPED_VEG: 1}),
                isNot(targetHas({POT_SOUP_COOKING: 1})),
                isNot(targetHas({POT_SOUP_READY: 1})),
                isNot(targetHas({POT_SOUP_BURNED: 1})),
            ],
            mutations=[
                updateActor({CHOPPED_MEAT: -1}),
                updateTarget({CHOPPED_VEG: -1, POT_SOUP_COOKING: 1, POT_TIMER: soup_cook_ticks, POT_READY_AGE: -999}),
                logActorAgentStat("soups_started"),
                logStatToGame("soups_started"),
            ],
        )
        handlers["start_soup_cook"] = Handler(
            filters=[
                targetHas({CHOPPED_VEG: 1}),
                targetHas({CHOPPED_MEAT: 1}),
                isNot(targetHas({POT_SOUP_COOKING: 1})),
                isNot(targetHas({POT_SOUP_READY: 1})),
                isNot(targetHas({POT_SOUP_BURNED: 1})),
            ],
            mutations=[
                updateTarget(
                    {
                        CHOPPED_VEG: -1,
                        CHOPPED_MEAT: -1,
                        POT_SOUP_COOKING: 1,
                        POT_TIMER: soup_cook_ticks,
                        POT_READY_AGE: -999,
                    }
                ),
                logActorAgentStat("soups_started"),
                logStatToGame("soups_started"),
            ],
        )
        handlers["load_soup_veg"] = Handler(
            filters=[
                actorHas({CHOPPED_VEG: 1}),
                isNot(targetHas({CHOPPED_VEG: 1})),
                isNot(targetHas({POT_SOUP_COOKING: 1})),
                isNot(targetHas({POT_SOUP_READY: 1})),
                isNot(targetHas({POT_SOUP_BURNED: 1})),
            ],
            mutations=[updateActor({CHOPPED_VEG: -1}), updateTarget({CHOPPED_VEG: 1})],
        )
        handlers["load_soup_meat"] = Handler(
            filters=[
                actorHas({CHOPPED_MEAT: 1}),
                isNot(targetHas({CHOPPED_MEAT: 1})),
                isNot(targetHas({POT_SOUP_COOKING: 1})),
                isNot(targetHas({POT_SOUP_READY: 1})),
                isNot(targetHas({POT_SOUP_BURNED: 1})),
            ],
            mutations=[updateActor({CHOPPED_MEAT: -1}), updateTarget({CHOPPED_MEAT: 1})],
        )
    if enable_soup_burn:
        handlers["clear_burned_pot"] = Handler(
            filters=[targetHas({POT_SOUP_BURNED: 1})],
            mutations=[
                updateTarget({POT_SOUP_BURNED: -1, POT_TIMER: -999, POT_READY_AGE: -999}),
                logActorAgentStat("pots_cleared"),
                logStatToGame("pots_cleared"),
            ],
        )

    return GridObjectConfig(
        name="cooking_station",
        inventory=InventoryConfig(initial={POT_TIMER: 0, POT_READY_AGE: 0, CHOPPED_VEG: 0, CHOPPED_MEAT: 0}),
        on_use_handlers=handlers,
    )


def fryer_station_config(
    fries_cook_ticks: int,
    *,
    enable_fries_recipe: bool = True,
    enable_fries_burn: bool = True,
) -> GridObjectConfig:
    handlers: dict[str, Handler] = {}
    if enable_fries_recipe:
        handlers["collect_ready_fries"] = Handler(
            filters=[actorHas({CLEAN_PLATE: 1}), targetHas({FRYER_FRIES_READY: 1})],
            mutations=[
                updateActor({CLEAN_PLATE: -1, DISH_FRIES: 1}),
                updateTarget({FRYER_FRIES_READY: -1, FRYER_READY_AGE: -999}),
            ],
        )
        handlers["start_fries_cook"] = Handler(
            filters=[
                actorHas({CHOPPED_VEG: 1}),
                isNot(targetHas({FRYER_FRIES_COOKING: 1})),
                isNot(targetHas({FRYER_FRIES_READY: 1})),
                isNot(targetHas({FRYER_FRIES_BURNED: 1})),
            ],
            mutations=[
                updateActor({CHOPPED_VEG: -1}),
                updateTarget({FRYER_FRIES_COOKING: 1, FRYER_TIMER: fries_cook_ticks, FRYER_READY_AGE: -999}),
                logActorAgentStat("fries_started"),
                logStatToGame("fries_started"),
            ],
        )
    if enable_fries_burn:
        handlers["clear_burned_fryer"] = Handler(
            filters=[targetHas({FRYER_FRIES_BURNED: 1})],
            mutations=[
                updateTarget({FRYER_FRIES_BURNED: -1, FRYER_TIMER: -999, FRYER_READY_AGE: -999}),
                logActorAgentStat("fryers_cleared"),
                logStatToGame("fryers_cleared"),
            ],
        )

    return GridObjectConfig(
        name="fryer_station",
        inventory=InventoryConfig(initial={FRYER_TIMER: 0, FRYER_READY_AGE: 0}),
        on_use_handlers=handlers,
    )


def _ticket_is_active(resource_name: str) -> GameValueFilter:
    return GameValueFilter(
        target=HandlerTarget.TARGET,
        value=QueryInventoryValue(query=ORDER_BOARD_QUERY, item=resource_name),
        min=1,
    )


def serving_station_config(
    ticket_specs: list[TicketSpec],
    *,
    enable_queue_orders: bool = True,
    enabled_recipes: set[RecipeName] | None = None,
) -> GridObjectConfig:
    handlers: dict[str, Handler] = {}
    if not enable_queue_orders:
        return GridObjectConfig(name="serving_station", on_use_handlers=handlers)

    allowed_recipes = enabled_recipes if enabled_recipes is not None else {"salad", "soup", "fries"}
    for ticket in ticket_specs:
        if ticket.recipe not in allowed_recipes:
            continue
        dish_resource = dish_resource_for_recipe(ticket.recipe)
        handlers[f"serve_ticket_{ticket.index:03d}_{ticket.recipe}"] = Handler(
            filters=[
                actorHas({dish_resource: 1}),
                _ticket_is_active(ticket.resource),
            ],
            mutations=[
                updateActor({dish_resource: -1, DIRTY_PLATE: 1}),
                queryDelta(
                    ORDER_BOARD_QUERY,
                    {
                        ticket.resource: -1,
                        ticket.queue_resource: -1,
                    },
                ),
                logActorAgentStat("orders_served"),
                logActorAgentStat(f"orders_served_{ticket.recipe}"),
                logStatToGame("orders_served"),
                logStatToGame("orders_served_total"),
                logStatToGame(f"orders_served_{ticket.recipe}"),
            ],
        )

    return GridObjectConfig(name="serving_station", on_use_handlers=handlers)


def wash_station_config(wash_ticks: int, *, enable_wash_cycle: bool = True) -> GridObjectConfig:
    handlers: dict[str, Handler] = {}
    if enable_wash_cycle:
        handlers["finish_wash_plate"] = Handler(
            filters=[targetHas({WASH_PROGRESS: wash_ticks - 1})],
            mutations=[
                updateActor({CLEAN_PLATE: 1}),
                updateTarget({WASH_PROGRESS: -999}),
                logActorAgentStat("plates_washed"),
                logStatToGame("plates_washed"),
            ],
        )
        handlers["continue_wash_plate"] = Handler(
            filters=[targetHas({WASH_PROGRESS: 1}), isNot(targetHas({WASH_PROGRESS: wash_ticks - 1}))],
            mutations=[updateTarget({WASH_PROGRESS: 1})],
        )
        handlers["start_wash_plate"] = Handler(
            filters=[actorHas({DIRTY_PLATE: 1}), isNot(targetHas({WASH_PROGRESS: 1}))],
            mutations=[updateActor({DIRTY_PLATE: -1}), updateTarget({WASH_PROGRESS: 1})],
        )
    return GridObjectConfig(
        name="wash_station",
        inventory=InventoryConfig(initial={WASH_PROGRESS: 0}),
        on_use_handlers=handlers,
    )


def order_board_config(
    ticket_specs: list[TicketSpec],
    order_queue_max: int,
    initial_orders: dict[str, int] | None = None,
) -> GridObjectConfig:
    initial = {
        QUEUE_SALAD: 0,
        QUEUE_SOUP: 0,
        QUEUE_FRIES: 0,
        **{ticket.resource: 0 for ticket in ticket_specs},
    }
    if initial_orders:
        initial.update(initial_orders)

    limits: dict[str, ResourceLimitsConfig] = {
        "queue_counts": ResourceLimitsConfig(
            base=order_queue_max,
            max=order_queue_max,
            resources=QUEUE_COUNTER_RESOURCES,
        ),
    }
    if ticket_specs:
        limits["active_tickets"] = ResourceLimitsConfig(
            base=order_queue_max,
            max=order_queue_max,
            resources=[ticket.resource for ticket in ticket_specs],
        )
        limits.update(
            {
                f"ticket_{ticket.index:03d}": ResourceLimitsConfig(base=1, max=1, resources=[ticket.resource])
                for ticket in ticket_specs
            }
        )

    return GridObjectConfig(
        name="order_board",
        inventory=InventoryConfig(initial=initial, limits=limits),
    )


def order_events(
    max_steps: int,
    ticket_specs: list[TicketSpec],
    *,
    order_queue_max: int = ORDER_QUEUE_MAX,
    enable_queue_orders: bool = True,
    enabled_recipes: set[RecipeName] | None = None,
) -> dict[str, EventConfig]:
    if not enable_queue_orders:
        return {}

    allowed_recipes = enabled_recipes or {"salad", "soup", "fries"}
    enabled_ticket_specs = [ticket for ticket in ticket_specs if ticket.recipe in allowed_recipes]
    if not enabled_ticket_specs:
        return {}

    active_tickets = SumGameValue(
        values=[QueryInventoryValue(query=ORDER_BOARD_QUERY, item=ticket.resource) for ticket in enabled_ticket_specs]
    )
    events: dict[str, EventConfig] = {}
    for ticket in enabled_ticket_specs:
        events[f"ticket_arrival_{ticket.index:03d}_{ticket.recipe}"] = EventConfig(
            name=f"ticket_arrival_{ticket.index:03d}_{ticket.recipe}",
            target_query=ORDER_BOARD_QUERY,
            timesteps=[ticket.arrival],
            filters=[isNot(GameValueFilter(target=HandlerTarget.TARGET, value=active_tickets, min=order_queue_max))],
            mutations=[
                updateTarget({ticket.resource: 1, ticket.queue_resource: 1}),
                logStatToGame("orders_arrived"),
                logStatToGame(f"orders_arrived_{ticket.recipe}"),
            ],
        )
        events[f"ticket_expiry_{ticket.index:03d}_{ticket.recipe}"] = EventConfig(
            name=f"ticket_expiry_{ticket.index:03d}_{ticket.recipe}",
            target_query=ORDER_BOARD_QUERY,
            timesteps=[ticket.expiry],
            filters=[targetHas({ticket.resource: 1})],
            mutations=[
                updateTarget({ticket.resource: -1, ticket.queue_resource: -1}),
                logStatToGame("orders_expired"),
                logStatToGame(f"orders_expired_{ticket.recipe}"),
            ],
        )
    return events


def cooking_events(
    max_steps: int,
    *,
    soup_burn_ticks: int,
    enable_soup_recipe: bool = True,
    enable_soup_burn: bool = True,
) -> dict[str, EventConfig]:
    if not enable_soup_recipe:
        return {}

    events: dict[str, EventConfig] = {
        "soup_cook_timer_tick": EventConfig(
            name="soup_cook_timer_tick",
            target_query=COOKING_STATION_QUERY,
            timesteps=periodic(start=0, period=1, end=max_steps),
            filters=[targetHas({POT_SOUP_COOKING: 1}), targetHas({POT_TIMER: 1})],
            mutations=[updateTarget({POT_TIMER: -1})],
        ),
        "soup_finish_cook": EventConfig(
            name="soup_finish_cook",
            target_query=COOKING_STATION_QUERY,
            timesteps=periodic(start=0, period=1, end=max_steps),
            filters=[targetHas({POT_SOUP_COOKING: 1}), isNot(targetHas({POT_TIMER: 1}))],
            mutations=[
                updateTarget({POT_SOUP_COOKING: -1, POT_SOUP_READY: 1, POT_READY_AGE: -999}),
                logStatToGame("soups_ready"),
            ],
        ),
    }
    if enable_soup_burn:
        events["soup_ready_age_tick"] = EventConfig(
            name="soup_ready_age_tick",
            target_query=COOKING_STATION_QUERY,
            timesteps=periodic(start=0, period=1, end=max_steps),
            filters=[targetHas({POT_SOUP_READY: 1})],
            mutations=[updateTarget({POT_READY_AGE: 1})],
        )
        events["soup_burn"] = EventConfig(
            name="soup_burn",
            target_query=COOKING_STATION_QUERY,
            timesteps=periodic(start=0, period=1, end=max_steps),
            filters=[targetHas({POT_SOUP_READY: 1}), targetHas({POT_READY_AGE: soup_burn_ticks})],
            mutations=[
                updateTarget({POT_SOUP_READY: -1, POT_SOUP_BURNED: 1, POT_READY_AGE: -999}),
                logStatToGame("soups_burned"),
            ],
        )
    return events


def fryer_events(
    max_steps: int,
    *,
    fries_burn_ticks: int,
    enable_fries_recipe: bool = True,
    enable_fries_burn: bool = True,
) -> dict[str, EventConfig]:
    if not enable_fries_recipe:
        return {}

    events: dict[str, EventConfig] = {
        "fries_cook_timer_tick": EventConfig(
            name="fries_cook_timer_tick",
            target_query=FRYER_STATION_QUERY,
            timesteps=periodic(start=0, period=1, end=max_steps),
            filters=[targetHas({FRYER_FRIES_COOKING: 1}), targetHas({FRYER_TIMER: 1})],
            mutations=[updateTarget({FRYER_TIMER: -1})],
        ),
        "fries_finish_cook": EventConfig(
            name="fries_finish_cook",
            target_query=FRYER_STATION_QUERY,
            timesteps=periodic(start=0, period=1, end=max_steps),
            filters=[targetHas({FRYER_FRIES_COOKING: 1}), isNot(targetHas({FRYER_TIMER: 1}))],
            mutations=[
                updateTarget({FRYER_FRIES_COOKING: -1, FRYER_FRIES_READY: 1, FRYER_READY_AGE: -999}),
                logStatToGame("fries_ready"),
            ],
        ),
    }
    if enable_fries_burn:
        events["fries_ready_age_tick"] = EventConfig(
            name="fries_ready_age_tick",
            target_query=FRYER_STATION_QUERY,
            timesteps=periodic(start=0, period=1, end=max_steps),
            filters=[targetHas({FRYER_FRIES_READY: 1})],
            mutations=[updateTarget({FRYER_READY_AGE: 1})],
        )
        events["fries_burn"] = EventConfig(
            name="fries_burn",
            target_query=FRYER_STATION_QUERY,
            timesteps=periodic(start=0, period=1, end=max_steps),
            filters=[targetHas({FRYER_FRIES_READY: 1}), targetHas({FRYER_READY_AGE: fries_burn_ticks})],
            mutations=[
                updateTarget({FRYER_FRIES_READY: -1, FRYER_FRIES_BURNED: 1, FRYER_READY_AGE: -999}),
                logStatToGame("fries_burned"),
            ],
        )
    return events


def queue_instrumentation_events(max_steps: int) -> dict[str, EventConfig]:
    active_orders = SumGameValue(
        values=[
            QueryInventoryValue(query=ORDER_BOARD_QUERY, item=QUEUE_SALAD),
            QueryInventoryValue(query=ORDER_BOARD_QUERY, item=QUEUE_SOUP),
            QueryInventoryValue(query=ORDER_BOARD_QUERY, item=QUEUE_FRIES),
        ]
    )
    return {
        "queue_pressure_tick": EventConfig(
            name="queue_pressure_tick",
            target_query=ORDER_BOARD_QUERY,
            timesteps=periodic(start=0, period=1, end=max_steps),
            mutations=[
                logStatToGame("queue_samples"),
                logStatToGame(
                    "queue_salad_depth_sum",
                    source=QueryInventoryValue(query=ORDER_BOARD_QUERY, item=QUEUE_SALAD),
                ),
                logStatToGame(
                    "queue_soup_depth_sum",
                    source=QueryInventoryValue(query=ORDER_BOARD_QUERY, item=QUEUE_SOUP),
                ),
                logStatToGame(
                    "queue_fries_depth_sum",
                    source=QueryInventoryValue(query=ORDER_BOARD_QUERY, item=QUEUE_FRIES),
                ),
                logStatToGame("orders_active_sum", source=active_orders),
            ],
        ),
    }


def _agent_config() -> AgentConfig:
    rewards: dict[str, AgentReward] = {
        "orders": reward(stat("orders_served"), weight=1.0),
        "soup_bonus": reward(stat("orders_served_soup"), weight=0.2),
        "fries_bonus": reward(stat("orders_served_fries"), weight=0.15),
        "expiry_penalty": reward(stat("game.orders_expired"), weight=-0.05),
    }
    return AgentConfig(
        inventory=InventoryConfig(
            initial={},
            limits={
                "carry": ResourceLimitsConfig(base=1, max=1, resources=BASE_AGENT_RESOURCES),
            },
        ),
        rewards=rewards,
    )


def counter_config() -> WallConfig:
    handlers: dict[str, Handler] = {}
    for resource in BASE_AGENT_RESOURCES:
        handlers[f"deposit_{resource}"] = Handler(
            filters=[actorHas({resource: 1}), isNot(targetHasAnyOf(BASE_AGENT_RESOURCES))],
            mutations=[deposit({resource: 1})],
        )
    for resource in BASE_AGENT_RESOURCES:
        handlers[f"withdraw_{resource}"] = Handler(
            filters=[isNot(actorHasAnyOf(BASE_AGENT_RESOURCES)), targetHas({resource: 1})],
            mutations=[withdraw({resource: 1})],
        )

    return WallConfig(
        name="wall",
        inventory=InventoryConfig(
            limits={
                "carry": ResourceLimitsConfig(base=1, max=1, resources=BASE_AGENT_RESOURCES),
            }
        ),
        on_use_handlers=handlers,
    )


class OvercookedGame(CoGameMission):
    default_variant: str | None = Field(default="full")
    max_steps: int = Field(default=300)
    ticket_first_arrival: int = Field(default=TICKET_FIRST_ARRIVAL, ge=0)
    ticket_interarrival: int = Field(default=TICKET_INTERARRIVAL, ge=1)
    ticket_deadline: int = Field(default=TICKET_DEADLINE, ge=1)
    chop_ticks: int = Field(default=CHOP_TICKS, ge=2)
    wash_ticks: int = Field(default=WASH_TICKS, ge=2)
    soup_cook_ticks: int = Field(default=SOUP_COOK_TICKS, ge=1)
    soup_burn_ticks: int = Field(default=SOUP_BURN_TICKS, ge=1)
    fries_cook_ticks: int = Field(default=FRIES_COOK_TICKS, ge=1)
    fries_burn_ticks: int = Field(default=FRIES_BURN_TICKS, ge=1)
    order_queue_max: int = Field(default=ORDER_QUEUE_MAX, ge=1)
    hub_layout: Literal["default", "tight", "cramped_room", "service_pass_room"] = Field(default="service_pass_room")
    station_order: list[str] = Field(default_factory=lambda: list(STATIONS))
    station_offsets: list[tuple[int, int]] = Field(default_factory=lambda: list(DEFAULT_KITCHEN_STATION_OFFSETS))
    hub_width: int = Field(default=18, ge=11)
    hub_height: int = Field(default=14, ge=11)
    randomize_spawn_positions: bool = Field(default=False)
    enable_queue_orders: bool = Field(default=False)
    enable_salad_recipe: bool = Field(default=False)
    enable_soup_recipe: bool = Field(default=False)
    enable_fries_recipe: bool = Field(default=False)
    enable_wash_cycle: bool = Field(default=False)
    enable_soup_burn: bool = Field(default=False)
    enable_fries_burn: bool = Field(default=False)

    @classmethod
    def create(cls, num_agents: int, max_steps: int) -> OvercookedGame:
        return cls(
            name="basic",
            description="Overcogged event-driven order queue game",
            map_builder=cls._map(num_agents),
            hub_layout="service_pass_room" if num_agents > 1 else "cramped_room",
            num_cogs=num_agents,
            min_cogs=1,
            max_cogs=num_agents,
            max_steps=max_steps,
        )

    @classmethod
    def variant_module_prefixes(cls) -> tuple[str, ...]:
        return ("cogames.games.overcogged.variants.",)

    def with_variants(self, variants: Sequence[str | CoGameMissionVariant]) -> Self:
        copy = super().with_variants(variants)
        requested_names = normalize_variant_names(
            [variant.name if isinstance(variant, CoGameMissionVariant) else variant for variant in variants]
        )
        if any(name in HIDDEN_VARIANT_NAMES for name in requested_names):
            copy.default_variant = None
        return copy

    def _active_variant_names(self) -> list[str]:
        names: list[str] = []
        if self.default_variant:
            names.append(self.default_variant)
        names.extend(self._base_variants)
        return normalize_variant_names(names)

    def _resolved_settings(self) -> OvercookedSettings:
        self._variant_registry = resolve_variant_selection(self._active_variant_names())
        settings = OvercookedSettings.from_mission(self)
        for variant in self._variant_registry.configured():
            if hasattr(variant, "modify_mission"):
                cast(SupportsModifyMission, variant).modify_mission(settings)
        return settings

    def make_base_env(self) -> MettaGridConfig:
        settings = self._resolved_settings()
        num_cogs = cast(int, self.num_cogs)
        if num_cogs == 1 and settings.enable_queue_orders:
            # Single-cog Overcogged now uses true one-item carry. Soften solo
            # ticket pacing to keep the mode viable without changing multi-cog
            # balance targets.
            if settings.ticket_interarrival <= 12:
                settings.ticket_interarrival = math.ceil(settings.ticket_interarrival * 2.25)
                settings.ticket_deadline = math.ceil(settings.ticket_deadline * 2.25)
            else:
                settings.ticket_interarrival *= 3
                settings.ticket_deadline *= 3
        soup_recipe_enabled = settings.enable_soup_recipe
        fries_recipe_enabled = settings.enable_fries_recipe
        soup_burn_enabled = soup_recipe_enabled and settings.enable_soup_burn
        fries_burn_enabled = fries_recipe_enabled and settings.enable_fries_burn
        enabled_recipes: set[RecipeName] = set()
        if settings.enable_salad_recipe:
            enabled_recipes.add("salad")
        if soup_recipe_enabled:
            enabled_recipes.add("soup")
        if fries_recipe_enabled:
            enabled_recipes.add("fries")

        ticket_specs = build_ticket_specs(
            settings.max_steps,
            first_arrival=settings.ticket_first_arrival,
            interarrival=settings.ticket_interarrival,
            deadline=settings.ticket_deadline,
        )
        resource_names = resource_names_for_tickets(ticket_specs)
        map_builder = cast(MapGenConfig, self.map_builder.model_copy(deep=True))
        hub = cast(CompoundConfig, map_builder.instance)
        hub.layout = settings.hub_layout
        hub.hub_width = settings.hub_width
        hub.hub_height = settings.hub_height
        map_builder.width = settings.hub_width
        map_builder.height = settings.hub_height
        hub.stations = validate_station_order(settings.station_order)
        hub.station_offsets = (
            validate_station_offsets(settings.station_offsets) if settings.hub_layout == "default" else None
        )
        hub.randomize_spawn_positions = settings.randomize_spawn_positions
        hub.spawn_count = num_cogs

        game = GameConfig(
            map_builder=map_builder,
            max_steps=settings.max_steps,
            num_agents=num_cogs,
            resource_names=resource_names,
            obs=ObsConfig(
                global_obs=GlobalObsConfig(
                    local_position=True,
                    last_action_move=True,
                ),
            ),
            actions=ActionsConfig(
                move=MoveActionConfig(),
                noop=NoopActionConfig(),
                change_vibe=ChangeVibeActionConfig(enabled=False, vibes=[]),
            ),
            talk=TalkConfig(enabled=True, max_length=140, cooldown_steps=0),
            agents=[_agent_config() for _ in range(num_cogs)],
            objects={
                "wall": counter_config(),
                "veg_station": veg_station_config(enable_veg_pickup=bool(enabled_recipes)),
                "meat_station": meat_station_config(enable_meat_pickup=soup_recipe_enabled),
                "plate_station": plate_station_config(
                    enable_plate_pickup=bool(enabled_recipes or settings.enable_wash_cycle)
                ),
                "chopping_station": chopping_station_config(
                    settings.chop_ticks,
                    enable_salad_recipe=settings.enable_salad_recipe,
                    enable_soup_recipe=soup_recipe_enabled,
                    enable_fries_recipe=fries_recipe_enabled,
                ),
                "cooking_station": cooking_station_config(
                    settings.soup_cook_ticks,
                    enable_soup_recipe=soup_recipe_enabled,
                    enable_soup_burn=soup_burn_enabled,
                ),
                "fryer_station": fryer_station_config(
                    settings.fries_cook_ticks,
                    enable_fries_recipe=fries_recipe_enabled,
                    enable_fries_burn=fries_burn_enabled,
                ),
                "serving_station": serving_station_config(
                    ticket_specs,
                    enable_queue_orders=settings.enable_queue_orders,
                    enabled_recipes=enabled_recipes,
                ),
                "wash_station": wash_station_config(settings.wash_ticks, enable_wash_cycle=settings.enable_wash_cycle),
                "order_board": order_board_config(ticket_specs, settings.order_queue_max),
            },
            events={
                **order_events(
                    max_steps=settings.max_steps,
                    ticket_specs=ticket_specs,
                    order_queue_max=settings.order_queue_max,
                    enable_queue_orders=settings.enable_queue_orders,
                    enabled_recipes=enabled_recipes,
                ),
                **cooking_events(
                    max_steps=settings.max_steps,
                    soup_burn_ticks=settings.soup_burn_ticks,
                    enable_soup_recipe=soup_recipe_enabled,
                    enable_soup_burn=soup_burn_enabled,
                ),
                **fryer_events(
                    max_steps=settings.max_steps,
                    fries_burn_ticks=settings.fries_burn_ticks,
                    enable_fries_recipe=fries_recipe_enabled,
                    enable_fries_burn=fries_burn_enabled,
                ),
                **queue_instrumentation_events(max_steps=settings.max_steps),
            },
            render=overcooked_render_config(settings, ticket_specs),
        )
        return MettaGridConfig(game=game)

    def make_env(self) -> MettaGridConfig:
        env = self.make_base_env()
        env.label = self.full_name()
        return env

    @staticmethod
    def _map(num_agents: int) -> MapGenConfig:
        return MapGenConfig(
            width=18,
            height=14,
            border_width=0,
            instance=CompoundConfig(
                spawn_count=num_agents,
                hub_object="empty",
                corner_bundle="none",
                cross_bundle="none",
                cross_distance=6,
                randomize_spawn_positions=False,
                stations=list(STATIONS),
                station_offsets=list(DEFAULT_KITCHEN_STATION_OFFSETS),
                layout="service_pass_room" if num_agents > 1 else "cramped_room",
                hub_width=18,
                hub_height=14,
                outer_clearance=0,
            ),
        )


class OvercookedCoGame(CoGame):
    def __init__(self) -> None:
        self.name = "overcogged"
        self._missions: list[CoGameMission] | None = None
        self._variant_registry: VariantRegistry | None = None
        self._eval_missions: list[CoGameMission] = []

    def _ensure_loaded(self) -> None:
        if self._variant_registry is not None:
            return

        _, _, variants = load_variants()
        self._missions = [make_basic_mission(), make_classic_mission()]
        self._variant_registry = VariantRegistry(list(variants))

    @property
    def missions(self) -> list[CoGameMission]:
        self._ensure_loaded()
        missions = self._missions
        assert missions is not None
        return missions

    @property
    def variant_registry(self) -> VariantRegistry:
        self._ensure_loaded()
        variant_registry = self._variant_registry
        assert variant_registry is not None
        return variant_registry

    @property
    def eval_missions(self) -> list[CoGameMission]:
        return self._eval_missions


register_game(OvercookedCoGame())
