"""Classic Overcogged variants preserved from the original Metta game."""

from __future__ import annotations

import random
from collections import Counter
from typing import cast

from cogames.core import CoGameMissionVariant, Deps
from cogames.games.overcogged.classic.game import (
    ALL_ELEMENTS,
    CARRY_RESOURCES,
    COOK_TIME,
    INGREDIENTS_NEEDED,
    _hub_render_assets_missing,
)
from cogames.games.overcogged.classic.map import overcogged_map
from mettagrid.config.event_config import EventConfig, periodic
from mettagrid.config.filter import (
    actorHas,
    actorHasAnyOf,
    isNot,
    query,
    targetHas,
    targetHasAnyOf,
)
from mettagrid.config.handler_config import Handler
from mettagrid.config.mettagrid_config import GridObjectConfig, MettaGridConfig, ResourceLimitsConfig
from mettagrid.config.mutation import ResourceDeltaMutation, deposit, updateActor, updateTarget, withdraw
from mettagrid.config.render_config import RenderStatusBarConfig
from mettagrid.config.tag import typeTag
from mettagrid.map_builder.ascii import AsciiMapBuilderConfig

EXTRA_CARRY = ["oxygen", "germanium", "silicon"]
ALL_CARRY = CARRY_RESOURCES + EXTRA_CARRY
NUM_RECIPES = 20
RECIPE_DURATION = 50
RECIPE_RESOURCES = ["recipe_carbon", "recipe_oxygen", "recipe_germanium", "recipe_silicon"]
HUB_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BURN_TIME = 40
FAST_BURN_TIME = BURN_TIME // 2
SHORT_COOK_TIME = COOK_TIME // 2
LONG_COOK_TIME = COOK_TIME * 2
TIP_WEIGHT = 0.5


def _generate_recipe_schedule(seed: int = 42) -> list[Counter[str]]:
    rng = random.Random(seed)
    all_elements = list(ALL_ELEMENTS) + EXTRA_CARRY
    recipes: list[Counter[str]] = []
    for _ in range(NUM_RECIPES):
        ingredients = [rng.choice(all_elements) for _ in range(INGREDIENTS_NEEDED)]
        recipes.append(Counter(ingredients))
    return recipes


def _recipe_counts(recipe: Counter[str]) -> dict[str, int]:
    return {f"recipe_{element}": count for element, count in recipe.items()}


def _classic_map_builder(env: MettaGridConfig) -> AsciiMapBuilderConfig:
    return cast(AsciiMapBuilderConfig, env.game.map_builder)


def _set_cook_complete_energy(env: MettaGridConfig, energy: int) -> None:
    for event_name, event in env.game.events.items():
        if not event_name.startswith("cook_complete"):
            continue
        event.mutations = [
            mutation.model_copy(update={"deltas": {**mutation.deltas, "energy": energy}})
            if isinstance(mutation, ResourceDeltaMutation) and "energy" in mutation.deltas
            else mutation
            for mutation in event.mutations
        ]


class RecipesVariant(CoGameMissionVariant):
    name: str = "recipes"
    description: str = "Adds alternative ingredients and per-hub recipe schedules."

    def modify_env(self, mission, env: MettaGridConfig) -> None:
        all_elements = list(ALL_ELEMENTS) + EXTRA_CARRY
        env.game.resource_names = list(env.game.resource_names) + EXTRA_CARRY + RECIPE_RESOURCES

        for agent in env.game.agents:
            carry = agent.inventory.limits["carry"]
            carry.resources = list(carry.resources) + EXTRA_CARRY

        for element in EXTRA_CARRY:
            env.game.objects[f"{element}_extractor"] = GridObjectConfig(
                name=f"{element}_extractor",
                on_use_handlers={
                    "pick_up": Handler(
                        filters=[actorHas({"miner": 1}), isNot(actorHasAnyOf(ALL_CARRY))],
                        mutations=[updateActor({element: 1, "num_items": 1})],
                    ),
                },
            )

        hub_handlers: dict[str, Handler] = {
            "pickup_heart": Handler(
                filters=[
                    actorHas({"scrambler": 1}),
                    isNot(actorHasAnyOf(ALL_CARRY)),
                    targetHas({"heart": 1}),
                ],
                mutations=[
                    updateActor({"heart": 1, "num_items": 1}),
                    updateTarget({"heart": -1}),
                ],
            )
        }
        for element in all_elements:
            hub_handlers[f"deposit_{element}"] = Handler(
                filters=[
                    actorHas({element: 1}),
                    targetHas({f"recipe_{element}": 1}),
                    targetHas({"missing": 1}),
                    isNot(targetHas({"heart": 1})),
                ],
                mutations=[
                    updateActor({element: -1, "num_items": -1}),
                    updateTarget({"missing": -1, f"recipe_{element}": -1}),
                ],
            )

        hub_base = env.game.objects["hub"]
        map_builder = _classic_map_builder(env)
        hub_positions = []
        for r, row in enumerate(map_builder.map_data):
            for c, cell in enumerate(row):
                if cell == "P":
                    hub_positions.append((r, c))

        schedules = [_generate_recipe_schedule(seed=42 + i) for i in range(len(hub_positions))]
        hub_tags = [f"hub_{i}" for i in range(len(hub_positions))]
        env.game.tags = list(env.game.tags) + hub_tags

        del env.game.objects["hub"]
        for i, (r, c) in enumerate(hub_positions):
            hub_char = HUB_CHARS[i]
            map_builder.map_data[r][c] = hub_char
            map_builder.char_to_map_name[hub_char] = f"hub_{i}"

            hub_config = hub_base.model_copy(deep=True)
            hub_config.name = "hub"
            hub_config.map_name = f"hub_{i}"
            hub_config.tags = [hub_tags[i]]
            hub_config.on_use_handlers = hub_handlers
            hub_config.inventory.limits.pop("ingredients", None)
            hub_config.inventory.limits["recipes"] = ResourceLimitsConfig(
                base=INGREDIENTS_NEEDED * len(RECIPE_RESOURCES),
                max=INGREDIENTS_NEEDED * len(RECIPE_RESOURCES),
                resources=RECIPE_RESOURCES,
            )
            hub_config.inventory.initial.pop("carbon", None)
            hub_config.inventory.initial.update(_recipe_counts(schedules[i][0]))
            env.game.objects[f"hub_{i}"] = hub_config

        del env.game.events["cook_complete"]
        for hub_idx, schedule in enumerate(schedules):
            for step, recipe in enumerate(schedule):
                start_t = step * RECIPE_DURATION
                end_t = (step + 1) * RECIPE_DURATION if step + 1 < len(schedule) else None
                timesteps = (
                    [t for t in range(start_t, end_t)] if end_t is not None else periodic(start=start_t, period=1)
                )
                env.game.events[f"cook_complete_hub{hub_idx}_r{step}"] = EventConfig(
                    name=f"cook_complete_hub{hub_idx}_r{step}",
                    target_query=query(hub_tags[hub_idx]),
                    timesteps=timesteps,
                    filters=[
                        isNot(targetHas({"missing": 1})),
                        isNot(targetHas({"energy": 1})),
                        isNot(targetHas({"heart": 1})),
                    ],
                    mutations=[
                        updateTarget(
                            {
                                "missing": INGREDIENTS_NEEDED,
                                "energy": COOK_TIME,
                                "heart": 1,
                                **_recipe_counts(recipe),
                            }
                        ),
                    ],
                    max_targets=None,
                )

        for hub_idx, schedule in enumerate(schedules):
            for step in range(1, len(schedule)):
                previous = schedule[step - 1]
                nxt = schedule[step]
                if previous == nxt:
                    continue
                clear = {
                    "missing": -INGREDIENTS_NEEDED,
                    "energy": -65535,
                    **{resource: -INGREDIENTS_NEEDED for resource in RECIPE_RESOURCES},
                }
                restore = {
                    "missing": INGREDIENTS_NEEDED,
                    "energy": COOK_TIME,
                    **_recipe_counts(nxt),
                }
                env.game.events[f"recipe_hub{hub_idx}_step{step}"] = EventConfig(
                    name=f"recipe_hub{hub_idx}_step{step}",
                    target_query=query(hub_tags[hub_idx]),
                    timesteps=[step * RECIPE_DURATION],
                    mutations=[updateTarget(clear), updateTarget(restore)],
                    max_targets=None,
                )

        env.game.render.assets["hub"] = _hub_render_assets_missing()
        env.game.render.object_status["hub"] = {
            "energy": RenderStatusBarConfig(
                resource="energy",
                short_name="E",
                max=COOK_TIME,
                divisions=COOK_TIME,
                rank=0,
            ),
        }

        chest = env.game.objects["chest"]
        chest.inventory.limits["items"].resources = all_elements
        for resource in EXTRA_CARRY:
            chest.on_use_handlers[f"deposit_{resource}"] = Handler(
                filters=[
                    actorHas({"miner": 1}),
                    actorHas({resource: 1}),
                    isNot(targetHasAnyOf(all_elements)),
                ],
                mutations=[deposit({resource: 1}), updateActor({"num_items": -1})],
            )
            chest.on_use_handlers[f"withdraw_{resource}"] = Handler(
                filters=[
                    actorHas({"miner": 1}),
                    isNot(actorHasAnyOf(ALL_CARRY)),
                    targetHas({resource: 1}),
                ],
                mutations=[withdraw({resource: 1}), updateActor({"num_items": 1})],
            )


class TipsVariant(CoGameMissionVariant):
    name: str = "tips"
    description: str = "Bonus reward weighted by delivery throughput."

    def modify_env(self, mission, env: MettaGridConfig) -> None:
        from mettagrid.config.game_value import stat as game_stat  # noqa: PLC0415
        from mettagrid.config.reward_config import reward  # noqa: PLC0415

        for agent in env.game.agents:
            agent.rewards["tips"] = reward(game_stat("delivery", delta=True), weight=TIP_WEIGHT)


class BurnVariant(CoGameMissionVariant):
    name: str = "burn"
    description: str = "Cooked hearts left in a hub too long become burned."

    def modify_env(self, mission, env: MettaGridConfig) -> None:
        env.game.resource_names = list(env.game.resource_names) + ["laser", "decoder"]

        hub_names = [name for name in env.game.objects if name == "hub" or name.startswith("hub_")]
        carry_resources = list(env.game.agents[0].inventory.limits["carry"].resources)

        for hub_name in hub_names:
            hub = env.game.objects[hub_name]
            hub.inventory.limits["burn"] = ResourceLimitsConfig(base=65535, max=65535, resources=["laser"])
            hub.inventory.limits["burned"] = ResourceLimitsConfig(base=1, max=1, resources=["decoder"])

            old_handlers = dict(hub.on_use_handlers)
            new_handlers: dict[str, Handler] = {}
            inserted = False
            for key, handler in old_handlers.items():
                if key.startswith("deposit_") and not inserted:
                    new_handlers["clear_burned"] = Handler(
                        filters=[
                            isNot(actorHasAnyOf(carry_resources)),
                            targetHas({"decoder": 1}),
                        ],
                        mutations=[updateTarget({"decoder": -1})],
                    )
                    inserted = True
                new_handlers[key] = handler
            if not inserted:
                new_handlers["clear_burned"] = Handler(
                    filters=[
                        isNot(actorHasAnyOf(carry_resources)),
                        targetHas({"decoder": 1}),
                    ],
                    mutations=[updateTarget({"decoder": -1})],
                )
            hub.on_use_handlers = new_handlers

        env.game.events["cook_tick"].filters = list(env.game.events["cook_tick"].filters) + [
            isNot(targetHas({"decoder": 1}))
        ]
        env.game.events["burn_tick"] = EventConfig(
            name="burn_tick",
            target_query=query(typeTag("hub")),
            timesteps=periodic(start=0, period=1),
            filters=[targetHas({"heart": 1})],
            mutations=[updateTarget({"laser": 1})],
            max_targets=None,
        )
        env.game.events["burn_complete"] = EventConfig(
            name="burn_complete",
            target_query=query(typeTag("hub")),
            timesteps=periodic(start=0, period=1),
            filters=[targetHas({"laser": BURN_TIME})],
            mutations=[updateTarget({"heart": -1, "laser": -BURN_TIME, "decoder": 1})],
            max_targets=None,
        )


class ShortCookVariant(CoGameMissionVariant):
    name: str = "short_cook"
    description: str = f"Cook time reduced to {SHORT_COOK_TIME}."

    def modify_env(self, mission, env: MettaGridConfig) -> None:
        for obj_name, obj in env.game.objects.items():
            if obj_name == "hub" or obj_name.startswith("hub_"):
                obj.inventory.initial["energy"] = SHORT_COOK_TIME
        _set_cook_complete_energy(env, SHORT_COOK_TIME)


class LongCookVariant(CoGameMissionVariant):
    name: str = "long_cook"
    description: str = f"Cook time increased to {LONG_COOK_TIME}."

    def modify_env(self, mission, env: MettaGridConfig) -> None:
        for obj_name, obj in env.game.objects.items():
            if obj_name == "hub" or obj_name.startswith("hub_"):
                obj.inventory.initial["energy"] = LONG_COOK_TIME
        _set_cook_complete_energy(env, LONG_COOK_TIME)


class FastBurnVariant(CoGameMissionVariant):
    name: str = "fast_burn"
    description: str = f"Burn time reduced to {FAST_BURN_TIME}."

    def dependencies(self) -> Deps:
        return Deps(required=[BurnVariant])

    def modify_env(self, mission, env: MettaGridConfig) -> None:
        burn_complete = env.game.events["burn_complete"]
        burn_complete.filters = [targetHas({"laser": FAST_BURN_TIME})]
        burn_complete.mutations = [updateTarget({"heart": -1, "laser": -FAST_BURN_TIME, "decoder": 1})]


class CrampedKitchenVariant(CoGameMissionVariant):
    name: str = "cramped_kitchen"
    description: str = "Smaller kitchen with tighter corridors and more counters."

    def modify_env(self, mission, env: MettaGridConfig) -> None:
        num_agents = env.game.num_agents
        cramped = overcogged_map(max(num_agents, 2), seed=42)

        target_h = min(10, len(cramped.map_data))
        target_w = min(10, len(cramped.map_data[0]) if cramped.map_data else 10)

        new_map: list[list[str]] = []
        for r in range(target_h):
            row = cramped.map_data[r][:target_w]
            if r == 0 or r == target_h - 1:
                row = ["#"] * target_w
            else:
                row[0] = "#"
                if len(row) < target_w:
                    row.extend(["#"] * (target_w - len(row)))
                row[target_w - 1] = "#"
            new_map.append(row)

        empty_cells = []
        for r in range(1, target_h - 1):
            for c in range(1, target_w - 1):
                if new_map[r][c] == ".":
                    empty_cells.append((r, c))

        rng = random.Random(42)
        rng.shuffle(empty_cells)
        fillable = empty_cells[num_agents:]
        for r, c in fillable[: len(fillable) * 3 // 5]:
            new_map[r][c] = "C"

        spawns = sum(cell == "@" for row in new_map for cell in row)
        if spawns < num_agents:
            open_cells = [
                (r, c) for r in range(1, target_h - 1) for c in range(1, target_w - 1) if new_map[r][c] == "."
            ]
            rng.shuffle(open_cells)
            for r, c in open_cells[: num_agents - spawns]:
                new_map[r][c] = "@"

        cramped.map_data = new_map
        env.game.map_builder = cramped


class FullVariant(CoGameMissionVariant):
    name: str = "full"
    description: str = "Recipes and burn mechanics combined."

    def dependencies(self) -> Deps:
        return Deps(required=[RecipesVariant, BurnVariant])
