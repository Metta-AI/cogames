"""Shared test infrastructure for variant tests.

Provides StationTestHarness for deterministic station interaction testing.
"""

from dataclasses import dataclass
from typing import Any

from cogsguard.game import GEAR
from cogsguard.game.elements import ElementsVariant
from cogsguard.game.teams import TeamConfig
from cogsguard.game.teams.junction import TeamJunctionVariant
from cogsguard.game.territory import (
    HUB_ALIGN_DISTANCE,
    JUNCTION_ALIGN_DISTANCE,
    net_materialized_query,
)

from mettagrid.config.filter import anyOf, hasTag, hasTagPrefix, isNear, isNot, sharedTagPrefix
from mettagrid.config.handler_config import Handler, actorHas, firstMatch, updateActor
from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    GameConfig,
    GridObjectConfig,
    InventoryConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ResourceLimitsConfig,
    WallConfig,
)
from mettagrid.config.mutation import (
    addTag,
    recomputeMaterializedQuery,
    removeTag,
    removeTagPrefix,
)
from mettagrid.config.mutation.stats_mutation import logActorAgentStat, logStatToGame
from mettagrid.config.query import MaterializedQuery, query
from mettagrid.config.tag import typeTag
from mettagrid.config.territory_config import TerritoryConfig
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.simulator import Simulation

ELEMENTS = ElementsVariant().elements
RESOURCES = ["energy", "heart", "hp", "solar", *ELEMENTS, *GEAR]


def resource_limits() -> dict[str, ResourceLimitsConfig]:
    """Create resource limits for all CvC resources.

    The 'gear' limit is required by TeamConfig.gear_station's ClearInventoryMutation.
    """
    return {
        "all": ResourceLimitsConfig(base=10000, max=10000, resources=RESOURCES),
        "gear": ResourceLimitsConfig(base=1, max=1, resources=GEAR),
    }


def team_tags(*teams: TeamConfig) -> list[str]:
    """Collect all tags from the given teams."""
    return [t.team_tag() for t in teams]


def hub_object(team: str, initial: dict[str, int]) -> GridObjectConfig:
    """Create a hub GridObjectConfig with the given team tag and initial inventory."""
    return GridObjectConfig(
        name="hub",
        map_name=f"{team}:hub",
        tags=[f"team:{team}"],
        inventory=InventoryConfig(initial=initial, limits=resource_limits()),
    )


def count_junctions_by_team(sim: Simulation, team_tag: str) -> int:
    """Count junctions that have the given team tag."""
    id_map = sim._config.game.id_map()
    tag_names = id_map.tag_names()
    tag_name_to_id = {name: idx for idx, name in enumerate(tag_names)}
    tag_id = tag_name_to_id.get(team_tag)
    if tag_id is None:
        return 0
    count = 0
    for obj in sim.grid_objects().values():
        if obj.get("type_name") == "junction" and obj.get("has_tag", lambda _: False)(tag_id):
            count += 1
    return count


def has_tag_at(sim: Simulation, tag_name: str, row: int, col: int, type_name: str | None = None) -> bool:
    """Check if an object at (row, col) has the given tag. Optionally filter by type_name."""
    id_map = sim._config.game.id_map()
    tag_names = id_map.tag_names()
    tag_name_to_id = {name: idx for idx, name in enumerate(tag_names)}
    tag_id = tag_name_to_id.get(tag_name)
    if tag_id is None:
        return False
    for obj in sim.grid_objects().values():
        if obj.get("r") == row and obj.get("c") == col:
            if type_name and obj.get("type_name") != type_name:
                continue
            if obj["has_tag"](tag_id):
                return True
    return False


@dataclass
class StationTestHarness:
    """Minimal environment for deterministic station interaction tests."""

    simulation: Simulation
    agent_id: int = 0

    @classmethod
    def create(
        cls,
        station: GridObjectConfig,
        agent_inventory: dict[str, int] | None = None,
        agent_team: str | None = None,
        tags: list[str] | None = None,
        extra_objects: list[GridObjectConfig] | None = None,
        materialize_queries: list[MaterializedQuery] | None = None,
        extra_resources: list[str] | None = None,
        territories: dict[str, TerritoryConfig] | None = None,
    ) -> "StationTestHarness":
        """Create a 5x5 map with agent at (1,2) and station at (2,2).

        Agent moves east to interact with station.
        """
        station_map_name = station.map_name or station.name

        objects: dict[str, Any] = {
            "wall": WallConfig(),
            station_map_name: station,
        }

        map_data = [
            ["#", "#", "#", "#", "#"],
            ["#", ".", ".", ".", "#"],
            ["#", "@", "S", ".", "#"],
            ["#", ".", ".", ".", "#"],
            ["#", "#", "#", "#", "#"],
        ]
        char_to_map_name: dict[str, str] = {
            "#": "wall",
            "@": "agent.agent",
            ".": "empty",
            "S": station_map_name,
        }

        available_positions = [(3, 2), (1, 3), (2, 3)]
        if extra_objects:
            for i, obj in enumerate(extra_objects):
                obj_name = obj.map_name or obj.name
                objects[obj_name] = obj
                char = chr(ord("A") + i)
                r, c = available_positions[i]
                map_data[r][c] = char
                char_to_map_name[char] = obj_name

        agent_cfg = AgentConfig(
            tags=[f"team:{agent_team}"] if agent_team else [],
            inventory=InventoryConfig(
                initial=agent_inventory or {},
                limits=resource_limits(),
            ),
        )

        resource_names = RESOURCES + (extra_resources or [])

        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                resource_names=resource_names,
                territories=territories or {"team_territory": TerritoryConfig(tag_prefix="team:")},
                actions=ActionsConfig(
                    noop=NoopActionConfig(),
                    move=MoveActionConfig(),
                ),
                agent=agent_cfg,
                objects=objects,
                tags=tags or [],
                materialize_queries=materialize_queries or [],
                map_builder=AsciiMapBuilder.Config(
                    map_data=map_data,
                    char_to_map_name=char_to_map_name,
                ),
            )
        )

        sim = Simulation(cfg, seed=42)

        if agent_inventory:
            sim.agent(0).set_inventory(agent_inventory)

        return cls(simulation=sim)

    def move_onto_station(self) -> bool:
        """Execute move_east to move onto station. Returns action success."""
        self.simulation.agent(self.agent_id).set_action("move_east")
        self.simulation.step()
        return self.simulation.agent(self.agent_id).last_action_success

    def step(self, n: int = 1) -> None:
        """Step simulation n times with noop action."""
        for _ in range(n):
            self.simulation.agent(self.agent_id).set_action("noop")
            self.simulation.step()

    def agent_inventory(self, agent_id: int = 0) -> dict[str, int]:
        """Get agent's current inventory."""
        return self.simulation.agent(agent_id).inventory

    def agent_has_gear(self, gear_type: str, agent_id: int = 0) -> bool:
        """Check if agent has specific gear."""
        return self.agent_inventory(agent_id).get(gear_type, 0) >= 1

    def station_exists(self, station_name: str = "station") -> bool:
        """Check if station still exists on map."""
        for obj in self.simulation.grid_objects().values():
            if station_name in obj.get("type_name", ""):
                return True
        return False

    def object_inventory(self, type_name: str) -> dict[str, int]:
        """Get a grid object's inventory by its type_name."""
        resource_names = self.simulation.resource_names
        for obj in self.simulation.grid_objects().values():
            if obj.get("type_name") == type_name:
                raw_inv = obj.get("inventory", {})
                return {resource_names[idx]: amount for idx, amount in raw_inv.items() if amount != 0}
        return {}

    def close(self) -> None:
        """Close the simulation."""
        self.simulation.close()


def make_junction_sim(
    teams: list[TeamConfig],
    map_data: list[list[str]],
    char_to_map_name: dict[str, str],
    agent_inventory: dict[str, int],
    agent_team: str,
) -> Simulation:
    """Create a simulation with hub(s), junction(s), and materialized queries for alignment tests."""
    objects: dict[str, Any] = {"wall": WallConfig()}

    for t in teams:
        objects[f"{t.short_name}:hub"] = GridObjectConfig(
            name="hub",
            map_name=f"{t.short_name}:hub",
            tags=[t.team_tag()],
        )

    junction = GridObjectConfig(name="junction")
    tj = TeamJunctionVariant(
        align_cost={"heart": 1},
        scramble_cost={"heart": 1},
        align_required_resources={"aligner": 1},
        scramble_required_resources={"scrambler": 1},
    )

    req_check_align = {**tj.align_required_resources, **tj.align_cost}
    req_check_scramble = {**tj.scramble_required_resources, **tj.scramble_cost}

    junction.on_tag_remove = {
        f"net:{t.name}": Handler(filters=[], mutations=[removeTag(f"team:{t.name}")]) for t in teams
    }
    handlers: list[Handler] = []
    for t in teams:
        align_filters: list = [
            actorHas(req_check_align),
            isNot(hasTagPrefix("team:")),
            anyOf(
                [
                    isNear(query(t.net_tag()), radius=JUNCTION_ALIGN_DISTANCE),
                    isNear(query(typeTag("hub"), hasTag(t.team_tag())), radius=HUB_ALIGN_DISTANCE),
                ]
            ),
        ]
        align_mutations: list = [
            updateActor({k: -v for k, v in tj.align_cost.items()}),
            logActorAgentStat("junction.aligned_by_agent"),
            logStatToGame(f"{t.name}/aligned.junction.gained"),
            addTag(t.team_tag()),
            addTag(t.net_tag()),
            recomputeMaterializedQuery(t.net_tag()),
        ]
        handlers.append(
            Handler(
                name=f"align_{t.name}",
                filters=align_filters,
                mutations=align_mutations,
            )
        )

        scramble_filters: list = [
            hasTag(t.team_tag()),
            isNot(sharedTagPrefix("team:")),
            actorHas(req_check_scramble),
        ]
        scramble_mutations: list = [
            updateActor({k: -v for k, v in tj.scramble_cost.items()}),
            removeTagPrefix("net:"),
            logActorAgentStat("junction.scrambled_by_agent"),
            logStatToGame(f"{t.name}/aligned.junction.lost"),
            recomputeMaterializedQuery("net:"),
        ]
        handlers.append(
            Handler(
                name=f"scramble_{t.name}",
                filters=scramble_filters,
                mutations=scramble_mutations,
            )
        )
    junction.on_use_handler = firstMatch(handlers)

    objects["junction"] = junction

    all_tags: list[str] = []
    for t in teams:
        all_tags.append(t.team_tag())

    mat_queries: list[MaterializedQuery] = []
    for t in teams:
        mat_queries.append(net_materialized_query(t))

    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=100,
            resource_names=RESOURCES,
            territories={"team_territory": TerritoryConfig(tag_prefix="team:")},
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(),
            ),
            agent=AgentConfig(
                tags=[f"team:{agent_team}"],
                inventory=InventoryConfig(
                    initial=agent_inventory,
                    limits=resource_limits(),
                ),
            ),
            objects=objects,
            tags=all_tags,
            materialize_queries=mat_queries,
            map_builder=AsciiMapBuilder.Config(
                map_data=map_data,
                char_to_map_name=char_to_map_name,
            ),
        )
    )

    sim = Simulation(cfg, seed=42)
    sim.agent(0).set_inventory(agent_inventory)
    return sim
