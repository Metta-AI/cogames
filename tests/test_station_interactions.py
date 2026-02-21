"""Integration tests for cogs_vs_clips station interactions.

Tests station mechanics using real MettaGrid environments with minimal setups.
Each test creates a small environment with one agent and one station to verify
specific interaction behaviors.
"""

from dataclasses import dataclass
from typing import Any

from cogames.cogs_vs_clips.cog import CogTeam
from cogames.cogs_vs_clips.config import CvCConfig
from cogames.cogs_vs_clips.hub import CvCHubConfig
from cogames.cogs_vs_clips.junction import CvCJunctionConfig
from cogames.cogs_vs_clips.stations import (
    CvCChestConfig,
    CvCExtractorConfig,
)
from cogames.cogs_vs_clips.team import TeamConfig
from mettagrid.config.filter import actorHasTag, hasTagPrefix, isNear, isNot
from mettagrid.config.handler_config import Handler, actorHas, updateActor
from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    CollectiveConfig,
    GameConfig,
    GridObjectConfig,
    InventoryConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ResourceLimitsConfig,
    WallConfig,
)
from mettagrid.config.mutation import addTag, recomputeMaterializedQuery, removeTag
from mettagrid.config.query import MaterializedQuery
from mettagrid.config.query import query as make_query
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.simulator import Simulation


def _resource_limits() -> dict[str, ResourceLimitsConfig]:
    """Create resource limits for all CvC resources.

    The 'gear' limit is required by CogTeam.gear_station's ClearInventoryMutation.
    """
    return {
        "all": ResourceLimitsConfig(min=10000, max=10000, resources=CvCConfig.RESOURCES),
        "gear": ResourceLimitsConfig(min=1, max=1, resources=CvCConfig.GEAR),
    }


def _team_tags(*teams: TeamConfig) -> list[str]:
    """Collect all tags from the given teams."""
    result: list[str] = []
    for t in teams:
        result.extend(t.all_tags())
    return result


def _hub_object(team: str, initial: dict[str, int]) -> GridObjectConfig:
    """Create a hub GridObjectConfig with the given team tag and initial inventory."""
    return GridObjectConfig(
        name=f"{team}:hub",
        tags=[f"team:{team}"],
        collective=team,
        inventory=InventoryConfig(initial=initial, limits=_resource_limits()),
    )


def _count_junctions_by_team(sim: Simulation, team_tag: str) -> int:
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
        collective_initial: dict[str, int] | None = None,
        materialize_queries: list[MaterializedQuery] | None = None,
        extra_resources: list[str] | None = None,
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

        collective_names: set[str] = set()
        if agent_team:
            collective_names.add(agent_team)
        if station.collective:
            collective_names.add(station.collective)
        if extra_objects:
            for obj in extra_objects:
                if obj.collective:
                    collective_names.add(obj.collective)

        collectives = {
            name: CollectiveConfig(
                name=name,
                inventory=InventoryConfig(
                    initial=collective_initial or {},
                    limits=_resource_limits(),
                ),
            )
            for name in collective_names
        }

        agent_cfg = AgentConfig(
            tags=[f"team:{agent_team}"] if agent_team else [],
            collective=agent_team,
            inventory=InventoryConfig(
                initial=agent_inventory or {},
                limits=_resource_limits(),
            ),
        )

        resource_names = CvCConfig.RESOURCES + (extra_resources or [])

        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                resource_names=resource_names,
                actions=ActionsConfig(
                    noop=NoopActionConfig(),
                    move=MoveActionConfig(),
                ),
                agent=agent_cfg,
                collectives=collectives,
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

    def collective_inventory(self, name: str) -> dict[str, int]:
        """Get collective's resource pool."""
        inventories = self.simulation._c_sim.get_collective_inventories()
        return inventories.get(name, {})

    def close(self) -> None:
        """Close the simulation."""
        self.simulation.close()


class TestExtractor:
    """Test CvCExtractorConfig station interactions."""

    def test_extract_without_gear(self):
        """Extracting without miner gear yields small amount."""
        station = CvCExtractorConfig(
            resource="oxygen",
            initial_amount=100,
            small_amount=1,
            large_amount=10,
        ).station_cfg()

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"oxygen": 0},
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        assert inv.get("oxygen", 0) == 1, f"Expected 1 oxygen, got {inv.get('oxygen', 0)}"

        harness.close()

    def test_extract_with_miner_gear(self):
        """Extracting with miner gear yields large amount."""
        station = CvCExtractorConfig(
            resource="oxygen",
            initial_amount=100,
            small_amount=1,
            large_amount=10,
        ).station_cfg()

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"oxygen": 0, "miner": 1},
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        assert inv.get("oxygen", 0) == 10, f"Expected 10 oxygen with miner, got {inv.get('oxygen', 0)}"

        harness.close()

    def test_extractor_depletion(self):
        """Extractor is removed when depleted."""
        station = CvCExtractorConfig(
            resource="oxygen",
            initial_amount=10,
            small_amount=1,
            large_amount=10,
        ).station_cfg()

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"oxygen": 0, "miner": 1},
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        assert inv.get("oxygen", 0) == 10, f"Expected 10 oxygen, got {inv.get('oxygen', 0)}"

        assert not harness.station_exists("oxygen_extractor"), "Extractor should be removed when depleted"

        harness.close()

    def test_extract_limited_by_inventory(self):
        """Extract amount is limited by extractor's remaining inventory."""
        station = CvCExtractorConfig(
            resource="oxygen",
            initial_amount=3,
            small_amount=1,
            large_amount=10,
        ).station_cfg()

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"oxygen": 0, "miner": 1},
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        assert inv.get("oxygen", 0) == 3, f"Expected 3 oxygen (limited by inventory), got {inv.get('oxygen', 0)}"

        harness.close()


class TestHub:
    """Test CvCHubConfig station interactions."""

    def test_deposit_resources(self):
        """Aligned agent deposits resources to hub collective."""
        station = CvCHubConfig().station_cfg(team=TeamConfig(name="cogs", short_name="c"))

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"oxygen": 50},
            agent_team="cogs",
            tags=["team:cogs"],
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        collective_inv = harness.collective_inventory("cogs")

        assert inv.get("oxygen", 0) == 0, f"Agent should have 0 oxygen after deposit, got {inv.get('oxygen', 0)}"
        assert collective_inv.get("oxygen", 0) == 50, (
            f"Collective should have 50 oxygen, got {collective_inv.get('oxygen', 0)}"
        )

        harness.close()

    def test_deposit_requires_team_tag(self):
        """Agent without team tag cannot deposit to hub."""
        station = CvCHubConfig().station_cfg(team=TeamConfig(name="cogs", short_name="c"))

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"oxygen": 50},
            agent_team=None,
            tags=["team:cogs"],
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        collective_inv = harness.collective_inventory("cogs")

        assert inv.get("oxygen", 0) == 50, f"Unaligned agent should keep oxygen, got {inv.get('oxygen', 0)}"
        assert collective_inv.get("oxygen", 0) == 0, (
            f"Collective should have 0 oxygen (no deposit), got {collective_inv.get('oxygen', 0)}"
        )

        harness.close()

    def test_hub_heal_aoe_heals_aligned(self):
        """Hub AOE heals aligned agents."""
        team = TeamConfig(name="cogs", short_name="c", base_aoe_deltas={"hp": 10, "energy": 10})
        station = CvCHubConfig().station_cfg(team=team)

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"hp": 50, "energy": 0},
            agent_team="cogs",
            tags=["team:cogs"],
        )

        harness.step(5)

        inv = harness.agent_inventory()
        assert inv.get("hp", 0) >= 100, f"Expected hp >= 100 after AOE healing, got {inv.get('hp', 0)}"
        assert inv.get("energy", 0) >= 50, f"Expected energy >= 50 after AOE, got {inv.get('energy', 0)}"

        harness.close()


class TestJunction:
    """Test CvCJunctionConfig station interactions."""

    def test_scramble_enemy_junction(self):
        """Agent with heart and scrambler can scramble enemy junction via tag removal."""
        station = GridObjectConfig(
            name="junction",
            render_symbol="📦",
            tags=["team:clips"],
            on_use_handlers={
                "scramble": Handler(
                    filters=[
                        hasTagPrefix("team:"),
                        actorHas({"scrambler": 1, **CvCConfig.SCRAMBLE_COST}),
                    ],
                    mutations=[
                        removeTag("team:cogs"),
                        removeTag("team:clips"),
                        updateActor({k: -v for k, v in CvCConfig.SCRAMBLE_COST.items()}),
                    ],
                ),
            },
        )

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"heart": 5, "scrambler": 1},
            agent_team="cogs",
            tags=["team:cogs", "team:clips"],
        )

        before = _count_junctions_by_team(harness.simulation, "team:clips")
        assert before == 1, f"Junction should start with team:clips, count={before}"

        harness.move_onto_station()

        after = _count_junctions_by_team(harness.simulation, "team:clips")
        assert after == 0, f"Junction should be scrambled (team:clips removed), count={after}"

        inv = harness.agent_inventory()
        assert inv.get("heart", 0) == 4, f"Expected 4 hearts after scramble, got {inv.get('heart', 0)}"

        harness.close()


class TestGearStation:
    """Test CogTeam.gear_station() interactions."""

    def test_change_gear_costs_resources(self):
        """Agent pays collective resources to get gear."""
        cogs = CogTeam()
        station = cogs.gear_station("miner")

        miner_cost = CvCConfig.GEAR_COSTS["miner"]
        initial_collective = {k: v * 10 for k, v in miner_cost.items()}

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={},
            agent_team="cogs",
            tags=cogs.all_tags(),
            extra_objects=[_hub_object("cogs", {})],
            collective_initial=initial_collective,
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        assert inv.get("miner", 0) == 1, f"Expected miner=1, got {inv.get('miner', 0)}"

        harness.close()

    def test_keep_gear_no_cost(self):
        """Agent with matching gear keeps it without paying."""
        cogs = CogTeam()
        station = cogs.gear_station("miner")

        miner_cost = CvCConfig.GEAR_COSTS["miner"]
        initial_collective = {k: 100 for k in miner_cost}

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"miner": 1},
            agent_team="cogs",
            tags=cogs.all_tags(),
            extra_objects=[_hub_object("cogs", {})],
            collective_initial=initial_collective,
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        collective_inv = harness.collective_inventory("cogs")

        assert inv.get("miner", 0) == 1, f"Should still have miner, got {inv.get('miner', 0)}"
        for resource, initial in initial_collective.items():
            assert collective_inv.get(resource, 0) == initial, (
                f"Collective {resource} should be unchanged at {initial}, got {collective_inv.get(resource, 0)}"
            )

        harness.close()

    def test_change_gear_clears_old_gear(self):
        """Getting new gear clears previous gear."""
        cogs = CogTeam()
        station = cogs.gear_station("miner")

        miner_cost = CvCConfig.GEAR_COSTS["miner"]
        initial_collective = {k: 100 for k in miner_cost}

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"scrambler": 1},
            agent_team="cogs",
            tags=cogs.all_tags(),
            extra_objects=[_hub_object("cogs", {})],
            collective_initial=initial_collective,
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        assert inv.get("miner", 0) == 1, f"Should have miner, got {inv.get('miner', 0)}"
        assert inv.get("scrambler", 0) == 0, f"Should have cleared scrambler, got {inv.get('scrambler', 0)}"

        harness.close()

    def test_insufficient_resources_no_change(self):
        """Agent cannot get gear if collective lacks resources."""
        cogs = CogTeam()
        station = cogs.gear_station("miner")

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={},
            agent_team="cogs",
            tags=cogs.all_tags(),
            extra_objects=[_hub_object("cogs", {})],
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        assert inv.get("miner", 0) == 0, f"Should not have miner (no resources), got {inv.get('miner', 0)}"

        harness.close()


class TestChest:
    """Test CvCChestConfig station interactions."""

    def test_get_heart_from_hub(self):
        """Aligned agent gets heart when hub collective has hearts."""
        station = CvCChestConfig().station_cfg(team="c", team_name="cogs")
        hub = _hub_object("cogs", {})

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"heart": 0},
            agent_team="cogs",
            tags=["team:cogs"],
            extra_objects=[hub],
            collective_initial={"heart": 10},
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        collective_inv = harness.collective_inventory("cogs")

        assert inv.get("heart", 0) == 1, f"Expected 1 heart, got {inv.get('heart', 0)}"
        assert collective_inv.get("heart", 0) == 9, (
            f"Collective should have 9 hearts, got {collective_inv.get('heart', 0)}"
        )

        harness.close()

    def test_make_heart_costs_elements(self):
        """Collective with elements can make heart."""
        station = CvCChestConfig().station_cfg(team="c", team_name="cogs")
        hub = _hub_object("cogs", {})

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"heart": 0},
            agent_team="cogs",
            tags=["team:cogs"],
            extra_objects=[hub],
            collective_initial={**CvCConfig.HEART_COST, "heart": 0},
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        assert inv.get("heart", 0) == 1, f"Expected 1 heart from making, got {inv.get('heart', 0)}"

        harness.close()

    def test_make_heart_requires_all_elements(self):
        """Cannot make heart without all required elements."""
        station = CvCChestConfig().station_cfg(team="c", team_name="cogs")
        hub = _hub_object("cogs", {})

        partial_cost = {k: v for k, v in CvCConfig.HEART_COST.items()}
        missing_element = list(partial_cost.keys())[0]
        del partial_cost[missing_element]

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"heart": 0},
            agent_team="cogs",
            tags=["team:cogs"],
            extra_objects=[hub],
            collective_initial={**partial_cost, "heart": 0},
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        assert inv.get("heart", 0) == 0, f"Should not make heart (missing element), got {inv.get('heart', 0)}"

        harness.close()

    def test_get_heart_requires_team_tag(self):
        """Agent without team tag cannot get heart."""
        station = CvCChestConfig().station_cfg(team="c", team_name="cogs")
        hub = _hub_object("cogs", {})

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"heart": 0},
            agent_team=None,
            tags=["team:cogs"],
            extra_objects=[hub],
            collective_initial={"heart": 10},
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        collective_inv = harness.collective_inventory("cogs")

        assert inv.get("heart", 0) == 0, f"Unaligned should not get heart, got {inv.get('heart', 0)}"
        assert collective_inv.get("heart", 0) == 10, (
            f"Collective should still have 10 hearts, got {collective_inv.get('heart', 0)}"
        )

        harness.close()


class TestAOE:
    """Test AOE (Area of Effect) station behaviors."""

    def test_heal_aoe_heals_aligned_agent(self):
        """Hub heal AOE heals agents with same team tag."""
        team = TeamConfig(name="cogs", short_name="c", base_aoe_deltas={"hp": 10, "energy": 10})
        station = CvCHubConfig().station_cfg(team=team)

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"hp": 50, "energy": 0},
            agent_team="cogs",
            tags=["team:cogs"],
            extra_resources=["influence:cogs"],
        )

        harness.step(5)

        inv = harness.agent_inventory()
        assert inv.get("hp", 0) >= 100, f"Aligned agent should be healed, hp={inv.get('hp', 0)}"

        harness.close()

    def test_multiple_aoe_sources_stack(self):
        """Mutating AOE effects stack across multiple overlapping sources."""
        team = TeamConfig(name="cogs", short_name="c", base_aoe_deltas={"hp": 5, "energy": 5})
        station = CvCHubConfig().station_cfg(team=team)

        station_map_name = station.map_name or station.name

        map_data = [
            ["#", "#", "#", "#", "#"],
            ["#", "H", "@", "H", "#"],
            ["#", ".", ".", ".", "#"],
            ["#", ".", ".", ".", "#"],
            ["#", "#", "#", "#", "#"],
        ]

        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                resource_names=CvCConfig.RESOURCES,
                actions=ActionsConfig(
                    noop=NoopActionConfig(),
                    move=MoveActionConfig(),
                ),
                agent=AgentConfig(
                    tags=["team:cogs"],
                    collective="cogs",
                    inventory=InventoryConfig(
                        initial={"hp": 0, "energy": 0},
                        limits=_resource_limits(),
                    ),
                ),
                collectives={
                    "cogs": CollectiveConfig(inventory=InventoryConfig(limits=_resource_limits())),
                },
                tags=["team:cogs"],
                objects={
                    "wall": WallConfig(),
                    station_map_name: station,
                },
                map_builder=AsciiMapBuilder.Config(
                    map_data=map_data,
                    char_to_map_name={
                        "#": "wall",
                        "@": "agent.agent",
                        ".": "empty",
                        "H": station_map_name,
                    },
                ),
            )
        )

        sim = Simulation(cfg, seed=42)
        sim.agent(0).set_inventory({"hp": 0, "energy": 0})

        for _ in range(5):
            sim.agent(0).set_action("noop")
            sim.step()

        inv = sim.agent(0).inventory

        # Weighted territory picks the winning side, but same-side mutating AOEs still stack
        assert inv.get("hp", 0) == 50, f"Expected hp=50 from stacked friendly AOEs, got {inv.get('hp', 0)}"

        sim.close()


def _has_tag_at(sim: Simulation, tag_name: str, row: int, col: int, type_name: str | None = None) -> bool:
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


def _make_junction_sim(
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
            collective=t.name,
        )

    junction = CvCJunctionConfig().station_cfg(teams)
    objects[junction.map_name] = junction

    all_tags: list[str] = []
    for t in teams:
        all_tags.extend(t.all_tags())

    mat_queries: list[MaterializedQuery] = []
    for t in teams:
        mat_queries.extend(t.materialized_queries())

    collectives = {
        t.name: CollectiveConfig(
            name=t.name,
            inventory=InventoryConfig(initial={}, limits=_resource_limits()),
        )
        for t in teams
    }

    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=100,
            resource_names=CvCConfig.RESOURCES,
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(),
            ),
            agent=AgentConfig(
                tags=[f"team:{agent_team}"],
                collective=agent_team,
                inventory=InventoryConfig(
                    initial=agent_inventory,
                    limits=_resource_limits(),
                ),
            ),
            collectives=collectives,
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


class TestJunctionAlignment:
    """Test junction alignment with isNear filter and recomputeMaterializedQuery."""

    def test_align_neutral_junction(self):
        """Agent with aligner+heart aligns a neutral junction near the hub.

        Map layout — junction must be 8-connected to hub so the ClosureQuery
        BFS can reach it after recomputeMaterializedQuery:
            #  #  #  #
            #  H  .  #
            #  @  J  #   ← J diagonal-adjacent to H
            #  .  .  #
            #  #  #  #
        """
        cogs = TeamConfig(name="cogs", short_name="c")
        sim = _make_junction_sim(
            teams=[cogs],
            map_data=[
                ["#", "#", "#", "#"],
                ["#", "H", ".", "#"],
                ["#", "@", "J", "#"],
                ["#", ".", ".", "#"],
                ["#", "#", "#", "#"],
            ],
            char_to_map_name={
                "#": "wall",
                "@": "agent.agent",
                ".": "empty",
                "H": "c:hub",
                "J": "junction",
            },
            agent_inventory={"aligner": 1, "heart": 5, "hp": 100, "energy": 100},
            agent_team="cogs",
        )

        assert _has_tag_at(sim, "net:cogs", 1, 1, "hub"), "Hub should have net:cogs from materialized query at init"

        sim.agent(0).set_action("move_east")
        sim.step()

        inv = sim.agent(0).inventory
        assert inv.get("heart", 0) == 4, f"Expected 4 hearts after alignment, got {inv.get('heart', 0)}"
        assert _has_tag_at(sim, "team:cogs", 2, 2, "junction"), "Junction should have team:cogs after alignment"
        assert _has_tag_at(sim, "net:cogs", 2, 2, "junction"), "Junction should have net:cogs after alignment"

        sim.close()

    def test_align_without_isNear_does_not_crash(self):
        """Alignment without isNear filter works (isolates isNear as crash source)."""
        junction = GridObjectConfig(
            name="junction",
            render_symbol="📦",
            on_use_handlers={
                "align_cogs": Handler(
                    filters=[
                        actorHasTag("team:cogs"),
                        actorHas({"aligner": 1, **CvCConfig.ALIGN_COST}),
                        isNot(hasTagPrefix("team:")),
                    ],
                    mutations=[
                        updateActor({k: -v for k, v in CvCConfig.ALIGN_COST.items()}),
                        addTag("team:cogs"),
                    ],
                ),
            },
        )

        harness = StationTestHarness.create(
            station=junction,
            agent_inventory={"aligner": 1, "heart": 5, "hp": 100, "energy": 100},
            agent_team="cogs",
            tags=["team:cogs"],
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        assert inv.get("heart", 0) == 4, f"Expected 4 hearts, got {inv.get('heart', 0)}"
        harness.close()

    def test_align_with_isNear_filter(self):
        """Alignment with isNear filter against materialized query tag."""
        cogs = TeamConfig(name="cogs", short_name="c")

        junction = GridObjectConfig(
            name="junction",
            render_symbol="📦",
            on_use_handlers={
                "align_cogs": Handler(
                    filters=[
                        actorHasTag("team:cogs"),
                        actorHas({"aligner": 1, **CvCConfig.ALIGN_COST}),
                        isNot(hasTagPrefix("team:")),
                        isNear(make_query(cogs.net_tag()), radius=CvCConfig.JUNCTION_DISTANCE),
                    ],
                    mutations=[
                        updateActor({k: -v for k, v in CvCConfig.ALIGN_COST.items()}),
                        addTag("team:cogs"),
                        recomputeMaterializedQuery(cogs.net_tag()),
                    ],
                ),
            },
        )

        hub = GridObjectConfig(
            name="hub",
            map_name=f"{cogs.short_name}:hub",
            tags=[cogs.team_tag()],
            collective=cogs.name,
        )

        harness = StationTestHarness.create(
            station=junction,
            agent_inventory={"aligner": 1, "heart": 5, "hp": 100, "energy": 100},
            agent_team="cogs",
            tags=cogs.all_tags(),
            extra_objects=[hub],
            materialize_queries=cogs.materialized_queries(),
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        assert inv.get("heart", 0) == 4, f"Expected 4 hearts, got {inv.get('heart', 0)}"
        harness.close()

    def test_alignment_updates_network_tag(self):
        """After alignment, junction receives the materialized net tag via recomputeMaterializedQuery."""
        cogs = TeamConfig(name="cogs", short_name="c")

        junction = GridObjectConfig(
            name="junction",
            render_symbol="📦",
            on_use_handlers={
                "align_cogs": Handler(
                    filters=[
                        actorHasTag("team:cogs"),
                        actorHas({"aligner": 1, **CvCConfig.ALIGN_COST}),
                        isNot(hasTagPrefix("team:")),
                        isNear(make_query(cogs.net_tag()), radius=CvCConfig.JUNCTION_DISTANCE),
                    ],
                    mutations=[
                        updateActor({k: -v for k, v in CvCConfig.ALIGN_COST.items()}),
                        addTag("team:cogs"),
                        recomputeMaterializedQuery(cogs.net_tag()),
                    ],
                ),
            },
        )

        hub = GridObjectConfig(
            name="hub",
            map_name=f"{cogs.short_name}:hub",
            tags=[cogs.team_tag()],
            collective=cogs.name,
        )

        harness = StationTestHarness.create(
            station=junction,
            agent_inventory={"aligner": 1, "heart": 5, "hp": 100, "energy": 100},
            agent_team="cogs",
            tags=cogs.all_tags(),
            extra_objects=[hub],
            materialize_queries=cogs.materialized_queries(),
        )

        harness.move_onto_station()

        assert _has_tag_at(harness.simulation, "net:cogs", 2, 2, "junction"), (
            "Junction should have net:cogs after alignment and recomputeMaterializedQuery"
        )

        harness.close()

    def test_scramble_removes_team_tag_via_on_tag_remove(self):
        """Scrambling a junction via removeTagPrefix('net:') cascades through on_tag_remove to remove team: tag.

        Uses the real CvCJunctionConfig (which uses removeTagPrefix('net:') + on_tag_remove cascade)
        rather than hardcoded removeTag calls.
        """
        cogs = TeamConfig(name="cogs", short_name="c")
        clips = TeamConfig(name="clips", short_name="clips")
        sim = _make_junction_sim(
            teams=[cogs, clips],
            map_data=[
                ["#", "#", "#", "#", "#"],
                ["#", "H", ".", ".", "#"],
                ["#", "J", "@", ".", "#"],
                ["#", ".", ".", ".", "#"],
                ["#", "#", "#", "#", "#"],
            ],
            char_to_map_name={
                "#": "wall",
                "@": "agent.agent",
                ".": "empty",
                "H": "clips:hub",
                "J": "junction",
            },
            agent_inventory={"scrambler": 1, "heart": 5, "hp": 100, "energy": 100},
            agent_team="cogs",
        )

        # Align the junction to clips first (via direct tag manipulation for setup)
        # The junction starts neutral; let's use the clips align event approach.
        # For simplicity, verify the hub gets net:clips from materialized query.
        assert _has_tag_at(sim, "net:clips", 1, 1, "hub"), "Hub should have net:clips"

        # Align junction to clips: move agent west onto junction, then manually set tags
        # Actually, let's align it by having a clips agent. Instead, let's create a pre-aligned junction.
        # Re-create with junction starting as clips-owned.
        sim.close()

        sim = _make_junction_sim(
            teams=[cogs, clips],
            map_data=[
                ["#", "#", "#", "#", "#"],
                ["#", "H", ".", ".", "#"],
                ["#", "J", "@", ".", "#"],
                ["#", ".", ".", ".", "#"],
                ["#", "#", "#", "#", "#"],
            ],
            char_to_map_name={
                "#": "wall",
                "@": "agent.agent",
                ".": "empty",
                "H": "clips:hub",
                "J": "junction",
            },
            agent_inventory={"scrambler": 1, "heart": 5, "hp": 100, "energy": 100},
            agent_team="cogs",
        )

        # The junction starts neutral. We need it to be clips-owned.
        # Use CvCJunctionConfig with owner_team to create a clips-owned junction instead.
        sim.close()

        # Build manually with a clips-owned junction
        junction_cfg = CvCJunctionConfig().station_cfg(teams=[cogs, clips], owner_team_name=clips.name)
        objects: dict[str, Any] = {"wall": WallConfig()}
        objects["clips:hub"] = GridObjectConfig(
            name="hub",
            map_name="clips:hub",
            tags=[clips.team_tag()],
            collective=clips.name,
        )
        objects[junction_cfg.map_name] = junction_cfg

        all_tags: list[str] = []
        for t in [cogs, clips]:
            all_tags.extend(t.all_tags())

        mat_queries: list[MaterializedQuery] = []
        for t in [cogs, clips]:
            mat_queries.extend(t.materialized_queries())

        collectives = {
            t.name: CollectiveConfig(
                name=t.name,
                inventory=InventoryConfig(initial={}, limits=_resource_limits()),
            )
            for t in [cogs, clips]
        }

        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                resource_names=CvCConfig.RESOURCES,
                actions=ActionsConfig(
                    noop=NoopActionConfig(),
                    move=MoveActionConfig(),
                ),
                agent=AgentConfig(
                    tags=["team:cogs"],
                    collective="cogs",
                    inventory=InventoryConfig(
                        initial={"scrambler": 1, "heart": 5, "hp": 100, "energy": 100},
                        limits=_resource_limits(),
                    ),
                ),
                collectives=collectives,
                objects=objects,
                tags=all_tags,
                materialize_queries=mat_queries,
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        ["#", "#", "#", "#", "#"],
                        ["#", "H", ".", ".", "#"],
                        ["#", "J", "@", ".", "#"],
                        ["#", ".", ".", ".", "#"],
                        ["#", "#", "#", "#", "#"],
                    ],
                    char_to_map_name={
                        "#": "wall",
                        "@": "agent.agent",
                        ".": "empty",
                        "H": "clips:hub",
                        "J": "junction",
                    },
                ),
            )
        )

        sim = Simulation(cfg, seed=42)
        sim.agent(0).set_inventory({"scrambler": 1, "heart": 5, "hp": 100, "energy": 100})

        # Junction should start with team:clips and net:clips
        assert _has_tag_at(sim, "team:clips", 2, 1, "junction"), "Junction should start with team:clips"
        assert _has_tag_at(sim, "net:clips", 2, 1, "junction"), "Junction should start with net:clips"

        # Agent scrambles junction by moving west
        sim.agent(0).set_action("move_west")
        sim.step()

        # After scramble: team:clips and net:clips should both be removed
        assert not _has_tag_at(sim, "net:clips", 2, 1, "junction"), "Junction should not have net:clips after scramble"
        assert not _has_tag_at(sim, "team:clips", 2, 1, "junction"), (
            "Junction should not have team:clips after scramble (via on_tag_remove cascade)"
        )

        sim.close()

    def test_chain_alignment_preserves_first_junction(self):
        """Aligning a second junction must not break the first junction's tags.

        This catches the on_tag_remove cascade bug: recomputeMaterializedQuery clears
        net:cogs from J1 → on_tag_remove fires → team:cogs removed from J1 → BFS
        can't find J1 as a bridge → network collapses.

        Map layout — junctions are vertically adjacent to hub so BFS can chain:
            #  #  #  #  #
            #  H  .  .  #   H at (1,1)
            #  J  @  .  #   J1 at (2,1), agent at (2,2)
            #  J  .  .  #   J2 at (3,1)
            #  #  #  #  #
        """
        cogs = TeamConfig(name="cogs", short_name="c")
        sim = _make_junction_sim(
            teams=[cogs],
            map_data=[
                ["#", "#", "#", "#", "#"],
                ["#", "H", ".", ".", "#"],
                ["#", "J", "@", ".", "#"],
                ["#", "J", ".", ".", "#"],
                ["#", "#", "#", "#", "#"],
            ],
            char_to_map_name={
                "#": "wall",
                "@": "agent.agent",
                ".": "empty",
                "H": "c:hub",
                "J": "junction",
            },
            agent_inventory={"aligner": 1, "heart": 10, "hp": 100, "energy": 100},
            agent_team="cogs",
        )

        # Agent at (2,2) aligns J1 at (2,1) by moving west
        sim.agent(0).set_action("move_west")
        sim.step()

        assert _has_tag_at(sim, "team:cogs", 2, 1, "junction"), "J1 should have team:cogs"
        assert _has_tag_at(sim, "net:cogs", 2, 1, "junction"), "J1 should have net:cogs"

        # Move south to (3,2), then align J2 at (3,1) by moving west
        sim.agent(0).set_action("move_south")
        sim.step()
        sim.agent(0).set_action("move_west")
        sim.step()

        # J2 should be aligned (it's adjacent to J1 which is in the network)
        assert _has_tag_at(sim, "team:cogs", 3, 1, "junction"), "J2 should have team:cogs"
        assert _has_tag_at(sim, "net:cogs", 3, 1, "junction"), "J2 should have net:cogs"

        # J1 must STILL have both tags after J2's alignment triggered recompute
        assert _has_tag_at(sim, "team:cogs", 2, 1, "junction"), (
            "J1 should still have team:cogs after J2 alignment (on_tag_remove cascade bug)"
        )
        assert _has_tag_at(sim, "net:cogs", 2, 1, "junction"), "J1 should still have net:cogs after J2 alignment"
