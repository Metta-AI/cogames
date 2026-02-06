"""Integration tests for cogs_vs_clips station interactions.

Tests station mechanics using real MettaGrid environments with minimal setups.
Each test creates a small environment with one agent and one station to verify
specific interaction behaviors.
"""

from dataclasses import dataclass
from typing import Any

from cogames.cogs_vs_clips.config import CvCConfig
from cogames.cogs_vs_clips.stations import (
    CvCChestConfig,
    CvCExtractorConfig,
    CvCGearStationConfig,
    CvCHubConfig,
    CvCJunctionConfig,
)
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
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.simulator import Simulation


def _resource_limits() -> dict[str, ResourceLimitsConfig]:
    """Create resource limits for all CvC resources.

    The 'gear' limit is required by CvCGearStationConfig's ClearInventoryMutation.
    """
    return {
        "all": ResourceLimitsConfig(min=10000, max=10000, resources=CvCConfig.RESOURCES),
        "gear": ResourceLimitsConfig(min=1, max=1, resources=CvCConfig.GEAR),
    }


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
        agent_collective: str | None = None,
        collectives: dict[str, CollectiveConfig] | None = None,
        extra_objects: dict[str, GridObjectConfig] | None = None,
    ) -> "StationTestHarness":
        """Create a 5x5 map with agent at (1,2) and station at (2,2).

        Agent moves east to interact with station.
        """
        # Default collectives for cogs_vs_clips
        if collectives is None:
            collectives = {
                "cogs": CollectiveConfig(
                    inventory=InventoryConfig(
                        initial={},
                        limits=_resource_limits(),
                    )
                ),
                "clips": CollectiveConfig(
                    inventory=InventoryConfig(
                        initial={},
                        limits=_resource_limits(),
                    )
                ),
            }

        # Use station's map_name for consistent references
        station_map_name = station.map_name or station.name

        # Build objects dict - use map_name as key
        objects: dict[str, Any] = {
            "wall": WallConfig(),
            station_map_name: station,
        }
        if extra_objects:
            objects.update(extra_objects)

        # Agent config
        agent_cfg = AgentConfig(
            collective=agent_collective,
            inventory=InventoryConfig(
                initial=agent_inventory or {},
                limits=_resource_limits(),
            ),
        )

        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=100,
                resource_names=CvCConfig.RESOURCES,
                actions=ActionsConfig(
                    noop=NoopActionConfig(),
                    move=MoveActionConfig(),
                ),
                agent=agent_cfg,
                collectives=collectives,
                objects=objects,
                map_builder=AsciiMapBuilder.Config(
                    map_data=[
                        ["#", "#", "#", "#", "#"],
                        ["#", ".", ".", ".", "#"],
                        ["#", "@", "S", ".", "#"],
                        ["#", ".", ".", ".", "#"],
                        ["#", "#", "#", "#", "#"],
                    ],
                    char_to_map_name={
                        "#": "wall",
                        "@": "agent.agent",
                        ".": "empty",
                        "S": station_map_name,
                    },
                ),
            )
        )

        sim = Simulation(cfg, seed=42)

        # Apply initial inventory if specified (after simulation creation)
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
        grid_objects = self.simulation.grid_objects()
        for obj in grid_objects.values():
            if station_name in obj.get("type", ""):
                return True
        return False

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

        # Move onto extractor
        harness.move_onto_station()

        # Should get small_amount (1)
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

        # Move onto extractor - should extract all 10 and remove it
        harness.move_onto_station()

        inv = harness.agent_inventory()
        assert inv.get("oxygen", 0) == 10, f"Expected 10 oxygen, got {inv.get('oxygen', 0)}"

        # Check extractor is removed
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

        # With miner wants 10, but only 3 available
        inv = harness.agent_inventory()
        assert inv.get("oxygen", 0) == 3, f"Expected 3 oxygen (limited by inventory), got {inv.get('oxygen', 0)}"

        harness.close()


class TestHub:
    """Test CvCHubConfig station interactions."""

    def test_deposit_resources(self):
        """Aligned agent deposits resources to collective."""
        station = CvCHubConfig().station_cfg(team="cogs", collective="cogs")

        collectives = {
            "cogs": CollectiveConfig(
                inventory=InventoryConfig(
                    initial={"oxygen": 0},
                    limits=_resource_limits(),
                )
            ),
        }

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"oxygen": 50},
            agent_collective="cogs",
            collectives=collectives,
        )

        harness.move_onto_station()

        # Agent should have deposited oxygen
        inv = harness.agent_inventory()
        collective_inv = harness.collective_inventory("cogs")

        assert inv.get("oxygen", 0) == 0, f"Agent should have 0 oxygen after deposit, got {inv.get('oxygen', 0)}"
        assert collective_inv.get("oxygen", 0) == 50, (
            f"Collective should have 50 oxygen, got {collective_inv.get('oxygen', 0)}"
        )

        harness.close()

    def test_deposit_requires_alignment(self):
        """Unaligned agent cannot deposit to hub."""
        station = CvCHubConfig().station_cfg(team="cogs", collective="cogs")

        collectives = {
            "cogs": CollectiveConfig(
                inventory=InventoryConfig(
                    initial={"oxygen": 0},
                    limits=_resource_limits(),
                )
            ),
        }

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"oxygen": 50},
            agent_collective=None,  # Not aligned
            collectives=collectives,
        )

        harness.move_onto_station()

        # Agent should still have oxygen (no deposit)
        inv = harness.agent_inventory()
        collective_inv = harness.collective_inventory("cogs")

        assert inv.get("oxygen", 0) == 50, f"Unaligned agent should keep oxygen, got {inv.get('oxygen', 0)}"
        assert collective_inv.get("oxygen", 0) == 0, (
            f"Collective should have 0 oxygen (no deposit), got {collective_inv.get('oxygen', 0)}"
        )

        harness.close()

    def test_hub_influence_aoe_heals_aligned(self):
        """Hub AOE heals aligned agents."""
        station = CvCHubConfig(
            aoe_range=5,
            influence_deltas={"hp": 10, "influence": 10, "energy": 10},
        ).station_cfg(team="cogs", collective="cogs")

        collectives = {
            "cogs": CollectiveConfig(inventory=InventoryConfig(limits=_resource_limits())),
        }

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"hp": 50, "influence": 0, "energy": 0},
            agent_collective="cogs",
            collectives=collectives,
        )

        # Step without moving - agent is in AOE range
        harness.step(5)

        inv = harness.agent_inventory()
        # After 5 ticks, should have gained 50 hp, influence, energy
        assert inv.get("hp", 0) >= 100, f"Expected hp >= 100 after AOE healing, got {inv.get('hp', 0)}"
        assert inv.get("influence", 0) >= 50, f"Expected influence >= 50 after AOE, got {inv.get('influence', 0)}"

        harness.close()

    def test_hub_attack_aoe_damages_enemies(self):
        """Hub AOE damages enemy agents."""
        station = CvCHubConfig(
            aoe_range=5,
            attack_deltas={"hp": -10},
        ).station_cfg(team="cogs", collective="cogs")

        collectives = {
            "cogs": CollectiveConfig(inventory=InventoryConfig(limits=_resource_limits())),
            "clips": CollectiveConfig(inventory=InventoryConfig(limits=_resource_limits())),
        }

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"hp": 100},
            agent_collective="clips",  # Enemy of cogs
            collectives=collectives,
        )

        # Step - agent is enemy and in range
        harness.step(5)

        inv = harness.agent_inventory()
        # After 5 ticks, should have lost 50 hp
        assert inv.get("hp", 0) <= 50, f"Expected hp <= 50 after AOE damage, got {inv.get('hp', 0)}"

        harness.close()


class TestJunction:
    """Test CvCJunctionConfig station interactions."""

    def test_align_neutral_junction(self):
        """Agent with heart and aligner can align neutral junction."""
        station = CvCJunctionConfig().station_cfg(team=None)  # Neutral

        collectives = {
            "cogs": CollectiveConfig(inventory=InventoryConfig(limits=_resource_limits())),
        }

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"heart": 5, "aligner": 1, "influence": 10},
            agent_collective="cogs",
            collectives=collectives,
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        # Should have spent 1 heart
        assert inv.get("heart", 0) == 4, f"Expected 4 hearts after alignment, got {inv.get('heart', 0)}"

        harness.close()

    def test_align_requires_heart_and_gear(self):
        """Agent without aligner cannot align junction."""
        station = CvCJunctionConfig().station_cfg(team=None)

        collectives = {
            "cogs": CollectiveConfig(inventory=InventoryConfig(limits=_resource_limits())),
        }

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"heart": 5, "influence": 10},  # No aligner
            agent_collective="cogs",
            collectives=collectives,
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        # Should still have all hearts (no alignment happened)
        assert inv.get("heart", 0) == 5, f"Expected 5 hearts (no alignment), got {inv.get('heart', 0)}"

        harness.close()

    def test_scramble_enemy_junction(self):
        """Agent with heart and scrambler can scramble enemy junction."""
        station = CvCJunctionConfig().station_cfg(team="clips")  # Enemy junction

        collectives = {
            "cogs": CollectiveConfig(inventory=InventoryConfig(limits=_resource_limits())),
            "clips": CollectiveConfig(inventory=InventoryConfig(limits=_resource_limits())),
        }

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"heart": 5, "scrambler": 1},
            agent_collective="cogs",
            collectives=collectives,
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        # Should have spent 1 heart
        assert inv.get("heart", 0) == 4, f"Expected 4 hearts after scramble, got {inv.get('heart', 0)}"

        harness.close()

    def test_deposit_to_aligned_junction(self):
        """Agent can deposit to junction aligned to same collective."""
        station = CvCJunctionConfig().station_cfg(team="cogs")

        collectives = {
            "cogs": CollectiveConfig(
                inventory=InventoryConfig(
                    initial={"oxygen": 0},
                    limits=_resource_limits(),
                )
            ),
        }

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"oxygen": 30},
            agent_collective="cogs",
            collectives=collectives,
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        collective_inv = harness.collective_inventory("cogs")

        assert inv.get("oxygen", 0) == 0, f"Agent should have deposited oxygen, got {inv.get('oxygen', 0)}"
        assert collective_inv.get("oxygen", 0) == 30, (
            f"Collective should have 30 oxygen, got {collective_inv.get('oxygen', 0)}"
        )

        harness.close()

    def test_cannot_deposit_to_enemy_junction(self):
        """Agent cannot deposit to enemy-aligned junction."""
        station = CvCJunctionConfig().station_cfg(team="clips")

        collectives = {
            "cogs": CollectiveConfig(inventory=InventoryConfig(limits=_resource_limits())),
            "clips": CollectiveConfig(
                inventory=InventoryConfig(
                    initial={"oxygen": 0},
                    limits=_resource_limits(),
                )
            ),
        }

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"oxygen": 30},
            agent_collective="cogs",
            collectives=collectives,
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        clips_inv = harness.collective_inventory("clips")

        assert inv.get("oxygen", 0) == 30, f"Agent should keep oxygen (enemy junction), got {inv.get('oxygen', 0)}"
        assert clips_inv.get("oxygen", 0) == 0, (
            f"Enemy collective should have 0 oxygen, got {clips_inv.get('oxygen', 0)}"
        )

        harness.close()


class TestGearStation:
    """Test CvCGearStationConfig station interactions."""

    def test_change_gear_costs_resources(self):
        """Agent pays collective resources to get gear."""
        station = CvCGearStationConfig(gear_type="miner").station_cfg(team="cogs", collective="cogs")

        miner_cost = CvCConfig.GEAR_COSTS["miner"]
        collectives = {
            "cogs": CollectiveConfig(
                inventory=InventoryConfig(
                    initial={**miner_cost, **{k: v * 10 for k, v in miner_cost.items()}},
                    limits=_resource_limits(),
                )
            ),
        }

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={},
            agent_collective="cogs",
            collectives=collectives,
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        assert inv.get("miner", 0) == 1, f"Expected miner=1, got {inv.get('miner', 0)}"

        harness.close()

    def test_keep_gear_no_cost(self):
        """Agent with matching gear keeps it without paying."""
        station = CvCGearStationConfig(gear_type="miner").station_cfg(team="cogs", collective="cogs")

        miner_cost = CvCConfig.GEAR_COSTS["miner"]
        initial_collective = {k: 100 for k in miner_cost}

        collectives = {
            "cogs": CollectiveConfig(
                inventory=InventoryConfig(
                    initial=initial_collective,
                    limits=_resource_limits(),
                )
            ),
        }

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"miner": 1},
            agent_collective="cogs",
            collectives=collectives,
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        collective_inv = harness.collective_inventory("cogs")

        assert inv.get("miner", 0) == 1, f"Should still have miner, got {inv.get('miner', 0)}"
        # Collective should not have been charged
        for resource, initial in initial_collective.items():
            assert collective_inv.get(resource, 0) == initial, (
                f"Collective {resource} should be unchanged at {initial}, got {collective_inv.get(resource, 0)}"
            )

        harness.close()

    def test_change_gear_clears_old_gear(self):
        """Getting new gear clears previous gear."""
        station = CvCGearStationConfig(gear_type="miner").station_cfg(team="cogs", collective="cogs")

        miner_cost = CvCConfig.GEAR_COSTS["miner"]
        collectives = {
            "cogs": CollectiveConfig(
                inventory=InventoryConfig(
                    initial={k: 100 for k in miner_cost},
                    limits=_resource_limits(),
                )
            ),
        }

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"scrambler": 1},  # Has different gear
            agent_collective="cogs",
            collectives=collectives,
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        assert inv.get("miner", 0) == 1, f"Should have miner, got {inv.get('miner', 0)}"
        assert inv.get("scrambler", 0) == 0, f"Should have cleared scrambler, got {inv.get('scrambler', 0)}"

        harness.close()

    def test_insufficient_resources_no_change(self):
        """Agent cannot get gear if collective lacks resources."""
        station = CvCGearStationConfig(gear_type="miner").station_cfg(team="cogs", collective="cogs")

        collectives = {
            "cogs": CollectiveConfig(
                inventory=InventoryConfig(
                    initial={},  # No resources
                    limits=_resource_limits(),
                )
            ),
        }

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={},
            agent_collective="cogs",
            collectives=collectives,
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        assert inv.get("miner", 0) == 0, f"Should not have miner (no resources), got {inv.get('miner', 0)}"

        harness.close()


class TestChest:
    """Test CvCChestConfig station interactions."""

    def test_get_heart_from_collective(self):
        """Aligned agent gets heart from collective."""
        station = CvCChestConfig().station_cfg(team="cogs", collective="cogs")

        collectives = {
            "cogs": CollectiveConfig(
                inventory=InventoryConfig(
                    initial={"heart": 10},
                    limits=_resource_limits(),
                )
            ),
        }

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"heart": 0},
            agent_collective="cogs",
            collectives=collectives,
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
        station = CvCChestConfig().station_cfg(team="cogs", collective="cogs")

        # Collective needs HEART_COST elements
        collectives = {
            "cogs": CollectiveConfig(
                inventory=InventoryConfig(
                    initial={**CvCConfig.HEART_COST, "heart": 0},
                    limits=_resource_limits(),
                )
            ),
        }

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"heart": 0},
            agent_collective="cogs",
            collectives=collectives,
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        # Should have made a heart
        assert inv.get("heart", 0) == 1, f"Expected 1 heart from making, got {inv.get('heart', 0)}"

        harness.close()

    def test_make_heart_requires_all_elements(self):
        """Cannot make heart without all required elements."""
        station = CvCChestConfig().station_cfg(team="cogs", collective="cogs")

        # Missing one element
        partial_cost = {k: v for k, v in CvCConfig.HEART_COST.items()}
        missing_element = list(partial_cost.keys())[0]
        del partial_cost[missing_element]

        collectives = {
            "cogs": CollectiveConfig(
                inventory=InventoryConfig(
                    initial={**partial_cost, "heart": 0},
                    limits=_resource_limits(),
                )
            ),
        }

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"heart": 0},
            agent_collective="cogs",
            collectives=collectives,
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        assert inv.get("heart", 0) == 0, f"Should not make heart (missing element), got {inv.get('heart', 0)}"

        harness.close()

    def test_get_heart_requires_alignment(self):
        """Unaligned agent cannot get heart."""
        station = CvCChestConfig().station_cfg(team="cogs", collective="cogs")

        collectives = {
            "cogs": CollectiveConfig(
                inventory=InventoryConfig(
                    initial={"heart": 10},
                    limits=_resource_limits(),
                )
            ),
        }

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"heart": 0},
            agent_collective=None,  # Not aligned
            collectives=collectives,
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
    """Test AOE (Area of Effect) station behaviors.

    These tests verify AOE effects using single-agent scenarios since multi-agent
    setup is complex. We test alignment filtering by running separate tests for
    aligned vs enemy agents.
    """

    def test_influence_aoe_heals_aligned_agent(self):
        """Hub influence AOE heals agents aligned to same collective."""
        station = CvCHubConfig(
            aoe_range=5,
            influence_deltas={"hp": 10, "energy": 10, "influence": 10},
        ).station_cfg(team="cogs", collective="cogs")

        collectives = {
            "cogs": CollectiveConfig(inventory=InventoryConfig(limits=_resource_limits())),
        }

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"hp": 50, "energy": 0, "influence": 0},
            agent_collective="cogs",  # Aligned to station
            collectives=collectives,
        )

        harness.step(5)

        inv = harness.agent_inventory()
        # After 5 ticks, should have gained 50 hp
        assert inv.get("hp", 0) >= 100, f"Aligned agent should be healed, hp={inv.get('hp', 0)}"

        harness.close()

    def test_attack_aoe_damages_enemy_agent(self):
        """Hub attack AOE damages agents aligned to different collective."""
        station = CvCHubConfig(
            aoe_range=5,
            influence_deltas={"hp": 0},  # No healing for aligned (not relevant here)
            attack_deltas={"hp": -10},
        ).station_cfg(team="cogs", collective="cogs")

        collectives = {
            "cogs": CollectiveConfig(inventory=InventoryConfig(limits=_resource_limits())),
            "clips": CollectiveConfig(inventory=InventoryConfig(limits=_resource_limits())),
        }

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"hp": 100},
            agent_collective="clips",  # Enemy of cogs station
            collectives=collectives,
        )

        harness.step(5)

        inv = harness.agent_inventory()
        # After 5 ticks of -10 hp, should have 50 hp
        assert inv.get("hp", 0) == 50, f"Enemy agent should lose 50 hp, hp={inv.get('hp', 0)}"

        harness.close()

    def test_attack_aoe_does_not_damage_aligned_agent(self):
        """Hub attack AOE does NOT damage agents aligned to same collective."""
        station = CvCHubConfig(
            aoe_range=5,
            influence_deltas={"hp": 0},  # No healing
            attack_deltas={"hp": -10},
        ).station_cfg(team="cogs", collective="cogs")

        collectives = {
            "cogs": CollectiveConfig(inventory=InventoryConfig(limits=_resource_limits())),
        }

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"hp": 100},
            agent_collective="cogs",  # Same collective - should not be damaged
            collectives=collectives,
        )

        harness.step(5)

        inv = harness.agent_inventory()
        # Aligned agent should not be damaged by attack AOE
        assert inv.get("hp", 0) == 100, f"Aligned agent should not be damaged, hp={inv.get('hp', 0)}"

        harness.close()

    def test_multiple_aoe_sources_stack(self):
        """Effects from multiple AOE sources stack."""
        station = CvCHubConfig(
            aoe_range=5,
            influence_deltas={"hp": 5, "energy": 5, "influence": 5},
        ).station_cfg(team="cogs", collective="cogs")

        station_map_name = station.map_name or station.name

        collectives = {
            "cogs": CollectiveConfig(inventory=InventoryConfig(limits=_resource_limits())),
        }

        # Map with two hubs (both instances of same station type)
        map_data = [
            ["#", "#", "#", "#", "#"],
            ["#", "H", "@", "H", "#"],  # Agent between two hubs
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
                    collective="cogs",
                    inventory=InventoryConfig(
                        initial={"hp": 0, "energy": 0, "influence": 0},
                        limits=_resource_limits(),
                    ),
                ),
                collectives=collectives,
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
        sim.agent(0).set_inventory({"hp": 0, "energy": 0, "influence": 0})

        # Step 5 times
        for _ in range(5):
            sim.agent(0).set_action("noop")
            sim.step()

        inv = sim.agent(0).inventory

        # With 2 hubs, each giving +5 hp per tick, after 5 ticks = 50 hp
        assert inv.get("hp", 0) >= 50, f"Expected hp >= 50 from 2 stacking AOEs, got {inv.get('hp', 0)}"

        sim.close()
