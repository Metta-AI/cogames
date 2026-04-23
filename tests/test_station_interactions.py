"""Integration tests for cogs_vs_clips station interactions.

Tests station mechanics using real MettaGrid environments with minimal setups.
Each test creates a small environment with one agent and one station to verify
specific interaction behaviors.

Variant-specific station tests live in tests/variants/.
"""

from cogsguard.game import GEAR
from cogsguard.game.elements import ElementsVariant
from cogsguard.game.teams import TeamConfig
from cogsguard.game.teams.hub import CvCHubConfig
from variants.conftest import StationTestHarness

from mettagrid.config.filter import sharedTagPrefix
from mettagrid.config.handler_config import Handler, updateTarget
from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    GameConfig,
    InventoryConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ResourceLimitsConfig,
    WallConfig,
)
from mettagrid.config.territory_config import TerritoryConfig, TerritoryControlConfig
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.simulator import Simulation

ELEMENTS = ElementsVariant().elements
RESOURCES = ["energy", "heart", "hp", "solar", *ELEMENTS, *GEAR]


def _resource_limits() -> dict[str, ResourceLimitsConfig]:
    return {
        "all": ResourceLimitsConfig(base=10000, max=10000, resources=RESOURCES),
        "gear": ResourceLimitsConfig(base=1, max=1, resources=GEAR),
    }


class TestHub:
    """Test CvCHubConfig station interactions."""

    def test_deposit_resources(self):
        """Aligned agent deposits resources to hub inventory."""
        station = CvCHubConfig(team=TeamConfig(name="cogs", short_name="c"), elements=["oxygen"]).station_cfg()

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"oxygen": 50},
            agent_team="cogs",
            tags=["team:cogs"],
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        hub_inv = harness.object_inventory("hub")

        assert inv.get("oxygen", 0) == 0, f"Agent should have 0 oxygen after deposit, got {inv.get('oxygen', 0)}"
        assert hub_inv.get("oxygen", 0) == 50, f"Hub should have 50 oxygen, got {hub_inv.get('oxygen', 0)}"

        harness.close()

    def test_deposit_requires_team_tag(self):
        """Agent without team tag cannot deposit to hub."""
        station = CvCHubConfig(team=TeamConfig(name="cogs", short_name="c"), elements=["oxygen"]).station_cfg()

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"oxygen": 50},
            agent_team=None,
            tags=["team:cogs"],
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        hub_inv = harness.object_inventory("hub")

        assert inv.get("oxygen", 0) == 50, f"Unaligned agent should keep oxygen, got {inv.get('oxygen', 0)}"
        assert hub_inv.get("oxygen", 0) == 0, f"Hub should have 0 oxygen (no deposit), got {hub_inv.get('oxygen', 0)}"

        harness.close()

    def test_hub_territory_heals_aligned(self):
        """Friendly territory heals aligned agents via presence handler."""
        team = TeamConfig(name="cogs", short_name="c")
        station = CvCHubConfig(team=team).station_cfg()
        station.territory_controls = [TerritoryControlConfig(territory="team_territory", strength=10)]

        heal_territory = {
            "team_territory": TerritoryConfig(
                tag_prefix="team:",
                presence={
                    "heal": Handler(
                        filters=[sharedTagPrefix("team:")],
                        mutations=[updateTarget({"hp": 10, "energy": 10})],
                    )
                },
            ),
        }

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"hp": 50, "energy": 0},
            agent_team="cogs",
            tags=["team:cogs"],
            territories=heal_territory,
        )

        harness.step(5)

        inv = harness.agent_inventory()
        assert inv.get("hp", 0) >= 100, f"Expected hp >= 100 from territory healing, got {inv.get('hp', 0)}"
        assert inv.get("energy", 0) >= 50, f"Expected energy >= 50 from territory, got {inv.get('energy', 0)}"

        harness.close()


class TestTerritoryHealing:
    """Test territory presence healing behaviors."""

    def test_territory_heals_aligned_agent(self):
        """Friendly territory presence heals agents with same team tag."""
        team = TeamConfig(name="cogs", short_name="c")
        station = CvCHubConfig(team=team).station_cfg()
        station.territory_controls = [TerritoryControlConfig(territory="team_territory", strength=10)]

        heal_territory = {
            "team_territory": TerritoryConfig(
                tag_prefix="team:",
                presence={
                    "heal": Handler(
                        filters=[sharedTagPrefix("team:")],
                        mutations=[updateTarget({"hp": 10, "energy": 10})],
                    )
                },
            ),
        }

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"hp": 50, "energy": 0},
            agent_team="cogs",
            tags=["team:cogs"],
            extra_resources=["influence:cogs"],
            territories=heal_territory,
        )

        harness.step(5)

        inv = harness.agent_inventory()
        assert inv.get("hp", 0) >= 100, f"Aligned agent should be healed, hp={inv.get('hp', 0)}"

        harness.close()

    def test_territory_heal_fires_once_per_tick(self):
        """Territory presence heal fires once per tick regardless of source count."""
        team = TeamConfig(name="cogs", short_name="c")
        station = CvCHubConfig(team=team).station_cfg()
        station.territory_controls = [TerritoryControlConfig(territory="team_territory", strength=10)]

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
                resource_names=RESOURCES,
                territories={
                    "team_territory": TerritoryConfig(
                        tag_prefix="team:",
                        presence={
                            "heal": Handler(
                                filters=[sharedTagPrefix("team:")],
                                mutations=[updateTarget({"hp": 5, "energy": 5})],
                            )
                        },
                    ),
                },
                actions=ActionsConfig(
                    noop=NoopActionConfig(),
                    move=MoveActionConfig(),
                ),
                agent=AgentConfig(
                    tags=["team:cogs"],
                    inventory=InventoryConfig(
                        initial={"hp": 0, "energy": 0},
                        limits=_resource_limits(),
                    ),
                ),
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

        assert inv.get("hp", 0) == 25, f"Expected hp=25 from territory presence, got {inv.get('hp', 0)}"

        sim.close()
