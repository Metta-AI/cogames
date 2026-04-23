"""Tests for gear station interactions: free gear (default) and costed gear."""

from cogsguard.game.teams import TeamConfig
from cogsguard.missions.machina_1 import GEAR_COSTS

from .conftest import StationTestHarness, hub_object


def _make_gear_station(team: TeamConfig, gear_type: str, symbol: str, cost: dict[str, int] | None = None):
    """Build a per-team gear station with handlers attached."""
    from cogsguard.game.teams.gear_stations import TeamGearStationsVariant  # noqa: PLC0415

    from mettagrid.config.mettagrid_config import GameConfig, MettaGridConfig  # noqa: PLC0415

    env = MettaGridConfig(game=GameConfig())
    v = TeamGearStationsVariant()
    v._add_station(env, team, gear_type, symbol, cost)
    return env.game.objects[f"{team.short_name}:{gear_type}"]


class TestGearStationWithCost:
    """Test gear station with explicit gear cost."""

    def test_change_gear_costs_resources(self):
        """Agent pays hub resources to get gear."""
        cogs = TeamConfig()
        station = _make_gear_station(cogs, "miner", "⛏️", GEAR_COSTS["miner"])

        miner_cost = GEAR_COSTS["miner"]
        hub_initial = {k: v * 10 for k, v in miner_cost.items()}

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={},
            agent_team="cogs",
            tags=[cogs.team_tag()],
            extra_objects=[hub_object("cogs", hub_initial)],
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        assert inv.get("miner", 0) == 1, f"Expected miner=1, got {inv.get('miner', 0)}"

        harness.close()

    def test_keep_gear_no_cost(self):
        """Agent with matching gear keeps it without paying."""
        cogs = TeamConfig()
        station = _make_gear_station(cogs, "miner", "⛏️", GEAR_COSTS["miner"])

        miner_cost = GEAR_COSTS["miner"]
        hub_initial = {k: 100 for k in miner_cost}

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"miner": 1},
            agent_team="cogs",
            tags=[cogs.team_tag()],
            extra_objects=[hub_object("cogs", hub_initial)],
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        hub_inv = harness.object_inventory("hub")

        assert inv.get("miner", 0) == 1, f"Should still have miner, got {inv.get('miner', 0)}"
        for resource, initial in hub_initial.items():
            assert hub_inv.get(resource, 0) == initial, (
                f"Hub {resource} should be unchanged at {initial}, got {hub_inv.get(resource, 0)}"
            )

        harness.close()

    def test_change_gear_clears_old_gear(self):
        """Getting new gear clears previous gear."""
        cogs = TeamConfig()
        station = _make_gear_station(cogs, "miner", "⛏️", GEAR_COSTS["miner"])

        miner_cost = GEAR_COSTS["miner"]
        hub_initial = {k: 100 for k in miner_cost}

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"scrambler": 1},
            agent_team="cogs",
            tags=[cogs.team_tag()],
            extra_objects=[hub_object("cogs", hub_initial)],
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        assert inv.get("miner", 0) == 1, f"Should have miner, got {inv.get('miner', 0)}"
        assert inv.get("scrambler", 0) == 0, f"Should have cleared scrambler, got {inv.get('scrambler', 0)}"

        harness.close()

    def test_insufficient_resources_no_change(self):
        """Agent cannot get gear if hub lacks resources."""
        cogs = TeamConfig()
        station = _make_gear_station(cogs, "miner", "⛏️", GEAR_COSTS["miner"])

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={},
            agent_team="cogs",
            tags=[cogs.team_tag()],
            extra_objects=[hub_object("cogs", {})],
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        assert inv.get("miner", 0) == 0, f"Should not have miner (no resources), got {inv.get('miner', 0)}"

        harness.close()


class TestFreeGearStation:
    """Test gear station with no cost (free gear, default)."""

    def test_free_gear_no_hub_needed(self):
        """Agent gets gear for free without needing a hub."""
        cogs = TeamConfig()
        station = _make_gear_station(cogs, "miner", "⛏️")

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={},
            agent_team="cogs",
            tags=[cogs.team_tag()],
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        assert inv.get("miner", 0) == 1, f"Expected free miner=1, got {inv.get('miner', 0)}"

        harness.close()

    def test_free_gear_clears_old_gear(self):
        """Free gear still clears previous gear."""
        cogs = TeamConfig()
        station = _make_gear_station(cogs, "scout", "🔭")

        harness = StationTestHarness.create(
            station=station,
            agent_inventory={"miner": 1},
            agent_team="cogs",
            tags=[cogs.team_tag()],
        )

        harness.move_onto_station()

        inv = harness.agent_inventory()
        assert inv.get("scout", 0) == 1, f"Should have scout, got {inv.get('scout', 0)}"
        assert inv.get("miner", 0) == 0, f"Should have cleared miner, got {inv.get('miner', 0)}"

        harness.close()
