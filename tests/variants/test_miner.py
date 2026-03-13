"""Tests for the miner role variant: extractor bonus and cargo capacity."""

from cogames.games.cogs_vs_clips.game.extractors import CvCExtractorConfig, ExtractorsVariant
from mettagrid.config.handler_config import Handler, actorHas, withdraw

from .conftest import StationTestHarness


def _extractor_with_miner_handler(resource: str, initial_amount: int, small_amount: int = 1, miner_amount: int = 10):
    """Build an extractor station with a miner extraction handler inserted before the default."""
    station = CvCExtractorConfig(
        resource=resource,
        initial_amount=initial_amount,
        small_amount=small_amount,
    ).station_cfg()
    default_extract = station.on_use_handlers.pop("extract")
    station.on_use_handlers["miner_extract"] = Handler(
        filters=[actorHas({"miner": 1})],
        mutations=[withdraw({resource: miner_amount}, remove_when_empty=True)],
    )
    station.on_use_handlers["extract"] = default_extract
    return station


class TestExtractor:
    """Test CvCExtractorConfig station interactions."""

    def test_extract_without_gear(self):
        """Extracting without miner gear yields small amount."""
        station = CvCExtractorConfig(resource="oxygen", initial_amount=100, small_amount=1).station_cfg()

        harness = StationTestHarness.create(station=station, agent_inventory={"oxygen": 0})
        harness.move_onto_station()

        inv = harness.agent_inventory()
        assert inv.get("oxygen", 0) == 1, f"Expected 1 oxygen, got {inv.get('oxygen', 0)}"
        harness.close()

    def test_extract_with_miner_gear(self):
        """Extracting with miner gear yields large amount via miner handler."""
        station = _extractor_with_miner_handler("oxygen", initial_amount=100)

        harness = StationTestHarness.create(station=station, agent_inventory={"oxygen": 0, "miner": 1})
        harness.move_onto_station()

        inv = harness.agent_inventory()
        assert inv.get("oxygen", 0) == 10, f"Expected 10 oxygen with miner, got {inv.get('oxygen', 0)}"
        harness.close()

    def test_extractor_depletion(self):
        """Extractor is removed when depleted."""
        station = _extractor_with_miner_handler("oxygen", initial_amount=10)

        harness = StationTestHarness.create(station=station, agent_inventory={"oxygen": 0, "miner": 1})
        harness.move_onto_station()

        inv = harness.agent_inventory()
        assert inv.get("oxygen", 0) == 10, f"Expected 10 oxygen, got {inv.get('oxygen', 0)}"
        assert not harness.station_exists("oxygen_extractor"), "Extractor should be removed when depleted"
        harness.close()

    def test_extract_limited_by_inventory(self):
        """Extract amount is limited by extractor's remaining inventory."""
        station = _extractor_with_miner_handler("oxygen", initial_amount=3)

        harness = StationTestHarness.create(station=station, agent_inventory={"oxygen": 0, "miner": 1})
        harness.move_onto_station()

        inv = harness.agent_inventory()
        assert inv.get("oxygen", 0) == 3, f"Expected 3 oxygen (limited by inventory), got {inv.get('oxygen', 0)}"
        harness.close()

    def test_without_miner_gear_gets_small_amount(self):
        """With miner handler registered, agents without miner gear still get small amount."""
        station = _extractor_with_miner_handler("oxygen", initial_amount=100)

        harness = StationTestHarness.create(station=station, agent_inventory={"oxygen": 0})
        harness.move_onto_station()

        inv = harness.agent_inventory()
        assert inv.get("oxygen", 0) == 1, f"Expected 1 oxygen without miner gear, got {inv.get('oxygen', 0)}"
        harness.close()


class TestExtractorsVariantExtractionHandlers:
    """Test the add_extraction_handler mechanism on ExtractorsVariant."""

    def test_add_extraction_handler(self):
        mining = ExtractorsVariant()
        mining.add_extraction_handler("miner_extract", required_resources={"miner": 1}, cost={}, amount=10)
        assert len(mining.extraction_handlers) == 1
        eh = mining.extraction_handlers[0]
        assert eh.name == "miner_extract"
        assert eh.required_resources == {"miner": 1}
        assert eh.cost == {}
        assert eh.amount == 10

    def test_multiple_extraction_handlers(self):
        mining = ExtractorsVariant()
        mining.add_extraction_handler("miner_extract", required_resources={"miner": 1}, cost={}, amount=10)
        mining.add_extraction_handler(
            "super_extract",
            required_resources={"super_miner": 1},
            cost={"energy": 5},
            amount=50,
        )
        assert len(mining.extraction_handlers) == 2
        assert mining.extraction_handlers[0].name == "super_extract"
        assert mining.extraction_handlers[1].name == "miner_extract"
