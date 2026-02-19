import pytest

from cogames.cogs_vs_clips.stations import CvCHubConfig, CvCJunctionConfig


@pytest.mark.parametrize(
    ("station_cfg", "team"),
    [
        (CvCJunctionConfig(), "blue"),
        (CvCHubConfig(), "red"),
    ],
)
def test_station_defines_non_mutating_territory_aoe(station_cfg, team: str) -> None:
    station = station_cfg.station_cfg(team=team)
    territory = station.aoes["territory"]
    assert territory.radius > 0
    assert territory.mutations == []
    assert territory.presence_deltas == {}
