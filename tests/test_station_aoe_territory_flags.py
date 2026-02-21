from cogames.cogs_vs_clips.clips import ClipsConfig
from cogames.cogs_vs_clips.cog import CogTeam
from cogames.cogs_vs_clips.hub import CvCHubConfig
from cogames.cogs_vs_clips.junction import CvCJunctionConfig


def _assert_non_mutating_territory_aoe(station) -> None:
    territory = station.aoes["territory"]
    assert territory.radius > 0
    assert territory.mutations == []
    assert territory.presence_deltas == {}


def test_junction_defines_non_mutating_territory_aoe() -> None:
    cogs = CogTeam()
    clips = ClipsConfig()
    station = CvCJunctionConfig().station_cfg(teams=[cogs, clips], owner_team_name=cogs.name)
    _assert_non_mutating_territory_aoe(station)


def test_hub_defines_non_mutating_territory_aoe() -> None:
    station = CvCHubConfig().station_cfg(team=CogTeam())
    _assert_non_mutating_territory_aoe(station)
