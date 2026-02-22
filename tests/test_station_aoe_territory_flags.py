from cogames.cogs_vs_clips.clips import ClipsConfig
from cogames.cogs_vs_clips.cog import CogTeam
from cogames.cogs_vs_clips.hub import CvCHubConfig
from cogames.cogs_vs_clips.junction import CvCJunctionConfig


def _assert_territory_controls(station) -> None:
    assert len(station.territory_controls) > 0
    tc = station.territory_controls[0]
    assert tc.territory == "team_territory"
    assert tc.strength > 0


def test_junction_defines_territory_controls() -> None:
    cogs = CogTeam()
    clips = ClipsConfig()
    station = CvCJunctionConfig().station_cfg(teams=[cogs, clips], owner_team_name=cogs.name)
    _assert_territory_controls(station)


def test_hub_defines_territory_controls() -> None:
    station = CvCHubConfig().station_cfg(team=CogTeam())
    _assert_territory_controls(station)
