"""Tests that TerritoryVariant assigns territory_controls to hub, junction, and ship."""

from cogsguard.game.teams import TeamConfig, TeamVariant
from cogsguard.game.territory import TerritoryVariant
from cogsguard.missions.arena import make_arena_map_builder
from cogsguard.missions.mission import CvCMission


def _assert_territory_controls(station) -> None:
    assert len(station.territory_controls) > 0
    tc = station.territory_controls[0]
    assert tc.territory == "team_territory"
    assert tc.strength > 0


def test_junction_defines_territory_controls() -> None:
    env = (
        CvCMission(
            name="test",
            description="test",
            map_builder=make_arena_map_builder(num_agents=4),
            min_cogs=4,
            max_cogs=4,
            num_cogs=4,
            max_steps=100,
        )
        .with_variants(
            [
                TeamVariant(default_teams={"cogs": TeamConfig(name="cogs", num_agents=4)}),
                TerritoryVariant(),
            ]
        )
        .make_env()
    )
    junction = env.game.objects["junction"]
    _assert_territory_controls(junction)


def test_hub_defines_territory_controls() -> None:
    env = (
        CvCMission(
            name="test",
            description="test",
            map_builder=make_arena_map_builder(num_agents=4),
            min_cogs=4,
            max_cogs=4,
            num_cogs=4,
            max_steps=100,
        )
        .with_variants(
            [
                TeamVariant(default_teams={"cogs": TeamConfig(name="cogs", num_agents=4)}),
                TerritoryVariant(),
            ]
        )
        .make_env()
    )
    hub = env.game.objects["c:hub"]
    _assert_territory_controls(hub)
