"""CogsGuard tutorial mission configuration."""

from cogames.cogs_vs_clips.mission import CvCMission
from cogames.cogs_vs_clips.team import CogTeam
from cogames.cogs_vs_clips.terrain import MachinaArena
from cogames.core import CoGameSite
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scenes.base_hub import BaseHubConfig


def make_cogsguard_tutorial_site() -> CoGameSite:
    """Create a smaller, simpler CogsGuard arena for the tutorial."""
    hub_config = BaseHubConfig(
        hub_object="c:hub",
        corner_bundle="extractors",
        cross_bundle="none",
        cross_distance=5,
        hub_width=15,
        hub_height=15,
        outer_clearance=2,
        stations=[
            "c:aligner",
            "c:scrambler",
            "c:miner",
            "c:scout",
        ],
    )
    map_builder = MapGen.Config(
        width=35,
        height=35,
        instance=MachinaArena.Config(
            spawn_count=1,
            building_coverage=0.05,
            hub=hub_config,
        ),
    )
    return CoGameSite(
        name="cogsguard_tutorial",
        description="CogsGuard tutorial arena - small map for learning",
        map_builder=map_builder,
        min_cogs=1,
        max_cogs=1,
    )


CogsGuardTutorialMission = CvCMission(
    name="tutorial",
    description="Learn the basics of CogsGuard: Roles, Resources, and Territory Control.",
    site=make_cogsguard_tutorial_site(),
    num_cogs=1,
    max_steps=2000,
    teams={
        "cogs": CogTeam(name="cogs", num_agents=1, wealth=1),
    },
)
