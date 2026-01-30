"""CogsGuard tutorial mission configuration."""

from cogames.cogs_vs_clips.mission import Mission, Site
from cogames.cogs_vs_clips.procedural import MachinaArena
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scenes.base_hub import BaseHubConfig


def make_cogsguard_tutorial_site() -> Site:
    """Create a smaller, simpler CogsGuard arena for the tutorial."""
    hub_config = BaseHubConfig(
        corner_bundle="extractors",
        cross_bundle="none",
        cross_distance=5,
        hub_width=15,
        hub_height=15,
        outer_clearance=2,
        stations=[
            "aligner_station",
            "scrambler_station",
            "miner_station",
            "scout_station",
            "chest",
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
    return Site(
        name="cogsguard_tutorial",
        description="CogsGuard tutorial arena - small map for learning",
        map_builder=map_builder,
        min_cogs=1,
        max_cogs=1,
    )


CogsGuardTutorialMission = Mission(
    name="tutorial",
    description="Learn the basics of CogsGuard: Roles, Resources, and Territory Control.",
    site=make_cogsguard_tutorial_site(),
    num_cogs=1,
    max_steps=2000,
    # Generous initial resources for learning
    collective_initial_carbon=50,
    collective_initial_oxygen=50,
    collective_initial_germanium=50,
    collective_initial_silicon=50,
    collective_initial_heart=10,
)
