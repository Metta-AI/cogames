# These evals are a spanning set of what might show up on the leaderboard.
# They are not exhaustive, but they should cover most situations.

from __future__ import annotations

import logging

from cogames.cogs_vs_clips.buildings import MachinaArena
from cogames.cogs_vs_clips.mission import CvCMission
from cogames.cogs_vs_clips.sites import HELLO_WORLD
from cogames.cogs_vs_clips.variants import (
    DarkSideVariant,
    DistantResourcesVariant,
    EmptyBaseVariant,
    EnergizedVariant,
    QuadrantBuildingsVariant,
)
from cogames.core import CoGameSite
from mettagrid.mapgen.mapgen import MapGen

logger = logging.getLogger(__name__)

SMALL_HELLO_WORLD = CoGameSite(
    name="small_hello_world",
    description="Small hello world map.",
    map_builder=MapGen.Config(width=50, height=50, instance=MachinaArena.Config(spawn_count=20)),
    min_cogs=1,
    max_cogs=20,
)

MEDIUM_HELLO_WORLD = CoGameSite(
    name="medium_hello_world",
    description="Medium hello world map.",
    map_builder=MapGen.Config(width=100, height=100, instance=MachinaArena.Config(spawn_count=20)),
    min_cogs=1,
    max_cogs=20,
)

LARGE_HELLO_WORLD = CoGameSite(
    name="large_hello_world",
    description="Large hello world map.",
    map_builder=MapGen.Config(width=500, height=500, instance=MachinaArena.Config(spawn_count=20)),
    min_cogs=1,
    max_cogs=20,
)

# Curated difficulty tiers per mission
# ------------------------------------------------------------

# Collect Distant Resources evals
DistantResources = CvCMission(
    name="distant_resources",
    description="Resources scattered far from base; heavy routing coordination.",
    site=HELLO_WORLD,
    variants=[
        EmptyBaseVariant(),
        DistantResourcesVariant(),
    ],
)

# Distant Resources tiers
DistantResourcesEasy = CvCMission(
    name="distant_resources_easy",
    description="Easy: simplified distribution with generous capacity.",
    site=HELLO_WORLD,
    variants=[
        EnergizedVariant(),
        DistantResourcesVariant(),
    ],
)

DistantResourcesStandard = CvCMission(
    name="distant_resources_standard",
    description="Standard: resources scattered far from base.",
    site=HELLO_WORLD,
    variants=[
        DistantResourcesVariant(),
    ],
)

DistantResourcesHard = CvCMission(
    name="distant_resources_hard",
    description="Hard: distant resources with dark side.",
    site=HELLO_WORLD,
    variants=[
        DistantResourcesVariant(),
        DarkSideVariant(),
    ],
)

# Divide and Conquer evals
QuadrantBuildings = CvCMission(
    name="quadrant_buildings",
    description="Place buildings in the four quadrants of the map.",
    site=HELLO_WORLD,
    variants=[
        EmptyBaseVariant(),
        QuadrantBuildingsVariant(),
    ],
)

# Quadrant Buildings tiers
QuadrantBuildingsEasy = CvCMission(
    name="quadrant_buildings_easy",
    description="Easy: buildings in quadrants with energy boost.",
    site=HELLO_WORLD,
    variants=[
        QuadrantBuildingsVariant(),
        EnergizedVariant(),
    ],
)

QuadrantBuildingsStandard = CvCMission(
    name="quadrant_buildings_standard",
    description="Standard: buildings placed in quadrants.",
    site=HELLO_WORLD,
    variants=[
        QuadrantBuildingsVariant(),
        EmptyBaseVariant(),
    ],
)

QuadrantBuildingsHard = CvCMission(
    name="quadrant_buildings_hard",
    description="Hard: quadrant distribution with empty base and dark side.",
    site=HELLO_WORLD,
    variants=[
        QuadrantBuildingsVariant(),
        EmptyBaseVariant(),
        DarkSideVariant(),
    ],
)

EVAL_MISSIONS: list[CvCMission] = [
    # Distant resources tiers
    DistantResourcesEasy,
    DistantResourcesStandard,
    DistantResourcesHard,
    # Quadrant buildings tiers
    QuadrantBuildingsEasy,
    QuadrantBuildingsStandard,
    QuadrantBuildingsHard,
]
