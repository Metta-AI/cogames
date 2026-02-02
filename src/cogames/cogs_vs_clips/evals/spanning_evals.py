# These evals are a spanning set of what might show up on the leaderboard.
# They are not exhaustive, but they should cover most situations.

from __future__ import annotations

import logging

from cogames.cogs_vs_clips.mission import CvCMission
from cogames.cogs_vs_clips.sites import HELLO_WORLD, TRAINING_FACILITY
from cogames.cogs_vs_clips.terrain import MachinaArena
from cogames.cogs_vs_clips.variants import (
    DarkSideVariant,
    DistantResourcesVariant,
    EmptyBaseVariant,
    EnergizedVariant,
    QuadrantBuildingsVariant,
    SingleResourceUniformVariant,
    SuperChargedVariant,
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

# Resource Bottleneck evals
OxygenBottleneck = CvCMission(
    name="oxygen_bottleneck",
    description="Oxygen is the limiting resource; agents must prioritize oxygen over other resources.",
    site=HELLO_WORLD,
    variants=[
        EmptyBaseVariant(missing=["oxygen_extractor"]),
        SingleResourceUniformVariant(building_name="oxygen_extractor"),
        EnergizedVariant(),
    ],
)

# Energy Starved evals
EnergyStarved = CvCMission(
    name="energy_starved",
    description="Energy is the limiting resource; agents must prioritize energy over other resources.",
    site=HELLO_WORLD,
    variants=[
        EmptyBaseVariant(),
        DarkSideVariant(),
    ],
)

# Curated difficulty tiers per mission
# ------------------------------------------------------------
# Oxygen Bottleneck
OxygenBottleneckEasy = CvCMission(
    name="oxygen_bottleneck_easy",
    description="Easy: tuned oxygen focus with simple layout and generous capacities.",
    site=HELLO_WORLD,
    variants=[
        SingleResourceUniformVariant(building_name="oxygen_extractor"),
        EnergizedVariant(),
    ],
)

OxygenBottleneckStandard = CvCMission(
    name="oxygen_bottleneck_standard",
    description="Standard: oxygen is the bottleneck; extractor missing at base.",
    site=HELLO_WORLD,
    variants=[
        EmptyBaseVariant(missing=["oxygen_extractor"]),
    ],
)

OxygenBottleneckHard = CvCMission(
    name="oxygen_bottleneck_hard",
    description="Hard: oxygen bottleneck with dark side.",
    site=HELLO_WORLD,
    variants=[
        EmptyBaseVariant(missing=["oxygen_extractor"]),
        DarkSideVariant(),
    ],
)

# Energy Starved
EnergyStarvedEasy = CvCMission(
    name="energy_starved_easy",
    description="Easy: abundant energy regen and capacity.",
    site=HELLO_WORLD,
    variants=[
        SuperChargedVariant(),
        EnergizedVariant(),
    ],
)

EnergyStarvedStandard = CvCMission(
    name="energy_starved_standard",
    description="Standard: energy is the limiting resource with dark-side regen.",
    site=HELLO_WORLD,
    variants=[
        DarkSideVariant(),
    ],
)

EnergyStarvedHard = CvCMission(
    name="energy_starved_hard",
    description="Hard: energy bottleneck with dark side.",
    site=HELLO_WORLD,
    variants=[
        DarkSideVariant(),
    ],
)

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

EasyHeartsTraining = CvCMission(
    name="easy_hearts_training",
    description="Simplified heart crafting with generous energy.",
    site=TRAINING_FACILITY,
    variants=[
        EnergizedVariant(),
    ],
)

EasyHeartsSmallWorld = CvCMission(
    name="easy_small_hearts",
    description="Simplified heart crafting with generous energy.",
    site=SMALL_HELLO_WORLD,
    variants=[
        EnergizedVariant(),
    ],
)

EasyHeartsMediumWorld = CvCMission(
    name="easy_medium_hearts",
    description="Simplified heart crafting with generous energy.",
    site=MEDIUM_HELLO_WORLD,
    variants=[
        EnergizedVariant(),
    ],
)

EasyHeartsLargeWorld = CvCMission(
    name="easy_large_hearts",
    description="Simplified heart crafting with generous energy.",
    site=LARGE_HELLO_WORLD,
    variants=[
        EnergizedVariant(),
    ],
)

EVAL_MISSIONS: list[CvCMission] = [
    # Oxygen bottleneck tiers
    OxygenBottleneckEasy,
    OxygenBottleneckStandard,
    OxygenBottleneckHard,
    # Energy starved tiers
    EnergyStarvedEasy,
    EnergyStarvedStandard,
    EnergyStarvedHard,
    # Distant resources tiers
    DistantResourcesEasy,
    DistantResourcesStandard,
    DistantResourcesHard,
    # Quadrant buildings tiers
    QuadrantBuildingsEasy,
    QuadrantBuildingsStandard,
    QuadrantBuildingsHard,
    # Hearts missions (easy only by design)
    EasyHeartsTraining,
    EasyHeartsSmallWorld,
    EasyHeartsMediumWorld,
    EasyHeartsLargeWorld,
]
