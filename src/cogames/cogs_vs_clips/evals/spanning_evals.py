# These evals are a spanning set of what might show up on the leaderboard.
# They are not exhaustive, but they should cover most situations.

from __future__ import annotations

import logging

from cogames.cogs_vs_clips.mission import Mission, Site
from cogames.cogs_vs_clips.procedural import MachinaArena
from cogames.cogs_vs_clips.sites import HELLO_WORLD, TRAINING_FACILITY
from cogames.cogs_vs_clips.variants import (
    CompassVariant,
    DarkSideVariant,
    DistantResourcesVariant,
    EmptyBaseVariant,
    EnergizedVariant,
    PackRatVariant,
    QuadrantBuildingsVariant,
    RoughTerrainVariant,
    SingleResourceUniformVariant,
    SuperChargedVariant,
)
from mettagrid.mapgen.mapgen import MapGen

logger = logging.getLogger(__name__)

SMALL_HELLO_WORLD = Site(
    name="small_hello_world",
    description="Small hello world map.",
    map_builder=MapGen.Config(width=50, height=50, instance=MachinaArena.Config(spawn_count=20)),
    min_cogs=1,
    max_cogs=20,
)

MEDIUM_HELLO_WORLD = Site(
    name="medium_hello_world",
    description="Medium hello world map.",
    map_builder=MapGen.Config(width=100, height=100, instance=MachinaArena.Config(spawn_count=20)),
    min_cogs=1,
    max_cogs=20,
)

LARGE_HELLO_WORLD = Site(
    name="large_hello_world",
    description="Large hello world map.",
    map_builder=MapGen.Config(width=500, height=500, instance=MachinaArena.Config(spawn_count=20)),
    min_cogs=1,
    max_cogs=20,
)

# Resource Bottleneck evals
OxygenBottleneck = Mission(
    name="oxygen_bottleneck",
    description="Oxygen is the limiting resource; agents must prioritize oxygen over other resources.",
    site=HELLO_WORLD,
    variants=[
        EmptyBaseVariant(missing=["oxygen_extractor"]),
        SingleResourceUniformVariant(building_name="oxygen_extractor"),
        PackRatVariant(),
    ],
)

# Energy Starved evals
EnergyStarved = Mission(
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
OxygenBottleneckEasy = Mission(
    name="oxygen_bottleneck_easy",
    description="Easy: tuned oxygen focus with simple layout and generous capacities.",
    site=HELLO_WORLD,
    variants=[
        SingleResourceUniformVariant(building_name="oxygen_extractor"),
        PackRatVariant(),
    ],
)

OxygenBottleneckStandard = Mission(
    name="oxygen_bottleneck_standard",
    description="Standard: oxygen is the bottleneck; extractor missing at base.",
    site=HELLO_WORLD,
    variants=[
        EmptyBaseVariant(missing=["oxygen_extractor"]),
    ],
)

OxygenBottleneckHard = Mission(
    name="oxygen_bottleneck_hard",
    description="Hard: oxygen bottleneck plus rough terrain.",
    site=HELLO_WORLD,
    variants=[
        EmptyBaseVariant(missing=["oxygen_extractor"]),
        RoughTerrainVariant(),
    ],
)

# Energy Starved
EnergyStarvedEasy = Mission(
    name="energy_starved_easy",
    description="Easy: abundant energy regen and capacity.",
    site=HELLO_WORLD,
    variants=[
        SuperChargedVariant(),
        EnergizedVariant(),
    ],
)

EnergyStarvedStandard = Mission(
    name="energy_starved_standard",
    description="Standard: energy is the limiting resource with dark-side regen.",
    site=HELLO_WORLD,
    variants=[
        DarkSideVariant(),
    ],
)

EnergyStarvedHard = Mission(
    name="energy_starved_hard",
    description="Hard: energy bottleneck with dark side and rough terrain.",
    site=HELLO_WORLD,
    variants=[
        DarkSideVariant(),
        RoughTerrainVariant(),
    ],
)

# Collect Distant Resources evals
DistantResources = Mission(
    name="distant_resources",
    description="Resources scattered far from base; heavy routing coordination.",
    site=HELLO_WORLD,
    variants=[
        EmptyBaseVariant(),
        DistantResourcesVariant(),
    ],
)

# Distant Resources tiers
DistantResourcesEasy = Mission(
    name="distant_resources_easy",
    description="Easy: simplified distribution and navigation aids.",
    site=HELLO_WORLD,
    variants=[
        CompassVariant(),
        PackRatVariant(),
        DistantResourcesVariant(),
    ],
)

DistantResourcesStandard = Mission(
    name="distant_resources_standard",
    description="Standard: resources scattered far from base.",
    site=HELLO_WORLD,
    variants=[
        CompassVariant(),
        DistantResourcesVariant(),
    ],
)

DistantResourcesHard = Mission(
    name="distant_resources_hard",
    description="Hard: distant resources with rough terrain.",
    site=HELLO_WORLD,
    variants=[
        DistantResourcesVariant(),
        RoughTerrainVariant(),
    ],
)

# Divide and Conquer evals
QuadrantBuildings = Mission(
    name="quadrant_buildings",
    description="Place buildings in the four quadrants of the map.",
    site=HELLO_WORLD,
    variants=[
        EmptyBaseVariant(),
        QuadrantBuildingsVariant(),
    ],
)

# Quadrant Buildings tiers
QuadrantBuildingsEasy = Mission(
    name="quadrant_buildings_easy",
    description="Easy: buildings in quadrants with navigation aid and inventory boost.",
    site=HELLO_WORLD,
    variants=[
        QuadrantBuildingsVariant(),
        CompassVariant(),
        PackRatVariant(),
    ],
)

QuadrantBuildingsStandard = Mission(
    name="quadrant_buildings_standard",
    description="Standard: buildings placed in quadrants.",
    site=HELLO_WORLD,
    variants=[
        QuadrantBuildingsVariant(),
        EmptyBaseVariant(),
    ],
)

QuadrantBuildingsHard = Mission(
    name="quadrant_buildings_hard",
    description="Hard: quadrant distribution with empty base and rough terrain.",
    site=HELLO_WORLD,
    variants=[
        QuadrantBuildingsVariant(),
        EmptyBaseVariant(),
        RoughTerrainVariant(),
    ],
)

EasyHeartsTraining = Mission(
    name="easy_hearts_training",
    description="Simplified heart crafting with generous caps and extractor base.",
    site=TRAINING_FACILITY,
    variants=[
        PackRatVariant(),
    ],
)

EasyHeartsSmallWorld = Mission(
    name="easy_small_hearts",
    description="Simplified heart crafting with generous caps and extractor base.",
    site=SMALL_HELLO_WORLD,
    variants=[
        PackRatVariant(),
    ],
)

EasyHeartsMediumWorld = Mission(
    name="easy_medium_hearts",
    description="Simplified heart crafting with generous caps and extractor base.",
    site=MEDIUM_HELLO_WORLD,
    variants=[
        PackRatVariant(),
    ],
)

EasyHeartsLargeWorld = Mission(
    name="easy_large_hearts",
    description="Simplified heart crafting with generous caps and extractor base.",
    site=LARGE_HELLO_WORLD,
    variants=[
        PackRatVariant(),
    ],
)

EVAL_MISSIONS: list[Mission] = [
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
