# These evals are a spanning set of what might show up on the leaderboard.
# They are not exhaustive, but they should cover most situations.

from __future__ import annotations

import logging

from cogames.games.cogs_vs_clips.evals.integrated_evals import HELLO_WORLD_MAP
from cogames.games.cogs_vs_clips.game import (
    DistantResourcesVariant,
    EmptyBaseVariant,
    EnergyVariant,
    HeartVariant,
    QuadrantBuildingsVariant,
)
from cogames.games.cogs_vs_clips.game.days import DayConfig, DaysVariant
from cogames.games.cogs_vs_clips.missions.mission import CvCMission
from cogames.games.cogs_vs_clips.missions.terrain import MachinaArena, RandomTransform
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scenes.compound import Compound

TRAINING_FACILITY_MAP = MapGen.Config(
    width=13,
    height=13,
    instance=RandomTransform.Config(
        scene=Compound.Config(
            spawn_count=4,
            corner_bundle="none",
            cross_bundle="none",
        )
    ),
)

logger = logging.getLogger(__name__)

SMALL_HELLO_WORLD_MAP = MapGen.Config(width=50, height=50, instance=MachinaArena.Config(spawn_count=20))

MEDIUM_HELLO_WORLD_MAP = MapGen.Config(width=100, height=100, instance=MachinaArena.Config(spawn_count=20))

LARGE_HELLO_WORLD_MAP = MapGen.Config(width=500, height=500, instance=MachinaArena.Config(spawn_count=20))


# Curated difficulty tiers per mission
# ------------------------------------------------------------

# Collect Distant Resources evals
DistantResources = CvCMission(
    name="distant_resources",
    description="Resources scattered far from base; heavy routing coordination.",
    map_builder=HELLO_WORLD_MAP,
    min_cogs=1,
    max_cogs=20,
).with_variants(
    [
        EmptyBaseVariant(),
        DistantResourcesVariant(),
    ]
)

# Distant Resources tiers
DistantResourcesEasy = CvCMission(
    name="distant_resources_easy",
    description="Easy: simplified distribution with generous capacity.",
    map_builder=HELLO_WORLD_MAP,
    min_cogs=1,
    max_cogs=20,
).with_variants(
    [
        DaysVariant(days_config=DayConfig(day_solar=255, night_solar=255)),
        EnergyVariant(limit=255),
        DistantResourcesVariant(),
    ]
)

DistantResourcesStandard = CvCMission(
    name="distant_resources_standard",
    description="Standard: resources scattered far from base.",
    map_builder=HELLO_WORLD_MAP,
    min_cogs=1,
    max_cogs=20,
).with_variants(
    [
        DistantResourcesVariant(),
    ]
)

DistantResourcesHard = CvCMission(
    name="distant_resources_hard",
    description="Hard: distant resources with dark side.",
    map_builder=HELLO_WORLD_MAP,
    min_cogs=1,
    max_cogs=20,
).with_variants(
    [
        DaysVariant(days_config=DayConfig(day_solar=0, night_solar=0)),
        DistantResourcesVariant(),
    ]
)

# Divide and Conquer evals
QuadrantBuildings = CvCMission(
    name="quadrant_buildings",
    description="Place buildings in the four quadrants of the map.",
    map_builder=HELLO_WORLD_MAP,
    min_cogs=1,
    max_cogs=20,
).with_variants(
    [
        EmptyBaseVariant(),
        QuadrantBuildingsVariant(),
    ]
)

# Quadrant Buildings tiers
QuadrantBuildingsEasy = CvCMission(
    name="quadrant_buildings_easy",
    description="Easy: buildings in quadrants with energy boost.",
    map_builder=HELLO_WORLD_MAP,
    min_cogs=1,
    max_cogs=20,
).with_variants(
    [
        DaysVariant(days_config=DayConfig(day_solar=255, night_solar=255)),
        EnergyVariant(limit=255),
        QuadrantBuildingsVariant(),
    ]
)

QuadrantBuildingsStandard = CvCMission(
    name="quadrant_buildings_standard",
    description="Standard: buildings placed in quadrants.",
    map_builder=HELLO_WORLD_MAP,
    min_cogs=1,
    max_cogs=20,
).with_variants(
    [
        QuadrantBuildingsVariant(),
        EmptyBaseVariant(),
    ]
)

QuadrantBuildingsHard = CvCMission(
    name="quadrant_buildings_hard",
    description="Hard: quadrant distribution with empty base and dark side.",
    map_builder=HELLO_WORLD_MAP,
    min_cogs=1,
    max_cogs=20,
).with_variants(
    [
        DaysVariant(days_config=DayConfig(day_solar=0, night_solar=0)),
        QuadrantBuildingsVariant(),
        EmptyBaseVariant(),
    ]
)

EasyHeartsTraining = CvCMission(
    name="easy_hearts_training",
    description="Simplified heart crafting with generous energy.",
    map_builder=TRAINING_FACILITY_MAP,
    min_cogs=1,
    max_cogs=4,
).with_variants(
    [
        HeartVariant(),
        DaysVariant(days_config=DayConfig(day_solar=255, night_solar=255)),
        EnergyVariant(limit=255),
    ]
)

EasyHeartsSmallWorld = CvCMission(
    name="easy_small_hearts",
    description="Simplified heart crafting with generous energy.",
    map_builder=SMALL_HELLO_WORLD_MAP,
    min_cogs=1,
    max_cogs=20,
).with_variants(
    [
        HeartVariant(),
        DaysVariant(days_config=DayConfig(day_solar=255, night_solar=255)),
        EnergyVariant(limit=255),
    ]
)

EasyHeartsMediumWorld = CvCMission(
    name="easy_medium_hearts",
    description="Simplified heart crafting with generous energy.",
    map_builder=MEDIUM_HELLO_WORLD_MAP,
    min_cogs=1,
    max_cogs=20,
).with_variants(
    [
        HeartVariant(),
        DaysVariant(days_config=DayConfig(day_solar=255, night_solar=255)),
        EnergyVariant(limit=255),
    ]
)

EasyHeartsLargeWorld = CvCMission(
    name="easy_large_hearts",
    description="Simplified heart crafting with generous energy.",
    map_builder=LARGE_HELLO_WORLD_MAP,
    min_cogs=1,
    max_cogs=20,
).with_variants(
    [
        HeartVariant(),
        DaysVariant(days_config=DayConfig(day_solar=255, night_solar=255)),
        EnergyVariant(limit=255),
    ]
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
    # Hearts missions (easy only by design)
    EasyHeartsTraining,
    EasyHeartsSmallWorld,
    EasyHeartsMediumWorld,
    EasyHeartsLargeWorld,
]
