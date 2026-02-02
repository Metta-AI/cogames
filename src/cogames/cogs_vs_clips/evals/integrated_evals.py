from __future__ import annotations

import logging

from cogames.cogs_vs_clips.mission import CvCMission
from cogames.cogs_vs_clips.sites import HELLO_WORLD
from cogames.cogs_vs_clips.terrain import MachinaArena
from cogames.cogs_vs_clips.variants import (
    DarkSideVariant,
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
    map_builder=MapGen.Config(width=150, height=150, instance=MachinaArena.Config(spawn_count=20)),
    min_cogs=1,
    max_cogs=20,
)

# Energy Starved evals
EnergyStarved = CvCMission(
    name="energy_starved",
    description="Energy is the limiting resource; agents must prioritize energy over other resources.",
    site=HELLO_WORLD,
    variants=[
        DarkSideVariant(),
    ],
)


EVAL_MISSIONS: list[CvCMission] = [
    EnergyStarved,
]
