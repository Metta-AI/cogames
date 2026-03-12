from __future__ import annotations

import logging

from cogames.games.cogs_vs_clips.game.damage import DamageVariant
from cogames.games.cogs_vs_clips.game.days import DayConfig, DaysVariant
from cogames.games.cogs_vs_clips.game.territory import TerritoryVariant as JunctionNetVariant
from cogames.games.cogs_vs_clips.missions.mission import CvCMission
from cogames.games.cogs_vs_clips.missions.terrain import MachinaArena
from mettagrid.mapgen.mapgen import MapGen

HELLO_WORLD_MAP = MapGen.Config(width=100, height=100, instance=MachinaArena.Config(spawn_count=20))

logger = logging.getLogger(__name__)

SMALL_HELLO_WORLD_MAP = MapGen.Config(width=50, height=50, instance=MachinaArena.Config(spawn_count=20))

MEDIUM_HELLO_WORLD_MAP = MapGen.Config(width=100, height=100, instance=MachinaArena.Config(spawn_count=20))

LARGE_HELLO_WORLD_MAP = MapGen.Config(width=150, height=150, instance=MachinaArena.Config(spawn_count=20))

# Energy Starved evals
EnergyStarved = CvCMission(
    name="energy_starved",
    description="Energy is the limiting resource; agents must prioritize energy over other resources.",
    map_builder=HELLO_WORLD_MAP,
    min_cogs=1,
    max_cogs=20,
).with_variants(
    [
        JunctionNetVariant(),
        DamageVariant(),
        DaysVariant(days_config=DayConfig(day_solar=0, night_solar=0)),
    ]
)


EVAL_MISSIONS: list[CvCMission] = [
    EnergyStarved,
]
