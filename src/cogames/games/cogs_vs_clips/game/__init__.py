from __future__ import annotations

from cogames.core import CoGameMissionVariant
from cogames.games.cogs_vs_clips.game.cargo import CargoLimitVariant
from cogames.games.cogs_vs_clips.game.clips import (
    AdaptiveClipsVariant,
    AngryClipsVariant,
    ClipsVariant,
    GreedyClipsVariant,
    NoClipsVariant,
)
from cogames.games.cogs_vs_clips.game.damage import DamageVariant
from cogames.games.cogs_vs_clips.game.days import DaysVariant
from cogames.games.cogs_vs_clips.game.energy import EnergyVariant
from cogames.games.cogs_vs_clips.game.extractors import CvCExtractorConfig, ExtractorsVariant
from cogames.games.cogs_vs_clips.game.forced_role_vibes import ForcedRoleVibesVariant
from cogames.games.cogs_vs_clips.game.gear import GearVariant
from cogames.games.cogs_vs_clips.game.gear_stations import GearStationsVariant, WildGearStationsVariant
from cogames.games.cogs_vs_clips.game.heart import HeartVariant
from cogames.games.cogs_vs_clips.game.junction import JunctionVariant
from cogames.games.cogs_vs_clips.game.multi_team import GEAR, MultiTeamVariant
from cogames.games.cogs_vs_clips.game.roles.aligner import AlignerVariant
from cogames.games.cogs_vs_clips.game.roles.miner import MinerVariant
from cogames.games.cogs_vs_clips.game.roles.scout import ScoutVariant
from cogames.games.cogs_vs_clips.game.roles.scrambler import ScramblerVariant
from cogames.games.cogs_vs_clips.game.solar import SolarVariant
from cogames.games.cogs_vs_clips.game.talk import TalkVariant
from cogames.games.cogs_vs_clips.game.teams import TeamVariant
from cogames.games.cogs_vs_clips.game.teams.four_corners import FourCornersVariant
from cogames.games.cogs_vs_clips.game.teams.gear_stations import TeamGearStationsVariant
from cogames.games.cogs_vs_clips.game.teams.hub import TeamHubVariant
from cogames.games.cogs_vs_clips.game.teams.hub_observations import HubObservationsVariant
from cogames.games.cogs_vs_clips.game.teams.junction import TeamJunctionVariant
from cogames.games.cogs_vs_clips.game.teams.junction_deposit import JunctionDepositVariant
from cogames.games.cogs_vs_clips.game.terrain import (
    BaseCompoundVariant,
    CavesVariant,
    CityVariant,
    DesertVariant,
    DistantResourcesVariant,
    EmptyBaseVariant,
    ForestVariant,
    QuadrantBuildingsVariant,
    RandomizeSpawnsVariant,
    Small50Variant,
)
from cogames.games.cogs_vs_clips.game.territory import DamageStrangersVariant, HealTeamVariant, TerritoryVariant
from cogames.games.cogs_vs_clips.game.territory import TerritoryVariant as JunctionNetVariant
from cogames.games.cogs_vs_clips.game.vibes import VibesVariant

__all__ = [
    "AdaptiveClipsVariant",
    "AlignerVariant",
    "AngryClipsVariant",
    "BaseCompoundVariant",
    "CargoLimitVariant",
    "CavesVariant",
    "CityVariant",
    "ClipsVariant",
    "NoClipsVariant",
    "CvCExtractorConfig",
    "DamageStrangersVariant",
    "DamageVariant",
    "DaysVariant",
    "DesertVariant",
    "DistantResourcesVariant",
    "EmptyBaseVariant",
    "EnergyVariant",
    "ForcedRoleVibesVariant",
    "ForestVariant",
    "FourCornersVariant",
    "GreedyClipsVariant",
    "GEAR",
    "GearStationsVariant",
    "GearVariant",
    "HealTeamVariant",
    "HeartVariant",
    "HubObservationsVariant",
    "JunctionDepositVariant",
    "JunctionNetVariant",
    "JunctionVariant",
    "MinerVariant",
    "MultiTeamVariant",
    "QuadrantBuildingsVariant",
    "RandomizeSpawnsVariant",
    "ScoutVariant",
    "ScramblerVariant",
    "Small50Variant",
    "SolarVariant",
    "TeamGearStationsVariant",
    "TeamHubVariant",
    "TeamJunctionVariant",
    "TeamVariant",
    "TerritoryVariant",
    "TalkVariant",
    "VibesVariant",
    "WildGearStationsVariant",
]


def _get_tutorial_variants() -> list[CoGameMissionVariant]:
    # Lazy import to break circular dependency:
    # game/__init__ -> missions.tutorial -> missions.machina_1 -> game.cargo -> game/__init__
    from cogames.games.cogs_vs_clips.missions.tutorial import (  # noqa: PLC0415
        AlignerRewardsVariant,
        MinerRewardsVariant,
        OverrunVariant,
        ScoutRewardsVariant,
        ScramblerRewardsVariant,
    )

    return [
        AlignerRewardsVariant(),
        MinerRewardsVariant(),
        ScoutRewardsVariant(),
        ScramblerRewardsVariant(),
        OverrunVariant(),
    ]


VARIANTS: list[CoGameMissionVariant] = [
    AlignerVariant(),
    CargoLimitVariant(),
    CavesVariant(),
    CityVariant(),
    AngryClipsVariant(),
    ClipsVariant(),
    GreedyClipsVariant(),
    NoClipsVariant(),
    AdaptiveClipsVariant(),
    DamageStrangersVariant(),
    DamageVariant(),
    DaysVariant(),
    DesertVariant(),
    EmptyBaseVariant(),
    EnergyVariant(),
    ExtractorsVariant(),
    ForcedRoleVibesVariant(),
    ForestVariant(),
    FourCornersVariant(),
    GearStationsVariant(),
    HealTeamVariant(),
    HeartVariant(),
    HubObservationsVariant(),
    JunctionDepositVariant(),
    JunctionVariant(),
    MinerVariant(),
    MultiTeamVariant(),
    QuadrantBuildingsVariant(),
    RandomizeSpawnsVariant(),
    ScoutVariant(),
    ScramblerVariant(),
    Small50Variant(),
    SolarVariant(),
    TeamGearStationsVariant(),
    TeamHubVariant(),
    TeamJunctionVariant(),
    TeamVariant(),
    TerritoryVariant(),
    TalkVariant(),
    VibesVariant(),
    WildGearStationsVariant(),
]


def _get_all_variants() -> list[CoGameMissionVariant]:
    return list(VARIANTS) + _get_tutorial_variants()
