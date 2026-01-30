from functools import lru_cache

from cogames.cogs_vs_clips.mission import AnyMission, Mission
from cogames.cogs_vs_clips.sites import (
    COGSGUARD_ARENA,
    COGSGUARD_MACHINA_1,
    make_cogsguard_machina1_site,
)
from mettagrid.config.mettagrid_config import MettaGridConfig

# CogsGuard Missions


def make_cogsguard_mission(num_agents: int = 10, max_steps: int = 10000) -> Mission:
    """Create a CogsGuard mission with configurable parameters (Machina1 layout)."""
    return Mission(
        name="basic",
        description="Basic CogsGuard mission (Machina1 layout)",
        site=make_cogsguard_machina1_site(num_agents),
        num_cogs=num_agents,
        max_steps=max_steps,
    )


CogsGuardMachina1Mission = Mission(
    name="basic",
    description="CogsGuard Machina1 - compete to control junctions with gear abilities.",
    site=COGSGUARD_MACHINA_1,
    num_cogs=8,
    max_steps=10000,
)

CogsGuardBasicMission = Mission(
    name="basic",
    description="CogsGuard Arena - compact training map with gear abilities.",
    site=COGSGUARD_ARENA,
    num_cogs=8,
    max_steps=1000,
)


_CORE_MISSIONS: list[AnyMission] = [
    CogsGuardMachina1Mission,
    CogsGuardBasicMission,
]


def get_core_missions() -> list[AnyMission]:
    return list(_CORE_MISSIONS)


def get_legacy_missions() -> list[Mission]:
    """Get legacy (pre-CogsGuard) missions for backward compatibility.

    Legacy missions have been removed. Returns empty list.
    """
    return []


def _build_eval_missions() -> list[Mission]:
    from cogames.cogs_vs_clips.evals.integrated_evals import EVAL_MISSIONS as INTEGRATED_EVAL_MISSIONS

    return [
        *INTEGRATED_EVAL_MISSIONS,
    ]


@lru_cache(maxsize=1)
def get_missions() -> list[AnyMission]:
    return [*_CORE_MISSIONS, *_build_eval_missions()]


def __getattr__(name: str) -> list[AnyMission]:
    if name == "MISSIONS":
        missions = get_missions()
        globals()["MISSIONS"] = missions
        return missions
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def make_game(num_cogs: int = 2, map_name: str = "training_facility_open_1.map") -> MettaGridConfig:
    """Create a default CogsGuard game configuration."""
    mission = make_cogsguard_mission(num_agents=num_cogs)
    return mission.make_env()
