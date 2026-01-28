"""Central site definitions shared across mission modules."""

from typing import cast

from cogames.cogs_vs_clips.mission import Site
from cogames.cogs_vs_clips.mission_utils import get_map
from cogames.cogs_vs_clips.procedural import MachinaArena, RandomTransform, SequentialMachinaArena
from mettagrid.mapgen.mapgen import MapGen, MapGenConfig
from mettagrid.mapgen.scenes.base_hub import BaseHub, BaseHubConfig

TRAINING_FACILITY = Site(
    name="training_facility",
    description="COG Training Facility. Basic training facility with open spaces and no obstacles.",
    map_builder=MapGen.Config(
        width=13,
        height=13,
        instance=RandomTransform.Config(
            scene=BaseHub.Config(
                spawn_count=4,
                corner_bundle="extractors",
                corner_objects=[
                    "carbon_extractor",
                    "oxygen_extractor",
                    "germanium_extractor",
                    "silicon_extractor",
                ],
                cross_bundle="none",
            )
        ),
    ),
    min_cogs=1,
    max_cogs=4,
)

HELLO_WORLD = Site(
    name="hello_world",
    description="Welcome to space.",
    map_builder=MapGen.Config(width=100, height=100, instance=MachinaArena.Config(spawn_count=20)),
    min_cogs=1,
    max_cogs=20,
)

MACHINA_1 = Site(
    name="machina_1",
    description="Your first mission. Collect resources and assemble HEARTs.",
    map_builder=MapGen.Config(width=88, height=88, instance=SequentialMachinaArena.Config(spawn_count=20)),
    min_cogs=1,
    max_cogs=20,
)


def _cogsguard_hub_config() -> BaseHubConfig:
    return BaseHubConfig(
        corner_bundle="extractors",
        cross_bundle="none",
        cross_distance=7,
        stations=[
            "aligner_station",
            "scrambler_station",
            "miner_station",
            "scout_station",
            "chest",
        ],
    )


# Evals site used by diagnostic evaluation missions
# Note: Individual diagnostic missions override this with their own specific maps
EVALS = Site(
    name="evals",
    description="Diagnostic evaluation arenas.",
    map_builder=get_map("diagnostic_evals/diagnostic_radial.map"),  # Default map (rarely used)
    min_cogs=1,
    max_cogs=8,
)


def make_cogsguard_arena_site(num_agents: int = 10) -> Site:
    """Create a CogsGuard arena site with configurable agent count."""
    map_builder = cast(MapGenConfig, MACHINA_1.map_builder).model_copy(deep=True)
    instance = map_builder.instance
    assert instance is not None
    map_builder = map_builder.model_copy(
        update={
            "instance": instance.model_copy(
                update={
                    "spawn_count": num_agents,
                    "hub": _cogsguard_hub_config(),
                }
            ),
        }
    )
    return Site(
        name="cogsguard_arena",
        description="CogsGuard arena map (Machina1 layout with gear stations)",
        map_builder=map_builder,
        min_cogs=num_agents,
        max_cogs=num_agents,
    )


_COGSGUARD_ARENA_MAP_BUILDER = cast(MapGenConfig, MACHINA_1.map_builder).model_copy(deep=True)
_COGSGUARD_ARENA_INSTANCE = _COGSGUARD_ARENA_MAP_BUILDER.instance
assert _COGSGUARD_ARENA_INSTANCE is not None
_COGSGUARD_ARENA_MAP_BUILDER = _COGSGUARD_ARENA_MAP_BUILDER.model_copy(
    update={
        "instance": _COGSGUARD_ARENA_INSTANCE.model_copy(
            update={
                "spawn_count": 20,
                "hub": _cogsguard_hub_config(),
            }
        ),
    }
)

# Default CogsGuard arena site with flexible agent count
COGSGUARD_ARENA = Site(
    name="cogsguard_arena",
    description="CogsGuard arena - compete to control junctions with gear abilities.",
    map_builder=_COGSGUARD_ARENA_MAP_BUILDER,
    min_cogs=1,
    max_cogs=20,
)


# Feature flag: Set to True to include legacy (pre-CogsGuard) sites in the CLI.
# To enable, add TRAINING_FACILITY, HELLO_WORLD, MACHINA_1 to SITES below.
# Also set _INCLUDE_LEGACY_MISSIONS = True in missions.py.
_INCLUDE_LEGACY_SITES = False

_LEGACY_SITES = [
    TRAINING_FACILITY,
    HELLO_WORLD,
    MACHINA_1,
]

SITES = [
    COGSGUARD_ARENA,
    EVALS,
    *(_LEGACY_SITES if _INCLUDE_LEGACY_SITES else []),
]
