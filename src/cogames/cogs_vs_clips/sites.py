"""Central site definitions shared across mission modules."""

from pathlib import Path
from typing import cast

from cogames.cogs_vs_clips.terrain import MachinaArena, MachinaArenaConfig, RandomTransform, SequentialMachinaArena
from cogames.core import CoGameSite
from mettagrid.map_builder.map_builder import MapBuilderConfig
from mettagrid.mapgen.mapgen import MapGen, MapGenConfig
from mettagrid.mapgen.scenes.base_hub import BaseHub, BaseHubConfig
from mettagrid.mapgen.scenes.building_distributions import DistributionConfig, DistributionType

MAPS_DIR = Path(__file__).resolve().parent.parent / "maps"


def get_map(map_name: str) -> MapBuilderConfig:
    """Load a map builder configuration from the maps directory."""
    normalized = map_name
    if normalized.startswith("evals/"):
        normalized = f"diagnostic_evals/{normalized.split('/', 1)[1]}"
    map_path = MAPS_DIR / normalized
    if not map_path.exists():
        raise FileNotFoundError(f"Map not found: {map_path}")
    return MapGen.Config(
        instance=MapBuilderConfig.from_uri(str(map_path)),
        instances=1,  # Force single instance - use spawn points from ASCII map directly
        fixed_spawn_order=False,
        instance_border_width=0,  # Don't add border - maps already have borders built in
    )


TRAINING_FACILITY = CoGameSite(
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

HELLO_WORLD = CoGameSite(
    name="hello_world",
    description="Welcome to space.",
    map_builder=MapGen.Config(width=100, height=100, instance=MachinaArena.Config(spawn_count=20)),
    min_cogs=1,
    max_cogs=20,
)

MACHINA_1 = CoGameSite(
    name="machina_1",
    description="Your first mission. Collect resources and assemble HEARTs.",
    map_builder=MapGen.Config(
        width=88,
        height=88,
        instance=SequentialMachinaArena.Config(
            spawn_count=20,
            map_perimeter_placements=[("clips:hub", 1)],
        ),
    ),
    min_cogs=1,
    max_cogs=20,
)


def _cogsguard_hub_config() -> BaseHubConfig:
    return BaseHubConfig(
        hub_object="c:hub",
        corner_bundle="extractors",
        cross_bundle="none",
        cross_distance=7,
        stations=[
            "c:aligner",
            "c:scrambler",
            "c:miner",
            "c:scout",
        ],
    )


# Evals site used by diagnostic evaluation missions
# Note: Individual diagnostic missions override this with their own specific maps
EVALS = CoGameSite(
    name="evals",
    description="Diagnostic evaluation arenas.",
    map_builder=get_map("diagnostic_evals/diagnostic_radial.map"),  # Default map (rarely used)
    min_cogs=1,
    max_cogs=8,
)


def make_cogsguard_arena_site(num_agents: int = 10) -> CoGameSite:
    """Create a CogsGuard arena site with configurable agent count."""
    map_builder = MapGen.Config(
        width=50,
        height=50,
        instance=MachinaArena.Config(
            spawn_count=num_agents,
            building_coverage=0.1,
            building_distributions={
                # Avoid "junction clusters" that can become unassailable when captured.
                "junction": DistributionConfig(type=DistributionType.POISSON),
            },
            hub=_cogsguard_hub_config(),
        ),
    )
    return CoGameSite(
        name="cogsguard_arena",
        description="CogsGuard arena map",
        map_builder=map_builder,
        min_cogs=num_agents,
        max_cogs=num_agents,
    )


def _build_cogsguard_machina1_map_builder(spawn_count: int) -> MapGenConfig:
    map_builder = cast(MapGenConfig, MACHINA_1.map_builder).model_copy(deep=True)
    instance = map_builder.instance
    assert instance is not None
    assert isinstance(instance, MachinaArenaConfig)
    existing_building_distributions = instance.building_distributions or {}
    existing_building_distributions = {
        k: (DistributionConfig.model_validate(v) if isinstance(v, dict) else v)
        for k, v in existing_building_distributions.items()
    }
    return map_builder.model_copy(
        update={
            "instance": instance.model_copy(
                update={
                    "spawn_count": spawn_count,
                    "hub": _cogsguard_hub_config(),
                    # Avoid "junction clusters" that can become unassailable when captured.
                    "building_distributions": {
                        **existing_building_distributions,
                        "junction": DistributionConfig(type=DistributionType.POISSON),
                    },
                }
            ),
        }
    )


def make_cogsguard_machina1_site(num_agents: int = 10) -> CoGameSite:
    """Create a CogsGuard Machina1 site with configurable agent count."""
    return CoGameSite(
        name="cogsguard_machina_1",
        description="CogsGuard Machina1 layout with gear stations.",
        map_builder=_build_cogsguard_machina1_map_builder(num_agents),
        min_cogs=num_agents,
        max_cogs=num_agents,
    )


# Default CogsGuard Machina1 site with flexible agent count
COGSGUARD_MACHINA_1 = CoGameSite(
    name="cogsguard_machina_1",
    description="CogsGuard Machina1 layout with gear stations.",
    map_builder=_build_cogsguard_machina1_map_builder(20),
    min_cogs=1,
    max_cogs=20,
)

# Default CogsGuard arena site with flexible agent count
COGSGUARD_ARENA = CoGameSite(
    name="cogsguard_arena",
    description="CogsGuard arena - compact training map with gear stations.",
    map_builder=MapGen.Config(
        width=50,
        height=50,
        instance=MachinaArena.Config(
            spawn_count=20,
            building_coverage=0.1,
            building_distributions={
                # Avoid "junction clusters" that can become unassailable when captured.
                "junction": DistributionConfig(type=DistributionType.POISSON),
            },
            hub=_cogsguard_hub_config(),
        ),
    ),
    min_cogs=1,
    max_cogs=20,
)


SITES = [
    COGSGUARD_MACHINA_1,
    COGSGUARD_ARENA,
    EVALS,
]
