from typing import cast

from cogames.cogs_vs_clips.missions import CogsGuardMachina1Mission
from cogames.cogs_vs_clips.sites import COGSGUARD_MACHINA_1
from cogames.cogs_vs_clips.terrain import MachinaArenaConfig
from mettagrid.config.tag import typeTag
from mettagrid.mapgen.mapgen import MapGenConfig
from mettagrid.mapgen.scenes.building_distributions import DistributionType


def test_cogsguard_machina1_base_junction_starts_aligned_to_cogs() -> None:
    env = CogsGuardMachina1Mission.make_env()

    # Only the base-hub junction starts aligned; regular junctions remain neutral.
    assert env.game.objects["junction"].collective is None

    base_junction = env.game.objects["c:junction"]
    assert base_junction.collective == "cogs"
    assert typeTag("junction").name in base_junction.tags


def test_cogsguard_machina1_site_places_c_junction_in_base_hub() -> None:
    map_builder = cast(MapGenConfig, COGSGUARD_MACHINA_1.map_builder)
    instance = map_builder.instance
    assert instance is not None
    assert isinstance(instance, MachinaArenaConfig)
    assert instance.hub.junction_object == "c:junction"
    assert instance.building_distributions is not None
    assert instance.building_distributions["junction"].type == DistributionType.POISSON
