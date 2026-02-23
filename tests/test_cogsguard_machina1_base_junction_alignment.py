from typing import cast

from cogames.cogs_vs_clips.missions import CogsGuardMachina1Mission
from cogames.cogs_vs_clips.sites import COGSGUARD_MACHINA_1
from cogames.cogs_vs_clips.terrain import MachinaArenaConfig
from mettagrid.config.tag import typeTag
from mettagrid.mapgen.mapgen import MapGenConfig
from mettagrid.mapgen.scenes.building_distributions import DistributionType


def test_cogsguard_machina1_neutral_junction_has_no_team_tag() -> None:
    env = CogsGuardMachina1Mission.make_env()

    junction = env.game.objects["junction"]
    assert not any(t.startswith("team:") for t in junction.tags), (
        f"Neutral junction should have no team tags, got {junction.tags}"
    )
    assert typeTag("junction") not in junction.tags, (
        "Junction type tag is auto-generated, should not be in explicit tags"
    )


def test_cogsguard_machina1_site_has_no_home_junction() -> None:
    map_builder = cast(MapGenConfig, COGSGUARD_MACHINA_1.map_builder)
    instance = map_builder.instance
    assert instance is not None
    assert isinstance(instance, MachinaArenaConfig)
    assert instance.building_distributions is not None
    assert instance.building_distributions["junction"].type == DistributionType.POISSON
