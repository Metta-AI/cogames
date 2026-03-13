from typing import cast

from cogames.games.cogs_vs_clips.game.territory import JUNCTION_ALIGN_DISTANCE
from cogames.games.cogs_vs_clips.missions.machina_1 import _build_machina1_map_builder, make_machina1_mission
from cogames.games.cogs_vs_clips.missions.terrain import MachinaArenaConfig
from mettagrid.config.tag import typeTag
from mettagrid.mapgen.mapgen import MapGenConfig
from mettagrid.mapgen.scenes.building_distributions import DistributionType
from mettagrid.simulator import Simulation


def test_cvc_machina1_neutral_junction_has_no_team_tag() -> None:
    env = make_machina1_mission().make_env()

    junction = env.game.objects["junction"]
    assert not any(t.startswith("team:") for t in junction.tags), (
        f"Neutral junction should have no team tags, got {junction.tags}"
    )
    assert typeTag("junction") not in junction.tags, (
        "Junction type tag is auto-generated, should not be in explicit tags"
    )


def test_cvc_machina1_site_has_no_home_junction() -> None:
    map_builder = cast(MapGenConfig, _build_machina1_map_builder(spawn_count=20))
    instance = map_builder.instance
    assert instance is not None
    assert isinstance(instance, MachinaArenaConfig)
    assert instance.map_corner_offset == 1
    assert instance.map_perimeter_placements == []
    assert instance.map_corner_placements == []
    assert instance.building_distributions is not None
    assert instance.building_distributions["junction"].type == DistributionType.POISSON


def test_cvc_machina1_has_junction_within_align_distance_of_cogs_hub() -> None:
    r2 = JUNCTION_ALIGN_DISTANCE * JUNCTION_ALIGN_DISTANCE

    for seed in range(10):
        env = make_machina1_mission().make_env()
        assert isinstance(env.game.map_builder, MapGenConfig)
        env.game.map_builder.seed = seed
        sim = Simulation(env)

        tag_names = sim.config.game.id_map().tag_names()
        cogs_hubs: list[tuple[int, int]] = []
        junctions: list[tuple[int, int]] = []

        for obj in sim.grid_objects().values():
            if obj["type_name"] == "junction":
                junctions.append((obj["r"], obj["c"]))
                continue
            if obj["type_name"] != "hub":
                continue
            tags = [tag_names[tag_id] for tag_id in obj["tag_ids"]]
            if "team:cogs" in tags:
                cogs_hubs.append((obj["r"], obj["c"]))

        assert cogs_hubs, "Expected at least one cogs hub on map"
        assert junctions, "Expected at least one junction on map"

        for hr, hc in cogs_hubs:
            assert any((jr - hr) * (jr - hr) + (jc - hc) * (jc - hc) <= r2 for jr, jc in junctions), (
                f"Expected junction within align range of cogs hub for seed={seed}; "
                f"hub={(hr, hc)} junction_count={len(junctions)}"
            )
