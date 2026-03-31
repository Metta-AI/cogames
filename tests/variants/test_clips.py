"""Tests for the clips variant: non-player faction with ships and events."""

from cogames.games.cogs_vs_clips.game import ClipsVariant, MultiTeamVariant, NoClipsVariant
from cogames.games.cogs_vs_clips.game.clips.ship import count_clips_ships_in_map_config
from cogames.games.cogs_vs_clips.game.damage import DamageVariant
from cogames.games.cogs_vs_clips.game.teams import TeamConfig, TeamVariant
from cogames.games.cogs_vs_clips.game.territory import HUB_ALIGN_DISTANCE, JUNCTION_ALIGN_DISTANCE
from cogames.games.cogs_vs_clips.missions.machina_1 import (
    MACHINA_1_MAP_BUILDER,
    make_machina1_mission,
)
from cogames.games.cogs_vs_clips.missions.mission import CvCMission
from cogames.games.cogs_vs_clips.missions.tutorial import ScramblerRewardsVariant, make_tutorial_mission
from cogames.games.cogs_vs_clips.train.cvc_curriculum import split_variants
from mettagrid.map_builder.ascii import AsciiMapBuilder


def _clips_event_group(events: dict, base_name: str) -> dict:
    prefix = f"{base_name}_"
    return {
        name: event
        for name, event in events.items()
        if (name == base_name or name.startswith(prefix)) and event.timesteps
    }


def _sum_max_targets(events: dict) -> int:
    return sum((event.max_targets or 0) for event in events.values())


def test_clips_event_targets_scale_with_default_clips_ship_count() -> None:
    env = make_machina1_mission().make_env()
    assert count_clips_ships_in_map_config(env.game.map_builder) == 4

    neutral_events = _clips_event_group(env.game.events, "neutral_to_clips")
    scramble_events = _clips_event_group(env.game.events, "cogs_to_neutral")
    assert len(neutral_events) == 4
    assert len(scramble_events) == 4
    assert _sum_max_targets(neutral_events) == 4
    assert _sum_max_targets(scramble_events) == 4
    assert all(event.max_targets == 1 for event in neutral_events.values())
    assert all(event.max_targets == 1 for event in scramble_events.values())


def test_clips_event_targets_split_per_corner_ship_count() -> None:
    mission = make_machina1_mission().with_variants([ClipsVariant(num_ships=2)])
    env = mission.make_env()

    neutral_events = _clips_event_group(env.game.events, "neutral_to_clips")
    scramble_events = _clips_event_group(env.game.events, "cogs_to_neutral")
    assert len(neutral_events) == 2
    assert len(scramble_events) == 2
    assert _sum_max_targets(neutral_events) == 2
    assert _sum_max_targets(scramble_events) == 2
    assert all(event.max_targets == 1 for event in neutral_events.values())
    assert all(event.max_targets == 1 for event in scramble_events.values())


def test_zero_ship_clips_does_not_register_ship_objects() -> None:
    env = make_machina1_mission().with_variants([ClipsVariant(num_ships=0)]).make_env()

    assert count_clips_ships_in_map_config(env.game.map_builder) == 0
    assert not any(name.startswith("clips:ship") for name in env.game.objects)
    assert not _clips_event_group(env.game.events, "neutral_to_clips")
    assert not _clips_event_group(env.game.events, "cogs_to_neutral")
    assert not any(mq.tag == "net:clips" for mq in env.game.materialize_queries)


def test_no_clips_variant_removes_ships_and_events() -> None:
    env = make_machina1_mission().with_variants([NoClipsVariant()]).make_env()

    assert count_clips_ships_in_map_config(env.game.map_builder) == 0
    assert not any(name.startswith("clips:ship") for name in env.game.objects)
    assert not _clips_event_group(env.game.events, "neutral_to_clips")
    assert not _clips_event_group(env.game.events, "cogs_to_neutral")
    assert not any(mq.tag == "net:clips" for mq in env.game.materialize_queries)


def test_no_clips_variant_removes_preseeded_ascii_ships() -> None:
    base = CvCMission(
        name="no_clips_ascii_cleanup",
        description="Remove pre-seeded clips ships from ASCII maps",
        map_builder=AsciiMapBuilder.Config(
            char_to_map_name={
                "#": "wall",
                ".": "empty",
                "a": "agent.cogs",
                "S": "clips:ship",
                "j": "junction",
            },
            map_data=[
                ["#", "#", "#", "#", "#"],
                ["#", "a", "S", "j", "#"],
                ["#", ".", "j", ".", "#"],
                ["#", ".", "S", ".", "#"],
                ["#", "#", "#", "#", "#"],
            ],
        ),
        min_cogs=1,
        max_cogs=1,
        max_steps=100,
    ).with_variants(
        [
            TeamVariant(default_teams={"cogs": TeamConfig(name="cogs", short_name="c", num_agents=1)}),
            DamageVariant(),
            NoClipsVariant(),
        ]
    )

    env = base.make_env()

    assert count_clips_ships_in_map_config(env.game.map_builder) == 0
    assert not any(name.startswith("clips:ship") for name in env.game.objects)
    assert not _clips_event_group(env.game.events, "neutral_to_clips")
    assert not _clips_event_group(env.game.events, "cogs_to_neutral")
    assert not any(mq.tag == "net:clips" for mq in env.game.materialize_queries)


def test_split_variants_keeps_clips_defaults_after_no_clips_override() -> None:
    no_clips_variants, _ = split_variants(["clips", "no_clips"])
    no_clips_env = make_machina1_mission().with_variants(no_clips_variants).make_env()

    assert count_clips_ships_in_map_config(no_clips_env.game.map_builder) == 0

    clips_variants, _ = split_variants(["clips"])
    clips_env = make_machina1_mission().with_variants(clips_variants).make_env()

    assert count_clips_ships_in_map_config(clips_env.game.map_builder) == 4


def test_clips_uses_ship_object_with_junction_territory_range() -> None:
    env = make_machina1_mission().make_env()

    ship_names = sorted(name for name in env.game.objects if name.startswith("clips:ship"))
    assert len(ship_names) == 4
    assert "clips:hub" not in env.game.objects
    assert "c:hub" in env.game.objects

    ship = env.game.objects[ship_names[0]]
    assert ship.name == "ship"


def test_clips_default_event_frequency() -> None:
    env = make_machina1_mission().make_env()
    align_steps = env.game.events["neutral_to_clips"].model_dump(mode="python")["timesteps"]
    scramble_steps = env.game.events["cogs_to_neutral"].model_dump(mode="python")["timesteps"]
    assert align_steps[1] - align_steps[0] == 70
    assert scramble_steps[1] - scramble_steps[0] == 70


def test_clips_alignment_range_uses_ship_and_junction_distance() -> None:
    env = make_machina1_mission().make_env()

    net_clips_query = next(mq for mq in env.game.materialize_queries if mq.tag == "net:clips").model_dump(mode="python")
    assert net_clips_query["query"]["source"]["source"] == "type:ship"
    assert net_clips_query["query"]["edge_filters"][0]["radius"] == max(JUNCTION_ALIGN_DISTANCE, HUB_ALIGN_DISTANCE)

    neutral_events = _clips_event_group(env.game.events, "neutral_to_clips")
    scramble_events = _clips_event_group(env.game.events, "cogs_to_neutral")
    assert len(neutral_events) == 4
    assert len(scramble_events) == 4

    neutral_lane_ship_tags: set[str] = set()
    for neutral_event in neutral_events.values():
        neutral_filters = neutral_event.model_dump(mode="python")["target_query"]["filters"]
        neutral_lane_filter = next(f for f in neutral_filters if f["filter_type"] == "max_distance")
        assert neutral_lane_filter["radius"] == JUNCTION_ALIGN_DISTANCE
        lane_query = neutral_lane_filter["query"]
        assert lane_query["query_type"] == "closure"
        assert lane_query["source"]["source"] == "type:ship"
        assert lane_query["candidates"]["source"] == "type:junction"
        assert lane_query["edge_filters"][0]["radius"] == JUNCTION_ALIGN_DISTANCE

        source_tags = {f["tag"] for f in lane_query["source"]["filters"] if f["filter_type"] == "tag"}
        assert "team:clips" in source_tags
        lane_ship_tags = [tag for tag in source_tags if tag.startswith("clips:ship")]
        assert len(lane_ship_tags) == 1
        candidate_tags = [f["tag"] for f in lane_query["candidates"]["filters"] if f["filter_type"] == "tag"]
        assert candidate_tags == lane_ship_tags
        neutral_lane_ship_tags.update(lane_ship_tags)

    scramble_lane_ship_tags: set[str] = set()
    for scramble_event in scramble_events.values():
        scramble_filters = scramble_event.model_dump(mode="python")["target_query"]["filters"]
        scramble_lane_filter = next(f for f in scramble_filters if f["filter_type"] == "max_distance")
        assert scramble_lane_filter["radius"] == JUNCTION_ALIGN_DISTANCE
        lane_query = scramble_lane_filter["query"]
        assert lane_query["query_type"] == "closure"
        source_tags = {f["tag"] for f in lane_query["source"]["filters"] if f["filter_type"] == "tag"}
        lane_ship_tags = [tag for tag in source_tags if tag.startswith("clips:ship")]
        assert len(lane_ship_tags) == 1
        scramble_lane_ship_tags.update(lane_ship_tags)

    assert len(neutral_lane_ship_tags) == 4
    assert len(scramble_lane_ship_tags) == 4


def test_clips_event_targets_use_clips_ship_map_placements_for_ascii_builder() -> None:
    base = CvCMission(
        name="clips_ship_map_config_scaling",
        description="Scale clips events by clips ship map placements",
        map_builder=AsciiMapBuilder.Config(
            char_to_map_name={
                "#": "wall",
                ".": "empty",
                "a": "agent.cogs",
                "S": "clips:ship",
                "j": "junction",
            },
            map_data=[
                ["#", "#", "#", "#", "#"],
                ["#", "a", "S", "j", "#"],
                ["#", ".", "j", ".", "#"],
                ["#", ".", "S", ".", "#"],
                ["#", "#", "#", "#", "#"],
            ],
        ),
        min_cogs=1,
        max_cogs=1,
        max_steps=100,
    ).with_variants(
        [
            TeamVariant(default_teams={"cogs": TeamConfig(name="cogs", short_name="c", num_agents=1)}),
            DamageVariant(),
            ClipsVariant(),
        ]
    )

    env = base.make_env()

    neutral_events = _clips_event_group(env.game.events, "neutral_to_clips")
    scramble_events = _clips_event_group(env.game.events, "cogs_to_neutral")
    assert len(neutral_events) == 2
    assert len(scramble_events) == 2
    assert _sum_max_targets(neutral_events) == 2
    assert _sum_max_targets(scramble_events) == 2
    assert all(event.max_targets == 1 for event in neutral_events.values())
    assert all(event.max_targets == 1 for event in scramble_events.values())


def test_clips_event_targets_scale_after_multi_team_map_rewrite() -> None:
    mission = make_machina1_mission().with_variants([MultiTeamVariant(num_teams=2)])
    env = mission.make_env()

    neutral_events = _clips_event_group(env.game.events, "neutral_to_clips")
    scramble_events = _clips_event_group(env.game.events, "cogs_to_neutral")
    assert len(neutral_events) == 8
    assert len(scramble_events) == 8
    assert _sum_max_targets(neutral_events) == 8
    assert _sum_max_targets(scramble_events) == 8


def test_multiteam_variant_does_not_mutate_shared_map_constants() -> None:
    assert count_clips_ships_in_map_config(MACHINA_1_MAP_BUILDER) == 0
    mission = CvCMission(
        name="basic",
        description="Constructor variant path should not mutate shared map state",
        map_builder=MACHINA_1_MAP_BUILDER,
        num_cogs=8,
        min_cogs=1,
        max_cogs=20,
        max_steps=1000,
    ).with_variants([DamageVariant(), ClipsVariant(), MultiTeamVariant(num_teams=2)])
    env = mission.make_env()

    assert count_clips_ships_in_map_config(env.game.map_builder) == 4 * 2
    assert count_clips_ships_in_map_config(MACHINA_1_MAP_BUILDER) == 0


def test_scrambler_tutorial_overrun_alignment_still_applies() -> None:
    tutorial = make_tutorial_mission()
    mission = tutorial.with_variants([ScramblerRewardsVariant()])
    env = mission.make_env()
    # Overrun sets initial clips tags on junctions instead of using events.
    junction = env.game.objects["junction"]
    assert "team:clips" in junction.tags
    assert "net:clips" in junction.tags
