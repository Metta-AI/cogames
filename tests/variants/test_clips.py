"""Tests for the clips variant: non-player faction with ships and events."""

from cogames.games.cogs_vs_clips.game import (
    AdaptiveClipsVariant,
    ClipsVariant,
    MultiTeamVariant,
    NoClipsVariant,
)
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
from mettagrid.config.filter import GameValueFilter, NotFilter, OrFilter
from mettagrid.config.game_value import QueryCountValue, SumGameValue
from mettagrid.config.query import Query
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


def test_adaptive_clips_variant_is_exported() -> None:
    assert AdaptiveClipsVariant().name == "adaptive_clips"


def test_adaptive_clips_split_variants_resolves() -> None:
    variants, _ = split_variants(["clips", "adaptive_clips"])
    adaptive = next(v for v in variants if v.name == "adaptive_clips")
    assert isinstance(adaptive, AdaptiveClipsVariant)


def test_adaptive_clips_wires_clips_config_after_configure() -> None:
    mission = make_machina1_mission().with_variants(
        [AdaptiveClipsVariant(dominance_ratio=4, dominant_targets_per_lane=2)]
    )
    _ = mission.make_env()
    clips = mission.required_variant(ClipsVariant).clips
    assert clips is not None
    assert clips.adaptive_enabled is True
    assert clips.adaptive_dominance_ratio == 4
    assert clips.adaptive_dominant_targets_per_lane == 2


def test_adaptive_clips_adds_balanced_and_burst_event_families() -> None:
    env = make_machina1_mission().with_variants([AdaptiveClipsVariant()]).make_env()
    neutral = _clips_event_group(env.game.events, "neutral_to_clips")
    scramble = _clips_event_group(env.game.events, "cogs_to_neutral")
    neutral_names = set(neutral.keys())
    scramble_names = set(scramble.keys())
    assert "neutral_to_clips_balanced" in neutral_names
    for n in (1, 2, 3):
        assert f"neutral_to_clips_burst_{n}" in neutral_names
    assert "cogs_to_neutral_balanced" in scramble_names
    for n in (1, 2, 3):
        assert f"cogs_to_neutral_burst_{n}" in scramble_names
    assert "neutral_to_clips_balanced_s1" in neutral_names
    assert "neutral_to_clips_burst_2_s3" in neutral_names
    assert "cogs_to_neutral_balanced_s1" in scramble_names
    assert "cogs_to_neutral_burst_2_s3" in scramble_names


def test_adaptive_clips_balanced_family_keeps_current_timesteps() -> None:
    baseline = make_machina1_mission().make_env()
    adaptive_env = make_machina1_mission().with_variants([AdaptiveClipsVariant()]).make_env()
    expected_align = baseline.game.events["neutral_to_clips"].timesteps
    balanced_align = adaptive_env.game.events["neutral_to_clips_balanced"]
    assert balanced_align.timesteps == expected_align
    assert balanced_align.max_targets == 1

    expected_scramble = baseline.game.events["cogs_to_neutral"].timesteps
    balanced_scramble = adaptive_env.game.events["cogs_to_neutral_balanced"]
    assert balanced_scramble.timesteps == expected_scramble
    assert balanced_scramble.max_targets == 1


_AdaptiveBalancedGateSigs = {
    frozenset({((1.0, -2.0), 0), ((1.0, 1.0), 1)}),
    frozenset({((-2.0, 1.0), 0), ((1.0, 1.0), 1)}),
}


def _or_neg_gate_signatures(o: OrFilter) -> set[tuple[tuple[float, ...], int]]:
    """Signatures (weights tuple, min) of GameValueFilters under each Not in an anyOf gate."""
    sigs: set[tuple[tuple[float, ...], int]] = set()
    for nf in o.inner:
        assert isinstance(nf, NotFilter)
        gvf = nf.inner
        assert isinstance(gvf, GameValueFilter)
        val = gvf.value
        assert isinstance(val, SumGameValue)
        assert val.weights is not None
        sigs.add((tuple(val.weights), gvf.min))
    return sigs


def _assert_adaptive_balanced_mutual_exclusion_gates(event) -> None:
    """Balanced align or scramble: two anyOf negated-dominance gates (see spec)."""
    assert isinstance(event.target_query, Query)
    bal_filters = event.target_query.filters
    ors = [f for f in bal_filters if isinstance(f, OrFilter)]
    assert len(ors) == 2, "balanced lane should stack two negated-dominance anyOf gates"
    for o in ors:
        assert len(o.inner) == 2
        assert all(isinstance(inner, NotFilter) for inner in o.inner)
    gate_sigs = {frozenset(_or_neg_gate_signatures(o)) for o in ors}
    assert gate_sigs == _AdaptiveBalancedGateSigs


def _assert_adaptive_burst_cogs_dominant_conjunction(event) -> None:
    """Burst align or scramble: cogs-dominant conjunction only (no OrFilter)."""
    assert isinstance(event.target_query, Query)
    burst_filters = event.target_query.filters
    assert not any(isinstance(f, OrFilter) for f in burst_filters)
    burst_gvfs = [f for f in burst_filters if isinstance(f, GameValueFilter)]
    assert len(burst_gvfs) == 2
    burst_weight_mins: set[tuple[tuple[float, ...], int]] = set()
    for filter_cfg in burst_gvfs:
        value = filter_cfg.value
        assert isinstance(value, SumGameValue)
        assert value.weights is not None
        burst_weight_mins.add((tuple(value.weights), filter_cfg.min))
    assert burst_weight_mins == {((1.0, -2.0), 0), ((1.0, 1.0), 1)}


def _representative_fallback_tier_name(
    names: set[str],
    innermost_name: str,
) -> str:
    """Pick an outer greedy tier such as `neutral_to_clips_balanced_r120`."""
    prefix = f"{innermost_name}_r"
    r_names = [n for n in names if n.startswith(prefix)]
    assert r_names, f"expected keys under {innermost_name!r}"
    return max(r_names, key=lambda n: int(n.split("_r")[-1]))


def _balanced_adaptive_gate_sigs(
    event,
) -> set[frozenset[tuple[tuple[float, ...], int]]]:
    assert isinstance(event.target_query, Query)
    ors = [f for f in event.target_query.filters if isinstance(f, OrFilter)]
    assert len(ors) == 2
    return {frozenset(_or_neg_gate_signatures(o)) for o in ors}


def _burst_adaptive_weight_mins(event) -> set[tuple[tuple[float, ...], int]]:
    assert isinstance(event.target_query, Query)
    burst_filters = event.target_query.filters
    assert not any(isinstance(f, OrFilter) for f in burst_filters)
    burst_gvfs = [f for f in burst_filters if isinstance(f, GameValueFilter)]
    assert len(burst_gvfs) == 2
    burst_weight_mins: set[tuple[tuple[float, ...], int]] = set()
    for filter_cfg in burst_gvfs:
        value = filter_cfg.value
        assert isinstance(value, SumGameValue)
        assert value.weights is not None
        burst_weight_mins.add((tuple(value.weights), filter_cfg.min))
    return burst_weight_mins


def _assert_adaptive_burst_cogs_dominant_game_value_shape(event) -> None:
    """Map-wide SumGameValue filters on burst lanes (neutral and scramble)."""
    assert isinstance(event.target_query, Query)
    gfs = [f for f in event.target_query.filters if isinstance(f, GameValueFilter)]
    assert len(gfs) == 2
    sums: list[SumGameValue] = []
    for filter_cfg in gfs:
        value = filter_cfg.value
        assert isinstance(value, SumGameValue)
        sums.append(value)
    weight_sets = {tuple(game_value.weights) for game_value in sums if game_value.weights is not None}
    assert (1.0, -2.0) in weight_sets
    assert (1.0, 1.0) in weight_sets
    for s in sums:
        assert len(s.values) == 2
        assert all(isinstance(v, QueryCountValue) for v in s.values)
    for f in gfs:
        s = f.value
        assert isinstance(s, SumGameValue)
        w = tuple(s.weights) if s.weights is not None else ()
        if w == (1.0, -2.0):
            assert f.min == 0
        elif w == (1.0, 1.0):
            assert f.min == 1


def test_adaptive_clips_burst_family_has_cogs_dominant_filters() -> None:
    env = make_machina1_mission().with_variants([AdaptiveClipsVariant()]).make_env()
    for key in ("neutral_to_clips_burst_1", "cogs_to_neutral_burst_1"):
        _assert_adaptive_burst_cogs_dominant_game_value_shape(env.game.events[key])


def test_adaptive_clips_balanced_and_burst_encode_mutual_exclusion_in_queries() -> None:
    """Balanced vs burst families encode band mutual exclusion in config only.

    The same adaptive filter stack is attached to both align (neutral_to_clips)
    and scramble (cogs_to_neutral) event families.

    Balanced events add two anyOf gates (OrFilter), each anyOf([isNot(A), isNot(B)]):
    one gate negates the cogs-dominant linear predicate together with the shared
    nonempty-junction predicate; the other negates the clips-dominant linear
    predicate with the same nonempty guard. Targets must satisfy both gates, so
    neither pure dominance band applies without the balanced nonempty escape.

    Burst events add only the cogs-dominant conjunction: two positive
    GameValueFilters (cogs-dominant linear and nonempty). There is no clips-dominant
    burst family and no OrFilter on the burst query.
    """
    env = make_machina1_mission().with_variants([AdaptiveClipsVariant()]).make_env()
    for balanced_key, burst_key in (
        ("neutral_to_clips_balanced", "neutral_to_clips_burst_1"),
        ("cogs_to_neutral_balanced", "cogs_to_neutral_burst_1"),
    ):
        _assert_adaptive_balanced_mutual_exclusion_gates(
            env.game.events[balanced_key],
        )
        _assert_adaptive_burst_cogs_dominant_conjunction(
            env.game.events[burst_key],
        )


def test_adaptive_clips_zero_ship_and_no_clips_still_remove_events() -> None:
    env = make_machina1_mission().with_variants([ClipsVariant(num_ships=0), AdaptiveClipsVariant()]).make_env()
    assert count_clips_ships_in_map_config(env.game.map_builder) == 0
    assert not _clips_event_group(env.game.events, "neutral_to_clips")
    assert not _clips_event_group(env.game.events, "cogs_to_neutral")


def test_no_clips_and_adaptive_clips_remove_events() -> None:
    env = make_machina1_mission().with_variants([AdaptiveClipsVariant(), NoClipsVariant()]).make_env()
    assert count_clips_ships_in_map_config(env.game.map_builder) == 0
    assert not _clips_event_group(env.game.events, "neutral_to_clips")
    assert not _clips_event_group(env.game.events, "cogs_to_neutral")


def test_adaptive_clips_has_no_clips_dominant_family_names() -> None:
    env = make_machina1_mission().with_variants([AdaptiveClipsVariant()]).make_env()
    for name in env.game.events:
        assert "clips_dominant" not in name


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
    assert align_steps[1] - align_steps[0] == 200
    assert scramble_steps[1] - scramble_steps[0] == 200


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
