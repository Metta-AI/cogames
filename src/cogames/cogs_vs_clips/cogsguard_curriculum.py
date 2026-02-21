from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from cogames.cogs_vs_clips.mission import CvCMission
from cogames.cogs_vs_clips.reward_variants import AVAILABLE_REWARD_VARIANTS
from cogames.cogs_vs_clips.variants import HIDDEN_VARIANTS, VARIANTS
from cogames.core import CoGameMissionVariant


@dataclass(frozen=True)
class EventProfile:
    name: str
    clips_overrides: dict[str, object]
    weather_overrides: dict[str, object]


COGSGUARD_FIXED_MAPS: list[str] = [
    "machina_100_stations.map",
    "machina_200_stations.map",
    "cave_base_50.map",
    "vanilla_large.map",
]

DEFAULT_EVENT_PROFILE = EventProfile("events_baseline", {}, {})
COGSGUARD_EVENT_PROFILES: list[EventProfile] = [
    DEFAULT_EVENT_PROFILE,
    EventProfile(
        "events_fast_clips_short_day",
        {
            "initial_clips_start": 5,
            "initial_clips_spots": 2,
            "scramble_start": 25,
            "scramble_interval": 50,
            "scramble_radius": 35,
            "align_start": 50,
            "align_interval": 50,
        },
        {"day_length": 100},
    ),
    EventProfile(
        "events_slow_clips_long_day",
        {
            "initial_clips_start": 50,
            "initial_clips_spots": 1,
            "scramble_start": 200,
            "scramble_interval": 200,
            "scramble_radius": 15,
            "align_start": 300,
            "align_interval": 200,
        },
        {"day_length": 400},
    ),
    EventProfile(
        "events_no_clips",
        {"disabled": True},
        {"day_length": 200},
    ),
]


def normalize_variant_names(variants: str | Sequence[str] | None) -> list[str]:
    if variants is None:
        return []
    if isinstance(variants, str):
        if variants.startswith("["):
            parsed = json.loads(variants)
            if isinstance(parsed, list):
                return [str(name) for name in parsed]
        return [variants]
    return list(variants)


def split_variants(
    variants: str | Sequence[str] | None,
) -> tuple[list[CoGameMissionVariant], list[str]]:
    if variants is None:
        names: list[str] = []
    else:
        names = normalize_variant_names(variants)
    all_variants = {variant.name: variant for variant in [*VARIANTS, *HIDDEN_VARIANTS]}
    reward_variants = set(AVAILABLE_REWARD_VARIANTS)

    resolved: list[CoGameMissionVariant] = []
    resolved_rewards: list[str] = []
    unknown: list[str] = []
    for name in names:
        if name in reward_variants:
            resolved_rewards.append(name)
            continue
        variant = all_variants.get(name)
        if variant is None:
            unknown.append(name)
            continue
        resolved.append(variant)

    if unknown:
        available_mission = ", ".join(v.name for v in VARIANTS)
        available_reward = ", ".join(AVAILABLE_REWARD_VARIANTS)
        missing = ", ".join(unknown)
        raise ValueError(
            f"Unknown variant(s): {missing}. Mission variants: {available_mission}. "
            f"Reward variants: {available_reward}."
        )

    return resolved, resolved_rewards


def resolve_event_profiles(event_profiles: Sequence[EventProfile] | None) -> list[EventProfile]:
    if event_profiles is None:
        return [DEFAULT_EVENT_PROFILE]
    return list(event_profiles)


def filter_compatible_variants(
    mission: CvCMission, variants: Sequence[CoGameMissionVariant]
) -> list[CoGameMissionVariant]:
    return [variant for variant in variants if variant.compat(mission)]
