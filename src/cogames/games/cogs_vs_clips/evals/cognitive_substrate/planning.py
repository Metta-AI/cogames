from __future__ import annotations

from cogames.games.cogs_vs_clips.evals.cognitive_substrate.common import (
    CognitiveSubstrateMission,
    rotated_options,
    success_handler,
)
from cogames.games.cogs_vs_clips.evals.cognitive_substrate.map_builder import ChoiceAsciiMapBuilder
from mettagrid.config.filter import AnyFilter, actorHasTag, isNot
from mettagrid.config.handler_config import Handler
from mettagrid.config.mettagrid_config import GridObjectConfig
from mettagrid.config.mutation import EntityTarget, addTag, logActorAgentStat, logStatToGame

_PLANNING_CHAR_TO_MAP_NAME = {
    "K": "key",
    "S": "switch",
    "D": "door",
    "G": "goal",
}


def _tag_progression_handler(name: str, required_tag: str | None, gained_tag: str, stat_name: str) -> Handler:
    filters: list[AnyFilter] = [isNot(actorHasTag(gained_tag))]
    if required_tag is not None:
        filters.insert(0, actorHasTag(required_tag))
    return Handler(
        name=name,
        filters=filters,
        mutations=[
            addTag(gained_tag, target=EntityTarget.ACTOR),
            logActorAgentStat(stat_name),
            logStatToGame(stat_name),
        ],
    )


def _planning_objects() -> dict[str, GridObjectConfig]:
    solve = success_handler(actorHasTag("state:door"))
    solve.name = "solve"
    return {
        "key": GridObjectConfig(
            name="key",
            map_name="key",
            tags=["planning:key"],
            on_use_handler=_tag_progression_handler("pickup", None, "state:key", "planning.key"),
        ),
        "switch": GridObjectConfig(
            name="switch",
            map_name="switch",
            tags=["planning:switch"],
            on_use_handler=_tag_progression_handler("flip", "state:key", "state:switch", "planning.switch"),
        ),
        "door": GridObjectConfig(
            name="door",
            map_name="door",
            tags=["planning:door"],
            on_use_handler=_tag_progression_handler("unlock", "state:switch", "state:door", "planning.door"),
        ),
        "goal": GridObjectConfig(
            name="goal",
            map_name="goal",
            tags=["planning:goal"],
            on_use_handler=solve,
        ),
    }


_PLANNING_EASY_MAP = [
    "#####################",
    "#.........K.........#",
    "#########.###########",
    "#.........#.........#",
    "#.........#.........#",
    "#S........@........D#",
    "#.........#.........#",
    "#.........#.........#",
    "#########.###########",
    "#.........G.........#",
    "#####################",
]

_PLANNING_MEDIUM_MAP = [
    "#############################",
    "#............K............###",
    "#############.###############",
    "#.............#...........###",
    "#.............#...........###",
    "#.............#...........###",
    "#S............@...........D##",
    "#.............#...........###",
    "#.............#...........###",
    "#.............#...........###",
    "#############.###############",
    "#.............G...........###",
    "#############################",
]

_PLANNING_HARD_MAP = [
    "#####################################",
    "#................K.................##",
    "#################.###################",
    "#.................#.................#",
    "#.................#.................#",
    "#.................#.................#",
    "#S................@................D#",
    "#.................#.................#",
    "#.................#.................#",
    "#.................#.................#",
    "#################.###################",
    "#.................G.................#",
    "#####################################",
]


EVAL_MISSIONS = [
    CognitiveSubstrateMission(
        name="planning_unlock_chain_easy",
        description="Three-step unlock chain on a short cross map.",
        map_builder=ChoiceAsciiMapBuilder.Config(
            map_options=rotated_options(_PLANNING_EASY_MAP),
            char_to_map_name=_PLANNING_CHAR_TO_MAP_NAME,
        ),
        object_configs=_planning_objects(),
        game_tags=["state:door", "state:key", "state:solved", "state:switch"],
        render_symbols={"key": "K", "switch": "S", "door": "D", "goal": "G"},
        max_steps=192,
    ),
    CognitiveSubstrateMission(
        name="planning_unlock_chain_medium",
        description="Longer unlock chain with higher backtracking cost.",
        map_builder=ChoiceAsciiMapBuilder.Config(
            map_options=rotated_options(_PLANNING_MEDIUM_MAP),
            char_to_map_name=_PLANNING_CHAR_TO_MAP_NAME,
        ),
        object_configs=_planning_objects(),
        game_tags=["state:door", "state:key", "state:solved", "state:switch"],
        render_symbols={"key": "K", "switch": "S", "door": "D", "goal": "G"},
        max_steps=320,
    ),
    CognitiveSubstrateMission(
        name="planning_unlock_chain_hard",
        description="Longest unlock chain with large branch separation.",
        map_builder=ChoiceAsciiMapBuilder.Config(
            map_options=rotated_options(_PLANNING_HARD_MAP),
            char_to_map_name=_PLANNING_CHAR_TO_MAP_NAME,
        ),
        object_configs=_planning_objects(),
        game_tags=["state:door", "state:key", "state:solved", "state:switch"],
        render_symbols={"key": "K", "switch": "S", "door": "D", "goal": "G"},
        max_steps=448,
    ),
]
