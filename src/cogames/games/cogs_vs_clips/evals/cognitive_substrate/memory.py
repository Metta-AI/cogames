from __future__ import annotations

from cogames.games.cogs_vs_clips.evals.cognitive_substrate.common import CognitiveSubstrateMission, success_handler
from cogames.games.cogs_vs_clips.evals.cognitive_substrate.map_builder import ChoiceAsciiMapBuilder
from mettagrid.config.handler_config import Handler
from mettagrid.config.mettagrid_config import GridObjectConfig

_CHAR_TO_MAP_NAME = {
    "1": "cue_1",
    "2": "cue_2",
    "3": "cue_3",
    "4": "cue_4",
    "R": "memory_goal_reward",
    "D": "memory_goal_decoy",
}


def _memory_objects() -> dict[str, GridObjectConfig]:
    cue_configs = {
        f"cue_{index}": GridObjectConfig(name=f"cue_{index}", map_name=f"cue_{index}", tags=["memory:cue"])
        for index in range(1, 5)
    }
    return {
        **cue_configs,
        "memory_goal_reward": GridObjectConfig(
            name="memory_goal",
            map_name="memory_goal_reward",
            tags=["memory:goal"],
            on_use_handler=success_handler().model_copy(update={"name": "solve"}),
        ),
        "memory_goal_decoy": GridObjectConfig(
            name="memory_goal",
            map_name="memory_goal_decoy",
            tags=["memory:goal"],
            on_use_handler=Handler(name="touch"),
        ),
    }


def _memory_variants(base_map: list[str], cue_count: int) -> list[list[list[str]]]:
    cue_chars = [str(index) for index in range(1, cue_count + 1)]
    options: list[list[list[str]]] = []
    reward_slots = [idx for idx, char in enumerate(base_map[-2]) if char == "G"]
    for cue_index, cue_char in enumerate(cue_chars):
        current = []
        for row in base_map:
            row_value = row.replace("C", cue_char)
            current.append(row_value)
        reward_row = list(current[-2])
        for goal_index, slot in enumerate(reward_slots):
            reward_row[slot] = "R" if goal_index == cue_index else "D"
        current[-2] = "".join(reward_row)
        options.append([list(row) for row in current])
    return options


_MEMORY_EASY_MAP = [
    "#####################",
    "#@.................##",
    "#C################.##",
    "##################.##",
    "#..................##",
    "#.###################",
    "#...................#",
    "#..G.............G..#",
    "#####################",
]

_MEMORY_MEDIUM_MAP = [
    "#############################",
    "#@........................###",
    "#C#######################.###",
    "#########################.###",
    "#.........................###",
    "#.###########################",
    "#..........................##",
    "##########################.##",
    "#..........................##",
    "#..G..........G..........G.##",
    "#############################",
]

_MEMORY_HARD_MAP = [
    "#####################################",
    "#@................................###",
    "#C###############################.###",
    "#################################.###",
    "#.................................###",
    "#.###################################",
    "#..................................##",
    "##################################.##",
    "#..................................##",
    "#..................................##",
    "#..G........G........G........G....##",
    "#####################################",
]


EVAL_MISSIONS = [
    CognitiveSubstrateMission(
        name="memory_mystery_path_easy",
        description="Delayed cue recall with two identical goal branches.",
        map_builder=ChoiceAsciiMapBuilder.Config(
            map_options=_memory_variants(_MEMORY_EASY_MAP, cue_count=2),
            char_to_map_name=_CHAR_TO_MAP_NAME,
        ),
        object_configs=_memory_objects(),
        game_tags=["state:solved"],
        render_symbols={
            "cue_1": "1",
            "cue_2": "2",
            "memory_goal_reward": "G",
            "memory_goal_decoy": "G",
        },
        max_steps=96,
    ),
    CognitiveSubstrateMission(
        name="memory_mystery_path_medium",
        description="Longer delayed cue recall with three identical goal branches.",
        map_builder=ChoiceAsciiMapBuilder.Config(
            map_options=_memory_variants(_MEMORY_MEDIUM_MAP, cue_count=3),
            char_to_map_name=_CHAR_TO_MAP_NAME,
        ),
        object_configs=_memory_objects(),
        game_tags=["state:solved"],
        render_symbols={
            "cue_1": "1",
            "cue_2": "2",
            "cue_3": "3",
            "memory_goal_reward": "G",
            "memory_goal_decoy": "G",
        },
        max_steps=144,
    ),
    CognitiveSubstrateMission(
        name="memory_mystery_path_hard",
        description="Long-horizon delayed cue recall with four identical goal branches.",
        map_builder=ChoiceAsciiMapBuilder.Config(
            map_options=_memory_variants(_MEMORY_HARD_MAP, cue_count=4),
            char_to_map_name=_CHAR_TO_MAP_NAME,
        ),
        object_configs=_memory_objects(),
        game_tags=["state:solved"],
        render_symbols={
            "cue_1": "1",
            "cue_2": "2",
            "cue_3": "3",
            "cue_4": "4",
            "memory_goal_reward": "G",
            "memory_goal_decoy": "G",
        },
        max_steps=256,
    ),
]
