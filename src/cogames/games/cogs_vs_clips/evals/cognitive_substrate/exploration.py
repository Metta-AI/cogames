from __future__ import annotations

from cogames.games.cogs_vs_clips.evals.cognitive_substrate.common import CognitiveSubstrateMission, success_handler
from mettagrid.config.mettagrid_config import GridObjectConfig
from mettagrid.map_builder.maze import MazePrimMapBuilder


def _exploration_goal() -> dict[str, GridObjectConfig]:
    return {
        "hub": GridObjectConfig(
            name="goal",
            map_name="hub",
            tags=["exploration:goal"],
            on_use_handlers={"solve": success_handler()},
        )
    }


EVAL_MISSIONS = [
    CognitiveSubstrateMission(
        name="exploration_sparse_search_easy",
        description="Sparse terminal maze search with a small random maze.",
        map_builder=MazePrimMapBuilder.Config(width=15, height=15, start_pos=(1, 1), end_pos=(13, 13), seed=None),
        object_configs=_exploration_goal(),
        game_tags=["state:solved"],
        render_symbols={"hub": "G"},
        max_steps=192,
    ),
    CognitiveSubstrateMission(
        name="exploration_sparse_search_medium",
        description="Sparse terminal maze search with a medium random maze.",
        map_builder=MazePrimMapBuilder.Config(width=23, height=23, start_pos=(1, 1), end_pos=(21, 21), seed=None),
        object_configs=_exploration_goal(),
        game_tags=["state:solved"],
        render_symbols={"hub": "G"},
        max_steps=448,
    ),
    CognitiveSubstrateMission(
        name="exploration_sparse_search_hard",
        description="Sparse terminal maze search with a large random maze.",
        map_builder=MazePrimMapBuilder.Config(width=31, height=31, start_pos=(1, 1), end_pos=(29, 29), seed=None),
        object_configs=_exploration_goal(),
        game_tags=["state:solved"],
        render_symbols={"hub": "G"},
        max_steps=896,
    ),
]
