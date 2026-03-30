from cogames.games.cogs_vs_clips.evals.cognitive_substrate.exploration import (
    EVAL_MISSIONS as EXPLORATION_EVAL_MISSIONS,
)
from cogames.games.cogs_vs_clips.evals.cognitive_substrate.memory import (
    EVAL_MISSIONS as MEMORY_EVAL_MISSIONS,
)
from cogames.games.cogs_vs_clips.evals.cognitive_substrate.planning import (
    EVAL_MISSIONS as PLANNING_EVAL_MISSIONS,
)

EVAL_MISSIONS = [
    *MEMORY_EVAL_MISSIONS,
    *EXPLORATION_EVAL_MISSIONS,
    *PLANNING_EVAL_MISSIONS,
]

CATEGORY_MISSIONS = {
    "memory": MEMORY_EVAL_MISSIONS,
    "exploration": EXPLORATION_EVAL_MISSIONS,
    "planning": PLANNING_EVAL_MISSIONS,
}
