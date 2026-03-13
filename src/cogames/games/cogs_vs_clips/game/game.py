from __future__ import annotations

from cogames.game import CoGame, register_game
from cogames.games.cogs_vs_clips.evals.diagnostic_evals import DIAGNOSTIC_EVALS
from cogames.games.cogs_vs_clips.evals.integrated_evals import (
    EVAL_MISSIONS as INTEGRATED_EVAL_MISSIONS,
)
from cogames.games.cogs_vs_clips.evals.spanning_evals import (
    EVAL_MISSIONS as SPANNING_EVAL_MISSIONS,
)
from cogames.games.cogs_vs_clips.game import _get_all_variants
from cogames.games.cogs_vs_clips.missions.arena import make_basic_mission
from cogames.games.cogs_vs_clips.missions.empty import make_empty_mission
from cogames.games.cogs_vs_clips.missions.machina_1 import make_machina1_mission
from cogames.games.cogs_vs_clips.missions.mission import CvCMission
from cogames.games.cogs_vs_clips.missions.tutorial import make_tutorial_mission


class CvCGame(CoGame):
    """Cogs vs Clips game. Subclasses CoGame with CvC missions and variants."""

    def __init__(self) -> None:
        eval_missions: list[CvCMission] = []
        eval_missions.extend(INTEGRATED_EVAL_MISSIONS)
        eval_missions.extend(SPANNING_EVAL_MISSIONS)
        eval_missions.extend(m() for m in DIAGNOSTIC_EVALS)  # type: ignore[call-arg]

        super().__init__(
            name="cogs_vs_clips",
            missions=[
                make_empty_mission(),
                make_machina1_mission(),
                make_basic_mission(),
                make_tutorial_mission(),
            ],
            variants=_get_all_variants(),
            eval_missions=eval_missions,
        )


# Register for CLI --game resolution
register_game(CvCGame())
