from __future__ import annotations

from cogames.cogs_vs_clips.mission import CvCMission
from cogames.cogs_vs_clips.tutorials.aligner_tutorial import AlignerTutorialMission
from cogames.cogs_vs_clips.tutorials.miner_tutorial import MinerTutorialMission
from cogames.cogs_vs_clips.tutorials.scout_tutorial import ScoutTutorialMission
from cogames.cogs_vs_clips.tutorials.scrambler_tutorial import ScramblerTutorialMission

EVAL_MISSIONS: list[CvCMission] = [
    AlignerTutorialMission,
    MinerTutorialMission,
    ScoutTutorialMission,
    ScramblerTutorialMission,
]
