"""Role-specific tutorial eval missions — one per role."""

from __future__ import annotations

from cogames.games.cogs_vs_clips.missions.mission import CvCMission
from cogames.games.cogs_vs_clips.missions.tutorial import make_tutorial_mission

_ROLES = ["aligner", "miner", "scout", "scrambler"]


def _make_role_tutorial(role: str) -> CvCMission:
    base = make_tutorial_mission()
    return base.with_variants([role]).model_copy(update={"name": f"{role}_tutorial", "num_agents": 4})


EVAL_MISSIONS: list[CvCMission] = [_make_role_tutorial(role) for role in _ROLES]
