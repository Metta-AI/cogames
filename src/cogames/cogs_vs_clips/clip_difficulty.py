from __future__ import annotations

from typing import Final, Optional

from cogames.cogs_vs_clips.mission import CvCMission
from cogames.cogs_vs_clips.ships import remove_clips_ships_from_map_config
from cogames.core import CoGameMissionVariant

MEDIUM_CLIPS_END: Final[int] = 300


class CogsGuardDifficulty(CoGameMissionVariant):
    name: str
    description: str = ""
    disable_clips: bool = False
    scramble_end: Optional[int] = None
    align_end: Optional[int] = None
    presence_end: Optional[int] = None

    def modify_mission(self, mission: CvCMission) -> None:
        if self.disable_clips:
            mission.site = mission.site.model_copy(
                deep=True,
                update={"map_builder": remove_clips_ships_from_map_config(mission.site.map_builder)},
            )
            mission.clips.disabled = False
            return

        mission.clips.disabled = False
        mission.clips.scramble_end = self.scramble_end
        mission.clips.align_end = self.align_end
        mission.clips.presence_end = self.presence_end


EASY = CogsGuardDifficulty(
    name="easy",
    description="No clips events.",
    disable_clips=True,
)

MEDIUM = CogsGuardDifficulty(
    name="medium",
    description="A few early clips events, then none.",
    scramble_end=MEDIUM_CLIPS_END,
    align_end=MEDIUM_CLIPS_END,
    presence_end=MEDIUM_CLIPS_END,
)

HARD = CogsGuardDifficulty(
    name="hard",
    description="Standard clips event system.",
)

COGSGUARD_DIFFICULTIES: tuple[CogsGuardDifficulty, ...] = (EASY, MEDIUM, HARD)


def get_cogsguard_difficulty(name: str) -> CogsGuardDifficulty:
    for difficulty in COGSGUARD_DIFFICULTIES:
        if difficulty.name == name:
            return difficulty
    available = ", ".join(d.name for d in COGSGUARD_DIFFICULTIES)
    raise ValueError(f"Unknown difficulty '{name}'. Available difficulties: {available}")
