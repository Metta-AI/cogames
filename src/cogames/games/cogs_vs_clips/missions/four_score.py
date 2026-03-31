"""Four Score mission: machina1 gameplay + 4-team corner compounds."""

from __future__ import annotations

from pydantic import Field

from cogames.games.cogs_vs_clips.game.teams.four_corners import FourCornersVariant
from cogames.games.cogs_vs_clips.game.terrain.four_corners import build_four_corners_map
from cogames.games.cogs_vs_clips.missions.mission import CvCMission
from mettagrid.mapgen.mapgen import MapGenConfig


class FourScoreMission(CvCMission):
    """Four Score: machina1 gameplay + 4-team corner compounds, no clips."""

    name: str = "four_score"
    description: str = "Multi-team corner bases competing for junction control."
    map_builder: MapGenConfig = Field(default_factory=lambda: build_four_corners_map(4, 8))
    num_cogs: int = 32
    min_cogs: int = 4
    max_cogs: int = 80
    max_steps: int = 10000
    default_variant: str = "machina_1"
    num_agents: int = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._base_variants["four_corners"] = FourCornersVariant()
