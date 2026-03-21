from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import Field

from cogames.cogs_vs_clips.config import CvCConfig
from cogames.cogs_vs_clips.stations import CvCStationConfig
from mettagrid.config.mettagrid_config import GridObjectConfig
from mettagrid.config.territory_config import TerritoryControlConfig

if TYPE_CHECKING:
    from cogames.cogs_vs_clips.team import TeamConfig


class CvCShipConfig(CvCStationConfig):
    """Simple clips ship station used as the clips network anchor."""

    control_range: int = Field(default=CvCConfig.TERRITORY_CONTROL_RADIUS, description="Range for territory control")

    def station_cfg(self, team: TeamConfig, map_name: Optional[str] = None) -> GridObjectConfig:
        return GridObjectConfig(
            name="ship",
            map_name=map_name or "ship",
            render_symbol="🚀",
            tags=[team.team_tag()],
            territory_controls=[
                TerritoryControlConfig(territory="team_territory", strength=self.control_range),
            ],
        )
