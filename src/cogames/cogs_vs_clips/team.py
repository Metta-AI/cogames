"""Team configuration for CogsGuard missions.

Teams are named collectives (resource pools) shared by agents.
"""

from pydantic import Field

from cogames.cogs_vs_clips.config import CvCConfig
from mettagrid.base_config import Config
from mettagrid.config.mettagrid_config import (
    CollectiveConfig,
    InventoryConfig,
    ResourceLimitsConfig,
)


class CogTeam(Config):
    """Configuration for a cogs team."""

    name: str = Field(default="cogs", description="Team name used for collectives and alignment")
    short_name: str = Field(default="c", description="Short prefix used for map object names")
    wealth: int = Field(default=1, description="Wealth multiplier for initial resources")
    initial_hearts: int | None = Field(default=None, description="Override initial hearts (default: 5 * wealth)")
    num_agents: int = Field(default=8, ge=1, description="Number of agents in the team")

    def collective_config(self) -> CollectiveConfig:
        """Create a CollectiveConfig for this team.

        Returns:
            CollectiveConfig with resource limits and initial inventory.
        """
        hearts = self.initial_hearts if self.initial_hearts is not None else 5 * self.wealth
        return CollectiveConfig(
            name=self.name,
            inventory=InventoryConfig(
                limits={
                    "resources": ResourceLimitsConfig(min=10000, resources=CvCConfig.ELEMENTS),
                    "hearts": ResourceLimitsConfig(min=65535, resources=["heart"]),
                },
                initial={
                    "carbon": 10 * self.wealth,
                    "oxygen": 10 * self.wealth,
                    "germanium": 10 * self.wealth,
                    "silicon": 10 * self.wealth,
                    "heart": hearts,
                },
            ),
        )
