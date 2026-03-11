"""Days variant: weather day/night cycle events."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from pydantic import Field

from cogames.cogs_vs_clips.solar import SolarVariant
from cogames.core import CoGameMissionVariant, Deps
from cogames.variants import ResolvedDeps
from mettagrid.base_config import Config
from mettagrid.config.event_config import EventConfig, periodic
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.config.mutation import updateTarget
from mettagrid.config.query import query

if TYPE_CHECKING:
    from cogames.core import CoGameMission


class DayConfig(Config):
    """Configuration for the day/night solar cycle."""

    day_length: int = Field(default=200)
    day_solar: int = Field(default=3)
    night_solar: int = Field(default=1)


class DaysVariant(CoGameMissionVariant):
    name: str = "days"

    days_config: DayConfig = Field(default_factory=DayConfig)

    @override
    def dependencies(self) -> Deps:
        return Deps(required=[SolarVariant])

    @override
    def configure(self, deps: ResolvedDeps) -> None:
        deps.required(SolarVariant)

    @override
    def modify_env(self, mission: CoGameMission, env: MettaGridConfig) -> None:
        cfg = self.days_config
        for agent in env.game.agents:
            agent.inventory.initial["solar"] = cfg.night_solar
        delta = cfg.day_solar - cfg.night_solar

        env.game.events["day"] = EventConfig(
            name="day",
            target_query=query("type:agent"),
            timesteps=periodic(start=0, period=cfg.day_length, end=env.game.max_steps),
            mutations=[updateTarget({"solar": delta})],
        )
        env.game.events["night"] = EventConfig(
            name="night",
            target_query=query("type:agent"),
            timesteps=periodic(start=cfg.day_length // 2, period=cfg.day_length, end=env.game.max_steps),
            mutations=[updateTarget({"solar": -delta})],
        )
