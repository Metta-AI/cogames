"""Days variant: weather day/night cycle events."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from cogames.cogs_vs_clips.solar import SolarVariant
from cogames.core import CoGameMissionVariant, Deps

if TYPE_CHECKING:
    from cogames.cogs_vs_clips.mission import CvCMission
    from mettagrid.config.mettagrid_config import MettaGridConfig


class DaysVariant(CoGameMissionVariant):
    name: str = "days"

    @override
    def dependencies(self) -> Deps:
        return Deps(required=[SolarVariant])

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        weather_events = mission.weather.events(max_steps=mission.max_steps)
        overlap = set(env.game.events) & set(weather_events)
        if overlap:
            raise ValueError(f"Overlapping event keys between existing events and weather: {overlap}")
        env.game.events.update(weather_events)
