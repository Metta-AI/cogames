"""Weather system events for CogsGuard missions.

Day/night cycle that applies resource deltas to entities at regular intervals.
"""

from pydantic import Field

from mettagrid.base_config import Config
from mettagrid.config.event_config import EventConfig, periodic
from mettagrid.config.mutation import updateTarget
from mettagrid.config.tag import typeTag


class WeatherConfig(Config):
    """Configuration for day/night weather cycle."""

    day_length: int = Field(default=200)
    day_deltas: dict[str, int] = Field(default_factory=lambda: {"solar": 3})
    night_deltas: dict[str, int] = Field(default_factory=lambda: {"solar": 1})
    target_tag: str = Field(default="agent")

    def events(self, max_steps: int) -> dict[str, EventConfig]:
        """Create weather events for a mission.

        Returns:
            Dictionary of event name to EventConfig.
        """
        events: dict[str, EventConfig] = {}
        tag = typeTag(self.target_tag)
        half = self.day_length // 2

        def _merge(apply: dict[str, int], reverse: dict[str, int]) -> dict[str, int]:
            keys = set(apply) | set(reverse)
            return {k: apply.get(k, 0) - reverse.get(k, 0) for k in keys}

        # Dawn: reverse night deltas, apply day deltas
        events["weather_day"] = EventConfig(
            name="weather_day",
            target_tag=tag,
            timesteps=periodic(start=0, period=self.day_length, end=max_steps),
            mutations=[updateTarget(_merge(self.day_deltas, self.night_deltas))],
        )

        # Dusk: reverse day deltas, apply night deltas
        events["weather_night"] = EventConfig(
            name="weather_night",
            target_tag=tag,
            timesteps=periodic(start=half, period=self.day_length, end=max_steps),
            mutations=[updateTarget(_merge(self.night_deltas, self.day_deltas))],
        )

        return events
