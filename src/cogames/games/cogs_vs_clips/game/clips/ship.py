from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from cogames.core import CvCStationConfig
from cogames.games.cogs_vs_clips.missions.terrain import MachinaArenaConfig
from mettagrid.config.mettagrid_config import GridObjectConfig
from mettagrid.map_builder.ascii import AsciiMapBuilderConfig
from mettagrid.map_builder.map_builder import AnyMapBuilderConfig, MapBuilderConfig
from mettagrid.mapgen.mapgen import MapGenConfig
from mettagrid.mapgen.scene import SceneConfig

if TYPE_CHECKING:
    from cogames.games.cogs_vs_clips.game.teams import TeamConfig

CLIPS_SHIP_MAP_NAME = "clips:ship"
DEFAULT_CLIPS_SHIP_COUNT = 4


class CvCShipConfig(CvCStationConfig):
    """Simple clips ship station used as the clips network anchor."""

    def station_cfg(
        self,
        team: TeamConfig,
        map_name: Optional[str] = None,
    ) -> GridObjectConfig:
        return GridObjectConfig(
            name="ship",
            map_name=map_name or "ship",
            tags=[team.team_tag()],
        )


def is_clips_ship_map_name(name: str) -> bool:
    return name == CLIPS_SHIP_MAP_NAME or name.startswith(f"{CLIPS_SHIP_MAP_NAME}:")


def clips_ship_map_names_in_map_config(config: MapBuilderConfig | SceneConfig) -> list[str]:
    """Return clips ship map names present in a map builder config, one entry per ship placement."""
    if isinstance(config, MapGenConfig):
        if config.instance is None:
            return []
        names = clips_ship_map_names_in_map_config(config.instance)
        return names * (config.instances or 1)

    if isinstance(config, MachinaArenaConfig):
        corner = [name for name, _ in config.map_corner_placements if is_clips_ship_map_name(name)]
        perimeter: list[str] = []
        for name, count in config.map_perimeter_placements:
            if is_clips_ship_map_name(name):
                perimeter.extend([name] * max(0, count))
        return [*corner, *perimeter]

    if isinstance(config, AsciiMapBuilderConfig):
        return [
            map_name
            for row in config.map_data
            for cell in row
            for map_name in [config.char_to_map_name.get(cell, "")]
            if is_clips_ship_map_name(map_name)
        ]

    return []


def remove_clips_ships_from_map_config(config: AnyMapBuilderConfig | SceneConfig) -> AnyMapBuilderConfig | SceneConfig:
    """Return a copy of *config* with all clips ship placements removed."""
    if isinstance(config, AsciiMapBuilderConfig):
        return _remove_from_ascii(config)

    if isinstance(config, MachinaArenaConfig):
        return _remove_from_machina(config)

    if isinstance(config, MapGenConfig):
        if config.instance is None:
            return config.model_copy(deep=True)
        updated = remove_clips_ships_from_map_config(config.instance)
        return config.model_copy(deep=True, update={"instance": updated})

    return config.model_copy(deep=True)


def _remove_from_machina(config: MachinaArenaConfig) -> MachinaArenaConfig:
    return config.model_copy(
        update={
            "map_corner_placements": [p for p in config.map_corner_placements if not is_clips_ship_map_name(p[0])],
            "map_perimeter_placements": [
                p for p in config.map_perimeter_placements if not is_clips_ship_map_name(p[0])
            ],
        }
    )


def add_clips_ships_to_map_config(
    config: AnyMapBuilderConfig | SceneConfig,
    num_ships: int,
) -> AnyMapBuilderConfig | SceneConfig:
    """Return a copy of *config* with clips ship corner placements added."""
    if isinstance(config, MachinaArenaConfig):
        placements = [(f"{CLIPS_SHIP_MAP_NAME}:{i}", i) for i in range(num_ships)]
        return config.model_copy(
            update={"map_corner_placements": [*config.map_corner_placements, *placements]},
        )

    if isinstance(config, MapGenConfig):
        if config.instance is None:
            return config.model_copy(deep=True)
        updated = add_clips_ships_to_map_config(config.instance, num_ships)
        return config.model_copy(deep=True, update={"instance": updated})

    return config.model_copy(deep=True)


def set_clips_ships_in_map_config(
    config: AnyMapBuilderConfig | SceneConfig,
    num_ships: Optional[int],
) -> AnyMapBuilderConfig | SceneConfig:
    """Return a copy of *config* with the resolved clips ship count applied."""
    if isinstance(config, MapGenConfig):
        if config.instance is None:
            return config.model_copy(deep=True)
        updated = set_clips_ships_in_map_config(config.instance, num_ships)
        return config.model_copy(deep=True, update={"instance": updated})

    if num_ships is None:
        if clips_ship_map_names_in_map_config(config):
            return config.model_copy(deep=True)
        if isinstance(config, MachinaArenaConfig):
            return add_clips_ships_to_map_config(config, DEFAULT_CLIPS_SHIP_COUNT)
        return config.model_copy(deep=True)

    if isinstance(config, AsciiMapBuilderConfig):
        if num_ships == 0:
            return _remove_from_ascii(config)
        current_ships = len(clips_ship_map_names_in_map_config(config))
        if current_ships == num_ships:
            return config.model_copy(deep=True)
        raise ValueError(
            f"ASCII maps define clips ship placements explicitly; cannot set clips ship count to {num_ships} "
            f"when the map contains {current_ships} placements."
        )

    config_without_ships = remove_clips_ships_from_map_config(config)
    if num_ships <= 0:
        return config_without_ships
    return add_clips_ships_to_map_config(config_without_ships, num_ships)


def _remove_from_ascii(config: AsciiMapBuilderConfig) -> AsciiMapBuilderConfig:
    empty_char = next((ch for ch, name in config.char_to_map_name.items() if name == "empty"), None)
    if empty_char is None:
        raise ValueError("No empty character mapping available to remove clips:ship placements")
    return config.model_copy(
        deep=True,
        update={
            "map_data": [
                [empty_char if is_clips_ship_map_name(config.char_to_map_name.get(cell, "")) else cell for cell in row]
                for row in config.map_data
            ]
        },
    )
