from __future__ import annotations

from cogames.cogs_vs_clips.terrain import MachinaArenaConfig
from mettagrid.map_builder.ascii import AsciiMapBuilderConfig
from mettagrid.map_builder.map_builder import AnyMapBuilderConfig, MapBuilderConfig
from mettagrid.mapgen.mapgen import MapGenConfig
from mettagrid.mapgen.scene import SceneConfig

CLIPS_SHIP_MAP_NAME = "clips:ship"


def count_clips_ships_in_map_config(config: MapBuilderConfig | SceneConfig) -> int:
    """Count clips ship placements in a map builder config."""
    if isinstance(config, MapGenConfig):
        if config.instance is None:
            return 0
        return count_clips_ships_in_map_config(config.instance) * (config.instances or 1)

    if isinstance(config, MachinaArenaConfig):
        corner = sum(1 for name, _ in config.map_corner_placements if name == CLIPS_SHIP_MAP_NAME)
        perimeter = sum(max(0, count) for name, count in config.map_perimeter_placements if name == CLIPS_SHIP_MAP_NAME)
        return corner + perimeter

    if isinstance(config, AsciiMapBuilderConfig):
        return sum(
            1 for row in config.map_data for cell in row if config.char_to_map_name.get(cell) == CLIPS_SHIP_MAP_NAME
        )

    return 0


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
            "map_corner_placements": [p for p in config.map_corner_placements if p[0] != CLIPS_SHIP_MAP_NAME],
            "map_perimeter_placements": [p for p in config.map_perimeter_placements if p[0] != CLIPS_SHIP_MAP_NAME],
        }
    )


def _remove_from_ascii(config: AsciiMapBuilderConfig) -> AsciiMapBuilderConfig:
    empty_char = next((ch for ch, name in config.char_to_map_name.items() if name == "empty"), None)
    if empty_char is None:
        raise ValueError("No empty character mapping available to remove clips:ship placements")
    return config.model_copy(
        deep=True,
        update={
            "map_data": [
                [empty_char if config.char_to_map_name.get(cell) == CLIPS_SHIP_MAP_NAME else cell for cell in row]
                for row in config.map_data
            ]
        },
    )
