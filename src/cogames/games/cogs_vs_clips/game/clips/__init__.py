from cogames.games.cogs_vs_clips.game.clips.clips import ClipsConfig, ClipsVariant
from cogames.games.cogs_vs_clips.game.clips.ship import (
    CLIPS_SHIP_MAP_NAME,
    CvCShipConfig,
    add_clips_ships_to_map_config,
    clips_ship_map_names_in_map_config,
    count_clips_ships_in_map_config,
    is_clips_ship_map_name,
    remove_clips_ships_from_map_config,
)

__all__ = [
    "CLIPS_SHIP_MAP_NAME",
    "ClipsConfig",
    "ClipsVariant",
    "CvCShipConfig",
    "add_clips_ships_to_map_config",
    "clips_ship_map_names_in_map_config",
    "count_clips_ships_in_map_config",
    "is_clips_ship_map_name",
    "remove_clips_ships_from_map_config",
]
