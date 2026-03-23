from cogames.games.cogs_vs_clips.game.clips.clips import (
    AngryClipsVariant,
    ClipsConfig,
    ClipsVariant,
    GreedyClipsVariant,
    NoClipsVariant,
)
from cogames.games.cogs_vs_clips.game.clips.ship import (
    CLIPS_SHIP_MAP_NAME,
    DEFAULT_CLIPS_SHIP_COUNT,
    CvCShipConfig,
    add_clips_ships_to_map_config,
    clips_ship_map_names_in_map_config,
    count_clips_ships_in_map_config,
    is_clips_ship_map_name,
    remove_clips_ships_from_map_config,
    set_clips_ships_in_map_config,
)

__all__ = [
    "CLIPS_SHIP_MAP_NAME",
    "DEFAULT_CLIPS_SHIP_COUNT",
    "AngryClipsVariant",
    "ClipsConfig",
    "ClipsVariant",
    "GreedyClipsVariant",
    "NoClipsVariant",
    "CvCShipConfig",
    "add_clips_ships_to_map_config",
    "clips_ship_map_names_in_map_config",
    "count_clips_ships_in_map_config",
    "is_clips_ship_map_name",
    "remove_clips_ships_from_map_config",
    "set_clips_ships_in_map_config",
]
