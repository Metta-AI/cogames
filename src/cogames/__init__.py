"""CoGames package."""

from cogames.game import register_game
from cogames.games import overcogged as _overcogged_game  # noqa: F401
from cogames.games.cogs_vs_clips.game.game import CvCGame

register_game(CvCGame())
