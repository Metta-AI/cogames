from types import SimpleNamespace

from mettagrid.config.vibes import Vibe

_GEAR = ["aligner", "scrambler", "miner", "scout"]
_ELEMENTS = ["oxygen", "carbon", "germanium", "silicon"]
_VIBES = [
    Vibe("😐", "default"),
    Vibe("❤️", "heart"),
    Vibe("⚙️", "gear"),
    Vibe("🌀", "scrambler"),
    Vibe("🔗", "aligner"),
    Vibe("⛏️", "miner"),
    Vibe("🔭", "scout"),
]

CvCConfig = SimpleNamespace(
    GEAR=_GEAR,
    ELEMENTS=_ELEMENTS,
    HEART_COST={e: 10 for e in _ELEMENTS},
    ALIGN_COST={"heart": 1},
    SCRAMBLE_COST={"heart": 1},
    GEAR_COSTS={
        "aligner": {"carbon": 3, "oxygen": 1, "germanium": 1, "silicon": 1},
        "scrambler": {"carbon": 1, "oxygen": 3, "germanium": 1, "silicon": 1},
        "miner": {"carbon": 1, "oxygen": 1, "germanium": 3, "silicon": 1},
        "scout": {"carbon": 1, "oxygen": 1, "germanium": 1, "silicon": 3},
    },
    GEAR_SYMBOLS={
        "aligner": "🔗",
        "scrambler": "🌀",
        "miner": "⛏️",
        "scout": "🔭",
    },
    RESOURCES=["energy", "heart", "hp", "influence", *_ELEMENTS, *_GEAR],
    VIBES=_VIBES,
    VIBE_NAMES=[vibe.name for vibe in _VIBES],
)
