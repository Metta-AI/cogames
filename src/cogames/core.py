"""Re-export shim for the core cogame mission/variant types.

The implementation now lives in `mettagrid.cogame.core`. This shim keeps
existing in-tree imports of `cogames.core` working until external repos
migrate; PR 6 deletes the cogames package entirely.
"""

from __future__ import annotations

from mettagrid.cogame.core import (
    CoGameMission,
    CoGameMissionVariant,
    CvCStationConfig,
    Deps,
)

__all__ = [
    "CoGameMission",
    "CoGameMissionVariant",
    "CvCStationConfig",
    "Deps",
]
