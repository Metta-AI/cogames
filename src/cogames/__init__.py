"""Tombstone for the cogames package.

CoGames has been retired in favor of Coworld
(https://github.com/Metta-AI/coworld). This wheel exists only to give
existing `pip install cogames` users a clear deprecation signal.

Importing this module emits a `DeprecationWarning`. No functional
surface remains; submodules that previously lived here (`cogames.core`,
`cogames.game`, `cogames.variants`, `cogames.sdk.cogsguard`) have moved
to `mettagrid.cogame` and `mettagrid.sdk.cogsguard`.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "cogames is retired in favor of coworld "
    "(https://github.com/Metta-AI/coworld). The variant framework "
    "moved to mettagrid.cogame; the cogsguard SDK moved to "
    "mettagrid.sdk.cogsguard. Update your imports and remove cogames "
    "from your dependencies.",
    DeprecationWarning,
    stacklevel=2,
)
