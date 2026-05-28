"""Re-export shim. Implementation moved to `mettagrid.sdk.cogsguard`. PR 6 deletes this."""

# ruff: noqa: F401, F403

from __future__ import annotations

from mettagrid.sdk.cogsguard import *  # re-export the same public surface
from mettagrid.sdk.cogsguard import __all__ as __all__
