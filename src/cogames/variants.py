"""Re-export shim for the variant framework.

The implementation now lives in `mettagrid.cogame.variants`. This shim keeps
existing in-tree imports of `cogames.variants` working until external repos
migrate; PR 6 deletes the cogames package entirely.
"""

from __future__ import annotations

from mettagrid.cogame.variants import ResolvedDeps, VariantRegistry

__all__ = ["ResolvedDeps", "VariantRegistry"]
