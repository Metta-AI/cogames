"""Shared typing helpers for Overcogged variant graph composition."""

from __future__ import annotations

from typing import Protocol, TypeVar

from cogames.core import CoGameMissionVariant

T = TypeVar("T", bound=CoGameMissionVariant)


class VariantGraphAccess(Protocol):
    """Minimal access surface shared by both variant-registry implementations."""

    def required(self, variant_type: type[T]) -> T: ...

    def optional(self, variant_type: type[T]) -> T | None: ...
