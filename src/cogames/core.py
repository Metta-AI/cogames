"""Core base classes for CoGame missions and variants."""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from pydantic import Field, PrivateAttr
from typing_extensions import Self

import cogames.variants as variants
from mettagrid.base_config import Config
from mettagrid.config.mettagrid_config import GridObjectConfig, MettaGridConfig
from mettagrid.map_builder.map_builder import AnyMapBuilderConfig

if TYPE_CHECKING:
    from cogames.variants import ResolvedDeps, VariantRegistry

# Type variable for mission types
TMission = TypeVar("TMission", bound="CoGameMission")
T = TypeVar("T", bound="CoGameMissionVariant")

MAP_MISSION_DELIMITER = "."


@dataclass
class Deps:
    """Declared dependencies for a variant, resolved before configure runs."""

    required: list[type[CoGameMissionVariant]] = field(default_factory=list)
    optional: list[type[CoGameMissionVariant]] = field(default_factory=list)


class CvCStationConfig(Config):
    def station_cfg(self) -> GridObjectConfig:
        raise NotImplementedError("Subclasses must implement this method")


class CoGameMissionVariant(Config, ABC):
    name: str
    description: str = Field(default="")
    depends_on: list[str] = Field(default_factory=list)

    _type_registry: ClassVar[dict[str, type[CoGameMissionVariant]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        name_val = cls.__dict__.get("name")
        if isinstance(name_val, str) and name_val:
            CoGameMissionVariant._type_registry[name_val] = cls

    @classmethod
    def create(cls, name: str) -> CoGameMissionVariant:
        variant_cls = cls._type_registry.get(name)
        assert variant_cls is not None, f"Unknown variant '{name}'. Available: {sorted(cls._type_registry)}"
        return variant_cls()  # pyright: ignore[reportCallIssue]

    def dependencies(self) -> Deps:
        """Declare required and optional variant dependencies.

        Called before configure to build the full dependency graph. The registry
        auto-creates missing required deps and repeats until stable.
        """
        return Deps()

    def configure(self, deps: ResolvedDeps) -> None:
        """Cross-configure with other active variants via resolved deps.

        Called after dependency resolution. Only declared deps are accessible.
        """

    def modify_mission(self, mission: CoGameMission) -> None:
        # Override this method to modify the mission.
        # Variants are allowed to modify the mission in-place - it's guaranteed to be a one-time only instance.
        pass

    def modify_env(self, mission: CoGameMission, env: MettaGridConfig) -> None:
        # Override this method to modify the produced environment.
        # Variants are allowed to modify the environment in-place.
        pass

    def compat(self, mission: CoGameMission) -> bool:
        """Check if this variant is compatible with the given mission.

        Returns True if the variant can be safely applied to the mission.
        Override this method to add compatibility checks.
        """
        return True

    def apply(self, mission: TMission) -> TMission:
        mission = mission.model_copy(deep=True)
        mission.variants.append(self)
        self.modify_mission(mission)
        return mission


class CoGameSite(Config):
    name: str
    description: str
    map_builder: AnyMapBuilderConfig

    min_cogs: int = Field(default=1, ge=1)
    max_cogs: int = Field(default=1000, ge=1)


class CoGameMission(Config, ABC):
    """Base class for Mission configurations with common fields and methods."""

    name: str
    description: str
    site: CoGameSite
    num_cogs: int | None = None

    default_variant: str | None = None
    sub_missions: list[str] = Field(default_factory=list)

    # Variants are applied to the mission immediately, and to its env when make_env is called
    variants: list[CoGameMissionVariant] = Field(default_factory=list)

    max_steps: int = Field(default=10000)

    _variant_registry: VariantRegistry = PrivateAttr(default_factory=lambda: variants.VariantRegistry())

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Can't call `variant.apply` here because it will create a new mission instance
        for variant in self.variants:
            variant.modify_mission(self)

    def required_variant(self, variant_type: type[T]) -> T:
        """Look up a configured variant by type. Only valid after configure phase."""
        return self._variant_registry.required(variant_type)

    def optional_variant(self, variant_type: type[T]) -> T | None:
        """Look up an optional variant by type. Returns None if not active."""
        return self._variant_registry.optional(variant_type)

    def has_variant(self, name: str) -> bool:
        return self._variant_registry.has(name)

    def with_variants(self, variants: list[CoGameMissionVariant] | Sequence[str | CoGameMissionVariant]) -> Self:
        has_strings = any(isinstance(v, str) for v in variants)
        if not has_strings:
            # Legacy path: all variant objects, apply eagerly
            mission = self
            for variant in variants:
                assert isinstance(variant, CoGameMissionVariant)
                mission = variant.apply(mission)
            return mission
        # Registry path: resolve names to objects and apply via the standard path
        copy = self.model_copy(deep=True)
        for v in variants:
            variant = v if isinstance(v, CoGameMissionVariant) else CoGameMissionVariant.create(v)
            copy.variants.append(variant)
            variant.modify_mission(copy)
        return copy

    def full_name(self) -> str:
        return f"{self.site.name}{MAP_MISSION_DELIMITER}{self.name}"
