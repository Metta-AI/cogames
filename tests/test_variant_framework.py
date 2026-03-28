"""Tests for the variant framework (VariantRegistry, ResolvedDeps, Deps)."""

from __future__ import annotations

import pytest

from cogames.core import CoGameMissionVariant, Deps
from cogames.variants import ResolvedDeps, VariantRegistry
from metta.games.games import GAMES, make_game, register

# --- Test variant subclasses (not ABC, so we can instantiate them) ---


class AlphaVariant(CoGameMissionVariant):
    name: str = "alpha"
    description: str = "Alpha variant"

    def dependencies(self) -> Deps:
        return Deps(required=[BetaVariant])

    def configure(self, deps: ResolvedDeps) -> None:
        self._beta = deps.required(BetaVariant)


class BetaVariant(CoGameMissionVariant):
    name: str = "beta"
    description: str = "Beta variant"


class GammaVariant(CoGameMissionVariant):
    name: str = "gamma"
    description: str = "Gamma variant"

    def dependencies(self) -> Deps:
        return Deps(optional=[AlphaVariant])

    def configure(self, deps: ResolvedDeps) -> None:
        self._alpha = deps.optional(AlphaVariant)


class DeltaVariant(CoGameMissionVariant):
    name: str = "delta"
    description: str = "Delta variant"


# --- _type_registry / __init_subclass__ ---


class TestTypeRegistry:
    def test_subclass_auto_registered(self):
        assert "alpha" in CoGameMissionVariant._type_registry
        assert issubclass(CoGameMissionVariant._type_registry["alpha"], CoGameMissionVariant)
        assert CoGameMissionVariant._type_registry["alpha"].__name__ == "AlphaVariant"

    def test_all_test_variants_registered(self):
        for name in ("alpha", "beta", "gamma", "delta"):
            assert name in CoGameMissionVariant._type_registry


# --- create() factory ---


class TestCreate:
    def test_create_known_variant(self):
        v = CoGameMissionVariant.create("alpha")
        assert v.name == "alpha"
        assert type(v).__name__ == "AlphaVariant"

    def test_create_unknown_raises(self):
        with pytest.raises(AssertionError, match="Unknown variant 'nonexistent'"):
            CoGameMissionVariant.create("nonexistent")


# --- VariantRegistry basics ---


class TestVariantRegistry:
    def test_register_and_get(self):
        reg = VariantRegistry([AlphaVariant(), BetaVariant()])
        assert reg.get("alpha") is not None
        assert reg.get("beta") is not None
        assert reg.get("nonexistent") is None

    def test_has(self):
        reg = VariantRegistry([AlphaVariant()])
        assert reg.has("alpha")
        assert not reg.has("beta")

    def test_all(self):
        variants = [AlphaVariant(), BetaVariant()]
        reg = VariantRegistry(variants)
        assert len(reg.all()) == 2


# --- Dependency resolution ---


class TestDependencyResolution:
    def test_auto_creates_required_dep(self):
        """Alpha requires Beta. If only Alpha is provided, Beta is auto-created."""
        reg = VariantRegistry([AlphaVariant()])
        reg._resolve_dependencies()
        assert reg.has("beta"), "Beta should have been auto-created as a required dep of Alpha"

    def test_no_auto_create_when_present(self):
        """If Beta already exists, it shouldn't be duplicated."""
        beta = BetaVariant()
        reg = VariantRegistry([AlphaVariant(), beta])
        reg._resolve_dependencies()
        assert reg.get("beta") is beta

    def test_optional_not_auto_created(self):
        """Gamma optionally depends on Alpha. Alpha should NOT be auto-created."""
        reg = VariantRegistry([GammaVariant()])
        reg._resolve_dependencies()
        assert not reg.has("alpha"), "Optional deps should not be auto-created"


# --- Configure phase ---


class TestConfigure:
    def test_configure_called_with_resolved_deps(self):
        """Alpha declares required=[BetaVariant]. After configure, Alpha._beta should be set."""
        alpha = AlphaVariant()
        beta = BetaVariant()
        reg = VariantRegistry([alpha, beta])
        reg.run_configure(["alpha", "beta"])
        assert alpha._beta is beta

    def test_configure_auto_creates_and_configures(self):
        """run_configure with just alpha should auto-create beta and configure both."""
        reg = VariantRegistry()
        reg.run_configure(["alpha"])
        assert reg.has("alpha")
        assert reg.has("beta")

    def test_optional_dep_present(self):
        """Gamma optionally depends on Alpha. When Alpha is present, it should be accessible."""
        gamma = GammaVariant()
        alpha = AlphaVariant()
        beta = BetaVariant()  # needed by Alpha
        reg = VariantRegistry([gamma, alpha, beta])
        reg.run_configure(["gamma", "alpha", "beta"])
        assert gamma._alpha is alpha

    def test_optional_dep_absent(self):
        """Gamma optionally depends on Alpha. When Alpha is absent, optional returns None."""
        gamma = GammaVariant()
        reg = VariantRegistry([gamma])
        reg.run_configure(["gamma"])
        assert gamma._alpha is None

    def test_configured_accessors_follow_topological_order(self):
        reg = VariantRegistry([AlphaVariant(), BetaVariant()])
        reg.run_configure(["alpha", "beta"])
        assert reg.configured_names() == ["beta", "alpha"]
        assert [variant.name for variant in reg.configured()] == ["beta", "alpha"]


# --- ResolvedDeps enforcement ---


class TestResolvedDeps:
    def test_undeclared_required_raises(self):
        """Accessing an undeclared required dep raises AssertionError."""
        reg = VariantRegistry()
        resolved = ResolvedDeps(reg, declared_required=set(), declared_optional=set())
        with pytest.raises(AssertionError, match="not declared in dependencies"):
            resolved.required(BetaVariant)

    def test_undeclared_optional_raises(self):
        """Accessing an undeclared optional dep raises AssertionError."""
        reg = VariantRegistry()
        resolved = ResolvedDeps(reg, declared_required=set(), declared_optional=set())
        with pytest.raises(AssertionError, match="not declared in dependencies"):
            resolved.optional(BetaVariant)


# --- Existing variants still work ---


class TestBackwardCompat:
    def test_default_dependencies_empty(self):
        """Variants that don't override dependencies() return empty Deps."""
        v = BetaVariant()
        deps = v.dependencies()
        assert deps.required == []
        assert deps.optional == []

    def test_default_configure_is_noop(self):
        """Variants that don't override configure() don't crash."""
        v = BetaVariant()
        reg = VariantRegistry([v])
        resolved = ResolvedDeps(reg, declared_required=set(), declared_optional=set())
        v.configure(resolved)  # should not raise

    def test_modify_env_still_callable(self):
        """The default modify_env is a no-op and still works."""
        v = DeltaVariant()
        # Just verify it doesn't crash with None mission/env (default is pass)
        v.modify_env(None, None)  # type: ignore[arg-type]

    def test_compat_default_true(self):
        v = DeltaVariant()
        assert v.compat(None) is True  # type: ignore[arg-type]


# --- build_dependency_graph ---


class TestDependencyGraph:
    def test_graph_edges(self):
        """build_dependency_graph returns edges between variants."""
        reg = VariantRegistry([AlphaVariant(), BetaVariant()])
        edges = reg.build_dependency_graph()
        # Alpha -> Beta (required)
        assert any(e[0] == "alpha" and e[1] == "beta" and e[2] == "required" for e in edges)


class TestTopologicalOrder:
    def test_deps_configured_before_dependents(self):
        """Beta (dep) should appear before Alpha (dependent) in configure order."""
        reg = VariantRegistry([AlphaVariant(), BetaVariant()])
        reg.run_configure(["alpha", "beta"])
        order = reg.configured_names()
        assert order.index("beta") < order.index("alpha")

    def test_circular_dependency_detected(self):
        """Circular deps should raise AssertionError."""

        class CycleA(CoGameMissionVariant):
            name: str = "cycle_a"
            description: str = ""

            def dependencies(self) -> Deps:
                return Deps(required=[CycleB])

        class CycleB(CoGameMissionVariant):
            name: str = "cycle_b"
            description: str = ""

            def dependencies(self) -> Deps:
                return Deps(required=[CycleA])

        reg = VariantRegistry([CycleA(), CycleB()])
        with pytest.raises(AssertionError, match="Circular dependency"):
            reg.run_configure(["cycle_a", "cycle_b"])


class _FakeMission:
    model_fields = {"max_steps": type("_Field", (), {"default": 17})()}

    def __init__(self, variants: list[str] | None = None) -> None:
        self._variants = list(variants or [])

    @classmethod
    def create(cls, num_agents: int, max_steps: int) -> "_FakeMission":
        assert num_agents == 3
        assert max_steps == 17
        return cls()

    def with_variants(self, variants: list[str]) -> "_FakeMission":
        return type(self)(variants)

    def make_env(self):
        return type(
            "_Env",
            (),
            {
                "label": "fake",
                "game": type("_Game", (), {"max_steps": 17})(),
            },
        )()


class TestMakeGameVariantValidation:
    def test_make_game_rejects_non_string_variant_entries(self) -> None:
        register("variant_validation_test", _FakeMission)
        try:
            with pytest.raises(TypeError, match="Game variants must be strings, got int"):
                make_game("variant_validation_test", num_agents=3, variants=[1])  # type: ignore[list-item]
        finally:
            GAMES.pop("variant_validation_test", None)
