"""Tests for the energy -> solar -> days variant chain."""

from __future__ import annotations

from cogsguard.game.energy import EnergyVariant
from cogsguard.missions.machina_1 import make_machina1_mission
from cogsguard.missions.mission import CvCMission

from cogames.variants import VariantRegistry
from mettagrid.config.handler_config import AllOf, FirstMatch, Handler

_CVC_VARIANT_MODULES = ("cogsguard.",)


def _handler_names(handler) -> set[str]:
    """Collect all handler names from a handler tree."""
    if handler is None:
        return set()
    if isinstance(handler, Handler):
        return {handler.name} if handler.name else set()
    if isinstance(handler, (FirstMatch, AllOf)):
        names = set()
        for h in handler.handlers:
            names |= _handler_names(h)
        return names
    return set()


def _make_mission(default_variant: str | None = "machina_1") -> CvCMission:
    mission = make_machina1_mission(num_agents=2, max_steps=1000)
    if default_variant is None:
        mission = mission.model_copy(update={"default_variant": None})
    return mission


class TestEnergyVariant:
    def test_adds_energy_limits_and_cost(self):
        mission = _make_mission(default_variant=None)
        env = mission.make_env()

        # Without energy variant, agents should not have energy limits
        for agent in env.game.agents:
            assert "energy" not in agent.inventory.limits
            assert "energy" not in agent.inventory.initial
        assert env.game.actions.move.consumed_resources == {}

    def test_with_energy_variant(self):
        registry = VariantRegistry()
        registry.run_configure(["energy"], preferred_modules=_CVC_VARIANT_MODULES)

        mission = _make_mission(default_variant=None)
        env = mission.make_env()
        registry.apply_to_env(mission, env)

        energy_v = registry.required(EnergyVariant)
        for agent in env.game.agents:
            assert "energy" in agent.inventory.limits
            assert agent.inventory.initial["energy"] == energy_v.initial
        assert env.game.actions.move.consumed_resources == {"energy": 4}


class TestSolarVariant:
    def test_auto_creates_energy(self):
        registry = VariantRegistry()
        registry.run_configure(["solar"], preferred_modules=_CVC_VARIANT_MODULES)

        assert registry.has("energy")
        assert registry.has("solar")

    def test_adds_solar_and_handler(self):
        registry = VariantRegistry()
        registry.run_configure(["solar"], preferred_modules=_CVC_VARIANT_MODULES)

        mission = _make_mission(default_variant=None)
        env = mission.make_env()
        registry.apply_to_env(mission, env)

        for agent in env.game.agents:
            assert "solar" in agent.inventory.initial
            assert "solar_to_energy" in _handler_names(agent.on_tick)
            # Energy should also be set (auto-created dependency)
            assert "energy" in agent.inventory.limits


class TestDaysVariant:
    def test_auto_creates_solar_and_energy(self):
        registry = VariantRegistry()
        registry.run_configure(["days"], preferred_modules=_CVC_VARIANT_MODULES)

        assert registry.has("energy")
        assert registry.has("solar")
        assert registry.has("days")

    def test_creates_weather_events(self):
        registry = VariantRegistry()
        registry.run_configure(["days"], preferred_modules=_CVC_VARIANT_MODULES)

        mission = _make_mission(default_variant=None)
        env = mission.make_env()
        registry.apply_to_env(mission, env)

        assert "day" in env.game.events
        assert "night" in env.game.events


class TestMakeEnvWithDefaultVariant:
    def test_default_variant_produces_full_config(self):
        mission = _make_mission()  # default_variant="machina_1"
        env = mission.make_env()

        energy_v = mission.required_variant(EnergyVariant)
        # Energy
        for agent in env.game.agents:
            assert "energy" in agent.inventory.limits
            assert agent.inventory.initial["energy"] == energy_v.initial
        assert env.game.actions.move.consumed_resources == {"energy": 4}

        # Solar
        for agent in env.game.agents:
            assert "solar" in agent.inventory.initial
            assert "solar_to_energy" in _handler_names(agent.on_tick)

        # Weather
        assert "day" in env.game.events
        assert "night" in env.game.events

    def test_no_default_variant_gives_bare_env(self):
        mission = _make_mission(default_variant=None)
        env = mission.make_env()

        for agent in env.game.agents:
            assert "energy" not in agent.inventory.limits
            assert "energy" not in agent.inventory.initial
            assert "solar" not in agent.inventory.initial
            assert "solar_to_energy" not in _handler_names(agent.on_tick)
        assert "day" not in env.game.events
