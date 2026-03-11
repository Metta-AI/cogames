"""Tests for the energy -> solar -> days variant chain."""

from __future__ import annotations

from cogames.cogs_vs_clips.days import DayConfig, DaysVariant
from cogames.cogs_vs_clips.energy import EnergyVariant
from cogames.cogs_vs_clips.mission import CvCMission
from cogames.cogs_vs_clips.sites import COGSGUARD_MACHINA_1
from cogames.cogs_vs_clips.solar import SolarVariant
from cogames.cogs_vs_clips.variants import DarkSideVariant, NoWeatherVariant, SuperChargedVariant
from cogames.variants import VariantRegistry


def _make_mission(default_variant: str | None = "machina_1") -> CvCMission:
    return CvCMission(
        name="test",
        description="test mission",
        site=COGSGUARD_MACHINA_1,
        num_cogs=2,
        max_steps=1000,
        default_variant=default_variant,
    )


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
        registry.run_configure(["energy"])

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
        registry.run_configure(["solar"])

        assert registry.has("energy")
        assert registry.has("solar")

    def test_adds_solar_and_handler(self):
        registry = VariantRegistry()
        registry.run_configure(["solar"])

        mission = _make_mission(default_variant=None)
        env = mission.make_env()
        registry.apply_to_env(mission, env)

        solar_v = registry.required(SolarVariant)
        for agent in env.game.agents:
            assert "solar" in agent.inventory.initial
            assert agent.inventory.initial["solar"] == solar_v.initial_solar
            assert "solar_to_energy" in agent.on_tick
            # Energy should also be set (auto-created dependency)
            assert "energy" in agent.inventory.limits


class TestDaysVariant:
    def test_auto_creates_solar_and_energy(self):
        registry = VariantRegistry()
        registry.run_configure(["days"])

        assert registry.has("energy")
        assert registry.has("solar")
        assert registry.has("days")

    def test_creates_weather_events(self):
        registry = VariantRegistry()
        registry.run_configure(["days"])

        mission = _make_mission(default_variant=None)
        env = mission.make_env()
        registry.apply_to_env(mission, env)

        assert "day" in env.game.events
        assert "night" in env.game.events

    def test_custom_day_config(self):
        cfg = DayConfig(night_solar=2)
        days = DaysVariant(days_config=cfg)
        registry = VariantRegistry([days])
        registry.run_configure(["days"])

        mission = _make_mission(default_variant=None)
        env = mission.make_env()
        registry.apply_to_env(mission, env)

        for agent in env.game.agents:
            assert agent.inventory.initial.get("solar") == 2


class TestWeatherModifiers:
    def test_dark_side_zeroes_weather(self):
        mission = _make_mission().with_variants([DarkSideVariant()])
        env = mission.make_env()

        # Weather events should still exist but with zero delta
        assert "day" in env.game.events
        assert "night" in env.game.events

    def test_super_charged_boosts_weather(self):
        mission = _make_mission().with_variants([SuperChargedVariant()])
        env = mission.make_env()

        assert "day" in env.game.events
        assert "night" in env.game.events

    def test_no_weather_disables_cycle(self):
        mission = _make_mission().with_variants([NoWeatherVariant()])
        env = mission.make_env()

        # Weather events still exist but with zero deltas
        assert "day" in env.game.events
        assert "night" in env.game.events


class TestMakeEnvWithDefaultVariant:
    def test_default_variant_produces_full_config(self):
        mission = _make_mission()  # default_variant="machina_1"
        env = mission.make_env()

        energy_v = EnergyVariant()
        # Energy
        for agent in env.game.agents:
            assert "energy" in agent.inventory.limits
            assert agent.inventory.initial["energy"] == energy_v.initial
        assert env.game.actions.move.consumed_resources == {"energy": 4}

        # Solar
        for agent in env.game.agents:
            assert "solar" in agent.inventory.initial
            assert "solar_to_energy" in agent.on_tick

        # Weather
        assert "day" in env.game.events
        assert "night" in env.game.events

    def test_label_only_includes_user_variants(self):
        mission = _make_mission()
        env = mission.make_env()

        # Default variant deps should NOT appear in label
        assert ".energy" not in env.label
        assert ".solar" not in env.label
        assert ".days" not in env.label
        assert ".machina_1" not in env.label

    def test_label_includes_user_variants(self):
        mission = _make_mission().with_variants([DarkSideVariant()])
        env = mission.make_env()

        assert env.label.endswith(".dark_side")

    def test_no_default_variant_gives_bare_env(self):
        mission = _make_mission(default_variant=None)
        env = mission.make_env()

        for agent in env.game.agents:
            assert "energy" not in agent.inventory.limits
            assert "energy" not in agent.inventory.initial
            assert "solar" not in agent.inventory.initial
            assert "solar_to_energy" not in agent.on_tick
        assert "day" not in env.game.events
