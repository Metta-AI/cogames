"""Tests for variant config modifications via make_env.

Each test creates a CvCMission with specific variants and verifies the env config
is correctly modified.
"""

import pytest

from cogames.games.cogs_vs_clips.game import GEAR
from cogames.games.cogs_vs_clips.game.cargo import CargoLimitVariant
from cogames.games.cogs_vs_clips.game.damage import DamageVariant
from cogames.games.cogs_vs_clips.game.days import DayConfig, DaysVariant
from cogames.games.cogs_vs_clips.game.elements import ElementsVariant
from cogames.games.cogs_vs_clips.game.endless import EndlessVariant
from cogames.games.cogs_vs_clips.game.energy import EnergyVariant
from cogames.games.cogs_vs_clips.game.extractors import ExtractorsVariant
from cogames.games.cogs_vs_clips.game.forced_role_vibes import ForcedRoleVibesVariant
from cogames.games.cogs_vs_clips.game.gear import GearVariant
from cogames.games.cogs_vs_clips.game.gear_stations import GearStationsVariant
from cogames.games.cogs_vs_clips.game.heart import HeartVariant
from cogames.games.cogs_vs_clips.game.junction import JunctionVariant
from cogames.games.cogs_vs_clips.game.roles.aligner import AlignerVariant
from cogames.games.cogs_vs_clips.game.roles.miner import MinerVariant
from cogames.games.cogs_vs_clips.game.roles.scout import ScoutVariant
from cogames.games.cogs_vs_clips.game.roles.scrambler import ScramblerVariant
from cogames.games.cogs_vs_clips.game.solar import SolarVariant
from cogames.games.cogs_vs_clips.game.talk import TalkVariant
from cogames.games.cogs_vs_clips.game.teams import TeamConfig, TeamVariant
from cogames.games.cogs_vs_clips.game.teams.gear_stations import TeamGearStationsVariant
from cogames.games.cogs_vs_clips.game.teams.hub import TeamHubVariant
from cogames.games.cogs_vs_clips.game.teams.hub_observations import HubObservationsVariant
from cogames.games.cogs_vs_clips.game.territory import DamageStrangersVariant, HealTeamVariant, TerritoryVariant
from cogames.games.cogs_vs_clips.game.vibes import NoVibesVariant, VibesVariant
from cogames.games.cogs_vs_clips.missions.arena import make_arena_map_builder
from cogames.games.cogs_vs_clips.missions.machina_1 import CvCMachina1Variant
from cogames.games.cogs_vs_clips.missions.mission import CvCMission
from mettagrid.config.filter import GameValueFilter, ResourceFilter
from mettagrid.config.filter.periodic_filter import PeriodicFilter
from mettagrid.config.game_value import GameValue, RatioGameValue
from mettagrid.config.handler_config import AllOf, FirstMatch, Handler
from mettagrid.config.mutation.query_inventory_mutation import QueryInventoryMutation
from mettagrid.config.mutation.resource_mutation import ResourceTransferMutation
from mettagrid.simulator import Simulation
from variants.conftest import StationTestHarness


def _handler_by_name(obj, name) -> Handler:
    """Find a handler by name in a FirstMatch on_use_handler."""
    assert isinstance(obj.on_use_handler, FirstMatch), f"Expected FirstMatch, got {type(obj.on_use_handler)}"
    for h in obj.on_use_handler.handlers:
        if isinstance(h, Handler) and h.name == name:
            return h
    names = [h.name for h in obj.on_use_handler.handlers if isinstance(h, Handler)]
    raise KeyError(f"Handler '{name}' not found in {names}")


def _has_handler(obj, name) -> bool:
    """Check if a handler with the given name exists in a FirstMatch on_use_handler."""
    if not isinstance(obj.on_use_handler, FirstMatch):
        return False
    return any(isinstance(h, Handler) and h.name == name for h in obj.on_use_handler.handlers)


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


def _find_handler(handler, name: str):
    """Find a handler by name in a handler tree."""
    if handler is None:
        return None
    if isinstance(handler, Handler):
        return handler if handler.name == name else None
    if isinstance(handler, (FirstMatch, AllOf)):
        for h in handler.handlers:
            found = _find_handler(h, name)
            if found is not None:
                return found
    return None


ELEMENTS = ElementsVariant().elements
VIBE_NAMES = [v.name for v in VibesVariant().vibes]


def _make_mission(variants, num_cogs=4, max_steps=100):
    return CvCMission(
        name="test",
        description="test",
        map_builder=make_arena_map_builder(num_agents=num_cogs),
        num_cogs=num_cogs,
        min_cogs=num_cogs,
        max_cogs=num_cogs,
        max_steps=max_steps,
    ).with_variants([TeamVariant(default_teams={"cogs": TeamConfig(name="cogs", num_agents=num_cogs)}), *variants])


def _make_env_without_default(variants, num_cogs=4):
    mission = _make_mission(variants, num_cogs=num_cogs)
    return mission.model_copy(update={"default_variant": None, "num_agents": num_cogs}).make_env()


class TestElementsVariant:
    def test_adds_element_resources(self):
        env = _make_mission([ElementsVariant()]).make_env()
        for element in ELEMENTS:
            assert element in env.game.resource_names

    def test_idempotent(self):
        env = _make_mission([ElementsVariant()]).make_env()
        count = env.game.resource_names.count("oxygen")
        assert count == 1


class TestEnergyVariant:
    def test_adds_energy_limits_and_initial(self):
        env = _make_mission([EnergyVariant()]).make_env()
        for agent in env.game.agents:
            assert "energy" in agent.inventory.limits
            assert agent.inventory.initial["energy"] == 100

    def test_move_action_cost(self):
        env = _make_mission([EnergyVariant()]).make_env()
        assert env.game.actions.move.consumed_resources == {"energy": 4}

    def test_energy_resource_added(self):
        env = _make_mission([EnergyVariant()]).make_env()
        assert "energy" in env.game.resource_names


class TestSolarVariant:
    def test_adds_solar_to_energy_tick_handler(self):
        env = _make_mission([SolarVariant()]).make_env()
        for agent in env.game.agents:
            assert "solar_to_energy" in _handler_names(agent.on_tick)


class TestDaysVariant:
    def test_adds_weather_events(self):
        env = _make_mission([DaysVariant()], max_steps=1000).make_env()
        assert "day" in env.game.events
        assert "night" in env.game.events

    def test_initial_solar_set(self):
        cfg = DayConfig(night_solar=2)
        env = _make_mission([DaysVariant(days_config=cfg)]).make_env()
        for agent in env.game.agents:
            assert agent.inventory.initial.get("solar") == 2


class TestCargoLimitVariant:
    def test_adds_cargo_limit_to_agents(self):
        env = _make_mission([CargoLimitVariant()]).make_env()
        for agent in env.game.agents:
            assert "cargo" in agent.inventory.limits
            cargo = agent.inventory.limits["cargo"]
            assert set(cargo.resources) == set(ELEMENTS)


class TestExtractorsVariant:
    def test_adds_extractors_for_each_element(self):
        env = _make_mission([ExtractorsVariant()]).make_env()
        for element in ELEMENTS:
            key = f"{element}_extractor"
            assert key in env.game.objects, f"Missing {key}"

    def test_default_limit_set_to_initial_amount(self):
        initial = 300
        env = _make_mission([ExtractorsVariant(initial_amount=initial)]).make_env()
        for element in ELEMENTS:
            obj = env.game.objects[f"{element}_extractor"]
            assert obj.inventory.default_limit == initial

    def test_remove_when_empty_defaults_true(self):
        env = _make_mission([ExtractorsVariant()]).make_env()
        for element in ELEMENTS:
            obj = env.game.objects[f"{element}_extractor"]
            assert isinstance(obj.on_use_handler, FirstMatch)
            for handler in obj.on_use_handler.handlers:
                if not isinstance(handler, Handler):
                    continue
                for mutation in handler.mutations:
                    if isinstance(mutation, ResourceTransferMutation):
                        assert mutation.remove_source_when_empty is True


class TestEndlessVariant:
    def test_sets_max_steps_zero(self):
        env = _make_mission([EndlessVariant()]).make_env()
        assert env.game.max_steps == 0

    def test_works_without_extractors(self):
        env = _make_env_without_default([EndlessVariant()])
        assert env.game.max_steps == 0
        # No on_tick refill handlers when extractors not present
        for key in _handler_names(env.game.on_tick):
            assert "refill" not in key

    def test_sets_remove_when_empty_false(self):
        env = _make_mission([EndlessVariant(), ExtractorsVariant()]).make_env()
        for element in ELEMENTS:
            obj = env.game.objects[f"{element}_extractor"]
            assert isinstance(obj.on_use_handler, FirstMatch)
            for handler in obj.on_use_handler.handlers:
                if not isinstance(handler, Handler):
                    continue
                for mutation in handler.mutations:
                    if isinstance(mutation, ResourceTransferMutation):
                        assert mutation.remove_source_when_empty is False

    def test_adds_refill_tick_handlers(self):
        env = _make_mission([EndlessVariant(), ExtractorsVariant()]).make_env()
        for element in ELEMENTS:
            key = f"{element}_extractor_refill"
            assert key in _handler_names(env.game.on_tick), f"Missing on_tick handler {key}"

    def test_refill_handler_uses_periodic_filter(self):
        period = 500
        env = _make_mission([EndlessVariant(refill_period=period), ExtractorsVariant()]).make_env()
        handler = _find_handler(env.game.on_tick, f"{ELEMENTS[0]}_extractor_refill")
        assert handler is not None
        periodic_filters = [f for f in handler.filters if isinstance(f, PeriodicFilter)]
        assert len(periodic_filters) == 1
        assert periodic_filters[0].period == period

    def test_refill_handler_uses_gamevalue_max_items(self):
        env = _make_mission([EndlessVariant(refill_fraction=4), ExtractorsVariant()]).make_env()
        handler = _find_handler(env.game.on_tick, f"{ELEMENTS[0]}_extractor_refill")
        assert handler is not None
        qi_mutations = [m for m in handler.mutations if isinstance(m, QueryInventoryMutation)]
        assert len(qi_mutations) == 1
        q = qi_mutations[0].query
        assert isinstance(q.max_items, GameValue)
        assert isinstance(q.max_items, RatioGameValue)

    def test_produces_runnable_simulation(self):
        """EndlessVariant + ExtractorsVariant produces a config that can be compiled to C++ and stepped."""
        env = _make_env_without_default([EndlessVariant(refill_period=5), ExtractorsVariant()])
        sim = Simulation(env, seed=42)
        for _ in range(20):
            for i in range(sim.num_agents):
                sim.agent(i).set_action("noop")
            sim.step()
        sim.close()


class TestGearVariant:
    def test_adds_gear_limit_to_agents(self):
        env = _make_mission([AlignerVariant(), ScramblerVariant(), MinerVariant(), ScoutVariant()]).make_env()
        for agent in env.game.agents:
            assert "gear" in agent.inventory.limits
            gear = agent.inventory.limits["gear"]
            assert gear.base == 1
            assert set(gear.resources) == set(GEAR)

    def test_adds_gear_stations(self):
        env = _make_env_without_default(
            [
                GearVariant(station_costs={"aligner": {"carbon": 2}}, station_symbols={"aligner": "A"}),
                AlignerVariant(),
                ScramblerVariant(),
                MinerVariant(),
                ScoutVariant(),
                GearStationsVariant(),
            ]
        )
        assert set(GEAR) <= env.game.objects.keys()
        assert env.game.render.symbols["aligner"] == "A"
        cost_filter = _handler_by_name(env.game.objects["aligner"], "change_gear").filters[0]
        assert isinstance(cost_filter, ResourceFilter)
        assert cost_filter.resources == {"carbon": 2}

    def test_adds_team_gear_stations(self):
        env = _make_env_without_default(
            [
                GearVariant(station_costs={"miner": {"carbon": 2}}, station_symbols={"miner": "M"}),
                AlignerVariant(),
                ScramblerVariant(),
                MinerVariant(),
                ScoutVariant(),
                TeamGearStationsVariant(),
            ]
        )
        assert {f"c:{role}" for role in GEAR} <= env.game.objects.keys()
        assert set(GEAR).isdisjoint(env.game.objects)
        assert env.game.objects["c:miner"].name == "miner"
        assert env.game.render.symbols["c:miner"] == "M"
        cost_filter = _handler_by_name(env.game.objects["c:miner"], "change_gear").filters[1]
        assert isinstance(cost_filter, GameValueFilter)
        assert cost_filter.min == 2


class TestTeamHubVariant:
    def test_creates_hub_for_each_team(self):
        env = _make_mission([TeamHubVariant()]).make_env()
        assert "c:hub" in env.game.objects
        hub = env.game.objects["c:hub"]
        assert hub.name == "hub"
        assert "team:cogs" in hub.tags

    def test_hub_has_deposit_handler(self):
        env = _make_mission([TeamHubVariant()]).make_env()
        hub = env.game.objects["c:hub"]
        assert _has_handler(hub, "deposit")

    def test_initial_hearts_override_preserves_default_element_inventory(self):
        env = _make_env_without_default(
            [
                ElementsVariant(),
                HeartVariant(),
                TeamHubVariant(initial_hearts={"cogs": 120}),
            ]
        )
        hub = env.game.objects["c:hub"]
        expected_element_inventory = len(env.game.agents) * 3
        assert hub.inventory.initial["heart"] == 120
        assert hub.inventory.initial["oxygen"] == expected_element_inventory
        assert hub.inventory.initial["carbon"] == expected_element_inventory
        assert hub.inventory.initial["germanium"] == expected_element_inventory
        assert hub.inventory.initial["silicon"] == expected_element_inventory


class TestHeartVariant:
    def test_adds_heart_limit_to_agents(self):
        env = _make_mission([HeartVariant()]).make_env()
        for agent in env.game.agents:
            assert "heart" in agent.inventory.limits
            heart = agent.inventory.limits["heart"]
            assert heart.base == 10

    def test_adds_heart_handlers_to_hub(self):
        env = _make_mission([TeamHubVariant(), HeartVariant(cost={"oxygen": 7})]).make_env()
        hub = env.game.objects["c:hub"]
        assert _has_handler(hub, "get_heart")
        assert _has_handler(hub, "make_and_get_heart")


class TestJunctionVariant:
    def test_adds_junction_object(self):
        env = _make_mission([JunctionVariant()]).make_env()
        assert "junction" in env.game.objects
        junction = env.game.objects["junction"]
        assert junction.name == "junction"

    def test_adds_junction_render_assets(self):
        env = _make_mission([JunctionVariant()]).make_env()
        assert "junction" in env.game.render.assets


class TestTerritoryVariant:
    def test_adds_territory_config(self):
        env = _make_mission([TerritoryVariant()]).make_env()
        assert "team_territory" in env.game.territories
        territory = env.game.territories["team_territory"]
        assert territory.tag_prefix == "team:"

    def test_adds_materialized_queries(self):
        env = _make_mission([TerritoryVariant()]).make_env()
        tags = {mq.tag for mq in env.game.materialize_queries}
        assert "net:cogs" in tags

    def test_adds_territory_controls_to_hub_and_junction(self):
        env = _make_mission([TerritoryVariant()]).make_env()
        hub = env.game.objects["c:hub"]
        assert len(hub.territory_controls) > 0
        junction = env.game.objects["junction"]
        assert len(junction.territory_controls) > 0

    def test_adds_alignment_queries(self):
        env = _make_mission([TerritoryVariant()]).make_env()
        query_tags = [mq.tag for mq in env.game.materialize_queries]
        assert "net:cogs" in query_tags


class TestDamageVariant:
    def test_adds_hp_limit_and_initial(self):
        env = _make_mission([DamageVariant()]).make_env()
        for agent in env.game.agents:
            assert "hp" in agent.inventory.limits
            assert agent.inventory.initial["hp"] == 50

    def test_adds_regen_tick_handler(self):
        env = _make_mission([DamageVariant()]).make_env()
        for agent in env.game.agents:
            assert "hp_regen" in _handler_names(agent.on_tick)

    def test_hp_resource_added(self):
        env = _make_mission([DamageVariant()]).make_env()
        assert "hp" in env.game.resource_names


class TestDamageStrangersVariant:
    def test_adds_damage_strangers_to_territory(self):
        env = _make_mission([DamageStrangersVariant()]).make_env()
        territory = env.game.territories["team_territory"]
        assert "damage_strangers" in territory.presence


class TestHealTeamVariant:
    def test_adds_heal_energy_to_territory(self):
        env = _make_mission([HealTeamVariant()]).make_env()
        territory = env.game.territories["team_territory"]
        assert "heal_energy" in territory.presence

    @pytest.mark.parametrize(
        "variant_types",
        [(DamageVariant, HealTeamVariant), (HealTeamVariant, DamageVariant)],
    )
    def test_adds_heal_hp_when_damage_variant_is_present(self, variant_types):
        env = _make_mission([variant_type() for variant_type in variant_types]).make_env()
        territory = env.game.territories["team_territory"]
        assert "heal_hp" in territory.presence


class TestHubObservationsVariant:
    def test_adds_element_observations(self):
        env = _make_mission([HubObservationsVariant()]).make_env()
        for element in ELEMENTS:
            key = f"team:{element}"
            assert key in env.game.obs.global_obs.obs, f"Missing obs {key}"


class TestVibesVariant:
    def test_adds_vibe_names(self):
        env = _make_mission([VibesVariant()]).make_env()
        assert env.game.vibe_names == VIBE_NAMES

    def test_enables_change_vibe_action(self):
        env = _make_mission([VibesVariant()]).make_env()
        assert env.game.actions.change_vibe.enabled is True


class TestNoVibesVariant:
    def test_disables_change_vibe_action(self):
        env = _make_mission([NoVibesVariant()]).make_env()
        assert env.game.actions.change_vibe.enabled is False
        assert env.game.vibe_names == VIBE_NAMES


class TestForcedRoleVibesVariant:
    def test_assigns_roles_from_custom_role_order(self):
        env = _make_mission(
            [
                ForcedRoleVibesVariant(
                    role_order=["miner", "aligner", "scrambler"],
                    per_team=False,
                )
            ]
        ).make_env()

        assert env.game.actions.change_vibe.enabled is False
        assert env.game.actions.change_vibe.vibes == []
        assert [agent.vibe for agent in env.game.agents] == [
            VIBE_NAMES.index("miner"),
            VIBE_NAMES.index("aligner"),
            VIBE_NAMES.index("scrambler"),
            VIBE_NAMES.index("miner"),
            VIBE_NAMES.index("aligner"),
            VIBE_NAMES.index("scrambler"),
            VIBE_NAMES.index("miner"),
            VIBE_NAMES.index("aligner"),
        ]


class TestTalkVariant:
    def test_replaces_change_vibe_with_talk(self):
        env = _make_mission([TalkVariant()]).make_env()
        assert env.game.actions.change_vibe.enabled is False
        assert env.game.vibe_names == VIBE_NAMES
        assert env.game.talk.enabled is True
        assert env.game.talk.max_length == 140
        assert env.game.talk.cooldown_steps == 50


class TestTeamVariant:
    def test_sets_team_sizes(self):
        env = _make_mission([TeamVariant(team_sizes={"cogs": 3})], num_cogs=4).make_env()
        assert env.game.num_agents == 3


class TestRoleVariants:
    def test_miner_adds_junction_deposit_handlers(self):
        from cogames.games.cogs_vs_clips.game.teams.junction_deposit import JunctionDepositVariant  # noqa: PLC0415

        env = _make_mission([JunctionVariant(), MinerVariant(), JunctionDepositVariant()]).make_env()
        junction = env.game.objects.get("junction")
        assert junction is not None
        assert isinstance(junction.on_use_handler, FirstMatch)
        deposit_handlers = [
            h.name for h in junction.on_use_handler.handlers if isinstance(h, Handler) and h.name.startswith("deposit_")
        ]
        assert len(deposit_handlers) > 0


class TestMachina1Variant:
    def test_produces_complete_env(self):
        mission = CvCMission(
            name="test",
            description="test",
            map_builder=make_arena_map_builder(num_agents=4),
            min_cogs=4,
            max_cogs=4,
            num_cogs=4,
            num_agents=4,
            max_steps=100,
        ).with_variants([CvCMachina1Variant()])
        env = mission.make_env()
        assert "c:hub" in env.game.objects
        assert "junction" in env.game.objects
        for element in ELEMENTS:
            assert f"{element}_extractor" in env.game.objects
        for role in GEAR:
            assert f"c:{role}" in env.game.objects
        assert "team_territory" in env.game.territories
        assert len(env.game.agents) > 0
        assert "day" in env.game.events
        junction = env.game.objects["junction"]
        assert isinstance(junction.on_use_handler, FirstMatch)
        junction_handler_names = {h.name for h in junction.on_use_handler.handlers if isinstance(h, Handler)}
        assert {"deposit_cogs", "deposit_clips"} <= junction_handler_names
        for agent in env.game.agents:
            assert "hp" in agent.inventory.limits
            assert "energy" in agent.inventory.limits
            assert agent.inventory.limits["gear"].base == 1
            assert agent.inventory.limits["heart"].base == 10

    def test_junction_deposit_forwards_resources_to_team_hub(self):
        mission = CvCMission(
            name="test",
            description="test",
            map_builder=make_arena_map_builder(num_agents=4),
            min_cogs=4,
            max_cogs=4,
            num_cogs=4,
            num_agents=4,
            max_steps=100,
        ).with_variants([CvCMachina1Variant()])
        env = mission.make_env()
        junction = env.game.objects["junction"].model_copy(
            update={"tags": [*env.game.objects["junction"].tags, "team:cogs"]}
        )
        harness = StationTestHarness.create(
            station=junction,
            agent_inventory={"oxygen": 50},
            agent_team="cogs",
            tags=list(env.game.tags),
            extra_objects=[env.game.objects["c:hub"]],
        )
        hub_oxygen_before = harness.object_inventory("hub").get("oxygen", 0)

        harness.move_onto_station()

        assert harness.agent_inventory().get("oxygen", 0) == 0
        assert harness.object_inventory("hub").get("oxygen", 0) == hub_oxygen_before + 50

        harness.close()

    def test_includes_clips(self):
        mission = CvCMission(
            name="test",
            description="test",
            map_builder=make_arena_map_builder(num_agents=4),
            min_cogs=4,
            max_cogs=4,
            num_cogs=4,
            num_agents=4,
            max_steps=100,
        ).with_variants([CvCMachina1Variant()])
        env = mission.make_env()
        clips_events = [k for k in env.game.events if "clips" in k or "neutral" in k]
        assert len(clips_events) > 0
