"""Tests for the scrambler role variant: junction scramble handlers."""

from typing import Any

from cogames.games.cogs_vs_clips.game.teams import TeamConfig
from cogames.games.cogs_vs_clips.game.territory import (
    HUB_ALIGN_DISTANCE,
    JUNCTION_ALIGN_DISTANCE,
    net_materialized_query,
)
from mettagrid.config.filter import anyOf, hasTag, hasTagPrefix, isNear, isNot, sharedTagPrefix
from mettagrid.config.handler_config import Handler, actorHas, updateActor
from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    GameConfig,
    GridObjectConfig,
    InventoryConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    WallConfig,
)
from mettagrid.config.mutation import (
    addTag,
    recomputeMaterializedQuery,
    removeTag,
    removeTagPrefix,
)
from mettagrid.config.mutation.stats_mutation import logActorAgentStat, logStatToGame
from mettagrid.config.query import MaterializedQuery, query
from mettagrid.config.tag import typeTag
from mettagrid.config.territory_config import TerritoryConfig
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.simulator import Simulation

from .conftest import RESOURCES, StationTestHarness, count_junctions_by_team, has_tag_at, resource_limits

SCRAMBLE_COST = {"heart": 1}


def test_scrambler_constants():
    assert SCRAMBLE_COST == {"heart": 1}


def test_scramble_enemy_junction():
    """Agent with heart and scrambler can scramble enemy junction via tag removal."""
    station = GridObjectConfig(
        name="junction",
        tags=["team:clips"],
        on_use_handlers={
            "scramble": Handler(
                filters=[
                    hasTagPrefix("team:"),
                    actorHas({"scrambler": 1, **SCRAMBLE_COST}),
                ],
                mutations=[
                    removeTag("team:cogs"),
                    removeTag("team:clips"),
                    updateActor({k: -v for k, v in SCRAMBLE_COST.items()}),
                ],
            ),
        },
    )

    harness = StationTestHarness.create(
        station=station,
        agent_inventory={"heart": 5, "scrambler": 1},
        agent_team="cogs",
        tags=["team:cogs", "team:clips"],
    )

    before = count_junctions_by_team(harness.simulation, "team:clips")
    assert before == 1, f"Junction should start with team:clips, count={before}"

    harness.move_onto_station()

    after = count_junctions_by_team(harness.simulation, "team:clips")
    assert after == 0, f"Junction should be scrambled (team:clips removed), count={after}"

    inv = harness.agent_inventory()
    assert inv.get("heart", 0) == 4, f"Expected 4 hearts after scramble, got {inv.get('heart', 0)}"

    harness.close()


def test_scramble_removes_team_tag_via_on_tag_remove():
    """Scrambling a junction via removeTagPrefix('net:') cascades through on_tag_remove to remove team: tag.

    Uses the real CvCJunctionConfig (which uses removeTagPrefix('net:') + on_tag_remove cascade)
    rather than hardcoded removeTag calls.
    """
    cogs = TeamConfig(name="cogs", short_name="c")
    clips = TeamConfig(name="clips", short_name="clips")

    teams = [cogs, clips]
    align_required_resources = {"aligner": 1, "heart": 1}
    scramble_required_resources = {"scrambler": 1, "heart": 1}
    junction_cfg = GridObjectConfig(
        name="junction",
        tags=[f"team:{clips.name}"],
        on_tag_remove={f"net:{t.name}": Handler(filters=[], mutations=[removeTag(f"team:{t.name}")]) for t in teams},
    )
    for t in teams:
        junction_cfg.on_use_handlers[f"align_{t.name}"] = Handler(
            filters=[
                actorHas(align_required_resources),
                isNot(hasTagPrefix("team:")),
                anyOf(
                    [
                        isNear(query(t.net_tag()), radius=JUNCTION_ALIGN_DISTANCE),
                        isNear(query(typeTag("hub"), hasTag(t.team_tag())), radius=HUB_ALIGN_DISTANCE),
                    ]
                ),
            ],
            mutations=[
                updateActor({"heart": -1}),
                logActorAgentStat("junction.aligned_by_agent"),
                logStatToGame(f"{t.name}/aligned.junction.gained"),
                addTag(t.team_tag()),
                addTag(t.net_tag()),
                recomputeMaterializedQuery(t.net_tag()),
            ],
        )
        junction_cfg.on_use_handlers[f"scramble_{t.name}"] = Handler(
            filters=[
                hasTag(t.team_tag()),
                isNot(sharedTagPrefix("team:")),
                actorHas(scramble_required_resources),
            ],
            mutations=[
                updateActor({"heart": -1}),
                removeTagPrefix("net:"),
                logActorAgentStat("junction.scrambled_by_agent"),
                logStatToGame(f"{t.name}/aligned.junction.lost"),
                recomputeMaterializedQuery("net:"),
            ],
        )
    objects: dict[str, Any] = {"wall": WallConfig()}
    objects["clips:hub"] = GridObjectConfig(
        name="hub",
        map_name="clips:hub",
        tags=[clips.team_tag()],
    )
    objects["junction"] = junction_cfg

    all_tags: list[str] = [t.team_tag() for t in [cogs, clips]]

    mat_queries: list[MaterializedQuery] = []
    for t in [cogs, clips]:
        mat_queries.append(net_materialized_query(t))

    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            max_steps=100,
            resource_names=RESOURCES,
            territories={"team_territory": TerritoryConfig(tag_prefix="team:")},
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(),
            ),
            agent=AgentConfig(
                tags=["team:cogs"],
                inventory=InventoryConfig(
                    initial={"scrambler": 1, "heart": 5, "hp": 100, "energy": 100},
                    limits=resource_limits(),
                ),
            ),
            objects=objects,
            tags=all_tags,
            materialize_queries=mat_queries,
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    ["#", "#", "#", "#", "#"],
                    ["#", "H", ".", ".", "#"],
                    ["#", "J", "@", ".", "#"],
                    ["#", ".", ".", ".", "#"],
                    ["#", "#", "#", "#", "#"],
                ],
                char_to_map_name={
                    "#": "wall",
                    "@": "agent.agent",
                    ".": "empty",
                    "H": "clips:hub",
                    "J": "junction",
                },
            ),
        )
    )

    sim = Simulation(cfg, seed=42)
    sim.agent(0).set_inventory({"scrambler": 1, "heart": 5, "hp": 100, "energy": 100})

    assert has_tag_at(sim, "team:clips", 2, 1, "junction"), "Junction should start with team:clips"
    assert has_tag_at(sim, "net:clips", 2, 1, "junction"), "Junction should start with net:clips"

    sim.agent(0).set_action("move_west")
    sim.step()

    assert not has_tag_at(sim, "net:clips", 2, 1, "junction"), "Junction should not have net:clips after scramble"
    assert not has_tag_at(sim, "team:clips", 2, 1, "junction"), (
        "Junction should not have team:clips after scramble (via on_tag_remove cascade)"
    )

    sim.close()
