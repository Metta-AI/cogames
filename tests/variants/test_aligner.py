"""Tests for the aligner role variant: junction alignment handlers."""

from cogames.games.cogs_vs_clips.game.teams import TeamConfig
from cogames.games.cogs_vs_clips.game.territory import net_materialized_query
from mettagrid.config.filter import actorHasTag, hasTagPrefix, isNear, isNot
from mettagrid.config.handler_config import Handler, actorHas, updateActor
from mettagrid.config.mettagrid_config import GridObjectConfig
from mettagrid.config.mutation import addTag, recomputeMaterializedQuery
from mettagrid.config.query import query as make_query

from .conftest import StationTestHarness, has_tag_at, make_junction_sim

ALIGN_COST = {"heart": 1}


def test_aligner_constants():
    assert ALIGN_COST == {"heart": 1}


def test_align_neutral_junction():
    """Agent with aligner+heart aligns a neutral junction near the hub.

    Map layout — junction must be 8-connected to hub so the ClosureQuery
    BFS can reach it after recomputeMaterializedQuery:
        #  #  #  #
        #  H  .  #
        #  @  J  #   ← J diagonal-adjacent to H
        #  .  .  #
        #  #  #  #
    """
    cogs = TeamConfig(name="cogs", short_name="c")
    sim = make_junction_sim(
        teams=[cogs],
        map_data=[
            ["#", "#", "#", "#"],
            ["#", "H", ".", "#"],
            ["#", "@", "J", "#"],
            ["#", ".", ".", "#"],
            ["#", "#", "#", "#"],
        ],
        char_to_map_name={
            "#": "wall",
            "@": "agent.agent",
            ".": "empty",
            "H": "c:hub",
            "J": "junction",
        },
        agent_inventory={"aligner": 1, "heart": 5, "hp": 100, "energy": 100},
        agent_team="cogs",
    )

    assert has_tag_at(sim, "net:cogs", 1, 1, "hub"), "Hub should have net:cogs from materialized query at init"

    sim.agent(0).set_action("move_east")
    sim.step()

    inv = sim.agent(0).inventory
    assert inv.get("heart", 0) == 4, f"Expected 4 hearts after alignment, got {inv.get('heart', 0)}"
    assert has_tag_at(sim, "team:cogs", 2, 2, "junction"), "Junction should have team:cogs after alignment"
    assert has_tag_at(sim, "net:cogs", 2, 2, "junction"), "Junction should have net:cogs after alignment"

    sim.close()


def test_align_without_isNear_does_not_crash():
    """Alignment without isNear filter works (isolates isNear as crash source)."""
    junction = GridObjectConfig(
        name="junction",
        on_use_handler=Handler(
            name="align_cogs",
            filters=[
                actorHasTag("team:cogs"),
                actorHas({"aligner": 1, **ALIGN_COST}),
                isNot(hasTagPrefix("team:")),
            ],
            mutations=[
                updateActor({k: -v for k, v in ALIGN_COST.items()}),
                addTag("team:cogs"),
            ],
        ),
    )

    harness = StationTestHarness.create(
        station=junction,
        agent_inventory={"aligner": 1, "heart": 5, "hp": 100, "energy": 100},
        agent_team="cogs",
        tags=["team:cogs"],
    )

    harness.move_onto_station()

    inv = harness.agent_inventory()
    assert inv.get("heart", 0) == 4, f"Expected 4 hearts, got {inv.get('heart', 0)}"
    harness.close()


def test_align_with_isNear_filter():
    """Alignment with isNear filter against materialized query tag."""
    cogs = TeamConfig(name="cogs", short_name="c")

    junction = GridObjectConfig(
        name="junction",
        on_use_handler=Handler(
            name="align_cogs",
            filters=[
                actorHasTag("team:cogs"),
                actorHas({"aligner": 1, **ALIGN_COST}),
                isNot(hasTagPrefix("team:")),
                isNear(make_query(cogs.net_tag()), radius=10),
            ],
            mutations=[
                updateActor({k: -v for k, v in ALIGN_COST.items()}),
                addTag("team:cogs"),
                recomputeMaterializedQuery(cogs.net_tag()),
            ],
        ),
    )

    hub = GridObjectConfig(
        name="hub",
        map_name=f"{cogs.short_name}:hub",
        tags=[cogs.team_tag()],
    )

    harness = StationTestHarness.create(
        station=junction,
        agent_inventory={"aligner": 1, "heart": 5, "hp": 100, "energy": 100},
        agent_team="cogs",
        tags=[cogs.team_tag()],
        extra_objects=[hub],
        materialize_queries=[net_materialized_query(cogs)],
    )

    harness.move_onto_station()

    inv = harness.agent_inventory()
    assert inv.get("heart", 0) == 4, f"Expected 4 hearts, got {inv.get('heart', 0)}"
    harness.close()


def test_alignment_updates_network_tag():
    """After alignment, junction receives the materialized net tag via recomputeMaterializedQuery."""
    cogs = TeamConfig(name="cogs", short_name="c")

    junction = GridObjectConfig(
        name="junction",
        on_use_handler=Handler(
            name="align_cogs",
            filters=[
                actorHasTag("team:cogs"),
                actorHas({"aligner": 1, **ALIGN_COST}),
                isNot(hasTagPrefix("team:")),
                isNear(make_query(cogs.net_tag()), radius=10),
            ],
            mutations=[
                updateActor({k: -v for k, v in ALIGN_COST.items()}),
                addTag("team:cogs"),
                recomputeMaterializedQuery(cogs.net_tag()),
            ],
        ),
    )

    hub = GridObjectConfig(
        name="hub",
        map_name=f"{cogs.short_name}:hub",
        tags=[cogs.team_tag()],
    )

    harness = StationTestHarness.create(
        station=junction,
        agent_inventory={"aligner": 1, "heart": 5, "hp": 100, "energy": 100},
        agent_team="cogs",
        tags=[cogs.team_tag()],
        extra_objects=[hub],
        materialize_queries=[net_materialized_query(cogs)],
    )

    harness.move_onto_station()

    assert has_tag_at(harness.simulation, "net:cogs", 2, 2, "junction"), (
        "Junction should have net:cogs after alignment and recomputeMaterializedQuery"
    )

    harness.close()


def test_chain_alignment_preserves_first_junction():
    """Aligning a second junction must not break the first junction's tags.

    This catches the on_tag_remove cascade bug: recomputeMaterializedQuery clears
    net:cogs from J1 → on_tag_remove fires → team:cogs removed from J1 → BFS
    can't find J1 as a bridge → network collapses.

    Map layout — junctions are vertically adjacent to hub so BFS can chain:
        #  #  #  #  #
        #  H  .  .  #   H at (1,1)
        #  J  @  .  #   J1 at (2,1), agent at (2,2)
        #  J  .  .  #   J2 at (3,1)
        #  #  #  #  #
    """
    cogs = TeamConfig(name="cogs", short_name="c")
    sim = make_junction_sim(
        teams=[cogs],
        map_data=[
            ["#", "#", "#", "#", "#"],
            ["#", "H", ".", ".", "#"],
            ["#", "J", "@", ".", "#"],
            ["#", "J", ".", ".", "#"],
            ["#", "#", "#", "#", "#"],
        ],
        char_to_map_name={
            "#": "wall",
            "@": "agent.agent",
            ".": "empty",
            "H": "c:hub",
            "J": "junction",
        },
        agent_inventory={"aligner": 1, "heart": 10, "hp": 100, "energy": 100},
        agent_team="cogs",
    )

    sim.agent(0).set_action("move_west")
    sim.step()

    assert has_tag_at(sim, "team:cogs", 2, 1, "junction"), "J1 should have team:cogs"
    assert has_tag_at(sim, "net:cogs", 2, 1, "junction"), "J1 should have net:cogs"

    sim.agent(0).set_action("move_south")
    sim.step()
    sim.agent(0).set_action("move_west")
    sim.step()

    assert has_tag_at(sim, "team:cogs", 3, 1, "junction"), "J2 should have team:cogs"
    assert has_tag_at(sim, "net:cogs", 3, 1, "junction"), "J2 should have net:cogs"

    assert has_tag_at(sim, "team:cogs", 2, 1, "junction"), (
        "J1 should still have team:cogs after J2 alignment (on_tag_remove cascade bug)"
    )
    assert has_tag_at(sim, "net:cogs", 2, 1, "junction"), "J1 should still have net:cogs after J2 alignment"
