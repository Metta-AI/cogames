"""Tests for the ForcedRoleVibesVariant."""

from cogsguard.game import ForcedRoleVibesVariant
from cogsguard.game.damage import DamageVariant
from cogsguard.game.teams import TeamConfig, TeamVariant
from cogsguard.game.vibes import VibesVariant
from cogsguard.missions.arena import make_arena_map_builder
from cogsguard.missions.mission import CvCMission


def test_forced_role_vibes_variant_forces_vibe_without_role_id_tokens() -> None:
    mission = CvCMission(
        name="basic",
        description="test",
        map_builder=make_arena_map_builder(num_agents=4),
        min_cogs=4,
        max_cogs=4,
        max_steps=100,
    ).with_variants(
        [
            TeamVariant(default_teams={"cogs": TeamConfig(num_agents=4)}),
            DamageVariant(),
            VibesVariant(),
            ForcedRoleVibesVariant(),
        ]
    )
    env = mission.make_env()

    assert "role_id" not in env.game.resource_names
    assert "inv:own:role_id" not in env.game.obs.global_obs.obs

    vibe_id_by_name = {name: idx for idx, name in enumerate(env.game.vibe_names)}
    expected_roles = ["miner", "aligner", "scrambler", "scout"]
    for agent_id, agent_cfg in enumerate(env.game.agents):
        expected_role_name = expected_roles[agent_id % 4]
        assert "role_id" not in agent_cfg.inventory.initial
        assert agent_cfg.vibe == vibe_id_by_name[expected_role_name]

    assert env.game.actions.change_vibe.enabled is False
