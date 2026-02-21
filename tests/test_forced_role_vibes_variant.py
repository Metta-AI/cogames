from cogames.cogs_vs_clips.cog import CogTeam
from cogames.cogs_vs_clips.mission import CvCMission
from cogames.cogs_vs_clips.sites import make_cogsguard_arena_site
from cogames.cogs_vs_clips.variants import ForcedRoleVibesVariant
from mettagrid.config.game_value import InventoryValue, Scope


def test_forced_role_vibes_variant_adds_global_role_id_and_forces_vibe() -> None:
    mission = CvCMission(
        name="basic",
        description="test",
        site=make_cogsguard_arena_site(num_agents=4),
        teams={"cogs": CogTeam(num_agents=4)},
        max_steps=100,
        variants=[ForcedRoleVibesVariant()],
    )
    env = mission.make_env()

    assert "role_id" in env.game.resource_names
    assert InventoryValue(item="role_id", scope=Scope.AGENT) in list(env.game.obs.global_obs.obs)

    vibe_id_by_name = {name: idx for idx, name in enumerate(env.game.vibe_names)}
    expected_roles = ["miner", "aligner", "scrambler", "scout"]
    for agent_id, agent_cfg in enumerate(env.game.agents):
        expected_role_id = agent_id % 4
        assert agent_cfg.inventory.initial["role_id"] == expected_role_id
        assert agent_cfg.vibe == vibe_id_by_name[expected_roles[expected_role_id]]

    assert env.game.actions.change_vibe.enabled is False
