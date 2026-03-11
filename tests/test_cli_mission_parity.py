from pathlib import Path

import pytest

from cogames.cli.mission import get_mission
from cogames.cogs_vs_clips.buildings import MachinaArenaConfig
from mettagrid.mapgen.mapgen import MapGenConfig


def test_get_mission_accepts_reward_variants() -> None:
    _, env_cfg, _ = get_mission(
        "cogsguard_machina_1.basic",
        variants_arg=["aligner", "miner", "scrambler"],
        cogs=8,
    )

    rewards = env_cfg.game.agents[0].rewards
    assert "junction_aligned_by_agent" in rewards
    assert "junction_scrambled_by_agent" in rewards
    assert "gain_diversity" in rewards
    assert env_cfg.label.endswith(".miner.aligner.scrambler")


def test_get_mission_accepts_mixed_mission_and_reward_variants() -> None:
    _, env_cfg, _ = get_mission(
        "cogsguard_machina_1.basic",
        variants_arg=["dark_side", "aligner"],
        cogs=8,
    )

    assert ".dark_side" in env_cfg.label
    assert env_cfg.label.endswith(".aligner")
    assert "junction_aligned_by_agent" in env_cfg.game.agents[0].rewards


def test_get_mission_steps_rebuilds_timestep_dependent_config() -> None:
    _, default_env, _ = get_mission("cogsguard_machina_1.basic", cogs=8)
    _, short_env, _ = get_mission("cogsguard_machina_1.basic", cogs=8, steps=1000)

    assert default_env.game.max_steps == 10000
    assert short_env.game.max_steps == 1000

    default_timesteps = default_env.game.events["neutral_to_clips"].timesteps
    short_timesteps = short_env.game.events["neutral_to_clips"].timesteps
    assert short_timesteps == [t for t in default_timesteps if t < short_env.game.max_steps]
    assert len(short_timesteps) < len(default_timesteps)

    reward_cfg = short_env.game.agents[0].rewards["aligned_junction_held"].model_dump(mode="python")
    assert reward_cfg["reward"]["weights"] == pytest.approx([1 / 1000.0])


def test_steps_applied_before_timestep_dependent_variants() -> None:
    _, _, mission = get_mission(
        "cogsguard_machina_1.basic",
        variants_arg=["clips_wave_only"],
        cogs=8,
        steps=20000,
    )
    assert mission is not None
    assert mission.max_steps == 20000
    assert mission.clips.scramble_start == 20001
    assert mission.clips.scramble_interval == 20001
    assert mission.clips.align_start == 20001
    assert mission.clips.align_interval == 20001


def test_file_mission_honors_steps_override(tmp_path: Path) -> None:
    mission_file = tmp_path / "mission.py"
    mission_file.write_text(
        "\n".join(
            [
                "from mettagrid.config.mettagrid_config import MettaGridConfig",
                "",
                "def get_config():",
                "    cfg = MettaGridConfig.EmptyRoom(num_agents=1)",
                "    cfg.game.max_steps = 777",
                "    return cfg",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _, env_cfg, mission = get_mission(str(mission_file), steps=123)
    assert mission is None
    assert env_cfg.game.max_steps == 123


@pytest.mark.parametrize("mission_name", ["cogsguard_machina_1.basic", "cogsguard_arena.basic"])
def test_num_cogs_variant_updates_machina_spawn_count(mission_name: str) -> None:
    _, env_cfg, _ = get_mission(mission_name, cogs=8)

    assert isinstance(env_cfg.game.map_builder, MapGenConfig)
    map_instance = env_cfg.game.map_builder.instance
    assert isinstance(map_instance, MachinaArenaConfig)
    assert map_instance.spawn_count == 8
