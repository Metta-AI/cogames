from pathlib import Path

from cogames.cli.mission import get_mission
from cogames.games.cogs_vs_clips.missions.terrain import find_machina_arena


def test_get_mission_accepts_reward_variants() -> None:
    _, env_cfg, _ = get_mission(
        "machina_1",
        variants_arg=["aligner", "miner", "scrambler"],
        cogs=8,
    )

    rewards = env_cfg.game.agents[0].rewards
    assert "junction_aligned_by_agent" in rewards
    assert "junction_scrambled_by_agent" in rewards
    assert "gain_diversity" in rewards


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

    _, env_cfg, mission = get_mission(str(mission_file))
    assert mission is None
    assert env_cfg.game.max_steps == 777


def test_cogs_override_updates_spawn_count_and_mission_counts() -> None:
    _, env_cfg, mission = get_mission("machina_1", cogs=2)

    assert mission is not None
    arena = find_machina_arena(env_cfg.game.map_builder)
    assert arena is not None
    assert mission.num_cogs == 2
    assert mission.model_dump()["num_agents"] == 2
    assert env_cfg.game.num_agents == 2
    assert arena.spawn_count == 2
