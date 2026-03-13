from pathlib import Path

from cogames.cli.mission import get_mission


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
