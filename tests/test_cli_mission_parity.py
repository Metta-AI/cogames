from pathlib import Path

import pytest
from cogsguard.game.clips import AngryClipsVariant, ClipsVariant
from cogsguard.game.clips.clips import JUNCTION_ALIGN_DISTANCE
from cogsguard.game.game import CvCGame
from cogsguard.missions.machina_1 import MachinaOneMission
from cogsguard.missions.terrain import find_machina_arena

from cogames.cli.mission import find_mission, get_mission, resolve_mission


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


def test_get_mission_accepts_angry_clips_variant() -> None:
    _, env_cfg, mission = get_mission(
        "machina_1",
        variants_arg=["angry_clips"],
        cogs=8,
    )

    assert mission is not None
    clips_v = mission.required_variant(ClipsVariant)
    assert clips_v.clips is not None
    assert clips_v.clips.angry_target_enemy_hub is True
    assert clips_v.clips.greedy_expand_from_ships is False
    assert clips_v.clips.scramble_radius == JUNCTION_ALIGN_DISTANCE
    assert "neutral_to_clips" in env_cfg.game.events


def test_parametrized_angry_clips_variant_updates_mission_settings() -> None:
    mission = MachinaOneMission().with_variants(
        [
            AngryClipsVariant(
                initial_clips_start=25,
                align_start=50,
                scramble_start=75,
                align_interval=10,
                scramble_interval=15,
            )
        ]
    )
    env_cfg = mission.make_env()

    clips_v = mission.required_variant(ClipsVariant)
    assert clips_v.clips is not None
    assert clips_v.clips.angry_target_enemy_hub is True
    assert clips_v.clips.greedy_expand_from_ships is False
    assert clips_v.clips.scramble_radius == JUNCTION_ALIGN_DISTANCE
    assert clips_v.clips.initial_clips_start == 25
    assert clips_v.clips.align_start == 50
    assert clips_v.clips.scramble_start == 75
    assert clips_v.clips.align_interval == 10
    assert clips_v.clips.scramble_interval == 15
    assert "neutral_to_clips" in env_cfg.game.events


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


def test_cogs_selection_updates_spawn_count_and_mission_counts() -> None:
    _, env_cfg, mission = get_mission("machina_1", cogs=2)

    assert mission is not None
    arena = find_machina_arena(env_cfg.game.map_builder)
    assert arena is not None
    assert mission.num_cogs == 2
    assert mission.model_dump()["num_agents"] == 2
    assert env_cfg.game.num_agents == 2
    assert arena.spawn_count == 2


def test_all_listed_sub_missions_resolve_via_cli_lookup() -> None:
    game = CvCGame()

    for mission in game.missions:
        for sub_name in mission.sub_missions:
            assert find_mission(game, f"{mission.name}.{sub_name}") is not None


def test_get_mission_accepts_talk_variant() -> None:
    _, env_cfg, _ = get_mission("machina_1", variants_arg=["talk"])

    assert env_cfg.game.actions.change_vibe.enabled is False
    assert env_cfg.game.talk.enabled is True
    assert env_cfg.game.talk.max_length == 140
    assert env_cfg.game.talk.cooldown_steps == 50


def test_resolve_mission_rejects_variants_from_other_games() -> None:
    game = CvCGame()

    with pytest.raises(ValueError, match="Unknown variant 'full'"):
        resolve_mission(game, "machina_1", variants_arg=["full"])
