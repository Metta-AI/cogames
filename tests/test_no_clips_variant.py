from cogames.cogs_vs_clips.clip_difficulty import EASY
from cogames.cogs_vs_clips.mission import CvCMission
from cogames.cogs_vs_clips.ships import count_clips_ships_in_map_config
from cogames.cogs_vs_clips.sites import make_cogsguard_machina1_site
from cogames.cogs_vs_clips.variants import NoClipsVariant


def test_no_clips_variant_removes_clips_ships_and_events() -> None:
    mission = CvCMission(
        name="basic",
        description="test",
        site=make_cogsguard_machina1_site(num_agents=4),
        num_cogs=4,
        max_steps=100,
        variants=[NoClipsVariant()],
    )
    env = mission.make_env()

    assert count_clips_ships_in_map_config(mission.site.map_builder) == 0
    assert "neutral_to_clips" not in env.game.events
    assert "cogs_to_neutral" not in env.game.events


def test_easy_difficulty_removes_clips_ships_and_events() -> None:
    mission = CvCMission(
        name="basic",
        description="test",
        site=make_cogsguard_machina1_site(num_agents=4),
        num_cogs=4,
        max_steps=100,
        variants=[EASY],
    )
    env = mission.make_env()

    assert count_clips_ships_in_map_config(mission.site.map_builder) == 0
    assert "neutral_to_clips" not in env.game.events
    assert "cogs_to_neutral" not in env.game.events
