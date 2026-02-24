from cogames.cogs_vs_clips.mission import CvCMission
from cogames.cogs_vs_clips.sites import make_cogsguard_machina1_site
from cogames.cogs_vs_clips.variants import NoClipsVariant


def test_no_clips_variant_disables_clips_system() -> None:
    mission = CvCMission(
        name="basic",
        description="test",
        site=make_cogsguard_machina1_site(num_agents=4),
        num_cogs=4,
        max_steps=100,
        variants=[NoClipsVariant()],
    )
    env = mission.make_env()

    assert mission.clips.disabled
    assert "neutral_to_clips" not in env.game.events
    assert "cogs_to_neutral" not in env.game.events
