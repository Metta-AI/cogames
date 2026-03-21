from cogames.policy.llm_aligner_prompt import build_llm_aligner_prompt
from cogames.policy.machina_llm_roles_policy import _parse_role_skill_choice


def test_build_llm_aligner_prompt_mentions_skills_and_state() -> None:
    prompt = build_llm_aligner_prompt(
        has_aligner=True,
        has_heart=False,
        hub_visible=True,
        known_neutral_junctions=3,
        known_friendly_junctions=1,
        current_skill="explore",
        no_move_steps=2,
        recent_events=["discovered 3 neutral junctions"],
    )

    assert "get_heart" in prompt
    assert "align_neutral" in prompt
    assert "explore" in prompt
    assert "unstuck" in prompt
    assert "known_neutral_junctions: 3" in prompt
    assert "known_friendly_junctions: 1" in prompt


def test_parse_role_skill_choice_accepts_json() -> None:
    skill, reason = _parse_role_skill_choice(
        '{"skill":"align_neutral","reason":"neutral junctions are known"}',
        {"get_heart", "align_neutral", "explore", "unstuck"},
    )
    assert skill == "align_neutral"
    assert reason == "neutral junctions are known"
