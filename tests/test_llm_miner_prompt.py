from cogames.policy.llm_miner_policy import _parse_skill_choice
from cogames.policy.llm_miner_prompt import build_llm_miner_prompt


def test_build_llm_miner_prompt_mentions_skills_and_state() -> None:
    prompt = build_llm_miner_prompt(
        carried_total=20,
        return_load=40,
        has_miner=True,
        hub_visible=False,
        remembered_hub=(3, -2),
        current_skill="mine_until_full",
        no_move_steps=2,
        recent_events=["cargo increased from 10 to 20"],
    )

    assert "mine_until_full" in prompt
    assert "deposit_to_hub" in prompt
    assert "unstuck" in prompt
    assert "carried_total: 20" in prompt
    assert "remembered_hub: spawn_relative_row=3, spawn_relative_col=-2" in prompt


def test_parse_skill_choice_accepts_json() -> None:
    skill, reason = _parse_skill_choice('{"skill":"deposit_to_hub","reason":"cargo full"}')
    assert skill == "deposit_to_hub"
    assert reason == "cargo full"


def test_parse_skill_choice_rejects_unknown_skill() -> None:
    skill, reason = _parse_skill_choice('{"skill":"do_anything","reason":"bad"}')
    assert skill is None
    assert reason == "bad"
