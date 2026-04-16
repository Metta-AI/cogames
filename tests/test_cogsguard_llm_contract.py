from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from cogames.sdk.cogsguard.llm_contract import (
    JunctionSnapshot,
    PlannerDirective,
    PlannerSkillOption,
    PlannerSummary,
    SkillName,
    SkillStatus,
    StrategyMode,
    build_planner_prompt,
    parse_planner_response,
    render_planner_library,
    render_skill_options,
)


def test_planner_directive_normalizes_talk() -> None:
    directive = PlannerDirective(
        mode=StrategyMode.FRONTIER_EXPAND,
        skill=SkillName.ALIGN_FRONTIER,
        talk="  claim   east frontier  ",
    )

    assert directive.talk == "claim east frontier"


def test_render_planner_library_lists_modes_and_skills() -> None:
    rendered = render_planner_library()

    assert "STRATEGY_MODES" in rendered
    assert "bootstrap_economy" in rendered
    assert "SKILLS" in rendered
    assert "align_frontier" in rendered


def test_render_skill_options_lists_status_and_role() -> None:
    rendered = render_skill_options(
        [
            PlannerSkillOption(
                name=SkillName.ALIGN_FRONTIER,
                status=SkillStatus.SETUP,
                reason="will acquire a heart first",
                preferred_role="aligner",
            )
        ]
    )

    assert "SKILL_OPTIONS" in rendered
    assert "align_frontier [setup, role=aligner]" in rendered
    assert "will acquire a heart first" in rendered


def _make_summary(**overrides) -> PlannerSummary:
    summary = PlannerSummary(
        step=500,
        agent_id=0,
        hp=80,
        position=(22, 44),
        role="miner",
        strategy_mode=StrategyMode.COVER_MISSING_RESOURCES,
        active_skill=SkillName.MINE_BIASED_RESOURCE,
        resource_bias="carbon",
        team_resources={"carbon": 10, "oxygen": 5, "germanium": 0, "silicon": 3},
        inventory={"heart": 3, "miner": 1, "carbon": 2},
        junctions=JunctionSnapshot(friendly=3, enemy=5, neutral=2),
        safe_distance=12,
        stalled=False,
        oscillating=False,
        has_gear=True,
        emergency_mining=False,
        roles="miner=4, aligner=2, scrambler=2",
        active_skill_steps=12,
        last_skill_exit_reason="explore completed: extractor seen",
        recent_events=["extractor_seen: Extractor oxygen@10,10 became visible."],
        skill_options=[
            PlannerSkillOption(
                name=SkillName.ALIGN_FRONTIER,
                status=SkillStatus.SETUP,
                reason="will acquire a heart before aligning",
                preferred_role="aligner",
            ),
            PlannerSkillOption(
                name=SkillName.EXPLORE,
                status=SkillStatus.READY,
                reason="reveals new map info, extractors, and junction targets",
            ),
        ],
        visible_talk=["agent-3: claim east frontier"],
        talk_enabled=True,
        talk_cooldown_steps=8,
    )
    return summary.model_copy(update=overrides)


def test_build_planner_prompt_uses_runtime_summary_contract() -> None:
    prompt = build_planner_prompt(_make_summary(strategy_mode=StrategyMode.BOOTSTRAP_ECONOMY, resource_bias="oxygen"))

    assert "CvC game step 500/10000" in prompt
    assert "Agent 0: HP=80, Hearts=3, Role=miner" in prompt
    assert "Current mode: bootstrap_economy" in prompt
    assert "Current active skill: mine_biased_resource (steps=12)" in prompt
    assert "Last skill exit: explore completed: extractor seen" in prompt
    assert "Current resource bias: oxygen" in prompt
    assert "RECENT_EVENTS" in prompt
    assert "extractor_seen: Extractor oxygen@10,10 became visible." in prompt
    assert "SKILL_OPTIONS" in prompt
    assert "align_frontier [setup, role=aligner]" in prompt
    assert "Visible teammate talk: agent-3: claim east frontier" in prompt
    assert "Talk enabled: True, talk cooldown steps: 8" in prompt


def test_build_planner_prompt_lists_supported_modes_and_skills() -> None:
    prompt = build_planner_prompt(_make_summary())

    assert "STRATEGY_MODES" in prompt
    assert "- scramble_pressure:" in prompt
    assert "- ready: can do the main job immediately" in prompt
    assert "SKILLS" in prompt
    assert "- align_frontier:" in prompt
    assert '"mode":"frontier_expand","skill":"align_frontier"' in prompt


def test_parse_planner_response_accepts_supported_fields() -> None:
    result = parse_planner_response(
        json.dumps(
            {
                "mode": "frontier_expand",
                "skill": "align_frontier",
                "resource_bias": "carbon",
                "talk": "claim east frontier",
                "analysis": "Push the frontier while coordinating locally.",
            }
        )
    )

    assert result.mode == StrategyMode.FRONTIER_EXPAND
    assert result.skill == SkillName.ALIGN_FRONTIER
    assert result.resource_bias == "carbon"
    assert result.talk == "claim east frontier"
    assert result.analysis == "Push the frontier while coordinating locally."


def test_parse_planner_response_accepts_fenced_json() -> None:
    result = parse_planner_response(
        """```json
{
  "mode": "frontier_expand",
  "skill": "explore",
  "analysis": "Scout first."
}
```"""
    )

    assert result.mode == StrategyMode.FRONTIER_EXPAND
    assert result.skill == SkillName.EXPLORE
    assert result.analysis == "Scout first."


def test_parse_planner_response_accepts_prefixed_json() -> None:
    result = parse_planner_response(
        """Looking at the current situation, we should pressure the frontier.

{"mode":"convert_frontier","skill":"align_frontier","analysis":"Convert the known frontier now."}"""
    )

    assert result.mode == StrategyMode.CONVERT_FRONTIER
    assert result.skill == SkillName.ALIGN_FRONTIER
    assert result.analysis == "Convert the known frontier now."


def test_parse_planner_response_recovers_from_truncated_json() -> None:
    result = parse_planner_response(
        "Looking at the current situation, we have pressure opportunities.\n\n"
        '{"mode":"scramble_pressure","skill":"scramble_frontier","analysis":"Keep enemy junctions unstable"'
    )

    assert result.mode == StrategyMode.SCRAMBLE_PRESSURE
    assert result.skill == SkillName.SCRAMBLE_FRONTIER
    assert "Looking at the current situation" in result.analysis


@pytest.mark.parametrize("mode", [mode.value for mode in StrategyMode])
def test_parse_planner_response_accepts_supported_modes(mode: str) -> None:
    result = parse_planner_response(json.dumps({"mode": mode, "skill": "explore", "analysis": "ok"}))

    assert result.mode.value == mode


def test_planner_directive_rejects_unknown_skill() -> None:
    with pytest.raises(ValidationError):
        PlannerDirective.model_validate_json(
            json.dumps({"mode": "frontier_expand", "skill": "invent_new_skill", "analysis": "ok"})
        )


def test_planner_directive_rejects_unknown_mode() -> None:
    with pytest.raises(ValidationError):
        PlannerDirective.model_validate_json(
            json.dumps({"mode": "economy_bootstrap", "skill": "explore", "analysis": "ok"})
        )


def test_parse_planner_response_falls_back_for_plain_text() -> None:
    result = parse_planner_response("This is not JSON at all, just plain text analysis.")

    assert result.mode == StrategyMode.FRONTIER_EXPAND
    assert result.skill == SkillName.EXPLORE
    assert result.analysis == "This is not JSON at all, just plain text analysis."
