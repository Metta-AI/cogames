from cogames.policy.llm_aligner_prompt import build_llm_aligner_prompt
from cogames.policy.machina_llm_roles_policy import (
    LLMAlignerPolicyImpl,
    LLMMinerPlannerClient,
    MachinaLLMRolesPolicy,
    _parse_role_skill_choice,
)
from cogames.cli.mission import get_mission
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import AgentObservation


def test_build_llm_aligner_prompt_mentions_skills_and_state() -> None:
    prompt = build_llm_aligner_prompt(
        has_aligner=True,
        has_heart=False,
        hub_visible=True,
        known_hubs=1,
        known_neutral_junctions=3,
        known_alignable_junctions=2,
        known_friendly_junctions=1,
        current_skill="explore",
        no_move_steps=2,
        recent_events=["discovered 3 neutral junctions"],
    )

    assert "get_heart" in prompt
    assert "align_neutral" in prompt
    assert "explore" in prompt
    assert "unstuck" in prompt
    assert "known_hubs: 1" in prompt
    assert "known_neutral_junctions: 3" in prompt
    assert "known_alignable_junctions: 2" in prompt
    assert "known_friendly_junctions: 1" in prompt


def test_parse_role_skill_choice_accepts_json() -> None:
    skill, reason = _parse_role_skill_choice(
        '{"skill":"align_neutral","reason":"neutral junctions are known"}',
        {"get_heart", "align_neutral", "explore", "unstuck"},
    )
    assert skill == "align_neutral"
    assert reason == "neutral junctions are known"


def _make_aligner_policy(responder) -> LLMAlignerPolicyImpl:
    env_cfg = get_mission("cogsguard_arena.basic")[1]
    return LLMAlignerPolicyImpl(
        PolicyEnvInterface.from_mg_cfg(env_cfg),
        agent_id=0,
        planner=LLMMinerPlannerClient(responder=responder),
        stuck_threshold=6,
        unstuck_horizon=4,
    )


def test_aligner_planner_overrides_explore_to_get_heart_when_hub_is_known() -> None:
    policy = _make_aligner_policy(lambda _: '{"skill":"explore","reason":"look around"}')
    state = policy.initial_agent_state()
    state.known_hubs = {(0, 0)}
    obs = AgentObservation(agent_id=0, tokens=[])

    policy._current_gear = lambda _: "aligner"  # type: ignore[method-assign]
    policy._inventory_count = lambda _obs, item: 0 if item == "heart" else 0  # type: ignore[method-assign]

    policy._plan_skill(obs, state)

    assert state.current_skill == "get_heart"
    assert "hub is known" in state.current_reason


def test_aligner_planner_overrides_explore_to_align_neutral_when_target_known() -> None:
    policy = _make_aligner_policy(lambda _: '{"skill":"explore","reason":"look around"}')
    state = policy.initial_agent_state()
    state.known_hubs = {(0, 0)}
    state.known_neutral_junctions = {(0, 5)}
    obs = AgentObservation(agent_id=0, tokens=[])

    policy._current_gear = lambda _: "aligner"  # type: ignore[method-assign]
    policy._inventory_count = lambda _obs, item: 1 if item == "heart" else 0  # type: ignore[method-assign]

    policy._plan_skill(obs, state)

    assert state.current_skill == "align_neutral"
    assert "alignable neutral junction" in state.current_reason


def test_machina_llm_roles_defaults_to_one_aligner() -> None:
    env_cfg = get_mission("cogsguard_machina_1")[1]
    policy = MachinaLLMRolesPolicy(
        PolicyEnvInterface.from_mg_cfg(env_cfg),
        llm_responder=lambda _: '{"skill":"explore","reason":"test"}',
    )

    assert policy._aligner_ids == frozenset({0})


def test_machina_llm_roles_accepts_explicit_aligner_ids() -> None:
    env_cfg = get_mission("cogsguard_machina_1")[1]
    policy = MachinaLLMRolesPolicy(
        PolicyEnvInterface.from_mg_cfg(env_cfg),
        aligner_ids="0,2",
        llm_responder=lambda _: '{"skill":"explore","reason":"test"}',
    )

    assert policy._aligner_ids == frozenset({0, 2})
