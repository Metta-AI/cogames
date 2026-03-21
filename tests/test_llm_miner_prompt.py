from cogames.cli.mission import get_mission
from cogames.policy.llm_miner_policy import LLMMinerPlannerClient, LLMMinerPolicyImpl, _parse_skill_choice
from cogames.policy.llm_miner_prompt import build_llm_miner_prompt
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import AgentObservation


def test_build_llm_miner_prompt_mentions_skills_and_state() -> None:
    prompt = build_llm_miner_prompt(
        carried_total=20,
        return_load=40,
        has_miner=True,
        hub_visible=False,
        remembered_hub=(3, -2),
        known_extractors=0,
        frontier_count=12,
        current_skill="mine_until_full",
        no_move_steps=2,
        recent_events=["cargo increased from 10 to 20"],
    )

    assert "mine_until_full" in prompt
    assert "deposit_to_hub" in prompt
    assert "explore" in prompt
    assert "unstuck" in prompt
    assert "carried_total: 20" in prompt
    assert "known_extractors: 0" in prompt
    assert "frontier_count: 12" in prompt
    assert "remembered_hub: spawn_relative_row=3, spawn_relative_col=-2" in prompt


def test_parse_skill_choice_accepts_json() -> None:
    skill, reason = _parse_skill_choice('{"skill":"deposit_to_hub","reason":"cargo full"}')
    assert skill == "deposit_to_hub"
    assert reason == "cargo full"


def test_parse_skill_choice_rejects_unknown_skill() -> None:
    skill, reason = _parse_skill_choice('{"skill":"do_anything","reason":"bad"}')
    assert skill is None
    assert reason == "bad"


def _make_miner_policy(responder) -> LLMMinerPolicyImpl:
    env_cfg = get_mission("cogsguard_arena.basic")[1]
    return LLMMinerPolicyImpl(
        PolicyEnvInterface.from_mg_cfg(env_cfg),
        agent_id=0,
        planner=LLMMinerPlannerClient(responder=responder),
        return_load=40,
        stuck_threshold=6,
        unstuck_horizon=4,
    )


def test_miner_planner_keeps_unstuck_when_gear_is_missing() -> None:
    policy = _make_miner_policy(lambda _: '{"skill":"unstuck","reason":"blocked near station"}')
    state = policy.initial_agent_state()
    obs = AgentObservation(agent_id=0, tokens=[])

    policy._starter._current_gear = lambda _items: "none"  # type: ignore[method-assign]
    policy._starter._inventory_items = lambda _obs: []  # type: ignore[method-assign]

    policy._plan_skill(obs, state)

    assert state.current_skill == "unstuck"
    assert state.current_reason == "blocked near station"
