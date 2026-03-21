from cogames.cli.mission import get_mission
from cogames.policy.aligner_agent import AlignerPolicyImpl
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import AgentObservation


def _make_policy() -> AlignerPolicyImpl:
    env_cfg = get_mission("cogsguard_arena.basic")[1]
    return AlignerPolicyImpl(PolicyEnvInterface.from_mg_cfg(env_cfg), agent_id=0)


def test_move_toward_target_uses_frontier_when_goal_is_unmapped() -> None:
    policy = _make_policy()
    state = policy.initial_agent_state()
    state.known_free_cells = {(0, 0), (0, 1)}

    action, _ = policy._move_toward_target(state, current_abs=(0, 0), target_abs=(0, 2))

    assert action.name == "move_east"


def test_explore_uses_map_memory_before_wandering() -> None:
    policy = _make_policy()
    state = policy.initial_agent_state()
    state.known_free_cells = {(0, 0), (0, 1), (0, 2)}
    state.blocked_cells = {(-1, 0), (1, 0), (0, -1)}
    obs = AgentObservation(agent_id=0, tokens=[])

    action, _ = policy._explore(obs, state)

    assert action.name == "move_east"
