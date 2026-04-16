from __future__ import annotations

import pytest

from cogames.games.cogs_vs_clips.missions.arena import make_basic_mission
from cogames.policy.starter_agent import StarterCogPolicyImpl
from mettagrid.config.id_map import ObservationFeatureSpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import AgentObservation, ObservationToken
from mettagrid.simulator.interface import PackedCoordinate


def _mock_policy_env_info() -> PolicyEnvInterface:
    action_names = ["noop", "move_north", "move_south", "move_east", "move_west"]
    tags = [
        "type:miner",
        "type:junction",
        "type:hub",
        "type:aligner",
        "type:scrambler",
        "type:scout",
        "type:carbon_extractor",
        "type:oxygen_extractor",
        "type:germanium_extractor",
        "type:silicon_extractor",
        "team:cogs",
        "team:clips",
        "type:agent",
        "type:wall",
    ]
    return PolicyEnvInterface(
        obs_features=[],
        tags=tags,
        action_names=action_names,
        vibe_action_names=[],
        num_agents=1,
        observation_shape=(5, 5, 3),
        egocentric_shape=(5, 5),
    )


def _inv_token(
    feature_id: int,
    name: str,
    value: int,
    *,
    row: int = 2,
    col: int = 2,
    normalization: float = 1.0,
) -> ObservationToken:
    feature = ObservationFeatureSpec(id=feature_id, name=name, normalization=normalization)
    packed = PackedCoordinate.pack(row, col)
    return ObservationToken(feature=feature, value=value, raw_token=(packed, feature_id, value))


def _tag_token(tag_id: int, *, row: int, col: int) -> ObservationToken:
    feature = ObservationFeatureSpec(id=100, name="tag", normalization=1.0)
    packed = PackedCoordinate.pack(row, col)
    return ObservationToken(feature=feature, value=tag_id, raw_token=(packed, 100, tag_id))


def _feature_token(feature_id: int, name: str, value: int, *, row: int = 2, col: int = 2) -> ObservationToken:
    feature = ObservationFeatureSpec(id=feature_id, name=name, normalization=1.0)
    packed = PackedCoordinate.pack(row, col)
    return ObservationToken(feature=feature, value=value, raw_token=(packed, feature_id, value))


def test_preferred_role_targets_its_station_when_not_equipped() -> None:
    impl = StarterCogPolicyImpl(_mock_policy_env_info(), agent_id=0, role="miner")
    state = impl.initial_agent_state()
    obs = AgentObservation(
        agent_id=0,
        tokens=[
            _inv_token(1, "inv:miner", 0),
            _inv_token(2, "inv:scout", 0),
            _inv_token(3, "inv:aligner", 0),
            _inv_token(4, "inv:scrambler", 0),
            _inv_token(5, "inv:heart", 0),
            _tag_token(10, row=2, col=2),  # team:cogs on self
            _tag_token(0, row=2, col=3),  # type:miner (east)
            _tag_token(10, row=2, col=3),  # team:cogs on station
            _tag_token(1, row=1, col=2),  # type:junction (north)
        ],
    )

    action, _ = impl.step_with_state(obs, state)
    assert action.name == "move_east"


def test_default_roles_cycle_by_agent_id() -> None:
    policy_env_info = _mock_policy_env_info()
    assert StarterCogPolicyImpl(policy_env_info, agent_id=0)._role == "miner"
    assert StarterCogPolicyImpl(policy_env_info, agent_id=1)._role == "aligner"
    assert StarterCogPolicyImpl(policy_env_info, agent_id=2)._role == "miner"
    assert StarterCogPolicyImpl(policy_env_info, agent_id=3)._role == "aligner"
    assert StarterCogPolicyImpl(policy_env_info, agent_id=4)._role == "miner"


def test_aligner_without_heart_targets_hub() -> None:
    impl = StarterCogPolicyImpl(_mock_policy_env_info(), agent_id=1)
    state = impl.initial_agent_state()
    obs = AgentObservation(
        agent_id=1,
        tokens=[
            _inv_token(1, "inv:aligner", 1),
            _inv_token(2, "inv:heart", 0),
            _tag_token(2, row=3, col=2),  # type:hub (south)
            _tag_token(1, row=1, col=2),  # type:junction (north)
        ],
    )

    action, _ = impl.step_with_state(obs, state)
    assert action.name == "move_south"


def test_aligner_with_heart_targets_neutral_junction() -> None:
    impl = StarterCogPolicyImpl(_mock_policy_env_info(), agent_id=1)
    state = impl.initial_agent_state()
    obs = AgentObservation(
        agent_id=1,
        tokens=[
            _inv_token(1, "inv:aligner", 1),
            _inv_token(2, "inv:heart", 1),
            _tag_token(1, row=3, col=2),  # neutral junction south
            _tag_token(1, row=1, col=2),  # enemy junction north
            _tag_token(11, row=1, col=2),  # team:clips on north junction
        ],
    )

    action, _ = impl.step_with_state(obs, state)
    assert action.name == "move_south"


def test_aligner_without_visible_hub_uses_remembered_hub() -> None:
    impl = StarterCogPolicyImpl(_mock_policy_env_info(), agent_id=1)
    state = impl.initial_agent_state()
    state.seen_tags_by_position[(2, 0)] = {2, 10}  # remembered hub on our team
    obs = AgentObservation(
        agent_id=1,
        tokens=[
            _inv_token(1, "inv:aligner", 1),
            _inv_token(2, "inv:heart", 0),
            _tag_token(10, row=2, col=2),  # team:cogs on self
        ],
    )

    action, _ = impl.step_with_state(obs, state)
    assert action.name == "move_south"


def test_aligner_with_heart_explores_instead_of_chasing_far_remembered_junction() -> None:
    impl = StarterCogPolicyImpl(_mock_policy_env_info(), agent_id=1)
    state = impl.initial_agent_state()
    state.seen_tags_by_position[(30, 0)] = {1}  # remembered neutral junction far beyond local frontier
    obs = AgentObservation(
        agent_id=1,
        tokens=[
            _inv_token(1, "inv:aligner", 1),
            _inv_token(2, "inv:heart", 1),
            _tag_token(10, row=2, col=2),  # team:cogs on self
        ],
    )

    action, _ = impl.step_with_state(obs, state)
    assert action.name == "move_south"


def test_scrambler_with_heart_targets_enemy_junction() -> None:
    impl = StarterCogPolicyImpl(_mock_policy_env_info(), agent_id=2, role="scrambler")
    state = impl.initial_agent_state()
    obs = AgentObservation(
        agent_id=2,
        tokens=[
            _inv_token(1, "inv:scrambler", 1),
            _inv_token(2, "inv:heart", 1),
            _tag_token(10, row=2, col=2),  # team:cogs on self
            _tag_token(1, row=1, col=2),  # enemy junction north
            _tag_token(11, row=1, col=2),  # team:clips on north junction
            _tag_token(1, row=3, col=2),  # neutral junction south
        ],
    )

    action, _ = impl.step_with_state(obs, state)
    assert action.name == "move_north"


def test_miner_with_cargo_targets_friendly_deposit() -> None:
    impl = StarterCogPolicyImpl(_mock_policy_env_info(), agent_id=0)
    state = impl.initial_agent_state()
    obs = AgentObservation(
        agent_id=0,
        tokens=[
            _inv_token(1, "inv:miner", 1),
            _inv_token(2, "inv:carbon:p1", 2, normalization=10.0),
            _tag_token(10, row=2, col=2),  # team:cogs on self
            _tag_token(2, row=1, col=2),  # hub north
            _tag_token(10, row=1, col=2),  # team:cogs on hub
            _tag_token(1, row=3, col=2),  # enemy junction south
            _tag_token(11, row=3, col=2),  # team:clips on junction
        ],
    )

    action, _ = impl.step_with_state(obs, state)
    assert action.name == "move_north"


def test_pathing_avoids_visible_agent_blockers() -> None:
    impl = StarterCogPolicyImpl(_mock_policy_env_info(), agent_id=1)
    state = impl.initial_agent_state()
    obs = AgentObservation(
        agent_id=1,
        tokens=[
            _tag_token(12, row=3, col=2),  # type:agent blocker directly south
            _tag_token(3, row=4, col=3),  # type:aligner station (south-east)
        ],
    )

    action, _ = impl.step_with_state(obs, state)
    assert action.name == "move_east"


def test_unreachable_visible_target_falls_back_to_explore() -> None:
    impl = StarterCogPolicyImpl(_mock_policy_env_info(), agent_id=0)
    state = impl.initial_agent_state()
    obs = AgentObservation(
        agent_id=0,
        tokens=[
            _inv_token(1, "inv:miner", 1),
            _tag_token(6, row=0, col=0),  # carbon extractor in the corner
            _tag_token(12, row=0, col=1),  # blocker
            _tag_token(12, row=1, col=0),  # blocker
        ],
    )

    action, _ = impl.step_with_state(obs, state)
    assert action.name == "move_east"


def test_duplicate_default_roles_start_with_different_wander_priorities() -> None:
    policy_env_info = _mock_policy_env_info()
    obs = AgentObservation(agent_id=0, tokens=[])
    first_impl = StarterCogPolicyImpl(policy_env_info, agent_id=0)
    second_impl = StarterCogPolicyImpl(policy_env_info, agent_id=4)
    first_action, _ = first_impl.step_with_state(obs, first_impl.initial_agent_state())
    second_action, _ = second_impl.step_with_state(obs, second_impl.initial_agent_state())
    assert first_action.name == "move_east"
    assert second_action.name == "move_west"


def test_starter_policy_requires_canonical_move_actions() -> None:
    policy_env_info = _mock_policy_env_info().model_copy(
        update={"action_names": ["noop", "move_north", "move_south", "move_west"]}
    )
    with pytest.raises(AssertionError, match="Starter policy requires actions"):
        StarterCogPolicyImpl(policy_env_info, agent_id=0)


def test_starter_policy_requires_canonical_tags() -> None:
    policy_env_info = _mock_policy_env_info().model_copy(
        update={"tags": [tag for tag in _mock_policy_env_info().tags if tag != "type:wall"]}
    )
    with pytest.raises(AssertionError, match="Starter policy requires tags"):
        StarterCogPolicyImpl(policy_env_info, agent_id=0)


def test_explore_persists_heading_instead_of_immediate_backtrack() -> None:
    impl = StarterCogPolicyImpl(_mock_policy_env_info(), agent_id=0, role="scout")
    state = impl.initial_agent_state()

    first_obs = AgentObservation(
        agent_id=0,
        tokens=[
            _inv_token(1, "inv:scout", 1),
            _tag_token(12, row=1, col=2),  # north blocked
            _tag_token(12, row=2, col=1),  # west blocked
            _tag_token(12, row=2, col=3),  # east blocked
        ],
    )
    first_action, state = impl.step_with_state(first_obs, state)

    second_obs = AgentObservation(
        agent_id=0,
        tokens=[
            _inv_token(1, "inv:scout", 1),
            _feature_token(2, "last_action_move", 1),
            _tag_token(12, row=2, col=3),  # east blocked
            _tag_token(12, row=3, col=2),  # south blocked
        ],
    )
    second_action, state = impl.step_with_state(second_obs, state)

    third_obs = AgentObservation(
        agent_id=0,
        tokens=[
            _inv_token(1, "inv:scout", 1),
            _feature_token(2, "last_action_move", 1),
            _tag_token(12, row=2, col=1),  # west blocked
            _tag_token(12, row=3, col=2),  # south blocked
        ],
    )
    third_action, _ = impl.step_with_state(third_obs, state)

    assert first_action.name == "move_south"
    assert second_action.name == "move_west"
    assert third_action.name == "move_north"


def test_failed_move_into_agent_is_not_remembered_as_wall() -> None:
    impl = StarterCogPolicyImpl(_mock_policy_env_info(), agent_id=0, role="scout")
    state = impl.initial_agent_state()
    state.last_move_direction = "east"

    blocked_obs = AgentObservation(
        agent_id=0,
        tokens=[
            _inv_token(1, "inv:scout", 1),
            _tag_token(12, row=2, col=3),  # type:agent blocker east
        ],
    )
    _, state = impl.step_with_state(blocked_obs, state)

    assert (0, 1) not in state.blocked


def test_cogsguard_tags_resolve_role_station_targets() -> None:
    policy_env_info = PolicyEnvInterface.from_mg_cfg(make_basic_mission().make_env())
    for role in ["miner", "aligner", "scrambler", "scout"]:
        impl = StarterCogPolicyImpl(policy_env_info, agent_id=0, role=role)
        expected_tag_name = f"type:{role}"
        expected_tag_id = policy_env_info.tags.index(expected_tag_name)
        assert expected_tag_id in impl._role_station_tags[role]
