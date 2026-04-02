from __future__ import annotations

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


def test_inventory_items_ignore_zero_values() -> None:
    impl = StarterCogPolicyImpl(_mock_policy_env_info(), agent_id=0, preferred_gear="miner")
    obs = AgentObservation(
        agent_id=0,
        tokens=[
            _inv_token(1, "inv:miner", 0),
            _inv_token(2, "inv:scout", 0),
            _inv_token(3, "inv:aligner", 0),
            _inv_token(4, "inv:scrambler", 0),
            _inv_token(5, "inv:heart", 0),
        ],
    )

    items = impl._inventory_amounts(obs)
    assert items == {}
    assert impl._current_gear(items) is None


def test_preferred_role_targets_its_station_when_not_equipped() -> None:
    impl = StarterCogPolicyImpl(_mock_policy_env_info(), agent_id=0, preferred_gear="miner")
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


def test_inventory_items_parse_power_suffix_and_non_power_colon_names() -> None:
    impl = StarterCogPolicyImpl(_mock_policy_env_info(), agent_id=0, preferred_gear=None)
    obs = AgentObservation(
        agent_id=0,
        tokens=[
            _inv_token(1, "inv:carbon", 34, normalization=100.0),
            _inv_token(2, "inv:carbon:p1", 12, normalization=100.0),
            _inv_token(3, "inv:own:policy", 7, normalization=100.0),
        ],
    )

    items = impl._inventory_amounts(obs)
    assert items["carbon"] == 1234
    assert items["own:policy"] == 7


def test_inventory_items_ignore_empty_suffix_tokens() -> None:
    impl = StarterCogPolicyImpl(_mock_policy_env_info(), agent_id=0, preferred_gear=None)
    obs = AgentObservation(
        agent_id=0,
        tokens=[
            _inv_token(1, "inv:", 5),
            _inv_token(2, "inv:miner", 1),
        ],
    )

    items = impl._inventory_amounts(obs)
    assert "" not in items
    assert items["miner"] == 1


def test_aligner_without_heart_targets_hub() -> None:
    impl = StarterCogPolicyImpl(_mock_policy_env_info(), agent_id=0, preferred_gear=None)
    state = impl.initial_agent_state()
    obs = AgentObservation(
        agent_id=0,
        tokens=[
            _inv_token(1, "inv:aligner", 1),
            _inv_token(2, "inv:heart", 0),
            _tag_token(2, row=3, col=2),  # type:hub (south)
            _tag_token(1, row=1, col=2),  # type:junction (north)
        ],
    )

    action, _ = impl.step_with_state(obs, state)
    assert action.name == "move_south"


def test_cogsguard_tags_resolve_role_station_targets() -> None:
    policy_env_info = PolicyEnvInterface.from_mg_cfg(make_basic_mission().make_env())
    for role in ["miner", "aligner", "scrambler", "scout"]:
        impl = StarterCogPolicyImpl(policy_env_info, agent_id=0, preferred_gear=role)
        expected_tag_name = f"type:{role}"
        expected_tag_id = policy_env_info.tags.index(expected_tag_name)
        assert expected_tag_id in impl._gear_station_tags_by_gear[role]
