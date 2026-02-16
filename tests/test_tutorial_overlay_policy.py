from typing import cast

from cogames.policy.tutorial_overlay_policy import TutorialOverlayPolicy
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import AgentObservation


def _step_tutorial_policy(tutorial: str) -> dict[str, object]:
    cfg = MettaGridConfig.EmptyRoom(num_agents=1, width=3, height=3, with_walls=False)
    policy_env_info = PolicyEnvInterface.from_mg_cfg(cfg)
    policy = TutorialOverlayPolicy(policy_env_info, tutorial=tutorial)
    agent = policy.agent_policy(0)
    obs = AgentObservation(agent_id=0, tokens=[])

    agent.step(obs)
    return agent.infos


def test_tutorial_overlay_policy_play_emits_mission_phases() -> None:
    infos = _step_tutorial_policy("play")
    phases = cast(list[str], infos["tutorial_overlay_phases"])

    assert phases[0].startswith("Welcome to CogsGuard")
    assert phases[1].startswith("Camera and selection")
    assert "WASD" in phases[2]
    assert len(phases) == 7
    assert "tutorial_overlay" not in infos


def test_tutorial_overlay_policy_cogsguard_emits_mission_phases() -> None:
    infos = _step_tutorial_policy("cogsguard")
    phases = cast(list[str], infos["tutorial_overlay_phases"])

    assert phases[0].startswith("CogsGuard:")
    assert "clips" in phases[0].lower()
    assert "tutorial_overlay" not in infos
