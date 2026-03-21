"""No-op tutorial policy that emits in-game tutorial phases."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from mettagrid.mettagrid_c import dtype_actions
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation

_PLAY_PHASES: tuple[str, ...] = (
    (
        "Welcome to CogsGuard\n"
        "Your goal: control junctions to outscore the opposing Clips team.\n"
        "Explore the arena using the controls in the next step."
    ),
    (
        "Camera and selection\n"
        "Scroll or pinch to zoom. Drag to pan.\n"
        "Click buildings to inspect them in the left pane.\n"
        "Click your Cog to take control of it."
    ),
    (
        "Movement and energy\n"
        "Use WASD or Arrow Keys to move.\n"
        "Movement costs energy. Your battery bar shows how much you have.\n"
        "Stand near the Hub or a friendly junction to recharge.\n"
        "You lose health when away from the Hub and your junctions, so don't stray too far."
    ),
    (
        "Gear up\n"
        "Walk into a Gear Station to pick a role.\n"
        "From left to right:\n"
        "  Aligner - captures neutral junctions for your team\n"
        "  Scrambler - neutralizes enemy junctions\n"
        "  Miner - gathers resources\n"
        "  Scout - explores fast\n"
        "Try equipping one now."
    ),
    (
        "Resources and hearts\n"
        "Walk into an Extractor to gather resources (Carbon, Oxygen, Germanium, Silicon).\n"
        "Visit the Assembler to craft Hearts from resources.\n"
        "Hearts are needed to capture and scramble junctions."
    ),
    (
        "Junction control\n"
        "Junctions are the key to winning.\n"
        "As an Aligner: get Influence from the Hub, carry a Heart, then walk into a neutral junction.\n"
        "As a Scrambler: carry a Heart, walk into an enemy junction to neutralize it.\n"
        "Friendly junctions recharge your team's energy."
    ),
    (
        "Tutorial complete\n"
        "You know the basics: move, gear up, gather, craft, and control junctions.\n"
        "Close this overlay and try it all together."
    ),
)


_COGSGUARD_PHASES: tuple[str, ...] = (
    (
        "CogsGuard: stabilization sim\n"
        "Outscore Clips by controlling more junctions over time.\n"
        "Each phase introduces a new layer of strategy."
    ),
    (
        "Phase 1: Gear up and get a heart\n"
        "Grab the Aligner gear to capture neutral junctions.\n"
        "Then grab the Scrambler gear to neutralize Clip junctions.\n"
        "Craft a heart at the Assembler - you need one for each capture or scramble."
    ),
    (
        "Phase 2: Expand from the Hub\n"
        "Move out to nearby neutral junctions and capture them.\n"
        "Stay close to the Hub early so you can recharge."
    ),
    (
        "Phase 3: Handle Clips pressure\n"
        "Clips will scramble your junctions in waves.\n"
        "Retake them quickly - every moment without control costs you points."
    ),
    (
        "Phase 4: Use territory\n"
        "The Hub and your junctions form safe territory - they restore energy and health.\n"
        "You lose health when away from them, so expand your junction network to push further."
    ),
    (
        "Phase 5: Coordinate roles\n"
        "Miners feed resources to the Hub.\n"
        "Aligners and Scramblers rotate across contested junctions.\n"
        "Keep all roles active."
    ),
    (
        "Phase 6: Keep crafting hearts\n"
        "Hearts fuel captures and scrambles.\n"
        "Convert resources into hearts continuously so you never stall."
    ),
    (
        "Phase 7: The control loop\n"
        "Repeat: scout the map, gather resources, craft hearts, capture junctions, defend them.\n"
        "Run this loop faster than Clips to win."
    ),
    ("Mission complete\nYou're ready for full CogsGuard missions."),
)


class TutorialOverlayAgentPolicy(AgentPolicy):
    """Per-agent no-op policy that broadcasts tutorial phases to the renderer."""

    def __init__(self, policy_env_info: PolicyEnvInterface, phases: Sequence[str]):
        super().__init__(policy_env_info)
        self._phases = tuple(phases)

    def step(self, obs: AgentObservation) -> Action:
        self._infos = {"tutorial_overlay_phases": list(self._phases)}
        return Action(name="noop")


class TutorialOverlayPolicy(MultiAgentPolicy):
    """Policy for tutorial sessions that keeps agents idle and emits overlay phases."""

    short_names = ["tutorial_noop"]

    def __init__(self, policy_env_info: PolicyEnvInterface, tutorial: str = "play", device: str = "cpu"):
        super().__init__(policy_env_info, device=device)
        self._phases = _COGSGUARD_PHASES if tutorial == "cogsguard" else _PLAY_PHASES
        self._noop_action_value = dtype_actions.type(policy_env_info.action_names.index("noop"))

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return TutorialOverlayAgentPolicy(self._policy_env_info, self._phases)

    def step_batch(self, raw_observations: np.ndarray, raw_actions: np.ndarray) -> None:
        raw_actions[...] = self._noop_action_value
