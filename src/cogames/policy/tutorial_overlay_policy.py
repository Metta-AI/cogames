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
        "Mission briefing: CogsGuard training overview\n"
        "Welcome. This simulation mirrors core CogsGuard operations. "
        "We will launch the Mettascope visual interface now."
    ),
    (
        "Step 1 - Interface and controls\n"
        "Left Pane (Intel): Shows details for selected objects (Stations, Tiles, Cogs).\n"
        "Right Pane (Vibe Deck): Select icons here to change your Cog's broadcast resonance.\n"
        "Zoom/Pan: Scroll or pinch to zoom the arena; drag to pan.\n"
        "Click various buildings to view details in the Left Pane.\n"
        "Look for the Hub (Hub), Junctions, Gear Stations, and Extractors.\n"
        "Click your Cog to assume control."
    ),
    (
        "Step 2 - Movement and energy\n"
        "Use WASD or Arrow Keys to move your Cog.\n"
        "Every move costs Energy, and aligned hubs/junctions recharge you.\n"
        "Watch your battery bar on the Cog or in the HUD.\n"
        "If low, rest (skip turn), lean against a wall (walk into it), or\n"
        "stand near the Hub or an aligned Junction."
    ),
    (
        "Step 3 - Gear up\n"
        "Primary interaction mode is walking into things.\n"
        "Locate a Gear Station and walk into it to equip a role:\n"
        "Miner, Scout, Aligner, or Scrambler.\n"
        "Gear costs are paid from the team commons."
    ),
    (
        "Step 4 - Resources and hearts\n"
        "Find an Extractor station to gather elements:\n"
        "C (Carbon), O (Oxygen), G (Germanium), S (Silicon).\n"
        "Visit the Chest to assemble or withdraw Hearts from the commons."
    ),
    (
        "Step 5 - Junction control\n"
        "Junctions can be aligned to your team.\n"
        "As an Aligner: get Influence (stand near the Hub) and a Heart, then bump a neutral junction.\n"
        "As a Scrambler: get a Heart, then bump an enemy-aligned junction to neutralize it.\n"
        "Aligned junctions recharge energy for your team."
    ),
    (
        "Step 6 - Objective complete\n"
        "Congratulations. You have completed the tutorial.\n"
        "You've mastered movement, gear, resources, and junction control.\n"
        "You're now ready to tackle the full CogsGuard arena."
    ),
)


_COGSGUARD_PHASES: tuple[str, ...] = (
    (
        "Mission briefing: CogsGuard stabilization sim\n"
        "Objective: outscore Clips by sustaining junction control under constant pressure."
    ),
    ("Phase 1: Open safely\nExpand from the hub to two nearby lanes and secure your first neutral junction."),
    (
        "Phase 2: Manage Clips pressure\n"
        "Clips will scramble nearby junctions in waves. Retake quickly to prevent score drift."
    ),
    (
        "Phase 3: Use territory effects\n"
        "Friendly territory restores HP, energy, and influence; enemy territory drains HP and influence."
    ),
    (
        "Phase 4: Coordinate roles\n"
        "Keep miners feeding the hub while aligners and scramblers rotate across contested lanes."
    ),
    (
        "Phase 5: Maintain heart economy\n"
        "Convert resources into hearts continuously so capture and scramble actions never stall."
    ),
    ("Phase 6: Win the control cycle\nRun the loop every minute: scout -> gather -> craft -> capture -> defend."),
    ("Mission complete\nYou are ready for full CogsGuard missions."),
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
