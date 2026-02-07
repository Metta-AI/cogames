from __future__ import annotations

from cogames.policy.starter_agent import StarterCogPolicyImpl, StarterCogState
from mettagrid.policy.policy import MultiAgentPolicy, StatefulAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


class _StarterRolePolicy(MultiAgentPolicy):
    short_names: list[str]
    _preferred_gear: str

    def __init__(self, policy_env_info: PolicyEnvInterface, device: str = "cpu"):
        super().__init__(policy_env_info, device=device)
        self._agent_policies: dict[int, StatefulAgentPolicy[StarterCogState]] = {}

    def agent_policy(self, agent_id: int) -> StatefulAgentPolicy[StarterCogState]:
        if agent_id not in self._agent_policies:
            self._agent_policies[agent_id] = StatefulAgentPolicy(
                StarterCogPolicyImpl(self._policy_env_info, agent_id, preferred_gear=self._preferred_gear),
                self._policy_env_info,
                agent_id=agent_id,
            )
        return self._agent_policies[agent_id]


class MinerRolePolicy(_StarterRolePolicy):
    short_names = ["role_miner"]
    _preferred_gear = "miner"


class ScoutRolePolicy(_StarterRolePolicy):
    short_names = ["role_scout"]
    _preferred_gear = "scout"


class AlignerRolePolicy(_StarterRolePolicy):
    short_names = ["role_aligner"]
    _preferred_gear = "aligner"


class ScramblerRolePolicy(_StarterRolePolicy):
    short_names = ["role_scrambler"]
    _preferred_gear = "scrambler"
