from __future__ import annotations

from cogames.policy.aligner_agent import AlignerPolicyImpl, AlignerState
from cogames.policy.starter_agent import StarterCogPolicyImpl, StarterCogState
from mettagrid.policy.policy import MultiAgentPolicy, StatefulAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


class MachinaRolesPolicy(MultiAgentPolicy):
    short_names = ["machina_roles", "machina_mixed"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        device: str = "cpu",
        num_aligners: int = 2,
        aligner_ids: str = "0,5",
    ):
        super().__init__(policy_env_info, device=device)
        parsed_aligner_ids = tuple(
            int(part.strip()) for part in aligner_ids.split(",") if part.strip()
        )
        if parsed_aligner_ids:
            self._aligner_ids = frozenset(parsed_aligner_ids)
        else:
            self._aligner_ids = frozenset(range(num_aligners))
        self._agent_policies: dict[int, StatefulAgentPolicy[AlignerState | StarterCogState]] = {}

    def agent_policy(self, agent_id: int) -> StatefulAgentPolicy[AlignerState | StarterCogState]:
        if agent_id not in self._agent_policies:
            if agent_id in self._aligner_ids:
                impl = AlignerPolicyImpl(self._policy_env_info, agent_id)
            else:
                impl = StarterCogPolicyImpl(self._policy_env_info, agent_id, preferred_gear="miner")
            self._agent_policies[agent_id] = StatefulAgentPolicy(
                impl,
                self._policy_env_info,
                agent_id=agent_id,
            )
        return self._agent_policies[agent_id]
