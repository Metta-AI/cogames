from __future__ import annotations

from mettagrid.policy.policy import MultiAgentPolicy, StatefulAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface

from cogames.policy.llm_skills import MinerSkillImpl, MinerSkillState


class MineClosestPolicy(MultiAgentPolicy):
    """Legacy scripted miner wrapper over the shared LLM-miner skill implementation."""

    short_names = ["mine_closest"]

    def __init__(self, policy_env_info: PolicyEnvInterface, device: str = "cpu", return_load: int | str = 40):
        super().__init__(policy_env_info, device=device)
        self._return_load = int(return_load)
        self._agent_policies: dict[int, StatefulAgentPolicy[MinerSkillState]] = {}

    def agent_policy(self, agent_id: int) -> StatefulAgentPolicy[MinerSkillState]:
        if agent_id not in self._agent_policies:
            self._agent_policies[agent_id] = StatefulAgentPolicy(
                MinerSkillImpl(self._policy_env_info, agent_id, self._return_load),
                self._policy_env_info,
                agent_id=agent_id,
            )
        return self._agent_policies[agent_id]
