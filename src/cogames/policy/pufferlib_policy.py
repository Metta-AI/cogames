"""PufferLib-trained policy shim for CoGames submissions.

This bridges checkpoints produced by PufferLib training (state_dict of
``pufferlib.environments.cogames.torch.Policy``) to the CoGames
`MultiAgentPolicy` interface so they can be used with ``cogames eval`` /
``cogames submit`` without requiring the full PufferLib repo at runtime.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional, Sequence, Union

import torch

import pufferlib.models  # type: ignore[import-untyped]
import pufferlib.pytorch  # type: ignore[import-untyped]
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy, StatefulAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.policy.pufferlib import PufferlibStatefulImpl
from mettagrid.simulator import Action, AgentObservation, Simulation


class PufferlibCogsPolicy(MultiAgentPolicy, AgentPolicy):
    """Loads and runs checkpoints trained with PufferLib's CoGames policy.

    This policy serves as both the MultiAgentPolicy factory and AgentPolicy
    implementation, returning per-agent wrappers that track state.
    """

    short_names = ["pufferlib_cogs"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        *,
        hidden_size: int = 256,
        device: str = "cpu",
    ):
        MultiAgentPolicy.__init__(self, policy_env_info, device=device)
        AgentPolicy.__init__(self, policy_env_info)
        self._hidden_size = hidden_size
        self._device = torch.device(device)
        self._shim_env = SimpleNamespace(
            single_observation_space=policy_env_info.observation_space,
            single_action_space=policy_env_info.action_space,
            observation_space=policy_env_info.observation_space,
            action_space=policy_env_info.action_space,
            num_agents=policy_env_info.num_agents,
        )
        self._shim_env.env = self._shim_env
        self._net = pufferlib.models.Default(self._shim_env, hidden_size=hidden_size).to(self._device)  # type: ignore[arg-type]
        self._action_names = policy_env_info.action_names
        self._is_recurrent = False
        self._stateful_impl = PufferlibStatefulImpl(
            self._net,
            policy_env_info,
            self._device,
            is_recurrent=self._is_recurrent,
        )
        self._agent_policies: dict[int, StatefulAgentPolicy[dict[str, torch.Tensor | None]]] = {}
        self._state_initialized = False
        self._state: dict[str, torch.Tensor | None] = {}

    def network(self) -> torch.nn.Module:  # type: ignore[override]
        return self._net

    def agent_policy(self, agent_id: int) -> AgentPolicy:  # type: ignore[override]
        if agent_id not in self._agent_policies:
            self._agent_policies[agent_id] = StatefulAgentPolicy(
                self._stateful_impl,
                self._policy_env_info,
                agent_id=agent_id,
            )
        return self._agent_policies[agent_id]

    def is_recurrent(self) -> bool:
        return self._is_recurrent

    def reset(self, simulation: Optional[Simulation] = None) -> None:  # type: ignore[override]
        for policy in self._agent_policies.values():
            policy.reset(simulation)
        self._reset_state()

    def load_policy_data(self, policy_data_path: str) -> None:
        state = torch.load(policy_data_path, map_location=self._device)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        uses_rnn = any(key.startswith(("lstm.", "cell.")) for key in state)
        base_net = pufferlib.models.Default(self._shim_env, hidden_size=self._hidden_size)  # type: ignore[arg-type]
        net = (
            pufferlib.models.LSTMWrapper(
                self._shim_env,
                base_net,
                input_size=base_net.hidden_size,
                hidden_size=base_net.hidden_size,
            )
            if uses_rnn
            else base_net
        )
        net.load_state_dict(state)
        self._net = net.to(self._device)
        self._is_recurrent = uses_rnn
        self._stateful_impl = PufferlibStatefulImpl(
            self._net,
            self._policy_env_info,
            self._device,
            is_recurrent=self._is_recurrent,
        )
        self._agent_policies.clear()
        self._state_initialized = False
        self._state = {}

    def save_policy_data(self, policy_data_path: str) -> None:
        torch.save(self._net.state_dict(), policy_data_path)

    def step(self, obs: Union[AgentObservation, torch.Tensor, Sequence[Any]]) -> Action:  # type: ignore[override]
        if isinstance(obs, AgentObservation):
            if not self._state_initialized:
                self._reset_state()
            with torch.no_grad():
                action, self._state = self._stateful_impl.step_with_state(obs, self._state)
            return action
        obs_tensor = torch.as_tensor(obs, device=self._device, dtype=torch.float32)
        if obs_tensor.ndim == 2:
            obs_tensor = obs_tensor.unsqueeze(0)
        with torch.no_grad():
            self._net.eval()
            logits, _ = self._net.forward_eval(obs_tensor, None)
            sampled, _, _ = pufferlib.pytorch.sample_logits(logits)
        action_idx = max(0, min(int(sampled.item()), len(self._action_names) - 1))
        return Action(name=self._action_names[action_idx])

    def _reset_state(self) -> None:
        self._stateful_impl.reset()
        self._state = self._stateful_impl.initial_agent_state()
        self._state_initialized = True
