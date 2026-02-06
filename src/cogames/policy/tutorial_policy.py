"""CNN + LSTM policy for CoGames tutorials.

Standard CNN + LSTM architecture with:
- Token-to-grid conversion for spatial observations
- 2D CNN encoder
- LSTM for temporal state
- Actor/Critic heads

Usage:
    cogames train -m miner_tutorial -p class=tutorial --device auto --steps 100000
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy, StatefulAgentPolicy, StatefulPolicyImpl
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.policy.token_encoder import coordinates
from mettagrid.policy.utils import LSTMState, LSTMStateDict
from mettagrid.simulator import Action, AgentObservation


class TutorialPolicyNet(nn.Module):
    """Standard CNN + LSTM network."""

    _feature_scale: torch.Tensor

    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__()

        self.hidden_size = 512
        self._obs_height = policy_env_info.obs_height
        self._obs_width = policy_env_info.obs_width
        self._num_features = max((int(f.id) for f in policy_env_info.obs_features), default=0) + 1

        # Feature normalization
        feature_norms = {f.id: f.normalization for f in policy_env_info.obs_features}
        max_id = max((int(fid) for fid in feature_norms.keys()), default=-1)
        feature_scale = torch.ones(max(256, max_id + 1), dtype=torch.float32)
        for fid, norm in feature_norms.items():
            feature_scale[fid] = max(float(norm), 1.0)
        self.register_buffer("_feature_scale", feature_scale)

        # CNN encoder (standard 3x3 convs with stride 2)
        self._cnn = nn.Sequential(
            nn.Conv2d(self._num_features, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, self._num_features, self._obs_height, self._obs_width)
            cnn_out_size = self._cnn(dummy).shape[1]

        self._cnn_fc = nn.Linear(cnn_out_size, 256)
        self._self_encoder = nn.Linear(self._num_features, 256)
        self._rnn = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=1, batch_first=True)

        # Output heads
        num_actions = len(policy_env_info.action_names)
        self._action_head = nn.Linear(self.hidden_size, num_actions)
        self._value_head = nn.Linear(self.hidden_size, 1)

    def _tokens_to_grid(self, observations: torch.Tensor) -> torch.Tensor:
        """Convert token observations [B, T, 3] to grid [B, C, H, W]."""
        batch_size = observations.shape[0]
        device = observations.device

        x_coords, y_coords = coordinates(observations, torch.long)
        feature_ids = observations[..., 1].to(torch.long)
        values = observations[..., 2].to(torch.float32)

        valid_mask = observations[..., 0] != 0xFF
        x_coords = torch.clamp(x_coords, 0, self._obs_width - 1)
        y_coords = torch.clamp(y_coords, 0, self._obs_height - 1)
        feature_ids_clamped = torch.clamp(feature_ids, 0, self._num_features - 1)

        scale = self._feature_scale[torch.clamp(feature_ids, 0, self._feature_scale.shape[0] - 1)]
        values = (values / (scale + 1e-6)) * valid_mask.float()

        grid = torch.zeros(batch_size, self._num_features, self._obs_height, self._obs_width, device=device)
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(x_coords)
        linear_idx = (
            batch_idx * (self._num_features * self._obs_height * self._obs_width)
            + feature_ids_clamped * (self._obs_height * self._obs_width)
            + y_coords * self._obs_width
            + x_coords
        )
        grid.view(-1).scatter_add_(0, linear_idx.view(-1), values.view(-1))
        return grid

    def forward(
        self,
        observations: torch.Tensor,
        state: Optional[Union[LSTMState, tuple[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: observations [B, T, 3] or [B, S, T, 3] -> (logits, values)."""
        orig_shape = observations.shape

        if observations.dim() == 4:
            segments, bptt_horizon = orig_shape[0], orig_shape[1]
            observations = observations.reshape(segments * bptt_horizon, *orig_shape[2:])
        else:
            segments, bptt_horizon = orig_shape[0], 1

        # CNN encoder
        grid = self._tokens_to_grid(observations)
        cnn_out = torch.relu(self._cnn_fc(self._cnn(grid)))

        # Center features (agent's own state)
        center = grid[:, :, self._obs_height // 2, self._obs_width // 2]
        self_out = torch.relu(self._self_encoder(center))

        hidden = torch.cat([cnn_out, self_out], dim=-1)
        hidden = rearrange(hidden, "(b t) h -> b t h", t=bptt_horizon, b=segments)

        # LSTM
        rnn_state = self._parse_state(state)
        hidden, new_state = self._rnn(hidden, rnn_state)

        if isinstance(state, dict) and "lstm_h" in state:
            h, c = new_state
            state["lstm_h"] = h.transpose(0, 1) if h.dim() == 3 else h
            state["lstm_c"] = c.transpose(0, 1) if c.dim() == 3 else c

        hidden = rearrange(hidden, "b t h -> (b t) h")
        return self._action_head(hidden), self._value_head(hidden)

    def forward_eval(
        self,
        observations: torch.Tensor,
        state: Optional[Union[LSTMState, tuple[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for evaluation (same as forward)."""
        return self.forward(observations, state)

    def _parse_state(
        self,
        state: Optional[Union[LSTMState, tuple[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]],
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Parse LSTM state from various formats."""
        if state is None:
            return None

        if isinstance(state, dict):
            h, c = state.get("lstm_h"), state.get("lstm_c")
            if h is None or c is None:
                return None
        elif isinstance(state, LSTMState):
            h, c = state.to_tuple()
        else:
            h, c = state

        if h.dim() == 3:
            h, c = h.transpose(0, 1), c.transpose(0, 1)
        elif h.dim() == 2:
            h, c = h.unsqueeze(0), c.unsqueeze(0)
        elif h.dim() == 1:
            h, c = h.unsqueeze(0).unsqueeze(0), c.unsqueeze(0).unsqueeze(0)
        return (h, c)


class TutorialAgentPolicy(StatefulPolicyImpl[LSTMState]):
    """Per-agent policy for inference."""

    def __init__(self, net: TutorialPolicyNet, device: torch.device, policy_env_info: PolicyEnvInterface):
        self._net = net
        self._device = device
        self._policy_env_info = policy_env_info

    def initial_agent_state(self) -> LSTMState:
        h = torch.zeros((1, self._net.hidden_size), device=self._device)
        c = torch.zeros((1, self._net.hidden_size), device=self._device)
        return LSTMState(hidden=h, cell=c)

    def step_with_state(self, obs: AgentObservation, state: LSTMState) -> tuple[Action, LSTMState]:
        obs_tensor = self._obs_to_tensor(obs)
        state_dict: LSTMStateDict = {"lstm_h": state.hidden, "lstm_c": state.cell}

        with torch.no_grad():
            logits, _ = self._net(obs_tensor, state_dict)

        new_state = LSTMState(hidden=state_dict["lstm_h"].detach(), cell=state_dict["lstm_c"].detach())
        action_idx = int(torch.distributions.Categorical(logits=logits).sample().item())
        return Action(name=self._policy_env_info.action_names[action_idx]), new_state

    def _obs_to_tensor(self, obs: AgentObservation) -> torch.Tensor:
        num_tokens, token_dim = self._policy_env_info.observation_space.shape
        obs_array = np.full((num_tokens, token_dim), 255, dtype=np.uint8)
        for i, token in enumerate(obs.tokens[:num_tokens]):
            obs_array[i, : len(token.raw_token)] = token.raw_token
        return torch.from_numpy(obs_array).unsqueeze(0).to(self._device)


class TutorialPolicy(MultiAgentPolicy):
    """Standard CNN + LSTM policy."""

    short_names = ["tutorial"]

    def __init__(self, policy_env_info: PolicyEnvInterface, device: str = "cpu", **kwargs):
        super().__init__(policy_env_info, device=device, **kwargs)
        self._device = torch.device(device)
        self._policy_env_info = policy_env_info
        self._net = TutorialPolicyNet(policy_env_info).to(self._device)
        self._agent_policy_impl = TutorialAgentPolicy(self._net, self._device, policy_env_info)

    def network(self) -> nn.Module:
        return self._net

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return StatefulAgentPolicy(self._agent_policy_impl, self._policy_env_info, agent_id=agent_id)

    def is_recurrent(self) -> bool:
        return True

    def load_policy_data(self, path: str) -> None:
        self._net.load_state_dict(torch.load(path, map_location=self._device, weights_only=True))

    def save_policy_data(self, path: str) -> None:
        torch.save(self._net.state_dict(), path)
