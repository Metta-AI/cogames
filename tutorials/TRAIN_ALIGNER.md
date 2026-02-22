---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# CoGames Puffer Training - Aligner Tutorial

This notebook replicates the training setup from:
```
cogames train -m aligner_tutorial -p tutorial
```

It walks through:
1. Building the environment from the aligner_tutorial mission
2. Token-to-grid observation preprocessing
3. Defining a CNN + LSTM policy network from scratch
4. Vectorizing with PufferLib
5. Running the PuffeRL training loop
6. Uploading to the CoGames leaderboard

**Aligner role**: Collect hearts and align neutral junctions to team control.
Aligners have increased influence limit (+20) and heart limit (3), and start
with 120 hearts per team to enable immediate junction alignment practice.

```python
%pip install mettagrid cogames pufferlib-core --quiet
```

```python
import torch
import torch.nn as nn
from einops import rearrange
import pufferlib.vector as pvector
from pufferlib import pufferl
from pufferlib.pufferlib import set_buffers

from mettagrid import MettaGridConfig
from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
from mettagrid.envs.early_reset_handler import EarlyResetHandler
from mettagrid.envs.stats_tracker import StatsTracker
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Simulator
from mettagrid.simulator.replay_log_writer import InMemoryReplayWriter
from mettagrid.util.stats_writer import NoopStatsWriter

from cogames.cogs_vs_clips.tutorials.aligner_tutorial import AlignerRewardsVariant
from cogames.cogs_vs_clips.clip_difficulty import EASY
from cogames.cogs_vs_clips.cog import CogConfig
from cogames.cogs_vs_clips.mission import CvCMission
from cogames.cogs_vs_clips.sites import COGSGUARD_ARENA
from cogames.cogs_vs_clips.team import CogTeam
from cogames.cogs_vs_clips.variants import BraveheartVariant
```

## 1. Build the mission and environment config

- **Site**: CogsGuard Arena (50x50 compact training map with gear stations in hub)
- **EASY difficulty**: No clips pressure
- **heart_limit=3**: Aligners can hold 3 hearts at once
- **initial_hearts=120**: Team starts with 120 hearts for immediate practice
- **1000 max steps** per episode

```python
NUM_AGENTS = 4
MAX_STEPS = 1000

mission = CvCMission(
    name="aligner_tutorial",
    description="Learn aligner role - collect hearts, align neutral junctions.",
    site=COGSGUARD_ARENA,
    num_cogs=NUM_AGENTS,
    max_steps=MAX_STEPS,
    cog=CogConfig(heart_limit=3),
    teams={"cogs": CogTeam(name="cogs", num_agents=NUM_AGENTS, wealth=3, initial_hearts=120)},
    variants=[EASY, AlignerRewardsVariant(), BraveheartVariant()],
)

env_cfg: MettaGridConfig = mission.make_env()

print(f"Map builder: {type(env_cfg.game.map_builder).__name__}")
print(f"Max steps: {env_cfg.game.max_steps}")
print(f"Num agents: {env_cfg.game.num_agents}")
print(f"Events: {list(env_cfg.game.events.keys())}")
print(f"Collectives: {list(env_cfg.game.collectives.keys())}")
```

## 2. Create a single environment

MettaGridPufferEnv wraps the C++ simulator with PufferLib's PufferEnv interface.

```python
SEED = 42

def make_env(buf=None, seed=None):
    """Environment factory for PufferLib vectorization."""
    cfg = env_cfg.model_copy(deep=True)

    map_builder = cfg.game.map_builder
    if isinstance(map_builder, MapGen.Config) and seed is not None:
        map_builder.seed = SEED + seed

    simulator = Simulator()
    simulator.add_event_handler(StatsTracker(NoopStatsWriter()))
    simulator.add_event_handler(EarlyResetHandler())
    env = MettaGridPufferEnv(simulator, cfg, buf=buf, seed=seed or 0)
    set_buffers(env, buf)
    return env


driver_env = make_env(seed=0)
policy_env_info = PolicyEnvInterface.from_mg_cfg(driver_env.env_cfg)

print(f"Observation space: {driver_env.single_observation_space}")
print(f"Action space: {driver_env.single_action_space}")
print(f"Num agents: {driver_env.num_agents}")
print(f"Action names: {policy_env_info.action_names}")
print(f"Obs features: {len(policy_env_info.obs_features)} features")
print(f"Obs grid: {policy_env_info.obs_height}x{policy_env_info.obs_width}")
driver_env.close()
```

## 3. Observation preprocessing

MettaGrid observations are sparse tokens `[B, T, 3]` where each token is `[packed_xy, feature_id, value]`.
The packed byte encodes grid coordinates as nibbles: `y = byte >> 4, x = byte & 0x0F`.

We scatter these into a dense spatial grid `[B, C, H, W]` so a CNN can process them.

```python
def tokens_to_grid(
    observations: torch.Tensor,
    obs_height: int,
    obs_width: int,
    num_features: int,
    feature_scale: torch.Tensor,
) -> torch.Tensor:
    """Convert sparse token observations [B, T, 3] into a dense grid [B, C, H, W]."""
    batch_size = observations.shape[0]
    device = observations.device

    coords_byte = observations[..., 0].to(torch.long)
    x_coords = coords_byte & 0x0F
    y_coords = (coords_byte >> 4) & 0x0F
    feature_ids = observations[..., 1].to(torch.long)
    values = observations[..., 2].to(torch.float32)

    valid_mask = (observations[..., 0] != 0xFF).float()
    x_coords = torch.clamp(x_coords, 0, obs_width - 1)
    y_coords = torch.clamp(y_coords, 0, obs_height - 1)
    feature_ids_clamped = torch.clamp(feature_ids, 0, num_features - 1)

    scale = feature_scale[torch.clamp(feature_ids, 0, feature_scale.shape[0] - 1)]
    values = (values / (scale + 1e-6)) * valid_mask

    grid = torch.zeros(batch_size, num_features, obs_height, obs_width, device=device)
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(x_coords)
    linear_idx = (
        batch_idx * (num_features * obs_height * obs_width)
        + feature_ids_clamped * (obs_height * obs_width)
        + y_coords * obs_width
        + x_coords
    )
    grid.view(-1).scatter_add_(0, linear_idx.view(-1), values.view(-1))
    return grid
```

## 4. Build the neural network

CNN + LSTM actor-critic:
- **CNN encoder**: Two 3x3 conv layers (64 → 128) with stride 2, projected to 256-dim
- **Self encoder**: Linear on the center cell (agent's own state) → 256-dim
- **LSTM**: 512 hidden units, 1 layer
- **Action head**: Linear → num_actions logits
- **Value head**: Linear → scalar value

```python
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")


class AlignerPolicyNet(nn.Module):
    """CNN + LSTM actor-critic."""

    _feature_scale: torch.Tensor

    def __init__(self, env_info: PolicyEnvInterface):
        super().__init__()

        self.hidden_size = 512
        self._obs_height = env_info.obs_height
        self._obs_width = env_info.obs_width
        self._num_features = max((int(f.id) for f in env_info.obs_features), default=0) + 1

        feature_norms = {f.id: f.normalization for f in env_info.obs_features}
        max_id = max((int(fid) for fid in feature_norms.keys()), default=-1)
        feature_scale = torch.ones(max(256, max_id + 1), dtype=torch.float32)
        for fid, norm in feature_norms.items():
            feature_scale[fid] = max(float(norm), 1.0)
        self.register_buffer("_feature_scale", feature_scale)

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

        num_actions = len(env_info.action_names)
        self._action_head = nn.Linear(self.hidden_size, num_actions)
        self._value_head = nn.Linear(self.hidden_size, 1)

    def forward(self, observations: torch.Tensor, state: dict[str, torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = observations.shape
        if observations.dim() == 4:
            segments, bptt_horizon = orig_shape[0], orig_shape[1]
            observations = observations.reshape(segments * bptt_horizon, *orig_shape[2:])
        else:
            segments, bptt_horizon = orig_shape[0], 1

        grid = tokens_to_grid(observations, self._obs_height, self._obs_width, self._num_features, self._feature_scale)
        cnn_out = torch.relu(self._cnn_fc(self._cnn(grid)))
        center = grid[:, :, self._obs_height // 2, self._obs_width // 2]
        self_out = torch.relu(self._self_encoder(center))

        hidden = torch.cat([cnn_out, self_out], dim=-1)
        hidden = rearrange(hidden, "(b t) h -> b t h", t=bptt_horizon, b=segments)

        rnn_state = None
        if state is not None:
            h, c = state.get("lstm_h"), state.get("lstm_c")
            if h is not None and c is not None:
                h = h.transpose(0, 1) if h.dim() == 3 else h.unsqueeze(0)
                c = c.transpose(0, 1) if c.dim() == 3 else c.unsqueeze(0)
                rnn_state = (h, c)

        hidden, (h_out, c_out) = self._rnn(hidden, rnn_state)

        if state is not None and "lstm_h" in state:
            state["lstm_h"] = h_out.transpose(0, 1)
            state["lstm_c"] = c_out.transpose(0, 1)

        hidden = rearrange(hidden, "b t h -> (b t) h")
        return self._action_head(hidden), self._value_head(hidden)

    forward_eval = forward


net = AlignerPolicyNet(policy_env_info).to(DEVICE)

print(f"\nArchitecture:\n{net}")
total_params = sum(p.numel() for p in net.parameters())
print(f"\nTotal parameters: {total_params:,}")
```

## 5. Vectorize environments with PufferLib

```python
NUM_ENVS = 4

# Serial backend — spawn multiprocessing can't find notebook-defined functions
vecenv = pvector.make(
    make_env,
    num_envs=NUM_ENVS,
    num_workers=1,
    batch_size=NUM_ENVS,
    backend=pvector.Serial,
)

total_agents = vecenv.num_agents
print(f"Vectorized envs: {NUM_ENVS}")
print(f"Total agents across all envs: {total_agents}")
print(f"Agents per env: {total_agents // NUM_ENVS}")
```

## 6. Configure and run PuffeRL training

```python
TOTAL_TIMESTEPS = 10_000_000
BPTT_HORIZON = 64
BATCH_SIZE = max(4096, total_agents * BPTT_HORIZON)
MINIBATCH_SIZE = min(4096, BATCH_SIZE)

train_config = dict(
    env="cogames.cogs_vs_clips",
    device=DEVICE.type,
    total_timesteps=max(TOTAL_TIMESTEPS, BATCH_SIZE),
    batch_size=BATCH_SIZE,
    minibatch_size=MINIBATCH_SIZE,
    bptt_horizon=BPTT_HORIZON,
    seed=SEED,
    use_rnn=True,
    torch_deterministic=True,
    cpu_offload=False,
    compile=False,
    optimizer="adam",
    learning_rate=0.00092,
    anneal_lr=True,
    min_lr_ratio=0.0,
    adam_beta1=0.95,
    adam_beta2=0.999,
    adam_eps=1e-8,
    precision="float32",
    gamma=0.995,
    gae_lambda=0.90,
    update_epochs=1,
    clip_coef=0.2,
    vf_coef=2.0,
    vf_clip_coef=0.2,
    max_grad_norm=1.5,
    ent_coef=0.01,
    vtrace_rho_clip=1.0,
    vtrace_c_clip=1.0,
    prio_alpha=0.8,
    prio_beta0=0.2,
    data_dir="./train_dir",
    checkpoint_interval=50,
    max_minibatch_size=32768,
)

print("Training config:")
for k, v in train_config.items():
    print(f"  {k}: {v}")
```

```python
trainer = pufferl.PuffeRL(train_config, vecenv, net)
print(f"Model size: {trainer.model_size:,} params")
print(f"Batch size: {trainer.config['batch_size']}")
print(f"Total epochs: {trainer.total_epochs}")
```

```python
from IPython.display import clear_output

while trainer.global_step < train_config["total_timesteps"]:
    trainer.evaluate()
    trainer.train()

    clear_output(wait=True)
    trainer.print_dashboard()

    has_nan = any(
        (p.grad is not None and not p.grad.isfinite().all()) or not p.isfinite().all()
        for p in net.parameters()
    )
    if has_nan:
        print(f"Training diverged at step {trainer.global_step}!")
        break

trainer.close()
print(f"Training complete. Steps: {trainer.global_step}, Epochs: {trainer.epoch}")
```

```python
save_path = "./train_dir/tutorial_aligner.pt"
torch.save(net.state_dict(), save_path)
print(f"Saved to {save_path}")
```

## 7. Watch the trained policy in MettaScope

Run the trained network for a full episode and view the replay in the interactive MettaScope viewer.

```python
import base64
import json
import zlib
from IPython.display import HTML, Javascript, display

# Run an episode and capture replay
replay_writer = InMemoryReplayWriter()
render_env = make_env(seed=99)
render_env._simulator.add_event_handler(replay_writer)
obs, _ = render_env.reset()

num_agents = render_env.num_agents
state = {
    "lstm_h": torch.zeros(num_agents, 1, net.hidden_size, device=DEVICE),
    "lstm_c": torch.zeros(num_agents, 1, net.hidden_size, device=DEVICE),
}

net.eval()
num_steps = 0
for _step in range(MAX_STEPS):
    obs_tensor = torch.from_numpy(obs).to(DEVICE)
    with torch.no_grad():
        logits, _ = net(obs_tensor, state)
    actions = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
    obs, rewards, terms, truncs, infos = render_env.step(actions)
    num_steps += 1
    if terms.all() or truncs.all():
        break

print(f"Episode finished after {num_steps} steps")

# Get replay data before closing (needs live simulation for episode stats)
replays = replay_writer.get_completed_replays()
replay_data = json.dumps(replays[0].get_replay_data())
render_env.close()

# Compress and encode
compressed = zlib.compress(replay_data.encode("utf-8"))
b64 = base64.b64encode(compressed).decode("utf-8")

# Display MettaScope iframe
iframe_src = "https://metta-ai.github.io/metta/mettascope/mettascope.html"
iframe_id = "mettascope_iframe"

display(HTML(f'''
<div>
    <iframe id="{iframe_id}" src="{iframe_src}" width="100%" height="800"
            style="border: 1px solid #ccc; border-radius: 4px;"></iframe>
</div>
'''))

b64_escaped = b64.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n')
display(Javascript(f'''
(function() {{
    const iframe = document.getElementById('{iframe_id}');
    const base64Data = '{b64_escaped}';
    function sendReplayData() {{
        iframe.contentWindow.postMessage({{
            type: 'replayData',
            base64: base64Data,
            fileName: 'tutorial_replay.json.z'
        }}, '*');
    }}
    window.addEventListener('message', function(event) {{
        if (event.data.type === 'mettascopeReady') {{
            sendReplayData();
        }}
    }});
}})();
'''))
```

## 8. Upload to the leaderboard

Submit the trained weights to the CoGames tournament. This uses the `tutorial` policy class
(same architecture as `AlignerPolicyNet`) with our saved weights.

Prerequisites: run `cogames login` in a terminal first to authenticate.

```python
POLICY_NAME = "my-aligner"  # Change this to your desired policy name

!cogames upload -p "class=tutorial,data={save_path}" -n {POLICY_NAME} --skip-validation
```
