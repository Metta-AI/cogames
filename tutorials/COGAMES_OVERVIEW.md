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

# CogsGuard: Game Overview

CogsGuard is a cooperative territory-control game where teams of specialized agents capture
and defend junctions against automated Clips expansion. This notebook covers the game rules,
shows a sample game in MettaScope, and trains a miner policy from scratch.

```python
%pip install mettagrid cogames pufferlib-core --quiet
```

## Game Rules

**Objective**: Hold junctions. Reward per tick = `junctions_held / max_steps`.

### The Map

The arena contains:
- **Hub** — your team's home base, projects friendly territory
- **Junctions** — capturable nodes that project territory in a radius
- **Extractors** — resource nodes where Miners gather raw materials
- **Gear Stations** — swap roles by spending resources
- **Assembler** — craft hearts from deposited resources

### 4 Roles

| Role | Stats | Ability |
|-----------|--------------------------------|------------------------------------------|
| Miner | +40 cargo, 10x extraction | Gathers resources at extractors |
| Aligner | Heart limit 3 | Captures neutral junctions (1 heart per capture) |
| Scrambler | +200 HP | Disrupts enemy junctions (1 heart) |
| Scout | +100 energy, +400 HP | Mobile reconnaissance |

**No single role can succeed alone.**

### Cooperation Loop

1. **Miners** extract resources and deposit them into their team hub inventory
2. The **Assembler** crafts hearts from pooled resources
3. **Aligners** spend hearts to capture neutral junctions → expand territory
4. **Scramblers** spend hearts to neutralize enemy junctions → push the front line
5. **Scouts** explore and apply pressure with their high HP and energy

### Clips (Automated Enemy)

Clips expand territory automatically:
- Neutralize enemy junctions adjacent to Clips territory
- Capture neutral junctions adjacent to Clips territory

This creates constant pressure — your team must expand faster than Clips consume.

### Territory Effects

**Friendly territory** (near aligned junctions or hub): HP and energy fully restored.

**Outside friendly territory**: −1 HP/tick, energy drained.

## Watch a Game

Before diving into training, watch a game in MettaScope to see the mechanics in action.

```python
from IPython.display import HTML, display

iframe_src = "https://metta-ai.github.io/metta/mettascope/mettascope.html"

display(HTML(f'''
<div>
    <iframe src="{iframe_src}" width="100%" height="800"
            style="border: 1px solid #ccc; border-radius: 4px;"></iframe>
</div>
'''))
```

### What You're Seeing

- **Agents** are directional sprites that change appearance based on their equipped gear
  (miner, aligner, scrambler, scout). Health pips appear above each agent, colored by team.
- **Junctions** show working/clipped/depleted states as they're captured or neutralized
- **Extractors** animate through working and depletion stages as miners gather resources
- **Territory** is shown as a colored overlay radiating from controlled junctions and hubs

The full game requires all four roles cooperating. To keep this notebook short, we'll
train just the **miner** role below — see the role-specific tutorials at the end for
the others.

```python
import torch

import pufferlib.vector as pvector
from cogames.cogs_vs_clips.clip_difficulty import EASY
from cogames.cogs_vs_clips.tutorials.miner_tutorial import MinerRewardsVariant
from cogames.cogs_vs_clips.mission import CvCMission
from cogames.cogs_vs_clips.sites import make_cogsguard_machina1_site
from cogames.cogs_vs_clips.cog import CogTeam
from cogames.cogs_vs_clips.variants import NoVibesVariant
from cogames.policy.tutorial_policy import TutorialPolicyNet
from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
from mettagrid.envs.early_reset_handler import EarlyResetHandler
from mettagrid.envs.stats_tracker import StatsTracker
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Simulator
from mettagrid.util.stats_writer import NoopStatsWriter
from pufferlib import pufferl
from pufferlib.pufferlib import set_buffers
```

## 1. Build the environment

We build a miner tutorial mission — 4 agents on a compact Machina1 map with dense reward
shaping for resource extraction and deposits. Clips are disabled (EASY) so the agents can
focus on learning to mine.

```python
SEED = 42
NUM_AGENTS = 4
MAX_STEPS = 1000

mission = CvCMission(
    name="miner_tutorial",
    description="Learn miner role - resource extraction and deposits.",
    site=make_cogsguard_machina1_site(NUM_AGENTS),
    num_cogs=NUM_AGENTS,
    max_steps=MAX_STEPS,
    teams={"cogs": CogTeam(name="cogs", num_agents=NUM_AGENTS)},
    variants=[EASY, NoVibesVariant(), MinerRewardsVariant()],
)

env_cfg = mission.make_env()


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


policy_env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)
```

## 2. Policy network

`TutorialPolicyNet` is a CNN + LSTM actor-critic that converts sparse token observations
into a spatial grid, encodes with convolutions, and outputs action logits + value estimates.
See `TRAIN_MINER.ipynb` for a layer-by-layer walkthrough, or
`cogames/policy/tutorial_policy.py` for the source.

```python
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

net = TutorialPolicyNet(policy_env_info).to(DEVICE)
print(f"Device: {DEVICE}, Parameters: {sum(p.numel() for p in net.parameters()):,}")
```

## 3. Train

4 vectorized environments, 1M timesteps (~2-5 min on CPU). The `MinerRewardsVariant` adds
dense reward shaping for gear pickup, resource extraction, and deposits — so agents learn
meaningful behavior even at this short training scale.

```python
from IPython.display import clear_output

NUM_ENVS = 4
TOTAL_TIMESTEPS = 1_000_000
BPTT_HORIZON = 64

vecenv = pvector.make(make_env, num_envs=NUM_ENVS, num_workers=1, batch_size=NUM_ENVS, backend=pvector.Serial)
total_agents = vecenv.num_agents
BATCH_SIZE = max(4096, total_agents * BPTT_HORIZON)

trainer = pufferl.PuffeRL(
    dict(
        env="cogames.cogs_vs_clips",
        device=DEVICE.type,
        total_timesteps=max(TOTAL_TIMESTEPS, BATCH_SIZE),
        batch_size=BATCH_SIZE,
        minibatch_size=min(4096, BATCH_SIZE),
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
    ),
    vecenv,
    net,
)

while trainer.global_step < TOTAL_TIMESTEPS:
    trainer.evaluate()
    trainer.train()
    clear_output(wait=True)
    trainer.print_dashboard()
    if any((p.grad is not None and not p.grad.isfinite().all()) or not p.isfinite().all() for p in net.parameters()):
        print(f"Training diverged at step {trainer.global_step}!")
        break

trainer.close()
print(f"Training complete. Steps: {trainer.global_step}, Epochs: {trainer.epoch}")
```

## 4. Evaluate

Run several episodes with the trained policy and measure performance. Episodes may end
early if agents die (the `EarlyResetHandler` terminates when all agents are dead), which
can produce negative rewards on short episodes.

```python
NUM_EVAL_EPISODES = 5

net.eval()
episode_rewards = []

for ep in range(NUM_EVAL_EPISODES):
    eval_env = make_env(seed=1000 + ep)
    obs, _ = eval_env.reset()
    num_agents = eval_env.num_agents
    state = {
        "lstm_h": torch.zeros(num_agents, 1, net.hidden_size, device=DEVICE),
        "lstm_c": torch.zeros(num_agents, 1, net.hidden_size, device=DEVICE),
    }
    ep_reward = 0.0
    num_steps = 0
    for _step in range(MAX_STEPS):
        obs_tensor = torch.from_numpy(obs).to(DEVICE)
        with torch.no_grad():
            logits, _ = net(obs_tensor, state)
        actions = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        obs, rewards, terms, truncs, infos = eval_env.step(actions)
        ep_reward += rewards.sum()
        num_steps += 1
        if terms.all() or truncs.all():
            break
    eval_env.close()
    episode_rewards.append(ep_reward)
    print(f"  Episode {ep + 1}: {num_steps} steps, reward {ep_reward:.3f}")

mean_reward = sum(episode_rewards) / len(episode_rewards)
print(f"\nMean reward over {NUM_EVAL_EPISODES} episodes: {mean_reward:.3f}")
```

<!-- #region -->
## Try It Yourself

**Interactive tutorial** — play the game yourself with a guided walkthrough:
```bash
cogames tutorial play
```

**Watch scripted agents** — see baseline behavior on the Arena map:
```bash
cogames play -m cogsguard_machina_1.basic -p class=baseline
```

**Role-specific training tutorials** — deep-dive into each role:
- `TRAIN_MINER.ipynb` — resource extraction and deposits
- `TRAIN_ALIGNER.ipynb` — junction capture and territory expansion
- `TRAIN_SCRAMBLER.ipynb` — enemy disruption
- `TRAIN_SCOUT.ipynb` — reconnaissance and map control

**Build a custom policy** — scaffold a trainable policy class:
```bash
cogames tutorial make-policy --trainable
```
<!-- #endregion -->

## Upload to the Leaderboard

Save the trained weights and submit to the CoGames tournament.
Run `cogames login` in a terminal first to authenticate.

```python
POLICY_NAME = "my-overview-policy"  # Change this to your desired policy name

save_path = "./train_dir/overview_policy.pt"
torch.save(net.state_dict(), save_path)
print(f"Saved to {save_path}")

!cogames upload -p "class=tutorial,data={save_path}" -n {POLICY_NAME} --skip-validation
```

