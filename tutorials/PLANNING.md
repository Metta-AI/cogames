# CoGames + PufferLib "Hello World" Tutorial - Technical Planning Document

**Date:** October 4, 2025  
**Purpose:** Comprehensive technical specification for building a 3-notebook tutorial series introducing newcomers to CoGames and PufferLib for multi-agent reinforcement learning.

---

## Executive Summary

This document provides detailed answers to all codebase-specific questions required to build a tutorial series for CoGames (a MettagGrid-based grid world environment) with PufferLib integration. The tutorial will progressively introduce:

1. **Basic goal achievement** - Agent learns to deposit a heart
2. **Crafting mechanics** - Agent learns to craft hearts from resources
3. **Full cycle & multi-agent** - Complete resource gathering loop with multi-agent scaling

**Key Finding:** The CoGames environment uses **MettagGrid** (Gymnasium-compatible) as its core, not a direct PufferLib Binding. PufferLib is used for vectorized training via `pufferlib.vector` and training via `pufferl.PuffeRL`.

---

## Phase 0: Foundation & Environment Setup

### Action Items

1. **Create Dockerfile** with:
   - `cogames` (from source or PyPI when available)
   - `mettagrid` (workspace dependency in pyproject.toml)
   - `pufferlib-core` (already a dependency)
   - `jupyterlab`
   - `imageio` for GIF creation
   - `torch` (CPU or CUDA depending on target)

### Codebase-Specific Questions & Answers

#### Environment Instantiation

**Q: What is the exact class name for the main grid world environment?**

**A:** `mettagrid.MettaGridEnv` (imported as `from mettagrid import MettaGridEnv`)

**Q: What arguments does its `__init__` method take?**

**A:** The environment takes an `env_cfg` parameter of type `MettaGridConfig`:

```python
from mettagrid import MettaGridEnv, MettaGridConfig

# Load a pre-configured game
from cogames.game import get_game
config = get_game("assembler_1_simple")

# Create environment
env = MettaGridEnv(env_cfg=config)
```

Configuration is a **Pydantic model** (`MettaGridConfig`) that can be:
- Loaded from pre-built scenarios via `cogames.game.get_game(name)`
- Loaded from YAML/JSON files via `cogames.game.load_game_config(path)`
- Created programmatically using `cogames.cogs_vs_clips.scenarios.make_game(...)`

#### PufferLib Integration

**Q: How is the cogworks environment wrapped for pufferlib?**

**A:** There is **no explicit PufferLib Binding subclass**. Instead:

1. **MettagGrid is Gymnasium-compatible** - it implements the standard Gymnasium API (reset, step, observation_space, action_space)
2. **PufferLib uses it directly** via vectorization in `train.py`:

```python
# From src/cogames/train.py, lines 40-102
def env_creator(cfg: MettaGridConfig, buf: Optional[Any] = None, seed: Optional[int] = None):
    env = MettaGridEnv(env_cfg=cfg)
    set_buffers(env, buf)  # PufferLib's buffer management
    return env

vecenv = pufferlib.vector.make(
    env_creator,
    num_envs=256,
    num_workers=8,
    batch_size=128,
    backend=pufferlib.vector.Multiprocessing,
    env_kwargs={"cfg": env_cfg},
)
```

3. **Training uses PuffeRL** (not the old PufferLib API):

```python
trainer = pufferl.PuffeRL(train_args, vecenv, policy_network)
while trainer.global_step < num_steps:
    trainer.evaluate()
    trainer.train()
```

**Q: What specific methods are essential for the tutorial?**

**A:** Standard Gymnasium methods:
- `env.reset(seed=42)` → returns `(obs, info)`
- `env.step(actions)` → returns `(obs, rewards, dones, truncated, info)`
- `env.single_observation_space` → Gymnasium `Box` space (shape: `(7, 7, 3)`, dtype: `uint8`)
- `env.single_action_space` → Gymnasium `MultiDiscrete([6, 4])` (6 move directions, 4 action arguments)
- `env.num_agents` → int
- `env.action_names`, `env.resource_names`, `env.object_type_names` → for display

#### Configuration System

**Q: Where are the default configuration parameters defined?**

**A:** In `src/cogames/cogs_vs_clips/scenarios.py`, function `_base_game_config(num_agents)` (lines 36-87)

**Q: What is the precise key for setting the map size?**

**A:**
```python
config.game.map_builder.width = 15
config.game.map_builder.height = 15
```

The `map_builder` can be:
- `RandomMapBuilder.Config` - for procedurally generated maps
- `AsciiMapBuilder.Config` - for loading from `.map` files

**Q: What is the key for setting the number of agents?**

**A:**
```python
config.game.num_agents = 4
config.game.map_builder.agents = 4  # Also needs to match in map_builder
```

---

## Phase 1: Tutorial Notebook 1 - Basic Goal Achievement

### Objective
Create `01_goal_achievement.ipynb` where an agent starts with a heart and learns to navigate to a chest and deposit it.

### Codebase-Specific Questions & Answers

#### Scenario Configuration

**Q: How do you configure an agent's `initial_inventory`?**

**A:** Via the agent configuration in the game config:

```python
config.game.agent.initial_inventory = {
    "energy": 100,
    "heart": 1,  # Start with 1 heart
}
```

**Format:** Dictionary with `{resource_name: quantity}`

**Key path:** `config.game.agent.initial_inventory`

**Q: How are entities like chests placed?**

**A:** Two methods:

1. **For RandomMapBuilder** (procedural generation):
```python
from mettagrid.map_builder.random import RandomMapBuilder

map_builder = RandomMapBuilder.Config(
    width=10,
    height=10,
    agents=1,
    objects={
        "chest": 1,  # Place 1 chest randomly
        "assembler": 1,
    },
    seed=42,
)
config.game.map_builder = map_builder
```

2. **For AsciiMapBuilder** (from .map file):
```python
# Entities are placed in the ASCII map file using map_char
# Example: 'C' = chest, 'Z' = assembler, '@' = agent spawn
```

#### Reward Function

**Q: Where is the reward for depositing a heart defined?**

**A:** In the agent configuration via `AgentRewards`:

```python
# From scenarios.py, lines 73-78
config.game.agent.rewards = AgentRewards(
    stats={"heart.lost": 1},  # 1 reward per heart lost from inventory (agent-specific)
    # Alternative: {"chest.heart.amount": 1} for team-based chest tracking
)
```

The reward is **sparse** - agents receive +1 reward for each heart that leaves their inventory (deposited).

**Key difference:**
- `heart.lost`: **Agent-specific stat** - each agent gets +1 when THEY deposit a heart
- `chest.heart.amount`: **Global stat** - all agents get +1 when ANY heart is deposited (team reward)

**Location:** `config.game.agent.rewards.stats`

**Reward calculation happens automatically** in MettagGrid's step function when the stats change.

**Q: What is the name of the action for depositing?**

**A:** There is **no explicit "DEPOSIT" action**. Instead:

- Depositing happens via **MOVE** action
- Agent must move **into** the chest from a valid deposit position
- Chests have `deposit_positions` (e.g., `["E"]` = east side of chest)

```python
# Default configuration from stations.py
def chest() -> ChestConfig:
    return ChestConfig(
        name="chest",
        type_id=17,
        map_char="C",
        render_symbol="📦",
        resource_type="heart",
        deposit_positions=["E"],  # Deposit by moving from East
        withdrawal_positions=["W"],  # Withdraw by moving from West
    )

# Recommended configuration for tutorials (easier to learn)
config.game.objects["chest"].deposit_positions = ["N", "S", "E", "W"]  # All sides
config.game.objects["chest"].withdrawal_positions = []  # Disable withdrawals
```

**Rationale for tutorial configuration:**
- Allow deposits from all sides → easier for agents to learn
- Disable withdrawals → task is unidirectional (hearts only go in, never out)

**Actions available:**
1. `move` (with direction argument: N, S, E, W, NE, NW, SE, SW)
2. `noop` (do nothing)
3. `change_glyph` (communication symbol)

#### Training & Evaluation

**Q: What is the main training function?**

**A:** `cogames.train.train()` which wraps PuffeRL:

```python
from cogames.train import train
from cogames.game import get_game
import torch
from pathlib import Path

env_cfg = get_game("assembler_1_simple")

train(
    env_cfg=env_cfg,
    policy_class_path="cogames.policy.simple.SimplePolicy",
    initial_weights_path=None,
    device=torch.device("cpu"),
    num_steps=10000,
    checkpoints_path=Path("./checkpoints"),
    seed=42,
    batch_size=4096,
    minibatch_size=4096,
)
```

**Mandatory arguments:**
- `env_cfg`: MettaGridConfig
- `policy_class_path`: str (full Python path to policy class)
- `device`: torch.device
- `initial_weights_path`: Optional[str]
- `num_steps`: int
- `checkpoints_path`: Path
- `seed`: int
- `batch_size`: int
- `minibatch_size`: int

**Q: How to load a saved model for inference?**

**A:** Via the Policy's `load_policy_data()` method:

```python
from cogames.policy.simple import SimplePolicy
from mettagrid import MettaGridEnv
import torch

env = MettaGridEnv(env_cfg=config)
policy = SimplePolicy(env, torch.device("cpu"))
policy.load_policy_data("./checkpoints/cogames.cogs_vs_clips/model_1000.pt")

# Create per-agent policies
agent_policy = policy.agent_policy(agent_id=0)
action = agent_policy.step(observation)
```

**Q: What method renders the environment as an image?**

**A:** MettagGrid doesn't have a `render(mode='rgb_array')` method in the traditional sense. Instead:

1. **For interactive play with GUI:** Uses `mettascope` (built-in GUI)
2. **For generating images:** Would need to access internal grid state and render manually

**Alternative approach for tutorial:**
- Use text-based rendering via `miniscope` renderer
- Or create a simple function to visualize the grid state from `env.grid_objects()`

---

## Phase 2: Tutorial Notebook 2 - Introducing Crafting

### Objective
Create `02_crafting_hearts.ipynb` where agent starts with raw materials, finds an assembler to craft a heart, then deposits it.

### Codebase-Specific Questions & Answers

#### Crafting Configuration

**Q: How are crafting recipes defined?**

**A:** In the `AssemblerConfig` via recipes:

```python
from mettagrid.config.mettagrid_config import AssemblerConfig, RecipeConfig

# From stations.py, lines 239-279
assembler = AssemblerConfig(
    name="assembler",
    type_id=8,
    map_char="Z",
    render_symbol="🔄",
    recipes=[
        (
            ["E"],  # Formation pattern (agent approaches from East)
            RecipeConfig(
                input_resources={"energy": 3},
                output_resources={"heart": 1},
                cooldown=1,
            ),
        ),
        (
            ["N"],  # Approach from North
            RecipeConfig(
                input_resources={"germanium": 1},
                output_resources={"decoder": 1},
                cooldown=1,
            ),
        ),
        # ... more recipes
    ],
)
```

**Structure:**
- Each recipe is a tuple: `(formation_pattern, RecipeConfig)`
- `formation_pattern`: List of positions where agents must stand (e.g., `["E"]`, `["N", "S"]`)
- `RecipeConfig` has:
  - `input_resources`: Dict[str, int] - resources consumed
  - `output_resources`: Dict[str, int] - resources produced
  - `cooldown`: int - steps before recipe can be used again

**Q: How to spawn an assembler?**

**A:** Via map_builder objects:

```python
map_builder = RandomMapBuilder.Config(
    width=10,
    height=10,
    agents=1,
    objects={
        "assembler": 1,  # Spawn 1 assembler
        "chest": 1,
    },
    seed=42,
)
```

#### Crafting Logic

**Q: What action triggers crafting?**

**A:** **MOVE** action - agent moves **into** the assembler from a valid position.

The system automatically:
1. Checks if agent is at a valid formation position
2. Verifies agent has required `input_resources`
3. Consumes resources from agent's inventory
4. Adds `output_resources` to agent's inventory
5. Applies cooldown to the assembler

**Q: Where is the crafting logic that checks requirements?**

**A:** This is handled internally by **MettagGrid**, not in CoGames code. The logic:
1. Checks agent position relative to assembler
2. Matches position against recipe `formation_pattern`
3. Verifies inventory has sufficient `input_resources`
4. If valid, executes recipe

**Q: How is inventory updated after crafting?**

**A:** Automatically by MettagGrid's step function. The process:
1. Resources in `input_resources` are subtracted from agent inventory
2. Resources in `output_resources` are added to agent inventory
3. Agent resource limits are enforced (from `config.game.agent.resource_limits`)

**Resource limits example:**
```python
config.game.agent.resource_limits = {
    "heart": 1,  # Can only carry 1 heart at a time
    "energy": 100,
    ("carbon", "oxygen", "germanium", "silicon"): 100,  # Shared limit
}
```

---

## Phase 3: Tutorial Notebook 3 - Full Cycle & Multi-Agent

### Objective
Create `03_full_cycle_and_multi_agent.ipynb` demonstrating:
- Part A: Full single-agent cycle (forage → craft → deposit)
- Part B: Scaling to multi-agent

### Codebase-Specific Questions & Answers

#### Foraging Configuration

**Q: How to spawn resource nodes?**

**A:** Resource "nodes" in CoGames are **extractor stations**:

```python
from cogames.cogs_vs_clips.scenarios import make_game

config = make_game(
    num_cogs=1,
    width=15,
    height=15,
    num_assemblers=1,
    num_chargers=1,  # Energy source
    num_carbon_extractors=1,
    num_oxygen_extractors=1,
    num_germanium_extractors=1,
    num_silicon_extractors=1,
    num_chests=1,
)
```

**Q: What is the agent action for collecting a resource?**

**A:** **MOVE** action - agent moves into the extractor.

**How extractors work:**
- Each extractor auto-generates resources over time (via recipes with cooldown)
- Agent moves into extractor to collect stored resources
- Resources are transferred to agent inventory

Example:
```python
# Carbon extractor (from stations.py, lines 39-54)
def carbon_extractor(max_uses: Optional[int] = None) -> AssemblerConfig:
    return AssemblerConfig(
        name="carbon_extractor",
        recipes=[
            (
                ["Any"],  # Agent can approach from any direction
                RecipeConfig(
                    output_resources={"carbon": 5},
                ),
            )
        ],
    )
```

#### Multi-Agent Dynamics

**Q: Besides changing `num_agents`, is any other code modification required?**

**A:** **NO** - The system automatically handles multi-agent:

```python
# Single agent
config = make_game(num_cogs=1, ...)

# Multi-agent - just change num_cogs
config = make_game(num_cogs=4, ...)
```

The environment and training code automatically adapt:
- Observation space: `(num_agents, 7, 7, 3)`
- Action space: `(num_agents, 2)` - [action_id, argument]
- Rewards: `(num_agents,)` array

**Q: How does the environment handle observations and actions for multiple agents?**

**A:** 

**Observations:**
```python
obs, info = env.reset()
# obs.shape = (num_agents, 7, 7, 3)
# Each agent gets a 7x7x3 egocentric view of the world
```

**Actions:**
```python
actions = np.zeros((env.num_agents, 2), dtype=np.int32)
for agent_id in range(env.num_agents):
    actions[agent_id] = agent_policies[agent_id].step(obs[agent_id])
    
obs, rewards, dones, truncated, info = env.step(actions)
# rewards.shape = (num_agents,)
```

**Q: Does pufferlib default to shared team reward or per-agent rewards?**

**A:** **Per-agent rewards** are returned. The reward structure depends on which stat you track:

```python
# Option 1: Agent-specific rewards (recommended for tutorials)
config.game.agent.rewards = AgentRewards(
    stats={"heart.lost": 1},  # Each agent rewarded when THEY deposit
)

# Option 2: Team-based rewards
config.game.agent.rewards = AgentRewards(
    stats={"chest.heart.amount": 1},  # All agents rewarded when ANY heart deposited
)
```

**Important distinction:**
- `heart.lost`: Per-agent stat - only the agent who deposited gets the reward
- `chest.heart.amount`: Global stat - all agents receive the same reward signal (team reward)

For tutorials, **use `heart.lost`** to provide clearer individual reward signals.

#### (Optional) Cooperative Crafting

**Q: Can crafting logic check for multiple agents?**

**A:** **YES** - via formation patterns in recipes:

```python
# Multi-agent recipe example
recipes=[
    (
        ["N", "S"],  # Requires 2 agents: one North, one South
        RecipeConfig(
            input_resources={"carbon": 10, "silicon": 10},
            output_resources={"heart": 1},
            cooldown=5,
        ),
    ),
]
```

The MettagGrid engine automatically checks:
1. Are there agents at positions `["N", "S"]` relative to the assembler?
2. Do those agents collectively have the required resources?
3. If yes, execute recipe and distribute output to the activator

**Where is this logic?** Inside MettagGrid's core (not exposed in CoGames code).

---

## Phase 4: Review and Finalization

### Codebase-Specific Questions & Answers

#### Performance & Hyperparameters

**Q: What are recommended hyperparameters for solving tutorial tasks?**

**A:** Based on `train.py` (lines 122-210):

**For Feedforward Policies (SimplePolicy):**
```python
learning_rate = 0.015
bptt_horizon = 1  # No RNN
optimizer = "muon"  # Fast convergence
batch_size = 4096
minibatch_size = 4096
update_epochs = 1
gamma = 0.995
gae_lambda = 0.90
ent_coef = 0.001
```

**For RNN Policies (LSTMPolicy):**
```python
learning_rate = 0.0003  # Much lower for stability
bptt_horizon = 1
optimizer = "adam"
adam_eps = 1e-8
batch_size = 4096
minibatch_size = 4096
# Same other hyperparameters as above
```

**Recommended training steps for tutorials:**
- Tutorial 1 (Goal Achievement): 10,000-20,000 steps (~2-3 minutes on CPU)
- Tutorial 2 (Crafting): 50,000 steps (~5-7 minutes on CPU)
- Tutorial 3 (Full Cycle): 100,000-200,000 steps (~10-15 minutes on CPU)

**Note:** These are estimates. Actual times depend on hardware. For Colab, use GPU for faster training.

#### Robustness & Error Handling

**Q: How does configuration parser handle errors?**

**A:** The system uses **Pydantic validation**:

**Invalid item names:**
```python
config.game.agent.initial_inventory = {"invalid_item": 1}
# Will NOT raise error during config creation
# Will cause issues during environment initialization or runtime

# Validation happens when:
# 1. MettaGridEnv(env_cfg=config) is called
# 2. Resources are referenced that don't exist in config.game.resource_names
```

**Resource names are defined in `stations.py`:**
```python
resources = [
    "energy",
    "carbon", 
    "oxygen",
    "germanium",
    "silicon",
    "heart",
    "decoder",
    "modulator",
    "resonator",
    "scrambler",
]
```

**Recommendation for tutorial:**
Add validation helper:
```python
def validate_resources(config, inventory_dict):
    """Validate that all resources in inventory exist in config."""
    valid_resources = set(config.game.resource_names)
    for resource in inventory_dict.keys():
        if resource not in valid_resources:
            raise ValueError(
                f"Invalid resource '{resource}'. "
                f"Valid resources: {', '.join(valid_resources)}"
            )
```

---

## Implementation Architecture Summary

### Core Components

1. **Environment Framework:** MettagGrid (Gymnasium-compatible)
2. **Training Framework:** PufferLib (via `pufferl.PuffeRL`)
3. **Configuration System:** Pydantic models (`MettaGridConfig`)
4. **Policy System:** Custom Policy interface with TrainablePolicy for RL

### Key Classes & Locations

| Component | Location | Class/Function |
|-----------|----------|----------------|
| Environment | `mettagrid` | `MettaGridEnv` |
| Game configs | `cogames.cogs_vs_clips.scenarios` | `make_game()`, `games()` |
| Policy base | `cogames.policy.policy` | `Policy`, `TrainablePolicy`, `AgentPolicy` |
| Simple policy | `cogames.policy.simple` | `SimplePolicy` |
| LSTM policy | `cogames.policy.lstm` | `LSTMPolicy` |
| Training | `cogames.train` | `train()` |
| Playing | `cogames.play` | `play()` |
| CLI | `cogames.main` | `app` (Typer CLI) |

### Data Flow

```
User → CLI (cogames) → game.get_game() → MettaGridConfig
                     ↓
              MettaGridEnv(config)
                     ↓
              Policy(env, device)
                     ↓
              train(env_cfg, policy, ...) 
                     ↓
              pufferlib.vector.make(env_creator)
                     ↓
              pufferl.PuffeRL(vecenv, policy.network())
                     ↓
              Checkpoints saved to disk
```

---

## Tutorial-Specific Simplified Scenarios

### Tutorial 1: Goal Achievement (Simplest)

```python
from cogames.cogs_vs_clips.scenarios import make_game

config = make_game(
    num_cogs=1,
    width=8,
    height=8,
    num_assemblers=0,
    num_chests=1,
    num_chargers=0,
    num_carbon_extractors=0,
    num_oxygen_extractors=0,
    num_germanium_extractors=0,
    num_silicon_extractors=0,
)

# Start agent with a heart
config.game.agent.initial_inventory = {
    "energy": 100,
    "heart": 1,
}

# Configure chest: easy deposits, no withdrawals
config.game.objects["chest"].deposit_positions = ["N", "S", "E", "W"]
config.game.objects["chest"].withdrawal_positions = []

# Reward for depositing (agent-specific stat)
config.game.agent.rewards.stats = {"heart.lost": 1.0}
```

### Tutorial 2: Crafting (Medium)

```python
config = make_game(
    num_cogs=1,
    width=10,
    height=10,
    num_assemblers=1,
    num_chests=1,
    num_chargers=0,
    num_carbon_extractors=0,
    num_oxygen_extractors=0,
    num_germanium_extractors=0,
    num_silicon_extractors=0,
)

# Start with crafting materials
config.game.agent.initial_inventory = {
    "energy": 100,  # Enough to craft and move
}

# Configure chest: easy deposits, no withdrawals
config.game.objects["chest"].deposit_positions = ["N", "S", "E", "W"]
config.game.objects["chest"].withdrawal_positions = []

# Simplify assembler recipe to use only energy
from mettagrid.config.mettagrid_config import RecipeConfig
config.game.objects["assembler"].recipes = [
    (
        ["Any"],  # Can approach from any direction
        RecipeConfig(
            input_resources={"energy": 10},
            output_resources={"heart": 1},
            cooldown=1,
        ),
    )
]

# Reward for depositing (agent-specific stat)
config.game.agent.rewards.stats = {"heart.lost": 1.0}
```

### Tutorial 3: Full Cycle (Complex)

```python
config = make_game(
    num_cogs=1,  # Part A: single agent
    width=15,
    height=15,
    num_assemblers=1,
    num_chests=1,
    num_chargers=1,
    num_carbon_extractors=1,
    num_oxygen_extractors=1,
    num_germanium_extractors=0,
    num_silicon_extractors=0,
)

# Start with minimal resources
config.game.agent.initial_inventory = {
    "energy": 100,
}

# Configure chest: easy deposits, no withdrawals
config.game.objects["chest"].deposit_positions = ["N", "S", "E", "W"]
config.game.objects["chest"].withdrawal_positions = []

# Recipe requires foraging
config.game.objects["assembler"].recipes = [
    (
        ["Any"],  # Can approach from any direction
        RecipeConfig(
            input_resources={"carbon": 3, "oxygen": 3},
            output_resources={"heart": 1},
            cooldown=1,
        ),
    )
]

# Reward for depositing (agent-specific stat)
config.game.agent.rewards.stats = {"heart.lost": 1.0}

# For Part B: just change num_cogs to 4
config.game.num_agents = 4
config.game.map_builder.agents = 4
```

---

## Docker Environment Specification

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    jupyterlab \
    torch --index-url https://download.pytorch.org/whl/cpu \
    imageio \
    imageio-ffmpeg \
    matplotlib \
    numpy \
    pufferlib-core \
    gymnasium

# Clone and install cogames + mettagrid
WORKDIR /workspace
RUN git clone https://github.com/[org]/cogames.git
WORKDIR /workspace/cogames
RUN pip install -e .

# Install mettagrid (workspace dependency)
# This will be handled automatically by the cogames installation

# Expose JupyterLab port
EXPOSE 8888

# Start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
```

---

## Critical Implementation Notes

### 1. Actions are Index-Based

Actions are `np.array([action_id, argument])`:
- `action_id`: Index in action space (0=move, 1=noop, 2=change_glyph)
- `argument`: Direction for move (0=N, 1=S, 2=E, 3=W, ...) or glyph ID

### 2. Observations are Egocentric

Each agent sees a 7x7x3 view centered on itself, not the full map.

### 3. Sparse Rewards

Rewards are only given when hearts are deposited. This makes learning difficult - may need:
- Reward shaping (e.g., small reward for approaching chest)
- Curriculum learning (start simple, add complexity)
- Longer training times

### 4. Multi-Agent Coordination

Agents do not explicitly communicate. Coordination emerges from:
- Shared reward signal
- Observing other agents in their egocentric view
- Optional: using `change_glyph` action to signal

### 5. Energy Management

Energy regenerates slowly (+1 per step) but is consumed by movement (2 per move). Agents must balance exploration vs conservation.

---

## Validation Checklist

Before finalizing tutorials, verify:

- [ ] All code examples run without errors
- [ ] Configuration objects serialize/deserialize correctly
- [ ] Training converges in reasonable time on Colab
- [ ] Saved checkpoints can be loaded for evaluation
- [ ] Rendered outputs are visually clear
- [ ] Error messages are beginner-friendly
- [ ] Multi-agent transition requires only 1 line change
- [ ] Notebooks include timing estimates per cell

---

## Appendix: Quick Reference

### Essential Imports

```python
from mettagrid import MettaGridEnv, MettaGridConfig
from cogames.game import get_game
from cogames.cogs_vs_clips.scenarios import make_game
from cogames.policy.simple import SimplePolicy
from cogames.policy.lstm import LSTMPolicy
from cogames.train import train
import torch
import numpy as np
```

### Basic Environment Usage

```python
# Create environment
config = get_game("assembler_1_simple")
env = MettaGridEnv(env_cfg=config)

# Reset
obs, info = env.reset(seed=42)

# Step
actions = np.array([[0, 2]])  # Move east
obs, rewards, dones, truncated, info = env.step(actions)
```

### Basic Training

```python
train(
    env_cfg=config,
    policy_class_path="cogames.policy.simple.SimplePolicy",
    initial_weights_path=None,
    device=torch.device("cpu"),
    num_steps=10000,
    checkpoints_path=Path("./checkpoints"),
    seed=42,
    batch_size=4096,
    minibatch_size=4096,
)
```

### Basic Policy Evaluation

```python
policy = SimplePolicy(env, torch.device("cpu"))
policy.load_policy_data("./checkpoints/cogames.cogs_vs_clips/model_10000.pt")

agent_policy = policy.agent_policy(0)
action = agent_policy.step(obs[0])
```

---

## Tutorial Visualizations

### Overview

Each tutorial notebook will include comprehensive visualizations to demonstrate learning progress. These visualizations serve multiple purposes:
- **Proof of learning**: Show that the agent is improving
- **Debugging**: Identify issues if training fails
- **Educational**: Help users understand RL concepts
- **Engagement**: Keep learners interested with visual feedback

### Core Visualizations (All Notebooks)

#### 1. Episode Return Curve with Episode Length

**Purpose:** Show learning progress and efficiency gains

```python
def plot_training_progress(episode_returns, episode_lengths, window=50):
    """Plot returns and episode lengths over training."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Episode returns
    ax1.plot(episode_returns, alpha=0.2, color='blue', label='Raw')
    smoothed_returns = smooth_curve(episode_returns, window)
    ax1.plot(smoothed_returns, color='blue', linewidth=2, 
             label=f'Smoothed ({window} episodes)')
    ax1.axhline(2.0, color='red', linestyle='--', label='Success Threshold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Return')
    ax1.set_title('Episode Returns Over Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Episode lengths (efficiency)
    ax2.plot(episode_lengths, alpha=0.2, color='green', label='Raw')
    smoothed_lengths = smooth_curve(episode_lengths, window)
    ax2.plot(smoothed_lengths, color='green', linewidth=2,
             label=f'Smoothed ({window} episodes)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Length Over Training (Lower = More Efficient)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def smooth_curve(values, window=50):
    """Apply moving average smoothing."""
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window)/window, mode='valid')
```

**Expected pattern:**
- Returns: Start near 0 → Increase to 2-3
- Episode length: Start high (~200 steps) → Decrease to ~50 steps (more efficient)

#### 2. Success Rate Over Time

```python
def plot_success_rate(episode_returns, window=50, threshold=2.0):
    """Plot rolling success rate."""
    successes = [1 if r >= threshold else 0 for r in episode_returns]
    success_rate = smooth_curve(successes, window)
    
    plt.figure(figsize=(10, 4))
    plt.plot(success_rate, color='green', linewidth=2)
    plt.axhline(0.6, color='orange', linestyle='--', label='60% Target')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title(f'Success Rate (Rolling {window} episodes)')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    return plt.gcf()
```

#### 3. Value Function Heatmap

**Purpose:** Visualize what the agent has learned about different positions

```python
def plot_value_heatmap(policy, env, map_height=10, map_width=10):
    """Visualize learned value function across map positions.
    
    Shows which positions the agent thinks are valuable.
    High values near chest/assembler indicate good learning.
    """
    import torch
    
    values = np.zeros((map_height, map_width))
    
    # Get value for each position
    for y in range(map_height):
        for x in range(map_width):
            # Create observation as if agent is at this position
            obs = create_observation_at_position(env, x, y)
            
            # Get value estimate from policy
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                _, value = policy.network().forward_eval(obs_tensor)
                values[y, x] = value.item()
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(values, cmap='viridis', interpolation='nearest', origin='lower')
    plt.colorbar(im, label='Estimated Value')
    plt.title('Learned Value Function Heatmap\n(Brighter = Higher Expected Return)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    # Overlay object positions if available
    overlay_map_objects(env)
    
    return plt.gcf()

def create_observation_at_position(env, x, y):
    """Helper to create observation for value function visualization."""
    # This is a simplified version - actual implementation depends on
    # how MettagGrid constructs observations
    # May need to reset env and move agent to position
    pass

def overlay_map_objects(env):
    """Overlay chest, assembler positions on heatmap."""
    # Extract object positions from env configuration
    # Plot them as markers on the heatmap
    pass
```

**Interpretation guide for notebooks:**
- Bright spots should appear near goal objects (chest, assembler)
- Dark areas indicate low-value positions (far from goals)
- Shows agent has learned spatial structure of task

### Notebook 1: Goal Delivery Visualizations

#### Subtask Completion: Navigation Progress

```python
def plot_navigation_progress(episode_data):
    """Track agent's distance to chest over training.
    
    Shows agent learning to move toward goal efficiently.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Distance to chest over episode (sample episode)
    sample_episode = episode_data['distances'][-1]  # Last episode
    ax1.plot(sample_episode, color='blue', linewidth=2)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Distance to Chest')
    ax1.set_title('Distance to Goal During Episode\n(Final Episode)')
    ax1.grid(True, alpha=0.3)
    
    # Average initial vs final distance
    initial_distances = [ep[0] for ep in episode_data['distances']]
    final_distances = [ep[-1] for ep in episode_data['distances']]
    
    ax2.plot(initial_distances, alpha=0.3, label='Initial Distance')
    ax2.plot(smooth_curve(initial_distances, 20), linewidth=2, label='Initial (Smoothed)')
    ax2.plot(final_distances, alpha=0.3, label='Final Distance')
    ax2.plot(smooth_curve(final_distances, 20), linewidth=2, label='Final (Smoothed)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Distance')
    ax2.set_title('Distance to Chest Over Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

#### Position Heatmap

```python
def plot_position_heatmap(visited_positions, map_height=10, map_width=10):
    """Show where agent spent time during training.
    
    Helps visualize exploration vs exploitation.
    """
    heatmap = np.zeros((map_height, map_width))
    
    for pos in visited_positions:
        x, y = pos
        if 0 <= x < map_width and 0 <= y < map_height:
            heatmap[y, x] += 1
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(heatmap, cmap='hot', interpolation='nearest', origin='lower')
    plt.colorbar(im, label='Visit Count')
    plt.title('Agent Position Heatmap\n(Brighter = More Time Spent)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    # Mark chest location
    overlay_map_objects(env)
    
    return plt.gcf()
```

### Notebook 2: Crafting - Subtask Completion Tracking

```python
def plot_crafting_subtasks(history):
    """Track completion of crafting sub-goals.
    
    Shows agent learning the sequence: navigate → craft → deposit
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Assembler visits per episode
    axes[0, 0].plot(history['assembler_visits_per_episode'])
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Visits')
    axes[0, 0].set_title('Assembler Visits Per Episode')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Hearts crafted per episode
    axes[0, 1].plot(history['hearts_crafted_per_episode'], color='orange')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Hearts Crafted')
    axes[0, 1].set_title('Crafting Success Per Episode')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Hearts deposited per episode
    axes[1, 0].plot(history['hearts_deposited_per_episode'], color='green')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Hearts Deposited')
    axes[1, 0].set_title('Deposit Success Per Episode')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Subtask completion rate
    subtask_rates = []
    for i in range(len(history['assembler_visits_per_episode'])):
        visited_assembler = history['assembler_visits_per_episode'][i] > 0
        crafted_heart = history['hearts_crafted_per_episode'][i] > 0
        deposited_heart = history['hearts_deposited_per_episode'][i] > 0
        
        completed = sum([visited_assembler, crafted_heart, deposited_heart])
        subtask_rates.append(completed / 3.0)
    
    axes[1, 1].plot(smooth_curve(subtask_rates, 20), color='purple', linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Completion Rate')
    axes[1, 1].set_title('Subtask Completion Rate\n(All 3 steps completed)')
    axes[1, 1].set_ylim(0, 1.1)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

#### Resource Inventory Timeline

```python
def plot_inventory_timeline(inventory_history):
    """Show resource flow during episode.
    
    Visualizes: Resources consumed → Heart created → Heart deposited
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Resource inventory over time
    steps = range(len(inventory_history['carbon']))
    axes[0].plot(steps, inventory_history['carbon'], label='Carbon', marker='o')
    axes[0].plot(steps, inventory_history['oxygen'], label='Oxygen', marker='s')
    axes[0].plot(steps, inventory_history['germanium'], label='Germanium', marker='^')
    axes[0].plot(steps, inventory_history['silicon'], label='Silicon', marker='d')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Quantity')
    axes[0].set_title('Crafting Resources Over Episode')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Heart inventory (shows crafting and deposit moments)
    axes[1].plot(steps, inventory_history['heart'], 
                 color='red', linewidth=3, marker='o', markersize=8)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Hearts')
    axes[1].set_title('Heart Inventory Over Episode\n(Increase = Crafted, Decrease = Deposited)')
    axes[1].set_ylim(-0.5, max(inventory_history['heart']) + 0.5)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

### Notebook 3: Multi-Agent Coordination Views

#### Individual vs Team Performance

```python
def plot_multiagent_returns(agent_histories, num_agents=4):
    """Compare individual agent and team performance.
    
    Shows coordination emerging from individual learning.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Individual agent returns
    for i in range(num_agents):
        agent_returns = agent_histories[i]['returns']
        axes[0, 0].plot(smooth_curve(agent_returns, 50), 
                        label=f'Agent {i}', alpha=0.7, linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].set_title('Individual Agent Returns')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Team total return
    team_returns = []
    max_len = max(len(agent_histories[i]['returns']) for i in range(num_agents))
    for ep in range(max_len):
        episode_sum = sum(
            agent_histories[i]['returns'][ep] 
            for i in range(num_agents) 
            if ep < len(agent_histories[i]['returns'])
        )
        team_returns.append(episode_sum)
    
    axes[0, 1].plot(smooth_curve(team_returns, 50), 
                    color='purple', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Team Return')
    axes[0, 1].set_title('Team Total Return')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Episode length comparison
    for i in range(num_agents):
        agent_lengths = agent_histories[i]['episode_lengths']
        axes[1, 0].plot(smooth_curve(agent_lengths, 50),
                        label=f'Agent {i}', alpha=0.7, linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].set_title('Episode Lengths (Efficiency)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Success rate per agent
    for i in range(num_agents):
        agent_returns = agent_histories[i]['returns']
        successes = [1 if r >= 1.0 else 0 for r in agent_returns]
        success_rate = smooth_curve(successes, 50)
        axes[1, 1].plot(success_rate, label=f'Agent {i}', alpha=0.7, linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Success Rate')
    axes[1, 1].set_title('Per-Agent Success Rates')
    axes[1, 1].set_ylim(0, 1.1)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

#### Agent Activity Specialization

```python
def plot_agent_specialization(agent_activities, num_agents=4):
    """Visualize if agents develop specialized roles.
    
    Shows what each agent spends time doing.
    """
    import seaborn as sns
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Activity heatmap
    activities = ['Extractor Visits', 'Assembler Visits', 'Chest Visits']
    activity_matrix = np.array([
        [agent_activities[i]['extractor_visits'] for i in range(num_agents)],
        [agent_activities[i]['assembler_visits'] for i in range(num_agents)],
        [agent_activities[i]['chest_visits'] for i in range(num_agents)],
    ])
    
    sns.heatmap(activity_matrix, 
                xticklabels=[f'Agent {i}' for i in range(num_agents)],
                yticklabels=activities,
                annot=True, fmt='.0f', cmap='Blues', ax=ax1)
    ax1.set_title('Agent Activity Patterns\n(Total Visits)')
    
    # Time spent distribution
    time_matrix = np.array([
        [agent_activities[i]['time_at_extractors'] for i in range(num_agents)],
        [agent_activities[i]['time_at_assembler'] for i in range(num_agents)],
        [agent_activities[i]['time_at_chest'] for i in range(num_agents)],
        [agent_activities[i]['time_moving'] for i in range(num_agents)],
    ])
    
    # Normalize to percentages
    time_matrix = time_matrix / time_matrix.sum(axis=0) * 100
    
    sns.heatmap(time_matrix,
                xticklabels=[f'Agent {i}' for i in range(num_agents)],
                yticklabels=['Extractors', 'Assembler', 'Chest', 'Moving'],
                annot=True, fmt='.1f', cmap='YlOrRd', ax=ax2,
                cbar_kws={'label': 'Percentage of Time'})
    ax2.set_title('Time Allocation Per Agent\n(Percentage)')
    
    plt.tight_layout()
    return fig
```

#### Coordination Efficiency Metrics

```python
def plot_coordination_metrics(episode_data):
    """Measure coordination efficiency.
    
    Tracks collisions, resource conflicts, and task distribution.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Collision rate over training
    collision_rate = [ep['collisions'] / ep['steps'] for ep in episode_data]
    axes[0, 0].plot(smooth_curve(collision_rate, 20), color='red', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Collisions per Step')
    axes[0, 0].set_title('Agent Collision Rate\n(Lower = Better Coordination)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Resource gathering balance
    resource_balance = [np.std(ep['resources_per_agent']) for ep in episode_data]
    axes[0, 1].plot(smooth_curve(resource_balance, 20), color='orange', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Std Dev of Resources')
    axes[0, 1].set_title('Resource Gathering Balance\n(Lower = More Equal Distribution)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Task distribution (who completes hearts)
    hearts_per_agent = np.array([ep['hearts_per_agent'] for ep in episode_data[-100:]])
    axes[1, 0].boxplot(hearts_per_agent, labels=[f'Agent {i}' for i in range(4)])
    axes[1, 0].set_ylabel('Hearts Deposited')
    axes[1, 0].set_title('Task Distribution (Last 100 Episodes)')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Concurrent activity (agents working simultaneously)
    concurrent_activity = [ep['avg_active_agents'] for ep in episode_data]
    axes[1, 1].plot(smooth_curve(concurrent_activity, 20), 
                    color='green', linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Avg Active Agents')
    axes[1, 1].set_title('Concurrent Activity\n(Higher = Better Parallelization)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

### Utility Functions

```python
# Save to tutorial_viz.py module

def create_gif_from_episode(env, policy, filename='episode.gif', max_steps=200):
    """Record episode and save as animated GIF.
    
    Args:
        env: The environment
        policy: Trained policy
        filename: Output filename
        max_steps: Maximum steps to record
    
    Returns:
        Path to saved GIF
    """
    import imageio
    
    frames = []
    obs, _ = env.reset()
    
    for step in range(max_steps):
        # Render frame (need to implement RGB rendering)
        # For now, can use text representation or placeholder
        frame = render_env_to_rgb(env)
        frames.append(frame)
        
        # Get action from policy
        actions = []
        for agent_id in range(env.num_agents):
            agent_policy = policy.agent_policy(agent_id)
            action = agent_policy.step(obs[agent_id])
            actions.append(action)
        
        # Step environment
        obs, rewards, dones, truncated, info = env.step(np.array(actions))
        
        if all(dones) or all(truncated):
            break
    
    imageio.mimsave(filename, frames, fps=10)
    return filename

def render_env_to_rgb(env):
    """Convert environment state to RGB image.
    
    This is a placeholder - actual implementation depends on
    MettagGrid's rendering capabilities.
    """
    # TODO: Implement based on env.grid_objects() or similar
    # For now, return a placeholder
    return np.zeros((256, 256, 3), dtype=np.uint8)

def create_metrics_table(episode_returns, episode_lengths, training_time):
    """Create summary table of training results."""
    import pandas as pd
    
    # Calculate metrics
    final_success_rate = np.mean([1 if r >= 2.0 else 0 
                                   for r in episode_returns[-100:]])
    avg_return = np.mean(episode_returns[-100:])
    avg_length = np.mean(episode_lengths[-100:])
    best_return = np.max(episode_returns)
    
    metrics = pd.DataFrame({
        'Metric': [
            'Final Success Rate',
            'Avg Return (Last 100)',
            'Best Return',
            'Avg Episode Length',
            'Training Time',
            'Total Episodes'
        ],
        'Value': [
            f'{final_success_rate:.1%}',
            f'{avg_return:.2f}',
            f'{best_return:.2f}',
            f'{avg_length:.0f} steps',
            f'{training_time:.1f}s',
            f'{len(episode_returns)}'
        ]
    })
    
    return metrics
```

### Implementation in Notebooks

Each notebook should include a visualization section:

```python
# Example notebook structure

## Training Progress
plot_training_progress(episode_returns, episode_lengths)

## Success Rate
plot_success_rate(episode_returns)

## Value Function (What Agent Learned)
plot_value_heatmap(policy, env)

## Subtask Analysis (Notebook 2 & 3)
plot_crafting_subtasks(history)  # Notebook 2
plot_multiagent_returns(agent_histories)  # Notebook 3

## Coordination Analysis (Notebook 3 only)
plot_agent_specialization(agent_activities)
plot_coordination_metrics(episode_data)

## Summary Table
display(create_metrics_table(episode_returns, episode_lengths, training_time))
```

---

## Next Steps

1. Create Dockerfile and test environment setup
2. Implement Tutorial 1 notebook with all explanations and visualizations
3. Create `tutorial_viz.py` module with visualization utilities
4. Test training convergence and adjust hyperparameters
5. Implement Tutorial 2 and 3 with progressive visualizations
6. Create sample outputs (GIFs, plots) for documentation
7. Final testing on clean Colab instance
8. Documentation review and polish

---

**Document Version:** 1.1  
**Last Updated:** October 4, 2025  
**Status:** Ready for Implementation  
**Update:** Added comprehensive visualization specifications including:
- Episode return curves with episode length tracking
- Value function heatmaps for all stages
- Subtask completion tracking for progressive stages
- Multi-agent coordination views and metrics

