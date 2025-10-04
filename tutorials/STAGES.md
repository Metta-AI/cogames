# Simplified Tutorial Stages for CoGames

This document provides simplified stage configurations for creating the 3-notebook tutorial series. Based on the cogology curriculum but dramatically simplified for learning purposes.

## Key Configuration Notes

**Reward System:**
- Uses `heart.lost` stat (agent-specific) instead of `chest.heart.amount` (global stat)
- Agents receive +1 reward when a heart leaves their inventory (deposited into chest)
- This is a per-agent reward signal, making it suitable for both single and multi-agent scenarios

**Chest Configuration:**
- Deposits allowed from all directions: `deposit_positions = ["N", "S", "E", "W"]`
- Withdrawals disabled: `withdrawal_positions = []`
- This ensures hearts can only go into chests, never come out
- Makes the task unidirectional and simpler to learn

## Stage 1: Goal Delivery (Tutorial Notebook 1)

**Objective:** Learn to deposit hearts that are already in inventory.

**Configuration:**
```python
from cogames.cogs_vs_clips.scenarios import make_game
from mettagrid.config.mettagrid_config import RecipeConfig

# Create simple map with 1 agent, 1 chest, no extractors
config = make_game(
    num_cogs=1,
    width=10,
    height=10,
    num_assemblers=0,  # No crafting yet
    num_chests=1,
    num_chargers=0,
    num_carbon_extractors=0,
    num_oxygen_extractors=0,
    num_germanium_extractors=0,
    num_silicon_extractors=0,
)

# Agent starts with hearts
config.game.agent.initial_inventory = {
    "energy": 100,
    "heart": 3,  # Start with 3 hearts
}

# Increase heart carrying capacity
config.game.agent.resource_limits["heart"] = 5

# Configure chest to accept deposits but not withdrawals
config.game.objects["chest"].deposit_positions = ["N", "S", "E", "W"]  # Accept from all sides
config.game.objects["chest"].withdrawal_positions = []  # No withdrawals allowed

# Reward for depositing hearts (agent-specific stat)
config.game.agent.rewards.stats = {
    "heart.lost": 1.0  # +1 reward per heart lost from inventory (deposited)
}
```

**Learning Goals:**
- Navigate to chest
- Understand move action
- Deposit items by moving into chest from correct position
- Observe sparse rewards

**Expected Training Time:** 10,000-20,000 steps (~2-3 minutes)

**Visualizations to Include:**
1. **Episode return curve + Episode length** - Shows learning and efficiency
2. **Success rate over time** - Should reach 60%+
3. **Value function heatmap** - Should show high values near chest
4. **Navigation progress** - Distance to chest decreasing over episodes
5. **Position heatmap** - Should show concentration near chest after training

---

## Stage 2: Simple Assembly (Tutorial Notebook 2)

**Objective:** Craft hearts from resources, then deposit them.

**Configuration:**
```python
config = make_game(
    num_cogs=1,
    width=12,
    height=12,
    num_assemblers=1,  # Add assembler for crafting
    num_chests=1,
    num_chargers=0,
    num_carbon_extractors=0,
    num_oxygen_extractors=0,
    num_germanium_extractors=0,
    num_silicon_extractors=0,
)

# Agent starts with crafting materials
config.game.agent.initial_inventory = {
    "energy": 100,
    "carbon": 5,
    "oxygen": 5,
    "germanium": 5,
    "silicon": 5,
}

# Simplified recipe: use all 4 resources to craft 1 heart
config.game.objects["assembler"].recipes = [
    (
        ["Any"],  # Can approach from any direction
        RecipeConfig(
            input_resources={
                "carbon": 1,
                "oxygen": 1,
                "germanium": 1,
                "silicon": 1,
            },
            output_resources={"heart": 1},
            cooldown=1,
        ),
    )
]

# Configure chest to accept deposits but not withdrawals
config.game.objects["chest"].deposit_positions = ["N", "S", "E", "W"]  # Accept from all sides
config.game.objects["chest"].withdrawal_positions = []  # No withdrawals allowed

# Same reward as Stage 1 (agent-specific stat)
config.game.agent.rewards.stats = {
    "heart.lost": 1.0
}
```

**Learning Goals:**
- Navigate to assembler
- Understand crafting via move action
- Multi-step planning: craft → navigate → deposit
- Resource management

**Expected Training Time:** 50,000 steps (~5-7 minutes)

**Visualizations to Include:**
1. **Episode return curve + Episode length** - Shows learning multi-step task
2. **Success rate over time** - Should reach 60%+
3. **Value function heatmap** - High values near assembler AND chest
4. **Subtask completion tracking**:
   - Assembler visits per episode
   - Hearts crafted per episode
   - Hearts deposited per episode
   - Overall subtask completion rate
5. **Resource inventory timeline** - Shows resources → heart → deposit sequence
6. **Before/after GIF comparison** - Random vs trained behavior

---

## Stage 3: Single Resource Foraging (Tutorial Notebook 3)

**Objective:** Forage ONE resource, craft hearts, deposit them.

**Configuration:**
```python
config = make_game(
    num_cogs=1,
    width=15,
    height=15,
    num_assemblers=1,
    num_chests=1,
    num_chargers=0,
    num_carbon_extractors=3,  # Add carbon extractors
    num_oxygen_extractors=0,
    num_germanium_extractors=0,
    num_silicon_extractors=0,
)

# Agent starts with 3 resources but must forage carbon
config.game.agent.initial_inventory = {
    "energy": 100,
    "oxygen": 5,
    "germanium": 5,
    "silicon": 5,
    # NO carbon - must forage it
}

# Configure carbon extractors to give 1 carbon per use
config.game.objects["carbon_extractor"].recipes[0][1].output_resources = {
    "carbon": 1
}

# Configure chest to accept deposits but not withdrawals
config.game.objects["chest"].deposit_positions = ["N", "S", "E", "W"]  # Accept from all sides
config.game.objects["chest"].withdrawal_positions = []  # No withdrawals allowed

# Same crafting recipe as Stage 2
config.game.objects["assembler"].recipes = [
    (
        ["Any"],
        RecipeConfig(
            input_resources={
                "carbon": 1,
                "oxygen": 1,
                "germanium": 1,
                "silicon": 1,
            },
            output_resources={"heart": 1},
            cooldown=1,
        ),
    )
]

# Same reward as Stage 1 (agent-specific stat)
config.game.agent.rewards.stats = {
    "heart.lost": 1.0
}
```

**Learning Goals:**
- Navigate to extractor
- Collect resources via move action
- Multi-step planning: forage → craft → deposit
- Handle multiple stations

**Expected Training Time:** 100,000 steps (~10-15 minutes)

**Visualizations to Include:**
1. **Episode return curve + Episode length** - Shows foraging behavior emergence
2. **Success rate over time** - Should reach 50%+ (harder task)
3. **Value function heatmap** - High values near extractors, assembler, AND chest
4. **Subtask completion tracking**:
   - Extractor visits per episode
   - Resources collected per episode
   - Assembler visits per episode
   - Hearts crafted and deposited
5. **Resource collection efficiency** - Carbon gathering rate over time
6. **Multi-step sequence visualization** - Forage → Craft → Deposit timeline

---

## Multi-Agent Scaling (Extension of Tutorial 3)

**Objective:** Scale Stage 3 to multiple agents with shared reward.

**Configuration:**
```python
# Same as Stage 3, but change num_cogs
config = make_game(
    num_cogs=4,  # Now 4 agents
    width=15,
    height=15,
    num_assemblers=1,
    num_chests=1,
    num_chargers=0,
    num_carbon_extractors=3,
    num_oxygen_extractors=0,
    num_germanium_extractors=0,
    num_silicon_extractors=0,
)

# IMPORTANT: Also update map_builder
config.game.num_agents = 4
config.game.map_builder.agents = 4

# Configure chest to accept deposits but not withdrawals
config.game.objects["chest"].deposit_positions = ["N", "S", "E", "W"]
config.game.objects["chest"].withdrawal_positions = []

# All other settings identical to single-agent Stage 3
```

**Learning Goals:**
- Coordination emerges from shared reward
- Agents learn to avoid conflicts
- Shared resources require implicit cooperation

**Expected Training Time:** 200,000 steps (~15-20 minutes)

**Visualizations to Include:**
1. **Multi-agent return curves**:
   - Individual agent returns (4 separate lines)
   - Team total return
   - Episode lengths per agent
   - Success rates per agent
2. **Coordination metrics**:
   - Agent activity specialization heatmap
   - Time allocation per agent
   - Collision rate over training (should decrease)
   - Resource gathering balance
3. **Task distribution** - Boxplot showing hearts deposited per agent
4. **Concurrent activity** - Shows parallelization improving
5. **Multi-agent GIF** - Shows all 4 agents working together
6. **Value function comparison** - Different agents may learn different values

---

## Stage 4 (Optional): Full Foraging

If you want to extend the tutorial further, Stage 4 would require foraging ALL resources:

**Configuration:**
```python
config = make_game(
    num_cogs=1,
    width=20,
    height=20,
    num_assemblers=1,
    num_chests=1,
    num_chargers=1,  # Add energy source
    num_carbon_extractors=2,
    num_oxygen_extractors=2,
    num_germanium_extractors=2,
    num_silicon_extractors=2,
)

# Agent starts with NOTHING
config.game.agent.initial_inventory = {
    "energy": 100,  # Just starting energy
}

# Configure chest to accept deposits but not withdrawals
config.game.objects["chest"].deposit_positions = ["N", "S", "E", "W"]
config.game.objects["chest"].withdrawal_positions = []

# All extractors give 1 resource per use
for extractor_name in ["carbon_extractor", "oxygen_extractor", 
                       "germanium_extractor", "silicon_extractor"]:
    if extractor_name in config.game.objects:
        config.game.objects[extractor_name].recipes[0][1].output_resources = {
            extractor_name.replace("_extractor", ""): 1
        }

# Charger gives 100 energy
config.game.objects["charger"].recipes[0][1].output_resources = {"energy": 100}
```

---

## Implementation Helper Functions

### Create Stage 1 Environment
```python
def create_stage1_env():
    """Create Stage 1: Goal Delivery environment."""
    config = make_game(
        num_cogs=1, width=10, height=10,
        num_assemblers=0, num_chests=1,
        num_chargers=0, num_carbon_extractors=0,
        num_oxygen_extractors=0, num_germanium_extractors=0,
        num_silicon_extractors=0,
    )
    config.game.agent.initial_inventory = {"energy": 100, "heart": 3}
    config.game.agent.resource_limits["heart"] = 5
    
    # Configure chest: deposits allowed from all sides, no withdrawals
    config.game.objects["chest"].deposit_positions = ["N", "S", "E", "W"]
    config.game.objects["chest"].withdrawal_positions = []
    
    # Reward for depositing hearts (agent-specific stat)
    config.game.agent.rewards.stats = {"heart.lost": 1.0}
    return config
```

### Create Stage 2 Environment
```python
def create_stage2_env():
    """Create Stage 2: Simple Assembly environment."""
    config = make_game(
        num_cogs=1, width=12, height=12,
        num_assemblers=1, num_chests=1,
        num_chargers=0, num_carbon_extractors=0,
        num_oxygen_extractors=0, num_germanium_extractors=0,
        num_silicon_extractors=0,
    )
    config.game.agent.initial_inventory = {
        "energy": 100, "carbon": 5, "oxygen": 5,
        "germanium": 5, "silicon": 5,
    }
    
    # Configure chest: deposits allowed from all sides, no withdrawals
    config.game.objects["chest"].deposit_positions = ["N", "S", "E", "W"]
    config.game.objects["chest"].withdrawal_positions = []
    
    # Simplified recipe
    from mettagrid.config.mettagrid_config import RecipeConfig
    config.game.objects["assembler"].recipes = [
        (["Any"], RecipeConfig(
            input_resources={"carbon": 1, "oxygen": 1, "germanium": 1, "silicon": 1},
            output_resources={"heart": 1},
            cooldown=1,
        ))
    ]
    
    # Reward for depositing hearts (agent-specific stat)
    config.game.agent.rewards.stats = {"heart.lost": 1.0}
    return config
```

### Create Stage 3 Environment
```python
def create_stage3_env(num_agents=1):
    """Create Stage 3: Single Resource Foraging environment."""
    config = make_game(
        num_cogs=num_agents, width=15, height=15,
        num_assemblers=1, num_chests=1,
        num_chargers=0, num_carbon_extractors=3,
        num_oxygen_extractors=0, num_germanium_extractors=0,
        num_silicon_extractors=0,
    )
    config.game.num_agents = num_agents
    config.game.map_builder.agents = num_agents
    
    config.game.agent.initial_inventory = {
        "energy": 100, "oxygen": 5, "germanium": 5, "silicon": 5,
    }
    
    # Configure chest: deposits allowed from all sides, no withdrawals
    config.game.objects["chest"].deposit_positions = ["N", "S", "E", "W"]
    config.game.objects["chest"].withdrawal_positions = []
    
    # Fix carbon extractor output
    config.game.objects["carbon_extractor"].recipes[0][1].output_resources = {"carbon": 1}
    
    # Recipe
    from mettagrid.config.mettagrid_config import RecipeConfig
    config.game.objects["assembler"].recipes = [
        (["Any"], RecipeConfig(
            input_resources={"carbon": 1, "oxygen": 1, "germanium": 1, "silicon": 1},
            output_resources={"heart": 1},
            cooldown=1,
        ))
    ]
    
    # Reward for depositing hearts (agent-specific stat)
    config.game.agent.rewards.stats = {"heart.lost": 1.0}
    return config
```

---

## Training Hyperparameters (From Original Code)

### For SimplePolicy (Feedforward)
```python
learning_rate = 0.015
optimizer = "muon"
batch_size = 4096
minibatch_size = 4096
num_steps = {
    "stage1": 20_000,
    "stage2": 50_000,
    "stage3": 100_000,
}
```

### For LSTMPolicy (Recurrent)
```python
learning_rate = 0.0003
optimizer = "adam"
batch_size = 4096
minibatch_size = 4096
num_steps = {
    "stage1": 20_000,
    "stage2": 50_000,
    "stage3": 100_000,
}
```

---

## Visualization Helpers

### Render Episode as GIF
```python
def render_episode_to_gif(env, policy, filename="episode.gif", max_steps=200):
    """Render a full episode and save as GIF."""
    import imageio
    import numpy as np
    
    frames = []
    obs, _ = env.reset()
    
    for step in range(max_steps):
        # Get action from policy
        agent_policy = policy.agent_policy(0)
        action = agent_policy.step(obs[0])
        
        # Step environment
        obs, rewards, dones, truncated, info = env.step(np.array([action]))
        
        # Render frame (would need custom rendering logic here)
        # frame = env.render_to_rgb()
        # frames.append(frame)
        
        if all(dones) or all(truncated):
            break
    
    # Save as GIF
    # imageio.mimsave(filename, frames, fps=10)
    print(f"Episode completed in {step+1} steps")
```

---

## Success Criteria (Simplified)

**Stage 1:**
- Average return > 2.0 (deposited 2+ hearts)
- Success in 60%+ of episodes

**Stage 2:**
- Average return > 2.0 (crafted and deposited 2+ hearts)
- Success in 60%+ of episodes

**Stage 3:**
- Average return > 1.5 (foraged, crafted, deposited)
- Success in 50%+ of episodes (harder due to foraging)

---

## Docker Environment for Tutorials

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y git build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    jupyterlab \
    torch --index-url https://download.pytorch.org/whl/cpu \
    imageio imageio-ffmpeg \
    matplotlib numpy scipy \
    gymnasium pufferlib-core

WORKDIR /workspace
COPY . /workspace/cogames
WORKDIR /workspace/cogames
RUN pip install -e .

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
```

---

## Key Simplifications from Original Code

1. **Removed multi-room architecture** - Single room per environment
2. **Removed variants A-L** - Just one configuration per stage
3. **Removed per-agent chest tracking** - Global reward only
4. **Removed automatic progression** - Manual stage transitions
5. **Removed learning progress sampling** - Simple task generation
6. **Removed success tracking callbacks** - Manual evaluation
7. **Removed stochastic reward shaping** - Pure sparse rewards
8. **Removed position-dependent recipes** - All recipes use "Any" position
9. **Removed extractor depletion** - Infinite resources for simplicity
10. **Removed chargers from early stages** - Focus on core mechanics

This simplified version is perfect for tutorial notebooks where clarity and learning are more important than advanced features.

**Document Version:** 1.1  
**Last Updated:** October 4, 2025  
**Update:** Added visualization specifications for each stage including episode lengths, subtask tracking, value heatmaps, and multi-agent coordination views.

---

## Visualization Summary

### Essential Visualizations for All Stages

| Visualization | Purpose | Expected Pattern |
|--------------|---------|------------------|
| **Episode Returns** | Shows learning progress | Start ~0 → Increase to 2-3 |
| **Episode Length** | Shows efficiency gains | Start ~200 → Decrease to ~50 |
| **Success Rate** | Shows task completion | Start ~0% → Reach 60%+ |
| **Value Heatmap** | Shows spatial understanding | Bright spots near goal objects |

### Stage-Specific Visualizations

**Stage 1: Goal Delivery**
- Navigation progress (distance to chest)
- Position heatmap (where agent explores)

**Stage 2: Simple Assembly**
- Subtask completion (visit → craft → deposit)
- Resource inventory timeline
- Before/after GIF comparison

**Stage 3: Single Resource Foraging**
- Extractor visits and collection efficiency
- Multi-step sequence visualization
- Resource gathering rate

**Multi-Agent Extension**
- Individual vs team returns
- Agent specialization heatmap
- Coordination metrics (collisions, balance)
- Task distribution across agents

### Visualization Module Structure

Create `tutorial_viz.py` with these functions:

```python
"""Visualization utilities for CoGames tutorials."""

# Core visualizations (all stages)
def plot_training_progress(returns, lengths, window=50)
def plot_success_rate(returns, window=50, threshold=2.0)
def plot_value_heatmap(policy, env, map_height, map_width)
def smooth_curve(values, window=50)

# Stage 1 visualizations
def plot_navigation_progress(episode_data)
def plot_position_heatmap(visited_positions, map_height, map_width)

# Stage 2 visualizations
def plot_crafting_subtasks(history)
def plot_inventory_timeline(inventory_history)

# Stage 3 visualizations
def plot_resource_efficiency(collection_history)

# Multi-agent visualizations
def plot_multiagent_returns(agent_histories, num_agents=4)
def plot_agent_specialization(agent_activities, num_agents=4)
def plot_coordination_metrics(episode_data)

# Utility functions
def create_gif_from_episode(env, policy, filename, max_steps=200)
def create_metrics_table(returns, lengths, training_time)
```

### Interpretation Guides for Notebooks

**Episode Returns:**
- Flat at 0: Agent not finding chest/completing task
- Gradual increase: Agent learning
- Oscillating: Normal RL variance
- Target: Average > 2.0 (deposited 2+ hearts)

**Episode Length:**
- Decreasing trend: Agent getting more efficient
- Very short (<10 steps): May be stuck in local optimum
- Very long (>200 steps): Not learning efficient paths

**Value Function:**
- Bright spots at goals: Good spatial understanding
- Uniform values: Agent hasn't learned anything
- Bright paths: Agent learned navigation routes

**Success Rate:**
- Should steadily increase to 60%+
- Plateau below 50%: May need more training or tuning
- Rapid increase then plateau: Good convergence

**Multi-Agent Coordination:**
- Decreasing collisions: Learning to avoid conflicts
- Balanced resource gathering: Fair task distribution
- Increasing concurrent activity: Better parallelization
- Similar success rates: No agent left behind

