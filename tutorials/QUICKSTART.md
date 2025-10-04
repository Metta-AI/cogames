# CoGames Tutorial Quickstart Guide

## 🚀 How to Run the Tutorials

This guide walks you through the actual steps to train agents using the tutorial notebooks.

---

## Prerequisites

### 1. Install CoGames and Dependencies

From the cogames repository root:

```bash
cd /Users/bullm/Documents/GitHub/cogames

# Install cogames in development mode
pip install -e .

# Verify installation
python -c "import cogames; print('CoGames installed successfully!')"
```

**Required packages** (should be installed automatically):
- `torch` (PyTorch)
- `numpy`
- `matplotlib`
- `pufferlib-core`
- `mettagrid`
- `pydantic`

### 2. Navigate to Tutorials Directory

```bash
cd tutorials
```

---

## Option 1: Running in Jupyter Notebook (Recommended)

### Step 1: Start Jupyter

```bash
# Install jupyter if you haven't
pip install jupyter

# Start Jupyter Notebook
jupyter notebook

# Or use JupyterLab (recommended)
pip install jupyterlab
jupyter lab
```

This will open a browser window with the file browser.

### Step 2: Open Tutorial 1

1. Click on `01_goal_delivery.ipynb`
2. You'll see all the cells we created

### Step 3: Run the Notebook

**Option A: Run All Cells at Once**
- Menu: `Cell` → `Run All`
- This will execute the entire tutorial start to finish

**Option B: Run Cell-by-Cell (Recommended for Learning)**
- Click on first cell
- Press `Shift + Enter` to run and move to next cell
- Read the output, understand what's happening
- Continue through all cells

### Step 4: Monitor Training

When you reach the training cell (cell 8), you'll see:
```
🚀 Starting training...
============================================================
[PufferLib output with training stats]
============================================================
✅ Training complete!
CPU times: user 2min 30s, sys: 15s, total: 2min 45s
```

**What to expect:**
- Training takes ~2-3 minutes for Tutorial 1
- You'll see live updates from PufferLib
- Learning rates, losses, episode stats will print
- Progress bar shows steps completed

### Step 5: View Results

After training completes, continue running cells to see:
- Learning curves (returns over time)
- Success rate plots
- Performance metrics
- Saved checkpoint locations

---

## Option 2: Running as Python Script

If you prefer running without Jupyter:

### Step 1: Convert Notebook to Python

```bash
cd tutorials
jupyter nbconvert --to python 01_goal_delivery.ipynb
```

This creates `01_goal_delivery.py`

### Step 2: Run the Script

```bash
python 01_goal_delivery.py
```

**Note:** You won't see interactive plots with this method. They'll be displayed briefly or saved to files.

---

## Option 3: Running Individual Sections

You can also run specific parts of the tutorial:

```python
# In a Python REPL or script
import sys
sys.path.append('/Users/bullm/Documents/GitHub/cogames/tutorials')

import numpy as np
import torch
from pathlib import Path
from cogames.cogs_vs_clips.scenarios import make_game
from cogames.policy.simple import SimplePolicy
from cogames.train import train
from mettagrid import MettaGridEnv

# Configure environment
config = make_game(
    num_cogs=1,
    width=10,
    height=10,
    num_assemblers=0,
    num_chests=1,
    num_chargers=0,
    num_carbon_extractors=0,
    num_oxygen_extractors=0,
    num_germanium_extractors=0,
    num_silicon_extractors=0,
)

config.game.agent.initial_inventory = {"energy": 100, "heart": 3}
config.game.agent.resource_limits["heart"] = 5
config.game.objects["chest"].deposit_positions = ["N", "S", "E", "W"]
config.game.objects["chest"].withdrawal_positions = []
config.game.agent.rewards.stats = {"heart.lost": 1.0}

# Train
checkpoint_dir = Path("./checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

train(
    env_cfg=config,
    policy_class_path="cogames.policy.simple.SimplePolicy",
    device=torch.device("cpu"),
    initial_weights_path=None,
    num_steps=20_000,
    checkpoints_path=checkpoint_dir,
    seed=42,
    batch_size=512,
    minibatch_size=512,
    vector_num_envs=4,
    vector_num_workers=1,
)

print("Training complete!")
```

---

## Expected Training Times

| Tutorial | Steps | CPU Time | GPU Time |
|----------|-------|----------|----------|
| Tutorial 1 | 20k | ~2-3 min | ~30-60 sec |
| Tutorial 2 | 50k | ~5-7 min | ~1-2 min |
| Tutorial 3 (Single) | 100k | ~10-15 min | ~2-4 min |
| Tutorial 3 (Multi) | 200k | ~20-25 min | ~5-8 min |

**Total for all 3 tutorials:** ~35-50 minutes on CPU

---

## During Training: What You'll See

### PufferLib Output Example

```
Initializing PuffeRL trainer...
Creating vectorized environments (4 parallel envs)...
Initializing policy network...
Starting PPO training for 20000 steps...

Step: 512/20000 | FPS: 1250 | Return: 0.12 | Length: 150 | Loss: 0.45
Step: 1024/20000 | FPS: 1280 | Return: 0.25 | Length: 145 | Loss: 0.42
Step: 2048/20000 | FPS: 1300 | Return: 0.85 | Length: 120 | Loss: 0.38
...
Step: 19968/20000 | FPS: 1350 | Return: 2.85 | Length: 52 | Loss: 0.15

Training complete!
Checkpoints saved to: ./checkpoints/cogames.cogs_vs_clips/
Final checkpoint: ./checkpoints/cogames.cogs_vs_clips/model_20000.pt
```

### Key Metrics to Watch

1. **Return**: Should increase over time
   - Tutorial 1: 0 → ~3.0
   - Tutorial 2: 0 → ~2-5
   - Tutorial 3: 0 → ~1-5

2. **Episode Length**: Should decrease (more efficient)
   - Tutorial 1: ~200 → ~50 steps
   - Tutorial 2: ~250 → ~100 steps

3. **Loss**: Should decrease and stabilize
   - Starts high (~0.5-1.0)
   - Ends lower (~0.1-0.3)

4. **FPS**: Steps per second
   - CPU: ~1000-1500 FPS
   - GPU: ~3000-5000 FPS

---

## After Training: Checkpoint Files

Training creates checkpoint files:

```
tutorials/
├── checkpoints/
│   └── cogames.cogs_vs_clips/
│       ├── model_200.pt
│       ├── model_400.pt
│       ├── ...
│       └── model_20000.pt  ← Latest checkpoint
├── checkpoints_stage2/
│   └── cogames.cogs_vs_clips/
│       └── model_50000.pt
└── checkpoints_stage3_single/
    └── cogames.cogs_vs_clips/
        └── model_100000.pt
```

**Checkpoint sizes:** ~1-5 MB each

---

## Visualizing Results

After training, the notebooks will automatically:

1. **Load the trained policy**
2. **Run 100 evaluation episodes**
3. **Generate plots:**
   - Episode returns over time
   - Episode lengths over time
   - Success rate curves
   - Subtask completion (Tutorial 2+)
   - Multi-agent comparisons (Tutorial 3)

### Example Plot Output

You'll see matplotlib figures showing:
- Blue/red lines for learning curves
- Smoothed curves overlaid on raw data
- Grid lines for readability
- Legends and axis labels

---

## Watching Your Agent Play

To see your trained agent in action (GUI visualization):

```bash
cd /Users/bullm/Documents/GitHub/cogames

# For Tutorial 1 agent
cogames play \
    --policy simple \
    --policy-data tutorials/checkpoints/cogames.cogs_vs_clips/model_20000.pt
    
# You'll need to specify a game scenario
# Check available games:
cogames games

# Play with your trained agent
cogames play tutorial_assembler_simple \
    --policy simple \
    --policy-data tutorials/checkpoints/cogames.cogs_vs_clips/model_20000.pt
```

This opens a GUI window showing the agent playing in real-time.

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'cogames'"

**Solution:**
```bash
cd /Users/bullm/Documents/GitHub/cogames
pip install -e .
```

### Issue: "ModuleNotFoundError: No module named 'tutorial_viz'"

**Solution:** Make sure you're running from the `tutorials/` directory:
```bash
cd /Users/bullm/Documents/GitHub/cogames/tutorials
jupyter notebook
```

Or add to path:
```python
import sys
sys.path.append('/Users/bullm/Documents/GitHub/cogames/tutorials')
```

### Issue: Training is very slow

**Solutions:**
- Reduce `num_steps` for testing (e.g., 5000 instead of 20000)
- Reduce `vector_num_envs` if memory constrained
- Use GPU if available: `device=torch.device("cuda")`

### Issue: Training diverges (NaN errors)

**Solutions:**
- Reduce learning rate
- Check that environment config is valid
- Try starting from scratch (no transfer learning)

### Issue: Plots don't display

**In Jupyter:**
```python
%matplotlib inline
import matplotlib.pyplot as plt
```

**In terminal:**
```python
plt.savefig('my_plot.png')  # Save instead of show
```

### Issue: Out of memory

**Solutions:**
- Reduce `batch_size` (e.g., 256 instead of 512)
- Reduce `vector_num_envs` (e.g., 2 instead of 4)
- Close other programs

---

## Progressive Tutorial Path

### Start Here: Tutorial 1 (Required)
**File:** `01_goal_delivery.ipynb`  
**Time:** ~5 minutes to run  
**Purpose:** Learn basics, generate checkpoint for Tutorial 2

### Then: Tutorial 2 (Uses Tutorial 1 checkpoint)
**File:** `02_simple_assembly.ipynb`  
**Time:** ~10 minutes to run  
**Purpose:** Learn crafting, generate checkpoint for Tutorial 3

### Finally: Tutorial 3 (Uses Tutorial 2 checkpoint)
**File:** `03_full_cycle_multiagent.ipynb`  
**Time:** ~40 minutes to run  
**Purpose:** Full pipeline + multi-agent

**Total time:** ~55 minutes for complete series

---

## Quick Test Run

Want to verify everything works without full training?

**Reduce training steps for quick test:**

In each notebook, change the training cell:

```python
# Instead of:
num_steps=20_000

# Use:
num_steps=2_000  # 10x faster, just for testing
```

This lets you verify:
- ✅ Environment configures correctly
- ✅ Training starts without errors
- ✅ Checkpoints save properly
- ✅ Policy loads successfully
- ✅ Visualizations work

Then run full training later for actual results.

---

## Command Summary

```bash
# 1. Install
cd /Users/bullm/Documents/GitHub/cogames
pip install -e .

# 2. Start Jupyter
cd tutorials
jupyter lab

# 3. Open notebook
# Click: 01_goal_delivery.ipynb

# 4. Run all cells
# Menu: Cell → Run All

# 5. Wait ~3 minutes for training

# 6. View results in plots

# 7. Optional: visualize agent
cd ..
cogames play <game> --policy simple --policy-data tutorials/checkpoints/cogames.cogs_vs_clips/model_20000.pt
```

---

## Tips for Best Experience

1. **Use JupyterLab** (better UI than Jupyter Notebook)
2. **Run cell-by-cell first** to understand each step
3. **Read the markdown cells** for context
4. **Watch the training metrics** to see learning happen
5. **Save your checkpoints** for later experimentation
6. **Try both tutorials in one session** to see transfer learning benefits

---

## Next Steps After Completing Tutorials

1. **Modify hyperparameters** (learning rate, network size)
2. **Create custom scenarios** (different map sizes, resources)
3. **Try LSTM policy** for better performance
4. **Scale to more agents** (8, 16, 32)
5. **Design curriculum learning** (progressive difficulty)
6. **Submit to competitions** (test against other policies)

---

**Ready to start? Open `01_goal_delivery.ipynb` and press Shift+Enter!** 🚀

