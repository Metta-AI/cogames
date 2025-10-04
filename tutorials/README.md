# CoGames Tutorials

This directory contains materials for creating a comprehensive "Hello World" tutorial series for CoGames with PufferLib integration.

## 📚 Contents

### Planning Documents

- **[PLANNING.md](PLANNING.md)** - Comprehensive technical specification
  - Complete codebase research and API documentation
  - Answers to all implementation questions
  - Phase-by-phase breakdown of tutorial requirements
  - Configuration examples and best practices
  - Docker environment specifications
  - Full visualization implementation guide

- **[STAGES.md](STAGES.md)** - Simplified stage configurations
  - Three progressive tutorial stages
  - Copy-paste ready configuration code
  - Helper functions for each stage
  - Visualization requirements per stage
  - Multi-agent scaling guidance

## 🎯 Tutorial Overview

The tutorial series consists of 3 progressive Jupyter notebooks:

### Notebook 1: Goal Delivery (10-20k steps, ~2-3 min)
**Objective:** Agent learns to deposit hearts already in inventory

**Key Concepts:**
- Basic navigation
- Sparse rewards
- Move action for interactions
- Observation space structure

**Visualizations:**
- Episode returns + lengths
- Success rate
- Value function heatmap
- Navigation progress

### Notebook 2: Simple Assembly (50k steps, ~5-7 min)
**Objective:** Agent learns to craft hearts then deposit them

**Key Concepts:**
- Multi-step planning
- Crafting via move action
- Resource management
- Subtask decomposition

**Visualizations:**
- All from Notebook 1 +
- Subtask completion tracking
- Resource inventory timeline
- Before/after GIF comparison

### Notebook 3: Full Cycle + Multi-Agent (100-200k steps, ~10-20 min)
**Objective:** Complete foraging → crafting → deposit cycle, then scale to 4 agents

**Key Concepts:**
- Resource extraction
- Complex multi-step tasks
- Multi-agent coordination
- Emergent cooperation

**Visualizations:**
- All from Notebook 2 +
- Resource collection efficiency
- Multi-agent return curves
- Agent specialization analysis
- Coordination metrics

## 🛠️ Implementation Status

- [x] Phase 0: Foundation research complete
- [x] Codebase analysis complete
- [x] Configuration system documented
- [x] Visualization specifications complete
- [ ] Dockerfile creation
- [ ] `tutorial_viz.py` module implementation
- [ ] Notebook 1 implementation
- [ ] Notebook 2 implementation
- [ ] Notebook 3 implementation
- [ ] Testing on Colab
- [ ] Sample outputs (GIFs, plots)

## 🚀 Quick Start for Implementation

### 1. Create Environment
```bash
# Build Docker container with all dependencies
docker build -t cogames-tutorial -f tutorials/Dockerfile .
docker run -p 8888:8888 cogames-tutorial
```

### 2. Create Visualization Module
```bash
# Implement tutorial_viz.py based on PLANNING.md specifications
touch tutorials/tutorial_viz.py
```

### 3. Create Notebooks
```bash
# Create three progressive notebooks
touch tutorials/01_goal_delivery.ipynb
touch tutorials/02_simple_assembly.ipynb
touch tutorials/03_full_cycle_multiagent.ipynb
```

## 📖 Key Technical Details

### Environment
- **Framework:** MettagGrid (Gymnasium-compatible)
- **Training:** PufferLib with PuffeRL trainer
- **Policies:** SimplePolicy (feedforward) or LSTMPolicy (recurrent)

### Configuration
- **Reward:** `heart.lost` stat (agent-specific, +1 per heart deposited)
- **Chest:** Accept deposits from all sides, no withdrawals
- **Actions:** Move-based interactions (no separate deposit/craft actions)

### Training Hyperparameters
**SimplePolicy:**
- Learning rate: 0.015
- Optimizer: Muon
- Batch size: 4096

**LSTMPolicy:**
- Learning rate: 0.0003
- Optimizer: Adam
- Batch size: 4096

## 📊 Expected Results

| Stage | Success Rate | Avg Return | Episode Length | Key Milestone |
|-------|-------------|------------|----------------|---------------|
| Stage 1 | 60%+ | 2.0+ | ~50 steps | Navigate to chest |
| Stage 2 | 60%+ | 2.0+ | ~80 steps | Craft then deposit |
| Stage 3 | 50%+ | 1.5+ | ~120 steps | Complete cycle |
| Multi-Agent | 50%+ | 6.0+ team | ~100 steps | Coordination emerges |

## 🔗 Related Documentation

- **Main README:** [/README.md](../README.md) - CoGames overview
- **Scenarios:** [/src/cogames/cogs_vs_clips/scenarios.py](../src/cogames/cogs_vs_clips/scenarios.py) - Game configurations
- **Stations:** [/src/cogames/cogs_vs_clips/stations.py](../src/cogames/cogs_vs_clips/stations.py) - Object definitions
- **MISSION:** [/MISSION.md](../MISSION.md) - Game lore and mechanics

## 💡 Design Principles

1. **Progressive Complexity:** Each stage builds on previous skills
2. **Copy-Paste Ready:** All code examples work out of the box
3. **Visual Feedback:** Comprehensive plots prove learning is happening
4. **Educational:** Clear explanations of RL concepts
5. **Practical:** Realistic training times for Colab/local execution

## 📝 Notes

- All configurations use `heart.lost` for agent-specific rewards
- Chests configured for easy learning (all sides, no withdrawals)
- Recipes simplified to `["Any"]` formation for accessibility
- Episode lengths tracked alongside returns for efficiency analysis
- Value function heatmaps show spatial understanding

## 🤝 Contributing

When implementing notebooks:
1. Follow the stage configurations in STAGES.md
2. Include all visualizations specified
3. Add interpretation guides for each plot
4. Test on clean environment (Docker/Colab)
5. Ensure training completes in specified time ranges

---

**Version:** 1.1  
**Last Updated:** October 4, 2025  
**Status:** Planning Complete - Ready for Implementation

