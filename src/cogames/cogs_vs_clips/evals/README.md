# CoGames Evaluation Missions

This directory contains evaluation missions for testing CoGames agents:

1. **Diagnostic Missions** - Fixed-map missions testing specific skills in controlled environments
2. **CogsGuard Eval Missions** - Fixed-map missions from curated evaluation arenas
3. **Integrated Eval Missions** - Procedural missions combining multiple challenges

---

## Diagnostic Missions

**Location:** `diagnostic_evals.py` **Access:** `cogames play --mission evals.diagnostic_*` **Map Type:** Fixed ASCII
maps (deterministic layouts)

Diagnostic missions test specific skills in isolation with controlled, repeatable environments.

### Available Diagnostic Missions

#### Navigation & Delivery

- `diagnostic_chest_navigation1/2/3` - Navigate to chest and deposit hearts (varying difficulty)
- `diagnostic_chest_deposit_near` - Chest nearby, test deposit mechanics
- `diagnostic_chest_deposit_search` - Find chest through exploration

#### Energy Management

- `diagnostic_charge_up` - Test charging mechanics and energy management

#### Memory

- `diagnostic_memory` - Test memory and state tracking over longer distances

#### Hard Variants

Most diagnostic missions have `_hard` variants with increased difficulty and longer time limits (e.g.,
`diagnostic_chest_navigation1_hard`, `diagnostic_charge_up_hard`, `diagnostic_memory_hard`).

### Playing Diagnostic Missions

```bash
# Basic diagnostic
uv run cogames play --mission evals.diagnostic_chest_navigation1 --cogs 1

# Hard variant
uv run cogames play --mission evals.diagnostic_charge_up_hard --cogs 1

# With policy
uv run cogames play --mission evals.diagnostic_chest_deposit_search -p baseline --cogs 1
```

---

## CogsGuard Eval Missions

**Location:** `cogsguard_evals.py` **Access:** `cogames play --mission cogsguard_evals.<name>` **Map Type:** Fixed ASCII
maps (curated layouts)

CogsGuard eval missions use hand-crafted maps designed to test specific scenarios at various scales.

### Available CogsGuard Eval Missions

- `eval_balanced_spread` - Balanced resource spread
- `eval_clip_oxygen` - Clips pressure with oxygen constraints
- `eval_collect_resources` / `_medium` / `_hard` - Resource collection at increasing difficulty
- `eval_divide_and_conquer` - Multi-zone coordination
- `eval_energy_starved` - Low energy environments
- `eval_multi_coordinated_collect_hard` - Complex multi-agent coordination
- `eval_oxygen_bottleneck` - Oxygen-limited scenarios
- `eval_single_use_world` - Single-use extractors
- `extractor_hub_30x30` / `50x50` / `70x70` / `80x80` / `100x100` - Hub-centric maps at various scales

---

## Integrated Eval Missions

**Location:** `integrated_evals.py` **Access:** `cogames play --mission hello_world.<name>` **Map Type:** Procedural
generation (MachinaArena)

### Available Integrated Missions

#### energy_starved

**Challenge:** Low energy regen requires careful energy management.

**Variants Applied:**

- DarkSide (reduced energy regen)

```bash
uv run cogames play --mission hello_world.energy_starved --cogs 2
```

---

## Design Philosophy

### Diagnostic Missions

- **Focused**: Each tests a specific skill in isolation
- **Deterministic**: Fixed maps ensure reproducible results
- **Minimal**: Small maps, simple layouts, clear objectives
- **Scalable**: Work well with 1-4 agents

### CogsGuard Eval Missions

- **Curated**: Hand-designed maps for specific scenarios
- **Fixed agent counts**: Each map defines its own agent count based on spawn pads

### Integrated Missions

- **Procedural**: Different map each run for generalization
- **Composable**: Built from reusable variants

### Evaluation Best Practices

1. Use diagnostic missions to identify specific skill deficits
2. Use CogsGuard eval missions for standardized benchmarking
3. Use integrated missions to evaluate overall performance
4. Run multiple seeds to account for procedural variation
5. Compare against scripted baselines for context
