# Scripted Agent Policies

Teaching-friendly scripted agents for CoGames evaluation and ablation studies, plus a tiny demo policy and the CogsGuard
team-play scripted policy.

## Overview

This package provides the CogsGuard team-play policy, two progressively capable scripted agents, and one tiny demo
policy:

1. **CogsGuard** - Vibe-based multi-role policy for the CogsGuard arena
2. **BaselineAgent** - Core functionality for legacy heart-production game: exploration, resource gathering, heart
   assembly
3. **UnclippingAgent** - Extends BaselineAgent with extractor unclipping capability (legacy game)

## Architecture

### File Structure

```
scripted_agent/
├── cogsguard/                   # CogsGuard scripted policy (vibe-based roles)
├── baseline_agent.py            # Base agent + BaselinePolicy wrapper
├── unclipping_agent.py          # Unclipping extension + UnclippingPolicy wrapper
├── demo_policy.py               # Tiny demo policy (short name: tiny_baseline)
└── pathfinding.py               # Pathfinding utilities (shared)
```

Each agent file contains:

- Agent class with core logic and state management
- Policy wrapper classes at the bottom for CLI integration

### Design Philosophy

These agents are designed for **ablation studies** and **baseline evaluation**:

- Simple, readable implementations
- Clear separation of capabilities
- Minimal dependencies

## Agents

### 1. CogsGuard Scripted Agent

CogsGuard is the team-play focus for scripted policies. Agents are controlled by **vibes** that map to roles and gear
acquisition.

**Vibes**:

| Vibe        | Behavior                                 |
| ----------- | ---------------------------------------- |
| `default`   | Idle (noop)                              |
| `heart`     | Idle (noop)                              |
| `gear`      | Smart role selection                     |
| `miner`     | Gather and deposit resources             |
| `scout`     | Explore and discover structures          |
| `aligner`   | Align neutral supply depots to cogs      |
| `scrambler` | Scramble clips-aligned depots to neutral |

**Gear costs** (paid from cogs commons):

| Gear      | Cost                                       | Bonus                |
| --------- | ------------------------------------------ | -------------------- |
| Miner     | 3 carbon, 1 oxygen, 1 germanium, 1 silicon | +40 cargo            |
| Scout     | 1 carbon, 1 oxygen, 1 germanium, 3 silicon | +100 energy, +400 HP |
| Aligner   | 3 carbon, 1 oxygen, 1 germanium, 1 silicon | +20 influence        |
| Scrambler | 1 carbon, 3 oxygen, 1 germanium, 1 silicon | +200 HP              |

**Supply depots** start clips-aligned. Scramblers neutralize them; aligners convert neutral depots to cogs for AOE
energy regen.

**Usage**:

```bash
# Default role distribution (1 scrambler, 4 miners, rest smart-gear)
./tools/run.py recipes.experiment.cogsguard.play policy_uri=metta://policy/role

# Custom role counts
./tools/run.py recipes.experiment.cogsguard.play \
    policy_uri="metta://policy/role?miner=3&scout=2&aligner=2&scrambler=3"
```

**Full documentation**: `packages/cogames-agents/src/cogames_agents/policy/scripted_agent/cogsguard/README.md`

### 2. BaselineAgent

**Purpose**: Minimal working agent for single/multi-agent missions

**Capabilities**:

- ✅ Visual discovery (explores to find stations and extractors)
- ✅ Resource gathering (navigates to extractors, handles cooldowns)
- ✅ Heart assembly (deposits resources at hub)
- ✅ Heart delivery (brings hearts to chest)
- ✅ Energy management (recharges when low)
- ✅ Extractor tracking (remembers positions, cooldowns, remaining uses)
- ✅ Agent occupancy avoidance (multi-agent collision avoidance via pathfinding)

**Limitations**:

- ❌ No unclipping support (can't handle clipped extractors)
- ⚠️ Multi-agent coordination is basic (agents avoid each other but don't explicitly coordinate)

**Usage**:

```python
from cogames_agents.policy.scripted_agent.baseline_agent import BaselinePolicy
from mettagrid import MettaGridEnv

env = MettaGridEnv(env_config)
policy = BaselinePolicy(env)

obs, info = env.reset()
policy.reset(obs, info)

agent = policy.agent_policy(0)
action = agent.step(obs[0])
```

**CLI**:

```bash
# Single agent
uv run cogames play --mission evals.diagnostic_radial -p baseline --cogs 1

# Multi-agent
uv run cogames play --mission evals.diagnostic_radial -p baseline --cogs 4
```

### 3. UnclippingAgent

**Purpose**: Handle missions with clipped extractors

**Extends BaselineAgent with**:

- ✅ Clipped extractor detection
- ✅ Unclip item crafting
- ✅ Extractor restoration
- ✅ Resource deficit management (ensures enough resources for both unclipping and hearts)

**Unclip Item Mapping**: | Clipped Resource | Unclip Item | Crafted From | Glyph |
|-----------------|-------------|--------------|-------| | Oxygen | decoder | carbon | gear | | Carbon | modulator |
oxygen | gear | | Germanium | resonator | silicon | gear | | Silicon | scrambler | germanium | gear |

**Workflow**:

1. Detects clipped extractor blocking progress
2. Gathers craft resource (e.g., carbon for decoder)
3. Changes glyph to "gear"
4. Crafts unclip item at hub
5. Navigates to clipped extractor
6. Uses item to unclip
7. Resumes normal gathering

**Usage**:

```python
from cogames_agents.policy.scripted_agent.unclipping_agent import UnclippingPolicy

policy = UnclippingPolicy(env)
# ... same as BaselinePolicy
```

### 4. TinyBaseline (demo policy)

**Purpose**: Minimal, readable demo policy used for quick experiments.

**Short name**: `tiny_baseline` (defined in `demo_policy.py`).

## StarterAgent

**Purpose**: Intro-friendly agent that mirrors the high-level flow described in docs.

**Decision tree**:

1. Low energy → go recharge
2. Carrying a heart → deliver it
3. Have all recipe inputs → assemble
4. Otherwise → gather missing resources in a fixed order (carbon, oxygen, germanium, silicon)

**Why it exists**: Shows the simplest possible if/else controller that still completes missions, ideal for external
readers who want a tiny, readable starting point before diving into the full Baseline/Unclipping logic.

**Location**: The starter policy lives in the core `cogames` package at `cogames.policy.starter_agent` so it is always
available without installing `cogames-agents`.

## Shared Components

### Phase System

All agents use a phase-based state machine:

```python
class Phase(Enum):
    GATHER = "gather"          # Collecting resources
    ASSEMBLE = "assemble"      # Crafting heart at hub
    DELIVER = "deliver"        # Bringing heart to chest
    RECHARGE = "recharge"      # Restoring energy
    CRAFT_UNCLIP = "craft_unclip"  # UnclippingAgent only
    UNCLIP = "unclip"          # UnclippingAgent only
```

### Navigation

Shared `pathfinding.py` module provides:

- **BFS pathfinding** with occupancy grid
- **Greedy fallback** when path blocked
- **Adjacent positioning** for station interactions
- **Agent occupancy avoidance** for multi-agent scenarios

### Observation Parsing

Agents parse egocentric observations (11×11 grid) to detect:

- Stations (hub, chest, junction, extractors)
- Other agents
- Walls and obstacles
- Agent state (resources, energy, inventory)

### Extractor Tracking

```python
@dataclass
class ExtractorInfo:
    position: tuple[int, int]
    resource_type: str  # "carbon", "oxygen", "germanium", "silicon"
    remaining_uses: int
    clipped: bool       # For UnclippingAgent
```

## Testing

### Quick Tests

#### BaselineAgent (Diagnostic Missions)

```bash
# Basic diagnostic (single agent)
uv run cogames play --mission evals.diagnostic_radial -p baseline --cogs 1 --steps 1000

# Chest navigation
uv run cogames play --mission evals.diagnostic_chest_navigation1 -p baseline --cogs 1 --steps 1000

# Resource extraction
uv run cogames play --mission evals.diagnostic_extract_missing_oxygen -p baseline --cogs 1 --steps 1000

# Hard version
uv run cogames play --mission evals.diagnostic_radial_hard -p baseline --cogs 1 --steps 2000

# Multi-agent (2, 4 agents)
uv run cogames play --mission evals.diagnostic_radial -p baseline --cogs 2 --steps 1500
uv run cogames play --mission evals.diagnostic_radial -p baseline --cogs 4 --steps 2000

# Assembly test
uv run cogames play --mission evals.diagnostic_assemble_seeded_search -p baseline --cogs 1 --steps 1000
```

### Comprehensive Evaluation

```bash
# Run full evaluation suite
uv run python packages/cogames/scripts/run_evaluation.py

# Evaluate specific agent
uv run python packages/cogames/scripts/run_evaluation.py --policy baseline
uv run python packages/cogames/scripts/run_evaluation.py --policy ladybug
```

## Evaluation Results

**Summary**:

- **BaselineAgent**: Works best for non-clipped missions with straightforward resource gathering
- **UnclippingAgent**: Best overall performance, handles clipping scenarios well

## Extending

### Adding New Agent Capabilities

To create a new agent variant:

1. **Create new file** (e.g., `my_agent.py`)
2. **Extend base class**:

```python
from .baseline_agent import BaselineAgent, SimpleAgentState

class MyAgent(BaselineAgent):
    def _update_phase(self, s: SimpleAgentState) -> None:
        # Add custom phase logic
        super()._update_phase(s)

    def _execute_phase(self, s: SimpleAgentState) -> int:
        # Add custom phase execution
        return super()._execute_phase(s)
```

3. **Add policy wrapper** at bottom of file:

```python
class MyAgentPolicy:
    """Per-agent policy wrapper."""
    def __init__(self, impl: MyAgent, agent_id: int):
        self._impl = impl
        self._agent_id = agent_id

    def step(self, obs) -> int:
        return self._impl.step(self._agent_id, obs)

class MyPolicy:
    """Policy wrapper for MyAgent."""
    def __init__(self, simulation=None):
        self._simulation = simulation
        self._impl = None
        self._agent_policies = {}

    def reset(self, obs, info):
        # Initialize impl from simulation
        pass

    def agent_policy(self, agent_id: int):
        # Return per-agent policy
        pass
```

4. **Register in `__init__.py`**:

```python
from cogames_agents.policy.scripted_agent.my_agent import MyPolicy

__all__ = [..., "MyPolicy"]
```

### Resource Management

Agents track deficits and gather in priority order:

1. Germanium (5 needed, highest priority)
2. Silicon (50 needed)
3. Carbon (20 needed)
4. Oxygen (20 needed)

UnclippingAgent adds special logic:

- Ensures enough craft resource for both unclipping AND hearts
- Prevents resource deficits when crafting decoders

## Future Work

- [ ] Dynamic heart recipe detection
- [ ] Charger clipping strategies
- [ ] Clip spread handling
- [ ] Learned extractor efficiency
- [ ] Advanced multi-agent coordination (task assignment, resource reservation)
- [ ] Frontier-based exploration improvements
