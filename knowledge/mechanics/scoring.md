# Scoring Deep-Dive

Source: `territory.py:30-41`, `machina_1.py:174-180`, verified against 20 episodes with 0.00 error

## Reward Formula (Verified)
```
reward = cogs_aligned_junction_held / 10000
```
Where `cogs_aligned_junction_held = sum over all 10,000 ticks of (net_connected_count - 1)`

Verified across 20 Softy v24 episodes with **0.00 error** (exact match to 2 decimal places).

## Network = ClosureQuery
```python
ClosureQuery(
    source=hub_query,                    # Start from hub
    candidates=query("type:junction", hasTag("team:cogs")),  # Team junctions
    edge_filters=[maxDistance(25)],       # 25-cell edges
)
```
- Network is a GRAPH CLOSURE from hub through team-tagged junctions
- Edge filter: any two nodes within 25 cells are connected
- Materialized as `net:cogs` tag, recomputed on every align/scramble

## Alignment vs Network
| Check | Radius | Purpose |
|-------|--------|---------|
| Alignment action | 15 cells from net, 25 from hub | Can I capture this junction? |
| Network closure | 25 cells between any nodes | Does this junction score? |

## Per-Agent Metrics Are Misleading

Reward is **per-team** — every agent gets identical reward regardless of individual contribution.

Per-agent metric correlations with reward:
| Metric | Correlation | Why |
|--------|-------------|-----|
| junction.aligned_by_agent | +0.002 | Essentially zero — individual alignment doesn't predict team reward |
| death | -0.718 | Reflects match difficulty, not causation |
| unique_visited | -0.830 | Harder matches = more exploration = lower team score |
| aligner.gained | -0.686 | More aligner pickups = more deaths = harder match |

**Exception**: For dinky (multi-agent), junction.aligned_by_agent has +0.832 correlation — because when dinky has more agents, they align more AND the team scores higher. This is agent-count mediated, not causal.

## What Actually Matters for Score

1. **Speed of capture**: junction at tick 100 earns 9900/10000 = 99% of max value
2. **Network connectivity**: disconnected junctions = 0 reward
3. **Network resilience**: redundant paths prevent cascade failure from clips scrambling
4. **Cascade failure**: scrambling a bridge junction disconnects everything beyond it
5. **Team coordination**: 8 agents, ~69 junctions — avoid fighting for same targets

## Theoretical Max
- Map: 69 capturable junctions, 10,000 ticks
- Max: 68.0 reward (all junctions from tick 0)
- Best observed: 50.56 (74.4% of max)
- Average (Softy v24): 40.0 (59% of max)
