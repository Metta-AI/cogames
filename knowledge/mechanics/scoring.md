# Scoring Deep-Dive

Source: `territory.py:30-41`, `machina_1.py:174-180`

## VOR Formula
```
reward_per_tick = (count(net:{team}) - root_count) / max_steps
```
- `root_count` = 1 for cogs (hub), 4 for clips (ships)
- `max_steps` = 10,000
- Summed across all 10,000 ticks for total episode reward
- VOR = reward vs replacement (random) policy baseline

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

- Alignment range (15) is the bottleneck for CAPTURING
- Network closure (25) is more generous for CONNECTIVITY
- A junction captured via hub (within 25) can bridge to junctions 25 cells away in the closure

## Key Implications
1. **Disconnected junctions = 0 reward** — the ONLY thing that matters is net-connected count
2. **Speed of capture**: junction at tick 100 earns 9900/10000 = 99% of max value
3. **Network resilience**: redundant paths (multiple junctions within 25 cells) protect against cascade
4. **Cascade failure**: scrambling a bridge junction disconnects everything beyond it
