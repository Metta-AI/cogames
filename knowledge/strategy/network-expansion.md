# Network Expansion Strategy

## The Core Problem
Reward = net-connected junctions per tick. A disconnected junction = 0 reward.
The network is a closure from hub through team junctions within 25 cells of each other.
Clips can scramble any junction near their frontier, causing cascading disconnection.

## Network Topology
The cogs network is a **graph**, not a tree. Multiple paths between junctions provide redundancy.

### Cascade Risk Assessment
- **Bridge junction**: only one path from hub. If scrambled, everything beyond it disconnects.
- **Redundant junction**: multiple paths from hub. Scrambling one path doesn't disconnect.
- **Hub-adjacent** (within 25 cells): always connected directly to hub. Lowest risk.
- **Frontier junction**: near clips ships. Highest scramble risk.

### Optimal Expansion Pattern
1. **Phase 1 (ticks 0-500)**: Capture junctions within 25 cells of hub (hub-connected, immune to cascade from bridge loss)
2. **Phase 2 (ticks 500-2000)**: Expand outward in CLUSTERS, not chains. Each new capture should be within 25 cells of 2+ existing junctions
3. **Phase 3 (ticks 2000+)**: Push toward clips frontiers only with scrambler defense

### Current Implementation Gap
Softy's `nearest_alignable_junction` uses `frontier * 8.0 - dist` scoring:
- `frontier` = count of neutral junctions within 15 cells that would become newly alignable
- This encourages expansion but doesn't penalize bridge vulnerability
- **Missing**: cascade risk penalty for junctions that create single-path chains

### Proposed Enhancement
Add a **redundancy bonus** to junction scoring:
```
redundancy = count of cogs junctions within 25 cells of target
score = frontier * 8.0 + redundancy * 5.0 - dist
```
Junctions near multiple existing cogs junctions score higher because:
1. They create redundant network paths
2. They're within existing territory (HP safety)
3. They're harder for clips to isolate

## Alignment Range vs Network Closure

| Context | Radius | Notes |
|---------|--------|-------|
| Alignment action | 15 from net, 25 from hub | The bottleneck for capturing |
| Network scoring | 25 between any nodes | More generous connectivity |
| Clips scramble range | 15 from ship frontier | How far clips can reach |
| Territory heal | 10 from junction, 20 from hub | Safe zone for HP |

### Implication
A junction at 20 cells from the nearest cogs junction:
- CANNOT be aligned directly (15-cell limit)
- WOULD be net-connected if somehow captured (25-cell closure)
- Must build stepping stones: capture intermediate junctions within 15 cells

## Defense Zones
Categorize junctions by risk:
- **Safe** (>30 cells from any clips ship frontier): low priority for defense
- **Contested** (15-30 cells from clips frontier): needs monitoring
- **Threatened** (<15 cells from clips frontier): active scramble target, needs scrambler

## Clips Ship Positions
Ships start at corners of 88x88 map. Initial safe zone is the center.
As clips expand, their frontier pushes inward. By tick 2000, frontier is ~30 cells from corners.
