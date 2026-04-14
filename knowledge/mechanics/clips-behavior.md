# Clips AI Behavior

Source: `clips.py:55-298`

## Configuration
- 4 ships, placed at map corners
- Each ship has an independent lane (ClosureQuery frontier)
- `greedy_expand_from_ships=True`, `max_search_radius=120`

## Timing (PREDICTABLE)
```
Scramble ticks: 100, 300, 500, 700, 900, ..., 9900
Align ticks:    100, 300, 500, 700, 900, ..., 9900
```
- Both scramble AND align fire on the same ticks
- Order: scramble first, then align (firstMatch handler)
- 200-tick safe window between each action

## Scramble Mechanics
- Target: cogs-tagged junctions within 15 cells of ship frontier
- Effect: `removeTagPrefix("net:")` → triggers `on_tag_remove` → `removeTag("team:cogs")`
- Then: `recomputeMaterializedQuery("net:")` recalculates ALL network closures

## CASCADE FAILURE (critical!)
When a junction is scrambled:
1. Its `net:cogs` tag is removed
2. `on_tag_remove` fires → removes `team:cogs` tag too
3. Network closure is recomputed from hub
4. Any junction that was ONLY connected through the scrambled one loses `net:cogs`
5. Those junctions then trigger `on_tag_remove` → lose `team:cogs` too
6. They become NEUTRAL (can be re-captured, but costs hearts + time)

**One scramble near the hub can disconnect an entire branch of the network.**

## Align Mechanics
- Target: neutral (no team tag) junctions within 15 cells of ship frontier
- Effect: adds `team:clips`, `net:clips`, ship lane tag
- Clips expand outward from corners along junction chains

## Ship Frontier
```python
ClosureQuery(
    source=ship_query,
    candidates=query("type:junction", hasTag(ship_map_name)),
    edge_filters=[maxDistance(15)],  # Note: 15, not 25
)
```
- Clips frontier uses 15-cell edges (tighter than cogs' 25-cell network closure)
- Ships only see their own lane's junctions

## Strategic Implications
1. **Center is initially safe** — ships start at corners
2. **Border junctions are high-risk** — near clips frontier = scramble target
3. **Predictable timing enables defense** — position scramblers before tick 100+200n
4. **Cascade makes defense critical** — losing one bridge junction costs many
5. **Redundant paths protect against cascade** — capture clusters, not chains
