# Game Rules Reference

## Core Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Map size | 88x88 | machina_1.py:51 |
| Episode length | 10,000 steps | machina_1.py:203 |
| Cogs agents | 8 | mission config |
| Clips ships | 4 (at corners) | ship.py:17 |
| Observation | 13x13 egocentric | mettagrid default |
| Actions | move_N/S/E/W, noop | mission.py:35-37 |

## Scoring
- Reward = `(net-connected junctions - 1) / 10000` per tick per agent
- Only junctions with `net:cogs` tag count (see scoring.md)
- Junction captured at tick 100 earns 99% of its max value — **speed is everything**

## Resources & Hearts
- Hub starts with: 24 each element + 5 hearts
- Heart cost: 7 C + 7 O + 7 Ge + 7 Si = 28 total
- Initial budget: ~8 hearts (5 pre-made + 24/7 ≈ 3 craftable)

## Territory
- Junction AOE: 10-cell radius
- Hub AOE: 20-cell radius (2x)
- Inside: +100 HP/tick, +100 energy/tick
- Outside: -1 HP/tick, no energy bonus

## Alignment
- Requires: 1 heart + aligner gear
- Range: within 15 cells of net junction OR 25 cells of hub
- Network closure: 25-cell edges (more generous than alignment range)

## Clips Timing
- Scramble + align every 200 ticks starting tick 100
- Ticks: 100, 300, 500, 700, 900, ..., 9900
- Each ship operates independently in its own lane
- **Cascade risk**: scrambling one junction can disconnect entire branches (see clips-behavior.md)

## Death
- At 0 HP: all hearts and gear destroyed
- Must re-acquire gear at station + hearts at hub
