# Competitor Analysis

## Leaderboard (April 14, 2026)

| Rank | Policy | VOR | Notes |
|------|--------|-----|-------|
| 1 | dinky:v27 | 27.21 | 123 matches, dominant |
| 2 | slanky:v129 | 23.92 | Many versions (v113-v147) |
| 3 | dinky:v26 | 23.24 | Previous version |
| 13 | Paz-Bot-9000:v17 | 19.94 | Third competitor |
| 26 | alpha.0:v922 | 17.73 | 439 matches, trained? |
| 224 | **Softy:v5** | **6.03** | Our policy |

## Gap Analysis: 6.03 vs 27.21 (4.5x)

### Likely Differentiators
1. **Trained neural policies**: dinky/slanky likely use RL (PufferLib + LSTM). Trained models learn implicit strategies that scripted policies can't replicate.
2. **Dynamic role allocation**: top policies may adapt roles based on game state
3. **Clips defense**: top policies almost certainly defend against cascade failure
4. **Efficient heart economy**: timing heart crafting vs consumption
5. **Map-aware expansion**: valuing specific junctions based on position, not just distance
6. **Energy exploitation**: staying in territory during night, aggressive during day

### Scripted Policy Ceiling
Based on game mechanics analysis, a well-optimized scripted policy should reach:
- **12-16 VOR**: with proper scrambler defense + network-aware targeting
- **16-20 VOR**: with energy exploitation + optimal expansion paths
- **20+ VOR**: likely requires trained component for mid-game adaptation

### Key Insight
slanky has 30+ versions on the leaderboard (v113-v147), suggesting rapid iteration.
Softy has only 5 versions. The improvement loop infrastructure is as important as any single change.

## What We Can Learn
- dinky's high match count (123) suggests consistent performance across seeds
- slanky's version spread (v113-v147 all scoring 16-24) suggests diminishing returns from iteration
- Paz-Bot's jump from v15 (15.68) to v17 (19.94) suggests a strategic pivot, not incremental improvement
