# Baseline Game Profile — v24
Updated: April 14, 2026. Seeds 42-51, 10 per scenario.

## Per-Scenario Performance
| Scenario | Weight | Mean Reward | StDev | CV | Min | Max |
|----------|--------|-------------|-------|----|-----|-----|
| c2 | 32% | 8.05 | 5.50 | 68.3% | 2.34 | 19.57 |
| c4 | 27% | 25.75 | 5.60 | 21.8% | 17.35 | 35.48 |
| c6 | 33% | 32.87 | 2.57 | 7.8% | 29.10 | 37.08 |
| c8 | 8% | 38.45 | 3.60 | 9.4% | 29.93 | 42.30 |

Tournament-weighted mean: 23.45

## Per-Agent Metrics (averaged across seeds)
| Scenario | Deaths | Hearts | Alignments | UniqueCells | FailedMoves |
|----------|--------|--------|------------|-------------|-------------|
| c2 | 4.2±7.0 | 21.6±11.1 | 4.4±6.3 | 637±132 | 229±652 |
| c4 | 30.4±16.4 | 35.7±9.9 | 19.4±8.5 | 935±111 | 497±403 |
| c6 | 43.9±12.1 | 36.9±6.2 | 23.9±5.5 | 1007±112 | 545±336 |
| c8 | 27.0±13.1 | 30.5±8.8 | 19.5±8.6 | 1131±123 | 1233±764 |

## Key Findings
1. **c2 is the high-variance scenario** (CV=68.3%). With 32% tournament weight, stabilizing c2 is highest leverage.
2. c2 deaths are LOW (4.2) because only 2 agents → less crowding. But alignments also low (4.4) = under-resourced.
3. c6 is our most stable scenario (CV=7.8%) — best for paired testing signal detection.
4. c8 failed moves are very high (1233) — agent-agent collisions with 8 agents in tight spaces.
5. Hearts per agent peaks at c6 (36.9) not c8 (30.5) — more agents means more competition for hearts.
6. Deaths peak at c6 (43.9) — 6 agents creating more contested zones.

## Improvement Opportunities
- **c2**: High variance suggests some seeds have fundamentally different map layouts. 1M/1A (P102) is correct but execution varies wildly. Could benefit from smarter 2-agent coordination.
- **c4**: 30 deaths/agent is high. P37 (faster hub retreat) could help here more than c6.
- **c8**: 1233 failed moves = 12% failure rate. Agent collision avoidance could help.
