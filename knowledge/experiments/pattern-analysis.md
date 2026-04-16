# Pattern Analysis — Deep Analysis Cycle 1
Updated: April 14, 2026

## Category Win Rates (126 experiments)
| Category | Tested | Won | Rate | Rule |
|----------|--------|-----|------|------|
| Bug fixes / capability | 10 | 10 | 100% | ONLY category that translates |
| Behavior tuning | 50+ | 0 | 0% | Never works in tournament |
| Deconfliction | 8 | 0 | 0% | Natural resolution > coordination |
| Hub-reducing | 6 | 0 | 0% | Hub = safety mechanism |
| Offense/scramble | 5 | 0 | 0% | Waste vs random opponents |
| Role distribution | 6 | 3 | 50% | Only helps severely under-resourced |
| Exploration changes | 6 | 1 | 17% | Reduction always hurts |

## Directional Rules (never violated)
1. Reducing hub visits ALWAYS hurts (P29, P72, P75, P45, P98)
2. Deconfliction ALWAYS hurts (P35, P36, P48, P55, P60, P93)
3. Timeout optimal at 60; below hurts (25=-53%, 35=-26%, 50=-6%), above is flat noise. Full 7-point sweep confirmed with p-values. Asymmetric curve — game forgives patience, punishes impatience.
4. Exploration reduction ALWAYS hurts (P16, P41, P45, P53, P98)
5. Offense vs random is ALWAYS wasteful (P1, P21, P27, P38, P44)
6. Miner count < 3 ALWAYS devastates (P1 -43%, P40 -38%)

## Near-Miss Candidates for Paired Retesting
| ID | Change | Delta | Base | Independence | Status |
|----|--------|-------|------|-------------|--------|
| ~~P86~~ | Deposit threshold 30→40 | +3.5% | v19 | Miner only | P156: REVERT (c2 regression, c4 inconsistent) |
| ~~P37~~ | Hub retreat stuck 5→3 | +3.6% | v13 | Stuck recovery | P157: NOISE (-4.1%, p=0.42) |
| P101 | ALIGN_NET_RADIUS 20→15 | +1.2% | v21 | Parameter, independent | Untested on v24+P154 |
| P49 | Sector radius 30→45 | +2.5% | v16 | Exploration, independent | Untested on v24+P154 |

**Dead candidates**: P33b noise, P86/P37 both noise on v24+P154

## Measurement Discovery (Cycle 59)
**c4+ has run-to-run non-determinism.** Base VOR varies ~6% between runs with same seed. Cause: likely Python hash randomization affecting dict/set iteration order in policy code. c2 is deterministic (fewer agents → fewer hash-dependent interactions).
- **Implication**: effects <10% on c4 require 20+ paired seeds to reliably detect.
- **P154 survived**: +18.7% is well above the noise floor.
- **P156 failed**: +9.3% in one run, +2.3% in another. Below noise floor.
- **Workaround**: could set PYTHONHASHSEED=0 in test scripts for determinism, but this wouldn't match tournament conditions.

## Negative Space (untried directions)
1. ~~**Timeout 50→60**~~: RESOLVED. Optimum at 60 locally but tournament-regresses (P127).
2. **Miner extractor selection**: Miners use _closest(). No scoring like aligners have.
3. ~~**Heartless wander bias**~~: RESOLVED. P151 catastrophic, P152 noise.
4. **Junction deposits are relay terminals**: Confirmed resources transfer to hub (not junction inventory). Current hub+visible-junction deposit behavior is optimal.
5. **P3 clips timing**: +7% on v4, never tested on current base. P4 interference is gone (reverted). Estimated max impact ~1.2% VOR (cascade events are rare). Below noise floor.

## Compound Test Priority
Near-misses exhausted (P86/P37 both noise on v24+P154). Remaining candidates P101/P49 are very small effects — unlikely to survive measurement noise.
