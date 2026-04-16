# Retrospective — April 14, 2026 (Cycle 20)

## Cycle Range
Cycles 1 through 20, hypotheses P1 through P59 (~40+ tested)

## Tournament Breakthrough
**v17 = 12.67 VOR, rank 58** (up from v11's 11.33, rank 75). P15 junction timeout fix translated to tournament (+11.8%). Confirms: bug fixes translate.

## Pattern Summary

| Category | Tested | Won | Win Rate | Best | Worst |
|----------|--------|-----|----------|------|-------|
| Economy | 8 | 1 | 12% | P12 +46% | P56 -18% |
| Bug fixes | 2 | 2 | 100% | P18 +6% tourney, P15 +12% tourney | — |
| Pathfinding | 5 | 0 | 0% | P30 +1.7% | P31 -66% |
| Defense/Offense | 6 | 0 | 0% | P27 +0.9% | P1 -43% |
| Scoring/Targeting | 9 | 0 | 0% | P19 +4.9% | P32 -8.3% |
| Behavior tuning | 3 | 0 | 0% | P11 +21% local→-28% tourney | — |
| Exploration | 6 | 0 | 0% | P49 +2.5% | P31 -66% |
| Coordination | 5 | 0 | 0% | P37 +3.6% | P35 -7.3% |

**What works**: Bug fixes (2/2 = 100%) and economy fixes (1/8). ALL three tournament wins were bugs.
**What fails**: Everything else. 35+ experiments, 0% win rate.
**Rule**: "Capability > behavior tuning" — STRONGLY CONFIRMED. Extended to: "Bug fix > capability > behavior tuning"

## Re-evaluation Candidates

### P3: Clips timing exploitation (RETRY)
- **Original**: +7.0% solo on v4 base. Blocked by P4 interference. P4 now REVERTED.
- **Recommendation**: Retry on v17. The interference is gone. But it's behavior tuning (risky category).

### P9 as filter: Connectivity-only targeting (CONSIDER)
- **Original**: +2.6% as bonus (too weak). Retry idea: FILTER junctions not adjacent to connected network.
- **Risk**: P32 (align-range filter) regressed -8.3%. But P32 was range-based, this is connectivity-based.

### ALL OTHERS: SKIP
- P13 already retested on v17 (-5.7%). P59 already tested (-1.6%). All coordination/scoring/exploration changes fail.

## Strategy Assessment

### Trajectory: PLATEAUING
- v2→v7: +8.36 VOR in 5 versions (economy fixes, massive jumps)
- v7→v17: +2.15 VOR in 10 versions (bug fixes, diminishing returns)
- Last 8 cycles: ALL failed. 0/16 hypotheses passed.
- **Hit rate**: 3/40+ = ~7%. We're mining depleted territory.

### Gap Analysis
- **Current**: 12.67 VOR (rank 58 of 355+)
- **Top 25**: 17.85 VOR. Need +41%.
- **#1 (dinky:v27)**: 27.25. Need +115%.
- **Competitors**: slanky has 25+ versions (19-24 VOR = massive iteration). dinky jumped from 23→27 in one version (qualitative breakthrough).

### What would close the gap?
Micro-optimizations CANNOT close a 2.15x gap. Need qualitative leaps:

1. **Junction discovery speed** — Finding all ~69 junctions faster = more alignment time. Currently relying on random wandering.
2. **Multi-agent spatial coordination** — 5 aligners should divide the map, not cluster. Each aligner "owns" a region.
3. **Cascade defense** — Protecting network from clips scrambling (but 4/4 defense attempts failed).
4. **Pathfinding** — Walls cause massive detours. Better routing = more alignment cycles per game.

## Updated Priority Queue

### High Priority (new capability class)
1. **P60: Map sector division for aligners** — Each aligner is assigned 1/5 of the map. Only targets junctions in their sector. Reduces travel, prevents clustering, improves coverage. TYPE: coordination capability.
2. **P61: Heartless junction pre-positioning** — Heartless aligners navigate TOWARD their next target junction (picked from known junctions) instead of random wandering. When heart arrives, they're already close. TYPE: efficiency capability.
3. **P62: Hub heart queue coordination** — Track how many aligners are headed to hub. If 2+ are en route, redirect excess to explore. TYPE: coordination bug fix.

### Medium Priority
4. P3 retry (clips timing, risky category)
5. P63: Miner route memory (remember extractor locations between trips)
6. P9 retry as filter (connectivity-only targeting)

### Low Priority
7. P64: Network-distance timeout (BFS instead of Manhattan for junction timeout)
