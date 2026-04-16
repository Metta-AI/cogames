# Experiment Log

### Cycle 62: P159/P160/P161 — three parallel hypotheses, all NOISE (April 15)
**Game engine distance discovery**: Confirmed game uses L2² (dr²+dc² ≤ r²), NOT Manhattan. Role IDs always 0..N-1.

- **P159** (L2 distance fix + ALIGN_NET_RADIUS 20→15): Fixed all distance calculations from Manhattan to L2², corrected radius to match game. c2 -2.6% p=0.52, c6 -1.3% p=0.67. **NOISE.** Bug is real but junctions rarely land in the Manhattan/L2 mismatch zone on machina_1.
- **P160** (failed_junctions blacklist reset every 2000 ticks): Prevents permanent blacklisting after death+respawn. c2 0% (identical on 9/10 seeds), c6 +2.1% p=0.33. **NOISE.** Marginal positive trend but failed threshold.
- **P161** (claim release when aligner loses heart): Release stale claims so other aligners can target. c2 0% (identical), c6 -0.2% p=0.92. **NOISE.**

**Meta-insight**: All remaining bugs have practical effects below c6's ~6% noise floor. Need fundamentally new capability or 20+ seed testing for <6% effects.

### Cycle 61: P158 ALIGN_NET_RADIUS 20→15 — NOISE (April 15)
- **Method**: 10 seeds c2 paired same-seed (c2 is deterministic — perfect precision).
- **Hypothesis**: Match game's actual alignment range (15 cells from net). Current radius 20 targets out-of-range junctions, wasting 50-tick timeout.
- **Result**: -1.7%, p=0.479, Cohen's d=-0.234. **NOISE.** 6 seeds differ by ±0.01-0.04.
- **Why no effect**: Junctions at 16-20 cells from network are rare. The radius mismatch almost never causes wasted targeting. Confirms P87 (0%) and P101 (+1.2%) results.
- Category: Parameter tuning. Not significant.
- **UPLOADED v28** after this test. P154 is the only win. Near-misses exhausted. Small effects below noise floor.

### Cycle 60: P157 stuck threshold 5→3 — NOISE (April 15)
- **Method**: 10 seeds c4 paired same-seed.
- **Hypothesis**: Faster stuck recovery (hub reset at stuck_count > 3 instead of > 5). Saves ~4 wasted ticks per stuck event.
- **Result**: -4.1%, p=0.424, Cohen's d=-0.265. **NOISE.**
- **Why failed**: Lowering stuck threshold increases unnecessary hub retreats. Agents that are briefly stuck (2-3 ticks) often resolve naturally. Threshold 5 is the right balance.
- Category: Parameter tuning. Not significant.

### Cycle 59: P156 deposit threshold 30→40 — REVERT (April 15)
- **Method**: Phase 1: 10 seeds c4. Phase 2: 5 seeds each c2+c6+c8. P156b: adaptive (30 for c2, 40 for c4+).
- **Hypothesis**: Miners deposit less often → more mining time → more resources.
- **Phase 1 (flat 40)**: +9.3%, p=0.064. Promising.
- **Phase 2**: c2 -8.0% regression! c6 +7.2% (but c6 non-deterministic). c8 noise.
- **P156b (adaptive threshold)**: c4 +2.3%, p=0.72. NOISE. Inconsistent with Phase 1.
- **Root cause**: c4 base values vary between runs (non-determinism). Phase 1's +9.3% was partly noise. c2 regression is real — single miner with higher threshold means longer heart supply gaps for the lone aligner.
- **Key finding**: c4 has run-to-run non-determinism (base VOR varies ~6% between runs with same seed). Likely Python hash randomization affecting dict/set iteration order. This means effects <10% require 20+ seeds to detect reliably.
- Category: Parameter tuning. REVERTED.

### Cycle 58: P154 c4 role 2M/2A → 1M/3A — WIN (April 15)
- **Method**: Phase 1: 10 seeds c4 paired. Phase 2: 5 seeds each c2+c6+c8. Phase 3: 10 more c4 seeds (20 total).
- **Hypothesis**: Hearts over-produced at 23% utilization. c4 currently 2M/2A (P103). Reducing to 1M/3A gives +50% alignment capacity. Hub starts with 8 hearts (enough for initial burst). 1 miner produces ~60 hearts in 10K ticks — adequate for 3 aligners.
- **Category**: Role distribution (now 4/7 win rate: P102 +82% c2, P103 +9.6% c4, P154 +18.7% c4).
- **Phase 1 (c4, 10 seeds)**: +18.2%, p=0.0176, Cohen's d=0.917. 9/10 positive.
- **Phase 2 (multi-scenario)**: c2 identical (same code path). c6/c8 noise (same code path; c6/c8 non-deterministic at higher agent counts).
- **Phase 3 (c4, 20 seeds total)**: +18.7%, p=0.0025, Cohen's d=0.779. 15/20 positive.
- **Tournament-weighted**: +5.05% (c4 at 27% weight).
- **Applied to softy.py.** Breaks 4-cycle fail streak (P150-P153).
- **Discovery**: c6/c8 have inherent non-determinism even with same seed (base c6 seed 31 gave 1.04 then 1.14 on rerun). c2 is deterministic. Paired testing still valid but variance is higher for c6+.
- Only affects c4 (team_size==4). c2/c6/c8 unchanged.

### Cycle 57: P153 BFS passability fix — WRONG HYPOTHESIS (April 15)
- **Hypothesis**: Junctions/extractors/hub are GridObjectConfig (vs WallConfig for walls). Only walls+agents should be blocked in BFS — junctions passable.
- **Investigation**: Game engine uses "relocate-to-empty, on-use" move chain. ALL tagged cells are impassable (occupied = not empty). WallConfig vs GridObjectConfig distinction is about handler complexity, not passability. Confirmed via source: `MoveActionConfig` default chain relocates to EMPTY cells only.
- **Result**: VOR dropped to ~0.00 on c6. Agents tried walking through everything, all moves failed.
- **Confirmed**: `blocked = set(tags)` in BFS is CORRECT. All tagged cells block movement.
- Category: Bug fix (wrong hypothesis).

### Cycle 56: P152 Heartless Wander Sector Bias — NOISE (April 15)
- **Method**: 5 seeds each c2+c4+c6+c8. Paired same-seed.
- **Results**:
  - c2: +1.0%, p=0.907. Noise.
  - c4: -2.1%, p=0.797. Noise.
  - c6: +0.2%, p=0.972. Noise.
  - c8: 4 valid pairs (seed 25 corrupt), insufficient.
- **Tournament-weighted**: ~0%. **NOISE.**
- **Hypothesis**: Lighter P151 — bias wander sector toward best known junction without replacing exploration. Sets explore_dir_idx to sector closest to nearest alignable/enemy junction.
- **Why failed**: Wander exploration is already effective. Biasing toward known junctions doesn't meaningfully speed up heart acquisition or alignment. The sector system already provides good coverage.
- Category: Capability improvement (aligner efficiency). Not significant.

### Cycle 55: P151 Heartless Pre-Positioning — REVERT (catastrophic c2) (April 15)
- **Method**: Phase 1: 10 seeds c6. Phase 2: 5 seeds each c2+c4+c6+c8.
- **Phase 1 (c6)**: +6.3%, p=0.184, Cohen's d=+0.455. Marginal positive. Passed screening.
- **Phase 2 (multi-scenario)**:
  - c2: **-100.0%, p=0.003, VOR=0.000 ALL seeds.** CATASTROPHIC.
  - c4: +4.2%, p=0.862. Noise.
  - c6: +0.8%, p=0.886. Noise.
  - c8: -0.3%, p=0.891. Noise.
- **Tournament-weighted**: -29.6%. **REVERT.**
- **Root cause**: Heartless aligner navigates directly TO junction → bumps it → impassable → stuck loop → hub retreat → repeat. In c2 with 1 aligner, the ONLY alignment agent wastes all time in navigate→stuck→hub→navigate loops. Zero exploration → zero new junctions → zero network.
- **Same pattern as P139**: Helps c6 (multiple aligners compensate), destroys c2 (single aligner = 100% of alignment capacity).
- **Confirms Rule #4**: Exploration reduction ALWAYS hurts (P16, P41, P45, P53, P98, P151).
- **Salvageable idea**: Instead of direct navigation, bias wander sector toward known junction (preserves exploration + pre-positioning). → P152.
- Category: Capability improvement (aligner efficiency). REVERTED due to c2 failure.

### Cycle 54: P150 Network Redundancy Scoring — NOISE (April 15)
- **Method**: Paired same-seed test. 10 seeds (1-10) x machina_1.clips c6.
- **P150 (add redundancy * 5.0 bonus for junctions near 2+ net-connected cogs junctions)**: -2.3%, p=0.700, Cohen's d=-0.126. **NOISE.**
- **Hypothesis**: Junctions creating redundant network paths resist cascade failure from clips scrambles.
- **Why failed**: Consistent with P2 (+2.8% on arena, neutral). Frontier scoring already implicitly captures some redundancy benefit. The explicit bonus doesn't add enough value. One catastrophic seed (-11.43) but mostly flat.
- **Context**: P2 tested redundancy * 4.0 on arena (no clips). P150 tested * 5.0 on machina_1.clips (with clips). Neither helps.
- Category: Capability improvement (junction selection). Not significant.

### Cycle 53: P141 2M/4A for c6+ — NOISE/NEGATIVE (April 15)
- **Method**: Paired same-seed test. 20 seeds (41-60) x machina_1.clips c6. 4 parallel batches.
- **P141 (c5-c6 uses 2M + rest aligners instead of ROLE_CYCLE 3M)**: -7.9%, p=0.395, Cohen's d=-0.195. **NOISE but NEGATIVE direction.**
- **Hypothesis**: With P139 (always hub), hearts overproduced. 3rd miner redundant. Extra aligner worth more.
- **Why failed**: Three catastrophic seeds (-29.65, -35.03, -29.22) where heart pipeline collapsed with only 2 miners. SD=13.16 shows extreme variance. Most seeds are flat but some game states require all 3 miners for balanced element production.
- **Confirms**: P105 (+0.6%), P40 (-38%). 3M is the minimum safe miner count for c6+.
- Category: Capability improvement (role optimization). Not significant.

### Cycle 52: P155 progress-based junction timeout — NOISE (April 15)
- **Method**: Paired same-seed test. 20 seeds (21-40) x machina_1.clips c6. 4 parallel batches.
- **P155 (blacklist junction if Manhattan distance doesn't decrease for 15 ticks)**: +1.1%, p=0.615, Cohen's d=+0.114. **NOISE.**
- **Hypothesis**: Agents circling walls waste time on unreachable junctions. Detecting lack of distance progress and blacklisting early would save 15-30 ticks per failed attempt.
- **Why noise**: Manhattan distance is a poor progress proxy with 43% walls. Agents navigating around wall clusters temporarily increase Manhattan distance before eventually decreasing it. False triggers offset genuine saves.
- Category: Capability improvement (progress detection). Not significant.
- **Tournament update**: v26 at 23.52 (20m), v25 at 23.40 — both below v24 at 25.60. P127 (timeout 50→60) likely hurts tournament. v27 (v24+P139, no P127) in qualifying.

### Cycle 51: P140 post-death state reset — NOISE (April 14)
- **Method**: Paired same-seed test. 20 seeds x machina_1.clips c6 (scrimmage). 4 parallel batches.
- **P140 (clear failed_junctions, stuck_count, target on death)**: +1.6%, p=0.452, Cohen's d=+0.172. **NOISE.**
- **Hypothesis**: When agents die and respawn, stale state from previous life persists (failed_junctions, stuck_count, target). Clearing on death gives fresh targeting.
- **Why noise**: Deaths are only 12.5/agent. Blacklist is small (~5 entries). Clearing it barely changes behavior. Fresh gear acquisition naturally resets targeting flow.
- Category: Bug fix. Not significant.

### Cycle 50: P139 always hub for heart — KEEP +4.9% (April 14)
- **Method**: Paired same-seed test. 20 seeds x machina_1.clips c6 (scrimmage).
- **P139 (skip heartless exploration, always go to hub)**: +4.9%, p=0.099, Cohen's d=+0.388. **KEEP!**
- **Hypothesis**: Hub has 300+ hearts worth of deposits. Heartless exploration wastes aligner time when hearts are always available. Skip exploration, go directly to hub.
- **Why it works**: Eliminates ~20-40 ticks of wasted exploration per alignment cycle. Aligners get hearts faster → more alignment cycles per game. Hub starts with 8 hearts (5 starting + 3 craftable from initial 24 elements) and miners deposit continuously.
- **Win #12!** First win since P127 (cycle 44). Breaks 8-cycle fail streak. Deep analysis identified this via game profiling: 345 hearts craftable but only 81 used = 23% utilization.
- **Applied to softy.py**. v26 = v25 + P139.

### Cycle 49: P138 unified clips+neutral scoring — NOISE (April 14)
- **Method**: Paired same-seed test. 20 seeds x machina_1.clips c6 (scrimmage).
- **P138 (include clips junctions in nearest_alignable_junction scoring)**: +0.0%, p=0.998, Cohen's d=+0.001. **DEAD ZERO.**
- **Hypothesis**: Off-screen targeting only considers neutral junctions. On machina_1.clips with NPC clips ships, most junctions are clips. Aligners should target clips junctions with frontier scoring.
- **Why failed**: Stale junction data. Coordinator records junction alignment when agent SEES it. Clips ships align junctions to clips, but if no agent re-visits, coordinator still says "neutral." So the neutral-only filter was ALREADY targeting clips junctions (it thought they were neutral). P138 changed a label, not behavior.
- **Deep analysis game profiling**: deaths=12.5/agent (NOT 55), move failures=17.2%, 43% walls. 345 hearts craftable but only 81 alignments. Aligner cycle time is 370 ticks/alignment.

### Cycle 48: P137 junction heal staleness buffer — REVERT (April 14)
- **Method**: Paired same-seed test. 20 seeds × machina_1.clips c6 (scrimmage).
- **P137 (nearest_healing +10 junction buffer)**: -3.8%, p=0.107, Cohen's d=-0.378. **MARGINAL NEGATIVE → REVERT**.
- **Hypothesis**: Stale junction data causes nearest_healing() to underestimate distance to actual healing when clips scramble junctions. Adding +10 buffer accounts for staleness.
- **Why failed**: Same pattern as P136. Adding distance buffer makes agents retreat TOO EARLY (think healing is further away → retreat sooner → abandon territory → cascade). Deaths are from clips territory disruption (sudden HP drain), NOT from underestimated heal distance.
- **Key learning**: The death problem on machina_1.clips is NOT solvable by retreat timing adjustments. P136 (+margin) and P137 (+buffer) both hurt. Deaths come from clips ships disrupting territory, causing sudden HP loss. Need fundamentally different approach.
- **Consecutive fails**: 7. **DEEP ANALYSIS MODE TRIGGERED.**

### Cycle 47: P135b + P136 on machina_1.clips — NOOP/REVERT (April 14)
- **Method**: Paired same-seed tests. 19 seeds × machina_1.clips c6 (scrimmage). **First tests on tournament map!**
- **P135b (respawn recalibration)**: +0.0%, exact zero all seeds. LP is continuous through death — no reset, no position jump. P135 hypothesis is DEAD.
- **P136 (HP_SAFETY_MARGIN 15→25)**: -4.7%, p=0.096. **MARGINAL NEGATIVE**. More conservative retreat = more travel time wasted. Deaths NOT reduced (46→50 avg). Increased retreat causes territory abandonment.
- **Key discovery**: machina_1.clips has ~55 deaths/agent vs arena's ~0.3! Clips ships cause massive death pressure. Testing on arena missed this entirely.
- **Key learning**: HP_SAFETY_MARGIN=15 is already near-optimal. Deaths are from clips territory disruption, not insufficient retreat.

### Cycle 46: P134 BFS blocking + P135 respawn recalibration — REVERT/NOOP (April 14)
- **Method**: Paired same-seed tests. 20 seeds × arena c6 (scrimmage).
- **P134 (agent-only BFS blocking)**: -89.5%, p≈0. **CATASTROPHIC**. Only blocking agents (not buildings) in `_go_absolute` and `_wander` BFS. Buildings ARE impassable in the game engine. Agents walked into buildings constantly, triggering stuck. Confirms: all tagged cells in observation = impassable walls for BFS.
- **P135 (respawn recalibration)**: +0.0%, exact zero on all 20 seeds. **NOOP**. Position jump detection (>20 Manhattan) never triggered. Deaths are extremely rare in self-play arena, or LP is continuous through death.
- **Decision**: Both reverted. P134 teaches us BFS blocking is correct as-is. P135 is a noop in the test scenario.
- **Key learning**: Tagged entities in observation window = impassable. Current BFS is the right approach.

### Cycle 45c: Timeout parameter sweep — RESEARCH (April 15)
- **Method**: Full paired parameter sweep. 7 timeout values (25/35/50/60/70/80/90/100), 20 seeds each, 280 total episodes on c6.
- **Results**: Steep left wall (25=-53.4%, 35=-26.3%, 50=-5.6%, all p<0.025), peak at 60, flat plateau right (70/80/90/100 all noise p>0.22).
- **Curve shape**: Asymmetric — game punishes impatience catastrophically, forgives excess patience.
- **Mechanism**: At 0.4 cells/tick, 60 ticks = 24 cells. Covers alignment range (15-25 cells) + wall detours. Below 60, agents abandon reachable targets. Above 60, extra patience is free.
- **Research value**: Proves the depth method doesn't just find improvements — it characterizes entire parameter spaces.

### Cycle 45b: P129 miner round-trip extractor scoring — NOISE (April 14)
- **Method**: Paired same-seed test. 20 seeds × c6.
- **P129 (miner extractor scoring by round-trip cost)**: -0.9%, p=0.675, Cohen's d=-0.10. **NOISE**.
- **Decision**: Revert. Round-trip scoring (approach + return distance) didn't beat simple closest-extractor. Miners' approach distance is already a good proxy for efficiency — extractors near the miner are also generally near deposit points.

### Cycle 45: P128 timeout 60→70 — NOISE (April 14)
- **Method**: Paired same-seed test. 20 seeds × c6.
- **P128 (timeout 60→70)**: +0.9%, p=0.611, Cohen's d=0.12. **NOISE**.
- **Decision**: Revert. Timeout series plateaus at 60. Monotonic trend (15→25→35→50→60 all helped) finally broke.
- **Key insight**: First directional rule violation attempt. Timeout ≤60 always helps, but 70 is past the sweet spot. ~60 ticks ≈ 24 cells at 0.4 cells/tick — matches alignment range + detour budget. Above this, agents are wasting time on unreachable targets.

### Cycle 44: DEEP ANALYSIS — Paired retesting of near-misses (April 14)
- **Method**: First paired same-seed test. 20 seeds × c6, baseline vs variant on identical maps.
- **P86 (deposit 30→40)**: +2.1%, p=0.512, Cohen's d=0.15. **NOISE** — collapsed in late seeds.
- **P37 (stuck 5→3)**: +3.8%, p=0.200, Cohen's d=0.29. **MARGINAL** — positive but not significant.
- **P127 (timeout 50→60)**: +5.9%, p=0.012, Cohen's d=0.57. **CONFIRMED** — 4th timeout win in series.
- **Decision**: P127 applied. Timeout series: 15→25(+14%)→35(+8.1%)→50(+3.8%)→60(+5.9%). Win #11!
- **Compound test (P127+P37)**: c4 +0.5% (p=0.93), c6 +0.3% (p=0.87). P37 effect subsumed by P127 — both help stuck-near-junction agents.
- **Key insight**: Paired testing detected P127 at p=0.012 — unpaired VOR showed 1.69/1.73 (invisible noise). The depth approach works.

### Cycle 43: P126 navigate-to-spawn-when-lost on v24 base (April 14)
- **Hypothesis**: Profiling showed agents stuck in GEAR phase for entire games (no hub_offset → can't navigate to station). Fix: navigate to spawn point (0,0 lp) when lost.
- **Broad fix (station + heart + retreat)**: Run 1: 1.69 vs 1.72 (-1.7%). Run 2: 1.69 vs 1.71 (-1.2%). AVG -1.5%.
- **Refined (station-only)**: 1.66 vs ~1.72 (-3.5%).
- **Why failed**: Navigating to spawn creates oscillation (go to (0,0), wander away, back to (0,0)). Heart source fix hurt exploration. Retreat fix pulled agents away from nearby healing. 9 consecutive fails.
- **Key discovery from profiling**: Aligner station NOT visible at spawn in c=8 (outside 13×13 window). Agents find stations by navigating to hub then wandering nearby. This is slow but works.

### Cycle 42: P125 HP_SAFETY_MARGIN 15→25 on v24 base (April 14)
- **Hypothesis**: 42.62 deaths/agent = 21% time wasted. Higher retreat threshold → fewer deaths → more productive ticks.
- **Run 1**: Baseline 1.66, P125 1.73 (+4.2%). **Run 2**: Baseline 1.73, P125 1.69 (-2.3%).
- **Average**: +0.9% = NOISE. First run's baseline was anomalously low.
- **Conclusion**: Earlier retreat doesn't help. Agents already retreat appropriately. The deaths come from sudden HP loss (wall hits, long paths) not gradual drain. 8 consecutive fails.

