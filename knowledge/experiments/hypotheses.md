# Improvement Hypotheses (Prioritized Queue)

## P200: Scrambler Highway + Role Switching [IMPLEMENTED — v30, server down]
- **Status**: Code in softy.py. v30-v33 all failed qualifying (server issue). NEEDS c1/c2 fix (see P200 spec).
- **Urgent fix**: ROLE_DISTRIBUTIONS[1]=("aligner",), [2]=("aligner","aligner"), [3]=("aligner","aligner","scrambler")
- **Full spec**: `knowledge/experiments/P200-scrambler-highway.md`
- **Date**: April 15, 2026

## ~~P210: Scrambler HP Buffer Strategy~~ [DISPROVED]
- **Hypothesis**: Scrambler gear gives +200 HP. Start as scrambler for HP buffer, switch to aligner.
- **Result**: DISPROVED. HP bonus is DYNAMIC — calculated from current inventory, not persistent.
- **Source code proof**: `gear_stations.py:50-55` executes `ClearInventoryMutation` on gear switch, destroying old gear. HP limit = `min(max, max(base, sum(modifier_bonus * quantity_held)))` — quantity_held drops to 0 when old gear cleared. Test `test_change_gear_clears_old_gear` confirms.
- **Implication**: Scrambler→aligner switch gives ZERO HP benefit.
- **HP efficiency gap RESOLVED**: Hearts do NOT give HP — they're independent systems. `hp.gained` = all HP increases (spawn + territory healing at +100/tick). `heart.gained` = hearts for alignment actions. The 162 vs 280-313 "HP/heart" ratio is dividing two unrelated metrics. The real driver is territory presence time and death rate, not hearts.
- **Date**: April 15, 2026

## P211: Explore More Territory [NEW — MEDIUM]
- **Hypothesis**: Softy explores 1714 unique cells vs competitors' 2223-2508 (30-46% less). More exploration → discover more junctions → more alignment targets. Currently 8-sector sweep, but sectors may be too small.
- **Evidence**: dinky explores 2508 cells, scores 27.31. Softy explores 1714 cells, scores 25.60.
- **Test**: Increase SECTOR_RADIUS from 30 to 50. Or: explore ALL discovered junction neighbors, not just sector direction.
- **Risk**: More exploration = more time outside territory = more deaths. Must balance.
- **Category**: Capability improvement
- **Date**: April 15, 2026

## P212: Optimal c1 Role — Is Aligner Really Best? [NEW — INVESTIGATION]
- **Hypothesis**: At c1, teammates align 150+ junctions and Softy adds ~61. What if Softy's 1 agent would be more valuable as a miner (boosting team heart supply) or scrambler (creating neutrals for teammates)?
- **Evidence**: Reward is team-based. Softy's individual junction count doesn't affect reward directly. The team might benefit more from resource support than one more aligner.
- **Test**: Run c1 as miner, c1 as scrambler, compare team-level cogs.junction.held vs c1 as aligner.
- **Problem**: Can't test cooperative play locally — `cogames pickup` runs all-Softy teams.
- **Category**: Strategic investigation
- **Date**: April 15, 2026

## ~~P1: Add scramblers to counter cascade failure~~ [TESTED — FAILED]
- **Result**: 2M/1Sc/5A → VOR 0.44 vs baseline 0.77 (-43%)
- **Root cause**: Losing a miner cripples heart economy. Scrambler works but tradeoff is net negative.
- **Date**: April 14, 2026

## ~~P2: Network-aware junction targeting (redundancy bonus)~~ [TESTED — NEUTRAL]
- **Result**: VOR 0.73 vs baseline 0.71 (+2.8%). Below 5% threshold.
- **Root cause**: The `redundancy * 4.0` bonus was too small vs existing frontier scoring. Junction selection already decent.
- **Date**: April 14, 2026

## P3: Clips timing exploitation [TESTED — PROMISING BUT INTERFERES WITH P4]
- **Result solo**: VOR 0.76 vs baseline 0.71 (+7.0%). Above 5% threshold.
- **Result stacked with P4**: VOR 0.72 — interference, worse than either alone.
- **Note**: May be worth revisiting as a standalone if P4 is ever removed. Do NOT combine with P4.
- **Date**: April 14, 2026

## ~~P4: Energy-aware navigation~~ [TESTED — KEPT, v6]
- **Result**: VOR 0.77 (5ep) / 0.78 (10ep confirmed) vs baseline 0.71 (+10%)
- **Implementation**: Night energy threshold 5 (vs 2 for day) in `_low_energy()`
- **Uploaded**: Softy:v6
- **Date**: April 14, 2026

## ~~P5: Hub initial heart rush optimization~~ [TESTED — FAILED]
- **Result**: VOR 0.73 vs baseline 0.78 (-6.4%). Hub congestion + lost exploration.
- **Date**: April 14, 2026

## P6: Heartless exploration cycling [TESTED — NEUTRAL, POSITIVE DIRECTION]
- **Result**: VOR 0.80 vs baseline 0.78 (+2.6%). Below 5% threshold.
- **Note**: Marginal gain. Could try even faster (every 3 ticks) or combine with other changes.
- **Date**: April 14, 2026

## ~~P7: Scrambler timing at tick ~90~~ [SKIPPED — P1 failed]
- **Depends on**: P1 which regressed. No scramblers in current build.

## ~~P8: Dynamic role switching mid-game~~ [TESTED — FAILED]
- **Result**: VOR 0.72 vs baseline 0.78 (-7.7%). Economy collapses without miners.
- **Date**: April 14, 2026

---

## P9: Track net:cogs tags — only target connected junctions [TESTED — NEUTRAL]
- **Result**: VOR 0.80 vs baseline 0.78 (+2.6%). Below 5% threshold.
- **Root cause**: Connectivity bonus (6.0 * connected neighbors) too weak relative to frontier scoring. May need stronger weight or different approach (filter instead of bonus).
- **Date**: April 14, 2026
- **Retry idea**: Instead of bonus, FILTER OUT targets not adjacent to any connected junction.

## ~~P10: Parse clips ship positions — anti-scramble awareness~~ [TESTED — FAILED]
- **Result**: VOR 0.72 vs baseline 0.78 (-7.7%).
- **Root cause**: Ship threat penalty (8-20 points) was too aggressive, steering aligners away from high-value frontier junctions near ships. Ships are at map corners — the best junctions are often near ships.
- **Date**: April 14, 2026
- **Never retry this approach**: Avoidance doesn't work. Need proactive defense (scramble back) instead.

## ~~P11: Game-phase strategy adaptation~~ [TESTED — REVERTED, hurt on leaderboard]
- **Local result**: VOR 1.26 (10ep) vs baseline 1.04 (+21%).
- **Tournament result**: v9 scored 7.51 vs v7/v8's 10.52 — **-28% REGRESSION**
- **Why it failed on leaderboard**: Aggressive early expansion (frontier_weight 12, dist_weight 0.7) sends aligners too far from base, making them vulnerable. Late consolidation gives up territory real opponents contest.
- **REVERTED**: v10 removes all P11 changes.
- **Lesson**: Behavior tuning that helps vs random gets exploited by real opponents. ONLY capability improvements translate. Same pattern as P4.
- **Date**: April 14, 2026

## P13: Aligner failed junction expiry [TESTING — 10ep confirmation]
- **5ep screen**: VOR 1.13 vs baseline 1.07 (+5.6%). Above threshold.
- **Awaiting**: 10ep confirmation run.
- **Implementation**: Dict with tick timestamp, expire after 200 ticks.
- **Date**: April 14, 2026

## P14: Smarter deposit threshold scaling [MEDIUM — NEW]
- **Hypothesis**: MINER_DEPOSIT_THRESHOLD=30 is static. Early game: deposit frequently (hearts needed ASAP). Late game: deposit less often (economy established).
- **Test**: Scale threshold: 15 before tick 1000, 30 after. Gets hearts flowing faster early.
- **Expected impact**: 5-8% VOR — faster early-game heart supply
- **Implementation**: In _miner(), use `threshold = 15 if s.step_count < 1000 else MINER_DEPOSIT_THRESHOLD`

## ~~P15: Increase aligner junction timeout 15→25~~ [TESTED — KEPT, v16]
- **Result**: VOR 1.22 (10ep) vs baseline 1.07 (+14%). BIGGEST WIN SINCE P12.
- **Implementation**: `target_ticks > 25` in _aligner()
- **Uploaded**: Softy:v16
- **Date**: April 14, 2026

## P18: Fix _in_align_range to use network BFS closure [TESTED — NEUTRAL LOCALLY]
- **Result**: VOR 1.14 vs baseline ~1.12 (+1.8%). Below 5% threshold locally.
- **Root cause**: Random opponents rarely cause cascade failure, so the fix doesn't help locally. May help in tournament where opponents scramble strategically.
- **Note**: Consider applying anyway for tournament — inverse of P4 situation (locally neutral but could be positive on leaderboard).
- **Date**: April 14, 2026

## P19: Aligner wave expansion (coordinate growth direction) [TESTED — NEUTRAL]
- **Result**: VOR 1.28 (10ep) vs baseline 1.22 (+4.9%). Below 5% threshold.
- **Root cause**: Marginal benefit. Existing frontier scoring already encourages contiguous growth. The +5 wave bonus helped slightly but added complexity.
- **Date**: April 14, 2026

## P20: Soft clips junction bonus (+6, within 20 cells) [TESTED — NEUTRAL]
- **Result**: VOR 1.17 vs baseline ~1.12 (+4.5%). Just below 5% threshold.
- **Root cause**: Even with reduced bonus (+6 vs P17's +12) and distance cap (20 cells), clips targeting still slightly distorts phase-adjusted scoring. The benefit of disrupting clips isn't enough to offset.
- **Date**: April 14, 2026

## ~~P16: Aggressive exploration sweep for heartless aligners~~ [TESTED — FAILED]
- **Hypothesis**: Heartless aligners waste time wandering near hub in random sectors. By sending them to coordinates far from ALL known junctions, they'll discover new junctions much faster → more alignment targets → faster network growth.
- **Test**: When heartless and hub can't craft, compute least-explored quadrant (farthest from all known junction centroid) and navigate there instead of sector wander.
- **Expected impact**: 5-10% VOR — faster discovery of the ~69 junctions on the map
- **Implementation**: In `_aligner()` heartless branch, replace sector wander with targeted exploration toward map regions with no known junctions. Use `_go_absolute()` to navigate to computed exploration target.

## P17: Clips junction priority scoring (dual value) [TESTED — PROMISING BUT INTERFERES WITH P11]
- **Hypothesis**: Re-aligning clips junctions provides dual value: we gain a junction AND potentially cascade-disconnect their network. Currently clips junctions are only targeted as a fallback after no neutral junctions remain.
- **Test**: In `nearest_alignable_junction()`, also consider clips junctions with a bonus score. Score clips junctions as: `frontier * 8.0 - dist + 12.0` (clips bonus).
- **Expected impact**: 5-10% VOR — strategic advantage from disrupting clips network
- **Implementation**: Modify `nearest_alignable_junction()` to include clips junctions with bonus scoring. Remove separate `nearest_enemy_alignable_junction()` fallback.

## ~~P27: Scramble detection + priority re-alignment~~ [TESTED — NEUTRAL]
- **Result**: VOR 1.11 vs baseline 1.10 (+0.9%). Neutral.
- **Root cause**: Random opponents don't scramble. May help in tournament.
- **Date**: April 14, 2026

## ~~P28: Heart-count-aware aligner dispatch~~ [TESTED — FAILED]
- **Result**: VOR 1.01 vs baseline 1.10 (-8.2%). Regression.
- **Root cause**: Too restrictive. Heart count snapshot stale, multiple aligners CAN benefit from simultaneous hub trips.
- **Date**: April 14, 2026

## ~~P29: Smarter stuck recovery~~ [TESTED — KEPT v12, then REVERTED v13]
- **Local result**: VOR 1.21 (10ep) vs baseline 1.10 (+10%)
- **Tournament result**: v12 scored 9.92 vs v11's 11.33 (-12.4%)
- **Why it failed in tournament**: Hub retreat when stuck provides healing, reset, and resource management under competitive pressure. Skipping hub exposes agents to opponent pressure.
- **REVERTED**: v13 = v11 code (P12 + P18 only)
- **Date**: April 14, 2026

## ~~P30: A* pathfinding on shared wall map~~ [TESTED — NEUTRAL]
- **Result**: VOR 1.26 (5ep) / 1.21 (10ep confirmed) vs baseline 1.19 (+1.7%). Neutral.
- **Implementation**: Wall tag detection, shared wall map in coordinator, A* with 2000 iteration cap, local BFS fallback.
- **Root cause**: Partial wall map doesn't contain enough data for A* to find meaningfully better paths. By the time sufficient walls are mapped (late game), agents have already learned routes through stuck detection.
- **Date**: April 14, 2026

## ~~P31: Visited area tracking for exploration~~ [TESTED — FAILED]
- **Result**: VOR 0.41 vs baseline ~1.21 (-66%). Catastrophic regression.
- **Root cause**: All agents converge on same "unexplored" sector, causing massive congestion.
- **Date**: April 14, 2026

## ~~P32: Align-range check for visible junctions~~ [TESTED — FAILED]
- **Result**: VOR 1.11 vs baseline ~1.21 (-8.3%). Regression.
- **Root cause**: Preemptive range-filtering rejects junctions about to become in-range as network grows. The 15-tick timeout is better for dynamic network.
- **Date**: April 14, 2026

## ~~P33/P33b: Frontier scoring for visible junctions~~ [TESTED — NEUTRAL]
- **Result**: VOR 1.24 vs baseline 1.19 (+4.2%). Below 5% threshold. Tested twice (with/without BFS caching).
- **Root cause**: Visible junctions picked by _closest are usually the same ones frontier scoring would pick.
- **Date**: April 14, 2026

## ~~P34: Mine near cogs junctions~~ [TESTED — NEUTRAL]
- **Result**: VOR 1.22 vs baseline 1.19 (+2.5%). Below 5% threshold.
- **Root cause**: Junction proximity bonus rarely changes extractor selection.
- **Date**: April 14, 2026

## P35: Shared out-of-range junction tracking [TESTING]
- **Hypothesis**: When one aligner times out on an out-of-range junction, other aligners still try it (each has own blacklist). Sharing confirmed out-of-range data saves up to 5 × 15 = 75 ticks of wasted targeting.
- **Implementation**: Coordinator-level `out_of_range_junctions` set, populated on timeout + range check fail. Clears when network grows.
- **Type**: Coordination bug fix
- **Date**: April 14, 2026

## P36: Visible junction claim check [TESTING]
- **Hypothesis**: Visible junctions bypass `is_claimed` check, so two aligners seeing the same junction both target it. One wastes their heart.
- **Implementation**: Add `is_claimed` check + `claim_target` call for visible junction targeting path.
- **Type**: Deconfliction bug fix
- **Date**: April 14, 2026

## ~~P37: Faster hub retreat on stuck (threshold 5→3)~~ [TESTED — NEUTRAL]
- **Result**: VOR 1.14 vs baseline 1.10 (+3.6%). Below threshold.
- **Root cause**: Right direction but stuck events too infrequent. Saves ~2 ticks per event.
- **Date**: April 14, 2026

## P38: Clips junction priority (+10 bonus, unified scoring) [TESTING]
- **Hypothesis**: Flipping clips→cogs is worth 2 points (we gain + they lose). Currently clips junctions are only targeted as fallback. By scoring them alongside neutral junctions with +10 bonus, we attack clips network while expanding ours.
- **Previously**: P17 (+5.8% on v9 with P11 interference), P20 (+4.5% on v9). Both untested on clean v13 base.
- **Type**: Strategic capability (offense)
- **Date**: April 14, 2026

## P39: Distance-only targeting (remove frontier scoring) [TESTING]
- **Hypothesis**: DIAGNOSTIC — test whether frontier scoring (`frontier * 8.0 - dist`) helps or hurts. If distance-only is better, our scoring algorithm is the bottleneck.
- **Type**: Diagnostic test
- **Date**: April 14, 2026

## ~~P38: Clips junction priority~~ [TESTED — FAILED]
- **Result**: VOR 1.06 vs baseline 1.10 (-3.6%). Few clips targets vs random.
- **Date**: April 14, 2026

## ~~P39: Distance-only targeting~~ [TESTED — DIAGNOSTIC]
- **Result**: VOR 1.07 vs baseline 1.10 (-2.7%). Confirms frontier scoring IS helpful.
- **Date**: April 14, 2026

## ~~P40: 4 miners + 4 aligners~~ [TESTED — CATASTROPHIC]
- **Result**: VOR 0.68 vs baseline 1.10 (-38%). Proves alignment capacity > heart production.
- **Date**: April 14, 2026

## P41: Hub-proximate heartless exploration [TESTING]
- **Hypothesis**: Heartless aligners stay within 20 cells of hub. When hearts become available, they're already close → faster heart sourcing → more alignment cycles.
- **Type**: Efficiency capability (reduces heart-sourcing lag)
- **Date**: April 14, 2026

## P42: Frontier weight 12.0 (from 8.0) [TESTING]
- **Hypothesis**: DIAGNOSTIC — since frontier scoring helps (P39), test if MORE frontier weighting is better. Pushes aligners toward highest-expansion junctions.
- **Type**: Weight optimization diagnostic
- **Date**: April 14, 2026

## P43: Frontier weight 4.0 (from 8.0) [TESTING]
- **Hypothesis**: DIAGNOSTIC — test if LESS frontier weighting (more distance-biased) is better. Brackets with P42 to find optimal weight.
- **Type**: Weight optimization diagnostic
- **Date**: April 14, 2026

## ~~P44: Scramble detection + re-alignment priority~~ [TESTED — FAILED]
- **Result**: VOR 1.02 vs baseline 1.07 (-4.7%).
- **Root cause**: Random opponents don't scramble. Detection never triggers.
- **Date**: April 14, 2026

## P45: Hub check interval 30→15 [TESTING]
- **Hypothesis**: Heartless aligners check hub every 30 ticks. If miners deposit hearts sooner, aligner doesn't know for up to 25 ticks. Faster checks = faster heart pickup.
- **Type**: Bug fix (stale data lag)
- **Date**: April 14, 2026

## P46: Distance-adjusted junction timeout [TESTING]
- **Hypothesis**: Fixed 25-tick timeout (P15) is suboptimal. Close junctions should blacklist faster (15 ticks), far junctions slower (dist/2 ticks). More precise than flat 25.
- **Type**: Capability refinement
- **Date**: April 14, 2026

## P48: Visible junction claim write (no check) [TESTING]
- **Hypothesis**: Visible junctions bypass claim_target, so off-screen scoring doesn't know this agent is targeting that junction. Write claim (inform) without check (don't block). Improves deconfliction without the P36 regression.
- **Type**: Bug fix (stale claims)
- **Date**: April 14, 2026

## ~~P12: Miner element coordination via hub inventory~~ [TESTED — KEPT, v7]
- **Result**: VOR 1.05 (5ep) / 1.14 (10ep) vs baseline 0.78 (+46%). BIGGEST SINGLE WIN.
- **Implementation**: `bottleneck_element()` in coordinator, dynamic targeting in `_miner()` Phase 3
- **Uploaded**: Softy:v7
- **Date**: April 14, 2026
- **Note**: Expected 3-4% but got 46%. Heart production was the dominant bottleneck.
