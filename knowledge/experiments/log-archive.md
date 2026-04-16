### Cycle 41: P33b re-test (user request) on v24 base (April 14)
- **Run 1**: Baseline 1.71, P33b 1.76 (+2.9%). **Run 2**: Baseline 1.73, P33b 1.69 (-2.3%).
- **Average**: +0.3% = NOISE. The first +2.9% was variance, not signal. Confirmed dead.

### Cycle 40: P122 frontier radius correction + P121 claimed set opt on v24 (April 14)
- **P122 (frontier scoring with ALIGN_ACTUAL_RADIUS=15 instead of 20)**: VOR 1.73 (+1.2%) — NEUTRAL
- **P121 (pre-compute claimed set for O(1) lookup)**: VOR 1.68 (-1.8%) — REVERTED
- Both too small to measure. 8-agent claimed set is fast enough. Frontier radius 20 vs 15 negligible.

### Cycle 39: P120 wall memory + expanded BFS on v24 base (April 14)
- **Baseline (v24)**: VOR 1.71 (10ep)
- **P120 (shared wall positions, 19×19 BFS with wall memory)**: VOR 1.60 (-6.5%) — **REVERTED**
- Expanded BFS (19×19 vs 13×13) too slow. Wall set conversion per tick is expensive.
- The 13×13 BFS + stuck detection is already efficient enough. Speed > smarts here.

### Cycle 38: P33b revisit frontier scoring visible junctions on v24 base (April 14)
- **Baseline (v24)**: VOR 1.71 (10ep)
- **P33b (unified frontier scoring for visible+offscreen junctions)**: VOR 1.76 (+2.9%) — **REVERTED** (below 3%)
- Real but small. +4.2% on v12, +2.9% on v24. May be worth bundling with another change.

### Cycle 37: P106 BFS queue + net cache on v24 base (April 14)
- **Baseline (v24)**: VOR 1.71 (10ep)
- **P106 (deque popleft + junction-versioned caching for net_connected_junctions)**: VOR 1.68 (-1.8%) — **REVERTED**
- P104 already removed the 300x BFS multiplier. Remaining 5→1 calls per tick too small to measure.

### Cycle 36: P105 dynamic roles c5/c6 on v24 base (April 14)
- **Baseline (v24)**: VOR 1.71 (10ep)
- **P105 (threshold >=7 instead of >=5, c5=2M/3A, c6=2M/4A)**: VOR 1.72 (+0.6%) — **REVERTED**
- c6 already has 50/50 split (3M/3A). Extra aligner doesn't compensate for lost miner.
- Rule: dynamic roles help when alignment severely under-resourced (c2:0A, c4:1A). c6 at 3A is fine.

### Cycle 35: P104 BFS caching on v23 base (April 14)
- **Baseline (v23)**: VOR 1.60 (10ep, seed 50)
- **P104 (pre-compute align range set)**: VOR 1.73 (+8.1%) — **KEPT**
  - Eliminated ~300x redundant BFS computations per aligner per tick
  - Broad improvement across c4-c7 suggests faster execution is materially helping
  - Bug: _in_align_range() called net_connected_junctions() for EVERY candidate junction
    AND every frontier neighbor. _net_cache fields existed but were NEVER USED.
- **Decision**: P104 applied. v24 = v23 + P104. Uploaded.
- **Win #10!** Three back-to-back: P102 (dynamic c2), P103 (dynamic c4), P104 (BFS cache).
- **Retrospective + tournament audit due** (cycle 35 divisible by 5).

### Cycle 34: P103 c4 dynamic roles on v22 base (April 14)
- **Baseline (v22)**: VOR 1.60 (10ep, seed 50)
- **P103 (c4: 2M+2A instead of 3M+1A)**: VOR 1.69 (+5.6%) — **KEPT**
  - c4: 1.14→1.25 (+9.6%) — targeted improvement
  - c6: 1.46→1.88 (+28.8%) — NOISE, code unchanged for c6
  - c8/c7/c5 approximately unchanged as expected
  - Overall inflated by c6 noise, but c4 signal is real
- **Tournament reasoning**: c4 is 27% of tournament matches. P40 proved alignment capacity > mining. 
  2M+2A doubles aligner count for c4 at cost of 1 miner (P12 bottleneck mining sustains with 2).
- **Decision**: P103 applied. v23 = v22 + P103. Uploaded.
- **Win #9**. Dynamic roles now: c1=0M/1A, c2=1M/1A, c3=1M/2A, c4=2M/2A, c5+=ROLE_CYCLE.

### Cycle 33: P102 + Tournament Audit on v21 base (April 14)
- **P102 (conservative dynamic roles)**: VOR 1.69 (+2.4%) — LOCAL below threshold BUT KEPT.
  - c2: 0.45→0.82 (+82%), c3: 0.81→1.18 (+46%), c1: 0.22→0.25 (+14%)
  - c4-c8: unchanged (uses ROLE_CYCLE for teams≥4)
  - **TOURNAMENT JUSTIFICATION**: c2 is 31% of tournament matches but only 5.6% of local VOR. 
    P102 gives c2 alignment capability (1M+1A instead of 2M/0A).
    If tournament c2 improves by 50%+ (5.1→7.7+), that's +0.8+ overall VOR → pass slanky.
- **Tournament Audit (first ever)**: Created `/audit-tournament-softy` command and `tournament-audit.md`.
  - CRITICAL FINDING: Tournament only runs c2/c4/c6/c8, NOT c1/c3/c5/c7.
  - v21 per-scenario: c2=5.1 (weakest), c4=21.5, c6=30.9, c8=35.1.
  - c2 is 31% of matches — 5.5x more weight than local testing.
- **Decision**: P102 applied to softy.py. v22 = v21 + P102. Uploaded.
- **Win #8** (first tournament-audit-informed decision). New rule: use tournament weights for threshold.

### Cycle 32: P99/P100/P101 on v21 base (April 14)
- **P99 (simple aligner-first)**: VOR 1.62 (-1.8%) — FAILED. c6-c7 lose miners, kills VOR.
  - c3 improved +10.3%, c2 +11.9%, but c7 -5.0%, c6 -7.3%.
  - Key learning: VOR is count-weighted (c8=22.2%, c1=2.8%). Must preserve c6-c8 performance.
- **P100 (dynamic role assignment)**: VOR 1.64 (-0.6%) — FAILED. c3 +51.5% but c2 -16.7%, c6 -10.5%.
  - Key: Only ≤3 agent changes are safe. c6+ must keep 3 miners.
- **P101 (ALIGN_NET_RADIUS 20→15)**: VOR 1.67 (+1.2%) — NEUTRAL. Below 3% threshold.
- **Decision**: No changes. v21 stands. 4 consecutive fails (P97-P101).

### Cycle 31: P97/P98 on v21 base (April 14)
- **Baseline VOR**: 1.65 (v21, 10ep)
- **V1 (P97) Hub calibration normalization (NW-most cell)**: VOR 1.59 (10ep) = -3.6% — FAILED
  - If hub is single-cell, this is a noop. Regression suggests the change slightly disrupted calibration timing.
- **V2 (P98) Direct-to-hub when 5+ neutral junctions known**: VOR 1.58 (10ep) = -4.2% — FAILED
  - Same pattern as P45: reducing exploration time hurts. Junction discovery is ALWAYS valuable.
- **Decision**: No changes. v21 stands.
- **v21 tournament: 22.34 VOR rank 6 — TOP 10! +18.3% from v20 (18.88).**
- **KEY INSIGHT (user)**: dinky dominates 1-2 player scenarios. Our ROLE_CYCLE puts miners first → with 1-3 agents, zero alignment capability. Testing P99 (aligner-first) next.

### Cycle 30: P93/P96 on v21 base (April 14)
- **Baseline VOR**: 1.65 (v21, 10ep)
- **V1 (P93) Clear stale claims on timeout/heartless**: VOR 1.63 (10ep) = -1.21% — FAILED
  - Same as P60: stale claims PROTECT other agents from unreachable junctions. Don't clear.
- **V2 (P96) Use shared hub resources for hub_can_craft**: VOR 1.58 (10ep) = -4.24% — FAILED
  - Stale coordinator data worse than periodic proximity checks. Far-away data unreliable.
- **Decision**: No changes. v21 stands.
- **v20 tournament: 18.88 VOR rank 21 — TOP 25! +17% from v18 (16.14).**

### Cycle 29: P89/P92 on v20 base (April 14)
- **Baseline VOR**: 1.59 (v20, 10ep)
- **V1 (P89) Net radius 20 + timeout 50**: VOR 1.65 (10ep) = **+3.77% — KEPT** (new 3% threshold)
  - Combined P87 (radius 15→20) and P88 (timeout 35→50). Improvement entirely from timeout.
- **V2 (P92) Per-node BFS direction in `_go_absolute`**: VOR 1.55 (10ep) = -2.52% — FAILED
  - Off-screen targets are so far that fixed direction priority is fine within 13×13 window.
- **Decision**: P89 applied to softy.py. v21 = v20 + P89. Uploaded.
- **Win #7** of 92 hypotheses. 7 for 7 on bug fixes + capability improvements.
- **Threshold lowered**: 5% → 3% (scraping phase). 10ep direct, no 5ep screen.

### Cycle 28: P87/P88 on v20 base (April 14)
- **Baseline VOR**: 1.41 (v20, 5ep) / 1.59 (10ep)
- **V1 (P87) Net radius 15→20**: VOR 1.63 (5ep) → **1.59 (10ep) = 0% — NOISE**. 5ep was misleading.
- **V2 (P88) Timeout 35→50**: VOR 1.54 (5ep) → **1.65 (10ep) = +3.8% — NEUTRAL**. Below 5% threshold.
- **Decision**: No changes. v20 stands. Game alignment radius appears to be exactly 15 Manhattan.

### Cycle 27: P76/P86 on v19 base (April 14)
- **Baseline VOR**: 1.41 (v19, 5ep) / 1.49 (10ep)
- **V1 (P76) BFS wander**: VOR 1.54 (5ep) → **1.59 (10ep) = +6.7% — KEPT**
  - Bug: `_wander` used simple 4-direction tries that bounce off walls. BFS navigates around obstacles.
  - Fix: Wander calls `_go_absolute(fallback_to_wander=False)` for sector navigation.
- **V2 (P86) Deposit threshold 30→40**: VOR 1.48 (+3.5%) — NEUTRAL. Slight improvement but below threshold.
- **Decision**: P76 applied to softy.py. v20 = v19 + P76. Uploaded.
- **Win #6** of 65+ hypotheses. 6 for 6 on bug fixes.

### Cycle 26: P73/P75 bug fixes on v18 base (April 14)
- **Baseline VOR**: 1.31 (v18, 5ep)
- **V1 (P73) Sector collision fix**: VOR 1.43 (5ep) → **1.49 (10ep) = +13.7% — KEPT**
  - Bug: `_explore_offset = agent_id % 4` caused only 4/8 sectors used, agents doubled up.
  - Fix: `_explore_offset = 0` → all 8 agents get unique sectors. Also fixed stuck rotation to % 8.
- **V2 (P75) Territory-aware heal distance**: VOR 1.24 (-5.3%) — FAILED. Agents stay too long, die.
- **Decision**: P73 applied to softy.py. v19 = v18 + P73. Uploaded.
- **Win #5** of 60+ hypotheses. All 5 are bug fixes.

### Cycle 25: P69/P72 new obs features on v18 base (April 14)
- **Baseline VOR**: 1.31 (v18, 5ep)
- **V1 (P69) net:cogs junction tracking**: VOR 1.24 (-5.3%) — FAILED. Distinguishing connected/disconnected cogs junctions hurts retreat distance calculations.
- **V2 (P72) aoe_mask territory-aware retreat**: VOR 1.25 (-4.6%) — FAILED. Skipping retreat in friendly territory prevents necessary hub trips for hearts.
- **Decision**: No changes. v18 stands.
- **Insight**: Both new observation features hurt. net:cogs redundant with BFS calculation. aoe_mask retreat skip removes beneficial hub visits.
- **v18 tournament**: 16.14 VOR rank 35 (+27.5% from v17's 12.67). P65 = best tournament gain ever.

### Cycle 24: P70/P71 vibes on v18 base (April 14)
- **Baseline VOR**: 1.33 (v18)
- **V1 (P70) Role vibes**: VOR 1.39 (5ep) = +4.5%. Below threshold.
- **V2 (P71) Heart vibes for aligners**: VOR 1.43 (5ep) → **1.25 (10ep) = -6.0% — FAILED**. 5ep was noise.
- **Decision**: Vibes don't help. The 1-tick cost may outweigh any interaction benefit.
- **Insight**: Tech manual says vibes "impact interactions" but this may be for RL-trained policies, not scripted.

### Cycle 23: P67/P68 on v18 base (April 14)
- **Baseline VOR**: 1.33 (v18)
- **V1 (P67) Timeout 30**: VOR 1.27 (-4.5%) — FAILED. 30 is worse than 35. Improvement is monotonic 25→35.
- **V2 (P68) Stuck threshold 7**: VOR 1.33 (0%) — NEUTRAL. Threshold doesn't matter with 35-tick timeout.
- **Decision**: No changes. v18 stands. Timeout 35 confirmed optimal (30<35>45 bracket).

### Cycle 22: P65/P66 timeout bracket on v17 base (April 14)
- **Baseline VOR**: 1.23 (v17)
- **V1 (P65) Timeout 25→35**: VOR 1.28 (5ep) → **1.33 (10ep confirmed) = +8.1% — KEPT**
- **V2 (P66) Timeout 25→45**: VOR 1.25 (+1.6%) — NEUTRAL. 45 is too long.
- **Decision**: P65 applied to softy.py. v18 = v17 + P65. Uploading.
- **Insight**: Same bug class as P15. At ~0.4 cells/tick, 25 ticks only covers ~10 cells, but alignment range is 15 cells from network.

### Parallel Cycle 21: P60/P42 on v17 base (April 14)
- **Baseline VOR**: 1.23 (v17)
- **V1 (P60) Stale claims fix**: VOR 1.18 (-4.1%) — FAILED. Stale claims PROTECT others from unreachable junctions.
- **V2 (P42) Frontier weight 12.0**: VOR 1.24 (+0.8%) — NEUTRAL. Weight 8.0 is near-optimal.
- **Decision**: No changes. v17 stands. 7 consecutive fails.

### Parallel Cycle 20: P56/P59 on v17 base (April 14)
- **Baseline VOR**: 1.23 (v17)
- **V1 (P56) Staggered mining**: VOR 1.01 (-17.9%) — FAILED. All miners should target the same bottleneck element.
- **V2 (P59) Blacklist clear on growth**: VOR 1.21 (-1.6%) — NEUTRAL. Too aggressive; most blacklisted junctions are genuinely unreachable.
- **Decision**: No changes. v17 stands. 5 consecutive fails.

### Parallel Cycle 19: P13/P54/P55 on v17 base (April 14, ~3:30 AM)
- **Baseline VOR**: 1.23 (10ep, v17)
- **V1 (P13) Junction expiry**: VOR 1.16 (-5.7%) — FAILED on v17. Neutral-only filter means smaller blacklist, expiry re-adds bad junctions.
- **V2 (P54) Miner threshold 20**: VOR 1.12 (-8.9%) — FAILED. More hub trips = more travel overhead.
- **V3 (P55) Aligner spacing penalty**: VOR 1.18 (-4.1%) — FAILED. Steers to suboptimal junctions.
- **Decision**: No changes. v17 stands.

### BUG FIX: Incomplete P38 revert — v17 uploaded (April 14, ~3:00 AM)
- **Bug**: P38 revert only removed clips_bonus but left `jalign == "cogs"` filter (includes clips). Original v11 had `jalign != "neutral"` (neutral-only). Also missing `nearest_enemy_junction` fallback method.
- **Impact**: v15/v16 were targeting clips junctions WITHOUT bonus — pure waste. Explains v15's 7.52 and v16's 10.19 tournament scores.
- **Fix**: Restored neutral-only filter + enemy junction fallback method. Uploaded as v17.
- **Lesson**: When reverting a change, check EVERY modified line, not just the obvious ones.

### Parallel Cycle 18: P49/P51/P53 exploration+heartless (April 14, ~2:30 AM)
- **Baseline VOR**: 1.20 (5ep, v16)
- **V1 (P49) Sector radius 45**: VOR 1.23 (+2.5%) — NEUTRAL
- **V2 (P51) Opportunistic hub check**: VOR 1.22 (+1.7%) — NEUTRAL
- **V3 (P53) Pre-select target while heartless**: VOR 1.18 (-1.7%) — FAILED (P41 pattern)
- **Decision**: No changes. v17 uploaded with P38 bug fix.

### Parallel Cycle 17: P45/P46/P48 aligner optimizations (April 14, ~2:00 AM)
- **Baseline VOR**: 1.20 (5ep, v16)
- **V1 (P45) Hub check 30→15**: VOR 1.15 (-4.2%) — FAILED. Less exploration time.
- **V2 (P46) Dist-adjusted timeout**: VOR 1.06 (-11.7%) — FAILED. Undoes P15 for close junctions.
- **V3 (P48) Visible claim write**: VOR 1.18 (-1.7%) — FAILED. Unnecessary overhead.
- **Decision**: No changes. v16 stands.
- **Learning**: P15's fixed 25-tick timeout is robust. Walls make distance ≠ travel time, so dynamic timeouts don't work. Hub check frequency and claim writes are already at correct balance.

### Upload: Softy:v16 — April 14, 2026
P15 (timeout 25) applied. v16 = P12 + P18 + P15. P38 reverted (v14 bombed at 5.34).

### Upload: Softy:v15 — April 14, 2026
P38 reverted. v15 = P12 + P18 only (clean base matching v11/v13).

### Parallel Cycle 16: P13/P15/P44 bug fixes + scramble detection (April 14, ~1:00 AM)
- **Baseline VOR**: 1.07 (5ep)
- **V1 (P13) Junction expiry 200 ticks**: VOR 1.13/1.15 (5ep/10ep, +7.5%) — **CONFIRMED**
- **V2 (P15) Timeout 15→25**: VOR 1.17/1.22 (5ep/10ep, +14.0%) — **CONFIRMED — BIGGEST SINCE P12**
- **V3 (P44) Scramble detection +15 bonus**: VOR 1.02 (-4.7%) — FAILED. Random opponents don't scramble.
- **Combined P13+P15**: VOR 1.23 (10ep) — marginal over P15 alone (1.22). Not worth combining.
- **Decision**: P15 applied to softy.py. Uploaded as v16. P13 deferred (subsumbed by P15).
- **Learning**: Aligner timeout of 15 was a MAJOR bug — 15 ticks ≈ 4 cells travel, but most junctions are 10-30 cells away. Aligners were blacklisting reachable junctions constantly. This is the bug fix category winning again.

### Parallel Cycle 15: P41/P42/P43 exploration+weight (April 15, ~12:00 AM)
- **Format**: Exploration optimization + frontier weight bracketing on v13
- **Baseline VOR**: 1.10 (5ep, v13)
- **V1 (P41) Hub-proximate exploration (20 cell cap)**: VOR 0.98 (-10.9%) — FAILED. Heartless exploration is load-bearing. Don't restrict range.
- **V2 (P42) Frontier weight 12.0**: VOR 1.02 (-7.3%) — FAILED. Too much frontier bias = too far travel.
- **V3 (P43) Frontier weight 4.0**: VOR 1.06 (-3.6%) — FAILED. Too little frontier = suboptimal networks.
- **Decision**: No changes. Frontier weight 8.0 CONFIRMED OPTIMAL (bracketed). v13 stands.
- **Learning**: Frontier weight 8.0 is the sweet spot. Heartless exploration range is critical for discovery. Both confirmed — stop tweaking these parameters.
- **15 cycles. 43 hypotheses. 11 consecutive failed. INCREMENTAL APPROACH HAS HIT BEDROCK.**

### Parallel Cycle 14: P38/P39/P40 DIAGNOSTIC (April 14, ~11:30 PM)
- **Format**: Diagnostic cycle — testing fundamentally different approaches to identify bottleneck
- **Baseline VOR**: 1.10 (5ep, v13)
- **V1 (P38) Clips junction priority (+10 bonus)**: VOR 1.06 (-3.6%) — FAILED. Few clips junctions vs random opponents. Bonus sends aligners chasing rare targets.
- **V2 (P39) Distance-only targeting (no frontier)**: VOR 1.07 (-2.7%) — FAILED. **Confirms frontier scoring IS helping.** Contiguous network growth > greedy nearest.
- **V3 (P40) 4 miners + 4 aligners**: VOR 0.68 (-38%) — **CATASTROPHIC. Hearts are NOT the bottleneck. ALIGNMENT CAPACITY is.** Losing 1 aligner costs far more than gaining 1 miner.
- **Decision**: No changes. v13 stands.
- **CRITICAL LEARNING**: With P12 fixing heart production, the NEXT bottleneck is ALIGNER THROUGHPUT. Focus on making each alignment cycle faster/more effective, NOT on getting hearts faster.
- **Diagnostic conclusions**: (1) Frontier scoring is correct, keep it. (2) 3M/5A is near-optimal. (3) Offense (clips targeting) doesn't help vs random.
- **14 cycles completed. 40 hypotheses tested. P12 + P18 survive.**

### Parallel Cycle 13: P35/P36/P37 race (April 14, ~11:00 PM)
- **Format**: Bug fix + coordination improvements on v13 base (P12 + P18)
- **Baseline VOR**: 1.10 (5ep, v13)
- **V1 (P35) Shared out-of-range tracking**: VOR 1.02 (-7.3%) — REGRESSION. Shared blacklists hurt because reachability is agent-specific.
- **V2 (P36) Visible junction claim check**: VOR 1.05 (-4.5%) — REGRESSION. Claim check forces fallthrough to distant off-screen targets. Natural "grab nearest" resolution is better.
- **V3 (P37) Faster hub retreat (threshold 5→3)**: VOR 1.14 (+3.6%) — neutral. Right direction but stuck events too infrequent to matter.
- **Decision**: No changes. v13 stands.
- **Learning**: Individual blacklists are CORRECT (reachability is position-dependent). Visible-junction targeting SHOULD bypass deconfliction (speed > coordination at close range). Hub retreat threshold is not a bottleneck.
- **13 cycles completed. 37 hypotheses tested. P12 + P18 survive. PLATEAU CONFIRMED — 8 consecutive neutral/failed cycles since P12.**

### v13: Revert P29 stuck recovery (April 14, ~10:30 PM)
- **Reason**: Leaderboard showed v12 (P29) scored **9.92 VOR** vs v11's **11.33**. THIRD leaderboard regression from locally positive changes.
- **Pattern confirmed**: P4 (+10% local → -23% tourney), P11 (+21% → -28%), P29 (+10% → -12%). ALL three were changes that help vs random but expose you vs real opponents.
- **Root cause**: When stuck in tournament (opponents contesting territory), hub retreat provides healing, resource management, and strategic reset. Skipping hub means staying in hostile territory longer.
- **Change**: Reverted P29. softy.py back to v11 code (P12 bottleneck miner + P18 BFS closure fix).
- **Uploaded**: Softy:v13
- **RULE STRENGTHENED**: Not just "no behavior tuning" — any change that REDUCES hub visits hurts in tournament. Hub provides critical safety and reset functionality under competitive pressure.

### Parallel Cycle 12: P33b/P34 race (April 14, ~10:00 PM)
- **Format**: Capability-focused improvements on v12 base (P29 applied — now reverted)
- **Baseline VOR**: 1.19 (5ep, v12)
- **V1 (P33b) Cached frontier scoring for visible junctions**: VOR 1.24 (+4.2%) — below 5% threshold. Identical to P33 without caching — computation overhead wasn't the issue. The visible-junction frontier scoring improvement itself is marginal.
- **V2 (P34) Mine near cogs junctions**: VOR 1.22 (+2.5%) — below 5% threshold. Scoring extractors by junction proximity doesn't meaningfully change which extractor is closest.
- **Decision**: No changes. Both below threshold.
- **Learning**: Visible junction frontier scoring (P33/P33b) is a real but small improvement (~4%). Mining near junctions doesn't help because extractors near junctions are usually already the closest ones. Need BIGGER improvements.
- **12 cycles completed. 34 hypotheses tested. P12 + P18 survive on leaderboard. P29 REVERTED.**

### Parallel Cycle 11: P30/P31/P32 race (April 14, ~8:45 PM)
- **Format**: Capability-focused improvements on v12 base (P29 applied)
- **Baseline VOR**: 1.19 (5ep, v12)
- **V1 (P30) A* pathfinding**: VOR 1.26 (5ep) / 1.21 (10ep, +1.7%) — neutral. Partial wall map not enough for A* benefit.
- **V2 (P31) Visited area tracking**: VOR 0.41 (-66%) — CATASTROPHIC. All agents converge on same "unexplored" sector.
- **V3 (P32) Align-range visible check**: VOR 1.11 (-8.3%) — FAILED. Preemptive range-filtering rejects junctions about to become in-range.
- **Decision**: No changes. v12 stands.
- **Learning**: Pathfinding improvements need COMPLETE wall data to help (partial maps don't). Exploration coordination is very hard — deterministic sector selection causes clustering. Dynamic network growth means static filtering loses opportunities (timeout-blacklist > preemptive check).
- **11 cycles completed. 32 hypotheses tested. P12 + P18 + P29 survive.**

### Parallel Cycle 10: P27/P28/P29 race (April 14, ~8:00 PM)
- **Format**: Capability-focused improvements on v11 base
- **Baseline VOR**: 1.10 (5ep)
- **V1 (P27) Scramble detection**: VOR 1.11 (+0.9%) — neutral. Random opponents don't scramble.
- **V2 (P28) Heart coordination**: VOR 1.01 (-8.2%) — FAILED. Too restrictive, stale heart data causes missed opportunities.
- **V3 (P29) Smarter stuck recovery**: VOR 1.20 (5ep) / **1.21 (10ep confirmed, +10%)** — **WINNER**
- **Decision**: KEEP V3 (P29). Applied to softy.py. Ready for upload.
- **Learning**: Travel time is a major efficiency sink. Avoiding unnecessary hub retreats saves 20-40 ticks per stuck event. Conditional hub trips (only when functionally needed) is the right approach. Heart coordination was too aggressive — multiple aligners CAN benefit from simultaneous hub visits.
- **10 cycles completed. 29 hypotheses tested. P12 + P18 + P29 survive.**

---

## April 13-14 overnight (autonomous loop)

### v1 (April 13, ~8:32 PM)
- Initial upload, starter template
- Qualifying only (retired)

### v1 → v2 (April 13, ~9:21 PM)
- **Changes**: Hub-relative coords, role system, observation parsing, BFS navigation
- **Leaderboard**: 2.16 VOR, rank 333
- **Local VOR**: 1.13

### v2 → v3 (April 14, ~12:05 AM)
- **Changes**: Bugfixes (aligner stuck-loop, heartless cycle, deposit threshold)
- **Leaderboard**: 3.43 VOR, rank 289 (+59%)
- **Local VOR**: 0.74 (then engine update, recalibrated to 0.30)

### v3 → v4 (April 14, ~2:11 AM)
- **Changes**: Hub inventory awareness (+117% local VOR), `last_action_move` stuck detection, 3M/5A/0S role split, miner element preference, frontier-aware junction targeting
- **Leaderboard**: 4.36 VOR, rank 257 (+27%)
- **Local VOR**: 0.71

### v4 → v5 (April 14, ~3:28 AM)
- **Changes**: Visible-only junction deposits, minor tuning
- **Leaderboard**: 6.03 VOR, rank 224 (+38%)
- **Local VOR**: 0.73

### Benchmark: April 14, 11:55 AM (recovery session)
- **Softy VOR (local)**: 0.81 (3 episodes vs random)
- **8v0 score**: 1.56
- **Note**: Local VOR vs random is much lower than tournament VOR

### v8: Revert P4 night energy (April 14, ~3:00 PM)
- **Reason**: Leaderboard showed v6 (P4+P12 precursors) scored 4.64, LOWER than v5 (6.03). Night energy conservation makes agents idle during night, giving opponents free territory. P4 helps locally vs random but hurts vs real opponents.
- **Change**: Reverted `_low_energy()` to simple `energy < 2` check. No night awareness.
- **Local VOR**: 1.11 (10ep). Essentially same as v7's 1.14 — P12 does all the work.
- **Uploaded**: Softy:v8
- **Key insight**: LOCAL BENCHMARKS vs RANDOM are unreliable predictors of TOURNAMENT performance. Must check leaderboard results and be willing to revert locally-positive changes that hurt on the board.

### v11: Apply P18 BFS closure fix (April 14, ~6:15 PM)
- **Change**: `_in_align_range` now uses BFS network closure instead of raw cogs junctions. Prevents targeting junctions near disconnected cogs junctions after cascade failure.
- **Local impact**: 0% (exactly neutral vs random — random opponents rarely cause cascade failure)
- **Rationale**: Bug fix, not behavior tuning. Zero local risk. Should help in tournament where cascade failure is common.
- **Uploaded**: Softy:v11

### Parallel Cycle 8: P18/P21/P24 race (April 14, ~6:00 PM)
- **Format**: Capability-focused improvements on v10 base
- **Baseline VOR**: 1.09 (5ep, v10)
- **V1 (P18) BFS closure fix**: VOR 1.09 (0%) — perfectly neutral. Applied to v11 as zero-risk bug fix.
- **V2 (P21) Late-game scrambler**: VOR 1.03 (-5.5%) — FAILED. Mid-game miner switch disrupts economy despite threshold check.
- **V3 (P24) Wall tracking**: TBD — still running.
- **8 cycles completed. P18 applied as bug fix. P21 scrambler approach abandoned (3rd failure for defense via role changes).**

### v10: Revert P11 game-phase adaptation (April 14, ~5:45 PM)
- **Reason**: Leaderboard showed v9 (P11) scored **7.51 VOR** vs v7/v8's **10.52**. SECOND leaderboard regression from locally positive changes.
- **Pattern**: P4 (night energy) +10% local / -23% tournament. P11 (game-phase) +21% local / -28% tournament. P12 (bottleneck) +46% local / +75% tournament.
- **Root cause**: Behavior modifications that help vs random get EXPLOITED by real opponents. P11's aggressive early expansion sends aligners too far from base. P11's late consolidation gives up territory. Only CAPABILITY improvements (P12 = more heart production) translate to tournament.
- **Change**: Reverted all P11 changes. softy.py back to v7/v8 code (P12 bottleneck miner only).
- **Uploaded**: Softy:v10
- **CRITICAL RULE**: Stop testing behavior tuning (scoring weights, timing changes, phase adaptation). Only test CAPABILITY improvements (new abilities, bug fixes, efficiency gains).

### Parallel Cycle 7: P21/P22/P23 race (April 14, ~5:30 PM)
- **Note**: Run against v9 baseline which itself was a regression. Results less meaningful.
- **V1 (P21) Late-game scrambler**: VOR 1.14
- **V2 (P22) Octant exploration**: VOR 1.15
- **V3 (P18+P19 combo)**: VOR 1.14
- **Decision**: All neutral. Obsoleted by P11 reversion.
- **7 cycles completed. 24 hypotheses tested. Only P12 survives on leaderboard.**

### Parallel Cycle 6: P18/P19/P20 race (April 14, ~5:00 PM)
- **Format**: Parallel test — 3 strategic improvements on v9 baseline
- **Baseline VOR**: 1.22 (10ep, v9)
- **V1 (P18) BFS network closure**: VOR 1.14 (+1.8% at 5ep) — neutral. Fix is correct but random opponents rarely cause cascade failure. May help in tournament.
- **V2 (P19) Wave expansion bonus**: VOR 1.19 (5ep) / 1.28 (10ep, +4.9%) — just below threshold. Clustering aligners helps marginally.
- **V3 (P20) Soft clips bonus**: VOR 1.17 (+4.5% at 5ep) — neutral. Even reduced bonus distorts phase scoring.
- **Decision**: No changes. v9 stands.
- **Learning**: Junction targeting improvements are hitting diminishing returns on local benchmarks. The frontier scoring + game-phase adaptation is already quite good vs random. Next improvements likely need to target behaviors that matter in TOURNAMENT (defense, opponent-awareness) or fundamental capability gaps (map discovery, pathfinding efficiency).
- **6 cycles completed. 21 hypotheses tested. 3 kept (P12 bottleneck miner, P11 game-phase). P18 BFS fix noted for potential tournament value.**

### v9: P11 game-phase adaptation (April 14, ~4:30 PM)
- **Leaderboard context**: v7 scored **10.52 VOR, rank 80** (19 matches). P12 translated +75% to tournament (vs +46% local). 
- **Change**: Game-phase adaptation in junction scoring and exploration radius:
  - Early (<3000): frontier_weight=12, dist_weight=0.7, sector_radius=45 (aggressive expansion)
  - Mid (3000-6000): frontier_weight=8, dist_weight=1.0, sector_radius=30 (balanced)
  - Late (>6000): frontier_weight=4, dist_weight=1.5, sector_radius=20 (consolidation)
- **Local VOR**: 1.26 (10ep) vs baseline 1.04 (+21%)
- **Uploaded**: Softy:v9
- **Key insight**: Phase-adjusted scoring is high-value. Early aggressive expansion discovers more territory; late consolidation maximizes contiguous network reward.

### Parallel Cycle 5: P11/P16/P17 race (April 14, ~4:00 PM)
- **Format**: Parallel test — 3 strategic improvements
- **Baseline VOR**: 1.04 (5ep)
- **V1 (P11) Game-phase adaptation**: VOR 1.12 (+7.7%) → **confirmed 1.26 (10ep, +21%)** — **WINNER**
- **V2 (P16) Exploration sweep**: VOR 1.00 (-3.8%) — FAILED. Sending heartless aligners to map edges wastes time vs nearby discovery.
- **V3 (P17) Clips junction priority**: VOR 1.10 (+5.8%) — promising solo, but combo with P11 was only 1.13 (interference)
- **V4 (P11+P17 combo)**: VOR 1.13 — clips bonus overrides phase-adjusted distance, sending aligners to distant clips junctions. Don't stack.
- **Decision**: KEEP V1 (P11). Uploaded as Softy:v9.
- **Learning**: Game-phase adaptation is the second biggest improvement after P12. Early aggressive expansion with higher frontier weight (12.0 vs 8.0) and lower distance penalty (0.7x) discovers more junctions. Late consolidation (4.0 frontier, 1.5x distance) prevents overextension. Clips priority interferes because the +12 bonus overrides phase-adjusted scoring.
- **5 cycles completed. 18 hypotheses tested. 3 kept (P4 night energy reverted, P12 bottleneck miner, P11 game-phase).**
- **Discovery**: `_in_align_range` has a bug — checks ALL cogs junctions instead of net-connected only. Filed as P18 for next cycle.

### Parallel Cycle 4: P13/P14/P15 race (April 14, ~2:30 PM)
- **Format**: Parallel test — 3 mechanical improvements
- **Baseline VOR**: 1.14 (v7)
- **V1 (P13) Failed junction expiry**: VOR 1.10 (-3.5%) — clearing blacklist hurts, junctions stay unreachable
- **V2 (P14) Early deposit threshold 15**: VOR 1.05 (-7.9%) — FAILED. More trips = less mining. P12 bottleneck miner already solved early heart supply.
- **V3 (P15) Aligner timeout 25**: VOR 1.17 (+2.6%) — neutral, slight positive. Longer patience helps marginally.
- **Decision**: No changes. v7 stands.
- **Learning**: P12's bottleneck miner already fixed the economy pipeline so thoroughly that P14 (faster deposits) became counterproductive — more trips just waste mining time. P13's blacklist clearing re-exposes aligners to unreachable junctions. The 15-tick timeout (P15) may be slightly too aggressive but 25 isn't enough improvement to justify.
- **4 cycles completed. 12 hypotheses tested. 2 kept (P4 night energy + P12 bottleneck miner).**

### Parallel Cycle 3: P9/P10/P12 race (April 14, ~2:00 PM)
- **Format**: Parallel test — 3 deep-research hypotheses
- **Baseline VOR**: 0.78 (v6)
- **V1 (P9) Net:cogs tracking**: VOR 0.80 (+2.6%) — neutral, connectivity bonus too weak
- **V2 (P10) Ship tracking**: VOR 0.72 (-7.7%) — FAILED. Ship threat penalty too aggressive, steers aligners away from good targets.
- **V3 (P12) Miner bottleneck element**: VOR 1.05 (5ep) / **1.14 (10ep confirmed)** — **+46% MASSIVE WIN**
- **Decision**: KEEP V3 (P12). Uploaded as Softy:v7.
- **Learning**: The biggest bottleneck was HEART PRODUCTION. Static miner element assignment meant one element would lag, stalling heart crafting for all aligners. Dynamic bottleneck targeting ensures balanced element delivery → continuous heart supply → more alignments per game. This is the single biggest improvement since the initial architecture.

### Parallel Cycle 2: P5/P6/P8 race (April 14, ~1:30 PM)
- **Format**: Parallel test — 3 variants benchmarked simultaneously
- **Baseline VOR**: 0.78 (v6, 10-episode confirmed)
- **V1 (P5) Early heart rush**: VOR 0.73 (-6.4%) — FAILED. Always-to-hub early game hurts exploration.
- **V2 (P6) Faster heartless cycling**: VOR 0.80 (+2.6%) — Neutral, below 5% threshold. Positive direction.
- **V3 (P8) Dynamic role switch**: VOR 0.72 (-7.7%) — FAILED. Late-game aligner conversion hurts economy.
- **Decision**: No changes. All hypotheses P1-P8 now tested. Need fresh hypotheses.
- **Learning**: Early-game behavior is already well-tuned. Forcing aligners to hub wastes exploration time. Miners converting to aligners late-game costs resources (gear switch) and the late-game alignment rate isn't worth it. Exploration cycling speed (8→5) shows marginal benefit — may be worth fine-tuning.

### Parallel Cycle 1: P2/P3/P4 race (April 14, ~1:00 PM)
- **Format**: First parallel test — 3 variants benchmarked simultaneously
- **Baseline VOR**: 0.71 (5 episodes)
- **V1 (P2) Network redundancy**: VOR 0.73 (+2.8%) — neutral, not kept
- **V2 (P3) Clips timing**: VOR 0.76 (+7.0%) — promising, queued for later
- **V3 (P4) Night energy conservation**: VOR 0.77 (+8.5%) — **WINNER**
- **Combo (P3+P4)**: VOR 0.72 (+1.4%) — interference, don't stack
- **10-episode confirmation**: VOR 0.78 — confirmed
- **Decision**: KEEP V3 (P4). Uploaded as Softy:v6.
- **Learning**: Night energy conservation is high-value. Clips timing helps solo but interferes with energy conservation (both modify movement patterns). Network redundancy bonus too small — junction selection already decent.

### P1 scrambler test (April 14, ~12:30 PM)
- **Change**: ROLE_CYCLE 3M/5A → 2M/1Sc/5A
- **Baseline VOR**: 0.77 (5 episodes vs random)
- **Test VOR**: 0.44 (5 episodes vs random, -43%)
- **Decision**: REVERTED. Heart economy too damaged by losing a miner.
- **Learning**: Defense via scramblers costs too much economy. Need network resilience through smarter junction selection (P2) or timing exploitation (P3) instead.

---
