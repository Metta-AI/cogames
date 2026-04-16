# Failed Experiments — DO NOT RETRY

## Multi-heart collection (HEART_COLLECT_TARGET=3)
- **What**: Aligners wait for 3 hearts before leaving hub
- **Result**: VOR dropped to 0.17
- **Why failed**: Deadlock — multiple aligners compete for same hearts, all waiting forever
- **Never retry because**: fundamental resource contention with shared hub

## Junction deposit at aligned junctions (off-screen)
- **What**: Miners navigate to known cogs junctions to deposit instead of hub
- **Result**: VOR 0.74 → 0.32
- **Why failed**: Junctions get scrambled by Clips during miner transit; resources lost
- **Never retry because**: scramble timing makes off-screen junction deposits unreliable
- **Note**: visible-only junction deposits (on-screen, confirmed cogs) DO work (v5)

## Phased startup (30-tick delay for agents 5-7)
- **What**: Stagger gear acquisition to reduce hub contention
- **Result**: VOR 0.71 → 0.65
- **Why failed**: 30-tick delay costs early junction captures; tick value is front-loaded
- **Never retry because**: any delay in early game directly reduces cumulative reward

## Visible junction deconfliction (strict)
- **What**: Aligners avoid junctions visible to other aligners
- **Result**: VOR 0.69 → 0.48
- **Why failed**: Too restrictive — aligners sit idle when all visible junctions are "claimed"
- **Never retry because**: observation window is only 13x13; too few junctions visible

## 4 miners / 4 aligners
- **What**: Add a 4th miner, remove 1 aligner
- **Result**: Same as 3M/5A
- **Why failed**: Extra miner doesn't offset lost aligner output
- **Never retry because**: 3 miners already saturate heart crafting pipeline

## Deposit threshold 20 (reduced from 30)
- **What**: Miners deposit earlier with less cargo
- **Result**: Minor regression
- **Why failed**: More travel trips per resource collected
- **Never retry because**: higher threshold = fewer trips = more mining time

## P1: Add scrambler (2M/1Sc/5A composition)
- **What**: Replace 1 miner with 1 scrambler to counter clips cascade failure
- **Result**: VOR 0.44 vs baseline 0.77 (-43% regression)
- **Why failed**: Losing a miner cripples the heart economy — fewer hearts means fewer alignments. The scrambler role works mechanically but the economic tradeoff isn't worth it at 8-agent team size.
- **Never retry because**: 2 miners can't sustain the heart pipeline for 5 aligners. Would need a way to add scrambling without losing a miner (e.g., dynamic role switch mid-game, or scrambler that also mines).
- **Date**: April 14, 2026

## P5: Early-game heart rush (always go to hub first 200 ticks)
- **What**: Heartless aligners always go to hub in first 200 ticks instead of exploring
- **Result**: VOR 0.73 vs baseline 0.78 (-6.4%)
- **Why failed**: Forces all aligners to hub simultaneously, causing congestion. Exploration during heartless periods discovers junctions that are needed later.
- **Never retry because**: Early exploration is load-bearing — it populates the coordinator's junction map
- **Date**: April 14, 2026

## P8: Dynamic role switch (miners → aligners at tick 3000)
- **What**: After tick 3000, miners deposit cargo and start running aligner logic
- **Result**: VOR 0.72 vs baseline 0.78 (-7.7%)
- **Why failed**: Gear switch costs hub resources. Late-game miners still need to mine to sustain heart crafting. Converting removes economic engine.
- **Never retry because**: Economy needs continuous mining throughout the game. Heart pipeline stalls without miners.
- **Date**: April 14, 2026

## P13: Failed junction expiry (clear blacklist every 200 ticks)
- **What**: Clear `failed_junctions` set every 200 ticks to allow retargeting
- **Result**: VOR 1.10 vs baseline 1.14 (-3.5%)
- **Why failed**: Junctions that were blacklisted were genuinely unreachable (walls, out of range). Clearing the blacklist sends aligners back to waste time on them again.
- **Never retry because**: The blacklist is accurate. Junctions don't move — if you couldn't reach it once, you likely can't reach it 200 ticks later.
- **Date**: April 14, 2026

## P14: Early deposit threshold (15 before tick 1000)
- **What**: Lower miner deposit threshold from 30 to 15 in first 1000 ticks
- **Result**: VOR 1.05 vs baseline 1.14 (-7.9%)
- **Why failed**: With P12 bottleneck miner already balancing elements, early deposits just increase travel overhead. More trips = less time mining = fewer total resources.
- **Never retry because**: P12 solved the heart production bottleneck; depositing earlier is now counterproductive
- **Date**: April 14, 2026

## P21: Late-game miner→scrambler conversion (miner 0 at tick 5000+)
- **What**: After tick 5000, if hub has 40+ of each element, miner 0 switches to scrambler logic
- **Result**: VOR 1.03 vs baseline 1.09 (-5.5%)
- **Why failed**: Third attempt at adding defense via role changes. Even with late-game timing and economy threshold, losing a miner disrupts the bottleneck element balance. 2 miners can't maintain balanced element supply for heart crafting.
- **Never retry because**: Defense via role changes has failed 3 times (P1 scrambler, P8 role switch, P21 late scrambler). The fundamental issue is 3 miners is the minimum for balanced heart production. Need defense without losing miners.
- **Date**: April 14, 2026

## P16: Aggressive exploration sweep (heartless aligners navigate to map edges)
- **What**: Heartless aligners compute centroid of known junctions and navigate away from it to discover new junctions
- **Result**: VOR 1.00 vs baseline 1.04 (-3.8%)
- **Why failed**: Sending aligners to distant map edges wastes time. Nearby wandering discovers junctions more efficiently because junctions cluster. Map-edge exploration means long travel for sparse returns.
- **Never retry because**: Local sector wandering IS the optimal exploration strategy. Junctions near hub are higher value (closer to network).
- **Date**: April 14, 2026

## P27: Scramble detection + priority re-alignment
- **What**: Detect cogs→neutral transitions, give +20 score bonus to recently-scrambled junctions
- **Result**: VOR 1.11 vs baseline 1.10 (+0.9%) — neutral
- **Why failed**: Random opponents rarely scramble cogs junctions. May help in tournament but not enough signal locally to keep.
- **Never retry because**: The scramble detection code works, but the bonus doesn't help vs random. Could revisit if tournament data shows cascade failure is a problem.
- **Date**: April 14, 2026

## P31: Visited area tracking for exploration guidance
- **What**: Track visited cells in coordinator, bias wander toward sectors with fewest visited cells (sample 5 points along each sector direction)
- **Result**: VOR 0.41 vs baseline ~1.21 (-66%) — CATASTROPHIC
- **Why failed**: The sector scoring (sampling 5 points per direction) completely disrupted exploration. Agents clustered toward the same "unexplored" sectors simultaneously, causing massive congestion. The sampling also added computational overhead to every wander call.
- **Never retry because**: Deterministic sector scoring causes all agents to converge on the same sector. Would need per-agent randomization that defeats the purpose of coordination.
- **Date**: April 14, 2026

## P32: Align-range check for visible junctions
- **What**: Add `_in_align_range()` check when targeting visible junctions. Previously visible junctions skipped the range check, causing aligners to waste 15 ticks on out-of-range junctions.
- **Result**: VOR 1.11 vs baseline ~1.21 (-8.3%) — REGRESSION
- **Why failed**: The network is dynamic — junctions that are currently out of alignment range may become in-range within a few ticks as other aligners capture nearby junctions. The 15-tick timeout + blacklist approach is BETTER because it gives the network time to grow. Preemptive filtering loses these "about to be reachable" junctions.
- **Never retry because**: Dynamic network growth means static range checks reject too many viable targets. The timeout-blacklist mechanism is the correct approach.
- **Date**: April 14, 2026

## P28: Heart-count-aware aligner dispatch
- **What**: Limit simultaneous aligner hub trips based on available hearts + craftable hearts
- **Result**: VOR 1.01 vs baseline 1.10 (-8.2%) — REGRESSION
- **Why failed**: Too restrictive. Aligners that can't claim a heart trip just wander instead of heading to hub. Heart count from observations may be stale. Multiple aligners CAN benefit from going to hub simultaneously (different hearts, crafting while waiting).
- **Never retry because**: Heart availability changes rapidly; gating based on snapshot data causes missed opportunities.
- **Date**: April 14, 2026

## P33/P33b: Frontier scoring for visible junctions
- **What**: Apply the same frontier scoring (frontier * 8.0 - dist) to visible junctions, which currently use distance-only _closest. P33b added BFS caching optimization.
- **Result**: VOR 1.24 vs baseline 1.19 (+4.2%) — below 5% threshold
- **Why failed**: The improvement is real but small. Most visible junctions that _closest picks are also the ones frontier scoring would pick. The inconsistency between visible and off-screen targeting rarely matters.
- **Never retry because**: Tested twice (with and without caching). The ceiling is ~4%, not enough to keep.
- **Date**: April 14, 2026

## P34: Mine near cogs junctions (scored extractor selection)
- **What**: Score extractors by distance + proximity to cogs junctions (+10 bonus within 15 cells) and hub (+8 within 20 cells)
- **Result**: VOR 1.22 vs baseline 1.19 (+2.5%) — below 5% threshold
- **Why failed**: Nearest extractor is usually already the best one. Junction proximity bonus rarely changes the selection because extractors near junctions are typically already closest.
- **Never retry because**: Marginal at best. Mining location isn't the bottleneck.
- **Date**: April 14, 2026

## P29: Smarter stuck recovery (conditional hub skip) — REVERTED from tournament
- **What**: When stuck, check if hub trip is functionally needed. If not, just rotate sector and wander instead of retreating to hub. Saves 20+ ticks of unnecessary travel per stuck event.
- **Local result**: VOR 1.21 vs baseline 1.10 (+10%) — KEPT as v12
- **Tournament result**: v12 scored 9.92 vs v11's 11.33 — **-12.4% REGRESSION**
- **Why failed in tournament**: Hub retreat when stuck provides critical benefits vs real opponents: healing (stay in territory), strategic reset (new targets from central position), resource management. Skipping hub means staying in contested territory where opponents apply pressure.
- **Never retry because**: Third case of locally-positive / tournament-negative. Same pattern as P4 (+10%/−23%) and P11 (+21%/−28%). Any change that reduces hub visits hurts under competitive pressure.
- **Date**: April 14, 2026

## P38: Clips junction priority (+10 bonus, unified scoring)
- **What**: Merge clips junctions into main scoring with +10 bonus (flipping clips→cogs worth 2 points). Remove separate enemy fallback.
- **Result**: VOR 1.06 vs baseline 1.10 (-3.6%)
- **Why failed**: Random opponents barely capture junctions — few clips targets exist. The +10 bonus sends aligners hunting rare clips junctions instead of abundant neutral ones.
- **Note**: May help in tournament where clips is active. But can't verify safely (local regression is a red flag).
- **Date**: April 14, 2026

## P39: Distance-only targeting (remove frontier scoring) — DIAGNOSTIC
- **What**: Remove `frontier * 8.0` from junction scoring, use pure `-dist`
- **Result**: VOR 1.07 vs baseline 1.10 (-2.7%)
- **Why important**: CONFIRMS frontier scoring IS helping. Contiguous network growth (frontier bonus) produces better networks than greedy nearest-junction selection.
- **Never retry because**: This PROVES frontier scoring is correct. Keep it.
- **Date**: April 14, 2026

## P40: 4 miners + 4 aligners — DIAGNOSTIC
- **What**: Change ROLE_CYCLE from 3M/5A to 4M/4A with per-element miner assignment
- **Result**: VOR 0.68 vs baseline 1.10 (-38%) — CATASTROPHIC
- **Why important**: PROVES hearts are NOT the bottleneck (P12 fixed that). ALIGNMENT CAPACITY is the bottleneck. Losing 1 aligner (-20% capacity) hurts far more than gaining 1 miner (+33% heart production).
- **Never retry because**: 3M/5A is near-optimal. Don't add miners. Focus on making aligners more effective.
- **Date**: April 14, 2026

## P35: Shared out-of-range junction tracking
- **What**: Coordinator-level `out_of_range_junctions` set. When aligner times out on a junction and `_in_align_range` returns False, add to shared set. All agents skip these. Clear when new cogs junction captured.
- **Result**: VOR 1.02 vs baseline 1.10 (-7.3%)
- **Why failed**: Junction reachability is agent-SPECIFIC (depends on position/path), not just range-based. Agent A failing doesn't mean agent B would fail from a different direction. Clearing on every capture also flushes genuinely unreachable junctions.
- **Never retry because**: Individual per-agent blacklists are the correct design. Sharing junction failure data across agents with different positions is fundamentally wrong.
- **Date**: April 14, 2026

## P36: Visible junction claim check
- **What**: Add `is_claimed` check + `claim_target` for visible junction targeting in aligner
- **Result**: VOR 1.05 vs baseline 1.10 (-4.5%)
- **Why failed**: The visible-junction path is intentionally lightweight ("grab nearest"). Adding claim checks forces fallthrough to the off-screen targeting path which picks FARTHER junctions. Natural resolution (first aligner wins, second blacklists after 15 ticks) is better than routing to distant alternatives.
- **Never retry because**: Visible-junction targeting should bypass deconfliction. Speed > coordination at close range.
- **Date**: April 14, 2026

## P37: Faster hub retreat (stuck threshold 5→3)
- **What**: Lower severe stuck threshold from stuck_count > 5 to > 3
- **Result**: VOR 1.14 vs baseline 1.10 (+3.6%) — below 5% threshold, neutral
- **Why neutral**: Stuck events are infrequent. Saving ~2 ticks per event doesn't compound enough.
- **Status**: Direction correct (hub safety) but effect too small to keep.
- **Date**: April 14, 2026

## Aligner target timeout 15 → 10
- **What**: Reduce junction stuck threshold
- **Result**: Neutral (no improvement)
- **Why failed**: 10 ticks too aggressive — abandonsjunctions that just needed 2 more ticks
- **Status**: Not harmful, just not helpful. Could retry at 12.

## P38: Clips junction priority — TOURNAMENT REGRESSION
- **What**: Unified clips+neutral junction scoring with +10 clips bonus in `nearest_alignable_junction`
- **Local result**: VOR 1.06 vs baseline 1.10 (-3.6%)
- **Tournament result**: v14 scored 5.34 (2 matches) — catastrophic regression from v11's 11.33
- **Why failed**: Clips bonus distorts frontier scoring. Random opponents don't capture many junctions, so bonus chases ghosts. In tournament, sends aligners toward contested enemy territory instead of safe expansion.
- **Never retry because**: Fourth tournament regression from local-neutral change. Offensive junction targeting doesn't work.
- **REVERTED**: v15 = P12 + P18 only (clean revert)
- **Date**: April 14, 2026

## P13: Junction expiry (on v17 base)
- **What**: Expire failed junction blacklist entries after 200 ticks
- **Result**: VOR 1.16 vs baseline 1.23 (-5.7%)
- **Why failed**: On v17's neutral-only targeting, blacklist is small and mostly correct. Expiry re-enables genuinely unreachable junctions.
- **Date**: April 14, 2026

## P54: Miner deposit threshold 30→20
- **What**: Lower miner cargo deposit threshold to get fresher bottleneck data
- **Result**: VOR 1.12 vs baseline 1.23 (-8.9%)
- **Why failed**: 50% more hub trips. Travel overhead far exceeds benefit of fresher data.
- **Never retry because**: 30 confirmed optimal. Lower = too much travel.
- **Date**: April 14, 2026

## P55: Aligner target spacing penalty
- **What**: -5 penalty per nearby aligner within 10 cells in junction scoring
- **Result**: VOR 1.18 vs baseline 1.23 (-4.1%)
- **Why failed**: Steers aligners to suboptimal junctions. Distance term already provides implicit spacing.
- **Date**: April 14, 2026

## P49: Sector radius 30→45
- **What**: Increase heartless exploration sector radius
- **Result**: VOR 1.23 vs baseline 1.20 (+2.5%) — NEUTRAL
- **Date**: April 14, 2026

## P51: Opportunistic hub check near hub (8 cells)
- **What**: Heartless aligners check hub for hearts when they happen to be within 8 cells
- **Result**: VOR 1.22 vs baseline 1.20 (+1.7%) — NEUTRAL
- **Date**: April 14, 2026

## P53: Pre-select target while heartless
- **What**: Heartless aligners navigate toward their next target junction instead of wandering
- **Result**: VOR 1.18 vs baseline 1.20 (-1.7%)
- **Why failed**: Moves agent away from hub. Same pattern as P41 — increases heart-sourcing lag.
- **Date**: April 14, 2026

## P45: Hub check interval 30→15
- **What**: Heartless aligners check hub every 15 ticks instead of 30
- **Result**: VOR 1.15 vs baseline 1.20 (-4.2%)
- **Why failed**: More hub checks = less exploration time. 20% hubward vs 17%. Net negative.
- **Date**: April 14, 2026

## P46: Distance-adjusted junction timeout
- **What**: Replace fixed 25-tick timeout with `max(15, dist/2)`
- **Result**: VOR 1.06 vs baseline 1.20 (-11.7%)
- **Why failed**: Close junctions get 7-15 tick timeout, undoing P15. Walls make distance ≠ travel time.
- **Never retry because**: Fixed 25 is robust. Dynamic timeout is fundamentally wrong — can't predict wall-based travel time from Manhattan distance.
- **Date**: April 14, 2026

## P48: Visible junction claim write (no check)
- **What**: Write claim_target for visible junction targeting (inform other agents)
- **Result**: VOR 1.18 vs baseline 1.20 (-1.7%)
- **Why failed**: Claims for visible junctions cause off-screen scoring to skip them. But visible alignment is fast — resolved before others target.
- **Date**: April 14, 2026

## P56: Staggered bottleneck mining (miner rank diversity)
- **What**: Each miner targets a different scarce element (rank 0=scarcest, 1=2nd, 2=3rd) instead of all targeting the same bottleneck
- **Result**: VOR 1.01 vs baseline 1.23 (-17.9%)
- **Why failed**: The bottleneck IS the bottleneck — all miners should fix it. Spreading across elements means 2 miners mine non-bottleneck resources, creating the same imbalance P12 originally fixed.
- **Never retry because**: P12's all-on-bottleneck is fundamentally correct. Diversifying is anti-P12.
- **Date**: April 14, 2026

## P59: Clear blacklist on network growth
- **What**: When new cogs junction captured, clear failed_junctions since previously unreachable junctions may now be in alignment range
- **Result**: VOR 1.21 vs baseline 1.23 (-1.6%) — NEUTRAL
- **Why failed**: Clearing entire blacklist on every network growth is too aggressive. Most blacklisted junctions are behind walls or genuinely out of range, not just barely-out-of-network.
- **Note**: A more targeted version (only clear junctions near the NEW cogs junction) might work but complexity isn't worth ~2%.
- **Date**: April 14, 2026

## P44: Scramble detection + re-alignment priority
- **What**: Track cogs→neutral/clips transitions in coordinator. Add +15 bonus for recently-scrambled junctions in scoring.
- **Result**: VOR 1.02 vs baseline 1.07 (-4.7%)
- **Why failed**: Random opponents don't scramble. Detection logic never triggers. Same pattern as P27/P38 — tournament-specific offense is dead weight locally.
- **Note**: Could be uploaded separately as tournament bet, but P38's tournament failure suggests offense doesn't translate either.
- **Date**: April 14, 2026

## P69: net:cogs tag for junction tracking
- **What**: Use `net:cogs` tag to distinguish connected vs disconnected cogs junctions. Only count connected junctions for healing distance and frontier scoring.
- **Result**: VOR 1.24 vs baseline 1.31 (-5.3%)
- **Why failed**: Disconnected cogs junctions still provide territory (HP regen). Excluding them from healing calculations makes retreat distance too large, triggering premature retreat.
- **Never retry because**: Our BFS `net_connected_junctions()` is the correct abstraction for scoring. For healing/territory, ALL cogs junctions matter.
- **Date**: April 14, 2026

## P72: aoe_mask territory-aware retreat
- **What**: Parse `aoe_mask` at center position (0=neutral, 1=friendly, 2=enemy). Skip retreat when in friendly territory (heals +100 HP/tick).
- **Result**: VOR 1.25 vs baseline 1.31 (-4.6%)
- **Why failed**: Retreat serves dual purpose: HP recovery AND hub access (hearts, deposits). Skipping retreat in territory prevents necessary hub trips. Agents stay in field too long without hearts.
- **Never retry because**: Retreat is overloaded — it's the only mechanism that sends agents back to hub for hearts. Can't disable HP-based retreat without alternative hub-visit trigger.
- **Date**: April 14, 2026

## P75: Territory-aware heal distance (nearest_healing minus territory radius)
- **What**: Adjusted `nearest_healing()` to subtract territory radius (10 for junctions, 20 for hub). Agent 8 cells from junction → heal_dist=0 instead of 8.
- **Result**: VOR 1.24 vs baseline 1.31 (-5.3%)
- **Why failed**: Agents stay in field too long with reduced retreat threshold. Territory radius might not match Manhattan distance in the game engine. Also similar to P72 — reducing retreat frequency costs hub visits for hearts.
- **Never retry because**: Same class of error as P72 (territory-based retreat reduction hurts). The retreat mechanism provides incidental hub access. Any retreat threshold reduction costs hearts.
- **Date**: April 14, 2026

## P99: Simple aligner-first role assignment
- **What**: Swap ROLE_CYCLE to aligners first, miners last. With 1-3 agents, all aligners instead of all miners.
- **Result**: VOR 1.62 vs baseline 1.65 (-1.8%)
- **Why failed**: VOR is count-weighted (c8=22.2%, c1=2.8%). c6-c7 scenarios lost a miner (3→2 or 3→1), devastating heart production. Small-team improvements (c2 +11.9%, c3 +10.3%) don't compensate due to low weight.
- **Key insight**: With 6-7 agents, having 3 miners is essential. But static role order can't serve both large and small teams. Need DYNAMIC role assignment.
- **Date**: April 14, 2026

## P97: Hub calibration normalization
- **What**: Pick northwest-most hub cell (min row, then col) for calibration instead of first-found.
- **Result**: VOR 1.59 vs baseline 1.65 (-3.6%)
- **Why failed**: Likely hub is single-cell (change is noop), and the different iteration pattern slightly disrupted calibration timing.
- **Date**: April 14, 2026

## P98: Direct-to-hub when junctions plentiful
- **What**: Heartless aligners go straight to hub when 5+ known neutral junctions exist instead of exploring.
- **Result**: VOR 1.58 vs baseline 1.65 (-4.2%)
- **Why failed**: Same as P45 — exploration during heartless phase is load-bearing. Even with plenty of known junctions, continued exploration discovers MORE junctions beyond the frontier.
- **Never retry because**: Exploration is ALWAYS valuable. Never cut it to save hub travel time.
- **Date**: April 14, 2026

## P93: Clear stale coordinator claims on target loss
- **What**: Clear coordinator claim_target(None) when aligner times out on a junction or goes heartless.
- **Result**: VOR 1.63 vs baseline 1.65 (-1.21%)
- **Why failed**: Same as P60 — stale claims PROTECT other agents from targeting junctions one agent already tried and failed. Claims act as implicit shared blacklist.
- **Never retry because**: Tested twice now (P60 at -4.1%, P93 at -1.21%). Stale claims are beneficial.
- **Date**: April 14, 2026

## P96: Use shared hub resources for hub_can_craft
- **What**: Use `self._coord.last_hub_resources` instead of `s.hub_resources` for heartless aligner hub_can_craft check.
- **Result**: VOR 1.58 vs baseline 1.65 (-4.24%)
- **Why failed**: Coordinator data is stale — set by whichever agent last saw the hub. If hub resources were consumed since then, stale "can craft" data sends aligners on wasted hub trips. Per-agent observation + periodic checks is more reliable.
- **Never retry because**: Stale shared data actively misleads. The periodic 5/30 check drives agents close enough to get fresh data.
- **Date**: April 14, 2026

## P105: Extend dynamic roles to c5/c6 (2M + rest aligners)
- **What**: Change threshold from `>= 5` to `>= 7` so c5 gets 2M/3A and c6 gets 2M/4A instead of ROLE_CYCLE's 3M/2A and 3M/3A.
- **Result**: VOR 1.72 vs baseline 1.71 (+0.6%) — NEUTRAL
- **Why failed**: c6 already has a 50/50 miner/aligner split (3M/3A). Going to 33/67 doesn't help because 3 aligners is adequate. Dynamic roles only help when alignment is severely under-resourced (c2 had 0 aligners, c4 had 1).
- **Never retry because**: The pattern is clear: c2 (0A→1A = +82%), c4 (1A→2A = +9.6%), c6 (3A→4A = +0.6%). Diminishing returns on aligner additions. 3 miners in c6 is the minimum for balanced element production.
- **Date**: April 14, 2026

## P92: Per-node BFS direction in `_go_absolute`
- **What**: Recompute direction priority from each BFS node toward target instead of using fixed direction from agent position.
- **Result**: VOR 1.55 vs baseline 1.59 (-2.52%)
- **Why failed**: Off-screen targets are so distant that direction barely changes across the 13×13 observation window. Fixed direction is effectively optimal.
- **Never retry because**: Observation window is too small for per-node direction to matter. Would need much larger BFS scope.
- **Date**: April 14, 2026

## P86: Miner deposit threshold 30→40
- **What**: Increase mining per trip before depositing.
- **Result**: VOR 1.48 vs baseline 1.41 (+3.5%) — NEUTRAL. Below 5% threshold.
- **Why neutral**: 30 is near-optimal. 20 was -8.9%, 40 is +3.5%. Diminishing returns above 30.
- **Date**: April 14, 2026

## P100: Dynamic role assignment (all team sizes)
- **What**: Dynamically assign roles based on team size: ≤2→0M, 3-4→1M, 5→2M, 6+→3M. Role determined on first step.
- **Result**: VOR 1.64 vs baseline 1.65 (-0.6%) — FAILED (neutral)
- **Why failed**: c2 dropped -16.7% (0M/2A = zero heart production). c5-c6 changed behavior from ROLE_CYCLE (2M/3A for c5 instead of 3M/2A). Count-weighted VOR means c6 regression (-10.5%) dominates c3 gain (+51.5%).
- **Key data**: c3=1.03 (+51.5%) with 1M/2A is massive. But c6=1.71 (-10.5%) with 2M/4A is fatal.
- **Lesson**: Only change ≤3 agent scenarios. Preserve ROLE_CYCLE for 4+ agents exactly.
- **Date**: April 14, 2026

## P136: HP_SAFETY_MARGIN 15→25 (more conservative retreat)
- **What**: Increase retreat threshold to trigger earlier retreat, hoping to reduce 55 deaths/agent
- **Result**: -4.7%, p=0.096 on machina_1.clips
- **Why failed**: More retreat = more travel time + territory abandonment. Deaths NOT reduced (46→50 avg). Deaths are from clips territory disruption, not insufficient retreat.
- **Never retry because**: Retreat timing adjustments CANNOT fix clips-caused deaths. The damage is sudden (territory loss → instant HP drain), not gradual.
- **Date**: April 14, 2026

## P137: Junction heal distance staleness buffer (+10)
- **What**: Add +10 to junction distances in nearest_healing() to account for stale coordinator data after clips scramble
- **Result**: -3.8%, p=0.107 on machina_1.clips
- **Why failed**: Same mechanism as P136. Buffer inflates perceived heal distance → earlier retreat → territory abandonment. The staleness hypothesis was wrong — agents aren't dying because they think healing is closer than it is.
- **Never retry because**: Both P136 and P137 prove retreat-based death reduction is counterproductive. Need different approach entirely.
- **Date**: April 14, 2026

## P101: ALIGN_NET_RADIUS 20→15 (restore actual range)
- **What**: Restore alignment net radius to actual game range (15) from expanded 20.
- **Result**: VOR 1.67 vs baseline 1.65 (+1.2%) — NEUTRAL. Below 3% threshold.
- **Why neutral**: The expanded radius generates false-positive targets, but P89's timeout (50 ticks) handles them gracefully. No harm, no benefit.
- **Date**: April 14, 2026
