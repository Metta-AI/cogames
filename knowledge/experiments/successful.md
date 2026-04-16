# Successful Changes

## Hub inventory awareness (v4)
- **What**: Parse `team:*` global tokens to track hub resources. Aligners check if hub can craft hearts before going.
- **Impact**: Local VOR 0.30 → 0.65 (+117%)
- **Why it works**: Aligners no longer camp at hub waiting for hearts that can't be crafted. They explore and discover junctions instead.
- **Lines**: `SoftyState.hub_resources`, aligner `_aligner()` hub_can_craft check

## `last_action_move` stuck detection (v4)
- **What**: Parse `last_action_move` token. Failed move (+2 stuck). Reduced severe threshold from 8 to 5.
- **Impact**: Contributed to 0.30 → 0.65 jump
- **Why it works**: Agents hitting walls or blocked by others recover faster

## 3M/5A role split (v4)
- **What**: Remove scramblers and scout. 3 miners sustain economy, 5 aligners maximize captures.
- **Impact**: Better than 3M/2S/2A/1Sc baseline
- **Why it works**: Pure alignment at 1 heart/junction beats scramble+align at 2 hearts/junction
- **Caveat**: This was BEFORE discovering cascade failure. May need re-evaluation with scramblers.

## Frontier-aware junction targeting (v4)
- **What**: Score junctions by `frontier * 8.0 - distance` where frontier = neutral junctions that become alignable
- **Impact**: Marginal improvement, kept for directional correctness
- **Why it works**: Prioritizes junctions that unlock new territory

## Dynamic miner bottleneck element targeting (v7)
- **What**: Miners dynamically target the element with lowest hub inventory instead of static assignment. Coordinator tracks hub resources, exposes `bottleneck_element()`.
- **Impact**: Local VOR 0.78 → 1.14 (+46%, 10-episode confirmed). BIGGEST SINGLE IMPROVEMENT.
- **Why it works**: Static element preference (carbon for miner 0, oxygen for miner 1, germanium for miner 2) meant one element would run low while others accumulated. Hearts need 7 of EACH element. One lagging element blocks ALL heart crafting. Dynamic targeting eliminates this bottleneck → continuous heart supply → aligners never idle.
- **Lines**: `SoftyCoordinator.bottleneck_element()`, `SoftyCoordinator.last_hub_resources`, hub resources sync in `_parse()`, `_miner()` Phase 3
- **Key insight**: The economy was the bottleneck all along. Not defense, not pathfinding, not timing — just keeping hearts flowing.

## ~~Night energy conservation (v6)~~ — REVERTED in v8
- **What**: During night ticks, require 5+ energy before moving instead of 2
- **Local impact**: +10% vs random
- **Tournament impact**: v6 scored 4.64 vs v5's 6.03 — NEGATIVE (-23% on leaderboard!)
- **Why it failed on leaderboard**: Idling during night gives real opponents free territory expansion. Random opponents don't exploit this but tournament policies do.
- **REVERTED**: v8 restores simple `energy < 2` check
- **Lesson**: Local vs random benchmarks can mislead. Always verify on leaderboard.

## ~~Game-phase strategy adaptation (v9)~~ — REVERTED in v10
- **What**: Scale junction scoring weights and exploration radius by step_count
- **Local impact**: +21% vs random
- **Tournament impact**: v9 scored 7.51 vs v7/v8's 10.52 — NEGATIVE (-28% on leaderboard!)
- **Why it failed on leaderboard**: Aggressive early expansion sends aligners too far, vulnerable to opponents. Late consolidation gives up territory real opponents contest.
- **REVERTED**: v10 restores original scoring weights
- **Lesson**: Same as P4 — behavior tuning that helps vs random gets exploited in tournament.

## ~~Smarter stuck recovery (v12)~~ — REVERTED in v13
- **What**: Stuck agents check if hub trip is functionally needed. If not, wander instead of retreating.
- **Local impact**: +10% vs random
- **Tournament impact**: v12 scored 9.92 vs v11's 11.33 — NEGATIVE (-12% on leaderboard!)
- **Why it failed on leaderboard**: Hub retreat when stuck provides healing, strategic reset, and resource management under competitive pressure. Skipping hub leaves agents exposed in contested territory.
- **REVERTED**: v13 restores always-retreat-to-hub on severe stuck
- **Lesson**: Third case of "helps vs random, hurts vs real opponents." Hub is a safety mechanism, not just a waypoint.

## Aligner junction timeout 15→25 (v16) — P15
- **What**: Increased aligner target_ticks timeout from 15 to 25 before blacklisting a junction
- **Impact**: Local VOR 1.07 → 1.22 (+14%, 10-episode confirmed). BIGGEST WIN SINCE P12.
- **Why it works**: 15-tick timeout ≈ 4 cells of travel. Most target junctions are 10-30 cells away. Aligners were prematurely blacklisting reachable junctions, wasting hearts on hub return trips and running out of targets. 25 ticks gives them enough time to actually reach distant junctions.
- **Lines**: `_aligner()` target_ticks check
- **Key insight**: Bug fix category wins again. The blacklisting system had TWO bugs: timeout too short (P15) and no expiry (P13, +7.5%). Both confirmed independently.
- **Category**: Bug fix / capability improvement

## Aligner junction timeout 25→35 (v18) — P65
- **What**: Increased aligner target_ticks timeout from 25 to 35 before blacklisting a junction
- **Impact**: Local VOR 1.23 → 1.33 (+8.1%, 10-episode confirmed). Extends P15's logic.
- **Why it works**: At ~0.4 cells/tick (accounting for energy cycles), 25 ticks only covers ~10 cells of travel. Alignment range extends 15 cells from network. Border junctions need 35+ ticks to reach. 45 ticks was neutral (too much waste on genuinely unreachable junctions).
- **Lines**: `_aligner()` target_ticks > 35
- **Key insight**: THIRD consecutive bug fix win (P12, P15, P65). The timeout was STILL too short. Now covers ~14 cells of effective travel, matching the 15-cell alignment radius.
- **Category**: Bug fix (timeout parameter still suboptimal)

## Exploration sector collision fix (v19) — P73
- **What**: Fixed `_explore_offset = agent_id % 4` → `_explore_offset = 0`. Also fixed stuck rotation from `% 4` to `% 8`.
- **Impact**: Local VOR 1.31 → 1.49 (+13.7%, 10-episode confirmed). 5th win overall.
- **Why it works**: The 8-sector exploration system was broken — `(agent_id + agent_id % 4) % 8` produced only 4 unique sectors (0,2,4,6). Agents 0&6, 1&7, 2&4, 3&5 explored the SAME sectors. Fix gives all 8 agents unique sectors → 100% more map coverage → more junctions discovered → more alignments.
- **Lines**: `_explore_offset` in `__init__`, `_update_stuck` rotation
- **Key insight**: 5th consecutive bug fix win. Pattern holds: only bugs/capability fixes translate.
- **Category**: Bug fix (exploration coverage halved by math error)

## BFS wander navigation (v20) — P76
- **What**: Replaced simple 4-direction tries in `_wander` with `_go_absolute` BFS pathfinding for sector exploration.
- **Impact**: Local VOR 1.49 → 1.59 (+6.7%, 10-episode confirmed). 6th win overall.
- **Why it works**: Previous wander tested 4 adjacent cells toward sector target. If the preferred direction was blocked by a wall, it tried the next (possibly AWAY from target). BFS navigates around walls within the 13x13 visible window, always finding the best first step toward the sector target.
- **Lines**: `_go_absolute` new `fallback_to_wander` param, `_wander` calls `_go_absolute`
- **Key insight**: 6th consecutive capability improvement win. Wander navigation was the last major area without BFS.
- **Category**: Capability improvement (pathfinding upgrade)

## Combined net radius 20 + timeout 50 (v21) — P89
- **What**: Increased ALIGN_NET_RADIUS from 15 to 20 and aligner timeout from 35 to 50.
- **Impact**: Local VOR 1.59 → 1.65 (+3.77%, 10-episode). 7th win overall.
- **Why it works**: Timeout 50 extends the P65 pattern — more patience for border junctions. Radius 20 is neutral (tested individually as P87 = 0%). The gain is entirely from timeout.
- **Lines**: `ALIGN_NET_RADIUS = 20`, `_aligner()` target_ticks > 50
- **Key insight**: Timeout wins keep compounding: 15→25 (+14%), 25→35 (+8.1%), 35→50 (+3.77%). Diminishing returns but still above 3% threshold.
- **Category**: Capability improvement (extended patience for border junctions)

## c4 role 1M/3A (v24+P154) — P154
- **What**: Changed c4 (team_size==4) from 2M/2A to 1M/3A. Only affects team_size==4 branch.
- **Impact**: c4 +18.7% (p=0.0025, 20 paired seeds, Cohen's d=0.779). Tournament-weighted +5.05%.
- **Why it works**: Hearts are over-produced at 23% utilization. Hub starts with 8 hearts (24 each element + 5 pre-crafted). Single miner produces ~60 hearts in 10K ticks — enough for 3 aligners (~20 each). Extra aligner = +50% alignment capacity for 27% of tournament.
- **Lines**: `_assign_role()` team_size==4 branch
- **Key insight**: Diminishing returns on miners: beyond 1 miner for c4, extra miners produce hearts that go unused. Alignment capacity was the bottleneck, not heart supply.
- **Category**: Role distribution (now 4/7 win rate in this category)

## Visible-only junction deposits (v5)
- **What**: Miners deposit only at junctions visible on screen (confirmed cogs), fall back to hub for off-screen
- **Impact**: Local 8v0 score 1.11 → 1.27
- **Why it works**: Eliminates risk of depositing at scrambled junctions
