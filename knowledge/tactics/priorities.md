# Current Priorities

Updated: April 15, 2026 (v28 uploaded with P154)

## Next Action
**IMPLEMENT P200: Scrambler Highway + Role Switching**
Full implementation spec: `knowledge/experiments/P200-scrambler-highway.md`
- This is the #1 missing capability. Every top-5 policy uses scramblers (30-177/ep). Softy uses 0.
- 6 code edits to softy.py (constants, state, _assign_role, _should_switch_role, tracking, step_with_state)
- Random pool WILL regress — this is expected and acceptable (scrambler idle vs random)
- Upload directly to tournament after benchmarks pass (capability improvement, 11/11 track record)
- **P1 failed because it replaced miners. P8 failed because miners could switch. P200 locks miners.**

## Upload Strategy
- Accumulated delta reset to 0% (just uploaded v28).
- Near-misses (P86/P37/P101) all NOISE on v24+P154 base.
- Remaining hypotheses have very small expected effects.
- Need fundamentally new approach or bug discovery for next win.
- **P139 lesson**: +4.9% on c6 but -51% on c2, -15% on c4. Net tournament: -12%.
- **P127 lesson**: +5.9% on c6 but v25 dropped from 25.60 to 23.40 in tournament.

## Critical Rules
- **ONLY bug fixes and capability improvements translate to tournament** (10 for 10; P127/P139 REVERTED)
- **Behavior tuning HURTS in tournament** — 0 for 100+
- **Hub-reducing changes HURT** (hub provides safety under competitive pressure)
- **Dynamic roles only help severely under-resourced scenarios** (c2/c4 fixed, c6+ leave alone)
- **NEVER test on one scenario only** — c2 is 31%, c4 is 27%, c6 is 33%, c8 is 9%

## Testing Protocol
- **Phase 1 (screening)**: Paired same-seed, 10 seeds, c6 only. REVERT if p>0.30 or delta<-3%.
- **Phase 2 (multi-scenario)**: If Phase 1 passes, test c2+c4+c6+c8, 5 seeds each.
  Compute tournament-weighted delta. REVERT if ANY scenario drops >-5%.
- **Phase 3 (full validation)**: If Phase 2 passes, 20 seeds across scenarios.
  Tournament-weighted p<0.10 AND delta>+1% = KEEP.
- **Map**: machina_1.clips (88x88 with NPC clips ships — the tournament map!)
- **Upload gate**: Must pass ALL 3 phases. No exceptions.

## Deep Analysis Findings (Cycle 49-50)
- **Deaths: 12.5/agent (NOT 55)** — earlier estimate was wrong
- **Move failures: 17.2%** — high but includes intentional bumps (entity interactions)
- **Wall density: 43%** — causes navigation challenges
- **Hearts: 300+ craftable, only 81 used = 23% utilization** → P139 fix
- **Aligner cycle time: 370 ticks/alignment** — primary bottleneck
- **Stale junction data**: Clips junctions appear "neutral" in coordinator (agents rarely re-visit)
- **P138 noop**: Unified scoring no effect because stale data already makes clips look neutral

## Remaining Capability Ideas
1. **P142**: Miner path optimization (43% walls → many failed moves during mining travel)
2. **P143**: Adaptive timeout based on network size (60 might be too long once network is large)
3. **P107**: Wall memory (persist walls across ticks for better BFS) — P120 tried and failed (-6.5%)
4. **P3 clips timing**: Predict clips scramble timing (every 200 ticks from 100), avoid aligning vulnerable junctions. Never tested on current base.
5. **c8 role tuning**: c8 currently uses ROLE_CYCLE (3M/5A). With P154 logic, 2M/6A might work for c8 (9% weight).
6. **Aligner travel reduction**: Closest-first junction targeting improvements (currently frontier*8.0-dist)

## Bottleneck Analysis
- **P12 fixed**: Heart production (element balance)
- **P102/P103 fixed**: Small-team role distribution (c2/c4)
- **P104 fixed**: BFS computation waste
- **P127 fixed**: Aligner patience at junctions (timeout 50→60)
- **P139 fixed**: Heartless exploration waste (always hub for heart)
- **Current bottleneck**: ALIGNER TRAVEL TIME — 88x88 map with 43% walls means long paths between junctions and hub
- **Secondary**: Move failure rate (17.2%) — agents walk into walls during off-screen navigation

## Key Numbers
- Current: v28 = v24+P154 (P12+P18+P15+P65+P73+P76+P89+P102+P103+P104+P154)
- **11 wins out of 158 hypotheses. All bug fixes / capability improvements / role distribution.**
- Tournament: v28 pending. Previous: v24 at 25.60 VOR, rank 2. Gap to #1 = 6.7%.
- Trajectory: v2(2.16) → v7(10.52) → v18(16.14) → v21(23.56) → v24(25.60) → v28(??)
- P154 +18.7% on c4, tournament-weighted +5.05%. Expected v28 ~26.85.
- Non-determinism: c4+ varies ~6% between runs. c2 is deterministic.
