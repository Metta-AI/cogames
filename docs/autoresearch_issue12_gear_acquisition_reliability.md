# Autoresearch Issue 12: Gear Acquisition and Change Reliability

Branch: `autoresearch/issue-12-gear-acquisition-reliability`

**Issue direction:** Fix gear acquisition reliability as a standalone problem, upstream of the full cross-role policy from issue #9. The problem: agents fail to equip intended gear at episode start, and accidentally pick up wrong gear (scout/scrambler) while navigating.

**Test harness (from issue):**
- 400-step episodes with 8 agents (`num_aligners=3`)
- Phase 1 (steps 0–200): agents 0,1,2 → aligner gear; agents 3–7 → miner gear
- Phase 2 (steps 200–400): force ALL agents to switch (aligners → miner, miners → aligner)
- Primary metrics: `initial_gear_success_rate` = fraction holding correct gear at step 200
- Secondary: `gear_change_success_rate` at step 400, `gear_contamination_rate` (scout/scrambler)
- Run at least 3 seeds to account for LLM timing variability

**Known root causes from issue:**
1. Navigation contamination: path to gear station routes through other stations (scrambler, scout)
2. `avoid_hazards=False` fallback path is the main culprit — BFS w/ hazards fails → falls back to optimistic BFS without hazard avoidance → routes through other stations
3. Map topology (seed=42): Agent 3 spawns far from miner station (proved in issue-9)
4. No gear-change path: once agent has wrong gear, navigating back also routes through hazard stations
5. LLM timing variability: ~2s per LLM call changes agent positions slightly

**Starting point:** Best result from issue-9 was cross_role_v9 (0.55 reward) with 2 aligners + 5 miners.
Current `initial_gear_success_rate` baseline from issue-9: ~0.25 (2/8 agents reliably).

---

## 2026-03-28T18:52:34Z: autoresearch starting, my plan is to...

**Plan:**
1. Restore cross_role_policy.py from issue-9 branch (v18 = best version with preferred_role hint)
2. Implement `GearTestPolicy`: 400-step episodes, phase-switch at step 200, gear metrics logging
3. Run baseline to measure current `initial_gear_success_rate`
4. Key experiments:
   a. Fix optimistic BFS fallback to also avoid hazard stations (buffer zone)
   b. Improve miner `_gear_up` to use aligner-style `_navigate_to_station` with `avoid_hazards=True`
   c. Add adjacency buffer around hazard stations (gear contamination from walking NEAR stations)
   d. Direct path planning that bypasses contamination zones entirely

**Hypothesis:**
The root cause is that optimistic BFS (fallback when BFS-with-hazards fails) completely ignores hazard stations. Adding hazard avoidance to the optimistic BFS fallback should prevent agents from routing through contaminating stations.

---

## 2026-03-28T18:52:34Z: starting to run baseline

**Gear test policy**: 400-step episode, `num_aligners=3`, phase_switch at step 200.
Looking at current gear acquisition:
- Gear test uses `GearTestPolicy` registered as `gear_test`
- Baseline metrics tracked via `gear_state` log lines

Run: `EPISODE_RUNNER_USE_ISOLATED_VENVS=0 cogames run -m cogsguard_machina_1 -c 8 -p "class=gear_test,kw.num_aligners=3,kw.llm_timeout_s=30" -e 1 -s 400 --action-timeout-ms 3000 --seed 42`

## 2026-03-28T19:30:00Z: baseline result

**Baseline result: 0.04 mission reward (400 steps)**

**Phase 1 gear states at step 200 (from PHASE_SWITCH logs):**
- Agent 0: old_preferred=aligner, gear=none → FAIL
- Agent 1: old_preferred=aligner, gear=aligner → SUCCESS
- Agent 2: old_preferred=aligner, gear=aligner → SUCCESS
- Agent 3: old_preferred=miner, gear=miner → SUCCESS
- Agent 4: old_preferred=miner, gear=miner → SUCCESS
- Agent 5: old_preferred=miner, gear=miner → SUCCESS
- Agent 6: old_preferred=miner, gear=miner → SUCCESS
- Agent 7: old_preferred=miner, gear=none → FAIL

`initial_gear_success_rate` = **6/8 = 0.75** (vs ~0.25 from issue-9 baseline!)
→ Hazard-aware miner gear_up is already a big improvement

**Phase 2 gear states at step 400:**
- Agents 1,2: still have aligner gear (phase 2 intended=miner) → FAIL
- Agents 3,4,5: still have miner gear (phase 2 intended=aligner) → FAIL
- Agents 0,6,7: gear=none → FAIL

`gear_change_success_rate` = **0/8 = 0.00**
`gear_contamination_rate` = 0/8 = 0.00 (no scout/scrambler contamination!)

**Root cause of phase 2 failure:**
The bootstrap logic only fires when `gear == "none"`. After the phase switch at step 200, agents still have their OLD gear (aligner or miner). The bootstrap doesn't recognize they need to switch gear:
- Agents 1,2 have aligner gear → prompt shows aligner skills only → LLM can't pick gear_up_miner
- Agents 3-7 have miner gear → prompt shows miner skills only → LLM can't pick gear_up_aligner

**Fix needed:** Bootstrap must also fire when `gear != effective_preferred` (wrong gear for current phase).

**Key finding:** The hazard-aware miner gear_up (_gear_up_miner_safe) WORKED:
- 5/5 miners (agents 3-7) acquired miner gear in phase 1 (except agent 7 who failed)
- 2/3 aligners (agents 1,2) acquired aligner gear in phase 1
- 0 contamination events (no scout/scrambler gear) in phase 1!

---

## 2026-03-28T19:30:00Z: starting new experiment loop (gear_switch_v1: phase 2 bootstrap fix)

**Hypothesis:** Phase 2 fails because bootstrap only checks `gear=="none"`. Need to add:
- If phase==2 and gear != effective_preferred and not gear_up_completed: bootstrap gear_up_{effective_preferred}
- This lets agents 1,2 navigate to miner station; agents 3-7 navigate to aligner station

**Changes (gear_switch_v1):**
- In `_plan_skill`: add phase 2 bootstrap that fires when `gear != effective_preferred`
- Rename "gear=none" bootstrap condition more precisely

---

## 2026-03-28T22:00:00Z: experiments v2-v11 summary

**v2 (32788d6):** discard — hazard adjacency buffer caused contamination
**v3-v4:** discard — reverted to 200-step timeout

**v5-v8: Navigation improvements**
- Key insight: `_navigate_to_station` always returns a direction via greedy fallback
- Greedy fallback can route through hazard stations → contamination
- v5: remove own-gear station from hazards (helps some miner→aligner switches)
- v6: BFS-without-hazards fallback (never fires due to navigate_to_station always returning direction — bug)
- v7: proper BFS cascade, gear_up_completed fix, phase2 persistent retry → p1=7/8 FIRST TIME!
- v8: remove optimistic-without-hazards (same results, confirming it was strict that caused contamination)
- v9: fix infinite loop (wrong-gear completion increments failures)

**v10: Multi-seed testing reveals**
- 3-seed average p1=0.71, p2=0.25
- BFS-without-hazards entirely removed (v10)
- Contamination still happens from greedy fallback in `_navigate_to_station`

**v11: Hazard-safe greedy**
- `_navigate_to_station_safe` checks if next step lands in hazard station
- If yes, returns None (caller explores instead of contaminating)
- Hypothesis: this should eliminate contamination while maintaining navigation quality

**Key findings:**
1. p1=7/8 is achievable (v7/v8/v9/v10 on seed 42)
2. p2 target (6/8) was met on seed 43 v9 but it was "lucky" (agents with no gear navigating to aligner)
3. Miner→aligner switch is hard due to map topology (scout/scrambler in path)
4. Aligner→miner switch should work in theory but gets blocked by navigation/time issues
5. LLM timing variability is large (3-seed variance is huge)
6. Bootstrap infinite loops must be prevented (v9 fix)
7. Greedy fallback contamination must be prevented (v11 fix)

