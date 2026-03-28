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

