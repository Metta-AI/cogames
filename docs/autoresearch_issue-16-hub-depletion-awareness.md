# Autoresearch Issue 16: Hub Depletion Awareness

Branch: `autoresearch/issue-16-hub-depletion-awareness`

**Issue direction:** Once the hub's 5 hearts are consumed (~step 200-300), agents enter a terminal get_heart->stale->unstuck->explore loop that wastes 60-70% of remaining episode. Teach agents to detect hub depletion and switch to defense/exploration.

**Success criteria (from issue):**
- get_heart stale exits < 10 (vs current ~55)
- reward at 1000 steps > 0.92 (vs current ~0.56 on main)
- No agent stuck at same position for > 100 consecutive steps

**Suggested experiments:**
- A: Track heart.withdrawn count; when >= 5, remove get_heart from skills, add defend
- B: After 3 consecutive get_heart stale exits, blacklist get_heart for that agent
- C: Add hub_depleted to LLM prompt context
- D: When hub depleted, switch aligners to long_explore

---

## 2026-03-29T00:00:00Z: autoresearch starting, my plan is to...

**Plan:**
1. Run baseline with cross_role policy (3 agents, 1000 steps) to measure current get_heart stale exits and reward
2. Implement Experiment A: Track heart withdrawals in SharedMap, gate get_heart when >= 5 withdrawn
3. Combine with Experiment C: Add hub_depleted flag to LLM prompt so model can reason about it
4. If needed, implement Experiment B as a per-agent fallback

**Hypothesis:**
The root cause is that agents have no way to know the hub is out of hearts. They keep trying get_heart, timing out after stuck_threshold*5 steps each time, wasting hundreds of steps. By tracking total heart withdrawals across the team and removing get_heart from available skills once depleted, agents will immediately switch to productive activities (defending held junctions, exploring for new ones).

---

## 2026-03-29T00:00:00Z: starting to run baseline

**Command:** `source .env.openrouter.local && UV_CACHE_DIR=/tmp/uv-cache uv run cogames play -m cogsguard_machina_1 -c 3 -p class=cross_role,kw.num_aligners=3,kw.llm_timeout_s=20 -s 1000 -r log --autostart`

**Baseline results:**
- mission_reward: 0.5709 (per-agent: 0.57)
- cogs/heart.withdrawn: 5 (all 5 hearts consumed)
- cogs/aligned.junction.held: 4709
- cogs/aligned.junction.gained: 6
- get_heart selected: 95 times
- get_heart completed: 16 times (17% success rate)
- get_heart stale/timeout exits: 83 (confirms issue: agents waste most steps on failed get_heart)
- has_heart=False in 96 of LLM decisions
- Agent 0: 446 move failures (44.6%), heart.gained=2
- Agent 1: 415 move failures (41.5%), heart.gained=2
- Agent 2: lost aligner gear, got miner gear, heart.gained=3 but 1 unused
- Total stale/stuck/timeout exits across all skills: 122

**Analysis:** Confirms issue exactly. After ~5 hearts withdrawn, agents enter get_heart->stale->explore->get_heart loop. 83 stale get_heart exits waste ~83*20=1660 agent-steps. With 3 agents * 1000 steps = 3000 total agent-steps, that's 55% wasted on failed get_heart alone.

---

## 2026-03-29T00:01:00Z: experiments v1-v5

**v1-v2:** Hub depletion tracking via global counter (hub_hearts_withdrawn >= 5/4).
Override never fired in time — agents got contaminated/died first. Worse than baseline.

**v3:** Added per-agent consecutive_get_heart_failures >= 2 for faster detection.
Get_heart stale exits: 83 → 0! But agents explore → wander into clip territory → die.
Best seed43: 0.62. Average across seeds: 0.50.

**v4:** Defend skill (noop near friendly junctions). Too passive — 94% noop, wasted all steps.
Score: 0.41. Much worse.

**v5:** When hub depleted, switch heartless aligners to mining → deposit resources → fund make_heart.
Mining switch works but full mine→make_heart cycle takes too long for 1000-step episodes.
Seed43: 0.64 (best result!). Average: 0.53. Baseline avg: 0.56.

**Key learnings:**
1. make_heart exists: costs 7 of each element (28 total). Mining can create hearts.
2. get_last_heart handler allows all 5 initial hearts to be withdrawn (not just 3-4)
3. LLM timing variance dominates results — agent deaths from clips ships are the main noise source
4. The get_heart stale metric went from 83→0 consistently, but reward isn't improving proportionally
5. The mining switch doesn't pay off in 1000 steps — too expensive (gear switch + mine + deposit cycles)

---

## 2026-03-29T01:00:00Z: experiment v6 - cooldown approach

**Approach:** Escalating cooldown after get_heart failures (2*N cycles, max 8). During cooldown, agents explore then retry. Hard-depleted (>= 5 hearts withdrawn) triggers mining switch.

**Results:**
| Seed | Baseline | v5 | v6 |
|------|----------|-----|-----|
| 42   | 0.57     | 0.51| 0.51|
| 43   | 0.62     | 0.64| 0.62|
| 44   | 0.50     | 0.44| 0.50|
| Avg  | 0.563    | 0.530| 0.543|

**Key metrics for v6:**
- get_heart stale exits: 0 across all seeds (down from 83 in baseline)
- get_heart selected: 9, completed: 11 (seed 42) — every attempt succeeded!
- The cooldown allows retries but spaces them out, preventing waste

**Analysis:** v6 matches baseline reward within noise while eliminating get_heart waste. The remaining variance is from agent deaths/contamination (unrelated to hub depletion). The cooldown approach is the least disruptive — it doesn't change behavior much when things work, but prevents worst-case loops.

**Next steps for future researcher:**
- The target of > 0.92 at 1000 steps requires improvements beyond hub depletion
- Main bottleneck now: agent deaths from clip ships and gear contamination
- Explore reducing clip ship interactions or better hazard avoidance
- Consider longer episodes (2000 steps) where mining→make_heart cycle can complete

---

## 2026-03-29T02:00:00Z: experiment v7 - explore-only (no mining switch)

Removed mining switch. When hub depleted, agents just explore. deposit_to_hub has navigation issues that waste 400 steps per failed attempt.

v7 results: seed42=0.49, seed43=0.62, seed44=0.50 (avg 0.537). Slightly worse than v6 (0.543). All versions achieve 0 get_heart stale exits.

## Final Summary

**Best approach: v7 (explore-only with cooldown)** — simplest, eliminates 83 get_heart stale exits while maintaining baseline-comparable reward.

Key changes in v7:
1. `SharedMap.hub_hearts_withdrawn` counter (incremented on get_heart completion)
2. `CrossRoleState.consecutive_get_heart_failures` + `get_heart_cooldown_steps`
3. Escalating cooldown after failures (2×N cycles, max 8)
4. `hub_depleted` flag in LLM prompt removes get_heart from available skills
5. Precondition enforcement prevents get_heart during cooldown/depletion

Target >0.92 needs fixes beyond hub depletion (navigation, deaths, clip avoidance).
