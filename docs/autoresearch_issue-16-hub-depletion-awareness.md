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
