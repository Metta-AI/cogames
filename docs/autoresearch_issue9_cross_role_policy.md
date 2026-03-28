# Autoresearch Issue 9: Cross Role Policy

Branch: `autoresearch/issue-9-cross-role-policy`

**Setup:** `cogsguard_machina_1.basic`, 1000 steps, 4 agents, `class=machina_llm_roles,kw.num_aligners=4`, seed=42, cloud LLM (nvidia/llama-3.3-nemotron-super-49b-v1.5 via OpenRouter)

**Issue direction:** Make the policy not role-dependent. Instead of fixed aligners/miners, ask "what does the team need now?":
- Not just mine any resource, but find what resource we need most and mine that
- If lots of resources and no one making hearts → turn to aligner and go align
- If enemy junction → become a scrambler
- Start simple, later try inter-robot coordination

**Best previous policy (from issue):**
`7321afc  0.92  aligned=24/held=8195/total=3.68  1000 steps  action-timeout-3000ms+miner-optimistic-bfs`
(Note: 22_march session also found 1.24 reward at 2000 steps with 4A+0S)

---

## 2026-03-28T08:14:29Z: autoresearch starting, my plan is to...

**Plan:**
1. Run baseline: 4A+0S at 1000 steps (reproducing the 0.92 result from best previous policy)
2. Implement cross-role policy: agents can switch between aligner/miner roles based on team needs
3. Start simple: unified LLM prompt that sees all skills (aligner + miner) and team state
4. The LLM decides: "should I mine resources or align junctions based on what the team needs?"
5. Key state to expose to LLM: known_resources, known_junctions, team_resource_deposits, hub_hearts

**Architecture approach:**
- Create a new `cross_role_policy` that combines aligner + miner skill sets
- The LLM gets a unified view: team resources, junction status, heart economy
- Skills available: gear_up_aligner, gear_up_miner, mine_until_full, deposit_to_hub, align_neutral, get_heart, explore, unstuck
- Agents start as undefined role and ask LLM what the team needs
- Guard rails: scripted overrides ensure preconditions are met

**Key fix discovered:** Must set `EPISODE_RUNNER_USE_ISOLATED_VENVS=0` or the policy server subprocess runs in an isolated venv without the OpenRouter key, causing httpx ReadTimeout on the second LLM call.

---

## 2026-03-28T08:14:29Z: starting to run baseline

Running: `EPISODE_RUNNER_USE_ISOLATED_VENVS=0 cogames run -m cogsguard_machina_1 -c 8 -p "class=machina_llm_roles,kw.num_aligners=3,kw.llm_timeout_s=30" -e 1 -s 1000 --action-timeout-ms 3000 --seed 42`

## 2026-03-28T08:48:00Z: baseline result is 0.66

**Baseline:** mission_reward=0.66, aligned=8 (held=5633)
- 8 agents: 3 LLM aligners + 5 LLM miners
- Hub resources: silicon=72+82deposited, germanium=32+50deposited, carbon=10+24deposited, oxygen=0+10deposited
- Heart usage: 6 hearts withdrawn
- Action timeouts: 1 (nearly all LLM calls succeed within 3000ms)
- Enemy: 43 junctions aligned, 21040 held (we're losing badly on junctions!)
- `aligner.gained=0.25/agent` = only 2 agents actually got aligner gear

**Key observations:**
1. Agents NOT getting miner gear - agents ending up with scout/scrambler gear accidentally
2. Hub has silicon/germanium surplus but oxygen shortage (0 deposited vs 10 is low)
3. Only 2 of 3 intended aligners got aligner gear
4. Enemy controls 43 junctions vs our 8 - big gap
5. Hub has resources but NOT making hearts efficiently (only 6 hearts used)

**Resource deficit for heart crafting:** need 7 of each type per heart.
- Silicon: 82 deposited (good)
- Germanium: 50 deposited (good)
- Carbon: 24 deposited (enough for ~3 hearts)
- Oxygen: 10 deposited (bottleneck! only ~1 heart worth)

---

## 2026-03-28: starting new experiment loop (cross-role v1)

**Hypothesis:** Give all 8 agents freedom to choose aligner or miner based on LLM judgment of team needs. Unified prompt with all 8 skills.

**Result (commit 15f24c4): 0.49 reward** — WORSE than baseline 0.66.

**What happened:**
1. All 8 agents chose `gear_up_miner` at start (LLM logic: "no known junctions, mining is the safe default")
2. After ~30 steps, junctions discovered, agents tried to switch to aligner
3. But gear switching is expensive (navigate back to station, get new gear)
4. 2 agents accidentally got scrambler/scout gear during navigation
5. Only 6 junctions aligned vs 8 in baseline; 3891 held-steps vs 5633

**Root cause:** Cold-start problem. Without a forced initial role split, all miners, then too late to switch efficiently.

**Next:** Add `num_aligners=3` to bootstrap initial role split (skips LLM for first plan). Add team gear counts to prompt so LLM can reason about team balance. This should recover baseline performance while enabling dynamic switching later.

---

## 2026-03-28: starting new experiment loop (cross-role v2: num_aligners + team context)

**Plan:**
- `num_aligners=3`: agents 0,1,2 start as aligners (no LLM for first plan), agents 3-7 start as miners
- Team state in prompt: `team_aligners`, `team_miners` so LLM can make better switching decisions
- Hint logic: if team has too few aligners and junctions available, suggest aligner role
- `kw.num_aligners=3,kw.llm_timeout_s=60,kw.stuck_threshold=30` for stability


