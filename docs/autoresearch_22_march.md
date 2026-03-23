# Autoresearch 22 March

Branch: `autoresearch_22_march`

**Setup:** `cogsguard_machina_1`, 1000 steps, 8 agents, `class=machina_llm_roles,kw.num_aligners=3`, seed=42, cloud LLM (nvidia/llama-3.3-nemotron-super-49b-v1.5 via OpenRouter)

---

## 2026-03-22T08:51: Session start

**Context:**
- Local Nemotron Nano 9B V2 is available at `/workspace/models/nemotron-nano-9b-v2`
- With local LLM, we can call it more frequently than cloud — no rate limits, no cost
- Previous best: `2.260000` reward on `cogsguard_machina_1` 1000 steps (3 aligners, 0 miners)
- Regression: the local-LLM PR (#5) lost many improvements from autoresearch_21_march

**Key observations from autoresearch_21_march:**
- Best team is 3 aligners (no miners) with reward 2.260
- Only 7 junctions exist on this map; all were aligned in best runs
- Heart economy: hub starts with 5 hearts, `make_heart` requires 7 of each element (28 total)
- Previous runs: `heart.withdrawn=5` with 1 miner depositing 371 resources → hearts weren't being crafted from resources
- Root cause: missing `stationary_on_valid_target` fix → aligner was getting kicked away from hub before `make_heart` + `get_last_heart` could complete
- LLM is the key differentiator vs pure scripted (0.588 scripted aligners vs 2.260 LLM)

**Plan:**
1. Fix regressions from local-LLM PR: restore `num_aligners`, better overrides, `stationary_on_valid_target`
2. Add per-agent `replan_interval` for more frequent LLM calls (leveraging local speed)
3. Improve heart economy: ensure aligners stay at hub to trigger `make_heart` + pickup
4. Improve miner resource diversity for heart crafting
5. Run experiments: baseline → heart economy → frequent LLM

**Default mission setup:** `cogsguard_machina_1`, 3 agents, 1000 steps, local LLM

---

## Experiment Log

### `7d3ac20` — 0.10 — fix-skill-completion-loops

**Result:** mission_reward=0.10, aligned=0, total=0.30

3 aligners 0 miners. agent0=22moves, agent1=14moves+gear+8hearts, agent2=61moves. 0 junctions aligned. Navigation BFS deadlock dominant issue — agents acquire gear and hearts but can't navigate to junctions. BFS fails when path goes through unexplored territory.

---

### `5859a9f` — 0.20 — navigation-shake-greedy-gearup

**Result:** mission_reward=0.20, aligned=1, held=959, total=0.80

3 LLM aligners + 1 scripted miner. Added navigation shake (periodic random moves to break BFS deadlocks) and greedy fallback. move.success=359/agent, move.failed=633/agent. aligner.gained=0.25/agent (only 1/4 got gear). heart.gained=0.75/agent. 1 junction aligned. max_steps_without_motion=532. 12 LLM timeouts. Beats scripted 0.131 baseline.

---

### `9e5a8aa` — 0.20 — skill-step-timeout

**Result:** mission_reward=0.20, aligned=1, held=959, total=0.80

Same as nav_shake but added skill-level step timeout. More LLM calls (32 timeouts). Skill timeout fires but same bottleneck: 2/3 aligners never reach station because BFS fails when path requires unexplored territory. `was_stuck` fix not yet applied.

---

### `218af61` — 0.21 — was_stuck-timeout-fix

**Result:** mission_reward=0.21, aligned=2, total=0.84

`was_stuck` now detects skill timeouts. `align_neutral→unstuck` override fires when stuck. 2 junctions aligned (vs 1). Still max_stuck=533. 47 LLM timeouts. Marginal improvement — core navigation problem remains.

---

### `11e5ac0` — 0.21 — no-hazard-avoid-when-equipped

**Result:** mission_reward=0.21, aligned=2, held=1054, total=0.84

Skip hazard avoidance in align_neutral/get_heart BFS (agents already have gear, can't re-equip at hazard stations). 2 junctions aligned. aligner.gained=0.25 (1/4 still). 47 LLM timeouts. Same bottleneck: agents can't find path through unexplored territory.

---

### `c05ef9f` — 0.21 — greedy-nav-fallback (**dead code bug**)

**Result:** mission_reward=0.21, aligned=2, held=1054, total=0.84

Attempted greedy navigation fallback when BFS fails. **Bug: dead code** — `_move_toward_target` never returns noop so greedy was never called. Same result as previous commit. `forget-stuck-junction` committed separately.

---

### `95c7706` — **0.40** — greedy-bfs-fallback+forget-stuck-junction (**first big jump**)

**Result:** mission_reward=0.40, aligned=9, held=3046, total=1.60

**+90% over previous best.** Two fixes landed together:
1. **Fixed dead code bug**: greedy fallback now actually activates when BFS fails
2. **Forget stuck junction**: after timeout on `align_neutral`, forget the stuck junction and try a different one

9 junctions aligned (vs 2). 3046 held-steps (vs 1054). heart.gained=1.12/agent. 423 moves/agent. **3× better than scripted baseline.**

---

### `(unstuck16)` — 0.20 — **unstuck_horizon=16 WORSE** (discard)

**Result:** mission_reward=0.20, aligned=1, held=959, total=0.80

Tried `unstuck_horizon=16` (vs default 4). **Significantly worse** — longer unstuck sequence wastes too many steps. **Keep unstuck_horizon=4.**

---

### `af41f51` — **0.63** — gear-up-fix-all-aligners (**second big jump**)

**Result:** mission_reward=0.63, aligned=16, held=5326, total=2.52

**+57.5% over previous best 0.40.** Root cause of gear-up failures identified and fixed:
1. **BFS was targeting the station cell itself** (which is in `blocked_cells`) → BFS immediately returned `None`
2. **`_explore_near_hub` was sending agents OUTSIDE hub area** → agents never found the aligner station

Fix: `_navigate_to_station` now targets the adjacent approach cell, not the station itself. Expected station position hardcoded at `hub_center + (4, -3)`. Now 3/3 aligners gear up (aligner.gained=0.38 vs 0.25 before). 16 aligned (vs 9). 357 LLM timeouts (high — action_timeout_ms=250ms causing most calls to time out and produce noops).

---

### `0f7a64a` — 0.69 — get-heart-optimistic-bfs+max-cells-20k

**Result:** mission_reward=0.69, aligned=22, held=5894, total=2.76

Hub visible but BFS failed because path through unknown cells. Fix: **optimistic BFS** (treat unknown cells as traversable, only avoid known walls) + increased max_cells 5k→20k for 100×100 map. 22 aligned (vs 16). 5894 held-steps (vs 5326). 38 skill timeouts. 351 LLM timeouts.

---

### `802fcdd` — 0.71 — forget-stuck-junction-after-1-timeout

**Result:** mission_reward=0.71, aligned=19, held=6122, total=2.84

Changed `align_neutral` to forget stuck junction after 1st timeout (vs 2nd). 10 align_neutral timeouts (vs 38), 10 junctions forgotten+retried. 21 get_heart timeouts. 6122 held-steps (vs 5894) — more held time despite fewer aligned junctions (19 vs 22). 374 LLM timeouts. Net new best due to held-step bonus, but junction count actually decreased — less exploration hurts coverage.

---

### `a07334f` — **0.90** — get-heart-navigate-to-approach-cell (**third big jump**)

**Result:** mission_reward=0.90, aligned=22, held=7982, total=3.60

**+26% over previous best 0.71.** Critical bug: **hub cell was in `blocked_cells`** (hub has a blocking tag), so BFS targeting hub cell immediately failed and returned `None`. With optimistic BFS, it could still find an optimistic path, but it was unreliable. Fix: `_get_heart` now uses `_navigate_to_station` approach (navigate to adjacent non-blocked cell, then step into hub). Result: heart.gained=5.00/agent (+25%). 22 aligned (same count), 7982 held-steps (+30%). 14 align_neutral timeouts + 24 get_heart timeouts. 399 LLM timeouts.

---

### `7321afc` — **0.92** — action-timeout-3000ms+miner-optimistic-bfs (**best result**)

**Result:** mission_reward=0.92, aligned=24, held=8195, total=3.68

**+2% over 0.90.** Two changes:
1. **`action_timeout_ms` 250ms → 3000ms**: With 250ms, 397 of 399 LLM calls exceeded timeout → noops. With 3000ms, only 2 timeouts. Eliminated 397 wasted action steps.
2. **Miner optimistic BFS in `_move_toward_target`**: miners now use optimistic BFS (treat unknown cells as traversable) when standard BFS fails — essential for navigating back to hub when path goes through unexplored territory.

24 aligned (+2). 8195 held (+213). 24 align_neutral timeouts + 26 get_heart timeouts. **BEST RESULT OF SESSION.**

---

### `e043101` — ~0.10 — revert-miner-optimistic-bfs (**discard, aborted**)

**Result:** aborted, ~aligned=4, estimated total=~0.40

Attempted to remove miner optimistic BFS based on analysis showing 0.92 had fewer completed deposits (75 vs 83 in 0.90). This was the **wrong conclusion** — the hypothesis was flawed. Without optimistic BFS:
- 3/5 miners completely stuck in `gear_up`/`unstuck` loops (couldn't reach miner station)
- Hub depleted → aligners couldn't get hearts → aligners stuck in `get_heart` timeouts
- LLM hallucinating `"unstick"` instead of `"unstuck"` (normalized in current code)

Aborted after confirming catastrophic failure (~4 friendly junctions vs 24). Root cause: when BFS fails and there's no optimistic fallback, miners' frontier navigation sends them **away** from hub into unexplored territory, making the situation worse.

---

## Final Session Summary (first half)

### What worked (biggest impact, in order)

1. **Fix gear-up navigation (`af41f51`, +57.5%)** — Station object is in `blocked_cells`; BFS must target adjacent approach cell. Use `_navigate_to_station` pattern everywhere.
2. **Fix hub navigation (`a07334f`, +26%)** — Hub is also in `blocked_cells`. `_get_heart` must use approach cell, not hub cell directly.
3. **Greedy BFS fallback + forget stuck junction (`95c7706`, +90%)** — When BFS fails (path through unexplored cells), fall back to greedy navigation toward nearest frontier cell closest to target. Forget unreachable junctions after timeout.
4. **action_timeout_ms=3000ms (`7321afc`, +2%)** — With 250ms, nearly every LLM call timed out → noop. 3000ms eliminates wasted steps.
5. **Optimistic BFS (`0f7a64a`, `7321afc`)** — Treat unknown cells as traversable when standard BFS fails. Critical for both aligner hub navigation and miner deposit routing.
6. **`unstick` typo normalization** — LLM sometimes generates `"unstick"` not `"unstuck"`; fixed with `{"unstick": "unstuck"}` mapping in `_parse_role_skill_choice`.

### What didn't work

- **`unstuck_horizon=16`** — Longer unstuck sequences waste too many steps. Keep at 4.
- **Reverting miner optimistic BFS** — Catastrophic. The BFS is load-bearing for miner navigation.
- **`forget-stuck-junction` after 1 timeout** — Improves timeout recovery but hurts junction coverage (fewer total junctions aligned, just held longer). The tradeoff isn't clearly positive.

### Key findings

**Navigation architecture insight**: All structure objects (station, hub, junction) are in `blocked_cells`. Standard BFS immediately fails if goal is in `blocked_cells`. The fix pattern is always: target an adjacent non-blocked approach cell, then step onto the object from there. This pattern (`_navigate_to_station`) must be used consistently everywhere.

**Optimistic BFS is essential**: On a 100×100 map, large portions are unexplored. Standard BFS (only known-free cells) frequently fails when path goes through unexplored territory. Optimistic BFS (treat unknown as free, avoid only known walls) enables navigation through partially-explored territory. This is critical especially for miners returning to hub.

**action_timeout_ms vs LLM latency**: Cloud LLM (Nemotron Super 49B) takes ~750ms–1800ms per call. With `action_timeout_ms=250ms`, every call exceeds the timeout → noop action. The 250ms limit was calibrated for a much faster model. Always set `action_timeout_ms` to at least 2× the 90th-percentile LLM latency.

**Miner BFS failure cascade**: Without optimistic BFS, when miner `_deposit_to_hub` can't find a path (unexplored gap in path to hub), frontier navigation sends the miner toward the frontier (exploration) rather than hub. This creates a feedback loop: miner explores further → path to hub grows longer → harder to return → eventually stuck at frontier far from hub.

**Hub depletion chain**: Hub has finite hearts. If miners can't deposit, hub can't craft new hearts. If hub has no hearts, aligners get stuck in `get_heart` timeout loops (100 steps per attempt). This silently kills aligner performance even when navigation works.

**Remaining bottlenecks at 0.92**:
- 24 `align_neutral` timeouts × 100 steps = 2400 wasted aligner steps
- 26 `get_heart` timeouts × 100 steps = 2600 wasted aligner steps
- Out of 3 aligners × 1000 steps = 3000 total aligner steps, ~5000 wasted steps implies many timeouts are cut short by game end or overlap

---

## 2026-03-22T(new session): Scout Grid Exploration

**Research direction:** Use dedicated Scouts with systematic grid exploration to fix navigation failures. The core insight: if scouts pre-explore the map in a systematic serpentine sweep, they build complete map knowledge enabling clean BFS navigation (no more optimistic BFS hacks). Run at 2000 steps (vs 1000 previously) to see if reward saturates.

**Plan:**
1. Run 2000-step baseline first (same 3-aligner setup as 0.92)
2. Implement ScoutExplorerPolicyImpl with systematic grid exploration
3. Team: 2 aligners + 1 scout (3 agents total)
4. Scout: serpentine grid sweep, avoid enemy corners, HP retreat
5. Re-architected navigation: systematic hub-centric exploration for aligners
6. Experiment: 2A+1S at 2000 steps vs 3A at 2000 steps

---

### `521deca` — 0.485 — scout-grid-explore-broken-coords (DISCARD)

**Result:** reward=0.485, aligned=7, held=2854

Scout had broken coordinate system (grid targets in absolute coords not spawn-relative). Scout died in 16 moves (stuck 1949/2000 steps). Worse than 0.490 baseline.

**Root cause identified during debugging**: resource extractors (33 silicon + 40 oxygen + 35 germanium + 37 carbon = 145 total on map) block movement but aren't tagged as "wall". BFS routes through them → repeated move failures → agents stuck.

---

### `6f1147a` — **0.612** — move-failure-tracking (3A+0S) (**new 2000-step best**)

**Result:** reward=0.612/agent, aligned=6, held=4117

**+25% over 2000-step baseline 0.490!** Key fix: `move_blocked_cells` mechanism — when a move fails (position unchanged from last step), mark the target cell as blocked. This persists across observation updates (unlike standard `blocked_cells` which gets cleared by `difference_update(visible_cells)`).

- Agent 0: 570 success / 1429 failed moves, 2 junctions
- Agent 1: 600 success / 1400 failed moves, 3 junctions
- Agent 2: 571 success / 1429 failed moves, 1 junction

Also tested 2A+1S config with same code: reward=0.581 — scout hurts (replaces aligner). With 3-agent budget, 3A beats 2A+1S. Use 3A as primary config.

**Remaining bottleneck**: still ~70% move failure rate. 145 extractors on map = many obstacles. Move-failure detection is reactive (fail then learn). Could be improved with proactive extractor tag detection.

**Next experiments to try:**
- Detect extractor/object tags proactively to pre-populate blocked_cells
- Faster alignment: reduce time in gear_up/get_heart
- Scout with 4 agents (3A+1S)

---

### `cc137e7` — **1.190** — 4-aligners (-c 4) (**confirmed best**)

**Result:** reward=1.190, aligned_by_agent=1.50 (6 junctions total), heart.gained=1.50/agent, max_steps_without_motion=11, 1 LLM timeout

**Confirmed previous 4A result.** Using `cogames run -m cogsguard_machina_1.basic -c 4 -p class=machina_llm_roles -e 1 -s 2000 --action-timeout-ms 10000 --seed 42`. Key insight: `-c 4` gives 4 total agents all aligners. `status.max_steps_without_motion=11` shows the move-failure-tracking is working perfectly (vs 965 before).

**Bottleneck analysis:**
- Hub starts with 5 hearts, no miners → hearts depleted quickly → only 6 total alignments
- With all 7 junctions on map: agents aligned 6/7
- Each heart buys one junction alignment; with no miners, heart supply is finite

**Next hypothesis:** Add 1 scripted miner (-c 5, num_aligners=4, scripted_miners=true) to replenish hub hearts.
If miners deposit resources, hub crafts hearts, aligners can cycle through more alignments = higher held-steps and reward.

---


### `6857db1` — **1.240** — reclaim-enemy-junctions + OOM fix (**new best**)

**Result:** reward=1.240, junction.aligned_by_agent=1.50 (6 total), heart.gained=1.50/agent, max_steps_without_motion=11.25, 1 action timeout

**+4.2% over 1.190 baseline!** Two changes:
1. `_align_neutral` falls back to enemy junctions when no neutral ones available
2. Miner move-failure tracking now writes to `shared_map.move_blocked_cells`
3. **Critical fix:** `PYTORCH_ALLOC_CONF=expandable_segments:True` — model uses ~26 GiB, leaving ~2.7 GiB free; inference needs ~3 GiB; expandable segments allows non-contiguous allocation.

Also fixed: `num_aligners` was accidentally reverted to 3 in the reclaim commit; restored to 4. The gc/delete fix (running gc before empty_cache) was less impactful than the allocator fix.

**Key observation:** `junction.aligned_by_agent=1.50` is the same as before (6 total). Reward improvement may come from more stable navigation (fewer OOM crashes = cleaner episodes) rather than more alignments. Could also be small timing differences in junction alignment order.

**Bottleneck unchanged:** Only 6 hearts available. Need heart replenishment to break past 6 alignments.

**Next hypotheses:**
1. **5 agents (-c 5, 4 aligners + 1 scripted miner)**: miner replenishes hearts enabling >6 alignments
2. **Explore timeout**: when all junctions aligned, agents waste cycles on get_heart; add explicit "sit and defend" mode
3. **Different seed**: try seed=0 to see if reward varies with map layout



---

### `11600c3` — **N/A** — 5aligners (-c 5) (**discard, OOM**)

**Result:** OOM — CUDA out of memory (same as 4A+1M)

5 agents (4 LLM aligners + 1 scripted scout) uses slightly more GPU memory than 4 agents, leaving only 2.68 GiB free (vs ~2.71 GiB for 4 agents). LLM inference needs exactly 3 GiB contiguous allocation; expandable_segments can't help when total free VRAM is below 3 GiB.

**Confirmed hardware limit:** Maximum 4 agents on this GPU (A40 44.43 GiB, model 26.27 GiB, ~2.7 GiB free).

**5A as config is dead.** To break past 6 alignments (current ceiling with 4A), need a different approach.

**Revised analysis — achievable headroom with 4A:**
- Theoretical max held_steps: 7 junctions × 2000 steps = 14,000 held
- Current: 6 junctions × ~1717 avg held steps ≈ 10,300 held → reward=1.240
- 86% of 6-junction theoretical maximum already achieved
- To get 7th junction: need 7th heart (mining impossible with 4A only)
- To improve with 4A: either (a) align junctions faster for more held time, or (b) prevent enemy recapture

**Next hypothesis options:**
1. **Junction defense**: When hub empty, agents stand on aligned junctions to prevent enemy recapture
2. **Proactive enemy reclaim**: Monitor junction status; immediately return to re-align lost junctions
3. **Faster first-alignment**: Optimize gear_up/navigation path so agents align junctions earlier in game

---

### `HEAD` — **0.88** — 3aligners-1scout (-c 4, SharedMap) (**discard**)

**Result:** reward=0.88, cogs/aligned.junction.gained=6, cogs/aligned.junction.held=6800, heart.gained=1.75/agent, 112 LLM calls

**SharedMap validation:** The scout correctly shares map knowledge with all 3 aligners — `known_neutral_junctions` reaches 12-13 (vs 8-9 without scout), `known_hubs=4` discovered faster. Scout explored systematically while aligners focused on alignment.

**However, 3A+1S is worse than 4A+0S (0.88 vs 1.24):**
- The 3 aligners actually held junctions longer (6800 > 5517 held steps for 4A+0S baseline)
- But reward is per-agent average — scout earns 0 reward, diluting by 25%
- The map exploration benefit (12→13 known junctions vs 8→9) doesn't translate to more alignments because heart supply (5 total) is the bottleneck, not map knowledge
- At 2000 steps with shared map, even 4A+0S discovers enough of the map through aligner exploration alone

**Key insight:** SharedMap enables fast map sharing, but the bottleneck is heart supply (only 5 hearts from hub, no mining). A scout that improves map coverage doesn't help when the constraint is hearts, not navigation. The 4A+0S SharedMap config at 1.24 remains best.

**True bottleneck:** `get_heart timeout` loop dominates late game — agents repeatedly try to get hearts from depleted hub, timing out after 100 steps each attempt. With 5 hearts and 4 agents, hearts are consumed in the first ~500 steps. The remaining ~1500 steps are wasted on futile `get_heart` attempts.

---

### `ff6913d` — **1.24** — 4A+defend (neutral)

**Result:** reward=1.24, aligned=6, held=10378, heart=1.50/agent

Added "defend" skill: when hub is depleted (get_heart timeouts >= 1), agents navigate to and stand on friendly junctions instead of futile get_heart attempts. Tested both threshold=2 (original) and threshold=1 (linter's more aggressive version).

**No improvement.** Enemies don't recapture our junctions, so physically standing on them has no effect. The 48 defend events replaced futile get_heart timeouts but the held-step count is identical because junctions weren't being lost anyway.

**Real bottleneck confirmed:** 24 align_neutral timeouts — agents know about 7 junctions but can only reach/align 6 of them. The 7th is likely in a region that BFS can't navigate to efficiently (enemy territory, blocked paths, or insufficient map coverage in that corridor).

---

## Summary of SharedMap Research Direction

**SharedMap implementation: SUCCESS** — All agents share one map object by reference. Scout's exploration instantly benefits all aligners' BFS. Verified working with both mock and real LLM.

**SharedMap impact on reward: MARGINAL** — With 4A+0S, SharedMap helps agents discover junctions faster but the binding constraint is heart supply (5 hearts), not map knowledge. By step 500, even without SharedMap, agents have explored enough map to find all 7 junctions.

**Best result: 1.24 reward** (4A+0S, SharedMap, enemy junction reclaim, move-failure tracking)

**Ceiling analysis:**
- 5 hearts → 6 alignments max → 6 × ~1717 avg held steps = ~10,300 held → reward 1.24
- Already at 86% of 6-junction theoretical max
- Breaking through requires either: (a) mining hearts (impossible with 4 agents/GPU limit), or (b) aligning the 7th junction (24 timeouts suggest unreachable)
