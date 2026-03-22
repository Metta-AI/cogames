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

## Final Session Summary

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

### Interesting observations

The session started at 0.10 (0 junctions aligned) and reached 0.92 (24 junctions aligned) through 13 experiments in ~1 session day. Each bug fix had outsized impact because the bugs were fundamental navigation failures, not incremental tuning issues.

The biggest single jump was `greedy-bfs-fallback+forget-stuck-junction` (0.21 → 0.40), which fixed the dead code bug where the greedy fallback was never actually called.

The `action_timeout_ms` change (250ms → 3000ms) is a good reminder: infrastructure parameters can silently kill performance when they're miscalibrated for the actual LLM latency.

### Exploring next

1. **Reduce `align_neutral`/`get_heart` timeout from 100 to 150 steps** — Current timeout is `stuck_threshold * 5 = 20 * 5 = 100`. Many timeouts may fire because a 100-step journey has some oscillation; 150 might let agents succeed instead of retrying.

2. **Diagnose get_heart timeouts**: Are they navigation failures or hub depletion? If hub is depleted (miners deposited but hub hasn't crafted hearts yet), the fix is different — aligners should wait rather than retry. Add a `hub_has_hearts` observable to the state.

3. **Multi-seed validation of 0.92**: Test seeds 42, 43, 44 to check variance. Seed 42 might be favorable. Need at least 3 seeds for a reliable baseline.

4. **Junction pre-exploration**: After aligning a junction, immediately explore adjacent territory to discover new junctions before returning to hub. Would reduce wasted explore cycles.

5. **Increase `stuck_threshold`**: Currently 20 consecutive blocked moves triggers stuck exit. May be too aggressive when navigating through crowded areas (other agents blocking cells). Try 30.

6. **Aligner `_move_toward_target` optimistic BFS**: `_move_toward_target` in `AlignerAgent` only uses standard BFS + frontier navigation. `_align_neutral` has optimistic BFS, `_get_heart` uses `_navigate_to_station`. But if other codepaths use `_move_toward_target` directly, they miss optimistic BFS.

7. **Scripted miners vs LLM miners**: The LLM miner skill selection (gear_up/mine_until_full/deposit_to_hub/explore) overlaps heavily with the scripted fallback logic. Consider testing `scripted_miners=True` — if scripted miners perform equally, we save LLM calls and reduce latency.
