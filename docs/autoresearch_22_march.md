# Autoresearch 22 March

Branch: `autoresearch_22_march`

## 2026-03-22T08:51: autoresearch starting, my plan is to...

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

## 2026-03-22T08:51: starting to run baseline

First, restoring critical regressions from local-LLM PR, then running baseline with local LLM.

**Changes:**
- Restored `num_aligners` + `aligner_ids` parameters to `MachinaLLMRolesPolicy`
- Restored `stationary_on_valid_target` in aligner (prevents stuck-timeout when waiting at hub for hearts)
- Restored `_known_alignable_junctions` helper
- Restored smarter override logic (overrode explore→get_heart when hub known, overrode to align_neutral when target known)
- Restored better explore completion (only completes when NEW junctions discovered)
- Restored explore strategies (explore_for_alignment when has_heart, explore_near_hub when hub known)
- Restored `no_progress_on_target_steps` stuck detection
- Added `known_hubs` count to aligner prompt for better LLM reasoning
- Added `replan_interval` parameter for more frequent LLM calls
