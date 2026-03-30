# Director Notes
_Written: 2026-03-30 (Session 4)_

## What I observed in the replay

### 3-agent replay on main (3A, machina_llm_roles, 1000 steps)
- **Reward: 0.563/agent** (total 1.689)
- Reward growth: 0.08/100 steps through step 500, then decelerates to 0.03/100 by step 1000
- **93% of LLM calls have has_heart=False** — hub depletes around step 500
- 213 stale exits, 79 get_heart attempts (only 12 successful)
- 176 LLM calls total, 0 LLM errors (new error handling works)
- Skill distribution: 79 get_heart, 75 explore, 24 align_neutral, 3 gear_up

### 8-agent replay (4A4M scripted miners, 1000 steps)
- **Reward: 0.4195/agent = 3.356 total** (4.2x over old 8-agent 0.80 total)
- Same hub depletion pattern: 2.3% has_heart=True
- Reward rate: 0.06/100 steps until step 600, then drops to 0.01/100
- 255 LLM calls, 4 LLM errors (handled gracefully), 393 stale exits
- Scripted miners work — mine_until_full and deposit_to_hub skills fire correctly
- Per-agent at step 400: 0.215 (vs 0.259 for 3A = 17% worse per-agent but 2.2x total)

## Current bottleneck

**Hub depletion on main branch** is the single most limiting factor. PR #18 (hub depletion awareness + make_heart cycle) is still not merged. Until it is, all 8-agent configs are limited to ~0.42/agent because hearts run out at step 500-600.

Secondary: LLM timeout crashes were killing episodes silently. Fixed by adding try/except around `_planner.complete()` in `machina_llm_roles_policy.py`.

## What I expected to happen vs. what I found

**Expected**: PR #18 would have been merged by now based on session 3's recommendation.

**Found**: PR #18 is still open. Three researchers built on its branch (#16, #24, #10) but none got merged back. Added urgency comments to PR #18.

**Surprise discovery**: 8-agent scaling works with scripted miners! 4A4M(scripted) achieves 3.356 total at 1000 steps — only limited by hub depletion. This is the clearest path to the 8-agent goal.

## Issues updated this session
- **#16**: Removed in-progress (watchdog killed). Still priority:1, PR #18 is critical.
- **#24**: Removed in-progress (watchdog killed). Best result 0.81/agent preserved.
- **#10**: Removed in-progress, deprioritized to priority:2. Superseded by #24.
- **#21**: Removed in-progress (watchdog killed, no results).
- **#15**: Closed — superseded by #25.
- **#25**: Created — 8-Agent Scaling with Scripted Miners (priority:1, blocked by PR #18).
- **PR #18**: Added urgency comment with session 4 data.

## Code changes this session
- `machina_llm_roles_policy.py`: Added try/except around `_planner.complete()` to handle LLM timeouts gracefully instead of crashing.
- `scripts/capture_frames.py`: Added `--policy-class` and `--policy-kw` CLI args for flexible replay configs.

## Research roadmap after this session
```
CRITICAL PATH (merge + 8-agent optimization):
  PR #18: Hub Depletion Fix → MERGE ASAP
    └─ #25: 8-Agent 4A4M(scripted) Scaling (priority:1, blocked)
    └─ #24: Fast-Extractor-Abandon (priority:1, 0.81/agent best)

OPTIMIZATION (after #25 unblocked):
  #10: Role Tuning (priority:2) — no element cycling, scarce_element preferred
  #20: Coordinated Exploration (priority:2) — spatial partitioning
  #17: LLM Skill Validation (priority:2)

RESEARCH DIRECTIONS:
  #19: LLM Code Generation (priority:2)
  #21: Intrinsic Motivation (priority:2)
  #11: Active Inference (priority:2)
  #22: Social Influence (priority:3)
  #23: Meta-Learning (priority:3)
```

## Open questions for next director
1. **PR #18 merge**: Has it been merged yet? If not, escalate further.
2. **8-agent projection**: With hub depletion fixed, 4A4M should maintain 0.06/100 growth → ~0.60/agent (4.8 total). Verify this.
3. **Optimal aligner/miner split**: 4A4M was tested, but 3A5M, 5A3M, 6A2M should also be tried.
4. **Fast-extractor-abandon + 8 agents**: Apply issue #24's 3-step threshold to 8-agent config.
5. **gemma-3-12b model**: Issue #16 showed 24% improvement with faster model. Test with 8 agents.
6. **LLM error handling**: The fix works (4 graceful fallbacks in 1000 steps). Should this be committed to main?
