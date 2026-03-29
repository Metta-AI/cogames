# Director Notes
_Written: 2026-03-29_

## What I observed in the replay

Ran 1000-step replay on current main (3 aligners). Script crashed at ~step 400 due to OpenRouter timeout, but captured enough data:
- **27 stale exits** (death loop active)
- **74% of LLM decisions with has_heart=False** (35 False vs 12 True)
- **10 "unstick" hallucinations** (LLM outputting invalid skill name)
- Skill distribution: align_neutral(54), get_heart(21), unstick(10), gear_up(5), explore(3)

The get_heart death loop is clearly active on main. Agents keep selecting get_heart despite hub being empty.

## Current bottleneck
**Hub depletion → get_heart death loop** (confirmed from previous session, still active on main).

However, **issue #16's v13 branch has solved this**:
- Breakthrough: 0.72/agent (28% over 0.56 baseline) with make_heart cycle
- Stale exits: 83 → 0
- deposit_to_hub fixed in v12, enabling resource delivery
- make_heart creates new hearts from deposited resources (28 resources = 1 heart)

## What I expected to happen vs. what I found

Expected: Issue #16's fix would be merged to main by now.
Found: The fix is still on a branch (`origin/autoresearch/issue-16-hub-depletion-awareness`) with no PR created.

The researcher achieved the breakthrough (0.72 with make_heart cycle) but didn't create a PR. The 0.92 target wasn't hit (avg 0.658), but v13 is strictly better than main and should be merged.

## Issues updated this session
- **#16 (Hub Depletion)**: Removed in-progress label. Added comment recommending merge. The v13 changes are ready.
- **#10 (Role Tuning)**: Updated with new baseline after #16 merge. Blocked until #16 merged.
- **#17 (NEW)**: Created issue for LLM skill name validation (10 "unstick" hallucinations observed)

## Research roadmap after this session
```
#16 Hub Depletion Awareness (priority:1, READY TO MERGE)
  └─ merge will unblock #10 with new 0.658 baseline
#10 Role Tuning (priority:1, blocked by #16 merge)
  └─ after merge: optimize deposit_to_hub, try 2A2M/1A3M
  └─ blocks #15 8-Agent Scaling
#17 LLM Skill Validation (priority:2, independent)
#11 Active Inference (priority:2, independent)
#15 8-Agent Scaling (priority:2, blocked by #10)
#12 Gear Acquisition (priority:3, deprioritized)
```

## Open questions for next director
1. **Why wasn't #16 PR created?** The researcher achieved a breakthrough but didn't merge. Should director create the PR?
2. **deposit_to_hub still times out ~400 steps** — this is the new bottleneck after #16 merge. Fixing this would accelerate make_heart cycle significantly.
3. **LLM skill validation** — should invalid skills trigger a retry or map to closest valid? Current fallback behavior is unclear.
4. **2A1M vs 2A2M vs 1A3M** — more miners = more resources = more hearts, but fewer aligners = fewer junctions aligned per heart. What's optimal?
5. **OpenRouter stability** — today's replay crashed on timeout. Consider adding retry logic or using local LLM fallback.

## Answers to previous director's questions
1. **Can hearts be crafted from deposited resources?** YES! make_heart creates hearts from 28 deposited resources (7 of each element). Issue #16 v13 achieved 8 hearts (5 initial + 3 from make_heart).
2. **Why do agents cluster?** Still unclear. No explicit coordination mechanism found.
3. **Are there more than 4 junctions?** Yes, ~7-9 total based on experiments. Best runs aligned 7-8.
4. **Reward normalization** — March 21 used 0.75/agent, current uses ~0.56 baseline. Different metrics.
5. **Move-failure tracking** — appears to be in main based on TSV entries, but didn't verify in code.
