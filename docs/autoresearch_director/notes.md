# Director Notes
_Written: 2026-03-29 (Session 3)_

## What I observed in the replay

Ran replays with both `machina_llm_roles` (old) and `cross_role` (v13) policies on the issue-16 branch:

### Old policy (machina_llm_roles)
- **All 3 agents severely stuck**: A0 600+ steps, A1 400+ steps, A2 800+ steps at same positions
- **154 stale exits** in 1000 steps
- **88.5% of LLM decisions have has_heart=False** (hub depleted early)
- **36 "unstick" hallucinations** (LLM outputting invalid skill name)
- No hub_depleted awareness — agents loop get_heart→stale→unstuck forever

### v13 cross_role policy
- **Reward: 0.72/agent** (28% over baseline)
- **8 junctions aligned** (vs 5-6 baseline)
- **Stale exits: 48** (81% reduction from 257)
- **0 hallucinated skills**
- **make_heart cycle working**: miner deposited 47 resources → hub created 3 new hearts
- **hub_depleted flag in prompts**: 88 True vs 42 False

## Current bottleneck

**Navigation failures** remain the primary issue after hub depletion fix:
1. 48 stale exits still occur (was 257)
2. deposit_to_hub sometimes times out ~400 steps
3. Agent deaths from clip ship interactions are the main variance source

The hub depletion fix (PR #18) eliminates the get_heart death loop but doesn't fix the underlying navigation issues.

## What I expected to happen vs. what I found

**Expected**: v13 fix would be merged to main after previous director session identified it as ready.

**Found**:
1. No PR was created — researcher achieved breakthrough but didn't create PR
2. The fix is in `cross_role_policy.py`, but `capture_frames.py` and other tools use `machina_llm_roles_policy.py`
3. Running with wrong policy shows the old broken behavior

**Action taken**: Created PR #18 to merge v13 fixes to main.

## Issues updated this session

- **#16 (Hub Depletion)**: Added comment with verified results, linked to PR #18
- **PR #18**: Created PR for issue-16 branch → main

## Research roadmap after this session
```
## IMMEDIATE (merge + optimize existing cycle)
#18 PR: Hub Depletion Fix (IN REVIEW - merge ASAP)
  └─ enables make_heart cycle, +28% reward
#24 Balanced Mining (priority:1) — deposits are 30:1 skewed, make_heart needs 1:1
  └─ fixing this alone could double make_heart output
#10 Role Tuning (priority:1, unblocked after #18)
  └─ optimize deposit_to_hub, try 2A2M/1A3M

## NEAR-TERM (research directions to unlock next ceiling)
#20 Coordinated Exploration (priority:2) — agents block each other (55% move fail vs 0.2% solo)
#19 LLM Code Generation (priority:2) — runtime skill composition, not fixed menus
#21 Intrinsic Motivation (priority:2) — empowerment-driven exploration for junction discovery
#17 LLM Skill Validation (priority:2) — fixed in cross_role, low priority

## LONGER-TERM (paradigm shifts)
#22 Social Influence & Role Specialization (priority:3) — emergent roles from interaction
#23 Meta-Learning: In-Context Adaptation (priority:3) — agents learn across episodes
#15 8-Agent Scaling (priority:2, blocked by #10) — LLM contention
#11 Active Inference (priority:2, independent)
#12 Gear Acquisition (priority:3, deprioritized)
```

## New issues created this session
- **#19**: LLM Dynamic Code Generation — runtime skill composition (priority:2)
- **#20**: Coordinated Multi-Agent Exploration — spatial partitioning (priority:2)
- **#21**: Intrinsic Motivation & Empowerment — exploration drives (priority:2)
- **#22**: Social Influence & Role Specialization — emergent coordination (priority:3)
- **#23**: Meta-Learning: In-Context Adaptation — cross-episode learning (priority:3)
- **#24**: Balanced Mining Strategy — element diversity for make_heart (priority:1)

## Key replay insight driving these issues
Single-agent efficiency is 100x better than multi-agent: 0.2% vs 55% move failures, 43k vs 12k cells visited. The multi-agent degradation is the fundamental research challenge. Issues #20, #21, #22 all attack this from different angles.

## Open questions for next director

1. **Should cross_role become the default policy?** machina_llm_roles is still hardcoded in capture_frames.py.
2. **What's the optimal agent count?** 1 agent is most efficient per-step but 3-4 give more total reward. Is there a sweet spot?
3. **LLM model choice**: Could a more capable model (Claude/GPT-4) via #19 beat a faster model (Nemotron 49B) that makes more decisions per episode?
4. **2000 steps vs 1000**: v13 hits 1.08 at 2000 steps. Should we optimize for the longer horizon?

5. **8-agent scaling**: Still blocked by LLM contention (A40 GPU limit). With better navigation fixing single-agent efficiency first, then try 4-agent configs before 8.
