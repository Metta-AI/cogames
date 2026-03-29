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
#18 PR: Hub Depletion Fix (IN REVIEW - merge ASAP)
  └─ enables make_heart cycle, +28% reward

#10 Role Tuning (priority:1, unblocked after #18 merge)
  └─ optimize deposit_to_hub, try 2A2M/1A3M compositions
  └─ blocks #15 8-Agent Scaling

#17 LLM Skill Validation (priority:2, independent)
  └─ Already fixed in cross_role — only affects old machina_llm_roles

#15 8-Agent Scaling (priority:2, blocked by #10)

#11 Active Inference (priority:2, independent)

#12 Gear Acquisition (priority:3, deprioritized)
```

## Open questions for next director

1. **Should cross_role become the default policy?** It has all the fixes, but machina_llm_roles is still hardcoded in capture_frames.py and possibly other scripts.

2. **Why 48 stale exits remain?** Navigation still fails regularly. Root causes:
   - BFS can't find paths through unexplored territory
   - Resource extractors (📦) are invisible obstacles
   - Agent clustering near hub creates congestion

3. **2A1M vs other compositions**: v13 used 2A1M (2 aligners, 1 miner). With make_heart working, would 1A2M or 2A2M be better? More miners = more resources = more hearts.

4. **2000 steps vs 1000**: At 2000 steps, v13 achieved 1.08/agent (exceeds 0.92 target). Is the goal to optimize for 1000 steps or accept longer episodes?

5. **8-agent scaling**: Still blocked by LLM contention (A40 GPU limit). With better navigation fixing single-agent efficiency first, then try 4-agent configs before 8.
