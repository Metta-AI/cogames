# Director Notes
_Written: 2026-03-28 (updated with replay observations)_

## What I observed in the replay

Ran 3 configurations on current main:
- 3 aligners, 1000 steps → 0.563/agent (1.689 total)
- 3 aligners, 400 steps → 0.259/agent
- 1 aligner, 500 steps → 0.294/agent
- 8 agents → CRASHED (LLM ReadTimeout)

### Agent behavior timeline (3A, 1000 steps)
- **Steps 0-100**: All 3 agents gear up and start exploring. A0 heads south, A1/A2 head north-west.
- **Steps 100-200**: Agents return to hub area, start get_heart/align_neutral cycles. First junctions aligned.
- **Steps 200-300**: A1 reaches row 45, col 48 and gets stuck. A2 returns to hub.
- **Steps 300-400**: A2 settles on gear station row (row 52-53). Both A1 and A2 now stuck.
- **Steps 400-1000**: Only A0 moves. A1 at EXACT same position for 700 steps. A2 on gear station row for 600 steps. Reward growth decays from 0.08→0.03 per 100 steps.

### The get_heart death loop
55 stale exits total. has_heart=False for 81% (51/63) of LLM decisions. Once hub's 5 hearts are consumed (~step 200-300), agents loop: get_heart → stale after 20 steps → unstuck → explore → get_heart → forever.

### Corner junctions unreachable
4 junctions at map corners (rows 6-7, 91-92). Agents spawn at rows 49-52. In 1000 steps, NO agent gets within 30 rows of any corner junction. The policy explores only ~20 cells from center.

### LLM decision pattern
- Skill selections: explore(37), get_heart(27), align_neutral(24), gear_up(4)
- LLM hallucinated "unstick" instead of "unstuck" once (mapped to explore by fallback)

## Current bottleneck
**Hub depletion → get_heart death loop.** This is MORE specific than my initial assessment of "skill timeout waste." The hub has 5 hearts and they're consumed by step 200-300. After that, 2 of 3 agents become permanently stuck because the LLM keeps choosing get_heart (which always fails) instead of exploring or defending.

## What I expected to happen vs. what I found
Initial analysis (from TSV) suggested the bottleneck was skill timeouts (24+26 = 50 timeouts wasting 2500 steps). The replay CONFIRMED this but revealed the ROOT CAUSE: hub depletion. The timeouts aren't random — they're ALL from get_heart failing because the hub is empty. Fixing get_heart without fixing hub awareness will just change the timeout count, not the outcome.

## Issues updated this session
- **#9 (Cross-Role Policy):** CLOSED. 18 variants, best 0.55 vs baseline 0.66.
- **#12 (Gear Acquisition):** Deprioritized. PR reverted.
- **#10 (Role Tuning):** Reprioritized to top. Added 4 experiments + detailed replay observations.
- **#15 (8-Agent Scaling):** NEW. Blocked by #10.
- **#16 (Hub Depletion Awareness):** NEW. Highest-leverage fix from replay observation. Can be worked on independently or as part of #10.

## Research roadmap after this session
```
#16 Hub Depletion Awareness (priority:1, HIGHEST LEVERAGE)
#10 Role Tuning (priority:1, umbrella for #16 + other fixes)
  └─ blocks #15 8-Agent Scaling
#11 Active Inference (independent, priority:2)
#12 Gear Acquisition (deprioritized)
#9  Cross-Role Policy (CLOSED)
```

## Open questions for next director
1. **Can hearts be crafted from deposited resources?** If yes, a miner could create more hearts beyond the initial 5. This would break the hard ceiling.
2. **Why do agents cluster?** All 3 agents stay within 20 cells of each other. Is there a coordination mechanism to spread them across different map quadrants?
3. **Are there more than 4 junctions?** Only 4 visible at map corners. Best runs aligned 7+, suggesting junctions are discovered during exploration. How many total junctions exist?
4. **Reward normalization question still open** — March 21 vs March 22 use different normalization.
5. **Move-failure tracking** from March 22 session — is it in main or does it need cherry-picking?
