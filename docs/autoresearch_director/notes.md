# Director Notes
_Written: 2026-03-31 (Session 5)_

## What I observed in the replay

### 3-agent pre-merge (machina_llm_roles, 1000 steps)
- **Reward: 0.5149/agent** (total 2.06)
- Linear growth ~0.05/100 steps, mild deceleration
- has_heart: 16.9% True (better than session 4's 2.3%)
- 300 stale exits, 118 LLM calls, 3 errors

### 3-agent post-merge (cross_role, 2A1M, 1000 steps)
- **Reward: 0.7055/agent** (total 2.117) — **+37% improvement!**
- has_heart: 19.2% True
- Stale exits: 25 (vs 300 pre-merge = 92% reduction)
- make_heart cycle active: mine_until_full (7 calls) + deposit_to_hub (5 calls)

### 8-agent pre-merge (machina_llm_roles, 4A4M scripted, 1000 steps)
- **Reward: 0.4195/agent** (total 3.356) — identical to session 4 (deterministic)
- has_heart: 3.2%, hub depletes at step 600, 724 stale exits

### 8-agent post-merge (cross_role, 4A4M LLM, 1000 steps)
- **Reward: 0.4043/agent** (total 3.234) — slightly WORSE than scripted miners
- has_heart: 10.7% (better but not enough)
- 345 stale exits, gear_up_miner: 36 calls many stale

## Current bottleneck

**Make_heart throughput can't keep up at 8 agents.** PR #18 merged and fixed 3-agent (+37%) but barely helped 8-agent. New bottleneck chain:
1. 4 aligners drain hearts faster than 4 miners produce via make_heart
2. LLM miners waste time on gear_up_miner stale exits; scripted miners are more reliable
3. Gear acquisition fails more often at 8-agent scale

## What I expected to happen vs. what I found

**Expected**: PR #18 merge would push 8-agent to ~0.60/agent (4.8 total).
**Found**: 3-agent got +37% (0.51→0.71) but 8-agent stayed flat (0.42→0.40). Make_heart can't keep pace with 4 aligners. The optimal 8-agent config needs cross_role aligners + scripted miners — not yet combined.

## Issues updated this session
- **#16**: Closed — PR #18 merged, verified +37% for 3-agent
- **PR #18**: Merged after 4 sessions identifying it as #1 bottleneck
- **#25**: Unblocked, updated with post-merge 8-agent data
- **#24**: Updated with session 5 context
- **#12**: Bumped priority:3→priority:2 (gear acquisition relevant for 8-agent)
- README leaderboard updated

## Open questions for next director
1. **Hybrid config**: Has anyone combined cross_role aligners + scripted miners? Highest-leverage experiment.
2. **Optimal aligner/miner split**: 3A5M or 5A3M might beat 4A4M with make_heart.
3. **gemma-3-12b + 8 agents**: +24% at 3 agents — does it scale to 8?
4. **Make_heart throughput**: How many hearts/1000 steps? Element-limited or time-limited?
5. **Gear contamination at scale**: Navigation issue or station congestion?
