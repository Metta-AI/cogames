# Softy Improvement Loops

All automated loops that drive the improvement cycle. Each has a trigger, protocol, and exit condition.

## 1. Improvement Loop (`/loop-softy`)
**Trigger**: Manual start or continuation from previous cycle.
**Protocol**:
1. Read cycle_state.md + priorities.md
2. Generate hypothesis (bug fix or capability improvement ONLY)
3. Implement in softy.py
4. Run 20 paired same-seed tests on machina_1.clips, c6
5. Analyze: paired t-test, Cohen's d
6. Decide: KEEP (p<0.10 AND delta>+1%), REVERT (p>0.30), or NOISE
7. Log to log.md, update cycle_state.md, update priorities.md
8. Repeat

**Exit**: Manual stop or context compaction. Resumes from cycle_state.md.

## 2. Deep Analysis Loop (`/deep-analysis-softy`)
**Trigger**: 7+ consecutive fails in improvement loop.
**Protocol**:
1. Run extended game profiling (100+ episodes with detailed output)
2. Parse: deaths/agent, move failures, heart utilization, aligner cycle time, junction throughput
3. Statistical pattern mining across episodes
4. Generate NEW hypotheses from profiling data (not recycled ideas)
5. Test hypotheses with paired testing
6. On first win, exit deep analysis and return to improvement loop

**Exit**: First KEEP result. Updates priorities.md with profiling findings.

## 3. Discovery Loop (NEW)
**Trigger**: N consecutive fails after deep analysis has already been done (double-stall).
**Protocol**:
1. Run MASS simulations (100-200 episodes) with detailed logging
2. Build behavioral dataset: per-tick agent positions, actions, outcomes
3. Model patterns: where do agents spend time? What actions succeed/fail? Where do deaths cluster?
4. Compare top-scoring vs bottom-scoring episodes — what's different?
5. Identify structural inefficiencies invisible to hypothesis testing
6. Generate capability-improvement hypotheses from data patterns

**Exit**: Novel hypothesis identified that scores >+3% in preliminary test.

## 4. Retrospective Loop (`/retrospective-softy`)
**Trigger**: Every 5 cycles in improvement loop.
**Protocol**:
1. Review last 5 experiments
2. Look for patterns in failures (common themes, repeated mistakes)
3. Check if any near-miss results (p=0.10-0.20) could be compounded
4. Update failed.md if patterns suggest permanent dead-ends
5. Adjust priorities.md based on accumulated learning

**Exit**: Updated priorities, optionally compound test proposal.

## 5. Tournament Audit Loop (`/audit-tournament-softy`)
**Trigger**: Every 5 cycles, or after upload.
**Protocol**:
1. Pull tournament match data via leaderboard API
2. Per-scenario breakdown (c2, c4, c6, c8 separately)
3. Compare local VOR translation ratio per scenario
4. Detect regressions (v25 at 23.40 vs v24 at 25.60)
5. Identify scenarios where we're weakest vs strongest
6. Weight improvement priorities by tournament scenario distribution

**Exit**: Updated tournament-audit.md, optional revert recommendation.

## 6. Parallel Testing (`/parallel-softy`)
**Trigger**: 3+ hypothesis candidates available simultaneously.
**Protocol**:
1. Create variant files (softy_pNNN.py) for each hypothesis
2. Run all variants + baseline in parallel (4 batches x 5 seeds)
3. Analyze all results simultaneously
4. Keep winners, revert losers
5. If multiple winners, compound test them together

**Exit**: All hypotheses tested and decided.

## Loop Interactions
- Improvement → Deep Analysis: triggered by 7+ consecutive fails
- Deep Analysis → Discovery: triggered by double-stall (deep analysis didn't break the streak)
- Improvement → Retrospective: every 5 cycles
- Improvement → Tournament Audit: every 5 cycles or post-upload
- Parallel Testing: can be invoked from any loop when multiple candidates exist

## Key Rules Across All Loops
- **ONLY bug fixes and capability improvements** — behavior tuning is 0/100+
- **Paired same-seed testing** — 20 seeds, machina_1.clips, paired t-test
- **KEEP threshold**: p<0.10 AND delta>+1%
- **REVERT threshold**: p>0.30
- **Dashboard update**: around context compaction boundaries
- **Batch uploads**: accumulate 3-5 wins before uploading to prevent match dilution
