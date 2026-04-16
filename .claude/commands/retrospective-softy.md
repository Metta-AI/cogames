Run a retrospective analysis of the Softy improvement loop. Re-examine past experiments with current context and identify opportunities we missed.

## SUBAGENT PROTOCOL
Delegate the heavy data reading to a Sonnet subagent. Keep synthesis + creative re-evaluation in Opus.

## Step 1: Data Gathering (delegated to Sonnet)
```
Agent(model: "sonnet", prompt: "Read these files in /Users/maxwellstarr/cogames and produce a structured analysis:

1. knowledge/experiments/failed.md — every failed experiment + root cause
2. knowledge/experiments/successful.md — every kept change + why it works
3. knowledge/experiments/hypotheses.md — full hypothesis queue with results
4. knowledge/experiments/log.md — complete experiment timeline
5. knowledge/tactics/priorities.md — current focus + rules

Categorize ALL tested hypotheses by type:
- Economy, Pathfinding, Defense, Scoring, Timing, Exploration

For each category compute: attempts, wins, win rate, best result, worst result.

Also identify:
- All near-miss candidates (+1.5% to +4% sub-threshold)
- All 'never retry' entries with their reasoning
- All capability improvements and whether they translated to tournament
- All behavior tuning and whether it translated

Return:
1. Category table (type | tested | won | rate | best | worst)
2. Near-miss candidate list (hypothesis, delta, p-value)
3. Capability vs behavior translation summary
4. 'Never retry' entries with reasoning (for re-evaluation)")
```

## Step 2: Creative Re-evaluation (in Opus)
Read the Sonnet subagent's structured output. Also read softy.py's current capabilities (relevant sections only, max 150 lines).

For EACH failed experiment in the near-miss list, ask:
> "Has anything changed since this failed that could make it work now?"

Check:
1. **Capability dependencies**: Did this fail because a capability was missing that we've SINCE added?
2. **Bug fix dependencies**: Did this fail because of a bug we've SINCE fixed?
3. **Near-misses**: Were any results 2-5% positive that might tip over with cumulative improvements?
4. **"Never retry" validation**: Re-examine reasoning with current context. Flag any where reasoning no longer holds.
5. **Successful experiment extensions**: For each kept change, is there a STRONGER version we haven't tried?

## Step 3: Strategy Check
Run leaderboard check:
```
uv run cogames leaderboard beta-cvc --policy Softy
uv run cogames leaderboard beta-cvc
```

Assess:
- Trajectory: converging, plateauing, or climbing?
- Gap to target and competitor landscape
- Is current approach still optimal?

## Step 4: Persist
Write to `knowledge/experiments/retrospective.md` (overwrite with latest):
```markdown
# Retrospective — [date]

## Pattern Summary
| Category | Tested | Won | Win Rate | Best | Worst |

**What works**: [summary]
**What fails**: [summary]
**Rule check**: "Capability > behavior tuning" — [still holds / needs update]

## Re-evaluation Candidates
[hypothesis — retry / modified retry / still skip, with rationale]

## Strategy Assessment
- Trajectory, current VOR, gap, approach recommendation

## Updated Priority Queue
1-3 items with rationale
```

Update `knowledge/experiments/hypotheses.md` — add re-eval candidates.
Update `knowledge/tactics/priorities.md` if priority queue changed.
Regenerate `knowledge/orient-state.md`.
Print concise console summary.
