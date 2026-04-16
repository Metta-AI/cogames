Analyze tournament match data to learn from competitive results. This reveals per-scenario strengths/weaknesses that local testing vs random cannot.

## AUTONOMY
- Run all phases without stopping. Write results directly.
- This can be triggered manually or auto-triggered every 5 cycles from /loop-softy.

## SUBAGENT PROTOCOL
Delegate the ENTIRE data fetch + computation to a Sonnet subagent. Only the synthesis and priority updates stay in Opus.

## Step 1: Fetch + Compute (delegated to Sonnet)
```
Agent(model: "sonnet", prompt: "Fetch and analyze Softy tournament data in /Users/maxwellstarr/cogames.

Run these commands:
  uv run cogames matches --policy Softy --limit 200 --json > /tmp/softy_matches.json
  uv run cogames leaderboard beta-cvc 2>&1 | head -25

Read /tmp/softy_matches.json. For each Softy version with 5+ matches:
- Group matches by num_agents (c2, c4, c6, c8)
- Compute: avg score, match count, min, max per scenario
- Tournament weights: c2≈32%, c4≈27%, c6≈33%, c8≈8%

Also compute:
- Regressions between consecutive versions per scenario
- Translation ratios: local VOR delta vs tournament VOR delta per version
- Scenario gap analysis: which scenario has most room for improvement
- Contribution of each scenario to overall VOR for best version

Return ALL of the following:
1. Per-version per-scenario table (version | c2 avg(N) | c4 avg(N) | c6 avg(N) | c8 avg(N) | weighted)
2. Regressions detected (list or 'none')
3. Translation ratios per version
4. Scenario contribution table for best version
5. Leaderboard top 5
6. Softy latest VOR and rank
7. Scenario-specific improvement opportunities (top 3)")
```

## Step 2: Synthesize + Update (in Opus)
Read the subagent results. Use creative judgment to:
- Identify if tournament data changes any priorities
- Generate hypotheses prioritized by TOURNAMENT-WEIGHTED impact
- Check if the "only bug fixes work" rule still holds

## Step 3: Persist
Write findings to `knowledge/experiments/tournament-audit.md` using this format:

```markdown
# Tournament Audit — [date]

## Scenario Breakdown
| Version | c2 avg (N) | c4 avg (N) | c6 avg (N) | c8 avg (N) | Overall |

## Scenario Contribution (best version)
- c2: X% of matches, avg Y
- c4: X% of matches, avg Y
- c6: X% of matches, avg Y
- c8: X% of matches, avg Y

## Regressions Detected
[list or "none"]

## Translation Ratios
[per-version local to tournament translation]

## Scenario-Specific Opportunities
1. [highest impact hypothesis with tournament-weighted estimate]

## Rules Update
[any amendments to improvement rules based on tournament data]
```

Update `knowledge/tactics/priorities.md` if tournament data changes priority order.
Regenerate `knowledge/orient-state.md`.
Print 5-line console summary.
