Run the autonomous Softy improvement loop. This MUST run continuously without human intervention.

## AUTONOMY RULES (NON-NEGOTIABLE)
- **NEVER ask permission** to edit files, run benchmarks, upload, or make decisions
- **NEVER enter plan mode during execution** — plan mode OK for retrospective/audit only
- **NEVER pause for confirmation** — the process IS the confirmation
- **NEVER output more than 10 lines** between actions — keep reports tight
- **Edit files directly** using the Edit tool. Do NOT ask "shall I edit?" — just edit.
- **Run benchmarks directly** — do NOT ask "shall I run?" — just run.
- **Upload directly** when threshold is met — do NOT ask "shall I upload?"
- If a tool call is denied by the user, retry with a different approach — do NOT stop the loop.

## SUBAGENT PROTOCOL (TOKEN EFFICIENCY)
Mechanical tasks MUST run in subagents to keep the Opus context small. Subagents start with fresh context — no accumulated cache reads.

| Task | Model | Why |
|------|-------|-----|
| Benchmarks + result parsing | `Agent(model: "haiku")` | Runs bash, returns one number |
| Import verification | `Agent(model: "haiku")` | Runs one command, returns pass/fail |
| Paired same-seed testing (20+ seeds) | `Agent(model: "sonnet")` | Runs 40+ commands, computes statistics |
| Tournament data fetch + table computation | `Agent(model: "sonnet")` | Fetches 200 matches, builds tables |
| Retrospective pattern analysis | `Agent(model: "sonnet")` | Reads 1000+ lines of history, finds patterns |

**Only these stay in the main Opus conversation:**
- Orient (read orient-state.md — 25 lines)
- Hypothesis selection (creative judgment)
- Code editing of softy.py (implementation)
- KEEP/REVERT decision (micro-audit)
- Knowledge file updates (short writes)

## CONTEXT MANAGEMENT (CRITICAL)
- **Read ONLY what you need** for the current step — never dump entire files into context
- **knowledge/ files are the source of truth** — not conversation history
- **Write results to knowledge/ IMMEDIATELY** after each benchmark (before any other step)
- **Compact every cycle** — but ONLY after verifying results are persisted to knowledge/
- If you don't know the cycle number, read `knowledge/orient-state.md`

## Step 0: Orient (every session start, <30 seconds)
Read ONE file: `knowledge/orient-state.md` (under 30 lines).
It contains: cycle number, VOR, mode, consecutive fails, next hypotheses, recent results, do-not-retry list.

Print 3 lines: cycle number, current VOR, next hypothesis. Then GO.

## Loop (repeat until stopped)

### 0b. Depth Triggers (check before picking hypothesis)
- **consecutive_fails >= 5**: Switch to **paired testing** (20 same-seed episodes) instead of standard 10ep pickup
- **consecutive_fails >= 7**: Run `/deep-analysis-softy` FIRST, then resume loop with updated priorities
- **3+ compound candidates accumulated**: Run compound test (bundle all into one variant, 20 paired episodes)
- Read `knowledge/experiments/pattern-analysis.md` and `game-profile.md` when they exist

### 1. Pick Hypothesis
Take the highest-priority UNTESTED hypothesis from orient-state.md that:
- Is NOT in the do-not-retry list
- Is a **capability improvement** (not behavior tuning, not hub-reducing)
- Can be implemented in one edit

If you need more detail than orient-state.md provides, read `knowledge/tactics/priorities.md` (first 20 lines only).

If 3+ untested hypotheses exist, use parallel testing (Step 1b). Otherwise sequential (Steps 2-5).

### 1b. Parallel Path (3+ hypotheses available)
1. Copy softy.py → softy_variant_1.py, softy_variant_2.py, softy_variant_3.py
2. Apply one hypothesis per variant
3. Verify imports — use 3 parallel `Agent(model: "haiku")` subagents, one per variant:
   Prompt: "Run `uv run python -c \"import softy_variant_N; print('OK')\"` in /Users/maxwellstarr/cogames and return pass/fail."
4. Run baseline + all 3 variants via 4 parallel `Agent(model: "haiku")` subagents (see Step 2 format)
5. Collect results, pick winner, cleanup variant files
6. Skip to Step 6

### 2. Baseline
Delegate to a Haiku subagent. Use the Agent tool:
```
Agent(model: "haiku", prompt: "cd /Users/maxwellstarr/cogames && uv run cogames pickup -p 'class=softy.SoftyPolicy' --pool random --episodes 5 -m machina_1 -c 8. Run this command and return ONLY the VOR number from the output, nothing else.")
```
Record the VOR number from the subagent result.

### 3. Implement
Edit softy.py directly. ONE change only.

softy.py structure (use offset/limit to read ONLY the function being changed):
```
SoftyCoordinator: ~1-150
SoftyState: ~151-250
SoftyAgentImpl parsing: ~251-400
Aligner logic: ~401-550
Miner logic: ~551-650
Navigation: ~651-750
SoftyPolicy entry: ~751-850
Role assignment: ~851-950
Constants: ~951-1027
```

After editing, verify import via Haiku subagent:
```
Agent(model: "haiku", prompt: "Run `uv run python -c \"import softy; print('OK')\"` in /Users/maxwellstarr/cogames and tell me if it succeeded or failed. If failed, return the error message.")
```

### 4. Test
**Standard testing**: Delegate to Haiku subagent (same as Step 2).

**Paired testing (when consecutive_fails >= 5)**: Delegate the entire batch to a Sonnet subagent:
```
Agent(model: "sonnet", prompt: "Run paired same-seed benchmark in /Users/maxwellstarr/cogames.
Baseline: class=softy.SoftyPolicy
Variant: class=softy_variant_1.SoftyPolicy (or softy.SoftyPolicy if editing in-place — use the current softy.py)
Map: machina_1, scenario: c6, seeds 42 through 61 (20 pairs).

For each seed from 42 to 61:
  Run: uv run cogames scrimmage -p 'class=softy.SoftyPolicy' -m machina_1 -c 6 --seed $SEED -e 1 --format json
  Parse the score from JSON output.
  Run the same for the variant.
  Record the delta (variant - baseline).

After all 20 seeds: compute mean delta, standard deviation, paired t-test p-value, and count of positive deltas.
Return EXACTLY this format:
  mean_delta_pct: X.X%
  p_value: 0.XXX
  positive_seeds: N/20
  raw_deltas: [list of 20 numbers]")
```

### 4b. Compound Test (when 3+ compound candidates exist)
Bundle all confirmed sub-threshold improvements into one variant.
Delegate to Sonnet subagent: 20 paired episodes on c4 + c6 (60% tournament weight).
If compound > +3% with p < 0.05: apply to softy.py. If < +1%: discard.

### 5. Micro-Audit + Decide (3 lines max)
Print exactly:
```
AUDIT [PNN]: VOR X.XX → Y.YY (±Z%). [KEEP/REVERT]
WHY: [one sentence]
MODEL: [one sentence — new rule or "no update"]
```

- **Standard testing**: >= 3% improvement: Keep. < 3% or regression: Revert.
- **Paired testing thresholds**: p < 0.10 AND delta > +1%: KEEP. p > 0.30: NOISE (revert). 0.10-0.30: MARGINAL (add to compound list).
- Log to log.md. Update accumulated_delta in cycle_state.md.
- **UPLOAD STRATEGY**: Do NOT upload every win. Track accumulated local VOR delta across wins.
  Upload when: (a) accumulated delta >= +10% local VOR, OR (b) 3+ wins batched, OR (c) tournament validation specifically needed.
  This prevents match dilution (each version competes for match slots) and hides strategy from competitors.

### 6. Persist Results (BEFORE anything else)
Append to `knowledge/experiments/log.md` (4 lines max per entry).
Update `knowledge/experiments/hypotheses.md` — mark tested with result.
Update `knowledge/experiments/cycle_state.md` — increment cycle number.
Update `knowledge/tactics/priorities.md` if next priority changed.
**Regenerate `knowledge/orient-state.md`** — overwrite with current state:
  - cycle number, base version, local VOR, consecutive fails, mode
  - next 3 untested hypotheses (1 line each)
  - last 3 results (1 line each)
  - top 5 do-not-retry categories from failed.md

### 7. Retrospective (every 5 cycles)
If cycle number is divisible by 5, delegate to a Sonnet subagent:
```
Agent(model: "sonnet", prompt: "Read these files in /Users/maxwellstarr/cogames and provide pattern analysis:
- knowledge/experiments/log.md (last 30 entries)
- knowledge/experiments/failed.md (all entries)
- knowledge/experiments/hypotheses.md (tested entries only)
- knowledge/experiments/pattern-analysis.md (if exists)

Categorize all experiments by type (bug fix, capability, behavior tuning, economy, pathfinding, defense, scoring, timing, exploration).
For each category: count attempts, count wins, win rate.
List any near-miss candidates: experiments with +1.5% to +4% improvement that were below keep threshold.
Flag any rule violations (behavior tuning attempts, hub-reducing).
Check if any failed hypotheses are worth retrying given recent capability wins.

Return:
1. Category table (type | attempts | wins | rate)
2. Near-miss candidates (max 5, one line each)
3. Retry recommendations (max 3, one line each)
4. Updated priority suggestions (max 3, one line each)")
```
Use the subagent results to update priorities.md and write to `knowledge/experiments/retrospective.md`.

### 7b. Tournament Audit (every 5 cycles, after retrospective)
If cycle number is divisible by 5, delegate to a Sonnet subagent:
```
Agent(model: "sonnet", prompt: "Fetch and analyze Softy tournament data in /Users/maxwellstarr/cogames.

Run: uv run cogames matches --policy Softy --limit 200 --json
Also run: uv run cogames leaderboard beta-cvc

From the matches JSON, group by version + num_agents (c2/c4/c6/c8).
For each version with 5+ matches per scenario, compute avg score.
Tournament weights: c2≈32%, c4≈27%, c6≈33%, c8≈8%.

Return:
1. Per-version per-scenario table (version | c2 | c4 | c6 | c8 | weighted)
2. Regressions between consecutive versions (version X→Y regressed on cN by Z%)
3. Translation ratio (local VOR improvement vs tournament VOR improvement)
4. Current leaderboard top 5
5. Softy's latest tournament VOR and rank")
```
Use results to update `knowledge/experiments/tournament-audit.md` and priorities.md if needed.

### 8. Compact
After every cycle: verify knowledge files have all results, then compact conversation.
After compaction, re-orient from Step 0 (knowledge files have everything needed).

## Stopping Conditions
- 10 consecutive reverts with no new hypothesis categories → stop and report
- VOR > 15.0 → celebrate and report
- User interrupts

## Recovery
If context was lost or session restarted:
1. Read `knowledge/orient-state.md` for full current state
2. If orient-state.md is missing, read `knowledge/experiments/cycle_state.md` + last 5 entries of log.md
3. Resume from Step 0 — knowledge files have everything
