Run a deep analysis session. Use when consecutive_fails >= 7 or when standard loop is stuck.

## AUTONOMY RULES (same as /loop-softy)
- **NEVER ask permission** — edit, test, decide directly
- **NEVER enter plan mode** — execute immediately
- **Write results to knowledge/ IMMEDIATELY** after each phase

## SUBAGENT PROTOCOL
- Baseline profiling (Phase 2) → `Agent(model: "sonnet")` — runs 40 scrimmages, returns stats table
- Paired retesting (Phase 3) → `Agent(model: "sonnet")` per candidate — runs 20 paired seeds, returns p-value + delta
- Compound testing (Phase 4) → `Agent(model: "sonnet")` — runs 20 paired episodes, returns results
- Pattern mining (Phase 1) and synthesis (Phase 5) stay in Opus — they require creative judgment

## Phase 1: Pattern Mining (5 min, read-only)
Read `knowledge/experiments/log.md`, `failed.md`, `successful.md`.
1. Build category x result matrix (bug fix, behavior, deconfliction, hub-reducing, offense, role, exploration)
2. Extract directional rules (directions that ALWAYS succeed or ALWAYS fail)
3. Identify **near-miss candidates**: experiments showing +1.5% to +4% (sub-threshold but possibly real)
4. Map **negative space**: what directions/categories have NEVER been tried?
5. Write to `knowledge/experiments/pattern-analysis.md`

## Phase 2: Baseline Profiling (10 min)
Delegate to a Sonnet subagent:
```
Agent(model: "sonnet", prompt: "Run baseline profiling for Softy in /Users/maxwellstarr/cogames.
For each scenario (c2, c4, c6, c8), run seeds 42-51 (10 each):
  uv run cogames scrimmage -p 'class=softy.SoftyPolicy' -m machina_1 -c $C --seed $SEED -e 1 --format json

Parse each JSON result. Extract: rewards, deaths, alignments, hearts, unique cells, failed moves.
Compute per-scenario: mean, stdev, CV%, min, max.
Identify highest-variance scenario.
Return as a markdown table with one row per scenario.")
```
Write results to `knowledge/experiments/game-profile.md`.

## Phase 3: Paired Retesting (15 min)
Take top 3 near-miss candidates from Phase 1.
For each candidate, delegate to a Sonnet subagent:
```
Agent(model: "sonnet", prompt: "Run paired retesting in /Users/maxwellstarr/cogames.
Baseline: class=softy.SoftyPolicy
Variant: class=softy_variant_N.SoftyPolicy
Map: machina_1, scenario: c6, seeds 42-61 (20 pairs).
For each seed, run baseline then variant. Record score delta.
Compute: mean delta %, paired t-test p-value, positive seeds count.
Return: mean_delta_pct, p_value, positive_seeds/20, raw_deltas list.")
```

Classify each:
- p < 0.10 AND delta > +1%: **CONFIRMED**
- p < 0.10 AND delta <= 1%: **COMPOUND_CANDIDATE**
- p > 0.30: **NOISE** (confirmed dead)
- 0.10 < p < 0.30: **MARGINAL** (retest with more seeds or add to compound)

## Phase 4: Compound Testing (10 min)
Bundle all CONFIRMED + MARGINAL positives into one variant.
Delegate to Sonnet subagent: 20 paired episodes on c4 + c6 (60% of tournament weight).
- Compound > +3% with p < 0.05: proceed to full validation (c2+c4+c6+c8)
- Compound < +1%: effects were noise or canceling

## Phase 5: Synthesis (2 min)
- Ranked hypothesis list with confidence levels
- Update `knowledge/tactics/priorities.md` with paired-test-validated candidates
- Update `knowledge/experiments/cycle_state.md`
- Regenerate `knowledge/orient-state.md`
- Produce actionable next steps for `/loop-softy`
- Clean up variant files

## Decision Thresholds (paired testing)
| p-value | Delta | Action |
|---------|-------|--------|
| < 0.10 | > +1% | KEEP (apply to softy.py) |
| < 0.10 | <= 1% | Add to compound candidate list |
| 0.10-0.30 | any | Retest with more episodes OR compound |
| > 0.30 | any | REVERT (confirmed noise) |

## Output Files
- `knowledge/experiments/pattern-analysis.md` — category matrix + directional rules
- `knowledge/experiments/game-profile.md` — per-scenario baseline stats
- `knowledge/tactics/priorities.md` — updated with confidence-rated hypotheses
