Run parallel hypothesis testing — benchmark 3 changes simultaneously, keep the winner.

## AUTONOMY RULES
- **NEVER ask permission** — edit, benchmark, decide, upload without pausing
- **NEVER enter plan mode** — just execute
- Keep all output concise — tables and 1-line summaries only

## SUBAGENT PROTOCOL
- All benchmarks → `Agent(model: "haiku")` — run in parallel, return VOR only
- Import verification → `Agent(model: "haiku")` — run in parallel, return pass/fail
- Hypothesis selection + decision stays in main Opus context

## Step 0: Orient
Read `knowledge/orient-state.md` (or fall back to first 20 lines of priorities.md + hypotheses.md).

## Step 1: Select 3 Hypotheses
Pick top 3 untested hypotheses that modify different parts of softy.py.
Print: `TESTING: P[X], P[Y], P[Z]` — one line each with hypothesis name.

## Step 2: Baseline
Delegate to Haiku subagent:
```
Agent(model: "haiku", prompt: "cd /Users/maxwellstarr/cogames && uv run cogames pickup -p 'class=softy.SoftyPolicy' --pool random --episodes 5 -m machina_1 -c 8. Return ONLY the VOR number.")
```

## Step 3: Create Variants
For each (N = 1, 2, 3):
1. `cp softy.py softy_variant_N.py`
2. Apply ONE change to softy_variant_N.py
3. Verify import — dispatch 3 parallel Haiku subagents:
   ```
   Agent(model: "haiku", prompt: "Run `uv run python -c \"import softy_variant_N; print('OK')\"` in /Users/maxwellstarr/cogames. Return pass/fail.")
   ```

## Step 4: Race (run ALL via parallel subagents)
Dispatch 3 parallel Haiku subagents, one per variant:
```
Agent(model: "haiku", prompt: "cd /Users/maxwellstarr/cogames && uv run cogames pickup -p 'class=softy_variant_N.SoftyPolicy' --pool random --episodes 5 -m machina_1 -c 8. Return ONLY the VOR number.")
```

## Step 5: Results + Micro-Audit
Print table:
```
| # | Hypothesis | VOR | Delta | Decision |
```
Then for each: `AUDIT [PNN]: [1 sentence why] MODEL: [1 sentence or "no update"]`

## Step 6: Decide
- Winner >= 5% above baseline: Apply to softy.py. Upload.
- All failed: Log all to failed.md.
- Close race (within 3%): Run 10ep confirmation for top 2 (via Haiku subagents).

## Step 7: Cleanup + Log
```
rm -f softy_variant_*.py
```
Log ALL results to knowledge/ files (log.md, hypotheses.md, failed.md or successful.md).
Regenerate `knowledge/orient-state.md`.
