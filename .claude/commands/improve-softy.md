Execute one improvement cycle on softy.py. ONE change, test, keep or revert.

## AUTONOMY RULES
- **NEVER ask permission** — edit, benchmark, decide, upload without pausing
- **NEVER enter plan mode** — just execute
- Keep output concise — micro-audit format only

## SUBAGENT PROTOCOL
- Benchmarks → `Agent(model: "haiku")` — returns VOR number only
- Import checks → `Agent(model: "haiku")` — returns pass/fail only
- Keep hypothesis selection + code editing + decision in main Opus context

## Pre-reads (first 20 lines each)
1. `knowledge/orient-state.md` — full current state (if exists, use instead of below)
2. `knowledge/tactics/priorities.md` — what to work on
3. `knowledge/experiments/failed.md` — what NOT to try (last 10 entries only)

## Steps

1. Read relevant section of softy.py (NOT the whole file — just the function you're changing, max 150 lines with offset/limit).

2. Select highest-priority change NOT in failed.md.

3. Baseline — delegate to Haiku subagent:
   ```
   Agent(model: "haiku", prompt: "cd /Users/maxwellstarr/cogames && uv run cogames pickup -p 'class=softy.SoftyPolicy' --pool random --episodes 5 -m machina_1 -c 8. Run this command and return ONLY the VOR number.")
   ```

4. Implement the change in softy.py. Verify import via Haiku subagent:
   ```
   Agent(model: "haiku", prompt: "Run `uv run python -c \"import softy; print('OK')\"` in /Users/maxwellstarr/cogames and return pass/fail.")
   ```

5. Re-run benchmark via Haiku subagent (same as Step 3).

6. Micro-audit (print exactly):
   ```
   AUDIT [PNN]: VOR X.XX → Y.YY (±Z%). [KEEP/REVERT]
   WHY: [one sentence]
   MODEL: [one sentence or "no update"]
   ```

7. If >= 5%: Keep. Upload via `uv run cogames upload -p "class=softy.SoftyPolicy" -f ~/cogames/softy.py -n Softy --season beta-cvc --skip-validation`
   If < 5%: Revert the edit.

8. Update knowledge files: log.md, hypotheses.md, failed.md or successful.md, priorities.md.
   Regenerate `knowledge/orient-state.md` with current state.
