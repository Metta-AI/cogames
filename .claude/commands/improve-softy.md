Execute one improvement cycle on softy.py. Make exactly ONE change, test it, and keep or revert.

## Rules
- Make exactly ONE targeted change per cycle
- Always establish a VOR baseline BEFORE making changes
- Always re-test AFTER the change
- If VOR improves >= 5%: keep the change and upload via `/upload-softy`
- If VOR regresses or is neutral: revert the change
- Log everything to the knowledge base

## Pre-reads (mandatory)
1. `knowledge/tactics/priorities.md` — what to work on
2. `knowledge/experiments/failed.md` — what NOT to try
3. `knowledge/experiments/hypotheses.md` — available ideas
4. `knowledge/strategy/current-approach.md` — active strategy

## Steps

1. Read `~/cogames/softy.py` to understand current state.

2. Select the highest-priority change from `knowledge/tactics/priorities.md` that is NOT in `knowledge/experiments/failed.md`.

3. Run baseline:
   ```
   uv run cogames pickup -p "class=softy.SoftyPolicy" --pool random --episodes 5 -m machina_1 -c 8
   ```

4. Implement the change in softy.py.

5. Verify import: `uv run python -c "import softy; print('OK')"`

6. Re-run benchmark (same command as step 3).

7. Compare VOR:
   - **Improved >= 5%**: Keep change. Run `/upload-softy`.
   - **Regressed or neutral**: Revert the edit.

8. **Update knowledge base**:
   - Append result to `knowledge/experiments/log.md`
   - If KEPT: move hypothesis to `knowledge/experiments/successful.md`, update `knowledge/strategy/current-approach.md`
   - If REVERTED: move hypothesis to `knowledge/experiments/failed.md` with root cause analysis
   - Update `knowledge/tactics/priorities.md` with next action
