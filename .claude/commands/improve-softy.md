Execute one improvement cycle on softy.py. Make exactly ONE change, test it, and keep or revert.

## Rules
- Make exactly ONE targeted change per cycle
- Always establish a VOR baseline BEFORE making changes
- Always re-test AFTER the change
- If VOR improves >= 5%: keep the change and upload via `/upload-softy`
- If VOR regresses or is neutral: revert the change
- Log everything to `softy-log.md`

## Steps

1. Read `softy-log.md` to see what's been tried and the current trajectory.

2. Read `~/cogames/softy.py` to understand current state.

3. Run baseline:
   ```
   uv run cogames pickup -p "class=softy.SoftyPolicy" --pool random --episodes 5 -m machina_1 -c 8
   ```

4. Identify ONE high-leverage change. Priority areas:
   - Aligner target selection (frontier expansion value)
   - Miner efficiency (deposit threshold, element cycling)
   - Heartless aligner behavior (explore vs wait)
   - Stuck detection sensitivity
   - Network expansion strategy (hub radius, net radius)
   - Role distribution tuning

5. Implement the change in softy.py.

6. Verify import: `uv run python -c "import softy; print('OK')"`

7. Re-run benchmark (same command as step 3).

8. Compare VOR:
   - **Improved >= 5%**: Keep change. Run `/upload-softy`.
   - **Regressed or neutral**: Revert the edit. Note why.

9. **Log the cycle** — append to `softy-log.md`:
   ```
   ### Improve: [date and time]
   - **Baseline VOR**: X.XX
   - **Change**: [what and why, with line refs]
   - **New VOR**: X.XX
   - **Delta**: +/-X.XX (+/-X%)
   - **Decision**: KEPT / REVERTED
   - **Uploaded**: vN (if kept)
   - **Next recommended**: [what to try next]
   ```
