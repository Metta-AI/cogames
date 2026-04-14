Quick status check on Softy without running benchmarks.

## Steps

1. Read `knowledge/experiments/log.md` — last 5 entries.

2. Read `knowledge/tactics/priorities.md` — current priorities.

3. Check leaderboard:
   ```
   uv run cogames leaderboard beta-cvc --policy Softy
   ```

4. Check submissions:
   ```
   uv run cogames submissions --season beta-cvc --policy Softy
   ```

5. Report:
   - **Leaderboard**: rank and VOR
   - **Last change**: what was tried and result
   - **Next priority**: from priorities.md
   - **Gap to #1**: current vs top
   - **Versions submitted**: count and trajectory
