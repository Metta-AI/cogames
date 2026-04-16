Upload current softy.py to the Alignment League leaderboard. Run immediately — no confirmation needed.

## Steps

1. Verify import: `uv run python -c "import softy; print('OK')"`

2. Upload:
   ```
   uv run cogames upload -p "class=softy.SoftyPolicy" -f ~/cogames/softy.py -n Softy --season beta-cvc --skip-validation
   ```

3. Check submission:
   ```
   uv run cogames submissions --season beta-cvc --policy Softy
   ```

4. Append to `knowledge/experiments/log.md` (2 lines max):
   ```
   ### Upload: Softy:vN — [date]
   Status: [submitted/qualifying/competition]
   ```

5. Print: `UPLOADED Softy:vN — [status]`
