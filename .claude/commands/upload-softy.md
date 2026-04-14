Upload current softy.py to the Alignment League leaderboard and log the result.

## Steps

1. Verify import first:
   ```
   uv run python -c "import softy; print('OK')"
   ```

2. Upload:
   ```
   uv run cogames upload -p "class=softy.SoftyPolicy" -f ~/cogames/softy.py -n Softy --season beta-cvc --skip-validation
   ```

3. Check submission status:
   ```
   uv run cogames submissions --season beta-cvc --policy Softy
   ```

4. **Log the upload** — append to `softy-log.md`:
   ```
   ### Upload: [date and time]
   - **Version**: Softy:vN
   - **Status**: [submitted/qualifying/competition]
   ```

5. Report the version number and status.
