Run the Softy VOR benchmark and log results.

## Steps

1. Verify import: `uv run python -c "import softy; print('OK')"`

2. Run the full benchmark:
   ```
   uv run cogames pickup -p "class=softy.SoftyPolicy" --pool random --episodes 5 -m machina_1 -c 8
   ```

3. Run starter baseline for comparison:
   ```
   uv run cogames pickup -p starter --pool random --episodes 5 -m machina_1 -c 8
   ```

4. Check current leaderboard position:
   ```
   uv run cogames leaderboard beta-cvc --policy Softy
   ```

5. **Log results** — append to `softy-log.md`:
   ```
   ### Benchmark: [date and time]
   - **Softy VOR**: X.XX (8v0 score: X.XX)
   - **Starter VOR**: X.XX
   - **Improvement over starter**: X.Xx
   - **Leaderboard**: Softy:vN at rank #X, VOR X.XX
   ```

6. Report in this format:
   - **Softy VOR**: X.XX (8v0 score: X.XX)
   - **Starter VOR**: X.XX
   - **Improvement over starter**: X.Xx
   - **Leaderboard rank**: #X (VOR X.XX)
