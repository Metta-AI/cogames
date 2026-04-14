Run the Softy VOR benchmark, log results to the knowledge base.

## Steps

1. Read `knowledge/tactics/priorities.md` to understand current context.

2. Verify import: `uv run python -c "import softy; print('OK')"`

3. Run the full benchmark:
   ```
   uv run cogames pickup -p "class=softy.SoftyPolicy" --pool random --episodes 5 -m machina_1 -c 8
   ```

4. Run starter baseline for comparison:
   ```
   uv run cogames pickup -p starter --pool random --episodes 5 -m machina_1 -c 8
   ```

5. Check current leaderboard position:
   ```
   uv run cogames leaderboard beta-cvc --policy Softy
   ```

6. **Log results** — append to `knowledge/experiments/log.md`:
   ```
   ### Benchmark: [date and time]
   - **Softy VOR**: X.XX (8v0 score: X.XX)
   - **Starter VOR**: X.XX
   - **Improvement over starter**: X.Xx
   - **Leaderboard**: Softy:vN at rank #X, VOR X.XX
   ```

7. Report concisely: VOR, 8v0 score, leaderboard rank, gap to top.
