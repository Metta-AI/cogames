Audit Softy's competitive position and identify the highest-leverage improvement.

## Steps

1. Read `softy-log.md` to understand recent performance trajectory.

2. Run local benchmark:
   ```
   uv run cogames pickup -p "class=softy.SoftyPolicy" --pool random --episodes 5 -m machina_1 -c 8
   ```

3. Check leaderboard position and gap to top:
   ```
   uv run cogames leaderboard beta-cvc --policy Softy
   uv run cogames leaderboard beta-cvc
   ```

4. Analyze the gap:
   - How many junctions does our 8v0 score imply we hold on average?
   - What's the gap to the top player?
   - Where are agents likely wasting time? (heartless camping, stuck, long travel, missed junctions)

5. Read `~/cogames/softy.py` and identify the single most impactful bottleneck.

6. **Log the audit** — append to `softy-log.md`:
   ```
   ### Audit: [date and time]
   - **Current VOR**: X.XX (rank #X)
   - **Gap to #1**: X.XX VOR
   - **Bottleneck identified**: [description]
   - **Recommended change**: [specific code change with line refs]
   - **Expected impact**: [estimate]
   ```

7. Recommend ONE specific code change with expected impact. Reference exact line numbers and functions.
