Audit Softy's competitive position and identify the highest-leverage improvement.

## Pre-reads (mandatory)
1. `knowledge/strategy/current-approach.md` — active strategy
2. `knowledge/experiments/failed.md` — what NOT to recommend
3. `knowledge/experiments/hypotheses.md` — existing improvement ideas
4. `knowledge/strategy/competitor-analysis.md` — competitive landscape

## Steps

1. Run local benchmark:
   ```
   uv run cogames pickup -p "class=softy.SoftyPolicy" --pool random --episodes 5 -m machina_1 -c 8
   ```

2. Check leaderboard position and gap to top:
   ```
   uv run cogames leaderboard beta-cvc --policy Softy
   uv run cogames leaderboard beta-cvc
   ```

3. Analyze the gap:
   - How many junctions does our score imply we hold on average?
   - What's the gap to the top player?
   - Where are agents wasting time? (heartless camping, stuck, cascade losses, slow exploration)

4. Read `~/cogames/softy.py` and identify the single most impactful bottleneck.

5. Cross-reference with `knowledge/experiments/failed.md` — do NOT recommend anything listed there.

6. **Log the audit** — append to `knowledge/experiments/log.md`:
   ```
   ### Audit: [date and time]
   - **Current VOR**: X.XX (rank #X)
   - **Gap to #1**: X.XX VOR
   - **Bottleneck identified**: [description]
   - **Recommended change**: [specific code change with line refs]
   - **Expected impact**: [estimate]
   ```

7. If the recommendation is new, add it to `knowledge/experiments/hypotheses.md`.

8. Update `knowledge/strategy/competitor-analysis.md` with any new leaderboard findings.

9. Recommend ONE specific code change with expected impact. Reference exact lines.
