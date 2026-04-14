Run the autonomous Softy improvement loop with persistent knowledge.

## Critical: Knowledge-First Architecture
- All state lives in `knowledge/` files, NOT conversation context
- Read knowledge files at the START of every cycle
- Write results back to knowledge files AFTER every action
- Compact conversation after every 2 cycles
- If session dies, a new `/loop-softy` picks up from knowledge files

## Step 0: Orient (every session start)
Read these files in order:
1. `knowledge/tactics/priorities.md` — what to work on
2. `knowledge/experiments/log.md` — last 10 entries only
3. `knowledge/experiments/failed.md` — what to avoid
4. `knowledge/experiments/hypotheses.md` — available ideas
5. `knowledge/strategy/current-approach.md` — what we're doing

Report current state: rank, VOR, last change, next priority.

## Loop (repeat until stopped)

### 1. Select Change
Pick the highest-priority item from `knowledge/tactics/priorities.md` that:
- Is NOT in `knowledge/experiments/failed.md`
- Has a clear hypothesis and expected impact
- Can be implemented in a single code change

### 2. Baseline
```
uv run cogames pickup -p "class=softy.SoftyPolicy" --pool random --episodes 5 -m machina_1 -c 8
```

### 3. Implement
Make exactly ONE change in `softy.py`. Verify import.

### 4. Test
Re-run the same benchmark command.

### 5. Decide
- **Improved >= 5%**: KEEP. Upload via `/upload-softy`. Record in `knowledge/experiments/successful.md`.
- **Regressed or neutral**: REVERT. Record in `knowledge/experiments/failed.md` with root cause.

### 6. Update Knowledge
- Append result to `knowledge/experiments/log.md`
- Update `knowledge/tactics/priorities.md` (next action)
- Update `knowledge/strategy/current-approach.md` if strategy changed

### 7. Compact
After every 2 complete cycles, run `/compact`.
**Before compacting**: verify all results are written to knowledge files.

## Stopping Conditions
- 10 consecutive reverts → need strategic pivot, stop and report
- VOR > 15.0 → stretch target hit, celebrate and report
- User interrupts
