Run the autonomous Softy improvement loop. Each cycle: audit → improve → upload → compact.

## Critical: Session Resilience
- All state lives in `softy-log.md`, NOT in conversation context
- Read `softy-log.md` at the START of every cycle to get current state
- Compact conversation after every 2 cycles to prevent context blowup
- If the session dies, a new `/loop-softy` picks up from the log

## Loop Structure

Repeat the following cycle until told to stop:

### 1. Read State
Read `softy-log.md` to understand:
- Current VOR and leaderboard rank
- What changes have been tried (don't repeat failed ideas)
- The trajectory (improving? plateaued?)

### 2. Audit (every 3rd cycle, or first cycle)
Run `/audit-softy` to get a fresh competitive analysis. On other cycles, skip to step 3 using the last audit's recommendation.

### 3. Improve
Run `/improve-softy` to execute one change cycle. This handles:
- Baseline measurement
- ONE targeted change
- Re-measurement
- Keep/revert decision
- Logging to `softy-log.md`

### 4. Upload (if improved)
If the change was kept, `/upload-softy` handles the upload and logging.

### 5. Compact
After every 2 complete cycles, run `/compact` to compress conversation context. This prevents the compaction-freeze that killed the overnight loop.

### 6. Continue
Go back to step 1. The log file has everything needed to continue.

## Stopping Conditions
- Stop after 10 consecutive reverts (we've plateaued — need a new strategy)
- Stop if VOR exceeds 15.0 (stretch target hit — report to user)
- Stop if user interrupts

## On Startup
If `softy-log.md` already has entries, report the current state:
- Latest VOR and rank
- How many cycles have run
- Last change and result
Then continue the loop from where it left off.
