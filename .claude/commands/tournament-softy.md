Multi-round tournament: parallel screen, combine winners, fine-tune. The exponential growth engine.

## Round 1: Parallel Screen (3 hypotheses)
Run `/parallel-softy` to test top 3 untested hypotheses.
Record the winner and any close runners-up.

## Round 2: Combinations
Take the Round 1 winner and combine it with:
- Runner-up from Round 1 (if any showed promise, even if < 5%)
- Next untested hypothesis from the queue

Create 3 combination variants:
1. Winner + Runner-up
2. Winner + Next-in-queue
3. Winner + Runner-up + Next-in-queue (triple stack)

Run parallel benchmark (same process as /parallel-softy Steps 3-6).

## Round 3: Fine-tune
Take the best combination from Round 2. Create 3 parameter-tuning variants:
- Variant A: Tune the key parameter +20%
- Variant B: Tune the key parameter -20%
- Variant C: Adjust a secondary parameter

Run parallel benchmark.

## Round 4: Confirm & Upload
Take the tournament champion. Run a 10-episode confirmation:
```
uv run cogames pickup -p "class=softy.SoftyPolicy" --pool random --episodes 10 -m machina_1 -c 8
```

If confirmed >= 5% above the pre-tournament baseline:
```
uv run cogames upload -p "class=softy.SoftyPolicy" -f ~/cogames/softy.py -n Softy --season beta-cvc --skip-validation
```

## Logging
After each round, update ALL knowledge files (log.md, failed.md, successful.md, priorities.md).
Document the full tournament bracket in `knowledge/experiments/log.md`.

## Stopping Conditions
- Round 1 produces no winners → stop, report, pivot strategy
- Round 2 combinations all regress vs Round 1 winner → skip to Round 4 (upload Round 1 winner as-is)
- Any round produces > 30% improvement → fast-track to confirmation

## Cleanup
```
rm -f softy_variant_*.py
```
Run after EVERY round — never leave variant files.
