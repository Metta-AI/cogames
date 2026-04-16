# Cogames — Softy Policy Development

## AUTONOMY (READ THIS FIRST)
- This project uses an autonomous improvement loop (`/loop-softy`)
- **NEVER ask permission** to edit softy.py, run benchmarks, upload, or update knowledge files
- **NEVER enter plan mode during execution** — plan mode OK for retrospective/audit only
- **Edit directly, test directly, decide directly, upload directly**
- Knowledge files in `knowledge/` are the source of truth — persist everything there

## When Resuming
Read `knowledge/orient-state.md` (under 30 lines) for full current state: cycle, VOR, next hypotheses, recent results, do-not-retry rules.

## Key Files
- `softy.py` — The policy (single file, ~1027 lines)
- `knowledge/` — Persistent knowledge base (see `knowledge/README.md`)
- `.claude/commands/` — Slash commands for the improvement workflow

## Critical Rules
- **ONLY bug fixes and capability improvements translate to tournament** (11 for 11)
- **Behavior tuning HURTS in tournament** — 0 for 100+
- **Hub-reducing changes HURT** (hub provides safety under competitive pressure)
- **Test across c2/c4/c6/c8 before uploading** — weights: c2(31%), c4(27%), c6(33%), c8(9%)

## CLI
```bash
uv run python -c "import softy; print('OK')"  # import check
uv run cogames pickup -p "class=softy.SoftyPolicy" --pool random --episodes 5 -m machina_1 -c 8
uv run cogames leaderboard beta-cvc --policy Softy
uv run cogames upload -p "class=softy.SoftyPolicy" -f ~/cogames/softy.py -n Softy --season beta-cvc --skip-validation
```

## Slash Commands
- `/loop-softy` — Autonomous improvement loop (subagent-accelerated)
- `/improve-softy` — Single improvement cycle
- `/deep-analysis-softy` — Deep analysis (auto-triggers after 7+ fails)
- `/parallel-softy` — Test 3 hypotheses simultaneously
- `/retrospective-softy` — Re-examine past experiments (auto every 5 cycles)
- `/audit-tournament-softy` — Tournament match analysis (auto every 5 cycles)
- `/test-softy` — Benchmark + log
- `/upload-softy` — Upload to leaderboard
- `/audit-softy` — Competitive analysis
- `/research-softy` — Deep-dive specific mechanic
- `/status-softy` — Quick status check
