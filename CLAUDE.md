# Cogames — Softy Policy Development

## Project
Building **Softy**, a scripted multi-agent policy for the **Cogs vs Clips** game in the Softmax Alignment League (`beta-cvc` season). The policy competes on a tournament leaderboard against other teams' policies.

## Key Files
- `softy.py` — The policy implementation (941 lines). Single-file, no external dependencies beyond mettagrid.
- `softy-plan.html` — Strategy plan with game mechanics, role design, and architecture.
- `softy-log.md` — Improvement cycle log (append-only). Check this first when resuming.
- `.claude/commands/` — Slash commands for the improvement workflow.

## Architecture
- **SoftyCoordinator** — Shared team state (hub, junctions, extractors, targets)
- **SoftyState** — Per-agent persistent state (position, stuck detection, role phase)
- **SoftyAgentImpl** — Per-agent decision logic with role dispatch
- **SoftyPolicy** — Entry point, creates coordinator, assigns roles

## Current Role Distribution
3 miners + 5 aligners (no scrambler/scout — pure alignment is more efficient).

## Commands
```bash
# Quick import check
uv run python -c "import softy; print('OK')"

# Local benchmark (the standard test)
uv run cogames pickup -p "class=softy.SoftyPolicy" --pool random --episodes 3 -m machina_1 -c 8

# Starter baseline for comparison
uv run cogames pickup -p starter --pool random --episodes 3 -m machina_1 -c 8

# Upload to leaderboard
uv run cogames upload -p "class=softy.SoftyPolicy" -f ~/cogames/softy.py -n Softy --season beta-cvc --skip-validation

# Check leaderboard
uv run cogames leaderboard beta-cvc --policy Softy

# Check submissions
uv run cogames submissions --season beta-cvc --policy Softy
```

## Slash Commands
- `/test-softy` — Benchmark + log results
- `/audit-softy` — Competitive analysis + identify bottleneck
- `/improve-softy` — One improvement cycle (baseline → change → test → keep/revert)
- `/upload-softy` — Upload to leaderboard + log
- `/loop-softy` — Autonomous improvement loop (orchestrates all above with compaction)

## Improvement Workflow
The autonomous loop follows: **audit → improve → upload → compact**. Each cycle:
1. Read `softy-log.md` for current state (critical for session resilience)
2. Identify highest-leverage change
3. Make exactly ONE change
4. Re-benchmark
5. If VOR improves >= 5%: keep + upload. Otherwise: revert.
6. Log results to `softy-log.md`
7. Compact every 2 cycles to prevent context blowup

**All state lives in `softy-log.md`** — if a session dies, a new one picks up from the log.

## Game Facts (from source)
- Map: 88x88 procedural Machina arena
- Agents: 8 cogs vs automated clips
- Duration: 10,000 steps
- Reward: `junctions_held / max_steps` per tick (speed of capture is everything)
- Heart cost: 7 of each element (C, O, Ge, Si) = 28 total
- Align: 1 heart + aligner gear, must be within 15 cells of network or 25 of hub
- Energy: 20 base, 4/move, solar regen 1-3/tick
- HP: -1/tick outside friendly territory, full heal inside 10-cell AOE
- Observation: 13x13 egocentric, token-based

## Current Performance
Check `softy-log.md` for the latest. As of April 14, 2026:
- Softy:v5 at rank #224, VOR 6.03 (21 matches on leaderboard)
- Local VOR: ~0.81 (3 episodes vs random — leaderboard uses competitive matchmaking)
- Progression: v2(2.16) → v3(3.43) → v4(4.36) → v5(6.03)
- Target: >10.0 VOR (stretch), >20.0 for top-10
- Top of leaderboard: dinky:v27 at 27.21

## Known Limitations of softmax.auth
The installed `softmax-cli` package is behind the cogames code. Missing functions were shimmed locally in `.venv/lib/python3.12/site-packages/softmax/auth.py`. If you reinstall softmax-cli, re-add:
- `load_current_cogames_token`
- `DEFAULT_COGAMES_API_SERVER`
- `load_cogames_user_token`, `save_cogames_active_token`, `fetch_cogames_whoami`, `restore_cogames_user_session`
