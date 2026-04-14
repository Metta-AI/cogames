# Cogames — Softy Policy Development

## Current Performance
- **Softy:v5** | Rank #224 | VOR 6.03 | Target: >15 VOR
- Top of board: dinky:v27 at 27.21 (4.5x gap)
- Trajectory: v2(2.16) → v3(3.43) → v4(4.36) → v5(6.03)

## When Resuming, Read These First
1. **`knowledge/tactics/priorities.md`** — what to work on next
2. **`knowledge/experiments/failed.md`** — what NOT to try
3. **`knowledge/strategy/current-approach.md`** — active strategy

## What This Is
Softy: a scripted multi-agent policy for Cogs vs Clips (Alignment League, beta-cvc season).
Goal: maximize VOR by capturing and holding **network-connected** junctions.

## Key Files
- `softy.py` — The policy (941 lines, single file, no external deps beyond mettagrid)
- `knowledge/` — Persistent knowledge base (see `knowledge/README.md` for index)
- `.claude/commands/` — 7 slash commands for the improvement workflow

## Architecture (4 classes in softy.py)
- **SoftyCoordinator** — shared team state (hub, junctions, extractors, targets)
- **SoftyState** — per-agent state (position, stuck detection, role phase)
- **SoftyAgentImpl** — per-agent decision logic with role dispatch
- **SoftyPolicy** — entry point, creates coordinator + role assignment (ROLE_CYCLE)

## Current Role Distribution
3 miners + 5 aligners. No scrambler/scout. **Needs scramblers** — see cascade failure risk in `knowledge/mechanics/clips-behavior.md`.

## Scoring (critical)
```
reward = (net-connected junctions - 1) / 10000, per tick
```
ONLY junctions with `net:cogs` tag count. Network = closure from hub through team junctions within 25 cells. Alignment requires 15 cells from net (or 25 from hub). **Disconnected junctions = zero reward.**

## Slash Commands
- `/test-softy` — Benchmark + log to knowledge base
- `/audit-softy` — Competitive analysis + identify bottleneck
- `/improve-softy` — One improvement cycle (baseline → change → test → keep/revert)
- `/upload-softy` — Upload to leaderboard + log
- `/loop-softy` — Autonomous improvement loop with knowledge + compaction
- `/research-softy` — Deep-dive specific game mechanic
- `/status-softy` — Quick check without benchmarks

## CLI Commands
```bash
uv run python -c "import softy; print('OK')"  # import check
uv run cogames pickup -p "class=softy.SoftyPolicy" --pool random --episodes 5 -m machina_1 -c 8
uv run cogames leaderboard beta-cvc --policy Softy
uv run cogames upload -p "class=softy.SoftyPolicy" -f ~/cogames/softy.py -n Softy --season beta-cvc --skip-validation
```

## Game Quick Reference
| Param | Value |
|-------|-------|
| Map | 88x88, ~69 junctions, 10K steps |
| Clips | 4 ships, scramble+align every 200 ticks from tick 100 |
| Hearts | 7 each element = 28 total. Hub starts with 24 each + 5 hearts |
| Align | 1 heart + gear, within 15 of net or 25 of hub |
| Territory | 10-cell AOE (junctions), 20-cell (hub), +100 HP/tick inside |
| Energy | 20 base, 4/move, solar 3 day / 1 night (200-tick cycle) |
| HP | 100 max, starts 50, -1/tick outside territory |
| **Cascade** | Scrambling one junction disconnects all junctions beyond it |

## Known Env Limitations
`softmax-cli` was shimmed locally in `.venv/.../softmax/auth.py`. If reinstalled, re-add: `load_current_cogames_token`, `DEFAULT_COGAMES_API_SERVER`, `load_cogames_user_token`, `save_cogames_active_token`, `fetch_cogames_whoami`, `restore_cogames_user_session`.
