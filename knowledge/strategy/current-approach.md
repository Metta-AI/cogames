# Current Approach: Softy v24 (P12+P18+P15+P38rev+P65+P73+P76+P89+P102+P103+P104)

## Role Distribution
Dynamic (P102+P103): c1=0M/1A, c2=1M/1A, c3=1M/2A, c4=2M/2A, c5+=ROLE_CYCLE (3M+rest A).

### Rationale
- Scramblers cost 2 hearts per junction (scramble + re-align) vs aligners at 1 heart
- Scout discovery is handled by wandering aligners during heartless phases
- Pure alignment maximizes junction capture rate early game
- P1 tested adding a scrambler — -43% regression from losing a miner

### Known Weaknesses
1. **No clips defense** — zero protection against cascade failure (see clips-behavior.md)
2. **No map discovery** — aligners only discover junctions within their wander range
3. **Stale junction data** — off-screen junctions can be scrambled without our knowledge
4. **Junction scoring ignores network connectivity** — `_in_align_range` treats disconnected cogs junctions as connected

## Key Features
- **SoftyCoordinator**: shared hub/junction/extractor positions, target deconfliction
- **Hub-relative coordinates**: all agents share a coordinate system calibrated to hub
- **Frontier scoring**: `score = frontier_count * 8.0 - distance` for junction targeting
- **Hub inventory awareness**: aligners check if hub can craft hearts before going
- **Dynamic bottleneck mining (P12)**: miners target whichever element has lowest hub inventory
- **Stuck detection**: position history + failed-move tracking, rotation recovery
- **Failed junction blacklist**: junctions stuck on for 15+ ticks get skipped

## Architecture (softy.py, ~955 lines)
- `SoftyCoordinator` — shared team state + bottleneck_element()
- `SoftyState` — per-agent state (position, stuck, role phase)
- `SoftyAgentImpl` — decision logic with role dispatch
- `SoftyPolicy` — entry point, assigns roles via ROLE_CYCLE

## Performance
- **Leaderboard: v24 rank #2, VOR 25.60.** Gap to #1 (dinky:v27, 27.31) = 6.7%.
- v25 (v24+P127) at 23.40, v26 (v25+P139) at 23.52 — both REGRESSED. P127+P139 REVERTED.
- **RESEARCH MODE**: No uploads until validated across c2+c4+c6+c8.
- **10 total wins** (P127/P139 reverted from count). All bug fixes / capability improvements.

## What Needs to Change
See `../tactics/priorities.md` for the ordered improvement queue.
Key areas: **network resilience** (cascade defense), **aligner efficiency** (pre-positioning), **multi-scenario validation**.
All code review bugs exhausted (P134/P32/P140/P87 all tested). Need capability improvements.

## Confirmed Game Mechanics (from source audit)
- Walls have `type:wall` tag (index 20) — already blocked by BFS via `set(tags)`
- Alignment range: 15 from net junction, 25 from hub (ALIGN_NET_RADIUS=20 is WIDER than actual)
- Network connectivity: 25-cell edges between cogs junctions (confirmed correct)
- VOR is count-weighted: c8 = 22.2%, c1 = 2.8% of local VOR
- Tournament scoring may weight differently — dinky's c1/c2 dominance suggests it matters there
