# Current Approach: Softy v5

## Role Distribution
3 Miners + 5 Aligners (no scrambler, no scout)

### Rationale
- Scramblers cost 2 hearts per junction (scramble + re-align) vs aligners at 1 heart
- Scout discovery is handled by wandering aligners during heartless phases
- Pure alignment maximizes junction capture rate early game

### Known Weaknesses
1. **No clips defense** — zero protection against cascade failure (see clips-behavior.md)
2. **No map discovery** — aligners only discover junctions within their wander range
3. **Heart pipeline dependency** — all 5 aligners compete for hearts from 3 miners' output
4. **Night energy starvation** — agents outside territory crawl at 1 move per 4 ticks

## Key Features
- **SoftyCoordinator**: shared hub/junction/extractor positions, target deconfliction
- **Hub-relative coordinates**: all agents share a coordinate system calibrated to hub
- **Frontier scoring**: `score = frontier_count * 8.0 - distance` for junction targeting
- **Hub inventory awareness**: aligners check if hub can craft hearts before going
- **Stuck detection**: position history + failed-move tracking, rotation recovery
- **Failed junction blacklist**: junctions stuck on for 15+ ticks get skipped

## Architecture (softy.py, 941 lines)
- `SoftyCoordinator` — shared team state
- `SoftyState` — per-agent state (position, stuck, role phase)
- `SoftyAgentImpl` — decision logic with role dispatch
- `SoftyPolicy` — entry point, assigns roles via ROLE_CYCLE

## Performance
- Leaderboard: rank #224, VOR 6.03 (Softy:v5, 21 matches)
- Local: VOR 0.81 (3 episodes vs random)
- Top of board: dinky:v27 at 27.21 (4.5x gap)

## What Needs to Change
See `../tactics/priorities.md` for the ordered improvement queue.
The #1 priority is adding scramblers to prevent cascade failure.
