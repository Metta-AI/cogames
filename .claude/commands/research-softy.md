Deep-dive a specific game mechanic or strategy question. Updates knowledge files with findings.

## Usage
Provide a topic as argument: scoring, clips, energy, alignment, network, roles, map

## Steps

1. Read the relevant `knowledge/mechanics/` file for current understanding.

2. Read the game source code for the topic:
   - **scoring**: `src/cogames/games/cogs_vs_clips/missions/machina_1.py`, `game/territory/territory.py`
   - **clips**: `src/cogames/games/cogs_vs_clips/game/clips/clips.py`, `clips/ship.py`
   - **energy**: `src/cogames/games/cogs_vs_clips/game/energy.py`, `solar.py`, `days.py`
   - **alignment**: `src/cogames/games/cogs_vs_clips/game/teams/junction.py`, `territory/territory.py`
   - **network**: `src/cogames/games/cogs_vs_clips/game/territory/territory.py` (ClosureQuery)
   - **roles**: `src/cogames/games/cogs_vs_clips/game/roles/*.py`, `teams/gear_stations.py`
   - **map**: `src/cogames/games/cogs_vs_clips/missions/terrain.py`, `game/junction.py`

3. Cross-reference with current `softy.py` implementation.

4. Identify gaps between game rules and our implementation.

5. Update the relevant `knowledge/mechanics/` file with new findings.

6. If findings suggest improvements, add to `knowledge/experiments/hypotheses.md`.

7. Report findings concisely with source file references.
