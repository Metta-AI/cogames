"""Classic Overcogged map definitions.

Provides the original hand-crafted kitchen layout plus a procedural fallback
for larger agent counts.
"""

from __future__ import annotations

import random

from mettagrid.map_builder.ascii import AsciiMapBuilderConfig

CHAR_TO_MAP_NAME = {
    "#": "wall",
    ".": "empty",
    "P": "hub",
    "M": "miner_station",
    "D": "scrambler_station",
    "S": "junction",
    "I": "carbon_extractor",
    "C": "chest",
    "@": "agent.agent",
}

LEVEL_1: list[list[str]] = [
    list("###########"),
    list("#..M.S.D..#"),
    list("#.........#"),
    list("#..CCCCC..#"),
    list("#P..@.@..P#"),
    list("#P.......P#"),
    list("#..CCCCC..#"),
    list("#.........#"),
    list("#.I.I.I.I.#"),
    list("###########"),
]


def _level_1_map(num_agents: int) -> AsciiMapBuilderConfig:
    """Return the fixed classic layout, placing extra agents on open tiles."""

    grid = [row[:] for row in LEVEL_1]
    existing = sum(cell == "@" for row in grid for cell in row)

    if num_agents > existing:
        open_tiles = [(r, c) for r in range(1, len(grid) - 1) for c in range(1, len(grid[0]) - 1) if grid[r][c] == "."]
        rng = random.Random(42)
        rng.shuffle(open_tiles)
        for r, c in open_tiles[: num_agents - existing]:
            grid[r][c] = "@"
    elif num_agents < existing:
        removed = 0
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if cell != "@":
                    continue
                if removed >= num_agents:
                    grid[r][c] = "."
                removed += 1

    return AsciiMapBuilderConfig(map_data=grid, char_to_map_name=CHAR_TO_MAP_NAME)


def _procedural_map(num_agents: int, seed: int | None = None) -> AsciiMapBuilderConfig:
    """Build a procedural classic layout for larger agent counts."""

    rng = random.Random(seed)

    base = max(10, 6 + num_agents * 2)
    inner_w = min(base, 20)
    inner_h = min(base, 16)
    width = inner_w + 2
    height = inner_h + 2

    grid = []
    for r in range(height):
        row: list[str] = []
        for c in range(width):
            if r == 0 or r == height - 1 or c == 0 or c == width - 1:
                row.append("#")
            else:
                row.append(".")
        grid.append(row)

    wall_adjacent = []
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            if r == 1 or r == height - 2 or c == 1 or c == width - 2:
                wall_adjacent.append((r, c))

    rng.shuffle(wall_adjacent)
    placed = 0
    for char, count in [("P", 2), ("M", 1), ("D", 1), ("S", 1), ("I", 2)]:
        for r, c in wall_adjacent[placed : placed + count]:
            grid[r][c] = char
        placed += count

    chest_positions = []
    for r in range(3, height - 2, 2):
        for c in range(3, width - 3):
            if grid[r][c] == ".":
                chest_positions.append((r, c))
    rng.shuffle(chest_positions)
    for r, c in chest_positions[: max(2, len(chest_positions) * 2 // 5)]:
        grid[r][c] = "C"

    open_tiles = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1) if grid[r][c] == "."]
    rng.shuffle(open_tiles)
    for r, c in open_tiles[:num_agents]:
        grid[r][c] = "@"

    return AsciiMapBuilderConfig(map_data=grid, char_to_map_name=CHAR_TO_MAP_NAME)


_LEVEL_1_MAX_AGENTS = 4


def overcogged_map(num_agents: int, seed: int | None = None) -> AsciiMapBuilderConfig:
    """Return a classic layout appropriate for the given agent count."""

    if num_agents <= _LEVEL_1_MAX_AGENTS:
        return _level_1_map(num_agents)
    return _procedural_map(num_agents, seed=seed)
