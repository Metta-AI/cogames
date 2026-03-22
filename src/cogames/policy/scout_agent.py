"""Scout policy with systematic grid exploration and HP protection.

Architecture:
  Phase 1 (grid_explore): Serpentine sweep across the full map at even spacing.
    - Skips map corners (enemy ship territory) to avoid HP drain.
    - Navigates using BFS with optimistic fallback.
    - Transitions to Phase 2 when all grid points are visited/skipped.
  Phase 2 (frontier_explore): Standard frontier exploration for remaining unknowns.
    - Prefers frontiers away from known enemy ships.

The scout does NOT need to align junctions — its sole job is map coverage.
Complete map knowledge enables other agents (aligners) to use reliable BFS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

from cogames.policy.aligner_agent import AlignerPolicyImpl, AlignerState, Coord, _DIRECTION_DELTAS
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action
from mettagrid.simulator.interface import AgentObservation

logger = logging.getLogger("cogames.policy.scout_agent")

# Map dimensions for cogsguard_machina_1 (88x88)
_MAP_WIDTH = 88
_MAP_HEIGHT = 88

# Grid exploration spacing (cells between visited points).
# With obs_radius ≈ 5, spacing = 8 gives ~90% coverage with ~100 grid points.
_GRID_SPACING = 8

# Fraction of map edge to stay away from (enemy ships in corners).
# Corner zone = _CORNER_FRACTION * min(map_width, map_height) cells from each corner.
_CORNER_FRACTION = 0.20  # 20% → ~17 cells

# HP retreat threshold: retreat to hub when HP < this fraction of observed max HP.
_HP_RETREAT_THRESHOLD = 0.55

# Tags used for enemy ships (HP drain source).
_ENEMY_SHIP_TAG_NAMES = ["clips:ship", "ship"]


@dataclass
class ScoutState(AlignerState):
    """State for the scout explorer agent."""
    # Grid exploration phase
    phase: str = "init"                         # "init" | "grid_explore" | "frontier_explore"
    grid_targets: list[Coord] = field(default_factory=list)
    grid_index: int = 0

    # HP tracking for retreat logic
    max_hp_seen: int = 0
    retreating: bool = False

    # Enemy position tracking
    known_enemy_ships: set[Coord] = field(default_factory=set)


class ScoutExplorerPolicyImpl(AlignerPolicyImpl):
    """Systematic grid-then-frontier explorer.

    The scout sweeps the map in a serpentine grid pattern, building complete
    map knowledge, then continues frontier exploration for any remaining unknowns.
    Enemy corners are avoided to prevent HP drain.
    """

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        agent_id: int,
        grid_offset_fraction: float = 0.0,
        map_width: int = _MAP_WIDTH,
        map_height: int = _MAP_HEIGHT,
        grid_spacing: int = _GRID_SPACING,
    ) -> None:
        super().__init__(policy_env_info, agent_id)
        self._map_width = map_width
        self._map_height = map_height
        self._grid_spacing = grid_spacing
        self._grid_offset_fraction = grid_offset_fraction
        # Tags for HP-draining enemy structures
        self._enemy_ship_tags = self._starter._resolve_tag_ids(_ENEMY_SHIP_TAG_NAMES)
        # Also treat clips ships as hazard stations to avoid
        self._corner_zone = int(min(map_width, map_height) * _CORNER_FRACTION)

    # ─────────────────────────────────────────────────────────────────────────
    # State management
    # ─────────────────────────────────────────────────────────────────────────

    def initial_agent_state(self) -> ScoutState:
        base = super().initial_agent_state()
        return ScoutState(
            wander_direction_index=base.wander_direction_index,
            wander_steps_remaining=base.wander_steps_remaining,
            last_mode=base.last_mode,
        )

    def _copy_with_scout(self, state: ScoutState, base: AlignerState) -> ScoutState:
        return replace(
            state,
            wander_direction_index=base.wander_direction_index,
            wander_steps_remaining=base.wander_steps_remaining,
            last_mode=base.last_mode,
            known_free_cells=set(base.known_free_cells),
            blocked_cells=set(base.blocked_cells),
            known_hubs=set(base.known_hubs),
            known_aligner_stations=set(base.known_aligner_stations),
            known_neutral_junctions=set(base.known_neutral_junctions),
            known_friendly_junctions=set(base.known_friendly_junctions),
            known_enemy_junctions=set(base.known_enemy_junctions),
            known_hazard_stations=set(base.known_hazard_stations),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Grid construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_grid_targets(self) -> list[Coord]:
        """Build a serpentine grid of (row, col) targets across the map.

        Starts near the hub area (center) and spirals outward in a boustrophedon
        (snake) pattern, skipping dangerous corners.
        """
        spacing = self._grid_spacing
        margin = spacing
        targets: list[Coord] = []
        direction = 1
        row = margin
        while row <= self._map_height - margin:
            if direction == 1:
                col = margin
                while col <= self._map_width - margin:
                    if not self._is_corner_zone(row, col):
                        targets.append((row, col))
                    col += spacing
            else:
                col = self._map_width - margin
                while col >= margin:
                    if not self._is_corner_zone(row, col):
                        targets.append((row, col))
                    col -= spacing
            row += spacing
            direction = -direction

        # Allow different scouts to start from different parts of the grid
        if self._grid_offset_fraction > 0.0 and targets:
            offset = int(len(targets) * self._grid_offset_fraction) % len(targets)
            targets = targets[offset:] + targets[:offset]

        return targets

    def _is_corner_zone(self, row: int, col: int) -> bool:
        """Return True if (row, col) is in a dangerous corner zone."""
        z = self._corner_zone
        w, h = self._map_width, self._map_height
        return (
            (row < z and col < z)
            or (row < z and col > w - z)
            or (row > h - z and col < z)
            or (row > h - z and col > w - z)
        )

    # ─────────────────────────────────────────────────────────────────────────
    # HP and danger tracking
    # ─────────────────────────────────────────────────────────────────────────

    def _read_hp(self, obs: AgentObservation) -> int | None:
        """Read current HP from observation tokens."""
        center = self._starter._center
        for token in obs.tokens:
            if token.location != center:
                continue
            name = token.feature.name
            if name in ("hp", "energy", "hp:cogs", "hp:agent", "current_hp"):
                return int(token.value)
        return None

    def _update_enemy_ships(self, obs: AgentObservation, state: ScoutState, current_abs: Coord) -> None:
        """Track known enemy ship positions from observation."""
        for token in obs.tokens:
            if token.feature.name != "tag" or token.location is None:
                continue
            if int(token.value) in self._enemy_ship_tags:
                abs_cell = self._visible_abs_cell(current_abs, token.location)
                state.known_enemy_ships.add(abs_cell)

    def _is_dangerous(self, cell: Coord, state: ScoutState) -> bool:
        """Check whether a cell is in enemy territory."""
        # Known enemy ships: stay away from them
        for ship in state.known_enemy_ships:
            if abs(cell[0] - ship[0]) + abs(cell[1] - ship[1]) < 12:
                return True
        # Structural: avoid corners even before seeing ships
        return self._is_corner_zone(cell[0], cell[1])

    def _should_retreat(self, obs: AgentObservation, state: ScoutState) -> bool:
        """Return True if HP is low and we should retreat to hub."""
        hp = self._read_hp(obs)
        if hp is None:
            return False
        if hp > state.max_hp_seen:
            state.max_hp_seen = hp
        if state.max_hp_seen > 0 and hp < state.max_hp_seen * _HP_RETREAT_THRESHOLD:
            return True
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Navigation helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _navigate_to(self, state: ScoutState, current_abs: Coord, target: Coord) -> str | None:
        """Navigate toward target using BFS then optimistic BFS."""
        # Standard BFS (only known free cells)
        direction = self._bfs_first_direction(state, current_abs, target, avoid_hazards=False)
        if direction is not None:
            return direction
        # Optimistic BFS (unknown cells treated as free, blocked cells avoided)
        direction = self._bfs_optimistic_direction(state, current_abs, target, avoid_hazards=False)
        return direction

    def _safe_move_toward(self, state: ScoutState, current_abs: Coord, target: Coord) -> tuple[Action, ScoutState]:
        """Navigate toward target; avoid dangerous cells; fall back to safe wander."""
        direction = self._navigate_to(state, current_abs, target)
        if direction is not None:
            # Check that the next step isn't directly into danger
            for name, (dr, dc) in _DIRECTION_DELTAS:
                if name == direction:
                    next_cell = (current_abs[0] + dr, current_abs[1] + dc)
                    if self._is_dangerous(next_cell, state):
                        return self._safe_wander(state, current_abs)
                    break
            return self._starter._action(f"move_{direction}"), state
        # Greedy fallback toward target, avoiding danger
        dr = target[0] - current_abs[0]
        dc = target[1] - current_abs[1]
        if abs(dr) >= abs(dc):
            direction = "south" if dr > 0 else "north"
        else:
            direction = "east" if dc > 0 else "west"
        next_cell = (
            current_abs[0] + (1 if direction == "south" else -1 if direction == "north" else 0),
            current_abs[1] + (1 if direction == "east" else -1 if direction == "west" else 0),
        )
        if self._is_dangerous(next_cell, state):
            return self._safe_wander(state, current_abs)
        return self._starter._action(f"move_{direction}"), state

    # ─────────────────────────────────────────────────────────────────────────
    # Exploration phases
    # ─────────────────────────────────────────────────────────────────────────

    def _grid_explore_step(
        self, obs: AgentObservation, state: ScoutState, current_abs: Coord
    ) -> tuple[Action, ScoutState]:
        """One step of Phase 1: advance to next unvisited grid target."""
        # Advance past targets that are already covered or dangerous
        while state.grid_index < len(state.grid_targets):
            target = state.grid_targets[state.grid_index]
            dist = abs(target[0] - current_abs[0]) + abs(target[1] - current_abs[1])
            # Skip: already within observation radius (already explored)
            if dist <= self._obs_radius_row + 1:
                state.grid_index += 1
                continue
            # Skip: target in enemy corner zone or near known enemy ship
            if self._is_dangerous(target, state):
                state.grid_index += 1
                continue
            break

        if state.grid_index >= len(state.grid_targets):
            state.phase = "frontier_explore"
            logger.info(
                "agent=%s scout_grid_complete known_free=%d junctions=%d+%d+%d",
                obs.agent_id,
                len(state.known_free_cells),
                len(state.known_neutral_junctions),
                len(state.known_friendly_junctions),
                len(state.known_enemy_junctions),
            )
            return self._frontier_explore_step(obs, state, current_abs)

        target = state.grid_targets[state.grid_index]
        if state.last_mode != "scout_grid":
            logger.info(
                "agent=%s mode=scout_grid target=%s idx=%d/%d",
                obs.agent_id, target, state.grid_index, len(state.grid_targets),
            )
            state.last_mode = "scout_grid"

        action, new_state = self._safe_move_toward(state, current_abs, target)
        return action, new_state

    def _frontier_explore_step(
        self, obs: AgentObservation, state: ScoutState, current_abs: Coord
    ) -> tuple[Action, ScoutState]:
        """Phase 2: frontier exploration, avoiding danger zones."""
        if state.last_mode != "scout_frontier":
            logger.info("agent=%s mode=scout_frontier", obs.agent_id)
            state.last_mode = "scout_frontier"

        # Safe frontier: exclude known dangerous cells
        all_frontier = self._frontier_cells(state)
        safe_frontier = {c for c in all_frontier if not self._is_dangerous(c, state)}
        frontier = safe_frontier if safe_frontier else all_frontier

        if not frontier:
            return self._safe_wander(state, current_abs)

        target = self._nearest_known(current_abs, frontier)
        action, base_state = self._move_toward_target(state, current_abs, target)
        return action, self._copy_with_scout(state, base_state)

    # ─────────────────────────────────────────────────────────────────────────
    # Main step
    # ─────────────────────────────────────────────────────────────────────────

    def step_with_state(self, obs: AgentObservation, state: ScoutState) -> tuple[Action, ScoutState]:
        current_abs = self._update_map_memory(obs, state)
        self._update_enemy_ships(obs, state, current_abs)

        # ── Initialize grid on first step ──────────────────────────────────
        if state.phase == "init":
            state.grid_targets = self._build_grid_targets()
            state.phase = "grid_explore"
            logger.info(
                "agent=%s scout_init grid_points=%d spacing=%d map=%dx%d corner_zone=%d",
                obs.agent_id, len(state.grid_targets), self._grid_spacing,
                self._map_width, self._map_height, self._corner_zone,
            )

        # ── HP-based retreat ───────────────────────────────────────────────
        if self._should_retreat(obs, state):
            if not state.retreating:
                logger.info("agent=%s scout_retreat hp_low", obs.agent_id)
                state.retreating = True
            if state.known_hubs:
                hub = self._nearest_known(current_abs, state.known_hubs)
                direction = self._navigate_to_station(state, current_abs, hub, avoid_hazards=False)
                if direction:
                    return self._starter._action(f"move_{direction}"), state
            return self._safe_wander(state, current_abs)
        state.retreating = False

        # ── Grid exploration (Phase 1) ─────────────────────────────────────
        if state.phase == "grid_explore":
            return self._grid_explore_step(obs, state, current_abs)

        # ── Frontier exploration (Phase 2) ────────────────────────────────
        return self._frontier_explore_step(obs, state, current_abs)
