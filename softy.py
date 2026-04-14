"""
Softy — Smart scripted policy for Cogs vs Clips.

Improvements over starter:
- Shared team coordinator for discovered structures
- Position tracking from lp:* global observations
- Long-range navigation to off-screen known targets
- HP-aware retreat to prevent gear loss
- Target deconfliction between agents
- Miner resource specialization
- Stuck detection and recovery
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

from mettagrid.policy.policy import MultiAgentPolicy, StatefulAgentPolicy, StatefulPolicyImpl
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action
from mettagrid.simulator.interface import AgentObservation

# ── Constants ─────────────────────────────────────────────────────────────────

ELEMENTS = ("carbon", "oxygen", "germanium", "silicon")
MOVE_DELTAS = {"north": (-1, 0), "south": (1, 0), "west": (0, -1), "east": (0, 1)}
DIRECTIONS = ["north", "east", "south", "west"]
TEAM_TAG_PREFIX = "team:"

# Role distribution for 8 agents — max expansion.
# 3 miners sustain heart pipeline; 5 aligners rush junctions.
# No scrambler — pure alignment is 2x more heart-efficient than scramble+align.
ROLE_CYCLE = ("miner", "miner", "miner", "aligner", "aligner", "aligner", "aligner", "aligner")

# Miner element preference — diversify across 3 elements, silicon covered by fallback.
MINER_ELEMENT_PREF_IDX = {0: 0, 1: 1, 2: 2}  # carbon, oxygen, germanium; silicon via opportunistic
SECTOR_RADIUS = 30  # explore beyond hub alignment range to discover junctions for network expansion
HP_SAFETY_MARGIN = 15
ENERGY_MOVE_COST = 4
STUCK_THRESHOLD = 3  # same position this many times in recent history triggers rotation
MINER_DEPOSIT_THRESHOLD = 30  # mine more before depositing to reduce travel overhead
ALIGN_HUB_RADIUS = 25  # junctions within this distance of hub can be aligned
ALIGN_NET_RADIUS = 15  # junctions within this distance of team network can be aligned


# ── Shared Coordinator ────────────────────────────────────────────────────────

@dataclass
class SoftyCoordinator:
    """Shared state across all Softy agents in a team."""

    hub_pos: tuple[int, int] | None = None
    stations: dict[str, tuple[int, int]] = field(default_factory=dict)
    junctions: dict[tuple[int, int], str] = field(default_factory=dict)
    extractors: dict[tuple[int, int], str] = field(default_factory=dict)
    agent_targets: dict[int, tuple[int, int] | None] = field(default_factory=dict)
    agent_positions: dict[int, tuple[int, int]] = field(default_factory=dict)

    def claim_target(self, agent_id: int, target: tuple[int, int] | None) -> None:
        self.agent_targets[agent_id] = target

    def is_claimed(self, target: tuple[int, int], exclude: int) -> bool:
        return any(t == target for aid, t in self.agent_targets.items() if aid != exclude)

    def nearest_junction(self, pos: tuple[int, int], alignment: str, agent_id: int) -> tuple[int, int] | None:
        best, best_dist = None, float("inf")
        for jpos, jalign in self.junctions.items():
            if jalign != alignment:
                continue
            if self.is_claimed(jpos, agent_id):
                continue
            dist = abs(jpos[0] - pos[0]) + abs(jpos[1] - pos[1])
            if dist < best_dist:
                best_dist = dist
                best = jpos
        return best

    def nearest_alignable_junction(
        self, pos: tuple[int, int], agent_id: int, blacklist: set[tuple[int, int]] | None = None,
    ) -> tuple[int, int] | None:
        """Find best neutral junction: balance distance with frontier expansion value."""
        best, best_score = None, float("-inf")
        for jpos, jalign in self.junctions.items():
            if jalign != "neutral":
                continue
            if self.is_claimed(jpos, agent_id):
                continue
            if blacklist and jpos in blacklist:
                continue
            if not self._in_align_range(jpos):
                continue
            dist = abs(jpos[0] - pos[0]) + abs(jpos[1] - pos[1])
            # Frontier value: count known non-cogs junctions that would become
            # newly alignable if we capture this one (within 15 cells)
            frontier = 0
            for opos, oalign in self.junctions.items():
                if opos == jpos or oalign == "cogs":
                    continue
                if abs(opos[0] - jpos[0]) + abs(opos[1] - jpos[1]) <= ALIGN_NET_RADIUS:
                    if not self._in_align_range(opos):
                        frontier += 1
            # Score: frontier bonus vs distance penalty
            score = frontier * 8.0 - dist
            if score > best_score:
                best_score = score
                best = jpos
        return best

    def nearest_enemy_alignable_junction(
        self, pos: tuple[int, int], agent_id: int, blacklist: set[tuple[int, int]] | None = None,
    ) -> tuple[int, int] | None:
        """Find nearest enemy (clips) junction within alignment range."""
        best, best_dist = None, float("inf")
        for jpos, jalign in self.junctions.items():
            if jalign != "clips":
                continue
            if self.is_claimed(jpos, agent_id):
                continue
            if blacklist and jpos in blacklist:
                continue
            if not self._in_align_range(jpos):
                continue
            dist = abs(jpos[0] - pos[0]) + abs(jpos[1] - pos[1])
            if dist < best_dist:
                best_dist = dist
                best = jpos
        return best

    def _in_align_range(self, jpos: tuple[int, int]) -> bool:
        """Check if a junction position is within alignment range of hub or cogs network."""
        # Within 25 cells of hub
        if self.hub_pos:
            if abs(jpos[0] - self.hub_pos[0]) + abs(jpos[1] - self.hub_pos[1]) <= ALIGN_HUB_RADIUS:
                return True
        # Within 15 cells of any cogs-aligned junction
        for cpos, calign in self.junctions.items():
            if calign == "cogs":
                if abs(jpos[0] - cpos[0]) + abs(jpos[1] - cpos[1]) <= ALIGN_NET_RADIUS:
                    return True
        return False

    def nearest_extractor(self, pos: tuple[int, int], element: str | None, agent_id: int) -> tuple[int, int] | None:
        best, best_dist = None, float("inf")
        for epos, etype in self.extractors.items():
            if element is not None and etype != element:
                continue
            if self.is_claimed(epos, agent_id):
                continue
            dist = abs(epos[0] - pos[0]) + abs(epos[1] - pos[1])
            if dist < best_dist:
                best_dist = dist
                best = epos
        return best

    def nearest_healing(self, pos: tuple[int, int]) -> int:
        """Manhattan distance to nearest cogs-aligned building."""
        dist = 999
        if self.hub_pos:
            dist = abs(pos[0] - self.hub_pos[0]) + abs(pos[1] - self.hub_pos[1])
        for jpos, jalign in self.junctions.items():
            if jalign == "cogs":
                dist = min(dist, abs(pos[0] - jpos[0]) + abs(pos[1] - jpos[1]))
        return dist


# ── Per-Agent State ───────────────────────────────────────────────────────────

@dataclass
class SoftyState:
    row: int = 0
    col: int = 0
    has_position: bool = False
    hub_offset_r: int = 0  # agent's lp row when standing at hub
    hub_offset_c: int = 0  # agent's lp col when standing at hub
    has_hub_offset: bool = False
    explore_dir_idx: int = 0
    recent_positions: list[tuple[int, int]] = field(default_factory=list)
    stuck_count: int = 0
    step_count: int = 0
    failed_junctions: set[tuple[int, int]] = field(default_factory=set)  # shared coords of junctions we couldn't align
    last_target_pos: tuple[int, int] | None = None  # shared coords of current target junction
    target_ticks: int = 0  # how many ticks we've been going toward last_target_pos
    had_heart_last_step: bool = False  # track if we had a heart last step (for alignment failure detection)
    heartless_ticks: int = 0  # ticks spent without a heart (aligners explore after threshold)
    hub_resources: dict[str, int] = field(default_factory=dict)  # team hub inventory (carbon, oxygen, etc.)
    last_move_succeeded: bool = True  # from last_action_move token


# ── Agent Implementation ─────────────────────────────────────────────────────

class SoftyAgentImpl(StatefulPolicyImpl[SoftyState]):

    def __init__(
        self,
        env: PolicyEnvInterface,
        agent_id: int,
        role: str,
        coordinator: SoftyCoordinator,
    ):
        self._env = env
        self._id = agent_id
        self._role = role
        self._coord = coordinator

        names = env.action_names
        self._action_set = set(names)
        self._noop = "noop" if "noop" in self._action_set else names[0]
        self._center = (env.obs_height // 2, env.obs_width // 2)

        # Tag resolution
        self._tag_id = {name: idx for idx, name in enumerate(env.tags)}
        self._team_tags = {idx for idx, name in enumerate(env.tags) if name.startswith(TEAM_TAG_PREFIX)}

        self._agent_tags = self._tags(["agent"])
        self._junction_tags = self._tags(["junction"])
        self._hub_tags = self._tags(["hub"])
        self._extractor_tags_by_elem = {e: self._tags([f"{e}_extractor"]) for e in ELEMENTS}
        self._all_extractor_tags: set[int] = set()
        for s in self._extractor_tags_by_elem.values():
            self._all_extractor_tags |= s
        self._station_tags = {r: self._tags([r, f"c:{r}"]) for r in ("miner", "aligner", "scrambler", "scout")}
        self._deposit_tags = self._tags(["hub", "junction"])
        self._heart_tags = self._tags(["hub", "chest"])

        self._preferred_element = ELEMENTS[MINER_ELEMENT_PREF_IDX.get(agent_id, agent_id % len(ELEMENTS))]
        self._explore_offset = agent_id % len(DIRECTIONS)

    def _tags(self, names: Iterable[str]) -> set[int]:
        ids: set[int] = set()
        for n in names:
            if n in self._tag_id:
                ids.add(self._tag_id[n])
            tn = f"type:{n}"
            if tn in self._tag_id:
                ids.add(self._tag_id[tn])
        return ids

    def _act(self, name: str) -> Action:
        return Action(name=name if name in self._action_set else self._noop)

    # ── Coordinate Conversion ────────────────────────────────────────────────
    # Coordinator stores hub-relative coords (hub = 0,0).
    # Each agent converts at the boundary: lp ↔ hub-relative.

    @staticmethod
    def _to_shared(lp_r: int, lp_c: int, s: SoftyState) -> tuple[int, int]:
        """Convert agent-local lp position to hub-relative shared coords."""
        return (lp_r - s.hub_offset_r, lp_c - s.hub_offset_c)

    @staticmethod
    def _to_local(shared_r: int, shared_c: int, s: SoftyState) -> tuple[int, int]:
        """Convert hub-relative shared coords to agent-local lp position."""
        return (shared_r + s.hub_offset_r, shared_c + s.hub_offset_c)

    # ── Observation Parsing ───────────────────────────────────────────────────

    def _parse(self, obs: AgentObservation, s: SoftyState):
        tags: dict[tuple[int, int], set[int]] = {}
        inv: dict[str, int] = {}
        globs: dict[str, int] = {}
        s.hub_resources = {}  # reset each step (multi-part tokens accumulate)


        for tok in obs.tokens:
            name = tok.feature.name

            if name.startswith("lp:") or name in (
                "last_action", "last_action_move", "episode_completion_pct", "last_reward", "agent_id",
            ):
                globs[name] = int(tok.value)
                continue

            # Hub team inventory (team:carbon, team:oxygen, etc.) — multi-part encoded
            if name.startswith("team:"):
                suffix = name[5:]
                if not suffix:
                    continue
                item, sep, pstr = suffix.rpartition(":p")
                if sep and item and pstr.isdigit():
                    scale = max(int(tok.feature.normalization), 1) ** int(pstr)
                else:
                    item = suffix
                    scale = 1
                val = int(tok.value)
                if val > 0:
                    s.hub_resources[item] = s.hub_resources.get(item, 0) + val * scale
                continue

            if name.startswith("inv:") and tok.location == self._center:
                suffix = name[4:]
                if not suffix:
                    continue
                item, sep, pstr = suffix.rpartition(":p")
                if sep and item and pstr.isdigit():
                    scale = max(int(tok.feature.normalization), 1) ** int(pstr)
                else:
                    item = suffix
                    scale = 1
                val = int(tok.value)
                if val > 0:
                    inv[item] = inv.get(item, 0) + val * scale
                continue

            if name == "tag" and tok.location is not None:
                tags.setdefault(tok.location, set()).add(int(tok.value))

        # Position from lp:* offsets
        lp_n = globs.get("lp:north", 0)
        lp_s = globs.get("lp:south", 0)
        lp_e = globs.get("lp:east", 0)
        lp_w = globs.get("lp:west", 0)
        if lp_n or lp_s or lp_e or lp_w or s.step_count > 0:
            s.row = lp_s - lp_n
            s.col = lp_e - lp_w
            s.has_position = True

        s.last_move_succeeded = bool(globs.get("last_action_move", 1))

        self._discover(tags, s)
        return tags, inv

    def _discover(self, tags: dict[tuple[int, int], set[int]], s: SoftyState) -> None:
        if not s.has_position:
            return
        cr, cc = self._center
        own_team = tags.get(self._center, set()) & self._team_tags

        # First pass: calibrate hub offset if hub is visible and not yet calibrated
        if not s.has_hub_offset:
            for loc, lt in tags.items():
                if lt & self._hub_tags and (lt & own_team):
                    # Hub's lp position in this agent's frame
                    hub_lp_r = s.row + (loc[0] - cr)
                    hub_lp_c = s.col + (loc[1] - cc)
                    s.hub_offset_r = hub_lp_r
                    s.hub_offset_c = hub_lp_c
                    s.has_hub_offset = True
                    break

        # Only write shared data once calibrated
        if not s.has_hub_offset:
            return

        for loc, lt in tags.items():
            # Compute hub-relative (shared) position
            lp_r = s.row + (loc[0] - cr)
            lp_c = s.col + (loc[1] - cc)
            sp = self._to_shared(lp_r, lp_c, s)

            if lt & self._hub_tags and (lt & own_team):
                self._coord.hub_pos = sp  # Should be (0, 0)

            if lt & self._junction_tags:
                if lt & own_team:
                    self._coord.junctions[sp] = "cogs"
                elif lt & (self._team_tags - own_team):
                    self._coord.junctions[sp] = "clips"
                else:
                    self._coord.junctions[sp] = "neutral"

            for elem, etags in self._extractor_tags_by_elem.items():
                if lt & etags:
                    self._coord.extractors[sp] = elem

            for role_name, stags in self._station_tags.items():
                if lt & stags and (lt & own_team):
                    self._coord.stations[role_name] = sp

        # Agent position in shared coords for deconfliction
        self._coord.agent_positions[self._id] = self._to_shared(s.row, s.col, s)

    # ── Navigation ────────────────────────────────────────────────────────────

    def _closest(
        self, tags: dict[tuple[int, int], set[int]],
        include: set[int],
        require: set[int] | None = None,
        exclude: set[int] | None = None,
    ) -> tuple[int, int] | None:
        cr, cc = self._center
        best, best_d = None, 999
        for loc, lt in tags.items():
            if not (lt & include):
                continue
            if require and not (lt & require):
                continue
            if exclude and (lt & exclude):
                continue
            d = abs(loc[0] - cr) + abs(loc[1] - cc)
            if d < best_d:
                best_d = d
                best = loc
        return best

    def _go_visible(self, target: tuple[int, int], tags: dict[tuple[int, int], set[int]]) -> Action | None:
        cr, cc = self._center
        dr, dc = target[0] - cr, target[1] - cc

        if dr == 0 and dc == 0:
            # Standing ON the target — move away so we can bump it again next step
            blocked = set(tags)
            blocked.discard(self._center)
            h, w = self._env.obs_height, self._env.obs_width
            for d in DIRECTIONS:
                delta = MOVE_DELTAS[d]
                nxt = (cr + delta[0], cc + delta[1])
                if nxt not in blocked and 0 <= nxt[0] < h and 0 <= nxt[1] < w:
                    return self._act(f"move_{d}")
            return self._act(self._noop)

        # Adjacent: step onto it (interaction)
        if abs(dr) + abs(dc) == 1:
            if dr == -1:
                return self._act("move_north")
            if dr == 1:
                return self._act("move_south")
            if dc == -1:
                return self._act("move_west")
            return self._act("move_east")

        # BFS to reach cell adjacent to target
        blocked = set(tags)
        blocked.discard(self._center)
        if not (tags.get(target, set()) & self._agent_tags):
            blocked.discard(target)

        h, w = self._env.obs_height, self._env.obs_width
        goals = set()
        for delta in MOVE_DELTAS.values():
            adj = (target[0] + delta[0], target[1] + delta[1])
            if 0 <= adj[0] < h and 0 <= adj[1] < w and adj not in blocked:
                goals.add(adj)
        if not goals:
            return None

        queue = deque([self._center])
        visited = {self._center}
        first_dir: dict[tuple[int, int], str] = {}

        while queue:
            cur = queue.popleft()
            dt_r, dt_c = target[0] - cur[0], target[1] - cur[1]
            # Prioritize directions toward target, but try ALL 4 to navigate around walls
            cands: list[str] = []
            if abs(dt_r) >= abs(dt_c):
                if dt_r > 0:
                    cands.append("south")
                elif dt_r < 0:
                    cands.append("north")
                if dt_c > 0:
                    cands.append("east")
                elif dt_c < 0:
                    cands.append("west")
            else:
                if dt_c > 0:
                    cands.append("east")
                elif dt_c < 0:
                    cands.append("west")
                if dt_r > 0:
                    cands.append("south")
                elif dt_r < 0:
                    cands.append("north")
            for d in DIRECTIONS:
                if d not in cands:
                    cands.append(d)

            for d in cands:
                delta = MOVE_DELTAS[d]
                nxt = (cur[0] + delta[0], cur[1] + delta[1])
                if nxt in visited or nxt in blocked or not (0 <= nxt[0] < h) or not (0 <= nxt[1] < w):
                    continue
                visited.add(nxt)
                first_dir[nxt] = first_dir.get(cur, d)
                if nxt in goals:
                    return self._act(f"move_{first_dir[nxt]}")
                queue.append(nxt)

        return None

    def _go_absolute(self, target: tuple[int, int], s: SoftyState, tags: dict) -> Action:
        """Navigate toward an off-screen target using BFS within visible window."""
        dr, dc = target[0] - s.row, target[1] - s.col
        h, w = self._env.obs_height, self._env.obs_width
        blocked = set(tags)
        blocked.discard(self._center)

        # Use BFS toward the edge of the observation window closest to the target.
        # Pick the edge cell that minimizes remaining distance to target.
        best_edge, best_dist, best_dir_str = None, float("inf"), None

        # BFS within obs window
        queue = deque([self._center])
        visited = {self._center}
        first_dir: dict[tuple[int, int], str] = {}
        edge_cells: list[tuple[tuple[int, int], str]] = []

        while queue:
            cur = queue.popleft()
            # Prioritize directions toward the absolute target
            cands: list[str] = []
            if abs(dr) >= abs(dc):
                if dr > 0:
                    cands.append("south")
                elif dr < 0:
                    cands.append("north")
                if dc > 0:
                    cands.append("east")
                elif dc < 0:
                    cands.append("west")
            else:
                if dc > 0:
                    cands.append("east")
                elif dc < 0:
                    cands.append("west")
                if dr > 0:
                    cands.append("south")
                elif dr < 0:
                    cands.append("north")
            for d in DIRECTIONS:
                if d not in cands:
                    cands.append(d)

            for d in cands:
                delta = MOVE_DELTAS[d]
                nxt = (cur[0] + delta[0], cur[1] + delta[1])
                if nxt in visited or nxt in blocked or not (0 <= nxt[0] < h) or not (0 <= nxt[1] < w):
                    continue
                visited.add(nxt)
                fd = first_dir.get(cur, d)
                first_dir[nxt] = fd
                # Check if this cell is at the edge of the obs window toward our target
                abs_r = s.row + (nxt[0] - self._center[0])
                abs_c = s.col + (nxt[1] - self._center[1])
                remaining = abs(target[0] - abs_r) + abs(target[1] - abs_c)
                if remaining < best_dist:
                    best_dist = remaining
                    best_edge = nxt
                    best_dir_str = fd
                queue.append(nxt)

        if best_dir_str is not None:
            return self._act(f"move_{best_dir_str}")

        return self._wander(tags, s)

    def _wander(self, tags: dict[tuple[int, int], set[int]], s: SoftyState) -> Action:
        """Explore outward from hub toward agent's assigned sector of the map."""
        blocked = set(tags)
        blocked.discard(self._center)
        h, w = self._env.obs_height, self._env.obs_width

        # If we know our position and hub, move away from hub toward our sector
        if s.has_position and s.has_hub_offset and self._coord.hub_pos is not None:
            sector_angles = [
                (-1, 0), (-1, 1), (0, 1), (1, 1),
                (1, 0), (1, -1), (0, -1), (-1, -1),
            ]
            # Use explore_dir_idx (shifted by stuck handler) mixed with agent id for initial diversity
            sector_idx = (self._id + s.explore_dir_idx) % len(sector_angles)
            dr, dc = sector_angles[sector_idx]
            # hub_pos is (0,0) in shared coords; sector target within alignment range
            sector_shared = (dr * SECTOR_RADIUS, dc * SECTOR_RADIUS)
            # Convert to local lp for direction calculation
            sector_local = self._to_local(sector_shared[0], sector_shared[1], s)

            # Prioritize directions that move toward sector target
            dirs = self._dirs_toward(sector_local[0] - s.row, sector_local[1] - s.col)
            for d in dirs:
                delta = MOVE_DELTAS[d]
                nxt = (self._center[0] + delta[0], self._center[1] + delta[1])
                if nxt not in blocked and 0 <= nxt[0] < h and 0 <= nxt[1] < w:
                    return self._act(f"move_{d}")

        # Fallback: use rotational exploration
        for i in range(len(DIRECTIONS)):
            d = DIRECTIONS[(s.explore_dir_idx + i) % len(DIRECTIONS)]
            delta = MOVE_DELTAS[d]
            nxt = (self._center[0] + delta[0], self._center[1] + delta[1])
            if nxt not in blocked and 0 <= nxt[0] < h and 0 <= nxt[1] < w:
                return self._act(f"move_{d}")

        return self._act(self._noop)

    @staticmethod
    def _dirs_toward(dr: int, dc: int) -> list[str]:
        """Return directions prioritized toward (dr, dc) offset, with all 4 directions."""
        dirs: list[str] = []
        if abs(dr) >= abs(dc):
            if dr > 0:
                dirs.append("south")
            elif dr < 0:
                dirs.append("north")
            if dc > 0:
                dirs.append("east")
            elif dc < 0:
                dirs.append("west")
        else:
            if dc > 0:
                dirs.append("east")
            elif dc < 0:
                dirs.append("west")
            if dr > 0:
                dirs.append("south")
            elif dr < 0:
                dirs.append("north")
        for d in DIRECTIONS:
            if d not in dirs:
                dirs.append(d)
        return dirs

    def _go_to_known(self, shared_target: tuple[int, int], tags: dict, s: SoftyState) -> Action:
        """Navigate to a known shared (hub-relative) position."""
        if s.has_position and s.has_hub_offset:
            local = self._to_local(shared_target[0], shared_target[1], s)
            cr, cc = self._center
            obs_row = local[0] - s.row + cr
            obs_col = local[1] - s.col + cc
            if 0 <= obs_row < self._env.obs_height and 0 <= obs_col < self._env.obs_width:
                action = self._go_visible((obs_row, obs_col), tags)
                if action is not None:
                    return action
            return self._go_absolute(local, s, tags)
        return self._wander(tags, s)

    def _update_stuck(self, s: SoftyState) -> None:
        pos = (s.row, s.col)
        s.recent_positions.append(pos)
        if len(s.recent_positions) > 8:
            s.recent_positions.pop(0)
        # Failed move (wall or blocked) is an immediate stuck signal
        if not s.last_move_succeeded:
            s.stuck_count += 2
            s.explore_dir_idx = (s.explore_dir_idx + 1) % len(DIRECTIONS)
        elif len(s.recent_positions) >= 4 and s.recent_positions.count(pos) >= STUCK_THRESHOLD:
            s.stuck_count += 1
            s.explore_dir_idx = (s.explore_dir_idx + 1) % len(DIRECTIONS)
        else:
            s.stuck_count = max(0, s.stuck_count - 1)

    # ── HP / Energy ───────────────────────────────────────────────────────────

    def _should_retreat(self, inv: dict, s: SoftyState) -> bool:
        hp = inv.get("hp", 0)
        if hp == 0:
            return False  # hp not observable or already dead
        if s.has_position and s.has_hub_offset:
            shared_pos = self._to_shared(s.row, s.col, s)
            heal_dist = self._coord.nearest_healing(shared_pos)
        else:
            heal_dist = 20
        return hp < heal_dist + HP_SAFETY_MARGIN

    def _retreat(self, tags: dict, s: SoftyState) -> Action:
        own_team = tags.get(self._center, set()) & self._team_tags
        vis = self._closest(tags, self._deposit_tags, require=own_team)
        if vis is not None:
            action = self._go_visible(vis, tags)
            if action is not None:
                return action
        # Find nearest cogs building in shared coords
        if s.has_hub_offset:
            shared_pos = self._to_shared(s.row, s.col, s)
            best_target = self._coord.hub_pos
            best_dist = float("inf")
            if best_target:
                best_dist = abs(shared_pos[0] - best_target[0]) + abs(shared_pos[1] - best_target[1])
            for jpos, jalign in self._coord.junctions.items():
                if jalign == "cogs":
                    jd = abs(shared_pos[0] - jpos[0]) + abs(shared_pos[1] - jpos[1])
                    if jd < best_dist:
                        best_dist = jd
                        best_target = jpos
            if best_target and s.has_position:
                return self._go_to_known(best_target, tags, s)
        return self._wander(tags, s)

    def _low_energy(self, inv: dict) -> bool:
        energy = inv.get("energy", 999)
        # Only gate on energy if we actually observe it AND it's critically low
        return energy < 2  # Need at least 2 for any action attempt

    # ── Role: Miner ───────────────────────────────────────────────────────────

    def _miner(self, tags: dict, inv: dict, s: SoftyState) -> Action:
        own = tags.get(self._center, set()) & self._team_tags
        has_gear = inv.get("miner", 0) > 0
        cargo = sum(inv.get(e, 0) for e in ELEMENTS)

        # Phase 1: get gear
        if not has_gear:
            return self._go_to_role_station("miner", tags, own, s)

        # Phase 2: deposit when cargo is above threshold
        # On-screen: deposit at nearest visible friendly building (hub or junction)
        # Off-screen: always navigate to hub (junctions might get scrambled during transit)
        if cargo >= MINER_DEPOSIT_THRESHOLD:
            vis = self._closest(tags, self._deposit_tags, require=own)
            if vis is not None:
                action = self._go_visible(vis, tags)
                if action is not None:
                    return action
            if self._coord.hub_pos is not None:
                return self._go_to_known(self._coord.hub_pos, tags, s)
            return self._wander(tags, s)

        # Phase 3: mine — prefer assigned element visible, fallback to any visible
        pref_tags = self._extractor_tags_by_elem.get(self._preferred_element, self._all_extractor_tags)
        vis = self._closest(tags, pref_tags)
        if vis is None:
            vis = self._closest(tags, self._all_extractor_tags)
        if vis is not None:
            action = self._go_visible(vis, tags)
            if action is not None:
                return action

        if s.has_position and s.has_hub_offset:
            shared_pos = self._to_shared(s.row, s.col, s)
            target = self._coord.nearest_extractor(shared_pos, self._preferred_element, self._id)
            if target is None:
                target = self._coord.nearest_extractor(shared_pos, None, self._id)
            if target:
                self._coord.claim_target(self._id, target)
                return self._go_to_known(target, tags, s)

        return self._wander(tags, s)

    # ── Role: Aligner ─────────────────────────────────────────────────────────

    def _aligner(self, tags: dict, inv: dict, s: SoftyState) -> Action:
        own = tags.get(self._center, set()) & self._team_tags
        has_gear = inv.get("aligner", 0) > 0
        has_heart = inv.get("heart", 0) > 0

        if not has_gear:
            s.had_heart_last_step = False
            return self._go_to_role_station("aligner", tags, own, s)

        if not has_heart:
            if s.had_heart_last_step and s.last_target_pos is not None:
                s.last_target_pos = None
                s.target_ticks = 0
            s.had_heart_last_step = False
            s.heartless_ticks += 1
            # Check if hub can craft a heart (needs >= 7 of each element)
            hub_can_craft = s.hub_resources and all(
                s.hub_resources.get(e, 0) >= 7 for e in ELEMENTS
            )
            if hub_can_craft:
                return self._go_to_heart_source(tags, own, s)
            # Hub can't craft — explore to discover junctions, check back periodically
            if (s.heartless_ticks % 30) < 5:
                return self._go_to_heart_source(tags, own, s)
            # Cycle through sectors to maximize junction discovery
            if s.heartless_ticks % 8 == 0:
                s.explore_dir_idx = (s.explore_dir_idx + 1) % 8
            return self._wander(tags, s)

        s.had_heart_last_step = True
        s.heartless_ticks = 0

        # Detect stuck on unalignable junction: if we've been targeting the same
        # junction for 15+ ticks with heart, it's probably out of range (walls).
        if s.last_target_pos is not None:
            s.target_ticks += 1
            if s.target_ticks > 15:
                s.failed_junctions.add(s.last_target_pos)
                s.last_target_pos = None
                s.target_ticks = 0

        # Find any alignable junction visible in observation (not our team, not blacklisted)
        vis = self._closest(tags, self._junction_tags, exclude=own)
        if vis is not None and s.has_hub_offset:
            junc_shared = self._to_shared(
                s.row + (vis[0] - self._center[0]),
                s.col + (vis[1] - self._center[1]),
                s,
            )
            if junc_shared not in s.failed_junctions:
                if junc_shared != s.last_target_pos:
                    s.last_target_pos = junc_shared
                    s.target_ticks = 0
                action = self._go_visible(vis, tags)
                if action is not None:
                    return action

        # Use coordinator to find nearest ALIGNABLE junction (within range, not blacklisted)
        if s.has_position and s.has_hub_offset:
            shared_pos = self._to_shared(s.row, s.col, s)
            target = self._coord.nearest_alignable_junction(shared_pos, self._id, s.failed_junctions)
            if target is None:
                target = self._coord.nearest_enemy_alignable_junction(shared_pos, self._id, s.failed_junctions)
            if target:
                self._coord.claim_target(self._id, target)
                if target != s.last_target_pos:
                    s.last_target_pos = target
                    s.target_ticks = 0
                return self._go_to_known(target, tags, s)

        return self._wander(tags, s)

    # ── Role: Scrambler ───────────────────────────────────────────────────────

    def _scrambler(self, tags: dict, inv: dict, s: SoftyState) -> Action:
        own = tags.get(self._center, set()) & self._team_tags
        enemy = self._team_tags - own
        has_gear = inv.get("scrambler", 0) > 0
        has_heart = inv.get("heart", 0) > 0

        if not has_gear:
            return self._go_to_role_station("scrambler", tags, own, s)

        if not has_heart:
            return self._go_to_heart_source(tags, own, s)

        # Find enemy (clips) junction
        vis = self._closest(tags, self._junction_tags, require=enemy)
        if vis is not None:
            action = self._go_visible(vis, tags)
            if action is not None:
                return action

        if s.has_position and s.has_hub_offset:
            shared_pos = self._to_shared(s.row, s.col, s)
            target = self._coord.nearest_junction(shared_pos, "clips", self._id)
            if target:
                self._coord.claim_target(self._id, target)
                return self._go_to_known(target, tags, s)

        return self._wander(tags, s)

    # ── Role: Scout ───────────────────────────────────────────────────────────

    def _scout(self, tags: dict, inv: dict, s: SoftyState) -> Action:
        own = tags.get(self._center, set()) & self._team_tags
        has_gear = inv.get("scout", 0) > 0

        if not has_gear:
            return self._go_to_role_station("scout", tags, own, s)

        # Phase 1 (first 2000 ticks): aggressive frontier exploration — cycle through all sectors
        if s.step_count < 2000:
            # Advance sector every 25 ticks to sweep the whole map
            if s.step_count % 25 == 0:
                s.explore_dir_idx = (s.explore_dir_idx + 1) % 8
            return self._wander(tags, s)

        # Phase 2: patrol known junctions to keep alignment status current
        if s.has_position and s.has_hub_offset:
            for jpos in self._coord.junctions:
                if not self._coord.is_claimed(jpos, self._id):
                    self._coord.claim_target(self._id, jpos)
                    return self._go_to_known(jpos, tags, s)

        return self._wander(tags, s)

    # ── Shared Helpers ────────────────────────────────────────────────────────

    def _go_to_role_station(self, role: str, tags: dict, own: set[int], s: SoftyState) -> Action:
        vis = self._closest(tags, self._station_tags[role], require=own)
        if vis is not None:
            action = self._go_visible(vis, tags)
            if action is not None:
                return action
        if role in self._coord.stations and s.has_position and s.has_hub_offset:
            return self._go_to_known(self._coord.stations[role], tags, s)
        if self._coord.hub_pos is not None and s.has_position and s.has_hub_offset:
            return self._go_to_known(self._coord.hub_pos, tags, s)
        return self._wander(tags, s)

    def _go_to_heart_source(self, tags: dict, own: set[int], s: SoftyState) -> Action:
        vis = self._closest(tags, self._heart_tags, require=own)
        if vis is not None:
            action = self._go_visible(vis, tags)
            if action is not None:
                return action
        if self._coord.hub_pos is not None and s.has_position and s.has_hub_offset:
            return self._go_to_known(self._coord.hub_pos, tags, s)
        return self._wander(tags, s)

    # ── Main Step ─────────────────────────────────────────────────────────────

    def step_with_state(self, obs: AgentObservation, s: SoftyState) -> tuple[Action, SoftyState]:
        tags, inv = self._parse(obs, s)

        self._update_stuck(s)
        s.step_count += 1

        # Wait for energy regen if too low to move
        if self._low_energy(inv):
            return self._act(self._noop), s

        # Retreat if HP is critical and we have gear to lose
        if self._should_retreat(inv, s):
            return self._retreat(tags, s), s

        # Severely stuck: head to hub to reset
        if s.stuck_count > 5 and self._coord.hub_pos is not None and s.has_position and s.has_hub_offset:
            # Aligner stuck near a junction it can't align → blacklist it
            if self._role == "aligner" and s.last_target_pos is not None:
                s.failed_junctions.add(s.last_target_pos)
                self._coord.claim_target(self._id, None)
                s.last_target_pos = None
                s.target_ticks = 0
            s.stuck_count = 0
            s.explore_dir_idx = (s.explore_dir_idx + 3) % 8  # shift sector by 135° to explore a different area
            return self._go_to_known(self._coord.hub_pos, tags, s), s

        # Role dispatch
        if self._role == "miner":
            return self._miner(tags, inv, s), s
        if self._role == "aligner":
            return self._aligner(tags, inv, s), s
        if self._role == "scrambler":
            return self._scrambler(tags, inv, s), s
        if self._role == "scout":
            return self._scout(tags, inv, s), s

        return self._wander(tags, s), s

    def initial_agent_state(self) -> SoftyState:
        return SoftyState(explore_dir_idx=self._explore_offset)


# ── Top-Level Policy ──────────────────────────────────────────────────────────

class SoftyPolicy(MultiAgentPolicy):
    short_names = ["softy"]

    def __init__(self, policy_env_info: PolicyEnvInterface, device: str = "cpu"):
        super().__init__(policy_env_info, device=device)
        self._coordinator = SoftyCoordinator()
        self._agents: dict[int, StatefulAgentPolicy[SoftyState]] = {}

    def agent_policy(self, agent_id: int) -> StatefulAgentPolicy[SoftyState]:
        if agent_id not in self._agents:
            role = ROLE_CYCLE[agent_id % len(ROLE_CYCLE)]
            impl = SoftyAgentImpl(self._policy_env_info, agent_id, role, self._coordinator)
            self._agents[agent_id] = StatefulAgentPolicy(impl, self._policy_env_info, agent_id=agent_id)
        return self._agents[agent_id]
