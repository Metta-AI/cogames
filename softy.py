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

from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy, StatefulPolicyImpl
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action
from mettagrid.simulator.interface import AgentObservation

# ── Constants ─────────────────────────────────────────────────────────────────

ELEMENTS = ("carbon", "oxygen", "germanium", "silicon")
MOVE_DELTAS = {"north": (-1, 0), "south": (1, 0), "west": (0, -1), "east": (0, 1)}
DIRECTIONS = ["north", "east", "south", "west"]
TEAM_TAG_PREFIX = "team:"

# P200: Role distributions by team size — includes scramblers for territory recapture.
# Competitive analysis: top policies (dinky/slinky) all use scramblers (30-177 per episode).
# Without scrambling, once clips takes a junction it's permanently lost.
# c1/c2: pure aligners — 96% of tournament is c1, teammates handle mining/scrambling.
# c3+: scramblers come from aligner slots, miners never switch.
ROLE_DISTRIBUTIONS = {
    1: ("aligner",),
    2: ("aligner", "aligner"),
    3: ("aligner", "aligner", "scrambler"),
    4: ("miner", "aligner", "aligner", "scrambler"),
    5: ("miner", "miner", "aligner", "aligner", "scrambler"),
    6: ("miner", "miner", "aligner", "aligner", "aligner", "scrambler"),
    7: ("miner", "miner", "aligner", "aligner", "aligner", "aligner", "scrambler"),
    8: ("miner", "miner", "miner", "aligner", "aligner", "aligner", "aligner", "aligner"),
}

# Role Switching Constants (tunable by improvement loop)
SWITCH_HEARTLESS_THRESHOLD = 100  # ticks heartless before aligner → scrambler (was 30, too eager)
SWITCH_NO_TARGET_THRESHOLD = 80   # ticks with no alignable target → scrambler (was 20, too eager)
SWITCH_NEUTRAL_AVAILABLE = 1      # neutral junctions visible → scrambler → aligner (switch back fast)
SWITCH_COOLDOWN = 30              # min ticks between role switches (was 50, faster switch-back)
SWITCH_ENABLED_MAX_TEAM = 4       # enable dynamic switching at c2-c4 (tournament-relevant sizes)
SCRAMBLE_LINGER_TICKS = 5         # ticks to stay in AOE after scrambling a junction

# Miner element preference — diversify across 3 elements, silicon covered by fallback.
MINER_ELEMENT_PREF_IDX = {0: 0, 1: 1, 2: 2}  # carbon, oxygen, germanium; silicon via opportunistic
SECTOR_RADIUS = 30  # explore beyond hub alignment range to discover junctions for network expansion
HP_SAFETY_MARGIN = 15
ENERGY_MOVE_COST = 4
STUCK_THRESHOLD = 3  # same position this many times in recent history triggers rotation
MINER_DEPOSIT_THRESHOLD = 30  # mine more before depositing to reduce travel overhead
ALIGN_HUB_RADIUS = 25  # junctions within this distance of hub can be aligned
ALIGN_NET_RADIUS = 20  # P89: expanded from 15 — wider net reach helps frontier junctions


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
    last_hub_resources: dict[str, int] = field(default_factory=dict)
    agent_ids: list[int] = field(default_factory=list)
    # P211: Spatial memory — shared wall map and explored cell tracking
    walls: set[tuple[int, int]] = field(default_factory=set)
    explored: set[tuple[int, int]] = field(default_factory=set)
    _net_cache: set[tuple[int, int]] = field(default_factory=set)
    _net_cache_tick: int = field(default=-1)
    _junctions_version: int = field(default=0)
    # P213: Adaptive team playbook — detect self-pairing at c2
    cogs_junction_snapshot: int | None = field(default=None)
    playbook_applied: bool = field(default=False)

    def net_connected_junctions(self) -> set[tuple[int, int]]:
        """BFS from hub through cogs junctions within 25 cells. Cached by junction version."""
        if self._net_cache_tick == self._junctions_version:
            return self._net_cache
        connected: set[tuple[int, int]] = set()
        if self.hub_pos is None:
            self._net_cache = connected
            self._net_cache_tick = self._junctions_version
            return connected
        queue = [self.hub_pos]
        visited = {self.hub_pos}
        while queue:
            pos = queue.pop(0)
            for jpos, jalign in self.junctions.items():
                if jalign != "cogs" or jpos in visited:
                    continue
                if abs(jpos[0] - pos[0]) + abs(jpos[1] - pos[1]) <= 25:
                    visited.add(jpos)
                    connected.add(jpos)
                    queue.append(jpos)
        self._net_cache = connected
        self._net_cache_tick = self._junctions_version
        return connected

    def bottleneck_element(self) -> str | None:
        """Return the element with lowest hub inventory, or None if no data."""
        if not self.last_hub_resources:
            return None
        return min(ELEMENTS, key=lambda e: self.last_hub_resources.get(e, 0))

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
        """Find best alignable NEUTRAL junction using frontier scoring.
        P104: Pre-compute align range set to avoid redundant BFS (~300x per call)."""
        # Build the set of all positions that are in align range (BFS once, not per-junction)
        in_range: set[tuple[int, int]] = set()
        net = self.net_connected_junctions()
        for jpos in self.junctions:
            # Check hub range
            if self.hub_pos and abs(jpos[0] - self.hub_pos[0]) + abs(jpos[1] - self.hub_pos[1]) <= ALIGN_HUB_RADIUS:
                in_range.add(jpos)
                continue
            # Check net range
            for cpos in net:
                if abs(jpos[0] - cpos[0]) + abs(jpos[1] - cpos[1]) <= ALIGN_NET_RADIUS:
                    in_range.add(jpos)
                    break

        best, best_score = None, float("-inf")
        for jpos, jalign in self.junctions.items():
            if jalign != "neutral":
                continue
            if self.is_claimed(jpos, agent_id):
                continue
            if blacklist and jpos in blacklist:
                continue
            if jpos not in in_range:
                continue
            dist = abs(jpos[0] - pos[0]) + abs(jpos[1] - pos[1])
            frontier = 0
            for opos, oalign in self.junctions.items():
                if opos == jpos or oalign == "cogs":
                    continue
                if abs(opos[0] - jpos[0]) + abs(opos[1] - jpos[1]) <= ALIGN_NET_RADIUS:
                    if opos not in in_range:
                        frontier += 1
            score = frontier * 8.0 - dist
            if score > best_score:
                best_score = score
                best = jpos
        return best

    def nearest_hub_junction(
        self, pos: tuple[int, int], agent_id: int, blacklist: set[tuple[int, int]] | None = None,
    ) -> tuple[int, int] | None:
        """P167: For small teams, find closest non-cogs junction within hub range.
        Pure distance, no frontier scoring — speed over expansion."""
        if not self.hub_pos:
            return None
        best, best_dist = None, float("inf")
        for jpos, jalign in self.junctions.items():
            if jalign == "cogs":
                continue
            if self.is_claimed(jpos, agent_id):
                continue
            if blacklist and jpos in blacklist:
                continue
            # Must be within hub alignment range (25 cells)
            if abs(jpos[0] - self.hub_pos[0]) + abs(jpos[1] - self.hub_pos[1]) > ALIGN_HUB_RADIUS:
                continue
            dist = abs(jpos[0] - pos[0]) + abs(jpos[1] - pos[1])
            if dist < best_dist:
                best_dist = dist
                best = jpos
        return best

    def nearest_enemy_junction(
        self, pos: tuple[int, int], agent_id: int, blacklist: set[tuple[int, int]] | None = None,
    ) -> tuple[int, int] | None:
        """Fallback: find nearest clips junction in align range."""
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
        """P18: Check alignment range using NET-CONNECTED junctions only.
        Prevents targeting junctions near disconnected cogs junctions after cascade failure."""
        # Within 25 cells of hub
        if self.hub_pos:
            if abs(jpos[0] - self.hub_pos[0]) + abs(jpos[1] - self.hub_pos[1]) <= ALIGN_HUB_RADIUS:
                return True
        # Within 15 cells of any NET-CONNECTED cogs junction
        net = self.net_connected_junctions()
        for cpos in net:
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
    # Role switching state
    last_switch_tick: int = 0
    switch_count: int = 0
    role_ticks: int = 0
    no_target_ticks: int = 0
    switching_to: str | None = None  # non-None = mid-switch, navigating to gear station
    # P211: Spatial memory — track move direction for wall detection, frontier for exploration
    last_move_dir: str | None = None  # direction of last attempted move
    frontier_target: tuple[int, int] | None = None  # shared coords of current exploration frontier
    frontier_ticks: int = 0  # ticks navigating toward frontier_target


# ── Agent Implementation ─────────────────────────────────────────────────────

class SoftyAgentImpl(StatefulPolicyImpl[SoftyState]):

    def __init__(
        self,
        env: PolicyEnvInterface,
        agent_id: int,
        role: str | None,
        coordinator: SoftyCoordinator,
    ):
        self._env = env
        self._id = agent_id
        self._role = role  # None = assign dynamically on first step
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
        self._all_station_tags: set[int] = set()
        for st in self._station_tags.values():
            self._all_station_tags |= st
        self._deposit_tags = self._tags(["hub", "junction"])
        self._heart_tags = self._tags(["hub", "chest"])

        self._preferred_element = ELEMENTS[MINER_ELEMENT_PREF_IDX.get(agent_id, agent_id % len(ELEMENTS))]
        # P73: Fix sector collision. Was agent_id % 4 → only 4 of 8 sectors used.
        # With 0, sector_idx = agent_id % 8 → all 8 sectors covered, no overlap.
        self._explore_offset = 0

    def _assign_role(self) -> None:
        """P200: Role assignment using ROLE_DISTRIBUTIONS — includes scramblers at all team sizes.

        Competitive analysis: dinky/slinky use scramblers (30-177/episode). Without scrambling,
        clips junctions are permanently lost. ROLE_DISTRIBUTIONS ensures every team gets at least
        one scrambler (except c1/c2 where economy can't support it).
        """
        team_size = len(self._coord.agent_ids)
        dist = ROLE_DISTRIBUTIONS.get(team_size, ROLE_DISTRIBUTIONS[8])
        sorted_ids = sorted(self._coord.agent_ids)
        idx = sorted_ids.index(self._id) if self._id in sorted_ids else self._id % len(dist)
        self._role = dist[idx % len(dist)]
        if self._role == "miner":
            self._preferred_element = ELEMENTS[idx % len(ELEMENTS)]

    def _should_switch_role(self, tags: dict, inv: dict, s: SoftyState) -> str | None:
        """P200: Check if agent should switch roles. Returns new role or None.

        Tunable constants allow the improvement loop to optimize switching behavior.
        dinky/slinky switch 3-38 times per episode — this enables the same capability.
        """
        team_size = len(self._coord.agent_ids)
        if team_size < 3:
            return None  # c1-c2: pure aligners, never switch — replay data shows switching at c2 adds stuck + scrambling overhead
        if team_size > SWITCH_ENABLED_MAX_TEAM:
            return None
        if s.step_count - s.last_switch_tick < SWITCH_COOLDOWN:
            return None
        if self._role == "miner":
            return None
        if self._role == "aligner":
            if s.heartless_ticks > SWITCH_HEARTLESS_THRESHOLD:
                return "scrambler"
            if s.no_target_ticks > SWITCH_NO_TARGET_THRESHOLD:
                return "scrambler"
        elif self._role == "scrambler":
            neutral_count = sum(1 for a in self._coord.junctions.values() if a == "neutral")
            if neutral_count >= SWITCH_NEUTRAL_AVAILABLE:
                return "aligner"
        return None

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
        # P211: Track move direction for wall detection on next tick
        self._pending_move_dir = name[5:] if name.startswith("move_") else None
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

        # Sync hub resources to coordinator for cross-agent visibility
        if s.hub_resources:
            self._coord.last_hub_resources = dict(s.hub_resources)

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

        # P211: Record wall when last move failed — the cell we tried to enter is impassable
        # Skip if the blocking cell has an agent, junction, hub, extractor, or station
        # (those are interactive objects, not walls)
        if not s.last_move_succeeded and s.last_move_dir:
            delta = MOVE_DELTAS.get(s.last_move_dir)
            if delta:
                obs_r = cr + delta[0]
                obs_c = cc + delta[1]
                blocked_tags = tags.get((obs_r, obs_c), set())
                is_interactive = blocked_tags & (
                    self._agent_tags | self._junction_tags | self._hub_tags
                    | self._all_extractor_tags | self._heart_tags
                    | self._all_station_tags
                )
                if not is_interactive:
                    wall_shared = self._to_shared(s.row + delta[0], s.col + delta[1], s)
                    self._coord.walls.add(wall_shared)

        for loc, lt in tags.items():
            # Compute hub-relative (shared) position
            lp_r = s.row + (loc[0] - cr)
            lp_c = s.col + (loc[1] - cc)
            sp = self._to_shared(lp_r, lp_c, s)

            if lt & self._hub_tags and (lt & own_team):
                self._coord.hub_pos = sp  # Should be (0, 0)

            if lt & self._junction_tags:
                if lt & own_team:
                    new_align = "cogs"
                elif lt & (self._team_tags - own_team):
                    new_align = "clips"
                else:
                    new_align = "neutral"
                if self._coord.junctions.get(sp) != new_align:
                    self._coord.junctions[sp] = new_align
                    self._coord._junctions_version += 1

            for elem, etags in self._extractor_tags_by_elem.items():
                if lt & etags:
                    self._coord.extractors[sp] = elem

            for role_name, stags in self._station_tags.items():
                if lt & stags and (lt & own_team):
                    self._coord.stations[role_name] = sp

        # Agent position in shared coords for deconfliction
        self._coord.agent_positions[self._id] = self._to_shared(s.row, s.col, s)

        # P211: Mark visible cells as explored (shared coords)
        for obs_r in range(self._env.obs_height):
            for obs_c in range(self._env.obs_width):
                lp_r = s.row + (obs_r - cr)
                lp_c = s.col + (obs_c - cc)
                self._coord.explored.add(self._to_shared(lp_r, lp_c, s))

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

    def _go_absolute(self, target: tuple[int, int], s: SoftyState, tags: dict,
                     fallback_to_wander: bool = True) -> Action | None:
        """Navigate toward an off-screen target using BFS within visible window.
        P76: Added fallback_to_wander param to allow _wander to call this without recursion."""
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

        if fallback_to_wander:
            return self._wander(tags, s)
        return None

    def _wander(self, tags: dict[tuple[int, int], set[int]], s: SoftyState) -> Action:
        """P211+P215: Frontier-seeking exploration with deconfliction, fallback to sector cycling."""
        blocked = set(tags)
        blocked.discard(self._center)
        h, w = self._env.obs_height, self._env.obs_width

        if s.has_position and s.has_hub_offset and self._coord.hub_pos is not None:
            agent_shared = self._to_shared(s.row, s.col, s)

            # Use cached frontier target if still valid
            if (s.frontier_target is not None
                    and s.frontier_ticks < 15
                    and s.stuck_count == 0
                    and s.last_move_succeeded):
                if agent_shared != s.frontier_target:
                    s.frontier_ticks += 1
                    local = self._to_local(s.frontier_target[0], s.frontier_target[1], s)
                    action = self._go_absolute(local, s, tags, fallback_to_wander=False)
                    if action is not None:
                        return action
                # Reached or stuck — clear
                s.frontier_target = None
                s.frontier_ticks = 0

            # Search for nearest unexplored frontier cell (within radius 15)
            # Deconflict: offset search center by agent index
            offsets = [(0, 0), (8, 0), (0, 8), (-8, 0), (0, -8), (8, 8), (-8, -8), (8, -8), (-8, 8)]
            sorted_ids = sorted(self._coord.agent_ids)
            my_idx = sorted_ids.index(self._id) if self._id in sorted_ids else 0
            offset = offsets[my_idx % len(offsets)]
            search_center = (agent_shared[0] + offset[0], agent_shared[1] + offset[1])

            best_frontier = None
            best_dist = float("inf")
            for r in range(-15, 16):
                for c in range(-15, 16):
                    candidate = (search_center[0] + r, search_center[1] + c)
                    if candidate in self._coord.explored or candidate in self._coord.walls:
                        continue
                    # Must be adjacent to explored (it's a reachable frontier)
                    is_frontier = False
                    for dr, dc in MOVE_DELTAS.values():
                        if (candidate[0] + dr, candidate[1] + dc) in self._coord.explored:
                            is_frontier = True
                            break
                    if not is_frontier:
                        continue
                    dist = abs(candidate[0] - agent_shared[0]) + abs(candidate[1] - agent_shared[1])
                    if dist < best_dist:
                        best_dist = dist
                        best_frontier = candidate

            if best_frontier is not None:
                s.frontier_target = best_frontier
                s.frontier_ticks = 0
                local = self._to_local(best_frontier[0], best_frontier[1], s)
                action = self._go_absolute(local, s, tags, fallback_to_wander=False)
                if action is not None:
                    return action

            # Fallback: sector cycling (pre-calibration or all explored)
            sector_angles = [
                (-1, 0), (-1, 1), (0, 1), (1, 1),
                (1, 0), (1, -1), (0, -1), (-1, -1),
            ]
            sector_idx = (self._id + s.explore_dir_idx) % len(sector_angles)
            dr, dc = sector_angles[sector_idx]
            sector_shared = (dr * SECTOR_RADIUS, dc * SECTOR_RADIUS)
            sector_local = self._to_local(sector_shared[0], sector_shared[1], s)
            action = self._go_absolute(sector_local, s, tags, fallback_to_wander=False)
            if action is not None:
                return action

        # Final fallback: rotational exploration
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
        # P73: Use % 8 (not % 4) so stuck rotation covers all 8 sector angles
        if not s.last_move_succeeded:
            s.stuck_count += 2
            s.explore_dir_idx = (s.explore_dir_idx + 1) % 8
        elif len(s.recent_positions) >= 4 and s.recent_positions.count(pos) >= STUCK_THRESHOLD:
            s.stuck_count += 1
            s.explore_dir_idx = (s.explore_dir_idx + 1) % 8
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
        # P168: In c2 (≤3 agents), deposit at threshold 4 instead of 30.
        # Hub starts with 6 of each element, needs 7 of ALL FOUR to craft a heart.
        # Low threshold forces fast bottleneck rotation: 4 carbon → deposit → 4 oxygen → ...
        # First heart craftable after ~4 trips (~136 ticks) instead of ~500+ with threshold 30.
        deposit_thresh = 4 if len(self._coord.agent_ids) <= 3 else MINER_DEPOSIT_THRESHOLD
        # On-screen: deposit at nearest visible friendly building (hub or junction)
        # Off-screen: always navigate to hub (junctions might get scrambled during transit)
        if cargo >= deposit_thresh:
            vis = self._closest(tags, self._deposit_tags, require=own)
            if vis is not None:
                action = self._go_visible(vis, tags)
                if action is not None:
                    return action
            if self._coord.hub_pos is not None:
                return self._go_to_known(self._coord.hub_pos, tags, s)
            return self._wander(tags, s)

        # Phase 3: mine — dynamically prefer the bottleneck element (lowest hub inventory)
        bottleneck = self._coord.bottleneck_element()
        pref_elem = bottleneck if bottleneck else self._preferred_element
        pref_tags = self._extractor_tags_by_elem.get(pref_elem, self._all_extractor_tags)
        vis = self._closest(tags, pref_tags)
        if vis is None:
            vis = self._closest(tags, self._all_extractor_tags)
        if vis is not None:
            action = self._go_visible(vis, tags)
            if action is not None:
                return action

        if s.has_position and s.has_hub_offset:
            shared_pos = self._to_shared(s.row, s.col, s)
            target = self._coord.nearest_extractor(shared_pos, pref_elem, self._id)
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
            # Heartless: explore to discover junctions, periodically check hub for hearts.
            hub_can_craft = False
            if s.hub_resources:
                counts = [s.hub_resources.get(e, 0) for e in ELEMENTS]
                hub_can_craft = min(counts) >= 7
            # P167: Small teams check hub more often (every 20 ticks vs 30).
            # Hearts are scarce in c2 — minimize wait time.
            hub_interval = 20 if len(self._coord.agent_ids) <= 3 else 30
            if hub_can_craft or s.heartless_ticks % hub_interval == 5:
                return self._go_to_heart_source(tags, own, s)
            # P162: Pre-position toward known junction while heartless (c4+ only).
            # Replay data (cycle 65): pre-positioning at c2 causes 100-6000+ stuck ticks
            # because agents navigate to impassable junctions → stuck → hub → repeat.
            # v29 (no pre-positioning) had 6 stuck ticks vs v37's 100-6142.
            # Only enable at c4+ where teammates provide enough junction discovery.
            if len(self._coord.agent_ids) > 3 and s.stuck_count < 3 and s.has_position and s.has_hub_offset:
                shared_pos = self._to_shared(s.row, s.col, s)
                target = self._coord.nearest_alignable_junction(shared_pos, self._id, s.failed_junctions)
                if target is None:
                    target = self._coord.nearest_enemy_junction(shared_pos, self._id, s.failed_junctions)
                if target:
                    self._coord.claim_target(self._id, target)
                    return self._go_to_known(target, tags, s)
            return self._wander(tags, s)

        s.had_heart_last_step = True
        s.heartless_ticks = 0

        # P65→P89: Timeout 15→25→35→50. P127 (50→60) REVERTED — locally +5.9%
        # but tournament regression: v25(60) at 23.40 vs v24(50) at 25.60.
        # 50 ticks ≈ 20 cells — covers alignment range. 60 ticks wastes time
        # on junctions scrambled by real opponents during approach.
        if s.last_target_pos is not None:
            s.target_ticks += 1
            if s.target_ticks > 50:
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
                # P48: Write claim for visible junction — deconflicts c2 agents
                # targeting the same junction when both see it on-screen.
                self._coord.claim_target(self._id, junc_shared)
                s.no_target_ticks = 0
                action = self._go_visible(vis, tags)
                if action is not None:
                    return action

        # Find best junction to align
        if s.has_position and s.has_hub_offset:
            shared_pos = self._to_shared(s.row, s.col, s)
            # P167: c1 uses hub-adjacent targeting (closest, no frontier scoring) —
            # solo agent needs fast alignment cycles near hub for healing/hearts.
            # c2+: frontier scoring leverages teammate network expansion.
            if len(self._coord.agent_ids) <= 1:
                target = self._coord.nearest_hub_junction(shared_pos, self._id, s.failed_junctions)
            else:
                target = self._coord.nearest_alignable_junction(shared_pos, self._id, s.failed_junctions)
            if target is None:
                target = self._coord.nearest_enemy_junction(shared_pos, self._id, s.failed_junctions)
            if target:
                self._coord.claim_target(self._id, target)
                if target != s.last_target_pos:
                    s.last_target_pos = target
                    s.target_ticks = 0
                s.no_target_ticks = 0
                return self._go_to_known(target, tags, s)

        s.no_target_ticks += 1
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
            s.heartless_ticks += 1
            return self._go_to_heart_source(tags, own, s)
        s.heartless_ticks = 0

        # P212: Timeout on scrambler targets — prevents infinite navigation to unreachable junctions
        # Same pattern as aligner timeout (50 ticks). Blacklists unreachable junctions.
        if s.last_target_pos is not None:
            s.target_ticks += 1
            if s.target_ticks > 50:
                s.failed_junctions.add(s.last_target_pos)
                self._coord.claim_target(self._id, None)
                s.last_target_pos = None
                s.target_ticks = 0

        # Find enemy (clips) junction
        vis = self._closest(tags, self._junction_tags, require=enemy)
        if vis is not None:
            if s.has_hub_offset:
                junc_shared = self._to_shared(
                    s.row + (vis[0] - self._center[0]),
                    s.col + (vis[1] - self._center[1]),
                    s,
                )
                if junc_shared not in s.failed_junctions:
                    s.last_target_pos = junc_shared
                    s.target_ticks = 0
                    s.no_target_ticks = 0
                    action = self._go_visible(vis, tags)
                    if action is not None:
                        return action

        if s.has_position and s.has_hub_offset:
            shared_pos = self._to_shared(s.row, s.col, s)
            target = self._coord.nearest_junction(shared_pos, "clips", self._id)
            if target and target not in s.failed_junctions:
                self._coord.claim_target(self._id, target)
                if target != s.last_target_pos:
                    s.last_target_pos = target
                    s.target_ticks = 0
                s.no_target_ticks = 0
                return self._go_to_known(target, tags, s)

        s.no_target_ticks += 1
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
        # P211: Save pending move direction before parse (for wall detection)
        s.last_move_dir = getattr(self, '_pending_move_dir', None)
        tags, inv = self._parse(obs, s)

        # P102: Assign role dynamically on first step (all agents registered by now)
        if self._role is None:
            self._assign_role()

        self._update_stuck(s)
        s.step_count += 1
        # P214: Clear junction blacklist periodically — junctions change state over time
        if s.step_count % 500 == 0:
            s.failed_junctions.clear()
        s.role_ticks += 1

        # P200: If mid-switch, continue navigating to gear station
        # P212: Add timeout — if stuck navigating to station for 100 ticks, cancel switch
        if s.switching_to is not None:
            has_new_gear = inv.get(s.switching_to, 0) > 0
            if has_new_gear:
                self._role = s.switching_to
                s.switching_to = None
                s.last_switch_tick = s.step_count
                s.switch_count += 1
                s.role_ticks = 0
                s.no_target_ticks = 0
                s.heartless_ticks = 0
            elif s.step_count - s.last_switch_tick > 100:
                # Timed out finding gear station — cancel switch, stay in current role
                s.switching_to = None
                s.last_switch_tick = s.step_count
            else:
                own = tags.get(self._center, set()) & self._team_tags
                return self._go_to_role_station(s.switching_to, tags, own, s), s

        # P200: Check for dynamic role switch (before role dispatch)
        new_role = self._should_switch_role(tags, inv, s)
        if new_role is not None:
            s.switching_to = new_role
            s.last_switch_tick = s.step_count  # P212: mark switch start for timeout
            own = tags.get(self._center, set()) & self._team_tags
            return self._go_to_role_station(new_role, tags, own, s), s

        # Wait for energy regen if too low to move
        if self._low_energy(inv):
            return self._act(self._noop), s

        # Retreat if HP is critical and we have gear to lose
        if self._should_retreat(inv, s):
            return self._retreat(tags, s), s

        # Severely stuck — go to hub as waypoint to reset
        if s.stuck_count > 5 and s.has_position and s.has_hub_offset and self._coord.hub_pos is not None:
            if self._role == "aligner" and s.last_target_pos is not None:
                s.failed_junctions.add(s.last_target_pos)
                self._coord.claim_target(self._id, None)
                s.last_target_pos = None
                s.target_ticks = 0
            s.stuck_count = 0
            s.explore_dir_idx = (s.explore_dir_idx + 1) % 8
            return self._go_to_known(self._coord.hub_pos, tags, s), s

        # P213: Adaptive team playbook — detect excess alignment capacity at c2
        team_size = len(self._coord.agent_ids)
        if team_size == 2 and s.step_count == 400 and not self._coord.playbook_applied:
            cogs_count = sum(1 for a in self._coord.junctions.values() if a == "cogs")
            self._coord.cogs_junction_snapshot = cogs_count
            if cogs_count >= 12:
                # High alignment rate — teammate is likely another aligner (Softy or similar)
                # Switch second agent (higher ID) to scrambler for diversity
                sorted_ids = sorted(self._coord.agent_ids)
                if self._id == sorted_ids[-1]:
                    self._role = "scrambler"
                    s.switching_to = "scrambler"
                    s.last_switch_tick = s.step_count
            self._coord.playbook_applied = True

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


# ── Torch-Free Agent Wrapper ──────────────────────────────────────────────────
# mettagrid's StatefulAgentPolicy.step() unconditionally imports torch and wraps
# calls in torch.no_grad(). The beta-cvc server (compat 0.25) sandbox doesn't
# install torch, so all qualifying matches fail with ModuleNotFoundError.
# This wrapper provides identical behavior without the torch dependency.


class _SoftyAgentPolicy(AgentPolicy):
    """Lightweight AgentPolicy that manages SoftyState without torch."""

    def __init__(self, impl: SoftyAgentImpl, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)
        self._impl = impl
        self._state: SoftyState = impl.initial_agent_state()

    def step(self, obs: AgentObservation) -> Action:
        action, self._state = self._impl.step_with_state(obs, self._state)
        return action


# ── Top-Level Policy ──────────────────────────────────────────────────────────

class SoftyPolicy(MultiAgentPolicy):
    short_names = ["softy"]

    def __init__(self, policy_env_info: PolicyEnvInterface, device: str = "cpu"):
        super().__init__(policy_env_info, device=device)
        self._coordinator = SoftyCoordinator()
        self._agents: dict[int, _SoftyAgentPolicy] = {}

    def agent_policy(self, agent_id: int) -> _SoftyAgentPolicy:
        if agent_id not in self._agents:
            self._coordinator.agent_ids.append(agent_id)
            # P102: role=None → assigned dynamically on first step based on team size
            impl = SoftyAgentImpl(self._policy_env_info, agent_id, None, self._coordinator)
            self._agents[agent_id] = _SoftyAgentPolicy(impl, self._policy_env_info)
        return self._agents[agent_id]
