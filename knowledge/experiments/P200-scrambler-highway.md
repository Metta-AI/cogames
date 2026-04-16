# P200: Scrambler Highway — REVISED with Tournament Replay Data

## Status: PARTIALLY REVERTED — c8 scramblers caused catastrophic regression

## What P200 Does (in softy.py)
1. ROLE_DISTRIBUTIONS replacing ROLE_CYCLE (scramblers at c3-c7 only, NOT c8)
2. Dynamic role switching: aligner↔scrambler at c3-c4 only (SWITCH_ENABLED_MAX_TEAM=4)
3. Switching constants: HEARTLESS=100, NO_TARGET=80, COOLDOWN=30, NEUTRAL_AVAILABLE=1
4. State tracking: no_target_ticks, switching_to, role_ticks in SoftyState
5. State machine in step_with_state for mid-switch gear station navigation

## CRITICAL FINDING: c8 Scramblers Destroy Performance (Cycle 65 Replay Research)

### Tournament replay data (beta-teams-tiny-fixed stage-1, c8 solo)
| Version | Distribution | c8 Score | Notes |
|---------|-------------|----------|-------|
| v29 | 3M/5A/0S | **38.6** | Pre-P200, best c8 |
| v24 | pre-P200 | 34.1 | |
| v33 | 2M/4A/2S | 31.4 | Scramblers broken (never got gear) — "lucky bug" |
| v34 | 2M/4A/2S | 9.5 | Scramblers functional, performance collapses |
| v37 | 2M/4A/2S | **4.6** | Scramblers very active (18 scrambled/agent) |

### Why v33's scramblers didn't work
v33 had SWITCH_HEARTLESS_THRESHOLD=30. At c4 (switching enabled), scramblers switched
to aligner in 30 ticks before finding the scrambler station. At c8 (switching disabled),
scramblers wandered forever without finding the station. Result: scramblers were useless
wanderers — bad but not actively harmful. v33's 31.4 at c8 was propped up by this bug.

### Why functional scramblers destroy c8
1. 2 scramblers convert 18 enemy→neutral per agent (costs heart each)
2. Only 4 aligners to re-align (vs 5 in v29) — can't keep up
3. 1344 stuck ticks navigating to distant enemy junctions
4. 4579 failed moves (43% failure rate vs v29's 16%)
5. Hearts consumed by scrambling → fewer for alignment

### Fix applied (cycle 65)
ROLE_DISTRIBUTIONS[8] reverted to 3M/5A/0S (v29's proven distribution).
c5-c7 keep 1 scrambler. c8 gets 0 scramblers.

## CRITICAL FIX NEEDED: c1/c2 Role Distributions

### The Problem

Tournament episode data (from `cogames episode list --policy <UUID> --json`):

| Softy Version | Agent distribution | Most common |
|---------------|-------------------|-------------|
| v24 (rank 2, 25.60) | 25× c1, 1× c2 | **c1 (96%)** |
| v29 (rank 5, 24.43) | 12× c1, 5× c2, 3× c3 | **c1 (60%)** |

**Softy almost always gets 1 agent.** The 8 cogs agents are split across multiple policies on the SAME TEAM. Everyone shares hub, hearts, junctions.

Current `ROLE_DISTRIBUTIONS[1] = ("scrambler",)` → solo agent scrambles.

v24's old code made c1 an aligner. That solo aligner:
- Got hearts from shared hub (teammates mine)
- Aligned 48-99 junctions per episode
- Earned 27-51 reward per episode

A solo scrambler:
- Converts enemy→neutral (costs hearts from shared hub)
- NO follow-up aligner — relies on teammates to align neutralized junctions
- Teammates already have their own scramblers (dinky: 30-177 scrambler.gained/ep)
- Softy's contribution is duplicating work teammates already do

### The Fix

```python
ROLE_DISTRIBUTIONS = {
    1: ("aligner",),                    # Solo: align. Teammates mine + scramble.
    2: ("aligner", "aligner"),          # Both align. Teammates provide economy.
    3: ("aligner", "aligner", "scrambler"),  # Add scrambler once we can spare one
    4: ("miner", "aligner", "aligner", "scrambler"),
    # ...c5-c8 unchanged...
}
```

**Why no miners at c1-c3:** Teammates mine. dinky:v27 at c2 has 0 miners — their teammate mines for them. Hub starts with elements + pre-crafted hearts. At c1-c2, every agent spent on mining is wasted alignment capacity.

**Why no scrambler at c1-c2:** Teammates already scramble (dinky: 30-177/ep). One more scrambler adds marginal value. One more aligner adds direct scoring capacity.

**Why scrambler at c3:** With 3 agents, we can spare one for scrambling. This lets us create our own neutral junctions instead of relying entirely on teammates.

### How to verify the fix is correct (tournament monitoring)

After upload, check episodes:
```bash
uv run cogames episode list --policy <NEW_UUID> --json | python3 -c "
import json, sys
for ep in json.loads(sys.stdin.read()):
    for pr in ep.get('policy_results', []):
        if pr['policy']['name'] == 'Softy':
            n = pr['num_agents']
            m = pr.get('avg_metrics', {})
            print(f'agents={n} reward={pr[\"avg_reward\"]:.1f} aligner.gained={m.get(\"aligner.gained\",0)} scrambler.gained={m.get(\"scrambler.gained\",0)} junction.aligned={m.get(\"junction.aligned_by_agent\",0)}')
"
```

Look for:
- c1: junction.aligned_by_agent > 40 (matching v24 levels) ← proves aligner is working
- c3+: scrambler.gained > 0 ← proves scrambler capability is active
- avg_reward ≥ v24 levels (40+ for c1)

## Role Switching Still Applies at c3+

Dynamic switching is gated on `team_size >= 3` (was already `<= SWITCH_ENABLED_MAX_TEAM=4`). At c3, the scrambler can switch to aligner when neutrals accumulate, and the aligner can switch to scrambler when heartless too long.

But add a floor: `if team_size < 3: return None` in `_should_switch_role()`. c1/c2 agents should never switch — they should stay as pure aligners.

## How to Reproduce the Tournament Research

### Get any policy's episodes
```bash
# 1. Find policy UUID
uv run cogames leaderboard beta-cvc --json | python3 -c "
import json, sys
for r in json.loads(sys.stdin.read()):
    p = r['policy']
    if r['score'] > 20:
        print(f\"{p['name']}:v{p['version']} id={p['id']} score={r['score']:.1f}\")
"

# 2. Get episodes with full per-agent metrics
uv run cogames episode list --policy <UUID> --json

# 3. Analyze agent count distribution + metrics
uv run cogames episode list --policy <UUID> --json | python3 -c "
import json, sys
data = json.loads(sys.stdin.read())
counts = {}
for ep in data:
    for pr in ep.get('policy_results', []):
        if pr['policy']['name'] == 'TARGET_POLICY':
            n = pr['num_agents']
            counts[n] = counts.get(n, 0) + 1
            m = pr.get('avg_metrics', {})
            print(f'agents={n} reward={pr[\"avg_reward\"]:.1f} aligner.gained={m.get(\"aligner.gained\",0)} scrambler.gained={m.get(\"scrambler.gained\",0)}')
print('Distribution:', dict(sorted(counts.items())))
"
```

### Key findings from this research

**Every top-5 policy uses scramblers** (30-177 scrambler.gained per episode). But they have 2-6 agents, enough to dedicate some to scrambling while others align.

**Softy with 1 agent cannot afford to scramble.** The single agent must maximize direct scoring (alignment). Scrambling is a support action — valuable when you have spare agents, wasteful when you're the whole team.

**Tournament is NOT adversarial between cogs policies.** All cogs policies cooperate on the same team vs clips NPC. Reward is shared based on team-wide junction.held. Individual score depends on how much YOUR agents contributed to junctions the team holds.

## Future Hypotheses (after c1/c2 fix validates)

- P201-P207: Tune switching constants for c3-c4 (where switching applies)
- P208: c3 distribution — ("aligner", "scrambler", "aligner") vs ("aligner", "aligner", "scrambler")
- P209: At c1, should switching be enabled to go scrambler when heartless too long? (probably not — better to just wander and discover junctions)
- P210: c2 distribution — add 1 scrambler? Depends on whether teammates saturate scrambling
