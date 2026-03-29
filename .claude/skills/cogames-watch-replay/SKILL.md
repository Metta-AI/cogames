---
name: cogames-watch-replay
description: Run a cogames episode and capture emoji map frames so you can observe agent behavior without a GUI. Use this to diagnose navigation, gear acquisition, or routing issues.
argument-hint: "[--steps 1000] [--every 100] [--aligners 3]"
---

**Important**: The env file lives at `../cogames/.env.openrouter.local`. Source it before running:

```bash
source ../cogames/.env.openrouter.local && export PATH="$HOME/.nimble/bin:$PATH" && .venv/bin/python scripts/capture_frames.py $ARGUMENTS --out docs/autoresearch_director/frames.txt 2>&1
```

After the script completes, read the frames file and run the analyses below.

## Step 1: Extract agent positions programmatically

The emoji grid is 98×98 — too large to eyeball. Write a python snippet to find agent symbols (`🟦`=A0, `🟧`=A1, `🟩`=A2, `🟨`=A3) and key landmarks in each frame. Record (row, col) per frame.

```python
# Parse each frame, find agent positions, junctions, stations
for sym, name in [('🟦','A0'), ('🟧','A1'), ('🟩','A2'), ('🟨','A3')]:
    if sym in line:
        col = line.index(sym)
        print(f'{name} at row={row_idx} col={col//2}')
```

Also find: `🔗`=aligner station, `⛏️`=miner station, `🔭`=scout station, `🌀`=scrambler station, `🚀`=junction.

## Step 2: Compute movement deltas and detect stuck agents

For each agent, compute Manhattan distance between consecutive frames. If distance ≤ 2 across a frame interval, the agent is STUCK.

Report: how many frame intervals (out of total) each agent is stuck. An agent stuck for >30% of the episode is a major problem.

## Step 3: Analyze reward growth rate

Rewards appear in each frame header: `rewards=[0.1234 0.1234 0.1234]`. Compute the delta per interval. A decelerating reward rate means agents are becoming less productive — usually because they're stuck or out of hearts.

Key pattern: if reward growth drops below 50% of its peak rate, something broke (hub depleted, agents stuck, navigation failure).

## Step 4: Zoom into stuck areas

Once you identify where an agent is frozen, extract the 15-row subgrid around that position to see what's blocking it:
- ⬛ walls: agent stuck against a wall
- 📦 resource extractors: invisible obstacles the agent can't navigate around
- 🔗⛏️🔭🌀 stations: agent may have accidentally picked up wrong gear
- Other agents: collision/congestion

## Step 5: Parse LLM logs from the run output

The script's stderr contains LLM decision logs. Analyze:
- **Skill selection distribution**: count how often each skill is chosen per agent
- **Stale exits**: count lines containing "stale" — these are failed navigation attempts
- **has_heart ratio**: count True vs False — if False dominates (>60%), hub depletion is active
- **Hallucinated skill names**: check if LLM outputs invalid skill names (e.g., "unstick" instead of "unstuck")
- **LLM response times**: look for `llm_response_ms` values — >3000ms indicates contention

## What to look for (diagnostic checklist)

### Gear acquisition
- Do agents reach gear stations within first 50-100 steps?
- Does any agent sit ON a station without moving? (gear contamination)
- Do agents pass near wrong-type stations? (scrambler/scout contamination risk)

### Hub and hearts
- When does has_heart flip from True to False and stay False? (hub depletion point)
- How many get_heart→stale cycles happen? (>10 means hub awareness is broken)
- The hub has exactly 5 hearts. Mining does NOT create more.

### Navigation and exploration
- Do agents reach the map corners? Junctions are often at the edges (rows 6-7, 91-92 on the 98×98 map)
- Do all agents cluster in the same area? (no dispersion = wasted capacity)
- What's the maximum distance any agent reaches from spawn (~row 50, col 40)?

### Multi-agent issues
- Are agents blocking each other near the hub or stations?
- Does 8-agent config crash on LLM timeout? (confirms contention problem)
- Compare per-agent reward: 1-agent vs 3-agent vs 8-agent — does it scale linearly?

## Recommended configs

- **Gear-up phase**: `--steps 200 --every 10` — watch first 200 steps closely
- **Standard diagnostic**: `--steps 500 --every 50`
- **Full episode**: `--steps 1000 --every 100`
- **Isolate single agent**: `--agents 1 --aligners 1 --steps 500 --every 50`
- **Scaling test**: run 1-agent, 3-agent, 8-agent and compare reward curves
