---
name: cogames-watch-replay
description: Run a cogames episode and capture emoji map frames so you can observe agent behavior without a GUI. Use this to diagnose navigation, gear acquisition, or routing issues.
argument-hint: "[--steps 1000] [--every 100] [--aligners 3]"
---

Run the frame capture script with the provided arguments (or defaults), then read and analyze the output.

```bash
python scripts/capture_frames.py $ARGUMENTS --out docs/autoresearch_director/frames.txt 2>&1
```

After the script completes, read the frames file and analyze it:

## How to read the output

**Parse programmatically, not visually** — the emoji grid is 98×98, too large to eyeball. Extract agent positions by searching for each symbol (`🟦`, `🟧`, `🟩`, `🟨`) and record their (row, col) coordinates across frames. Then compute movement deltas to detect stuck agents.

**Questions to answer:**
- Are agents moving between frames, or returning to the same position?
- Are they spreading across the map or clustering near the hub?
- Do they reach gear stations (🔗⛏️🔭🌀) and change gear?
- When do alignments happen (watch reward jumps in the step header)?

**Zoom into stuck areas** — once you identify where an agent is frozen, extract just the 15×15 subgrid around that position to see what's blocking it (walls ⬛, resource extractors 📦, stations).

## Recommended configs

- Short episode, fine-grained: `--steps 200 --every 10`
- Standard diagnostic: `--steps 500 --every 50`
- Full episode: `--steps 1000 --every 100`
- Isolate single agent: `--agents 1 --steps 500 --every 50`
- Multi-agent contention: run 1-agent vs 3-agent vs 8-agent and compare reward/alignment counts
