---
name: cogames-watch-replay
description: Run a cogames episode and capture emoji map frames so you can observe agent behavior without a GUI. Use this to diagnose navigation, gear acquisition, or routing issues.
argument-hint: "[--steps 1000] [--every 100] [--aligners 3]"
---

Run the frame capture script with the provided arguments (or defaults), then read and analyze the output.

```bash
python scripts/capture_frames.py $ARGUMENTS --out docs/autoresearch_director/frames.txt 2>&1
```

After the script completes, read the frames file and describe what you observe:
- Where are the agents at each snapshot?
- Are they moving or stuck?
- Are they spreading across the map or clustering?
- Do you see agents near the gear stations (🔗⛏️🔭🌀) or hub?
- Any agents that stop moving between frames?
