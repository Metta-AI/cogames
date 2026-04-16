# Orient State (auto-regenerated each cycle)

cycle: 65
base_version: v37 (P200 + switching fixes, deployed to beta-teams-tiny-fixed)
local_vor: N/A (cooperative sim — scrimmage misleading)
consecutive_fails: 0
mode: fixing_tournament_regression
last_working_upload: v37 (April 15)

## CRITICAL: v37 c8 Regression (stage-1 = 4.6 reward, LAST PLACE)

### Root cause: P200 scramblers at c8
- v29 (3M/5A/0S): 38.6 reward at c8 — BEST Softy version
- v37 (2M/4A/2S): 4.6 reward at c8 — WORST (scramblers tank performance)
- v33 (2M/4A/2S): 31.4 at c8 — scramblers assigned but never got gear ("lucky bug")
- **FIX**: Remove scramblers from c8, revert to 3M/5A/0S

### Stuck behavior from pre-positioning (line 903)
- v29: 6 stuck ticks EVERY episode (healthy)
- v37: 100-6142 stuck ticks (catastrophic)
- Cause: heartless pre-positioning navigates to impassable junctions
- **FIX**: Remove or guard pre-positioning code

### c2 switching adds stuck + scrambling overhead
- v37 at c2 scrambles 1-14 junctions/agent (from switching), costs alignment
- At c2, teammates mask this. At c8, it's fatal.
- **FIX**: Disable switching at c2 (guard: team_size < 3)

## TWO TOURNAMENTS — Both Matter

### beta-teams-tiny-fixed (ranked, complete)
Tournament stages: stage-1 (c8 solo) → stage-2 (c4 2-way) → stage-3 (c2 4-way) → team rounds

| Version | Stage-1 (c8) | Stage-2 (c4) | Stage-3 (c2) | Leaderboard |
|---------|-------------|-------------|-------------|-------------|
| v29 | **38.6** | 26.5 | 27.5 | 18.00 (#7) |
| v24 | 34.1 | 26.5 | 27.8 | 35.00 (#10) |
| v33 | 31.4 | **29.7** | 27.2 | — |
| v34 | 9.5 | 19.4 | 20.8 | — |
| v37 | **4.6** | 27.8 | 27.4 | — |

Top: slinky:v2=10.00, dinky:v28=10.00 (lower=better)

### beta-cvc (freeplay)
- v24 at rank #2 (score 25.60). dinky:v27 is #1 at 27.31.
- v30-v37 not on beta-cvc leaderboard (server issues during qualifying)

## Cross-Policy Stage Rankings

**Stage-3 (c2, most data, ~600 matches each):**
dinky:v28=29.3, slanky:v152=29.2, slanky-teams-tiny:v3=29.0, slanky:v145=28.8
...Softy:v24=27.8, v29=27.4, v37=27.3, v33=27.1
**Consistent ~1.5 pt gap to slanky. Slanky explores 2x territory (2051 vs 838 cells).**

## CHANGES THIS CYCLE (cycle 65)
- Removing scramblers from ROLE_DISTRIBUTIONS[8] → 3M/5A/0S
- Disabling switching at c2 (team_size < 3 guard)
- Fixing stuck behavior from pre-positioning
- Testing across c2/c4/c6/c8

## Do NOT Retry
- Behavior tuning (0 for 100+)
- Hub-reducing changes
- ALIGN_NET_RADIUS 15 (v28 proved wrong, stay at 20)
- timeout 60 (v28 proved wrong, stay at 50)
- c1/c2 scramblers or miners
- HP_SAFETY_MARGIN > 15
- Always-hub heartless at c1/c2
- SWITCH_ENABLED_MAX_TEAM > 4
- Night retreat scaling (heal_dist * 3)
