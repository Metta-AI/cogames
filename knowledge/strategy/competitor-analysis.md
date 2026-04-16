# Competitor Analysis (April 15, 2026 — Deep Research)

Source: Episode-level analysis of 78+ episodes via `cogames episode list --policy <UUID> --json`

## Current Leaderboard

| Rank | Policy | Score | Matches |
|------|--------|-------|---------|
| 1 | dinky:v27 | 27.31 | 127 |
| 2 | Softy:v24 | 25.60 | 20 |
| 3 | Softy:v22 | 24.98 | 20 |
| 5 | Softy:v29 | 24.43 | 21 |
| 6 | slanky:v129 | 23.92 | 20 |
| 34 | Softy:v28 | 18.47 | 20 |
| — | slinky:v1 | 20.65 | 25 |
| — | rohit_test_slanky:v1 | 20.57 | 20 |

Gap to #1: 1.71 points (25.60 vs 27.31). Softy is a scripted policy competing with likely-trained policies.

## Cross-Policy Comparison (Episode Data)

| Metric | Softy v24 | dinky v27 | slinky v1 | rohit_test v1 |
|--------|-----------|-----------|-----------|---------------|
| **Agent distribution** | 96% c1 | 42/42/17% c2/c4/c6 | 35/25/40% | 35/30/35% |
| **Deaths/agent** | 8.6 | 9.4 | **3.5** | 8.5 |
| **junction.aligned/agent** | **61.2** | 34.3 | 27.1 | 26.1 |
| **scrambler.gained** | **0** | 4.0 | 1.7 | 2.9 |
| **Move success rate** | **99.6%** | 84.1% | 99.4% | 95.0% |
| **Max stuck time (steps)** | **4.8** | 2094 | 188 | 1089 |
| **Cells explored** | 1714 | **2508** | 2223 | 2312 |
| **HP per heart** | 162 | **313** | 280 | 299 |
| **Hearts collected** | **64.9** | 38.5 | 41.1 | 41.0 |
| **Consistency (CV)** | **21%** | 42% | 56% | 99% |
| **Resources collected** | **1028 avg** | 173 | 539 | 611 |

## Softy's Strengths (DO NOT BREAK)

1. **Navigation**: 99.6% move success, 4.8 stuck ticks vs 188-2094 for competitors
2. **Junction alignment**: 61.2/agent — 2x competitors (but they have more agents)
3. **Consistency**: 21% CV in junction alignment vs 42-99% for competitors
4. **Resource collection**: 2-7x more resources per agent than competitors

## Softy's Gaps (IMPROVEMENT TARGETS)

### Gap 1: No Scrambler Usage
- Softy: 0 scrambler.gained across ALL episodes
- dinky: 4.0/ep, slinky: 1.7/ep, rohit: 2.9/ep
- All competitors use scramblers; Softy is the only one without

### ~~Gap 2: HP Efficiency Deficit~~ [RESOLVED — RED HERRING]
- The 162 vs 280-313 "HP per heart" ratio is **meaningless** — hearts do NOT give HP
- Hearts = alignment/scramble currency (1 per action). HP = separate system (territory healing +100/tick)
- `hp.gained / heart.gained` divides two independent resource systems
- The real death rate gap (Softy 8.6 vs slinky 3.5) is caused by territory presence time and gear HP modifiers (+200 for scrambler, +400 for scout), NOT heart collection

### Gap 3: Less Exploration (30-46%)
- Softy: 1714 unique cells visited
- Competitors: 2223-2508 (30-46% more)
- Competitors explore more territory despite having worse navigation

### Gap 4: Aligner Turnover
- Softy gains 5.8 aligners but loses 8.7 (net -2.8)
- Competitors: dinky gains 13.8/loses 13.5 (net +0.2)
- Softy loses more aligners than it picks up — frequent deaths while geared

## Stage-3 Team Dynamics (Cycle 65 Research)

### Teammate Impact on Softy:v29 Reward (stage-3, c2)
| Teammate | Softy reward | Delta |
|----------|-------------|-------|
| slanky-teams-tiny:v3 | 29.6 | +2.2 |
| slanky-teams-tiny:v2 | 29.3 | +1.9 |
| dinky:v28 | 28.8 | +1.4 |
| slanky:v152 | 28.5 | +1.1 |
| slinky:v2 | 26.9 | -0.5 |
| Softy:v24 | 27.2 | -0.2 |
| Softy:v33 | 26.1 | -1.3 |
| Softy:v34 | 20.1 | -7.3 |

### Softy self-pairing penalty
| Softy versions on team | Avg reward | Bad game rate |
|------------------------|-----------|--------------|
| 1 | 29.6 | 11% |
| 2 | 27.2 | 17% |
| 3 | 21.8 | 33% |
| 4 | 21.3 | 50% |

**Each additional Softy costs ~4 points.** All Softy versions are pure aligners at c2 — they compete for junctions and lack scrambling diversity. slanky/dinky variants provide complementary value.

### Best team compositions (stage-3)
Top teams: dinky + slanky variants + slinky (48.3 max reward)
Best with Softy: Softy:v24 + dinky:v28 + slanky-teams-tiny:v2 + slinky:v2 (45.0)
Worst: 3+ Softy versions together (5.6-6.7)

### Strategic implication
The 1.5pt gap to slanky is primarily matchup distribution (32% of matches have 2+ Softy versions), not code quality. Softy's alignment output (32.3/agent) is competitive with dinky (34.3). Reducing the number of Softy submissions in stage-3 would have more impact than code changes.

## dinky Deep-Dive (The #1 Policy)

### Reward Correlations
| Metric | Correlation with dinky's reward |
|--------|-------------------------------|
| junction.aligned_by_agent | **+0.832** (dominant) |
| agent count | +0.330 |
| deaths | -0.416 |
| action.failed | -0.485 |
| scrambler.gained | -0.495 |
| aligner.gained | -0.386 |

**Junction alignment is THE predictor** — not aligner collection, not scrambling.

### dinky's Reward by Agent Count
- 2 agents: 11.9 avg (range 1.1-42.3)
- 4 agents: 18.0 avg (range 1.6-46.8)
- 6 agents: 27.8 avg (range 11.8-43.8)

### Head-to-Head (Same Episode)
- Softy 6 agents → 36 aligned vs dinky 2 agents → 58 aligned
- Softy 4 agents → 16 aligned vs dinky 4 agents → 50 aligned
- **dinky achieves 3-5x more junction alignment per agent than Softy in the same episode**

### dinky's Pattern: More Agents → Less Scrambling
- 2 agents: scrambler/aligner ratio = 0.34
- 4 agents: ratio = 0.27
- 6 agents: ratio = 0.15
- With more agents, dinky focuses MORE on alignment and LESS on scrambling

## v24 vs v29 vs v28 — Why Rank Changed

| Version | Rank | Score | c1% | c1 reward | Key change |
|---------|------|-------|-----|-----------|------------|
| v24 | 2 | 25.60 | 95% | 41.0 | timeout 50, radius 20 |
| v29 | 5 | 24.43 | 60% | **41.5** | +P139 (always hub) |
| v28 | 34 | 18.47 | **0%** | N/A | radius 15, timeout 60 |

**v29 is SLIGHTLY BETTER than v24 at c1** (41.5 vs 41.0). The rank difference is matchup distribution (v29 got more multi-agent matches which score lower), NOT code quality.

**v28 crashed** because: (1) radius 15 is wrong, (2) timeout 60 hurts, (3) got zero c1 matches, (4) harder opponents.

## How to Reproduce This Analysis

```bash
# Get policy UUID
uv run cogames leaderboard beta-cvc --json | python3 -c "
import json, sys
for r in json.loads(sys.stdin.read()):
    p = r['policy']
    if r['score'] > 20:
        print(f\"{p['name']}:v{p['version']} id={p['id']} score={r['score']:.1f}\")
"

# Episode analysis for any policy
uv run cogames episode list --policy <UUID> --json | python3 -c "
import json, sys
data = json.loads(sys.stdin.read())
for ep in data:
    for pr in ep.get('policy_results', []):
        if pr['policy']['name'] == 'TARGET':
            n = pr['num_agents']
            m = pr.get('avg_metrics', {})
            print(f'agents={n} reward={pr[\"avg_reward\"]:.1f} aligned={m.get(\"junction.aligned_by_agent\",0):.0f} scrambled={m.get(\"scrambler.gained\",0):.0f} deaths={m.get(\"death\",0):.0f} hp.gained={m.get(\"hp.gained\",0):.0f} heart.gained={m.get(\"heart.gained\",0):.0f}')
"
```
