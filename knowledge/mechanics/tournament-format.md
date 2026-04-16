# Tournament Format

Source: Episode analysis of 100+ matches, `cogames season show`, `cogames matches`

## TWO TOURNAMENTS — Both Matter

### beta-teams-tiny-fixed (THE ranked tournament)
| Parameter | Value |
|-----------|-------|
| Format | Multi-stage elimination → team rounds → policy scoring |
| Score metric | **Lower = better** (sum of top 4 team placements) |
| Runs | Daily (new version each day ~17:00) |
| Stages | 7: entry → stage-1/2/3 → sample → team-round-1/2/3 |

#### Stage Structure (verified from 2533-match tournament)
| Stage | Policies | Agents/policy | Matches | Softy agent count |
|-------|----------|--------------|---------|------------------|
| stage-1 | 1 (solo) | 8 | 51 | c8 |
| stage-2 | 2 | 4 | 240 | c4 |
| stage-3 | 4 | 2 | 2058 | c2 |
| team-round-1 | 8+ | 1-2 | 120 | c1-c2 |
| team-round-2 | 8+ | 1-2 | 60 | c1-c2 |

**stage-3 has 80% of all matches** — c2 performance is the most important factor.

### beta-cvc (freeplay)
| Parameter | Value |
|-----------|-------|
| Policies per match | **2** |
| Agent splits | 6-2 (70%) or 4-4 (30%) |
| Format | Qualify via self-play, then 20 matches vs random partners |
| Score metric | **Higher = better** (mean reward) |
| Softy usually gets | 1 agent (96% of matches) |

## Cooperative, NOT Competitive

**All policies are on the SAME TEAM.** Evidence:
1. All agents are `agent.red.*` (no blue team)
2. Both policies receive IDENTICAL reward in every episode (verified across 100+ episodes, zero exceptions)
3. Both share hub, heart economy, junction network
4. Game stats show single `cogs/` metric pool for all 8 agents

## Reward Formula (verified, 0.00 error)

```
reward = cogs_aligned_junction_held / 10000
```

| Episode | cogs.junction.held | Calculated | Observed | Error |
|---------|-------------------|------------|----------|-------|
| 770a878e | 402,693 | 40.27 | 40.27 | 0.00 |
| 81101bd6 | 431,877 | 43.19 | 43.19 | 0.00 |
| da8aa4a8 | 480,960 | 48.10 | 48.10 | 0.00 |
| fe6223ed | 505,600 | 50.56 | 50.56 | 0.00 |

Theoretical max: 68.0 (all 69 junctions held from tick 0). Best observed: 50.56 (74.4%).

## Leaderboard Score

Score = mean(avg_reward) across all matches. High variance (stddev ~12, or 47% of mean).

## Softy's Agent Count Distribution

| Version | Rank | c1 | c2 | c3 | c4 | c6 |
|---------|------|----|----|----|----|----|
| v24 | 2 | 95% | 5% | — | — | — |
| v29 | 5 | 60% | 25% | 15% | — | — |
| v28 | 34 | 0% | 35% | — | 30% | 35% |

**Softy almost always gets 1 agent.** Competitor policies (dinky, slinky) get 2-6 agents.

## beta-teams-tiny-fixed: v33 Cross-Policy Data (20 episodes, stage-3)

| Policy | Eps | Reward | Aligned/agent | Deaths/agent | Scrambled | Hearts |
|--------|-----|--------|---------------|-------------|-----------|--------|
| slanky:v152 | 6 | 32.6 | **62** | 10 | 5 | 93 |
| dinky:v28 | 4 | 30.2 | 58 | 11 | 4 | 64 |
| **Softy:v33** | **20** | **29.6** | **54** | **15** | **0** | **60** |
| Softy:v29 | 6 | 27.8 | 52 | 15 | 0 | 59 |
| slinky:v2 (#1) | 7 | 28.2 | 39 | **6** | 4 | 67 |
| slanky:v145 | 10 | 31.3 | 20 | **5** | 2 | 63 |

**Key finding**: Top-ranked policies die 3x LESS (4-6 deaths vs Softy's 15). Death reduction is the #1 improvement target.

Deaths-reward correlation: **-0.43** (moderate — fewer deaths = higher team reward).

## Strategic Implications

1. **Softy doesn't need its own miners at c2** — 6 teammate agents mine
2. **Softy doesn't need its own scramblers at c2** — teammates scramble
3. **Reward is team-based** — individual per-agent metrics don't directly affect personal reward
4. **Death reduction is the biggest gap** — Softy 15 deaths/agent vs competitors 4-6
5. **Local benchmarks are misleading** — `cogames pickup` runs self-play (all Softy), tournament runs cooperative
6. **Both tournaments matter** — optimize for c2 (beta-teams-tiny-fixed) AND c1 (beta-cvc)

## How to Check Match Data

```bash
# Agent count distribution for any policy
uv run cogames episode list --policy <UUID> --json | python3 -c "
import json, sys
data = json.loads(sys.stdin.read())
counts = {}
for ep in data:
    for pr in ep.get('policy_results', []):
        if pr['policy']['name'] == 'TARGET':
            counts[pr['num_agents']] = counts.get(pr['num_agents'], 0) + 1
print(dict(sorted(counts.items())))
"

# Verify reward formula
uv run cogames episode list --policy <UUID> --json | python3 -c "
import json, sys
for ep in json.loads(sys.stdin.read()):
    gs = ep.get('game_stats', {})
    held = gs.get('cogs/aligned.junction.held', 0)
    calculated = held / 10000
    observed = ep['policy_results'][0]['avg_reward']
    print(f'held={held:.0f} calc={calculated:.2f} obs={observed:.2f} err={abs(calculated-observed):.2f}')
"
```
