# Tournament Audit — April 14, 2026

## CRITICAL: Tournament Only Runs c2, c4, c6, c8
NO c1/c3/c5/c7. Distribution: c2≈31%, c4≈27%, c6≈33%, c8≈9%.
Local VOR count-weights c2 at 5.6% — tournament weights it 5.5x more.

## Scenario Breakdown (versions with 10+ matches)
| Version | c2 avg (N) | c4 avg (N) | c6 avg (N) | c8 avg (N) | Overall |
|---------|-----------|-----------|-----------|-----------|---------|
| v21     | 5.1 (7)   | 21.5 (6)  | 30.9 (7)  | 35.1 (2)  | 23.56   |
| v20     | 2.9 (7)   | 16.2 (6)  | 25.7 (7)  | 22.8 (2)  | 18.88   |
| v19     | 2.9 (6)   | 12.0 (6)  | 21.9 (8)  | 24.3 (2)  | 16.38   |
| v18     | 6.7 (7)   | 13.0 (6)  | 21.1 (7)  | 28.4 (2)  | 16.14   |
| v17     | 4.1 (7)   | 7.4 (6)   | 18.5 (7)  | 21.8 (2)  | 12.67   |

## Scenario Contribution (v21)
- c2 (31% of matches): avg 5.1 — OUR WEAKEST. 2 miners, 0 aligners under ROLE_CYCLE.
- c4 (27% of matches): avg 21.5 — Good. 3M+1A under ROLE_CYCLE.
- c6 (33% of matches): avg 30.9 — Strong. 3M+3A.
- c8 (9% of matches): avg 35.1 — Strongest (but only 2 matches).

## Per-Scenario Progression (avg score)
- c2: v17=4.1 → v18=6.7(+63%) → v19=2.9(-58%) → v20=2.9(+0%) → v21=5.1(+78%)
- c4: v17=7.4 → v18=13.0(+76%) → v19=12.0(-7%) → v20=16.2(+35%) → v21=21.5(+32%)
- c6: v17=18.5 → v18=21.1(+14%) → v19=21.9(+4%) → v20=25.7(+17%) → v21=30.9(+20%)
- c8: v17=21.8 → v18=28.4(+30%) → v19=24.3(-14%) → v20=22.8(-7%) → v21=35.1(+54%)

## Regressions Detected
1. **v18→v19 c2**: 6.7→2.9 (-58%). v19 = v18 + P73 (sector collision fix). May have changed exploration for 2-agent teams. But only 7 matches — could be variance.
2. **v19→v20 c8**: 24.3→22.8 (-7%). v20 = v19 + P76 (BFS wander). Small sample (2 matches).
3. **v20→v21 c2 recovered**: 2.9→5.1 (+78%). v21 = v20 + P89 (radius+timeout). Longer timeout helps even 2-agent teams.

## Translation Ratios (local → tournament)
- v17→v18 (P65 timeout 25→35): local +8.1% → tournament c4 +76%, c6 +14%. **Huge c4 translation.**
- v18→v19 (P73 sector fix): local +14% → tournament c6 +4%, BUT c2 -58%. **c2 regression masked by overall gain.**
- v19→v20 (P76 BFS wander): local +7% → tournament c4 +35%, c6 +17%. **Strong translation in mid-scenarios.**
- v20→v21 (P89 radius+timeout): local +4% → tournament c4 +32%, c6 +20%, c8 +54%. **Best translation ever.**

## Scenario-Specific Opportunities
1. **c2 (HIGHEST PRIORITY)**: avg 5.1, 31% of matches. With ROLE_CYCLE, both agents are miners → zero alignment. Hub starts with 5 hearts that go UNUSED. P102 gives 1M+1A. Local c2 showed +82%. If tournament c2 reaches 8-10, that's +1.0-1.5 overall VOR → could reach 24.5-25.0 (pass slanky).
2. **c4**: avg 21.5, 27% of matches. 3M+1A = only 1 aligner. A 2M+2A split could help but risk is high (P40 showed 4M = catastrophic). Don't touch yet.
3. **c8**: avg 35.1, 9% of matches. Strong but small sample. Not a priority.

## Rules Update
- **Local VOR threshold must account for tournament weighting**: A change that improves c2 by 50% shows as only +2.8% local (c2 weight 5.6%) but would be +15.5% tournament (c2 weight 31%).
- **P102 should be uploaded despite local +2.4%**: It's a capability improvement (enables alignment for c2), and c2 is our weakest tournament scenario at 31% weight.
- **All future hypotheses should estimate tournament-weighted impact** alongside local VOR.
