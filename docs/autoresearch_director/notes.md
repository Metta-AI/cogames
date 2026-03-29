# Director Notes
_Written: 2026-03-28_

## What I observed
No replay available (no API key on this machine). Analysis based on TSV data, issue comments, and git history from two completed research sessions (#9: 18 variants, #12: 19 variants).

## Current bottleneck
**Skill timeout waste.** The 0.92 baseline (7321afc) has 24 align_neutral + 26 get_heart timeouts = ~2500 wasted agent-steps out of ~8000 total (31%). This is the single highest-leverage fix because:
1. Issue #9 v9 proved timeout extension works: 100→400 steps cut failures from 45-64% to <5%
2. The fix is independent of gear acquisition (no gear switching needed in fixed-role baseline)
3. The second bottleneck (5 hearts from hub, mining doesn't help) is a hard ceiling that needs architectural changes

## What I expected to happen vs. what I found
This is the first director session, so no prior expectations. Key surprise: PR #13 (gear fix from #12) improved 300-step runs but regressed 1000-step runs — short-horizon A/B testing is insufficient for this environment.

## Issues updated this session
- **#9 (Cross-Role Policy):** CLOSED. 18 variants, best 0.55 vs baseline 0.66. Cherry-pickable findings documented.
- **#12 (Gear Acquisition):** Deprioritized. 19 variants, PR reverted. Gear switching not needed for fixed-role baseline.
- **#10 (Role Tuning):** Reprioritized to top. Added 4 concrete experiments cherry-picked from #9/#12 learnings.
- **#15 (8-Agent Scaling):** NEW. Created to track the 3→8 agent scaling problem separately. Blocked by #10.

## Research roadmap after this session
```
#10 Role Tuning (priority:1, NEXT)
  └─ blocks #15 8-Agent Scaling
#11 Active Inference (independent, priority:2)
#12 Gear Acquisition (deprioritized, revisit if #10 hits gear-related ceiling)
#9  Cross-Role Policy (CLOSED)
```

## Open questions for next director
1. Does the timeout extension from #9 v9 transfer cleanly to the fixed-role baseline? The code paths differ.
2. Is the 5-heart ceiling breakable? Mining deposits resources but doesn't create hearts. Is there a game mechanic to craft hearts from deposited resources?
3. The March 21 normalization gave 2.260 reward vs March 22's 0.92 for similar behavior. Which normalization is the competition metric? This affects how we report progress.
4. Move-failure tracking (+42% held-steps in March 22) — is this already in main or does it need cherry-picking?
