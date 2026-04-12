# Director Notes
_Written: 2026-04-12 (Session 6 — offline→online bridge)_

## What changed since session 5

Session 5 celebrated PR #18 merging and a +37% offline 3-agent reward (0.51 → 0.71/agent). This session I connected to the online tournament for the first time. The picture is much worse than offline suggests.

## Offline observations (carry-over)

- Best offline: 0.77/agent (cross_role v15, 2A1M, 3 agents, 1000 steps, commit `97886dd`).
- Post-PR#18 main: 0.71/agent (cross_role, 3 agents, 1000 steps).
- 8-agent stuck: 0.42/agent (4A4M scripted miners). LLM miners slightly worse (0.40).
- Open offline issues: #25, #24, #12, plus external advice #26/#27.

## Online observations

Leaderboard `beta-cvc` pulled 2026-04-12:
- 322 policies on leaderboard; **top is `dinky:v27` at 27.90**, then `slanky:v129` at 23.92.
- Our best entry is **rank 265** (`lessandro-fast-llm-v1:v1`, 3.415). Our other two qualified entries are rank 271 and 279.
- That's an **8× gap** from the top. Median is ~16.

**Uploaded policies vs qualified**: 12 uploaded, only 3 reached competition. The others:
- `lessandro-crossrole-v1:v1` — **0/4 qualifying (BackoffLimitExceeded)** — this is our 0.71/agent offline champion!
- `lessandro-codex-nemotron-27-march:v1` — 0/4
- `lessandro-machina-retry-v1:v1` — 0/4
- `lessandro-scripted-v1:v1` / v2 — 0/4 each
- `lessandro-scripted-v3:v1` — 2 pass, 1 fail
- The 3 passing ones all happen to be LLM variants that don't use cross_role.

**Match-level distribution** (107 matches, 72 competition + 35 qualifying):
- Score ranges **0.00 → 11.81**, heavily bimodal.
- Zero-score matches: partnered with `anoop_agi.absolute_garbage_intelligence_*` (0.00/0.00)
- 8+ score matches: partnered with `slanky` (7.56 avg over 3) or `scissors_v1_v*` (8+ several times).
- Our three qualified policies have avg 3.01 / 3.09 / 3.39 — nearly indistinguishable. **The partner, not our policy, decides the score.**

**Replay analysis** (3 replays downloaded: our best, our worst, top-1 dinky match):

BEST — `lessandro-machina-paid-v1:v1 + slanky`, score 11.81, 10000 steps:
- `cogs/aligned.junction.held` = 118057 (avg 11.8 junctions per tick)
- `cogs/aligned.junction.gained` = 66 (captures)
- `cogs/*.deposited` ≈ 3500 each (balanced mining)
- **cogs/heart.withdrawn = 5** (only the hub's initial 5 hearts — we never crafted)
- Our 2 agents steps=8944/8963, slanky agents steps=1624–5816 (slanky dying early)

WORST — `lessandro-machina-paid-v1:v1` 8-agent self-play, score 0.11:
- `cogs/aligned.junction.gained` = 1 (ONE junction captured in 10k steps)
- `cogs/carbon.deposited` = 1, silicon=1, oxygen=4, germanium=3 (NO mining at all)
- Per-agent `action.move.failed` = 1200 avg (worst 2520 — agent pinned against wall)
- `cell.unique_visited` = 467 (out of ~5000 → 9% coverage)
- `cell.max_distance_from_spawn` = 42 (agents stay in spawn quadrant)
- `miner.amount` = 0.0, `aligner.amount` = 0.125 (1 in 8 ever wore aligner gear)
- `death` = 2/agent on average

DINKY (top-1) — `dinky:v27 + anoop_agi`, score 1.14 in this match:
- `cogs/aligned.junction.gained` = 9, held = 11439
- All 8 agents: **vibes = 0** (see below)
- dinky's leaderboard rank comes from **118 match-matches**, not a single brilliant episode.

**Critical instrumentation finding**: across all 24 sampled agents in 3 replays, **zero `change_vibe_*` actions** were ever emitted. Yet the env clearly has role state (aligner.amount, miner.amount are non-zero for slanky). Either (a) the replay `action_id` field strips vibe transitions, or (b) role state is set by walking into gear stations (not by the `change_vibe_*` action). Big implication: our offline cross_role "switching" experiments may have been measuring the wrong thing.

## Offline → online gap

1. **Offline best**: 0.71/agent @ 3-agent cross_role post-PR#18 (commit post-`801a0a2`).
2. **Online best**: rank 265/322, score 3.415 with `lessandro-fast-llm-v1:v1`. Our post-PR#18 cross_role policy is NOT on the leaderboard at all — it crashes qualifying.
3. **Gap is NOT closing** because the offline winner cannot enter the tournament. Session 5's +37% offline improvement was invisible online.
4. **Gap causes (quantified)**:
   - *Submission blocker*: crossrole-v1:v1 fails 4/4 qualifying (BackoffLimitExceeded on 8-agent self-play). 6 of our 12 policies fail this way.
   - *Horizon mismatch*: offline eval is 1k steps, online is 10k. Online score is cumulative junction-held, not peak capture. We never test past 1k so we don't know what happens after step 1000.
   - *Format mismatch*: qualifying is 8-agent self-play (assignments=[0,0,0,0,0,0,0,0]). We never test our policy alone at 8-agent scale offline.
   - *Partner variance dominates our signal*: avg 3.0 hides a 0.0 ↔ 11.8 distribution driven almost entirely by partner quality.
   - *Role-assignment silently broken*: 0/24 agents ever emit change_vibe_* in replays, suggesting our "dynamic role" work may not be executing online.
5. **Bottleneck is the GAP itself**, not offline or online alone. Offline reward is improving (session 5 +37%); that improvement just isn't landing on the leaderboard.

## Issues created this session (all priority:1)

- **#28** [AUTORESEARCH][ONLINE] Fix qualifying BackoffLimitExceeded — unblocks submission of our best offline work. Single highest-leverage action.
- **#29** [AUTORESEARCH][ONLINE] 10k-step eval + held-per-tick metric — make offline reward correlate with online score.
- **#30** [AUTORESEARCH][ONLINE] 8-agent self-play collapse (1200 failed moves/agent, 0 mining) — fix three specific failure modes in the worst replay.
- **#31** [AUTORESEARCH][ONLINE] Zero `change_vibe_*` actions observed — instrumentation / root-cause investigation for role assignment.
- **#32** [AUTORESEARCH][ONLINE] Partner robustness (0.00 → 11.81 score swing) — build a "carries-bad-partners" policy.

Existing offline issues (#25, #24, #12, #17, etc.) remain at priority:2 — they are still correct work, but the online issues unblock them or change their evaluation target, so they should run after #28/#29 land.

## Current bottleneck

**The submission pipeline**, not the policy. Our research leaderboard lists a 0.71/agent cross_role winner that the tournament has never seen. Fix #28 first, then re-evaluate everything else against a 10k-step, held-per-tick benchmark (#29).

## Open questions for next director

1. After #28 lands, does `lessandro-crossrole-v2` (re-uploaded with `scripted_miners=true`) actually score ≥ 5 on the leaderboard, or does the 10k-step horizon kill it for reasons we can't see from 1k offline?
2. Do any other top-10 policies (slanky family, dinky) emit `change_vibe_*` in their replays? If NO → vibe isn't action-mediated and we've misread issue #9 results for 4 sessions. If YES → our policy is uniquely broken in role emission.
3. Given partner variance (σ ≈ 3.3), how many matches do we need to rank-order policies with statistical significance? Current 20-match samples may be too noisy.
4. When held_per_tick becomes the primary metric, does session 5's v15 config still win, or does the make_heart cycle fall apart past 1000 steps?
5. Is there a Matchups API endpoint that tells us WHICH partners we've been paired with most often? Our 20 samples include heavy overlap with `scissors_v1_v*` — we may be overfitting to that partner family.

## Next session preparation

Before running, check:
- `gh issue list --label priority:1` — any of #28–#32 moved to closed or in progress?
- `GET /tournament/seasons/beta-cvc/leaderboard` — did our rank move above 200?
- Are there new uploaded policies (`crossrole-v2`, `carry-v1`) in competition?
- Does any of our policies now have > 0 `change_vibe_*` actions in a fresh replay?
