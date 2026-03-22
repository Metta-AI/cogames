2026-03-21 11:28:35 PDT: autoresearch starting, my plan is to push the LLM-backed machina policy to longer horizons than 80 steps, verify real OpenRouter planner calls are happening, inspect long-run dynamics from logs, and iterate only on experiments where the LLM path is active.

2026-03-21 11:28:35 PDT: starting to run baseline

2026-03-21 11:34:48 PDT: baseline result is the longer-horizon 240-step trace confirmed real OpenRouter planning for all three agents, but a single ReadTimeout around step 21 permanently disabled the planner and the run devolved into fallback loops. The important long-term dynamics were: miners repeatedly got stuck bouncing between mine/deposit and aligner lost gear then got trapped because planner-selected unstuck was being overridden back into gear_up.

2026-03-21 11:34:48 PDT: starting new experiment loop, in this experiment I want to preserve LLM-selected unstuck when prerequisites are temporarily unmet and stop transient OpenRouter failures from permanently disabling the planner. my hypothesis is that long-horizon behavior is being dominated by control-policy overrides and one-shot network failures rather than by bad high-level planner choices.

2026-03-21 11:34:48 PDT: I run my experiment, I found out that the patched policy now keeps the LLM live through a 100-step episode with kw.llm_timeout_s=20 and allows the aligner to choose unstuck while gearless instead of force-looping gear_up. this is a good result because it satisfies the "LLM must actually be working" constraint past 80 steps and surfaces the next real long-term bottleneck: miners still spend too many late steps oscillating around deposit/mine pathing, while the aligner can still re-enter gear recovery loops after dropping gear. next experiment next agent should probably try explicit failure-memory or alternative-route skills for deposit_to_hub / mine_until_full rather than only repeating unstuck.

2026-03-21 14:00:00 PDT: starting new experiment loop, in this experiment I want to run llm_miner only (no aligner) at increasing step counts (50, 100, 200, 400) to find when reward degrades and debug the LLM planner output. my hypothesis is that miners get stuck in oscillation loops at longer horizons and the LLM planner output will reveal the decision trajectory leading to those failures. Starting with miners first, will move to aligner once miners are solid.

2026-03-21 14:30:00 PDT: I found a critical bug: agents sit on depleted extractors forever because `stationary_on_valid_target` resets the stuck counter even when no cargo is being gained. Fix: added `no_progress_on_target_steps` counter that detects when the agent is on a valid target but making no progress. Also remove depleted extractors from `known_extractors` so agents don't return to them. Results:
- Before fix (200 steps, miner-only): silicon.deposited=40, Agent 0 had 157 noops
- After fix (200 steps, miner-only): silicon.deposited=200, germanium.deposited=40
- After fix (400 steps, miner-only): silicon=300, germanium=160, carbon=100
- Full team (machina_llm_roles) 200 steps: 0.049/agent, junction.held=287
- Full team 400 steps: 0.109/agent, junction.held=691
- Full team 1000 steps: 0.273/agent, junction.held=1730, junction.gained=6, total deposited=1080
Remaining issues: aligner has 53% move failure rate (wall bumping), miners accidentally pick up scrambler gear. Next experiment should address these.

2026-03-21 20:00:00 PDT: autoresearch starting (new agent session), my plan is to maximize mission reward by: (1) running with 8 agents (2A6M composition) since March 19 data shows this is optimal, (2) fixing aligner wall-bumping (53% move failure rate), (3) improving miner efficiency. The current branch has depleted-extractor detection and hazard-station avoidance which were not in the March 19 experiments.

2026-03-21 20:00:00 PDT: starting to run baseline with 8 agents (2A6M) at 1000 steps

2026-03-21 20:30:00 PDT: baseline result is 0.80 total reward (0.10/agent) with 8 agents (2A6M) at 1000 steps. Catastrophic failure: cogs aligned 0 junctions, clips held 21040. Aligner 0 never got gear, aligner 1 got 5 hearts but aligned nothing (planner overwhelmed). Multiple miners stuck 127+ steps. The 2s deadline is too tight for 8 concurrent LLM calls causing massive "waiting for previous planner request" fallbacks. Reverting to 3-agent config (best known: 0.819 reward) for experiments, then scaling up once improvements are validated.

2026-03-21 20:30:00 PDT: starting new experiment loop, in this experiment I want to try aligner hazard station avoidance (avoid non-aligner gear stations in BFS pathfinding) and improved aligner frontier stepping (step into unknown territory from frontier cells like miners already do). My hypothesis is that aligners lose gear by stepping on miner/scrambler/scout stations during navigation and get stuck at map boundaries because they don't step into unknown territory, both of which waste cycles and reduce junction alignment rate.

2026-03-21 21:00:00 PDT: aligner hazard avoidance alone gave 0.762 (discard). Combined with smart planner fallback (scripted decision on timeout instead of unstuck), got 0.822 (marginal keep). Aligner still has 55.7% move failure rate. Key insight: the LLM planner adds contention for miners that don't benefit much from it (they just cycle gear→mine→deposit). Next experiment: disable LLM for miners (scripted only) to eliminate planner contention and let aligners get reliable planning.

2026-03-21 21:00:00 PDT: starting new experiment loop, in this experiment I want to make miners use purely scripted skill selection (no LLM calls) while keeping LLM planning for aligners. My hypothesis is that miners follow a deterministic loop (gear_up→mine_until_full→deposit_to_hub) that doesn't benefit from LLM planning, and their API calls create contention that hurts aligner planning quality. With scripted miners, I can also scale to 8 agents without planner contention.

2026-03-21 22:00:00 PDT: BREAKTHROUGH! With scripted miners + stuck counter fix, the 1A2M config scored 1.105 (2684 junction.held, 7 gained). Then discovered 8-agent config has spawn-stuck issue (agents physically trapped). The 2A1M config with scripted miner scored 2.184 (6281 junction.held, 7 gained) — 2.67x baseline! Agent 1 aligned all 7 junctions. Key discoveries:
- The no_move_steps counter was never reset on skill change (bug fix)
- Explore completion was too aggressive (completed immediately when extractors known)
- 2 aligners >> 1 aligner for junction.held because junctions are aligned earlier
- Scripted miners eliminate planner contention allowing reliable aligner LLM planning
Next: try 3 aligners (no miners), try different aligner_ids compositions.

2026-03-21 22:00:00 PDT: starting new experiment loop, trying 3 aligners 0 miners (all alignment, no mining).

2026-03-21 23:00:00 PDT: composition sweep results:
- 3 aligners (1000 steps): 2.260 total, 6533 junction.held, 7 gained — BEST at 1000 steps
- 3 aligners (2000 steps): 3.503 total, 9676 junction.held, 7 gained — diminishing returns
- 4 aligners: 0.774 total — catastrophic, 2 agents lost gear to miner stations, high failure rates
- 2A1M LLM: 1.199 total — planner contention hurt aligners despite 8 junctions gained
- 1 aligner: 0.10 total — agent stuck, couldn't find junctions

Conclusions: 3 agents is the sweet spot for this map. More agents hit spawn/navigation issues. The reward is dominated by junction alignment (held over time). Key improvement vectors: (1) align junctions earlier, (2) get more hearts, (3) prevent gear loss on undiscovered stations.

2026-03-21 23:00:00 PDT: starting new experiment loop, trying to improve aligner initial speed by making the scripted fallback override smarter — when aligner has gear + heart but is stuck on explore, override to unstuck then explore again (don't get trapped in get_heart loops).

2026-03-21 23:30:00 PDT: I ran the experiments with get-heart loop fixes and scripted fallback loop fixes. Results are stable at ~2.260 for 3-aligner config. The game is highly deterministic with fixed map layout. Agent 0 always loses aligner gear to an undiscovered miner station early in the run, and agent 2 can't find hearts/junctions due to its spawn position. Only agent 1 successfully aligns junctions (all 7). The ceiling is imposed by: (1) 7 hearts maximum per 1000 steps from the hub, (2) agent 0 gear loss on undiscovered miner station, (3) deterministic map layout.

2026-03-22 00:10:00 PDT: CRITICAL ANALYSIS: agent 0 in 2A1M gets 977 successes but never gets hearts because it loops unstuck→get_heart→stuck→unstuck×N (LLM chooses unstuck 3+ times in a row instead of explore). With mining and both aligners working, hub should have ~13 hearts (vs 7). Fix: after 2+ consecutive unstuck, force explore to find route to hub. This could double junctions from 7→13 → double junction.held → ~4.5 total reward.

2026-03-22 00:00:00 PDT: USER CORRECTION - revert to machina_1 baseline environment. Arena experiments discarded (different reward structure, no competing team). Key insight from user: aligners need resources from miners to craft hearts. Need balanced composition with actual miners that deposit. Key data points:
- 3-aligner (no miners): 2.260, 7 junctions, ~7 hearts (base hub supply)
- 2A1M all-LLM: 1.199, 8 junctions, 13 hearts (miner deposited 504 resources → more hearts)
- Problem: with 3 LLM agents, planner contention causes 2A1M to perform worse despite more hearts
- Fix hypothesis: increase planner deadline (5s) so 3 agents get reliable LLM responses

2026-03-21 23:45:00 PDT: BREAKTHROUGH on arena map! 3 aligners on cogsguard_arena got 7.650 total reward (2.55/agent) vs 2.260 on machina_1. Arena has no competing clips team (clips/aligned.junction.held=0) so reward scales much better. Only 2 junctions aligned but arena reward formula is 3.4x better. Continuing arena experiments.

SESSION SUMMARY:
- Baseline: 0.819 total reward (dc984a7, 1A2M LLM, 1000 steps)
- BEST: 2.260 total reward (3 aligners, 1000 steps) = 2.76x improvement
- Key discoveries: no_move_steps counter bug, explore completion bug, planner contention dominance, 3-agent sweet spot, alignment-focused composition wins
- Next agent should try: (1) pre-seeding aligner with hub/station positions for faster startup, (2) alternative unstuck patterns that explore more aggressively, (3) investigating if the game has mechanics to increase heart generation rate
