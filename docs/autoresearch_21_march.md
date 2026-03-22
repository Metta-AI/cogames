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
