greeting agent! you will be given a GitHub issue number. your task is to improve the reward score on cogames by improving our LLM policies, following the direction in that issue.

# Setup

You will be invoked with an issue number, e.g. `autoresearch.md 10`.

1. Fetch the issue: `gh issue view <N> --repo SolbiatiAlessandro/cogames`
2. Read the issue title and body — that is your research direction for this session
3. Create and checkout a branch named after the issue: `git checkout -b autoresearch/issue-<N>-<slug>` where slug is a short kebab-case summary of the issue title
4. All experiment results and discussion go as **comments on the issue** via `gh issue comment <N> --repo SolbiatiAlessandro/cogames --body "..."` — do NOT write a separate discussion doc

# Experimentation

We are iterating on a Voyager (Minecraft LLM Agents) like solution to cogames.
We have a set of hardocded skills like mine until full, go back to home
And in the game we have prompt fot a lightweight LLM we call through open router that uses those skills. For now the LLM doesn't write code.
Current implementation details that matter:

- The LLM is used as a planner over bounded scripted skills, not as a per-tick action model.
- The scripted skills are the execution layer and the main place where reward improvements currently come from.
- The current OpenRouter model is `nvidia/llama-3.3-nemotron-super-49b-v1.5`.
- The current local OpenRouter key is expected to be loaded from `.env.openrouter.local` and must never be committed.
- We already know from experiments that:
  - the LLM planner is often reasonable
  - the bigger failures are usually in skill execution, navigation, or state summarization
  - logs are extremely useful because they let us inspect the actual decision trajectory

Here is what you can do to improve game reward

1. add new skills with hardwritten python code
2. change the prompt of the LLMs

Your first task is to  make in docs/ a file called <branch_name>.md
That file is your experiment report notebook, add logs
WRITE TO <branch_name>.md: "<timestamp>: autoresearch starting, my plan is to..."

WRITE TO <branch_name>.md: "<timestamp>: starting to run baseline"
Your second setup task run should be running baseline , so you will run the policy as is
WRITE TO <branch_name>.md: "<timestamp>: baseline result is "

# Reward target

**Issue-specific override (check first):**
Before defaulting to mission reward, read the issue body. If the issue defines its own success criteria or custom metrics (e.g. `initial_gear_success_rate`, `gear_change_success_rate`, or any explicit numeric thresholds), use those as your primary optimization target instead of mission reward. Log the issue-defined metrics in the TSV `secondary_rewards` column and track them as your main signal.

Primary objective (default — use when issue defines no custom metrics):

- maximize the mission reward reported by the game

Secondary objectives:

- maximize `aligned.junction.held`
  - `cogs/carbon.deposited`
  - `cogs/oxygen.deposited`
  - `cogs/germanium.deposited`
  - `cogs/silicon.deposited`


# Output and logging

once you have run your training

When an experiment is done, log it to docs/results_<branch_name>.tsv (tab-separated, NOT comma-separated — commas break in descriptions).
Log also in docs/<branch>.md , that is your experiment logs! write your findigns and learnings so next researcher can also follow your direction. If you don't will be hard to figure out what you did later

The TSV has a header row and 5 columns:

commit	mission_reward secondary_rewards steps		status	description
git commit hash (short, 7 chars)
mission_reward reward achieved in the episode (e.g. 3.5) — use 0.000000 for crashes
secondary_reward
how many steps in the episode
status: keep, discard, or crash
short text description of what this experiment tried


# The experiment loop
The experiment runs on the branch created in Setup above.
That branch must also exist on GitHub so progress is backed up and visible remotely.

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune skills and LLM policy with experimental idea by directly hacking the code, brainstorm what you should try, read other experiments in /docs and other previous commit,
WRITE TO <branch_name>.md: "<timestamp>: starting new experiment loop, in this experiment I want to try.. my hypothesis is.."
3. git commit in the form "[EXPERIMENT=...][EXPERIMENT_START] wrote code.." and push to github
5. Run the experiment
6. Read out the results
7. Record the results in the tsv and commit the results
8. WRITE TO <branch_name>.md: "<timestamp>: I run my experiment, I found out that.. this is a good/bad result because.. next experiment next agent should probably try.."
9. Post a comment on the issue with the results and interpretation: `gh issue comment <N> --repo SolbiatiAlessandro/cogames --body "..."`
10. git commit in the form "[EXPERIMENT=...][EXPERIMENT_RESULTS] added ... reard to TSV" and push to github
11. Push the branch to GitHub. If the branch does not exist remotely yet, create it with `git push -u origin <branch_name>`. After that, use `git push` after each kept result so the remote branch stays current.
12. If reward improved you "advance" the branch, keeping the git commit
13. If reward is equal or worse, you git reset back to where you started. If you had already pushed a discarded commit, bring the remote branch back in sync too.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

Crashes: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

WATCHING THE GAME (optional): If you want to see what the policy is actually doing spatially — agents moving, gear stations, junctions — run the `/cogames-watch-replay` skill. It captures emoji map frames every N steps and shows you where agents are across the episode. Useful when a metric looks wrong and you want to understand why. Example: `/cogames-watch-replay --steps 500 --every 50`.

NEVER STOP: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.
