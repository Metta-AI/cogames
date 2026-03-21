# Autoresearch 21 March

This file backfills the experiment narrative for branch `autoresearch_21_march`.
The existing git history and TSV log captured code changes and some results, but they did not consistently record the experiment thesis and findings in commit messages.

## Baseline

- Baseline code reference: `ea1802b`
- Mission: `cogsguard_machina_1`
- Command shape: `uv run cogames play -m cogsguard_machina_1 -c 3 -p class=machina_llm_roles,kw.llm_timeout_s=10 -s 80`
- Result: reward `0.040800`
- Secondary signal: `aligned.junction.held=56`, `germanium.deposited=80`

Finding:
The starting point already achieved non-zero alignment and deposit reward, so improvements needed to beat a functioning baseline rather than recover from a broken policy.

## Experiment 1

- Code commit: `c7605e0`
- Thesis: improve aligner navigation by using map memory and frontier-following instead of falling back to wandering too early.
- Changes:
  - added frontier-based exploration in the aligner policy
  - moved toward frontier cells when the final target was not yet reachable through mapped free cells
  - biased some exploration toward areas near known hubs
  - added tests for remembered-map exploration behavior
- Expected outcome:
  - aligners should spend less time wandering blindly
  - aligners should discover useful routes and targets faster

Finding:
This introduced the first serious navigation upgrade, but there is no corresponding TSV row for a clean run from exactly `c7605e0`. The experiment thesis is visible in the code, but the run/result logging was missing.

## Experiment 2

- Code commit: `48b6985`
- Thesis: make the LLM aligner planner more state-aware and reduce bad skill choices after gear or target discovery.
- Changes:
  - prompt now included `known_hubs` and `known_alignable_junctions`
  - planner override forced `get_heart` when aligner gear was equipped and a hub was already known
  - planner override forced `align_neutral` when a valid alignable target was already known
  - stuck tracking no longer penalized standing on a valid target tile
  - added tests for planner override behavior
- Expected outcome:
  - fewer wasted `explore` calls
  - faster transition into heart collection and neutral alignment

Runs recorded in TSV:
- discard, reward `0.024000`: "2-aligner default for 3-cog machina_llm_roles plus planner crash fallback"
- discard, reward `0.024000`: "Offline planner fallback with 1 aligner 2 miners and miner hub-search override"

Finding:
This direction underperformed the baseline badly. In practice the planner/fallback behavior did not improve team reward, and one configuration collapsed alignment entirely. The repeated low reward suggests that planner refinements alone were not enough when execution and routing were still weak.

## Experiment 3

- Base commit before dirty worktree run: `863939c`
- Final committed form of the same work: `e81c8a9`
- Thesis: improve scripted execution without relying on the online planner by preserving static world knowledge and making routing more map-aware.
- Changes:
  - hubs, stations, and extractors were treated as static remembered objects instead of being forgotten when out of sight
  - miner and aligner movement toward visible targets was converted to absolute-coordinate, map-aware routing
  - aligner exploration was biased toward frontier that could expand the aligned network
  - aligner `explore` behavior became conditional on whether the cog had a heart or knew hubs
  - added an alignment-frontier test
- Expected outcome:
  - better recovery when the planner is offline or unreachable
  - less rediscovery cost for static objectives
  - more useful expansion around the current aligned network

Run recorded in TSV:
- discard, reward `0.037800`: OpenRouter was unreachable, so the policy effectively ran in scripted fallback mode

Finding:
This was better than the failed `48b6985` runs and recovered meaningful alignment progress (`aligned.junction.held=46`), but it still did not beat the baseline `0.040800`. The result suggests the static-memory and frontier-bias ideas were directionally useful, especially for offline fallback, but not yet enough to produce a net reward improvement.

## Summary

- Best recorded run remained the baseline at `0.040800`
- Planner-prompt and override tuning alone did not help and may have hurt
- Scripted navigation and memory improvements partially recovered performance
- The strongest signal from this branch is that execution-layer improvements mattered more than LLM prompt changes

## Logging Gaps

This branch had several process issues:

- `c7605e0` has no corresponding TSV result row
- `48b6985` has two result rows, meaning multiple runs were logged against the same code commit
- the run described against `863939c` was actually performed on a dirty worktree and only later committed as `e81c8a9`
- commit messages did not consistently record thesis and findings in the format requested by `autoresearch.md`

This file exists to preserve that missing narrative for the branch.
