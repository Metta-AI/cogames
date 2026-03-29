---
name: cogames-issues
description: View, create, and prioritize CoGames autoresearch issues. Use this to check the current research stack, create new experiment issues, or reprioritize after a director session.
argument-hint: "[list | create | reprioritize]"
---

# CoGames Issue Management

All issues live in `SolbiatiAlessandro/cogames`. The label system is how the three-tier research system communicates:

- **Director** writes labels after synthesizing experiment results
- **OpenClaw** reads labels to decide which autoresearcher to spawn next
- **Autoresearcher** reads the issue body for experiment instructions, posts results as comments

## Label Scheme

Every open research issue MUST have exactly one priority label and optionally one status label.

| Label | Meaning | Color |
|-------|---------|-------|
| `priority:1` | Spawn next — highest leverage | red |
| `priority:2` | Second tier — ready but not urgent | pink |
| `priority:3` | Deprioritized — revisit later | light pink |
| `blocked` | Has unresolved dependencies, do NOT spawn | black |
| `in-progress` | Researcher currently active on this issue | green |

## How OpenClaw Reads the Stack

OpenClaw picks the next issue to work on with this logic:

```bash
# 1. Find highest-priority unblocked issue
gh issue list --repo SolbiatiAlessandro/cogames --label "priority:1" --state open --json number,title,labels

# 2. Skip any that also have "blocked" or "in-progress"
# 3. Spawn autoresearcher on the first remaining issue
# 4. Add "in-progress" label to that issue
gh issue edit N --repo SolbiatiAlessandro/cogames --add-label "in-progress"

# 5. When researcher finishes, remove "in-progress"
gh issue edit N --repo SolbiatiAlessandro/cogames --remove-label "in-progress"
```

If no `priority:1` issues are available, fall back to `priority:2`, then `priority:3`.

## How the Director Writes Labels

After synthesizing results and watching replays, the director updates labels to reflect the new research roadmap.

### Reprioritize an existing issue

```bash
# Promote to priority:1
gh issue edit N --repo SolbiatiAlessandro/cogames \
  --remove-label "priority:2" --add-label "priority:1"

# Demote to priority:3
gh issue edit N --repo SolbiatiAlessandro/cogames \
  --remove-label "priority:1" --add-label "priority:3"

# Mark as blocked (dependency not resolved)
gh issue edit N --repo SolbiatiAlessandro/cogames --add-label "blocked"

# Unblock (dependency resolved)
gh issue edit N --repo SolbiatiAlessandro/cogames --remove-label "blocked"
```

Always leave an audit comment explaining WHY:

```bash
gh issue comment N --repo SolbiatiAlessandro/cogames \
  --body "**[DIRECTOR UPDATE <date>]** Promoted to priority:1 because <reason>"
```

### Create a new research issue

Every issue must have: hypothesis, metric, success criteria, dependencies, and suggested experiments.

```bash
gh issue create \
  --repo SolbiatiAlessandro/cogames \
  --title "[AUTORESEARCH][STATUS=notstarted] <Short hypothesis title>" \
  --label "priority:N" \
  --body "$(cat <<'EOF'
## Hypothesis
<one sentence: what we think will improve reward and why>

## Metric / Success Criteria
<what to measure — issue-specific metric overrides default mission reward>
<e.g.: mission_reward > 0.92 at 1000 steps, or get_heart_stale_exits < 10>

## Blocked by
<issue numbers this depends on, or "none">

## Background
<what evidence from prior experiments or replay observations motivates this>

## Suggested experiments
- [ ] Experiment A: <description>
- [ ] Experiment B: <description>
EOF
)"
```

If the issue is blocked, also add the `blocked` label:

```bash
gh issue edit N --repo SolbiatiAlessandro/cogames --add-label "blocked"
```

### Close a resolved or failed issue

```bash
gh issue close N --repo SolbiatiAlessandro/cogames \
  --comment "Resolved by <commit> / Failed: <reason> / Superseded by #M"
```

## Checking Current State

To see the full research stack:

```bash
# All open issues with labels
gh issue list --repo SolbiatiAlessandro/cogames --state open --json number,title,labels

# Just the spawn queue (what OpenClaw will pick next)
gh issue list --repo SolbiatiAlessandro/cogames --label "priority:1" --state open

# What's currently being worked on
gh issue list --repo SolbiatiAlessandro/cogames --label "in-progress" --state open

# What's blocked
gh issue list --repo SolbiatiAlessandro/cogames --label "blocked" --state open
```

## Rules

- An issue without a `priority:N` label is invisible to OpenClaw — always assign one
- Only the director changes priority labels (not researchers, not OpenClaw)
- Researchers post results as issue comments, never change labels
- `blocked` + `priority:1` = "important but not ready" — OpenClaw skips it
- Multiple `priority:1` issues are fine — OpenClaw picks the first one returned
- When a researcher finishes, it removes `in-progress` but does NOT change priority — that's the director's job
