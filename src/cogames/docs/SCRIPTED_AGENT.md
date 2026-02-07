# Scripted Agent Policies

`cogames` ships a small set of teaching-friendly scripted policies intended for tutorials and debugging.

## Included Policies

### 1. StarterPolicy

Location: `cogames.policy.starter_agent`

Short name: `starter`

This is a simple multi-role heuristic policy for the CogsGuard arena. It tries to acquire gear and then follows a
minimal objective loop (e.g., miners go to extractors; scouts wander).

### 2. Role Policies (Fixed Gear Preference)

Location: `cogames.policy.role_policies`

Short names:

- `role_miner`
- `role_scout`
- `role_aligner`
- `role_scrambler`

These are thin wrappers around the starter logic that try to acquire a specific gear type first.

### 3. TutorialPolicy (Trainable Reference)

Location: `cogames.policy.tutorial_policy`

Short name: `tutorial`

A simple CNN + LSTM policy used in CoGames tutorials.

## Usage

```bash
# Run a scripted policy by short name
uv run cogames play --mission evals.diagnostic_radial -p starter --cogs 4

# Force a fixed role
uv run cogames play --mission evals.diagnostic_radial -p role_miner --cogs 4
```

## More Advanced Agents

The repository also contains richer scripted and Nim-compiled agent policies under `packages/cogames-agents/`, but that
package is not published to PyPI.
