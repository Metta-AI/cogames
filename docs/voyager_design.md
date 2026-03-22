# Voyager-Style Multi-Agent LLM Design

## Goal

Explore a CoGames agent architecture inspired by Voyager: agents use an LLM to plan, write reusable skills, execute those skills in-game, and revise strategy over time. The initial target is a 4-cog team with fixed role bias: 3 miners and 1 aligner.

## Proposed Shape

- Run one local LLM server on GPU and share it across four agent sessions.
- Give each cog its own context, memory, and current assignment.
- Maintain a shared skill repo and shared team memory/blackboard.
- Use the LLM sparsely for planning and code generation, not for every environment tick.
- Execute generated skills for bounded horizons, then return control for review or replanning.

## Why This Fits This Repo

- CoGames already supports multi-agent policy injection.
- Existing role-specialized policies suggest a clean insertion point for a new `voyager`-style policy layer.
- The environment already exposes structured observations and discrete actions, which makes skill execution easier to sandbox than raw UI control.

## Initial Architecture

1. **Strategist layer**
   Assigns roles and task priorities such as `mine carbon`, `return to hub`, or `capture nearest neutral junction`.

2. **Per-agent worker layer**
   Each cog has an LLM session that interprets its assignment, selects an existing skill, or writes/refines a new one.

3. **Shared skill repo**
   Stores reusable behavior snippets such as navigation, mining loops, deposit loops, and junction capture routines.

4. **Skill runtime**
   Runs generated skills in a sandbox with a limited API over CoGames actions and observations.

5. **Shared memory**
   Tracks hub needs, discovered resource locations, blocked paths, claimed tasks, and recent failures.

## Infra Assumption

Yes, this should run on GPU if we want a practical loop. Four agents plus code-revision turns will be too slow on CPU for useful iteration.

Recommended first setup:

- One Runpod GPU instance hosting a single local model server.
- One 7B-class instruct or coder model, likely quantized.
- Four logical agent sessions multiplexed onto that one model.

The critical point is to avoid running four separate heavyweight model processes. We want one server, shared weights, separate contexts.

## First Prototype Scope

- Fixed team shape: 3 miners, 1 aligner.
- Fixed planning cadence: replan every `N` steps or on failure/completion.
- Bounded skill authoring only inside a dedicated skills directory.
- No arbitrary repo-wide code edits.
- No free-form self-modification of the full system prompt.
- Strategy edits allowed only through a versioned strategy file or structured config.

## Main Open Questions

1. What model is small enough to run cheaply on Runpod but still reliable at writing short executable skills?
2. What observation summary should we provide so the LLM can reason without huge prompts?
3. What is the skill API: pure Python helpers, DSL, or action macros?
4. How do we sandbox generated code safely and deterministically?
5. Should coordination be centralized through one strategist, or fully decentralized through shared memory?
6. How often should agents call the LLM so that planning quality improves without stalling gameplay?
7. What failures trigger skill revision versus fallback to a scripted baseline?
8. How do we evaluate success against simpler baselines such as existing role policies or trained policies?

## Recommendation

Start with a hybrid system:

- scripted low-level execution
- LLM-written bounded skills
- shared blackboard coordination
- one strategist cadence, not continuous deliberation

That is the lowest-risk path to something Voyager-like in CoGames without building an expensive or unstable fully agentic controller.
