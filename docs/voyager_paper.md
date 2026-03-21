# VOYAGER Paper Notes

Source paper:
- VOYAGER: An Open-Ended Embodied Agent with Large Language Models
- ArXiv: https://arxiv.org/abs/2305.16291
- Local PDF: [voyager_2305.16291.pdf](./voyager_2305.16291.pdf)

## Why this note exists

This file is a repo-local summary of the VOYAGER paper so the agent can read the core ideas without internet access. It is not a verbatim copy of the paper. It focuses on the parts most relevant to building an LLM-driven CoGames agent.

## Main idea

VOYAGER is a Minecraft agent that uses an LLM as a lifelong learner rather than just a per-step controller. Its core claim is that long-horizon embodied behavior improves when the LLM operates over executable skills and iterative feedback instead of only primitive actions.

The paper emphasizes three ingredients:

1. Automatic curriculum
The agent proposes its own next tasks based on current state, prior successes, and prior failures. The curriculum is meant to keep the agent near its capability frontier instead of repeatedly asking fixed benchmark tasks.

2. Skill library
The agent stores successful behaviors as reusable code. Skills are indexed by text embeddings of their descriptions and later retrieved when a new task is similar. This lets the agent compose old solutions into new ones and reduces forgetting.

3. Iterative prompting with feedback
The LLM writes code, executes it, observes environment feedback and interpreter errors, then revises the code. A separate self-verification step checks whether the intended task was actually completed.

## Why code is the action space

The paper argues for code instead of low-level actions because code:

- naturally represents temporally extended behavior
- is interpretable
- is reusable
- is compositional
- supports long-horizon control better than single-step action selection

This is the key idea we should transfer into CoGames. The LLM should mostly choose or revise bounded skills, not drive every move every tick.

## VOYAGER architecture

At a high level, the loop is:

1. Observe current world state.
2. Propose the next task via the automatic curriculum.
3. Retrieve relevant prior skills from the skill library.
4. Ask the LLM to generate or revise code for the task.
5. Execute the code in the environment.
6. Collect:
   - environment feedback
   - execution errors
   - self-verification result
7. Iterate until the task succeeds.
8. Store the successful code as a new skill.
9. Move on to the next task.

## Automatic curriculum details

The curriculum prompt includes:

- current inventory and equipment
- nearby entities / blocks
- biome / time / health / hunger / position
- completed tasks
- failed tasks
- directives that encourage diverse exploration but avoid tasks that are too hard

The objective is not a fixed benchmark reward. It is open-ended exploration and capability growth.

For CoGames, the equivalent would be:

- current role
- inventory / gear
- visible stations / extractors / junctions / hub
- known map memory
- recent successes / recent failures
- current team situation
- candidate next objectives that are just beyond current progress

## Skill library details

Each skill is stored as code plus a textual description. The description is embedded and used as the retrieval key. On a new task, the system retrieves a few relevant prior skills and includes them in the LLM context.

This matters for our design because it suggests:

- skills should be explicit units
- each skill should have a human-readable description
- retrieval should be over descriptions, not just file names
- successful skills should be stable and reusable across matches

For CoGames, skills should look like:

- `gear_up(role)`
- `mine_until_full`
- `deposit_to_hub`
- `get_heart`
- `align_neutral`
- `explore`
- `unstuck`

## Iterative prompting details

The paper’s code-generation loop uses three feedback channels:

1. Environment feedback
Natural-language or structured feedback from the world about what happened during execution.

2. Execution errors
Interpreter/runtime errors from the generated program.

3. Self-verification
A second LLM check that decides whether the task succeeded and, if not, critiques the attempt.

This is important for us because our current system already has the beginning of this pattern:

- planner selects a skill
- skill executes
- we log stuck / completed / acquired gear / acquired heart / discovered target

The next step is to make that feedback richer and more structured, not necessarily to jump straight to arbitrary code generation.

## What the paper shows empirically

The paper reports that VOYAGER:

- discovers more unique items than prior baselines
- travels farther
- reaches major Minecraft milestones much faster
- transfers learned skills to new worlds better than comparison methods

The point is not just raw benchmark score. It is open-ended accumulation and transfer of reusable competence.

## What maps directly to CoGames

Good transfer:

- skill library
- bounded code / bounded skills
- iterative feedback loop
- curriculum over next objectives
- retrieval of prior skills
- using the LLM sparsely at decision boundaries

Bad transfer if copied literally:

- fully open-ended single-agent exploration as the only objective
- assuming one agent owns the whole world state
- assuming task success is always easy to verify from inventory alone

CoGames has team structure, contention, and role specialization, so we need explicit coordination and shared state.

## How this should influence our CoGames design

The paper supports the following design choices:

1. The LLM should operate mostly over skills, not primitive per-tick control.

2. Skills should be explicit, named, reusable, and stored with descriptions.

3. The planner should be called at sparse boundaries:
   - skill complete
   - skill failed
   - agent stuck
   - role objective changed
   - important world event happened

4. We should preserve execution feedback as structured trajectory data.

5. We should eventually add retrieval over prior successful skills or strategy snippets.

## Important differences from our current system

VOYAGER in the paper is:

- single-agent
- open-ended
- centered on code generation and lifelong accumulation

Our current CoGames prototype is:

- multi-agent
- role-specialized
- more like LLM planning over scripted skills than free code generation

That is still aligned with the paper. It is a narrower and safer starting point.

## Recommended adaptation path for this repo

Stage 1:
- keep LLM as a planner over fixed skills
- improve structured feedback
- improve skill execution and navigation

Stage 2:
- store successful trajectories and stable skill variants
- add retrieval over prior strategies / skill descriptions

Stage 3:
- allow bounded skill revision or code synthesis in a restricted runtime
- validate and promote successful generated skills into the shared library

Stage 4:
- introduce team-level curriculum / strategy updates
- let the planner coordinate role allocation and shared priorities

## Key takeaways

- The strongest idea in VOYAGER is not “LLM controls everything.”
- The strongest idea is “LLM grows a reusable library of executable behaviors through iterative feedback.”
- For CoGames, the right first version is an LLM that selects and later revises bounded skills, not an LLM that emits every low-level action.
- Our current 3-agent prototype is already on the right path if we keep improving skill quality, feedback quality, and eventually retrieval.
