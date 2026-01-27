# CoGames Tutorial: make-policy

This notebook walks through the two tutorial policy generators: `--scripted` and `--trainable`.


## Prerequisites

- Run from the repo root with your virtual environment activated.
- If `cogames` is not found, activate `.venv` and retry.


Optional: confirm the CLI is available:

```bash
cogames --help
```


## Step 1 — Scripted policy template

The scripted template is a rule-based policy you can edit by hand. It runs immediately with `cogames play` and does not require training.


```bash
cogames tutorial make-policy --scripted -o my_scripted_policy.py
```

Expected output (example):
```
Scripted policy template copied to: /path/to/your/project/my_scripted_policy.py
Play with: cogames play -m cogsguard_arena.basic -p class=my_scripted_policy.StarterPolicy
```

Note: Replace `/path/to/your/project/` with your local repo path.


Common pitfalls:
- These commands overwrite existing files; use `-o` to choose a new filename.


Run the scripted policy (no training required):

```bash
cogames play -m cogsguard_arena.basic -p class=my_scripted_policy.StarterPolicy
```


Expected terminal output (example):
```
Playing cogsguard_arena.basic
Max Steps: 1000, Render: gui
Initializing Mettascope...
Episode Complete!
Steps: <N>
Total Rewards: [<value>]
Final Reward Sum: <value>
```


## Step 2 — Trainable policy template

The trainable template defines a neural policy. You edit the model/logic, then train it with `cogames tutorial train`, and run it using the saved weights.

```bash
cogames tutorial make-policy --trainable -o my_trainable_policy.py
```

Expected output (example):
```
Trainable policy template copied to: /path/to/your/project/my_trainable_policy.py
Train with: cogames tutorial train -m cogsguard_arena.basic -p class=my_trainable_policy.MyTrainablePolicy --steps 2000
```

Note: Replace `/path/to/your/project/` with your local repo path.


Common pitfalls:
- These commands overwrite existing files; use `-o` to choose a new filename.


Train and run the trainable policy:

```bash
cogames tutorial train -m cogsguard_arena.basic -p class=my_trainable_policy.MyTrainablePolicy --steps 2000
cogames play -m cogsguard_arena.basic -p class=my_trainable_policy.MyTrainablePolicy,data=train_dir/<your_run>/checkpoints/<your_run>:v<X>/weights.safetensors  # Replace <your_run> with your actual run name and <X> with the version number from your training output
```

Note: Add `--steps` for quick tutorial runs; the default is very large.


Expected terminal output (example):
```
Training on mission: cogsguard_arena.basic
...progress logs...
Training complete. Checkpoints saved to: ./train_dir
Final checkpoint: train_dir/<run>/checkpoints/<run>:vX
```


Expected terminal output (example):
```
Playing cogsguard_arena.basic
Max Steps: 1000, Render: gui
Initializing Mettascope...
Episode Complete!
Steps: <N>
Total Rewards: [<value>]
Final Reward Sum: <value>
```


## Step 3 — Customize your own policy

You can edit the generated policy files to make your own behavior. For scripted policies, change rules directly and run immediately. For trainable policies, modify the model or logic, then retrain and run with the new checkpoint.


## Summary
- **Scripted** = rule-based, runs immediately without training.
- **Trainable** = neural policy, train with `cogames tutorial train`.


## What to do next
- **Scripted**: run the `cogames play ...` command printed by the CLI.
- **Trainable**: run the `cogames tutorial train ...` command printed by the CLI.

