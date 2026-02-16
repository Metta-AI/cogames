# CoGames Tutorial: train

This notebook walks through `cogames tutorial train` with short, beginner-friendly runs.


## Prerequisites

- Run from the repo root with your virtual environment activated.
- If `cogames` is not found, activate `.venv` and retry.


Optional: list missions to choose a different training target:

```bash
cogames missions
```


Optional: confirm the CLI is available:

```bash
cogames --help
```


## Step 1 — Train with the built-in LSTM policy

Use a small `--steps` value for a quick tutorial run. The default is very large.

```bash
cogames tutorial train -m cogsguard_machina_1.basic -p class=lstm --steps 2000
```

Expected terminal output (example):
```
Training on mission: cogsguard_machina_1.basic
...progress logs...
Training complete. Checkpoints saved to: ./train_dir
Final checkpoint: ./train_dir/<run_id>/model_000001.pt

To play with this policy:
  cogames play -m cogsguard_machina_1.basic -p class=lstm,data=./train_dir/<run_id>/model_000001.pt
```

Replace `<run_id>` with your actual run ID from your training output.


Common pitfalls:
- If you omit `--steps`, training may run for a very long time.


Troubleshooting:
- Seeing "CUDA not available; falling back to CPU" is fine for local runs.
- Pressing Ctrl-C stops training and prints resume/play/eval commands.


## Step 2 — Train your own policy (from the template)

First generate the trainable template (if you have not already):

```bash
cogames tutorial make-policy --trainable -o my_trainable_policy.py
```

Then run:
```bash
cogames tutorial train -m cogsguard_machina_1.basic -p class=my_trainable_policy.MyTrainablePolicy --steps 2000
```

Expected terminal output (example):
```
Training on mission: cogsguard_machina_1.basic
...progress logs...
Training complete. Checkpoints saved to: ./train_dir
Final checkpoint: ./train_dir/<run_id>/model_000001.pt
```


## What to do next
- Play using the saved checkpoint:
  `cogames play -m cogsguard_machina_1.basic -p class=lstm,data=./train_dir/<run_id>/model_000001.pt`
- Evaluate using the same checkpoint:
  `cogames eval -m cogsguard_machina_1.basic -p class=lstm,data=./train_dir/<run_id>/model_000001.pt`
