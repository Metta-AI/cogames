# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CoGames Tutorial: train
#
# This notebook walks through `cogames tutorial train` with short, beginner-friendly runs.
#

# %% [markdown]
# ## Prerequisites
#
# - Run from the repo root with your virtual environment activated.
# - If `cogames` is not found, activate `.venv` and retry.
#

# %% [markdown]
# Optional: list missions to choose a different training target:
#
# ```bash
# cogames missions
# ```
#

# %% [markdown]
# Optional: confirm the CLI is available:
#
# ```bash
# cogames --help
# ```
#

# %% [markdown]
# ## Step 1 — Train with the built-in LSTM policy
#
# Use a small `--steps` value for a quick tutorial run. The default is very large.
#

# %% [markdown]
# Tip: the default `--steps` is extremely large; use a small number for quick tutorials.
# Checkpoint bundles are written under `./train_dir/<run>/checkpoints/<run>:vX`.
#
# ```bash
# cogames tutorial train -m cogsguard_arena.basic -p class=lstm --steps 2000
# ```

# %% [markdown]
# Expected terminal output (example):
# ```
# Training on mission: cogsguard_arena.basic
# ...progress logs...
# Training complete. Checkpoints saved to: ./train_dir
# Checkpoint saved to: ./train_dir/<run_id>/model_000001.pt
# ```
#
# Replace `<run_id>` with your actual run ID from your training output.
#
# Note: To run the checkpoint, use the class+data form with the `-p` flag, e.g. `-p class=lstm,data=./train_dir/<run_id>/model_000001.pt`.
#

# %% [markdown]
# Common pitfalls:
# - If you omit `--steps`, training may run for a very long time.
#

# %% [markdown]
# Troubleshooting:
# - Seeing “CUDA not available; falling back to CPU” is fine for local runs.
# - Pressing Ctrl‑C stops training and prints resume/play/eval commands.
#

# %% [markdown]
# ## Step 2 — Train your own policy (from the template)
#
# First generate the trainable template (if you have not already):
#
# ```bash
# cogames tutorial make-policy --trainable -o my_trainable_policy.py
# ```

# %% [markdown]
# Then run:
# ```bash
# cogames tutorial train -m cogsguard_arena.basic -p class=my_trainable_policy.MyTrainablePolicy --steps 2000
# ````


# %% [markdown]
# Expected terminal output (example):
# ```
# Training on mission: cogsguard_arena.basic
# ...progress logs...
# Training complete. Checkpoints saved to: ./train_dir
# Final checkpoint: train_dir/<run>/checkpoints/<run>:vX
# ```
#
# **Note:** When using the checkpoint in your code, you'll need to use the format `train_dir/<run>` rather than the full path shown in the output above.
#

# %% [markdown]
# ## What to do next
# - Play using the saved checkpoint bundle:
#   `cogames play -m cogsguard_arena.basic -p class=lstm,data=./train_dir/<run_id>/model_000001.pt`
# - Evaluate using the same bundle:
#   `cogames eval -m cogsguard_arena.basic -p class=lstm,data=./train_dir/<run_id>/model_000001.pt`
#
# Note: tutorial train writes `model_*.pt` under `./train_dir/<run_id>/`. Use `class=...` + `data=...` to run it.
#
