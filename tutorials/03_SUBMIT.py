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
# # CoGames Tutorial: Submit & View Results
#
# This notebook walks through submitting a policy to the CoGames leaderboard and viewing your results.
#

# %% [markdown]
# ## Prerequisites
#
# - Run from the repo root with your virtual environment activated.
# - You need a policy checkpoint or script ready to submit.
#

# %% [markdown]
# Optional: confirm the CLI is available:
#
# ```bash
# cogames --help
# ```
#

# %% [markdown]
# ## Step 1 — Log in
#
# Authenticate before submitting or checking the leaderboard.
#
# ```bash
# cogames login
# ```

# %% [markdown]
# ## Step 2 — Choose a policy to submit
#
# You can submit either:
# - A **checkpoint bundle** (directory or .zip with `policy_spec.json`), or
# - A **policy class + weights** using `class=...` and `data=...`.
#
# Examples below use placeholders. Replace them with your actual paths.
#

# %% [markdown]
# Tip: find your run id and checkpoint by listing `./train_dir` and inspecting the latest run folder. For example:
#
# ```bash
# ls -lt train_dir
# ls -lt train_dir/<RUN_ID>/checkpoints
# ```
#
# Expected output (example):
#
# ```text
# train_dir/
#   176850340101/
#   176850219234/
# ```
#
# What this shows:
# - The newest folder name is your `RUN_ID`.
#
# Then, inside that run:
#
# ```text
# train_dir/176850340101/checkpoints/
#   model_000001.pt
#   trainer_state.pt
# ```
#
# What this shows:
# - `model_*.pt` is the weights file you can submit with `class=... ,data=...`.
#

# %% [markdown]
# ### Option A — Submit a checkpoint bundle
#
# ```bash
# cogames upload -p ./train_dir/<RUN_ID>/checkpoints/<RUN_ID>:<CHECKPOINT> -n my_policy_name
# ```
#
# Replace `<RUN_ID>` with your run name and `<CHECKPOINT>` with the checkpoint version.
#

# %% [markdown]
# ### Option B — Submit a policy class + weights
#
# ```bash
# cogames upload -p class=my_policy.MyTrainablePolicy,data=./train_dir/<run_id>/model_000001.pt -n my_policy_name
# ```
#
# Use this form if your training output is `model_*.pt` (not a bundle).
#

# %% [markdown]
# Tip: Use `--dry-run` first to validate your policy bundle before uploading.
#

# %% [markdown]
# Note: Scores can take a while to appear after submission (pending → scored).
#

# %% [markdown]
# Tip: It can take a while for scores to appear after submission.
#

# %% [markdown]
# ## Step 3 — Dry run upload (recommended)
#
# Validate the upload package without sending it.
#
#

# %% [markdown]
# ```bash
# cogames upload -p ./train_dir/{RUN_ID}/checkpoints/{RUN_ID}:{CHECKPOINT} -n {POLICY_NAME} --dry-run
# ```
#

# %% [markdown]
# Expected output (example):
# ```text
# Dry run complete - validation passed, zip created!
# ```
#

# %% [markdown]
# ## Step 4 — Upload policy
#
# Upload the policy bundle to CoGames.
#
#

# %% [markdown]
# ```bash
# cogames upload -p ./train_dir/{RUN_ID}/checkpoints/{RUN_ID}:{CHECKPOINT} -n {POLICY_NAME}
# ```
#

# %% [markdown]
# Expected output (example):
# ```text
# Upload complete: my_policy_name:v1
# To submit to a tournament: cogames submit my_policy_name:v1 --season <SEASON>
# ```
#

# %% [markdown]
# ## Step 5 — Submit to a season
#
# Once uploaded, submit the policy version to a season.
#
#
# Pick a season from the list below, then submit your policy to that season.
#

# %% [markdown]
# ```bash
# cogames seasons
# ```
# ```
#                                Tournament Seasons
#
#  Season           Description                           Pools
#  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  beta             Policies start in qualifying;         qualifying, competition
#                   promoted to competition if score
#                   meets threshold
# ```

# %% [markdown]
# ```bash
# cogames submit {POLICY_NAME} --season {SEASON}
# ```
#

# %% [markdown]
# Expected output (example):
# ```text
# Submitted successfully!
# Submission ID: <uuid>
# ```
#

# %% [markdown]
# ## Step 6 — View your submissions
#
# List your submissions and see when scores appear.
#
#

# %% [markdown]
# ```bash
# cogames submissions
# ```
#
# Expected output
# ```
#                              Your Uploaded Policies
#
#  Policy                                              Uploaded           Seasons
#  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  your-name.2026-01-16.test-starter-policy.14:48:00…   2026-01-16 22:49   beta
# ```

# %% [markdown]
# ## Step 7 — View the leaderboard
#
# View the leaderboard.
#
#
#

# %% [markdown]
# Then run:
#
# ```bash
# cogames leaderboard --season {SEASON}
# ```
#

# %% [markdown]
# ## Troubleshooting
#
# - **Auth errors**: run `cogames login` again.
# - **Module not found**: use `class=...` with a fully qualified path or include the file in submission.
# - **Invalid policy path**: ensure `-p` points to an actual bundle or weights file.
#

# %% [markdown]
# ## Common submission issues
# - **Module not found**: use a fully qualified class path or include the policy file in your submission.
# - **Invalid policy path**: ensure `-p` points to an existing bundle or weights file.
#

# %% [markdown]
# ## Local vs S3 checkpoints
# - Local training usually saves files under `./train_dir/`.
# - If you trained in a sandbox or cloud job, you may need to download or reference the S3 bundle.
#
