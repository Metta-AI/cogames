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
# - You need the files required to build a submission bundle.
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
# ## Step 2 — Build a submission bundle
#
# The canonical workflow is:
# 1. Build `submission.zip` with `cogames create-bundle`
# 2. Upload that bundle with `cogames upload`
# 3. Submit it to a season
#

# %% [markdown]
# Tip: find your run id and checkpoint by listing `./train_dir`:
#
# ```bash
# ls -lt train_dir
# ls -lt train_dir/<RUN_ID>
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
# The newest folder name is your `RUN_ID`. Inside that run:
#
# ```text
# train_dir/176850340101/
#   model_000001.pt
# ```
#
# Use the policy or checkpoint path, plus any extra runtime files or setup your policy needs, when creating the bundle.
#
# ```bash
# cogames create-bundle -p <policy-or-checkpoint> -o submission.zip [-f <extra-path> ...] [--setup-script <setup.py>]
# ```
#
# If your policy needs extra runtime files or setup, include them here. `agent/COGAMES_SUBMISSION.md` has a full repo example.
#

# %% [markdown]
# ## Step 3 — Upload the bundle
#
# Upload the prepared bundle:
#
# ```bash
# cogames upload -p ./submission.zip -n my_policy_name --no-submit
# ```
#

# %% [markdown]
# ## Step 4 — Dry run (optional)
#
# Validate the bundle locally without uploading:
#
# ```bash
# cogames upload -p ./submission.zip -n my_policy_name --dry-run
# ```
#

# %% [markdown]
# ## Step 5 — Submit to a season
#
# Submit the uploaded policy to a season:
#
# ```bash
# cogames submit my_policy_name --season beta-teams-small
# ```
#
# List available seasons:
#
# ```bash
# cogames seasons
# ```
#
# Note: Scores can take a while to appear after submission.
#

# %% [markdown]
# ## Step 6 — View your submissions
#
# ```bash
# cogames submissions
# ```
#

# %% [markdown]
# ## Step 7 — View the leaderboard
#
# ```bash
# cogames leaderboard --season beta-teams-small
# ```
#

# %% [markdown]
# ## Troubleshooting
#
# - **Auth errors**: run `cogames login` again.
# - **Module not found / 1011 during qualifying**: rebuild `submission.zip` with every runtime file and setup step your policy needs.
#   `agent/COGAMES_SUBMISSION.md` has a full repo example.
# - **Invalid policy path**: ensure `-p` points to an existing policy, checkpoint, or bundle.
# - **Local vs S3 checkpoints**: local training saves files under `./train_dir/`. Cloud training may require downloading or referencing the S3 bundle.
