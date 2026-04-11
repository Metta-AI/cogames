# CoGames Tutorial: Submit & View Results

This notebook walks through submitting a policy to the CoGames leaderboard and viewing your results.


## Prerequisites

- Run from the repo root with your virtual environment activated.
- You need the files required to build a submission bundle.


Optional: confirm the CLI is available:

```bash
cogames --help
```


## Step 1 — Log in

Authenticate before submitting or checking the leaderboard.

```bash
cogames login
```

## Step 2 — Build a submission bundle

The canonical workflow is:
1. Build `submission.zip` with `cogames create-bundle`
2. Upload that bundle with `cogames upload`
3. Submit it to a season


Tip: find your run id and checkpoint by listing `./train_dir`:

```bash
ls -lt train_dir
ls -lt train_dir/<RUN_ID>
```

Expected output (example):

```text
train_dir/
  176850340101/
  176850219234/
```

The newest folder name is your `RUN_ID`. Inside that run:

```text
train_dir/176850340101/
  model_000001.pt
```

Use the checkpoint path plus any runtime files your policy needs when creating the bundle.


### Example A — Build a bundle from a Python policy

```bash
cogames create-bundle -p class=my_policy.MyPolicy -o submission.zip -f my_policy.py
```


### Example B — Build a bundle from a checkpoint plus runtime files

For a checkpoint-backed policy, include the files the policy imports at runtime:

```bash
cogames create-bundle -p ./train_dir/<RUN_ID>:latest -o submission.zip \
  -f agent \
  -f packages/cortex/pyproject.toml \
  -f packages/cortex/src \
  --setup-script cogames-agents/trained_setup_script.py
cogames upload -p ./submission.zip -n my_policy_name
```

If your policy needs extra runtime files or setup, include them in the bundle (more details in `agent/COGAMES_SUBMISSION.md`).


## Step 3 — Upload the bundle

Upload the prepared bundle:

```bash
cogames upload -p ./submission.zip -n my_policy_name --no-submit
```


## Step 4 — Dry run (optional)

Run the Docker smoke test without sending the bundle:

```bash
cogames upload -p ./submission.zip -n my_policy_name --dry-run
```

Dry-run is a smoke test, not a guarantee that later tournament matches will succeed.


## Step 5 — Submit to a season

Submit the uploaded policy to a season:

```bash
cogames submit my_policy_name --season beta-teams-small
```

`cogames upload` can also upload and submit in one command if you pass `--season`.

List available seasons:

```bash
cogames seasons
```

Note: Scores can take a while to appear after submission.


## Step 6 — View your submissions

```bash
cogames submissions
```


## Step 7 — View the leaderboard

```bash
cogames leaderboard --season beta-teams-small
```


## Troubleshooting

- **Auth errors**: run `cogames login` again.
- **Module not found / 1011 during qualifying**: rebuild `submission.zip` with the runtime files your policy imports
  and a `--setup-script` if needed (more details in `agent/COGAMES_SUBMISSION.md`).
- **Invalid policy path**: ensure `-p` points to an existing bundle or weights file.
- **Local vs S3 checkpoints**: local training saves files under `./train_dir/`. Cloud training may require downloading or referencing the S3 bundle.
- **Dry-run passed but qualifying failed**: the default validation run is only a short smoke test. Check the season
  match artifacts to debug full-match failures.
