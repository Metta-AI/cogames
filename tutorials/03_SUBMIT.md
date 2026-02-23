# CoGames Tutorial: Submit & View Results

This notebook walks through submitting a policy to the CoGames leaderboard and viewing your results.


## Prerequisites

- Run from the repo root with your virtual environment activated.
- You need a policy checkpoint or script ready to submit.


Optional: confirm the CLI is available:

```bash
cogames --help
```


## Step 1 — Log in

Authenticate before submitting or checking the leaderboard.

```bash
cogames login
```

## Step 2 — Choose a policy to submit

You can submit either:
- A **policy class + weights** using `class=...` and `data=...`, or
- A **checkpoint bundle** (directory with `policy_spec.json`).

Examples below use placeholders. Replace them with your actual paths.

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
  policy_spec.json
```

`model_*.pt` is the weights file you can submit with `class=...,data=...`.


### Option A — Upload with class + weights

```bash
cogames upload -p class=my_policy.MyTrainablePolicy,data=./train_dir/<RUN_ID>/model_000001.pt -n my_policy_name --skip-validation
```


### Option B — Upload a checkpoint bundle

If your training run produced a `policy_spec.json`, you can upload the run directory directly:

```bash
cogames upload -p ./train_dir/<RUN_ID> -n my_policy_name --skip-validation
```


## Step 3 — Dry run (optional)

Validate the upload package without sending it:

```bash
cogames upload -p ./train_dir/<RUN_ID> -n my_policy_name --dry-run --skip-validation
```


## Step 4 — Submit to a season

By default, `cogames upload` both uploads and submits to a season. You can specify a season explicitly:

```bash
cogames upload -p ./train_dir/<RUN_ID> -n my_policy_name --season beta-cvc-no-clips --skip-validation
```

Or submit a previously uploaded policy to a season:

```bash
cogames submit my_policy_name --season beta-cvc-no-clips
```

List available seasons:

```bash
cogames seasons
```

Note: Scores can take a while to appear after submission.


## Step 5 — View your submissions

```bash
cogames submissions
```


## Step 6 — View the leaderboard

```bash
cogames leaderboard --season beta-cvc-no-clips
```


## Limits

- **Max upload size**: 500 MB. Submissions exceeding this limit are rejected with an HTTP 413 error. If your submission is too large, try excluding unnecessary files or compressing model weights.

## Troubleshooting

- **Auth errors**: run `cogames login` again.
- **Module not found**: use `class=...` with a fully qualified path or include the file in submission.
- **Invalid policy path**: ensure `-p` points to an existing bundle or weights file.
- **Local vs S3 checkpoints**: local training saves files under `./train_dir/`. Cloud training may require downloading or referencing the S3 bundle.
- **Policy too large**: the server enforces a 500 MB upload limit. Reduce your submission size by removing unnecessary files from your checkpoint directory before uploading.
