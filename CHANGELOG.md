# Changelog

## Jan 26, 2026

**CogsGuard is the New Default Game**

- `cogames missions` now shows CogsGuard (`cogsguard_arena`) as the default and only visible game
- Legacy missions (training_facility, hello_world, machina_1) are hidden from the CLI but remain available for internal
  use
- Default season changed to `beta-cogsguard`

**Season-Aware Policy Validation**

- `cogames upload` and `cogames validate-policy` now accept `--season` to validate against the correct game for that
  season
- Validation uses the game appropriate for the target season (e.g., `beta` season validates against
  `training_facility.harvest`, `beta-cogsguard` validates against `cogsguard_machina_1.basic` using the Machina1 map)
- This ensures policies are validated against the same game they'll compete in

## Jan 23, 2026

**CLI Flag Updates**

Breaking changes to command-line arguments for improved consistency:

- `cogames submissions` - replaced `policy_name` with an optional `--policy/-p` flag instead of a positional argument
- `cogames diagnose` - removed `-m` short form for `--experiments`
- `cogames diagnose` - replaced `--repeats` with `--episodes/-e`
- `cogames make-mission` - removed `-h` and `-w` short form for `--height` and `--width`
- `cogames login` - replaced `--server/-s` with `--login-server`
- `cogames pickup` - added `--mission/-m` and `--variant/-v` flags (previously hardcoded to `machina_1.open_world`)
- `cogames validate-policy` - replaced `policy` positional argument with `--policy/-p` flag

## Jan 7, 2026

**Tournament CLI Updates**

- `cogames upload` - Upload a policy without submitting to a tournament
- `cogames submit` - Submit an uploaded policy to a tournament season
- `cogames submissions` - View your uploaded policies and tournament submissions
- `cogames leaderboard --season <name>` - View the leaderboard for a season
- `cogames seasons` - List available tournament seasons
- `cogames pickup` - Evaluate a candidate policy against a fixed pool and compute VOR

## Dec 16, 2025

**CLI Command Restructuring**

- `cogames eval` has been renamed to `cogames run`

- Commands intended for demonstration purposes to get you up and running are housed under `cogames tutorial`:
  - `cogames train` -> `cogames tutorial train`. This command offers a thin wrapper around
    [pufferlib](https://github.com/PufferAI/PufferLib/tree/3.0) train
  - `cogames make-policy` -> `cogames tutorial make-policy`. This command creates an example python implementation of a
    scripted policy for you to modify
  - Introduced `--scripted` and `--trainable` modes for `cogames make-policy`. Each creates an example policy
    implementation from a template in different style: scripted or neural-net based.
  - `cogames tutorial` -> `cogames tutorial play`. This is an interactive tutorial that walks you through the basic
    mechanics of the game.

## Dec 3, 2025

**What's new**

Breaking change: The format for specifying a policy in `cogames` command is now updated to support passing keyword
arguments.

- cogames `-p` or `--policy` arguments should now take the form `class=...[,data=...][,proportion=...] [,kw.name=val]`

- Example: uv run cogames play -m hello_world -p class=path.to.MyPolicyClass,data=my_data/policy.pt,kw.policy_mode=fast

Update: `cogames submit` now accepts much larger checkpoint files. It makes use of presigned s3 URLs for uploading your
checkpoint data instead of going through web servers that enforced lower file size caps.

Update: `cogames submit` does a light round of validation on your proposed submission before uploading it. It ensures
that your policy is loadable (using the latest copy of `cogames` in pypi, not necessarily the one you have installed),
and can perform a step on a simple environment.
