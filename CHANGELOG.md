# Changelog

## May 5, 2026

**Among Them / BitWorld Integration**

- BitWorld episodes now run through the native server with real policy loading, reward collection, and server-side config
- Added BitWorld replay viewer (`cogames bitworld replay`, `cogames episode replay`) with format auto-detection
- Added `cogames bitworld quick-run` without a hardcoded player cap — forwards to the installed BitWorld launcher
- Added the NotTooDumb bot as an Among Them policy
- Added `cogames tutorial make-policy --amongthem` template and `cogames docs amongthem_policy` walkthrough
- Policy submissions now bundle environment metadata so the runner can validate observation/action contracts directly
- BitWorld remains an optional dependency (`cogames[bitworld]`) — base installs are unaffected

**CLI Changes (Breaking)**

- Removed `cogames missions`, `cogames evals`, `cogames variants`, `cogames describe`, and `cogames make-mission` commands (mission infrastructure is retained internally for `play`/`train`/`run`)
- Removed old hidden aliases: `cogames login`, `cogames train`, `cogames eval`, `cogames games`, top-level `cogames make-policy`
- Current commands: `cogames auth login`, `cogames tutorial train`, `cogames run`, `cogames tutorial make-policy`, `cogames season list`

**Tournament & Season Improvements**

- Authenticated leaderboard requests — logged-in users can see private/internal seasons
- Season CLI loads login token for private season access
- Pool config endpoint for season pools (game engine + env config) used by validation dry-runs
- Restored `PoolInfo.config_id` in season responses (regression from pool-config addition)
- Submission validation now uses the pool's `game_engine` instead of guessing
- Post-submit browser opens Observatory home instead of profile page
- `cogames validate-policy` rejects filenames that cannot be imported as Python modules

**Platform**

- Python requirement lowered from 3.12 to 3.11 (PEP 695 syntax replaced with 3.11-compatible equivalents)
- BitWorld env config is now first-class with a discriminated union (`EnvConfig` / `BitWorldEnvConfig` / `AnyEnvConfig`)
- Decoupled Gridworks mission routes from CogsGuard internals — routes now use the CoGames registry API
- CI test suite trimmed ~50% for cogames package (subprocess launches → in-process CliRunner)

**Documentation**

- Regenerated README CLI reference to include `auth`, `season`, `episode`, `assay`, and `bitworld` subgroups
- Fixed `cogames[neural]` install rendering in tutorial help (Rich was stripping the bracket)
- Removed dead `agent/COGAMES_SUBMISSION.md` references from submit tutorial
- Fixed `cogames docs readme` rendering to show raw Markdown links
- Updated all docs/help text to stop referencing removed mission commands
- Updated `cogames play --help` to use starter policy instead of `class=baseline`
- Fixed `change_vibe` technical manual description to clarify vibes are signaling-only in machina_1

## Jan 26, 2026

**CogsGuard is the New Default Game**

- `cogames missions` now shows CogsGuard (`cogsguard_arena`) as the default and only visible game
- Legacy missions (training_facility, hello_world, machina_1) are hidden from the CLI but still accessible by name
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
  - Introduced `--scripted` and `--trainable` modes for `cogames tutorial make-policy`. Each creates an example policy
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
