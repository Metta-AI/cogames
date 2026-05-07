# Coworld Episode Executor

`cogames.coworld.executor` is the hosted entrypoint for a single Coworld episode.
It mirrors the MettaGrid episode runner contract: read one job spec, run the full
episode, then upload artifacts to the URIs provided in environment variables.

## Command

```bash
python -m cogames.coworld.executor
```

The process expects Docker to be available because Coworld games and players are
Docker-first runnables.

## Required Input

```bash
JOB_SPEC_URI
```

`JOB_SPEC_URI` points to a JSON `CoworldEpisodeJobSpec`:

```json
{
  "manifest": {
    "game": {
      "runnable": {
        "image": "example-game:latest",
        "run": ["python", "/app/game/server.py"],
        "env": {}
      },
      "config_schema": {},
      "results_schema": {}
    }
  },
  "game_config": {},
  "players": [
    {
      "image": "example-player:latest",
      "run": ["python", "/app/player.py"],
      "env": {},
      "initial_params": {}
    }
  ],
  "episode_tags": {},
  "policy_names": ["policy:v1"]
}
```

The executor generates one token per player and injects those tokens into the
concrete `game_config` file before starting the game. Player `initial_params`
become query params on the `COGAMES_ENGINE_WS_URL` passed to that player.

## Optional Inputs

```bash
COWORLD_WORKDIR=/coworld
COWORLD_TIMEOUT_SECONDS=3600
GIT_COMMIT=...
COGAMES_VERSION=...
LOG_LEVEL=...
```

`COWORLD_WORKDIR` is the shared workspace mounted into nested game/player
containers. The executor writes local artifacts there while the episode runs.

## Output URIs

All output environment variables are optional, but hosted jobs normally provide
them:

```bash
RESULTS_URI
REPLAY_URI
RUNTIME_INFO_URI
DEBUG_URI
ERROR_INFO_URI
POLICY_LOG_URLS
```

Outputs:

- `RESULTS_URI`: game-defined `results.json`, validated against
  `manifest.game.results_schema`.
- `REPLAY_URI`: gzip-compressed replay uploaded as `replay.json.z`.
- `RUNTIME_INFO_URI`: JSON with `git_commit` and `cogames_version`.
- `DEBUG_URI`: zip containing game logs and per-player logs.
- `ERROR_INFO_URI`: `RunnerError` JSON if the runner crashes.
- `POLICY_LOG_URLS`: JSON object mapping player position to a destination URI.
  Each player log is uploaded from `policy_agent_{position}.txt`.

## Local CLI

```bash
uv run cogames coworld run-episode spec.json --output-dir ./coworld-episode-results
uv run metta/setup/tools/dev/cli.py run-coworld-episode spec.json --output-dir ./coworld-episode-results
```
