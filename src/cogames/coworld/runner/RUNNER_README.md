# Coworld Episode Runner

`cogames.coworld.runner.runner` is the fully local Coworld episode
runner. It starts the game and players with Docker on the current machine, waits
for the episode to finish, and writes artifacts to a local workspace.

Use it through the public CLI:

```bash
uv run cogames coworld run-episode spec.json --output-dir ./coworld-episode-results
```

The hosted production runner is `cogames.coworld.runner.kubernetes_runner`.
That path runs the game and players as Kubernetes containers instead of requiring
Docker inside the runner container.
