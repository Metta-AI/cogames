# AGENTS.md — cogames

Public game configs for the Alignment League Benchmark. Published to PyPI; depends on `mettagrid` (workspace source,
pinned version in `pyproject.toml`). Keep changes compatible with the pinned `mettagrid` API.

## CLI

The package installs a `cogames` entrypoint (`cogames.main:app`):

```bash
uv run cogames --help
uv run cogames play <mission>        # see skills cg.play / cg.test for renderer details
```

## Tests

```bash
uv run metta pytest packages/cogames/tests -v
uv run metta pytest --changed
```

## Lint

```bash
uv run metta lint --fix              # ruff (also runs via the Edit/Write hook)
```

## Reference

- `TECHNICAL_MANUAL.md` — game mechanics and config reference.
- `MISSION.md` — mission definitions.
- Source lives under `src/cogames` and `src/metta_alo`.
