# cogames — retired

CoGames has been retired in favor of [Coworld](https://github.com/Metta-AI/coworld).

This package exists only as a tombstone for `pip install cogames`. It has no
runtime surface; importing it emits a `DeprecationWarning`.

## Migration

| Old import | New import |
|---|---|
| `from cogames.core import …` | `from mettagrid.cogame.core import …` |
| `from cogames.game import …` | `from mettagrid.cogame.game import …` |
| `from cogames.variants import …` | `from mettagrid.cogame.variants import …` |
| `from cogames.sdk.cogsguard import …` | `from mettagrid.sdk.cogsguard import …` |

The cogames CLI (`cogames play`, `cogames train`, …), `cogames.policy`,
`cogames.standalone_games`, `metta_alo`, and the embedded game configs
(`cogames.games.cogs_vs_clips`, etc.) are not migrated; they were tied to
infrastructure that no longer exists. Use Coworld for an equivalent runtime.

Drop `cogames` from your dependencies.
