# README Spec

Restructure `cogames` README using nbdev notebooks.

## Motivation

Goals:

- Website describes "why", README describes "how"
- Quick-start-first: 2 min to play, ~5 min to submit
- README is concise
- Standard open source practices: LICENSE, security.txt, CI badges
- README codeblocks are tested and stay in sync with CLI
- Long-running flows (tournament submission) tested nightly

Why nbdev:

- README.ipynb compiles to README.md, testable via CI
- Users can single-click launch in CoLab

References: [Gymnasium README](https://github.com/Farama-Foundation/Gymnasium), [nbdev](https://nbdev.fast.ai/)

## Structure

| Section                | Content                                                              |
| ---------------------- | -------------------------------------------------------------------- |
| Hero                   | One-liner, badges (PyPI, Python, Discord), link to site              |
| Quick Start            | Play tutorial, then submit a policy. Training is opt-in.             |
| About the Game         | Brief description + GIF, link to MISSION.md                          |
| About the Tournament   | Describes rough tournament rules, links to site                      |
| Tutorials              | Links to notebooks: make-policy, train, tournament (see tutorial.md) |
| Command Reference      | Auto-generated from typer                                            |
| Installing from Source | For contributors; requires nim, bazel                                |
| Resources              | Links: MISSION.md, TECHNICAL_MANUAL.md, Discord                      |
| Citation               | BibTeX                                                               |

## Command Reference

Auto-generated from typer CLI.

### Local

| Command                 | Description                                  |
| ----------------------- | -------------------------------------------- |
| `cogames tutorial play` | Interactive tutorial to learn game mechanics |
| `cogames play`          | Play an episode with a policy                |
| `cogames run`           | Evaluate policies on missions                |
| `cogames missions`      | List available missions                      |
| `cogames variants`      | List mission variants/modifiers              |
| `cogames evals`         | List evaluation mission sets                 |
| `cogames policies`      | List available policy shorthands             |
| `cogames make-mission`  | Create custom mission configuration          |
| `cogames pickup`        | _(upcoming)_                                 |
| `cogames scrimmage`     | _(upcoming)_                                 |

### Tournament

| Command               | Description                         |
| --------------------- | ----------------------------------- |
| `cogames login`       | Authenticate with tournament server |
| `cogames upload`      | Upload a policy to the server       |
| `cogames submit`      | Submit a policy to a tournament     |
| `cogames submissions` | View your tournament submissions    |
| `cogames seasons`     | List available tournament seasons   |
| `cogames leaderboard` | View tournament leaderboard         |

## Installing from Source

For contributors or development. Most users should install from PyPI.

Requires: [nim](https://nim-lang.org/install.html), [bazel](https://bazel.build/install)

```bash
git clone https://github.com/metta-ai/metta.git
cd metta/packages/cogames
uv pip install -e .
```

## Testing & CI

### CI Flow

1. Devs edit README.ipynb or tutorial notebooks
2. Devs regenerate README.md
3. CI runs notebooks and ensures README.md matches commit
4. CI checks command reference against `cogames --help` output

### Nightly Runner

Long-running integration tests that can't run in CI on every commit:

- Run on both public package and installed-from-source
- Test against live production tournament service
- Create two policies: one that will definitely error, one that's a trained net that will not
- Submit both to leaderboard
- Track progression through tournament matches
