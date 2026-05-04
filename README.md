# CoGames: A Game Environment for the Alignment League Benchmark

<p align="center">
  <a href="https://pypi.org/project/cogames/">
    <img src="https://img.shields.io/pypi/v/cogames" alt="PyPi version">
  </a>
  <a href="https://pypi.org/project/cogames/">
    <img src="https://img.shields.io/pypi/pyversions/cogames" alt="Python version">
  </a>
  <a href="https://discord.gg/secret-hologenesis">
    <img src="https://img.shields.io/discord/1309708848730345493?logo=discord&logoColor=white&label=Discord" alt="Discord">
  </a>
  <a href="https://deepwiki.com/Metta-AI/cogames">
    <img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki">
  </a>
  <a href="https://colab.research.google.com/github/Metta-AI/cogames/blob/main/README.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
  </a>

  <a href="https://softmax.com/">
    <img src="https://img.shields.io/badge/Softmax-Website-849EBE?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyBpZD0iTGF5ZXJfMSIgZGF0YS1uYW1lPSJMYXllciAxIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB2aWV3Qm94PSIwIDAgNTI5LjIyIDUzNy40NyI+CiAgPGRlZnM+CiAgICA8c3R5bGU+CiAgICAgIC5jbHMtMSB7CiAgICAgICAgY2xpcC1wYXRoOiB1cmwoI2NsaXBwYXRoKTsKICAgICAgfQoKICAgICAgLmNscy0yIHsKICAgICAgICBmaWxsOiBub25lOwogICAgICB9CgogICAgICAuY2xzLTIsIC5jbHMtMywgLmNscy00LCAuY2xzLTUgewogICAgICAgIHN0cm9rZS13aWR0aDogMHB4OwogICAgICB9CgogICAgICAuY2xzLTMgewogICAgICAgIGZpbGw6ICMwZTI3NTg7CiAgICAgIH0KCiAgICAgIC5jbHMtNCB7CiAgICAgICAgZmlsbDogI2JiY2NmMzsKICAgICAgfQoKICAgICAgLmNscy01IHsKICAgICAgICBmaWxsOiAjODU5ZWJlOwogICAgICB9CiAgICA8L3N0eWxlPgogICAgPGNsaXBQYXRoIGlkPSJjbGlwcGF0aCI+CiAgICAgIDxyZWN0IGNsYXNzPSJjbHMtMiIgd2lkdGg9IjUyOS4yMSIgaGVpZ2h0PSI1MzcuNDciLz4KICAgIDwvY2xpcFBhdGg+CiAgPC9kZWZzPgogIDxnIGNsYXNzPSJjbHMtMSI+CiAgICA8cGF0aCBjbGFzcz0iY2xzLTMiIGQ9Ik00MzUuNzksMTY3LjA5YzEuMzksMTQuNzIsMi4yOCwzNS4xNS4wNyw1OS4xOC0yLjc2LDMwLTguNDEsOTEuNDItNTMuMDYsMTQ0Ljg1LTEyLjcxLDE1LjIxLTMxLjUsMzcuMTctNjQuMjcsNTAuNzUtMjAuMSw4LjMzLTM5LjcyLDExLjE0LTU3LjA4LDExLjE0LTI3LjIxLDAtNDguODctNi44OS01OC4xNC0xMC4xOC0yNC4yOC04LjcxLTQ2LjI2LTEzLjg4LTY0LjgyLTE3LjAzLTkuODEtMS42Ny0xNy4yLTIuODktMjcuNTEtMy4zNy01LjMtLjI0LTExLjItLjU4LTE3LjUyLS41OC0xNC45NywwLTMyLjMxLDEuOTEtNDkuNjYsMTEuNTUtNy4yLDQtMjIuMzYsMTEuODItMzMuMjgsMjguNC0yLjMxLDMuNS01LjczLDcuMjYtNy4xLDEzLjE3LS43NywzLjMzLS44Myw2LjQ1LS41NSw5LjE3LDEwLjE4LDE2LjUsMzEuOSwyNS4xLDcwLjMxLDM5Ljc5LDQwLjU4LDE1LjUyLDc2LjQ2LDIzLjA4LDEwMy4zNiwyNy4yLDM5LjYyLDYuMDcsNzAuMjEsNi4yOSw4OC44NSw2LjM1LjY5LDAsMS4zOSwwLDIuMDgsMCw1MS4zMSwwLDg4Ljg1LTUuOTYsOTYuNzQtNy4yNiwyMS4xNS0zLjQ2LDUwLjY1LTguNDUsODYuOC0yMi4zOSwzOS41Mi0xNS4yNCw2Ni42MS0yNS42OCw3Ni4yNS01MC42OSwxLjUtMy44OCw0LjYzLTEzLjQzLTIuODYtNTUuMDctNy41Ny00Mi4xNS0xOC4xMi03My4xOS0xOS42Ny03Ny42OC0yMC45MS02MC43My0zMS4zNy05MS4wOS00Ny4xNS0xMjAuNTktNy4xNi0xMy4zOC0xNC4zNy0yNS40My0yMS44Mi0zNi43MiIvPgogICAgPHBhdGggY2xhc3M9ImNscy01IiBkPSJNNDA1LjgsMTI2LjM4Yy0yLjcsMTQuMTMtNy40MywzMy40Ny0xNi4xOSw1NS4zOS05LjMzLDIzLjMzLTE3LjQzLDQzLjU5LTM2LjExLDYzLjctOS43MywxMC40Ny0zNC4xMSwzNi43LTcwLjQ5LDQwLjg5LTMuMjguMzgtNi40Ny41NS05LjYuNTUtMTUuMjQsMC0yOS4xMy00LjE2LTQ2LjQ4LTkuMzYtMjIuNjQtNi43OC0zMy45OC0xNC4zNy02MS42My0xOS4xOC0xMC4xNy0xLjc3LTE2LjI2LTIuODMtMjQuNTMtMi44M2gtLjM0cy02Mi4zLjIzLTEwOC45MSw2NS4xMmMtLjI4LjM5LS41NS43Ny0uNTUuNzctNy4zMiwxMC44Ny0xMS45MSwyMC44OC0xNC44NiwyOC43OC0yLjU1LDYuNzktMy45NywxMi4yNi01LjU0LDE4LjI3LTEuNiw2LjE1LTIuNzksMTEuNjItNi4zMSwzMS4xNy0xLjE0LDYuMzYtMi42MSwxNC41OS00LjI3LDI0LjI1LDYuNC0xMC45MSwxNy4xMi0yNS45LDM0LjItMzkuMywxNC41OS0xMS40NSwyNy45OC0xNy4xMywzMy4wNi0xOS4xNiwyLjg1LTEuMTMsMTMuNzUtNS4zNSwyOC45NS03LjgsMy45My0uNjMsMTIuMTgtMS44LDIzLjItMS44LDguMTMsMCwxNy43Ny42MywyOC4zLDIuNTgsNi42NywxLjIzLDE2LjYxLDMuMTMsMjguNDEsOC4xNSw0LjAxLDEuNywxMS4yMSw1LjAzLDE4Ljg1LDksNC45MiwyLjU2LDguMzcsNC41MywxNC4xOCw3LjE0LDQuOSwyLjIxLDkuMDMsMy43NiwxMS43OCw0Ljc0LDAsMCwxOS4yMyw2LjM2LDQwLjI0LDYuOTguOTkuMDMsMS45Ni4wNCwyLjkxLjA0LDUuMzQsMCw5LjY4LS4zOSw5LjY4LS4zOSw2LjYtLjI2LDE1LjktMS4xOCwyNi41NS00LjIsMzkuMjUtMTEuMTQsNjEuNDQtNDEuMDMsNzQuMDctNTguMDIsNDkuOTUtNjcuMTksNDcuOTMtMTY3Ljg1LDQ3LjQxLTE4NC43Ny01LjE3LTcuMDItMTAuNDktMTMuODYtMTYuMDItMjAuNyIvPgogICAgPHBhdGggY2xhc3M9ImNscy00IiBkPSJNMjYzLjg1LDBjLS4xNywwLS4zMywwLS40OSwwLTkuNTYuMTMtMTguOTcsMy45OC01MS40NSwzMy4zNC0zNC4xLDMwLjgzLTQ4Ljk2LDQ4LjA2LTQ4Ljk2LDQ4LjA2LTQ1Ljg0LDUzLjEzLTY4Ljc3LDc5LjY5LTkyLjQ4LDEyMS40OS0zMC4zLDUzLjQxLTQ0LjkxLDEwMC4wOS01MS4yMywxMjMuMDItLjU4LDIuMS0xLjE1LDQuMzItMS43Myw2LjcxLDIuMzItNS40OSw0Ljk0LTExLjE5LDcuOS0xNy4wNCwxMy4zOS0yNi40MiwzNi42MS03Mi4yMSw4My45My04OC44NiwxMy41NC00Ljc2LDI1Ljk2LTYuMDYsMzMuNzktNi4zNywxLjQ3LS4wNiwyLjkxLS4wOCw0LjMtLjA4LDI3LjM5LDAsMzguNzMsMTAuODIsNzIsMTguODMsMTUuMjcsMy42NywzMS43OCw3Ljg4LDQ5LjUsNy44OCw5LjM2LDAsMTkuMDYtMS4xNywyOS4xLTQuMjIsMzYuNDktMTEuMDYsNTYuNDQtNDEuMzMsNjMuMjUtNTEuNTMsMTUuMzYtMjMuMDIsMTkuOTUtNDUuMjMsMjEuOS01NS4xNCwyLjQ3LTEyLjU3LDMuMTQtMjMuODUsMy4wNC0zMy4xNC02LjMyLTcuMzUtMTIuOTItMTQuOTEtMTkuODctMjIuODYsMCwwLTE3LjM5LTE5Ljg5LTUxLjk4LTQ5LjQ4QzI4Mi41MSwzLjM5LDI3Mi41LDAsMjYzLjg1LDAiLz4KICA8L2c+Cjwvc3ZnPg==" alt="Softmax website">
  </a>
</p>

The [Alignment League Benchmark (ALB)](https://www.softmax.com/alignmentleague) is a suite of multi-agent games, designed to measure how well AI agents align, coordinate, and collaborate with others (both AIs and humans).

CoGames is the games environment for ALB. You can use it to:

* create new games
* train agents to play existing ALB games
* submit those agents to the ALB leaderboard

There's one ALB game right now: Cogs vs Clips.

# Quick Start

## Step 1: Install CoGames

Install [cogames](https://pypi.org/project/cogames/) as a Python package.
```bash
pip install cogames
```

<details><summary>Using uv</summary>

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment
uv venv .venv
source .venv/bin/activate

# Install cogames
uv pip install cogames
```

</details>
<details><summary>Using Docker</summary>

```dockerfile
# Ensure Python 3.12 is available
FROM python:3.12-slim

# Ensure C/C++ compiler is available
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
  rm -rf /var/lib/apt/lists/*

# Install cogames
RUN pip install --no-cache-dir cogames
```

</details>
<details><summary>Using Colab</summary>

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Metta-AI/cogames/blob/main/README.ipynb)

</details>

## Step 2: Run your first game

Start with `cogames play`. This is the default command for running a local episode.
```bash
cogames play -m arena -p starter -r log -s 300
```

If your environment supports GUI rendering and you want the guided onboarding flow, run:
```bash
cogames tutorial play
```
The tutorial opens a new window, and the terminal gives you instructions for the training mission.

Command guide:

- `cogames play`: first local run; one episode with GUI, unicode, or log rendering
- `cogames scrimmage`: repeat one policy across many episodes
- `cogames pickup`: compare one policy against a pool and compute VOR; use this later, not as the default
  "play the game" command

## Step 3: Submit a policy to the leaderboard

1. Log into the ALB leaderboard with your GitHub account.

    ```bash
    cogames auth login
    ```

2. Create, upload, and submit a starter bundle.

    ```bash
    cogames create-bundle -p class=cogames.policy.starter_agent.StarterPolicy -o submission.zip
    cogames upload -p ./submission.zip -n "$USER.README-quickstart-starter-policy" --no-submit
    cogames submit "$USER.README-quickstart-starter-policy" --season beta-teams-small
    ```

3. Check your submission status.

    ```bash
    cogames submissions --season beta-teams-small --policy "$USER.README-quickstart-starter-policy"
    cogames season matches beta-teams-small --limit 5
    ```

# Tutorials

To learn more, see:

1. [Creating a policy](tutorials/01_MAKE_POLICY.md): Creating a custom policy and evaluating it
2. [Training](tutorials/02_TRAIN.md): Training a custom policy and evaluating it
3. [Submitting](tutorials/03_SUBMIT.md): Submitting to the leaderboard and understanding the results

If you want help, or to share your experience, join [the Discord](https://discord.gg/secret-hologenesis).

# About the game

CvC is a cooperative territory-control game. Teams of AI agents ("Cogs") work together to capture and defend
junctions against automated opponents ("Clips") by:

* gathering resources and depositing them at controlled junctions
* acquiring specialized roles (Miner, Aligner, Scrambler, Scout) at gear stations
* capturing neutral junctions using Aligners (costs 1 heart)
* disrupting enemy junctions using Scramblers (costs 1 heart)
* defending territory from Clips expansion

Read [MISSION.md](MISSION.md) for a thorough description of the game mechanics.

<p align="center">
  <img src="assets/cvc-reel.gif" alt="CvC reel">
<br>

There are many mission configurations available, with different map sizes, junction layouts, and game rules.

Overall, CvC aims to present rich environments with:

- **Territory control**: Capture and defend junctions to score points each tick
- **Role specialization**: Four roles (Miner, Aligner, Scrambler, Scout) with distinct capabilities and dependencies
- **Dense rewards**: Agents receive reward every tick proportional to territory controlled
- **Partial observability**: Agents have limited visibility of the environment
- **Required multi-agent cooperation**: No single role can succeed alone; Miners need Aligners to capture junctions,
  Aligners need Miners for resources, Scramblers must clear enemy territory for Aligners to advance

# About the tournament

We run multiple tournament seasons at a time, and each has a different structure. Our freeplay seasons are cheap to
submit to and evergreen: we give you indicative scores, showing how you play with others. Our team tournaments support
limited submissions and usually follow this pattern:

- **Play-ins**: New submissions are evaluated in one or more early stages, with eliminations based on stage results.
- **Team stages**: Surviving policies are sampled into teams and evaluated in repeated team matches.
- **Progressive culling**: Lower-performing teams/policies are removed across later stages.
- **Final scoring**: Remaining policies are ranked on the season leaderboard using season-specific scoring.

To inspect the exact rules for a specific season:
```bash
cogames season list
cogames season show <SEASON>
cogames season stages <SEASON>
cogames season progress <SEASON>
cogames season teams <SEASON>
cogames season leaderboard <SEASON>
cogames season leaderboard <SEASON> --pool <POOL> --type team
cogames season leaderboard <SEASON> --pool <POOL> --type score-policies
cogames season pool-config <SEASON> <POOL>
```

Note: `cogames season show` takes only one positional argument (`<SEASON>`).
Use separate subcommands (`stages`, `progress`, `teams`, `leaderboard`) for details.

## API Docs
The tournament API is documented at [api.observatory.softmax-research.net/docs](https://api.observatory.softmax-research.net/docs). The interactive
OpenAPI spec describes all public endpoints for seasons, matches, leaderboards, and submissions.

## Intended submit workflow (CLI)

Recommended end-to-end sequence once you are ready to submit:

```bash
# 1) Login
cogames auth login

# 2) Pick a season
cogames season list
cogames season show <SEASON>

# 3) Create a submission bundle
cogames create-bundle -p <POLICY_OR_CHECKPOINT> -o submission.zip [-f <EXTRA_PATH> ...] [--setup-script <SETUP_SCRIPT.py>]

# 4) Upload the bundle
cogames upload -p ./submission.zip -n <POLICY_NAME> --no-submit

# 5) Submit to the season
cogames submit <POLICY_NAME[:vN]> --season <SEASON>

# 6) Track status
cogames submissions --season <SEASON> --policy <POLICY_NAME>
cogames season matches <SEASON> --limit 20

# 7) Debug specific outcomes
cogames matches <MATCH_ID>
cogames match-artifacts <MATCH_ID>
cogames episode show <EPISODE_ID>
cogames episode replay <EPISODE_ID>
```

## Policy Secrets

If your policy needs credentials at runtime (e.g. an API key for an LLM provider), use `--secret-env` to attach
secret environment variables to your upload:

```bash
cogames upload \
  -p ./my_policy -n my-llm-policy \
  --secret-env ANTHROPIC_API_KEY=sk-ant-... \
  --secret-env OTHER_SECRET=value
```

See [POLICY_SECRETS.md](POLICY_SECRETS.md) for details on how secrets are stored, scoped, and cleaned up.

# Command Reference

To specify a `MISSION`, you can:

- Use a mission name from the registry given by `cogames missions` (e.g. `training_facility_1`).
- Use a path to a mission configuration file (e.g. `path/to/mission.yaml`).

To specify a `POLICY`, use one of two formats:

- **URI format** (for checkpoint bundles):

    - Point directly at a checkpoint bundle (directory or `.zip` containing `policy_spec.json`)
    - Examples: `./train_dir/my_run:v5`, `./train_dir/my_run:v5.zip`, `s3://bucket/path/run:v5.zip`
    - Use `:latest` suffix to auto-resolve the highest version: `./train_dir/checkpoints:latest`

- **Key-value format** (for explicit class + weights):

    - `class=`: Policy shorthand or full class path from `cogames policies`, e.g. `class=lstm` or
    `class=cogames.policy.random.RandomPolicy`.
    - `data=`: Optional path to a weights file (e.g., `weights.safetensors`). Must be a file, not a directory.
    - `proportion=`: Optional positive float specifying the relative share of agents that use this policy (default: 1.0).
    - `kw.<arg>=`: Optional policy `__init__` keyword arguments (all values parsed as strings).

You can view all the commands with
```bash
cogames --help
```
and you can view help for a given command with:
```bash
cogames [COMMAND] --help
```


## Play Commands



### `cogames play`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames play [OPTIONS]                                                                                     </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Play a game interactively.                                                                                        

 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">This runs a single episode of the game using one or more policies.</span>                                                

 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">By default, the policy is 'noop', so agents won't move unless manually controlled.</span>                                
 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">To see agents move by themselves, use `</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">--policy</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> class=random` or `</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">--policy</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> class=baseline`.</span>                       

 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Multiple </span><span style="color: #7fbf7f; text-decoration-color: #7fbf7f; font-weight: bold">-p</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> flags assign one policy per team (in team order).</span>                                                     

 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">You can manually control the actions of a specific cog by clicking on a cog</span>                                       
 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">in GUI mode or pressing M in unicode mode and using your arrow or WASD keys.</span>                                      
 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Log mode is non-interactive and doesn't support manual control.</span>                                                   

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Game Setup ────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--game</span>             <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">GAME   </span>  Game to play (default: cogsguard). <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: cogsguard]</span>                             <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--mission</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-m</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">MISSION</span>  Mission to play (run <span style="font-weight: bold">cogames missions</span> to list).                                     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--variant</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-v</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">VARIANT</span>  Apply variant modifier (repeatable).                                                <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--cogs</span>     <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-c</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">N      </span>  Number of cogs/agents. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: (from mission)]</span>                                    <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Policy ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--policy</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">POLICY</span>  Policy per team. One <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span> applies to all teams; multiple <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span> assigns one per team.       <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--device</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">DEVICE</span>  Policy device (auto, cpu, cuda, cuda:0, etc.). <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: auto]</span>                        <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Simulation ────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--steps</span>              <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">N                          </span>  Max steps per episode (note: <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span> is steps, not seed).  <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                                                           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: 1000]                                     </span>  <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--action-timeout-ms</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">MS [x&gt;=1</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">]</span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">                  </span>  Max ms per action before noop. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: 10000]</span>       <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--render</span>             <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-r</span>      <span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">[</span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">auto</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">|</span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">gui</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">|</span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">unicode</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">|</span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">log</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">|</span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">none</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">]</span>  <span style="font-weight: bold">auto</span>=gui when display is available, otherwise         <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                                                           unicode; <span style="font-weight: bold">gui</span>=MettaScope, <span style="font-weight: bold">unicode</span>=terminal,            <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                                                           <span style="font-weight: bold">log</span>=metrics only.                                     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                                                           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: auto]                                      </span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--seed</span>                       <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">INTEGER                    </span>  RNG seed for reproducibility (use <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--seed</span>, not <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span>).    <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                                                           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: 42]                                     </span>    <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--map-seed</span>                   <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">SEED                       </span>  Separate seed for procedural map generation.          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                                                           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: (same as --seed)]                 </span>          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--autostart</span>                  <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">                           </span>  Start simulation immediately without waiting for user <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                                                           input.                                                <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Output ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--save-replay-dir</span>         <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">DIR </span>  Save replay file for later viewing with <span style="font-weight: bold">cogames replay</span>.                         <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--save-replay-file</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">FILE</span>  Save replay to a fixed file path (overwrites existing file)                     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Other ─────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-h</span>        Show this message and exit.                                                                   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                                                                                                                   
 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Examples:</span>                                                                                                         
 <span style="color: #008080; text-decoration-color: #008080">cogames play </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-m</span><span style="color: #008080; text-decoration-color: #008080"> arena</span>                                  Interactive                                                
 <span style="color: #008080; text-decoration-color: #008080">cogames play </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-m</span><span style="color: #008080; text-decoration-color: #008080"> arena </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span><span style="color: #008080; text-decoration-color: #008080"> class=random</span>                  Random policy                                              
 <span style="color: #008080; text-decoration-color: #008080">cogames play </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-m</span><span style="color: #008080; text-decoration-color: #008080"> arena </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-c</span><span style="color: #008080; text-decoration-color: #008080"> 4 </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span><span style="color: #008080; text-decoration-color: #008080"> class=baseline</span>           Baseline, 4 cogs                                           
 <span style="color: #008080; text-decoration-color: #008080">cogames play </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-m</span><span style="color: #008080; text-decoration-color: #008080"> four_score </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span><span style="color: #008080; text-decoration-color: #008080"> nlanky </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span><span style="color: #008080; text-decoration-color: #008080"> baseline </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span><span style="color: #008080; text-decoration-color: #008080"> random </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span><span style="color: #008080; text-decoration-color: #008080"> noop</span>                                                
 One policy per team                                                                                               
 <span style="color: #008080; text-decoration-color: #008080">cogames play </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-m</span><span style="color: #008080; text-decoration-color: #008080"> four_score </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span><span style="color: #008080; text-decoration-color: #008080"> nlanky:1 </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span><span style="color: #008080; text-decoration-color: #008080"> random:2</span>     Mixed teams (cycling pattern)                              
 <span style="color: #008080; text-decoration-color: #008080">cogames play </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-m</span><span style="color: #008080; text-decoration-color: #008080"> arena </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--save-replay-file</span><span style="color: #008080; text-decoration-color: #008080"> ./latest.json.z</span> Overwrite fixed replay file                              
 <span style="color: #008080; text-decoration-color: #008080">cogames play </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-m</span><span style="color: #008080; text-decoration-color: #008080"> machina_1 </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-v</span><span style="color: #008080; text-decoration-color: #008080"> talk </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-r</span><span style="color: #008080; text-decoration-color: #008080"> gui</span>             Speech bubbles over cogs                                     
 <span style="color: #008080; text-decoration-color: #008080">cogames play </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-m</span><span style="color: #008080; text-decoration-color: #008080"> machina_1 </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-r</span><span style="color: #008080; text-decoration-color: #008080"> unicode</span>                   Terminal mode                                              

</pre>



    



### `cogames replay`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames replay [OPTIONS] FILE                                                                              </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Replay a saved game episode from a file in the GUI.                                                               

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #800000; text-decoration-color: #800000">*</span>    replay_path      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">FILE</span>  Path to a MettaGrid replay (.json.z, .replay, .bin) or a BitWorld replay            <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                             (.bitreplay).                                                                       <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                             <span style="color: #bf7f7f; text-decoration-color: #bf7f7f">[required]                                                                         </span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>      <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-h</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">     </span>  Show this message and exit.                                                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--duration</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">FLOAT</span>  Seconds to keep a BitWorld replay server alive. MettaScope replays ignore this       <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                            option.                                                                              <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                                                                                                                   
 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Examples:</span>                                                                                                         
 <span style="color: #008080; text-decoration-color: #008080">cogames replay ./replays/game.json.z</span>              Replay Cogs vs Clips in MettaScope                              
 <span style="color: #008080; text-decoration-color: #008080">cogames replay ./train_dir/my_run/replay.bin</span>      Replay a legacy MettaGrid run                                   
 <span style="color: #008080; text-decoration-color: #008080">cogames replay ./among_them.bitreplay</span>             Replay BitWorld in the global client                            

</pre>



    



## Evaluate Commands



### `cogames scrimmage`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames scrimmage [OPTIONS]                                                                                </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Evaluate a single policy controlling all agents.                                                                  

 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">This command is equivalent to running `cogames run` with a single policy.</span>                                         

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Mission ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--mission</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-m</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">MISSION</span>  Missions to evaluate (supports wildcards).                                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--cogs</span>     <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-c</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">N      </span>  Number of cogs (agents).                                                            <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--variant</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-v</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">VARIANT</span>  Mission variant (repeatable).                                                       <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Policy ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--policy</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">POLICY</span>  Policies to evaluate: (URI[,proportion=N] or NAME or                                  <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                           class=NAME[,data=FILE][,proportion=N]...).                                            <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--device</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">DEVICE</span>  Policy device (auto, cpu, cuda, cuda:0, etc.). <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: auto]</span>                        <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Simulation ────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--episodes</span>           <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-e</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">N [x&gt;=1</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">]</span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> </span>  Number of evaluation episodes. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: 10]</span>                            <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--steps</span>              <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">N [x&gt;=1</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">]</span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> </span>  Max steps per episode (note: <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span> is steps, not seed).                    <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                                         <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: (from mission)]                           </span>                    <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--seed</span>                       <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">N [x&gt;=0</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">]</span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> </span>  Seed for evaluation RNG (use <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--seed</span>, not <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span>). <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: 42]</span>             <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--map-seed</span>                   <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">N [x&gt;=0</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">]</span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> </span>  MapGen seed for procedural maps. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: (same as --seed)]</span>            <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--action-timeout-ms</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">MS [x&gt;=1</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">]</span>  Max ms per action before noop. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: 250]</span>                           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Output ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--format</span>                 <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">FMT</span>  Output format: yaml or json.                                                      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--save-replay-dir</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">DIR</span>  Directory to save replays.                                                        <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Other ─────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-h</span>        Show this message and exit.                                                                   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                                                                                                                   
 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Examples:</span>                                                                                                         
 <span style="color: #008080; text-decoration-color: #008080">cogames scrimmage </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-m</span><span style="color: #008080; text-decoration-color: #008080"> arena </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span><span style="color: #008080; text-decoration-color: #008080"> lstm</span>                          Single policy eval                                    

</pre>



    



### `cogames run`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames run [OPTIONS]                                                                                      </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Evaluate one or more policies on missions.                                                                        

 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">With multiple policies (e.g., 2 policies, 4 agents), each policy always controls 2 agents,</span>                        
 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">but which agents swap between policies each episode.</span>                                                              

 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">With one policy, this command is equivalent to `cogames scrimmage`.</span>                                               

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Mission ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--mission</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-m</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">MISSION</span>  Missions to evaluate (supports wildcards).                                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--cogs</span>     <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-c</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">N      </span>  Number of cogs (agents).                                                            <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--variant</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-v</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">VARIANT</span>  Mission variant (repeatable).                                                       <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Policy ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--policy</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">POLICY</span>  Policies to evaluate: (URI[,proportion=N] or NAME or                                  <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                           class=NAME[,data=FILE][,proportion=N]...).                                            <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--device</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">DEVICE</span>  Policy device (auto, cpu, cuda, cuda:0, etc.). <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: auto]</span>                        <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Simulation ────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--episodes</span>           <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-e</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">N [x&gt;=1</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">]</span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> </span>  Number of evaluation episodes. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: 10]</span>                            <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--steps</span>              <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">N [x&gt;=1</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">]</span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> </span>  Max steps per episode (note: <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span> is steps, not seed).                    <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                                         <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: (from mission)]                           </span>                    <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--seed</span>                       <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">N [x&gt;=0</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">]</span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> </span>  Seed for evaluation RNG (use <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--seed</span>, not <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span>). <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: 42]</span>             <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--map-seed</span>                   <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">N [x&gt;=0</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">]</span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> </span>  MapGen seed for procedural maps. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: (same as --seed)]</span>            <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--action-timeout-ms</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">MS [x&gt;=1</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">]</span>  Max ms per action before noop. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: 250]</span>                           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Output ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--format</span>                 <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">FMT</span>  Output format: yaml or json.                                                      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--save-replay-dir</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">DIR</span>  Directory to save replays.                                                        <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Other ─────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-h</span>        Show this message and exit.                                                                   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                                                                                                                   
 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Examples:</span>                                                                                                         
 <span style="color: #008080; text-decoration-color: #008080">cogames run </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-m</span><span style="color: #008080; text-decoration-color: #008080"> arena </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span><span style="color: #008080; text-decoration-color: #008080"> lstm</span>                         Evaluate single policy                                       
 <span style="color: #008080; text-decoration-color: #008080">cogames run </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-m</span><span style="color: #008080; text-decoration-color: #008080"> machina_1 </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span><span style="color: #008080; text-decoration-color: #008080"> ./train_dir/my_run:v5</span>     Evaluate a checkpoint bundle                                
 <span style="color: #008080; text-decoration-color: #008080">cogames run </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-m</span><span style="color: #008080; text-decoration-color: #008080"> 'arena.*' </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span><span style="color: #008080; text-decoration-color: #008080"> lstm </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span><span style="color: #008080; text-decoration-color: #008080"> random </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-e</span><span style="color: #008080; text-decoration-color: #008080"> 20</span>            Evaluate multiple policies together                   
 <span style="color: #008080; text-decoration-color: #008080">cogames run </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-m</span><span style="color: #008080; text-decoration-color: #008080"> machina_1 </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span><span style="color: #008080; text-decoration-color: #008080"> ./train_dir/my_run:v5,proportion=3 </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span><span style="color: #008080; text-decoration-color: #008080"> class=random,proportion=5</span>                       
 Evaluate policies in 3:5 mix                                                                                      

</pre>



    



### `cogames pickup`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames pickup [OPTIONS]                                                                                   </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Evaluate a policy against a pool of other policies and compute VOR.                                               

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Mission ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--mission</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-m</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">MISSION </span>  Mission to evaluate on. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: arena]</span>                                           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--cogs</span>     <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-c</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">N [x&gt;=1</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">]</span>  Number of cogs (agents). <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: 4]</span>                                              <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--variant</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-v</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">VARIANT </span>  Mission variant (repeatable).                                                      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Policy ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--policy</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">POLICY</span>  Candidate policy to evaluate.                                                         <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--pool</span>            <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">POLICY</span>  Pool policy (repeatable).                                                             <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--device</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">DEVICE</span>  Policy device (auto, cpu, cuda, cuda:0, etc.). <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: auto]</span>                        <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Simulation ────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--episodes</span>           <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-e</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">N [x&gt;=1</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">]</span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> </span>  Episodes per scenario. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: 1]</span>                                     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--steps</span>              <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">N [x&gt;=1</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">]</span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> </span>  Max steps per episode (note: <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span> is steps, not seed). <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: 1000]</span>    <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--seed</span>                       <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">N [x&gt;=0</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">]</span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> </span>  Base random seed (use <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--seed</span>, not <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span>). <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: 50]</span>                    <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--map-seed</span>                   <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">N [x&gt;=0</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">]</span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold"> </span>  MapGen seed for procedural maps. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: (same as --seed)]</span>            <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--action-timeout-ms</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">MS [x&gt;=1</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">]</span>  Max ms per action before noop. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: 250]</span>                           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Output ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--save-replay-dir</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">DIR</span>  Directory to save replays.                                                        <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Other ─────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-h</span>        Show this message and exit.                                                                   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                                                                                                                   
 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Examples:</span>                                                                                                         
 <span style="color: #008080; text-decoration-color: #008080">cogames pickup </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span><span style="color: #008080; text-decoration-color: #008080"> greedy </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--pool</span><span style="color: #008080; text-decoration-color: #008080"> random</span>                      Test greedy against pool of random                    

</pre>



    



### `cogames diagnose`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames diagnose [OPTIONS]                                                                                 </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Run diagnostic evals for a policy checkpoint.                                                                     

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                                                                                                                   
 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Examples:</span>                                                                                                         
 <span style="color: #008080; text-decoration-color: #008080">cogames diagnose ./train_dir/my_run</span>                         Default CvC evals                                     
 <span style="color: #008080; text-decoration-color: #008080">cogames diagnose lstm </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--scripted-baseline-policy</span><span style="color: #008080; text-decoration-color: #008080"> scripted.basic</span>   Compare against scripted baseline               
 <span style="color: #008080; text-decoration-color: #008080">cogames diagnose lstm </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--known-strong-policy</span><span style="color: #008080; text-decoration-color: #008080"> my_best_policy</span>     Normalize against known-strong policy              
 <span style="color: #008080; text-decoration-color: #008080">cogames diagnose lstm </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--compare-run-dir</span><span style="color: #008080; text-decoration-color: #008080"> outputs/cogames-diagnose/prev_run</span>  Stability comparison                   

</pre>



    



## Info Commands



### `cogames version`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames version [OPTIONS]                                                                                  </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Show version information for cogames and dependencies.                                                            

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>          Show this message and exit.                                                                     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    



### `cogames docs`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames docs [OPTIONS] DOC                                                                                 </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Print documentation (run without arguments to see available docs).                                                

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>   doc_name      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">DOC</span>  Document name (readme, mission, technical_manual, scripted_agent, evals, mapgen).          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-h</span>        Show this message and exit.                                                                   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                                                                                                                   
 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Examples:</span>                                                                                                         
 <span style="color: #008080; text-decoration-color: #008080">cogames docs</span>                             List available documents                                                 
 <span style="color: #008080; text-decoration-color: #008080">cogames docs readme</span>                      Print README                                                             
 <span style="color: #008080; text-decoration-color: #008080">cogames docs mission</span>                     Print mission briefing                                                   

</pre>



    



## Policies Commands



### `cogames policies`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames policies [OPTIONS]                                                                                 </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Show available policy shorthand names.                                                                            

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>          Show this message and exit.                                                                     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                                                                                                                   
 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Usage:</span>                                                                                                            
 Use these shorthand names with <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--policy</span> or <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span>:                                                                    
 <span style="color: #008080; text-decoration-color: #008080">cogames play </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-m</span><span style="color: #008080; text-decoration-color: #008080"> arena </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span><span style="color: #008080; text-decoration-color: #008080"> class=random</span>     Use random policy                                                       
 <span style="color: #008080; text-decoration-color: #008080">cogames play </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-m</span><span style="color: #008080; text-decoration-color: #008080"> arena </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span><span style="color: #008080; text-decoration-color: #008080"> class=baseline</span>   Use baseline policy                                                     

</pre>



    



### `cogames create-bundle`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames create-bundle [OPTIONS]                                                                            </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Create a submission bundle zip from a policy.                                                                     

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Policy ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #800000; text-decoration-color: #800000">*</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--policy</span>      <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">POLICY </span>  Policy specification: URI (bundle dir or .zip) or NAME or                     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                                   class=NAME[,data=FILE][,kw.x=val].                                            <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                                   <span style="color: #bf7f7f; text-decoration-color: #bf7f7f">[required]                                                                   </span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>    <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--init-kwarg</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-k</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">KEY=VAL</span>  Policy init kwargs (can be repeated).                                         <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Output ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--output</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-o</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">PATH</span>  Output path for the bundle zip. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: submission.zip]</span>                               <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Files ─────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--include-files</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-f</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">PATH</span>  Files or directories to include (can be repeated).                               <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--setup-script</span>           <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">PATH</span>  Python setup script to include in the bundle.                                    <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Other ─────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-h</span>        Show this message and exit.                                                                   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                                                                                                                   
 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Examples:</span>                                                                                                         
 <span style="color: #008080; text-decoration-color: #008080">cogames create-bundle </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span><span style="color: #008080; text-decoration-color: #008080"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">&lt;POLICY_OR_CHECKPOINT&gt;</span><span style="color: #008080; text-decoration-color: #008080"> </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-o</span><span style="color: #008080; text-decoration-color: #008080"> submission.zip</span>   Create a submission bundle                    
 <span style="color: #008080; text-decoration-color: #008080">cogames create-bundle </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span><span style="color: #008080; text-decoration-color: #008080"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">&lt;POLICY_OR_CHECKPOINT&gt;</span><span style="color: #008080; text-decoration-color: #008080"> </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-o</span><span style="color: #008080; text-decoration-color: #008080"> submission.zip   </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-f</span><span style="color: #008080; text-decoration-color: #008080"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">&lt;EXTRA_PATH&gt;</span><span style="color: #008080; text-decoration-color: #008080"> ... </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--setup-script</span><span style="color: #008080; text-decoration-color: #008080"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">&lt;SETUP.py&gt;</span> 
 Include extra runtime files or setup when needed                                                                  

</pre>



    



### `cogames validate-bundle`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames validate-bundle [OPTIONS]                                                                          </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Validate a policy bundle runs correctly in Docker.                                                                

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #800000; text-decoration-color: #800000">*</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--policy</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URI</span>  Bundle URI (file://, s3://, or local path to .zip or directory). <span style="color: #bf7f7f; text-decoration-color: #bf7f7f">[required]</span>           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Tournament ────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--season</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">SEASON</span>  Tournament season (determines which game to validate against).                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Server ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--server</span>              <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Tournament server URL (used to resolve default season).                              <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                            <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://api.observatory.softmax-research.net]</span>                              <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--login-server</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Authentication server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://softmax.com/api]</span>                        <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Validation ────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--image</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">TEXT</span>  Docker image for container validation. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: ghcr.io/metta-ai/episode-runner:latest]</span>   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Other ─────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-h</span>        Show this message and exit.                                                                   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    



## Tournament Commands



### `cogames submissions`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames submissions [OPTIONS]                                                                              </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Show your uploads and tournament submissions.                                                                     

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Filter ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--policy</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">POLICY</span>  Filter by policy name (e.g., 'my-policy' or 'my-policy:v3').                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--season</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">SEASON</span>  Filter by tournament season.                                                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Server ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--login-server</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Authentication server URL <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://softmax.com/api]</span>                       <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--server</span>        <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Tournament server URL <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://api.observatory.softmax-research.net]</span>      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Output ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--json</span>          Print raw JSON instead of table.                                                                <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Other ─────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-h</span>        Show this message and exit.                                                                   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                                                                                                                   
 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Examples:</span>                                                                                                         
 <span style="color: #008080; text-decoration-color: #008080">cogames submissions</span>                         All your uploads                                                      
 <span style="color: #008080; text-decoration-color: #008080">cogames submissions </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--season</span><span style="color: #008080; text-decoration-color: #008080"> beta-cvc</span>           Submissions in a season                                           
 <span style="color: #008080; text-decoration-color: #008080">cogames submissions </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span><span style="color: #008080; text-decoration-color: #008080"> my-policy</span>            Info on a specific policy                                             

</pre>



    



### `cogames leaderboard`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames leaderboard [OPTIONS] SEASON                                                                       </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Show tournament leaderboard for a season.                                                                         

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>   season_arg      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">SEASON</span>  Tournament season name (positional shorthand for <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--season</span>).                           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Tournament ────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--season</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">SEASON</span>  Tournament season name (default: server default).                                       <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Filter ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--policy</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-P</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">POLICY</span>  Filter by policy name (e.g., 'slanky' or 'slanky:v88').                               <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--mine</span>    <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-M</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>  Show only your own policies (requires auth).                                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Server ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--login-server</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Authentication server URL <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://softmax.com/api]</span>                       <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--server</span>        <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Tournament server URL <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://api.observatory.softmax-research.net]</span>      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Output ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--json</span>          Print raw JSON instead of table.                                                                <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Other ─────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-h</span>        Show this message and exit.                                                                   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                                                                                                                   
 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Examples:</span>                                                                                                         
 <span style="color: #008080; text-decoration-color: #008080">cogames leaderboard beta-cvc</span>                          View rankings (positional season)                           
 <span style="color: #008080; text-decoration-color: #008080">cogames leaderboard </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--season</span><span style="color: #008080; text-decoration-color: #008080"> beta-cvc</span>                 View rankings (option)                                      
 <span style="color: #008080; text-decoration-color: #008080">cogames leaderboard beta-cvc </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--policy</span><span style="color: #008080; text-decoration-color: #008080"> slanky</span>          Filter by policy name                                       
 <span style="color: #008080; text-decoration-color: #008080">cogames leaderboard beta-cvc </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--mine</span>                   Show only your policies                                     

</pre>



    



### `cogames matches`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames matches [OPTIONS] MATCH_ID                                                                         </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Show your recent matches and policy logs.                                                                         

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>   match_id      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">TEXT</span>  Match ID to show details for. If omitted, lists recent matches.                           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Filter ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--season</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">SEASON</span>  Tournament season (for listing matches).                                              <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--policy</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-P</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">POLICY</span>  Filter matches to those involving a specific policy (e.g., 'slanky' or 'slanky:v88'). <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--limit</span>   <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-n</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">N     </span>  Number of matches to show. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: 20]</span>                                              <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Output ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--logs</span>           <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-l</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>  Show available policy logs for the match.                                         <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--download-logs</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-d</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">DIR</span>  Download all accessible logs to directory.                                        <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--json</span>                   <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>  Print raw JSON instead of table.                                                  <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Server ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--login-server</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Authentication server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://softmax.com/api]</span>                        <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--server</span>              <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Tournament server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://api.observatory.softmax-research.net]</span>       <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Other ─────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-h</span>        Show this message and exit.                                                                   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                                                                                                                   
 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Examples:</span>                                                                                                         
 <span style="color: #008080; text-decoration-color: #008080">cogames matches</span>                              List recent matches                                                  
 <span style="color: #008080; text-decoration-color: #008080">cogames matches </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--policy</span><span style="color: #008080; text-decoration-color: #008080"> slanky</span>               Filter by policy name                                               
 <span style="color: #008080; text-decoration-color: #008080">cogames matches </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">&lt;match-id&gt;</span>                   Show match details                                                   
 <span style="color: #008080; text-decoration-color: #008080">cogames matches </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">&lt;match-id&gt;</span><span style="color: #008080; text-decoration-color: #008080"> </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--logs</span>            Show available logs                                                  
 <span style="color: #008080; text-decoration-color: #008080">cogames match-artifacts </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">&lt;match-id&gt;</span><span style="color: #008080; text-decoration-color: #008080"> error-info</span>  Show runner error info                                             
 <span style="color: #008080; text-decoration-color: #008080">cogames matches </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">&lt;match-id&gt;</span><span style="color: #008080; text-decoration-color: #008080"> </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-d</span><span style="color: #008080; text-decoration-color: #008080"> ./logs</span>         Download logs                                                        

</pre>



    



### `cogames match-artifacts`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames match-artifacts [OPTIONS] MATCH_ID ARTIFACT_TYPE                                                   </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Retrieve artifacts for a match (logs, etc.).                                                                      

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #800000; text-decoration-color: #800000">*</span>    match_id           <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">TEXT</span>  Match ID to fetch artifacts for. <span style="color: #bf7f7f; text-decoration-color: #bf7f7f">[required]</span>                                       <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>      artifact_type      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">TEXT</span>  Type of artifact to retrieve (e.g. 'logs', 'error-info'). <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: logs]</span>         <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Filter ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--policy</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">POLICY_VERSION_ID</span>  Policy version ID. If omitted, uses your first policy in the match.        <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Output ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--output</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-o</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">FILE</span>  Save artifact to file.                                                                  <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Server ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--login-server</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Authentication server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://softmax.com/api]</span>                        <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--server</span>              <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Tournament server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://api.observatory.softmax-research.net]</span>       <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Other ─────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-h</span>        Show this message and exit.                                                                   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                                                                                                                   
 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Examples:</span>                                                                                                         
 <span style="color: #008080; text-decoration-color: #008080">cogames match-artifacts </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">&lt;match-id&gt;</span>                     Get match logs                                             
 <span style="color: #008080; text-decoration-color: #008080">cogames match-artifacts </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">&lt;match-id&gt;</span><span style="color: #008080; text-decoration-color: #008080"> error-info</span>          Get runner error info                                      
 <span style="color: #008080; text-decoration-color: #008080">cogames match-artifacts </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">&lt;match-id&gt;</span><span style="color: #008080; text-decoration-color: #008080"> logs </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-o</span><span style="color: #008080; text-decoration-color: #008080"> out.txt</span>     Save to file                                               

</pre>



    



### `cogames upload`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames upload [OPTIONS]                                                                                   </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Upload a policy to CoGames.                                                                                       

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Upload ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #800000; text-decoration-color: #800000">*</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--name</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-n</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">NAME</span>  Name for your uploaded policy. <span style="color: #bf7f7f; text-decoration-color: #bf7f7f">[required]</span>                                              <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Policy ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #800000; text-decoration-color: #800000">*</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--policy</span>      <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">POLICY </span>  Policy specification: URI (bundle dir or .zip) or NAME or                     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                                   class=NAME[,data=FILE][,kw.x=val].                                            <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                                   <span style="color: #bf7f7f; text-decoration-color: #bf7f7f">[required]                                                                   </span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>    <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--init-kwarg</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-k</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">KEY=VAL</span>  Policy init kwargs (can be repeated).                                         <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Files ─────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--include-files</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-f</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">PATH</span>  Files or directories to include (can be repeated).                               <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--setup-script</span>           <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">PATH</span>  Python setup script to run before loading the policy.                            <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Secrets ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--secret-env</span>         <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">KEY=VALUE</span>  Secret environment variable for policy execution (can be repeated). Stored in   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                                 AWS Secrets Manager.                                                            <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--use-bedrock</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">         </span>  Enable AWS Bedrock access for this policy. Sets USE_BEDROCK=true in policy      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                                 environment.                                                                    <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Tournament ────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--season</span>           <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">SEASON</span>  Tournament season (default: server's default season).                                <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--no-submit</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">      </span>  Upload without submitting to a season.                                               <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Validation ────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--dry-run</span>                <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    </span>  Run the Docker smoke test only without uploading.                                <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--skip-validation</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    </span>  Skip the Docker smoke test.                                                      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--image</span>                  <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">TEXT</span>  Docker image for container validation.                                           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                                <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: ghcr.io/metta-ai/episode-runner:latest]</span>                                <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Server ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--login-server</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Authentication server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://softmax.com/api]</span>                        <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--server</span>              <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Tournament server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://api.observatory.softmax-research.net]</span>       <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Other ─────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-h</span>        Show this message and exit.                                                                   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                                                                                                                   
 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Examples:</span>                                                                                                         
 <span style="color: #008080; text-decoration-color: #008080">cogames upload </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span><span style="color: #008080; text-decoration-color: #008080"> ./submission.zip </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-n</span><span style="color: #008080; text-decoration-color: #008080"> my-policy </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--no-submit</span>   Upload a submission bundle without submitting       
 <span style="color: #008080; text-decoration-color: #008080">cogames upload </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span><span style="color: #008080; text-decoration-color: #008080"> ./submission.zip </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-n</span><span style="color: #008080; text-decoration-color: #008080"> my-policy </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--dry-run</span>   Validate a submission bundle locally without          
 uploading                                                                                                         

</pre>



    



### `cogames submit`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames submit [OPTIONS] POLICY                                                                            </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Submit a policy to a tournament season.                                                                           

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #800000; text-decoration-color: #800000">*</span>    policy_name      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">POLICY</span>  Policy name (e.g., 'my-policy' or 'my-policy:v3' for specific version).           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                               <span style="color: #bf7f7f; text-decoration-color: #bf7f7f">[required]                                                             </span>           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Tournament ────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--season</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">SEASON</span>  Tournament season name.                                                                 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Server ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--login-server</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Authentication server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://softmax.com/api]</span>                      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--server</span>        <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Tournament server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://api.observatory.softmax-research.net]</span>     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Other ─────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-h</span>        Show this message and exit.                                                                   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                                                                                                                   
 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Examples:</span>                                                                                                         
 <span style="color: #008080; text-decoration-color: #008080">cogames submit my-policy</span>                                   Submit to default season                               
 <span style="color: #008080; text-decoration-color: #008080">cogames submit my-policy:v3 </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--season</span><span style="color: #008080; text-decoration-color: #008080"> beta-cvc</span>              Submit specific version to specific season             

</pre>



    



### `cogames ship`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames ship [OPTIONS]                                                                                     </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Bundle, validate, upload, and submit a policy in one command.                                                     

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Upload ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #800000; text-decoration-color: #800000">*</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--name</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-n</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">NAME</span>  Name for your uploaded policy. <span style="color: #bf7f7f; text-decoration-color: #bf7f7f">[required]</span>                                              <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Policy ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #800000; text-decoration-color: #800000">*</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--policy</span>      <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">POLICY </span>  Policy specification: URI (bundle dir or .zip) or NAME or                     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                                   class=NAME[,data=FILE][,kw.x=val].                                            <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                                   <span style="color: #bf7f7f; text-decoration-color: #bf7f7f">[required]                                                                   </span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>    <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--init-kwarg</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-k</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">KEY=VAL</span>  Policy init kwargs (can be repeated).                                         <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Files ─────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--include-files</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-f</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">PATH</span>  Files or directories to include (can be repeated).                               <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--setup-script</span>           <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">PATH</span>  Python setup script to run before loading the policy.                            <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Tournament ────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--season</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">SEASON</span>  Tournament season (default: server's default season).                                   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Validation ────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--dry-run</span>                <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    </span>  Run the Docker smoke test only without uploading.                                <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--skip-validation</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    </span>  Skip the Docker smoke test.                                                      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--image</span>                  <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">TEXT</span>  Docker image for container validation.                                           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                                <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: ghcr.io/metta-ai/episode-runner:latest]</span>                                <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Server ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--login-server</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Authentication server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://softmax.com/api]</span>                        <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--server</span>              <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Tournament server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://api.observatory.softmax-research.net]</span>       <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Other ─────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-h</span>        Show this message and exit.                                                                   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                                                                                                                   
 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Examples:</span>                                                                                                         
 <span style="color: #008080; text-decoration-color: #008080">cogames ship </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span><span style="color: #008080; text-decoration-color: #008080"> ./submission.zip </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-n</span><span style="color: #008080; text-decoration-color: #008080"> my-policy </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--season</span><span style="color: #008080; text-decoration-color: #008080"> beta-cvc</span>   Ship a prepared submission bundle               
 <span style="color: #008080; text-decoration-color: #008080">cogames ship </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span><span style="color: #008080; text-decoration-color: #008080"> ./submission.zip </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-n</span><span style="color: #008080; text-decoration-color: #008080"> my-policy </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--dry-run</span>   Validate a prepared submission bundle locally           

</pre>



    



### `cogames auth login`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames auth login [OPTIONS]                                                                               </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Log in to CoGames.                                                                                                

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--login-server</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Authentication server URL <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://softmax.com/api]</span>                       <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--no-browser</span>            <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>  Skip opening browser automatically.                                                <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--force</span>         <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-f</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>  Re-authenticate even if already logged in                                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>                  <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>  Show this message and exit.                                                        <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    



### `cogames auth logout`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames auth logout [OPTIONS]                                                                              </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Log out of CoGames.                                                                                               

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--login-server</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Authentication server URL <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://softmax.com/api]</span>                         <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>                <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>  Show this message and exit.                                                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    



### `cogames auth get-login-url`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames auth get-login-url [OPTIONS]                                                                       </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Print the CoGames login URL.                                                                                      

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--login-server</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Authentication server URL <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://softmax.com/api]</span>                         <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>                <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>  Show this message and exit.                                                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    



### `cogames auth status`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames auth status [OPTIONS]                                                                              </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Show CoGames authentication status.                                                                               

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--login-server</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Authentication server URL <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://softmax.com/api]</span>                       <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--server</span>        <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  API server URL for /whoami verification.                                           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>                  <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>  Show this message and exit.                                                        <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    



### `cogames auth get-token`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames auth get-token [OPTIONS]                                                                           </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Print the saved CoGames token.                                                                                    

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--login-server</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Authentication server URL <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://softmax.com/api]</span>                         <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>                <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>  Show this message and exit.                                                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    



### `cogames auth set-token`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames auth set-token [OPTIONS] TOKEN                                                                     </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Save a CoGames token.                                                                                             

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #800000; text-decoration-color: #800000">*</span>    token      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">TEXT</span>  Bearer token to save <span style="color: #bf7f7f; text-decoration-color: #bf7f7f">[required]</span>                                                           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--login-server</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Authentication server URL <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://softmax.com/api]</span>                         <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>                <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>  Show this message and exit.                                                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    



### `cogames season list`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames season list [OPTIONS]                                                                              </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> List tournament seasons.                                                                                          

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>          Show this message and exit.                                                                     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Server ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--login-server</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Authentication server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://softmax.com/api]</span>                      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--server</span>        <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Tournament server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://api.observatory.softmax-research.net]</span>     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Output ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--json</span>          Print raw JSON.                                                                                 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    



### `cogames season show`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames season show [OPTIONS] SEASON                                                                       </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Show details for a season.                                                                                        

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #800000; text-decoration-color: #800000">*</span>    season_name      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">SEASON</span>  Season name. <span style="color: #bf7f7f; text-decoration-color: #bf7f7f">[required]</span>                                                           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>          Show this message and exit.                                                                     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Server ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--login-server</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Authentication server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://softmax.com/api]</span>                      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--server</span>        <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Tournament server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://api.observatory.softmax-research.net]</span>     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Output ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--json</span>          Print raw JSON.                                                                                 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    



### `cogames season versions`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames season versions [OPTIONS] SEASON                                                                   </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> List versions of a season.                                                                                        

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #800000; text-decoration-color: #800000">*</span>    season_name      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">SEASON</span>  Season name. <span style="color: #bf7f7f; text-decoration-color: #bf7f7f">[required]</span>                                                           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>          Show this message and exit.                                                                     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Server ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--login-server</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Authentication server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://softmax.com/api]</span>                      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--server</span>        <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Tournament server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://api.observatory.softmax-research.net]</span>     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Output ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--json</span>          Print raw JSON.                                                                                 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    



### `cogames season stages`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames season stages [OPTIONS] SEASON                                                                     </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Show stages for a season.                                                                                         

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #800000; text-decoration-color: #800000">*</span>    season_name      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">SEASON</span>  Season name. <span style="color: #bf7f7f; text-decoration-color: #bf7f7f">[required]</span>                                                           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>          Show this message and exit.                                                                     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Server ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--login-server</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Authentication server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://softmax.com/api]</span>                      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--server</span>        <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Tournament server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://api.observatory.softmax-research.net]</span>     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Output ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--json</span>          Print raw JSON.                                                                                 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    



### `cogames season progress`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames season progress [OPTIONS] SEASON                                                                   </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Show season progress summary.                                                                                     

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #800000; text-decoration-color: #800000">*</span>    season_name      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">SEASON</span>  Season name. <span style="color: #bf7f7f; text-decoration-color: #bf7f7f">[required]</span>                                                           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>          Show this message and exit.                                                                     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Server ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--login-server</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Authentication server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://softmax.com/api]</span>                      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--server</span>        <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Tournament server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://api.observatory.softmax-research.net]</span>     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Output ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--json</span>          Print raw JSON.                                                                                 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    



### `cogames season leaderboard`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames season leaderboard [OPTIONS] SEASON                                                                </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Show season leaderboard.                                                                                          

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>   season_name      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">SEASON</span>  Season name (default: server default).                                               <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--pool</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">TEXT                        </span>  Pool name for stage-specific leaderboard.                         <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--type</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-t</span>      <span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">[</span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">policy</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">|</span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">team</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">|</span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">score-policies</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">]</span>  Leaderboard type (policy, team, score-policies).                  <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                                               <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: policy]                               </span>                  <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">                            </span>  Show this message and exit.                                       <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Server ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--login-server</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Authentication server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://softmax.com/api]</span>                      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--server</span>        <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Tournament server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://api.observatory.softmax-research.net]</span>     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Output ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--json</span>          Print raw JSON.                                                                                 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    



### `cogames season teams`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames season teams [OPTIONS] SEASON                                                                      </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Show teams in a season.                                                                                           

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #800000; text-decoration-color: #800000">*</span>    season_name      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">SEASON</span>  Season name. <span style="color: #bf7f7f; text-decoration-color: #bf7f7f">[required]</span>                                                           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>          Show this message and exit.                                                                     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Server ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--login-server</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Authentication server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://softmax.com/api]</span>                      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--server</span>        <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Tournament server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://api.observatory.softmax-research.net]</span>     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Output ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--json</span>          Print raw JSON.                                                                                 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    



### `cogames season matches`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames season matches [OPTIONS] SEASON                                                                    </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Show matches in a season.                                                                                         

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #800000; text-decoration-color: #800000">*</span>    season_name      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">SEASON</span>  Season name. <span style="color: #bf7f7f; text-decoration-color: #bf7f7f">[required]</span>                                                           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--limit</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-n</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">INTEGER</span>  Number of matches to show. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: 20]</span>                                              <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>           <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">       </span>  Show this message and exit.                                                           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Server ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--login-server</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Authentication server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://softmax.com/api]</span>                      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--server</span>        <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Tournament server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://api.observatory.softmax-research.net]</span>     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Output ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--json</span>          Print raw JSON.                                                                                 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    



### `cogames season pool-config`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames season pool-config [OPTIONS] SEASON POOL                                                           </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Show configuration for a season pool.                                                                             

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #800000; text-decoration-color: #800000">*</span>    season_name      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">SEASON</span>  Season name. <span style="color: #bf7f7f; text-decoration-color: #bf7f7f">[required]</span>                                                           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #800000; text-decoration-color: #800000">*</span>    pool_name        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">POOL  </span>  Pool name. <span style="color: #bf7f7f; text-decoration-color: #bf7f7f">[required]</span>                                                             <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>          Show this message and exit.                                                                     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Server ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--login-server</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Authentication server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://softmax.com/api]</span>                      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--server</span>        <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Tournament server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://api.observatory.softmax-research.net]</span>     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Output ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--json</span>          Print raw JSON.                                                                                 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    



### `cogames episode list`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames episode list [OPTIONS]                                                                             </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> List game episodes.                                                                                               

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Filter ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--policy</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">POLICY_VERSION_ID</span>  Filter by policy version ID.                                               <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--tags</span>    <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-t</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">TAGS             </span>  Comma-separated key:value tag filters.                                     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--limit</span>   <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-n</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">N                </span>  Number of episodes to show. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: 20]</span>                                  <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Server ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--server</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Tournament server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://api.observatory.softmax-research.net]</span>             <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Output ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--json</span>          Print raw JSON instead of table.                                                                <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Other ─────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-h</span>        Show this message and exit.                                                                   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    



### `cogames episode show`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames episode show [OPTIONS] EPISODE_ID                                                                  </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Show details for a specific episode.                                                                              

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #800000; text-decoration-color: #800000">*</span>    episode_id      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">TEXT</span>  Episode ID to show. <span style="color: #bf7f7f; text-decoration-color: #bf7f7f">[required]</span>                                                       <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Server ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--server</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Tournament server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://api.observatory.softmax-research.net]</span>             <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Output ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--json</span>          Print raw JSON instead of table.                                                                <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Other ─────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-h</span>        Show this message and exit.                                                                   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    



### `cogames episode replay`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames episode replay [OPTIONS] EPISODE_ID                                                                </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Download and replay a game episode in MettaScope or the BitWorld global client.                                   

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #800000; text-decoration-color: #800000">*</span>    episode_id      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">TEXT</span>  Episode ID to replay. <span style="color: #bf7f7f; text-decoration-color: #bf7f7f">[required]</span>                                                     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Output ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--output</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-o</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">FILE</span>  Save replay to file instead of launching viewer.                                        <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Server ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--server</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Tournament server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://api.observatory.softmax-research.net]</span>             <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Other ─────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-h</span>        Show this message and exit.                                                                   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    



### `cogames assay status`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames assay status [OPTIONS] POLICY_OR_RUN_ID                                                            </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Show status of an assay run or the latest run for a policy.                                                       

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #800000; text-decoration-color: #800000">*</span>    policy_or_run_id      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">TEXT</span>  Assay run UUID, or policy name (name[:version]). <span style="color: #bf7f7f; text-decoration-color: #bf7f7f">[required]</span>                    <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>          Show this message and exit.                                                                     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Server ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--login-server</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Authentication server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://softmax.com/api]</span>                      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--server</span>        <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Tournament server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://api.observatory.softmax-research.net]</span>     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Output ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--json</span>          Print raw JSON.                                                                                 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    



### `cogames assay list`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames assay list [OPTIONS]                                                                               </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> List assay runs.                                                                                                  

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--policy</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">TEXT</span>  Filter by policy name[:version] or UUID.                                                <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>            <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">    </span>  Show this message and exit.                                                             <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Server ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--login-server</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Authentication server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://softmax.com/api]</span>                      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--server</span>        <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Tournament server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://api.observatory.softmax-research.net]</span>     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Output ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--json</span>          Print raw JSON.                                                                                 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    



### `cogames assay results`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames assay results [OPTIONS] POLICY_OR_RUN_ID                                                           </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Get scoring results for an assay run.                                                                             

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #800000; text-decoration-color: #800000">*</span>    policy_or_run_id      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">TEXT</span>  Assay run UUID, or policy name (name[:version]). <span style="color: #bf7f7f; text-decoration-color: #bf7f7f">[required]</span>                    <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>          Show this message and exit.                                                                     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Server ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--login-server</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Authentication server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://softmax.com/api]</span>                      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--server</span>        <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Tournament server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://api.observatory.softmax-research.net]</span>     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Output ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--json</span>          Print raw JSON.                                                                                 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    



### `cogames assay submit`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames assay submit [OPTIONS] POLICY                                                                      </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Submit an assay run for a policy.                                                                                 

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #800000; text-decoration-color: #800000">*</span>    policy      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">TEXT</span>  Policy name[:version] or UUID. <span style="color: #bf7f7f; text-decoration-color: #bf7f7f">[required]</span>                                                <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--source</span>                  <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">TEXT   </span>  cogames mission set name to run (cvc_evals, role_specific_evals, …).         <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                                    <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: cvc_evals]                                                </span>         <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--name</span>            <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-n</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">TEXT   </span>  Optional label for this assay run (used for deduplication).                  <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--compat-version</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-c</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">TEXT   </span>  Compat version override.                                                     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--episodes</span>        <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-e</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">INTEGER</span>  Episodes per mission. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: 3]</span>                                           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--max-steps</span>               <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">INTEGER</span>  Max steps per episode. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: 10000]</span>                                      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--watch</span>           <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-w</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">       </span>  Poll until the run completes.                                                <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>                    <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">       </span>  Show this message and exit.                                                  <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Server ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--login-server</span>          <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Authentication server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://softmax.com/api]</span>                      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--server</span>        <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">URL</span>  Tournament server URL. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: https://api.observatory.softmax-research.net]</span>     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Output ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--json</span>          Print raw JSON.                                                                                 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    



## Tutorial Commands



### `cogames tutorial play`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames tutorial play [OPTIONS]                                                                            </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Interactive tutorial - learn to play Cogs vs Clips.                                                               

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>          Show this message and exit.                                                                     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    



### `cogames tutorial cvc`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames tutorial cvc [OPTIONS]                                                                             </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Interactive CvC tutorial - learn roles and territory control.                                                     

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>          Show this message and exit.                                                                     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    



### `cogames tutorial make-policy`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames tutorial make-policy [OPTIONS]                                                                     </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Create a new policy from a template. Requires exactly one policy type.                                            

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Policy Type ───────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--trainable</span>          Create a trainable (neural network) policy.                                                <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--scripted</span>           Create a scripted (rule-based) policy.                                                     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--amongthem</span>          Create an AmongThem BitWorld scripted practice policy.                                     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Output ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--output</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-o</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">FILE</span>  Output file path. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: my_policy.py]</span>                                               <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Other ─────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-h</span>        Show this message and exit.                                                                   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                                                                                                                   
 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Examples:</span>                                                                                                         
 <span style="color: #008080; text-decoration-color: #008080">cogames tutorial make-policy </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-t</span><span style="color: #008080; text-decoration-color: #008080"> </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-o</span><span style="color: #008080; text-decoration-color: #008080"> my_nn_policy.py</span>        Trainable (neural network)                              
 <span style="color: #008080; text-decoration-color: #008080">cogames tutorial make-policy </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-s</span><span style="color: #008080; text-decoration-color: #008080"> </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-o</span><span style="color: #008080; text-decoration-color: #008080"> my_scripted_policy.py</span>  Scripted (rule-based)                                   
 <span style="color: #008080; text-decoration-color: #008080">cogames tutorial make-policy </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--amongthem</span><span style="color: #008080; text-decoration-color: #008080"> </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-o</span><span style="color: #008080; text-decoration-color: #008080"> amongthem_policy.py</span>                                                   
 AmongThem scripted practice                                                                                       

</pre>



    



### `cogames tutorial train`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames tutorial train [OPTIONS]                                                                           </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Train a policy on one or more missions.                                                                           

 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Requires the ``neural`` extra (PyTorch + PufferLib).</span>                                                              
 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Install with: ``pip install cogames[neural]``.</span>                                                                            

 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">By default, our 'lstm' policy architecture is used. You can select a different architecture</span>                       
 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">(like 'stateless' or 'baseline'), or define your own implementing the MultiAgentPolicy</span>                            
 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">interface with a trainable network() method (see mettagrid/policy/policy.py).</span>                                     

 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Continue training from a checkpoint using URI format, or load weights into an explicit class</span>                      
 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">with class=...,data=... syntax.</span>                                                                                   

 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Supply repeated </span><span style="color: #7fbf7f; text-decoration-color: #7fbf7f; font-weight: bold">-m</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> flags to create a training curriculum that rotates through missions.</span>                           
 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Use wildcards (*) in mission names to match multiple missions at once.</span>                                            

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Mission Setup ─────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--mission</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-m</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">MISSION</span>  Missions to train on (wildcards supported, repeatable for curriculum).              <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--cogs</span>     <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-c</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">N      </span>  Number of cogs (agents). <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: (from mission)]</span>                                  <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--variant</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-v</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">VARIANT</span>  Mission variant (repeatable).                                                       <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Policy ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--policy</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span>      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">POLICY</span>  Policy to train (URI (bundle dir or .zip) or NAME or                                  <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                           class=NAME[,data=FILE][,kw.x=val]).                                                   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: class=lstm]                                                                </span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Training ──────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--steps</span>                 <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">N [x&gt;=1</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">]</span>  Number of training steps. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: 10000000000]</span>                              <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--minibatch-size</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">N [x&gt;=1</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">]</span>  Minibatch size for training. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: 4096]</span>                                  <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Hardware ──────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--device</span>                   <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">DEVICE  </span>  Device to train on (auto, cpu, cuda, mps). <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: auto]</span>                 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--num-workers</span>              <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">N [x&gt;=1</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">]</span>  Number of worker processes. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: (CPU cores)]</span>                         <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--parallel-envs</span>            <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">N [x&gt;=1</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">]</span>  Number of parallel environments.                                           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--vector-batch-size</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">N [x&gt;=1</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">]</span>  Vectorized environment batch size.                                         <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Reproducibility ───────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--seed</span>            <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">N [x&gt;=0</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">]</span>  Seed for training RNG. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: 42]</span>                                                <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--map-seed</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">N [x&gt;=0</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">]</span>  MapGen seed for procedural map layout. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: (same as --seed)]</span>                  <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Output ────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--checkpoints</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">DIR</span>  Path to save training checkpoints. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: ./train_dir]</span>                             <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--log-outputs</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">   </span>  Log training outputs.                                                                 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Other ─────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>  <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-h</span>        Show this message and exit.                                                                   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                                                                                                                   
 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Examples:</span>                                                                                                         
 <span style="color: #008080; text-decoration-color: #008080">cogames tutorial train </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-m</span><span style="color: #008080; text-decoration-color: #008080"> arena</span>                             Basic training                                        
 <span style="color: #008080; text-decoration-color: #008080">cogames tutorial train </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-m</span><span style="color: #008080; text-decoration-color: #008080"> arena </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span><span style="color: #008080; text-decoration-color: #008080"> class=baseline</span>                                                                 
 Train baseline policy                                                                                             
 <span style="color: #008080; text-decoration-color: #008080">cogames tutorial train </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span><span style="color: #008080; text-decoration-color: #008080"> ./train_dir/my_run:v5</span>                  Continue from checkpoint                         
 <span style="color: #008080; text-decoration-color: #008080">cogames tutorial train </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-p</span><span style="color: #008080; text-decoration-color: #008080"> class=lstm,data=./weights.safetensors</span>  Load weights into class                          
 <span style="color: #008080; text-decoration-color: #008080">cogames tutorial train </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-m</span><span style="color: #008080; text-decoration-color: #008080"> mission_1 </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-m</span><span style="color: #008080; text-decoration-color: #008080"> mission_2</span>                 Curriculum (rotates)                             
 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">Wildcard patterns:</span>                                                                                                
 <span style="color: #008080; text-decoration-color: #008080">cogames tutorial train </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-m</span><span style="color: #008080; text-decoration-color: #008080"> 'machina_2_bigger:*'</span>                   All missions on machina_2_bigger                 
 <span style="color: #008080; text-decoration-color: #008080">cogames tutorial train </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-m</span><span style="color: #008080; text-decoration-color: #008080"> '*:shaped'</span>                             All "shaped" missions                            
 <span style="color: #008080; text-decoration-color: #008080">cogames tutorial train </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">-m</span><span style="color: #008080; text-decoration-color: #008080"> 'machina*:shaped'</span>                      All "shaped" on machina maps                     

</pre>



    



## BitWorld Commands



### `cogames bitworld games`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames bitworld games [OPTIONS]                                                                           </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> List installed BitWorld games.                                                                                    

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>          Show this message and exit.                                                                     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    



### `cogames bitworld quick-run`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames bitworld quick-run [OPTIONS] GAME [PORT]                                                           </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Compile and run a BitWorld game with local player clients.                                                        

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #800000; text-decoration-color: #800000">*</span>    game      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">TEXT  </span>  BitWorld game folder name. <span style="color: #bf7f7f; text-decoration-color: #bf7f7f">[required]</span>                                                    <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>      port      <span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">[</span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">PORT</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">]</span>  Port to bind. If omitted, BitWorld chooses one.                                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--players</span>            <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">INTEGER RANGE [x&gt;=1</span><span style="color: #bfbf7f; text-decoration-color: #bfbf7f; font-weight: bold">]</span>  Number of local player clients. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: 1]</span>                         <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--address</span>            <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">TEXT                </span>  Server bind address. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: 127.0.0.1]</span>                            <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--save-replay</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">PATH                </span>  Path for BitWorld replay output.                                     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--nim</span>                <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">TEXT                </span>  Nim compiler executable. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: nim]</span>                              <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>               <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">                    </span>  Show this message and exit.                                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    



### `cogames bitworld replay`



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">                                                                                                                   </span>
<span style="font-weight: bold"> </span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Usage: </span><span style="font-weight: bold">cogames bitworld replay [OPTIONS] REPLAY_PATH                                                              </span>
<span style="font-weight: bold">                                                                                                                   </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Replay a BitWorld recording in the global web viewer.                                                             

</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #800000; text-decoration-color: #800000">*</span>    replay_path      <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">PATH</span>  BitWorld replay file. <span style="color: #bf7f7f; text-decoration-color: #bf7f7f">[required]</span>                                                    <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--game</span>                   <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">TEXT   </span>  BitWorld game folder override. Defaults to the replay header game.            <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--port</span>                   <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">INTEGER</span>  Port to bind. If omitted, one is chosen.                                      <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--address</span>                <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">TEXT   </span>  Server bind address. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: 127.0.0.1]</span>                                     <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--browser-address</span>        <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">TEXT   </span>  Address the browser should use to reach the replay server.                    <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>                                   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: 127.0.0.1]                                      </span>                    <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--duration</span>               <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">FLOAT  </span>  Seconds to keep the replay server alive. If omitted, wait until interrupted.  <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--nim</span>                    <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">TEXT   </span>  Nim compiler executable. <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[default: nim]</span>                                       <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">--help</span>                   <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">       </span>  Show this message and exit.                                                   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    


# Citation

If you use CoGames in your research, please cite:

```bibtex
@software{cogames2025,
  title={CoGames: Multi-Agent Cooperative Game Environments},
  author={Softmax},
  year={2025},
  url={https://github.com/Metta-AI/cogames}
}
```

