# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CoGames: A Game Environment for the Alignment League Benchmark

# %% [markdown]
# <p align="center">
#   <a href="https://pypi.org/project/cogames/">
#     <img src="https://img.shields.io/pypi/v/cogames" alt="PyPi version">
#   </a>
#   <a href="https://pypi.org/project/cogames/">
#     <img src="https://img.shields.io/pypi/pyversions/cogames" alt="Python version">
#   </a>
#   <a href="https://discord.gg/secret-hologenesis">
#     <img src="https://img.shields.io/discord/1309708848730345493?logo=discord&logoColor=white&label=Discord" alt="Discord">
#   </a>
#   <a href="https://deepwiki.com/Metta-AI/cogames">
#     <img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki">
#   </a>
#   <a href="<<colab-link>>">
#     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
#   </a>
#
#   <a href="https://softmax.com/">
#     <img src="https://img.shields.io/badge/Softmax-Website-849EBE?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyBpZD0iTGF5ZXJfMSIgZGF0YS1uYW1lPSJMYXllciAxIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB2aWV3Qm94PSIwIDAgNTI5LjIyIDUzNy40NyI+CiAgPGRlZnM+CiAgICA8c3R5bGU+CiAgICAgIC5jbHMtMSB7CiAgICAgICAgY2xpcC1wYXRoOiB1cmwoI2NsaXBwYXRoKTsKICAgICAgfQoKICAgICAgLmNscy0yIHsKICAgICAgICBmaWxsOiBub25lOwogICAgICB9CgogICAgICAuY2xzLTIsIC5jbHMtMywgLmNscy00LCAuY2xzLTUgewogICAgICAgIHN0cm9rZS13aWR0aDogMHB4OwogICAgICB9CgogICAgICAuY2xzLTMgewogICAgICAgIGZpbGw6ICMwZTI3NTg7CiAgICAgIH0KCiAgICAgIC5jbHMtNCB7CiAgICAgICAgZmlsbDogI2JiY2NmMzsKICAgICAgfQoKICAgICAgLmNscy01IHsKICAgICAgICBmaWxsOiAjODU5ZWJlOwogICAgICB9CiAgICA8L3N0eWxlPgogICAgPGNsaXBQYXRoIGlkPSJjbGlwcGF0aCI+CiAgICAgIDxyZWN0IGNsYXNzPSJjbHMtMiIgd2lkdGg9IjUyOS4yMSIgaGVpZ2h0PSI1MzcuNDciLz4KICAgIDwvY2xpcFBhdGg+CiAgPC9kZWZzPgogIDxnIGNsYXNzPSJjbHMtMSI+CiAgICA8cGF0aCBjbGFzcz0iY2xzLTMiIGQ9Ik00MzUuNzksMTY3LjA5YzEuMzksMTQuNzIsMi4yOCwzNS4xNS4wNyw1OS4xOC0yLjc2LDMwLTguNDEsOTEuNDItNTMuMDYsMTQ0Ljg1LTEyLjcxLDE1LjIxLTMxLjUsMzcuMTctNjQuMjcsNTAuNzUtMjAuMSw4LjMzLTM5LjcyLDExLjE0LTU3LjA4LDExLjE0LTI3LjIxLDAtNDguODctNi44OS01OC4xNC0xMC4xOC0yNC4yOC04LjcxLTQ2LjI2LTEzLjg4LTY0LjgyLTE3LjAzLTkuODEtMS42Ny0xNy4yLTIuODktMjcuNTEtMy4zNy01LjMtLjI0LTExLjItLjU4LTE3LjUyLS41OC0xNC45NywwLTMyLjMxLDEuOTEtNDkuNjYsMTEuNTUtNy4yLDQtMjIuMzYsMTEuODItMzMuMjgsMjguNC0yLjMxLDMuNS01LjczLDcuMjYtNy4xLDEzLjE3LS43NywzLjMzLS44Myw2LjQ1LS41NSw5LjE3LDEwLjE4LDE2LjUsMzEuOSwyNS4xLDcwLjMxLDM5Ljc5LDQwLjU4LDE1LjUyLDc2LjQ2LDIzLjA4LDEwMy4zNiwyNy4yLDM5LjYyLDYuMDcsNzAuMjEsNi4yOSw4OC44NSw2LjM1LjY5LDAsMS4zOSwwLDIuMDgsMCw1MS4zMSwwLDg4Ljg1LTUuOTYsOTYuNzQtNy4yNiwyMS4xNS0zLjQ2LDUwLjY1LTguNDUsODYuOC0yMi4zOSwzOS41Mi0xNS4yNCw2Ni42MS0yNS42OCw3Ni4yNS01MC42OSwxLjUtMy44OCw0LjYzLTEzLjQzLTIuODYtNTUuMDctNy41Ny00Mi4xNS0xOC4xMi03My4xOS0xOS42Ny03Ny42OC0yMC45MS02MC43My0zMS4zNy05MS4wOS00Ny4xNS0xMjAuNTktNy4xNi0xMy4zOC0xNC4zNy0yNS40My0yMS44Mi0zNi43MiIvPgogICAgPHBhdGggY2xhc3M9ImNscy01IiBkPSJNNDA1LjgsMTI2LjM4Yy0yLjcsMTQuMTMtNy40MywzMy40Ny0xNi4xOSw1NS4zOS05LjMzLDIzLjMzLTE3LjQzLDQzLjU5LTM2LjExLDYzLjctOS43MywxMC40Ny0zNC4xMSwzNi43LTcwLjQ5LDQwLjg5LTMuMjguMzgtNi40Ny41NS05LjYuNTUtMTUuMjQsMC0yOS4xMy00LjE2LTQ2LjQ4LTkuMzYtMjIuNjQtNi43OC0zMy45OC0xNC4zNy02MS42My0xOS4xOC0xMC4xNy0xLjc3LTE2LjI2LTIuODMtMjQuNTMtMi44M2gtLjM0cy02Mi4zLjIzLTEwOC45MSw2NS4xMmMtLjI4LjM5LS41NS43Ny0uNTUuNzctNy4zMiwxMC44Ny0xMS45MSwyMC44OC0xNC44NiwyOC43OC0yLjU1LDYuNzktMy45NywxMi4yNi01LjU0LDE4LjI3LTEuNiw2LjE1LTIuNzksMTEuNjItNi4zMSwzMS4xNy0xLjE0LDYuMzYtMi42MSwxNC41OS00LjI3LDI0LjI1LDYuNC0xMC45MSwxNy4xMi0yNS45LDM0LjItMzkuMywxNC41OS0xMS40NSwyNy45OC0xNy4xMywzMy4wNi0xOS4xNiwyLjg1LTEuMTMsMTMuNzUtNS4zNSwyOC45NS03LjgsMy45My0uNjMsMTIuMTgtMS44LDIzLjItMS44LDguMTMsMCwxNy43Ny42MywyOC4zLDIuNTgsNi42NywxLjIzLDE2LjYxLDMuMTMsMjguNDEsOC4xNSw0LjAxLDEuNywxMS4yMSw1LjAzLDE4Ljg1LDksNC45MiwyLjU2LDguMzcsNC41MywxNC4xOCw3LjE0LDQuOSwyLjIxLDkuMDMsMy43NiwxMS43OCw0Ljc0LDAsMCwxOS4yMyw2LjM2LDQwLjI0LDYuOTguOTkuMDMsMS45Ni4wNCwyLjkxLjA0LDUuMzQsMCw5LjY4LS4zOSw5LjY4LS4zOSw2LjYtLjI2LDE1LjktMS4xOCwyNi41NS00LjIsMzkuMjUtMTEuMTQsNjEuNDQtNDEuMDMsNzQuMDctNTguMDIsNDkuOTUtNjcuMTksNDcuOTMtMTY3Ljg1LDQ3LjQxLTE4NC43Ny01LjE3LTcuMDItMTAuNDktMTMuODYtMTYuMDItMjAuNyIvPgogICAgPHBhdGggY2xhc3M9ImNscy00IiBkPSJNMjYzLjg1LDBjLS4xNywwLS4zMywwLS40OSwwLTkuNTYuMTMtMTguOTcsMy45OC01MS40NSwzMy4zNC0zNC4xLDMwLjgzLTQ4Ljk2LDQ4LjA2LTQ4Ljk2LDQ4LjA2LTQ1Ljg0LDUzLjEzLTY4Ljc3LDc5LjY5LTkyLjQ4LDEyMS40OS0zMC4zLDUzLjQxLTQ0LjkxLDEwMC4wOS01MS4yMywxMjMuMDItLjU4LDIuMS0xLjE1LDQuMzItMS43Myw2LjcxLDIuMzItNS40OSw0Ljk0LTExLjE5LDcuOS0xNy4wNCwxMy4zOS0yNi40MiwzNi42MS03Mi4yMSw4My45My04OC44NiwxMy41NC00Ljc2LDI1Ljk2LTYuMDYsMzMuNzktNi4zNywxLjQ3LS4wNiwyLjkxLS4wOCw0LjMtLjA4LDI3LjM5LDAsMzguNzMsMTAuODIsNzIsMTguODMsMTUuMjcsMy42NywzMS43OCw3Ljg4LDQ5LjUsNy44OCw5LjM2LDAsMTkuMDYtMS4xNywyOS4xLTQuMjIsMzYuNDktMTEuMDYsNTYuNDQtNDEuMzMsNjMuMjUtNTEuNTMsMTUuMzYtMjMuMDIsMTkuOTUtNDUuMjMsMjEuOS01NS4xNCwyLjQ3LTEyLjU3LDMuMTQtMjMuODUsMy4wNC0zMy4xNC02LjMyLTcuMzUtMTIuOTItMTQuOTEtMTkuODctMjIuODYsMCwwLTE3LjM5LTE5Ljg5LTUxLjk4LTQ5LjQ4QzI4Mi41MSwzLjM5LDI3Mi41LDAsMjYzLjg1LDAiLz4KICA8L2c+Cjwvc3ZnPg==" alt="Softmax website">
#   </a>
# </p>

# %% [markdown]
# The [Alignment League Benchmark (ALB)](https://www.softmax.com/alignmentleague) is a suite of multi-agent games, designed to measure how well AI agents align, coordinate, and collaborate with others (both AIs and humans).
#
# CoGames is the games environment for ALB. You can use it to:
#
# * create new games
# * train agents to play existing ALB games
# * submit those agents to the ALB leaderboard
#
# There's one ALB game right now: Cogs vs Clips.

# %% [markdown]
# # Quick Start

# %% [markdown]
# ## Step 1: Install CoGames
#
# Install [cogames](https://pypi.org/project/cogames/) as a Python package.
# ```bash
# pip install cogames
# ```
#
# <details><summary>Using uv</summary>
#
# ```bash
# # Install uv
# curl -LsSf https://astral.sh/uv/install.sh | sh
#
# # Create a virtual environment
# uv venv .venv
# source .venv/bin/activate
#
# # Install cogames
# uv pip install cogames
# ```
#
# </details>
# <details><summary>Using Docker</summary>
#
# ```dockerfile
# # Ensure Python 3.12 is available
# FROM python:3.12-slim
#
# # Ensure C/C++ compiler is available
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends build-essential && \
#   rm -rf /var/lib/apt/lists/*
#
# # Install cogames
# RUN pip install --no-cache-dir cogames
# ```
#
# </details>
# <details><summary>Using Colab</summary>
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](<<colab-link>>)
#
# </details>

# %% [markdown]
# ## Step 2: Play Cogs vs Clips
#
# Play an easy mission in Cogs vs Clips using:
# ```bash
# cogames tutorial play
# ```
# The game will open in a new window, and the terminal will give you instructions to complete training mission.

# %% [markdown]
# ## Step 3: Submit a policy to the leaderboard

# %% [markdown]
# 1. Log into the ALB leaderboard with your GitHub account.
#
#     ```bash
#     cogames login
#     ```
#
# 2. Upload one of our starter policies under your name.
#
#     ```bash
#     cogames upload --policy "class=cogames.policy.scripted_agent.starter_agent.StarterPolicy" --name "$USER.README-quickstart-starter-policy"
#     ```
#
# 3. List the seasons currently running.
#
#     ```bash
#     cogames seasons
#     ```
#
# 4. Submit the policy to one of the running seasons.
#
#     ```bash
#     cogames submit "$USER.README-quickstart-starter-policy:v1" --season beta-cogsguard
#     ```

# %% [markdown]
# # Tutorials
#
# To learn more, see:
#
# 1. [Creating a policy](tutorials/01_MAKE_POLICY.md): Creating a custom policy and evaluating it
# 2. [Training](tutorials/02_TRAIN.md): Training a custom policy and evaluating it
# 3. [Submitting](tutorials/03_SUBMIT.md): Submitting to the leaderboard and understanding the results
#
# If you want help, or to share your experience, join [the Discord](https://discord.gg/secret-hologenesis).

# %% [markdown]
# # About the game
#
# CogsGuard is a cooperative territory-control game. Teams of AI agents ("Cogs") work together to capture and defend
# junctions against automated opponents ("Clips") by:
#
# * gathering resources and depositing them at controlled junctions
# * acquiring specialized roles (Miner, Aligner, Scrambler, Scout) at gear stations
# * capturing neutral junctions using Aligners (costs 1 heart + 1 influence)
# * disrupting enemy junctions using Scramblers (costs 1 heart)
# * defending territory from Clips expansion
#
# Read [MISSION.md](MISSION.md) for a thorough description of the game mechanics.
#
# <p align="center">
#   <img src="assets/cvc-reel.gif" alt="CogsGuard reel">
# <br>
#
# There are many mission configurations available, with different map sizes, junction layouts, and game rules.
#
# Overall, CogsGuard aims to present rich environments with:
#
# - **Territory control**: Capture and defend junctions to score points each tick
# - **Role specialization**: Four roles (Miner, Aligner, Scrambler, Scout) with distinct capabilities and dependencies
# - **Dense rewards**: Agents receive reward every tick proportional to territory controlled
# - **Partial observability**: Agents have limited visibility of the environment
# - **Required multi-agent cooperation**: No single role can succeed alone; Miners need Aligners to capture junctions,
#   Aligners need Miners for resources, Scramblers must clear enemy territory for Aligners to advance

# %% [markdown]
# # About the tournament
#
# ## How seasons work
#
# The ALB leaderboard runs in seasons. Each season has two pools:
#
# - **Qualifying pool**: Where new submissions start. Your policy plays matches against other policies in the pool.
# - **Competition pool**: Policies that score above a threshold in qualifying get promoted here.
#
# To see active seasons and their pools:
# ```bash
# cogames seasons
# ```
#
# ## How scoring works
#
# When you submit a policy, it gets queued for matches against other policies in its pool. Our focal metric is VORP (Value Over Replacement Policy), which estimates how much your agent improves team performance compared to a baseline.
#
# VORP is calculated by comparing:
#
# - Replacement mean: The average score when only other pool policies play (no candidate)
# - Candidate score: The score when your policy plays
#
# The difference tells us how much value your policy adds to a team. A positive VORP means your policy makes teams better; a negative VORP means teams perform worse with your policy than without it.
#
# You can evaluate VOR locally before submitting:
#
# ```bash
# cogames pickup --policy <YOUR_POLICY> --pool <POOL_POLICY>
# ```
#
# ## Viewing results
#
# To check your submission status and match results:
# ```bash
# cogames submissions
# cogames leaderboard --season beta-cogsguard
# ```

# %% [markdown]
# # Command Reference
#
# Most commands are of the form `cogames <command> --mission [MISSION] --policy [POLICY] [OPTIONS]`
#
# To specify a `MISSION`, you can:
#
# - Use a mission name from the registry given by `cogames missions` (e.g. `training_facility_1`).
# - Use a path to a mission configuration file (e.g. `path/to/mission.yaml`).
# - Alternatively, specify a set of missions with `--mission-set`.
#
# To specify a `POLICY`, use one of two formats:
#
# - **URI format** (for checkpoint bundles):
#
#     - Point directly at a checkpoint bundle (directory or `.zip` containing `policy_spec.json`)
#     - Examples: `./train_dir/my_run:v5`, `./train_dir/my_run:v5.zip`, `s3://bucket/path/run:v5.zip`
#     - Use `:latest` suffix to auto-resolve the highest version: `./train_dir/checkpoints:latest`
#
# - **Key-value format** (for explicit class + weights):
#
#     - `class=`: Policy shorthand or full class path from `cogames policies`, e.g. `class=lstm` or
#     `class=cogames.policy.random.RandomPolicy`.
#     - `data=`: Optional path to a weights file (e.g., `weights.safetensors`). Must be a file, not a directory.
#     - `proportion=`: Optional positive float specifying the relative share of agents that use this policy (default: 1.0).
#     - `kw.<arg>=`: Optional policy `__init__` keyword arguments (all values parsed as strings).
#
# You can view all the commands with
# ```bash
# cogames --help
# ```
# and you can view help for a given command with:
# ```bash
# cogames [COMMAND] --help
# ```

# %%
# <<hide-input>>
import sys
from collections import defaultdict

# Clear cached modules
for mod in list(sys.modules.keys()):
    if mod.startswith("cogames") or mod.startswith("rich"):
        del sys.modules[mod]

import typer
from click import Context
from IPython.display import Markdown, display

from cogames.main import app


def build_command_groups_from_cli():
    """Build command groups by introspecting the CLI's rich_help_panel attributes."""
    panel_to_commands = defaultdict(list)

    for cmd_info in app.registered_commands:
        if getattr(cmd_info, "hidden", False):
            continue
        panel = getattr(cmd_info, "rich_help_panel", None)
        if panel is None or isinstance(panel, typer.models.DefaultPlaceholder):
            continue
        name = cmd_info.name or (cmd_info.callback.__name__ if cmd_info.callback else None)
        if name:
            panel_to_commands[panel].append(name)

    for group_info in app.registered_groups:
        group_name = group_info.name
        if not group_name or getattr(group_info, "hidden", False):
            continue
        sub_app = group_info.typer_instance
        if sub_app:
            for sub_cmd_info in sub_app.registered_commands:
                if getattr(sub_cmd_info, "hidden", False):
                    continue
                panel = getattr(sub_cmd_info, "rich_help_panel", None)
                if panel is None or isinstance(panel, typer.models.DefaultPlaceholder):
                    continue
                sub_name = sub_cmd_info.name or (sub_cmd_info.callback.__name__ if sub_cmd_info.callback else None)
                if sub_name:
                    panel_to_commands[panel].append(f"{group_name} {sub_name}")

    return dict(panel_to_commands)


def get_click_command(cmd_name: str):
    """Get the Click command object for a command name."""
    click_app = typer.main.get_command(app)
    if " " in cmd_name:
        parent, child = cmd_name.split(" ", 1)
        parent_cmd = click_app.commands.get(parent)
        return parent_cmd.commands.get(child) if parent_cmd and hasattr(parent_cmd, "commands") else None
    return click_app.commands.get(cmd_name)


def display_command_reference():
    """Display command reference with proper heading structure."""
    command_groups = build_command_groups_from_cli()

    for group_name, cmd_names in command_groups.items():
        display(Markdown(f"## {group_name} Commands"))
        for cmd_name in cmd_names:
            display(Markdown(f"### `cogames {cmd_name}`"))
            cmd = get_click_command(cmd_name)
            if cmd:
                ctx = Context(cmd, info_name=f"cogames {cmd_name}")
                print(cmd.get_help(ctx))


display_command_reference()

# %% [markdown]
# # Citation
#
# If you use CoGames in your research, please cite:
#
# ```bibtex
# @software{cogames2025,
#   title={CoGames: Multi-Agent Cooperative Game Environments},
#   author={Softmax},
#   year={2025},
#   url={https://github.com/metta-ai/metta}
# }
# ```
#
