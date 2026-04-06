"""Canonical local-game commands for ``softmax cogames``."""

import typer

from cogames.main import create_bundle_cmd, make_policy, play_cmd, run_cmd, train_cmd, tutorial_cmd

app = typer.Typer(
    help="Local CoGames workflows exposed under softmax.",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    rich_markup_mode="rich",
)

tutorial_app = typer.Typer(
    help="Tutorial commands for CoGames.",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    rich_markup_mode="rich",
)

tutorial_app.command(
    name="play",
    help="Interactive tutorial for Cogs vs Clips.",
    add_help_option=False,
)(tutorial_cmd)
tutorial_app.command(
    name="train",
    help="Tutorial-friendly local training flow.",
    add_help_option=False,
)(train_cmd)
tutorial_app.command(
    name="make-policy",
    help="Create a starter policy from a template.",
    add_help_option=False,
)(make_policy)

app.add_typer(tutorial_app, name="tutorial", rich_help_panel="Tutorials")
app.command(
    name="play",
    help="Play a CoGames mission locally.",
    rich_help_panel="Local Games",
    add_help_option=False,
)(play_cmd)
app.command(
    name="train",
    help="Train a CoGames policy locally.",
    rich_help_panel="Local Games",
    add_help_option=False,
)(train_cmd)
app.command(
    name="eval",
    help="Evaluate CoGames policies locally.",
    rich_help_panel="Local Games",
    add_help_option=False,
)(run_cmd)
app.command(
    name="bundle",
    help="Create a policy bundle for submission.",
    rich_help_panel="Local Games",
    add_help_option=False,
)(create_bundle_cmd)
