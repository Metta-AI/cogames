from __future__ import annotations

import importlib.util
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def test_cogs_vs_clips_snapshot_exposes_admin_slot_state(tmp_path: Path) -> None:
    game = _new_game(tmp_path)

    snapshot = game.snapshot()

    assert snapshot["protocol"] == "coworld.player.v1"
    assert snapshot["global_protocol"] == "mettagrid.mettascope.live.v1"
    assert snapshot["tick_mode"] == "fixed"
    assert snapshot["human_action_timeout_seconds"] == 5.0
    assert snapshot["slots"][0]["control_state"] == {
        "control_mode": "policy",
        "human_controller_connection_id": None,
        "tick_mode": "fixed",
        "human_action_timeout_seconds": 5.0,
    }


def test_cogs_vs_clips_admin_snapshot_exposes_takeover_player_links(tmp_path: Path) -> None:
    server_module = _load_cogs_vs_clips_server_module()
    game = server_module.CogsVsClipsGame(
        {
            "mission": "machina_1",
            "tokens": ["token 0", "token/1"],
            "max_steps": 3,
            "seed": 0,
            "step_seconds": 0.02,
        },
        results_path=tmp_path / "results.json",
        replay_path=None,
        request_shutdown=lambda: None,
    )

    snapshot = game.admin_snapshot()

    assert all("player_client_url" not in slot for slot in game.snapshot()["slots"])
    for slot_index, slot in enumerate(snapshot["slots"]):
        player_link = urlparse(slot["player_client_url"])
        assert player_link.path == "/player"
        assert parse_qs(player_link.query) == {
            "slot": [str(slot_index)],
            "token": [game.tokens[slot_index]],
            "takeover": ["1"],
        }


def test_cogs_vs_clips_global_action_updates_policy_action(tmp_path: Path) -> None:
    game = _new_game(tmp_path)
    action_name = next(action_name for action_name in game.episode.action_names if action_name != "noop")

    game.handle_global_message({"type": "action", "agent_id": 0, "action_name": action_name})
    game.episode.apply_actions()

    assert game.episode.latest_policy_actions[0].action_name == action_name
    assert game.episode.latest_action_indices[0] == game.episode.action_names.index(action_name)


def _new_game(tmp_path: Path):
    server_module = _load_cogs_vs_clips_server_module()
    return server_module.CogsVsClipsGame(
        {
            "mission": "machina_1",
            "tokens": ["token-0", "token-1"],
            "max_steps": 3,
            "seed": 0,
            "step_seconds": 0.02,
        },
        results_path=tmp_path / "results.json",
        replay_path=None,
        request_shutdown=lambda: None,
    )


def _cogs_vs_clips_root() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "cogames" / "coworld" / "examples" / "cogs_vs_clips"


def _load_cogs_vs_clips_server_module():
    spec = importlib.util.spec_from_file_location(
        "cogs_vs_clips_server_test",
        _cogs_vs_clips_root() / "game" / "server.py",
    )
    assert spec is not None
    assert spec.loader is not None
    server_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server_module)
    return server_module
