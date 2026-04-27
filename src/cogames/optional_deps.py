"""Dependency availability checks for optional extras."""

import importlib.util


def _is_importable(module_name: str) -> bool:
    """Check if a module can be imported without actually importing it."""
    return importlib.util.find_spec(module_name) is not None


def has_neural() -> bool:
    """Return True if the ``neural`` extra (PyTorch + PufferLib) is installed."""
    return _is_importable("torch") and _is_importable("pufferlib")


def require_neural(command: str) -> None:
    """Raise a clear error if the ``neural`` extra is not installed.

    Call this at the top of CLI commands that need PyTorch or PufferLib
    before doing any deferred imports so users see a friendly message
    instead of a raw ImportError traceback.

    Args:
        command: The CLI command name (e.g. ``"cogames tutorial train"``) shown in the
            error message so users know what triggered the check.
    """
    if has_neural():
        return
    raise SystemExit(
        f"'{command}' requires PyTorch and PufferLib, which are not installed.\n"
        "\n"
        "Install them with:\n"
        "  pip install cogames[neural]\n"
    )
