from __future__ import annotations

import ctypes
import ctypes.util
import os
import shutil
import subprocess
import sys


def _macos_session_flag(core_foundation: ctypes.CDLL, core_graphics: ctypes.CDLL, session: int, key: str) -> bool:
    core_foundation.CFDictionaryGetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    core_foundation.CFDictionaryGetValue.restype = ctypes.c_void_p
    core_foundation.CFBooleanGetValue.argtypes = [ctypes.c_void_p]
    core_foundation.CFBooleanGetValue.restype = ctypes.c_bool
    try:
        symbol = ctypes.c_void_p.in_dll(core_graphics, key)
    except ValueError:
        return False
    value = core_foundation.CFDictionaryGetValue(session, symbol)
    return bool(value and core_foundation.CFBooleanGetValue(value))


def _macos_session_has_display() -> bool:
    core_graphics_path = ctypes.util.find_library("CoreGraphics")
    if core_graphics_path is None:
        return False

    core_foundation_path = ctypes.util.find_library("CoreFoundation")
    if core_foundation_path is None:
        return False

    try:
        core_graphics = ctypes.CDLL(core_graphics_path)
        core_foundation = ctypes.CDLL(core_foundation_path)

        core_graphics.CGSessionCopyCurrentDictionary.argtypes = []
        core_graphics.CGSessionCopyCurrentDictionary.restype = ctypes.c_void_p
        core_foundation.CFRelease.argtypes = [ctypes.c_void_p]
        session = core_graphics.CGSessionCopyCurrentDictionary()
        if not session:
            return False

        try:
            return _macos_session_flag(
                core_foundation, core_graphics, session, "kCGSessionOnConsoleKey"
            ) and _macos_session_flag(
                core_foundation,
                core_graphics,
                session,
                "kCGSessionLoginDoneKey",
            )
        finally:
            core_foundation.CFRelease(session)
    except (OSError, TypeError, ValueError):
        return False


def _macos_launchctl_has_display() -> bool:
    if shutil.which("launchctl") is None:
        return False

    try:
        result = subprocess.run(
            ["launchctl", "print", f"gui/{os.getuid()}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=0.25,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False

    if result.returncode != 0:
        return False

    return "session = Aqua" in (result.stdout or "") or "session = Aqua" in (result.stderr or "")


def _macos_has_display() -> bool:
    # Prefer session-based checks on macOS.
    # Env vars can be set in SSH/XQuartz sessions without an active WindowServer GUI,
    # so we avoid treating them as sufficient on their own.
    if _macos_session_has_display():
        return True

    return _macos_launchctl_has_display()


def _linux_has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _windows_has_display() -> bool:
    return True


def has_display() -> bool:
    if sys.platform.startswith("linux"):
        return _linux_has_display()
    if sys.platform == "darwin":
        return _macos_has_display()
    if sys.platform == "win32":
        return _windows_has_display()
    return False
