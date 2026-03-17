"""Interactive CLI login flow."""

from __future__ import annotations

import asyncio
import html
import socket
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Sequence

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from rich.panel import Panel

from cogames.auth import build_browser_login_url, save_token
from cogames.cli.base import console
from cogames.token_storage import TokenKind


@dataclass
class _CLIAuthSession:
    token: str | None = None
    error: str | None = None
    auth_completed: threading.Event = field(default_factory=threading.Event)
    completion_lock: threading.Lock = field(default_factory=threading.Lock)


def _render_html(
    *,
    title: str,
    headline: str,
    message_lines: Sequence[str],
    status: Literal["success", "error"],
    auto_close_seconds: int | None = None,
    extra_html: str = "",
) -> str:
    icon = "&#10003;" if status == "success" else "&#9888;"
    escaped_title = html.escape(title)
    escaped_headline = html.escape(headline)
    messages = "".join(f"<p class='smx-auth__message'>{html.escape(line)}</p>" for line in message_lines)
    current_year = datetime.now().year
    auto_close_script = ""
    if auto_close_seconds is not None:
        auto_close_script = f"""
        <script>
            window.setTimeout(function () {{
                try {{
                    window.close();
                }} catch (err) {{
                    console.debug("Auto-close suppressed", err);
                }}
            }}, {int(auto_close_seconds * 1000)});
        </script>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{escaped_title}</title>
    <link rel="stylesheet" href="https://softmax.com/Assets/softmax.css" />
    <style>
        :root {{
            color-scheme: light;
        }}
        * {{
            box-sizing: border-box;
        }}
        body.smx-auth-page {{
            margin: 0;
            min-height: 100vh;
            background-color: #fffdf4;
            color: #0E2758;
            font-family: "ABC Marfa Variable", "Roboto", -apple-system, BlinkMacSystemFont, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: clamp(2.4rem, 8vw, 4rem) 1.5rem;
            overflow: hidden;
            text-rendering: optimizeLegibility;
        }}
        .smx-auth-card {{
            width: min(560px, 100%);
            background: rgba(255, 254, 248, 0.95);
            border-radius: 24px;
            border: 1px solid rgba(14, 39, 88, 0.12);
            box-shadow: 0 32px 60px rgba(14, 39, 88, 0.12);
            padding: clamp(2rem, 6vw, 3.25rem);
            text-align: center;
        }}
        .smx-auth-card--success .smx-auth-icon {{
            background: rgba(26, 107, 63, 0.16);
            color: #195C38;
        }}
        .smx-auth-card--error .smx-auth-icon {{
            background: rgba(176, 46, 38, 0.16);
            color: #952F2B;
        }}
        .smx-auth-icon {{
            height: 76px;
            width: 76px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.5rem;
            margin: 0 auto 20px;
            border: 1px solid rgba(14, 39, 88, 0.12);
        }}
        .smx-auth-headline {{
            margin: 0 0 12px;
            font-size: clamp(1.8rem, 5vw, 2.35rem);
            font-weight: 600;
            letter-spacing: -0.01em;
        }}
        .smx-auth__message {{
            margin: 0 0 12px;
            font-size: 1.02rem;
            line-height: 1.6;
            color: rgba(14, 39, 88, 0.72);
        }}
        .smx-auth__body {{
            display: grid;
            gap: 8px;
        }}
        .smx-auth__actions {{
            margin-top: 32px;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }}
        .smx-auth-button {{
            appearance: none;
            border-radius: 999px;
            border: 2px solid #0E2758;
            background: #0E2758;
            color: #fffdf4;
            cursor: pointer;
            padding: 0.9rem 1.8rem;
            font-size: 0.95rem;
            font-family: "Marfa Mono", "Courier New", monospace;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            transition: transform 0.15s ease, box-shadow 0.2s ease, background 0.2s ease;
        }}
        .smx-auth-button:hover {{
            transform: translateY(-1px);
            background: #1a3875;
            border-color: #1a3875;
            box-shadow: 0 14px 28px rgba(26, 56, 117, 0.18);
        }}
        .smx-auth-button:active {{
            transform: translateY(0);
            box-shadow: 0 8px 16px rgba(14, 39, 88, 0.18);
        }}
        .smx-auth-footnote {{
            margin-top: 28px;
            font-size: 0.85rem;
            color: rgba(14, 39, 88, 0.55);
        }}
        @media (max-width: 540px) {{
            .smx-auth-card {{
                padding: 2.4rem 1.8rem;
            }}
        }}
    </style>
</head>
<body class="smx-auth-page">
    <main class="smx-auth-card smx-auth-card--{status}" role="dialog" aria-live="polite">
        <div class="smx-auth-icon" aria-hidden="true">{icon}</div>
        <h1 class="smx-auth-headline">{escaped_headline}</h1>
        <div class="smx-auth__body">
            {messages}
            {extra_html}
        </div>
        <div class="smx-auth-footnote">may we all find alignment - softmax, {current_year}</div>
    </main>
    {auto_close_script}
</body>
</html>"""


def _success_html() -> str:
    return _render_html(
        title="Authentication Successful",
        headline="You're all set!",
        message_lines=[
            "Authentication complete. You can return to the terminal.",
            "This window will close automatically in a moment.",
        ],
        status="success",
        auto_close_seconds=3,
        extra_html="""
            <div class="smx-auth__actions">
                <button class="smx-auth-button" type="button" onclick="window.close()">Close this window</button>
            </div>
            """,
    )


def _error_html(*, error_message: str) -> str:
    return _render_html(
        title="Authentication Error",
        headline="Something went wrong",
        message_lines=[
            error_message,
            "Please retry the login process or contact support if the issue persists.",
        ],
        status="error",
    )


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _open_browser(*, url: str) -> None:
    webbrowser.open(url)


def _finish_authentication(session: _CLIAuthSession, *, token: str | None = None, error: str | None = None) -> bool:
    with session.completion_lock:
        if session.auth_completed.is_set():
            return False
        if token is not None:
            session.token = token
        if error is not None:
            session.error = error
        session.auth_completed.set()
        return True


def _create_app(session: _CLIAuthSession) -> FastAPI:
    app = FastAPI(title="CLI OAuth2 Callback Server")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/callback")
    async def callback(request: Request) -> HTMLResponse:
        try:
            token = request.query_params.get("token")
            if not token:
                _finish_authentication(session, error="No token received in callback")
                return HTMLResponse(content=_error_html(error_message="No token received"), status_code=400)

            _finish_authentication(session, token=token)
            return HTMLResponse(content=_success_html())
        except Exception as exc:
            _finish_authentication(session, error=f"Callback error: {exc}")
            return HTMLResponse(content=_error_html(error_message=f"Error: {exc}"), status_code=500)

    return app


def _validate_token(*, login_server: str, token: str) -> bool | None:
    validate_url = f"{login_server.rstrip('/')}/validate"
    try:
        response = httpx.get(
            validate_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=5.0,
        )
    except httpx.HTTPError:
        return None

    if response.status_code == 200:
        data = response.json()
        return bool(data.get("valid"))
    if response.status_code in {400, 401, 403, 404}:
        return False
    return None


def _print_login_instructions(*, auth_url: str, agent_hint: str | None) -> None:
    if agent_hint:
        console.print(
            Panel(
                agent_hint,
                title="🤖 Agent Hint",
                border_style="cyan",
            )
        )
        console.print()
    console.print("Open this URL in any browser to sign in:")
    console.print()
    console.print(auth_url)
    console.print()


def _start_manual_token_prompt(*, session: _CLIAuthSession, login_server: str) -> None:
    def prompt_loop() -> None:
        while not session.auth_completed.is_set():
            try:
                token = input("Paste token here when ready: ").strip()
            except EOFError:
                return

            if session.auth_completed.is_set():
                return
            if not token:
                continue

            validation_result = _validate_token(login_server=login_server, token=token)
            if validation_result is False:
                console.print("Invalid token. Please try again.", style="red")
                continue
            if validation_result is None:
                console.print("Could not validate token right now. Saving it anyway.", style="yellow")

            _finish_authentication(session, token=token)
            return

    threading.Thread(target=prompt_loop, daemon=True).start()


def _run_server(*, session: _CLIAuthSession, port: int) -> None:
    try:
        app = _create_app(session)
        config = uvicorn.Config(
            app=app,
            host="127.0.0.1",
            port=port,
            log_level="error",
            access_log=False,
        )
        asyncio.run(uvicorn.Server(config).serve())
    except Exception as exc:
        session.error = f"Server error: {exc}"


def _wait_for_callback_server_to_start(*, session: _CLIAuthSession, port: int, timeout_seconds: float = 3.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if session.error:
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.05)
    return False


def do_interactive_login_for_token(
    *,
    login_server: str,
    server_to_save_token_under: str,
    token_kind: TokenKind,
    agent_hint: str | None,
    open_browser: bool,
) -> None:
    """Run the CLI browser login flow and save the resulting token."""
    assert sys.stdin.isatty(), "This function should only be called when stdin is a TTY"

    session = _CLIAuthSession()
    callback_url: str | None = None
    port = _find_free_port()

    threading.Thread(target=_run_server, kwargs={"session": session, "port": port}, daemon=True).start()
    if _wait_for_callback_server_to_start(session=session, port=port):
        callback_url = f"http://127.0.0.1:{port}/callback"
    session.error = None

    auth_url = build_browser_login_url(login_server, callback_url=callback_url)
    _print_login_instructions(auth_url=auth_url, agent_hint=agent_hint)
    if open_browser:
        _open_browser(url=auth_url)

    _start_manual_token_prompt(session=session, login_server=login_server)
    session.auth_completed.wait()
    if session.error:
        raise RuntimeError(session.error)
    if not session.token:
        raise RuntimeError("No token received")

    save_token(token_kind=token_kind, server=server_to_save_token_under, token=session.token)
    print(f"\nToken saved for {server_to_save_token_under}")
    print()
