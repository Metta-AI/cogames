import signal
import threading
from collections.abc import Callable
from types import FrameType, TracebackType
from typing import Any

SignalHandler = Callable[[int, FrameType | None], Any] | int | signal.Handlers | None


def _can_set_signal() -> bool:
    """Check if signal handlers can be set in the current context."""
    return threading.current_thread() is threading.main_thread()


class DeferSigintContextManager:
    """Entering the contextmanager re-associates the active SIGINT handler.

    The first observed SIGINT follows python's default behavior (raises a KeyboardInterrupt).

    After any subsequent SIGINTs, or when exiting this context manager's suite, any previously
    active SIGINT handler is re-associated. If a SIGINT was received while the context manager was
    active, and if the previously-active handler was not the default or the IGNORE handler, it is
    called once.

    If called from a non-main thread, signal handling is silently skipped since
    signal.signal() can only be called from the main thread of the main interpreter.
    """

    def __init__(self) -> None:
        self._previous_handler: SignalHandler = None
        self._have_rebound_handler = False
        self._can_handle_signals = _can_set_signal()
        self.sigint_count = 0

    def __enter__(self) -> "DeferSigintContextManager":
        if self._can_handle_signals:
            self._previous_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self.handle)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if not self._have_rebound_handler:
            self._rebind_and_call_old_handler()

    def _rebind_and_call_old_handler(self) -> None:
        if not self._can_handle_signals:
            self._have_rebound_handler = True
            return
        handler = self._previous_handler
        signal.signal(signal.SIGINT, handler)
        self._have_rebound_handler = True
        if self.sigint_count > 0:
            if handler == signal.SIG_IGN:
                pass
            elif handler == signal.SIG_DFL:
                signal.default_int_handler(signal.SIGINT, None)
            elif callable(handler):
                handler(signal.SIGINT, None)

    def handle(self, sig: int, frame: FrameType | None) -> None:
        self.sigint_count += 1
        if self.sigint_count > 1:
            self._rebind_and_call_old_handler()
        else:
            signal.default_int_handler(signal.SIGINT, None)
