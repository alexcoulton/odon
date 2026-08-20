"""Lifecycle helpers for long-running, separately packaged Odon extensions."""

from __future__ import annotations

import logging
import signal
import threading
from collections.abc import Callable
from typing import Any, Protocol, TypeVar

from .client import Client, connect
from .discovery import Instance

logger = logging.getLogger("odon.extensions")


class Extension(Protocol):
    """The small lifecycle contract understood by :func:`run`."""

    def close(self) -> None: ...


ExtensionT = TypeVar("ExtensionT", bound=Extension)


def run(
    factory: Callable[[Client], ExtensionT],
    *,
    instance: Instance | str | None = None,
    reconnect: bool = False,
    reconnect_delay: float = 1.0,
    stop_event: threading.Event | None = None,
) -> None:
    """Run an extension until interrupted or its Odon connection closes.

    ``factory`` is normally an extension class whose constructor accepts an Odon
    client. Signal handlers are installed only when called from the main thread.
    Reconnection creates a fresh client and extension registration; extension
    implementations therefore do not need to preserve invalid session handles.
    """

    if reconnect_delay < 0:
        raise ValueError("reconnect_delay must not be negative")
    stopped = stop_event or threading.Event()
    previous_handlers: dict[int, Any] = {}

    def request_stop(_signum: int, _frame: Any) -> None:
        stopped.set()

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)

    try:
        while not stopped.is_set():
            client: Client | None = None
            extension: ExtensionT | None = None
            try:
                client = connect(instance=instance)
                extension = factory(client)
                logger.info(
                    "started extension %s for Odon instance %s",
                    type(extension).__name__,
                    client.hello.instance_id,
                )
                while not stopped.wait(0.25) and not client.closed:
                    pass
            except Exception:
                if not reconnect or stopped.is_set():
                    raise
                logger.exception("Odon extension disconnected; reconnecting")
            finally:
                if extension is not None:
                    try:
                        extension.close()
                    except Exception:
                        logger.exception("failed to close Odon extension cleanly")
                if client is not None:
                    client.close()
            if not reconnect or stopped.is_set():
                break
            stopped.wait(reconnect_delay)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
