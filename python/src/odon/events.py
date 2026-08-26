"""Synchronous event subscriptions for Odon."""

from __future__ import annotations

import queue
import logging
import threading
from collections.abc import Callable, Iterable, Iterator
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Mapping

from .errors import ConnectionClosedError
from .models import Event

if TYPE_CHECKING:
    from .client import Client

EventCallback = Callable[[Event], Any]
logger = logging.getLogger("odon.events")
_callback_scope: ContextVar[bool] = ContextVar(
    "odon_sync_event_callback_scope", default=False
)


def pattern_matches(pattern: str, event: str) -> bool:
    return pattern == "*" or pattern == event or (
        pattern.endswith("*") and event.startswith(pattern[:-1])
    )


class Events:
    """Manage subscriptions and consume server-pushed events."""

    def __init__(self, client: "Client", *, max_queue: int = 1024) -> None:
        self._client = client
        self._queue: queue.Queue[Event | None] = queue.Queue(maxsize=max_queue)
        self._callback_queue: queue.Queue[Event | None] = queue.Queue(maxsize=max_queue)
        self._callbacks: list[tuple[tuple[str, ...], EventCallback]] = []
        self._lock = threading.Lock()
        self._closed = False
        self.dropped_events = 0
        self._worker = threading.Thread(
            target=self._run_callbacks, name="odon-event-callbacks", daemon=True
        )
        self._worker.start()

    def subscribe(
        self,
        events: str | Iterable[str],
        callback: EventCallback | None = None,
    ) -> Mapping[str, Any]:
        patterns = (events,) if isinstance(events, str) else tuple(events)
        registration = (patterns, callback) if callback is not None else None
        if callback is not None:
            assert registration is not None
            with self._lock:
                self._callbacks.append(registration)
        try:
            return self._client.call("events.subscribe", {"events": list(patterns)})
        except BaseException:
            if registration is not None:
                with self._lock:
                    self._callbacks = [
                        item for item in self._callbacks if item is not registration
                    ]
            raise

    def unsubscribe(self, events: str | Iterable[str] | None = None) -> Mapping[str, Any]:
        patterns = None if events is None else ((events,) if isinstance(events, str) else tuple(events))
        result = self._client.call(
            "events.unsubscribe", {} if patterns is None else {"events": list(patterns)}
        )
        if patterns is None:
            with self._lock:
                self._callbacks.clear()
        return result

    def status(self) -> Mapping[str, Any]:
        return self._client.call("events.get_status")

    @property
    def in_callback(self) -> bool:
        """Whether the caller is currently running in a synchronous event callback."""

        return _callback_scope.get()

    def remove_callback(self, callback: EventCallback) -> None:
        with self._lock:
            self._callbacks = [item for item in self._callbacks if item[1] is not callback]

    def next(self, timeout: float | None = None) -> Event:
        """Block until the next event arrives, or raise ``queue.Empty``."""

        event = self._queue.get(timeout=timeout)
        if event is None:
            raise ConnectionClosedError("Odon event stream is closed")
        return event

    def iter(self, timeout: float | None = None) -> Iterator[Event]:
        while not self._client.closed:
            try:
                yield self.next(timeout)
            except queue.Empty:
                if timeout is not None:
                    return
            except ConnectionClosedError:
                return

    def _receive(self, params: Mapping[str, Any]) -> None:
        if self._closed:
            return
        event = Event.from_params(params)
        for destination in (self._queue, self._callback_queue):
            try:
                destination.put_nowait(event)
            except queue.Full:
                self.dropped_events += 1

    def _run_callbacks(self) -> None:
        while (event := self._callback_queue.get()) is not None:
            with self._lock:
                callbacks = tuple(self._callbacks)
            for patterns, callback in callbacks:
                if any(pattern_matches(pattern, event.name) for pattern in patterns):
                    token = _callback_scope.set(True)
                    try:
                        callback(event)
                    except Exception:
                        # User callbacks are isolated from connection IO and each other.
                        logger.exception("Odon event callback failed for %s", event.name)
                        continue
                    finally:
                        _callback_scope.reset(token)

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for destination in (self._queue, self._callback_queue):
            try:
                destination.put_nowait(None)
            except queue.Full:
                # Closing takes precedence over delivery of already-buffered events.
                try:
                    destination.get_nowait()
                except queue.Empty:
                    pass
                destination.put_nowait(None)
