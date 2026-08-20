"""Asyncio event subscriptions for Odon."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from typing import TYPE_CHECKING, Any, Mapping

from .errors import ConnectionClosedError
from .events import pattern_matches
from .models import Event

if TYPE_CHECKING:
    from .async_client import AsyncClient

AsyncEventCallback = Callable[[Event], Any | Awaitable[Any]]
logger = logging.getLogger("odon.events")


class AsyncEvents:
    def __init__(self, client: "AsyncClient", *, max_queue: int = 1024) -> None:
        self._client = client
        self._queue: asyncio.Queue[Event | None] = asyncio.Queue(maxsize=max_queue)
        self._callbacks: list[tuple[tuple[str, ...], AsyncEventCallback]] = []
        self._closed = False
        self.dropped_events = 0

    async def subscribe(
        self,
        events: str | Iterable[str],
        callback: AsyncEventCallback | None = None,
    ) -> Mapping[str, Any]:
        patterns = (events,) if isinstance(events, str) else tuple(events)
        registration = (patterns, callback) if callback is not None else None
        if callback is not None:
            assert registration is not None
            self._callbacks.append(registration)
        try:
            return await self._client.call("events.subscribe", {"events": list(patterns)})
        except BaseException:
            if registration is not None:
                self._callbacks = [
                    item for item in self._callbacks if item is not registration
                ]
            raise

    async def unsubscribe(self, events: str | Iterable[str] | None = None) -> Mapping[str, Any]:
        patterns = None if events is None else ((events,) if isinstance(events, str) else tuple(events))
        result = await self._client.call(
            "events.unsubscribe", {} if patterns is None else {"events": list(patterns)}
        )
        if patterns is None:
            self._callbacks.clear()
        return result

    async def status(self) -> Mapping[str, Any]:
        return await self._client.call("events.get_status")

    def remove_callback(self, callback: AsyncEventCallback) -> None:
        self._callbacks = [item for item in self._callbacks if item[1] is not callback]

    async def next(self, timeout: float | None = None) -> Event:
        if timeout is None:
            event = await self._queue.get()
        else:
            event = await asyncio.wait_for(self._queue.get(), timeout)
        if event is None:
            raise ConnectionClosedError("Odon event stream is closed")
        return event

    async def iter(self) -> AsyncIterator[Event]:
        while True:
            try:
                yield await self.next()
            except ConnectionClosedError:
                return

    def _receive(self, params: Mapping[str, Any]) -> None:
        if self._closed:
            return
        event = Event.from_params(params)
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self.dropped_events += 1
        for patterns, callback in tuple(self._callbacks):
            if any(pattern_matches(pattern, event.name) for pattern in patterns):
                try:
                    result = callback(event)
                    if inspect.isawaitable(result):
                        asyncio.create_task(self._await_callback(result, event))
                except Exception:
                    logger.exception("Odon async event callback failed for %s", event.name)
                    continue

    async def _await_callback(self, result: Awaitable[Any], event: Event) -> None:
        try:
            await result
        except Exception:
            logger.exception("Odon async event callback failed for %s", event.name)

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            # Closing takes precedence over delivery of already-buffered events.
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait(None)
