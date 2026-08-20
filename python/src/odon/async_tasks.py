"""Awaitable handles for Odon-managed long-running operations."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from .errors import RequestTimeoutError, TaskCancelledError, TaskFailedError
from .models import Event, TaskSnapshot

if TYPE_CHECKING:
    from .async_client import AsyncClient

logger = logging.getLogger("odon.tasks")


async def _run_progress_callback(awaitable: Any, task_id: str) -> None:
    try:
        await awaitable
    except Exception:
        logger.exception("Odon task progress callback failed for %s", task_id)


class AsyncTask:
    def __init__(self, tasks: "AsyncTasks", snapshot: TaskSnapshot) -> None:
        self._tasks = tasks
        self.snapshot = snapshot

    @property
    def task_id(self) -> str:
        return self.snapshot.task_id

    def __await__(self):
        return self.wait().__await__()

    async def refresh(self) -> TaskSnapshot:
        self.snapshot = (await self._tasks.get(self.task_id)).snapshot
        return self.snapshot

    async def wait(
        self,
        timeout: float | None = None,
        *,
        progress: Callable[[TaskSnapshot], Any] | None = None,
    ) -> Any:
        completed = asyncio.Event()

        def receive(event: Event) -> None:
            if event.source == self.task_id and isinstance(event.data, Mapping):
                self.snapshot = TaskSnapshot.from_result(event.data)
                if self.snapshot.done:
                    completed.set()
                if progress is not None:
                    result = progress(self.snapshot)
                    if inspect.isawaitable(result):
                        asyncio.create_task(
                            _run_progress_callback(result, self.task_id)
                        )

        await self._tasks._client.events.subscribe("tasks.*", receive)
        try:
            if (await self.refresh()).done:
                return self._result()
            try:
                if timeout is None:
                    await completed.wait()
                else:
                    await asyncio.wait_for(completed.wait(), timeout)
            except TimeoutError as error:
                raise RequestTimeoutError(
                    f"timed out waiting for task {self.task_id!r}"
                ) from error
            return self._result()
        finally:
            self._tasks._client.events.remove_callback(receive)

    async def cancel(self) -> TaskSnapshot:
        self.snapshot = (await self._tasks.cancel(self.task_id)).snapshot
        return self.snapshot

    async def forget(self) -> None:
        await self._tasks.forget(self.task_id)

    def _result(self) -> Any:
        if self.snapshot.state == "cancelled":
            raise TaskCancelledError(f"task {self.task_id!r} was cancelled")
        if self.snapshot.state == "failed":
            raise TaskFailedError(self.task_id, self.snapshot.error)
        return self.snapshot.result


class AsyncTasks:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def start(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        label: str | None = None,
    ) -> AsyncTask:
        request: dict[str, Any] = {"method": method, "params": dict(params or {})}
        if label is not None:
            request["label"] = label
        result = await self._client.call("tasks.start", request)
        return AsyncTask(self, TaskSnapshot.from_result(result))

    async def get(self, task_id: str) -> AsyncTask:
        result = await self._client.call("tasks.get", {"task_id": task_id})
        return AsyncTask(self, TaskSnapshot.from_result(result))

    async def list(self, *, include_finished: bool = True) -> list[AsyncTask]:
        result = await self._client.call(
            "tasks.list", {"include_finished": include_finished}
        )
        return [AsyncTask(self, TaskSnapshot.from_result(item)) for item in result["tasks"]]

    async def cancel(self, task_id: str) -> AsyncTask:
        result = await self._client.call("tasks.cancel", {"task_id": task_id})
        return AsyncTask(self, TaskSnapshot.from_result(result))

    async def forget(self, task_id: str) -> None:
        await self._client.call("tasks.forget", {"task_id": task_id})
