"""Synchronous handles for Odon-managed long-running operations."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from .errors import (
    RequestTimeoutError,
    TaskCancelledError,
    TaskFailedError,
    UnsafeCallbackWaitError,
)
from .models import Event, TaskSnapshot

if TYPE_CHECKING:
    from .client import Client


class Task:
    def __init__(self, tasks: "Tasks", snapshot: TaskSnapshot) -> None:
        self._tasks = tasks
        self.snapshot = snapshot

    @property
    def task_id(self) -> str:
        return self.snapshot.task_id

    def refresh(self) -> TaskSnapshot:
        self.snapshot = self._tasks.get(self.task_id).snapshot
        return self.snapshot

    def wait(
        self,
        timeout: float | None = None,
        *,
        progress: Callable[[TaskSnapshot], Any] | None = None,
    ) -> Any:
        if bool(getattr(self._tasks._client.events, "in_callback", False)):
            raise UnsafeCallbackWaitError(self.task_id)
        completed = threading.Event()

        def receive(event: Event) -> None:
            if event.source == self.task_id and isinstance(event.data, Mapping):
                self.snapshot = TaskSnapshot.from_result(event.data)
                if self.snapshot.done:
                    completed.set()
                if progress is not None:
                    progress(self.snapshot)

        self._tasks._client.events.subscribe("tasks.*", receive)
        try:
            if self.refresh().done:
                return self._result()
            if not completed.wait(timeout):
                raise RequestTimeoutError(
                    f"timed out waiting for task {self.task_id!r}; the retained Odon "
                    "task may still be running (call task.cancel() explicitly to request "
                    "cooperative cancellation)"
                )
            return self._result()
        finally:
            self._tasks._client.events.remove_callback(receive)

    def cancel(self) -> TaskSnapshot:
        self.snapshot = self._tasks.cancel(self.task_id).snapshot
        return self.snapshot

    def forget(self) -> None:
        self._tasks.forget(self.task_id)

    def _result(self) -> Any:
        if self.snapshot.state == "cancelled":
            raise TaskCancelledError(f"task {self.task_id!r} was cancelled")
        if self.snapshot.state == "failed":
            raise TaskFailedError(self.task_id, self.snapshot.error)
        return self.snapshot.result


class Tasks:
    def __init__(self, client: "Client") -> None:
        self._client = client

    def start(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        label: str | None = None,
    ) -> Task:
        request: dict[str, Any] = {"method": method, "params": dict(params or {})}
        if label is not None:
            request["label"] = label
        return Task(self, TaskSnapshot.from_result(self._client.call("tasks.start", request)))

    def get(self, task_id: str) -> Task:
        return Task(
            self,
            TaskSnapshot.from_result(self._client.call("tasks.get", {"task_id": task_id})),
        )

    def list(self, *, include_finished: bool = True) -> list[Task]:
        result = self._client.call("tasks.list", {"include_finished": include_finished})
        return [Task(self, TaskSnapshot.from_result(item)) for item in result["tasks"]]

    def cancel(self, task_id: str) -> Task:
        return Task(
            self,
            TaskSnapshot.from_result(
                self._client.call("tasks.cancel", {"task_id": task_id})
            ),
        )

    def forget(self, task_id: str) -> None:
        self._client.call("tasks.forget", {"task_id": task_id})
