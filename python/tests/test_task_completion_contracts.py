from __future__ import annotations

import asyncio
import threading
import unittest
from typing import Any, Callable

from odon.async_tasks import AsyncTask
from odon.errors import RequestTimeoutError, TaskCancelledError
from odon.models import Event, TaskSnapshot
from odon.tasks import Task


def snapshot(
    state: str = "running",
    *,
    phase: str = "waiting_for_presentation",
    result: Any = None,
    phase_details: Any = None,
) -> TaskSnapshot:
    return TaskSnapshot.from_result(
        {
            "task_id": "task:capture",
            "label": "Capture screenshot",
            "state": state,
            "progress": None,
            "phase": phase,
            "phase_details": phase_details,
            "result": result,
            "error": None,
            "created_at_unix_ms": 1,
            "completed_at_unix_ms": 2 if state != "running" else None,
            "cancellation_supported": True,
            "owner_session_id": "test",
        }
    )


def completion_event(value: TaskSnapshot) -> Event:
    return Event(
        name=f"tasks.{value.state}",
        sequence=1,
        revision=1,
        source=value.task_id,
        data={
            **value.__dict__,
            "state": value.state,
        },
    )


class SyncEvents:
    def __init__(self) -> None:
        self.callback: Callable[[Event], None] | None = None

    def subscribe(self, _pattern: str, callback: Callable[[Event], None]) -> None:
        self.callback = callback

    def remove_callback(self, callback: Callable[[Event], None]) -> None:
        if self.callback is callback:
            self.callback = None

    def emit(self, event: Event) -> None:
        assert self.callback is not None
        self.callback(event)


class SyncTasks:
    def __init__(self, current: TaskSnapshot) -> None:
        self.current = current
        self._client = type("Client", (), {"events": SyncEvents()})()

    def get(self, _task_id: str) -> Task:
        return Task(self, self.current)  # type: ignore[arg-type]


class AsyncEvents:
    def __init__(self) -> None:
        self.callback: Callable[[Event], None] | None = None

    async def subscribe(
        self, _pattern: str, callback: Callable[[Event], None]
    ) -> None:
        self.callback = callback

    def remove_callback(self, callback: Callable[[Event], None]) -> None:
        if self.callback is callback:
            self.callback = None

    def emit(self, event: Event) -> None:
        assert self.callback is not None
        self.callback(event)


class AsyncTasks:
    def __init__(self, current: TaskSnapshot) -> None:
        self.current = current
        self._client = type("Client", (), {"events": AsyncEvents()})()

    async def get(self, _task_id: str) -> AsyncTask:
        return AsyncTask(self, self.current)  # type: ignore[arg-type]


class SyncCompletionContractTests(unittest.TestCase):
    def test_phase_details_are_typed_and_wait_timeout_does_not_cancel(self) -> None:
        details = {
            "capture_id": 7,
            "desired_projection_revision": 42,
            "resource_generations": {"document": 3},
        }
        tasks = SyncTasks(snapshot(phase_details=details))
        task = Task(tasks, tasks.current)  # type: ignore[arg-type]
        with self.assertRaises(RequestTimeoutError):
            task.wait(timeout=0.01)
        self.assertEqual(task.snapshot.state, "running")
        self.assertEqual(task.snapshot.phase_details, details)

    def test_sync_wait_observes_success_and_cancellation(self) -> None:
        for terminal, expected in (
            (snapshot("completed", phase="completed", result={"ok": True}), {"ok": True}),
            (snapshot("cancelled", phase="cancelled"), TaskCancelledError),
        ):
            with self.subTest(state=terminal.state):
                tasks = SyncTasks(snapshot())
                task = Task(tasks, tasks.current)  # type: ignore[arg-type]
                timer = threading.Timer(
                    0.01,
                    tasks._client.events.emit,
                    args=(completion_event(terminal),),
                )
                timer.start()
                try:
                    if isinstance(expected, type):
                        with self.assertRaises(expected):
                            task.wait(timeout=1)
                    else:
                        self.assertEqual(task.wait(timeout=1), expected)
                finally:
                    timer.join(timeout=1)


class AsyncCompletionContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_wait_yields_while_unrelated_work_progresses(self) -> None:
        tasks = AsyncTasks(snapshot())
        task = AsyncTask(tasks, tasks.current)  # type: ignore[arg-type]
        waiting = asyncio.create_task(task.wait(timeout=1))
        await asyncio.sleep(0)
        self.assertFalse(waiting.done())

        unrelated_completed = False

        async def unrelated() -> None:
            nonlocal unrelated_completed
            await asyncio.sleep(0)
            unrelated_completed = True

        await unrelated()
        self.assertTrue(unrelated_completed)
        tasks._client.events.emit(
            completion_event(snapshot("completed", phase="completed", result=9))
        )
        self.assertEqual(await waiting, 9)

    async def test_async_timeout_and_cancellation_are_distinct(self) -> None:
        running_tasks = AsyncTasks(snapshot())
        running = AsyncTask(running_tasks, running_tasks.current)  # type: ignore[arg-type]
        with self.assertRaises(RequestTimeoutError):
            await running.wait(timeout=0.01)
        self.assertEqual(running.snapshot.state, "running")

        cancelled_tasks = AsyncTasks(snapshot("cancelled", phase="cancelled"))
        cancelled = AsyncTask(cancelled_tasks, cancelled_tasks.current)  # type: ignore[arg-type]
        with self.assertRaises(TaskCancelledError):
            await cancelled.wait(timeout=1)


if __name__ == "__main__":
    unittest.main()
