from __future__ import annotations

import asyncio
import threading
import unittest
from typing import Any, Mapping

from odon.async_events import AsyncEvents
from odon.async_tasks import AsyncTask
from odon.errors import UnsafeCallbackWaitError
from odon.events import Events
from odon.models import Event, TaskSnapshot
from odon.tasks import Task


def task_snapshot(state: str = "running") -> TaskSnapshot:
    return TaskSnapshot.from_result(
        {
            "task_id": "task:callback-wait",
            "label": "Callback wait regression",
            "state": state,
            "progress": None,
            "phase": state,
            "phase_details": None,
            "result": {"ok": True} if state == "completed" else None,
            "error": None,
            "created_at_unix_ms": 1,
            "completed_at_unix_ms": 2 if state == "completed" else None,
            "cancellation_supported": False,
            "owner_session_id": "test",
        }
    )


class FakeClient:
    def __init__(self) -> None:
        self.closed = False
        self.calls: list[tuple[str, Mapping[str, Any]]] = []
        self.events = Events(self)  # type: ignore[arg-type]
        self.current = task_snapshot()

    def call(
        self, method: str, params: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        self.calls.append((method, dict(params or {})))
        if method == "tasks.get":
            return dict(self.current.__dict__)
        return {}


class FakeTasks:
    def __init__(self, client: FakeClient) -> None:
        self._client = client

    def get(self, _task_id: str) -> Task:
        return Task(self, self._client.current)  # type: ignore[arg-type]


def ui_event(sequence: int) -> Mapping[str, Any]:
    return {
        "event": "ui.extension:org.example.action",
        "sequence": sequence,
        "revision": sequence,
        "source": "ui:run",
        "data": {
            "component_id": "run",
            "value": None,
            "action": {"type": "emit", "event": "run"},
        },
    }


class EventTaskSafetyTests(unittest.TestCase):
    def test_wait_from_callback_fails_immediately_and_delivery_continues(self) -> None:
        client = FakeClient()
        task = Task(FakeTasks(client), client.current)  # type: ignore[arg-type]
        calls: list[int] = []
        errors: list[UnsafeCallbackWaitError] = []
        delivered = threading.Event()

        def callback(event: Event) -> None:
            calls.append(event.sequence)
            try:
                task.wait(timeout=5)
            except UnsafeCallbackWaitError as error:
                errors.append(error)
            if len(calls) == 2:
                delivered.set()

        try:
            client.events.subscribe("ui.extension:org.example.*", callback)
            client.events._receive(ui_event(1))
            client.events._receive(ui_event(2))
            self.assertTrue(delivered.wait(1), "callback delivery stalled after unsafe wait")
        finally:
            client.events._close()

        self.assertEqual(calls, [1, 2])
        self.assertEqual(len(errors), 2)
        self.assertEqual(errors[0].task_id, task.task_id)
        self.assertIn("extension action worker", str(errors[0]))
        self.assertNotIn("tasks.get", [method for method, _ in client.calls])

    def test_completed_task_is_still_rejected_inside_callback(self) -> None:
        client = FakeClient()
        client.current = task_snapshot("completed")
        task = Task(FakeTasks(client), client.current)  # type: ignore[arg-type]
        observed = threading.Event()

        def callback(_event: Event) -> None:
            with self.assertRaises(UnsafeCallbackWaitError):
                task.wait(timeout=1)
            observed.set()

        try:
            client.events.subscribe("ui.extension:org.example.*", callback)
            client.events._receive(ui_event(1))
            self.assertTrue(observed.wait(1))
        finally:
            client.events._close()

    def test_wait_outside_callback_is_unchanged(self) -> None:
        client = FakeClient()
        client.current = task_snapshot("completed")
        task = Task(FakeTasks(client), client.current)  # type: ignore[arg-type]
        try:
            self.assertEqual(task.wait(timeout=1), {"ok": True})
        finally:
            client.events._close()


class AsyncFakeClient:
    def __init__(self) -> None:
        self.closed = False
        self.events = AsyncEvents(self)  # type: ignore[arg-type]
        self.current = task_snapshot()

    async def call(
        self, method: str, params: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        if method == "tasks.get":
            return dict(self.current.__dict__)
        return {}


class AsyncFakeTasks:
    def __init__(self, client: AsyncFakeClient) -> None:
        self._client = client

    async def get(self, _task_id: str) -> AsyncTask:
        return AsyncTask(self, self._client.current)  # type: ignore[arg-type]


class AsyncEventTaskSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_event_callback_may_await_task_without_starving_delivery(self) -> None:
        client = AsyncFakeClient()
        task = AsyncTask(AsyncFakeTasks(client), client.current)  # type: ignore[arg-type]
        result: list[Any] = []
        done = asyncio.Event()

        async def callback(_event: Event) -> None:
            result.append(await task.wait(timeout=1))
            done.set()

        await client.events.subscribe("ui.extension:org.example.*", callback)
        client.events._receive(ui_event(1))
        await asyncio.sleep(0)
        terminal = task_snapshot("completed")
        client.current = terminal
        client.events._receive(
            {
                "event": "tasks.completed",
                "sequence": 2,
                "revision": 2,
                "source": terminal.task_id,
                "data": dict(terminal.__dict__),
            }
        )
        await asyncio.wait_for(done.wait(), 1)
        self.assertEqual(result, [{"ok": True}])
        client.events._close()


if __name__ == "__main__":
    unittest.main()
