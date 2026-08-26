from __future__ import annotations

import asyncio
import threading
import time
import unittest
from typing import Any, Mapping

from odon.async_events import AsyncEvents
from odon.async_ui import AsyncUi
from odon.errors import (
    ActionQueueFullError,
    ActionRejectedError,
    UnsafeCallbackWaitError,
)
from odon.events import Events
from odon.models import Event, TaskSnapshot
from odon.tasks import Task
from odon.ui import Ui
from odon.ui_actions import (
    ActionContext,
    ActionRegistration,
    AsyncActionRegistration,
    UiInteraction,
)


EXTENSION_ID = "org.example.actions"


def snapshot(state: str = "running", *, result: Any = None) -> TaskSnapshot:
    return TaskSnapshot.from_result(
        {
            "task_id": "task:action",
            "label": "Action task",
            "state": state,
            "progress": None,
            "phase": state,
            "phase_details": None,
            "result": result,
            "error": None,
            "created_at_unix_ms": 1,
            "completed_at_unix_ms": 2 if state != "running" else None,
            "cancellation_supported": True,
            "owner_session_id": "test",
        }
    )


def completion_event(value: TaskSnapshot) -> Mapping[str, Any]:
    return {
        "event": f"tasks.{value.state}",
        "sequence": 50,
        "revision": 50,
        "source": value.task_id,
        "data": {**value.__dict__, "state": value.state},
    }


def interaction(action: str, *, value: Any = None, sequence: int = 1) -> UiInteraction:
    event = Event(
        name=f"ui.extension:{EXTENSION_ID}.action",
        sequence=sequence,
        revision=sequence,
        source=f"ui:{action}",
        data={
            "component_id": action,
            "value": value,
            "action": {"type": "emit", "event": action},
        },
    )
    return UiInteraction.from_event(event, extension_id=EXTENSION_ID)


def interaction_params(action: str, *, sequence: int = 1) -> Mapping[str, Any]:
    item = interaction(action, sequence=sequence).event
    return {
        "event": item.name,
        "sequence": item.sequence,
        "revision": item.revision,
        "source": item.source,
        "data": item.data,
    }


class SyncTasks:
    def __init__(self, client: "SyncClient") -> None:
        self._client = client

    def get(self, _task_id: str) -> Task:
        return Task(self, self._client.current)  # type: ignore[arg-type]


class SyncClient:
    def __init__(self) -> None:
        self.closed = False
        self.calls: list[tuple[str, Mapping[str, Any]]] = []
        self.events = Events(self)  # type: ignore[arg-type]
        self.current = snapshot()

    def call(
        self, method: str, params: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        values = dict(params or {})
        self.calls.append((method, values))
        if method == "ui.extensions.register":
            return {**values, "granted_capabilities": values["capabilities"]}
        if method == "tasks.get":
            return dict(self.current.__dict__)
        return {}


class Contribution:
    def __init__(self) -> None:
        self.patches: list[Mapping[str, Any]] = []

    def patch_values(self, values: Mapping[str, Any]) -> None:
        self.patches.append(dict(values))


class CancellableTask:
    snapshot = type("Snapshot", (), {"cancellation_supported": True})()

    def __init__(self) -> None:
        self.cancelled = threading.Event()

    def cancel(self) -> None:
        self.cancelled.set()


class SyncActionWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = SyncClient()
        self.extension = Ui(self.client).register_extension(
            id=EXTENSION_ID, name="Actions", version="1"
        )

    def tearDown(self) -> None:
        self.extension._close_local()
        self.client.events._close()

    def test_serial_worker_can_wait_without_starving_task_completion(self) -> None:
        tasks = SyncTasks(self.client)
        task = Task(tasks, self.client.current)  # type: ignore[arg-type]
        completed = threading.Event()
        result: list[Any] = []

        def handler(_context: ActionContext, _interaction: UiInteraction) -> None:
            terminal = snapshot("completed", result={"loaded": True})

            def finish() -> None:
                self.client.current = terminal
                self.client.events._receive(completion_event(terminal))

            timer = threading.Timer(0.02, finish)
            timer.start()
            try:
                result.append(task.wait(timeout=1))
            finally:
                timer.join(timeout=1)
            completed.set()

        registration = self.extension.on_action("load", handler)
        self.assertIsInstance(registration, ActionRegistration)
        self.client.events._receive(interaction_params("load"))
        self.assertTrue(completed.wait(1))
        self.assertEqual(result, [{"loaded": True}])
        status = self.extension.action_status()
        self.assertEqual(status.completed, 1)
        self.assertEqual(status.failed, 0)
        self.assertEqual(status.executed, 1)

    def test_callback_execution_reports_unsafe_wait_and_keeps_delivery_alive(self) -> None:
        task = Task(SyncTasks(self.client), self.client.current)  # type: ignore[arg-type]
        errors: list[BaseException] = []
        delivered = threading.Event()

        def handler(_context: ActionContext, _interaction: UiInteraction) -> None:
            task.wait(timeout=1)

        def on_error(error: BaseException, _context: ActionContext | None) -> None:
            errors.append(error)
            if len(errors) == 2:
                delivered.set()

        self.extension.on_action(
            "unsafe", handler, execution="callback", on_error=on_error
        )
        self.client.events._receive(interaction_params("unsafe", sequence=1))
        self.client.events._receive(interaction_params("unsafe", sequence=2))
        self.assertTrue(delivered.wait(1))
        self.assertTrue(all(isinstance(error, UnsafeCallbackWaitError) for error in errors))
        self.assertEqual(self.extension.action_status().failed, 2)

    def test_serial_worker_contains_failures_and_runs_the_next_action(self) -> None:
        calls: list[int] = []
        done = threading.Event()

        def handler(_context: ActionContext, item: UiInteraction) -> None:
            calls.append(item.event.sequence)
            if item.event.sequence == 1:
                raise RuntimeError("first action failed")
            done.set()

        errors: list[str] = []
        registration = self.extension.on_action(
            "run",
            handler,
            on_error=lambda error, _context: errors.append(str(error)),
        )
        assert isinstance(registration, ActionRegistration)
        registration.submit(interaction("run", sequence=1))
        registration.submit(interaction("run", sequence=2))
        self.assertTrue(done.wait(1))
        self.assertEqual(calls, [1, 2])
        self.assertEqual(errors, ["first action failed"])
        status = self.extension.action_status()
        self.assertEqual(status.failed, 1)
        self.assertEqual(status.completed, 1)
        self.assertEqual(status.executed, 2)

    def test_latest_coalesces_pending_actions_and_status_is_generation_safe(self) -> None:
        release = threading.Event()
        first_started = threading.Event()
        seen: list[int] = []
        contribution = Contribution()

        def handler(context: ActionContext, item: UiInteraction) -> None:
            if item.event.sequence == 1:
                first_started.set()
                release.wait(1)
            seen.append(item.event.sequence)
            context.result(f"Ready {item.event.sequence}")

        registration = self.extension.on_action(
            "latest",
            handler,
            coalesce="latest",
            queue_key="selection",
            contribution=contribution,  # type: ignore[arg-type]
            status_component_id="status",
        )
        assert isinstance(registration, ActionRegistration)
        registration.submit(interaction("latest", sequence=1))
        self.assertTrue(first_started.wait(1))
        registration.submit(interaction("latest", sequence=2))
        registration.submit(interaction("latest", sequence=3))
        release.set()
        deadline = time.monotonic() + 1
        while self.extension.action_status().completed < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(seen, [1, 3])
        status = self.extension.action_status()
        self.assertEqual(status.submitted, 3)
        self.assertEqual(status.completed, 1)
        self.assertEqual(status.cancelled, 1)
        self.assertEqual(status.coalesced, 1)
        self.assertEqual(contribution.patches, [{"status": "Ready 3"}])

    def test_accumulate_combines_navigation_deltas(self) -> None:
        release = threading.Event()
        started = threading.Event()
        deltas: list[float] = []

        def navigate(context: ActionContext, item: UiInteraction) -> None:
            if item.event.sequence == 1:
                started.set()
                release.wait(1)
            deltas.append(context.delta)

        next_action = self.extension.on_action(
            "next",
            navigate,
            queue_key="navigation",
            coalesce="accumulate",
            delta=1,
        )
        previous_action = self.extension.on_action(
            "previous",
            navigate,
            queue_key="navigation",
            coalesce="accumulate",
            delta=-1,
        )
        assert isinstance(next_action, ActionRegistration)
        assert isinstance(previous_action, ActionRegistration)
        next_action.submit(interaction("next", sequence=1))
        self.assertTrue(started.wait(1))
        next_action.submit(interaction("next", sequence=2))
        next_action.submit(interaction("next", sequence=3))
        previous_action.submit(interaction("previous", sequence=4))
        release.set()
        deadline = time.monotonic() + 1
        while len(deltas) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(deltas, [1.0, 1.0])
        status = self.extension.action_status()
        self.assertEqual(status.submitted, 4)
        self.assertEqual(status.coalesced, 2)
        self.assertEqual(status.executed, 2)

    def test_all_policy_does_not_stale_an_active_ordered_action(self) -> None:
        release = threading.Event()
        started = threading.Event()
        completed: list[int] = []

        def handler(context: ActionContext, item: UiInteraction) -> None:
            if item.event.sequence == 1:
                started.set()
                release.wait(1)
            context.ensure_current()
            completed.append(item.event.sequence)

        registration = self.extension.on_action("ordered", handler, coalesce="all")
        assert isinstance(registration, ActionRegistration)
        registration.submit(interaction("ordered", sequence=1))
        self.assertTrue(started.wait(1))
        registration.submit(interaction("ordered", sequence=2))
        release.set()
        deadline = time.monotonic() + 1
        while len(completed) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(completed, [1, 2])

    def test_worker_policy_allows_independent_actions_to_overlap(self) -> None:
        release = threading.Event()
        both_started = threading.Event()
        started: list[int] = []
        lock = threading.Lock()

        def handler(_context: ActionContext, item: UiInteraction) -> None:
            with lock:
                started.append(item.event.sequence)
                if len(started) == 2:
                    both_started.set()
            release.wait(1)

        registration = self.extension.on_action(
            "parallel", handler, execution="worker", concurrent_workers=2
        )
        assert isinstance(registration, ActionRegistration)
        registration.submit(interaction("parallel", sequence=1))
        registration.submit(interaction("parallel", sequence=2))
        self.assertTrue(both_started.wait(1))
        release.set()
        deadline = time.monotonic() + 1
        while (
            self.extension.action_status().completed < 2
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        self.assertCountEqual(started, [1, 2])

    def test_extension_local_close_removes_actions_and_closes_runner(self) -> None:
        registration = self.extension.on_action(
            "cleanup", lambda _context, _item: None
        )
        assert isinstance(registration, ActionRegistration)
        runner = self.extension._action_runner
        assert runner is not None
        self.extension._close_local()
        self.assertTrue(registration.removed)
        self.assertTrue(runner.snapshot().closed)
        self.assertIsNone(self.extension._action_runner)
        self.assertFalse(self.extension._interaction_callbacks)

    def test_callback_policy_does_not_start_worker_infrastructure(self) -> None:
        registration = self.extension.on_action(
            "inspect", lambda _context, _item: None, execution="callback"
        )
        assert isinstance(registration, ActionRegistration)
        runner = self.extension._action_runner
        assert runner is not None
        self.assertIsNone(runner._serial_worker)
        self.assertIsNone(runner._executor)
        registration.submit(interaction("inspect"))
        self.assertIsNone(runner._serial_worker)
        self.assertIsNone(runner._executor)

    def test_accumulate_can_cancel_a_pending_delta_to_zero(self) -> None:
        release = threading.Event()
        started = threading.Event()
        deltas: list[float] = []

        def navigate(context: ActionContext, item: UiInteraction) -> None:
            if item.event.sequence == 1:
                started.set()
                release.wait(1)
            deltas.append(context.delta)

        next_action = self.extension.on_action(
            "next", navigate, queue_key="navigation", coalesce="accumulate", delta=1
        )
        previous_action = self.extension.on_action(
            "previous",
            navigate,
            queue_key="navigation",
            coalesce="accumulate",
            delta=-1,
        )
        assert isinstance(next_action, ActionRegistration)
        assert isinstance(previous_action, ActionRegistration)
        next_action.submit(interaction("next", sequence=1))
        self.assertTrue(started.wait(1))
        next_action.submit(interaction("next", sequence=2))
        previous_action.submit(interaction("previous", sequence=3))
        release.set()
        deadline = time.monotonic() + 1
        while not deltas and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.03)
        self.assertEqual(deltas, [1.0])
        status = self.extension.action_status()
        self.assertEqual(status.submitted, 3)
        self.assertEqual(status.coalesced, 1)

    def test_bounded_queue_rejects_overflow_and_reports_reason(self) -> None:
        release = threading.Event()
        started = threading.Event()
        errors: list[BaseException] = []

        def handler(_context: ActionContext, _item: UiInteraction) -> None:
            started.set()
            release.wait(1)

        registration = self.extension.on_action(
            "bounded", handler, max_queue=1, on_error=lambda error, _ctx: errors.append(error)
        )
        assert isinstance(registration, ActionRegistration)
        registration.submit(interaction("bounded", sequence=1))
        self.assertTrue(started.wait(1))
        self.assertTrue(registration.submit(interaction("bounded", sequence=2)))
        self.assertFalse(registration.submit(interaction("bounded", sequence=3)))
        release.set()
        self.assertIsInstance(errors[0], ActionQueueFullError)
        self.assertEqual(self.extension.action_status().rejected, 1)

    def test_extension_close_is_bounded_and_cancels_pending_work(self) -> None:
        release = threading.Event()
        started = threading.Event()

        def handler(_context: ActionContext, _item: UiInteraction) -> None:
            started.set()
            release.wait(2)

        registration = self.extension.on_action("close", handler)
        assert isinstance(registration, ActionRegistration)
        registration.submit(interaction("close", sequence=1))
        self.assertTrue(started.wait(1))
        registration.submit(interaction("close", sequence=2))
        began = time.monotonic()
        assert self.extension._action_runner is not None
        self.extension._action_runner.close(timeout=0.05)
        elapsed = time.monotonic() - began
        release.set()
        self.assertLess(elapsed, 0.3)
        self.assertGreaterEqual(self.extension.action_status().cancelled, 1)

    def test_removing_registration_cooperatively_cancels_attached_task(self) -> None:
        task = CancellableTask()
        started = threading.Event()
        release = threading.Event()

        def handler(context: ActionContext, _item: UiInteraction) -> None:
            context.attach(task)
            started.set()
            release.wait(1)
            context.check_cancelled()

        registration = self.extension.on_action("cancel", handler)
        assert isinstance(registration, ActionRegistration)
        registration.submit(interaction("cancel"))
        self.assertTrue(started.wait(1))
        registration.remove()
        self.assertTrue(task.cancelled.wait(1))
        release.set()

    def test_reject_while_busy_is_explicit(self) -> None:
        release = threading.Event()
        started = threading.Event()
        errors: list[BaseException] = []

        def handler(_context: ActionContext, _item: UiInteraction) -> None:
            started.set()
            release.wait(1)

        registration = self.extension.on_action(
            "export",
            handler,
            coalesce="reject-while-busy",
            on_error=lambda error, _context: errors.append(error),
        )
        assert isinstance(registration, ActionRegistration)
        self.assertTrue(registration.submit(interaction("export", sequence=1)))
        self.assertTrue(started.wait(1))
        self.assertFalse(registration.submit(interaction("export", sequence=2)))
        release.set()
        self.assertIsInstance(errors[0], ActionRejectedError)
        self.assertEqual(self.extension.action_status().rejected, 1)


class AsyncClient:
    def __init__(self) -> None:
        self.closed = False
        self.calls: list[tuple[str, Mapping[str, Any]]] = []
        self.events = AsyncEvents(self)  # type: ignore[arg-type]

    async def call(
        self, method: str, params: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        values = dict(params or {})
        self.calls.append((method, values))
        if method == "ui.extensions.register":
            return {**values, "granted_capabilities": values["capabilities"]}
        return {}


class AsyncActionWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_serial_worker_preserves_order_and_contains_failure(self) -> None:
        client = AsyncClient()
        extension = await AsyncUi(client).register_extension(
            id=EXTENSION_ID, name="Actions", version="1"
        )
        calls: list[int] = []
        errors: list[str] = []
        finished = asyncio.Event()

        async def handler(_context: Any, item: UiInteraction) -> None:
            calls.append(item.event.sequence)
            await asyncio.sleep(0)
            if item.event.sequence == 1:
                raise RuntimeError("async failure")
            finished.set()

        registration = await extension.on_action(
            "run",
            handler,
            on_error=lambda error, _context: errors.append(str(error)),
        )
        self.assertIsInstance(registration, AsyncActionRegistration)
        registration.submit(interaction("run", sequence=1))
        registration.submit(interaction("run", sequence=2))
        await asyncio.wait_for(finished.wait(), 1)
        self.assertEqual(calls, [1, 2])
        self.assertEqual(errors, ["async failure"])
        self.assertEqual(extension.action_status().completed, 1)
        self.assertEqual(extension.action_status().executed, 2)
        runner = extension._action_runner
        assert runner is not None
        await extension._close_local()
        self.assertTrue(registration.removed)
        self.assertTrue(runner.snapshot().closed)
        client.events._close()

    async def test_async_worker_latest_cancels_pending_and_stales_active(self) -> None:
        client = AsyncClient()
        extension = await AsyncUi(client).register_extension(
            id=EXTENSION_ID, name="Actions", version="1"
        )
        started = asyncio.Event()
        release = asyncio.Event()
        seen: list[int] = []

        async def handler(context: Any, item: UiInteraction) -> None:
            if item.event.sequence == 1:
                started.set()
                await release.wait()
            context.ensure_current()
            seen.append(item.event.sequence)

        registration = await extension.on_action(
            "latest-worker",
            handler,
            execution="worker",
            coalesce="latest",
            queue_key="selection",
        )
        registration.submit(interaction("latest-worker", sequence=1))
        await asyncio.wait_for(started.wait(), 1)
        registration.submit(interaction("latest-worker", sequence=2))
        registration.submit(interaction("latest-worker", sequence=3))
        release.set()
        deadline = asyncio.get_running_loop().time() + 1
        while extension.action_status().completed < 1:
            if asyncio.get_running_loop().time() >= deadline:
                self.fail(f"latest async action did not finish: {extension.action_status()}")
            await asyncio.sleep(0.005)
        self.assertEqual(seen, [3])
        self.assertEqual(extension.action_status().coalesced, 1)
        await extension._close_local()
        client.events._close()


if __name__ == "__main__":
    unittest.main()
