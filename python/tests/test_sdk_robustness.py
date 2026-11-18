"""Regression tests for connection, discovery, event and task failure modes."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Mapping
from unittest import mock

import odon
from odon import async_tasks, tasks
from odon.async_client import AsyncClient
from odon.async_events import AsyncEvents
from odon.async_tasks import AsyncTask
from odon.client import Client
from odon.discovery import Instance, list_instances
from odon.errors import OdonError, RequestTimeoutError
from odon.events import Events
from odon.launch import _launched_instance
from odon.models import TaskSnapshot
from odon.tasks import Task

ROOT = Path(__file__).resolve().parents[2]


def snapshot(state: str) -> TaskSnapshot:
    done = state == "completed"
    return TaskSnapshot.from_result(
        {
            "task_id": "task:robustness",
            "label": "Robustness",
            "state": state,
            "progress": None,
            "phase": state,
            "phase_details": None,
            "result": {"ok": True} if done else None,
            "error": None,
            "created_at_unix_ms": 1,
            "completed_at_unix_ms": 2 if done else None,
            "cancellation_supported": False,
            "owner_session_id": "test",
        }
    )


def event(sequence: int) -> Mapping[str, Any]:
    return {
        "event": "viewer.camera.changed",
        "sequence": sequence,
        "revision": sequence,
        "source": "test",
        "data": {},
    }


class _SyncClient:
    def __init__(self) -> None:
        self.closed = False
        self.events = Events(self)  # type: ignore[arg-type]
        self.current = snapshot("running")
        self.gets = 0

    def call(self, method: str, params: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        return {}


class _SyncTasks:
    def __init__(self, client: _SyncClient) -> None:
        self._client = client

    def get(self, _task_id: str) -> Task:
        self._client.gets += 1
        # The task finishes in Odon after the first read, but its completion event is lost.
        if self._client.gets >= 2:
            self._client.current = snapshot("completed")
        return Task(self, self._client.current)  # type: ignore[arg-type]


class _AsyncClient:
    def __init__(self) -> None:
        self.closed = False
        self.events = AsyncEvents(self)  # type: ignore[arg-type]
        self.current = snapshot("running")
        self.gets = 0

    async def call(
        self, method: str, params: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        return {}


class _AsyncTasks:
    def __init__(self, client: _AsyncClient) -> None:
        self._client = client

    async def get(self, _task_id: str) -> AsyncTask:
        self._client.gets += 1
        if self._client.gets >= 2:
            self._client.current = snapshot("completed")
        return AsyncTask(self, self._client.current)  # type: ignore[arg-type]


class TaskWaitTests(unittest.TestCase):
    def test_wait_without_timeout_recovers_when_the_completion_event_is_lost(self) -> None:
        client = _SyncClient()
        task = Task(_SyncTasks(client), client.current)  # type: ignore[arg-type]
        try:
            with mock.patch.object(tasks, "COMPLETION_RECHECK_SECONDS", 0.02):
                self.assertEqual(task.wait(), {"ok": True})
        finally:
            client.events._close()
        self.assertGreaterEqual(client.gets, 2)

    def test_wait_still_times_out_while_the_task_is_running(self) -> None:
        client = _SyncClient()
        tasks_resource = _SyncTasks(client)
        tasks_resource.get = lambda _task_id: Task(tasks_resource, snapshot("running"))  # type: ignore[method-assign,arg-type]
        task = Task(tasks_resource, client.current)  # type: ignore[arg-type]
        try:
            with mock.patch.object(tasks, "COMPLETION_RECHECK_SECONDS", 0.01):
                with self.assertRaises(RequestTimeoutError):
                    task.wait(timeout=0.05)
        finally:
            client.events._close()


class AsyncTaskWaitTests(unittest.IsolatedAsyncioTestCase):
    async def test_wait_without_timeout_recovers_when_the_completion_event_is_lost(self) -> None:
        client = _AsyncClient()
        task = AsyncTask(_AsyncTasks(client), client.current)  # type: ignore[arg-type]
        with mock.patch.object(async_tasks, "COMPLETION_RECHECK_SECONDS", 0.02):
            self.assertEqual(await asyncio.wait_for(task.wait(), 2), {"ok": True})
        client.events._close()

    async def test_call_timeout_is_reported_as_a_request_timeout(self) -> None:
        async def never_answers(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            await reader.read()
            writer.close()

        server = await asyncio.start_server(never_answers, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        client = AsyncClient(reader, writer, timeout=0.05)
        try:
            with self.assertRaises(RequestTimeoutError):
                await client.call("app.get_state")
        finally:
            await client.close()
            server.close()
            await server.wait_closed()


class EventQueueTests(unittest.TestCase):
    def test_an_unread_polling_queue_does_not_report_dropped_events(self) -> None:
        client = _SyncClient()
        events = Events(client, max_queue=4)  # type: ignore[arg-type]
        try:
            for sequence in range(1, 21):
                events._receive(event(sequence))
                time.sleep(0.001)  # let the callback worker drain its own queue
            self.assertEqual(events.dropped_events, 0)
            # The most recent events are still there for a consumer that starts late.
            self.assertEqual(events.next(timeout=1).sequence, 17)
            # Once something polls, falling behind is a real loss and is reported.
            for sequence in range(21, 31):
                events._receive(event(sequence))
                time.sleep(0.001)
            self.assertGreater(events.dropped_events, 0)
        finally:
            events._close()
            client.events._close()


class AsyncEventQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_an_unread_polling_queue_does_not_report_dropped_events(self) -> None:
        events = AsyncEvents(_AsyncClient(), max_queue=4)  # type: ignore[arg-type]
        for sequence in range(1, 21):
            events._receive(event(sequence))
        self.assertEqual(events.dropped_events, 0)
        self.assertEqual((await events.next(timeout=1)).sequence, 17)
        for sequence in range(21, 31):
            events._receive(event(sequence))
        self.assertGreater(events.dropped_events, 0)
        events._close()


class DiscoveryTests(unittest.TestCase):
    def _manifest(self, directory: Path, name: str, pid: int, port: int) -> Path:
        path = directory / f"instance-{name}.json"
        path.write_text(
            json.dumps(
                {
                    "instance_id": name,
                    "pid": pid,
                    "endpoint": f"tcp://127.0.0.1:{port}",
                    "token": "secret",
                    "app_version": "0.0.0",
                    "protocol_versions": [1],
                    "started_at_unix_ms": 1,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_only_manifests_of_dead_processes_are_removed(self) -> None:
        # A port that nothing listens on.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            closed_port = probe.getsockname()[1]
        finished = subprocess.Popen([sys.executable, "-c", "pass"])
        finished.wait()
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            busy = self._manifest(runtime, "busy", os.getpid(), closed_port)
            dead = self._manifest(runtime, "dead", finished.pid, closed_port)
            with mock.patch.dict(os.environ, {"ODON_RUNTIME_DIR": directory}):
                self.assertEqual(list_instances(), [])
            self.assertTrue(busy.exists(), "a live process's manifest must survive a missed probe")
            self.assertFalse(dead.exists())


class HandshakeTests(unittest.TestCase):
    def test_a_rejected_handshake_releases_the_connection(self) -> None:
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        peer_closed = threading.Event()

        def serve() -> None:
            connection, _ = listener.accept()
            with connection, connection.makefile("rw", encoding="utf-8") as stream:
                request = json.loads(stream.readline())
                stream.write(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": request["id"],
                            "error": {"code": -32001, "message": "invalid token"},
                        }
                    )
                    + "\n"
                )
                stream.flush()
                # The client must hang up; otherwise this read never returns.
                if stream.readline() == "":
                    peer_closed.set()

        server = threading.Thread(target=serve, daemon=True)
        server.start()
        try:
            with self.assertRaises(OdonError):
                Client("127.0.0.1", port, token="wrong", timeout=2)
            self.assertTrue(peer_closed.wait(2), "the failed client left its socket open")
        finally:
            listener.close()
            server.join(2)


class LaunchSelectionTests(unittest.TestCase):
    def _instance(self, name: str, pid: int, started: int) -> Instance:
        return Instance(
            instance_id=name,
            pid=pid,
            host="127.0.0.1",
            port=1,
            token="t",
            app_version="0",
            protocol_versions=(1,),
            started_at_unix_ms=started,
            manifest_path=Path(name),
        )

    def test_the_child_pid_wins_and_ambiguous_newcomers_are_not_adopted(self) -> None:
        old = self._instance("old", 10, 500)
        ours = self._instance("ours", 42, 1_500)
        other = self._instance("other", 77, 1_600)
        self.assertEqual(_launched_instance([old, other, ours], 42, {"old"}, 1_000), ours)
        # A wrapper started the application under another PID: the single newcomer is ours.
        self.assertEqual(_launched_instance([old, other], 42, {"old"}, 1_000), other)
        # Two newcomers cannot be told apart, and an older manifest is never a newcomer.
        late = self._instance("late", 78, 1_700)
        self.assertIsNone(_launched_instance([old, other, late], 42, {"old"}, 1_000))
        stale = self._instance("stale", 79, 900)
        self.assertIsNone(_launched_instance([old, stale], 42, {"old"}, 1_000))


class PackagingTests(unittest.TestCase):
    def test_the_version_has_one_source(self) -> None:
        from odon._version import __version__

        self.assertEqual(odon.__version__, __version__)
        pyproject = (ROOT / "python" / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('dynamic = ["version"]', pyproject)
        self.assertNotIn(f'version = "{__version__}"', pyproject)

    def test_generated_async_resources_are_current(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_async_resources.py"), "--check"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
