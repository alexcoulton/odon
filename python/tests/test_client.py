from __future__ import annotations

import asyncio
import json
import socketserver
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import odon


class _ControlHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        hello_complete = False
        tasks: dict[str, dict[str, Any]] = {}
        resources: dict[str, dict[str, Any]] = {}
        layers: dict[str, dict[str, Any]] = {}
        for raw_line in self.rfile:
            notification: dict[str, Any] | None = None
            request = json.loads(raw_line)
            request_id = request.get("id")
            method = request.get("method")
            method = {
                "app.get_state": "get_current_view",
                "viewer.camera.get": "get_camera",
                "viewer.camera.set": "set_camera",
            }.get(method, method)
            if method == "system.hello":
                expected_token = getattr(self.server, "expected_token", None)
                if expected_token is not None and request["params"].get("token") != expected_token:
                    response = self._error(request_id, "AUTHENTICATION_FAILED", -32003)
                    self.wfile.write((json.dumps(response) + "\n").encode())
                    self.wfile.flush()
                    return
                expected_client = getattr(self.server, "expected_client", None)
                if (
                    expected_client is not None
                    and request["params"].get("client") != expected_client
                ):
                    response = self._error(request_id, "INVALID_CLIENT", -32602)
                    self.wfile.write((json.dumps(response) + "\n").encode())
                    self.wfile.flush()
                    return
                hello_complete = True
                response: dict[str, Any] = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocol_version": 1,
                        "app_name": "odon",
                        "app_version": "test",
                        "control_api_version": "0.1.0",
                        "instance_id": "test-instance",
                        "session_id": "test-session",
                        "capabilities": ["viewer.read", "viewer.write"],
                        "max_inline_payload_bytes": 1048576,
                        "permission_policy": "local_authenticated_standard",
                        "client": request["params"]["client"],
                    },
                }
            elif not hello_complete:
                response = self._error(request_id, "HANDSHAKE_REQUIRED", -32001)
            elif method == "get_camera":
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "mode": "single",
                        "camera": {
                            "center_world_lvl0": [10.0, 20.0],
                            "zoom_screen_per_lvl0_px": 0.5,
                        },
                    },
                }
            elif method == "set_camera":
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"camera": request["params"]},
                }
            elif method == "events.subscribe":
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"events": request["params"]["events"], "revision": 7},
                }
                notification = {
                    "jsonrpc": "2.0",
                    "method": "events.event",
                    "params": {
                        "event": "viewer.camera.changed",
                        "sequence": 1,
                        "revision": 7,
                        "source": "viewer:active",
                        "data": {
                            "zoom": 2.0,
                            "viewport_id": "viewport-1",
                            "affected_viewport_ids": ["viewport-1", "viewport-2"],
                            "link_transaction_id": "viewport-1-8",
                        },
                        "initiating_session_id": "another-session",
                        "initiating_request_id": 99,
                    },
                }
            elif method == "tasks.start":
                task_id = "task:test"
                completed = {
                    "task_id": task_id,
                    "label": request["params"].get("label", "test"),
                    "state": "completed",
                    "progress": 1.0,
                    "phase": "completed",
                    "result": {"answer": 42},
                    "error": None,
                    "created_at_unix_ms": 1,
                    "completed_at_unix_ms": 2,
                    "cancellation_supported": False,
                    "owner_session_id": "test-session",
                }
                tasks[task_id] = completed
                queued = dict(completed, state="queued", progress=0.0, phase="queued", result=None)
                response = {"jsonrpc": "2.0", "id": request_id, "result": queued}
                notification = {
                    "jsonrpc": "2.0",
                    "method": "events.event",
                    "params": {
                        "event": "tasks.completed",
                        "sequence": 2,
                        "revision": 7,
                        "source": task_id,
                        "data": completed,
                    },
                }
            elif method == "tasks.get":
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": tasks[request["params"]["task_id"]],
                }
            elif method == "data.resources.register":
                item = {
                    **request["params"],
                    "resource_id": request["params"].get("resource_id", "resource:test"),
                    "owner_session_id": "test-session",
                    "revision": 8,
                }
                resources[item["resource_id"]] = item
                response = {"jsonrpc": "2.0", "id": request_id, "result": item}
            elif method == "viewer.layers.add":
                item = {
                    **request["params"],
                    "layer_id": request["params"].get("layer_id", "layer:test"),
                    "owner_session_id": "test-session",
                    "order": len(layers),
                    "revision": 9,
                }
                layers[item["layer_id"]] = item
                response = {"jsonrpc": "2.0", "id": request_id, "result": item}
            elif method == "ui.extensions.register":
                item = {
                    **request["params"],
                    "requested_capabilities": request["params"]["capabilities"],
                    "granted_capabilities": request["params"]["capabilities"],
                    "owner_session_id": "test-session",
                    "connected": True,
                    "revision": 10,
                }
                response = {"jsonrpc": "2.0", "id": request_id, "result": item}
            elif method == "ui.contributions.register":
                item = {
                    **request["params"],
                    "contribution_id": "contribution:test",
                    "revision": 11,
                }
                response = {"jsonrpc": "2.0", "id": request_id, "result": item}
            elif method == "fail":
                response = self._error(request_id, "NOT_READY", -32010)
            elif method == "disconnect":
                return
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"method": method, "params": request.get("params", {})},
                }
            self.wfile.write((json.dumps(response) + "\n").encode())
            if notification is not None:
                self.wfile.write((json.dumps(notification) + "\n").encode())
            self.wfile.flush()

    @staticmethod
    def _error(request_id: Any, kind: str, code: int) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": "test failure",
                "data": {"kind": kind},
            },
        }


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    expected_token: str | None = None
    expected_client: dict[str, str] | None = None


class ClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = _Server(("127.0.0.1", 0), _ControlHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.host, cls.port = cls.server.server_address

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_sync_client_handshake_resources_and_structured_errors(self) -> None:
        with odon.connect(self.host, self.port) as client:
            self.assertEqual(client.hello.protocol_version, 1)
            self.assertIn("viewer.read", client.hello.capabilities)
            self.assertIs(client.viewer.viewport_links, client.viewport_links)
            camera = client.viewer.get_camera()
            self.assertEqual(camera["camera"]["center_world_lvl0"], [10.0, 20.0])
            changed = client.viewer.set_camera(
                center=(30, 40), zoom=2, if_revision=17
            )
            self.assertEqual(changed["camera"]["zoom"], 2.0)
            self.assertEqual(changed["camera"]["if_revision"], 17)

            with self.assertRaises(odon.RemoteError) as raised:
                client.call("fail")
            self.assertEqual(raised.exception.kind, "NOT_READY")

    def test_raw_call_is_available(self) -> None:
        with odon.connect(self.host, self.port) as client:
            result = client.call("custom.method", {"value": 7})
            self.assertEqual(result["params"]["value"], 7)

    def test_connect_forwards_custom_client_identity(self) -> None:
        self.server.expected_client = {
            "name": "odon-two-viewer-demo",
            "version": "demo-1",
        }
        try:
            with odon.connect(
                self.host,
                self.port,
                client_name="odon-two-viewer-demo",
                client_version="demo-1",
            ) as client:
                self.assertEqual(client.hello.instance_id, "test-instance")
        finally:
            self.server.expected_client = None

    def test_close_finishes_cleanup_after_server_disconnects(self) -> None:
        client = odon.connect(self.host, self.port)
        with self.assertRaises(odon.ConnectionClosedError):
            client.call("disconnect", timeout=1)
        client.close()
        self.assertTrue(client._cleanup_complete)

    def test_sync_events_support_iterators_and_callbacks(self) -> None:
        received: list[odon.Event] = []
        with odon.connect(self.host, self.port) as client:
            client.events.subscribe("viewer.camera.*", received.append)
            event = client.events.next(timeout=1)
            self.assertEqual(event.name, "viewer.camera.changed")
            self.assertEqual(event.revision, 7)
            self.assertEqual(event.data["viewport_id"], "viewport-1")
            self.assertEqual(
                event.data["affected_viewport_ids"],
                ["viewport-1", "viewport-2"],
            )
            self.assertEqual(event.data["link_transaction_id"], "viewport-1-8")
            self.assertEqual(event.initiating_session_id, "another-session")
            self.assertEqual(event.initiating_request_id, 99)
            deadline = threading.Event()
            for _ in range(100):
                if received:
                    break
                deadline.wait(0.01)
            self.assertEqual([item.sequence for item in received], [1])

    def test_sync_event_iterator_wakes_when_client_closes(self) -> None:
        client = odon.connect(self.host, self.port)
        iterator = client.events.iter()
        finished = threading.Event()

        def consume() -> None:
            list(iterator)
            finished.set()

        thread = threading.Thread(target=consume)
        thread.start()
        client.close()
        thread.join(timeout=1)
        self.assertTrue(finished.is_set())

    def test_sync_task_wait_is_event_driven_and_race_safe(self) -> None:
        with odon.connect(self.host, self.port) as client:
            task = client.tasks.start("open_project", {"path": "test.odon"})
            self.assertEqual(task.wait(timeout=1), {"answer": 42})

    def test_data_layer_and_declarative_ui_resources(self) -> None:
        with odon.connect(self.host, self.port) as client:
            data = client.data.register(
                "file:///tmp/labels.zarr",
                format="ome-zarr",
                coordinate_space=odon.CoordinateSpace(
                    axes=("y", "x"), scale=(0.5, 0.5)
                ),
            )
            layer = client.layers.add(data, name="Cellpose", kind="labels")
            self.assertEqual(layer.snapshot["data_resource_id"], data.resource_id)

            extension = client.ui.register_extension(
                id="org.example.cellpose", name="Cellpose", version="0.1"
            )
            contribution = extension.register(
                odon.ui.Panel(
                    "cellpose",
                    title="Cellpose",
                    children=[
                        odon.ui.Slider(
                            "diameter",
                            "Diameter",
                            minimum=1,
                            maximum=100,
                            value=30,
                            event_policy=odon.ui.Debounce(milliseconds=100),
                        ),
                        odon.ui.Button(
                            "run", "Run", action=odon.ui.emit("run-segmentation")
                        ),
                    ],
                )
            )
            self.assertEqual(contribution.contribution_id, "contribution:test")

    def test_discovery_selects_authenticated_instance(self) -> None:
        self.server.expected_token = "manifest-secret"
        try:
            with tempfile.TemporaryDirectory() as directory:
                manifest = {
                    "instance_id": "discovered-instance",
                    "pid": 123,
                    "endpoint": f"tcp://{self.host}:{self.port}",
                    "token": "manifest-secret",
                    "app_version": "test",
                    "protocol_versions": [1],
                    "started_at_unix_ms": 123456,
                }
                Path(directory, "instance-discovered-instance.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                with mock.patch.dict("os.environ", {"ODON_RUNTIME_DIR": directory}):
                    instances = odon.list_instances()
                    self.assertEqual([item.instance_id for item in instances], ["discovered-instance"])
                    with odon.connect() as client:
                        self.assertEqual(client.hello.instance_id, "test-instance")
        finally:
            self.server.expected_token = None


class CoordinateSpaceTests(unittest.TestCase):
    def test_coordinate_space_validates_and_roundtrips_affine_coordinates(self) -> None:
        space = odon.CoordinateSpace(
            axes=("y", "x"), scale=(0.5, 2.0), translation=(10.0, -4.0)
        )
        world = space.pixel_to_world((6.0, 3.0))
        self.assertEqual(world, (13.0, 2.0))
        self.assertEqual(space.world_to_pixel(world), (6.0, 3.0))
        with self.assertRaises(ValueError):
            odon.CoordinateSpace(axes=("x", "x"))
        with self.assertRaises(ValueError):
            odon.CoordinateSpace(axes=("y", "x"), scale=(1.0,))


class AsyncClientTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = _Server(("127.0.0.1", 0), _ControlHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.host, cls.port = cls.server.server_address

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    async def test_async_context_and_calls(self) -> None:
        async with odon.connect_async(self.host, self.port) as client:
            self.assertIs(client.viewer.viewport_links, client.viewport_links)
            camera, changed = await asyncio.gather(
                client.viewer.get_camera(),
                client.viewer.set_camera(center=(1, 2), zoom=3, if_revision=23),
            )
            self.assertEqual(camera["camera"]["zoom_screen_per_lvl0_px"], 0.5)
            self.assertEqual(changed["camera"]["center_world_lvl0"], [1.0, 2.0])
            self.assertEqual(changed["camera"]["if_revision"], 23)

    async def test_connector_is_awaitable(self) -> None:
        client = await odon.connect_async(self.host, self.port)
        try:
            self.assertEqual(client.hello.app_name, "odon")
        finally:
            await client.close()

    async def test_connect_async_forwards_custom_client_identity(self) -> None:
        self.server.expected_client = {
            "name": "odon-async-demo",
            "version": "demo-2",
        }
        try:
            async with odon.connect_async(
                self.host,
                self.port,
                client_name="odon-async-demo",
                client_version="demo-2",
            ) as client:
                self.assertEqual(client.hello.instance_id, "test-instance")
        finally:
            self.server.expected_client = None

    async def test_async_close_finishes_cleanup_after_server_disconnects(self) -> None:
        client = await odon.connect_async(self.host, self.port)
        with self.assertRaises(odon.ConnectionClosedError):
            await client.call("disconnect", timeout=1)
        await client.close()
        self.assertTrue(client._cleanup_complete)

    async def test_async_events_are_server_pushed(self) -> None:
        async with odon.connect_async(self.host, self.port) as client:
            await client.events.subscribe("viewer.*")
            event = await client.events.next(timeout=1)
            self.assertEqual(event.name, "viewer.camera.changed")
            self.assertEqual(event.data["zoom"], 2.0)

    async def test_async_event_iterator_wakes_when_client_closes(self) -> None:
        client = await odon.connect_async(self.host, self.port)
        iterator = client.events.iter()
        pending = asyncio.create_task(anext(iterator))
        await asyncio.sleep(0)
        await client.close()
        with self.assertRaises(StopAsyncIteration):
            await asyncio.wait_for(pending, timeout=1)

    async def test_async_task_is_directly_awaitable(self) -> None:
        async with odon.connect_async(self.host, self.port) as client:
            task = await client.tasks.start("open_project", {"path": "test.odon"})
            self.assertEqual(await task, {"answer": 42})
