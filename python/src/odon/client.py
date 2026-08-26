"""Thread-safe synchronous Odon control client."""

from __future__ import annotations

import itertools
import json
import socket
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .errors import (
    ConnectionClosedError,
    ProtocolError,
    RequestTimeoutError,
    remote_error_from_message,
)
from .models import DEFAULT_REQUESTED_CAPABILITIES, Hello
from .events import Events
from .tasks import Tasks
from .data import DataResources
from .layers import Layers
from .ui import Ui
from .discovery import Instance, select_instance
from .resources import Analysis, Annotations, Application, Channels, Datasets, DeepLinks, Labels, Masks, Measurements, Memory, Mosaic, NativeLayers, ObjectExports, Objects, Planes, ProjectDiscovery, ProjectObjects, ProjectRois, ProjectSamplesheets, Projects, ProjectViews, S3Datasets, Screenshots, Thresholds, Viewer, ViewportLinks, Viewports, ViewportWorkspace

logger = logging.getLogger("odon.client")


@dataclass
class _PendingRequest:
    event: threading.Event = field(default_factory=threading.Event)
    response: Mapping[str, Any] | None = None
    error: BaseException | None = None


class Client:
    """A persistent synchronous connection to one running Odon instance."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        *,
        token: str | None = None,
        instance: Instance | str | None = None,
        timeout: float = 10.0,
        client_name: str = "odon-client",
        client_version: str = "0.1.0",
        requested_capabilities: Sequence[str] | None = None,
    ) -> None:
        if host is None and port is None:
            selected = select_instance(instance)
            host, port, token = selected.host, selected.port, selected.token
        elif host is None or port is None:
            raise ValueError("host and port must be provided together")
        elif instance is not None:
            raise ValueError("instance cannot be combined with an explicit host and port")
        self.host = host
        self.port = port
        self.timeout = timeout
        self._socket = socket.create_connection((host, port), timeout=timeout)
        self._socket.settimeout(None)
        self._reader = self._socket.makefile("r", encoding="utf-8", newline="\n")
        self._ids = itertools.count(1)
        self._send_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: dict[int, _PendingRequest] = {}
        self._closed = False
        self._cleanup_complete = False
        self._max_inline_payload_bytes = 1_048_576
        self._launched_process: Any = None
        self.events = Events(self)
        self.tasks = Tasks(self)
        self.data = DataResources(self)
        self.ui = Ui(self)
        self._reader_thread = threading.Thread(
            target=self._read_messages,
            name="odon-client-reader",
            daemon=True,
        )
        self._reader_thread.start()

        requested = tuple(
            DEFAULT_REQUESTED_CAPABILITIES
            if requested_capabilities is None
            else requested_capabilities
        )
        if any(not isinstance(capability, str) or not capability for capability in requested):
            self.close()
            raise ValueError("requested_capabilities must contain non-empty strings")

        hello_result = self.call(
            "system.hello",
            {
                "client": {"name": client_name, "version": client_version},
                "protocol_versions": [1],
                "requested_capabilities": list(requested),
                **({"token": token} if token is not None else {}),
            },
        )
        if not isinstance(hello_result, Mapping):
            self.close()
            raise ProtocolError("system.hello returned a non-object result")
        self.hello = Hello.from_result(hello_result)
        self._max_inline_payload_bytes = self.hello.max_inline_payload_bytes
        logger.debug(
            "connected to Odon instance=%s session=%s protocol=%s",
            self.hello.instance_id,
            self.hello.session_id,
            self.hello.protocol_version,
        )

        self.application = Application(self)
        self.datasets = Datasets(self)
        self.s3_datasets = S3Datasets(self)
        self.datasets.s3 = self.s3_datasets
        self.deep_links = DeepLinks(self)
        self.viewer = Viewer(self)
        self.viewports = Viewports(self)
        self.viewport_workspace = ViewportWorkspace(self)
        self.viewport_links = ViewportLinks(self)
        self.viewer.viewports = self.viewports
        self.viewer.workspace = self.viewport_workspace
        self.viewer.viewport_links = self.viewport_links
        self.channels = Channels(self)
        self.planes = Planes(self)
        self.native_layers = NativeLayers(self)
        self.layers = Layers(self)
        self.viewer.channels = self.channels
        self.viewer.planes = self.planes
        self.viewer.native_layers = self.native_layers
        self.viewer.layers = self.layers
        self.projects = Projects(self)
        self.project_samplesheets = ProjectSamplesheets(self)
        self.projects.samplesheets = self.project_samplesheets
        self.project_discovery = ProjectDiscovery(self)
        self.projects.discovery = self.project_discovery
        self.project_objects = ProjectObjects(self)
        self.projects.objects = self.project_objects
        self.project_rois = ProjectRois(self)
        self.projects.rois = self.project_rois
        self.project_views = ProjectViews(self)
        self.projects.views = self.project_views
        self.screenshots = Screenshots(self)
        self.labels = Labels(self)
        self.memory = Memory(self)
        self.objects = Objects(self)
        self.annotations = Annotations(self)
        self.masks = Masks(self)
        self.thresholds = Thresholds(self)
        self.analysis = Analysis(self)
        self.measurements = Measurements(self)
        self.object_exports = ObjectExports(self)
        self.mosaic = Mosaic(self)
        self.viewer.objects = self.objects
        self.viewer.annotations = self.annotations
        self.viewer.masks = self.masks
        self.viewer.thresholds = self.thresholds
        self.viewer.analysis = self.analysis
        self.viewer.measurements = self.measurements
        self.viewer.memory = self.memory
        self.viewer.labels = self.labels
        self.objects.exports = self.object_exports

    def call(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        if self._closed:
            raise ConnectionClosedError("Odon control connection is closed")
        request_id = next(self._ids)
        logger.debug("Odon request id=%s method=%s", request_id, method)
        pending = _PendingRequest()
        with self._pending_lock:
            self._pending[request_id] = pending
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": dict(params or {}),
        }
        try:
            encoded = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
            if len(encoded) > self._max_inline_payload_bytes:
                with self._pending_lock:
                    self._pending.pop(request_id, None)
                raise ValueError(
                    f"request exceeds Odon's {self._max_inline_payload_bytes}-byte inline payload limit"
                )
            with self._send_lock:
                self._socket.sendall(encoded)
        except OSError as error:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise ConnectionClosedError(f"failed to send request: {error}") from error

        if not pending.event.wait(self.timeout if timeout is None else timeout):
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise RequestTimeoutError(
                f"timed out waiting for Odon method {method!r}; the operation may still be running"
            )
        if pending.error is not None:
            raise pending.error
        if pending.response is None:
            raise ProtocolError("request completed without a response")
        if "error" in pending.response:
            raise remote_error_from_message(
                pending.response, method=method, request_id=request_id
            )
        if "result" not in pending.response:
            raise ProtocolError("JSON-RPC response has neither result nor error")
        return pending.response["result"]

    def close(self) -> None:
        if self._cleanup_complete:
            return
        self.ui._close()
        self._closed = True
        hello = getattr(self, "hello", None)
        logger.debug("closing Odon session=%s", getattr(hello, "session_id", "unnegotiated"))
        self.data._close()
        self.events._close()
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._socket.close()
        self._reader.close()
        self._fail_pending(ConnectionClosedError("Odon control connection closed"))
        self._cleanup_complete = True

    def batch(
        self,
        operations: Sequence[tuple[str, Mapping[str, Any] | None]],
        *,
        atomic: bool = False,
    ) -> Any:
        return self.call(
            "system.batch",
            {
                "atomic": atomic,
                "operations": [
                    {"method": method, "params": dict(params or {})}
                    for method, params in operations
                ],
            },
        )

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def launched_process(self) -> Any:
        """The ``subprocess.Popen`` handle when created by :func:`odon.launch`."""

        return self._launched_process

    def _read_messages(self) -> None:
        try:
            for line in self._reader:
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as error:
                    self._fail_pending(ProtocolError(f"invalid JSON from Odon: {error}"))
                    return
                if not isinstance(message, Mapping):
                    self._fail_pending(ProtocolError("Odon sent a non-object message"))
                    return
                request_id = message.get("id")
                if message.get("method") == "events.event":
                    params = message.get("params")
                    if isinstance(params, Mapping):
                        self.events._receive(params)
                    continue
                if not isinstance(request_id, int):
                    continue  # Reserved for future event notifications.
                with self._pending_lock:
                    pending = self._pending.pop(request_id, None)
                if pending is not None:
                    pending.response = message
                    pending.event.set()
        except (OSError, ValueError) as error:
            if not self._closed:
                self._fail_pending(ConnectionClosedError(f"connection reader failed: {error}"))
        finally:
            if not self._closed:
                self._closed = True
                self._fail_pending(ConnectionClosedError("Odon closed the control connection"))
                self.data._close()
            self.events._close()

    def _fail_pending(self, error: BaseException) -> None:
        with self._pending_lock:
            pending_requests = list(self._pending.values())
            self._pending.clear()
        for pending in pending_requests:
            pending.error = error
            pending.event.set()


def connect(
    host: str | None = None,
    port: int | None = None,
    *,
    timeout: float = 10.0,
    token: str | None = None,
    instance: Instance | str | None = None,
    client_name: str = "odon-client",
    client_version: str = "0.1.0",
    requested_capabilities: Sequence[str] | None = None,
) -> Client:
    """Connect to a running Odon instance."""

    return Client(
        host,
        port,
        timeout=timeout,
        token=token,
        instance=instance,
        client_name=client_name,
        client_version=client_version,
        requested_capabilities=requested_capabilities,
    )
