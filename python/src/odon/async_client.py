"""Asyncio Odon control client using response-driven futures, not polling."""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
from typing import Any, Mapping, Sequence

from .async_resources import (
    AsyncAnalysis,
    AsyncAnnotations,
    AsyncApplication,
    AsyncChannels,
    AsyncDatasets,
    AsyncDeepLinks,
    AsyncLabels,
    AsyncMasks,
    AsyncMemory,
    AsyncMeasurements,
    AsyncMosaic,
    AsyncNativeLayers,
    AsyncObjects,
    AsyncObjectExports,
    AsyncPlanes,
    AsyncProjectDiscovery,
    AsyncProjectObjects,
    AsyncProjectRois,
    AsyncProjectSamplesheets,
    AsyncProjects,
    AsyncProjectViews,
    AsyncScreenshots,
    AsyncS3Datasets,
    AsyncThresholds,
    AsyncViewer,
    AsyncViewportLinks,
    AsyncViewports,
    AsyncViewportWorkspace,
)
from .errors import (
    ConnectionClosedError,
    ProtocolError,
    RequestTimeoutError,
    remote_error_from_message,
)
from .models import Hello
from .async_events import AsyncEvents
from .async_tasks import AsyncTasks
from .async_data import AsyncDataResources
from .async_layers import AsyncLayers
from .async_ui import AsyncUi
from .discovery import Instance, select_instance

logger = logging.getLogger("odon.async_client")


class AsyncClient:
    """A persistent asyncio connection to one running Odon instance."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        timeout: float,
    ) -> None:
        self.timeout = timeout
        self._reader = reader
        self._writer = writer
        self._ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._send_lock = asyncio.Lock()
        self._closed = False
        self._cleanup_complete = False
        self._max_inline_payload_bytes = 1_048_576
        self._launched_process: Any = None
        self.events = AsyncEvents(self)
        self.tasks = AsyncTasks(self)
        self.data = AsyncDataResources(self)
        self.ui = AsyncUi(self)
        self._reader_task = asyncio.create_task(
            self._read_messages(), name="odon-client-reader"
        )

        self.application = AsyncApplication(self)
        self.datasets = AsyncDatasets(self)
        self.s3_datasets = AsyncS3Datasets(self)
        self.datasets.s3 = self.s3_datasets
        self.deep_links = AsyncDeepLinks(self)
        self.viewer = AsyncViewer(self)
        self.viewports = AsyncViewports(self)
        self.viewport_workspace = AsyncViewportWorkspace(self)
        self.viewport_links = AsyncViewportLinks(self)
        self.viewer.viewports = self.viewports
        self.viewer.workspace = self.viewport_workspace
        self.viewer.viewport_links = self.viewport_links
        self.channels = AsyncChannels(self)
        self.planes = AsyncPlanes(self)
        self.native_layers = AsyncNativeLayers(self)
        self.layers = AsyncLayers(self)
        self.viewer.channels = self.channels
        self.viewer.planes = self.planes
        self.viewer.native_layers = self.native_layers
        self.viewer.layers = self.layers
        self.projects = AsyncProjects(self)
        self.project_samplesheets = AsyncProjectSamplesheets(self)
        self.projects.samplesheets = self.project_samplesheets
        self.project_discovery = AsyncProjectDiscovery(self)
        self.projects.discovery = self.project_discovery
        self.project_objects = AsyncProjectObjects(self)
        self.projects.objects = self.project_objects
        self.project_rois = AsyncProjectRois(self)
        self.projects.rois = self.project_rois
        self.project_views = AsyncProjectViews(self)
        self.projects.views = self.project_views
        self.screenshots = AsyncScreenshots(self)
        self.labels = AsyncLabels(self)
        self.memory = AsyncMemory(self)
        self.objects = AsyncObjects(self)
        self.annotations = AsyncAnnotations(self)
        self.masks = AsyncMasks(self)
        self.thresholds = AsyncThresholds(self)
        self.analysis = AsyncAnalysis(self)
        self.measurements = AsyncMeasurements(self)
        self.object_exports = AsyncObjectExports(self)
        self.mosaic = AsyncMosaic(self)
        self.viewer.objects = self.objects
        self.viewer.annotations = self.annotations
        self.viewer.masks = self.masks
        self.viewer.thresholds = self.thresholds
        self.viewer.analysis = self.analysis
        self.viewer.measurements = self.measurements
        self.viewer.memory = self.memory
        self.viewer.labels = self.labels
        self.objects.exports = self.object_exports
        self.hello: Hello

    @classmethod
    async def connect(
        cls,
        host: str | None = None,
        port: int | None = None,
        *,
        token: str | None = None,
        instance: Instance | str | None = None,
        timeout: float = 10.0,
        client_name: str = "odon-client",
        client_version: str = "0.1.0",
    ) -> "AsyncClient":
        if host is None and port is None:
            selected = select_instance(instance)
            host, port, token = selected.host, selected.port, selected.token
        elif host is None or port is None:
            raise ValueError("host and port must be provided together")
        elif instance is not None:
            raise ValueError("instance cannot be combined with an explicit host and port")
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
        client = cls(reader, writer, timeout=timeout)
        try:
            result = await client.call(
                "system.hello",
                {
                    "client": {"name": client_name, "version": client_version},
                    "protocol_versions": [1],
                    **({"token": token} if token is not None else {}),
                },
            )
            if not isinstance(result, Mapping):
                raise ProtocolError("system.hello returned a non-object result")
            client.hello = Hello.from_result(result)
            client._max_inline_payload_bytes = client.hello.max_inline_payload_bytes
            logger.debug(
                "connected to Odon instance=%s session=%s protocol=%s",
                client.hello.instance_id,
                client.hello.session_id,
                client.hello.protocol_version,
            )
            return client
        except BaseException:
            await client.close()
            raise

    async def call(
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
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": dict(params or {}),
        }
        try:
            encoded = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
            if len(encoded) > self._max_inline_payload_bytes:
                self._pending.pop(request_id, None)
                raise ValueError(
                    f"request exceeds Odon's {self._max_inline_payload_bytes}-byte inline payload limit"
                )
            async with self._send_lock:
                self._writer.write(encoded)
                await self._writer.drain()
        except (ConnectionError, OSError) as error:
            self._pending.pop(request_id, None)
            raise ConnectionClosedError(f"failed to send request: {error}") from error

        try:
            response = await asyncio.wait_for(
                future, timeout=self.timeout if timeout is None else timeout
            )
        except TimeoutError as error:
            raise RequestTimeoutError(
                f"timed out waiting for Odon method {method!r}; the operation may still be running"
            ) from error
        finally:
            self._pending.pop(request_id, None)
        if "error" in response:
            raise remote_error_from_message(response, method=method, request_id=request_id)
        if "result" not in response:
            raise ProtocolError("JSON-RPC response has neither result nor error")
        return response["result"]

    async def close(self) -> None:
        if self._cleanup_complete:
            return
        self._closed = True
        self.events._close()
        await self.data._close()
        self._writer.close()
        try:
            await self._writer.wait_closed()
        except (ConnectionError, OSError):
            pass
        if asyncio.current_task() is not self._reader_task:
            self._reader_task.cancel()
            await asyncio.gather(self._reader_task, return_exceptions=True)
        self._fail_pending(ConnectionClosedError("Odon control connection closed"))
        self._cleanup_complete = True

    async def batch(
        self,
        operations: Sequence[tuple[str, Mapping[str, Any] | None]],
        *,
        atomic: bool = False,
    ) -> Any:
        return await self.call(
            "system.batch",
            {
                "atomic": atomic,
                "operations": [
                    {"method": method, "params": dict(params or {})}
                    for method, params in operations
                ],
            },
        )

    async def __aenter__(self) -> "AsyncClient":
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def launched_process(self) -> Any:
        return self._launched_process

    async def _read_messages(self) -> None:
        try:
            while line := await self._reader.readline():
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ProtocolError(f"invalid JSON from Odon: {error}") from error
                if not isinstance(message, Mapping):
                    raise ProtocolError("Odon sent a non-object message")
                request_id = message.get("id")
                if message.get("method") == "events.event":
                    params = message.get("params")
                    if isinstance(params, Mapping):
                        self.events._receive(params)
                    continue
                if not isinstance(request_id, int):
                    continue  # Reserved for future event notifications.
                future = self._pending.get(request_id)
                if future is not None and not future.done():
                    future.set_result(message)
            if not self._closed:
                raise ConnectionClosedError("Odon closed the control connection")
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            if not self._closed:
                self._closed = True
                self._fail_pending(error)
                await self.data._close()
            self.events._close()

    def _fail_pending(self, error: BaseException) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)


class _AsyncConnector:
    def __init__(
        self,
        host: str | None,
        port: int | None,
        timeout: float,
        token: str | None,
        instance: Instance | str | None,
        client_name: str,
        client_version: str,
    ) -> None:
        self._host = host
        self._port = port
        self._timeout = timeout
        self._token = token
        self._instance = instance
        self._client_name = client_name
        self._client_version = client_version
        self._client: AsyncClient | None = None

    def __await__(self):
        return self._connect().__await__()

    async def _connect(self) -> AsyncClient:
        if self._client is None:
            self._client = await AsyncClient.connect(
                self._host,
                self._port,
                timeout=self._timeout,
                token=self._token,
                instance=self._instance,
                client_name=self._client_name,
                client_version=self._client_version,
            )
        return self._client

    async def __aenter__(self) -> AsyncClient:
        return await self._connect()

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._client is not None:
            await self._client.close()


def connect_async(
    host: str | None = None,
    port: int | None = None,
    *,
    timeout: float = 10.0,
    token: str | None = None,
    instance: Instance | str | None = None,
    client_name: str = "odon-client",
    client_version: str = "0.1.0",
) -> _AsyncConnector:
    """Return an awaitable and async-context-manager connection factory."""

    return _AsyncConnector(
        host,
        port,
        timeout,
        token,
        instance,
        client_name,
        client_version,
    )
