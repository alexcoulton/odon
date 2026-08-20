from __future__ import annotations

import unittest
from typing import Any, Mapping

from odon.async_resources import AsyncLabels, AsyncPlanes, AsyncViewer
from odon.resources import Labels, Planes, Viewer


class Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        call = (method, dict(params or {}))
        self.calls.append(call)
        return call


class AsyncClient(Client):
    async def call(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        call = (method, dict(params or {}))
        self.calls.append(call)
        return call


class ResourceTests(unittest.TestCase):
    def test_label_wrappers(self) -> None:
        client = Client()
        labels = Labels(client)  # type: ignore[arg-type]
        labels.list()
        labels.get()
        labels.load("cells", if_revision=2)
        labels.set_visibility(False)
        labels.set_visibility(True, name="nuclei")
        labels.unload()
        self.assertEqual(client.calls[2][1], {"name": "cells", "if_revision": 2})
        self.assertEqual(client.calls[4][1], {"visible": True, "name": "nuclei"})

    def test_rendering_and_plane_availability_wrappers(self) -> None:
        client = Client()
        viewer = Viewer(client)  # type: ignore[arg-type]
        planes = Planes(client)  # type: ignore[arg-type]
        viewer.get_rendering_state()
        planes.get_operation_availability()
        self.assertEqual(client.calls[0][0], "viewer.rendering.get_state")
        self.assertEqual(client.calls[1][0], "viewer.planes.operation_availability")


class AsyncResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_wrappers(self) -> None:
        client = AsyncClient()
        labels = AsyncLabels(client)  # type: ignore[arg-type]
        viewer = AsyncViewer(client)  # type: ignore[arg-type]
        planes = AsyncPlanes(client)  # type: ignore[arg-type]
        await labels.load()
        await viewer.get_rendering_state()
        await planes.get_operation_availability()
        self.assertEqual(client.calls[0][0], "viewer.labels.load")
        self.assertEqual(client.calls[2][0], "viewer.planes.operation_availability")


if __name__ == "__main__":
    unittest.main()
