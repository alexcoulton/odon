from __future__ import annotations

import unittest
from typing import Any, Mapping

from odon.async_resources import AsyncPlanes
from odon.resources import Planes


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        recorded = (method, dict(params or {}))
        self.calls.append(recorded)
        return recorded


class AsyncRecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        recorded = (method, dict(params or {}))
        self.calls.append(recorded)
        return recorded


class PlaneResourceTests(unittest.TestCase):
    def test_sync_plane_wrappers(self) -> None:
        client = RecordingClient()
        planes = Planes(client)  # type: ignore[arg-type]

        self.assertEqual(planes.get()[0], "viewer.planes.get")
        planes.set(mode="xz", slice=4, if_revision=7)
        planes.next(2, wrap=True)
        planes.previous()

        self.assertEqual(
            client.calls[1],
            ("viewer.planes.set", {"mode": "xz", "slice": 4, "if_revision": 7}),
        )
        self.assertEqual(
            client.calls[2], ("viewer.planes.next", {"step": 2, "wrap": True})
        )
        self.assertEqual(
            client.calls[3], ("viewer.planes.previous", {"step": 1, "wrap": False})
        )
        with self.assertRaises(ValueError):
            planes.set()


class AsyncPlaneResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_plane_wrappers(self) -> None:
        client = AsyncRecordingClient()
        planes = AsyncPlanes(client)  # type: ignore[arg-type]

        await planes.get()
        await planes.set(mode="yz", slice=8, if_revision=11)
        await planes.next(5, wrap=True)

        self.assertEqual(client.calls[0], ("viewer.planes.get", {}))
        self.assertEqual(
            client.calls[1],
            ("viewer.planes.set", {"mode": "yz", "slice": 8, "if_revision": 11}),
        )
        self.assertEqual(
            client.calls[2], ("viewer.planes.next", {"step": 5, "wrap": True})
        )


if __name__ == "__main__":
    unittest.main()
