from __future__ import annotations

import unittest
from typing import Any, Mapping

from odon.async_resources import AsyncNativeLayers
from odon.resources import NativeLayers


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


class NativeLayerResourceTests(unittest.TestCase):
    def test_native_layer_wrappers(self) -> None:
        client = RecordingClient()
        layers = NativeLayers(client)  # type: ignore[arg-type]

        layers.list()
        layers.get("channel:0")
        layers.set_active("channel:1", if_revision=2)
        layers.set_visibility("channel:1", False)
        layers.set_order("channels", ["channel:1", "channel:0"])
        layers.set_offset("channel:1", [4, -2])
        layers.reset_offset("channel:1")

        self.assertEqual(client.calls[0], ("viewer.native_layers.list", {}))
        self.assertEqual(
            client.calls[2],
            (
                "viewer.native_layers.set_active",
                {"layer_id": "channel:1", "if_revision": 2},
            ),
        )
        self.assertEqual(
            client.calls[4][1],
            {"stack": "channels", "layers": ["channel:1", "channel:0"]},
        )
        self.assertEqual(client.calls[5][1]["offset_world"], [4.0, -2.0])


class AsyncNativeLayerResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_native_layer_wrappers(self) -> None:
        client = AsyncRecordingClient()
        layers = AsyncNativeLayers(client)  # type: ignore[arg-type]

        await layers.list()
        await layers.set_visibility("channel:0", True, if_revision=8)
        await layers.set_offset("channel:0", [1, 2])

        self.assertEqual(client.calls[0], ("viewer.native_layers.list", {}))
        self.assertEqual(client.calls[1][1]["if_revision"], 8)
        self.assertEqual(client.calls[2][1]["offset_world"], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
