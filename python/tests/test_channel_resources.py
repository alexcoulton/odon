from __future__ import annotations

import unittest
from typing import Any, Mapping

from odon.async_resources import AsyncChannels
from odon.resources import Channels


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


class ChannelResourceTests(unittest.TestCase):
    def test_channel_property_wrappers(self) -> None:
        client = RecordingClient()
        channels = Channels(client)  # type: ignore[arg-type]

        channels.set_color("DAPI", [12, 34, 56], if_revision=2)
        channels.set_note(1, "nuclear")
        channels.get_transform("DAPI")
        channels.set_transform(
            "DAPI", offset_world=[3, 4], scale=[1.2, 0.8], rotation_rad=0.25
        )
        channels.reset_transform(1)
        channels.get_presentation()
        channels.set_presentation(search="nuclear", sort="visible_first")

        self.assertEqual(
            client.calls[0],
            (
                "viewer.channels.set_color",
                {"name": "DAPI", "color_rgb": [12, 34, 56], "if_revision": 2},
            ),
        )
        self.assertEqual(
            client.calls[1],
            ("viewer.channels.set_note", {"index": 1, "note": "nuclear"}),
        )
        self.assertEqual(
            client.calls[3][1],
            {
                "name": "DAPI",
                "offset_world": [3.0, 4.0],
                "scale": [1.2, 0.8],
                "rotation_rad": 0.25,
            },
        )
        with self.assertRaises(ValueError):
            channels.set_transform("DAPI")
        self.assertEqual(client.calls[-1][1], {"search": "nuclear", "sort": "visible_first"})
        with self.assertRaises(ValueError):
            channels.set_presentation()


class AsyncChannelResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_channel_property_wrappers(self) -> None:
        client = AsyncRecordingClient()
        channels = AsyncChannels(client)  # type: ignore[arg-type]

        await channels.set_color(0, [1, 2, 3])
        await channels.set_note("DAPI", "nuclear", if_revision=4)
        await channels.set_transform("DAPI", scale=[2, 2])
        await channels.set_presentation(sort="name_asc")

        self.assertEqual(
            client.calls[0],
            ("viewer.channels.set_color", {"index": 0, "color_rgb": [1, 2, 3]}),
        )
        self.assertEqual(client.calls[1][1]["if_revision"], 4)
        self.assertEqual(client.calls[2][1]["scale"], [2.0, 2.0])
        self.assertEqual(client.calls[3][0], "viewer.channels.presentation.set")


if __name__ == "__main__":
    unittest.main()
