from __future__ import annotations

import unittest
from typing import Any, Mapping

from odon.async_resources import AsyncDeepLinks
from odon.resources import DeepLinks


class RecordingTasks:
    def __init__(self, calls: list[tuple[str, dict[str, Any]]]) -> None:
        self.calls = calls

    def start(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        label: str | None = None,
    ) -> Any:
        recorded = (method, {**dict(params or {}), "_label": label})
        self.calls.append(recorded)
        return recorded


class AsyncRecordingTasks:
    def __init__(self, calls: list[tuple[str, dict[str, Any]]]) -> None:
        self.calls = calls

    async def start(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        label: str | None = None,
    ) -> Any:
        recorded = (method, {**dict(params or {}), "_label": label})
        self.calls.append(recorded)
        return recorded


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.tasks = RecordingTasks(self.calls)

    def call(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        recorded = (method, dict(params or {}))
        self.calls.append(recorded)
        return recorded


class AsyncRecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.tasks = AsyncRecordingTasks(self.calls)

    async def call(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        recorded = (method, dict(params or {}))
        self.calls.append(recorded)
        return recorded


class DeepLinkResourceTests(unittest.TestCase):
    def test_deep_link_wrappers(self) -> None:
        client = RecordingClient()
        links = DeepLinks(client)  # type: ignore[arg-type]

        links.parse("odon://open?roi=ROI-1")
        links.resolve({"roi": "ROI-1"})
        links.get_filters({"object_query": "area > 10"})
        links.generate({"visible_channels": ["DAPI"]}, roi="ROI-2")
        links.apply("odon://open?channel=CD3", if_revision=7)
        links.apply({"zoom": 0.5})

        self.assertEqual(
            client.calls,
            [
                ("deep_links.parse", {"url": "odon://open?roi=ROI-1"}),
                ("deep_links.resolve", {"request": {"roi": "ROI-1"}}),
                (
                    "deep_links.filters.get",
                    {"request": {"object_query": "area > 10"}},
                ),
                (
                    "deep_links.generate",
                    {
                        "include_project": True,
                        "request": {"visible_channels": ["DAPI"]},
                        "roi": "ROI-2",
                    },
                ),
                (
                    "deep_links.apply",
                    {
                        "url": "odon://open?channel=CD3",
                        "if_revision": 7,
                        "_label": "Apply Odon deep link",
                    },
                ),
                (
                    "deep_links.apply",
                    {
                        "request": {"zoom": 0.5},
                        "_label": "Apply Odon deep link",
                    },
                ),
            ],
        )


class AsyncDeepLinkResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_deep_link_wrappers(self) -> None:
        client = AsyncRecordingClient()
        links = AsyncDeepLinks(client)  # type: ignore[arg-type]

        await links.parse("odon:open")
        await links.resolve("odon://open?roi=ROI-1")
        await links.get_filters("odon://open?object_query=area%20%3E%2010")
        await links.generate(include_project=False)
        await links.apply({"fill_cells": True}, if_revision=3)

        self.assertEqual(client.calls[0], ("deep_links.parse", {"url": "odon:open"}))
        self.assertEqual(
            client.calls[3],
            ("deep_links.generate", {"include_project": False}),
        )
        self.assertEqual(
            client.calls[4],
            (
                "deep_links.apply",
                {
                    "request": {"fill_cells": True},
                    "if_revision": 3,
                    "_label": "Apply Odon deep link",
                },
            ),
        )


if __name__ == "__main__":
    unittest.main()
