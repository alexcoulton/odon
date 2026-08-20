from __future__ import annotations

import unittest
from typing import Any, Mapping

from odon.async_resources import AsyncProjectViews
from odon.resources import ProjectViews


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


class ProjectViewResourceTests(unittest.TestCase):
    def test_project_view_wrappers(self) -> None:
        client = RecordingClient()
        views = ProjectViews(client)  # type: ignore[arg-type]

        views.list()
        views.get("Review")
        views.create("Review", {"visible_channels": ["DAPI"]}, if_revision=3)
        views.capture("Current")
        views.rename(0, "Overview")
        views.apply("Overview")
        views.delete(0)

        self.assertEqual(client.calls[0], ("project.views.list", {}))
        self.assertEqual(client.calls[1], ("project.views.get", {"name": "Review"}))
        self.assertEqual(
            client.calls[2],
            (
                "project.views.create",
                {
                    "name": "Review",
                    "spec": {"visible_channels": ["DAPI"]},
                    "if_revision": 3,
                },
            ),
        )
        self.assertEqual(
            client.calls[4],
            ("project.views.rename", {"index": 0, "new_name": "Overview"}),
        )


class AsyncProjectViewResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_project_view_wrappers(self) -> None:
        client = AsyncRecordingClient()
        views = AsyncProjectViews(client)  # type: ignore[arg-type]

        await views.create("Review")
        await views.capture("Current", if_revision=4)
        await views.apply(1)

        self.assertEqual(client.calls[0], ("project.views.create", {"name": "Review"}))
        self.assertEqual(client.calls[1][1]["if_revision"], 4)
        self.assertEqual(client.calls[2], ("project.views.apply", {"index": 1}))


if __name__ == "__main__":
    unittest.main()
