from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any, Mapping

from odon.async_resources import (
    AsyncProjectDiscovery,
    AsyncProjectObjects,
    AsyncProjectSamplesheets,
)
from odon.resources import ProjectDiscovery, ProjectObjects, ProjectSamplesheets


class RecordingTasks:
    def __init__(self, calls: list[tuple[str, dict[str, Any]]]) -> None:
        self.calls = calls

    def start(
        self, method: str, params: Mapping[str, Any], *, label: str | None = None
    ) -> Any:
        recorded = (method, dict(params))
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


class AsyncRecordingTasks:
    def __init__(self, calls: list[tuple[str, dict[str, Any]]]) -> None:
        self.calls = calls

    async def start(
        self, method: str, params: Mapping[str, Any], *, label: str | None = None
    ) -> Any:
        recorded = (method, dict(params))
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


class ProjectSamplesheetTests(unittest.TestCase):
    def test_project_samplesheet_wrappers(self) -> None:
        client = RecordingClient()
        sheets = ProjectSamplesheets(client)  # type: ignore[arg-type]
        discovery = ProjectDiscovery(client)  # type: ignore[arg-type]

        sheets.inspect(Path("samples.csv"), offset=5, limit=25)
        sheets.import_("samples.csv", if_revision=2)
        sheets.export("out.csv", overwrite=True)
        discovery.add_root("datasets", if_revision=3)

        self.assertEqual(
            client.calls,
            [
                (
                    "project.samplesheets.inspect",
                    {"path": "samples.csv", "offset": 5, "limit": 25},
                ),
                ("project.samplesheets.import", {"path": "samples.csv", "if_revision": 2}),
                ("project.samplesheets.export", {"path": "out.csv", "overwrite": True}),
                ("project.discovery.add_root", {"path": "datasets", "if_revision": 3}),
            ],
        )

    def test_project_object_preload_wrappers(self) -> None:
        client = RecordingClient()
        objects = ProjectObjects(client)  # type: ignore[arg-type]
        objects.get_preload()
        objects.list_preload_sources(offset=5, limit=10)
        objects.preload(mode="centroid_points", lazy_properties=False, if_revision=7)
        objects.clear_preload()
        self.assertEqual(client.calls[0], ("project.objects.preload.get", {}))
        self.assertEqual(client.calls[1][1], {"offset": 5, "limit": 10})
        self.assertEqual(
            client.calls[2],
            (
                "project.objects.preload.start",
                {
                    "mode": "centroid_points",
                    "lazy_properties": False,
                    "if_revision": 7,
                },
            ),
        )


class AsyncProjectSamplesheetTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_project_samplesheet_wrappers(self) -> None:
        client = AsyncRecordingClient()
        sheets = AsyncProjectSamplesheets(client)  # type: ignore[arg-type]
        discovery = AsyncProjectDiscovery(client)  # type: ignore[arg-type]

        await sheets.inspect("samples.csv")
        await sheets.import_("samples.csv")
        await sheets.export("out.csv")
        await discovery.add_root("datasets")

        self.assertEqual(client.calls[0][0], "project.samplesheets.inspect")
        self.assertEqual(client.calls[1][0], "project.samplesheets.import")
        self.assertEqual(client.calls[2][1]["overwrite"], False)
        self.assertEqual(client.calls[3][0], "project.discovery.add_root")

    async def test_async_project_object_preload_wrappers(self) -> None:
        client = AsyncRecordingClient()
        objects = AsyncProjectObjects(client)  # type: ignore[arg-type]
        await objects.get_preload()
        await objects.preload()
        await objects.clear_preload(if_revision=9)
        self.assertEqual(
            [call[0] for call in client.calls],
            [
                "project.objects.preload.get",
                "project.objects.preload.start",
                "project.objects.preload.clear",
            ],
        )


if __name__ == "__main__":
    unittest.main()
