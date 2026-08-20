from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any, Mapping

from odon.async_resources import AsyncProjectRois
from odon.resources import ProjectRois, Projects


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


class ProjectRoiResourceTests(unittest.TestCase):
    def test_project_and_roi_wrappers(self) -> None:
        client = RecordingClient()
        projects = Projects(client)  # type: ignore[arg-type]
        rois = ProjectRois(client)  # type: ignore[arg-type]

        projects.get()
        projects.create(default_dataset="image", if_revision=1)
        projects.save_as(Path("study.project.json"))
        projects.update_metadata(default_threshold_marker="DAPI")
        rois.add(
            "ROI-A",
            Path("a.ome.zarr"),
            display_name="A",
            metadata={"cohort": "treated"},
        )
        rois.update("ROI-A", display_name="Alpha")
        rois.select(["ROI-A"], mode="replace")
        rois.next(2, wrap=False)

        self.assertEqual(client.calls[0], ("project.get", {}))
        self.assertEqual(
            client.calls[1],
            ("project.create", {"default_dataset": "image", "if_revision": 1}),
        )
        self.assertEqual(client.calls[2][1]["path"], "study.project.json")
        self.assertEqual(client.calls[4][1]["metadata"], {"cohort": "treated"})
        self.assertEqual(
            client.calls[5],
            (
                "project.rois.update",
                {"target_id": "ROI-A", "changes": {"display_name": "Alpha"}},
            ),
        )


class AsyncProjectRoiResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_roi_wrappers(self) -> None:
        client = AsyncRecordingClient()
        rois = AsyncProjectRois(client)  # type: ignore[arg-type]

        await rois.get("ROI-A")
        await rois.focus("ROI-A", if_revision=3)
        await rois.previous(wrap=True)

        self.assertEqual(client.calls[0], ("project.rois.get", {"id": "ROI-A"}))
        self.assertEqual(client.calls[1][1]["if_revision"], 3)
        self.assertEqual(client.calls[2][0], "project.rois.previous")


if __name__ == "__main__":
    unittest.main()
