from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any, Mapping

from odon.async_resources import AsyncApplication, AsyncDatasets, AsyncS3Datasets
from odon.resources import Application, Datasets, S3Datasets


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.tasks = RecordingTasks(self.calls)
        self.application = Application(self)  # type: ignore[arg-type]

    def call(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        recorded = (method, dict(params or {}))
        self.calls.append(recorded)
        return recorded


class AsyncRecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.tasks = AsyncRecordingTasks(self.calls)
        self.application = AsyncApplication(self)  # type: ignore[arg-type]

    async def call(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        recorded = (method, dict(params or {}))
        self.calls.append(recorded)
        return recorded


class RecordingTasks:
    def __init__(self, calls: list[tuple[str, dict[str, Any]]]) -> None:
        self.calls = calls

    def start(
        self, method: str, params: Mapping[str, Any], *, label: str | None = None
    ) -> Any:
        recorded = (method, dict(params))
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


class DatasetResourceTests(unittest.TestCase):
    def test_inspect_forwards_path(self) -> None:
        client = RecordingClient()
        datasets = Datasets(client)  # type: ignore[arg-type]
        datasets.inspect(Path("sample.zarr"))
        self.assertEqual(client.calls, [("datasets.inspect", {"path": "sample.zarr"})])

    def test_typed_dataset_open_wrappers(self) -> None:
        client = RecordingClient()
        datasets = Datasets(client)  # type: ignore[arg-type]
        datasets.open_spatialdata(
            "spatial.zarr",
            image="images/full",
            labels="cells",
            shapes=["cell_boundaries"],
            points="transcripts",
            points_max=500,
            if_revision=4,
        )
        datasets.open_xenium(
            "experiment", imagery="tiff", load_transcripts=False
        )
        self.assertEqual(client.calls[0][0], "datasets.open_spatialdata")
        self.assertEqual(client.calls[0][1]["shapes"], ["cell_boundaries"])
        self.assertEqual(client.calls[0][1]["if_revision"], 4)
        self.assertEqual(
            client.calls[1],
            (
                "datasets.open_xenium",
                {
                    "path": "experiment",
                    "imagery": "tiff",
                    "load_cells": True,
                    "load_transcripts": False,
                },
            ),
        )

    def test_typed_tiff_plane_wrapper(self) -> None:
        client = RecordingClient()
        datasets = Datasets(client)  # type: ignore[arg-type]
        datasets.open_tiff("stack.ome.tif", z=3, t=2, if_revision=8)
        self.assertEqual(
            client.calls,
            [
                (
                    "datasets.open_tiff",
                    {"path": "stack.ome.tif", "z": 3, "t": 2, "if_revision": 8},
                )
            ],
        )

    def test_remote_dataset_wrappers(self) -> None:
        client = RecordingClient()
        datasets = Datasets(client)  # type: ignore[arg-type]
        s3 = S3Datasets(client)  # type: ignore[arg-type]
        datasets.open_http("https://example.test/image.ome.zarr")
        s3.configure_session(
            endpoint="https://s3.example.test",
            bucket="images",
            access_key="test-key",
            secret_key="test-secret",
        )
        s3.get_session()
        s3.list("study")
        s3.open("study/roi.ome.zarr")
        s3.clear_session()
        self.assertEqual(client.calls[0][0], "datasets.open_http")
        self.assertEqual(client.calls[1][0], "datasets.s3.configure_session")
        self.assertEqual(client.calls[2], ("datasets.s3.get_session", {}))
        self.assertEqual(client.calls[3], ("datasets.s3.list", {"prefix": "study"}))
        self.assertEqual(client.calls[4][0], "datasets.open_s3")
        self.assertEqual(client.calls[5], ("datasets.s3.clear_session", {}))


class AsyncDatasetResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_inspect_forwards_path(self) -> None:
        client = AsyncRecordingClient()
        datasets = AsyncDatasets(client)  # type: ignore[arg-type]
        await datasets.inspect("sample.tif")
        self.assertEqual(client.calls, [("datasets.inspect", {"path": "sample.tif"})])

    async def test_async_typed_dataset_open_wrappers(self) -> None:
        client = AsyncRecordingClient()
        datasets = AsyncDatasets(client)  # type: ignore[arg-type]
        await datasets.open_spatialdata("spatial.zarr", image="image")
        await datasets.open_xenium("experiment", load_cells=False)
        self.assertEqual(client.calls[0][0], "datasets.open_spatialdata")
        self.assertEqual(client.calls[1][1]["load_cells"], False)

    async def test_async_remote_dataset_wrappers(self) -> None:
        client = AsyncRecordingClient()
        datasets = AsyncDatasets(client)  # type: ignore[arg-type]
        s3 = AsyncS3Datasets(client)  # type: ignore[arg-type]
        await datasets.open_http("https://example.test/image.ome.zarr")
        await s3.configure_session(
            endpoint="s3.example.test",
            bucket="images",
            access_key="key",
            secret_key="secret",
        )
        await s3.list()
        await s3.open("roi.zarr")
        self.assertEqual([call[0] for call in client.calls], [
            "datasets.open_http",
            "datasets.s3.configure_session",
            "datasets.s3.list",
            "datasets.open_s3",
        ])


if __name__ == "__main__":
    unittest.main()
