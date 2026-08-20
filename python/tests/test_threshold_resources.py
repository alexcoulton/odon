from __future__ import annotations

import unittest
from typing import Any, Mapping

from odon.async_resources import AsyncThresholds
from odon.resources import Thresholds


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
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        label: str | None = None,
    ) -> Any:
        recorded = (method, {**dict(params or {}), "_label": label})
        self.calls.append(recorded)
        return recorded


class AsyncRecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.tasks = AsyncRecordingTasks(self.calls)

    async def call(
        self, method: str, params: Mapping[str, Any] | None = None
    ) -> Any:
        recorded = (method, dict(params or {}))
        self.calls.append(recorded)
        return recorded


class ThresholdResourceTests(unittest.TestCase):
    def test_threshold_preview_workflow(self) -> None:
        client = RecordingClient()
        thresholds = Thresholds(client)  # type: ignore[arg-type]

        thresholds.list_levels()
        thresholds.get_preview()
        thresholds.configure(threshold=1200, min_component_pixels=24)
        thresholds.start_preview(
            scope="entire_image", level=2, channel="DAPI", threshold=1400
        )
        thresholds.refresh_preview()
        thresholds.apply_preview(if_revision=4)
        thresholds.cancel_preview()

        self.assertEqual(client.calls[0], ("viewer.thresholds.levels.list", {}))
        self.assertEqual(client.calls[2][1]["min_component_pixels"], 24)
        self.assertEqual(client.calls[3][0], "viewer.thresholds.preview.start")
        self.assertEqual(client.calls[3][1]["channel"], "DAPI")
        self.assertEqual(client.calls[5][1]["if_revision"], 4)
        self.assertEqual(client.calls[6][0], "viewer.thresholds.preview.cancel")


class AsyncThresholdResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_threshold_preview_workflow(self) -> None:
        client = AsyncRecordingClient()
        thresholds = AsyncThresholds(client)  # type: ignore[arg-type]

        await thresholds.start_preview(scope="visible", channel=1)
        await thresholds.configure(threshold=900)
        await thresholds.apply_preview()
        await thresholds.cancel_preview(if_revision=5)

        self.assertEqual(client.calls[0][0], "viewer.thresholds.preview.start")
        self.assertEqual(client.calls[0][1]["channel"], 1)
        self.assertEqual(client.calls[2][0], "viewer.thresholds.preview.apply")
        self.assertEqual(client.calls[3][1]["if_revision"], 5)


if __name__ == "__main__":
    unittest.main()
