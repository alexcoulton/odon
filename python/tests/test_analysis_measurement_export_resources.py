from __future__ import annotations

import unittest
from typing import Any, Mapping

from odon.async_resources import AsyncAnalysis, AsyncMeasurements, AsyncObjectExports
from odon.resources import Analysis, Measurements, ObjectExports


class Tasks:
    def __init__(self, calls: list[tuple[str, dict[str, Any]]]) -> None:
        self.calls = calls

    def start(self, method: str, params: Mapping[str, Any] | None = None, *, label: str | None = None) -> Any:
        call = (method, {**dict(params or {}), "_label": label})
        self.calls.append(call)
        return call


class Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.tasks = Tasks(self.calls)

    def call(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        call = (method, dict(params or {}))
        self.calls.append(call)
        return call


class AsyncTasks(Tasks):
    async def start(self, method: str, params: Mapping[str, Any] | None = None, *, label: str | None = None) -> Any:
        return super().start(method, params, label=label)


class AsyncClient(Client):
    def __init__(self) -> None:
        super().__init__()
        self.tasks = AsyncTasks(self.calls)

    async def call(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        call = (method, dict(params or {}))
        self.calls.append(call)
        return call


class ResourceTests(unittest.TestCase):
    def test_analysis_resource_wrappers(self) -> None:
        client = Client()
        analysis = Analysis(client)  # type: ignore[arg-type]
        analysis.get()
        analysis.histogram("DAPI", bins=64, transform="arcsinh")
        analysis.suggest_thresholds("DAPI", method="kmeans", count=4)
        analysis.set({"threshold_elements": []}, if_revision=3)
        analysis.warmup()
        analysis.import_preset("calls.json")
        analysis.export_preset("calls-out.json", overwrite=True)
        self.assertEqual(client.calls[1][1]["bins"], 64)
        self.assertEqual(client.calls[2][1]["method"], "kmeans")
        self.assertEqual(client.calls[3][1]["if_revision"], 3)
        self.assertEqual(client.calls[4][0], "viewer.analysis.warmup.start")

    def test_measurement_resource_wrappers(self) -> None:
        client = Client()
        measurements = Measurements(client)  # type: ignore[arg-type]
        measurements.configure(metric="median", level=2, filtered_only=True)
        measurements.start(metric="mean", prefix="mean_")
        measurements.list_generated_properties()
        measurements.cancel(if_revision=7)
        self.assertEqual(client.calls[0][1]["filtered_only"], True)
        self.assertEqual(client.calls[1][0], "viewer.measurements.start")
        self.assertEqual(client.calls[3][1]["if_revision"], 7)

    def test_export_resource_wrappers(self) -> None:
        client = Client()
        exports = ObjectExports(client)  # type: ignore[arg-type]
        exports.list_columns()
        exports.export_csv("cells.csv", scope="selected", overwrite=True)
        exports.export_geoparquet("cells.parquet", columns=["id"])
        self.assertEqual(client.calls[1][0], "exports.objects.export_csv")
        self.assertEqual(client.calls[1][1]["scope"], "selected")
        self.assertEqual(client.calls[2][1]["columns"], ["id"])


class AsyncResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_wrappers(self) -> None:
        client = AsyncClient()
        analysis = AsyncAnalysis(client)  # type: ignore[arg-type]
        measurements = AsyncMeasurements(client)  # type: ignore[arg-type]
        exports = AsyncObjectExports(client)  # type: ignore[arg-type]
        await analysis.get_warmup()
        await measurements.start(metric="median")
        await exports.export_csv("cells.csv")
        self.assertEqual(client.calls[0][0], "viewer.analysis.warmup.get")
        self.assertEqual(client.calls[1][0], "viewer.measurements.start")
        self.assertEqual(client.calls[2][0], "exports.objects.export_csv")


if __name__ == "__main__":
    unittest.main()
