from __future__ import annotations

import unittest
from typing import Any, Mapping

from odon.async_resources import AsyncObjects
from odon.resources import Objects


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


class AsyncRecordingTasks(RecordingTasks):
    async def start(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        label: str | None = None,
    ) -> Any:
        return super().start(method, params, label=label)


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


class ObjectResourceTests(unittest.TestCase):
    def test_object_style_wrappers(self) -> None:
        client = RecordingClient()
        objects = Objects(client)  # type: ignore[arg-type]

        objects.get_style(target="objects")
        objects.set_style(opacity=0.4, fill_cells=True, if_revision=5)
        objects.set_legend(
            [{"value": "Tumor", "visible": False, "color_rgb": [255, 0, 0]}]
        )
        objects.get_fast_rendering()
        objects.set_fast_rendering(False, if_revision=6)

        self.assertEqual(client.calls[0], ("viewer.objects.style.get", {"target": "objects"}))
        self.assertEqual(client.calls[1][1]["fill_cells"], True)
        self.assertEqual(client.calls[2][0], "viewer.objects.legend.set")
        self.assertEqual(client.calls[4][1], {"enabled": False, "if_revision": 6})

    def test_object_filter_model_wrappers(self) -> None:
        client = RecordingClient()
        objects = Objects(client)  # type: ignore[arg-type]

        objects.set_filter_model(
            clauses=[{"property": "class", "query": "tumor"}],
            logic="any",
            if_revision=8,
        )
        objects.set_filter_model(mode="query", query="area_px > 50")
        objects.get_filter_revision(target="objects")

        self.assertEqual(
            client.calls[0],
            (
                "viewer.objects.filters.set_model",
                {
                    "mode": "simple",
                    "clauses": [{"property": "class", "query": "tumor"}],
                    "logic": "any",
                    "if_revision": 8,
                },
            ),
        )
        self.assertEqual(client.calls[1][1]["query"], "area_px > 50")
        self.assertEqual(client.calls[2][0], "viewer.objects.filters.get_revision")

    def test_object_geometry_wrappers(self) -> None:
        client = RecordingClient()
        objects = Objects(client)  # type: ignore[arg-type]
        points = [[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]]

        objects.query_lasso(points)
        objects.select_lasso(points, mode="toggle", if_revision=3)
        objects.select_rect([0, 0, 5, 5], mode="remove")

        self.assertEqual(client.calls[0][0], "viewer.objects.query_lasso")
        self.assertEqual(client.calls[1][1]["mode"], "toggle")
        self.assertEqual(client.calls[2][1]["mode"], "remove")

    def test_object_selection_wrappers(self) -> None:
        client = RecordingClient()
        objects = Objects(client)  # type: ignore[arg-type]

        objects.select_ids(["cell-1", "cell-2"], mode="add", if_revision=11)
        objects.select_filtered(mode="replace", viewport_id="viewport-left")
        objects.focus("cell-2", fit=False)
        objects.focus(3)
        objects.clear_focus(if_revision=12)

        self.assertEqual(client.calls[0][0], "viewer.objects.selection.select_ids")
        self.assertEqual(client.calls[0][1]["mode"], "add")
        self.assertEqual(client.calls[1][1]["viewport_id"], "viewport-left")
        self.assertEqual(client.calls[2][1], {"id": "cell-2", "fit": False})
        self.assertEqual(client.calls[3][1], {"index": 3, "fit": True})
        self.assertEqual(client.calls[4][1]["if_revision"], 12)

    def test_object_source_wrappers(self) -> None:
        client = RecordingClient()
        objects = Objects(client)  # type: ignore[arg-type]

        objects.load("cells.parquet", downsample_factor=2.0, if_revision=3)
        objects.reload()
        objects.cancel_load()
        objects.clear(if_revision=4)

        self.assertEqual(client.calls[0][0], "viewer.objects.source.load")
        self.assertEqual(client.calls[0][1]["downsample_factor"], 2.0)
        self.assertEqual(client.calls[1][0], "viewer.objects.source.reload")
        self.assertEqual(client.calls[2], ("viewer.objects.source.cancel_load", {}))
        self.assertEqual(
            client.calls[3], ("viewer.objects.source.clear", {"if_revision": 4})
        )

    def test_object_property_wrappers(self) -> None:
        client = RecordingClient()
        objects = Objects(client)  # type: ignore[arg-type]

        objects.get_state(target="objects")
        objects.list_properties(offset=10, limit=25, target="objects")
        objects.load_property("cell_type", if_revision=4, target="objects")
        objects.get_property_values("cell_type", offset=100, limit=50)

        self.assertEqual(
            client.calls,
            [
                ("viewer.objects.get_state", {"target": "objects"}),
                (
                    "viewer.objects.properties.list",
                    {"target": "objects", "offset": 10, "limit": 25},
                ),
                (
                    "viewer.objects.properties.load",
                    {
                        "target": "objects",
                        "property": "cell_type",
                        "if_revision": 4,
                        "_label": "Load object property cell_type",
                    },
                ),
                (
                    "viewer.objects.properties.values",
                    {"property": "cell_type", "offset": 100, "limit": 50},
                ),
            ],
        )


class AsyncObjectResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_object_style_wrappers(self) -> None:
        client = AsyncRecordingClient()
        objects = AsyncObjects(client)  # type: ignore[arg-type]

        await objects.get_style()
        await objects.set_style(color_rgb=[1, 2, 3])
        await objects.set_legend([{"value": "A", "visible": True}])
        await objects.get_fast_rendering()
        await objects.set_fast_rendering(True)

        self.assertEqual(client.calls[1][0], "viewer.objects.style.set")
        self.assertEqual(client.calls[4][1], {"enabled": True})

    async def test_async_object_filter_model_wrappers(self) -> None:
        client = AsyncRecordingClient()
        objects = AsyncObjects(client)  # type: ignore[arg-type]

        await objects.set_filter_model(
            clauses=[{"property": "id", "query": "cell"}], logic="all"
        )
        await objects.get_filter_revision()

        self.assertEqual(client.calls[0][0], "viewer.objects.filters.set_model")
        self.assertEqual(client.calls[1][0], "viewer.objects.filters.get_revision")

    async def test_async_object_geometry_wrappers(self) -> None:
        client = AsyncRecordingClient()
        objects = AsyncObjects(client)  # type: ignore[arg-type]
        points = [[0, 0], [2, 0], [1, 2]]

        await objects.query_lasso(points)
        await objects.select_lasso(points, mode="add")

        self.assertEqual(client.calls[0][0], "viewer.objects.query_lasso")
        self.assertEqual(client.calls[1][1]["mode"], "add")

    async def test_async_object_selection_wrappers(self) -> None:
        client = AsyncRecordingClient()
        objects = AsyncObjects(client)  # type: ignore[arg-type]

        await objects.select_ids(["a"], mode="toggle")
        await objects.select_filtered(mode="remove", filter_query="score > 2")
        await objects.focus("a")
        await objects.clear_focus()

        self.assertEqual(client.calls[0][1]["mode"], "toggle")
        self.assertEqual(client.calls[1][1]["filter_query"], "score > 2")
        self.assertEqual(client.calls[2][0], "viewer.objects.focus.set")

    async def test_async_object_source_wrappers(self) -> None:
        client = AsyncRecordingClient()
        objects = AsyncObjects(client)  # type: ignore[arg-type]

        await objects.load("cells.csv")
        await objects.reload(if_revision=2)
        await objects.cancel_load()
        await objects.clear()

        self.assertEqual(client.calls[0][0], "viewer.objects.source.load")
        self.assertEqual(client.calls[1][1]["if_revision"], 2)

    async def test_async_object_property_wrappers(self) -> None:
        client = AsyncRecordingClient()
        objects = AsyncObjects(client)  # type: ignore[arg-type]

        await objects.get_state()
        await objects.list_properties(limit=5)
        await objects.load_property("cluster")
        await objects.get_property_values("cluster", limit=10)

        self.assertEqual(client.calls[0], ("viewer.objects.get_state", {}))
        self.assertEqual(client.calls[2][0], "viewer.objects.properties.load")
        self.assertEqual(client.calls[3][1]["limit"], 10)


if __name__ == "__main__":
    unittest.main()
