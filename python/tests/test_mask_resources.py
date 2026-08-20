from __future__ import annotations

import unittest
from typing import Any, Mapping

from odon.async_resources import AsyncMasks
from odon.resources import Masks


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

    async def call(
        self, method: str, params: Mapping[str, Any] | None = None
    ) -> Any:
        recorded = (method, dict(params or {}))
        self.calls.append(recorded)
        return recorded


class MaskResourceTests(unittest.TestCase):
    def test_mask_crud_wrappers(self) -> None:
        client = RecordingClient()
        masks = Masks(client)  # type: ignore[arg-type]
        vertices = [[0, 0], [10, 0], [5, 10]]

        masks.list_layers()
        masks.get_layer(2)
        masks.create_layer("Review", if_revision=1)
        masks.update_layer(2, opacity=0.5, visible=False)
        masks.list_polygons(2, offset=5, limit=10)
        masks.add_polygon(2, vertices)
        masks.update_polygon(2, 0, vertices, coordinate_space="local")
        masks.remove_polygon(2, 0)
        masks.get_selection()
        masks.select(2, 0, vertex_index=1)
        masks.clear_selection()
        masks.undo()
        masks.delete_layer(2)
        masks.import_geojson("masks.geojson", name="Imported")
        masks.export_geojson("result.geojson", layer_id=2, overwrite=True)
        masks.get_persistence()
        masks.sync_to_project()

        self.assertEqual(client.calls[0], ("viewer.masks.layers.list", {}))
        self.assertEqual(client.calls[2][1]["name"], "Review")
        self.assertEqual(client.calls[5][1]["coordinate_space"], "world")
        self.assertEqual(client.calls[6][1]["coordinate_space"], "local")
        self.assertEqual(client.calls[8], ("viewer.masks.selection.get", {}))
        self.assertEqual(client.calls[9][1]["vertex_index"], 1)
        self.assertEqual(client.calls[10][0], "viewer.masks.selection.clear")
        self.assertEqual(client.calls[11], ("viewer.masks.undo", {}))
        self.assertEqual(client.calls[13][1]["name"], "Imported")
        self.assertEqual(client.calls[14][1]["overwrite"], True)
        self.assertEqual(client.calls[16][0], "viewer.masks.persistence.sync")


class AsyncMaskResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_mask_crud_wrappers(self) -> None:
        client = AsyncRecordingClient()
        masks = AsyncMasks(client)  # type: ignore[arg-type]
        vertices = [[0, 0], [1, 0], [0, 1]]

        await masks.create_layer()
        await masks.add_polygon(1, vertices)
        await masks.list_polygons(1)
        await masks.remove_polygon(1, 0)
        await masks.select(1, 0)
        await masks.clear_selection()
        await masks.undo(if_revision=3)
        await masks.import_geojson("masks.geojson")
        await masks.export_geojson("result.geojson")
        await masks.get_persistence()
        await masks.sync_to_project()

        self.assertEqual(client.calls[0][0], "viewer.masks.layers.create")
        self.assertEqual(client.calls[1][0], "viewer.masks.polygons.add")
        self.assertEqual(client.calls[4][0], "viewer.masks.selection.set")
        self.assertEqual(client.calls[6][1]["if_revision"], 3)
        self.assertEqual(client.calls[7][0], "viewer.masks.import_geojson")
        self.assertEqual(client.calls[10][0], "viewer.masks.persistence.sync")


if __name__ == "__main__":
    unittest.main()
