from __future__ import annotations

import unittest
from typing import Any, Mapping

from odon.async_resources import AsyncMosaic
from odon.resources import Mosaic


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


class AsyncRecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.tasks = AsyncRecordingTasks(self.calls)

    async def call(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
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


class MosaicResourceTests(unittest.TestCase):
    def test_mosaic_focus_wrappers(self) -> None:
        client = RecordingClient()
        mosaic = Mosaic(client)  # type: ignore[arg-type]

        mosaic.get_state()
        mosaic.get_focus()
        mosaic.set_focus("ROI-A", fit=False, if_revision=2)
        mosaic.next(2, wrap=False)
        mosaic.previous()
        mosaic.fit_focus()
        mosaic.list_items(offset=10, limit=20)
        mosaic.get_selection()
        mosaic.select(["ROI-A", "ROI-B"], mode="add", if_revision=5)
        mosaic.select_all()
        mosaic.select_range("ROI-A", "ROI-C")
        mosaic.clear_selection()
        mosaic.clear_focus()
        mosaic.fit_all()

        self.assertEqual(client.calls[0], ("mosaic.get_state", {}))
        self.assertEqual(client.calls[1], ("mosaic.focus.get", {}))
        self.assertEqual(
            client.calls[2],
            (
                "mosaic.focus.set",
                {"roi_id": "ROI-A", "fit": False, "if_revision": 2},
            ),
        )
        self.assertEqual(
            client.calls[3],
            ("mosaic.focus.next", {"step": 2, "wrap": False}),
        )
        self.assertIn(
            ("mosaic.items.list", {"offset": 10, "limit": 20}), client.calls
        )
        self.assertIn(
            (
                "mosaic.selection.set",
                {"ids": ["ROI-A", "ROI-B"], "mode": "add", "if_revision": 5},
            ),
            client.calls,
        )
        self.assertIn(
            (
                "mosaic.selection.set",
                {"mode": "range", "start": "ROI-A", "end": "ROI-C"},
            ),
            client.calls,
        )

    def test_mosaic_object_wrappers(self) -> None:
        client = RecordingClient()
        mosaic = Mosaic(client)  # type: ignore[arg-type]

        mosaic.get_object_state()
        mosaic.get_object_style()
        mosaic.set_object_style(fill_cells=True, opacity=0.4)
        mosaic.get_object_selection(roi_id="ROI-A")
        mosaic.replace_object_selection(
            item_id=2, selected_indices=[1, 3], primary_index=3
        )
        mosaic.clear_object_selection()
        mosaic.load_selected_objects(if_revision=9)
        mosaic.load_objects(item_ids=[1, 2], downsample_factor=2.0)
        mosaic.cancel_object_load(if_revision=10)

        self.assertEqual(client.calls[0], ("mosaic.objects.get_state", {}))
        self.assertEqual(client.calls[1], ("mosaic.objects.style.get", {}))
        self.assertEqual(
            client.calls[2],
            ("mosaic.objects.style.set", {"style": {"fill_cells": True, "opacity": 0.4}}),
        )
        self.assertEqual(
            client.calls[3], ("mosaic.objects.selection.get", {"roi_id": "ROI-A"})
        )
        self.assertEqual(client.calls[4][0], "mosaic.objects.selection.replace")
        self.assertEqual(client.calls[5], ("mosaic.objects.selection.clear", {}))
        self.assertEqual(
            client.calls[6],
            (
                "mosaic.objects.load_selected",
                {"if_revision": 9, "_label": "Load selected mosaic objects"},
            ),
        )
        self.assertEqual(
            client.calls[7][0],
            "mosaic.objects.load",
        )
        self.assertEqual(client.calls[7][1]["item_ids"], [1, 2])
        self.assertEqual(
            client.calls[8],
            ("mosaic.objects.cancel_load", {"if_revision": 10}),
        )

    def test_mosaic_presentation_wrappers(self) -> None:
        client = RecordingClient()
        mosaic = Mosaic(client)  # type: ignore[arg-type]

        mosaic.set_left_tab("project", if_revision=3)
        mosaic.set_right_tab("layout")
        mosaic.set_rendering(smooth_pixels=False, show_tile_debug=True, if_revision=4)

        self.assertEqual(
            client.calls[0],
            ("mosaic.ui.set_left_tab", {"tab": "project", "if_revision": 3}),
        )
        self.assertEqual(
            client.calls[1], ("mosaic.ui.set_right_tab", {"tab": "layout"})
        )
        self.assertEqual(
            client.calls[2],
            (
                "mosaic.rendering.set",
                {
                    "smooth_pixels": False,
                    "show_tile_debug": True,
                    "if_revision": 4,
                },
            ),
        )


class AsyncMosaicResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_mosaic_focus_wrappers(self) -> None:
        client = AsyncRecordingClient()
        mosaic = AsyncMosaic(client)  # type: ignore[arg-type]

        await mosaic.set_focus(1)
        await mosaic.previous(if_revision=4)
        await mosaic.select_all()
        await mosaic.clear_focus()
        await mosaic.fit_all()

        self.assertEqual(
            client.calls[0],
            ("mosaic.focus.set", {"index": 1, "fit": True}),
        )
        self.assertEqual(client.calls[1][1]["if_revision"], 4)
        self.assertEqual(client.calls[2], ("mosaic.selection.set", {"mode": "all"}))

    async def test_async_mosaic_object_wrappers(self) -> None:
        client = AsyncRecordingClient()
        mosaic = AsyncMosaic(client)  # type: ignore[arg-type]

        await mosaic.get_object_state()
        await mosaic.get_object_style()
        await mosaic.set_object_style(fill_cells=True)
        await mosaic.get_object_selection(item_id=1)
        await mosaic.replace_object_selection(item_id=1, selected_indices=[2])
        await mosaic.clear_object_selection(roi_id="ROI-A")
        await mosaic.load_selected_objects()
        await mosaic.load_objects(roi_ids=["ROI-A"])
        await mosaic.cancel_object_load()

        self.assertEqual(client.calls[0], ("mosaic.objects.get_state", {}))
        self.assertEqual(client.calls[1], ("mosaic.objects.style.get", {}))
        self.assertEqual(client.calls[2][0], "mosaic.objects.style.set")
        self.assertEqual(client.calls[3][0], "mosaic.objects.selection.get")
        self.assertEqual(client.calls[4][0], "mosaic.objects.selection.replace")
        self.assertEqual(client.calls[5][0], "mosaic.objects.selection.clear")
        self.assertEqual(client.calls[6][0], "mosaic.objects.load_selected")
        self.assertEqual(client.calls[7][0], "mosaic.objects.load")
        self.assertEqual(client.calls[7][1]["roi_ids"], ["ROI-A"])
        self.assertEqual(client.calls[8], ("mosaic.objects.cancel_load", {}))

    async def test_async_mosaic_presentation_wrappers(self) -> None:
        client = AsyncRecordingClient()
        mosaic = AsyncMosaic(client)  # type: ignore[arg-type]

        await mosaic.set_left_tab("layers")
        await mosaic.set_right_tab("memory", if_revision=5)
        await mosaic.set_rendering(show_tile_debug=False)

        self.assertEqual(
            client.calls[0], ("mosaic.ui.set_left_tab", {"tab": "layers"})
        )
        self.assertEqual(client.calls[1][1]["if_revision"], 5)
        self.assertEqual(
            client.calls[2],
            ("mosaic.rendering.set", {"show_tile_debug": False}),
        )


if __name__ == "__main__":
    unittest.main()
