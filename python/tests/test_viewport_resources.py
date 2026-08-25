from __future__ import annotations

import unittest
from typing import Any

from odon.async_resources import AsyncViewportLinks, AsyncViewportWorkspace, AsyncViewports
from odon.resources import ViewportLinks, ViewportWorkspace, Viewports


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        params = dict(params or {})
        self.calls.append((method, params))
        if method == "viewer.workspace.get":
            return {
                "active_viewport_id": "viewport-1",
                "layout": "single",
                "viewports": [{"viewport_id": "viewport-1"}],
            }
        if method in {"viewer.viewports.create", "viewer.viewports.clone"}:
            return {"viewport_id": "viewport-2", "_control": {"revision": 12}}
        if method == "viewer.viewports.rename":
            return {"renamed": True, "_control": {"revision": 11}}
        return {"method": method, "params": params}


class AsyncRecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        params = dict(params or {})
        self.calls.append((method, params))
        if method == "viewer.workspace.get":
            return {
                "active_viewport_id": "viewport-1",
                "layout": "single",
                "viewports": [{"viewport_id": "viewport-1"}],
            }
        if method in {"viewer.viewports.create", "viewer.viewports.clone"}:
            return {"viewport_id": "viewport-2", "_control": {"revision": 12}}
        if method == "viewer.viewports.rename":
            return {"renamed": True, "_control": {"revision": 11}}
        return {"method": method, "params": params}


class ViewportResourceTests(unittest.TestCase):
    def test_comparison_builds_two_stable_handles_and_independent_styles(self) -> None:
        client = RecordingClient()
        viewports = Viewports(client)  # type: ignore[arg-type]

        comparison = viewports.compare(
            titles=("Marker A", "Marker B"),
            linked=("camera", "plane", "selection"),
        )
        comparison.left.objects.set_style(
            color_property="marker_a", fill_cells=True, fill_opacity=0.6
        )
        comparison.right.objects.set_style(
            color_property="marker_b", fill_cells=True, fill_opacity=0.6
        )

        self.assertEqual(comparison.left.id, "viewport-1")
        self.assertEqual(comparison.right.id, "viewport-2")
        self.assertIn(
            (
                "viewer.viewports.clone",
                {
                    "viewport_id": "viewport-1",
                    "title": "Marker B",
                    "layout": "horizontal",
                    "ratio": 0.5,
                    "activate": True,
                    "if_revision": 11,
                },
            ),
            client.calls,
        )
        self.assertIn(
            (
                "viewer.viewport_links.create",
                {
                    "link_group_id": "comparison-navigation",
                    "viewports": ["viewport-1", "viewport-2"],
                    "fields": ["camera", "plane", "selection"],
                    "if_revision": 12,
                },
            ),
            client.calls,
        )
        self.assertEqual(
            client.calls[-2],
            (
                "viewer.viewports.objects.style.set",
                {
                    "viewport_id": "viewport-1",
                    "color_property": "marker_a",
                    "fill_cells": True,
                    "fill_opacity": 0.6,
                },
            ),
        )
        self.assertEqual(
            client.calls[-1][1]["color_property"],
            "marker_b",
        )

    def test_viewport_navigation_channels_and_workspace_methods(self) -> None:
        client = RecordingClient()
        viewport = Viewports(client).handle("viewport-7")  # type: ignore[arg-type]
        workspace = ViewportWorkspace(client)  # type: ignore[arg-type]

        viewport.set_camera(center=(10, 20), zoom=2.5)
        viewport.fit_camera(if_navigation_revision=7)
        viewport.set_plane(mode="xz", slice=4)
        viewport.set_visible_channels(["DAPI", 2])
        viewport.set_channels(["DAPI", "CD3"], if_presentation_revision=38)
        viewport.set_channel_color("DAPI", [10, 20, 30])
        viewport.set_channel_contrast("DAPI", 100, 2000)
        viewport.set_object_filter(
            mode="simple",
            clauses=[{"property": "cell_type", "query": "tumor"}],
            logic="all",
        )
        viewport.set_object_legend(
            [{"value": "tumor", "color_rgb": [255, 0, 0]}],
            if_presentation_revision=39,
        )
        viewport.set_layer_visibility("segmentation_objects", False)
        viewport.set_layer(
            "mask:42",
            opacity=0.25,
            color_rgb=[10, 20, 30],
            if_presentation_revision=40,
        )
        viewport.get_layer("mask:42")
        viewport.set_rendering(
            smooth_pixels=False,
            show_scale_bar=False,
            show_hud=True,
            show_tile_debug=True,
            if_presentation_revision=41,
        )
        viewport.get_rendering()
        viewport.rename("Comparison")
        workspace.get_layout()
        workspace.get_links()
        workspace.swap()
        workspace.set_layout("vertical")
        workspace.set_layout(
            split="horizontal",
            viewports=[viewport, "viewport-8"],
            ratio=0.4,
        )
        workspace.set_links(camera=False, plane=True, selection=True)

        self.assertEqual(
            client.calls[0],
            (
                "viewer.viewports.camera.set",
                {
                    "viewport_id": "viewport-7",
                    "center_world_lvl0": [10.0, 20.0],
                    "zoom": 2.5,
                },
            ),
        )
        self.assertEqual(
            client.calls[1],
            (
                "viewer.viewports.camera.fit",
                {"viewport_id": "viewport-7", "if_navigation_revision": 7},
            ),
        )
        self.assertEqual(
            client.calls[-1],
            (
                "viewer.viewport_links.set",
                {"camera": False, "plane": True, "selection": True},
            ),
        )
        self.assertIn(
            (
                "viewer.workspace.layout.set",
                {
                    "layout": "horizontal",
                    "viewports": ["viewport-7", "viewport-8"],
                    "ratio": 0.4,
                },
            ),
            client.calls,
        )
        self.assertIn(
            (
                "viewer.viewports.channels.set",
                {
                    "viewport_id": "viewport-7",
                    "channels": ["DAPI", "CD3"],
                    "mode": "only",
                    "if_presentation_revision": 38,
                },
            ),
            client.calls,
        )
        self.assertIn(
            (
                "viewer.viewports.objects.legend.set",
                {
                    "viewport_id": "viewport-7",
                    "entries": [
                        {"value": "tumor", "color_rgb": [255, 0, 0]}
                    ],
                    "if_presentation_revision": 39,
                },
            ),
            client.calls,
        )
        self.assertIn(
            (
                "viewer.viewports.objects.filter.set",
                {
                    "viewport_id": "viewport-7",
                    "mode": "simple",
                    "clauses": [
                        {"property": "cell_type", "query": "tumor"}
                    ],
                    "logic": "all",
                },
            ),
            client.calls,
        )
        self.assertIn(
            (
                "viewer.viewports.layers.set",
                {
                    "viewport_id": "viewport-7",
                    "layer_id": "mask:42",
                    "presentation": {
                        "opacity": 0.25,
                        "color_rgb": [10, 20, 30],
                    },
                    "if_presentation_revision": 40,
                },
            ),
            client.calls,
        )
        self.assertIn(
            (
                "viewer.viewports.layers.get",
                {"viewport_id": "viewport-7", "layer_id": "mask:42"},
            ),
            client.calls,
        )
        self.assertIn(
            (
                "viewer.viewports.rendering.set",
                {
                    "viewport_id": "viewport-7",
                    "smooth_pixels": False,
                    "show_scale_bar": False,
                    "show_hud": True,
                    "show_tile_debug": True,
                    "if_presentation_revision": 41,
                },
            ),
            client.calls,
        )
        self.assertIn(
            ("viewer.viewports.rendering.get", {"viewport_id": "viewport-7"}),
            client.calls,
        )
        self.assertIn(
            (
                "viewer.viewports.layers.set_visibility",
                {
                    "viewport_id": "viewport-7",
                    "layer_id": "segmentation_objects",
                    "visible": False,
                },
            ),
            client.calls,
        )

    def test_invalid_comparison_arguments_fail_locally(self) -> None:
        client = RecordingClient()
        viewports = Viewports(client)  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            viewports.compare(titles=("only one",))
        with self.assertRaises(ValueError):
            viewports.compare(linked=("camera", "unknown"))

    def test_canonical_link_group_resource(self) -> None:
        client = RecordingClient()
        links = ViewportLinks(client)  # type: ignore[arg-type]
        left, right = Viewports(client).handle("viewport-1"), "viewport-2"  # type: ignore[arg-type]

        links.list()
        links.create(viewports=[left, right], fields=["camera"])
        links.update(fields=["plane"], if_revision=20)
        links.remove(if_revision=21)

        self.assertEqual(client.calls[0], ("viewer.viewport_links.list", {}))
        self.assertEqual(
            client.calls[1],
            (
                "viewer.viewport_links.create",
                {
                    "link_group_id": "comparison-navigation",
                    "viewports": ["viewport-1", "viewport-2"],
                    "fields": ["camera", "selection"],
                },
            ),
        )
        self.assertEqual(
            client.calls[2],
            (
                "viewer.viewport_links.update",
                {
                    "link_group_id": "comparison-navigation",
                    "fields": ["plane", "selection"],
                    "if_revision": 20,
                },
            ),
        )
        self.assertEqual(
            client.calls[3],
            (
                "viewer.viewport_links.remove",
                {
                    "link_group_id": "comparison-navigation",
                    "if_revision": 21,
                },
            ),
        )
        with self.assertRaises(ValueError):
            links.update(fields=["unknown"])

    def test_comparison_chains_global_revision_guards(self) -> None:
        client = RecordingClient()
        Viewports(client).compare(if_revision=10)  # type: ignore[arg-type]

        self.assertEqual(client.calls[1][1]["if_revision"], 10)
        self.assertEqual(client.calls[2][1]["if_revision"], 11)
        self.assertEqual(client.calls[3][1]["if_revision"], 12)

    def test_viewport_revision_guards_are_scoped_to_the_requested_domain(self) -> None:
        client = RecordingClient()
        viewport = Viewports(client).handle("viewport-9")  # type: ignore[arg-type]

        viewport.set_camera(zoom=2.0, if_navigation_revision=12)
        viewport.set_object_style(
            fill_cells=True, if_presentation_revision=34
        )
        viewport.rename("Guarded", if_presentation_revision=35)
        viewport.set_channel_group(
            ["DAPI", "CD3"],
            group="Nuclei",
            color_rgb=[10, 20, 30],
            if_presentation_revision=36,
        )

        self.assertEqual(client.calls[0][1]["if_navigation_revision"], 12)
        self.assertNotIn("if_presentation_revision", client.calls[0][1])
        self.assertEqual(client.calls[1][1]["if_presentation_revision"], 34)
        self.assertEqual(client.calls[2][1]["if_presentation_revision"], 35)
        self.assertEqual(client.calls[3][0], "viewer.viewports.channels.set_group")
        self.assertEqual(client.calls[3][1]["if_presentation_revision"], 36)

    def test_viewport_continuous_object_color_helper(self) -> None:
        client = RecordingClient()
        viewport = Viewports(client).handle("viewport-color")  # type: ignore[arg-type]

        viewport.objects.color_by_continuous(
            "mean_dapi",
            palette="viridis",
            domain=(0, 4095),
            fill_cells=True,
            if_presentation_revision=12,
        )

        method, params = client.calls[0]
        self.assertEqual(method, "viewer.viewports.objects.style.set")
        self.assertEqual(params["viewport_id"], "viewport-color")
        self.assertEqual(params["color_mapping"]["property"], "mean_dapi")
        self.assertEqual(params["if_presentation_revision"], 12)


class AsyncViewportResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_comparison_and_style_parity(self) -> None:
        client = AsyncRecordingClient()
        viewports = AsyncViewports(client)  # type: ignore[arg-type]

        comparison = await viewports.compare(titles=("A", "B"))
        await comparison.left.objects.set_style(
            color_property="property_a", fill_cells=True
        )
        await comparison.right.objects.set_style(
            color_property="property_b", fill_cells=True
        )
        await comparison.right.set_object_filter("cell_type == 'tumor'")
        await comparison.left.set_channel_order(
            ["DAPI", "CD3", "PanCK", "Ki67", "Collagen"]
        )
        await comparison.right.set_active_layer("segmentation_objects")
        await AsyncViewportWorkspace(client).get_layout()  # type: ignore[arg-type]
        await AsyncViewportWorkspace(client).get_links()  # type: ignore[arg-type]
        await AsyncViewportWorkspace(client).swap()  # type: ignore[arg-type]
        await AsyncViewportWorkspace(client).set_layout("vertical")  # type: ignore[arg-type]

        self.assertEqual(comparison.left.id, "viewport-1")
        self.assertEqual(comparison.right.id, "viewport-2")
        self.assertEqual(
            client.calls[-1],
            ("viewer.workspace.layout.set", {"layout": "vertical"}),
        )

        with self.assertRaises(ValueError):
            await AsyncViewportWorkspace(client).set_layout(  # type: ignore[arg-type]
                "horizontal", split="vertical"
            )

    async def test_async_viewport_continuous_object_color_helper(self) -> None:
        client = AsyncRecordingClient()
        viewport = AsyncViewports(client).handle("viewport-color")  # type: ignore[arg-type]

        await viewport.objects.color_by_continuous(
            "score", scale="log10", domain=(1, 1000), reverse=True
        )

        mapping = client.calls[0][1]["color_mapping"]
        self.assertEqual(mapping["scale"], "log10")
        self.assertEqual(mapping["domain"], [1.0, 1000.0])

    async def test_async_viewport_revision_guard_parity(self) -> None:
        client = AsyncRecordingClient()
        viewport = AsyncViewports(client).handle("viewport-3")  # type: ignore[arg-type]

        await viewport.set_plane(
            mode="xy", slice=2, if_navigation_revision=8
        )
        await viewport.fit_camera(if_navigation_revision=9)
        await viewport.set_channel_color(
            "DAPI", [1, 2, 3], if_presentation_revision=9
        )
        await viewport.set_channels(["DAPI"], if_presentation_revision=9)
        await viewport.set_object_legend(
            [{"value": "immune", "visible": False}],
            if_presentation_revision=10,
        )
        await viewport.list_channel_groups()
        await viewport.set_layer("channel:0", color_rgb=[3, 2, 1])
        await viewport.get_layer("channel:0")
        await viewport.set_rendering(show_hud=False, if_presentation_revision=10)
        await viewport.get_rendering()

        self.assertEqual(client.calls[0][1]["if_navigation_revision"], 8)
        self.assertEqual(client.calls[1][0], "viewer.viewports.camera.fit")
        self.assertEqual(client.calls[1][1]["if_navigation_revision"], 9)
        self.assertEqual(client.calls[2][1]["if_presentation_revision"], 9)
        self.assertEqual(client.calls[3][0], "viewer.viewports.channels.set")
        self.assertEqual(client.calls[3][1]["if_presentation_revision"], 9)
        self.assertEqual(client.calls[4][0], "viewer.viewports.objects.legend.set")
        self.assertEqual(client.calls[4][1]["if_presentation_revision"], 10)
        self.assertEqual(client.calls[5][0], "viewer.viewports.channels.list_groups")
        self.assertEqual(client.calls[6][0], "viewer.viewports.layers.set")
        self.assertEqual(client.calls[7][0], "viewer.viewports.layers.get")
        self.assertEqual(
            client.calls[8],
            (
                "viewer.viewports.rendering.set",
                {
                    "viewport_id": "viewport-3",
                    "show_hud": False,
                    "if_presentation_revision": 10,
                },
            ),
        )
        self.assertEqual(client.calls[9][0], "viewer.viewports.rendering.get")

    async def test_async_comparison_chains_global_revision_guards(self) -> None:
        client = AsyncRecordingClient()
        await AsyncViewports(client).compare(if_revision=10)  # type: ignore[arg-type]

        self.assertEqual(client.calls[1][1]["if_revision"], 10)
        self.assertEqual(client.calls[2][1]["if_revision"], 11)
        self.assertEqual(client.calls[3][1]["if_revision"], 12)

    async def test_async_canonical_link_group_resource(self) -> None:
        client = AsyncRecordingClient()
        links = AsyncViewportLinks(client)  # type: ignore[arg-type]

        await links.list()
        await links.create(viewports=["viewport-1", "viewport-2"], fields=["plane"])
        await links.update(fields=["camera", "selection"])
        await links.remove()

        self.assertEqual(client.calls[0][0], "viewer.viewport_links.list")
        self.assertEqual(client.calls[1][0], "viewer.viewport_links.create")
        self.assertEqual(client.calls[1][1]["fields"], ["plane", "selection"])
        self.assertEqual(client.calls[2][0], "viewer.viewport_links.update")
        self.assertEqual(client.calls[3][0], "viewer.viewport_links.remove")


if __name__ == "__main__":
    unittest.main()
