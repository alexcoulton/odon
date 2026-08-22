"""Manual platform acceptance for Odon's no-frame control actor.

Build and launch the current Odon tree, put its window in the condition named by
``--condition``, then run from the repository root:

    uv run --project python python scripts/verify_background_control.py \
        --condition covered

Repeat with ``visible``, ``minimized``, and ``separate-space``.  Do not switch
to Odon while this script runs.  Success means semantic/resource work completed
without a GUI frame; switch back afterwards and inspect the final two-view
projection.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import odon


ACTOR_METHODS = (
    "app.get_state",
    "datasets.open_ome_zarr",
    "datasets.open_tiff",
    "datasets.open_spatialdata",
    "datasets.open_xenium",
    "datasets.open_http",
    "datasets.open_s3",
    "datasets.s3.get_session",
    "datasets.s3.configure_session",
    "datasets.s3.clear_session",
    "datasets.s3.list",
    "viewer.workspace.get",
    "viewer.viewports.clone",
    "viewer.viewports.rename",
    "viewer.viewport_links.set",
    "viewer.viewports.channels.set_visible",
    "viewer.viewports.channels.set_color",
    "viewer.viewports.channels.set_order",
    "viewer.viewports.channels.set_group",
    "viewer.channels.set_note",
    "viewer.channels.set_transform",
    "viewer.channels.intensity_stats",
    "viewer.channels.presentation.set",
    "viewer.panels.set",
    "viewer.ui.set_right_tab",
    "viewer.scale_bar.get",
    "viewer.scale_bar.set",
    "viewer.viewports.rendering.set",
    "viewer.rendering.get_state",
    "viewer.objects.source.load",
    "viewer.objects.properties.values",
    "viewer.objects.get_visibility",
    "viewer.objects.set_visibility",
    "viewer.viewports.objects.style.set",
    "viewer.viewports.objects.filter.set",
    "viewer.viewports.layers.list",
    "viewer.viewports.layers.get",
    "viewer.viewports.layers.set",
    "viewer.viewports.layers.set_visibility",
    "viewer.viewports.layers.set_order",
    "viewer.viewports.layers.set_active",
    "viewer.native_layers.list",
    "viewer.native_layers.get",
    "viewer.native_layers.set_active",
    "viewer.native_layers.set_visibility",
    "viewer.native_layers.set_order",
    "viewer.native_layers.set_offset",
    "viewer.native_layers.reset_offset",
    "viewer.labels.list",
    "viewer.labels.get",
    "viewer.labels.load",
    "viewer.labels.unload",
    "viewer.labels.set_visibility",
    "viewer.thresholds.levels.list",
    "viewer.masks.layers.create",
    "viewer.masks.polygons.add",
    "viewer.masks.selection.set",
    "viewer.masks.layers.list",
    "viewer.masks.persistence.sync",
    "viewer.viewports.camera.fit",
    "project.views.capture",
)

HYBRID_METHODS = (
    "viewer.objects.selection.select_filtered",
    "viewer.objects.query_rect",
    "viewer.objects.get_selection",
    "project.views.apply",
)


def call(app: Any, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    result = app.call(method, params or {})
    if not isinstance(result, dict):
        raise AssertionError(f"{method} returned a non-object result: {result!r}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        required=True,
        choices=("visible", "covered", "minimized", "separate-space"),
        help="Native-window condition maintained for the complete run.",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("fixtures/synthetic_5ch.ome.zarr"),
    )
    parser.add_argument(
        "--objects-fixture",
        type=Path,
        default=Path("fixtures/synthetic_objects.geojson"),
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    fixture = args.fixture.expanduser().resolve()
    if not fixture.exists():
        raise FileNotFoundError(fixture)
    objects_fixture = args.objects_fixture.expanduser().resolve()
    if not objects_fixture.exists():
        raise FileNotFoundError(objects_fixture)

    started = time.monotonic()
    app = odon.connect()
    open_task = app.datasets.open_ome_zarr(fixture)
    open_result = open_task.wait(timeout=args.timeout)

    workspace = call(app, "viewer.workspace.get")
    viewports = workspace["viewports"]
    if len(viewports) == 2:
        call(
            app,
            "viewer.viewports.remove",
            {"viewport_id": viewports[1]["viewport_id"]},
        )
        workspace = call(app, "viewer.workspace.get")
        viewports = workspace["viewports"]
    if len(viewports) != 1:
        raise AssertionError(f"expected one reset viewport, got {len(viewports)}")

    left = viewports[0]["viewport_id"]
    call(app, "viewer.viewports.rename", {"viewport_id": left, "title": "Left"})
    right = call(
        app,
        "viewer.viewports.clone",
        {
            "viewport_id": left,
            "title": "Right",
            "layout": "horizontal",
            "ratio": 0.5,
        },
    )["viewport_id"]
    call(
        app,
        "viewer.viewport_links.set",
        {"camera": True, "plane": True, "selection": True},
    )
    objects_loaded = call(
        app,
        "viewer.objects.source.load",
        {"path": str(objects_fixture), "downsample_factor": 1.0},
    )
    object_visibility = call(
        app,
        "viewer.objects.set_visibility",
        {"target": "objects", "visible": True},
    )
    labels = call(app, "viewer.labels.list")
    loaded_labels = call(app, "viewer.labels.load", {"name": "cells"})
    call(app, "viewer.labels.set_visibility", {"visible": False})
    shown_labels = call(app, "viewer.labels.set_visibility", {"visible": True})
    threshold_levels = call(app, "viewer.thresholds.levels.list")
    mask_id = call(
        app,
        "viewer.masks.layers.create",
        {"name": f"Background acceptance ({args.condition})"},
    )["id"]
    call(
        app,
        "viewer.masks.polygons.add",
        {
            "id": mask_id,
            "vertices": [[8.0, 8.0], [48.0, 8.0], [48.0, 48.0], [8.0, 48.0]],
        },
    )
    call(
        app,
        "viewer.masks.selection.set",
        {"id": mask_id, "index": 0, "vertex_index": 1},
    )
    masks = call(app, "viewer.masks.layers.list")
    mask_persistence = call(app, "viewer.masks.persistence.sync")
    mask_layer_id = f"mask:{mask_id}"
    call(
        app,
        "viewer.viewports.layers.set_visibility",
        {"viewport_id": left, "layer_id": mask_layer_id, "visible": True},
    )
    hidden_mask = call(
        app,
        "viewer.viewports.layers.set_visibility",
        {"viewport_id": right, "layer_id": mask_layer_id, "visible": False},
    )
    native_mask = call(app, "viewer.native_layers.get", {"layer_id": mask_layer_id})
    moved_channel = call(
        app,
        "viewer.native_layers.set_offset",
        {"layer_id": "channel:0", "offset_world": [1.0, -2.0]},
    )
    reset_channel = call(
        app,
        "viewer.native_layers.reset_offset",
        {"layer_id": "channel:0"},
    )

    for viewport_id, channel, color in (
        (left, 0, [80, 140, 255]),
        (right, 1, [255, 90, 120]),
    ):
        call(
            app,
            "viewer.viewports.channels.set_visible",
            {"viewport_id": viewport_id, "channels": [channel], "mode": "only"},
        )
        call(
            app,
            "viewer.viewports.channels.set_color",
            {"viewport_id": viewport_id, "channel": channel, "color_rgb": color},
        )
        call(
            app,
            "viewer.viewports.rendering.set",
            {"viewport_id": viewport_id, "smooth_pixels": viewport_id == right},
        )

    for viewport_id, fill_opacity, query in (
        (left, 0.25, "phenotype == 'tumour'"),
        (right, 0.75, "phenotype == 'immune'"),
    ):
        call(
            app,
            "viewer.viewports.objects.style.set",
            {
                "viewport_id": viewport_id,
                "visible": True,
                "fill_cells": True,
                "fill_opacity": fill_opacity,
                "color_property": "phenotype",
            },
        )
        filtered = call(
            app,
            "viewer.viewports.objects.filter.set",
            {"viewport_id": viewport_id, "mode": "query", "query": query},
        )
        assert filtered["result"]["visible_count"] == 1, filtered

    property_values = call(
        app,
        "viewer.objects.properties.values",
        {"property": "phenotype", "offset": 0, "limit": 10},
    )
    selected = call(
        app,
        "viewer.objects.selection.select_filtered",
        {
            "target": "segmentation_objects",
            "filter_query": "phenotype == 'immune'",
            "mode": "replace",
        },
    )
    queried = call(
        app,
        "viewer.objects.query_rect",
        {
            "target": "segmentation_objects",
            "viewport_id": left,
            "rect": [0.0, 0.0, 15.0, 15.0],
        },
    )
    current_selection = call(
        app,
        "viewer.objects.get_selection",
        {"target": "segmentation_objects", "limit": 10},
    )

    call(app, "viewer.channels.set_note", {"channel": 1, "note": args.condition})
    call(
        app,
        "viewer.channels.set_transform",
        {
            "channel": 1,
            "offset_world": [2.0, -1.0],
            "scale": [1.05, 0.95],
            "rotation_rad": 0.05,
        },
    )
    call(
        app,
        "viewer.viewports.channels.set_order",
        {"viewport_id": left, "channels": [4, 3, 2, 1, 0], "mode": "exact"},
    )
    call(
        app,
        "viewer.viewports.channels.set_group",
        {"viewport_id": right, "channels": [0, 1], "name": "Acceptance"},
    )
    call(app, "viewer.channels.presentation.set", {"search": "", "sort": "manual"})
    call(app, "viewer.panels.set", {"left": False, "right": False})
    right_tab = call(app, "viewer.ui.set_right_tab", {"tab": "measurements"})
    call(app, "viewer.scale_bar.set", {"visible": True})
    stats = call(app, "viewer.channels.intensity_stats", {"channel": 0, "level": 0})
    fit = call(app, "viewer.viewports.camera.fit", {"viewport_id": left})
    captured_view = call(
        app,
        "project.views.capture",
        {"name": f"Background acceptance ({args.condition})", "viewport_id": left},
    )
    applied_view = call(
        app,
        "project.views.apply",
        {"name": f"Background acceptance ({args.condition})"},
    )

    final_workspace = call(app, "viewer.workspace.get")
    application_state = call(app, "app.get_state")
    rendering_state = call(app, "viewer.rendering.get_state")
    loading = call(app, "app.get_loading_state")
    diagnostics = call(app, "system.get_diagnostics")
    routes = diagnostics["dispatch"]["method_routes"]
    wrong_routes = {method: routes.get(method) for method in ACTOR_METHODS if routes.get(method) != "actor"}
    wrong_hybrid_routes = {
        method: routes.get(method)
        for method in HYBRID_METHODS
        if routes.get(method) != "hybrid"
    }

    assert stats["n"] > 0, stats
    assert objects_loaded["object_count"] == 2, objects_loaded
    assert object_visibility["overlay"]["segmentation_objects"] is True
    assert labels["available"] == ["cells"], labels
    assert loaded_labels["loaded"] == "cells", loaded_labels
    assert shown_labels["visible"] is True, shown_labels
    assert threshold_levels["levels"][0]["width"] == 512, threshold_levels
    assert masks["layers"][-1]["polygon_count"] == 1, masks
    assert mask_persistence["persistence"]["dirty"] is False, mask_persistence
    assert hidden_mask["result"]["layer"]["visible"] is False, hidden_mask
    assert native_mask["layer"]["kind"] == "mask", native_mask
    assert moved_channel["result"]["layer"]["offset_world"] == [1.0, -2.0]
    assert reset_channel["result"]["layer"]["offset_world"] == [0.0, 0.0]
    assert [item["value"] for item in property_values["values"]] == [
        "tumour",
        "immune",
    ], property_values
    assert selected["selection"]["primary"]["id"] == "cell-b", selected
    assert queried["objects"]["query"]["matches"][0]["id"] == "cell-a", queried
    assert (
        current_selection["objects"]["selection"]["primary"]["id"] == "cell-b"
    ), current_selection
    assert fit["result"]["zoom_screen_per_lvl0_px"] > 0, fit
    assert captured_view["captured"] is True, captured_view
    assert applied_view["applied"] is True, applied_view
    assert right_tab["tab"]["right_tab"] == "measurements", right_tab
    assert final_workspace["ui"]["right_tab"] == "measurements", final_workspace
    assert final_workspace["layout"] == "horizontal", final_workspace
    assert len(final_workspace["viewports"]) == 2, final_workspace
    assert application_state["mode"] == "single", application_state
    assert application_state["view"]["channel_count"] == 5, application_state
    assert rendering_state["compositing"] == "additive", rendering_state
    assert loading["loading"]["model_ready"] is True, loading
    assert loading["loading"]["resources_ready"] is True, loading
    assert loading["loading"]["geometry_ready"] is True, loading
    assert not wrong_routes, wrong_routes
    assert not wrong_hybrid_routes, wrong_hybrid_routes

    report = {
        "condition": args.condition,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "open_result": open_result,
        "model_ready": loading["loading"]["model_ready"],
        "resources_ready": loading["loading"]["resources_ready"],
        "geometry_ready": loading["loading"]["geometry_ready"],
        "presentation_ready": loading["loading"]["presentation_ready"],
        "projection_revision": loading["loading"]["projection_revision"],
        "presented_projection_revision": loading["loading"][
            "presented_projection_revision"
        ],
        "viewports": [item["viewport_id"] for item in final_workspace["viewports"]],
        "channel_stat_count": stats["n"],
        "actor_methods_checked": len(ACTOR_METHODS),
    }
    print(json.dumps(report, indent=2, default=str))
    print(
        "PASS: semantic/resource work completed. Switch to Odon now and verify "
        "that the final two-view projection appears without another API call."
    )


if __name__ == "__main__":
    main()
