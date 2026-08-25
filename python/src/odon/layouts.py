"""Reusable actor-owned application-shell layouts for common Odon workflows."""

from __future__ import annotations

from collections.abc import Iterable

from .ui import ShellLayout, ShellLayoutNode, ShellMountId, ShellSize


def review(*, panel_mounts: Iterable[str] = ()) -> ShellLayout:
    """Build a single-view review workspace with project, views, and ROI panels."""

    return _single_workspace(
        "review",
        left=(ShellMountId.LAYERS, ShellMountId.PROJECT),
        right=(ShellMountId.PROPERTIES, ShellMountId.VIEWS, ShellMountId.ROI_SELECTOR),
        panel_mounts=panel_mounts,
        left_ratio=0.24,
        canvas_ratio=0.68,
    )


def analysis(*, panel_mounts: Iterable[str] = ()) -> ShellLayout:
    """Build a single-view analysis workspace with analysis and measurement panels."""

    return _single_workspace(
        "analysis",
        left=(ShellMountId.LAYERS, ShellMountId.PROJECT),
        right=(ShellMountId.ANALYSIS, ShellMountId.MEASUREMENTS, ShellMountId.MEMORY),
        panel_mounts=panel_mounts,
        left_ratio=0.22,
        canvas_ratio=0.66,
    )


def comparison(*, panel_mounts: Iterable[str] = ()) -> ShellLayout:
    """Build a single-view comparison shell around Odon's native viewport workspace."""

    return _single_workspace(
        "comparison",
        left=(ShellMountId.LAYERS,),
        right=(ShellMountId.VIEWS, ShellMountId.PROPERTIES),
        panel_mounts=panel_mounts,
        left_ratio=0.18,
        canvas_ratio=0.74,
    )


def mosaic_triage(*, panel_mounts: Iterable[str] = ()) -> ShellLayout:
    """Build a mosaic triage workspace with project, layout, property, and memory panels."""

    prefix = "layout:workflow.mosaic-triage"
    extension_ids, extension_nodes = _extension_panels(prefix, panel_mounts)
    left_ids, left_nodes = _builtin_panels(
        prefix, "left", (ShellMountId.LAYERS, ShellMountId.PROJECT)
    )
    right_ids, right_nodes = _builtin_panels(
        prefix,
        "right",
        (ShellMountId.MOSAIC_LAYOUT, ShellMountId.PROPERTIES, ShellMountId.MEMORY),
    )
    right_children = (*right_ids, *extension_ids, f"{prefix}.right.default-extensions")
    return ShellLayout(
        f"{prefix}.root",
        (
            ShellLayoutNode.application(
                f"{prefix}.root", [f"{prefix}.top", f"{prefix}.body", f"{prefix}.status"]
            ),
            ShellLayoutNode.toolbar(
                f"{prefix}.top", [f"{prefix}.native-top", f"{prefix}.top-actions"]
            ),
            ShellLayoutNode.builtin(f"{prefix}.native-top", ShellMountId.MOSAIC_TOP_BAR),
            ShellLayoutNode.builtin(
                f"{prefix}.top-actions", ShellMountId.EXTENSION_TOP_BAR_ACTIONS
            ),
            ShellLayoutNode.split_node(
                f"{prefix}.body",
                f"{prefix}.left",
                f"{prefix}.center-right",
                ratio=0.24,
            ),
            ShellLayoutNode.panel(
                f"{prefix}.left", f"{prefix}.left-tabs", size=ShellSize(width=340)
            ),
            ShellLayoutNode.tabs(
                f"{prefix}.left-tabs", (*left_ids, f"{prefix}.left.default-extensions"),
                selected=left_ids[0],
            ),
            *left_nodes,
            ShellLayoutNode.builtin(
                f"{prefix}.left.default-extensions", ShellMountId.EXTENSION_LEFT_SECTIONS,
                title="Extensions",
            ),
            ShellLayoutNode.split_node(
                f"{prefix}.center-right",
                f"{prefix}.canvas-column",
                f"{prefix}.right",
                ratio=0.68,
            ),
            ShellLayoutNode.column(
                f"{prefix}.canvas-column", [f"{prefix}.canvas-controls", f"{prefix}.canvas"]
            ),
            ShellLayoutNode.builtin(
                f"{prefix}.canvas-controls", ShellMountId.EXTENSION_CANVAS_CONTROLS
            ),
            ShellLayoutNode.canvas(
                f"{prefix}.canvas", ShellMountId.MOSAIC_CANVAS,
                size=ShellSize(min_width=256, min_height=256),
            ),
            ShellLayoutNode.panel(
                f"{prefix}.right", f"{prefix}.right-tabs", size=ShellSize(width=380)
            ),
            ShellLayoutNode.tabs(
                f"{prefix}.right-tabs", right_children, selected=right_ids[0]
            ),
            *right_nodes,
            *extension_nodes,
            ShellLayoutNode.builtin(
                f"{prefix}.right.default-extensions", ShellMountId.EXTENSION_RIGHT_TABS,
                title="Extensions",
            ),
            ShellLayoutNode.status_bar(f"{prefix}.status", [f"{prefix}.status-items"]),
            ShellLayoutNode.builtin(
                f"{prefix}.status-items", ShellMountId.EXTENSION_STATUS_BAR
            ),
        ),
    )


def presentation(*, mode: str = "single", show_toolbar: bool = False) -> ShellLayout:
    """Build a canvas-first single or mosaic presentation workspace."""

    if mode not in {"single", "mosaic"}:
        raise ValueError("presentation mode must be 'single' or 'mosaic'")
    prefix = f"layout:workflow.presentation.{mode}"
    canvas = ShellMountId.VIEWER_CANVAS if mode == "single" else ShellMountId.MOSAIC_CANVAS
    top = ShellMountId.VIEWER_TOP_BAR if mode == "single" else ShellMountId.MOSAIC_TOP_BAR
    root_children = [f"{prefix}.canvas"]
    nodes = [
        ShellLayoutNode.canvas(
            f"{prefix}.canvas", canvas, size=ShellSize(min_width=256, min_height=256)
        )
    ]
    if show_toolbar:
        root_children.insert(0, f"{prefix}.top")
        nodes.insert(0, ShellLayoutNode.builtin(f"{prefix}.top", top))
    return ShellLayout(
        f"{prefix}.root",
        (ShellLayoutNode.application(f"{prefix}.root", root_children), *nodes),
    )


def _single_workspace(
    name: str,
    *,
    left: tuple[ShellMountId, ...],
    right: tuple[ShellMountId, ...],
    panel_mounts: Iterable[str],
    left_ratio: float,
    canvas_ratio: float,
) -> ShellLayout:
    prefix = f"layout:workflow.{name}"
    extension_ids, extension_nodes = _extension_panels(prefix, panel_mounts)
    left_ids, left_nodes = _builtin_panels(prefix, "left", left)
    right_ids, right_nodes = _builtin_panels(prefix, "right", right)
    return ShellLayout(
        f"{prefix}.root",
        (
            ShellLayoutNode.application(
                f"{prefix}.root", [f"{prefix}.top", f"{prefix}.body", f"{prefix}.status"]
            ),
            ShellLayoutNode.toolbar(
                f"{prefix}.top", [f"{prefix}.native-top", f"{prefix}.top-actions"]
            ),
            ShellLayoutNode.builtin(f"{prefix}.native-top", ShellMountId.VIEWER_TOP_BAR),
            ShellLayoutNode.builtin(
                f"{prefix}.top-actions", ShellMountId.EXTENSION_TOP_BAR_ACTIONS
            ),
            ShellLayoutNode.split_node(
                f"{prefix}.body", f"{prefix}.left", f"{prefix}.center-right",
                ratio=left_ratio,
            ),
            ShellLayoutNode.panel(
                f"{prefix}.left", f"{prefix}.left-tabs", size=ShellSize(width=340)
            ),
            ShellLayoutNode.tabs(
                f"{prefix}.left-tabs", (*left_ids, f"{prefix}.left.default-extensions"),
                selected=left_ids[0],
            ),
            *left_nodes,
            ShellLayoutNode.builtin(
                f"{prefix}.left.default-extensions", ShellMountId.EXTENSION_LEFT_SECTIONS,
                title="Extensions",
            ),
            ShellLayoutNode.split_node(
                f"{prefix}.center-right", f"{prefix}.canvas-column", f"{prefix}.right",
                ratio=canvas_ratio,
            ),
            ShellLayoutNode.column(
                f"{prefix}.canvas-column", [f"{prefix}.canvas-controls", f"{prefix}.canvas"]
            ),
            ShellLayoutNode.builtin(
                f"{prefix}.canvas-controls", ShellMountId.EXTENSION_CANVAS_CONTROLS
            ),
            ShellLayoutNode.canvas(
                f"{prefix}.canvas", ShellMountId.VIEWER_CANVAS,
                size=ShellSize(min_width=256, min_height=256),
            ),
            ShellLayoutNode.panel(
                f"{prefix}.right", f"{prefix}.right-tabs", size=ShellSize(width=380)
            ),
            ShellLayoutNode.tabs(
                f"{prefix}.right-tabs",
                (*right_ids, *extension_ids, f"{prefix}.right.default-extensions"),
                selected=right_ids[0],
            ),
            *right_nodes,
            *extension_nodes,
            ShellLayoutNode.builtin(
                f"{prefix}.right.default-extensions", ShellMountId.EXTENSION_RIGHT_TABS,
                title="Extensions",
            ),
            ShellLayoutNode.status_bar(f"{prefix}.status", [f"{prefix}.status-items"]),
            ShellLayoutNode.builtin(
                f"{prefix}.status-items", ShellMountId.EXTENSION_STATUS_BAR
            ),
        ),
    )


def _builtin_panels(
    prefix: str, side: str, mounts: tuple[ShellMountId, ...]
) -> tuple[tuple[str, ...], tuple[ShellLayoutNode, ...]]:
    ids = tuple(f"{prefix}.{side}.native.{index}" for index in range(len(mounts)))
    return ids, tuple(
        ShellLayoutNode.builtin(node_id, mount) for node_id, mount in zip(ids, mounts)
    )


def _extension_panels(
    prefix: str, mounts: Iterable[str]
) -> tuple[tuple[str, ...], tuple[ShellLayoutNode, ...]]:
    mounts = tuple(mounts)
    if len(mounts) != len(set(mounts)):
        raise ValueError("panel_mounts must not contain duplicate extension mounts")
    for mount in mounts:
        if not isinstance(mount, str) or not mount.startswith("extension:") or "/" not in mount:
            raise ValueError("panel_mounts must contain registered extension shell mount IDs")
    ids = tuple(f"{prefix}.extension.{index}" for index in range(len(mounts)))
    return ids, tuple(
        ShellLayoutNode.extension(node_id, mount) for node_id, mount in zip(ids, mounts)
    )
