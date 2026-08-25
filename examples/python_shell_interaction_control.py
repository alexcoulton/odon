"""Qualify shell interaction state and protected recovery from Python.

The checked-in five-channel OME-Zarr fixture is used by default. Run Odon separately and connect:

    uv run --project python python examples/python_shell_interaction_control.py

Or launch the repository debug binary and pause at each evidence stage:

    uv run --project python python examples/python_shell_interaction_control.py --launch --capture

The staged patches use the same actor-owned ``ui.shell.patch_layout`` state that native split,
tab, collapse, activation, and focus interactions commit. Recovery is invoked through the shared
``ui.commands.execute`` path. The workflow restores the previous shell before exiting.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import odon
from odon import layouts, ui


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "fixtures" / "synthetic_5ch.ome.zarr"
DEFAULT_EXECUTABLE = ROOT / "target" / "debug" / "odon"

PREFIX = "layout:workflow.review"
BODY_ID = f"{PREFIX}.body"
LEFT_COLLAPSIBLE_ID = f"{PREFIX}.left-collapsible"
LEFT_PANEL_ID = f"{PREFIX}.left"
LEFT_TABS_ID = f"{PREFIX}.left-tabs"
LAYERS_ID = f"{PREFIX}.left.native.0"
PROJECT_ID = f"{PREFIX}.left.native.1"
CANVAS_ID = f"{PREFIX}.canvas"
RECOVERY_ROOT_ID = "layout:recovery.single.root"
RECOVERY_CANVAS_ID = "layout:recovery.single.canvas"

BASELINE_RATIO = 0.24
RESIZED_RATIO = 0.36


def interaction_layout() -> ui.ShellLayout:
    """Add a collapsible left region to the nested review workflow."""

    base = layouts.review()
    nodes: list[ui.ShellLayoutNode] = []
    for node in base.nodes:
        if node.id == BODY_ID:
            node = replace(
                node,
                children=(LEFT_COLLAPSIBLE_ID, node.children[1]),
                split=ui.ShellSplit(
                    ratio=BASELINE_RATIO,
                    resizable=True,
                    axis="horizontal",
                ),
            )
        nodes.append(node)
    nodes.append(
        ui.ShellLayoutNode.collapsible(
            LEFT_COLLAPSIBLE_ID,
            LEFT_PANEL_ID,
            title="Review panels",
        )
    )
    return ui.ShellLayout(base.root_id, tuple(nodes))


def plan_summary() -> dict[str, Any]:
    layout = interaction_layout()
    body = layout.node(BODY_ID)
    collapsible = layout.node(LEFT_COLLAPSIBLE_ID)
    tabs = layout.node(LEFT_TABS_ID)
    return {
        "dataset": DEFAULT_DATASET.name,
        "mode": "single",
        "layout_root": layout.root_id,
        "layout_nodes": len(layout.nodes),
        "split_node_id": body.id,
        "baseline_ratio": body.split.ratio if body.split is not None else None,
        "resized_ratio": RESIZED_RATIO,
        "collapsible_node_id": collapsible.id,
        "collapsible_child_id": collapsible.children[0],
        "tabs_node_id": tabs.id,
        "baseline_selected_id": tabs.selected_id,
        "resized_selected_id": PROJECT_ID,
        "focused_node_id": PROJECT_ID,
        "recovery_command_id": "app.shell.recover",
        "recovery_root_id": RECOVERY_ROOT_ID,
    }


def _node(snapshot: ui.ShellSnapshot, node_id: str) -> ui.ShellLayoutNode:
    if snapshot.layout is None:
        raise RuntimeError("Odon did not publish the recursive desired layout")
    return snapshot.layout.node(node_id)


def _split_ratio(snapshot: ui.ShellSnapshot) -> float:
    split = _node(snapshot, BODY_ID).split
    if split is None:
        raise RuntimeError(f"shell node {BODY_ID!r} did not publish split state")
    return split.ratio


def _stage_state(stage: str, snapshot: ui.ShellSnapshot) -> dict[str, Any]:
    result: dict[str, Any] = {
        "stage": stage,
        "shell_revision": snapshot.revision,
        "layout_root": snapshot.layout.root_id if snapshot.layout is not None else None,
        "active_region_id": snapshot.active_region_id,
        "focused_node_id": snapshot.focused_node_id,
    }
    if snapshot.layout is not None and BODY_ID in {node.id for node in snapshot.layout.nodes}:
        result.update(
            {
                "split_ratio": _split_ratio(snapshot),
                "left_collapsed": _node(snapshot, LEFT_COLLAPSIBLE_ID).collapsed,
                "left_selected_id": _node(snapshot, LEFT_TABS_ID).selected_id,
            }
        )
    return result


def _print_stage(stage: str, snapshot: ui.ShellSnapshot, capture: bool) -> None:
    print(json.dumps(_stage_state(stage, snapshot), indent=2, sort_keys=True), flush=True)
    if capture:
        print(f"ODON_SHELL_INTERACTION_CAPTURE_STAGE={stage}", flush=True)
        input()


def _wait_for_recovery(
    app: odon.Client, *, after_revision: int, timeout: float = 5.0
) -> ui.ShellSnapshot:
    deadline = time.monotonic() + timeout
    latest = app.ui.shell.get(mode="single")
    while time.monotonic() < deadline:
        if (
            latest.revision > after_revision
            and latest.layout is not None
            and latest.layout.root_id == RECOVERY_ROOT_ID
        ):
            return latest
        time.sleep(0.05)
        latest = app.ui.shell.get(mode="single")
    raise TimeoutError(
        "protected recovery layout did not replace the single-view shell "
        f"after revision {after_revision}; latest revision={latest.revision}"
    )


def run_live(*, launch: bool, executable: Path, dataset: Path, capture: bool) -> None:
    if not dataset.exists():
        raise FileNotFoundError(dataset)
    if launch and not executable.exists():
        raise FileNotFoundError(executable)

    app = (
        odon.launch(executable, timeout=30.0)
        if launch
        else odon.connect(client_name="python-shell-interaction")
    )
    launched_process = app.launched_process
    original_layout: ui.ShellLayoutDocument | None = None
    try:
        app.datasets.open_ome_zarr(dataset).wait(timeout=120.0)
        original_layout = app.ui.shell.export_layout(mode="single")

        current = app.ui.shell.get(mode="single")
        baseline = app.ui.shell.replace_layout(
            interaction_layout(),
            mode="single",
            if_revision=current.revision,
            transaction_id="python-shell-interaction-baseline",
        )
        app.channels.set_visible(["DAPI", "CD3"], mode="only")
        app.viewer.fit()
        _print_stage("baseline", baseline, capture)

        resized = app.ui.shell.patch_layout(
            splits={
                BODY_ID: ui.ShellSplit(
                    ratio=RESIZED_RATIO,
                    resizable=True,
                    axis="horizontal",
                )
            },
            selected={LEFT_TABS_ID: PROJECT_ID},
            active_region_id=PROJECT_ID,
            focused_node_id=PROJECT_ID,
            mode="single",
            if_revision=baseline.revision,
            transaction_id="python-shell-interaction-resize-select-focus",
        )
        _print_stage("resized-selected-focused", resized, capture)

        collapsed = app.ui.shell.patch_layout(
            collapsed={LEFT_COLLAPSIBLE_ID: True},
            active_region_id=CANVAS_ID,
            focused_node_id=CANVAS_ID,
            mode="single",
            if_revision=resized.revision,
            transaction_id="python-shell-interaction-collapse",
        )
        _print_stage("collapsed", collapsed, capture)

        execution = app.ui.commands.execute("app.shell.recover")
        recovered = _wait_for_recovery(app, after_revision=collapsed.revision)
        recovery_canvas = _node(recovered, RECOVERY_CANVAS_ID)
        print(
            json.dumps(
                {
                    **_stage_state("recovered", recovered),
                    "command_id": "app.shell.recover",
                    "handler_type": execution.get("handler_type"),
                    "recovery_canvas_mount": recovery_canvas.mount,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        if capture:
            print("ODON_SHELL_INTERACTION_CAPTURE_STAGE=recovered", flush=True)
            input()
    finally:
        try:
            if original_layout is not None:
                current = app.ui.shell.get(mode="single")
                app.ui.shell.import_layout(
                    original_layout,
                    mode="single",
                    if_revision=current.revision,
                    transaction_id="python-shell-interaction-restore",
                )
        finally:
            app.close()
            if launched_process is not None and launched_process.poll() is None:
                launched_process.terminate()
                launched_process.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    args = parser.parse_args()
    if args.plan_only:
        print(json.dumps(plan_summary(), indent=2, sort_keys=True))
        return
    run_live(
        launch=args.launch,
        executable=args.executable,
        dataset=args.dataset,
        capture=args.capture,
    )


if __name__ == "__main__":
    main()
