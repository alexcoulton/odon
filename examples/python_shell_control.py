"""Install and exercise a Python-defined Odon application shell.

The checked-in five-channel OME-Zarr fixture is used by default. Run Odon separately and connect:

    uv run --project python python examples/python_shell_control.py

Or let the example launch the repository's debug binary:

    uv run --project python python examples/python_shell_control.py --launch

Use ``--plan-only`` to validate and print the declarative layout and toolbar without a running app.
The live workflow restores the previous layout, toolbar, and scale-bar state before exiting.
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
TOOLBAR_NODE_ID = "layout:workflow.review.command-toolbar"
RIGHT_PANEL_ID = "layout:workflow.review.right"


def shell_layout() -> ui.ShellLayout:
    """Build a nested review shell with command-state-driven right-panel visibility."""

    base = layouts.review()
    nodes: list[ui.ShellLayoutNode] = []
    for node in base.nodes:
        if node.id == "layout:workflow.review.top":
            node = replace(node, children=(*node.children, TOOLBAR_NODE_ID))
        elif node.id == RIGHT_PANEL_ID:
            node = replace(
                node,
                state_bindings={
                    "visible": ui.command_state(
                        "viewer.scale_bar.toggle", state="checked"
                    )
                },
            )
        nodes.append(node)
    nodes.append(
        ui.ShellLayoutNode.builtin(
            TOOLBAR_NODE_ID,
            ui.ShellMountId.COMMAND_TOOLBAR,
            title="Review commands",
        )
    )
    return ui.ShellLayout(base.root_id, tuple(nodes))


def command_toolbar() -> ui.CommandToolbar:
    """Build a toolbar mixing ready, checked, disabled, and protected commands."""

    return ui.CommandToolbar.toolbar(
        "toolbar:python-shell-example",
        (
            ui.CommandToolbarGroup.group(
                "toolbar-group:project",
                (
                    ui.CommandToolbarItem.command(
                        "toolbar-item:save",
                        "project.save",
                        label="Save",
                        tooltip="Save the current Odon project.",
                    ),
                    ui.CommandToolbarItem.command(
                        "toolbar-item:recover",
                        "app.shell.recover",
                        label="Recover layout",
                        tooltip="Install Odon's protected recovery layout.",
                    ),
                ),
                title="Project",
            ),
            ui.CommandToolbarGroup.group(
                "toolbar-group:view",
                (
                    ui.CommandToolbarItem.command(
                        "toolbar-item:scale-bar",
                        "viewer.scale_bar.toggle",
                        label="Scale bar + inspector",
                        tooltip="Toggle the scale bar and the bound inspector region.",
                    ),
                    ui.CommandToolbarItem.command(
                        "toolbar-item:export-masks",
                        "viewer.masks.export_geojson",
                        label="Export masks",
                        tooltip="Disabled until a mask layer exists.",
                    ),
                ),
                title="View",
            ),
        ),
    )


def plan_summary() -> dict[str, Any]:
    layout = shell_layout()
    toolbar = command_toolbar()
    return {
        "dataset": DEFAULT_DATASET.name,
        "mode": "single",
        "layout_root": layout.root_id,
        "layout_nodes": len(layout.nodes),
        "toolbar_groups": len(toolbar.groups),
        "toolbar_commands": [
            item.command_id for group in toolbar.groups for item in group.items
        ],
        "binding": layout.node(RIGHT_PANEL_ID).state_bindings["visible"],
    }


def _wait_for_checked(app: odon.Client, expected: bool, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        command = next(
            item
            for item in app.ui.commands.list()
            if item.id == "viewer.scale_bar.toggle"
        )
        if command.state.get("checked") is expected:
            return
        time.sleep(0.05)
    raise TimeoutError(f"scale-bar command did not reconcile to checked={expected}")


def _desired_node(snapshot: ui.ShellSnapshot, node_id: str) -> ui.ShellLayoutNode:
    if snapshot.layout is None:
        raise RuntimeError("Odon did not publish the recursive desired layout")
    return snapshot.layout.node(node_id)


def run_live(*, launch: bool, executable: Path, dataset: Path, capture: bool) -> None:
    if not dataset.exists():
        raise FileNotFoundError(dataset)
    if launch and not executable.exists():
        raise FileNotFoundError(executable)

    app = (
        odon.launch(executable, timeout=30.0)
        if launch
        else odon.connect(client_name="python-shell-workflow")
    )
    launched_process = app.launched_process
    original_layout: ui.ShellLayoutDocument | None = None
    original_toolbar: ui.CommandToolbarSnapshot | None = None
    original_scale_bar: bool | None = None
    try:
        opened = app.datasets.open_ome_zarr(dataset).wait(timeout=120.0)
        original_layout = app.ui.shell.export_layout(mode="single")
        original_toolbar = app.ui.toolbars.get()
        original_scale_bar = bool(app.viewer.get_scale_bar()["visible"])

        app.viewer.set_scale_bar(True)
        toolbar = app.ui.toolbars.replace(
            command_toolbar(),
            if_revision=original_toolbar.revision,
            transaction_id="python-shell-example-toolbar",
        )
        current = app.ui.shell.get(mode="single")
        shell = app.ui.shell.replace_layout(
            shell_layout(),
            mode="single",
            if_revision=current.revision,
            transaction_id="python-shell-example-layout",
        )
        app.channels.set_visible(["DAPI", "CD3"], mode="only")
        app.viewer.fit()
        _wait_for_checked(app, True)

        output = {
            **plan_summary(),
            "opened": bool(opened),
            "shell_revision": shell.revision,
            "toolbar_revision": toolbar.revision,
            "active_region_id": shell.active_region_id,
            "right_panel_visible": _desired_node(shell, RIGHT_PANEL_ID).visible,
            "scale_bar_checked": True,
        }
        print(json.dumps(output, indent=2, sort_keys=True), flush=True)

        if capture:
            print("ODON_SHELL_CAPTURE_STAGE=expanded", flush=True)
            input()
            app.ui.commands.execute("viewer.scale_bar.toggle", checked=False)
            _wait_for_checked(app, False)
            collapsed = app.ui.shell.get(mode="single")
            print(
                json.dumps(
                    {
                        "shell_revision": collapsed.revision,
                        "right_panel_visible": _desired_node(
                            collapsed, RIGHT_PANEL_ID
                        ).visible,
                        "scale_bar_checked": False,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
            print("ODON_SHELL_CAPTURE_STAGE=bound-hidden", flush=True)
            input()
    finally:
        try:
            if original_scale_bar is not None:
                app.viewer.set_scale_bar(original_scale_bar)
            if original_layout is not None:
                current = app.ui.shell.get(mode="single")
                app.ui.shell.import_layout(
                    original_layout,
                    mode="single",
                    if_revision=current.revision,
                    transaction_id="python-shell-example-restore-layout",
                )
            if original_toolbar is not None:
                current_toolbar = app.ui.toolbars.get()
                app.ui.toolbars.replace(
                    original_toolbar.toolbar,
                    if_revision=current_toolbar.revision,
                    transaction_id="python-shell-example-restore-toolbar",
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
