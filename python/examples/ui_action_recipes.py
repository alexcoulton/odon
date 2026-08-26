#!/usr/bin/env python3
"""Runnable counterparts for the six native UI action recipes.

Run from the repository root, for example::

    uv run --project python python python/examples/ui_action_recipes.py \
      --objects /path/to/cells.geoparquet --property score

The script connects to one running Odon instance and installs a native panel.
It is intentionally small enough to copy from; production extensions should
also use ``odon.run_extension`` for reconnect lifecycle management.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import threading
from typing import Any, Sequence

import odon
from odon import ui


def long_running_button(
    extension: ui.Extension, contribution: ui.Contribution, app: odon.Client, path: Path
) -> None:
    """Recipe 1: a button may safely wait because its handler uses a serial worker."""

    @extension.on_action(
        "run-load",
        execution="serial-worker",
        coalesce="reject-while-busy",
        contribution=contribution,
        status_component_id="status",
        progress_component_id="progress",
    )
    def run(context: odon.ActionContext, _interaction: odon.UiInteraction) -> None:
        with context.busy("Loading objects…"):
            task = context.attach(app.objects.load(path))
            task.wait(timeout=60, progress=context.report_task)


def resource_load_from_interaction(
    extension: ui.Extension, contribution: ui.Contribution, app: odon.Client, path: Path
) -> None:
    """Recipe 2: latest-wins source selection waits outside callback delivery."""

    @extension.on_action(
        "reload-source",
        coalesce="latest",
        queue_key="source-selection",
        contribution=contribution,
        status_component_id="status",
    )
    def load(context: odon.ActionContext, _interaction: odon.UiInteraction) -> None:
        task = context.attach(app.objects.load(path))
        task.wait(timeout=60, progress=context.report_task)
        context.ensure_current()
        app.objects.get_state()  # inspect the semantic source after task completion
        odon.wait_for_viewer_readiness(app, timeout=30)
        context.result("Ready")


def previous_next_navigation(
    extension: ui.Extension,
    contribution: ui.Contribution,
    viewer: Any,
    channels: Sequence[str],
) -> None:
    """Recipe 3: opposite rapid clicks reduce to one accumulated delta."""

    current = [0]

    def move(context: odon.ActionContext, _interaction: odon.UiInteraction) -> None:
        current[0] = (current[0] + int(context.delta)) % len(channels)
        channel = channels[current[0]]
        viewer.set_visible_channels([channel], mode="only")
        viewer.set_active_channel(channel)
        context.patch({"channel": channel})

    extension.on_action(
        "previous-channel",
        move,
        queue_key="channel-navigation",
        coalesce="accumulate",
        delta=-1,
        contribution=contribution,
    )
    extension.on_action(
        "next-channel",
        move,
        queue_key="channel-navigation",
        coalesce="accumulate",
        delta=1,
        contribution=contribution,
    )


def latest_wins_select(
    extension: ui.Extension,
    contribution: ui.Contribution,
    app: odon.Client,
    viewer: Any,
) -> None:
    """Recipe 4: only the newest pending select value may commit."""

    @extension.on_action(
        "channel-selected",
        coalesce="latest",
        queue_key="channel-selection",
        contribution=contribution,
        status_component_id="status",
    )
    def select(context: odon.ActionContext, interaction: odon.UiInteraction) -> None:
        channel = str(interaction.value)
        context.status(f"Selecting {channel}…")

        def install_channel() -> None:
            viewer.set_visible_channels([channel], mode="only")
            viewer.set_active_channel(channel)

        if not context.commit(install_channel):
            context.ensure_current()
        odon.wait_for_viewer_readiness(app, timeout=30)
        context.ensure_current()
        context.patch({"channel": channel, "status": "Ready"})


def busy_status_error_panel(
    extension: ui.Extension, contribution: ui.Contribution, app: odon.Client, path: Path
) -> None:
    """Recipe 5: short panel errors coexist with retained structured exceptions."""

    diagnostics: list[BaseException] = []

    def failed(error: BaseException, context: odon.ActionContext | None) -> None:
        diagnostics.append(error)
        if context is not None:
            context.patch({"status": f"Failed: {type(error).__name__}"})

    extension.on_action(
        "checked-load",
        lambda context, _interaction: context.attach(app.objects.load(path)).wait(
            timeout=60, progress=context.report_task
        ),
        coalesce="reject-while-busy",
        contribution=contribution,
        status_component_id="status",
        progress_component_id="progress",
        on_error=failed,
    )


def safe_source_style_swap(
    extension: ui.Extension,
    contribution: ui.Contribution,
    app: odon.Client,
    viewer: Any,
    path: Path,
    property_name: str,
) -> None:
    """Recipe 6: never retain a property mapping across an incompatible source."""

    @extension.on_action(
        "safe-swap",
        coalesce="latest",
        queue_key="object-source-style",
        contribution=contribution,
        status_component_id="status",
        progress_component_id="progress",
    )
    def swap(context: odon.ActionContext, _interaction: odon.UiInteraction) -> None:
        odon.replace_object_source_and_style(
            app,
            str(path),
            property_name,
            domain="auto",
            context=context,
            presentation_objects=viewer.objects,
        )


def panel(channels: Sequence[str]) -> ui.Panel:
    return ui.Panel(
        "action-recipes",
        title="SDK action recipes",
        children=[
            ui.Select(
                "channel",
                "Channel",
                options=channels,
                value=channels[0],
                action=ui.emit("channel-selected"),
                event_policy=ui.Immediate(),
            ),
            ui.Grid(
                "navigation",
                columns=2,
                children=[
                    ui.Button("previous", "Previous", action=ui.emit("previous-channel")),
                    ui.Button("next", "Next", action=ui.emit("next-channel")),
                ],
            ),
            ui.Button("load", "Long load", action=ui.emit("run-load")),
            ui.Button("reload", "Reload source", action=ui.emit("reload-source")),
            ui.Button("checked", "Checked load", action=ui.emit("checked-load")),
            ui.Button("swap", "Safe source/style swap", action=ui.emit("safe-swap")),
            ui.Progress("progress", value=0),
            ui.Status("status", "Ready"),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", required=True, type=Path)
    parser.add_argument("--property", required=True)
    args = parser.parse_args()
    app = odon.connect(client_name="ui-action-recipes")
    extension = app.ui.register_extension(
        id="org.odon.examples.ui_action_recipes",
        name="SDK action recipes",
        version="1.0.0",
        capabilities=("ui.panels", "viewer.read", "viewer.write"),
    )
    workspace = app.viewer.workspace.get()
    viewer = app.viewer.viewports.handle(workspace["viewports"][0]["viewport_id"])
    listed = viewer.list_channels()
    raw_channels = listed.get("channels", listed.get("result", []))
    channels = tuple(str(item.get("name", item.get("label"))) for item in raw_channels)
    if not channels:
        raise RuntimeError("open an image with at least one channel first")
    contribution = extension.register(panel(channels), location="right.tabs")
    long_running_button(extension, contribution, app, args.objects)
    resource_load_from_interaction(extension, contribution, app, args.objects)
    previous_next_navigation(extension, contribution, viewer, channels)
    latest_wins_select(extension, contribution, app, viewer)
    busy_status_error_panel(extension, contribution, app, args.objects)
    safe_source_style_swap(
        extension, contribution, app, viewer, args.objects, args.property
    )
    stopped = threading.Event()
    print("Recipes installed in Odon; press Ctrl+C to remove them.")
    try:
        stopped.wait()
    except KeyboardInterrupt:
        pass
    finally:
        extension.remove()
        app.close()


if __name__ == "__main__":
    main()
