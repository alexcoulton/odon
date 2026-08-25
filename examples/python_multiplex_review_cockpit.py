"""Install an adaptive five-channel multiplex-review cockpit controlled by Python.

The workflow turns the checked-in synthetic OME-Zarr fixture into a small review
application: Python defines the right-hand panel, channel presets, command handlers,
native menu and toolbar entries, command-palette metadata, predicates, and event-driven
review state. Every change is restored when the example exits.

Inspect the declarative plan without Odon:

    uv run --project python python examples/python_multiplex_review_cockpit.py --plan-only

Launch the repository release build and keep the cockpit interactive:

    uv run --project python python examples/python_multiplex_review_cockpit.py --launch --serve
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import odon
from odon import layouts, ui


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "fixtures" / "synthetic_5ch.ome.zarr"
DEFAULT_EXECUTABLE = ROOT / "target" / "release" / "odon"
EXTENSION_ID = "org.odon.multiplex_review"
EXTENSION_VERSION = "1.0.0"
CONTRIBUTION_ID = "review-cockpit"
EXPECTED_SHELL_MOUNT = f"extension:{EXTENSION_ID}/{CONTRIBUTION_ID}"
PROFILE_NAME = "Multiplex review cockpit"
TOOLBAR_NODE_ID = "layout:workflow.review.command-toolbar"
RIGHT_TABS_ID = "layout:workflow.review.right-tabs"
COCKPIT_NODE_ID = "layout:workflow.review.extension.0"

SESSION_CAPABILITIES = (
    "application.open",
    "events.read",
    "ui.shell.application_control",
    "ui.shell.chrome",
    "ui.shell.compose",
    "ui.shell.extension_place",
    "ui.shell.persistence",
    "ui.shell.read",
    "ui.shell.recovery",
    "ui.shell.shortcuts",
    "viewer.channels.read",
    "viewer.channels.write",
    "viewer.read",
    "viewer.write",
)


def command_id(local_id: str) -> str:
    return f"extension:{EXTENSION_ID}/{local_id}"


COMMANDS: tuple[dict[str, Any], ...] = (
    {
        "id": "overview",
        "title": "Multiplex Overview",
        "description": "Show epithelial, stromal, and nuclear context.",
        "event": "overview",
    },
    {
        "id": "nuclear-qc",
        "title": "Nuclear QC",
        "description": "Show DAPI and Ki67 for nuclear quality review.",
        "event": "nuclear-qc",
    },
    {
        "id": "immune-context",
        "title": "Immune Context",
        "description": "Show DAPI and CD3 for immune-cell context.",
        "event": "immune-context",
    },
    {
        "id": "stromal-context",
        "title": "Stromal Context",
        "description": "Show DAPI and Collagen for stromal context.",
        "event": "stromal-context",
    },
    {
        "id": "flag-view",
        "title": "Flag Current View",
        "description": "Add the current camera and channel state to the review log.",
        "event": "flag-view",
    },
    {
        "id": "inspect-selection",
        "title": "Inspect Selected Objects",
        "description": "Summarise the currently selected segmented objects.",
        "event": "inspect-selection",
        "kind": "objects",
    },
    {
        "id": "export-review",
        "title": "Export Review Package",
        "description": "Export review notes with the current mask context.",
        "event": "export-review",
        "kind": "masks",
    },
)


PRESETS: Mapping[str, Mapping[str, Any]] = {
    "overview": {
        "label": "Multiplex overview",
        "visible": ("DAPI", "PanCK", "Collagen"),
        "active": "PanCK",
        "colors": {
            "DAPI": (70, 105, 255),
            "PanCK": (255, 70, 175),
            "Collagen": (70, 225, 135),
        },
        "contrast": {"DAPI": (0, 60000), "PanCK": (0, 42000), "Collagen": (0, 36000)},
    },
    "nuclear-qc": {
        "label": "Nuclear QC",
        "visible": ("DAPI", "Ki67"),
        "active": "Ki67",
        "colors": {"DAPI": (55, 95, 255), "Ki67": (255, 210, 40)},
        "contrast": {"DAPI": (0, 60000), "Ki67": (0, 50000)},
    },
    "immune-context": {
        "label": "Immune context",
        "visible": ("DAPI", "CD3"),
        "active": "CD3",
        "colors": {"DAPI": (55, 95, 255), "CD3": (20, 235, 235)},
        "contrast": {"DAPI": (0, 60000), "CD3": (0, 42000)},
    },
    "stromal-context": {
        "label": "Stromal context",
        "visible": ("DAPI", "Collagen"),
        "active": "Collagen",
        "colors": {"DAPI": (55, 95, 255), "Collagen": (60, 235, 120)},
        "contrast": {"DAPI": (0, 60000), "Collagen": (0, 36000)},
    },
}


def _dataset_predicates() -> ui.CommandPredicates:
    return ui.CommandPredicates.predicates(
        visible=ui.CommandPredicate.capability(
            "viewer.read", reason="Viewer access is required."
        ),
        enabled=ui.CommandPredicate.state(
            "resources.dataset", reason="Open a dataset to use this action."
        ),
    )


def command_predicates(kind: str | None = None) -> ui.CommandPredicates:
    if kind == "objects":
        return ui.CommandPredicates.predicates(
            visible=ui.CommandPredicate.state(
                "resources.objects", reason="Load an object table to show this action."
            ),
            enabled=ui.CommandPredicate.state(
                "selection.objects.count",
                operator="greater_than",
                value=0,
                reason="Select at least one object.",
            ),
        )
    if kind == "masks":
        return ui.CommandPredicates.predicates(
            visible=ui.CommandPredicate.capability(
                "viewer.read", reason="Viewer access is required."
            ),
            enabled=ui.CommandPredicate.state(
                "resources.masks", reason="Create or load a mask before exporting."
            ),
        )
    return _dataset_predicates()


def review_panel(*, smooth_pixels: bool = True) -> ui.Panel:
    """Build the Python-authored native panel used by the live workflow."""

    inspect = ui.Button(
        "inspect-selection",
        "Inspect selected objects",
        action=ui.command(
            "ui.commands.execute", {"command_id": command_id("inspect-selection")}
        ),
    ).when(
        visible=ui.command_state(command_id("inspect-selection"), state="visible"),
        enabled=ui.command_state(command_id("inspect-selection"), state="enabled"),
    )
    export = ui.Button(
        "export-review",
        "Export review package",
        action=ui.command(
            "ui.commands.execute", {"command_id": command_id("export-review")}
        ),
    ).when(enabled=ui.command_state(command_id("export-review"), state="enabled"))

    return ui.Panel(
        "multiplex-review-cockpit",
        title="Multiplex review",
        children=[
            ui.Markdown(
                "introduction",
                "**Adaptive five-channel review**\n\n"
                "Python owns this workflow while Odon renders and evaluates it natively.",
            ),
            ui.Status("stage", "Ready · Multiplex overview"),
            ui.Select(
                "preset",
                "Channel preset",
                options=list(PRESETS),
                value="overview",
                action=ui.emit("preset-selected"),
                event_policy=ui.Immediate(),
            ),
            ui.Grid(
                "preset-actions",
                columns=2,
                children=[
                    ui.Button(
                        "nuclear-qc",
                        "Nuclear QC",
                        action=ui.command(
                            "ui.commands.execute", {"command_id": command_id("nuclear-qc")}
                        ),
                    ),
                    ui.Button(
                        "immune-context",
                        "Immune context",
                        action=ui.command(
                            "ui.commands.execute",
                            {"command_id": command_id("immune-context")},
                        ),
                    ),
                ],
            ),
            ui.Toggle(
                "smooth-pixels",
                "Smooth pixels",
                value=smooth_pixels,
                action=ui.bind("viewer", property="smooth_pixels"),
                event_policy=ui.Immediate(),
            ),
            ui.Button("fit-view", "Fit image", action=ui.command("viewer.camera.fit")),
            ui.Separator("review-separator"),
            ui.TextInput(
                "notes",
                "Review note",
                action=ui.emit("notes-changed"),
                event_policy=ui.OnCommit(),
            ),
            ui.Button(
                "flag-view",
                "Flag current view",
                action=ui.command(
                    "ui.commands.execute", {"command_id": command_id("flag-view")}
                ),
            ),
            inspect,
            export,
            ui.Progress("progress", value=0.0, label="Review progress"),
            ui.Status("status", "No regions flagged yet"),
            ui.Markdown("summary", "Review log is empty."),
        ],
    )


def shell_layout(panel_mount: str = EXPECTED_SHELL_MOUNT) -> ui.ShellLayout:
    base = layouts.review(panel_mounts=(panel_mount,))
    nodes: list[ui.ShellLayoutNode] = []
    for node in base.nodes:
        if node.id == "layout:workflow.review.top":
            node = replace(node, children=(*node.children, TOOLBAR_NODE_ID))
        elif node.id == RIGHT_TABS_ID:
            node = replace(node, selected_id=COCKPIT_NODE_ID)
        elif node.id == COCKPIT_NODE_ID:
            node = replace(node, title="Multiplex review")
        nodes.append(node)
    nodes.append(
        ui.ShellLayoutNode.builtin(
            TOOLBAR_NODE_ID,
            ui.ShellMountId.COMMAND_TOOLBAR,
            title="Multiplex review commands",
        )
    )
    return ui.ShellLayout(base.root_id, tuple(nodes))


def command_toolbar() -> ui.CommandToolbar:
    def item(local_id: str, label: str) -> ui.CommandToolbarItem:
        return ui.CommandToolbarItem.command(
            f"toolbar-item:multiplex-{local_id}",
            command_id(local_id),
            label=label,
            tooltip=next(
                command["description"]
                for command in COMMANDS
                if command["id"] == local_id
            ),
        )

    return ui.CommandToolbar.toolbar(
        "toolbar:multiplex-review",
        (
            ui.CommandToolbarGroup.group(
                "toolbar-group:multiplex-presets",
                (
                    item("overview", "Overview"),
                    item("nuclear-qc", "Nuclear QC"),
                    item("immune-context", "Immune"),
                    item("stromal-context", "Stroma"),
                ),
                title="Channels",
            ),
            ui.CommandToolbarGroup.group(
                "toolbar-group:multiplex-review",
                (
                    item("inspect-selection", "Inspect objects"),
                    item("flag-view", "Flag view"),
                    item("export-review", "Export review"),
                ),
                title="Review",
            ),
        ),
    )


def review_menu(original: ui.CommandMenuNode | None = None) -> ui.CommandMenuNode:
    review = ui.CommandMenuNode.menu(
        "menu:multiplex-review",
        "Review",
        (
            ui.CommandMenuNode.command("menu-item:multiplex-overview", command_id("overview")),
            ui.CommandMenuNode.command("menu-item:multiplex-nuclear", command_id("nuclear-qc")),
            ui.CommandMenuNode.command("menu-item:multiplex-immune", command_id("immune-context")),
            ui.CommandMenuNode.command(
                "menu-item:multiplex-stroma", command_id("stromal-context")
            ),
            ui.CommandMenuNode.separator("menu-separator:multiplex-review"),
            ui.CommandMenuNode.command(
                "menu-item:multiplex-inspect", command_id("inspect-selection")
            ),
            ui.CommandMenuNode.command("menu-item:multiplex-flag", command_id("flag-view")),
            ui.CommandMenuNode.command("menu-item:multiplex-export", command_id("export-review")),
            ui.CommandMenuNode.separator("menu-separator:multiplex-native"),
            ui.CommandMenuNode.command(
                "menu-item:multiplex-scale-bar", "viewer.scale_bar.toggle", label="Show scale bar"
            ),
        ),
    )
    children = () if original is None else original.children
    root_id = "menu-bar:multiplex-review" if original is None else original.id
    return ui.CommandMenuNode.menu_bar(root_id, (review, *children))


def command_palette() -> ui.CommandPalette:
    return ui.CommandPalette.palette(
        "palette:multiplex-review",
        title="Multiplex review commands",
        placeholder="Search channels, review actions, and Odon commands…",
        shortcut={"key": "p", "modifiers": ["primary", "shift"]},
        show_descriptions=True,
        max_results=30,
    )


def _component_ids(component: ui.Component) -> list[str]:
    return [
        component.id,
        *(
            child_id
            for child in component.children
            for child_id in _component_ids(child)
        ),
    ]


def plan_summary() -> dict[str, Any]:
    layout = shell_layout()
    toolbar = command_toolbar()
    menu = review_menu()
    panel = review_panel()
    inspect = next(child for child in panel.children if child.id == "inspect-selection")
    export = next(child for child in panel.children if child.id == "export-review")
    return {
        "dataset": DEFAULT_DATASET.name,
        "extension_id": EXTENSION_ID,
        "layout_root": layout.root_id,
        "layout_nodes": len(layout.nodes),
        "cockpit_node_id": COCKPIT_NODE_ID,
        "cockpit_mount": layout.node(COCKPIT_NODE_ID).mount,
        "cockpit_title": layout.node(COCKPIT_NODE_ID).title,
        "selected_right_tab": layout.node(RIGHT_TABS_ID).selected_id,
        "profile": {"name": PROFILE_NAME, "scope": "session"},
        "presets": {name: list(spec["visible"]) for name, spec in PRESETS.items()},
        "command_ids": [command_id(command["id"]) for command in COMMANDS],
        "toolbar_commands": [item.command_id for group in toolbar.groups for item in group.items],
        "menu_commands": [
            child.command_id
            for child in menu.children[0].children
            if child.type == "command"
        ],
        "palette": command_palette().to_dict(),
        "panel_component_ids": _component_ids(panel),
        "inspect_bindings": inspect.state_bindings,
        "export_bindings": export.state_bindings,
    }


class ReviewController:
    """Own command/event handlers and patch the native panel with review state."""

    def __init__(self, app: odon.Client, contribution: ui.Contribution) -> None:
        self.app = app
        self.contribution = contribution
        self.notes = ""
        self.current_preset = "overview"
        self.flags: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._condition = threading.Condition(threading.RLock())

    def subscribe(self) -> None:
        self.app.events.subscribe(f"ui.extension:{EXTENSION_ID}.*", self.handle_event)

    def close(self) -> None:
        try:
            if not self.app.closed:
                self.app.events.unsubscribe(f"ui.extension:{EXTENSION_ID}.*")
        finally:
            self.app.events.remove_callback(self.handle_event)

    def _patch(self, **values: Any) -> None:
        self.contribution.patch_values(values)

    def _summary(self) -> str:
        if not self.flags:
            return "Review log is empty."
        latest = self.flags[-1]
        channels = ", ".join(latest["visible_channels"])
        return (
            f"**{len(self.flags)} flagged view(s)**  \n"
            f"Latest: {channels}  \n{latest['note'] or '_No note_'}"
        )

    def apply_preset(self, preset_id: str) -> None:
        preset = PRESETS[preset_id]
        self.app.channels.set_visible(preset["visible"], mode="only")
        for channel, color in preset["colors"].items():
            self.app.channels.set_color(channel, color)
        for channel, limits in preset["contrast"].items():
            self.app.channels.set_contrast(channel, minimum=limits[0], maximum=limits[1])
        self.app.channels.set_active(preset["active"])
        self.app.viewer.fit()
        self.current_preset = preset_id
        self._patch(
            preset=preset_id,
            stage=f"Ready · {preset['label']}",
            status=f"Showing {', '.join(preset['visible'])}",
        )

    def flag_view(self) -> None:
        visible = self.app.channels.list_visible()
        if isinstance(visible, Mapping):
            raw_channels = visible.get("channels", visible.get("visible", []))
        else:
            raw_channels = visible
        names = [
            str(item.get("name", item.get("label", item)))
            if isinstance(item, Mapping)
            else str(item)
            for item in (raw_channels or [])
        ]
        self.flags.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "preset": self.current_preset,
                "visible_channels": names,
                "camera": self.app.viewer.get_camera(),
                "note": self.notes,
            }
        )
        self._patch(
            progress=min(len(self.flags) / 5.0, 1.0),
            status=f"Flagged view {len(self.flags)}",
            summary=self._summary(),
        )

    def handle_event(self, event: odon.Event) -> None:
        data = dict(event.data) if isinstance(event.data, Mapping) else {}
        record = {"name": event.name, "source": event.source, "data": data}
        try:
            suffix = event.name.removeprefix(f"ui.extension:{EXTENSION_ID}.")
            if suffix in PRESETS:
                self.apply_preset(suffix)
            elif suffix == "flag-view":
                self.flag_view()
            elif suffix == "inspect-selection":
                self._patch(status="Selected objects are ready for Python analysis")
            elif suffix == "export-review":
                self._patch(status=f"Review package ready · {len(self.flags)} flagged view(s)")
            elif suffix == "input":
                action = data.get("action")
                action = dict(action) if isinstance(action, Mapping) else {}
                semantic_event = action.get("event")
                if semantic_event == "preset-selected" and data.get("value") in PRESETS:
                    self.apply_preset(str(data["value"]))
                elif semantic_event == "notes-changed":
                    self.notes = str(data.get("value") or "")
                    self._patch(status="Review note committed")
        except Exception as error:  # surfaced in the panel and the smoke result
            self.errors.append(f"{type(error).__name__}: {error}")
            try:
                self._patch(status="Python handler failed", summary=self.errors[-1])
            except Exception:
                pass
        finally:
            with self._condition:
                self.events.append(record)
                self._condition.notify_all()

    def wait_for_command(self, local_id: str, previous_count: int, timeout: float = 10.0) -> None:
        expected = f"ui.extension:{EXTENSION_ID}.{local_id}"
        deadline = time.monotonic() + timeout
        with self._condition:
            while sum(item["name"] == expected for item in self.events) <= previous_count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"did not observe {expected}")
                self._condition.wait(remaining)
        if self.errors:
            raise RuntimeError(self.errors[-1])


def _connect(*, launch: bool, executable: Path) -> tuple[odon.Client, Any | None]:
    if launch:
        bootstrap = odon.launch(executable, timeout=30.0)
        process = bootstrap.launched_process
        instance_id = bootstrap.hello.instance_id
        bootstrap.close()
        return (
            odon.connect(
                instance=instance_id,
                timeout=15.0,
                client_name="multiplex-review-cockpit",
                requested_capabilities=SESSION_CAPABILITIES,
            ),
            process,
        )
    return (
        odon.connect(
            client_name="multiplex-review-cockpit",
            requested_capabilities=SESSION_CAPABILITIES,
        ),
        None,
    )


def _register_commands(extension: ui.Extension) -> None:
    for command in COMMANDS:
        extension.register_command(
            command["id"],
            command["title"],
            command["description"],
            command["event"],
            modes=("single",),
            predicates=command_predicates(command.get("kind")),
        )


def _capability_evidence(app: odon.Client) -> dict[str, Any]:
    restricted = odon.connect(
        instance=app.hello.instance_id,
        client_name="multiplex-review-restricted-proof",
        requested_capabilities=("ui.shell.read",),
    )
    try:
        command = next(
            item
            for item in restricted.ui.commands.list()
            if item.id == command_id("overview")
        )
        evidence: dict[str, Any] = {
            "visible": command.state.get("visible"),
            "enabled": command.state.get("enabled"),
            "missing_capabilities": command.state.get("missing_capabilities", []),
        }
        try:
            restricted.ui.commands.execute(command)
        except odon.RemoteError as error:
            evidence["denial"] = {
                "kind": error.kind,
                "required_state": error.data.get("state"),
            }
        else:
            raise RuntimeError("restricted command execution unexpectedly succeeded")
        return evidence
    finally:
        restricted.close()


def run_live(
    *, launch: bool, executable: Path, dataset: Path, serve: bool, smoke: bool
) -> None:
    if not dataset.exists():
        raise FileNotFoundError(dataset)
    if launch and not executable.exists():
        raise FileNotFoundError(executable)

    app, launched_process = _connect(launch=launch, executable=executable)
    original_layout: ui.ShellLayoutDocument | None = None
    original_menu: ui.CommandMenuSnapshot | None = None
    original_toolbar: ui.CommandToolbarSnapshot | None = None
    original_palette: ui.CommandPaletteSnapshot | None = None
    extension: ui.Extension | None = None
    controller: ReviewController | None = None
    profile_saved = False
    try:
        opened = app.datasets.open_ome_zarr(dataset).wait(timeout=120.0)
        original_layout = app.ui.shell.export_layout(mode="single")
        original_menu = app.ui.menus.get()
        original_toolbar = app.ui.toolbars.get()
        original_palette = app.ui.palette.get()
        smooth = bool(app.viewer.get_smooth_pixels().get("smooth", True))

        extension = app.ui.register_extension(
            id=EXTENSION_ID,
            name="Multiplex review cockpit",
            version=EXTENSION_VERSION,
            capabilities=("ui.panels", "ui.actions", "viewer.read", "viewer.write"),
            disconnect_policy="retain",
        )
        _register_commands(extension)
        contribution = extension.register(
            review_panel(smooth_pixels=smooth),
            location="right.tabs",
            contribution_id=CONTRIBUTION_ID,
        )
        if contribution.shell_mount != EXPECTED_SHELL_MOUNT:
            raise RuntimeError(f"unexpected shell mount {contribution.shell_mount!r}")

        controller = ReviewController(app, contribution)
        controller.subscribe()

        current_menu = app.ui.menus.get()
        app.ui.menus.replace(
            review_menu(original_menu.menu),
            if_revision=current_menu.revision,
            transaction_id="multiplex-review-menu",
        )
        current_toolbar = app.ui.toolbars.get()
        app.ui.toolbars.replace(
            command_toolbar(),
            if_revision=current_toolbar.revision,
            transaction_id="multiplex-review-toolbar",
        )
        current_palette = app.ui.palette.get()
        app.ui.palette.replace(
            command_palette(),
            if_revision=current_palette.revision,
            transaction_id="multiplex-review-palette",
        )
        current = app.ui.shell.get(mode="single")
        installed = app.ui.shell.replace_layout(
            shell_layout(contribution.shell_mount),
            mode="single",
            if_revision=current.revision,
            transaction_id="multiplex-review-layout",
        )
        app.ui.shell.save_profile(PROFILE_NAME, scope="session", mode="single")
        profile_saved = True

        controller.apply_preset("overview")
        command_states = {
            item.id: dict(item.state)
            for item in app.ui.commands.list()
            if item.id.startswith(f"extension:{EXTENSION_ID}/")
        }
        evidence = _capability_evidence(app)

        exercised: list[str] = []
        if smoke:
            for local_id in ("nuclear-qc", "immune-context", "flag-view"):
                event_name = f"ui.extension:{EXTENSION_ID}.{local_id}"
                previous = sum(item["name"] == event_name for item in controller.events)
                app.ui.commands.execute(command_id(local_id))
                controller.wait_for_command(local_id, previous)
                exercised.append(local_id)

        print(
            json.dumps(
                {
                    **plan_summary(),
                    "opened": bool(opened),
                    "instance_id": app.hello.instance_id,
                    "shell_revision": installed.revision,
                    "command_states": command_states,
                    "restricted_session": evidence,
                    "exercised": exercised,
                    "event_count": len(controller.events),
                    "flagged_views": len(controller.flags),
                    "interactive": serve,
                },
                indent=2,
                sort_keys=True,
                default=str,
            ),
            flush=True,
        )

        if serve:
            print(
                "\nCockpit ready. Use the Review menu, toolbar, Shift+Cmd/Ctrl+P, "
                "or the Python panel. Press Ctrl+C here to restore Odon.",
                flush=True,
            )
            while True:
                time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if controller is not None:
                controller.close()
            if not app.closed:
                if profile_saved:
                    app.ui.shell.remove_profile(PROFILE_NAME, scope="session")
                if original_menu is not None:
                    current_menu = app.ui.menus.get()
                    app.ui.menus.replace(
                        original_menu.menu, if_revision=current_menu.revision
                    )
                if original_toolbar is not None:
                    current_toolbar = app.ui.toolbars.get()
                    app.ui.toolbars.replace(
                        original_toolbar.toolbar, if_revision=current_toolbar.revision
                    )
                if original_palette is not None:
                    current_palette = app.ui.palette.get()
                    app.ui.palette.replace(
                        original_palette.palette, if_revision=current_palette.revision
                    )
                if original_layout is not None:
                    current = app.ui.shell.get(mode="single")
                    app.ui.shell.import_layout(
                        original_layout,
                        mode="single",
                        if_revision=current.revision,
                        transaction_id="multiplex-review-restore",
                    )
                if extension is not None:
                    extension.remove()
        finally:
            app.close()
            if launched_process is not None and launched_process.poll() is None:
                launched_process.terminate()
                launched_process.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument(
        "--no-smoke",
        action="store_true",
        help="Do not invoke the deterministic command sequence after installation.",
    )
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
        serve=args.serve,
        smoke=not args.no_smoke,
    )


if __name__ == "__main__":
    main()
