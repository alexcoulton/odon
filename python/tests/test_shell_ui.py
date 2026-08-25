from __future__ import annotations

import unittest
from typing import Any, Mapping

from odon.async_ui import AsyncUi
from odon.ui import (
    ApplicationCommand,
    Button,
    CommandMenuNode,
    CommandMenuSnapshot,
    CommandPalette,
    CommandPaletteSnapshot,
    CommandPredicate,
    CommandPredicates,
    CommandToolbar,
    CommandToolbarGroup,
    CommandToolbarItem,
    CommandToolbarSnapshot,
    ExtensionLayoutTemplate,
    ShellComponentDescriptor,
    ShellId,
    ShellLayout,
    ShellLayoutDocument,
    ShellLayoutProfile,
    ShellLayoutNode,
    ShellMountId,
    ShellSize,
    ShellSnapshot,
    Ui,
    command_state,
)


def command_surface_result(
    method: str, params: Mapping[str, Any] | None
) -> dict[str, Any]:
    commands = [
        {
            "id": "app.lifecycle.quit",
            "title": "Quit odon",
            "description": "Safely quit Odon.",
            "handler": {"type": "native", "action": "quit"},
            "availability": {"modes": ["project", "single", "mosaic"]},
            "protected": True,
            "shortcut": {"key": "q", "modifiers": ["primary"]},
            "icon": None,
            "predicates": {},
            "state": {
                "visible": True,
                "enabled": True,
                "checkable": False,
                "checked": None,
                "reasons": [],
                "missing_capabilities": [],
            },
        },
        {
            "id": "app.shell.recover",
            "title": "Recover Application Layout",
            "description": "Install the protected recovery layout.",
            "handler": {"type": "control", "method": "ui.shell.recover", "params": {}},
            "availability": {"modes": ["project", "single", "mosaic"]},
            "protected": True,
            "shortcut": None,
            "icon": None,
            "predicates": {},
            "state": {
                "visible": True,
                "enabled": True,
                "checkable": False,
                "checked": None,
                "reasons": [],
                "missing_capabilities": [],
            },
        },
    ]
    menu = (params or {}).get(
        "menu",
        {
            "id": "menu:root",
            "type": "menu_bar",
            "children": [
                {
                    "id": "menu:application",
                    "type": "menu",
                    "title": "odon",
                    "children": [
                        {
                            "id": "menu-item:quit",
                            "type": "command",
                            "command_id": "app.lifecycle.quit",
                        }
                    ],
                },
                {
                    "id": "menu:help",
                    "type": "menu",
                    "title": "Help",
                    "children": [
                        {
                            "id": "menu-item:recover",
                            "type": "command",
                            "command_id": "app.shell.recover",
                        }
                    ],
                },
            ],
        },
    )
    if method == "ui.commands.describe_schema":
        return {"schema_version": 1, "menu_node_types": ["menu_bar", "menu", "command", "separator"]}
    if method == "ui.commands.list":
        return {"schema_version": 1, "revision": 3, "commands": commands}
    if method == "ui.commands.register":
        spec = dict((params or {})["command"])
        command = {
            "id": f"extension:{(params or {})['extension_id']}/{spec['id']}",
            "title": spec["title"],
            "description": spec["description"],
            "handler": {"type": "event", "event": spec["event"]},
            "availability": {"modes": spec["modes"]},
            "protected": False,
            "shortcut": spec.get("shortcut"),
            "icon": spec.get("icon"),
            "predicates": spec.get("predicates", {}),
            "state": {
                "visible": True,
                "enabled": True,
                "checkable": "checked" in spec.get("predicates", {}),
                "checked": False
                if "checked" in spec.get("predicates", {})
                else None,
                "reasons": [],
                "missing_capabilities": [],
            },
            "ownership": {
                "scope": "extension",
                "owner_id": (params or {})["extension_id"],
                "owner_session_id": "session-test",
                "protected": False,
            },
            "readiness": {"state": "ready"},
            "disconnect_policy": "retain",
        }
        return {"schema_version": 1, "revision": 4, "command": command}
    if method == "ui.commands.remove":
        return {**dict(params or {}), "removed": True, "revision": 4}
    if method == "ui.commands.execute":
        return {**dict(params or {}), "dispatched": True}
    if method.startswith("ui.toolbars."):
        toolbar = (params or {}).get(
            "toolbar", {"id": "toolbar:main", "type": "toolbar", "groups": []}
        )
        return {
            "schema_version": 1,
            "revision": 4 if method == "ui.toolbars.replace" else 3,
            "toolbar": toolbar,
            **(
                {
                    "change": {
                        "operation": "replace",
                        "changed": True,
                        "transaction_id": (params or {}).get("transaction_id"),
                    }
                }
                if method == "ui.toolbars.replace"
                else {}
            ),
        }
    if method.startswith("ui.palette."):
        palette = (params or {}).get(
            "palette",
            CommandPalette.palette("palette:main").to_dict(),
        )
        return {
            "schema_version": 1,
            "revision": 4 if method == "ui.palette.replace" else 3,
            "palette": palette,
            **(
                {
                    "change": {
                        "operation": "replace",
                        "changed": True,
                        "transaction_id": (params or {}).get("transaction_id"),
                    }
                }
                if method == "ui.palette.replace"
                else {}
            ),
        }
    return {
        "schema_version": 1,
        "revision": 4 if method == "ui.menus.replace" else 3,
        "menu": menu,
        **(
            {
                "change": {
                    "operation": "replace",
                    "changed": True,
                    "transaction_id": (params or {}).get("transaction_id"),
                }
            }
            if method == "ui.menus.replace"
            else {}
        ),
    }


def component_catalog(params: Mapping[str, Any] | None) -> dict[str, Any]:
    mode = str((params or {}).get("mode", "single"))
    return {
        "schema_version": 1,
        "mode": mode,
        "components": [
            {
                "id": "builtin:viewer-canvas",
                "version": 1,
                "title": "Image viewport workspace",
                "kind": "canvas",
                "modes": ["single"],
                "readiness": ["model"],
                "legal_parent_types": ["row", "column", "split"],
                "singleton": True,
                "configuration_schema": {"type": "object"},
                "commands": ["viewer.viewports.camera.set"],
                "events": ["viewer.camera.changed"],
                "minimum_size": {"width": 256, "height": 256},
                "recommended_size": {"width": 1200, "height": 800},
                "persistence": "project",
                "ownership": {
                    "scope": "application",
                    "owner_id": "odon",
                    "owner_session_id": None,
                    "protected": True,
                },
            }
        ],
    }


def shell_result(method: str, params: Mapping[str, Any] | None) -> dict[str, Any]:
    mode = str((params or {}).get("mode", "project"))
    root_id = f"builtin:{mode}.root"
    result: dict[str, Any] = {
        "schema_version": 1,
        "revision": 4,
        "mode": mode,
        "root_id": root_id,
        "active_region_id": root_id,
        "focused_node_id": None,
        "nodes": [
            {
                "id": root_id,
                "type": "application",
                "parent_id": None,
                "content": None,
                "visible": True,
                "mutable": {"visibility": False, "order": False, "selection": False},
                "children": [],
                "selected_id": None,
                "ownership": {
                    "scope": "application",
                    "owner_id": "odon",
                    "protected": True,
                },
            }
        ],
    }
    if method in {
        "ui.shell.patch",
        "ui.shell.patch_layout",
        "ui.shell.reset",
        "ui.shell.replace_layout",
        "ui.shell.import_layout",
        "ui.shell.recover",
        "ui.shell.profiles.load",
    }:
        result["change"] = {
            "operation": (
                "load_profile"
                if method == "ui.shell.profiles.load"
                else method.removeprefix("ui.shell.")
            ),
            "mode": mode,
            "previous_revision": 4,
            "revision": 4,
            "changed": False,
            "transaction_id": (params or {}).get("transaction_id"),
            "changes": [],
        }
    return result


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        call = (method, dict(params or {}))
        self.calls.append(call)
        if method.startswith(("ui.commands.", "ui.menus.", "ui.toolbars.", "ui.palette.")):
            return command_surface_result(method, params)
        if method == "ui.shell.describe_schema":
            return {"schema_version": 1}
        if method == "ui.shell.components.list":
            return component_catalog(params)
        if method == "ui.shell.export_layout":
            return ShellLayoutDocument(
                str((params or {}).get("mode", "single")),
                ShellUiTests.single_layout(),
            ).to_dict()
        if method == "ui.shell.profiles.list":
            return {
                "schema_version": 1,
                "scope": str((params or {}).get("scope", "session")),
                "profiles": [
                    {
                        "name": "review",
                        "mode": "single",
                        "document_schema_version": 1,
                        "valid": True,
                        "error": None,
                        "startup_modes": ["single"],
                    }
                ],
            }
        if method in {"ui.shell.profiles.save", "ui.shell.profiles.remove"}:
            return dict(params or {})
        if method == "ui.extensions.register":
            return {
                **dict(params or {}),
                "granted_capabilities": list((params or {}).get("capabilities", [])),
                "revision": 10,
            }
        if method == "ui.extensions.set_readiness":
            return {
                "id": (params or {})["extension_id"],
                "ready": (params or {})["ready"],
                "readiness_reason": (params or {}).get("reason"),
            }
        if method == "ui.extensions.layouts.register":
            return {
                **dict(params or {}),
                "revision": 11,
                "ownership": {
                    "scope": "extension",
                    "owner_id": (params or {})["extension_id"],
                    "owner_session_id": "session-test",
                    "protected": False,
                },
            }
        if method == "ui.extensions.layouts.list":
            return {
                "extension_id": (params or {})["extension_id"],
                "layouts": [
                    {
                        "extension_id": (params or {})["extension_id"],
                        "name": "Review",
                        "document": ShellLayoutDocument(
                            "single", ShellUiTests.single_layout()
                        ).to_dict(),
                        "revision": 11,
                        "ownership": {
                            "scope": "extension",
                            "owner_id": (params or {})["extension_id"],
                            "owner_session_id": "session-test",
                            "protected": False,
                        },
                    }
                ],
                "revision": 11,
            }
        if method == "ui.extensions.layouts.remove":
            return {**dict(params or {}), "removed": True}
        return shell_result(method, params)


class AsyncRecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        call = (method, dict(params or {}))
        self.calls.append(call)
        if method.startswith(("ui.commands.", "ui.menus.", "ui.toolbars.", "ui.palette.")):
            return command_surface_result(method, params)
        if method == "ui.shell.describe_schema":
            return {"schema_version": 1}
        if method == "ui.shell.components.list":
            return component_catalog(params)
        if method == "ui.shell.export_layout":
            return ShellLayoutDocument(
                str((params or {}).get("mode", "single")),
                ShellUiTests.single_layout(),
            ).to_dict()
        if method == "ui.shell.profiles.list":
            return {
                "schema_version": 1,
                "scope": str((params or {}).get("scope", "session")),
                "profiles": [
                    {
                        "name": "review",
                        "mode": "single",
                        "document_schema_version": 1,
                        "valid": True,
                        "error": None,
                        "startup_modes": ["single"],
                    }
                ],
            }
        if method in {"ui.shell.profiles.save", "ui.shell.profiles.remove"}:
            return dict(params or {})
        if method == "ui.extensions.register":
            return {
                **dict(params or {}),
                "granted_capabilities": list((params or {}).get("capabilities", [])),
                "revision": 10,
            }
        if method == "ui.extensions.set_readiness":
            return {
                "id": (params or {})["extension_id"],
                "ready": (params or {})["ready"],
                "readiness_reason": (params or {}).get("reason"),
            }
        if method == "ui.extensions.layouts.register":
            return {
                **dict(params or {}),
                "revision": 11,
                "ownership": {
                    "scope": "extension",
                    "owner_id": (params or {})["extension_id"],
                    "owner_session_id": "session-test",
                    "protected": False,
                },
            }
        if method == "ui.extensions.layouts.list":
            return {
                "extension_id": (params or {})["extension_id"],
                "layouts": [
                    {
                        "extension_id": (params or {})["extension_id"],
                        "name": "Review",
                        "document": ShellLayoutDocument(
                            "single", ShellUiTests.single_layout()
                        ).to_dict(),
                        "revision": 11,
                        "ownership": {
                            "scope": "extension",
                            "owner_id": (params or {})["extension_id"],
                            "owner_session_id": "session-test",
                            "protected": False,
                        },
                    }
                ],
                "revision": 11,
            }
        if method == "ui.extensions.layouts.remove":
            return {**dict(params or {}), "removed": True}
        return shell_result(method, params)


class ShellUiTests(unittest.TestCase):
    def test_component_state_bindings_are_bounded_and_serialized(self) -> None:
        component = Button(
            "run",
            "Run",
            action={"type": "emit", "event": "run"},
        ).when(
            visible=command_state("extension:org.example.binding/run", state="visible"),
            enabled=command_state("extension:org.example.binding/run"),
        )
        self.assertEqual(
            component.to_dict()["state_bindings"],
            {
                "visible": {
                    "type": "command_state",
                    "command_id": "extension:org.example.binding/run",
                    "state": "visible",
                    "equals": True,
                },
                "enabled": {
                    "type": "command_state",
                    "command_id": "extension:org.example.binding/run",
                    "state": "enabled",
                    "equals": True,
                },
            },
        )
        self.assertNotIn(
            "state_bindings",
            Button("plain", "Plain", action={"type": "emit", "event": "plain"}).to_dict(),
        )
        with self.assertRaises(ValueError):
            command_state("project.save", state="private")
        with self.assertRaises(ValueError):
            command_state("project.save", equals=1)  # type: ignore[arg-type]

        shell_node = ShellLayoutNode.builtin(
            "layout:bound-help",
            ShellMountId.HELP,
            state_bindings={
                "visible": command_state("app.shell.recover", state="enabled")
            },
        )
        serialized = shell_node.to_dict()
        self.assertEqual(
            serialized["state_bindings"]["visible"]["command_id"],
            "app.shell.recover",
        )
        self.assertEqual(ShellLayoutNode.from_result(serialized), shell_node)
        with self.assertRaises(ValueError):
            ShellLayoutNode.builtin(
                "layout:invalid-binding",
                ShellMountId.HELP,
                state_bindings={
                    "enabled": command_state("app.shell.recover")
                },
            )

    @staticmethod
    def single_layout() -> ShellLayout:
        return ShellLayout(
            "layout:test.root",
            (
                ShellLayoutNode.application("layout:test.root", ["layout:test.body"]),
                ShellLayoutNode.row(
                    "layout:test.body",
                    ["layout:test.canvas", "layout:test.panel"],
                ),
                ShellLayoutNode.canvas(
                    "layout:test.canvas", "builtin:viewer-canvas"
                ),
                ShellLayoutNode.panel(
                    "layout:test.panel",
                    "layout:test.layers",
                    size=ShellSize(width=320),
                ),
                ShellLayoutNode.builtin("layout:test.layers", "builtin:layers"),
            ),
        )

    def test_commands_and_platform_menu_are_typed_separate_resources(self) -> None:
        client = RecordingClient()
        ui = Ui(client)  # type: ignore[arg-type]

        commands = ui.commands.list()
        snapshot = ui.menus.get()
        swapped = CommandMenuNode.menu_bar(
            snapshot.menu.id, reversed(snapshot.menu.children)
        )
        changed = ui.menus.replace(
            swapped, if_revision=snapshot.revision, transaction_id="menu-python-3"
        )

        self.assertIsInstance(commands[0], ApplicationCommand)
        self.assertTrue(commands[0].protected)
        self.assertIsInstance(snapshot, CommandMenuSnapshot)
        self.assertEqual(changed.menu.children[0].id, "menu:help")
        self.assertEqual(changed.change["transaction_id"], "menu-python-3")
        self.assertEqual(client.calls[-1][0], "ui.menus.replace")
        self.assertEqual(client.calls[-1][1]["if_command_revision"], 3)

    def test_command_menu_builders_reject_non_menu_bar_roots(self) -> None:
        ui = Ui(RecordingClient())  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "root must be a menu_bar"):
            ui.menus.replace(CommandMenuNode.separator("not-a-menu"))

    def test_command_toolbar_builders_share_the_command_revision(self) -> None:
        client = RecordingClient()
        ui = Ui(client)  # type: ignore[arg-type]
        current = ui.toolbars.get()
        toolbar = CommandToolbar.toolbar(
            "toolbar:analysis",
            (
                CommandToolbarGroup.group(
                    "toolbar-group:analysis",
                    (
                        CommandToolbarItem.command(
                            "toolbar-item:quit",
                            "app.lifecycle.quit",
                            label="Quit",
                            icon="⏻",
                            tooltip="Safely quit Odon.",
                            show_label=False,
                        ),
                    ),
                    title="Analysis",
                ),
            ),
        )
        changed = ui.toolbars.replace(
            toolbar,
            if_revision=current.revision,
            transaction_id="toolbar-python-3",
        )

        self.assertIsInstance(changed, CommandToolbarSnapshot)
        self.assertEqual(changed.toolbar.groups[0].items[0].command_id, "app.lifecycle.quit")
        self.assertEqual(changed.toolbar.groups[0].items[0].icon, "⏻")
        self.assertEqual(changed.toolbar.groups[0].items[0].tooltip, "Safely quit Odon.")
        self.assertFalse(changed.toolbar.groups[0].items[0].show_label)
        self.assertEqual(changed.change["transaction_id"], "toolbar-python-3")
        self.assertEqual(client.calls[-1][0], "ui.toolbars.replace")

    def test_command_palette_is_typed_revisioned_and_configurable(self) -> None:
        client = RecordingClient()
        ui = Ui(client)  # type: ignore[arg-type]
        current = ui.palette.get()
        changed = ui.palette.replace(
            CommandPalette.palette(
                "palette:review",
                title="Review commands",
                placeholder="Find an action…",
                shortcut={"key": "k", "modifiers": ["primary"]},
                show_descriptions=False,
                max_results=12,
            ),
            if_revision=current.revision,
            transaction_id="palette-python-3",
        )

        self.assertIsInstance(current, CommandPaletteSnapshot)
        self.assertEqual(changed.palette.shortcut["key"], "k")
        self.assertFalse(changed.palette.show_descriptions)
        self.assertEqual(changed.change["transaction_id"], "palette-python-3")
        self.assertEqual(client.calls[-1][0], "ui.palette.replace")
        self.assertEqual(client.calls[-1][1]["if_command_revision"], 3)

    def test_extension_commands_are_typed_owned_and_invokable(self) -> None:
        client = RecordingClient()
        ui = Ui(client)  # type: ignore[arg-type]
        extension = ui.register_extension(
            id="org.example.commands",
            name="Commands",
            version="1.0.0",
            capabilities=("ui.actions",),
            disconnect_policy="retain",
        )
        registered = extension.register_command(
            "measure.cells",
            "Measure Cells",
            "Measure selected cells.",
            "measure",
            modes=("single", "mosaic"),
            shortcut={"key": "m", "modifiers": ["primary", "shift"]},
            predicates=CommandPredicates.predicates(
                visible=CommandPredicate.capability("viewer.read"),
                enabled=CommandPredicate.all(
                    CommandPredicate.state("resources.objects"),
                    CommandPredicate.state(
                        "selection.objects.count",
                        operator="greater_than",
                        value=0,
                        reason="Select at least one object.",
                    ),
                ),
                checked=CommandPredicate.state(
                    "presentation.left_panel.visible"
                ),
            ),
            if_revision=3,
            transaction_id="python-command-3",
        )
        invoked = ui.commands.execute(registered)
        toggled = ui.commands.execute("viewer.scale_bar.toggle", checked=False)
        with self.assertRaisesRegex(ValueError, "checked must be a boolean"):
            ui.commands.execute("viewer.scale_bar.toggle", checked=0)  # type: ignore[arg-type]
        extension.remove_command(registered, if_revision=4)

        self.assertEqual(
            registered.id, "extension:org.example.commands/measure.cells"
        )
        self.assertEqual(registered.ownership["owner_id"], "org.example.commands")
        self.assertEqual(registered.readiness["state"], "ready")
        self.assertEqual(
            registered.predicates.enabled.spec["type"],  # type: ignore[union-attr]
            "all",
        )
        self.assertTrue(registered.state["enabled"])
        self.assertTrue(invoked["dispatched"])
        self.assertFalse(toggled["checked"])
        execute_calls = [call for call in client.calls if call[0] == "ui.commands.execute"]
        self.assertEqual(
            execute_calls[-1][1],
            {"command_id": "viewer.scale_bar.toggle", "checked": False},
        )
        self.assertEqual(client.calls[-1][0], "ui.commands.remove")
        self.assertEqual(client.calls[-1][1]["if_command_revision"], 4)

        with self.assertRaisesRegex(ValueError, "ASCII letters"):
            extension.register_command(
                "bad/id", "Bad", "Bad command.", "bad"
            )
        with self.assertRaisesRegex(ValueError, "unsupported command state path"):
            CommandPredicate.state("private.actor.path")

    def test_shell_wrappers_forward_composition_and_revision(self) -> None:
        client = RecordingClient()
        shell = Ui(client).shell  # type: ignore[arg-type]

        snapshot = shell.get(mode="single")
        patched = shell.patch(
            mode="single",
            visibility={ShellId.SINGLE_TOP_BAR: False},
            orders={
                ShellId.SINGLE_LEFT_TABS: [
                    ShellId.SINGLE_PROJECT,
                    ShellId.SINGLE_LAYERS,
                ]
            },
            selected={ShellId.SINGLE_LEFT_TABS: ShellId.SINGLE_PROJECT},
            if_revision=4,
            transaction_id="review-layout-17",
        )
        shell.reset(mode="single", if_revision=5)

        self.assertIsInstance(snapshot, ShellSnapshot)
        self.assertEqual(snapshot.mode, "single")
        self.assertEqual(snapshot.nodes[0].ownership.owner_id, "odon")
        self.assertTrue(snapshot.nodes[0].ownership.protected)
        self.assertIsNotNone(patched.change)
        self.assertEqual(patched.change.transaction_id, "review-layout-17")
        self.assertEqual(client.calls[0], ("ui.shell.get", {"mode": "single"}))
        self.assertEqual(client.calls[1][0], "ui.shell.patch")
        self.assertEqual(client.calls[1][1]["if_shell_revision"], 4)
        self.assertEqual(
            client.calls[1][1]["orders"]["builtin:single.left-tabs"],
            ["builtin:single.project", "builtin:single.layers"],
        )
        self.assertEqual(
            client.calls[2],
            ("ui.shell.reset", {"mode": "single", "if_shell_revision": 5}),
        )

    def test_replace_layout_builds_one_atomic_desired_tree(self) -> None:
        client = RecordingClient()
        shell = Ui(client).shell  # type: ignore[arg-type]
        layout = self.single_layout()

        snapshot = shell.replace_layout(layout, mode="single", if_revision=4)

        self.assertIsInstance(snapshot, ShellSnapshot)
        self.assertEqual(client.calls[0][0], "ui.shell.replace_layout")
        request = client.calls[0][1]
        self.assertEqual(request["mode"], "single")
        self.assertEqual(request["if_shell_revision"], 4)
        self.assertEqual(request["desired_tree"]["root_id"], "layout:test.root")
        self.assertEqual(len(request["desired_tree"]["nodes"]), 5)

    def test_application_owned_extension_hosts_have_stable_typed_mount_ids(self) -> None:
        layout = ShellLayout(
            "layout:host.root",
            (
                ShellLayoutNode.application(
                    "layout:host.root",
                    ["layout:host.top", "layout:host.body", "layout:host.status"],
                ),
                ShellLayoutNode.toolbar(
                    "layout:host.top",
                    [
                        "layout:host.native-top",
                        "layout:host.command-toolbar",
                        "layout:host.actions",
                    ],
                ),
                ShellLayoutNode.builtin(
                    "layout:host.native-top", ShellMountId.VIEWER_TOP_BAR
                ),
                ShellLayoutNode.builtin(
                    "layout:host.actions", ShellMountId.EXTENSION_TOP_BAR_ACTIONS
                ),
                ShellLayoutNode.row(
                    "layout:host.body", ["layout:host.canvas", "layout:host.inspector"]
                ),
                ShellLayoutNode.canvas(
                    "layout:host.canvas", ShellMountId.VIEWER_CANVAS
                ),
                ShellLayoutNode.builtin(
                    "layout:host.inspector", ShellMountId.SHELL_INSPECTOR
                ),
                ShellLayoutNode.builtin(
                    "layout:host.command-toolbar", ShellMountId.COMMAND_TOOLBAR
                ),
                ShellLayoutNode.status_bar(
                    "layout:host.status", ["layout:host.status-items"]
                ),
                ShellLayoutNode.builtin(
                    "layout:host.status-items", ShellMountId.EXTENSION_STATUS_BAR
                ),
            ),
        )

        layout.validate_for_mode("single")
        self.assertEqual(
            layout.node("layout:host.actions").to_dict()["mount"],
            "builtin:extension-host.top-bar-actions",
        )
        self.assertEqual(
            layout.node("layout:host.inspector").mount,
            "builtin:shell-inspector",
        )
        self.assertEqual(
            layout.node("layout:host.command-toolbar").mount,
            "builtin:command-toolbar",
        )
        self.assertEqual(ShellMountId.HELP, "builtin:help")
        self.assertEqual(
            ShellMountId.RECOVERY_CONTROLS,
            "builtin:recovery-controls",
        )
        self.assertEqual(ShellMountId.CHANNELS, "builtin:channels")
        self.assertEqual(
            ShellMountId.VIEWER_VIEWPORT_CONTROLS,
            "builtin:viewer-viewport-controls",
        )

    def test_component_catalogue_is_typed_and_mode_scoped(self) -> None:
        client = RecordingClient()
        shell = Ui(client).shell  # type: ignore[arg-type]

        components = shell.list_components(mode="single")

        self.assertEqual(len(components), 1)
        self.assertIsInstance(components[0], ShellComponentDescriptor)
        self.assertEqual(components[0].id, "builtin:viewer-canvas")
        self.assertEqual(components[0].ownership.scope, "application")
        self.assertTrue(components[0].ownership.protected)
        self.assertEqual(
            client.calls[0], ("ui.shell.components.list", {"mode": "single"})
        )

    def test_patch_layout_forwards_typed_interactive_state(self) -> None:
        client = RecordingClient()
        shell = Ui(client).shell  # type: ignore[arg-type]

        snapshot = shell.patch_layout(
            selected={"layout:tabs": "layout:second"},
            sizes={"layout:panel": ShellSize(width=420, min_width=220)},
            splits={"layout:split": {"axis": "vertical", "ratio": 0.6}},
            visibility={"layout:first": False},
            collapsed={"layout:section": True},
            configurations={"layout:top": {"show_title": False}},
            active_region_id="layout:second",
            focused_node_id="layout:second",
            mode="single",
            if_revision=4,
            transaction_id="interactive-layout-18",
        )

        self.assertIsInstance(snapshot, ShellSnapshot)
        method, params = client.calls[0]
        self.assertEqual(method, "ui.shell.patch_layout")
        self.assertEqual(params["sizes"]["layout:panel"]["width"], 420.0)
        self.assertEqual(params["splits"]["layout:split"]["axis"], "vertical")
        self.assertEqual(params["active_region_id"], "layout:second")
        self.assertEqual(params["focused_node_id"], "layout:second")
        self.assertEqual(params["configurations"]["layout:top"]["show_title"], False)
        self.assertEqual(params["if_shell_revision"], 4)
        self.assertEqual(params["transaction_id"], "interactive-layout-18")
        self.assertEqual(snapshot.change.transaction_id, "interactive-layout-18")

        shell.patch_layout(clear_focus=True)
        self.assertEqual(client.calls[1], ("ui.shell.patch_layout", {"clear_focus": True}))
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            shell.patch_layout(focused_node_id="layout:second", clear_focus=True)

    def test_layout_documents_export_import_and_recover(self) -> None:
        client = RecordingClient()
        shell = Ui(client).shell  # type: ignore[arg-type]

        document = shell.export_layout(mode="single")
        imported = shell.import_layout(document, mode="single", if_revision=4)
        recovered = shell.recover(mode="single", if_revision=5)

        self.assertIsInstance(document, ShellLayoutDocument)
        self.assertEqual(document.format, "odon.shell-layout")
        self.assertEqual(document.layout.root_id, "layout:test.root")
        self.assertIsInstance(imported, ShellSnapshot)
        self.assertIsInstance(recovered, ShellSnapshot)
        self.assertEqual(client.calls[0], ("ui.shell.export_layout", {"mode": "single"}))
        self.assertEqual(client.calls[1][0], "ui.shell.import_layout")
        self.assertEqual(client.calls[1][1]["document"]["schema_version"], 1)
        self.assertEqual(client.calls[1][1]["if_shell_revision"], 4)
        self.assertEqual(
            client.calls[2],
            ("ui.shell.recover", {"mode": "single", "if_shell_revision": 5}),
        )

    def test_named_layout_profile_wrappers(self) -> None:
        client = RecordingClient()
        shell = Ui(client).shell  # type: ignore[arg-type]

        profiles = shell.list_profiles(scope="application")
        shell.save_profile("review", scope="application", mode="single")
        loaded = shell.load_profile(
            "review", scope="application", mode="single", if_revision=4
        )
        shell.remove_profile("review", scope="application")
        shell.save_profile("team", scope="project", mode="single")

        self.assertEqual(profiles[0].name, "review")
        self.assertEqual(profiles[0].mode, "single")
        self.assertEqual(profiles[0].startup_modes, ("single",))
        self.assertIsInstance(loaded, ShellSnapshot)
        self.assertEqual(client.calls[0][0], "ui.shell.profiles.list")
        self.assertEqual(client.calls[1][0], "ui.shell.profiles.save")
        self.assertEqual(client.calls[2][1]["if_shell_revision"], 4)
        self.assertEqual(client.calls[3][0], "ui.shell.profiles.remove")
        self.assertEqual(client.calls[4][1]["scope"], "project")

    def test_extension_default_layout_wrappers_and_apply(self) -> None:
        client = RecordingClient()
        ui = Ui(client)  # type: ignore[arg-type]
        extension = ui.register_extension(
            id="org.example.review",
            name="Review",
            version="1",
            disconnect_policy="retain",
        )
        document = ShellLayoutDocument("single", self.single_layout())

        registered = extension.register_layout("Review", document)
        listed = extension.list_layouts()
        applied = listed[0].apply(ui.shell, if_revision=4)
        extension.remove_layout("Review")

        self.assertIsInstance(registered, ExtensionLayoutTemplate)
        self.assertEqual(registered.extension_id, "org.example.review")
        self.assertEqual(registered.document.mode, "single")
        self.assertEqual(registered.ownership.scope, "extension")
        self.assertEqual(registered.ownership.owner_session_id, "session-test")
        self.assertEqual(listed[0].name, "Review")
        self.assertIsInstance(applied, ShellSnapshot)
        self.assertEqual(client.calls[1][0], "ui.extensions.layouts.register")
        self.assertEqual(client.calls[1][1]["document"]["schema_version"], 1)
        self.assertEqual(client.calls[2][0], "ui.extensions.layouts.list")
        self.assertEqual(client.calls[3][0], "ui.shell.import_layout")
        self.assertEqual(client.calls[3][1]["if_shell_revision"], 4)
        self.assertEqual(client.calls[4][0], "ui.extensions.layouts.remove")

        before = len(client.calls)
        with self.assertRaisesRegex(ValueError, "name"):
            extension.remove_layout("\n")
        self.assertEqual(len(client.calls), before)

    def test_extension_readiness_and_mount_state_are_typed(self) -> None:
        client = RecordingClient()
        extension = Ui(client).register_extension(  # type: ignore[arg-type]
            id="org.example.ready",
            name="Ready",
            version="1",
            ready=False,
            readiness_reason="warming model",
        )
        extension.set_readiness(False, reason="loading weights")
        self.assertFalse(extension.snapshot["ready"])
        self.assertEqual(extension.snapshot["readiness_reason"], "loading weights")
        self.assertEqual(
            client.calls[1],
            (
                "ui.extensions.set_readiness",
                {
                    "extension_id":"org.example.ready",
                    "ready":False,
                    "reason":"loading weights",
                },
            ),
        )
        node = ShellLayoutNode.from_result(
            {
                "id":"layout:extension",
                "type":"extension_mount",
                "mount":"extension:org.example.ready/panel",
                "readiness":{
                    "state":"incompatible",
                    "reason":None,
                    "expected_extension_version":"1",
                    "current_extension_version":"2",
                },
            }
        )
        self.assertEqual(node.readiness.state, "incompatible")
        self.assertEqual(node.readiness.expected_extension_version, "1")

    def test_shell_local_validation_rejects_invalid_patches(self) -> None:
        client = RecordingClient()
        shell = Ui(client).shell  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "mode"):
            shell.get(mode="detached")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            shell.patch(orders={ShellId.SINGLE_LEFT_TABS: ["same", "same"]})
        with self.assertRaisesRegex(ValueError, "visibility"):
            shell.patch(visibility={ShellId.SINGLE_TOP_BAR: 1})  # type: ignore[dict-item]
        with self.assertRaisesRegex(ValueError, "revision"):
            shell.reset(if_revision=0)
        with self.assertRaisesRegex(ValueError, "transaction_id"):
            shell.reset(transaction_id="")
        with self.assertRaisesRegex(ValueError, "required mount"):
            shell.replace_layout(
                ShellLayout(
                    "layout:root",
                    (
                        ShellLayoutNode.application("layout:root", ["layout:layers"]),
                        ShellLayoutNode.builtin("layout:layers", "builtin:layers"),
                    ),
                ),
                mode="single",
            )
        with self.assertRaisesRegex(ValueError, "booleans"):
            shell.patch_layout(
                collapsed={"layout:section": 1}  # type: ignore[dict-item]
            )
        self.assertEqual(client.calls, [])

    def test_invalid_profile_diagnostics_remain_typed_and_inspectable(self) -> None:
        profile = ShellLayoutProfile.from_result(
            {
                "name": "Future",
                "mode": "future-mode",
                "document_schema_version": 99,
                "valid": False,
                "error": "schema version 99 is unsupported",
                "error_kind": "UNSUPPORTED",
                "recovery_method": "ui.shell.recover",
                "startup_modes": [],
            }
        )
        self.assertFalse(profile.valid)
        self.assertEqual(profile.mode, "future-mode")
        self.assertEqual(profile.error_kind, "UNSUPPORTED")
        self.assertEqual(profile.recovery_method, "ui.shell.recover")


class AsyncShellUiTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_commands_and_menus_match_sync_surface(self) -> None:
        client = AsyncRecordingClient()
        ui = AsyncUi(client)  # type: ignore[arg-type]

        commands = await ui.commands.list()
        snapshot = await ui.menus.get()
        changed = await ui.menus.replace(
            CommandMenuNode.menu_bar(snapshot.menu.id, snapshot.menu.children),
            if_revision=snapshot.revision,
            transaction_id="async-menu-4",
        )

        self.assertIsInstance(commands[0], ApplicationCommand)
        self.assertIsInstance(snapshot, CommandMenuSnapshot)
        self.assertEqual(changed.change["transaction_id"], "async-menu-4")
        self.assertEqual(client.calls[2][0], "ui.menus.replace")

    async def test_async_extension_commands_match_sync_surface(self) -> None:
        client = AsyncRecordingClient()
        ui = AsyncUi(client)  # type: ignore[arg-type]
        extension = await ui.register_extension(
            id="org.example.commands",
            name="Commands",
            version="1.0.0",
            capabilities=("ui.actions",),
            disconnect_policy="retain",
        )
        registered = await extension.register_command(
            "measure",
            "Measure",
            "Measure cells.",
            "measure",
            predicates=CommandPredicates.predicates(
                enabled=CommandPredicate.state("resources.objects")
            ),
        )
        invoked = await ui.commands.execute(registered)
        toggled = await ui.commands.execute("viewer.scale_bar.toggle", checked=True)
        await extension.remove_command(registered)

        self.assertEqual(registered.handler["type"], "event")
        self.assertEqual(
            registered.predicates.enabled.spec["path"],  # type: ignore[union-attr]
            "resources.objects",
        )
        self.assertTrue(invoked["dispatched"])
        self.assertTrue(toggled["checked"])
        execute_calls = [call for call in client.calls if call[0] == "ui.commands.execute"]
        self.assertEqual(
            execute_calls[-1][1],
            {"command_id": "viewer.scale_bar.toggle", "checked": True},
        )
        self.assertEqual(client.calls[-1][0], "ui.commands.remove")

    async def test_async_command_toolbars_match_sync_surface(self) -> None:
        client = AsyncRecordingClient()
        ui = AsyncUi(client)  # type: ignore[arg-type]
        current = await ui.toolbars.get()
        changed = await ui.toolbars.replace(
            CommandToolbar.toolbar(
                "toolbar:main",
                (
                    CommandToolbarGroup.group(
                        "toolbar-group:file",
                        (
                            CommandToolbarItem.command(
                                "toolbar-item:quit", "app.lifecycle.quit"
                            ),
                        ),
                    ),
                ),
            ),
            if_revision=current.revision,
        )
        self.assertEqual(changed.toolbar.groups[0].id, "toolbar-group:file")
        self.assertEqual(client.calls[-1][0], "ui.toolbars.replace")

    async def test_async_command_palette_matches_sync_surface(self) -> None:
        client = AsyncRecordingClient()
        ui = AsyncUi(client)  # type: ignore[arg-type]
        current = await ui.palette.get()
        changed = await ui.palette.replace(
            CommandPalette.palette(
                "palette:analysis",
                title="Analysis commands",
                max_results=8,
            ),
            if_revision=current.revision,
            transaction_id="palette-async-3",
        )

        self.assertEqual(changed.palette.title, "Analysis commands")
        self.assertEqual(changed.palette.max_results, 8)
        self.assertEqual(changed.change["transaction_id"], "palette-async-3")
        self.assertEqual(client.calls[-1][0], "ui.palette.replace")

    async def test_async_shell_wrappers_match_sync_surface(self) -> None:
        client = AsyncRecordingClient()
        shell = AsyncUi(client).shell  # type: ignore[arg-type]

        snapshot = await shell.get()
        components = await shell.list_components(mode="single")
        patched = await shell.patch(visibility={ShellId.PROJECT_EXTENSION_TOP_BAR: False})
        reset = await shell.reset()
        replaced = await shell.replace_layout(ShellUiTests.single_layout(), mode="single")
        layout_patched = await shell.patch_layout(
            sizes={"layout:test.panel": ShellSize(width=300)},
            transaction_id="async-layout-3",
        )
        document = await shell.export_layout(mode="single")
        imported = await shell.import_layout(document, mode="single")
        recovered = await shell.recover(mode="single")
        profiles = await shell.list_profiles()
        await shell.save_profile("review", mode="single")
        profile_loaded = await shell.load_profile("review", mode="single")
        await shell.remove_profile("review")

        self.assertIsInstance(snapshot, ShellSnapshot)
        self.assertIsInstance(components[0], ShellComponentDescriptor)
        self.assertIsInstance(patched, ShellSnapshot)
        self.assertIsInstance(reset, ShellSnapshot)
        self.assertIsInstance(replaced, ShellSnapshot)
        self.assertIsInstance(layout_patched, ShellSnapshot)
        self.assertIsInstance(document, ShellLayoutDocument)
        self.assertIsInstance(imported, ShellSnapshot)
        self.assertIsInstance(recovered, ShellSnapshot)
        self.assertEqual(profiles[0].name, "review")
        self.assertIsInstance(profile_loaded, ShellSnapshot)
        self.assertEqual(client.calls[0], ("ui.shell.get", {}))
        self.assertEqual(client.calls[1][0], "ui.shell.components.list")
        self.assertEqual(client.calls[2][0], "ui.shell.patch")
        self.assertEqual(client.calls[3], ("ui.shell.reset", {}))
        self.assertEqual(client.calls[4][0], "ui.shell.replace_layout")
        self.assertEqual(client.calls[5][0], "ui.shell.patch_layout")
        self.assertEqual(client.calls[5][1]["transaction_id"], "async-layout-3")
        self.assertEqual(layout_patched.change.transaction_id, "async-layout-3")
        self.assertEqual(client.calls[6][0], "ui.shell.export_layout")
        self.assertEqual(client.calls[7][0], "ui.shell.import_layout")
        self.assertEqual(client.calls[8][0], "ui.shell.recover")
        self.assertEqual(client.calls[9][0], "ui.shell.profiles.list")
        self.assertEqual(client.calls[10][0], "ui.shell.profiles.save")
        self.assertEqual(client.calls[11][0], "ui.shell.profiles.load")
        self.assertEqual(client.calls[12][0], "ui.shell.profiles.remove")

    async def test_async_extension_default_layout_wrappers(self) -> None:
        client = AsyncRecordingClient()
        ui = AsyncUi(client)  # type: ignore[arg-type]
        extension = await ui.register_extension(
            id="org.example.review",
            name="Review",
            version="1",
            disconnect_policy="disable",
        )
        document = ShellLayoutDocument("single", ShellUiTests.single_layout())

        registered = await extension.register_layout("Review", document)
        listed = await extension.list_layouts()
        await extension.remove_layout("Review")

        self.assertIsInstance(registered, ExtensionLayoutTemplate)
        self.assertEqual(listed[0].document.layout.root_id, "layout:test.root")
        self.assertEqual(client.calls[1][0], "ui.extensions.layouts.register")
        self.assertEqual(client.calls[2][0], "ui.extensions.layouts.list")
        self.assertEqual(client.calls[3][0], "ui.extensions.layouts.remove")
