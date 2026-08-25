"""Versioned declarative UI components rendered natively by Odon/egui."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
import math
import unicodedata
from typing import TYPE_CHECKING, Any, Iterable, Mapping

if TYPE_CHECKING:
    from .client import Client


def emit(event: str, **data: Any) -> dict[str, Any]:
    return {"type": "emit", "event": event, "data": data}


def command(method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"type": "command", "method": method, "params": dict(params or {})}


def bind(target: str, **selector: Any) -> dict[str, Any]:
    return {"type": "bind", "target": target, **selector}


def command_state(
    command_id: str, *, state: str = "enabled", equals: bool = True
) -> dict[str, Any]:
    """Bind a component property to evaluated actor-owned command state."""

    command_id = _command_text(command_id, "command state binding command ID")
    if state not in {"visible", "enabled", "checked"}:
        raise ValueError("command state binding state must be visible, enabled, or checked")
    if not isinstance(equals, bool):
        raise ValueError("command state binding equals must be a boolean")
    return {
        "type": "command_state",
        "command_id": command_id,
        "state": state,
        "equals": equals,
    }


@dataclass(frozen=True)
class EventPolicy:
    type: str
    milliseconds: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            **({"milliseconds": self.milliseconds} if self.milliseconds is not None else {}),
        }


class Immediate(EventPolicy):
    def __init__(self) -> None:
        super().__init__("immediate")


class OnCommit(EventPolicy):
    def __init__(self) -> None:
        super().__init__("commit")


class Throttle(EventPolicy):
    def __init__(self, *, milliseconds: int) -> None:
        super().__init__("throttle", milliseconds)


class Debounce(EventPolicy):
    def __init__(self, *, milliseconds: int) -> None:
        super().__init__("debounce", milliseconds)


@dataclass
class Component:
    id: str
    type: str = field(init=False)
    label: str | None = None
    title: str | None = None
    help: str | None = None
    visible: bool = True
    enabled: bool = True
    value: Any = None
    minimum: float | None = None
    maximum: float | None = None
    options: list[Any] = field(default_factory=list)
    columns: int | None = None
    action: Mapping[str, Any] | None = None
    event_policy: EventPolicy | Mapping[str, Any] | None = None
    state_bindings: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    children: list["Component"] = field(default_factory=list)

    def when(
        self,
        *,
        visible: Mapping[str, Any] | None = None,
        enabled: Mapping[str, Any] | None = None,
    ) -> "Component":
        """Bind visibility or enablement to actor-evaluated command state."""

        if visible is not None:
            self.state_bindings["visible"] = dict(visible)
        if enabled is not None:
            self.state_bindings["enabled"] = dict(enabled)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "title": self.title,
            "help": self.help,
            "visible": self.visible,
            "enabled": self.enabled,
            "value": self.value,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "options": self.options,
            "columns": self.columns,
            "action": dict(self.action) if self.action is not None else None,
            "event_policy": (
                self.event_policy.to_dict()
                if isinstance(self.event_policy, EventPolicy)
                else dict(self.event_policy)
                if self.event_policy is not None
                else None
            ),
            **(
                {
                    "state_bindings": {
                        property: dict(binding)
                        for property, binding in self.state_bindings.items()
                    }
                }
                if self.state_bindings
                else {}
            ),
            "children": [child.to_dict() for child in self.children],
        }


class Container(Component):
    def __init__(
        self,
        id: str,
        *,
        children: Iterable[Component] = (),
        label: str | None = None,
        title: str | None = None,
    ) -> None:
        super().__init__(id=id, label=label, title=title, children=list(children))


class Panel(Container):
    type = "panel"


class Column(Container):
    type = "column"


class Row(Container):
    type = "row"


class Grid(Container):
    type = "grid"

    def __init__(
        self, id: str, *, columns: int = 2, children: Iterable[Component] = (), label: str | None = None
    ) -> None:
        Component.__init__(
            self, id=id, label=label, columns=columns, children=list(children)
        )


class Tabs(Container):
    type = "tabs"

    def __init__(
        self, id: str, *, children: Iterable[Component] = (), selected: str | None = None
    ) -> None:
        Component.__init__(self, id=id, value=selected, children=list(children))


class Scroll(Container):
    type = "scroll"


class Group(Container):
    type = "group"


class Collapsible(Container):
    type = "collapsible"


class Text(Component):
    type = "text"

    def __init__(self, id: str, text: str) -> None:
        super().__init__(id=id, value=text)


class Markdown(Text):
    type = "markdown"


class Status(Text):
    type = "status"


class Warning(Text):
    type = "warning"


class Error(Text):
    type = "error"


class Spinner(Component):
    type = "spinner"

    def __init__(self, id: str, label: str | None = None) -> None:
        super().__init__(id=id, label=label)


class Button(Component):
    type = "button"

    def __init__(
        self, id: str, label: str, *, action: Mapping[str, Any], enabled: bool = True
    ) -> None:
        super().__init__(id=id, label=label, action=action, enabled=enabled)


class Toggle(Component):
    type = "toggle"

    def __init__(
        self,
        id: str,
        label: str,
        *,
        value: bool = False,
        action: Mapping[str, Any] | None = None,
        event_policy: EventPolicy | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            id=id, label=label, value=value, action=action, event_policy=event_policy
        )


class Checkbox(Toggle):
    type = "checkbox"


class Slider(Component):
    type = "slider"

    def __init__(
        self,
        id: str,
        label: str,
        *,
        minimum: float,
        maximum: float,
        value: float,
        action: Mapping[str, Any] | None = None,
        event_policy: EventPolicy | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            id=id,
            label=label,
            value=value,
            minimum=minimum,
            maximum=maximum,
            action=action,
            event_policy=event_policy,
        )


class Number(Slider):
    type = "number"


class Integer(Slider):
    type = "integer"


class TextInput(Component):
    type = "text_input"

    def __init__(
        self,
        id: str,
        label: str,
        *,
        value: str = "",
        action: Mapping[str, Any] | None = None,
        event_policy: EventPolicy | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            id=id, label=label, value=value, action=action, event_policy=event_policy
        )


class Select(Component):
    type = "select"

    def __init__(
        self,
        id: str,
        label: str,
        *,
        options: Iterable[Any],
        value: Any = None,
        action: Mapping[str, Any] | None = None,
        event_policy: EventPolicy | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            id=id,
            label=label,
            options=list(options),
            value=value,
            action=action,
            event_policy=event_policy,
        )


class Radio(Select):
    type = "radio"


class MultiSelect(Select):
    type = "multi_select"

    def __init__(
        self,
        id: str,
        label: str,
        *,
        options: Iterable[Any],
        value: Iterable[Any] = (),
        action: Mapping[str, Any] | None = None,
        event_policy: EventPolicy | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            id,
            label,
            options=options,
            value=list(value),
            action=action,
            event_policy=event_policy,
        )


class Color(Component):
    type = "color"

    def __init__(
        self,
        id: str,
        label: str,
        *,
        value: str = "#ffffffff",
        action: Mapping[str, Any] | None = None,
        event_policy: EventPolicy | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            id=id,
            label=label,
            value=value,
            action=action,
            event_policy=event_policy,
        )


class Progress(Component):
    type = "progress"

    def __init__(self, id: str, *, value: float = 0.0, label: str | None = None) -> None:
        super().__init__(id=id, label=label, value=value)


class Separator(Component):
    type = "separator"

    def __init__(self, id: str) -> None:
        super().__init__(id=id)


class Spacer(Component):
    type = "spacer"

    def __init__(self, id: str, *, points: float = 8.0) -> None:
        super().__init__(id=id, value=points)


class Contribution:
    def __init__(self, extension: "Extension", snapshot: Mapping[str, Any]) -> None:
        self._extension = extension
        self.snapshot = dict(snapshot)

    @property
    def contribution_id(self) -> str:
        return str(self.snapshot["contribution_id"])

    @property
    def revision(self) -> int:
        return int(self.snapshot["revision"])

    @property
    def shell_mount(self) -> str:
        return str(self.snapshot["shell_mount"])

    @property
    def ownership(self) -> "ShellOwnership | None":
        value = self.snapshot.get("ownership")
        if value is None:
            return None
        return ShellOwnership.from_result(
            _require_mapping(value, "contribution ownership")
        )

    def mount(self, node_id: str, **kwargs: Any) -> "ShellLayoutNode":
        """Create a keyed shell node that renders this contribution."""
        return ShellLayoutNode.extension(node_id, self.shell_mount, **kwargs)

    def patch_values(
        self, values: Mapping[str, Any], *, if_revision: int | None = None
    ) -> "Contribution":
        result = self._extension._ui._client.call(
            "ui.contributions.patch_values",
            {
                "contribution_id": self.contribution_id,
                "values": dict(values),
                **({"if_revision": if_revision} if if_revision is not None else {}),
            },
        )
        self.snapshot = dict(result)
        return self

    def remove(self) -> None:
        self._extension._ui._client.call(
            "ui.contributions.remove", {"contribution_id": self.contribution_id}
        )


class Extension:
    def __init__(self, ui: "Ui", snapshot: Mapping[str, Any]) -> None:
        self._ui = ui
        self.snapshot = dict(snapshot)

    @property
    def id(self) -> str:
        return str(self.snapshot["id"])

    @property
    def granted_capabilities(self) -> frozenset[str]:
        return frozenset(self.snapshot.get("granted_capabilities", ()))

    def set_readiness(self, ready: bool, *, reason: str | None = None) -> "Extension":
        """Publish whether this extension is ready for its retained mounts to render."""

        if not isinstance(ready, bool):
            raise ValueError("extension readiness must be a boolean")
        params: dict[str, Any] = {"extension_id": self.id, "ready": ready}
        if reason is not None:
            if not isinstance(reason, str) or not reason or len(reason.encode("utf-8")) > 256:
                raise ValueError("extension readiness reason must contain 1 to 256 bytes")
            params["reason"] = reason
        self.snapshot = dict(
            self._ui._client.call("ui.extensions.set_readiness", params)
        )
        return self

    def register(
        self,
        root: Component,
        *,
        location: str = "shell",
        contribution_id: str | None = None,
    ) -> Contribution:
        params: dict[str, Any] = {
            "extension_id": self.id,
            "location": location,
            "root": root.to_dict(),
        }
        if contribution_id is not None:
            params["contribution_id"] = contribution_id
        return Contribution(
            self, self._ui._client.call("ui.contributions.register", params)
        )

    def register_command(
        self,
        id: str,
        title: str,
        description: str,
        event: str,
        *,
        modes: Iterable[str] = ("project", "single", "mosaic"),
        shortcut: Mapping[str, Any] | None = None,
        icon: str | None = None,
        predicates: "CommandPredicates | Mapping[str, Any] | None" = None,
        if_revision: int | None = None,
        transaction_id: str | None = None,
    ) -> "ApplicationCommand":
        """Register an owned action as ``extension:<extension-id>/<id>``."""

        result = _require_mapping(
            self._ui._client.call(
                "ui.commands.register",
                _extension_command_register_params(
                    self.id,
                    id,
                    title,
                    description,
                    event,
                    modes=modes,
                    shortcut=shortcut,
                    icon=icon,
                    predicates=predicates,
                    if_revision=if_revision,
                    transaction_id=transaction_id,
                ),
            ),
            "extension command registration",
        )
        return ApplicationCommand.from_result(
            _require_mapping(result.get("command"), "registered extension command")
        )

    def remove_command(
        self,
        command: "ApplicationCommand | str",
        *,
        if_revision: int | None = None,
        transaction_id: str | None = None,
    ) -> None:
        """Remove an owned command and all menu items that present it."""

        command_id = command.id if isinstance(command, ApplicationCommand) else command
        params: dict[str, Any] = {
            "extension_id": self.id,
            "command_id": _command_text(command_id, "command id"),
        }
        revision = _validate_command_revision(if_revision)
        if revision is not None:
            params["if_command_revision"] = revision
        checked_transaction = _validate_shell_transaction_id(transaction_id)
        if checked_transaction is not None:
            params["transaction_id"] = checked_transaction
        self._ui._client.call("ui.commands.remove", params)

    def register_layout(
        self,
        name: str,
        document: "ShellLayoutDocument | Mapping[str, Any]",
    ) -> "ExtensionLayoutTemplate":
        """Register or replace a named default layout owned by this extension."""

        result = self._ui._client.call(
            "ui.extensions.layouts.register",
            {
                "extension_id": self.id,
                "name": _extension_layout_name(name),
                "document": _extension_layout_document(document),
            },
        )
        return ExtensionLayoutTemplate.from_result(
            _require_mapping(result, "extension layout template")
        )

    def list_layouts(self) -> tuple["ExtensionLayoutTemplate", ...]:
        """List this extension's registered default layouts by name."""

        result = _require_mapping(
            self._ui._client.call(
                "ui.extensions.layouts.list", {"extension_id": self.id}
            ),
            "extension layout list",
        )
        raw = result.get("layouts")
        if not isinstance(raw, list):
            raise ValueError("extension layout list requires a layouts array")
        return tuple(
            ExtensionLayoutTemplate.from_result(
                _require_mapping(item, "extension layout template")
            )
            for item in raw
        )

    def remove_layout(self, name: str) -> None:
        """Remove a named default layout owned by this extension."""

        self._ui._client.call(
            "ui.extensions.layouts.remove",
            {"extension_id": self.id, "name": _extension_layout_name(name)},
        )

    def remove(self) -> None:
        self._ui._client.call("ui.extensions.remove", {"extension_id": self.id})


class ShellId(str, Enum):
    """Stable shell schema-version-1 built-in and extension-host IDs."""

    PROJECT_ROOT = "builtin:project.root"
    PROJECT_TOP_BAR = "builtin:project.top-bar"
    PROJECT_WORKSPACE = "builtin:project.workspace"
    PROJECT_EXTENSION_TOP_BAR = "extension:project.top-bar"
    PROJECT_EXTENSION_STATUS_BAR = "extension:project.status-bar"
    PROJECT_EXTENSION_CARDS = "extension:project.project-cards"

    SINGLE_ROOT = "builtin:single.root"
    SINGLE_TOP_BAR = "builtin:single.top-bar"
    SINGLE_LEFT_PANEL = "builtin:single.left-panel"
    SINGLE_LEFT_TABS = "builtin:single.left-tabs"
    SINGLE_LAYERS = "builtin:single.layers"
    SINGLE_PROJECT = "builtin:single.project"
    SINGLE_CANVAS = "builtin:single.canvas"
    SINGLE_RIGHT_PANEL = "builtin:single.right-panel"
    SINGLE_RIGHT_TABS = "builtin:single.right-tabs"
    SINGLE_PROPERTIES = "builtin:single.properties"
    SINGLE_VIEWS = "builtin:single.views"
    SINGLE_ANALYSIS = "builtin:single.analysis"
    SINGLE_MEASUREMENTS = "builtin:single.measurements"
    SINGLE_MEMORY = "builtin:single.memory"
    SINGLE_ROI_SELECTOR = "builtin:single.roi-selector"
    SINGLE_EXTENSION_TOP_BAR = "extension:single.top-bar"
    SINGLE_EXTENSION_STATUS_BAR = "extension:single.status-bar"
    SINGLE_EXTENSION_LEFT_PANEL = "extension:single.left-panel"
    SINGLE_EXTENSION_RIGHT_PANEL = "extension:single.right-panel"
    SINGLE_EXTENSION_CANVAS_CONTROLS = "extension:single.canvas-controls"

    MOSAIC_ROOT = "builtin:mosaic.root"
    MOSAIC_TOP_BAR = "builtin:mosaic.top-bar"
    MOSAIC_LEFT_PANEL = "builtin:mosaic.left-panel"
    MOSAIC_LEFT_TABS = "builtin:mosaic.left-tabs"
    MOSAIC_LAYERS = "builtin:mosaic.layers"
    MOSAIC_PROJECT = "builtin:mosaic.project"
    MOSAIC_CANVAS = "builtin:mosaic.canvas"
    MOSAIC_RIGHT_PANEL = "builtin:mosaic.right-panel"
    MOSAIC_RIGHT_TABS = "builtin:mosaic.right-tabs"
    MOSAIC_PROPERTIES = "builtin:mosaic.properties"
    MOSAIC_VIEWS = "builtin:mosaic.views"
    MOSAIC_LAYOUT = "builtin:mosaic.layout"
    MOSAIC_MEMORY = "builtin:mosaic.memory"
    MOSAIC_EXTENSION_TOP_BAR = "extension:mosaic.top-bar"
    MOSAIC_EXTENSION_STATUS_BAR = "extension:mosaic.status-bar"
    MOSAIC_EXTENSION_LEFT_PANEL = "extension:mosaic.left-panel"
    MOSAIC_EXTENSION_RIGHT_PANEL = "extension:mosaic.right-panel"
    MOSAIC_EXTENSION_CANVAS_CONTROLS = "extension:mosaic.canvas-controls"


class ShellLayoutType(str, Enum):
    """Supported node types in an actor-owned desired application layout."""

    APPLICATION = "application"
    ROW = "row"
    COLUMN = "column"
    SPLIT = "split"
    TABS = "tabs"
    PANEL = "panel"
    COLLAPSIBLE = "collapsible"
    TOOLBAR = "toolbar"
    STATUS_BAR = "status_bar"
    MENU_HOST = "menu_host"
    CANVAS_SLOT = "canvas_slot"
    BUILTIN_MOUNT = "builtin_mount"
    EXTENSION_MOUNT = "extension_mount"


class ShellMountId(str, Enum):
    """Stable built-in component and application-owned extension-host mount IDs."""

    PROJECT_TOP_BAR = "builtin:project-top-bar"
    PROJECT_WORKSPACE = "builtin:project-workspace"
    VIEWER_TOP_BAR = "builtin:viewer-top-bar"
    VIEWER_CANVAS = "builtin:viewer-canvas"
    MOSAIC_TOP_BAR = "builtin:mosaic-top-bar"
    MOSAIC_CANVAS = "builtin:mosaic-canvas"
    LAYERS = "builtin:layers"
    PROJECT = "builtin:project"
    PROPERTIES = "builtin:properties"
    VIEWS = "builtin:views"
    ANALYSIS = "builtin:analysis"
    MEASUREMENTS = "builtin:measurements"
    MEMORY = "builtin:memory"
    ROI_SELECTOR = "builtin:roi-selector"
    MOSAIC_LAYOUT = "builtin:mosaic-layout"
    EXTENSION_TOP_BAR_ACTIONS = "builtin:extension-host.top-bar-actions"
    EXTENSION_STATUS_BAR = "builtin:extension-host.status-bar"
    EXTENSION_LEFT_SECTIONS = "builtin:extension-host.left-sections"
    EXTENSION_RIGHT_TABS = "builtin:extension-host.right-tabs"
    EXTENSION_CANVAS_CONTROLS = "builtin:extension-host.canvas-controls"
    EXTENSION_PROJECT_CARDS = "builtin:extension-host.project-cards"
    SHELL_INSPECTOR = "builtin:shell-inspector"
    COMMAND_TOOLBAR = "builtin:command-toolbar"
    HELP = "builtin:help"
    RECOVERY_CONTROLS = "builtin:recovery-controls"
    CHANNELS = "builtin:channels"
    VIEWER_VIEWPORT_CONTROLS = "builtin:viewer-viewport-controls"


@dataclass(frozen=True)
class ShellOwnership:
    """Server-derived ownership and protection metadata for a shell node or component."""

    scope: str
    owner_id: str
    protected: bool
    owner_session_id: str | None = None

    @classmethod
    def from_result(cls, value: Mapping[str, Any]) -> "ShellOwnership":
        scope = value.get("scope")
        owner_id = value.get("owner_id")
        protected = value.get("protected")
        owner_session_id = value.get("owner_session_id")
        if (
            scope not in {"application", "extension"}
            or not isinstance(owner_id, str)
            or not owner_id
            or not isinstance(protected, bool)
            or (owner_session_id is not None and not isinstance(owner_session_id, str))
        ):
            raise ValueError("invalid shell ownership metadata")
        return cls(scope, owner_id, protected, owner_session_id)


@dataclass(frozen=True)
class ShellMountReadiness:
    """Actor-reported usability of one retained extension mount."""

    state: str
    reason: str | None = None
    expected_extension_version: str | None = None
    current_extension_version: str | None = None

    @classmethod
    def from_result(cls, value: Mapping[str, Any]) -> "ShellMountReadiness":
        state = value.get("state")
        reason = value.get("reason")
        expected = value.get("expected_extension_version")
        current = value.get("current_extension_version")
        if state not in {"ready", "not_ready", "disconnected", "incompatible", "missing"}:
            raise ValueError("invalid shell mount readiness state")
        if any(item is not None and not isinstance(item, str) for item in (reason, expected, current)):
            raise ValueError("invalid shell mount readiness metadata")
        return cls(state, reason, expected, current)


@dataclass(frozen=True)
class ShellSize:
    """Advisory size constraints for one desired-layout node."""

    width: float | None = None
    height: float | None = None
    min_width: float | None = None
    min_height: float | None = None
    max_width: float | None = None
    max_height: float | None = None
    flex: float | None = None

    def to_dict(self) -> dict[str, float]:
        values = {
            name: value
            for name, value in (
                ("width", self.width),
                ("height", self.height),
                ("min_width", self.min_width),
                ("min_height", self.min_height),
                ("max_width", self.max_width),
                ("max_height", self.max_height),
                ("flex", self.flex),
            )
            if value is not None
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"shell size {name} must be a number")
            if not math.isfinite(value):
                raise ValueError(f"shell size {name} must be finite")
            if value <= 0 and name not in {"min_width", "min_height"}:
                raise ValueError(f"shell size {name} must be positive")
            if value < 0:
                raise ValueError(f"shell size {name} must not be negative")
        if self.min_width is not None and self.max_width is not None and self.min_width > self.max_width:
            raise ValueError("shell min_width must not exceed max_width")
        if self.min_height is not None and self.max_height is not None and self.min_height > self.max_height:
            raise ValueError("shell min_height must not exceed max_height")
        return {name: float(value) for name, value in values.items()}

    @classmethod
    def from_result(cls, value: Mapping[str, Any]) -> "ShellSize":
        allowed = {
            "width", "height", "min_width", "min_height", "max_width", "max_height", "flex"
        }
        if set(value) - allowed:
            raise ValueError("shell size contains unknown fields")
        size = cls(**{name: value.get(name) for name in allowed})
        size.to_dict()
        return size


@dataclass(frozen=True)
class ShellSplit:
    """Resizable two-child split configuration."""

    ratio: float = 0.5
    resizable: bool = True
    axis: str = "horizontal"

    def to_dict(self) -> dict[str, Any]:
        if isinstance(self.ratio, bool) or not isinstance(self.ratio, (int, float)):
            raise ValueError("shell split ratio must be a number")
        if not math.isfinite(self.ratio):
            raise ValueError("shell split ratio must be finite")
        if not 0.05 <= self.ratio <= 0.95:
            raise ValueError("shell split ratio must be between 0.05 and 0.95")
        if not isinstance(self.resizable, bool):
            raise ValueError("shell split resizable must be a boolean")
        if self.axis not in {"horizontal", "vertical"}:
            raise ValueError("shell split axis must be 'horizontal' or 'vertical'")
        return {"axis": self.axis, "ratio": float(self.ratio), "resizable": self.resizable}

    @classmethod
    def from_result(cls, value: Mapping[str, Any]) -> "ShellSplit":
        if set(value) - {"axis", "ratio", "resizable"} or "ratio" not in value:
            raise ValueError("invalid shell split options")
        split = cls(value["ratio"], value.get("resizable", True), value.get("axis", "horizontal"))
        split.to_dict()
        return split


@dataclass(frozen=True)
class ShellLayoutNode:
    """One stable keyed node in a complete desired application layout."""

    id: str
    type: ShellLayoutType | str
    children: tuple[str, ...] = ()
    visible: bool = True
    title: str | None = None
    mount: str | None = None
    selected_id: str | None = None
    size: ShellSize = field(default_factory=ShellSize)
    split: ShellSplit | None = None
    collapsed: bool = False
    configuration: Mapping[str, Any] = field(default_factory=dict)
    state_bindings: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    parent_id: str | None = field(default=None, compare=False)
    ownership: ShellOwnership | None = field(default=None, compare=False)
    readiness: ShellMountReadiness | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _shell_id(self.id))
        try:
            kind = self.type if isinstance(self.type, ShellLayoutType) else ShellLayoutType(self.type)
        except ValueError as error:
            raise ValueError(f"unknown shell layout node type: {self.type!r}") from error
        object.__setattr__(self, "type", kind)
        children = tuple(_shell_id(child) for child in self.children)
        if len(children) != len(set(children)):
            raise ValueError(f"shell layout node '{self.id}' has duplicate children")
        object.__setattr__(self, "children", children)
        if not isinstance(self.visible, bool) or not isinstance(self.collapsed, bool):
            raise ValueError("shell layout visibility and collapsed values must be booleans")
        for label, value in (("title", self.title), ("mount", self.mount), ("selected_id", self.selected_id), ("parent_id", self.parent_id)):
            if value is not None:
                _shell_id(value)
        self.size.to_dict()
        if self.split is not None:
            self.split.to_dict()
        if not isinstance(self.configuration, Mapping):
            raise ValueError("shell mount configuration must be a mapping")
        object.__setattr__(self, "configuration", dict(self.configuration))
        if not isinstance(self.state_bindings, Mapping):
            raise ValueError("shell state bindings must be a mapping")
        checked_bindings: dict[str, Mapping[str, Any]] = {}
        for property, binding in self.state_bindings.items():
            if property != "visible" or not isinstance(binding, Mapping):
                raise ValueError("shell nodes can bind only visible to command state")
            if set(binding) - {"type", "command_id", "state", "equals"}:
                raise ValueError("shell visible binding contains unknown fields")
            if binding.get("type") != "command_state":
                raise ValueError("shell visible binding type must be 'command_state'")
            checked_bindings[property] = command_state(
                binding.get("command_id"),
                state=binding.get("state", ""),
                equals=binding.get("equals", True),
            )
        object.__setattr__(self, "state_bindings", checked_bindings)

    def to_dict(self, *, include_parent: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "type": self.type.value,
            "children": list(self.children),
            "visible": self.visible,
            "size": self.size.to_dict(),
            "collapsed": self.collapsed,
            "configuration": dict(self.configuration),
            "state_bindings": {
                property: dict(binding) for property, binding in self.state_bindings.items()
            },
        }
        if include_parent:
            result["parent_id"] = self.parent_id
        if self.title is not None:
            result["title"] = self.title
        if self.mount is not None:
            result["mount"] = self.mount
        if self.selected_id is not None:
            result["selected_id"] = self.selected_id
        if self.split is not None:
            result["split"] = self.split.to_dict()
        return result

    @classmethod
    def from_result(cls, value: Mapping[str, Any]) -> "ShellLayoutNode":
        return cls(
            id=value.get("id", ""),
            type=value.get("type", ""),
            parent_id=value.get("parent_id"),
            children=tuple(value.get("children", ())),
            visible=value.get("visible", True),
            title=value.get("title"),
            mount=value.get("mount"),
            selected_id=value.get("selected_id"),
            size=ShellSize.from_result(_require_mapping(value.get("size", {}), "shell size")),
            split=(
                ShellSplit.from_result(_require_mapping(value["split"], "shell split"))
                if value.get("split") is not None
                else None
            ),
            collapsed=value.get("collapsed", False),
            configuration=dict(
                _require_mapping(value.get("configuration", {}), "shell mount configuration")
            ),
            state_bindings={
                property: dict(_require_mapping(binding, "shell state binding"))
                for property, binding in _require_mapping(
                    value.get("state_bindings", {}), "shell state bindings"
                ).items()
            },
            ownership=(
                ShellOwnership.from_result(
                    _require_mapping(value["ownership"], "shell ownership")
                )
                if value.get("ownership") is not None
                else None
            ),
            readiness=(
                ShellMountReadiness.from_result(
                    _require_mapping(value["readiness"], "shell mount readiness")
                )
                if value.get("readiness") is not None
                else None
            ),
        )

    @classmethod
    def application(cls, id: str, children: Iterable[str]) -> "ShellLayoutNode":
        return cls(id, ShellLayoutType.APPLICATION, tuple(children))

    @classmethod
    def row(cls, id: str, children: Iterable[str], **kwargs: Any) -> "ShellLayoutNode":
        return cls(id, ShellLayoutType.ROW, tuple(children), **kwargs)

    @classmethod
    def column(cls, id: str, children: Iterable[str], **kwargs: Any) -> "ShellLayoutNode":
        return cls(id, ShellLayoutType.COLUMN, tuple(children), **kwargs)

    @classmethod
    def panel(cls, id: str, child: str, **kwargs: Any) -> "ShellLayoutNode":
        return cls(id, ShellLayoutType.PANEL, (_shell_id(child),), **kwargs)

    @classmethod
    def collapsible(cls, id: str, child: str, **kwargs: Any) -> "ShellLayoutNode":
        return cls(id, ShellLayoutType.COLLAPSIBLE, (_shell_id(child),), **kwargs)

    @classmethod
    def toolbar(cls, id: str, children: Iterable[str], **kwargs: Any) -> "ShellLayoutNode":
        return cls(id, ShellLayoutType.TOOLBAR, tuple(children), **kwargs)

    @classmethod
    def status_bar(cls, id: str, children: Iterable[str], **kwargs: Any) -> "ShellLayoutNode":
        return cls(id, ShellLayoutType.STATUS_BAR, tuple(children), **kwargs)

    @classmethod
    def menu_host(cls, id: str, children: Iterable[str], **kwargs: Any) -> "ShellLayoutNode":
        return cls(id, ShellLayoutType.MENU_HOST, tuple(children), **kwargs)

    @classmethod
    def tabs(cls, id: str, children: Iterable[str], *, selected: str, **kwargs: Any) -> "ShellLayoutNode":
        return cls(id, ShellLayoutType.TABS, tuple(children), selected_id=selected, **kwargs)

    @classmethod
    def split_node(cls, id: str, first: str, second: str, *, axis: str = "horizontal", ratio: float = 0.5, resizable: bool = True, **kwargs: Any) -> "ShellLayoutNode":
        return cls(id, ShellLayoutType.SPLIT, (first, second), split=ShellSplit(ratio, resizable, axis), **kwargs)

    @classmethod
    def builtin(cls, id: str, mount: str, **kwargs: Any) -> "ShellLayoutNode":
        return cls(id, ShellLayoutType.BUILTIN_MOUNT, mount=mount, **kwargs)

    @classmethod
    def canvas(cls, id: str, mount: str, **kwargs: Any) -> "ShellLayoutNode":
        return cls(id, ShellLayoutType.CANVAS_SLOT, mount=mount, **kwargs)

    @classmethod
    def extension(cls, id: str, mount: str, **kwargs: Any) -> "ShellLayoutNode":
        return cls(id, ShellLayoutType.EXTENSION_MOUNT, mount=mount, **kwargs)


@dataclass(frozen=True)
class ShellLayout(Mapping[str, Any]):
    """A complete actor-validated desired layout tree."""

    root_id: str
    nodes: tuple[ShellLayoutNode, ...]
    _raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "root_id", _shell_id(self.root_id))
        object.__setattr__(self, "nodes", tuple(self.nodes))
        self._validate_topology()

    @classmethod
    def from_result(cls, value: Mapping[str, Any]) -> "ShellLayout":
        raw_nodes = value.get("nodes")
        if not isinstance(raw_nodes, list):
            raise ValueError("shell layout nodes must be an array")
        return cls(
            root_id=value.get("root_id", ""),
            nodes=tuple(
                ShellLayoutNode.from_result(_require_mapping(node, "shell layout node"))
                for node in raw_nodes
            ),
            _raw=dict(value),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"root_id": self.root_id, "nodes": [node.to_dict() for node in self.nodes]}

    def node(self, node_id: str) -> ShellLayoutNode:
        wanted = _shell_id(node_id)
        for node in self.nodes:
            if node.id == wanted:
                return node
        raise KeyError(wanted)

    def validate_for_mode(self, mode: str) -> None:
        mode = _validate_shell_mode(mode, required=True)
        required = {
            "project": "builtin:project-workspace",
            "single": "builtin:viewer-canvas",
            "mosaic": "builtin:mosaic-canvas",
        }[mode]
        mounts = [node.mount for node in self.nodes if node.mount is not None]
        if required not in mounts:
            raise ValueError(f"shell layout must retain required mount '{required}'")
        builtins = [
            node.mount
            for node in self.nodes
            if node.type in {ShellLayoutType.BUILTIN_MOUNT, ShellLayoutType.CANVAS_SLOT}
        ]
        if len(builtins) != len(set(builtins)):
            raise ValueError("shell layouts cannot duplicate singleton built-in mounts")
        allowed = {
            "project": {
                "builtin:project-top-bar", "builtin:project-workspace",
                "builtin:extension-host.top-bar-actions",
                "builtin:extension-host.status-bar",
                "builtin:extension-host.project-cards",
                "builtin:shell-inspector",
                "builtin:command-toolbar",
            },
            "single": {
                "builtin:viewer-top-bar", "builtin:viewer-canvas", "builtin:layers",
                "builtin:project", "builtin:properties", "builtin:views", "builtin:analysis",
                "builtin:measurements", "builtin:memory", "builtin:roi-selector",
                "builtin:extension-host.top-bar-actions",
                "builtin:extension-host.status-bar",
                "builtin:extension-host.left-sections",
                "builtin:extension-host.right-tabs",
                "builtin:extension-host.canvas-controls",
                "builtin:shell-inspector",
                "builtin:command-toolbar",
            },
            "mosaic": {
                "builtin:mosaic-top-bar", "builtin:mosaic-canvas", "builtin:layers",
                "builtin:project", "builtin:properties", "builtin:views",
                "builtin:mosaic-layout", "builtin:memory",
                "builtin:extension-host.top-bar-actions",
                "builtin:extension-host.status-bar",
                "builtin:extension-host.left-sections",
                "builtin:extension-host.right-tabs",
                "builtin:extension-host.canvas-controls",
                "builtin:shell-inspector",
                "builtin:command-toolbar",
            },
        }[mode]
        for node in self.nodes:
            if node.type in {ShellLayoutType.BUILTIN_MOUNT, ShellLayoutType.CANVAS_SLOT}:
                if node.mount not in allowed:
                    raise ValueError(f"built-in mount '{node.mount}' is not available in {mode} mode")
            if node.type is ShellLayoutType.CANVAS_SLOT and node.mount != required:
                raise ValueError(f"canvas slot must mount '{required}' in {mode} mode")
            if node.type is ShellLayoutType.EXTENSION_MOUNT and not str(node.mount).startswith("extension:"):
                raise ValueError("extension mounts must use the extension: namespace")

    def _validate_topology(self) -> None:
        if not self.nodes or len(self.nodes) > 256:
            raise ValueError("shell layout must contain between 1 and 256 nodes")
        by_id = {node.id: node for node in self.nodes}
        if len(by_id) != len(self.nodes):
            raise ValueError("shell layout contains duplicate node IDs")
        root = by_id.get(self.root_id)
        if root is None or root.type is not ShellLayoutType.APPLICATION:
            raise ValueError("shell layout root must reference an application node")
        parents: dict[str, str] = {}
        for node in self.nodes:
            for child in node.children:
                if child not in by_id:
                    raise ValueError(f"shell layout node '{node.id}' has unknown child '{child}'")
                if child in parents:
                    raise ValueError(f"shell layout node '{child}' has multiple parents")
                parents[child] = node.id
            self._validate_node_shape(node)
        if self.root_id in parents:
            raise ValueError("shell layout root cannot be a child")
        for node in self.nodes:
            expected_parent = parents.get(node.id)
            if node.parent_id is not None and node.parent_id != expected_parent:
                raise ValueError(f"shell layout node '{node.id}' has inconsistent parent_id")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str, depth: int) -> None:
            if depth > 32:
                raise ValueError("shell layout exceeds maximum depth 32")
            if node_id in visiting:
                raise ValueError(f"shell layout contains a cycle at '{node_id}'")
            visiting.add(node_id)
            for child in by_id[node_id].children:
                visit(child, depth + 1)
            visiting.remove(node_id)
            visited.add(node_id)

        visit(self.root_id, 1)
        if len(visited) != len(by_id):
            raise ValueError("every shell layout node must be reachable from root_id")

    @staticmethod
    def _validate_node_shape(node: ShellLayoutNode) -> None:
        count = len(node.children)
        kind = node.type
        valid_count = (
            (kind in {ShellLayoutType.APPLICATION, ShellLayoutType.ROW, ShellLayoutType.COLUMN, ShellLayoutType.TOOLBAR, ShellLayoutType.STATUS_BAR, ShellLayoutType.MENU_HOST} and count >= 1)
            or (kind is ShellLayoutType.SPLIT and count == 2)
            or (kind is ShellLayoutType.TABS and count >= 1)
            or (kind in {ShellLayoutType.PANEL, ShellLayoutType.COLLAPSIBLE} and count == 1)
            or (kind in {ShellLayoutType.CANVAS_SLOT, ShellLayoutType.BUILTIN_MOUNT, ShellLayoutType.EXTENSION_MOUNT} and count == 0)
        )
        if not valid_count:
            raise ValueError(f"shell layout node '{node.id}' has invalid child count")
        if kind is ShellLayoutType.TABS:
            if node.selected_id not in node.children:
                raise ValueError(f"tabs node '{node.id}' must select one child")
        elif node.selected_id is not None:
            raise ValueError("only tabs nodes may define selected_id")
        if (kind is ShellLayoutType.SPLIT) != (node.split is not None):
            raise ValueError("split options must be defined exactly for split nodes")
        is_mount = kind in {ShellLayoutType.CANVAS_SLOT, ShellLayoutType.BUILTIN_MOUNT, ShellLayoutType.EXTENSION_MOUNT}
        if is_mount != (node.mount is not None):
            raise ValueError("mount must be defined exactly for mount nodes")

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(("root_id", "nodes"))

    def __len__(self) -> int:
        return 2


@dataclass(frozen=True)
class ShellLayoutDocument(Mapping[str, Any]):
    """Portable, versioned serialization of one application-mode layout."""

    mode: str
    layout: ShellLayout
    schema_version: int = 1
    format: str = "odon.shell-layout"

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", _validate_shell_mode(self.mode, required=True))
        if self.schema_version != 1:
            raise ValueError("ShellLayoutDocument only constructs schema version 1")
        if self.format != "odon.shell-layout":
            raise ValueError("shell layout document format must be 'odon.shell-layout'")
        self.layout.validate_for_mode(self.mode)

    @classmethod
    def from_result(cls, value: Mapping[str, Any]) -> "ShellLayoutDocument":
        raw = _require_mapping(value, "shell layout document")
        if raw.get("format") != "odon.shell-layout" or raw.get("schema_version") != 1:
            raise ValueError("unsupported shell layout document format or schema version")
        return cls(
            mode=_validate_shell_mode(raw.get("mode"), required=True),
            layout=ShellLayout.from_result(
                _require_mapping(raw.get("layout"), "shell layout document layout")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "mode": self.mode,
            "layout": self.layout.to_dict(),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(("format", "schema_version", "mode", "layout"))

    def __len__(self) -> int:
        return 4


@dataclass(frozen=True)
class ExtensionLayoutTemplate:
    """A canonical version-1 layout template owned by one extension."""

    extension_id: str
    name: str
    document: ShellLayoutDocument
    revision: int
    ownership: ShellOwnership | None = None
    extension_version: str | None = None
    readiness: str = "ready"

    @classmethod
    def from_result(cls, value: Mapping[str, Any]) -> "ExtensionLayoutTemplate":
        extension_id = value.get("extension_id")
        name = value.get("name")
        revision = value.get("revision")
        if (
            not isinstance(extension_id, str)
            or not extension_id
            or not isinstance(name, str)
            or not name
            or isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision < 1
        ):
            raise ValueError("invalid extension layout template metadata")
        readiness = str(value.get("readiness", "ready"))
        if readiness not in {"ready", "not_ready", "disconnected", "incompatible"}:
            raise ValueError("invalid extension layout template readiness")
        return cls(
            extension_id,
            name,
            ShellLayoutDocument.from_result(
                _require_mapping(value.get("document"), "extension layout document")
            ),
            revision,
            (
                ShellOwnership.from_result(
                    _require_mapping(value["ownership"], "extension layout ownership")
                )
                if value.get("ownership") is not None
                else None
            ),
            (
                str(value["extension_version"])
                if value.get("extension_version") is not None
                else None
            ),
            readiness,
        )

    def apply(
        self, shell: "Shell", *, if_revision: int | None = None
    ) -> "ShellSnapshot":
        """Atomically apply this template to its document mode."""

        if self.readiness != "ready":
            raise ValueError(
                f"extension layout template is not usable: {self.readiness}"
            )

        return shell.import_layout(
            self.document,
            mode=self.document.mode,
            if_revision=if_revision,
        )


@dataclass(frozen=True)
class ShellLayoutProfile:
    """Metadata for one named session, application, or project layout."""

    name: str
    mode: str | None
    document_schema_version: int | None
    valid: bool
    error: str | None = None
    startup_modes: tuple[str, ...] = ()
    error_kind: str | None = None
    recovery_method: str | None = None

    @classmethod
    def from_result(cls, value: Mapping[str, Any]) -> "ShellLayoutProfile":
        name = value.get("name")
        raw_mode = value.get("mode")
        version = value.get("document_schema_version")
        valid = value.get("valid", True)
        error = value.get("error")
        startup_modes = value.get("startup_modes", [])
        error_kind = value.get("error_kind")
        recovery_method = value.get("recovery_method")
        mode = (
            _validate_shell_mode(raw_mode)
            if valid
            else raw_mode if isinstance(raw_mode, str) else None
        )
        if (
            not isinstance(name, str)
            or not name
            or (raw_mode is not None and not isinstance(raw_mode, str))
            or (version is not None and (isinstance(version, bool) or not isinstance(version, int)))
            or not isinstance(valid, bool)
            or (error is not None and not isinstance(error, str))
            or not isinstance(startup_modes, list)
            or any(
                not isinstance(mode, str)
                or mode not in {"project", "single", "mosaic"}
                for mode in startup_modes
            )
            or (error_kind is not None and not isinstance(error_kind, str))
            or (recovery_method is not None and not isinstance(recovery_method, str))
        ):
            raise ValueError("invalid shell layout profile metadata")
        return cls(
            name,
            mode,
            version,
            valid,
            error,
            tuple(startup_modes),
            error_kind,
            recovery_method,
        )


@dataclass(frozen=True)
class ShellComponentDescriptor:
    """Introspected constraints for one mountable built-in GUI component."""

    id: str
    version: int
    title: str
    kind: str
    modes: tuple[str, ...]
    readiness: tuple[str, ...]
    legal_parent_types: tuple[str, ...]
    singleton: bool
    configuration_schema: Mapping[str, Any]
    commands: tuple[str, ...]
    events: tuple[str, ...]
    minimum_size: Mapping[str, float]
    recommended_size: Mapping[str, float]
    persistence: str
    ownership: ShellOwnership | None = None

    @classmethod
    def from_result(cls, value: Mapping[str, Any]) -> "ShellComponentDescriptor":
        component_id = _shell_id(value.get("id", ""))
        version = value.get("version")
        title = value.get("title")
        kind = value.get("kind")
        singleton = value.get("singleton")
        persistence = value.get("persistence")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ValueError(f"component '{component_id}' has invalid version")
        if not all(isinstance(item, str) and item for item in (title, kind, persistence)):
            raise ValueError(f"component '{component_id}' has incomplete identity")
        if not isinstance(singleton, bool):
            raise ValueError(f"component '{component_id}' has invalid singleton flag")

        def strings(name: str) -> tuple[str, ...]:
            raw = value.get(name)
            if not isinstance(raw, list) or any(not isinstance(item, str) or not item for item in raw):
                raise ValueError(f"component '{component_id}' has invalid {name}")
            return tuple(raw)

        minimum = _component_size(value.get("minimum_size"), component_id, "minimum_size")
        recommended = _component_size(
            value.get("recommended_size"), component_id, "recommended_size"
        )
        return cls(
            component_id,
            version,
            title,
            kind,
            strings("modes"),
            strings("readiness"),
            strings("legal_parent_types"),
            singleton,
            dict(_require_mapping(value.get("configuration_schema"), "configuration schema")),
            strings("commands"),
            strings("events"),
            minimum,
            recommended,
            persistence,
            (
                ShellOwnership.from_result(
                    _require_mapping(value["ownership"], "component ownership")
                )
                if value.get("ownership") is not None
                else None
            ),
        )


def _component_size(value: Any, component_id: str, label: str) -> Mapping[str, float]:
    size = _require_mapping(value, label)
    if set(size) != {"width", "height"}:
        raise ValueError(f"component '{component_id}' has invalid {label}")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(item)
        or item < 0
        for item in size.values()
    ):
        raise ValueError(f"component '{component_id}' has invalid {label}")
    return {name: float(item) for name, item in size.items()}


@dataclass(frozen=True)
class ShellMutability:
    visibility: bool
    order: bool
    selection: bool

    @classmethod
    def from_result(cls, value: Mapping[str, Any]) -> "ShellMutability":
        try:
            fields = (value["visibility"], value["order"], value["selection"])
        except (KeyError, TypeError) as error:
            raise ValueError("shell node mutability is incomplete") from error
        if not all(isinstance(item, bool) for item in fields):
            raise ValueError("shell node mutability values must be booleans")
        return cls(*fields)


@dataclass(frozen=True)
class ShellPropertyChange:
    node_id: str
    property: str
    before: Any
    after: Any

    @classmethod
    def from_result(cls, value: Mapping[str, Any]) -> "ShellPropertyChange":
        node_id = str(value.get("node_id", ""))
        property_name = str(value.get("property", ""))
        if not node_id or property_name not in {
            "visibility", "order", "selection", "size", "split", "collapse", "layout",
            "configuration", "active_region", "focus"
        }:
            raise ValueError("invalid shell property change")
        return cls(node_id, property_name, value.get("before"), value.get("after"))


@dataclass(frozen=True)
class ShellChange:
    operation: str
    mode: str
    previous_revision: int
    revision: int
    changed: bool
    transaction_id: str | None
    changes: tuple[ShellPropertyChange, ...]

    @classmethod
    def from_result(cls, value: Mapping[str, Any]) -> "ShellChange":
        operation = str(value.get("operation", ""))
        mode = _validate_shell_mode(value.get("mode"), required=True)
        previous = _validate_shell_revision(value.get("previous_revision"), required=True)
        revision = _validate_shell_revision(value.get("revision"), required=True)
        changed = value.get("changed")
        transaction_id = _validate_shell_transaction_id(value.get("transaction_id"))
        raw_changes = value.get("changes")
        if operation not in {
            "patch",
            "reset",
            "native_sync",
            "replace_layout",
            "patch_layout",
            "import_layout",
            "recover",
            "load_profile",
        }:
            raise ValueError("unknown shell change operation")
        if not isinstance(changed, bool) or not isinstance(raw_changes, list):
            raise ValueError("invalid shell change payload")
        changes = tuple(
            ShellPropertyChange.from_result(_require_mapping(item, "shell change"))
            for item in raw_changes
        )
        if changed != bool(changes):
            raise ValueError("shell change flag does not match its property changes")
        return cls(operation, mode, previous, revision, changed, transaction_id, changes)


@dataclass(frozen=True)
class ShellNode:
    id: str
    type: str
    parent_id: str | None
    content: str | None
    visible: bool
    mutable: ShellMutability
    children: tuple[str, ...]
    selected_id: str | None
    ownership: ShellOwnership | None = None

    @classmethod
    def from_result(cls, value: Mapping[str, Any]) -> "ShellNode":
        node_id = str(value.get("id", ""))
        node_type = str(value.get("type", ""))
        parent_id = value.get("parent_id")
        content = value.get("content")
        visible = value.get("visible")
        children = value.get("children")
        selected_id = value.get("selected_id")
        if not node_id or not node_type:
            raise ValueError("shell nodes require non-empty id and type")
        if parent_id is not None and not isinstance(parent_id, str):
            raise ValueError(f"parent_id for '{node_id}' must be a string or None")
        if content is not None and not isinstance(content, str):
            raise ValueError(f"content for '{node_id}' must be a string or None")
        if not isinstance(visible, bool) or not isinstance(children, list):
            raise ValueError(f"shell node '{node_id}' has invalid visibility or children")
        if any(not isinstance(child, str) or not child for child in children):
            raise ValueError(f"shell node '{node_id}' has an invalid child ID")
        if len(children) != len(set(children)):
            raise ValueError(f"shell node '{node_id}' has duplicate children")
        if selected_id is not None and not isinstance(selected_id, str):
            raise ValueError(f"selected_id for '{node_id}' must be a string or None")
        return cls(
            id=node_id,
            type=node_type,
            parent_id=parent_id,
            content=content,
            visible=visible,
            mutable=ShellMutability.from_result(
                _require_mapping(value.get("mutable"), "shell node mutability")
            ),
            children=tuple(children),
            selected_id=selected_id,
            ownership=(
                ShellOwnership.from_result(
                    _require_mapping(value["ownership"], "shell ownership")
                )
                if value.get("ownership") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class ShellSnapshot(Mapping[str, Any]):
    schema_version: int
    revision: int
    mode: str
    root_id: str
    nodes: tuple[ShellNode, ...]
    active_region_id: str
    focused_node_id: str | None
    layout: ShellLayout | None = None
    change: ShellChange | None = None
    _raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_result(cls, value: Mapping[str, Any]) -> "ShellSnapshot":
        raw = _require_mapping(value, "shell snapshot")
        schema_version = raw.get("schema_version")
        if schema_version != 1:
            raise ValueError(f"unsupported shell schema version: {schema_version!r}")
        revision = _validate_shell_revision(raw.get("revision"), required=True)
        mode = _validate_shell_mode(raw.get("mode"), required=True)
        root_id = raw.get("root_id")
        raw_nodes = raw.get("nodes")
        active_region_id = raw.get("active_region_id")
        focused_node_id = raw.get("focused_node_id")
        if not isinstance(root_id, str) or not root_id or not isinstance(raw_nodes, list):
            raise ValueError("shell snapshot requires root_id and nodes")
        if not isinstance(active_region_id, str) or not active_region_id:
            raise ValueError("shell snapshot requires active_region_id")
        if focused_node_id is not None and (
            not isinstance(focused_node_id, str) or not focused_node_id
        ):
            raise ValueError("shell focused_node_id must be a string or None")
        nodes = tuple(
            ShellNode.from_result(_require_mapping(item, "shell node"))
            for item in raw_nodes
        )
        by_id = {node.id: node for node in nodes}
        if len(by_id) != len(nodes):
            raise ValueError("shell snapshot contains duplicate node IDs")
        if root_id not in by_id or by_id[root_id].parent_id is not None:
            raise ValueError("shell root_id must reference a root node")
        for node in nodes:
            if node.parent_id is not None and node.parent_id not in by_id:
                raise ValueError(f"shell node '{node.id}' has an unknown parent")
            if any(child not in by_id for child in node.children):
                raise ValueError(f"shell node '{node.id}' has an unknown child")
            if node.selected_id is not None and node.selected_id not in node.children:
                raise ValueError(f"shell node '{node.id}' selects a non-child")
        change_value = raw.get("change")
        layout_value = raw.get("layout")
        layout = (
            ShellLayout.from_result(_require_mapping(layout_value, "shell layout"))
            if layout_value is not None
            else None
        )
        change = (
            ShellChange.from_result(_require_mapping(change_value, "shell change"))
            if change_value is not None
            else None
        )
        return cls(
            schema_version,
            revision,
            mode,
            root_id,
            nodes,
            active_region_id,
            focused_node_id,
            layout,
            change,
            dict(raw),
        )

    def node(self, node_id: str | ShellId) -> ShellNode:
        wanted = _shell_id(node_id)
        for node in self.nodes:
            if node.id == wanted:
                return node
        raise KeyError(wanted)

    def __getitem__(self, key: str) -> Any:
        return self._raw[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._raw)

    def __len__(self) -> int:
        return len(self._raw)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _validate_shell_mode(value: Any, *, required: bool = False) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or value not in {"project", "single", "mosaic"}:
        raise ValueError("mode must be 'project', 'single', or 'mosaic'")
    return value


def _validate_shell_revision(value: Any, *, required: bool = False) -> int | None:
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("shell revision must be an integer greater than or equal to 1")
    return value


def _validate_shell_transaction_id(value: Any) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > 128
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("transaction_id must contain 1 to 128 non-control bytes")
    return value


def _shell_id(value: str | ShellId) -> str:
    if isinstance(value, ShellId):
        return value.value
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("shell node IDs must contain 1 to 256 characters")
    return value


_COMMAND_STATE_PATHS = frozenset(
    {
        "mode",
        "resources.project",
        "resources.dataset",
        "resources.mosaic",
        "resources.objects",
        "resources.labels",
        "resources.masks",
        "resources.gpu",
        "selection.objects.count",
        "selection.mosaic_items.count",
        "presentation.scale_bar.checked",
        "presentation.left_panel.visible",
        "presentation.right_panel.visible",
    }
)
@dataclass(frozen=True)
class CommandPredicate:
    """One bounded condition evaluated against actor-owned command context."""

    spec: Mapping[str, Any]

    @classmethod
    def always(cls, value: bool, *, reason: str | None = None) -> "CommandPredicate":
        if not isinstance(value, bool):
            raise ValueError("always predicate value must be a boolean")
        return cls._build("always", value=value, reason=reason)

    @classmethod
    def capability(
        cls, capability: str, *, reason: str | None = None
    ) -> "CommandPredicate":
        return cls._build(
            "capability",
            capability=_command_text(capability, "predicate capability"),
            reason=reason,
        )

    @classmethod
    def state(
        cls,
        path: str,
        *,
        operator: str = "truthy",
        value: Any = None,
        reason: str | None = None,
    ) -> "CommandPredicate":
        if path not in _COMMAND_STATE_PATHS:
            raise ValueError(f"unsupported command state path: {path!r}")
        comparisons = {"equals", "not_equals", "greater_than", "at_least"}
        if operator not in {"truthy", "falsy", *comparisons}:
            raise ValueError(f"unsupported command state operator: {operator!r}")
        if (operator in comparisons) != (value is not None):
            raise ValueError(f"state predicate operator {operator!r} has incompatible value")
        if value is not None and isinstance(value, (list, tuple, dict, set)):
            raise ValueError("command state predicate value must be a scalar")
        if operator in {"greater_than", "at_least"} and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            raise ValueError("numeric command state predicate requires a number")
        return cls._build(
            "state",
            path=path,
            operator=operator,
            **({"value": value} if value is not None else {}),
            reason=reason,
        )

    @classmethod
    def all(
        cls, *predicates: "CommandPredicate | Mapping[str, Any]", reason: str | None = None
    ) -> "CommandPredicate":
        return cls._group("all", predicates, reason)

    @classmethod
    def any(
        cls, *predicates: "CommandPredicate | Mapping[str, Any]", reason: str | None = None
    ) -> "CommandPredicate":
        return cls._group("any", predicates, reason)

    @classmethod
    def not_(
        cls,
        predicate: "CommandPredicate | Mapping[str, Any]",
        *,
        reason: str | None = None,
    ) -> "CommandPredicate":
        return cls._build(
            "not", predicate=cls.from_result(predicate).to_dict(), reason=reason
        )

    @classmethod
    def from_result(
        cls, value: "CommandPredicate | Mapping[str, Any]"
    ) -> "CommandPredicate":
        if isinstance(value, cls):
            return value
        raw = dict(_require_mapping(value, "command predicate"))
        kind = raw.get("type")
        reason = raw.get("reason")
        if kind == "always":
            return cls.always(raw.get("value"), reason=reason)
        if kind == "capability":
            return cls.capability(raw.get("capability"), reason=reason)
        if kind == "state":
            return cls.state(
                raw.get("path"),
                operator=raw.get("operator", "truthy"),
                **({"value": raw["value"]} if "value" in raw else {}),
                reason=reason,
            )
        if kind in {"all", "any"}:
            children = raw.get("predicates")
            if not isinstance(children, list):
                raise ValueError(f"{kind} predicate requires a predicates array")
            return cls._group(kind, children, reason)
        if kind == "not":
            return cls.not_(
                _require_mapping(raw.get("predicate"), "not predicate child"),
                reason=reason,
            )
        raise ValueError(f"unsupported command predicate type: {kind!r}")

    @classmethod
    def _group(
        cls,
        kind: str,
        predicates: Iterable["CommandPredicate | Mapping[str, Any]"],
        reason: str | None,
    ) -> "CommandPredicate":
        checked = tuple(cls.from_result(predicate) for predicate in predicates)
        if not checked:
            raise ValueError(f"{kind} predicate requires at least one child")
        return cls._build(
            kind,
            predicates=[predicate.to_dict() for predicate in checked],
            reason=reason,
        )

    @classmethod
    def _build(cls, kind: str, *, reason: str | None = None, **data: Any) -> "CommandPredicate":
        checked_reason = (
            _command_text(reason, "predicate reason") if reason is not None else None
        )
        return cls(
            {
                "type": kind,
                **data,
                **({"reason": checked_reason} if checked_reason is not None else {}),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(self.spec)


@dataclass(frozen=True)
class CommandPredicates:
    visible: CommandPredicate | None = None
    enabled: CommandPredicate | None = None
    checked: CommandPredicate | None = None

    @classmethod
    def predicates(
        cls,
        *,
        visible: CommandPredicate | Mapping[str, Any] | None = None,
        enabled: CommandPredicate | Mapping[str, Any] | None = None,
        checked: CommandPredicate | Mapping[str, Any] | None = None,
    ) -> "CommandPredicates":
        return cls(
            CommandPredicate.from_result(visible) if visible is not None else None,
            CommandPredicate.from_result(enabled) if enabled is not None else None,
            CommandPredicate.from_result(checked) if checked is not None else None,
        )

    @classmethod
    def from_result(
        cls, value: "CommandPredicates | Mapping[str, Any] | None"
    ) -> "CommandPredicates":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        raw = _require_mapping(value, "command predicates")
        unknown = set(raw) - {"visible", "enabled", "checked"}
        if unknown:
            raise ValueError(f"unsupported command predicate slots: {sorted(unknown)!r}")
        return cls.predicates(
            visible=raw.get("visible"),
            enabled=raw.get("enabled"),
            checked=raw.get("checked"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            name: predicate.to_dict()
            for name, predicate in (
                ("visible", self.visible),
                ("enabled", self.enabled),
                ("checked", self.checked),
            )
            if predicate is not None
        }


@dataclass(frozen=True)
class ApplicationCommand(Mapping[str, Any]):
    """A stable application action, independent of where it is presented."""

    id: str
    title: str
    description: str
    handler: Mapping[str, Any]
    modes: tuple[str, ...]
    protected: bool
    predicates: CommandPredicates = field(default_factory=CommandPredicates)
    state: Mapping[str, Any] = field(default_factory=dict)
    shortcut: Mapping[str, Any] | None = None
    icon: str | None = None
    ownership: Mapping[str, Any] | None = None
    readiness: Mapping[str, Any] | None = None
    disconnect_policy: str | None = None
    _raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_result(cls, value: Mapping[str, Any]) -> "ApplicationCommand":
        raw = _require_mapping(value, "application command")
        command_id = _command_text(raw.get("id"), "command id")
        title = _command_text(raw.get("title"), "command title")
        description = _command_text(raw.get("description"), "command description")
        handler = _require_mapping(raw.get("handler"), "command handler")
        availability = _require_mapping(raw.get("availability"), "command availability")
        modes = availability.get("modes")
        if not isinstance(modes, list) or not modes or any(
            mode not in {"project", "single", "mosaic"} for mode in modes
        ):
            raise ValueError("command availability requires application modes")
        protected = raw.get("protected")
        if not isinstance(protected, bool):
            raise ValueError("command protected must be a boolean")
        shortcut_value = raw.get("shortcut")
        shortcut = (
            dict(_require_mapping(shortcut_value, "command shortcut"))
            if shortcut_value is not None
            else None
        )
        icon = raw.get("icon")
        if icon is not None and not isinstance(icon, str):
            raise ValueError("command icon must be a string or None")
        ownership_value = raw.get("ownership")
        ownership = (
            dict(_require_mapping(ownership_value, "command ownership"))
            if ownership_value is not None
            else None
        )
        readiness_value = raw.get("readiness")
        readiness = (
            dict(_require_mapping(readiness_value, "command readiness"))
            if readiness_value is not None
            else None
        )
        disconnect_policy = raw.get("disconnect_policy")
        if disconnect_policy is not None and disconnect_policy not in {
            "remove",
            "disable",
            "retain",
        }:
            raise ValueError("unsupported command disconnect policy")
        predicates = CommandPredicates.from_result(raw.get("predicates"))
        state_value = raw.get("state", {})
        state = dict(_require_mapping(state_value, "evaluated command state"))
        for field_name in ("visible", "enabled", "checkable"):
            if field_name in state and not isinstance(state[field_name], bool):
                raise ValueError(f"command state {field_name} must be a boolean")
        if state.get("checked") is not None and not isinstance(state["checked"], bool):
            raise ValueError("command state checked must be a boolean or None")
        return cls(
            id=command_id,
            title=title,
            description=description,
            handler=dict(handler),
            modes=tuple(str(mode) for mode in modes),
            protected=protected,
            predicates=predicates,
            state=state,
            shortcut=shortcut,
            icon=icon,
            ownership=ownership,
            readiness=readiness,
            disconnect_policy=disconnect_policy,
            _raw=dict(raw),
        )

    def __getitem__(self, key: str) -> Any:
        return self._raw[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._raw)

    def __len__(self) -> int:
        return len(self._raw)


@dataclass(frozen=True)
class CommandMenuNode:
    """A menu bar, nested menu, command presentation, or separator."""

    id: str
    type: str
    title: str | None = None
    command_id: str | None = None
    children: tuple["CommandMenuNode", ...] = ()
    label: str | None = None
    icon: str | None = None
    show_shortcut: bool | None = None

    @classmethod
    def from_result(cls, value: Mapping[str, Any]) -> "CommandMenuNode":
        raw = _require_mapping(value, "command menu node")
        node_id = _command_text(raw.get("id"), "menu node id")
        kind = raw.get("type")
        if kind not in {"menu_bar", "menu", "command", "separator"}:
            raise ValueError(f"unsupported command menu node type: {kind!r}")
        title = raw.get("title")
        command_id = raw.get("command_id")
        children_value = raw.get("children", [])
        if kind in {"menu_bar", "menu"}:
            if kind == "menu" and not isinstance(title, str):
                raise ValueError("menu nodes require a title")
            if not isinstance(children_value, list):
                raise ValueError("menu children must be an array")
        elif children_value:
            raise ValueError("command and separator nodes cannot have children")
        if kind == "command":
            command_id = _command_text(command_id, "menu command_id")
        elif command_id is not None:
            raise ValueError("only command menu nodes may have command_id")
        show_shortcut = raw.get("show_shortcut")
        if show_shortcut is not None and not isinstance(show_shortcut, bool):
            raise ValueError("show_shortcut must be a boolean or None")
        return cls(
            node_id,
            str(kind),
            str(title) if title is not None else None,
            str(command_id) if command_id is not None else None,
            tuple(
                cls.from_result(_require_mapping(child, "command menu child"))
                for child in children_value
            ),
            str(raw["label"]) if isinstance(raw.get("label"), str) else None,
            str(raw["icon"]) if isinstance(raw.get("icon"), str) else None,
            show_shortcut,
        )

    @classmethod
    def menu_bar(cls, id: str, children: Iterable["CommandMenuNode"]) -> "CommandMenuNode":
        return cls(_command_text(id, "menu bar id"), "menu_bar", children=tuple(children))

    @classmethod
    def menu(cls, id: str, title: str, children: Iterable["CommandMenuNode"]) -> "CommandMenuNode":
        return cls(
            _command_text(id, "menu id"),
            "menu",
            title=_command_text(title, "menu title"),
            children=tuple(children),
        )

    @classmethod
    def command(
        cls,
        id: str,
        command_id: str,
        *,
        label: str | None = None,
        icon: str | None = None,
        show_shortcut: bool | None = None,
    ) -> "CommandMenuNode":
        return cls(
            _command_text(id, "menu item id"),
            "command",
            command_id=_command_text(command_id, "command id"),
            label=label,
            icon=icon,
            show_shortcut=show_shortcut,
        )

    @classmethod
    def separator(cls, id: str) -> "CommandMenuNode":
        return cls(_command_text(id, "menu separator id"), "separator")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            **({"title": self.title} if self.title is not None else {}),
            **({"command_id": self.command_id} if self.command_id is not None else {}),
            **(
                {"children": [child.to_dict() for child in self.children]}
                if self.type in {"menu_bar", "menu"}
                else {}
            ),
            **({"label": self.label} if self.label is not None else {}),
            **({"icon": self.icon} if self.icon is not None else {}),
            **(
                {"show_shortcut": self.show_shortcut}
                if self.show_shortcut is not None
                else {}
            ),
        }


@dataclass(frozen=True)
class CommandMenuSnapshot(Mapping[str, Any]):
    schema_version: int
    revision: int
    menu: CommandMenuNode
    change: Mapping[str, Any] | None = None
    _raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_result(cls, value: Mapping[str, Any]) -> "CommandMenuSnapshot":
        raw = _require_mapping(value, "command menu snapshot")
        if raw.get("schema_version") != 1:
            raise ValueError("unsupported command menu schema version")
        revision = _validate_command_revision(raw.get("revision"), required=True)
        menu = CommandMenuNode.from_result(_require_mapping(raw.get("menu"), "command menu"))
        if menu.type != "menu_bar":
            raise ValueError("command menu root must be a menu_bar")
        change_value = raw.get("change")
        change = (
            dict(_require_mapping(change_value, "command menu change"))
            if change_value is not None
            else None
        )
        return cls(1, revision, menu, change, dict(raw))

    def __getitem__(self, key: str) -> Any:
        return self._raw[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._raw)

    def __len__(self) -> int:
        return len(self._raw)


@dataclass(frozen=True)
class CommandToolbarItem:
    id: str
    command_id: str
    label: str | None = None
    icon: str | None = None
    tooltip: str | None = None
    show_label: bool = True

    @classmethod
    def command(
        cls,
        id: str,
        command: ApplicationCommand | str,
        *,
        label: str | None = None,
        icon: str | None = None,
        tooltip: str | None = None,
        show_label: bool = True,
    ) -> "CommandToolbarItem":
        if not isinstance(show_label, bool):
            raise ValueError("toolbar item show_label must be a boolean")
        command_id = command.id if isinstance(command, ApplicationCommand) else command
        return cls(
            _command_text(id, "toolbar item id"),
            _command_text(command_id, "toolbar command id"),
            _command_text(label, "toolbar item label") if label is not None else None,
            _command_text(icon, "toolbar item icon") if icon is not None else None,
            _command_text(tooltip, "toolbar item tooltip")
            if tooltip is not None
            else None,
            show_label,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "command_id": self.command_id,
            **({"label": self.label} if self.label is not None else {}),
            **({"icon": self.icon} if self.icon is not None else {}),
            **({"tooltip": self.tooltip} if self.tooltip is not None else {}),
            "show_label": self.show_label,
        }

    @classmethod
    def from_result(cls, value: Mapping[str, Any]) -> "CommandToolbarItem":
        raw = _require_mapping(value, "command toolbar item")
        return cls.command(
            _command_text(raw.get("id"), "toolbar item id"),
            _command_text(raw.get("command_id"), "toolbar command id"),
            label=raw.get("label"),
            icon=raw.get("icon"),
            tooltip=raw.get("tooltip"),
            show_label=raw.get("show_label", True),
        )


@dataclass(frozen=True)
class CommandToolbarGroup:
    id: str
    items: tuple[CommandToolbarItem, ...]
    title: str | None = None

    @classmethod
    def group(
        cls,
        id: str,
        items: Iterable[CommandToolbarItem],
        *,
        title: str | None = None,
    ) -> "CommandToolbarGroup":
        return cls(
            _command_text(id, "toolbar group id"),
            tuple(items),
            _command_text(title, "toolbar group title") if title is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "items": [item.to_dict() for item in self.items],
            **({"title": self.title} if self.title is not None else {}),
        }

    @classmethod
    def from_result(cls, value: Mapping[str, Any]) -> "CommandToolbarGroup":
        raw = _require_mapping(value, "command toolbar group")
        items = raw.get("items")
        if not isinstance(items, list):
            raise ValueError("command toolbar group requires an items array")
        return cls.group(
            _command_text(raw.get("id"), "toolbar group id"),
            (
                CommandToolbarItem.from_result(
                    _require_mapping(item, "command toolbar item")
                )
                for item in items
            ),
            title=raw.get("title"),
        )


@dataclass(frozen=True)
class CommandToolbar:
    id: str
    groups: tuple[CommandToolbarGroup, ...]

    @classmethod
    def toolbar(
        cls, id: str, groups: Iterable[CommandToolbarGroup]
    ) -> "CommandToolbar":
        return cls(_command_text(id, "toolbar id"), tuple(groups))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "toolbar",
            "groups": [group.to_dict() for group in self.groups],
        }

    @classmethod
    def from_result(cls, value: Mapping[str, Any]) -> "CommandToolbar":
        raw = _require_mapping(value, "command toolbar")
        if raw.get("type") != "toolbar":
            raise ValueError("command toolbar type must be 'toolbar'")
        groups = raw.get("groups")
        if not isinstance(groups, list):
            raise ValueError("command toolbar requires a groups array")
        return cls.toolbar(
            _command_text(raw.get("id"), "toolbar id"),
            (
                CommandToolbarGroup.from_result(
                    _require_mapping(group, "command toolbar group")
                )
                for group in groups
            ),
        )


@dataclass(frozen=True)
class CommandToolbarSnapshot(Mapping[str, Any]):
    schema_version: int
    revision: int
    toolbar: CommandToolbar
    change: Mapping[str, Any] | None = None
    _raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_result(cls, value: Mapping[str, Any]) -> "CommandToolbarSnapshot":
        raw = _require_mapping(value, "command toolbar snapshot")
        if raw.get("schema_version") != 1:
            raise ValueError("unsupported command toolbar schema version")
        change_value = raw.get("change")
        return cls(
            1,
            _validate_command_revision(raw.get("revision"), required=True),
            CommandToolbar.from_result(
                _require_mapping(raw.get("toolbar"), "command toolbar")
            ),
            dict(_require_mapping(change_value, "command toolbar change"))
            if change_value is not None
            else None,
            dict(raw),
        )

    def __getitem__(self, key: str) -> Any:
        return self._raw[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._raw)

    def __len__(self) -> int:
        return len(self._raw)


@dataclass(frozen=True)
class CommandPalette:
    id: str
    title: str
    placeholder: str
    shortcut: Mapping[str, Any]
    show_descriptions: bool = True
    max_results: int = 20

    @classmethod
    def palette(
        cls,
        id: str,
        *,
        title: str = "Commands",
        placeholder: str = "Search commands…",
        shortcut: Mapping[str, Any] | None = None,
        show_descriptions: bool = True,
        max_results: int = 20,
    ) -> "CommandPalette":
        if not isinstance(show_descriptions, bool):
            raise ValueError("palette show_descriptions must be a boolean")
        if (
            isinstance(max_results, bool)
            or not isinstance(max_results, int)
            or not 1 <= max_results <= 100
        ):
            raise ValueError("palette max_results must be an integer between 1 and 100")
        if shortcut is None:
            shortcut = {"key": "p", "modifiers": ["primary", "shift"]}
        checked_shortcut = dict(_require_mapping(shortcut, "palette shortcut"))
        if not isinstance(checked_shortcut.get("key"), str) or not isinstance(
            checked_shortcut.get("modifiers"), list
        ):
            raise ValueError("palette shortcut requires key and modifiers")
        return cls(
            _command_text(id, "palette id"),
            _command_text(title, "palette title"),
            _command_text(placeholder, "palette placeholder"),
            checked_shortcut,
            show_descriptions,
            max_results,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "command_palette",
            "title": self.title,
            "placeholder": self.placeholder,
            "shortcut": dict(self.shortcut),
            "show_descriptions": self.show_descriptions,
            "max_results": self.max_results,
        }

    @classmethod
    def from_result(cls, value: Mapping[str, Any]) -> "CommandPalette":
        raw = _require_mapping(value, "command palette")
        if raw.get("type") != "command_palette":
            raise ValueError("command palette type must be 'command_palette'")
        return cls.palette(
            _command_text(raw.get("id"), "palette id"),
            title=_command_text(raw.get("title"), "palette title"),
            placeholder=_command_text(raw.get("placeholder"), "palette placeholder"),
            shortcut=_require_mapping(raw.get("shortcut"), "palette shortcut"),
            show_descriptions=raw.get("show_descriptions", True),
            max_results=raw.get("max_results", 20),
        )


@dataclass(frozen=True)
class CommandPaletteSnapshot(Mapping[str, Any]):
    schema_version: int
    revision: int
    palette: CommandPalette
    change: Mapping[str, Any] | None = None
    _raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_result(cls, value: Mapping[str, Any]) -> "CommandPaletteSnapshot":
        raw = _require_mapping(value, "command palette snapshot")
        if raw.get("schema_version") != 1:
            raise ValueError("unsupported command palette schema version")
        change_value = raw.get("change")
        return cls(
            1,
            _validate_command_revision(raw.get("revision"), required=True),
            CommandPalette.from_result(
                _require_mapping(raw.get("palette"), "command palette")
            ),
            dict(_require_mapping(change_value, "command palette change"))
            if change_value is not None
            else None,
            dict(raw),
        )

    def __getitem__(self, key: str) -> Any:
        return self._raw[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._raw)

    def __len__(self) -> int:
        return len(self._raw)


def _command_text(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > 256
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{field} must contain 1 to 256 non-control bytes")
    return value


def _extension_command_register_params(
    extension_id: str,
    command_id: str,
    title: str,
    description: str,
    event: str,
    *,
    modes: Iterable[str],
    shortcut: Mapping[str, Any] | None,
    icon: str | None,
    predicates: CommandPredicates | Mapping[str, Any] | None,
    if_revision: int | None,
    transaction_id: str | None,
) -> dict[str, Any]:
    local_id = _command_text(command_id, "command id")
    if (
        local_id.startswith(".")
        or local_id.endswith(".")
        or any(not (character.isascii() and (character.isalnum() or character in "._-")) for character in local_id)
    ):
        raise ValueError(
            "command id must use ASCII letters, digits, '.', '_', or '-' and cannot begin or end with '.'"
        )
    checked_modes = tuple(modes)
    if not checked_modes or len(set(checked_modes)) != len(checked_modes) or any(
        mode not in {"project", "single", "mosaic"} for mode in checked_modes
    ):
        raise ValueError("command modes must be unique application modes")
    command_spec: dict[str, Any] = {
        "id": local_id,
        "title": _command_text(title, "command title"),
        "description": _command_text(description, "command description"),
        "event": _command_text(event, "command event"),
        "modes": list(checked_modes),
        "shortcut": dict(shortcut) if shortcut is not None else None,
        "icon": _command_text(icon, "command icon") if icon is not None else None,
        "predicates": CommandPredicates.from_result(predicates).to_dict(),
    }
    params: dict[str, Any] = {
        "extension_id": _command_text(extension_id, "extension id"),
        "command": command_spec,
    }
    revision = _validate_command_revision(if_revision)
    if revision is not None:
        params["if_command_revision"] = revision
    checked_transaction = _validate_shell_transaction_id(transaction_id)
    if checked_transaction is not None:
        params["transaction_id"] = checked_transaction
    return params


def _validate_command_revision(value: Any, *, required: bool = False) -> int | None:
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("command revision must be an integer greater than or equal to 1")
    return value


class Commands:
    def __init__(self, ui: "Ui") -> None:
        self._ui = ui

    def describe_schema(self) -> Mapping[str, Any]:
        return self._ui._client.call("ui.commands.describe_schema")

    def list(self) -> tuple[ApplicationCommand, ...]:
        result = _require_mapping(
            self._ui._client.call("ui.commands.list"), "application command list"
        )
        commands = result.get("commands")
        if not isinstance(commands, list):
            raise ValueError("application command list requires a commands array")
        return tuple(
            ApplicationCommand.from_result(_require_mapping(command, "application command"))
            for command in commands
        )

    def execute(
        self, command: ApplicationCommand | str, *, checked: bool | None = None
    ) -> Mapping[str, Any]:
        """Invoke a ready native, control, or extension-event command."""

        command_id = command.id if isinstance(command, ApplicationCommand) else command
        params: dict[str, Any] = {
            "command_id": _command_text(command_id, "command id")
        }
        if checked is not None:
            if not isinstance(checked, bool):
                raise ValueError("checked must be a boolean")
            params["checked"] = checked
        return self._ui._client.call(
            "ui.commands.execute",
            params,
        )


class Menus:
    def __init__(self, ui: "Ui") -> None:
        self._ui = ui

    def get(self) -> CommandMenuSnapshot:
        return CommandMenuSnapshot.from_result(self._ui._client.call("ui.menus.get"))

    def replace(
        self,
        menu: CommandMenuNode | Mapping[str, Any],
        *,
        if_revision: int | None = None,
        transaction_id: str | None = None,
    ) -> CommandMenuSnapshot:
        raw = menu.to_dict() if isinstance(menu, CommandMenuNode) else dict(menu)
        checked = CommandMenuNode.from_result(raw)
        if checked.type != "menu_bar":
            raise ValueError("platform menu root must be a menu_bar")
        params: dict[str, Any] = {"menu": checked.to_dict()}
        revision = _validate_command_revision(if_revision)
        if revision is not None:
            params["if_command_revision"] = revision
        checked_transaction = _validate_shell_transaction_id(transaction_id)
        if checked_transaction is not None:
            params["transaction_id"] = checked_transaction
        return CommandMenuSnapshot.from_result(
            self._ui._client.call("ui.menus.replace", params)
        )


class Toolbars:
    def __init__(self, ui: "Ui") -> None:
        self._ui = ui

    def get(self) -> CommandToolbarSnapshot:
        return CommandToolbarSnapshot.from_result(
            self._ui._client.call("ui.toolbars.get")
        )

    def replace(
        self,
        toolbar: CommandToolbar | Mapping[str, Any],
        *,
        if_revision: int | None = None,
        transaction_id: str | None = None,
    ) -> CommandToolbarSnapshot:
        raw = toolbar.to_dict() if isinstance(toolbar, CommandToolbar) else dict(toolbar)
        checked = CommandToolbar.from_result(raw)
        params: dict[str, Any] = {"toolbar": checked.to_dict()}
        revision = _validate_command_revision(if_revision)
        if revision is not None:
            params["if_command_revision"] = revision
        checked_transaction = _validate_shell_transaction_id(transaction_id)
        if checked_transaction is not None:
            params["transaction_id"] = checked_transaction
        return CommandToolbarSnapshot.from_result(
            self._ui._client.call("ui.toolbars.replace", params)
        )


class Palette:
    def __init__(self, ui: "Ui") -> None:
        self._ui = ui

    def get(self) -> CommandPaletteSnapshot:
        return CommandPaletteSnapshot.from_result(
            self._ui._client.call("ui.palette.get")
        )

    def replace(
        self,
        palette: CommandPalette | Mapping[str, Any],
        *,
        if_revision: int | None = None,
        transaction_id: str | None = None,
    ) -> CommandPaletteSnapshot:
        raw = palette.to_dict() if isinstance(palette, CommandPalette) else dict(palette)
        checked = CommandPalette.from_result(raw)
        params: dict[str, Any] = {"palette": checked.to_dict()}
        revision = _validate_command_revision(if_revision)
        if revision is not None:
            params["if_command_revision"] = revision
        checked_transaction = _validate_shell_transaction_id(transaction_id)
        if checked_transaction is not None:
            params["transaction_id"] = checked_transaction
        return CommandPaletteSnapshot.from_result(
            self._ui._client.call("ui.palette.replace", params)
        )


class Shell:
    """Inspect and compose Odon's native application shell."""

    def __init__(self, ui: "Ui") -> None:
        self._ui = ui

    def describe_schema(self) -> Mapping[str, Any]:
        return self._ui._client.call("ui.shell.describe_schema")

    def get(self, *, mode: str | None = None) -> ShellSnapshot:
        return ShellSnapshot.from_result(
            self._ui._client.call("ui.shell.get", self._get_params(mode))
        )

    def list_components(
        self, *, mode: str | None = None
    ) -> tuple[ShellComponentDescriptor, ...]:
        """Discover built-in component mounts and their layout constraints."""

        result = self._ui._client.call(
            "ui.shell.components.list", self._get_params(mode)
        )
        raw = _require_mapping(result, "shell component catalogue").get("components")
        if not isinstance(raw, list):
            raise ValueError("shell component catalogue requires a components array")
        return tuple(
            ShellComponentDescriptor.from_result(
                _require_mapping(component, "shell component descriptor")
            )
            for component in raw
        )

    def export_layout(self, *, mode: str | None = None) -> ShellLayoutDocument:
        """Export one mode as a portable versioned layout document."""

        return ShellLayoutDocument.from_result(
            self._ui._client.call("ui.shell.export_layout", self._get_params(mode))
        )

    def import_layout(
        self,
        document: ShellLayoutDocument | Mapping[str, Any],
        *,
        mode: str | None = None,
        if_revision: int | None = None,
        transaction_id: str | None = None,
    ) -> ShellSnapshot:
        """Atomically validate, migrate, and import a layout document."""

        return ShellSnapshot.from_result(
            self._ui._client.call(
                "ui.shell.import_layout",
                self._import_layout_params(document, mode, if_revision, transaction_id),
            )
        )

    def list_profiles(
        self, *, scope: str = "session"
    ) -> tuple[ShellLayoutProfile, ...]:
        """List named layouts from session, application, or project scope."""

        result = self._ui._client.call(
            "ui.shell.profiles.list", self._profile_scope_params(scope)
        )
        profiles = _require_mapping(result, "shell profile list").get("profiles")
        if not isinstance(profiles, list):
            raise ValueError("shell profile list requires a profiles array")
        return tuple(
            ShellLayoutProfile.from_result(_require_mapping(item, "shell profile"))
            for item in profiles
        )

    def save_profile(
        self,
        name: str,
        *,
        scope: str = "session",
        mode: str | None = None,
    ) -> Mapping[str, Any]:
        """Save the current layout under a session, application, or project name."""

        return self._ui._client.call(
            "ui.shell.profiles.save", self._profile_params(name, scope, mode)
        )

    def load_profile(
        self,
        name: str,
        *,
        scope: str = "session",
        mode: str | None = None,
        if_revision: int | None = None,
        transaction_id: str | None = None,
    ) -> ShellSnapshot:
        """Atomically load a named layout into the active mode."""

        params = self._profile_params(name, scope, mode)
        checked_revision = _validate_shell_revision(if_revision)
        if checked_revision is not None:
            params["if_shell_revision"] = checked_revision
        checked_transaction_id = _validate_shell_transaction_id(transaction_id)
        if checked_transaction_id is not None:
            params["transaction_id"] = checked_transaction_id
        return ShellSnapshot.from_result(
            self._ui._client.call("ui.shell.profiles.load", params)
        )

    def remove_profile(
        self, name: str, *, scope: str = "session"
    ) -> Mapping[str, Any]:
        """Remove a named session, application, or project layout."""

        params = self._profile_params(name, scope, None)
        return self._ui._client.call("ui.shell.profiles.remove", params)

    def patch(
        self,
        *,
        visibility: Mapping[str | ShellId, bool] | None = None,
        orders: Mapping[str | ShellId, Iterable[str | ShellId]] | None = None,
        selected: Mapping[str | ShellId, str | ShellId] | None = None,
        mode: str | None = None,
        if_revision: int | None = None,
        transaction_id: str | None = None,
    ) -> ShellSnapshot:
        return ShellSnapshot.from_result(
            self._ui._client.call(
                "ui.shell.patch",
                self._patch_params(
                    visibility=visibility,
                    orders=orders,
                    selected=selected,
                    mode=mode,
                    if_revision=if_revision,
                    transaction_id=transaction_id,
                ),
            )
        )

    def reset(
        self, *, mode: str | None = None, if_revision: int | None = None,
        transaction_id: str | None = None,
    ) -> ShellSnapshot:
        return ShellSnapshot.from_result(
            self._ui._client.call(
                "ui.shell.reset", self._reset_params(mode, if_revision, transaction_id)
            )
        )

    def recover(
        self, *, mode: str | None = None, if_revision: int | None = None,
        transaction_id: str | None = None,
    ) -> ShellSnapshot:
        """Replace the active shell with Odon's protected minimal recovery layout."""

        return ShellSnapshot.from_result(
            self._ui._client.call(
                "ui.shell.recover", self._reset_params(mode, if_revision, transaction_id)
            )
        )

    def replace_layout(
        self,
        layout: ShellLayout | Mapping[str, Any],
        *,
        mode: str | None = None,
        if_revision: int | None = None,
        transaction_id: str | None = None,
    ) -> ShellSnapshot:
        """Atomically replace the active mode's complete desired layout tree."""

        return ShellSnapshot.from_result(
            self._ui._client.call(
                "ui.shell.replace_layout",
                self._replace_layout_params(layout, mode, if_revision, transaction_id),
            )
        )

    def patch_layout(
        self,
        *,
        visibility: Mapping[str, bool] | None = None,
        selected: Mapping[str, str] | None = None,
        sizes: Mapping[str, ShellSize | Mapping[str, Any]] | None = None,
        splits: Mapping[str, ShellSplit | Mapping[str, Any]] | None = None,
        collapsed: Mapping[str, bool] | None = None,
        configurations: Mapping[str, Mapping[str, Any]] | None = None,
        active_region_id: str | None = None,
        focused_node_id: str | None = None,
        clear_focus: bool = False,
        mode: str | None = None,
        if_revision: int | None = None,
        transaction_id: str | None = None,
    ) -> ShellSnapshot:
        """Atomically update interactive state within the active desired tree."""

        return ShellSnapshot.from_result(
            self._ui._client.call(
                "ui.shell.patch_layout",
                self._patch_layout_params(
                    visibility=visibility,
                    selected=selected,
                    sizes=sizes,
                    splits=splits,
                    collapsed=collapsed,
                    configurations=configurations,
                    active_region_id=active_region_id,
                    focused_node_id=focused_node_id,
                    clear_focus=clear_focus,
                    mode=mode,
                    if_revision=if_revision,
                    transaction_id=transaction_id,
                ),
            )
        )

    @staticmethod
    def _get_params(mode: str | None) -> dict[str, Any]:
        checked_mode = _validate_shell_mode(mode)
        return {**({"mode": checked_mode} if checked_mode is not None else {})}

    @staticmethod
    def _profile_scope_params(scope: str) -> dict[str, Any]:
        if scope not in {"session", "application", "project"}:
            raise ValueError(
                "shell profile scope must be 'session', 'application', or 'project'"
            )
        return {"scope": scope}

    @classmethod
    def _profile_params(
        cls, name: str, scope: str, mode: str | None
    ) -> dict[str, Any]:
        if (
            not isinstance(name, str)
            or not name.strip()
            or len(name) > 128
            or any(ord(character) < 32 or ord(character) == 127 for character in name)
        ):
            raise ValueError(
                "shell profile names must contain 1 to 128 characters without control characters"
            )
        params = {"name": name, **cls._profile_scope_params(scope)}
        checked_mode = _validate_shell_mode(mode)
        if checked_mode is not None:
            params["mode"] = checked_mode
        return params

    @staticmethod
    def _patch_params(
        *,
        visibility: Mapping[str | ShellId, bool] | None,
        orders: Mapping[str | ShellId, Iterable[str | ShellId]] | None,
        selected: Mapping[str | ShellId, str | ShellId] | None,
        mode: str | None,
        if_revision: int | None,
        transaction_id: str | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if visibility is not None:
            checked_visibility: dict[str, bool] = {}
            for node_id, visible in visibility.items():
                if not isinstance(visible, bool):
                    raise ValueError("shell visibility values must be booleans")
                checked_visibility[_shell_id(node_id)] = visible
            params["visibility"] = checked_visibility
        if orders is not None:
            checked_orders: dict[str, list[str]] = {}
            for parent_id, child_ids in orders.items():
                if isinstance(child_ids, (str, bytes)):
                    raise ValueError("shell orders must contain iterables of child IDs")
                children = [_shell_id(child_id) for child_id in child_ids]
                if len(children) != len(set(children)):
                    raise ValueError("shell orders cannot contain duplicate child IDs")
                checked_orders[_shell_id(parent_id)] = children
            params["orders"] = checked_orders
        if selected is not None:
            params["selected"] = {
                _shell_id(parent_id): _shell_id(child_id)
                for parent_id, child_id in selected.items()
            }
        checked_mode = _validate_shell_mode(mode)
        checked_revision = _validate_shell_revision(if_revision)
        if checked_mode is not None:
            params["mode"] = checked_mode
        if checked_revision is not None:
            params["if_shell_revision"] = checked_revision
        checked_transaction_id = _validate_shell_transaction_id(transaction_id)
        if checked_transaction_id is not None:
            params["transaction_id"] = checked_transaction_id
        return params

    @staticmethod
    def _reset_params(
        mode: str | None,
        if_revision: int | None,
        transaction_id: str | None = None,
    ) -> dict[str, Any]:
        checked_mode = _validate_shell_mode(mode)
        checked_revision = _validate_shell_revision(if_revision)
        return {
            **({"mode": checked_mode} if checked_mode is not None else {}),
            **(
                {"if_shell_revision": checked_revision}
                if checked_revision is not None
                else {}
            ),
            **(
                {"transaction_id": checked_transaction_id}
                if (checked_transaction_id := _validate_shell_transaction_id(transaction_id))
                is not None
                else {}
            ),
        }

    @staticmethod
    def _replace_layout_params(
        layout: ShellLayout | Mapping[str, Any],
        mode: str | None,
        if_revision: int | None,
        transaction_id: str | None,
    ) -> dict[str, Any]:
        checked_mode = _validate_shell_mode(mode)
        checked_revision = _validate_shell_revision(if_revision)
        checked_layout = (
            layout
            if isinstance(layout, ShellLayout)
            else ShellLayout.from_result(_require_mapping(layout, "shell layout"))
        )
        if checked_mode is not None:
            checked_layout.validate_for_mode(checked_mode)
        return {
            "desired_tree": checked_layout.to_dict(),
            **({"mode": checked_mode} if checked_mode is not None else {}),
            **(
                {"if_shell_revision": checked_revision}
                if checked_revision is not None
                else {}
            ),
            **(
                {"transaction_id": checked_transaction_id}
                if (checked_transaction_id := _validate_shell_transaction_id(transaction_id))
                is not None
                else {}
            ),
        }

    @staticmethod
    def _import_layout_params(
        document: ShellLayoutDocument | Mapping[str, Any],
        mode: str | None,
        if_revision: int | None,
        transaction_id: str | None,
    ) -> dict[str, Any]:
        checked_mode = _validate_shell_mode(mode)
        checked_revision = _validate_shell_revision(if_revision)
        raw_document = _extension_layout_document(document)
        if checked_mode is not None:
            document_mode = raw_document.get("mode")
            if isinstance(document_mode, str) and document_mode != checked_mode:
                raise ValueError("shell layout document mode does not match requested mode")
        return {
            "document": raw_document,
            **({"mode": checked_mode} if checked_mode is not None else {}),
            **(
                {"if_shell_revision": checked_revision}
                if checked_revision is not None
                else {}
            ),
            **(
                {"transaction_id": checked_transaction_id}
                if (checked_transaction_id := _validate_shell_transaction_id(transaction_id))
                is not None
                else {}
            ),
        }

    @staticmethod
    def _patch_layout_params(
        *,
        visibility: Mapping[str, bool] | None,
        selected: Mapping[str, str] | None,
        sizes: Mapping[str, ShellSize | Mapping[str, Any]] | None,
        splits: Mapping[str, ShellSplit | Mapping[str, Any]] | None,
        collapsed: Mapping[str, bool] | None,
        configurations: Mapping[str, Mapping[str, Any]] | None,
        active_region_id: str | None,
        focused_node_id: str | None,
        clear_focus: bool,
        mode: str | None,
        if_revision: int | None,
        transaction_id: str | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if visibility is not None:
            params["visibility"] = _shell_bool_map(visibility, "visibility")
        if selected is not None:
            params["selected"] = {
                _shell_id(node_id): _shell_id(child_id)
                for node_id, child_id in selected.items()
            }
        if sizes is not None:
            params["sizes"] = {
                _shell_id(node_id): (
                    value.to_dict()
                    if isinstance(value, ShellSize)
                    else ShellSize.from_result(
                        _require_mapping(value, "shell size")
                    ).to_dict()
                )
                for node_id, value in sizes.items()
            }
        if splits is not None:
            params["splits"] = {
                _shell_id(node_id): (
                    value.to_dict()
                    if isinstance(value, ShellSplit)
                    else ShellSplit.from_result(
                        _require_mapping(value, "shell split")
                    ).to_dict()
                )
                for node_id, value in splits.items()
            }
        if collapsed is not None:
            params["collapsed"] = _shell_bool_map(collapsed, "collapsed")
        if configurations is not None:
            params["configurations"] = {
                _shell_id(node_id): dict(
                    _require_mapping(configuration, "shell mount configuration")
                )
                for node_id, configuration in configurations.items()
            }
        if active_region_id is not None:
            params["active_region_id"] = _shell_id(active_region_id)
        if focused_node_id is not None:
            params["focused_node_id"] = _shell_id(focused_node_id)
        if clear_focus:
            if focused_node_id is not None:
                raise ValueError("clear_focus and focused_node_id are mutually exclusive")
            params["clear_focus"] = True
        checked_mode = _validate_shell_mode(mode)
        checked_revision = _validate_shell_revision(if_revision)
        if checked_mode is not None:
            params["mode"] = checked_mode
        if checked_revision is not None:
            params["if_shell_revision"] = checked_revision
        checked_transaction_id = _validate_shell_transaction_id(transaction_id)
        if checked_transaction_id is not None:
            params["transaction_id"] = checked_transaction_id
        return params


def _shell_bool_map(values: Mapping[str, bool], label: str) -> dict[str, bool]:
    checked: dict[str, bool] = {}
    for node_id, value in values.items():
        if not isinstance(value, bool):
            raise ValueError(f"shell layout {label} values must be booleans")
        checked[_shell_id(node_id)] = value
    return checked


def _extension_layout_name(name: str) -> str:
    if (
        not isinstance(name, str)
        or not name.strip()
        or len(name) > 128
        or any(unicodedata.category(character) == "Cc" for character in name)
    ):
        raise ValueError(
            "extension layout name must contain 1 to 128 characters without control characters"
        )
    return name


def _extension_layout_document(
    document: ShellLayoutDocument | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(document, ShellLayoutDocument):
        return document.to_dict()
    return dict(_require_mapping(document, "shell layout document"))


class Ui:
    def __init__(self, client: "Client") -> None:
        self._client = client
        self.commands = Commands(self)
        self.menus = Menus(self)
        self.toolbars = Toolbars(self)
        self.palette = Palette(self)
        self.shell = Shell(self)

    def register_extension(
        self,
        *,
        id: str,
        name: str,
        version: str,
        capabilities: Iterable[str] = ("ui.panels",),
        disconnect_policy: str = "remove",
        ready: bool = True,
        readiness_reason: str | None = None,
    ) -> Extension:
        if not isinstance(ready, bool):
            raise ValueError("extension readiness must be a boolean")
        result = self._client.call(
            "ui.extensions.register",
            {
                "id": id,
                "name": name,
                "version": version,
                "capabilities": list(capabilities),
                "disconnect_policy": disconnect_policy,
                "ready":ready,
                **(
                    {"readiness_reason":readiness_reason}
                    if readiness_reason is not None
                    else {}
                ),
            },
        )
        return Extension(self, result)

    def list_extensions(self) -> list[Mapping[str, Any]]:
        return self._client.call("ui.extensions.list")["extensions"]

    def list_contributions(self) -> list[Mapping[str, Any]]:
        return self._client.call("ui.contributions.list")["contributions"]

    def describe_schema(self) -> Mapping[str, Any]:
        return self._client.call("ui.describe_schema")
