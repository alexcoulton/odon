"""Versioned declarative UI components rendered natively by Odon/egui."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Mapping

if TYPE_CHECKING:
    from .client import Client


def emit(event: str, **data: Any) -> dict[str, Any]:
    return {"type": "emit", "event": event, "data": data}


def command(method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"type": "command", "method": method, "params": dict(params or {})}


def bind(target: str, **selector: Any) -> dict[str, Any]:
    return {"type": "bind", "target": target, **selector}


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
    children: list["Component"] = field(default_factory=list)

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

    def register(
        self,
        root: Component,
        *,
        location: str = "right.tabs",
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

    def remove(self) -> None:
        self._ui._client.call("ui.extensions.remove", {"extension_id": self.id})


class Ui:
    def __init__(self, client: "Client") -> None:
        self._client = client

    def register_extension(
        self,
        *,
        id: str,
        name: str,
        version: str,
        capabilities: Iterable[str] = ("ui.panels",),
        disconnect_policy: str = "remove",
    ) -> Extension:
        result = self._client.call(
            "ui.extensions.register",
            {
                "id": id,
                "name": name,
                "version": version,
                "capabilities": list(capabilities),
                "disconnect_policy": disconnect_policy,
            },
        )
        return Extension(self, result)

    def list_extensions(self) -> list[Mapping[str, Any]]:
        return self._client.call("ui.extensions.list")["extensions"]

    def list_contributions(self) -> list[Mapping[str, Any]]:
        return self._client.call("ui.contributions.list")["contributions"]

    def describe_schema(self) -> Mapping[str, Any]:
        return self._client.call("ui.describe_schema")
