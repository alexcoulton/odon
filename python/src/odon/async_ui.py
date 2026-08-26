"""Async registration wrappers for declarative Odon UI."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any
import weakref

from .ui import (
    ApplicationCommand,
    CommandMenuNode,
    CommandMenuSnapshot,
    CommandPalette,
    CommandPaletteSnapshot,
    CommandPredicates,
    CommandToolbar,
    CommandToolbarSnapshot,
    Component,
    ExtensionLayoutTemplate,
    Shell,
    ShellComponentDescriptor,
    ShellId,
    ShellLayout,
    ShellLayoutDocument,
    ShellLayoutProfile,
    ShellLayoutNode,
    ShellSnapshot,
    _extension_layout_document,
    _extension_layout_name,
    _extension_command_register_params,
    _require_mapping,
    _validate_command_revision,
    _validate_shell_transaction_id,
)
from .models import Event
from .ui_actions import (
    AsyncActionContext,
    AsyncActionRegistration,
    AsyncActionRunner,
    AsyncInteractionSubscription,
    ActionWorkerSnapshot,
    CoalescePolicy,
    ExecutionPolicy,
    UiInteraction,
    UiInteractionDecodeError,
    _validate_coalesce,
    _validate_execution,
)

import logging

if TYPE_CHECKING:
    from .async_client import AsyncClient

logger = logging.getLogger("odon.ui")


class AsyncContribution:
    def __init__(self, extension: "AsyncExtension", snapshot: Mapping[str, Any]) -> None:
        self._extension = extension
        self.snapshot = dict(snapshot)

    @property
    def contribution_id(self) -> str:
        return str(self.snapshot["contribution_id"])

    @property
    def shell_mount(self) -> str:
        return str(self.snapshot["shell_mount"])

    def mount(self, node_id: str, **kwargs: Any) -> ShellLayoutNode:
        """Create a keyed shell node that renders this contribution."""
        return ShellLayoutNode.extension(node_id, self.shell_mount, **kwargs)

    async def patch_values(
        self, values: Mapping[str, Any], *, if_revision: int | None = None
    ) -> "AsyncContribution":
        result = await self._extension._ui._client.call(
            "ui.contributions.patch_values",
            {
                "contribution_id": self.contribution_id,
                "values": dict(values),
                **({"if_revision": if_revision} if if_revision is not None else {}),
            },
        )
        self.snapshot = dict(result)
        return self

    async def remove(self) -> None:
        await self._extension._ui._client.call(
            "ui.contributions.remove", {"contribution_id": self.contribution_id}
        )


class AsyncExtension:
    def __init__(self, ui: "AsyncUi", snapshot: Mapping[str, Any]) -> None:
        self._ui = ui
        self.snapshot = dict(snapshot)
        self._interaction_callbacks: dict[
            AsyncInteractionSubscription, Callable[[Event], Any | Awaitable[Any]]
        ] = {}
        self._action_runner: AsyncActionRunner | None = None
        self._action_registrations: set[AsyncActionRegistration] = set()

    @property
    def id(self) -> str:
        return str(self.snapshot["id"])

    async def set_readiness(
        self, ready: bool, *, reason: str | None = None
    ) -> "AsyncExtension":
        if not isinstance(ready, bool):
            raise ValueError("extension readiness must be a boolean")
        params: dict[str, Any] = {"extension_id": self.id, "ready": ready}
        if reason is not None:
            if not isinstance(reason, str) or not reason or len(reason.encode("utf-8")) > 256:
                raise ValueError("extension readiness reason must contain 1 to 256 bytes")
            params["reason"] = reason
        self.snapshot = dict(
            await self._ui._client.call("ui.extensions.set_readiness", params)
        )
        return self

    async def register(
        self,
        root: Component,
        *,
        location: str = "shell",
        contribution_id: str | None = None,
    ) -> AsyncContribution:
        params: dict[str, Any] = {
            "extension_id": self.id,
            "location": location,
            "root": root.to_dict(),
        }
        if contribution_id is not None:
            params["contribution_id"] = contribution_id
        result = await self._ui._client.call("ui.contributions.register", params)
        return AsyncContribution(self, result)

    async def on_interaction(
        self,
        callback: Callable[[UiInteraction], Any | Awaitable[Any]],
        *,
        action: str | None = None,
        component_id: str | None = None,
    ) -> AsyncInteractionSubscription:
        """Subscribe to normalized interactions from this extension's components."""

        if not callable(callback):
            raise TypeError("interaction callback must be callable")
        if action is not None and (not isinstance(action, str) or not action.strip()):
            raise ValueError("interaction action filter must be a non-empty string")
        if component_id is not None and (
            not isinstance(component_id, str) or not component_id.strip()
        ):
            raise ValueError("interaction component_id filter must be a non-empty string")
        pattern = f"ui.extension:{self.id}.*"

        def receive(event: Event) -> Any | Awaitable[Any]:
            try:
                interaction = UiInteraction.from_event(event, extension_id=self.id)
            except UiInteractionDecodeError as error:
                logger.warning("Ignoring malformed Odon UI interaction: %s", error)
                return None
            if action is not None and interaction.action != action:
                return None
            if component_id is not None and interaction.component_id != component_id:
                return None
            return callback(interaction)

        subscription: AsyncInteractionSubscription

        async def remove() -> None:
            receive_callback = self._interaction_callbacks.pop(subscription, None)
            if receive_callback is None:
                return
            self._ui._client.events.remove_callback(receive_callback)
            if not self._interaction_callbacks and not self._ui._client.closed:
                await self._ui._client.events.unsubscribe(pattern)

        subscription = AsyncInteractionSubscription(remove)
        await self._ui._client.events.subscribe(pattern, receive)
        self._interaction_callbacks[subscription] = receive
        return subscription

    async def on_action(
        self,
        action: str,
        callback: Callable[
            [AsyncActionContext, UiInteraction], Any | Awaitable[Any]
        ],
        *,
        execution: ExecutionPolicy = "serial-worker",
        component_id: str | None = None,
        queue_key: str | None = None,
        coalesce: CoalescePolicy = "all",
        delta: float = 1,
        max_queue: int = 128,
        contribution: AsyncContribution | None = None,
        status_component_id: str | None = None,
        progress_component_id: str | None = None,
        on_error: Callable[
            [BaseException, AsyncActionContext | None], Any | Awaitable[Any]
        ]
        | None = None,
    ) -> AsyncActionRegistration:
        """Register a normalized action with an explicit async execution policy."""

        if not isinstance(action, str) or not action.strip():
            raise ValueError("extension action must be a non-empty string")
        if not callable(callback):
            raise TypeError("extension action callback must be callable")
        checked_execution = _validate_execution(execution)
        checked_coalesce = _validate_coalesce(coalesce)
        checked_key = action if queue_key is None else queue_key
        if not isinstance(checked_key, str) or not checked_key.strip():
            raise ValueError("extension action queue_key must be a non-empty string")
        if isinstance(delta, bool) or not isinstance(delta, (int, float)):
            raise ValueError("extension action delta must be numeric")
        if self._action_runner is None:
            self._action_runner = AsyncActionRunner(max_queue=max_queue)
        elif self._action_runner.max_queue != max_queue:
            raise ValueError("all actions on one extension must use the same max_queue")

        def removed(registration: AsyncActionRegistration) -> None:
            self._action_registrations.discard(registration)

        registration = AsyncActionRegistration(
            self._action_runner,
            action,
            callback,
            execution=checked_execution,
            coalesce=checked_coalesce,
            queue_key=checked_key,
            delta=float(delta),
            contribution=contribution,
            status_component_id=status_component_id,
            progress_component_id=progress_component_id,
            on_error=on_error,
            on_remove=removed,
        )
        subscription = await self.on_interaction(
            registration.submit,
            action=action,
            component_id=component_id,
        )
        registration._subscription = subscription
        self._action_registrations.add(registration)
        return registration

    def action_status(self) -> ActionWorkerSnapshot:
        if self._action_runner is None:
            return ActionWorkerSnapshot(
                submitted=0,
                executed=0,
                completed=0,
                failed=0,
                cancelled=0,
                rejected=0,
                coalesced=0,
                queue_depth=0,
                running_actions=(),
                closed=False,
            )
        return self._action_runner.snapshot()

    async def _remove_action_registrations(self) -> None:
        for registration in tuple(self._action_registrations):
            await registration.remove()
        if self._action_runner is not None:
            await self._action_runner.close()
            self._action_runner = None

    async def _remove_interaction_subscriptions(self) -> None:
        for subscription in tuple(self._interaction_callbacks):
            await subscription.remove()

    async def register_command(
        self,
        id: str,
        title: str,
        description: str,
        event: str,
        *,
        modes: Iterable[str] = ("project", "single", "mosaic"),
        shortcut: Mapping[str, Any] | None = None,
        icon: str | None = None,
        predicates: CommandPredicates | Mapping[str, Any] | None = None,
        if_revision: int | None = None,
        transaction_id: str | None = None,
    ) -> ApplicationCommand:
        result = _require_mapping(
            await self._ui._client.call(
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

    async def remove_command(
        self,
        command: ApplicationCommand | str,
        *,
        if_revision: int | None = None,
        transaction_id: str | None = None,
    ) -> None:
        command_id = command.id if isinstance(command, ApplicationCommand) else command
        params: dict[str, Any] = {
            "extension_id": self.id,
            "command_id": str(command_id),
        }
        revision = _validate_command_revision(if_revision)
        if revision is not None:
            params["if_command_revision"] = revision
        checked_transaction = _validate_shell_transaction_id(transaction_id)
        if checked_transaction is not None:
            params["transaction_id"] = checked_transaction
        await self._ui._client.call("ui.commands.remove", params)

    async def register_layout(
        self,
        name: str,
        document: ShellLayoutDocument | Mapping[str, Any],
    ) -> ExtensionLayoutTemplate:
        """Register or replace a named default layout owned by this extension."""

        result = await self._ui._client.call(
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

    async def list_layouts(self) -> tuple[ExtensionLayoutTemplate, ...]:
        """List this extension's registered default layouts by name."""

        result = _require_mapping(
            await self._ui._client.call(
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

    async def remove_layout(self, name: str) -> None:
        """Remove a named default layout owned by this extension."""

        await self._ui._client.call(
            "ui.extensions.layouts.remove",
            {"extension_id": self.id, "name": _extension_layout_name(name)},
        )

    async def remove(self) -> None:
        await self._remove_action_registrations()
        await self._remove_interaction_subscriptions()
        await self._ui._client.call("ui.extensions.remove", {"extension_id": self.id})
        self._ui._extensions.discard(self)

    async def _close_local(self) -> None:
        if self._action_runner is not None:
            await self._action_runner.close()
            self._action_runner = None
        for registration in tuple(self._action_registrations):
            registration._close_local()
        for subscription in tuple(self._interaction_callbacks):
            subscription._close_local()
        for callback in tuple(self._interaction_callbacks.values()):
            self._ui._client.events.remove_callback(callback)
        self._interaction_callbacks.clear()
        self._action_registrations.clear()


class AsyncCommands:
    def __init__(self, ui: "AsyncUi") -> None:
        self._ui = ui

    async def describe_schema(self) -> Mapping[str, Any]:
        return await self._ui._client.call("ui.commands.describe_schema")

    async def list(self) -> tuple[ApplicationCommand, ...]:
        result = _require_mapping(
            await self._ui._client.call("ui.commands.list"),
            "application command list",
        )
        commands = result.get("commands")
        if not isinstance(commands, list):
            raise ValueError("application command list requires a commands array")
        return tuple(
            ApplicationCommand.from_result(
                _require_mapping(command, "application command")
            )
            for command in commands
        )

    async def execute(
        self, command: ApplicationCommand | str, *, checked: bool | None = None
    ) -> Mapping[str, Any]:
        command_id = command.id if isinstance(command, ApplicationCommand) else command
        params: dict[str, Any] = {"command_id": str(command_id)}
        if checked is not None:
            if not isinstance(checked, bool):
                raise ValueError("checked must be a boolean")
            params["checked"] = checked
        return await self._ui._client.call(
            "ui.commands.execute", params
        )


class AsyncMenus:
    def __init__(self, ui: "AsyncUi") -> None:
        self._ui = ui

    async def get(self) -> CommandMenuSnapshot:
        return CommandMenuSnapshot.from_result(
            await self._ui._client.call("ui.menus.get")
        )

    async def replace(
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
            await self._ui._client.call("ui.menus.replace", params)
        )


class AsyncToolbars:
    def __init__(self, ui: "AsyncUi") -> None:
        self._ui = ui

    async def get(self) -> CommandToolbarSnapshot:
        return CommandToolbarSnapshot.from_result(
            await self._ui._client.call("ui.toolbars.get")
        )

    async def replace(
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
            await self._ui._client.call("ui.toolbars.replace", params)
        )


class AsyncPalette:
    def __init__(self, ui: "AsyncUi") -> None:
        self._ui = ui

    async def get(self) -> CommandPaletteSnapshot:
        return CommandPaletteSnapshot.from_result(
            await self._ui._client.call("ui.palette.get")
        )

    async def replace(
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
            await self._ui._client.call("ui.palette.replace", params)
        )


class AsyncShell:
    """Async access to Odon's native application shell."""

    def __init__(self, ui: "AsyncUi") -> None:
        self._ui = ui

    async def describe_schema(self) -> Mapping[str, Any]:
        return await self._ui._client.call("ui.shell.describe_schema")

    async def get(self, *, mode: str | None = None) -> ShellSnapshot:
        return ShellSnapshot.from_result(
            await self._ui._client.call("ui.shell.get", Shell._get_params(mode))
        )

    async def list_components(
        self, *, mode: str | None = None
    ) -> tuple[ShellComponentDescriptor, ...]:
        result = await self._ui._client.call(
            "ui.shell.components.list", Shell._get_params(mode)
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

    async def export_layout(
        self, *, mode: str | None = None
    ) -> ShellLayoutDocument:
        return ShellLayoutDocument.from_result(
            await self._ui._client.call(
                "ui.shell.export_layout", Shell._get_params(mode)
            )
        )

    async def import_layout(
        self,
        document: ShellLayoutDocument | Mapping[str, Any],
        *,
        mode: str | None = None,
        if_revision: int | None = None,
        transaction_id: str | None = None,
    ) -> ShellSnapshot:
        return ShellSnapshot.from_result(
            await self._ui._client.call(
                "ui.shell.import_layout",
                Shell._import_layout_params(document, mode, if_revision, transaction_id),
            )
        )

    async def list_profiles(
        self, *, scope: str = "session"
    ) -> tuple[ShellLayoutProfile, ...]:
        result = await self._ui._client.call(
            "ui.shell.profiles.list", Shell._profile_scope_params(scope)
        )
        profiles = _require_mapping(result, "shell profile list").get("profiles")
        if not isinstance(profiles, list):
            raise ValueError("shell profile list requires a profiles array")
        return tuple(
            ShellLayoutProfile.from_result(_require_mapping(item, "shell profile"))
            for item in profiles
        )

    async def save_profile(
        self,
        name: str,
        *,
        scope: str = "session",
        mode: str | None = None,
    ) -> Mapping[str, Any]:
        return await self._ui._client.call(
            "ui.shell.profiles.save", Shell._profile_params(name, scope, mode)
        )

    async def load_profile(
        self,
        name: str,
        *,
        scope: str = "session",
        mode: str | None = None,
        if_revision: int | None = None,
        transaction_id: str | None = None,
    ) -> ShellSnapshot:
        params = Shell._profile_params(name, scope, mode)
        if if_revision is not None:
            params["if_shell_revision"] = Shell._reset_params(None, if_revision)[
                "if_shell_revision"
            ]
        if transaction_id is not None:
            params["transaction_id"] = Shell._reset_params(
                None, None, transaction_id
            )["transaction_id"]
        return ShellSnapshot.from_result(
            await self._ui._client.call("ui.shell.profiles.load", params)
        )

    async def remove_profile(
        self, name: str, *, scope: str = "session"
    ) -> Mapping[str, Any]:
        return await self._ui._client.call(
            "ui.shell.profiles.remove", Shell._profile_params(name, scope, None)
        )

    async def patch(
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
            await self._ui._client.call(
                "ui.shell.patch",
                Shell._patch_params(
                    visibility=visibility,
                    orders=orders,
                    selected=selected,
                    mode=mode,
                    if_revision=if_revision,
                    transaction_id=transaction_id,
                ),
            )
        )

    async def reset(
        self, *, mode: str | None = None, if_revision: int | None = None,
        transaction_id: str | None = None,
    ) -> ShellSnapshot:
        return ShellSnapshot.from_result(
            await self._ui._client.call(
                "ui.shell.reset", Shell._reset_params(mode, if_revision, transaction_id)
            )
        )

    async def recover(
        self, *, mode: str | None = None, if_revision: int | None = None,
        transaction_id: str | None = None,
    ) -> ShellSnapshot:
        return ShellSnapshot.from_result(
            await self._ui._client.call(
                "ui.shell.recover", Shell._reset_params(mode, if_revision, transaction_id)
            )
        )

    async def replace_layout(
        self,
        layout: ShellLayout | Mapping[str, Any],
        *,
        mode: str | None = None,
        if_revision: int | None = None,
        transaction_id: str | None = None,
    ) -> ShellSnapshot:
        return ShellSnapshot.from_result(
            await self._ui._client.call(
                "ui.shell.replace_layout",
                Shell._replace_layout_params(layout, mode, if_revision, transaction_id),
            )
        )

    async def patch_layout(
        self,
        *,
        visibility: Mapping[str, bool] | None = None,
        selected: Mapping[str, str] | None = None,
        sizes: Mapping[str, Any] | None = None,
        splits: Mapping[str, Any] | None = None,
        collapsed: Mapping[str, bool] | None = None,
        configurations: Mapping[str, Mapping[str, Any]] | None = None,
        active_region_id: str | None = None,
        focused_node_id: str | None = None,
        clear_focus: bool = False,
        mode: str | None = None,
        if_revision: int | None = None,
        transaction_id: str | None = None,
    ) -> ShellSnapshot:
        return ShellSnapshot.from_result(
            await self._ui._client.call(
                "ui.shell.patch_layout",
                Shell._patch_layout_params(
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


class AsyncUi:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client
        self.commands = AsyncCommands(self)
        self.menus = AsyncMenus(self)
        self.toolbars = AsyncToolbars(self)
        self.palette = AsyncPalette(self)
        self.shell = AsyncShell(self)
        self._extensions: weakref.WeakSet[AsyncExtension] = weakref.WeakSet()

    async def register_extension(
        self,
        *,
        id: str,
        name: str,
        version: str,
        capabilities: Iterable[str] = ("ui.panels",),
        disconnect_policy: str = "remove",
        ready: bool = True,
        readiness_reason: str | None = None,
    ) -> AsyncExtension:
        if not isinstance(ready, bool):
            raise ValueError("extension readiness must be a boolean")
        result = await self._client.call(
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
        extension = AsyncExtension(self, result)
        self._extensions.add(extension)
        return extension

    async def _close(self) -> None:
        for extension in tuple(self._extensions):
            await extension._close_local()
        self._extensions.clear()

    async def list_extensions(self) -> list[Mapping[str, Any]]:
        return (await self._client.call("ui.extensions.list"))["extensions"]

    async def list_contributions(self) -> list[Mapping[str, Any]]:
        return (await self._client.call("ui.contributions.list"))["contributions"]

    async def describe_schema(self) -> Mapping[str, Any]:
        return await self._client.call("ui.describe_schema")
