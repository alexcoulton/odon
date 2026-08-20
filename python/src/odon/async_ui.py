"""Async registration wrappers for declarative Odon UI."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from .ui import Component

if TYPE_CHECKING:
    from .async_client import AsyncClient


class AsyncContribution:
    def __init__(self, extension: "AsyncExtension", snapshot: Mapping[str, Any]) -> None:
        self._extension = extension
        self.snapshot = dict(snapshot)

    @property
    def contribution_id(self) -> str:
        return str(self.snapshot["contribution_id"])

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

    @property
    def id(self) -> str:
        return str(self.snapshot["id"])

    async def register(
        self,
        root: Component,
        *,
        location: str = "right.tabs",
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

    async def remove(self) -> None:
        await self._ui._client.call("ui.extensions.remove", {"extension_id": self.id})


class AsyncUi:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def register_extension(
        self,
        *,
        id: str,
        name: str,
        version: str,
        capabilities: Iterable[str] = ("ui.panels",),
        disconnect_policy: str = "remove",
    ) -> AsyncExtension:
        result = await self._client.call(
            "ui.extensions.register",
            {
                "id": id,
                "name": name,
                "version": version,
                "capabilities": list(capabilities),
                "disconnect_policy": disconnect_policy,
            },
        )
        return AsyncExtension(self, result)

    async def list_extensions(self) -> list[Mapping[str, Any]]:
        return (await self._client.call("ui.extensions.list"))["extensions"]

    async def list_contributions(self) -> list[Mapping[str, Any]]:
        return (await self._client.call("ui.contributions.list"))["contributions"]

    async def describe_schema(self) -> Mapping[str, Any]:
        return await self._client.call("ui.describe_schema")
