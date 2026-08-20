"""Async stable external layer wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .async_data import AsyncDataResource

if TYPE_CHECKING:
    from .async_client import AsyncClient


class AsyncLayer:
    def __init__(self, layers: "AsyncLayers", snapshot: Mapping[str, Any]) -> None:
        self._layers = layers
        self.snapshot = dict(snapshot)

    @property
    def layer_id(self) -> str:
        return str(self.snapshot["layer_id"])

    @property
    def revision(self) -> int:
        return int(self.snapshot["revision"])

    async def refresh(self) -> "AsyncLayer":
        self.snapshot = (await self._layers.get(self.layer_id)).snapshot
        return self

    async def update(self, **changes: Any) -> "AsyncLayer":
        self.snapshot = (await self._layers.update(self.layer_id, **changes)).snapshot
        return self

    async def remove(self) -> None:
        await self._layers.remove(self.layer_id)


class AsyncLayers:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def add(
        self,
        data: AsyncDataResource | str,
        *,
        name: str,
        kind: str,
        layer_id: str | None = None,
        visible: bool = True,
        opacity: float = 1.0,
        ownership: str = "session",
        style: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> AsyncLayer:
        params: dict[str, Any] = {
            "data_resource_id": data.resource_id if isinstance(data, AsyncDataResource) else data,
            "name": name,
            "kind": kind,
            "visible": visible,
            "opacity": opacity,
            "ownership": ownership,
            "style": dict(style or {}),
            "provenance": dict(provenance or {}),
        }
        if layer_id is not None:
            params["layer_id"] = layer_id
        return AsyncLayer(self, await self._client.call("viewer.layers.add", params))

    async def get(self, layer_id: str) -> AsyncLayer:
        return AsyncLayer(
            self, await self._client.call("viewer.layers.get", {"layer_id": layer_id})
        )

    async def list(self) -> list[AsyncLayer]:
        result = await self._client.call("viewer.layers.list")
        return [AsyncLayer(self, item) for item in result["layers"]]

    async def update(self, layer_id: str, **changes: Any) -> AsyncLayer:
        params = {"layer_id": layer_id, **{key: value for key, value in changes.items() if value is not None}}
        return AsyncLayer(self, await self._client.call("viewer.layers.update", params))

    async def replace_data(
        self, layer_id: str, data: AsyncDataResource | str
    ) -> AsyncLayer:
        resource_id = data.resource_id if isinstance(data, AsyncDataResource) else data
        return await self.update(layer_id, data_resource_id=resource_id)

    async def remove(self, layer_id: str) -> None:
        await self._client.call("viewer.layers.remove", {"layer_id": layer_id})

    async def reorder(self, layers: Sequence[AsyncLayer | str]) -> list[AsyncLayer]:
        order = [layer.layer_id if isinstance(layer, AsyncLayer) else layer for layer in layers]
        result = await self._client.call("viewer.layers.reorder", {"order": order})
        return [AsyncLayer(self, item) for item in result["layers"]]
