"""Stable external layer resources."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .data import DataResource

if TYPE_CHECKING:
    from .client import Client


class Layer:
    def __init__(self, layers: "Layers", snapshot: Mapping[str, Any]) -> None:
        self._layers = layers
        self.snapshot = dict(snapshot)

    @property
    def layer_id(self) -> str:
        return str(self.snapshot["layer_id"])

    @property
    def revision(self) -> int:
        return int(self.snapshot["revision"])

    def refresh(self) -> "Layer":
        self.snapshot = self._layers.get(self.layer_id).snapshot
        return self

    def update(
        self,
        *,
        name: str | None = None,
        visible: bool | None = None,
        opacity: float | None = None,
        style: Mapping[str, Any] | None = None,
        if_revision: int | None = None,
    ) -> "Layer":
        self.snapshot = self._layers.update(
            self.layer_id,
            name=name,
            visible=visible,
            opacity=opacity,
            style=style,
            if_revision=if_revision,
        ).snapshot
        return self

    def remove(self) -> None:
        self._layers.remove(self.layer_id)


class Layers:
    def __init__(self, client: "Client") -> None:
        self._client = client

    def add(
        self,
        data: DataResource | str,
        *,
        name: str,
        kind: str,
        layer_id: str | None = None,
        visible: bool = True,
        opacity: float = 1.0,
        ownership: str = "session",
        style: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> Layer:
        params: dict[str, Any] = {
            "data_resource_id": data.resource_id if isinstance(data, DataResource) else data,
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
        return Layer(self, self._client.call("viewer.layers.add", params))

    def get(self, layer_id: str) -> Layer:
        return Layer(self, self._client.call("viewer.layers.get", {"layer_id": layer_id}))

    def list(self) -> list[Layer]:
        result = self._client.call("viewer.layers.list")
        return [Layer(self, item) for item in result["layers"]]

    def update(self, layer_id: str, **changes: Any) -> Layer:
        params = {"layer_id": layer_id, **{key: value for key, value in changes.items() if value is not None}}
        return Layer(self, self._client.call("viewer.layers.update", params))

    def replace_data(self, layer_id: str, data: DataResource | str) -> Layer:
        resource_id = data.resource_id if isinstance(data, DataResource) else data
        return self.update(layer_id, data_resource_id=resource_id)

    def remove(self, layer_id: str) -> None:
        self._client.call("viewer.layers.remove", {"layer_id": layer_id})

    def reorder(self, layers: Sequence[Layer | str]) -> list[Layer]:
        order = [layer.layer_id if isinstance(layer, Layer) else layer for layer in layers]
        result = self._client.call("viewer.layers.reorder", {"order": order})
        return [Layer(self, item) for item in result["layers"]]
