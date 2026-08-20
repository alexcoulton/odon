"""Async data-resource wrappers."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .data import CoordinateSpace, _write_numpy_zarr

if TYPE_CHECKING:
    from .async_client import AsyncClient


class AsyncDataResource:
    def __init__(self, resources: "AsyncDataResources", snapshot: Mapping[str, Any]) -> None:
        self._resources = resources
        self.snapshot = dict(snapshot)

    @property
    def resource_id(self) -> str:
        return str(self.snapshot["resource_id"])

    async def refresh(self) -> "AsyncDataResource":
        self.snapshot = (await self._resources.get(self.resource_id)).snapshot
        return self

    async def remove(self) -> None:
        await self._resources.remove(self.resource_id)


class AsyncDataResources:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client
        self._temporary_directories: dict[str, Path] = {}

    async def register(
        self,
        uri: str | Path,
        *,
        format: str,
        coordinate_space: CoordinateSpace,
        resource_id: str | None = None,
        ownership: str = "session",
        metadata: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> AsyncDataResource:
        uri_value = uri.resolve().as_uri() if isinstance(uri, Path) else str(uri)
        params: dict[str, Any] = {
            "uri": uri_value,
            "format": format,
            "ownership": ownership,
            "coordinate_space": coordinate_space.to_dict(),
            "metadata": dict(metadata or {}),
            "provenance": dict(provenance or {}),
        }
        if resource_id is not None:
            params["resource_id"] = resource_id
        return AsyncDataResource(
            self, await self._client.call("data.resources.register", params)
        )

    async def get(self, resource_id: str) -> AsyncDataResource:
        result = await self._client.call(
            "data.resources.get", {"resource_id": resource_id}
        )
        return AsyncDataResource(self, result)

    async def list(self) -> list[AsyncDataResource]:
        result = await self._client.call("data.resources.list")
        return [AsyncDataResource(self, item) for item in result["resources"]]

    async def remove(self, resource_id: str) -> None:
        await self._client.call("data.resources.remove", {"resource_id": resource_id})
        directory = self._temporary_directories.pop(resource_id, None)
        if directory is not None:
            await asyncio.to_thread(shutil.rmtree, directory, ignore_errors=True)

    async def register_numpy(
        self,
        array: Any,
        *,
        axes: Sequence[str],
        units: Sequence[str] = (),
        scale: Sequence[float] = (),
        translation: Sequence[float] = (),
        provenance: Mapping[str, Any] | None = None,
    ) -> AsyncDataResource:
        directory, path, metadata = await asyncio.to_thread(
            _write_numpy_zarr,
            array,
            axes=axes,
            units=units,
            scale=scale,
            translation=translation,
        )
        try:
            resource = await self.register(
                path,
                format="ome-zarr",
                coordinate_space=CoordinateSpace(
                    axes=tuple(axes),
                    units=tuple(units),
                    scale=tuple(float(value) for value in scale),
                    translation=tuple(float(value) for value in translation),
                ),
                metadata=metadata,
                provenance=provenance,
            )
        except BaseException:
            await asyncio.to_thread(shutil.rmtree, directory, ignore_errors=True)
            raise
        self._temporary_directories[resource.resource_id] = directory
        return resource

    async def _close(self) -> None:
        directories = tuple(self._temporary_directories.values())
        self._temporary_directories = {}
        await asyncio.gather(
            *(asyncio.to_thread(shutil.rmtree, directory, ignore_errors=True) for directory in directories)
        )
