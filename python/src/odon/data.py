"""Referenced data resources and optional NumPy-to-Zarr exchange."""

from __future__ import annotations

import shutil
import tempfile
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from .client import Client


@dataclass(frozen=True)
class CoordinateSpace:
    axes: tuple[str, ...]
    units: tuple[str, ...] = ()
    scale: tuple[float, ...] = ()
    translation: tuple[float, ...] = ()
    reference_layer_id: str | None = None

    def __post_init__(self) -> None:
        if not self.axes or any(not axis.strip() for axis in self.axes):
            raise ValueError("axes must contain non-empty names")
        if len(set(self.axes)) != len(self.axes):
            raise ValueError("axis names must be unique")
        for name, values in (
            ("units", self.units),
            ("scale", self.scale),
            ("translation", self.translation),
        ):
            if values and len(values) != len(self.axes):
                raise ValueError(f"{name} must be empty or match the number of axes")
        if any(not math.isfinite(value) or value <= 0 for value in self.scale):
            raise ValueError("scale values must be finite and positive")
        if any(not math.isfinite(value) for value in self.translation):
            raise ValueError("translation values must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "axes": list(self.axes),
            "units": list(self.units),
            "scale": list(self.scale),
            "translation": list(self.translation),
            **(
                {"reference_layer_id": self.reference_layer_id}
                if self.reference_layer_id is not None
                else {}
            ),
        }

    def pixel_to_world(self, coordinates: Sequence[float]) -> tuple[float, ...]:
        """Apply the level-zero affine transform to pixel-centre coordinates."""

        if len(coordinates) != len(self.axes):
            raise ValueError("coordinates must match the number of axes")
        scale = self.scale or (1.0,) * len(self.axes)
        translation = self.translation or (0.0,) * len(self.axes)
        return tuple(
            float(coordinate) * scale[index] + translation[index]
            for index, coordinate in enumerate(coordinates)
        )

    def world_to_pixel(self, coordinates: Sequence[float]) -> tuple[float, ...]:
        """Invert the level-zero affine transform for world coordinates."""

        if len(coordinates) != len(self.axes):
            raise ValueError("coordinates must match the number of axes")
        scale = self.scale or (1.0,) * len(self.axes)
        translation = self.translation or (0.0,) * len(self.axes)
        return tuple(
            (float(coordinate) - translation[index]) / scale[index]
            for index, coordinate in enumerate(coordinates)
        )


class DataResource:
    def __init__(self, resources: "DataResources", snapshot: Mapping[str, Any]) -> None:
        self._resources = resources
        self.snapshot = dict(snapshot)

    @property
    def resource_id(self) -> str:
        return str(self.snapshot["resource_id"])

    def refresh(self) -> "DataResource":
        self.snapshot = self._resources.get(self.resource_id).snapshot
        return self

    def remove(self) -> None:
        self._resources.remove(self.resource_id)


class DataResources:
    def __init__(self, client: "Client") -> None:
        self._client = client
        self._temporary_directories: dict[str, Path] = {}

    def register(
        self,
        uri: str | Path,
        *,
        format: str,
        coordinate_space: CoordinateSpace,
        resource_id: str | None = None,
        ownership: str = "session",
        metadata: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> DataResource:
        uri_value = str(uri)
        if isinstance(uri, Path):
            uri_value = uri.resolve().as_uri()
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
        return DataResource(self, self._client.call("data.resources.register", params))

    def register_numpy(
        self,
        array: Any,
        *,
        axes: Sequence[str],
        units: Sequence[str] = (),
        scale: Sequence[float] = (),
        translation: Sequence[float] = (),
        provenance: Mapping[str, Any] | None = None,
    ) -> DataResource:
        """Write an array to session-scoped Zarr and register it by reference.

        This method requires the optional ``odon-client[arrays]`` dependencies.
        """

        directory, path, metadata = _write_numpy_zarr(
            array, axes=axes, units=units, scale=scale, translation=translation
        )
        try:
            resource = self.register(
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
            shutil.rmtree(directory, ignore_errors=True)
            raise
        self._temporary_directories[resource.resource_id] = directory
        return resource

    def get(self, resource_id: str) -> DataResource:
        return DataResource(
            self,
            self._client.call("data.resources.get", {"resource_id": resource_id}),
        )

    def list(self) -> list[DataResource]:
        result = self._client.call("data.resources.list")
        return [DataResource(self, item) for item in result["resources"]]

    def remove(self, resource_id: str) -> None:
        self._client.call("data.resources.remove", {"resource_id": resource_id})
        directory = self._temporary_directories.pop(resource_id, None)
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)

    def _close(self) -> None:
        for directory in self._temporary_directories.values():
            shutil.rmtree(directory, ignore_errors=True)
        self._temporary_directories.clear()


def _write_numpy_zarr(
    array: Any,
    *,
    axes: Sequence[str],
    units: Sequence[str] = (),
    scale: Sequence[float] = (),
    translation: Sequence[float] = (),
) -> tuple[Path, Path, dict[str, Any]]:
    try:
        import numpy as np
        import zarr
    except ImportError as error:
        raise ImportError(
            "register_numpy requires `pip install odon-client[arrays]`"
        ) from error
    data = np.asarray(array)
    coordinate_space = CoordinateSpace(
        axes=tuple(axes),
        units=tuple(units),
        scale=tuple(float(value) for value in scale),
        translation=tuple(float(value) for value in translation),
    )
    if len(coordinate_space.axes) != data.ndim:
        raise ValueError("the number of axes must match the array rank")
    directory = Path(tempfile.mkdtemp(prefix="odon-array-"))
    path = directory / "array.zarr"
    try:
        root = zarr.open_group(str(path), mode="w")
        try:
            root.create_array("0", data=data)
        except AttributeError:  # zarr 2.x
            root.create_dataset("0", data=data, shape=data.shape, dtype=data.dtype)
        transforms: list[dict[str, Any]] = []
        if scale:
            transforms.append({"type": "scale", "scale": [float(value) for value in scale]})
        if translation:
            transforms.append(
                {"type": "translation", "translation": [float(value) for value in translation]}
            )
        root.attrs["multiscales"] = [
            {
                "version": "0.4",
                "axes": [
                    {"name": name, **({"unit": units[index]} if index < len(units) else {})}
                    for index, name in enumerate(axes)
                ],
                "datasets": [
                    {"path": "0", **({"coordinateTransformations": transforms} if transforms else {})}
                ],
            }
        ]
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    metadata: dict[str, Any] = {"shape": list(data.shape), "dtype": str(data.dtype)}
    if data.size and np.issubdtype(data.dtype, np.number):
        try:
            minimum = data.min().item()
            maximum = data.max().item()
            if math.isfinite(float(minimum)) and math.isfinite(float(maximum)):
                metadata.update(value_min=minimum, value_max=maximum)
        except (TypeError, ValueError):
            pass
    return directory, path, metadata
