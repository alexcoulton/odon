"""Asynchronous, resource-oriented wrappers over Odon control methods."""

from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

if TYPE_CHECKING:
    from .async_client import AsyncClient


def _with_revision(params: Mapping[str, Any], if_revision: int | None) -> dict[str, Any]:
    result = dict(params)
    if if_revision is not None:
        result["if_revision"] = if_revision
    return result


class AsyncApplication:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client
        self._cached_state: Any = None

    async def get_state(self) -> Any:
        state = await self._client.call("app.get_state")
        self._cached_state = deepcopy(state)
        return state

    @property
    def cached_state(self) -> Any:
        return deepcopy(self._cached_state)

    async def get_loading_state(self) -> Any:
        return await self._client.call("app.get_loading_state")

    async def list_methods(self) -> Any:
        return await self._client.call("system.describe_methods")

    async def describe_events(self) -> Any:
        return await self._client.call("system.describe_events")

    async def get_application_surface(self) -> Any:
        """Return Odon's machine-readable native/API/Python parity manifest."""

        return await self._client.call("system.get_application_surface")

    async def get_method_availability(
        self, methods: Iterable[str] | None = None
    ) -> Any:
        params = {} if methods is None else {"methods": list(methods)}
        return await self._client.call("app.get_method_availability", params)

    async def get_diagnostics(self) -> Any:
        return await self._client.call("system.get_diagnostics")

    async def get_settings(self) -> Any:
        return await self._client.call("app.settings.get")

    async def update_settings(
        self,
        *,
        auto_contrast: Mapping[str, Any] | None = None,
        fast_object_rendering: bool | None = None,
        if_revision: int | None = None,
    ) -> Any:
        params: dict[str, Any] = {}
        if auto_contrast is not None:
            params["auto_contrast"] = dict(auto_contrast)
        if fast_object_rendering is not None:
            params["fast_object_rendering"] = fast_object_rendering
        return await self._client.call(
            "app.settings.set", _with_revision(params, if_revision)
        )

    async def list_recent_projects(self) -> Any:
        return await self._client.call("app.recent_projects.list")

    async def forget_recent_project(
        self, path: str | Path, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "app.recent_projects.forget",
            _with_revision({"path": str(path)}, if_revision),
        )

    async def clear_recent_projects(self, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "app.recent_projects.clear", _with_revision({}, if_revision)
        )

    async def get_lifecycle(self) -> Any:
        return await self._client.call("app.lifecycle.get")

    async def request_close(
        self, *, save: str = "prompt", if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "app.lifecycle.request_close",
            _with_revision({"save": save}, if_revision),
        )

    async def request_quit(
        self, *, save: str = "prompt", if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "app.lifecycle.request_quit",
            _with_revision({"save": save}, if_revision),
        )

    async def open_ome_zarr(
        self, path: str | Path, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.tasks.start(
            "datasets.open_ome_zarr",
            _with_revision({"path": str(path)}, if_revision),
            label=f"Open {path}",
        )

    async def open_tiff(
        self,
        path: str | Path,
        *,
        z: int = 0,
        t: int = 0,
        if_revision: int | None = None,
    ) -> Any:
        params: dict[str, Any] = {"path": str(path)}
        if z:
            params["z"] = z
        if t:
            params["t"] = t
        return await self._client.tasks.start(
            "datasets.open_tiff",
            _with_revision(params, if_revision),
            label=f"Open {path}",
        )

    async def open_mosaic_samplesheet(
        self, path: str | Path, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.tasks.start(
            "datasets.open_mosaic_samplesheet",
            _with_revision({"path": str(path)}, if_revision),
            label=f"Open mosaic {path}",
        )

    async def show_project_page(self, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "app.navigation.show_project", _with_revision({}, if_revision)
        )


class AsyncDatasets:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def inspect(self, path: str | Path) -> Any:
        return await self._client.call("datasets.inspect", {"path": str(path)})

    async def open_ome_zarr(
        self, path: str | Path, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.application.open_ome_zarr(
            path, if_revision=if_revision
        )

    async def open_tiff(
        self,
        path: str | Path,
        *,
        z: int = 0,
        t: int = 0,
        if_revision: int | None = None,
    ) -> Any:
        return await self._client.application.open_tiff(
            path, z=z, t=t, if_revision=if_revision
        )

    async def open_mosaic_samplesheet(
        self, path: str | Path, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.application.open_mosaic_samplesheet(
            path, if_revision=if_revision
        )

    async def open_spatialdata(
        self,
        path: str | Path,
        *,
        image: str,
        extra_images: Iterable[str] = (),
        labels: str | None = None,
        shapes: Iterable[str] = (),
        points: str | None = None,
        points_max: int = 200_000,
        if_revision: int | None = None,
    ) -> Any:
        params: dict[str, Any] = {
            "path": str(path),
            "image": image,
            "extra_images": list(extra_images),
            "shapes": list(shapes),
            "points_max": points_max,
        }
        if labels is not None:
            params["labels"] = labels
        if points is not None:
            params["points"] = points
        return await self._client.tasks.start(
            "datasets.open_spatialdata",
            _with_revision(params, if_revision),
            label=f"Open SpatialData {path}",
        )

    async def open_xenium(
        self,
        path: str | Path,
        *,
        imagery: str = "auto",
        load_cells: bool = True,
        load_transcripts: bool = True,
        if_revision: int | None = None,
    ) -> Any:
        return await self._client.tasks.start(
            "datasets.open_xenium",
            _with_revision(
                {
                    "path": str(path),
                    "imagery": imagery,
                    "load_cells": load_cells,
                    "load_transcripts": load_transcripts,
                },
                if_revision,
            ),
            label=f"Open Xenium {path}",
        )

    async def open_http(
        self, url: str, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.tasks.start(
            "datasets.open_http",
            _with_revision({"url": url}, if_revision),
            label=f"Open remote OME-Zarr {url}",
        )


class AsyncS3Datasets:
    """Asynchronous session-only S3 operations."""

    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def get_session(self) -> Any:
        return await self._client.call("datasets.s3.get_session")

    async def configure_session(
        self,
        *,
        endpoint: str,
        bucket: str,
        access_key: str,
        secret_key: str,
        region: str = "auto",
        if_revision: int | None = None,
    ) -> Any:
        return await self._client.call(
            "datasets.s3.configure_session",
            _with_revision(
                {
                    "endpoint": endpoint,
                    "region": region,
                    "bucket": bucket,
                    "access_key": access_key,
                    "secret_key": secret_key,
                },
                if_revision,
            ),
        )

    async def clear_session(self, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "datasets.s3.clear_session", _with_revision({}, if_revision)
        )

    async def list(self, prefix: str = "") -> Any:
        return await self._client.tasks.start(
            "datasets.s3.list", {"prefix": prefix}, label=f"List S3 {prefix or '/'}"
        )

    async def open(self, prefix: str, *, if_revision: int | None = None) -> Any:
        return await self._client.tasks.start(
            "datasets.open_s3",
            _with_revision({"prefix": prefix}, if_revision),
            label=f"Open S3 OME-Zarr {prefix}",
        )


class AsyncDeepLinks:
    """Asynchronous deep-link operations."""

    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def parse(self, url: str) -> Any:
        return await self._client.call("deep_links.parse", {"url": url})

    async def resolve(self, value: str | Mapping[str, Any]) -> Any:
        params = {"url": value} if isinstance(value, str) else {"request": dict(value)}
        return await self._client.call("deep_links.resolve", params)

    async def get_filters(self, value: str | Mapping[str, Any]) -> Any:
        params = {"url": value} if isinstance(value, str) else {"request": dict(value)}
        return await self._client.call("deep_links.filters.get", params)

    async def generate(
        self,
        request: Mapping[str, Any] | None = None,
        *,
        include_project: bool = True,
        roi: str | None = None,
    ) -> Any:
        params: dict[str, Any] = {"include_project": include_project}
        if request is not None:
            params["request"] = dict(request)
        if roi is not None:
            params["roi"] = roi
        return await self._client.call("deep_links.generate", params)

    async def apply(
        self,
        value: str | Mapping[str, Any],
        *,
        if_revision: int | None = None,
    ) -> Any:
        params = {"url": value} if isinstance(value, str) else {"request": dict(value)}
        return await self._client.tasks.start(
            "deep_links.apply",
            _with_revision(params, if_revision),
            label="Apply Odon deep link",
        )


class AsyncViewer:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def get_camera(self) -> Any:
        return await self._client.call("viewer.camera.get")

    async def get_state(self) -> Any:
        return await self._client.application.get_state()

    @property
    def cached_state(self) -> Any:
        return self._client.application.cached_state

    async def set_camera(
        self,
        *,
        center: Sequence[float] | None = None,
        zoom: float | None = None,
        if_revision: int | None = None,
    ) -> Any:
        params: dict[str, Any] = {}
        if center is not None:
            if len(center) != 2:
                raise ValueError("center must contain exactly two coordinates")
            params["center_world_lvl0"] = [float(center[0]), float(center[1])]
        if zoom is not None:
            params["zoom"] = float(zoom)
        return await self._client.call(
            "viewer.camera.set", _with_revision(params, if_revision)
        )

    async def fit(self, *, if_revision: int | None = None) -> Any:
        return await self._client.call("viewer.camera.fit", _with_revision({}, if_revision))

    async def zoom_in(
        self, factor: float | None = None, *, if_revision: int | None = None
    ) -> Any:
        params = {} if factor is None else {"factor": factor}
        return await self._client.call(
            "viewer.camera.zoom_in", _with_revision(params, if_revision)
        )

    async def zoom_out(
        self, factor: float | None = None, *, if_revision: int | None = None
    ) -> Any:
        params = {} if factor is None else {"factor": factor}
        return await self._client.call(
            "viewer.camera.zoom_out", _with_revision(params, if_revision)
        )

    async def get_smooth_pixels(self) -> Any:
        return await self._client.call("viewer.rendering.get_smooth_pixels")

    async def get_rendering_state(self) -> Any:
        return await self._client.call("viewer.rendering.get_state")

    async def set_smooth_pixels(
        self, smooth: bool, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "viewer.rendering.set_smooth_pixels",
            _with_revision({"smooth": smooth}, if_revision),
        )

    async def set_right_tab(self, tab: str, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "viewer.ui.set_right_tab", _with_revision({"tab": tab}, if_revision)
        )

    async def get_scale_bar(self) -> Any:
        return await self._client.call("viewer.scale_bar.get")

    async def set_scale_bar(
        self, visible: bool, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "viewer.scale_bar.set",
            _with_revision({"visible": visible}, if_revision),
        )

    async def get_side_panels(self) -> Any:
        return await self._client.call("viewer.panels.get")

    async def set_side_panels(
        self,
        *,
        left: bool | None = None,
        right: bool | None = None,
        if_revision: int | None = None,
    ) -> Any:
        params = {
            key: value
            for key, value in {"left": left, "right": right}.items()
            if value is not None
        }
        if not params:
            raise ValueError("left and/or right is required")
        return await self._client.call(
            "viewer.panels.set", _with_revision(params, if_revision)
        )


class AsyncChannels:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def list(self) -> Any:
        return await self._client.call("viewer.channels.list")

    async def list_visible(self) -> Any:
        return await self._client.call("viewer.channels.list_visible")

    async def get_active(self) -> Any:
        return await self._client.call("viewer.channels.get_active")

    async def set_active(
        self, channel: str | int, *, if_revision: int | None = None
    ) -> Any:
        key = "index" if isinstance(channel, int) else "name"
        return await self._client.call(
            "viewer.channels.set_active", _with_revision({key: channel}, if_revision)
        )

    async def set_visible(
        self,
        channels: Iterable[str | int],
        *,
        mode: str = "only",
        if_revision: int | None = None,
    ) -> Any:
        return await self._client.call(
            "viewer.channels.set_visible",
            _with_revision({"channels": list(channels), "mode": mode}, if_revision),
        )

    async def get_contrast(self, channel: str | int | None = None) -> Any:
        params: dict[str, Any] = {}
        if channel is not None:
            params["index" if isinstance(channel, int) else "name"] = channel
        return await self._client.call("viewer.channels.get_contrast", params)

    async def set_contrast(
        self,
        channel: str | int,
        *,
        minimum: float,
        maximum: float,
        if_revision: int | None = None,
    ) -> Any:
        params: dict[str, Any] = {
            "index" if isinstance(channel, int) else "name": channel,
            "min": float(minimum),
            "max": float(maximum),
        }
        return await self._client.call(
            "viewer.channels.set_contrast", _with_revision(params, if_revision)
        )

    async def set_color(
        self,
        channel: str | int,
        color_rgb: Sequence[int],
        *,
        if_revision: int | None = None,
    ) -> Any:
        if len(color_rgb) != 3:
            raise ValueError("color_rgb must contain exactly three components")
        params = {
            "index" if isinstance(channel, int) else "name": channel,
            "color_rgb": [int(component) for component in color_rgb],
        }
        return await self._client.call(
            "viewer.channels.set_color", _with_revision(params, if_revision)
        )

    async def set_note(
        self, channel: str | int, note: str, *, if_revision: int | None = None
    ) -> Any:
        params = {
            "index" if isinstance(channel, int) else "name": channel,
            "note": note,
        }
        return await self._client.call(
            "viewer.channels.set_note", _with_revision(params, if_revision)
        )

    async def get_transform(self, channel: str | int) -> Any:
        key = "index" if isinstance(channel, int) else "name"
        return await self._client.call("viewer.channels.get_transform", {key: channel})

    async def set_transform(
        self,
        channel: str | int,
        *,
        offset_world: Sequence[float] | None = None,
        scale: Sequence[float] | None = None,
        rotation_rad: float | None = None,
        if_revision: int | None = None,
    ) -> Any:
        params: dict[str, Any] = {
            "index" if isinstance(channel, int) else "name": channel
        }
        for key, value in (("offset_world", offset_world), ("scale", scale)):
            if value is not None:
                if len(value) != 2:
                    raise ValueError(f"{key} must contain exactly two components")
                params[key] = [float(value[0]), float(value[1])]
        if rotation_rad is not None:
            params["rotation_rad"] = float(rotation_rad)
        if len(params) == 1:
            raise ValueError("offset_world, scale, and/or rotation_rad is required")
        return await self._client.call(
            "viewer.channels.set_transform", _with_revision(params, if_revision)
        )

    async def reset_transform(
        self, channel: str | int, *, if_revision: int | None = None
    ) -> Any:
        key = "index" if isinstance(channel, int) else "name"
        return await self._client.call(
            "viewer.channels.reset_transform",
            _with_revision({key: channel}, if_revision),
        )

    async def set_order(
        self, channels: Iterable[str | int], *, if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "viewer.channels.set_order",
            _with_revision({"channels": list(channels)}, if_revision),
        )

    async def get_presentation(self) -> Any:
        return await self._client.call("viewer.channels.presentation.get")

    async def set_presentation(
        self,
        *,
        search: str | None = None,
        sort: str | None = None,
        if_revision: int | None = None,
    ) -> Any:
        params = {key: value for key, value in (("search", search), ("sort", sort)) if value is not None}
        if not params:
            raise ValueError("search and/or sort is required")
        return await self._client.call(
            "viewer.channels.presentation.set", _with_revision(params, if_revision)
        )

    async def list_groups(self) -> Any:
        return await self._client.call("viewer.channels.list_groups")

    async def set_group(self, *, if_revision: int | None = None, **params: Any) -> Any:
        return await self._client.call(
            "viewer.channels.set_group", _with_revision(params, if_revision)
        )

    async def intensity_stats(self, **params: Any) -> Any:
        return await self._client.tasks.start(
            "viewer.channels.intensity_stats",
            params,
            label="Compute channel intensity statistics",
        )


class AsyncPlanes:
    """Orientation and slice navigation for multidimensional datasets."""

    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def get(self) -> Any:
        return await self._client.call("viewer.planes.get")

    async def get_operation_availability(self) -> Any:
        return await self._client.call("viewer.planes.operation_availability")

    async def set(
        self,
        *,
        mode: str | None = None,
        slice: int | None = None,
        if_revision: int | None = None,
    ) -> Any:
        params: dict[str, Any] = {}
        if mode is not None:
            params["mode"] = mode
        if slice is not None:
            params["slice"] = slice
        if not params:
            raise ValueError("mode and/or slice is required")
        return await self._client.call(
            "viewer.planes.set", _with_revision(params, if_revision)
        )

    async def next(
        self, step: int = 1, *, wrap: bool = False, if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "viewer.planes.next",
            _with_revision({"step": step, "wrap": wrap}, if_revision),
        )

    async def previous(
        self, step: int = 1, *, wrap: bool = False, if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "viewer.planes.previous",
            _with_revision({"step": step, "wrap": wrap}, if_revision),
        )


class AsyncNativeLayers:
    """Odon's built-in channel, segmentation, mask, annotation, and spatial layers."""

    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def list(self) -> Any:
        return await self._client.call("viewer.native_layers.list")

    async def get(self, layer_id: str) -> Any:
        return await self._client.call(
            "viewer.native_layers.get", {"layer_id": layer_id}
        )

    async def set_active(
        self, layer_id: str, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "viewer.native_layers.set_active",
            _with_revision({"layer_id": layer_id}, if_revision),
        )

    async def set_visibility(
        self,
        layer_id: str,
        visible: bool,
        *,
        if_revision: int | None = None,
    ) -> Any:
        return await self._client.call(
            "viewer.native_layers.set_visibility",
            _with_revision({"layer_id": layer_id, "visible": visible}, if_revision),
        )

    async def set_order(
        self,
        stack: str,
        layers: Iterable[str],
        *,
        if_revision: int | None = None,
    ) -> Any:
        return await self._client.call(
            "viewer.native_layers.set_order",
            _with_revision({"stack": stack, "layers": list(layers)}, if_revision),
        )

    async def set_offset(
        self,
        layer_id: str,
        offset_world: Sequence[float],
        *,
        if_revision: int | None = None,
    ) -> Any:
        if len(offset_world) != 2:
            raise ValueError("offset_world must contain exactly two components")
        return await self._client.call(
            "viewer.native_layers.set_offset",
            _with_revision(
                {
                    "layer_id": layer_id,
                    "offset_world": [float(offset_world[0]), float(offset_world[1])],
                },
                if_revision,
            ),
        )

    async def reset_offset(
        self, layer_id: str, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "viewer.native_layers.reset_offset",
            _with_revision({"layer_id": layer_id}, if_revision),
        )


class AsyncProjects:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def list_rois(self) -> Any:
        return await self._client.call("project.rois.list")

    async def get(self) -> Any:
        return await self._client.call("project.get")

    async def create(
        self, *, default_dataset: str | None = None, if_revision: int | None = None
    ) -> Any:
        params = {} if default_dataset is None else {"default_dataset": default_dataset}
        return await self._client.call(
            "project.create", _with_revision(params, if_revision)
        )

    async def open(self, path: str | Path, *, if_revision: int | None = None) -> Any:
        return await self._client.tasks.start(
            "project.open",
            _with_revision({"path": str(path)}, if_revision),
            label=f"Open project {path}",
        )

    async def save(self, *, if_revision: int | None = None) -> Any:
        return await self._client.call("project.save", _with_revision({}, if_revision))

    async def save_as(
        self, path: str | Path, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "project.save_as", _with_revision({"path": str(path)}, if_revision)
        )

    async def update_metadata(
        self, *, if_revision: int | None = None, **changes: Any
    ) -> Any:
        return await self._client.call(
            "project.update_metadata", _with_revision(changes, if_revision)
        )

    async def open_roi(
        self, roi: str | int, *, if_revision: int | None = None, **params: Any
    ) -> Any:
        key = "index" if isinstance(roi, int) else "id"
        return await self._client.tasks.start(
            "project.rois.open",
            _with_revision({key: roi, **params}, if_revision),
            label=f"Open ROI {roi}",
        )


class AsyncProjectSamplesheets:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def inspect(
        self, path: str | Path, *, offset: int = 0, limit: int = 200
    ) -> Any:
        return await self._client.call(
            "project.samplesheets.inspect",
            {"path": str(path), "offset": offset, "limit": limit},
        )

    async def validate(
        self, path: str | Path, *, offset: int = 0, limit: int = 200
    ) -> Any:
        return await self._client.call(
            "project.samplesheets.validate",
            {"path": str(path), "offset": offset, "limit": limit},
        )

    async def import_(
        self, path: str | Path, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.tasks.start(
            "project.samplesheets.import",
            _with_revision({"path": str(path)}, if_revision),
            label=f"Import samplesheet {path}",
        )

    async def export(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
        if_revision: int | None = None,
    ) -> Any:
        return await self._client.call(
            "project.samplesheets.export",
            _with_revision(
                {"path": str(path), "overwrite": overwrite}, if_revision
            ),
        )


class AsyncProjectDiscovery:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def add_root(
        self, path: str | Path, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.tasks.start(
            "project.discovery.add_root",
            _with_revision({"path": str(path)}, if_revision),
            label=f"Discover datasets under {path}",
        )


class AsyncProjectObjects:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def get_preload(self) -> Any:
        return await self._client.call("project.objects.preload.get")

    async def list_preload_sources(
        self, *, offset: int = 0, limit: int = 200
    ) -> Any:
        return await self._client.call(
            "project.objects.preload.list_sources",
            {"offset": offset, "limit": limit},
        )

    async def preload(
        self,
        *,
        mode: str = "full_geometry",
        lazy_properties: bool = True,
        if_revision: int | None = None,
    ) -> Any:
        return await self._client.tasks.start(
            "project.objects.preload.start",
            _with_revision(
                {"mode": mode, "lazy_properties": lazy_properties}, if_revision
            ),
            label="Preload project objects",
        )

    async def clear_preload(self, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "project.objects.preload.clear", _with_revision({}, if_revision)
        )


class AsyncProjectRois:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def list(self) -> Any:
        return await self._client.call("project.rois.list")

    async def get(self, roi_id: str) -> Any:
        return await self._client.call("project.rois.get", {"id": roi_id})

    async def add(
        self,
        roi_id: str,
        path: str | Path,
        *,
        display_name: str | None = None,
        dataset: str | None = None,
        segmentation_path: str | Path | None = None,
        metadata: Mapping[str, str] | None = None,
        if_revision: int | None = None,
    ) -> Any:
        params: dict[str, Any] = {"id": roi_id, "path": str(path)}
        if display_name is not None:
            params["display_name"] = display_name
        if dataset is not None:
            params["dataset"] = dataset
        if segmentation_path is not None:
            params["segmentation_path"] = str(segmentation_path)
        if metadata is not None:
            params["metadata"] = dict(metadata)
        return await self._client.call(
            "project.rois.add", _with_revision(params, if_revision)
        )

    async def update(
        self, roi_id: str, *, if_revision: int | None = None, **changes: Any
    ) -> Any:
        normalized = dict(changes)
        for key in ("path", "segmentation_path"):
            if key in normalized and normalized[key] is not None:
                normalized[key] = str(normalized[key])
        return await self._client.call(
            "project.rois.update",
            _with_revision({"target_id": roi_id, "changes": normalized}, if_revision),
        )

    async def remove(self, roi_id: str, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "project.rois.remove", _with_revision({"id": roi_id}, if_revision)
        )

    async def reorder(
        self, ids: Iterable[str], *, if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "project.rois.reorder", _with_revision({"ids": list(ids)}, if_revision)
        )

    async def get_selection(self) -> Any:
        return await self._client.call("project.rois.get_selection")

    async def select(
        self,
        ids: Iterable[str],
        *,
        mode: str = "replace",
        if_revision: int | None = None,
    ) -> Any:
        return await self._client.call(
            "project.rois.select",
            _with_revision({"ids": list(ids), "mode": mode}, if_revision),
        )

    async def focus(self, roi_id: str, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "project.rois.focus", _with_revision({"id": roi_id}, if_revision)
        )

    async def next(
        self, step: int = 1, *, wrap: bool = True, if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "project.rois.next",
            _with_revision({"step": step, "wrap": wrap}, if_revision),
        )

    async def previous(
        self, step: int = 1, *, wrap: bool = True, if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "project.rois.previous",
            _with_revision({"step": step, "wrap": wrap}, if_revision),
        )

    async def open(
        self, roi: str | int, *, if_revision: int | None = None, **params: Any
    ) -> Any:
        key = "index" if isinstance(roi, int) else "id"
        return await self._client.tasks.start(
            "project.rois.open",
            _with_revision({key: roi, **params}, if_revision),
            label=f"Open ROI {roi}",
        )

    async def open_selected_mosaic(self, *, if_revision: int | None = None) -> Any:
        return await self._client.tasks.start(
            "project.rois.open_selected_mosaic",
            _with_revision({}, if_revision),
            label="Open selected ROIs as mosaic",
        )


class AsyncProjectViews:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    @staticmethod
    def _selector(view: str | int) -> dict[str, str | int]:
        return {"index" if isinstance(view, int) else "name": view}

    async def list(self) -> Any:
        return await self._client.call("project.views.list")

    async def get(self, view: str | int) -> Any:
        return await self._client.call("project.views.get", self._selector(view))

    async def create(
        self,
        name: str,
        spec: Mapping[str, Any] | None = None,
        *,
        if_revision: int | None = None,
    ) -> Any:
        params: dict[str, Any] = {"name": name}
        if spec is not None:
            params["spec"] = dict(spec)
        return await self._client.call(
            "project.views.create", _with_revision(params, if_revision)
        )

    async def capture(self, name: str, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "project.views.capture",
            _with_revision({"name": name}, if_revision),
        )

    async def rename(
        self,
        view: str | int,
        new_name: str,
        *,
        if_revision: int | None = None,
    ) -> Any:
        return await self._client.call(
            "project.views.rename",
            _with_revision({**self._selector(view), "new_name": new_name}, if_revision),
        )

    async def delete(
        self, view: str | int, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "project.views.delete",
            _with_revision(self._selector(view), if_revision),
        )

    async def apply(self, view: str | int, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "project.views.apply",
            _with_revision(self._selector(view), if_revision),
        )


class AsyncScreenshots:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def capture(
        self,
        path: str | Path | None = None,
        *,
        overwrite: bool = False,
        if_revision: int | None = None,
    ) -> Any:
        params = {} if path is None else {"path": str(path)}
        params["overwrite"] = overwrite
        return await self._client.tasks.start(
            "viewer.screenshot.capture",
            _with_revision(params, if_revision),
            label="Capture screenshot",
        )

    async def capture_window(
        self,
        path: str | Path | None = None,
        *,
        if_revision: int | None = None,
        **params: Any,
    ) -> Any:
        if path is not None:
            params["path"] = str(path)
        return await self._client.tasks.start(
            "app.screenshot.capture",
            _with_revision(params, if_revision),
            label="Capture Odon window",
        )

    async def capture_project(
        self,
        path: str | Path | None = None,
        *,
        if_revision: int | None = None,
        **params: Any,
    ) -> Any:
        if path is not None:
            params["path"] = str(path)
        return await self._client.tasks.start(
            "project.screenshot.capture",
            _with_revision(params, if_revision),
            label="Capture project page",
        )

    async def get_settings(self) -> Any:
        return await self._client.call("viewer.screenshot.settings.get")

    async def set_settings(
        self,
        *,
        output_dir: str | Path | None = None,
        clear_output_dir: bool = False,
        include_scale_bar: bool | None = None,
        include_legend: bool | None = None,
        scale_bar_scale: float | None = None,
        legend_scale: float | None = None,
        if_revision: int | None = None,
    ) -> Any:
        if output_dir is not None and clear_output_dir:
            raise ValueError("output_dir and clear_output_dir are mutually exclusive")
        params: dict[str, Any] = {}
        if output_dir is not None:
            params["output_dir"] = str(output_dir)
        elif clear_output_dir:
            params["output_dir"] = None
        for key, value in (
            ("include_scale_bar", include_scale_bar),
            ("include_legend", include_legend),
            ("scale_bar_scale", scale_bar_scale),
            ("legend_scale", legend_scale),
        ):
            if value is not None:
                params[key] = value
        return await self._client.call(
            "viewer.screenshot.settings.set",
            _with_revision(params, if_revision),
        )


class AsyncLabels:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def list(self) -> Any:
        return await self._client.call("viewer.labels.list")

    async def get(self) -> Any:
        return await self._client.call("viewer.labels.get")

    async def load(
        self, name: str | None = None, *, if_revision: int | None = None
    ) -> Any:
        params = {} if name is None else {"name": name}
        return await self._client.call(
            "viewer.labels.load", _with_revision(params, if_revision)
        )

    async def unload(self, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "viewer.labels.unload", _with_revision({}, if_revision)
        )

    async def set_visibility(
        self,
        visible: bool,
        *,
        name: str | None = None,
        if_revision: int | None = None,
    ) -> Any:
        params: dict[str, Any] = {"visible": visible}
        if name is not None:
            params["name"] = name
        return await self._client.call(
            "viewer.labels.set_visibility", _with_revision(params, if_revision)
        )


class AsyncMemory:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def get(self) -> Any:
        return await self._client.call("memory.get")

    async def pin(
        self,
        level: int,
        *,
        channels: Sequence[str | int] | None = None,
        scope: str | None = None,
        item: str | int | None = None,
        force: bool = False,
        if_revision: int | None = None,
    ) -> Any:
        params: dict[str, Any] = {"level": level, "force": force}
        if channels is not None:
            params["channels"] = list(channels)
        if scope is not None:
            params["scope"] = scope
        if item is not None:
            params["item"] = item
        return await self._client.tasks.start(
            "memory.pin",
            _with_revision(params, if_revision),
            label=f"Pin image level {level} in RAM",
        )

    async def unpin(
        self,
        level: int,
        *,
        scope: str | None = None,
        item: str | int | None = None,
        if_revision: int | None = None,
    ) -> Any:
        params: dict[str, Any] = {"level": level}
        if scope is not None:
            params["scope"] = scope
        if item is not None:
            params["item"] = item
        return await self._client.call(
            "memory.unpin",
            _with_revision(params, if_revision),
        )

    async def unpin_all(self, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "memory.unpin_all",
            _with_revision({}, if_revision),
        )

    async def get_tile_loading(self) -> Any:
        return await self._client.call("memory.tiles.get")

    async def set_tile_loading(
        self,
        *,
        workers: int | None = None,
        prefetch_mode: str | None = None,
        prefetch_aggressiveness: str | None = None,
        prefer_pinned_finer_levels: bool | None = None,
        if_revision: int | None = None,
    ) -> Any:
        params: dict[str, Any] = {}
        for key, value in (
            ("workers", workers),
            ("prefetch_mode", prefetch_mode),
            ("prefetch_aggressiveness", prefetch_aggressiveness),
            ("prefer_pinned_finer_levels", prefer_pinned_finer_levels),
        ):
            if value is not None:
                params[key] = value
        return await self._client.call(
            "memory.tiles.set",
            _with_revision(params, if_revision),
        )


class AsyncObjects:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def get_overlay_visibility(self, **selector: Any) -> Any:
        return await self._client.call("viewer.objects.get_visibility", selector)

    async def get_state(self, **selector: Any) -> Any:
        return await self._client.call("viewer.objects.get_state", selector)

    async def load(
        self,
        path: str | Path,
        *,
        downsample_factor: float = 1.0,
        if_revision: int | None = None,
    ) -> Any:
        return await self._client.tasks.start(
            "viewer.objects.source.load",
            _with_revision(
                {"path": str(path), "downsample_factor": downsample_factor},
                if_revision,
            ),
            label=f"Load objects from {path}",
        )

    async def reload(self, *, if_revision: int | None = None) -> Any:
        return await self._client.tasks.start(
            "viewer.objects.source.reload",
            _with_revision({}, if_revision),
            label="Reload object source",
        )

    async def clear(self, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "viewer.objects.source.clear", _with_revision({}, if_revision)
        )

    async def cancel_load(self, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "viewer.objects.source.cancel_load", _with_revision({}, if_revision)
        )

    async def get_style(self, **selector: Any) -> Any:
        return await self._client.call("viewer.objects.style.get", selector)

    async def set_style(
        self, *, if_revision: int | None = None, **params: Any
    ) -> Any:
        return await self._client.call(
            "viewer.objects.style.set", _with_revision(params, if_revision)
        )

    async def set_legend(
        self,
        entries: Iterable[Mapping[str, Any]],
        *,
        if_revision: int | None = None,
        **selector: Any,
    ) -> Any:
        return await self._client.call(
            "viewer.objects.legend.set",
            _with_revision(
                {**selector, "entries": [dict(entry) for entry in entries]},
                if_revision,
            ),
        )

    async def get_fast_rendering(self, **selector: Any) -> Any:
        return await self._client.call("viewer.objects.rendering.get_fast", selector)

    async def set_fast_rendering(
        self, enabled: bool, *, if_revision: int | None = None, **selector: Any
    ) -> Any:
        return await self._client.call(
            "viewer.objects.rendering.set_fast",
            _with_revision({**selector, "enabled": enabled}, if_revision),
        )

    async def list_properties(
        self, *, offset: int = 0, limit: int = 200, **selector: Any
    ) -> Any:
        return await self._client.call(
            "viewer.objects.properties.list",
            {**selector, "offset": offset, "limit": limit},
        )

    async def load_property(
        self, property: str, *, if_revision: int | None = None, **selector: Any
    ) -> Any:
        return await self._client.tasks.start(
            "viewer.objects.properties.load",
            _with_revision({**selector, "property": property}, if_revision),
            label=f"Load object property {property}",
        )

    async def get_property_values(
        self,
        property: str,
        *,
        offset: int = 0,
        limit: int = 200,
        **selector: Any,
    ) -> Any:
        return await self._client.call(
            "viewer.objects.properties.values",
            {**selector, "property": property, "offset": offset, "limit": limit},
        )

    async def set_overlay_visibility(
        self, visible: bool, *, if_revision: int | None = None, **selector: Any
    ) -> Any:
        return await self._client.call(
            "viewer.objects.set_visibility",
            _with_revision({**selector, "visible": visible}, if_revision),
        )

    async def get_selection(self, **selector: Any) -> Any:
        return await self._client.call("viewer.objects.get_selection", selector)

    async def query_rect(self, rect: Sequence[float], **selector: Any) -> Any:
        return await self._client.call(
            "viewer.objects.query_rect", {**selector, "rect": list(rect)}
        )

    async def query_view(self, **selector: Any) -> Any:
        return await self._client.call("viewer.objects.query_view", selector)

    async def query_lasso(
        self, points: Iterable[Sequence[float]], **selector: Any
    ) -> Any:
        return await self._client.call(
            "viewer.objects.query_lasso",
            {**selector, "world_points": [list(point) for point in points]},
        )

    async def select_rect(
        self,
        rect: Sequence[float],
        *,
        mode: str = "replace",
        if_revision: int | None = None,
        **selector: Any,
    ) -> Any:
        return await self._client.call(
            "viewer.objects.select_rect",
            _with_revision(
                {**selector, "rect": list(rect), "mode": mode}, if_revision
            ),
        )

    async def select_lasso(
        self,
        points: Iterable[Sequence[float]],
        *,
        mode: str = "replace",
        if_revision: int | None = None,
        **selector: Any,
    ) -> Any:
        return await self._client.call(
            "viewer.objects.select_lasso",
            _with_revision(
                {
                    **selector,
                    "world_points": [list(point) for point in points],
                    "mode": mode,
                },
                if_revision,
            ),
        )

    async def clear_selection(
        self, *, if_revision: int | None = None, **selector: Any
    ) -> Any:
        return await self._client.call(
            "viewer.objects.clear_selection", _with_revision(selector, if_revision)
        )

    async def select_ids(
        self,
        ids: Iterable[str],
        *,
        mode: str = "replace",
        if_revision: int | None = None,
        **selector: Any,
    ) -> Any:
        return await self._client.call(
            "viewer.objects.selection.select_ids",
            _with_revision({**selector, "ids": list(ids), "mode": mode}, if_revision),
        )

    async def select_filtered(
        self,
        *,
        mode: str = "replace",
        if_revision: int | None = None,
        **selector: Any,
    ) -> Any:
        return await self._client.call(
            "viewer.objects.selection.select_filtered",
            _with_revision({**selector, "mode": mode}, if_revision),
        )

    async def focus(
        self,
        value: str | int,
        *,
        fit: bool = True,
        if_revision: int | None = None,
        **selector: Any,
    ) -> Any:
        key = "index" if isinstance(value, int) else "id"
        return await self._client.call(
            "viewer.objects.focus.set",
            _with_revision({**selector, key: value, "fit": fit}, if_revision),
        )

    async def clear_focus(
        self, *, if_revision: int | None = None, **selector: Any
    ) -> Any:
        return await self._client.call(
            "viewer.objects.focus.clear", _with_revision(selector, if_revision)
        )

    async def get_filter(self, **selector: Any) -> Any:
        return await self._client.call("viewer.objects.get_filter", selector)

    async def get_filter_revision(self, **selector: Any) -> Any:
        return await self._client.call("viewer.objects.filters.get_revision", selector)

    async def set_filter(
        self, query: str, *, if_revision: int | None = None, **selector: Any
    ) -> Any:
        return await self._client.call(
            "viewer.objects.set_filter",
            _with_revision({**selector, "query": query}, if_revision),
        )

    async def set_filter_model(
        self,
        *,
        mode: str = "simple",
        clauses: Iterable[Mapping[str, Any]] | None = None,
        logic: str = "all",
        query: str | None = None,
        if_revision: int | None = None,
        **selector: Any,
    ) -> Any:
        params: dict[str, Any] = {**selector, "mode": mode}
        if mode == "simple":
            params.update(
                clauses=[dict(clause) for clause in (clauses or [])], logic=logic
            )
        if query is not None:
            params["query"] = query
        return await self._client.call(
            "viewer.objects.filters.set_model",
            _with_revision(params, if_revision),
        )

    async def clear_filter(
        self, *, if_revision: int | None = None, **selector: Any
    ) -> Any:
        return await self._client.call(
            "viewer.objects.clear_filter", _with_revision(selector, if_revision)
        )


class AsyncMasks:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def list_layers(self) -> Any:
        return await self._client.call("viewer.masks.layers.list")

    async def get_layer(self, layer_id: int) -> Any:
        return await self._client.call("viewer.masks.layers.get", {"id": layer_id})

    async def create_layer(
        self,
        name: str | None = None,
        *,
        editable: bool = True,
        if_revision: int | None = None,
    ) -> Any:
        params: dict[str, Any] = {"editable": editable}
        if name is not None:
            params["name"] = name
        return await self._client.call(
            "viewer.masks.layers.create", _with_revision(params, if_revision)
        )

    async def update_layer(
        self, layer_id: int, *, if_revision: int | None = None, **changes: Any
    ) -> Any:
        return await self._client.call(
            "viewer.masks.layers.update",
            _with_revision({"id": layer_id, **changes}, if_revision),
        )

    async def delete_layer(
        self, layer_id: int, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "viewer.masks.layers.delete",
            _with_revision({"id": layer_id}, if_revision),
        )

    async def list_polygons(
        self, layer_id: int, *, offset: int = 0, limit: int = 200
    ) -> Any:
        return await self._client.call(
            "viewer.masks.polygons.list",
            {"id": layer_id, "offset": offset, "limit": limit},
        )

    async def add_polygon(
        self,
        layer_id: int,
        vertices: Iterable[Sequence[float]],
        *,
        coordinate_space: str = "world",
        if_revision: int | None = None,
    ) -> Any:
        return await self._client.call(
            "viewer.masks.polygons.add",
            _with_revision(
                {
                    "id": layer_id,
                    "vertices": [list(vertex) for vertex in vertices],
                    "coordinate_space": coordinate_space,
                },
                if_revision,
            ),
        )

    async def update_polygon(
        self,
        layer_id: int,
        index: int,
        vertices: Iterable[Sequence[float]],
        *,
        coordinate_space: str = "world",
        if_revision: int | None = None,
    ) -> Any:
        return await self._client.call(
            "viewer.masks.polygons.update",
            _with_revision(
                {
                    "id": layer_id,
                    "index": index,
                    "vertices": [list(vertex) for vertex in vertices],
                    "coordinate_space": coordinate_space,
                },
                if_revision,
            ),
        )

    async def remove_polygon(
        self, layer_id: int, index: int, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "viewer.masks.polygons.remove",
            _with_revision({"id": layer_id, "index": index}, if_revision),
        )

    async def get_selection(self) -> Any:
        return await self._client.call("viewer.masks.selection.get")

    async def select(
        self,
        layer_id: int,
        index: int,
        *,
        vertex_index: int | None = None,
        if_revision: int | None = None,
    ) -> Any:
        return await self._client.call(
            "viewer.masks.selection.set",
            _with_revision(
                {
                    "id": layer_id,
                    "index": index,
                    "vertex_index": vertex_index,
                },
                if_revision,
            ),
        )

    async def clear_selection(self, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "viewer.masks.selection.clear", _with_revision({}, if_revision)
        )

    async def undo(self, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "viewer.masks.undo", _with_revision({}, if_revision)
        )

    async def import_geojson(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        editable: bool = True,
        downsample_factor: float = 1.0,
        if_revision: int | None = None,
    ) -> Any:
        params: dict[str, Any] = {
            "path": str(path),
            "editable": editable,
            "downsample_factor": downsample_factor,
        }
        if name is not None:
            params["name"] = name
        return await self._client.call(
            "viewer.masks.import_geojson", _with_revision(params, if_revision)
        )

    async def export_geojson(
        self,
        path: str | Path,
        *,
        layer_id: int | None = None,
        overwrite: bool = False,
    ) -> Any:
        params: dict[str, Any] = {"path": str(path), "overwrite": overwrite}
        if layer_id is not None:
            params["id"] = layer_id
        return await self._client.call("viewer.masks.export_geojson", params)

    async def get_persistence(self) -> Any:
        return await self._client.call("viewer.masks.persistence.get")

    async def sync_to_project(self, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "viewer.masks.persistence.sync", _with_revision({}, if_revision)
        )


class AsyncThresholds:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    @staticmethod
    def _params(
        *,
        scope: str | None = None,
        level: int | None = None,
        channel: int | str | None = None,
        threshold: int | None = None,
        min_component_pixels: int | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if scope is not None:
            params["scope"] = scope
        if level is not None:
            params["level"] = level
        if channel is not None:
            params["channel"] = channel
        if threshold is not None:
            params["threshold"] = threshold
        if min_component_pixels is not None:
            params["min_component_pixels"] = min_component_pixels
        return params

    async def list_levels(self) -> Any:
        return await self._client.call("viewer.thresholds.levels.list")

    async def get_preview(self) -> Any:
        return await self._client.call("viewer.thresholds.preview.get")

    async def configure(
        self,
        *,
        scope: str | None = None,
        level: int | None = None,
        channel: int | str | None = None,
        threshold: int | None = None,
        min_component_pixels: int | None = None,
        if_revision: int | None = None,
    ) -> Any:
        params = self._params(
            scope=scope,
            level=level,
            channel=channel,
            threshold=threshold,
            min_component_pixels=min_component_pixels,
        )
        return await self._client.call(
            "viewer.thresholds.preview.configure",
            _with_revision(params, if_revision),
        )

    async def start_preview(
        self,
        *,
        scope: str | None = None,
        level: int | None = None,
        channel: int | str | None = None,
        threshold: int | None = None,
        min_component_pixels: int | None = None,
        if_revision: int | None = None,
    ) -> Any:
        params = self._params(
            scope=scope,
            level=level,
            channel=channel,
            threshold=threshold,
            min_component_pixels=min_component_pixels,
        )
        return await self._client.tasks.start(
            "viewer.thresholds.preview.start",
            _with_revision(params, if_revision),
            label="Start threshold preview",
        )

    async def refresh_preview(self, *, if_revision: int | None = None) -> Any:
        return await self._client.tasks.start(
            "viewer.thresholds.preview.refresh",
            _with_revision({}, if_revision),
            label="Refresh threshold preview",
        )

    async def apply_preview(self, *, if_revision: int | None = None) -> Any:
        return await self._client.tasks.start(
            "viewer.thresholds.preview.apply",
            _with_revision({}, if_revision),
            label="Apply threshold preview",
        )

    async def cancel_preview(self, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "viewer.thresholds.preview.cancel", _with_revision({}, if_revision)
        )


def _object_target(target: str, layer_id: int | None) -> dict[str, Any]:
    params: dict[str, Any] = {"target": target}
    if layer_id is not None:
        params["layer_id"] = layer_id
    return params


class AsyncAnalysis:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def get(self, *, target: str = "objects", layer_id: int | None = None) -> Any:
        return await self._client.call("viewer.analysis.get", _object_target(target, layer_id))

    async def set(self, state: Mapping[str, Any], *, target: str = "objects", layer_id: int | None = None, if_revision: int | None = None) -> Any:
        params = {**_object_target(target, layer_id), "state": dict(state)}
        return await self._client.call("viewer.analysis.set", _with_revision(params, if_revision))

    async def histogram(self, property: str, *, bins: int = 128, transform: str = "none", target: str = "objects", layer_id: int | None = None) -> Any:
        return await self._client.call("viewer.analysis.histogram", {**_object_target(target, layer_id), "property": property, "bins": bins, "transform": transform})

    async def suggest_thresholds(self, property: str, *, method: str = "quantiles", count: int = 3, transform: str = "none", target: str = "objects", layer_id: int | None = None) -> Any:
        return await self._client.call("viewer.analysis.suggest_thresholds", {**_object_target(target, layer_id), "property": property, "method": method, "count": count, "transform": transform})

    async def get_warmup(self, *, target: str = "objects", layer_id: int | None = None) -> Any:
        return await self._client.call("viewer.analysis.warmup.get", _object_target(target, layer_id))

    async def warmup(self, *, target: str = "objects", layer_id: int | None = None, if_revision: int | None = None) -> Any:
        return await self._client.tasks.start("viewer.analysis.warmup.start", _with_revision(_object_target(target, layer_id), if_revision), label="Warm object analysis")

    async def import_preset(self, path: str | Path, *, target: str = "objects", layer_id: int | None = None, if_revision: int | None = None) -> Any:
        params = {**_object_target(target, layer_id), "path": str(path)}
        return await self._client.call("viewer.analysis.presets.import", _with_revision(params, if_revision))

    async def export_preset(self, path: str | Path, *, overwrite: bool = False, target: str = "objects", layer_id: int | None = None) -> Any:
        return await self._client.call("viewer.analysis.presets.export", {**_object_target(target, layer_id), "path": str(path), "overwrite": overwrite})


class AsyncMeasurements:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def get(self, *, target: str = "objects", layer_id: int | None = None) -> Any:
        return await self._client.call("viewer.measurements.get", _object_target(target, layer_id))

    async def configure(self, *, metric: str | None = None, level: int | None = None, concurrency: int | None = None, filtered_only: bool | None = None, prefix: str | None = None, target: str = "objects", layer_id: int | None = None, if_revision: int | None = None) -> Any:
        params = _object_target(target, layer_id)
        for key, value in {"metric": metric, "level": level, "concurrency": concurrency, "filtered_only": filtered_only, "prefix": prefix}.items():
            if value is not None:
                params[key] = value
        return await self._client.call("viewer.measurements.configure", _with_revision(params, if_revision))

    async def start(self, *, if_revision: int | None = None, **configuration: Any) -> Any:
        target = str(configuration.pop("target", "objects"))
        layer_id = configuration.pop("layer_id", None)
        params = {**_object_target(target, layer_id), **configuration}
        return await self._client.tasks.start("viewer.measurements.start", _with_revision(params, if_revision), label="Measure polygon intensities")

    async def cancel(self, *, target: str = "objects", layer_id: int | None = None, if_revision: int | None = None) -> Any:
        return await self._client.call("viewer.measurements.cancel", _with_revision(_object_target(target, layer_id), if_revision))

    async def list_generated_properties(self, *, target: str = "objects", layer_id: int | None = None) -> Any:
        return await self._client.call("viewer.measurements.properties.list", _object_target(target, layer_id))


class AsyncObjectExports:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def list_columns(self, *, target: str = "objects", layer_id: int | None = None) -> Any:
        return await self._client.call("exports.objects.columns", _object_target(target, layer_id))

    async def get_state(self, *, target: str = "objects", layer_id: int | None = None) -> Any:
        return await self._client.call("exports.objects.get_state", _object_target(target, layer_id))

    async def export(self, path: str | Path, *, format: str | None = None, scope: str = "all", columns: Iterable[str] | None = None, overwrite: bool = False, target: str = "objects", layer_id: int | None = None, if_revision: int | None = None) -> Any:
        params: dict[str, Any] = {**_object_target(target, layer_id), "path": str(path), "scope": scope, "overwrite": overwrite}
        if format is not None:
            params["format"] = format
        if columns is not None:
            params["columns"] = list(columns)
        return await self._client.tasks.start("exports.objects.start", _with_revision(params, if_revision), label=f"Export objects to {path}")

    async def export_csv(self, path: str | Path, **options: Any) -> Any:
        params = self._export_params(path, **options)
        return await self._client.tasks.start("exports.objects.export_csv", params, label=f"Export CSV to {path}")

    async def export_geoparquet(self, path: str | Path, **options: Any) -> Any:
        params = self._export_params(path, **options)
        return await self._client.tasks.start("exports.objects.export_geoparquet", params, label=f"Export GeoParquet to {path}")

    @staticmethod
    def _export_params(path: str | Path, **options: Any) -> dict[str, Any]:
        target = str(options.pop("target", "objects"))
        layer_id = options.pop("layer_id", None)
        if_revision = options.pop("if_revision", None)
        columns = options.pop("columns", None)
        params: dict[str, Any] = {**_object_target(target, layer_id), "path": str(path), **options}
        if columns is not None:
            params["columns"] = list(columns)
        return _with_revision(params, if_revision)


class AsyncMosaic:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def configure_layout(
        self, *, if_revision: int | None = None, **params: Any
    ) -> Any:
        return await self._client.call(
            "mosaic.layout.configure", _with_revision(params, if_revision)
        )

    async def get_state(self) -> Any:
        return await self._client.call("mosaic.get_state")

    async def list_items(self, *, offset: int = 0, limit: int = 200) -> Any:
        return await self._client.call(
            "mosaic.items.list", {"offset": offset, "limit": limit}
        )

    async def get_selection(self) -> Any:
        return await self._client.call("mosaic.selection.get")

    async def select(
        self,
        ids: Iterable[str],
        *,
        mode: str = "replace",
        if_revision: int | None = None,
    ) -> Any:
        return await self._client.call(
            "mosaic.selection.set",
            _with_revision({"ids": list(ids), "mode": mode}, if_revision),
        )

    async def select_all(self, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "mosaic.selection.set", _with_revision({"mode": "all"}, if_revision)
        )

    async def select_range(
        self,
        start: str,
        end: str,
        *,
        if_revision: int | None = None,
    ) -> Any:
        return await self._client.call(
            "mosaic.selection.set",
            _with_revision(
                {"mode": "range", "start": start, "end": end}, if_revision
            ),
        )

    async def clear_selection(self, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "mosaic.selection.clear", _with_revision({}, if_revision)
        )

    async def get_focus(self) -> Any:
        return await self._client.call("mosaic.focus.get")

    async def set_focus(
        self,
        roi: str | int,
        *,
        fit: bool = True,
        if_revision: int | None = None,
    ) -> Any:
        key = "index" if isinstance(roi, int) else "roi_id"
        return await self._client.call(
            "mosaic.focus.set",
            _with_revision({key: roi, "fit": fit}, if_revision),
        )

    async def next(
        self, step: int = 1, *, wrap: bool = True, if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "mosaic.focus.next",
            _with_revision({"step": step, "wrap": wrap}, if_revision),
        )

    async def previous(
        self, step: int = 1, *, wrap: bool = True, if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "mosaic.focus.previous",
            _with_revision({"step": step, "wrap": wrap}, if_revision),
        )

    async def fit_focus(self, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "mosaic.focus.fit", _with_revision({}, if_revision)
        )

    async def clear_focus(self, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "mosaic.focus.clear", _with_revision({}, if_revision)
        )

    async def fit_all(self, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "mosaic.fit_all", _with_revision({}, if_revision)
        )

    async def get_object_state(self) -> Any:
        return await self._client.call("mosaic.objects.get_state")

    async def load_selected_objects(
        self, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.tasks.start(
            "mosaic.objects.load_selected",
            _with_revision({}, if_revision),
            label="Load selected mosaic objects",
        )

    async def cancel_object_load(self, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "mosaic.objects.cancel_load", _with_revision({}, if_revision)
        )

    async def set_right_tab(
        self, tab: str, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "mosaic.ui.set_right_tab", _with_revision({"tab": tab}, if_revision)
        )
