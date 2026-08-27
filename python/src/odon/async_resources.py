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


def _compact(params: Mapping[str, Any]) -> dict[str, Any]:
    """Drop only absent optional parameters while preserving nested JSON nulls."""

    return {key: value for key, value in params.items() if value is not None}


def _continuous_color_mapping(
    property: str,
    *,
    palette: str | Iterable[Mapping[str, Any]] = "viridis",
    domain: str | Sequence[float] = "auto",
    scale: str = "linear",
    reverse: bool = False,
    out_of_range: str = "clamp",
    missing_color_rgb: Sequence[int] | None = None,
) -> dict[str, Any]:
    if not property.strip():
        raise ValueError("property must not be empty")
    palette_value: Any = palette if isinstance(palette, str) else [dict(stop) for stop in palette]
    if not isinstance(palette, str) and len(palette_value) < 2:
        raise ValueError("custom palettes require at least two stops")
    if isinstance(domain, str):
        if domain != "auto":
            raise ValueError("domain must be 'auto' or a two-number sequence")
        domain_value: Any = domain
    else:
        domain_value = [float(value) for value in domain]
        if len(domain_value) != 2 or domain_value[0] >= domain_value[1]:
            raise ValueError("domain must contain two numbers with min < max")
    if scale not in {"linear", "log10"}:
        raise ValueError("scale must be 'linear' or 'log10'")
    if scale == "log10" and domain_value != "auto" and domain_value[0] <= 0:
        raise ValueError("log10 domains must be greater than zero")
    if out_of_range not in {"clamp", "hide"}:
        raise ValueError("out_of_range must be 'clamp' or 'hide'")
    missing = None if missing_color_rgb is None else list(missing_color_rgb)
    if missing is not None and (
        len(missing) != 3
        or any(not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 255 for value in missing)
    ):
        raise ValueError("missing_color_rgb must contain three integers from 0 to 255")
    return {
        "mode": "continuous",
        "property": property,
        "palette": palette_value,
        "domain": domain_value,
        "scale": scale,
        "reverse": reverse,
        "out_of_range": out_of_range,
        "missing_color_rgb": missing,
    }


def _with_viewport_revision(
    params: Mapping[str, Any],
    if_revision: int | None,
    *,
    navigation: int | None = None,
    presentation: int | None = None,
) -> dict[str, Any]:
    result = _with_revision(params, if_revision)
    if navigation is not None:
        result["if_navigation_revision"] = navigation
    if presentation is not None:
        result["if_presentation_revision"] = presentation
    return result


def _next_control_revision(result: Any) -> int | None:
    """Extract the global revision returned by the control protocol."""

    if not isinstance(result, Mapping):
        return None
    control = result.get("_control")
    if not isinstance(control, Mapping):
        return None
    revision = control.get("revision")
    return revision if isinstance(revision, int) and not isinstance(revision, bool) else None


def _filter_source(
    *,
    viewport_id: str | None = None,
    filter_query: str | None = None,
    use_all_objects: bool = False,
    use_active_viewport_filter: bool = False,
) -> dict[str, Any]:
    selected = sum(
        (
            viewport_id is not None,
            filter_query is not None,
            use_all_objects,
            use_active_viewport_filter,
        )
    )
    if selected > 1:
        raise ValueError("choose exactly one filter source")
    if viewport_id is not None:
        return {"viewport_id": viewport_id}
    if filter_query is not None:
        return {"filter_query": filter_query}
    if use_all_objects:
        return {"use_all_objects": True}
    if use_active_viewport_filter:
        return {"use_active_viewport_filter": True}
    return {}


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
        show_extension_manager: bool | None = None,
        shell_layout_startup_profiles: Mapping[str, str] | None = None,
        if_revision: int | None = None,
    ) -> Any:
        params: dict[str, Any] = {}
        if auto_contrast is not None:
            params["auto_contrast"] = dict(auto_contrast)
        if fast_object_rendering is not None:
            params["fast_object_rendering"] = fast_object_rendering
        if show_extension_manager is not None:
            params["show_extension_manager"] = show_extension_manager
        if shell_layout_startup_profiles is not None:
            params["shell_layout_startup_profiles"] = dict(
                shell_layout_startup_profiles
            )
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
        """Fit content after Rust has completed the first canvas layout."""
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

    async def set_left_tab(
        self, tab: str, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "viewer.ui.set_left_tab", _with_revision({"tab": tab}, if_revision)
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


class AsyncViewport:
    """A stable asynchronous handle to one native Odon viewport."""

    def __init__(self, client: "AsyncClient", viewport_id: str) -> None:
        self._client = client
        self.id = viewport_id
        self.objects = AsyncViewportObjects(self)

    async def get(self) -> Any:
        return await self._client.call(
            "viewer.viewports.get", {"viewport_id": self.id}
        )

    async def set_active(self, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "viewer.viewports.set_active",
            _with_revision({"viewport_id": self.id}, if_revision),
        )

    async def rename(
        self,
        title: str,
        *,
        if_revision: int | None = None,
        if_presentation_revision: int | None = None,
    ) -> Any:
        return await self._client.call(
            "viewer.viewports.rename",
            _with_viewport_revision(
                {"viewport_id": self.id, "title": title},
                if_revision,
                presentation=if_presentation_revision,
            ),
        )

    async def remove(self, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "viewer.viewports.remove",
            _with_revision({"viewport_id": self.id}, if_revision),
        )

    async def get_camera(self) -> Any:
        return await self._client.call(
            "viewer.viewports.camera.get", {"viewport_id": self.id}
        )

    async def set_camera(
        self,
        *,
        center: Sequence[float] | None = None,
        zoom: float | None = None,
        if_revision: int | None = None,
        if_navigation_revision: int | None = None,
    ) -> Any:
        params: dict[str, Any] = {"viewport_id": self.id}
        if center is not None:
            if len(center) != 2:
                raise ValueError("center must contain exactly two coordinates")
            params["center_world_lvl0"] = [float(center[0]), float(center[1])]
        if zoom is not None:
            params["zoom"] = float(zoom)
        return await self._client.call(
            "viewer.viewports.camera.set",
            _with_viewport_revision(
                params, if_revision, navigation=if_navigation_revision
            ),
        )

    async def fit_camera(
        self,
        *,
        if_revision: int | None = None,
        if_navigation_revision: int | None = None,
    ) -> Any:
        """Fit the image after Rust has laid out this viewport."""
        return await self._client.call(
            "viewer.viewports.camera.fit",
            _with_viewport_revision(
                {"viewport_id": self.id},
                if_revision,
                navigation=if_navigation_revision,
            ),
        )

    async def get_plane(self) -> Any:
        return await self._client.call(
            "viewer.viewports.planes.get", {"viewport_id": self.id}
        )

    async def set_plane(
        self,
        *,
        mode: str | None = None,
        slice: int | None = None,
        if_revision: int | None = None,
        if_navigation_revision: int | None = None,
    ) -> Any:
        params: dict[str, Any] = {"viewport_id": self.id}
        if mode is not None:
            params["mode"] = mode
        if slice is not None:
            params["slice"] = slice
        return await self._client.call(
            "viewer.viewports.planes.set",
            _with_viewport_revision(
                params, if_revision, navigation=if_navigation_revision
            ),
        )

    async def list_channels(self) -> Any:
        return await self._client.call(
            "viewer.viewports.channels.get", {"viewport_id": self.id}
        )

    async def set_visible_channels(
        self,
        channels: Iterable[str | int],
        *,
        mode: str = "only",
        if_revision: int | None = None,
        if_presentation_revision: int | None = None,
    ) -> Any:
        return await self._client.call(
            "viewer.viewports.channels.set_visible",
            _with_viewport_revision(
                {
                    "viewport_id": self.id,
                    "channels": list(channels),
                    "mode": mode,
                },
                if_revision,
                presentation=if_presentation_revision,
            ),
        )

    async def set_channels(
        self,
        channels: Iterable[str | int],
        *,
        mode: str = "only",
        if_revision: int | None = None,
        if_presentation_revision: int | None = None,
    ) -> Any:
        """Set the viewport's visible channel collection via the canonical API."""
        return await self._client.call(
            "viewer.viewports.channels.set",
            _with_viewport_revision(
                {
                    "viewport_id": self.id,
                    "channels": list(channels),
                    "mode": mode,
                },
                if_revision,
                presentation=if_presentation_revision,
            ),
        )

    async def set_active_channel(
        self,
        channel: str | int,
        *,
        if_revision: int | None = None,
        if_presentation_revision: int | None = None,
    ) -> Any:
        return await self._client.call(
            "viewer.viewports.channels.set_active",
            _with_viewport_revision(
                {"viewport_id": self.id, "channel": channel},
                if_revision,
                presentation=if_presentation_revision,
            ),
        )

    async def set_channel_color(
        self,
        channel: str | int,
        color_rgb: Iterable[int],
        *,
        if_revision: int | None = None,
        if_presentation_revision: int | None = None,
    ) -> Any:
        return await self._client.call(
            "viewer.viewports.channels.set_color",
            _with_viewport_revision(
                {
                    "viewport_id": self.id,
                    "channel": channel,
                    "color_rgb": list(color_rgb),
                },
                if_revision,
                presentation=if_presentation_revision,
            ),
        )

    async def set_channel_contrast(
        self,
        channel: str | int,
        minimum: float,
        maximum: float,
        *,
        if_revision: int | None = None,
        if_presentation_revision: int | None = None,
    ) -> Any:
        return await self._client.call(
            "viewer.viewports.channels.set_contrast",
            _with_viewport_revision(
                {
                    "viewport_id": self.id,
                    "channel": channel,
                    "min": minimum,
                    "max": maximum,
                },
                if_revision,
                presentation=if_presentation_revision,
            ),
        )

    async def set_channel_order(
        self,
        channels: Iterable[str | int],
        *,
        mode: str = "exact",
        if_revision: int | None = None,
        if_presentation_revision: int | None = None,
    ) -> Any:
        return await self._client.call(
            "viewer.viewports.channels.set_order",
            _with_viewport_revision(
                {
                    "viewport_id": self.id,
                    "channels": list(channels),
                    "mode": mode,
                },
                if_revision,
                presentation=if_presentation_revision,
            ),
        )

    async def list_channel_groups(self) -> Any:
        return await self._client.call(
            "viewer.viewports.channels.list_groups",
            {"viewport_id": self.id},
        )

    async def set_channel_group(
        self,
        channels: Iterable[str | int],
        *,
        group: str | None = None,
        group_id: int | None = None,
        color_rgb: Iterable[int] | None = None,
        inherit_color: bool = True,
        replace_group_members: bool = False,
        if_revision: int | None = None,
        if_presentation_revision: int | None = None,
    ) -> Any:
        params: dict[str, Any] = {
            "viewport_id": self.id,
            "channels": list(channels),
            "inherit_color": inherit_color,
            "replace_group_members": replace_group_members,
        }
        if group is not None:
            params["group"] = group
        if group_id is not None:
            params["group_id"] = group_id
        if color_rgb is not None:
            params["color_rgb"] = list(color_rgb)
        return await self._client.call(
            "viewer.viewports.channels.set_group",
            _with_viewport_revision(
                params,
                if_revision,
                presentation=if_presentation_revision,
            ),
        )

    async def get_object_style(self) -> Any:
        return await self._client.call(
            "viewer.viewports.objects.style.get", {"viewport_id": self.id}
        )

    async def set_object_style(
        self,
        *,
        if_revision: int | None = None,
        if_presentation_revision: int | None = None,
        **style: Any,
    ) -> Any:
        return await self._client.call(
            "viewer.viewports.objects.style.set",
            _with_viewport_revision(
                {"viewport_id": self.id, **style},
                if_revision,
                presentation=if_presentation_revision,
            ),
        )

    async def set_object_legend(
        self,
        entries: Iterable[Mapping[str, Any]],
        *,
        if_revision: int | None = None,
        if_presentation_revision: int | None = None,
    ) -> Any:
        """Set per-value colours/visibility for this viewport's object property."""
        return await self._client.call(
            "viewer.viewports.objects.legend.set",
            _with_viewport_revision(
                {
                    "viewport_id": self.id,
                    "entries": [dict(entry) for entry in entries],
                },
                if_revision,
                presentation=if_presentation_revision,
            ),
        )

    async def get_object_filter(self) -> Any:
        return await self._client.call(
            "viewer.viewports.objects.filter.get", {"viewport_id": self.id}
        )

    async def set_object_filter(
        self,
        query: str | None = None,
        *,
        mode: str | None = None,
        clauses: Iterable[Mapping[str, Any]] | None = None,
        logic: str | None = None,
        if_revision: int | None = None,
        if_presentation_revision: int | None = None,
    ) -> Any:
        params: dict[str, Any] = {"viewport_id": self.id}
        if query is not None:
            params["query"] = query
        if mode is not None:
            params["mode"] = mode
        if clauses is not None:
            params["clauses"] = [dict(clause) for clause in clauses]
        if logic is not None:
            params["logic"] = logic
        return await self._client.call(
            "viewer.viewports.objects.filter.set",
            _with_viewport_revision(
                params, if_revision, presentation=if_presentation_revision
            ),
        )

    async def clear_object_filter(
        self,
        *,
        if_revision: int | None = None,
        if_presentation_revision: int | None = None,
    ) -> Any:
        return await self._client.call(
            "viewer.viewports.objects.filter.clear",
            _with_viewport_revision(
                {"viewport_id": self.id},
                if_revision,
                presentation=if_presentation_revision,
            ),
        )

    async def get_rendering(self) -> Any:
        """Return this viewport's independent display preferences."""
        return await self._client.call(
            "viewer.viewports.rendering.get", {"viewport_id": self.id}
        )

    async def set_rendering(
        self,
        *,
        smooth_pixels: bool | None = None,
        show_scale_bar: bool | None = None,
        show_hud: bool | None = None,
        show_tile_debug: bool | None = None,
        if_revision: int | None = None,
        if_presentation_revision: int | None = None,
    ) -> Any:
        """Set independent sampling and decoration preferences for this viewport."""
        params: dict[str, Any] = {"viewport_id": self.id}
        for name, value in (
            ("smooth_pixels", smooth_pixels),
            ("show_scale_bar", show_scale_bar),
            ("show_hud", show_hud),
            ("show_tile_debug", show_tile_debug),
        ):
            if value is not None:
                params[name] = value
        return await self._client.call(
            "viewer.viewports.rendering.set",
            _with_viewport_revision(
                params, if_revision, presentation=if_presentation_revision
            ),
        )

    async def list_layers(self) -> Any:
        return await self._client.call(
            "viewer.viewports.layers.list", {"viewport_id": self.id}
        )

    async def get_layer(self, layer_id: str) -> Any:
        """Return one layer and its presentation in this viewport."""
        return await self._client.call(
            "viewer.viewports.layers.get",
            {"viewport_id": self.id, "layer_id": layer_id},
        )

    async def set_layer(
        self,
        layer_id: str,
        presentation: Mapping[str, Any] | None = None,
        *,
        if_revision: int | None = None,
        if_presentation_revision: int | None = None,
        **changes: Any,
    ) -> Any:
        """Update one layer's independent presentation in this viewport."""
        payload = dict(presentation or {})
        payload.update(changes)
        return await self._client.call(
            "viewer.viewports.layers.set",
            _with_viewport_revision(
                {
                    "viewport_id": self.id,
                    "layer_id": layer_id,
                    "presentation": payload,
                },
                if_revision,
                presentation=if_presentation_revision,
            ),
        )

    async def set_layer_visibility(
        self,
        layer_id: str,
        visible: bool,
        *,
        if_revision: int | None = None,
        if_presentation_revision: int | None = None,
    ) -> Any:
        return await self._client.call(
            "viewer.viewports.layers.set_visibility",
            _with_viewport_revision(
                {
                    "viewport_id": self.id,
                    "layer_id": layer_id,
                    "visible": visible,
                },
                if_revision,
                presentation=if_presentation_revision,
            ),
        )

    async def set_layer_order(
        self,
        stack: str,
        layers: Iterable[str],
        *,
        if_revision: int | None = None,
        if_presentation_revision: int | None = None,
    ) -> Any:
        return await self._client.call(
            "viewer.viewports.layers.set_order",
            _with_viewport_revision(
                {
                    "viewport_id": self.id,
                    "stack": stack,
                    "layers": list(layers),
                },
                if_revision,
                presentation=if_presentation_revision,
            ),
        )

    async def set_active_layer(
        self,
        layer_id: str,
        *,
        if_revision: int | None = None,
        if_presentation_revision: int | None = None,
    ) -> Any:
        return await self._client.call(
            "viewer.viewports.layers.set_active",
            _with_viewport_revision(
                {"viewport_id": self.id, "layer_id": layer_id},
                if_revision,
                presentation=if_presentation_revision,
            ),
        )


class AsyncViewportObjects:
    """Async object presentation/filter resource bound to a stable viewport."""

    def __init__(self, viewport: AsyncViewport) -> None:
        self._viewport = viewport

    async def get_style(self) -> Any:
        return await self._viewport.get_object_style()

    async def set_style(
        self,
        *,
        if_revision: int | None = None,
        if_presentation_revision: int | None = None,
        **style: Any,
    ) -> Any:
        return await self._viewport.set_object_style(
            if_revision=if_revision,
            if_presentation_revision=if_presentation_revision,
            **style,
        )

    async def color_by_continuous(
        self,
        property: str,
        *,
        palette: str | Iterable[Mapping[str, Any]] = "viridis",
        domain: str | Sequence[float] = "auto",
        scale: str = "linear",
        reverse: bool = False,
        out_of_range: str = "clamp",
        missing_color_rgb: Sequence[int] | None = None,
        fill_cells: bool | None = None,
        fill_opacity: float | None = None,
        if_revision: int | None = None,
        if_presentation_revision: int | None = None,
    ) -> Any:
        style = {
            "color_mapping": _continuous_color_mapping(
                property,
                palette=palette,
                domain=domain,
                scale=scale,
                reverse=reverse,
                out_of_range=out_of_range,
                missing_color_rgb=missing_color_rgb,
            ),
            **_compact({"fill_cells": fill_cells, "fill_opacity": fill_opacity}),
        }
        return await self.set_style(
            if_revision=if_revision,
            if_presentation_revision=if_presentation_revision,
            **style,
        )

    async def set_legend(
        self,
        entries: Iterable[Mapping[str, Any]],
        *,
        if_revision: int | None = None,
        if_presentation_revision: int | None = None,
    ) -> Any:
        return await self._viewport.set_object_legend(
            entries,
            if_revision=if_revision,
            if_presentation_revision=if_presentation_revision,
        )

    async def get_filter(self) -> Any:
        return await self._viewport.get_object_filter()

    async def set_filter(
        self,
        query: str | None = None,
        *,
        mode: str | None = None,
        clauses: Iterable[Mapping[str, Any]] | None = None,
        logic: str | None = None,
        if_revision: int | None = None,
        if_presentation_revision: int | None = None,
    ) -> Any:
        return await self._viewport.set_object_filter(
            query,
            mode=mode,
            clauses=clauses,
            logic=logic,
            if_revision=if_revision,
            if_presentation_revision=if_presentation_revision,
        )

    async def clear_filter(
        self,
        *,
        if_revision: int | None = None,
        if_presentation_revision: int | None = None,
    ) -> Any:
        return await self._viewport.clear_object_filter(
            if_revision=if_revision,
            if_presentation_revision=if_presentation_revision,
        )


class AsyncViewportComparison:
    def __init__(self, left: AsyncViewport, right: AsyncViewport) -> None:
        self.left = left
        self.right = right
        self.viewports = (left, right)


class AsyncViewportWorkspace:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def get(self) -> Any:
        return await self._client.call("viewer.workspace.get")

    async def get_layout(self) -> Any:
        return await self._client.call("viewer.workspace.layout.get")

    async def set_layout(
        self,
        layout: str | None = None,
        *,
        split: str | None = None,
        viewports: Iterable[AsyncViewport | str] | None = None,
        ratio: float | None = None,
        if_revision: int | None = None,
    ) -> Any:
        if layout is None:
            layout = split
        elif split is not None and split != layout:
            raise ValueError("layout and split must agree when both are provided")
        if layout is None:
            raise ValueError("layout or split is required")
        params: dict[str, Any] = {"layout": layout}
        if viewports is not None:
            params["viewports"] = [
                item.id if isinstance(item, AsyncViewport) else item
                for item in viewports
            ]
        if ratio is not None:
            params["ratio"] = ratio
        return await self._client.call(
            "viewer.workspace.layout.set",
            _with_revision(params, if_revision),
        )

    async def swap(self, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "viewer.workspace.swap", _with_revision({}, if_revision)
        )

    async def get_links(self) -> Any:
        return await self._client.call("viewer.viewport_links.get")

    async def set_links(
        self,
        *,
        camera: bool | None = None,
        plane: bool | None = None,
        selection: bool | None = None,
        if_revision: int | None = None,
    ) -> Any:
        params = {
            key: value
            for key, value in {
                "camera": camera,
                "plane": plane,
                "selection": selection,
            }.items()
            if value is not None
        }
        return await self._client.call(
            "viewer.viewport_links.set", _with_revision(params, if_revision)
        )


class AsyncViewportLinks:
    """Async canonical resource for the fixed comparison link group."""

    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    @staticmethod
    def _fields(fields: Iterable[str]) -> list[str]:
        result = list(dict.fromkeys(fields))
        unknown = set(result).difference({"camera", "plane", "selection"})
        if unknown:
            raise ValueError(f"unknown linked fields: {sorted(unknown)}")
        if "selection" not in result:
            result.append("selection")
        return result

    @staticmethod
    def _viewport_ids(viewports: Iterable[AsyncViewport | str]) -> list[str]:
        return [
            item.id if isinstance(item, AsyncViewport) else item for item in viewports
        ]

    async def list(self) -> Any:
        return await self._client.call("viewer.viewport_links.list")

    async def create(
        self,
        *,
        viewports: Iterable[AsyncViewport | str],
        fields: Iterable[str] = ("camera", "plane", "selection"),
        link_group_id: str = "comparison-navigation",
        if_revision: int | None = None,
    ) -> Any:
        return await self._client.call(
            "viewer.viewport_links.create",
            _with_revision(
                {
                    "link_group_id": link_group_id,
                    "viewports": self._viewport_ids(viewports),
                    "fields": self._fields(fields),
                },
                if_revision,
            ),
        )

    async def update(
        self,
        *,
        fields: Iterable[str],
        viewports: Iterable[AsyncViewport | str] | None = None,
        link_group_id: str = "comparison-navigation",
        if_revision: int | None = None,
    ) -> Any:
        params: dict[str, Any] = {
            "link_group_id": link_group_id,
            "fields": self._fields(fields),
        }
        if viewports is not None:
            params["viewports"] = self._viewport_ids(viewports)
        return await self._client.call(
            "viewer.viewport_links.update", _with_revision(params, if_revision)
        )

    async def remove(
        self,
        link_group_id: str = "comparison-navigation",
        *,
        if_revision: int | None = None,
    ) -> Any:
        return await self._client.call(
            "viewer.viewport_links.remove",
            _with_revision({"link_group_id": link_group_id}, if_revision),
        )


class AsyncViewports:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def list(self) -> Any:
        return await self._client.call("viewer.viewports.list")

    def handle(self, viewport_id: str) -> AsyncViewport:
        return AsyncViewport(self._client, viewport_id)

    async def active(self) -> AsyncViewport:
        workspace = await self._client.call("viewer.workspace.get")
        return self.handle(str(workspace["active_viewport_id"]))

    async def create(
        self,
        *,
        source: AsyncViewport | str | None = None,
        title: str | None = None,
        layout: str = "horizontal",
        ratio: float | None = None,
        activate: bool = True,
        if_revision: int | None = None,
    ) -> AsyncViewport:
        params: dict[str, Any] = {"layout": layout, "activate": activate}
        if ratio is not None:
            params["ratio"] = ratio
        if source is not None:
            params["source_viewport_id"] = (
                source.id if isinstance(source, AsyncViewport) else source
            )
        if title is not None:
            params["title"] = title
        result = await self._client.call(
            "viewer.viewports.create", _with_revision(params, if_revision)
        )
        return self.handle(str(result["viewport_id"]))

    async def clone(
        self,
        source: AsyncViewport | str,
        *,
        title: str | None = None,
        layout: str = "horizontal",
        ratio: float | None = None,
        activate: bool = True,
        if_revision: int | None = None,
    ) -> AsyncViewport:
        """Clone an explicit viewport into the second workspace slot."""
        params: dict[str, Any] = {
            "viewport_id": source.id if isinstance(source, AsyncViewport) else source,
            "layout": layout,
            "activate": activate,
        }
        if ratio is not None:
            params["ratio"] = ratio
        if title is not None:
            params["title"] = title
        result = await self._client.call(
            "viewer.viewports.clone", _with_revision(params, if_revision)
        )
        return self.handle(str(result["viewport_id"]))

    async def compare(
        self,
        *,
        layout: str = "horizontal",
        ratio: float = 0.5,
        titles: Sequence[str] = ("View 1", "View 2"),
        linked: Iterable[str] = ("camera", "plane", "selection"),
        if_revision: int | None = None,
    ) -> AsyncViewportComparison:
        if len(titles) != 2:
            raise ValueError("titles must contain exactly two values")
        fields = list(dict.fromkeys(linked))
        unknown = set(fields).difference({"camera", "plane", "selection"})
        if unknown:
            raise ValueError(f"unknown linked fields: {sorted(unknown)}")
        left = await self.active()
        renamed = await left.rename(titles[0], if_revision=if_revision)
        next_revision = _next_control_revision(renamed)
        created = await self._client.call(
            "viewer.viewports.clone",
            _with_revision(
                {
                    "viewport_id": left.id,
                    "title": titles[1],
                    "layout": layout,
                    "ratio": ratio,
                    "activate": True,
                },
                next_revision,
            ),
        )
        right = self.handle(str(created["viewport_id"]))
        next_revision = _next_control_revision(created)
        await AsyncViewportLinks(self._client).create(
            viewports=[left, right],
            fields=fields,
            if_revision=next_revision,
        )
        return AsyncViewportComparison(left, right)


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

    async def auto_contrast(
        self,
        *,
        channels: Sequence[str | int] | None = None,
        viewport_id: str | None = None,
        overwrite_manual: bool = True,
    ) -> Any:
        params: dict[str, Any] = {"overwrite_manual": overwrite_manual}
        if channels is not None:
            params["channels"] = list(channels)
        if viewport_id is not None:
            params["viewport_id"] = viewport_id
        return await self._client.tasks.start(
            "viewer.channels.auto_contrast",
            params,
            label="Apply automatic channel contrast",
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

    async def capture(
        self,
        name: str,
        *,
        viewport: AsyncViewport | str | None = None,
        if_revision: int | None = None,
    ) -> Any:
        params: dict[str, Any] = {"name": name}
        if viewport is not None:
            params["viewport_id"] = (
                viewport.id if isinstance(viewport, AsyncViewport) else viewport
            )
        return await self._client.call(
            "project.views.capture",
            _with_revision(params, if_revision),
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
        viewport: AsyncViewport | str | None = None,
        overwrite: bool = False,
        if_revision: int | None = None,
    ) -> Any:
        params = {} if path is None else {"path": str(path)}
        if viewport is not None:
            params["viewport_id"] = (
                viewport.id if isinstance(viewport, AsyncViewport) else viewport
            )
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

    async def capture_workspace(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
        if_revision: int | None = None,
    ) -> Any:
        return await self._client.tasks.start(
            "viewer.workspace.screenshot.capture",
            _with_revision(
                {"path": str(path), "overwrite": overwrite}, if_revision
            ),
            label="Capture viewport workspace",
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

    async def get_segmentation_geojson_source(self) -> Any:
        return await self._client.call("viewer.segmentation_geojson.source.get")

    async def load_segmentation_geojson(
        self,
        path: str | Path,
        *,
        downsample_factor: float = 1.0,
        if_revision: int | None = None,
    ) -> Any:
        return await self._client.tasks.start(
            "viewer.segmentation_geojson.source.load",
            _with_revision(
                {"path": str(path), "downsample_factor": downsample_factor},
                if_revision,
            ),
            label=f"Load segmentation GeoJSON from {path}",
        )

    async def reload_segmentation_geojson(
        self, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.tasks.start(
            "viewer.segmentation_geojson.source.reload",
            _with_revision({}, if_revision),
            label="Reload segmentation GeoJSON",
        )

    async def clear_segmentation_geojson(
        self, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "viewer.segmentation_geojson.source.clear",
            _with_revision({}, if_revision),
        )

    async def get_style(self, **selector: Any) -> Any:
        return await self._client.call("viewer.objects.style.get", selector)

    async def set_style(
        self, *, if_revision: int | None = None, **params: Any
    ) -> Any:
        return await self._client.call(
            "viewer.objects.style.set", _with_revision(params, if_revision)
        )

    async def color_by_continuous(
        self,
        property: str,
        *,
        palette: str | Iterable[Mapping[str, Any]] = "viridis",
        domain: str | Sequence[float] = "auto",
        scale: str = "linear",
        reverse: bool = False,
        out_of_range: str = "clamp",
        missing_color_rgb: Sequence[int] | None = None,
        fill_cells: bool | None = None,
        fill_opacity: float | None = None,
        if_revision: int | None = None,
        **selector: Any,
    ) -> Any:
        return await self.set_style(
            if_revision=if_revision,
            **selector,
            color_mapping=_continuous_color_mapping(
                property,
                palette=palette,
                domain=domain,
                scale=scale,
                reverse=reverse,
                out_of_range=out_of_range,
                missing_color_rgb=missing_color_rgb,
            ),
            **_compact({"fill_cells": fill_cells, "fill_opacity": fill_opacity}),
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
        viewport_id: str | None = None,
        filter_query: str | None = None,
        use_all_objects: bool = False,
        use_active_viewport_filter: bool = False,
        if_revision: int | None = None,
        **selector: Any,
    ) -> Any:
        return await self._client.call(
            "viewer.objects.selection.select_filtered",
            _with_revision(
                {
                    **selector,
                    **_filter_source(
                        viewport_id=viewport_id,
                        filter_query=filter_query,
                        use_all_objects=use_all_objects,
                        use_active_viewport_filter=use_active_viewport_filter,
                    ),
                    "mode": mode,
                },
                if_revision,
            ),
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


class AsyncAnnotations:
    """Async actor-owned point annotation layers."""

    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def list_layers(self) -> Any:
        return await self._client.call("viewer.annotations.layers.list")

    async def get_layer(self, layer_id: int) -> Any:
        return await self._client.call(
            "viewer.annotations.layers.get", {"layer_id": layer_id}
        )

    async def create_layer(
        self,
        name: str | None = None,
        *,
        if_revision: int | None = None,
        **state: Any,
    ) -> Any:
        if name is not None:
            state["name"] = name
        return await self._client.call(
            "viewer.annotations.layers.create", _with_revision(state, if_revision)
        )

    async def update_layer(
        self, layer_id: int, *, if_revision: int | None = None, **state: Any
    ) -> Any:
        return await self._client.call(
            "viewer.annotations.layers.update",
            _with_revision({"layer_id": layer_id, "state": state}, if_revision),
        )

    async def delete_layer(
        self, layer_id: int, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "viewer.annotations.layers.delete",
            _with_revision({"layer_id": layer_id}, if_revision),
        )

    @staticmethod
    def _source_params(
        layer_id: int,
        path: str | Path | None = None,
        **columns: str,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"layer_id": layer_id, **columns}
        if path is not None:
            params["path"] = str(path)
        return params

    async def inspect(
        self,
        layer_id: int,
        path: str | Path,
        *,
        if_revision: int | None = None,
        **columns: str,
    ) -> Any:
        return await self._client.tasks.start(
            "viewer.annotations.source.inspect",
            _with_revision(self._source_params(layer_id, path, **columns), if_revision),
            label=f"Inspect annotations from {path}",
        )

    async def load(
        self,
        layer_id: int,
        path: str | Path | None = None,
        *,
        if_revision: int | None = None,
        **columns: str,
    ) -> Any:
        return await self._client.tasks.start(
            "viewer.annotations.source.load",
            _with_revision(self._source_params(layer_id, path, **columns), if_revision),
            label=f"Load annotation layer {layer_id}",
        )

    async def reload(
        self, layer_id: int, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.tasks.start(
            "viewer.annotations.source.reload",
            _with_revision({"layer_id": layer_id}, if_revision),
            label=f"Reload annotation layer {layer_id}",
        )

    async def clear_source(
        self, layer_id: int, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "viewer.annotations.source.clear",
            _with_revision({"layer_id": layer_id}, if_revision),
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
        replace_layer_id: int | None = None,
        expected_generation: int | None = None,
        if_revision: int | None = None,
    ) -> Any:
        params: dict[str, Any] = {
            "path": str(path),
            "editable": editable,
            "downsample_factor": downsample_factor,
        }
        if name is not None:
            params["name"] = name
        if replace_layer_id is not None:
            params["replace_layer_id"] = replace_layer_id
        if expected_generation is not None:
            params["expected_generation"] = expected_generation
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

    async def append_to_geojson(
        self,
        path: str | Path,
        *,
        name: str = "Exclusion masks",
        downsample_factor: float = 1.0,
        roi_root: str | Path | None = None,
        expected_generation: int | None = None,
        if_revision: int | None = None,
    ) -> Any:
        params: dict[str, Any] = {
            "path": str(path),
            "name": name,
            "downsample_factor": downsample_factor,
        }
        if roi_root is not None:
            params["roi_root"] = str(roi_root)
        if expected_generation is not None:
            params["expected_generation"] = expected_generation
        return await self._client.call(
            "viewer.masks.persistence.append_geojson",
            _with_revision(params, if_revision),
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

    async def histogram(
        self,
        property: str,
        *,
        bins: int = 128,
        transform: str = "none",
        target: str = "objects",
        layer_id: int | None = None,
        viewport_id: str | None = None,
        filter_query: str | None = None,
        use_all_objects: bool = False,
        use_active_viewport_filter: bool = False,
    ) -> Any:
        return await self._client.call(
            "viewer.analysis.histogram",
            {
                **_object_target(target, layer_id),
                **_filter_source(
                    viewport_id=viewport_id,
                    filter_query=filter_query,
                    use_all_objects=use_all_objects,
                    use_active_viewport_filter=use_active_viewport_filter,
                ),
                "property": property,
                "bins": bins,
                "transform": transform,
            },
        )

    async def suggest_thresholds(
        self,
        property: str,
        *,
        method: str = "quantiles",
        count: int = 3,
        transform: str = "none",
        target: str = "objects",
        layer_id: int | None = None,
        viewport_id: str | None = None,
        filter_query: str | None = None,
        use_all_objects: bool = False,
        use_active_viewport_filter: bool = False,
    ) -> Any:
        return await self._client.call(
            "viewer.analysis.suggest_thresholds",
            {
                **_object_target(target, layer_id),
                **_filter_source(
                    viewport_id=viewport_id,
                    filter_query=filter_query,
                    use_all_objects=use_all_objects,
                    use_active_viewport_filter=use_active_viewport_filter,
                ),
                "property": property,
                "method": method,
                "count": count,
                "transform": transform,
            },
        )

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

    async def start(
        self,
        *,
        viewport_id: str | None = None,
        filter_query: str | None = None,
        use_all_objects: bool = False,
        use_active_viewport_filter: bool = False,
        if_revision: int | None = None,
        **configuration: Any,
    ) -> Any:
        target = str(configuration.pop("target", "objects"))
        layer_id = configuration.pop("layer_id", None)
        params = {
            **_object_target(target, layer_id),
            **_filter_source(
                viewport_id=viewport_id,
                filter_query=filter_query,
                use_all_objects=use_all_objects,
                use_active_viewport_filter=use_active_viewport_filter,
            ),
            **configuration,
        }
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

    async def export(
        self,
        path: str | Path,
        *,
        format: str | None = None,
        scope: str = "all",
        columns: Iterable[str] | None = None,
        overwrite: bool = False,
        target: str = "objects",
        layer_id: int | None = None,
        viewport_id: str | None = None,
        filter_query: str | None = None,
        use_all_objects: bool = False,
        use_active_viewport_filter: bool = False,
        if_revision: int | None = None,
    ) -> Any:
        params: dict[str, Any] = {
            **_object_target(target, layer_id),
            **_filter_source(
                viewport_id=viewport_id,
                filter_query=filter_query,
                use_all_objects=use_all_objects,
                use_active_viewport_filter=use_active_viewport_filter,
            ),
            "path": str(path),
            "scope": scope,
            "overwrite": overwrite,
        }
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
        filter_source = _filter_source(
            viewport_id=options.pop("viewport_id", None),
            filter_query=options.pop("filter_query", None),
            use_all_objects=bool(options.pop("use_all_objects", False)),
            use_active_viewport_filter=bool(
                options.pop("use_active_viewport_filter", False)
            ),
        )
        params: dict[str, Any] = {
            **_object_target(target, layer_id),
            **filter_source,
            "path": str(path),
            **options,
        }
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

    async def get_object_style(self) -> Any:
        return await self._client.call("mosaic.objects.style.get")

    async def set_object_style(self, **style: Any) -> Any:
        return await self._client.call("mosaic.objects.style.set", {"style": style})

    async def get_object_property_cache(self) -> Any:
        """Get the lazy mosaic object-property retention policy."""
        return await self._client.call("mosaic.objects.property_cache.get")

    async def set_object_property_cache(
        self,
        *,
        policy: str = "lru",
        capacity: int | None = 2,
        if_revision: int | None = None,
    ) -> Any:
        """Bound lazily loaded mosaic properties without unloading object geometry."""
        if policy not in {"lru", "unbounded"}:
            raise ValueError("policy must be 'lru' or 'unbounded'")
        if policy == "lru" and (capacity is None or capacity < 1):
            raise ValueError("LRU property cache capacity must be at least 1")
        if policy == "unbounded":
            capacity = None
        return await self._client.call(
            "mosaic.objects.property_cache.set",
            _with_revision(
                _compact({"policy": policy, "capacity": capacity}), if_revision
            ),
        )

    async def color_objects_by_continuous(
        self,
        property: str,
        *,
        palette: str | Iterable[Mapping[str, Any]] = "viridis",
        domain: str | Sequence[float] = "auto",
        scale: str = "linear",
        reverse: bool = False,
        out_of_range: str = "clamp",
        missing_color_rgb: Sequence[int] | None = None,
        fill_cells: bool | None = None,
        fill_opacity: float | None = None,
    ) -> Any:
        return await self.set_object_style(
            color_mapping=_continuous_color_mapping(
                property,
                palette=palette,
                domain=domain,
                scale=scale,
                reverse=reverse,
                out_of_range=out_of_range,
                missing_color_rgb=missing_color_rgb,
            ),
            **_compact({"fill_cells": fill_cells, "fill_opacity": fill_opacity}),
        )

    async def get_object_selection(
        self, *, item_id: int | None = None, roi_id: str | None = None
    ) -> Any:
        return await self._client.call(
            "mosaic.objects.selection.get",
            _compact({"item_id": item_id, "roi_id": roi_id}),
        )

    async def replace_object_selection(
        self,
        *,
        selected_indices: Sequence[int],
        item_id: int | None = None,
        roi_id: str | None = None,
        primary_index: int | None = None,
        expected_generation: int | None = None,
    ) -> Any:
        return await self._client.call(
            "mosaic.objects.selection.replace",
            _compact(
                {
                    "item_id": item_id,
                    "roi_id": roi_id,
                    "expected_generation": expected_generation,
                    "state": {
                        "selected_indices": list(selected_indices),
                        "primary_index": primary_index,
                    },
                }
            ),
        )

    async def clear_object_selection(
        self, *, item_id: int | None = None, roi_id: str | None = None
    ) -> Any:
        return await self._client.call(
            "mosaic.objects.selection.clear",
            _compact({"item_id": item_id, "roi_id": roi_id}),
        )

    async def load_selected_objects(
        self, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.tasks.start(
            "mosaic.objects.load_selected",
            _with_revision({}, if_revision),
            label="Load selected mosaic objects",
        )

    async def load_objects(
        self,
        *,
        item_ids: Sequence[int] | None = None,
        roi_ids: Sequence[str] | None = None,
        scope: str | None = None,
        downsample_factor: float = 1.0,
        if_revision: int | None = None,
    ) -> Any:
        return await self._client.tasks.start(
            "mosaic.objects.load",
            _with_revision(
                _compact(
                    {
                        "item_ids": list(item_ids) if item_ids is not None else None,
                        "roi_ids": list(roi_ids) if roi_ids is not None else None,
                        "scope": scope,
                        "downsample_factor": downsample_factor,
                    }
                ),
                if_revision,
            ),
            label="Load mosaic objects",
        )

    async def cancel_object_load(self, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "mosaic.objects.cancel_load", _with_revision({}, if_revision)
        )

    async def set_left_tab(
        self, tab: str, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "mosaic.ui.set_left_tab", _with_revision({"tab": tab}, if_revision)
        )

    async def set_rendering(
        self,
        *,
        smooth_pixels: bool | None = None,
        show_tile_debug: bool | None = None,
        if_revision: int | None = None,
    ) -> Any:
        params: dict[str, Any] = {}
        if smooth_pixels is not None:
            params["smooth_pixels"] = smooth_pixels
        if show_tile_debug is not None:
            params["show_tile_debug"] = show_tile_debug
        return await self._client.call(
            "mosaic.rendering.set", _with_revision(params, if_revision)
        )

    async def set_right_tab(
        self, tab: str, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "mosaic.ui.set_right_tab", _with_revision({"tab": tab}, if_revision)
        )
