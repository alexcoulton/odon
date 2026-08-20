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
        return await self._client.call("get_loading_state")

    async def list_methods(self) -> Any:
        return await self._client.call("system.describe_methods")

    async def describe_events(self) -> Any:
        return await self._client.call("system.describe_events")

    async def get_diagnostics(self) -> Any:
        return await self._client.call("system.get_diagnostics")

    async def open_ome_zarr(
        self, path: str | Path, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.tasks.start(
            "open_ome_zarr",
            _with_revision({"path": str(path)}, if_revision),
            label=f"Open {path}",
        )

    async def open_tiff(
        self, path: str | Path, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.tasks.start(
            "open_tiff",
            _with_revision({"path": str(path)}, if_revision),
            label=f"Open {path}",
        )

    async def open_mosaic_samplesheet(
        self, path: str | Path, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.tasks.start(
            "open_mosaic_samplesheet",
            _with_revision({"path": str(path)}, if_revision),
            label=f"Open mosaic {path}",
        )

    async def show_project_page(self, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "show_project_page", _with_revision({}, if_revision)
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
        return await self._client.call("set_camera", _with_revision(params, if_revision))

    async def fit(self, *, if_revision: int | None = None) -> Any:
        return await self._client.call("fit_to_view", _with_revision({}, if_revision))

    async def zoom_in(
        self, factor: float | None = None, *, if_revision: int | None = None
    ) -> Any:
        params = {} if factor is None else {"factor": factor}
        return await self._client.call("zoom_in", _with_revision(params, if_revision))

    async def zoom_out(
        self, factor: float | None = None, *, if_revision: int | None = None
    ) -> Any:
        params = {} if factor is None else {"factor": factor}
        return await self._client.call("zoom_out", _with_revision(params, if_revision))

    async def get_smooth_pixels(self) -> Any:
        return await self._client.call("get_smooth_pixels")

    async def set_smooth_pixels(
        self, smooth: bool, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "set_smooth_pixels", _with_revision({"smooth": smooth}, if_revision)
        )

    async def set_right_tab(self, tab: str, *, if_revision: int | None = None) -> Any:
        return await self._client.call(
            "set_right_tab", _with_revision({"tab": tab}, if_revision)
        )

    async def get_side_panels(self) -> Any:
        return await self._client.call("get_side_panels")

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
            "set_side_panels", _with_revision(params, if_revision)
        )


class AsyncChannels:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def list(self) -> Any:
        return await self._client.call("list_channels")

    async def list_visible(self) -> Any:
        return await self._client.call("list_visible_channels")

    async def get_active(self) -> Any:
        return await self._client.call("get_active_channel")

    async def set_active(
        self, channel: str | int, *, if_revision: int | None = None
    ) -> Any:
        key = "index" if isinstance(channel, int) else "name"
        return await self._client.call(
            "set_active_channel", _with_revision({key: channel}, if_revision)
        )

    async def set_visible(
        self,
        channels: Iterable[str | int],
        *,
        mode: str = "only",
        if_revision: int | None = None,
    ) -> Any:
        return await self._client.call(
            "set_visible_channels",
            _with_revision({"channels": list(channels), "mode": mode}, if_revision),
        )

    async def get_contrast(self, channel: str | int | None = None) -> Any:
        params: dict[str, Any] = {}
        if channel is not None:
            params["index" if isinstance(channel, int) else "name"] = channel
        return await self._client.call("get_channel_contrast", params)

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
            "set_channel_contrast", _with_revision(params, if_revision)
        )

    async def set_order(
        self, channels: Iterable[str | int], *, if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "set_channel_order",
            _with_revision({"channels": list(channels)}, if_revision),
        )

    async def list_groups(self) -> Any:
        return await self._client.call("list_channel_groups")

    async def set_group(self, *, if_revision: int | None = None, **params: Any) -> Any:
        return await self._client.call(
            "set_channel_group", _with_revision(params, if_revision)
        )

    async def intensity_stats(self, **params: Any) -> Any:
        return await self._client.tasks.start(
            "get_channel_intensity_stats", params, label="Compute channel intensity statistics"
        )


class AsyncProjects:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def list_rois(self) -> Any:
        return await self._client.call("list_project_rois")

    async def open(self, path: str | Path, *, if_revision: int | None = None) -> Any:
        return await self._client.tasks.start(
            "open_project",
            _with_revision({"path": str(path)}, if_revision),
            label=f"Open project {path}",
        )

    async def save(self, *, if_revision: int | None = None) -> Any:
        return await self._client.call("save_project", _with_revision({}, if_revision))

    async def open_roi(
        self, roi: str | int, *, if_revision: int | None = None, **params: Any
    ) -> Any:
        key = "index" if isinstance(roi, int) else "id"
        return await self._client.tasks.start(
            "open_roi",
            _with_revision({key: roi, **params}, if_revision),
            label=f"Open ROI {roi}",
        )


class AsyncScreenshots:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def capture(
        self, path: str | Path | None = None, *, if_revision: int | None = None
    ) -> Any:
        params = {} if path is None else {"path": str(path)}
        return await self._client.tasks.start(
            "capture_screenshot",
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
            "capture_window_screenshot",
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
            "capture_project_screenshot",
            _with_revision(params, if_revision),
            label="Capture project page",
        )


class AsyncObjects:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def get_overlay_visibility(self, **selector: Any) -> Any:
        return await self._client.call("get_object_overlay_visibility", selector)

    async def set_overlay_visibility(
        self, visible: bool, *, if_revision: int | None = None, **selector: Any
    ) -> Any:
        return await self._client.call(
            "set_object_overlay_visibility",
            _with_revision({**selector, "visible": visible}, if_revision),
        )

    async def get_selection(self, **selector: Any) -> Any:
        return await self._client.call("get_object_selection", selector)

    async def query_rect(self, rect: Sequence[float], **selector: Any) -> Any:
        return await self._client.call("query_object_ids_in_rect", {**selector, "rect": list(rect)})

    async def query_view(self, **selector: Any) -> Any:
        return await self._client.call("query_object_ids_in_view", selector)

    async def select_rect(
        self,
        rect: Sequence[float],
        *,
        if_revision: int | None = None,
        **selector: Any,
    ) -> Any:
        return await self._client.call(
            "select_object_ids_in_rect",
            _with_revision({**selector, "rect": list(rect)}, if_revision),
        )

    async def clear_selection(
        self, *, if_revision: int | None = None, **selector: Any
    ) -> Any:
        return await self._client.call(
            "clear_object_selection", _with_revision(selector, if_revision)
        )

    async def get_filter(self, **selector: Any) -> Any:
        return await self._client.call("get_object_filter", selector)

    async def set_filter(
        self, query: str, *, if_revision: int | None = None, **selector: Any
    ) -> Any:
        return await self._client.call(
            "set_object_filter_query",
            _with_revision({**selector, "query": query}, if_revision),
        )

    async def clear_filter(
        self, *, if_revision: int | None = None, **selector: Any
    ) -> Any:
        return await self._client.call(
            "clear_object_filter", _with_revision(selector, if_revision)
        )


class AsyncMosaic:
    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def configure_layout(
        self, *, if_revision: int | None = None, **params: Any
    ) -> Any:
        return await self._client.call(
            "configure_mosaic_layout", _with_revision(params, if_revision)
        )

    async def set_right_tab(
        self, tab: str, *, if_revision: int | None = None
    ) -> Any:
        return await self._client.call(
            "set_mosaic_right_tab", _with_revision({"tab": tab}, if_revision)
        )
