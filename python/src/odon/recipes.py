"""Inspectable building blocks for safe Python-authored Odon workflows.

These helpers deliberately compose the public SDK.  They do not introduce hidden
protocol commands or make actor mutations atomic; callers can still inspect every
retained task, resource response, and readiness snapshot.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import time
from typing import Any

from .errors import RequestTimeoutError, ResourceNotFoundError


READINESS_FIELDS = (
    "model_ready",
    "resources_ready",
    "geometry_ready",
    "canvas_ready",
    "presentation_ready",
)


class ObjectPropertyUnavailableError(ResourceNotFoundError):
    """A requested numeric property is unavailable in the installed object source."""

    def __init__(self, property_name: str, columns: Sequence[Mapping[str, Any]]) -> None:
        available = sorted(
            str(column.get("name"))
            for column in columns
            if isinstance(column.get("name"), str)
        )
        super().__init__(
            f"object property {property_name!r} is absent or non-numeric; "
            f"available properties: {', '.join(available) or '(none)'}",
            code=-32004,
            kind="RESOURCE_NOT_FOUND",
            data={"property": property_name, "available_properties": available},
        )
        self.property_name = property_name
        self.columns = tuple(dict(column) for column in columns)


@dataclass(frozen=True)
class ObjectSourceStyleResult:
    """Evidence returned after a safe source/style transition is presented."""

    task: Any
    task_result: Any
    object_state: Mapping[str, Any]
    property_descriptor: Mapping[str, Any]
    readiness: Mapping[str, Any]


def _loading(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    loading = value.get("loading", value)
    return loading if isinstance(loading, Mapping) else {}


def _is_ready(loading: Mapping[str, Any]) -> bool:
    if not all(loading.get(field) is True for field in READINESS_FIELDS):
        return False
    projected = loading.get("projection_revision")
    presented = loading.get("presented_projection_revision")
    if isinstance(projected, int) and isinstance(presented, int):
        return presented >= projected
    return True


def wait_for_viewer_readiness(
    app: Any, *, timeout: float = 30.0, poll_interval: float = 0.05
) -> Mapping[str, Any]:
    """Wait for model, resources, geometry, canvas, and presentation readiness."""

    if timeout < 0:
        raise ValueError("readiness timeout must be non-negative")
    if poll_interval <= 0:
        raise ValueError("readiness poll_interval must be positive")
    deadline = time.monotonic() + timeout
    last: Mapping[str, Any] = {}
    while True:
        last = _loading(app.application.get_loading_state())
        if _is_ready(last):
            return last
        if time.monotonic() >= deadline:
            raise RequestTimeoutError(
                "timed out waiting for Odon viewer readiness; actor or renderer work "
                f"may still be running (last readiness: {dict(last)!r})"
            )
        time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))


async def async_wait_for_viewer_readiness(
    app: Any, *, timeout: float = 30.0, poll_interval: float = 0.05
) -> Mapping[str, Any]:
    """Async counterpart of :func:`wait_for_viewer_readiness`."""

    if timeout < 0:
        raise ValueError("readiness timeout must be non-negative")
    if poll_interval <= 0:
        raise ValueError("readiness poll_interval must be positive")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    last: Mapping[str, Any] = {}
    while True:
        last = _loading(await app.application.get_loading_state())
        if _is_ready(last):
            return last
        if loop.time() >= deadline:
            raise RequestTimeoutError(
                "timed out waiting for Odon viewer readiness; actor or renderer work "
                f"may still be running (last readiness: {dict(last)!r})"
            )
        await asyncio.sleep(min(poll_interval, max(0.0, deadline - loop.time())))


def require_numeric_object_property(
    objects: Any, property_name: str
) -> Mapping[str, Any]:
    """Return a loaded numeric property descriptor or raise a structured error."""
    offset = 0
    columns: list[Mapping[str, Any]] = []
    while True:
        page = objects.list_properties(offset=offset, limit=200)
        if not isinstance(page, Mapping) or not isinstance(page.get("columns"), list):
            raise ValueError("object property listing returned an invalid response")
        page_columns = [item for item in page["columns"] if isinstance(item, Mapping)]
        columns.extend(page_columns)
        for column in page_columns:
            if column.get("name") == property_name and column.get("numeric") is True:
                return dict(column)
        if page.get("has_more") is not True:
            raise ObjectPropertyUnavailableError(property_name, columns)
        offset += len(page_columns)
        if not page_columns:
            raise ValueError("object property listing did not advance its pagination")


async def async_require_numeric_object_property(
    objects: Any, property_name: str
) -> Mapping[str, Any]:
    offset = 0
    columns: list[Mapping[str, Any]] = []
    while True:
        page = await objects.list_properties(offset=offset, limit=200)
        if not isinstance(page, Mapping) or not isinstance(page.get("columns"), list):
            raise ValueError("object property listing returned an invalid response")
        page_columns = [item for item in page["columns"] if isinstance(item, Mapping)]
        columns.extend(page_columns)
        for column in page_columns:
            if column.get("name") == property_name and column.get("numeric") is True:
                return dict(column)
        if page.get("has_more") is not True:
            raise ObjectPropertyUnavailableError(property_name, columns)
        offset += len(page_columns)
        if not page_columns:
            raise ValueError("object property listing did not advance its pagination")


def replace_object_source_and_style(
    app: Any,
    path: str,
    property_name: str,
    *,
    palette: Any = "viridis",
    domain: str | Sequence[float] = "auto",
    fill_cells: bool = True,
    fill_opacity: float = 0.65,
    downsample_factor: float = 1.0,
    timeout: float = 60.0,
    poll_interval: float = 0.05,
    context: Any | None = None,
    presentation_objects: Any | None = None,
    publish_ready: bool = True,
) -> ObjectSourceStyleResult:
    """Safely replace the active one-view object source and continuous style.

    The overlay is hidden and its property-dependent mapping is neutralized before
    source replacement.  ``Ready`` is only published after the final projection has
    been observed by the renderer.
    """

    deadline = time.monotonic() + timeout

    def remaining() -> float:
        return max(0.0, deadline - time.monotonic())

    def current() -> None:
        if context is not None:
            context.ensure_current()

    if context is not None:
        context.status(f"Loading {property_name}…")
    current()
    presentation = presentation_objects or app.objects

    def neutralize() -> None:
        if presentation is app.objects:
            presentation.set_overlay_visibility(False)
            presentation.set_style(color_mapping={"mode": "single"})
        else:
            presentation.set_style(visible=False, color_mapping={"mode": "single"})

    if context is None:
        neutralize()
    elif not context.commit(neutralize):
        context.ensure_current()
    wait_for_viewer_readiness(
        app, timeout=remaining(), poll_interval=poll_interval
    )
    current()
    app.objects.clear()
    task = app.objects.load(path, downsample_factor=downsample_factor)
    if context is not None:
        context.attach(task)
    task_result = task.wait(
        timeout=remaining(),
        progress=context.report_task if context is not None else None,
    )
    readiness = wait_for_viewer_readiness(
        app, timeout=remaining(), poll_interval=poll_interval
    )
    current()
    object_state = app.objects.get_state()
    descriptor = require_numeric_object_property(app.objects, property_name)
    current()
    def install_style() -> None:
        presentation.color_by_continuous(
            property_name,
            palette=palette,
            domain=domain,
            fill_cells=fill_cells,
            fill_opacity=fill_opacity,
        )
        if presentation is app.objects:
            presentation.set_overlay_visibility(True)
        else:
            presentation.set_style(visible=True)

    if context is None:
        install_style()
    elif not context.commit(install_style):
        context.ensure_current()
    readiness = wait_for_viewer_readiness(
        app, timeout=remaining(), poll_interval=poll_interval
    )
    current()
    if context is not None and publish_ready:
        context.result("Ready")
    return ObjectSourceStyleResult(
        task=task,
        task_result=task_result,
        object_state=dict(object_state) if isinstance(object_state, Mapping) else {},
        property_descriptor=descriptor,
        readiness=dict(readiness),
    )


async def async_replace_object_source_and_style(
    app: Any,
    path: str,
    property_name: str,
    *,
    palette: Any = "viridis",
    domain: str | Sequence[float] = "auto",
    fill_cells: bool = True,
    fill_opacity: float = 0.65,
    downsample_factor: float = 1.0,
    timeout: float = 60.0,
    poll_interval: float = 0.05,
    context: Any | None = None,
    presentation_objects: Any | None = None,
    publish_ready: bool = True,
) -> ObjectSourceStyleResult:
    """Async safe source/style transition for the active one-view object layer."""

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    def remaining() -> float:
        return max(0.0, deadline - loop.time())

    async def current() -> None:
        if context is not None:
            context.ensure_current()

    if context is not None:
        await context.status(f"Loading {property_name}…")
    await current()
    presentation = presentation_objects or app.objects
    if presentation is app.objects:
        await presentation.set_overlay_visibility(False)
        await presentation.set_style(color_mapping={"mode": "single"})
    else:
        await presentation.set_style(visible=False, color_mapping={"mode": "single"})
    await async_wait_for_viewer_readiness(
        app, timeout=remaining(), poll_interval=poll_interval
    )
    await current()
    await app.objects.clear()
    task = await app.objects.load(path, downsample_factor=downsample_factor)
    if context is not None:
        context.attach(task)
    task_result = await task.wait(
        timeout=remaining(),
        progress=context.report_task if context is not None else None,
    )
    readiness = await async_wait_for_viewer_readiness(
        app, timeout=remaining(), poll_interval=poll_interval
    )
    await current()
    object_state = await app.objects.get_state()
    descriptor = await async_require_numeric_object_property(
        app.objects, property_name
    )
    await current()
    await presentation.color_by_continuous(
        property_name,
        palette=palette,
        domain=domain,
        fill_cells=fill_cells,
        fill_opacity=fill_opacity,
    )
    if presentation is app.objects:
        await presentation.set_overlay_visibility(True)
    else:
        await presentation.set_style(visible=True)
    readiness = await async_wait_for_viewer_readiness(
        app, timeout=remaining(), poll_interval=poll_interval
    )
    await current()
    if context is not None and publish_ready:
        await context.result("Ready")
    return ObjectSourceStyleResult(
        task=task,
        task_result=task_result,
        object_state=dict(object_state) if isinstance(object_state, Mapping) else {},
        property_descriptor=descriptor,
        readiness=dict(readiness),
    )


@dataclass(frozen=True)
class MarkerComparisonState:
    """Synchronized semantic state owned by :class:`MarkerComparisonController`."""

    roi_id: str
    marker: str
    fill: str
    generation: int = 0
    phase: str = "ready"


@dataclass(frozen=True)
class MarkerComparisonComponents:
    """Native panel component IDs patched by the reference controller."""

    roi: str = "roi"
    marker: str = "marker"
    fill: str = "fill-mode"
    status: str = "status"
    progress: str | None = None


class MarkerComparisonController:
    """One-view controller keeping marker channel, object fill, and panel in sync.

    It is intentionally configured with plain mappings/callables rather than a new
    protocol concept.  Every transition runs through one extension-owned serial
    worker and uses one queue key, so a newer ROI, marker, or fill generation can
    prevent an older load from committing stale presentation state.
    """

    def __init__(
        self,
        app: Any,
        extension: Any,
        contribution: Any,
        viewer: Any,
        *,
        rois: Sequence[str],
        markers: Mapping[str, Sequence[str]],
        fills: Sequence[str],
        channel_for: Callable[[str, str], str],
        property_for: Callable[[str, str, str], str],
        source_for: Callable[[str, str], str],
        domain_for: Callable[[str, str, str], str | Sequence[float]],
        initial_state: MarkerComparisonState,
        palette_for: Callable[[str, str, str], Any] | None = None,
        fill_label: Callable[[str], str] | None = None,
        open_roi: Callable[[str], Any] | None = None,
        components: MarkerComparisonComponents = MarkerComparisonComponents(),
        fill_opacity: float = 0.7,
        timeout: float = 60.0,
        queue_key: str = "marker-comparison",
    ) -> None:
        if not rois or not fills:
            raise ValueError("marker comparison requires at least one ROI and fill")
        self.app = app
        self.extension = extension
        self.contribution = contribution
        self.viewer = viewer
        self.rois = tuple(rois)
        self.markers = {key: tuple(value) for key, value in markers.items()}
        if any(not self.markers.get(roi) for roi in self.rois):
            raise ValueError("every marker comparison ROI requires at least one marker")
        self.fills = tuple(fills)
        self.channel_for = channel_for
        self.property_for = property_for
        self.source_for = source_for
        self.domain_for = domain_for
        self.palette_for = palette_for or (lambda _roi, _marker, _fill: "viridis")
        self.fill_label = fill_label or str
        self.open_roi_callback = open_roi
        self.components = components
        self.fill_opacity = fill_opacity
        self.timeout = timeout
        self.queue_key = queue_key
        self.state = initial_state
        self._committed = initial_state
        self._loaded_source: str | None = None
        self._registrations: list[Any] = []
        self.errors: list[BaseException] = []
        self._validate_state(initial_state)

    def _validate_state(self, state: MarkerComparisonState) -> None:
        if state.roi_id not in self.rois:
            raise ValueError(f"unknown ROI {state.roi_id!r}")
        if state.marker not in self.markers[state.roi_id]:
            raise ValueError(
                f"marker {state.marker!r} is unavailable in ROI {state.roi_id!r}"
            )
        if state.fill not in self.fills:
            raise ValueError(f"unknown fill {state.fill!r}")

    def _patch(self, values: Mapping[str, Any]) -> None:
        self.contribution.patch_values(dict(values))

    def _panel_values(self, state: MarkerComparisonState, status: str) -> dict[str, Any]:
        return {
            self.components.roi: state.roi_id,
            self.components.marker: state.marker,
            self.components.fill: self.fill_label(state.fill),
            self.components.status: status,
        }

    def _on_error(self, error: BaseException, context: Any | None) -> None:
        self.errors.append(error)

        def commit_failure() -> None:
            committed = self._committed
            self.state = MarkerComparisonState(
                committed.roi_id,
                committed.marker,
                committed.fill,
                context.generation,
                "failed",
            )
            self._patch(
                self._panel_values(
                    self.state, f"Failed: {type(error).__name__}: {error}"
                )
            )

        if context is not None:
            context.commit(commit_failure)

    def install_actions(self) -> "MarkerComparisonController":
        """Register standard selects and previous/next actions on the extension."""

        common = {
            "execution": "serial-worker",
            "queue_key": self.queue_key,
            "contribution": self.contribution,
            "status_component_id": self.components.status,
            "progress_component_id": self.components.progress,
            "on_error": self._on_error,
        }
        specs = (
            ("roi-selected", self._select_roi, "latest", 1.0),
            ("previous-roi", self._move_roi, "accumulate", -1.0),
            ("next-roi", self._move_roi, "accumulate", 1.0),
            ("marker-selected", self._select_marker, "latest", 1.0),
            ("previous-marker", self._move_marker, "accumulate", -1.0),
            ("next-marker", self._move_marker, "accumulate", 1.0),
            ("fill-selected", self._select_fill, "latest", 1.0),
            ("previous-fill", self._move_fill, "accumulate", -1.0),
            ("next-fill", self._move_fill, "accumulate", 1.0),
        )
        for action, callback, coalesce, delta in specs:
            self._registrations.append(
                self.extension.on_action(
                    action,
                    callback,
                    coalesce=coalesce,
                    delta=delta,
                    **common,
                )
            )
        return self

    def close(self) -> None:
        for registration in tuple(self._registrations):
            registration.remove()
        self._registrations.clear()

    def apply_initial(self) -> None:
        """Apply initial state outside callback scope during extension setup."""

        class InitialContext:
            generation = 0
            is_current = True

            def ensure_current(self) -> None:
                return None

            def status(self, message: str) -> None:
                self_message[0] = message

            def attach(self, _task: Any) -> None:
                return None

            def report_task(self, _snapshot: Any) -> None:
                return None

            def result(self, _message: str) -> None:
                return None

            def commit(self, callback: Callable[[], Any]) -> bool:
                callback()
                return True

        self_message = [""]
        self._transition(self._committed, InitialContext())

    def _select_roi(self, context: Any, interaction: Any) -> None:
        roi = str(interaction.value)
        marker = self._committed.marker
        if marker not in self.markers.get(roi, ()):
            marker = self.markers.get(roi, ("",))[0]
        self._transition(
            MarkerComparisonState(roi, marker, self._committed.fill), context
        )

    def _move_roi(self, context: Any, _interaction: Any) -> None:
        current = self.rois.index(self._committed.roi_id)
        roi = self.rois[(current + int(context.delta)) % len(self.rois)]
        marker = self._committed.marker
        if marker not in self.markers[roi]:
            marker = self.markers[roi][0]
        self._transition(
            MarkerComparisonState(roi, marker, self._committed.fill), context
        )

    def _select_marker(self, context: Any, interaction: Any) -> None:
        self._transition(
            MarkerComparisonState(
                self._committed.roi_id,
                str(interaction.value),
                self._committed.fill,
            ),
            context,
        )

    def _move_marker(self, context: Any, _interaction: Any) -> None:
        values = self.markers[self._committed.roi_id]
        current = values.index(self._committed.marker)
        marker = values[(current + int(context.delta)) % len(values)]
        self._transition(
            MarkerComparisonState(
                self._committed.roi_id, marker, self._committed.fill
            ),
            context,
        )

    def _select_fill(self, context: Any, interaction: Any) -> None:
        requested = str(interaction.value)
        fill = next(
            (item for item in self.fills if requested in {item, self.fill_label(item)}),
            requested,
        )
        self._transition(
            MarkerComparisonState(
                self._committed.roi_id, self._committed.marker, fill
            ),
            context,
        )

    def _move_fill(self, context: Any, _interaction: Any) -> None:
        current = self.fills.index(self._committed.fill)
        fill = self.fills[(current + int(context.delta)) % len(self.fills)]
        self._transition(
            MarkerComparisonState(
                self._committed.roi_id, self._committed.marker, fill
            ),
            context,
        )

    def _transition(self, requested: MarkerComparisonState, context: Any) -> None:
        self._validate_state(requested)
        target = MarkerComparisonState(
            requested.roi_id,
            requested.marker,
            requested.fill,
            context.generation,
            "loading",
        )
        self.state = target
        context.status(f"Loading {target.marker} · {self.fill_label(target.fill)}…")
        if target.roi_id != self._committed.roi_id and self.open_roi_callback:
            operation = self.open_roi_callback(target.roi_id)
            wait = getattr(operation, "wait", None)
            if callable(wait):
                context.attach(operation)
                wait(timeout=self.timeout, progress=context.report_task)
            wait_for_viewer_readiness(self.app, timeout=self.timeout)
            context.ensure_current()
            self._loaded_source = None

        channel = self.channel_for(target.roi_id, target.marker)

        def install_channel() -> None:
            self.viewer.set_visible_channels([channel], mode="only")
            self.viewer.set_active_channel(channel)

        if not context.commit(install_channel):
            context.ensure_current()

        source = self.source_for(target.roi_id, target.marker)
        property_name = self.property_for(
            target.roi_id, target.marker, target.fill
        )
        domain = self.domain_for(target.roi_id, target.marker, target.fill)
        palette = self.palette_for(target.roi_id, target.marker, target.fill)
        if source != self._loaded_source:
            self._loaded_source = None
            replace_object_source_and_style(
                self.app,
                source,
                property_name,
                palette=palette,
                domain=domain,
                fill_opacity=self.fill_opacity,
                timeout=self.timeout,
                context=context,
                presentation_objects=self.viewer.objects,
                publish_ready=False,
            )
            self._loaded_source = source
        else:
            require_numeric_object_property(self.app.objects, property_name)
            context.ensure_current()

            def install_style() -> None:
                self.viewer.objects.color_by_continuous(
                    property_name,
                    palette=palette,
                    domain=domain,
                    fill_cells=True,
                    fill_opacity=self.fill_opacity,
                )
                self.viewer.objects.set_style(visible=True)

            if not context.commit(install_style):
                context.ensure_current()
            wait_for_viewer_readiness(self.app, timeout=self.timeout)
            context.ensure_current()

        ready_state = MarkerComparisonState(
            target.roi_id,
            target.marker,
            target.fill,
            target.generation,
            "ready",
        )
        status = (
            f"Ready · {ready_state.roi_id} · {ready_state.marker} · "
            f"{self.fill_label(ready_state.fill)}"
        )
        def commit() -> None:
            self._patch(self._panel_values(ready_state, status))
            self._committed = ready_state
            self.state = ready_state

        if not context.commit(commit):
            context.ensure_current()
