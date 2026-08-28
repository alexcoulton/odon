"""Adapt official Python SpatialData objects to generic Odon resources.

This module deliberately never calls Odon's native ``datasets.open_spatialdata``
compatibility method. Raster pixels remain in their existing backing store and
are read by Odon's Rust OME-Zarr loader.
"""

from __future__ import annotations

import asyncio
import math
import re
import secrets
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

from ..async_layers import AsyncLayer
from ..data import AffineTransform, CoordinateSpace, DataResource
from ..layers import Layer, LayerSpec
from ..providers import ArrayProvider

if TYPE_CHECKING:
    from ..async_client import AsyncClient
    from ..async_data import AsyncDataResource
    from ..async_tasks import AsyncTask
    from ..client import Client
    from ..tasks import Task


_SUPPORTED_SPATIALDATA = ((0, 6), (0, 9))


class SpatialDataAdapterError(ValueError):
    """A structured SpatialData adaptation failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        element: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.element = element
        self.details = dict(details or {})


@dataclass(frozen=True)
class TransferDecision:
    requested: str
    chosen: str
    reason: str


@dataclass
class ElementDisplay:
    name: str
    kind: str
    group_path: str | None
    transfer: TransferDecision
    resource: DataResource | AsyncDataResource
    source: str | Path
    format: str
    source_lifetime: str
    coordinate_space: CoordinateSpace
    ownership: str
    layer: Layer | AsyncLayer | None = None


@dataclass(frozen=True)
class TableProjection:
    table_name: str
    target_element: str
    columns: tuple[str, ...]
    transfer: TransferDecision
    resource_id: str


@dataclass(frozen=True)
class ElementSelection:
    """A bounded Odon selection snapshot mapped back to SpatialData element IDs."""

    element: str
    ids: tuple[str, ...]
    selection_count: int
    truncated: bool
    generation: int
    primary_id: str | None = None


@dataclass
class SpatialDataDisplay:
    """Retained handles and identities for a synchronous SpatialData display."""

    task: Task
    source_uri: str
    coordinate_system: str
    elements: dict[str, ElementDisplay]
    warnings: list[dict[str, Any]] = field(default_factory=list)
    unsupported: list[dict[str, Any]] = field(default_factory=list)
    table_projections: dict[str, TableProjection] = field(default_factory=dict)
    document: Mapping[str, Any] | None = None
    _app: Client = field(repr=False, default=None)  # type: ignore[assignment]
    _temporary_directories: tuple[Path, ...] = field(repr=False, default=())
    _providers: tuple[ArrayProvider, ...] = field(repr=False, default=())
    _pending_created_sources: tuple[Path, ...] = field(repr=False, default=())
    _selection: Mapping[str, Any] = field(repr=False, default_factory=dict)
    _closed: bool = field(repr=False, default=False)

    def wait_ready(self, timeout: float | None = None) -> Mapping[str, Any]:
        try:
            result = self.task.wait(timeout=timeout)
        except BaseException:
            for element in reversed(tuple(self.elements.values())):
                try:
                    element.resource.remove()
                except Exception:
                    pass
            try:
                self._remove_temporary_directories()
            finally:
                _remove_created_sources(self._pending_created_sources)
            self._pending_created_sources = ()
            raise
        self.document = dict(result)
        self._pending_created_sources = ()
        _attach_sync_layer_handles(self._app, self.elements, result.get("layers", ()))
        return self.document

    @property
    def readiness(self) -> dict[str, bool]:
        document = self.document or {}
        return {
            "source": True,
            "resources": bool(document.get("resources_ready", False)),
            "model": bool(document.get("model_ready", False)),
            "presentation": bool(document.get("presentation_ready", False)),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for element in reversed(tuple(self.elements.values())):
            if element.layer is not None and element.ownership == "session":
                element.layer.remove()
        for element in reversed(tuple(self.elements.values())):
            if element.ownership == "session":
                element.resource.remove()
        self._remove_temporary_directories()

    def selection(self, element: str, *, limit: int = 200) -> ElementSelection:
        """Return selected object IDs for one displayed shapes or points element."""

        displayed = _selectable_element(self.elements, element)
        response = self._app.objects.get_selection(
            **_object_selector(displayed), limit=limit
        )
        return _element_selection(element, response)

    def selected_ids(self, element: str, *, limit: int = 200) -> tuple[str, ...]:
        """Convenience form of :meth:`selection` for caller-managed subsetting."""

        return self.selection(element, limit=limit).ids

    def persist(
        self,
        destination: str | Path,
        *,
        overwrite: bool = False,
    ) -> "SpatialDataDisplay":
        """Create and open a project-owned generation with durable dependencies."""

        durable, created = _materialize_durable_sources(
            self.elements, self._providers, destination, overwrite=overwrite
        )
        resources: list[DataResource] = []
        elements: dict[str, ElementDisplay] = {}
        specs: list[LayerSpec] = []
        try:
            for name, element in self.elements.items():
                source = durable[name]
                snapshot = getattr(element.resource, "snapshot", {})
                resource = self._app.data.register(
                    source,
                    format=element.format,
                    group_path=(
                        element.group_path
                        if element.source_lifetime == "durable"
                        else None
                    ),
                    coordinate_space=element.coordinate_space,
                    ownership="project",
                    source_lifetime="durable",
                    metadata=snapshot.get("metadata", {}),
                    provenance={
                        **dict(snapshot.get("provenance", {})),
                        "promoted_from_resource_id": element.resource.resource_id,
                    },
                )
                resources.append(resource)
                promoted = ElementDisplay(
                    name=name,
                    kind=element.kind,
                    group_path=(
                        element.group_path
                        if element.source_lifetime == "durable"
                        else None
                    ),
                    transfer=TransferDecision(
                        requested=(
                            "snapshot"
                            if element.source_lifetime == "session"
                            else "reference"
                        ),
                        chosen=(
                            "snapshot"
                            if element.source_lifetime == "session"
                            else "reference"
                        ),
                        reason="explicit_durable_promotion",
                    ),
                    resource=resource,
                    source=source,
                    format=element.format,
                    source_lifetime="durable",
                    coordinate_space=element.coordinate_space,
                    ownership="project",
                )
                elements[name] = promoted
                if element.kind != "image":
                    specs.append(
                        LayerSpec.from_resource(
                            resource,
                            name=name,
                            kind=element.kind,
                            layer_id=(
                                _object_layer_id()
                                if element.kind in {"shapes", "points", "objects"}
                                else None
                            ),
                            ownership="project",
                            provenance={"promoted_from": element.resource.resource_id},
                        )
                    )
            primary = elements[next(iter(self.elements))].resource
            task = self._app.datasets.open_resource(primary, layers=specs)
        except BaseException:
            for resource in reversed(resources):
                resource.remove()
            _remove_created_sources(created)
            raise
        projections = {
            name: TableProjection(
                table_name=projection.table_name,
                target_element=projection.target_element,
                columns=projection.columns,
                transfer=TransferDecision(
                    requested="snapshot", chosen="snapshot", reason="durable_vector_snapshot"
                ),
                resource_id=elements[projection.target_element].resource.resource_id,
            )
            for name, projection in self.table_projections.items()
        }
        return SpatialDataDisplay(
            task=task,
            source_uri=Path(destination).expanduser().resolve().as_uri(),
            coordinate_system=self.coordinate_system,
            elements=elements,
            warnings=list(self.warnings),
            unsupported=list(self.unsupported),
            table_projections=projections,
            _app=self._app,
            _pending_created_sources=created,
            _selection=dict(self._selection),
        )

    def refresh(
        self,
        sdata: Any,
        *,
        element: str | None = None,
        transfer: str | None = None,
        timeout: float | None = None,
    ) -> "SpatialDataDisplay":
        """Open a fresh document generation, retaining the old one until ready."""

        if (
            element is not None
            and element not in self.elements
            and element not in self.table_projections
        ):
            raise SpatialDataAdapterError(
                "element_not_displayed",
                "the requested element is not part of this display",
                element=element,
            )
        policies = {
            name: displayed.transfer.requested for name, displayed in self.elements.items()
        }
        policies.update(
            {
                name: projection.transfer.requested
                for name, projection in self.table_projections.items()
            }
        )
        if transfer is not None:
            _validate_transfer(transfer)
            if element is None:
                policies = {"*": transfer}
            else:
                policies[element] = transfer
        refreshed = display(
            self._app,
            sdata,
            **dict(self._selection),
            transfer=policies,
            ownership=next(iter(self.elements.values())).ownership,
        )
        refreshed.wait_ready(timeout=timeout)
        self.close()
        return refreshed

    def _remove_temporary_directories(self) -> None:
        for directory in self._temporary_directories:
            shutil.rmtree(directory, ignore_errors=True)
        for provider in self._providers:
            provider.close()


@dataclass
class AsyncSpatialDataDisplay:
    """Retained handles and identities for an asynchronous SpatialData display."""

    task: AsyncTask
    source_uri: str
    coordinate_system: str
    elements: dict[str, ElementDisplay]
    warnings: list[dict[str, Any]] = field(default_factory=list)
    unsupported: list[dict[str, Any]] = field(default_factory=list)
    table_projections: dict[str, TableProjection] = field(default_factory=dict)
    document: Mapping[str, Any] | None = None
    _app: AsyncClient = field(repr=False, default=None)  # type: ignore[assignment]
    _temporary_directories: tuple[Path, ...] = field(repr=False, default=())
    _providers: tuple[ArrayProvider, ...] = field(repr=False, default=())
    _pending_created_sources: tuple[Path, ...] = field(repr=False, default=())
    _selection: Mapping[str, Any] = field(repr=False, default_factory=dict)
    _closed: bool = field(repr=False, default=False)

    async def wait_ready(self, timeout: float | None = None) -> Mapping[str, Any]:
        try:
            result = await self.task.wait(timeout=timeout)
        except BaseException:
            await asyncio.gather(
                *(
                    element.resource.remove()
                    for element in reversed(tuple(self.elements.values()))
                ),
                return_exceptions=True,
            )
            try:
                await self._remove_temporary_directories()
            finally:
                await asyncio.to_thread(
                    _remove_created_sources, self._pending_created_sources
                )
            self._pending_created_sources = ()
            raise
        self.document = dict(result)
        self._pending_created_sources = ()
        _attach_async_layer_handles(self._app, self.elements, result.get("layers", ()))
        return self.document

    @property
    def readiness(self) -> dict[str, bool]:
        document = self.document or {}
        return {
            "source": True,
            "resources": bool(document.get("resources_ready", False)),
            "model": bool(document.get("model_ready", False)),
            "presentation": bool(document.get("presentation_ready", False)),
        }

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for element in reversed(tuple(self.elements.values())):
            if element.layer is not None and element.ownership == "session":
                await element.layer.remove()
        for element in reversed(tuple(self.elements.values())):
            if element.ownership == "session":
                await element.resource.remove()
        await self._remove_temporary_directories()

    async def selection(self, element: str, *, limit: int = 200) -> ElementSelection:
        """Return selected object IDs for one displayed shapes or points element."""

        displayed = _selectable_element(self.elements, element)
        response = await self._app.objects.get_selection(
            **_object_selector(displayed), limit=limit
        )
        return _element_selection(element, response)

    async def selected_ids(self, element: str, *, limit: int = 200) -> tuple[str, ...]:
        return (await self.selection(element, limit=limit)).ids

    async def persist(
        self,
        destination: str | Path,
        *,
        overwrite: bool = False,
    ) -> "AsyncSpatialDataDisplay":
        durable, created = await asyncio.to_thread(
            _materialize_durable_sources,
            self.elements,
            self._providers,
            destination,
            overwrite=overwrite,
        )
        resources: list[AsyncDataResource] = []
        elements: dict[str, ElementDisplay] = {}
        specs: list[LayerSpec] = []
        try:
            for name, element in self.elements.items():
                source = durable[name]
                snapshot = getattr(element.resource, "snapshot", {})
                resource = await self._app.data.register(
                    source,
                    format=element.format,
                    group_path=(
                        element.group_path
                        if element.source_lifetime == "durable"
                        else None
                    ),
                    coordinate_space=element.coordinate_space,
                    ownership="project",
                    source_lifetime="durable",
                    metadata=snapshot.get("metadata", {}),
                    provenance={
                        **dict(snapshot.get("provenance", {})),
                        "promoted_from_resource_id": element.resource.resource_id,
                    },
                )
                resources.append(resource)
                elements[name] = ElementDisplay(
                    name=name,
                    kind=element.kind,
                    group_path=(
                        element.group_path
                        if element.source_lifetime == "durable"
                        else None
                    ),
                    transfer=TransferDecision(
                        requested=(
                            "snapshot"
                            if element.source_lifetime == "session"
                            else "reference"
                        ),
                        chosen=(
                            "snapshot"
                            if element.source_lifetime == "session"
                            else "reference"
                        ),
                        reason="explicit_durable_promotion",
                    ),
                    resource=resource,
                    source=source,
                    format=element.format,
                    source_lifetime="durable",
                    coordinate_space=element.coordinate_space,
                    ownership="project",
                )
                if element.kind != "image":
                    specs.append(
                        LayerSpec(
                            data_resource_id=resource.resource_id,
                            name=name,
                            kind=element.kind,
                            layer_id=(
                                _object_layer_id()
                                if element.kind in {"shapes", "points", "objects"}
                                else None
                            ),
                            ownership="project",
                            provenance={"promoted_from": element.resource.resource_id},
                        )
                    )
            primary = elements[next(iter(self.elements))].resource
            task = await self._app.datasets.open_resource(primary, layers=specs)
        except BaseException:
            for resource in reversed(resources):
                await resource.remove()
            await asyncio.to_thread(_remove_created_sources, created)
            raise
        projections = {
            name: TableProjection(
                table_name=projection.table_name,
                target_element=projection.target_element,
                columns=projection.columns,
                transfer=TransferDecision(
                    requested="snapshot", chosen="snapshot", reason="durable_vector_snapshot"
                ),
                resource_id=elements[projection.target_element].resource.resource_id,
            )
            for name, projection in self.table_projections.items()
        }
        return AsyncSpatialDataDisplay(
            task=task,
            source_uri=Path(destination).expanduser().resolve().as_uri(),
            coordinate_system=self.coordinate_system,
            elements=elements,
            warnings=list(self.warnings),
            unsupported=list(self.unsupported),
            table_projections=projections,
            _app=self._app,
            _pending_created_sources=created,
            _selection=dict(self._selection),
        )

    async def refresh(
        self,
        sdata: Any,
        *,
        element: str | None = None,
        transfer: str | None = None,
        timeout: float | None = None,
    ) -> "AsyncSpatialDataDisplay":
        if (
            element is not None
            and element not in self.elements
            and element not in self.table_projections
        ):
            raise SpatialDataAdapterError(
                "element_not_displayed",
                "the requested element is not part of this display",
                element=element,
            )
        policies = {
            name: displayed.transfer.requested for name, displayed in self.elements.items()
        }
        policies.update(
            {
                name: projection.transfer.requested
                for name, projection in self.table_projections.items()
            }
        )
        if transfer is not None:
            _validate_transfer(transfer)
            if element is None:
                policies = {"*": transfer}
            else:
                policies[element] = transfer
        refreshed = await display_async(
            self._app,
            sdata,
            **dict(self._selection),
            transfer=policies,
            ownership=next(iter(self.elements.values())).ownership,
        )
        await refreshed.wait_ready(timeout=timeout)
        await self.close()
        return refreshed

    async def _remove_temporary_directories(self) -> None:
        await asyncio.gather(
            *(
                asyncio.to_thread(shutil.rmtree, directory, ignore_errors=True)
                for directory in self._temporary_directories
            ),
            return_exceptions=True,
        )
        await asyncio.gather(
            *(asyncio.to_thread(provider.close) for provider in self._providers),
            return_exceptions=True,
        )


@dataclass(frozen=True)
class _SpatialDataApi:
    spatialdata_type: type[Any]
    version: str
    get_axes_names: Callable[[Any], Sequence[str]]
    get_transformation: Callable[..., Any]
    get_table_keys: Callable[[Any], tuple[Any, str, str]]
    get_dask_backing_files: Callable[[Any], Sequence[str]]
    read_zarr: Callable[..., Any]
    tokenize: Callable[[Any], str]


@dataclass(frozen=True)
class _PreparedElement:
    name: str
    kind: str
    source: str | Path
    format: str
    source_lifetime: str
    group_path: str | None
    coordinate_space: CoordinateSpace
    metadata: dict[str, Any]
    provenance: dict[str, Any]
    transfer: TransferDecision


@dataclass(frozen=True)
class _PreparedTableProjection:
    table_name: str
    target_element: str
    columns: tuple[str, ...]
    transfer: TransferDecision


@dataclass(frozen=True)
class _TableJoin:
    table_name: str
    target_element: str
    region_key: str
    instance_key: str
    columns: tuple[str, ...]
    table: Any
    requested_transfer: str


@dataclass(frozen=True)
class _PreparedDisplay:
    root: str | Path | None
    source_uri: str
    coordinate_system: str
    elements: tuple[_PreparedElement, ...]
    table_projections: tuple[_PreparedTableProjection, ...]
    temporary_directories: tuple[Path, ...]
    providers: tuple[ArrayProvider, ...]
    warnings: tuple[dict[str, Any], ...]


def _materialize_durable_sources(
    elements: Mapping[str, ElementDisplay],
    providers: Sequence[ArrayProvider],
    destination: str | Path,
    *,
    overwrite: bool,
) -> tuple[dict[str, str | Path], tuple[Path, ...]]:
    root = Path(destination).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    providers_by_uri = {provider.uri: provider for provider in providers}
    durable: dict[str, str | Path] = {}
    created: list[Path] = []
    try:
        for name, element in elements.items():
            if element.source_lifetime == "durable":
                durable[name] = element.source
                continue
            stem = _safe_file_stem(name)
            provider = providers_by_uri.get(str(element.source))
            if provider is not None:
                path = root / f"{stem}.ome.zarr"
                existed = path.exists()
                provider.materialize(path, overwrite=overwrite)
            elif element.format in {"parquet", "geoparquet"}:
                path = root / f"{stem}.parquet"
                existed = path.exists()
                _copy_file_snapshot(Path(element.source), path, overwrite=overwrite)
            else:
                raise SpatialDataAdapterError(
                    "persistence_unavailable",
                    "this ephemeral resource cannot be promoted to a durable format",
                    element=name,
                    details={"format": element.format},
                )
            durable[name] = path
            if not existed:
                created.append(path)
    except BaseException:
        _remove_created_sources(created)
        raise
    return durable, tuple(created)


def _copy_file_snapshot(source: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.odon-staging-", dir=destination.parent
        )
    )
    staging = staging_root / destination.name
    try:
        shutil.copy2(source, staging)
        staging.replace(destination)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def _remove_created_sources(paths: Sequence[Path]) -> None:
    for path in reversed(tuple(paths)):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def display(
    app: Client,
    sdata: Any,
    *,
    image: str | None = None,
    labels: Iterable[str] = (),
    shapes: Iterable[str] = (),
    points: Iterable[str] = (),
    tables: Iterable[str] = (),
    table_columns: Mapping[str, Iterable[str]] | None = None,
    coordinate_system: str | None = None,
    transfer: str | Mapping[str, str] = "auto",
    ownership: str = "session",
) -> SpatialDataDisplay:
    """Compose selected SpatialData elements through generic Odon resources."""

    prepared = _prepare_display(
        sdata,
        image=image,
        labels=labels,
        shapes=shapes,
        points=points,
        tables=tables,
        table_columns=table_columns,
        coordinate_system=coordinate_system,
        transfer=transfer,
    )
    if (
        any(item.source_lifetime == "session" for item in prepared.elements)
        and ownership != "session"
    ):
        for directory in prepared.temporary_directories:
            shutil.rmtree(directory, ignore_errors=True)
        for provider in prepared.providers:
            provider.close()
        raise SpatialDataAdapterError(
            "ephemeral_ownership_invalid",
            "provider-backed and temporary adapted resources must remain session-owned",
        )
    resources: list[DataResource] = []
    elements: dict[str, ElementDisplay] = {}
    layer_specs: list[LayerSpec] = []
    try:
        for item in prepared.elements:
            resource = app.data.register(
                item.source,
                format=item.format,
                group_path=item.group_path,
                coordinate_space=item.coordinate_space,
                ownership=ownership,
                source_lifetime=item.source_lifetime,
                metadata=item.metadata,
                provenance=item.provenance,
            )
            resources.append(resource)
            if item.kind != "image":
                layer_specs.append(
                    LayerSpec(
                        data_resource_id=resource.resource_id,
                        name=item.name,
                        kind=item.kind,
                        layer_id=(
                            _object_layer_id()
                            if item.kind in {"shapes", "points", "objects"}
                            else None
                        ),
                        ownership=ownership,
                        provenance=item.provenance,
                    )
                )
            elements[item.name] = ElementDisplay(
                name=item.name,
                kind=item.kind,
                group_path=item.group_path,
                transfer=item.transfer,
                resource=resource,
                source=item.source,
                format=item.format,
                source_lifetime=item.source_lifetime,
                coordinate_space=item.coordinate_space,
                ownership=ownership,
            )
        primary = elements[prepared.elements[0].name].resource
        task = app.datasets.open_resource(primary, layers=layer_specs)
    except BaseException:
        for resource in reversed(resources):
            resource.remove()
        for directory in prepared.temporary_directories:
            shutil.rmtree(directory, ignore_errors=True)
        for provider in prepared.providers:
            provider.close()
        raise
    table_projections = {
        item.table_name: TableProjection(
            table_name=item.table_name,
            target_element=item.target_element,
            columns=item.columns,
            transfer=item.transfer,
            resource_id=elements[item.target_element].resource.resource_id,
        )
        for item in prepared.table_projections
    }
    return SpatialDataDisplay(
        task=task,
        source_uri=prepared.source_uri,
        coordinate_system=prepared.coordinate_system,
        elements=elements,
        warnings=list(prepared.warnings),
        table_projections=table_projections,
        _app=app,
        _temporary_directories=prepared.temporary_directories,
        _providers=prepared.providers,
        _selection={
            "image": image,
            "labels": tuple(_names(labels)),
            "shapes": tuple(_names(shapes)),
            "points": tuple(_names(points)),
            "tables": tuple(_names(tables)),
            "table_columns": {
                name: tuple(_names(columns))
                for name, columns in (table_columns or {}).items()
            },
            "coordinate_system": coordinate_system,
        },
    )


async def display_async(
    app: AsyncClient,
    sdata: Any,
    *,
    image: str | None = None,
    labels: Iterable[str] = (),
    shapes: Iterable[str] = (),
    points: Iterable[str] = (),
    tables: Iterable[str] = (),
    table_columns: Mapping[str, Iterable[str]] | None = None,
    coordinate_system: str | None = None,
    transfer: str | Mapping[str, str] = "auto",
    ownership: str = "session",
) -> AsyncSpatialDataDisplay:
    """Asynchronous counterpart of :func:`display`."""

    prepared = _prepare_display(
        sdata,
        image=image,
        labels=labels,
        shapes=shapes,
        points=points,
        tables=tables,
        table_columns=table_columns,
        coordinate_system=coordinate_system,
        transfer=transfer,
    )
    if (
        any(item.source_lifetime == "session" for item in prepared.elements)
        and ownership != "session"
    ):
        await asyncio.gather(
            *(
                asyncio.to_thread(shutil.rmtree, directory, ignore_errors=True)
                for directory in prepared.temporary_directories
            )
        )
        await asyncio.gather(
            *(asyncio.to_thread(provider.close) for provider in prepared.providers)
        )
        raise SpatialDataAdapterError(
            "ephemeral_ownership_invalid",
            "provider-backed and temporary adapted resources must remain session-owned",
        )
    resources: list[AsyncDataResource] = []
    elements: dict[str, ElementDisplay] = {}
    layer_specs: list[LayerSpec] = []
    try:
        for item in prepared.elements:
            resource = await app.data.register(
                item.source,
                format=item.format,
                group_path=item.group_path,
                coordinate_space=item.coordinate_space,
                ownership=ownership,
                source_lifetime=item.source_lifetime,
                metadata=item.metadata,
                provenance=item.provenance,
            )
            resources.append(resource)
            if item.kind != "image":
                layer_specs.append(
                    LayerSpec(
                        data_resource_id=resource.resource_id,
                        name=item.name,
                        kind=item.kind,
                        layer_id=(
                            _object_layer_id()
                            if item.kind in {"shapes", "points", "objects"}
                            else None
                        ),
                        ownership=ownership,
                        provenance=item.provenance,
                    )
                )
            elements[item.name] = ElementDisplay(
                name=item.name,
                kind=item.kind,
                group_path=item.group_path,
                transfer=item.transfer,
                resource=resource,
                source=item.source,
                format=item.format,
                source_lifetime=item.source_lifetime,
                coordinate_space=item.coordinate_space,
                ownership=ownership,
            )
        primary = elements[prepared.elements[0].name].resource
        task = await app.datasets.open_resource(primary, layers=layer_specs)
    except BaseException:
        for resource in reversed(resources):
            await resource.remove()
        await asyncio.gather(
            *(
                asyncio.to_thread(shutil.rmtree, directory, ignore_errors=True)
                for directory in prepared.temporary_directories
            )
        )
        await asyncio.gather(
            *(asyncio.to_thread(provider.close) for provider in prepared.providers)
        )
        raise
    table_projections = {
        item.table_name: TableProjection(
            table_name=item.table_name,
            target_element=item.target_element,
            columns=item.columns,
            transfer=item.transfer,
            resource_id=elements[item.target_element].resource.resource_id,
        )
        for item in prepared.table_projections
    }
    return AsyncSpatialDataDisplay(
        task=task,
        source_uri=prepared.source_uri,
        coordinate_system=prepared.coordinate_system,
        elements=elements,
        warnings=list(prepared.warnings),
        table_projections=table_projections,
        _app=app,
        _temporary_directories=prepared.temporary_directories,
        _providers=prepared.providers,
        _selection={
            "image": image,
            "labels": tuple(_names(labels)),
            "shapes": tuple(_names(shapes)),
            "points": tuple(_names(points)),
            "tables": tuple(_names(tables)),
            "table_columns": {
                name: tuple(_names(columns))
                for name, columns in (table_columns or {}).items()
            },
            "coordinate_system": coordinate_system,
        },
    )


def _prepare_display(
    sdata: Any,
    *,
    image: str | None,
    labels: Iterable[str],
    shapes: Iterable[str],
    points: Iterable[str],
    tables: Iterable[str],
    table_columns: Mapping[str, Iterable[str]] | None,
    coordinate_system: str | None,
    transfer: str | Mapping[str, str],
) -> _PreparedDisplay:
    api = _load_spatialdata_api()
    if not isinstance(sdata, api.spatialdata_type):
        raise SpatialDataAdapterError(
            "invalid_spatialdata",
            "sdata must be an instance of spatialdata.SpatialData",
        )
    _validate_transfer(transfer)
    image_name = _select_image(sdata, image)
    label_names = _names(labels)
    shape_names = _names(shapes)
    point_names = _names(points)
    table_names = _names(tables)
    selected_names = [
        image_name,
        *label_names,
        *shape_names,
        *point_names,
        *table_names,
    ]
    duplicate_names = sorted(
        {name for name in selected_names if selected_names.count(name) > 1}
    )
    if duplicate_names:
        raise SpatialDataAdapterError(
            "ambiguous_element_name",
            "selected element names must be unique across SpatialData families",
            details={"names": duplicate_names},
        )
    selected: list[tuple[str, str, Any]] = [
        ("image", image_name, sdata.images[image_name])
    ]
    for name in label_names:
        _require_element(sdata.labels, "labels", name)
        selected.append(("labels", name, sdata.labels[name]))
    for name in shape_names:
        _require_element(sdata.shapes, "shapes", name)
        selected.append(("shapes", name, sdata.shapes[name]))
    for name in point_names:
        _require_element(sdata.points, "points", name)
        selected.append(("points", name, sdata.points[name]))
    for name in table_names:
        _require_element(sdata.tables, "tables", name)
    root = _backing_source(sdata) if bool(sdata.is_backed()) else None
    if (
        isinstance(root, str)
        and urlsplit(root).scheme.lower() == "s3"
        and any(kind == "labels" for kind, _, _ in selected)
    ):
        raise SpatialDataAdapterError(
            "s3_secondary_raster_unavailable",
            "S3 credentials are actor-session state and cannot yet be projected "
            "into secondary raster layers",
            details={"available": ["primary_image"], "alternative": "durable_http_url"},
        )
    target = _select_coordinate_system(api, selected, coordinate_system)
    disk_paths = set(sdata.elements_paths_on_disk()) if root is not None else set()
    fresh = (
        api.read_zarr(
            root,
            selection=tuple(
                sorted(
                    {
                        "images" if kind == "image" else kind
                        for kind, _, _ in selected
                    }
                )
            ),
        )
        if root is not None
        else None
    )
    joins = _prepare_table_joins(
        api,
        sdata,
        table_names,
        table_columns or {},
        shape_names,
        transfer,
    )
    joins_by_target: dict[str, list[_TableJoin]] = {}
    for join in joins:
        joins_by_target.setdefault(join.target_element, []).append(join)
    absolute_affines = {
        name: _element_affine(api, element, target, name)
        for _, name, element in selected
    }
    document_inverse = _invert_affine(absolute_affines[image_name], image_name)
    prepared: list[_PreparedElement] = []
    temporary_directories: list[Path] = []
    providers: list[ArrayProvider] = []
    try:
        for kind, name, element in selected:
            requested = _transfer_for(transfer, name)
            affine = _multiply_affine(document_inverse, absolute_affines[name])
            coordinate_space = _coordinate_space(api, element, affine, name)
            metadata: dict[str, Any]
            source: Path
            format: str
            source_lifetime: str
            group_path: str | None
            if kind in {"image", "labels"}:
                if requested == "snapshot":
                    raise SpatialDataAdapterError(
                        "raster_snapshot_unavailable",
                        "the adapter never materializes a raster implicitly; use a "
                        "provider or create a durable snapshot explicitly",
                        element=name,
                    )
                metadata = _raster_metadata(element)
                reference_error: SpatialDataAdapterError | None = None
                if root is not None and requested in {"auto", "reference"}:
                    try:
                        group_path = _prove_direct_reference(
                            api,
                            sdata,
                            fresh,
                            root,
                            disk_paths,
                            kind,
                            name,
                            element,
                        )
                    except SpatialDataAdapterError as error:
                        reference_error = error
                    else:
                        source = root
                        format = "ome-zarr"
                        decision = TransferDecision(
                            requested=requested,
                            chosen="reference",
                            reason="backed_self_contained_persisted_raster",
                        )
                        source_lifetime = (
                            "session"
                            if isinstance(root, str)
                            and urlsplit(root).scheme.lower() == "s3"
                            else "durable"
                        )
                if root is None or requested == "provider" or reference_error is not None:
                    if requested == "reference":
                        if reference_error is not None:
                            raise reference_error
                        raise SpatialDataAdapterError(
                            "unbacked_raster",
                            "reference transfer requires a backed SpatialData raster",
                            element=name,
                        )
                    axes = tuple(str(axis) for axis in api.get_axes_names(element))
                    payloads = _raster_payloads(element, name)
                    provider = ArrayProvider(payloads[0], axes=axes)
                    providers.append(provider)
                    source = provider.uri
                    format = "ome-zarr"
                    group_path = None
                    metadata.update(
                        provider_generation=provider.snapshot.generation,
                        provider_multiscales=1,
                    )
                    decision = TransferDecision(
                        requested=requested,
                        chosen="provider",
                        reason=(
                            "unbacked_or_computed_raster"
                            if reference_error is None
                            else f"reference_ineligible:{reference_error.code}"
                        ),
                    )
                    source_lifetime = "session"
            else:
                source, format, group_path, decision, temporary = _prepare_vector(
                    api=api,
                    sdata=sdata,
                    fresh=fresh,
                    root=root,
                    disk_paths=disk_paths,
                    kind=kind,
                    name=name,
                    element=element,
                    requested=requested,
                    joins=joins_by_target.get(name, ()),
                )
                if temporary is not None:
                    temporary_directories.append(temporary)
                source_lifetime = "session" if temporary is not None else "durable"
                metadata = {
                    "spatialdata_table_projections": [
                        {
                            "table": join.table_name,
                            "columns": list(join.columns),
                            "target_key": "instance_id",
                        }
                        for join in joins_by_target.get(name, ())
                    ]
                }
            metadata.update(
                spatialdata_element=name,
                spatialdata_kind=kind,
                spatialdata_coordinate_system=target,
                spatialdata_affine_xy=absolute_affines[name],
                odon_document_affine_xy=affine,
            )
            provenance = {
                "adapter": "odon.integrations.spatialdata",
                "adapter_version": "0.2",
                "spatialdata_version": api.version,
                "spatialdata_element": name,
                "spatialdata_kind": kind,
                "coordinate_system": target,
                "transfer": decision.chosen,
                "source_uri": _source_uri(source),
            }
            if group_path is not None:
                provenance["group_path"] = group_path
            prepared.append(
                _PreparedElement(
                    name=name,
                    kind=kind,
                    source=source,
                    format=format,
                    source_lifetime=source_lifetime,
                    group_path=group_path,
                    coordinate_space=coordinate_space,
                    metadata=metadata,
                    provenance=provenance,
                    transfer=decision,
                )
            )
    except BaseException:
        for directory in temporary_directories:
            shutil.rmtree(directory, ignore_errors=True)
        for provider in providers:
            provider.close()
        raise
    source_uri = _source_uri(root) if root is not None else _source_uri(prepared[0].source)
    return _PreparedDisplay(
        root=root,
        source_uri=source_uri,
        coordinate_system=target,
        elements=tuple(prepared),
        table_projections=tuple(
            _PreparedTableProjection(
                table_name=join.table_name,
                target_element=join.target_element,
                columns=join.columns,
                transfer=TransferDecision(
                    requested=join.requested_transfer,
                    chosen="snapshot",
                    reason="selected_columns_joined_into_vector_snapshot",
                ),
            )
            for join in joins
        ),
        temporary_directories=tuple(temporary_directories),
        providers=tuple(providers),
        warnings=(),
    )


def _attach_sync_layer_handles(
    app: Client,
    elements: Mapping[str, ElementDisplay],
    snapshots: Iterable[Mapping[str, Any]],
) -> None:
    by_resource = {element.resource.resource_id: element for element in elements.values()}
    for snapshot in snapshots:
        element = by_resource.get(str(snapshot.get("data_resource_id", "")))
        if element is not None:
            element.layer = Layer(app.layers, snapshot)


def _selectable_element(
    elements: Mapping[str, ElementDisplay], name: str
) -> ElementDisplay:
    try:
        element = elements[name]
    except KeyError as error:
        raise SpatialDataAdapterError(
            "element_not_displayed",
            "the requested element is not part of this display",
            element=name,
        ) from error
    if element.kind not in {"shapes", "points", "objects"} or element.layer is None:
        raise SpatialDataAdapterError(
            "element_not_selectable",
            "the requested element is not an actor-owned object layer",
            element=name,
        )
    return element


def _object_selector(element: ElementDisplay) -> dict[str, Any]:
    assert element.layer is not None
    layer_id = element.layer.layer_id
    prefix = "spatial_shape:"
    if not layer_id.startswith(prefix):
        raise SpatialDataAdapterError(
            "element_not_selectable",
            "the displayed layer does not expose a stable spatial object target",
            element=element.name,
            details={"layer_id": layer_id},
        )
    try:
        numeric_id = int(layer_id[len(prefix) :])
    except ValueError as error:
        raise SpatialDataAdapterError(
            "element_not_selectable",
            "the displayed layer has an invalid spatial object target",
            element=element.name,
            details={"layer_id": layer_id},
        ) from error
    return {"target": "spatial_shape", "layer_id": numeric_id}


def _element_selection(element: str, response: Mapping[str, Any]) -> ElementSelection:
    selection = response.get("selection", {})
    if not isinstance(selection, Mapping):
        selection = {}
    selected = selection.get("selected", ())
    ids = tuple(
        str(item["id"])
        for item in selected
        if isinstance(item, Mapping) and item.get("id") is not None
    )
    primary = selection.get("primary")
    return ElementSelection(
        element=element,
        ids=ids,
        selection_count=int(selection.get("selection_count", len(ids))),
        truncated=bool(selection.get("truncated", False)),
        generation=int(selection.get("generation", 0)),
        primary_id=(
            str(primary["id"])
            if isinstance(primary, Mapping) and primary.get("id") is not None
            else None
        ),
    )


def _object_layer_id() -> str:
    # Keep the numeric target exactly representable by JSON/JavaScript consumers.
    return f"spatial_shape:{secrets.randbelow((1 << 52) - 1) + 1}"


def _attach_async_layer_handles(
    app: AsyncClient,
    elements: Mapping[str, ElementDisplay],
    snapshots: Iterable[Mapping[str, Any]],
) -> None:
    by_resource = {element.resource.resource_id: element for element in elements.values()}
    for snapshot in snapshots:
        element = by_resource.get(str(snapshot.get("data_resource_id", "")))
        if element is not None:
            element.layer = AsyncLayer(app.layers, snapshot)


def _load_spatialdata_api() -> _SpatialDataApi:
    try:
        import spatialdata
        from dask.base import tokenize
        from spatialdata.models import get_axes_names, get_table_keys
        from spatialdata.transformations import get_transformation
    except ImportError as error:
        raise ImportError(
            "SpatialData integration requires `pip install odon-client[spatialdata]`"
        ) from error
    version = str(getattr(spatialdata, "__version__", "0.0.0"))
    match = re.match(r"^(\d+)\.(\d+)", version)
    parsed = (int(match.group(1)), int(match.group(2))) if match else (0, 0)
    minimum, maximum = _SUPPORTED_SPATIALDATA
    if not minimum <= parsed < maximum:
        raise ImportError(
            f"spatialdata {version} is unsupported; expected >=0.6,<0.9"
        )
    return _SpatialDataApi(
        spatialdata_type=spatialdata.SpatialData,
        version=version,
        get_axes_names=get_axes_names,
        get_transformation=get_transformation,
        get_table_keys=get_table_keys,
        get_dask_backing_files=spatialdata.get_dask_backing_files,
        read_zarr=spatialdata.read_zarr,
        tokenize=tokenize,
    )


def _select_image(sdata: Any, image: str | None) -> str:
    if image is not None:
        if image not in sdata.images:
            raise SpatialDataAdapterError(
                "element_not_found", f"image element {image!r} was not found", element=image
            )
        return image
    candidates = tuple(sdata.images)
    if len(candidates) != 1:
        raise SpatialDataAdapterError(
            "ambiguous_primary_image",
            "image must be specified unless exactly one image is available",
            details={"candidates": list(candidates)},
        )
    return candidates[0]


def _names(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        values = (values,)
    result = tuple(values)
    if any(not isinstance(value, str) or not value for value in result):
        raise SpatialDataAdapterError(
            "invalid_element_name", "element names must be non-empty strings"
        )
    if len(set(result)) != len(result):
        raise SpatialDataAdapterError(
            "duplicate_element", "selected element names must be unique"
        )
    return result


def _require_element(collection: Mapping[str, Any], kind: str, name: str) -> None:
    if name not in collection:
        raise SpatialDataAdapterError(
            "element_not_found", f"{kind} element {name!r} was not found", element=name
        )


def _validate_transfer(transfer: str | Mapping[str, str]) -> None:
    allowed = {"auto", "reference", "provider", "snapshot"}
    if isinstance(transfer, str):
        policies = {"*": transfer}
    elif isinstance(transfer, Mapping):
        policies = transfer
        if any(not isinstance(name, str) or not name for name in policies):
            raise SpatialDataAdapterError(
                "invalid_transfer_policy", "transfer policy keys must be element names"
            )
    else:
        raise SpatialDataAdapterError(
            "invalid_transfer_policy", "transfer must be a policy or element-policy mapping"
        )
    invalid = {name: policy for name, policy in policies.items() if policy not in allowed}
    if invalid:
        raise SpatialDataAdapterError(
            "invalid_transfer_policy",
            "transfer policies must be auto, reference, provider, or snapshot",
            details={"invalid": invalid},
        )


def _transfer_for(transfer: str | Mapping[str, str], name: str) -> str:
    if isinstance(transfer, str):
        return transfer
    return transfer.get(name, transfer.get("*", "auto"))


def _prepare_table_joins(
    api: _SpatialDataApi,
    sdata: Any,
    table_names: Sequence[str],
    table_columns: Mapping[str, Iterable[str]],
    shape_names: Sequence[str],
    transfer: str | Mapping[str, str],
) -> tuple[_TableJoin, ...]:
    unknown = set(table_columns).difference(table_names)
    if unknown:
        raise SpatialDataAdapterError(
            "unselected_table_columns",
            "table_columns contains a table that was not selected",
            details={"tables": sorted(unknown)},
        )
    joins: list[_TableJoin] = []
    for table_name in table_names:
        columns = _names(table_columns.get(table_name, ()))
        if not columns:
            raise SpatialDataAdapterError(
                "table_columns_required",
                "selected tables require an explicit, non-empty column list",
                element=table_name,
            )
        requested = _transfer_for(transfer, table_name)
        if requested in {"reference", "provider"}:
            raise SpatialDataAdapterError(
                "table_transfer_unavailable",
                "selected table columns currently require snapshot projection",
                element=table_name,
                details={"requested": requested, "available": ["auto", "snapshot"]},
            )
        table = sdata.tables[table_name]
        try:
            regions, region_key, instance_key = api.get_table_keys(table)
        except Exception as error:
            raise SpatialDataAdapterError(
                "invalid_table_annotation",
                "the selected table does not expose a valid SpatialData annotation target",
                element=table_name,
            ) from error
        if isinstance(regions, str):
            region_names = {regions}
        else:
            region_names = {str(region) for region in regions}
        targets = sorted(region_names.intersection(shape_names))
        if len(targets) != 1:
            raise SpatialDataAdapterError(
                "table_target_unavailable",
                "a selected table must annotate exactly one selected shapes element",
                element=table_name,
                details={
                    "regions": sorted(region_names),
                    "selected_shapes": list(shape_names),
                    "matching_targets": targets,
                },
            )
        obs = getattr(table, "obs", None)
        if obs is None:
            raise SpatialDataAdapterError(
                "invalid_table_annotation",
                "the selected table has no obs frame",
                element=table_name,
            )
        missing = [column for column in columns if column not in obs.columns]
        if missing:
            raise SpatialDataAdapterError(
                "table_column_not_found",
                "one or more selected table columns were not found",
                element=table_name,
                details={"columns": missing},
            )
        joins.append(
            _TableJoin(
                table_name=table_name,
                target_element=targets[0],
                region_key=str(region_key),
                instance_key=str(instance_key),
                columns=columns,
                table=table,
                requested_transfer=requested,
            )
        )
    return tuple(joins)


def _prepare_vector(
    *,
    api: _SpatialDataApi,
    sdata: Any,
    fresh: Any,
    root: str | Path | None,
    disk_paths: set[str],
    kind: str,
    name: str,
    element: Any,
    requested: str,
    joins: Sequence[_TableJoin],
) -> tuple[Path, str, None, TransferDecision, Path | None]:
    if requested == "provider":
        raise SpatialDataAdapterError(
            "provider_unavailable",
            "provider transfer is available for rasters, not vector elements",
            element=name,
            details={"available": ["auto", "reference", "snapshot"]},
        )
    referenced = None if joins else _prove_vector_reference(
        sdata, fresh, root, disk_paths, kind, name, element
    )
    if referenced is not None and requested in {"auto", "reference"}:
        return (
            referenced,
            "geoparquet",
            None,
            TransferDecision(
                requested=requested,
                chosen="reference",
                reason="backed_self_contained_equivalent_geoparquet",
            ),
            None,
        )
    if requested == "reference":
        reason = (
            "table projection requires a new geometry resource"
            if joins
            else "no equivalent Odon-readable GeoParquet payload could be proven"
        )
        raise SpatialDataAdapterError(
            "vector_reference_unavailable", reason, element=name
        )
    directory = Path(tempfile.mkdtemp(prefix="odon-spatialdata-vector-"))
    destination = directory / f"{_safe_file_stem(name)}.parquet"
    try:
        _write_vector_snapshot(kind, name, element, destination, joins)
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return (
        destination,
        "geoparquet",
        None,
        TransferDecision(
            requested=requested,
            chosen="snapshot",
            reason=(
                "selected_table_columns_joined_into_vector"
                if joins
                else "selected_vector_materialized_as_geoparquet"
            ),
        ),
        directory,
    )


def _prove_vector_reference(
    sdata: Any,
    fresh: Any,
    root: str | Path | None,
    disk_paths: set[str],
    kind: str,
    name: str,
    element: Any,
) -> Path | None:
    if kind != "shapes" or not isinstance(root, Path) or not bool(sdata.is_self_contained(name)):
        return None
    expected = f"shapes/{name}"
    if tuple(sdata.locate_element(element)) != (expected,) or expected not in disk_paths:
        return None
    path = root / expected / "shapes.parquet"
    if not path.is_file():
        return None
    fresh_element = getattr(fresh, "shapes", {}).get(name)
    equals = getattr(element, "equals", None)
    if fresh_element is None or not callable(equals):
        return None
    try:
        equivalent = bool(equals(fresh_element))
    except Exception:
        return None
    return path if equivalent else None


def _write_vector_snapshot(
    kind: str,
    name: str,
    element: Any,
    destination: Path,
    joins: Sequence[_TableJoin],
) -> None:
    if kind == "shapes":
        frame = element.copy()
        frame = _apply_table_joins(frame, joins, name)
    elif kind == "points":
        try:
            import geopandas as gpd
        except ImportError as error:
            raise ImportError(
                "point snapshots require `pip install odon-client[spatialdata]`"
            ) from error
        frame = (
            element.compute()
            if callable(getattr(element, "compute", None))
            else element.copy()
        )
        if "x" not in frame.columns or "y" not in frame.columns:
            raise SpatialDataAdapterError(
                "unsupported_points_model",
                "selected points require x and y columns",
                element=name,
            )
        frame = gpd.GeoDataFrame(
            frame.copy(), geometry=gpd.points_from_xy(frame["x"], frame["y"])
        )
        if "instance_id" not in frame.columns:
            frame["instance_id"] = frame.index
    else:  # pragma: no cover - guarded by caller
        raise SpatialDataAdapterError("unsupported_vector_kind", kind, element=name)
    writer = getattr(frame, "to_parquet", None)
    if not callable(writer):
        raise SpatialDataAdapterError(
            "unsupported_vector_model",
            "selected vector cannot be written as GeoParquet",
            element=name,
        )
    try:
        writer(destination, index=False)
    except Exception as error:
        raise SpatialDataAdapterError(
            "vector_snapshot_failed",
            "failed to materialize the selected vector as GeoParquet",
            element=name,
            details={"destination": str(destination)},
        ) from error


def _apply_table_joins(frame: Any, joins: Sequence[_TableJoin], name: str) -> Any:
    if not joins:
        if "instance_id" not in frame.columns:
            frame["instance_id"] = frame.index
        return frame
    if "instance_id" not in frame.columns:
        frame["instance_id"] = frame.index
    for join in joins:
        obs = join.table.obs
        if join.region_key in obs.columns:
            obs = obs.loc[obs[join.region_key].astype(str) == join.target_element]
        if join.instance_key in obs.columns:
            keys = obs[join.instance_key]
        else:
            keys = obs.index
        if bool(keys.duplicated().any()):
            raise SpatialDataAdapterError(
                "duplicate_table_instance_key",
                "table instance keys must be unique for projection",
                element=join.table_name,
            )
        for column in join.columns:
            if column in frame.columns:
                raise SpatialDataAdapterError(
                    "table_column_conflict",
                    f"table column {column!r} already exists on shapes {name!r}",
                    element=join.table_name,
                )
            values = obs[column]
            if str(getattr(values, "dtype", "")) == "category":
                values = values.astype("string")
            lookup = dict(zip(keys, values))
            frame[column] = frame["instance_id"].map(lookup)
    return frame


def _safe_file_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return stem or "element"


def _backing_source(sdata: Any) -> str | Path:
    if not bool(sdata.is_backed()) or sdata.path is None:
        raise SpatialDataAdapterError(
            "unbacked_raster",
            "reference transfer requires a backed SpatialData Zarr store",
        )
    raw_path = sdata.path
    value = str(raw_path)
    scheme = urlsplit(value).scheme.lower()
    if scheme in {"http", "https", "s3"}:
        return value.rstrip("/")
    if scheme and scheme != "file":
        raise SpatialDataAdapterError(
            "remote_source_unavailable",
            "the backed SpatialData store uses an unsupported URI scheme",
            details={"path": value, "scheme": scheme},
        )
    if scheme == "file":
        from urllib.request import url2pathname

        root = Path(url2pathname(urlsplit(value).path)).expanduser().resolve()
    else:
        root = Path(raw_path).expanduser().resolve()
    if not root.is_dir():
        raise SpatialDataAdapterError(
            "source_unavailable",
            f"SpatialData backing store does not exist: {root}",
        )
    return root


def _source_uri(source: str | Path) -> str:
    return source.resolve().as_uri() if isinstance(source, Path) else source


def _select_coordinate_system(
    api: _SpatialDataApi,
    selected: Sequence[tuple[str, str, Any]],
    requested: str | None,
) -> str:
    systems: list[set[str]] = []
    for _, name, element in selected:
        transforms = api.get_transformation(element, get_all=True)
        if not isinstance(transforms, Mapping):
            raise SpatialDataAdapterError(
                "invalid_transform_metadata",
                "SpatialData returned invalid coordinate-system metadata",
                element=name,
            )
        systems.append(set(transforms))
    common = set.intersection(*systems)
    if requested is not None:
        if requested not in common:
            raise SpatialDataAdapterError(
                "coordinate_system_unavailable",
                f"coordinate system {requested!r} is not shared by all selected elements",
                details={"common": sorted(common)},
            )
        return requested
    if len(common) != 1:
        raise SpatialDataAdapterError(
            "ambiguous_coordinate_system",
            "coordinate_system must be specified unless exactly one system is shared",
            details={"common": sorted(common)},
        )
    return next(iter(common))


def _prove_direct_reference(
    api: _SpatialDataApi,
    sdata: Any,
    fresh: Any,
    root: str | Path,
    disk_paths: set[str],
    kind: str,
    name: str,
    element: Any,
) -> str:
    if not bool(sdata.is_self_contained(name)):
        raise SpatialDataAdapterError(
            "external_backing",
            "the selected element depends on files outside the SpatialData store",
            element=name,
        )
    locations = tuple(sdata.locate_element(element))
    expected = f"{'images' if kind == 'image' else 'labels'}/{name}"
    if locations != (expected,) or expected not in disk_paths:
        raise SpatialDataAdapterError(
            "unproven_backing",
            "the selected in-memory element cannot be mapped uniquely to a persisted Zarr group",
            element=name,
            details={"locations": list(locations), "expected": expected},
        )
    backing_files = tuple(str(path) for path in api.get_dask_backing_files(element))
    if not backing_files:
        raise SpatialDataAdapterError(
            "unproven_backing",
            "the selected raster exposes no Dask backing files",
            element=name,
        )
    for backing_file in backing_files:
        if not _source_contains(root, backing_file):
            raise SpatialDataAdapterError(
                "external_backing",
                "the selected raster references data outside the SpatialData store",
                element=name,
                details={"backing_file": backing_file},
            )
    fresh_element = getattr(fresh, "images" if kind == "image" else "labels")[name]
    current_payloads = _raster_payloads(element, name)
    fresh_payloads = _raster_payloads(fresh_element, name)
    fresh_backing_files = tuple(
        str(path) for path in api.get_dask_backing_files(fresh_element)
    )
    if (
        set(backing_files) != set(fresh_backing_files)
        or len(current_payloads) != len(fresh_payloads)
        or any(
            not _equivalent_backed_payload(api, current, persisted)
            for current, persisted in zip(current_payloads, fresh_payloads)
        )
    ):
        raise SpatialDataAdapterError(
            "in_memory_raster_diverged",
            "the in-memory raster graph differs from the persisted element; "
            "reference transfer would show stale pixels",
            element=name,
        )
    return expected


def _equivalent_backed_payload(api: _SpatialDataApi, current: Any, persisted: Any) -> bool:
    if api.tokenize(current) == api.tokenize(persisted):
        return True
    for attribute in ("shape", "dtype", "chunks"):
        if getattr(current, attribute, None) != getattr(persisted, attribute, None):
            return False
    # Opening the same Zarr array through two store instances can give its Dask layers different
    # generated hashes. Compare the read graph's operation structure after removing only those
    # hashes; computed/sliced/rechunked graphs retain extra or different operations and fail.
    return _dask_graph_signature(current) not in {None, ()} and (
        _dask_graph_signature(current) == _dask_graph_signature(persisted)
    )


def _dask_graph_signature(payload: Any) -> tuple[tuple[str, str], ...] | None:
    layers = getattr(getattr(payload, "dask", None), "layers", None)
    if not isinstance(layers, Mapping):
        return None
    return tuple(
        sorted(
            (
                re.sub(r"-[0-9a-f]{32}$", "", str(name)),
                type(layer).__name__,
            )
            for name, layer in layers.items()
        )
    )


def _source_contains(root: str | Path, backing_file: str) -> bool:
    if isinstance(root, Path):
        parsed = urlsplit(backing_file)
        if parsed.scheme == "file":
            from urllib.request import url2pathname

            candidate = url2pathname(parsed.path)
        elif (
            len(parsed.scheme) == 1
            and len(backing_file) > 2
            and backing_file[1] == ":"
        ):
            candidate = backing_file
        elif parsed.scheme:
            return False
        else:
            candidate = backing_file
        try:
            Path(candidate).expanduser().resolve().relative_to(root)
        except ValueError:
            return False
        return True
    normalized_root = root.rstrip("/")
    normalized_file = backing_file.rstrip("/")
    return normalized_file == normalized_root or normalized_file.startswith(
        normalized_root + "/"
    )


def _raster_payloads(element: Any, name: str) -> tuple[Any, ...]:
    if hasattr(element, "data"):
        return (element.data,)
    subtree = getattr(element, "subtree", None)
    if subtree is not None:
        payloads: list[Any] = []
        for node in subtree:
            dataset = getattr(node, "dataset", None)
            if dataset is None:
                dataset = getattr(node, "ds", None)
            if dataset is None:
                continue
            for variable in dataset.data_vars.values():
                payloads.append(variable.data)
        if payloads:
            return tuple(payloads)
    raise SpatialDataAdapterError(
        "unsupported_raster_model",
        "the selected raster model does not expose comparable array payloads",
        element=name,
    )


def _element_affine(
    api: _SpatialDataApi, element: Any, target: str, name: str
) -> list[list[float]]:
    axes = tuple(str(axis) for axis in api.get_axes_names(element))
    if "x" not in axes or "y" not in axes:
        raise SpatialDataAdapterError(
            "unsupported_axes",
            "selected elements require x and y axes",
            element=name,
            details={"axes": list(axes)},
        )
    transform = api.get_transformation(element, to_coordinate_system=target)
    matrix = transform.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    affine = [[float(value) for value in row] for row in matrix]
    if len(affine) != 3 or any(len(row) != 3 for row in affine):
        raise SpatialDataAdapterError(
            "unsupported_transform",
            "SpatialData produced a non-2D affine transform",
            element=name,
        )
    tolerance = 1e-9
    values = [value for row in affine for value in row]
    if any(not math.isfinite(value) for value in values):
        raise SpatialDataAdapterError(
            "unsupported_transform", "the affine transform is not finite", element=name
        )
    if (
        abs(affine[2][0]) > tolerance
        or abs(affine[2][1]) > tolerance
        or abs(affine[2][2] - 1.0) > tolerance
    ):
        raise SpatialDataAdapterError(
            "unsupported_transform",
            "non-affine 2D mappings are not supported",
            element=name,
            details={"matrix": affine},
        )
    determinant = affine[0][0] * affine[1][1] - affine[0][1] * affine[1][0]
    if abs(determinant) <= tolerance:
        raise SpatialDataAdapterError(
            "unsupported_transform",
            "the selected affine transform is not invertible",
            element=name,
            details={"matrix": affine},
        )
    return affine


def _coordinate_space(
    api: _SpatialDataApi, element: Any, affine: list[list[float]], name: str
) -> CoordinateSpace:
    axes = tuple(str(axis) for axis in api.get_axes_names(element))
    scale = [1.0] * len(axes)
    translation = [0.0] * len(axes)
    if abs(affine[0][1]) <= 1e-12 and abs(affine[1][0]) <= 1e-12:
        scale[axes.index("x")] = affine[0][0]
        scale[axes.index("y")] = affine[1][1]
        translation[axes.index("x")] = affine[0][2]
        translation[axes.index("y")] = affine[1][2]
        legacy_scale = tuple(scale) if all(value > 0.0 for value in scale) else ()
        legacy_translation = tuple(translation) if legacy_scale else ()
    else:
        legacy_scale = ()
        legacy_translation = ()
    return CoordinateSpace(
        axes=axes,
        units=_axis_units(element, axes),
        scale=legacy_scale,
        translation=legacy_translation,
        affine=AffineTransform(
            matrix=tuple(  # type: ignore[arg-type]
                tuple(float(value) for value in row) for row in affine
            )
        ),
    )


def _invert_affine(matrix: Sequence[Sequence[float]], name: str) -> list[list[float]]:
    a, b, tx = matrix[0]
    c, d, ty = matrix[1]
    determinant = a * d - b * c
    if abs(determinant) <= 1e-12:
        raise SpatialDataAdapterError(
            "unsupported_transform", "primary image affine is not invertible", element=name
        )
    return [
        [d / determinant, -b / determinant, (b * ty - d * tx) / determinant],
        [-c / determinant, a / determinant, (c * tx - a * ty) / determinant],
        [0.0, 0.0, 1.0],
    ]


def _multiply_affine(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> list[list[float]]:
    return [
        [
            sum(float(left[row][index]) * float(right[index][column]) for index in range(3))
            for column in range(3)
        ]
        for row in range(3)
    ]


def _axis_units(element: Any, axes: Sequence[str]) -> tuple[str, ...]:
    coordinates = getattr(element, "coords", None)
    if coordinates is None:
        return ()
    units: list[str] = []
    for axis in axes:
        try:
            coordinate = coordinates[axis]
        except (KeyError, TypeError):
            return ()
        unit = getattr(coordinate, "attrs", {}).get("unit")
        if not isinstance(unit, str) or not unit:
            return ()
        units.append(unit)
    return tuple(units)


def _raster_metadata(element: Any) -> dict[str, Any]:
    if hasattr(element, "shape") and hasattr(element, "dtype"):
        return {"shape": list(element.shape), "dtype": str(element.dtype), "multiscales": 1}
    payloads = _raster_payloads(element, "<metadata>")
    first = payloads[0]
    metadata: dict[str, Any] = {"multiscales": len(payloads)}
    if hasattr(first, "shape"):
        metadata["shape"] = list(first.shape)
    if hasattr(first, "dtype"):
        metadata["dtype"] = str(first.dtype)
    return metadata


__all__ = [
    "AsyncSpatialDataDisplay",
    "ElementDisplay",
    "ElementSelection",
    "SpatialDataAdapterError",
    "SpatialDataDisplay",
    "TableProjection",
    "TransferDecision",
    "display",
    "display_async",
]
