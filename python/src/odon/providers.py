"""Session-scoped, read-only data providers for Odon.

The initial transport is a small loopback Zarr v2 service.  It is intentionally
generic: SpatialData is only one caller, and the provider accepts any
NumPy-compatible object that supports bounded slicing.
"""

from __future__ import annotations

import itertools
import json
import secrets
import shutil
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ProviderSnapshot:
    uri: str
    generation: int
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: str
    axes: tuple[str, ...]
    cached_chunks: int


class ArrayProvider:
    """Expose bounded chunks of an array as a session-only OME-Zarr source."""

    def __init__(
        self,
        array: Any,
        *,
        axes: Sequence[str],
        chunks: Sequence[int] | None = None,
        max_concurrency: int = 8,
        cache_chunks: int = 128,
    ) -> None:
        try:
            import numpy as np
        except ImportError as error:
            raise ImportError("array providers require NumPy") from error
        candidate = getattr(array, "data", None)
        payload = (
            candidate
            if candidate is not None
            and hasattr(candidate, "shape")
            and hasattr(candidate, "dtype")
            else array
        )
        shape = tuple(int(value) for value in payload.shape)
        if not shape or any(value <= 0 for value in shape):
            raise ValueError("provider arrays must have a non-empty positive shape")
        axes = tuple(str(axis) for axis in axes)
        if len(shape) != len(axes):
            raise ValueError("provider axes must match the array rank")
        dtype = np.dtype(payload.dtype)
        if dtype.hasobject:
            raise ValueError("provider arrays cannot use object dtype")
        if chunks is None:
            chunks = _default_chunks(payload, axes, shape)
        chunks = tuple(int(value) for value in chunks)
        if len(chunks) != len(shape) or any(value <= 0 for value in chunks):
            raise ValueError("provider chunks must be positive and match the array rank")
        max_concurrency = int(max_concurrency)
        if max_concurrency <= 0:
            raise ValueError("provider max_concurrency must be positive")
        cache_chunks = int(cache_chunks)
        if cache_chunks < 0:
            raise ValueError("provider cache_chunks must not be negative")
        self._array = payload
        self._shape = shape
        self._axes = axes
        self._chunks = tuple(min(chunk, size) for chunk, size in zip(chunks, shape))
        self._dtype = dtype
        self._generation = 1
        self._token = secrets.token_urlsafe(32)
        self._semaphore = threading.BoundedSemaphore(max_concurrency)
        self._cache_limit = cache_chunks
        self._cache: OrderedDict[tuple[int, ...], bytes] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._closed = False
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def do_HEAD(self) -> None:  # noqa: N802
                provider._serve(self, include_body=False)

            def do_GET(self) -> None:  # noqa: N802
                provider._serve(self, include_body=True)

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="odon-array-provider",
            daemon=True,
        )
        self._thread.start()
        host, port = self._server.server_address[:2]
        self._uri = f"http://{host}:{port}/{self._token}/{self._generation}"

    @property
    def uri(self) -> str:
        return self._uri

    @property
    def snapshot(self) -> ProviderSnapshot:
        with self._cache_lock:
            cached = len(self._cache)
        return ProviderSnapshot(
            uri=self.uri,
            generation=self._generation,
            shape=self._shape,
            chunks=self._chunks,
            dtype=self._dtype.str,
            axes=self._axes,
            cached_chunks=cached,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)
        with self._cache_lock:
            self._cache.clear()

    def materialize(
        self,
        destination: str | Path,
        *,
        overwrite: bool = False,
        cancel: threading.Event | None = None,
    ) -> Path:
        """Explicitly write the complete provider generation as OME-Zarr."""

        try:
            import numpy as np
            import zarr
        except ImportError as error:
            raise ImportError(
                "provider snapshots require `pip install odon-client[arrays]`"
            ) from error
        path = Path(destination).expanduser().resolve()
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        parent = path.parent
        parent.mkdir(parents=True, exist_ok=True)
        estimated = int(self._dtype.itemsize)
        for size in self._shape:
            estimated *= size
        if shutil.disk_usage(parent).free < estimated:
            raise OSError(f"insufficient free space for {estimated} snapshot bytes")
        staging_parent = Path(
            tempfile.mkdtemp(prefix=f".{path.name}.odon-staging-", dir=parent)
        )
        staging = staging_parent / path.name
        try:
            try:
                root = zarr.open_group(str(staging), mode="w", zarr_format=2)
            except TypeError:  # zarr 2.x
                root = zarr.open_group(str(staging), mode="w", zarr_version=2)
            try:
                target = root.create_array(
                    "0", shape=self._shape, chunks=self._chunks, dtype=self._dtype
                )
            except AttributeError:  # zarr 2.x
                target = root.create_dataset(
                    "0", shape=self._shape, chunks=self._chunks, dtype=self._dtype
                )
            grid = tuple(
                range((size + chunk - 1) // chunk)
                for size, chunk in zip(self._shape, self._chunks)
            )
            for index in itertools.product(*grid):
                if cancel is not None and cancel.is_set():
                    raise InterruptedError("provider snapshot cancelled")
                slices = tuple(
                    slice(chunk_index * chunk, min((chunk_index + 1) * chunk, size))
                    for chunk_index, chunk, size in zip(index, self._chunks, self._shape)
                )
                actual_shape = tuple(value.stop - value.start for value in slices)
                chunk = np.frombuffer(self._chunk(index), dtype=self._dtype).reshape(
                    self._chunks
                )
                target[slices] = chunk[tuple(slice(0, size) for size in actual_shape)]
            root.attrs["multiscales"] = [
                {
                    "version": "0.4",
                    "name": "provider-snapshot",
                    "axes": [_axis_descriptor(axis) for axis in self._axes],
                    "datasets": [{"path": "0"}],
                }
            ]
            target.attrs["_ARRAY_DIMENSIONS"] = list(self._axes)
            _commit_snapshot(staging, path, overwrite=overwrite)
        except BaseException:
            shutil.rmtree(staging_parent, ignore_errors=True)
            raise
        shutil.rmtree(staging_parent, ignore_errors=True)
        return path

    def _serve(self, request: BaseHTTPRequestHandler, *, include_body: bool) -> None:
        if self._closed:
            _reply(request, HTTPStatus.SERVICE_UNAVAILABLE, b"provider closed", include_body)
            return
        expected_prefix = f"/{self._token}/{self._generation}/"
        path = urlsplit(request.path).path
        if not path.startswith(expected_prefix):
            _reply(request, HTTPStatus.NOT_FOUND, b"not found", include_body)
            return
        key = path[len(expected_prefix) :]
        static = self._static_payload(key)
        if static is not None:
            payload, content_type = static
            _reply(request, HTTPStatus.OK, payload, include_body, content_type)
            return
        if not key.startswith("0/"):
            _reply(request, HTTPStatus.NOT_FOUND, b"not found", include_body)
            return
        try:
            index = tuple(int(value) for value in key[2:].split("."))
        except ValueError:
            _reply(request, HTTPStatus.NOT_FOUND, b"not found", include_body)
            return
        if len(index) != len(self._shape) or any(
            value < 0 or value * self._chunks[axis] >= self._shape[axis]
            for axis, value in enumerate(index)
        ):
            _reply(request, HTTPStatus.NOT_FOUND, b"not found", include_body)
            return
        if not include_body:
            _reply(request, HTTPStatus.OK, b"", False, "application/octet-stream")
            return
        if not self._semaphore.acquire(blocking=False):
            _reply(request, HTTPStatus.SERVICE_UNAVAILABLE, b"provider busy", True)
            return
        try:
            payload = self._chunk(index)
        except Exception as error:
            _reply(
                request,
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"chunk computation failed: {type(error).__name__}".encode(),
                True,
            )
            return
        finally:
            self._semaphore.release()
        _reply(request, HTTPStatus.OK, payload, True, "application/octet-stream")

    def _static_payload(self, key: str) -> tuple[bytes, str] | None:
        if key == ".zgroup":
            return _json_bytes({"zarr_format": 2}), "application/json"
        if key == ".zattrs":
            axes = [_axis_descriptor(axis) for axis in self._axes]
            return (
                _json_bytes(
                    {
                        "multiscales": [
                            {
                                "version": "0.4",
                                "name": "provider",
                                "axes": axes,
                                "datasets": [{"path": "0"}],
                            }
                        ]
                    }
                ),
                "application/json",
            )
        if key == "0/.zarray":
            return (
                _json_bytes(
                    {
                        "zarr_format": 2,
                        "shape": list(self._shape),
                        "chunks": list(self._chunks),
                        "dtype": self._dtype.str,
                        "compressor": None,
                        "fill_value": 0,
                        "order": "C",
                        "filters": None,
                        "dimension_separator": ".",
                    }
                ),
                "application/json",
            )
        if key == "0/.zattrs":
            return _json_bytes({"_ARRAY_DIMENSIONS": list(self._axes)}), "application/json"
        return None

    def _chunk(self, index: tuple[int, ...]) -> bytes:
        with self._cache_lock:
            cached = self._cache.pop(index, None)
            if cached is not None:
                self._cache[index] = cached
                return cached
        import numpy as np

        slices = tuple(
            slice(chunk_index * chunk, min((chunk_index + 1) * chunk, size))
            for chunk_index, chunk, size in zip(index, self._chunks, self._shape)
        )
        selected = self._array[slices]
        if callable(getattr(selected, "compute", None)):
            selected = selected.compute()
        selected = getattr(selected, "values", selected)
        array = np.asarray(selected, dtype=self._dtype, order="C")
        if array.shape != self._chunks:
            padded = np.zeros(self._chunks, dtype=self._dtype)
            padded[tuple(slice(0, size) for size in array.shape)] = array
            array = padded
        payload = array.tobytes(order="C")
        if self._cache_limit:
            with self._cache_lock:
                self._cache[index] = payload
                while len(self._cache) > self._cache_limit:
                    self._cache.popitem(last=False)
        return payload


def _default_chunks(array: Any, axes: Sequence[str], shape: Sequence[int]) -> tuple[int, ...]:
    chunks = getattr(array, "chunks", None)
    if chunks and len(chunks) == len(shape):
        try:
            return tuple(int(axis_chunks[0]) for axis_chunks in chunks)
        except (TypeError, IndexError):
            pass
    return tuple(
        min(size, 256) if axis in {"x", "y"} else 1
        for axis, size in zip(axes, shape)
    )


def _axis_descriptor(axis: str) -> dict[str, str]:
    axis_type = (
        "space"
        if axis in {"x", "y", "z"}
        else "channel"
        if axis == "c"
        else "time"
        if axis == "t"
        else "unknown"
    )
    return {"name": axis, "type": axis_type}


def _commit_snapshot(staging: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        staging.replace(destination)
        return
    if not overwrite:
        raise FileExistsError(destination)
    backup_parent = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.odon-backup-", dir=destination.parent)
    )
    backup = backup_parent / destination.name
    destination.replace(backup)
    try:
        staging.replace(destination)
    except BaseException:
        if not destination.exists() and backup.exists():
            backup.replace(destination)
        raise
    finally:
        shutil.rmtree(backup_parent, ignore_errors=True)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def _reply(
    request: BaseHTTPRequestHandler,
    status: HTTPStatus,
    payload: bytes,
    include_body: bool,
    content_type: str = "text/plain; charset=utf-8",
) -> None:
    request.send_response(status)
    request.send_header("Content-Type", content_type)
    request.send_header("Content-Length", str(len(payload)))
    request.send_header("Cache-Control", "private, max-age=31536000, immutable")
    request.end_headers()
    if include_body:
        request.wfile.write(payload)


__all__ = ["ArrayProvider", "ProviderSnapshot"]
