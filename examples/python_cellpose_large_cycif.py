# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "cellpose==3.1.1.2",
#   "packaging>=24",
#   "zarr>=2.16,<3",
# ]
# ///
"""Segment a large CyCIF OME-Zarr in resumable tiles and compare it in Odon.

This laptop-oriented workflow adapts the overlap strategy in the earlier
``cellpose.test.py`` pipeline without allocating a whole-slide label image.
Every 2048px tile overlaps its neighbours by 256px. A cell is retained by the
tile whose inner ownership region contains its centroid, so duplicate cells are
discarded while the chosen prediction keeps a full overlap margin.

Inspect the job without importing Cellpose:

    uv run --project python python examples/python_cellpose_large_cycif.py --plan-only

Benchmark one representative tile on Apple MPS:

    uv run --script examples/python_cellpose_large_cycif.py --benchmark --device mps

Run a four-tile resumable pilot and display its partial results:

    uv run --script examples/python_cellpose_large_cycif.py \
        --max-tiles 4 --device mps --launch --serve
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = ROOT / "python" / "src"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

import odon  # noqa: E402
from odon import layouts, ui  # noqa: E402


DEFAULT_DATASET = Path(
    "/Volumes/Extreme SSD/synapse/TNP_pilot_cycif.qupath-full.ome.zarr"
)
DEFAULT_MARKERS = Path("/Volumes/Extreme SSD/synapse/synapse_zarr/markers.csv")
DEFAULT_OUTPUT_BASE = Path(
    "/Volumes/Extreme SSD/synapse/TNP_pilot_cycif.cellpose-comparison"
)
DEFAULT_EXECUTABLE = ROOT / "target" / "release" / "odon"
DEFAULT_NUCLEAR = "DNA1"
DEFAULT_CYTOPLASM = ("ECAD", "panCK", "CD45", "CD3D", "CD8A")
EXTENSION_ID = "org.odon.large_cycif_cellpose"
CONTRIBUTION_ID = "large-cycif-cellpose"
EXPECTED_SHELL_MOUNT = f"extension:{EXTENSION_ID}/{CONTRIBUTION_ID}"
RIGHT_TABS_ID = "layout:workflow.comparison.right-tabs"
COMPARISON_NODE_ID = "layout:workflow.comparison.extension.0"

SESSION_CAPABILITIES = (
    "application.open",
    "events.read",
    "ui.shell.application_control",
    "ui.shell.compose",
    "ui.shell.extension_place",
    "ui.shell.read",
    "viewer.channels.read",
    "viewer.channels.write",
    "viewer.read",
    "viewer.write",
)


@dataclass(frozen=True)
class RunSpec:
    id: str
    title: str
    description: str
    cellprob_threshold: float
    flow_threshold: float
    min_size: int = 200


RUN_SPECS: tuple[RunSpec, ...] = (
    RunSpec(
        id="legacy-permissive",
        title="Legacy permissive",
        description="Reproduces the earlier pipeline thresholds.",
        cellprob_threshold=-6.0,
        flow_threshold=2.0,
    ),
    RunSpec(
        id="balanced",
        title="Balanced",
        description="Retains marginal cells without using the full legacy relaxation.",
        cellprob_threshold=-2.0,
        flow_threshold=0.8,
    ),
    RunSpec(
        id="conservative",
        title="Conservative",
        description="Uses the standard probability and flow-quality thresholds.",
        cellprob_threshold=0.0,
        flow_threshold=0.4,
    ),
)
RUN_BY_ID = {spec.id: spec for spec in RUN_SPECS}


@dataclass(frozen=True)
class TileSpec:
    index: int
    row: int
    column: int
    y0: int
    y1: int
    x0: int
    x1: int
    inner_y0: int
    inner_y1: int
    inner_x0: int
    inner_x1: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.y1 - self.y0, self.x1 - self.x0


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def ome_metadata(dataset: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return _read_json(dataset / ".zattrs"), _read_json(dataset / "0" / ".zarray")


def marker_names(dataset: Path, marker_file: Path | None) -> list[str]:
    attrs, array = ome_metadata(dataset)
    channel_count = int(array["shape"][0])
    if marker_file is not None and marker_file.exists():
        with marker_file.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        names = [str(row.get("marker_name") or row.get("marker") or "") for row in rows]
        if len(names) >= channel_count and all(names[:channel_count]):
            return names[:channel_count]
    channels = attrs.get("omero", {}).get("channels", [])
    names = [str(channel.get("label", f"Channel {index + 1}")) for index, channel in enumerate(channels)]
    if len(names) != channel_count:
        raise ValueError(
            f"expected {channel_count} channel names, found {len(names)} in OME metadata"
        )
    return names


def viewer_channel_names(dataset: Path) -> list[str]:
    attrs, array = ome_metadata(dataset)
    channels = attrs.get("omero", {}).get("channels", [])
    names = [str(channel.get("label", f"Channel {index + 1}")) for index, channel in enumerate(channels)]
    if len(names) != int(array["shape"][0]):
        return [f"Channel {index + 1}" for index in range(int(array["shape"][0]))]
    return names


def channel_indices(names: Sequence[str], requested: Sequence[str]) -> list[int]:
    lookup = {name.casefold(): index for index, name in enumerate(names)}
    missing = [name for name in requested if name.casefold() not in lookup]
    if missing:
        raise ValueError(f"unknown marker(s) {missing}; available markers: {list(names)}")
    return [lookup[name.casefold()] for name in requested]


def iter_tiles(
    height: int,
    width: int,
    *,
    tile_size: int,
    overlap: int,
) -> Iterator[TileSpec]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be in the range 0 <= overlap < tile_size")
    stride = tile_size - overlap
    y_starts = list(range(0, height, stride))
    x_starts = list(range(0, width, stride))
    y_starts = list(dict.fromkeys(max(0, min(y, height - tile_size)) for y in y_starts))
    x_starts = list(dict.fromkeys(max(0, min(x, width - tile_size)) for x in x_starts))
    y_ends = [min(start + tile_size, height) for start in y_starts]
    x_ends = [min(start + tile_size, width) for start in x_starts]
    y_boundaries = [0, *[(left + right) // 2 for left, right in zip(y_ends[:-1], y_starts[1:])], height]
    x_boundaries = [0, *[(left + right) // 2 for left, right in zip(x_ends[:-1], x_starts[1:])], width]
    index = 0
    for row, y0 in enumerate(y_starts):
        y1 = y_ends[row]
        for column, x0 in enumerate(x_starts):
            x1 = x_ends[column]
            yield TileSpec(
                index=index,
                row=row,
                column=column,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                inner_y0=y_boundaries[row],
                inner_y1=y_boundaries[row + 1],
                inner_x0=x_boundaries[column],
                inner_x1=x_boundaries[column + 1],
            )
            index += 1


def center_first(tiles: Sequence[TileSpec], height: int, width: int) -> list[TileSpec]:
    center_y = height / 2.0
    center_x = width / 2.0
    return sorted(
        tiles,
        key=lambda tile: (
            (0.5 * (tile.y0 + tile.y1) - center_y) ** 2
            + (0.5 * (tile.x0 + tile.x1) - center_x) ** 2,
            tile.index,
        ),
    )


def dataset_plan(
    dataset: Path,
    marker_file: Path | None,
    *,
    tile_size: int,
    overlap: int,
    nuclear: str,
    cytoplasm: Sequence[str],
) -> dict[str, Any]:
    attrs, array = ome_metadata(dataset)
    names = marker_names(dataset, marker_file)
    shape = [int(value) for value in array["shape"]]
    tiles = list(iter_tiles(shape[1], shape[2], tile_size=tile_size, overlap=overlap))
    requested = [nuclear, *cytoplasm]
    indices = channel_indices(names, requested)
    return {
        "dataset": str(dataset),
        "dataset_name": attrs.get("multiscales", [{}])[0].get("name", dataset.name),
        "shape": shape,
        "dtype": array["dtype"],
        "chunks": array["chunks"],
        "nuclear": {"name": nuclear, "index": indices[0]},
        "cytoplasm": [
            {"name": name, "index": index}
            for name, index in zip(cytoplasm, indices[1:])
        ],
        "tile_size": tile_size,
        "overlap": overlap,
        "tile_count": len(tiles),
        "dense_label_bytes_per_run": shape[1] * shape[2] * 4,
        "runs": [asdict(spec) for spec in RUN_SPECS],
    }


def plan_summary() -> dict[str, Any]:
    if DEFAULT_DATASET.exists():
        plan = dataset_plan(
            DEFAULT_DATASET,
            DEFAULT_MARKERS,
            tile_size=2048,
            overlap=256,
            nuclear=DEFAULT_NUCLEAR,
            cytoplasm=DEFAULT_CYTOPLASM,
        )
    else:
        plan = {
            "dataset": str(DEFAULT_DATASET),
            "missing": True,
            "nuclear": DEFAULT_NUCLEAR,
            "cytoplasm": list(DEFAULT_CYTOPLASM),
            "tile_size": 2048,
            "overlap": 256,
            "runs": [asdict(spec) for spec in RUN_SPECS],
        }
    plan["strategy"] = {
        "checkpoint": "one atomic GeoJSONL file per tile and setting",
        "deduplication": "centroid ownership inside half-overlap regions",
        "finalization": "stream tile features into one Odon GeoJSON per setting",
        "resume": True,
    }
    return plan


def _dataset_signature(dataset: Path, plan: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(str(dataset.resolve()).encode())
    digest.update((dataset / ".zattrs").read_bytes())
    digest.update((dataset / "0" / ".zarray").read_bytes())
    digest.update(json.dumps(plan, sort_keys=True).encode())
    digest.update(json.dumps(config, sort_keys=True).encode())
    return digest.hexdigest()


def _cellpose_version() -> str:
    try:
        return importlib.metadata.version("cellpose")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _scientific_imports() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import cv2
        import numpy as np
        import torch
        import zarr
        from cellpose import dynamics, models, utils
    except ImportError as error:  # pragma: no cover - dependency guidance
        raise RuntimeError(
            "Cellpose dependencies are unavailable. Run this file with "
            "`uv run --script examples/python_cellpose_large_cycif.py ...`. "
            f"Original import error: {error}"
        ) from error
    return np, zarr, torch, cv2, (dynamics, models, utils)


def estimate_normalization(
    array: Any,
    indices: Sequence[int],
    *,
    grid: int = 8,
    sample_size: int = 256,
    low: float = 1.0,
    high: float = 99.8,
) -> dict[int, tuple[float, float]]:
    import numpy as np

    _channels, height, width = array.shape
    ys = np.linspace(0, max(0, height - sample_size), grid, dtype=int)
    xs = np.linspace(0, max(0, width - sample_size), grid, dtype=int)
    values: dict[int, list[Any]] = {index: [] for index in indices}
    for y0 in ys.tolist():
        for x0 in xs.tolist():
            for index in indices:
                values[index].append(
                    np.asarray(
                        array[index, y0 : y0 + sample_size, x0 : x0 + sample_size]
                    ).ravel()
                )
    result: dict[int, tuple[float, float]] = {}
    for index, samples in values.items():
        joined = np.concatenate(samples)
        lo, hi = np.percentile(joined, (low, high)).tolist()
        result[index] = (float(lo), float(max(hi, lo + 1.0)))
    return result


def normalize_plane(plane: Any, limits: tuple[float, float]) -> Any:
    import numpy as np

    lo, hi = limits
    normalized = np.asarray(plane, dtype=np.float32)
    normalized = np.clip(normalized, lo, hi)
    normalized -= np.float32(lo)
    normalized /= np.float32(hi - lo)
    return normalized


def load_tile_input(
    array: Any,
    tile: TileSpec,
    *,
    nuclear_index: int,
    cytoplasm_indices: Sequence[int],
    normalization: Mapping[int, tuple[float, float]],
) -> tuple[Any, float]:
    import numpy as np

    nuclear = normalize_plane(
        array[nuclear_index, tile.y0 : tile.y1, tile.x0 : tile.x1],
        normalization[nuclear_index],
    )
    cytoplasm_planes = [
        normalize_plane(
            array[index, tile.y0 : tile.y1, tile.x0 : tile.x1],
            normalization[index],
        )
        for index in cytoplasm_indices
    ]
    cytoplasm = np.mean(np.stack(cytoplasm_planes), axis=0, dtype=np.float32)
    tissue = (nuclear > 0.05) | (cytoplasm > 0.05)
    return np.stack((cytoplasm, nuclear), axis=-1), float(tissue.mean())


def make_model(model_type: str, device: str) -> tuple[Any, str]:
    _np, _zarr, torch, _cv2, modules = _scientific_imports()
    _dynamics, models, _utils = modules
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS was requested but is not available")
    if device == "auto":
        use_gpu = bool(torch.cuda.is_available() or torch.backends.mps.is_available())
    elif device == "cpu":
        use_gpu = False
    elif device in {"mps", "cuda"}:
        use_gpu = True
    else:
        raise ValueError(f"unsupported device {device!r}")
    model = models.CellposeModel(gpu=use_gpu, model_type=model_type)
    actual = str(getattr(model, "device", "gpu" if use_gpu else "cpu"))
    return model, actual


def infer_shared_flows(model: Any, image: Any, *, diameter: float) -> tuple[Any, Any, float]:
    import numpy as np

    started = time.monotonic()
    result = model.eval(
        image,
        channel_axis=-1,
        diameter=diameter,
        normalize=False,
        compute_masks=False,
    )
    flows = result[1]
    return np.asarray(flows[1]), np.asarray(flows[2]), time.monotonic() - started


def recover_mask(
    model: Any,
    dP: Any,
    cellprob: Any,
    spec: RunSpec,
    *,
    diameter: float,
) -> Any:
    import numpy as np
    from cellpose import dynamics

    model_diameter = float(getattr(model, "diam_mean", 30.0))
    rescale = 1.0 if diameter <= 0 else model_diameter / diameter
    computed = dynamics.resize_and_compute_masks(
        dP,
        cellprob,
        niter=int(200 / rescale),
        cellprob_threshold=spec.cellprob_threshold,
        flow_threshold=spec.flow_threshold,
        min_size=spec.min_size,
        device=getattr(model, "device", None),
    )
    if isinstance(computed, tuple):
        computed = computed[0]
    return np.asarray(computed)


def mask_features(
    mask: Any,
    tile: TileSpec,
    spec: RunSpec,
    *,
    simplify: float,
) -> list[dict[str, Any]]:
    import cv2
    import numpy as np

    mask_array = np.asarray(mask)
    counts = np.bincount(mask_array.astype(np.int64, copy=False).ravel())
    contours, _hierarchy = cv2.findContours(
        mask_array.astype(np.int32, copy=False),
        cv2.RETR_FLOODFILL,
        cv2.CHAIN_APPROX_NONE,
    )
    outlines: dict[int, Any] = {}
    for contour in contours:
        if contour is None or len(contour) < 3:
            continue
        x, y = contour[0, 0]
        label = int(mask_array[int(y), int(x)])
        if label <= 0:
            continue
        previous = outlines.get(label)
        if previous is None or abs(cv2.contourArea(contour)) > abs(cv2.contourArea(previous)):
            outlines[label] = contour

    features: list[dict[str, Any]] = []
    for label in sorted(outlines):
        outline = outlines[label].reshape((-1, 2))
        center_x = float(np.mean(outline[:, 0])) + tile.x0
        center_y = float(np.mean(outline[:, 1])) + tile.y0
        if not (
            tile.inner_x0 <= center_x < tile.inner_x1
            and tile.inner_y0 <= center_y < tile.inner_y1
        ):
            continue
        points = np.asarray(outline, dtype=np.float32).reshape((-1, 1, 2))
        if simplify > 0:
            points = cv2.approxPolyDP(points, float(simplify), True)
        points = points.reshape((-1, 2))
        if len(points) < 3:
            continue
        ring = [
            [round(float(point[0]) + tile.x0, 2), round(float(point[1]) + tile.y0, 2)]
            for point in points
        ]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        features.append(
            {
                "type": "Feature",
                "id": f"{spec.id}-t{tile.index}-c{label}",
                "properties": {
                    "cellpose_run": spec.id,
                    "tile": tile.index,
                    "tile_cell": label,
                    "area_px": int(counts[label]) if label < len(counts) else 0,
                    "centroid_x": round(center_x, 2),
                    "centroid_y": round(center_y, 2),
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
    return features


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def write_tile_checkpoint(
    run_dir: Path,
    tile: TileSpec,
    features_by_run: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    tissue_fraction: float,
    inference_seconds: float,
    recovery_seconds: Mapping[str, float],
) -> None:
    for run_id, features in features_by_run.items():
        target = run_dir / "tiles" / run_id / f"tile-{tile.index:04d}.geojsonl"
        _atomic_write_text(
            target,
            "".join(json.dumps(feature, separators=(",", ":")) + "\n" for feature in features),
        )
    done = {
        "tile": asdict(tile),
        "tissue_fraction": tissue_fraction,
        "inference_seconds": inference_seconds,
        "recovery_seconds": dict(recovery_seconds),
        "cells": {run_id: len(features) for run_id, features in features_by_run.items()},
    }
    _atomic_write_text(
        run_dir / "tiles" / "done" / f"tile-{tile.index:04d}.json",
        json.dumps(done, indent=2, sort_keys=True),
    )


def finalize_geojson(run_dir: Path, run_id: str) -> tuple[Path, dict[str, Any]]:
    target = (run_dir / f"{run_id}.geojson").resolve()
    temporary = target.with_suffix(".geojson.tmp")
    areas: list[int] = []
    feature_count = 0
    with temporary.open("w") as output:
        output.write(
            '{"type":"FeatureCollection","name":'
            + json.dumps(f"Cellpose {RUN_BY_ID[run_id].title}")
            + ',"odon":{"kind":"large-cycif-cellpose","run":'
            + json.dumps(run_id)
            + '},"features":['
        )
        first = True
        tile_dir = run_dir / "tiles" / run_id
        for path in sorted(tile_dir.glob("tile-*.geojsonl")):
            for line in path.open():
                feature = json.loads(line)
                if not first:
                    output.write(",")
                output.write(json.dumps(feature, separators=(",", ":")))
                first = False
                feature_count += 1
                areas.append(int(feature.get("properties", {}).get("area_px", 0)))
        output.write("]}")
    temporary.replace(target)
    areas.sort()
    if areas:
        middle = len(areas) // 2
        median = (
            float(areas[middle])
            if len(areas) % 2
            else 0.5 * (areas[middle - 1] + areas[middle])
        )
    else:
        median = 0.0
    return target, {
        "cell_count": feature_count,
        "mean_area_px": round(sum(areas) / len(areas), 2) if areas else 0.0,
        "median_area_px": round(median, 2),
        "segmented_pixels": sum(areas),
    }


def completed_tile_indices(run_dir: Path) -> set[int]:
    return {
        int(path.stem.removeprefix("tile-"))
        for path in (run_dir / "tiles" / "done").glob("tile-*.json")
    }


def benchmark_tile(
    dataset: Path,
    marker_file: Path | None,
    *,
    tile_size: int,
    overlap: int,
    nuclear: str,
    cytoplasm: Sequence[str],
    diameter: float,
    model_type: str,
    device: str,
) -> dict[str, Any]:
    np, zarr, _torch, _cv2, _modules = _scientific_imports()
    del np
    plan = dataset_plan(
        dataset,
        marker_file,
        tile_size=tile_size,
        overlap=overlap,
        nuclear=nuclear,
        cytoplasm=cytoplasm,
    )
    names = marker_names(dataset, marker_file)
    indices = channel_indices(names, [nuclear, *cytoplasm])
    array = zarr.open_group(str(dataset), mode="r")["0"]
    normalization = estimate_normalization(array, indices)
    tiles = list(iter_tiles(plan["shape"][1], plan["shape"][2], tile_size=tile_size, overlap=overlap))
    tile = center_first(tiles, plan["shape"][1], plan["shape"][2])[0]
    image, tissue_fraction = load_tile_input(
        array,
        tile,
        nuclear_index=indices[0],
        cytoplasm_indices=indices[1:],
        normalization=normalization,
    )
    model, actual_device = make_model(model_type, device)
    dP, cellprob, inference_seconds = infer_shared_flows(model, image, diameter=diameter)
    runs: dict[str, Any] = {}
    for spec in RUN_SPECS:
        started = time.monotonic()
        mask = recover_mask(model, dP, cellprob, spec, diameter=diameter)
        runs[spec.id] = {
            "cells": int(mask.max()),
            "recovery_seconds": round(time.monotonic() - started, 3),
        }
    return {
        "tile": asdict(tile),
        "tissue_fraction": round(tissue_fraction, 4),
        "device": actual_device,
        "model": model_type,
        "inference_seconds": round(inference_seconds, 3),
        "projected_all_tiles_hours": round(inference_seconds * len(tiles) / 3600.0, 2),
        "total_tiles": len(tiles),
        "runs": runs,
    }


def run_segmentation(
    dataset: Path,
    marker_file: Path | None,
    output_base: Path,
    *,
    tile_size: int,
    overlap: int,
    nuclear: str,
    cytoplasm: Sequence[str],
    diameter: float,
    model_type: str,
    device: str,
    min_tissue_fraction: float,
    simplify: float,
    max_tiles: int | None,
) -> dict[str, Any]:
    np, zarr, _torch, _cv2, _modules = _scientific_imports()
    del np
    plan = dataset_plan(
        dataset,
        marker_file,
        tile_size=tile_size,
        overlap=overlap,
        nuclear=nuclear,
        cytoplasm=cytoplasm,
    )
    names = marker_names(dataset, marker_file)
    indices = channel_indices(names, [nuclear, *cytoplasm])
    config = {
        "cellpose_version": _cellpose_version(),
        "model": model_type,
        "device_request": device,
        "diameter": diameter,
        "min_tissue_fraction": min_tissue_fraction,
        "simplify": simplify,
    }
    signature = _dataset_signature(dataset, plan, config)
    run_dir = output_base / f"run-{signature[:12]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(output_base / "latest.json", json.dumps({"run_dir": str(run_dir.resolve())}))

    array = zarr.open_group(str(dataset), mode="r")["0"]
    normalization_path = run_dir / "normalization.json"
    if normalization_path.exists():
        normalization = {
            int(index): (float(limits[0]), float(limits[1]))
            for index, limits in _read_json(normalization_path).items()
        }
        print(f"Reusing normalization from {normalization_path}", flush=True)
    else:
        print("Estimating global full-resolution normalization…", flush=True)
        normalization = estimate_normalization(array, indices)
        _atomic_write_text(
            normalization_path,
            json.dumps(
                {str(index): list(limits) for index, limits in normalization.items()},
                indent=2,
                sort_keys=True,
            ),
        )
    normalization_json = {str(index): list(limits) for index, limits in normalization.items()}
    tiles = list(iter_tiles(plan["shape"][1], plan["shape"][2], tile_size=tile_size, overlap=overlap))
    completed = completed_tile_indices(run_dir)
    pending = [tile for tile in center_first(tiles, plan["shape"][1], plan["shape"][2]) if tile.index not in completed]
    if max_tiles is not None:
        pending = pending[:max_tiles]
    model = None
    actual_device = "not-needed"
    started_all = time.monotonic()
    for position, tile in enumerate(pending, start=1):
        image, tissue_fraction = load_tile_input(
            array,
            tile,
            nuclear_index=indices[0],
            cytoplasm_indices=indices[1:],
            normalization=normalization,
        )
        print(
            f"Tile {tile.index + 1}/{len(tiles)} ({position}/{len(pending)} this run) "
            f"tissue={100 * tissue_fraction:.2f}%",
            flush=True,
        )
        if tissue_fraction < min_tissue_fraction:
            write_tile_checkpoint(
                run_dir,
                tile,
                {spec.id: [] for spec in RUN_SPECS},
                tissue_fraction=tissue_fraction,
                inference_seconds=0.0,
                recovery_seconds={spec.id: 0.0 for spec in RUN_SPECS},
            )
            continue
        if model is None:
            model, actual_device = make_model(model_type, device)
            print(f"Cellpose {model_type} loaded on {actual_device}", flush=True)
        dP, cellprob, inference_seconds = infer_shared_flows(model, image, diameter=diameter)
        features_by_run: dict[str, list[dict[str, Any]]] = {}
        recovery_seconds: dict[str, float] = {}
        for spec in RUN_SPECS:
            recovery_started = time.monotonic()
            mask = recover_mask(model, dP, cellprob, spec, diameter=diameter)
            features_by_run[spec.id] = mask_features(
                mask, tile, spec, simplify=simplify
            )
            recovery_seconds[spec.id] = time.monotonic() - recovery_started
        write_tile_checkpoint(
            run_dir,
            tile,
            features_by_run,
            tissue_fraction=tissue_fraction,
            inference_seconds=inference_seconds,
            recovery_seconds=recovery_seconds,
        )
        print(
            f"  inference={inference_seconds:.2f}s cells="
            + ", ".join(f"{key}:{len(value)}" for key, value in features_by_run.items()),
            flush=True,
        )

    completed = completed_tile_indices(run_dir)
    results: dict[str, Any] = {}
    for spec in RUN_SPECS:
        geojson_path, metrics = finalize_geojson(run_dir, spec.id)
        results[spec.id] = {
            "id": spec.id,
            "title": spec.title,
            "description": spec.description,
            "settings": asdict(spec),
            "geojson_path": str(geojson_path),
            **metrics,
        }
    manifest = {
        "format": "odon-large-cycif-cellpose-v1",
        "signature": signature,
        "dataset": str(dataset.resolve()),
        "marker_file": str(marker_file.resolve()) if marker_file is not None else None,
        "plan": plan,
        "config": config,
        "normalization": normalization_json,
        "viewer_channels": viewer_channel_names(dataset),
        "display_channel_indices": [indices[0], *indices[1:3]],
        "device": actual_device,
        "completed_tiles": len(completed),
        "total_tiles": len(tiles),
        "complete": len(completed) == len(tiles),
        "elapsed_seconds_this_run": round(time.monotonic() - started_all, 3),
        "results": results,
    }
    _atomic_write_text(run_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def load_latest_manifest(output_base: Path) -> dict[str, Any]:
    latest = output_base / "latest.json"
    if not latest.exists():
        raise FileNotFoundError(f"no large Cellpose run found under {output_base}")
    run_dir = Path(_read_json(latest)["run_dir"])
    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("format") != "odon-large-cycif-cellpose-v1":
        raise ValueError(f"unsupported manifest format in {manifest_path}")
    return manifest


def command_id(run_id: str) -> str:
    return f"extension:{EXTENSION_ID}/show-{run_id}"


def _result_text(result: Mapping[str, Any]) -> str:
    settings = result["settings"]
    object_label = str(result.get("object_label", "cells"))
    details = (
        f"{result['title']}\n\n{result['description']}\n\n"
        f"{result['cell_count']:,} {object_label}\n"
        f"Median area: {result['median_area_px']:.1f} px² · "
        f"Mean area: {result['mean_area_px']:.1f} px²"
    )
    if "cellprob_threshold" in settings:
        return (
            details
            + f"\nCell probability: {settings['cellprob_threshold']} · "
            f"Flow: {settings['flow_threshold']} · Minimum: {settings['min_size']} px"
        )
    method = settings.get("model", settings.get("method"))
    input_channels = settings.get("input_channels")
    extra = []
    if method:
        extra.append(f"Model: {method}")
    if input_channels:
        extra.append("Input: " + ", ".join(str(value) for value in input_channels))
    return details + ("\n" + " · ".join(extra) if extra else "")


def _comparison_text(results: Mapping[str, Mapping[str, Any]], selected: str) -> str:
    rows = ["All runs", ""]
    for run_id, result in results.items():
        prefix = "→ " if run_id == selected else ""
        object_label = str(result.get("object_label", "cells"))
        rows.append(f"{prefix}{result['title']}: {result['cell_count']:,} {object_label}")
    return "\n".join(rows)


def _result_ids(manifest: Mapping[str, Any]) -> list[str]:
    results = manifest["results"]
    requested = [str(run_id) for run_id in manifest.get("result_order", ())]
    return [run_id for run_id in requested if run_id in results] + [
        run_id for run_id in results if run_id not in requested
    ]


def comparison_panel(manifest: Mapping[str, Any]) -> ui.Panel:
    results = manifest["results"]
    run_ids = _result_ids(manifest)
    ordered_results = {run_id: results[run_id] for run_id in run_ids}
    initial = "balanced" if "balanced" in results else run_ids[0]
    progress = float(manifest["completed_tiles"]) / max(1, int(manifest["total_tiles"]))
    return ui.Panel(
        "large-cellpose-comparison",
        title=str(manifest.get("workflow_title", "Large Cellpose comparison")),
        children=[
            ui.Markdown(
                "introduction",
                str(
                    manifest.get(
                        "workflow_description",
                        "Large-image Cellpose\n\n"
                        "Resumable overlapping tiles with one shared inference per tile.",
                    )
                ),
            ),
            ui.Status(
                "status",
                f"Showing {results[initial]['title']} · "
                f"{manifest['completed_tiles']}/{manifest['total_tiles']} tiles",
            ),
            ui.Progress("tile-progress", value=progress, label="Segmentation progress"),
            ui.Select(
                "selected-run",
                "Segmentation run",
                options=run_ids,
                value=initial,
                action=ui.emit("run-selected"),
                event_policy=ui.Immediate(),
            ),
            ui.Grid(
                "run-buttons",
                columns=1,
                children=[
                    ui.Button(
                        f"show-{run_id}",
                        str(results[run_id]["title"]),
                        action=ui.command(
                            "ui.commands.execute", {"command_id": command_id(run_id)}
                        ),
                    )
                    for run_id in run_ids
                ],
            ),
            ui.Toggle(
                "overlay-visible",
                "Show outlines",
                value=True,
                action=ui.emit("overlay-visible"),
                event_policy=ui.Immediate(),
            ),
            ui.Button("fit-view", "Fit image", action=ui.command("viewer.camera.fit")),
            ui.Separator("metrics-separator"),
            ui.Markdown("metrics", _result_text(results[initial])),
            ui.Markdown("comparison", _comparison_text(ordered_results, initial)),
        ],
    )


def comparison_layout(panel_mount: str = EXPECTED_SHELL_MOUNT) -> ui.ShellLayout:
    base = layouts.comparison(panel_mounts=(panel_mount,))
    nodes: list[ui.ShellLayoutNode] = []
    for node in base.nodes:
        if node.id == RIGHT_TABS_ID:
            node = replace(node, selected_id=COMPARISON_NODE_ID)
        elif node.id == COMPARISON_NODE_ID:
            node = replace(node, title="Large Cellpose comparison")
        nodes.append(node)
    return ui.ShellLayout(base.root_id, tuple(nodes))


class ComparisonController:
    def __init__(self, app: odon.Client, contribution: ui.Contribution, manifest: Mapping[str, Any]) -> None:
        self.app = app
        self.contribution = contribution
        self.manifest = manifest
        self.results = manifest["results"]
        run_ids = _result_ids(manifest)
        self.selected = "balanced" if "balanced" in self.results else run_ids[0]

    def _overlay_target(self, run_id: str) -> str:
        result = self.results[run_id]
        return "objects" if result.get("object_source") == "objects" else "geojson"

    def subscribe(self) -> None:
        self.app.events.subscribe(f"ui.extension:{EXTENSION_ID}.*", self.handle_event)

    def close(self) -> None:
        try:
            if not self.app.closed:
                self.app.events.unsubscribe(f"ui.extension:{EXTENSION_ID}.*")
        except odon.ConnectionClosedError:
            pass
        finally:
            self.app.events.remove_callback(self.handle_event)

    def show(self, run_id: str, *, wait_for_resource: bool) -> None:
        result = self.results[run_id]
        overlay_target = self._overlay_target(run_id)
        if overlay_target == "objects":
            task = self.app.objects.load(result["geojson_path"])
        else:
            task = self.app.objects.load_segmentation_geojson(result["geojson_path"])
        if wait_for_resource:
            task.wait(timeout=300.0)
        self.app.objects.set_overlay_visibility(
            overlay_target != "objects", target="geojson"
        )
        self.app.objects.set_overlay_visibility(
            overlay_target == "objects", target="objects"
        )
        style = result.get("object_style")
        if overlay_target == "objects" and isinstance(style, Mapping):
            self.app.objects.set_style(target="objects", **dict(style))
            legend = result.get("object_legend")
            if isinstance(legend, Sequence) and not isinstance(legend, (str, bytes)):
                self.app.objects.set_legend(legend, target="objects")
        display_indices = self.manifest["display_channel_indices"]
        viewer_names = self.manifest["viewer_channels"]
        channels = [viewer_names[index] for index in display_indices]
        self.app.channels.set_visible(channels, mode="only")
        for name, color in zip(channels, ((65, 95, 255), (255, 60, 170), (20, 225, 220))):
            self.app.channels.set_color(name, color)
        self.selected = run_id
        self.contribution.patch_values(
            {
                "selected-run": run_id,
                "overlay-visible": True,
                "status": f"Showing {result['title']} · {result['cell_count']:,} "
                f"{result.get('object_label', 'cells')} · "
                f"{self.manifest['completed_tiles']}/{self.manifest['total_tiles']} tiles",
                "metrics": _result_text(result),
                "comparison": _comparison_text(self.results, run_id),
            }
        )

    def handle_event(self, event: odon.Event) -> None:
        data = dict(event.data) if isinstance(event.data, Mapping) else {}
        try:
            suffix = event.name.removeprefix(f"ui.extension:{EXTENSION_ID}.")
            if suffix.startswith("show-"):
                self.show(suffix.removeprefix("show-"), wait_for_resource=False)
                return
            if suffix != "input":
                return
            action = data.get("action")
            action = dict(action) if isinstance(action, Mapping) else {}
            if action.get("event") == "run-selected" and str(data.get("value")) in self.results:
                self.show(str(data["value"]), wait_for_resource=False)
            elif action.get("event") == "overlay-visible":
                visible = bool(data.get("value"))
                self.app.objects.set_overlay_visibility(
                    visible, target=self._overlay_target(self.selected)
                )
                self.contribution.patch_values({"overlay-visible": visible})
        except Exception as error:
            try:
                self.contribution.patch_values(
                    {"status": "Cellpose comparison failed", "metrics": f"{type(error).__name__}: {error}"}
                )
            except Exception:
                pass


def _connect(*, launch: bool, executable: Path) -> tuple[odon.Client, Any | None]:
    if launch:
        bootstrap = odon.launch(executable, timeout=30.0)
        process = bootstrap.launched_process
        instance_id = bootstrap.hello.instance_id
        bootstrap.close()
        return (
            odon.connect(
                instance=instance_id,
                timeout=15.0,
                client_name="large-cycif-cellpose",
                requested_capabilities=SESSION_CAPABILITIES,
            ),
            process,
        )
    return (
        odon.connect(
            client_name="large-cycif-cellpose",
            requested_capabilities=SESSION_CAPABILITIES,
        ),
        None,
    )


def view_results(
    manifest: Mapping[str, Any],
    *,
    launch: bool,
    executable: Path,
    serve: bool,
) -> None:
    dataset = Path(str(manifest["dataset"]))
    app, launched_process = _connect(launch=launch, executable=executable)
    original_layout: ui.ShellLayoutDocument | None = None
    extension: ui.Extension | None = None
    controller: ComparisonController | None = None
    try:
        app.datasets.open_ome_zarr(dataset).wait(timeout=300.0)
        original_layout = app.ui.shell.export_layout(mode="single")
        extension = app.ui.register_extension(
            id=EXTENSION_ID,
            name="Large CyCIF Cellpose comparison",
            version="1.0.0",
            capabilities=("ui.panels", "ui.actions", "viewer.read", "viewer.write"),
        )
        for run_id in _result_ids(manifest):
            result = manifest["results"][run_id]
            extension.register_command(
                f"show-{run_id}",
                f"Show Cellpose: {result['title']}",
                str(result["description"]),
                f"show-{run_id}",
                modes=("single",),
            )
        contribution = extension.register(
            comparison_panel(manifest),
            location="right.tabs",
            contribution_id=CONTRIBUTION_ID,
        )
        controller = ComparisonController(app, contribution, manifest)
        controller.subscribe()
        current = app.ui.shell.get(mode="single")
        app.ui.shell.replace_layout(
            comparison_layout(contribution.shell_mount),
            mode="single",
            if_revision=current.revision,
            transaction_id="large-cycif-cellpose-layout",
        )
        controller.show(controller.selected, wait_for_resource=True)
        app.viewer.fit()
        print(
            json.dumps(
                {
                    "instance_id": app.hello.instance_id,
                    "complete": manifest["complete"],
                    "tiles": [manifest["completed_tiles"], manifest["total_tiles"]],
                    "results": manifest["results"],
                    "interactive": serve,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        if serve:
            print("\nLarge Cellpose comparison ready. Press Ctrl+C to restore Odon.", flush=True)
            while True:
                time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if controller is not None:
                controller.close()
            if not app.closed:
                if original_layout is not None:
                    current = app.ui.shell.get(mode="single")
                    app.ui.shell.import_layout(
                        original_layout,
                        mode="single",
                        if_revision=current.revision,
                        transaction_id="large-cycif-cellpose-restore",
                    )
                if extension is not None:
                    extension.remove()
        except odon.ConnectionClosedError:
            pass
        finally:
            app.close()
            if launched_process is not None and launched_process.poll() is None:
                launched_process.terminate()
                launched_process.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--view-only", action="store_true")
    parser.add_argument("--segment-only", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--markers", type=Path, default=DEFAULT_MARKERS)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--nuclear", default=DEFAULT_NUCLEAR)
    parser.add_argument("--cytoplasm", nargs="+", default=list(DEFAULT_CYTOPLASM))
    parser.add_argument("--model", default="cyto2")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--diameter", type=float, default=35.0)
    parser.add_argument("--tile-size", type=int, default=2048)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.002)
    parser.add_argument("--simplify", type=float, default=1.5)
    parser.add_argument("--max-tiles", type=int)
    args = parser.parse_args()

    if args.plan_only:
        print(json.dumps(plan_summary(), indent=2, sort_keys=True))
        return
    if args.view_only and (args.benchmark or args.segment_only):
        parser.error("--view-only cannot be combined with --benchmark or --segment-only")
    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)
    marker_file = args.markers if args.markers.exists() else None
    if args.benchmark:
        print(
            json.dumps(
                benchmark_tile(
                    args.dataset,
                    marker_file,
                    tile_size=args.tile_size,
                    overlap=args.overlap,
                    nuclear=args.nuclear,
                    cytoplasm=args.cytoplasm,
                    diameter=args.diameter,
                    model_type=args.model,
                    device=args.device,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.view_only:
        manifest = load_latest_manifest(args.output_base)
    else:
        manifest = run_segmentation(
            args.dataset,
            marker_file,
            args.output_base,
            tile_size=args.tile_size,
            overlap=args.overlap,
            nuclear=args.nuclear,
            cytoplasm=args.cytoplasm,
            diameter=args.diameter,
            model_type=args.model,
            device=args.device,
            min_tissue_fraction=args.min_tissue_fraction,
            simplify=args.simplify,
            max_tiles=args.max_tiles,
        )
    if not args.segment_only:
        if args.launch and not args.executable.exists():
            raise FileNotFoundError(args.executable)
        view_results(
            manifest,
            launch=args.launch,
            executable=args.executable,
            serve=args.serve,
        )


if __name__ == "__main__":
    main()
