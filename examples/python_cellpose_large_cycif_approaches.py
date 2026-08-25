# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "cellpose==3.1.1.2",
#   "packaging>=24",
#   "zarr>=2.16,<3",
# ]
# ///
"""Compare DAPI-only and membrane-guided Cellpose inputs on large CyCIF tiles.

This is an intentionally bounded pilot. It uses the same centre-out tiles and
centroid ownership regions as ``python_cellpose_large_cycif.py``, but writes to
a separate output directory so it cannot contaminate a resumable whole-slide
run.

    uv run --script examples/python_cellpose_large_cycif_approaches.py \
        --device mps --launch --serve
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


EXAMPLES = Path(__file__).resolve().parent
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import python_cellpose_large_cycif as large  # noqa: E402


DEFAULT_OUTPUT_BASE = Path(
    "/Volumes/Extreme SSD/synapse/TNP_pilot_cycif.cellpose-approach-pilot"
)


@dataclass(frozen=True)
class ApproachSpec:
    id: str
    title: str
    description: str
    model_type: str
    markers: tuple[str, ...]
    aggregation: str
    diameter: float
    cellprob_threshold: float
    flow_threshold: float
    min_size: int


APPROACHES: tuple[ApproachSpec, ...] = (
    ApproachSpec(
        "dapi-nuclei",
        "DAPI nuclei",
        "DAPI-only nuclei model; outlines are nuclei rather than whole-cell boundaries.",
        "nuclei",
        (),
        "zeros",
        18.0,
        -1.0,
        0.4,
        80,
    ),
    ApproachSpec(
        "ecad-membrane",
        "DAPI + ECAD",
        "Epithelial membrane-guided cyto2 input using locally normalized ECAD.",
        "cyto2",
        ("ECAD",),
        "single",
        25.0,
        -2.0,
        0.8,
        100,
    ),
    ApproachSpec(
        "cd45-membrane",
        "DAPI + CD45",
        "Immune membrane-guided cyto2 input using locally normalized CD45.",
        "cyto2",
        ("CD45",),
        "single",
        25.0,
        -2.0,
        0.8,
        100,
    ),
    ApproachSpec(
        "dual-membrane-max",
        "DAPI + membrane max",
        "Pixelwise maximum of locally normalized ECAD and CD45 preserves either membrane signal.",
        "cyto2",
        ("ECAD", "CD45"),
        "max",
        25.0,
        -2.0,
        0.8,
        100,
    ),
    ApproachSpec(
        "broad-local-mean",
        "DAPI + broad mean",
        "Local-normalization control: mean of ECAD, panCK, CD45, CD3D, and CD8A.",
        "cyto2",
        ("ECAD", "panCK", "CD45", "CD3D", "CD8A"),
        "mean",
        25.0,
        -2.0,
        0.8,
        100,
    ),
    ApproachSpec(
        "broad-local-max",
        "DAPI + broad max",
        "Pixelwise maximum of five locally normalized cell-associated markers avoids dilution.",
        "cyto2",
        ("ECAD", "panCK", "CD45", "CD3D", "CD8A"),
        "max",
        25.0,
        -2.0,
        0.8,
        100,
    ),
)


def normalize_local(plane: Any, *, low: float = 1.0, high: float = 99.8) -> Any:
    import numpy as np

    values = np.asarray(plane, dtype=np.float32)
    lo, hi = np.percentile(values, (low, high)).tolist()
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((values - np.float32(lo)) / np.float32(hi - lo), 0.0, 1.0)


def approach_input(
    approach: ApproachSpec,
    normalized: Mapping[str, Any],
) -> Any:
    import numpy as np

    nuclear = normalized[large.DEFAULT_NUCLEAR]
    if approach.model_type == "nuclei":
        return np.stack((nuclear, np.zeros_like(nuclear)), axis=-1)
    planes = [normalized[name] for name in approach.markers]
    if approach.aggregation == "single":
        cytoplasm = planes[0]
    elif approach.aggregation == "mean":
        cytoplasm = np.mean(np.stack(planes), axis=0, dtype=np.float32)
    elif approach.aggregation == "max":
        cytoplasm = np.max(np.stack(planes), axis=0)
    else:
        raise ValueError(f"unsupported aggregation {approach.aggregation!r}")
    return np.stack((cytoplasm, nuclear), axis=-1)


def _metrics(features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    areas = sorted(int(feature["properties"]["area_px"]) for feature in features)
    if not areas:
        return {"cell_count": 0, "mean_area_px": 0.0, "median_area_px": 0.0, "segmented_pixels": 0}
    middle = len(areas) // 2
    median = float(areas[middle]) if len(areas) % 2 else 0.5 * (areas[middle - 1] + areas[middle])
    return {
        "cell_count": len(areas),
        "mean_area_px": round(sum(areas) / len(areas), 2),
        "median_area_px": round(median, 2),
        "segmented_pixels": sum(areas),
    }


def _write_geojson(path: Path, approach: ApproachSpec, features: Sequence[Mapping[str, Any]]) -> None:
    document = {
        "type": "FeatureCollection",
        "name": f"Cellpose approach: {approach.title}",
        "odon": {"kind": "large-cycif-cellpose-approach", "approach": approach.id},
        "features": list(features),
    }
    large._atomic_write_text(path, json.dumps(document, separators=(",", ":")))


def approach_plan(max_tiles: int = 4) -> dict[str, Any]:
    return {
        "normalization": "per-tile percentile 1.0–99.8",
        "tile_count": max_tiles,
        "approaches": [asdict(approach) for approach in APPROACHES],
        "output_base": str(DEFAULT_OUTPUT_BASE),
    }


def run_pilot(
    dataset: Path,
    markers: Path | None,
    output_base: Path,
    *,
    device: str,
    max_tiles: int,
    tile_size: int,
    overlap: int,
    simplify: float,
    approaches: Sequence[ApproachSpec] = APPROACHES,
    format_name: str = "odon-large-cycif-cellpose-approaches-v1",
    display_markers: Sequence[str] = (large.DEFAULT_NUCLEAR, "ECAD", "CD45"),
) -> dict[str, Any]:
    import numpy as np
    import zarr

    names = large.marker_names(dataset, markers)
    required_markers = [large.DEFAULT_NUCLEAR]
    for approach in approaches:
        for marker in approach.markers:
            if marker not in required_markers:
                required_markers.append(marker)
    for marker in display_markers:
        if marker not in required_markers:
            required_markers.append(marker)
    indices = large.channel_indices(names, required_markers)
    marker_indices = dict(zip(required_markers, indices))
    array = zarr.open_group(str(dataset), mode="r")["0"]
    tiles = list(
        large.iter_tiles(
            int(array.shape[1]),
            int(array.shape[2]),
            tile_size=tile_size,
            overlap=overlap,
        )
    )
    selected_tiles = large.center_first(tiles, int(array.shape[1]), int(array.shape[2]))[:max_tiles]
    config = {
        "format": format_name,
        "cellpose_version": large._cellpose_version(),
        "normalization": "per-tile-p1-p99.8",
        "tile_size": tile_size,
        "overlap": overlap,
        "simplify": simplify,
        "tile_indices": [tile.index for tile in selected_tiles],
        "approaches": [asdict(approach) for approach in approaches],
    }
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    run_dir = output_base / f"run-{signature}"
    run_dir.mkdir(parents=True, exist_ok=True)
    large._atomic_write_text(output_base / "latest.json", json.dumps({"run_dir": str(run_dir.resolve())}))

    features: dict[str, list[dict[str, Any]]] = {approach.id: [] for approach in approaches}
    timings: dict[str, float] = {approach.id: 0.0 for approach in approaches}
    models: dict[str, Any] = {}
    actual_devices: dict[str, str] = {}
    started = time.monotonic()
    for position, tile in enumerate(selected_tiles, start=1):
        raw = {
            marker: np.asarray(array[index, tile.y0 : tile.y1, tile.x0 : tile.x1])
            for marker, index in marker_indices.items()
        }
        normalized = {marker: normalize_local(plane) for marker, plane in raw.items()}
        tissue_fraction = float((normalized[large.DEFAULT_NUCLEAR] > 0.05).mean())
        print(
            f"Tile {tile.index + 1}/{len(tiles)} ({position}/{len(selected_tiles)}) "
            f"DNA tissue={100 * tissue_fraction:.2f}%",
            flush=True,
        )
        flows: dict[tuple[Any, ...], tuple[Any, Any]] = {}
        for approach in approaches:
            if approach.model_type not in models:
                models[approach.model_type], actual_devices[approach.model_type] = large.make_model(
                    approach.model_type, device
                )
                print(
                    f"  loaded {approach.model_type} on {actual_devices[approach.model_type]}",
                    flush=True,
                )
            model = models[approach.model_type]
            image = approach_input(approach, normalized)
            inference_started = time.monotonic()
            flow_key = (
                approach.model_type,
                approach.markers,
                approach.aggregation,
                approach.diameter,
            )
            if flow_key not in flows:
                dP, cellprob, _inference_seconds = large.infer_shared_flows(
                    model, image, diameter=approach.diameter
                )
                flows[flow_key] = (dP, cellprob)
            else:
                dP, cellprob = flows[flow_key]
            run_spec = large.RunSpec(
                approach.id,
                approach.title,
                approach.description,
                approach.cellprob_threshold,
                approach.flow_threshold,
                approach.min_size,
            )
            mask = large.recover_mask(
                model,
                dP,
                cellprob,
                run_spec,
                diameter=approach.diameter,
            )
            tile_features = large.mask_features(mask, tile, run_spec, simplify=simplify)
            features[approach.id].extend(tile_features)
            timings[approach.id] += time.monotonic() - inference_started
            print(f"  {approach.id}: {len(tile_features)} objects", flush=True)

    results: dict[str, Any] = {}
    for approach in approaches:
        path = (run_dir / f"{approach.id}.geojson").resolve()
        _write_geojson(path, approach, features[approach.id])
        results[approach.id] = {
            "id": approach.id,
            "title": approach.title,
            "description": approach.description,
            "settings": asdict(approach),
            "geojson_path": str(path),
            "elapsed_seconds": round(timings[approach.id], 3),
            **_metrics(features[approach.id]),
        }
    manifest = {
        "format": config["format"],
        "signature": signature,
        "dataset": str(dataset.resolve()),
        "marker_file": str(markers.resolve()) if markers is not None else None,
        "config": config,
        "viewer_channels": large.viewer_channel_names(dataset),
        "display_channel_indices": [marker_indices[marker] for marker in display_markers],
        "device": actual_devices,
        "completed_tiles": len(selected_tiles),
        "total_tiles": len(selected_tiles),
        "complete": True,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "result_order": [approach.id for approach in approaches],
        "results": results,
    }
    large._atomic_write_text(run_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def load_manifest(output_base: Path) -> dict[str, Any]:
    latest = json.loads((output_base / "latest.json").read_text())
    manifest = json.loads((Path(latest["run_dir"]) / "manifest.json").read_text())
    if manifest.get("format") != "odon-large-cycif-cellpose-approaches-v1":
        raise ValueError("latest output is not an approach pilot")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--view-only", action="store_true")
    parser.add_argument("--segment-only", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--dataset", type=Path, default=large.DEFAULT_DATASET)
    parser.add_argument("--markers", type=Path, default=large.DEFAULT_MARKERS)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--executable", type=Path, default=large.DEFAULT_EXECUTABLE)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--max-tiles", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=2048)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--simplify", type=float, default=1.5)
    args = parser.parse_args()

    if args.plan_only:
        print(json.dumps(approach_plan(args.max_tiles), indent=2, sort_keys=True))
        return
    if args.view_only:
        manifest = load_manifest(args.output_base)
    else:
        manifest = run_pilot(
            args.dataset,
            args.markers if args.markers.exists() else None,
            args.output_base,
            device=args.device,
            max_tiles=args.max_tiles,
            tile_size=args.tile_size,
            overlap=args.overlap,
            simplify=args.simplify,
        )
    if not args.segment_only:
        large.view_results(
            manifest,
            launch=args.launch,
            executable=args.executable,
            serve=args.serve,
        )


if __name__ == "__main__":
    main()
