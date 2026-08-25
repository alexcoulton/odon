# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "instanseg-torch==0.1.1",
#   "opencv-python-headless>=4.9",
#   "zarr>=2.16,<3",
# ]
# ///
"""Compare InstanSeg with the selected DAPI-only Cellpose result in Odon.

The pilot deliberately uses the same centre-out tiles as the DAPI Cellpose
sweep. InstanSeg is evaluated once with DAPI only and once with DAPI plus a
small phenotype-spanning marker panel. The multiplex inference yields both
nuclear and whole-cell boundaries.

    uv run --script examples/python_instanseg_large_cycif_pilot.py \
        --device mps --launch --serve
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


EXAMPLES = Path(__file__).resolve().parent
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import python_cellpose_dapi_nuclei_sweep as dapi_sweep  # noqa: E402
import python_cellpose_large_cycif as large  # noqa: E402
import python_cellpose_large_cycif_approaches as cellpose_pilot  # noqa: E402


DEFAULT_OUTPUT_BASE = Path(
    "/Volumes/Extreme SSD/synapse/TNP_pilot_cycif.instanseg-pilot"
)
FORMAT = "odon-large-cycif-instanseg-pilot-v1"
MODEL = "fluorescence_nuclei_and_cells"
MULTIPLEX_MARKERS = ("DNA1", "ECAD", "panCK", "CD45", "CD3D", "CD8A")
INTENSITY_PROPERTY = "mean_channel_1"
INTENSITY_PALETTE = "viridis"


def plan(max_tiles: int = 2, pixel_size: float = 0.5) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "model": MODEL,
        "max_tiles": max_tiles,
        "pixel_size_um": pixel_size,
        "pixel_size_note": "explicit assumption; source OME metadata records an implausible 1 nm/px",
        "comparisons": [
            "Cellpose nuclei: DAPI, diameter 18, probability -1, min area 40 px",
            "InstanSeg nuclei: DAPI only",
            "InstanSeg nuclei: DAPI plus phenotype-spanning markers",
            "InstanSeg cells: DAPI plus phenotype-spanning markers",
        ],
        "multiplex_markers": list(MULTIPLEX_MARKERS),
    }


def _write_geojson(path: Path, title: str, features: Sequence[Mapping[str, Any]]) -> None:
    document = {
        "type": "FeatureCollection",
        "name": title,
        "odon": {"kind": "large-cycif-instanseg-pilot"},
        "features": list(features),
    }
    large._atomic_write_text(path, json.dumps(document, separators=(",", ":")))


def _result(
    run_dir: Path,
    run_id: str,
    title: str,
    description: str,
    input_channels: Sequence[str],
    target: str,
    features: Sequence[Mapping[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    path = (run_dir / f"{run_id}.geojson").resolve()
    _write_geojson(path, title, features)
    return {
        "id": run_id,
        "title": title,
        "description": description,
        "settings": {
            "model": MODEL,
            "input_channels": list(input_channels),
            "target": target,
        },
        "geojson_path": str(path),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "object_label": target,
        **cellpose_pilot._metrics(features),
    }


def _cellpose_reference(max_tiles: int) -> dict[str, Any]:
    manifest = dapi_sweep.load_manifest(dapi_sweep.DEFAULT_OUTPUT_BASE)
    expected_tiles = list(manifest["config"]["tile_indices"][:max_tiles])
    if int(manifest["completed_tiles"]) != max_tiles or len(expected_tiles) != max_tiles:
        raise ValueError(
            "the latest DAPI sweep does not contain exactly the requested comparison tiles"
        )
    result = dict(manifest["results"]["dapi-d18-small"])
    result.update(
        id="cellpose-dapi-d18-small",
        title="Cellpose nuclei · DAPI d18",
        description="Selected DAPI-only Cellpose setting: diameter 18, retain small nuclei.",
        object_label="nuclei",
    )
    return {"manifest": manifest, "result": result, "tile_indices": expected_tiles}


def _intensity_cache_signature(source: Path, dataset: Path, channel_index: int) -> str:
    digest = hashlib.sha256()
    digest.update(str(source.resolve()).encode())
    digest.update(str(source.stat().st_size).encode())
    digest.update(str(source.stat().st_mtime_ns).encode())
    digest.update((dataset / "0" / ".zarray").read_bytes())
    digest.update(f"{channel_index}:mean-level-0-independent-polygons-v3".encode())
    return digest.hexdigest()[:12]


def add_channel_intensity_results(
    manifest: Mapping[str, Any], *, channel_index: int = 0
) -> dict[str, Any]:
    """Attach exact means to every result and apply one shared continuous scale."""
    import cv2
    import numpy as np
    import zarr

    updated = dict(manifest)
    updated["results"] = dict(manifest["results"])
    dataset = Path(str(manifest["dataset"]))
    summaries: list[dict[str, Any]] = []
    for run_id in updated.get("result_order", updated["results"].keys()):
        source_result = dict(updated["results"][run_id])
        source = Path(source_result["geojson_path"])
        signature = _intensity_cache_signature(source, dataset, channel_index)
        output = source.with_name(f"{source.stem}-mean-channel-1-{signature}.geojson")
        sidecar = output.with_suffix(".result.json")
        if sidecar.exists() and output.exists():
            measured_result = json.loads(sidecar.read_text())
            updated["results"][run_id] = measured_result
            summaries.append(measured_result["intensity"])
            continue
        document = json.loads(source.read_text())
        features = document.get("features", [])
        if not isinstance(features, list):
            raise ValueError("segmentation GeoJSON features must be a list")
        config = manifest["config"]
        array = zarr.open_group(str(dataset), mode="r")["0"]
        tiles = {
            tile.index: tile
            for tile in large.iter_tiles(
                int(array.shape[1]),
                int(array.shape[2]),
                tile_size=int(config["tile_size"]),
                overlap=int(config["overlap"]),
            )
        }
        by_tile: dict[int, list[dict[str, Any]]] = {}
        for feature in features:
            tile_index = int(feature["properties"]["tile"])
            by_tile.setdefault(tile_index, []).append(feature)
        measured = 0
        for tile_index, tile_features in by_tile.items():
            tile = tiles[tile_index]
            plane = np.asarray(
                array[channel_index, tile.y0 : tile.y1, tile.x0 : tile.x1]
            )
            for feature in tile_features:
                geometry = feature.get("geometry", {})
                rings = geometry.get("coordinates", [])
                if geometry.get("type") != "Polygon" or not rings:
                    continue
                local_rings = []
                for ring in rings:
                    points = np.asarray(ring, dtype=np.float64)
                    points[:, 0] -= tile.x0
                    points[:, 1] -= tile.y0
                    local_rings.append(points)
                outer = local_rings[0]
                x0 = max(0, int(np.floor(outer[:, 0].min())))
                y0 = max(0, int(np.floor(outer[:, 1].min())))
                x1 = min(plane.shape[1] - 1, int(np.ceil(outer[:, 0].max())))
                y1 = min(plane.shape[0] - 1, int(np.ceil(outer[:, 1].max())))
                if x0 > x1 or y0 > y1:
                    continue
                mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
                offset = np.asarray([x0, y0], dtype=np.float64)
                cv2.fillPoly(
                    mask,
                    [np.rint(outer - offset).astype(np.int32)],
                    1,
                )
                for hole in local_rings[1:]:
                    cv2.fillPoly(
                        mask,
                        [np.rint(hole - offset).astype(np.int32)],
                        0,
                    )
                pixels = plane[y0 : y1 + 1, x0 : x1 + 1][mask != 0]
                if pixels.size == 0:
                    continue
                feature["properties"][INTENSITY_PROPERTY] = round(float(pixels.mean()), 3)
                measured += 1
        if measured != len(features):
            raise ValueError(
                f"measured {measured:,} of {len(features):,} nuclei; refusing partial intensity output"
            )
        values = np.asarray(
            [float(feature["properties"][INTENSITY_PROPERTY]) for feature in features],
            dtype=np.float64,
        )
        intensity_summary = {
            "minimum": round(float(values.min()), 3),
            "maximum": round(float(values.max()), 3),
            "mean": round(float(values.mean()), 3),
            "median": round(float(np.median(values)), 3),
            "count": int(values.size),
        }
        document["name"] = f"{source_result['title']}: Channel 1 mean intensity"
        document["odon"] = {
            "kind": "large-cycif-instanseg-intensity",
            "metric": "mean",
            "channel_index": channel_index,
            "channel_name": "Channel 1",
            "level": 0,
            "property": INTENSITY_PROPERTY,
        }
        large._atomic_write_text(output, json.dumps(document, separators=(",", ":")))
        measured_result = {
            **source_result,
            "description": f"{source_result['description']} Fills show exact mean Channel 1 intensity.",
            "geojson_path": str(output.resolve()),
            "object_source": "objects",
            "intensity": intensity_summary,
            "settings": {
                **dict(source_result["settings"]),
                "measurement": "mean",
                "measurement_channel": "Channel 1",
                "measurement_level": 0,
            },
        }
        large._atomic_write_text(sidecar, json.dumps(measured_result, indent=2, sort_keys=True))
        updated["results"][run_id] = measured_result
        summaries.append(intensity_summary)

    minimum = min(float(summary["minimum"]) for summary in summaries)
    maximum = max(float(summary["maximum"]) for summary in summaries)
    if minimum >= maximum:
        epsilon = max(abs(minimum), 1.0) * 1.0e-9
        minimum -= epsilon
        maximum += epsilon
    shared_domain = [minimum, maximum]
    for run_id in updated.get("result_order", updated["results"].keys()):
        result = updated["results"][run_id]
        result["object_style"] = {
            "visible": True,
            "fill_cells": True,
            "fill_opacity": 0.72,
            "opacity": 0.85,
            "color_mapping": {
                "mode": "continuous",
                "property": INTENSITY_PROPERTY,
                "palette": INTENSITY_PALETTE,
                "domain": shared_domain,
                "scale": "linear",
                "reverse": False,
                "out_of_range": "clamp",
                "missing_color_rgb": None,
            },
        }
        result.pop("object_legend", None)
    updated["shared_intensity_domain"] = shared_domain
    return updated


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
    pixel_size: float,
) -> dict[str, Any]:
    import numpy as np
    import torch
    import zarr
    from instanseg import InstanSeg

    reference = _cellpose_reference(max_tiles)
    names = large.marker_names(dataset, markers)
    indices = large.channel_indices(names, MULTIPLEX_MARKERS)
    marker_indices = dict(zip(MULTIPLEX_MARKERS, indices))
    array = zarr.open_group(str(dataset), mode="r")["0"]
    all_tiles = list(
        large.iter_tiles(
            int(array.shape[1]), int(array.shape[2]), tile_size=tile_size, overlap=overlap
        )
    )
    selected_tiles = large.center_first(
        all_tiles, int(array.shape[1]), int(array.shape[2])
    )[:max_tiles]
    if [tile.index for tile in selected_tiles] != reference["tile_indices"]:
        raise ValueError("InstanSeg and Cellpose tile selections do not match")

    config = {
        **plan(max_tiles, pixel_size),
        "tile_size": tile_size,
        "overlap": overlap,
        "simplify": simplify,
        "tile_indices": [tile.index for tile in selected_tiles],
    }
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    run_dir = output_base / f"run-{signature}"
    run_dir.mkdir(parents=True, exist_ok=True)
    large._atomic_write_text(
        output_base / "latest.json", json.dumps({"run_dir": str(run_dir.resolve())})
    )

    model = InstanSeg(MODEL, device=device, verbosity=1)
    actual_device = str(model.inference_device)
    collected: dict[str, list[dict[str, Any]]] = {
        "instanseg-dapi-nuclei": [],
        "instanseg-multiplex-nuclei": [],
        "instanseg-multiplex-cells": [],
    }
    timings = {key: 0.0 for key in collected}
    started_all = time.monotonic()
    for position, tile in enumerate(selected_tiles, start=1):
        raw = np.stack(
            [
                np.asarray(array[index, tile.y0 : tile.y1, tile.x0 : tile.x1])
                for index in indices
            ]
        )
        print(
            f"Tile {tile.index + 1}/{len(all_tiles)} "
            f"({position}/{len(selected_tiles)}) on {actual_device}",
            flush=True,
        )

        started = time.monotonic()
        dapi_labels = model.eval_medium_image(
            raw[:1],
            pixel_size=pixel_size,
            tile_size=512,
            batch_size=1,
            target="nuclei",
            return_image_tensor=False,
        )[0, 0].to(dtype=torch.int32).numpy()
        timings["instanseg-dapi-nuclei"] += time.monotonic() - started

        started = time.monotonic()
        multiplex_labels = model.eval_medium_image(
            raw,
            pixel_size=pixel_size,
            tile_size=512,
            batch_size=1,
            target="all_outputs",
            return_image_tensor=False,
        )[0].to(dtype=torch.int32).numpy()
        multiplex_seconds = time.monotonic() - started
        timings["instanseg-multiplex-nuclei"] += multiplex_seconds
        timings["instanseg-multiplex-cells"] += multiplex_seconds

        masks = {
            "instanseg-dapi-nuclei": dapi_labels,
            "instanseg-multiplex-nuclei": multiplex_labels[0],
            "instanseg-multiplex-cells": multiplex_labels[1],
        }
        for run_id, mask in masks.items():
            spec = large.RunSpec(run_id, run_id, run_id, 0.0, 0.0, 0)
            features = large.mask_features(mask, tile, spec, simplify=simplify)
            collected[run_id].extend(features)
            print(f"  {run_id}: {len(features)} objects", flush=True)

    results = {
        "cellpose-dapi-d18-small": reference["result"],
        "instanseg-dapi-nuclei": _result(
            run_dir,
            "instanseg-dapi-nuclei",
            "InstanSeg nuclei · DAPI",
            "Independent DAPI-only nuclei prediction from InstanSeg.",
            ("DNA1",),
            "nuclei",
            collected["instanseg-dapi-nuclei"],
            timings["instanseg-dapi-nuclei"],
        ),
        "instanseg-multiplex-nuclei": _result(
            run_dir,
            "instanseg-multiplex-nuclei",
            "InstanSeg nuclei · multiplex",
            "Channel-invariant nuclei prediction using DAPI and five cell-associated markers.",
            MULTIPLEX_MARKERS,
            "nuclei",
            collected["instanseg-multiplex-nuclei"],
            timings["instanseg-multiplex-nuclei"],
        ),
        "instanseg-multiplex-cells": _result(
            run_dir,
            "instanseg-multiplex-cells",
            "InstanSeg cells · multiplex",
            "Whole-cell prediction from the same channel-invariant multiplex inference.",
            MULTIPLEX_MARKERS,
            "cells",
            collected["instanseg-multiplex-cells"],
            timings["instanseg-multiplex-cells"],
        ),
    }
    manifest = {
        "format": FORMAT,
        "signature": signature,
        "dataset": str(dataset.resolve()),
        "marker_file": str(markers.resolve()) if markers is not None else None,
        "config": config,
        "viewer_channels": large.viewer_channel_names(dataset),
        "display_channel_indices": [marker_indices["DNA1"]],
        "device": actual_device,
        "completed_tiles": len(selected_tiles),
        "total_tiles": len(selected_tiles),
        "complete": True,
        "elapsed_seconds": round(time.monotonic() - started_all, 3),
        "result_order": list(results),
        "results": results,
        "workflow_title": "Cell segmentation comparison",
        "workflow_description": (
            "Cellpose versus InstanSeg\n\n"
            "Matched tiles compare DAPI-only nuclei with multiplex nuclei and whole-cell boundaries."
        ),
    }
    large._atomic_write_text(
        run_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True)
    )
    return manifest


def load_manifest(output_base: Path) -> dict[str, Any]:
    latest = json.loads((output_base / "latest.json").read_text())
    manifest = json.loads((Path(latest["run_dir"]) / "manifest.json").read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError("latest output is not an InstanSeg pilot")
    # Keep every comparison route visually matched: changing the segmentation
    # must not also introduce phenotype channels into the canvas.
    manifest["display_channel_indices"] = [0]
    for result in manifest.get("results", {}).values():
        if "object_label" not in result:
            target = result.get("settings", {}).get("target")
            result["object_label"] = target if target in {"nuclei", "cells"} else "nuclei"
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
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--max-tiles", type=int, default=2)
    parser.add_argument("--tile-size", type=int, default=2048)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--simplify", type=float, default=1.0)
    parser.add_argument("--pixel-size", type=float, default=0.5)
    args = parser.parse_args()

    if args.plan_only:
        print(json.dumps(plan(args.max_tiles, args.pixel_size), indent=2, sort_keys=True))
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
            pixel_size=args.pixel_size,
        )
    manifest = add_channel_intensity_results(manifest, channel_index=0)
    if not args.segment_only:
        large.view_results(
            manifest, launch=args.launch, executable=args.executable, serve=args.serve
        )


if __name__ == "__main__":
    main()
