# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "cellpose==3.1.1.2",
#   "packaging>=24",
#   "zarr>=2.16,<3",
# ]
# ///
"""Tune DAPI-only Cellpose nuclei segmentation on representative CyCIF tiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


EXAMPLES = Path(__file__).resolve().parent
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import python_cellpose_large_cycif as large  # noqa: E402
import python_cellpose_large_cycif_approaches as pilot  # noqa: E402


DEFAULT_OUTPUT_BASE = Path(
    "/Volumes/Extreme SSD/synapse/TNP_pilot_cycif.cellpose-dapi-sweep"
)
FORMAT = "odon-large-cycif-cellpose-dapi-sweep-v1"


DAPI_VARIANTS: tuple[pilot.ApproachSpec, ...] = (
    pilot.ApproachSpec(
        "dapi-d14",
        "DAPI · diameter 14",
        "Smaller expected nuclei; probability -1, minimum area 40 px.",
        "nuclei",
        (),
        "zeros",
        14.0,
        -1.0,
        0.4,
        40,
    ),
    pilot.ApproachSpec(
        "dapi-d18-recall",
        "DAPI · high recall",
        "Diameter 18 with probability -2 and minimum area 40 px.",
        "nuclei",
        (),
        "zeros",
        18.0,
        -2.0,
        0.4,
        40,
    ),
    pilot.ApproachSpec(
        "dapi-d18-small",
        "DAPI · retain small",
        "Diameter 18 with probability -1 and minimum area 40 px.",
        "nuclei",
        (),
        "zeros",
        18.0,
        -1.0,
        0.4,
        40,
    ),
    pilot.ApproachSpec(
        "dapi-d18-control",
        "DAPI · current control",
        "Current result: diameter 18, probability -1, minimum area 80 px.",
        "nuclei",
        (),
        "zeros",
        18.0,
        -1.0,
        0.4,
        80,
    ),
    pilot.ApproachSpec(
        "dapi-d18-precision",
        "DAPI · higher precision",
        "Diameter 18 with probability 0 and minimum area 80 px.",
        "nuclei",
        (),
        "zeros",
        18.0,
        0.0,
        0.4,
        80,
    ),
    pilot.ApproachSpec(
        "dapi-d22",
        "DAPI · diameter 22",
        "Larger expected nuclei; probability -1, minimum area 80 px.",
        "nuclei",
        (),
        "zeros",
        22.0,
        -1.0,
        0.4,
        80,
    ),
)


def plan(max_tiles: int = 2) -> dict[str, object]:
    return {
        "format": FORMAT,
        "normalization": "per-tile percentile 1.0–99.8",
        "max_tiles": max_tiles,
        "shared_inferences_per_tile": 3,
        "variants": [variant.__dict__ for variant in DAPI_VARIANTS],
    }


def load_manifest(output_base: Path) -> dict[str, object]:
    latest = json.loads((output_base / "latest.json").read_text())
    manifest = json.loads((Path(latest["run_dir"]) / "manifest.json").read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError("latest output is not a DAPI nuclei sweep")
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
    parser.add_argument("--max-tiles", type=int, default=2)
    args = parser.parse_args()

    if args.plan_only:
        print(json.dumps(plan(args.max_tiles), indent=2, sort_keys=True))
        return
    if args.view_only:
        manifest = load_manifest(args.output_base)
    else:
        manifest = pilot.run_pilot(
            args.dataset,
            args.markers if args.markers.exists() else None,
            args.output_base,
            device=args.device,
            max_tiles=args.max_tiles,
            tile_size=2048,
            overlap=256,
            simplify=1.0,
            approaches=DAPI_VARIANTS,
            format_name=FORMAT,
            display_markers=(large.DEFAULT_NUCLEAR,),
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
