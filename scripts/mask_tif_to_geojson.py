#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.features import shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a labeled TIFF mask into GeoJSON polygons."
    )
    parser.add_argument("input_tif", type=Path)
    parser.add_argument("output_geojson", type=Path)
    parser.add_argument(
        "--background",
        type=int,
        default=0,
        help="Label value to treat as background and exclude from output.",
    )
    parser.add_argument(
        "--band",
        type=int,
        default=1,
        help="Raster band to polygonize.",
    )
    return parser.parse_args()


def dataset_metadata(src: rasterio.io.DatasetReader) -> tuple[dict[str, object], Affine]:
    transform = src.transform
    pixel_space = src.crs is None and transform == Affine.identity()
    if pixel_space:
        metadata = {
            "coordinate_space": "pixel",
            "origin": "upper-left",
            "note": "Source TIFF is not georeferenced; polygon coordinates are pixel coordinates.",
        }
        transform = Affine.identity()
    else:
        metadata = {
            "coordinate_space": "georeferenced",
            "crs": src.crs.to_string() if src.crs is not None else None,
            "transform": list(transform)[:6],
        }
    return metadata, transform


def write_geojson(
    data: np.ndarray,
    output_geojson: Path,
    background: int,
    metadata: dict[str, object],
    transform: Affine,
) -> int:
    if not np.issubdtype(data.dtype, np.integer):
        raise ValueError(f"expected an integer label mask, got {data.dtype}")
    if np.any(data < 0):
        raise ValueError("negative label values are not supported")

    mask = data != background
    if not np.any(mask):
        raise ValueError("no non-background labels found")

    areas = np.bincount(data[mask].ravel())
    output_geojson.parent.mkdir(parents=True, exist_ok=True)

    feature_count = 0
    with output_geojson.open("w", encoding="utf-8") as fh:
        fh.write('{"type":"FeatureCollection","metadata":')
        json.dump(metadata, fh, separators=(",", ":"))
        fh.write(',"features":[')

        first = True
        for geometry, value in shapes(data, mask=mask, transform=transform):
            label_id = int(value)
            feature = {
                "type": "Feature",
                "properties": {
                    "label_id": label_id,
                    "area_pixels": int(areas[label_id]) if label_id < len(areas) else None,
                },
                "geometry": geometry,
            }
            if not first:
                fh.write(",")
            json.dump(feature, fh, separators=(",", ":"))
            first = False
            feature_count += 1

        fh.write("]}")

    return feature_count


def main() -> None:
    args = parse_args()

    with rasterio.open(args.input_tif) as src:
        data = src.read(args.band)
        metadata, transform = dataset_metadata(src)

    feature_count = write_geojson(
        data=data,
        output_geojson=args.output_geojson,
        background=args.background,
        metadata=metadata,
        transform=transform,
    )
    print(
        f"Wrote {feature_count} features from {args.input_tif} to {args.output_geojson}"
    )


if __name__ == "__main__":
    main()
