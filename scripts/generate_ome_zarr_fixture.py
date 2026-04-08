from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from array import array
from pathlib import Path


CHANNEL_NAMES = ["DAPI", "CD3", "PanCK", "Ki67", "Collagen"]


def clamp_u16(value: float) -> int:
    value = int(round(value))
    if value < 0:
        return 0
    if value > 65535:
        return 65535
    return value


def taper(value: float, width: float) -> float:
    if width <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(value) / width)


def disc(x: float, y: float, cx: float, cy: float, radius: float) -> float:
    if radius <= 0:
        return 0.0
    dx = x - cx
    dy = y - cy
    d2 = dx * dx + dy * dy
    r2 = radius * radius
    if d2 >= r2:
        return 0.0
    return 1.0 - d2 / r2


def ring(x: float, y: float, cx: float, cy: float, radius: float, width: float) -> float:
    distance = math.hypot(x - cx, y - cy)
    return taper(distance - radius, width)


def tissue_profile(x: int, y: int, width: int, height: int) -> float:
    cx = width * 0.52
    cy = height * 0.50
    dx = (x - cx) / (width * 0.42)
    dy = (y - cy) / (height * 0.31)
    main = max(0.0, 1.0 - (dx * dx + dy * dy))

    dx2 = (x - width * 0.25) / (width * 0.18)
    dy2 = (y - height * 0.63) / (height * 0.15)
    lobe = max(0.0, 1.0 - (dx2 * dx2 + dy2 * dy2))

    notch = taper(x - width * 0.08, width * 0.12) * taper(y - height * 0.15, height * 0.18)
    tissue = min(1.0, main + 0.6 * lobe)
    return max(0.0, tissue - 0.25 * notch)


def generate_level0(width: int, height: int) -> list[array]:
    planes: list[array] = []

    for channel in range(len(CHANNEL_NAMES)):
        plane = array("H", [0]) * (width * height)

        for y in range(height):
            row_offset = y * width
            for x in range(width):
                tissue = tissue_profile(x, y, width, height)
                if tissue <= 0.0:
                    continue

                gx = ((x + 16) % 32) - 16
                gy = ((y + 16) % 32) - 16
                nuclei = max(0.0, 1.0 - (gx * gx + gy * gy) / 121.0)

                immune_band = taper(x - width * 0.28, width * 0.11)
                epithelium = taper(y - height * 0.56, height * 0.20) * taper(
                    x - width * 0.64, width * 0.34
                )
                hotspot = max(
                    disc(x, y, width * 0.67, height * 0.36, 72.0),
                    0.7 * disc(x, y, width * 0.41, height * 0.60, 58.0),
                )
                boundary = ring(
                    x, y, width * 0.52, height * 0.50, min(width, height) * 0.34, 18.0
                )
                fiber = max(
                    0.0,
                    math.sin((x + 1.4 * y) / 18.0) * 0.5 + 0.5,
                ) * taper(y - height * 0.70, height * 0.25)

                if channel == 0:
                    value = tissue * (4000.0 + nuclei * 52000.0)
                elif channel == 1:
                    value = tissue * immune_band * (6000.0 + nuclei * 28000.0)
                    value += tissue * immune_band * (
                        math.sin((x + 2.0 * y) / 21.0) * 0.5 + 0.5
                    ) * 16000.0
                elif channel == 2:
                    value = tissue * epithelium * (
                        18000.0 + (math.cos(x / 27.0) * 0.5 + 0.5) * 22000.0
                    )
                elif channel == 3:
                    value = tissue * hotspot * nuclei * 50000.0
                else:
                    value = (boundary * 34000.0 + tissue * fiber * 22000.0) * (0.6 + 0.4 * tissue)

                plane[row_offset + x] = clamp_u16(value)

        planes.append(plane)

    return planes


def downsample_plane(plane: array, width: int, height: int) -> tuple[array, int, int]:
    out_width = max(width // 2, 1)
    out_height = max(height // 2, 1)
    out = array("H", [0]) * (out_width * out_height)

    for y in range(out_height):
        top = (2 * y) * width
        bottom = top + width
        out_row = y * out_width
        for x in range(out_width):
            left = 2 * x
            value = (
                plane[top + left]
                + plane[top + left + 1]
                + plane[bottom + left]
                + plane[bottom + left + 1]
            ) // 4
            out[out_row + x] = value

    return out, out_width, out_height


def build_levels(width: int, height: int, levels: int) -> list[tuple[int, int, list[array]]]:
    current_width = width
    current_height = height
    current_planes = generate_level0(width, height)
    out = [(current_width, current_height, current_planes)]

    while len(out) < levels and current_width > 1 and current_height > 1:
        next_planes: list[array] = []
        next_width = max(current_width // 2, 1)
        next_height = max(current_height // 2, 1)
        for plane in current_planes:
            downsampled, _, _ = downsample_plane(plane, current_width, current_height)
            next_planes.append(downsampled)
        current_width = next_width
        current_height = next_height
        current_planes = next_planes
        out.append((current_width, current_height, current_planes))

    return out


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_chunk(path: Path, values: array) -> None:
    payload = array("H", values)
    if sys.byteorder != "little":
        payload.byteswap()
    path.write_bytes(payload.tobytes())


def write_level(
    root: Path,
    level_index: int,
    width: int,
    height: int,
    planes: list[array],
    chunk_size: int,
) -> None:
    level_dir = root / str(level_index)
    level_dir.mkdir(parents=True, exist_ok=True)

    y_chunk = min(chunk_size, height)
    x_chunk = min(chunk_size, width)
    write_json(
        level_dir / ".zarray",
        {
            "zarr_format": 2,
            "shape": [len(planes), height, width],
            "chunks": [1, y_chunk, x_chunk],
            "dtype": "<u2",
            "compressor": None,
            "fill_value": 0,
            "filters": None,
            "order": "C",
            "dimension_separator": ".",
        },
    )

    for c, plane in enumerate(planes):
        for y0 in range(0, height, y_chunk):
            for x0 in range(0, width, x_chunk):
                chunk = array("H")
                for y in range(y0, y0 + y_chunk):
                    row = y * width
                    chunk.extend(plane[row + x0 : row + x0 + x_chunk])
                write_chunk(level_dir / f"{c}.{y0 // y_chunk}.{x0 // x_chunk}", chunk)


def write_root_metadata(root: Path, levels: list[tuple[int, int, list[array]]]) -> None:
    datasets = []
    for level_index, _ in enumerate(levels):
        scale = float(2**level_index)
        datasets.append(
            {
                "path": str(level_index),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, scale, scale]}
                ],
            }
        )

    write_json(root / ".zgroup", {"zarr_format": 2})
    write_json(
        root / ".zattrs",
        {
            "multiscales": [
                {
                    "version": "0.4",
                    "name": "synthetic-5ch",
                    "axes": [
                        {"name": "c", "type": "channel"},
                        {"name": "y", "type": "space", "unit": "micrometer"},
                        {"name": "x", "type": "space", "unit": "micrometer"},
                    ],
                    "datasets": datasets,
                }
            ],
            "omero": {
                "channels": [
                    {"label": "DAPI", "window": {"start": 0, "end": 60000}},
                    {"label": "CD3", "window": {"start": 0, "end": 42000}},
                    {"label": "PanCK", "window": {"start": 0, "end": 42000}},
                    {"label": "Ki67", "window": {"start": 0, "end": 50000}},
                    {"label": "Collagen", "window": {"start": 0, "end": 36000}},
                ]
            },
        },
    )


def generate_fixture(output: Path, width: int, height: int, levels: int, chunk_size: int) -> None:
    pyramid = build_levels(width, height, levels)
    write_root_metadata(output, pyramid)
    for level_index, (level_width, level_height, planes) in enumerate(pyramid):
        write_level(output, level_index, level_width, level_height, planes, chunk_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="fixtures/synthetic_5ch.ome.zarr",
        help="Output OME-Zarr directory.",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"{output} already exists; pass --overwrite to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    generate_fixture(output, args.width, args.height, args.levels, args.chunk)

    print(f"wrote {output}")
    for level_index in range(args.levels):
        width = max(args.width // (2**level_index), 1)
        height = max(args.height // (2**level_index), 1)
        if width == 1 or height == 1:
            print(f"level[{level_index}] shape=(5, {height}, {width})")
            break
        print(f"level[{level_index}] shape=(5, {height}, {width})")


if __name__ == "__main__":
    main()
