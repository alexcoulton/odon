from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile


def crop_center(data: np.ndarray, size: int) -> np.ndarray:
    _, height, width = data.shape
    size = min(size, height, width)
    y0 = max((height - size) // 2, 0)
    x0 = max((width - size) // 2, 0)
    return data[:, y0 : y0 + size, x0 : x0 + size]


def downsample2(data: np.ndarray) -> np.ndarray:
    channels, height, width = data.shape
    out_h = max(height // 2, 1)
    out_w = max(width // 2, 1)
    trimmed = data[:, : out_h * 2, : out_w * 2].astype(np.uint32, copy=False)
    pooled = (
        trimmed[:, 0::2, 0::2]
        + trimmed[:, 0::2, 1::2]
        + trimmed[:, 1::2, 0::2]
        + trimmed[:, 1::2, 1::2]
    ) // 4
    return pooled.astype(np.uint16, copy=False)


def build_levels(base: np.ndarray, levels: int) -> list[np.ndarray]:
    out = [base]
    while len(out) < levels:
        prev = out[-1]
        if prev.shape[1] <= 1 or prev.shape[2] <= 1:
            break
        out.append(downsample2(prev))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="1.tif")
    parser.add_argument("--dst", default="1_pyramid_crop.ome.tif")
    parser.add_argument("--crop", type=int, default=512)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--tile", type=int, default=128)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    with tifffile.TiffFile(src) as tf:
        series = tf.series[0]
        data = series.asarray()
        axes = series.axes
        if axes != "CYX":
            raise SystemExit(f"expected CYX source axes, got {axes!r}")

    cropped = crop_center(data, args.crop)
    levels = build_levels(cropped, args.levels)
    channel_names = [f"channel {i + 1}" for i in range(levels[0].shape[0])]

    with tifffile.TiffWriter(dst, ome=True) as tif:
        tif.write(
            levels[0],
            metadata={
                "axes": "CYX",
                "Channel": {"Name": channel_names},
            },
            photometric="minisblack",
            tile=(args.tile, args.tile),
            subifds=len(levels) - 1,
        )
        for level in levels[1:]:
            tif.write(
                level,
                metadata=None,
                photometric="minisblack",
                tile=(min(args.tile, level.shape[1]), min(args.tile, level.shape[2])),
                subfiletype=1,
            )

    print(f"wrote {dst}")
    for i, level in enumerate(levels):
        print(f"level[{i}] shape={tuple(level.shape)}")


if __name__ == "__main__":
    main()
