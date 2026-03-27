#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_marker_names(markers_csv: Path) -> list[str]:
    with markers_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{markers_csv} is missing a header row")
        required = {"channel_number", "marker_name"}
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"{markers_csv} is missing required columns: {', '.join(missing)}"
            )

        indexed: dict[int, str] = {}
        for row in reader:
            raw_idx = (row.get("channel_number") or "").strip()
            raw_name = (row.get("marker_name") or "").strip()
            if not raw_idx:
                continue
            idx = int(raw_idx)
            if idx in indexed:
                raise ValueError(f"duplicate channel_number in {markers_csv}: {idx}")
            indexed[idx] = raw_name or f"channel {idx - 1}"

    if not indexed:
        raise ValueError(f"{markers_csv} contained no marker rows")

    max_idx = max(indexed)
    expected = list(range(1, max_idx + 1))
    seen = sorted(indexed)
    if seen != expected:
        missing = [str(i) for i in expected if i not in indexed]
        raise ValueError(
            f"{markers_csv} channel_number values are not contiguous from 1: missing {', '.join(missing)}"
        )

    return [indexed[i] for i in expected]


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def channel_count_for_dataset(root_json: dict, dataset_root: Path) -> int:
    attrs = root_json.get("attributes")
    if not isinstance(attrs, dict):
        raise ValueError(f"{dataset_root}/zarr.json is missing object-valued 'attributes'")

    ome = attrs.get("ome")
    if not isinstance(ome, dict):
        raise ValueError(f"{dataset_root}/zarr.json is missing attributes.ome")

    multiscales = ome.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        raise ValueError(f"{dataset_root}/zarr.json is missing ome.multiscales")

    axes = multiscales[0].get("axes")
    if not isinstance(axes, list):
        raise ValueError(f"{dataset_root}/zarr.json has invalid ome.multiscales[0].axes")

    c_index = None
    for i, axis in enumerate(axes):
        if isinstance(axis, dict) and axis.get("name") == "c":
            c_index = i
            break
    if c_index is None:
        raise ValueError(f"{dataset_root}/zarr.json has no channel axis")

    level0_json = read_json(dataset_root / "0" / "zarr.json")
    shape = level0_json.get("shape")
    if not isinstance(shape, list):
        raise ValueError(f"{dataset_root}/0/zarr.json has invalid shape")

    try:
        return int(shape[c_index])
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"{dataset_root}/0/zarr.json has invalid channel shape") from exc


def patch_dataset(dataset_root: Path, marker_names: list[str], backup_suffix: str) -> None:
    zarr_json = dataset_root / "zarr.json"
    root_json = read_json(zarr_json)
    channel_count = channel_count_for_dataset(root_json, dataset_root)
    if channel_count != len(marker_names):
        raise ValueError(
            f"{dataset_root} has {channel_count} channels, markers.csv has {len(marker_names)}"
        )

    backup_path = zarr_json.with_name(zarr_json.name + backup_suffix)
    if backup_path.exists():
        raise FileExistsError(f"backup already exists: {backup_path}")
    backup_path.write_text(json.dumps(root_json, indent=2) + "\n", encoding="utf-8")

    attrs = root_json["attributes"]
    omero = attrs.get("omero")
    if omero is None:
        omero = {}
        attrs["omero"] = omero
    if not isinstance(omero, dict):
        raise ValueError(f"{zarr_json} has non-object attributes.omero")

    omero["channels"] = [{"label": name} for name in marker_names]
    zarr_json.write_text(json.dumps(root_json, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch synapse OME-Zarr roots with channel names from markers.csv."
    )
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument(
        "--markers-csv",
        type=Path,
        default=None,
        help="Path to markers.csv. Defaults to <dataset_dir>/markers.csv",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".bak",
        help="Suffix for backup copies of zarr.json",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    markers_csv = args.markers_csv or (dataset_dir / "markers.csv")
    marker_names = load_marker_names(markers_csv)

    roots = sorted(
        p for p in dataset_dir.glob("*.zarr") if (p / "zarr.json").is_file()
    )
    if not roots:
        raise ValueError(f"no dataset roots found under {dataset_dir}")

    for root in roots:
        patch_dataset(root, marker_names, args.backup_suffix)

    print(
        f"patched {len(roots)} datasets in {dataset_dir} with {len(marker_names)} channel names"
    )


if __name__ == "__main__":
    main()
