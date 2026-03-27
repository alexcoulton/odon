#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def parse_axis_label(value: str, axis_name: str) -> tuple[str, int]:
    text = value.strip()
    match = re.fullmatch(r"[A-Za-z]*([0-9]+)", text)
    if not match:
        raise ValueError(f"could not parse {axis_name} label: {value!r}")
    return text, int(match.group(1))


def build_rows(input_csv: Path) -> list[dict[str, str]]:
    with input_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{input_csv} is missing a header row")

        required = {"dearray_id", "X", "Y"}
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"{input_csv} missing required columns: {', '.join(missing)}")

        rows: list[dict[str, str]] = []
        for raw in reader:
            dearray_id = (raw.get("dearray_id") or "").strip()
            if not dearray_id:
                continue

            x_label, x_num = parse_axis_label(raw.get("X", ""), "X")
            y_label, y_num = parse_axis_label(raw.get("Y", ""), "Y")
            path = f"{dearray_id}.zarr"
            if not (input_csv.parent / path).exists():
                raise FileNotFoundError(f"missing dataset for dearray_id={dearray_id}: {path}")

            row = {
                "id": dearray_id,
                "path": path,
                "dearray_id": dearray_id,
                "X": x_label,
                "Y": y_label,
                "layout_col": f"{x_num:03d}",
                "layout_row": f"{y_num:03d}",
                "layout_order": f"{y_num:03d}_{x_num:03d}",
            }

            for key, value in raw.items():
                if key in {"dearray_id", "X", "Y"}:
                    continue
                row[key.strip()] = (value or "").strip()

            rows.append(row)

    rows.sort(key=lambda row: (row["layout_row"], row["layout_col"], row["dearray_id"]))
    return rows


def write_rows(rows: list[dict[str, str]], output_csv: Path) -> None:
    if not rows:
        raise ValueError("no rows to write")

    fieldnames = list(rows[0].keys())
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert synapse tissues.csv into a odon mosaic samplesheet."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()

    rows = build_rows(args.input_csv)
    write_rows(rows, args.output_csv)


if __name__ == "__main__":
    main()
