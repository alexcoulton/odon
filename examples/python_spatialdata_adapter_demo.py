"""Display a real SpatialData fixture through Odon's Python-led adapter.

The fixture is written once as the original SpatialData store. The adapter then
passes Odon the store URI plus the image/labels group paths; it does not create
a second raster store.

Create and inspect the fixture without running Odon:

    uv run --project python python examples/python_spatialdata_adapter_demo.py \
        --fixture-only

Launch the repository build, display the fixture, and capture visual evidence:

    uv run --project python python examples/python_spatialdata_adapter_demo.py \
        --launch --screenshot /tmp/odon-spatialdata-adapter-demo.png
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import odon
from odon.integrations.spatialdata import SpatialDataDisplay, display


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXECUTABLE = ROOT / "target" / "debug" / "odon"
DEFAULT_FIXTURE = Path(tempfile.gettempdir()) / "odon-spatialdata-adapter-demo.zarr"


def build_synthetic_spatialdata(*, size: int = 384, seed: int = 7) -> Any:
    """Return a deterministic, unbacked SpatialData scene with all element types."""

    try:
        import anndata as ad
        import geopandas as gpd
        import numpy as np
        import pandas as pd
        from shapely.geometry import Point
        from spatialdata import SpatialData
        from spatialdata.models import (
            Image2DModel,
            Labels2DModel,
            PointsModel,
            ShapesModel,
            TableModel,
        )
        from spatialdata.transformations import Identity
    except ImportError as error:  # pragma: no cover - command-line dependency guard
        raise RuntimeError(
            "This example needs the SpatialData extra. Install it with "
            "`python -m pip install -e './python[spatialdata]'`."
        ) from error

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    image = rng.normal(80.0, 10.0, size=(3, size, size))
    # Odon accepts integer label rasters through the same generic image-layer path;
    # uint16 is ample for this fixture and is supported by the renderer.
    labels = np.zeros((size, size), dtype=np.uint16)
    rows: list[dict[str, Any]] = []
    cell_properties: list[dict[str, Any]] = []
    polygons = []
    point_rows: list[dict[str, Any]] = []

    x_positions = np.linspace(48, size - 48, 5)
    y_positions = np.linspace(48, size - 48, 4)
    for instance_id, (base_y, base_x) in enumerate(
        ((y, x) for y in y_positions for x in x_positions), start=1
    ):
        center_x = float(base_x + rng.uniform(-9, 9))
        center_y = float(base_y + rng.uniform(-9, 9))
        radius = float(rng.uniform(19, 26))
        phenotype = "marker-high" if instance_id % 3 == 0 else "marker-low"
        score = float(
            rng.uniform(0.65, 1.0)
            if phenotype == "marker-high"
            else rng.uniform(0.05, 0.5)
        )

        distance = np.hypot(xx - center_x, yy - center_y)
        nucleus = np.exp(-0.5 * (distance / (radius * 0.38)) ** 2)
        membrane = np.exp(-0.5 * ((distance - radius * 0.82) / 2.4) ** 2)
        marker = np.exp(-0.5 * (distance / (radius * 0.68)) ** 2)
        image[0] += nucleus * rng.uniform(900, 1_500)
        image[1] += membrane * rng.uniform(700, 1_200)
        image[2] += marker * (1_350 if phenotype == "marker-high" else 260)

        labels[distance <= radius * 0.72] = instance_id
        rows.append({"instance_id": instance_id})
        cell_properties.append(
            {"cell_type": phenotype, "score": score}
        )
        polygons.append(Point(center_x, center_y).buffer(radius, resolution=24))

        for transcript_index in range(12):
            angle = rng.uniform(0, 2 * np.pi)
            radial = radius * 0.75 * np.sqrt(rng.uniform())
            point_rows.append(
                {
                    "x": center_x + radial * np.cos(angle),
                    "y": center_y + radial * np.sin(angle),
                    "instance_id": f"tx-{instance_id:02d}-{transcript_index:02d}",
                    "feature": "GeneA" if transcript_index % 2 == 0 else "GeneB",
                }
            )

    rendered = np.empty_like(image, dtype=np.uint16)
    for channel_index, channel in enumerate(image):
        upper = float(np.percentile(channel, 99.8))
        rendered[channel_index] = (
            np.clip(channel / max(upper, 1.0), 0.0, 1.0) * 60_000
        ).astype(np.uint16)

    transform = {"global": Identity()}
    image_element = Image2DModel.parse(
        rendered,
        dims=("c", "y", "x"),
        c_coords=["DAPI", "Membrane", "Marker"],
        transformations=transform,
    )
    labels_element = Labels2DModel.parse(
        labels,
        dims=("y", "x"),
        transformations=transform,
    )
    shapes_element = ShapesModel.parse(
        gpd.GeoDataFrame(rows, geometry=polygons),
        transformations=transform,
    )
    points_element = PointsModel.parse(
        pd.DataFrame(point_rows),
        transformations=transform,
    )
    obs = pd.DataFrame(
        {
            "region": pd.Categorical(["cells"] * len(rows)),
            "instance_id": [row["instance_id"] for row in rows],
            "cell_type": pd.Categorical(
                [properties["cell_type"] for properties in cell_properties]
            ),
            "score": [properties["score"] for properties in cell_properties],
        },
        index=[f"cell-{index:02d}" for index in range(1, len(rows) + 1)],
    )
    table_element = TableModel.parse(
        ad.AnnData(X=np.zeros((len(rows), 1), dtype=np.float32), obs=obs),
        region="cells",
        region_key="region",
        instance_key="instance_id",
    )
    return SpatialData(
        images={"morphology": image_element},
        labels={"nuclei": labels_element},
        shapes={"cells": shapes_element},
        points={"transcripts": points_element},
        tables={"cell_table": table_element},
    )


def ensure_fixture(path: Path) -> tuple[Any, bool]:
    """Create the original store if absent, then return it as a backed object."""

    import spatialdata

    path = path.expanduser().resolve()
    if path.exists():
        return spatialdata.read_zarr(path), False

    path.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{path.name}.staging-", dir=path.parent)
    )
    staging_store = staging_root / path.name
    try:
        build_synthetic_spatialdata().write(staging_store)
        staging_store.replace(path)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return spatialdata.read_zarr(path), True


def fixture_summary(sdata: Any, path: Path, created: bool) -> dict[str, Any]:
    return {
        "fixture": str(path),
        "created": created,
        "backed": bool(sdata.is_backed()),
        "coordinate_systems": sorted(sdata.coordinate_systems),
        "elements": {
            "images": sorted(sdata.images),
            "labels": sorted(sdata.labels),
            "shapes": sorted(sdata.shapes),
            "points": sorted(sdata.points),
            "tables": sorted(sdata.tables),
        },
    }


def adaptation_summary(shown: SpatialDataDisplay, fixture: Path) -> dict[str, Any]:
    elements = {
        name: {
            "kind": element.kind,
            "transfer": element.transfer.chosen,
            "reason": element.transfer.reason,
            "source": str(element.source),
            "group_path": element.group_path,
            "source_lifetime": element.source_lifetime,
            "resource_id": element.resource.resource_id,
            "layer_id": element.layer.layer_id if element.layer is not None else None,
        }
        for name, element in shown.elements.items()
    }
    fixture = fixture.resolve()
    for name in ("morphology", "nuclei"):
        element = shown.elements[name]
        if element.transfer.chosen != "reference":
            raise RuntimeError(f"{name} unexpectedly used {element.transfer.chosen!r}")
        if Path(element.source).resolve() != fixture:
            raise RuntimeError(f"{name} unexpectedly points outside the fixture store")
    return {
        "source_uri": shown.source_uri,
        "coordinate_system": shown.coordinate_system,
        "raster_copies_created_by_adapter": 0,
        "elements": elements,
        "table_projections": {
            name: {
                "target_element": projection.target_element,
                "columns": list(projection.columns),
                "transfer": projection.transfer.chosen,
            }
            for name, projection in shown.table_projections.items()
        },
    }


def run_live(args: argparse.Namespace, sdata: Any, fixture: Path) -> None:
    if args.launch and not args.executable.is_file():
        raise FileNotFoundError(args.executable)
    app = (
        odon.launch(args.executable, timeout=30.0)
        if args.launch
        else odon.connect(
            instance=args.instance,
            client_name="python-spatialdata-adapter-demo",
        )
    )
    launched_process = app.launched_process
    shown: SpatialDataDisplay | None = None
    try:
        before = app.application.get_state()
        catalog = app.application.list_methods()
        entries = catalog.get("methods", []) if isinstance(catalog, dict) else catalog
        methods = {
            str(entry.get("name"))
            for entry in entries
            if isinstance(entry, dict) and entry.get("name") is not None
        }
        required = {"data.resources.register", "datasets.open_resource"}
        missing = sorted(required - methods)
        if missing:
            raise RuntimeError(
                f"connected Odon build is missing required methods: {missing}"
            )
        print(
            json.dumps(
                {
                    "connected_instance": app.hello.instance_id,
                    "app_version": app.hello.app_version,
                    "state_before_display": before,
                    "required_methods": sorted(required),
                },
                indent=2,
                default=str,
            )
        )

        shown = display(
            app,
            sdata,
            image="morphology",
            labels=["nuclei"],
            shapes=["cells"],
            points=["transcripts"],
            tables=["cell_table"],
            table_columns={"cell_table": ["cell_type", "score"]},
            coordinate_system="global",
            transfer="auto",
        )
        document = shown.wait_ready(timeout=120.0)
        app.viewer.fit()
        readiness = odon.recipes.wait_for_viewer_readiness(app, timeout=60.0)
        print(json.dumps(adaptation_summary(shown, fixture), indent=2))
        print(json.dumps({"document": document, "readiness": readiness}, indent=2))

        if args.screenshot is not None:
            screenshot = args.screenshot.expanduser().resolve()
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            result = app.screenshots.capture_window(
                screenshot, overwrite=True
            ).wait(timeout=60.0)
            print(json.dumps({"screenshot": str(screenshot), "result": result}, indent=2))

        if args.hold_seconds > 0:
            time.sleep(args.hold_seconds)
        elif sys.stdin.isatty() and args.screenshot is None:
            input("SpatialData is displayed in Odon. Press Enter to clean up and exit... ")
    finally:
        if shown is not None:
            shown.close()
        app.close()
        if launched_process is not None and launched_process.poll() is None:
            launched_process.terminate()
            launched_process.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--fixture-only", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--instance")
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--hold-seconds", type=float, default=0.0)
    args = parser.parse_args()
    if args.launch and args.instance is not None:
        parser.error("--launch and --instance cannot be used together")
    if args.hold_seconds < 0:
        parser.error("--hold-seconds must be non-negative")

    fixture = args.fixture.expanduser().resolve()
    sdata, created = ensure_fixture(fixture)
    print(json.dumps(fixture_summary(sdata, fixture, created), indent=2))
    if not args.fixture_only:
        run_live(args, sdata, fixture)


if __name__ == "__main__":
    main()
