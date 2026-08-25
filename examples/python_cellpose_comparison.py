# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "cellpose>=4.2,<5",
#   "packaging>=24",
#   "zarr>=2.16,<3",
# ]
# ///
"""Run Cellpose on Odon's five-channel fixture and compare the results in Odon.

The script reads PanCK and CD3 as a cytoplasmic composite and DAPI as the
nuclear channel. It runs three Cellpose-SAM parameter sets, writes labelled
arrays and GeoJSON outlines, then installs a Python-authored comparison panel
in Odon. Generated results are cached in ``test_data/cellpose_comparison``.

Inspect the workflow without Cellpose or Odon:

    uv run --project python python examples/python_cellpose_comparison.py --plan-only

Run the complete workflow in a new release build of Odon:

    uv run --script examples/python_cellpose_comparison.py --launch --serve

The first Cellpose run downloads its pretrained weights. Cellpose's pretrained
models and annotated training data have their own non-commercial licensing;
review that licensing before using the resulting workflow commercially.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = ROOT / "python" / "src"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

import odon  # noqa: E402
from odon import layouts, ui  # noqa: E402


DEFAULT_DATASET = ROOT / "fixtures" / "synthetic_5ch.ome.zarr"
DEFAULT_OUTPUT_DIR = ROOT / "test_data" / "cellpose_comparison"
DEFAULT_EXECUTABLE = ROOT / "target" / "release" / "odon"
EXTENSION_ID = "org.odon.cellpose_comparison"
CONTRIBUTION_ID = "cellpose-comparison"
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
    diameter: float | None = 30.0
    min_size: int = 15


RUN_SPECS: tuple[RunSpec, ...] = (
    RunSpec(
        id="permissive",
        title="Permissive",
        description="Keeps dimmer candidate pixels and tolerates less consistent flows.",
        cellprob_threshold=-1.0,
        flow_threshold=0.6,
    ),
    RunSpec(
        id="balanced",
        title="Balanced",
        description="Uses the standard Cellpose probability and flow thresholds.",
        cellprob_threshold=0.0,
        flow_threshold=0.4,
    ),
    RunSpec(
        id="conservative",
        title="Conservative",
        description="Requires stronger cell evidence and more consistent flows.",
        cellprob_threshold=1.0,
        flow_threshold=0.3,
    ),
)
RUN_BY_ID = {spec.id: spec for spec in RUN_SPECS}


def command_id(run_id: str) -> str:
    return f"extension:{EXTENSION_ID}/show-{run_id}"


def comparison_panel(results: Mapping[str, Mapping[str, Any]]) -> ui.Panel:
    """Build the native panel Python contributes to Odon."""

    initial = "balanced" if "balanced" in results else next(iter(results))
    return ui.Panel(
        "cellpose-comparison-panel",
        title="Cellpose comparison",
        children=[
            ui.Markdown(
                "introduction",
                "Cellpose-SAM parameter comparison\n\n"
                "Switch runs without moving the camera. The outline source and "
                "summary below update together.",
            ),
            ui.Status("status", f"Showing {RUN_BY_ID[initial].title}"),
            ui.Select(
                "selected-run",
                "Segmentation run",
                options=[spec.id for spec in RUN_SPECS if spec.id in results],
                value=initial,
                action=ui.emit("run-selected"),
                event_policy=ui.Immediate(),
            ),
            ui.Grid(
                "run-buttons",
                columns=3,
                children=[
                    ui.Button(
                        f"show-{spec.id}",
                        spec.title,
                        action=ui.command(
                            "ui.commands.execute", {"command_id": command_id(spec.id)}
                        ),
                    )
                    for spec in RUN_SPECS
                    if spec.id in results
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
            ui.Markdown("metrics", _result_markdown(results[initial])),
            ui.Markdown("comparison", _comparison_markdown(results, initial)),
        ],
    )


def comparison_layout(panel_mount: str = EXPECTED_SHELL_MOUNT) -> ui.ShellLayout:
    """Select the Cellpose panel in Odon's reusable comparison workspace."""

    base = layouts.comparison(panel_mounts=(panel_mount,))
    nodes: list[ui.ShellLayoutNode] = []
    for node in base.nodes:
        if node.id == RIGHT_TABS_ID:
            node = replace(node, selected_id=COMPARISON_NODE_ID)
        elif node.id == COMPARISON_NODE_ID:
            node = replace(node, title="Cellpose comparison")
        nodes.append(node)
    return ui.ShellLayout(base.root_id, tuple(nodes))


def plan_summary() -> dict[str, Any]:
    placeholder = {
        spec.id: {
            "id": spec.id,
            "title": spec.title,
            "cell_count": 0,
            "segmented_pixels": 0,
            "coverage_percent": 0.0,
            "median_area_px": 0.0,
            "mean_area_px": 0.0,
            "geojson_path": f"{spec.id}.geojson",
            "settings": asdict(spec),
        }
        for spec in RUN_SPECS
    }
    layout = comparison_layout()
    return {
        "dataset": DEFAULT_DATASET.name,
        "input": {"cytoplasm": ["PanCK", "CD3"], "nuclear": "DAPI"},
        "model": "cpsam_v2",
        "runs": [asdict(spec) for spec in RUN_SPECS],
        "outputs": ["labelled NumPy array", "GeoJSON outlines", "manifest.json"],
        "viewer": {
            "extension_id": EXTENSION_ID,
            "panel_mount": layout.node(COMPARISON_NODE_ID).mount,
            "selected_right_tab": layout.node(RIGHT_TABS_ID).selected_id,
            "panel": comparison_panel(placeholder).to_dict(),
        },
    }


def _channel_names(dataset: Path) -> list[str]:
    metadata = json.loads((dataset / ".zattrs").read_text())
    channels = metadata.get("omero", {}).get("channels", [])
    names = [str(channel.get("label", index)) for index, channel in enumerate(channels)]
    if not names:
        raise ValueError(f"{dataset} does not declare OME-Zarr channel metadata")
    return names


def _dataset_fingerprint(dataset: Path) -> str:
    digest = hashlib.sha256()
    files = [dataset / ".zattrs", *(dataset / "0").glob("*")]
    for path in sorted((path for path in files if path.is_file()), key=lambda p: str(p)):
        digest.update(str(path.relative_to(dataset)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_cellpose_input(
    dataset: Path,
    *,
    cytoplasm_channels: Sequence[str],
    nuclear_channel: str,
) -> tuple[Any, dict[str, Any]]:
    """Load a two-channel YXC array from level zero of an OME-Zarr image."""

    try:
        import numpy as np
        import zarr
    except ImportError as error:  # pragma: no cover - dependency guidance
        raise RuntimeError(
            "Cellpose dependencies are missing. Run this file with "
            "`uv run --script examples/python_cellpose_comparison.py ...`."
        ) from error

    names = _channel_names(dataset)
    indices = {name: index for index, name in enumerate(names)}
    missing = [name for name in (*cytoplasm_channels, nuclear_channel) if name not in indices]
    if missing:
        raise ValueError(f"unknown channel(s) {missing}; available channels: {names}")

    root = zarr.open_group(str(dataset), mode="r")
    level_zero = root["0"]
    cytoplasm_planes = [
        np.asarray(level_zero[indices[name], :, :], dtype=np.float32)
        for name in cytoplasm_channels
    ]
    cytoplasm = np.maximum.reduce(cytoplasm_planes)
    nuclei = np.asarray(level_zero[indices[nuclear_channel], :, :], dtype=np.float32)
    image = np.stack((cytoplasm, nuclei), axis=-1)
    return image, {
        "channel_names": names,
        "cytoplasm_channels": list(cytoplasm_channels),
        "nuclear_channel": nuclear_channel,
        "shape": list(image.shape),
        "dtype": str(image.dtype),
    }


def _cellpose_version() -> str:
    try:
        return importlib.metadata.version("cellpose")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _mask_metrics(mask: Any) -> dict[str, Any]:
    import numpy as np

    labels, counts = np.unique(mask, return_counts=True)
    areas = counts[labels != 0]
    segmented = int(areas.sum()) if areas.size else 0
    return {
        "cell_count": int(areas.size),
        "segmented_pixels": segmented,
        "coverage_percent": round(100.0 * segmented / int(mask.size), 3),
        "mean_area_px": round(float(areas.mean()), 2) if areas.size else 0.0,
        "median_area_px": round(float(np.median(areas)), 2) if areas.size else 0.0,
    }


def mask_to_geojson(mask: Any, spec: RunSpec, model_name: str) -> dict[str, Any]:
    """Convert Cellpose's labelled image into Odon-readable polygon outlines."""

    import numpy as np
    from cellpose import utils

    areas = np.bincount(np.asarray(mask, dtype=np.int64).ravel())
    features: list[dict[str, Any]] = []
    for label, outline in enumerate(utils.outlines_list(mask), start=1):
        if outline is None or len(outline) < 3:
            continue
        ring = [[float(point[0]), float(point[1])] for point in outline]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        features.append(
            {
                "type": "Feature",
                "id": f"{spec.id}-{label}",
                "properties": {
                    "label": label,
                    "cellpose_run": spec.id,
                    "cellpose_model": model_name,
                    "cellprob_threshold": spec.cellprob_threshold,
                    "flow_threshold": spec.flow_threshold,
                    "diameter": spec.diameter,
                    "area_px": int(areas[label]) if label < len(areas) else 0,
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
    return {
        "type": "FeatureCollection",
        "name": f"Cellpose {spec.title}",
        "odon": {"kind": "cellpose-comparison", "run": spec.id},
        "features": features,
    }


def run_segmentations(
    dataset: Path,
    output_dir: Path,
    *,
    model_name: str,
    cytoplasm_channels: Sequence[str],
    nuclear_channel: str,
    gpu: bool,
    force: bool,
) -> dict[str, Any]:
    """Execute and cache every configured Cellpose run."""

    try:
        import numpy as np
        from cellpose import dynamics, models
    except ImportError as error:  # pragma: no cover - dependency guidance
        raise RuntimeError(
            "Cellpose or one of its runtime dependencies is unavailable. Use "
            "`uv run --script examples/python_cellpose_comparison.py ...`. "
            f"Original import error: {error}"
        ) from error

    dataset = dataset.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    fingerprint = _dataset_fingerprint(dataset)
    signature = {
        "dataset": str(dataset),
        "dataset_sha256": fingerprint,
        "cellpose_version": _cellpose_version(),
        "model": model_name,
        "gpu": gpu,
        "cytoplasm_channels": list(cytoplasm_channels),
        "nuclear_channel": nuclear_channel,
        "runs": [asdict(spec) for spec in RUN_SPECS],
    }
    if not force and manifest_path.exists():
        cached = json.loads(manifest_path.read_text())
        if cached.get("signature") == signature and all(
            Path(result["geojson_path"]).exists() for result in cached.get("results", {}).values()
        ):
            print(f"Reusing Cellpose results from {output_dir}", flush=True)
            return cached

    image, input_metadata = load_cellpose_input(
        dataset,
        cytoplasm_channels=cytoplasm_channels,
        nuclear_channel=nuclear_channel,
    )
    print(
        f"Loading Cellpose model {model_name!r} on {'GPU/MPS' if gpu else 'CPU'}...",
        flush=True,
    )
    model = models.CellposeModel(gpu=gpu, pretrained_model=model_name)
    diameters = {spec.diameter for spec in RUN_SPECS}
    if len(diameters) != 1:
        raise ValueError(
            "threshold comparisons must share a diameter so they can reuse one inference"
        )
    diameter = next(iter(diameters))
    print(
        f"Inferring shared Cellpose flows at diameter={diameter}; "
        "each run will recover masks from these same flows.",
        flush=True,
    )
    inference_started = time.monotonic()
    _unused_mask, flows, _styles = model.eval(
        image,
        channel_axis=-1,
        diameter=diameter,
        compute_masks=False,
    )
    inference_seconds = round(time.monotonic() - inference_started, 3)
    dP = np.asarray(flows[1])
    cellprob = np.asarray(flows[2])
    rescale = 1.0 if diameter is None or diameter <= 0 else 30.0 / diameter
    niter = int(200 / rescale)
    print(f"Shared inference completed in {inference_seconds:.2f}s", flush=True)

    results: dict[str, dict[str, Any]] = {}
    for spec in RUN_SPECS:
        print(
            f"Recovering {spec.title}: cellprob={spec.cellprob_threshold}, "
            f"flow={spec.flow_threshold}, diameter={spec.diameter}",
            flush=True,
        )
        started = time.monotonic()
        mask = dynamics.resize_and_compute_masks(
            dP,
            cellprob,
            niter=niter,
            flow_threshold=spec.flow_threshold,
            cellprob_threshold=spec.cellprob_threshold,
            min_size=spec.min_size,
            device=model.device,
        )
        mask = np.asarray(mask, dtype=np.uint32)
        recovery_seconds = round(time.monotonic() - started, 3)
        mask_path = (output_dir / f"{spec.id}.npy").resolve()
        geojson_path = (output_dir / f"{spec.id}.geojson").resolve()
        np.save(mask_path, mask)
        geojson = mask_to_geojson(mask, spec, model_name)
        geojson_path.write_text(json.dumps(geojson, separators=(",", ":")))
        metrics = _mask_metrics(mask)
        results[spec.id] = {
            "id": spec.id,
            "title": spec.title,
            "description": spec.description,
            "settings": asdict(spec),
            "inference_seconds": inference_seconds,
            "mask_recovery_seconds": recovery_seconds,
            "mask_path": str(mask_path),
            "geojson_path": str(geojson_path),
            "outline_count": len(geojson["features"]),
            **metrics,
        }
        print(
            f"  {metrics['cell_count']} cells, {metrics['coverage_percent']:.2f}% coverage "
            f"in {recovery_seconds:.2f}s after shared inference",
            flush=True,
        )

    manifest = {
        "format": "odon-cellpose-comparison-v1",
        "signature": signature,
        "input": input_metadata,
        "shared_inference_seconds": inference_seconds,
        "results": results,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def load_manifest(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No cached Cellpose comparison exists at {manifest_path}. "
            "Run without --view-only first."
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "odon-cellpose-comparison-v1":
        raise ValueError(f"unsupported Cellpose manifest: {manifest_path}")
    return manifest


def _result_markdown(result: Mapping[str, Any]) -> str:
    settings = result["settings"]
    return (
        f"{result['title']}\n\n"
        f"{result.get('description', '')}\n\n"
        f"{result['cell_count']} cells · {result['coverage_percent']:.2f}% coverage\n"
        f"Median area: {result['median_area_px']:.1f} px² · "
        f"Mean area: {result['mean_area_px']:.1f} px²\n"
        f"Cell probability: {settings['cellprob_threshold']} · "
        f"Flow: {settings['flow_threshold']} · Diameter: {settings['diameter']}"
    )


def _comparison_markdown(results: Mapping[str, Mapping[str, Any]], selected: str) -> str:
    rows = ["All runs", ""]
    for spec in RUN_SPECS:
        if spec.id not in results:
            continue
        run_id = spec.id
        result = results[run_id]
        title = str(result["title"])
        if run_id == selected:
            title = f"→ {title}"
        rows.append(
            f"{title}: {result['cell_count']} cells · {result['coverage_percent']:.2f}%"
        )
    return "\n".join(rows)


class ComparisonController:
    def __init__(
        self,
        app: odon.Client,
        contribution: ui.Contribution,
        results: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.app = app
        self.contribution = contribution
        self.results = results
        self.selected = "balanced" if "balanced" in results else next(iter(results))
        self.errors: list[str] = []
        self._lock = threading.RLock()

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

    def show(self, run_id: str, *, wait_for_resource: bool = True) -> None:
        if run_id not in self.results:
            raise ValueError(f"unknown Cellpose run {run_id!r}")
        result = self.results[run_id]
        if int(result["outline_count"]) == 0:
            self.app.objects.clear_segmentation_geojson()
        else:
            task = self.app.objects.load_segmentation_geojson(result["geojson_path"])
            # Keep interactive callbacks responsive while the actor-owned resource
            # task completes; startup still waits before presenting the first run.
            if wait_for_resource:
                task.wait(timeout=120.0)
            self.app.objects.set_overlay_visibility(True, target="geojson")
        self.app.channels.set_visible(("DAPI", "PanCK", "CD3"), mode="only")
        self.app.channels.set_color("DAPI", (60, 95, 255))
        self.app.channels.set_color("PanCK", (255, 65, 170))
        self.app.channels.set_color("CD3", (30, 230, 225))
        self.selected = run_id
        self.contribution.patch_values(
            {
                "selected-run": run_id,
                "overlay-visible": int(result["outline_count"]) > 0,
                "status": f"Showing {result['title']} · {result['cell_count']} cells",
                "metrics": _result_markdown(result),
                "comparison": _comparison_markdown(self.results, run_id),
            }
        )

    def set_visibility(self, visible: bool) -> None:
        self.app.objects.set_overlay_visibility(bool(visible), target="geojson")
        self.contribution.patch_values(
            {
                "overlay-visible": bool(visible),
                "status": (
                    f"Showing {self.results[self.selected]['title']}"
                    if visible
                    else "Cellpose outlines hidden"
                ),
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
            semantic_event = action.get("event")
            if semantic_event == "run-selected" and str(data.get("value")) in self.results:
                self.show(str(data["value"]), wait_for_resource=False)
            elif semantic_event == "overlay-visible":
                self.set_visibility(bool(data.get("value")))
        except Exception as error:
            with self._lock:
                self.errors.append(f"{type(error).__name__}: {error}")
            try:
                self.contribution.patch_values(
                    {"status": "Cellpose comparison failed", "metrics": self.errors[-1]}
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
                client_name="cellpose-comparison",
                requested_capabilities=SESSION_CAPABILITIES,
            ),
            process,
        )
    return (
        odon.connect(
            client_name="cellpose-comparison",
            requested_capabilities=SESSION_CAPABILITIES,
        ),
        None,
    )


def view_results(
    manifest: Mapping[str, Any],
    *,
    dataset: Path,
    launch: bool,
    executable: Path,
    serve: bool,
) -> None:
    results = manifest.get("results")
    if not isinstance(results, Mapping) or not results:
        raise ValueError("Cellpose manifest contains no results")
    app, launched_process = _connect(launch=launch, executable=executable)
    original_layout: ui.ShellLayoutDocument | None = None
    extension: ui.Extension | None = None
    controller: ComparisonController | None = None
    try:
        app.datasets.open_ome_zarr(dataset).wait(timeout=120.0)
        original_layout = app.ui.shell.export_layout(mode="single")
        extension = app.ui.register_extension(
            id=EXTENSION_ID,
            name="Cellpose comparison",
            version="1.0.0",
            capabilities=("ui.panels", "ui.actions", "viewer.read", "viewer.write"),
        )
        for spec in RUN_SPECS:
            if spec.id in results:
                extension.register_command(
                    f"show-{spec.id}",
                    f"Show Cellpose: {spec.title}",
                    spec.description,
                    f"show-{spec.id}",
                    modes=("single",),
                )
        contribution = extension.register(
            comparison_panel(results),
            location="right.tabs",
            contribution_id=CONTRIBUTION_ID,
        )
        controller = ComparisonController(app, contribution, results)
        controller.subscribe()
        current = app.ui.shell.get(mode="single")
        app.ui.shell.replace_layout(
            comparison_layout(contribution.shell_mount),
            mode="single",
            if_revision=current.revision,
            transaction_id="cellpose-comparison-layout",
        )
        controller.show(controller.selected)
        app.viewer.fit()
        print(
            json.dumps(
                {
                    "dataset": str(dataset),
                    "instance_id": app.hello.instance_id,
                    "selected": controller.selected,
                    "results": results,
                    "interactive": serve,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        if serve:
            print(
                "\nCellpose comparison ready. Switch runs in the right panel. "
                "Press Ctrl+C here to restore the previous layout.",
                flush=True,
            )
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
                        transaction_id="cellpose-comparison-restore",
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
    parser.add_argument("--view-only", action="store_true")
    parser.add_argument("--segment-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Ignore cached segmentation runs.")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA or Apple MPS when available.")
    parser.add_argument("--launch", action="store_true", help="Launch a new Odon release build.")
    parser.add_argument("--serve", action="store_true", help="Keep the comparison interactive.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--model", default="cpsam_v2")
    parser.add_argument("--cytoplasm-channels", nargs="+", default=["PanCK", "CD3"])
    parser.add_argument("--nuclear-channel", default="DAPI")
    args = parser.parse_args()

    if args.plan_only:
        print(json.dumps(plan_summary(), indent=2, sort_keys=True))
        return
    if args.view_only and args.segment_only:
        parser.error("--view-only and --segment-only are mutually exclusive")
    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)
    if args.launch and not args.executable.exists():
        raise FileNotFoundError(args.executable)

    if args.view_only:
        manifest = load_manifest(args.output_dir)
    else:
        manifest = run_segmentations(
            args.dataset,
            args.output_dir,
            model_name=args.model,
            cytoplasm_channels=args.cytoplasm_channels,
            nuclear_channel=args.nuclear_channel,
            gpu=args.gpu,
            force=args.force,
        )
    if not args.segment_only:
        view_results(
            manifest,
            dataset=args.dataset,
            launch=args.launch,
            executable=args.executable,
            serve=args.serve,
        )


if __name__ == "__main__":
    main()
