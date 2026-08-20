"""A deliberately small end-to-end analysis extension."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Mapping

import odon
from odon import ui


class _Cancelled(RuntimeError):
    pass


class CellposeExtension:
    """Install a native Cellpose panel and run inference outside Odon."""

    def __init__(self, app: odon.Client) -> None:
        self.app = app
        self.extension = app.ui.register_extension(
            id="org.odon.cellpose",
            name="Cellpose Segmentation",
            version="0.1.0",
            capabilities=("ui.panels", "viewer.read", "viewer.layers.write", "data.write"),
            disconnect_policy="disable",
        )
        self.contribution = self.extension.register(
            ui.Panel(
                "cellpose-panel",
                title="Cellpose",
                children=[
                    ui.Select(
                        "model",
                        "Model",
                        options=("cyto3", "nuclei"),
                        value="cyto3",
                    ),
                    ui.Slider(
                        "diameter",
                        "Diameter",
                        minimum=1,
                        maximum=200,
                        value=30,
                    ),
                    ui.Toggle("gpu", "Use GPU", value=False),
                    ui.Select(
                        "extent",
                        "Extent",
                        options=("viewport", "whole image"),
                        value="viewport",
                    ),
                    ui.Button(
                        "run",
                        "Run Cellpose",
                        action=ui.emit("run-segmentation"),
                    ),
                    ui.Button(
                        "cancel",
                        "Cancel",
                        action=ui.emit("cancel-segmentation"),
                    ),
                    ui.Progress("progress", value=0.0),
                    ui.Status("status", "Ready"),
                ],
            ),
            location="right.tabs",
            contribution_id="org.odon.cellpose.panel",
        )
        self._values: dict[str, Any] = {
            "model": "cyto3",
            "diameter": 30.0,
            "gpu": False,
            "extent": "viewport",
        }
        self._running = False
        self._cancel_requested = threading.Event()
        self._resource: odon.DataResource | None = None
        app.events.subscribe("ui.extension:org.odon.cellpose.*", self._on_ui_event)

    def close(self) -> None:
        self._cancel_requested.set()
        self.extension.remove()

    def _on_ui_event(self, event: odon.Event) -> None:
        if not isinstance(event.data, Mapping):
            return
        component_id = event.data.get("component_id")
        if event.name.endswith(".input") and component_id in self._values:
            self._values[str(component_id)] = event.data.get("value")
            return
        action = event.data.get("action")
        if (
            event.name.endswith(".action")
            and isinstance(action, Mapping)
            and action.get("type") == "emit"
            and action.get("event") == "run-segmentation"
            and not self._running
        ):
            self._cancel_requested.clear()
            self._running = True
            threading.Thread(
                target=self._run,
                name="odon-cellpose",
                daemon=True,
            ).start()
        elif (
            event.name.endswith(".action")
            and isinstance(action, Mapping)
            and action.get("type") == "emit"
            and action.get("event") == "cancel-segmentation"
            and self._running
        ):
            self._cancel_requested.set()

    def _run(self) -> None:
        try:
            self._patch(progress=0.05, status="Reading active OME-Zarr image…")
            image, axes, scale, translation = self._read_active_plane()
            self._check_cancelled()
            self._patch(progress=0.2, status="Running Cellpose…")

            from cellpose import models

            model = models.CellposeModel(
                gpu=bool(self._values["gpu"]),
                model_type=str(self._values["model"]),
            )
            result = model.eval(image, diameter=float(self._values["diameter"]))
            masks = result[0]
            self._check_cancelled()

            self._patch(progress=0.85, status="Registering label layer…")
            resource = self.app.data.register_numpy(
                masks,
                axes=axes,
                scale=scale,
                translation=translation,
                provenance={
                    "software": "cellpose",
                    "model": self._values["model"],
                    "diameter": self._values["diameter"],
                },
            )
            previous_resource = self._resource
            try:
                self.app.layers.replace_data("layer:org.odon.cellpose.labels", resource)
            except odon.ResourceNotFoundError:
                self.app.layers.add(
                    resource,
                    name="Cellpose labels",
                    kind="labels",
                    layer_id="layer:org.odon.cellpose.labels",
                    opacity=0.65,
                    style={"rendering": "labels"},
                    provenance={"extension": "org.odon.cellpose"},
                )
            self._resource = resource
            if previous_resource is not None:
                previous_resource.remove()
            self._patch(progress=1.0, status="Cellpose labels ready")
        except _Cancelled:
            self._patch(progress=0.0, status="Cellpose cancelled")
        except Exception as error:
            self._patch(progress=0.0, status=f"Cellpose failed: {error}")
        finally:
            self._running = False

    def _check_cancelled(self) -> None:
        if self._cancel_requested.is_set():
            raise _Cancelled()

    def _patch(self, *, progress: float, status: str) -> None:
        self.contribution.patch_values({"progress": progress, "status": status})

    def _read_active_plane(
        self,
    ) -> tuple[Any, tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
        import numpy as np
        import zarr

        state = self.app.application.get_state()
        view = state.get("view") if isinstance(state, Mapping) else None
        source = view.get("dataset") if isinstance(view, Mapping) else None
        if not isinstance(source, str) or not source.startswith("local:"):
            raise RuntimeError("the reference extension currently requires a local OME-Zarr dataset")
        root = zarr.open_group(str(Path(source.removeprefix("local:"))), mode="r")
        multiscale = root.attrs["multiscales"][0]
        axes = tuple(
            axis["name"] if isinstance(axis, Mapping) else str(axis)
            for axis in multiscale["axes"]
        )
        dataset = multiscale["datasets"][0]
        array = root[dataset["path"]]
        active_channel = view.get("active_channel") if isinstance(view, Mapping) else None
        channel = int(active_channel.get("index", 0)) if isinstance(active_channel, Mapping) else 0
        selection: list[Any] = []
        output_axes: list[str] = []
        for axis in axes:
            if axis.lower() in {"y", "x"}:
                selection.append(slice(None))
                output_axes.append(axis)
            elif axis.lower() == "c":
                selection.append(channel)
            else:
                selection.append(0)
        image = np.asarray(array[tuple(selection)])
        translation = [0.0, 0.0]
        if self._values["extent"] == "viewport":
            camera = self.app.viewer.get_camera().get("camera", {})
            viewport = camera.get("viewport") if isinstance(camera, Mapping) else None
            bounds = viewport.get("visible_world_lvl0") if isinstance(viewport, Mapping) else None
            if isinstance(bounds, list) and len(bounds) == 4:
                x0 = max(0, int(bounds[0]))
                y0 = max(0, int(bounds[1]))
                x1 = min(image.shape[-1], int(bounds[2]) + 1)
                y1 = min(image.shape[-2], int(bounds[3]) + 1)
                if x1 > x0 and y1 > y0:
                    image = image[y0:y1, x0:x1]
                    translation = [float(y0), float(x0)]
        return image, tuple(output_axes), (1.0, 1.0), tuple(translation)
