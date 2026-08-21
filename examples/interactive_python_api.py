"""Interactive tour of Odon's Python API.

This file is deliberately split into ``# %%`` cells. Open it in Neovim and
send one cell (or a few lines) at a time to IPython with vim-slime. Start
IPython from the repository with either:

    uv run --project python ipython

or, if the client is already installed in your environment:

    ipython

Odon must already be running with its control API enabled. Most cells are
read-only. Reversible UI demonstrations say so explicitly, while examples
that write files or create image layers are disabled by default.
"""

# %% Connect and choose the optional demonstrations you want to run.
from __future__ import annotations

from pathlib import Path
from pprint import pprint
from typing import Any, Callable

import odon
from odon import ui

app = odon.connect(client_name="odon-interactive-demo")

# Change either flag in IPython, then resend its corresponding cell below.
RUN_SCREENSHOT_DEMO = False
RUN_NUMPY_LAYER_DEMO = False

print(
    f"Connected to {app.hello.app_name} {app.hello.app_version} "
    f"(instance {app.hello.instance_id}, session {app.hello.session_id})"
)
print("Capabilities:", ", ".join(sorted(app.hello.capabilities)))


# %% Small helpers: unavailable mode-specific methods become clear skips.
def availability(method: str) -> dict[str, Any] | None:
    result = app.application.get_method_availability([method])
    return next(
        (entry for entry in result.get("methods", []) if entry["method"] == method),
        None,
    )


def api_call(label: str, method: str, call: Callable[[], Any]) -> Any:
    entry = availability(method)
    if entry is None:
        print(f"\n{label}: skipped (method is not in this Odon build)")
        return None
    if not entry["available"]:
        modes = ", ".join(entry.get("available_in", []))
        print(
            f"\n{label}: skipped ({entry.get('reason')}; current mode is "
            f"{entry.get('mode')}, available in: {modes})"
        )
        return None
    try:
        result = call()
    except odon.OdonError as error:
        print(f"\n{label}: Odon reported {type(error).__name__}: {error}")
        return None
    print(f"\n{label}:")
    pprint(result, sort_dicts=False, depth=6)
    return result


# %% Discover the running application and the API available in its current mode.
state = api_call("Application state", "app.get_state", app.application.get_state)
lifecycle = api_call(
    "Application lifecycle", "app.lifecycle.get", app.application.get_lifecycle
)
settings = api_call("Application settings", "app.settings.get", app.application.get_settings)

interesting_methods = [
    "viewer.camera.get",
    "viewer.channels.list",
    "viewer.planes.get",
    "viewer.objects.get_state",
    "viewer.masks.layers.list",
    "mosaic.get_state",
    "project.get",
    "ui.contributions.register",
]
print("\nSelected method availability:")
pprint(
    app.application.get_method_availability(interesting_methods),
    sort_dicts=False,
)


# %% Inspect the current viewer without changing it.
camera = api_call("Camera", "viewer.camera.get", app.viewer.get_camera)
rendering = api_call(
    "Rendering", "viewer.rendering.get_state", app.viewer.get_rendering_state
)
panels = api_call("Side panels", "viewer.panels.get", app.viewer.get_side_panels)
scale_bar = api_call("Scale bar", "viewer.scale_bar.get", app.viewer.get_scale_bar)
channels = api_call("Channels", "viewer.channels.list", app.channels.list)
visible_channels = api_call(
    "Visible channels", "viewer.channels.list_visible", app.channels.list_visible
)
plane = api_call("Current Z/T plane", "viewer.planes.get", app.planes.get)
plane_operations = api_call(
    "Plane navigation availability",
    "viewer.planes.operation_availability",
    app.planes.get_operation_availability,
)


# %% Reversible control: visibly toggle the scale bar.
scale_bar = api_call("Scale bar before toggle", "viewer.scale_bar.get", app.viewer.get_scale_bar)
if scale_bar and scale_bar.get("supported"):
    original_scale_bar = bool(scale_bar["visible"])
    api_call(
        "Scale bar toggled",
        "viewer.scale_bar.set",
        lambda: app.viewer.set_scale_bar(not original_scale_bar),
    )


# %% Send this cell after you have seen the scale-bar change.
if "original_scale_bar" in globals():
    app.viewer.set_scale_bar(original_scale_bar)
    print("Scale bar restored.")


# %% Reversible channel control: isolate one channel.
channel_items = channels.get("channels", []) if isinstance(channels, dict) else []
if channel_items:
    original_visible_channels = [
        item["name"] for item in channel_items if item.get("visible")
    ]
    original_active_channel = next(
        (item["name"] for item in channel_items if item.get("selected")),
        None,
    )
    demo_channel = channel_items[0]["name"]
    pprint(app.channels.set_visible([demo_channel], mode="only"), sort_dicts=False)
    print(f"Showing only {demo_channel!r}; send the next cell to restore the view.")


# %% Restore channel visibility and the active channel exactly as they were.
if "original_visible_channels" in globals():
    app.channels.set_visible(original_visible_channels, mode="only")
    if original_active_channel is not None:
        app.channels.set_active(original_active_channel)
    print("Channel visibility and active channel restored.")


# %% Camera commands are immediate; try these one line at a time.
# app.viewer.zoom_in()
# app.viewer.zoom_out()
# app.viewer.fit()
# app.viewer.set_camera(center=(1000, 750), zoom=0.5)


# %% Create two native viewports with linked navigation.
# Send this cell once; the first milestone intentionally allows at most two views.
comparison = app.viewer.viewports.compare(
    layout="horizontal",
    ratio=0.55,
    titles=("Property A", "Property B"),
    linked=("camera", "plane", "selection"),
)
comparison.left.id, comparison.right.id


# %% Give the two canvases independent channel presentations.
comparison.left.set_visible_channels(["DAPI", "CD3"])
comparison.right.set_visible_channels(["DAPI", "PanCK"])
comparison.left.set_channel_color("CD3", [0, 255, 255])
comparison.right.set_channel_color("PanCK", [255, 80, 180])
comparison.left.set_rendering(smooth_pixels=False, show_hud=True)
comparison.right.set_rendering(smooth_pixels=True, show_hud=False)


# %% After loading segmentation properties, fill each canvas by a different column.
# Replace the example names with columns from app.objects.list_properties().
# comparison.left.objects.set_style(
#     fill_cells=True, fill_opacity=0.65, color_property="property_a"
# )
# comparison.right.objects.set_style(
#     fill_cells=True, fill_opacity=0.65, color_property="property_b"
# )
# comparison.left.objects.set_legend(
#     [{"value": "high", "color_rgb": [255, 80, 80]}]
# )


# %% Object filters and native-overlay visibility can also differ per canvas.
# comparison.left.objects.set_filter(
#     mode="simple",
#     clauses=[{"property": "cell_type", "query": "immune"}],
# )
# comparison.right.objects.set_filter("area > 250")
left_layers = comparison.left.list_layers()
right_layers = comparison.right.list_layers()
pprint(left_layers, sort_dicts=False, depth=4)
# presentation = comparison.left.get_layer("segmentation_objects")
# comparison.right.set_layer("segmentation_objects", presentation={"visible": True})
# comparison.left.set_layer_visibility("segmentation_labels", False)
# comparison.right.set_active_layer("segmentation_objects")


# %% Linked camera changes update both; unlinking enables independent navigation.
comparison.left.set_camera(center=(256, 256), zoom=2.0)
comparison.left.fit_camera()
app.viewer.viewport_links.update(fields=("plane", "selection"))
comparison.right.set_camera(center=(380, 280), zoom=3.0)


# %% Change the arrangement, or close one view to return to the original workspace.
app.viewer.workspace.set_layout("vertical", ratio=0.6)
# comparison.right.remove()


# %% Inspect sharing and live frame-planning diagnostics.
workspace = app.viewer.workspace.get()
pprint(workspace["shared_resources"], sort_dicts=False)
pprint(workspace["performance"], sort_dicts=False)


# %% In a multi-view workspace, filter-sensitive operations name their source.
# Replace "area" with a numeric property available in this object layer.
# app.analysis.histogram("area", viewport=comparison.left)
# app.analysis.histogram("area", filter_query="area > 250")
# app.analysis.histogram("area", use_all_objects=True)


# %% Inspect layers, labels, memory controls, and screenshot configuration.
native_layers = api_call(
    "Dataset-native layers", "viewer.native_layers.list", app.native_layers.list
)
external_layers = api_call(
    "Python/external layers",
    "viewer.layers.list",
    lambda: [layer.snapshot for layer in app.layers.list()],
)
labels = api_call("Label overlay", "viewer.labels.get", app.labels.get)
memory = api_call("Memory and pinning", "memory.get", app.memory.get)
tile_loading = api_call(
    "Tile-loading policy", "memory.tiles.get", app.memory.get_tile_loading
)
screenshot_settings = api_call(
    "Screenshot settings",
    "viewer.screenshot.settings.get",
    app.screenshots.get_settings,
)


# %% Inspect whichever higher-level features are meaningful for this dataset.
project = api_call("Project", "project.get", app.projects.get)
rois = api_call("Project ROIs", "project.rois.list", app.projects.rois.list)
saved_views = api_call("Saved views", "project.views.list", app.projects.views.list)
mosaic = api_call("Mosaic", "mosaic.get_state", app.mosaic.get_state)
objects = api_call("Objects", "viewer.objects.get_state", app.objects.get_state)
properties = api_call(
    "Object properties", "viewer.objects.properties.list", app.objects.list_properties
)
masks = api_call("Mask layers", "viewer.masks.layers.list", app.masks.list_layers)
thresholds = api_call(
    "Threshold levels", "viewer.thresholds.levels.list", app.thresholds.list_levels
)
analysis = api_call("Analysis state", "viewer.analysis.get", app.analysis.get)
measurements = api_call(
    "Measurement state", "viewer.measurements.get", app.measurements.get
)


# %% Batch independent reads into one round trip.
batch_methods = [
    ("viewer.camera.get", {}),
    ("viewer.channels.list", {}),
    ("viewer.rendering.get_state", {}),
]
if all((entry := availability(method)) and entry["available"] for method, _ in batch_methods):
    print("Batch results:")
    pprint(app.batch(batch_methods), sort_dicts=False, depth=5)
else:
    print("Batch skipped because one of its viewer methods is unavailable.")


# %% Add a native Odon panel whose widgets are declared entirely in Python.
DEMO_EXTENSION_ID = "org.odon.interactive_demo"

# This makes the cell safe to resend in the same session.
if any(item["id"] == DEMO_EXTENSION_ID for item in app.ui.list_extensions()):
    app.call("ui.extensions.remove", {"extension_id": DEMO_EXTENSION_ID})


def print_ui_event(event: odon.Event) -> None:
    print(
        f"UI event {event.name!r} from {event.source!r}: ",
        event.data,
    )


app.events.subscribe(f"ui.extension:{DEMO_EXTENSION_ID}.*", print_ui_event)

extension = app.ui.register_extension(
    id=DEMO_EXTENSION_ID,
    name="Interactive Python Demo",
    version="0.1.0",
    capabilities=("ui.panels", "viewer.read", "viewer.write"),
)

demo_panel = ui.Panel(
    "python-demo",
    title="Python Demo",
    children=[
        ui.Markdown(
            "intro",
            "**This native egui panel was described by a Python script.**",
        ),
        ui.Select(
            "model",
            "Example model",
            options=("cyto3", "nuclei", "custom"),
            value="cyto3",
            action=ui.emit("model-changed"),
        ),
        ui.Slider(
            "diameter",
            "Cell diameter",
            minimum=1,
            maximum=200,
            value=30,
            action=ui.emit("diameter-changed"),
            event_policy=ui.Debounce(milliseconds=150),
        ),
        ui.Toggle(
            "smooth",
            "Smooth pixels",
            action=ui.bind("viewer", property="smooth_pixels"),
        ),
        ui.Button(
            "fit",
            "Fit image in view",
            action=ui.command("viewer.camera.fit"),
        ),
        ui.Button(
            "run",
            "Pretend to run Cellpose",
            action=ui.emit("run-cellpose"),
        ),
        ui.Progress("progress", value=0.0, label="Progress"),
        ui.Status("status", "Ready"),
    ],
)

contribution = extension.register(demo_panel, location="right.tabs")
print("Created contribution:", contribution.contribution_id)
print("Granted capabilities:", sorted(extension.granted_capabilities))
print("Interact with the new Python Demo tab in Odon and watch events here.")


# %% Python can update the existing native panel without rebuilding it.
contribution.patch_values({"progress": 0.35, "status": "Loading model..."})
# Send this later to simulate completion:
# contribution.patch_values({"progress": 1.0, "status": "Complete"})


# %% Long-running operations return immediately as task handles.
print("Known tasks:")
for known_task in app.tasks.list(include_finished=True):
    snapshot = known_task.snapshot
    print(snapshot.task_id, snapshot.state, snapshot.label, snapshot.progress)

# This example writes a PNG, so it runs only after you set the flag to True.
if RUN_SCREENSHOT_DEMO:
    screenshot_path = Path.cwd() / "odon-python-api-demo.png"
    screenshot_task = app.screenshots.capture(
        screenshot_path, viewport=comparison.left
    )
    print("Started:", screenshot_task.task_id, screenshot_task.snapshot.state)

    def show_progress(snapshot: odon.TaskSnapshot) -> None:
        print(snapshot.state, snapshot.phase, snapshot.progress)

    screenshot_result = screenshot_task.wait(timeout=30, progress=show_progress)
    print("Screenshot result:")
    pprint(screenshot_result, sort_dicts=False)

    workspace_path = Path.cwd() / "odon-python-api-comparison.png"
    workspace_task = app.screenshots.capture_workspace(workspace_path)
    pprint(workspace_task.wait(timeout=30), sort_dicts=False)
else:
    print("Set RUN_SCREENSHOT_DEMO = True and resend this cell to capture a PNG.")


# %% Saved project views can be captured from either viewport explicitly.
# app.projects.views.capture("Property A", viewport=comparison.left)
# app.projects.views.capture("Property B", viewport=comparison.right)


# %% Optional: create a synthetic NumPy label image as a live Odon layer.
# This requires the client's optional array dependencies:
#     uv sync --project python --extra arrays
if RUN_NUMPY_LAYER_DEMO:
    import numpy as np

    yy, xx = np.ogrid[:512, :512]
    synthetic_labels = np.zeros((512, 512), dtype=np.uint16)
    synthetic_labels[(xx - 170) ** 2 + (yy - 250) ** 2 < 80**2] = 1
    synthetic_labels[(xx - 350) ** 2 + (yy - 220) ** 2 < 60**2] = 2

    demo_resource = app.data.register_numpy(
        synthetic_labels,
        axes=("y", "x"),
        units=("pixel", "pixel"),
        scale=(1.0, 1.0),
        translation=(0.0, 0.0),
        provenance={"generator": "examples/interactive_python_api.py"},
    )
    demo_layer = app.layers.add(
        demo_resource,
        name="Synthetic Python labels",
        kind="labels",
        opacity=0.65,
    )
    print("Resource:", demo_resource.snapshot)
    print("Layer:", demo_layer.snapshot)
else:
    print("Set RUN_NUMPY_LAYER_DEMO = True and resend this cell to add a label layer.")


# %% Cleanup for the optional NumPy layer (safe to send even if it was not run).
if "demo_layer" in globals():
    demo_layer.remove()
    del demo_layer
if "demo_resource" in globals():
    demo_resource.remove()
    del demo_resource
print("Optional data/layer demo cleaned up.")


# %% Cleanup the panel when you are finished with it.
app.events.unsubscribe(f"ui.extension:{DEMO_EXTENSION_ID}.*")
app.events.remove_callback(print_ui_event)
extension.remove()
print("Interactive panel removed.")


# %% Close the interactive connection last. Re-run the first cell to reconnect.
# app.close()
