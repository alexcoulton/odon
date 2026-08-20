# Python API

Status: experimental protocol v1 implementation

Odon remains a standalone Rust application. The separately installed,
pure-Python `odon-client` package controls a running Odon process over the Odon
Control Protocol; Odon does not embed or require Python.

This page is the guided introduction. The documentation set also includes:

- the generated [complete member reference](python-api-reference.md), containing
  every synchronous and asynchronous signature plus its control method, modes,
  task classification, and event;
- [behavioural and domain contracts](python-api-contracts.md), covering state,
  selectors, revisions, tasks, events, ownership, errors, and resource-specific
  semantics; and
- [current Python API limitations](../advanced/python-api-limitations.md),
  including the single-viewport and predefined-UI-host boundaries.

## Install

For development from this repository:

```bash
python -m pip install -e ./python
```

Install the optional array adapter with:

```bash
python -m pip install -e './python[arrays]'
```

Python 3.10 or newer is required.

## Connect, discover, or launch

Each Odon process binds a dynamic loopback port and publishes a private runtime
manifest containing an instance ID and random bearer token. With exactly one
Odon process running:

```python
import odon

with odon.connect() as app:
    print(app.hello.instance_id)
    print(app.application.get_state())
```

Use `odon.list_instances()` and `odon.connect(instance="…")` when several
instances are open. An explicit `host`, `port`, and `token` may also be supplied.

The SDK can start an installed Odon executable without distributing it:

```python
app = odon.launch("/Applications/Odon.app/Contents/MacOS/odon")
print(app.launched_process.pid)
```

`odon.launch_async(...)` is the asyncio equivalent. Closing the client closes
the control session; it does not terminate an Odon process that launched
successfully.

## Synchronous control

```python
with odon.connect() as app:
    print(app.channels.list())
    app.channels.set_visible(["DAPI", "CD3"])
    app.viewer.set_camera(center=(500, 700), zoom=0.5)
    app.viewer.set_side_panels(left=False, right=True)

    task = app.screenshots.capture("view.png")
    result = task.wait(timeout=30)
```

The resource-oriented API covers application state, projects and ROIs, viewer
and camera state, channels and contrast, objects, selection and filtering,
mosaic layout, screenshots, external data, layers, tasks, events, and native UI
extensions. `app.call(method, params)` remains available for introspection and
experimental methods.

For a vim-slime/IPython tour made of independent `# %%` cells, open
[`examples/interactive_python_api.py`](../../examples/interactive_python_api.py).

## Application surface

The high-level sync and async clients expose the same semantic resources. The
async names below have identical arguments and add `await` to network calls:

| Resource | Representative operations |
| --- | --- |
| `application` | state/readiness, settings, recent projects, diagnostics, navigation, guarded close/quit |
| `datasets` | inspect/open local OME-Zarr, TIFF, SpatialData, Xenium, HTTP and authenticated S3 sources |
| `projects` | create/open/save, metadata, samplesheets, discovery, ROI CRUD/selection/focus, saved views, object preload |
| `viewer` | camera, panels, interpolation, renderer readiness, scale bar, active right tab |
| `planes` | orientation/slice navigation and XY-only operation availability |
| `channels` | visibility, active channel, colour, notes, contrast, transforms, order, groups, search and sorting |
| `native_layers` | complete native-layer inventory, active layer, visibility, order and alignment offsets |
| `labels` | NGFF label discovery, load/unload, render state and visibility |
| `objects` | source lifecycle, paged properties, styling/legend, filter models, spatial queries, selection and focus |
| `masks` | layer/polygon CRUD, selection, undo, GeoJSON import/export and project synchronization |
| `thresholds` | levels, bounded preview configuration, refresh, polygon application and cancellation |
| `analysis` | calls/selections/mappings, histograms, threshold suggestions, presets and cache warmup |
| `measurements` | configuration, background execution, cancellation and generated property discovery |
| `object_exports` | enriched CSV/GeoParquet columns, scoped export and progress |
| `mosaic` | items, selection/focus, layout, object loading and cancellation |
| `memory` | single/mosaic RAM estimates and pin lifecycle; single-view tile worker/prefetch policy |
| `screenshots` | viewer/window/project capture, overlays, scaling, output folder and overwrite policy |
| `data`, `layers`, `ui` | external resources, extension-owned layers and declarative native UI |

The checked-in `api/application-surface.json` manifest is the authoritative
native/control/Python parity map. Odon validates it against the central command
registry at test time, including canonical methods, capabilities, events and
sync/async Python references. `application.get_application_surface()` returns
the same manifest at runtime.

Mode restrictions are discoverable rather than inferred:

```python
availability = app.application.get_method_availability([
    "viewer.labels.load",
    "memory.pin",
    "mosaic.focus.set",
])
plane_rules = app.planes.get_operation_availability()  # single-image mode
```

Calls are explicit network operations. `cached_state` only returns a deep copy
of the last state fetched through `get_state()` and never performs hidden I/O.

## Asyncio and long-running work

```python
import asyncio
import odon

async def main():
    async with odon.connect_async() as app:
        camera, channels = await asyncio.gather(
            app.viewer.get_camera(),
            app.channels.list(),
        )

        task = await app.projects.open("experiment.odon")
        result = await task
        print(camera, channels, result)

asyncio.run(main())
```

The async client registers one `asyncio.Future` per JSON-RPC request. A reader
coroutine resolves the matching future when Odon responds, so concurrent calls
can complete out of order without polling.

Operations that may take minutes return retained task handles. Starting a task
returns promptly; `task.wait()` blocks only the calling Python thread and
`await task` suspends only the calling coroutine. Odon stays responsive and
pushes progress and completion events. `wait(timeout=...)` stops waiting but
does not assume the Odon operation stopped. Cancellation is cooperative: it can
cancel queued work and discard late results, but it cannot forcibly interrupt
an operation already executing inside an indivisible library call.

RAM pinning illustrates risk confirmation and mosaic scopes:

```python
state = app.memory.get()
task = app.memory.pin(2, channels=["DAPI"], scope="all")
result = task.wait()
if result.get("confirmation_required"):
    app.memory.pin(2, channels=["DAPI"], scope="all", force=True).wait()
```

In a mosaic, `scope` is `focused`, `item`, or `all`; pass `item=` with the item
ID or sample name for item scope. In single-image mode, scope fields are ignored.

## Revisions and conflicts

Successful mutations advance a shared revision. High-level mutating methods
accept `if_revision`:

```python
state = app.application.get_state()
revision = state["_control"]["revision"]
app.viewer.set_camera(zoom=2.0, if_revision=revision)
```

Odon raises `ConflictError` if another client or native UI action changed state
first. Layers and UI contributions also expose their own revision guards for
atomic updates.

## Events

```python
app.events.subscribe("viewer.camera.*", print)
event = app.events.next(timeout=5)

for event in app.events.iter(timeout=1):
    print(event.name, event.revision, event.data)
```

Async code can use `await app.events.next()` or `async for event in
app.events.iter()`. Callbacks are isolated from connection I/O and from each
other. Queues are bounded; slow consumers can inspect `dropped_events` and
`events.status()`. Closing a client wakes blocked sync and async iterators.

## Data and layers

Large data moves by reference rather than through JSON:

```python
space = odon.CoordinateSpace(
    axes=("y", "x"),
    units=("micrometer", "micrometer"),
    scale=(0.5, 0.5),
)
resource = app.data.register(
    "file:///analysis/cellpose-labels.zarr",
    format="ome-zarr",
    coordinate_space=space,
    provenance={"software": "cellpose", "model": "cyto3"},
)
layer = app.layers.add(
    resource, name="Cellpose", kind="labels", opacity=0.7
)
layer.update(opacity=0.9, if_revision=layer.revision)
```

`CoordinateSpace.pixel_to_world()` and `world_to_pixel()` make the affine
coordinate contract explicit. Axis names are unique and scale/translation
vectors are validated.

`register_numpy()` and its async equivalent write a session-scoped temporary
Zarr store, register it by reference, and clean it when removed or when the
client closes:

```python
labels = app.data.register_numpy(
    mask,
    axes=("y", "x"),
    translation=(y0, x0),
    provenance={"software": "my-segmenter"},
)
preview = app.layers.add(labels, name="Preview", kind="labels")
```

Resource descriptors accept OME-Zarr/Zarr, Arrow IPC, Parquet, GeoParquet, and
GeoJSON references. Live rendering currently supports these local-file paths:

| Resource | Live Odon rendering |
| --- | --- |
| OME-Zarr/Zarr image or labels | Raster overlay |
| Parquet/GeoParquet shapes, objects, or points | Native spatial/object overlay |
| Arrow IPC or GeoJSON registered resource | Descriptor/lifecycle only |

Session-owned resources and layers are removed on disconnect. User-owned items
are not deleted. Project-owned resource and layer descriptors round-trip through
project JSON, but project layers may not reference a session-temporary resource.

## Declarative native UI

Python sends a validated component tree; Rust retains it and renders egui
widgets on Odon's UI thread:

```python
from odon import ui

extension = app.ui.register_extension(
    id="org.example.analysis",
    name="Analysis",
    version="0.1.0",
    capabilities=("ui.panels", "viewer.read", "viewer.layers.write"),
    disconnect_policy="disable",
)

panel = ui.Panel("analysis", title="Analysis", children=[
    ui.Select("model", "Model", options=("fast", "accurate"), value="fast"),
    ui.Slider(
        "threshold", "Threshold", minimum=0, maximum=1, value=0.5,
        event_policy=ui.Debounce(milliseconds=100),
    ),
    ui.Button("run", "Run", action=ui.emit("run-analysis")),
    ui.Progress("progress", value=0),
    ui.Status("status", "Ready"),
])
contribution = extension.register(panel, location="right.tabs")
app.events.subscribe("ui.extension:org.example.analysis.*", print)
contribution.patch_values({"progress": 0.5, "status": "Working"})
```

Rust-owned hosts are available at `right.tabs`, `left.sections`,
`top_bar.actions`, `status_bar`, and `canvas.controls`. `project.cards` currently
uses a dedicated extension window. The component vocabulary includes row,
column, grid, tabs, scroll, group/collapsible, text/Markdown/status/warning/error,
spinner/separator/spacer, buttons, toggles/checkboxes, sliders, numeric and text
inputs, select/radio/multi-select, colour, and progress controls.

Actions can:

- `emit` an event for Python;
- invoke a validated native Odon command; or
- `bind` a supported property directly to layer visibility/opacity, active or
  visible channels, camera zoom, or smooth-pixel state.

Native bindings remain responsive while Python is busy. Commit, immediate,
throttle, and debounce event policies are supported. Disconnect policies are
`remove`, `disable`, and `retain`; Odon also exposes native diagnostics and
removal controls. `odon.run_extension(factory, reconnect=True)` supplies signal
handling, cleanup, and reconnect/re-registration for packaged extensions.

## Cellpose reference extension

Install `odon-client[arrays]`, Cellpose, and the separate `odon-cellpose`
package, then run:

```bash
odon-cellpose
```

It registers a native panel, reads the active local OME-Zarr plane, runs
Cellpose in the external Python process, registers a temporary label result,
and displays it in Odon. The Cancel button is cooperative. The current reference
implementation reads the chosen extent into memory and is a proof of the full
extension path, not yet a tiled durable segmentation pipeline.

## Errors, introspection, and security

```python
try:
    app.viewer.set_camera(zoom=-1)
except odon.InvalidParametersError as error:
    print(error.kind, error.code, error.data)

print(app.application.list_methods())
print(app.application.describe_events())
print(app.ui.describe_schema())
```

Stable error kinds map to subclasses including `AuthenticationError`,
`InvalidParametersError`, `NotReadyError`, `WrongModeError`, `ConflictError`,
`ResourceNotFoundError`, `ResourceLimitError`, and `PermissionDeniedError`.

Protocol v1 is loopback-only, requires the per-instance token, limits inline
payloads and queue sizes, and never executes Python inside Odon. The API remains
experimental: method aliases, snapshot shapes, UI vocabulary, and renderer
coverage may change before a stable v1 subset is declared.

The current implementation has one native canvas and one viewer state. Mosaic
mode places several ROI items inside that canvas rather than creating
independent viewports. Python cannot currently create side-by-side viewers with
different per-view presentation. See the
[limitations page](../advanced/python-api-limitations.md) for the complete
boundary.

Semantic feature families in the parity manifest are implemented. Entries
classified as `adapter_only` or `presentation_only` cover packaging, operating-
system URL registration, CLI/MCP adapters, fixture utilities, and visual-only
chrome; they are deliberately not Python application-state commands. Packaged
cross-platform smoke tests and a stable-v1 compatibility declaration remain
release work rather than missing loopback control methods.
