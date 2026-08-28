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
  including the two-viewport milestone limit and predefined-UI-host boundaries.

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

## Multi-viewport comparisons

Single-image mode can contain one or two native Rust viewports. They share the
open dataset, raw tile cache, object geometry, edits, and object selection while
retaining independent camera, plane, channel, object, filter, overlay visibility,
layer order, and active-layer presentation:

```python
comparison = app.viewer.viewports.compare(
    layout="horizontal",
    ratio=0.55,
    titles=("Marker A", "Marker B"),
    linked=("camera", "plane", "selection"),
)
comparison.left.set_visible_channels(["DAPI", "CD3"])
comparison.right.set_visible_channels(["DAPI", "PanCK"])
comparison.left.objects.set_style(
    fill_cells=True, fill_opacity=0.65, color_property="marker_a"
)
comparison.right.objects.set_style(
    fill_cells=True, fill_opacity=0.65, color_property="marker_b"
)
comparison.left.objects.set_legend([
    {"value": "positive", "color_rgb": [255, 80, 80]}
])
comparison.right.objects.set_filter("area > 250")
comparison.left.set_rendering(smooth_pixels=False, show_hud=True)
comparison.right.set_rendering(smooth_pixels=True, show_hud=False)
```

For numeric comparisons, apply one explicit domain so equal colours have equal meaning:

```python
for viewport in (comparison.left, comparison.right):
    viewport.objects.color_by_continuous(
        "mean_channel_1",
        palette="viridis",
        domain=(4_000.0, 42_000.0),
        fill_cells=True,
        fill_opacity=0.72,
    )
```

Stable `Viewport` handles continue to address their ID when another canvas is
active. Use `app.viewer.viewports.clone(viewport, ...)` when the source view
must be explicit, and `viewport.fit_camera()` to fit a named canvas. Legacy
`app.viewer`, `app.channels`, and `app.objects` calls target the active
viewport. Configure the canonical comparison link group with
`app.viewer.viewport_links.create(viewports=[left, right], fields=[...])`,
`update(...)`, or `remove()`; `workspace.get_links()` and `set_links()` remain
compatibility conveniences. Arrange the canvases with
`set_layout("horizontal" | "vertical", ratio=...)`; capture either one
canvas with `app.screenshots.capture(..., viewport=...)` or both with
`app.screenshots.capture_workspace(...)`. Versioned project state restores a
two-view comparison, while older project files migrate to one viewport.
The planned declarative spelling
`set_layout(split="horizontal", viewports=[left, right], ratio=...)` is also
accepted and validates the current workspace order. Object presentation is
available as `viewport.objects.*`; the earlier flat
`viewport.set_object_*()` methods remain compatible aliases.

Filter-sensitive analysis, measurement, selection, and export calls must name
their provenance once two views exist: pass `viewport=...`, `filter_query=...`,
or `use_all_objects=True`. This prevents a background script from silently
using whichever canvas a person most recently clicked. Inspect resource-sharing
and timing counters with `app.viewer.workspace.get()`.

## Application surface

The high-level sync and async clients expose the same semantic resources. The
async names below have identical arguments and add `await` to network calls:

| Resource | Representative operations |
| --- | --- |
| `application` | state/readiness, settings, recent projects, diagnostics, navigation, guarded close/quit |
| `datasets` | inspect/open local OME-Zarr, TIFF, SpatialData, Xenium, HTTP and authenticated S3 sources |
| `projects` | create/open/save, metadata, samplesheets, discovery, ROI CRUD/selection/focus, saved views, object preload |
| `viewer` | camera, panels, interpolation, renderer readiness, scale bar, active right tab, linked one/two-viewport workspaces |
| `planes` | orientation/slice navigation and XY-only operation availability |
| `channels` | visibility, active channel, colour, notes, contrast, transforms, order, groups, search and sorting |
| `native_layers` | complete native-layer inventory, active layer, visibility, order and alignment offsets |
| `labels` | NGFF label discovery, load/unload, render state and visibility |
| `objects` | source lifecycle, paged properties, styling/legend, filter models, spatial queries, selection and focus |
| `annotations` | point-layer CRUD and styling, Parquet schema inspection, background loading/reloading and source clearing |
| `masks` | layer/polygon CRUD, selection, undo, GeoJSON import/export and project synchronization |
| `thresholds` | levels, bounded preview configuration, refresh, polygon application and cancellation |
| `analysis` | calls/selections/mappings, histograms, threshold suggestions, presets and cache warmup |
| `measurements` | configuration, background execution, cancellation and generated property discovery |
| `object_exports` | enriched CSV/GeoParquet columns, scoped export and progress |
| `mosaic` | items, selection/focus, layout, object loading and cancellation |
| `memory` | single/mosaic RAM estimates and pin lifecycle; tile worker/prefetch controls; mosaic image-tile byte budgets, channel history, and diagnostics |
| `screenshots` | explicit viewport/workspace/window/project capture, overlays, scaling, output folder and overwrite policy |
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

Do not call synchronous `Task.wait()` from an `app.events` callback. Completion is delivered by
that same callback worker, so the SDK raises `UnsafeCallbackWaitError` immediately—including for an
already-complete task—to make behavior independent of timing. Register resource work through
`extension.on_action(..., execution="serial-worker")`, return from the raw callback and wait on
another thread, or use the async client. Async event callbacks may await `AsyncTask` because their
awaitable bodies run as independent asyncio tasks while event dispatch continues.

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

Synchronous callbacks execute serially on the SDK callback worker. They should decode, validate,
or enqueue and return; a blocking task wait raises `UnsafeCallbackWaitError` rather than starving
later event delivery. Raw wildcard subscriptions remain available for diagnostics and advanced
correlation.

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
contribution = extension.register(panel)
app.events.subscribe("ui.extension:org.example.analysis.*", print)
contribution.patch_values({"progress": 0.5, "status": "Working"})
# Use contribution.mount("layout:analysis") in a ShellLayout desired tree.
```

Extension-scoped normalized interactions remove the difference between `.action` and `.input`
event envelopes:

```python
def inspect(interaction):
    print(
        interaction.component_id,
        interaction.action,
        interaction.value,
        interaction.kind,
        interaction.event.revision,
    )

subscription = extension.on_interaction(inspect, action="run-analysis")
subscription.remove()
```

Use the extension-owned action runner when a control starts a task or coordinates viewer state:

```python
@extension.on_action(
    "run-analysis",
    execution="serial-worker",       # documented default
    coalesce="reject-while-busy",
    contribution=contribution,
    status_component_id="status",
    progress_component_id="progress",
)
def run(context, interaction):
    with context.busy("Loading…"):
        task = context.attach(app.objects.load(path))
        result = task.wait(timeout=60, progress=context.report_task)
```

Execution policies are `callback`, `worker`, and `serial-worker`. Queue policies are `all`,
`latest`, `accumulate`, and `reject-while-busy`; none is inferred from the component type.
`extension.action_status()` provides submitted, executed, terminal, coalescing, rejection, queue,
running-action, and shutdown diagnostics. `context.ensure_current()`, generation-checked
`patch()`/`commit()`, retained-task cancellation, bounded shutdown, and error callbacks keep rapid
interactions and failures from leaving channel, source, style, and panel state out of sync. An
accumulated action already in progress completes; pending relative deltas are then applied against
its committed state. See the [safe UI action recipes](../examples/python-ui-action-recipes.md) for
complete runnable patterns and unsafe counterexamples.

Contributions default to first-class shell mounts and appear in
`app.ui.shell.list_components()`. The `right.tabs`, `left.sections`,
`top_bar.actions`, `status_bar`, `canvas.controls`, and `project.cards`
locations remain available as placement hints. Their default hosts are ordinary
`builtin:extension-host.*` components in the actor-owned tree, not separate native panels or
windows. The component vocabulary includes row,
column, grid, tabs, scroll, group/collapsible, text/Markdown/status/warning/error,
spinner/separator/spacer, buttons, toggles/checkboxes, sliders, numeric and text
inputs, select/radio/multi-select, colour, and progress controls.

`ShellMountId` provides typed stable component IDs. `CHANNELS`,
`VIEWER_VIEWPORT_CONTROLS`, `HELP`, `RECOVERY_CONTROLS`, `SHELL_INSPECTOR`, and
`COMMAND_TOOLBAR` expose independently mountable native surfaces. The inspector reports actor
revision, ownership, mutation capability, readiness, and current geometry. The catalogue remains
authoritative for modes and legal parents, and native conformance tests cover every advertised
built-in dispatcher. Reusable application layouts are available from `odon.layouts`; see
[Python application-shell workflows](../examples/python-shell-workflows.md).

Complete shell layouts can be shared or retained without manually handling the
tree mapping:

```python
document = app.ui.shell.export_layout(mode="single")
app.ui.shell.import_layout(document)
app.ui.shell.save_profile("review", scope="application")
app.ui.shell.load_profile("review", scope="application")
app.application.update_settings(
    shell_layout_startup_profiles={"single": "review"}
)
# Project scope joins the normal project dirty/save/open lifecycle.
app.ui.shell.save_profile("team-review", scope="project")
# Explicit protected fallback after an incompatible external document:
app.ui.shell.recover()
```

Normal application profiles live in the current OS user's Odon settings file. Restart tests and
isolated automation can direct one launched process to a disposable file without changing that
normal profile:

```python
app = odon.launch(
    executable,
    env={"ODON_SETTINGS_PATH": str(temporary_settings_file)},
)
```

An empty `ODON_SETTINGS_PATH` is rejected. Omitting it preserves the ordinary platform settings
location.

An extension can publish one or more owned defaults for users or workflows to
apply explicitly:

```python
document = app.ui.shell.export_layout(mode="single")
template = extension.register_layout("Analysis review", document)
template = extension.list_layouts()[0]
template.apply(app.ui.shell)
extension.remove_layout("Analysis review")
```

Odon validates extension-mount ownership and normalizes each registered
template to layout-document v1. Templates are retained across reconnect only
for `disable` or `retain` extensions and never replace the current layout until
an import is requested.

The platform application menu is a separate command presentation rather than an egui shell node:

```python
commands = {command.id: command for command in app.ui.commands.list()}
current = app.ui.menus.get()
menu = ui.CommandMenuNode.menu_bar(
    current.menu.id,
    reversed(current.menu.children),
)
current = app.ui.menus.replace(menu, if_revision=current.revision)
```

The real macOS menu is rebuilt after the actor commit. Nested menu builders and separators are
supported; close, quit, and recovery presentations are protected. Python can also replace the
bounded mounted-toolbar presentation and configure the native searchable palette through
`app.ui.toolbars` and `app.ui.palette`. Toolbar items support label/icon/tooltip overrides and
checked presentation. Extension commands can use `ui.CommandPredicates` and
`ui.CommandPredicate` to declare actor-evaluated visibility, enabled, and checked conditions over a
bounded capability/resource/selection/presentation context. User-remappable shortcut bindings
remain future command-surface work, but declared shortcuts are live: macOS realizes menu
accelerators and Windows/Linux match every eligible descriptor from the latest actor projection.
`primary` maps to Command or Ctrl, effective aliases are conflict checked, and the schema reports
platform support plus diagnostics for non-realizable modifiers. Extensions that declared
`ui.actions` can call
`extension.register_command(...)`; Odon assigns a namespaced ID, checks overlapping-mode shortcut
conflicts, follows extension readiness/disconnect policy, and publishes the declared extension
event when the command is invoked from Python or the native menu.

Actions can:

- `emit` an event for Python;
- invoke a validated native Odon command; or
- `bind` a supported property directly to layer visibility/opacity, active or
  visible channels, camera zoom, or smooth-pixel state.

Component visibility and enablement can reuse the actor-evaluated command model:

```python
run = ui.Button("run", "Run", action=ui.emit("run")).when(
    visible=ui.command_state("extension:org.example.review/run", state="visible"),
    enabled=ui.command_state("extension:org.example.review/run"),
)
```

These bounded bindings reconcile from the actor projection; a missing command resolves false and
no Python callback runs on the GUI thread.
`ShellLayoutNode.state_bindings={"visible": ui.command_state(...)}` uses the same projection for a
shell region and persists through layout export/import and profiles.

Native bindings remain responsive while Python is busy. Commit, immediate,
throttle, and debounce event policies are supported. Disconnect policies are
`remove`, `disable`, and `retain`; Odon also exposes native diagnostics and
removal controls. `odon.run_extension(factory, reconnect=True)` supplies signal
handling, cleanup, and reconnect/re-registration for packaged extensions.
Immediate interactions are render-cadence limited, while throttle/debounce changes coalesce to the
latest value per component. Python observes native semantic outcomes but cannot synchronously
intercept or cancel native interactions.

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

The first multi-viewport milestone intentionally permits at most two canvases
inside a single-image viewer. Mosaic remains a separate one-canvas model for
positioned ROI items. See the [limitations page](../advanced/python-api-limitations.md)
for the remaining boundaries.

Semantic feature families in the parity manifest are implemented. Entries
classified as `adapter_only` or `presentation_only` cover packaging, operating-
system URL registration, CLI/MCP adapters, fixture utilities, and visual-only
chrome; they are deliberately not Python application-state commands. Packaged
cross-platform smoke tests and a stable-v1 compatibility declaration remain
release work rather than missing loopback control methods.
