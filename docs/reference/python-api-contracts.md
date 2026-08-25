# Python API contracts

This page defines how the current `odon-client` API behaves. Use the
[guided Python API page](python-api.md) for examples and the generated
[member reference](python-api-reference.md) for every sync and async signature.
The central Rust command registry remains authoritative for request admission,
capabilities, mode availability, and emitted events.

## Status and compatibility

The SDK and control protocol are experimental. Canonical control methods are
provisional rather than stable-v1. Python resource names are intended to be the
durable user-facing surface, but snapshot fields, component vocabulary, and
renderer coverage can still change before a stable compatibility declaration.

The package requires Python 3.10 or newer and has no mandatory third-party
dependencies. The `arrays` extra adds NumPy and Zarr support for
`register_numpy()`.

## Root resource tree

`odon.connect()` and `odon.connect_async()` expose the same resource tree:

```text
app
├── application
├── datasets
│   └── s3
├── deep_links
├── viewer
│   ├── workspace
│   ├── viewports
│   ├── viewport_links
│   ├── channels          (also app.channels)
│   ├── planes            (also app.planes)
│   ├── native_layers     (also app.native_layers)
│   ├── layers            (also app.layers)
│   ├── objects           (also app.objects)
│   ├── annotations       (also app.annotations)
│   ├── masks             (also app.masks)
│   ├── thresholds        (also app.thresholds)
│   ├── analysis          (also app.analysis)
│   ├── measurements      (also app.measurements)
│   ├── memory            (also app.memory)
│   └── labels            (also app.labels)
├── projects
│   ├── samplesheets
│   ├── discovery
│   ├── objects
│   ├── rois
│   └── views
├── mosaic
├── screenshots
├── data
├── layers
├── tasks
├── events
└── ui
```

Resources are lightweight wrappers around one persistent authenticated
connection. They do not cache network results unless explicitly documented.
`application.cached_state` and `viewer.cached_state` are the sole general state
caches; they return deep copies of the last `get_state()` result.

## Connections and discovery

With exactly one running instance, `odon.connect()` discovers its private
runtime manifest and authenticates automatically. With several instances, pass
an `Instance` or instance ID to `instance=`. Supplying `host` and `port`
requires supplying both and cannot be combined with `instance=`.

The handshake populates `app.hello`:

| Field | Meaning |
| --- | --- |
| `protocol_version` | Negotiated JSON-RPC protocol version. |
| `app_name`, `app_version` | Connected Odon build. |
| `control_api_version` | Control surface version reported by Odon. |
| `instance_id`, `session_id` | Process and authenticated connection identities. |
| `capabilities` | Granted capability names. |
| `max_inline_payload_bytes` | Maximum encoded request size. |
| `permission_policy` | Active server permission policy. |

`Client.close()` closes only the control connection. It does not close Odon.
Use the guarded lifecycle methods when closing or quitting the application is
intentional. A client created by `odon.launch()` retains the process handle at
`launched_process`, but a successful client close still does not implicitly
terminate that process.

## Calls, JSON values, and snapshots

High-level methods return JSON-compatible values: mappings, lists, strings,
numbers, booleans, and `None`. Retained resource, layer, task, event, and
handshake values use small Python handle or dataclass types.

Snapshot mappings are detached values. Mutating them does not mutate Odon.
Make an explicit SDK call to change application state. Fields prefixed with
`_control` are control metadata rather than native domain fields; the shared
revision used for optimistic concurrency is available from application state.

`app.call(method, params)` is the low-level escape hatch. It accepts canonical
control methods and returns their unwrapped JSON-RPC `result`. Prefer the typed
resource wrappers because they normalize paths and selectors, validate common
sequence lengths, create task handles, and map remote errors.

The request timeout applies only to waiting for a JSON-RPC response. If it
expires, `RequestTimeoutError` means that Python stopped waiting; it does not
prove that Odon stopped the operation. Operations designed to run for a long
time return tasks instead.

## Background execution and readiness

Odon separates semantic work from native presentation. For actor-owned
single-image operations, local OME-Zarr opening, viewport workspace changes,
camera fitting, plane changes, complete channel presentation (including notes,
transforms, order, groups, and search), channel intensity reads, side-panel
visibility, explicit per-viewport object appearance/property/legend presentation, viewport
rendering preferences, local NGFF label discovery/loading/visibility, project create/open/save and
metadata/ROI/saved-view CRUD, and referenced-resource/external-layer commands
continue while the native window is covered or occluded. Project file and
dataset metadata I/O, samplesheet operations, and recursive project discovery
run on bounded workers; resource/layer mutations and
project persistence retain actor mailbox ordering. The actor retains one
immutable latest-value projection and the renderer consumes
that final state when frames resume. Intermediate visual states are coalesced;
semantic command results, revisions, and events are not.

Mask layer/polygon CRUD, selection, undo, GeoJSON import/export, and project synchronization use
the same no-frame path. Import/export filesystem work runs on bounded workers. Successful edits
mean the canonical mask generation has committed; successful export additionally reports
`output_ready`. Native mask commits carry an expected generation and are rejected on conflict
rather than overwriting a newer Python transaction.

Primary-object query, selection, and focus commands also complete against actor state without a
render frame. Standalone `filter_query` selection is evaluated on a bounded worker, and committed
native primary-object selection uses generation checking. Explicit active/spatial-shape targets
and screen-coordinate rectangle requests remain compatibility-routed during the spatial-layer
migration.

The active-view compatibility methods for camera, plane navigation, channel
presentation, panels, and smooth-pixel sampling are routed to the
same actor-owned viewport. `system.list_methods` reports each method's
`execution_class`, `readiness_requirements`, `completion_contract`, exact
`completion_point`, and `cancellation` policy. `system.get_diagnostics`
reports the current `actor`/`hybrid`/`legacy_ui`/`control_service` route per method plus
separate queue, model, reply, worker, projection, and presentation-wait
measurements.

`application.get_loading_state()` can report these readiness fields:

| Field | Meaning |
| --- | --- |
| `model_ready` | The canonical semantic state has accepted the operation. |
| `resources_ready` | Required metadata and storage resources are usable. |
| `geometry_ready` | Logical viewport geometry exists for geometry commands. |
| `geometry` | Geometry source (`bootstrap`, `derived`, or `observed`), confidence, and retained workspace size. |
| `presentation_ready` | A rendered frame has consumed the latest projection revision. |
| `projection_revision` | Latest renderer projection produced by the actor. |
| `presented_projection_revision` | Latest projection acknowledged after a UI frame. |

Dataset-open tasks complete at model/resource/geometry readiness. They do not
wait for a covered macOS window to paint. Pixel-dependent work such as window
or canvas screenshots can still require presentation readiness. Legacy fields,
including `canvas_ready`, remain in diagnostic snapshots for compatibility but
are not the completion condition for actor-owned dataset opening.

The completion contract is one of:

| Contract | What completion means |
| --- | --- |
| `immediate_semantic` | The actor has committed the semantic result. |
| `resource_ready` | Required resources or output have been installed. |
| `retained_background` | The returned task has reached a terminal state. |
| `presentation_dependent` | The required projection was rendered and output committed atomically. |

Python and MCP receive this metadata from the same central Rust registry.

An actor-owned command submitted while an asynchronous open is still in its
transition phase fails immediately with `NOT_READY` and declared readiness
requirements. It is not sent to the legacy GUI dispatcher. Await the open task
and retry; unrelated actor queries and cancellation remain responsive.

## Modes and availability

Odon has four reported modes:

| Mode | Meaning |
| --- | --- |
| `project` | Project page with no active dataset canvas. |
| `single` | One dataset viewer. |
| `mosaic` | A mosaic of project ROIs. |
| `transition` | Dataset or view transition; most mutations are temporarily unavailable. |

The generated member reference lists registry modes for every direct method.
Check the running instance when writing adaptable extensions:

```python
availability = app.application.get_method_availability([
    "viewer.camera.get",
    "viewer.objects.style.set",
    "mosaic.get_state",
])
```

Each entry reports the canonical method, current mode, `available`,
`available_in`, required capability, and a `reason` such as `wrong_mode` or
`not_ready`. Mode availability does not guarantee that optional data exists;
for example, an object method can be valid in single mode while no object
source is loaded.

## Queries, mutations, and revisions

Queries do not advance the shared application revision. Successful mutations
do. Mutating high-level methods accept `if_revision` where concurrent edits
could otherwise overwrite one another:

```python
state = app.application.get_state()
revision = state["_control"]["revision"]
app.viewer.set_camera(zoom=2.0, if_revision=revision)
```

A stale guard raises `ConflictError`; fetch fresh state and decide whether to
retry or merge. Omitting the guard requests last-writer-wins behaviour.

Layer and UI contribution handles also expose resource-local revisions.
Passing a handle revision protects that specific retained resource. Do not
assume that application revisions and resource-local revisions are
interchangeable.

## Retained tasks

Methods marked `task` return `Task` or `AsyncTask` promptly. Every task has a
`TaskSnapshot` containing:

```text
task_id, label, state, progress, phase, phase_details, result, error,
created_at_unix_ms, completed_at_unix_ms,
cancellation_supported, owner_session_id
```

States are `queued`, `running`, `completed`, `failed`, or `cancelled`.
`TaskSnapshot.done` is true for the three terminal states. Progress is either a
floating-point fraction or `None` when the operation cannot report meaningful
progress.

`phase_details` is optional structured data for the current phase. A screenshot
waiting behind a covered or minimized window uses
`waiting_for_presentation` and reports its `capture_id`,
`desired_projection_revision`, and exact renderer `resource_generations`.
During encoding it changes to `writing_output` with the target path and format.

Synchronous code can block its own thread:

```python
task = app.datasets.open_ome_zarr(path)
result = task.wait(timeout=120, progress=print)
```

Async code suspends only its coroutine:

```python
task = await app.datasets.open_ome_zarr(path)
result = await task
```

Waiting subscribes to pushed `tasks.*` events and performs one refresh to close
the registration race. It does not poll in a loop. A wait timeout leaves the
task retained and potentially running. `cancel()` is cooperative: queued work
can stop immediately, while an indivisible native library call may finish
before cancellation takes effect. `forget()` is for terminal tasks and removes
their retained server record.

Task failures raise `TaskFailedError`; cancellations raise
`TaskCancelledError`. The structured server error remains available on the
snapshot and failure exception.

## Async semantics

The async client uses one persistent stream and one future per request.
Concurrent requests may complete out of order:

```python
camera, channels = await asyncio.gather(
    app.viewer.get_camera(),
    app.channels.list(),
)
```

Network methods on async resources are coroutines even when their names match
the synchronous API. Task-starting calls therefore require one await to obtain
the task and another await to obtain its result. Async event callbacks may be
coroutines. Blocking analysis libraries such as Cellpose should run in their
own worker thread or process rather than on the asyncio event-loop thread.

## Events

Subscriptions accept exact event names or suffix wildcards such as
`viewer.camera.*`. The server emits ordered `Event` values:

| Field | Meaning |
| --- | --- |
| `name` | Semantic event name. |
| `sequence` | Monotonic event sequence. |
| `revision` | Application revision after the change. |
| `source` | Resource, component, or task source ID. |
| `data` | JSON-compatible event payload. |
| `initiating_session_id` | Session that initiated the change, when known. |
| `initiating_request_id` | Request that initiated the change, when known. |

Callbacks are dispatched away from the connection reader and isolated from
one another. A failing callback is logged and does not close the connection.
The callback and iterator queues are bounded; slow consumers can inspect
`dropped_events` and `events.status()`. Closing the client wakes blocked event
consumers.

`unsubscribe(pattern)` changes the server subscription. Remove a locally
registered callback separately with `remove_callback(callback)` if the client
will remain subscribed through another pattern.

## Batches

`app.batch()` sends an ordered sequence of `(method, params)` pairs. The return
value contains one result or error entry per operation. With `atomic=False`,
later operations can run after an earlier failure according to server batch
semantics. With `atomic=True`, Odon validates that the requested operations can
use the supported atomic path; it does not turn arbitrary long-running native
work into a database transaction. Do not put task-starting operations in a
batch when individual task handles are required.

## Common selectors and enumerations

High-level wrappers deliberately accept stable names as well as indexes where
the native domain supports them:

| Contract | Values |
| --- | --- |
| Channel selector | Channel name (`str`) or zero-based index (`int`). |
| Channel visibility mode | `only`, `show`, `hide`. |
| Channel presentation sort | `manual`, `name_asc`, `name_desc`, `visible_first`, `hidden_first`. |
| Plane mode | `xy`, `xz`, `yz`; slice is zero-based. |
| Selection mode | Usually `replace`, `add`, `remove`, or `toggle`; mosaic also supports `all` and `range`. |
| Project view selector | Saved-view name (`str`) or zero-based index (`int`). |
| Mosaic focus selector | ROI ID/name (`str`) or zero-based index (`int`). |
| Object selector | Main objects by default; mask-derived targets use the method's `target` and `layer_id` arguments. |
| Filter model mode | `simple` clauses or an advanced query. Simple `logic` is `all` or `any`. |
| Mask coordinate space | `world` by default; supported imports can also use their documented source coordinates. |
| Lifecycle save decision | `prompt`, `save`, `discard`. |
| Memory mosaic scope | `focused`, `item`, `all`; `item=` identifies the ROI for item scope. |
| Tile prefetch mode | `off`, `target_halo`, `target_and_finer_halo`. |
| Prefetch aggressiveness | `conservative`, `balanced`, `aggressive`. |
| Object preload mode | `full_geometry`, `centroid_points`. |
| Mosaic layout | `fit_cells`, `native_pixels`. |
| Xenium imagery | `auto`, `ome_zarr`, `tiff`. |
| Analysis transform | `none` plus transforms reported by the active method schema. |
| Threshold suggestion | `quantiles` or one-dimensional K-means where supported. |
| Export scope | `all`, `filtered`, `selected`. |
| Ownership | `session`, `project`, or `user` where the resource accepts ownership. |
| UI disconnect policy | `remove`, `disable`, `retain`. |

Call `application.list_methods()` for the exact request schema accepted by the
connected build. This is especially important for methods that intentionally
accept extensible `**params` dictionaries.

## Domain contracts

### Application and settings

`application.get_state()` is the broad current-state snapshot and refreshes
`cached_state`. `get_loading_state()` is narrower and intended for readiness
and progress diagnostics. Persistent settings currently cover auto-contrast
and fast-object-rendering preferences. Auto-contrast methods are
`zero_to_p97`, `p1_to_p99`, and `zero_to_max`; percentile bounds are validated
by Odon.

Recent-project mutations change only Odon's recent-list preference. They do
not delete projects from disk. Lifecycle requests require an explicit save
decision and are the only SDK methods intended to close a window or quit Odon.

### Datasets and remote sources

Dataset inspection is read-only. Open methods return tasks and settle only
after the resulting view reaches its defined readiness point. Local paths are
sent as filesystem paths; referenced external resources use URIs.

S3 credentials are session-only, redacted from responses, and removed from
Odon memory by `clear_session()` or process exit. HTTP and S3 open methods are
subject to the connected build's networking and format support.

SpatialData opening requires a selected image and can additionally select
extra images, one labels element, shape elements, and a bounded points element.
Xenium opening makes imagery and cell/transcript loading choices explicit.

### Projects, ROIs, samplesheets, and saved views

ROI IDs are stable project identities. Reordering requires the complete desired
order. ROI selection and focus are separate states. Opening an ROI or selected
mosaic returns a task because it changes application mode and loads data.

Project `save()` requires an existing project path; `save_as()` supplies one.
Samplesheet inspection and validation do not mutate the project. Import
replaces project ROI definitions only after validation. Export never overwrites
unless requested.

Saved views contain reproducible viewer presentation state. `capture()` reads
the active viewer or an explicitly selected viewport, while `create()` accepts an explicit spec.
Capture is background-safe. Applying a saved view is background-safe when its referenced object
or label resources are already ready; an apply that must load a missing resource uses the
documented hybrid compatibility route.

### Viewer, camera, planes, and channels

Camera centers use level-0 world coordinates. Zoom is screen pixels per
level-0 pixel and must be finite and greater than zero. The current API exposes
one camera because Odon currently has one viewer canvas.

Plane navigation is single-view only. Some analysis and edit operations remain
XY-only; query `get_operation_availability()` before presenting them in XZ or
YZ mode.

Channels can be selected by name or index. `set_visible(..., mode="only")`
replaces the complete visible set; `show` and `hide` modify it incrementally.
RGB colours contain exactly three integers from 0 to 255. Channel transforms
use two-component world offsets and scales plus radians for rotation.

Native-layer IDs describe Odon-owned render layers. External-layer IDs describe
resources registered through `app.layers`; these namespaces and lifecycle
rules are distinct.

In single-image mode, native-layer inventory, active identity, visibility, stack order,
presentation, and world translation are canonical actor state. The global
`viewer.native_layers.*` family addresses the active viewport (translations remain shared by the
document); `viewer.viewports.layers.*` addresses an explicit viewport. These calls complete
without a rendered frame. Native UI changes use an internal atomic viewport-state transaction
with an optimistic presentation revision, so a stale frame cannot overwrite newer Python state.

### Labels and external layers

NGFF labels are discovered from the active dataset and loaded by name. The
metadata resource is opened and committed by the background actor; creating the native tile
loader and drawing it remain asynchronous renderer work. The current native labels renderer draws
outlines rather than arbitrary per-label property fills.

External data is moved by reference. Supported descriptor formats include
OME-Zarr/Zarr, Arrow IPC, Parquet, GeoParquet, and GeoJSON, but descriptor
registration does not imply live rendering for every format. See the guided
API page for the live-rendering matrix.

Layer kinds describe how Rust should render a resource. A layer has a stable
ID, visibility, opacity, style, order, ownership, provenance, and revision.
Removing a referenced data resource fails until dependent layers are removed.

### Objects, properties, filters, and selection

The main object source supports Parquet/GeoParquet and related object data
accepted by the native loader. Loading and property materialization can be
long-running tasks. Property listing and value retrieval are paginated;
extensions must not assume every property or value is returned in one call.
Source parsing, property indexing, and per-viewport filter evaluation are background-safe actor
operations. Their successful synchronous reply means the semantic resource/filter has committed;
`presentation_ready` may remain false until a renderer frame consumes the latest projection.

Rectangles and lasso vertices are world-coordinate values. Query methods do
not change selection. Selection methods accept replace/add/remove/toggle modes,
and selection is distinct from focus. Focusing can optionally fit the camera using retained
logical viewport geometry. Primary-object selection identity is actor-owned and document-shared;
the renderer projection carries only indices and a generation, not object geometry.

Simple filter models contain typed clauses joined with `all` or `any`; advanced
filters carry a query string. Object style, legend, analysis, measurement, and
export dictionaries are validated by Rust. Use runtime method schemas and
current state snapshots for the supported fields of the connected experimental
build.

Numeric object properties can be mapped without manufacturing categorical bins. The sync and
async `objects.color_by_continuous(...)` helpers, viewport-bound equivalent, and
`mosaic.color_objects_by_continuous(...)` all emit the same `color_mapping` contract. Domains are
either `"auto"` over the full unfiltered source or an explicit `(minimum, maximum)` pair; use an
explicit shared domain when comparing sources. Named palettes are `viridis`, `magma`, `plasma`,
`inferno`, `cividis`, `turbo`, and `gray`. Linear and log10 scales, reversal, clamp/hide handling,
and a missing-value colour are actor-validated. Existing `color_property="phenotype"` calls remain
categorical and retain their per-value legend overrides.

### Point annotations

`app.annotations` manages actor-owned point layers in both single-image and mosaic modes. Layer
state includes stable numeric identity, visibility, point/stroke style, world offset, column
mapping, categorical colours/shapes, and continuous value range. `inspect()`, `load()`, and
`reload()` return retained tasks; Parquet I/O and indexing proceed on Odon's bounded worker service
without waiting for an egui frame. The renderer receives immutable generation-tagged shared data,
so a covered or minimized window catches up from the latest projection when it paints again.

### Masks and threshold regions

Mask layers have integer IDs and contain indexed polygons. Polygon vertices
must contain coordinate pairs. Layer/polygon edits participate in the native
actor-owned undo history. GeoJSON export requires explicit overwrite permission. CRUD, selection,
undo, import/export, and project synchronization are background-safe; renderer presentation is
asynchronous. `viewer.masks.state.replace` is the atomic state-transfer primitive used for native
commits and accepts `expected_generation` for optimistic concurrency.

Threshold preview is a bounded, single-view workflow: choose scope, pyramid
level, channel, threshold, and minimum component size; start or refresh the
preview; then apply it to create an editable mask layer or cancel it. Starting,
refreshing, and applying return tasks.

### Analysis, measurements, and exports

Analysis is scoped to the primary object source by default and can target
supported mask-derived objects with `target`/`layer_id`. Histograms are bounded
by a requested bin count. Threshold suggestions are advisory and do not mutate
calls until analysis state is set.

Measurement configuration specifies metric, pyramid level, concurrency,
filtered-only scope, and output-property prefix. Measurement and warmup work
return tasks. Cancellation is cooperative.

Object export is explicit about format, scope, columns, target, output path,
and overwrite. CSV omits geometry encoding beyond supported columns;
GeoParquet carries WKB geometry and spatial metadata. Exports return tasks and
write files from the Odon process.

### Mosaic

Mosaic state has one shared canvas and camera containing multiple positioned
ROI items. This is not a multi-viewport model. Selection and focus are separate;
object loading can be requested for selected items and cancelled cooperatively.
Item/property lists are paginated.

Layout configuration supports grouping, primary/secondary sorting, labels,
gaps, columns, and fitted-cell or native-pixel layout. Mosaic channels are
shared presentation state across the mosaic rather than independent per-item
channel configurations.

### Viewport workspaces

Single-image mode owns a versioned workspace with one or two stable viewport
IDs. Creating the second viewport clones presentation and navigation state but
does not reopen or duplicate the dataset, loaders, raw tile cache, object data,
edit history, or selection identities. Explicit `viewer.viewports.*` methods
target an ID; legacy `viewer.*` methods target the active viewport.

Camera and plane links propagate one causal change to the peer. Disabling a
link leaves subsequent changes independent. Selection is document-shared and
cannot be unlinked in the first milestone. Presentation changes—including
channels, object fill/property/filter, native-layer visibility/order, and
active layer—do not mutate the peer. The last viewport cannot be removed.
Existing project files load as one viewport; version-1 workspace state restores
two-view layouts, split ratio, links, stable IDs, navigation, presentation, and
per-view sampling/decorations. A stale handle continues targeting its stable ID
and raises `ResourceNotFoundError` after that viewport is removed.

Each viewport handle exposes an `objects` child resource for the planned
`viewport.objects.set_style/set_legend/set_filter` spelling. Its calls remain
ID-bound when activity changes. Flat `set_object_*` methods are retained as
compatible aliases. Workspace layout accepts either `layout="horizontal"` or
the declarative `split="horizontal", viewports=[...]` form; an explicit list
must match the current stable-ID order.

`app.viewer.viewport_links` exposes the canonical fixed link-group resource:
`list()`, `create(viewports=..., fields=...)`, `update(fields=...)`, and
`remove()`. Its ID is `comparison-navigation`. Removing it disables camera and
plane propagation but its returned snapshot still contains `selection`, since
selection identity cannot be separated from the document in this milestone.
The older `workspace.get_links()` and `workspace.set_links()` calls remain
supported compatibility wrappers.

Filter-sensitive selection, analysis, measurement, and export calls in a
two-view workspace require `viewport`, `filter_query`, `use_all_objects=True`,
or an explicit active-view opt-in. Viewport navigation responses include the
affected IDs and one link transaction ID; events retain initiating session and
request metadata. `viewer.workspace.get` exposes sharing, decoded-byte, I/O,
and frame-planning counters for runtime verification.

### Memory and screenshots

`memory.get()` reports estimates and current pin state. Pinning returns a task
and may return a confirmation requirement rather than allocate unexpectedly
large memory; repeat with `force=True` only after presenting that estimate to
the user. In mosaic mode, scopes determine which ROI items are pinned.

Tile worker count is bounded from 1 to 12. Screenshot capture returns a task
that settles when output has been written. While Odon cannot draw a frame, the
task remains observable in `waiting_for_presentation`; unrelated actor commands
continue to complete. Returning to Odon releases only the capture whose
projection revision and resource generations match. The renderer returns
pixels and the actor's bounded writer performs an atomic no-clobber commit,
removing temporary output on failure or cancellation. Explicit viewport,
composed workspace, complete-window, and project-page captures are separate operations.
File output does not overwrite unless
the relevant method explicitly accepts and receives overwrite permission.

### Deep links

Parsing and filter extraction are read-only. Resolution checks a link against
current project state without applying it. Generation canonicalizes structured
or current viewer state. Applying a link is a task because it may open projects
or datasets and change multiple state domains.

## Data ownership and cleanup

Session-owned resources, external layers, and `disconnect_policy="remove"` UI
contributions are cleaned up when their authenticated connection closes.
`register_numpy()` writes a session-scoped temporary Zarr store and the SDK
removes it when its resource is removed or the client closes.

Project-owned descriptors persist in project JSON. A project-owned layer may
not reference a session-temporary resource. User-owned resources outlive a
session only according to the specific domain contract; Odon never interprets
removing a descriptor as permission to delete an arbitrary user file.

## Declarative UI

Python registers a bounded component tree; Rust validates, retains, and renders
it on the egui thread. New contributions default to the first-class `shell`
location. Registration returns a stable `Contribution.shell_mount`, and
`Contribution.mount(node_id)` creates the corresponding keyed
`ShellLayoutNode`.

The predefined locations are placement hints backed by application-owned mounts in each default
desired tree:

```text
right.tabs
left.sections
top_bar.actions
status_bar
canvas.controls
project.cards
```

The corresponding components are `builtin:extension-host.top-bar-actions`,
`builtin:extension-host.status-bar`, `builtin:extension-host.left-sections`,
`builtin:extension-host.right-tabs`, `builtin:extension-host.canvas-controls`, and
`builtin:extension-host.project-cards`. They are catalogue-described and can be moved, hidden, or
removed like other non-protected built-ins. An empty host releases its geometry locally while its
declared actor visibility remains unchanged. A contribution explicitly mounted elsewhere in the
tree is excluded from its default host, so it is never rendered twice. No compatibility location
creates an independent egui panel, overlay, or project-card window. The complete component
constructor list is generated in the member reference.

The native shell around those hosts is separately inspectable and composable. Shell methods
return a typed `ShellSnapshot` (which also implements the mapping interface for compatibility),
and stable version-1 IDs are available through `ShellId`:

```python
from odon.ui import ShellId, ShellLayout, ShellLayoutNode, ShellSize

schema = app.ui.shell.describe_schema()  # formal draft-2020-12 JSON Schemas
shell = app.ui.shell.get()
components = app.ui.shell.list_components(mode="single")

# Hide native chrome and an extension host in one atomic update.
shell = app.ui.shell.patch(
    visibility={
        ShellId.SINGLE_TOP_BAR: False,
        ShellId.SINGLE_EXTENSION_STATUS_BAR: False,
    },
    if_revision=shell.revision,
)

# Reorder an exact tab child set and select the first tab.
shell = app.ui.shell.patch(
    orders={
        ShellId.SINGLE_LEFT_TABS: [
            ShellId.SINGLE_PROJECT,
            ShellId.SINGLE_LAYERS,
        ]
    },
    selected={ShellId.SINGLE_LEFT_TABS: ShellId.SINGLE_PROJECT},
)

app.ui.shell.reset()  # restore defaults for the active mode

# Replace the active mode's complete topology as one validated transaction.
layout = ShellLayout(
    "layout:review.root",
    (
        ShellLayoutNode.application("layout:review.root", ["layout:review.body"]),
        ShellLayoutNode.row(
            "layout:review.body", ["layout:review.canvas", "layout:review.layers"]
        ),
        ShellLayoutNode.canvas("layout:review.canvas", "builtin:viewer-canvas"),
        ShellLayoutNode.panel(
            "layout:review.layers",
            "layout:review.layers-body",
            size=ShellSize(width=320),
        ),
        ShellLayoutNode.builtin("layout:review.layers-body", "builtin:layers"),
    ),
)
shell = app.ui.shell.replace_layout(
    layout, mode="single", if_revision=shell.revision
)

# Portable interchange, named profiles, and protected recovery.
document = app.ui.shell.export_layout(mode="single")
shell = app.ui.shell.import_layout(document, if_revision=shell.revision)
app.ui.shell.save_profile("review", scope="session")
app.ui.shell.save_profile("review-default", scope="application")
app.ui.shell.save_profile("team-review", scope="project")
profiles = app.ui.shell.list_profiles(scope="application")
shell = app.ui.shell.load_profile("review-default", scope="application")
shell = app.ui.shell.recover(if_revision=shell.revision)
```

Shell nodes, desired-layout nodes, and component descriptors expose typed `ownership` metadata:
`scope`, stable `owner_id`, optional `owner_session_id`, and `protected`. Application roots and the
required project/viewer/mosaic workspace mounts are protected. A control session that owns a
registered extension can mutate its own extension nodes but receives `PermissionDeniedError` for
another extension's node. Error data identifies the node, mount, owner, required
`ui.shell.application_control` capability, and suggested resolution. An authenticated session that
is explicitly granted `ui.shell.application_control` can compose all registered mounts; native UI
retains unconditional recovery authority. Authentication by itself grants no shell mutation
authority.

Method introspection separates shell access into `ui.shell.read`, `ui.shell.compose`,
`ui.shell.extension_place`, `ui.shell.persistence`, and `ui.shell.recovery`. It also advertises the
reserved `ui.shell.chrome` and `ui.shell.window_control` boundaries. `ui.shell.shortcuts` is now
enforced for extension command registration and removal.
Extension sessions are enforced against these ownership classes inside the actor and can mutate
only their own extension-mount nodes after receiving `ui.shell.extension_place`.
`system.hello` negotiates `requested_capabilities` into `granted_capabilities`, and transport
dispatch enforces the method boundary before actor ownership checks. The standard Python client
requests the complete current controller set by default; `requested_capabilities=()` or a narrower
sequence creates a least-privilege session.

Application commands and their platform-menu presentation are separate actor-owned resources.
`app.ui.commands.list()` returns stable `ApplicationCommand` records with handler, mode
availability, protected status, icon, and shortcut metadata. `app.ui.menus.get()` returns a
revisioned `CommandMenuSnapshot`; `CommandMenuNode` builds menu bars, nested menus, command items,
and separators without duplicating command semantics:

```python
current = app.ui.menus.get()
menu = ui.CommandMenuNode.menu_bar(
    current.menu.id,
    reversed(current.menu.children),
)
updated = app.ui.menus.replace(
    menu,
    if_revision=current.revision,
    transaction_id="reorder-native-menu",
)
```

`ui.menus.replace` requires `ui.shell.chrome`, is bounded to 256 nodes and depth 12, validates all
command references, rejects duplicate node IDs, and returns the same refetch/merge/retry conflict
metadata pattern under the independent `application_command_surface` domain. Protected close,
quit, and recovery commands must remain reachable. On macOS, the actual application menu is rebuilt
from the actor projection; nested menus, separators, labels, accelerators, recovery dispatch, and
scale-bar checked state are native platform items rather than an egui imitation.

An extension that declared `ui.actions` can register an event command with
`extension.register_command(...)`. Odon assigns the canonical
`extension:<extension-id>/<local-id>` identity, rejects shortcuts already claimed in an
overlapping application mode, and requires the owning session plus `ui.shell.shortcuts`.
Conflicts use the effective desktop modifier mapping: `primary` means Command on macOS and Ctrl on
Windows/Linux. `describe_schema()` publishes neutral labels, the active mapping, supported
modifiers, and the unsupported-modifier policy. Unsupported platform modifiers return diagnostic
platform and resolution data instead of silently registering an inert shortcut.
`app.ui.commands.execute(command, checked=...)` uses the same actor-resolved dispatcher as native
menus, mounted toolbars, the command palette, and shortcuts. Extension handlers publish the declared
`ui.extension:<extension-id>.<event>` without executing Python on the GUI thread; built-in native
and control handlers are permission checked before Rust realizes their platform effect. Every
successful dispatch publishes `ui.commands.executed`. Remove-policy
disconnects remove the descriptor and every menu item that refers to it; disable/retain
disconnects preserve it as unavailable. Compatible reconnects and readiness changes update the
same descriptor, while version mismatches remain non-invokable. The native macOS menu and mounted
`builtin:command-toolbar` consume the same evaluated command state. Toolbar items can override
label, icon, tooltip, and label visibility; checked commands render as selected buttons and disabled
tooltips include actor-provided reasons. Mounted buttons publish their button role, toggled state,
and tooltip/disabled reason to the native accessibility tree; AccessKit click and keyboard
activation submit the same toggled command invocation. The actor starts with an empty toolbar
presentation and default layouts do not mount an empty command-toolbar region. A custom layout or
saved profile opts in by mounting `ShellMountId.COMMAND_TOOLBAR`.

Extension commands can declare `CommandPredicates` with independent `visible`, `enabled`, and
`checked` slots. `CommandPredicate` builds bounded `always`, `capability`, published actor `state`,
`all`, `any`, and `not_` expressions. State access is restricted to the paths advertised by
`app.ui.commands.describe_schema()`; it covers resource presence, object/mosaic selection counts,
mode, GPU readiness, panel visibility, and scale-bar state rather than arbitrary model internals.
The actor evaluates node/depth-bounded predicates for every projection and for direct execution.
`app.ui.commands.list()` evaluates capability predicates against the calling session, while native
menu/toolbar/palette projection uses native authority. `ApplicationCommand.state` exposes
`visible`, `enabled`, optional `checked`, reasons, and missing capabilities.

`app.ui.palette.get()` returns the actor-owned searchable-palette presentation. Python can replace
its title, placeholder, platform-neutral shortcut, description visibility, and bounded result count
with `app.ui.palette.replace(ui.CommandPalette.palette(...), if_revision=...)`. Replacement requires
`ui.shell.chrome`, shares the command-surface revision, publishes `ui.palette.changed`, and rejects
shortcuts claimed by a command. The native palette searches command ID, title, and description,
filters unavailable modes and non-ready extensions, and submits the selected command through
`ui.commands.execute`; it does not maintain a second registry or execute Python on the GUI thread.
On Windows and Linux, every eligible command descriptor is likewise matched from the latest actor
projection and dispatched through `ui.commands.execute`. Exact key events are consumed before
legacy widgets render; visible, enabled, and checked state comes from the actor. Because there is no
detached OS-side registry, command and palette replacements reconcile on the next projection with
no stale shortcut registrations.

Desired trees are bounded to 256 nodes and depth 32. Per-mount `configuration` is also bounded by
encoded bytes per node and per tree, nesting depth, value count, key length, and string length.
Clients should read the exact current values from `describe_schema()["layout_limits"]`; an
over-limit replacement or patch is rejected atomically without advancing the shell revision.
Boundary tests construct and validate all 256 tree nodes, then reject node 257. Command-surface
tests likewise accept all 128 toolbar items, exercise all 32 predicate nodes and depth 8, and
reconcile repeated selection-state changes before verifying the next item/node is rejected.

Stale `if_shell_revision` failures include `expected_revision`, `current_revision`,
`conflicting_domain="application_shell"`, `snapshot_method="ui.shell.get"`, and
`retry_strategy="refetch_merge_retry"` so clients can implement an explicit merge loop.
Mutating shell methods also accept an optional 1-to-128-byte `transaction_id`. The same value is
returned as `ShellChange.transaction_id` and included in `ui.shell.changed`, allowing a client to
correlate its atomic mutation without relying on timing. It is a correlation identifier, not an
atomic batch spanning several calls.

An extension contribution can be incorporated into the same transaction with
`contribution.mount("layout:review.extension")`; its `shell_mount` is also
available for clients that build layout mappings directly.

`ShellMountId` exposes the stable built-in mount vocabulary. The
`builtin:shell-inspector` component is implemented in Rust and can be mounted in a panel, tabs,
collapsible, row, column, or split. It diagnoses the actor revision, active/focused nodes,
ownership/protection, required mutation capability, readiness, validation problems, and geometry
without granting additional mutation authority. `odon.layouts` provides complete typed review,
analysis, comparison, mosaic-triage, and presentation trees; these are SDK-local builders and
still require the same server-side validation and revision guard when applied.

Extensions can also publish named default templates without mutating the active
shell implicitly:

```python
template = extension.register_layout("Review", document)
templates = extension.list_layouts()
shell = templates[0].apply(app.ui.shell, if_revision=shell.revision)
extension.remove_layout("Review")
```

An extension can register before a long warm-up and then publish its state with
`extension.set_readiness(False, reason="loading weights")` and
`extension.set_readiness(True)`. Retained contributions and templates require an exact extension
version match after reconnect. `ShellLayoutNode.readiness` distinguishes `ready`, `not_ready`,
`disconnected`, `incompatible`, and `missing`, with expected/current versions and an optional
reason; a template whose readiness is not `ready` cannot be applied by the typed Python helper.

The service validates the complete document and every referenced extension
mount, then normalizes accepted v0 or v1 input to canonical v1. Templates are
owned by the registering extension, capped at 64 per extension, and require
the granted `ui.panels` capability. Other sessions cannot inspect, replace, or
remove them. A `remove` disconnect policy deletes templates; `disable` and
`retain` preserve them for a later registration that reclaims the same
extension identity. Templates are deliberately not applied automatically:
application remains a revision-guarded `ui.shell.import_layout` transaction.

Snapshots have `schema_version`, `revision`, `mode`, `root_id`, `active_region_id`, an optional
`focused_node_id`, and a flat
`nodes` collection with stable IDs, parents, child order, selection, visibility,
content identity, and per-property mutability. They also carry `layout`, a typed
`ShellLayout` desired tree. `get(mode=...)` may inspect an inactive mode. `patch()`,
`replace_layout()`, and `reset()` mutate only the active mode. Legacy `orders`
must contain exactly the built-in container's existing direct children, once
each. `replace_layout()` instead submits a complete keyed tree and can add,
remove, or reparent layout containers atomically.

`patch_layout(active_region_id=..., focused_node_id=...)` updates interaction ownership through
the same revision-guarded actor transaction as geometry. `patch_layout(clear_focus=True)` clears
focus explicitly. Native tabs, collapsibles, mount activation, and completed split drags submit
that same command; split drag frames are local previews and unchanged final ratios are not sent.
`ui.shell.changed` reports per-node `visibility`, `order`, `selection`, `size`, `split`, `collapse`,
`active_region`, and `focus` changes. A topology change is reported as `layout`.

The application root is realized as a vertical chrome/content stack. Menu and toolbar hosts have
a default 36-point desired height, status hosts 24 points, and explicit node sizing overrides
those hints. The built-in project, viewer, and mosaic top bars are ordinary catalogue mounts, so
their order and visibility in the desired tree affect the rendered window. The former fixed native
top panels are used only if the renderer has no desired-layout projection.

Every mount node carries a `configuration` mapping. Replace-layout/import/profile operations
persist it, while `patch_layout(configurations={node_id: {...}})` changes it atomically and emits a
per-node `configuration` change. Component descriptors are authoritative for configuration JSON
Schema. The native top bars currently expose boolean `show_*` groups (title, navigation, panel,
viewport/status, rendering, and contrast as applicable); omitted flags default to `true`.

`export_layout()` returns a typed `ShellLayoutDocument` with format
`odon.shell-layout` and schema version 1. `import_layout()` validates the whole
document before mutation and currently accepts v1 plus the documented v0
`{schema_version, mode, desired_tree}` migration shape. Unsupported future
versions return `UNSUPPORTED` without changing shell state. Missing extension
mounts remain in the tree as diagnosable placeholders.

Named profiles use the same document contract. Session profiles live in the
actor until Odon exits. Application profiles are written atomically through
the settings worker under the current OS user's Odon configuration directory and are available
after restart; the `application` scope name does not mean machine-global. Project profiles live in
canonical project state, mark it dirty, and round-trip through ordinary
`project.save`/`project.open` transactions. Profile lists report invalid or
unsupported stored documents instead of hiding them. Extension defaults use
the separately owned `ui.extensions.layouts.*` template resource described
above. `recover()` replaces the
active mode with a protected two-node workspace containing its required native
workspace/canvas, so clients have an explicit safe path after an incompatible
restore.

Checked-in compatibility fixtures preserve the v0 and v1 inputs and cover corruption, unsupported
future schemas, startup restore/recovery, and missing or version-incompatible retained extension
mounts. A newer readable schema must add its migration and fixture together; Odon does not invent
or predeclare a v2 migration before v2 exists.

An application profile can be selected for automatic startup by updating the
complete per-mode selection map:

```python
app.application.update_settings(
    shell_layout_startup_profiles={"single": "review-default"}
)
```

Passing `{}` disables all automatic shell-profile restores. The setting takes
effect on the next process start. For each configured mode, the actor attempts
restore once, when that mode is first activated. A missing profile, malformed
tree, unsupported schema, or mode mismatch installs the protected recovery
layout instead. `app.application.get_settings()` reports both the configured
map and per-mode `default`, `restored`, or `recovered` outcome. Application
profile metadata includes `startup_modes`; invalid profiles include their exact
`error`, `error_kind`, and `recovery_method`.

Desired layouts support application roots, rows, columns, horizontal or vertical
splits, tabs, panels, collapsible regions, toolbar/status/menu hosts, canvas
slots, built-in mounts, and extension mounts. The actor rejects cycles, unknown
children, multiple parents, unreachable nodes, invalid child counts, duplicate
singleton mounts, illegal catalogue parent/mode combinations, trees deeper than
32 levels, trees larger than 256 nodes, and layouts without the mode's required
workspace/canvas. The recursive native renderer realizes nested row, column,
split, tab, panel, collapsible, canvas, built-in, and extension nodes. Desired,
minimum, maximum, and flex sizing constrain geometry; split handles commit
their ratio through `ui.shell.patch_layout`.

`list_components()` returns typed `ShellComponentDescriptor` records rather
than requiring clients to guess mount IDs. Each descriptor includes a stable
component/version identity, supported modes, legal parent node types, readiness,
singleton policy, configuration schema, commands, events, minimum/recommended
size, and persistence semantics. Registered extension contributions appear in
the same catalogue with their stable shell mount and connected/disconnected
readiness.
`ShellMountId.CHANNELS`, `VIEWER_VIEWPORT_CONTROLS`, `HELP`, and `RECOVERY_CONTROLS` expose native
surfaces that were previously nested inside composite chrome. They join `SHELL_INSPECTOR` and
`COMMAND_TOOLBAR` as independently placeable mounts. The server validates mode and legal parent;
the native test suite compares every mode-scoped catalogue entry against its renderer dispatcher.

No-op patches, replacements, imports, profile loads, recoveries, and resets preserve the shell revision. Mutation
results include a typed change summary with old/new shell revisions and each
changed node property; native panel/tab commands emit the same
`ui.shell.changed` compatibility event. Rust reconciles projection nodes by
stable key and retains renderer-local geometry, tab, collapse, scroll, and
split interaction state for surviving keys. Single-viewer, mosaic, and project content trees use
the recursive renderer. Native application top bars use ordinary mounts; platform windows remain
separately hosted.

Actions are one of:

- `ui.emit(...)`: emit an extension event for Python;
- `ui.command(...)`: execute a known control command natively; or
- `ui.bind(...)`: bind a supported viewer/layer property natively.

Component values can follow those bounded viewer/channel/layer targets. Visibility and enablement
can follow the already evaluated actor command state without exposing another arbitrary state path:

```python
run = ui.Button("run", "Run", action=ui.emit("run"))
run.when(
    visible=ui.command_state("extension:org.example.review/run", state="visible"),
    enabled=ui.command_state("extension:org.example.review/run"),
)
```

`state_bindings` accepts only `visible` and `enabled` properties and only `visible`, `enabled`, or
`checked` command state. Missing command IDs resolve false. Evaluation and reconciliation use the
actor projection and never execute Python on the GUI thread.

Desired shell nodes use the same descriptor for their `visible` property:

```python
help_node = ui.ShellLayoutNode.builtin(
    "layout:context.help",
    ui.ShellMountId.HELP,
    state_bindings={
        "visible": ui.command_state("extension:org.example.review/help", state="enabled")
    },
)
```

The binding round-trips through layout export/import and profiles. Its effective visibility is
derived for the active actor projection without replacing the persisted desired value.

Command and binding actions require the matching granted capability. Native
bindings remain responsive while Python is busy. Component IDs must be unique
within a contribution. Contribution patches are atomic and address components
by ID. Extension IDs use reverse-domain-style names and are unique while
connected. Contribution IDs form bounded stable mount IDs. Retained or disabled
contributions survive disconnect and reconnect under a new owning session;
remove-policy contributions leave a safe missing-component placeholder if a
retained shell tree still refers to them.

High-frequency component interactions coalesce by stable extension/component key. Immediate
events are render-cadence limited; `Throttle` and `Debounce` clamp their interval and retain only
the latest deferred value. Slow subscribers use bounded drop-on-pressure queues. Python cannot
synchronously intercept or cancel a native interaction; it can observe the semantic event and
submit a later revision-guarded command.
The native command catalogue and Rust realization arms are checked as exact sets. Native shell
tabs, collapse, focus, activation, and completed split gestures use the semantic shell patch/event
path rather than an unregistered callback channel.

The current API composes native and extension component bodies in a recursive single-window shell
and declaratively controls the command catalogue plus its macOS menu, Windows/Linux descriptor
shortcuts, mounted toolbar, native searchable-palette presentation, and contextual command state.
Arbitrary docking, detached native windows, monitor placement, and cross-window canvas transfer are
outside the scope of this API initiative. User-remappable shortcut bindings are not currently
defined.
Viewport workspace methods can instantiate a second native canvas. See
[Python API limitations](../advanced/python-api-limitations.md).

## Errors

All SDK exceptions derive from `OdonError`. Local validation can raise standard
`ValueError` before a request is sent. Remote failures map to:

| Exception | Meaning |
| --- | --- |
| `AuthenticationError` | Token or handshake authentication failed. |
| `ProtocolVersionError` | No compatible protocol version. |
| `InvalidParametersError` | Request failed schema or semantic validation. |
| `ResourceNotFoundError` | Requested stable ID, task, file, or resource is absent. |
| `NotReadyError` | Odon is transitioning or required data is not ready. |
| `UnsupportedCapabilityError` | Connected build does not implement the capability. |
| `ConflictError` | Revision guard is stale or identity conflicts. |
| `PermissionDeniedError` | Session lacks permission for the operation. |
| `WrongModeError` | Method is not valid in the current application mode. |
| `ResourceLimitError` | Bounded queue, payload, item, or memory policy was exceeded. |
| `RequestTimeoutError` | Client stopped waiting for a direct response. |
| `ConnectionClosedError` | Connection ended before completion. |

`RemoteError` retains the structured error kind, JSON-RPC code, data, method,
and request ID. Do not parse human-readable error messages to branch program
logic.

## Introspection and documentation guarantees

The running application can describe itself:

```python
app.application.list_methods()                 # methods and request schemas
app.application.describe_events()              # event envelope and families
app.application.get_method_availability()      # current-mode availability
app.application.get_application_surface()      # native/control/SDK parity
app.ui.describe_schema()                        # component vocabulary and limits
```

The generated member reference is checked against the SDK during tests. If a
public method signature, direct canonical method, mode, task classification, or
event changes, regenerating the reference is required. The hand-written guide
and contracts explain semantics; runtime introspection remains authoritative
for the connected experimental build.
