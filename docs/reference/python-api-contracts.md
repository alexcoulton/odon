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
`execution_class` and `readiness_requirements`. `system.get_diagnostics`
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
task_id, label, state, progress, phase, result, error,
created_at_unix_ms, completed_at_unix_ms,
cancellation_supported, owner_session_id
```

States are `queued`, `running`, `completed`, `failed`, or `cancelled`.
`TaskSnapshot.done` is true for the three terminal states. Progress is either a
floating-point fraction or `None` when the operation cannot report meaningful
progress.

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
that settles when output has been written. Explicit viewport, composed
workspace, complete-window, and project-page captures are separate operations.
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
it on the egui thread. Supported insertion hosts are:

```text
right.tabs
left.sections
top_bar.actions
status_bar
canvas.controls
project.cards
```

`project.cards` currently renders through a dedicated extension window. The
complete component constructor list is generated in the member reference.

Actions are one of:

- `ui.emit(...)`: emit an extension event for Python;
- `ui.command(...)`: execute a known control command natively; or
- `ui.bind(...)`: bind a supported viewer/layer property natively.

Command and binding actions require the matching granted capability. Native
bindings remain responsive while Python is busy. Component IDs must be unique
within a contribution. Contribution patches are atomic and address components
by ID. Extension IDs use reverse-domain-style names and are unique while
connected.

The current API adds UI at predefined hosts; it cannot replace the native
application shell. Viewport workspace methods can instantiate a second native
canvas, but not arbitrary shell regions. See
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
