# Odon Python API and Extension Platform Plan

Status: Core experimental implementation complete; stabilization work remains
Date: 2026-08-20
Target: Odon 0.x, with a stable control API designed before Odon 1.0

## Executive Summary

Odon should remain a standalone Rust application. A separately installed Python
package should connect to a running Odon process and control it through a local,
versioned protocol. Odon should not bundle or embed Python.

The architecture should have four layers:

```text
                         Python SDK and extensions
                                   |
MCP client -> odon_mcp adapter -> Odon Control Protocol <- other future clients
                                   |
                           typed Rust control core
                                   |
               Odon state, egui interface, data loading, GPU rendering
```

The Odon Control Protocol is the stable product boundary. MCP is one adapter to
that protocol; it is not the protocol used by the Python SDK. The Python API
should provide typed synchronous and asynchronous clients, state inspection,
commands, events, long-running tasks, data and layer exchange, and a declarative
UI system.

For custom UI, Python sends a serializable component tree to Odon. Rust retains
the specification and renders native egui widgets. Widgets can bind directly to
Odon state, invoke native Odon commands, or emit events to Python callbacks.
This provides extensive customization without executing Python on Odon's render
thread.

A Cellpose extension is the end-to-end reference implementation. It exercises
data discovery, custom native UI, Python-side computation, cooperative
cancellation, temporary previews, rendered label layers, and provenance.

## Implementation Snapshot

As of 2026-08-20, the first experimental implementation is present in this
repository:

- `src/control/` provides the central registry, command validation, structured
  errors, discovery/authentication, events, retained tasks, data/layer
  registries, revisions, ownership, and declarative UI registry.
- Both the published protocol server and MCP adapter admit application commands
  through that shared registry and `ControlCommand` path. MCP keeps only its
  adapter-specific tool descriptions and result formatting.
- `odon-client` is a separately buildable, pure-Python sync/async SDK with
  discovery, launch helpers, resource wrappers, callbacks/iterators, awaitable
  tasks, NumPy/Zarr exchange, optimistic revision guards, and native UI classes.
- Rust renders extension UI in bounded native hosts and can execute validated
  Odon commands and supported state bindings without a Python round trip.
- Session cleanup, disconnect policies, extension reconnection, project-owned
  resource/layer descriptor persistence, local raster/shape rendering, and a
  separately packaged Cellpose reference extension are implemented.

This is not yet a stable-v1 declaration. Snapshot shapes and method aliases are
provisional; UI manifests remain session-owned; live referenced rendering is
local-file and single-view focused; Arrow IPC and GeoJSON descriptors do not yet
have live renderer adapters; and Cellpose still uses an in-memory extent plus a
temporary preview rather than tiled inference and durable output selection.

## Goals

The platform should allow Python to:

- Inspect and control all stable viewer, project, layer, channel, camera,
  selection, annotation, layout, screenshot, and export operations.
- Subscribe to meaningful Odon state changes and user interactions.
- Run deterministic automation from scripts, notebooks, tests, and pipelines.
- Add, update, style, reorder, hide, and remove supported data layers.
- Describe custom panels, controls, menus, dialogs, and actions that Rust renders
  as native egui components.
- Connect native widgets directly to Odon state and commands without requiring a
  Python round trip for each interaction.
- Run external Python analysis packages, including Cellpose, and display their
  results in Odon.
- Support multiple independently developed extensions without requiring an Odon
  fork or recompilation.
- Preserve a responsive viewer if Python is busy, fails, or disconnects.
- Use the same typed Rust command definitions to drive the Python API, MCP
  schemas, protocol schemas, documentation, and tests.

## Non-Goals

The initial platform will not:

- Embed CPython, ship a Python runtime, or manage user Conda environments.
- Execute arbitrary Python code in the Odon process.
- Allow Python callbacks to participate in per-frame rendering.
- Transfer full-resolution images or large object tables as JSON.
- Promise that arbitrary Qt, Tk, web, or notebook widgets can be inserted into
  egui.
- Expose unstable Rust implementation details as part of the public API.
- Make remote network access available by default.
- Replace Odon's Rust-native core interface with a mandatory Python-defined UI.
- Turn MCP tool names or MCP content envelopes into the public SDK contract.

## Design Principles

### Rust owns the viewer

Rust remains authoritative for application state, data loading, coordinate
systems, egui rendering, GPU resources, and immediate user interaction. Python
requests changes and receives snapshots or events.

### The protocol is semantic, not widget-driven

Commands should express operations such as `viewer.camera.fit` and
`viewer.channels.set_visible`, rather than simulated mouse clicks or knowledge
of widget coordinates. Built-in UI and Python clients should ultimately use the
same semantic command layer where practical.

### Network effects are explicit in Python

Python method calls that perform RPC should look like method calls. Properties
may expose cached snapshots, but assigning an ordinary attribute should not
silently perform network IO.

### Local interactions stay local

A declarative slider bound to layer opacity should update Odon directly in Rust.
Python should only be involved when the workflow requires custom computation or
external coordination.

### Data moves by reference

Large raster arrays, label images, tables, shapes, and point sets should normally
be shared through OME-Zarr, Arrow IPC, Parquet, GeoParquet, or another explicit
data resource. JSON is reserved for commands, metadata, small geometry, and UI
specifications.

### Failure is contained

Python failures must not crash Odon. Extension-owned UI should remain visibly
disconnected or be removed according to policy. Native viewing must continue.

### Start narrow and make the boundary durable

The first SDK may cover the existing MCP operations, but it should be built on a
protocol with structured errors, discovery, version negotiation, and extension
points. Avoid freezing the current ad hoc bridge format as the permanent API.

## Primary Use Cases

### Reproducible viewer automation

A script opens each ROI, applies standard channels and contrast, waits for all
required data, and captures a consistent QC image.

### Notebook-linked exploration

A user selects cells or a region in Odon, receives IDs and coordinates in a
notebook, analyses them with pandas or Scanpy, and sends colours or derived
properties back to Odon.

### Analysis-driven layers

Python creates segmentations, classifications, points, shapes, measurements, or
heatmaps and registers them as Odon layers with an explicit coordinate space and
provenance.

### Workflow-specific applications

A laboratory package installs a native-looking review panel, project actions,
validation rules, and exports without modifying Odon itself.

### Model review and correction

Python runs a segmentation or classification model. Odon provides rapid visual
review and selection, and corrections flow back to Python or a durable data
store.

### Testing and integration

Automated tests launch or connect to Odon, load fixtures, assert state, exercise
commands, wait for rendering readiness, and compare screenshots.

## Implemented Foundation

Odon now contains the intended architectural seam:

- The GUI owns a dynamic authenticated loopback TCP listener in
  `src/mcp/bridge.rs`.
- Bridge requests are queued through `crossbeam_channel`.
- `RootApp` drains the queue and executes operations on the egui/UI thread.
- `odon_mcp` speaks MCP over stdio and forwards tools through the bridge.
- Shared control operations cover projects, ROIs, channels, contrast, camera,
  panels, object visibility and selection, filters, mosaics, screenshots, and
  loading state.
- `src/control/` owns method admission, request validation, revisions, errors,
  events, tasks, resources, layers, UI extensions, and discovery manifests.
- The Python SDK and MCP adapter connect independently through the same protocol
  boundary.

The implementation remains experimental because several legacy application
operations still expose provisional JSON snapshot shapes and MCP maintains
client-specific input-schema presentation. These are stabilization concerns,
not separate execution paths.

## Target Architecture

### Components

| Component | Responsibility |
| --- | --- |
| Odon control core | Typed commands, queries, results, errors, tasks, events, and capability registry. |
| Odon protocol server | Connection handling, authentication, JSON-RPC framing, version negotiation, and subscriptions. |
| Odon application adapter | Executes commands against `RootApp` and active viewer modes on the UI thread. |
| `odon_mcp` | Maps MCP tools to control methods and formats MCP results. |
| `odon-client` | Typed Python connection, resource API, events, tasks, and error classes. |
| Extension packages | Domain workflows such as `odon-cellpose`, installed in the user's Python environment. |
| Data exchange layer | URI/path resources, temporary resources, optional array adapters, and provenance. |
| Declarative UI engine | Rust-side retained UI specifications rendered through egui. |

### Process model

The normal configuration has three possible processes:

```text
odon GUI                 owns data and rendering
odon_mcp                 optional, launched by an MCP client
user Python process      optional, owns Python packages and extension logic
```

`odon_mcp` and Python connect independently to the same Odon control server.
Neither needs to launch the other.

### Instance model

Every running Odon GUI has an instance ID. The protocol must not assume that
port `17870` uniquely identifies the desired instance.

At startup, Odon should:

1. Bind only to a loopback address, using an available port.
2. Generate an instance UUID and a cryptographically random bearer token.
3. Write a user-readable-only runtime manifest.
4. Remove the manifest during orderly shutdown.
5. Allow clients to discard manifests whose process no longer exists.

An illustrative manifest is:

```json
{
  "instance_id": "6293946e-f01a-451d-9a63-129101e78527",
  "pid": 48132,
  "endpoint": "tcp://127.0.0.1:49381",
  "token": "base64-random-token",
  "app_version": "0.2.0",
  "protocol_versions": [1],
  "started_at": "2026-08-20T14:32:10Z",
  "project_path": "/data/example/odon.project.json"
}
```

The project path may be omitted when no project is open. It is informational;
clients must query authoritative state after connecting.

## Odon Control Protocol

### Recommended transport

Use JSON-RPC 2.0 messages, one JSON object per line, over a persistent loopback
TCP connection for version 1.

Reasons:

- It evolves the current bridge with minimal architectural churn.
- It supports requests, responses, notifications, and multiple clients.
- Rust and Python can implement it with small dependency footprints.
- Persistent connections support events and task progress.
- A later WebSocket, Unix socket, or named-pipe transport can carry the same
  messages without changing the semantic API.

The connection implementation must support request IDs and must not assume that
responses arrive in request order.

### Authentication and handshake

The first request must authenticate and negotiate a protocol version:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "system.hello",
  "params": {
    "token": "base64-random-token",
    "client": {
      "name": "odon-client-python",
      "version": "0.1.0"
    },
    "protocol_versions": [1]
  }
}
```

The response should include:

- Negotiated protocol version.
- Odon application and control API versions.
- Instance ID.
- Supported capabilities.
- Optional feature versions.
- Maximum inline payload size.
- Current permission policy.
- A connection/session ID.

Authentication failure closes the connection after a structured error.

### Naming

Method names should be namespaced by semantic resource:

- `system.*`
- `app.*`
- `project.*`
- `viewer.*`
- `viewer.channels.*`
- `viewer.camera.*`
- `viewer.layers.*`
- `viewer.selection.*`
- `viewer.objects.*`
- `viewer.annotations.*`
- `mosaic.*`
- `ui.*`
- `tasks.*`
- `events.*`
- `data.*`

Avoid putting the version in every method name. Negotiate the major protocol
version during `system.hello` and advertise granular capabilities.

### Requests and responses

Example request:

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "viewer.channels.set_visible",
  "params": {
    "channels": ["DAPI", "CD3"],
    "mode": "only"
  }
}
```

Example result:

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "result": {
    "visible": [
      {"id": "channel:0", "index": 0, "name": "DAPI"},
      {"id": "channel:2", "index": 2, "name": "CD3"}
    ],
    "revision": 118
  }
}
```

Application failures must use JSON-RPC error responses rather than successful
results containing an `error` field.

### Structured errors

Define stable machine-readable error names in `error.data.kind`:

| Error kind | Meaning |
| --- | --- |
| `INVALID_PARAMS` | Parameters failed validation. |
| `NOT_FOUND` | The requested resource or object does not exist. |
| `WRONG_MODE` | The operation is unavailable in the current project/single/mosaic mode. |
| `NOT_READY` | Required data or viewer state is still loading. |
| `UNSUPPORTED` | The connected Odon build lacks the required capability. |
| `CONFLICT` | A revision, ownership, or state precondition failed. |
| `PERMISSION_DENIED` | The session cannot perform the requested operation. |
| `RESOURCE_LIMIT` | Payload, memory, or other configured limit was exceeded. |
| `CANCELLED` | A task was cancelled. |
| `INTERNAL` | An unexpected Odon error occurred. |

Errors should include relevant resource IDs, current mode, expected capability,
or validation paths in structured data. Human-readable messages may change;
error kinds must remain stable within an API major version.

### Resource identifiers

Stable IDs are required for resources manipulated across events and requests:

- Instance ID.
- Connection/session ID.
- Project and ROI ID.
- Viewer ID.
- Layer ID.
- Channel ID in addition to channel index and name.
- UI extension and component ID.
- Task ID.
- Temporary data resource ID.

IDs should not be derived solely from display names. Display names are mutable
and may not be unique.

### State and revisions

Odon should expose coherent resource snapshots rather than forcing clients to
assemble state from numerous unrelated calls:

- `app.get_state`
- `project.get_state`
- `viewer.get_state`
- `viewer.layers.list`
- `viewer.channels.list`
- `ui.extensions.list`

Every mutable top-level resource should carry a monotonically increasing
revision. Mutations may accept `if_revision` to prevent stale updates:

```json
{
  "layer_id": "layer:cells",
  "opacity": 0.5,
  "if_revision": 27
}
```

On conflict, Odon returns the current revision and enough information for the
client to refresh.

### Events

Clients explicitly subscribe to event patterns:

```json
{
  "jsonrpc": "2.0",
  "id": 9,
  "method": "events.subscribe",
  "params": {
    "events": [
      "viewer.selection.changed",
      "viewer.camera.changed",
      "viewer.layers.*",
      "tasks.*",
      "ui.extension:org.example.cellpose.*"
    ]
  }
}
```

Events are JSON-RPC notifications and include an event sequence number, source
resource, resource revision, and optional initiating session/request IDs.

Initial event families should include:

- Application mode and project changes.
- Viewer created, ready, busy, and closed.
- Active ROI and plane changes.
- Channel visibility, order, colour, contrast, and active channel changes.
- Camera changes, with coalescing during continuous navigation.
- Layer added, changed, reordered, and removed.
- Object and annotation selection changes.
- Task progress, completion, failure, and cancellation.
- Declarative UI input and action events.
- Odon shutdown and extension disconnect state.

High-frequency events must be coalesced or rate-limited. A subscriber can request
an appropriate maximum rate where supported.

### Long-running tasks

Operations such as loading a large dataset, exporting data, writing labels, or
running an Odon-side analysis should return a task handle promptly:

```json
{
  "task_id": "task:01J5S8Y2B2NSR68JCY5M3W4Q9F",
  "state": "queued"
}
```

Tasks expose:

- State: queued, running, completed, failed, or cancelled.
- Progress fraction where known.
- Human-readable phase.
- Structured result or error.
- Creation and completion timestamps.
- Whether cancellation is supported.

Methods:

- `tasks.get`
- `tasks.list`
- `tasks.cancel`
- `tasks.forget`

The Python SDK provides `task.wait()`, `await task`, progress callbacks, and
timeouts without imposing a five-second application timeout.

### Batch operations

Provide a batch method for related operations that should avoid intermediate
redraws and partial UI states:

```json
{
  "jsonrpc": "2.0",
  "id": 15,
  "method": "system.batch",
  "params": {
    "atomic": false,
    "operations": [
      {"method": "viewer.channels.set_visible", "params": {"channels": ["DAPI", "CD3"]}},
      {"method": "viewer.channels.set_contrast", "params": {"channel": "CD3", "min": 20, "max": 900}},
      {"method": "viewer.camera.fit", "params": {}}
    ]
  }
}
```

Version 1 may support redraw suppression and ordered execution without promising
transactional rollback. `atomic: true` should only be advertised if the command
set can genuinely guarantee it.

### Introspection

The protocol should expose:

- `system.get_capabilities`
- `system.describe_methods`
- JSON Schemas for request and response types.
- Event schemas.
- Declarative UI component schemas.

This supports debugging and generation, but the Python SDK should retain a
curated ergonomic layer rather than exposing only generated method stubs.

## Typed Rust Control Core

### Proposed module structure

An illustrative structure is:

```text
src/control/
  mod.rs
  command.rs
  error.rs
  event.rs
  registry.rs
  response.rs
  task.rs
  protocol/
    mod.rs
    connection.rs
    discovery.rs
    framing.rs
    server.rs
  ui/
    mod.rs
    component.rs
    render.rs
    state.rs
```

MCP-specific code remains under `src/mcp/` and imports the public control
registry rather than maintaining a separate method allow-list.

### Command definitions

Use serializable typed request and response structures. Add schema generation,
for example with `schemars`, after validating that the dependency and generated
schema quality are suitable.

```rust
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SetVisibleChannelsRequest {
    pub channels: Vec<ChannelSelector>,
    #[serde(default)]
    pub mode: VisibilityMode,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct SetVisibleChannelsResponse {
    pub visible: Vec<ChannelSummary>,
    pub revision: u64,
}
```

The registry should associate each method with:

- Name and summary.
- Required capability and permission.
- Request and response schemas.
- Whether it mutates state.
- Whether it starts a task.
- MCP exposure metadata, if appropriate.
- Stability level: experimental, provisional, or stable.

### UI-thread execution

Protocol connections must never mutate egui or viewer state directly. They
decode and validate requests, enqueue typed commands, request an egui repaint,
and wait asynchronously for a typed result.

The UI thread should execute a bounded number of commands per frame or within a
time budget. A single client must not be able to starve rendering by flooding
the queue.

Commands that perform expensive IO or computation should start background work
and return a task handle rather than blocking the UI thread.

### Shared command path

Where practical, native UI actions, deep links, MCP calls, and Python calls
should converge on the same application command functions. This reduces
behavioral differences and makes protocol tests meaningful. It is not necessary
to rewrite every native widget before the first release; convergence can be
incremental.

## Python Package

### Naming and distribution

Recommended distribution name: `odon-client`
Recommended import name: `odon`

The wheel contains Python code only. It does not bundle Odon, Rust libraries,
Cellpose, NumPy, or an interpreter.

Suggested optional dependency groups:

- `odon-client[arrays]`: NumPy and Zarr adapters.
- `odon-client[tables]`: PyArrow and pandas/Polars adapters as appropriate.
- `odon-client[notebook]`: notebook event-loop and display helpers.
- Domain extensions such as `odon-cellpose` remain separate distributions.

The base package should keep required dependencies small. A standard-library
transport is possible; a light validation/model dependency may be justified if
it materially improves compatibility and error reporting.

### Package structure

```text
odon/
  __init__.py
  client.py
  async_client.py
  connection.py
  discovery.py
  errors.py
  events.py
  tasks.py
  types.py
  resources/
    app.py
    project.py
    viewer.py
    channels.py
    camera.py
    layers.py
    selection.py
    objects.py
    mosaic.py
    ui.py
  ui/
    components.py
    actions.py
    bindings.py
    extension.py
  adapters/
    numpy.py
    zarr.py
    arrow.py
    geopandas.py
```

### Connection API

```python
import odon

instances = odon.instances()
app = odon.connect()                  # only instance, or selected default
app = odon.connect(instance_id="...")
app = odon.connect(endpoint="...", token="...")
```

If discovery finds multiple plausible instances, `connect()` should raise a
typed `MultipleInstancesError` containing summaries rather than choosing
silently.

Odon may optionally be launched by the SDK:

```python
app = odon.launch(executable="/Applications/odon.app/Contents/MacOS/odon")
```

Launching locates or uses an explicitly supplied installed Odon executable. The
SDK still does not distribute Odon.

### Synchronous and asynchronous clients

Both forms should share resource types and behavior:

```python
app = odon.connect()
app.viewer.camera.fit()

app = await odon.async_connect()
await app.viewer.camera.fit()
```

The synchronous client can maintain a background reader thread for response
dispatch and event delivery. The asynchronous client uses the active event loop.

Notebook support must be tested explicitly because notebooks often already run
an asyncio loop.

### Resource-oriented API

Illustrative API:

```python
app = odon.connect()

task = app.open_dataset("/data/sample.ome.zarr")
task.wait()
app.viewer.wait_until_ready()

channels = app.viewer.channels.list()
app.viewer.channels.set_visible(["DAPI", "CD3"], mode="only")
app.viewer.channels.set_contrast("CD3", minimum=20, maximum=900)
app.viewer.camera.fit()

cells = app.viewer.layers["cells"]
cells.set_style(color_by="cell_type", fill=True, opacity=0.8)

selection = app.viewer.selection.get()
app.viewer.screenshot("figures/roi-001.png").wait()
```

Methods should accept resource objects and stable IDs, with names and indices as
convenient selectors where ambiguity is handled explicitly.

### Snapshots and cached state

Calls such as `viewer.get_state()` return immutable typed snapshots. The client
may maintain an event-updated cache, but cache use must be explicit:

```python
state = app.viewer.get_state()       # RPC
state = app.viewer.cached_state      # no RPC; may be stale or None
```

### Raw escape hatch

Expose a low-level call for provisional methods and debugging:

```python
result = app.call("viewer.experimental_operation", {"value": 4})
```

The raw method does not remove the need for curated, documented SDK methods.

### Error mapping

Map stable protocol errors to Python exceptions:

- `OdonError`
- `ConnectionError`
- `AuthenticationError`
- `ProtocolVersionError`
- `InvalidParametersError`
- `ResourceNotFoundError`
- `WrongModeError`
- `NotReadyError`
- `UnsupportedCapabilityError`
- `ConflictError`
- `PermissionDeniedError`
- `TaskCancelledError`

Each exception carries the protocol method, request ID, structured error data,
and human-readable message.

### Context and cleanup

```python
with odon.connect() as app:
    extension = app.ui.register_extension(...)
    ...
```

Closing a client connection should release session-scoped UI registrations,
subscriptions, and temporary resources according to their declared lifetimes.
Durable layers and project changes must not be silently discarded.

## API Surface

The initial semantic method groups should cover the following.

### System and application

- Connect, authenticate, inspect versions and capabilities.
- Inspect current mode and loading state.
- List running tasks and extensions.
- Request graceful app shutdown only with an explicit permission and user-facing
  policy.
- Batch commands.

### Projects and ROIs

- Open, inspect, save, and optionally save-as projects.
- List, select, focus, and open ROIs.
- Navigate previous and next ROI.
- Inspect and update project-backed view state where safe.
- Open selected ROIs as a mosaic.

### Viewer and plane state

- Inspect active viewer, data source, dimensions, axes, coordinate transforms,
  current plane, viewport, and readiness.
- Set supported Z, time, or other plane selectors.
- Fit, pan, zoom, and set camera center/scale.
- Configure smooth-pixel behavior and side panels.

### Channels

- List channels and stable channel IDs.
- Get/set active and visible channels.
- Get/set order, groups, colours, notes, and contrast.
- Request histograms and intensity statistics as tasks where necessary.

### Layers

- List all layers with type, source, visibility, style, transform, ownership, and
  provenance.
- Add, update, replace, reorder, group, hide, show, and remove layers.
- Support image, labels, objects/shapes, points, masks, annotations, and text or
  metadata overlays as Odon gains corresponding primitives.
- Distinguish project-owned, session-owned, extension-owned, and temporary
  layers.
- Apply typed style specifications rather than exposing internal renderer fields.

### Objects, selection, and annotations

- Query and modify object selections.
- Query objects in a rectangle, viewport, polygon, or layer-defined region.
- Get/set object filters and property colouring.
- Retrieve selected object properties by reference or in bounded inline results.
- Create and edit supported annotations.
- Subscribe to selection and annotation events.

### Mosaic

- Configure layout, grouping, sorting, columns, gaps, and labels.
- Query cell/ROI bounds and active ROI.
- Apply shared channel and layer state.
- Fit all or focus a specific mosaic cell.

### Screenshots and export

- Capture canvas, window, project page, or specified viewport.
- Return task completion only after the output is actually written.
- Add metadata describing dataset, view state, Odon version, and extension
  provenance where requested.
- Support future in-memory image results through a binary resource rather than
  base64 in ordinary JSON.

## Declarative UI System

### Model

Python registers a versioned, serializable component tree. Rust validates and
retains the tree, then renders it each egui frame.

```text
Python UI objects -> UI manifest/patches -> Rust UI state -> egui widgets
       ^                                                      |
       +---------------- structured events -------------------+
```

This is a remote declarative UI model, not remote drawing.

### Extension registration

Every custom UI contribution belongs to an extension:

```python
extension = app.ui.register_extension(
    id="org.odon.cellpose",
    name="Cellpose Segmentation",
    version="0.1.0",
    capabilities=["ui.panels", "viewer.layers.write", "data.read"],
)
```

Extension IDs use reverse-domain-style names or another collision-resistant
convention. Component IDs are unique within an extension.

Registration returns an extension session and the capabilities actually
granted. A future trust dialog can require user approval for sensitive
capabilities.

### UI insertion points

Initial named locations should be intentionally constrained:

- `left.sections`
- `right.tabs`
- `top_bar.actions`
- `canvas.context_menu`
- `canvas.controls`
- `status_bar`
- `project.cards`
- application dialogs and notifications through explicit commands

After the component model is stable, Odon may expose configurable overall
layouts with references to built-in panels and the central canvas. A minimal
Rust-owned shell and safe-mode reset must always remain available.

### Core component tree

All components have:

- Stable component ID.
- Component type and schema version.
- Optional label, help text, enabled/visible state, and style hints.
- Optional value or model binding.
- Optional actions.
- Optional children for container components.
- Ownership and revision metadata maintained by Rust.

Illustrative Python:

```python
from odon import ui

panel = ui.Panel(
    id="cell-qc",
    title="Cell QC",
    children=[
        ui.Select(
            id="cell-type",
            label="Cell type",
            options=["Tumour", "T cell", "Macrophage"],
        ),
        ui.Slider(
            id="confidence",
            label="Minimum confidence",
            minimum=0.0,
            maximum=1.0,
            value=0.8,
            event_policy=ui.Debounce(milliseconds=100),
        ),
        ui.Button(
            id="apply",
            label="Apply filter",
            action=ui.emit("apply-filter"),
        ),
        ui.Button(
            id="fit",
            label="Fit image",
            action=ui.command("viewer.camera.fit"),
        ),
    ],
)

extension.register(panel, location="right.tabs")
```

### Layout components

Initial layout vocabulary:

- Row and column.
- Grid.
- Group or framed section.
- Tabs.
- Collapsible section.
- Scroll region.
- Separator and spacer.

Layout properties should be semantic and constrained. Avoid attempting to
recreate CSS inside egui.

### Input components

Initial input vocabulary:

- Button and button group.
- Toggle and checkbox.
- Integer and floating-point sliders.
- Integer, number, and text input.
- Single and multi-select.
- Radio group.
- Colour input.
- File or directory request through an Odon-native picker and explicit
  permission policy.
- Channel, layer, ROI, object-property, and annotation selectors.

Domain-aware selectors are more valuable than only generic widgets because they
can bind to Odon state and remain valid as state changes.

### Display components

Initial display vocabulary:

- Text and restricted Markdown.
- Status, warning, and error callouts.
- Progress bar and spinner.
- Key/value properties.
- Image thumbnail by data resource reference.
- Small table, with a path to virtualized tables.
- Histogram and basic plot specifications.
- Legend and colour ramp.
- Selection and layer summaries.

Rich text and external links must be sanitized and follow a clear user consent
policy.

### Three interaction types

#### Binding

A binding directly reads and writes a supported Odon state property:

```python
ui.Toggle(
    id="fill-cells",
    label="Fill cells",
    bind=ui.bind("viewer.layers", layer_id="layer:cells", property="fill"),
)
```

Rust handles the interaction immediately and emits ordinary state events.

#### Native command

A native action invokes a registered Odon command:

```python
ui.Button(
    id="fit",
    label="Fit image",
    action=ui.command("viewer.camera.fit"),
)
```

Allowed command parameters are either fixed in the manifest or derived from
explicit component values through a constrained mapping. Do not evaluate
arbitrary expressions received from Python.

#### Python event

A custom action emits a structured event:

```python
ui.Button(
    id="run",
    label="Run Cellpose",
    action=ui.emit("run-segmentation"),
)
```

Python handles the event and updates UI or viewer state. The button can declare
busy and disconnected behavior.

### UI state

Separate component structure from mutable values:

- Manifests define the tree and defaults.
- Rust owns the live value model for rendered components.
- Python can query or patch values.
- Input events contain the relevant value snapshot and revision.
- Odon may preserve values across compatible manifest updates.

This prevents Python from needing to resend the tree on every input change.

### Patching and reconciliation

Support operations such as:

- Register or replace a complete contribution.
- Patch component properties.
- Patch multiple values in one request.
- Insert, move, or remove a child.
- Replace table rows or plot series by resource reference.
- Remove all contributions owned by an extension.

Stable component IDs let Rust reconcile changes without losing focus, scroll
position, or local values unnecessarily.

Patches include an expected revision. Invalid patches reject atomically and
leave the last valid tree active.

### Event policies

Each interactive component can request a supported policy:

- On commit.
- Immediate.
- Throttled to a maximum rate.
- Debounced for a specified interval.

Odon applies hard rate limits regardless of requested policy. Continuous slider
or pointer changes must not flood a Python process.

### Disconnection behavior

An extension declares one of these policies:

- Remove its UI when disconnected.
- Retain UI but disable Python-dependent actions.
- Retain UI for a reconnection grace period, then remove it.

Native bindings and native commands may remain active while disconnected.
Python-event actions display a clear unavailable state.

Odon must provide a visible way to inspect and remove extensions, recover the
default layout, and start without third-party UI contributions.

### Persistence

Default behavior is session-scoped UI. Persisting an extension manifest into a
project is opt-in and records:

- Extension ID and compatible version range.
- UI manifest.
- Required capabilities.
- Whether Python callbacks are required.
- User-approved trust metadata where appropriate.

A project must remain openable when the Python extension is not installed. Odon
can show the layout with unavailable actions or omit it with an explanatory
placeholder.

### Canvas customization

Canvas data and rendering should use the layer/overlay API, not ordinary UI
components. Declarative canvas controls may appear above the canvas, but large
geometry, imagery, and per-frame visuals remain Odon layer resources.

Future custom rendering may accept constrained, validated style or shader
specifications. Arbitrary native libraries or Python render callbacks are out of
scope for the protocol.

## Data Exchange

### Dataset descriptors

Odon should provide a normalized descriptor for the active data source:

- Source kind and URI/path.
- Dataset/group path.
- Shape, axes, chunks, pixel type, and multiscale levels.
- Channel metadata.
- Current plane selection.
- Image-to-world and layer transforms.
- Pixel size and physical units where known.
- Credentials/access limitations without exposing secrets.

Python should normally open the same underlying OME-Zarr, TIFF, SpatialData, or
table resource directly. For sources Python cannot access, Odon can later expose
bounded region export or a binary data service.

### Payload classes

| Data | Preferred exchange |
| --- | --- |
| Small parameters and metadata | Inline JSON. |
| Small shapes | Inline GeoJSON or typed coordinate lists. |
| Images and labels | OME-Zarr URI/path and group path. |
| Tables and point data | Arrow IPC or Parquet. |
| Large shapes | GeoParquet or Arrow representation. |
| Screenshots | Output path or managed binary resource. |
| NumPy arrays | Temporary Zarr initially; shared memory may follow. |

Set and advertise a conservative maximum inline payload size. Large inline
payload attempts should return `RESOURCE_LIMIT` with suggested alternatives.

### Managed temporary resources

The SDK can create session-scoped resources:

```python
resource = app.data.from_array(labels, format="ome-zarr", lifetime="session")
layer = app.viewer.layers.add_labels(resource, name="Cellpose preview")
```

Initially the SDK may write into a secure temporary directory and register the
resource path with Odon. The resource record includes owner, format, lifetime,
size, transform, and cleanup policy.

Cleanup rules must prevent Odon and Python from deleting user-owned source data.
Only explicitly managed temporary locations may be automatically removed.

### Coordinate contract

Every spatial data operation must identify its coordinate space. Define and
document:

- Array axis order.
- Level-zero pixel coordinates.
- Multiscale level coordinates.
- World or physical coordinates.
- Screen/viewport coordinates.
- ROI-local and mosaic-global coordinates.
- Pixel center versus pixel edge conventions.
- Transform direction and matrix layout.
- Units.

Layer creation requires a transform or a reference coordinate space. Avoid
implicit assumptions that all external geometry is level-zero XY.

Coordinate conversions should be available through typed Odon methods and pure
Python helpers, with shared conformance fixtures.

### Provenance

Derived layers should accept structured provenance:

```json
{
  "producer": "odon-cellpose",
  "producer_version": "0.1.0",
  "algorithm": "cellpose",
  "model": "user-model-2026-08",
  "parameters": {"channels": ["DAPI", "PanCK"]},
  "source_dataset": "...",
  "created_at": "..."
}
```

Secrets, full environment dumps, and non-serializable Python objects must not be
stored. Extensions may add domain metadata under namespaced keys.

### Future high-performance transport

Shared memory or binary frames may be added after measuring real workloads. It
must not be a prerequisite for the initial control and declarative UI API.

Any shared-memory design needs explicit dtype, shape, strides, endianness,
ownership, lifetime, acknowledgement, and cleanup semantics. Cross-platform
behavior must be tested before becoming public.

## Extension Lifecycle

An external Python extension follows this lifecycle:

1. Discover and connect to Odon.
2. Authenticate and negotiate capabilities.
3. Register extension identity and requested permissions.
4. Register UI, commands, subscriptions, and optional temporary resources.
5. Handle events and issue control commands.
6. Update progress, UI state, and layers during work.
7. Persist explicit outputs if requested.
8. Unregister and clean session resources on shutdown.

The SDK should provide a small extension runner:

```python
from odon.extensions import run
from odon_cellpose import CellposeExtension

run(CellposeExtension())
```

It handles connection selection, logging, reconnect policy, signal handling, and
event-loop lifetime.

## Cellpose Reference Extension

### Purpose

`odon-cellpose` should be built after the core layer, task, event, and
declarative UI APIs are usable. It is both a useful feature and an integration
test for the extension platform.

Odon does not install Cellpose. The user installs `odon-cellpose`, Cellpose, and
the desired ML runtime into a Python environment they control.

### User workflow

1. Start Odon and open an image or project ROI.
2. Start the Cellpose extension from a terminal or notebook.
3. Odon displays a native Cellpose tab.
4. Select model, input channels, execution device, and target extent.
5. Run a preview on the current viewport or selected region.
6. Display the result as a temporary labels or object layer.
7. Adjust settings and repeat without accumulating preview layers.
8. Run the accepted configuration over the desired full-resolution extent.
9. Write a durable OME-Zarr labels group or supported object representation.
10. Attach provenance and optionally add the output to the project.

### Sequence

```text
User          Odon/egui             odon-cellpose          data store
 |               |                       |                     |
 | click Preview |                       |                     |
 |-------------->| emit UI event         |                     |
 |               |---------------------->|                     |
 |               |                       | read source region  |
 |               |                       |-------------------->|
 |               |                       | run Cellpose        |
 |               |<----------------------| progress/UI patches |
 |               |                       | write temp labels   |
 |               |                       |-------------------->|
 |               |<----------------------| add/replace layer   |
 | see preview   |                       |                     |
 |<--------------|                       |                     |
```

### Required platform capabilities

- Active dataset and region descriptor.
- Stable channel selection and plane metadata.
- Viewport and user-drawn region coordinates.
- Temporary OME-Zarr data registration.
- Add/replace/remove labels or object layer.
- Extension UI panel and events.
- Task progress and cancellation.
- Provenance.
- Explicit persistence into a chosen output location.

### Tiling responsibility

The first Cellpose extension can own tiling, overlap, stitching, unique label
IDs, and output writing in Python. Odon supplies dataset geometry and displays
results. If several extensions later duplicate these mechanics, a generic Odon
or SDK tiling service can be designed from observed needs.

### Correction workflow

A later iteration can send selected mask IDs back to Python for deletion,
merging, relabeling, or localized reprocessing. Corrections should produce an
auditable operation log rather than mutating durable labels invisibly.

## Security and Trust

### Local does not mean trusted

The control server binds only to loopback by default, but another process owned
by the user could still attempt a connection. Use a random token stored in a
runtime manifest with user-only filesystem permissions.

### Permissions

Capabilities should distinguish at least:

- Read viewer/project state.
- Change viewer state.
- Read source data descriptors.
- Add temporary layers.
- Modify project state.
- Write or export files.
- Register UI.
- Register persistent UI.
- Request application shutdown.

Version 1 may grant a standard local-client set after token authentication, but
the protocol should carry permissions so Odon can add consent and policy later.

### File operations

File-writing methods require explicit paths or Odon-native save dialogs. Never
allow an extension to disguise arbitrary filesystem operations as a generic UI
binding.

Managed resource cleanup is restricted to directories created and registered by
the SDK or Odon for that purpose.

### UI safety

- Validate every UI manifest and patch against a schema.
- Limit component count, nesting depth, string sizes, event rates, and table
  payloads.
- Sanitize restricted Markdown and external links.
- Do not evaluate Python, JavaScript, expressions, or arbitrary shader code from
  a UI manifest.
- Namespace component IDs and enforce extension ownership.
- Provide extension inspection, disable, removal, and safe-mode controls.

### Remote control

Remote TCP binding is out of scope for the initial release. If added, it must
have a separate threat model with transport encryption, stronger authentication,
explicit user enablement, and filesystem/data access restrictions.

## MCP Integration

`odon_mcp` remains a small standalone executable. It should consume the typed
control registry and forward supported methods to the running GUI.

Not every control method must become an MCP tool. The registry can mark methods
as:

- Suitable for MCP with an AI-oriented description and schema.
- Hidden because they are too low-level, high-volume, or unsafe.
- Wrapped by a higher-level MCP tool.

MCP results should remain structured where the MCP protocol permits. The adapter
may provide concise textual summaries, but the Odon control result itself should
not be designed around an LLM text envelope.

The adapter negotiates the Odon control protocol and reports a clear version or
capability error when it is incompatible with the running GUI.

## Versioning and Compatibility

Version these independently:

- Odon application version.
- Odon Control Protocol major version.
- Control capability or schema versions.
- Python SDK version.
- Individual extension version.
- Declarative UI schema version.

Compatibility policy for protocol version 1:

- Additive optional fields are compatible.
- Clients ignore unknown response and event fields.
- Servers reject unknown request fields for stable typed methods unless a method
  explicitly permits extension metadata.
- Removing or changing field meaning requires a protocol major version.
- New methods and enum variants require capability discovery; enum handling must
  be designed carefully so old clients fail clearly.
- Experimental methods may change in minor releases and are namespaced or marked
  explicitly.
- Deprecations remain available for a documented transition period.

The Python SDK should publish its supported protocol range and fail early with a
helpful upgrade/downgrade message.

## Performance and Responsiveness

Initial performance requirements:

- Native bindings and native UI commands complete without a Python round trip.
- Protocol parsing and queueing do not occur on the render-critical path beyond
  bounded command execution.
- Camera and slider events are coalesced or throttled.
- No unbounded per-client command or event queue.
- Large payloads are rejected before expensive allocation where possible.
- UI manifests have component and depth limits.
- Tables and long lists use virtualization or referenced data.
- A disconnected or stalled Python client cannot block egui.
- Long-running work returns a task and performs computation off the UI thread.

Benchmark and instrument:

- Round-trip latency for a simple query.
- Native binding latency.
- Event delivery under camera movement.
- UI rendering cost by component count.
- Layer registration and replacement latency.
- Temporary-array write and load cost.
- Viewer frame rate while a busy extension is connected.

Set numeric budgets after the first prototype measurements rather than choosing
unsupported targets now.

## Observability and Diagnostics

Odon should expose a Control/Extensions diagnostics view containing:

- Listening state and instance ID, without displaying the full token by default.
- Connected clients and declared names.
- Negotiated protocol versions.
- Granted capabilities.
- Registered extensions and UI contributions.
- Queue depths and recent rate-limit warnings.
- Active tasks.
- Last structured error per extension.
- Controls to disconnect or remove an extension.

Protocol logging must redact tokens, credentials, and sensitive parameters.
Debug logging can include request IDs, methods, durations, result categories, and
payload sizes.

The Python SDK should use standard `logging` and include connection/session IDs
in debug messages.

## Testing Strategy

### Rust unit tests

- Request deserialization and validation.
- Response and error serialization.
- Command registry uniqueness and schema completeness.
- Capability checks.
- UI manifest validation, reconciliation, and ownership.
- Coordinate conversions.
- Task state transitions and cancellation.
- Discovery manifest creation and stale-manifest handling.

### Protocol conformance tests

Maintain request/response/event fixtures shared by Rust and Python. Test:

- Handshake and incompatible versions.
- Authentication failure.
- Out-of-order concurrent responses.
- Structured errors.
- Event subscriptions and sequence numbers.
- Queue and payload limits.
- Reconnection and shutdown.

### Python unit tests

Use a deterministic fake Odon server for:

- Sync and async clients.
- Resource wrappers.
- Error mapping.
- Event callbacks and filters.
- Task waiting, progress, timeout, and cancellation.
- Declarative UI serialization and patches.
- Discovery behavior with zero, one, multiple, and stale instances.

### Integration tests

Run against a real Odon build and small fixtures:

- Open OME-Zarr and wait for readiness.
- Manipulate channels and camera.
- Add and remove supported layers.
- Register a panel and simulate component events.
- Disconnect an extension and verify failure behavior.
- Capture a screenshot and verify completion.
- Exercise Cellpose with a lightweight or mocked model in continuous integration.

GPU-dependent screenshot assertions need platform-tolerant comparison or a
controlled rendering environment.

### Compatibility tests

Test supported combinations of:

- Current Python SDK with current Odon.
- Current SDK with the oldest supported Odon protocol implementation.
- Oldest supported SDK with current Odon.
- Current `odon_mcp` with current and oldest supported Odon.

Schema snapshots should be reviewed as compatibility artifacts, not updated
blindly.

## Documentation and Developer Experience

Required documentation:

- Installation and connecting to Odon.
- API quick start.
- Sync and async usage.
- Notebook guide.
- Resource and state model.
- Events and tasks.
- Adding and exchanging layers.
- Coordinate systems.
- Declarative UI component reference.
- Building and packaging an extension.
- Cellpose tutorial.
- Security and extension trust.
- Protocol reference and compatibility policy.
- Migration guide from raw bridge or MCP-oriented automation.

Provide templates for:

- Minimal script.
- Notebook integration.
- UI-only extension.
- Analysis extension with a background task.
- Layer-producing extension.
- Packaged extension with entry point and tests.

## Delivery Roadmap

### Phase 0: Decisions, inventory, and spike — implemented

Deliverables:

- Approve this architecture or record changes as focused decision records.
- Inventory every existing MCP control method, its parameters, responses,
  readiness behavior, and mode restrictions.
- Define protocol version 1 envelopes, error model, naming, and capability rules.
- Prototype a small Python client against the existing bridge to validate API
  ergonomics without publishing the bridge format as stable.
- Measure simple round-trip latency and event-loop impact.

Exit criteria:

- A reviewed protocol draft exists.
- Ten representative existing commands have proposed typed contracts.
- A Python spike can connect, inspect state, change channels, move the camera,
  and capture a screenshot.
- Open architectural decisions are explicitly recorded.

### Phase 1: Typed control core and protocol server — implemented provisionally

Deliverables:

- `src/control` typed command, result, error, and registry modules.
- JSON-RPC request IDs and structured application errors.
- `system.hello`, capability discovery, and schema introspection.
- Typed migration of the existing MCP-supported operations.
- Bounded UI-thread command execution.
- Compatibility-preserving temporary support for the existing bridge client if
  needed during migration.

Exit criteria:

- All current MCP operations pass through the typed control core.
- No control failure needs a successful `{"error": ...}` result.
- Protocol and schema tests cover the migrated methods.

### Phase 2: Discovery, security, and Python SDK foundation — implemented

Deliverables:

- Dynamic loopback endpoint and runtime manifest.
- Per-instance authentication token.
- Instance enumeration and stale-manifest cleanup.
- `odon-client` package with sync and async connections.
- Typed errors, application/project/viewer resources, raw call, and version
  checks.
- Initial packaging and API documentation.

Exit criteria:

- Python can reliably select among zero, one, or multiple Odon instances.
- The SDK covers the current stable MCP control surface.
- Odon does not require Python to be installed.
- Windows, macOS, and Linux connection behavior is tested.

### Phase 3: Events, revisions, and tasks — implemented

Deliverables:

- Persistent multi-request connections.
- Event subscription and delivery.
- State revisions and optional mutation preconditions.
- Task registry, progress, completion, failure, and cancellation.
- Python event iterators, callbacks, cached snapshots, and task handles.
- Diagnostics for connections, queues, and tasks.

Exit criteria:

- Python can react to selection, channel, layer, camera, and readiness changes.
- Opening and screenshot operations can be awaited to actual completion.
- A slow client cannot block Odon or grow queues without bound.

### Phase 4: General layer and data-resource API — implemented for local v1 resources

Deliverables:

- Stable layer IDs, ownership, lifecycle, list/add/update/replace/remove, ordering,
  and style APIs.
- Normalized dataset and coordinate-space descriptors.
- Temporary resource registry.
- OME-Zarr images/labels and Arrow/Parquet/GeoParquet resource references.
- Python NumPy-to-temporary-Zarr adapter.
- Provenance model.

Exit criteria:

- Python can add a derived label layer and replace a preview efficiently.
- Python can add object/point data with an explicit transform.
- Session cleanup cannot remove user-owned data.
- Coordinate conformance fixtures pass in Rust and Python.

### Phase 5: Declarative UI version 1 — implemented experimentally

Deliverables:

- Extension registration and ownership.
- Right-tab and top-bar insertion points, followed by other approved locations.
- Layout, input, and display component vocabulary.
- Bind, command, and emit interaction types.
- Retained values, patches, revisions, throttling, and disconnection policies.
- Extension diagnostics and safe-mode removal.
- Python UI component classes and extension runner.

Exit criteria:

- A Python package can add a native panel without modifying Rust.
- Bound opacity/channel controls remain responsive if Python is busy.
- A Python-event button can start a task and display progress.
- Invalid or abusive manifests are rejected without destabilizing the viewer.
- Disconnecting the extension produces the declared, visible behavior.

### Phase 6: Cellpose reference extension — vertical slice implemented

Deliverables:

- Separately packaged `odon-cellpose` proof of concept.
- Native panel for model/input/extent/output settings.
- Viewport or selected-region preview.
- Progress, cancellation, and error reporting.
- Temporary preview replacement.
- Tiled durable output with provenance.
- Tutorial and integration tests.

Exit criteria:

- A user-controlled Python environment can run Cellpose against an Odon-opened
  dataset without Python being bundled in Odon.
- Results align correctly at full resolution and after camera navigation.
- Preview and final output lifecycles are predictable.
- Cellpose failure does not interrupt native Odon viewing.

The reference extension covers the connection, native panel, event, inference,
temporary Zarr, replacement, rendering, provenance, reconnect, and cooperative
cancellation path. Tiled inference and user-selected durable output remain part
of stabilization rather than the initial proof of architecture.

### Phase 7: Stabilization and extension ecosystem — pending

Deliverables:

- Public compatibility policy and deprecation process.
- Extension templates and packaging guidance.
- Permission/trust UX informed by real extensions.
- Performance profiling and numeric budgets.
- Additional reference integrations selected from actual user demand.
- Evaluation of shared memory, virtualized tables, declarative plots, and
  persisted extension layouts.

Exit criteria:

- The stable subset of protocol version 1 is declared.
- At least two independently structured extensions validate that the API is not
  Cellpose-specific.
- Cross-version compatibility tests run in continuous integration.

## Recommended First Vertical Slice

The first vertical slice should be deliberately small but cross every important
boundary:

1. Connect and authenticate through the new protocol.
2. Query the active dataset and current viewport.
3. Register a right-panel tab containing text, a channel selector, a slider, and
   a button.
4. Bind the channel selector to Odon state locally.
5. Emit the button event to Python.
6. Have Python generate a small synthetic label image for the viewport.
7. Write it to a managed temporary OME-Zarr resource.
8. Add or replace a preview label layer.
9. Update a native progress/status component.
10. Disconnect Python and verify that Odon remains usable and cleans the
    temporary contribution safely.

This slice is more informative than implementing many additional control
methods first. It validates protocol, events, UI, data exchange, coordinates,
ownership, and failure behavior in one bounded experiment.

## Stable Version 1 Acceptance Criteria

The experimental API should only be declared stable version 1 when:

- Odon ships without Python and works normally when no client is connected.
- A separately installed Python wheel discovers and authenticates to Odon on all
  supported desktop platforms.
- The stable SDK covers normal project, viewer, channel, camera, layer,
  selection, mosaic, screenshot, and readiness workflows.
- Structured errors and capability negotiation replace string inspection.
- Events and tasks are reliable, bounded, and documented.
- Python can create and manage derived layers without large JSON transfers.
- Coordinate contracts are explicit and tested.
- Python can register native Rust-rendered UI using a versioned component schema.
- Native bindings remain responsive independently of Python.
- Python callbacks can run long work with progress and cancellation without
  blocking egui.
- Odon visibly handles extension disconnection and provides a safe reset.
- MCP uses the same typed control definitions rather than a parallel method
  implementation.
- A Cellpose reference extension demonstrates the complete workflow.
- Compatibility, security, and cleanup behavior have automated coverage.

## Risks and Mitigations

### The API mirrors internal state too closely

Mitigation: expose semantic resources and commands, use stable IDs, and keep
renderer-specific details out of stable schemas.

### Declarative UI becomes an imitation of HTML/CSS

Mitigation: use a compact egui-oriented component vocabulary, semantic layout
hints, and domain-specific Odon controls.

### Python latency makes controls feel poor

Mitigation: support Rust-local bindings and commands; throttle events; use
Python only for custom logic.

### Large data overwhelms the protocol

Mitigation: enforce inline limits and exchange data by OME-Zarr, Arrow, Parquet,
or managed resources.

### Coordinate mismatches produce incorrect scientific results

Mitigation: make coordinate spaces mandatory, supply transform utilities, and
maintain cross-language fixtures.

### Extensions leave files, UI, or tasks behind

Mitigation: explicit ownership and lifetimes, connection cleanup, managed
temporary directories, and extension diagnostics.

### API and MCP definitions drift

Mitigation: generate both from the typed Rust registry and test registry/schema
completeness.

### A malicious or broken local extension degrades Odon

Mitigation: authentication, capabilities, bounds on queues and UI manifests,
task isolation, user-visible extension controls, and no arbitrary code execution
inside Odon.

### The first design tries to support every future plugin

Mitigation: validate the primitives with a small vertical slice and Cellpose,
mark early APIs provisional, and stabilize only demonstrated abstractions.

## Resolved and Remaining Decisions

The initial implementation selected private platform runtime manifests,
loopback TCP, Python 3.10+, UUID-like stable IDs, session-only UI manifests,
Python-owned managed temporary Zarr stores, and OME-Zarr labels for the Cellpose
preview. Launch helpers require an explicit executable and then use ordinary
manifest discovery. Extension capabilities are validated against a conservative
local permission policy; a future trust/consent dialog is not implemented.

Before stabilization, the project still needs to decide which provisional
snapshot shapes and aliases become permanent, whether to generate richer schemas
from fully typed request/result models, whether project UI manifests are safe to
persist, and what durable/tiled Cellpose output workflow to support.

## Immediate Next Actions

1. Exercise the experimental wheel and Cellpose extension against packaged Odon
   on macOS, Windows, and Linux.
2. Gather real extension feedback before freezing method aliases, snapshots,
   component vocabulary, and permission UX.
3. Add a second independently structured extension and cross-version protocol
   compatibility tests.
4. Implement tiled/durable analysis output and the remaining live resource
   render adapters demanded by real workflows.
5. Publish a compatibility/deprecation policy, then declare only the proven
   subset stable.

The architecture should be reviewed after real extension use and before
declaring any Python or declarative UI surface stable.
