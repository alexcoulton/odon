# Complete Odon Application Surface: Control and Python API Plan

Status: proposed follow-on to the experimental Python API

Target: semantic control of the complete supported Odon application surface
before declaring the Python API and Control Protocol v1 stable

Related documents:

- `docs/design/python-api-plan.md`
- `docs/design/control-protocol-v1.md`
- `docs/odon-feature-inventory.md`
- `docs/design/test-coverage-matrix.md`
- `docs/reference/python-api.md`

## Executive Summary

The first Python API implementation established the correct architecture: Odon
remains a standalone Rust application, a central authenticated control protocol
owns the product boundary, and Python and MCP are independent clients of that
same boundary. It covers the former MCP surface and adds events, retained tasks,
external data and layers, declarative native UI, revisions, and extension
lifecycle management.

It does not yet expose the complete Odon application. Important native state and
operations remain GUI-only, including multidimensional plane navigation,
complete channel and layer properties, project construction, saved views,
object styling, mask editing, threshold regions, analysis calls, measurements,
exports, and memory pinning.

This plan extends the central Rust API first and then exposes it through both
sync and async Python resources. It does not automate widgets or reproduce the
egui layout in Python. The API operates on semantic application state: projects,
viewports, layers, masks, selections, measurements, and other domain concepts.

Completeness will be measurable. Odon will maintain a machine-readable parity
manifest that maps every supported user-visible capability to:

1. a query or state snapshot;
2. one or more semantic commands;
3. completion and event behavior;
4. a typed Python wrapper;
5. permission requirements; and
6. automated test evidence.

The stable API can intentionally exclude OS-owned presentation details and raw
input gestures, but it must not leave a meaningful Odon workflow accessible
only by clicking the GUI.

## Definition of Complete Application Control

A native capability is controllable when an authenticated client can:

- determine whether the capability is available in the current mode;
- inspect all scientifically or operationally meaningful state it owns;
- perform every meaningful operation supported by the GUI;
- identify targets through stable IDs rather than transient list positions;
- receive structured success, failure, conflict, progress, and cancellation;
- observe changes regardless of whether Python, MCP, or the native UI caused
  them;
- express coordinates, selectors, transforms, and output ownership explicitly;
- reproduce the resulting state through a project or script where persistence
  is supported; and
- use equivalent synchronous and asynchronous Python APIs.

Completeness does not mean:

- remote mouse or keyboard simulation;
- exposing internal Rust structs, egui widget IDs, renderer buffers, or GPU
  implementation details;
- allowing Python code to execute inside the Odon process;
- allowing an extension to replace arbitrary Odon rendering code;
- exposing transient hover, drag, animation, or per-frame state unless it has
  semantic meaning; or
- making features available in modes where native Odon does not support them.

For example, Python should be able to create, edit, select, and export a mask
polygon. It does not need to simulate each mouse-down event used to draw it.

## Architectural Requirements

### One central product API

Every operation must enter Odon through `src/control/`. The Python SDK, MCP,
deep-link application, native menus, and native egui controls should converge on
the same domain commands wherever their semantics are equivalent.

```text
native UI -----------\
native menus ---------\
deep links ------------> Rust domain services -> RootApp/viewer state
Python SDK -> protocol -/
MCP adapter -> protocol/
```

The native UI may still maintain gesture state and presentation details, but it
must call shared domain operations for meaningful mutations. A protocol handler
must not imitate a button click or directly duplicate the UI implementation.

The central registry and Python SDK target the complete semantic surface. MCP
remains a deliberately curated adapter: it may expose only operations that are
useful and safe for an AI client, but every MCP tool must map to the same
canonical registry entry. Deep links and native menus are similarly selective
entry points, not separate APIs.

### Domain modules, not a single bridge

Extend the control core into domain services with typed request and result
models:

```text
src/control/
  application.rs
  datasets.rs
  projects.rs
  viewer.rs
  channels.rs
  layers.rs
  objects.rs
  masks.rs
  analysis.rs
  measurements.rs
  mosaic.rs
  memory.rs
  exports.rs
  events.rs
  tasks.rs
  registry.rs
  protocol.rs
```

The exact file split can follow the Rust implementation, but ownership must be
clear and domain behavior must be testable without a native window.

### Canonical names and generated aliases

Freeze canonical hierarchical method names before stable v1, for example:

- `viewer.camera.get`
- `viewer.channels.set_visible`
- `project.rois.select`
- `viewer.masks.polygons.update`
- `viewer.measurements.run`

Legacy flat names such as `set_camera` remain temporary aliases generated from
the registry. They must not be independently dispatched or documented as the
stable API. Introspection reports canonical name, aliases, stability, version,
capability, mode constraints, task behavior, and schemas.

### Stable identity and snapshots

All mutable entities need stable IDs and typed snapshots:

- application instance and control session;
- project, ROI, dataset, view preset, and mosaic item;
- channel and channel group;
- native and external layer;
- object source and object row;
- mask layer and polygon;
- analysis threshold, call, mapping, and preset;
- measurement run and result;
- data resource, extension contribution, task, and export.

Snapshots include an entity revision where atomic updates matter. Names and
indices are ergonomic selectors, not durable identity.

### Queries, mutations, tasks, and transactions

Use four explicit operation classes:

- **Queries** return promptly and do not mutate state.
- **Mutations** execute atomically on authoritative Rust state.
- **Tasks** return promptly and represent loading, computation, rendering, or
  file IO that can take significant time.
- **Transactions/batches** apply related mutations against one expected global
  revision and either commit together or fail without partial application.

Do not turn every operation into a task. Do not leave slow file or analysis work
as a blocking JSON-RPC call.

### Events describe semantic changes

Every mutable domain must emit events for both native and remote changes. Event
payloads should contain stable IDs, new revisions, initiating session/request
IDs when available, and enough information to refresh only the affected entity.

High-frequency camera, slider, and progress events remain bounded and
coalescible. Durable state changes such as polygon deletion, completed export,
or project save must not be silently lost; clients should be able to detect a
sequence gap and resynchronize.

### Capabilities and mode restrictions

Introspection must distinguish:

- unsupported by this Odon version;
- supported but unavailable in the current viewer mode;
- temporarily not ready because data is loading;
- denied by the permission policy; and
- invalid for the selected entity or view plane.

Every Python resource should offer an `available()` or capability-inspection
path without relying on string matching from exceptions.

## Current Baseline and Target Coverage

The current SDK already provides connection/discovery, sync and async clients,
state inspection, dataset/project opening, camera, basic channels, basic object
selection/filtering, mosaic layout, screenshots, tasks, events, external
resources/layers, declarative UI, and extension lifecycle.

The target matrix below defines the remaining breadth. `Partial` means the
current central API exposes useful operations but not the complete native
surface.

| Domain | Current central/Python coverage | Stable target |
| --- | --- | --- |
| Application/session | Partial | Settings, recent items, navigation, window-safe close/quit, diagnostics, and complete state summaries |
| Dataset discovery/opening | Partial | All GUI-supported local/remote sources, discovery, drag/drop-equivalent path routing, SpatialData, Xenium, and credentials policy |
| Projects/workspaces | Partial | Create, open, save/save-as, metadata, ROI CRUD/selection, samplesheets, roots, search paths, cache controls, and exports |
| Viewer/camera | Mostly covered | Canonical camera/viewport methods, back navigation, tools, coordinate conversion, render readiness, and scale bar |
| Multidimensional planes | Missing | Plane listing/selection, slice axes/ranges/indices, and plane-specific availability |
| Channels/groups | Partial | Colour, notes, filtering/sorting state, complete groups, transforms, auto-contrast, histogram, and reset/max operations |
| Unified native layers | Partial | Inspect and mutate every native/external layer, including images, labels, masks, annotations, shapes, points, objects, and mosaic labels; active state, grouping, style, transforms, source lifecycle, and exports |
| Saved views | Missing | List, create, capture, validate, replace, rename, apply, delete, and serialize |
| Objects/overlays | Partial | Source lifecycle, properties, styling, legends, filters, arbitrary selection, focus, and paged queries |
| Masks | Missing | Layer and polygon CRUD, selection, import/export, transforms, display modes, undo/redo, and project persistence |
| Threshold regions | Missing | Configuration, preview tasks, component filtering, polygonization, apply-to-mask, and cleanup |
| Analysis | Missing | Histograms, transforms, thresholds, suggestions, mappings, calls, presets, review state, and failure marking |
| Measurements | Missing | Configure, estimate, run, progress, attach properties, inspect results, and export |
| Mosaic | Partial | Complete layout, ROI selection/focus/navigation, text labels, object state, and mosaic memory controls |
| Memory/performance | Missing | Loading metrics, estimates, pin/unpin/unload, scope, channel/level selection, and task progress |
| Screenshots/report output | Partial | Complete settings, regions/hosts, deterministic readiness, scale bar, metadata, and output ownership |
| Deep links | Indirect | Parse, validate, apply, generate, and report unsupported fields through the central API |
| Extensions/UI | Broad experimental | Complete supported schema, dialogs/actions where approved, persistence decision, trust UX, and stable compatibility rules |

This matrix must be kept aligned with `docs/odon-feature-inventory.md` and the
feature IDs in `docs/design/test-coverage-matrix.md`.

## Target Domain APIs

The method names below are a design inventory, not permission to freeze weak
contracts. Each work package must finalize typed schemas, mode restrictions,
events, revisions, and Python models before marking a method stable.

### 1. Application, settings, and navigation

Expose:

- complete application, mode, active dataset, and readiness summaries;
- auto-contrast settings and fast-object-rendering preferences;
- recent-project listing, forgetting, and clearing;
- back/return-to-project navigation;
- scale-bar visibility and screenshot settings;
- safe close-window and quit requests with dirty-state reporting;
- unsaved-change state and explicit discard/save decisions;
- runtime diagnostics, version/build information, queue statistics, and logs
  suitable for support without exposing secrets.

Candidate resources:

```python
app.application.get_state()
app.application.get_settings()
app.application.update_settings(auto_contrast={...})
app.application.list_recent_projects()
app.application.navigate_back()
app.application.request_close(save="prompt")
```

Close and quit are sensitive capabilities. They require explicit permission and
must never be hidden side effects of closing a Python client.

### 2. Dataset discovery and opening

Provide one typed source model covering:

- OME-Zarr/NGFF;
- TIFF and OME-TIFF;
- samplesheets and project files;
- OME-Zarr folder-tree discovery;
- SpatialData elements;
- Xenium experiments;
- supported remote HTTP/S3 stores; and
- supported annotation/object/point sources.

Expose source inspection before opening, supported element discovery, open
options, credential references, recent-source behavior, and task completion at
actual viewer readiness. Paths and URIs supplied by the caller replace native
file dialogs; the protocol does not attempt to drive OS file pickers.

Candidate resources:

```python
source = app.datasets.inspect(path)
elements = source.list_elements()
task = app.datasets.open(source, element="image", options={...})
task = app.datasets.add_ome_zarr_root(path)
task = app.datasets.open_spatialdata(path, images=[...], shapes=[...])
```

### 3. Projects, samplesheets, and ROIs

Expose the full project workspace lifecycle:

- create an empty project;
- open, validate, save, save-as, and inspect dirty state;
- get and update project metadata;
- import a samplesheet and export a samplesheet;
- add an OME-Zarr discovery root;
- list/add/update/remove/reorder ROIs;
- select ROIs using replace/add/remove/toggle/range/all-visible semantics;
- inspect focused and selected ROIs;
- open one ROI or a selected set as a mosaic;
- move to previous/next ROI with documented wrapping behavior;
- manage segmentation paths and search roots;
- inspect and control object preload/cache behavior; and
- preserve masks, layers, views, groups, analysis presets, and mosaic state.

All ROI operations use stable ROI IDs. Dataset paths remain data, not identity.
Large ROI lists use pagination or iterators with stable ordering and filtering.

### 4. Viewer, coordinates, tools, and multidimensional navigation

Complete viewer control includes:

- camera centre, zoom, viewport size, fit target, and focused target;
- pixel/world/screen/mosaic/ROI coordinate conversion;
- active viewer tool and tool availability;
- current plane, available planes, slice axes, extents, and indices;
- previous/next slice and direct slice selection;
- XY-only capability checks;
- rendered-frame and visible-tile readiness;
- scale-bar visibility; and
- back/focus/fit behavior for images and mosaic ROIs.

The stable coordinate contract must define axis order, pixel centres versus
edges, level-zero coordinates, transform direction, units, and mosaic-local
versus global spaces. Geometry APIs must identify their coordinate space.

### 5. Channels and channel groups

Extend channels to cover:

- stable channel IDs, names, visibility, active state, ordering, and colour;
- contrast get/set/reset/set-max/auto-contrast;
- histograms and intensity statistics as tasks where necessary;
- notes;
- channel-list search/filter/sort presentation state where it is project state;
- previous/next active channel;
- group create/update/delete/reorder and membership;
- group visibility, active group, and colour inheritance;
- channel translation, scale, and rotation;
- bulk atomic presets; and
- single-view/mosaic shared-state semantics.

Python should be able to capture and reapply a complete channel display preset
without reconstructing private project JSON.

### 6. Unified layers and alignment

The current external-layer registry must be joined to a read/write model of all
native viewer layers. `app.layers.list()` should describe image channels,
objects, annotations, points, labels, masks, text labels, and external derived
layers through a tagged union with stable common fields.

Common operations:

- list/get and subscribe to changes;
- get/set active layer;
- show/hide, rename, reorder, and group;
- inspect layer type and available operations;
- get/set opacity, tint/colour, blend/display style, and transform;
- translate, scale, rotate, reset transform, and apply atomic matrices;
- load/reload/replace/clear source;
- export where the layer type supports it; and
- manage session/project/user ownership.

Tagged type-specific state must cover current annotation and feature layers as
well as segmentation objects. This includes annotation source paths and column
mapping, categorical/continuous value modes, category visibility/colour/shape,
continuous ranges, stroke width, point size, feature-series enablement, tint,
and SpatialData/Xenium element identity. NGFF label layers expose their label
group, outline style, transform, readiness, and current renderer limitations.

Type-specific Python wrappers should supplement, not replace, a common layer
base:

```python
layer = app.layers.get(layer_id)
if isinstance(layer, odon.ObjectLayer):
    layer.style.update(fill=True, opacity=0.6)
```

### 7. Saved views and reproducible view state

Expose saved project views as first-class entities:

- list/get/capture;
- validate and preview;
- create/replace/rename/delete/reorder;
- apply to an optional ROI;
- inspect unresolved channel, object, or ROI references;
- convert to/from a documented serializable model; and
- generate a deep-link representation where possible.

Application of a view is a task when it loads data or objects. Completion means
the requested state has settled, not merely that fields were queued.

### 8. Objects, overlays, properties, legends, and selection

Complete the object surface with:

- load/reload/replace/clear sources for Parquet, GeoParquet, GeoJSON, CSV points,
  SpatialData, and Xenium-backed objects;
- source status, geometry type, row count, bounds, transforms, and lazy-property
  readiness;
- property schemas and paged/columnar value access;
- visibility, opacity, outlines, fill, fill opacity, selection opacity, and fast
  rendering;
- single-colour and property-based colouring;
- legend categories, visibility, colour overrides, and reset;
- simple filter clauses plus the complete boolean query language;
- filtered/visible/total counts;
- explicit selection by object ID, rectangle, lasso/polygon, query, or viewport;
- replace/add/remove/toggle selection modes;
- primary selection and focus/zoom to objects; and
- chunked result references for large object-ID or property queries.

Large tables and ID sets must not be returned as unbounded inline JSON. Use
pagination, Arrow/Parquet resources, or bounded iterators selected by measured
workloads.

### 9. Mask layers and polygon editing

Expose masks as structured editable geometry:

- list/create/rename/delete mask layers;
- visibility, active state, display mode, style, offset, and transform;
- list/get/add/update/move/delete polygons;
- bulk replace and batch edits;
- polygon selection and primary polygon;
- import/export GeoJSON;
- clear a mask layer;
- undo/redo with an inspectable history state; and
- explicit project persistence.

Polygon writes validate finiteness, vertex count, closure semantics, coordinate
space, size limits, and revision. A drag performed in the GUI and an atomic
Python polygon update must reach the same Rust edit operation and undo history.

### 10. Threshold-region workflows

Expose the complete threshold workflow:

- choose image channel and pyramid level;
- choose current viewport or complete-image extent;
- set threshold and minimum component size;
- estimate cost and enforce existing safeguards;
- create/refresh/cancel a raster preview task;
- inspect preview metadata and bounds;
- polygonize the preview;
- apply it to a new or existing mask layer; and
- clear temporary previews deterministically.

Preview state is session-owned. Applied masks follow the selected mask-layer
ownership and project persistence rules.

### 11. Analysis, thresholds, mappings, and calls

Expose analysis as typed domain state rather than attempting to manipulate the
Analysis tab:

- list eligible numeric and categorical properties;
- calculate histograms over all/filtered/selected objects;
- select raw or arcsinh transforms;
- create/update/delete threshold rules;
- request quantile and K-means suggestions as tasks;
- select objects from a rule with explicit selection mode;
- map image channels to object properties;
- manage follow-active-channel behavior;
- create/update/delete simple and composite calls;
- calculate and attach call columns;
- save/load/delete call presets;
- mark and clear marker-call failures; and
- inspect warmup/readiness and task progress.

Analysis results include source layer ID, source revision, filter revision, and
parameters so Python cannot mistake a stale calculation for current state.

### 12. Measurements and exports

Expose measurements as retained, cancellable operations:

- inspect available channels, polygon sources, pyramid levels, and target
  counts;
- configure mean or exact median measurements;
- choose all or filtered objects;
- estimate memory/work before execution;
- run with progress and cooperative cancellation;
- attach generated properties to the source object layer;
- inspect past session runs and result provenance; and
- export enriched GeoParquet or CSV with selected properties, calls, measured
  values, selection columns, and geometry policy.

All exports require an explicit destination, overwrite policy, and atomic-write
behavior. The returned result includes the final path, rows, columns, bytes,
source revisions, and warnings.

### 13. Mosaic state and navigation

Complete mosaic control includes:

- inspect all mosaic items, metadata fields, positions, sizes, and focus;
- select mosaic ROIs using the same selection modes as the project browser;
- fit all, fit/focus one, previous/next, and clear focus;
- choose fit-cells or native-pixels layout;
- set column count, group field, primary/secondary sort, gaps, and group labels;
- configure text-label columns and style;
- inspect unresolved/missing metadata values;
- control shared channel state;
- load/inspect per-ROI objects; and
- control mosaic-specific memory pinning.

Layout application should be atomic and return the normalized layout state and
revision.

### 14. Memory, tiles, and performance controls

Expose operational controls without publishing renderer internals:

- loading state and visible/coarse/fine tile readiness;
- memory usage and estimates by dataset, ROI, channel, and pyramid level;
- available pin targets;
- pin selected channels/levels for the focused ROI or all mosaic ROIs;
- task progress, partial failures, and cancellation;
- unpin/unload controls; and
- configured limits and warning/danger classifications.

Clients should not receive raw cache objects. Snapshots describe semantic
targets, estimates, status, and outcomes.

### 15. Screenshots, reports, deep links, and output

Complete output control includes:

- viewer, window, and project screenshots;
- size, scale, background, overlays, scale-bar, and output settings supported by
  native Odon;
- readiness policies such as current-frame versus wait-for-visible-tiles;
- deterministic metadata/provenance sidecars where requested;
- deep-link parse/validate/apply/generate operations;
- explicit reporting of unsupported or ambiguous deep-link fields; and
- common atomic output and overwrite policies.

Screenshot tasks complete only after the output is durably written or return a
managed resource when no path is supplied.

### 16. Declarative UI and extension lifecycle

Retain the bounded Rust-rendered component model. Stabilization requires:

- a versioned component and host schema;
- stable values, actions, bindings, event policies, and validation errors;
- complete introspection of limits and supported native command actions;
- extension trust/permission UI;
- documented disconnect and reconnect behavior;
- a decision on whether any contribution manifests can persist in projects;
- safe removal/reset diagnostics; and
- accessibility and keyboard behavior for the stable vocabulary.

Do not make arbitrary egui code, Python callbacks on the UI thread, CSS, or
unbounded custom drawing part of the stable surface.

## Python SDK Shape

Use parallel synchronous and asynchronous resource trees:

```python
app.application
app.datasets
app.projects
app.viewer
app.viewer.planes
app.channels
app.layers
app.objects
app.masks
app.analysis
app.measurements
app.mosaic
app.memory
app.screenshots
app.deep_links
app.data
app.tasks
app.events
app.ui
```

Requirements:

- Sync and async APIs have the same domain names and data models.
- Public snapshots use dataclasses or typed immutable models, not undocumented
  dictionaries, once their stable contracts are chosen.
- Entity handles retain IDs, expose `refresh()`, and accept revision guards.
- Bulk operations accept iterables but enforce negotiated limits.
- Large results return paged iterators or referenced resources.
- Local validation catches obvious shape/type/coordinate errors, while Rust
  remains authoritative.
- `app.call()` remains available for experimental methods and diagnostics, not
  as the normal route for a stable feature.
- Methods expose availability and required capabilities through introspection.
- Documentation labels each resource and method experimental, provisional, or
  stable.

Example target workflow:

```python
with odon.connect() as app:
    project = app.projects.open("study.odon").wait()
    app.projects.rois.select(query="cohort == 'validation'", mode="replace")
    app.projects.open_selected_as_mosaic().wait()

    app.channels.apply_preset({
        "visible": ["DAPI", "PanCK"],
        "colours": {"DAPI": "#4f6fff", "PanCK": "#ffcc33"},
    })

    cells = app.objects.active()
    cells.filter.set("area > 100 and phenotype == 'tumour'")
    cells.selection.select_query("Ki67 > 0.5", mode="replace")

    run = app.measurements.run(
        objects=cells,
        channels=["PanCK"],
        statistic="mean",
        filtered_only=True,
    )
    run.wait()
    app.exports.objects(cells, "enriched.geoparquet", include_selection=True).wait()
```

## Parity Manifest and Coverage Enforcement

Create a checked-in machine-readable manifest, for example
`api/application-surface.toml`, with one record per capability family or
operation:

```toml
[[capability]]
id = "MASK-01.delete-polygon"
native_entry_points = ["canvas.delete", "context-menu.delete", "properties.delete"]
query = "viewer.masks.polygons.get"
command = "viewer.masks.polygons.remove"
event = "viewer.masks.polygons.removed"
python_sync = "odon.MaskPolygon.remove"
python_async = "odon.AsyncMaskPolygon.remove"
permission = "viewer.masks.write"
tests = ["rust:mask_delete_uses_shared_edit", "python:test_mask_delete_round_trip"]
stability = "provisional"
```

CI validates:

- IDs are unique and reference the existing feature inventory/matrix;
- every registered stable method appears in the manifest;
- every stable mutating method declares an event or documents why none is
  appropriate;
- every stable task declares completion semantics and cancellation behavior;
- sync and async Python wrappers exist for every stable Python capability;
- MCP exposure is explicit rather than accidental;
- required tests exist; and
- removed or renamed stable entries follow the compatibility policy.

When a new native feature is added, its pull request must either add semantic
control coverage or explicitly mark it as presentation-only with review. This
is the mechanism that keeps “complete” true after the initial project.

## Events and State Synchronization

Define event families for all target domains:

- `application.*`, `settings.*`, and `navigation.*`;
- `datasets.*`, `project.*`, and `project.rois.*`;
- `viewer.camera.*`, `viewer.planes.*`, `viewer.tools.*`, and
  `viewer.readiness.*`;
- `viewer.channels.*`, `viewer.groups.*`, and `viewer.layers.*`;
- `viewer.objects.*`, `viewer.selection.*`, and `viewer.filters.*`;
- `viewer.masks.*` and `viewer.history.*`;
- `viewer.thresholds.*`, `viewer.analysis.*`, and `viewer.measurements.*`;
- `mosaic.*`, `memory.*`, `exports.*`, and `screenshots.*`; and
- existing `tasks.*`, `data.resources.*`, `ui.extensions.*`, and
  `ui.contributions.*`.

Add a lightweight state-synchronization helper to the SDK:

1. fetch a typed snapshot and global revision;
2. subscribe from the returned event sequence;
3. apply entity deltas where supported;
4. detect dropped events or revision gaps; and
5. refresh the affected snapshot.

This supports long-running dashboards without requiring them to refetch the
entire application state on every change.

## Security and Ownership

Expand capabilities by domain and operation:

- `.read`, `.write`, `.execute`, and `.export` where appropriate;
- separate application-close/quit, filesystem output, project persistence, and
  credential-use permissions;
- extensions request the minimum set and receive the granted set;
- a future Odon trust dialog identifies the extension, requested capabilities,
  package/version, and session/persistence scope;
- paths are canonicalized and output overwrite is explicit;
- bearer tokens, credentials, and environment secrets never appear in normal
  snapshots, events, diagnostics, provenance, or logs; and
- session cleanup deletes only Odon/Python-managed temporary resources.

The initial loopback-only transport remains appropriate. Remote control is a
separate security design and is not required for complete local application
control.

## Data Size and Performance Rules

- Keep command metadata and small state inline in JSON.
- Paginate ROI lists, properties, tasks, and history when they can grow large.
- Return object IDs inline only below a negotiated threshold.
- Use Arrow/Parquet/OME-Zarr or managed resources for large data.
- Coalesce camera, progress, and continuous-value events.
- Never block egui on Python callback execution.
- Measure control latency, serialization cost, task throughput, event loss, and
  repeated preview replacement before choosing shared memory.
- Add binary/shared-memory transport only for demonstrated bottlenecks and only
  with explicit shape, dtype, strides, endianness, ownership, acknowledgement,
  and cleanup semantics.

## Implementation Roadmap

### Phase A: Parity inventory and contract foundation

Deliverables:

- Add the machine-readable application-surface manifest.
- Map every feature ID in `docs/design/test-coverage-matrix.md` to native entry
  points and current control coverage.
- Choose canonical hierarchical names and mark flat aliases deprecated.
- Define stable identity, snapshot, revision, paging, output, and availability
  models.
- Add registry checks for schemas, aliases, permissions, events, tasks, and
  Python parity.
- Refactor native UI mutations onto shared domain operations where necessary.

Exit criteria:

- Every supported native capability is classified as semantic or
  presentation-only.
- Every semantic capability has an owner and target phase.
- No new GUI-only semantic mutation can enter unnoticed in CI.

### Phase B: Viewer, planes, channels, and unified layers

Coverage IDs: `CAM-*`, `CHAN-*`, `PLANE-*`, `LAYER-*`, `LABEL-01`, `VIEW-*`.

Deliverables:

- Complete camera/coordinate/tool/readiness contracts.
- Add plane and slice inspection/navigation.
- Complete channel colour, notes, contrast operations, groups, presets, and
  transforms.
- Unify native and external layer inspection and mutation.
- Add saved-view-independent complete viewer snapshots and events.
- Add sync/async Python wrappers and MCP mappings selected for AI use.

Exit criteria:

- A Python script can reproduce every normal viewing and layer-alignment action
  available in single-view mode.
- Native UI and Python produce identical state transitions and events.

### Phase C: Projects, data sources, saved views, and mosaic

Coverage IDs: `PROJ-*`, `SHEET-*`, `MOS-*`, `DATA-*`, `LINK-*`.

Deliverables:

- Complete project create/edit/save-as, metadata, ROI, samplesheet, discovery
  root, and cache APIs.
- Add typed SpatialData/Xenium/source discovery and open options.
- Add saved-view CRUD/application.
- Complete mosaic selection, layout, navigation, labels, and state.
- Centralize deep-link application and generation through domain commands.

Exit criteria:

- Python can build a project from sources, curate ROIs, save it, reopen it, open
  single and mosaic views, and reproduce all project/mosaic view state.

### Phase D: Objects, properties, styling, filters, and selection

Coverage IDs: `OBJ-*`, `FILT-*`, `SEL-*`, relevant `DATA-*`.

Deliverables:

- Complete object source lifecycle and readiness.
- Add property schema/paged access and referenced large results.
- Add full style, legend, and fast-rendering control.
- Complete filter clauses and boolean query models.
- Complete selection by IDs, geometry, query, and viewport with all modes.
- Add focus and selection-driven events.

Exit criteria:

- Every object and overlay property available in the native Properties tab can
  be inspected and changed semantically from Python.
- Large tables and selections remain bounded.

### Phase E: Masks, history, and threshold regions

Coverage IDs: `MASK-*`, `THRESH-*`, relevant `LAYER-*`.

Deliverables:

- Add mask-layer and polygon CRUD with stable IDs and revisions.
- Route GUI edits and Python edits through the same undoable operations.
- Add selection, import/export, transforms, display modes, and persistence.
- Add threshold configuration, preview tasks, polygonization, and apply-to-mask.

Exit criteria:

- A Python script can create the same mask project state as the complete native
  draw/edit/import/threshold workflow.
- Mixed GUI/Python edits have a coherent undo history and conflict behavior.

### Phase F: Analysis, measurements, and exports

Coverage IDs: `ANALYSIS-*`, `MEAS-*`, `EXPORT-*`.

Deliverables:

- Add histogram/transformation/threshold/suggestion operations.
- Add channel-property mappings, calls, composites, presets, and failures.
- Add retained measurement runs, provenance, attachment, and result inspection.
- Add atomic GeoParquet/CSV exports with explicit schemas and overwrite policy.

Exit criteria:

- Python can reproduce the supported Analysis and Measurements tabs without
  reading or mutating private Rust/project state.
- Results identify their source revisions and remain reproducible.

### Phase G: Memory, screenshots, settings, and lifecycle

Coverage IDs: `PERF-*`, screenshot portions of `MCP-04`, `VIEW-01`, `LINK-05`,
and application settings/lifecycle additions to the inventory.

Deliverables:

- Add memory estimates, pin/unpin/unload tasks, and loading summaries.
- Complete screenshot settings and readiness policies.
- Add application settings, recent projects, scale bar, and navigation.
- Add guarded close/quit and dirty-state decisions.
- Add packaged cross-platform smoke tests.

Exit criteria:

- All meaningful menus, top-bar actions, Memory controls, and output settings
  have semantic equivalents.
- Sensitive lifecycle actions are permissioned and explicit.

### Phase H: Ecosystem validation and stable-v1 declaration

Deliverables:

- Build at least three structurally different reference clients: an interactive
  notebook workflow, Cellpose or another layer-producing analysis extension,
  and a project/object review or measurement extension.
- Run cross-version compatibility tests for current and previous supported
  Odon/SDK combinations.
- Complete macOS, Windows, and Linux packaged tests.
- Publish compatibility, deprecation, and support policies.
- Freeze only the proven protocol, snapshot, component, and Python subsets.
- Remove or schedule removal of deprecated flat aliases.

Exit criteria:

- The parity manifest contains no unexplained GUI-only semantic capability.
- Every stable manifest record has Rust, protocol, sync Python, async Python,
  event, error, and compatibility evidence as applicable.
- Protocol v1 and `odon-client` 1.0 compatibility promises are documented.

## Testing Strategy

### Rust domain tests

- Test shared domain operations independently of egui.
- Assert success, validation, wrong-mode, not-ready, conflict, permission, and
  resource-limit behavior.
- Verify native UI and protocol adapters invoke the same operation.
- Test transactional failure leaves state unchanged.
- Use deterministic fixtures for coordinates, planes, geometry, analysis,
  measurements, and project round trips.

### Registry and schema tests

- Validate every registered method and alias.
- Reject unknown fields for stable typed methods.
- Snapshot generated request/result/event schemas.
- Check task and event declarations.
- Check manifest and registry completeness in both directions.

### Protocol integration tests

- Authenticate and negotiate versions/capabilities.
- Exercise concurrent requests and out-of-order responses.
- Verify revision conflicts between native UI and multiple clients.
- Verify bounded events, sequence-gap detection, and resynchronization.
- Verify task progress, timeout, cooperative cancellation, and retained results.
- Verify disconnect cleanup and ownership boundaries.

### Python contract tests

- Run equivalent sync and async workflows.
- Test typed model validation and exception mapping.
- Test every stable entity handle and collection.
- Test pagination and referenced large results.
- Test event-driven state mirrors and reconnection.
- Run against an in-process protocol fixture and a real packaged Odon smoke
  target.

### End-to-end parity scenarios

At minimum automate:

1. Build a project from a samplesheet, select ROIs, save, reopen, and open a
   mosaic.
2. Configure planes, channels, colours, groups, transforms, camera, and a saved
   view; reproduce the state from a fresh session.
3. Load objects, set style and legend, filter, select by geometry/query, measure,
   and export.
4. Create/import/edit/undo/export masks and apply a threshold preview.
5. Run analysis thresholds and composite calls, save a preset, and attach
   results.
6. Pin and unpin memory targets while observing progress and limits.
7. Mix native UI actions and Python mutations and verify events, revisions, and
   final state agree.
8. Disconnect Python during work and verify Odon remains responsive and cleans
   only session-owned state.

Visual/GPU output, native windows, URL registration, and packaged file dialogs
remain extended platform tests, but their semantic state and commands are
covered in the required suite.

## Compatibility and Migration

- The current experimental client remains usable while domains migrate.
- Canonical hierarchical methods are introduced alongside generated flat
  aliases.
- Introspection marks aliases deprecated and reports replacement names.
- Python high-level wrappers move to canonical methods first.
- MCP tools keep user-friendly names but map to canonical registry entries.
- No stable method changes request/result meaning without a protocol-major
  change; additive optional fields follow the published policy.
- Snapshot models declare which fields are stable and retain an extension field
  only where forward-compatible use is deliberate.
- Cross-version fixtures retain at least the previous supported SDK and Odon
  release pair.

## Documentation Deliverables

- Generate a Python API reference from the typed models and registry.
- Add capability recipes for notebook control, project construction, channel
  presets, object review, masks, measurements, and custom UI.
- Publish a native-feature/Python-equivalent table from the parity manifest.
- Document mode restrictions and availability checks next to each method.
- Document task completion and cancellation semantics per operation.
- Provide versioned extension templates and packaging examples.
- Keep `docs/reference/python-api.md` focused on supported user workflows and
  link the exhaustive generated reference separately.

## Risks and Mitigations

### Completeness creates an unmaintainable API

Expose stable semantic domain operations, not every widget field. Group related
state into typed snapshots and bulk mutations, and stabilize in phases after
real use.

### Native behavior and control behavior drift

Route both through shared domain operations and make parity-manifest validation
part of CI.

### Snapshots mirror private Rust state

Design public models around user concepts and stable IDs. Translate in an
application adapter rather than serializing internal structs directly.

### Large object and image results overwhelm JSON

Use paging and referenced resources, enforce limits, and measure before adding
binary or shared-memory transports.

### Mixed GUI/Python edits overwrite one another

Use entity/global revisions, transactions, semantic events, and resynchronizing
state mirrors.

### Long analysis blocks Odon

Run work off the UI thread, expose retained tasks, update Rust-owned progress,
and define cooperative cancellation boundaries.

### Full control increases local security impact

Use granular capabilities, an explicit trust UI, canonical paths, overwrite
policies, visible extension diagnostics, and special protection for persistence
and application lifecycle operations.

### The plan freezes the wrong abstraction too early

Keep new domains provisional through the parity scenarios and three reference
clients. Declare stable subsets domain by domain rather than stabilizing every
new method simultaneously.

## Stable-v1 Acceptance Gate

The complete application API is ready for a stable declaration only when:

- all existing semantic capabilities in `docs/odon-feature-inventory.md` are in
  the parity manifest;
- every semantic capability is either supported or has a documented native
  limitation that applies equally to GUI and API;
- there are no direct MCP-only or Python-only application execution paths;
- native UI and remote clients share domain operations;
- stable methods use canonical names, typed models, stable IDs, structured
  errors, revisions, and documented availability;
- long operations have verified completion and cancellation semantics;
- all mutable stable domains emit sufficient events for resynchronization;
- sync and async Python APIs have verified parity;
- large data and query results remain bounded;
- ownership, cleanup, persistence, and overwrite behavior are tested;
- at least three real workflows validate the abstractions;
- packaged behavior is exercised on macOS, Windows, and Linux; and
- compatibility and deprecation policies are published.

At that point “Python can completely control Odon” will be a testable product
claim rather than an architectural aspiration.
