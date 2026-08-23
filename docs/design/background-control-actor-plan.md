# Background-Safe Central Model and Control Actor Plan

Status: stabilization and acceptance in progress. All 264 registered application methods are now
actor-routed in every supported mode and target combination, leaving zero legacy application
routes. The legacy request channel, canvas-deferred semantic queue, and
`RootApp::reply_to_control_request` dispatcher have been removed. Alternate object resources,
retained compute, mosaic state, and presentation capture use actor-owned state, bounded workers,
or explicit presentation acknowledgements. Renderer integration is isolated behind projection
and native-observation adapters. Direct remote compatibility mutations and native before/after
snapshot translators are gone. Project and mosaic UI actions now emit typed actor commands at
their interaction points, including project creation, metadata, ROI and saved-view edits,
samplesheet/discovery work, TIFF opening, selected-ROI mosaic opening, mosaic layout/focus/camera,
and shared mosaic presentation. Viewport, mask, and native-layer commits likewise originate
directly at their interaction points, including worker-backed mask append-and-reload persistence
and revision-checked layer gesture transactions.

The control actor is organized as a small façade with dedicated runtime, dispatch, worker,
completion, projection, diagnostics, domain, and test modules. The former 27,951-line `src/app.rs`
is a responsibility-based `src/app/` tree. The canonical `AppModel` is likewise split into
construction, runtime, dispatch, project, compute, resource, object, mask, layer, saved-view, and
viewport modules; `src/model/app.rs` now holds shared types and pure helpers rather than a second
13,420-line implementation monolith. `RootApp` no longer contains the application command surface,
and its projection, remote-dialog, and tests are separate modules with structural size guards. The
transport-independent `OdonControlRequest` envelope now belongs to `src/control`, the required
local `OdonControlRuntime` is isolated in `src/mcp/runtime.rs`, and
`src/mcp/bridge.rs` is a small façade for connection-scoped shared state rather than actor
construction or renderer queues. The bridge implementation is split into TCP listener/framing and
worker queues (`bridge/transport.rs`), JSON-RPC and actor-mailbox dispatch (`bridge/dispatch.rs`),
batch and retained-task orchestration (`bridge/tasks.rs`), background readiness monitors
(`bridge/waits.rs`), and task/UI/event protocol registries (`bridge/services.rs`), with tests in
`bridge/tests.rs`. This is a source boundary only: the wire protocol and actor mailbox are
unchanged.
Remaining release work is cumulative regression evidence, performance evidence, and real platform
occlusion acceptance, not application-method migration.

The multi-ROI application shell follows the same rule. The former 8,024-line `src/mosaic/mod.rs`
is now a compact state/type façade with separate control, construction, lifecycle,
memory/navigation, layer/dialog, panel, canvas, tile, I/O, and test modules. Mosaic command
semantics remain actor-owned; this split only makes renderer and transient UI responsibilities
reviewable.

Object-layer renderer integration is no longer concentrated in the former 6,758-line
`src/objects/core.rs`. Its façade retains shared adapter/export types and small geometry helpers;
runtime loading, source formats, properties/presentation UI, selection/filtering, exports, and
tests live under `src/objects/core/`. Actor-side object resources remain immutable and shared; this
split organizes the renderer consumer and does not move canonical object semantics out of the
actor.

The custom threshold compatibility panel is split along its blocking-I/O boundary as well:
`custom/cell_thresholds.rs` owns panel state and renderer projection,
`cell_thresholds/data.rs` owns parquet loading and dataset/marker inference, and
`cell_thresholds/threshold_files.rs` owns CSV/JSON threshold parsing. These renderer-side adapters
do not replace the canonical actor analysis/resource path.

The former 4,654-line `src/project/space.rs` is also responsibility-split. Versioned project/view
DTOs and pure matching helpers remain in the façade; actor projection and persistence, ROI/view
state, browser UI, saved-view dialogs, samplesheet/discovery workflows, and tests live under
`src/project/space/`. Project semantics still enter the shared typed command outbox before any
renderer mirror changes.

Date: 2026-08-23

Target: make Python, MCP, native UI, and future clients operate on one canonical
Rust application model that continues to execute while Odon's window is
backgrounded, covered, occluded, or temporarily unable to paint

Related documents:

- `docs/design/python-api-plan.md`
- `docs/design/complete-application-python-api-plan.md`
- `docs/design/control-protocol-v1.md`
- `docs/design/multi-viewport-plan.md`
- `docs/design/layout-tree-milestone-plan.md`
- `docs/design/test-coverage-matrix.md`
- `docs/testing/background-control-acceptance.md`

## Implementation Status

The background-safe first vertical slice is now implemented in
`src/control/actor/` and `src/model/app.rs`:

- the protocol bridge submits commands to a dedicated control actor;
- native UI and TCP clients construct the same transport-independent control request envelope;
- local actor construction and renderer projection handles live in a dedicated required-runtime
  module, while TCP binding, authentication, JSON-RPC dispatch, and connections live in the bridge;
- local OME-Zarr metadata/storage opening runs on a worker and installs a
  generation-checked document model without a UI frame;
- viewport layout, titles, links, logical geometry, camera, plane, channel
  visibility/color/contrast/notes/transforms/order/groups/search presentation, side-panel state,
  per-viewport object appearance/property/legend presentation, and rendering preferences execute
  in the actor;
- the corresponding active-view compatibility methods (`viewer.camera.*`,
  `viewer.planes.*`, core `viewer.channels.*`, and smooth-pixel control) use the
  same actor state rather than falling back to a GUI frame;
- renderer state is published through a bounded latest-value projection, so
  covered-window changes coalesce and `RootApp::update` applies the final state
  atomically when frames resume;
- renderer observations are tagged with the projection revision they consumed and can update
  only renderer-owned compatibility fields; a delayed frame cannot write channel, viewport, or
  panel state back over a newer actor transaction;
- model/resource/geometry readiness is distinct from presentation readiness;
- the renderer reports measured viewport geometry and consumed projection
  revisions back to the actor; and
- a TCP regression executes the complete two-view comparison with no UI queue
  drain or frame tick;
- dataset opens use a bounded two-worker pool rather than spawning an
  unbounded thread per request; and
- local dataset inspection and OME-Zarr opening share source-neutral typed document and inspection
  contracts, with native adapters kept outside renderer/GPU construction; and
- channel intensity statistics use that resource-worker boundary and cannot block the actor or UI
  thread; actor-owned requests arriving during a dataset transition return explicit `NOT_READY`
  rather than silently falling through to the UI dispatcher; and
- project creation, metadata, ROI navigation/editing, saved-view CRUD/capture and resource-ready
  apply, project open, and project
  save execute against the canonical actor model; open/save filesystem work runs in the bounded
  worker pool with generation checks and complete version-6 state round trips;
- project samplesheet inspection/validation/import/export and recursive OME-Zarr discovery run in
  the bounded worker pool; imports and discovery commit complete ROI collections through
  generation-checked actor transactions without a render frame;
- referenced data resources and external layers are serialized through the actor, including
  registration, queries, visibility/style updates, order, removal, project-manifest persistence,
  and immediate save-after-layer ordering; project open installs its manifest before replying, so
  a delayed renderer frame cannot replace newer resource state; and
- primary object sources are parsed once on bounded workers into an immutable actor index plus an
  opaque renderer-native preload; source load/reload/clear, property schema/value access, and
  per-viewport simple or typed-query filters complete and report accurate counts without a frame;
  the renderer installs the shared preload without parsing the source a second time; and
- mask layer/polygon CRUD, selection, undo, atomic native commits, GeoJSON import/export, and
  in-memory project synchronization are actor-owned; imports/exports use the bounded worker pool,
  native edits carry generation conflict checks, and the renderer applies the latest versioned
  mask projection only when frames resume; and
- primary-object rectangle/lasso/view queries, shared selection identity, ID/filter selection,
  focus, and native selection commits are actor-owned; immutable polygon/point geometry stays in
  the shared object resource, standalone filter selection runs on a bounded worker, and selection
  projections contain only indices plus a generation; and
- native rectangle, lasso, clear, click, and point-to-object ID selection controls submit
  target-aware actor commands before renderer state changes; click transactions retain exact
  renderer hit-testing semantics and are rejected if their consumed selection generation is stale;
  and
- Odon-native channel/overlay inventory, order, active layer, visibility, presentation, and
  translation are actor-owned for single-image viewports; compatibility renderers may contribute
  newly discovered descriptors but cannot overwrite existing actor presentation, and native UI
  changes commit as one presentation-revision-checked viewport transaction; and
- local NGFF label discovery, metadata loading, visibility, and unload are actor-owned; label
  metadata opens on bounded workers without requiring a GPU or frame, while the renderer creates
  its tile loader from the shared generation-checked resource when presentation resumes; and
- operation readiness is keyed by kind and optional scope, so concurrent viewport filters and
  mask import/export work cannot complete, fail, or cancel one another accidentally; and
- settings, recent-project mutations, and save/discard/cancel lifecycle decisions are actor-owned;
  persistence uses the bounded worker pool, while the actor emits only a narrow close/quit platform
  effect for the UI thread to execute; and
- deep-link parsing, resolution against current/external/example projects, filter extraction,
  generation, and transactional application are frame-independent; application opens the target
  project/ROI/resources and commits its complete view atomically with generation and cancellation
  guards; and
- remote S3 session management, S3 browsing, and HTTP/S3 OME-Zarr opening are actor-owned; remote
  I/O runs on bounded workers, credentials remain in an actor-private redacted service, and
  clearing a session invalidates dependent listings and opens; and
- TIFF, SpatialData, and Xenium primary-document discovery and opening are actor-owned; TIFF
  projections retain the worker-prepared pyramid so resuming presentation does not reparse TIFF
  metadata, while typed secondary-resource preparation for SpatialData/Xenium remains in Wave 2D;
  and
- retained task creation, polling, progress, cancellation, completion, failure, and forgetting are
  ordered through the actor-owned task service while transport remains a thin client; and
- actor implementation code is split by responsibility and domain: the façade contains no command
  handlers, worker execution and result commitment are separate, completion dispatch is domain-
  aware, tests are split by behavior, and actor-capability metadata is registry-owned; and
- native macOS project Open, Save, and Save As menu actions enqueue the same actor commands as
  remote clients, after any semantic native edits collected in the current frame; and
- diagnostics expose actor health, per-method actor/control-service routing, queue wait,
  model time, reply time, projection coalescing, worker counts, and
  presentation wait separately. The method catalog now declares execution
  class and readiness requirements.

There is no compatibility dispatcher or production legacy semantic route. Native UI commits use
typed actor commands; renderer observations are limited to native-origin deltas and immutable
presentation state. The former renderer-snapshot event publisher has also been removed: canonical
actor commands are now the only source of semantic protocol events. Cross-platform manual
covered/minimized/Space acceptance remains outstanding.

As of this revision, 279 commands are listed as actor-capable: all 264 application-registry
methods, including `app.get_method_availability`, plus 15 of the 34 protocol-service methods. The
other 19 protocol services (handshake, event/task management, and declarative-UI registries) are
control-service operations that already execute outside the render loop and are not application
semantic routes.

## Updated Execution Plan (Authoritative)

The original phases later in this document describe how the architecture was introduced. They
remain useful historical context, but they are no longer the implementation schedule. The plan in
this section supersedes their sequencing and defines completion for the current worktree.

### Two completion gates

This project has two separate success conditions. Neither may be substituted for the other.

**Gate A — original backgrounding defect:**

- local OME-Zarr open reaches model and resource readiness without a frame;
- the complete `two_viewer_comparison.py` workflow finishes while frames are paused;
- Python does not wait for Odon to become visible;
- returning to Odon consumes the latest projection and shows the requested final state without
  another Python call; and
- visible, covered, minimized, and separate-Space behavior is recorded on macOS, with equivalent
  covered/minimized checks on Windows and Linux where supported.

Automated actor/TCP evidence covers no-frame semantic execution and resumed projection state. The
literal Python SDK workflow under each operating-system window condition must still be run and
recorded before Gate A is declared complete.

**Gate B — complete controllable application surface:**

- every registered application method has a mode- and target-aware execution owner;
- every semantic or resource method executes in the actor or a bounded worker without a GUI tick;
- every pixel-dependent method is an actor-owned presentation task with an explicit waiting state;
- every excluded method is documented as intentionally UI-transient and is not represented as a
  remotely controllable semantic operation;
- native UI, Python, MCP, menus, and deep links commit through the same command boundary; and
- the legacy semantic dispatcher and duplicate semantic ownership have been removed.

### Current method ledger

The registry contains 264 application methods, and all 264 are present in the actor-capable list.
The route-matrix audit evaluates every supported mode and target variant and reports zero
`legacy_ui` or hybrid routes. Pixel capture methods are included: they enter an actor-owned
presentation task and wait for a generation-specific renderer acknowledgement without blocking
the actor mailbox.

The actor-capability count still must not be used as the only completion signal. Registry, route,
no-frame execution, resumed projection, persistence, and platform-occlusion evidence are recorded
separately.

### Non-negotiable implementation invariants

Every remaining migration must preserve these rules:

1. A semantic request never replies merely because work was queued for `RootApp::update`.
2. The actor never waits synchronously for a GUI frame, filesystem operation, network request,
   image decode, table scan, inference run, export, or GPU operation.
3. Workers return immutable, generation-tagged results; only the actor commits them.
4. Once a field is migrated, the actor is its only canonical owner. Renderer observations cannot
   overwrite it, even if they were produced from a newer wall-clock time but an older projection.
5. Native UI changes use the same typed command as remote clients. Optimistic renderer-local state
   is permitted only for latency-sensitive gesture previews and must be revision-tagged.
6. Model/resource completion and presentation completion are distinct public states.
7. Heavy resource payloads are shared by stable `Arc` handles and generations, never copied into
   JSON projections or cloned for every camera change.
8. Command, worker, event, and presentation queues remain bounded and expose backpressure.
9. Successful response shapes, events, revision behavior, task cancellation, and persistence are
   characterized before migration and preserved unless an intentional correction is documented.
10. A method is counted as complete only after all supported modes and target variants have either
    passed actor-path tests or received an explicit, justified exclusion.

### Foundation work required before broadening the surface

The actor foundation works for the current vertical slice, but four pieces must be strengthened as
part of the next migrations rather than postponed to final cleanup.

#### Mode- and target-aware routing

Current state: complete as a foundation. The registry now evaluates routes by mode and, where
relevant, target/parameter variant; actor dispatch uses that evaluator rather than a duplicate
list. Introspection reports the matrix while retaining the old flat summary for compatibility.
Remaining work is coverage, not another routing design: each migration must add its route cases,
and the parity manifest, diagnostics, generated Python documentation, and tests must continue to
consume this one evaluator.

#### Operation-specific readiness

Replace the broad `resources_busy` boolean with an operation table or typed readiness submodels.
At minimum, track independent generations and phases for:

- document opening;
- project persistence;
- labels;
- primary objects and object filters;
- masks;
- threshold preview;
- analysis warmup and histogram work;
- measurements;
- exports;
- mosaic resources; and
- presentation outputs.

One operation finishing must never mark another concurrent operation ready. `app.get_loading_state`
may retain aggregate compatibility fields, but it must also expose the individual operations that
produce those aggregates.

Current state: the broad mutable busy/status pair has been replaced by kind-and-scope operation
keys for document open, project I/O, labels, primary objects, per-viewport/selection object
filters, mask I/O, and settings I/O. Generation checks prevent stale completion from replacing a
newer operation. New kinds must be added alongside each remaining domain—dataset inspection,
remote listing, preloading, threshold work, analysis, measurements, exports, mosaic resources, and
presentation outputs—rather than reintroducing a shared busy flag.

#### Actor-owned retained tasks

Task creation, progress, cancellation, worker completion, and final result installation need one
ordering authority. A task records its command, required readiness dimensions, source generations,
phase, progress, cancellation state, result, and failure. Cancellation must remain responsive while
another worker is busy and stale completions must be harmless.

Current state: complete as a foundation. Canonical task mutations are submitted through a bounded
actor-owned task-service mailbox; transport polling/subscription is a thin client, and the actor
continues accepting cancellation and unrelated semantic requests while a worker is blocked. The
remaining compute waves must add cancellation-before-start and queued-completion race coverage for
each new retained task family rather than reintroducing transport-owned mutation.

#### Actor source organization

Current state: complete as a prerequisite. The actor remains one runtime ordering authority, but
its source is no longer a monolith. `src/control/actor/mod.rs` is a small public façade; mailbox
runtime, dispatch, worker jobs, worker execution, projections, diagnostics, common request helpers,
domain preparation, and domain completion are separate modules. Tests are split by behavior under
`src/control/actor/tests/`. Actor-capable method metadata lives under the registry and has a
uniqueness/known-method audit. New migrations must place preparation and completion logic in their
domain modules and must not grow a new central dispatch or completion match.

The bounded resource-worker runtime follows the same rule. `control/actor/worker.rs` owns worker
thread creation and the single `LoadJob` dispatch loop; memory, thresholds, analysis, measurements,
project/ROI loading, and screenshot/label computations live in domain modules under
`control/actor/worker/`. This changes neither queue topology nor completion ordering, and a source
guard prevents domain computation from accumulating in the dispatch loop again.

The command registry is likewise one authoritative API surface without being one source-file
monolith. `src/control/registry.rs` defines shared descriptor types and lookup/introspection
behavior; `registry/catalog.rs` owns application descriptors; `protocol_catalog.rs` owns protocol
services and aliases; `schema.rs` owns generated request schemas; and actor-capability metadata and
tests have dedicated modules. A structural guard caps the façade and production modules and keeps
catalog data, schemas, and tests from accumulating in the façade again.

Typed command decoding is a separate companion boundary. `src/control/command.rs` now retains only
the public `ControlCommand` envelope, descriptor resolution, revision extraction, and service
method classification. Serde request DTOs and defaults live in `control/command/requests.rs`,
method-specific validation lives in `requests/validation.rs`, and characterization tests live in
`command/tests.rs`. Registry metadata remains authoritative; the split does not duplicate the
method catalog or introduce a second decoder.

#### Application source organization

Current state: complete as a prerequisite. The former `src/app.rs` mixed shared types, 184 fields,
frame orchestration, rendering, blocking compatibility workflows, control handlers, and 42 inline
tests in 27,951 lines. It is now `src/app/mod.rs` plus responsibility modules for construction,
projection, lifecycle, datasets, projects, viewports, canvas/rendering, layers, selection,
thresholds, memory, screenshots, and tile scheduling. Frame-driven compatibility handlers are
isolated as renderer snapshots/projection adapters under `src/app/renderer_bridge/`, and app tests
are split by behavior under `src/app/tests/`. Direct compatibility mutation helpers retained for
characterization are guarded by `cfg(test)` and are absent from production builds.

This is a source-boundary refactor, not a claim that the aggregate app state is already correctly
owned. `docs/design/app-state-ownership-inventory.md` classifies every field and gives its deletion
wave. Structural tests keep the façade and responsibility modules below generous size ceilings,
keep the `eframe::App` lifecycle out of the façade, and preserve the actor façade boundary. New
actor waves must delete their corresponding legacy handlers and compatibility fields rather than
merely adding more modules.

The viewport canvas now has an explicit interaction/render boundary. `app/canvas.rs` retains the
hard per-viewport clip, tile scheduling, ordered overlay painting, HUD, and capture sequencing;
`app/canvas/interactions.rs` owns camera, selection, mask, layer-move, and transform gestures and
returns the one camera-motion signal consumed by the paint phase. This keeps the clip and draw
order auditable while preventing gesture mutation from being buried in the render pipeline.

Native-layer renderer integration is likewise source-separated beneath the tiny
`app/layer_runtime.rs` façade. Dedicated modules own typed command submission, quick-contrast UI,
layer metadata/visibility access, offset persistence and commits, transformed visible-region math,
and layer-order rebuilding. All methods retain app-only visibility and continue to consume or
commit the same actor projection; this is not a new semantic layer model.

Annotation rendering is also separated from its source decoders. `annotations/mod.rs` retains the
layer model/UI and draw orchestration; hit testing, category/color LUT construction, parquet
schema/data decoding, and OpenGL execution live in dedicated child modules. They consume the same
shared annotation payloads and do not create a second resource or command owner.

Raw-tile OpenGL rendering follows the same source boundary. `render/tiles_gl.rs` retains the public
draw DTOs and `TilesGl` façade; child modules own paint orchestration, cache/GPU/FBO lifecycle,
geometry and affine math, R16 texture upload, and versioned shader compilation. The split exposes
no new application API and leaves tile rendering as a consumer of actor-owned state.

Line-bin OpenGL rendering is split behind its unchanged public façade as well. The generic
line-segment renderer, object-indexed selection renderer, and shared shader compiler live in
separate child modules. Their LRU buffers and selection-state textures remain renderer resources;
they consume actor-projected geometry and selection generations without owning either semantic
model.

The same guard now applies to the canonical model. Its former 13,420-line `src/model/app.rs`
combined all domain mutations, dispatch, shared types, helpers, and tests. `src/model/app.rs` is now
a compact type/helper façade, behavior is grouped under `src/model/app/`, and characterization
tests live in `src/model/app/tests.rs`. This is source organization only: all domain `impl
AppModel` blocks still mutate the same actor-owned value, and registry dispatch remains a single
typed entry point.

The viewport command surface is split one level further without splitting its owner.
`model/app/viewport_commands.rs` retains dataset/revision helpers and active-viewport compatibility
commands; `viewport_commands/workspace.rs` owns topology and link groups; `navigation.rs` owns
camera and plane mutations; and `presentation.rs` owns scoped channels, objects, and rendering.
Every method remains an inherent operation on the same actor-owned `AppModel` and retains the same
typed dispatch path.

The declarative UI service is split behind the same stable `UiRegistry` API. `control/ui.rs` owns
extension/contribution registry state and lifecycle operations; `ui/render.rs` owns native window
and contribution placement; `ui/render/components.rs` owns recursive widget rendering and
interaction event policy; `ui/validation.rs` owns tree limits, capability checks, patches, and
native binding synchronization; and tests live separately. Python and protocol clients continue to
submit the same declarative component trees and drain the same typed UI actions.

Object-layer implementation follows the same boundary. The former object core is split into
runtime/loading, properties UI, selection, export, and test modules. Object analysis now keeps its
UI and threshold-set orchestration in `objects/analysis.rs`, while property-cache/live-selection
derivation and pure histogram/threshold/scatter algorithms live in dedicated child modules.
Structural guards cap both façades and their production modules; this is organization of the same
renderer-side object layer, not a new semantic owner beside the actor.

Object presentation state is also separated beneath `objects/core/properties_ui.rs`. The façade
owns the egui properties panel; `properties_ui/style.rs` owns actor style projection and global
display settings; `filters.rs` owns filter/query state and property selection; and `colors.rs` owns
legend entries, per-value colors/visibility, project display persistence, and loaded-property
reconciliation. All remain inherent methods on one `ObjectsLayer` projection.

At the outer object-module boundary, `objects/mod.rs` now retains shared `ObjectsLayer` state and
supporting DTO/types only. The native actor object-resource service and filter evaluator live in
`objects/control_service.rs`; typed property-column storage and matching live in
`property_store.rs`; default renderer/UI projection construction lives in `defaults.rs`; and their
tests live in `objects/tests.rs`. Existing public re-exports preserve the service and UI-action API.

Object rendering is split at its own runtime boundaries. `objects/render.rs` owns the actual draw
pipeline and presentation styles; `render/selection.rs` owns hover, query, selection, and selected
render-state updates; `render/mesh.rs` owns fill meshes and render-cache identifiers;
`render/selection_geometry.rs` owns pure hit-test geometry; and `render/lods.rs` owns LOD and color
group construction. Geometry/cache regression tests live in `render/tests.rs` rather than the
production modules.

The actor-owned `MosaicModel` is independently split by semantic domain. Its façade retains shared
types, construction, resource installation, snapshot/projection, and command dispatch;
`model/mosaic/memory.rs` owns pin lifecycle; `objects.rs` owns object-resource loading;
`layout.rs` owns item layout, selection, focus, and camera navigation; `presentation.rs` owns
channels, native layers, panels, and rendering settings; and tests live separately. These modules
all implement the same actor-owned model and do not shard state or ordering authority.

The canonical mask model is split without splitting its owner. `model/masks.rs` retains the
`MaskModel` state type and defaults; `masks/state.rs` owns projection, persistence, and worker-result
installation; `commands.rs` owns generation-checked CRUD dispatch; `validation.rs` owns parameter
and polygon validation; `geojson.rs` owns file decoding; and tests live separately. The actor still
serializes every mutation through one model value and exposes the same crate-visible contracts.

Deep-link input is organized behind the unchanged `deep_link.rs` public façade. Dedicated modules
own canonical URL serialization, project/example/ROI resolution, URL and value parsing, and
actor-facing segmentation/filter derivation, with characterization tests isolated separately.
Native URL handlers and actor commands therefore consume one request contract without coupling
the parser to project loading, renderer construction, or command execution.

Canonical project state follows the same single-owner split. `model/project.rs` retains
`ProjectModelSnapshot` and the private `ProjectModel`; `project/state.rs` owns install, persistence,
manifest, sample-sheet, discovery, and worker-result transitions; `commands.rs` owns project/ROI/
saved-view dispatch and navigation; and `validation.rs` owns snapshot and request normalization.
The modules share one private snapshot and do not create an alternate project command path.

Renderer construction follows source boundaries rather than accumulating in one constructor file.
`mosaic/construction.rs` is the CLI/samplesheet entry façade; actor-resource, project, local,
remote, project-config, and samplesheet adapters live under `mosaic/construction/`. Each adapter
prepares source-specific resources and passes a typed `PreparedMosaicConstruction` to
`assembly.rs`, which is the only place that applies shared `MosaicViewerApp` defaults. This removes
six duplicated full-state initializers without changing the renderer state type or actor cutover.

Mosaic control integration is separated behind a matching `mosaic/control.rs` façade.
`control/intents.rs` owns typed native command submission; `project.rs` owns renderer-side project
serialization, restoration, and preload installation; `snapshots.rs` owns control projections and
test-only compatibility queries; `screenshots.rs` owns capture requests and naming; and `host.rs`
owns narrow platform-facing requests. Every method remains on the same `MosaicViewerApp`, and the
split neither restores renderer semantic ownership nor changes actor routing.

The native TIFF resource provider is also separated at its natural I/O boundaries.
`xenium/tiff_pyramid.rs` retains public TIFF/OME-TIFF types and `TiffPyramid` behavior;
`tiff_pyramid/inspection.rs` owns IFD inspection, channel ordering, and pyramid construction;
`metadata.rs` owns OME-XML parsing; `loaders.rs` owns chunk decoding and bounded asynchronous tile,
histogram, and channel-max loaders; and format/loader tests live in `tests.rs`. The public exports
through `xenium::` remain unchanged.

SpatialData renderer adapters are separated without changing their format or ownership boundary.
`spatialdata/layers.rs` now owns only the discovered-layer collection and public target types;
`layers/shapes.rs` owns polygon/point shape loading, drawing, and object-layer adaptation;
`layers/points.rs` owns point interaction, feature presentation, and rendering; and
`layers/points/prepare.rs` owns background coordinate normalization and feature-cache/bin
construction. Actor workers still prepare canonical document resources, while these modules remain
renderer adapters for the resulting SpatialData elements.

The underlying parquet-shape adapter is also split without changing its public loader contracts:
`spatialdata/parquet_shapes.rs` owns schema inspection and projected/cancellable batch loading,
`parquet_shapes/geometry.rs` owns the private WKB/Arrow geometry decoder and centroid summaries,
and `parquet_shapes/tests.rs` contains decoder characterization. Actor workers continue to call the
same public loader functions.

#### Ownership cleanup during each migration

Do not accumulate actor and renderer copies until a final bulk cleanup. Each domain migration must
remove or encapsulate its old semantic mutation path immediately. Compatibility DTOs may remain,
but opaque JSON should not become new canonical state when a typed Rust model is practical.

### Immediate execution sequence

The following is the concrete queue from the current worktree. It makes the dependency order
explicit and prevents the broad method waves from building on another global-busy abstraction.

| Order | Deliverable | Current state | Required evidence |
| ---: | --- | --- | --- |
| 1 | Scoped readiness keys and generation-safe aggregation | Complete | Simultaneous same-kind operations remain independent; stale completion and cancellation tests pass |
| 2 | Per-viewport and selection object-filter readiness | Complete | Two filters can run concurrently; clearing/replacing one cannot complete or cancel the other |
| 3 | Mask import/export retained operation state | Complete | Multiple mask I/O jobs have independent progress, failure, cancellation, and stale-result rejection |
| 4 | Mode- and target-aware route metadata | Complete | Registry introspection publishes the same evaluator used by actor dispatch |
| 5 | Wave 1 application settings/recent/lifecycle | Complete | No-frame mutation, persistence/restart, ordering, lifecycle-save, and native-command tests pass |
| 6 | Actor implementation modularization and registry-owned capability metadata | Complete | Façade/runtime/dispatch/workers/completions/domains/tests are separated; registry audit passes |
| 7 | Application implementation modularization and state-ownership inventory | Complete | Frame/update/domains/tests are separated; legacy boundary and source guardrails pass |
| 8 | Wave 2A source-neutral document contract and inspection | Complete for OME-Zarr/inspection foundation | OME-Zarr and inspection produce typed descriptors and complete without a frame; alternate adapters remain Wave 2D |
| 9 | Actor-owned retained-task kernel | Complete as a foundation | Actor orders task mutation; transport is a thin observer; blocked-worker cancellation remains responsive |
| 10 | Wave 2B deep-link resolution | Complete | Current, external, and example-project resolution works without frames |
| 11 | Wave 2C remote sessions and remote OME-Zarr | Complete | HTTP/S3 list/open no-frame tests cover cancellation, stale work, and credential redaction; native single-open uses the same actor commands |
| 12 | Wave 2D TIFF/SpatialData/Xenium adapters | Complete for registered surface | All three opens and target-aware secondary object resources are prepared without a frame |
| 13 | Wave 2E project preload and ROI-open transaction | Complete | Actor-owned preload resources are shared with atomic local/HTTP/S3 ROI opens; failure, cancellation, and stale completion retain the prior document |
| 14 | Wave 2F transactional deep-link apply | Complete | Current/external project, ROI, resources, saved view, filters, and camera commit atomically with cancellation and stale-completion guards; native URL/open entry points use the actor |
| 15 | Wave 3 retained compute/resources | Complete | Threshold, memory, analysis, measurement, and export tasks progress and cancel without frames |
| 16 | Wave 4 canonical mosaic and mosaic opening | Complete | Complete mosaic state and resource workflow executes with frames paused |
| 17 | Wave 5 presentation tasks | Complete | Pixel capture waits explicitly for presentation while unrelated commands continue |
| 18 | Wave 6 legacy semantic-path removal | Complete | Legacy channels, dispatchers, renderer event publishers, and native snapshot translators are absent; native commits originate as typed commands |
| 19 | Wave 7 release audit | In progress | Automated cumulative verification and manual platform acceptance remain |

The execution rule is deliberately strict: finish and test each row before using it as a
dependency for the next one. This does not require one commit per row, but each row should remain
independently reviewable.

### Progress accounting

Three numbers must be reported separately throughout implementation:

- **registry coverage:** 264/264 application methods actor-capable;
- **route coverage:** zero legacy or hybrid application routes across supported modes and targets;
- **acceptance coverage:** commands proven to complete under paused frames and real OS occlusion.

The retained platform matrix, commands, and visual-resume checklist live in
`docs/testing/background-control-acceptance.md`. The verifier can write one JSON semantic/resource
record per native-window condition; visual projection inspection is recorded separately so a
successful background command run cannot be mistaken for proof of renderer catch-up.

An increase in registry coverage alone does not close either completion gate. The verifier should
eventually derive all three figures from registry metadata and retained test evidence so that this
document does not become the only source of truth.

### Migration checklist for every command family

Each family follows the same reviewable sequence:

1. Freeze successful and error response shapes, events, revisions, persistence, and task behavior
   with characterization tests.
2. Classify every field as model-owned, shared resource, renderer-owned, presentation-only, or
   transient UI state.
3. Add a typed actor submodel and generation tokens for any heavyweight resource.
4. Move blocking work to a bounded worker and make cancellation/stale completion deterministic.
5. Dispatch Python, MCP, deep-link, menu, and native UI operations through the same command.
6. Publish the minimum immutable render projection needed to display the result.
7. Delete or restrict the old direct semantic mutation path.
8. Add a paused-frame actor test, native/remote parity test, projection-resume test, and relevant
   persistence round trip.
9. Update the route matrix, application-surface manifest, Python reference, and limitation notes.
10. Run the family tests plus the cumulative no-frame workflow before moving to the next family.

### Wave 1 — central application state and lifecycle

Status: complete in the current worktree. Settings types are shared library models; actor commands
validate and persist settings on bounded workers; settings writes cannot overlap out of order;
project open/save records recent paths; lifecycle prompt/save/discard is actor-owned; and actual
close/quit is delivered through a narrow platform-effect queue. Native settings, recent-project,
and close actions submit the same commands when the bridge is available.

Scope: the eight remaining `app.settings.*`, `app.recent_projects.*`, and `app.lifecycle.*`
methods, plus completion of the newly migrated application/rendering state queries.

Implementation:

- add actor-owned application preferences, recent-project records, dirty state, and close intent;
- bootstrap settings exactly once and project subsequent actor changes back to native controls;
- persist settings on a worker using generation-checked atomic replacement;
- make project open/save update recent-project state in the same semantic workflow;
- model close/quit decisions independently of the native window command;
- treat the actual window close or process quit as a small presentation/platform effect after the
  actor has validated save/discard/cancel semantics; and
- make right-tab and other application-level UI selections canonical where they affect workflows.

Exit evidence:

- settings and recent-project mutations complete with no frames and survive restart;
- concurrent settings writes cannot commit out of order;
- dirty close with `cancel`, `discard`, and `save` has deterministic actor tests; and
- native settings changes and Python settings changes produce the same model and events.

Release follow-ups retained from this completed wave: verify replace-over-existing settings writes
on Windows and use a platform-safe atomic replacement primitive if `rename` is insufficient; test
recent-project existence-cache bootstrap and restart behavior on all supported platforms; and emit
an observable settings-persistence failure event in addition to readiness/status state.

### Wave 2 — project resources, alternate datasets, and deep-link transactions

Status: in progress. Dataset inspection, remote session/list/open, and deep-link
parse/resolve/filter/generate are actor-owned and have no-frame actor tests. Six non-mosaic
methods remain outside the actor list in this wave. The
two mosaic-opening methods formerly grouped here are deliberately deferred to Wave 4 so they are
implemented once against the canonical mosaic model rather than temporarily reproducing legacy
mosaic ownership.

#### Wave 2A — characterize and introduce the source-neutral document boundary

Status: complete for the common contract, local inspection, and OME-Zarr adapter. Alternate source
adapters deliberately remain in Wave 2D.

Methods: `datasets.inspect`, plus the shared internals used by all `datasets.open_*` commands.

- freeze the current success/error payloads and source-specific option validation with
  characterization tests before changing routing;
- introduce Send-compatible `DatasetInspection`, `OpenedDocument`, and resource-handle types that
  separate semantic metadata from renderer construction;
- retain source identity, axes, extents, channels, planes, supported overlays, and stable resource
  generations in the result;
- keep tile decoders, GPU resources, event-loop handles, and renderer caches out of worker results;
- make local inspection a scoped worker operation with cancellation and stale-result rejection; and
- adapt the existing OME-Zarr open path to the same boundary first, proving that the abstraction
  preserves the already working no-frame workflow before adding new source kinds.

Checkpoint: `datasets.inspect` and OME-Zarr both use the common boundary, paused-frame tests pass,
and rendering can resume from the published resource generation without reopening the source.

#### Wave 2B — deep-link resolution

Status: complete. Current-project resolution avoids I/O; external and example-project resolution
runs on bounded workers and uses the same normalized project/ROI rules as native startup.

Method: `deep_links.resolve`.

- resolve against the current actor-owned project without I/O;
- when the link names another project, read that project on a bounded worker and return a typed,
  generation-tagged resolution candidate for the actor to validate;
- centralize example-project/default resolution rules outside `RootApp` so native startup and the
  protocol use exactly the same resolver;
- return an unambiguous project source, normalized project path, and ROI identity without changing
  application state; and
- never use the UI's pending-deep-link queue as a completion mechanism.

Checkpoint: current-project and external-project resolution complete with frames paused; ambiguous,
missing, malformed, cancelled, and superseded cases have stable errors and cannot alter state.

#### Wave 2C — remote session service and remote OME-Zarr

Status: complete. The actor owns an in-memory credential service and redacted session snapshot;
bounded workers perform S3 listing and HTTP/S3 opens; session generations reject stale work; and
the native single-dataset remote dialog submits the same actor commands as Python and MCP.

Methods: `datasets.s3.configure_session`, `datasets.s3.get_session`,
`datasets.s3.clear_session`, `datasets.s3.list`, `datasets.open_http`, and `datasets.open_s3`.

- keep S3 credentials in a dedicated in-memory control service; only redacted endpoint, region, and
  session-presence metadata may enter snapshots, diagnostics, events, or logs;
- perform S3 listing and HTTP/S3 metadata reads on bounded, cancellable workers;
- use explicit operation scopes per list/open request and preserve transport errors as structured
  control errors;
- make clear-session invalidate or cancel dependent work deterministically; and
- install remote documents through the same `OpenedDocument` commit used by local OME-Zarr.

Checkpoint: credential redaction tests, list/open cancellation tests, stale session-generation
tests, and no-frame HTTP/S3 opens pass. Network integration tests may use deterministic local test
servers/emulators; real-provider smoke tests remain optional release evidence.

#### Wave 2D — alternate local and structured dataset adapters

Status: in progress. TIFF, SpatialData, and Xenium opening now run as generation-checked actor
transactions on the bounded worker pool and publish source-neutral prepared documents. TIFF
metadata is parsed once and its immutable pyramid is reused when presentation resumes. SpatialData
and Xenium discovery, selection validation, and primary imagery preparation are also worker-owned;
their selected secondary imagery, labels, shapes, points, tables, cells, and transcripts still need
dedicated typed resource handles so no initial subresource I/O begins from renderer construction.

Methods: `datasets.open_tiff`, `datasets.open_spatialdata`, and `datasets.open_xenium`.

- move discovery, metadata parsing, table/shape enumeration, and initial resource loading to the
  bounded worker side of the source-neutral boundary;
- represent selected SpatialData/Xenium imagery, labels, shapes, points, and object tables as typed
  resources with stable generations;
- return only model-installable results from workers; renderer-specific construction happens later
  from resource handles when a frame is available;
- ensure a superseding dataset/project open cancels the semantic operation and makes every late
  source completion harmless; and
- preserve each command's current element-selection and validation contract.

Checkpoint: each source kind opens to model/resource readiness with frames paused and presents from
the installed generation after frames resume. Unsupported variants fail before mutating the active
document.

#### Wave 2E — project object preload and ROI-open transaction

Methods: `project.objects.preload.get`, `project.objects.preload.list_sources`,
`project.objects.preload.start`, `project.objects.preload.clear`, and `project.rois.open`.

Status: complete. The preload catalog is keyed by saved-project/load/resource identity, filesystem
inspection and parsing run on bounded workers, native UI uses the same actor commands, and renderer
caches are populated only from immutable projections. ROI open retains the prior document until
the selected local/HTTP/S3 source, saved labels, object resource, masks, and persisted view state
can commit together. Shared preload handles avoid reparsing; task cancellation, explicit clear,
project/source supersession, and late worker completions are generation guarded.

- add an actor-owned preload catalog keyed by project/resource identity and project generation;
- share immutable object resources between preload results and the document install path instead of
  parsing the same source twice;
- make clear/cancel invalidate only the intended preload scopes;
- implement ROI open as a retained actor transaction: validate ROI, open its dataset, install the
  document generation, attach masks/objects/labels and saved-view state, then emit the final mode/
  active-ROI event and reply;
- define transactional failure semantics explicitly: either retain the prior usable document or
  enter a typed transition failure, never expose a half-installed mixture; and
- remove renderer-side project resource auto-loads once the actor transaction owns them.

Checkpoint: preload reuse is observable, immediate save/state queries see the committed resources,
ROI open has atomic failure/cancellation tests, and native/Python/MCP calls produce identical model
snapshots with frames paused.

#### Wave 2F — deep-link application

Method: `deep_links.apply`.

Status: complete. The control actor now owns resolution, bounded-worker resource preparation,
generation/cancellation checks, and one atomic model commit. Current-ROI application reuses the
installed document and preserves an arbitrary viewport workspace; external-project failure,
cancellation, concurrent project edits, and invalidated remote sessions retain the prior usable
document. Native URL handling and project ROI open actions submit this same actor transaction, and
the former frame-driven API apply path has been removed.

- model apply as a retained transaction that composes the tested project-open, resolve, ROI-open,
  and view/filter commands rather than directly mutating their fields;
- capture source generations at every asynchronous boundary and stop promptly on cancellation;
- apply camera, planes, channel presentation, object styling/filtering, and selected saved view only
  after the target document/resources are installed;
- emit one ordered semantic outcome and make repeated application idempotent where the public
  contract permits; and
- delete the UI-owned pending-deep-link application path after native deep links submit this same
  transaction.

Checkpoint: a deep link can open an external project and ROI and apply its complete view while
frames are paused; returning to Odon renders the final projection without another command.

Wave 2 exit evidence:

- all six remaining non-mosaic methods in this wave have actor routes for every supported target;
- every supported single-document source can be inspected/opened without a frame;
- secrets and renderer-only handles never appear in serializable model state;
- cancelling or superseding a remote/local/project/deep-link open cannot install stale resources;
- opening through native UI, Python, MCP, or a deep link yields the same project/document model;
  and
- the application-surface manifest, route matrix, Python contracts, and limitation documentation
  are regenerated from the completed routes before Wave 3 starts.

### Wave 3 — single-view compute and resource operations

Scope: 18 remaining methods: analysis, measurements, and object export.

Completed Wave 3 slice: `viewer.screenshot.settings.get/set` are actor-owned in single-view mode.
Output-directory validation runs on a bounded worker and commits by generation; native dialog
changes submit the same typed command and the projection updates the renderer. Mosaic screenshot
preferences remain on the mosaic route until Wave 4, and actual pixel capture remains Wave 5.
`memory.tiles.get/set` are also actor-owned for single-view mode: semantic policy projects to the
renderer, while cache occupancy and policy-realization generation return through the guarded
renderer-observation channel. The mosaic route remains part of Wave 4.
`memory.get/pin/unpin/unpin_all` are actor-owned for single-view mode. Full-level channel data is
read on bounded workers into immutable shared resources, cancellation is checked between channel
reads, and unpinning or document replacement rejects late installs. System-memory risk is measured
on the worker before allocation. Mosaic pinning remains part of Wave 4.
All six `viewer.thresholds.preview.*` methods are actor-owned. Region reads, component filtering,
and polygonization run on bounded workers; configuration and preview resources are generation
guarded, cancellation rejects late results, and Apply commits the generated editable mask through
the canonical mask model. Native threshold controls submit the same commands, while the renderer
only realizes the projected shared preview resource.

Implementation:

- move analysis configuration, presets, threshold configuration, measurement definitions, export
  specifications, and memory policies into typed actor submodels;
- separate CPU/resource results from GPU cache realization;
- run histograms, threshold suggestions, mask computation, analysis warmup, measurements, and file
  exports as cancellable retained tasks on bounded workers;
- reuse immutable object/image resources and spatial indexes by generation;
- publish threshold masks, measurement results, and export metadata as shared resources rather than
  embedding large results in projections;
- make tile pinning a semantic residency request that can complete at the resource-cache boundary
  without requiring the target pixels to have been presented; and
- keep screenshot preferences semantic while leaving actual capture for Wave 5.

Exit evidence:

- every operation progresses and cancels while frames are paused;
- concurrent jobs have independent readiness and cannot clear each other's state;
- output tasks reply only after files are durably written;
- stale results after dataset/object replacement are rejected; and
- Python sync and async APIs observe identical task results without blocking the Python event loop.

### Wave 4 — canonical mosaic model

Scope: all 17 remaining `mosaic.*` methods, `project.rois.open_selected_mosaic`,
`datasets.open_mosaic_samplesheet`, and mosaic branches of otherwise shared viewer methods.

Implementation:

- add `MosaicModel` containing item identity, layout, selection, focus, cameras, shared channel
  presentation, object resource state, and logical geometry;
- publish a lightweight mosaic projection consumed by `MosaicViewerApp`;
- make selection, focus stepping/fitting, layout, tabs, object loading, and cancellation actor-owned;
- run per-item metadata and object loads through bounded workers with mosaic/document generations;
- construct mosaics from project ROI selection and samplesheets through the same actor-owned item
  transaction, instead of letting either opening command initialize renderer state directly;
- convert native mosaic interactions to commands with revision-tagged optimistic camera previews;
  and
- replace static route labels for shared methods with truthful single/mosaic route entries.

Exit evidence:

- the complete mosaic command surface works with no frames;
- selected object loads settle without renderer polling;
- returning to the window renders the latest mosaic layout/focus/selection atomically; and
- native and remote mosaic workflows have state, event, and revision parity.

### Wave 5 — explicit presentation tasks

Scope: the four screenshot/capture methods and any later operation whose contract promises rendered
pixels rather than semantic/resource readiness.

Sequence:

```text
command -> actor validates output and captures desired model revision
        -> task phase = waiting_for_presentation
        -> actor publishes/coalesces render projection and remains responsive
        -> renderer reports revision/resources visible
        -> UI performs pixel readback/crop
        -> output worker encodes and atomically writes the file
        -> actor commits output metadata and completes the task
```

Requirements:

- presentation waits never block the actor mailbox;
- tasks remain observable and cancellable while the window is covered;
- a task reports exactly which projection/resource generation it is waiting for;
- timeout policy is explicit and does not misreport model failure;
- overwrite checks and output writes are race-safe; and
- project screenshot composition uses an actor mode transition followed by the same presentation
  acknowledgement mechanism.

Exit evidence:

- covered-window capture reports `waiting_for_presentation` rather than appearing stalled;
- unrelated model commands continue to complete during that wait;
- foregrounding Odon advances the presentation acknowledgement and finishes the output; and
- cancellation leaves no partial destination file.

### Wave 6 — remove the semantic compatibility path

Status: complete for the production application surface. The registry audit reports no legacy
route, the bridge has no legacy request channel, and `RootApp` has no semantic request dispatcher.
Renderer snapshots and actor projection application live in explicitly named bridge modules;
test-only compatibility helpers are not compiled into Odon. Native-layer activation, visibility,
ordering, properties, offsets, and channel transforms now commit directly from their interaction
points; drag previews retain their starting presentation revision and control projections are
deferred until the gesture ends.

- generate a registry audit that fails if any semantic/resource method can reach
  `RootApp::reply_to_control_request`;
- delete the canvas-deferred request queue after only presentation tasks use presentation reports;
- remove duplicate model fields or make renderer copies private projection caches;
- stop periodic renderer observations from carrying actor-owned semantic fields;
- remove UI-side project/resource auto-loaders superseded by actor workers;
- make native menus and all committed egui actions submit typed commands; and
- retain a narrowly named platform-effect channel only for window focus/close/quit, file dialogs,
  clipboard integration, and rendered-pixel capture.

Exit evidence:

- pausing `RootApp::update` cannot prevent any non-presentation application method from completing;
- diagnostics show no legacy semantic routes in any supported mode/target combination; and
- compile-time module visibility makes direct semantic renderer mutation difficult.

Native project, single-viewer, and mosaic controls now submit direct actor commands. Viewport
workspace, camera, plane, rendering, channel, object-presentation, mosaic layout/focus, and
worker-backed object-filter interactions emit commands at their commit points. The final
before/after mosaic snapshot translator and its root module have been deleted; a structural test
prevents their reintroduction. Revision-tagged camera drags remain optimistic renderer previews,
then submit `viewer.camera.set`. Actor projections update renderer mirrors through projection-only
helpers and cannot feed commands back into the actor. Renderer-only observations (resource
discovery, tile-loading telemetry, and measured geometry) remain intentionally.

Native-layer projection no longer reuses renderer command emulators. A dedicated projection module
applies actor-validated presentation, ordering, offsets, and active-layer state without generating
responses or semantic intents; the old `control_set_*` layer methods are test-only characterization
fixtures. Regression coverage includes first-projection mask-to-channel activation feedback and
exact clearing of an actor-projected automatic contrast window.

Top-level project-workspace actions now originate from one `ProjectSpace` typed-command outbox
before their root, single-viewer, or mosaic host can create a semantic relay request. The single
viewer ROI selector follows the same route. Existing semantic `ViewerRequest` and `MosaicRequest`
semantic variants have been deleted. `RootApp` requires its actor runtime, and actor construction
precedes optional TCP-server binding and manifest publication; a TCP failure reports that
Python/MCP access is unavailable while local actor controls continue, whereas an
actor-construction failure aborts application construction instead of enabling a frame-driven
semantic fallback. Startup dataset opens also enter the actor outbox, and the old root direct-open
and direct deep-link chains are gone. Platform effects and transient dialogs intentionally remain
host-owned.

### Wave 7 — stabilization and release evidence

The detailed execution sequence for the remaining ownership cleanup, compatibility deletion, and
release sign-off is maintained in `actor-refactor-completion-plan.md`.

Run and retain evidence for:

- actor/model unit tests without egui;
- stale completion, cancellation, mailbox saturation, actor panic, and worker panic tests;
- concurrent operation/readiness race tests;
- native-command versus protocol-command replay parity;
- projection coalescing and resume rendering for every model subtree;
- project/settings persistence and restart round trips;
- complete Rust, Python SDK, protocol-schema, generated-reference, and application-surface suites;
- latency/allocation benchmarks against the performance requirements in this document; and
- visible, fully covered, minimized, and separate-Space manual runs on macOS, with supported
  Windows/Linux equivalents.

The release audit must inspect every registry method and every requirement above. Absence of a
known failure is not completion evidence; each item needs a passing test, recorded platform result,
or explicit intentional exclusion.

## Historical Architecture and Migration Record

The remainder of this document records the pre-cutover problem and the phases used to reach the
current architecture. References to a remaining or temporary legacy path describe that historical
state; the authoritative current status and release gates are above.

Odon originally accepted protocol requests on background network threads but
executed all application commands from `RootApp::update`. Those routes depended on eframe receiving
a native redraw. On macOS, a fully covered window can be reported as occluded and AppKit may
withhold redraws, so commands did not progress until Odon became visible.

The implemented architecture makes a central, UI-independent Rust model the
canonical owner of controllable application state. A dedicated control actor
serializes semantic commands against that model. Python, MCP, deep links,
native menus, and egui controls converge on the same typed commands. The egui
application consumes model projections and renders them when frames are available;
the production application surface now uses that boundary throughout.

```text
 Python SDK ----\
 MCP adapter ----\
 deep links ------> typed command mailbox -> ControlActor -> AppModel
 native UI -------/                            |   |   |
 native menus ---/                             |   |   +-> tasks/workers
                                                |   +-----> events/revisions
                                                +---------> RenderProjection
                                                               |
                                                       RootApp / egui / GPU
```

This is an incremental refactor, not a rewrite. The completed first vertical slice
moved dataset identity, viewport workspace, camera, plane, channel presentation,
and readiness into the actor. Subsequent work migrated substantial project, layer,
object, mask, and label state. The updated waves finish the remaining compute,
mosaic, presentation, and application-level domains.

## Original Problem Statement (Resolved)

The former legacy control path was:

```text
protocol thread -> OdonControlRequest queue -> request_repaint()
                                            -> RootApp::update()
                                            -> mutate viewer state
                                            -> protocol reply
```

Methods using this path had several consequences:

- A semantic command cannot complete without a GUI frame.
- macOS window occlusion becomes an application-readiness condition.
- Network request lifetime, model mutation, layout, tile planning, and painting
  are accidentally coupled.
- Commands that do not need graphics resources, such as renaming a viewport or
  changing channel visibility, still wait for the renderer.
- `open_task.wait()` cannot distinguish model readiness from presentation
  readiness without inspecting transient egui state.
- Tests must often construct or drive egui just to exercise domain behavior.
- The native UI and protocol handlers can drift because both mutate large app
  structs directly.

The deferred-canvas queue makes command completion honest when a frame
is available, but it cannot create frames while the OS suppresses them. It is a
correct short-term guard, not the final execution architecture.

## Goals

The architecture must:

- Execute ordinary commands while the Odon window is fully covered.
- Keep egui, OpenGL, wgpu, native window handles, and input gestures on the UI
  thread.
- Make one Rust model authoritative for both native and remote control.
- Preserve the existing JSON-RPC protocol and Python API wherever semantics are
  already sound.
- Give synchronous calls blocking semantics and asynchronous calls awaitable
  semantics without blocking Python's event loop.
- Define completion according to the resource affected, not according to the
  occurrence of an arbitrary paint.
- Retain deterministic logical viewport geometry independently of fresh egui
  layout.
- Maintain ordered revisions, events, task progress, cancellation, and conflict
  detection.
- Support incremental migration with characterization tests and reversible
  milestones.
- Keep rendering responsive and avoid copying full image or object payloads on
  every model update.

## Non-Goals

This architecture will not:

- Embed Python or run Python code inside Odon.
- Move GPU resources or egui widgets to the actor thread.
- Make an actual OS-window screenshot possible while the platform refuses to
  present a frame; such operations will expose presentation-waiting state.
- Convert every existing Odon feature in one change.
- Introduce a distributed multi-user consistency model.
- Replace referenced Zarr, Arrow, GeoJSON, or image resources with JSON copies.
- Promise that UI-only transient state such as hover, drag previews, or tooltip
  dwell is remotely controllable.

## Architectural Principles

### The actor owns canonical semantic state

There must not be one mutable camera in `RootApp` and another mutable camera in
the actor with bidirectional best-effort synchronization. Once a domain is
migrated, the actor's value is canonical. The UI reads a projection and sends
commands for mutations.

### Message passing, not shared mutable application locks

The actor runs on a dedicated Rust thread with a bounded mailbox. Clients send
typed command envelopes and receive typed results through one-shot reply
channels. The UI receives immutable projections and sends the same commands as
remote clients.

A large `Arc<Mutex<RootApp>>` is explicitly rejected:

- `RootApp` contains UI- and graphics-thread-bound values.
- paint stalls could block protocol execution;
- protocol work could block input and rendering; and
- lock ordering across loaders, events, and tasks would be fragile.

### Model readiness and presentation readiness are distinct

The model may be ready even when no new frame can be painted. Readiness will be
reported as explicit dimensions:

- `model_ready`: the requested document and semantic state are installed;
- `resources_ready`: required metadata or referenced resources are usable;
- `geometry_ready`: logical viewport geometry is sufficient for the command;
- `presentation_ready`: a UI projection has been consumed by a rendered frame;
- `output_ready`: a requested file or retained output is complete.

Most control calls require only model readiness. Camera fitting requires model
and logical geometry readiness. Window screenshots require presentation and
output readiness.

### Rendering is a projection, not the source of truth

The actor publishes a lightweight `RenderProjection` containing the current
document identity, viewport workspace, camera and plane state, layer/channel
presentation, selection signatures, and generations for larger resources.
`RootApp` uses this projection to update renderer-local caches and plan frames.

Decoded tiles, GPU textures, in-flight uploads, hover state, input gestures,
frame timing, and transient drawing state remain renderer-owned.

### No actor command waits synchronously for an occluded UI

The actor may enqueue a presentation request and return a retained task that
reports `waiting_for_presentation`, but the actor thread itself must remain able
to process queries, cancellation, and other commands. A UI acknowledgement is
allowed only for operations whose public contract explicitly requires pixels
to have been presented or captured.

## Proposed Components

### `AppModel`

`AppModel` contains Send-compatible, serializable or cheaply cloneable semantic
state. Initial submodels:

```text
AppModel
├── mode: Project | Single | Mosaic | Transition
├── project: ProjectModel
├── document: Option<DocumentModel>
├── workspace: Option<ViewportWorkspaceModel>
├── resources: ResourceCatalogModel
├── extensions: ExtensionModel
├── revision: ModelRevision
└── readiness: ReadinessModel
```

The initial `DocumentModel` should own:

- stable dataset source identity and metadata;
- axes, level-zero extent, channels, and plane bounds;
- stable native and external layer descriptors;
- shared object-selection identity;
- project-backed persistence descriptors; and
- generation IDs for heavyweight resources owned elsewhere.

The initial `ViewportWorkspaceModel` should own:

- stable viewport IDs and titles;
- layout tree or current two-slot layout representation;
- logical canvas geometry;
- camera, plane, and per-viewport presentation state;
- camera, plane, and selection link definitions;
- navigation and presentation revisions; and
- active viewport identity.

### `ControlActor`

The actor owns `AppModel` and processes one command at a time in mailbox order.
It provides:

- typed validation and dispatch;
- optimistic revision guards;
- deterministic mutation transactions;
- event creation;
- task creation and progress updates;
- render-projection publication;
- project persistence scheduling; and
- worker completion handling.

The actor must remain responsive during long work. Dataset reads, object-table
loads, image export, analysis, and inference remain worker tasks. Workers return
typed completion messages to the actor; they do not mutate the model directly.

### `ControlHandle`

All callers receive a cloneable `ControlHandle` containing bounded senders and
read-only shared snapshots where appropriate. It is the only supported entry
point for semantic commands.

```rust
pub struct CommandEnvelope {
    pub command: ControlCommand,
    pub origin: CommandOrigin,
    pub request_id: Option<RequestId>,
    pub expected_revision: Option<ModelRevision>,
    pub reply: oneshot::Sender<Result<ControlResult, ControlError>>,
}
```

`CommandOrigin` distinguishes Python, MCP, native UI, deep link, menu, loader,
and system recovery without changing command semantics.

### `RenderProjection`

The projection is immutable and versioned:

```rust
pub struct RenderProjection {
    pub model_revision: ModelRevision,
    pub document_generation: DocumentGeneration,
    pub workspace: Arc<ViewportWorkspaceProjection>,
    pub layers: Arc<LayerProjection>,
    pub selection: Arc<SelectionProjection>,
    pub resource_generations: ResourceGenerations,
}
```

Use structural sharing (`Arc`) and generation IDs so camera changes do not copy
object tables or image metadata. The actor publishes through a latest-value
channel; intermediate render projections may be coalesced because every
semantic event remains preserved separately.

### `PresentationReporter`

After consuming a projection, the UI reports:

- model revision consumed;
- actual window and canvas rectangles;
- viewport IDs rendered;
- renderer/resource generations visible;
- screenshot or capture completions; and
- recoverable rendering failures.

Reports are actor messages. The UI never directly edits canonical state while
reporting geometry.

## State Ownership Boundary

| State | Canonical owner | Notes |
| --- | --- | --- |
| Dataset identity, axes, extents, channel metadata | Actor/model | Renderer holds storage/cache handles by generation. |
| Project ROIs, saved views, metadata | Actor/model | Persistence runs on workers and commits through actor. |
| Viewport IDs, layout, titles, links | Actor/model | UI renders the declared layout. |
| Cameras, planes, channel visibility and colors | Actor/model | Native gestures emit commands. |
| Logical canvas geometry | Actor/model | Derived from last observation plus layout rules. |
| Actual egui rectangles for the last frame | UI report mirrored in model | Tagged by presentation generation. |
| Object selection and semantic filters | Actor/model | Heavy indexes remain shared resources. |
| Masks and annotations | Actor/model after migration | Gesture previews remain UI-local until committed. |
| Decoded image/object data | Resource workers/caches | Addressed by stable generations. |
| GPU textures and upload queues | Renderer/UI thread | Never crosses into actor. |
| Hover, drag, focus ring, tool transient state | UI thread | Commit produces semantic commands. |
| Tasks, events, revisions | Actor/control services | One ordering authority. |
| Protocol sessions and authentication | Transport/control services | Session lifecycle messages go to actor. |

## Logical Viewport Geometry

Background-safe fitting requires geometry that survives the absence of a fresh
frame. Introduce `LogicalWorkspaceGeometry`:

```text
LogicalWorkspaceGeometry
├── window_content_size_points
├── workspace_rect_points
├── per_viewport_canvas_rects
├── layout_revision
├── observation_revision
├── source: Observed | Derived | Default
└── confidence: Exact | StableEstimate | Bootstrap
```

Rules:

1. Every visible frame reports exact rectangles to the actor.
2. Exact geometry remains valid while the window is merely covered or
   backgrounded.
3. Layout mutations derive new viewport rectangles immediately from the last
   workspace rectangle, header metrics, split ratio, and layout tree.
4. Side-panel changes derive a new workspace rectangle from retained window and
   panel geometry.
5. Dataset replacement preserves workspace geometry; it does not clear canvas
   rectangles merely because renderer-local state was rebuilt.
6. On first launch, bootstrap geometry comes from persisted window dimensions
   and deterministic UI metrics. If unavailable, use documented default window
   dimensions and mark confidence `Bootstrap`.
7. The next visible frame reconciles derived geometry with observed geometry.
   Reconciliation updates future fits but does not undo an already acknowledged
   camera command.
8. Geometry is invalid only when dimensions are non-finite, non-positive, or no
   bootstrap can be constructed.

`fit_camera()` uses the image world extent and logical target canvas size. It
therefore completes while Odon is covered. Returning to Odon may slightly
adjust margins if actual font or panel metrics differ, but it must not reveal a
black, off-image camera.

## Command Execution Classes

Every registry entry will declare an execution class:

### Model commands

Examples: rename viewport, set channels, set camera, set plane, change links,
edit project metadata. These execute completely in the actor and reply without
a UI frame.

### Geometry commands

Examples: fit camera, query objects in the current view, capture a workspace
region definition. These execute in the actor once logical geometry is usable.
They never require foreground visibility.

### Resource commands

Examples: open dataset, load labels, read object properties, run measurements.
The actor starts workers and settles the task when model and required resource
state are installed. Presentation readiness is reported separately.

### Presentation commands

Examples: window screenshot and operations explicitly promising rendered
pixels. These create retained tasks and may remain in
`waiting_for_presentation` while the OS suppresses frames. Other commands and
cancellation remain operational.

### External-compute commands

Examples: registered extension resources or Cellpose output installation.
Python performs computation out of process; the actor validates referenced
results, installs model descriptors, and notifies the renderer.

The command registry and introspection output should expose the execution class
and readiness requirements so SDK documentation is generated from one source.

## Command and Frame Sequences

### Python command while Odon is covered

```text
Python       Protocol       ControlActor       RootApp
  | call()      |                |                 |
  |------------>| envelope      |                 |
  |             |-------------->| mutate model    |
  |             |               | publish proj.   |
  |             |<--------------| success         |
  |<------------|                |                 |
  | continues   |                |                 |
  |             |                |     no frame    |
```

When Odon becomes visible, `RootApp` consumes the latest projection and draws
the final state. It need not replay every intermediate visual state.

### Native UI gesture

```text
pointer drag -> UI-local preview -> camera.set command -> actor transaction
             <- latest projection <- event/revision
```

For smooth panning, the UI may optimistically render a local preview tagged with
the expected navigation revision. The actor remains authoritative and normally
acknowledges within one mailbox turn. On conflict, the next projection corrects
the preview.

### Dataset open

```text
open command
  -> actor creates task and enters resource transition
  -> worker opens metadata/storage
  -> actor installs DocumentModel and workspace model
  -> task reaches model_ready/resources_ready and completes
  -> renderer consumes projection when it can
  -> presentation report advances presentation_ready separately
```

Python can configure channels and viewports immediately after the open task
completes, even if no frame has occurred.

## Revisions, Events, and Transactions

- The actor is the sole allocator of semantic model revisions.
- One command produces at most one externally visible mutation transaction,
  even if it affects linked viewports.
- Per-viewport navigation and presentation revisions remain available for
  narrow optimistic guards.
- Events are emitted after the canonical model commits and before the command
  reply is observed out of order by another client.
- Render acknowledgements use a distinct presentation sequence and never
  increment model revisions.
- Worker completions carry the model generation they were started from. Stale
  completions are discarded or installed only when explicitly compatible.
- Project autosave consumes an immutable model snapshot at a known revision.
- Event subscribers may drop notifications according to current bounded-queue
  policy, but state queries always return the authoritative latest snapshot.

## Error and Timeout Semantics

- `NOT_READY` means a declared model, resource, geometry, or presentation
  prerequisite is genuinely unavailable—not merely that no repaint happened.
- Model commands do not have GUI-frame timeouts.
- Geometry commands report the missing geometry field and confidence state if
  bootstrap construction fails.
- Presentation tasks expose their waiting phase and remain cancellable.
- Actor mailbox saturation returns a bounded backpressure error rather than
  consuming unbounded memory.
- A panicked worker fails its task; it cannot poison the actor.
- A panicked actor is fatal and should cause the control endpoint to close
  rather than continue with divergent UI state.

## Original Incremental Migration Plan (Historical Baseline)

The phases below record the initial rollout strategy. The authoritative remaining-work sequence is
the updated execution plan above.

### Phase 0: Characterize and instrument the current boundary

- Add a regression harness that can deliberately stop UI ticks while protocol
  requests continue.
- Record command execution location, queue wait, model time, render wait, and
  reply time separately.
- Add a macOS manual acceptance script covering visible, covered, minimized,
  and separate-Space states.
- Inventory fields in `RootApp`, `OmeZarrViewerApp`, and `MosaicViewerApp` as
  model-owned, renderer-owned, transient UI, or not yet classified.
- Freeze protocol snapshots for the first migrated vertical slice.

Exit criteria:

- The foreground dependency is reproduced without relying on human timing.
- Every viewport-comparison field has an assigned future owner.

### Phase 1: Introduce actor infrastructure without changing behavior

- Add `src/model/` with identifiers, revisions, readiness, document, viewport,
  and projection types.
- Add the `src/control/actor/` module, handle/envelope types, and bounded channels.
- Start the actor before the protocol bridge and give the bridge a
  `ControlHandle`.
- Add actor health and mailbox metrics to diagnostics.
- Initially route a harmless query and test-only command through the actor.
- Keep the existing RootApp command queue behind a compatibility adapter.

Exit criteria:

- Actor commands and queries complete with zero calls to `RootApp::update`.
- Shutdown, session cleanup, cancellation, and panic behavior are tested.

### Phase 2: Migrate viewport geometry and camera as the vertical slice

- Move stable viewport IDs, titles, layout, links, cameras, planes, and logical
  geometry into `ViewportWorkspaceModel`.
- Add presentation reports from `ui_viewport_workspace`.
- Convert `viewer.workspace.*`, `viewer.viewports.*`, camera, and plane commands
  to actor transactions.
- Convert native viewport controls and pan/zoom commits to the same commands.
- Preserve optimistic local navigation preview for frame-rate responsiveness.
- Remove canvas-dependent deferral for migrated camera operations.

Exit criteria:

- `fit_camera()` completes while the Odon window is fully covered.
- Linked cameras and planes remain correct after returning to Odon.
- Existing viewport revision, persistence, and performance tests pass.

### Phase 3: Migrate dataset opening and channel presentation

- Split dataset metadata/storage ownership from renderer construction.
- Have workers return a Send-compatible `OpenedDocument` descriptor.
- Install document and default workspace models in the actor.
- Preserve logical geometry across document replacement.
- Migrate channels, groups, contrast, transforms, ordering, and rendering
  preferences.
- Redefine open-task completion as model/resource readiness.
- Retain optional presentation readiness for clients that explicitly need it.

Exit criteria:

- The complete `two_viewer_comparison.py` script finishes while Odon is covered.
- When Odon returns to the foreground, both canvases show the requested channels
  and fitted linked cameras without additional Python calls.

### Phase 4: Migrate projects, layers, and referenced resources

- Move project metadata, ROIs, saved views, external descriptors, layer order,
  visibility, and persistence state into model subtrees.
- Change deep links and native menus to actor commands.
- Make project save/load worker-based transactions with revision checks.
- Keep heavy stores, decoded tiles, and GPU resources outside the model.

Exit criteria:

- Project automation and layer installation work with zero UI ticks.
- Project save/load round trips reproduce the actor model.

### Phase 5: Migrate objects, masks, annotations, and selections

- Separate immutable geometry/index resources from semantic presentation,
  filters, analysis configuration, and selection identity.
- Convert committed mask and annotation edits to actor transactions.
- Keep drawing and drag previews UI-local until commit.
- Move view queries to logical geometry plus shared spatial indexes.

Exit criteria:

- Selection, filtering, mask editing through Python, and native editing produce
  equivalent events and project state.
- Large object resources are not cloned per projection.

### Phase 6: Migrate analysis, measurements, exports, and declarative UI

- Make long-running operations actor-owned retained tasks backed by workers.
- Move declarative extension descriptors and state bindings into the actor.
- Ensure extension actions invoke actor commands directly.
- Keep native widget focus and transient input state in egui.

Exit criteria:

- All entries in the application-surface parity manifest name an actor command
  or an intentional UI-only exclusion.

### Phase 7: Remove RootApp semantic mutation paths

- Delete the legacy RootApp control dispatch queue.
- Remove duplicate model fields from viewer structs.
- Replace direct native mutations with commands or explicitly renderer-local
  state.
- Make accidental direct mutation difficult through module visibility and
  constructors.
- Remove the temporary canvas-deferred request mechanism once no migrated
  command depends on it.

Exit criteria:

- Protocol and native semantic mutations share the actor path.
- RootApp can be paused without preventing model/query command completion.

### Phase 8: Stabilize and document

- Run cross-platform foreground/background acceptance testing.
- Publish readiness and execution-class contracts in the protocol reference.
- Regenerate sync and async Python documentation.
- Benchmark actor latency, projection allocation, and frame planning.
- Update the feature parity manifest and declare any remaining exclusions.

## Proposed Module Layout

```text
src/
├── model/
│   ├── mod.rs
│   ├── app.rs
│   ├── document.rs
│   ├── readiness.rs
│   ├── revisions.rs
│   ├── viewport.rs
│   ├── project.rs
│   ├── layers.rs
│   └── projection.rs
├── control/
│   ├── actor.rs
│   ├── handle.rs
│   ├── envelope.rs
│   ├── dispatch.rs
│   ├── registry.rs
│   ├── events.rs
│   └── tasks.rs
├── render/
│   ├── projection_sync.rs
│   └── ... existing render resources
└── root_app.rs
```

Domain modules may remain near their current implementation while being
extracted. The important boundary is ownership and dependency direction:
`model` must not depend on egui, eframe, OpenGL, wgpu, or native window types.

## Test Strategy

### Actor unit tests

- Execute commands without constructing egui or eframe.
- Verify revisions, link propagation, conflicts, events, and task states.
- Pause projection consumption and confirm commands still complete.
- Saturate mailboxes and verify bounded backpressure.
- Inject stale worker completions and verify generation safety.

### Model/render contract tests

- Feed projections into a deterministic fake renderer.
- Report exact and derived geometry and verify reconciliation.
- Compare native-command and protocol-command results from the same starting
  model.
- Replay command logs and compare final model snapshots.

### Protocol and Python integration tests

- Run Odon with a test renderer whose frame clock can be paused.
- Open the fixture, create two viewports, configure channels, and fit cameras
  while frames are paused.
- Assert the Python script completes and the final projection is correct.
- Resume frames and assert the rendered state matches without extra commands.
- Test synchronous blocking and asynchronous cancellation independently.

### Platform acceptance tests

On macOS:

1. Start Odon and expose the synthetic fixture.
2. Completely cover Odon with a maximized terminal.
3. Run `two_viewer_comparison.py`.
4. Confirm the script completes without switching applications.
5. Return to Odon and confirm both viewports render correctly.
6. Repeat with Odon minimized and on another Space.
7. Verify presentation tasks accurately report waiting if macOS prevents pixel
   capture.

Repeat the covered/minimized cases on Windows and Linux where supported.

### Existing regression suites

Every phase must keep:

- Rust unit and data-contract tests;
- Python SDK tests;
- application-surface documentation generation;
- multi-viewport ownership and persistence tests;
- project round-trip tests; and
- renderer cache/performance characterization.

## Performance Requirements

- Median local model-command latency below 5 ms when no worker is required.
- No frame waits in model-command latency.
- Bounded command and worker-completion queues.
- Camera/pan projection allocation independent of image and object data size.
- Projection coalescing permitted; semantic events never coalesced silently.
- No actor-held lock during filesystem, network, image decoding, Parquet, or GPU
  work.
- UI projection application should remain below the existing multi-viewport
  frame-planning budget.

## Compatibility and Rollout

- Preserve existing protocol method names, request shapes, and successful
  response shapes during migration.
- Treat corrected readiness behavior as a bug fix, documenting any task-phase
  additions.
- Keep a temporary compatibility dispatcher for unmigrated methods.
- Add diagnostics showing whether each method executed through `actor` or
  `legacy_ui`.
- Gate each migrated command family with an internal development switch until
  its parity tests pass; do not expose two public modes to SDK users.
- Commit by vertical slice so a problematic family can be reverted without
  reverting the actor foundation.

## Risks and Mitigations

### Duplicate state during migration

Risk: actor and RootApp copies diverge.

Mitigation: each field has exactly one canonical owner. Compatibility adapters
translate at the boundary; they do not establish ongoing bidirectional sync.

### Native interaction latency

Risk: routing panning through an actor makes the UI feel sluggish.

Mitigation: use optimistic renderer-local previews and coalesced navigation
commands, reconciled by actor revisions.

### Projection copying

Risk: frequent state publication copies large channel, object, or annotation
collections.

Mitigation: immutable `Arc` subtrees, structural sharing, stable resource
handles, and generation IDs.

### Worker races after document replacement

Risk: an old loader installs results into a new document.

Mitigation: every worker request and completion carries a document generation;
the actor rejects stale completions.

### Project persistence drift

Risk: saved projects reflect renderer-local state rather than canonical state.

Mitigation: serialize actor snapshots only; presentation reports update logical
geometry through explicit actor messages.

### Screenshot expectations

Risk: users assume every command can finish while macOS prevents drawing.

Mitigation: classify screenshot and rendered-pixel operations as presentation
tasks with visible waiting phases and cancellation. All non-pixel semantic
commands remain background-safe.

### Scope growth

Risk: attempting to extract the complete app blocks delivery of the original
fix.

Mitigation: viewport/camera and dataset/channel vertical slices are independently
shippable and must solve the comparison script before later domains migrate.

## Acceptance Criteria for the First Shippable Milestone

The first milestone is complete when all of the following hold:

- The actor remains responsive when `RootApp::update` is deliberately paused.
- Dataset open reaches model/resource readiness without a rendered frame.
- Viewport creation, layout, rename, links, camera, plane, channels, colors, and
  rendering preferences execute through the actor.
- Logical geometry is retained or derived across dataset and layout changes.
- `fit_camera()` completes with Odon completely covered by another application.
- `two_viewer_comparison.py` completes while Odon is covered.
- Returning to Odon shows the correct two-view comparison without another API
  call.
- Native UI operations and Python operations produce equivalent model state,
  revisions, and events.
- Existing Rust and Python test suites pass.
- Diagnostics contain no `legacy_ui` execution for the migrated methods.

Current evidence (2026-08-21): `comparison_workflow_completes_over_tcp_without_a_ui_frame` opens
the checked-in fixture and executes the comparison plus complete channel presentation, panels,
intensity I/O, and fit without draining the UI queue. Model/renderer projection tests apply the
coalesced state atomically and compare renderer snapshots. Direct characterization tests compare
native implementation results, revisions, topology, and actor results. These tests establish the
actor/projection behavior but do not alone prove the literal external Python process and native
window conditions. Before declaring the milestone complete, run the full Rust and Python suites
and record the macOS visible/covered/minimized/separate-Space workflow, plus corresponding
Windows/Linux cases where available, against the built application.

## Original First Implementation Sequence (Completed Baseline)

Use small reviewable commits:

1. Add paused-frame regression harness and command timing diagnostics.
2. Add model identifiers, readiness types, actor mailbox, and lifecycle.
3. Route read-only application/viewport queries through the actor.
4. Add logical geometry and UI presentation reports.
5. Migrate camera and plane mutations, including native gestures.
6. Migrate viewport workspace layout, links, and titles.
7. Split dataset open into worker result plus actor installation.
8. Migrate channel presentation and rendering preferences.
9. Switch open-task completion to model/resource readiness.
10. Run the macOS covered-window acceptance workflow and remove migrated
    deferred-UI paths.

This sequence delivers the background-control fix before the remainder of the
application is extracted, while establishing the architecture needed for the
complete Python API.
