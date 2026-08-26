# Current Python API limitations

The Python SDK exposes Odon's current semantic application surface, but that
does not mean every visual or computational detail is replaceable from Python.
This page is the explicit boundary for the current implementation.

## Background-control migration

The actor-owned local OME-Zarr and multi-viewport comparison slice progresses
without GUI frames: open tasks, viewport creation/layout/rename/linking,
logical camera fit, plane state, channel visibility/color/contrast, and
channel notes/transforms/order/groups/search, side-panel visibility, channel
intensity statistics, per-viewport object appearance/property/legend presentation, and
per-viewport rendering preferences are
background-safe. Project creation/open/save, metadata and ROI editing/navigation,
saved-view CRUD/capture, referenced-resource registration, and
external-layer installation/style/visibility/order/removal now use the same
no-frame actor path. Resource/layer mutations and project saves share one
serialized mailbox, so a save issued immediately after a layer call includes
that layer. Samplesheet inspection/validation/import/export and recursive
OME-Zarr project discovery likewise execute in bounded background workers and
commit through the actor. Their renderer
changes coalesce into one latest-value projection and are displayed when Odon
next receives a frame, rather than replaying every intermediate visual state.
Renderer observations carry the last applied projection revision and cannot
overwrite newer actor-owned values when an old frame resumes after occlusion.
The equivalent active-view camera, plane, channel, panel, and smooth-pixel
methods also use this path. Dataset metadata opening and channel-statistics I/O
are admitted through a bounded worker pool, so repeated requests cannot create
unbounded worker threads. If an actor-owned command arrives before an
asynchronous open reaches model/resource readiness, it returns explicit
`NOT_READY`; it is never rerouted to the GUI queue.

Single-image native layer inventory, active layer, visibility, independent viewport
presentation/order, and world offsets are actor-owned as well. Native layer changes made in the
Rust UI commit back as one viewport presentation transaction with a revision guard. The actor
accepts newly discovered compatibility-layer descriptors without allowing a delayed renderer
frame to replace presentation already committed through Python.

Local NGFF label discovery, load, visibility, and unload are background-safe as well. Label
metadata is opened by the bounded worker pool and committed independently of GPU presentation;
when frames resume, the renderer creates its label tile loader from that shared resource.

Mask layer and polygon CRUD, mask selection/undo, GeoJSON import/export, and
in-memory project synchronization are also actor-owned. GeoJSON filesystem work
runs on the bounded worker pool. A native committed mask edit is replayed as one
generation-checked actor transaction, so it cannot silently overwrite a newer
Python edit; the versioned mask projection is rendered when frames resume.

Point annotation identity, style, column mapping, categorical/continuous presentation, and
project persistence are actor-owned. Parquet schema inspection and point loading run on the bounded
worker service. Their immutable indexed datasets are projected to the single-image or mosaic
renderer by shared handle, so annotation commands and tasks progress while Odon is covered.

All registered application methods now have actor execution routes. `system.get_diagnostics`
reports queue/model/reply/presentation timing and remains the best way to distinguish completion of
semantic or resource work from its later display. Pixel capture is the intentional exception: it
waits for a generation-specific presentation acknowledgement while leaving the actor responsive.

Primary object source loading, reload/clear, paginated property access, per-viewport appearance,
legends, and filters now use the background actor. The source worker produces both the canonical
property/spatial index and a shared renderer preload; returning to Odon therefore installs the
already parsed geometry rather than starting the import again. Primary-object queries, selection,
focus, and native selection commits also use the actor, including worker-evaluated standalone
filter selection. Explicit `target="active"`, `target="spatial_shape"`, and screen-coordinate
rectangle requests retain the compatibility path until spatial layers and exact canvas origins
move into the canonical model. Analysis and export remain on the compatibility path at this stage.

## Two-viewport milestone limit

Single-image mode supports one or two native Rust viewports, arranged
horizontally or vertically. Python can address them by stable ID and configure
independent camera, plane, channels, object style/filter, overlay visibility,
layer order, sampling, HUD, scale-bar, tile-debug and active-layer state. The
split ratio is configurable from `0.1` through `0.9`. Camera and plane links are configurable;
object selection and scientific edits remain document-shared.

The current boundary is deliberately two canvases. There is no 2×2 grid,
arbitrary layout tree, detached viewport window, or cross-document split yet.
Both canvases always show the same open document and share its source, caches,
object geometry, edits, and selection. Mosaic mode remains a different
one-canvas layout containing several ROI items.

Filter-sensitive operations require an explicit viewport, standalone filter,
or all-object choice in a two-view workspace. This is a deliberate safety
contract, not an omitted convenience.

## Declarative single-window shell

Python can register first-class extension component trees and mount them beside
built-in components in a recursive keyed shell. The `right.tabs`,
`left.sections`, `top_bar.actions`, `status_bar`, `canvas.controls`, and
`project.cards` placement hints render through application-owned mounts in the actor tree rather
than external egui panels or windows. It can inspect
the versioned project, single-viewer, and mosaic shell trees through
`app.ui.shell`. For the active shell it can atomically replace the topology,
show or hide nodes, select tabs, resize nested splits, collapse regions, apply
an optimistic shell-revision guard, and restore defaults.
Layouts can be exported/imported as versioned documents, saved as session,
durable application, or project-owned profiles, migrated from the v0 document
shape, and replaced with a protected minimal recovery layout. Extensions can
register owned, validated default templates that follow their disconnect
policy and are applied explicitly through shell import. Application settings
can select a startup profile per mode; the actor restores it once on first
activation and falls back to the protected recovery tree with diagnostics.
Migrations beyond v0-to-v1 remain future work because no newer document schema
exists yet.
Inactive shells are inspectable, but must become active before mutation. The
formal schema is available from `app.ui.shell.describe_schema()`, and Python
returns typed snapshots/nodes with stable `ShellId` constants.
Built-in mount IDs and their compatibility, sizing, command, event, and
persistence metadata are discoverable through `app.ui.shell.list_components()`.
Snapshots and component descriptors also expose application/extension ownership and protected
status. Extension-session mutations are checked inside the actor and cannot change a foreign
extension or ungranted application-owned node. Application-controller authority is negotiated
explicitly during `system.hello`; authentication alone grants no shell mutation authority. Native
recovery remains privileged. Per-mutation transaction correlation is implemented, while a true
multi-method atomic batch remains future work if workflows require it.
Active region and focus are actor-owned and revision guarded. Native tabs, collapsibles, mount
activation, and completed split drags emit the same semantic shell changes as Python; focus can be
cleared explicitly. Disconnect/reconnect focus transfer, cross-mode isolation, and high-contention
revision races have actor/bridge coverage; cross-platform rendered evidence remains future work.
Mount configuration is retained and revision guarded. Native top bars publish and honor non-empty
schemas; most other built-ins intentionally publish empty schemas until they have meaningful
per-instance options, and extension configuration-version compatibility remains future work.
Channels, single-view viewport controls, the documentation browser, protected recovery controls,
the shell inspector, and the command toolbar can be mounted independently. Catalogue/renderer
conformance is tested for every advertised mode; further subdivision of composite controls remains
demand-driven rather than exposing arbitrary egui widgets.

Python can inspect stable application commands separately from their menu presentations and
atomically replace a bounded nested platform-menu tree. On macOS this rebuilds the real native menu,
including submenus, separators, accelerators, protected recovery, and checked scale-bar state.
Extensions can register owned, namespaced event commands. Shortcut conflicts are rejected across
overlapping modes using platform-effective `primary` aliases. macOS uses native menu accelerators;
Windows and Linux resolve every eligible descriptor from the current actor projection and dispatch
it through `ui.commands.execute`, so changed shortcuts do not leave stale registrations. The
schema reports the platform mapping and rejects non-realizable modifiers with diagnostics. The
native menu, mounted command toolbar, and searchable command palette apply mode and
extension-readiness enablement. Python can replace the bounded toolbar and palette
presentations. Bounded capability and actor-state predicates now produce shared visible, enabled,
checked, reason, and missing-capability state. Toolbar items support checked/icon/tooltip/label
presentation. Extension components can bind visibility and enablement to that evaluated command
state with bounded command IDs/state fields, and shell nodes can bind visibility to the same state.
Arbitrary predicate paths, Python predicate callbacks, and user-remappable shortcuts are
intentionally not implemented.

By design, this initiative does not attempt to:

- define arbitrary docking or create detached native windows;
- control monitor placement or move canvases between native windows;

The current API also does not:

- define user-remappable shortcut bindings;
- define arbitrary state expressions or Python callbacks outside the bounded predicate catalogue;
- create arbitrary egui widgets or run Python on the GUI thread;
- inject HTML, JavaScript, or CSS; or
- intercept every native interaction before Odon handles it.

The viewport workspace controls multiple canvases within the mounted viewer
canvas component, while the shell API controls its surrounding recursive
composition. Rust remains responsible for native rendering, input, GPU
resources, platform windows, validation, and safe placeholders. Arbitrary docking and multi-window
composition are outside the scope of the single-window shell API rather than unfinished work in
its completion plan.

## Renderer coverage differs from descriptor coverage

The data-resource registry accepts OME-Zarr/Zarr, Arrow IPC, Parquet,
GeoParquet, and GeoJSON descriptors. Registration establishes identity,
coordinates, ownership, and lifecycle; it does not promise live rendering for
every format.

Current live external rendering is strongest for:

- local OME-Zarr/Zarr images and labels; and
- local Parquet/GeoParquet shapes, objects, and points.

Arrow IPC and registered GeoJSON can be useful as descriptors even where no
live renderer consumes them. Renderer support must be checked separately from
successful registration.

## Labels and property-filled cells

The native NGFF label overlay currently renders outlines rather than arbitrary
per-label property fills. Property-driven fill, legends, filtering, and
selection are provided by object/polygon overlays. A workflow with a raster
label image plus a separate value table may need conversion to supported
object geometry or a future label-property join and fill renderer.

## Multidimensional operations

XY, XZ, and YZ plane navigation is exposed for supported datasets, but several
editing, thresholding, measurement, and analysis operations remain XY-only.
Use `app.planes.get_operation_availability()` before enabling such operations
in an extension.

## Mosaic differences

Mosaic mode shares camera and channel presentation across its positioned ROI
items. Some single-view object, mask, analysis, measurement, label, memory,
alignment, and screenshot operations either have mosaic-specific equivalents
or are unavailable. Query method availability instead of assuming a single-
view call will apply per mosaic item.

## Long-running and external computation

Odon tasks are retained and asynchronous from the caller's perspective, but
cancellation is cooperative. A queued operation can be cancelled; a native
library call that does not expose interruption may finish before Odon can
discard its result.

Four execution contexts have different contracts:

- the synchronous SDK callback worker delivers raw and normalized events serially; it must not
  block on `Task.wait()`, which raises `UnsafeCallbackWaitError` there;
- extension `worker` and `serial-worker` actions run outside callback delivery and may wait for
  retained tasks; `serial-worker` is the safe default for related viewer mutations;
- the Odon control actor owns semantic state, revision checks, retained-task lifecycle, and bounded
  resource workers; Python workers do not bypass actor validation; and
- the render thread realizes actor projections and acknowledges presentation. Semantic or resource
  completion does not by itself prove that final pixels have been presented.

The Python `replace_object_source_and_style()` recipe orders existing actor commands and verifies
the new property, but it is not one actor-atomic command. Its neutral and final states can be
observed as separate revisions. Generation checks prevent an older Python action from committing a
newer controller's final state; another independent client can still interleave mutations unless
the application also uses the exposed revision guards. A future actor-owned compound command would
be required if no intermediate state may ever be observed.

Synchronous `ActionContext.patch()` and `commit()` hold the runner's generation lock across their
short commit. Their async counterparts check before and after an awaited actor call, but cannot make
that call atomic with a UI event arriving on the same event loop. Async workflows should use actor
revision guards where an intermediate stale patch would be unacceptable and always perform a final
`ensure_current()` before publishing readiness.

Odon does not embed Python. Cellpose and similar algorithms execute in the
external Python process. The extension is responsible for worker-process or
thread management for external computation, dependency installation, model lifecycle, chunking,
and durable output. Native UI task actions do not need a hand-written queue or thread in the common
case because `extension.on_action()` owns that lifecycle. The current Cellpose example is an end-to-end reference, not a
tiled production segmentation engine.

## Large data does not travel through JSON

Inline requests are bounded. Image arrays, tables, and large geometry are
registered by URI. `register_numpy()` is a convenience that writes a temporary
Zarr store before registration; it is not zero-copy shared memory. Very large
analysis results should be written in a supported referenced format.

## Session lifetime and persistence

Session-owned resources and UI contributions can disappear when Python
disconnects. Project-owned descriptors persist only when their dependencies
are also persistable and the project is saved. S3 credentials are session-only
and are never persisted or returned.

Retained tasks belong to a session. A reconnecting extension must rediscover or
recreate its UI and session-owned resources according to its disconnect policy.

## Experimental compatibility

The control surface is not yet declared stable v1. Canonical methods and Python
resource organization are deliberately structured for stability, but response
snapshot fields, UI vocabulary, style dictionaries, and renderer coverage can
still evolve. Extensions should:

- use high-level resource methods rather than deprecated aliases;
- check capabilities and method availability;
- use stable IDs rather than list indexes when possible;
- use revision guards for coordinated mutations;
- tolerate additive response fields; and
- inspect the connected build at runtime when using extensible dictionaries.

See the [Python API contracts](../reference/python-api-contracts.md) and
[generated member reference](../reference/python-api-reference.md) for the
documented current surface.
