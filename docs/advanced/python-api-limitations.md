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

Other application domains still use the compatibility UI dispatcher while the
central-model migration continues. In particular, resource-loading saved-view apply,
project ROI opening, mosaics, other native image resource loading, spatial-shape object selection,
annotations, threshold-mask computation, measurements, object exports, screenshots, remote stores, TIFF,
SpatialData, and Xenium have not all been moved to the actor yet. A method in
one of those domains may still require Odon's native event loop to advance.
Use `system.get_diagnostics` to inspect every method's current `actor`, `hybrid`,
`legacy_ui`, or independent `control_service` route and the actor's
queue/model/reply/presentation timing.

Applying a saved view is actor-owned when its referenced object/label resources are already
available. A preset that itself requests a missing object or label resource remains an explicit
hybrid route until saved-view application is connected to the actor resource workers; diagnostics report
`project.views.apply` accordingly. Capturing channel, camera, and object presentation from any
existing viewport is fully background-safe.

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

## Declarative UI extends predefined hosts

Python can add native component trees at `right.tabs`, `left.sections`,
`top_bar.actions`, `status_bar`, `canvas.controls`, and `project.cards`. It can
show or hide major side panels and select supported native tabs.

It cannot currently:

- replace or arbitrarily rearrange the complete application shell;
- instantiate native panels by stable component ID;
- redefine all menus, toolbars, docking, or shortcuts;
- create arbitrary egui widgets or run Python on the GUI thread;
- inject HTML, JavaScript, or CSS; or
- intercept every native interaction before Odon handles it.

The viewport workspace controls canvas composition while Rust remains
responsible for native rendering. It does not make the rest of the application
shell arbitrarily dockable.

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

Odon does not embed Python. Cellpose and similar algorithms execute in the
external Python process. The extension is responsible for worker-process or
thread management, dependency installation, model lifecycle, chunking, and
durable output. The current Cellpose example is an end-to-end reference, not a
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
