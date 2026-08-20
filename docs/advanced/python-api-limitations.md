# Current Python API limitations

The Python SDK exposes Odon's current semantic application surface, but that
does not mean every visual or computational detail is replaceable from Python.
This page is the explicit boundary for the current implementation.

## One application canvas and one viewport

Odon currently has one native viewer canvas. `app.viewer`, its camera, active
plane, channel presentation, native layers, objects, masks, and overlays refer
to that canvas. Mosaic mode positions several ROI images inside the same canvas;
it does not create independent viewports.

Consequences:

- Python cannot create two side-by-side native viewers in one Odon window.
- Camera and most presentation state cannot be scoped by a `viewer_id`.
- The same segmentation cannot currently have different per-viewport styling.
- Linked overview/detail or raw/processed comparison views require separate
  Odon processes or future multi-viewport support.

Two Odon processes can be controlled by one Python program and synchronized
through events, but they have separate windows, sessions, and resource caches.

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

A future workspace/layout API can address shell composition while keeping Rust
responsible for native rendering. Multi-viewport support is a separate viewer-
architecture requirement, not merely a layout feature.

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
