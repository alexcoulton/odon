# Zoom Segmentation Freeze Remediation Plan

Status: implemented; automated and native validation in progress

Date: 2026-08-27

## Outcome

Odon must remain responsive when a large polygon segmentation is visible and the user pans or
zooms, including when **Use proxy points at low zoom** is disabled. Continuous fills must remain
available at useful overview scales, and switching property, palette, domain, filter, or opacity
must not rebuild segmentation geometry or raster tiles.

The target renderer is a hybrid:

- a lazy, world-aligned, multiresolution cell-ID tile pyramid for low and medium zoom fills;
- spatially culled vector polygons for close zoom, where relatively few cells are visible;
- ID-tile selection fills plus live object-indexed outlines for hover and selected cells; and
- the existing per-object state and RGBA lookup textures for visibility and colour.

This is a renderer change. It does not require changes to the Python SDK contract or the Amy
comparison workflow.

## Implementation Record

Milestones 0-4 are implemented in the native renderer. The implementation includes byte-accounted
mesh, lookup-texture, completed-tile, and unfinished-tile residency; spatial fill-bin queries and
scissoring; `R32UI` cell-ID rasterization and live colour lookup; world-aligned power-of-two tile
levels; coarse-first, centre-prioritized refinement; resumable vertex-budgeted tile rasterization;
bounded unfinished work; and geometry-only tile invalidation. Workspace performance observations
report residency, budgets, requests, hits, completions, discards, evictions, exact pending work,
draw counts, submitted vertices/triangles, and timing.

Analysis and interactive polygon selection now update a per-object state texture over the resident
ID tiles instead of rebuilding selected-only fill or outline geometry. The selected and normal
cell colours are resolved in one tile-composition pass, so a previous translucent selection cannot
survive as a separate retained layer. Close-zoom outlines reuse
the object-indexed line LODs prepared when the resource loads. Diagnostics separately report
selection tile-composition draws so this path can be verified without inferring it from appearance.

The scheduler deliberately performs small resumable GPU jobs from the current paint requests
instead of maintaining a background CPU result queue. Unfinished textures are never composited.
Only currently requested keys are advanced, and the strict unfinished-tile LRU bounds obsolete
work from navigation or multiple viewports. Because no asynchronous CPU result exists, there is no
stale CPU completion to upload; `tile_discarded` records unfinished GPU tiles evicted before
publication.

Milestone 5 remains intentionally unimplemented: runtime tiles are not persisted to disk because
no measured reopening-latency result yet justifies that complexity. Automated Rust validation is
part of this change. The final Amy interaction and memory smoke remains a user-run release gate, as
specified below.

## Current Behaviour and Working Diagnosis

The current continuous-fill path already has a good property-colour architecture:

- `ObjectFillMesh` stores an object index with every fill vertex;
- `ObjectFillGlRenderer` uploads one state texture and one optional RGBA colour texture indexed by
  object; and
- palette, domain, property, filter, and selection changes can update those small per-object
  payloads without rebuilding the triangle mesh.

The expensive part is the geometry submission. In the main fill path,
`src/objects/render.rs` submits `object_fill_mesh.vertices_local`, which is the complete
tessellated segmentation, whenever its overall bounds intersect the viewport. The mesh contains
spatial `bin_vertices`, but they are currently used only by the large-selection fill overlay.
Consequently, disabling low-zoom proxy points can cause the complete fill mesh to be drawn for
each requested interactive frame, even if only a small part is visible.

The OpenGL mesh cache is bounded by entry count rather than bytes. Entry-count bounds are not a
sufficient safety guarantee because individual VBOs and future tile textures can differ greatly
in size. Already-submitted OpenGL work cannot be cancelled, so long whole-mesh draws can continue
to delay newer frames even if application-level redraw requests coalesce.

The progressive slowdown observed on the large Amy ROI is consistent with repeated expensive GPU
work during zoom, but the exact contribution of GPU saturation, allocations, cache growth, and
driver behaviour has not yet been measured. The first milestone therefore adds evidence and hard
resource ceilings before treating the diagnosis as proven.

## Architectural Decision

### Fill representation

Each segmentation tile stores a dense object index rather than a finished colour:

```text
world polygons
    -> offscreen rasterization
R32UI object-index tile (0 = background, object index + 1 = cell)
    -> existing object state and RGBA lookup textures
displayed continuous or categorical colour
```

A 32-bit object-index tile uses four bytes per pixel, the same storage as an RGBA8 colour tile.
Changing the marker, Nimbus/median source, colour domain, palette, opacity, or filter updates the
existing per-object lookup payload; it does not invalidate geometry tiles.

Use compact internal object indices, not external feature IDs. Verify `R32UI` framebuffer and
sampling support through the current OpenGL 3.3/glow path in a small prototype before committing
the complete renderer. If a target backend cannot use an integer render target reliably, define a
tested lossless RGBA8 ID packing fallback rather than silently switching to pre-coloured tiles.

### Tile layout and levels

- Align the segmentation tile grid and level transforms with the primary image pyramid where
  practical, while keeping the object source transform explicit.
- Use fixed world-aligned `(level, tile_x, tile_y)` keys. Camera position is never part of a key.
- Include the object-resource identity, geometry generation/fingerprint, and spatial transform in
  the cache namespace. Styling generations are not part of the tile key.
- Rasterize every level deliberately. Do not use normal mipmap averaging because averaged object
  IDs are meaningless.
- Define deterministic overlap behaviour using source/object order and test it.
- Initially use nearest ID sampling and live vector outlines. Treat antialiased fill coverage as a
  later, separately measured enhancement; an optional coverage texture must not invent blended
  object IDs.

At extreme overview scales, several cells can occupy one screen pixel. The ordinary object view
can select a deterministic representative ID, but must not imply that this is an aggregate
measurement. A population-density or mean-value overview would be a separate analytical layer
with explicit aggregation semantics.

### Hybrid level selection

- At low and medium zoom, draw the closest available cell-ID tile level.
- If the requested level is absent, draw a coarser cached level immediately and request only the
  missing visible tiles.
- At close zoom, switch to spatially culled vector fills so boundaries remain crisp beyond the
  finest tile resolution.
- Draw selected-cell fills as a state lookup over the cached IDs. Reuse prebuilt object-indexed line
  LODs for close-zoom selection outlines and hover feedback rather than rebuilding selected-only
  geometry after every Analysis brush change.
- Preserve screen-space outline widths independently of fill LOD.

Disabling **Use proxy points at low zoom** means that the user sees filled segmentation rather
than centroid proxies. It must not mean that Odon submits the complete ROI mesh every frame.

### Expected performance trade-offs

The hybrid path should be selected from measured object count, visible geometry, zoom, and
available GPU features rather than applied indiscriminately:

- Small segmentations can retain the current direct renderer, avoiding tile-management overhead.
- Large low/medium-zoom views exchange repeated whole-mesh draws for a small number of texture
  draws and per-pixel object-colour lookups.
- First display of an uncached area has bounded background rasterization cost and may temporarily
  show a coarser level.
- The tile cache consumes additional GPU memory, but only up to an explicit byte budget; it must
  not grow by camera position or compete without accounting with the image-tile cache.
- Close-zoom spatial culling can increase draw-call count while greatly reducing submitted
  triangles. Bin/tile size and the vector-to-tile transition must therefore be benchmarked rather
  than guessed.
- Property and colour edits continue to update the existing small per-object payload and should
  not incur tile-generation cost.

The performance gate covers ordinary small datasets and image-only navigation as well as the
large Amy case. The new renderer must not fix large-ROI interaction by materially regressing those
paths.

## Implementation Plan

### Milestone 0: reproduce, instrument, and impose safety ceilings

Add renderer diagnostics before changing the architecture:

- CPU and GPU bytes for object fill meshes, object state/colour textures, and ID tiles;
- visible, resident, pending, completed, discarded, and evicted tile counts;
- triangles submitted and fill draw calls per frame;
- tile-rasterization and tile-composition timings;
- cache hits/misses and peak in-flight work; and
- a monotonically increasing viewport request generation for stale-work accounting.

Expose a compact form through the existing workspace/performance diagnostics and keep detailed
logging behind a debug option. Add strict configurable byte ceilings and bounded in-flight work;
do not rely only on LRU entry counts.

Create a deterministic large synthetic object fixture or generator for automated benchmarks.
Use the external Amy sample only for local manual acceptance; do not add Amy data, scripts, or
workflow artefacts to the repository. GUI input automation is not required for this plan: the
camera sequence can be driven by a renderer benchmark, and the user can perform the final native
interaction test.

Exit conditions:

- the current failure sequence has a recorded baseline;
- diagnostics distinguish cache growth from repeated expensive draws;
- caches and in-flight work cannot exceed their configured budgets; and
- exhausting a budget degrades quality or delays refinement without allocating without bound.

### Milestone 1: build the bounded spatial-geometry foundation

Build the shared spatial-geometry access required by both the tile rasterizer and the final
close-zoom vector fallback. This is not a separate user-facing renderer or a stopgap to ship in
place of the multiresolution solution:

- use `ObjectFillMesh::bin_range_for_local_rect` and its spatial bins to obtain only the triangles
  intersecting a requested world tile or close-zoom viewport;
- allow the bounded direct path to consume the same query for sufficiently large close-zoom
  fills, not only the selected-fill overlay;
- submit only bins intersecting the transformed visible rectangle;
- scissor each bin to its world/screen tile rectangle, or partition geometry without duplication,
  so triangles copied into adjacent bins do not blend repeatedly;
- make GPU mesh residency byte-budgeted and evictable;
- keep geometry cache keys independent of camera state; and
- cap work submitted in a frame so input and image-tile composition get regular opportunities to
  run.

This work is part of the proper architecture: the pyramid generator must find polygons for one
tile without scanning or drawing the whole ROI, and the hybrid renderer needs a safe exact-vector
path at high zoom and on unsupported GPUs. It may reduce the failure while development is in
progress, but it is not the planned end-user resolution on its own.

Exit conditions:

- panning a close view submits work proportional to visible bins, not the full ROI;
- repeated pan/zoom does not create new geometry entries for camera states;
- opacity is not multiplied by duplicated bin draws; and
- the large synthetic camera sequence has bounded peak memory and no progressive frame-time
  growth.

### Milestone 2: prove one cell-ID tile end to end

Add a narrow OpenGL prototype that:

1. creates an offscreen framebuffer with an integer object-index attachment;
2. rasterizes the intersecting indexed fill triangles into one world-aligned tile;
3. composites that tile through the existing object visibility/state and RGBA colour textures;
4. changes continuous property, palette, domain, opacity, and filter without rerasterizing it; and
5. compares the result with the current direct-vector fill at a reference camera.

Reuse the existing continuous `ObjectColorPayload` initially. It already converts the selected
numeric property and colour mapping into one RGBA value per object. The first tile milestone does
not need a new raw-value shader or SDK-facing colour contract.

Exit conditions:

- one ID tile renders continuous fills correctly;
- changing style updates only the per-object payload;
- background, missing, filtered, and out-of-range objects remain transparent as specified; and
- integer-ID precision and object-count limits are explicit and tested.

### Milestone 3: add lazy multiresolution scheduling

Introduce an object-fill tile scheduler with latest-visible-state priority:

- compute the visible tile set from the current camera and object transform;
- request the target level first, retaining a coarser available fallback;
- use a viewport generation token on every request and completion;
- discard results whose geometry namespace or requested visibility generation is stale;
- stop queued CPU work cooperatively and never upload a completed stale tile;
- bound worker concurrency and GPU uploads per frame; and
- atomically publish completed tiles so the current frame never sees a partial tile.

OpenGL commands already submitted cannot be cancelled. Keep tile rasterization jobs small enough
that any one obsolete submission has a bounded cost. Coalesce navigation changes before creating
new work and retain only the newest desired visible set.

Cache policy:

- GPU cache: strict byte-budgeted LRU, biased toward current visible tiles;
- CPU staging cache: separately byte-budgeted and released after upload unless reuse is measured
  to justify retention;
- coarse overview: retained while the object resource is active when it fits its budget; and
- no persistent disk cache in the first interactive milestone.

Exit conditions:

- zooming out immediately uses a coarser resident level instead of redrawing the full geometry;
- rapid pan/zoom does not produce an ever-growing queue;
- stale CPU results are not uploaded and stale GPU work is limited to at most the configured
  in-flight bound; and
- style changes do not change the ID-tile hit rate or enqueue raster work.

### Milestone 4: integrate the hybrid renderer and user semantics

Route object fills through the new hybrid policy:

- small object sets may continue to use the direct path;
- large low/medium-zoom views use cell-ID tiles;
- close zoom uses the culled direct-vector path;
- selection fills reuse ID tiles, while selection outlines and hover reuse prebuilt geometry; and
- proxy points remain an optional presentation preference, not the only performance safeguard.

Ensure that single-view, explicit viewport, and mosaic presentations share immutable object
geometry and tile resources where geometry/transform namespaces match, while retaining
viewport-owned style payloads and camera-dependent visible sets.

Handle renderer loss or unsupported GPU features safely: keep the bounded direct path as the
fallback and report the degraded mode in diagnostics. Never fall back to an unculled whole-ROI
draw for a large resource.

Exit conditions:

- **Use proxy points at low zoom** on and off are both safe on a large ROI;
- continuous fills remain visible through zoom-level changes;
- outlines keep a useful screen-space width;
- selection and hover are accurate during coarse fallback and after refinement; and
- multiple viewports do not duplicate the underlying geometry pyramid unnecessarily.

### Milestone 5: persistence only if runtime evidence justifies it

After the runtime cache is stable, measure first-view tile-generation latency. If it materially
impairs reopening large samples, add an optional persistent local cache:

- fingerprint source geometry, transform, tessellation/raster rules, tile size, and renderer
  format version;
- write tiles atomically and tolerate incomplete cache directories;
- enforce a disk byte budget with recoverable eviction;
- validate cached metadata before reuse; and
- make deletion safe because the cache is always regenerable.

Precomputed pyramids beside source data can be considered later as an optional import/data-pipeline
optimization. They are not required to fix the interactive freeze.

## Correctness Tests

- Tile and direct-vector output agree at representative cameras, allowing only documented edge
  sampling differences.
- Object transforms, negative offsets, non-unit scale, and viewport clipping select the correct
  tile and object IDs.
- More than 65,535 objects render without ID truncation; the documented 32-bit limit is enforced.
- Background ID zero, null values, out-of-range hiding, filters, and invisible categories are
  transparent.
- Continuous median/Nimbus switching, domain edits, palette reversal, and opacity edits do not
  invalidate ID tiles.
- Geometry reload, source replacement, and transform changes do invalidate the correct namespace.
- Overlapping polygons have deterministic precedence.
- Selection fills do not invalidate ID tiles, and selection/hover overlays remain aligned across
  tile-to-vector transitions.
- Context loss and renderer recreation release and regenerate GPU resources safely.

## Performance and Failure Tests

Run a repeatable camera trace containing small alternating zooms, a sustained zoom-out, a pan, and
a return to the initial view. Record:

- median and 99th-percentile frame-planning and fill-render time;
- peak CPU and GPU cache bytes;
- maximum queued/in-flight tile jobs;
- stale jobs discarded before rasterization and before upload;
- triangles submitted by the vector fallback; and
- whether memory and latency return to a steady state after interaction stops.

Required gates:

- no monotonic cache or queue growth across repeated traces;
- resident bytes never exceed their configured budgets apart from one documented bounded upload;
- whole-ROI triangle submission is absent from large low/medium-zoom frames;
- the final requested view reaches exact presentation readiness after interaction stops;
- colour/property changes complete without geometry-tile regeneration; and
- the application remains responsive enough to accept further navigation while refinement is
  pending.

Manual release smoke on the large Amy sample:

1. load one large ROI with continuous fill;
2. disable **Use proxy points at low zoom**;
3. alternate small zoom-in/zoom-out movements, then zoom out substantially;
4. switch median/Nimbus and several channels;
5. edit palette/domain/opacity;
6. return to the original view; and
7. confirm stable memory, responsive controls, correct fill, and eventual exact readiness.

The user performs this native interaction smoke; no AppleScript test is required.

## Rollout and Stop Conditions

Land the work behind an internal renderer capability flag until Milestones 0-3 pass. Keep the
bounded vector path available for comparison and fallback. Make the hybrid renderer the default
for large object resources only after the Amy smoke and synthetic performance gates pass.

Stop and revisit the ID-tile design if the prototype shows any of the following:

- integer framebuffer support is unreliable on a required platform;
- per-pixel ID plus state/colour lookups cost more than the culled direct path at the intended
  zoom ranges;
- tile generation cannot be divided into sufficiently small bounded jobs; or
- low-resolution ID sampling produces scientifically misleading overview behaviour that cannot be
  made explicit.

In that event, retain the diagnostics, budgets, and spatial-geometry foundation from Milestones
0-1 while comparing a pre-coloured tile cache or a more aggressively culled vector renderer. The
alternative must still satisfy the same bounded-memory and visible-work performance gates; it
must not be treated as an unmeasured temporary patch.

## Definition of Done

- The large-ROI zoom sequence cannot create unbounded queued work or cache growth.
- Disabling proxy points never triggers an unculled whole-segmentation draw at low/medium zoom.
- Continuous segmentation fills remain useful while zooming and become exact after refinement.
- Median/Nimbus, marker, palette, range, filter, and opacity changes do not rebuild ID tiles.
- GPU and CPU object-renderer memory are measured and enforced by byte budgets.
- Automated synthetic correctness/performance tests and the user-run Amy native smoke pass.
- Documentation explains the hybrid fill behaviour, fallback mode, cache diagnostics, and the
  distinction between cell-level display and any future aggregated overview.
