# Odon Low-Zoom Texture-Outline Panning Plan

## Status

Implemented in the shared object renderer; automated Rust tests and the release build pass. The
user-led single-view/three-ROI/ten-ROI visual and performance matrix remains before final
acceptance. Odon has not been launched automatically for visual testing.

Implemented components:

- camera-independent cached visibility and continuous-outline state, with explicit invalidation,
  retained-byte accounting, rebuild counters, and source-replacement cleanup;
- two-texel physical gutters around each logical 512 x 512 object-ID tile, including exact byte
  accounting and corrected ancestor/interior UV addressing;
- texture-derived continuous, categorical, single-colour, selected, and primary borders;
- a two-level-finer overview ID target with one-level fallback plus an adaptive 2 x 2,
  coverage-weighted compose resolve, so subpixel cells contribute proportionally instead of one
  nearest-neighbour ID winning each display pixel;
- an adaptive low-zoom texture/vector resolver with workload thresholds, hysteresis, tile-coverage
  fallback, and exact-vector inspection mode;
- per-layer single-view diagnostics and aggregated mosaic diagnostics for resolved mode, reason,
  submitted vector work, texture-border and multisample-compose work, planning time, coverage,
  transitions, and cached presentation state;
- regression coverage for state reuse/invalidation, resolver hysteresis, tile keys, gutter bounds,
  ancestor UVs, edge classification, selection semantics, diagnostics exposure, and source
  organization.

## Objective

Make panning very large object/segmentation layers smooth in both single-image and mosaic viewers,
without replacing cells with proxy points or weakening close-zoom polygon inspection. The Amy
ten-ROI mosaic is the primary stress benchmark, not the boundary of the implementation.

At low zoom, Odon should compose both continuously coloured fills and visible cell borders from
the existing world-aligned object-ID texture tiles. At inspection zoom, it should retain the
current exact vector-outline renderer. Panning must not rebuild segmentation tiles, property
colours, or polygon geometry.

## Scope

The core implementation belongs in the shared `ObjectsLayer`, object-ID tile renderer, and
texture/vector mode resolver. It must therefore apply consistently to:

- a large segmentation loaded in the ordinary single-image viewer;
- segmentation objects used by single-view Analysis and live selection;
- each object/segmentation layer presented inside a mosaic.

Mosaic-specific work is limited to aggregating per-layer diagnostics, using the Amy mosaic as the
stress test, and the optional later optimization that batches tile draws across ROIs. Do not gate
the texture-border path on mosaic mode.

This plan does not extend the renderer to masks, point annotations, NGFF labels, or unrelated
native layer types. Those can adopt the shared mechanism separately if their rendering models
later converge.

## Measured Stress Case

The Amy ten-ROI mosaic currently contains:

- 10 simultaneously visible ROIs;
- 1,480,967 segmented cells;
- 86 unified markers with median raw, median flat-field, and Nimbus fills;
- a camera zoom of approximately 0.00545 screen pixels per level-0 pixel when fitted;
- `Always polygons` enabled.

The live renderer snapshot after marker cycling showed:

| Component | Current retained/work state |
| --- | ---: |
| Image tiles | 14 tiles / 3.0 MB / no in-flight work |
| Object-ID fill tiles | 68 tiles / 71.3 MB / no tiles generated in the sampled frame |
| Object fill mesh pool | 321 entries / 267.1 MB |
| Object vector-outline buffers | 160 entries / 483.6 MB |
| Approximate cached outline records | 24.2 million |
| Resident lazy marker columns | 20 columns / 12.2 MB |

The image and object-ID tile caches were already warm and idle. The remaining fitted-view cost is
therefore primarily the vector outline pass: every ROI is visible, so spatial culling retains most
of the outline workload. The outline cache-key fix prevents duplicate uploads, but cached vector
buffers still have to be submitted and processed for every panning frame.

## Current and Target Rendering Paths

### Current

```text
Polygon fill geometry
    -> world-aligned R32UI object-ID tiles
    -> cell-ID-to-colour/state lookup in the compose shader
    -> continuous fills

Polygon outline geometry
    -> cached vector buffers and LOD bins
    -> vector line shader on every displayed frame
    -> cell outlines
```

The fill tiles are independent of marker, palette, domain, filter, and selection style. A tile
stores `object_index + 1`, with zero reserved for background. Small state and colour textures map
that object index to its current presentation.

### Target at low zoom

```text
World-aligned object-ID tiles
    -> cell-ID-to-colour/state lookup
    -> neighbouring-ID edge detection
    -> continuous fills + screen-space cell borders
```

### Target at inspection zoom

```text
Existing direct/vector fill and outline paths
    -> exact polygon edges and current interaction behaviour
```

The low-zoom path remains polygon based: the ID tiles are rasterizations of the true polygons, not
centroid/proxy markers.

## Visual Contract

The implementation must preserve these behaviours:

1. Continuous median/Nimbus colours remain identical to the existing mapping, including missing
   colours, opacity, clamping, reversal, and fixed/automatic domains.
2. Cells remain polygon-shaped at low zoom; `Always polygons` must never silently become proxy
   points.
3. Border width is expressed in screen pixels and remains visually stable while zooming.
4. Hidden/filtered cells remain hidden. A visible cell bordering a hidden cell receives an outer
   border.
5. Selected and primary cells preserve their current fill and border precedence.
6. Tile boundaries show no false seams and do not erase real cell boundaries.
7. Ancestor tiles provide a complete fallback while finer tiles are pending; stale zoom-level
   content must not remain after the finer level is ready.
8. The vector/texture transition has no flicker, repeated mode switching, double-dark borders, or
   obvious geometry pop.
9. Close zoom remains the current exact vector presentation.

Minor raster quantization is acceptable only while individual cell details are below meaningful
screen resolution. The renderer must switch back to vectors before that quantization becomes
visible during normal inspection.

## Design Decisions

1. **Use the existing ID tiles.** Do not introduce final-colour segmentation screenshots. Marker,
   palette, range, filter, and selection changes must continue to update without geometry
   rerasterization.
2. **Choose by projected workload and resolution, not ROI count.** One pathological ROI can be
   harder than ten small ROIs. The resolver should consider projected scale, tile coverage, and
   visible outline records.
3. **Do not make panning itself a cache key.** World-aligned tiles remain reusable across camera
   movement.
4. **Do not discard the vector renderer.** It remains authoritative at inspection zoom and as a
   fallback when texture composition is unavailable.
5. **Avoid a mandatory new user setting initially.** `Automatic` may continue to use proxy points;
   `Always polygons` can use polygon ID tiles at low zoom and exact vectors at high zoom. Add an
   advanced `Always vector outlines` override only if scientific/visual validation demonstrates a
   real need.
6. **Make the transition deterministic.** Use hysteresis around the switch threshold so tiny zoom
   changes cannot alternate modes frame by frame.
7. **Optimize the established bottleneck first.** Mosaic-wide draw batching is deferred until
   measurements show that the remaining tile-composition calls are material.

## Phase 0: Add Frame-Cost Diagnostics and Capture a Baseline

### Work

- Add per-frame outline statistics:
  - selected outline mode: `proxy`, `texture`, or `vector`;
  - visible vector bins;
  - visible vector records/vertices;
  - outline draw calls;
  - vector-outline paint/submit time;
  - texture-border compose draw calls and compose time.
- Separate retained cache bytes from work submitted in the latest frame.
- Add CPU render-plan timing around visibility-state preparation, fill planning, and outline
  planning.
- Aggregate the counters across mosaic layers in the existing renderer observation without
  scanning all geometry.
- Expose per-layer state through `viewer.objects.get_state()`, aggregate it through
  `mosaic.objects.get_state()`, and include the same bounded counters in application diagnostics.
- Record fitted-view idle and continuous-pan traces for one, three, and ten Amy ROIs.

GPU timer queries may be added if they can be collected asynchronously without stalling the GL
thread. CPU submission time and exact submitted record counts are the minimum required baseline.

### Acceptance

- The ten-ROI fitted view reports the approximate 24-million-record outline workload seen in the
  current cache snapshot.
- Panning does not increment ID-tile generation counters once the visible working set is warm.
- Diagnostic collection itself has no measurable interaction penalty.

## Phase 1: Cache Visibility and Selection Presentation State

The current render planning can build and scan per-object state arrays while preparing a frame,
even when only the camera changed.

### Work

- Add immutable cached `Arc<Vec<u8>>` payloads for:
  - base visible/filtered state used by direct and tiled fills;
  - continuous-outline visible/selected/primary state used by the vector fallback.
- Key/invalidate these payloads only on object resource, filter, selection, or relevant style
  generations—not camera or frame generation.
- Reuse the same state payload across the ID-tile and vector paths where their state encoding is
  compatible.
- Clear payloads on source replacement and ensure stale worker/model generations cannot reinstall
  them.
- Add retained-byte and rebuild-count diagnostics.

### Acceptance

- Panning with unchanged filter/selection performs zero full per-cell state rebuilds.
- Filter and selection changes rebuild exactly the required payloads and update the next presented
  frame without a residual previous layer.
- Existing single-view Analysis live-selection behaviour remains correct while also benefiting
  from the cached presentation state.

## Phase 2: Render Borders from Object-ID Tiles

### 2.1 Extend tile presentation state

Add low-zoom border parameters to `ObjectFillTileStyle`/`ObjectFillTileGlParams`:

- enabled/mode;
- outline width in screen pixels;
- base outline colour/opacity where continuous per-cell colour is not active;
- whether continuous object colour also colours the border;
- selected/primary border colours and precedence;
- physical-pixel scale for high-DPI displays.

These parameters are presentation-only and must not enter `ObjectFillTileKey`.

### 2.2 Add neighbour-ID edge detection

Extend the ID-tile compose fragment shader to:

1. fetch the current integer object ID;
2. resolve its visibility, selection, and continuous colour as today;
3. sample neighbouring ID texels using integer `texelFetch`;
4. classify the current fragment as a boundary when an effective neighbouring ID differs or is
   background;
5. blend the appropriate base/continuous/selected border colour;
6. retain the existing fill colour for interior fragments.

Use a bounded cardinal/diagonal sample kernel sized for the supported screen-space outline-width
range. Derive the texel offset from texture size, tile-to-screen scale, and pixels-per-point so the
border does not visibly grow or shrink at pyramid-level changes.

### 2.3 Eliminate tile seams with gutters

A shader cannot inspect an adjacent cell across two independently bound tile textures. Add a
small raster gutter instead of binding neighbouring textures:

- allocate each physical ID tile with at least a one-texel border around its logical 512 x 512
  interior;
- rasterize geometry for the correspondingly expanded world bounds;
- compose only the logical interior, while neighbour samples may read the gutter;
- extend the gutter if the supported outline kernel exceeds one texel;
- account the additional bytes precisely in the existing tile budget;
- make ancestor/fallback UV calculations address the correct logical interior.

A two-texel gutter around a 512-pixel tile adds less than two percent payload and is preferable to
extra texture bindings and cross-tile coordination.

### 2.4 Preserve selection and filtering semantics

- Treat a filtered/transparent neighbour as background for the visible current cell.
- Apply selected/primary fill and border presentation from the existing selection-state texture.
- Do not rebuild ID tiles when a filter, selection, marker, palette, opacity, or range changes.
- Verify objects spanning tile boundaries retain one coherent ID and do not acquire an internal
  seam.

### Acceptance

- A warmed low-zoom pan submits no base vector-outline geometry.
- Marker, palette, domain, filter, and selection edits change presentation without increasing ID
  tile-generation counts.
- No false grid appears at tile boundaries on solid-colour, categorical, or continuous fills.
- Border width is stable on standard and Retina/high-DPI displays.

## Phase 3: Add the Adaptive Texture/Vector Resolver

### Inputs

For each layer/frame, use already available or O(1)/bounded values:

- local screen pixels per source pixel;
- chosen polygon/outline LOD;
- estimated visible vector records from intersecting bins;
- target/fallback ID-tile coverage;
- whether fills and outlines are visible;
- GL integer-texture/compose-shader availability;
- current renderer preference (`Automatic` versus `Always polygons`).

### Policy

- Continue the existing proxy path when `Automatic` legitimately chooses proxy points.
- Under `Always polygons`, choose texture borders when:
  - polygon fills are active;
  - ID-tile composition is supported;
  - target or valid ancestor coverage can present the visible region; and
  - projected resolution or submitted vector work exceeds the calibrated threshold.
- Choose exact vectors before cells/edges become large enough for tile quantization to be
  inspectable.
- Keep vectors as a correctness fallback when tiles are unavailable.
- Use separate enter/leave thresholds. For example, enter vector mode at a higher projected scale
  than the scale at which texture mode is re-entered.
- Do not switch modes solely because the pointer started or stopped dragging.

The initial threshold should be calibrated from the one/three/ten-ROI matrix rather than frozen
from the Amy dataset. A defensive visible-vector-record budget may override the scale threshold
for unusually complex geometry.

### Transition

- Prefer hysteresis and stable coverage over a long crossfade.
- If a crossfade is required, keep it short and avoid fully drawing both expensive paths for more
  than a bounded number of frames.
- Advance presentation generation once per actual mode transition, not once per camera frame.
- Retain the most recent valid mode while the camera remains inside the hysteresis band.

### Acceptance

- Slowly zooming around the transition produces no flicker or rapid alternation.
- No blank/residual segmentation frame occurs when a finer tile level becomes available.
- Zooming in reaches exact vector outlines before boundary quantization is visually apparent.
- Zooming back out returns to the texture path without rebuilding geometry-keyed tiles.

## Phase 4: Reduce Remaining Tile-Composition Overhead if Measurements Require It

Only proceed if Phase 3 leaves a material CPU/draw-call bottleneck.

Potential work:

- collect visible tile draw descriptors across mosaic layers before registering paint callbacks;
- reuse one mapped/streamed quad buffer rather than uploading six vertices per tile draw;
- batch compatible tile quads while preserving per-ROI state/colour textures;
- investigate texture arrays or bindless-style grouping only where supported by Odon's GL target;
- retain per-ROI scissoring and stable composition order.

Do not introduce a texture atlas that forces ID-tile rebuilding on marker/style changes.

## Settings and API Semantics

The first implementation should not require a breaking SDK or project-format change.

- `Automatic` retains its current permission to use proxy points at low zoom.
- `Always polygons` guarantees real polygon-derived shapes, using texture borders at low zoom and
  exact vectors at inspection zoom.
- Existing `fast_rendering` API values remain valid.
- Add read-only diagnostics for the resolved low-zoom path and transition reason.
- If user testing identifies a need for exact vectors at every scale, replace the Boolean setting
  with a backward-compatible enum migration:
  - `automatic`;
  - `polygon_shapes` (texture low zoom, vector high zoom);
  - `vector_outlines_always`.

Any new setting must be actor-owned and projected to the renderer; the renderer must not become
the semantic source of truth.

## Automated Test Plan

### Unit tests

- Tile keys remain independent of camera and border/style state.
- Gutter world bounds and interior UVs are correct at several pyramid levels, negative tile
  coordinates, and ancestor fallback levels.
- The CPU mirror/test helper for edge classification handles:
  - same-ID interior;
  - different-ID boundaries;
  - object/background boundaries;
  - filtered neighbours;
  - selected and primary precedence.
- Adaptive mode resolution is deterministic at both hysteresis thresholds.
- A camera-only change does not rebuild visibility or selection state.
- Filter, selection, object-source, and property generations invalidate only their intended
  cached payloads.

### Renderer/control tests

- Shader creation failure falls back to vectors without hiding segmentation.
- ID-tile generation is unchanged by marker, palette, domain, opacity, filter, and selection
  updates.
- Mosaic renderer observations report the resolved mode and submitted work.
- Single-view and mosaic snapshots preserve existing API fields.
- Source-organization limits remain satisfied after splitting shader/planning/statistics files.

### Existing regression suites

- Object rendering and object-fill tile tests.
- Live selection and Analysis threshold tests.
- Mosaic model/control tests.
- Full `cargo check --all-targets`, formatting, and release app build.

## User-Led Visual and Performance Matrix

The user performs the visual interaction validation; automated checks provide counters and
readiness evidence but do not simulate GUI dragging.

Test each of the following with one large ROI, the three largest ROIs, and the ten-ROI Amy mosaic:

1. Fit the complete view and pan continuously in several directions.
2. Zoom slowly across the texture/vector transition in both directions.
3. Stop at the transition and make very small zoom changes.
4. Cycle median raw, median flat-field, and Nimbus repeatedly.
5. Cycle several image channels while panning.
6. Change palette, domain, opacity, and missing-value colour.
7. Toggle segmentation and channel visibility.
8. Apply and clear a filter.
9. Select cells and verify selected/primary fill and outline presentation.
10. Pan across ID-tile boundaries and inspect for seams or missing borders.
11. Zoom into individual cells and verify exact vector boundaries return.

Capture the internal counters before, during, and after each workload. Activity Monitor remains a
secondary process-level check, not the source of renderer attribution.

## Performance Acceptance Criteria

For the ten-ROI fitted view after the visible tile working set is warm:

- low-zoom panning submits zero base vector-outline records except for an explicitly documented
  correctness fallback;
- ID-tile generation remains flat while panning within the warmed visible region;
- no property decode or image-tile request is caused by camera movement alone;
- median and p95 pan frame time improve by at least 2x from the Phase 0 baseline, or reach a median
  below 16.7 ms and p95 below 33 ms on the benchmark machine;
- no repeated frame hitch above 50 ms occurs during a warmed pan;
- marker cycling and repeated panning reach a stable memory plateau;
- retained vector buffers do not grow with camera frames or marker changes;
- visual comparison shows no proxy dots, tile grid, residual prior zoom level, selection ghosting,
  or transition flicker.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| False seams at tile edges | Raster gutters and compose only the logical interior |
| Missing very narrow cells at coarse levels | Use multiresolution levels and switch to vectors before inspection scale |
| Border width changes at LOD transitions | Derivative/tile-to-screen-aware sampling and high-DPI tests |
| Transition flicker | Coverage requirement plus enter/leave hysteresis |
| Double-dark borders | One authoritative path outside a tightly bounded transition |
| Selected-cell presentation differs | Reuse the existing selection-state texture and explicit precedence tests |
| Shader unsupported/fails on a platform | Retain exact vector fallback and report the reason |
| Border shader becomes fill-rate heavy | Bound the sample kernel; compare shader cost with vector workload before defaulting |
| Extra gutter memory defeats tile budget | Exact byte accounting; expected overhead below two percent for a two-texel gutter |
| Per-frame CPU scans remain after GPU optimization | Complete Phase 1 and expose state rebuild counters |

## Delivery Sequence

1. Land diagnostics and record the one/three/ten-ROI baseline.
2. Cache visibility/selection presentation state and remeasure CPU frame planning.
3. Implement guttered ID tiles and texture-derived borders behind an internal feature flag.
4. Validate colour, filtering, selection, and tile-boundary correctness.
5. Add and calibrate the adaptive resolver and hysteresis.
6. Run automated suites and build the release app without launching it.
7. Launch the requested Amy workload for user-led visual/performance testing.
8. Make the path default only after the ten-ROI acceptance criteria pass.
9. Consider mosaic draw batching only if the post-change measurements justify it.

## Completion Definition

The work is complete when large segmentation layers in both single-image and mosaic viewers pan
smoothly using cached polygon-derived texture fills and borders, retain exact vectors at
inspection zoom, change continuous properties without rebuilding segmentation tiles, and expose
enough diagnostics to prove which path is active. The fitted ten-ROI Amy mosaic remains the main
stress gate. Automated regression coverage and the user-led visual matrix must pass without
flicker, seams, residual layers, or renewed memory growth.
