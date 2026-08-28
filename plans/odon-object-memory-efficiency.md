# Odon Object and Mosaic Memory-Efficiency Plan

## Objective

Reduce the memory required to load and interact with very large segmentation layers, especially
multi-ROI mosaics, without regressing the tiled segmentation renderer's smooth zooming, panning,
continuous fills, selection updates, or Python SDK behavior.

The immediate benchmark is the Amy ten-ROI mosaic:

- 10 large ROIs
- 1,480,967 cells
- 86 unified markers
- Median raw, median flat-field, and Nimbus fills
- Property-column cache capacity of two

## Implementation Status (2026-08-27)

Phases 1 and 2 are implemented and covered by the Rust and Python regression suites.

### Phase 1 delivered

- GeoParquet rows no longer build or retain per-cell JSON property maps.
- Identity and selected properties are exposed through a shared typed column source.
- Numeric and categorical lookup, filtering, Analysis values, exports, measurements, and SDK
  queries read through that source.
- Generated measurement columns use compact `f32` values plus a validity bitmap and layer over
  the shared source without cloning the feature table.
- GeoJSON and CSV retain their existing row-oriented fallback behavior.

### Phase 2 delivered

- The renderer and control actor now hold the same `Arc<Vec<ControlObjectFeature>>`; building the
  control resource does not clone IDs, bounds, polygons, centroids, or inline properties.
- The actor retains immutable shared bulk data while continuing to own mutations and generation
  synchronization.
- Memory diagnostics count the shared feature/geometry representation once and no longer publish
  an `actor_model.*` geometry mirror.

### Automated verification

- Rust library: 259 passed.
- Rust application binary: 270 passed, with six fixture/diagnostic tests intentionally ignored.
- Python SDK: 197 passed, one external-fixture test skipped.
- Full `cargo check --all-targets`, formatting, and diff whitespace checks passed.

### Ten-ROI object-retention benchmark

The checked-out ten Amy GeoParquet tables were loaded simultaneously through the same full-geometry,
lazy-property preload path used by mosaic resources. This exercises all 1,480,967 cells without
requiring the image volume:

| Measurement | Phase 1 + 2 result |
| --- | ---: |
| ROIs held concurrently | 10 |
| Cells | 1,480,967 |
| Accounted retained CPU object/render data | 3.191 GB |
| Retained opaque per-row property maps | 0 |
| Test-process peak memory footprint | 3.188 GB |
| Test-process maximum resident set size | 1.071 GB |
| Swap operations | 0 |
| Full load time, already-built debug test binary | 44.34 s |

The 3.191 GB component total is 428 MB below the previous 3.619 GB accounted geometry total,
consistent with removing the control geometry mirror. More importantly, the old approximately
2.18 GB of opaque duplicated B-tree map allocations is absent. The process-level result is not a
direct comparison with the earlier 11.5 GB GUI footprint because this diagnostic does not load
image channels, create a macOS window, or install GPU tiles. A like-for-like GUI measurement
remains to be taken when the external Amy image volume is mounted; visual interaction testing also
remains user-operated.

## Measured Baseline

The existing byte accounting and macOS heap measurements show:

| Component | Retained memory |
| --- | ---: |
| Accounted CPU geometry and derived render data | 3.619 GB |
| Total live malloc heap | 9.762 GB |
| Physical footprint during the measured run | 11.5 GB |
| GPU accelerator resident memory | Approximately 289 MB |
| IOSurface memory | Approximately 154 MB |

The geometry total includes canonical objects, the SDK/control mirror, outline LODs, selection
LODs, fill meshes, fill spatial bins, spatial indexes, and point payloads. It excludes allocator
rounding, JSON B-tree nodes, columnar properties, analysis caches, image data, and GPU resources.

### Per-cell property-map finding

Each normalized Amy GeoParquet row supplies three identifier properties:

- `label`
- `name`
- `cell_id`

The loader also inserts normalized `id`. These four values are retained in a row-oriented
`serde_json::Map`, which is backed by a Rust `BTreeMap` in the current build.

The renderer model retains one map per cell. Building the Python SDK/control resource clones the
map, resulting in:

```text
1,480,967 cells × 2 maps = 2,961,934 retained maps
```

An isolated allocator probe using the same four-entry map produced exactly one 736-byte live
allocation per map on the current macOS/Rust build. The map nodes therefore account for:

```text
2,961,934 × 736 bytes = 2,179,983,424 bytes
                           2.180 GB
                           2.030 GiB
```

This does not include separately allocated property keys, string values, or allocator metadata.
The application-level diagnostics report these maps as opaque allocations rather than treating
the platform-specific 736-byte allocator size class as a portable exact value.

### Current conclusion

The ten-ROI memory footprint is largely explainable by retained duplicate representations. There
is not currently evidence that the stable footprint is primarily caused by the compact `f32`
measurement columns or by GPU textures. There is also not yet evidence of an unbounded leak after
the configured channel/property caches have warmed; that must remain a specific regression test.

The largest known opportunities are:

1. Per-cell JSON property maps retained twice.
2. A second SDK/control representation of cell geometry and metadata.
3. Separate derived geometry for normal outlines, selected outlines, full fills, and spatially
   binned fills.

## Design Principles

1. Keep large scientific data columnar and shared.
2. Materialize row-oriented JSON only for the small number of cells explicitly requested.
3. Maintain one canonical identity for each cell.
4. Share immutable data between the renderer and SDK/control actor.
5. Keep renderer caches bounded and independently evictable.
6. Preserve the fast tiled renderer and avoid exchanging memory savings for visible flicker or
   interaction latency.
7. Measure every phase independently before proceeding to the next architectural change.
8. Preserve GeoJSON and CSV behavior while optimizing GeoParquet first.

## Phase 0: Finish and Preserve Memory Instrumentation

### Work

- Keep per-ROI and aggregate object-memory diagnostics available from single-view and mosaic
  object state.
- Report exact retained capacities for vectors, geometry, IDs, and JSON strings.
- Report opaque allocation counts for containers such as `serde_json::Map` whose internal byte
  capacity is not exposed by their public API.
- Add byte reporting for resident property columns, analysis caches, CPU render tiles, GPU object
  tiles, and image channels.
- Report cache entries, evictions, retained references, and current versus peak bytes.
- Retain the compatibility API field while documenting that the report now covers broader CPU
  object data, not geometry alone.

### Acceptance criteria

- Per-ROI component totals sum to the mosaic aggregate.
- Exact capacity totals do not include opaque estimates.
- Allocator estimates are clearly identified with their platform and measurement method.
- Diagnostic collection does not create per-frame scans or materially affect interaction speed.

## Phase 1: Replace Per-cell GeoParquet Maps with Shared Columns

This is the first implementation phase because it offers the largest certain saving without
changing the successful renderer.

### Proposed model

Introduce a shared object-attribute table along these lines:

```text
ObjectAttributeStore
├── canonical identity column
├── optional label/name/source identity columns
├── compact numeric columns (f32 + validity bitmap where appropriate)
├── integer and Boolean columns
├── dictionary-encoded categorical strings
└── fallback JSON columns only for unsupported values
```

Both the renderer and SDK/control actor reference the same immutable column storage.

### Work

1. Select a canonical identity using the existing precedence rules.
2. Keep the canonical identity and source-row index outside a JSON map.
3. Load `label`, `name`, and `cell_id` as compact shared columns only when their semantics are
   needed.
4. Stop inserting normalized `id` into every GeoParquet row map.
5. Keep GeoParquet `inline_properties` empty or remove that field from the GeoParquet-backed
   representation.
6. Update object-property lookup, filtering, export, measurement, selection details, and coloring
   to read through the shared attribute store.
7. Materialize a JSON object only when returning one cell's details or serializing a bounded SDK
   result.
8. Preserve row-oriented inline properties as a fallback for GeoJSON and CSV during this phase.
9. Verify that dropped property-cache columns have no remaining strong references.

### Compatibility requirements

- Existing Python SDK property names and returned values remain stable.
- Filter and analysis expressions produce identical selections.
- Cell details still expose identity fields.
- GeoParquet export preserves requested attributes.
- Missing values and `f32` conversion behavior remain unchanged.

### Expected result

- Guaranteed removal of approximately 2.18 GB of B-tree node allocations in the ten-ROI
  benchmark, plus duplicated key and string allocations.
- Fewer allocations, faster loading, and improved CPU cache locality.

## Phase 2: Remove the SDK/Control Geometry Mirror

The control resource currently builds a second feature representation containing cloned IDs,
properties, polygon coordinates, bounds, centroids, areas, and perimeters.

### Target architecture

```text
                         ┌── Renderer
Shared ObjectData ───────┤
                         └── SDK/control actor
```

### Work

1. Introduce a renderer-neutral immutable `ObjectData` or `ObjectFeatureTable`.
2. Store bounds, centroids, areas, perimeters, source-row indices, and identities in compact shared
   arrays.
3. Store polygon geometry once in a shared representation usable by the renderer.
4. Replace `ControlObjectFeature` ownership with lightweight row references or accessors.
5. Return polygon coordinates through the SDK only when explicitly requested.
6. Ensure selection, zoom-to-object, filtering, numeric summaries, export, and SDK queries operate
   against shared storage.
7. Keep actor ownership of mutations and synchronization; only immutable bulk storage is shared.

### Expected result

- Remove approximately 400–600 MB of duplicated feature and polygon storage in the current
  benchmark.
- Reduce load-time cloning and temporary allocation pressure.

## Phase 3: Consolidate Derived Geometry Caches

Measured derived caches in the ten-ROI run include:

| Cache | Retained memory |
| --- | ---: |
| Selection outline LODs | 798 MB |
| Fill spatial bins | 624 MB |
| Full fill mesh | 591 MB |
| Normal outline LODs | 594 MB |

These caches are performance-sensitive and should be optimized only after Phases 1 and 2 have
been measured.

### Work

1. Share outline segment vertices between normal and selected rendering.
2. Attach compact object IDs or object-index ranges to shared segments rather than retaining a
   second selection geometry.
3. Investigate whether spatial fill bins can reference indexed ranges in a canonical mesh rather
   than copying full vertex payloads.
4. Split CPU render resources into independently evictable tiles.
5. Place CPU object tiles under a global mosaic byte budget, analogous to the GPU object-tile
   budget.
6. Prioritize visible tiles and recently used zoom levels.
7. Drop stale tile-build results before installation when the camera, source, style, or selection
   generation has advanced.
8. Rebuild evicted derived tiles from canonical geometry on demand.
9. Preserve a bounded low-resolution fallback so zooming out never requires every detailed tile
   to be resident simultaneously.

### Required performance checks

- No return of fill flicker during pan or zoom.
- Histogram/live-selection changes update without a residual previous selection layer.
- No stale zoom-level texture remains visible after selection or style changes.
- Tile misses do not cause long UI-thread stalls.
- Rapid camera motion coalesces obsolete work.

## Phase 4: Make Channel and Property Eviction Observable

### Work

- Report resident image-channel CPU and GPU bytes.
- Report resident marker-property columns and their byte sizes.
- Record LRU hits, misses, insertions, and evictions.
- Detect strong references that prevent an evicted entry from being released.
- Ensure switching channels does not retain previous decoded arrays or texture uploads beyond the
  configured capacity.
- Ensure switching fill properties releases old columns after both renderer and actor generations
  have advanced.

### Acceptance criteria

- Repeatedly cycling through all channels and fill modes reaches a stable memory plateau.
- After warm-up, memory does not grow by more than 5% over another complete cycle, allowing for
  allocator high-water behavior to be reported separately from live allocations.
- Returning to a recently used item within capacity remains fast.

## Phase 5: Benchmark and Regression Matrix

Run each phase independently against the same datasets and settings.

### Benchmarks

1. One large Amy ROI.
2. Three large Amy ROIs.
3. The ten largest Amy ROIs.
4. The complete Amy mosaic after the ten-ROI target is stable.

### Record

- Live malloc bytes and allocation count.
- Physical footprint and peak footprint.
- Swap use.
- Per-component retained CPU bytes.
- CPU and GPU tile-cache bytes.
- Image-channel bytes.
- Property-column bytes.
- Initial load time.
- Channel and fill-switch latency.
- Pan and zoom frame time.
- Tile-build queue depth and stale-work drops.
- Selection-update latency.

### Interaction tests

- Pan and zoom continuously with Always polygons enabled.
- Cycle through every channel repeatedly.
- Cycle through median raw, median flat-field, and Nimbus fills repeatedly.
- Change continuous color limits and palettes.
- Toggle segmentation and channel visibility.
- Use histogram live selection in single-view Analysis mode.
- Move the histogram threshold continuously and verify there is no residual selection layer.
- Enter and leave the Analysis tab repeatedly.
- Query cell details and properties through the Python SDK.
- Export a filtered selection and compare values with the source GeoParquet.

## Delivery Sequence

1. Complete diagnostic coverage and capture the reproducible ten-ROI baseline.
2. Implement the shared columnar GeoParquet property store.
3. Validate SDK, filtering, export, coloring, and Analysis behavior.
4. Measure the ten-ROI memory reduction before continuing.
5. Replace the SDK/control geometry mirror with shared immutable object data.
6. Measure again.
7. Prototype derived-cache consolidation behind a feature flag.
8. Compare memory and interaction performance before making it the default.
9. Extend testing to the complete Amy mosaic only after the ten-ROI acceptance criteria pass.

## Success Criteria

The work is successful when:

- The ten-ROI mosaic loads without severe memory pressure or swap-driven system degradation.
- Its live heap is substantially below the current 9.762 GB baseline.
- Property and channel cycling reaches a stable plateau.
- The Python SDK retains its current capabilities without owning a second row-oriented copy of
  every cell.
- Zooming, panning, continuous fills, and selection remain visually stable and responsive.
- The complete mosaic either fits within a documented budget or degrades predictably through
  bounded eviction rather than exhausting system memory.

## Non-goals

- Removing numerical precision beyond the already accepted `f32` measurement representation.
- Replacing the tiled segmentation renderer that solved the original zoom/fill performance issue.
- Performing live Analysis across the entire mosaic before a concrete mosaic-analysis use case is
  defined.
- Treating allocator high-water marks as live retained memory without corroborating heap data.
