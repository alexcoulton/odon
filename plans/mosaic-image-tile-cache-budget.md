# Odon Mosaic Image-Tile Cache Budget Plan

## Status

Implemented through the byte-budgeted cache, channel-aware retention, bounded decoded-response
queue, automatic/preset/custom settings, actor API, Python SDK, and diagnostics. Automated
regression and the user-led ten-ROI visual validation remain the delivery gates recorded below.

## Objective

Replace mosaic image-channel caching's fixed 12,000-texture limit with a global, byte-budgeted,
channel-aware cache that remains fast on ordinary datasets and prevents large mosaics from
exhausting RAM, GPU-backed shared memory, and swap.

Automatic budgeting will be the default. Users will also have understandable presets and an
advanced byte-limit override. The setting will use MiB/GiB rather than tile count.

## Evidence and Current Failure Mode

The Amy ten-ROI comparison provides a reproducible stress case:

- 10 ROIs and 1,480,967 segmented cells.
- 86 unified markers.
- A unified marker can require several ROI-specific image-channel indices at once.
- Image chunks are normally 512 x 512, 16-bit, or approximately 0.5 MiB per full raw tile.
- `MosaicTilesGl` currently has a global 12,000-entry LRU whose key includes dataset, level,
  position, and channel.
- Completed tiles from previous channels remain resident until entry-count pressure evicts them.
- A full cache can therefore represent approximately 6 GiB of raw tile payload before graphics
  driver overhead.

Live measurements after channel cycling showed:

| Component | Clean ten-ROI launch | After channel cycling |
| --- | ---: | ---: |
| Odon physical footprint | 6.16 GB | 10.9 GB |
| Untagged/shared VM allocation | 612 MB | 4.58 GB |
| IOAccelerator graphics | 98 MB | 529 MB |
| IOSurface | 37 MB | 154 MB |
| Accounted segmentation/object data | 3.191 GB | 3.191 GB |

The object-property cache remained configured as a two-column-per-ROI LRU, and no per-cell JSON
maps were retained. The growth is therefore in the image-tile/graphics path rather than Nimbus or
median-value property storage.

The measured machine had 17.4 GB of 18.4 GB swap in use during the stressed run. Avoid using that
run as an interactive profiling target until the bounded cache is available.

## Design Decisions

1. A byte budget is always active; there is no separate "massive workload mode."
2. Small workloads behave as before because they never reach the budget.
3. The budget is global across the mosaic, not multiplied per ROI or channel.
4. Current visible tiles and a coarse fallback working set have priority over history.
5. A channel history policy retains either no previous marker group, one previous group, or an
   automatically selected amount.
6. A marker group means the complete set of ROI-specific channel indices needed to display one
   unified marker, not merely the active channel index.
7. Automatic policy responds to system pressure but uses hysteresis to avoid oscillation.
8. GPU deletion is performed on the GL thread and remains accounted until deletion actually runs.
9. Live-byte counters are authoritative for cache correctness; Activity Monitor remains useful for
   physical footprint and allocator/driver high-water behaviour.
10. Settings and control methods are actor-owned; the renderer consumes a projected policy and
    reports observations without becoming the semantic owner.

## Proposed User Settings

Add an **Image tile cache** section to Odon's performance/rendering settings:

### Cache mode

- `Automatic` — default and recommended.
- `Conservative` — fixed low-memory preset.
- `Balanced` — fixed middle preset.
- `Performance` — larger cache for machines with substantial memory.
- `Custom` — advanced explicit limit in MiB or GiB.

Initial preset values should be calibrated against the benchmark matrix. Starting candidates are:

| Preset | Candidate budget |
| --- | ---: |
| Conservative | 256 MiB |
| Balanced | 512 MiB |
| Performance | 1 GiB |
| Custom | 128 MiB–4 GiB |

### Channel history

- `Automatic` — retain the previous marker group only while pressure and budget allow.
- `Current only` — release all non-visible channel tiles.
- `Current + previous` — favor rapid before/after comparison.

Do not expose the existing 12,000-entry limit in the UI.

## Runtime Policy Model

Introduce serializable actor-owned policy types along these lines:

```text
ImageTileCachePolicy
├── mode: automatic | conservative | balanced | performance | custom
├── custom_budget_bytes: optional integer
├── channel_history: automatic | current_only | current_and_previous
└── low_watermark_ratio: internal/defaulted

ResolvedImageTileCachePolicy
├── effective_budget_bytes
├── low_watermark_bytes
├── pressure_state
├── visible_channel_group
├── retained_history_groups
└── reason
```

Persist the user policy in `AppSettings`. Do not persist transient resolved budgets, pressure
states, cache contents, or current marker history.

## Automatic Budget Resolution

Automatic mode should derive a conservative upper bound from:

- total physical memory;
- currently available memory;
- current macOS/Windows/Linux memory-pressure state;
- known Odon CPU allocations, especially segmentation/object data;
- the estimated current visible tile working set;
- active viewport count and mosaic visibility;
- existing pinned image levels.

The initial implementation should use a deterministic total-RAM tier as its stable ceiling, then
shrink below that ceiling in response to pressure. Suggested initial ceilings:

| Physical RAM | Automatic ceiling candidate |
| --- | ---: |
| Up to 16 GiB | 384 MiB |
| 16–32 GiB | 768 MiB |
| Above 32 GiB | 1 GiB |

These are calibration inputs, not contractual constants.

Pressure response:

- `normal`: use the resolved ceiling;
- `warning`: target at most 50% of the ceiling and discard history;
- `critical`: target at most 25% and retain only the visible/coarse working set.

Use high/low watermarks. For example, begin eviction when tracked bytes exceed 100% of the
effective budget and continue until no more than 80–85% remains. Require sustained lower pressure
before growing the budget again.

If the current visible working set exceeds the configured budget, correctness wins: retain the
minimum complete visible set, report an explicit temporary over-budget reason, stop historical
prefetch, and aggressively remove everything unprotected.

## Cache Refactor

### 1. Account actual bytes

Extend `MosaicTilesGl` so every entry records its retained cost:

- pending CPU raw bytes: `width * height * size_of::<u16>()`;
- uploaded texture payload estimate using the actual edge-tile dimensions and GL format;
- texture/resource bookkeeping overhead as a separately labelled estimate;
- bytes waiting for GL deletion;
- current and peak bytes.

Track bytes during every state transition:

```text
request -> pending CPU -> uploaded GPU -> queued for deletion -> deleted
```

Never decrement the GPU/deletion total when an entry merely leaves the LRU. Decrement only after
the GL-thread deletion executes.

### 2. Replace entry-count eviction

Retain LRU ordering for recency, but evict by total bytes rather than number of entries. The cache
must support:

- variable-size edge tiles;
- mixed pyramid levels and formats;
- an entry larger than the historical allowance;
- protected visible keys;
- eviction down to a low watermark;
- bounded queued-deletion bytes;
- a small defensive entry-count ceiling for metadata/pathological zero-byte cases.

### 3. Add channel-aware priorities

Classify cache entries in this order:

1. visible tiles required for the current complete composite;
2. coarse visible fallback tiles;
3. current marker-group tiles outside the immediate view;
4. previous marker-group tiles, when allowed;
5. recent camera history for the current group;
6. older marker groups and distant pyramid levels.

Evict from the lowest priority first, using LRU within each priority.

When the visible channel set changes:

- advance the existing tile-request generation;
- treat the entire new visible channel set as one marker group;
- cancel or reject stale pending work;
- immediately remove cache entries outside the permitted history groups;
- queue their GL textures for deletion on the next paint;
- repaint after deletion so the queue cannot remain indefinitely while the viewer is idle.

### 4. Bound transient work

The request queue is already bounded and workers perform pre/post-decode generation checks. Add:

- byte reservation before accepting a decode when possible;
- a bounded response queue or explicit pending-response byte accounting;
- stale-drop counters before read, after read, before install, and before upload;
- cancellation of obsolete pending CPU payloads when the visible channel group changes;
- a per-frame upload/deletion work budget so cleanup does not create a long UI-thread stall.

## Settings and Control Surface

### Native application

- Add the policy to `AppSettings` with backward-compatible defaults.
- Normalize invalid or old settings to `Automatic`.
- Add native Settings controls with explanatory memory/performance text.
- Show the currently resolved byte budget and pressure state beneath `Automatic`.
- Keep the existing mosaic Memory panel focused on operational status; link or mirror the cache
  summary there.

### Actor/API

Extend the existing `memory.tiles.get` and `memory.tiles.set` contract rather than creating a
parallel cache-policy API. Add typed fields such as:

```json
{
  "cache_mode": "automatic",
  "cache_budget_bytes": null,
  "channel_history": "automatic"
}
```

The returned snapshot should distinguish requested policy from resolved runtime policy.

Update:

- control registry schemas and method catalogue;
- application-surface manifest;
- optimistic revision handling;
- sync and async Python SDK wrappers;
- generated Python API reference;
- settings persistence and migration tests;
- MCP exposure only if interactive cache inspection materially helps debugging.

## Diagnostics

Make the following available in `memory.tiles.get`, application diagnostics, and the mosaic Memory
panel:

- requested mode and custom budget;
- resolved budget and low watermark;
- current pressure state and resolution reason;
- cache entry count;
- pending CPU bytes;
- uploaded texture bytes;
- queued-deletion bytes;
- total tracked bytes and peak bytes;
- protected visible working-set bytes;
- over-budget bytes and reason;
- hits, misses, inserts, and evictions;
- evictions by cause: byte budget, channel change, pressure, dataset replacement;
- stale drops at each pipeline stage;
- current and previous channel groups;
- bounded per-channel/group entry and byte summaries;
- GL deletion count and last successful cleanup frame;
- currently in-flight request/response bytes.

Counters must be O(1) to update. Diagnostic reads must not scan 12,000 entries every frame; bounded
per-group summaries should be maintained incrementally or computed only on explicit request.

## Implementation Phases

### Phase 0: Instrument without changing eviction

1. Add byte accounting and lifecycle counters to `MosaicTilesGl`.
2. Expose the counters through the renderer observation and Memory panel.
3. Record a clean ten-ROI baseline and a marker-cycling trace.
4. Reconcile internal totals with `footprint`, `vmmap -summary`, and Activity Monitor categories.

Acceptance:

- Internal pending/uploaded/deletion totals explain the direction and approximate magnitude of the
  process-level VM/graphics change.
- Counter collection does not scan cache contents per frame.

### Phase 1: Byte-budgeted global LRU

1. Replace the 12,000-entry policy with budget/high/low watermarks.
2. Preserve visible tile keys during eviction.
3. Delete evicted GL resources promptly and account deletion completion.
4. Add unit tests for variable tile sizes and every lifecycle transition.

Acceptance:

- Tracked cache bytes plateau at the configured budget, except for a reported visible-working-set
  overrun.
- No stale texture remains visible.
- No cache-induced flicker appears during ordinary pan or zoom.

### Phase 2: Channel-aware retention

1. Represent the current unified marker as a set of ROI-specific channel indices.
2. Implement `Current only`, `Current + previous`, and automatic history.
3. Evict older channel groups immediately on visibility replacement.
4. Coalesce rapid marker changes and drop stale work.

Acceptance:

- Repeated channel switching reaches a stable live-byte plateau.
- Returning to the immediately previous marker remains fast when history permits.
- Older markers reload without retaining their old textures afterward.

### Phase 3: Automatic budgeting and settings

1. Add persistent settings types, migration, presets, and native UI.
2. Resolve the automatic budget from RAM tier, known Odon allocations, and pressure.
3. Apply hysteretic pressure shrink/grow behaviour.
4. Project resolved policy to every mosaic renderer generation.

Acceptance:

- Small datasets show no measurable regression.
- Large mosaics constrain history automatically before swap pressure becomes severe.
- Changing the setting applies without reopening the mosaic.

### Phase 4: SDK, documentation, and compatibility

1. Extend `memory.tiles.get/set` and the Python SDK.
2. Update schemas, manifests, API documentation, and workflow documentation.
3. Add sync/async SDK tests and actor ownership/revision tests.
4. Document that the property-column LRU and image-tile cache are separate controls.

## Test Matrix

### Unit tests

- Byte accounting for full and edge tiles.
- Pending-to-uploaded transition does not double-count CPU and GPU payloads.
- Eviction remains accounted until GL deletion completes.
- LRU eviction reaches the low watermark.
- Visible keys survive normal budget eviction.
- An oversized visible working set reports rather than loops.
- Channel-group replacement retains the configured history only.
- Several ROI-specific aliases are treated as one unified marker group.
- Rapid generation changes drop stale requests and responses.
- Dataset replacement clears all cache state and deletion queues safely.
- Settings migration from configurations without the new fields.

### Existing regression suites

- Full Rust library and application tests.
- Full Python SDK tests.
- Existing tile worker, mosaic rendering, object-fill, and source-organization tests.
- Single-view rendering tests even if the first implementation changes mosaic only.

### Performance datasets

1. Synthetic small mosaic: prove the budget is dormant below its limit.
2. One large Amy ROI.
3. Three large Amy ROIs.
4. The ten largest Amy ROIs.
5. All Amy ROIs only after the ten-ROI criteria pass.

## Ten-ROI Validation Protocol

Use the same cached Amy environment and settings as the current comparison workflow:

```text
10 ROIs
1,480,967 cells
Always polygons
object property cache capacity = 2 per ROI
one unified marker group visible at a time
```

For every build:

1. Launch clean and wait for all object resources and presentation readiness.
2. Record internal cache counters, `footprint`, `vmmap -summary`, system swap, and frame timing.
3. Cycle through at least 20 unified markers at a fixed camera.
4. Repeat the same cycle once more.
5. Pan and zoom at overview and focused scales.
6. Alternate repeatedly between two markers to test previous-group retention.
7. Change median raw, median flat-field, and Nimbus fills independently of the image channel.
8. Record cache hits/misses, evictions, stale drops, live/peak bytes, physical footprint, and swap
   before and after each stage.
9. Let the viewer idle and verify pending responses and queued GL deletions drain to zero.
10. The user performs visual checks; automated tooling records semantic and memory state only.

## Acceptance Criteria

- Image-tile live bytes remain within the resolved budget plus an explicitly reported visible-set
  overrun.
- After warm-up, a second complete marker cycle increases internal live cache bytes by no more than
  1% and process physical footprint by no more than 5%, allowing allocator/driver high-water to be
  reported separately.
- The ten-ROI workflow does not materially increase system swap from its clean-launch baseline.
- Automatic mode prevents the observed growth from approximately 6.2 GB to 10.9 GB during the
  marker-cycling test on the reference machine.
- Current-marker panning and zooming remain smooth.
- The previous marker switches quickly when history is enabled.
- No partial-channel composite, stale channel, blank ROI, or fill flicker appears.
- Rapid marker switching cannot leave unbounded pending CPU payloads or queued GL deletions.
- Small single-image and small-mosaic workloads show no statistically meaningful frame-time or
  channel-switch regression.
- Settings, actor state, Python SDK state, native UI, and renderer-resolved state remain
  synchronized.

## Likely Code Areas

- `src/mosaic/tiles_gl.rs`: byte accounting, priority eviction, deletion accounting, statistics.
- `src/mosaic/io.rs`: pending/response byte bounds and stale-work counters.
- `src/mosaic/canvas.rs`: visible/protected key sets and channel-group transitions.
- `src/mosaic/construction/assembly.rs`: remove `MosaicTilesGl::new(12_000)` and inject resolved
  policy.
- `src/mosaic/control/snapshots.rs`: renderer cache observations.
- `src/mosaic/panels.rs` and settings UI: diagnostics and controls.
- `src/settings.rs`: persistent cache policy and migration/defaults.
- `src/model/app/preferences_memory.rs`: actor-owned requested/resolved policy.
- `src/control/registry/*`: typed method schemas and capability catalogue.
- `python/src/odon/resources.py` and `async_resources.py`: sync/async SDK controls.
- `api/application-surface.json` and generated API documentation.

## Risks and Mitigations

### Visible-set eviction causes flicker

Protect all keys used by the current complete composite and its coarse fallback. Permit a bounded,
reported overrun instead of evicting visible dependencies.

### Cleanup stalls the UI thread

Limit GL uploads and deletions per frame while maintaining prompt repaint until the deletion queue
is empty.

### Aggressive eviction makes comparisons slow

Retain one previous unified marker group when budget and pressure allow. Report history-disabled
decisions in diagnostics.

### Activity Monitor stays high after correct eviction

Use internal live-byte counters to prove release, then separately report allocator and graphics
driver high-water behaviour. Do not weaken the cache merely to force an immediate process-footprint
drop.

### Automatic policy oscillates

Use pressure-state hysteresis, separate high/low watermarks, and a minimum interval before budget
growth.

### Settings multiply memory across modes

Treat the configured value as an application-wide ceiling. The first implementation can apply it
to mosaic image tiles while retaining a path to a shared single-view/mosaic cache coordinator.

## Non-goals

- Replacing the segmentation object-tile renderer.
- Combining the object-property LRU with the image-tile cache.
- Keeping every historical channel instantly available on a memory-constrained machine.
- Using Activity Monitor alone as the cache-correctness test.
- Enabling live whole-mosaic Analysis as part of this work.

## Delivery Order

1. Merge diagnostics and reproduce the live growth with exact cache counters.
2. Merge byte-budgeted eviction behind the automatic policy default.
3. Add channel-group history and stale-work cleanup.
4. Add settings, presets, actor/API support, and documentation.
5. Pass automated tests and the one-/three-/ten-ROI benchmark ladder.
6. Enable the default for all mosaics.
7. Consider sharing the same global coordinator with single-view and multi-viewport image caches.
