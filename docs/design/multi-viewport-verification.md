# Multi-Viewport Milestone Verification

Status: verified complete for the one/two same-document viewport milestone,
2026-08-21

This is the requirement-by-requirement completion record for
`multi-viewport-plan.md`. It distinguishes the shipped first milestone from
explicit later work such as nested grids, unrelated datasets, detached windows,
and independent selection sets.

## Product and ownership requirements

| Requirement | Implementation evidence | Verification evidence |
| --- | --- | --- |
| One or two viewports, never zero or more than two | `ViewportWorkspace` and `MAX_VIEWPORTS` in `src/viewports.rs` | `workspace_starts_with_one_stable_active_viewport`; `workspace_enforces_limit_and_preserves_last_viewport` |
| Stable IDs, titles, active state, clone, rename, swap and removal | `ViewportId`, `ViewportSlot`, workspace lifecycle methods; canvas header and Views menu in `src/app.rs` | `cloning_creates_independent_state_and_split_layout`; `setting_active_layout_links_and_title_updates_revision_only_on_change`; `viewport_lifecycle_rejects_invalid_layouts_and_preserves_final_view` |
| Horizontal and vertical splits with usable ratios | `ViewportLayout`, validated `split_ratio`, proportional split allocation | ratio state tests; project round-trip test; `horizontal_split_stacks_each_header_above_an_adjacent_full_height_canvas`; live 55/45 GPU capture |
| One scientific document and source owner | one `OmeZarrViewerApp` retains dataset, store, loaders, scientific layers and history; `ViewerViewportState` contains only navigation/presentation state | ownership inventory; `motivating_two_property_comparison_runs_end_to_end_on_one_document` asserts one document, dataset and object geometry instance |
| Shared object identity, selection, masks, annotations and edits | these remain outside `ViewerViewportState`; selection is an invariant of the fixed link group | ownership inventory; motivating acceptance test selects one object once; `two_viewport_controls_keep_presentation_independent_and_navigation_linked` |
| Independent navigation and presentation | per-slot camera, plane, channels, native-layer state, object style/filter/legend and render preferences | `two_viewport_controls_keep_presentation_independent_and_navigation_linked`; channel-group, overlay, rendering, explicit-layer and revision tests |
| Only the active viewport edits | non-active canvas temporarily uses pan mode; activation is click/drag based; transient edit state is cancelled on activation/removal | `activating_or_removing_a_viewport_cancels_transient_edit_gestures`; canvas geometry/interaction characterization tests |

The detailed field classification and prohibited ownership regressions are
recorded in `multi-viewport-ownership-inventory.md`.

## Rendering and scheduling requirements

| Requirement | Implementation evidence | Verification evidence |
| --- | --- | --- |
| Union and deduplicate all visible raw keys | workspace frame builds aggregate raw, CPU, label and SpatialData-image key sets before publishing them | `multi_viewport_active_key_union_deduplicates_and_retains_peer_work` |
| Fair request budgets | active and peer canvases receive non-zero bounded budgets, with an active-view preference | `multi_viewport_scheduler_prioritizes_active_without_starving_peer` |
| Two simultaneous CPU presentations | active render IDs are a set; composed responses retain their render generation | `tile_worker_accepts_two_live_viewport_generations`; TIFF equivalent test |
| Presentation-independent decode reuse | `CpuDecodedTileCache` keys source samples without colour/contrast or viewport ID | Zarr and TIFF sharing tests assert two distinct RGBA outputs, one source read and one cache hit |
| Slow/failing shared work is safe | decoded-cache entries coordinate concurrent work and retain failures | `decoded_cache_shares_slow_work_and_reuses_failures` |
| Removing one viewport drops only its work | workers recheck active keys/generations before publishing; removal updates live render IDs | `removing_viewport_drops_only_its_cpu_generation_during_loading` |
| Independent coarse/fine continuity | target/fallback history and current/previous render IDs are in `ViewerViewportState` | ownership inventory; simultaneous generation and presentation tests |
| Shared GPU raw cache with independent composition | one `TilesGl`; each callback supplies its viewport channel presentation and sampling mode | live GPU capture reports one cache and shows different composites of the same source |
| No GPU callback state leakage | each compositor captures and restores viewport, scissor, blend function, blend/depth/cull enablement | `GlCapabilityState` capture/restore in all `TilesGl` compositor callbacks; native GPU smoke |
| Object data stored once while styles coexist | object geometry/properties stay in `ObjectsLayer`; presentation signatures and colour-group cache coexist | motivating two-property test; runtime geometry-instance counter; object presentation tests |

## Linking, interaction and events

- Camera and plane propagation happens inside one `control_in_viewport`
  transaction. It returns the source ID, affected IDs, navigation and
  presentation revisions, and one link transaction ID.
- `explicit_camera_fit_uses_target_canvas_and_propagates_one_link_transaction`
  verifies target-canvas fitting and one causal propagation.
- Disabling camera propagation leaves the peer unchanged in
  `two_viewport_controls_keep_presentation_independent_and_navigation_linked`.
- Selection remains document-shared and cannot be disabled in this milestone.
  `canonical_viewport_link_group_validates_members_and_preserves_shared_selection`
  verifies that invariant.
- Explicit viewport mutations publish structured viewport events. When the
  active view changes compatibly, `RootApp` also emits the legacy camera,
  plane, channel, rendering or layer event with causal metadata.
  `explicit_viewport_mutations_keep_active_view_legacy_event_compatibility`
  and the Python event tests cover the mapping and deserialization contract.
- Stable handles remain ID-addressed and stale IDs produce resource-not-found
  errors after removal; active-view changes do not retarget them.

## Central API and Python SDK

The central registry and root dispatcher expose:

- workspace get/layout/swap and workspace screenshot methods;
- viewport list/get/create/clone/rename/remove/activate methods;
- explicit camera, plane, aggregate/specialized channel, rendering, object
  style/legend/filter and native-layer methods; and
- canonical `viewer.viewport_links.list/create/update/remove` methods. The
  fixed group ID is `comparison-navigation`; legacy link get/set remains
  compatible.

Every explicit viewport method validates `viewport_id`. Navigation and
presentation revision guards are separate. The registry test checks method
presence, schemas, permissions, modes and events; the application-surface test
checks registry/manifest parity.

The sync and async Python clients expose matching `Viewport`, bound
`ViewportObjects`, `Viewports`, `ViewportWorkspace` and `ViewportLinks`
resources. Both `layout=` and the planned `split=..., viewports=...` layout
form are supported. `compare()` performs only
canonical protocol calls and returns two stable handles. The interactive
example demonstrates independent channels, styles, filters, rendering,
layers, links, screenshots and diagnostics. `test_viewport_resources.py`
covers sync/async forwarding, local validation, revision chaining and canonical
link calls. The generated exhaustive member reference is current.

Legacy `viewer.*`, MCP tools and existing Python resources continue to target
the active viewport. The complete Rust suite includes the MCP bridge/tool tests,
and the complete Python suite exercises existing resources unchanged.

Filter-sensitive selection, analysis, measurement and export operations reject
ambiguous multi-view calls. They require a viewport, standalone query, all-
objects scope, or explicit active-view opt-in. This is verified by
`filter_sensitive_operations_require_and_honor_an_explicit_source` and Python
resource tests.

## Output, persistence and lifecycle

| Requirement | Verification evidence |
| --- | --- |
| Explicit per-viewport screenshot queues and cleanup | `viewport_screenshot_queue_keeps_targets_independent_and_cleans_removed_view` |
| Whole-workspace crop uses both canvas rectangles in layout order | `workspace_canvas_rect_is_the_union_of_both_viewport_canvases`; root crop test; 1,584 by 1,032 live workspace capture |
| Canvas screenshot excludes persistent headers | live workspace capture contains the two canvas rectangles without canvas chrome |
| Explicit project-view capture | `ProjectViews.capture(viewport=...)` and `control_project_view_spec_for_viewport` |
| Versioned two-view workspace persistence | `multi_viewport_workspace_roundtrips_through_versioned_project_state` and channel/overlay round-trip tests |
| Legacy project migration | `legacy_project_view_migrates_to_one_viewport` plus the data-contract suite |
| Deep links keep active-view compatibility | complete deep-link suite; workspace deep links remain explicitly later scope |
| Session cleanup does not own native viewports/data | viewports live in the native document workspace, not session-owned control resources; full control resource/UI lifecycle tests pass |

## Performance and hardening evidence

`multi-viewport-performance.md` records the reproducible benchmark and live
counters. The latest debug-profile run measured a 0.1870 ms single-view and
0.2354 ms two-view frame-plan EMA. One document, dataset, decoded cache and
object-geometry owner remained present. The live GPU run reported one raw cache
with 18 entries while rendering both canvases.

Slow decode, cached failure, viewport removal during loading, transient gesture
cleanup, stale handle behavior and screenshot queue cleanup have deterministic
tests. Clippy completes successfully; its warnings are the repository's
existing lint backlog rather than errors introduced by the feature.

## Final gates

Run on 2026-08-21:

```text
cargo test --all-targets
  219 passed, 0 failed, 4 ignored

PYTHONPATH=python/src python/.venv/bin/python -m unittest discover -s python/tests -v
  86 passed, 0 failed

cargo clippy --all-targets
  completed successfully (warnings only)

cargo fmt --check
python/.venv/bin/python scripts/generate_python_api_reference.py --check
python/.venv/bin/python -m compileall -q python/src examples/interactive_python_api.py
jq empty api/application-surface.json
git diff --check
  all passed
```

The live native smoke launched the checked-in synthetic OME-Zarr, created and
configured both viewports exclusively through Python, populated the shared GPU
cache, and captured two aligned canvases with different channel composites.

## Explicitly deferred capabilities

These are architectural expansion points, not missing first-milestone work:

- more than two viewports and recursive/grid layouts;
- multiple named link groups and independent selection sets;
- unrelated or coordinate-mapped datasets in one workspace;
- detached operating-system windows;
- workspace deep links; and
- multi-viewport Mosaic mode.
