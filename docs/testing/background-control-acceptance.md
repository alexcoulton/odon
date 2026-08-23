# Background Control Acceptance Record

Status: in progress

Date opened: 2026-08-22

This record closes Wave 7 of the background control actor migration. It deliberately separates
automated no-frame evidence from literal native-window evidence. A green unit test cannot prove
that macOS treated a window as covered or minimized, and a successful Python run does not prove
that the renderer consumed the final projection when frames resumed.

## Completion gates

The actor release gate is complete only when all three columns below have passing evidence for
every required platform condition:

| Platform condition | Semantic/resource completion while Odon remains in the condition | Final projection appears after returning to Odon without another API call | Evidence |
| --- | --- | --- | --- |
| macOS — visible | Passed on current debug build | Provisional pass from immediate screen-capture inspection; user sign-off pending | `background-control-results/macos-visible-debug-final.json`; instance `46ae75fc-1c69-69c8-7594-5d6e5c82c974` |
| macOS — fully covered | Passed on current debug build | Provisional pass from immediate screen-capture inspection; user sign-off pending | `background-control-results/macos-covered-debug-final.json`; instance `8f1b7671-b1cc-4d7c-1497-4bc0fa2a05fb` |
| macOS — minimized | Passed on current debug build | Provisional pass from immediate screen-capture inspection; user sign-off pending | `background-control-results/macos-minimized-debug-final.json`; instance `46ae75fc-1c69-69c8-7594-5d6e5c82c974` |
| macOS — separate Space | Passed on current debug build | Provisional pass from first visible full-screen capture; user sign-off pending | `background-control-results/macos-separate-space-debug-final.json`; instance `46ae75fc-1c69-69c8-7594-5d6e5c82c974` |
| Windows — fully covered | Not yet run | Not yet run | Required when a supported Windows build is available |
| Windows — minimized | Not yet run | Not yet run | Required when a supported Windows build is available |
| Linux — fully covered | Not yet run | Not yet run | Required when a supported Linux window system/build is available |
| Linux — minimized | Not yet run | Not yet run | Required when a supported Linux window system/build is available |

An unsupported platform/build may be recorded as an explicit exclusion with the platform,
version, reason, and reviewer. Do not silently treat “not run” as passing.

## Automated evidence in the current tree

The following checks cover the parts that do not require a real native window condition:

| Requirement | Authoritative check | Current evidence |
| --- | --- | --- |
| Every registered application route is actor-owned in every supported mode | `registered_application_surface_has_no_legacy_execution_routes` | Passing; 263/263 application methods, zero legacy/hybrid routes |
| TCP comparison workflow completes with no UI frame | `comparison_workflow_completes_over_tcp_without_a_ui_frame` | Passing |
| Actor projection coalesces while unconsumed and retains the final workspace | Same TCP comparison test plus actor workspace projection tests | Passing |
| Resource/compute families complete without frames | Domain tests under `src/control/actor/tests/` | Passing in the Rust library suite |
| Legacy UI request dispatcher is absent | `production_control_path_has_no_legacy_ui_dispatcher` | Passing |
| Semantic events cannot bypass the actor through renderer snapshot diffs | Same source guard; `publish_native_event` and `control_observed_state` are forbidden | Passing |
| Native project/viewer/mosaic commits do not use before/after snapshot translators | `root_has_no_native_snapshot_translator` plus native direct-intent tests | Passing |
| Local actor remains available when optional TCP publication fails | `optional_tcp_failure_keeps_the_local_actor_available` | Passing |
| Mosaic actor projections cannot feed commands back into the actor | `actor_owned_mosaic_interactions_emit_commands_without_semantic_mutation` | Passing |
| Split-view overlays cannot paint into a sibling canvas | `viewport_canvas_establishes_a_hard_clip_before_painting` plus current-build zoomed screen-capture reproduction | Passing |
| Actor-controlled label projection does not leave an interactive native prompt | `label_control_state_and_channel_presentation_are_bounded` plus current-build covered-run inspection | Passing |
| Actor, application, and command-registry implementations remain modular | Source-organization tests including `app_source_stays_split_by_responsibility`, `root_app_source_stays_split_by_responsibility`, `control_registry_stays_split_by_responsibility`, and `actor_resource_worker_stays_split_by_domain` | Passing |
| Mosaic intent submission, project persistence, control snapshots, screenshots, and host effects remain modular | `mosaic_app_stays_split_by_responsibility` | Passing |
| Canonical mask state, CRUD commands, validation, GeoJSON I/O, and tests remain modular | `mask_model_stays_split_by_responsibility` | Passing |
| Deep-link DTOs, canonicalization, resolution, parsing, semantic derivation, and tests remain modular | `deep_link_surface_stays_split_by_responsibility` | Passing |
| Canonical project installation/persistence, command/navigation, and validation remain modular | `project_model_stays_split_by_responsibility` | Passing |
| Native-layer command, contrast, metadata, offset, geometry, and ordering responsibilities remain modular | `layer_runtime_stays_split_by_responsibility` | Passing |
| Annotation model/UI, hit testing, color LUTs, parquet decoding, and GL drawing remain modular | `annotation_points_layer_stays_split_by_responsibility` | Passing |
| Raw-tile GL façade, paint orchestration, resource lifecycle, geometry, upload, and shaders remain modular | `tile_gl_renderer_stays_split_by_responsibility` | Passing |
| Generic and object-selection line-bin renderers remain separate from shared GL program compilation | `line_bin_renderers_stay_split_by_responsibility` | Passing |
| TCP bridge transport, connection dispatch, tasks, readiness waits, protocol services, and tests remain modular | `local_control_runtime_and_tcp_bridge_remain_separate` | Passing |
| Native TIFF inspection, metadata, loaders, and tests remain modular | `tiff_pyramid_stays_split_by_responsibility` | Passing |
| SpatialData collection, shape, point, and payload-preparation adapters remain modular | `spatialdata_layers_stay_split_by_responsibility` | Passing |
| SpatialData parquet loading, WKB geometry decoding, and decoder tests remain modular | `spatialdata_parquet_shapes_stay_split_by_responsibility` | Passing |
| Declarative UI registry, rendering, validation/bindings, and tests remain modular | `declarative_ui_stays_split_by_responsibility` | Passing |
| Cell-threshold UI, parquet loading, and threshold-file parsing remain modular | `cell_threshold_panel_stays_split_by_responsibility` | Passing |
| Median local model-command round trip is below 5 ms | `local_model_command_median_round_trip_stays_below_five_milliseconds` | Passing; 19–39 µs median in current debug-tree runs |
| Camera projection does not copy object geometry | `camera_projection_reuses_large_object_resource_without_copying_geometry` | Passing with a 10,000-object shared resource |
| Multi-viewport frame planning remains within the existing characterization budget | `benchmark_single_and_two_viewport_frame_planning` | Passing; 0.2474 ms single and 0.2414 ms split EMA in the current debug-tree run |
| Python SDK surface and generated reference remain synchronized | Python SDK unit suite and `generate_python_api_reference.py --check` | Passing in the latest cumulative run |

Latest cumulative run before platform acceptance:

- Rust library: 170 passed, 0 failed.
- Rust `odon` binary: 188 passed, 0 failed, 4 tests intentionally ignored by the latest cumulative run; the frame-planning benchmark was previously run explicitly and passed, while three extended-fixture tests remain unavailable.
- Rust `data_contracts` integration: 10 passed, 0 failed.
- Python SDK: 88 passed, 0 failed.

These counts are working-tree evidence, not a substitute for rerunning the cumulative commands
immediately before release or commit.

The current-tree visible run completed 81 semantic/resource operations in 0.680 seconds with the
isolated Odon process frontmost. The actor and renderer both reported projection revision 112 at
completion. An atomic OS-level foreground-and-capture step, with no intervening Odon command,
showed the requested blue-left/pink-right comparison, distinct overlays, and correct viewport
clipping. This run used the same macOS 15.5 (24F74), Apple Silicon, Odon 0.1.5 working tree as the
minimized run. Its JSON retains `"projection_inspection": "pending"`; the table is the separate
visual-review record and still requires user sign-off.

The current-tree covered run completed 81 semantic/resource operations in 0.720 seconds while a
maximized terminal covered Odon. The final actor state asserted a linked horizontal comparison,
blue channel 0 on the left, pink channel 1 on the right, distinct object filters/fill opacities,
and per-viewport mask visibility. Odon was then foregrounded and captured without an intervening
Odon API or UI command. The projection was present, unobstructed by the label-discovery modal, and
all overlays remained within their viewport canvas. The generated JSON deliberately retains
`"projection_inspection": "pending"`; the table above is the separate review record and still
requires user sign-off.

The current-tree minimized run used a freshly built, isolated Odon process after confirming that
the instance registry was empty. macOS reported that process's only window as minimized for the
complete 81-operation verifier run, which completed in 0.676 seconds. The exact process was then
restored, foregrounded, and captured atomically without an intervening Odon command. Its first
visible frame contained the expected linked comparison: blue channel 0 on the left, pink channel 1
on the right, distinct overlays, and no cross-viewport painting. This run used macOS 15.5
(24F74), Apple Silicon, Odon 0.1.5 from current working tree base `7737aab`. The JSON likewise
retains `"projection_inspection": "pending"`; the table records the separate visual review and
still requires user sign-off.

The current-tree separate-Space run placed the isolated Odon process in its own macOS full-screen
Space, switched back to iTerm2, and confirmed that accessibility exposed zero Odon windows on the
active Space. All 81 operations completed there in 0.684 seconds. Activating the exact Odon PID
then switched to its full-screen Space; the first capture, made without an intervening Odon
command, contained the expected linked blue-left/pink-right comparison and correctly clipped
overlays. This run used macOS 15.5 (24F74), Apple Silicon, and Odon 0.1.5 from the current working
tree base `7737aab`. Its JSON retains `"projection_inspection": "pending"`; the table records the
separate visual review and still requires user sign-off.

## Native-window procedure

Build and launch the current working tree. For each macOS condition, keep Odon in that condition
for the complete Python run and do not switch to it while the verifier is executing:

```bash
uv run --project python python scripts/verify_background_control.py \
  --instance <instance-id-from-odon.list_instances()> \
  --condition covered \
  --output docs/testing/background-control-results/macos-covered.json
```

Always pass the instance ID for the freshly built application. This prevents a stale or separately
installed Odon process from satisfying the semantic checks. The generated record includes the
negotiated instance ID and application version.

Repeat with `visible`, `minimized`, and `separate-space`, changing the output filename. The JSON
record proves only semantic/resource completion and therefore writes
`"projection_inspection": "pending"`.

After the script exits:

1. Return to Odon without sending another MCP, Python, menu, or keyboard command.
2. Verify that the two horizontal viewports appear immediately.
3. Verify that the left viewport shows only channel 0 in blue and the right shows only channel 1
   in pink.
4. Verify that both object overlays and the generated mask are present, with the mask hidden only
   in the right viewport.
5. Verify that the two object filters and fill opacities differ as requested.
6. Record the visual result, macOS version, hardware, Odon build/commit, elapsed time, reviewer,
   and JSON evidence path in the table above.

Do not alter the generated JSON to claim visual success; the table is the human-reviewed visual
record. If a run fails, retain its JSON or terminal output with a `-failed` suffix and describe the
observed state before rerunning.

## Presentation-task check

Rendered-pixel methods are intentionally different from semantic/resource methods. While Odon is
covered or minimized, queue a screenshot capture and confirm that it reports an explicit
presentation-waiting state rather than claiming completion. Return to Odon and confirm that the
generation-matched capture completes. Record this separately from the semantic verifier because
pixel production legitimately requires a renderer frame.

## Release sign-off

Wave 7 may be marked complete only after:

- every required row above is passing or explicitly excluded;
- projection inspection is recorded for every passing semantic run;
- the complete Rust, Python, schema/reference, and application-surface suites pass on the final
  tree; and
- performance evidence satisfies the latency, queue-boundedness, projection-allocation, and frame
  planning requirements in the architecture plan.
