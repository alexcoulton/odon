# Background Control Acceptance Record

Status: complete for the macOS release candidate; Windows/Linux explicitly excluded from this local sign-off

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
| macOS — visible | Passed on commit `43a44f5` | Passed; first inspected frame showed the final comparison | `background-control-results/macos-visible-debug-43a44f5.json`; instance `9c221de2-a6ca-601b-458e-4696032a26fa`; Codex visual review |
| macOS — fully covered | Passed on commit `43a44f5` | Passed after foregrounding with no Odon command | `background-control-results/macos-covered-debug-43a44f5.json`; same instance; Codex visual review |
| macOS — minimized | Passed on commit `43a44f5` | Passed after restore; tiles materialized within four seconds without another Odon command | `background-control-results/macos-minimized-debug-43a44f5.json`; same instance; Codex visual review |
| macOS — separate Space | Passed on commit `43a44f5` | Passed after switching directly to the full-screen Space with no Odon command | `background-control-results/macos-separate-space-debug-43a44f5.json`; same instance; Codex visual review |
| Windows — fully covered | Excluded from this sign-off | Excluded | No Windows runner/build is available in the macOS workspace; Codex review. Reopen before claiming Windows release support. |
| Windows — minimized | Excluded from this sign-off | Excluded | Same Windows platform exclusion. |
| Linux — fully covered | Excluded from this sign-off | Excluded | No Linux display server/build is available in the macOS workspace; Codex review. Reopen before claiming Linux release support. |
| Linux — minimized | Excluded from this sign-off | Excluded | Same Linux platform exclusion. |

An unsupported platform/build may be recorded as an explicit exclusion with the platform,
version, reason, and reviewer. Do not silently treat “not run” as passing.

## Automated evidence in the current tree

The following checks cover the parts that do not require a real native window condition:

| Requirement | Authoritative check | Current evidence |
| --- | --- | --- |
| Every registered application route is actor-owned in every supported mode | `registered_application_surface_has_no_legacy_execution_routes` | Passing; 286/286 application methods, zero legacy/hybrid routes |
| TCP comparison workflow completes with no UI frame | `comparison_workflow_completes_over_tcp_without_a_ui_frame` | Passing |
| Actor projection coalesces while unconsumed and retains the final workspace | Same TCP comparison test plus actor workspace projection tests | Passing |
| Resource/compute families complete without frames | Domain tests under `src/control/actor/tests/` | Passing in the Rust library suite |
| Legacy UI request dispatcher is absent | `production_control_path_has_no_legacy_ui_dispatcher` | Passing |
| Renderer adapters contain no semantic command emulators | `renderer_has_no_semantic_command_emulators` | Passing; inventory reduced from 55 to zero |
| Native viewport topology has no renderer mutation fallback | `native_workspace_topology_commands_do_not_fall_back_to_renderer_mutation` and `native_workspace_topology_has_no_renderer_mutation_fallback` | Passing |
| Dataset bootstrap restores persisted project views in the actor without importing renderer workspace state | `dataset_bootstrap_restores_actor_project_workspace_and_supersedes_workers` plus the viewport source guard | Passing |
| Production project attachment and dataset switching cannot restore semantic view state directly in the renderer | viewport source guard | Passing; legacy restorer is test-only |
| Periodic renderer observation excludes actor-owned semantic state | `renderer_observation_excludes_actor_owned_workspace_semantics` | Passing |
| Renderer frames cannot advance viewport revisions or linked navigation | `horizontal_split_stacks_each_header_above_an_adjacent_full_height_canvas` plus the viewport source guard | Passing |
| Per-viewport render history and interaction/cache drafts are structurally separate from projected semantic fields | `viewport_render_history_is_explicitly_separated_from_projected_state` | Passing |
| Frame-driven workspace synchronization cannot recapture semantic state from the renderer | `viewport_render_history_is_explicitly_separated_from_projected_state` | Passing; runtime-only capture plus actor-derived native-layer topology |
| Native camera fit queues before the first projection without mutating renderer state | `native_camera_fit_queues_before_first_projection_without_mutating_renderer_state` | Passing |
| Native channel controls queue before the first projection without mutating renderer state | `native_channel_controls_queue_before_first_projection_without_mutating_renderer_state` plus the viewport source guard | Passing |
| Native layer commits queue before the first projection without mutating renderer state | `native_layer_commits_queue_before_first_projection_without_mutating_renderer_state` plus the viewport source guard | Passing |
| Actor-owned panel, tab, rendering, and scale-bar controls wait for projection | viewport source guard and actor-backed rendering preference tests | Passing; both single-view panel tabs are actor-owned |
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

Final cumulative run for commit `43a44f5`:

- Rust library: 188 passed, 0 failed.
- Rust `odon` binary: 203 passed, 0 failed, 4 tests intentionally ignored; the frame-planning benchmark was also run explicitly and passed, while three extended-fixture tests remain unavailable.
- Rust `data_contracts` integration: 10 passed, 0 failed.
- Python SDK: 100 passed, 0 failed.
- Frame planning: 0.1159 ms single-view and 0.1959 ms split-view EMA.

Formatting, all-target compilation, JSON validation, generated Python-reference checking,
application-surface coverage, ownership-ledger coverage, queue backpressure, model latency, and
shared-resource allocation checks passed on the same candidate.

## Final macOS candidate record

The signed-off native candidate is debug commit `43a44f5`, Odon 0.1.5, macOS 15.5 (24F74),
Apple Silicon (`arm64`), instance `9c221de2-a6ca-601b-458e-4696032a26fa`. The visible, covered,
minimized, and separate-Space runs completed 82 actor method checks and audited 286 application
routes each. Their elapsed times were 0.802 s, 0.843 s, 0.820 s, and 1.075 s respectively.
In every condition the final actor and presented projection revisions matched at verifier exit.

Visual review used an OS-level activation and screen capture with no intervening Odon API, menu,
or keyboard command. All four restored views showed the requested linked horizontal comparison:
blue channel 0 on the left, pink channel 1 on the right, distinct object overlays/fill opacity,
the acceptance mask visible only on the left, and no cross-viewport painting. The minimized case
showed the correct semantic projection immediately; its image tiles repopulated over the next four
seconds while Odon remained frontmost, without another command.

The native presentation probe queued `viewer.screenshot.capture` while the exact window was still
minimized. On this macOS/eframe combination the renderer continued offscreen presentation, so the
task had already advanced to structured phase `writing_output` at the 0.5-second observation and
then completed atomically (186,858-byte PNG, capture ID 1, projection 460). The deterministic
no-frame actor tests separately exercise the suppressed-frame branch, including structured
`waiting_for_presentation`, timeout, cancellation, stale acknowledgements, no-clobber behavior,
and partial-output cleanup.

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

Rendered-pixel methods are intentionally different from semantic/resource methods. On a platform
that suppresses frames while Odon is covered or minimized, queue a screenshot capture and confirm
that it reports an explicit `waiting_for_presentation` phase rather than claiming completion.
Return to Odon and confirm that the generation-matched capture completes. If the platform continues
presenting offscreen, record that behavior and verify the deterministic actor no-frame test instead.
Record this separately from the semantic verifier because pixel production legitimately requires a
renderer frame.

## Release sign-off

Wave 7 and Milestone 8 are complete for the recorded macOS candidate because:

- every required row above is passing or explicitly excluded;
- projection inspection is recorded for every passing semantic run;
- the complete Rust, Python, schema/reference, and application-surface suites pass on the final
  tree; and
- performance evidence satisfies the latency, queue-boundedness, projection-allocation, and frame
  planning requirements in the architecture plan.

Windows and Linux remain explicit platform exclusions for this sign-off. Their rows must be reopened
and exercised on real native runners before either platform is claimed as release-supported.
