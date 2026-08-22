# Odon Application State Ownership Inventory

Status: active migration guide

Date: 2026-08-22

This document classifies the state still assembled by `OmeZarrViewerApp` after the application
source split. It prevents file modularization from being mistaken for ownership modularization.
The long-term rule is that semantic state has one owner in the control actor, immutable CPU
resources cross the actor/renderer boundary through generation-tagged handles, and `app` retains
only renderer resources, frame-local interaction state, and narrow platform effects.

## Ownership classes

| Class | Meaning | Permitted mutation |
| --- | --- | --- |
| Actor semantic | Durable/queryable behavior exposed to native UI, Python, MCP, projects, or deep links | Typed actor commands only |
| Shared resource | Immutable or internally synchronized CPU data identified by a generation | Bounded worker produces it; actor validates and installs it |
| Renderer | GPU handles, caches, tile queues, and observations derived from an actor projection | Render thread only; never written back as semantic truth |
| Transient UI | Dialog drafts, hover/drag state, and optimistic gesture previews | Render thread; commit semantic results through a command |
| Legacy semantic | Existing frame-owned state scheduled for a named migration wave | Compatibility handlers only; no new API may be added here |

## Field inventory

The rows below cover every field of `OmeZarrViewerApp`. A row may describe a tightly coupled field
family, but each concrete field is named. “Destination” is the intended owner after the actor plan,
not necessarily the owner in the current compatibility implementation.

| Area | Fields | Current role | Destination / removal wave |
| --- | --- | --- | --- |
| Primary document | `dataset`, `store` | Renderer-compatible document metadata and storage | Actor document descriptor plus shared resource handle; renderer projection, Wave 6/7 cleanup |
| Remote execution | `remote_runtime` | Runtime retained by the viewer | Resource service outside UI; Wave 2C |
| Image loaders | `loader`, `raw_loader` | CPU/GPU tile worker handles | Renderer |
| NGFF labels | `label_cells`, `label_loader`, `label_cells_xform` | Shared label metadata and renderer loader/mapping | Actor label resource handle plus renderer loader; remove duplicate metadata in Wave 7 |
| Label dialog | `seg_label_names`, `seg_label_selected`, `seg_label_input`, `seg_label_status`, `seg_label_prompt_open`, `seg_label_prompt_always`, `seg_label_prompt_preference` | Mixed discovered data, preference, and dialog draft | Actor resource/preferences for names and policy; transient UI for open/input/status; Wave 2E/6 |
| Histogram/channel maxima | `hist_loader`, `chanmax_loader`, `chanmax_request_id`, `chanmax_level`, `chanmax_pending`, `chanmax_snapshot`, `hist`, `hist_request_id`, `hist_request_pending`, `hist_dirty`, `hist_navigation_dirty_since`, `hist_last_sent` | Async analysis scheduling and cached results | Retained actor compute operations and shared results; renderer-only plot cache; Wave 3 |
| Tile cache | `cache`, `pending` | Decoded texture cache and upload queue | Renderer |
| Camera/render revision | `camera`, `active_render_id`, `previous_render_id`, `active_render_smooth_pixels`, `previous_render_smooth_pixels` | Actor projection mirror and renderer invalidation | Actor owns camera/smoothing; render IDs remain renderer observations; Wave 7 deletes semantic mirror writes |
| Plane/render history | `previous_view_selection`, `previous_displayed_view_selection`, `last_render_view_selection`, `last_canvas_rect`, `last_target_level`, `fallback_ceiling_level`, `last_visible_world_tiles`, `zoom_out_floor_level`, `zoom_out_floor_until`, `zoom_out_floor_visible_world_tiles` | Rendering heuristics and measured geometry | Renderer |
| Channel and plane semantics | `selected_channel`, `view_plane_mode`, `draft_view_slice_level0`, `current_x_level0`, `current_y_level0`, `current_z_level0`, `channels`, `channel_window_overrides`, `auto_contrast_settings`, `fast_object_rendering`, `channel_list_search` | Actor-owned values with compatibility mirrors plus a transient slice draft | Actor semantic except `draft_view_slice_level0`; renderer consumes projection; Wave 7 mirror removal |
| Active layer/selection UI | `active_layer`, `selected_channel_layers`, `channel_select_anchor_idx`, `selected_channel_group_id`, `quick_contrast_target`, `selected_overlay_layers`, `overlay_select_anchor_pos` | Mixed semantic active target and UI multi-selection | Actor owns active layer/group where externally visible; anchors and multi-select are transient UI; Wave 6/7 |
| Panel/application UI | `show_left_panel`, `show_right_panel`, `close_dialog_open`, `left_tab`, `right_tab` | Projectable UI settings and platform dialog state | Actor settings for visible panels/tabs; close dialog remains transient platform UI; Wave 6/7 |
| Memory/pinning | `memory_selected_channels`, `pinned_levels`, `pending_memory_load`, `memory_status`, `system_memory`, `system_memory_last_refresh`, `prefer_pinned_finer_levels` | Legacy memory operation, resources, and UI status | Actor retained task and resource generations; system snapshot may remain UI cache; Wave 3 |
| Project/ROI panels | `project_space`, `project_cfg_seen`, `roi_selector`, `cell_thresholds` | Actor project mirror plus legacy panel models | Actor project/analysis models; UI keeps only drafts; Waves 2E and 3, mirror removal Wave 7 |
| Point/annotation data | `cell_points`, `annotation_layers`, `next_annotation_layer_id` | Legacy semantic resources and IDs | Actor resource registry/shared payloads; Wave 2E/6 |
| Masks | `mask_layers`, `next_mask_layer_id`, `mask_layers_project_dirty`, `control_actor_mask_generation`, `control_actor_mask_undo_available` | Actor-owned mask projection with compatibility bookkeeping | Actor semantic/resource state; renderer keeps generation only; Wave 7 |
| Mask gestures | `tool_mode`, `drawing_mask_layer`, `drawing_mask_polygon`, `selected_mask_polygon`, `selected_mask_vertex`, `dragging_mask_vertex`, `moving_mask_polygon`, `undo_stack`, `native_mask_actor_intent_emitted` | Frame-local edit previews plus legacy undo copy | Gestures stay transient; committed polygons and undo history stay actor-owned; Wave 7 removes duplicate undo stack |
| Object selection gestures | `selection_rect_start_world`, `selection_rect_current_world`, `selection_lasso_world` | Frame-local rectangle/lasso previews | Transient UI; completed selection commits to actor |
| Threshold analysis | `threshold_region_min_pixels`, `threshold_region_scope`, `threshold_region_full_level`, `threshold_region_status`, `threshold_region_preview`, `threshold_region_preview_generation` | Legacy compute parameters/result and render generation | Actor retained compute task plus shared result; renderer generation only; Wave 3 |
| Cell outline presentation | `cells_outlines_visible`, `cells_outlines_color_rgb`, `cells_outlines_opacity`, `cells_outlines_width_px` | Viewport presentation compatibility state | Actor per-viewport presentation; Wave 6/7 |
| GPU renderers | `points_gl`, `threshold_preview_gl`, `tiles_gl`, `labels_gl` | GPU resources | Renderer |
| Remote dialog/session | `remote_dialog_open`, `remote_mode`, `remote_http_url`, `remote_s3_endpoint`, `remote_s3_region`, `remote_s3_bucket`, `remote_s3_prefix`, `remote_s3_access_key`, `remote_s3_secret_key`, `remote_status`, `remote_s3_browser` | Dialog drafts mixed with credentials/session/listing | UI owns drafts/open/status; actor-side secret service owns credentials; shared listing result; Wave 2C |
| Platform requests | `pending_request` | Frame-to-root compatibility message | Replace semantic variants with actor commands; retain only platform effects; Wave 6 |
| Native command bridge | `native_control_intents` | Buffered native UI commands | Narrow per-frame command outbox; retain until direct actor handle is available |
| Projection generations | `control_actor_object_generation`, `control_actor_label_generation`, `control_actor_object_selection_generation` | Last actor generation consumed by renderer | Renderer observation |
| Dialog/help state | `group_layers_dialog`, `hover_tooltip_state`, `active_help_topic`, `roi_info_open` | Dialog drafts and hover/help presentation | Transient UI |
| Render preferences/debug | `smooth_pixels`, `show_tile_debug`, `mask_draw_debug_stats`, `show_scale_bar`, `show_hud` | Actor projection mirrors plus renderer diagnostics | Actor owns user-visible settings; debug statistics remain renderer; Wave 7 |
| Tile scheduling settings | `tile_loader_threads`, `tile_prefetch_mode`, `tile_prefetch_aggressiveness`, `tile_loading_status` | Renderer configuration/status | Renderer, with actor-owned persisted preference only if exposed remotely |
| Object/spatial resources | `seg_geojson`, `seg_objects`, `spatial_image_layers`, `spatial_layers`, `spatial_image_transform`, `spatial_label_transform`, `spatial_root`, `spatial_label_store`, `xenium_layers` | Mixed immutable resources and legacy semantic presentation | Actor descriptors/generations plus shared resources; renderer-native caches remain local; Waves 2D/2E/6 |
| Layer transforms | `channel_offsets_world`, `channel_scales`, `channel_rotations_rad`, `loaded_layer_offsets_world`, `points_offset_world`, `spatial_points_offset_world`, `seg_labels_offset_world`, `seg_geojson_offset_world`, `seg_objects_offset_world`, `xenium_cells_offset_world`, `xenium_transcripts_offset_world` | Actor-native presentation for migrated layers and legacy mirrors for others | Actor per-viewport presentation; Wave 6/7 |
| Layer ordering/gestures | `overlay_layer_order`, `channel_layer_order`, `channel_sort_mode`, `layer_drag`, `layer_move`, `layer_transform` | Actor ordering mirror plus transient gestures | Actor owns order; drag/move/transform previews remain transient; Wave 7 |
| TIFF plane state | `tiff_plane_state` | Alternate-document resource plus dialog draft | Actor document/plane model and transient draft; Wave 2D |
| Screenshots | `screenshot_settings`, `screenshot_settings_open`, `screenshot_worker`, `screenshot_next_id`, `screenshot_pending`, `screenshot_in_flight`, `screenshot_output_dir` | Settings, presentation-task queue, worker, and dialog | Actor owns settings/task lifecycle; render thread fulfills pixel request; Wave 5 |
| Multi-viewport | `viewport_workspace`, `viewport_layer_groups` | Actor projection mirror | Actor semantic model; renderer consumes projection; Wave 7 |
| Per-frame active resources | `viewport_raw_active_keys`, `viewport_cpu_active_keys`, `viewport_label_active_keys`, `viewport_spatial_image_active_keys` | Aggregate cache-retention observations | Renderer |
| Frame planning telemetry | `viewport_frame_plan_ms`, `viewport_frame_plan_ema_ms`, `viewport_frame_plan_samples` | Renderer performance telemetry | Renderer diagnostics |

## Source responsibility map

`src/app/mod.rs` defines shared viewer types, the compatibility aggregate, and small pure helpers.
Production behavior is organized as follows:

| Module | Responsibility |
| --- | --- |
| `construction.rs`, `lifecycle.rs`, `update.rs` | Construction, narrow platform lifecycle, top-level frame orchestration |
| `actor_projection.rs`, `viewport_runtime.rs`, `viewport_ui.rs` | Actor projection consumption, viewport runtime state, workspace chrome |
| `canvas.rs`, `overlay_rendering.rs`, `navigation.rs`, `tile_runtime.rs`, `loading.rs` | Canvas interaction, overlay drawing, navigation, tile scheduling/draining, loading diagnostics |
| `layer_runtime.rs`, `layer_properties.rs`, `project_view.rs`, `layers_ui.rs`, `contrast_ui.rs` | Layer semantics compatibility, properties, project-view conversion, layer panels/groups, contrast/histogram UI |
| `datasets.rs`, `tiff.rs`, `remote.rs`, `image_runtime.rs` | Dataset switching, TIFF planes, remote browser, image/view helpers |
| `selection.rs`, `thresholds.rs`, `mask_interaction.rs` | Spatial selection, threshold preview, transient mask gestures |
| `projects.rs`, `project_integration.rs`, `deep_links.rs` | Project state, project resources, deep-link compatibility |
| `memory_ui.rs`, `screenshots.rs` | Memory/pinning UI and screenshot presentation work |
| `legacy_control/` | Explicitly temporary frame-driven control handlers, split by domain |
| `tests/` | Characterization and structural regression tests, split by behavior |

## Dependency rules

1. `update.rs` may orchestrate domain modules but must not contain domain command handlers.
2. New remotely visible semantics are added to `src/model` and `src/control/actor`, never to
   `legacy_control`.
3. A sibling app module may expose an internal helper with `pub(super)`; public methods are kept
   only where `RootApp` or another crate-facing integration requires them.
4. UI modules do not perform blocking I/O. They submit actor work or consume already completed
   resources.
5. Renderer observations carry the projection/resource generation they consumed and cannot mutate
   actor semantic state directly.
6. Tests live under `src/app/tests`; the structural test enforces generous size ceilings so a new
   monolith is caught before it reaches the former scale.

## Migration deletion order

After each actor wave passes its paused-frame and native/remote parity tests:

1. remove that domain's methods from `legacy_control`;
2. remove semantic fields listed above from `OmeZarrViewerApp` or replace them with a consumed
   generation/resource handle;
3. route native UI commits through the typed command;
4. retain only renderer and transient UI fields in the corresponding app module; and
5. update this inventory in the same change.

`legacy_control` disappears in Wave 6. The remaining actor-owned compatibility mirrors disappear
in Wave 7; source modularization alone is not the completion condition.
