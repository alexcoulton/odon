# Odon Application State Ownership Inventory

Status: Wave 7 ownership and release audit

Date: 2026-08-23

This document classifies the state still assembled by `OmeZarrViewerApp` after the application
source split. It prevents file modularization from being mistaken for ownership modularization.
The long-term rule is that semantic state has one owner in the control actor, immutable CPU
resources cross the actor/renderer boundary through generation-tagged handles, and `app` retains
only renderer resources, frame-local interaction state, and narrow platform effects.

The executable ledger is `api/state-ownership-ledger.json`. It covers every field of
`OmeZarrViewerApp`, `RootApp`, and `MosaicViewerApp` exactly once and records each family's current
class, disposition, canonical writer, projection source, renderer and persistence consumers,
native commit point, and completion milestone. The structural test
`application_state_ownership_ledger_covers_every_host_field_exactly_once` compares the ledger to
the Rust struct definitions, rejects duplicate or unknown fields, and requires new fields to be
classified before the suite can pass.

Current executable-ledger baseline:

| Host | Fields | Retain | Narrow | Replace | Delete |
| --- | ---: | ---: | ---: | ---: | ---: |
| `OmeZarrViewerApp` | 193 | 64 | 100 | 29 | 0 |
| `RootApp` | 43 | 16 | 18 | 8 | 1 |
| `MosaicViewerApp` | 83 | 20 | 59 | 4 | 0 |
| **Total** | **319** | **100** | **177** | **41** | **1** |

`Retain` does not mean actor ownership: retained fields are renderer resources/observations,
transient UI, shared-resource handles, or narrow platform effects. `Narrow`, `replace`, and
`delete` are the remaining ownership-cleanup queue and must be updated as each slice lands.

The initial test-only renderer semantic-emulator baseline was 55 allowlisted methods. Milestone 2
retired all 55 across workspace topology, navigation, channels, layers, rendering preferences,
labels, object presentation, filters, and selection. The current inventory is zero.
`renderer_has_no_semantic_command_emulators` scans `renderer_bridge` and fails if a recognized
mutation family is reintroduced. Renderer observation, readiness, projection application, and
presentation acknowledgement remain valid responsibilities; application semantic tests now issue
commands through an in-process actor model and consume its projection.

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
| Mask gestures | `tool_mode`, `drawing_mask_layer`, `drawing_mask_polygon`, `selected_mask_polygon`, `selected_mask_vertex`, `dragging_mask_vertex`, `moving_mask_polygon`, `undo_stack` | Frame-local edit previews plus fallback-only undo storage before an actor projection is installed | Gestures stay transient; committed polygons and undo history are actor-owned |
| Object selection gestures | `selection_rect_start_world`, `selection_rect_current_world`, `selection_lasso_world` | Frame-local rectangle/lasso previews | Transient UI; completed selection commits to actor |
| Threshold analysis | `threshold_region_min_pixels`, `threshold_region_scope`, `threshold_region_full_level`, `threshold_region_status`, `threshold_region_preview`, `threshold_region_preview_generation` | Legacy compute parameters/result and render generation | Actor retained compute task plus shared result; renderer generation only; Wave 3 |
| Cell outline presentation | `cells_outlines_visible`, `cells_outlines_color_rgb`, `cells_outlines_opacity`, `cells_outlines_width_px` | Viewport presentation compatibility state | Actor per-viewport presentation; Wave 6/7 |
| GPU renderers | `points_gl`, `threshold_preview_gl`, `tiles_gl`, `labels_gl` | GPU resources | Renderer |
| Remote dialog/session | `remote_dialog_open`, `remote_mode`, `remote_http_url`, `remote_s3_endpoint`, `remote_s3_region`, `remote_s3_bucket`, `remote_s3_prefix`, `remote_s3_access_key`, `remote_s3_secret_key`, `remote_status`, `remote_s3_browser` | Dialog drafts mixed with credentials/session/listing | UI owns drafts/open/status; actor-side secret service owns credentials; shared listing result; Wave 2C |
| Platform requests | `pending_request` | Frame-to-root compatibility message | Replace semantic variants with actor commands; retain only platform effects; Wave 6 |
| Native command bridge | `native_control_intents` | Buffered native UI commands | Narrow per-frame command outbox; retain until direct actor handle is available |
| Projection generations | `control_actor_object_generation`, `control_actor_secondary_object_generations`, `control_actor_object_selection_generation`, `control_actor_secondary_object_selection_generations`, `control_actor_label_generation` | Last actor resource/selection generation consumed by renderer | Renderer observation |
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
| `renderer_bridge/` | Renderer observations, readiness, actor projection adapters, and presentation acknowledgement; semantic command emulators are forbidden |
| `tests/` | Characterization and structural regression tests, split by behavior |

The canvas boundary is split by frame phase. `app/canvas.rs` owns allocation, the hard viewport
clip, tile/overlay paint order, and screenshot capture. `app/canvas/interactions.rs` owns transient
camera, selection, mask, move, and transform gestures and submits actor-owned commits through the
existing interaction helpers. Neither file introduces canonical state; both operate on the same
renderer projection for one frame.

The native-layer runtime is split by responsibility while retaining one renderer projection.
`app/layer_runtime.rs` is a module façade; its child modules own actor-command submission and
gesture cancellation, quick-contrast calculations/widgets, layer metadata and visibility access,
offset baselines/commits, transformed visible-region geometry, and deterministic layer-order
rebuilding. Methods remain visible only within `crate::app`, and semantic commits still enter the
typed actor-command outbox.

The annotation renderer adapter keeps its data/presentation model in `annotations/mod.rs`, while
hit testing, category/color LUT construction, parquet schema/data decoding, and OpenGL drawing live
in `annotations/selection.rs`, `colors.rs`, `parquet.rs`, and `gl.rs`. CPU and GL rendering consume
the same immutable ROI payloads and category styles. These modules do not own durable annotation
commands; actor resources and generations remain authoritative.

The raw-tile OpenGL renderer is split below the same renderer-only boundary.
`render/tiles_gl.rs` retains the public draw DTOs and `TilesGl` resource façade;
`tiles_gl/orchestration.rs` owns cache-facing operations and paint sequencing;
`backend.rs` owns raw-tile cache entries, GPU objects, framebuffer targets, and resource lifecycle;
`geometry.rs` owns uniforms, NDC construction, affine transforms, and bounds; `upload.rs` owns R16
texture upload and filtering; and `shaders.rs` owns versioned GLSL sources and program compilation.
These modules consume actor-projected render inputs but do not own application semantics.

The line-bin OpenGL surface has a matching public façade. `render/line_bins_gl/bins.rs` owns the
generic line-segment bin renderer and its shader pair; `objects.rs` owns object-indexed bins,
selection-state textures, and selection-aware shaders; and `program.rs` owns shared shader compile
and link mechanics. The façade re-exports both existing draw APIs unchanged, and all cache, GPU,
and selection-texture state remains renderer-only.

The outer application shell follows the same boundary. `src/root_app.rs` retains construction,
mode transitions, and frame orchestration; `root_app/actor_projection.rs` applies projections and
reports renderer observations; `root_app/remote.rs` owns the native remote-dataset dialog backed
by actor commands; and `root_app/tests.rs` contains boundary tests, including a source guard that
prevents restoration of a native snapshot translator. Structural tests cap the root façade and
production submodules so these responsibilities cannot silently collapse back into a single file.

The control shell is split at the same ownership boundary. `src/control/request.rs` owns the
transport-independent actor request envelope. `src/mcp/runtime.rs` owns the required actor handle,
registries, renderer projections, and optional server-publication status. `src/mcp/bridge.rs` is
the connection-state façade. Its child modules separately own TCP listener/framing and worker
queues (`transport.rs`), JSON-RPC and actor-mailbox dispatch (`dispatch.rs`), batch and retained-task
orchestration (`tasks.rs`), background readiness monitors (`waits.rs`), and task/UI/event protocol
registries (`services.rs`); tests remain in `tests.rs`. A structural regression test prevents the
runtime, façade, or production bridge modules from absorbing one another's responsibilities.

The declarative extension UI remains one `UiRegistry` but no longer one implementation file.
`control/ui.rs` owns extension/contribution lifecycle and action queues; `ui/render.rs` owns native
placement; `ui/render/components.rs` owns recursive widget/event-policy rendering; and
`ui/validation.rs` owns validation, capability authorization, value patching, and native binding
synchronization. Tests are isolated under `control/ui/tests.rs`. This is a source boundary inside
the control service, not a second UI registry or semantic owner.

The actor's bounded resource-worker pool also has an explicit source boundary.
`src/control/actor/worker.rs` owns thread creation and `LoadJob` dispatch, while worker-side memory,
threshold, analysis, measurement, project/ROI, screenshot, and label computations live under
`control/actor/worker/`. They still consume the same bounded queue and return the same typed
completions to the actor; no new ordering authority is introduced.

The transport-independent command registry is also split by responsibility while retaining a
single authoritative surface. `src/control/registry.rs` owns descriptor types and public
lookup/introspection behavior; application descriptors, protocol services and aliases, request
schemas, actor-capability metadata, and tests live in dedicated `control/registry/` modules. This
changes source ownership only—the actor and all clients continue to consume the same registry.

The typed command decoder is organized around that registry rather than mixed into it.
`src/control/command.rs` owns the public command envelope and descriptor/revision resolution;
request DTOs and serde defaults, method-specific validation, and tests live under
`control/command/`. The registry remains the single method catalog and every decoded command still
enters the same actor mailbox.

The native TIFF resource provider is organized separately from actor state ownership.
`src/xenium/tiff_pyramid.rs` defines the shared TIFF/OME-TIFF resource types and public pyramid
API; IFD inspection and level construction, OME metadata parsing, asynchronous decoding/loaders,
and tests live under `xenium/tiff_pyramid/`. Actor workers and renderer setup continue to consume
the same public provider API and shared resource values.

SpatialData layer adapters have a matching renderer-side boundary. `src/spatialdata/layers.rs`
retains collection lifecycle and public selection targets, while shape loading/rendering, point
interaction/rendering, and point payload normalization/cache construction live under
`spatialdata/layers/`. These modules do not own dataset lifecycle or canonical commands; they adapt
actor-prepared SpatialData resources into native renderer overlays.

The parquet-shape source adapter is split below that renderer boundary. `parquet_shapes.rs` owns
schema inspection, projected batch reads, cancellation, and construction of public loaded-object
DTOs; `parquet_shapes/geometry.rs` owns WKB decoding, geometry classification/transforms, centroid
summaries, and Arrow scalar extraction; and its characterization test lives in `tests.rs`. The WKB
parser is private to the adapter and the public SpatialData exports remain unchanged.

The canonical semantic model is also organized by domain rather than by call order. The
`src/model/app.rs` façade defines the shared model types and pure projection/validation helpers;
`model/app/construction.rs`, `runtime.rs`, and `dispatch.rs` own lifecycle boundaries; and the
remaining modules own projects, settings/deep links, compute tasks, resources, objects, masks,
native layers, saved views, and viewport commands. All modules provide inherent implementations
for the same `AppModel`; this is not state sharding and does not create multiple semantic owners.
Tests are isolated in `model/app/tests.rs`, and structural guards cap both the façade and each
production domain module.

Within that model, viewport commands are divided by semantic revision domain. The
`viewport_commands.rs` façade owns shared dataset/revision helpers and active-viewport compatibility
methods; child modules own workspace topology/linking, navigation/plane state, and scoped
presentation respectively. They all mutate the same `DatasetModel.workspace` through the same
actor mailbox; the split introduces no secondary workspace owner.

`MosaicViewerApp` uses the same source-boundary pattern. `src/mosaic/mod.rs` retains shared mosaic
state and small host traits; the `control.rs` façade separates native actor-intent submission,
project serialization/restoration, control snapshots and test-only compatibility queries,
screenshot lifecycle, and host/platform effects beneath `mosaic/control/`; `construction.rs`
prepares renderer resources; `update.rs` owns the frame lifecycle; and memory/navigation,
layers/dialogs, panels, and canvas/tile work live in dedicated modules. Layout characterization
tests are in `mosaic/tests.rs`. The split does not make the mosaic renderer a semantic owner:
actor projections remain authoritative and native mutations still enter the typed command outbox.

Mosaic construction is itself source-adapter based. The `construction.rs` façade selects CLI or
samplesheet entry paths, while actor-resource, local, remote, project, project-config, and
samplesheet preparation live in child modules. All adapters converge on the typed assembly in
`construction/assembly.rs`, the sole initializer for shared renderer defaults. The adapters prepare
resources; they do not create additional semantic state owners.

The corresponding actor-owned `MosaicModel` keeps one semantic state value but no longer one large
implementation file. `src/model/mosaic.rs` owns shared types, installation, snapshot/projection,
and dispatch; memory pinning, object-resource loading, layout/selection/focus navigation, channel
and native-layer presentation, and tests live under `model/mosaic/`. This is source organization,
not state sharding: every command still executes through the same actor ordering authority.

The actor-owned `MaskModel` is likewise one state value behind a compact `model/masks.rs` façade.
Its child modules separate projection/import state transitions, typed CRUD dispatch, parameter and
geometry validation, GeoJSON file decoding, and characterization tests. Cross-module helpers stay
private to `model::masks`; the existing crate-visible model and GeoJSON loader contracts are
unchanged, and all mutations still run under the actor's ordering and generation checks.

The public deep-link surface is source-separated from actor execution as well. `deep_link.rs`
retains the serialized request DTOs, defaults, and stable re-exports; `deep_link/canonical.rs`
owns canonical URL production; `resolution.rs` owns example and project-ROI resolution;
`parsing.rs` owns URL recognition, decoding, aliases, and typed value parsing; `semantics.rs`
derives actor-facing segmentation and filter intent; and tests live separately. This preserves one
public deep-link contract while keeping parsing and resolution independent of renderer state.

The actor-owned `ProjectModel` retains its snapshot and model types in the compact
`model/project.rs` façade. `project/state.rs` owns project installation, persistence payloads,
manifest/sample-sheet/discovery updates, and normalized worker results; `commands.rs` owns typed
project, ROI, saved-view, selection, and focus dispatch; and `validation.rs` owns pure snapshot,
ROI, and parameter normalization. All modules mutate the same private snapshot under the actor's
ordering authority, and the normalized-load helper remains available at its original module path.

The renderer-side object implementation is separated by responsibility as well.
`src/objects/mod.rs` retains shared object-layer state and DTO/types; native actor-service filter
evaluation, typed property-column storage, default construction, and tests live in dedicated
`objects/` modules. `src/objects/core.rs` retains shared object types and small accessors, with runtime/loading,
properties UI, selection, export, and tests under `objects/core/`. `src/objects/analysis.rs` owns
analysis UI and threshold-set orchestration; `objects/analysis/data_selection.rs` owns derived
property caches and live-selection bookkeeping; and `objects/analysis/algorithms.rs` owns pure
histogram, threshold, polygon, fuzzy-name, and scatter helpers. These modules still operate on one
`ObjectsLayer` renderer projection and do not bypass actor ownership.

The object draw path is similarly bounded: `objects/render.rs` retains drawing and presentation,
while hover/query/selection behavior, fill-mesh/cache construction, pure selection geometry, LOD
and color-group construction, and tests live in dedicated `objects/render/` modules. Public and
crate-visible `ObjectsLayer` methods retain their prior API; the split does not change command
routing or introduce renderer-owned semantic commits.

The renderer-side `ObjectsLayer` is similarly split beneath `src/objects/core/`: `runtime.rs`
drains bounded loaders and installs prepared resources, `loading.rs` owns source-format parsing,
`properties_ui.rs` owns transient property/presentation controls, `selection.rs` owns renderer
selection/filter caches, and `export.rs` owns native export presentation and encoding orchestration.
`core.rs` is a small shared-type/helper façade, and characterization tests live in
`core/tests.rs`. Canonical selections, filters, resources, and export task state remain actor-owned;
these modules consume projections or prepare renderer-native data.

The properties boundary is subdivided by state domain rather than widget order:
`core/properties_ui.rs` renders the panel, while child modules own style projection, filter/query
state, and color/legend plus project-display persistence. These are renderer projection helpers on
the same `ObjectsLayer`; actor commands and revisions remain authoritative.

The custom cell-threshold panel also has an explicit I/O boundary. `custom/cell_thresholds.rs`
owns panel state, interactions, and projection into the renderer points layer;
`cell_thresholds/data.rs` owns dataset-path inference, parquet schema inspection, and the bounded
point loader; and `cell_thresholds/threshold_files.rs` owns CSV/JSON threshold configuration
parsing. The split does not promote the panel to a semantic owner: it remains renderer-side
compatibility UI over the actor-owned analysis/resource surface.

`ProjectSpace` is now a small DTO/helper façade plus responsibility modules under
`src/project/space/`. `control_persistence.rs` owns actor projection consumption and the typed
command outbox, `rois_views.rs` owns the renderer-side project projection and UI drafts,
`browser_ui.rs` and `views_ui.rs` own transient presentation, and `imports.rs` owns native
samplesheet/discovery preparation. Characterization tests live in `space/tests.rs`. The actor
remains the durable project owner; this split does not authorize the UI projection to commit
semantic state directly.

## Dependency rules

1. `update.rs` may orchestrate domain modules but must not contain domain command handlers.
2. New remotely visible semantics are added to `src/model` and `src/control/actor`, never to
   `renderer_bridge`; production code there may observe or project state but may not own command semantics.
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

1. remove or test-gate that domain's direct mutation helpers in `renderer_bridge`;
2. remove semantic fields listed above from `OmeZarrViewerApp` or replace them with a consumed
   generation/resource handle;
3. route native UI commits through the typed command;
4. retain only renderer and transient UI fields in the corresponding app module; and
5. update this inventory in the same change.

The production `legacy_control` boundary disappeared in Wave 6. `renderer_bridge` remains as the
narrow renderer integration boundary; test-only characterization helpers are retired as their
actor-plus-projection replacements reach equivalent coverage.

The first Milestone 3 topology cut removed native renderer fallbacks for viewport clone, remove,
activate, rename, layout, ratio, swap, and links. Those actions now remain pending until the actor
projection confirms them, even before the first projection has been consumed. The recurring
renderer-to-actor report no longer sends a workspace snapshot: its dedicated observation payload
contains only shared-resource/cache telemetry, frame and tile-loading observations, and missing
native-layer resource descriptors. Dataset bootstrap no longer serializes renderer workspace state
into the actor: the project snapshot is installed first and the actor restores its matching ROI
view and persisted masks before producing the initial projection. Production project attachment
and renderer-side dataset switching no longer invoke the in-process semantic restorer; it remains
test-only while its persistence characterizations are converted. Mutation of the combined
per-frame viewport container stays open in the `viewer.workspace_projection` ledger row, while
heavyweight project resource materialization is tracked in the resource milestones.
Frame capture may retain an optimistic camera or slice preview for painting, but it no longer
increments actor-owned navigation/presentation revisions or copies linked navigation to sibling
viewports. Camera fit and committed plane changes always queue typed actor commands; query-only
renderer fixtures restore the active frame state without writing their observation into the
workspace projection.
Channel visibility/order controls, active-channel stepping, contrast, RGB presets, and channel
groups now follow the same command-only rule before and after the first projection. Renderer
channel values are consumed projection data; the deleted readiness branches can no longer treat
them as a startup semantic authority.
Generic native-layer activation, visibility, ordering, offsets, transforms, and property edits no
longer branch on whether the first projection has arrived. Gesture fields retain only their
preview and starting revision; the committed state is a typed actor command. Panel visibility,
left- and right-tab selection, rendering preferences, and the native scale-bar action likewise wait
for projection instead of changing actor-owned renderer fields directly. The single-view panel
settings ledger row is therefore actor-owned; the combined renderer projection/frame-state
container is the remaining Milestone 3 workspace boundary.

## Native cutover status

The remote application surface has no legacy execution route, and native project, single-viewer,
and mosaic controls emit typed actor commands at their interaction points. The final native mosaic
before/after snapshot translator has been removed. Camera drag remains a deliberately optimistic,
revision-reconciled renderer preview and submits the final camera state directly. Source guards,
typed-command decoding tests, semantic non-mutation tests, and projection no-feedback tests protect
this boundary.

Native-layer projection is now isolated in `app/actor_layer_projection.rs`. It decodes canonical
actor state directly and cannot call the deleted renderer-side `control_set_*` command emulators.
First-projection active-layer replay is also command-silent, and an explicit actor `window: null`
clears a stale renderer contrast window instead of being discarded by command-validation
compatibility logic.

Top-level `ProjectSpaceAction` values now enter a shared typed-command outbox in `ProjectSpace`
before the single-viewer, mosaic, or root project-page hosts can translate them. This includes ROI
and saved-view opens, project open/save, TIFF open, recent-project changes, selected-ROI mosaic
open, and object-preload actions; the single-view ROI selector uses the same path. The semantic
semantic variants of `ViewerRequest` and `MosaicRequest` have been deleted. `RootApp` stores the
actor runtime as a required value, local actor construction happens before optional TCP
publication, an unavailable Python/MCP listener leaves the actor operational, and failure to
construct the canonical actor aborts application construction instead of selecting a frame-driven
semantic mode. The old root direct-open and direct deep-link chains are gone; startup dataset opens
also enter the typed actor outbox. Platform/transient actions such as opening a dialog, help,
close, and back remain host requests by design.

Viewport workspace/presentation, object selection, masks, and native-layer presentation are no
longer in this ledger. Rectangle,
lasso, clear, click, and native
ID-selection interactions submit target-aware actor commands before renderer selection changes.
Click commits preserve renderer hit-testing and modifier behavior through a generation-checked
selection transaction; actor projection is the only operation that installs the committed
selection into primary or spatial-shape renderers. Native-layer activation, visibility, bulk
visibility, ordering, property transactions, offsets, and channel transforms submit directly at
their interaction points; revision-tagged drag previews are restored before actor reconciliation.
Mask CRUD, selection, undo, gesture commits,
property edits, reloads, and project-GeoJSON append/save all submit generation-checked actor
commands; gesture previews are restored or reconciled when the corresponding projection arrives.
