use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::hash::Hash;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use eframe::egui;
use glow::HasContext;
use lyon_path::Path as LyonPath;
use lyon_path::math::point as lyon_point;
use lyon_tessellation::{BuffersBuilder, FillOptions, FillTessellator, FillVertex, VertexBuffers};
use ndarray::Array2;
use rfd::FileDialog;
#[cfg(test)]
use zarrs::array::{Array, ArraySubset};

use crate::annotations::{
    AnnotationCategoryStyle, AnnotationLayerStyle, AnnotationPointsLayer, AnnotationShape,
};
use crate::app_support::memory::{
    MemoryChannelRow, PendingMemoryAction, SystemMemorySnapshot, format_bytes, memory_risk,
    refresh_system_memory_if_needed, ui_memory_channel_selector, ui_memory_overview,
    ui_pending_memory_action_dialog,
};
use crate::app_support::repaint as repaint_control;
use crate::app_support::screenshot::{
    ScreenshotRequest, ScreenshotSettings, ScreenshotWorkerHandle, ScreenshotWorkerMsg,
    next_numbered_screenshot_path,
};
use crate::app_support::settings::AutoContrastSettings;
use crate::camera::Camera;
use crate::custom::cell_thresholds::CellThresholdsPanel;
use crate::custom::roi_selector::{RoiSelectorAction, RoiSelectorPanel};
use crate::data::dataset_kind::{LocalDatasetKind, classify_local_dataset_path};
#[cfg(test)]
use crate::data::ome::retrieve_image_subset_u16;
use crate::data::ome::{ChannelInfo, Dims, OmeZarrDataset};
use crate::data::project_config::{
    ProjectChannelGroup, ProjectChannelGroupMember, ProjectLayerGroups, ProjectRoi,
};
use crate::data::remote_store::{
    S3BrowseEntry, S3BrowseListing, S3Browser, S3Store, build_http_store, build_s3_browser,
    build_s3_store, list_s3_prefix,
};
use crate::geometry::threshold_regions::{ThresholdRegionMask, extract_threshold_region_mask};
use crate::imaging::channel_max::{
    ChannelMaxLoaderHandle, ChannelMaxRequest, spawn_channel_max_loader,
};
use crate::imaging::histogram::{HistogramLoaderHandle, HistogramResponse, spawn_histogram_loader};
use crate::imaging::pinned_levels::{PinnedLevelStatus, PinnedLevels};
use crate::imaging::tiling::{
    TileCoord, choose_level_auto, levels_to_draw, tiles_needed_lvl0_rect_for_axes,
};
#[cfg(test)]
use crate::imaging::view_plane::image_subset_ranges_for_view;
use crate::imaging::view_plane::{
    ViewPlaneMode, ViewPlaneSelection, clamp_selection as clamp_view_selection,
    display_axes as display_axes_for_mode, display_downsample, local_to_world_scale,
    slice_extent_level0, supported_modes,
};
use crate::masks::resolve_masks_geojson_path_and_downsample;
use crate::masks::{MaskDisplayMode, MaskLayer, MaskRasterDisplayCache};
use crate::objects::GeoJsonSegmentationLayer;
#[cfg(test)]
use crate::objects::ObjectFilterLogic;
use crate::objects::PreloadedObjectLayer;
use crate::objects::{
    ObjectProjectDisplayState, ObjectViewportFilterCacheState, ObjectViewportFilterState,
    ObjectsLayer,
};
#[cfg(test)]
use crate::project::ProjectAnnotationLayerState;
use crate::project::groups as layer_groups;
use crate::project::{
    ProjectCameraState, ProjectChannelViewState, ProjectObjectCacheUiState, ProjectRoiViewState,
    ProjectSegmentationViewState, ProjectSpace, ProjectSpaceAction, ProjectUiState,
    ProjectViewChannelRef, ProjectViewSpec, ProjectViewportViewState, ProjectWorkspaceViewState,
};
use crate::render::labels_gl::{LabelDraw, LabelsGl, OutlinesParams};
use crate::render::labels_raw::{
    LabelTileKey, LabelTileLoaderHandle, LabelTileRequest, spawn_label_tile_loader,
};
use crate::render::points::{PointsLayer, PointsStyle};
use crate::render::points_gl::{PointsGlDrawData, PointsGlDrawParams, PointsGlRenderer};
use crate::render::threshold_preview_gl::{
    ThresholdPreviewGlDrawData, ThresholdPreviewGlDrawParams, ThresholdPreviewGlRenderer,
};
use crate::render::tiles::{
    RenderChannel, TileCache, TileKey, TileRequest, TileResponse, TileWorkerResponse,
    recommended_tile_loader_threads, spawn_tile_loader,
};
use crate::render::tiles_gl::{ChannelDraw, TileDraw, TilesGl};
use crate::render::tiles_raw::{
    RawTileKey, RawTileLoaderHandle, RawTileRequest, RawTileWorkerResponse, spawn_raw_tile_loader,
};
use crate::spatialdata::SpatialDataLayers;
use crate::spatialdata::SpatialImageLayers;
use crate::spatialdata::{SpatialDataElement, SpatialDataTransform2, discover_spatialdata};
use crate::ui::canvas_overlays;
use crate::ui::channel_notes;
use crate::ui::channels_panel::{self, ChannelListHost, ChannelSortMode};
use crate::ui::contrast;
use crate::ui::group_layers::{GroupLayersDialog, GroupLayersTarget, default_group_name};
use crate::ui::icons::{Icon, icon_button};
use crate::ui::layer_list;
use crate::ui::left_panel;
use crate::ui::right_panel;
use crate::ui::style::apply_napari_like_dark;
use crate::ui::top_bar;
use crate::viewports::{
    ViewportId, ViewportLayout, ViewportLinks, ViewportSlot, ViewportWorkspace,
};
#[cfg(test)]
use odon::deep_link::{DeepLinkChannelOrder, DeepLinkObjectFilterLogic, DeepLinkRequest};

mod actor_layer_projection;
mod actor_projection;
mod canvas;
mod construction;
mod contrast_ui;
mod datasets;
mod deep_links;
mod image_runtime;
mod layer_properties;
mod layer_runtime;
mod layers_ui;
mod lifecycle;
mod loading;
mod mask_interaction;
mod memory_ui;
mod navigation;
mod overlay_rendering;
mod project_integration;
mod project_view;
mod projects;
mod remote;
mod renderer_bridge;
mod screenshots;
mod selection;
mod thresholds;
mod tiff;
mod tile_runtime;
mod update;
mod viewport_runtime;
mod viewport_ui;
use crate::xenium::XeniumLayers;
use odon::control::{DataResourceSnapshot, LayerSnapshot};
use odon::model::{LabelZarrDataset, discover_label_names_local};

// Single-dataset viewer shell.
//
// This file owns the top-level frame lifecycle for the primary viewer: input handling,
// side-panel UI, tile/overlay worker draining, canvas rendering, and cross-cutting viewer
// state such as the active tool, layer ordering, and screenshot flow. The lower-level data,
// rendering, and overlay modules do the heavy lifting; this file coordinates when they are
// polled, invalidated, or drawn.

const RAW_TILE_CACHE_CAPACITY_TILES: usize = 2048;
const RAW_TILE_CACHE_MAX_CAPACITY_TILES: usize = 4096;
const RAW_TILE_CACHE_HEADROOM_TILES: usize = 256;
const RAW_TILE_ADAPTIVE_CHANNEL_THRESHOLD: usize = 16;
const RAW_TILE_ADAPTIVE_BRIDGE_TILES_PER_FRAME: usize = 1;
const RAW_TILE_ADAPTIVE_COARSE_TILES_PER_FRAME: usize = 1;

fn viewport_image_request_budgets(multi_view: bool, active: bool) -> (usize, usize) {
    if multi_view && !active {
        // Keep the comparison view streaming on every frame while reserving
        // more decode/upload bandwidth for the canvas under interaction.
        (128, 32)
    } else {
        (256, 64)
    }
}

fn merge_viewport_active_keys<K, I>(aggregate: &mut HashSet<K>, keys: I)
where
    K: Eq + Hash,
    I: IntoIterator<Item = K>,
{
    aggregate.extend(keys);
}

const MASK_POLYGON_CLOSE_HIT_RADIUS_SCREEN_PX: f32 = 10.0;
const MASK_POLYGON_VERTEX_HIT_RADIUS_SCREEN_PX: f32 = 8.0;
const MASK_POLYGON_EDGE_HIT_RADIUS_SCREEN_PX: f32 = 6.0;
const HISTOGRAM_NAVIGATION_DEBOUNCE: Duration = Duration::from_millis(300);
const HISTOGRAM_REQUEST_THROTTLE: Duration = Duration::from_millis(200);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum LayerId {
    Channel(usize),
    SpatialImage(u64),
    SegmentationLabels,
    SegmentationGeoJson,
    SegmentationObjects,
    Mask(u64),
    Points,
    Annotation(u64),
    SpatialShape(u64),
    SpatialPoints,
    XeniumCells,
    XeniumTranscripts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LeftTab {
    Layers,
    Project,
}

impl LeftTab {
    fn storage_key(self) -> &'static str {
        match self {
            Self::Layers => "layers",
            Self::Project => "project",
        }
    }

    fn from_storage_key(value: &str) -> Option<Self> {
        match value {
            "layers" => Some(Self::Layers),
            "project" => Some(Self::Project),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RightTab {
    Properties,
    Views,
    Analysis,
    Measurements,
    Memory,
    RoiSelector,
}

impl RightTab {
    fn storage_key(self) -> &'static str {
        match self {
            Self::Properties => "properties",
            Self::Views => "views",
            Self::Analysis => "analysis",
            Self::Measurements => "measurements",
            Self::Memory => "memory",
            Self::RoiSelector => "roi_selector",
        }
    }

    fn from_storage_key(value: &str) -> Option<Self> {
        match value {
            "properties" => Some(Self::Properties),
            "views" => Some(Self::Views),
            "analysis" => Some(Self::Analysis),
            "measurements" => Some(Self::Measurements),
            "memory" => Some(Self::Memory),
            "roi_selector" => Some(Self::RoiSelector),
            _ => None,
        }
    }
}

type LayerGroup = layer_list::LayerGroup;
type LayerDragState = layer_list::LayerDragState<LayerId>;

impl ChannelListHost for OmeZarrViewerApp {
    type LayerId = LayerId;

    fn channel_search(&self) -> &str {
        &self.channel_list_search
    }

    fn channel_search_mut(&mut self) -> &mut String {
        &mut self.channel_list_search
    }

    fn channel_sort_mode(&self) -> ChannelSortMode {
        self.channel_sort_mode
    }

    fn set_channel_sort_mode(&mut self, mode: ChannelSortMode) {
        if let Some((viewport_id, revision)) = self.active_viewport_command_scope() {
            self.submit_native_viewport_intent(
                "viewer.viewports.channels.set_order",
                serde_json::json!({
                    "viewport_id":viewport_id,
                    "if_presentation_revision":revision,
                    "sort":mode.storage_key(),
                }),
            );
        }
    }

    fn channel_count(&self) -> usize {
        self.channels.len()
    }

    fn channel_order(&self) -> &[usize] {
        &self.channel_layer_order
    }

    fn channel_name(&self, idx: usize) -> Option<String> {
        self.channels.get(idx).map(|ch| ch.name.clone())
    }

    fn channel_visible(&self, idx: usize) -> Option<bool> {
        self.channels.get(idx).map(|ch| ch.visible)
    }

    fn set_channel_visible(&mut self, idx: usize, visible: bool) {
        if let Some((viewport_id, revision)) = self.active_viewport_command_scope() {
            self.submit_native_viewport_intent(
                "viewer.viewports.channels.set_visible",
                serde_json::json!({
                    "viewport_id":viewport_id,
                    "if_presentation_revision":revision,
                    "channels":[idx],
                    "mode":if visible { "show" } else { "hide" },
                }),
            );
        }
    }

    fn set_channels_visible(&mut self, indices: &[usize], visible: bool) {
        if let Some((viewport_id, revision)) = self.active_viewport_command_scope() {
            self.submit_native_viewport_intent(
                "viewer.viewports.channels.set_visible",
                serde_json::json!({
                    "viewport_id":viewport_id,
                    "if_presentation_revision":revision,
                    "channels":indices,
                    "mode":if visible { "show" } else { "hide" },
                }),
            );
        }
    }

    fn channel_available(&self, idx: usize) -> bool {
        self.layer_is_available(LayerId::Channel(idx))
    }

    fn is_channel_selected(&self, idx: usize) -> bool {
        self.active_layer == LayerId::Channel(idx) || self.selected_channel_layers.contains(&idx)
    }

    fn selected_channel_group_id(&self) -> Option<u64> {
        self.selected_channel_group_id
    }

    fn select_channel_group(&mut self, group_id: Option<u64>) {
        self.selected_channel_group_id = group_id;
        self.selected_channel_layers.clear();
        if let Some(gid) = group_id {
            if let Some(idx) = self.channel_indices_in_group(gid).into_iter().next() {
                self.commit_active_layer(LayerId::Channel(idx));
            }
        }
    }

    fn handle_channel_primary_click(
        &mut self,
        idx: usize,
        visible_indices: &[usize],
        modifiers: egui::Modifiers,
    ) {
        if modifiers.shift && self.channel_select_anchor_idx.is_some() {
            let anchor_idx = self.channel_select_anchor_idx.unwrap_or(idx);
            let anchor_pos = visible_indices.iter().position(|&idx2| idx2 == anchor_idx);
            let current_pos = visible_indices.iter().position(|&idx2| idx2 == idx);
            if let (Some(anchor_pos), Some(current_pos)) = (anchor_pos, current_pos) {
                let (a, b) = if anchor_pos <= current_pos {
                    (anchor_pos, current_pos)
                } else {
                    (current_pos, anchor_pos)
                };
                self.selected_channel_layers.clear();
                for idx2 in &visible_indices[a..=b] {
                    self.selected_channel_layers.insert(*idx2);
                }
            } else {
                self.selected_channel_layers.clear();
                self.selected_channel_layers.insert(idx);
            }
        } else if modifiers.command {
            if !self.selected_channel_layers.insert(idx) {
                self.selected_channel_layers.remove(&idx);
            }
            self.channel_select_anchor_idx = Some(idx);
            self.selected_channel_group_id = None;
        } else {
            self.selected_channel_layers.clear();
            self.selected_channel_layers.insert(idx);
            self.channel_select_anchor_idx = Some(idx);
            self.selected_channel_group_id = None;
        }
        self.commit_active_layer(LayerId::Channel(idx));
    }

    fn handle_channel_secondary_click(&mut self, idx: usize) {
        if !self.selected_channel_layers.contains(&idx) {
            self.selected_channel_layers.clear();
            self.selected_channel_layers.insert(idx);
            self.channel_select_anchor_idx = Some(idx);
            self.selected_channel_group_id = None;
            self.commit_active_layer(LayerId::Channel(idx));
        }
    }

    fn open_group_layers_dialog_channels(&mut self, members: Vec<usize>) {
        Self::open_group_layers_dialog_channels(self, members);
    }

    fn can_reset_selected_layer_positions(&mut self) -> bool {
        self.current_visible_move_targets_have_moved()
    }

    fn reset_selected_layer_positions(&mut self) -> bool {
        self.reset_current_visible_move_targets_to_loaded()
    }

    fn can_apply_rgb_preset(&self) -> bool {
        self.channels.len() == 3
    }

    fn apply_rgb_preset(&mut self) -> bool {
        self.apply_three_channel_rgb_preset()
    }

    fn layer_groups(&self) -> crate::data::project_config::ProjectLayerGroups {
        self.current_layer_groups()
    }

    fn set_layer_groups(&mut self, groups: crate::data::project_config::ProjectLayerGroups) {
        self.commit_current_channel_groups(groups);
    }

    fn channels_changed(&mut self) {
        self.bump_render_id();
    }

    fn layer_drag_mut(&mut self) -> &mut Option<LayerDragState> {
        &mut self.layer_drag
    }

    fn dragging_channel_idx(&self) -> Option<usize> {
        self.layer_drag.as_ref().and_then(|drag| {
            if drag.group != LayerGroup::Channels {
                return None;
            }
            match drag.dragged {
                LayerId::Channel(idx) => Some(idx),
                _ => None,
            }
        })
    }

    fn channel_layer_id(&self, idx: usize) -> Self::LayerId {
        LayerId::Channel(idx)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolMode {
    Select,
    Pan,
    MoveLayer,
    TransformLayer,
    DrawMaskPolygon,
    LassoSelect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MaskPolygonSelection {
    layer_id: u64,
    polygon_idx: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct MaskVertexDrag {
    selection: MaskPolygonSelection,
    vertex_idx: usize,
    undo_recorded: bool,
    start_polygon: Vec<egui::Pos2>,
    start_selection: Option<MaskPolygonSelection>,
    start_selected_vertex: Option<usize>,
    actor_generation: u64,
}

#[derive(Debug, Clone)]
struct MaskPolygonMoveState {
    selection: MaskPolygonSelection,
    start_polygon: Vec<egui::Pos2>,
    start_pointer_world: egui::Pos2,
    start_selection: Option<MaskPolygonSelection>,
    start_selected_vertex: Option<usize>,
    actor_generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MaskPolygonHit {
    polygon_idx: usize,
    vertex_idx: Option<usize>,
}

#[derive(Debug, Default, Clone)]
struct MaskDrawDebugStats {
    visible_layers: usize,
    painted_polygons: usize,
    painted_vertices: usize,
    screen_polygons: usize,
    screen_vertices: usize,
    fill_polygons: usize,
    fill_vertices: usize,
    raster_layers: usize,
    raster_pixels: usize,
    draw_time: Duration,
}

#[derive(Debug, Clone)]
struct LayerOffsetEntry {
    layer: LayerId,
    offset_world: egui::Vec2,
}

fn distance_to_screen_segment(p: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> f32 {
    let ab = b - a;
    let len_sq = ab.length_sq();
    if len_sq <= f32::EPSILON {
        return p.distance(a);
    }
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

fn point_in_mask_polygon(p: egui::Pos2, poly: &[egui::Pos2]) -> bool {
    let n = OmeZarrViewerApp::mask_polygon_unique_vertex_count(poly);
    if n < 3 {
        return false;
    }

    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let pi = poly[i];
        let pj = poly[j];
        if (pi.y > p.y) != (pj.y > p.y) {
            let x_intersection = (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y) + pi.x;
            if p.x < x_intersection {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TilePrefetchMode {
    Off,
    TargetHalo,
    TargetAndFinerHalo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TilePrefetchAggressiveness {
    Conservative,
    Balanced,
    Aggressive,
}

#[derive(Debug, Clone)]
struct HoverTooltipState {
    signature: String,
    lines: Vec<String>,
    first_seen: Instant,
    last_seen: Instant,
    visible: bool,
}

struct ThresholdRegionPreview {
    generation: u64,
    channel_index: usize,
    channel_name: String,
    scope: ThresholdRegionScope,
    level_index: usize,
    downsample: f32,
    x0: u64,
    y0: u64,
    plane: Array2<u16>,
    raw_values: Arc<Vec<u16>>,
    threshold: u16,
    min_component_pixels: usize,
    mask: ThresholdRegionMask,
    texture: Option<egui::TextureHandle>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThresholdRegionScope {
    VisibleRegion,
    EntireImage,
}

impl ThresholdRegionScope {
    fn label(self) -> &'static str {
        match self {
            Self::VisibleRegion => "visible region",
            Self::EntireImage => "entire image",
        }
    }
}

const THRESHOLD_REGION_MAX_INTERACTIVE_PIXELS: u64 = 10_000_000;

fn threshold_level_size(
    level: &crate::data::ome::LevelInfo,
    y_dim: usize,
    x_dim: usize,
) -> Option<(u64, u64)> {
    let height = level.shape.get(y_dim).copied()?;
    let width = level.shape.get(x_dim).copied()?;
    Some((width, height))
}

fn threshold_region_pixel_count(width: u64, height: u64) -> Option<u64> {
    width.checked_mul(height)
}

fn default_threshold_full_level(
    levels: &[crate::data::ome::LevelInfo],
    y_dim: usize,
    x_dim: usize,
    max_pixels: u64,
) -> Option<usize> {
    levels.iter().find_map(|level| {
        let (width, height) = threshold_level_size(level, y_dim, x_dim)?;
        let pixels = threshold_region_pixel_count(width, height)?;
        (pixels <= max_pixels).then_some(level.index)
    })
}

// Grouping dialog state lives in `ui_group_layers`.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RemoteMode {
    Http,
    S3,
}

struct RemoteS3BrowserState {
    session: S3Browser,
    signature: String,
    current_prefix: String,
    parent_prefix: Option<String>,
    entries: Vec<S3BrowseEntry>,
    current_is_dataset: bool,
    selected_dataset_prefixes: HashSet<String>,
    listing_cache: HashMap<String, S3BrowseListing>,
}

#[derive(Debug, Clone)]
pub struct S3DatasetSelection {
    pub endpoint: String,
    pub region: String,
    pub bucket: String,
    pub prefix: String,
    pub access_key: String,
    pub secret_key: String,
}

#[derive(Debug, Clone)]
struct ObjectLayerViewportPresentation {
    display: ObjectProjectDisplayState,
    filter: ObjectViewportFilterState,
    filter_cache: ObjectViewportFilterCacheState,
    visible: bool,
    opacity: f32,
    width_screen_px: f32,
    color_rgb: [u8; 3],
    show_selection_overlay: bool,
}

impl ObjectLayerViewportPresentation {
    fn capture(layer: &ObjectsLayer) -> Self {
        Self {
            display: layer.project_display_state(),
            filter: layer.viewport_filter_state(),
            filter_cache: layer.viewport_filter_cache_state(),
            visible: layer.visible,
            opacity: layer.opacity,
            width_screen_px: layer.width_screen_px,
            color_rgb: layer.color_rgb,
            show_selection_overlay: layer.show_selection_overlay,
        }
    }

    fn apply(&self, layer: &mut ObjectsLayer) {
        layer.apply_project_display_state(&self.display);
        layer.apply_viewport_filter_state(&self.filter);
        layer.apply_viewport_filter_cache_state(&self.filter_cache);
        layer.visible = self.visible;
        layer.opacity = self.opacity;
        layer.width_screen_px = self.width_screen_px;
        layer.color_rgb = self.color_rgb;
        layer.show_selection_overlay = self.show_selection_overlay;
    }
}

#[derive(Debug, Clone)]
struct MaskViewportPresentation {
    id: u64,
    visible: bool,
    opacity: f32,
    width_screen_px: f32,
    display_mode: MaskDisplayMode,
    color_rgb: [u8; 3],
}

#[derive(Debug, Clone)]
struct AnnotationViewportPresentation {
    id: u64,
    visible: bool,
    style: AnnotationLayerStyle,
    category_styles: Vec<AnnotationCategoryStyle>,
    continuous_shape: AnnotationShape,
    continuous_range: Option<(f32, f32)>,
}

#[derive(Debug, Clone)]
struct SpatialShapeViewportPresentation {
    id: u64,
    visible: bool,
    opacity: f32,
    width_screen_px: f32,
    color_rgb: [u8; 3],
    objects: Option<ObjectLayerViewportPresentation>,
}

#[derive(Debug, Clone)]
struct SpatialImageViewportPresentation {
    id: u64,
    visible: bool,
    opacity: f32,
    current_z_level0: u64,
    channels: Vec<ChannelInfo>,
}

#[derive(Debug, Clone)]
struct PendingViewportScreenshot {
    viewport_id: ViewportId,
    request: ScreenshotRequest,
}

#[derive(Debug, Clone)]
struct ViewportRenderState {
    last_canvas_rect: Option<egui::Rect>,
    active_render_id: u64,
    previous_render_id: Option<u64>,
    active_render_smooth_pixels: bool,
    previous_render_smooth_pixels: Option<bool>,
    previous_view_selection: Option<ViewPlaneSelection>,
    previous_displayed_view_selection: Option<ViewPlaneSelection>,
    last_render_view_selection: ViewPlaneSelection,
    last_target_level: Option<usize>,
    fallback_ceiling_level: Option<usize>,
    last_visible_world_tiles: Option<egui::Rect>,
    zoom_out_floor_level: Option<usize>,
    zoom_out_floor_until: Option<Instant>,
    zoom_out_floor_visible_world_tiles: Option<egui::Rect>,
}

#[derive(Debug, Clone)]
struct ViewportTransientState {
    draft_view_slice_level0: Option<u64>,
    selected_channel_layers: HashSet<usize>,
    selected_channel_group_id: Option<u64>,
    selected_overlay_layers: HashSet<LayerId>,
    object_filter_cache: ObjectViewportFilterCacheState,
}

#[derive(Debug, Clone)]
struct ViewerViewportState {
    camera: Camera,
    render: ViewportRenderState,
    transient: ViewportTransientState,
    selected_channel: usize,
    view_plane_mode: ViewPlaneMode,
    current_x_level0: u64,
    current_y_level0: u64,
    current_z_level0: u64,
    channels: Vec<ChannelInfo>,
    layer_groups: ProjectLayerGroups,
    channel_window_overrides: HashMap<String, (f32, f32)>,
    active_layer: LayerId,
    overlay_layer_order: Vec<LayerId>,
    channel_layer_order: Vec<usize>,
    channel_sort_mode: ChannelSortMode,
    object_display: ObjectProjectDisplayState,
    object_filter: ObjectViewportFilterState,
    object_visible: bool,
    object_opacity: f32,
    object_width_screen_px: f32,
    object_color_rgb: [u8; 3],
    object_show_selection_overlay: bool,
    cells_outlines_visible: bool,
    cells_outlines_color_rgb: [u8; 3],
    cells_outlines_opacity: f32,
    cells_outlines_width_px: f32,
    cell_points_visible: bool,
    cell_points_style: PointsStyle,
    masks: Vec<MaskViewportPresentation>,
    annotations: Vec<AnnotationViewportPresentation>,
    seg_geojson_visible: bool,
    seg_geojson_opacity: f32,
    seg_geojson_width_screen_px: f32,
    seg_geojson_color_rgb: [u8; 3],
    spatial_shapes: Vec<SpatialShapeViewportPresentation>,
    spatial_points: Option<(bool, PointsStyle, f32, usize)>,
    spatial_images: Vec<SpatialImageViewportPresentation>,
    xenium_cells: Option<(bool, f32, f32, [u8; 3])>,
    xenium_transcripts: Option<(bool, PointsStyle, String, usize)>,
    smooth_pixels: bool,
    show_scale_bar: bool,
    show_hud: bool,
    show_tile_debug: bool,
}

impl ViewerViewportState {
    /// Capture only state that the renderer is allowed to advance between actor projections.
    /// Camera is included as the explicit optimistic navigation preview; its actor revision and
    /// linked-camera semantics remain actor-owned.
    fn capture_runtime(&mut self, app: &OmeZarrViewerApp) {
        self.camera = app.camera.clone();
        self.render = ViewportRenderState {
            last_canvas_rect: app.last_canvas_rect,
            active_render_id: app.active_render_id,
            previous_render_id: app.previous_render_id,
            active_render_smooth_pixels: app.active_render_smooth_pixels,
            previous_render_smooth_pixels: app.previous_render_smooth_pixels,
            previous_view_selection: app.previous_view_selection,
            previous_displayed_view_selection: app.previous_displayed_view_selection,
            last_render_view_selection: app.last_render_view_selection,
            last_target_level: app.last_target_level,
            fallback_ceiling_level: app.fallback_ceiling_level,
            last_visible_world_tiles: app.last_visible_world_tiles,
            zoom_out_floor_level: app.zoom_out_floor_level,
            zoom_out_floor_until: app.zoom_out_floor_until,
            zoom_out_floor_visible_world_tiles: app.zoom_out_floor_visible_world_tiles,
        };
        self.transient = ViewportTransientState {
            draft_view_slice_level0: app.draft_view_slice_level0,
            selected_channel_layers: app.selected_channel_layers.clone(),
            selected_channel_group_id: app.selected_channel_group_id,
            selected_overlay_layers: app.selected_overlay_layers.clone(),
            object_filter_cache: app.seg_objects.viewport_filter_cache_state(),
        };
    }

    fn color_json(color: egui::Color32) -> serde_json::Value {
        serde_json::json!(color.to_array())
    }

    fn color_from_json(value: &serde_json::Value) -> Option<egui::Color32> {
        let values = value.as_array()?;
        if values.len() != 4 {
            return None;
        }
        let mut rgba = [0_u8; 4];
        for (dst, value) in rgba.iter_mut().zip(values) {
            *dst = u8::try_from(value.as_u64()?).ok()?;
        }
        Some(egui::Color32::from_rgba_unmultiplied(
            rgba[0], rgba[1], rgba[2], rgba[3],
        ))
    }

    fn rgb_from_json(value: &serde_json::Value) -> Option<[u8; 3]> {
        let values = value.as_array()?;
        if values.len() != 3 {
            return None;
        }
        Some([
            u8::try_from(values[0].as_u64()?).ok()?,
            u8::try_from(values[1].as_u64()?).ok()?,
            u8::try_from(values[2].as_u64()?).ok()?,
        ])
    }

    fn points_style_json(style: &PointsStyle) -> serde_json::Value {
        serde_json::json!({
            "radius_screen_px": style.radius_screen_px,
            "fill_positive_rgba": Self::color_json(style.fill_positive),
            "fill_negative_rgba": Self::color_json(style.fill_negative),
            "stroke_positive": {
                "width": style.stroke_positive.width,
                "color_rgba": Self::color_json(style.stroke_positive.color),
            },
            "stroke_negative": {
                "width": style.stroke_negative.width,
                "color_rgba": Self::color_json(style.stroke_negative.color),
            },
        })
    }

    fn apply_points_style_json(style: &mut PointsStyle, value: &serde_json::Value) {
        if let Some(radius) = value
            .get("radius_screen_px")
            .and_then(serde_json::Value::as_f64)
        {
            style.radius_screen_px = radius as f32;
        }
        if let Some(color) = value
            .get("fill_positive_rgba")
            .and_then(Self::color_from_json)
        {
            style.fill_positive = color;
        }
        if let Some(color) = value
            .get("fill_negative_rgba")
            .and_then(Self::color_from_json)
        {
            style.fill_negative = color;
        }
        for (key, stroke) in [
            ("stroke_positive", &mut style.stroke_positive),
            ("stroke_negative", &mut style.stroke_negative),
        ] {
            let Some(saved) = value.get(key) else {
                continue;
            };
            if let Some(width) = saved.get("width").and_then(serde_json::Value::as_f64) {
                stroke.width = width as f32;
            }
            if let Some(color) = saved.get("color_rgba").and_then(Self::color_from_json) {
                stroke.color = color;
            }
        }
    }

    fn object_layer_presentation_json(
        presentation: &ObjectLayerViewportPresentation,
    ) -> serde_json::Value {
        serde_json::json!({
            "display": presentation.display,
            "filter": presentation.filter.project_json(),
            "visible": presentation.visible,
            "opacity": presentation.opacity,
            "width_screen_px": presentation.width_screen_px,
            "color_rgb": presentation.color_rgb,
            "show_selection_overlay": presentation.show_selection_overlay,
        })
    }

    fn apply_object_layer_presentation_json(
        presentation: &mut ObjectLayerViewportPresentation,
        value: &serde_json::Value,
    ) -> Result<(), String> {
        if let Some(display) = value.get("display") {
            presentation.display = serde_json::from_value(display.clone())
                .map_err(|error| format!("invalid object display presentation: {error}"))?;
        }
        if let Some(property) = value.get("color_property") {
            presentation.display.color_property_key = property
                .as_str()
                .map(str::trim)
                .filter(|property| !property.is_empty())
                .map(str::to_string);
        }
        if let Some(overrides) = value.get("color_level_overrides") {
            presentation.display.color_level_overrides = serde_json::from_value(overrides.clone())
                .map_err(|error| format!("invalid object legend presentation: {error}"))?;
        }
        if let Some(fill) = value.get("fill_cells").and_then(serde_json::Value::as_bool) {
            presentation.display.fill_cells = fill;
        }
        if let Some(opacity) = value
            .get("fill_opacity")
            .and_then(serde_json::Value::as_f64)
        {
            presentation.display.fill_opacity = (opacity as f32).clamp(0.0, 1.0);
        }
        if let Some(opacity) = value
            .get("selected_fill_opacity")
            .and_then(serde_json::Value::as_f64)
        {
            presentation.display.selected_fill_opacity = (opacity as f32).clamp(0.0, 1.0);
        }
        if let Some(fast) = value
            .get("fast_rendering")
            .and_then(serde_json::Value::as_bool)
        {
            presentation.display.fast_rendering = fast;
        }
        if let Some(filter) = value.get("filter") {
            presentation.filter = ObjectViewportFilterState::from_project_json(filter)?;
            presentation.filter_cache = ObjectViewportFilterCacheState::empty();
        }
        if let Some(visible) = value.get("visible").and_then(serde_json::Value::as_bool) {
            presentation.visible = visible;
        }
        if let Some(opacity) = value.get("opacity").and_then(serde_json::Value::as_f64) {
            presentation.opacity = (opacity as f32).clamp(0.0, 1.0);
        }
        if let Some(width) = value
            .get("width_screen_px")
            .and_then(serde_json::Value::as_f64)
        {
            presentation.width_screen_px = (width as f32).max(0.0);
        }
        if let Some(color) = value.get("color_rgb").and_then(Self::rgb_from_json) {
            presentation.color_rgb = color;
        }
        if let Some(show) = value
            .get("show_selection_overlay")
            .and_then(serde_json::Value::as_bool)
        {
            presentation.show_selection_overlay = show;
        }
        Ok(())
    }

    fn project_presentation_json(&self) -> serde_json::Value {
        let masks = self
            .masks
            .iter()
            .map(|layer| {
                serde_json::json!({
                    "id": layer.id,
                    "visible": layer.visible,
                    "opacity": layer.opacity,
                    "width_screen_px": layer.width_screen_px,
                    "display_mode": layer.display_mode.storage_key(),
                    "color_rgb": layer.color_rgb,
                })
            })
            .collect::<Vec<_>>();
        let annotations = self
            .annotations
            .iter()
            .map(|layer| {
                serde_json::json!({
                    "id": layer.id,
                    "visible": layer.visible,
                    "style": {
                        "radius_screen_px": layer.style.radius_screen_px,
                        "opacity": layer.style.opacity,
                        "stroke_width": layer.style.stroke.width,
                        "stroke_color_rgba": Self::color_json(layer.style.stroke.color),
                    },
                    "category_styles": layer.category_styles.iter().map(|category| serde_json::json!({
                        "name": category.name,
                        "visible": category.visible,
                        "color_rgba": Self::color_json(category.color),
                        "shape": category.shape.storage_key(),
                    })).collect::<Vec<_>>(),
                    "continuous_shape": layer.continuous_shape.storage_key(),
                    "continuous_range": layer.continuous_range.map(|(lo, hi)| [lo, hi]),
                })
            })
            .collect::<Vec<_>>();
        let spatial_shapes = self
            .spatial_shapes
            .iter()
            .map(|layer| {
                serde_json::json!({
                    "id": layer.id,
                    "visible": layer.visible,
                    "opacity": layer.opacity,
                    "width_screen_px": layer.width_screen_px,
                    "color_rgb": layer.color_rgb,
                    "objects": layer.objects.as_ref().map(Self::object_layer_presentation_json),
                })
            })
            .collect::<Vec<_>>();
        let spatial_images = self
            .spatial_images
            .iter()
            .map(|layer| {
                serde_json::json!({
                    "id": layer.id,
                    "visible": layer.visible,
                    "opacity": layer.opacity,
                    "current_z_level0": layer.current_z_level0,
                    "channels": layer.channels.iter().map(|channel| serde_json::json!({
                        "index": channel.index,
                        "name": channel.name,
                        "visible": channel.visible,
                        "color_rgb": channel.color_rgb,
                        "window": channel.window.map(|(lo, hi)| [lo, hi]),
                    })).collect::<Vec<_>>(),
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "cell_points": {
                "visible": self.cell_points_visible,
                "style": Self::points_style_json(&self.cell_points_style),
            },
            "masks": masks,
            "annotations": annotations,
            "segmentation_geojson": {
                "visible": self.seg_geojson_visible,
                "opacity": self.seg_geojson_opacity,
                "width_screen_px": self.seg_geojson_width_screen_px,
                "color_rgb": self.seg_geojson_color_rgb,
            },
            "spatial_shapes": spatial_shapes,
            "spatial_points": self.spatial_points.as_ref().map(|(visible, style, threshold, max_points)| serde_json::json!({
                "visible": visible,
                "style": Self::points_style_json(style),
                "threshold": threshold,
                "max_render_points_total": max_points,
            })),
            "spatial_images": spatial_images,
            "xenium_cells": self.xenium_cells.map(|(visible, opacity, width, color)| serde_json::json!({
                "visible": visible,
                "opacity": opacity,
                "width_screen_px": width,
                "color_rgb": color,
            })),
            "xenium_transcripts": self.xenium_transcripts.as_ref().map(|(visible, style, gene_query, max_points)| serde_json::json!({
                "visible": visible,
                "style": Self::points_style_json(style),
                "gene_query": gene_query,
                "max_render_points_total": max_points,
            })),
            "display_preferences": {
                "smooth_pixels": self.smooth_pixels,
                "show_scale_bar": self.show_scale_bar,
                "show_hud": self.show_hud,
                "show_tile_debug": self.show_tile_debug,
            },
        })
    }

    fn apply_project_presentation_json(&mut self, value: &serde_json::Value) -> Result<(), String> {
        if let Some(saved) = value.get("display_preferences") {
            if let Some(value) = saved
                .get("smooth_pixels")
                .and_then(serde_json::Value::as_bool)
            {
                self.smooth_pixels = value;
            }
            if let Some(value) = saved
                .get("show_scale_bar")
                .and_then(serde_json::Value::as_bool)
            {
                self.show_scale_bar = value;
            }
            if let Some(value) = saved.get("show_hud").and_then(serde_json::Value::as_bool) {
                self.show_hud = value;
            }
            if let Some(value) = saved
                .get("show_tile_debug")
                .and_then(serde_json::Value::as_bool)
            {
                self.show_tile_debug = value;
            }
        }
        if let Some(points) = value.get("cell_points") {
            if let Some(visible) = points.get("visible").and_then(serde_json::Value::as_bool) {
                self.cell_points_visible = visible;
            }
            if let Some(style) = points.get("style") {
                Self::apply_points_style_json(&mut self.cell_points_style, style);
            }
        }
        if let Some(saved_layers) = value.get("masks").and_then(serde_json::Value::as_array) {
            for saved in saved_layers {
                let Some(id) = saved.get("id").and_then(serde_json::Value::as_u64) else {
                    continue;
                };
                let Some(layer) = self.masks.iter_mut().find(|layer| layer.id == id) else {
                    continue;
                };
                if let Some(visible) = saved.get("visible").and_then(serde_json::Value::as_bool) {
                    layer.visible = visible;
                }
                if let Some(opacity) = saved.get("opacity").and_then(serde_json::Value::as_f64) {
                    layer.opacity = (opacity as f32).clamp(0.0, 1.0);
                }
                if let Some(width) = saved
                    .get("width_screen_px")
                    .and_then(serde_json::Value::as_f64)
                {
                    layer.width_screen_px = (width as f32).max(0.0);
                }
                if let Some(mode) = saved
                    .get("display_mode")
                    .and_then(serde_json::Value::as_str)
                    .and_then(MaskDisplayMode::from_storage_key)
                {
                    layer.display_mode = mode;
                }
                if let Some(color) = saved.get("color_rgb").and_then(Self::rgb_from_json) {
                    layer.color_rgb = color;
                }
            }
        }
        if let Some(saved_layers) = value
            .get("annotations")
            .and_then(serde_json::Value::as_array)
        {
            for saved in saved_layers {
                let Some(id) = saved.get("id").and_then(serde_json::Value::as_u64) else {
                    continue;
                };
                let Some(layer) = self.annotations.iter_mut().find(|layer| layer.id == id) else {
                    continue;
                };
                if let Some(visible) = saved.get("visible").and_then(serde_json::Value::as_bool) {
                    layer.visible = visible;
                }
                if let Some(style) = saved.get("style") {
                    if let Some(radius) = style
                        .get("radius_screen_px")
                        .and_then(serde_json::Value::as_f64)
                    {
                        layer.style.radius_screen_px = radius as f32;
                    }
                    if let Some(opacity) = style.get("opacity").and_then(serde_json::Value::as_f64)
                    {
                        layer.style.opacity = (opacity as f32).clamp(0.0, 1.0);
                    }
                    if let Some(width) = style
                        .get("stroke_width")
                        .and_then(serde_json::Value::as_f64)
                    {
                        layer.style.stroke.width = (width as f32).max(0.0);
                    }
                    if let Some(color) = style
                        .get("stroke_color_rgba")
                        .and_then(Self::color_from_json)
                    {
                        layer.style.stroke.color = color;
                    }
                }
                if let Some(categories) = saved
                    .get("category_styles")
                    .and_then(serde_json::Value::as_array)
                {
                    for saved_category in categories {
                        let Some(name) = saved_category
                            .get("name")
                            .and_then(serde_json::Value::as_str)
                        else {
                            continue;
                        };
                        let Some(category) = layer
                            .category_styles
                            .iter_mut()
                            .find(|category| category.name == name)
                        else {
                            continue;
                        };
                        if let Some(visible) = saved_category
                            .get("visible")
                            .and_then(serde_json::Value::as_bool)
                        {
                            category.visible = visible;
                        }
                        if let Some(color) = saved_category
                            .get("color_rgba")
                            .and_then(Self::color_from_json)
                        {
                            category.color = color;
                        }
                        if let Some(shape) = saved_category
                            .get("shape")
                            .and_then(serde_json::Value::as_str)
                            .and_then(AnnotationShape::from_storage_key)
                        {
                            category.shape = shape;
                        }
                    }
                }
                if let Some(shape) = saved
                    .get("continuous_shape")
                    .and_then(serde_json::Value::as_str)
                    .and_then(AnnotationShape::from_storage_key)
                {
                    layer.continuous_shape = shape;
                }
                if let Some(range) = saved
                    .get("continuous_range")
                    .and_then(serde_json::Value::as_array)
                    .filter(|range| range.len() == 2)
                    && let (Some(lo), Some(hi)) = (range[0].as_f64(), range[1].as_f64())
                {
                    layer.continuous_range = Some((lo as f32, hi as f32));
                }
            }
        }
        if let Some(saved) = value.get("segmentation_geojson") {
            if let Some(visible) = saved.get("visible").and_then(serde_json::Value::as_bool) {
                self.seg_geojson_visible = visible;
            }
            if let Some(opacity) = saved.get("opacity").and_then(serde_json::Value::as_f64) {
                self.seg_geojson_opacity = (opacity as f32).clamp(0.0, 1.0);
            }
            if let Some(width) = saved
                .get("width_screen_px")
                .and_then(serde_json::Value::as_f64)
            {
                self.seg_geojson_width_screen_px = (width as f32).max(0.0);
            }
            if let Some(color) = saved.get("color_rgb").and_then(Self::rgb_from_json) {
                self.seg_geojson_color_rgb = color;
            }
        }
        if let Some(saved_layers) = value
            .get("spatial_shapes")
            .and_then(serde_json::Value::as_array)
        {
            for saved in saved_layers {
                let Some(id) = saved.get("id").and_then(serde_json::Value::as_u64) else {
                    continue;
                };
                let Some(layer) = self.spatial_shapes.iter_mut().find(|layer| layer.id == id)
                else {
                    continue;
                };
                if let Some(visible) = saved.get("visible").and_then(serde_json::Value::as_bool) {
                    layer.visible = visible;
                }
                if let Some(opacity) = saved.get("opacity").and_then(serde_json::Value::as_f64) {
                    layer.opacity = (opacity as f32).clamp(0.0, 1.0);
                }
                if let Some(width) = saved
                    .get("width_screen_px")
                    .and_then(serde_json::Value::as_f64)
                {
                    layer.width_screen_px = (width as f32).max(0.0);
                }
                if let Some(color) = saved.get("color_rgb").and_then(Self::rgb_from_json) {
                    layer.color_rgb = color;
                }
                if let (Some(objects), Some(saved_objects)) =
                    (layer.objects.as_mut(), saved.get("objects"))
                {
                    Self::apply_object_layer_presentation_json(objects, saved_objects)?;
                }
            }
        }
        if let (Some(saved), Some((visible, style, threshold, max_points))) =
            (value.get("spatial_points"), self.spatial_points.as_mut())
        {
            if let Some(value) = saved.get("visible").and_then(serde_json::Value::as_bool) {
                *visible = value;
            }
            if let Some(saved_style) = saved.get("style") {
                Self::apply_points_style_json(style, saved_style);
            }
            if let Some(value) = saved.get("threshold").and_then(serde_json::Value::as_f64) {
                *threshold = value as f32;
            }
            if let Some(value) = saved
                .get("max_render_points_total")
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
            {
                *max_points = value;
            }
        }
        if let Some(saved_layers) = value
            .get("spatial_images")
            .and_then(serde_json::Value::as_array)
        {
            for saved in saved_layers {
                let Some(id) = saved.get("id").and_then(serde_json::Value::as_u64) else {
                    continue;
                };
                let Some(layer) = self.spatial_images.iter_mut().find(|layer| layer.id == id)
                else {
                    continue;
                };
                if let Some(visible) = saved.get("visible").and_then(serde_json::Value::as_bool) {
                    layer.visible = visible;
                }
                if let Some(opacity) = saved.get("opacity").and_then(serde_json::Value::as_f64) {
                    layer.opacity = (opacity as f32).clamp(0.0, 1.0);
                }
                if let Some(z) = saved
                    .get("current_z_level0")
                    .and_then(serde_json::Value::as_u64)
                {
                    layer.current_z_level0 = z;
                }
                if let Some(channels) = saved.get("channels").and_then(serde_json::Value::as_array)
                {
                    for saved_channel in channels {
                        let Some(index) = saved_channel
                            .get("index")
                            .and_then(serde_json::Value::as_u64)
                            .and_then(|index| usize::try_from(index).ok())
                        else {
                            continue;
                        };
                        let Some(channel) = layer
                            .channels
                            .iter_mut()
                            .find(|channel| channel.index == index)
                        else {
                            continue;
                        };
                        if let Some(visible) = saved_channel
                            .get("visible")
                            .and_then(serde_json::Value::as_bool)
                        {
                            channel.visible = visible;
                        }
                        if let Some(color) =
                            saved_channel.get("color_rgb").and_then(Self::rgb_from_json)
                        {
                            channel.color_rgb = color;
                        }
                        if let Some(window) = saved_channel
                            .get("window")
                            .and_then(serde_json::Value::as_array)
                            .filter(|window| window.len() == 2)
                            && let (Some(lo), Some(hi)) = (window[0].as_f64(), window[1].as_f64())
                        {
                            channel.window = Some((lo as f32, hi as f32));
                        }
                    }
                }
            }
        }
        if let (Some(saved), Some((visible, opacity, width, color))) =
            (value.get("xenium_cells"), self.xenium_cells.as_mut())
        {
            if let Some(value) = saved.get("visible").and_then(serde_json::Value::as_bool) {
                *visible = value;
            }
            if let Some(value) = saved.get("opacity").and_then(serde_json::Value::as_f64) {
                *opacity = (value as f32).clamp(0.0, 1.0);
            }
            if let Some(value) = saved
                .get("width_screen_px")
                .and_then(serde_json::Value::as_f64)
            {
                *width = (value as f32).max(0.0);
            }
            if let Some(value) = saved.get("color_rgb").and_then(Self::rgb_from_json) {
                *color = value;
            }
        }
        if let (Some(saved), Some((visible, style, gene_query, max_points))) = (
            value.get("xenium_transcripts"),
            self.xenium_transcripts.as_mut(),
        ) {
            if let Some(value) = saved.get("visible").and_then(serde_json::Value::as_bool) {
                *visible = value;
            }
            if let Some(saved_style) = saved.get("style") {
                Self::apply_points_style_json(style, saved_style);
            }
            if let Some(value) = saved.get("gene_query").and_then(serde_json::Value::as_str) {
                gene_query.clear();
                gene_query.push_str(value);
            }
            if let Some(value) = saved
                .get("max_render_points_total")
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
            {
                *max_points = value;
            }
        }
        Ok(())
    }

    fn capture(app: &OmeZarrViewerApp) -> Self {
        Self {
            camera: app.camera.clone(),
            render: ViewportRenderState {
                last_canvas_rect: app.last_canvas_rect,
                active_render_id: app.active_render_id,
                previous_render_id: app.previous_render_id,
                active_render_smooth_pixels: app.active_render_smooth_pixels,
                previous_render_smooth_pixels: app.previous_render_smooth_pixels,
                previous_view_selection: app.previous_view_selection,
                previous_displayed_view_selection: app.previous_displayed_view_selection,
                last_render_view_selection: app.last_render_view_selection,
                last_target_level: app.last_target_level,
                fallback_ceiling_level: app.fallback_ceiling_level,
                last_visible_world_tiles: app.last_visible_world_tiles,
                zoom_out_floor_level: app.zoom_out_floor_level,
                zoom_out_floor_until: app.zoom_out_floor_until,
                zoom_out_floor_visible_world_tiles: app.zoom_out_floor_visible_world_tiles,
            },
            transient: ViewportTransientState {
                draft_view_slice_level0: app.draft_view_slice_level0,
                selected_channel_layers: app.selected_channel_layers.clone(),
                selected_channel_group_id: app.selected_channel_group_id,
                selected_overlay_layers: app.selected_overlay_layers.clone(),
                object_filter_cache: app.seg_objects.viewport_filter_cache_state(),
            },
            selected_channel: app.selected_channel,
            view_plane_mode: app.view_plane_mode,
            current_x_level0: app.current_x_level0,
            current_y_level0: app.current_y_level0,
            current_z_level0: app.current_z_level0,
            channels: app.channels.clone(),
            layer_groups: app.current_layer_groups(),
            channel_window_overrides: app.channel_window_overrides.clone(),
            active_layer: app.active_layer,
            overlay_layer_order: app.overlay_layer_order.clone(),
            channel_layer_order: app.channel_layer_order.clone(),
            channel_sort_mode: app.channel_sort_mode,
            object_display: app.seg_objects.project_display_state(),
            object_filter: app.seg_objects.viewport_filter_state(),
            object_visible: app.seg_objects.visible,
            object_opacity: app.seg_objects.opacity,
            object_width_screen_px: app.seg_objects.width_screen_px,
            object_color_rgb: app.seg_objects.color_rgb,
            object_show_selection_overlay: app.seg_objects.show_selection_overlay,
            cells_outlines_visible: app.cells_outlines_visible,
            cells_outlines_color_rgb: app.cells_outlines_color_rgb,
            cells_outlines_opacity: app.cells_outlines_opacity,
            cells_outlines_width_px: app.cells_outlines_width_px,
            cell_points_visible: app.cell_points.visible,
            cell_points_style: app.cell_points.style.clone(),
            masks: app
                .mask_layers
                .iter()
                .map(|layer| MaskViewportPresentation {
                    id: layer.id,
                    visible: layer.visible,
                    opacity: layer.opacity,
                    width_screen_px: layer.width_screen_px,
                    display_mode: layer.display_mode,
                    color_rgb: layer.color_rgb,
                })
                .collect(),
            annotations: app
                .annotation_layers
                .iter()
                .map(|layer| AnnotationViewportPresentation {
                    id: layer.id,
                    visible: layer.visible,
                    style: layer.style.clone(),
                    category_styles: layer.category_styles.clone(),
                    continuous_shape: layer.continuous_shape,
                    continuous_range: layer.continuous_range,
                })
                .collect(),
            seg_geojson_visible: app.seg_geojson.visible,
            seg_geojson_opacity: app.seg_geojson.opacity,
            seg_geojson_width_screen_px: app.seg_geojson.width_screen_px,
            seg_geojson_color_rgb: app.seg_geojson.color_rgb,
            spatial_shapes: app
                .spatial_layers
                .shapes
                .iter()
                .map(|layer| SpatialShapeViewportPresentation {
                    id: layer.id,
                    visible: layer.visible,
                    opacity: layer.opacity,
                    width_screen_px: layer.width_screen_px,
                    color_rgb: layer.color_rgb,
                    objects: layer
                        .object_layer()
                        .map(ObjectLayerViewportPresentation::capture),
                })
                .collect(),
            spatial_points: app.spatial_layers.points.as_ref().map(|layer| {
                (
                    layer.visible,
                    layer.style.clone(),
                    layer.threshold,
                    layer.max_render_points_total,
                )
            }),
            spatial_images: app
                .spatial_image_layers
                .images
                .iter()
                .map(|layer| SpatialImageViewportPresentation {
                    id: layer.id,
                    visible: layer.visible,
                    opacity: layer.opacity,
                    current_z_level0: layer.current_z_level0,
                    channels: layer.channels.clone(),
                })
                .collect(),
            xenium_cells: app.xenium_layers.cells.as_ref().map(|layer| {
                (
                    layer.visible,
                    layer.opacity,
                    layer.width_screen_px,
                    layer.color_rgb,
                )
            }),
            xenium_transcripts: app.xenium_layers.transcripts.as_ref().map(|layer| {
                (
                    layer.visible,
                    layer.style.clone(),
                    layer.gene_query.clone(),
                    layer.max_render_points_total,
                )
            }),
            smooth_pixels: app.smooth_pixels,
            show_scale_bar: app.show_scale_bar,
            show_hud: app.show_hud,
            show_tile_debug: app.show_tile_debug,
        }
    }

    fn apply(&self, app: &mut OmeZarrViewerApp) {
        app.camera = self.camera.clone();
        app.last_canvas_rect = self.render.last_canvas_rect;
        app.active_render_id = self.render.active_render_id;
        app.previous_render_id = self.render.previous_render_id;
        app.active_render_smooth_pixels = self.render.active_render_smooth_pixels;
        app.previous_render_smooth_pixels = self.render.previous_render_smooth_pixels;
        app.previous_view_selection = self.render.previous_view_selection;
        app.previous_displayed_view_selection = self.render.previous_displayed_view_selection;
        app.last_render_view_selection = self.render.last_render_view_selection;
        app.last_target_level = self.render.last_target_level;
        app.fallback_ceiling_level = self.render.fallback_ceiling_level;
        app.last_visible_world_tiles = self.render.last_visible_world_tiles;
        app.zoom_out_floor_level = self.render.zoom_out_floor_level;
        app.zoom_out_floor_until = self.render.zoom_out_floor_until;
        app.zoom_out_floor_visible_world_tiles = self.render.zoom_out_floor_visible_world_tiles;
        app.selected_channel = self.selected_channel;
        app.view_plane_mode = self.view_plane_mode;
        app.draft_view_slice_level0 = self.transient.draft_view_slice_level0;
        app.current_x_level0 = self.current_x_level0;
        app.current_y_level0 = self.current_y_level0;
        app.current_z_level0 = self.current_z_level0;
        // Channel identity, names, and notes are document metadata. Only the
        // visual fields are swapped with a viewport presentation.
        for source in &self.channels {
            if let Some(channel) = app
                .channels
                .iter_mut()
                .find(|channel| channel.index == source.index)
            {
                channel.color_rgb = source.color_rgb;
                channel.window = source.window;
                channel.visible = source.visible;
            }
        }
        app.viewport_layer_groups.clone_from(&self.layer_groups);
        app.channel_window_overrides
            .clone_from(&self.channel_window_overrides);
        app.active_layer = self.active_layer;
        app.selected_channel_layers
            .clone_from(&self.transient.selected_channel_layers);
        app.selected_channel_group_id = self.transient.selected_channel_group_id;
        app.selected_overlay_layers
            .clone_from(&self.transient.selected_overlay_layers);
        app.overlay_layer_order
            .clone_from(&self.overlay_layer_order);
        app.channel_layer_order
            .clone_from(&self.channel_layer_order);
        app.channel_sort_mode = self.channel_sort_mode;
        app.seg_objects
            .apply_project_display_state(&self.object_display);
        app.seg_objects
            .apply_viewport_filter_state(&self.object_filter);
        app.seg_objects
            .apply_viewport_filter_cache_state(&self.transient.object_filter_cache);
        app.seg_objects.visible = self.object_visible;
        app.seg_objects.opacity = self.object_opacity;
        app.seg_objects.width_screen_px = self.object_width_screen_px;
        app.seg_objects.color_rgb = self.object_color_rgb;
        app.seg_objects.show_selection_overlay = self.object_show_selection_overlay;
        app.cells_outlines_visible = self.cells_outlines_visible;
        app.cells_outlines_color_rgb = self.cells_outlines_color_rgb;
        app.cells_outlines_opacity = self.cells_outlines_opacity;
        app.cells_outlines_width_px = self.cells_outlines_width_px;
        app.cell_points.visible = self.cell_points_visible;
        app.cell_points.style.clone_from(&self.cell_points_style);
        for presentation in &self.masks {
            if let Some(layer) = app
                .mask_layers
                .iter_mut()
                .find(|layer| layer.id == presentation.id)
            {
                layer.visible = presentation.visible;
                layer.opacity = presentation.opacity;
                layer.width_screen_px = presentation.width_screen_px;
                layer.display_mode = presentation.display_mode;
                layer.color_rgb = presentation.color_rgb;
            }
        }
        for presentation in &self.annotations {
            if let Some(layer) = app
                .annotation_layers
                .iter_mut()
                .find(|layer| layer.id == presentation.id)
            {
                layer.visible = presentation.visible;
                layer.style.clone_from(&presentation.style);
                layer
                    .category_styles
                    .clone_from(&presentation.category_styles);
                layer.continuous_shape = presentation.continuous_shape;
                layer.continuous_range = presentation.continuous_range;
            }
        }
        app.seg_geojson.visible = self.seg_geojson_visible;
        app.seg_geojson.opacity = self.seg_geojson_opacity;
        app.seg_geojson.width_screen_px = self.seg_geojson_width_screen_px;
        app.seg_geojson.color_rgb = self.seg_geojson_color_rgb;
        for presentation in &self.spatial_shapes {
            if let Some(layer) = app
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|layer| layer.id == presentation.id)
            {
                layer.visible = presentation.visible;
                layer.opacity = presentation.opacity;
                layer.width_screen_px = presentation.width_screen_px;
                layer.color_rgb = presentation.color_rgb;
                if let (Some(objects), Some(layer_objects)) =
                    (&presentation.objects, layer.object_layer_mut())
                {
                    objects.apply(layer_objects);
                }
            }
        }
        if let (Some((visible, style, threshold, max_points)), Some(layer)) =
            (&self.spatial_points, app.spatial_layers.points.as_mut())
        {
            layer.visible = *visible;
            layer.style.clone_from(style);
            layer.threshold = *threshold;
            layer.max_render_points_total = *max_points;
        }
        for presentation in &self.spatial_images {
            if let Some(layer) = app
                .spatial_image_layers
                .images
                .iter_mut()
                .find(|layer| layer.id == presentation.id)
            {
                layer.visible = presentation.visible;
                layer.opacity = presentation.opacity;
                layer.current_z_level0 = presentation.current_z_level0;
                layer.channels.clone_from(&presentation.channels);
            }
        }
        if let (Some((visible, opacity, width, color)), Some(layer)) =
            (&self.xenium_cells, app.xenium_layers.cells.as_mut())
        {
            layer.visible = *visible;
            layer.opacity = *opacity;
            layer.width_screen_px = *width;
            layer.color_rgb = *color;
        }
        if let (Some((visible, style, gene_query, max_points)), Some(layer)) = (
            &self.xenium_transcripts,
            app.xenium_layers.transcripts.as_mut(),
        ) {
            layer.visible = *visible;
            layer.style.clone_from(style);
            layer.gene_query.clone_from(gene_query);
            layer.max_render_points_total = *max_points;
        }
        app.smooth_pixels = self.smooth_pixels;
        app.show_scale_bar = self.show_scale_bar;
        app.show_hud = self.show_hud;
        app.show_tile_debug = self.show_tile_debug;
    }

    fn layer_visible(&self, id: LayerId) -> Option<bool> {
        match id {
            LayerId::Channel(index) => self.channels.get(index).map(|channel| channel.visible),
            LayerId::SpatialImage(id) => self
                .spatial_images
                .iter()
                .find(|layer| layer.id == id)
                .map(|layer| layer.visible),
            LayerId::SegmentationLabels => Some(self.cells_outlines_visible),
            LayerId::SegmentationGeoJson => Some(self.seg_geojson_visible),
            LayerId::SegmentationObjects => Some(self.object_visible),
            LayerId::Mask(id) => self
                .masks
                .iter()
                .find(|layer| layer.id == id)
                .map(|layer| layer.visible),
            LayerId::Points => Some(self.cell_points_visible),
            LayerId::Annotation(id) => self
                .annotations
                .iter()
                .find(|layer| layer.id == id)
                .map(|layer| layer.visible),
            LayerId::SpatialShape(id) => self
                .spatial_shapes
                .iter()
                .find(|layer| layer.id == id)
                .map(|layer| layer.visible),
            LayerId::SpatialPoints => self.spatial_points.as_ref().map(|state| state.0),
            LayerId::XeniumCells => self.xenium_cells.as_ref().map(|state| state.0),
            LayerId::XeniumTranscripts => self.xenium_transcripts.as_ref().map(|state| state.0),
        }
    }

    #[cfg(test)]
    fn set_layer_visible(&mut self, id: LayerId, visible: bool) {
        match id {
            LayerId::Channel(index) => {
                if let Some(channel) = self.channels.get_mut(index) {
                    channel.visible = visible;
                }
            }
            LayerId::SpatialImage(id) => {
                if let Some(layer) = self.spatial_images.iter_mut().find(|layer| layer.id == id) {
                    layer.visible = visible;
                }
            }
            LayerId::SegmentationLabels => self.cells_outlines_visible = visible,
            LayerId::SegmentationGeoJson => self.seg_geojson_visible = visible,
            LayerId::SegmentationObjects => self.object_visible = visible,
            LayerId::Mask(id) => {
                if let Some(layer) = self.masks.iter_mut().find(|layer| layer.id == id) {
                    layer.visible = visible;
                }
            }
            LayerId::Points => self.cell_points_visible = visible,
            LayerId::Annotation(id) => {
                if let Some(layer) = self.annotations.iter_mut().find(|layer| layer.id == id) {
                    layer.visible = visible;
                }
            }
            LayerId::SpatialShape(id) => {
                if let Some(layer) = self.spatial_shapes.iter_mut().find(|layer| layer.id == id) {
                    layer.visible = visible;
                }
            }
            LayerId::SpatialPoints => {
                if let Some(state) = self.spatial_points.as_mut() {
                    state.0 = visible;
                }
            }
            LayerId::XeniumCells => {
                if let Some(state) = self.xenium_cells.as_mut() {
                    state.0 = visible;
                }
            }
            LayerId::XeniumTranscripts => {
                if let Some(state) = self.xenium_transcripts.as_mut() {
                    state.0 = visible;
                }
            }
        }
    }
}

pub struct OmeZarrViewerApp {
    dataset: OmeZarrDataset,
    store: Arc<dyn zarrs::storage::ReadableStorageTraits>,
    remote_runtime: Option<Arc<tokio::runtime::Runtime>>,
    loader: crate::render::tiles::TileLoaderHandle,
    raw_loader: Option<RawTileLoaderHandle>,
    label_cells: Option<LabelZarrDataset>,
    label_loader: Option<LabelTileLoaderHandle>,
    label_cells_xform: Option<Vec<LabelToWorld>>,
    seg_label_names: Vec<String>,
    seg_label_selected: String,
    seg_label_input: String,
    seg_label_status: String,
    seg_label_prompt_open: bool,
    seg_label_prompt_always: bool,
    seg_label_prompt_preference: LabelPromptSessionPreference,
    hist_loader: HistogramLoaderHandle,
    chanmax_loader: ChannelMaxLoaderHandle,
    chanmax_request_id: u64,
    chanmax_level: usize,
    chanmax_pending: Vec<bool>,
    chanmax_snapshot: Vec<Option<(f32, f32)>>,
    cache: TileCache<egui::TextureHandle>,
    pending: Vec<TileResponse>,
    hist: Option<HistogramResponse>,
    hist_request_id: u64,
    hist_request_pending: bool,
    hist_dirty: bool,
    hist_navigation_dirty_since: Option<Instant>,
    hist_last_sent: Instant,

    camera: Camera,
    active_render_id: u64,
    previous_render_id: Option<u64>,
    active_render_smooth_pixels: bool,
    previous_render_smooth_pixels: Option<bool>,
    previous_view_selection: Option<ViewPlaneSelection>,
    previous_displayed_view_selection: Option<ViewPlaneSelection>,
    last_render_view_selection: ViewPlaneSelection,
    last_canvas_rect: Option<egui::Rect>,
    last_target_level: Option<usize>,
    fallback_ceiling_level: Option<usize>,
    last_visible_world_tiles: Option<egui::Rect>,
    zoom_out_floor_level: Option<usize>,
    zoom_out_floor_until: Option<Instant>,
    zoom_out_floor_visible_world_tiles: Option<egui::Rect>,

    selected_channel: usize,
    view_plane_mode: ViewPlaneMode,
    draft_view_slice_level0: Option<u64>,
    current_x_level0: u64,
    current_y_level0: u64,
    current_z_level0: u64,
    channels: Vec<ChannelInfo>,
    channel_window_overrides: HashMap<String, (f32, f32)>,
    auto_contrast_settings: AutoContrastSettings,
    fast_object_rendering: bool,
    channel_list_search: String,

    active_layer: LayerId,
    selected_channel_layers: HashSet<usize>,
    memory_selected_channels: HashSet<usize>,
    channel_select_anchor_idx: Option<usize>,
    selected_channel_group_id: Option<u64>,
    quick_contrast_target: top_bar::QuickContrastTarget,
    selected_overlay_layers: HashSet<LayerId>,
    overlay_select_anchor_pos: Option<usize>,
    show_left_panel: bool,
    show_right_panel: bool,
    close_dialog_open: bool,
    pinned_levels: PinnedLevels,
    pending_memory_load: Option<PendingMemoryAction<Vec<PendingPinnedLevelLoadRequest>>>,
    memory_status: String,
    system_memory: Option<SystemMemorySnapshot>,
    system_memory_last_refresh: Option<Instant>,
    left_tab: LeftTab,
    right_tab: RightTab,
    project_space: ProjectSpace,
    project_cfg_seen: u64,
    roi_selector: RoiSelectorPanel,
    cell_thresholds: CellThresholdsPanel,
    cell_points: PointsLayer,
    annotation_layers: Vec<AnnotationPointsLayer>,
    mask_layers: Vec<MaskLayer>,
    tool_mode: ToolMode,
    drawing_mask_layer: Option<u64>,
    drawing_mask_polygon: Vec<egui::Pos2>,
    selected_mask_polygon: Option<MaskPolygonSelection>,
    selected_mask_vertex: Option<usize>,
    dragging_mask_vertex: Option<MaskVertexDrag>,
    moving_mask_polygon: Option<MaskPolygonMoveState>,
    selection_rect_start_world: Option<egui::Pos2>,
    selection_rect_current_world: Option<egui::Pos2>,
    selection_lasso_world: Vec<egui::Pos2>,
    threshold_region_min_pixels: usize,
    threshold_region_scope: ThresholdRegionScope,
    threshold_region_full_level: usize,
    threshold_region_status: String,
    threshold_region_preview: Option<ThresholdRegionPreview>,
    cells_outlines_visible: bool,
    cells_outlines_color_rgb: [u8; 3],
    cells_outlines_opacity: f32,
    cells_outlines_width_px: f32,
    points_gl: Option<PointsGlRenderer>,
    threshold_preview_gl: Option<ThresholdPreviewGlRenderer>,
    tiles_gl: Option<TilesGl>,
    labels_gl: Option<LabelsGl>,
    remote_dialog_open: bool,
    remote_mode: RemoteMode,
    remote_http_url: String,
    remote_s3_endpoint: String,
    remote_s3_region: String,
    remote_s3_bucket: String,
    remote_s3_prefix: String,
    remote_s3_access_key: String,
    remote_s3_secret_key: String,
    remote_status: String,
    remote_s3_browser: Option<RemoteS3BrowserState>,

    pending_request: Option<ViewerRequest>,
    native_control_intents: Vec<NativeControlIntent>,
    control_actor_object_generation: u64,
    control_actor_secondary_object_generations: HashMap<u64, u64>,
    control_actor_secondary_object_selection_generations: HashMap<u64, u64>,
    control_actor_secondary_object_analysis_generations: HashMap<u64, u64>,
    control_actor_label_generation: u64,
    control_actor_object_selection_generation: u64,
    control_actor_mask_generation: u64,
    control_actor_workspace_revision: u64,
    pending_control_actor_mask_projection: Option<serde_json::Value>,
    control_actor_threshold_generation: u64,
    control_actor_analysis_generation: u64,
    control_actor_measurement_generation: u64,
    control_actor_object_export_generation: u64,
    control_actor_mask_undo_available: bool,
    control_actor_tile_policy_generation: u64,
    group_layers_dialog: Option<GroupLayersDialog>,
    hover_tooltip_state: Option<HoverTooltipState>,
    active_help_topic: Option<crate::ui::help::HelpTopic>,
    roi_info_open: bool,

    smooth_pixels: bool,
    show_tile_debug: bool,
    mask_draw_debug_stats: MaskDrawDebugStats,
    show_scale_bar: bool,
    show_hud: bool,
    tile_loader_threads: usize,
    tile_prefetch_mode: TilePrefetchMode,
    tile_prefetch_aggressiveness: TilePrefetchAggressiveness,
    tile_loading_status: String,
    prefer_pinned_finer_levels: bool,

    seg_geojson: GeoJsonSegmentationLayer,
    seg_objects: ObjectsLayer,
    spatial_image_layers: SpatialImageLayers,
    spatial_layers: SpatialDataLayers,
    spatial_image_transform: SpatialDataTransform2,
    spatial_label_transform: SpatialDataTransform2,
    spatial_root: Option<PathBuf>,
    spatial_label_store: Option<Arc<dyn zarrs::storage::ReadableStorageTraits>>,
    xenium_layers: XeniumLayers,

    channel_offsets_world: Vec<egui::Vec2>,
    channel_scales: Vec<egui::Vec2>,
    channel_rotations_rad: Vec<f32>,
    loaded_layer_offsets_world: HashMap<LayerId, egui::Vec2>,
    points_offset_world: egui::Vec2,
    spatial_points_offset_world: egui::Vec2,
    seg_labels_offset_world: egui::Vec2,
    seg_geojson_offset_world: egui::Vec2,
    seg_objects_offset_world: egui::Vec2,
    xenium_cells_offset_world: egui::Vec2,
    xenium_transcripts_offset_world: egui::Vec2,

    overlay_layer_order: Vec<LayerId>,
    channel_layer_order: Vec<usize>,
    channel_sort_mode: ChannelSortMode,
    layer_drag: Option<LayerDragState>,
    layer_move: Option<LayerMoveState>,
    layer_transform: Option<LayerTransformState>,
    tiff_plane_state: Option<TiffPlaneState>,
    screenshot_settings: ScreenshotSettings,
    screenshot_settings_open: bool,
    screenshot_worker: ScreenshotWorkerHandle,
    screenshot_next_id: u64,
    screenshot_pending: VecDeque<PendingViewportScreenshot>,
    screenshot_in_flight: HashMap<u64, ViewportId>,
    screenshot_output_dir: Option<PathBuf>,
    viewport_workspace: Option<ViewportWorkspace<ViewerViewportState>>,
    native_viewport_command_scope: Option<NativeViewportCommandScope>,
    viewport_layer_groups: ProjectLayerGroups,
    viewport_raw_active_keys: Option<HashSet<RawTileKey>>,
    viewport_cpu_active_keys: Option<HashSet<TileKey>>,
    viewport_label_active_keys: Option<HashSet<LabelTileKey>>,
    viewport_spatial_image_active_keys: Option<HashMap<u64, HashSet<RawTileKey>>>,
    viewport_frame_plan_ms: f32,
    viewport_frame_plan_ema_ms: f32,
    viewport_frame_plan_samples: u64,
}

#[derive(Debug, Clone)]
struct NativeViewportCommandScope {
    viewport_id: String,
    navigation_revision: u64,
    presentation_revision: u64,
}

#[derive(Debug, Clone)]
struct LayerMoveState {
    targets: Vec<LayerOffsetEntry>,
    actor_scope: Option<(String, u64)>,
}

#[derive(Debug, Clone)]
struct PendingPinnedLevelLoadRequest {
    level: usize,
    selected_channels: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerTransformKind {
    Translate,
    Scale,
    Rotate,
}

#[derive(Debug, Clone)]
struct LayerTransformState {
    layer: LayerId,
    kind: LayerTransformKind,
    start_offset_world: egui::Vec2,
    start_scale: egui::Vec2,
    start_rotation_rad: f32,
    pivot_screen: egui::Pos2,
    start_pointer_screen: egui::Pos2,
    start_angle_rad: f32,
    start_len_screen: f32,
    actor_scope: Option<(String, u64)>,
}

#[derive(Debug, Clone)]
pub enum ViewerRequest {
    OpenRemoteS3Mosaic(Vec<S3DatasetSelection>),
}

#[derive(Debug, Clone)]
pub struct NativeControlIntent {
    pub method: &'static str,
    pub params: serde_json::Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelPromptSessionPreference {
    Ask,
    AlwaysSkip,
    AlwaysLoad,
}

#[derive(Debug, Clone, Copy)]
struct LabelToWorld {
    scale_x: f32,
    scale_y: f32,
    offset_x: f32,
    offset_y: f32,
}

#[derive(Debug, Clone)]
struct TiffPlaneState {
    dataset_root: PathBuf,
    image_path: PathBuf,
    dataset_name: String,
    channel_name: String,
    size_z: usize,
    size_t: usize,
    current_z: usize,
    current_t: usize,
    draft_z: usize,
    draft_t: usize,
    status: String,
}

struct TiffRuntimeAssets {
    dataset: OmeZarrDataset,
    store: Arc<dyn zarrs::storage::ReadableStorageTraits>,
    loader: crate::render::tiles::TileLoaderHandle,
    raw_loader: Option<RawTileLoaderHandle>,
    hist_loader: HistogramLoaderHandle,
    chanmax_loader: ChannelMaxLoaderHandle,
    chanmax_level: usize,
    tiff_plane_state: Option<TiffPlaneState>,
}

pub(crate) fn dummy_local_store_for_path(
    path: &Path,
) -> anyhow::Result<Arc<dyn zarrs::storage::ReadableStorageTraits>> {
    let store_root = if path.is_dir() {
        path.to_path_buf()
    } else if let Some(parent) = path.parent() {
        parent.to_path_buf()
    } else {
        std::env::current_dir().context("resolve current directory for TIFF store")?
    };
    Ok(Arc::new(zarrs::filesystem::FilesystemStore::new(
        &store_root,
    )?))
}

fn normalize_deep_link_name(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect()
}

fn marker_name_from_channel_label(value: &str) -> &str {
    let marker = value
        .split_once(" - ")
        .map(|(_, marker)| marker)
        .unwrap_or(value)
        .trim();
    marker
        .split_once(" (")
        .map(|(marker, _)| marker)
        .or_else(|| marker.split_once(" [").map(|(marker, _)| marker))
        .unwrap_or(marker)
        .trim()
}

fn suggest_channel_alias(value: &str) -> String {
    let marker = marker_name_from_channel_label(value);
    let tokens = marker
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .filter(|token| !looks_like_channel_id_token(token))
        .filter(|token| !looks_like_fluorophore_token(token))
        .map(|token| token.to_ascii_lowercase())
        .collect::<Vec<_>>();
    if !tokens.is_empty() {
        return tokens.join("_");
    }
    normalize_deep_link_name(marker)
}

fn looks_like_channel_id_token(token: &str) -> bool {
    let token = token.trim();
    let Some(rest) = token.strip_prefix('C').or_else(|| token.strip_prefix('c')) else {
        return false;
    };
    !rest.is_empty() && rest.chars().all(|ch| ch.is_ascii_digit())
}

fn looks_like_fluorophore_token(token: &str) -> bool {
    let token = normalize_deep_link_name(token);
    if token.is_empty() {
        return false;
    }
    if token.chars().all(|ch| ch.is_ascii_digit()) {
        return true;
    }
    matches!(
        token.as_str(),
        "fitc"
            | "apc"
            | "pe"
            | "cy3"
            | "cy5"
            | "cy7"
            | "tritc"
            | "texasred"
            | "dylight"
            | "alexa"
            | "fluor"
            | "af"
    ) || token
        .strip_prefix("opal")
        .is_some_and(|rest| !rest.is_empty() && rest.chars().all(|ch| ch.is_ascii_digit()))
        || token
            .strip_prefix("af")
            .is_some_and(|rest| !rest.is_empty() && rest.chars().all(|ch| ch.is_ascii_digit()))
}

#[cfg(test)]
fn cd_marker_digit_suffix(value: &str) -> Option<(&str, &str)> {
    let rest = value.strip_prefix("cd")?;
    let digit_len = rest
        .char_indices()
        .take_while(|(_, ch)| ch.is_ascii_digit())
        .map(|(idx, ch)| idx + ch.len_utf8())
        .last()
        .unwrap_or(0);
    if digit_len == 0 {
        return None;
    }
    let (digits, suffix) = rest.split_at(digit_len);
    Some((digits, suffix))
}

#[cfg(test)]
fn marker_alias_matches(requested: &str, candidate_marker: &str) -> bool {
    let requested = normalize_deep_link_name(requested);
    let candidate = normalize_deep_link_name(candidate_marker);
    if requested.is_empty() || candidate.is_empty() {
        return false;
    }
    if requested == candidate {
        return true;
    }

    let Some((requested_digits, requested_suffix)) = cd_marker_digit_suffix(&requested) else {
        return false;
    };
    let Some((candidate_digits, candidate_suffix)) = cd_marker_digit_suffix(&candidate) else {
        return false;
    };
    if requested_digits != candidate_digits {
        return false;
    }
    if requested_suffix.is_empty() {
        return candidate_suffix
            .chars()
            .next()
            .is_none_or(|ch| ch.is_ascii_alphabetic());
    }
    candidate_suffix == requested_suffix
}

#[cfg(test)]
fn deep_link_channel_groups(raw: &[String], alternatives: &[Vec<String>]) -> Vec<Vec<String>> {
    let mut groups = alternatives
        .iter()
        .filter_map(|terms| {
            let mut group = Vec::new();
            for term in terms {
                push_unique_term(&mut group, term);
            }
            (!group.is_empty()).then_some(group)
        })
        .collect::<Vec<_>>();
    for term in raw {
        let mut group = Vec::new();
        push_unique_term(&mut group, term);
        if !group.is_empty() {
            groups.push(group);
        }
    }
    groups
}

#[cfg(test)]
fn push_unique_term(dst: &mut Vec<String>, value: &str) {
    let value = value.trim();
    if !value.is_empty() && !dst.iter().any(|existing| existing == value) {
        dst.push(value.to_string());
    }
}

#[cfg(test)]
fn channel_intensity_stats_json(
    idx: usize,
    name: &str,
    level: usize,
    downsample: f32,
    data: &ndarray::ArrayD<u16>,
) -> serde_json::Value {
    let mut values = data.iter().copied().collect::<Vec<_>>();
    if values.is_empty() {
        return serde_json::json!({
            "index": idx,
            "name": name,
            "level": level,
            "downsample": downsample,
            "n": 0,
            "error": "empty image subset",
        });
    }
    values.sort_unstable();
    let n = values.len();
    let mut sum = 0u64;
    let mut nonzero = 0usize;
    for &value in &values {
        sum = sum.saturating_add(value as u64);
        if value != 0 {
            nonzero += 1;
        }
    }
    serde_json::json!({
        "index": idx,
        "name": name,
        "level": level,
        "downsample": downsample,
        "shape": data.shape(),
        "n": n,
        "nonzero": nonzero,
        "nonzero_fraction": nonzero as f64 / n as f64,
        "min": values[0],
        "q1": percentile_sorted_u16(&values, 0.25),
        "median": percentile_sorted_u16(&values, 0.50),
        "q3": percentile_sorted_u16(&values, 0.75),
        "p95": percentile_sorted_u16(&values, 0.95),
        "p99": percentile_sorted_u16(&values, 0.99),
        "max": values[n - 1],
        "mean": sum as f64 / n as f64,
    })
}

#[cfg(test)]
fn percentile_sorted_u16(values: &[u16], q: f64) -> u16 {
    if values.is_empty() {
        return 0;
    }
    let idx = ((values.len().saturating_sub(1)) as f64 * q.clamp(0.0, 1.0)).round() as usize;
    values[idx.min(values.len() - 1)]
}

fn channel_groups_snapshot(
    groups: &ProjectLayerGroups,
    channels: &[ChannelInfo],
) -> serde_json::Value {
    serde_json::Value::Array(
        groups
            .channel_groups
            .iter()
            .map(|group| {
                let members = channels
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, channel)| {
                        let member = groups.channel_members.get(channel.name.as_str())?;
                        (member.group_id == group.id).then(|| {
                            serde_json::json!({
                                "index": idx,
                                "name": channel.name,
                                "inherit_color": member.inherit_color,
                            })
                        })
                    })
                    .collect::<Vec<_>>();
                serde_json::json!({
                    "id": group.id,
                    "name": group.name,
                    "expanded": group.expanded,
                    "color_rgb": group.color_rgb,
                    "members": members,
                })
            })
            .collect(),
    )
}

pub(crate) fn build_tiff_dataset(
    dataset_root: PathBuf,
    dataset_name: String,
    levels: Vec<crate::data::ome::LevelInfo>,
    dims: Dims,
    channels: Vec<ChannelInfo>,
    abs_max: f32,
    pixel_size_xy: Option<([f32; 2], [Option<String>; 2])>,
) -> OmeZarrDataset {
    let axes = if dims.c.is_some() {
        vec![
            crate::data::ome::Axis {
                name: "c".to_string(),
                unit: None,
            },
            crate::data::ome::Axis {
                name: "y".to_string(),
                unit: pixel_size_xy
                    .as_ref()
                    .and_then(|(_, units)| units[0].clone()),
            },
            crate::data::ome::Axis {
                name: "x".to_string(),
                unit: pixel_size_xy
                    .as_ref()
                    .and_then(|(_, units)| units[1].clone()),
            },
        ]
    } else {
        vec![
            crate::data::ome::Axis {
                name: "y".to_string(),
                unit: pixel_size_xy
                    .as_ref()
                    .and_then(|(_, units)| units[0].clone()),
            },
            crate::data::ome::Axis {
                name: "x".to_string(),
                unit: pixel_size_xy
                    .as_ref()
                    .and_then(|(_, units)| units[1].clone()),
            },
        ]
    };
    let multiscale = crate::data::ome::Multiscale {
        name: Some(dataset_name),
        axes,
        datasets: levels
            .iter()
            .map(|l| crate::data::ome::MultiscaleDataset {
                path: l.path.clone(),
                coordinate_transformations: vec![crate::data::ome::CoordTransform::Scale {
                    scale: if dims.c.is_some() {
                        if let Some((sizes, _)) = pixel_size_xy.as_ref() {
                            vec![1.0, sizes[0] * l.downsample, sizes[1] * l.downsample]
                        } else {
                            vec![1.0, l.downsample, l.downsample]
                        }
                    } else {
                        if let Some((sizes, _)) = pixel_size_xy.as_ref() {
                            vec![sizes[0] * l.downsample, sizes[1] * l.downsample]
                        } else {
                            vec![l.downsample, l.downsample]
                        }
                    },
                }],
            })
            .collect(),
    };

    OmeZarrDataset {
        source: crate::data::dataset_source::DatasetSource::Local(dataset_root),
        multiscale,
        levels,
        channels,
        dims,
        abs_max,
        render_kind: crate::data::ome::DatasetRenderKind::Image,
    }
}

fn build_tiff_runtime_assets(
    gpu_available: bool,
    dataset_root: PathBuf,
    image_path: PathBuf,
    dataset_name: String,
    channel_name: String,
    plane_selection: crate::xenium::TiffPlaneSelection,
) -> anyhow::Result<TiffRuntimeAssets> {
    let pyramid = Arc::new(crate::xenium::TiffPyramid::open_with_selection(
        &image_path,
        plane_selection,
    )?);
    pyramid.validate_supported_ome_layout()?;
    let levels = pyramid.to_levels_info();
    crate::log_info!("tiff pyramid levels={}", levels.len());
    let dataset = build_tiff_dataset(
        dataset_root.clone(),
        dataset_name.clone(),
        levels,
        pyramid.dims(),
        pyramid.default_channels_named(&channel_name),
        pyramid.abs_max,
        pyramid.physical_pixel_size_xy(),
    );
    let store = dummy_local_store_for_path(&dataset_root)?;
    build_tiff_runtime_assets_from_parts(
        gpu_available,
        dataset_root,
        image_path,
        dataset_name,
        channel_name,
        pyramid,
        dataset,
        store,
    )
}

fn build_tiff_runtime_assets_from_resource(
    gpu_available: bool,
    resource: &crate::data::document::AlternateDocumentResource,
) -> anyhow::Result<TiffRuntimeAssets> {
    let pyramid = resource
        .payload::<crate::xenium::TiffPyramid>()
        .ok_or_else(|| anyhow::anyhow!("TIFF document has an incompatible native resource"))?;
    build_tiff_runtime_assets_from_prepared_resource(gpu_available, resource, pyramid)
}

fn build_tiff_runtime_assets_from_prepared_resource(
    gpu_available: bool,
    resource: &crate::data::document::AlternateDocumentResource,
    pyramid: Arc<crate::xenium::TiffPyramid>,
) -> anyhow::Result<TiffRuntimeAssets> {
    let dataset = resource.dataset.clone();
    let dataset_root = dataset
        .source
        .local_path()
        .map(Path::to_path_buf)
        .ok_or_else(|| anyhow::anyhow!("TIFF document has no local source path"))?;
    let image_path = pyramid.path.clone();
    let dataset_name = dataset
        .multiscale
        .name
        .clone()
        .unwrap_or_else(|| "tiff".to_string());
    build_tiff_runtime_assets_from_parts(
        gpu_available,
        dataset_root,
        image_path,
        dataset_name,
        "image".to_string(),
        pyramid,
        dataset,
        Arc::clone(&resource.store),
    )
}

#[allow(clippy::too_many_arguments)]
fn build_tiff_runtime_assets_from_parts(
    gpu_available: bool,
    dataset_root: PathBuf,
    image_path: PathBuf,
    dataset_name: String,
    channel_name: String,
    pyramid: Arc<crate::xenium::TiffPyramid>,
    dataset: OmeZarrDataset,
    store: Arc<dyn zarrs::storage::ReadableStorageTraits>,
) -> anyhow::Result<TiffRuntimeAssets> {
    let dims_yx = (dataset.dims.y, dataset.dims.x);
    let tile_loader_threads = recommended_tile_loader_threads();
    let loader =
        crate::xenium::spawn_tiff_tile_loader(pyramid.clone(), dims_yx, tile_loader_threads)?;
    let raw_loader = if gpu_available {
        crate::xenium::spawn_tiff_raw_tile_loader(pyramid.clone(), dims_yx, tile_loader_threads)
            .ok()
    } else {
        None
    };
    let hist_loader = crate::xenium::spawn_tiff_histogram_loader(pyramid.clone())?;
    let chanmax_loader = crate::xenium::spawn_tiff_channel_max_loader(pyramid.clone())?;
    let tiff_plane_state = pyramid.has_plane_selection().then(|| TiffPlaneState {
        dataset_root,
        image_path,
        dataset_name,
        channel_name,
        size_z: pyramid.size_z,
        size_t: pyramid.size_t,
        current_z: pyramid.plane_selection.z,
        current_t: pyramid.plane_selection.t,
        draft_z: pyramid.plane_selection.z,
        draft_t: pyramid.plane_selection.t,
        status: String::new(),
    });
    Ok(TiffRuntimeAssets {
        chanmax_level: update::choose_default_max_level(&dataset),
        dataset,
        store,
        loader,
        raw_loader,
        hist_loader,
        chanmax_loader,
        tiff_plane_state,
    })
}

impl OmeZarrViewerApp {
    fn drain_channel_maxes(&mut self) {
        let mut any_changed = false;
        let abs_max = self.dataset.abs_max.max(1.0);
        while let Ok(msg) = self.chanmax_loader.rx.try_recv() {
            if msg.request_id != self.chanmax_request_id {
                continue;
            }
            let idx = msg.channel as usize;
            if idx >= self.channels.len() {
                continue;
            }
            if !self.chanmax_pending.get(idx).copied().unwrap_or(false) {
                continue;
            }

            // Don't override if the user changed this channel's window since we requested.
            if self.chanmax_snapshot.get(idx).copied().unwrap_or(None) != self.channels[idx].window
            {
                if let Some(p) = self.chanmax_pending.get_mut(idx) {
                    *p = false;
                }
                continue;
            }

            let mut lo = (msg.lo as f32).clamp(0.0, abs_max);
            let mut hi = (msg.hi as f32).clamp(0.0, abs_max);
            if !lo.is_finite() || lo < 0.0 {
                lo = 0.0;
            }
            if lo >= abs_max {
                lo = (abs_max - 1.0).max(0.0);
            }
            if !hi.is_finite() || hi <= lo {
                hi = (lo + 1.0).min(abs_max);
            }
            self.channels[idx].window = Some((lo, hi));
            if let Some(p) = self.chanmax_pending.get_mut(idx) {
                *p = false;
            }
            if idx == self.selected_channel {
                self.hist_dirty = true;
            }
            any_changed = true;
        }
        if any_changed {
            self.bump_render_id();
        }
    }
}

fn apply_preserved_channel_settings(prev: &[ChannelInfo], new: &mut [ChannelInfo]) {
    use std::collections::HashMap;

    #[derive(Clone, Copy)]
    struct Settings {
        visible: bool,
        color_rgb: [u8; 3],
    }

    let mut by_name: HashMap<&str, Settings> = HashMap::with_capacity(prev.len());
    for ch in prev {
        by_name.insert(
            ch.name.as_str(),
            Settings {
                visible: ch.visible,
                color_rgb: ch.color_rgb,
            },
        );
    }

    for ch in new {
        if let Some(s) = by_name.get(ch.name.as_str()) {
            ch.visible = s.visible;
            ch.color_rgb = s.color_rgb;
        }
    }
}

fn compute_label_to_world_xforms(
    image: &OmeZarrDataset,
    labels: &LabelZarrDataset,
    image_transform: SpatialDataTransform2,
) -> Vec<LabelToWorld> {
    // Best-effort: lock label levels to the image pyramid by index.
    // This keeps the labels perfectly aligned during zoom because we use the exact same
    // world mapping per level as the imagery.
    let img0 = image.levels.get(0);
    let img0_w = img0.map(|l| l.shape[image.dims.x] as f32).unwrap_or(0.0);
    let img0_h = img0.map(|l| l.shape[image.dims.y] as f32).unwrap_or(0.0);

    let mut out = Vec::with_capacity(labels.levels.len());
    for lvl in &labels.levels {
        let mut scale_x = image
            .levels
            .get(lvl.index)
            .map(|l| l.downsample)
            .unwrap_or(lvl.downsample);
        let mut scale_y = scale_x;

        // Fallback when label pyramid length doesn't match image:
        // compute downsample by matching extents.
        if img0_w > 0.0 && img0_h > 0.0 {
            let lw = lvl.shape.get(labels.dims.x).copied().unwrap_or(0) as f32;
            let lh = lvl.shape.get(labels.dims.y).copied().unwrap_or(0) as f32;
            if lw > 0.0 && lh > 0.0 {
                let dsx = img0_w / lw;
                let dsy = img0_h / lh;
                let ds = dsx.max(dsy);
                if ds.is_finite() && ds > 0.0 {
                    scale_x = ds;
                    scale_y = ds;
                }
            }
        }

        let xform_scale_x = image_transform.scale[0].max(1e-6);
        let xform_scale_y = image_transform.scale[1].max(1e-6);
        let mapped_scale_x = scale_x.max(1e-6) * xform_scale_x;
        let mapped_scale_y = scale_y.max(1e-6) * xform_scale_y;

        out.push(LabelToWorld {
            scale_x: mapped_scale_x,
            scale_y: mapped_scale_y,
            offset_x: image_transform.translation[0],
            offset_y: image_transform.translation[1],
        });
    }

    out
}

fn xform_screen_point(
    p: egui::Pos2,
    pivot: egui::Pos2,
    translation: egui::Vec2,
    scale: egui::Vec2,
    rotation_rad: f32,
) -> egui::Pos2 {
    let v = p - pivot;
    let v = egui::vec2(v.x * scale.x, v.y * scale.y);
    let v = rotate_vec2(v, rotation_rad);
    pivot + translation + v
}

fn rotate_vec2(v: egui::Vec2, rotation_rad: f32) -> egui::Vec2 {
    let (s, c) = rotation_rad.sin_cos();
    egui::vec2(v.x * c - v.y * s, v.x * s + v.y * c)
}

fn vec2_to_array(v: egui::Vec2) -> [f32; 2] {
    [v.x, v.y]
}

fn layer_offsets_differ(a: egui::Vec2, b: egui::Vec2) -> bool {
    (a - b).length_sq() > 1e-12
}

fn mask_stroke_alpha(opacity: f32) -> u8 {
    (opacity.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn mask_fill_color(
    color_rgb: [u8; 3],
    opacity: f32,
    display_mode: MaskDisplayMode,
) -> Option<egui::Color32> {
    let alpha_scale = match display_mode {
        MaskDisplayMode::OutlineOnly => return None,
        MaskDisplayMode::TranslucentFill => 0.35,
        MaskDisplayMode::FilledPreview => 1.0,
    };
    let a = (opacity.clamp(0.0, 1.0) * alpha_scale * 255.0).round() as u8;
    (a > 0)
        .then(|| egui::Color32::from_rgba_unmultiplied(color_rgb[0], color_rgb[1], color_rgb[2], a))
}

fn paint_filled_polygon(ui: &egui::Ui, points: &[egui::Pos2], fill: egui::Color32) -> bool {
    let Some(clean) = cleaned_mask_fill_points(points) else {
        return false;
    };
    let mut builder = LyonPath::builder();
    builder.begin(lyon_point(clean[0].x, clean[0].y));
    for point in &clean[1..] {
        builder.line_to(lyon_point(point.x, point.y));
    }
    builder.close();
    let path = builder.build();

    let mut tess = FillTessellator::new();
    let mut geometry: VertexBuffers<egui::Pos2, u32> = VertexBuffers::new();
    if tess
        .tessellate_path(
            &path,
            &FillOptions::default(),
            &mut BuffersBuilder::new(&mut geometry, |vertex: FillVertex<'_>| {
                let p = vertex.position();
                egui::pos2(p.x, p.y)
            }),
        )
        .is_err()
        || geometry.indices.is_empty()
    {
        return false;
    }

    let mut mesh = egui::epaint::Mesh::default();
    mesh.indices = geometry.indices;
    mesh.vertices = geometry
        .vertices
        .into_iter()
        .map(|pos| egui::epaint::Vertex {
            pos,
            uv: egui::epaint::WHITE_UV,
            color: fill,
        })
        .collect();
    ui.painter().add(egui::Shape::mesh(mesh));
    true
}

fn cleaned_mask_fill_points(points: &[egui::Pos2]) -> Option<Vec<egui::Pos2>> {
    let mut clean = points
        .iter()
        .copied()
        .filter(|p| p.x.is_finite() && p.y.is_finite())
        .collect::<Vec<_>>();
    if clean.len() >= 2 && clean.first() == clean.last() {
        clean.pop();
    }
    clean.dedup_by(|a, b| a.distance_sq(*b) <= 1e-6);
    (clean.len() >= 3).then_some(clean)
}

fn bounds_for_points(points: &[egui::Pos2]) -> Option<egui::Rect> {
    let mut iter = points
        .iter()
        .copied()
        .filter(|p| p.x.is_finite() && p.y.is_finite());
    let first = iter.next()?;
    let mut min = first;
    let mut max = first;
    for p in iter {
        min.x = min.x.min(p.x);
        min.y = min.y.min(p.y);
        max.x = max.x.max(p.x);
        max.y = max.y.max(p.y);
    }
    Some(egui::Rect::from_min_max(min, max))
}

fn inv_xform_world_point(
    p: egui::Pos2,
    pivot: egui::Pos2,
    translation_world: egui::Vec2,
    scale: egui::Vec2,
    rotation_rad: f32,
) -> egui::Pos2 {
    let mut v = p - pivot - translation_world;
    v = rotate_vec2(v, -rotation_rad);
    let sx = scale.x.abs().max(1e-6);
    let sy = scale.y.abs().max(1e-6);
    v = egui::vec2(v.x / sx, v.y / sy);
    pivot + v
}

fn quad_center(corners: &[egui::Pos2; 4]) -> egui::Pos2 {
    let sum = corners
        .iter()
        .fold(egui::vec2(0.0, 0.0), |acc, &p| acc + p.to_vec2());
    let c = sum * 0.25;
    egui::pos2(c.x, c.y)
}

fn point_in_convex_quad(p: egui::Pos2, corners: &[egui::Pos2; 4]) -> bool {
    // Winding-agnostic: inside if all cross products have the same sign (or are zero).
    let mut has_pos = false;
    let mut has_neg = false;
    for i in 0..4 {
        let a = corners[i];
        let b = corners[(i + 1) % 4];
        let ab = b - a;
        let ap = p - a;
        let c = cross2(ab, ap);
        if c > 0.0 {
            has_pos = true;
        } else if c < 0.0 {
            has_neg = true;
        }
        if has_pos && has_neg {
            return false;
        }
    }
    true
}

fn cross2(a: egui::Vec2, b: egui::Vec2) -> f32 {
    a.x * b.y - a.y * b.x
}

#[cfg(test)]
mod tests;
