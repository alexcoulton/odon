use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use eframe::egui;
use glow::HasContext;
use ndarray::Array2;
use rfd::FileDialog;
use zarrs::array::{Array, ArraySubset};

use crate::annotations::AnnotationPointsLayer;
use crate::camera::Camera;
use crate::imaging::channel_max::{ChannelMaxLoaderHandle, ChannelMaxRequest, spawn_channel_max_loader};
use crate::custom::cell_thresholds::{CellThresholdsAction, CellThresholdsPanel};
use crate::custom::roi_selector::{RoiSelectorAction, RoiSelectorPanel};
use crate::masks::resolve_masks_geojson_path_and_downsample;
use crate::geometry::geojson::{PolygonRingMode, load_geojson_polylines_world};
use crate::imaging::histogram::{HistogramLoaderHandle, HistogramResponse, spawn_histogram_loader};
use crate::ui::icons::{Icon, icon_button};
use crate::render::labels::{LabelZarrDataset, discover_label_names_local};
use crate::render::labels_gl::{LabelDraw, LabelsGl, OutlinesParams};
use crate::render::labels_raw::{
    LabelTileKey, LabelTileLoaderHandle, LabelTileRequest, spawn_label_tile_loader,
};
use crate::project::groups as layer_groups;
use crate::masks::MaskLayer;
use crate::masks::save_mask_layers_geojson;
use crate::app_support::memory::{
    MemoryChannelRow, PendingMemoryAction, SystemMemorySnapshot, format_bytes, memory_risk,
    refresh_system_memory_if_needed, ui_memory_channel_selector, ui_memory_overview,
    ui_pending_memory_action_dialog,
};
use crate::data::ome::retrieve_image_subset_u16;
use crate::data::ome::{ChannelInfo, Dims, OmeZarrDataset};
use crate::imaging::pinned_levels::{PinnedLevelStatus, PinnedLevels};
use crate::render::points::PointsLayer;
use crate::render::points_gl::{PointsGlDrawData, PointsGlDrawParams, PointsGlRenderer};
use crate::data::project_config::ProjectRoi;
use crate::project::{ProjectSpace, ProjectSpaceAction};
use crate::data::remote_store::{
    S3BrowseEntry, S3BrowseListing, S3Browser, S3Store, build_http_store, build_s3_browser,
    build_s3_store, list_s3_prefix,
};
use crate::app_support::repaint as repaint_control;
use crate::app_support::screenshot::{
    ScreenshotRequest, ScreenshotSettings, ScreenshotWorkerHandle, ScreenshotWorkerMsg,
    next_numbered_screenshot_path,
};
use crate::objects::GeoJsonSegmentationLayer;
use crate::objects::ObjectsLayer;
use crate::spatialdata::SpatialImageLayers;
use crate::spatialdata::{SpatialDataElement, SpatialDataTransform2, discover_spatialdata};
use crate::spatialdata::SpatialDataLayers;
use crate::render::threshold_preview_gl::{
    ThresholdPreviewGlDrawData, ThresholdPreviewGlDrawParams, ThresholdPreviewGlRenderer,
};
use crate::geometry::threshold_regions::{
    ThresholdRegionMask, extract_threshold_region_mask, threshold_region_mask_to_polygons,
};
use crate::render::tiles::{
    RenderChannel, TileCache, TileKey, TileRequest, TileResponse, TileWorkerResponse,
    recommended_tile_loader_threads, spawn_tile_loader,
};
use crate::render::tiles_gl::{ChannelDraw, TileDraw, TilesGl};
use crate::render::tiles_raw::{
    RawTileKey, RawTileLoaderHandle, RawTileRequest, RawTileWorkerResponse, spawn_raw_tile_loader,
};
use crate::imaging::tiling::{TileCoord, choose_level_auto, levels_to_draw, tiles_needed_lvl0_rect};
use crate::ui::canvas_overlays;
use crate::ui::channels_panel::{self, ChannelListHost};
use crate::ui::contrast;
use crate::ui::group_layers::{GroupLayersDialog, GroupLayersTarget, default_group_name};
use crate::ui::layer_list;
use crate::ui::left_panel;
use crate::ui::right_panel;
use crate::ui::style::apply_napari_like_dark;
use crate::ui::top_bar;
use crate::xenium::XeniumLayers;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RightTab {
    Properties,
    Analysis,
    Measurements,
    Memory,
    RoiSelector,
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
        if let Some(ch) = self.channels.get_mut(idx) {
            ch.visible = visible;
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
                self.set_active_layer(LayerId::Channel(idx));
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
        self.set_active_layer(LayerId::Channel(idx));
    }

    fn handle_channel_secondary_click(&mut self, idx: usize) {
        if !self.selected_channel_layers.contains(&idx) {
            self.selected_channel_layers.clear();
            self.selected_channel_layers.insert(idx);
            self.channel_select_anchor_idx = Some(idx);
            self.selected_channel_group_id = None;
            self.set_active_layer(LayerId::Channel(idx));
        }
    }

    fn open_group_layers_dialog_channels(&mut self, members: Vec<usize>) {
        Self::open_group_layers_dialog_channels(self, members);
    }

    fn layer_groups(&self) -> crate::data::project_config::ProjectLayerGroups {
        self.project_space.layer_groups().clone()
    }

    fn set_layer_groups(&mut self, groups: crate::data::project_config::ProjectLayerGroups) {
        self.project_space.update_layer_groups(|g| *g = groups);
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
    Pan,
    MoveLayer,
    TransformLayer,
    DrawMaskPolygon,
    RectSelect,
    LassoSelect,
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
    hist_last_sent: Instant,

    camera: Camera,
    active_render_id: u64,
    previous_render_id: Option<u64>,
    last_canvas_rect: Option<egui::Rect>,
    last_target_level: Option<usize>,
    fallback_ceiling_level: Option<usize>,
    last_visible_world_tiles: Option<egui::Rect>,
    zoom_out_floor_level: Option<usize>,
    zoom_out_floor_until: Option<Instant>,
    zoom_out_floor_visible_world_tiles: Option<egui::Rect>,

    auto_level: bool,
    manual_level: usize,
    selected_channel: usize,
    channels: Vec<ChannelInfo>,
    channel_window_overrides: HashMap<String, (f32, f32)>,
    channel_list_search: String,

    active_layer: LayerId,
    selected_channel_layers: HashSet<usize>,
    memory_selected_channels: HashSet<usize>,
    channel_select_anchor_idx: Option<usize>,
    selected_channel_group_id: Option<u64>,
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
    next_annotation_layer_id: u64,
    mask_layers: Vec<MaskLayer>,
    next_mask_layer_id: u64,
    tool_mode: ToolMode,
    drawing_mask_layer: Option<u64>,
    drawing_mask_polygon: Vec<egui::Pos2>,
    selection_rect_start_world: Option<egui::Pos2>,
    selection_rect_current_world: Option<egui::Pos2>,
    selection_lasso_world: Vec<egui::Pos2>,
    threshold_region_min_pixels: usize,
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
    threshold_region_preview_generation: u64,

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
    group_layers_dialog: Option<GroupLayersDialog>,
    hover_tooltip_state: Option<HoverTooltipState>,

    smooth_pixels: bool,
    show_tile_debug: bool,
    show_scale_bar: bool,
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
    points_offset_world: egui::Vec2,
    spatial_points_offset_world: egui::Vec2,
    seg_labels_offset_world: egui::Vec2,
    seg_geojson_offset_world: egui::Vec2,
    seg_objects_offset_world: egui::Vec2,
    xenium_cells_offset_world: egui::Vec2,
    xenium_transcripts_offset_world: egui::Vec2,

    overlay_layer_order: Vec<LayerId>,
    channel_layer_order: Vec<usize>,
    layer_drag: Option<LayerDragState>,
    layer_move: Option<LayerMoveState>,
    layer_transform: Option<LayerTransformState>,
    tiff_plane_state: Option<TiffPlaneState>,
    screenshot_settings: ScreenshotSettings,
    screenshot_settings_open: bool,
    screenshot_worker: ScreenshotWorkerHandle,
    screenshot_next_id: u64,
    screenshot_pending: Option<ScreenshotRequest>,
    screenshot_in_flight: Option<u64>,
    screenshot_output_dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct LayerMoveState {
    layer: LayerId,
    start_offset_world: egui::Vec2,
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
}

#[derive(Debug, Clone)]
pub enum ViewerRequest {
    OpenMosaic(Vec<PathBuf>),
    OpenSingle(PathBuf),
    OpenProjectRoi(ProjectRoi),
    OpenProjectMosaic(Vec<ProjectRoi>),
    OpenRemoteS3Mosaic(Vec<S3DatasetSelection>),
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
    approx_downsample: f32,
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

fn dummy_local_store_for_path(
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

fn build_tiff_dataset(
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
                axis_type: Some("channel".to_string()),
                unit: None,
            },
            crate::data::ome::Axis {
                name: "y".to_string(),
                axis_type: Some("space".to_string()),
                unit: pixel_size_xy
                    .as_ref()
                    .and_then(|(_, units)| units[0].clone()),
            },
            crate::data::ome::Axis {
                name: "x".to_string(),
                axis_type: Some("space".to_string()),
                unit: pixel_size_xy
                    .as_ref()
                    .and_then(|(_, units)| units[1].clone()),
            },
        ]
    } else {
        vec![
            crate::data::ome::Axis {
                name: "y".to_string(),
                axis_type: Some("space".to_string()),
                unit: pixel_size_xy
                    .as_ref()
                    .and_then(|(_, units)| units[0].clone()),
            },
            crate::data::ome::Axis {
                name: "x".to_string(),
                axis_type: Some("space".to_string()),
                unit: pixel_size_xy
                    .as_ref()
                    .and_then(|(_, units)| units[1].clone()),
            },
        ]
    };
    let multiscale = crate::data::ome::Multiscale {
        version: None,
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
        r#type: Some("image".to_string()),
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
    let dims_yx = (dataset.dims.y, dataset.dims.x);
    let store = dummy_local_store_for_path(&dataset_root)?;
    let loader = crate::xenium::spawn_tiff_tile_loader(pyramid.clone(), dims_yx)?;
    let raw_loader = if gpu_available {
        crate::xenium::spawn_tiff_raw_tile_loader(pyramid.clone(), dims_yx).ok()
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
        chanmax_level: choose_default_max_level(&dataset),
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
    pub fn set_show_scale_bar(&mut self, show: bool) {
        self.show_scale_bar = show;
    }

    pub fn set_label_prompt_preference(&mut self, preference: LabelPromptSessionPreference) {
        self.seg_label_prompt_preference = preference;
        if preference == LabelPromptSessionPreference::Ask {
            self.seg_label_prompt_always = false;
        }
    }

    pub fn label_prompt_preference(&self) -> LabelPromptSessionPreference {
        self.seg_label_prompt_preference
    }

    fn default_tile_loader_threads() -> usize {
        recommended_tile_loader_threads()
    }

    fn supports_runtime_tile_loader_tuning(&self) -> bool {
        self.tiff_plane_state.is_none()
    }

    fn respawn_tile_loaders(&mut self) -> anyhow::Result<()> {
        if !self.supports_runtime_tile_loader_tuning() {
            anyhow::bail!("runtime tile loading settings are not available for this dataset");
        }
        let dims_cyx = (
            self.dataset.dims.c,
            self.dataset.dims.y,
            self.dataset.dims.x,
        );
        self.loader = spawn_tile_loader(
            self.store.clone(),
            self.dataset.levels.iter().map(|l| l.path.clone()).collect(),
            self.dataset
                .levels
                .iter()
                .map(|l| l.shape.clone())
                .collect(),
            self.dataset
                .levels
                .iter()
                .map(|l| l.chunks.clone())
                .collect(),
            self.dataset
                .levels
                .iter()
                .map(|l| l.dtype.clone())
                .collect(),
            dims_cyx,
            self.tile_loader_threads,
        )?;
        self.raw_loader = if self.tiles_gl.is_some() {
            Some(spawn_raw_tile_loader(
                self.store.clone(),
                self.dataset.levels.iter().map(|l| l.path.clone()).collect(),
                self.dataset
                    .levels
                    .iter()
                    .map(|l| l.shape.clone())
                    .collect(),
                self.dataset
                    .levels
                    .iter()
                    .map(|l| l.chunks.clone())
                    .collect(),
                self.dataset
                    .levels
                    .iter()
                    .map(|l| l.dtype.clone())
                    .collect(),
                dims_cyx,
                self.tile_loader_threads,
            )?)
        } else {
            None
        };
        self.pending.clear();
        if let Some(tiles_gl) = self.tiles_gl.as_ref() {
            tiles_gl.reset();
        }
        self.cache = TileCache::new(256);
        self.bump_render_id();
        Ok(())
    }

    fn try_get_raw_tile_from_pinned_finer(
        &self,
        key: RawTileKey,
        level: &crate::data::ome::LevelInfo,
    ) -> Option<crate::render::tiles_raw::RawTileResponse> {
        if !self.prefer_pinned_finer_levels {
            return None;
        }
        for source_level in 0..key.level {
            let Some(source_info) = self.dataset.levels.get(source_level) else {
                continue;
            };
            if let Some(resp) = self.pinned_levels.try_get_raw_tile_resampled_from_level(
                source_level,
                key,
                &self.dataset.dims,
                level,
                source_info,
            ) {
                return Some(resp);
            }
        }
        None
    }

    fn try_get_composited_tile_from_pinned_finer(
        &self,
        key: TileKey,
        channels: &[RenderChannel],
        level: &crate::data::ome::LevelInfo,
    ) -> Option<TileResponse> {
        if !self.prefer_pinned_finer_levels {
            return None;
        }
        for source_level in 0..key.level {
            let Some(source_info) = self.dataset.levels.get(source_level) else {
                continue;
            };
            if let Some(tile) = self
                .pinned_levels
                .try_get_composited_tile_resampled_from_level(
                    source_level,
                    key,
                    channels,
                    &self.dataset.dims,
                    level,
                    source_info,
                )
            {
                return Some(tile);
            }
        }
        None
    }

    fn axis_unit_to_um(unit: &str) -> Option<f32> {
        let u = unit.trim().to_ascii_lowercase();
        match u.as_str() {
            "um" | "µm" | "micrometer" | "micrometre" | "micron" | "microns" => Some(1.0),
            "nm" | "nanometer" | "nanometre" | "nanometers" | "nanometres" => Some(0.001),
            "mm" | "millimeter" | "millimetre" | "millimeters" | "millimetres" => Some(1000.0),
            "m" | "meter" | "metre" | "meters" | "metres" => Some(1_000_000.0),
            _ => None,
        }
    }

    fn dataset_pixel_size_um(&self) -> Option<f32> {
        let ms = &self.dataset.multiscale;
        let ds0 = ms.datasets.first()?;
        let mut scale: Option<&[f32]> = None;
        for ct in &ds0.coordinate_transformations {
            if let crate::data::ome::CoordTransform::Scale { scale: s } = ct {
                scale = Some(s.as_slice());
                break;
            }
        }
        let scale = scale?;
        if scale.len() != self.dataset.dims.ndim {
            return None;
        }

        let ax_x = ms.axes.get(self.dataset.dims.x)?;
        let ax_y = ms.axes.get(self.dataset.dims.y)?;
        let fx = ax_x.unit.as_deref().and_then(Self::axis_unit_to_um)?;
        let fy = ax_y.unit.as_deref().and_then(Self::axis_unit_to_um)?;
        let sx = scale[self.dataset.dims.x] * fx;
        let sy = scale[self.dataset.dims.y] * fy;
        if !(sx.is_finite() && sy.is_finite() && sx > 0.0 && sy > 0.0) {
            return None;
        }
        Some((sx + sy) * 0.5)
    }

    pub fn new(
        cc: &eframe::CreationContext<'_>,
        dataset: OmeZarrDataset,
        store: Arc<dyn zarrs::storage::ReadableStorageTraits>,
    ) -> Self {
        apply_napari_like_dark(&cc.egui_ctx);

        let level_paths: Vec<String> = dataset.levels.iter().map(|l| l.path.clone()).collect();
        let level_shapes: Vec<Vec<u64>> = dataset.levels.iter().map(|l| l.shape.clone()).collect();
        let level_chunks: Vec<Vec<u64>> = dataset.levels.iter().map(|l| l.chunks.clone()).collect();
        let level_dtypes: Vec<String> = dataset.levels.iter().map(|l| l.dtype.clone()).collect();
        let dims_cyx = (dataset.dims.c, dataset.dims.y, dataset.dims.x);
        let tile_loader_threads = Self::default_tile_loader_threads();

        let loader = spawn_tile_loader(
            store.clone(),
            level_paths,
            level_shapes,
            level_chunks,
            level_dtypes.clone(),
            dims_cyx,
            tile_loader_threads,
        )
        .expect("failed to spawn tile loader");

        let (raw_loader, tiles_gl) = if cc.gl.is_some() {
            let raw = spawn_raw_tile_loader(
                store.clone(),
                dataset.levels.iter().map(|l| l.path.clone()).collect(),
                dataset.levels.iter().map(|l| l.shape.clone()).collect(),
                dataset.levels.iter().map(|l| l.chunks.clone()).collect(),
                dataset.levels.iter().map(|l| l.dtype.clone()).collect(),
                dims_cyx,
                tile_loader_threads,
            )
            .ok();
            (raw, Some(TilesGl::new(RAW_TILE_CACHE_CAPACITY_TILES)))
        } else {
            (None, None)
        };

        let seg_label_names = dataset
            .source
            .local_path()
            .map(discover_label_names_local)
            .unwrap_or_default();
        let seg_label_selected = if seg_label_names.iter().any(|n| n == "cells") {
            "cells".to_string()
        } else if let Some(first) = seg_label_names.first() {
            first.clone()
        } else {
            "cells".to_string()
        };
        let seg_label_input = seg_label_selected.clone();
        let seg_label_status = String::new();
        let seg_label_prompt_open = cc.gl.is_some() && !seg_label_names.is_empty();

        let (label_cells, label_loader, label_cells_xform, labels_gl) = if cc.gl.is_some() {
            // Labels are not auto-loaded; we prompt on open if any are present.
            (None, None, None, Some(LabelsGl::new(1024)))
        } else {
            (None, None, None, None)
        };

        let hist_loader = spawn_histogram_loader(
            store.clone(),
            dataset.levels.iter().map(|l| l.path.clone()).collect(),
            dataset.levels.iter().map(|l| l.shape.clone()).collect(),
            dataset.levels.iter().map(|l| l.dtype.clone()).collect(),
            dims_cyx,
        )
        .expect("failed to spawn histogram loader");

        let chanmax_level = choose_default_max_level(&dataset);
        let chanmax_loader = spawn_channel_max_loader(
            store.clone(),
            dataset.levels.iter().map(|l| l.path.clone()).collect(),
            dataset.levels.iter().map(|l| l.shape.clone()).collect(),
            dataset.levels.iter().map(|l| l.chunks.clone()).collect(),
            dataset.levels.iter().map(|l| l.dtype.clone()).collect(),
            dims_cyx,
        )
        .expect("failed to spawn channel max loader");

        let mut camera = Camera::default();
        camera.center_world_lvl0 = egui::pos2(0.0, 0.0);
        camera.zoom_screen_per_lvl0_px = 0.1;

        let mut app = Self {
            dataset: dataset.clone(),
            store: store.clone(),
            remote_runtime: None,
            loader,
            raw_loader,
            label_cells,
            label_loader,
            label_cells_xform,
            seg_label_names,
            seg_label_selected,
            seg_label_input,
            seg_label_status,
            seg_label_prompt_open,
            seg_label_prompt_always: false,
            seg_label_prompt_preference: LabelPromptSessionPreference::Ask,
            hist_loader,
            chanmax_loader,
            chanmax_request_id: 1,
            chanmax_level,
            chanmax_pending: vec![true; dataset.channels.len()],
            chanmax_snapshot: dataset.channels.iter().map(|c| c.window).collect(),
            cache: TileCache::new(256),
            pending: Vec::new(),
            hist: None,
            hist_request_id: 0,
            hist_request_pending: false,
            hist_dirty: true,
            hist_last_sent: Instant::now()
                .checked_sub(Duration::from_secs(3600))
                .unwrap_or_else(Instant::now),
            camera,
            active_render_id: 1,
            previous_render_id: None,
            last_canvas_rect: None,
            last_target_level: None,
            fallback_ceiling_level: None,
            last_visible_world_tiles: None,
            zoom_out_floor_level: None,
            zoom_out_floor_until: None,
            zoom_out_floor_visible_world_tiles: None,
            auto_level: true,
            manual_level: 0,
            selected_channel: 0,
            channels: dataset.channels.clone(),
            channel_window_overrides: HashMap::new(),
            channel_list_search: String::new(),

            active_layer: if dataset.channels.is_empty() {
                LayerId::Points
            } else {
                LayerId::Channel(0)
            },
            selected_channel_layers: if dataset.channels.is_empty() {
                HashSet::new()
            } else {
                HashSet::from([0usize])
            },
            memory_selected_channels: (0..dataset.channels.len()).collect(),
            channel_select_anchor_idx: None,
            selected_channel_group_id: None,
            selected_overlay_layers: HashSet::new(),
            overlay_select_anchor_pos: None,
            show_left_panel: true,
            show_right_panel: true,
            close_dialog_open: false,
            pinned_levels: PinnedLevels::new(),
            pending_memory_load: None,
            memory_status: String::new(),
            system_memory: None,
            system_memory_last_refresh: None,
            left_tab: LeftTab::Layers,
            right_tab: RightTab::Properties,
            project_space: ProjectSpace::default(),
            project_cfg_seen: 0,
            roi_selector: RoiSelectorPanel::new(&dataset.source),
            cell_thresholds: CellThresholdsPanel::new(
                dataset
                    .source
                    .local_path()
                    .unwrap_or_else(|| std::path::Path::new("")),
                dataset.multiscale.name.as_deref(),
            ),
            cell_points: PointsLayer::new("cell_centroids"),
            annotation_layers: Vec::new(),
            next_annotation_layer_id: 1,
            mask_layers: Vec::new(),
            next_mask_layer_id: 1,
            tool_mode: ToolMode::Pan,
            drawing_mask_layer: None,
            drawing_mask_polygon: Vec::new(),
            selection_rect_start_world: None,
            selection_rect_current_world: None,
            selection_lasso_world: Vec::new(),
            threshold_region_min_pixels: 32,
            threshold_region_status: String::new(),
            threshold_region_preview: None,
            cells_outlines_visible: true,
            cells_outlines_color_rgb: [0, 255, 0],
            cells_outlines_opacity: 0.75,
            cells_outlines_width_px: 0.0,
            points_gl: cc.gl.as_ref().map(|_| PointsGlRenderer::default()),
            threshold_preview_gl: cc
                .gl
                .as_ref()
                .map(|_| ThresholdPreviewGlRenderer::default()),
            tiles_gl,
            labels_gl,

            remote_dialog_open: false,
            remote_mode: RemoteMode::Http,
            remote_http_url: String::new(),
            remote_s3_endpoint: String::new(),
            remote_s3_region: "auto".to_string(),
            remote_s3_bucket: String::new(),
            remote_s3_prefix: String::new(),
            remote_s3_access_key: String::new(),
            remote_s3_secret_key: String::new(),
            remote_status: String::new(),
            remote_s3_browser: None,

            pending_request: None,
            group_layers_dialog: None,
            hover_tooltip_state: None,
            smooth_pixels: true,
            show_tile_debug: false,
            show_scale_bar: true,
            tile_loader_threads,
            tile_prefetch_mode: TilePrefetchMode::TargetHalo,
            tile_prefetch_aggressiveness: TilePrefetchAggressiveness::Balanced,
            tile_loading_status: String::new(),
            prefer_pinned_finer_levels: false,

            seg_geojson: GeoJsonSegmentationLayer::default(),
            seg_objects: ObjectsLayer::default(),
            spatial_image_layers: SpatialImageLayers::default(),
            spatial_layers: SpatialDataLayers::default(),
            spatial_image_transform: SpatialDataTransform2::default(),
            spatial_label_transform: SpatialDataTransform2::default(),
            spatial_root: None,
            spatial_label_store: None,
            xenium_layers: XeniumLayers::default(),

            channel_offsets_world: vec![egui::Vec2::ZERO; dataset.channels.len()],
            channel_scales: vec![egui::Vec2::splat(1.0); dataset.channels.len()],
            channel_rotations_rad: vec![0.0; dataset.channels.len()],
            points_offset_world: egui::Vec2::ZERO,
            spatial_points_offset_world: egui::Vec2::ZERO,
            seg_labels_offset_world: egui::Vec2::ZERO,
            seg_geojson_offset_world: egui::Vec2::ZERO,
            seg_objects_offset_world: egui::Vec2::ZERO,
            xenium_cells_offset_world: egui::Vec2::ZERO,
            xenium_transcripts_offset_world: egui::Vec2::ZERO,

            overlay_layer_order: Vec::new(),
            channel_layer_order: (0..dataset.channels.len()).collect(),
            layer_drag: None,
            layer_move: None,
            layer_transform: None,
            tiff_plane_state: None,
            threshold_region_preview_generation: 1,
            screenshot_settings: ScreenshotSettings::default(),
            screenshot_settings_open: false,
            screenshot_worker: ScreenshotWorkerHandle::spawn(),
            screenshot_next_id: 1,
            screenshot_pending: None,
            screenshot_in_flight: None,
            screenshot_output_dir: None,
        };

        app.configure_root_label_dataset_if_needed();
        app.rebuild_layer_orders();
        app.request_default_channel_maxes();
        app.active_render_id = app.compute_render_id();

        // Initial fit (best effort).
        let shape0 = &app.dataset.levels[0].shape;
        let y = shape0[app.dataset.dims.y] as f32;
        let x = shape0[app.dataset.dims.x] as f32;
        let world = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(x, y));
        if let Some(viewport) = cc.egui_ctx.input(|i| i.viewport().inner_rect) {
            app.camera.fit_to_world_rect(viewport, world);
        }

        app
    }

    pub fn new_runtime(
        ctx: &egui::Context,
        gpu_available: bool,
        dataset: OmeZarrDataset,
        store: Arc<dyn zarrs::storage::ReadableStorageTraits>,
    ) -> Self {
        apply_napari_like_dark(ctx);

        let level_paths: Vec<String> = dataset.levels.iter().map(|l| l.path.clone()).collect();
        let level_shapes: Vec<Vec<u64>> = dataset.levels.iter().map(|l| l.shape.clone()).collect();
        let level_chunks: Vec<Vec<u64>> = dataset.levels.iter().map(|l| l.chunks.clone()).collect();
        let level_dtypes: Vec<String> = dataset.levels.iter().map(|l| l.dtype.clone()).collect();
        let dims_cyx = (dataset.dims.c, dataset.dims.y, dataset.dims.x);
        let tile_loader_threads = Self::default_tile_loader_threads();

        let loader = spawn_tile_loader(
            store.clone(),
            level_paths,
            level_shapes,
            level_chunks,
            level_dtypes.clone(),
            dims_cyx,
            tile_loader_threads,
        )
        .expect("failed to spawn tile loader");

        let (raw_loader, tiles_gl) = if gpu_available {
            let raw = spawn_raw_tile_loader(
                store.clone(),
                dataset.levels.iter().map(|l| l.path.clone()).collect(),
                dataset.levels.iter().map(|l| l.shape.clone()).collect(),
                dataset.levels.iter().map(|l| l.chunks.clone()).collect(),
                dataset.levels.iter().map(|l| l.dtype.clone()).collect(),
                dims_cyx,
                tile_loader_threads,
            )
            .ok();
            (raw, Some(TilesGl::new(RAW_TILE_CACHE_CAPACITY_TILES)))
        } else {
            (None, None)
        };

        let seg_label_names = dataset
            .source
            .local_path()
            .map(discover_label_names_local)
            .unwrap_or_default();
        let seg_label_selected = if seg_label_names.iter().any(|n| n == "cells") {
            "cells".to_string()
        } else if let Some(first) = seg_label_names.first() {
            first.clone()
        } else {
            "cells".to_string()
        };
        let seg_label_input = seg_label_selected.clone();
        let seg_label_status = String::new();
        let seg_label_prompt_open = gpu_available && !seg_label_names.is_empty();

        let (label_cells, label_loader, label_cells_xform, labels_gl) = if gpu_available {
            // Labels are not auto-loaded; we prompt on open if any are present.
            (None, None, None, Some(LabelsGl::new(1024)))
        } else {
            (None, None, None, None)
        };

        let hist_loader = spawn_histogram_loader(
            store.clone(),
            dataset.levels.iter().map(|l| l.path.clone()).collect(),
            dataset.levels.iter().map(|l| l.shape.clone()).collect(),
            dataset.levels.iter().map(|l| l.dtype.clone()).collect(),
            dims_cyx,
        )
        .expect("failed to spawn histogram loader");

        let chanmax_level = choose_default_max_level(&dataset);
        let chanmax_loader = spawn_channel_max_loader(
            store.clone(),
            dataset.levels.iter().map(|l| l.path.clone()).collect(),
            dataset.levels.iter().map(|l| l.shape.clone()).collect(),
            dataset.levels.iter().map(|l| l.chunks.clone()).collect(),
            dataset.levels.iter().map(|l| l.dtype.clone()).collect(),
            dims_cyx,
        )
        .expect("failed to spawn channel max loader");

        let mut camera = Camera::default();
        camera.center_world_lvl0 = egui::pos2(0.0, 0.0);
        camera.zoom_screen_per_lvl0_px = 0.1;

        let mut app = Self {
            dataset: dataset.clone(),
            store: store.clone(),
            remote_runtime: None,
            loader,
            raw_loader,
            label_cells,
            label_loader,
            label_cells_xform,
            seg_label_names,
            seg_label_selected,
            seg_label_input,
            seg_label_status,
            seg_label_prompt_open,
            seg_label_prompt_always: false,
            seg_label_prompt_preference: LabelPromptSessionPreference::Ask,
            hist_loader,
            chanmax_loader,
            chanmax_request_id: 1,
            chanmax_level,
            chanmax_pending: vec![true; dataset.channels.len()],
            chanmax_snapshot: dataset.channels.iter().map(|c| c.window).collect(),
            cache: TileCache::new(256),
            pending: Vec::new(),
            hist: None,
            hist_request_id: 0,
            hist_request_pending: false,
            hist_dirty: true,
            hist_last_sent: Instant::now()
                .checked_sub(Duration::from_secs(3600))
                .unwrap_or_else(Instant::now),
            camera,
            active_render_id: 1,
            previous_render_id: None,
            last_canvas_rect: None,
            last_target_level: None,
            fallback_ceiling_level: None,
            last_visible_world_tiles: None,
            zoom_out_floor_level: None,
            zoom_out_floor_until: None,
            zoom_out_floor_visible_world_tiles: None,
            auto_level: true,
            manual_level: 0,
            selected_channel: 0,
            channels: dataset.channels.clone(),
            channel_window_overrides: HashMap::new(),
            channel_list_search: String::new(),

            active_layer: if dataset.channels.is_empty() {
                LayerId::Points
            } else {
                LayerId::Channel(0)
            },
            selected_channel_layers: if dataset.channels.is_empty() {
                HashSet::new()
            } else {
                HashSet::from([0usize])
            },
            memory_selected_channels: (0..dataset.channels.len()).collect(),
            channel_select_anchor_idx: None,
            selected_channel_group_id: None,
            selected_overlay_layers: HashSet::new(),
            overlay_select_anchor_pos: None,
            show_left_panel: true,
            show_right_panel: true,
            close_dialog_open: false,
            pinned_levels: PinnedLevels::new(),
            pending_memory_load: None,
            memory_status: String::new(),
            system_memory: None,
            system_memory_last_refresh: None,
            left_tab: LeftTab::Layers,
            right_tab: RightTab::Properties,
            project_space: ProjectSpace::default(),
            project_cfg_seen: 0,
            roi_selector: RoiSelectorPanel::new(&dataset.source),
            cell_thresholds: CellThresholdsPanel::new(
                dataset
                    .source
                    .local_path()
                    .unwrap_or_else(|| std::path::Path::new("")),
                dataset.multiscale.name.as_deref(),
            ),
            cell_points: PointsLayer::new("cell_centroids"),
            annotation_layers: Vec::new(),
            next_annotation_layer_id: 1,
            mask_layers: Vec::new(),
            next_mask_layer_id: 1,
            tool_mode: ToolMode::Pan,
            drawing_mask_layer: None,
            drawing_mask_polygon: Vec::new(),
            selection_rect_start_world: None,
            selection_rect_current_world: None,
            selection_lasso_world: Vec::new(),
            threshold_region_min_pixels: 32,
            threshold_region_status: String::new(),
            threshold_region_preview: None,
            cells_outlines_visible: true,
            cells_outlines_color_rgb: [0, 255, 0],
            cells_outlines_opacity: 0.75,
            cells_outlines_width_px: 0.0,
            points_gl: gpu_available.then_some(PointsGlRenderer::default()),
            threshold_preview_gl: gpu_available.then_some(ThresholdPreviewGlRenderer::default()),
            tiles_gl,
            labels_gl,

            remote_dialog_open: false,
            remote_mode: RemoteMode::Http,
            remote_http_url: String::new(),
            remote_s3_endpoint: String::new(),
            remote_s3_region: "auto".to_string(),
            remote_s3_bucket: String::new(),
            remote_s3_prefix: String::new(),
            remote_s3_access_key: String::new(),
            remote_s3_secret_key: String::new(),
            remote_status: String::new(),
            remote_s3_browser: None,

            pending_request: None,
            group_layers_dialog: None,
            hover_tooltip_state: None,
            smooth_pixels: true,
            show_tile_debug: false,
            show_scale_bar: true,
            tile_loader_threads: Self::default_tile_loader_threads(),
            tile_prefetch_mode: TilePrefetchMode::TargetHalo,
            tile_prefetch_aggressiveness: TilePrefetchAggressiveness::Balanced,
            tile_loading_status: String::new(),
            prefer_pinned_finer_levels: false,

            seg_geojson: GeoJsonSegmentationLayer::default(),
            seg_objects: ObjectsLayer::default(),
            spatial_image_layers: SpatialImageLayers::default(),
            spatial_layers: SpatialDataLayers::default(),
            spatial_image_transform: SpatialDataTransform2::default(),
            spatial_label_transform: SpatialDataTransform2::default(),
            spatial_root: None,
            spatial_label_store: None,
            xenium_layers: XeniumLayers::default(),

            channel_offsets_world: vec![egui::Vec2::ZERO; dataset.channels.len()],
            channel_scales: vec![egui::Vec2::splat(1.0); dataset.channels.len()],
            channel_rotations_rad: vec![0.0; dataset.channels.len()],
            points_offset_world: egui::Vec2::ZERO,
            spatial_points_offset_world: egui::Vec2::ZERO,
            seg_labels_offset_world: egui::Vec2::ZERO,
            seg_geojson_offset_world: egui::Vec2::ZERO,
            seg_objects_offset_world: egui::Vec2::ZERO,
            xenium_cells_offset_world: egui::Vec2::ZERO,
            xenium_transcripts_offset_world: egui::Vec2::ZERO,

            overlay_layer_order: Vec::new(),
            channel_layer_order: (0..dataset.channels.len()).collect(),
            layer_drag: None,
            layer_move: None,
            layer_transform: None,
            tiff_plane_state: None,
            threshold_region_preview_generation: 1,
            screenshot_settings: ScreenshotSettings::default(),
            screenshot_settings_open: false,
            screenshot_worker: ScreenshotWorkerHandle::spawn(),
            screenshot_next_id: 1,
            screenshot_pending: None,
            screenshot_in_flight: None,
            screenshot_output_dir: None,
        };

        app.configure_root_label_dataset_if_needed();
        app.rebuild_layer_orders();
        app.request_default_channel_maxes();
        app.active_render_id = app.compute_render_id();

        // Initial fit (best effort).
        let shape0 = &app.dataset.levels[0].shape;
        let y = shape0[app.dataset.dims.y] as f32;
        let x = shape0[app.dataset.dims.x] as f32;
        let world = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(x, y));
        if let Some(viewport) = ctx.input(|i| i.viewport().inner_rect) {
            app.camera.fit_to_world_rect(viewport, world);
        }

        app
    }

    pub fn new_xenium_runtime(
        ctx: &egui::Context,
        gpu_available: bool,
        dataset_root: PathBuf,
        morphology_mip_tiff: PathBuf,
        cells_zarr_zip: Option<PathBuf>,
        transcripts_zarr_zip: Option<PathBuf>,
        pixel_size_um: f32,
    ) -> anyhow::Result<Self> {
        apply_napari_like_dark(ctx);

        let mut app = Self::new_tiff_runtime_named(
            ctx,
            gpu_available,
            dataset_root.clone(),
            morphology_mip_tiff,
            "xenium".to_string(),
            "morphology".to_string(),
        )?;
        app.attach_xenium_layers(
            dataset_root,
            cells_zarr_zip,
            transcripts_zarr_zip,
            pixel_size_um,
        );
        // Xenium default: cells ON, transcripts layer present but OFF until a gene is typed.
        if let Some(c) = app.xenium_layers.cells.as_mut() {
            c.visible = true;
        }
        if let Some(t) = app.xenium_layers.transcripts.as_mut() {
            t.visible = false;
        }
        Ok(app)
    }

    pub fn new_tiff_runtime(
        ctx: &egui::Context,
        gpu_available: bool,
        image_path: PathBuf,
    ) -> anyhow::Result<Self> {
        apply_napari_like_dark(ctx);

        let dataset_name = image_path
            .file_stem()
            .and_then(|s| s.to_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("tiff")
            .to_string();

        Self::new_tiff_runtime_named(
            ctx,
            gpu_available,
            image_path.clone(),
            image_path,
            dataset_name,
            "image".to_string(),
        )
    }

    fn new_tiff_runtime_named(
        ctx: &egui::Context,
        gpu_available: bool,
        dataset_root: PathBuf,
        image_path: PathBuf,
        dataset_name: String,
        channel_name: String,
    ) -> anyhow::Result<Self> {
        crate::log_info!(
            "open tiff: root={} image={}",
            dataset_root.to_string_lossy(),
            image_path.to_string_lossy()
        );
        let assets = build_tiff_runtime_assets(
            gpu_available,
            dataset_root,
            image_path,
            dataset_name,
            channel_name,
            crate::xenium::TiffPlaneSelection { z: 0, t: 0 },
        )?;
        let tiles_gl = gpu_available.then(|| TilesGl::new(RAW_TILE_CACHE_CAPACITY_TILES));

        let mut app = Self::new_runtime_with_handles(
            ctx,
            gpu_available,
            assets.dataset,
            assets.store,
            assets.loader,
            assets.raw_loader,
            tiles_gl,
            assets.hist_loader,
            assets.chanmax_loader,
            assets.chanmax_level,
        );
        app.tiff_plane_state = assets.tiff_plane_state;
        Ok(app)
    }

    fn new_runtime_with_handles(
        ctx: &egui::Context,
        gpu_available: bool,
        dataset: OmeZarrDataset,
        store: Arc<dyn zarrs::storage::ReadableStorageTraits>,
        loader: crate::render::tiles::TileLoaderHandle,
        raw_loader: Option<RawTileLoaderHandle>,
        tiles_gl: Option<TilesGl>,
        hist_loader: HistogramLoaderHandle,
        chanmax_loader: ChannelMaxLoaderHandle,
        chanmax_level: usize,
    ) -> Self {
        let mut camera = Camera::default();
        camera.center_world_lvl0 = egui::pos2(0.0, 0.0);
        camera.zoom_screen_per_lvl0_px = 0.1;

        let seg_label_names = dataset
            .source
            .local_path()
            .map(discover_label_names_local)
            .unwrap_or_default();
        let seg_label_selected = if seg_label_names.iter().any(|n| n == "cells") {
            "cells".to_string()
        } else if let Some(first) = seg_label_names.first() {
            first.clone()
        } else {
            "cells".to_string()
        };
        let seg_label_input = seg_label_selected.clone();
        let seg_label_status = String::new();
        let seg_label_prompt_open = gpu_available && !seg_label_names.is_empty();

        let (label_cells, label_loader, label_cells_xform, labels_gl) = if gpu_available {
            (None, None, None, Some(LabelsGl::new(1024)))
        } else {
            (None, None, None, None)
        };

        let mut app = Self {
            dataset: dataset.clone(),
            store: store.clone(),
            remote_runtime: None,
            loader,
            raw_loader,
            label_cells,
            label_loader,
            label_cells_xform,
            seg_label_names,
            seg_label_selected,
            seg_label_input,
            seg_label_status,
            seg_label_prompt_open,
            seg_label_prompt_always: false,
            seg_label_prompt_preference: LabelPromptSessionPreference::Ask,
            hist_loader,
            chanmax_loader,
            chanmax_request_id: 1,
            chanmax_level,
            chanmax_pending: vec![false; dataset.channels.len()],
            chanmax_snapshot: dataset.channels.iter().map(|c| c.window).collect(),
            cache: TileCache::new(256),
            pending: Vec::new(),
            hist: None,
            hist_request_id: 0,
            hist_request_pending: false,
            hist_dirty: true,
            hist_last_sent: Instant::now()
                .checked_sub(Duration::from_secs(3600))
                .unwrap_or_else(Instant::now),
            camera,
            active_render_id: 1,
            previous_render_id: None,
            last_canvas_rect: None,
            last_target_level: None,
            fallback_ceiling_level: None,
            last_visible_world_tiles: None,
            zoom_out_floor_level: None,
            zoom_out_floor_until: None,
            zoom_out_floor_visible_world_tiles: None,
            auto_level: true,
            manual_level: 0,
            selected_channel: 0,
            channels: dataset.channels.clone(),
            channel_window_overrides: HashMap::new(),
            channel_list_search: String::new(),
            active_layer: if dataset.channels.is_empty() {
                LayerId::Points
            } else {
                LayerId::Channel(0)
            },
            selected_channel_layers: if dataset.channels.is_empty() {
                HashSet::new()
            } else {
                HashSet::from([0usize])
            },
            memory_selected_channels: (0..dataset.channels.len()).collect(),
            channel_select_anchor_idx: None,
            selected_channel_group_id: None,
            selected_overlay_layers: HashSet::new(),
            overlay_select_anchor_pos: None,
            show_left_panel: true,
            show_right_panel: true,
            close_dialog_open: false,
            pinned_levels: PinnedLevels::new(),
            pending_memory_load: None,
            memory_status: String::new(),
            system_memory: None,
            system_memory_last_refresh: None,
            left_tab: LeftTab::Layers,
            right_tab: RightTab::Properties,
            project_space: ProjectSpace::default(),
            project_cfg_seen: 0,
            roi_selector: RoiSelectorPanel::new(&dataset.source),
            cell_thresholds: CellThresholdsPanel::new(
                dataset
                    .source
                    .local_path()
                    .unwrap_or_else(|| std::path::Path::new("")),
                dataset.multiscale.name.as_deref(),
            ),
            cell_points: PointsLayer::new("cell_centroids"),
            annotation_layers: Vec::new(),
            next_annotation_layer_id: 1,
            mask_layers: Vec::new(),
            next_mask_layer_id: 1,
            tool_mode: ToolMode::Pan,
            drawing_mask_layer: None,
            drawing_mask_polygon: Vec::new(),
            selection_rect_start_world: None,
            selection_rect_current_world: None,
            selection_lasso_world: Vec::new(),
            threshold_region_min_pixels: 32,
            threshold_region_status: String::new(),
            threshold_region_preview: None,
            cells_outlines_visible: true,
            cells_outlines_color_rgb: [0, 255, 0],
            cells_outlines_opacity: 0.75,
            cells_outlines_width_px: 0.0,
            points_gl: gpu_available.then(|| PointsGlRenderer::default()),
            threshold_preview_gl: gpu_available.then(|| ThresholdPreviewGlRenderer::default()),
            tiles_gl,
            labels_gl,
            remote_dialog_open: false,
            remote_mode: RemoteMode::Http,
            remote_http_url: String::new(),
            remote_s3_endpoint: String::new(),
            remote_s3_region: "auto".to_string(),
            remote_s3_bucket: String::new(),
            remote_s3_prefix: String::new(),
            remote_s3_access_key: String::new(),
            remote_s3_secret_key: String::new(),
            remote_status: String::new(),
            remote_s3_browser: None,
            pending_request: None,
            group_layers_dialog: None,
            hover_tooltip_state: None,
            smooth_pixels: true,
            show_tile_debug: false,
            show_scale_bar: true,
            tile_loader_threads: Self::default_tile_loader_threads(),
            tile_prefetch_mode: TilePrefetchMode::TargetHalo,
            tile_prefetch_aggressiveness: TilePrefetchAggressiveness::Balanced,
            tile_loading_status: String::new(),
            prefer_pinned_finer_levels: false,
            seg_geojson: GeoJsonSegmentationLayer::default(),
            seg_objects: ObjectsLayer::default(),
            spatial_image_layers: SpatialImageLayers::default(),
            spatial_layers: SpatialDataLayers::default(),
            spatial_image_transform: SpatialDataTransform2::default(),
            spatial_label_transform: SpatialDataTransform2::default(),
            spatial_root: None,
            spatial_label_store: None,
            xenium_layers: XeniumLayers::default(),
            channel_offsets_world: vec![egui::Vec2::ZERO; dataset.channels.len()],
            channel_scales: vec![egui::Vec2::splat(1.0); dataset.channels.len()],
            channel_rotations_rad: vec![0.0; dataset.channels.len()],
            points_offset_world: egui::Vec2::ZERO,
            spatial_points_offset_world: egui::Vec2::ZERO,
            seg_labels_offset_world: egui::Vec2::ZERO,
            seg_geojson_offset_world: egui::Vec2::ZERO,
            seg_objects_offset_world: egui::Vec2::ZERO,
            xenium_cells_offset_world: egui::Vec2::ZERO,
            xenium_transcripts_offset_world: egui::Vec2::ZERO,
            overlay_layer_order: Vec::new(),
            channel_layer_order: (0..dataset.channels.len()).collect(),
            layer_drag: None,
            layer_move: None,
            layer_transform: None,
            tiff_plane_state: None,
            threshold_region_preview_generation: 1,
            screenshot_settings: ScreenshotSettings::default(),
            screenshot_settings_open: false,
            screenshot_worker: ScreenshotWorkerHandle::spawn(),
            screenshot_next_id: 1,
            screenshot_pending: None,
            screenshot_in_flight: None,
            screenshot_output_dir: None,
        };

        app.configure_root_label_dataset_if_needed();
        app.rebuild_layer_orders();
        app.active_render_id = app.compute_render_id();

        // Initial fit.
        if let Some(shape0) = app.dataset.levels.get(0).map(|l| l.shape.clone()) {
            let y = shape0[app.dataset.dims.y] as f32;
            let x = shape0[app.dataset.dims.x] as f32;
            let world = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(x, y));
            if let Some(viewport) = ctx.input(|i| i.viewport().inner_rect) {
                app.camera.fit_to_world_rect(viewport, world);
            }
        }

        app
    }

    pub fn take_request(&mut self) -> Option<ViewerRequest> {
        self.pending_request.take()
    }

    pub fn take_project_space(&mut self) -> ProjectSpace {
        self.sync_mask_layers_into_project_space();
        std::mem::take(&mut self.project_space)
    }

    pub fn set_project_space(&mut self, project_space: ProjectSpace) {
        self.project_space = project_space;
        self.project_cfg_seen = u64::MAX;
        self.restore_mask_layers_from_project_space();
        self.auto_load_project_roi_segmentation();
    }

    pub fn set_remote_runtime(&mut self, runtime: Option<Arc<tokio::runtime::Runtime>>) {
        self.remote_runtime = runtime;
    }

    pub fn attach_spatialdata_layers(
        &mut self,
        spatial_root: PathBuf,
        image_transform: SpatialDataTransform2,
        extra_images: Vec<SpatialDataElement>,
        labels: Option<SpatialDataElement>,
        tables: Vec<SpatialDataElement>,
        shapes: Vec<SpatialDataElement>,
        points: Option<(SpatialDataElement, usize)>,
    ) {
        self.spatial_image_layers.clear();
        self.spatial_layers.clear();
        self.xenium_layers.clear();
        self.spatial_layers.set_root(spatial_root.clone());
        self.spatial_layers.set_tables(tables);
        self.spatial_image_transform = image_transform;
        self.spatial_label_transform = labels
            .as_ref()
            .map(|l| l.transform.relative_to(image_transform))
            .unwrap_or_default();
        self.spatial_root = Some(spatial_root.clone());
        self.spatial_label_store = zarrs::filesystem::FilesystemStore::new(&spatial_root)
            .ok()
            .map(|s| Arc::new(s) as Arc<dyn zarrs::storage::ReadableStorageTraits>);
        self.xenium_cells_offset_world = egui::Vec2::ZERO;
        self.xenium_transcripts_offset_world = egui::Vec2::ZERO;
        self.spatial_points_offset_world = egui::Vec2::ZERO;
        self.seg_objects_offset_world = egui::Vec2::ZERO;
        self.seg_objects.clear();
        self.label_cells = None;
        self.label_loader = None;
        self.label_cells_xform = None;
        self.seg_label_names = discover_label_names_local(&spatial_root);
        self.seg_label_selected = labels
            .as_ref()
            .map(|l| l.name.clone())
            .or_else(|| self.seg_label_names.first().cloned())
            .unwrap_or_default();
        self.seg_label_input = self.seg_label_selected.clone();
        self.seg_label_prompt_open = false;

        for image in &extra_images {
            let mut image = image.clone();
            image.transform = image.transform.relative_to(image_transform);
            if let Err(err) = self.spatial_image_layers.load_image(
                &spatial_root,
                &image,
                self.tiles_gl.is_some(),
                self.smooth_pixels,
            ) {
                eprintln!(
                    "failed to load SpatialData image layer {}: {err}",
                    image.name
                );
            }
        }

        for sh in &shapes {
            let mut sh = sh.clone();
            sh.transform = sh.transform.relative_to(image_transform);
            if sh.name == "cell_boundaries" {
                if let Some(rel) = sh.rel_parquet.as_ref() {
                    self.seg_objects.load_spatialdata_shapes(
                        spatial_root.join(rel),
                        sh.transform,
                        sh.name.as_str(),
                    );
                }
            } else {
                self.spatial_layers.load_shapes(&sh);
            }
        }
        if let Some((pt, max_points)) = points.as_ref() {
            let mut pt = pt.clone();
            pt.transform = pt.transform.relative_to(image_transform);
            let shape0 = self.dataset.levels.get(0).map(|l| l.shape.clone());
            let image_size = shape0.and_then(|s| {
                let x = s.get(self.dataset.dims.x).copied()? as f32;
                let y = s.get(self.dataset.dims.y).copied()? as f32;
                Some([x, y])
            });
            self.spatial_layers
                .load_points_with_image_size(&pt, *max_points, image_size);
        }
        if labels.is_some() && self.tiles_gl.is_some() && !self.seg_label_selected.is_empty() {
            let selected_label = self.seg_label_selected.clone();
            if let Err(err) = self.load_segmentation_labels(selected_label.as_str()) {
                self.seg_label_status =
                    format!("Load labels/{} failed: {err}", self.seg_label_selected);
            }
        }

        self.rebuild_layer_orders();
        self.bump_render_id();
    }

    pub fn attach_xenium_layers(
        &mut self,
        dataset_root: PathBuf,
        cells_zip: Option<PathBuf>,
        transcripts_zip: Option<PathBuf>,
        pixel_size_um: f32,
    ) {
        self.xenium_layers.clear();
        self.xenium_layers
            .attach(dataset_root, cells_zip, transcripts_zip, pixel_size_um);
        self.xenium_cells_offset_world = egui::Vec2::ZERO;
        self.xenium_transcripts_offset_world = egui::Vec2::ZERO;
        self.bump_render_id();
    }

    pub fn current_local_dataset_root(&self) -> Option<PathBuf> {
        self.dataset.source.local_path().map(|p| p.to_path_buf())
    }

    fn spatial_label_transform_for_name(&self, label_name: &str) -> SpatialDataTransform2 {
        let Some(root) = self.spatial_root.as_ref() else {
            return self.spatial_label_transform;
        };
        let Ok(discovery) = discover_spatialdata(root) else {
            return self.spatial_label_transform;
        };
        discovery
            .labels
            .iter()
            .find(|label| label.name == label_name)
            .map(|label| label.transform.relative_to(self.spatial_image_transform))
            .unwrap_or(self.spatial_label_transform)
    }

    pub fn set_status(&mut self, status: impl Into<String>) {
        self.roi_selector.set_status(status);
    }

    pub fn open_screenshot_settings(&mut self) {
        self.screenshot_settings_open = true;
    }

    pub fn screenshot_output_dir(&self) -> Option<&Path> {
        self.screenshot_output_dir.as_deref()
    }

    pub fn request_screenshot_png(&mut self, path: PathBuf) {
        let id = self.screenshot_next_id;
        self.screenshot_next_id = self.screenshot_next_id.wrapping_add(1).max(1);
        self.screenshot_pending = Some(ScreenshotRequest {
            id,
            path,
            settings: self.screenshot_settings,
        });
        self.screenshot_in_flight = Some(id);
        // Avoid capturing floating dialogs over the canvas.
        self.screenshot_settings_open = false;
        self.set_status("Capturing screenshot...".to_string());
    }

    pub fn request_quick_screenshot_png(&mut self) -> anyhow::Result<PathBuf> {
        let Some(dir) = self.screenshot_output_dir.as_deref() else {
            anyhow::bail!("No screenshot folder configured");
        };
        let path = next_numbered_screenshot_path(dir, &self.default_screenshot_filename())?;
        self.request_screenshot_png(path.clone());
        Ok(path)
    }

    pub fn default_screenshot_filename(&self) -> String {
        let base = self.dataset.source.display_name();
        let stem = base
            .strip_suffix(".ome.zarr")
            .or_else(|| base.strip_suffix(".zarr"))
            .unwrap_or(base.as_str());
        let sanitized = stem
            .chars()
            .map(|ch| match ch {
                '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
                _ => ch,
            })
            .collect::<String>();
        let sanitized = sanitized.trim().trim_matches('.').trim_matches('_');
        if sanitized.is_empty() {
            "odon.screenshot.png".to_string()
        } else {
            format!("{sanitized}.screenshot.png")
        }
    }

    pub fn export_masks_geojson(&self, path: &Path) -> anyhow::Result<()> {
        save_mask_layers_geojson(path, &self.mask_layers)
    }

    fn sync_mask_layers_into_project_space(&mut self) {
        let Some(local_root) = self.dataset.source.local_path() else {
            return;
        };
        let layers = self.mask_layers.iter().map(|l| l.to_project()).collect();
        self.project_space.set_roi_mask_layers(local_root, layers);
    }

    fn restore_mask_layers_from_project_space(&mut self) {
        let Some(local_root) = self.dataset.source.local_path() else {
            self.mask_layers.clear();
            self.next_mask_layer_id = 1;
            return;
        };
        let Some(layers) = self.project_space.roi_mask_layers(local_root) else {
            self.mask_layers.clear();
            self.next_mask_layer_id = 1;
            return;
        };

        self.mask_layers = layers.iter().map(MaskLayer::from_project).collect();
        self.next_mask_layer_id = self
            .mask_layers
            .iter()
            .map(|l| l.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
            .max(1);

        // Reset any in-progress drawing if it no longer targets a valid layer.
        if let Some(id) = self.drawing_mask_layer {
            if !self.mask_layers.iter().any(|l| l.id == id) {
                self.drawing_mask_layer = None;
                self.drawing_mask_polygon.clear();
            }
        }

        self.rebuild_layer_orders();
        self.bump_render_id();
    }

    fn ensure_editable_mask_layer(&mut self) -> u64 {
        if let LayerId::Mask(id) = self.active_layer {
            if let Some(l) = self.mask_layers.iter().find(|l| l.id == id) {
                if l.editable {
                    return id;
                }
            }
        }

        if let Some(l) = self.mask_layers.iter().rev().find(|l| l.editable) {
            return l.id;
        }

        self.create_editable_mask_layer(None)
    }

    fn create_editable_mask_layer(&mut self, name: Option<String>) -> u64 {
        let base = "Masks";
        let mut name = name.unwrap_or_else(|| base.to_string());
        if self
            .mask_layers
            .iter()
            .any(|l| l.name.eq_ignore_ascii_case(&name))
        {
            let mut i = 2;
            loop {
                let candidate = format!("{base} {i}");
                if !self
                    .mask_layers
                    .iter()
                    .any(|l| l.name.eq_ignore_ascii_case(&candidate))
                {
                    name = candidate;
                    break;
                }
                i += 1;
            }
        }

        let id = self.next_mask_layer_id.max(1);
        self.next_mask_layer_id = id.saturating_add(1);
        self.mask_layers.push(MaskLayer {
            id,
            name,
            visible: true,
            opacity: 0.9,
            width_screen_px: 2.0,
            color_rgb: [255, 210, 60],
            offset_world: egui::Vec2::ZERO,
            editable: true,
            polygons_world: Vec::new(),
            source_geojson: None,
        });
        id
    }

    pub fn request_close_dialog(&mut self) {
        self.close_dialog_open = true;
    }

    pub fn confirm_or_request_close_dialog(&mut self) -> bool {
        if self.close_dialog_open {
            self.close_dialog_open = false;
            return true;
        }
        self.close_dialog_open = true;
        false
    }

    fn configure_root_label_dataset_if_needed(&mut self) {
        if !self.dataset.is_root_label_mask() {
            return;
        }

        self.channels.clear();
        self.channel_offsets_world.clear();
        self.channel_scales.clear();
        self.channel_rotations_rad.clear();
        self.channel_layer_order.clear();
        self.selected_channel_layers.clear();
        self.channel_select_anchor_idx = None;
        self.selected_channel = 0;
        self.chanmax_pending.clear();
        self.chanmax_snapshot.clear();
        self.seg_label_names.clear();
        self.seg_label_prompt_open = false;
        self.active_layer = LayerId::SegmentationLabels;

        match self.load_root_segmentation_labels() {
            Ok(()) => {
                self.seg_label_status =
                    format!("Opened top-level label mask '{}'.", self.seg_label_selected);
            }
            Err(err) => {
                self.label_cells = None;
                self.label_loader = None;
                self.label_cells_xform = None;
                self.cells_outlines_visible = false;
                self.seg_label_selected = LabelZarrDataset::root_label_name(&self.dataset);
                self.seg_label_input = self.seg_label_selected.clone();
                self.seg_label_status = format!("Open top-level label mask failed: {err}");
            }
        }
    }

    fn load_root_segmentation_labels(&mut self) -> anyhow::Result<()> {
        if self.tiles_gl.is_none() {
            anyhow::bail!("top-level label masks require the GPU renderer");
        }
        self.labels_gl
            .get_or_insert_with(|| LabelsGl::new(1024))
            .reset();

        let lbl = LabelZarrDataset::from_root_dataset(&self.dataset);
        let label_loader = spawn_label_tile_loader(
            self.store.clone(),
            lbl.levels.iter().map(|l| l.path.clone()).collect(),
            lbl.levels.iter().map(|l| l.shape.clone()).collect(),
            lbl.levels.iter().map(|l| l.chunks.clone()).collect(),
            lbl.levels.iter().map(|l| l.dtype.clone()).collect(),
            (lbl.dims.y, lbl.dims.x),
        )?;

        self.label_loader = Some(label_loader);
        self.spatial_label_transform = SpatialDataTransform2::default();
        self.label_cells_xform = Some(compute_label_to_world_xforms(
            &self.dataset,
            &lbl,
            self.spatial_label_transform,
        ));
        self.seg_label_names.clear();
        self.seg_label_selected = lbl.label_name.clone();
        self.seg_label_input = self.seg_label_selected.clone();
        self.label_cells = Some(lbl);
        self.cells_outlines_visible = true;
        self.seg_label_prompt_open = false;
        Ok(())
    }

    fn request_default_channel_maxes(&mut self) {
        if self.channels.is_empty() {
            return;
        }
        // One epoch for all channels; ignore stale responses on ROI switches.
        let request_id = self.chanmax_request_id;
        let level = self
            .chanmax_level
            .min(self.dataset.levels.len().saturating_sub(1));
        self.chanmax_snapshot = self.channels.iter().map(|c| c.window).collect();
        for (i, ch) in self.channels.iter().enumerate() {
            if !self.chanmax_pending.get(i).copied().unwrap_or(false) {
                continue;
            }
            let _ = self.chanmax_loader.tx.send(ChannelMaxRequest {
                request_id,
                level,
                channel: ch.index as u64,
            });
        }
    }

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

            let mut hi = (msg.p97 as f32).clamp(0.0, abs_max);
            if !hi.is_finite() || hi <= 0.0 {
                hi = 1.0;
            }
            self.channels[idx].window = Some((0.0, hi));
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

impl eframe::App for OmeZarrViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Per-frame flow:
        // 1. absorb external input (drops, close requests, project-config changes)
        // 2. drain worker output and tick async subsystems into a consistent snapshot
        // 3. build chrome/panels
        // 4. draw the central canvas and overlays
        // 5. schedule the next repaint based on outstanding async work
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        if !dropped.is_empty() {
            let mut other_paths = Vec::new();
            for path in dropped.into_iter().filter_map(|f| f.path) {
                if !self.seg_objects.handle_dropped_path(path.clone()) {
                    other_paths.push(path);
                }
            }
            if !other_paths.is_empty() {
                self.project_space.handle_dropped_paths(other_paths);
            }
        }

        self.ui_seg_label_prompt(ctx);
        self.rebuild_layer_orders();

        // Push project config updates into custom panels that depend on it.
        let cfg_gen = self.project_space.config_generation();
        if cfg_gen != self.project_cfg_seen {
            self.project_cfg_seen = cfg_gen;
            let cfg = self.project_space.config().clone();
            self.roi_selector
                .set_project_config(cfg.clone(), &self.dataset.source);
            self.cell_thresholds.set_project_config(cfg);
        }

        // Napari-like "close window" prompt.
        // - Cmd/Ctrl+W opens confirmation
        // - Cmd/Ctrl+W again confirms close
        if top_bar::handle_cmd_w_close(ctx, &mut self.close_dialog_open) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }

        self.drain_tiles(ctx);
        self.drain_raw_tiles();
        self.drain_label_tiles();
        self.drain_screenshots();
        self.drain_histogram();
        self.drain_channel_maxes();
        self.seg_geojson.tick();
        self.seg_objects.tick();
        self.spatial_image_layers.tick();
        self.spatial_layers.tick();
        self.xenium_layers.tick();
        self.cell_thresholds.tick(&mut self.cell_points);
        self.roi_selector.tick();
        refresh_system_memory_if_needed(
            &mut self.system_memory,
            &mut self.system_memory_last_refresh,
            Duration::from_secs(2),
        );
        self.sync_analysis_follow_active_channel_state();
        let mut ann_changed = false;
        for layer in &mut self.annotation_layers {
            ann_changed |= layer.tick();
        }
        if ann_changed {
            self.bump_render_id();
            ctx.request_repaint();
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                top_bar::ui_title(
                    ui,
                    format!("OME-Zarr: {}", self.dataset.source.display_name()),
                );
                ui.separator();
                if top_bar::ui_fit(ui, "Fit (F)") {
                    self.fit_to_last_canvas();
                }
                ui.separator();
                top_bar::ui_auto_level(
                    ui,
                    &mut self.auto_level,
                    &mut self.manual_level,
                    self.dataset.levels.len().saturating_sub(1),
                );
                ui.separator();
                let have_channels = !self.channels.is_empty();
                if let Some(step) = top_bar::ui_prev_next_channel(ui, have_channels) {
                    self.step_selected_channel_visibility(step);
                }
                if self.roi_selector.has_multiple_rois() {
                    ui.separator();
                    if let Some(step) = top_bar::ui_prev_next_roi(ui, true) {
                        if let Some(action) = self.roi_selector.step_roi_action(step) {
                            self.handle_roi_selector_action(ctx, action);
                        }
                    }
                }
                ui.separator();
                top_bar::ui_panel_toggles(
                    ui,
                    &mut self.show_left_panel,
                    &mut self.show_right_panel,
                );

                if top_bar::ui_smooth_toggle(ui, &mut self.smooth_pixels) {
                    if let Some(tiles_gl) = self.tiles_gl.as_ref() {
                        tiles_gl.set_smooth_pixels(self.smooth_pixels);
                    } else {
                        // CPU fallback: texture filtering is chosen at texture creation time.
                        // Recreate the cache so tiles are re-uploaded with the new sampling.
                        self.cache = TileCache::new(256);
                        self.pending.clear();
                    }
                    self.spatial_image_layers
                        .set_smooth_pixels(self.smooth_pixels);
                }
                ui.checkbox(&mut self.show_tile_debug, "Tile Debug");

                // Compact contrast controls when both side panels are hidden.
                if !self.show_left_panel && !self.show_right_panel && have_channels {
                    ui.separator();
                    let abs_max = self.dataset.abs_max.max(1.0);
                    let ch_idx = self
                        .selected_channel
                        .min(self.channels.len().saturating_sub(1));
                    let ch_name = self
                        .channels
                        .get(ch_idx)
                        .map(|c| c.name.clone())
                        .unwrap_or_default();
                    let window = self.channels[ch_idx].window.unwrap_or((0.0, abs_max));
                    if let Some((lo, hi)) = top_bar::ui_compact_contrast(
                        ui,
                        top_bar::CompactContrastParams {
                            abs_max,
                            channel_name: &ch_name,
                            window,
                            step: 1.0,
                            id_salt: "top-contrast-range",
                        },
                    ) {
                        if let Some(dst) = self.channels.get_mut(ch_idx) {
                            dst.window = Some((lo, hi));
                            self.channel_window_overrides
                                .insert(dst.name.clone(), (lo, hi));
                        }
                        self.hist_dirty = true;
                        self.bump_render_id();
                    }
                }
            });
        });
        self.seg_objects.ui_load_dialog(ctx);

        if self.show_left_panel {
            let mut tab = self.left_tab;
            left_panel::show(
                ctx,
                "viewer-left",
                &mut tab,
                &[
                    left_panel::TabSpec {
                        tab: LeftTab::Layers,
                        label: "Layers",
                        panel_key: "layers",
                        default_width: 360.0,
                        scroll: false,
                    },
                    left_panel::TabSpec {
                        tab: LeftTab::Project,
                        label: "Project",
                        panel_key: "project",
                        default_width: 420.0,
                        scroll: false,
                    },
                ],
                |ui, tab| match tab {
                    LeftTab::Layers => self.ui_layers(ui, ctx),
                    LeftTab::Project => {
                        let cur = self.dataset.source.local_path();
                        if let Some(action) = self.project_space.ui(ui, cur) {
                            match action {
                                ProjectSpaceAction::Open(roi) => {
                                    self.project_space
                                        .set_status(format!("Opening: {}", roi.source_display()));
                                    self.pending_request = Some(ViewerRequest::OpenProjectRoi(roi));
                                }
                                ProjectSpaceAction::OpenMosaic(rois) => {
                                    self.project_space.set_status(format!(
                                        "Opening mosaic ({} items)...",
                                        rois.len()
                                    ));
                                    self.pending_request =
                                        Some(ViewerRequest::OpenProjectMosaic(rois));
                                }
                                ProjectSpaceAction::OpenRemoteDialog => {
                                    self.remote_dialog_open = true;
                                    self.remote_status.clear();
                                }
                            }
                        }
                    }
                },
            );
            self.left_tab = tab;
        }

        if self.show_right_panel {
            let mut tab = self.right_tab;
            right_panel::show(
                ctx,
                "right",
                380.0,
                &mut tab,
                &[
                    right_panel::TabSpec {
                        tab: RightTab::Properties,
                        label: "Properties",
                        scroll: false,
                    },
                    right_panel::TabSpec {
                        tab: RightTab::Analysis,
                        label: "Analysis",
                        scroll: true,
                    },
                    right_panel::TabSpec {
                        tab: RightTab::Measurements,
                        label: "Measurements",
                        scroll: true,
                    },
                    right_panel::TabSpec {
                        tab: RightTab::Memory,
                        label: "Memory",
                        scroll: true,
                    },
                    right_panel::TabSpec {
                        tab: RightTab::RoiSelector,
                        label: "ROI Selector",
                        scroll: true,
                    },
                ],
                |ui, tab| match tab {
                    RightTab::Properties => self.ui_layer_properties(ui, ctx),
                    RightTab::Analysis => {
                        let suspend_live_selection_sync =
                            matches!(self.tool_mode, ToolMode::RectSelect | ToolMode::LassoSelect);
                        if self.seg_objects.object_count() > 0 {
                            self.seg_objects.ui_analysis(
                                ui,
                                &self.dataset,
                                self.store.clone(),
                                &self.channels,
                                self.selected_channel,
                                suspend_live_selection_sync,
                                self.seg_objects_offset_world,
                                self.spatial_root.as_deref(),
                                self.spatial_layers.table_elements(),
                            );
                            if let Some(idx) = self.seg_objects.take_pending_zoom_object_index() {
                                self.fit_to_seg_object_index(idx);
                            }
                        } else if let LayerId::SpatialShape(id) = self.active_layer {
                            let spatial_tables = self.spatial_layers.table_elements().to_vec();
                            if let Some(layer) =
                                self.spatial_layers.shapes.iter_mut().find(|s| s.id == id)
                            {
                                let offset_world = layer.offset_world;
                                if let Some(objects) = layer.object_layer_mut() {
                                    objects.ui_analysis(
                                        ui,
                                        &self.dataset,
                                        self.store.clone(),
                                        &self.channels,
                                        self.selected_channel,
                                        suspend_live_selection_sync,
                                        offset_world,
                                        self.spatial_root.as_deref(),
                                        &spatial_tables,
                                    );
                                    if let Some(idx) = objects.take_pending_zoom_object_index() {
                                        if let Some(viewport) = self.last_canvas_rect
                                            && let Some(world) =
                                                objects.fit_object_bounds_world(idx, offset_world)
                                        {
                                            self.camera.fit_to_world_rect(viewport, world);
                                        }
                                    }
                                } else {
                                    ui.heading("Analysis");
                                    ui.label(
                                        "Analysis is available for object-backed shape layers.",
                                    );
                                }
                            } else {
                                ui.heading("Analysis");
                                ui.label("SpatialData shape layer not found.");
                            }
                        } else {
                            ui.heading("Analysis");
                            ui.label(
                                "Analysis is available for loaded Segmentation Objects and object-backed SpatialData shape layers.",
                            );
                        }
                    }
                    RightTab::Measurements => {
                        if self.seg_objects.object_count() > 0 {
                            self.seg_objects.ui_measurements(
                                ui,
                                &self.dataset,
                                self.store.clone(),
                                &self.channels,
                                self.seg_objects_offset_world,
                            );
                        } else if let LayerId::SpatialShape(id) = self.active_layer {
                            if let Some(layer) =
                                self.spatial_layers.shapes.iter_mut().find(|s| s.id == id)
                            {
                                let offset_world = layer.offset_world;
                                if let Some(objects) = layer.object_layer_mut() {
                                    objects.ui_measurements(
                                        ui,
                                        &self.dataset,
                                        self.store.clone(),
                                        &self.channels,
                                        offset_world,
                                    );
                                } else {
                                    ui.heading("Measurements");
                                    ui.label(
                                        "Measurements are available for object-backed shape layers.",
                                    );
                                }
                            } else {
                                ui.heading("Measurements");
                                ui.label("SpatialData shape layer not found.");
                            }
                        } else {
                            ui.heading("Measurements");
                            ui.label(
                                "Measurements are available for loaded Segmentation Objects and object-backed SpatialData shape layers.",
                            );
                        }
                    }
                    RightTab::Memory => self.ui_memory(ui),
                    RightTab::RoiSelector => {
                        if let Some(action) = self.roi_selector.ui(ui) {
                            self.handle_roi_selector_action(ctx, action);
                        }
                    }
                },
            );
            self.right_tab = tab;
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            self.ui_canvas(ui, ctx);
        });

        if self.remote_dialog_open {
            self.ui_remote_dialog(ctx);
        }

        self.ui_group_layers_dialog(ctx);
        self.ui_memory_load_dialog(ctx);
        self.ui_screenshot_settings_dialog(ctx);

        if top_bar::ui_close_dialog(ctx, &mut self.close_dialog_open) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }

        if ctx.input(|i| i.key_pressed(egui::Key::F)) {
            if self.active_layer == LayerId::SegmentationObjects
                && !self.fit_to_selected_seg_objects()
            {
                self.fit_to_last_canvas();
            } else if self.active_layer != LayerId::SegmentationObjects {
                self.fit_to_last_canvas();
            }
        }

        self.schedule_repaint(ctx);
    }
}

const DEFAULT_MAX_SCAN_PIXELS: u64 = 2_000_000;

fn choose_default_max_level(dataset: &OmeZarrDataset) -> usize {
    let y_dim = dataset.dims.y;
    let x_dim = dataset.dims.x;
    let mut chosen = dataset.levels.len().saturating_sub(1);
    for (i, level) in dataset.levels.iter().enumerate().rev() {
        let y = *level.shape.get(y_dim).unwrap_or(&0);
        let x = *level.shape.get(x_dim).unwrap_or(&0);
        let pixels = y.saturating_mul(x);
        if pixels > 0 && pixels <= DEFAULT_MAX_SCAN_PIXELS {
            chosen = i;
        }
    }
    chosen
}

impl OmeZarrViewerApp {
    fn apply_tiff_plane_selection(
        &mut self,
        ctx: &egui::Context,
        target_z: usize,
        target_t: usize,
    ) -> anyhow::Result<()> {
        let Some(prev_state) = self.tiff_plane_state.clone() else {
            anyhow::bail!("TIFF plane selection is not active");
        };

        let prev_channels = self.channels.clone();
        let prev_selected_name = self
            .channels
            .get(self.selected_channel)
            .map(|c| c.name.clone());
        let old_world_w = self
            .dataset
            .levels
            .first()
            .map(|l| l.shape[self.dataset.dims.x] as f32)
            .unwrap_or(0.0);
        let old_world_h = self
            .dataset
            .levels
            .first()
            .map(|l| l.shape[self.dataset.dims.y] as f32)
            .unwrap_or(0.0);
        let fx = if old_world_w > 0.0 {
            (self.camera.center_world_lvl0.x / old_world_w).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let fy = if old_world_h > 0.0 {
            (self.camera.center_world_lvl0.y / old_world_h).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let old_zoom = self.camera.zoom_screen_per_lvl0_px;
        let preserve_transforms = self.channels.len();

        let mut assets = build_tiff_runtime_assets(
            self.tiles_gl.is_some(),
            prev_state.dataset_root,
            prev_state.image_path,
            prev_state.dataset_name,
            prev_state.channel_name,
            crate::xenium::TiffPlaneSelection {
                z: target_z,
                t: target_t,
            },
        )?;

        let mut new_channels = assets.dataset.channels.clone();
        apply_preserved_channel_settings(&prev_channels, &mut new_channels);
        for ch in &mut new_channels {
            if let Some(w) = self.channel_window_overrides.get(&ch.name).copied() {
                ch.window = Some(w);
            }
        }
        assets.dataset.channels = new_channels.clone();

        if let Some(tiles_gl) = self.tiles_gl.as_ref() {
            tiles_gl.reset();
        }
        if let Some(labels_gl) = self.labels_gl.as_ref() {
            labels_gl.reset();
        }

        self.dataset = assets.dataset;
        self.store = assets.store;
        self.loader = assets.loader;
        self.raw_loader = assets.raw_loader;
        self.hist_loader = assets.hist_loader;
        self.chanmax_loader = assets.chanmax_loader;
        self.chanmax_level = assets.chanmax_level;
        self.channels = new_channels;
        self.tiff_plane_state = assets.tiff_plane_state.take();
        if let Some(state) = self.tiff_plane_state.as_mut() {
            state.status.clear();
        }

        if self.channels.len() != preserve_transforms {
            self.channel_offsets_world = vec![egui::Vec2::ZERO; self.channels.len()];
            self.channel_scales = vec![egui::Vec2::splat(1.0); self.channels.len()];
            self.channel_rotations_rad = vec![0.0; self.channels.len()];
        }

        if let Some(name) = prev_selected_name {
            if let Some(ch) = self.channels.iter().find(|c| c.name == name) {
                self.selected_channel = ch.index;
            } else {
                self.selected_channel = self
                    .selected_channel
                    .min(self.channels.len().saturating_sub(1));
            }
        } else {
            self.selected_channel = self
                .selected_channel
                .min(self.channels.len().saturating_sub(1));
        }
        if matches!(self.active_layer, LayerId::Channel(_)) {
            self.active_layer = LayerId::Channel(self.selected_channel);
        }

        let new_world_w = self
            .dataset
            .levels
            .first()
            .map(|l| l.shape[self.dataset.dims.x] as f32)
            .unwrap_or(0.0);
        let new_world_h = self
            .dataset
            .levels
            .first()
            .map(|l| l.shape[self.dataset.dims.y] as f32)
            .unwrap_or(0.0);
        self.camera.center_world_lvl0 = egui::pos2(new_world_w * fx, new_world_h * fy);
        self.camera.zoom_screen_per_lvl0_px = old_zoom;

        self.cache = TileCache::new(256);
        self.pending.clear();
        self.previous_render_id = None;
        self.active_render_id = self.compute_render_id();
        self.hist = None;
        self.hist_request_id = 0;
        self.hist_request_pending = false;
        self.hist_dirty = true;
        self.hist_last_sent = Instant::now()
            .checked_sub(Duration::from_secs(3600))
            .unwrap_or_else(Instant::now);
        self.chanmax_request_id = self.chanmax_request_id.wrapping_add(1).max(1);
        self.chanmax_pending = self
            .channels
            .iter()
            .map(|c| !self.channel_window_overrides.contains_key(&c.name))
            .collect();
        self.chanmax_snapshot = self.channels.iter().map(|c| c.window).collect();
        self.request_default_channel_maxes();
        ctx.request_repaint();
        Ok(())
    }

    fn ui_screenshot_settings_dialog(&mut self, ctx: &egui::Context) {
        if !self.screenshot_settings_open {
            return;
        }
        let mut open = self.screenshot_settings_open;
        egui::Window::new("Screenshot Settings")
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.label("These options affect canvas-only PNG screenshots.");
                ui.label(
                    "Quick Screenshot uses Cmd+Shift+S and saves directly to the folder below.",
                );
                ui.add_space(6.0);
                ui.label("Quick-save folder");
                ui.horizontal(|ui| {
                    let folder_text = self
                        .screenshot_output_dir
                        .as_deref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "Not set".to_string());
                    ui.monospace(folder_text);
                    if ui.button("Choose...").clicked() {
                        let mut dialog = FileDialog::new().set_title("Select Screenshot Folder");
                        if let Some(dir) = self.screenshot_output_dir.as_deref() {
                            dialog = dialog.set_directory(dir);
                        }
                        if let Some(dir) = dialog.pick_folder() {
                            self.screenshot_output_dir = Some(dir);
                        }
                    }
                    if ui
                        .add_enabled(
                            self.screenshot_output_dir.is_some(),
                            egui::Button::new("Clear"),
                        )
                        .clicked()
                    {
                        self.screenshot_output_dir = None;
                    }
                });
                ui.add_space(6.0);
                ui.checkbox(
                    &mut self.screenshot_settings.include_scale_bar,
                    "Include scale bar",
                );
                ui.checkbox(
                    &mut self.screenshot_settings.include_legend,
                    "Include legend (visible markers)",
                );
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label("Scale bar size");
                    ui.add(
                        egui::Slider::new(&mut self.screenshot_settings.scale_bar_scale, 0.5..=3.0)
                            .suffix("x"),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Legend size");
                    ui.add(
                        egui::Slider::new(&mut self.screenshot_settings.legend_scale, 0.5..=3.0)
                            .suffix("x"),
                    );
                });
            });
        self.screenshot_settings_open = open;
    }

    fn ui_tiff_plane_controls(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let Some(mut state) = self.tiff_plane_state.clone() else {
            return;
        };
        if state.size_z <= 1 && state.size_t <= 1 {
            return;
        }

        ui.heading("Plane");
        ui.label(format!(
            "Current: Z={}, T={}",
            state.current_z, state.current_t
        ));

        ui.horizontal(|ui| {
            ui.label("Z");
            ui.add_enabled(
                state.size_z > 1,
                egui::DragValue::new(&mut state.draft_z).range(0..=state.size_z.saturating_sub(1)),
            );
            ui.label("T");
            ui.add_enabled(
                state.size_t > 1,
                egui::DragValue::new(&mut state.draft_t).range(0..=state.size_t.saturating_sub(1)),
            );
        });

        let changed = state.draft_z != state.current_z || state.draft_t != state.current_t;
        ui.horizontal(|ui| {
            if ui
                .add_enabled(changed, egui::Button::new("Apply"))
                .clicked()
            {
                match self.apply_tiff_plane_selection(ctx, state.draft_z, state.draft_t) {
                    Ok(()) => return,
                    Err(err) => state.status = err.to_string(),
                }
            }
            if ui
                .add_enabled(changed, egui::Button::new("Reset"))
                .clicked()
            {
                state.draft_z = state.current_z;
                state.draft_t = state.current_t;
                state.status.clear();
            }
        });

        if !state.status.is_empty() {
            ui.colored_label(ui.visuals().warn_fg_color, &state.status);
        }
        self.tiff_plane_state = Some(state);
        ui.separator();
    }

    fn is_loading_scene(&self) -> bool {
        let mut busy = false;
        if let Some(tiles_gl) = self.tiles_gl.as_ref() {
            busy |= tiles_gl.is_busy();
        }
        busy |= self.cache.is_busy();
        if let Some(labels_gl) = self.labels_gl.as_ref() {
            busy |= labels_gl.is_busy();
        }
        busy |= self.seg_geojson.is_busy();
        busy |= self.seg_objects.is_busy();
        busy |= self.spatial_image_layers.is_busy();
        busy |= self.spatial_layers.is_busy();
        busy |= self.pinned_levels.has_loading();
        busy
    }

    fn image_tile_request_count(&self) -> usize {
        if let Some(tiles_gl) = self.tiles_gl.as_ref() {
            tiles_gl.in_flight_len()
        } else {
            self.cache.in_flight_len()
        }
    }

    fn desired_raw_tile_cache_capacity(&self, visible_target_raw_tiles: usize) -> usize {
        let headroom = visible_target_raw_tiles
            .saturating_div(4)
            .max(RAW_TILE_CACHE_HEADROOM_TILES);
        visible_target_raw_tiles
            .saturating_add(headroom)
            .max(RAW_TILE_CACHE_CAPACITY_TILES)
            .min(RAW_TILE_CACHE_MAX_CAPACITY_TILES)
    }

    fn maybe_grow_raw_tile_cache(&self, tiles_gl: &TilesGl, visible_target_raw_tiles: usize) {
        let desired = self.desired_raw_tile_cache_capacity(visible_target_raw_tiles);
        let current = tiles_gl.capacity();
        if desired <= current {
            return;
        }
        crate::log_info!(
            "growing raw tile cache from {} to {} tiles for visible target set {}",
            current,
            desired,
            visible_target_raw_tiles
        );
        tiles_gl.grow_capacity(desired);
    }

    fn loading_indicator_text(&self) -> Option<&'static str> {
        if self.seg_objects.is_loading() {
            Some("Loading segmentation objects...")
        } else if self.spatial_image_layers.is_busy() {
            Some("Loading SpatialData images...")
        } else if self.spatial_layers.is_loading_shapes() {
            Some("Loading SpatialData shapes...")
        } else if self.spatial_layers.is_loading_points() {
            Some("Loading SpatialData points...")
        } else if self.seg_objects.is_analyzing() {
            Some("Measuring cell intensities...")
        } else if self.seg_geojson.is_busy() {
            Some("Loading segmentation...")
        } else if self.pinned_levels.has_loading() {
            Some("Pinning image level into RAM...")
        } else if self.is_loading_scene() {
            Some("Loading image tiles...")
        } else {
            None
        }
    }

    fn step_selected_channel_visibility(&mut self, step: i32) {
        if self.channels.is_empty() || self.channel_layer_order.is_empty() {
            return;
        }
        let cur_idx = self
            .selected_channel
            .min(self.channels.len().saturating_sub(1));
        let cur_pos = self
            .channel_layer_order
            .iter()
            .position(|&idx| idx == cur_idx)
            .unwrap_or(0);
        let n = self.channel_layer_order.len() as i32;
        let next_pos = ((cur_pos as i32) + step).rem_euclid(n) as usize;
        let next_idx =
            self.channel_layer_order[next_pos].min(self.channels.len().saturating_sub(1));

        if let Some(cur) = self.channels.get_mut(cur_idx) {
            cur.visible = false;
        }
        if let Some(next) = self.channels.get_mut(next_idx) {
            next.visible = true;
        }

        self.selected_channel = next_idx;
        self.active_layer = LayerId::Channel(next_idx);
        self.hist_dirty = true;
        self.bump_render_id();

        if let Some(ch) = self.channels.get(next_idx) {
            let _ = self.cell_thresholds.sync_marker_from_channel_name(&ch.name);
        }
    }

    fn handle_cell_thresholds_action(&mut self, action: CellThresholdsAction) {
        match action {
            CellThresholdsAction::CycleImageChannel(step) => {
                self.cycle_image_channel(step);
            }
            CellThresholdsAction::NudgeImageContrastMax(delta) => {
                self.nudge_selected_channel_max(delta);
            }
        }
    }

    fn cycle_image_channel(&mut self, step: i32) {
        if self.channels.is_empty() {
            return;
        }
        let n = self.channels.len() as i32;
        let cur = (self
            .selected_channel
            .min(self.channels.len().saturating_sub(1))) as i32;
        let next = (cur + step).rem_euclid(n) as usize;

        for ch in &mut self.channels {
            ch.visible = false;
        }
        if let Some(ch) = self.channels.get_mut(next) {
            ch.visible = true;
        }
        self.selected_channel = next;
        self.active_layer = LayerId::Channel(next);
        self.hist_dirty = true;
        self.bump_render_id();

        if let Some(ch) = self.channels.get(next) {
            let _ = self.cell_thresholds.sync_marker_from_channel_name(&ch.name);
        }
    }

    fn sync_analysis_follow_active_channel_state(&mut self) {
        self.seg_objects
            .ensure_object_property_analysis_warmup_started(&self.channels, self.selected_channel);
        self.seg_objects
            .sync_analysis_follow_active_channel(&self.channels, self.selected_channel);

        let active_shape_id = match self.active_layer {
            LayerId::SpatialShape(id) => Some(id),
            _ => None,
        };
        if let Some(id) = active_shape_id
            && let Some(layer) = self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|shape| shape.id == id)
            && let Some(objects) = layer.object_layer_mut()
        {
            objects.ensure_object_property_analysis_warmup_started(
                &self.channels,
                self.selected_channel,
            );
            objects.sync_analysis_follow_active_channel(&self.channels, self.selected_channel);
        }
    }

    fn nudge_selected_channel_max(&mut self, delta: f32) {
        let abs_max = self.dataset.abs_max.max(1.0);
        let idx = self
            .selected_channel
            .min(self.channels.len().saturating_sub(1));
        let Some(ch) = self.channels.get_mut(idx) else {
            return;
        };
        let name = ch.name.clone();
        let (mut lo, mut hi) = ch.window.unwrap_or((0.0, abs_max));
        lo = lo.clamp(0.0, abs_max);
        hi = (hi + delta).clamp(0.0, abs_max);
        if hi <= lo {
            hi = (lo + 1.0).min(abs_max);
        }
        ch.window = Some((lo, hi));
        self.channel_window_overrides.insert(name, (lo, hi));
        self.hist_dirty = true;
        self.bump_render_id();
    }

    fn auto_load_project_roi_segmentation(&mut self) {
        if self.seg_objects.loaded_geojson.is_some()
            || self.seg_geojson.loaded_geojson.is_some()
            || self.seg_objects.is_loading()
            || self.seg_geojson.is_busy()
        {
            return;
        }

        let Some(roi) = self
            .project_space
            .config()
            .rois
            .iter()
            .find(|roi| match (roi.dataset_source(), &self.dataset.source) {
                (
                    Some(crate::data::dataset_source::DatasetSource::Local(path)),
                    crate::data::dataset_source::DatasetSource::Local(active),
                ) => path == *active || path.to_string_lossy() == active.to_string_lossy(),
                (Some(source), active) => source == *active,
                (None, _) => false,
            })
            .cloned()
        else {
            return;
        };

        let Some(segpath) = roi.segpath else {
            return;
        };

        let segpath = if segpath.is_relative() {
            self.project_space
                .project_dir()
                .map(|dir| dir.join(&segpath))
                .unwrap_or(segpath)
        } else {
            segpath
        };

        let Some(ext) = segpath.extension().and_then(|s| s.to_str()) else {
            self.roi_selector.set_status(format!(
                "Project segmentation path has no supported extension: {}",
                segpath.to_string_lossy()
            ));
            return;
        };

        if !segpath.exists() {
            self.roi_selector.set_status(format!(
                "Project segmentation path was not found: {}",
                segpath.to_string_lossy()
            ));
            return;
        }

        match ext.to_ascii_lowercase().as_str() {
            "geojson" | "json" | "geoparquet" | "parquet" => {
                self.seg_objects
                    .load_path(segpath.clone(), self.seg_objects.downsample_factor);
                self.set_active_layer(LayerId::SegmentationObjects);
                self.roi_selector.set_status(format!(
                    "Loading segmentation: {}",
                    segpath
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("segmentation")
                ));
            }
            _ => {
                self.roi_selector.set_status(format!(
                    "Project segmentation format is not supported for single view: {}",
                    segpath.to_string_lossy()
                ));
            }
        }
    }

    fn handle_roi_selector_action(&mut self, ctx: &egui::Context, action: RoiSelectorAction) {
        match action {
            RoiSelectorAction::OpenRoi(roi) => {
                let Some(source) = roi.dataset_source() else {
                    self.roi_selector
                        .set_status("Open ROI failed: ROI has no dataset source.".to_string());
                    return;
                };
                match source {
                    crate::data::dataset_source::DatasetSource::Local(path) => {
                        if let Err(err) = self.switch_roi(ctx, path) {
                            self.roi_selector
                                .set_status(format!("Open ROI failed: {err}"));
                        }
                    }
                    _ => {
                        self.pending_request = Some(ViewerRequest::OpenProjectRoi(roi));
                    }
                }
            }
            RoiSelectorAction::LoadLabels => {
                if let Err(err) = self.ensure_segmentation_labels_loaded() {
                    self.roi_selector
                        .set_status(format!("Load Labels failed: {err}"));
                } else {
                    self.roi_selector.set_status(format!(
                        "Loaded labels/{}.",
                        self.seg_label_selected.as_str()
                    ));
                }
            }
            RoiSelectorAction::LoadMasks => match self.ensure_exclusion_masks_loaded() {
                Ok(n) => {
                    self.roi_selector
                        .set_status(format!("Loaded masks ({n} shapes)."));
                }
                Err(err) => {
                    self.roi_selector
                        .set_status(format!("Load Masks failed: {err}"));
                }
            },
            RoiSelectorAction::SaveMasks => {
                let Some(local_root) = self.dataset.source.local_path() else {
                    self.roi_selector
                        .set_status("Save Masks is supported for local datasets only.".to_string());
                    return;
                };
                if !self.drawing_mask_polygon.is_empty() {
                    self.roi_selector.set_status(
                        "Finish polygon (Enter/double-click) or cancel (Esc) before saving."
                            .to_string(),
                    );
                    return;
                }

                // Legacy "Save Masks" appends editable (non-file-backed) mask polygons to the
                // napari-style masks file path inferred from the Project config.
                let mut polys: Vec<Vec<egui::Pos2>> = Vec::new();
                for l in &self.mask_layers {
                    if !l.editable || l.source_geojson.is_some() {
                        continue;
                    }
                    for poly in &l.polygons_world {
                        if poly.len() < 3 {
                            continue;
                        }
                        polys.push(
                            poly.iter()
                                .copied()
                                .map(|p| p + l.offset_world)
                                .collect::<Vec<_>>(),
                        );
                    }
                }
                if polys.is_empty() {
                    self.roi_selector
                        .set_status("No drawn masks to save.".to_string());
                    return;
                }

                let Some(cfg) = self.roi_selector.masks_config_for_roi(local_root) else {
                    self.roi_selector.set_status(
                        "Save Masks failed: no matching dataset in Project config.".to_string(),
                    );
                    return;
                };
                let entry = self.roi_selector.roi_entry_for_path(local_root);

                match resolve_masks_geojson_path_and_downsample(local_root, &cfg, entry.as_ref()) {
                    Ok(resolved) => {
                        let result: anyhow::Result<()> = (|| {
                            let ds = resolved.downsample_factor.max(1e-6);
                            let path = &resolved.geojson_path;
                            if let Some(parent) = path.parent() {
                                fs::create_dir_all(parent).with_context(|| {
                                    format!("failed to create {}", parent.to_string_lossy())
                                })?;
                            }

                            let mut root: serde_json::Value = if path.exists() {
                                let text = fs::read_to_string(path).with_context(|| {
                                    format!("failed to read {}", path.to_string_lossy())
                                })?;
                                serde_json::from_str(&text)
                                    .context("failed to parse existing GeoJSON")?
                            } else {
                                serde_json::json!({"type":"FeatureCollection","features":[]})
                            };

                            let feats = root
                                .get_mut("features")
                                .and_then(|v| v.as_array_mut())
                                .ok_or_else(|| {
                                    anyhow::anyhow!("GeoJSON missing 'features' array")
                                })?;

                            for (idx, poly) in polys.iter().enumerate() {
                                if poly.len() < 3 {
                                    continue;
                                }
                                let mut ring: Vec<Vec<f64>> = Vec::with_capacity(poly.len() + 1);
                                for &p in poly {
                                    ring.push(vec![
                                        (p.x as f64) * (ds as f64),
                                        (p.y as f64) * (ds as f64),
                                    ]);
                                }
                                if ring.first() != ring.last() {
                                    if let Some(first) = ring.first().cloned() {
                                        ring.push(first);
                                    }
                                }

                                feats.push(serde_json::json!({
                                    "type": "Feature",
                                    "geometry": { "type": "Polygon", "coordinates": [ ring ] },
                                    "properties": {
                                        "layer": "odon_masks",
                                        "shape_index": idx as i64,
                                        "roi_root": local_root.to_string_lossy(),
                                    }
                                }));
                            }

                            let text = serde_json::to_string_pretty(&root)
                                .context("failed to encode GeoJSON")?;
                            fs::write(path, text).with_context(|| {
                                format!("failed to write {}", path.to_string_lossy())
                            })?;
                            Ok(())
                        })();
                        if let Err(err) = result {
                            self.roi_selector
                                .set_status(format!("Save Masks failed: {err}"));
                            return;
                        }

                        // Clear saved (editable, non-file-backed) layers.
                        for l in &mut self.mask_layers {
                            if l.editable && l.source_geojson.is_none() {
                                l.clear();
                            }
                        }

                        // Refresh the read-only layer from disk so appended shapes show up there too.
                        if let Err(err) = self.ensure_exclusion_masks_loaded() {
                            self.roi_selector
                                .set_status(format!("Saved masks, but reload failed: {err}"));
                        } else {
                            self.roi_selector.set_status(format!(
                                "Saved masks (appended) -> {}",
                                resolved.geojson_path.to_string_lossy()
                            ));
                        }
                    }
                    Err(err) => {
                        self.roi_selector
                            .set_status(format!("Save Masks failed: {err}"));
                    }
                }
            }
        }
    }

    fn switch_roi(&mut self, ctx: &egui::Context, new_root: PathBuf) -> anyhow::Result<()> {
        if new_root.as_os_str().is_empty() {
            anyhow::bail!("empty ROI path");
        }
        if self
            .dataset
            .source
            .local_path()
            .is_some_and(|p| p == new_root.as_path())
        {
            return Ok(());
        }

        let (dataset, store) = crate::data::ome::OmeZarrDataset::open_local(&new_root)?;
        self.switch_dataset_with_store(ctx, dataset, store, None)
    }

    fn switch_dataset_with_store(
        &mut self,
        ctx: &egui::Context,
        mut dataset: OmeZarrDataset,
        store: Arc<dyn zarrs::storage::ReadableStorageTraits>,
        remote_runtime: Option<Arc<tokio::runtime::Runtime>>,
    ) -> anyhow::Result<()> {
        if dataset.source == self.dataset.source {
            return Ok(());
        }
        // Persist editable, project-backed state before replacing the dataset, then rebuild the
        // viewer around the new source while preserving the user's channel preferences and an
        // approximate camera position in normalized image coordinates.
        self.sync_mask_layers_into_project_space();

        let prev_channels = self.channels.clone();
        let prev_selected_name = self
            .channels
            .get(self.selected_channel)
            .map(|c| c.name.clone());

        let old_world_w = self
            .dataset
            .levels
            .get(0)
            .map(|l| l.shape[self.dataset.dims.x] as f32)
            .unwrap_or(0.0);
        let old_world_h = self
            .dataset
            .levels
            .get(0)
            .map(|l| l.shape[self.dataset.dims.y] as f32)
            .unwrap_or(0.0);
        let fx = if old_world_w > 0.0 {
            (self.camera.center_world_lvl0.x / old_world_w).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let fy = if old_world_h > 0.0 {
            (self.camera.center_world_lvl0.y / old_world_h).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let old_zoom = self.camera.zoom_screen_per_lvl0_px;
        let reload_exclusion_masks = self
            .mask_layers
            .iter()
            .any(|l| l.name == "Exclusion masks" && !l.editable && l.visible);

        let mut new_channels = dataset.channels.clone();
        apply_preserved_channel_settings(&prev_channels, &mut new_channels);
        for ch in &mut new_channels {
            if let Some(w) = self.channel_window_overrides.get(&ch.name).copied() {
                ch.window = Some(w);
            }
        }
        dataset.channels = new_channels.clone();

        let dims_cyx = (dataset.dims.c, dataset.dims.y, dataset.dims.x);

        let loader = spawn_tile_loader(
            store.clone(),
            dataset.levels.iter().map(|l| l.path.clone()).collect(),
            dataset.levels.iter().map(|l| l.shape.clone()).collect(),
            dataset.levels.iter().map(|l| l.chunks.clone()).collect(),
            dataset.levels.iter().map(|l| l.dtype.clone()).collect(),
            dims_cyx,
            self.tile_loader_threads,
        )?;

        let raw_loader = if self.tiles_gl.is_some() {
            Some(
                spawn_raw_tile_loader(
                    store.clone(),
                    dataset.levels.iter().map(|l| l.path.clone()).collect(),
                    dataset.levels.iter().map(|l| l.shape.clone()).collect(),
                    dataset.levels.iter().map(|l| l.chunks.clone()).collect(),
                    dataset.levels.iter().map(|l| l.dtype.clone()).collect(),
                    dims_cyx,
                    self.tile_loader_threads,
                )
                .ok(),
            )
            .flatten()
        } else {
            None
        };

        let hist_loader = spawn_histogram_loader(
            store.clone(),
            dataset.levels.iter().map(|l| l.path.clone()).collect(),
            dataset.levels.iter().map(|l| l.shape.clone()).collect(),
            dataset.levels.iter().map(|l| l.dtype.clone()).collect(),
            dims_cyx,
        )?;

        let chanmax_level = choose_default_max_level(&dataset);
        let chanmax_loader = spawn_channel_max_loader(
            store.clone(),
            dataset.levels.iter().map(|l| l.path.clone()).collect(),
            dataset.levels.iter().map(|l| l.shape.clone()).collect(),
            dataset.levels.iter().map(|l| l.chunks.clone()).collect(),
            dataset.levels.iter().map(|l| l.dtype.clone()).collect(),
            dims_cyx,
        )?;

        let label_cells: Option<LabelZarrDataset> = None;
        let label_loader: Option<LabelTileLoaderHandle> = None;
        let label_cells_xform: Option<Vec<LabelToWorld>> = None;
        let labels_gl = if self.tiles_gl.is_some() {
            Some(
                self.labels_gl
                    .clone()
                    .unwrap_or_else(|| LabelsGl::new(1024)),
            )
        } else {
            None
        };

        let seg_label_names = dataset
            .source
            .local_path()
            .map(discover_label_names_local)
            .unwrap_or_default();
        let mut seg_label_selected = self.seg_label_selected.clone();
        if seg_label_selected.is_empty() {
            seg_label_selected = if seg_label_names.iter().any(|n| n == "cells") {
                "cells".to_string()
            } else if let Some(first) = seg_label_names.first() {
                first.clone()
            } else {
                "cells".to_string()
            };
        }
        if !seg_label_names.is_empty() && !seg_label_names.iter().any(|n| n == &seg_label_selected)
        {
            seg_label_selected = if seg_label_names.iter().any(|n| n == "cells") {
                "cells".to_string()
            } else {
                seg_label_names[0].clone()
            };
        }
        let seg_label_input = seg_label_selected.clone();
        let seg_label_status = String::new();

        // Commit the switch (after all fallible operations succeed).
        if let Some(tiles_gl) = self.tiles_gl.as_ref() {
            tiles_gl.reset();
        }
        if let Some(labels_gl) = labels_gl.as_ref() {
            labels_gl.reset();
        }

        self.dataset = dataset;
        self.store = store;
        self.remote_runtime = remote_runtime;
        self.channels = new_channels;
        self.channel_offsets_world = vec![egui::Vec2::ZERO; self.channels.len()];
        self.channel_scales = vec![egui::Vec2::splat(1.0); self.channels.len()];
        self.channel_rotations_rad = vec![0.0; self.channels.len()];
        self.points_offset_world = egui::Vec2::ZERO;
        self.spatial_points_offset_world = egui::Vec2::ZERO;
        self.mask_layers.clear();
        self.next_mask_layer_id = 1;
        self.drawing_mask_layer = None;
        self.seg_labels_offset_world = egui::Vec2::ZERO;
        self.seg_geojson_offset_world = egui::Vec2::ZERO;
        self.seg_objects_offset_world = egui::Vec2::ZERO;
        self.seg_geojson.clear();
        self.seg_objects.clear();
        self.spatial_layers.clear();
        self.spatial_root = None;
        self.spatial_label_store = None;
        self.loader = loader;
        self.raw_loader = raw_loader;
        self.hist_loader = hist_loader;
        self.chanmax_loader = chanmax_loader;
        self.chanmax_request_id = self.chanmax_request_id.wrapping_add(1).max(1);
        self.chanmax_level = chanmax_level;
        self.chanmax_pending = self
            .channels
            .iter()
            .map(|c| !self.channel_window_overrides.contains_key(&c.name))
            .collect();
        self.chanmax_snapshot = self.channels.iter().map(|c| c.window).collect();
        self.label_cells = label_cells;
        self.label_loader = label_loader;
        self.label_cells_xform = label_cells_xform;
        self.labels_gl = labels_gl;
        self.seg_label_names = seg_label_names;
        self.seg_label_selected = seg_label_selected;
        self.seg_label_input = seg_label_input;
        self.seg_label_status = seg_label_status;
        self.seg_label_prompt_open = self.tiles_gl.is_some() && !self.seg_label_names.is_empty();
        self.cells_outlines_visible = false;
        self.tiff_plane_state = None;
        self.configure_root_label_dataset_if_needed();

        if let Some(name) = prev_selected_name {
            if let Some(ch) = self.channels.iter().find(|c| c.name == name) {
                self.selected_channel = ch.index;
            } else {
                self.selected_channel = self
                    .selected_channel
                    .min(self.channels.len().saturating_sub(1));
            }
        } else {
            self.selected_channel = self
                .selected_channel
                .min(self.channels.len().saturating_sub(1));
        }
        if matches!(self.active_layer, LayerId::Channel(_)) {
            self.active_layer = LayerId::Channel(self.selected_channel);
        }

        self.rebuild_layer_orders();

        let new_world_w = self
            .dataset
            .levels
            .get(0)
            .map(|l| l.shape[self.dataset.dims.x] as f32)
            .unwrap_or(0.0);
        let new_world_h = self
            .dataset
            .levels
            .get(0)
            .map(|l| l.shape[self.dataset.dims.y] as f32)
            .unwrap_or(0.0);
        self.camera.center_world_lvl0 = egui::pos2(new_world_w * fx, new_world_h * fy);
        self.camera.zoom_screen_per_lvl0_px = old_zoom;

        self.cache = TileCache::new(256);
        self.pending.clear();
        self.previous_render_id = None;
        self.active_render_id = self.compute_render_id();
        self.restore_mask_layers_from_project_space();

        self.hist = None;
        self.hist_request_id = 0;
        self.hist_request_pending = false;
        self.hist_dirty = true;
        self.hist_last_sent = Instant::now()
            .checked_sub(Duration::from_secs(3600))
            .unwrap_or_else(Instant::now);

        self.request_default_channel_maxes();
        self.roi_selector
            .sync_to_dataset_source(&self.dataset.source);
        if let Some(local_root) = self.dataset.source.local_path() {
            self.cell_thresholds.set_dataset_root(
                local_root,
                self.dataset.multiscale.name.as_deref(),
                &mut self.cell_points,
            );
        }
        self.auto_load_project_roi_segmentation();
        self.drawing_mask_polygon.clear();
        if reload_exclusion_masks {
            if self.dataset.source.local_path().is_some() {
                if let Err(err) = self.ensure_exclusion_masks_loaded() {
                    self.roi_selector
                        .set_status(format!("Load Masks failed: {err}"));
                }
            } else {
                self.roi_selector
                    .set_status("Masks are supported for local datasets only.".to_string());
            }
        }

        // Best-effort fit if the new ROI is wildly different in size.
        if let Some(viewport) = ctx.input(|i| i.viewport().inner_rect) {
            if self.camera.zoom_screen_per_lvl0_px <= 0.0 {
                self.camera.fit_to_world_rect(
                    viewport,
                    egui::Rect::from_min_size(
                        egui::pos2(0.0, 0.0),
                        egui::vec2(new_world_w.max(1.0), new_world_h.max(1.0)),
                    ),
                );
            }
        }

        Ok(())
    }

    fn ensure_exclusion_masks_loaded(&mut self) -> anyhow::Result<usize> {
        let Some(local_root) = self.dataset.source.local_path() else {
            anyhow::bail!("exclusion masks are supported for local datasets only");
        };
        let Some(cfg) = self.roi_selector.masks_config_for_roi(local_root) else {
            anyhow::bail!("no matching dataset entry in Project config");
        };
        let entry = self.roi_selector.roi_entry_for_path(local_root);

        let resolved = resolve_masks_geojson_path_and_downsample(local_root, &cfg, entry.as_ref())?;
        let polylines = load_geojson_polylines_world(
            &resolved.geojson_path,
            resolved.downsample_factor,
            PolygonRingMode::AllRings,
        )
        .with_context(|| {
            format!(
                "failed to load masks: {}",
                resolved.geojson_path.to_string_lossy()
            )
        })?;

        let existing_idx = self.mask_layers.iter().position(|l| {
            !l.editable
                && l.source_geojson
                    .as_ref()
                    .is_some_and(|p| p == &resolved.geojson_path)
        });

        let idx = match existing_idx {
            Some(i) => i,
            None => {
                let id = self.next_mask_layer_id.max(1);
                self.next_mask_layer_id = id.saturating_add(1);
                self.mask_layers.push(MaskLayer {
                    id,
                    name: "Exclusion masks".to_string(),
                    visible: true,
                    opacity: 0.85,
                    width_screen_px: 1.5,
                    color_rgb: [50, 220, 255],
                    offset_world: egui::Vec2::ZERO,
                    editable: false,
                    polygons_world: Vec::new(),
                    source_geojson: Some(resolved.geojson_path.clone()),
                });
                self.mask_layers.len().saturating_sub(1)
            }
        };

        if let Some(l) = self.mask_layers.get_mut(idx) {
            l.polygons_world = polylines;
            l.source_geojson = Some(resolved.geojson_path);
            l.visible = true;
            l.editable = false;
        }

        Ok(self
            .mask_layers
            .get(idx)
            .map(|l| l.polygons_world.len())
            .unwrap_or(0))
    }

    fn refresh_seg_label_names_for_current_roi(&mut self) {
        if self.dataset.is_root_label_mask() {
            self.seg_label_names.clear();
            self.seg_label_selected = LabelZarrDataset::root_label_name(&self.dataset);
            self.seg_label_input = self.seg_label_selected.clone();
            self.seg_label_prompt_open = false;
            return;
        }

        self.seg_label_names = self
            .spatial_root
            .as_deref()
            .or_else(|| self.dataset.source.local_path())
            .map(discover_label_names_local)
            .unwrap_or_default();
        if self.seg_label_selected.trim().is_empty() {
            self.seg_label_selected = if self.seg_label_names.iter().any(|n| n == "cells") {
                "cells".to_string()
            } else if let Some(first) = self.seg_label_names.first() {
                first.clone()
            } else {
                "cells".to_string()
            };
        }
        if !self.seg_label_names.is_empty()
            && !self
                .seg_label_names
                .iter()
                .any(|n| n == &self.seg_label_selected)
        {
            self.seg_label_selected = if self.seg_label_names.iter().any(|n| n == "cells") {
                "cells".to_string()
            } else {
                self.seg_label_names[0].clone()
            };
        }
        if self.seg_label_input.trim().is_empty() || self.seg_label_input == self.seg_label_selected
        {
            self.seg_label_input = self.seg_label_selected.clone();
        }
    }

    fn ensure_segmentation_labels_loaded(&mut self) -> anyhow::Result<()> {
        let name = self.seg_label_selected.trim().to_string();
        if name.is_empty() {
            anyhow::bail!("label name is empty");
        }
        self.load_segmentation_labels(name.as_str())
    }

    fn load_segmentation_labels(&mut self, label_name: &str) -> anyhow::Result<()> {
        if self.dataset.is_root_label_mask() {
            return self.load_root_segmentation_labels();
        }
        if self.tiles_gl.is_none() {
            anyhow::bail!("segmentation overlay requires the GPU path");
        }
        self.labels_gl
            .get_or_insert_with(|| LabelsGl::new(1024))
            .reset();

        let label_store = self
            .spatial_label_store
            .as_ref()
            .cloned()
            .unwrap_or_else(|| self.store.clone());
        match LabelZarrDataset::try_open(label_store.clone(), label_name)? {
            Some(lbl) => {
                self.spatial_label_transform = self.spatial_label_transform_for_name(label_name);
                self.label_loader = spawn_label_tile_loader(
                    label_store,
                    lbl.levels.iter().map(|l| l.path.clone()).collect(),
                    lbl.levels.iter().map(|l| l.shape.clone()).collect(),
                    lbl.levels.iter().map(|l| l.chunks.clone()).collect(),
                    lbl.levels.iter().map(|l| l.dtype.clone()).collect(),
                    (lbl.dims.y, lbl.dims.x),
                )
                .ok();
                self.label_cells_xform = Some(compute_label_to_world_xforms(
                    &self.dataset,
                    &lbl,
                    self.spatial_label_transform,
                ));
                self.label_cells = Some(lbl);
                self.cells_outlines_visible = true;
                self.seg_label_selected = label_name.to_string();
                self.seg_label_input = self.seg_label_selected.clone();
                self.seg_label_prompt_open = false;
                self.rebuild_layer_orders();
                Ok(())
            }
            None => {
                self.label_cells = None;
                self.label_loader = None;
                self.label_cells_xform = None;
                anyhow::bail!("no labels/{label_name} found in this ROI")
            }
        }
    }

    fn ui_seg_label_prompt(&mut self, ctx: &egui::Context) {
        if !self.seg_label_prompt_open {
            return;
        }
        if self.tiles_gl.is_none() {
            self.seg_label_prompt_open = false;
            return;
        }
        if self.seg_label_names.is_empty() {
            self.seg_label_prompt_open = false;
            return;
        }

        match self.seg_label_prompt_preference {
            LabelPromptSessionPreference::AlwaysSkip => {
                self.seg_label_status.clear();
                self.seg_label_prompt_open = false;
                return;
            }
            LabelPromptSessionPreference::AlwaysLoad => {
                let name = self.seg_label_selected.trim().to_string();
                if name.is_empty() {
                    self.seg_label_prompt_preference = LabelPromptSessionPreference::Ask;
                    self.seg_label_prompt_always = false;
                } else {
                    match self.load_segmentation_labels(name.as_str()) {
                        Ok(()) => {
                            self.seg_label_status.clear();
                            self.set_active_layer(LayerId::SegmentationLabels);
                            self.bump_render_id();
                            self.seg_label_prompt_open = false;
                            return;
                        }
                        Err(err) => {
                            self.seg_label_status = format!("Load labels/{name} failed: {err}");
                            self.seg_label_prompt_preference = LabelPromptSessionPreference::Ask;
                            self.seg_label_prompt_always = false;
                        }
                    }
                }
            }
            LabelPromptSessionPreference::Ask => {}
        }

        let mut open = true;
        let mut request_close = false;
        egui::Window::new("Load labels?")
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.label(format!(
                    "Found {} label group(s) under labels/.",
                    self.seg_label_names.len()
                ));
                ui.add_space(6.0);

                ui.horizontal(|ui| {
                    ui.label("Label");
                    egui::ComboBox::from_id_salt("seg_label_prompt_select")
                        .selected_text(self.seg_label_selected.clone())
                        .show_ui(ui, |ui| {
                            for name in self.seg_label_names.clone() {
                                ui.selectable_value(
                                    &mut self.seg_label_selected,
                                    name.clone(),
                                    name,
                                );
                            }
                        });
                });

                if !self.seg_label_status.trim().is_empty() {
                    ui.add_space(6.0);
                    ui.label(self.seg_label_status.clone());
                }

                ui.add_space(8.0);
                ui.checkbox(&mut self.seg_label_prompt_always, "Always");

                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    if ui.button("Skip").clicked() {
                        self.seg_label_prompt_preference = if self.seg_label_prompt_always {
                            LabelPromptSessionPreference::AlwaysSkip
                        } else {
                            LabelPromptSessionPreference::Ask
                        };
                        self.seg_label_status.clear();
                        request_close = true;
                    }
                    if ui.button("Load labels").clicked() {
                        let name = self.seg_label_selected.trim().to_string();
                        if name.is_empty() {
                            self.seg_label_status = "Label name is empty.".to_string();
                        } else {
                            match self.load_segmentation_labels(name.as_str()) {
                                Ok(()) => {
                                    self.seg_label_prompt_preference =
                                        if self.seg_label_prompt_always {
                                            LabelPromptSessionPreference::AlwaysLoad
                                        } else {
                                            LabelPromptSessionPreference::Ask
                                        };
                                    self.seg_label_status.clear();
                                    self.set_active_layer(LayerId::SegmentationLabels);
                                    self.bump_render_id();
                                    request_close = true;
                                }
                                Err(err) => {
                                    self.seg_label_status =
                                        format!("Load labels/{name} failed: {err}");
                                }
                            }
                        }
                    }
                });
            });

        if request_close {
            open = false;
        }
        if !open {
            self.seg_label_prompt_open = false;
        }
    }

    fn ui_remote_dialog(&mut self, ctx: &egui::Context) {
        let mut open = self.remote_dialog_open;
        let mut s3_inputs_changed = false;
        let mut connect_s3 = false;
        let mut refresh_s3 = false;
        let mut browse_to: Option<String> = None;
        let mut remote_open_mosaic = false;
        egui::Window::new("Open Remote OME-Zarr")
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.remote_mode, RemoteMode::Http, "HTTP(S)");
                    ui.selectable_value(&mut self.remote_mode, RemoteMode::S3, "S3 / R2");
                });
                ui.separator();

                match self.remote_mode {
                    RemoteMode::Http => {
                        ui.label("Dataset URL (points to the OME-Zarr directory):");
                        ui.text_edit_singleline(&mut self.remote_http_url);
                        ui.label("Example: https://host/path/ROI1.ome.zarr");
                    }
                    RemoteMode::S3 => {
                        ui.label("Endpoint (R2): https://<accountid>.r2.cloudflarestorage.com");
                        s3_inputs_changed |= ui
                            .text_edit_singleline(&mut self.remote_s3_endpoint)
                            .changed();
                        ui.horizontal(|ui| {
                            ui.label("Region:");
                            s3_inputs_changed |= ui
                                .text_edit_singleline(&mut self.remote_s3_region)
                                .changed();
                        });
                        ui.horizontal(|ui| {
                            ui.label("Bucket:");
                            s3_inputs_changed |= ui
                                .text_edit_singleline(&mut self.remote_s3_bucket)
                                .changed();
                        });
                        ui.label("Prefix (path to the OME-Zarr directory within the bucket):");
                        s3_inputs_changed |= ui
                            .text_edit_singleline(&mut self.remote_s3_prefix)
                            .changed();
                        ui.separator();
                        ui.label("Credentials (static):");
                        ui.horizontal(|ui| {
                            ui.label("Access key:");
                            s3_inputs_changed |= ui
                                .text_edit_singleline(&mut self.remote_s3_access_key)
                                .changed();
                        });
                        ui.horizontal(|ui| {
                            ui.label("Secret key:");
                            s3_inputs_changed |= ui
                                .add(
                                    egui::TextEdit::singleline(&mut self.remote_s3_secret_key)
                                        .password(true),
                                )
                                .changed();
                        });

                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            let connect_label = if self.remote_s3_browser.is_some() {
                                "Reconnect"
                            } else {
                                "Connect"
                            };
                            if ui.button(connect_label).clicked() {
                                connect_s3 = true;
                            }
                            if ui
                                .add_enabled(
                                    self.remote_s3_browser.is_some(),
                                    egui::Button::new("Refresh"),
                                )
                                .clicked()
                            {
                                refresh_s3 = true;
                            }
                        });

                        let browser_view = self.remote_s3_browser.as_ref().map(|state| {
                            (
                                state.current_prefix.clone(),
                                state.parent_prefix.clone(),
                                state.current_is_dataset,
                                state.entries.clone(),
                                state.selected_dataset_prefixes.clone(),
                            )
                        });

                        if let Some((
                            current_prefix,
                            parent_prefix,
                            current_is_dataset,
                            entries,
                            mut selected_prefixes,
                        )) = browser_view
                        {
                            ui.add_space(6.0);
                            ui.separator();
                            egui::Frame::group(ui.style()).show(ui, |ui| {
                                ui.set_min_width(620.0);
                                ui.horizontal(|ui| {
                                    ui.label("Browser");
                                    ui.label(if current_prefix.is_empty() {
                                        "<bucket root>".to_string()
                                    } else {
                                        current_prefix.clone()
                                    });
                                    if ui
                                        .add_enabled(
                                            parent_prefix.is_some() || !current_prefix.is_empty(),
                                            egui::Button::new("Up"),
                                        )
                                        .clicked()
                                    {
                                        browse_to = Some(parent_prefix.unwrap_or_default());
                                    }
                                });
                                if current_is_dataset {
                                    ui.horizontal(|ui| {
                                        let mut selected =
                                            selected_prefixes.contains(&current_prefix);
                                        if ui.checkbox(&mut selected, "Select current").changed() {
                                            if selected {
                                                selected_prefixes.insert(current_prefix.clone());
                                            } else {
                                                selected_prefixes.remove(&current_prefix);
                                            }
                                        }
                                        ui.label("This prefix looks like an OME-Zarr dataset.");
                                        if ui.button("Use this prefix").clicked() {
                                            self.remote_s3_prefix = current_prefix.clone();
                                        }
                                    });
                                }
                                ui.separator();
                                ui.horizontal(|ui| {
                                    ui.add_sized([28.0, 18.0], egui::Label::new("Sel"));
                                    ui.small("Name");
                                });
                                egui::ScrollArea::vertical()
                                    .auto_shrink([false, false])
                                    .max_height(260.0)
                                    .show(ui, |ui| {
                                        ui.set_min_width(ui.available_width());
                                        for entry in &entries {
                                            ui.horizontal(|ui| {
                                                if entry.is_dataset {
                                                    let mut selected =
                                                        selected_prefixes.contains(&entry.prefix);
                                                    if ui
                                                        .add_sized(
                                                            [28.0, 20.0],
                                                            egui::Checkbox::without_text(
                                                                &mut selected,
                                                            ),
                                                        )
                                                        .on_hover_text("Select this OME-Zarr")
                                                        .changed()
                                                    {
                                                        if selected {
                                                            selected_prefixes
                                                                .insert(entry.prefix.clone());
                                                        } else {
                                                            selected_prefixes.remove(&entry.prefix);
                                                        }
                                                    }
                                                    if ui
                                                        .selectable_label(
                                                            self.remote_s3_prefix.trim()
                                                                == entry.prefix,
                                                            format!(
                                                                "[{}] {}",
                                                                if entry
                                                                    .prefix
                                                                    .ends_with(".ome.zarr")
                                                                {
                                                                    "OME-Zarr"
                                                                } else {
                                                                    "Zarr"
                                                                },
                                                                entry.name
                                                            ),
                                                        )
                                                        .clicked()
                                                    {
                                                        self.remote_s3_prefix =
                                                            entry.prefix.clone();
                                                    }
                                                    if ui.small_button("Browse").clicked() {
                                                        browse_to = Some(entry.prefix.clone());
                                                    }
                                                } else {
                                                    ui.add_space(28.0);
                                                    if ui
                                                        .button(format!("[dir] {}", entry.name))
                                                        .clicked()
                                                    {
                                                        browse_to = Some(entry.prefix.clone());
                                                    }
                                                }
                                            });
                                        }
                                    });
                            });
                            if let Some(state) = self.remote_s3_browser.as_mut() {
                                state.selected_dataset_prefixes = selected_prefixes;
                            }
                        }
                    }
                }

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        self.remote_dialog_open = false;
                        self.remote_status.clear();
                    }
                    if ui.button("Open").clicked() {
                        match self.open_remote_dataset(ctx) {
                            Ok(_) => {
                                self.remote_dialog_open = false;
                                self.remote_status.clear();
                            }
                            Err(err) => {
                                self.remote_status = format!("{err}");
                            }
                        }
                    }
                    let selected_remote = self.selected_remote_s3_datasets();
                    if ui
                        .add_enabled(
                            self.remote_mode == RemoteMode::S3 && selected_remote.len() >= 2,
                            egui::Button::new(format!("Open Mosaic ({})", selected_remote.len())),
                        )
                        .clicked()
                    {
                        remote_open_mosaic = true;
                    }
                });

                if !self.remote_status.is_empty() {
                    ui.add_space(6.0);
                    ui.label(self.remote_status.clone());
                }
            });

        if s3_inputs_changed {
            self.clear_remote_s3_browser();
        }
        if connect_s3 {
            match self.connect_remote_s3_browser() {
                Ok(()) => self.remote_status.clear(),
                Err(err) => self.remote_status = format!("{err}"),
            }
        } else if refresh_s3 {
            match self.refresh_remote_s3_browser() {
                Ok(()) => self.remote_status.clear(),
                Err(err) => self.remote_status = format!("{err}"),
            }
        } else if let Some(prefix) = browse_to {
            match self.browse_remote_s3_prefix(prefix) {
                Ok(()) => self.remote_status.clear(),
                Err(err) => self.remote_status = format!("{err}"),
            }
        } else if remote_open_mosaic {
            let selected = self.selected_remote_s3_datasets();
            if selected.len() >= 2 {
                self.pending_request = Some(ViewerRequest::OpenRemoteS3Mosaic(selected));
                self.remote_dialog_open = false;
                self.remote_status.clear();
            } else {
                self.remote_status = "Select at least 2 S3 OME-Zarr datasets.".to_string();
            }
        }
        self.remote_dialog_open = open;
    }

    fn remote_s3_signature(&self) -> String {
        format!(
            "{}\n{}\n{}\n{}\n{}",
            self.remote_s3_endpoint.trim(),
            self.remote_s3_region.trim(),
            self.remote_s3_bucket.trim(),
            self.remote_s3_access_key.trim(),
            self.remote_s3_secret_key.trim()
        )
    }

    fn clear_remote_s3_browser(&mut self) {
        self.remote_s3_browser = None;
    }

    fn connect_remote_s3_browser(&mut self) -> anyhow::Result<()> {
        let browser = build_s3_browser(
            &self.remote_s3_endpoint,
            &self.remote_s3_region,
            &self.remote_s3_bucket,
            &self.remote_s3_access_key,
            &self.remote_s3_secret_key,
        )?;
        let signature = self.remote_s3_signature();
        let browse_prefix = if self.remote_s3_prefix.trim().ends_with(".ome.zarr")
            || self.remote_s3_prefix.trim().ends_with(".zarr")
        {
            self.remote_s3_prefix
                .trim()
                .trim_matches('/')
                .rsplit_once('/')
                .map(|(parent, _)| parent.to_string())
                .unwrap_or_default()
        } else {
            self.remote_s3_prefix.trim().trim_matches('/').to_string()
        };
        let listing = list_s3_prefix(&browser, &browse_prefix)?;
        self.apply_remote_s3_listing(browser, signature, listing, HashSet::new());
        Ok(())
    }

    fn refresh_remote_s3_browser(&mut self) -> anyhow::Result<()> {
        let Some(state) = self.remote_s3_browser.take() else {
            anyhow::bail!("not connected to S3");
        };
        let listing = list_s3_prefix(&state.session, &state.current_prefix)?;
        let mut cache = state.listing_cache;
        let selected = state.selected_dataset_prefixes;
        cache.insert(listing.prefix.clone(), listing.clone());
        self.apply_remote_s3_listing(state.session, state.signature, listing, selected);
        if let Some(next) = self.remote_s3_browser.as_mut() {
            next.listing_cache = cache;
        }
        Ok(())
    }

    fn browse_remote_s3_prefix(&mut self, prefix: String) -> anyhow::Result<()> {
        let Some(state) = self.remote_s3_browser.take() else {
            anyhow::bail!("not connected to S3");
        };
        let mut cache = state.listing_cache;
        let selected = state.selected_dataset_prefixes;
        let listing = if let Some(cached) = cache.get(&prefix).cloned() {
            cached
        } else {
            let listing = list_s3_prefix(&state.session, &prefix)?;
            cache.insert(prefix.clone(), listing.clone());
            listing
        };
        self.apply_remote_s3_listing(state.session, state.signature, listing, selected);
        if let Some(next) = self.remote_s3_browser.as_mut() {
            next.listing_cache = cache;
        }
        Ok(())
    }

    fn apply_remote_s3_listing(
        &mut self,
        session: S3Browser,
        signature: String,
        listing: S3BrowseListing,
        selected_dataset_prefixes: HashSet<String>,
    ) {
        let current_prefix = listing.prefix.clone();
        let parent_prefix = listing.parent_prefix.clone();
        let entries = listing.entries.clone();
        let current_is_dataset = listing.current_is_dataset;
        self.remote_s3_browser = Some(RemoteS3BrowserState {
            session,
            signature,
            current_prefix: current_prefix.clone(),
            parent_prefix,
            entries,
            current_is_dataset,
            selected_dataset_prefixes,
            listing_cache: HashMap::from([(current_prefix, listing)]),
        });
    }

    fn selected_remote_s3_datasets(&self) -> Vec<S3DatasetSelection> {
        let Some(state) = self.remote_s3_browser.as_ref() else {
            return Vec::new();
        };
        let endpoint = self.remote_s3_endpoint.trim().to_string();
        let region = self.remote_s3_region.trim().to_string();
        let bucket = self.remote_s3_bucket.trim().to_string();
        let access_key = self.remote_s3_access_key.trim().to_string();
        let secret_key = self.remote_s3_secret_key.trim().to_string();
        let mut prefixes = state
            .selected_dataset_prefixes
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        prefixes.sort();
        prefixes
            .into_iter()
            .map(|prefix| S3DatasetSelection {
                endpoint: endpoint.clone(),
                region: region.clone(),
                bucket: bucket.clone(),
                prefix,
                access_key: access_key.clone(),
                secret_key: secret_key.clone(),
            })
            .collect()
    }

    fn open_remote_dataset(&mut self, ctx: &egui::Context) -> anyhow::Result<()> {
        match self.remote_mode {
            RemoteMode::Http => {
                let mut url = self
                    .remote_http_url
                    .trim()
                    .trim_end_matches('/')
                    .to_string();
                if url.is_empty() {
                    anyhow::bail!("URL is empty");
                }
                if !url.starts_with("http://") && !url.starts_with("https://") {
                    url = format!("https://{url}");
                }
                let store = build_http_store(&url)?;
                let source = crate::data::dataset_source::DatasetSource::Http { base_url: url };
                let dataset = crate::data::ome::OmeZarrDataset::open_with_store(source, store.clone())?;
                self.switch_dataset_with_store(ctx, dataset, store, None)
            }
            RemoteMode::S3 => {
                let mut endpoint = self.remote_s3_endpoint.trim().to_string();
                let region = self.remote_s3_region.trim().to_string();
                let bucket = self.remote_s3_bucket.trim().to_string();
                let prefix = self.remote_s3_prefix.trim().trim_matches('/').to_string();
                let access_key = self.remote_s3_access_key.trim().to_string();
                let secret_key = self.remote_s3_secret_key.trim().to_string();

                if endpoint.is_empty() || bucket.is_empty() {
                    anyhow::bail!("endpoint and bucket are required");
                }
                if access_key.is_empty() || secret_key.is_empty() {
                    anyhow::bail!("access key / secret key are required");
                }
                if !endpoint.starts_with("http://") && !endpoint.starts_with("https://") {
                    endpoint = format!("https://{endpoint}");
                }

                let S3Store { store, runtime } = build_s3_store(
                    &endpoint,
                    &region,
                    &bucket,
                    &prefix,
                    &access_key,
                    &secret_key,
                )?;
                let source = crate::data::dataset_source::DatasetSource::S3 {
                    endpoint,
                    region: if region.is_empty() {
                        "auto".to_string()
                    } else {
                        region
                    },
                    bucket,
                    prefix,
                };
                let dataset = crate::data::ome::OmeZarrDataset::open_with_store(source, store.clone())?;
                self.switch_dataset_with_store(ctx, dataset, store, Some(runtime))
            }
        }
    }

    fn clear_spatial_selection_drag(&mut self) {
        self.selection_rect_start_world = None;
        self.selection_rect_current_world = None;
        self.selection_lasso_world.clear();
    }

    fn active_layer_supports_spatial_selection(&self) -> bool {
        match self.active_layer {
            LayerId::SegmentationObjects => self.seg_objects.object_count() > 0,
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter()
                .find(|layer| layer.id == id)
                .is_some_and(|layer| layer.has_object_layer()),
            _ => false,
        }
    }

    fn apply_rect_selection_to_active_layer(
        &mut self,
        world_rect: egui::Rect,
        additive: bool,
    ) -> usize {
        match self.active_layer {
            LayerId::SegmentationObjects => self.seg_objects.select_in_world_rect(
                world_rect,
                self.seg_objects_offset_world,
                additive,
            ),
            LayerId::SpatialShape(id) => {
                let Some(layer) = self
                    .spatial_layers
                    .shapes
                    .iter_mut()
                    .find(|layer| layer.id == id)
                else {
                    return 0;
                };
                let offset_world = layer.offset_world;
                layer.object_layer_mut().map_or(0, |objects| {
                    objects.select_in_world_rect(world_rect, offset_world, additive)
                })
            }
            _ => 0,
        }
    }

    fn apply_lasso_selection_to_active_layer(
        &mut self,
        world_points: &[egui::Pos2],
        additive: bool,
    ) -> usize {
        match self.active_layer {
            LayerId::SegmentationObjects => self.seg_objects.select_in_world_lasso(
                world_points,
                self.seg_objects_offset_world,
                additive,
            ),
            LayerId::SpatialShape(id) => {
                let Some(layer) = self
                    .spatial_layers
                    .shapes
                    .iter_mut()
                    .find(|layer| layer.id == id)
                else {
                    return 0;
                };
                let offset_world = layer.offset_world;
                layer.object_layer_mut().map_or(0, |objects| {
                    objects.select_in_world_lasso(world_points, offset_world, additive)
                })
            }
            _ => 0,
        }
    }

    fn active_object_selection_count(&self) -> usize {
        match self.active_layer {
            LayerId::SegmentationObjects => self.seg_objects.selection_count(),
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer())
                .map(|objects| objects.selection_count())
                .unwrap_or(0),
            _ => 0,
        }
    }

    fn active_object_selection_elements_snapshot(&self) -> Vec<(usize, String, usize)> {
        match self.active_layer {
            LayerId::SegmentationObjects => self.seg_objects.selection_elements_snapshot(),
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer())
                .map(|objects| objects.selection_elements_snapshot())
                .unwrap_or_default(),
            _ => Vec::new(),
        }
    }

    fn create_selection_element_from_active_selection(&mut self) -> usize {
        match self.active_layer {
            LayerId::SegmentationObjects => self
                .seg_objects
                .create_selection_element_from_current_selection_with_name(None),
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer_mut())
                .map(|objects| {
                    objects.create_selection_element_from_current_selection_with_name(None)
                })
                .unwrap_or(0),
            _ => 0,
        }
    }

    fn add_active_selection_to_element(&mut self, element_idx: usize) -> usize {
        match self.active_layer {
            LayerId::SegmentationObjects => self
                .seg_objects
                .add_current_selection_to_element(element_idx),
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer_mut())
                .map(|objects| objects.add_current_selection_to_element(element_idx))
                .unwrap_or(0),
            _ => 0,
        }
    }

    fn selected_channel_visible_data_rect_lvl0(
        &self,
        viewport: egui::Rect,
        ch_idx: usize,
    ) -> egui::Rect {
        let visible_world = self.visible_world_rect(viewport);
        let corners = [
            visible_world.left_top(),
            egui::pos2(visible_world.right(), visible_world.top()),
            visible_world.right_bottom(),
            egui::pos2(visible_world.left(), visible_world.bottom()),
        ];
        let pivot = self.image_world_rect_lvl0().center();
        let off = self
            .channel_offsets_world
            .get(ch_idx)
            .copied()
            .unwrap_or_default();
        let scale = self
            .channel_scales
            .get(ch_idx)
            .copied()
            .unwrap_or(egui::Vec2::splat(1.0));
        let rot = self
            .channel_rotations_rad
            .get(ch_idx)
            .copied()
            .unwrap_or(0.0);

        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for &corner in &corners {
            let local = inv_xform_world_point(corner, pivot, off, scale, rot);
            min_x = min_x.min(local.x);
            min_y = min_y.min(local.y);
            max_x = max_x.max(local.x);
            max_y = max_y.max(local.y);
        }

        egui::Rect::from_min_max(egui::pos2(min_x, min_y), egui::pos2(max_x, max_y))
            .intersect(self.image_local_rect_lvl0())
    }

    fn selected_channel_local_to_world(&self, ch_idx: usize, local: egui::Pos2) -> egui::Pos2 {
        let pivot = self.image_world_rect_lvl0().center();
        let off = self
            .channel_offsets_world
            .get(ch_idx)
            .copied()
            .unwrap_or_default();
        let scale = self
            .channel_scales
            .get(ch_idx)
            .copied()
            .unwrap_or(egui::Vec2::splat(1.0));
        let rot = self
            .channel_rotations_rad
            .get(ch_idx)
            .copied()
            .unwrap_or(0.0);
        xform_screen_point(local, pivot, off, scale, rot)
    }

    fn uses_gpu_threshold_region_preview(&self, preview: &ThresholdRegionPreview) -> bool {
        self.threshold_preview_gl.is_some() && preview.min_component_pixels <= 1
    }

    fn start_threshold_region_preview(&mut self, ctx: &egui::Context) -> anyhow::Result<()> {
        let viewport = self
            .last_canvas_rect
            .ok_or_else(|| anyhow::anyhow!("canvas viewport unavailable"))?;
        let ch_idx = self
            .selected_channel
            .min(self.channels.len().saturating_sub(1));
        let Some(level_info) = self.dataset.levels.get(self.choose_level()) else {
            anyhow::bail!("invalid image level");
        };

        let visible_rect_lvl0 = self.selected_channel_visible_data_rect_lvl0(viewport, ch_idx);
        if visible_rect_lvl0.width() <= 0.0 || visible_rect_lvl0.height() <= 0.0 {
            anyhow::bail!("no visible region intersects the active channel");
        }

        let downsample = level_info.downsample.max(1e-6);
        let y_dim = self.dataset.dims.y;
        let x_dim = self.dataset.dims.x;
        let c_dim = self.dataset.dims.c;
        let x0 = (visible_rect_lvl0.left() / downsample).floor().max(0.0) as u64;
        let y0 = (visible_rect_lvl0.top() / downsample).floor().max(0.0) as u64;
        let x1 = (visible_rect_lvl0.right() / downsample)
            .ceil()
            .min(level_info.shape[x_dim] as f32) as u64;
        let y1 = (visible_rect_lvl0.bottom() / downsample)
            .ceil()
            .min(level_info.shape[y_dim] as f32) as u64;
        if x1 <= x0 || y1 <= y0 {
            anyhow::bail!("visible region is empty at this level");
        }

        let mut ranges = Vec::with_capacity(level_info.shape.len());
        for dim in 0..level_info.shape.len() {
            if Some(dim) == c_dim {
                let channel_index = self
                    .channels
                    .get(ch_idx)
                    .map(|ch| ch.index as u64)
                    .unwrap_or(0);
                ranges.push(channel_index..(channel_index + 1));
            } else if dim == y_dim {
                ranges.push(y0..y1);
            } else if dim == x_dim {
                ranges.push(x0..x1);
            } else {
                ranges.push(0..1);
            }
        }

        let zarr_path = format!("/{}", level_info.path.trim_start_matches('/'));
        let array = Array::open(self.store.clone(), &zarr_path).with_context(|| {
            format!("failed to open image array for level {}", level_info.index)
        })?;
        let subset = ArraySubset::new_with_ranges(&ranges);
        let data = retrieve_image_subset_u16(&array, &subset, &level_info.dtype)
            .context("failed to read active-channel viewport subset")?;
        let plane = if c_dim.is_some() {
            data.into_dimensionality::<ndarray::Ix3>()
                .ok()
                .map(|a| a.index_axis(ndarray::Axis(0), 0).to_owned())
        } else {
            data.into_dimensionality::<ndarray::Ix2>().ok()
        }
        .context("unexpected dimensionality for threshold region subset")?;

        let threshold = self
            .channels
            .get(ch_idx)
            .and_then(|channel| channel.window)
            .map(|(lo, _)| lo.round().clamp(0.0, u16::MAX as f32) as u16)
            .unwrap_or(0);
        let channel_name = self
            .channels
            .get(ch_idx)
            .map(|channel| channel.name.clone())
            .unwrap_or_else(|| format!("Channel {ch_idx}"));
        let raw_values = Arc::new(plane.iter().copied().collect::<Vec<_>>());
        let generation = self.threshold_region_preview_generation;
        self.threshold_region_preview_generation =
            self.threshold_region_preview_generation.wrapping_add(1);
        let mut preview = ThresholdRegionPreview {
            generation,
            channel_index: ch_idx,
            channel_name,
            level_index: level_info.index,
            downsample,
            x0,
            y0,
            plane,
            raw_values,
            threshold,
            min_component_pixels: self.threshold_region_min_pixels.max(1),
            mask: ThresholdRegionMask {
                width: 0,
                height: 0,
                included: Vec::new(),
            },
            texture: None,
        };
        if !self.uses_gpu_threshold_region_preview(&preview) {
            Self::recompute_threshold_region_preview_cpu_data(ctx, &mut preview);
        }
        self.threshold_region_status = Self::threshold_region_preview_status_message(
            &preview,
            self.uses_gpu_threshold_region_preview(&preview),
        );
        self.threshold_region_preview = Some(preview);
        Ok(())
    }

    fn recompute_threshold_region_preview(&mut self, ctx: &egui::Context) {
        let gpu_available = self.threshold_preview_gl.is_some();
        if let Some(preview) = self.threshold_region_preview.as_mut() {
            preview.min_component_pixels = self.threshold_region_min_pixels.max(1);
            let uses_gpu = gpu_available && preview.min_component_pixels <= 1;
            if uses_gpu {
                preview.mask = ThresholdRegionMask {
                    width: 0,
                    height: 0,
                    included: Vec::new(),
                };
                preview.texture = None;
            } else {
                Self::recompute_threshold_region_preview_cpu_data(ctx, preview);
            }
            self.threshold_region_status =
                Self::threshold_region_preview_status_message(preview, uses_gpu);
        }
    }

    fn threshold_region_preview_status_message(
        preview: &ThresholdRegionPreview,
        uses_gpu: bool,
    ) -> String {
        if uses_gpu {
            format!(
                "Previewing {} at level {} on the GPU (threshold only; min component filtering is applied on Apply).",
                preview.channel_name, preview.level_index
            )
        } else {
            let included = preview
                .mask
                .included
                .iter()
                .filter(|included| **included)
                .count();
            format!(
                "Preview: {} pixels selected in {} at level {}.",
                included, preview.channel_name, preview.level_index
            )
        }
    }

    fn recompute_threshold_region_preview_cpu_data(
        ctx: &egui::Context,
        preview: &mut ThresholdRegionPreview,
    ) {
        preview.mask = extract_threshold_region_mask(
            &preview.plane,
            preview.threshold,
            preview.min_component_pixels,
        );
        let mut rgba = vec![0u8; preview.mask.width * preview.mask.height * 4];
        for (idx, included) in preview.mask.included.iter().copied().enumerate() {
            if !included {
                continue;
            }
            let base = idx * 4;
            rgba[base] = 255;
            rgba[base + 1] = 210;
            rgba[base + 2] = 80;
            rgba[base + 3] = 120;
        }
        let image = egui::ColorImage::from_rgba_unmultiplied(
            [preview.mask.width, preview.mask.height],
            &rgba,
        );
        let options = egui::TextureOptions::NEAREST;
        if let Some(texture) = preview.texture.as_mut() {
            texture.set(image, options);
        } else {
            preview.texture = Some(ctx.load_texture(
                format!(
                    "threshold-preview-{}-{}-{}-{}",
                    preview.channel_index, preview.level_index, preview.x0, preview.y0
                ),
                image,
                options,
            ));
        }
    }

    fn create_threshold_mask_from_preview(&mut self) -> anyhow::Result<usize> {
        let Some(preview) = self.threshold_region_preview.as_ref() else {
            anyhow::bail!("no threshold preview is active");
        };
        let mask = extract_threshold_region_mask(
            &preview.plane,
            preview.threshold,
            preview.min_component_pixels,
        );
        let polygons = threshold_region_mask_to_polygons(&mask);
        if polygons.is_empty() {
            anyhow::bail!("no visible regions found above the current threshold");
        }

        let channel_index = preview.channel_index;
        let channel_name = preview.channel_name.clone();
        let level_index = preview.level_index;
        let x0 = preview.x0;
        let y0 = preview.y0;
        let downsample = preview.downsample;
        let layer_id = self.create_editable_mask_layer(Some(format!("Threshold {channel_name}")));
        let mut created = 0usize;
        let world_polygons = polygons
            .into_iter()
            .map(|polygon| {
                polygon
                    .into_iter()
                    .map(|point| {
                        let local_lvl0 = egui::pos2(
                            (x0 as f32 + point.x) * downsample,
                            (y0 as f32 + point.y) * downsample,
                        );
                        self.selected_channel_local_to_world(channel_index, local_lvl0)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        if let Some(layer) = self
            .mask_layers
            .iter_mut()
            .find(|layer| layer.id == layer_id)
        {
            layer.offset_world = egui::Vec2::ZERO;
            layer.polygons_world.clear();
            for polygon in world_polygons {
                layer.add_closed_polygon(polygon);
            }
            layer.visible = true;
            created = layer.polygons_world.len();
        }

        self.threshold_region_preview = None;
        self.active_layer = LayerId::Mask(layer_id);
        self.threshold_region_status = format!(
            "Created {created} threshold region(s) from {channel_name} at level {level_index}."
        );
        self.rebuild_layer_orders();
        self.bump_render_id();
        Ok(created)
    }

    fn draw_threshold_region_preview(&self, ui: &mut egui::Ui, rect: egui::Rect) {
        let Some(preview) = self.threshold_region_preview.as_ref() else {
            return;
        };
        let width = preview.plane.dim().1;
        let height = preview.plane.dim().0;
        if width == 0 || height == 0 {
            return;
        }

        let x0 = preview.x0 as f32 * preview.downsample;
        let y0 = preview.y0 as f32 * preview.downsample;
        let x1 = (preview.x0 as f32 + width as f32) * preview.downsample;
        let y1 = (preview.y0 as f32 + height as f32) * preview.downsample;
        let corners_world = [
            self.selected_channel_local_to_world(preview.channel_index, egui::pos2(x0, y0)),
            self.selected_channel_local_to_world(preview.channel_index, egui::pos2(x1, y0)),
            self.selected_channel_local_to_world(preview.channel_index, egui::pos2(x1, y1)),
            self.selected_channel_local_to_world(preview.channel_index, egui::pos2(x0, y1)),
        ];
        let corners_screen = corners_world.map(|point| self.camera.world_to_screen(point, rect));
        let uses_gpu = self.uses_gpu_threshold_region_preview(preview);
        if uses_gpu {
            if let Some(renderer) = self.threshold_preview_gl.clone() {
                let data = ThresholdPreviewGlDrawData {
                    generation: preview.generation,
                    width,
                    height,
                    values: preview.raw_values.clone(),
                };
                let params = ThresholdPreviewGlDrawParams {
                    visible: true,
                    quad_screen: corners_screen,
                    threshold_u16: preview.threshold,
                    tint: egui::Color32::from_rgba_unmultiplied(255, 210, 80, 120),
                };
                let cb = egui_glow::CallbackFn::new(move |info, painter| {
                    renderer.paint(info, painter, &data, &params);
                });
                ui.painter().add(egui::PaintCallback {
                    rect,
                    callback: Arc::new(cb),
                });
            }
        } else {
            let Some(texture) = preview.texture.as_ref() else {
                return;
            };
            let mut mesh = egui::Mesh::with_texture(texture.id());
            let base = mesh.vertices.len() as u32;
            mesh.vertices.push(egui::epaint::Vertex {
                pos: corners_screen[0],
                uv: egui::pos2(0.0, 0.0),
                color: egui::Color32::WHITE,
            });
            mesh.vertices.push(egui::epaint::Vertex {
                pos: corners_screen[1],
                uv: egui::pos2(1.0, 0.0),
                color: egui::Color32::WHITE,
            });
            mesh.vertices.push(egui::epaint::Vertex {
                pos: corners_screen[2],
                uv: egui::pos2(1.0, 1.0),
                color: egui::Color32::WHITE,
            });
            mesh.vertices.push(egui::epaint::Vertex {
                pos: corners_screen[3],
                uv: egui::pos2(0.0, 1.0),
                color: egui::Color32::WHITE,
            });
            mesh.indices
                .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            ui.painter().add(egui::Shape::mesh(mesh));
        }

        ui.painter().add(egui::Shape::closed_line(
            vec![
                corners_screen[0],
                corners_screen[1],
                corners_screen[2],
                corners_screen[3],
            ],
            egui::Stroke::new(
                1.5,
                egui::Color32::from_rgba_unmultiplied(255, 210, 80, 220),
            ),
        ));

        let pixel_step_x_screen = {
            let p0 =
                self.selected_channel_local_to_world(preview.channel_index, egui::pos2(x0, y0));
            let p1 = self.selected_channel_local_to_world(
                preview.channel_index,
                egui::pos2(x0 + preview.downsample, y0),
            );
            self.camera
                .world_to_screen(p0, rect)
                .distance(self.camera.world_to_screen(p1, rect))
        };
        let pixel_step_y_screen = {
            let p0 =
                self.selected_channel_local_to_world(preview.channel_index, egui::pos2(x0, y0));
            let p1 = self.selected_channel_local_to_world(
                preview.channel_index,
                egui::pos2(x0, y0 + preview.downsample),
            );
            self.camera
                .world_to_screen(p0, rect)
                .distance(self.camera.world_to_screen(p1, rect))
        };
        let show_grid = pixel_step_x_screen.max(pixel_step_y_screen) >= 12.0
            && width.saturating_add(height) <= 2048;
        if show_grid {
            let grid_stroke = egui::Stroke::new(
                1.0,
                egui::Color32::from_rgba_unmultiplied(255, 255, 255, 72),
            );
            for x in 0..=width {
                let local_x = x0 + x as f32 * preview.downsample;
                let p0 = self.selected_channel_local_to_world(
                    preview.channel_index,
                    egui::pos2(local_x, y0),
                );
                let p1 = self.selected_channel_local_to_world(
                    preview.channel_index,
                    egui::pos2(local_x, y1),
                );
                ui.painter().line_segment(
                    [
                        self.camera.world_to_screen(p0, rect),
                        self.camera.world_to_screen(p1, rect),
                    ],
                    grid_stroke,
                );
            }
            for y in 0..=height {
                let local_y = y0 + y as f32 * preview.downsample;
                let p0 = self.selected_channel_local_to_world(
                    preview.channel_index,
                    egui::pos2(x0, local_y),
                );
                let p1 = self.selected_channel_local_to_world(
                    preview.channel_index,
                    egui::pos2(x1, local_y),
                );
                ui.painter().line_segment(
                    [
                        self.camera.world_to_screen(p0, rect),
                        self.camera.world_to_screen(p1, rect),
                    ],
                    grid_stroke,
                );
            }
        }
    }

    fn ui_layers(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        self.rebuild_layer_orders();

        ui.heading("Tools");
        // Tool availability depends on the current active layer. When a layer change invalidates
        // the current tool, fall back to pan and clear any partial selection gesture so we don't
        // leave stale drag state behind.
        let can_object_select = self.active_layer_supports_spatial_selection();
        if !can_object_select
            && matches!(self.tool_mode, ToolMode::RectSelect | ToolMode::LassoSelect)
        {
            self.clear_spatial_selection_drag();
            self.tool_mode = ToolMode::Pan;
        }
        ui.horizontal(|ui| {
            if icon_button(
                ui,
                Icon::Pan,
                self.tool_mode == ToolMode::Pan,
                egui::Sense::click(),
            )
            .on_hover_text("Pan")
            .clicked()
            {
                self.clear_spatial_selection_drag();
                self.tool_mode = ToolMode::Pan;
            }
            if icon_button(
                ui,
                Icon::Move,
                self.tool_mode == ToolMode::MoveLayer,
                egui::Sense::click(),
            )
            .on_hover_text("Move active layer")
            .clicked()
            {
                self.clear_spatial_selection_drag();
                self.tool_mode = ToolMode::MoveLayer;
            }
            let can_transform = matches!(self.active_layer, LayerId::Channel(_));
            let mut transform_clicked = false;
            ui.add_enabled_ui(can_transform, |ui| {
                if icon_button(
                    ui,
                    Icon::Transform,
                    self.tool_mode == ToolMode::TransformLayer,
                    egui::Sense::click(),
                )
                .on_hover_text("Transform active channel (scale/rotate)")
                .clicked()
                {
                    transform_clicked = true;
                }
            });
            if transform_clicked {
                self.clear_spatial_selection_drag();
                self.tool_mode = ToolMode::TransformLayer;
            }
            if icon_button(
                ui,
                Icon::Polygon,
                self.tool_mode == ToolMode::DrawMaskPolygon,
                egui::Sense::click(),
            )
            .on_hover_text("Draw mask polygon")
            .clicked()
            {
                self.clear_spatial_selection_drag();
                self.tool_mode = ToolMode::DrawMaskPolygon;
                let id = self.ensure_editable_mask_layer();
                self.active_layer = LayerId::Mask(id);
                self.drawing_mask_layer = Some(id);
            }
            ui.add_enabled_ui(can_object_select, |ui| {
                if icon_button(
                    ui,
                    Icon::RectSelect,
                    self.tool_mode == ToolMode::RectSelect,
                    egui::Sense::click(),
                )
                .on_hover_text("Drag a rectangle to select cells by centroid")
                .clicked()
                {
                    self.clear_spatial_selection_drag();
                    self.tool_mode = ToolMode::RectSelect;
                }
                if icon_button(
                    ui,
                    Icon::LassoSelect,
                    self.tool_mode == ToolMode::LassoSelect,
                    egui::Sense::click(),
                )
                .on_hover_text("Draw a freehand lasso to select cells by centroid")
                .clicked()
                {
                    self.clear_spatial_selection_drag();
                    self.tool_mode = ToolMode::LassoSelect;
                }
            });
        });

        ui.separator();
        ui.heading("Layers");

        egui::ScrollArea::vertical()
            .id_salt("layers-scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let mut groups_cfg = self.project_space.layer_groups().clone();
                let mut groups_changed = false;

                // Overlays master visibility toggle.
                let overlay_ids = self.overlay_layer_order.clone();
                if overlay_ids.is_empty() {
                    ui.horizontal(|ui| {
                        ui.label("Overlays");
                        ui.add_space(4.0);
                        ui.label("(none)");
                    });
                } else {
                    let mut overlays_all = true;
                    let mut overlays_none = true;
                    for id in overlay_ids.iter().copied() {
                        if !self.layer_is_available(id) {
                            continue;
                        }
                        if let Some(v) = self.layer_visible_mut(id).map(|v| *v) {
                            overlays_all &= v;
                            overlays_none &= !v;
                        }
                    }
                    let overlays_mixed = !overlays_all && !overlays_none;
                    ui.horizontal(|ui| {
                        ui.label("Overlays");
                        ui.add_space(4.0);
                        let mut all = overlays_all;
                        if ui
                            .add(egui::Checkbox::new(&mut all, "All").indeterminate(overlays_mixed))
                            .changed()
                        {
                            for id in overlay_ids.into_iter() {
                                if !self.layer_is_available(id) {
                                    continue;
                                }
                                if let Some(v) = self.layer_visible_mut(id) {
                                    *v = all;
                                }
                            }
                            self.bump_render_id();
                        }
                    });

                    // Annotation groups: show a collapsible header at the first member, and hide
                    // members when collapsed.
                    let mut ann_members_by_group: HashMap<u64, Vec<u64>> = HashMap::new();
                    for id in self.overlay_layer_order.iter().copied() {
                        let LayerId::Annotation(aid) = id else { continue };
                        let Some(m) = groups_cfg.annotation_members.get(&aid) else { continue };
                        if groups_cfg.annotation_groups.iter().any(|g| g.id == m.group_id) {
                            ann_members_by_group.entry(m.group_id).or_default().push(aid);
                        }
                    }
                    let mut ann_headers_shown: HashSet<u64> = HashSet::new();
                    let mut delete_ann_group: Option<u64> = None;

                    for i in 0..self.overlay_layer_order.len() {
                        let id = self.overlay_layer_order[i];

                        if let LayerId::Annotation(aid) = id {
                            if let Some(m) = groups_cfg.annotation_members.get(&aid) {
                                let gid = m.group_id;
                                if groups_cfg.annotation_groups.iter().any(|g| g.id == gid) {
                                    if !ann_headers_shown.contains(&gid) {
                                        ann_headers_shown.insert(gid);
                                        let Some(group_idx) = groups_cfg
                                            .annotation_groups
                                            .iter()
                                            .position(|g| g.id == gid)
                                        else {
                                            // Shouldn't happen due to check above.
                                            continue;
                                        };
                                        let members = ann_members_by_group
                                            .get(&gid)
                                            .map(|v| v.as_slice())
                                            .unwrap_or(&[]);

                                        let group = &mut groups_cfg.annotation_groups[group_idx];
                                        let mut all = true;
                                        let mut none = true;
                                        for &mid in members {
                                            let lid = LayerId::Annotation(mid);
                                            if let Some(v) = self.layer_visible_mut(lid).map(|v| *v) {
                                                all &= v;
                                                none &= !v;
                                            }
                                        }
                                        let mixed = !members.is_empty() && !all && !none;

                                        let header = egui::collapsing_header::CollapsingState::load_with_default_open(
                                            ui.ctx(),
                                            ui.make_persistent_id(("annotation-group", group.id)),
                                            group.expanded,
                                        )
                                        .show_header(ui, |ui| {
                                            let mut set_all = all;
                                            ui.add_enabled_ui(!members.is_empty(), |ui| {
                                                if ui
                                                    .add(
                                                        egui::Checkbox::new(&mut set_all, "")
                                                            .indeterminate(mixed),
                                                    )
                                                    .on_hover_text("Toggle all annotation layers in group")
                                                    .changed()
                                                {
                                                    for &mid in members {
                                                        if let Some(v) = self.layer_visible_mut(LayerId::Annotation(mid)) {
                                                            *v = set_all;
                                                        }
                                                    }
                                                    group.visible = set_all;
                                                    groups_changed = true;
                                                    self.bump_render_id();
                                                }
                                            });
                                            ui.add_space(4.0);
                                            ui.label(group.name.clone());
                                        });
                                        let open = header.is_open();
                                        let (_toggle, _hdr, _body) = header.body(|ui| {
                                            ui.horizontal(|ui| {
                                                ui.label("Name");
                                                groups_changed |= ui.text_edit_singleline(&mut group.name).changed();
                                            });
                                            ui.horizontal(|ui| {
                                                ui.label("Visible");
                                                if ui.checkbox(&mut group.visible, "").changed() {
                                                    groups_changed = true;
                                                }
                                                if ui.button("Delete group").clicked() {
                                                    delete_ann_group = Some(group.id);
                                                }
                                            });
                                            ui.horizontal(|ui| {
                                                let mut has_tint = group.tint_rgb.is_some();
                                                if ui.checkbox(&mut has_tint, "Tint").changed() {
                                                    if has_tint && group.tint_rgb.is_none() {
                                                        group.tint_rgb = Some([255, 255, 255]);
                                                    }
                                                    if !has_tint {
                                                        group.tint_rgb = None;
                                                    }
                                                    groups_changed = true;
                                                }
                                                if let Some(rgb) = group.tint_rgb.as_mut() {
                                                    let mut c = egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
                                                    if ui.color_edit_button_srgba(&mut c).changed() {
                                                        *rgb = [c.r(), c.g(), c.b()];
                                                        groups_changed = true;
                                                    }
                                                }
                                            });
                                            groups_changed |= ui
                                                .add(
                                                    egui::Slider::new(&mut group.tint_strength, 0.0..=1.0)
                                                        .text("Tint strength")
                                                        .clamping(egui::SliderClamping::Always),
                                                )
                                                .changed();
                                        });
                                        if open != group.expanded {
                                            group.expanded = open;
                                            groups_changed = true;
                                        }
                                    }

                                    // Hide grouped members when the group is collapsed.
                                    let expanded = groups_cfg
                                        .annotation_groups
                                        .iter()
                                        .find(|g| g.id == gid)
                                        .map(|g| g.expanded)
                                        .unwrap_or(true);
                                    if !expanded {
                                        continue;
                                    }
                                }
                            }
                        }

                        let available = self.layer_is_available(id);
                        let selected = self.active_layer == id || self.selected_overlay_layers.contains(&id);
                        let icon = self.layer_icon(id);
                        let name = self.layer_display_name(id);
                        let visible = self.layer_visible_mut(id).map(|v| *v);
                        let resp = layer_list::ui_layer_row(
                            ui,
                            ctx,
                            &mut self.layer_drag,
                            LayerGroup::Overlays,
                            i,
                            id,
                            &name,
                            layer_list::LayerRowOptions {
                                available,
                                selected,
                                icon,
                                visible,
                                color_rgb: None,
                            },
                        );
                        let mods = ctx.input(|i| i.modifiers);
                        if resp.selected_clicked {
                            // Primary click selection.
                            if mods.shift && self.overlay_select_anchor_pos.is_some() {
                                let anchor = self.overlay_select_anchor_pos.unwrap_or(i);
                                let (a, b) = if anchor <= i { (anchor, i) } else { (i, anchor) };
                                self.selected_overlay_layers.clear();
                                for pos in a..=b {
                                    if let Some(l) = self.overlay_layer_order.get(pos).copied() {
                                        self.selected_overlay_layers.insert(l);
                                    }
                                }
                            } else if mods.command {
                                if !self.selected_overlay_layers.insert(id) {
                                    self.selected_overlay_layers.remove(&id);
                                }
                                self.overlay_select_anchor_pos = Some(i);
                            } else {
                                self.selected_overlay_layers.clear();
                                self.selected_overlay_layers.insert(id);
                                self.overlay_select_anchor_pos = Some(i);
                            }
                            self.set_active_layer(id);
                        } else if resp.row_response.secondary_clicked() {
                            // Right-click selects the row (single) if it wasn't already selected.
                            if !self.selected_overlay_layers.contains(&id) {
                                self.selected_overlay_layers.clear();
                                self.selected_overlay_layers.insert(id);
                                self.overlay_select_anchor_pos = Some(i);
                                self.set_active_layer(id);
                            }
                        }
                        if let Some(v) = resp.visible_changed {
                            if let Some(dst) = self.layer_visible_mut(id) {
                                *dst = v;
                            }
                        }
                        if resp.changed {
                            self.bump_render_id();
                        }

                        // Context menu: group layers.
                        resp.row_response.context_menu(|ui| {
                            let selected_annotations: Vec<u64> = self
                                .selected_overlay_layers
                                .iter()
                                .filter_map(|l| match l {
                                    LayerId::Annotation(a) => Some(*a),
                                    _ => None,
                                })
                                .collect();
                            let can_group = selected_annotations.len() >= 2
                                && selected_annotations.len() == self.selected_overlay_layers.len();
                            if ui
                                .add_enabled(can_group, egui::Button::new("Group layers..."))
                                .clicked()
                            {
                                self.open_group_layers_dialog_annotations(selected_annotations);
                                ui.close_menu();
                            }
                        });
                    }

                    if let Some(group_id) = delete_ann_group {
                        groups_cfg.annotation_groups.retain(|g| g.id != group_id);
                        groups_cfg
                            .annotation_members
                            .retain(|_k, m| m.group_id != group_id);
                        groups_changed = true;
                    }
                }
                ui.separator();
                channels_panel::show(self, ui, ctx);
            });

        layer_list::paint_drag_preview(ctx, self.layer_drag.as_ref(), |id| {
            self.layer_display_name(id)
        });

        let mut dropped: Option<(LayerGroup, usize, usize)> = None;
        layer_list::finish_drag_if_released(ctx, &mut self.layer_drag, |group, from, to| {
            dropped = Some((group, from, to));
        });
        if let Some((group, from, to)) = dropped {
            match group {
                LayerGroup::Overlays => {
                    layer_list::reorder_vec(&mut self.overlay_layer_order, from, to)
                }
                LayerGroup::Channels => {
                    layer_list::reorder_vec(&mut self.channel_layer_order, from, to)
                }
            }
            self.bump_render_id();
        }
    }

    fn open_group_layers_dialog_channels(&mut self, members: Vec<usize>) {
        let existing = self
            .project_space
            .layer_groups()
            .channel_groups
            .iter()
            .map(|g| g.name.clone())
            .collect::<Vec<_>>();
        let default_name = default_group_name(existing);
        self.group_layers_dialog = Some(GroupLayersDialog::new(
            GroupLayersTarget::Channels(members),
            default_name,
        ));
    }

    fn open_group_layers_dialog_annotations(&mut self, members: Vec<u64>) {
        let existing = self
            .project_space
            .layer_groups()
            .annotation_groups
            .iter()
            .map(|g| g.name.clone())
            .collect::<Vec<_>>();
        let default_name = default_group_name(existing);
        self.group_layers_dialog = Some(GroupLayersDialog::new(
            GroupLayersTarget::Annotations(members),
            default_name,
        ));
    }

    fn ui_group_layers_dialog(&mut self, ctx: &egui::Context) {
        let Some(dialog) = self.group_layers_dialog.as_mut() else {
            return;
        };

        let mut open = true;
        let mut accept = false;
        let mut cancel = false;

        egui::Window::new("Group layers")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .open(&mut open)
            .show(ctx, |ui| {
                ui.label("Group name");
                let mut name_output = egui::TextEdit::singleline(&mut dialog.name).show(ui);
                if dialog.focus_name_on_open {
                    name_output.response.request_focus();
                    name_output
                        .state
                        .cursor
                        .set_char_range(Some(egui::text::CCursorRange::select_all(
                            &name_output.galley,
                        )));
                    name_output.state.store(ui.ctx(), name_output.response.id);
                    dialog.focus_name_on_open = false;
                }

                if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    accept = true;
                }
                if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    cancel = true;
                }

                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                    if ui.button("OK").clicked() {
                        accept = true;
                    }
                });
            });

        if !open || cancel {
            self.group_layers_dialog = None;
            return;
        }
        if accept {
            let name = dialog.resolved_name();
            let target = dialog.target.clone();
            self.group_layers_dialog = None;
            self.apply_new_group(name, target);
        }
    }

    fn apply_new_group(&mut self, name: String, target: GroupLayersTarget) {
        match target {
            GroupLayersTarget::Channels(indices) => {
                let first_color = indices
                    .first()
                    .and_then(|idx| self.channels.get(*idx))
                    .map(|ch| {
                        layer_groups::effective_channel_color_rgb(
                            self.project_space.layer_groups(),
                            &ch.name,
                            ch.color_rgb,
                        )
                    })
                    .unwrap_or([255, 255, 255]);

                self.project_space.update_layer_groups(|groups| {
                    let existing_ids = groups
                        .channel_groups
                        .iter()
                        .map(|g| g.id)
                        .collect::<Vec<_>>();
                    let gid = layer_groups::next_group_id(&existing_ids);
                    groups
                        .channel_groups
                        .push(crate::data::project_config::ProjectChannelGroup {
                            id: gid,
                            name,
                            expanded: true,
                            color_rgb: first_color,
                        });
                    for idx in indices {
                        if let Some(ch) = self.channels.get(idx) {
                            groups.channel_members.insert(
                                ch.name.clone(),
                                crate::data::project_config::ProjectChannelGroupMember {
                                    group_id: gid,
                                    inherit_color: true,
                                },
                            );
                        }
                    }
                });
                self.bump_render_id();
            }
            GroupLayersTarget::Annotations(layer_ids) => {
                self.project_space.update_layer_groups(|groups| {
                    let existing_ids = groups
                        .annotation_groups
                        .iter()
                        .map(|g| g.id)
                        .collect::<Vec<_>>();
                    let gid = layer_groups::next_group_id(&existing_ids);
                    groups
                        .annotation_groups
                        .push(crate::data::project_config::ProjectAnnotationGroup {
                            id: gid,
                            name,
                            expanded: true,
                            visible: true,
                            tint_rgb: None,
                            tint_strength: 0.35,
                        });
                    for id in layer_ids {
                        groups.annotation_members.insert(
                            id,
                            crate::data::project_config::ProjectAnnotationGroupMember {
                                group_id: gid,
                                inherit_tint: true,
                            },
                        );
                    }
                });
                self.bump_render_id();
            }
        }
    }

    fn selected_memory_channel_indices(&self) -> Vec<usize> {
        self.channel_layer_order
            .iter()
            .copied()
            .filter(|idx| self.memory_selected_channels.contains(idx))
            .collect()
    }

    fn selected_memory_channel_ids(&self) -> Vec<u64> {
        self.selected_memory_channel_indices()
            .into_iter()
            .filter_map(|idx| self.channels.get(idx).map(|channel| channel.index as u64))
            .collect()
    }

    fn memory_channel_rows(&self) -> Vec<MemoryChannelRow> {
        self.channel_layer_order
            .iter()
            .filter_map(|&idx| {
                self.channels.get(idx).map(|channel| MemoryChannelRow {
                    id: idx,
                    label: if channel.visible {
                        format!("{} (visible)", channel.name)
                    } else {
                        channel.name.clone()
                    },
                    visible: channel.visible,
                })
            })
            .collect()
    }

    fn estimate_level_ram_bytes_for_selected_channels(
        &self,
        level: usize,
        selected: &[usize],
    ) -> u64 {
        let Some(info) = self.dataset.levels.get(level) else {
            return 0;
        };
        if selected.is_empty() {
            return 0;
        }
        let Some(&shape_y) = info.shape.get(self.dataset.dims.y) else {
            return 0;
        };
        let Some(&shape_x) = info.shape.get(self.dataset.dims.x) else {
            return 0;
        };
        let channel_count = if self.dataset.dims.c.is_some() {
            selected.len() as u64
        } else {
            1
        };
        let bytes_per_sample = match info.dtype.as_str() {
            "|u1" | "|i1" => 1u64,
            "<u2" | ">u2" | "<i2" | ">i2" => 2u64,
            "<f4" | ">f4" | "<u4" | ">u4" | "<i4" | ">i4" => 4u64,
            _ => 2u64,
        };
        channel_count
            .checked_mul(shape_y)
            .and_then(|v| v.checked_mul(shape_x))
            .and_then(|v| v.checked_mul(bytes_per_sample))
            .unwrap_or(0)
    }

    fn memory_risk(&self, requested_bytes: u64) -> Option<crate::app_support::memory::MemoryRisk> {
        memory_risk(
            self.system_memory.as_ref(),
            self.pinned_levels.total_loaded_bytes(),
            requested_bytes,
        )
    }

    fn start_memory_load(
        &mut self,
        summary: String,
        requests: Vec<PendingPinnedLevelLoadRequest>,
        requested_bytes: u64,
    ) {
        if requests.is_empty() {
            self.memory_status = "No eligible channels selected for RAM pinning.".to_string();
            return;
        }
        if let Some(risk) = self.memory_risk(requested_bytes) {
            self.pending_memory_load = Some(PendingMemoryAction {
                summary,
                payload: requests,
                risk,
            });
        } else {
            self.execute_memory_load(summary, requests);
        }
    }

    fn execute_memory_load(
        &mut self,
        summary: String,
        requests: Vec<PendingPinnedLevelLoadRequest>,
    ) {
        if requests.is_empty() {
            self.memory_status = "No eligible channels selected for RAM pinning.".to_string();
            return;
        }
        for request in requests {
            self.pinned_levels.request_load(
                self.store.clone(),
                self.dataset.dims.clone(),
                self.dataset.levels.clone(),
                request.level,
                request.selected_channels,
            );
        }
        self.memory_status = summary;
    }

    fn ui_memory_load_dialog(&mut self, ctx: &egui::Context) {
        if let Some((summary, requests)) =
            ui_pending_memory_action_dialog(ctx, &mut self.pending_memory_load)
        {
            self.execute_memory_load(summary, requests);
        }
    }

    fn ui_memory(&mut self, ui: &mut egui::Ui) {
        ui_memory_overview(
            ui,
            "Manually pin selected OME-Zarr channels and levels in CPU RAM for the current image. Pinned levels feed the existing tile renderer instead of replacing it.",
            Some(("Pinned total", self.pinned_levels.total_loaded_bytes())),
            self.system_memory.as_ref(),
        );
        ui.add_space(6.0);

        ui.collapsing("Tile Loading", |ui| {
            if self.supports_runtime_tile_loader_tuning() {
                let mut threads = self.tile_loader_threads as u32;
                ui.horizontal(|ui| {
                    ui.label("Workers");
                    let changed = ui
                        .add(
                            egui::DragValue::new(&mut threads)
                                .range(1..=12)
                                .speed(0.2),
                        )
                        .changed();
                    if ui.button("Auto").clicked() {
                        threads = Self::default_tile_loader_threads() as u32;
                    }
                    if changed || threads as usize != self.tile_loader_threads {
                        let next = threads.max(1) as usize;
                        if next != self.tile_loader_threads {
                            self.tile_loader_threads = next;
                            match self.respawn_tile_loaders() {
                                Ok(()) => {
                                    self.tile_loading_status = format!(
                                        "Respawned tile loaders with {} worker(s).",
                                        self.tile_loader_threads
                                    );
                                }
                                Err(err) => {
                                    self.tile_loading_status =
                                        format!("Tile loader reconfigure failed: {err}");
                                }
                            }
                        }
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Prefetch");
                    egui::ComboBox::from_id_salt("tile-prefetch-mode")
                        .selected_text(match self.tile_prefetch_mode {
                            TilePrefetchMode::Off => "Off",
                            TilePrefetchMode::TargetHalo => "Target halo",
                            TilePrefetchMode::TargetAndFinerHalo => "Target + finer halo",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.tile_prefetch_mode,
                                TilePrefetchMode::Off,
                                "Off",
                            );
                            ui.selectable_value(
                                &mut self.tile_prefetch_mode,
                                TilePrefetchMode::TargetHalo,
                                "Target halo",
                            );
                            ui.selectable_value(
                                &mut self.tile_prefetch_mode,
                                TilePrefetchMode::TargetAndFinerHalo,
                                "Target + finer halo",
                            );
                        });
                });
                ui.horizontal(|ui| {
                    ui.label("Aggressiveness");
                    egui::ComboBox::from_id_salt("tile-prefetch-aggressiveness")
                        .selected_text(match self.tile_prefetch_aggressiveness {
                            TilePrefetchAggressiveness::Conservative => "Conservative",
                            TilePrefetchAggressiveness::Balanced => "Balanced",
                            TilePrefetchAggressiveness::Aggressive => "Aggressive",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.tile_prefetch_aggressiveness,
                                TilePrefetchAggressiveness::Conservative,
                                "Conservative",
                            );
                            ui.selectable_value(
                                &mut self.tile_prefetch_aggressiveness,
                                TilePrefetchAggressiveness::Balanced,
                                "Balanced",
                            );
                            ui.selectable_value(
                                &mut self.tile_prefetch_aggressiveness,
                                TilePrefetchAggressiveness::Aggressive,
                                "Aggressive",
                            );
                        });
                });
                ui.checkbox(
                    &mut self.prefer_pinned_finer_levels,
                    "Use pinned finer levels for missing coarser levels",
                )
                .on_hover_text(
                    "If the current zoom level is not pinned, render it from a finer pinned level before falling back to disk or network reads.",
                );
                ui.label(
                    "Target halo prefetches nearby tiles at the current level. Target + finer halo also warms the next finer level to reduce zoom-in stalls.",
                );
            } else {
                ui.label("Runtime tile loading controls are unavailable for this dataset backend.");
            }
            if !self.tile_loading_status.is_empty() {
                ui.label(self.tile_loading_status.clone());
            }
            ui.separator();
        });

        let rows = self.memory_channel_rows();
        ui_memory_channel_selector(
            ui,
            "viewer-memory-channel-list",
            &rows,
            &mut self.memory_selected_channels,
        );
        ui.separator();

        ui.label(format!(
            "Texture tile cache: {} / {} tiles, {} in flight",
            self.cache.len(),
            self.cache.capacity(),
            self.cache.in_flight_len()
        ));
        if let Some(level) = self.last_target_level {
            ui.label(format!("Current draw level: {level}"));
        }
        ui.label("Loading is manual. The app estimates RAM usage but does not enforce a system-memory limit.");
        if !self.memory_status.is_empty() {
            ui.label(self.memory_status.clone());
        }
        ui.separator();

        let selected_channels = self.selected_memory_channel_indices();
        let selected_channel_ids = self.selected_memory_channel_ids();
        egui::Grid::new("viewer-memory-grid")
            .num_columns(5)
            .striped(true)
            .show(ui, |ui| {
                ui.strong("Level");
                ui.strong("Shape");
                ui.strong("RAM");
                ui.strong("Status");
                ui.strong("Action");
                ui.end_row();

                for level_idx in 0..self.dataset.levels.len() {
                    let (shape_y, shape_x) = self
                        .dataset
                        .levels
                        .get(level_idx)
                        .map(|level| {
                            (
                                level.shape.get(self.dataset.dims.y).copied().unwrap_or(0),
                                level.shape.get(self.dataset.dims.x).copied().unwrap_or(0),
                            )
                        })
                        .unwrap_or((0, 0));
                    let selected_count = if self.dataset.dims.c.is_some() {
                        selected_channels.len()
                    } else if selected_channels.is_empty() {
                        0
                    } else {
                        1
                    };
                    let estimate = self.estimate_level_ram_bytes_for_selected_channels(
                        level_idx,
                        &selected_channels,
                    );

                    ui.label(level_idx.to_string());
                    ui.label(format!("{selected_count} x {shape_y} x {shape_x}"));
                    ui.label(if estimate == 0 {
                        "No selected channels".to_string()
                    } else {
                        format_bytes(estimate)
                    });
                    match self.pinned_levels.status(level_idx) {
                        PinnedLevelStatus::Unloaded => {
                            if self.last_target_level == Some(level_idx) {
                                ui.label("Streaming (current)");
                            } else if !self.auto_level && self.manual_level == level_idx {
                                ui.label("Streaming (manual)");
                            } else {
                                ui.label("Streaming");
                            }
                        }
                        PinnedLevelStatus::Loading => {
                            ui.label("Loading");
                        }
                        PinnedLevelStatus::Loaded {
                            bytes,
                            channels_loaded,
                        } => {
                            ui.label(format!(
                                "Pinned ({}; {} ch)",
                                format_bytes(bytes),
                                channels_loaded
                            ));
                        }
                        PinnedLevelStatus::Failed(err) => {
                            ui.colored_label(ui.visuals().warn_fg_color, format!("Failed: {err}"));
                        }
                    }
                    ui.horizontal(|ui| {
                        let risk = self.memory_risk(estimate);
                        let load_label = match risk.as_ref().map(|risk| risk.level) {
                            Some(crate::app_support::memory::MemoryRiskLevel::Danger) => "Load danger",
                            Some(crate::app_support::memory::MemoryRiskLevel::Warning) => "Load warning",
                            None => "Load",
                        };
                        if ui
                            .add_enabled(estimate > 0, egui::Button::new(load_label))
                            .clicked()
                        {
                            self.start_memory_load(
                                format!(
                                    "Loading {} channel(s) from level {level_idx} into RAM",
                                    selected_channel_ids.len()
                                ),
                                vec![PendingPinnedLevelLoadRequest {
                                    level: level_idx,
                                    selected_channels: selected_channel_ids.clone(),
                                }],
                                estimate,
                            );
                        }
                        let can_unload = !matches!(
                            self.pinned_levels.status(level_idx),
                            PinnedLevelStatus::Unloaded
                        );
                        if ui
                            .add_enabled(can_unload, egui::Button::new("Unload"))
                            .clicked()
                        {
                            self.pinned_levels.unload(level_idx);
                            self.memory_status =
                                format!("Unloaded pinned level {level_idx} from RAM.");
                        }
                    });
                    ui.end_row();
                }
            });
    }

    fn ui_layer_properties(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading(self.layer_display_name(self.active_layer));
        ui.separator();

        let mut changed = false;
        ui.label("Transform");

        let active_layer = self.active_layer;
        let mut off = self.layer_offset_world(active_layer);

        let mut reset_clicked = false;
        ui.horizontal(|ui| {
            changed |= ui
                .add(egui::DragValue::new(&mut off.x).speed(5.0).prefix("x "))
                .changed();
            changed |= ui
                .add(egui::DragValue::new(&mut off.y).speed(5.0).prefix("y "))
                .changed();
            reset_clicked = ui
                .button("Reset")
                .on_hover_text("Reset translation (and scale/rotation for channels)")
                .clicked();
        });

        if let Some(dst) = self.layer_offset_world_mut(active_layer) {
            if reset_clicked {
                *dst = egui::Vec2::ZERO;
                changed = true;
            } else if changed {
                *dst = off;
            }
        }

        if let LayerId::Channel(idx0) = active_layer {
            let idx = idx0.min(self.channels.len().saturating_sub(1));

            let mut scale = self
                .channel_scales
                .get(idx)
                .copied()
                .unwrap_or(egui::Vec2::splat(1.0));
            let mut rot = self.channel_rotations_rad.get(idx).copied().unwrap_or(0.0);
            let mut deg = rot.to_degrees();

            ui.horizontal(|ui| {
                ui.label("Scale");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut scale.x)
                            .speed(0.02)
                            .clamp_range(0.01..=100.0)
                            .prefix("x "),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut scale.y)
                            .speed(0.02)
                            .clamp_range(0.01..=100.0)
                            .prefix("y "),
                    )
                    .changed();
                if ui.button("1x").clicked() {
                    scale = egui::Vec2::splat(1.0);
                    changed = true;
                }
            });

            ui.horizontal(|ui| {
                ui.label("Rotate");
                if ui
                    .add(
                        egui::DragValue::new(&mut deg)
                            .speed(1.0)
                            .clamp_range(-360.0..=360.0)
                            .suffix(" deg"),
                    )
                    .changed()
                {
                    changed = true;
                }
                if ui.button("0").clicked() {
                    deg = 0.0;
                    changed = true;
                }
            });

            if reset_clicked {
                scale = egui::Vec2::splat(1.0);
                deg = 0.0;
                changed = true;
            }

            rot = deg.to_radians();
            if let Some(dst) = self.channel_scales.get_mut(idx) {
                *dst = scale;
            }
            if let Some(dst) = self.channel_rotations_rad.get_mut(idx) {
                *dst = rot;
            }
        }

        if changed {
            self.hist_dirty = true;
        }
        ui.separator();

        if let Some(group_id) = self.selected_channel_group_id {
            self.ui_group_contrast(ctx, ui, group_id);
            return;
        }

        match self.active_layer {
            LayerId::Channel(idx) => {
                self.selected_channel = idx.min(self.channels.len().saturating_sub(1));
                self.ui_contrast(ctx, ui);
                self.maybe_request_histogram(ctx);
            }
            LayerId::SpatialImage(id) => {
                if let Some(layer) = self
                    .spatial_image_layers
                    .images
                    .iter_mut()
                    .find(|l| l.id == id)
                {
                    if layer.ui_properties(ui) {
                        self.bump_render_id();
                    }
                } else {
                    ui.label("Not loaded.");
                }
            }
            LayerId::Points => {
                ui.checkbox(&mut self.cell_points.visible, "Visible");
                ui.add(
                    egui::Slider::new(&mut self.cell_points.style.radius_screen_px, 0.5..=20.0)
                        .text("Size")
                        .show_value(true)
                        .clamping(egui::SliderClamping::Always),
                );

                ui.separator();
                ui.label("Positive points");
                ui.horizontal(|ui| {
                    ui.label("Fill");
                    ui.color_edit_button_srgba(&mut self.cell_points.style.fill_positive);
                });
                ui.horizontal(|ui| {
                    ui.label("Stroke");
                    ui.add(
                        egui::DragValue::new(&mut self.cell_points.style.stroke_positive.width)
                            .speed(0.25)
                            .clamp_range(0.0..=10.0),
                    );
                    ui.color_edit_button_srgba(&mut self.cell_points.style.stroke_positive.color);
                });

                ui.separator();
                ui.label("Negative points");
                ui.horizontal(|ui| {
                    ui.label("Fill");
                    ui.color_edit_button_srgba(&mut self.cell_points.style.fill_negative);
                });
                ui.horizontal(|ui| {
                    ui.label("Stroke");
                    ui.add(
                        egui::DragValue::new(&mut self.cell_points.style.stroke_negative.width)
                            .speed(0.25)
                            .clamp_range(0.0..=10.0),
                    );
                    ui.color_edit_button_srgba(&mut self.cell_points.style.stroke_negative.color);
                });
            }
            LayerId::Annotation(id) => {
                let Some(idx) = self.annotation_layers.iter().position(|l| l.id == id) else {
                    ui.label("Annotation layer not found.");
                    return;
                };
                let mut groups_cfg = self.project_space.layer_groups().clone();
                let mut groups_changed = false;
                let mut delete_clicked = false;
                ui.horizontal(|ui| {
                    ui.label("Name");
                    ui.text_edit_singleline(&mut self.annotation_layers[idx].name);
                });
                ui.separator();

                // Grouping (optional): visibility/tint can be controlled at group level.
                let mut selected_group: Option<u64> = groups_cfg
                    .annotation_members
                    .get(&id)
                    .map(|m| m.group_id)
                    .filter(|gid| groups_cfg.annotation_groups.iter().any(|g| g.id == *gid));
                ui.horizontal(|ui| {
                    ui.label("Group");
                    egui::ComboBox::from_id_salt(("annotation-group-select", id))
                        .selected_text(
                            selected_group
                                .and_then(|gid| {
                                    groups_cfg.annotation_groups.iter().find(|g| g.id == gid)
                                })
                                .map(|g| g.name.as_str())
                                .unwrap_or("(none)"),
                        )
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut selected_group, None, "(none)");
                            for g in &groups_cfg.annotation_groups {
                                ui.selectable_value(
                                    &mut selected_group,
                                    Some(g.id),
                                    g.name.clone(),
                                );
                            }
                        });
                    if ui
                        .button("+ Group")
                        .on_hover_text("Create a new annotation group")
                        .clicked()
                    {
                        let existing = groups_cfg
                            .annotation_groups
                            .iter()
                            .map(|g| g.id)
                            .collect::<Vec<_>>();
                        let id2 = layer_groups::next_group_id(&existing);
                        groups_cfg.annotation_groups.push(
                            crate::data::project_config::ProjectAnnotationGroup {
                                id: id2,
                                name: format!("Group {id2}"),
                                expanded: true,
                                visible: true,
                                tint_rgb: None,
                                tint_strength: 0.35,
                            },
                        );
                        selected_group = Some(id2);
                        groups_changed = true;
                    }
                });

                let have_member = groups_cfg.annotation_members.get(&id).is_some();
                if selected_group.is_none() && have_member {
                    groups_cfg.annotation_members.remove(&id);
                    groups_changed = true;
                } else if let Some(gid) = selected_group {
                    match groups_cfg.annotation_members.get_mut(&id) {
                        Some(m) => {
                            if m.group_id != gid {
                                m.group_id = gid;
                                groups_changed = true;
                            }
                        }
                        None => {
                            groups_cfg.annotation_members.insert(
                                id,
                                crate::data::project_config::ProjectAnnotationGroupMember {
                                    group_id: gid,
                                    inherit_tint: true,
                                },
                            );
                            groups_changed = true;
                        }
                    }
                }

                if let Some(gid) = selected_group {
                    let mut inherit_tint = groups_cfg
                        .annotation_members
                        .get(&id)
                        .map(|m| m.inherit_tint)
                        .unwrap_or(true);
                    ui.horizontal(|ui| {
                        if ui
                            .checkbox(&mut inherit_tint, "Inherit group tint")
                            .changed()
                        {
                            if let Some(m) = groups_cfg.annotation_members.get_mut(&id) {
                                m.inherit_tint = inherit_tint;
                                groups_changed = true;
                            }
                        }
                    });

                    if let Some(group) = groups_cfg
                        .annotation_groups
                        .iter_mut()
                        .find(|g| g.id == gid)
                    {
                        ui.separator();
                        ui.label("Group settings");

                        ui.horizontal(|ui| {
                            ui.label("Name");
                            groups_changed |= ui.text_edit_singleline(&mut group.name).changed();
                        });
                        ui.horizontal(|ui| {
                            ui.label("Visible");
                            if ui.checkbox(&mut group.visible, "").changed() {
                                groups_changed = true;
                            }
                        });
                        ui.horizontal(|ui| {
                            let mut has_tint = group.tint_rgb.is_some();
                            if ui.checkbox(&mut has_tint, "Tint").changed() {
                                if has_tint && group.tint_rgb.is_none() {
                                    group.tint_rgb = Some([255, 255, 255]);
                                }
                                if !has_tint {
                                    group.tint_rgb = None;
                                }
                                groups_changed = true;
                            }
                            if let Some(rgb) = group.tint_rgb.as_mut() {
                                let mut c = egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
                                if ui.color_edit_button_srgba(&mut c).changed() {
                                    *rgb = [c.r(), c.g(), c.b()];
                                    groups_changed = true;
                                }
                            }
                        });
                        groups_changed |= ui
                            .add(
                                egui::Slider::new(&mut group.tint_strength, 0.0..=1.0)
                                    .text("Tint strength")
                                    .clamping(egui::SliderClamping::Always),
                            )
                            .changed();
                    }
                }
                ui.separator();
                if self.annotation_layers[idx].ui_properties(ui) {
                    self.bump_render_id();
                }
                ui.separator();
                delete_clicked = ui.button("Delete layer").clicked();
                if delete_clicked {
                    self.annotation_layers.remove(idx);
                    if self.active_layer == LayerId::Annotation(id) {
                        self.active_layer = if !self.channels.is_empty() {
                            LayerId::Channel(self.selected_channel.min(self.channels.len() - 1))
                        } else {
                            LayerId::Points
                        };
                    }
                    self.rebuild_layer_orders();
                    self.bump_render_id();
                    return;
                }

                if groups_changed {
                    let new_groups = groups_cfg;
                    self.project_space.update_layer_groups(|g| *g = new_groups);
                    self.bump_render_id();
                }
            }
            LayerId::SegmentationLabels => {
                if !self.layer_is_available(LayerId::SegmentationLabels) {
                    ui.label("Segmentation labels require the GPU renderer.");
                    return;
                }

                ui.horizontal(|ui| {
                    ui.label("Label");

                    if self.dataset.is_root_label_mask() {
                        ui.label(self.seg_label_selected.clone());
                    } else if !self.seg_label_names.is_empty() {
                        let before = self.seg_label_selected.clone();
                        egui::ComboBox::from_id_salt("seg_label_select")
                            .selected_text(self.seg_label_selected.clone())
                            .show_ui(ui, |ui| {
                                for name in self.seg_label_names.clone() {
                                    ui.selectable_value(
                                        &mut self.seg_label_selected,
                                        name.clone(),
                                        name,
                                    );
                                }
                            });

                        if ui.button("Reload").clicked() || self.seg_label_selected != before {
                            let name = self.seg_label_selected.trim().to_string();
                            if name.is_empty() {
                                self.seg_label_status = "Label name is empty.".to_string();
                            } else {
                                match self.load_segmentation_labels(name.as_str()) {
                                    Ok(()) => {
                                        self.seg_label_status =
                                            format!("Loaded labels/{}.", name.as_str());
                                    }
                                    Err(err) => {
                                        self.seg_label_status =
                                            format!("Load labels/{} failed: {err}", name.as_str());
                                    }
                                }
                            }
                        }
                    } else {
                        ui.text_edit_singleline(&mut self.seg_label_input);
                        if ui.button("Load").clicked() {
                            let name = self.seg_label_input.trim().to_string();
                            if name.is_empty() {
                                self.seg_label_status = "Label name is empty.".to_string();
                            } else {
                                self.seg_label_selected = name.clone();
                                match self.load_segmentation_labels(name.as_str()) {
                                    Ok(()) => {
                                        self.seg_label_status =
                                            format!("Loaded labels/{}.", name.as_str());
                                    }
                                    Err(err) => {
                                        self.seg_label_status =
                                            format!("Load labels/{} failed: {err}", name.as_str());
                                    }
                                }
                            }
                        }
                    }

                    if !self.dataset.is_root_label_mask() && ui.button("Refresh").clicked() {
                        self.refresh_seg_label_names_for_current_roi();
                    }
                });

                if !self.seg_label_status.trim().is_empty() {
                    ui.label(self.seg_label_status.clone());
                }

                if self.label_cells.is_none() {
                    ui.label("Not loaded for this ROI.");
                    return;
                }

                ui.separator();
                ui.checkbox(&mut self.cells_outlines_visible, "Visible");
                ui.add(
                    egui::Slider::new(&mut self.cells_outlines_opacity, 0.0..=1.0)
                        .text("Opacity")
                        .show_value(true)
                        .clamping(egui::SliderClamping::Always),
                );
            }
            LayerId::SegmentationGeoJson => {
                let default_dir = self
                    .dataset
                    .source
                    .local_path()
                    .and_then(|p| p.parent())
                    .unwrap_or_else(|| Path::new("."));
                self.seg_geojson.ui_properties(ui, default_dir);
            }
            LayerId::SegmentationObjects => {
                let default_dir = self
                    .dataset
                    .source
                    .local_path()
                    .and_then(|p| p.parent())
                    .unwrap_or_else(|| Path::new("."));
                self.seg_objects.ui_properties(ui, default_dir);
            }
            LayerId::SpatialShape(id) => {
                if let Some(layer) = self.spatial_layers.shapes.iter_mut().find(|s| s.id == id) {
                    let default_dir = self
                        .dataset
                        .source
                        .local_path()
                        .and_then(|p| p.parent())
                        .unwrap_or_else(|| Path::new("."));
                    if layer.ui_properties(ui, default_dir) {
                        self.bump_render_id();
                    }
                } else {
                    ui.label("Not loaded.");
                }
            }
            LayerId::SpatialPoints => {
                if self.spatial_layers.points.is_some() {
                    let positive_targets = self.available_object_selection_targets();
                    let mut changed = false;
                    let mut bounds: Option<egui::Rect> = None;
                    let positive_cell_request;
                    {
                        let layer = self.spatial_layers.points.as_mut().expect("checked");
                        changed = layer.ui_properties(ui, &positive_targets);
                        bounds = layer.bounds_world();
                        positive_cell_request = layer.take_positive_cell_selection_request();
                    }
                    if let Some((cell_ids, target)) = positive_cell_request {
                        let status = if let Some((matched_layers, matched_objects)) =
                            self.select_objects_by_ids_target(&cell_ids, target)
                        {
                            changed = true;
                            format!(
                                "Selected {matched_objects} object(s) across {matched_layers} layer(s)."
                            )
                        } else {
                            "No loaded object layers matched those cell IDs.".to_string()
                        };
                        if let Some(layer) = self.spatial_layers.points.as_mut() {
                            layer.set_cell_selection_status(status);
                        }
                    }
                    if changed {
                        self.bump_render_id();
                    }
                    if let Some(bounds) = bounds {
                        ui.separator();
                        ui.label(format!(
                            "Bounds: x [{:.0}, {:.0}]  y [{:.0}, {:.0}]",
                            bounds.min.x, bounds.max.x, bounds.min.y, bounds.max.y
                        ));
                        if ui.button("Fit to points").clicked() {
                            if let Some(viewport) = self.last_canvas_rect {
                                let off = self.layer_offset_world(LayerId::SpatialPoints);
                                self.camera
                                    .fit_to_world_rect(viewport, bounds.translate(off));
                                self.bump_render_id();
                            }
                        }
                    }
                } else {
                    ui.label("Not loaded.");
                }
            }
            LayerId::XeniumCells => {
                if let Some(layer) = self.xenium_layers.cells.as_mut() {
                    layer.ui_properties(ui);
                } else {
                    ui.label("Not loaded.");
                }
            }
            LayerId::XeniumTranscripts => {
                if let Some(layer) = self.xenium_layers.transcripts.as_mut() {
                    layer.ui_properties(ui);
                } else {
                    ui.label("Not loaded.");
                }
            }
            LayerId::Mask(id) => {
                let Some(idx) = self.mask_layers.iter().position(|l| l.id == id) else {
                    ui.label("Mask layer not found.");
                    return;
                };

                let mut changed = false;
                let mut new_layer_clicked = false;
                let mut draw_tool_clicked = false;
                let mut clear_clicked = false;
                let mut delete_clicked = false;
                let mut reload_from_roi_clicked = false;
                let mut reload_from_file: Option<PathBuf> = None;

                {
                    let layer = &mut self.mask_layers[idx];

                    ui.horizontal(|ui| {
                        ui.label("Name");
                        changed |= ui.text_edit_singleline(&mut layer.name).changed();
                    });

                    changed |= ui.checkbox(&mut layer.visible, "Visible").changed();
                    changed |= ui.checkbox(&mut layer.editable, "Editable").changed();

                    changed |= ui
                        .add(
                            egui::Slider::new(&mut layer.opacity, 0.0..=1.0)
                                .text("Opacity")
                                .show_value(true)
                                .clamping(egui::SliderClamping::Always),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut layer.width_screen_px, 0.25..=6.0)
                                .text("Width")
                                .show_value(true)
                                .clamping(egui::SliderClamping::Always),
                        )
                        .changed();

                    ui.horizontal(|ui| {
                        ui.label("Color");
                        let mut c = egui::Color32::from_rgb(
                            layer.color_rgb[0],
                            layer.color_rgb[1],
                            layer.color_rgb[2],
                        );
                        if ui.color_edit_button_srgba(&mut c).changed() {
                            layer.color_rgb = [c.r(), c.g(), c.b()];
                            changed = true;
                        }
                    });

                    if let Some(src) = layer.source_geojson.as_ref() {
                        ui.separator();
                        ui.label("Source (GeoJSON)");
                        ui.label(src.to_string_lossy());
                        ui.horizontal(|ui| {
                            if layer.name == "Exclusion masks" && !layer.editable {
                                reload_from_roi_clicked |= ui.button("Reload").clicked();
                            } else if ui.button("Reload").clicked() {
                                reload_from_file = Some(src.clone());
                            }
                        });
                    }

                    ui.separator();
                    ui.horizontal(|ui| {
                        new_layer_clicked |= ui.button("New layer").clicked();
                        draw_tool_clicked |= ui.button("Draw (tool)").clicked();
                        clear_clicked |= ui
                            .add_enabled(
                                !layer.polygons_world.is_empty(),
                                egui::Button::new("Clear"),
                            )
                            .clicked();
                    });

                    ui.horizontal(|ui| {
                        delete_clicked |= ui.button("Delete layer").clicked();
                    });
                }

                if reload_from_roi_clicked {
                    match self.ensure_exclusion_masks_loaded() {
                        Ok(_) => changed = true,
                        Err(err) => self
                            .roi_selector
                            .set_status(format!("Reload masks failed: {err}")),
                    }
                } else if let Some(path) = reload_from_file {
                    match load_geojson_polylines_world(&path, 1.0, PolygonRingMode::AllRings) {
                        Ok(polys) => {
                            if let Some(layer) = self.mask_layers.get_mut(idx) {
                                layer.polygons_world = polys;
                                changed = true;
                            }
                        }
                        Err(err) => self
                            .roi_selector
                            .set_status(format!("Reload masks failed: {err}")),
                    }
                }

                if new_layer_clicked {
                    let new_id = self.create_editable_mask_layer(None);
                    self.set_active_layer(LayerId::Mask(new_id));
                    changed = true;
                }
                if draw_tool_clicked {
                    self.tool_mode = ToolMode::DrawMaskPolygon;
                    self.drawing_mask_layer = Some(id);
                }
                if clear_clicked {
                    if let Some(layer) = self.mask_layers.get_mut(idx) {
                        layer.clear();
                        changed = true;
                    }
                    if self.drawing_mask_layer == Some(id) {
                        self.drawing_mask_polygon.clear();
                    }
                }
                if delete_clicked {
                    self.mask_layers.remove(idx);
                    if self.drawing_mask_layer == Some(id) {
                        self.drawing_mask_layer = None;
                        self.drawing_mask_polygon.clear();
                    }
                    if self.active_layer == LayerId::Mask(id) {
                        self.active_layer = if !self.channels.is_empty() {
                            LayerId::Channel(self.selected_channel.min(self.channels.len() - 1))
                        } else {
                            LayerId::Points
                        };
                    }
                    self.rebuild_layer_orders();
                    self.bump_render_id();
                    return;
                }

                if changed {
                    self.bump_render_id();
                }
            }
        }
    }

    fn set_active_layer(&mut self, id: LayerId) {
        self.active_layer = id;
        if let LayerId::Channel(idx) = id {
            self.selected_channel = idx.min(self.channels.len().saturating_sub(1));
            self.hist_dirty = true;
        } else {
            self.selected_channel_group_id = None;
        }
    }

    fn channel_indices_in_group(&self, group_id: u64) -> Vec<usize> {
        let groups = self.project_space.layer_groups();
        self.channel_layer_order
            .iter()
            .copied()
            .filter(|&idx| {
                self.channels.get(idx).is_some_and(|ch| {
                    groups
                        .channel_members
                        .get(ch.name.as_str())
                        .is_some_and(|m| m.group_id == group_id)
                })
            })
            .collect()
    }

    fn group_contrast_window_for_indices(
        &self,
        indices: &[usize],
        abs_max: f32,
    ) -> Option<((f32, f32), bool)> {
        let mut first_window: Option<(f32, f32)> = None;
        let mut mixed = false;
        for &idx in indices {
            let Some(ch) = self.channels.get(idx) else {
                continue;
            };
            let window = ch.window.unwrap_or((0.0, abs_max));
            if let Some(prev) = first_window {
                if (prev.0 - window.0).abs() > 1e-6 || (prev.1 - window.1).abs() > 1e-6 {
                    mixed = true;
                }
            } else {
                first_window = Some(window);
            }
        }
        first_window.map(|window| (window, mixed))
    }

    fn ui_group_contrast(&mut self, _ctx: &egui::Context, ui: &mut egui::Ui, group_id: u64) {
        let abs_max = self.dataset.abs_max.max(1.0);
        let Some(group) = self
            .project_space
            .layer_groups()
            .channel_groups
            .iter()
            .find(|g| g.id == group_id)
            .cloned()
        else {
            self.selected_channel_group_id = None;
            ui.label("Selected channel group no longer exists.");
            return;
        };

        let members = self.channel_indices_in_group(group_id);
        ui.heading("Contrast");
        ui.label(format!("Group: {}", group.name));
        ui.label(format!("Applies to {} channel(s).", members.len()));

        if members.is_empty() {
            ui.label("This group has no channels.");
            return;
        }

        let Some((window, mixed)) = self.group_contrast_window_for_indices(&members, abs_max)
        else {
            ui.label("No channels available in this group.");
            return;
        };
        if mixed {
            ui.label("Group channels currently have mixed contrast limits. Applying changes here will overwrite them.");
        }

        let out = contrast::ui_contrast_window(
            ui,
            abs_max,
            window,
            contrast::ContrastUiOptions {
                show_nudge_buttons: true,
                set_max_button_label: "Set Max -> Group",
            },
        );
        let (lo, hi) = out.window;

        if out.set_max_all_clicked {
            for &idx in &members {
                if let Some(dst) = self.channels.get_mut(idx) {
                    let (mut dlo, _) = dst.window.unwrap_or((0.0, abs_max));
                    dlo = dlo.clamp(0.0, abs_max);
                    let dhi = hi.clamp(0.0, abs_max);
                    let dlo = if dhi <= dlo {
                        (dhi - 1.0).clamp(0.0, abs_max)
                    } else {
                        dlo
                    };
                    dst.window = Some((dlo, dhi));
                    self.channel_window_overrides
                        .insert(dst.name.clone(), (dlo, dhi));
                }
            }
            self.bump_render_id();
            return;
        }

        if out.limits_touched {
            for &idx in &members {
                if let Some(dst) = self.channels.get_mut(idx) {
                    dst.window = Some((lo, hi));
                    self.channel_window_overrides
                        .insert(dst.name.clone(), (lo, hi));
                }
            }
            self.bump_render_id();
        }
    }

    fn add_annotation_layer(&mut self) {
        let id = self.next_annotation_layer_id.max(1);
        self.next_annotation_layer_id = id.wrapping_add(1).max(1);
        let name = format!("Annotations {id}");
        self.annotation_layers
            .push(AnnotationPointsLayer::new(id, name));
        self.set_active_layer(LayerId::Annotation(id));
        self.rebuild_layer_orders();
    }

    pub fn add_annotation_layer_from_menu(&mut self) {
        self.add_annotation_layer();
    }

    pub fn open_seg_geojson_dialog(&mut self) {
        let default_dir = self
            .dataset
            .source
            .local_path()
            .and_then(|p| p.parent())
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        self.seg_geojson.open_dialog(&default_dir);
    }

    pub fn open_seg_objects_dialog(&mut self) {
        let default_dir = self
            .dataset
            .source
            .local_path()
            .and_then(|p| p.parent())
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        self.seg_objects.open_dialog(&default_dir);
    }

    fn layer_is_available(&self, id: LayerId) -> bool {
        match id {
            LayerId::SegmentationLabels => self.tiles_gl.is_some(),
            _ => true,
        }
    }

    fn layer_display_name(&self, id: LayerId) -> String {
        match id {
            LayerId::Channel(idx) => self
                .channels
                .get(idx)
                .map(|c| c.name.clone())
                .unwrap_or_else(|| format!("Channel {idx}")),
            LayerId::SpatialImage(id) => self
                .spatial_image_layers
                .images
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.name.clone())
                .unwrap_or_else(|| format!("Image {id}")),
            LayerId::SegmentationLabels => {
                let name = self.seg_label_selected.trim();
                if name.is_empty() {
                    "Segmentation labels".to_string()
                } else {
                    format!("Segmentation ({name})")
                }
            }
            LayerId::SegmentationGeoJson => "Segmentation (GeoJSON)".to_string(),
            LayerId::SegmentationObjects => "Segmentation (Objects)".to_string(),
            LayerId::Mask(id) => self
                .mask_layers
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.name.clone())
                .unwrap_or_else(|| format!("Mask {id}")),
            LayerId::Points => "Points".to_string(),
            LayerId::Annotation(id) => self
                .annotation_layers
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.name.clone())
                .unwrap_or_else(|| format!("Annotations {id}")),
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter()
                .find(|s| s.id == id)
                .map(|s| s.name.clone())
                .unwrap_or_else(|| format!("Shapes {id}")),
            LayerId::SpatialPoints => self
                .spatial_layers
                .points
                .as_ref()
                .map(|p| p.name.clone())
                .unwrap_or_else(|| "Points (SpatialData)".to_string()),
            LayerId::XeniumCells => self
                .xenium_layers
                .cells
                .as_ref()
                .map(|c| c.name.clone())
                .unwrap_or_else(|| "Cells (Xenium)".to_string()),
            LayerId::XeniumTranscripts => self
                .xenium_layers
                .transcripts
                .as_ref()
                .map(|t| t.name.clone())
                .unwrap_or_else(|| "Transcripts (Xenium)".to_string()),
        }
    }

    fn layer_icon(&self, id: LayerId) -> Icon {
        match id {
            LayerId::Channel(_) => Icon::Image,
            LayerId::SpatialImage(_) => Icon::Image,
            LayerId::Points => Icon::Points,
            LayerId::Annotation(_) => Icon::Points,
            LayerId::SpatialPoints => Icon::Points,
            LayerId::XeniumTranscripts => Icon::Points,
            LayerId::SegmentationLabels
            | LayerId::SegmentationGeoJson
            | LayerId::SegmentationObjects
            | LayerId::Mask(_)
            | LayerId::SpatialShape(_)
            | LayerId::XeniumCells => Icon::Polygon,
        }
    }

    fn layer_visible_mut(&mut self, id: LayerId) -> Option<&mut bool> {
        match id {
            LayerId::Channel(idx) => self.channels.get_mut(idx).map(|c| &mut c.visible),
            LayerId::SpatialImage(id) => self
                .spatial_image_layers
                .images
                .iter_mut()
                .find(|l| l.id == id)
                .map(|l| &mut l.visible),
            LayerId::SegmentationLabels => Some(&mut self.cells_outlines_visible),
            LayerId::SegmentationGeoJson => Some(&mut self.seg_geojson.visible),
            LayerId::SegmentationObjects => Some(&mut self.seg_objects.visible),
            LayerId::Mask(id) => self
                .mask_layers
                .iter_mut()
                .find(|l| l.id == id)
                .map(|l| &mut l.visible),
            LayerId::Points => Some(&mut self.cell_points.visible),
            LayerId::Annotation(id) => self
                .annotation_layers
                .iter_mut()
                .find(|l| l.id == id)
                .map(|l| &mut l.visible),
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|s| s.id == id)
                .map(|s| s.visible_mut()),
            LayerId::SpatialPoints => self.spatial_layers.points.as_mut().map(|p| &mut p.visible),
            LayerId::XeniumCells => self.xenium_layers.cells.as_mut().map(|c| &mut c.visible),
            LayerId::XeniumTranscripts => self
                .xenium_layers
                .transcripts
                .as_mut()
                .map(|t| &mut t.visible),
        }
    }

    fn layer_offset_world(&self, id: LayerId) -> egui::Vec2 {
        match id {
            LayerId::Channel(idx) => self
                .channel_offsets_world
                .get(idx)
                .copied()
                .unwrap_or(egui::Vec2::ZERO),
            LayerId::SpatialImage(id) => self
                .spatial_image_layers
                .images
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.offset_world)
                .unwrap_or(egui::Vec2::ZERO),
            LayerId::Points => self.points_offset_world,
            LayerId::Annotation(id) => self
                .annotation_layers
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.offset_world)
                .unwrap_or(egui::Vec2::ZERO),
            LayerId::SpatialPoints => self.spatial_points_offset_world,
            LayerId::Mask(id) => self
                .mask_layers
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.offset_world)
                .unwrap_or(egui::Vec2::ZERO),
            LayerId::SegmentationLabels => self.seg_labels_offset_world,
            LayerId::SegmentationGeoJson => self.seg_geojson_offset_world,
            LayerId::SegmentationObjects => self.seg_objects_offset_world,
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter()
                .find(|s| s.id == id)
                .map(|s| s.offset_world)
                .unwrap_or(egui::Vec2::ZERO),
            LayerId::XeniumCells => self.xenium_cells_offset_world,
            LayerId::XeniumTranscripts => self.xenium_transcripts_offset_world,
        }
    }

    fn layer_offset_world_mut(&mut self, id: LayerId) -> Option<&mut egui::Vec2> {
        match id {
            LayerId::Channel(idx) => self.channel_offsets_world.get_mut(idx),
            LayerId::SpatialImage(id) => self
                .spatial_image_layers
                .images
                .iter_mut()
                .find(|l| l.id == id)
                .map(|l| &mut l.offset_world),
            LayerId::Points => Some(&mut self.points_offset_world),
            LayerId::Annotation(id) => self
                .annotation_layers
                .iter_mut()
                .find(|l| l.id == id)
                .map(|l| &mut l.offset_world),
            LayerId::SpatialPoints => Some(&mut self.spatial_points_offset_world),
            LayerId::Mask(id) => self
                .mask_layers
                .iter_mut()
                .find(|l| l.id == id)
                .map(|l| &mut l.offset_world),
            LayerId::SegmentationLabels => Some(&mut self.seg_labels_offset_world),
            LayerId::SegmentationGeoJson => Some(&mut self.seg_geojson_offset_world),
            LayerId::SegmentationObjects => Some(&mut self.seg_objects_offset_world),
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|s| s.id == id)
                .map(|s| &mut s.offset_world),
            LayerId::XeniumCells => Some(&mut self.xenium_cells_offset_world),
            LayerId::XeniumTranscripts => Some(&mut self.xenium_transcripts_offset_world),
        }
    }

    fn any_visible_channel_offset(&self) -> bool {
        for (i, ch) in self.channels.iter().enumerate() {
            if !ch.visible {
                continue;
            }
            let off = self
                .channel_offsets_world
                .get(i)
                .copied()
                .unwrap_or_default();
            if off.x.abs() > 1e-6 || off.y.abs() > 1e-6 {
                return true;
            }
        }
        false
    }

    fn any_visible_channel_affine(&self) -> bool {
        for (i, ch) in self.channels.iter().enumerate() {
            if !ch.visible {
                continue;
            }
            let scale = self
                .channel_scales
                .get(i)
                .copied()
                .unwrap_or(egui::Vec2::splat(1.0));
            let rot = self.channel_rotations_rad.get(i).copied().unwrap_or(0.0);
            if (scale.x - 1.0).abs() > 1e-6 || (scale.y - 1.0).abs() > 1e-6 || rot.abs() > 1e-6 {
                return true;
            }
        }
        false
    }

    fn union_visible_world_for_visible_channels(&self, visible_world: egui::Rect) -> egui::Rect {
        let mut min_off_x = 0.0f32;
        let mut max_off_x = 0.0f32;
        let mut min_off_y = 0.0f32;
        let mut max_off_y = 0.0f32;
        let mut any = false;
        for (i, ch) in self.channels.iter().enumerate() {
            if !ch.visible {
                continue;
            }
            let off = self
                .channel_offsets_world
                .get(i)
                .copied()
                .unwrap_or_default();
            if !any {
                min_off_x = off.x;
                max_off_x = off.x;
                min_off_y = off.y;
                max_off_y = off.y;
                any = true;
            } else {
                min_off_x = min_off_x.min(off.x);
                max_off_x = max_off_x.max(off.x);
                min_off_y = min_off_y.min(off.y);
                max_off_y = max_off_y.max(off.y);
            }
        }
        if !any {
            return visible_world;
        }

        // For a channel with offset `off`, the region of *data* that must be fetched is
        // `visible_world - off`. Union all of those to avoid missing tiles.
        egui::Rect::from_min_max(
            egui::pos2(
                visible_world.min.x - max_off_x,
                visible_world.min.y - max_off_y,
            ),
            egui::pos2(
                visible_world.max.x - min_off_x,
                visible_world.max.y - min_off_y,
            ),
        )
    }

    fn union_visible_world_for_visible_channels_xform(
        &self,
        visible_world: egui::Rect,
    ) -> egui::Rect {
        let img_world = self.image_local_rect_lvl0();
        let pivot = img_world.center();

        let corners = [
            visible_world.left_top(),
            egui::pos2(visible_world.right(), visible_world.top()),
            visible_world.right_bottom(),
            egui::pos2(visible_world.left(), visible_world.bottom()),
        ];

        let mut acc: Option<egui::Rect> = None;
        for (i, ch) in self.channels.iter().enumerate() {
            if !ch.visible {
                continue;
            }
            let off = self
                .channel_offsets_world
                .get(i)
                .copied()
                .unwrap_or_default();
            let scale = self
                .channel_scales
                .get(i)
                .copied()
                .unwrap_or(egui::Vec2::splat(1.0));
            let rot = self.channel_rotations_rad.get(i).copied().unwrap_or(0.0);

            let mut min_x = f32::INFINITY;
            let mut max_x = f32::NEG_INFINITY;
            let mut min_y = f32::INFINITY;
            let mut max_y = f32::NEG_INFINITY;
            for &c in &corners {
                let p = inv_xform_world_point(c, pivot, off, scale, rot);
                min_x = min_x.min(p.x);
                max_x = max_x.max(p.x);
                min_y = min_y.min(p.y);
                max_y = max_y.max(p.y);
            }
            let r = egui::Rect::from_min_max(egui::pos2(min_x, min_y), egui::pos2(max_x, max_y));
            acc = Some(match acc {
                None => r,
                Some(prev) => prev.union(r),
            });
        }

        acc.unwrap_or(visible_world)
    }

    fn rebuild_layer_orders(&mut self) {
        // Channels: retain valid indices, then append missing.
        let n = self.channels.len();
        self.channel_layer_order.retain(|&i| i < n);
        let mut seen = HashSet::new();
        self.channel_layer_order.retain(|i| seen.insert(*i));
        if self.channel_layer_order.len() != n {
            self.channel_layer_order = (0..n).collect();
        }

        let mut want: Vec<LayerId> = Vec::new();
        for layer in &self.spatial_image_layers.images {
            want.push(LayerId::SpatialImage(layer.id));
        }
        for l in &self.mask_layers {
            want.push(LayerId::Mask(l.id));
        }
        for l in &self.annotation_layers {
            want.push(LayerId::Annotation(l.id));
        }
        if self.seg_geojson.loaded_geojson.is_some() {
            want.push(LayerId::SegmentationGeoJson);
        }
        if self.seg_objects.has_data() {
            want.push(LayerId::SegmentationObjects);
        }
        if self.label_cells.is_some() {
            want.push(LayerId::SegmentationLabels);
        }
        if !self.cell_points.points.is_empty() {
            want.push(LayerId::Points);
        }
        for layer in &self.spatial_layers.shapes {
            want.push(LayerId::SpatialShape(layer.id));
        }
        if self.spatial_layers.points.is_some() {
            want.push(LayerId::SpatialPoints);
        }
        if self.xenium_layers.cells.is_some() {
            want.push(LayerId::XeniumCells);
        }
        if self.xenium_layers.transcripts.is_some() {
            want.push(LayerId::XeniumTranscripts);
        }

        let mut seen2 = HashSet::new();
        self.overlay_layer_order
            .retain(|id| want.contains(id) && seen2.insert(*id));
        for id in want {
            if !self.overlay_layer_order.contains(&id) {
                self.overlay_layer_order.push(id);
            }
        }

        if let LayerId::Channel(idx) = self.active_layer {
            if idx >= n {
                self.active_layer = if n > 0 {
                    LayerId::Channel(0)
                } else {
                    LayerId::Points
                };
            }
        }
        if matches!(
            self.active_layer,
            LayerId::SpatialImage(_)
                | LayerId::Mask(_)
                | LayerId::SegmentationGeoJson
                | LayerId::SegmentationObjects
                | LayerId::SegmentationLabels
                | LayerId::Points
                | LayerId::Annotation(_)
                | LayerId::SpatialShape(_)
                | LayerId::SpatialPoints
                | LayerId::XeniumCells
                | LayerId::XeniumTranscripts
        ) && !self.overlay_layer_order.contains(&self.active_layer)
        {
            self.active_layer = if n > 0 {
                LayerId::Channel(self.selected_channel.min(n - 1))
            } else {
                LayerId::Points
            };
        }
    }

    fn ui_contrast(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        self.ui_tiff_plane_controls(ctx, ui);
        ui.heading("Contrast");

        let abs_max = self.dataset.abs_max.max(1.0);

        let selected_channel = self.selected_channel;
        let Some(selected_info) = self.channels.get(selected_channel).cloned() else {
            ui.label("No channel selected.");
            return;
        };
        let selected_name = selected_info.name.clone();
        ui.label(format!("Channel: {selected_name}"));

        // Optional group + inherit/override semantics (Napari-like).
        let mut groups_cfg = self.project_space.layer_groups().clone();
        let mut groups_changed = false;
        let mut selected_group: Option<u64> = groups_cfg
            .channel_members
            .get(selected_name.as_str())
            .map(|m| m.group_id)
            .filter(|gid| groups_cfg.channel_groups.iter().any(|g| g.id == *gid));

        ui.horizontal(|ui| {
            ui.label("Group");
            egui::ComboBox::from_id_salt("channel-group-select")
                .selected_text(
                    selected_group
                        .and_then(|gid| groups_cfg.channel_groups.iter().find(|g| g.id == gid))
                        .map(|g| g.name.as_str())
                        .unwrap_or("(none)"),
                )
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut selected_group, None, "(none)");
                    for g in &groups_cfg.channel_groups {
                        ui.selectable_value(&mut selected_group, Some(g.id), g.name.clone());
                    }
                });

            if ui
                .button("+ Group")
                .on_hover_text("Create a new group")
                .clicked()
            {
                let existing = groups_cfg
                    .channel_groups
                    .iter()
                    .map(|g| g.id)
                    .collect::<Vec<_>>();
                let id = layer_groups::next_group_id(&existing);
                groups_cfg
                    .channel_groups
                    .push(crate::data::project_config::ProjectChannelGroup {
                        id,
                        name: format!("Group {id}"),
                        expanded: true,
                        color_rgb: [255, 255, 255],
                    });
                selected_group = Some(id);
                groups_changed = true;
            }
        });

        // Apply membership change.
        let have_member = groups_cfg
            .channel_members
            .get(selected_name.as_str())
            .is_some();
        if selected_group.is_none() && have_member {
            groups_cfg.channel_members.remove(selected_name.as_str());
            groups_changed = true;
        } else if let Some(gid) = selected_group {
            match groups_cfg.channel_members.get_mut(selected_name.as_str()) {
                Some(m) => {
                    if m.group_id != gid {
                        m.group_id = gid;
                        groups_changed = true;
                    }
                }
                None => {
                    groups_cfg.channel_members.insert(
                        selected_name.clone(),
                        crate::data::project_config::ProjectChannelGroupMember {
                            group_id: gid,
                            inherit_color: true,
                        },
                    );
                    groups_changed = true;
                }
            }
        }

        let mut inherit_group_color = true;
        let group_color_rgb: Option<[u8; 3]> = selected_group.and_then(|gid| {
            groups_cfg
                .channel_groups
                .iter()
                .find(|g| g.id == gid)
                .map(|g| g.color_rgb)
        });
        if let Some(m) = groups_cfg.channel_members.get(selected_name.as_str()) {
            inherit_group_color = m.inherit_color;
        }

        if let Some(gid) = selected_group {
            ui.horizontal(|ui| {
                ui.add_enabled_ui(
                    groups_cfg
                        .channel_members
                        .contains_key(selected_name.as_str()),
                    |ui| {
                        if ui
                            .checkbox(&mut inherit_group_color, "Inherit group color")
                            .changed()
                        {
                            if let Some(m) =
                                groups_cfg.channel_members.get_mut(selected_name.as_str())
                            {
                                m.inherit_color = inherit_group_color;
                                groups_changed = true;
                            }
                        }
                    },
                );
                if inherit_group_color {
                    if let Some(group) = groups_cfg.channel_groups.iter_mut().find(|g| g.id == gid)
                    {
                        ui.add_space(8.0);
                        ui.label("Group color");
                        let mut c = egui::Color32::from_rgb(
                            group.color_rgb[0],
                            group.color_rgb[1],
                            group.color_rgb[2],
                        );
                        if ui.color_edit_button_srgba(&mut c).changed() {
                            group.color_rgb = [c.r(), c.g(), c.b()];
                            groups_changed = true;
                        }
                    }
                }
            });
        }

        // Channel color (override or ungrouped).
        let allow_channel_color = selected_group.is_none() || !inherit_group_color;
        if let Some(ch) = self.channels.get(selected_channel) {
            let mut c = egui::Color32::from_rgb(ch.color_rgb[0], ch.color_rgb[1], ch.color_rgb[2]);
            let mut changed_color = false;
            ui.horizontal(|ui| {
                ui.label(if allow_channel_color {
                    "Color"
                } else {
                    "Color (override)"
                });
                ui.add_enabled_ui(allow_channel_color, |ui| {
                    changed_color = ui.color_edit_button_srgba(&mut c).changed();
                });
            });
            if changed_color {
                if let Some(dst) = self.channels.get_mut(selected_channel) {
                    dst.color_rgb = [c.r(), c.g(), c.b()];
                }
                self.bump_render_id();
            } else if !allow_channel_color {
                if let Some(rgb) = group_color_rgb {
                    ui.label(format!(
                        "Using group color: rgb({}, {}, {})",
                        rgb[0], rgb[1], rgb[2]
                    ));
                }
            }
        }

        if groups_changed {
            let new_groups = groups_cfg;
            self.project_space.update_layer_groups(|g| *g = new_groups);
            self.bump_render_id();
        }

        let window = selected_info.window.unwrap_or((0.0, abs_max));
        let out = contrast::ui_contrast_window(
            ui,
            abs_max,
            window,
            contrast::ContrastUiOptions {
                show_nudge_buttons: true,
                set_max_button_label: "Set Max -> All",
            },
        );
        let (lo, hi) = out.window;

        if out.set_max_all_clicked {
            let new_hi = hi;
            for dst in &mut self.channels {
                let (mut dlo, _) = dst.window.unwrap_or((0.0, abs_max));
                dlo = dlo.clamp(0.0, abs_max);
                let dhi = new_hi.clamp(0.0, abs_max);
                let dlo = if dhi <= dlo {
                    (dhi - 1.0).clamp(0.0, abs_max)
                } else {
                    dlo
                };
                dst.window = Some((dlo, dhi));
                self.channel_window_overrides
                    .insert(dst.name.clone(), (dlo, dhi));
            }
            self.bump_render_id();
        }

        if out.limits_touched {
            if let Some(dst) = self.channels.get_mut(selected_channel) {
                dst.window = Some((lo, hi));
            }
            self.channel_window_overrides
                .insert(selected_name, (lo, hi));
            self.bump_render_id();
        }

        ui.separator();
        self.ui_histogram(ui, abs_max, (lo, hi));

        ui.separator();
        ui.collapsing("Threshold Regions", |ui| {
            ui.label(
                "Capture the visible pixels from the active channel, preview the thresholded raster on the canvas, then apply it as a mask layer.",
            );
            if self.threshold_region_preview.is_none() {
                if ui.button("Start threshold preview from visible region").clicked() {
                    self.threshold_region_status.clear();
                    if let Err(err) = self.start_threshold_region_preview(ctx) {
                        self.threshold_region_status = format!("Threshold regions failed: {err}");
                    }
                }
            } else {
                let (
                    channel_name,
                    level_index,
                    width,
                    height,
                    plane_min,
                    plane_max,
                    mut threshold,
                    current_min_pixels,
                ) = {
                    let preview = self.threshold_region_preview.as_ref().expect("preview exists");
                    let plane_min = preview.plane.iter().copied().min().unwrap_or(0);
                    let plane_max = preview.plane.iter().copied().max().unwrap_or(0);
                    (
                        preview.channel_name.clone(),
                        preview.level_index,
                        preview.mask.width,
                        preview.mask.height,
                        plane_min,
                        plane_max.max(plane_min),
                        preview.threshold,
                        preview.min_component_pixels,
                    )
                };
                ui.label(format!(
                    "Previewing {channel_name} at level {level_index} ({width} x {height} px)."
                ));
                let threshold_changed = if plane_max > plane_min {
                    ui.add(
                        egui::Slider::new(&mut threshold, plane_min..=plane_max)
                            .text("Threshold")
                            .clamping(egui::SliderClamping::Always),
                    )
                    .changed()
                } else {
                    ui.label(format!("Threshold: {threshold}"));
                    false
                };
                let mut min_pixels = current_min_pixels;
                let min_pixels_changed = ui
                    .horizontal(|ui| {
                        ui.label("Min component pixels");
                        ui.add(
                            egui::DragValue::new(&mut min_pixels)
                                .range(1..=1_000_000)
                                .speed(1.0),
                        )
                        .changed()
                    })
                    .inner;
                let mut preview_changed = false;
                if threshold_changed {
                    if let Some(preview) = self.threshold_region_preview.as_mut() {
                        preview.threshold = threshold;
                    }
                    preview_changed = true;
                }
                if min_pixels_changed && min_pixels != self.threshold_region_min_pixels {
                    self.threshold_region_min_pixels = min_pixels;
                    preview_changed = true;
                }
                if preview_changed {
                    self.recompute_threshold_region_preview(ctx);
                }

                ui.horizontal(|ui| {
                    if ui.button("Refresh from visible region").clicked() {
                        self.threshold_region_status.clear();
                        if let Err(err) = self.start_threshold_region_preview(ctx) {
                            self.threshold_region_status =
                                format!("Threshold regions failed: {err}");
                        }
                    }
                    if ui.button("Apply mask from preview").clicked() {
                        self.threshold_region_status.clear();
                        if let Err(err) = self.create_threshold_mask_from_preview() {
                            self.threshold_region_status =
                                format!("Threshold regions failed: {err}");
                        }
                    }
                    if ui.button("Cancel preview").clicked() {
                        self.threshold_region_preview = None;
                        self.threshold_region_status.clear();
                    }
                });
                ui.small(
                    "The canvas overlay is a raster preview. The pixel grid appears automatically when you zoom in far enough.",
                );
            }
            if self.threshold_region_preview.is_none() {
                ui.horizontal(|ui| {
                    ui.label("Min component pixels");
                    ui.add(
                        egui::DragValue::new(&mut self.threshold_region_min_pixels)
                            .range(1..=1_000_000)
                            .speed(1.0),
                    );
                });
            }
            if !self.threshold_region_status.is_empty() {
                ui.label(self.threshold_region_status.clone());
            }
        });
    }

    fn ui_histogram(&mut self, ui: &mut egui::Ui, abs_max: f32, limits: (f32, f32)) {
        let (rect, _response) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), 140.0),
            egui::Sense::hover(),
        );
        ui.painter()
            .rect_filled(rect, 0.0, egui::Color32::from_gray(18));

        let Some(hist) = self
            .hist
            .as_ref()
            .filter(|h| h.request_id == self.hist_request_id)
        else {
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Histogram: (loading...)".to_string(),
                egui::FontId::proportional(12.0),
                egui::Color32::from_gray(200),
            );
            return;
        };

        let bins = &hist.bins;
        if bins.is_empty() {
            return;
        }
        let max_count = bins.iter().copied().max().unwrap_or(1).max(1) as f32;

        let w = rect.width().max(1.0);
        let h = rect.height().max(1.0);
        let bin_w = w / bins.len() as f32;
        for (i, &c) in bins.iter().enumerate() {
            let x0 = rect.left() + i as f32 * bin_w;
            let x1 = x0 + bin_w;
            let frac = (c as f32) / max_count;
            let y1 = rect.bottom();
            let y0 = y1 - frac * h;
            let r = egui::Rect::from_min_max(egui::pos2(x0, y0), egui::pos2(x1, y1));
            ui.painter()
                .rect_filled(r, 0.0, egui::Color32::from_gray(90));
        }

        let (lo, hi) = limits;
        let x_lo = rect.left() + (lo / abs_max.clamp(1.0, f32::MAX)) * w;
        let x_hi = rect.left() + (hi / abs_max.clamp(1.0, f32::MAX)) * w;
        ui.painter().line_segment(
            [
                egui::pos2(x_lo, rect.top()),
                egui::pos2(x_lo, rect.bottom()),
            ],
            egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 80, 80)),
        );
        ui.painter().line_segment(
            [
                egui::pos2(x_hi, rect.top()),
                egui::pos2(x_hi, rect.bottom()),
            ],
            egui::Stroke::new(2.0, egui::Color32::from_rgb(80, 255, 80)),
        );

        let stats_text = if let Some(s) = hist.stats.as_ref() {
            format!(
                "Min: {:.0} | Q1: {:.0} | Median: {:.0} | Q3: {:.0} | Max: {:.0} (n={})",
                s.min, s.q1, s.median, s.q3, s.max, s.n
            )
        } else {
            "Min: - | Q1: - | Median: - | Q3: - | Max: -".to_string()
        };
        ui.add_space(4.0);
        ui.label(stats_text);
    }

    fn ui_canvas(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let available = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(available, egui::Sense::click_and_drag());
        self.last_canvas_rect = Some(rect);
        ui.painter()
            .rect_filled(rect, 0.0, egui::Color32::from_gray(10));

        let space_down = ctx.input(|i| i.key_down(egui::Key::Space));

        // Gesture handling happens before any drawing so the frame is rendered from a coherent
        // camera/tool snapshot. Each tool owns a separate interaction path here; later code only
        // consumes the resulting state.
        if response.double_clicked() {
            match self.tool_mode {
                ToolMode::Pan => self.fit_to_rect(rect),
                ToolMode::MoveLayer => self.fit_to_rect(rect),
                ToolMode::TransformLayer => self.fit_to_rect(rect),
                ToolMode::DrawMaskPolygon => {
                    if self.drawing_mask_polygon.len() >= 3 {
                        let vertices = std::mem::take(&mut self.drawing_mask_polygon);
                        let id = self
                            .drawing_mask_layer
                            .unwrap_or_else(|| self.ensure_editable_mask_layer());
                        self.drawing_mask_layer = Some(id);
                        if let Some(layer) = self.mask_layers.iter_mut().find(|l| l.id == id) {
                            layer.add_closed_polygon(vertices);
                            layer.visible = true;
                        }
                    }
                }
                ToolMode::RectSelect | ToolMode::LassoSelect => {}
            }
        }

        // Camera navigation is global. It is intentionally independent from layer/tool logic so
        // panning and zooming behave consistently across all content types.
        let mut camera_changed = false;
        if response.hovered() {
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            let pinch = ui.input(|i| i.zoom_delta());
            if scroll != 0.0 {
                if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
                    let factor = (scroll * 0.0015).exp();
                    self.camera.zoom_about_screen_point(rect, pos, factor);
                    camera_changed = true;
                }
            }
            if pinch != 1.0 {
                if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
                    self.camera.zoom_about_screen_point(rect, pos, pinch);
                    camera_changed = true;
                }
            }
        }

        if self.tool_mode == ToolMode::DrawMaskPolygon
            && !space_down
            && response.clicked_by(egui::PointerButton::Primary)
        {
            if let Some(pos) = response.interact_pointer_pos() {
                let world = self.camera.screen_to_world(pos, rect);
                let id = self
                    .drawing_mask_layer
                    .unwrap_or_else(|| self.ensure_editable_mask_layer());
                self.drawing_mask_layer = Some(id);
                let off = self.layer_offset_world(LayerId::Mask(id));
                self.drawing_mask_polygon.push(world - off);
            }
        }

        if self.tool_mode == ToolMode::DrawMaskPolygon && !ctx.wants_keyboard_input() {
            if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                self.drawing_mask_polygon.clear();
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Backspace)) {
                self.drawing_mask_polygon.pop();
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Enter))
                && self.drawing_mask_polygon.len() >= 3
            {
                let vertices = std::mem::take(&mut self.drawing_mask_polygon);
                let id = self
                    .drawing_mask_layer
                    .unwrap_or_else(|| self.ensure_editable_mask_layer());
                self.drawing_mask_layer = Some(id);
                if let Some(layer) = self.mask_layers.iter_mut().find(|l| l.id == id) {
                    layer.add_closed_polygon(vertices);
                    layer.visible = true;
                }
            }
        }

        // Spatial selection tools operate in world coordinates, but they are only valid for a
        // subset of active layers. The drag state is kept separate from the final selection so
        // Escape can cancel the gesture without mutating layer selections.
        let can_rect_select = self.tool_mode == ToolMode::RectSelect
            && !space_down
            && !ctx.wants_keyboard_input()
            && self.active_layer_supports_spatial_selection();
        if can_rect_select && response.drag_started_by(egui::PointerButton::Primary) {
            if let Some(pos) = response.interact_pointer_pos() {
                let world = self.camera.screen_to_world(pos, rect);
                self.selection_rect_start_world = Some(world);
                self.selection_rect_current_world = Some(world);
            }
        }
        if can_rect_select && response.dragged_by(egui::PointerButton::Primary) {
            if let Some(pos) = response.interact_pointer_pos() {
                self.selection_rect_current_world = Some(self.camera.screen_to_world(pos, rect));
            }
        }
        if can_rect_select && response.drag_stopped_by(egui::PointerButton::Primary) {
            if let (Some(start), Some(end)) = (
                self.selection_rect_start_world,
                self.selection_rect_current_world,
            ) {
                let selection_rect = egui::Rect::from_two_pos(start, end);
                let additive = ctx.input(|i| i.modifiers.shift || i.modifiers.command);
                let _ = self.apply_rect_selection_to_active_layer(selection_rect, additive);
            }
            self.clear_spatial_selection_drag();
        }

        let can_lasso_select = self.tool_mode == ToolMode::LassoSelect
            && !space_down
            && !ctx.wants_keyboard_input()
            && self.active_layer_supports_spatial_selection();
        if can_lasso_select && response.drag_started_by(egui::PointerButton::Primary) {
            self.selection_lasso_world.clear();
            if let Some(pos) = response.interact_pointer_pos() {
                self.selection_lasso_world
                    .push(self.camera.screen_to_world(pos, rect));
            }
        }
        if can_lasso_select && response.dragged_by(egui::PointerButton::Primary) {
            if let Some(pos) = response.interact_pointer_pos() {
                let world = self.camera.screen_to_world(pos, rect);
                let min_step_world = 2.0 / self.camera.zoom_screen_per_lvl0_px.max(1e-6);
                let should_push = self
                    .selection_lasso_world
                    .last()
                    .is_none_or(|last| last.distance(world) >= min_step_world);
                if should_push {
                    self.selection_lasso_world.push(world);
                }
            }
        }
        if can_lasso_select && response.drag_stopped_by(egui::PointerButton::Primary) {
            if self.selection_lasso_world.len() >= 3 {
                let additive = ctx.input(|i| i.modifiers.shift || i.modifiers.command);
                let lasso_world = self.selection_lasso_world.clone();
                let _ = self.apply_lasso_selection_to_active_layer(&lasso_world, additive);
            }
            self.clear_spatial_selection_drag();
        }

        if self.tool_mode == ToolMode::Pan
            && !space_down
            && !ctx.wants_keyboard_input()
            && response.clicked_by(egui::PointerButton::Primary)
        {
            if let Some(pos) = response.interact_pointer_pos() {
                let world = self.camera.screen_to_world(pos, rect);
                let mods = ctx.input(|i| i.modifiers);
                match self.active_layer {
                    LayerId::SegmentationObjects => {
                        let off = self.layer_offset_world(LayerId::SegmentationObjects);
                        self.seg_objects.select_at(
                            world,
                            off,
                            &self.camera,
                            mods.shift,
                            mods.command,
                        );
                    }
                    LayerId::SpatialShape(id) => {
                        if let Some(layer) =
                            self.spatial_layers.shapes.iter_mut().find(|s| s.id == id)
                        {
                            layer.select_at(world, mods.shift, mods.command, &self.camera);
                        }
                    }
                    _ => {}
                }
            }
        }

        let cancel_selection_gesture = !ctx.wants_keyboard_input()
            && ctx.input(|i| i.key_pressed(egui::Key::Escape))
            && matches!(self.tool_mode, ToolMode::RectSelect | ToolMode::LassoSelect)
            && (self.selection_rect_start_world.is_some()
                || !self.selection_lasso_world.is_empty());
        if cancel_selection_gesture {
            self.clear_spatial_selection_drag();
        } else if !ctx.wants_keyboard_input() && ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            if matches!(self.tool_mode, ToolMode::RectSelect | ToolMode::LassoSelect) {
                self.clear_spatial_selection_drag();
            }
            match self.active_layer {
                LayerId::SegmentationObjects => self.seg_objects.clear_selection(),
                LayerId::SpatialShape(id) => {
                    if let Some(layer) = self.spatial_layers.shapes.iter_mut().find(|s| s.id == id)
                    {
                        layer.clear_selection();
                    }
                }
                _ => {}
            }
        }

        if self.tool_mode == ToolMode::TransformLayer && !ctx.wants_keyboard_input() {
            if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                self.tool_mode = ToolMode::Pan;
                self.layer_transform = None;
            }
        }

        let can_pan_primary = self.tool_mode == ToolMode::Pan || space_down;
        if can_pan_primary && response.dragged_by(egui::PointerButton::Primary) {
            let delta = ui.input(|i| i.pointer.delta());
            self.camera.pan_by_screen_delta(delta);
            camera_changed = true;
        }

        let can_move_primary = self.tool_mode == ToolMode::MoveLayer && !space_down;
        if can_move_primary && response.drag_started_by(egui::PointerButton::Primary) {
            let layer = self.active_layer;
            let start_offset_world = self.layer_offset_world(layer);
            self.layer_move = Some(LayerMoveState {
                layer,
                start_offset_world,
            });
        }
        if can_move_primary && response.dragged_by(egui::PointerButton::Primary) {
            let z = self.camera.zoom_screen_per_lvl0_px.max(1e-6);
            let (layer, start) = match self.layer_move.as_ref() {
                None => (
                    self.active_layer,
                    self.layer_offset_world(self.active_layer),
                ),
                Some(s) => (s.layer, s.start_offset_world),
            };
            if let Some(delta) = ui.input(|i| i.pointer.total_drag_delta()) {
                if let Some(off) = self.layer_offset_world_mut(layer) {
                    // total_drag_delta is in screen points; convert to world lvl0.
                    *off = start + delta / z;
                }
            }
        }
        if response.drag_stopped_by(egui::PointerButton::Primary) {
            self.layer_move = None;
        }

        // Transform tool (channels only): translate/scale/rotate with handles drawn on-canvas.
        // Use pointer-down instead of drag-start so the grab registers immediately (no drag threshold).
        let can_transform_primary = self.tool_mode == ToolMode::TransformLayer
            && !space_down
            && matches!(self.active_layer, LayerId::Channel(_));

        if can_transform_primary
            && response.is_pointer_button_down_on()
            && ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Primary))
        {
            let pointer = response
                .interact_pointer_pos()
                .or_else(|| ui.input(|i| i.pointer.interact_pos()));
            if let (Some(pointer), LayerId::Channel(ch_idx0)) = (pointer, self.active_layer) {
                let ch_idx = ch_idx0.min(self.channels.len().saturating_sub(1));

                let (pivot_screen, corners, rotate_handle) =
                    self.channel_transform_gizmo_screen(rect, ch_idx);

                let hit_r = 10.0;
                let mut kind = None;
                if rotate_handle.distance(pointer) <= hit_r {
                    kind = Some(LayerTransformKind::Rotate);
                } else {
                    for &c in corners.iter() {
                        if c.distance(pointer) <= hit_r {
                            kind = Some(LayerTransformKind::Scale);
                            break;
                        }
                    }
                }
                if kind.is_none() && point_in_convex_quad(pointer, &corners) {
                    kind = Some(LayerTransformKind::Translate);
                }

                if let Some(kind) = kind {
                    let start_offset_world = self
                        .channel_offsets_world
                        .get(ch_idx)
                        .copied()
                        .unwrap_or_default();
                    let start_scale = self
                        .channel_scales
                        .get(ch_idx)
                        .copied()
                        .unwrap_or(egui::Vec2::splat(1.0));
                    let start_rotation_rad = self
                        .channel_rotations_rad
                        .get(ch_idx)
                        .copied()
                        .unwrap_or(0.0);
                    let start_vec_screen = pointer - pivot_screen;
                    let start_angle_rad = start_vec_screen.y.atan2(start_vec_screen.x);
                    let start_len_screen = start_vec_screen.length().max(1e-6);
                    self.layer_transform = Some(LayerTransformState {
                        layer: LayerId::Channel(ch_idx),
                        kind,
                        start_offset_world,
                        start_scale,
                        start_rotation_rad,
                        pivot_screen,
                        start_pointer_screen: pointer,
                        start_angle_rad,
                        start_len_screen,
                    });
                } else {
                    self.layer_transform = None;
                }
            } else {
                self.layer_transform = None;
            }
        }

        if can_transform_primary
            && response.is_pointer_button_down_on()
            && ui.input(|i| i.pointer.button_down(egui::PointerButton::Primary))
        {
            let z = self.camera.zoom_screen_per_lvl0_px.max(1e-6);
            let pointer = response
                .interact_pointer_pos()
                .or_else(|| ui.input(|i| i.pointer.interact_pos()));
            if let (Some(state), Some(pointer)) = (self.layer_transform.clone(), pointer) {
                if let LayerId::Channel(ch_idx0) = state.layer {
                    let ch_idx = ch_idx0.min(self.channels.len().saturating_sub(1));
                    match state.kind {
                        LayerTransformKind::Translate => {
                            let delta_screen = pointer - state.start_pointer_screen;
                            if let Some(off) = self.channel_offsets_world.get_mut(ch_idx) {
                                *off = state.start_offset_world + delta_screen / z;
                            }
                        }
                        LayerTransformKind::Scale => {
                            let v = pointer - state.pivot_screen;
                            let len = v.length().max(1e-6);
                            let factor = (len / state.start_len_screen).clamp(0.01, 100.0);
                            if let Some(scale) = self.channel_scales.get_mut(ch_idx) {
                                let candidate = state.start_scale * factor;
                                scale.x = candidate.x.clamp(0.01, 100.0);
                                scale.y = candidate.y.clamp(0.01, 100.0);
                            }
                        }
                        LayerTransformKind::Rotate => {
                            let v = pointer - state.pivot_screen;
                            let angle = v.y.atan2(v.x);
                            let delta = angle - state.start_angle_rad;
                            if let Some(rot) = self.channel_rotations_rad.get_mut(ch_idx) {
                                *rot = state.start_rotation_rad + delta;
                            }
                        }
                    }
                    self.hist_dirty = true;
                } else {
                    self.layer_transform = None;
                }
            }
        }

        if can_transform_primary
            && ui.input(|i| i.pointer.button_released(egui::PointerButton::Primary))
        {
            self.layer_transform = None;
        }

        if camera_changed {
            self.hist_dirty = true;
        }

        let visible_world = self.visible_world_rect(rect);
        let visible_world_tiles_world = if self.any_visible_channel_affine() {
            self.union_visible_world_for_visible_channels_xform(visible_world)
        } else if self.any_visible_channel_offset() {
            self.union_visible_world_for_visible_channels(visible_world)
        } else {
            visible_world
        };
        let visible_world_tiles = self.primary_image_world_rect_to_local(visible_world_tiles_world);
        let prev_visible_world_tiles = self.last_visible_world_tiles.unwrap_or(visible_world_tiles);
        let target_level = self.choose_level();

        // Short-lived "zoom-out floor": when zooming out, keep drawing the previous (finer) target
        // level over the previously-visible region for a moment. This avoids sudden blur jumps
        // while the new coarser target is still loading.
        const ZOOM_OUT_FLOOR_MS: u64 = 400;
        let now = Instant::now();
        if let Some(until) = self.zoom_out_floor_until {
            if now > until {
                self.zoom_out_floor_level = None;
                self.zoom_out_floor_until = None;
                self.zoom_out_floor_visible_world_tiles = None;
            }
        }
        if let Some(prev_target) = self.last_target_level {
            if target_level > prev_target {
                self.zoom_out_floor_level = Some(prev_target);
                self.zoom_out_floor_until = Some(now + Duration::from_millis(ZOOM_OUT_FLOOR_MS));
                self.zoom_out_floor_visible_world_tiles = Some(prev_visible_world_tiles);
            } else if target_level < prev_target {
                self.zoom_out_floor_level = None;
                self.zoom_out_floor_until = None;
                self.zoom_out_floor_visible_world_tiles = None;
            }
        }

        // Sticky "fallback ceiling": when zooming in, keep requesting/drawing intermediate levels
        // between the current target and the last coarser target we came from. This avoids a
        // situation where we fall back all the way to the coarsest level once the target settles.
        let coarsest = self.dataset.levels.len().saturating_sub(1);
        let mut ceiling = self.fallback_ceiling_level.unwrap_or(target_level);
        if let Some(prev_target) = self.last_target_level {
            if target_level < prev_target {
                ceiling = ceiling.max(prev_target);
            } else if target_level > prev_target {
                // Zooming out: reset the ceiling to match the new coarser target.
                ceiling = target_level;
            }
        } else {
            ceiling = target_level;
        }
        ceiling = ceiling.min(coarsest);
        self.fallback_ceiling_level = Some(ceiling);

        // Request levels: normal coarse/mid/target plus (when zooming in) the intermediate ladder
        // up to the sticky ceiling. Zoom-out uses a separate "floor" draw-only overlay.
        let mut levels_to_draw = levels_to_draw(self.dataset.levels.len(), target_level);
        if ceiling > target_level {
            for l in target_level..=ceiling {
                levels_to_draw.push(l);
            }
        }
        levels_to_draw.sort_unstable_by(|a, b| b.cmp(a)); // coarse -> fine
        levels_to_draw.dedup();

        let mut needed_per_level: Vec<(usize, crate::data::ome::LevelInfo, Vec<TileKey>)> =
            Vec::with_capacity(levels_to_draw.len());
        for &level in &levels_to_draw {
            let level_info = self.dataset.levels[level].clone();
            let coords: Vec<TileCoord> =
                tiles_needed_lvl0_rect(visible_world_tiles, &level_info, &self.dataset.dims, 1);
            let mut needed: Vec<TileKey> = coords
                .into_iter()
                .map(|c| TileKey {
                    render_id: self.active_render_id,
                    level,
                    tile_y: c.tile_y,
                    tile_x: c.tile_x,
                })
                .collect();
            self.sort_tile_keys_near_center(&level_info, &mut needed);
            needed_per_level.push((level, level_info, needed));
        }
        let render_channels = self.render_channels_for_request(target_level);
        let visible_target_raw_request_count = needed_per_level
            .iter()
            .find(|(level, _, _)| *level == target_level)
            .map(|(_, _, needed)| needed.len())
            .unwrap_or_default()
            .saturating_mul(render_channels.len());
        let raw_tile_cache_capacity = if let Some(tiles_gl) = self.tiles_gl.as_ref() {
            self.maybe_grow_raw_tile_cache(tiles_gl, visible_target_raw_request_count);
            tiles_gl.capacity()
        } else {
            0
        };
        let visible_raw_request_count = needed_per_level
            .iter()
            .map(|(_, _, needed)| needed.len())
            .sum::<usize>()
            .saturating_mul(render_channels.len());
        let high_fanout_raw_request_mode =
            self.tiles_gl.is_some() && render_channels.len() >= RAW_TILE_ADAPTIVE_CHANNEL_THRESHOLD;
        let adaptive_raw_request_mode = self.tiles_gl.is_some()
            && (high_fanout_raw_request_mode
                || visible_raw_request_count > raw_tile_cache_capacity);

        let mut prefetch_needed_per_level: Vec<(usize, Vec<TileKey>)> = Vec::new();
        if !adaptive_raw_request_mode && !self.pinned_levels.has_loading() {
            let target_level_prefetch_needed = needed_per_level
                .iter()
                .find(|(level, _, _)| *level == target_level)
                .map(|(level, level_info, needed)| {
                    self.prefetch_keys_for_level(*level, level_info, visible_world_tiles, needed)
                })
                .unwrap_or_default();
            if !target_level_prefetch_needed.is_empty() {
                prefetch_needed_per_level.push((target_level, target_level_prefetch_needed));
            }
            if self.tile_prefetch_mode == TilePrefetchMode::TargetAndFinerHalo && target_level > 0 {
                let finer_level = target_level - 1;
                if let Some(level_info) = self.dataset.levels.get(finer_level) {
                    let finer_visible: Vec<TileKey> = tiles_needed_lvl0_rect(
                        visible_world_tiles,
                        level_info,
                        &self.dataset.dims,
                        1,
                    )
                    .into_iter()
                    .map(|c| TileKey {
                        render_id: self.active_render_id,
                        level: finer_level,
                        tile_y: c.tile_y,
                        tile_x: c.tile_x,
                    })
                    .collect();
                    let finer_prefetch = self.prefetch_keys_for_level(
                        finer_level,
                        level_info,
                        visible_world_tiles,
                        &finer_visible,
                    );
                    if !finer_prefetch.is_empty() {
                        prefetch_needed_per_level.push((finer_level, finer_prefetch));
                    }
                }
            }
        }

        self.last_target_level = Some(target_level);
        self.last_visible_world_tiles = Some(visible_world_tiles);

        if let (Some(tiles_gl), Some(raw_loader)) =
            (self.tiles_gl.clone(), self.raw_loader.as_ref())
        {
            let raw_tx = raw_loader.tx.clone();
            // If the coarser target level is already "ready enough" over the previous visible
            // region, drop the zoom-out floor early. If not, extend it a bit past the nominal
            // timeout so we don't get a sudden blur jump when IO is slower than expected.
            const ZOOM_OUT_FLOOR_EXTEND_MS: u64 = 200;
            if let Some(floor_level) = self.zoom_out_floor_level {
                if floor_level >= self.dataset.levels.len() || floor_level >= target_level {
                    self.zoom_out_floor_level = None;
                    self.zoom_out_floor_until = None;
                    self.zoom_out_floor_visible_world_tiles = None;
                } else {
                    let floor_rect = self
                        .zoom_out_floor_visible_world_tiles
                        .unwrap_or(prev_visible_world_tiles);
                    if let Some(level_info_tgt) = self.dataset.levels.get(target_level) {
                        let coords_tgt = tiles_needed_lvl0_rect(
                            floor_rect,
                            level_info_tgt,
                            &self.dataset.dims,
                            0,
                        );
                        let sample_max = 16usize;
                        let stride = (coords_tgt.len() / sample_max).max(1);
                        let mut total = 0usize;
                        let mut ready = 0usize;
                        if !coords_tgt.is_empty() && !render_channels.is_empty() {
                            for c in coords_tgt.iter().step_by(stride).take(sample_max) {
                                for ch in &render_channels {
                                    total += 1;
                                    let k = RawTileKey {
                                        level: target_level,
                                        tile_y: c.tile_y,
                                        tile_x: c.tile_x,
                                        channel: ch.index,
                                    };
                                    if tiles_gl.contains(&k) {
                                        ready += 1;
                                    }
                                }
                            }
                        }
                        let ready_enough = total == 0 || ready * 10 >= total * 8; // >=80%
                        if ready_enough {
                            self.zoom_out_floor_level = None;
                            self.zoom_out_floor_until = None;
                            self.zoom_out_floor_visible_world_tiles = None;
                        } else if self.zoom_out_floor_until.map(|u| now > u).unwrap_or(true) {
                            self.zoom_out_floor_until =
                                Some(now + Duration::from_millis(ZOOM_OUT_FLOOR_EXTEND_MS));
                        }
                    }
                }
            }

            let fallback_floor: Option<(usize, crate::data::ome::LevelInfo, Vec<TileKey>)> =
                (|| -> Option<(usize, crate::data::ome::LevelInfo, Vec<TileKey>)> {
                    let floor_level = self.zoom_out_floor_level?;
                    if floor_level >= self.dataset.levels.len() || floor_level >= target_level {
                        return None;
                    }
                    if let Some(until) = self.zoom_out_floor_until {
                        if now > until {
                            return None;
                        }
                    }
                    let floor_rect = self
                        .zoom_out_floor_visible_world_tiles
                        .unwrap_or(prev_visible_world_tiles);
                    if floor_rect.width() <= 0.0 || floor_rect.height() <= 0.0 {
                        return None;
                    }
                    let level_info = self.dataset.levels[floor_level].clone();
                    let coords: Vec<TileCoord> =
                        tiles_needed_lvl0_rect(floor_rect, &level_info, &self.dataset.dims, 1);
                    if coords.is_empty() {
                        return None;
                    }
                    // Keep the draw-only floor lightweight even if the previous visible area was large.
                    let mut needed: Vec<TileKey> = coords
                        .into_iter()
                        .take(1024)
                        .map(|c| TileKey {
                            render_id: self.active_render_id,
                            level: floor_level,
                            tile_y: c.tile_y,
                            tile_x: c.tile_x,
                        })
                        .collect();
                    self.sort_tile_keys_near_center(&level_info, &mut needed);
                    Some((floor_level, level_info, needed))
                })();

            let mut requested_this_frame = 0usize;
            let max_requests_per_frame = 256usize;
            let mut request_levels: Vec<usize> = Vec::new();
            if adaptive_raw_request_mode {
                let coarsest_level = needed_per_level.first().map(|(level, _, _)| *level);
                let bridge_level = target_level.checked_add(1).and_then(|level| {
                    needed_per_level
                        .iter()
                        .any(|(l, _, _)| *l == level)
                        .then_some(level)
                });

                if let Some(level) = bridge_level {
                    request_levels.push(level);
                }
                if let Some(level) = coarsest_level {
                    if high_fanout_raw_request_mode {
                        // With many visible channels, requesting the full coarse ladder causes
                        // the same fallback tiles to dominate the queue. Keep only the bridge
                        // level plus the target level in this mode.
                    } else if Some(level) != bridge_level && level != target_level {
                        request_levels.push(level);
                    }
                }
                request_levels.push(target_level);
            } else {
                request_levels.extend(needed_per_level.iter().map(|(level, _, _)| *level));
            }

            request_levels.sort_unstable();
            request_levels.dedup();

            if adaptive_raw_request_mode {
                crate::log_debug!(
                    "raw request mode: adaptive={} high_fanout={} target={} bridge={:?} levels={:?} channels={} visible_raw={} cache_cap={}",
                    adaptive_raw_request_mode,
                    high_fanout_raw_request_mode,
                    target_level,
                    target_level.checked_add(1).and_then(|level| {
                        needed_per_level
                            .iter()
                            .any(|(l, _, _)| *l == level)
                            .then_some(level)
                    }),
                    request_levels,
                    render_channels.len(),
                    visible_raw_request_count,
                    raw_tile_cache_capacity
                );
            }

            for level in &request_levels {
                let Some((_, _, needed)) = needed_per_level.iter().find(|(l, _, _)| l == level)
                else {
                    continue;
                };
                let phase_max = if adaptive_raw_request_mode && *level != target_level {
                    let tiles_per_phase = if *level == target_level.saturating_add(1) {
                        RAW_TILE_ADAPTIVE_BRIDGE_TILES_PER_FRAME
                    } else {
                        RAW_TILE_ADAPTIVE_COARSE_TILES_PER_FRAME
                    };
                    (requested_this_frame + render_channels.len().saturating_mul(tiles_per_phase))
                        .min(max_requests_per_frame)
                } else {
                    max_requests_per_frame
                };
                self.request_raw_tiles_with_budget(
                    &tiles_gl,
                    &raw_tx,
                    *level,
                    needed,
                    &render_channels,
                    &mut requested_this_frame,
                    phase_max,
                );
            }
            if !adaptive_raw_request_mode {
                for (level, needed) in &prefetch_needed_per_level {
                    if requested_this_frame >= max_requests_per_frame {
                        break;
                    }
                    self.request_raw_tiles_with_budget(
                        &tiles_gl,
                        &raw_tx,
                        *level,
                        needed,
                        &render_channels,
                        &mut requested_this_frame,
                        max_requests_per_frame,
                    );
                }
            }

            let keep_levels: Vec<usize> = if adaptive_raw_request_mode {
                request_levels.clone()
            } else {
                needed_per_level
                    .iter()
                    .map(|(level, _, _)| *level)
                    .collect()
            };

            // Prune stale in-flight requests so the app can go idle immediately after a fast pan/zoom.
            if let Some(tiles_gl_ref) = self.tiles_gl.as_ref() {
                let mut keep: HashSet<RawTileKey> = HashSet::new();
                for (level, _level_info, needed) in needed_per_level.iter() {
                    if !keep_levels.contains(level) {
                        continue;
                    }
                    for key in needed {
                        for ch in &render_channels {
                            keep.insert(RawTileKey {
                                level: *level,
                                tile_y: key.tile_y,
                                tile_x: key.tile_x,
                                channel: ch.index,
                            });
                        }
                    }
                }
                for (level, needed) in &prefetch_needed_per_level {
                    if !keep_levels.contains(level) {
                        continue;
                    }
                    for key in needed {
                        for ch in &render_channels {
                            keep.insert(RawTileKey {
                                level: *level,
                                tile_y: key.tile_y,
                                tile_x: key.tile_x,
                                channel: ch.index,
                            });
                        }
                    }
                }
                if let Some((level, _level_info, needed)) = fallback_floor.as_ref() {
                    for key in needed {
                        for ch in &render_channels {
                            keep.insert(RawTileKey {
                                level: *level,
                                tile_y: key.tile_y,
                                tile_x: key.tile_x,
                                channel: ch.index,
                            });
                        }
                    }
                }
                tiles_gl_ref.prune_in_flight(&keep);
            }

            // Build draw list coarse -> fine.
            let mut draws: Vec<TileDraw> = Vec::new();
            draws.reserve(512);
            let any_affine_visible = self.any_visible_channel_affine();
            let mut max_abs_off_screen = egui::Vec2::ZERO;
            if !any_affine_visible && self.any_visible_channel_offset() {
                let z = self.camera.zoom_screen_per_lvl0_px;
                for (i, ch) in self.channels.iter().enumerate() {
                    if !ch.visible {
                        continue;
                    }
                    let off = self
                        .channel_offsets_world
                        .get(i)
                        .copied()
                        .unwrap_or_default()
                        * z;
                    max_abs_off_screen.x = max_abs_off_screen.x.max(off.x.abs());
                    max_abs_off_screen.y = max_abs_off_screen.y.max(off.y.abs());
                }
            }
            let draw_rect = if max_abs_off_screen.x > 0.0 || max_abs_off_screen.y > 0.0 {
                rect.expand2(max_abs_off_screen + egui::vec2(2.0, 2.0))
            } else {
                rect
            };
            for (level, level_info, needed) in needed_per_level {
                for key in needed {
                    let (_tile_world_rect, tile_screen_rect) =
                        self.tile_rects(&key, rect, &level_info);
                    if any_affine_visible || tile_screen_rect.intersects(draw_rect) {
                        draws.push(TileDraw {
                            level,
                            tile_y: key.tile_y,
                            tile_x: key.tile_x,
                            screen_rect: tile_screen_rect,
                        });
                    }
                }
            }
            // Draw-only zoom-out floor overlay last (finer than the current target).
            if let Some((level, level_info, needed)) = fallback_floor {
                for key in needed {
                    let (_tile_world_rect, tile_screen_rect) =
                        self.tile_rects(&key, rect, &level_info);
                    if any_affine_visible || tile_screen_rect.intersects(draw_rect) {
                        draws.push(TileDraw {
                            level,
                            tile_y: key.tile_y,
                            tile_x: key.tile_x,
                            screen_rect: tile_screen_rect,
                        });
                    }
                }
            }

            let zoom = self.camera.zoom_screen_per_lvl0_px;
            let pivot_screen = self
                .camera
                .world_to_screen(self.image_world_rect_lvl0().center(), rect);
            let mut channel_offsets_world: Vec<egui::Vec2> =
                Vec::with_capacity(render_channels.len());
            let mut channel_xforms_screen: Vec<crate::render::tiles_gl::ChannelScreenTransform> =
                Vec::with_capacity(render_channels.len());
            let mut any_offset = false;
            let mut any_affine = false;
            for ch in &render_channels {
                let idx = self
                    .channels
                    .iter()
                    .position(|c| c.index as u64 == ch.index)
                    .unwrap_or(0);
                let off_world = self
                    .channel_offsets_world
                    .get(idx)
                    .copied()
                    .unwrap_or_default();
                let scale = self
                    .channel_scales
                    .get(idx)
                    .copied()
                    .unwrap_or(egui::Vec2::splat(1.0));
                let rot = self.channel_rotations_rad.get(idx).copied().unwrap_or(0.0);
                any_offset |= off_world.x.abs() > 1e-6 || off_world.y.abs() > 1e-6;
                any_affine |= (scale.x - 1.0).abs() > 1e-6
                    || (scale.y - 1.0).abs() > 1e-6
                    || rot.abs() > 1e-6;
                channel_offsets_world.push(off_world);
                channel_xforms_screen.push(crate::render::tiles_gl::ChannelScreenTransform {
                    pivot_screen,
                    translation_screen: off_world * zoom,
                    scale,
                    rotation_rad: rot,
                });
            }

            let channels: Vec<ChannelDraw> =
                render_channels.into_iter().map(ChannelDraw::from).collect();
            let cb = egui_glow::CallbackFn::new(move |info, painter| {
                if any_affine || any_offset {
                    if any_affine {
                        tiles_gl.paint_with_channel_transforms_screen(
                            info,
                            painter,
                            &draws,
                            &channels,
                            &channel_xforms_screen,
                        );
                        return;
                    }
                    tiles_gl.paint_with_channel_offsets(
                        info,
                        painter,
                        &draws,
                        &channels,
                        &channel_offsets_world,
                        zoom,
                    );
                } else {
                    tiles_gl.paint(info, painter, &draws, &channels);
                }
            });
            ui.painter().add(egui::PaintCallback {
                rect,
                callback: Arc::new(cb),
            });
        } else {
            // CPU (RGBA) path.
            // Request tiles fine -> coarse so zoom-in upgrades quickly.
            let mut requested_this_frame = 0usize;
            let max_requests_per_frame = 64usize;
            for (level, _level_info, needed) in needed_per_level.iter().rev() {
                if requested_this_frame >= max_requests_per_frame {
                    break;
                }
                self.request_tiles_with_budget(
                    *level,
                    needed,
                    &mut requested_this_frame,
                    max_requests_per_frame,
                );
            }
            for (level, needed) in &prefetch_needed_per_level {
                if requested_this_frame >= max_requests_per_frame {
                    break;
                }
                self.request_tiles_with_budget(
                    *level,
                    needed,
                    &mut requested_this_frame,
                    max_requests_per_frame,
                );
            }

            // Prune stale in-flight requests so we don't keep repainting while the loader finishes
            // work that is no longer visible.
            let mut keep: HashSet<TileKey> = HashSet::new();
            for (_level, _level_info, needed) in needed_per_level.iter() {
                for key in needed {
                    keep.insert(*key);
                }
            }
            for (_level, needed) in &prefetch_needed_per_level {
                for key in needed {
                    keep.insert(*key);
                }
            }
            self.cache.prune_in_flight(&keep);

            // Draw tiles coarse -> fine (fallback first, then refine).
            for (level, level_info, needed) in needed_per_level {
                for key in needed {
                    if let Some(tex) = self.get_tile_texture(&key) {
                        let (_tile_world_rect, tile_screen_rect) =
                            self.tile_rects(&key, rect, &level_info);
                        if tile_screen_rect.intersects(rect) {
                            ui.painter().image(
                                tex.id(),
                                tile_screen_rect,
                                egui::Rect::from_min_max(
                                    egui::pos2(0.0, 0.0),
                                    egui::pos2(1.0, 1.0),
                                ),
                                egui::Color32::WHITE,
                            );
                        }
                    } else if level == target_level {
                        let (_tile_world_rect, tile_screen_rect) =
                            self.tile_rects(&key, rect, &level_info);
                        ui.painter().rect_stroke(
                            tile_screen_rect,
                            0.0,
                            egui::Stroke::new(1.0, egui::Color32::from_gray(30)),
                            egui::StrokeKind::Inside,
                        );
                    }
                }
            }
        }

        // If segmentation is hidden, clear any in-flight label tile requests so we don't keep
        // repainting while the background loader drains work we won't display.
        if !self.cells_outlines_visible {
            if let Some(labels_gl) = self.labels_gl.as_ref() {
                let keep: HashSet<LabelTileKey> = HashSet::new();
                labels_gl.prune_in_flight(&keep);
            }
        }

        // Overlays in the user-controlled layer order (bottom -> top).
        self.rebuild_layer_orders();
        let overlay_order = self.overlay_layer_order.clone();
        for layer in overlay_order.into_iter().rev() {
            match layer {
                LayerId::Channel(_) => {}
                LayerId::SpatialImage(id) => {
                    if let Some(layer) = self
                        .spatial_image_layers
                        .images
                        .iter_mut()
                        .find(|l| l.id == id)
                    {
                        layer.draw(ui, &self.camera, rect, visible_world);
                    }
                }
                LayerId::SegmentationLabels => {
                    self.draw_cells_segmentation_overlay(ui, rect, visible_world, target_level);
                }
                LayerId::SegmentationGeoJson => {
                    let off = self.layer_offset_world(LayerId::SegmentationGeoJson);
                    let mut cam = self.camera.clone();
                    cam.center_world_lvl0 -= off;
                    self.seg_geojson.draw(
                        ui,
                        &cam,
                        rect,
                        visible_world.translate(-off),
                        self.tiles_gl.is_some(),
                    );
                }
                LayerId::SegmentationObjects => {
                    let off = self.layer_offset_world(LayerId::SegmentationObjects);
                    self.seg_objects.draw(
                        ui,
                        &self.camera,
                        rect,
                        visible_world,
                        off,
                        self.tiles_gl.is_some(),
                    );
                }
                LayerId::Mask(id) => self.draw_mask_layer_overlay(ui, rect, id),
                LayerId::Points => self.draw_points_overlay(ui, rect, visible_world),
                LayerId::Annotation(id) => {
                    let Some(local_root) = self.current_local_dataset_root() else {
                        continue;
                    };
                    let roi_id = self
                        .project_space
                        .rois()
                        .iter()
                        .find(|r| r.local_path().is_some_and(|path| path == local_root))
                        .map(|r| r.id.clone())
                        .or_else(|| {
                            local_root
                                .file_name()
                                .and_then(|s| s.to_str())
                                .map(|s| s.to_string())
                        })
                        .unwrap_or_else(|| "ROI".to_string());
                    let off = self.layer_offset_world(LayerId::Annotation(id));
                    if let Some(layer) = self.annotation_layers.iter_mut().find(|l| l.id == id) {
                        let group_tint = layer_groups::effective_annotation_tint(
                            self.project_space.layer_groups(),
                            id,
                        );
                        layer.offset_world = off;
                        layer.draw_single(
                            ui,
                            rect,
                            self.camera.center_world_lvl0,
                            self.camera.zoom_screen_per_lvl0_px,
                            roi_id.as_str(),
                            group_tint,
                            self.tiles_gl.is_some(),
                        );
                        if self.active_layer == LayerId::Annotation(id) {
                            if let Some(pointer) = ui.input(|i| i.pointer.hover_pos()) {
                                if rect.contains(pointer) {
                                    let world = self.camera.screen_to_world(pointer, rect);
                                    layer.maybe_hover_tooltip(
                                        ui.ctx(),
                                        rect,
                                        world,
                                        self.camera.zoom_screen_per_lvl0_px,
                                        roi_id.as_str(),
                                        egui::Vec2::ZERO,
                                        1.0,
                                    );
                                }
                            }
                        }
                    }
                }
                LayerId::SpatialShape(id) => {
                    let off = self.layer_offset_world(LayerId::SpatialShape(id));
                    if let Some(layer) = self.spatial_layers.shapes.iter_mut().find(|s| s.id == id)
                    {
                        layer.draw(
                            ui,
                            &self.camera,
                            rect,
                            visible_world,
                            self.tiles_gl.is_some(),
                            off,
                        );
                    }
                }
                LayerId::SpatialPoints => {
                    let off = self.layer_offset_world(LayerId::SpatialPoints);
                    if let Some(layer) = self.spatial_layers.points.as_ref() {
                        layer.draw(ui, rect, &self.camera, off, self.tiles_gl.is_some());
                    }
                }
                LayerId::XeniumCells => {
                    let off = self.layer_offset_world(LayerId::XeniumCells);
                    if let Some(layer) = self.xenium_layers.cells.as_ref() {
                        layer.draw(
                            ui,
                            &self.camera,
                            rect,
                            visible_world,
                            self.tiles_gl.is_some(),
                            off,
                        );
                    }
                }
                LayerId::XeniumTranscripts => {
                    let off = self.layer_offset_world(LayerId::XeniumTranscripts);
                    if let Some(layer) = self.xenium_layers.transcripts.as_ref() {
                        layer.draw(ui, rect, &self.camera, off, self.tiles_gl.is_some());
                    }
                }
            }
        }

        self.draw_threshold_region_preview(ui, rect);

        // In-progress polygon preview (Draw mask tool).
        if self.tool_mode == ToolMode::DrawMaskPolygon && !self.drawing_mask_polygon.is_empty() {
            let mask_id = self.drawing_mask_layer.or_else(|| {
                if let LayerId::Mask(id) = self.active_layer {
                    Some(id)
                } else {
                    None
                }
            });
            let (c, off) = mask_id
                .and_then(|id| {
                    self.mask_layers
                        .iter()
                        .find(|l| l.id == id)
                        .map(|l| (l.color_rgb, l.offset_world))
                })
                .unwrap_or(([255, 210, 60], egui::Vec2::ZERO));

            let color = egui::Color32::from_rgb(c[0], c[1], c[2]);
            let stroke = egui::Stroke::new(2.0, color);

            let pts = self
                .drawing_mask_polygon
                .iter()
                .copied()
                .map(|p| self.camera.world_to_screen(p + off, rect))
                .collect::<Vec<_>>();

            if pts.len() >= 2 {
                ui.painter().add(egui::Shape::line(pts.clone(), stroke));
            }

            if let Some(cursor) = ui.input(|i| i.pointer.hover_pos()) {
                if let Some(last) = pts.last().copied() {
                    ui.painter()
                        .line_segment([last, cursor], egui::Stroke::new(1.0, color));
                }
            }

            for (i, p) in pts.iter().copied().enumerate() {
                let r = if i == 0 { 4.0 } else { 3.0 };
                ui.painter().circle_filled(p, r, color);
            }
        }

        let selection_color = egui::Color32::from_rgba_unmultiplied(255, 210, 80, 180);
        let selection_stroke = egui::Stroke::new(2.0, selection_color);
        if self.tool_mode == ToolMode::RectSelect
            && let (Some(start), Some(end)) = (
                self.selection_rect_start_world,
                self.selection_rect_current_world,
            )
        {
            let rect_screen = egui::Rect::from_two_pos(
                self.camera.world_to_screen(start, rect),
                self.camera.world_to_screen(end, rect),
            );
            ui.painter().rect_filled(
                rect_screen,
                0.0,
                egui::Color32::from_rgba_unmultiplied(255, 210, 80, 36),
            );
            ui.painter()
                .rect_stroke(rect_screen, 0.0, selection_stroke, egui::StrokeKind::Inside);
        }
        if self.tool_mode == ToolMode::LassoSelect && self.selection_lasso_world.len() >= 2 {
            let lasso_screen = self
                .selection_lasso_world
                .iter()
                .copied()
                .map(|point| self.camera.world_to_screen(point, rect))
                .collect::<Vec<_>>();
            ui.painter()
                .add(egui::Shape::line(lasso_screen.clone(), selection_stroke));
            if let (Some(first), Some(last)) =
                (lasso_screen.first().copied(), lasso_screen.last().copied())
            {
                ui.painter().line_segment(
                    [last, first],
                    egui::Stroke::new(1.0, selection_color.gamma_multiply(0.7)),
                );
            }
        }

        let selection_count = self.active_object_selection_count();
        let selection_elements = self.active_object_selection_elements_snapshot();
        response.context_menu(|ui| {
            if selection_count == 0 || !self.active_layer_supports_spatial_selection() {
                ui.label("No selected cells.");
                return;
            }

            ui.label(format!("Selected cells: {selection_count}"));
            if ui
                .button("New selection element from selected cells")
                .clicked()
            {
                let _ = self.create_selection_element_from_active_selection();
                ui.close();
            }
            ui.menu_button("Add selected cells to element", |ui| {
                if selection_elements.is_empty() {
                    ui.label("No selection elements.");
                    return;
                }
                for (idx, name, count) in &selection_elements {
                    if ui.button(format!("{name} ({count})")).clicked() {
                        let _ = self.add_active_selection_to_element(*idx);
                        ui.close();
                    }
                }
            });
            if ui.button("Clear selection").clicked() {
                match self.active_layer {
                    LayerId::SegmentationObjects => self.seg_objects.clear_selection(),
                    LayerId::SpatialShape(id) => {
                        if let Some(layer) = self
                            .spatial_layers
                            .shapes
                            .iter_mut()
                            .find(|layer| layer.id == id)
                        {
                            layer.clear_selection();
                        }
                    }
                    _ => {}
                }
                ui.close();
            }
        });

        let screenshot = self.screenshot_pending.take();
        let screenshot_active = screenshot.is_some();

        // HUD (disabled while capturing screenshots).
        if !screenshot_active {
            let hud = format!(
                "level {target_level} zoom {:.3}  center ({:.0}, {:.0})",
                self.camera.zoom_screen_per_lvl0_px,
                self.camera.center_world_lvl0.x,
                self.camera.center_world_lvl0.y
            );
            canvas_overlays::paint_hud(ui, rect, hud);
        }

        // Scale bar (bottom-left). Uses microns if the dataset encodes physical units;
        // otherwise falls back to pixels.
        let draw_scale_bar = screenshot
            .as_ref()
            .map(|s| s.settings.include_scale_bar)
            .unwrap_or(self.show_scale_bar);
        if draw_scale_bar {
            canvas_overlays::paint_scale_bar(
                ui,
                rect,
                canvas_overlays::ScaleBarParams {
                    zoom_screen_per_lvl0_px: self.camera.zoom_screen_per_lvl0_px,
                    um_per_lvl0_px: self.dataset_pixel_size_um(),
                    scale: screenshot
                        .as_ref()
                        .map(|s| s.settings.scale_bar_scale)
                        .unwrap_or(1.0),
                },
            );
        }

        // Legend (bottom-right) for visible channels (screenshot-only for now).
        if screenshot
            .as_ref()
            .is_some_and(|s| s.settings.include_legend)
        {
            let groups = self.project_space.layer_groups();
            let order = if self.channel_layer_order.len() == self.channels.len() {
                self.channel_layer_order.clone()
            } else {
                (0..self.channels.len()).collect()
            };
            let mut entries: Vec<(egui::Color32, String)> = Vec::new();
            for idx in order {
                let Some(ch) = self.channels.get(idx) else {
                    continue;
                };
                if !ch.visible {
                    continue;
                }
                let rgb = layer_groups::effective_channel_color_rgb(
                    groups,
                    ch.name.as_str(),
                    ch.color_rgb,
                );
                entries.push((
                    egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]),
                    ch.name.clone(),
                ));
            }
            canvas_overlays::paint_marker_legend(
                ui,
                rect,
                &entries,
                screenshot
                    .as_ref()
                    .map(|s| s.settings.legend_scale)
                    .unwrap_or(1.0),
            );
        }

        // Transform gizmo overlay (for channels only).
        if self.tool_mode == ToolMode::TransformLayer {
            if let LayerId::Channel(ch_idx) = self.active_layer {
                self.draw_channel_transform_gizmo(ui, rect, ch_idx);
            }
        }

        // Loading indicator (top-right). Avoid capturing transient spinners in screenshots.
        if !screenshot_active {
            let loading_text = self.loading_indicator_text();
            let tile_loading_count = self.image_tile_request_count();
            let spinner_text = if self.show_tile_debug && tile_loading_count > 0 {
                Some(format!("{tile_loading_count} tiles"))
            } else {
                None
            };
            canvas_overlays::paint_spinner(
                ui,
                rect,
                loading_text.is_some(),
                spinner_text.as_deref(),
            );
            if let Some(text) = loading_text {
                canvas_overlays::paint_loading_badge(ui, rect, text);
            }
        }

        // Always-on hover tooltip (active layer only). Avoid capturing tooltips in screenshots.
        if !screenshot_active {
            // Important: when Segmentation Objects is the active layer, hover picking can be expensive
            // at low zoom. Avoid doing that work while the camera is actively moving.
            self.ui_active_layer_tooltip(ui, ctx, rect, &response, camera_changed);
        }

        // Screenshot capture: read back the canvas pixels after overlays have been drawn.
        if let Some(spec) = screenshot {
            let tx = self.screenshot_worker.tx.clone();
            let id = spec.id;
            let path = spec.path.clone();
            let capture_rect = rect;
            let cb = egui_glow::CallbackFn::new(move |info, painter| {
                let viewport = info.viewport;
                let ppp = info.pixels_per_point.max(1e-6);
                let viewport_w_px = (viewport.width().max(1.0) * ppp).round().max(1.0) as i32;
                let viewport_h_px = (viewport.height().max(1.0) * ppp).round().max(1.0) as i32;

                let mut x_px = ((capture_rect.min.x - viewport.min.x) * ppp)
                    .round()
                    .max(0.0) as i32;
                let mut y_px = ((viewport.max.y - capture_rect.max.y) * ppp)
                    .round()
                    .max(0.0) as i32;
                let mut w_px = (capture_rect.width() * ppp).round().max(1.0) as i32;
                let mut h_px = (capture_rect.height() * ppp).round().max(1.0) as i32;

                if x_px >= viewport_w_px || y_px >= viewport_h_px {
                    return;
                }
                if x_px + w_px > viewport_w_px {
                    w_px = (viewport_w_px - x_px).max(1);
                }
                if y_px + h_px > viewport_h_px {
                    h_px = (viewport_h_px - y_px).max(1);
                }
                if w_px <= 0 || h_px <= 0 {
                    return;
                }

                let gl = painter.gl();
                let mut rgba = vec![0u8; (w_px as usize) * (h_px as usize) * 4];
                unsafe {
                    let gl_ref = gl.as_ref();
                    gl_ref.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
                    gl_ref.read_pixels(
                        x_px,
                        y_px,
                        w_px,
                        h_px,
                        glow::RGBA,
                        glow::UNSIGNED_BYTE,
                        glow::PixelPackData::Slice(Some(rgba.as_mut_slice())),
                    );
                }
                let _ = tx.send(ScreenshotWorkerMsg::SavePng {
                    id,
                    path: path.clone(),
                    width: w_px as usize,
                    height: h_px as usize,
                    rgba_bottom_up: rgba,
                });
            });
            ui.painter().add(egui::PaintCallback {
                rect,
                callback: Arc::new(cb),
            });
        }
    }

    fn ui_active_layer_tooltip(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        rect: egui::Rect,
        response: &egui::Response,
        camera_changed: bool,
    ) {
        const HOVER_TOOLTIP_DWELL: Duration = Duration::from_millis(120);
        const HOVER_TOOLTIP_GRACE: Duration = Duration::from_millis(180);

        if !response.hovered() {
            self.hover_tooltip_state = None;
            return;
        }
        // Don't show tooltips while dragging/panning/transforming.
        if ui.input(|i| i.pointer.any_down()) {
            self.hover_tooltip_state = None;
            return;
        }
        if ctx.wants_keyboard_input() {
            self.hover_tooltip_state = None;
            return;
        }
        let Some(pointer_screen) = ui.input(|i| i.pointer.hover_pos()) else {
            self.hover_tooltip_state = None;
            return;
        };
        if !rect.contains(pointer_screen) {
            self.hover_tooltip_state = None;
            return;
        }
        let now = Instant::now();

        if camera_changed {
            self.hover_tooltip_state = None;
            return;
        }

        let pointer_world = self.camera.screen_to_world(pointer_screen, rect);
        let lines: Option<Vec<String>> = match self.active_layer {
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter()
                .find(|s| s.id == id)
                .and_then(|s| s.hover_tooltip(pointer_world, &self.camera)),
            LayerId::SpatialPoints => {
                let off = self.layer_offset_world(LayerId::SpatialPoints);
                self.spatial_layers
                    .points
                    .as_ref()
                    .and_then(|p| p.hover_tooltip(pointer_world, off, &self.camera))
            }
            LayerId::XeniumTranscripts => {
                let off = self.layer_offset_world(LayerId::XeniumTranscripts);
                self.xenium_layers
                    .transcripts
                    .as_ref()
                    .and_then(|t| t.hover_tooltip(pointer_world, off, &self.camera))
            }
            LayerId::SegmentationObjects => {
                let off = self.layer_offset_world(LayerId::SegmentationObjects);
                self.seg_objects
                    .hover_tooltip(pointer_world, off, &self.camera)
            }
            _ => None,
        };
        let has_lines = lines.is_some();

        if let Some(lines) = lines {
            let signature = lines.join("\n");
            match self.hover_tooltip_state.as_mut() {
                Some(state) if state.signature == signature => {
                    state.last_seen = now;
                    if !state.visible && now.duration_since(state.first_seen) >= HOVER_TOOLTIP_DWELL
                    {
                        state.visible = true;
                    }
                    state.lines = lines;
                }
                Some(state) => {
                    *state = HoverTooltipState {
                        signature,
                        lines,
                        first_seen: now,
                        last_seen: now,
                        visible: false,
                    };
                }
                None => {
                    self.hover_tooltip_state = Some(HoverTooltipState {
                        signature,
                        lines,
                        first_seen: now,
                        last_seen: now,
                        visible: false,
                    });
                }
            }
        } else if self
            .hover_tooltip_state
            .as_ref()
            .is_some_and(|state| now.duration_since(state.last_seen) > HOVER_TOOLTIP_GRACE)
        {
            self.hover_tooltip_state = None;
        }

        if let Some(state) = self.hover_tooltip_state.as_ref() {
            if !state.visible {
                let elapsed = now.duration_since(state.first_seen);
                if elapsed < HOVER_TOOLTIP_DWELL {
                    ctx.request_repaint_after(HOVER_TOOLTIP_DWELL - elapsed);
                } else {
                    ctx.request_repaint();
                }
            } else if !has_lines {
                let elapsed = now.duration_since(state.last_seen);
                if elapsed < HOVER_TOOLTIP_GRACE {
                    ctx.request_repaint_after(HOVER_TOOLTIP_GRACE - elapsed);
                }
            }
        }

        if let Some(state) = self.hover_tooltip_state.as_ref()
            && state.visible
            && now.duration_since(state.last_seen) <= HOVER_TOOLTIP_GRACE
        {
            let lines = state.lines.clone();
            crate::ui::tooltip::show_tooltip_at_pointer(
                ctx,
                ui.id().with("hover_layer_tooltip"),
                |ui| {
                    for l in lines {
                        ui.label(l);
                    }
                },
            );
        }
    }

    fn draw_cells_segmentation_overlay(
        &mut self,
        ui: &mut egui::Ui,
        rect: egui::Rect,
        visible_world: egui::Rect,
        target_level: usize,
    ) {
        if !self.cells_outlines_visible {
            return;
        }

        let off = self.layer_offset_world(LayerId::SegmentationLabels);
        let visible_world = visible_world.translate(-off);
        let off_screen = off * self.camera.zoom_screen_per_lvl0_px;

        let (Some(lbl), Some(loader), Some(renderer)) = (
            self.label_cells.as_ref(),
            self.label_loader.as_ref(),
            self.labels_gl.clone(),
        ) else {
            return;
        };
        let Some(xforms) = self.label_cells_xform.as_ref() else {
            return;
        };

        // Keep segmentation level selection locked to the image level to avoid drift when the
        // label pyramid scale metadata or rounding differs.
        let target_label_level = target_level.min(lbl.levels.len().saturating_sub(1));
        let levels_to_draw = vec![target_label_level];

        let mut needed_per_level: Vec<(usize, crate::data::ome::LevelInfo, Vec<LabelTileKey>)> =
            Vec::with_capacity(levels_to_draw.len());
        for &level in &levels_to_draw {
            let level_info = lbl.levels[level].clone();
            let xform = xforms.get(level).copied().unwrap_or(LabelToWorld {
                scale_x: level_info.downsample,
                scale_y: level_info.downsample,
                offset_x: 0.0,
                offset_y: 0.0,
                approx_downsample: level_info.downsample,
            });

            let inv_x = 1.0 / xform.scale_x.max(1e-6);
            let inv_y = 1.0 / xform.scale_y.max(1e-6);
            let visible_lvl = egui::Rect::from_min_max(
                egui::pos2(
                    (visible_world.min.x - xform.offset_x) * inv_x,
                    (visible_world.min.y - xform.offset_y) * inv_y,
                ),
                egui::pos2(
                    (visible_world.max.x - xform.offset_x) * inv_x,
                    (visible_world.max.y - xform.offset_y) * inv_y,
                ),
            );

            let y_dim = lbl.dims.y;
            let x_dim = lbl.dims.x;
            let shape_y = level_info.shape[y_dim] as f32;
            let shape_x = level_info.shape[x_dim] as f32;
            let image_rect_lvl =
                egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(shape_x, shape_y));
            let visible_lvl = visible_lvl.intersect(image_rect_lvl.expand(1.0));

            let chunk_y = level_info.chunks[y_dim] as f32;
            let chunk_x = level_info.chunks[x_dim] as f32;

            let tile_y0 = (visible_lvl.min.y / chunk_y).floor().max(0.0) as i64 - 1;
            let tile_x0 = (visible_lvl.min.x / chunk_x).floor().max(0.0) as i64 - 1;
            let tile_y1 = (visible_lvl.max.y / chunk_y).ceil().max(0.0) as i64 + 1;
            let tile_x1 = (visible_lvl.max.x / chunk_x).ceil().max(0.0) as i64 + 1;

            let needed = self.label_tiles_needed_with_xform(
                level,
                tile_y0,
                tile_y1,
                tile_x0,
                tile_x1,
                &level_info,
                &lbl.dims,
                xform,
            );
            needed_per_level.push((level, level_info, needed));
        }

        // Prune stale in-flight label tile requests so we don't keep repainting after a fast pan/zoom.
        if let Some(labels_gl_ref) = self.labels_gl.as_ref() {
            let mut keep: HashSet<LabelTileKey> = HashSet::new();
            for (_level, _level_info, needed) in needed_per_level.iter() {
                for k in needed {
                    keep.insert(*k);
                }
            }
            labels_gl_ref.prune_in_flight(&keep);
        }

        // Request fine -> coarse so zoom-in upgrades quickly.
        let mut requested_this_frame = 0usize;
        let max_requests_per_frame = 128usize;
        for (_level, _level_info, needed) in needed_per_level.iter().rev() {
            if requested_this_frame >= max_requests_per_frame {
                break;
            }
            for key in needed {
                if requested_this_frame >= max_requests_per_frame {
                    break;
                }
                if renderer.mark_in_flight(*key) {
                    let _ = loader.tx.send(LabelTileRequest { key: *key });
                    requested_this_frame += 1;
                }
            }
        }

        // Draw list coarse -> fine.
        let mut draws: Vec<LabelDraw> = Vec::new();
        draws.reserve(512);
        for (level, level_info, needed) in needed_per_level {
            let xform = xforms.get(level).copied().unwrap_or(LabelToWorld {
                scale_x: level_info.downsample,
                scale_y: level_info.downsample,
                offset_x: 0.0,
                offset_y: 0.0,
                approx_downsample: level_info.downsample,
            });
            for key in needed {
                let (_world_rect, screen_rect) =
                    self.label_tile_rects(&key, rect, &level_info, &lbl.dims, xform);
                let screen_rect = screen_rect.translate(off_screen);
                if screen_rect.intersects(rect) {
                    draws.push(LabelDraw {
                        level,
                        tile_y: key.tile_y,
                        tile_x: key.tile_x,
                        screen_rect,
                    });
                }
            }
        }

        let c = self.cells_outlines_color_rgb;
        let params = OutlinesParams {
            visible: true,
            color_rgb: [
                c[0] as f32 / 255.0,
                c[1] as f32 / 255.0,
                c[2] as f32 / 255.0,
            ],
            opacity: self.cells_outlines_opacity,
            width_screen_px: self.cells_outlines_width_px,
        };
        let cb = egui_glow::CallbackFn::new(move |info, painter| {
            renderer.paint(info, painter, &draws, params);
        });
        ui.painter().add(egui::PaintCallback {
            rect,
            callback: Arc::new(cb),
        });
    }

    fn draw_mask_layer_overlay(&self, ui: &mut egui::Ui, rect: egui::Rect, id: u64) {
        let Some(layer) = self.mask_layers.iter().find(|l| l.id == id) else {
            return;
        };
        if !layer.visible || layer.polygons_world.is_empty() {
            return;
        }

        let off = layer.offset_world;
        let c = layer.color_rgb;
        let a = (layer.opacity.clamp(0.0, 1.0) * 255.0).round() as u8;
        let color = egui::Color32::from_rgba_premultiplied(c[0], c[1], c[2], a);
        let stroke = egui::Stroke::new(layer.width_screen_px, color);

        for poly in &layer.polygons_world {
            if poly.len() < 2 {
                continue;
            }
            let pts = poly
                .iter()
                .copied()
                .map(|p| self.camera.world_to_screen(p + off, rect))
                .collect::<Vec<_>>();
            ui.painter().add(egui::Shape::line(pts, stroke));
        }
    }

    fn draw_points_overlay(
        &mut self,
        ui: &mut egui::Ui,
        rect: egui::Rect,
        visible_world: egui::Rect,
    ) {
        let off = self.layer_offset_world(LayerId::Points);
        if let Some(renderer) = self.points_gl.clone() {
            if let Some((generation, positions_world, values)) = self.cell_thresholds.gpu_points() {
                let data = PointsGlDrawData {
                    generation,
                    positions_world,
                    values,
                };
                let params = PointsGlDrawParams {
                    center_world: self.camera.center_world_lvl0,
                    zoom_screen_per_world: self.camera.zoom_screen_per_lvl0_px,
                    threshold: self.cell_thresholds.threshold(),
                    style: self.cell_points.style.clone(),
                    visible: self.cell_points.visible,
                    local_to_world_offset: off,
                    local_to_world_scale: egui::vec2(1.0, 1.0),
                };
                let cb = egui_glow::CallbackFn::new(move |info, painter| {
                    renderer.paint(info, painter, &data, &params);
                });
                ui.painter().add(egui::PaintCallback {
                    rect,
                    callback: Arc::new(cb),
                });
            }
        } else {
            let world_to_screen = |p: egui::Pos2| self.camera.world_to_screen(p + off, rect);
            self.cell_points.draw(
                ui.painter(),
                rect,
                world_to_screen,
                visible_world.translate(-off),
                self.camera.zoom_screen_per_lvl0_px,
            );
        }
    }

    fn label_tiles_needed_with_xform(
        &self,
        level: usize,
        tile_y0: i64,
        tile_y1: i64,
        tile_x0: i64,
        tile_x1: i64,
        level_info: &crate::data::ome::LevelInfo,
        dims: &crate::data::ome::Dims,
        xform: LabelToWorld,
    ) -> Vec<LabelTileKey> {
        let mut keys = Vec::new();

        let y_dim = dims.y;
        let x_dim = dims.x;
        let max_tiles_y = ((level_info.shape[y_dim] + level_info.chunks[y_dim] - 1)
            / level_info.chunks[y_dim]) as i64;
        let max_tiles_x = ((level_info.shape[x_dim] + level_info.chunks[x_dim] - 1)
            / level_info.chunks[x_dim]) as i64;

        let y0 = tile_y0.clamp(0, max_tiles_y);
        let y1 = tile_y1.clamp(0, max_tiles_y);
        let x0 = tile_x0.clamp(0, max_tiles_x);
        let x1 = tile_x1.clamp(0, max_tiles_x);

        for ty in y0..y1 {
            for tx in x0..x1 {
                keys.push(LabelTileKey {
                    level,
                    tile_y: ty as u64,
                    tile_x: tx as u64,
                });
            }
        }

        // Near-to-center priority for request ordering.
        let center_world = self.camera.center_world_lvl0;
        let inv_x = 1.0 / xform.scale_x.max(1e-6);
        let inv_y = 1.0 / xform.scale_y.max(1e-6);
        let center_lvl = egui::pos2(
            (center_world.x - xform.offset_x) * inv_x,
            (center_world.y - xform.offset_y) * inv_y,
        );
        let chunk_y = level_info.chunks[y_dim] as f32;
        let chunk_x = level_info.chunks[x_dim] as f32;
        keys.sort_by(|a, b| {
            let ay = (a.tile_y as f32 + 0.5) * chunk_y;
            let ax = (a.tile_x as f32 + 0.5) * chunk_x;
            let by = (b.tile_y as f32 + 0.5) * chunk_y;
            let bx = (b.tile_x as f32 + 0.5) * chunk_x;
            let da = (ax - center_lvl.x).powi(2) + (ay - center_lvl.y).powi(2);
            let db = (bx - center_lvl.x).powi(2) + (by - center_lvl.y).powi(2);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });

        keys
    }

    fn label_tile_rects(
        &self,
        key: &LabelTileKey,
        viewport: egui::Rect,
        level_info: &crate::data::ome::LevelInfo,
        dims: &crate::data::ome::Dims,
        xform: LabelToWorld,
    ) -> (egui::Rect, egui::Rect) {
        let y_dim = dims.y;
        let x_dim = dims.x;
        let chunk_y = level_info.chunks[y_dim] as f32;
        let chunk_x = level_info.chunks[x_dim] as f32;

        let y0 = key.tile_y as f32 * chunk_y;
        let x0 = key.tile_x as f32 * chunk_x;
        let y1 = (y0 + chunk_y).min(level_info.shape[y_dim] as f32);
        let x1 = (x0 + chunk_x).min(level_info.shape[x_dim] as f32);

        let world_min = egui::pos2(
            x0 * xform.scale_x + xform.offset_x,
            y0 * xform.scale_y + xform.offset_y,
        );
        let world_max = egui::pos2(
            x1 * xform.scale_x + xform.offset_x,
            y1 * xform.scale_y + xform.offset_y,
        );
        let world_rect = egui::Rect::from_min_max(world_min, world_max);

        let screen_min = self.camera.world_to_screen(world_min, viewport);
        let screen_max = self.camera.world_to_screen(world_max, viewport);
        let screen_rect = egui::Rect::from_min_max(screen_min, screen_max);

        (world_rect, screen_rect)
    }

    fn visible_world_rect(&self, viewport: egui::Rect) -> egui::Rect {
        let world_min = self.camera.screen_to_world(viewport.left_top(), viewport);
        let world_max = self
            .camera
            .screen_to_world(viewport.right_bottom(), viewport);
        egui::Rect::from_min_max(world_min, world_max)
    }

    fn image_local_rect_lvl0(&self) -> egui::Rect {
        let shape0 = &self.dataset.levels[0].shape;
        let y = shape0[self.dataset.dims.y] as f32;
        let x = shape0[self.dataset.dims.x] as f32;
        egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(x, y))
    }

    fn primary_image_local_to_world(&self, p: egui::Pos2) -> egui::Pos2 {
        p
    }

    fn primary_image_world_to_local(&self, p: egui::Pos2) -> egui::Pos2 {
        p
    }

    fn primary_image_world_rect_to_local(&self, rect: egui::Rect) -> egui::Rect {
        rect
    }

    fn image_world_rect_lvl0(&self) -> egui::Rect {
        self.image_local_rect_lvl0()
    }

    fn channel_transform_gizmo_screen(
        &self,
        viewport: egui::Rect,
        ch_idx: usize,
    ) -> (egui::Pos2, [egui::Pos2; 4], egui::Pos2) {
        let img_world = self.image_world_rect_lvl0();
        let pivot_world = img_world.center();
        let pivot_screen = self.camera.world_to_screen(pivot_world, viewport);

        let zoom = self.camera.zoom_screen_per_lvl0_px;
        let off_world = self
            .channel_offsets_world
            .get(ch_idx)
            .copied()
            .unwrap_or_default();
        let trans_screen = off_world * zoom;
        let pivot_screen_effective = pivot_screen + trans_screen;
        let scale = self
            .channel_scales
            .get(ch_idx)
            .copied()
            .unwrap_or(egui::Vec2::splat(1.0));
        let rot = self
            .channel_rotations_rad
            .get(ch_idx)
            .copied()
            .unwrap_or(0.0);

        // Base image corners in screen space (untransformed).
        let tl = self.camera.world_to_screen(img_world.left_top(), viewport);
        let tr = self
            .camera
            .world_to_screen(egui::pos2(img_world.right(), img_world.top()), viewport);
        let br = self
            .camera
            .world_to_screen(img_world.right_bottom(), viewport);
        let bl = self
            .camera
            .world_to_screen(egui::pos2(img_world.left(), img_world.bottom()), viewport);

        let corners = [
            xform_screen_point(tl, pivot_screen, trans_screen, scale, rot),
            xform_screen_point(tr, pivot_screen, trans_screen, scale, rot),
            xform_screen_point(br, pivot_screen, trans_screen, scale, rot),
            xform_screen_point(bl, pivot_screen, trans_screen, scale, rot),
        ];

        let center = quad_center(&corners);
        let top_mid = (corners[0].to_vec2() + corners[1].to_vec2()) * 0.5;
        let outward = {
            let v = top_mid - center.to_vec2();
            if v.length() > 1e-6 {
                v / v.length()
            } else {
                egui::vec2(0.0, -1.0)
            }
        };
        let rotate_handle_v = top_mid + outward * 26.0;
        let rotate_handle = egui::pos2(rotate_handle_v.x, rotate_handle_v.y);

        (pivot_screen_effective, corners, rotate_handle)
    }

    fn draw_channel_transform_gizmo(&self, ui: &mut egui::Ui, viewport: egui::Rect, ch_idx: usize) {
        if ch_idx >= self.channels.len() {
            return;
        }

        let (_pivot, corners, rotate_handle) =
            self.channel_transform_gizmo_screen(viewport, ch_idx);
        let base_color = egui::Color32::from_rgb(120, 200, 255);
        let hover_color = egui::Color32::from_rgb(255, 180, 60);
        let stroke = egui::Stroke::new(1.6, base_color);
        let handle_r = 4.5;

        let mut hover_corner: Option<usize> = None;
        let mut hover_rotate = false;
        let hit_r = 10.0;
        if let Some(pointer) = ui.input(|i| i.pointer.hover_pos()) {
            if viewport.contains(pointer) {
                for (i, &c) in corners.iter().enumerate() {
                    if c.distance(pointer) <= hit_r {
                        hover_corner = Some(i);
                        break;
                    }
                }
                hover_rotate = rotate_handle.distance(pointer) <= hit_r;
            }
        }

        // Outline.
        let mut pts = Vec::with_capacity(5);
        pts.push(corners[0]);
        pts.push(corners[1]);
        pts.push(corners[2]);
        pts.push(corners[3]);
        pts.push(corners[0]);
        ui.painter().add(egui::Shape::line(pts, stroke));

        // Corner handles.
        for (i, &c) in corners.iter().enumerate() {
            let fill = if hover_corner == Some(i) {
                hover_color
            } else {
                base_color
            };
            ui.painter().circle_filled(c, handle_r, fill);
        }

        // Rotate handle + connector.
        let top_mid_v = (corners[0].to_vec2() + corners[1].to_vec2()) * 0.5;
        let top_mid = egui::pos2(top_mid_v.x, top_mid_v.y);
        let rotate_stroke = if hover_rotate {
            egui::Stroke::new(stroke.width, hover_color)
        } else {
            stroke
        };
        ui.painter()
            .line_segment([top_mid, rotate_handle], rotate_stroke);
        ui.painter()
            .circle_stroke(rotate_handle, handle_r + 1.0, rotate_stroke);
    }

    fn fit_to_last_canvas(&mut self) {
        let Some(viewport) = self.last_canvas_rect else {
            return;
        };
        self.fit_to_rect(viewport);
    }

    pub fn fit_to_viewport(&mut self, viewport: egui::Rect) {
        self.fit_to_rect(viewport);
    }

    fn available_object_selection_targets(
        &self,
    ) -> Vec<(
        crate::spatialdata::PositiveCellSelectionTarget,
        String,
    )> {
        let mut targets = Vec::new();
        if self.seg_objects.has_data() {
            targets.push((
                crate::spatialdata::PositiveCellSelectionTarget::SegmentationObjects,
                "Segmentation Objects".to_string(),
            ));
        }
        targets.extend(self.spatial_layers.positive_cell_selection_targets());
        targets
    }

    fn select_objects_by_ids_target(
        &mut self,
        cell_ids: &[String],
        target: crate::spatialdata::PositiveCellSelectionTarget,
    ) -> Option<(usize, usize)> {
        let id_set = cell_ids.iter().cloned().collect::<HashSet<_>>();
        if id_set.is_empty() {
            return None;
        }

        let mut matched_layers = 0usize;
        let mut matched_objects = 0usize;
        match target {
            crate::spatialdata::PositiveCellSelectionTarget::SegmentationObjects => {
                let selected = self.seg_objects.select_objects_by_ids(&id_set);
                if selected > 0 {
                    matched_layers += 1;
                    matched_objects += selected;
                }
            }
            crate::spatialdata::PositiveCellSelectionTarget::AllObjectLayers => {
                let selected = self.seg_objects.select_objects_by_ids(&id_set);
                if selected > 0 {
                    matched_layers += 1;
                    matched_objects += selected;
                }
                if let Some((layers, objects)) = self
                    .spatial_layers
                    .select_positive_cells_by_ids(cell_ids, target)
                {
                    matched_layers += layers;
                    matched_objects += objects;
                }
            }
            crate::spatialdata::PositiveCellSelectionTarget::ShapeLayer(_) => {
                if let Some((layers, objects)) = self
                    .spatial_layers
                    .select_positive_cells_by_ids(cell_ids, target)
                {
                    matched_layers += layers;
                    matched_objects += objects;
                }
            }
        }

        (matched_layers > 0).then_some((matched_layers, matched_objects))
    }

    fn fit_to_selected_seg_objects(&mut self) -> bool {
        let Some(viewport) = self.last_canvas_rect else {
            return false;
        };
        let off = self.layer_offset_world(LayerId::SegmentationObjects);
        let Some(world) = self.seg_objects.fit_bounds_world(off) else {
            return false;
        };
        self.camera.fit_to_world_rect(viewport, world);
        true
    }

    fn fit_to_seg_object_index(&mut self, object_index: usize) -> bool {
        let Some(viewport) = self.last_canvas_rect else {
            return false;
        };
        let off = self.layer_offset_world(LayerId::SegmentationObjects);
        let Some(world) = self.seg_objects.fit_object_bounds_world(object_index, off) else {
            return false;
        };
        self.camera.fit_to_world_rect(viewport, world);
        true
    }

    fn fit_to_rect(&mut self, viewport: egui::Rect) {
        let world = self.image_world_rect_lvl0();
        self.camera.fit_to_world_rect(viewport, world);
    }

    fn choose_level(&self) -> usize {
        if !self.auto_level {
            return self
                .manual_level
                .min(self.dataset.levels.len().saturating_sub(1));
        }
        choose_level_auto(
            &self.dataset.levels,
            self.camera.zoom_screen_per_lvl0_px,
            1.0,
        )
    }

    fn sort_tile_keys_near_center(&self, level_info: &crate::data::ome::LevelInfo, keys: &mut [TileKey]) {
        // Request the tiles nearest the current viewport center first so zoom-in refines where the
        // user is looking before spending bandwidth on peripheral tiles.
        let y_dim = self.dataset.dims.y;
        let x_dim = self.dataset.dims.x;
        let center_world = self.camera.center_world_lvl0;
        let center_local = self.primary_image_world_to_local(center_world);
        let downsample = level_info.downsample;
        let center_lvl = egui::pos2(center_local.x / downsample, center_local.y / downsample);
        let chunk_y = level_info.chunks[y_dim] as f32;
        let chunk_x = level_info.chunks[x_dim] as f32;
        let _ = x_dim;

        keys.sort_by(|a, b| {
            let ay = (a.tile_y as f32 + 0.5) * chunk_y;
            let ax = (a.tile_x as f32 + 0.5) * chunk_x;
            let by = (b.tile_y as f32 + 0.5) * chunk_y;
            let bx = (b.tile_x as f32 + 0.5) * chunk_x;
            let da = (ax - center_lvl.x).powi(2) + (ay - center_lvl.y).powi(2);
            let db = (bx - center_lvl.x).powi(2) + (by - center_lvl.y).powi(2);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    fn schedule_repaint(&self, ctx: &egui::Context) {
        if self.is_loading_scene() {
            repaint_control::request_repaint_busy(ctx);
            return;
        }

        if self.has_pending_async_ui_work() {
            ctx.request_repaint_after(Duration::from_millis(50));
        }
    }

    fn has_pending_async_ui_work(&self) -> bool {
        let properties_hist_active = self.show_right_panel
            && self.right_tab == RightTab::Properties
            && matches!(self.active_layer, LayerId::Channel(_));
        (properties_hist_active && (self.hist_dirty || self.hist_request_pending))
            || self.chanmax_pending.iter().any(|pending| *pending)
            || self.screenshot_in_flight.is_some()
            || self
                .annotation_layers
                .iter()
                .any(|layer| layer.has_pending_work())
    }

    fn request_tiles_with_budget(
        &mut self,
        level: usize,
        needed: &[TileKey],
        sent: &mut usize,
        max_per_frame: usize,
    ) {
        if *sent >= max_per_frame {
            return;
        }

        let channels = self.render_channels_for_request(level);
        for key in needed {
            if *sent >= max_per_frame {
                break;
            }
            if !self.cache.mark_in_flight(*key) {
                continue;
            }
            if let Some(level_info) = self.dataset.levels.get(level) {
                if let Some(tile) = self.pinned_levels.try_get_composited_tile(
                    *key,
                    &channels,
                    &self.dataset.dims,
                    level_info,
                ) {
                    self.cache.cancel_in_flight(key);
                    self.pending.push(tile);
                    *sent += 1;
                    continue;
                }
                if let Some(tile) =
                    self.try_get_composited_tile_from_pinned_finer(*key, &channels, level_info)
                {
                    self.cache.cancel_in_flight(key);
                    self.pending.push(tile);
                    *sent += 1;
                    continue;
                }
            }
            let _ = self.loader.tx.send(TileRequest {
                key: *key,
                channels: channels.clone(),
            });
            *sent += 1;
        }
    }

    fn request_raw_tiles_with_budget(
        &mut self,
        tiles_gl: &TilesGl,
        raw_tx: &crossbeam_channel::Sender<RawTileRequest>,
        level: usize,
        needed: &[TileKey],
        render_channels: &[RenderChannel],
        sent: &mut usize,
        max_per_frame: usize,
    ) {
        if *sent >= max_per_frame {
            return;
        }

        for key in needed {
            if *sent >= max_per_frame {
                break;
            }
            for ch in render_channels {
                if *sent >= max_per_frame {
                    break;
                }
                let raw_key = RawTileKey {
                    level,
                    tile_y: key.tile_y,
                    tile_x: key.tile_x,
                    channel: ch.index,
                };
                if !tiles_gl.mark_in_flight(raw_key) {
                    continue;
                }
                if let Some(level_info) = self.dataset.levels.get(level) {
                    if let Some(resp) =
                        self.pinned_levels
                            .try_get_raw_tile(raw_key, &self.dataset.dims, level_info)
                    {
                        tiles_gl.insert_pending(resp);
                        *sent += 1;
                        continue;
                    }
                    if let Some(resp) = self.try_get_raw_tile_from_pinned_finer(raw_key, level_info)
                    {
                        tiles_gl.insert_pending(resp);
                        *sent += 1;
                        continue;
                    }
                }
                let _ = raw_tx.send(RawTileRequest { key: raw_key });
                *sent += 1;
            }
        }
    }

    fn prefetch_spec(&self, visible_count: usize) -> Option<(i64, usize)> {
        match self.tile_prefetch_mode {
            TilePrefetchMode::Off => None,
            TilePrefetchMode::TargetHalo | TilePrefetchMode::TargetAndFinerHalo => {
                let (small_pad, small_budget, medium_pad, medium_budget) =
                    match self.tile_prefetch_aggressiveness {
                        TilePrefetchAggressiveness::Conservative => (1, 16usize, 1, 8usize),
                        TilePrefetchAggressiveness::Balanced => (2, 48usize, 1, 24usize),
                        TilePrefetchAggressiveness::Aggressive => (2, 96usize, 2, 48usize),
                    };
                if visible_count <= 16 {
                    Some((small_pad, small_budget))
                } else if visible_count <= 48 {
                    Some((medium_pad, medium_budget))
                } else {
                    None
                }
            }
        }
    }

    fn prefetch_keys_for_level(
        &self,
        level: usize,
        level_info: &crate::data::ome::LevelInfo,
        visible_world_tiles: egui::Rect,
        visible_needed: &[TileKey],
    ) -> Vec<TileKey> {
        let visible_count = visible_needed.len();
        let Some((pad_tiles, prefetch_budget)) = self.prefetch_spec(visible_count) else {
            return Vec::new();
        };

        // Prefetch only a halo around the already-visible set. This keeps ahead of short pans
        // without letting speculative IO dominate the queue when the viewport is large.
        let visible_set: HashSet<(u64, u64)> = visible_needed
            .iter()
            .map(|key| (key.tile_y, key.tile_x))
            .collect();
        let mut prefetch: Vec<TileKey> = tiles_needed_lvl0_rect(
            visible_world_tiles,
            level_info,
            &self.dataset.dims,
            pad_tiles,
        )
        .into_iter()
        .filter_map(|coord| {
            (!visible_set.contains(&(coord.tile_y, coord.tile_x))).then_some(TileKey {
                render_id: self.active_render_id,
                level,
                tile_y: coord.tile_y,
                tile_x: coord.tile_x,
            })
        })
        .collect();
        self.sort_tile_keys_near_center(level_info, &mut prefetch);
        prefetch.truncate(prefetch_budget);
        prefetch
    }

    fn render_channels_for_request(&self, _level: usize) -> Vec<RenderChannel> {
        let mut out = Vec::new();
        let groups = self.project_space.layer_groups();
        let order = if self.channel_layer_order.len() == self.channels.len() {
            self.channel_layer_order.clone()
        } else {
            (0..self.channels.len()).collect()
        };

        for idx in order {
            let Some(ch) = self.channels.get(idx) else {
                continue;
            };
            if !ch.visible {
                continue;
            }
            let rgb =
                layer_groups::effective_channel_color_rgb(groups, ch.name.as_str(), ch.color_rgb);
            out.push(RenderChannel {
                index: ch.index as u64,
                color_rgb: [
                    rgb[0] as f32 / 255.0,
                    rgb[1] as f32 / 255.0,
                    rgb[2] as f32 / 255.0,
                ],
                window: ch.window.unwrap_or((0.0, 65535.0)),
            });
        }
        out
    }

    fn tile_rects(
        &self,
        key: &TileKey,
        viewport: egui::Rect,
        level_info: &crate::data::ome::LevelInfo,
    ) -> (egui::Rect, egui::Rect) {
        let y_dim = self.dataset.dims.y;
        let x_dim = self.dataset.dims.x;
        let chunk_y = level_info.chunks[y_dim] as f32;
        let chunk_x = level_info.chunks[x_dim] as f32;

        let y0 = key.tile_y as f32 * chunk_y;
        let x0 = key.tile_x as f32 * chunk_x;
        let y1 = (y0 + chunk_y).min(level_info.shape[y_dim] as f32);
        let x1 = (x0 + chunk_x).min(level_info.shape[x_dim] as f32);

        let downsample = level_info.downsample;
        let world_min =
            self.primary_image_local_to_world(egui::pos2(x0 * downsample, y0 * downsample));
        let world_max =
            self.primary_image_local_to_world(egui::pos2(x1 * downsample, y1 * downsample));
        let world_rect = egui::Rect::from_min_max(world_min, world_max);

        let screen_min = self.camera.world_to_screen(world_min, viewport);
        let screen_max = self.camera.world_to_screen(world_max, viewport);
        let screen_rect = egui::Rect::from_min_max(screen_min, screen_max);

        (world_rect, screen_rect)
    }

    fn get_tile_texture(&mut self, key: &TileKey) -> Option<egui::TextureHandle> {
        if let Some(tex) = self.cache.get(key).cloned() {
            return Some(tex);
        }

        if let Some(prev) = self.previous_render_id {
            let prev_key = TileKey {
                render_id: prev,
                level: key.level,
                tile_y: key.tile_y,
                tile_x: key.tile_x,
            };
            if let Some(tex) = self.cache.get(&prev_key).cloned() {
                return Some(tex);
            }
        }

        None
    }

    fn bump_render_id(&mut self) {
        let new_id = self.compute_render_id();
        if new_id != self.active_render_id {
            self.previous_render_id = Some(self.active_render_id);
            self.active_render_id = new_id;
        }
    }

    fn compute_render_id(&self) -> u64 {
        use std::hash::{Hash, Hasher};

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.dataset.source.hash(&mut hasher);
        self.channel_layer_order.hash(&mut hasher);
        let groups = self.project_space.layer_groups();
        for &idx in &self.channel_layer_order {
            if let Some(ch) = self.channels.get(idx) {
                ch.index.hash(&mut hasher);
                ch.visible.hash(&mut hasher);
                let rgb = layer_groups::effective_channel_color_rgb(
                    groups,
                    ch.name.as_str(),
                    ch.color_rgb,
                );
                rgb.hash(&mut hasher);
                let (w0, w1) = ch.window.unwrap_or((0.0, 65535.0));
                w0.to_bits().hash(&mut hasher);
                w1.to_bits().hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    fn drain_histogram(&mut self) {
        while let Ok(msg) = self.hist_loader.rx.try_recv() {
            if msg.request_id != self.hist_request_id {
                continue;
            }
            self.hist_request_pending = false;
            self.hist = Some(msg);
        }
    }

    fn maybe_request_histogram(&mut self, ctx: &egui::Context) {
        if !self.hist_dirty {
            return;
        }
        if self.channels.is_empty() {
            self.hist_dirty = false;
            self.hist_request_pending = false;
            self.hist = None;
            return;
        }
        let Some(viewport) = self.last_canvas_rect else {
            return;
        };
        let elapsed = Instant::now().duration_since(self.hist_last_sent);
        let throttle = Duration::from_millis(200);
        if elapsed < throttle {
            ctx.request_repaint_after(throttle - elapsed);
            return;
        }

        let level = self.dataset.levels.len().saturating_sub(1);
        let level_info = &self.dataset.levels[level];
        let downsample = level_info.downsample.max(1.0);

        let visible_world = self.visible_world_rect(viewport);
        let mut y0 = (visible_world.min.y / downsample).floor().max(0.0) as u64;
        let mut y1 = (visible_world.max.y / downsample).ceil().max(0.0) as u64;
        let mut x0 = (visible_world.min.x / downsample).floor().max(0.0) as u64;
        let mut x1 = (visible_world.max.x / downsample).ceil().max(0.0) as u64;

        let y_dim = self.dataset.dims.y;
        let x_dim = self.dataset.dims.x;
        let shape_y = *level_info.shape.get(y_dim).unwrap_or(&0);
        let shape_x = *level_info.shape.get(x_dim).unwrap_or(&0);
        y0 = y0.min(shape_y);
        y1 = y1.min(shape_y).max(y0);
        x0 = x0.min(shape_x);
        x1 = x1.min(shape_x).max(x0);

        // Hard cap the sampled area to keep it responsive.
        let max_dim = 1024u64;
        if y1.saturating_sub(y0) > max_dim {
            let cy = (y0 + y1) / 2;
            y0 = cy.saturating_sub(max_dim / 2);
            y1 = (y0 + max_dim).min(shape_y);
        }
        if x1.saturating_sub(x0) > max_dim {
            let cx = (x0 + x1) / 2;
            x0 = cx.saturating_sub(max_dim / 2);
            x1 = (x0 + max_dim).min(shape_x);
        }

        self.hist_request_id = self.hist_request_id.wrapping_add(1);
        let req = crate::imaging::histogram::HistogramRequest {
            request_id: self.hist_request_id,
            level,
            channel: self.selected_channel as u64,
            y0,
            y1,
            x0,
            x1,
            bins: 256,
            abs_max: self.dataset.abs_max.max(1.0),
        };
        let _ = self.hist_loader.tx.send(req);
        self.hist_last_sent = Instant::now();
        self.hist_request_pending = true;
        self.hist_dirty = false;
    }

    fn drain_tiles(&mut self, ctx: &egui::Context) {
        while let Ok(msg) = self.loader.rx.try_recv() {
            match msg {
                TileWorkerResponse::Tile(msg) => {
                    self.cache.cancel_in_flight(&msg.key);

                    // Loader responses can outlive the frame that requested them. Accept the
                    // current render epoch and, briefly, the immediately previous one so the draw
                    // path can finish a coarse->fine transition without showing obviously stale
                    // tiles from unrelated dataset/tool states.
                    if msg.key.render_id != self.active_render_id
                        && self.previous_render_id != Some(msg.key.render_id)
                    {
                        continue;
                    }
                    self.pending.push(msg);
                }
                TileWorkerResponse::Failed { key, error } => {
                    self.cache.cancel_in_flight(&key);
                    crate::log_warn!("tile load failed for {:?}: {}", key, error);
                    ctx.request_repaint_after(Duration::from_millis(100));
                }
            }
        }

        if self.pending.is_empty() {
            return;
        }

        for TileResponse {
            key,
            width,
            height,
            rgba,
        } in self.pending.drain(..)
        {
            let image = egui::ColorImage::from_rgba_unmultiplied([width, height], &rgba);
            let options = if self.smooth_pixels {
                egui::TextureOptions::LINEAR
            } else {
                egui::TextureOptions::NEAREST
            };
            let tex = ctx.load_texture(
                format!(
                    "tile-{}-{}-{}-{}",
                    key.render_id, key.level, key.tile_y, key.tile_x
                ),
                image,
                options,
            );
            self.cache.put(key, tex);
        }
    }

    fn drain_raw_tiles(&mut self) {
        let (Some(loader), Some(tiles_gl)) = (self.raw_loader.as_ref(), self.tiles_gl.as_ref())
        else {
            return;
        };
        while let Ok(msg) = loader.rx.try_recv() {
            match msg {
                RawTileWorkerResponse::Tile(msg) => tiles_gl.insert_pending(msg),
                RawTileWorkerResponse::Failed { key, error } => {
                    tiles_gl.cancel_in_flight(&key);
                    crate::log_warn!("raw tile load failed for {:?}: {}", key, error);
                }
            }
        }
    }

    fn drain_label_tiles(&mut self) {
        let (Some(loader), Some(labels_gl)) = (self.label_loader.as_ref(), self.labels_gl.as_ref())
        else {
            return;
        };
        while let Ok(msg) = loader.rx.try_recv() {
            labels_gl.insert_pending(msg);
        }
    }

    fn drain_screenshots(&mut self) {
        while let Ok(resp) = self.screenshot_worker.rx.try_recv() {
            match resp {
                crate::app_support::screenshot::ScreenshotWorkerResp::Saved { id, path, result } => {
                    if self.screenshot_in_flight == Some(id) {
                        self.screenshot_in_flight = None;
                    }
                    match result {
                        Ok(()) => {
                            self.set_status(format!(
                                "Saved screenshot -> {}",
                                path.to_string_lossy()
                            ));
                        }
                        Err(err) => {
                            self.set_status(format!("Save screenshot failed: {err}"));
                        }
                    }
                }
            }
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
            approx_downsample: mapped_scale_x.max(mapped_scale_y).max(1e-6),
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
