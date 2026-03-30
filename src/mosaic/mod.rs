mod io;
mod segmentation_geojson;
mod tiles_gl;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use eframe::egui;
use glow::HasContext;
use rfd::FileDialog;

use crate::annotations::{AnnotationCategoryStyle, AnnotationPointsLayer, AnnotationShape};
use crate::app::S3DatasetSelection;
use crate::camera::Camera;
use crate::data::dataset_source::DatasetSource;
use crate::ui::icons::Icon;
use crate::project::groups as layer_groups;
use crate::app_support::memory::{
    MemoryChannelRow, MemoryRisk, MemoryRiskLevel, PendingMemoryAction, SystemMemorySnapshot,
    format_bytes, memory_risk, refresh_system_memory_if_needed, ui_memory_channel_selector,
    ui_memory_overview, ui_pending_memory_action_dialog,
};
use self::io::{
    MosaicPinnedLevelStatus, MosaicPinnedLevels, MosaicRawTileKey, MosaicRawTileLoaderHandle,
    MosaicRawTileRequest, MosaicRawTileWorkerResponse, MosaicSource,
    estimate_level_ram_bytes_for_channels, spawn_mosaic_raw_tile_loader,
};
use self::segmentation_geojson::MosaicGeoJsonSegmentationOverlay;
use self::tiles_gl::{ChannelDraw, MosaicTileDraw, MosaicTilesGl};
use crate::data::ome::OmeZarrDataset;
use crate::data::project_config::{ProjectLayerGroups, ProjectRoi};
use crate::project::{
    ProjectAnnotationCategoryStyleState, ProjectAnnotationLayerState, ProjectCameraState,
    ProjectChannelViewState, ProjectMosaicViewState, ProjectSpace, ProjectUiState,
};
use crate::data::remote_store::{build_http_store, build_s3_store};
use crate::app_support::repaint as repaint_control;
use crate::data::samplesheet::load_samplesheet_csv;
use crate::app_support::screenshot::{
    ScreenshotRequest, ScreenshotSettings, ScreenshotWorkerHandle, ScreenshotWorkerMsg,
    next_numbered_screenshot_path,
};
use crate::imaging::tiling::{TileCoord, choose_level_auto, tiles_needed_lvl0_rect};
use crate::ui::canvas_overlays;
use crate::ui::channels_panel::{self, ChannelListHost};
use crate::ui::contrast;
use crate::ui::group_layers::{GroupLayersDialog, GroupLayersTarget, default_group_name};
use crate::ui::layer_list;
use crate::ui::left_panel;
use crate::ui::right_panel;
use crate::ui::style::apply_napari_like_dark;
use crate::ui::top_bar;

// Mosaic viewer shell.
//
// This file owns the multi-ROI view: layout/sort/group state, shared channel controls, coarse-
// to-fine tile refinement across many items, and mosaic-only overlays such as text labels and
// grouped annotations. Lower-level tile loaders and GL code stay elsewhere; this module decides
// what should be requested, drawn, or kept focused from frame to frame.

#[derive(Debug, Clone)]
pub struct MosaicCliArgs {
    pub dataset_names: Vec<String>,
    pub columns: Option<usize>,
    pub samplesheet_csv: Option<PathBuf>,
    pub project_path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct GlobalChannel {
    name: String,
    color_rgb: [u8; 3],
    window: Option<(f32, f32)>,
    visible: bool,
}

#[derive(Debug, Clone)]
struct MosaicItem {
    id: usize,
    sample_id: String,
    meta: HashMap<String, String>,
    dataset: OmeZarrDataset,
    offset: egui::Vec2,
    scale: f32,
    placed_size: egui::Vec2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum MosaicLayerId {
    TextLabels,
    SegmentationGeoJson,
    Annotation(u64),
    Channel(usize),
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
    Layout,
    Memory,
}

impl RightTab {
    fn storage_key(self) -> &'static str {
        match self {
            Self::Properties => "properties",
            Self::Layout => "layout",
            Self::Memory => "memory",
        }
    }

    fn from_storage_key(value: &str) -> Option<Self> {
        match value {
            "properties" => Some(Self::Properties),
            "layout" => Some(Self::Layout),
            "memory" => Some(Self::Memory),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MosaicLayoutMode {
    FitCells,
    NativePixels,
}

impl MosaicLayoutMode {
    fn label(self) -> &'static str {
        match self {
            Self::FitCells => "Fit cells",
            Self::NativePixels => "Plot to scale",
        }
    }

    fn storage_key(self) -> &'static str {
        match self {
            Self::FitCells => "fit_cells",
            Self::NativePixels => "native_pixels",
        }
    }

    fn from_storage_key(value: &str) -> Option<Self> {
        match value {
            "fit_cells" => Some(Self::FitCells),
            "native_pixels" => Some(Self::NativePixels),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
struct PendingMemoryLoadRequest {
    dataset_id: usize,
    source: MosaicSource,
    level: usize,
    selected_global_channels: Vec<u64>,
}

pub struct MosaicViewerApp {
    items: Vec<MosaicItem>,
    sources: Arc<Vec<MosaicSource>>,
    pinned_levels: MosaicPinnedLevels,
    loader: MosaicRawTileLoaderHandle,
    tiles_gl: MosaicTilesGl,
    remote_runtimes: Vec<Arc<tokio::runtime::Runtime>>,

    camera: Camera,
    last_canvas_rect: Option<egui::Rect>,
    mosaic_bounds: egui::Rect,

    focused_core_id: Option<usize>,

    abs_max: f32,
    channels: Vec<GlobalChannel>,
    selected_channel: usize,
    channel_list_search: String,
    active_layer: MosaicLayerId,
    selected_channel_layers: HashSet<usize>,
    memory_selected_channels: HashSet<usize>,
    channel_select_anchor_idx: Option<usize>,
    selected_channel_group_id: Option<u64>,
    selected_overlay_layers: HashSet<MosaicLayerId>,
    overlay_select_anchor_pos: Option<usize>,
    overlay_layer_order: Vec<MosaicLayerId>,
    channel_layer_order: Vec<usize>,
    annotation_layers: Vec<AnnotationPointsLayer>,
    next_annotation_layer_id: u64,
    last_target_level_by_dataset_id: Vec<Option<usize>>,
    fallback_ceiling_by_dataset_id: Vec<Option<usize>>,
    zoom_out_floor_by_dataset_id: Vec<Option<usize>>,
    zoom_out_floor_until_by_dataset_id: Vec<Option<Instant>>,
    zoom_out_floor_world_by_dataset_id: Vec<Option<egui::Rect>>,
    last_visible_world: Option<egui::Rect>,
    layer_groups: ProjectLayerGroups,
    layer_drag: Option<layer_list::LayerDragState<MosaicLayerId>>,
    left_tab: LeftTab,
    right_tab: RightTab,
    metadata_columns: Vec<String>,
    sort_by: String,
    sort_secondary_enabled: bool,
    sort_by_secondary: String,
    group_by: String,
    show_group_labels: bool,
    group_gap: f32,
    layout_mode: MosaicLayoutMode,
    group_blocks: Vec<GroupBlock>,
    show_text_labels: bool,
    label_columns: Vec<String>,
    grid_cols: usize,
    grid_cell_w: f32,
    grid_cell_h: f32,
    grid_pad: f32,
    show_left_panel: bool,
    show_right_panel: bool,
    close_dialog_open: bool,
    system_memory: Option<SystemMemorySnapshot>,
    system_memory_last_refresh: Option<Instant>,
    pending_memory_load: Option<PendingMemoryAction<Vec<PendingMemoryLoadRequest>>>,
    tile_request_generation: u64,
    last_tile_request_signature: Option<TileRequestSignature>,

    status: String,
    allow_back: bool,
    pending_request: Option<MosaicRequest>,
    group_layers_dialog: Option<GroupLayersDialog>,
    smooth_pixels: bool,
    show_tile_debug: bool,
    screenshot_settings: ScreenshotSettings,
    screenshot_settings_open: bool,
    screenshot_worker: ScreenshotWorkerHandle,
    screenshot_next_id: u64,
    screenshot_pending: Option<ScreenshotRequest>,
    screenshot_in_flight: Option<u64>,
    screenshot_output_dir: Option<PathBuf>,
    seg_geojson: MosaicGeoJsonSegmentationOverlay,
    seg_geojson_pending_visible: bool,
    project_space: ProjectSpace,
}

impl ChannelListHost for MosaicViewerApp {
    type LayerId = MosaicLayerId;

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
        self.set_layer_visible(MosaicLayerId::Channel(idx), visible);
    }

    fn channel_available(&self, idx: usize) -> bool {
        self.layer_available(MosaicLayerId::Channel(idx))
    }

    fn is_channel_selected(&self, idx: usize) -> bool {
        self.active_layer == MosaicLayerId::Channel(idx)
            || self.selected_channel_layers.contains(&idx)
    }

    fn selected_channel_group_id(&self) -> Option<u64> {
        self.selected_channel_group_id
    }

    fn select_channel_group(&mut self, group_id: Option<u64>) {
        self.selected_channel_group_id = group_id;
        self.selected_channel_layers.clear();
        if let Some(gid) = group_id {
            if let Some(idx) = self.channel_indices_in_group(gid).into_iter().next() {
                self.set_active_layer(MosaicLayerId::Channel(idx));
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
        self.set_active_layer(MosaicLayerId::Channel(idx));
    }

    fn handle_channel_secondary_click(&mut self, idx: usize) {
        if !self.selected_channel_layers.contains(&idx) {
            self.selected_channel_layers.clear();
            self.selected_channel_layers.insert(idx);
            self.channel_select_anchor_idx = Some(idx);
            self.selected_channel_group_id = None;
            self.set_active_layer(MosaicLayerId::Channel(idx));
        }
    }

    fn open_group_layers_dialog_channels(&mut self, members: Vec<usize>) {
        Self::open_group_layers_dialog_channels(self, members);
    }

    fn layer_groups(&self) -> ProjectLayerGroups {
        self.layer_groups.clone()
    }

    fn set_layer_groups(&mut self, groups: ProjectLayerGroups) {
        self.layer_groups = groups;
    }

    fn channels_changed(&mut self) {}

    fn layer_drag_mut(&mut self) -> &mut Option<layer_list::LayerDragState<Self::LayerId>> {
        &mut self.layer_drag
    }

    fn dragging_channel_idx(&self) -> Option<usize> {
        self.layer_drag.as_ref().and_then(|drag| {
            if drag.group != layer_list::LayerGroup::Channels {
                return None;
            }
            match drag.dragged {
                MosaicLayerId::Channel(idx) => Some(idx),
                _ => None,
            }
        })
    }

    fn channel_layer_id(&self, idx: usize) -> Self::LayerId {
        MosaicLayerId::Channel(idx)
    }
}

#[derive(Debug, Clone)]
pub enum MosaicRequest {
    BackToSingle,
    OpenSingle(PathBuf),
    OpenMosaic(Vec<PathBuf>),
    OpenProjectRoi(ProjectRoi),
    OpenProjectMosaic(Vec<ProjectRoi>),
    OpenRemoteDialog,
}

#[derive(Debug, Clone)]
struct GroupBlock {
    name: String,
    world_rect: egui::Rect,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TileRequestSignature {
    viewport_width_bits: u32,
    viewport_height_bits: u32,
    visible_world_min_x_bits: u32,
    visible_world_min_y_bits: u32,
    visible_world_max_x_bits: u32,
    visible_world_max_y_bits: u32,
    visible_channels: Vec<u64>,
}

impl MosaicViewerApp {
    fn layer_id_storage_key(id: MosaicLayerId) -> String {
        match id {
            MosaicLayerId::TextLabels => "text_labels".to_string(),
            MosaicLayerId::SegmentationGeoJson => "segmentation_geojson".to_string(),
            MosaicLayerId::Annotation(id) => format!("annotation:{id}"),
            MosaicLayerId::Channel(idx) => format!("channel:{idx}"),
        }
    }

    fn parse_layer_id_storage_key(&self, value: &str) -> Option<MosaicLayerId> {
        if let Some(raw) = value.strip_prefix("annotation:") {
            return raw.parse::<u64>().ok().map(MosaicLayerId::Annotation);
        }
        if let Some(raw) = value.strip_prefix("channel:") {
            return raw.parse::<usize>().ok().map(MosaicLayerId::Channel);
        }
        match value {
            "text_labels" => Some(MosaicLayerId::TextLabels),
            "segmentation_geojson" => Some(MosaicLayerId::SegmentationGeoJson),
            _ => None,
        }
    }

    fn project_camera_state(&self) -> ProjectCameraState {
        ProjectCameraState {
            center_world_lvl0: [self.camera.center_world_lvl0.x, self.camera.center_world_lvl0.y],
            zoom_screen_per_lvl0_px: self.camera.zoom_screen_per_lvl0_px,
        }
    }

    fn apply_project_camera_state(&mut self, state: &ProjectCameraState) {
        self.camera.center_world_lvl0 =
            egui::pos2(state.center_world_lvl0[0], state.center_world_lvl0[1]);
        if state.zoom_screen_per_lvl0_px.is_finite() && state.zoom_screen_per_lvl0_px > 0.0 {
            self.camera.zoom_screen_per_lvl0_px = state.zoom_screen_per_lvl0_px;
        }
    }

    fn project_ui_state(&self) -> ProjectUiState {
        ProjectUiState {
            show_left_panel: Some(self.show_left_panel),
            show_right_panel: Some(self.show_right_panel),
            left_tab: Some(self.left_tab.storage_key().to_string()),
            right_tab: Some(self.right_tab.storage_key().to_string()),
            smooth_pixels: Some(self.smooth_pixels),
            show_tile_debug: Some(self.show_tile_debug),
            show_scale_bar: None,
            auto_level: None,
            manual_level: None,
        }
    }

    fn project_path_string(&self, path: &Path) -> String {
        if let Some(project_dir) = self.project_space.project_dir()
            && let Ok(relative) = path.strip_prefix(&project_dir)
        {
            return relative.to_string_lossy().to_string();
        }
        path.to_string_lossy().to_string()
    }

    fn resolve_project_path(&self, path: &str) -> PathBuf {
        let path_buf = PathBuf::from(path);
        if path_buf.is_absolute() {
            path_buf
        } else {
            self.project_space
                .project_dir()
                .map(|dir| dir.join(&path_buf))
                .unwrap_or(path_buf)
        }
    }

    fn project_annotation_layer_state(
        &self,
        layer: &AnnotationPointsLayer,
    ) -> ProjectAnnotationLayerState {
        ProjectAnnotationLayerState {
            id: layer.id,
            name: layer.name.clone(),
            visible: layer.visible,
            radius_screen_px: layer.style.radius_screen_px,
            opacity: layer.style.opacity,
            stroke_width: layer.style.stroke.width,
            stroke_color_rgb: [
                layer.style.stroke.color.r(),
                layer.style.stroke.color.g(),
                layer.style.stroke.color.b(),
            ],
            stroke_color_alpha: layer.style.stroke.color.a(),
            offset_world: [layer.offset_world.x, layer.offset_world.y],
            parquet_path: layer
                .parquet
                .path
                .as_deref()
                .map(|path| self.project_path_string(path)),
            roi_id_column: layer.parquet.roi_id_column.clone(),
            x_column: layer.parquet.x_column.clone(),
            y_column: layer.parquet.y_column.clone(),
            value_column: layer.parquet.value_column.clone(),
            selected_value_column: layer.selected_value_column.clone(),
            category_styles: layer
                .category_styles
                .iter()
                .map(|style| ProjectAnnotationCategoryStyleState {
                    name: style.name.clone(),
                    visible: style.visible,
                    color_rgb: [style.color.r(), style.color.g(), style.color.b()],
                    shape: style.shape.storage_key().to_string(),
                })
                .collect(),
            continuous_shape: Some(layer.continuous_shape.storage_key().to_string()),
            continuous_range: layer.continuous_range.map(|(lo, hi)| [lo, hi]),
        }
    }

    fn restore_annotation_layers(&mut self, layers: &[ProjectAnnotationLayerState]) {
        self.annotation_layers.clear();
        self.next_annotation_layer_id = 1;
        for saved in layers {
            let mut layer = AnnotationPointsLayer::new(saved.id, saved.name.clone());
            layer.visible = saved.visible;
            layer.style.radius_screen_px = saved.radius_screen_px;
            layer.style.opacity = saved.opacity;
            layer.style.stroke.width = saved.stroke_width;
            layer.style.stroke.color = egui::Color32::from_rgba_unmultiplied(
                saved.stroke_color_rgb[0],
                saved.stroke_color_rgb[1],
                saved.stroke_color_rgb[2],
                saved.stroke_color_alpha,
            );
            layer.offset_world = egui::vec2(saved.offset_world[0], saved.offset_world[1]);
            layer.parquet.path = saved
                .parquet_path
                .as_deref()
                .map(|path| self.resolve_project_path(path));
            layer.parquet.roi_id_column = saved.roi_id_column.clone();
            layer.parquet.x_column = saved.x_column.clone();
            layer.parquet.y_column = saved.y_column.clone();
            layer.parquet.value_column = saved.value_column.clone();
            layer.selected_value_column = saved.selected_value_column.clone();
            layer.category_styles = saved
                .category_styles
                .iter()
                .map(|style| AnnotationCategoryStyle {
                    name: style.name.clone(),
                    visible: style.visible,
                    color: egui::Color32::from_rgb(
                        style.color_rgb[0],
                        style.color_rgb[1],
                        style.color_rgb[2],
                    ),
                    shape: AnnotationShape::from_storage_key(&style.shape)
                        .unwrap_or(AnnotationShape::Circle),
                })
                .collect();
            if let Some(shape) = saved
                .continuous_shape
                .as_deref()
                .and_then(AnnotationShape::from_storage_key)
            {
                layer.continuous_shape = shape;
            }
            layer.continuous_range = saved.continuous_range.map(|[lo, hi]| (lo, hi));
            if layer.parquet.path.is_some() {
                layer.request_schema_load();
                layer.request_load();
            }
            self.next_annotation_layer_id = self.next_annotation_layer_id.max(saved.id + 1);
            self.annotation_layers.push(layer);
        }
    }

    fn apply_project_ui_state(&mut self, state: &ProjectUiState) {
        if let Some(show_left_panel) = state.show_left_panel {
            self.show_left_panel = show_left_panel;
        }
        if let Some(show_right_panel) = state.show_right_panel {
            self.show_right_panel = show_right_panel;
        }
        if let Some(left_tab) = state.left_tab.as_deref().and_then(LeftTab::from_storage_key) {
            self.left_tab = left_tab;
        }
        if let Some(right_tab) = state
            .right_tab
            .as_deref()
            .and_then(RightTab::from_storage_key)
        {
            self.right_tab = right_tab;
        }
        if let Some(smooth_pixels) = state.smooth_pixels {
            self.smooth_pixels = smooth_pixels;
            self.tiles_gl.set_smooth_pixels(self.smooth_pixels);
        }
        if let Some(show_tile_debug) = state.show_tile_debug {
            self.show_tile_debug = show_tile_debug;
        }
    }

    pub fn set_layer_groups(&mut self, groups: ProjectLayerGroups) {
        self.layer_groups = groups;
    }

    pub fn take_project_space(&mut self) -> ProjectSpace {
        self.project_space.set_mosaic_view_state(ProjectMosaicViewState {
            channel_order: self.channel_layer_order.clone(),
            channels: self
                .channels
                .iter()
                .map(|ch| ProjectChannelViewState {
                    visible: Some(ch.visible),
                    color_rgb: Some(ch.color_rgb),
                    window: ch.window.map(|(lo, hi)| [lo, hi]),
                    offset_world: None,
                    scale: None,
                    rotation_rad: None,
                })
                .collect(),
            active_channel: Some(self.selected_channel),
            active_layer: Some(Self::layer_id_storage_key(self.active_layer)),
            overlay_order: self
                .overlay_layer_order
                .iter()
                .copied()
                .map(Self::layer_id_storage_key)
                .collect(),
            overlay_visibility: self
                .overlay_layer_order
                .iter()
                .copied()
                .filter_map(|id| {
                    self.layer_visible_value(id)
                        .map(|visible| (Self::layer_id_storage_key(id), visible))
                })
                .collect::<BTreeMap<_, _>>(),
            sort_by: Some(self.sort_by.clone()),
            sort_secondary_enabled: Some(self.sort_secondary_enabled),
            sort_by_secondary: Some(self.sort_by_secondary.clone()),
            group_by: Some(self.group_by.clone()),
            show_group_labels: Some(self.show_group_labels),
            group_gap: Some(self.group_gap),
            layout_mode: Some(self.layout_mode.storage_key().to_string()),
            show_text_labels: Some(self.show_text_labels),
            label_columns: self.label_columns.clone(),
            camera: Some(self.project_camera_state()),
            ui: Some(self.project_ui_state()),
            annotation_layers: self
                .annotation_layers
                .iter()
                .map(|layer| self.project_annotation_layer_state(layer))
                .collect(),
        });
        self.project_space.update_layer_groups(|g| {
            *g = self.layer_groups.clone();
        });
        std::mem::take(&mut self.project_space)
    }

    pub fn project_space_mut(&mut self) -> &mut ProjectSpace {
        &mut self.project_space
    }

    pub fn set_project_space(&mut self, mut project_space: ProjectSpace) {
        self.layer_groups = project_space.layer_groups().clone();
        if let Some(view) = project_space.mosaic_view_state() {
            if let Some(ui) = view.ui.as_ref() {
                self.apply_project_ui_state(ui);
            }
            if !view.channel_order.is_empty() {
                let mut channel_order = view
                    .channel_order
                    .iter()
                    .copied()
                    .filter(|&idx| idx < self.channels.len())
                    .collect::<Vec<_>>();
                for idx in 0..self.channels.len() {
                    if !channel_order.contains(&idx) {
                        channel_order.push(idx);
                    }
                }
                self.channel_layer_order = channel_order;
            }
            for (idx, saved) in view.channels.iter().enumerate() {
                let Some(ch) = self.channels.get_mut(idx) else {
                    continue;
                };
                if let Some(visible) = saved.visible {
                    ch.visible = visible;
                }
                if let Some(color_rgb) = saved.color_rgb {
                    ch.color_rgb = color_rgb;
                }
                if let Some([lo, hi]) = saved.window {
                    ch.window = Some((lo, hi));
                }
            }
            if let Some(active_channel) = view.active_channel {
                self.selected_channel = active_channel.min(self.channels.len().saturating_sub(1));
            }
            if !view.overlay_order.is_empty() {
                let mut overlay_order = view
                    .overlay_order
                    .iter()
                    .filter_map(|id| self.parse_layer_id_storage_key(id))
                    .collect::<Vec<_>>();
                for id in self.overlay_layer_order.iter().copied() {
                    if !overlay_order.contains(&id) {
                        overlay_order.push(id);
                    }
                }
                self.overlay_layer_order = overlay_order;
            }
            if let Some(sort_by) = view.sort_by.as_ref() {
                self.sort_by = sort_by.clone();
            }
            if let Some(sort_secondary_enabled) = view.sort_secondary_enabled {
                self.sort_secondary_enabled = sort_secondary_enabled;
            }
            if let Some(sort_by_secondary) = view.sort_by_secondary.as_ref() {
                self.sort_by_secondary = sort_by_secondary.clone();
            }
            if let Some(group_by) = view.group_by.as_ref() {
                self.group_by = group_by.clone();
            }
            if let Some(show_group_labels) = view.show_group_labels {
                self.show_group_labels = show_group_labels;
            }
            if let Some(group_gap) = view.group_gap {
                self.group_gap = group_gap;
            }
            if let Some(layout_mode) = view.layout_mode.as_deref()
                && let Some(parsed) = MosaicLayoutMode::from_storage_key(layout_mode)
            {
                self.layout_mode = parsed;
            }
            if let Some(show_text_labels) = view.show_text_labels {
                self.show_text_labels = show_text_labels;
            }
            self.label_columns = view.label_columns.clone();
            self.apply_sort_and_layout();
            for (id, visible) in &view.overlay_visibility {
                if let Some(layer_id) = self.parse_layer_id_storage_key(id) {
                    self.set_layer_visible(layer_id, *visible);
                }
            }
            if let Some(camera) = view.camera.as_ref() {
                self.apply_project_camera_state(camera);
            }
            self.restore_annotation_layers(&view.annotation_layers);
            if let Some(active_layer) = view
                .active_layer
                .as_deref()
                .and_then(|id| self.parse_layer_id_storage_key(id))
            {
                self.set_active_layer(active_layer);
            } else if !self.channels.is_empty() {
                self.set_active_layer(MosaicLayerId::Channel(
                    self.selected_channel.min(self.channels.len().saturating_sub(1)),
                ));
            }
        }
        project_space.update_layer_groups(|g| {
            *g = self.layer_groups.clone();
        });
        self.project_space = project_space;
    }

    pub fn take_layer_groups(&mut self) -> ProjectLayerGroups {
        std::mem::take(&mut self.layer_groups)
    }

    pub fn layer_groups(&self) -> &ProjectLayerGroups {
        &self.layer_groups
    }

    pub fn take_request(&mut self) -> Option<MosaicRequest> {
        self.pending_request.take()
    }

    pub fn request_close_dialog(&mut self) {
        self.close_dialog_open = true;
    }

    pub fn set_status(&mut self, status: impl Into<String>) {
        self.status = status.into();
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
        self.screenshot_settings_open = false;
        self.status = "Capturing screenshot...".to_string();
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
        let base = self
            .focused_item()
            .or_else(|| self.items.first())
            .map(|it| it.sample_id.clone())
            .filter(|name| !name.trim().is_empty())
            .unwrap_or_else(|| "mosaic".to_string());
        let sanitized = base
            .chars()
            .map(|ch| match ch {
                '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
                _ => ch,
            })
            .collect::<String>();
        let sanitized = sanitized.trim().trim_matches('.').trim_matches('_');
        if sanitized.is_empty() {
            "odon.mosaic.screenshot.png".to_string()
        } else {
            format!("{sanitized}.mosaic.screenshot.png")
        }
    }

    pub fn confirm_or_request_close_dialog(&mut self) -> bool {
        if self.close_dialog_open {
            self.close_dialog_open = false;
            return true;
        }
        self.close_dialog_open = true;
        false
    }

    pub fn from_args(
        cc: &eframe::CreationContext<'_>,
        args: MosaicCliArgs,
    ) -> anyhow::Result<Self> {
        if let Some(sheet) = args.samplesheet_csv.as_deref() {
            return Self::from_samplesheet(cc, sheet, args.columns);
        }
        Self::from_config(cc, args)
    }

    pub fn from_local_paths(
        ctx: &egui::Context,
        gpu_available: bool,
        roi_paths: Vec<PathBuf>,
        columns: Option<usize>,
    ) -> anyhow::Result<Self> {
        apply_napari_like_dark(ctx);
        if !gpu_available {
            anyhow::bail!("mosaic mode requires GPU (OpenGL) backend");
        }

        let mut items: Vec<MosaicItem> = Vec::new();
        let mut stores: Vec<Arc<dyn zarrs::storage::ReadableStorageTraits>> = Vec::new();
        for p in roi_paths {
            match OmeZarrDataset::open_local(&p) {
                Ok((ds, store)) => {
                    let id = items.len();
                    let sample_id = p
                        .file_name()
                        .and_then(|s| s.to_str())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| ds.source.display_name());
                    items.push(MosaicItem {
                        id,
                        sample_id,
                        meta: Default::default(),
                        dataset: ds,
                        offset: egui::vec2(0.0, 0.0),
                        scale: 1.0,
                        placed_size: egui::vec2(1.0, 1.0),
                    });
                    stores.push(store);
                }
                Err(err) => eprintln!("skipping ROI {}: {err}", p.to_string_lossy()),
            }
        }

        if items.len() < 2 {
            anyhow::bail!("need at least 2 valid OME-Zarr roots to open mosaic");
        }

        let abs_max = items
            .iter()
            .map(|it| it.dataset.abs_max)
            .fold(0.0f32, f32::max)
            .max(1.0);

        let channels: Vec<GlobalChannel> =
            build_global_channels(items.iter().map(|it| &it.dataset));

        let n = items.len();
        let cols = columns
            .filter(|&c| c > 0)
            .unwrap_or_else(|| ((n as f32).sqrt().ceil() as usize).max(1));

        let pad = 64.0f32;
        let (cell_w, cell_h) = max_level0_size_items(&items);
        let cell_w = cell_w.max(1.0);
        let cell_h = cell_h.max(1.0);

        let (mosaic_bounds, group_blocks) = layout_items_grouped(
            &mut items,
            cols,
            cell_w,
            cell_h,
            pad,
            None,
            0.0,
            MosaicLayoutMode::FitCells,
        );

        let sources = Arc::new(
            items
                .iter()
                .zip(stores.iter())
                .map(|(it, store)| MosaicSource {
                    source: it.dataset.source.clone(),
                    store: store.clone(),
                    levels: it.dataset.levels.clone(),
                    dims: it.dataset.dims.clone(),
                    channel_map: build_channel_map(&channels, &it.dataset),
                })
                .collect::<Vec<_>>(),
        );

        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8)
            .clamp(2, 16);
        let pinned_levels = MosaicPinnedLevels::new();
        let loader = spawn_mosaic_raw_tile_loader(
            Arc::clone(&sources),
            pinned_levels.clone(),
            threads,
            8192,
        )?;

        let mut camera = Camera::default();
        if let Some(viewport) = ctx.input(|i| i.viewport().inner_rect) {
            camera.fit_to_world_rect(viewport, mosaic_bounds);
        } else {
            camera.center_world_lvl0 = mosaic_bounds.center();
            camera.zoom_screen_per_lvl0_px = 0.01;
        }

        let focused_core_id = items.first().map(|it| it.id);

        let channel_layer_order = (0..channels.len()).collect::<Vec<_>>();
        let overlay_layer_order = vec![
            MosaicLayerId::SegmentationGeoJson,
            MosaicLayerId::TextLabels,
        ];
        let active_layer = channel_layer_order
            .first()
            .copied()
            .map(MosaicLayerId::Channel)
            .unwrap_or(MosaicLayerId::TextLabels);

        let sources_len = sources.len();
        let initial_selected_channel_layers = if channels.is_empty() {
            HashSet::new()
        } else {
            HashSet::from([0usize])
        };
        let initial_memory_selected_channels = (0..channels.len()).collect::<HashSet<_>>();

        Ok(Self {
            items,
            sources,
            pinned_levels,
            loader,
            tiles_gl: MosaicTilesGl::new(12_000),
            remote_runtimes: Vec::new(),
            camera,
            last_canvas_rect: None,
            mosaic_bounds,
            focused_core_id,
            abs_max,
            channels,
            selected_channel: 0,
            channel_list_search: String::new(),
            active_layer,
            selected_channel_layers: initial_selected_channel_layers,
            memory_selected_channels: initial_memory_selected_channels,
            channel_select_anchor_idx: None,
            selected_channel_group_id: None,
            selected_overlay_layers: HashSet::new(),
            overlay_select_anchor_pos: None,
            overlay_layer_order,
            channel_layer_order,
            annotation_layers: Vec::new(),
            next_annotation_layer_id: 1,
            last_target_level_by_dataset_id: vec![None; sources_len],
            fallback_ceiling_by_dataset_id: vec![None; sources_len],
            zoom_out_floor_by_dataset_id: vec![None; sources_len],
            zoom_out_floor_until_by_dataset_id: vec![None; sources_len],
            zoom_out_floor_world_by_dataset_id: vec![None; sources_len],
            last_visible_world: None,
            layer_groups: ProjectLayerGroups::default(),
            layer_drag: None,
            left_tab: LeftTab::Layers,
            right_tab: RightTab::Properties,
            metadata_columns: Vec::new(),
            sort_by: "id".to_string(),
            sort_secondary_enabled: false,
            sort_by_secondary: "id".to_string(),
            group_by: String::new(),
            show_group_labels: true,
            group_gap: 96.0,
            layout_mode: MosaicLayoutMode::FitCells,
            group_blocks,
            show_text_labels: true,
            label_columns: vec!["id".to_string()],
            grid_cols: cols,
            grid_cell_w: cell_w,
            grid_cell_h: cell_h,
            grid_pad: pad,
            show_left_panel: true,
            show_right_panel: true,
            close_dialog_open: false,
            system_memory: None,
            system_memory_last_refresh: None,
            pending_memory_load: None,
            status: "Ready.".to_string(),
            allow_back: true,
            pending_request: None,
            group_layers_dialog: None,
            smooth_pixels: true,
            show_tile_debug: false,
            screenshot_settings: ScreenshotSettings::default(),
            screenshot_settings_open: false,
            screenshot_worker: ScreenshotWorkerHandle::spawn(),
            screenshot_next_id: 1,
            screenshot_pending: None,
            screenshot_in_flight: None,
            screenshot_output_dir: None,
            seg_geojson: MosaicGeoJsonSegmentationOverlay::default(),
            seg_geojson_pending_visible: false,
            tile_request_generation: 1,
            last_tile_request_signature: None,
            project_space: ProjectSpace::default(),
        })
    }

    pub fn from_remote_s3_sources(
        ctx: &egui::Context,
        gpu_available: bool,
        datasets: Vec<S3DatasetSelection>,
        columns: Option<usize>,
    ) -> anyhow::Result<Self> {
        apply_napari_like_dark(ctx);
        if !gpu_available {
            anyhow::bail!("mosaic mode requires GPU (OpenGL) backend");
        }

        let mut items: Vec<MosaicItem> = Vec::new();
        let mut stores: Vec<Arc<dyn zarrs::storage::ReadableStorageTraits>> = Vec::new();
        let mut remote_runtimes: Vec<Arc<tokio::runtime::Runtime>> = Vec::new();

        for spec in datasets {
            let endpoint = spec.endpoint.trim().to_string();
            let region = spec.region.trim().to_string();
            let bucket = spec.bucket.trim().to_string();
            let prefix = spec.prefix.trim().trim_matches('/').to_string();
            let access_key = spec.access_key.trim().to_string();
            let secret_key = spec.secret_key.trim().to_string();

            let crate::data::remote_store::S3Store { store, runtime } = build_s3_store(
                &endpoint,
                &region,
                &bucket,
                &prefix,
                &access_key,
                &secret_key,
            )?;
            let source = DatasetSource::S3 {
                endpoint,
                region: if region.is_empty() {
                    "auto".to_string()
                } else {
                    region
                },
                bucket,
                prefix,
            };
            let ds = OmeZarrDataset::open_with_store(source, store.clone())?;
            let id = items.len();
            let sample_id = ds.source.display_name();
            items.push(MosaicItem {
                id,
                sample_id,
                meta: Default::default(),
                dataset: ds,
                offset: egui::vec2(0.0, 0.0),
                scale: 1.0,
                placed_size: egui::vec2(1.0, 1.0),
            });
            stores.push(store);
            remote_runtimes.push(runtime);
        }

        if items.len() < 2 {
            anyhow::bail!("need at least 2 valid S3 OME-Zarr roots to open mosaic");
        }

        let abs_max = items
            .iter()
            .map(|it| it.dataset.abs_max)
            .fold(0.0f32, f32::max)
            .max(1.0);

        let channels: Vec<GlobalChannel> =
            build_global_channels(items.iter().map(|it| &it.dataset));

        let n = items.len();
        let cols = columns
            .filter(|&c| c > 0)
            .unwrap_or_else(|| ((n as f32).sqrt().ceil() as usize).max(1));

        let pad = 64.0f32;
        let (cell_w, cell_h) = max_level0_size_items(&items);
        let cell_w = cell_w.max(1.0);
        let cell_h = cell_h.max(1.0);

        let (mosaic_bounds, group_blocks) = layout_items_grouped(
            &mut items,
            cols,
            cell_w,
            cell_h,
            pad,
            None,
            0.0,
            MosaicLayoutMode::FitCells,
        );

        let sources = Arc::new(
            items
                .iter()
                .zip(stores.iter())
                .map(|(it, store)| MosaicSource {
                    source: it.dataset.source.clone(),
                    store: store.clone(),
                    levels: it.dataset.levels.clone(),
                    dims: it.dataset.dims.clone(),
                    channel_map: build_channel_map(&channels, &it.dataset),
                })
                .collect::<Vec<_>>(),
        );

        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8)
            .clamp(2, 16);
        let pinned_levels = MosaicPinnedLevels::new();
        let loader = spawn_mosaic_raw_tile_loader(
            Arc::clone(&sources),
            pinned_levels.clone(),
            threads,
            8192,
        )?;

        let mut camera = Camera::default();
        if let Some(viewport) = ctx.input(|i| i.viewport().inner_rect) {
            camera.fit_to_world_rect(viewport, mosaic_bounds);
        } else {
            camera.center_world_lvl0 = mosaic_bounds.center();
            camera.zoom_screen_per_lvl0_px = 0.01;
        }

        let focused_core_id = items.first().map(|it| it.id);
        let channel_layer_order = (0..channels.len()).collect::<Vec<_>>();
        let overlay_layer_order = vec![
            MosaicLayerId::SegmentationGeoJson,
            MosaicLayerId::TextLabels,
        ];
        let active_layer = channel_layer_order
            .first()
            .copied()
            .map(MosaicLayerId::Channel)
            .unwrap_or(MosaicLayerId::TextLabels);

        let sources_len = sources.len();
        let initial_selected_channel_layers = if channels.is_empty() {
            HashSet::new()
        } else {
            HashSet::from([0usize])
        };
        let initial_memory_selected_channels = (0..channels.len()).collect::<HashSet<_>>();

        Ok(Self {
            items,
            sources,
            pinned_levels,
            loader,
            tiles_gl: MosaicTilesGl::new(12_000),
            remote_runtimes,
            camera,
            last_canvas_rect: None,
            mosaic_bounds,
            focused_core_id,
            abs_max,
            channels,
            selected_channel: 0,
            channel_list_search: String::new(),
            active_layer,
            selected_channel_layers: initial_selected_channel_layers,
            memory_selected_channels: initial_memory_selected_channels,
            channel_select_anchor_idx: None,
            selected_channel_group_id: None,
            selected_overlay_layers: HashSet::new(),
            overlay_select_anchor_pos: None,
            overlay_layer_order,
            channel_layer_order,
            annotation_layers: Vec::new(),
            next_annotation_layer_id: 1,
            last_target_level_by_dataset_id: vec![None; sources_len],
            fallback_ceiling_by_dataset_id: vec![None; sources_len],
            zoom_out_floor_by_dataset_id: vec![None; sources_len],
            zoom_out_floor_until_by_dataset_id: vec![None; sources_len],
            zoom_out_floor_world_by_dataset_id: vec![None; sources_len],
            last_visible_world: None,
            layer_groups: ProjectLayerGroups::default(),
            layer_drag: None,
            left_tab: LeftTab::Layers,
            right_tab: RightTab::Properties,
            metadata_columns: Vec::new(),
            sort_by: "id".to_string(),
            sort_secondary_enabled: false,
            sort_by_secondary: "id".to_string(),
            group_by: String::new(),
            show_group_labels: true,
            group_gap: 96.0,
            layout_mode: MosaicLayoutMode::FitCells,
            group_blocks,
            show_text_labels: true,
            label_columns: vec!["id".to_string()],
            grid_cols: cols,
            grid_cell_w: cell_w,
            grid_cell_h: cell_h,
            grid_pad: pad,
            show_left_panel: true,
            show_right_panel: true,
            close_dialog_open: false,
            system_memory: None,
            system_memory_last_refresh: None,
            pending_memory_load: None,
            status: "Ready.".to_string(),
            allow_back: true,
            pending_request: None,
            group_layers_dialog: None,
            smooth_pixels: true,
            show_tile_debug: false,
            screenshot_settings: ScreenshotSettings::default(),
            screenshot_settings_open: false,
            screenshot_worker: ScreenshotWorkerHandle::spawn(),
            screenshot_next_id: 1,
            screenshot_pending: None,
            screenshot_in_flight: None,
            screenshot_output_dir: None,
            seg_geojson: MosaicGeoJsonSegmentationOverlay::default(),
            seg_geojson_pending_visible: false,
            tile_request_generation: 1,
            last_tile_request_signature: None,
            project_space: ProjectSpace::default(),
        })
    }

    pub fn from_project_rois(
        ctx: &egui::Context,
        gpu_available: bool,
        rois: Vec<ProjectRoi>,
        project_dir: Option<PathBuf>,
        columns: Option<usize>,
    ) -> anyhow::Result<Self> {
        apply_napari_like_dark(ctx);
        if !gpu_available {
            anyhow::bail!("mosaic mode requires GPU (OpenGL) backend");
        }

        let mut meta_keys: HashSet<String> = HashSet::new();
        let mut items: Vec<MosaicItem> = Vec::new();
        let mut stores: Vec<Arc<dyn zarrs::storage::ReadableStorageTraits>> = Vec::new();
        let mut remote_runtimes: Vec<Arc<tokio::runtime::Runtime>> = Vec::new();
        for roi in &rois {
            let Some(source) = roi.dataset_source().map(|source| match source {
                DatasetSource::Local(path) if path.is_relative() => DatasetSource::Local(
                    project_dir
                        .as_ref()
                        .map(|dir| dir.join(&path))
                        .unwrap_or(path),
                ),
                other => other,
            }) else {
                continue;
            };

            let opened = match &source {
                DatasetSource::Local(path) => {
                    OmeZarrDataset::open_local(path).map(|(ds, store)| (ds, store, None))
                }
                DatasetSource::Http { base_url } => build_http_store(base_url).and_then(|store| {
                    OmeZarrDataset::open_with_store(source.clone(), store.clone())
                        .map(|ds| (ds, store, None))
                }),
                DatasetSource::S3 { .. } => Err(anyhow::anyhow!(
                    "project-backed S3 mosaic requires credentials via the S3 browser path"
                )),
            };

            match opened {
                Ok((ds, store, runtime)) => {
                    let id = items.len();
                    let mut meta = roi.meta.clone();
                    if let Some(seg) = roi.segpath.as_ref() {
                        meta.insert("segpath".to_string(), seg.to_string_lossy().to_string());
                    }
                    for key in meta.keys() {
                        if !key.trim().is_empty() {
                            meta_keys.insert(key.clone());
                        }
                    }
                    items.push(MosaicItem {
                        id,
                        sample_id: roi
                            .display_name
                            .as_deref()
                            .unwrap_or(roi.id.as_str())
                            .to_string(),
                        meta,
                        dataset: ds,
                        offset: egui::vec2(0.0, 0.0),
                        scale: 1.0,
                        placed_size: egui::vec2(1.0, 1.0),
                    });
                    stores.push(store);
                    if let Some(runtime) = runtime {
                        remote_runtimes.push(runtime);
                    }
                }
                Err(err) => eprintln!(
                    "skipping ROI id='{}' source='{}': {err}",
                    roi.id,
                    roi.source_display()
                ),
            }
        }

        if items.len() < 2 {
            anyhow::bail!("need at least 2 valid ROIs to open mosaic");
        }

        let mut meta_columns = meta_keys.into_iter().collect::<Vec<_>>();
        meta_columns.sort();

        let mut seg_geojson = MosaicGeoJsonSegmentationOverlay::default();
        seg_geojson.set_samplesheet_dir(project_dir);
        for it in &items {
            seg_geojson.discover_from_meta(it.id, &it.meta);
        }

        let abs_max = items
            .iter()
            .map(|it| it.dataset.abs_max)
            .fold(0.0f32, f32::max)
            .max(1.0);

        let channels: Vec<GlobalChannel> =
            build_global_channels(items.iter().map(|it| &it.dataset));

        let n = items.len();
        let cols = columns
            .filter(|&c| c > 0)
            .unwrap_or_else(|| ((n as f32).sqrt().ceil() as usize).max(1));

        let pad = 64.0f32;
        let (cell_w, cell_h) = max_level0_size_items(&items);
        let cell_w = cell_w.max(1.0);
        let cell_h = cell_h.max(1.0);

        let (mosaic_bounds, group_blocks) = layout_items_grouped(
            &mut items,
            cols,
            cell_w,
            cell_h,
            pad,
            None,
            0.0,
            MosaicLayoutMode::FitCells,
        );

        let sources = Arc::new(
            items
                .iter()
                .zip(stores.iter())
                .map(|(it, store)| MosaicSource {
                    source: it.dataset.source.clone(),
                    store: store.clone(),
                    levels: it.dataset.levels.clone(),
                    dims: it.dataset.dims.clone(),
                    channel_map: build_channel_map(&channels, &it.dataset),
                })
                .collect::<Vec<_>>(),
        );

        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8)
            .clamp(2, 16);
        let pinned_levels = MosaicPinnedLevels::new();
        let loader = spawn_mosaic_raw_tile_loader(
            Arc::clone(&sources),
            pinned_levels.clone(),
            threads,
            8192,
        )?;

        let mut camera = Camera::default();
        if let Some(viewport) = ctx.input(|i| i.viewport().inner_rect) {
            camera.fit_to_world_rect(viewport, mosaic_bounds);
        } else {
            camera.center_world_lvl0 = mosaic_bounds.center();
            camera.zoom_screen_per_lvl0_px = 0.01;
        }

        let focused_core_id = items.first().map(|it| it.id);

        let channel_layer_order = (0..channels.len()).collect::<Vec<_>>();
        let overlay_layer_order = vec![
            MosaicLayerId::SegmentationGeoJson,
            MosaicLayerId::TextLabels,
        ];
        let active_layer = channel_layer_order
            .first()
            .copied()
            .map(MosaicLayerId::Channel)
            .unwrap_or(MosaicLayerId::TextLabels);

        let sources_len = sources.len();
        let initial_selected_channel_layers = if channels.is_empty() {
            HashSet::new()
        } else {
            HashSet::from([0usize])
        };
        let initial_memory_selected_channels = (0..channels.len()).collect::<HashSet<_>>();

        Ok(Self {
            items,
            sources,
            pinned_levels,
            loader,
            tiles_gl: MosaicTilesGl::new(12_000),
            remote_runtimes,
            camera,
            last_canvas_rect: None,
            mosaic_bounds,
            focused_core_id,
            abs_max,
            channels,
            selected_channel: 0,
            channel_list_search: String::new(),
            active_layer,
            selected_channel_layers: initial_selected_channel_layers,
            memory_selected_channels: initial_memory_selected_channels,
            channel_select_anchor_idx: None,
            selected_channel_group_id: None,
            selected_overlay_layers: HashSet::new(),
            overlay_select_anchor_pos: None,
            overlay_layer_order,
            channel_layer_order,
            annotation_layers: Vec::new(),
            next_annotation_layer_id: 1,
            last_target_level_by_dataset_id: vec![None; sources_len],
            fallback_ceiling_by_dataset_id: vec![None; sources_len],
            zoom_out_floor_by_dataset_id: vec![None; sources_len],
            zoom_out_floor_until_by_dataset_id: vec![None; sources_len],
            zoom_out_floor_world_by_dataset_id: vec![None; sources_len],
            last_visible_world: None,
            layer_groups: ProjectLayerGroups::default(),
            layer_drag: None,
            left_tab: LeftTab::Layers,
            right_tab: RightTab::Properties,
            metadata_columns: meta_columns,
            sort_by: "id".to_string(),
            sort_secondary_enabled: false,
            sort_by_secondary: "id".to_string(),
            group_by: String::new(),
            show_group_labels: true,
            group_gap: 96.0,
            layout_mode: MosaicLayoutMode::FitCells,
            group_blocks,
            show_text_labels: true,
            label_columns: vec!["id".to_string()],
            grid_cols: cols,
            grid_cell_w: cell_w,
            grid_cell_h: cell_h,
            grid_pad: pad,
            show_left_panel: true,
            show_right_panel: true,
            close_dialog_open: false,
            system_memory: None,
            system_memory_last_refresh: None,
            pending_memory_load: None,
            status: "Ready.".to_string(),
            allow_back: true,
            pending_request: None,
            group_layers_dialog: None,
            smooth_pixels: true,
            show_tile_debug: false,
            screenshot_settings: ScreenshotSettings::default(),
            screenshot_settings_open: false,
            screenshot_worker: ScreenshotWorkerHandle::spawn(),
            screenshot_next_id: 1,
            screenshot_pending: None,
            screenshot_in_flight: None,
            screenshot_output_dir: None,
            seg_geojson,
            seg_geojson_pending_visible: false,
            tile_request_generation: 1,
            last_tile_request_signature: None,
            project_space: ProjectSpace::default(),
        })
    }

    fn from_config(cc: &eframe::CreationContext<'_>, args: MosaicCliArgs) -> anyhow::Result<Self> {
        apply_napari_like_dark(&cc.egui_ctx);

        let _gl = cc
            .gl
            .as_ref()
            .context("mosaic mode requires GPU (OpenGL) backend")?;

        let project_path = args
            .project_path
            .as_deref()
            .context("mosaic config mode requires --project")?;
        let mut ps = ProjectSpace::default();
        ps.load_from_file(project_path).with_context(|| {
            format!("failed to load project: {}", project_path.to_string_lossy())
        })?;
        let cfg = ps.config().clone();
        let project_dir = project_path.parent().map(|p| p.to_path_buf());
        let default_dataset = cfg
            .default_dataset
            .as_deref()
            .unwrap_or("default")
            .to_string();

        let dataset_names = args.dataset_names;
        let want_all = dataset_names.is_empty();

        let mut meta_keys: HashSet<String> = HashSet::new();
        let mut items: Vec<MosaicItem> = Vec::new();
        let mut stores: Vec<Arc<dyn zarrs::storage::ReadableStorageTraits>> = Vec::new();
        let mut remote_runtimes: Vec<Arc<tokio::runtime::Runtime>> = Vec::new();
        for roi in &cfg.rois {
            let ds_key = roi.dataset.as_deref().unwrap_or(default_dataset.as_str());
            if !want_all && !dataset_names.iter().any(|n| n == ds_key) {
                continue;
            }

            let Some(source) = roi.dataset_source().map(|source| match source {
                DatasetSource::Local(path) if path.is_relative() => DatasetSource::Local(
                    project_dir.as_ref().map(|d| d.join(&path)).unwrap_or(path),
                ),
                other => other,
            }) else {
                continue;
            };

            let opened = match &source {
                DatasetSource::Local(path) => {
                    OmeZarrDataset::open_local(path).map(|(ds, store)| (ds, store, None))
                }
                DatasetSource::Http { base_url } => build_http_store(base_url).and_then(|store| {
                    OmeZarrDataset::open_with_store(source.clone(), store.clone())
                        .map(|ds| (ds, store, None))
                }),
                DatasetSource::S3 { .. } => Err(anyhow::anyhow!(
                    "project-backed S3 mosaic requires credentials via the S3 browser path"
                )),
            };

            match opened {
                Ok((ds, store, runtime)) => {
                    let id = items.len();
                    let mut meta = roi.meta.clone();
                    if let Some(seg) = roi.segpath.as_ref() {
                        meta.insert("segpath".to_string(), seg.to_string_lossy().to_string());
                    }
                    for k in meta.keys() {
                        if !k.trim().is_empty() {
                            meta_keys.insert(k.clone());
                        }
                    }
                    items.push(MosaicItem {
                        id,
                        sample_id: roi
                            .display_name
                            .as_deref()
                            .unwrap_or(roi.id.as_str())
                            .to_string(),
                        meta,
                        dataset: ds,
                        offset: egui::vec2(0.0, 0.0),
                        scale: 1.0,
                        placed_size: egui::vec2(1.0, 1.0),
                    });
                    stores.push(store);
                    if let Some(runtime) = runtime {
                        remote_runtimes.push(runtime);
                    }
                }
                Err(err) => eprintln!(
                    "skipping ROI id='{}' source='{}': {err}",
                    roi.id,
                    roi.source_display()
                ),
            }
        }
        if items.len() < 2 {
            anyhow::bail!(
                "need at least 2 valid ROIs to open mosaic (filtered by datasets={:?})",
                dataset_names
            );
        }

        let mut meta_columns = meta_keys.into_iter().collect::<Vec<_>>();
        meta_columns.sort();

        let mut seg_geojson = MosaicGeoJsonSegmentationOverlay::default();
        seg_geojson.set_samplesheet_dir(project_dir);
        for it in &items {
            seg_geojson.discover_from_meta(it.id, &it.meta);
        }

        let abs_max = items
            .iter()
            .map(|it| it.dataset.abs_max)
            .fold(0.0f32, f32::max)
            .max(1.0);

        let channels: Vec<GlobalChannel> =
            build_global_channels(items.iter().map(|it| &it.dataset));

        let n = items.len();
        let cols = args
            .columns
            .filter(|&c| c > 0)
            .unwrap_or_else(|| ((n as f32).sqrt().ceil() as usize).max(1));

        let pad = 64.0f32;
        let (cell_w, cell_h) = max_level0_size_items(&items);
        let cell_w = cell_w.max(1.0);
        let cell_h = cell_h.max(1.0);

        let (mosaic_bounds, group_blocks) = layout_items_grouped(
            &mut items,
            cols,
            cell_w,
            cell_h,
            pad,
            None,
            0.0,
            MosaicLayoutMode::FitCells,
        );

        let sources = Arc::new(
            items
                .iter()
                .zip(stores.iter())
                .map(|(it, store)| MosaicSource {
                    source: it.dataset.source.clone(),
                    store: store.clone(),
                    levels: it.dataset.levels.clone(),
                    dims: it.dataset.dims.clone(),
                    channel_map: build_channel_map(&channels, &it.dataset),
                })
                .collect::<Vec<_>>(),
        );

        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8)
            .clamp(2, 16);
        let pinned_levels = MosaicPinnedLevels::new();
        let loader = spawn_mosaic_raw_tile_loader(
            Arc::clone(&sources),
            pinned_levels.clone(),
            threads,
            8192,
        )?;

        let mut camera = Camera::default();
        if let Some(viewport) = cc.egui_ctx.input(|i| i.viewport().inner_rect) {
            camera.fit_to_world_rect(viewport, mosaic_bounds);
        } else {
            camera.center_world_lvl0 = mosaic_bounds.center();
            camera.zoom_screen_per_lvl0_px = 0.01;
        }

        let focused_core_id = items.first().map(|it| it.id);

        let channel_layer_order = (0..channels.len()).collect::<Vec<_>>();
        let overlay_layer_order = vec![
            MosaicLayerId::SegmentationGeoJson,
            MosaicLayerId::TextLabels,
        ];
        let active_layer = channel_layer_order
            .first()
            .copied()
            .map(MosaicLayerId::Channel)
            .unwrap_or(MosaicLayerId::TextLabels);

        let sources_len = sources.len();
        let initial_selected_channel_layers = if channels.is_empty() {
            HashSet::new()
        } else {
            HashSet::from([0usize])
        };
        let initial_memory_selected_channels = (0..channels.len()).collect::<HashSet<_>>();

        Ok(Self {
            items,
            sources,
            pinned_levels,
            loader,
            tiles_gl: MosaicTilesGl::new(12_000),
            remote_runtimes,
            camera,
            last_canvas_rect: None,
            mosaic_bounds,
            focused_core_id,
            abs_max,
            channels,
            selected_channel: 0,
            channel_list_search: String::new(),
            active_layer,
            selected_channel_layers: initial_selected_channel_layers,
            memory_selected_channels: initial_memory_selected_channels,
            channel_select_anchor_idx: None,
            selected_channel_group_id: None,
            selected_overlay_layers: HashSet::new(),
            overlay_select_anchor_pos: None,
            overlay_layer_order,
            channel_layer_order,
            annotation_layers: Vec::new(),
            next_annotation_layer_id: 1,
            last_target_level_by_dataset_id: vec![None; sources_len],
            fallback_ceiling_by_dataset_id: vec![None; sources_len],
            zoom_out_floor_by_dataset_id: vec![None; sources_len],
            zoom_out_floor_until_by_dataset_id: vec![None; sources_len],
            zoom_out_floor_world_by_dataset_id: vec![None; sources_len],
            last_visible_world: None,
            layer_groups: ProjectLayerGroups::default(),
            layer_drag: None,
            left_tab: LeftTab::Layers,
            right_tab: RightTab::Properties,
            metadata_columns: meta_columns,
            sort_by: "id".to_string(),
            sort_secondary_enabled: false,
            sort_by_secondary: "id".to_string(),
            group_by: String::new(),
            show_group_labels: true,
            group_gap: 96.0,
            layout_mode: MosaicLayoutMode::FitCells,
            group_blocks,
            show_text_labels: true,
            label_columns: vec!["id".to_string()],
            grid_cols: cols,
            grid_cell_w: cell_w,
            grid_cell_h: cell_h,
            grid_pad: pad,
            show_left_panel: true,
            show_right_panel: true,
            close_dialog_open: false,
            system_memory: None,
            system_memory_last_refresh: None,
            pending_memory_load: None,
            status: "Ready.".to_string(),
            allow_back: false,
            pending_request: None,
            group_layers_dialog: None,
            smooth_pixels: true,
            show_tile_debug: false,
            screenshot_settings: ScreenshotSettings::default(),
            screenshot_settings_open: false,
            screenshot_worker: ScreenshotWorkerHandle::spawn(),
            screenshot_next_id: 1,
            screenshot_pending: None,
            screenshot_in_flight: None,
            screenshot_output_dir: None,
            seg_geojson,
            seg_geojson_pending_visible: false,
            tile_request_generation: 1,
            last_tile_request_signature: None,
            project_space: ProjectSpace::default(),
        })
    }

    fn from_samplesheet(
        cc: &eframe::CreationContext<'_>,
        samplesheet_csv: &Path,
        columns: Option<usize>,
    ) -> anyhow::Result<Self> {
        apply_napari_like_dark(&cc.egui_ctx);

        let _gl = cc
            .gl
            .as_ref()
            .context("mosaic mode requires GPU (OpenGL) backend")?;

        let sheet = load_samplesheet_csv(samplesheet_csv)?;
        let samplesheet_dir = samplesheet_csv.parent().map(|p| p.to_path_buf());
        let mut items: Vec<MosaicItem> = Vec::with_capacity(sheet.rows.len());
        let mut stores: Vec<Arc<dyn zarrs::storage::ReadableStorageTraits>> =
            Vec::with_capacity(sheet.rows.len());
        for row in &sheet.rows {
            match OmeZarrDataset::open_local(&row.path) {
                Ok((ds, store)) => {
                    let id = items.len();
                    items.push(MosaicItem {
                        id,
                        sample_id: row.id.clone(),
                        meta: row.meta.clone(),
                        dataset: ds,
                        offset: egui::vec2(0.0, 0.0),
                        scale: 1.0,
                        placed_size: egui::vec2(1.0, 1.0),
                    });
                    stores.push(store);
                }
                Err(err) => eprintln!(
                    "skipping samplesheet row id='{}' path='{}': {err}",
                    row.id,
                    row.path.to_string_lossy()
                ),
            }
        }
        if items.is_empty() {
            anyhow::bail!(
                "failed to open any ROIs from samplesheet: {}",
                samplesheet_csv.to_string_lossy()
            );
        }

        let mut seg_geojson = MosaicGeoJsonSegmentationOverlay::default();
        seg_geojson.set_samplesheet_dir(samplesheet_dir);
        for it in &items {
            seg_geojson.discover_from_meta(it.id, &it.meta);
        }

        let abs_max = items
            .iter()
            .map(|it| it.dataset.abs_max)
            .fold(0.0f32, f32::max)
            .max(1.0);

        let channels: Vec<GlobalChannel> =
            build_global_channels(items.iter().map(|it| &it.dataset));

        let n = items.len();
        let cols = columns
            .filter(|&c| c > 0)
            .unwrap_or_else(|| ((n as f32).sqrt().ceil() as usize).max(1));

        let pad = 64.0f32;
        let (cell_w, cell_h) = max_level0_size_items(&items);
        let cell_w = cell_w.max(1.0);
        let cell_h = cell_h.max(1.0);
        for it in &mut items {
            if it.sample_id.trim().is_empty() {
                it.sample_id = it.dataset.source.display_name();
            }
        }

        let (mosaic_bounds, group_blocks) = layout_items_grouped(
            &mut items,
            cols,
            cell_w,
            cell_h,
            pad,
            None,
            0.0,
            MosaicLayoutMode::FitCells,
        );

        let sources = Arc::new(
            items
                .iter()
                .zip(stores.iter())
                .map(|(it, store)| MosaicSource {
                    source: it.dataset.source.clone(),
                    store: store.clone(),
                    levels: it.dataset.levels.clone(),
                    dims: it.dataset.dims.clone(),
                    channel_map: build_channel_map(&channels, &it.dataset),
                })
                .collect::<Vec<_>>(),
        );

        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8)
            .clamp(2, 16);
        let pinned_levels = MosaicPinnedLevels::new();
        let loader = spawn_mosaic_raw_tile_loader(
            Arc::clone(&sources),
            pinned_levels.clone(),
            threads,
            8192,
        )?;

        let mut camera = Camera::default();
        if let Some(viewport) = cc.egui_ctx.input(|i| i.viewport().inner_rect) {
            camera.fit_to_world_rect(viewport, mosaic_bounds);
        } else {
            camera.center_world_lvl0 = mosaic_bounds.center();
            camera.zoom_screen_per_lvl0_px = 0.01;
        }

        let focused_core_id = items.first().map(|it| it.id);

        let channel_layer_order = (0..channels.len()).collect::<Vec<_>>();
        let overlay_layer_order = vec![
            MosaicLayerId::SegmentationGeoJson,
            MosaicLayerId::TextLabels,
        ];
        let active_layer = channel_layer_order
            .first()
            .copied()
            .map(MosaicLayerId::Channel)
            .unwrap_or(MosaicLayerId::TextLabels);

        let sources_len = sources.len();
        let initial_selected_channel_layers = if channels.is_empty() {
            HashSet::new()
        } else {
            HashSet::from([0usize])
        };
        let initial_memory_selected_channels = (0..channels.len()).collect::<HashSet<_>>();

        Ok(Self {
            items,
            sources,
            pinned_levels,
            loader,
            tiles_gl: MosaicTilesGl::new(12_000),
            remote_runtimes: Vec::new(),
            camera,
            last_canvas_rect: None,
            mosaic_bounds,
            focused_core_id,
            abs_max,
            channels,
            selected_channel: 0,
            channel_list_search: String::new(),
            active_layer,
            selected_channel_layers: initial_selected_channel_layers,
            memory_selected_channels: initial_memory_selected_channels,
            channel_select_anchor_idx: None,
            selected_channel_group_id: None,
            selected_overlay_layers: HashSet::new(),
            overlay_select_anchor_pos: None,
            overlay_layer_order,
            channel_layer_order,
            annotation_layers: Vec::new(),
            next_annotation_layer_id: 1,
            last_target_level_by_dataset_id: vec![None; sources_len],
            fallback_ceiling_by_dataset_id: vec![None; sources_len],
            zoom_out_floor_by_dataset_id: vec![None; sources_len],
            zoom_out_floor_until_by_dataset_id: vec![None; sources_len],
            zoom_out_floor_world_by_dataset_id: vec![None; sources_len],
            last_visible_world: None,
            layer_groups: ProjectLayerGroups::default(),
            layer_drag: None,
            left_tab: LeftTab::Layers,
            right_tab: RightTab::Properties,
            metadata_columns: sheet.meta_columns,
            sort_by: "id".to_string(),
            sort_secondary_enabled: false,
            sort_by_secondary: "id".to_string(),
            group_by: String::new(),
            show_group_labels: true,
            group_gap: 96.0,
            layout_mode: MosaicLayoutMode::FitCells,
            group_blocks,
            show_text_labels: true,
            label_columns: vec!["id".to_string()],
            grid_cols: cols,
            grid_cell_w: cell_w,
            grid_cell_h: cell_h,
            grid_pad: pad,
            show_left_panel: true,
            show_right_panel: true,
            close_dialog_open: false,
            system_memory: None,
            system_memory_last_refresh: None,
            pending_memory_load: None,
            status: format!(
                "Loaded samplesheet: {}",
                samplesheet_csv
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("<samplesheet>")
            ),
            allow_back: false,
            pending_request: None,
            group_layers_dialog: None,
            smooth_pixels: true,
            show_tile_debug: false,
            screenshot_settings: ScreenshotSettings::default(),
            screenshot_settings_open: false,
            screenshot_worker: ScreenshotWorkerHandle::spawn(),
            screenshot_next_id: 1,
            screenshot_pending: None,
            screenshot_in_flight: None,
            screenshot_output_dir: None,
            seg_geojson,
            seg_geojson_pending_visible: false,
            tile_request_generation: 1,
            last_tile_request_signature: None,
            project_space: ProjectSpace::default(),
        })
    }
}

impl eframe::App for MosaicViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Per-frame flow mirrors the single-view app but with shared mosaic state:
        // refresh/tick async overlays, build chrome and side panels, then draw the current
        // viewport while progressively refining visible ROIs.
        self.refresh_system_memory_if_needed();
        self.seg_geojson.tick();
        self.drain_screenshots();
        for layer in &mut self.annotation_layers {
            if layer.tick() {
                ctx.request_repaint();
            }
        }

        // Napari-like "close window" prompt.
        // - Cmd/Ctrl+W opens confirmation
        // - Cmd/Ctrl+W again confirms close
        if top_bar::handle_cmd_w_close(ctx, &mut self.close_dialog_open) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                top_bar::ui_title(ui, format!("Mosaic: {} ROIs", self.items.len()));
                ui.separator();
                if top_bar::ui_back(ui, self.allow_back) {
                    self.pending_request = Some(MosaicRequest::BackToSingle);
                }
                if top_bar::ui_fit(ui, "Fit Mosaic (F)") {
                    self.fit_mosaic();
                }
                ui.separator();
                top_bar::ui_status(ui, &self.status);
                ui.separator();
                let have_items = !self.items.is_empty();
                if let Some(step) = top_bar::ui_prev_next_core(ui, have_items) {
                    self.step_focused_core(ctx, step);
                }
                top_bar::ui_core_index(ui, self.focused_core_summary());
                ui.separator();
                let have_channels = !self.channels.is_empty();
                if let Some(step) = top_bar::ui_prev_next_channel(ui, have_channels) {
                    self.step_selected_channel_visibility(step);
                }
                ui.separator();
                top_bar::ui_panel_toggles(
                    ui,
                    &mut self.show_left_panel,
                    &mut self.show_right_panel,
                );
                if top_bar::ui_smooth_toggle(ui, &mut self.smooth_pixels) {
                    self.tiles_gl.set_smooth_pixels(self.smooth_pixels);
                }
                ui.checkbox(&mut self.show_tile_debug, "Tile Debug");

                // Compact contrast controls when both side panels are hidden.
                if !self.show_left_panel && !self.show_right_panel && have_channels {
                    ui.separator();
                    let abs_max = self.abs_max.max(1.0);
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
                        }
                    }
                }
            });
        });

        if self.show_left_panel {
            let mut tab = self.left_tab;
            left_panel::show(
                ctx,
                "mosaic-left",
                &mut tab,
                &[
                    left_panel::TabSpec {
                        tab: LeftTab::Layers,
                        label: "Layers",
                        panel_key: "layers",
                        default_width: 360.0,
                        scroll: true,
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
                    LeftTab::Project => self.ui_project(ui),
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
                        scroll: true,
                    },
                    right_panel::TabSpec {
                        tab: RightTab::Layout,
                        label: "Layout",
                        scroll: true,
                    },
                    right_panel::TabSpec {
                        tab: RightTab::Memory,
                        label: "Memory",
                        scroll: true,
                    },
                ],
                |ui, tab| match tab {
                    RightTab::Properties => self.ui_properties(ui),
                    RightTab::Layout => self.ui_layout(ui, ctx),
                    RightTab::Memory => self.ui_memory(ui),
                },
            );
            self.right_tab = tab;
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            self.ui_canvas(ui, ctx);
        });

        self.ui_group_layers_dialog(ctx);
        self.ui_memory_load_dialog(ctx);
        self.ui_screenshot_settings_dialog(ctx);

        if top_bar::ui_close_dialog(ctx, &mut self.close_dialog_open) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }

        if ctx.input(|i| i.key_pressed(egui::Key::F)) {
            self.fit_mosaic();
        }

        // Avoid a busy loop when idle. Only repaint while we are interacting or still streaming.
        if self.tiles_gl.is_busy()
            || self.seg_geojson.is_busy()
            || self.seg_geojson_pending_visible
            || self.pinned_levels.has_loading()
            || self.screenshot_pending.is_some()
            || self.screenshot_in_flight.is_some()
        {
            repaint_control::request_repaint_busy(ctx);
        }
    }
}

impl MosaicViewerApp {
    fn focused_item(&self) -> Option<&MosaicItem> {
        let id = self.focused_core_id?;
        self.items.iter().find(|it| it.id == id)
    }

    fn selected_memory_global_channels(&self) -> Vec<u64> {
        self.channel_layer_order
            .iter()
            .copied()
            .filter(|gid| self.memory_selected_channels.contains(gid))
            .map(|gid| gid as u64)
            .collect()
    }

    fn refresh_system_memory_if_needed(&mut self) {
        refresh_system_memory_if_needed(
            &mut self.system_memory,
            &mut self.system_memory_last_refresh,
            Duration::from_secs(2),
        );
    }

    fn memory_risk(&self, requested_bytes: u64) -> Option<MemoryRisk> {
        memory_risk(
            self.system_memory.as_ref(),
            self.pinned_levels.total_loaded_bytes(),
            requested_bytes,
        )
    }

    fn start_memory_load(
        &mut self,
        summary: String,
        requests: Vec<PendingMemoryLoadRequest>,
        requested_bytes: u64,
    ) {
        if requests.is_empty() {
            self.status = "No eligible channels selected for RAM pinning.".to_string();
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

    fn execute_memory_load(&mut self, summary: String, requests: Vec<PendingMemoryLoadRequest>) {
        let count = requests.len();
        for request in requests {
            self.pinned_levels.request_load(
                request.dataset_id,
                request.source,
                request.level,
                request.selected_global_channels,
            );
        }
        self.status = if count == 0 {
            "No eligible channels selected for RAM pinning.".to_string()
        } else {
            summary
        };
    }

    fn memory_load_requests_for_all_rois(
        &self,
        level: usize,
        selected_global_channels: &[u64],
    ) -> (Vec<PendingMemoryLoadRequest>, u64) {
        let mut requests = Vec::new();
        let mut total_bytes = 0u64;
        for item in &self.items {
            let Some(source) = self.sources.get(item.id).cloned() else {
                continue;
            };
            if source.levels.get(level).is_none() {
                continue;
            }
            let estimate = estimate_level_ram_bytes_for_channels(
                &source,
                level,
                Some(selected_global_channels),
            )
            .unwrap_or(0);
            if estimate == 0 {
                continue;
            }
            total_bytes = total_bytes.saturating_add(estimate);
            requests.push(PendingMemoryLoadRequest {
                dataset_id: item.id,
                source,
                level,
                selected_global_channels: selected_global_channels.to_vec(),
            });
        }
        (requests, total_bytes)
    }

    fn memory_load_request_for_dataset(
        &self,
        dataset_id: usize,
        source: MosaicSource,
        level: usize,
        selected_global_channels: &[u64],
    ) -> Option<(PendingMemoryLoadRequest, u64)> {
        let requested_bytes =
            estimate_level_ram_bytes_for_channels(&source, level, Some(selected_global_channels))
                .unwrap_or(0);
        if requested_bytes == 0 {
            return None;
        }
        Some((
            PendingMemoryLoadRequest {
                dataset_id,
                source,
                level,
                selected_global_channels: selected_global_channels.to_vec(),
            },
            requested_bytes,
        ))
    }

    fn unload_level_for_all_rois(&mut self, level: usize) -> usize {
        let mut count = 0usize;
        for item in &self.items {
            if self
                .sources
                .get(item.id)
                .and_then(|s| s.levels.get(level))
                .is_none()
            {
                continue;
            }
            self.pinned_levels.unload(item.id, level);
            count += 1;
        }
        count
    }

    fn refine_item_order(&self, visible_world: egui::Rect) -> Vec<usize> {
        if self.items.is_empty() {
            return Vec::new();
        }

        let center_world = self.camera.center_world_lvl0;
        let center_item_id = self
            .items
            .iter()
            .find(|it| item_rect(it).contains(center_world))
            .map(|it| it.id);
        let focused_id = self.focused_core_id;

        let mut out: Vec<(u8, f32, usize)> = Vec::new();
        out.reserve(self.items.len().min(256));
        for (idx, it) in self.items.iter().enumerate() {
            let r = item_rect(it);
            if !r.intersects(visible_world) {
                continue;
            }
            let pri = if Some(it.id) == center_item_id {
                0u8
            } else if Some(it.id) == focused_id {
                1u8
            } else {
                2u8
            };
            let d = r.center() - center_world;
            let dist2 = d.x * d.x + d.y * d.y;
            out.push((pri, dist2, idx));
        }

        out.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));
        out.into_iter().map(|t| t.2).collect()
    }

    fn step_selected_channel_visibility(&mut self, step: i32) {
        if self.channels.is_empty() || self.channel_layer_order.is_empty() {
            return;
        }
        let cur_gid = self
            .selected_channel
            .min(self.channels.len().saturating_sub(1));
        let cur_pos = self
            .channel_layer_order
            .iter()
            .position(|&g| g == cur_gid)
            .unwrap_or(0);
        let n = self.channel_layer_order.len() as i32;
        let next_pos = ((cur_pos as i32) + step).rem_euclid(n) as usize;
        let next_gid =
            self.channel_layer_order[next_pos].min(self.channels.len().saturating_sub(1));

        if let Some(cur) = self.channels.get_mut(cur_gid) {
            cur.visible = false;
        }
        if let Some(next) = self.channels.get_mut(next_gid) {
            next.visible = true;
        }
        self.selected_channel = next_gid;
        self.active_layer = MosaicLayerId::Channel(next_gid);
    }

    fn drain_raw_tiles(&mut self) {
        while let Ok(msg) = self.loader.rx.try_recv() {
            match msg {
                MosaicRawTileWorkerResponse::Tile(msg) => {
                    if msg.generation == self.tile_request_generation {
                        self.tiles_gl.insert_pending(msg);
                    } else {
                        self.tiles_gl.cancel_in_flight(&msg.key);
                    }
                }
                MosaicRawTileWorkerResponse::Dropped { key, .. } => {
                    self.tiles_gl.cancel_in_flight(&key);
                }
            }
        }
    }

    fn sync_tile_request_generation(
        &mut self,
        visible_world: egui::Rect,
        viewport: egui::Rect,
        channels_draw: &[ChannelDraw],
    ) {
        let signature = TileRequestSignature {
            viewport_width_bits: viewport.width().to_bits(),
            viewport_height_bits: viewport.height().to_bits(),
            visible_world_min_x_bits: visible_world.min.x.to_bits(),
            visible_world_min_y_bits: visible_world.min.y.to_bits(),
            visible_world_max_x_bits: visible_world.max.x.to_bits(),
            visible_world_max_y_bits: visible_world.max.y.to_bits(),
            visible_channels: channels_draw.iter().map(|ch| ch.index).collect(),
        };

        if self.last_tile_request_signature.as_ref() != Some(&signature) {
            self.tile_request_generation = self.tile_request_generation.wrapping_add(1).max(1);
            self.last_tile_request_signature = Some(signature);
            self.loader
                .set_latest_generation(self.tile_request_generation);
        }
    }

    fn fit_mosaic(&mut self) {
        if let Some(viewport) = self.last_canvas_rect {
            self.camera.fit_to_world_rect(viewport, self.mosaic_bounds);
        }
    }

    fn focused_core_summary(&self) -> Option<(usize, usize, String)> {
        let n = self.items.len();
        if n == 0 {
            return None;
        }
        let Some(id) = self.focused_core_id else {
            return None;
        };
        let idx = self.items.iter().position(|it| it.id == id)?;
        let name = self
            .items
            .get(idx)
            .map(|it| it.sample_id.clone())
            .unwrap_or_default();
        Some((idx + 1, n, name))
    }

    fn fit_focused_core(&mut self, ctx: &egui::Context) {
        if self.items.is_empty() {
            return;
        }
        let id = self
            .focused_core_id
            .filter(|id| self.items.iter().any(|it| it.id == *id))
            .unwrap_or(self.items[0].id);
        self.focused_core_id = Some(id);

        let Some(it) = self.items.iter().find(|it| it.id == id) else {
            return;
        };
        let world = item_rect(it);
        let viewport = self
            .last_canvas_rect
            .or_else(|| ctx.input(|i| i.viewport().inner_rect));
        if let Some(viewport) = viewport {
            self.camera.fit_to_world_rect(viewport, world);
        }
    }

    fn step_focused_core(&mut self, ctx: &egui::Context, step: i32) {
        let n = self.items.len();
        if n == 0 {
            return;
        }
        let cur_id = self.focused_core_id;
        let cur_idx = cur_id
            .and_then(|id| self.items.iter().position(|it| it.id == id))
            .unwrap_or(0);
        let next_idx = ((cur_idx as i32) + step).rem_euclid(n as i32) as usize;
        self.focused_core_id = Some(self.items[next_idx].id);
        self.fit_focused_core(ctx);
    }

    fn set_active_layer(&mut self, id: MosaicLayerId) {
        self.active_layer = id;
        if let MosaicLayerId::Channel(idx) = id {
            self.selected_channel = idx.min(self.channels.len().saturating_sub(1));
        } else {
            self.selected_channel_group_id = None;
        }
    }

    fn channel_indices_in_group(&self, group_id: u64) -> Vec<usize> {
        self.channel_layer_order
            .iter()
            .copied()
            .filter(|&idx| {
                self.channels.get(idx).is_some_and(|ch| {
                    self.layer_groups
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

    fn ui_group_contrast(&mut self, ui: &mut egui::Ui, group_id: u64) {
        let abs_max = self.abs_max.max(1.0);
        let Some(group) = self
            .layer_groups
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
        ui.heading("Contrast (global)");
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
                show_nudge_buttons: false,
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
                }
            }
            ui.ctx().request_repaint();
            return;
        }

        if out.limits_touched {
            for &idx in &members {
                if let Some(dst) = self.channels.get_mut(idx) {
                    dst.window = Some((lo, hi));
                }
            }
            ui.ctx().request_repaint();
        }
    }

    fn add_annotation_layer(&mut self) {
        let id = self.next_annotation_layer_id.max(1);
        self.next_annotation_layer_id = id.wrapping_add(1).max(1);
        let name = format!("Annotations {id}");
        self.annotation_layers
            .push(AnnotationPointsLayer::new(id, name));
        let layer_id = MosaicLayerId::Annotation(id);
        if !self.overlay_layer_order.contains(&layer_id) {
            self.overlay_layer_order.push(layer_id);
        }
        self.set_active_layer(layer_id);
    }

    fn layer_display_name(&self, id: MosaicLayerId) -> String {
        match id {
            MosaicLayerId::TextLabels => "Text labels".to_string(),
            MosaicLayerId::SegmentationGeoJson => "Segmentation (GeoJSON)".to_string(),
            MosaicLayerId::Annotation(id) => self
                .annotation_layers
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.name.clone())
                .unwrap_or_else(|| format!("Annotations {id}")),
            MosaicLayerId::Channel(idx) => self
                .channels
                .get(idx)
                .map(|c| c.name.clone())
                .unwrap_or_else(|| format!("Channel {idx}")),
        }
    }

    fn layer_icon(&self, id: MosaicLayerId) -> Icon {
        match id {
            MosaicLayerId::Channel(_) => Icon::Image,
            MosaicLayerId::SegmentationGeoJson => Icon::Polygon,
            MosaicLayerId::TextLabels => Icon::Text,
            MosaicLayerId::Annotation(_) => Icon::Points,
        }
    }

    fn layer_available(&self, id: MosaicLayerId) -> bool {
        match id {
            MosaicLayerId::TextLabels => true,
            MosaicLayerId::SegmentationGeoJson => self.seg_geojson.has_any_segpaths(),
            MosaicLayerId::Annotation(_) => true,
            MosaicLayerId::Channel(idx) => idx < self.channels.len(),
        }
    }

    fn layer_visible_value(&self, id: MosaicLayerId) -> Option<bool> {
        match id {
            MosaicLayerId::TextLabels => Some(self.show_text_labels),
            MosaicLayerId::SegmentationGeoJson => Some(self.seg_geojson.visible),
            MosaicLayerId::Annotation(id) => self
                .annotation_layers
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.visible),
            MosaicLayerId::Channel(idx) => self.channels.get(idx).map(|c| c.visible),
        }
    }

    fn set_layer_visible(&mut self, id: MosaicLayerId, visible: bool) {
        match id {
            MosaicLayerId::TextLabels => self.show_text_labels = visible,
            MosaicLayerId::SegmentationGeoJson => self.seg_geojson.visible = visible,
            MosaicLayerId::Annotation(id) => {
                if let Some(l) = self.annotation_layers.iter_mut().find(|l| l.id == id) {
                    l.visible = visible;
                }
            }
            MosaicLayerId::Channel(idx) => {
                if let Some(ch) = self.channels.get_mut(idx) {
                    ch.visible = visible;
                }
            }
        }
    }

    fn ui_layers(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("Layers");
        ui.separator();

        // Mosaic layers are mostly shared overlays plus globally visible channels. The layer list
        // is therefore less about per-item state and more about shared visibility/group controls.
        // Overlays master visibility toggle.
        let overlay_ids = self.overlay_layer_order.clone();
        let mut overlays_all = true;
        let mut overlays_none = true;
        for id in overlay_ids.iter().copied() {
            if !self.layer_available(id) {
                continue;
            }
            if let Some(v) = self.layer_visible_value(id) {
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
                for id in overlay_ids.iter().copied() {
                    if !self.layer_available(id) {
                        continue;
                    }
                    self.set_layer_visible(id, all);
                }
            }
            if ui.button("+ Annotations").clicked() {
                self.add_annotation_layer();
            }
        });

        let mut ann_members_by_group: HashMap<u64, Vec<u64>> = HashMap::new();
        for id in self.overlay_layer_order.iter().copied() {
            let MosaicLayerId::Annotation(aid) = id else {
                continue;
            };
            let Some(m) = self.layer_groups.annotation_members.get(&aid) else {
                continue;
            };
            if self
                .layer_groups
                .annotation_groups
                .iter()
                .any(|g| g.id == m.group_id)
            {
                ann_members_by_group
                    .entry(m.group_id)
                    .or_default()
                    .push(aid);
            }
        }
        let mut ann_headers_shown: HashSet<u64> = HashSet::new();
        let mut delete_ann_group: Option<u64> = None;

        for i in 0..self.overlay_layer_order.len() {
            let id = self.overlay_layer_order[i];

            if let MosaicLayerId::Annotation(aid) = id {
                if let Some(m) = self.layer_groups.annotation_members.get(&aid) {
                    let gid = m.group_id;
                    if self
                        .layer_groups
                        .annotation_groups
                        .iter()
                        .any(|g| g.id == gid)
                    {
                        if !ann_headers_shown.contains(&gid) {
                            ann_headers_shown.insert(gid);
                            let Some(group_idx) = self
                                .layer_groups
                                .annotation_groups
                                .iter()
                                .position(|g| g.id == gid)
                            else {
                                continue;
                            };
                            let members = ann_members_by_group
                                .get(&gid)
                                .map(|v| v.as_slice())
                                .unwrap_or(&[]);
                            let (
                                mut group_name,
                                mut group_expanded,
                                mut group_visible,
                                mut group_tint_rgb,
                                mut group_tint_strength,
                            ) = {
                                let g = &self.layer_groups.annotation_groups[group_idx];
                                (
                                    g.name.clone(),
                                    g.expanded,
                                    g.visible,
                                    g.tint_rgb,
                                    g.tint_strength,
                                )
                            };

                            let mut all = true;
                            let mut none = true;
                            for &mid in members {
                                let lid = MosaicLayerId::Annotation(mid);
                                if let Some(v) = self.layer_visible_value(lid) {
                                    all &= v;
                                    none &= !v;
                                }
                            }
                            let mixed = !members.is_empty() && !all && !none;

                            let header =
                                egui::collapsing_header::CollapsingState::load_with_default_open(
                                    ui.ctx(),
                                    ui.make_persistent_id(("mosaic-annotation-group", gid)),
                                    group_expanded,
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
                                                self.set_layer_visible(
                                                    MosaicLayerId::Annotation(mid),
                                                    set_all,
                                                );
                                            }
                                            group_visible = set_all;
                                        }
                                    });
                                    ui.add_space(4.0);
                                    ui.label(group_name.clone());
                                });
                            let open = header.is_open();
                            let (_toggle, _hdr, _body) = header.body(|ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Name");
                                    ui.text_edit_singleline(&mut group_name);
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Visible");
                                    ui.checkbox(&mut group_visible, "");
                                    if ui.button("Delete group").clicked() {
                                        delete_ann_group = Some(gid);
                                    }
                                });
                                ui.horizontal(|ui| {
                                    let mut has_tint = group_tint_rgb.is_some();
                                    if ui.checkbox(&mut has_tint, "Tint").changed() {
                                        if has_tint && group_tint_rgb.is_none() {
                                            group_tint_rgb = Some([255, 255, 255]);
                                        }
                                        if !has_tint {
                                            group_tint_rgb = None;
                                        }
                                    }
                                    if let Some(rgb) = group_tint_rgb.as_mut() {
                                        let mut c = egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
                                        if ui.color_edit_button_srgba(&mut c).changed() {
                                            *rgb = [c.r(), c.g(), c.b()];
                                        }
                                    }
                                });
                                ui.add(
                                    egui::Slider::new(&mut group_tint_strength, 0.0..=1.0)
                                        .text("Tint strength")
                                        .clamping(egui::SliderClamping::Always),
                                );
                            });
                            group_expanded = open;

                            let g = &mut self.layer_groups.annotation_groups[group_idx];
                            g.name = group_name;
                            g.expanded = group_expanded;
                            g.visible = group_visible;
                            g.tint_rgb = group_tint_rgb;
                            g.tint_strength = group_tint_strength;
                        }

                        let expanded = self
                            .layer_groups
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

            let available = self.layer_available(id);
            let selected = self.active_layer == id || self.selected_overlay_layers.contains(&id);
            let icon = self.layer_icon(id);
            let name = self.layer_display_name(id);
            let visible = self.layer_visible_value(id);
            let resp = layer_list::ui_layer_row(
                ui,
                ctx,
                &mut self.layer_drag,
                layer_list::LayerGroup::Overlays,
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
                if mods.shift && self.overlay_select_anchor_pos.is_some() {
                    let anchor = self.overlay_select_anchor_pos.unwrap_or(i);
                    let (a, b) = if anchor <= i {
                        (anchor, i)
                    } else {
                        (i, anchor)
                    };
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
                if !self.selected_overlay_layers.contains(&id) {
                    self.selected_overlay_layers.clear();
                    self.selected_overlay_layers.insert(id);
                    self.overlay_select_anchor_pos = Some(i);
                    self.set_active_layer(id);
                }
            }
            if let Some(v) = resp.visible_changed {
                self.set_layer_visible(id, v);
            }

            resp.row_response.context_menu(|ui| {
                let selected_annotations: Vec<u64> = self
                    .selected_overlay_layers
                    .iter()
                    .filter_map(|l| match l {
                        MosaicLayerId::Annotation(a) => Some(*a),
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
            self.layer_groups
                .annotation_groups
                .retain(|g| g.id != group_id);
            self.layer_groups
                .annotation_members
                .retain(|_k, m| m.group_id != group_id);
        }

        ui.separator();
        channels_panel::show(self, ui, ctx);

        layer_list::paint_drag_preview(ctx, self.layer_drag.as_ref(), |id| {
            self.layer_display_name(id)
        });
        let mut dropped: Option<(layer_list::LayerGroup, usize, usize)> = None;
        layer_list::finish_drag_if_released(ctx, &mut self.layer_drag, |group, from, to| {
            dropped = Some((group, from, to));
        });
        if let Some((group, from, to)) = dropped {
            match group {
                layer_list::LayerGroup::Overlays => {
                    layer_list::reorder_vec(&mut self.overlay_layer_order, from, to)
                }
                layer_list::LayerGroup::Channels => {
                    layer_list::reorder_vec(&mut self.channel_layer_order, from, to)
                }
            }
        }
    }

    fn open_group_layers_dialog_channels(&mut self, members: Vec<usize>) {
        let existing = self
            .layer_groups
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
            .layer_groups
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
            ctx.request_repaint();
        }
    }

    fn ui_memory_load_dialog(&mut self, ctx: &egui::Context) {
        if let Some((summary, requests)) =
            ui_pending_memory_action_dialog(ctx, &mut self.pending_memory_load)
        {
            self.execute_memory_load(summary, requests);
        }
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
                    &mut self.screenshot_settings.include_legend,
                    "Include legend (visible channels)",
                );
                ui.add_space(8.0);
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

    fn drain_screenshots(&mut self) {
        while let Ok(resp) = self.screenshot_worker.rx.try_recv() {
            match resp {
                crate::app_support::screenshot::ScreenshotWorkerResp::Saved { id, path, result } => {
                    if self.screenshot_in_flight == Some(id) {
                        self.screenshot_in_flight = None;
                    }
                    self.status = match result {
                        Ok(()) => format!("Saved screenshot -> {}", path.to_string_lossy()),
                        Err(err) => format!("Save screenshot failed: {err}"),
                    };
                }
            }
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
                            &self.layer_groups,
                            &ch.name,
                            ch.color_rgb,
                        )
                    })
                    .unwrap_or([255, 255, 255]);

                let existing_ids = self
                    .layer_groups
                    .channel_groups
                    .iter()
                    .map(|g| g.id)
                    .collect::<Vec<_>>();
                let gid = layer_groups::next_group_id(&existing_ids);
                self.layer_groups
                    .channel_groups
                    .push(crate::data::project_config::ProjectChannelGroup {
                        id: gid,
                        name,
                        expanded: true,
                        color_rgb: first_color,
                    });
                for idx in indices {
                    if let Some(ch) = self.channels.get(idx) {
                        self.layer_groups.channel_members.insert(
                            ch.name.clone(),
                            crate::data::project_config::ProjectChannelGroupMember {
                                group_id: gid,
                                inherit_color: true,
                            },
                        );
                    }
                }
            }
            GroupLayersTarget::Annotations(layer_ids) => {
                let existing_ids = self
                    .layer_groups
                    .annotation_groups
                    .iter()
                    .map(|g| g.id)
                    .collect::<Vec<_>>();
                let gid = layer_groups::next_group_id(&existing_ids);
                self.layer_groups.annotation_groups.push(
                    crate::data::project_config::ProjectAnnotationGroup {
                        id: gid,
                        name,
                        expanded: true,
                        visible: true,
                        tint_rgb: None,
                        tint_strength: 0.35,
                    },
                );
                for id in layer_ids {
                    self.layer_groups.annotation_members.insert(
                        id,
                        crate::data::project_config::ProjectAnnotationGroupMember {
                            group_id: gid,
                            inherit_tint: true,
                        },
                    );
                }
            }
        }
    }

    fn ui_project(&mut self, ui: &mut egui::Ui) {
        if let Some(action) = self.project_space.ui(ui, None) {
            match action {
                crate::project::ProjectSpaceAction::Open(roi) => {
                    self.pending_request = Some(MosaicRequest::OpenProjectRoi(roi));
                }
                crate::project::ProjectSpaceAction::OpenMosaic(rois) => {
                    self.pending_request = Some(MosaicRequest::OpenProjectMosaic(rois));
                }
                crate::project::ProjectSpaceAction::OpenRemoteDialog => {
                    self.pending_request = Some(MosaicRequest::OpenRemoteDialog);
                }
            }
        }
    }

    fn ui_properties(&mut self, ui: &mut egui::Ui) {
        if let Some(group_id) = self.selected_channel_group_id {
            self.ui_group_contrast(ui, group_id);
            return;
        }
        match self.active_layer {
            MosaicLayerId::Channel(idx) => {
                self.selected_channel = idx.min(self.channels.len().saturating_sub(1));
                self.ui_contrast(ui);
            }
            MosaicLayerId::TextLabels => {
                ui.heading("Text labels");
                ui.separator();
                ui.checkbox(&mut self.show_text_labels, "Visible");
                ui.add_enabled_ui(self.show_text_labels, |ui| {
                    let mut available_columns = vec!["id".to_string()];
                    available_columns.extend(self.metadata_columns.iter().cloned());

                    ui.horizontal(|ui| {
                        let mut add_clicked = false;
                        if ui.button("+ Add label line").clicked() {
                            add_clicked = true;
                        }
                        if ui
                            .add_enabled(
                                !self.label_columns.is_empty(),
                                egui::Button::new("Clear lines"),
                            )
                            .clicked()
                        {
                            self.label_columns.clear();
                        }
                        if add_clicked {
                            let next = available_columns
                                .iter()
                                .find(|col| !self.label_columns.contains(*col))
                                .cloned()
                                .or_else(|| available_columns.first().cloned());
                            if let Some(next) = next {
                                self.label_columns.push(next);
                            }
                        }
                    });

                    if self.label_columns.is_empty() {
                        ui.label("No label lines selected.");
                    }

                    let mut remove_idx = None;
                    for (idx, column) in self.label_columns.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(format!("Line {}", idx + 1));
                            egui::ComboBox::from_id_salt(("mosaic-label-line", idx))
                                .selected_text(column.clone())
                                .show_ui(ui, |ui| {
                                    for col in &available_columns {
                                        ui.selectable_value(column, col.clone(), col);
                                    }
                                });
                            if ui.small_button("Remove").clicked() {
                                remove_idx = Some(idx);
                            }
                        });
                    }
                    if let Some(idx) = remove_idx {
                        self.label_columns.remove(idx);
                    }
                });
            }
            MosaicLayerId::SegmentationGeoJson => {
                let have_any_seg = self.seg_geojson.has_any_segpaths();
                let zoom_selected = self.seg_geojson.ui_left_panel(ui, have_any_seg);
                if zoom_selected
                    && let (Some(bounds), Some(viewport)) = (
                        self.seg_geojson.selected_bounds_world(),
                        self.last_canvas_rect,
                    )
                {
                    self.camera.fit_to_world_rect(viewport, bounds);
                }
                if have_any_seg {
                    let (loaded, loading, total) = self.seg_geojson.loaded_stats();
                    ui.label(format!(
                        "GeoJSON: {loaded}/{total} loaded ({loading} loading)"
                    ));
                    let missing = self.seg_geojson.last_missing_bins();
                    if self.seg_geojson.visible && missing > 0 {
                        ui.label(format!("GeoJSON bins: {missing} pending GPU uploads"));
                    }
                }
            }
            MosaicLayerId::Annotation(id) => {
                let Some(idx) = self.annotation_layers.iter().position(|l| l.id == id) else {
                    ui.label("Annotation layer not found.");
                    return;
                };
                ui.heading(self.annotation_layers[idx].name.clone());
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Name");
                    ui.text_edit_singleline(&mut self.annotation_layers[idx].name);
                });
                ui.separator();

                let mut selected_group: Option<u64> = self
                    .layer_groups
                    .annotation_members
                    .get(&id)
                    .map(|m| m.group_id)
                    .filter(|gid| {
                        self.layer_groups
                            .annotation_groups
                            .iter()
                            .any(|g| g.id == *gid)
                    });
                let mut groups_changed = false;

                ui.horizontal(|ui| {
                    ui.label("Group");
                    egui::ComboBox::from_id_salt(("mosaic-annotation-group-select", id))
                        .selected_text(
                            selected_group
                                .and_then(|gid| {
                                    self.layer_groups
                                        .annotation_groups
                                        .iter()
                                        .find(|g| g.id == gid)
                                })
                                .map(|g| g.name.as_str())
                                .unwrap_or("(none)"),
                        )
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut selected_group, None, "(none)");
                            for g in &self.layer_groups.annotation_groups {
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
                        let existing = self
                            .layer_groups
                            .annotation_groups
                            .iter()
                            .map(|g| g.id)
                            .collect::<Vec<_>>();
                        let id2 = layer_groups::next_group_id(&existing);
                        self.layer_groups.annotation_groups.push(
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

                let have_member = self.layer_groups.annotation_members.get(&id).is_some();
                if selected_group.is_none() && have_member {
                    self.layer_groups.annotation_members.remove(&id);
                    groups_changed = true;
                } else if let Some(gid) = selected_group {
                    match self.layer_groups.annotation_members.get_mut(&id) {
                        Some(m) => {
                            if m.group_id != gid {
                                m.group_id = gid;
                                groups_changed = true;
                            }
                        }
                        None => {
                            self.layer_groups.annotation_members.insert(
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
                    let mut inherit_tint = self
                        .layer_groups
                        .annotation_members
                        .get(&id)
                        .map(|m| m.inherit_tint)
                        .unwrap_or(true);
                    ui.horizontal(|ui| {
                        if ui
                            .checkbox(&mut inherit_tint, "Inherit group tint")
                            .changed()
                        {
                            if let Some(m) = self.layer_groups.annotation_members.get_mut(&id) {
                                m.inherit_tint = inherit_tint;
                                groups_changed = true;
                            }
                        }
                    });

                    if let Some(group) = self
                        .layer_groups
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
                let changed = self.annotation_layers[idx].ui_properties(ui);
                if changed {
                    // Ensure repaint (GL uniforms changed).
                }
                ui.separator();
                if ui.button("Delete layer").clicked() {
                    let layer_id = MosaicLayerId::Annotation(id);
                    self.annotation_layers.remove(idx);
                    self.overlay_layer_order.retain(|l| *l != layer_id);
                    if self.active_layer == layer_id {
                        self.active_layer = self
                            .channel_layer_order
                            .first()
                            .copied()
                            .map(MosaicLayerId::Channel)
                            .unwrap_or(MosaicLayerId::TextLabels);
                    }
                }
                if groups_changed {
                    ui.ctx().request_repaint();
                }
            }
        }
    }

    fn ui_layout(&mut self, ui: &mut egui::Ui, _ctx: &egui::Context) {
        ui.heading("Arrange");
        ui.label("Sort and optionally group ROIs by samplesheet columns.");
        ui.add_space(6.0);

        let mut changed = false;
        egui::ComboBox::from_label("Group by")
            .selected_text(if self.group_by.is_empty() {
                "(none)".to_string()
            } else {
                self.group_by.clone()
            })
            .show_ui(ui, |ui| {
                changed |= ui
                    .selectable_value(&mut self.group_by, String::new(), "(none)")
                    .changed();
                for col in &self.metadata_columns {
                    changed |= ui
                        .selectable_value(&mut self.group_by, col.clone(), col)
                        .changed();
                }
            });
        ui.horizontal(|ui| {
            ui.add_enabled_ui(!self.group_by.is_empty(), |ui| {
                changed |= ui
                    .checkbox(&mut self.show_group_labels, "Show group labels")
                    .changed();
                ui.label("Gap");
                changed |= ui
                    .add(egui::DragValue::new(&mut self.group_gap).speed(5.0))
                    .changed();
            });
        });

        ui.add_space(8.0);
        egui::ComboBox::from_label("Layout")
            .selected_text(self.layout_mode.label())
            .show_ui(ui, |ui| {
                changed |= ui
                    .selectable_value(
                        &mut self.layout_mode,
                        MosaicLayoutMode::FitCells,
                        MosaicLayoutMode::FitCells.label(),
                    )
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut self.layout_mode,
                        MosaicLayoutMode::NativePixels,
                        MosaicLayoutMode::NativePixels.label(),
                    )
                    .changed();
            });

        ui.add_space(8.0);
        egui::ComboBox::from_label("Sort by")
            .selected_text(self.sort_by.clone())
            .show_ui(ui, |ui| {
                changed |= ui
                    .selectable_value(&mut self.sort_by, "id".to_string(), "id")
                    .changed();
                for col in &self.metadata_columns {
                    changed |= ui
                        .selectable_value(&mut self.sort_by, col.clone(), col)
                        .changed();
                }
            });

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            changed |= ui
                .checkbox(&mut self.sort_secondary_enabled, "Then by")
                .changed();
            ui.add_enabled_ui(self.sort_secondary_enabled, |ui| {
                egui::ComboBox::from_id_salt("sort-by-secondary")
                    .selected_text(self.sort_by_secondary.clone())
                    .show_ui(ui, |ui| {
                        changed |= ui
                            .selectable_value(&mut self.sort_by_secondary, "id".to_string(), "id")
                            .changed();
                        for col in &self.metadata_columns {
                            changed |= ui
                                .selectable_value(&mut self.sort_by_secondary, col.clone(), col)
                                .changed();
                        }
                    });
            });
        });

        if ui.button("Apply sort").clicked() || changed {
            self.apply_sort_and_layout();
        }

        ui.add_space(8.0);
        ui.label(format!(
            "Layout: {}, {} columns",
            self.layout_mode.label(),
            self.grid_cols
        ));
    }

    fn ui_memory(&mut self, ui: &mut egui::Ui) {
        ui_memory_overview(
            ui,
            "Manually pin selected OME-Zarr channels and levels in CPU RAM.",
            Some(("Pinned total", self.pinned_levels.total_loaded_bytes())),
            self.system_memory.as_ref(),
        );
        ui.add_space(6.0);

        let rows = self
            .channel_layer_order
            .iter()
            .filter_map(|&gid| {
                self.channels.get(gid).map(|ch| MemoryChannelRow {
                    id: gid,
                    label: if ch.visible {
                        format!("{} (visible)", ch.name)
                    } else {
                        ch.name.clone()
                    },
                    visible: ch.visible,
                })
            })
            .collect::<Vec<_>>();
        ui_memory_channel_selector(
            ui,
            "mosaic-memory-channel-list",
            &rows,
            &mut self.memory_selected_channels,
        );
        ui.separator();

        let selected_global_channels = self.selected_memory_global_channels();
        let selected_channel_count = selected_global_channels.len();

        let max_levels = self
            .sources
            .iter()
            .map(|src| src.levels.len())
            .max()
            .unwrap_or(0);
        if max_levels > 0 {
            ui.label("All ROIs");
            egui::Grid::new("mosaic-memory-all-grid")
                .num_columns(5)
                .striped(true)
                .show(ui, |ui| {
                    ui.strong("Level");
                    ui.strong("Eligible");
                    ui.strong("RAM");
                    ui.strong("State");
                    ui.strong("Action");
                    ui.end_row();

                    for level_idx in 0..max_levels {
                        let mut eligible = 0usize;
                        let mut loaded = 0usize;
                        let mut loading = 0usize;
                        let mut failed = 0usize;
                        let mut bytes = 0u64;

                        for item in &self.items {
                            let Some(source) = self.sources.get(item.id) else {
                                continue;
                            };
                            if source.levels.get(level_idx).is_none() {
                                continue;
                            }
                            let estimate = estimate_level_ram_bytes_for_channels(
                                source,
                                level_idx,
                                Some(&selected_global_channels),
                            )
                            .unwrap_or(0);
                            if estimate == 0 {
                                continue;
                            }
                            eligible += 1;
                            bytes = bytes.saturating_add(estimate);
                            match self.pinned_levels.status(item.id, level_idx) {
                                MosaicPinnedLevelStatus::Unloaded => {}
                                MosaicPinnedLevelStatus::Loading => loading += 1,
                                MosaicPinnedLevelStatus::Loaded { .. } => loaded += 1,
                                MosaicPinnedLevelStatus::Failed(_) => failed += 1,
                            }
                        }

                        if eligible == 0 {
                            continue;
                        }

                        ui.label(level_idx.to_string());
                        ui.label(format!("{eligible} ROI(s)"));
                        let risk = self.memory_risk(bytes);
                        let risk_text = match risk.as_ref().map(|r| r.level) {
                            Some(MemoryRiskLevel::Danger) => " danger",
                            Some(MemoryRiskLevel::Warning) => " warning",
                            None => "",
                        };
                        ui.label(format!("{}{}", format_bytes(bytes), risk_text));
                        if loading > 0 {
                            ui.label(format!("Loading {loading}, loaded {loaded}/{eligible}"));
                        } else if loaded == eligible {
                            ui.label("Loaded for all");
                        } else if loaded > 0 || failed > 0 {
                            ui.label(format!("Loaded {loaded}/{eligible}, failed {failed}"));
                        } else {
                            ui.label("Not loaded");
                        }
                        ui.horizontal(|ui| {
                            if ui
                                .add_enabled(
                                    selected_channel_count > 0 && eligible > 0 && loading == 0,
                                    egui::Button::new("Load all"),
                                )
                                .clicked()
                            {
                                let (requests, requested_bytes) = self
                                    .memory_load_requests_for_all_rois(
                                        level_idx,
                                        &selected_global_channels,
                                    );
                                let count = requests.len();
                                self.start_memory_load(
                                    format!(
                                        "Loading {} channel(s) from level {level_idx} into RAM for {count} ROI(s)",
                                        selected_channel_count
                                    ),
                                    requests,
                                    requested_bytes,
                                );
                            }
                            if ui
                                .add_enabled(
                                    loaded > 0 || loading > 0 || failed > 0,
                                    egui::Button::new("Unload all"),
                                )
                                .clicked()
                            {
                                let count = self.unload_level_for_all_rois(level_idx);
                                self.status =
                                    format!("Unloaded level {level_idx} for {count} ROI(s)");
                            }
                        });
                        ui.end_row();
                    }
                });
            ui.separator();
        }

        let Some(item) = self.focused_item() else {
            ui.label("No focused ROI.");
            return;
        };
        let dataset_id = item.id;
        let item_dims = item.dataset.dims.clone();
        let sample_id = item.sample_id.clone();
        let levels = item.dataset.levels.clone();

        ui.label(format!("Focused ROI: {sample_id}"));
        ui.label("Loading is manual. The app estimates RAM usage but does not enforce a system-memory limit.");
        ui.separator();

        let Some(source) = self.sources.get(dataset_id).cloned() else {
            ui.label("Missing mosaic source metadata.");
            return;
        };

        egui::Grid::new(("mosaic-memory-grid", dataset_id))
            .num_columns(5)
            .striped(true)
            .show(ui, |ui| {
                ui.strong("Level");
                ui.strong("Shape");
                ui.strong("RAM");
                ui.strong("State");
                ui.strong("Action");
                ui.end_row();

                for (level_idx, level) in levels.iter().enumerate() {
                    let shape_y = level.shape.get(item_dims.y).copied().unwrap_or(0);
                    let shape_x = level.shape.get(item_dims.x).copied().unwrap_or(0);
                    let channels = item_dims
                        .c
                        .and_then(|c| level.shape.get(c).copied())
                        .unwrap_or(1);
                    let estimate = estimate_level_ram_bytes_for_channels(
                        &source,
                        level_idx,
                        Some(&selected_global_channels),
                    )
                    .unwrap_or(0);
                    let status = self.pinned_levels.status(dataset_id, level_idx);

                    ui.label(level_idx.to_string());
                    ui.label(format!("{channels} x {shape_y} x {shape_x}"));
                    let risk = self.memory_risk(estimate);
                    let risk_text = match risk.as_ref().map(|r| r.level) {
                        Some(MemoryRiskLevel::Danger) => " danger",
                        Some(MemoryRiskLevel::Warning) => " warning",
                        None => "",
                    };
                    ui.label(format!("{}{}", format_bytes(estimate), risk_text));
                    if estimate == 0 {
                        ui.label("No selected channels");
                    } else {
                        match &status {
                            MosaicPinnedLevelStatus::Unloaded => {
                                ui.label("Not loaded");
                            }
                            MosaicPinnedLevelStatus::Loading => {
                                ui.label("Loading");
                            }
                            MosaicPinnedLevelStatus::Loaded {
                                bytes,
                                channels_loaded,
                            } => {
                                ui.label(format!(
                                    "Loaded ({}; {} ch)",
                                    format_bytes(*bytes),
                                    channels_loaded
                                ));
                            }
                            MosaicPinnedLevelStatus::Failed(err) => {
                                ui.colored_label(
                                    ui.visuals().warn_fg_color,
                                    format!("Failed: {err}"),
                                );
                            }
                        }
                    }

                    ui.horizontal(|ui| {
                        if ui
                            .add_enabled(
                                estimate > 0 && !matches!(status, MosaicPinnedLevelStatus::Loading),
                                egui::Button::new("Load"),
                            )
                            .clicked()
                        {
                            if let Some((request, requested_bytes)) = self
                                .memory_load_request_for_dataset(
                                    dataset_id,
                                    source.clone(),
                                    level_idx,
                                    &selected_global_channels,
                                )
                            {
                                self.start_memory_load(
                                    format!(
                                        "Loading {} channel(s) from ROI '{}' level {} into RAM",
                                        selected_channel_count, sample_id, level_idx
                                    ),
                                    vec![request],
                                    requested_bytes,
                                );
                            }
                        }
                        if ui
                            .add_enabled(
                                !matches!(status, MosaicPinnedLevelStatus::Unloaded),
                                egui::Button::new("Unload"),
                            )
                            .clicked()
                        {
                            self.pinned_levels.unload(dataset_id, level_idx);
                            self.status = format!(
                                "Unloaded ROI '{}' level {} from RAM",
                                sample_id, level_idx
                            );
                        }
                    });
                    ui.end_row();
                }
            });
    }

    fn ui_contrast(&mut self, ui: &mut egui::Ui) {
        ui.heading("Contrast (global)");
        if self.channels.is_empty() {
            ui.label("No channels.");
            return;
        }

        let mut changed_channel = false;
        egui::ComboBox::from_label("Channel")
            .selected_text(
                self.channels
                    .get(self.selected_channel)
                    .map(|c| c.name.as_str())
                    .unwrap_or("-"),
            )
            .show_ui(ui, |ui| {
                let order = self.channel_layer_order.clone();
                for idx in order.into_iter() {
                    let Some(ch) = self.channels.get(idx) else {
                        continue;
                    };
                    changed_channel |= ui
                        .selectable_value(&mut self.selected_channel, idx, &ch.name)
                        .changed();
                }
            });

        let abs_max = self.abs_max.max(1.0);
        let Some(sel) = self.channels.get(self.selected_channel).cloned() else {
            return;
        };

        let selected_name = sel.name.clone();
        let mut selected_group: Option<u64> = self
            .layer_groups
            .channel_members
            .get(selected_name.as_str())
            .map(|m| m.group_id)
            .filter(|gid| {
                self.layer_groups
                    .channel_groups
                    .iter()
                    .any(|g| g.id == *gid)
            });
        let mut groups_changed = false;

        ui.horizontal(|ui| {
            ui.label("Group");
            egui::ComboBox::from_id_salt("mosaic-channel-group-select")
                .selected_text(
                    selected_group
                        .and_then(|gid| {
                            self.layer_groups
                                .channel_groups
                                .iter()
                                .find(|g| g.id == gid)
                        })
                        .map(|g| g.name.as_str())
                        .unwrap_or("(none)"),
                )
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut selected_group, None, "(none)");
                    for g in &self.layer_groups.channel_groups {
                        ui.selectable_value(&mut selected_group, Some(g.id), g.name.clone());
                    }
                });
            if ui
                .button("+ Group")
                .on_hover_text("Create a new group")
                .clicked()
            {
                let existing = self
                    .layer_groups
                    .channel_groups
                    .iter()
                    .map(|g| g.id)
                    .collect::<Vec<_>>();
                let id = layer_groups::next_group_id(&existing);
                self.layer_groups
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

        // Apply membership.
        let have_member = self
            .layer_groups
            .channel_members
            .get(selected_name.as_str())
            .is_some();
        if selected_group.is_none() && have_member {
            self.layer_groups
                .channel_members
                .remove(selected_name.as_str());
            groups_changed = true;
        } else if let Some(gid) = selected_group {
            match self
                .layer_groups
                .channel_members
                .get_mut(selected_name.as_str())
            {
                Some(m) => {
                    if m.group_id != gid {
                        m.group_id = gid;
                        groups_changed = true;
                    }
                }
                None => {
                    self.layer_groups.channel_members.insert(
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
        if let Some(m) = self
            .layer_groups
            .channel_members
            .get(selected_name.as_str())
        {
            inherit_group_color = m.inherit_color;
        }
        if let Some(gid) = selected_group {
            ui.horizontal(|ui| {
                if ui
                    .checkbox(&mut inherit_group_color, "Inherit group color")
                    .changed()
                {
                    if let Some(m) = self
                        .layer_groups
                        .channel_members
                        .get_mut(selected_name.as_str())
                    {
                        m.inherit_color = inherit_group_color;
                        groups_changed = true;
                    }
                }
                if inherit_group_color {
                    if let Some(group) = self
                        .layer_groups
                        .channel_groups
                        .iter_mut()
                        .find(|g| g.id == gid)
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

        let allow_channel_color = selected_group.is_none() || !inherit_group_color;
        if let Some(ch) = self.channels.get_mut(self.selected_channel) {
            ui.horizontal(|ui| {
                ui.label(if allow_channel_color {
                    "Color"
                } else {
                    "Color (override)"
                });
                ui.add_enabled_ui(allow_channel_color, |ui| {
                    let mut c =
                        egui::Color32::from_rgb(ch.color_rgb[0], ch.color_rgb[1], ch.color_rgb[2]);
                    if ui.color_edit_button_srgba(&mut c).changed() {
                        ch.color_rgb = [c.r(), c.g(), c.b()];
                    }
                });
            });
        }
        if groups_changed {
            ui.ctx().request_repaint();
        }
        let window = sel.window.unwrap_or((0.0, abs_max));
        let out = contrast::ui_contrast_window(
            ui,
            abs_max,
            window,
            contrast::ContrastUiOptions {
                show_nudge_buttons: false,
                set_max_button_label: "Set Max -> All",
            },
        );
        let (lo, hi) = out.window;

        if out.set_max_all_clicked {
            for dst in &mut self.channels {
                let (mut dlo, _) = dst.window.unwrap_or((0.0, abs_max));
                dlo = dlo.clamp(0.0, abs_max);
                let dhi = hi.clamp(0.0, abs_max);
                let dlo = if dhi <= dlo {
                    (dhi - 1.0).clamp(0.0, abs_max)
                } else {
                    dlo
                };
                dst.window = Some((dlo, dhi));
            }
            return;
        }

        if out.limits_touched || changed_channel {
            if let Some(dst) = self.channels.get_mut(self.selected_channel) {
                dst.window = Some((lo, hi));
            }
        }
    }

    fn ui_canvas(&mut self, ui: &mut egui::Ui, _ctx: &egui::Context) {
        let available = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(available, egui::Sense::drag());
        self.last_canvas_rect = Some(rect);
        ui.painter()
            .rect_filled(rect, 0.0, egui::Color32::from_gray(10));

        // Zoom + pan
        if response.hovered() {
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            let pinch = ui.input(|i| i.zoom_delta());
            if scroll != 0.0 {
                if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
                    let factor = (scroll * 0.0015).exp();
                    self.camera.zoom_about_screen_point(rect, pos, factor);
                }
            }
            if pinch != 1.0 {
                if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
                    self.camera.zoom_about_screen_point(rect, pos, pinch);
                }
            }
        }
        if response.dragged_by(egui::PointerButton::Primary) {
            let delta = ui.input(|i| i.pointer.delta());
            self.camera.pan_by_screen_delta(delta);
        }

        if response.double_clicked() {
            if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
                let world = self.camera.screen_to_world(pos, rect);
                if let Some(it) = self.items.iter().find(|it| item_rect(it).contains(world)) {
                    self.focused_core_id = Some(it.id);
                    self.camera.fit_to_world_rect(rect, item_rect(it));
                }
            }
        }

        if self.active_layer == MosaicLayerId::SegmentationGeoJson
            && self.seg_geojson.visible
            && response.clicked_by(egui::PointerButton::Primary)
            && let Some(pos) = ui.input(|i| i.pointer.hover_pos())
            && rect.contains(pos)
        {
            let world = self.camera.screen_to_world(pos, rect);
            let mods = ui.input(|i| i.modifiers);
            if let Some(it) = self.items.iter().find(|it| item_rect(it).contains(world)) {
                self.seg_geojson
                    .select_at(it.id, world, &self.camera, mods.shift, mods.command);
            } else if !mods.shift && !mods.command {
                self.seg_geojson.clear_selection();
            }
        }

        // Draw
        //
        // The mosaic renderer works in two passes each frame:
        // 1. ensure every visible ROI has at least coarse coverage, so the whole mosaic appears
        // 2. spend the remaining request budget refining ROIs near the viewport center/current focus
        //
        // This prevents a zoomed-out mosaic from showing "holes" while still biasing bandwidth
        // toward the area the user is inspecting.
        let visible_world = visible_world_rect(&self.camera, rect);
        let prev_visible_world = self.last_visible_world.unwrap_or(visible_world);
        let channels_draw = self.visible_channel_draws();
        self.sync_tile_request_generation(visible_world, rect, &channels_draw);
        self.drain_raw_tiles();
        let request_generation = self.tile_request_generation;
        let mut draws: Vec<MosaicTileDraw> = Vec::new();

        let mut sent = 0usize;
        let max_requests_per_frame = 2048usize;
        let max_coarse_tiles_per_item_per_frame = 2usize;

        // Phase A: ensure coarsest level tiles for all items (so everything appears when zoomed out).
        for it in &self.items {
            let _ = self.collect_draws_and_requests_for_item(
                it,
                visible_world,
                Some(prev_visible_world),
                rect,
                Phase::CoarseOnly,
                request_generation,
                None,
                None,
                None,
                None,
                None,
                &mut draws,
                &mut sent,
                max_requests_per_frame,
                max_coarse_tiles_per_item_per_frame,
            );
        }
        // Phase B: refine near the current zoom.
        let refine_order = self.refine_item_order(visible_world);
        for idx in refine_order {
            if sent >= max_requests_per_frame {
                break;
            }
            let (id, target, ceiling) = {
                let Some(it) = self.items.get(idx) else {
                    continue;
                };
                let prev = self
                    .last_target_level_by_dataset_id
                    .get(it.id)
                    .copied()
                    .flatten();
                let prev_ceiling = self
                    .fallback_ceiling_by_dataset_id
                    .get(it.id)
                    .copied()
                    .flatten();
                let prev_floor = self
                    .zoom_out_floor_by_dataset_id
                    .get(it.id)
                    .copied()
                    .flatten();
                let prev_floor_until = self
                    .zoom_out_floor_until_by_dataset_id
                    .get(it.id)
                    .copied()
                    .flatten();
                let prev_floor_world = self
                    .zoom_out_floor_world_by_dataset_id
                    .get(it.id)
                    .copied()
                    .flatten();
                let (target, ceiling, floor, floor_until, floor_world) = self
                    .collect_draws_and_requests_for_item(
                        it,
                        visible_world,
                        Some(prev_visible_world),
                        rect,
                        Phase::Refine,
                        request_generation,
                        prev,
                        prev_ceiling,
                        prev_floor,
                        prev_floor_until,
                        prev_floor_world,
                        &mut draws,
                        &mut sent,
                        max_requests_per_frame,
                        max_coarse_tiles_per_item_per_frame,
                    );
                if let Some(dst) = self.zoom_out_floor_by_dataset_id.get_mut(it.id) {
                    *dst = floor;
                }
                if let Some(dst) = self.zoom_out_floor_until_by_dataset_id.get_mut(it.id) {
                    *dst = floor_until;
                }
                if let Some(dst) = self.zoom_out_floor_world_by_dataset_id.get_mut(it.id) {
                    *dst = floor_world;
                }
                (it.id, target, ceiling)
            };
            if let Some(t) = target {
                if let Some(dst) = self.last_target_level_by_dataset_id.get_mut(id) {
                    *dst = Some(t);
                }
            }
            if let Some(c) = ceiling {
                if let Some(dst) = self.fallback_ceiling_by_dataset_id.get_mut(id) {
                    *dst = Some(c);
                }
            }
        }
        self.last_visible_world = Some(visible_world);

        // If the user navigated quickly, we may have in-flight requests for tiles that are no longer
        // relevant to the current view. Prune them so we can actually go idle.
        let mut keep: HashSet<MosaicRawTileKey> =
            HashSet::with_capacity(draws.len() * channels_draw.len());
        for td in &draws {
            for ch in &channels_draw {
                keep.insert(MosaicRawTileKey {
                    dataset_id: td.dataset_id,
                    level: td.level,
                    tile_y: td.tile_y,
                    tile_x: td.tile_x,
                    channel: ch.index,
                });
            }
        }
        self.tiles_gl.prune_in_flight(&keep);

        let tiles_gl = self.tiles_gl.clone();
        let sources = Arc::clone(&self.sources);
        let cb = egui_glow::CallbackFn::new(move |info, painter| {
            tiles_gl.paint(info, painter, &sources, &draws, &channels_draw);
        });
        ui.painter().add(egui::PaintCallback {
            rect,
            callback: Arc::new(cb),
        });

        let screenshot = self.screenshot_pending.take();
        let screenshot_active = screenshot.is_some();

        // Overlay layers (in the user-controlled order).
        self.seg_geojson_pending_visible = false;
        let overlay_order = self.overlay_layer_order.clone();

        let mut visible_rois: Vec<(String, egui::Vec2, f32)> = Vec::new();
        visible_rois.reserve(self.items.len().min(256));
        for it in &self.items {
            let r = item_rect(it);
            if r.intersects(visible_world) {
                visible_rois.push((it.sample_id.clone(), it.offset, it.scale));
            }
        }

        for layer in overlay_order.into_iter().rev() {
            match layer {
                MosaicLayerId::Channel(_) => {}
                MosaicLayerId::TextLabels => {
                    if self.show_text_labels {
                        self.draw_text_labels(ui, rect);
                    }
                }
                MosaicLayerId::SegmentationGeoJson => {
                    if self.seg_geojson.visible && self.seg_geojson.has_any_segpaths() {
                        let mut visible_items: Vec<(usize, egui::Rect, egui::Vec2, f32)> =
                            Vec::new();
                        visible_items.reserve(self.items.len().min(128));
                        for it in &self.items {
                            let r = item_rect(it);
                            if r.intersects(visible_world) {
                                visible_items.push((it.id, r, it.offset, it.scale));
                            }
                        }
                        self.seg_geojson_pending_visible = self
                            .seg_geojson
                            .ensure_visible_items_loading(&visible_items, visible_world);
                        let pending_gpu = self.seg_geojson.paint(
                            ui,
                            &self.camera,
                            rect,
                            visible_world,
                            &visible_items,
                        );
                        self.seg_geojson_pending_visible |= pending_gpu;
                    }
                }
                MosaicLayerId::Annotation(id) => {
                    if let Some(layer) = self.annotation_layers.iter_mut().find(|l| l.id == id) {
                        let group_tint =
                            layer_groups::effective_annotation_tint(&self.layer_groups, id);
                        layer.draw_mosaic(
                            ui,
                            rect,
                            self.camera.center_world_lvl0,
                            self.camera.zoom_screen_per_lvl0_px,
                            &visible_rois,
                            group_tint,
                            true,
                        );
                        if self.active_layer == MosaicLayerId::Annotation(id) {
                            if let Some(pointer) = ui.input(|i| i.pointer.hover_pos()) {
                                if rect.contains(pointer) {
                                    let world = self.camera.screen_to_world(pointer, rect);
                                    if let Some(it) =
                                        self.items.iter().find(|it| item_rect(it).contains(world))
                                    {
                                        layer.maybe_hover_tooltip(
                                            ui.ctx(),
                                            rect,
                                            world,
                                            self.camera.zoom_screen_per_lvl0_px,
                                            it.sample_id.as_str(),
                                            it.offset,
                                            it.scale,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if !screenshot_active
            && self.active_layer == MosaicLayerId::SegmentationGeoJson
            && self.seg_geojson.visible
            && let Some(pointer) = ui.input(|i| i.pointer.hover_pos())
            && rect.contains(pointer)
        {
            let world = self.camera.screen_to_world(pointer, rect);
            if let Some(it) = self.items.iter().find(|it| item_rect(it).contains(world))
                && let Some(lines) = self.seg_geojson.hover_tooltip(it.id, world, &self.camera)
            {
                crate::ui::tooltip::show_tooltip_at_pointer(
                    ui.ctx(),
                    egui::Id::new(("mosaic-segmentation-object-tooltip", it.id)),
                    |ui| {
                        for line in lines {
                            ui.label(line);
                        }
                    },
                );
            }
        }

        if self.show_group_labels && !self.group_by.is_empty() {
            self.draw_group_labels(ui, rect);
        }

        let tile_loading_count = self.tiles_gl.loading_tile_count_for(&keep);

        if !screenshot_active {
            let hud = format!(
                "zoom {:.5} center ({:.0}, {:.0})",
                self.camera.zoom_screen_per_lvl0_px,
                self.camera.center_world_lvl0.x,
                self.camera.center_world_lvl0.y
            );
            canvas_overlays::paint_hud(ui, rect, hud);
        }

        if screenshot
            .as_ref()
            .is_some_and(|spec| spec.settings.include_legend)
        {
            let mut entries: Vec<(egui::Color32, String)> = Vec::new();
            for idx in self.channel_layer_order.iter().copied() {
                let Some(ch) = self.channels.get(idx) else {
                    continue;
                };
                if !ch.visible {
                    continue;
                }
                let rgb = layer_groups::effective_channel_color_rgb(
                    &self.layer_groups,
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
                    .map(|spec| spec.settings.legend_scale)
                    .unwrap_or(1.0),
            );
        }

        if !screenshot_active {
            let spinner_text = if self.show_tile_debug && tile_loading_count > 0 {
                Some(format!("{tile_loading_count} tiles"))
            } else {
                None
            };
            canvas_overlays::paint_spinner(
                ui,
                rect,
                tile_loading_count > 0
                    || self.seg_geojson.is_busy()
                    || self.seg_geojson_pending_visible,
                spinner_text.as_deref(),
            );
        }

        if let Some(spec) = screenshot {
            let tx = self.screenshot_worker.tx.clone();
            let id = spec.id;
            let path = spec.path.clone();
            let cb = egui_glow::CallbackFn::new(move |info, painter| {
                let viewport = info.viewport_in_pixels();
                let x_px = viewport.left_px;
                let y_px = viewport.from_bottom_px;
                let w_px = viewport.width_px;
                let h_px = viewport.height_px;

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

    fn visible_channel_draws(&self) -> Vec<ChannelDraw> {
        let mut out = Vec::new();
        for gid in self.channel_layer_order.iter().copied() {
            let Some(gch) = self.channels.get(gid) else {
                continue;
            };
            if !gch.visible {
                continue;
            }
            let rgb = layer_groups::effective_channel_color_rgb(
                &self.layer_groups,
                gch.name.as_str(),
                gch.color_rgb,
            );
            out.push(ChannelDraw {
                index: gid as u64,
                color_rgb: [
                    rgb[0] as f32 / 255.0,
                    rgb[1] as f32 / 255.0,
                    rgb[2] as f32 / 255.0,
                ],
                window: gch.window.unwrap_or((0.0, self.abs_max)),
            });
        }
        out
    }

    fn sort_tile_coords_near_center(
        &self,
        item: &MosaicItem,
        level_info: &crate::data::ome::LevelInfo,
        keys: &mut [TileCoord],
    ) {
        let y_dim = item.dataset.dims.y;
        let x_dim = item.dataset.dims.x;
        let center_world = self.camera.center_world_lvl0;
        let center_local = (center_world - item.offset) / item.scale;
        let downsample = level_info.downsample.max(1e-6);
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

    fn collect_draws_and_requests_for_item(
        &self,
        it: &MosaicItem,
        visible_world: egui::Rect,
        prev_visible_world: Option<egui::Rect>,
        viewport: egui::Rect,
        phase: Phase,
        request_generation: u64,
        prev_target_level: Option<usize>,
        prev_ceiling_level: Option<usize>,
        prev_floor_level: Option<usize>,
        prev_floor_until: Option<Instant>,
        prev_floor_world: Option<egui::Rect>,
        draws_out: &mut Vec<MosaicTileDraw>,
        sent: &mut usize,
        max_per_frame: usize,
        max_coarse_tiles_per_item_per_frame: usize,
    ) -> (
        Option<usize>,
        Option<usize>,
        Option<usize>,
        Option<Instant>,
        Option<egui::Rect>,
    ) {
        let item_rect = item_rect(it);
        if !item_rect.intersects(visible_world) {
            return (None, None, None, None, None);
        }

        // Translate the current mosaic viewport into this item's local level-0 pixel space.
        // All level choice and tile-key generation below happens in per-item local coordinates,
        // then the resulting draws are mapped back into shared mosaic world coordinates.
        //
        // The refinement path also tracks two short-lived pieces of history:
        // - a "ceiling" when zooming in, so intermediate levels remain eligible instead of
        //   jumping directly from very coarse to very fine
        // - a "floor" when zooming out, so the previous finer level can linger briefly and avoid
        //   a sudden blur jump while the new coarser level is still loading
        //
        // The return value carries that per-item refinement state back to the caller for reuse on
        // the next frame.
        // visible local (lvl0 px), intersection with ROI bounds
        let visible_in_item = visible_world.intersect(item_rect);
        let local_min = (visible_in_item.min.to_vec2() - it.offset) / it.scale;
        let local_max = (visible_in_item.max.to_vec2() - it.offset) / it.scale;
        let visible_local = egui::Rect::from_min_max(local_min.to_pos2(), local_max.to_pos2());

        let mut target_out: Option<usize> = None;
        let mut ceiling_out: Option<usize> = None;
        let mut zoom_out_floor_level_out: Option<usize> = prev_floor_level;
        let mut zoom_out_floor_until_out: Option<Instant> = prev_floor_until;
        let mut zoom_out_floor_world_out: Option<egui::Rect> = prev_floor_world;
        let levels = match phase {
            Phase::CoarseOnly => vec![it.dataset.levels.len().saturating_sub(1)],
            Phase::Refine => {
                let target_level = choose_level_auto(
                    &it.dataset.levels,
                    self.camera.zoom_screen_per_lvl0_px,
                    it.scale,
                );
                target_out = Some(target_level);
                let coarsest = it.dataset.levels.len().saturating_sub(1);
                let mut ceiling = prev_ceiling_level
                    .or(prev_target_level)
                    .unwrap_or(target_level);
                if let Some(prev_target) = prev_target_level {
                    if target_level < prev_target {
                        ceiling = ceiling.max(prev_target);
                    } else if target_level > prev_target {
                        ceiling = target_level;
                    }
                } else {
                    ceiling = target_level;
                }
                ceiling = ceiling.min(coarsest);
                ceiling_out = Some(ceiling);

                // We already have Phase::CoarseOnly ensuring coarsest coverage for all items; here
                // we focus on progressively refining between target and the sticky ceiling.
                let mut levels = Vec::new();
                for l in target_level..=ceiling {
                    levels.push(l);
                }
                levels.sort_unstable_by(|a, b| b.cmp(a)); // coarse -> fine
                levels.dedup();

                // Short-lived zoom-out floor: keep drawing the previous finer target level over the
                // previously-visible region for a moment to avoid sudden blur jumps.
                const ZOOM_OUT_FLOOR_MS: u64 = 400;
                let now = Instant::now();
                let prev_vis_world = prev_visible_world.unwrap_or(visible_world);
                if let Some(prev_target) = prev_target_level {
                    if target_level > prev_target {
                        zoom_out_floor_level_out = Some(prev_target);
                        zoom_out_floor_until_out =
                            Some(now + Duration::from_millis(ZOOM_OUT_FLOOR_MS));
                        zoom_out_floor_world_out = Some(prev_vis_world);
                    } else if target_level < prev_target {
                        zoom_out_floor_level_out = None;
                        zoom_out_floor_until_out = None;
                        zoom_out_floor_world_out = None;
                    }
                }
                levels
            }
        };

        let Some(src) = self.sources.get(it.id) else {
            return (
                target_out,
                ceiling_out,
                zoom_out_floor_level_out,
                zoom_out_floor_until_out,
                zoom_out_floor_world_out,
            );
        };

        // Keep the zoom-out floor until the new (coarser) target has enough tiles, so we don't get
        // a sudden blur jump if IO is slower than expected.
        const ZOOM_OUT_FLOOR_EXTEND_MS: u64 = 200;
        if let (Some(target_level), Some(floor_level), Some(floor_world)) = (
            target_out,
            zoom_out_floor_level_out,
            zoom_out_floor_world_out,
        ) {
            if floor_level >= it.dataset.levels.len() || floor_level >= target_level {
                zoom_out_floor_level_out = None;
                zoom_out_floor_until_out = None;
                zoom_out_floor_world_out = None;
            } else {
                let probe_gid = self.channel_layer_order.iter().copied().find(|&gid| {
                    let visible = self.channels.get(gid).is_some_and(|ch| ch.visible);
                    visible && src.channel_map.get(gid).copied().flatten().is_some()
                });
                let probe_channel = probe_gid.map(|gid| gid as u64);

                let visible_floor_in_item = floor_world.intersect(item_rect);
                let mut ready_enough = true;
                if let (Some(probe_channel), Some(level_info_tgt)) =
                    (probe_channel, it.dataset.levels.get(target_level))
                {
                    if visible_floor_in_item.width() > 0.0 && visible_floor_in_item.height() > 0.0 {
                        let local_min =
                            (visible_floor_in_item.min.to_vec2() - it.offset) / it.scale;
                        let local_max =
                            (visible_floor_in_item.max.to_vec2() - it.offset) / it.scale;
                        let visible_local_floor =
                            egui::Rect::from_min_max(local_min.to_pos2(), local_max.to_pos2());
                        let coords_tgt = tiles_needed_lvl0_rect(
                            visible_local_floor,
                            level_info_tgt,
                            &it.dataset.dims,
                            0,
                        );
                        let sample_max = 8usize;
                        let stride = (coords_tgt.len() / sample_max).max(1);
                        let mut total = 0usize;
                        let mut ready = 0usize;
                        for c in coords_tgt.iter().step_by(stride).take(sample_max) {
                            total += 1;
                            let k = MosaicRawTileKey {
                                dataset_id: it.id,
                                level: target_level,
                                tile_y: c.tile_y,
                                tile_x: c.tile_x,
                                channel: probe_channel,
                            };
                            if self.tiles_gl.contains(&k) {
                                ready += 1;
                            }
                        }
                        ready_enough = total == 0 || ready * 10 >= total * 8; // >=80%
                    }
                }

                let now = Instant::now();
                if ready_enough {
                    zoom_out_floor_level_out = None;
                    zoom_out_floor_until_out = None;
                    zoom_out_floor_world_out = None;
                } else if zoom_out_floor_until_out.map(|u| now > u).unwrap_or(true) {
                    zoom_out_floor_until_out =
                        Some(now + Duration::from_millis(ZOOM_OUT_FLOOR_EXTEND_MS));
                }
            }
        }

        for &level in &levels {
            let Some(level_info) = it.dataset.levels.get(level) else {
                continue;
            };
            let mut needed_tiles =
                tiles_needed_lvl0_rect(visible_local, level_info, &it.dataset.dims, 1);
            self.sort_tile_coords_near_center(it, level_info, &mut needed_tiles);

            let tile_limit = if matches!(phase, Phase::CoarseOnly) {
                max_coarse_tiles_per_item_per_frame.max(1)
            } else {
                usize::MAX
            };

            for (ti, key) in needed_tiles.iter().enumerate() {
                if ti >= tile_limit {
                    break;
                }
                if *sent >= max_per_frame {
                    break;
                }
                for gid in self.channel_layer_order.iter().copied() {
                    if *sent >= max_per_frame {
                        break;
                    }
                    let Some(gch) = self.channels.get(gid) else {
                        continue;
                    };
                    if !gch.visible {
                        continue;
                    }
                    if src.channel_map.get(gid).copied().flatten().is_none() {
                        continue;
                    }
                    let raw_key = MosaicRawTileKey {
                        dataset_id: it.id,
                        level,
                        tile_y: key.tile_y,
                        tile_x: key.tile_x,
                        channel: gid as u64,
                    };
                    if self.tiles_gl.mark_in_flight(raw_key) {
                        if self
                            .loader
                            .tx
                            .try_send(MosaicRawTileRequest {
                                key: raw_key,
                                generation: request_generation,
                            })
                            .is_ok()
                        {
                            *sent += 1;
                        } else {
                            self.tiles_gl.cancel_in_flight(&raw_key);
                            break;
                        }
                    }
                }
            }

            // Draw tiles (coarse -> fine).
            for key in needed_tiles {
                let screen_rect =
                    tile_screen_rect_mosaic(&self.camera, it, level_info, &key, viewport);
                if screen_rect.intersects(viewport) {
                    draws_out.push(MosaicTileDraw {
                        dataset_id: it.id,
                        level,
                        tile_y: key.tile_y,
                        tile_x: key.tile_x,
                        screen_rect,
                    });
                }
            }
        }

        // Draw-only zoom-out floor overlay last (finer than the current target).
        if matches!(phase, Phase::Refine) {
            if let (Some(target_level), Some(floor_level)) = (target_out, zoom_out_floor_level_out)
            {
                if floor_level < target_level {
                    let now = Instant::now();
                    if zoom_out_floor_until_out.map(|u| now <= u).unwrap_or(false) {
                        if let Some(floor_world) = zoom_out_floor_world_out.or(prev_visible_world) {
                            let visible_floor_in_item =
                                floor_world.intersect(item_rect).intersect(visible_world);
                            if visible_floor_in_item.width() > 0.0
                                && visible_floor_in_item.height() > 0.0
                            {
                                let local_min =
                                    (visible_floor_in_item.min.to_vec2() - it.offset) / it.scale;
                                let local_max =
                                    (visible_floor_in_item.max.to_vec2() - it.offset) / it.scale;
                                let visible_local_floor = egui::Rect::from_min_max(
                                    local_min.to_pos2(),
                                    local_max.to_pos2(),
                                );
                                if let Some(level_info) = it.dataset.levels.get(floor_level) {
                                    let needed_tiles = tiles_needed_lvl0_rect(
                                        visible_local_floor,
                                        level_info,
                                        &it.dataset.dims,
                                        1,
                                    );
                                    for key in needed_tiles.into_iter().take(512) {
                                        let screen_rect = tile_screen_rect_mosaic(
                                            &self.camera,
                                            it,
                                            level_info,
                                            &key,
                                            viewport,
                                        );
                                        if screen_rect.intersects(viewport) {
                                            draws_out.push(MosaicTileDraw {
                                                dataset_id: it.id,
                                                level: floor_level,
                                                tile_y: key.tile_y,
                                                tile_x: key.tile_x,
                                                screen_rect,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        (
            target_out,
            ceiling_out,
            zoom_out_floor_level_out,
            zoom_out_floor_until_out,
            zoom_out_floor_world_out,
        )
    }

    fn apply_sort_and_layout(&mut self) {
        // Sorting/grouping is allowed to reorder items freely, but we preserve user context by:
        // keeping the focused ROI selected if it still exists, and remapping the camera center as
        // a fraction of the old mosaic bounds into the new bounds after layout.
        let keep_focused = self.focused_core_id;
        let sort_by = self.sort_by.clone();
        let secondary = if self.sort_secondary_enabled {
            Some(self.sort_by_secondary.clone())
        } else {
            None
        };
        let group_by = self.group_by.clone();
        self.items.sort_by(|a, b| {
            if !group_by.is_empty() {
                let ga = group_label_for_item(a, &group_by);
                let gb = group_label_for_item(b, &group_by);
                let cg = cmp_sort_key(&ga, &gb);
                if cg != std::cmp::Ordering::Equal {
                    return cg;
                }
            }
            let c0 = cmp_sort_key(
                &sort_value_for_item(a, &sort_by),
                &sort_value_for_item(b, &sort_by),
            );
            if c0 != std::cmp::Ordering::Equal {
                return c0;
            }
            if let Some(sec) = secondary.as_deref() {
                let c1 = cmp_sort_key(&sort_value_for_item(a, sec), &sort_value_for_item(b, sec));
                if c1 != std::cmp::Ordering::Equal {
                    return c1;
                }
            }
            a.sample_id.cmp(&b.sample_id)
        });

        self.focused_core_id = keep_focused
            .filter(|id| self.items.iter().any(|it| it.id == *id))
            .or_else(|| self.items.first().map(|it| it.id));

        // Preserve camera center fraction within mosaic bounds.
        let old = self.mosaic_bounds;
        let fx = if old.width() > 0.0 {
            ((self.camera.center_world_lvl0.x - old.min.x) / old.width()).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let fy = if old.height() > 0.0 {
            ((self.camera.center_world_lvl0.y - old.min.y) / old.height()).clamp(0.0, 1.0)
        } else {
            0.5
        };

        let (bounds, blocks) = layout_items_grouped(
            &mut self.items,
            self.grid_cols,
            self.grid_cell_w,
            self.grid_cell_h,
            self.grid_pad,
            (!self.group_by.is_empty()).then_some(self.group_by.as_str()),
            self.group_gap.max(0.0),
            self.layout_mode,
        );
        self.mosaic_bounds = bounds;
        self.group_blocks = blocks;
        let newb = self.mosaic_bounds;
        self.camera.center_world_lvl0 = egui::pos2(
            newb.min.x + newb.width() * fx,
            newb.min.y + newb.height() * fy,
        );
    }

    fn draw_text_labels(&self, ui: &mut egui::Ui, viewport: egui::Rect) {
        let visible_world = visible_world_rect(&self.camera, viewport);
        let painter = ui.painter();
        let font = egui::FontId::proportional(13.0);
        let fg = egui::Color32::from_gray(240);
        let bg = egui::Color32::from_black_alpha(160);
        let line_gap = 1.0;

        for it in &self.items {
            let world_rect = item_rect(it);
            if !world_rect.intersects(visible_world) {
                continue;
            }
            let screen_min = self.camera.world_to_screen(world_rect.left_top(), viewport);
            let pos = screen_min + egui::vec2(6.0, 6.0);

            let lines = label_values_for_item(it, &self.label_columns);
            if lines.is_empty() {
                continue;
            }

            let galleys = lines
                .into_iter()
                .map(|line| painter.layout_no_wrap(line, font.clone(), fg))
                .collect::<Vec<_>>();
            let width = galleys
                .iter()
                .map(|galley| galley.size().x)
                .fold(0.0, f32::max);
            let height = galleys.iter().map(|galley| galley.size().y).sum::<f32>()
                + line_gap * galleys.len().saturating_sub(1) as f32;
            let rect = egui::Rect::from_min_size(pos, egui::vec2(width, height)).expand(2.0);
            painter.rect_filled(rect, 3.0, bg);

            let mut y = pos.y;
            for galley in galleys {
                painter.galley(egui::pos2(pos.x, y), galley.clone(), fg);
                y += galley.size().y + line_gap;
            }
        }
    }

    fn draw_group_labels(&self, ui: &mut egui::Ui, viewport: egui::Rect) {
        if self.group_blocks.is_empty() {
            return;
        }
        let visible_world = visible_world_rect(&self.camera, viewport);
        let painter = ui.painter();
        let font = egui::FontId::proportional(15.0);
        let fg = egui::Color32::from_gray(245);
        let bg = egui::Color32::from_black_alpha(200);

        for g in &self.group_blocks {
            if !g.world_rect.intersects(visible_world) {
                continue;
            }
            let screen_min = self
                .camera
                .world_to_screen(g.world_rect.left_top(), viewport);
            let pos = screen_min + egui::vec2(8.0, 8.0);
            let galley = painter.layout_no_wrap(g.name.clone(), font.clone(), fg);
            let rect = egui::Rect::from_min_size(pos, galley.size()).expand2(egui::vec2(4.0, 3.0));
            painter.rect_filled(rect, 4.0, bg);
            painter.galley(pos, galley, fg);
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Phase {
    CoarseOnly,
    Refine,
}

fn item_rect(it: &MosaicItem) -> egui::Rect {
    egui::Rect::from_min_size(it.offset.to_pos2(), it.placed_size)
}

fn sort_value_for_item(it: &MosaicItem, column: &str) -> String {
    if column == "id" {
        it.sample_id.clone()
    } else {
        it.meta.get(column).cloned().unwrap_or_default()
    }
}

fn label_value_for_item(it: &MosaicItem, column: &str) -> String {
    sort_value_for_item(it, column)
}

fn label_values_for_item(it: &MosaicItem, columns: &[String]) -> Vec<String> {
    columns
        .iter()
        .map(|column| label_value_for_item(it, column))
        .filter(|value| !value.trim().is_empty())
        .collect()
}

fn group_label_for_item(it: &MosaicItem, column: &str) -> String {
    let v = sort_value_for_item(it, column);
    let v = v.trim();
    if v.is_empty() {
        "(missing)".to_string()
    } else {
        v.to_string()
    }
}

fn cmp_sort_key(a: &str, b: &str) -> std::cmp::Ordering {
    let empty_a = a.trim().is_empty();
    let empty_b = b.trim().is_empty();
    if empty_a != empty_b {
        return empty_a.cmp(&empty_b); // non-empty first
    }
    a.to_ascii_lowercase().cmp(&b.to_ascii_lowercase())
}

fn visible_world_rect(camera: &Camera, viewport: egui::Rect) -> egui::Rect {
    let world_min = camera.screen_to_world(viewport.left_top(), viewport);
    let world_max = camera.screen_to_world(viewport.right_bottom(), viewport);
    egui::Rect::from_min_max(world_min, world_max)
}

fn layout_items(
    items: &mut [MosaicItem],
    cols: usize,
    cell_w: f32,
    cell_h: f32,
    pad: f32,
) -> egui::Rect {
    // Fit-cell layout preserves each ROI aspect ratio inside a regular grid cell. The item's
    // local level-0 pixel space is scaled uniformly into that cell; later drawing code relies on
    // `offset` and `scale` being the full item->mosaic transform.
    let n = items.len();
    let cols = cols.max(1);
    let rows = (n + cols - 1) / cols;

    for (pos, it) in items.iter_mut().enumerate() {
        let (w0, h0) = level0_size(&it.dataset);
        let s = (cell_w / w0.max(1.0)).min(cell_h / h0.max(1.0)).max(1e-6);
        let placed_w = w0 * s;
        let placed_h = h0 * s;
        let col = (pos % cols) as f32;
        let row = (pos / cols) as f32;
        let cell_origin = egui::vec2(col * (cell_w + pad), row * (cell_h + pad));
        let inset = egui::vec2((cell_w - placed_w) * 0.5, (cell_h - placed_h) * 0.5);
        it.offset = cell_origin + inset;
        it.scale = s;
        it.placed_size = egui::vec2(placed_w, placed_h);
    }

    let total_w = cols as f32 * (cell_w + pad) - pad;
    let total_h = rows as f32 * (cell_h + pad) - pad;
    egui::Rect::from_min_size(
        egui::pos2(0.0, 0.0),
        egui::vec2(total_w.max(1.0), total_h.max(1.0)),
    )
}

fn layout_items_native(items: &mut [MosaicItem], cols: usize, pad: f32) -> egui::Rect {
    if items.is_empty() {
        return egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1.0, 1.0));
    }

    let cols = cols.max(1);
    let mut max_w = 1.0f32;
    let mut y_cursor = 0.0f32;

    for row in items.chunks_mut(cols) {
        let row_h = row
            .iter()
            .map(|it| {
                let (_, h) = level0_size(&it.dataset);
                h
            })
            .fold(1.0f32, f32::max);

        let mut x_cursor = 0.0f32;
        for it in row {
            let (w0, h0) = level0_size(&it.dataset);
            it.scale = 1.0;
            it.placed_size = egui::vec2(w0, h0);
            it.offset = egui::vec2(x_cursor, y_cursor + (row_h - h0) * 0.5);
            x_cursor += w0 + pad;
        }

        let row_w = (x_cursor - pad).max(1.0);
        max_w = max_w.max(row_w);
        y_cursor += row_h + pad;
    }

    let total_h = (y_cursor - pad).max(1.0);
    egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(max_w, total_h))
}

fn layout_items_grouped(
    items: &mut [MosaicItem],
    cols: usize,
    cell_w: f32,
    cell_h: f32,
    pad: f32,
    group_by: Option<&str>,
    group_gap: f32,
    layout_mode: MosaicLayoutMode,
) -> (egui::Rect, Vec<GroupBlock>) {
    let Some(group_col) = group_by else {
        let bounds = match layout_mode {
            MosaicLayoutMode::FitCells => layout_items(items, cols, cell_w, cell_h, pad),
            MosaicLayoutMode::NativePixels => layout_items_native(items, cols, pad),
        };
        return (bounds, Vec::new());
    };
    if items.is_empty() {
        return (
            egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1.0, 1.0)),
            Vec::new(),
        );
    }

    let cols = cols.max(1);
    let header_h = 48.0f32;
    let gap = group_gap.max(0.0);

    let mut block_specs: Vec<(String, f32, f32)> = Vec::new();
    let mut total_w = 1.0f32;
    let mut y_cursor = 0.0f32;

    let mut i = 0usize;
    while i < items.len() {
        let gname = group_label_for_item(&items[i], group_col);
        let gkey = gname.to_ascii_lowercase();
        let mut j = i + 1;
        while j < items.len()
            && group_label_for_item(&items[j], group_col).to_ascii_lowercase() == gkey
        {
            j += 1;
        }

        let group_len = j - i;
        let group_h;
        let group_w;
        match layout_mode {
            MosaicLayoutMode::FitCells => {
                let current_total_w = cols as f32 * (cell_w + pad) - pad;
                let group_rows = (group_len + cols - 1) / cols;
                group_h = header_h + group_rows as f32 * (cell_h + pad) - pad;
                group_w = current_total_w.max(1.0);

                for (pos_in_group, it) in items[i..j].iter_mut().enumerate() {
                    let (w0, h0) = level0_size(&it.dataset);
                    let s = (cell_w / w0.max(1.0)).min(cell_h / h0.max(1.0)).max(1e-6);
                    let placed_w = w0 * s;
                    let placed_h = h0 * s;
                    let col = (pos_in_group % cols) as f32;
                    let row = (pos_in_group / cols) as f32;
                    let cell_origin = egui::vec2(
                        col * (cell_w + pad),
                        y_cursor + header_h + row * (cell_h + pad),
                    );
                    let inset = egui::vec2((cell_w - placed_w) * 0.5, (cell_h - placed_h) * 0.5);
                    it.offset = cell_origin + inset;
                    it.scale = s;
                    it.placed_size = egui::vec2(placed_w, placed_h);
                }
            }
            MosaicLayoutMode::NativePixels => {
                let mut content_y = y_cursor + header_h;
                let mut max_group_w = 1.0f32;
                for row in items[i..j].chunks_mut(cols) {
                    let row_h = row
                        .iter()
                        .map(|it| {
                            let (_, h) = level0_size(&it.dataset);
                            h
                        })
                        .fold(1.0f32, f32::max);
                    let mut x_cursor = 0.0f32;
                    for it in row {
                        let (w0, h0) = level0_size(&it.dataset);
                        it.scale = 1.0;
                        it.placed_size = egui::vec2(w0, h0);
                        it.offset = egui::vec2(x_cursor, content_y + (row_h - h0) * 0.5);
                        x_cursor += w0 + pad;
                    }
                    max_group_w = max_group_w.max((x_cursor - pad).max(1.0));
                    content_y += row_h + pad;
                }
                group_h = header_h + (content_y - (y_cursor + header_h) - pad).max(1.0);
                group_w = max_group_w;
            }
        }

        total_w = total_w.max(group_w);
        block_specs.push((gname.clone(), y_cursor, group_h));

        y_cursor += group_h + gap;
        i = j;
    }

    let total_h = (y_cursor - gap).max(1.0);
    let blocks = block_specs
        .into_iter()
        .map(|(name, y, h)| GroupBlock {
            name,
            world_rect: egui::Rect::from_min_size(egui::pos2(0.0, y), egui::vec2(total_w, h)),
        })
        .collect::<Vec<_>>();
    let bounds =
        egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(total_w.max(1.0), total_h));
    (bounds, blocks)
}

fn tile_screen_rect_mosaic(
    camera: &Camera,
    it: &MosaicItem,
    level: &crate::data::ome::LevelInfo,
    key: &TileCoord,
    viewport: egui::Rect,
) -> egui::Rect {
    let y_dim = it.dataset.dims.y;
    let x_dim = it.dataset.dims.x;
    let chunk_y = level.chunks[y_dim] as f32;
    let chunk_x = level.chunks[x_dim] as f32;

    let y0 = (key.tile_y as f32) * chunk_y;
    let x0 = (key.tile_x as f32) * chunk_x;
    let y1 = (y0 + chunk_y).min(level.shape[y_dim] as f32);
    let x1 = (x0 + chunk_x).min(level.shape[x_dim] as f32);

    // local world in lvl0 px
    let downsample = level.downsample;
    let local_min = egui::pos2(x0 * downsample, y0 * downsample);
    let local_max = egui::pos2(x1 * downsample, y1 * downsample);

    // mosaic world
    let world_min = (it.offset + local_min.to_vec2() * it.scale).to_pos2();
    let world_max = (it.offset + local_max.to_vec2() * it.scale).to_pos2();

    let screen_min = camera.world_to_screen(world_min, viewport);
    let screen_max = camera.world_to_screen(world_max, viewport);
    egui::Rect::from_min_max(screen_min, screen_max)
}

fn level0_size(ds: &OmeZarrDataset) -> (f32, f32) {
    let shape0 = ds.levels.get(0).map(|l| &l.shape);
    let Some(shape0) = shape0 else {
        return (1.0, 1.0);
    };
    let w = shape0[ds.dims.x] as f32;
    let h = shape0[ds.dims.y] as f32;
    (w.max(1.0), h.max(1.0))
}

fn max_level0_size_items(items: &[MosaicItem]) -> (f32, f32) {
    let mut max_w = 1.0f32;
    let mut max_h = 1.0f32;
    for it in items {
        let (w, h) = level0_size(&it.dataset);
        max_w = max_w.max(w);
        max_h = max_h.max(h);
    }
    (max_w, max_h)
}

fn build_channel_map(global: &[GlobalChannel], ds: &OmeZarrDataset) -> Vec<Option<u64>> {
    let mut out = vec![None; global.len()];
    for (gid, gch) in global.iter().enumerate() {
        if let Some(ds_ch) = ds.channels.iter().find(|c| c.name == gch.name) {
            out[gid] = Some(ds_ch.index as u64);
        }
    }
    out
}

fn build_global_channels<'a>(
    datasets: impl IntoIterator<Item = &'a OmeZarrDataset>,
) -> Vec<GlobalChannel> {
    let mut out: Vec<GlobalChannel> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for ds in datasets {
        for c in &ds.channels {
            if !seen.insert(c.name.clone()) {
                continue;
            }
            out.push(GlobalChannel {
                name: c.name.clone(),
                color_rgb: c.color_rgb,
                window: c.window,
                visible: c.visible,
            });
        }
    }

    out
}
