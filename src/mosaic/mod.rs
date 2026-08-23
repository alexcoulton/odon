mod canvas;
mod construction;
mod control;
mod io;
mod layers_ui;
mod memory_navigation;
mod panels;
mod segmentation_geojson;
mod tiles_gl;
mod update;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use eframe::egui;
use glow::HasContext;
use rfd::FileDialog;

use self::io::{
    MosaicPinnedLevelStatus, MosaicPinnedLevels, MosaicRawTileKey, MosaicRawTileLoaderHandle,
    MosaicRawTileRequest, MosaicRawTileWorkerResponse, MosaicSource,
    estimate_level_ram_bytes_for_channels, spawn_mosaic_raw_tile_loader,
};
use self::segmentation_geojson::MosaicGeoJsonSegmentationOverlay;
use self::tiles_gl::{ChannelDraw, MosaicTileDraw, MosaicTilesGl};
use crate::annotations::AnnotationPointsLayer;
use crate::app::{NativeControlIngress, NativeControlIntent};
use crate::app_support::memory::{
    MemoryChannelRow, MemoryRisk, MemoryRiskLevel, PendingMemoryAction, SystemMemorySnapshot,
    format_bytes, memory_risk, refresh_system_memory_if_needed, ui_memory_channel_selector,
    ui_memory_overview, ui_pending_memory_action_dialog,
};
use crate::app_support::repaint as repaint_control;
use crate::app_support::screenshot::{
    RendererScreenshotRequest, ScreenshotDialogState, ScreenshotSettings,
};
use crate::camera::Camera;
use crate::data::dataset_source::DatasetSource;
use crate::data::ome::OmeZarrDataset;
use crate::data::project_config::ProjectLayerGroups;
use crate::data::remote_store::build_http_store;
use crate::data::samplesheet::load_samplesheet_csv;
use crate::imaging::tiling::{TileCoord, choose_level_auto, tiles_needed_lvl0_rect};
use crate::objects::PreloadedObjectLayer;
use crate::project::groups as layer_groups;
use crate::project::{
    ProjectCameraState, ProjectChannelViewState, ProjectMosaicViewState, ProjectObjectCacheUiState,
    ProjectSpace, ProjectUiState,
};
use crate::ui::canvas_overlays;
use crate::ui::channel_notes;
use crate::ui::channels_panel::{self, ChannelListHost, ChannelSortMode};
use crate::ui::contrast;
use crate::ui::group_layers::{GroupLayersDialog, GroupLayersTarget, default_group_name};
use crate::ui::icons::Icon;
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
    note: String,
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
    Views,
    Layout,
    Memory,
}

impl RightTab {
    fn storage_key(self) -> &'static str {
        match self {
            Self::Properties => "properties",
            Self::Views => "views",
            Self::Layout => "layout",
            Self::Memory => "memory",
        }
    }

    fn from_storage_key(value: &str) -> Option<Self> {
        match value {
            "properties" => Some(Self::Properties),
            "views" => Some(Self::Views),
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

pub struct MosaicViewerApp {
    items: Vec<MosaicItem>,
    sources: Arc<Vec<MosaicSource>>,
    pinned_levels: MosaicPinnedLevels,
    loader: MosaicRawTileLoaderHandle,
    tiles_gl: MosaicTilesGl,
    _remote_runtimes: Vec<Arc<tokio::runtime::Runtime>>,

    camera: Camera,
    last_canvas_rect: Option<egui::Rect>,
    mosaic_bounds: egui::Rect,

    focused_core_id: Option<usize>,
    selected_core_ids: HashSet<usize>,

    abs_max: f32,
    channels: Vec<GlobalChannel>,
    selected_channel: usize,
    channel_list_search: String,
    active_layer: MosaicLayerId,
    selected_channel_layers: HashSet<usize>,
    memory_selected_channels: HashSet<usize>,
    channel_select_anchor_idx: Option<usize>,
    selected_channel_group_id: Option<u64>,
    quick_contrast_target: top_bar::QuickContrastTarget,
    selected_overlay_layers: HashSet<MosaicLayerId>,
    overlay_select_anchor_pos: Option<usize>,
    overlay_layer_order: Vec<MosaicLayerId>,
    channel_layer_order: Vec<usize>,
    channel_sort_mode: ChannelSortMode,
    annotation_layers: Vec<AnnotationPointsLayer>,
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
    show_left_panel: bool,
    show_right_panel: bool,
    close_dialog_open: bool,
    system_memory: Option<SystemMemorySnapshot>,
    system_memory_last_refresh: Option<Instant>,
    pending_memory_load: Option<PendingMemoryAction<serde_json::Value>>,
    control_actor_memory_state: serde_json::Value,
    tile_request_generation: u64,
    last_tile_request_signature: Option<TileRequestSignature>,

    renderer_status: String,
    show_return_navigation: bool,
    return_dataset_root: Option<PathBuf>,
    pending_platform_effect: Option<MosaicPlatformEffect>,
    group_layers_dialog: Option<GroupLayersDialog>,
    smooth_pixels: bool,
    show_tile_debug: bool,
    screenshot_dialog: ScreenshotDialogState,
    screenshot_capture: MosaicScreenshotCaptureAdapter,
    seg_geojson: MosaicGeoJsonSegmentationOverlay,
    seg_geojson_pending_visible: bool,
    project_space: ProjectSpace,
    active_help_topic: Option<crate::ui::help::HelpTopic>,
    consumed_mosaic_resource_generation: u64,
    consumed_mosaic_object_generation: u64,
    native_command_ingress: NativeControlIngress,
}

#[derive(Debug, Default)]
struct MosaicScreenshotCaptureAdapter {
    pending: Option<RendererScreenshotRequest>,
}

impl ChannelListHost for MosaicViewerApp {
    type LayerId = MosaicLayerId;

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
        self.submit_native_control_intent(
            "viewer.channels.presentation.set",
            serde_json::json!({"sort":mode.storage_key()}),
        );
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
        self.submit_native_control_intent(
            "viewer.channels.set_visible",
            serde_json::json!({
                "channels":[idx],
                "mode":if visible { "show" } else { "hide" },
            }),
        );
    }

    fn set_channels_visible(&mut self, indices: &[usize], visible: bool) {
        self.submit_native_control_intent(
            "viewer.channels.set_visible",
            serde_json::json!({
                "channels":indices,
                "mode":if visible { "show" } else { "hide" },
            }),
        );
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

    fn can_reset_selected_layer_positions(&mut self) -> bool {
        false
    }

    fn reset_selected_layer_positions(&mut self) -> bool {
        false
    }

    fn can_apply_rgb_preset(&self) -> bool {
        self.channels.len() == 3
    }

    fn apply_rgb_preset(&mut self) -> bool {
        if self.channels.len() != 3 {
            return false;
        }
        let rgb = [[255, 0, 0], [0, 255, 0], [0, 0, 255]];
        let hi = self.abs_max.clamp(1.0, 255.0);
        self.submit_native_control_intent(
            "viewer.channels.set_visible",
            serde_json::json!({"channels":[0,1,2],"mode":"only"}),
        );
        for (index, color_rgb) in rgb.into_iter().enumerate() {
            self.submit_native_control_intent(
                "viewer.channels.set_color",
                serde_json::json!({"index":index,"color_rgb":color_rgb}),
            );
            self.submit_native_control_intent(
                "viewer.channels.set_contrast",
                serde_json::json!({"index":index,"min":0.0,"max":hi}),
            );
        }
        self.submit_native_control_intent(
            "viewer.channels.set_active",
            serde_json::json!({"index":0}),
        );
        self.renderer_status = "Applying RGB preset to channels 0-2...".to_string();
        true
    }

    fn layer_groups(&self) -> ProjectLayerGroups {
        self.layer_groups.clone()
    }

    fn set_layer_groups(&mut self, groups: ProjectLayerGroups) {
        self.submit_native_control_intent(
            "viewer.channels.set_group",
            serde_json::json!({"state":groups}),
        );
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
pub enum MosaicPlatformEffect {
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

#[derive(Debug, Clone, Copy)]
enum Phase {
    CoarseOnly,
    Refine,
}

fn item_rect(it: &MosaicItem) -> egui::Rect {
    egui::Rect::from_min_size(it.offset.to_pos2(), it.placed_size)
}

fn json_vec2(value: Option<&serde_json::Value>) -> Option<[f32; 2]> {
    let values = value?.as_array()?;
    if values.len() != 2 {
        return None;
    }
    let x = values[0].as_f64()? as f32;
    let y = values[1].as_f64()? as f32;
    (x.is_finite() && y.is_finite()).then_some([x, y])
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
                note: String::new(),
            });
        }
    }

    out
}

#[cfg(test)]
#[path = "tests.rs"]
mod layout_tests;
