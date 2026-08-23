use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use eframe::egui;
use rfd::FileDialog;
use serde::{Deserialize, Serialize};

use crate::app_support::memory::format_bytes;
use crate::app_support::settings::RecentProject;
use crate::data::dataset_kind::{
    LocalDatasetKind, can_open_in_mosaic, classify_local_dataset_path, normalize_local_dataset_path,
};
use crate::data::dataset_source::DatasetSource;
#[cfg(test)]
use crate::data::project_config::ProjectMaskLayer;
use crate::data::project_config::{ProjectConfig, ProjectLayerGroups, ProjectRoi};
use crate::data::samplesheet::{
    SampleRow, SampleSheet, load_samplesheet_csv, write_samplesheet_csv,
};
use crate::objects::{
    ObjectPreloadMode, ObjectPreloadSettings, ObjectProjectAnalysisState, ObjectProjectDisplayState,
};
use crate::ui::help::HelpTopic;
use crate::ui::roi_browser::RoiBrowseState;
use odon::deep_link::DeepLinkRequest;

mod browser_ui;
mod control_persistence;
mod imports;
mod rois_views;
mod views_ui;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProjectFileV1 {
    version: u32,
    items: Vec<ProjectItem>,
    selected: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProjectFileV2 {
    version: u32,
    items: Vec<ProjectItem>,
    focused: Option<PathBuf>,
    selected: Vec<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProjectFileV3Legacy {
    version: u32,
    items: Vec<ProjectItem>,
    focused: Option<PathBuf>,
    selected: Vec<PathBuf>,
    #[serde(default)]
    napari_gui_config_yaml: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProjectFileV4 {
    version: u32,
    #[serde(default)]
    config: ProjectConfig,
    focused: Option<PathBuf>,
    selected: Vec<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProjectFileV5 {
    version: u32,
    #[serde(default)]
    config: ProjectConfig,
    focused: Option<String>,
    #[serde(default)]
    selected: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProjectFileV6 {
    version: u32,
    #[serde(default)]
    config: ProjectConfig,
    #[serde(default)]
    state: ProjectState,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
struct ProjectState {
    #[serde(default)]
    browser: ProjectBrowserState,
    #[serde(default)]
    roi_views: BTreeMap<String, ProjectRoiViewState>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    view_presets: Vec<ProjectViewPreset>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    mosaic: Option<ProjectMosaicViewState>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
struct ProjectBrowserState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    focused: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    selected: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ProjectRoiViewState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_groups: Option<ProjectLayerGroups>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub channel_order: Vec<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub channels: Vec<ProjectChannelViewState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_channel: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_layer: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub overlay_order: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub overlay_visibility: BTreeMap<String, bool>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub overlay_offsets_world: BTreeMap<String, [f32; 2]>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub overlay_original_offsets_world: BTreeMap<String, [f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segmentation: Option<ProjectSegmentationViewState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub analysis: Option<ObjectProjectAnalysisState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<ProjectCameraState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ui: Option<ProjectUiState>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub annotation_layers: Vec<ProjectAnnotationLayerState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace: Option<ProjectWorkspaceViewState>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProjectWorkspaceViewState {
    pub version: u32,
    pub layout: String,
    #[serde(default = "default_split_ratio")]
    pub split_ratio: f32,
    pub active_viewport_id: String,
    #[serde(default = "default_true")]
    pub link_camera: bool,
    #[serde(default = "default_true")]
    pub link_plane: bool,
    #[serde(default = "default_true")]
    pub link_selection: bool,
    pub viewports: Vec<ProjectViewportViewState>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ProjectViewportViewState {
    pub id: String,
    pub title: String,
    #[serde(default = "default_viewport_revision")]
    pub navigation_revision: u64,
    #[serde(default = "default_viewport_revision")]
    pub presentation_revision: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_groups: Option<ProjectLayerGroups>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<ProjectCameraState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plane_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x_level0: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub y_level0: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub z_level0: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub channel_order: Vec<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub channels: Vec<ProjectChannelViewState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_channel: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_layer: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub overlay_order: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub overlay_visibility: BTreeMap<String, bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segmentation: Option<ProjectSegmentationViewState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object_filter: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object_visible: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object_opacity: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object_width_screen_px: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object_color_rgb: Option<[u8; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object_show_selection_overlay: Option<bool>,
    /// Complete per-viewport overlay presentation. This is additive to the
    /// typed core fields above so older workspace files remain readable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presentation: Option<serde_json::Value>,
}

fn default_true() -> bool {
    true
}

fn default_split_ratio() -> f32 {
    0.5
}

fn default_viewport_revision() -> u64 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ProjectChannelViewState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visible: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_rgb: Option<[u8; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset_world: Option<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_offset_world: Option<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotation_rad: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ProjectCameraState {
    pub center_world_lvl0: [f32; 2],
    pub zoom_screen_per_lvl0_px: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ProjectSegmentationViewState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outlines_color_rgb: Option<[u8; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outlines_opacity: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outlines_width_px: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object_display: Option<ObjectProjectDisplayState>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ProjectUiState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_left_panel: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_right_panel: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub left_tab: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub right_tab: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel_sort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub smooth_pixels: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_tile_debug: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_scale_bar: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_hud: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_level: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manual_level: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ProjectAnnotationCategoryStyleState {
    pub name: String,
    pub visible: bool,
    pub color_rgb: [u8; 3],
    pub shape: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ProjectAnnotationLayerState {
    pub id: u64,
    pub name: String,
    pub visible: bool,
    pub radius_screen_px: f32,
    pub opacity: f32,
    pub stroke_width: f32,
    pub stroke_color_rgb: [u8; 3],
    pub stroke_color_alpha: u8,
    pub offset_world: [f32; 2],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parquet_path: Option<String>,
    pub roi_id_column: String,
    pub x_column: String,
    pub y_column: String,
    pub value_column: String,
    pub selected_value_column: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub category_styles: Vec<ProjectAnnotationCategoryStyleState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuous_shape: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuous_range: Option<[f32; 2]>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ProjectMosaicViewState {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub channel_order: Vec<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub channels: Vec<ProjectChannelViewState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_channel: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_layer: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub overlay_order: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub overlay_visibility: BTreeMap<String, bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sort_by: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sort_secondary_enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sort_by_secondary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_by: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_group_labels: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_gap: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_text_labels: Option<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub label_columns: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<ProjectCameraState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ui: Option<ProjectUiState>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub annotation_layers: Vec<ProjectAnnotationLayerState>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ProjectViewChannelRef {
    pub label: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub alias: String,
}

impl ProjectViewChannelRef {
    fn search_terms(&self) -> Vec<String> {
        let mut terms = Vec::new();
        push_unique_non_empty(&mut terms, self.alias.trim());
        push_unique_non_empty(&mut terms, self.label.trim());
        terms
    }

    fn display_name(&self) -> String {
        if self.alias.trim().is_empty() {
            self.label.clone()
        } else {
            format!("{} ({})", self.alias.trim(), self.label)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ProjectViewSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel_ref: Option<ProjectViewChannelRef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub visible_channels: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub visible_channel_refs: Vec<ProjectViewChannelRef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hidden_channels: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segmentation_source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_labels: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cell_color_by: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub visible_cell_types: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hidden_cell_types: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fill_cells: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_selection_overlay: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<ProjectCameraState>,
}

impl ProjectViewSpec {
    pub fn to_deep_link_request(&self, roi: Option<String>) -> DeepLinkRequest {
        let channel_alternatives = self
            .channel_ref
            .as_ref()
            .map(ProjectViewChannelRef::search_terms)
            .unwrap_or_default();
        let channel = if channel_alternatives.is_empty() {
            self.channel.clone()
        } else {
            channel_alternatives.first().cloned()
        };
        let visible_channel_alternatives = self
            .visible_channel_refs
            .iter()
            .map(ProjectViewChannelRef::search_terms)
            .filter(|terms| !terms.is_empty())
            .collect::<Vec<_>>();
        DeepLinkRequest {
            example: None,
            project_path: None,
            roi,
            sample: None,
            channel,
            channel_alternatives,
            visible_channels: self.visible_channels.clone(),
            visible_channel_alternatives,
            group_visible_channels: false,
            visible_channel_group: None,
            visible_channel_group_color: None,
            channel_order: None,
            hidden_channels: self.hidden_channels.clone(),
            hidden_channel_alternatives: Vec::new(),
            contrast_min: None,
            contrast_max: None,
            channel_contrasts: Vec::new(),
            channel_colors: Vec::new(),
            segmentation: None,
            segmentation_source: self.segmentation_source.clone(),
            load_segmentation_labels: self.load_labels,
            cell_color_by: self.cell_color_by.clone(),
            fill_cells: self.fill_cells,
            show_selection_overlay: self.show_selection_overlay,
            fast_object_rendering: None,
            visible_cell_types: self.visible_cell_types.clone(),
            hidden_cell_types: self.hidden_cell_types.clone(),
            object_level_colors: Vec::new(),
            object_filters: Vec::new(),
            object_filter_logic: None,
            object_query: None,
            center_world: self.camera.as_ref().map(|camera| camera.center_world_lvl0),
            zoom: self
                .camera
                .as_ref()
                .map(|camera| camera.zoom_screen_per_lvl0_px),
        }
    }
}

fn push_unique_non_empty(dst: &mut Vec<String>, value: &str) {
    let value = value.trim();
    if !value.is_empty() && !dst.iter().any(|existing| existing == value) {
        dst.push(value.to_string());
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ProjectViewPreset {
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default)]
    pub spec: ProjectViewSpec,
}

fn view_preset_summary(preset: &ProjectViewPreset) -> String {
    let channel_names = visible_channel_display_names(&preset.spec);
    let channels = if channel_names.is_empty() {
        "(current channels)".to_string()
    } else {
        channel_names.join(", ")
    };
    let cell_types = if preset.spec.visible_cell_types.is_empty() {
        "(all cell types)".to_string()
    } else {
        preset.spec.visible_cell_types.join(", ")
    };
    let color_by = preset
        .spec
        .cell_color_by
        .as_deref()
        .unwrap_or("(single color)");
    format!("Markers: {channels}\nColor by: {color_by}\nCell types: {cell_types}")
}

fn visible_channel_display_names(spec: &ProjectViewSpec) -> Vec<String> {
    if !spec.visible_channel_refs.is_empty() {
        spec.visible_channel_refs
            .iter()
            .map(ProjectViewChannelRef::display_name)
            .collect()
    } else {
        spec.visible_channels.clone()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProjectItem {
    pub path: PathBuf,
    #[serde(default)]
    pub display_name: Option<String>,
}

#[derive(Debug, Clone)]
pub enum ProjectSpaceAction {
    Open(ProjectRoi),
    OpenView(ProjectRoi, ProjectViewSpec),
    OpenLocalPath(PathBuf),
    OpenProject(PathBuf),
    SaveProject(PathBuf),
    ForgetRecentProject(PathBuf),
    ClearRecentProjects,
    CaptureCurrentView,
    OpenMosaic,
    OpenRemoteDialog,
    PreloadObjectSegmentations(ObjectPreloadSettings),
    ClearObjectCache,
    ShowHelp(HelpTopic),
}

#[derive(Debug, Clone)]
pub struct ProjectControlIntent {
    pub method: &'static str,
    pub params: serde_json::Value,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ProjectObjectCacheUiState {
    pub available_count: usize,
    pub on_disk_bytes: u64,
    pub cached: usize,
    pub total: usize,
    pub done: usize,
    pub failed: usize,
    pub loading: bool,
    pub cached_settings: ObjectPreloadSettings,
}

#[derive(Debug, Default, Clone)]
pub struct ProjectSpace {
    config: ProjectConfig,
    state: ProjectState,
    focused: Option<String>,
    selected: HashSet<String>,
    config_generation: u64,
    project_file_path: Option<PathBuf>,
    save_path: String,
    load_path: String,
    status: String,
    config_json: String,
    config_json_dirty: bool,
    config_json_status: String,
    new_meta_key: String,
    new_meta_value: String,
    roi_browse: RoiBrowseState,
    object_cache_ui: ProjectObjectCacheUiState,
    object_cache_settings: ObjectPreloadSettings,
    recent_projects: Vec<RecentProject>,
    selected_view_preset: usize,
    view_preset_name_input: String,
    views_dialog_open: bool,
    view_preset_draft: Option<ProjectViewSpec>,
    save_toast: Option<ProjectSaveToast>,
    roi_list_hover_rect: Option<egui::Rect>,
    /// Last complete actor project replacement materialized into this renderer-side view model.
    /// Ordinary actor projections deliberately do not replace the complete persisted state.
    control_actor_load_generation: u64,
    control_actor_owned: bool,
    pending_control_intents: Vec<ProjectControlIntent>,
}

#[derive(Debug, Clone)]
struct ProjectSaveToast {
    message: String,
    created_at: Instant,
}

#[derive(Debug, Clone)]
struct SegmentationCandidate {
    path: PathBuf,
    normalized_stem: String,
    tokens: HashSet<String>,
    format_rank: i32,
}

fn collect_segmentation_candidates(
    roots: &[PathBuf],
    max_depth: usize,
) -> Vec<SegmentationCandidate> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut stack = roots
        .iter()
        .cloned()
        .map(|root| (root, 0usize))
        .collect::<Vec<_>>();

    while let Some((dir, depth)) = stack.pop() {
        let Ok(read_dir) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in read_dir.flatten() {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                if depth < max_depth {
                    stack.push((path, depth + 1));
                }
                continue;
            }
            if !file_type.is_file() || !is_segmentation_candidate_path(&path) {
                continue;
            }
            let canonical = path.canonicalize().unwrap_or(path);
            if !seen.insert(canonical.clone()) {
                continue;
            }
            let normalized_stem = normalize_match_string(&path_stem_without_multi_ext(&canonical));
            let tokens = segmentation_match_tokens_for_path(&canonical);
            let format_rank = segmentation_candidate_format_rank(&canonical);
            out.push(SegmentationCandidate {
                path: canonical,
                normalized_stem,
                tokens,
                format_rank,
            });
        }
    }

    out
}

fn best_segmentation_match_for_roi<'a>(
    roi: &ProjectRoi,
    candidates: &'a [SegmentationCandidate],
) -> Option<&'a SegmentationCandidate> {
    let local_path = roi.local_path()?;
    let target_stem = path_stem_without_multi_ext(local_path);
    let target_norm = normalize_match_string(&target_stem);
    let target_tokens = roi_match_tokens(roi);
    let parent = local_path
        .parent()
        .map(|p| p.canonicalize().unwrap_or_else(|_| p.to_path_buf()));

    let mut ranked = candidates
        .iter()
        .filter_map(|candidate| {
            let mut score = 0i32;
            if !target_norm.is_empty() && candidate.normalized_stem == target_norm {
                score += 1000;
            }
            if !target_tokens.is_empty() {
                let overlap = target_tokens
                    .iter()
                    .filter(|token| candidate.tokens.contains(*token))
                    .count() as i32;
                score += overlap * 120;
            }
            if let (Some(parent), Some(candidate_parent)) =
                (parent.as_ref(), candidate.path.parent())
            {
                let candidate_parent = candidate_parent
                    .canonicalize()
                    .unwrap_or_else(|_| candidate_parent.to_path_buf());
                if &candidate_parent == parent {
                    score += 250;
                }
            }
            let stem_text = candidate
                .path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            if stem_text.contains("seg") || stem_text.contains("mask") {
                score += 40;
            }
            (score >= 180).then_some((score, candidate))
        })
        .collect::<Vec<_>>();

    ranked.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| b.1.format_rank.cmp(&a.1.format_rank))
            .then_with(|| a.1.path.to_string_lossy().cmp(&b.1.path.to_string_lossy()))
    });
    ranked.first().map(|(_, candidate)| *candidate)
}

fn roi_match_tokens(roi: &ProjectRoi) -> HashSet<String> {
    let mut tokens = HashSet::new();
    insert_match_tokens(&mut tokens, &roi.id);
    if let Some(name) = roi.display_name.as_ref() {
        insert_match_tokens(&mut tokens, name);
    }
    if let Some(local_path) = roi.local_path() {
        insert_match_tokens(&mut tokens, &path_stem_without_multi_ext(local_path));
    }
    for value in roi.meta.values() {
        insert_match_tokens(&mut tokens, value);
    }
    tokens
}

fn resolve_project_relative_path(project_dir: Option<&Path>, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        return path;
    }
    project_dir.map_or(path.clone(), |dir| dir.join(path))
}

#[cfg(test)]
#[path = "space/tests.rs"]
mod tests;

fn segmentation_match_tokens_for_path(path: &Path) -> HashSet<String> {
    let mut tokens = HashSet::new();
    insert_match_tokens(&mut tokens, &path_stem_without_multi_ext(path));
    if let Some(parent) = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
    {
        insert_match_tokens(&mut tokens, parent);
    }
    tokens
}

fn insert_match_tokens(tokens: &mut HashSet<String>, text: &str) {
    let lowered = text.to_ascii_lowercase();
    let parts = lowered
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|part| !part.is_empty());
    for part in parts {
        if is_generic_match_token(part) {
            continue;
        }
        tokens.insert(part.to_string());
    }
    let collapsed = normalize_match_string(text);
    if !collapsed.is_empty() {
        tokens.insert(collapsed);
    }
}

fn normalize_match_string(text: &str) -> String {
    text.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_lowercase())
        .collect()
}

fn is_generic_match_token(token: &str) -> bool {
    matches!(
        token,
        "ome"
            | "zarr"
            | "geojson"
            | "json"
            | "seg"
            | "segmentation"
            | "mask"
            | "masks"
            | "cells"
            | "cell"
            | "objects"
            | "object"
            | "polygon"
            | "polygons"
            | "outline"
            | "outlines"
            | "boundaries"
            | "boundary"
            | "image"
            | "images"
    )
}

fn path_stem_without_multi_ext(path: &Path) -> String {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_string();
    let lowered = name.to_ascii_lowercase();
    for suffix in [
        ".ome.zarr",
        ".spatialdata.zarr",
        ".zarr",
        ".geoparquet",
        ".parquet",
        ".geojson",
        ".json",
    ] {
        if lowered.ends_with(suffix) && name.len() > suffix.len() {
            return name[..name.len() - suffix.len()].to_string();
        }
    }
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_string()
}

fn is_segmentation_candidate_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "geoparquet" | "parquet" | "geojson" | "json"
            )
        })
        .unwrap_or(false)
}

fn segmentation_candidate_format_rank(path: &Path) -> i32 {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("geoparquet") => 2,
        Some("parquet") => 1,
        _ => 0,
    }
}

fn discover_omezarr_roots_under(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut stack = vec![root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let Ok(read_dir) = fs::read_dir(&dir) else {
            continue;
        };

        let mut is_omezarr_root = false;
        let mut child_dirs = Vec::new();
        for entry in read_dir.flatten() {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_file() {
                if path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|name| name == ".zattrs" || name == "zarr.json")
                {
                    is_omezarr_root = true;
                }
            } else if file_type.is_dir() {
                child_dirs.push(path);
            }
        }

        if is_omezarr_root {
            let canonical = dir.canonicalize().unwrap_or(dir.clone());
            if seen.insert(canonical.clone()) {
                out.push(canonical);
            }
            continue;
        }

        for child in child_dirs {
            stack.push(child);
        }
    }

    out.sort();
    out
}
