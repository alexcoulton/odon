use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use eframe::egui;
use rfd::FileDialog;
use serde::{Deserialize, Serialize};

use crate::data::dataset_kind::{
    LocalDatasetKind, can_open_in_mosaic, classify_local_dataset_path, normalize_local_dataset_path,
};
use crate::data::dataset_source::DatasetSource;
use crate::data::project_config::{
    ProjectConfig, ProjectLayerGroups, ProjectMaskLayer, ProjectRoi,
};
use crate::data::samplesheet::{
    SampleRow, SampleSheet, load_samplesheet_csv, write_samplesheet_csv,
};
use crate::objects::{ObjectProjectAnalysisState, ObjectProjectDisplayState};
use crate::ui::roi_browser::RoiBrowseState;

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
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ProjectChannelViewState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visible: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_rgb: Option<[u8; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset_world: Option<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotation_rad: Option<f32>,
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
    pub smooth_pixels: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_tile_debug: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_scale_bar: Option<bool>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProjectItem {
    pub path: PathBuf,
    #[serde(default)]
    pub display_name: Option<String>,
}

#[derive(Debug, Clone)]
pub enum ProjectSpaceAction {
    Open(ProjectRoi),
    OpenMosaic(Vec<ProjectRoi>),
    OpenRemoteDialog,
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
}

impl ProjectSpace {
    fn browser_state(&self) -> ProjectBrowserState {
        let mut selected = self.selected.iter().cloned().collect::<Vec<_>>();
        selected.sort();
        ProjectBrowserState {
            focused: self.focused.clone(),
            selected,
        }
    }

    fn state_for_save(&self) -> ProjectState {
        let mut state = self.state.clone();
        state.browser = self.browser_state();
        let valid_roi_keys = self
            .config
            .rois
            .iter()
            .filter_map(ProjectRoi::source_key)
            .collect::<HashSet<_>>();
        state
            .roi_views
            .retain(|source_key, _| valid_roi_keys.contains(source_key));
        state
    }

    pub fn config(&self) -> &ProjectConfig {
        &self.config
    }

    pub fn layer_groups(&self) -> &ProjectLayerGroups {
        &self.config.layer_groups
    }

    pub fn update_layer_groups(&mut self, f: impl FnOnce(&mut ProjectLayerGroups)) {
        f(&mut self.config.layer_groups);
        self.config_generation = self.config_generation.wrapping_add(1);
        self.config_json_dirty = true;
    }

    pub fn config_generation(&self) -> u64 {
        self.config_generation
    }

    pub fn load_from_file(&mut self, path: &Path) -> anyhow::Result<()> {
        self.load_path = path.to_string_lossy().to_string();
        self.load_from_path();
        if self.status.starts_with("Load failed:")
            || self.status.starts_with("Unsupported project version:")
        {
            anyhow::bail!("{}", self.status);
        }
        Ok(())
    }

    pub fn save_to_file(&mut self, path: &Path) -> anyhow::Result<()> {
        self.save_path = path.to_string_lossy().to_string();
        self.state = self.state_for_save();
        let file = ProjectFileV6 {
            version: 6,
            config: self.config.clone(),
            state: self.state.clone(),
        };
        let text = serde_json::to_string_pretty(&file)?;
        fs::write(path, text)?;
        self.project_file_path = Some(path.to_path_buf());
        self.status = format!("Saved: {}", path.to_string_lossy());
        Ok(())
    }

    pub fn set_status(&mut self, status: impl Into<String>) {
        self.status = status.into();
    }

    fn resolved_segmentation_search_roots(&self) -> Vec<PathBuf> {
        let mut roots = Vec::new();
        let mut seen = HashSet::new();
        let project_dir = self.project_dir();

        for root in &self.config.mosaic_segmentation_search_roots {
            let resolved = if root.is_relative() {
                project_dir
                    .as_ref()
                    .map(|dir| dir.join(root))
                    .unwrap_or_else(|| root.clone())
            } else {
                root.clone()
            };
            let resolved = resolved.canonicalize().unwrap_or(resolved);
            if seen.insert(resolved.clone()) {
                roots.push(resolved);
            }
        }

        if let Some(dir) = project_dir {
            let dir = dir.canonicalize().unwrap_or(dir);
            if seen.insert(dir.clone()) {
                roots.push(dir);
            }
        }

        for roi in &self.config.rois {
            if let Some(parent) = roi.local_path().and_then(|path| path.parent()) {
                let parent = parent
                    .canonicalize()
                    .unwrap_or_else(|_| parent.to_path_buf());
                if seen.insert(parent.clone()) {
                    roots.push(parent);
                }
            }
        }

        roots
    }

    fn add_segmentation_search_root(&mut self, root: PathBuf) {
        let root = root.canonicalize().unwrap_or(root);
        if self
            .config
            .mosaic_segmentation_search_roots
            .iter()
            .any(|existing| existing == &root)
        {
            return;
        }
        self.config.mosaic_segmentation_search_roots.push(root);
        self.config_generation = self.config_generation.wrapping_add(1);
    }

    fn auto_match_segmentations(&mut self, selected_only: bool) {
        let roots = self.resolved_segmentation_search_roots();
        if roots.is_empty() {
            self.status = "No segmentation search roots available.".to_string();
            return;
        }
        let candidates = collect_segmentation_candidates(&roots, 6);
        if candidates.is_empty() {
            self.status = "No segmentation candidates found in search roots.".to_string();
            return;
        }

        let mut matched = 0usize;
        let mut unmatched = 0usize;
        for roi in &mut self.config.rois {
            if selected_only
                && !roi
                    .source_key()
                    .as_ref()
                    .is_some_and(|key| self.selected.contains(key))
            {
                continue;
            }
            if roi.local_path().is_none() {
                continue;
            }
            if let Some(best) = best_segmentation_match_for_roi(roi, &candidates) {
                roi.segpath = Some(best.path.clone());
                matched += 1;
            } else {
                unmatched += 1;
            }
        }
        self.config_generation = self.config_generation.wrapping_add(1);
        self.status = match (matched, unmatched) {
            (0, _) => "No segmentation matches found.".to_string(),
            (_, 0) => format!("Matched segmentation for {matched} ROI(s)."),
            _ => format!("Matched {matched} ROI(s); {unmatched} unmatched."),
        };
    }

    pub fn rois(&self) -> &[ProjectRoi] {
        &self.config.rois
    }

    pub fn roi_mask_layers(&self, roi_path: &Path) -> Option<&[ProjectMaskLayer]> {
        let key = roi_path
            .canonicalize()
            .unwrap_or_else(|_| roi_path.to_path_buf());
        let key_s = key.to_string_lossy();
        self.config
            .rois
            .iter()
            .find(|it| {
                it.local_path()
                    .is_some_and(|path| path == key.as_path() || path.to_string_lossy() == key_s)
            })
            .map(|it| it.mask_layers.as_slice())
    }

    fn ensure_roi_for_source(&mut self, source: &DatasetSource) {
        let source_key = source.source_key();
        if self
            .config
            .rois
            .iter()
            .any(|roi| roi.source_key().as_deref() == Some(source_key.as_str()))
        {
            return;
        }

        let display_name = source.display_name();
        let default_dataset = self
            .config
            .default_dataset
            .clone()
            .unwrap_or_else(|| "default".to_string());
        let mut roi = ProjectRoi {
            id: display_name.clone(),
            source: None,
            path: None,
            dataset: source.is_local().then_some(default_dataset),
            display_name: Some(display_name),
            segpath: None,
            mask_layers: Vec::new(),
            channel_order: Vec::new(),
            meta: Default::default(),
        };
        roi.set_dataset_source(source.clone());
        self.config.rois.push(roi);
    }

    fn roi_view_state_mut(&mut self, source: &DatasetSource) -> &mut ProjectRoiViewState {
        self.ensure_roi_for_source(source);
        self.state.roi_views.entry(source.source_key()).or_default()
    }

    pub fn roi_view_state(&self, source: &DatasetSource) -> Option<&ProjectRoiViewState> {
        self.state.roi_views.get(&source.source_key())
    }

    pub fn set_roi_view_state(&mut self, source: &DatasetSource, view: ProjectRoiViewState) {
        let dst = self.roi_view_state_mut(source);
        if *dst == view {
            return;
        }
        *dst = view;
        self.config_generation = self.config_generation.wrapping_add(1);
    }

    pub fn mosaic_view_state(&self) -> Option<&ProjectMosaicViewState> {
        self.state.mosaic.as_ref()
    }

    pub fn set_mosaic_view_state(&mut self, view: ProjectMosaicViewState) {
        if self.state.mosaic.as_ref() == Some(&view) {
            return;
        }
        self.state.mosaic = Some(view);
        self.config_generation = self.config_generation.wrapping_add(1);
    }

    pub fn set_roi_mask_layers(&mut self, roi_path: &Path, layers: Vec<ProjectMaskLayer>) {
        let key = roi_path
            .canonicalize()
            .unwrap_or_else(|_| roi_path.to_path_buf());
        let key_s = key.to_string_lossy();
        if let Some(it) = self.config.rois.iter_mut().find(|it| {
            it.local_path()
                .is_some_and(|path| path == key.as_path() || path.to_string_lossy() == key_s)
        }) {
            it.mask_layers = layers;
            self.config_generation = self.config_generation.wrapping_add(1);
            return;
        }

        // If the ROI isn't part of the explicit list yet, add it (best-effort) so masks can be
        // persisted when the user saves the Project JSON.
        let display_name = key
            .file_name()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string());
        let id = display_name.clone().unwrap_or_else(|| "ROI".to_string());
        let mut roi = ProjectRoi {
            id,
            source: None,
            path: None,
            dataset: None,
            display_name,
            segpath: None,
            mask_layers: layers,
            channel_order: Vec::new(),
            meta: Default::default(),
        };
        roi.set_dataset_source(DatasetSource::Local(key));
        self.config.rois.push(roi);
        self.config_generation = self.config_generation.wrapping_add(1);
    }

    pub fn focused_roi(&self) -> Option<&ProjectRoi> {
        let key = self.focused.as_ref()?;
        self.config
            .rois
            .iter()
            .find(|roi| roi.source_key().as_deref() == Some(key.as_str()))
    }

    pub fn selected_rois(&self) -> Vec<ProjectRoi> {
        self.config
            .rois
            .iter()
            .filter(|roi| {
                roi.source_key()
                    .is_some_and(|key| self.selected.contains(key.as_str()))
            })
            .cloned()
            .collect()
    }

    pub fn rois_for_local_paths(&self, paths: &[PathBuf]) -> Vec<ProjectRoi> {
        let selected = paths.iter().collect::<HashSet<_>>();
        self.config
            .rois
            .iter()
            .filter(|roi| {
                roi.local_path()
                    .is_some_and(|path| selected.contains(&path.to_path_buf()))
            })
            .cloned()
            .collect()
    }

    pub fn add_roi_source(&mut self, source: DatasetSource) {
        let source_key = source.source_key();
        if self
            .config
            .rois
            .iter()
            .any(|roi| roi.source_key().as_deref() == Some(source_key.as_str()))
        {
            self.focused = Some(source_key.clone());
            self.selected.insert(source_key);
            return;
        }
        let display_name = source.display_name();
        let default_dataset = self
            .config
            .default_dataset
            .clone()
            .unwrap_or_else(|| "default".to_string());
        let mut roi = ProjectRoi {
            id: display_name.clone(),
            source: None,
            path: None,
            dataset: source.is_local().then_some(default_dataset),
            display_name: Some(display_name),
            segpath: None,
            mask_layers: Vec::new(),
            channel_order: Vec::new(),
            meta: Default::default(),
        };
        roi.set_dataset_source(source);
        self.config.rois.push(roi);
        if self.focused.is_none() {
            self.focused = Some(source_key.clone());
            self.selected.insert(source_key);
        }
        self.status.clear();
        self.config_generation = self.config_generation.wrapping_add(1);
    }

    pub fn project_dir(&self) -> Option<PathBuf> {
        self.project_file_path
            .as_ref()
            .and_then(|p| p.parent().map(Path::to_path_buf))
    }

    pub fn saved_project_path(&self) -> Option<PathBuf> {
        self.project_file_path.clone()
    }

    pub fn current_project_path(&self) -> Option<PathBuf> {
        self.project_file_path.clone().or_else(|| {
            (!self.save_path.trim().is_empty()).then(|| PathBuf::from(self.save_path.trim()))
        })
    }

    pub fn save_as_project(&mut self) {
        self.save_project_via_dialog("Save Project");
    }

    pub fn save_new_project(&mut self) {
        self.save_project_via_dialog("Save New Project");
    }

    fn save_project_via_dialog(&mut self, title: &str) {
        let default_name = self
            .current_project_path()
            .as_ref()
            .and_then(|path| path.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("project.json")
            .to_string();
        let mut dialog = FileDialog::new()
            .add_filter("Project JSON", &["json"])
            .set_file_name(&default_name)
            .set_title(title);
        if let Some(parent) = self
            .current_project_path()
            .as_ref()
            .and_then(|path| path.parent())
        {
            dialog = dialog.set_directory(parent);
        }
        if let Some(path) = dialog.save_file() {
            self.save_path = path.to_string_lossy().to_string();
            self.save_to_path();
        }
    }

    fn exportable_local_roi_count(&self) -> usize {
        self.config
            .rois
            .iter()
            .filter(|roi| roi.local_path().is_some())
            .count()
    }

    fn default_samplesheet_export_path(&self) -> PathBuf {
        let project_path = self.current_project_path();

        let stem = project_path
            .as_ref()
            .and_then(|path| path.file_stem())
            .and_then(|stem| stem.to_str())
            .filter(|stem| !stem.trim().is_empty())
            .unwrap_or("samplesheet");
        let stem = stem.strip_suffix(".project").unwrap_or(stem);
        let file_name = format!("{stem}.samplesheet.csv");

        project_path
            .as_ref()
            .and_then(|path| path.parent())
            .map(|dir| dir.join(&file_name))
            .unwrap_or_else(|| PathBuf::from(file_name))
    }

    fn export_samplesheet_csv(&mut self, path: &Path) -> anyhow::Result<()> {
        let mut meta_columns = BTreeSet::new();
        let mut rows = Vec::new();
        let mut skipped_non_local = 0usize;

        for roi in &self.config.rois {
            let Some(local_path) = roi.local_path() else {
                skipped_non_local += 1;
                continue;
            };

            let mut meta = roi
                .meta
                .iter()
                .filter_map(|(key, value)| {
                    let key = key.trim();
                    (!key.is_empty()).then(|| (key.to_string(), value.clone()))
                })
                .collect::<std::collections::HashMap<_, _>>();

            if let Some(dataset) = roi.dataset.as_ref().filter(|s| !s.trim().is_empty()) {
                meta.insert("dataset".to_string(), dataset.clone());
            }

            if let Some(segpath) = roi.segpath.as_ref() {
                let segpath = segpath
                    .canonicalize()
                    .unwrap_or_else(|_| segpath.to_path_buf());
                meta.insert("segpath".to_string(), segpath.to_string_lossy().to_string());
            }

            for key in meta.keys() {
                meta_columns.insert(key.clone());
            }

            let id = if roi.id.trim().is_empty() {
                roi.display_name
                    .clone()
                    .filter(|s| !s.trim().is_empty())
                    .unwrap_or_else(|| {
                        local_path
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("ROI")
                            .to_string()
                    })
            } else {
                roi.id.trim().to_string()
            };

            rows.push(SampleRow {
                id,
                path: local_path
                    .canonicalize()
                    .unwrap_or_else(|_| local_path.to_path_buf()),
                meta,
            });
        }

        if rows.is_empty() {
            anyhow::bail!("project has no local ROIs to export");
        }

        let exported_count = rows.len();
        let sheet = SampleSheet {
            meta_columns: meta_columns.into_iter().collect(),
            rows,
        };
        write_samplesheet_csv(path, &sheet)?;

        self.status = if skipped_non_local > 0 {
            format!(
                "Exported {exported_count} ROI(s) to {}; skipped {skipped_non_local} non-local ROI(s).",
                path.to_string_lossy()
            )
        } else {
            format!(
                "Exported {exported_count} ROI(s) to {}.",
                path.to_string_lossy()
            )
        };

        Ok(())
    }

    pub fn set_current_dataset_root(&mut self, root: Option<&Path>) {
        let Some(root) = root else {
            if self.save_path.trim().is_empty() {
                self.save_path = "odon.project.json".to_string();
            }
            if self.load_path.trim().is_empty() {
                self.load_path = self.save_path.clone();
            }
            return;
        };
        let root_s = root.to_string_lossy();
        if self.save_path.is_empty() {
            self.save_path = format!("{root_s}.project.json");
        }
        if self.load_path.is_empty() {
            self.load_path = self.save_path.clone();
        }
    }

    pub fn handle_dropped_paths(&mut self, paths: impl IntoIterator<Item = PathBuf>) {
        for p in paths {
            if let Some(root) = normalize_local_dataset_path(&p) {
                self.add_roi(root);
                continue;
            }
            self.status = format!("Unsupported dataset: {}", p.to_string_lossy());
        }
    }

    fn add_roi(&mut self, root: PathBuf) {
        let root = root.canonicalize().unwrap_or(root);
        let Some(kind) = classify_local_dataset_path(&root) else {
            self.status = format!("Not a supported dataset root: {}", root.to_string_lossy());
            return;
        };
        let mut source = DatasetSource::Local(root);
        if !matches!(kind, LocalDatasetKind::OmeZarr) {
            // TIFF/Xenium stay local but don't default into the mosaic dataset selector.
            source = DatasetSource::Local(source.local_path().unwrap().to_path_buf());
        }
        self.add_roi_source(source);
    }

    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        current_dataset_root: Option<&Path>,
    ) -> Option<ProjectSpaceAction> {
        self.set_current_dataset_root(current_dataset_root);
        let mut action = None;

        ui.heading("Project");
        ui.label("Drag OME-Zarr folders, Xenium folders, or TIFF files onto the window.");
        ui.label(format!(
            "{} ROI(s), {} selected",
            self.config.rois.len(),
            self.selected.len()
        ));
        ui.add_space(6.0);

        ui.separator();
        ui.horizontal(|ui| {
            if ui.button("Import Samplesheet CSV...").clicked() {
                if let Some(path) = FileDialog::new()
                    .add_filter("CSV", &["csv"])
                    .set_title("Import Samplesheet CSV")
                    .pick_file()
                {
                    if let Err(err) = self.import_rois_from_csv(&path) {
                        self.status = format!("Import failed: {err}");
                    }
                }
            }
            if ui
                .add_enabled(
                    self.exportable_local_roi_count() > 0,
                    egui::Button::new("Export Samplesheet CSV..."),
                )
                .clicked()
            {
                let default_path = self.default_samplesheet_export_path();
                let default_name = default_path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("samplesheet.csv")
                    .to_string();
                let mut dialog = FileDialog::new()
                    .add_filter("CSV", &["csv"])
                    .set_file_name(&default_name)
                    .set_title("Export Samplesheet CSV");
                if let Some(parent) = default_path.parent() {
                    dialog = dialog.set_directory(parent);
                }
                if let Some(path) = dialog.save_file() {
                    if let Err(err) = self.export_samplesheet_csv(&path) {
                        self.status = format!("Export failed: {err}");
                    }
                }
            }
            if ui.button("Add OME-Zarr Root...").clicked() {
                if let Some(path) = FileDialog::new()
                    .set_title("Choose Folder Containing OME-Zarr ROIs")
                    .pick_folder()
                {
                    if let Err(err) = self.import_rois_from_root(&path) {
                        self.status = format!("OME-Zarr root import failed: {err}");
                    }
                }
            }
            if ui.button("Open Remote...").clicked() {
                action = Some(ProjectSpaceAction::OpenRemoteDialog);
            }
            if ui.button("Add Seg Search Root...").clicked() {
                if let Some(path) = FileDialog::new()
                    .set_title("Add Segmentation Search Root")
                    .pick_folder()
                {
                    self.add_segmentation_search_root(path);
                }
            }
            if ui
                .add_enabled(
                    !self.config.rois.is_empty(),
                    egui::Button::new("Auto-match Seg"),
                )
                .clicked()
            {
                self.auto_match_segmentations(false);
            }
            if ui.button("Edit config (JSON)").clicked() {
                self.config_json_dirty = false;
                self.config_json =
                    serde_json::to_string_pretty(&self.config).unwrap_or_else(|_| "{}".to_string());
                self.config_json_status.clear();
            }
        });
        ui.add_space(6.0);
        ui.label("Browse ROIs");
        let browse = crate::ui::roi_browser::ui(
            ui,
            "project-space-roi-browser",
            &self.config.rois,
            None,
            &mut self.roi_browse,
        );
        ui.label(format!(
            "Showing {} / {} ROI(s).",
            browse.filtered_indices.len(),
            browse.total_count
        ));
        if browse.changed || self.roi_browse.has_filters() {
            self.sync_filtered_selection(&browse.filtered_indices);
        }
        let seg_roots = self.resolved_segmentation_search_roots();
        if !seg_roots.is_empty() {
            ui.add_space(4.0);
            ui.collapsing("Segmentation search roots", |ui| {
                let mut remove_idx = None;
                for (idx, root) in self
                    .config
                    .mosaic_segmentation_search_roots
                    .iter()
                    .enumerate()
                {
                    ui.horizontal(|ui| {
                        ui.label(root.to_string_lossy());
                        if ui.small_button("Remove").clicked() {
                            remove_idx = Some(idx);
                        }
                    });
                }
                if self.config.mosaic_segmentation_search_roots.is_empty() {
                    ui.label("Using project directory and ROI parent folders.");
                }
                if let Some(idx) = remove_idx {
                    self.config.mosaic_segmentation_search_roots.remove(idx);
                    self.config_generation = self.config_generation.wrapping_add(1);
                }
            });
        }
        if !self.config_json.is_empty() {
            ui.add_space(6.0);
            ui.label("Project config (JSON)");
            let resp = ui.add(
                egui::TextEdit::multiline(&mut self.config_json)
                    .desired_rows(10)
                    .font(egui::TextStyle::Monospace),
            );
            if resp.changed() {
                self.config_json_dirty = true;
            }
            ui.horizontal(|ui| {
                if ui.button("Apply").clicked() {
                    match serde_json::from_str::<ProjectConfig>(&self.config_json) {
                        Ok(cfg) => {
                            self.config = cfg;
                            self.config_generation = self.config_generation.wrapping_add(1);
                            self.config_json_dirty = false;
                            self.config_json_status = "Applied.".to_string();
                        }
                        Err(err) => self.config_json_status = format!("Parse error: {err}"),
                    }
                }
                if ui.button("Refresh").clicked() {
                    self.config_json_dirty = false;
                    self.config_json = serde_json::to_string_pretty(&self.config)
                        .unwrap_or_else(|_| "{}".to_string());
                    self.config_json_status.clear();
                }
                if ui.button("Close").clicked() {
                    self.config_json.clear();
                    self.config_json_status.clear();
                    self.config_json_dirty = false;
                }
            });
            if !self.config_json_status.is_empty() {
                ui.label(self.config_json_status.clone());
            }
            ui.separator();
        }

        let mut clicked_index: Option<usize> = None;
        let mut clicked_key: Option<String> = None;
        let mut clicked_double: bool = false;

        let select_all_shortcut = ui.input(|i| i.key_pressed(egui::Key::A) && i.modifiers.command);
        if select_all_shortcut
            && !ui.ctx().wants_keyboard_input()
            && !browse.filtered_indices.is_empty()
        {
            self.selected.clear();
            for &roi_idx in &browse.filtered_indices {
                if let Some(key) = self
                    .config
                    .rois
                    .get(roi_idx)
                    .and_then(ProjectRoi::source_key)
                {
                    self.selected.insert(key);
                }
            }
            if self.focused.is_none() {
                self.focused = browse
                    .filtered_indices
                    .first()
                    .and_then(|&idx| self.config.rois.get(idx).and_then(ProjectRoi::source_key));
            }
        }

        let visible_indices = browse.filtered_indices;

        ui.label(format!("ROIs ({})", visible_indices.len()));
        egui::Frame::group(ui.style()).show(ui, |ui| {
            let list_height = ui.available_height().clamp(180.0, 280.0);
            ui.set_min_height(list_height);
            egui::ScrollArea::vertical()
                .id_salt("project-roi-list")
                .max_height(list_height)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    if visible_indices.is_empty() {
                        ui.label("No ROIs match the current browse filters.");
                    }
                    for &roi_idx in &visible_indices {
                        let Some(it) = self.config.rois.get(roi_idx) else {
                            continue;
                        };
                        ui.push_id(roi_idx, |ui| {
                            let name = it.display_name.clone().unwrap_or_else(|| it.id.clone());
                            let source_key = it.source_key();
                            let tooltip = it.source_display();
                            let is_selected = source_key
                                .as_ref()
                                .is_some_and(|key| self.selected.contains(key));
                            let is_focused = source_key
                                .as_ref()
                                .zip(self.focused.as_ref())
                                .is_some_and(|(key, focused)| key == focused);
                            let row_label = if it.segpath.is_some() {
                                format!("{name}  [seg]")
                            } else {
                                name
                            };

                            let row_width = ui.available_width();
                            let resp = ui.add_sized(
                                [row_width, 0.0],
                                egui::Button::new(row_label).selected(is_selected),
                            );
                            let resp = resp.on_hover_text(tooltip);
                            if resp.clicked() || resp.double_clicked() {
                                clicked_index = Some(roi_idx);
                                clicked_key = source_key;
                                clicked_double = resp.double_clicked();
                            }

                            if is_focused {
                                let stroke = ui.visuals().selection.stroke;
                                let stroke = egui::Stroke::new(stroke.width.max(2.0), stroke.color);
                                ui.painter().rect_stroke(
                                    resp.rect.shrink(0.5),
                                    egui::CornerRadius::same(6),
                                    stroke,
                                    egui::StrokeKind::Middle,
                                );
                            }
                            ui.separator();
                        });
                    }
                });
        });

        if let (Some(i), Some(source_key)) = (clicked_index, clicked_key.clone()) {
            let modifiers = ui.input(|inp| inp.modifiers);
            let cmd = modifiers.command;
            let shift = modifiers.shift;

            let focused_idx = self.focused.as_ref().and_then(|p| {
                visible_indices.iter().position(|&idx| {
                    self.config
                        .rois
                        .get(idx)
                        .and_then(ProjectRoi::source_key)
                        .as_deref()
                        == Some(p.as_str())
                })
            });
            let clicked_visible_idx = visible_indices
                .iter()
                .position(|&idx| idx == i)
                .unwrap_or_default();

            if shift {
                self.selected.clear();
                if let Some(fi) = focused_idx {
                    let a = fi.min(clicked_visible_idx);
                    let b = fi.max(clicked_visible_idx);
                    for &roi_idx in &visible_indices[a..=b] {
                        if let Some(key) = self
                            .config
                            .rois
                            .get(roi_idx)
                            .and_then(ProjectRoi::source_key)
                        {
                            self.selected.insert(key);
                        }
                    }
                } else {
                    self.selected.insert(source_key.clone());
                }
                self.focused = Some(source_key.clone());
            } else if cmd {
                if self.selected.contains(source_key.as_str()) {
                    self.selected.remove(source_key.as_str());
                } else {
                    self.selected.insert(source_key.clone());
                }
                self.focused = Some(source_key.clone());
            } else {
                self.selected.clear();
                self.selected.insert(source_key.clone());
                self.focused = Some(source_key.clone());
            }

            if clicked_double {
                if let Some(roi) = self.config.rois.get(i).cloned() {
                    action = Some(ProjectSpaceAction::Open(roi));
                }
            }
        }

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            let selected_count = self.selected.len();
            let all_visible_selected = !visible_indices.is_empty()
                && visible_indices.iter().all(|&idx| {
                    self.config
                        .rois
                        .get(idx)
                        .and_then(ProjectRoi::source_key)
                        .as_ref()
                        .is_some_and(|key| self.selected.contains(key))
                });
            let mosaic_enabled = selected_count >= 2
                && self
                    .selected_rois()
                    .iter()
                    .all(|roi| roi.local_path().map(can_open_in_mosaic).unwrap_or(true));
            let open_clicked = ui
                .add_enabled(self.focused_roi().is_some(), egui::Button::new("Open"))
                .clicked();
            let open_mosaic_clicked = ui
                .add_enabled(
                    mosaic_enabled,
                    egui::Button::new(format!("Open mosaic ({selected_count})")),
                )
                .clicked();
            let select_all_clicked = ui
                .add_enabled(
                    !visible_indices.is_empty(),
                    egui::Button::new(if self.roi_browse.has_filters() {
                        if all_visible_selected {
                            "All visible selected"
                        } else {
                            "Select visible"
                        }
                    } else {
                        "Select all"
                    }),
                )
                .clicked();
            let remove_clicked = ui
                .add_enabled(selected_count > 0, egui::Button::new("Remove"))
                .clicked();
            let clear_clicked = ui
                .add_enabled(!self.config.rois.is_empty(), egui::Button::new("Clear"))
                .clicked();

            if open_clicked {
                if let Some(roi) = self.focused_roi().cloned() {
                    action = Some(ProjectSpaceAction::Open(roi));
                }
            }
            if open_mosaic_clicked {
                let mut rois = self.selected_rois();
                rois.sort_by(|a, b| a.source_display().cmp(&b.source_display()));
                action = Some(ProjectSpaceAction::OpenMosaic(rois));
            }
            if select_all_clicked {
                self.selected.clear();
                for &roi_idx in &visible_indices {
                    if let Some(key) = self
                        .config
                        .rois
                        .get(roi_idx)
                        .and_then(ProjectRoi::source_key)
                    {
                        self.selected.insert(key);
                    }
                }
                if self.focused.is_none() || self.roi_browse.has_filters() {
                    self.focused = visible_indices.first().and_then(|&idx| {
                        self.config.rois.get(idx).and_then(ProjectRoi::source_key)
                    });
                }
            }
            if remove_clicked {
                if !self.selected.is_empty() {
                    let mut keep: Vec<ProjectRoi> = Vec::with_capacity(self.config.rois.len());
                    for it in self.config.rois.drain(..) {
                        if !it
                            .source_key()
                            .as_ref()
                            .is_some_and(|key| self.selected.contains(key))
                        {
                            keep.push(it);
                        }
                    }
                    self.config.rois = keep;
                    self.selected.clear();
                    self.focused = self.config.rois.first().and_then(ProjectRoi::source_key);
                    if let Some(p) = self.focused.clone() {
                        self.selected.insert(p);
                    }
                    self.config_generation = self.config_generation.wrapping_add(1);
                }
            }
            if clear_clicked {
                self.config.rois.clear();
                self.focused = None;
                self.selected.clear();
                self.config_generation = self.config_generation.wrapping_add(1);
            }
        });

        ui.add_space(8.0);
        egui::ScrollArea::vertical()
            .id_salt("project-details-scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.separator();
                ui.heading("ROI Details");
                let focused_idx = self.focused.as_ref().and_then(|focused| {
                    self.config
                        .rois
                        .iter()
                        .position(|roi| roi.source_key().as_deref() == Some(focused.as_str()))
                });
                if let Some(idx) = focused_idx {
                    let mut changed = false;
                    let mut choose_seg = false;
                    let mut clear_seg = false;
                    let mut remove_key = None::<String>;
                    let mut rename_meta = None::<(String, String, String)>;
                    let mut add_meta = None::<(String, String)>;
                    let mut new_meta_key = self.new_meta_key.clone();
                    let mut new_meta_value = self.new_meta_value.clone();

                    {
                        let roi = &mut self.config.rois[idx];
                        ui.horizontal(|ui| {
                            ui.label("ID");
                            changed |= ui.text_edit_singleline(&mut roi.id).changed();
                        });
                        ui.horizontal(|ui| {
                            ui.label("Display");
                            let display = roi.display_name.get_or_insert_with(|| roi.id.clone());
                            changed |= ui.text_edit_singleline(display).changed();
                        });
                        ui.horizontal(|ui| {
                            ui.label("Image");
                            let mut path_text = roi.source_display();
                            ui.add_sized(
                                [ui.available_width(), 0.0],
                                egui::TextEdit::singleline(&mut path_text),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("Segmentation");
                            let mut seg_text = roi
                                .segpath
                                .as_ref()
                                .map(|p| p.to_string_lossy().to_string())
                                .unwrap_or_default();
                            ui.add(egui::TextEdit::singleline(&mut seg_text).desired_width(320.0));
                            if ui.button("Choose...").clicked() {
                                choose_seg = true;
                            }
                            if ui
                                .add_enabled(roi.segpath.is_some(), egui::Button::new("Clear"))
                                .clicked()
                            {
                                clear_seg = true;
                            }
                        });

                        ui.add_space(6.0);
                        ui.label("Metadata");
                        let mut meta_keys = roi.meta.keys().cloned().collect::<Vec<_>>();
                        meta_keys.sort();
                        for key in meta_keys {
                            let existing = roi.meta.get(&key).cloned().unwrap_or_default();
                            let mut label = key.clone();
                            let mut value = existing.clone();
                            let mut remove_clicked = false;
                            ui.horizontal(|ui| {
                                changed |= ui.text_edit_singleline(&mut label).changed();
                                changed |= ui.text_edit_singleline(&mut value).changed();
                                if ui.small_button("Remove").clicked() {
                                    remove_clicked = true;
                                }
                            });
                            if remove_clicked {
                                remove_key = Some(key.clone());
                            } else {
                                if value != existing {
                                    roi.meta.insert(key.clone(), value.clone());
                                }
                                if label != key
                                    && !label.trim().is_empty()
                                    && !roi.meta.contains_key(&label)
                                {
                                    rename_meta =
                                        Some((key.clone(), label.trim().to_string(), value));
                                }
                            }
                        }
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::TextEdit::singleline(&mut new_meta_key).hint_text("column"),
                            );
                            ui.add(
                                egui::TextEdit::singleline(&mut new_meta_value).hint_text("value"),
                            );
                            let can_add = !new_meta_key.trim().is_empty();
                            if ui
                                .add_enabled(can_add, egui::Button::new("Add metadata"))
                                .clicked()
                            {
                                add_meta = Some((
                                    new_meta_key.trim().to_string(),
                                    new_meta_value.trim().to_string(),
                                ));
                            }
                        });
                    }

                    self.new_meta_key = new_meta_key;
                    self.new_meta_value = new_meta_value;

                    if clear_seg {
                        self.config.rois[idx].segpath = None;
                        changed = true;
                    }
                    if choose_seg {
                        if let Some(path) = FileDialog::new()
                            .add_filter("GeoJSON", &["geojson", "json"])
                            .add_filter("GeoParquet", &["geoparquet", "parquet"])
                            .set_title("Choose ROI Segmentation")
                            .pick_file()
                        {
                            self.config.rois[idx].segpath = Some(path);
                            changed = true;
                        }
                    }
                    if let Some(key) = remove_key {
                        self.config.rois[idx].meta.remove(&key);
                        changed = true;
                    }
                    if let Some((old_key, new_key, value)) = rename_meta {
                        self.config.rois[idx].meta.remove(&old_key);
                        self.config.rois[idx].meta.insert(new_key, value);
                        changed = true;
                    }
                    if let Some((key, value)) = add_meta {
                        self.config.rois[idx].meta.insert(key, value);
                        self.new_meta_key.clear();
                        self.new_meta_value.clear();
                        changed = true;
                    }

                    if changed {
                        self.config_generation = self.config_generation.wrapping_add(1);
                    }
                } else {
                    ui.label("Select an ROI to edit its segmentation path and metadata.");
                }

                ui.add_space(8.0);
                ui.separator();
                ui.label("Save / Load");

                ui.horizontal(|ui| {
                    if ui.button("New Project").clicked() {
                        self.config = ProjectConfig::default();
                        self.focused = None;
                        self.selected.clear();
                        self.roi_browse.clear();
                        self.config_generation = self.config_generation.wrapping_add(1);
                        self.status = "New project.".to_string();
                    }
                    if ui.button("Save").clicked() {
                        if self.saved_project_path().is_some() {
                            if let Some(path) = self.saved_project_path() {
                                self.save_path = path.to_string_lossy().to_string();
                            }
                            self.save_to_path();
                        } else {
                            self.save_as_project();
                        }
                    }
                    if ui.button("Save As...").clicked() {
                        self.save_as_project();
                    }
                    if ui.button("Load Project...").clicked() {
                        if let Some(path) = FileDialog::new()
                            .add_filter("Project JSON", &["json"])
                            .set_title("Load Project")
                            .pick_file()
                        {
                            self.load_path = path.to_string_lossy().to_string();
                            self.load_from_path();
                        }
                    }
                });

                if !self.status.is_empty() {
                    ui.add_space(6.0);
                    ui.label(self.status.clone());
                }
            });

        action
    }

    fn import_rois_from_csv(&mut self, path: &Path) -> anyhow::Result<()> {
        let sheet = load_samplesheet_csv(path)?;
        let base_dir = path.parent();
        self.config.rois.clear();
        self.roi_browse.clear();
        let default_dataset = self
            .config
            .default_dataset
            .clone()
            .unwrap_or_else(|| "default".to_string());
        for row in sheet.rows {
            let meta = row.meta;
            let resolved_path = if row.path.is_relative() {
                base_dir
                    .map(|dir| dir.join(&row.path))
                    .unwrap_or_else(|| row.path.clone())
            } else {
                row.path.clone()
            };
            let resolved_path = resolved_path.canonicalize().unwrap_or(resolved_path);
            let segpath = meta
                .get("segpath")
                .filter(|s| !s.trim().is_empty())
                .map(PathBuf::from)
                .map(|seg| {
                    if seg.is_relative() {
                        base_dir.map(|dir| dir.join(&seg)).unwrap_or(seg)
                    } else {
                        seg
                    }
                })
                .map(|seg| seg.canonicalize().unwrap_or(seg));
            let dataset = meta
                .get("dataset")
                .filter(|s| !s.trim().is_empty())
                .cloned()
                .or_else(|| Some(default_dataset.clone()));
            let mut roi = ProjectRoi {
                id: row.id.clone(),
                source: None,
                path: None,
                dataset,
                display_name: Some(row.id),
                segpath,
                mask_layers: Vec::new(),
                channel_order: Vec::new(),
                meta,
            };
            roi.set_dataset_source(DatasetSource::Local(resolved_path));
            self.config.rois.push(roi);
        }
        self.focused = self.config.rois.first().and_then(ProjectRoi::source_key);
        self.selected.clear();
        if let Some(p) = self.focused.clone() {
            self.selected.insert(p);
        }
        self.config_generation = self.config_generation.wrapping_add(1);
        self.status = format!(
            "Imported {} ROIs from samplesheet ({} metadata columns).",
            self.config.rois.len(),
            sheet.meta_columns.len()
        );
        Ok(())
    }

    fn sync_filtered_selection(&mut self, visible_indices: &[usize]) {
        let visible_keys = visible_indices
            .iter()
            .filter_map(|&idx| self.config.rois.get(idx).and_then(ProjectRoi::source_key))
            .collect::<HashSet<_>>();

        self.selected.retain(|key| visible_keys.contains(key));

        if self
            .focused
            .as_ref()
            .is_some_and(|key| !visible_keys.contains(key))
        {
            self.focused = None;
        }
        if self.focused.is_none() {
            self.focused = visible_indices
                .first()
                .and_then(|&idx| self.config.rois.get(idx).and_then(ProjectRoi::source_key));
        }
        if self.selected.is_empty() {
            if let Some(key) = self.focused.clone() {
                self.selected.insert(key);
            }
        }
    }

    fn import_rois_from_root(&mut self, root: &Path) -> anyhow::Result<()> {
        let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
        if !root.is_dir() {
            anyhow::bail!("not a directory: {}", root.to_string_lossy());
        }

        let before = self.config.rois.len();
        let roots = discover_omezarr_roots_under(&root);
        if roots.is_empty() {
            anyhow::bail!(
                "no OME-Zarr datasets found under {}",
                root.to_string_lossy()
            );
        }
        for roi_root in roots {
            self.add_roi(roi_root);
        }
        let added = self.config.rois.len().saturating_sub(before);
        self.status = format!(
            "Added {added} OME-Zarr ROI(s) from {}.",
            root.to_string_lossy()
        );
        Ok(())
    }

    fn save_to_path(&mut self) {
        let path = PathBuf::from(self.save_path.trim());
        if path.as_os_str().is_empty() {
            self.status = "Save path is empty.".to_string();
            return;
        }
        self.state = self.state_for_save();
        let file = ProjectFileV6 {
            version: 6,
            config: self.config.clone(),
            state: self.state.clone(),
        };
        match serde_json::to_string_pretty(&file) {
            Ok(text) => match fs::write(&path, text) {
                Ok(_) => {
                    self.project_file_path = Some(path.clone());
                    self.status = format!("Saved: {}", path.to_string_lossy());
                }
                Err(e) => self.status = format!("Save failed: {e}"),
            },
            Err(e) => self.status = format!("Save failed: {e}"),
        }
    }

    fn load_from_path(&mut self) {
        let path = PathBuf::from(self.load_path.trim());
        if path.as_os_str().is_empty() {
            self.status = "Load path is empty.".to_string();
            return;
        }
        let text = match fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                self.status = format!("Load failed: {e}");
                return;
            }
        };
        let version = serde_json::from_str::<serde_json::Value>(&text)
            .ok()
            .and_then(|v| v.get("version").and_then(|x| x.as_u64()))
            .unwrap_or(1);

        self.config = ProjectConfig::default();
        self.state = ProjectState::default();
        self.roi_browse.clear();

        let (mut config, mut state): (ProjectConfig, ProjectState) = match version {
            1 => {
                let file: ProjectFileV1 = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(e) => {
                        self.status = format!("Load failed: {e}");
                        return;
                    }
                };
                let focused = file.selected.and_then(|i| {
                    file.items
                        .get(i)
                        .map(|it| DatasetSource::Local(it.path.clone()).source_key())
                });
                let rois = file
                    .items
                    .into_iter()
                    .map(|it| {
                        let mut roi = ProjectRoi {
                            id: it
                                .display_name
                                .clone()
                                .unwrap_or_else(|| it.path.to_string_lossy().to_string()),
                            source: None,
                            path: None,
                            dataset: None,
                            display_name: it.display_name,
                            segpath: None,
                            mask_layers: Vec::new(),
                            channel_order: Vec::new(),
                            meta: Default::default(),
                        };
                        roi.set_dataset_source(DatasetSource::Local(it.path));
                        roi
                    })
                    .collect();
                (
                    ProjectConfig {
                        rois,
                        ..Default::default()
                    },
                    ProjectState {
                        browser: ProjectBrowserState {
                            focused,
                            selected: Vec::new(),
                        },
                        ..Default::default()
                    },
                )
            }
            2 => {
                let file: ProjectFileV2 = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(e) => {
                        self.status = format!("Load failed: {e}");
                        return;
                    }
                };
                let rois = file
                    .items
                    .into_iter()
                    .map(|it| {
                        let mut roi = ProjectRoi {
                            id: it
                                .display_name
                                .clone()
                                .unwrap_or_else(|| it.path.to_string_lossy().to_string()),
                            source: None,
                            path: None,
                            dataset: None,
                            display_name: it.display_name,
                            segpath: None,
                            mask_layers: Vec::new(),
                            channel_order: Vec::new(),
                            meta: Default::default(),
                        };
                        roi.set_dataset_source(DatasetSource::Local(it.path));
                        roi
                    })
                    .collect();
                (
                    ProjectConfig {
                        rois,
                        ..Default::default()
                    },
                    ProjectState {
                        browser: ProjectBrowserState {
                            focused: file
                                .focused
                                .map(|path| DatasetSource::Local(path).source_key()),
                            selected: file
                                .selected
                                .into_iter()
                                .map(|path| DatasetSource::Local(path).source_key())
                                .collect(),
                        },
                        ..Default::default()
                    },
                )
            }
            3 => {
                let file: ProjectFileV3Legacy = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(e) => {
                        self.status = format!("Load failed: {e}");
                        return;
                    }
                };
                let rois = file
                    .items
                    .into_iter()
                    .map(|it| {
                        let mut roi = ProjectRoi {
                            id: it
                                .display_name
                                .clone()
                                .unwrap_or_else(|| it.path.to_string_lossy().to_string()),
                            source: None,
                            path: None,
                            dataset: None,
                            display_name: it.display_name,
                            segpath: None,
                            mask_layers: Vec::new(),
                            channel_order: Vec::new(),
                            meta: Default::default(),
                        };
                        roi.set_dataset_source(DatasetSource::Local(it.path));
                        roi
                    })
                    .collect();
                (
                    ProjectConfig {
                        rois,
                        ..Default::default()
                    },
                    ProjectState {
                        browser: ProjectBrowserState {
                            focused: file
                                .focused
                                .map(|path| DatasetSource::Local(path).source_key()),
                            selected: file
                                .selected
                                .into_iter()
                                .map(|path| DatasetSource::Local(path).source_key())
                                .collect(),
                        },
                        ..Default::default()
                    },
                )
            }
            4 => {
                let file: ProjectFileV4 = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(e) => {
                        self.status = format!("Load failed: {e}");
                        return;
                    }
                };
                (
                    file.config,
                    ProjectState {
                        browser: ProjectBrowserState {
                            focused: file
                                .focused
                                .map(|path| DatasetSource::Local(path).source_key()),
                            selected: file
                                .selected
                                .into_iter()
                                .map(|path| DatasetSource::Local(path).source_key())
                                .collect(),
                        },
                        ..Default::default()
                    },
                )
            }
            5 => {
                let file: ProjectFileV5 = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(e) => {
                        self.status = format!("Load failed: {e}");
                        return;
                    }
                };
                (
                    file.config,
                    ProjectState {
                        browser: ProjectBrowserState {
                            focused: file.focused,
                            selected: file.selected,
                        },
                        ..Default::default()
                    },
                )
            }
            6 => {
                let file: ProjectFileV6 = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(e) => {
                        self.status = format!("Load failed: {e}");
                        return;
                    }
                };
                (file.config, file.state)
            }
            _ => {
                self.status = format!("Unsupported project version: {version}");
                return;
            }
        };
        self.project_file_path = Some(path.clone());
        self.save_path = path.to_string_lossy().to_string();
        self.load_path = path.to_string_lossy().to_string();

        let mut seen: HashSet<String> = HashSet::new();
        let mut cleaned: Vec<ProjectRoi> = Vec::new();
        let default_dataset = config
            .default_dataset
            .clone()
            .unwrap_or_else(|| "default".to_string());
        let rois = std::mem::take(&mut config.rois);
        for mut roi in rois.into_iter() {
            let Some(source) = roi.dataset_source() else {
                continue;
            };
            let dedupe_key = source.source_key();
            if !seen.insert(dedupe_key) {
                continue;
            }
            match source {
                DatasetSource::Local(path) => {
                    let p = path.canonicalize().unwrap_or(path);
                    let Some(kind) = classify_local_dataset_path(&p) else {
                        continue;
                    };
                    if roi.display_name.is_none() {
                        roi.display_name = p
                            .file_name()
                            .and_then(|s| s.to_str())
                            .map(|s| s.to_string());
                    }
                    if roi.id.trim().is_empty() {
                        roi.id = roi
                            .display_name
                            .clone()
                            .unwrap_or_else(|| p.to_string_lossy().to_string());
                    }
                    if roi
                        .dataset
                        .as_deref()
                        .map(|s| s.trim().is_empty())
                        .unwrap_or(true)
                        && matches!(kind, LocalDatasetKind::OmeZarr)
                    {
                        roi.dataset = Some(default_dataset.clone());
                    }
                    roi.set_dataset_source(DatasetSource::Local(p));
                }
                other => {
                    if roi.display_name.is_none() {
                        roi.display_name = Some(other.display_name());
                    }
                    if roi.id.trim().is_empty() {
                        roi.id = roi
                            .display_name
                            .clone()
                            .unwrap_or_else(|| other.display_name());
                    }
                    roi.set_dataset_source(other);
                }
            }
            cleaned.push(roi);
        }

        for roi in &mut cleaned {
            let key = roi.source_key();
            if let Some(key) = key {
                if !roi.channel_order.is_empty() {
                    let view = state.roi_views.entry(key).or_default();
                    if view.channel_order.is_empty() {
                        view.channel_order = std::mem::take(&mut roi.channel_order);
                    } else {
                        roi.channel_order.clear();
                    }
                }
            }
        }

        config.rois = cleaned;
        self.config = config;
        self.state = state;
        self.config_generation = self.config_generation.wrapping_add(1);

        self.focused = self
            .state
            .browser
            .focused
            .clone()
            .filter(|key| {
                self.config
                    .rois
                    .iter()
                    .any(|it| it.source_key().as_deref() == Some(key.as_str()))
            })
            .or_else(|| self.config.rois.first().and_then(ProjectRoi::source_key));

        self.selected.clear();
        for key in self.state.browser.selected.clone() {
            if self
                .config
                .rois
                .iter()
                .any(|it| it.source_key().as_deref() == Some(key.as_str()))
            {
                self.selected.insert(key);
            }
        }
        if self.selected.is_empty() {
            if let Some(p) = self.focused.clone() {
                self.selected.insert(p);
            }
        }
        self.status = format!("Loaded: {}", path.to_string_lossy());
    }
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
