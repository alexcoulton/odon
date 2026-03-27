use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use eframe::egui;

use crate::data::dataset_source::DatasetSource;
use crate::data::project_config::{ProjectConfig, ProjectDatasetConfig, ProjectRoi};
use crate::ui::roi_browser::RoiBrowseState;

#[derive(Debug, Clone)]
pub struct RoiEntry {
    pub dataset_name: String,
    pub display_name: String,
    pub key: String,
    pub roi_short: String,
    pub source_key: Option<String>,
    pub local_path: Option<PathBuf>,
    pub sample_name: Option<String>,
    pub roi: ProjectRoi,
}

#[derive(Debug, Clone)]
pub struct MasksConfig {
    pub layout: String,
    pub base_dir_full_res: Option<PathBuf>,
    pub base_dir_downsampled: Option<PathBuf>,
    pub masks_dir: Option<PathBuf>,
    pub masks_downsample_full_res: f32,
    pub masks_downsample_downsampled: f32,
}

#[derive(Debug, Clone)]
pub enum RoiSelectorAction {
    OpenRoi(ProjectRoi),
    LoadLabels,
    LoadMasks,
    SaveMasks,
}

pub struct RoiSelectorPanel {
    project: ProjectConfig,
    cfg_status: String,

    datasets: Vec<String>,
    selected_dataset: usize,

    roi_entries: Vec<RoiEntry>,
    selected_roi: usize,
    active_source_key: Option<String>,

    status: String,
    roi_browse: RoiBrowseState,
}

impl RoiSelectorPanel {
    pub fn new(current_source: &DatasetSource) -> Self {
        let mut panel = Self {
            project: ProjectConfig::default(),
            cfg_status: "No project loaded.".to_string(),
            datasets: Vec::new(),
            selected_dataset: 0,
            roi_entries: Vec::new(),
            selected_roi: 0,
            active_source_key: Some(current_source.source_key()),
            status: "Ready.".to_string(),
            roi_browse: RoiBrowseState::default(),
        };
        panel.rebuild();
        panel
    }

    pub fn set_project_config(&mut self, project: ProjectConfig, current_source: &DatasetSource) {
        self.project = project;
        self.active_source_key = Some(current_source.source_key());
        self.roi_browse.clear();
        self.rebuild();
        self.sync_to_dataset_source(current_source);
    }

    pub fn set_status(&mut self, status: impl Into<String>) {
        self.status = status.into();
    }

    pub fn sync_to_dataset_source(&mut self, source: &DatasetSource) {
        self.active_source_key = Some(source.source_key());
        if self.datasets.is_empty() {
            return;
        }

        // Prefer the dataset assigned to this ROI in the project list (fall back to project default).
        if let Some(roi) = self
            .project
            .rois
            .iter()
            .find(|r| roi_matches_source(r, source))
        {
            let ds = dataset_key_for_roi(&self.project, roi);
            if let Some(i) = self.datasets.iter().position(|n| n == &ds) {
                self.selected_dataset = i;
                self.rebuild_roi_entries();
            }
        }

        self.set_active_source_key(source);
    }

    pub fn set_active_source_key(&mut self, source: &DatasetSource) {
        let key = source.source_key();
        self.active_source_key = Some(key.clone());
        match source {
            DatasetSource::Local(active_path) => {
                if let Some(i) = self.roi_entries.iter().position(|e| {
                    e.local_path
                        .as_deref()
                        .is_some_and(|p| paths_equal(p, active_path))
                }) {
                    self.selected_roi = i;
                    return;
                }
            }
            _ => {}
        }
        if let Some(i) = self
            .roi_entries
            .iter()
            .position(|e| e.source_key.as_deref() == Some(&key))
        {
            self.selected_roi = i;
        }
    }

    pub fn masks_config_for_roi(&self, dataset_root: &Path) -> Option<MasksConfig> {
        let entry = self.roi_entry_for_path(dataset_root)?;
        let ds = self.project.datasets.get(&entry.dataset_name)?;
        Some(masks_config_from_dataset(ds))
    }

    pub fn roi_entry_for_path(&self, dataset_root: &Path) -> Option<RoiEntry> {
        self.roi_entries
            .iter()
            .find(|e| {
                e.local_path
                    .as_deref()
                    .is_some_and(|p| paths_equal(p, dataset_root))
            })
            .cloned()
    }

    pub fn tick(&mut self) {}

    pub fn has_multiple_rois(&self) -> bool {
        self.roi_entries.len() > 1
    }

    pub fn step_roi_action(&mut self, delta: i32) -> Option<RoiSelectorAction> {
        if self.roi_entries.is_empty() {
            self.selected_roi = 0;
            return None;
        }
        self.step_roi(delta);
        self.selected_roi_action()
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) -> Option<RoiSelectorAction> {
        ui.heading("ROI Selector");

        if self.datasets.is_empty() {
            ui.label(&self.cfg_status);
            ui.label("Add ROIs in the Project tab to enable ROI selection.");
            return None;
        }

        let mut dataset_changed = false;
        egui::ComboBox::from_label("Dataset")
            .selected_text(
                self.datasets
                    .get(self.selected_dataset)
                    .cloned()
                    .unwrap_or_else(|| "-".to_string()),
            )
            .show_ui(ui, |ui| {
                for (i, name) in self.datasets.iter().enumerate() {
                    dataset_changed |= ui
                        .selectable_value(&mut self.selected_dataset, i, name)
                        .changed();
                }
            });
        if dataset_changed {
            self.rebuild_roi_entries();
        }

        ui.add_space(4.0);
        ui.label("Browse");
        let dataset = self
            .datasets
            .get(self.selected_dataset)
            .cloned()
            .unwrap_or_default();
        let browse = crate::ui::roi_browser::ui(
            ui,
            "roi-selector-browser",
            &self.project.rois,
            Some(dataset.as_str()),
            &mut self.roi_browse,
        );
        ui.label(format!(
            "Showing {} / {} ROI(s).",
            browse.filtered_indices.len(),
            browse.total_count
        ));
        if browse.changed {
            self.rebuild_roi_entries();
        }

        ui.add_space(4.0);

        let roi_text = self
            .roi_entries
            .get(self.selected_roi)
            .map(|e| e.display_name.as_str())
            .unwrap_or("-");
        if self.roi_entries.is_empty() {
            ui.label("No ROIs match the current browse filters.");
        }
        let mut roi_changed = false;
        egui::ComboBox::from_label("ROI")
            .selected_text(roi_text)
            .show_ui(ui, |ui| {
                for (i, entry) in self.roi_entries.iter().enumerate() {
                    roi_changed |= ui
                        .selectable_value(&mut self.selected_roi, i, &entry.display_name)
                        .changed();
                }
            });

        ui.horizontal(|ui| {
            if ui.button("Prev ROI").clicked() {
                roi_changed = true;
                self.step_roi(-1);
            }
            if ui.button("Next ROI").clicked() {
                roi_changed = true;
                self.step_roi(1);
            }
        });

        if roi_changed {
            return self.selected_roi_action();
        }

        ui.separator();

        let mut action = None;
        ui.horizontal(|ui| {
            if ui.button("Load Labels").clicked() {
                action = Some(RoiSelectorAction::LoadLabels);
            }
            if ui.button("Load Masks").clicked() {
                action = Some(RoiSelectorAction::LoadMasks);
            }
            if ui.button("Save Masks").clicked() {
                action = Some(RoiSelectorAction::SaveMasks);
            }
        });

        ui.add_space(6.0);
        ui.label(format!("Status: {}", self.status));

        action
    }

    fn rebuild(&mut self) {
        self.rebuild_datasets();
        self.rebuild_roi_entries();
    }

    fn rebuild_datasets(&mut self) {
        let mut ds: BTreeMap<String, ()> = BTreeMap::new();
        for r in &self.project.rois {
            let name = dataset_key_for_roi(&self.project, r);
            if !name.trim().is_empty() {
                ds.insert(name, ());
            }
        }
        // Also include any configured datasets even if no ROI is assigned yet.
        for name in self.project.datasets.keys() {
            ds.insert(name.clone(), ());
        }
        self.datasets = ds.keys().cloned().collect();
        if self.datasets.is_empty() {
            self.cfg_status = "No datasets configured.".to_string();
            self.selected_dataset = 0;
            self.roi_entries.clear();
            self.selected_roi = 0;
            return;
        }
        self.cfg_status = "Ready.".to_string();

        if let Some(want) = self.project.default_dataset.as_deref() {
            if let Some(i) = self.datasets.iter().position(|n| n == want) {
                self.selected_dataset = i;
            }
        }
        self.selected_dataset = self
            .selected_dataset
            .min(self.datasets.len().saturating_sub(1));
    }

    fn rebuild_roi_entries(&mut self) {
        self.roi_entries.clear();
        self.selected_roi = 0;

        let dataset = self
            .datasets
            .get(self.selected_dataset)
            .cloned()
            .unwrap_or_default();

        let filtered_indices = crate::ui::roi_browser::filtered_roi_indices(
            &self.project.rois,
            Some(dataset.as_str()),
            &self.roi_browse.clauses,
        );
        for idx in filtered_indices {
            if let Some(r) = self.project.rois.get(idx) {
                self.roi_entries.push(roi_to_entry(r, &dataset));
            }
        }
        self.roi_entries
            .sort_by(|a, b| a.display_name.cmp(&b.display_name));

        if let Some(active) = self.active_source_key.clone() {
            if let Some(i) = self
                .roi_entries
                .iter()
                .position(|e| e.source_key.as_deref() == Some(&active))
            {
                self.selected_roi = i;
            }
        }
    }

    fn step_roi(&mut self, delta: i32) {
        if self.roi_entries.is_empty() {
            self.selected_roi = 0;
            return;
        }
        let n = self.roi_entries.len() as i32;
        let i = self.selected_roi as i32;
        let next = (i + delta).rem_euclid(n) as usize;
        self.selected_roi = next;
    }

    fn selected_roi_action(&mut self) -> Option<RoiSelectorAction> {
        let entry = self.roi_entries.get(self.selected_roi).cloned()?;
        if let Some(source) = entry.roi.dataset_source() {
            self.set_active_source_key(&source);
        } else {
            self.active_source_key = None;
        }
        Some(RoiSelectorAction::OpenRoi(entry.roi))
    }
}

fn roi_to_entry(r: &ProjectRoi, dataset_name: &str) -> RoiEntry {
    let display_name = r
        .display_name
        .clone()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| r.id.clone());
    let roi_short = r
        .local_path()
        .and_then(|path| path.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();
    let local_path = r.local_path().map(|p| p.to_path_buf());
    let source_key = r.source_key();
    RoiEntry {
        dataset_name: dataset_name.to_string(),
        display_name,
        key: r.id.clone(),
        roi_short,
        sample_name: None,
        source_key,
        local_path,
        roi: r.clone(),
    }
}

fn masks_config_from_dataset(ds: &ProjectDatasetConfig) -> MasksConfig {
    MasksConfig {
        layout: ds
            .layout
            .as_deref()
            .unwrap_or("flat_roi")
            .trim()
            .to_ascii_lowercase(),
        base_dir_full_res: ds
            .base_dir_full_res
            .as_deref()
            .map(expand_tilde)
            .map(PathBuf::from),
        base_dir_downsampled: ds
            .base_dir_downsampled
            .as_deref()
            .map(expand_tilde)
            .map(PathBuf::from),
        masks_dir: ds.masks_dir.as_deref().map(expand_tilde).map(PathBuf::from),
        masks_downsample_full_res: ds.masks_downsample_full_res.unwrap_or(1.0),
        masks_downsample_downsampled: ds.masks_downsample_downsampled.unwrap_or(1.0),
    }
}

fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest).to_string_lossy().to_string();
        }
    }
    path.to_string()
}

fn dataset_key_for_roi(project: &ProjectConfig, roi: &ProjectRoi) -> String {
    roi.dataset
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.to_string())
        .or_else(|| project.default_dataset.clone())
        .unwrap_or_else(|| "default".to_string())
}

fn roi_matches_source(roi: &ProjectRoi, source: &DatasetSource) -> bool {
    match (roi.dataset_source(), source) {
        (Some(DatasetSource::Local(p)), DatasetSource::Local(active)) => paths_equal(&p, active),
        (Some(s), _) => s == *source,
        (None, _) => false,
    }
}

fn paths_equal(a: &Path, b: &Path) -> bool {
    a.to_string_lossy() == b.to_string_lossy()
}
