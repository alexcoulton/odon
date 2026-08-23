use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use serde_json::{Value, json};

use crate::control::ControlError;
use crate::data::dataset_kind::{LocalDatasetKind, classify_local_dataset_path};
use crate::data::dataset_source::DatasetSource;
use crate::data::project_config::{ProjectConfig, ProjectRoi};

use super::validation::*;
use super::*;

impl ProjectModel {
    pub(crate) fn roi_view_state_json(&self, source_key: &str) -> Option<&Value> {
        self.snapshot
            .state
            .get("roi_views")
            .and_then(Value::as_object)
            .and_then(|views| views.get(source_key))
    }

    pub(crate) fn activate_roi(&mut self, roi: &ProjectRoi) -> Result<(), ControlError> {
        let index = self.roi_index_by_id(&roi.id)?;
        let key = required_source_key(&self.snapshot.rois[index])?;
        self.snapshot.focused_source_key = Some(key.clone());
        self.snapshot.selected_source_keys = vec![key];
        self.mark_navigation_change();
        Ok(())
    }

    pub(crate) fn set_roi_view_state_json(
        &mut self,
        source_key: &str,
        view: Value,
    ) -> Result<(), ControlError> {
        let state = self
            .snapshot
            .state
            .as_object_mut()
            .ok_or_else(|| invalid("project state must be an object"))?;
        let views = state
            .entry("roi_views")
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .ok_or_else(|| invalid("project ROI view state must be an object"))?;
        if views.get(source_key) == Some(&view) {
            return Ok(());
        }
        views.insert(source_key.to_string(), view);
        self.mark_structural_change();
        Ok(())
    }

    pub(crate) fn replace(&mut self, mut snapshot: ProjectModelSnapshot) {
        normalize_snapshot_shape(&mut snapshot);
        self.snapshot = snapshot;
        self.normalize_selection();
        self.sync_persisted_project();
    }

    pub(crate) fn snapshot(&self) -> ProjectModelSnapshot {
        self.snapshot.clone()
    }

    pub(crate) fn mask_layer_count_for_source(&self, source: &DatasetSource) -> Option<usize> {
        let key = source.source_key();
        self.snapshot
            .rois
            .iter()
            .find(|roi| roi.source_key().as_deref() == Some(key.as_str()))
            .map(|roi| roi.mask_layers.len())
    }

    pub(crate) fn sync_mask_layers_for_source(
        &mut self,
        source: DatasetSource,
        layers: Vec<crate::data::project_config::ProjectMaskLayer>,
    ) -> Result<bool, ControlError> {
        let key = source.source_key();
        if let Some(roi) = self
            .snapshot
            .rois
            .iter_mut()
            .find(|roi| roi.source_key().as_deref() == Some(key.as_str()))
        {
            if roi.mask_layers == layers {
                return Ok(false);
            }
            roi.mask_layers = layers;
        } else {
            let display_name = source.display_name();
            let mut id = display_name.clone();
            let mut suffix = 2_u64;
            while self.snapshot.rois.iter().any(|roi| roi.id == id) {
                id = format!("{display_name} {suffix}");
                suffix = suffix.saturating_add(1);
            }
            let mut roi = ProjectRoi {
                id,
                display_name: Some(display_name),
                mask_layers: layers,
                ..ProjectRoi::default()
            };
            roi.set_dataset_source(source);
            normalize_roi_id(&mut roi)?;
            self.snapshot.rois.push(roi);
        }
        self.mark_structural_change();
        Ok(true)
    }

    pub(crate) fn install_loaded(
        &mut self,
        path: PathBuf,
        mut config: ProjectConfig,
        mut state: Value,
    ) -> Result<(), ControlError> {
        if !state.is_object() {
            state = json!({});
        }
        let project_dir = path.parent();
        let default_dataset = config
            .default_dataset
            .clone()
            .unwrap_or_else(|| "default".to_string());
        let mut seen = HashSet::new();
        let mut source_key_replacements = HashMap::new();
        let mut rois = Vec::new();
        for mut roi in std::mem::take(&mut config.rois) {
            let Some(source) = roi.dataset_source() else {
                continue;
            };
            let old_key = source.source_key();
            match source {
                DatasetSource::Local(source_path) => {
                    let resolved = if source_path.is_absolute() {
                        source_path
                    } else {
                        project_dir.map_or(source_path.clone(), |dir| dir.join(&source_path))
                    };
                    let resolved = resolved.canonicalize().unwrap_or(resolved);
                    if let Some(segmentation) = roi.segpath.take() {
                        let segmentation = if segmentation.is_absolute() {
                            segmentation
                        } else {
                            project_dir.map_or(segmentation.clone(), |dir| dir.join(&segmentation))
                        };
                        roi.segpath = Some(segmentation.canonicalize().unwrap_or(segmentation));
                    }
                    roi.set_dataset_source(DatasetSource::Local(resolved));
                    if matches!(
                        classify_local_dataset_path(
                            roi.local_path()
                                .expect("local ROI retains its resolved path")
                        ),
                        Some(LocalDatasetKind::OmeZarr)
                    ) && roi
                        .dataset
                        .as_deref()
                        .is_none_or(|dataset| dataset.trim().is_empty())
                    {
                        roi.dataset = Some(default_dataset.clone());
                    }
                }
                source => roi.set_dataset_source(source),
            }
            let new_key = required_source_key(&roi)?;
            if !seen.insert(new_key.clone()) {
                continue;
            }
            source_key_replacements.insert(old_key, new_key);
            if roi.display_name.is_none() {
                roi.display_name = Some(
                    roi.dataset_source()
                        .expect("cleaned ROI retains a source")
                        .display_name(),
                );
            }
            normalize_roi_id(&mut roi)?;
            rois.push(roi);
        }
        let roi_views = state
            .as_object_mut()
            .expect("project state was normalized")
            .entry("roi_views")
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .ok_or_else(|| invalid("project ROI view state must be an object"))?;
        for roi in &mut rois {
            if roi.channel_order.is_empty() {
                continue;
            }
            let key = required_source_key(roi)?;
            let view = roi_views
                .entry(key)
                .or_insert_with(|| json!({}))
                .as_object_mut()
                .ok_or_else(|| invalid("project ROI view entry must be an object"))?;
            let has_order = view
                .get("channel_order")
                .and_then(Value::as_array)
                .is_some_and(|order| !order.is_empty());
            if !has_order {
                view.insert("channel_order".to_string(), json!(roi.channel_order));
            }
            roi.channel_order.clear();
        }
        config.rois.clone_from(&rois);
        let browser = state
            .as_object_mut()
            .expect("project state was normalized")
            .entry("browser")
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .ok_or_else(|| invalid("project browser state must be an object"))?;
        let remap = |key: &str| {
            source_key_replacements
                .get(key)
                .cloned()
                .unwrap_or_else(|| key.to_string())
        };
        let focused_source_key = browser.get("focused").and_then(Value::as_str).map(&remap);
        let selected_source_keys = browser
            .get("selected")
            .and_then(Value::as_array)
            .map(|selected| {
                selected
                    .iter()
                    .filter_map(Value::as_str)
                    .map(&remap)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let next_load_generation = self.snapshot.load_generation.wrapping_add(1).max(1);
        let next_config_generation = self.snapshot.config_generation.wrapping_add(1).max(1);
        let default_dataset = config.default_dataset.clone();
        let secondary_dataset = config.secondary_dataset.clone();
        let default_threshold_marker = config.default_threshold_marker.clone();
        let mosaic_segmentation_search_roots = config.mosaic_segmentation_search_roots.clone();
        let dataset_keys = config.datasets.keys().cloned().collect();
        let view_presets = state
            .get("view_presets")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let view_count = view_presets.len();
        self.snapshot = ProjectModelSnapshot {
            config,
            state,
            load_generation: next_load_generation,
            rois,
            default_dataset,
            secondary_dataset,
            default_threshold_marker,
            mosaic_segmentation_search_roots,
            dataset_keys,
            selected_source_keys,
            focused_source_key,
            saved_path: Some(path),
            config_generation: next_config_generation,
            view_presets,
            view_count,
            dirty: false,
            ..ProjectModelSnapshot::default()
        };
        self.normalize_selection();
        if self.snapshot.focused_source_key.is_none() {
            self.snapshot.focused_source_key =
                self.snapshot.rois.first().and_then(ProjectRoi::source_key);
        }
        if self.snapshot.selected_source_keys.is_empty()
            && let Some(key) = self.snapshot.focused_source_key.clone()
        {
            self.snapshot.selected_source_keys.push(key);
        }
        self.sync_persisted_project();
        Ok(())
    }

    pub(crate) fn persistence_payload(&self) -> Result<(Value, u64), ControlError> {
        Ok((
            json!({
                "version": 6,
                "config": self.snapshot.config,
                "state": self.snapshot.state,
            }),
            self.snapshot.config_generation,
        ))
    }

    pub(crate) fn mark_saved(&mut self, path: PathBuf, saved_config_generation: u64) {
        self.snapshot.saved_path = Some(path);
        if self.snapshot.config_generation == saved_config_generation {
            self.snapshot.dirty = false;
        }
    }

    pub(crate) fn update_manifest(&mut self, resources: Vec<Value>, layers: Vec<Value>) -> bool {
        if self.snapshot.config.control_resources == resources
            && self.snapshot.config.control_layers == layers
        {
            return false;
        }
        self.snapshot.config.control_resources = resources;
        self.snapshot.config.control_layers = layers;
        self.snapshot.config_generation = self.snapshot.config_generation.wrapping_add(1);
        self.snapshot.dirty = true;
        true
    }

    pub(crate) fn replace_rois_from_samplesheet(
        &mut self,
        rois: Vec<ProjectRoi>,
    ) -> Result<Value, ControlError> {
        validate_replacement_rois(&rois)?;
        self.snapshot.rois = rois;
        self.snapshot.selected_source_keys.clear();
        self.snapshot.focused_source_key =
            self.snapshot.rois.first().and_then(ProjectRoi::source_key);
        if let Some(key) = self.snapshot.focused_source_key.clone() {
            self.snapshot.selected_source_keys.push(key);
        }
        self.mark_structural_change();
        Ok(self.rois_json())
    }

    pub(crate) fn add_discovered_roots(
        &mut self,
        roots: Vec<PathBuf>,
    ) -> Result<(usize, Value), ControlError> {
        let existing_sources = self
            .snapshot
            .rois
            .iter()
            .filter_map(ProjectRoi::source_key)
            .collect::<HashSet<_>>();
        let mut source_keys = existing_sources;
        let mut ids = self
            .snapshot
            .rois
            .iter()
            .map(|roi| roi.id.clone())
            .collect::<HashSet<_>>();
        let default_dataset = self
            .snapshot
            .default_dataset
            .clone()
            .unwrap_or_else(|| "default".to_string());
        let before = self.snapshot.rois.len();
        for root in roots {
            let root = root.canonicalize().unwrap_or(root);
            let source = DatasetSource::Local(root.clone());
            if !source_keys.insert(source.source_key()) {
                continue;
            }
            let base = root
                .file_name()
                .and_then(|name| name.to_str())
                .filter(|name| !name.trim().is_empty())
                .unwrap_or("ROI")
                .to_string();
            let mut id = base.clone();
            let mut suffix = 2usize;
            while !ids.insert(id.clone()) {
                id = format!("{base}-{suffix}");
                suffix += 1;
            }
            let mut roi = ProjectRoi {
                id,
                display_name: Some(base),
                dataset: Some(default_dataset.clone()),
                ..ProjectRoi::default()
            };
            roi.set_dataset_source(source);
            self.snapshot.rois.push(roi);
        }
        let added = self.snapshot.rois.len().saturating_sub(before);
        if added > 0 {
            if self.snapshot.focused_source_key.is_none() {
                self.snapshot.focused_source_key =
                    self.snapshot.rois.first().and_then(ProjectRoi::source_key);
            }
            if self.snapshot.selected_source_keys.is_empty()
                && let Some(key) = self.snapshot.focused_source_key.clone()
            {
                self.snapshot.selected_source_keys.push(key);
            }
            self.mark_structural_change();
        }
        Ok((added, self.rois_json()))
    }
}

pub(crate) fn normalized_loaded_project_snapshot(
    path: PathBuf,
    config: ProjectConfig,
    state: Value,
) -> Result<ProjectModelSnapshot, ControlError> {
    let mut project = ProjectModel::default();
    project.install_loaded(path, config, state)?;
    Ok(project.snapshot())
}
