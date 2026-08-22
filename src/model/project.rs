use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use serde_json::{Value, json};

use crate::control::{ControlError, ControlErrorKind};
use crate::data::dataset_kind::{LocalDatasetKind, classify_local_dataset_path};
use crate::data::dataset_source::DatasetSource;
use crate::data::project_config::{ProjectConfig, ProjectRoi};

#[derive(Debug, Clone)]
pub struct ProjectModelSnapshot {
    /// Complete persisted project configuration. The explicit fields below are indexed copies
    /// used by actor commands; `ProjectModel` keeps both representations synchronized.
    pub config: ProjectConfig,
    /// Complete version-6 project state. This remains JSON in the renderer-independent crate so
    /// UI-specific view-state DTOs do not become actor dependencies.
    pub state: Value,
    /// Changes only when a complete project is replaced (open/create), allowing the renderer to
    /// distinguish a full materialization from an ordinary semantic projection.
    pub load_generation: u64,
    pub rois: Vec<ProjectRoi>,
    pub default_dataset: Option<String>,
    pub secondary_dataset: Option<String>,
    pub default_threshold_marker: Option<String>,
    pub mosaic_segmentation_search_roots: Vec<PathBuf>,
    pub dataset_keys: Vec<String>,
    pub selected_source_keys: Vec<String>,
    pub focused_source_key: Option<String>,
    pub saved_path: Option<PathBuf>,
    pub config_generation: u64,
    pub view_presets: Vec<Value>,
    pub view_count: usize,
    pub dirty: bool,
}

impl Default for ProjectModelSnapshot {
    fn default() -> Self {
        Self {
            config: ProjectConfig::default(),
            state: json!({}),
            load_generation: 0,
            rois: Vec::new(),
            default_dataset: None,
            secondary_dataset: None,
            default_threshold_marker: None,
            mosaic_segmentation_search_roots: Vec::new(),
            dataset_keys: Vec::new(),
            selected_source_keys: Vec::new(),
            focused_source_key: None,
            saved_path: None,
            config_generation: 0,
            view_presets: Vec::new(),
            view_count: 0,
            dirty: false,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ProjectModel {
    snapshot: ProjectModelSnapshot,
}

impl ProjectModel {
    pub(crate) fn activate_roi(&mut self, roi: &ProjectRoi) -> Result<(), ControlError> {
        let index = self.roi_index_by_id(&roi.id)?;
        let key = required_source_key(&self.snapshot.rois[index])?;
        self.snapshot.focused_source_key = Some(key.clone());
        self.snapshot.selected_source_keys = vec![key];
        self.mark_navigation_change();
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

    pub(crate) fn dispatch(&mut self, method: &str, params: &Value) -> Result<Value, ControlError> {
        match method {
            "project.rois.list" => Ok(self.rois_json()),
            "project.get" => Ok(self.project_json()),
            "project.create" => self.create(params),
            "project.update_metadata" => self.update_metadata(params),
            "project.rois.get" => self.get_roi(params),
            "project.rois.add" => self.add_roi(params),
            "project.rois.update" => self.update_roi(params),
            "project.rois.remove" => self.remove_roi(params),
            "project.rois.reorder" => self.reorder_rois(params),
            "project.rois.get_selection" => Ok(self.selection_json()),
            "project.rois.select" => self.select_rois(params),
            "project.rois.focus" => self.focus_roi(params),
            "project.rois.next" => self.step_roi(params, true),
            "project.rois.previous" => self.step_roi(params, false),
            "project.views.list" => Ok(self.views_json()),
            "project.views.get" => self.get_view(params),
            "project.views.create" => self.create_view(params),
            "project.views.rename" => self.rename_view(params),
            "project.views.delete" => self.delete_view(params),
            _ => Err(ControlError::new(
                ControlErrorKind::MethodNotFound,
                format!("unknown project-model method '{method}'"),
            )),
        }
    }

    fn project_json(&self) -> Value {
        json!({
            "path": self.snapshot.saved_path.as_ref().map(|path| path.to_string_lossy().to_string()),
            "config_generation": self.snapshot.config_generation,
            "roi_count": self.snapshot.rois.len(),
            "view_count": self.snapshot.view_count,
            "metadata": {
                "default_dataset": self.snapshot.default_dataset,
                "secondary_dataset": self.snapshot.secondary_dataset,
                "default_threshold_marker": self.snapshot.default_threshold_marker,
                "mosaic_segmentation_search_roots": self.snapshot.mosaic_segmentation_search_roots,
                "dataset_keys": self.snapshot.dataset_keys,
            },
        })
    }

    pub(crate) fn rois_json(&self) -> Value {
        let selected = self
            .snapshot
            .selected_source_keys
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        let rois = self
            .snapshot
            .rois
            .iter()
            .map(|roi| {
                let source_key = roi.source_key();
                json!({
                    "id": roi.id,
                    "display_name": roi.display_name,
                    "dataset": roi.dataset,
                    "source_key": source_key,
                    "source": roi.source_display(),
                    "segmentation_path": roi.segpath.as_ref().map(|path| path.to_string_lossy().to_string()),
                    "selected": source_key.as_deref().is_some_and(|key| selected.contains(key)),
                    "focused": source_key == self.snapshot.focused_source_key,
                })
            })
            .collect::<Vec<_>>();
        json!({
            "project_path": self.snapshot.saved_path.as_ref().map(|path| path.to_string_lossy().to_string()),
            "roi_count": rois.len(),
            "rois": rois,
        })
    }

    fn selection_json(&self) -> Value {
        let selected = self
            .snapshot
            .rois
            .iter()
            .filter(|roi| {
                roi.source_key().is_some_and(|key| {
                    self.snapshot
                        .selected_source_keys
                        .iter()
                        .any(|item| item == &key)
                })
            })
            .map(|roi| roi.id.clone())
            .collect::<Vec<_>>();
        let focused = self
            .snapshot
            .focused_source_key
            .as_deref()
            .and_then(|key| self.roi_by_source_key(key))
            .map(|roi| roi.id.clone());
        json!({"focused": focused, "selected": selected})
    }

    fn views_json(&self) -> Value {
        json!({
            "views": self.snapshot.view_presets.iter().enumerate().map(|(index, preset)| {
                view_response(index, preset)
            }).collect::<Vec<_>>()
        })
    }

    fn get_view(&self, params: &Value) -> Result<Value, ControlError> {
        let index = self.view_index(params)?;
        Ok(view_response(index, &self.snapshot.view_presets[index]))
    }

    fn create_view(&mut self, params: &Value) -> Result<Value, ControlError> {
        let name = required_string(params, "name")?.trim();
        let mut spec = params.get("spec").cloned().unwrap_or_else(|| json!({}));
        normalize_view_channel_alias(&mut spec);
        let preset = json!({"name":name,"description":"","spec":spec});
        let index = self
            .snapshot
            .view_presets
            .iter()
            .position(|existing| existing.get("name").and_then(Value::as_str) == Some(name));
        let index = if let Some(index) = index {
            self.snapshot.view_presets[index] = preset;
            index
        } else {
            self.snapshot.view_presets.push(preset);
            self.snapshot.view_presets.len() - 1
        };
        self.snapshot.view_count = self.snapshot.view_presets.len();
        self.mark_structural_change();
        Ok(view_response(index, &self.snapshot.view_presets[index]))
    }

    fn rename_view(&mut self, params: &Value) -> Result<Value, ControlError> {
        let index = self.view_index(params)?;
        let new_name = required_string(params, "new_name")?.trim();
        if self
            .snapshot
            .view_presets
            .iter()
            .enumerate()
            .any(|(candidate, preset)| {
                candidate != index && preset.get("name").and_then(Value::as_str) == Some(new_name)
            })
        {
            return Err(conflict(format!(
                "a view preset named '{new_name}' already exists"
            )));
        }
        let changed = self.snapshot.view_presets[index]
            .get("name")
            .and_then(Value::as_str)
            != Some(new_name);
        if changed {
            self.snapshot.view_presets[index]
                .as_object_mut()
                .ok_or_else(|| invalid("saved view preset must be an object"))?
                .insert("name".to_string(), Value::String(new_name.to_string()));
            self.mark_structural_change();
        }
        Ok(view_response(index, &self.snapshot.view_presets[index]))
    }

    fn delete_view(&mut self, params: &Value) -> Result<Value, ControlError> {
        let index = self.view_index(params)?;
        let removed = self.snapshot.view_presets.remove(index);
        self.snapshot.view_count = self.snapshot.view_presets.len();
        self.mark_structural_change();
        Ok(json!({
            "deleted": true,
            "index": index,
            "name": removed.get("name").cloned().unwrap_or(Value::Null),
        }))
    }

    fn view_index(&self, params: &Value) -> Result<usize, ControlError> {
        if let Some(index) = params.get("index").and_then(Value::as_u64) {
            let index = usize::try_from(index).unwrap_or(usize::MAX);
            return (index < self.snapshot.view_presets.len())
                .then_some(index)
                .ok_or_else(|| not_found(format!("view preset index {index} is out of range")));
        }
        let name = required_string(params, "name")?.trim();
        self.snapshot
            .view_presets
            .iter()
            .position(|preset| preset.get("name").and_then(Value::as_str) == Some(name))
            .ok_or_else(|| not_found(format!("view preset '{name}' was not found")))
    }

    fn create(&mut self, params: &Value) -> Result<Value, ControlError> {
        let default_dataset = params
            .get("default_dataset")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string);
        self.snapshot = ProjectModelSnapshot {
            default_dataset,
            load_generation: self.snapshot.load_generation.wrapping_add(1),
            ..ProjectModelSnapshot::default()
        };
        self.sync_persisted_project();
        Ok(json!({"created": true, "project": self.project_json()}))
    }

    fn update_metadata(&mut self, params: &Value) -> Result<Value, ControlError> {
        if let Some(value) = params.get("default_dataset") {
            self.snapshot.default_dataset = optional_string(value, "default_dataset")?;
        }
        if let Some(value) = params.get("secondary_dataset") {
            self.snapshot.secondary_dataset = optional_string(value, "secondary_dataset")?;
        }
        if let Some(value) = params.get("default_threshold_marker") {
            self.snapshot.default_threshold_marker =
                optional_string(value, "default_threshold_marker")?;
        }
        if let Some(values) = params.get("mosaic_segmentation_search_roots") {
            let values = values.as_array().ok_or_else(|| {
                invalid("mosaic_segmentation_search_roots must be an array of paths")
            })?;
            self.snapshot.mosaic_segmentation_search_roots = values
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .map(PathBuf::from)
                        .ok_or_else(|| invalid("every segmentation search root must be a string"))
                })
                .collect::<Result<Vec<_>, _>>()?;
        }
        self.mark_structural_change();
        Ok(json!({"updated": true, "project": self.project_json()}))
    }

    fn get_roi(&self, params: &Value) -> Result<Value, ControlError> {
        let id = required_string(params, "id")?;
        let index = self.roi_index_by_id(id)?;
        Ok(json!({"index": index, "roi": self.snapshot.rois[index]}))
    }

    fn add_roi(&mut self, params: &Value) -> Result<Value, ControlError> {
        let mut roi = roi_from_params(params, None)?;
        normalize_roi_id(&mut roi)?;
        if self
            .snapshot
            .rois
            .iter()
            .any(|existing| existing.id == roi.id)
        {
            return Err(conflict(format!("ROI '{}' already exists", roi.id)));
        }
        let source_key = required_source_key(&roi)?;
        if self
            .snapshot
            .rois
            .iter()
            .any(|existing| existing.source_key().as_deref() == Some(source_key.as_str()))
        {
            return Err(conflict(
                "ROI dataset source is already present in the project",
            ));
        }
        self.snapshot.rois.push(roi);
        let index = self.snapshot.rois.len() - 1;
        if self.snapshot.focused_source_key.is_none() {
            self.snapshot.focused_source_key = Some(source_key.clone());
        }
        if !self.snapshot.selected_source_keys.contains(&source_key) {
            self.snapshot.selected_source_keys.push(source_key);
        }
        self.mark_structural_change();
        let id = self.snapshot.rois[index].id.clone();
        self.get_roi(&json!({"id": id}))
    }

    fn update_roi(&mut self, params: &Value) -> Result<Value, ControlError> {
        let target_id = params
            .get("target_id")
            .or_else(|| params.get("id"))
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("target_id is required"))?;
        let index = self.roi_index_by_id(target_id)?;
        let existing = self.snapshot.rois[index].clone();
        let patch = params.get("changes").unwrap_or(params);
        let mut roi = roi_from_params(patch, Some(&existing))?;
        normalize_roi_id(&mut roi)?;
        if self
            .snapshot
            .rois
            .iter()
            .enumerate()
            .any(|(candidate, existing)| candidate != index && existing.id == roi.id)
        {
            return Err(conflict(format!("ROI '{}' already exists", roi.id)));
        }
        let new_key = required_source_key(&roi)?;
        if self
            .snapshot
            .rois
            .iter()
            .enumerate()
            .any(|(candidate, existing)| {
                candidate != index && existing.source_key().as_deref() == Some(new_key.as_str())
            })
        {
            return Err(conflict(
                "ROI dataset source is already present in the project",
            ));
        }
        let old_key = existing.source_key();
        self.snapshot.rois[index] = roi;
        if let Some(old_key) = old_key.filter(|old_key| old_key != &new_key) {
            replace_key(&mut self.snapshot.selected_source_keys, &old_key, &new_key);
            if self.snapshot.focused_source_key.as_deref() == Some(old_key.as_str()) {
                self.snapshot.focused_source_key = Some(new_key);
            }
        }
        self.mark_structural_change();
        let id = self.snapshot.rois[index].id.clone();
        self.get_roi(&json!({"id": id}))
    }

    fn remove_roi(&mut self, params: &Value) -> Result<Value, ControlError> {
        let id = required_string(params, "id")?;
        let index = self.roi_index_by_id(id)?;
        let removed = self.snapshot.rois.remove(index);
        if let Some(key) = removed.source_key() {
            self.snapshot
                .selected_source_keys
                .retain(|item| item != &key);
            if self.snapshot.focused_source_key.as_deref() == Some(key.as_str()) {
                self.snapshot.focused_source_key = None;
            }
        }
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
        Ok(json!({"removed": true, "roi": removed}))
    }

    fn reorder_rois(&mut self, params: &Value) -> Result<Value, ControlError> {
        let ids = string_array(params, "ids")?;
        if ids.len() != self.snapshot.rois.len() {
            return Err(invalid(
                "ROI order must contain every project ROI exactly once",
            ));
        }
        let unique = ids.iter().collect::<HashSet<_>>();
        if unique.len() != ids.len() {
            return Err(invalid("ROI order must not contain duplicate IDs"));
        }
        let mut next = Vec::with_capacity(ids.len());
        for id in &ids {
            let index = self.roi_index_by_id(id)?;
            next.push(self.snapshot.rois[index].clone());
        }
        let changed = next.iter().map(|roi| roi.id.as_str()).ne(self
            .snapshot
            .rois
            .iter()
            .map(|roi| roi.id.as_str()));
        if changed {
            self.snapshot.rois = next;
            self.mark_structural_change();
        }
        Ok(self.rois_json())
    }

    fn select_rois(&mut self, params: &Value) -> Result<Value, ControlError> {
        let ids = string_array(params, "ids")?;
        let keys = ids
            .iter()
            .map(|id| {
                let index = self.roi_index_by_id(id)?;
                required_source_key(&self.snapshot.rois[index])
            })
            .collect::<Result<Vec<_>, _>>()?;
        match params
            .get("mode")
            .and_then(Value::as_str)
            .unwrap_or("replace")
        {
            "replace" => self.snapshot.selected_source_keys.clone_from(&keys),
            "add" => {
                for key in &keys {
                    if !self.snapshot.selected_source_keys.contains(key) {
                        self.snapshot.selected_source_keys.push(key.clone());
                    }
                }
            }
            "remove" => self
                .snapshot
                .selected_source_keys
                .retain(|item| !keys.contains(item)),
            "toggle" => {
                for key in &keys {
                    if let Some(index) = self
                        .snapshot
                        .selected_source_keys
                        .iter()
                        .position(|item| item == key)
                    {
                        self.snapshot.selected_source_keys.remove(index);
                    } else {
                        self.snapshot.selected_source_keys.push(key.clone());
                    }
                }
            }
            _ => {
                return Err(invalid(
                    "selection mode must be replace, add, remove, or toggle",
                ));
            }
        }
        if let Some(key) = keys.last() {
            self.snapshot.focused_source_key = Some(key.clone());
        }
        self.mark_navigation_change();
        Ok(self.selection_json())
    }

    fn focus_roi(&mut self, params: &Value) -> Result<Value, ControlError> {
        let id = required_string(params, "id")?;
        let index = self.roi_index_by_id(id)?;
        self.snapshot.focused_source_key = Some(required_source_key(&self.snapshot.rois[index])?);
        self.mark_navigation_change();
        Ok(self.selection_json())
    }

    fn step_roi(&mut self, params: &Value, forward: bool) -> Result<Value, ControlError> {
        if self.snapshot.rois.is_empty() {
            return Err(not_found("project has no ROIs"));
        }
        let raw_step = params.get("step").and_then(Value::as_u64).unwrap_or(1);
        let step = i64::try_from(raw_step).unwrap_or(i64::MAX);
        let step = if forward { step } else { -step };
        let wrap = params.get("wrap").and_then(Value::as_bool).unwrap_or(true);
        let current = self
            .snapshot
            .focused_source_key
            .as_deref()
            .and_then(|key| {
                self.snapshot
                    .rois
                    .iter()
                    .position(|roi| roi.source_key().as_deref() == Some(key))
            })
            .unwrap_or_default();
        let len = self.snapshot.rois.len() as i64;
        let candidate = current as i64 + step;
        let index = if wrap {
            candidate.rem_euclid(len) as usize
        } else {
            candidate.clamp(0, len - 1) as usize
        };
        self.snapshot.focused_source_key = Some(required_source_key(&self.snapshot.rois[index])?);
        self.mark_navigation_change();
        Ok(self.selection_json())
    }

    fn roi_index_by_id(&self, id: &str) -> Result<usize, ControlError> {
        let id = id.trim();
        if id.is_empty() {
            return Err(invalid("ROI id must not be empty"));
        }
        let matches = self
            .snapshot
            .rois
            .iter()
            .enumerate()
            .filter(|(_, roi)| roi.id == id)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [index] => Ok(*index),
            [] => Err(not_found(format!("ROI '{id}' was not found"))),
            _ => Err(conflict(format!("ROI id '{id}' is ambiguous"))),
        }
    }

    fn roi_by_source_key(&self, key: &str) -> Option<&ProjectRoi> {
        self.snapshot
            .rois
            .iter()
            .find(|roi| roi.source_key().as_deref() == Some(key))
    }

    fn normalize_selection(&mut self) {
        let valid = self
            .snapshot
            .rois
            .iter()
            .filter_map(ProjectRoi::source_key)
            .collect::<HashSet<_>>();
        self.snapshot
            .selected_source_keys
            .retain(|key| valid.contains(key));
        self.snapshot.selected_source_keys.sort();
        self.snapshot.selected_source_keys.dedup();
        if self
            .snapshot
            .focused_source_key
            .as_ref()
            .is_some_and(|key| !valid.contains(key))
        {
            self.snapshot.focused_source_key = None;
        }
    }

    fn mark_structural_change(&mut self) {
        self.snapshot.config_generation = self.snapshot.config_generation.wrapping_add(1);
        self.snapshot.dirty = true;
        self.sync_persisted_project();
    }

    fn mark_navigation_change(&mut self) {
        self.snapshot.config_generation = self.snapshot.config_generation.wrapping_add(1);
        self.sync_persisted_project();
    }

    fn sync_persisted_project(&mut self) {
        self.snapshot.config.rois.clone_from(&self.snapshot.rois);
        self.snapshot
            .config
            .default_dataset
            .clone_from(&self.snapshot.default_dataset);
        self.snapshot
            .config
            .secondary_dataset
            .clone_from(&self.snapshot.secondary_dataset);
        self.snapshot
            .config
            .default_threshold_marker
            .clone_from(&self.snapshot.default_threshold_marker);
        self.snapshot
            .config
            .mosaic_segmentation_search_roots
            .clone_from(&self.snapshot.mosaic_segmentation_search_roots);
        let state = self
            .snapshot
            .state
            .as_object_mut()
            .expect("project state is normalized to an object");
        let browser = state
            .entry("browser")
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .expect("project browser state is normalized to an object");
        browser.insert(
            "focused".to_string(),
            self.snapshot
                .focused_source_key
                .clone()
                .map(Value::String)
                .unwrap_or(Value::Null),
        );
        browser.insert(
            "selected".to_string(),
            json!(self.snapshot.selected_source_keys),
        );
        state.insert(
            "view_presets".to_string(),
            Value::Array(self.snapshot.view_presets.clone()),
        );
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

fn validate_replacement_rois(rois: &[ProjectRoi]) -> Result<(), ControlError> {
    if rois.is_empty() {
        return Err(invalid("samplesheet contained no usable ROIs"));
    }
    let mut ids = HashSet::new();
    let mut sources = HashSet::new();
    for roi in rois {
        let id = roi.id.trim();
        if id.is_empty() {
            return Err(invalid("samplesheet ROI id must not be empty"));
        }
        if !ids.insert(id) {
            return Err(invalid(format!("duplicate samplesheet ROI id '{id}'")));
        }
        let source = required_source_key(roi)?;
        if !sources.insert(source) {
            return Err(invalid("samplesheet contains a duplicate dataset source"));
        }
    }
    Ok(())
}

fn normalize_snapshot_shape(snapshot: &mut ProjectModelSnapshot) {
    if !snapshot.state.is_object() {
        snapshot.state = json!({});
    }
    let state = snapshot
        .state
        .as_object_mut()
        .expect("project state was normalized to an object");
    if !state.get("browser").is_some_and(Value::is_object) {
        state.insert("browser".to_string(), json!({}));
    }
    if snapshot.view_presets.is_empty()
        && let Some(view_presets) = state.get("view_presets").and_then(Value::as_array)
    {
        snapshot.view_presets.clone_from(view_presets);
    }
    snapshot.view_count = snapshot.view_presets.len();
}

fn view_response(index: usize, preset: &Value) -> Value {
    json!({
        "index": index,
        "name": preset.get("name").cloned().unwrap_or(Value::Null),
        "description": preset.get("description").cloned().unwrap_or_else(|| json!("")),
        "spec": preset.get("spec").cloned().unwrap_or_else(|| json!({})),
    })
}

fn normalize_view_channel_alias(spec: &mut Value) {
    let Some(label) = spec
        .get("channel_ref")
        .and_then(Value::as_object)
        .and_then(|active| active.get("label"))
        .and_then(Value::as_str)
        .map(str::to_string)
    else {
        return;
    };
    let alias = spec
        .get("visible_channel_refs")
        .and_then(Value::as_array)
        .and_then(|visible| {
            visible
                .iter()
                .find(|entry| entry.get("label").and_then(Value::as_str) == Some(label.as_str()))
        })
        .and_then(|entry| entry.get("alias"))
        .cloned();
    if let (Some(alias), Some(active)) = (
        alias,
        spec.get_mut("channel_ref").and_then(Value::as_object_mut),
    ) {
        active.insert("alias".to_string(), alias);
    }
}

fn roi_from_params(
    params: &Value,
    existing: Option<&ProjectRoi>,
) -> Result<ProjectRoi, ControlError> {
    if let Some(replacement) = params.get("replacement") {
        return serde_json::from_value(replacement.clone())
            .map_err(|error| invalid(format!("invalid replacement ROI: {error}")));
    }
    let mut roi = existing.cloned().unwrap_or_default();
    if let Some(id) = params.get("id").and_then(Value::as_str) {
        roi.id = id.to_string();
    }
    if let Some(value) = params.get("display_name") {
        roi.display_name = optional_string(value, "display_name")?;
    }
    if let Some(value) = params.get("dataset") {
        roi.dataset = optional_string(value, "dataset")?;
    }
    if let Some(value) = params.get("source") {
        let source = serde_json::from_value::<DatasetSource>(value.clone())
            .map_err(|error| invalid(format!("invalid dataset source: {error}")))?;
        roi.set_dataset_source(source);
    }
    if let Some(path) = params.get("path").and_then(Value::as_str) {
        roi.set_dataset_source(DatasetSource::Local(PathBuf::from(path)));
    }
    if let Some(value) = params.get("segmentation_path") {
        roi.segpath = optional_string(value, "segmentation_path")?.map(PathBuf::from);
    }
    if let Some(metadata) = params.get("metadata") {
        let metadata = metadata
            .as_object()
            .ok_or_else(|| invalid("metadata must be an object of string values"))?;
        roi.meta = metadata
            .iter()
            .map(|(key, value)| {
                value
                    .as_str()
                    .map(|value| (key.clone(), value.to_string()))
                    .ok_or_else(|| invalid(format!("metadata value '{key}' must be a string")))
            })
            .collect::<Result<HashMap<_, _>, _>>()?;
    }
    Ok(roi)
}

fn normalize_roi_id(roi: &mut ProjectRoi) -> Result<(), ControlError> {
    roi.id = roi.id.trim().to_string();
    if roi.id.is_empty() {
        return Err(invalid("ROI id must not be empty"));
    }
    Ok(())
}

fn required_source_key(roi: &ProjectRoi) -> Result<String, ControlError> {
    roi.source_key()
        .ok_or_else(|| invalid("ROI must have a dataset source"))
}

fn replace_key(items: &mut [String], old: &str, new: &str) {
    for item in items {
        if item == old {
            *item = new.to_string();
        }
    }
}

fn optional_string(value: &Value, name: &str) -> Result<Option<String>, ControlError> {
    match value {
        Value::Null => Ok(None),
        Value::String(value) => Ok(Some(value.clone())),
        _ => Err(invalid(format!("{name} must be a string or null"))),
    }
}

fn required_string<'a>(params: &'a Value, name: &str) -> Result<&'a str, ControlError> {
    params
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("{name} is required")))
}

fn string_array(params: &Value, name: &str) -> Result<Vec<String>, ControlError> {
    params
        .get(name)
        .and_then(Value::as_array)
        .ok_or_else(|| invalid(format!("{name} is required")))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| invalid(format!("every {name} entry must be a string")))
        })
        .collect()
}

fn invalid(message: impl Into<String>) -> ControlError {
    ControlError::new(ControlErrorKind::InvalidParams, message)
}

fn not_found(message: impl Into<String>) -> ControlError {
    ControlError::new(ControlErrorKind::ResourceNotFound, message)
}

fn conflict(message: impl Into<String>) -> ControlError {
    ControlError::new(ControlErrorKind::Conflict, message)
}
