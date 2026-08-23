use std::collections::HashSet;
use std::path::PathBuf;

use serde_json::{Value, json};

use crate::control::{ControlError, ControlErrorKind};
use crate::data::project_config::{ProjectConfig, ProjectRoi};

use super::validation::*;
use super::*;

impl ProjectModel {
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
        let config = params
            .get("config")
            .map(|value| {
                serde_json::from_value::<ProjectConfig>(value.clone())
                    .map_err(|error| invalid(format!("invalid project config: {error}")))
            })
            .transpose()?;
        let default_dataset = params
            .get("default_dataset")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string);
        let load_generation = self.snapshot.load_generation.wrapping_add(1);
        self.snapshot = if let Some(config) = config {
            ProjectModelSnapshot {
                rois: config.rois.clone(),
                default_dataset: config.default_dataset.clone(),
                secondary_dataset: config.secondary_dataset.clone(),
                default_threshold_marker: config.default_threshold_marker.clone(),
                mosaic_segmentation_search_roots: config.mosaic_segmentation_search_roots.clone(),
                dataset_keys: config.datasets.keys().cloned().collect(),
                config,
                load_generation,
                ..ProjectModelSnapshot::default()
            }
        } else {
            ProjectModelSnapshot {
                default_dataset,
                load_generation,
                ..ProjectModelSnapshot::default()
            }
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
            "replace" => {
                self.snapshot.selected_source_keys.clone_from(&keys);
                if keys.is_empty() {
                    self.snapshot.focused_source_key = None;
                }
            }
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

    pub(super) fn roi_index_by_id(&self, id: &str) -> Result<usize, ControlError> {
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

    pub(super) fn normalize_selection(&mut self) {
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

    pub(super) fn mark_structural_change(&mut self) {
        self.snapshot.config_generation = self.snapshot.config_generation.wrapping_add(1);
        self.snapshot.dirty = true;
        self.sync_persisted_project();
    }

    pub(super) fn mark_navigation_change(&mut self) {
        self.snapshot.config_generation = self.snapshot.config_generation.wrapping_add(1);
        self.sync_persisted_project();
    }

    pub(super) fn sync_persisted_project(&mut self) {
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
