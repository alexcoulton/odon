use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use serde_json::{Value, json};

use crate::control::{ControlError, ControlErrorKind};
use crate::data::dataset_source::DatasetSource;
use crate::data::project_config::ProjectRoi;

use super::ProjectModelSnapshot;

pub(super) fn validate_replacement_rois(rois: &[ProjectRoi]) -> Result<(), ControlError> {
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

pub(super) fn normalize_snapshot_shape(snapshot: &mut ProjectModelSnapshot) {
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

pub(super) fn view_response(index: usize, preset: &Value) -> Value {
    json!({
        "index": index,
        "name": preset.get("name").cloned().unwrap_or(Value::Null),
        "description": preset.get("description").cloned().unwrap_or_else(|| json!("")),
        "spec": preset.get("spec").cloned().unwrap_or_else(|| json!({})),
    })
}

pub(super) fn normalize_view_channel_alias(spec: &mut Value) {
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

pub(super) fn roi_from_params(
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

pub(super) fn normalize_roi_id(roi: &mut ProjectRoi) -> Result<(), ControlError> {
    roi.id = roi.id.trim().to_string();
    if roi.id.is_empty() {
        return Err(invalid("ROI id must not be empty"));
    }
    Ok(())
}

pub(super) fn required_source_key(roi: &ProjectRoi) -> Result<String, ControlError> {
    roi.source_key()
        .ok_or_else(|| invalid("ROI must have a dataset source"))
}

pub(super) fn replace_key(items: &mut [String], old: &str, new: &str) {
    for item in items {
        if item == old {
            *item = new.to_string();
        }
    }
}

pub(super) fn optional_string(value: &Value, name: &str) -> Result<Option<String>, ControlError> {
    match value {
        Value::Null => Ok(None),
        Value::String(value) => Ok(Some(value.clone())),
        _ => Err(invalid(format!("{name} must be a string or null"))),
    }
}

pub(super) fn required_string<'a>(params: &'a Value, name: &str) -> Result<&'a str, ControlError> {
    params
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("{name} is required")))
}

pub(super) fn string_array(params: &Value, name: &str) -> Result<Vec<String>, ControlError> {
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

pub(super) fn invalid(message: impl Into<String>) -> ControlError {
    ControlError::new(ControlErrorKind::InvalidParams, message)
}

pub(super) fn not_found(message: impl Into<String>) -> ControlError {
    ControlError::new(ControlErrorKind::ResourceNotFound, message)
}

pub(super) fn conflict(message: impl Into<String>) -> ControlError {
    ControlError::new(ControlErrorKind::Conflict, message)
}
