use std::path::PathBuf;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::control::{ControlError, ControlErrorKind};
use crate::data::annotations::{
    AnnotationColumnInfo, AnnotationDataset, AnnotationValueMode, ProjectAnnotationLayerState,
    default_category_style_states,
};

#[derive(Debug, Clone)]
pub struct ControlAnnotationResource {
    pub dataset: Arc<AnnotationDataset>,
}

#[derive(Debug, Clone)]
pub struct ControlAnnotationLayerProjection {
    pub state: ProjectAnnotationLayerState,
    pub generation: u64,
    pub resource_generation: u64,
    pub resource: Option<Arc<ControlAnnotationResource>>,
    pub schema: Arc<Vec<AnnotationColumnInfo>>,
    pub pending: bool,
    pub status: String,
}

#[derive(Debug, Clone)]
pub(crate) struct AnnotationLoadSpec {
    pub document_generation: u64,
    pub layer_id: u64,
    pub source_generation: u64,
    pub operation_generation: u64,
    pub path: PathBuf,
    pub roi_id_column: String,
    pub x_column: String,
    pub y_column: String,
    pub value_column: String,
    pub load_dataset: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct AnnotationLoadResult {
    pub schema: Vec<AnnotationColumnInfo>,
    pub dataset: Option<AnnotationDataset>,
}

#[derive(Debug, Clone)]
struct AnnotationLayerModel {
    state: ProjectAnnotationLayerState,
    generation: u64,
    source_generation: u64,
    resource_generation: u64,
    operation_generation: u64,
    resource: Option<Arc<ControlAnnotationResource>>,
    schema: Arc<Vec<AnnotationColumnInfo>>,
    pending: bool,
    status: String,
}

impl AnnotationLayerModel {
    fn new(state: ProjectAnnotationLayerState) -> Result<Self, ControlError> {
        validate_state(&state)?;
        Ok(Self {
            state,
            generation: 1,
            source_generation: 1,
            resource_generation: 0,
            operation_generation: 0,
            resource: None,
            schema: Arc::new(Vec::new()),
            pending: false,
            status: String::new(),
        })
    }

    fn projection(&self) -> ControlAnnotationLayerProjection {
        ControlAnnotationLayerProjection {
            state: self.state.clone(),
            generation: self.generation,
            resource_generation: self.resource_generation,
            resource: self.resource.clone(),
            schema: Arc::clone(&self.schema),
            pending: self.pending,
            status: self.status.clone(),
        }
    }

    fn snapshot(&self) -> Value {
        let mut value = serde_json::to_value(&self.state).expect("annotation state serializes");
        let object = value
            .as_object_mut()
            .expect("annotation state is an object");
        object.extend([
            ("generation".to_string(), json!(self.generation)),
            (
                "resource_generation".to_string(),
                json!(self.resource_generation),
            ),
            ("pending".to_string(), json!(self.pending)),
            ("status".to_string(), json!(self.status)),
            (
                "schema".to_string(),
                json!(
                    self.schema
                        .iter()
                        .map(|column| &column.name)
                        .collect::<Vec<_>>()
                ),
            ),
            (
                "resource".to_string(),
                self.resource.as_ref().map_or(Value::Null, |resource| {
                    json!({
                        "mode":match resource.dataset.mode {
                            AnnotationValueMode::Categorical => "categorical",
                            AnnotationValueMode::Continuous => "continuous",
                        },
                        "total_points":resource.dataset.total_points,
                        "total_rois":resource.dataset.total_rois,
                        "categories":resource.dataset.categories,
                        "value_min":resource.dataset.value_min,
                        "value_max":resource.dataset.value_max,
                    })
                }),
            ),
        ]);
        value
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AnnotationModel {
    layers: Vec<AnnotationLayerModel>,
    next_id: u64,
    generation: u64,
}

impl Default for AnnotationModel {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            next_id: 1,
            generation: 1,
        }
    }
}

impl AnnotationModel {
    pub(crate) fn restorable_layer_ids(&self) -> Vec<u64> {
        self.layers
            .iter()
            .filter(|layer| {
                layer.state.parquet_path.is_some()
                    && layer.resource.is_none()
                    && !layer.pending
                    && layer.source_generation == 1
            })
            .map(|layer| layer.state.id)
            .collect()
    }

    pub(crate) fn restore(
        &mut self,
        states: Vec<ProjectAnnotationLayerState>,
    ) -> Result<(), ControlError> {
        let mut ids = std::collections::HashSet::new();
        let mut layers = Vec::with_capacity(states.len());
        for state in states {
            if !ids.insert(state.id) {
                return Err(invalid("annotation layer IDs must be unique"));
            }
            layers.push(AnnotationLayerModel::new(state)?);
        }
        self.next_id = layers
            .iter()
            .map(|layer| layer.state.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
            .max(1);
        self.layers = layers;
        self.generation = self.generation.wrapping_add(1).max(1);
        Ok(())
    }

    pub(crate) fn states(&self) -> Vec<ProjectAnnotationLayerState> {
        self.layers
            .iter()
            .map(|layer| layer.state.clone())
            .collect()
    }

    pub(crate) fn projections(&self) -> Vec<ControlAnnotationLayerProjection> {
        self.layers
            .iter()
            .map(AnnotationLayerModel::projection)
            .collect()
    }

    pub(crate) fn snapshot(&self) -> Value {
        json!({
            "generation":self.generation,
            "layers":self.layers.iter().map(AnnotationLayerModel::snapshot).collect::<Vec<_>>(),
        })
    }

    pub(crate) fn layer_snapshot(&self, id: u64) -> Result<Value, ControlError> {
        self.layer(id).map(AnnotationLayerModel::snapshot)
    }

    pub(crate) fn create(&mut self, params: &Value) -> Result<Value, ControlError> {
        let id = self.next_id.max(1);
        self.next_id = id.saturating_add(1).max(1);
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| format!("Annotations {id}"));
        let mut state = default_state(id, name);
        apply_state_patch(&mut state, params)?;
        self.layers.push(AnnotationLayerModel::new(state)?);
        self.generation = self.generation.wrapping_add(1).max(1);
        self.layer_snapshot(id)
    }

    pub(crate) fn update(&mut self, id: u64, params: &Value) -> Result<Value, ControlError> {
        let layer = self.layer_mut(id)?;
        let before_source = source_identity(&layer.state);
        apply_state_patch(&mut layer.state, params)?;
        validate_state(&layer.state)?;
        if source_identity(&layer.state) != before_source {
            layer.source_generation = layer.source_generation.wrapping_add(1).max(1);
            layer.operation_generation = layer.operation_generation.wrapping_add(1).max(1);
            layer.resource = None;
            layer.resource_generation = layer.resource_generation.wrapping_add(1).max(1);
            layer.pending = false;
            layer.status = "Annotation source configuration changed; load to refresh.".to_string();
        }
        layer.generation = layer.generation.wrapping_add(1).max(1);
        self.generation = self.generation.wrapping_add(1).max(1);
        self.layer_snapshot(id)
    }

    pub(crate) fn delete(&mut self, id: u64) -> Result<Value, ControlError> {
        let index = self
            .layers
            .iter()
            .position(|layer| layer.state.id == id)
            .ok_or_else(|| not_found(id))?;
        self.layers.remove(index);
        self.generation = self.generation.wrapping_add(1).max(1);
        Ok(json!({"id":id,"deleted":true,"generation":self.generation}))
    }

    pub(crate) fn clear_source(&mut self, id: u64) -> Result<Value, ControlError> {
        let layer = self.layer_mut(id)?;
        layer.state.parquet_path = None;
        layer.resource = None;
        layer.schema = Arc::new(Vec::new());
        layer.pending = false;
        layer.operation_generation = layer.operation_generation.wrapping_add(1).max(1);
        layer.source_generation = layer.source_generation.wrapping_add(1).max(1);
        layer.resource_generation = layer.resource_generation.wrapping_add(1).max(1);
        layer.generation = layer.generation.wrapping_add(1).max(1);
        layer.status = "Annotation source cleared.".to_string();
        self.generation = self.generation.wrapping_add(1).max(1);
        self.layer_snapshot(id)
    }

    pub(crate) fn begin_load(
        &mut self,
        document_generation: u64,
        id: u64,
        params: &Value,
        load_dataset: bool,
    ) -> Result<AnnotationLoadSpec, ControlError> {
        let layer = self.layer_mut(id)?;
        apply_source_params(&mut layer.state, params)?;
        validate_state(&layer.state)?;
        let path = layer
            .state
            .parquet_path
            .as_deref()
            .map(PathBuf::from)
            .ok_or_else(|| invalid("annotation source path is required"))?;
        layer.source_generation = layer.source_generation.wrapping_add(1).max(1);
        layer.operation_generation = layer.operation_generation.wrapping_add(1).max(1);
        layer.pending = true;
        layer.status = if load_dataset {
            format!("Loading annotations from {}", path.display())
        } else {
            format!("Inspecting annotation schema from {}", path.display())
        };
        layer.generation = layer.generation.wrapping_add(1).max(1);
        let source_generation = layer.source_generation;
        let operation_generation = layer.operation_generation;
        let roi_id_column = layer.state.roi_id_column.clone();
        let x_column = layer.state.x_column.clone();
        let y_column = layer.state.y_column.clone();
        let value_column = layer.state.selected_value_column.clone();
        self.generation = self.generation.wrapping_add(1).max(1);
        Ok(AnnotationLoadSpec {
            document_generation,
            layer_id: id,
            source_generation,
            operation_generation,
            path,
            roi_id_column,
            x_column,
            y_column,
            value_column,
            load_dataset,
        })
    }

    pub(crate) fn load_is_current(&self, spec: &AnnotationLoadSpec) -> bool {
        self.layers.iter().any(|layer| {
            layer.state.id == spec.layer_id
                && layer.source_generation == spec.source_generation
                && layer.operation_generation == spec.operation_generation
        })
    }

    pub(crate) fn finish_load(
        &mut self,
        spec: &AnnotationLoadSpec,
        result: AnnotationLoadResult,
    ) -> Option<Value> {
        if !self.load_is_current(spec) {
            return None;
        }
        let layer = self.layer_mut(spec.layer_id).ok()?;
        layer.schema = Arc::new(result.schema);
        if let Some(dataset) = result.dataset {
            if dataset.mode == AnnotationValueMode::Categorical {
                let defaults = default_category_style_states(&dataset.categories);
                let existing = layer
                    .state
                    .category_styles
                    .iter()
                    .map(|style| (style.name.clone(), style.clone()))
                    .collect::<std::collections::HashMap<_, _>>();
                layer.state.category_styles = defaults
                    .into_iter()
                    .map(|style| existing.get(&style.name).cloned().unwrap_or(style))
                    .collect();
                layer.state.continuous_range = None;
            } else {
                layer.state.category_styles.clear();
                if layer.state.continuous_range.is_none() {
                    layer.state.continuous_range = Some([dataset.value_min, dataset.value_max]);
                }
            }
            layer.state.value_column = layer.state.selected_value_column.clone();
            layer.resource = Some(Arc::new(ControlAnnotationResource {
                dataset: Arc::new(dataset),
            }));
            layer.resource_generation = layer.resource_generation.wrapping_add(1).max(1);
        }
        layer.pending = false;
        layer.status = layer.resource.as_ref().map_or_else(
            || format!("Read {} annotation columns.", layer.schema.len()),
            |resource| {
                format!(
                    "Loaded {} points across {} ROIs.",
                    resource.dataset.total_points, resource.dataset.total_rois
                )
            },
        );
        layer.generation = layer.generation.wrapping_add(1).max(1);
        self.generation = self.generation.wrapping_add(1).max(1);
        self.layer_snapshot(spec.layer_id).ok()
    }

    pub(crate) fn fail_load(&mut self, spec: &AnnotationLoadSpec, message: String) -> bool {
        if !self.load_is_current(spec) {
            return false;
        }
        let Ok(layer) = self.layer_mut(spec.layer_id) else {
            return false;
        };
        layer.pending = false;
        layer.status = message;
        layer.generation = layer.generation.wrapping_add(1).max(1);
        self.generation = self.generation.wrapping_add(1).max(1);
        true
    }

    fn layer(&self, id: u64) -> Result<&AnnotationLayerModel, ControlError> {
        self.layers
            .iter()
            .find(|layer| layer.state.id == id)
            .ok_or_else(|| not_found(id))
    }

    fn layer_mut(&mut self, id: u64) -> Result<&mut AnnotationLayerModel, ControlError> {
        self.layers
            .iter_mut()
            .find(|layer| layer.state.id == id)
            .ok_or_else(|| not_found(id))
    }
}

fn default_state(id: u64, name: String) -> ProjectAnnotationLayerState {
    ProjectAnnotationLayerState {
        id,
        name,
        visible: true,
        radius_screen_px: 4.0,
        opacity: 0.9,
        stroke_width: 1.0,
        stroke_color_rgb: [0, 0, 0],
        stroke_color_alpha: 140,
        offset_world: [0.0, 0.0],
        parquet_path: None,
        roi_id_column: "id".to_string(),
        x_column: "x_centroid".to_string(),
        y_column: "y_centroid".to_string(),
        value_column: "cluster_label".to_string(),
        selected_value_column: "cluster_label".to_string(),
        category_styles: Vec::new(),
        continuous_shape: Some("circle".to_string()),
        continuous_range: None,
    }
}

fn apply_state_patch(
    state: &mut ProjectAnnotationLayerState,
    params: &Value,
) -> Result<(), ControlError> {
    let patch = params.get("state").unwrap_or(params);
    let patch = patch
        .as_object()
        .ok_or_else(|| invalid("annotation layer patch must be an object"))?;
    let allowed = [
        "name",
        "visible",
        "radius_screen_px",
        "opacity",
        "stroke_width",
        "stroke_color_rgb",
        "stroke_color_alpha",
        "offset_world",
        "roi_id_column",
        "x_column",
        "y_column",
        "value_column",
        "selected_value_column",
        "category_styles",
        "continuous_shape",
        "continuous_range",
    ];
    for key in patch.keys() {
        if !allowed.contains(&key.as_str()) && !matches!(key.as_str(), "id" | "if_revision") {
            return Err(invalid(format!("unknown annotation layer field '{key}'")));
        }
    }
    let mut value = serde_json::to_value(&*state).expect("annotation state serializes");
    let object = value
        .as_object_mut()
        .expect("annotation state is an object");
    for (key, value) in patch {
        if allowed.contains(&key.as_str()) {
            object.insert(key.clone(), value.clone());
        }
    }
    *state = serde_json::from_value(value)
        .map_err(|error| invalid(format!("invalid annotation layer patch: {error}")))?;
    Ok(())
}

fn apply_source_params(
    state: &mut ProjectAnnotationLayerState,
    params: &Value,
) -> Result<(), ControlError> {
    let object = params
        .as_object()
        .ok_or_else(|| invalid("annotation source parameters must be an object"))?;
    for (key, value) in object {
        match key.as_str() {
            "id" | "layer_id" | "if_revision" => {}
            "path" | "parquet_path" => {
                state.parquet_path = value.as_str().map(str::to_string);
            }
            "roi_id_column" => state.roi_id_column = required_string(value, key)?,
            "x_column" => state.x_column = required_string(value, key)?,
            "y_column" => state.y_column = required_string(value, key)?,
            "value_column" | "selected_value_column" => {
                state.selected_value_column = required_string(value, key)?;
            }
            other => {
                return Err(invalid(format!(
                    "unknown annotation source field '{other}'"
                )));
            }
        }
    }
    Ok(())
}

fn validate_state(state: &ProjectAnnotationLayerState) -> Result<(), ControlError> {
    if state.id == 0 || state.name.trim().is_empty() {
        return Err(invalid(
            "annotation layers require a positive ID and non-empty name",
        ));
    }
    for (name, value, min, max) in [
        ("radius_screen_px", state.radius_screen_px, 0.1, 128.0),
        ("opacity", state.opacity, 0.0, 1.0),
        ("stroke_width", state.stroke_width, 0.0, 64.0),
    ] {
        if !value.is_finite() || !(min..=max).contains(&value) {
            return Err(invalid(format!("{name} is outside its supported range")));
        }
    }
    if state.offset_world.iter().any(|value| !value.is_finite()) {
        return Err(invalid("annotation offset must be finite"));
    }
    for (name, value) in [
        ("roi_id_column", &state.roi_id_column),
        ("x_column", &state.x_column),
        ("y_column", &state.y_column),
        ("value_column", &state.value_column),
        ("selected_value_column", &state.selected_value_column),
    ] {
        if value.trim().is_empty() {
            return Err(invalid(format!("{name} must not be empty")));
        }
    }
    if state
        .continuous_shape
        .as_deref()
        .is_some_and(|shape| !matches!(shape, "circle" | "square" | "diamond" | "cross"))
        || state.category_styles.iter().any(|style| {
            style.name.trim().is_empty()
                || !matches!(
                    style.shape.as_str(),
                    "circle" | "square" | "diamond" | "cross"
                )
        })
    {
        return Err(invalid("annotation shape is invalid"));
    }
    if state
        .continuous_range
        .is_some_and(|[low, high]| !low.is_finite() || !high.is_finite() || high < low)
    {
        return Err(invalid("annotation continuous range is invalid"));
    }
    Ok(())
}

fn source_identity(
    state: &ProjectAnnotationLayerState,
) -> (Option<String>, String, String, String, String) {
    (
        state.parquet_path.clone(),
        state.roi_id_column.clone(),
        state.x_column.clone(),
        state.y_column.clone(),
        state.selected_value_column.clone(),
    )
}

fn required_string(value: &Value, name: &str) -> Result<String, ControlError> {
    value
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| invalid(format!("{name} must be a non-empty string")))
}

fn invalid(message: impl Into<String>) -> ControlError {
    ControlError::new(ControlErrorKind::InvalidParams, message)
}

fn not_found(id: u64) -> ControlError {
    ControlError::new(
        ControlErrorKind::ResourceNotFound,
        format!("annotation layer {id} was not found"),
    )
}
