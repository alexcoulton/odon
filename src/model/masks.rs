use std::fs;
use std::path::Path;

use anyhow::Context;
use serde_json::{Value, json};

use crate::control::{ControlError, ControlErrorKind};
use crate::data::project_config::ProjectMaskLayer;

const MAX_UNDO_STATES: usize = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MaskSelection {
    layer_id: u64,
    polygon_index: usize,
    vertex_index: Option<usize>,
}

#[derive(Debug, Clone)]
struct MaskUndoState {
    layers: Vec<ProjectMaskLayer>,
    next_id: u64,
    active_layer_id: Option<u64>,
    selection: Option<MaskSelection>,
    dirty: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct MaskModel {
    layers: Vec<ProjectMaskLayer>,
    next_id: u64,
    active_layer_id: Option<u64>,
    selection: Option<MaskSelection>,
    undo: Vec<MaskUndoState>,
    generation: u64,
    dirty: bool,
}

impl Default for MaskModel {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            next_id: 1,
            active_layer_id: None,
            selection: None,
            undo: Vec::new(),
            generation: 1,
            dirty: false,
        }
    }
}

impl MaskModel {
    pub(crate) fn replace(&mut self, layers: Vec<ProjectMaskLayer>, active_layer_id: Option<u64>) {
        self.next_id = layers
            .iter()
            .map(|layer| layer.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
            .max(1);
        self.layers = layers;
        self.active_layer_id =
            active_layer_id.filter(|id| self.layers.iter().any(|layer| layer.id == *id));
        self.selection = None;
        self.undo.clear();
        self.generation = self.generation.wrapping_add(1).max(1);
        self.dirty = false;
    }

    pub(crate) fn restore_projection(&mut self, projection: &Value) -> Result<(), ControlError> {
        let layers = projection
            .get("layers")
            .cloned()
            .map(serde_json::from_value::<Vec<ProjectMaskLayer>>)
            .transpose()
            .map_err(|error| invalid(format!("renderer mask layers are invalid: {error}")))?
            .unwrap_or_default();
        let active_layer_id = projection.get("active_layer_id").and_then(Value::as_u64);
        self.replace(layers, active_layer_id);
        self.dirty = projection
            .get("dirty")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        self.selection = parse_projection_selection(projection.get("selection"), &self.layers)?;
        self.generation = projection
            .get("generation")
            .and_then(Value::as_u64)
            .unwrap_or(self.generation)
            .max(1);
        Ok(())
    }

    pub(crate) fn projection_json(&self) -> Value {
        json!({
            "generation": self.generation,
            "active_layer_id": self.active_layer_id,
            "layers": self.layers,
            "selection": self.selection_json(),
            "dirty": self.dirty,
            "undo_available": !self.undo.is_empty(),
        })
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn dirty(&self) -> bool {
        self.dirty
    }

    pub(crate) fn mark_persisted(&mut self) {
        self.dirty = false;
    }

    pub(crate) fn export_layers(
        &self,
        layer_id: Option<u64>,
    ) -> Result<Vec<ProjectMaskLayer>, ControlError> {
        match layer_id {
            Some(id) => Ok(vec![self.layer(id)?.clone()]),
            None => Ok(self.layers.clone()),
        }
    }

    pub(crate) fn install_imported_layer(
        &mut self,
        name: String,
        editable: bool,
        polygons_world: Vec<Vec<[f32; 2]>>,
        source_geojson: std::path::PathBuf,
    ) -> Value {
        self.push_undo();
        let id = self.next_id.max(1);
        self.next_id = id.saturating_add(1).max(1);
        let polygon_count = polygons_world.len();
        self.layers.push(ProjectMaskLayer {
            id,
            name,
            visible: true,
            opacity: 0.85,
            width_screen_px: 1.5,
            display_mode: Some("translucent_fill".to_string()),
            color_rgb: [50, 220, 255],
            offset_world: [0.0, 0.0],
            editable,
            polygons_world,
            source_geojson: Some(source_geojson.clone()),
        });
        self.active_layer_id = Some(id);
        self.commit();
        json!({
            "imported": true,
            "path": source_geojson.to_string_lossy(),
            "layer_id": id,
            "polygon_count": polygon_count,
            "layer": self.layer_json(self.layers.last().expect("imported mask layer exists")),
        })
    }

    pub(crate) fn install_generated_threshold_layer(
        &mut self,
        name: String,
        polygons_world: Vec<Vec<[f32; 2]>>,
    ) -> Value {
        self.push_undo();
        let id = self.next_id.max(1);
        self.next_id = id.saturating_add(1).max(1);
        let polygon_count = polygons_world.len();
        self.layers.push(ProjectMaskLayer {
            id,
            name,
            visible: true,
            opacity: 0.85,
            width_screen_px: 1.5,
            display_mode: Some("filled_preview".to_string()),
            color_rgb: [255, 210, 80],
            offset_world: [0.0, 0.0],
            editable: true,
            polygons_world,
            source_geojson: None,
        });
        self.active_layer_id = Some(id);
        self.commit();
        let mut response = self.layer_json(self.layers.last().expect("threshold mask exists"));
        response["polygon_count"] = json!(polygon_count);
        response
    }

    pub(crate) fn dispatch(&mut self, method: &str, params: &Value) -> Result<Value, ControlError> {
        match method {
            "viewer.masks.layers.list" => Ok(self.list_layers()),
            "viewer.masks.layers.get" => self.get_layer(params),
            "viewer.masks.layers.create" => self.create_layer(params),
            "viewer.masks.layers.update" => self.update_layer(params),
            "viewer.masks.layers.delete" => self.delete_layer(params),
            "viewer.masks.polygons.list" => self.list_polygons(params),
            "viewer.masks.polygons.add" => self.add_polygon(params),
            "viewer.masks.polygons.update" => self.update_polygon(params),
            "viewer.masks.polygons.remove" => self.remove_polygon(params),
            "viewer.masks.selection.get" => Ok(json!({"selection": self.selection_json()})),
            "viewer.masks.selection.set" => self.set_selection(params),
            "viewer.masks.selection.clear" => Ok(self.clear_selection()),
            "viewer.masks.undo" => Ok(self.undo()),
            "viewer.masks.state.replace" => self.replace_transaction(params),
            _ => Err(ControlError::new(
                ControlErrorKind::MethodNotFound,
                format!("unknown mask model method '{method}'"),
            )),
        }
    }

    fn list_layers(&self) -> Value {
        json!({
            "total": self.layers.len(),
            "layers": self.layers.iter().map(|layer| self.layer_json(layer)).collect::<Vec<_>>(),
            "undo_available": !self.undo.is_empty(),
        })
    }

    fn get_layer(&self, params: &Value) -> Result<Value, ControlError> {
        let id = required_id(params)?;
        let layer = self.layer(id)?;
        Ok(self.layer_json(layer))
    }

    fn create_layer(&mut self, params: &Value) -> Result<Value, ControlError> {
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| format!("Mask {}", self.next_id));
        let editable = params
            .get("editable")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        self.push_undo();
        let id = self.next_id.max(1);
        self.next_id = id.saturating_add(1).max(1);
        self.layers.push(ProjectMaskLayer {
            id,
            name,
            visible: true,
            opacity: 0.85,
            width_screen_px: 1.5,
            display_mode: Some("translucent_fill".to_string()),
            color_rgb: [50, 220, 255],
            offset_world: [0.0, 0.0],
            editable,
            polygons_world: Vec::new(),
            source_geojson: None,
        });
        self.active_layer_id = Some(id);
        self.commit();
        Ok(self.layer_json(self.layers.last().expect("created mask layer exists")))
    }

    fn update_layer(&mut self, params: &Value) -> Result<Value, ControlError> {
        let id = required_id(params)?;
        let index = self.layer_index(id)?;
        let name = optional_nonempty_string(params, "name")?;
        let opacity = optional_bounded_f32(params, "opacity", 0.0, 1.0, true)?;
        let width = optional_bounded_f32(params, "width_screen_px", 0.0, f32::MAX, false)?;
        let display_mode = match params.get("display_mode") {
            Some(value) => {
                let value = value
                    .as_str()
                    .ok_or_else(|| invalid("display_mode must be a string"))?;
                if !matches!(
                    value,
                    "outline_only"
                        | "outline"
                        | "translucent_fill"
                        | "fill_outline"
                        | "semi_transparent_fill"
                        | "filled_preview"
                        | "mask_preview"
                ) {
                    return Err(invalid("unknown mask display_mode"));
                }
                Some(
                    match value {
                        "outline" => "outline_only",
                        "fill_outline" | "semi_transparent_fill" => "translucent_fill",
                        "mask_preview" => "filled_preview",
                        other => other,
                    }
                    .to_string(),
                )
            }
            None => None,
        };
        let color = optional_rgb(params, "color_rgb")?;
        let offset = optional_pair(params, "offset_world")?;
        let visible = optional_bool(params, "visible")?;
        let editable = optional_bool(params, "editable")?;
        self.push_undo();
        let layer = &mut self.layers[index];
        if let Some(value) = name {
            layer.name = value;
        }
        if let Some(value) = opacity {
            layer.opacity = value;
        }
        if let Some(value) = width {
            layer.width_screen_px = value;
        }
        if let Some(value) = display_mode {
            layer.display_mode = Some(value);
        }
        if let Some(value) = color {
            layer.color_rgb = value;
        }
        if let Some(value) = offset {
            layer.offset_world = value;
        }
        if let Some(value) = visible {
            layer.visible = value;
        }
        if let Some(value) = editable {
            layer.editable = value;
        }
        self.commit();
        Ok(self.layer_json(&self.layers[index]))
    }

    fn delete_layer(&mut self, params: &Value) -> Result<Value, ControlError> {
        let id = required_id(params)?;
        let index = self.layer_index(id)?;
        self.push_undo();
        self.layers.remove(index);
        if self.active_layer_id == Some(id) {
            self.active_layer_id = self.layers.first().map(|layer| layer.id);
        }
        if self
            .selection
            .is_some_and(|selection| selection.layer_id == id)
        {
            self.selection = None;
        }
        self.commit();
        Ok(json!({"deleted": true, "id": id}))
    }

    fn list_polygons(&self, params: &Value) -> Result<Value, ControlError> {
        let id = required_id(params)?;
        let layer = self.layer(id)?;
        let offset = usize_param(params, "offset", 0)?;
        let limit = usize_param(params, "limit", 200)?.min(10_000);
        let polygons = layer
            .polygons_world
            .iter()
            .enumerate()
            .skip(offset)
            .take(limit)
            .map(|(index, polygon)| polygon_json(layer, index, polygon))
            .collect::<Vec<_>>();
        Ok(json!({
            "layer_id": id,
            "total": layer.polygons_world.len(),
            "offset": offset,
            "limit": limit,
            "has_more": offset.saturating_add(polygons.len()) < layer.polygons_world.len(),
            "polygons": polygons,
        }))
    }

    fn add_polygon(&mut self, params: &Value) -> Result<Value, ControlError> {
        let id = required_id(params)?;
        let index = self.layer_index(id)?;
        if !self.layers[index].editable {
            return Err(invalid(format!("mask layer {id} is read-only")));
        }
        let vertices = parse_vertices(params, self.layers[index].offset_world)?;
        self.push_undo();
        self.layers[index]
            .polygons_world
            .push(close_polygon(vertices));
        let polygon_index = self.layers[index].polygons_world.len() - 1;
        self.commit();
        Ok(json!({"added": true, "layer_id": id, "index": polygon_index}))
    }

    fn update_polygon(&mut self, params: &Value) -> Result<Value, ControlError> {
        let id = required_id(params)?;
        let layer_index = self.layer_index(id)?;
        let polygon_index = required_index(params)?;
        if !self.layers[layer_index].editable {
            return Err(invalid(format!("mask layer {id} is read-only")));
        }
        if polygon_index >= self.layers[layer_index].polygons_world.len() {
            return Err(not_found(format!(
                "mask polygon index {polygon_index} is out of range"
            )));
        }
        let vertices = parse_vertices(params, self.layers[layer_index].offset_world)?;
        self.push_undo();
        self.layers[layer_index].polygons_world[polygon_index] = close_polygon(vertices);
        self.commit();
        Ok(json!({"updated": true, "layer_id": id, "index": polygon_index}))
    }

    fn remove_polygon(&mut self, params: &Value) -> Result<Value, ControlError> {
        let id = required_id(params)?;
        let layer_index = self.layer_index(id)?;
        let polygon_index = required_index(params)?;
        if !self.layers[layer_index].editable {
            return Err(invalid(format!("mask layer {id} is read-only")));
        }
        if polygon_index >= self.layers[layer_index].polygons_world.len() {
            return Err(not_found(format!(
                "mask polygon index {polygon_index} is out of range"
            )));
        }
        self.push_undo();
        self.layers[layer_index]
            .polygons_world
            .remove(polygon_index);
        if self.selection.is_some_and(|selection| {
            selection.layer_id == id && selection.polygon_index == polygon_index
        }) {
            self.selection = None;
        }
        self.commit();
        Ok(json!({"removed": true, "layer_id": id, "index": polygon_index}))
    }

    fn set_selection(&mut self, params: &Value) -> Result<Value, ControlError> {
        let layer_id = required_id(params)?;
        let polygon_index = required_index(params)?;
        let layer = self.layer(layer_id)?;
        let polygon = layer.polygons_world.get(polygon_index).ok_or_else(|| {
            not_found(format!(
                "mask polygon index {polygon_index} is out of range"
            ))
        })?;
        let vertex_index = match params.get("vertex_index") {
            None | Some(Value::Null) => None,
            Some(value) => {
                let index = value
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| {
                        invalid("vertex_index must be a non-negative integer or null")
                    })?;
                if index >= unique_vertex_count(polygon) {
                    return Err(not_found(format!(
                        "mask vertex index {index} is out of range"
                    )));
                }
                Some(index)
            }
        };
        self.selection = Some(MaskSelection {
            layer_id,
            polygon_index,
            vertex_index,
        });
        self.active_layer_id = Some(layer_id);
        self.generation = self.generation.wrapping_add(1).max(1);
        Ok(json!({"selection": self.selection_json()}))
    }

    fn clear_selection(&mut self) -> Value {
        let cleared = self.selection.take().is_some();
        if cleared {
            self.generation = self.generation.wrapping_add(1).max(1);
        }
        json!({"cleared": cleared, "selection": Value::Null})
    }

    fn undo(&mut self) -> Value {
        let Some(state) = self.undo.pop() else {
            return json!({"undone": false, "undo_available": false});
        };
        self.layers = state.layers;
        self.next_id = state.next_id;
        self.active_layer_id = state.active_layer_id;
        self.selection = state.selection;
        self.dirty = state.dirty;
        self.generation = self.generation.wrapping_add(1).max(1);
        json!({"undone": true, "undo_available": !self.undo.is_empty()})
    }

    fn replace_transaction(&mut self, params: &Value) -> Result<Value, ControlError> {
        if let Some(expected) = params.get("expected_generation").and_then(Value::as_u64)
            && expected != self.generation
        {
            return Err(ControlError::new(
                ControlErrorKind::Conflict,
                format!(
                    "mask generation conflict: expected {expected}, current {}",
                    self.generation
                ),
            )
            .with_data(json!({
                "expected_generation": expected,
                "current_generation": self.generation,
            })));
        }
        let state = params.get("state").unwrap_or(params);
        let layers = state
            .get("layers")
            .cloned()
            .map(serde_json::from_value::<Vec<ProjectMaskLayer>>)
            .transpose()
            .map_err(|error| invalid(format!("mask replacement layers are invalid: {error}")))?
            .ok_or_else(|| invalid("mask replacement requires layers"))?;
        validate_layers(&layers)?;
        let active_layer_id = state.get("active_layer_id").and_then(Value::as_u64);
        if active_layer_id.is_some_and(|id| !layers.iter().any(|layer| layer.id == id)) {
            return Err(invalid("active_layer_id references an unknown mask layer"));
        }
        let selection = parse_projection_selection(state.get("selection"), &layers)?;
        self.push_undo();
        self.next_id = layers
            .iter()
            .map(|layer| layer.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
            .max(1);
        self.layers = layers;
        self.active_layer_id = active_layer_id;
        self.selection = selection;
        self.commit();
        Ok(self.projection_json())
    }

    fn layer_json(&self, layer: &ProjectMaskLayer) -> Value {
        json!({
            "id": layer.id,
            "name": layer.name,
            "visible": layer.visible,
            "opacity": layer.opacity,
            "width_screen_px": layer.width_screen_px,
            "display_mode": layer.display_mode.as_deref().unwrap_or("outline_only"),
            "color_rgb": layer.color_rgb,
            "offset_world": layer.offset_world,
            "editable": layer.editable,
            "polygon_count": layer.polygons_world.len(),
            "source_geojson": layer.source_geojson,
            "active": self.active_layer_id == Some(layer.id),
        })
    }

    fn selection_json(&self) -> Value {
        let Some(selection) = self.selection else {
            return Value::Null;
        };
        let Some(layer) = self
            .layers
            .iter()
            .find(|layer| layer.id == selection.layer_id)
        else {
            return Value::Null;
        };
        let Some(polygon) = layer.polygons_world.get(selection.polygon_index) else {
            return Value::Null;
        };
        json!({
            "layer_id": selection.layer_id,
            "polygon_index": selection.polygon_index,
            "vertex_index": selection.vertex_index,
            "vertices_local": polygon,
            "vertices_world": polygon.iter().map(|point| [
                point[0] + layer.offset_world[0],
                point[1] + layer.offset_world[1],
            ]).collect::<Vec<_>>(),
        })
    }

    fn layer(&self, id: u64) -> Result<&ProjectMaskLayer, ControlError> {
        self.layers
            .iter()
            .find(|layer| layer.id == id)
            .ok_or_else(|| not_found(format!("mask layer {id} not found")))
    }

    fn layer_index(&self, id: u64) -> Result<usize, ControlError> {
        self.layers
            .iter()
            .position(|layer| layer.id == id)
            .ok_or_else(|| not_found(format!("mask layer {id} not found")))
    }

    fn push_undo(&mut self) {
        if self.undo.len() == MAX_UNDO_STATES {
            self.undo.remove(0);
        }
        self.undo.push(MaskUndoState {
            layers: self.layers.clone(),
            next_id: self.next_id,
            active_layer_id: self.active_layer_id,
            selection: self.selection,
            dirty: self.dirty,
        });
    }

    fn commit(&mut self) {
        self.dirty = true;
        self.generation = self.generation.wrapping_add(1).max(1);
    }
}

fn required_id(params: &Value) -> Result<u64, ControlError> {
    params
        .get("id")
        .or_else(|| params.get("layer_id"))
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid("mask layer id is required"))
}

fn required_index(params: &Value) -> Result<usize, ControlError> {
    params
        .get("index")
        .or_else(|| params.get("polygon_index"))
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| invalid("polygon index is required"))
}

fn optional_nonempty_string(params: &Value, name: &str) -> Result<Option<String>, ControlError> {
    params
        .get(name)
        .map(|value| {
            value
                .as_str()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .ok_or_else(|| invalid(format!("mask layer {name} must not be empty")))
        })
        .transpose()
}

fn optional_bool(params: &Value, name: &str) -> Result<Option<bool>, ControlError> {
    params
        .get(name)
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| invalid(format!("{name} must be a boolean")))
        })
        .transpose()
}

fn optional_bounded_f32(
    params: &Value,
    name: &str,
    min: f32,
    max: f32,
    inclusive_min: bool,
) -> Result<Option<f32>, ControlError> {
    params
        .get(name)
        .map(|value| {
            let value = value
                .as_f64()
                .filter(|value| value.is_finite())
                .map(|value| value as f32)
                .ok_or_else(|| invalid(format!("{name} must be finite")))?;
            let lower = if inclusive_min {
                value >= min
            } else {
                value > min
            };
            if !lower || value > max {
                return Err(invalid(format!("{name} is outside its allowed range")));
            }
            Ok(value)
        })
        .transpose()
}

fn optional_rgb(params: &Value, name: &str) -> Result<Option<[u8; 3]>, ControlError> {
    params
        .get(name)
        .map(|value| {
            let values = value
                .as_array()
                .filter(|values| values.len() == 3)
                .ok_or_else(|| invalid(format!("{name} must contain three integers")))?;
            Ok([
                u8::try_from(
                    values[0]
                        .as_u64()
                        .ok_or_else(|| invalid("invalid red value"))?,
                )
                .map_err(|_| invalid("invalid red value"))?,
                u8::try_from(
                    values[1]
                        .as_u64()
                        .ok_or_else(|| invalid("invalid green value"))?,
                )
                .map_err(|_| invalid("invalid green value"))?,
                u8::try_from(
                    values[2]
                        .as_u64()
                        .ok_or_else(|| invalid("invalid blue value"))?,
                )
                .map_err(|_| invalid("invalid blue value"))?,
            ])
        })
        .transpose()
}

fn optional_pair(params: &Value, name: &str) -> Result<Option<[f32; 2]>, ControlError> {
    params
        .get(name)
        .map(|value| {
            let values = value
                .as_array()
                .filter(|values| values.len() == 2)
                .ok_or_else(|| invalid(format!("{name} must contain two finite numbers")))?;
            let x = values[0]
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| invalid(format!("{name}[0] must be finite")))?
                as f32;
            let y = values[1]
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| invalid(format!("{name}[1] must be finite")))?
                as f32;
            Ok([x, y])
        })
        .transpose()
}

fn usize_param(params: &Value, name: &str, default: usize) -> Result<usize, ControlError> {
    params
        .get(name)
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| invalid(format!("{name} must be a non-negative integer")))
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn parse_vertices(params: &Value, layer_offset: [f32; 2]) -> Result<Vec<[f32; 2]>, ControlError> {
    let values = params
        .get("vertices")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("vertices is required"))?;
    if values.len() < 3 {
        return Err(invalid("vertices must contain at least three points"));
    }
    let world = match params
        .get("coordinate_space")
        .and_then(Value::as_str)
        .unwrap_or("world")
    {
        "world" => true,
        "local" => false,
        _ => return Err(invalid("coordinate_space must be 'world' or 'local'")),
    };
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let pair = value
                .as_array()
                .filter(|pair| pair.len() == 2)
                .ok_or_else(|| invalid(format!("vertices[{index}] must be [x, y]")))?;
            let x = pair[0]
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| invalid(format!("vertices[{index}][0] must be finite")))?
                as f32;
            let y = pair[1]
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| invalid(format!("vertices[{index}][1] must be finite")))?
                as f32;
            Ok(if world {
                [x - layer_offset[0], y - layer_offset[1]]
            } else {
                [x, y]
            })
        })
        .collect()
}

fn close_polygon(mut vertices: Vec<[f32; 2]>) -> Vec<[f32; 2]> {
    if vertices.first() != vertices.last()
        && let Some(first) = vertices.first().copied()
    {
        vertices.push(first);
    }
    vertices
}

fn unique_vertex_count(polygon: &[[f32; 2]]) -> usize {
    polygon.len().saturating_sub(usize::from(
        polygon.len() > 1 && polygon.first() == polygon.last(),
    ))
}

fn polygon_json(layer: &ProjectMaskLayer, index: usize, polygon: &[[f32; 2]]) -> Value {
    json!({
        "index": index,
        "vertices_local": polygon,
        "vertices_world": polygon.iter().map(|point| [
            point[0] + layer.offset_world[0],
            point[1] + layer.offset_world[1],
        ]).collect::<Vec<_>>(),
    })
}

fn parse_projection_selection(
    value: Option<&Value>,
    layers: &[ProjectMaskLayer],
) -> Result<Option<MaskSelection>, ControlError> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let layer_id = value
        .get("layer_id")
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid("renderer mask selection has no layer_id"))?;
    let polygon_index = value
        .get("polygon_index")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| invalid("renderer mask selection has no polygon_index"))?;
    let layer = layers
        .iter()
        .find(|layer| layer.id == layer_id)
        .ok_or_else(|| invalid("renderer mask selection references an unknown layer"))?;
    let polygon = layer
        .polygons_world
        .get(polygon_index)
        .ok_or_else(|| invalid("renderer mask selection references an unknown polygon"))?;
    let vertex_index = value
        .get("vertex_index")
        .filter(|value| !value.is_null())
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|index| *index < unique_vertex_count(polygon))
                .ok_or_else(|| invalid("renderer mask selection references an unknown vertex"))
        })
        .transpose()?;
    Ok(Some(MaskSelection {
        layer_id,
        polygon_index,
        vertex_index,
    }))
}

fn validate_layers(layers: &[ProjectMaskLayer]) -> Result<(), ControlError> {
    let mut ids = std::collections::HashSet::new();
    for layer in layers {
        if layer.id == 0 || !ids.insert(layer.id) {
            return Err(invalid("mask layer IDs must be unique positive integers"));
        }
        if layer.name.trim().is_empty() {
            return Err(invalid(format!(
                "mask layer {} has an empty name",
                layer.id
            )));
        }
        if !layer.opacity.is_finite() || !(0.0..=1.0).contains(&layer.opacity) {
            return Err(invalid(format!(
                "mask layer {} opacity must be between 0 and 1",
                layer.id
            )));
        }
        if !layer.width_screen_px.is_finite() || layer.width_screen_px <= 0.0 {
            return Err(invalid(format!(
                "mask layer {} width_screen_px must be greater than zero",
                layer.id
            )));
        }
        if layer.offset_world.iter().any(|value| !value.is_finite()) {
            return Err(invalid(format!(
                "mask layer {} offset_world must be finite",
                layer.id
            )));
        }
        if layer.display_mode.as_deref().is_some_and(|mode| {
            !matches!(
                mode,
                "outline_only"
                    | "outline"
                    | "translucent_fill"
                    | "fill_outline"
                    | "semi_transparent_fill"
                    | "filled_preview"
                    | "mask_preview"
            )
        }) {
            return Err(invalid(format!(
                "mask layer {} has an unknown display_mode",
                layer.id
            )));
        }
        for polygon in &layer.polygons_world {
            if polygon.len() < 2
                || polygon
                    .iter()
                    .flatten()
                    .any(|coordinate| !coordinate.is_finite())
            {
                return Err(invalid(format!(
                    "mask layer {} contains invalid polygon geometry",
                    layer.id
                )));
            }
        }
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> ControlError {
    ControlError::new(ControlErrorKind::InvalidParams, message)
}

fn not_found(message: impl Into<String>) -> ControlError {
    ControlError::new(ControlErrorKind::ResourceNotFound, message)
}

pub(crate) fn load_geojson_mask_polylines(
    path: &Path,
    downsample_factor: f32,
) -> anyhow::Result<Vec<Vec<[f32; 2]>>> {
    if !path.exists() {
        anyhow::bail!("missing GeoJSON file: {}", path.to_string_lossy());
    }
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.to_string_lossy()))?;
    let root: Value = serde_json::from_str(&text).context("failed to parse GeoJSON")?;
    let features = root
        .get("features")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let scale = downsample_factor.max(1.0e-6);
    let mut polygons = Vec::new();
    for feature in features {
        let Some(geometry) = feature.get("geometry") else {
            continue;
        };
        let geometry_type = geometry
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        let coordinates = geometry.get("coordinates");
        match geometry_type.as_str() {
            "polygon" => {
                if let Some(rings) = coordinates.and_then(Value::as_array) {
                    polygons.extend(
                        rings
                            .iter()
                            .filter_map(|ring| parse_geojson_points(ring, scale, true)),
                    );
                }
            }
            "multipolygon" => {
                if let Some(values) = coordinates.and_then(Value::as_array) {
                    for polygon in values {
                        if let Some(rings) = polygon.as_array() {
                            polygons.extend(
                                rings
                                    .iter()
                                    .filter_map(|ring| parse_geojson_points(ring, scale, true)),
                            );
                        }
                    }
                }
            }
            "linestring" => {
                if let Some(points) =
                    coordinates.and_then(|value| parse_geojson_points(value, scale, false))
                {
                    polygons.push(points);
                }
            }
            "multilinestring" => {
                if let Some(lines) = coordinates.and_then(Value::as_array) {
                    polygons.extend(
                        lines
                            .iter()
                            .filter_map(|line| parse_geojson_points(line, scale, false)),
                    );
                }
            }
            _ => {}
        }
    }
    if polygons.is_empty() {
        anyhow::bail!("no supported shapes in GeoJSON");
    }
    Ok(polygons)
}

fn parse_geojson_points(node: &Value, scale: f32, close: bool) -> Option<Vec<[f32; 2]>> {
    let coordinates = node.as_array()?;
    let mut points = coordinates
        .iter()
        .filter_map(|coordinate| {
            let pair = coordinate.as_array()?;
            let x = pair.first()?.as_f64()? as f32 * scale;
            let y = pair.get(1)?.as_f64()? as f32 * scale;
            (x.is_finite() && y.is_finite()).then_some([x, y])
        })
        .collect::<Vec<_>>();
    if points.len() < 2 {
        return None;
    }
    if points.first() == points.last() {
        points.pop();
    }
    if close && let Some(first) = points.first().copied() {
        points.push(first);
    }
    Some(points)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_crud_selection_and_undo_are_renderer_independent() {
        let mut masks = MaskModel::default();
        let layer = masks
            .dispatch("viewer.masks.layers.create", &json!({"name":"Cells"}))
            .unwrap();
        let id = layer["id"].as_u64().unwrap();
        masks
            .dispatch(
                "viewer.masks.polygons.add",
                &json!({"id":id,"vertices":[[1,2],[4,2],[4,5]]}),
            )
            .unwrap();
        let polygons = masks
            .dispatch("viewer.masks.polygons.list", &json!({"id":id}))
            .unwrap();
        assert_eq!(
            polygons["polygons"][0]["vertices_local"]
                .as_array()
                .unwrap()
                .len(),
            4
        );
        let selected = masks
            .dispatch(
                "viewer.masks.selection.set",
                &json!({"id":id,"index":0,"vertex_index":1}),
            )
            .unwrap();
        assert_eq!(selected["selection"]["vertex_index"], 1);
        assert_eq!(
            masks.dispatch("viewer.masks.undo", &json!({})).unwrap()["undone"],
            true
        );
        assert_eq!(
            masks
                .dispatch("viewer.masks.polygons.list", &json!({"id":id}))
                .unwrap()["total"],
            0
        );
    }

    #[test]
    fn atomic_replacement_rejects_a_stale_native_generation() {
        let mut masks = MaskModel::default();
        masks
            .dispatch("viewer.masks.layers.create", &json!({"name":"Python"}))
            .unwrap();
        let error = masks
            .dispatch(
                "viewer.masks.state.replace",
                &json!({
                    "expected_generation":1,
                    "state":{"layers":[],"active_layer_id":null,"selection":null},
                }),
            )
            .unwrap_err();
        assert_eq!(error.kind, ControlErrorKind::Conflict);
        assert_eq!(masks.projection_json()["layers"][0]["name"], "Python");
    }
}
