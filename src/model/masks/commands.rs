use serde_json::{Value, json};

use crate::control::{ControlError, ControlErrorKind};

use super::validation::*;
use super::*;

impl MaskModel {
    pub(crate) fn dispatch(&mut self, method: &str, params: &Value) -> Result<Value, ControlError> {
        if matches!(
            method,
            "viewer.masks.layers.create"
                | "viewer.masks.layers.update"
                | "viewer.masks.layers.delete"
                | "viewer.masks.polygons.add"
                | "viewer.masks.polygons.update"
                | "viewer.masks.polygons.remove"
                | "viewer.masks.selection.set"
                | "viewer.masks.selection.clear"
                | "viewer.masks.undo"
        ) {
            self.validate_expected_generation(params)?;
        }
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
        let active = optional_bool(params, "active")?;
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
        if let Some(true) = active {
            self.active_layer_id = Some(id);
        } else if active == Some(false) && self.active_layer_id == Some(id) {
            self.active_layer_id = None;
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
        self.validate_expected_generation(params)?;
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

    fn validate_expected_generation(&self, params: &Value) -> Result<(), ControlError> {
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
        Ok(())
    }

    pub(super) fn layer_json(&self, layer: &ProjectMaskLayer) -> Value {
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

    pub(super) fn selection_json(&self) -> Value {
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

    pub(super) fn layer(&self, id: u64) -> Result<&ProjectMaskLayer, ControlError> {
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

    pub(super) fn push_undo(&mut self) {
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

    pub(super) fn commit(&mut self) {
        self.dirty = true;
        self.generation = self.generation.wrapping_add(1).max(1);
    }
}
