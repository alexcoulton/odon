use serde_json::{Value, json};

use crate::control::ControlError;

use super::validation::{invalid, parse_projection_selection};
use super::*;

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

    pub(crate) fn appendable_layers(&self) -> Vec<ProjectMaskLayer> {
        self.layers
            .iter()
            .filter(|layer| {
                layer.editable
                    && layer.source_geojson.is_none()
                    && layer
                        .polygons_world
                        .iter()
                        .any(|polygon| polygon.len() >= 3)
            })
            .cloned()
            .collect()
    }

    pub(crate) fn reconcile_appended_file(
        &mut self,
        saved_layers: &[ProjectMaskLayer],
        name: String,
        polygons_world: Vec<Vec<[f32; 2]>>,
        source_geojson: std::path::PathBuf,
    ) -> Value {
        let mut cleared_polygon_count = 0usize;
        for saved in saved_layers {
            let Some(current) = self.layers.iter_mut().find(|layer| layer.id == saved.id) else {
                continue;
            };
            for saved_polygon in &saved.polygons_world {
                if let Some(index) = current
                    .polygons_world
                    .iter()
                    .position(|polygon| polygon == saved_polygon)
                {
                    current.polygons_world.remove(index);
                    cleared_polygon_count = cleared_polygon_count.saturating_add(1);
                }
            }
        }

        let source_index = self.layers.iter().position(|layer| {
            !layer.editable
                && layer
                    .source_geojson
                    .as_ref()
                    .is_some_and(|path| path == &source_geojson)
        });
        let source_layer_id = if let Some(index) = source_index {
            let layer = &mut self.layers[index];
            layer.polygons_world = polygons_world;
            layer.source_geojson = Some(source_geojson.clone());
            layer.visible = true;
            layer.editable = false;
            layer.id
        } else {
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
                editable: false,
                polygons_world,
                source_geojson: Some(source_geojson.clone()),
            });
            id
        };
        self.selection = self.selection.filter(|selection| {
            !saved_layers
                .iter()
                .any(|layer| layer.id == selection.layer_id)
                && selection.layer_id != source_layer_id
                && self
                    .layers
                    .iter()
                    .find(|layer| layer.id == selection.layer_id)
                    .and_then(|layer| layer.polygons_world.get(selection.polygon_index))
                    .is_some()
        });
        self.commit();
        json!({
            "source_layer_id": source_layer_id,
            "source_polygon_count": self
                .layers
                .iter()
                .find(|layer| layer.id == source_layer_id)
                .map_or(0, |layer| layer.polygons_world.len()),
            "cleared_polygon_count": cleared_polygon_count,
        })
    }

    pub(crate) fn install_imported_layer(
        &mut self,
        name: String,
        editable: bool,
        replace_layer_id: Option<u64>,
        polygons_world: Vec<Vec<[f32; 2]>>,
        source_geojson: std::path::PathBuf,
    ) -> Option<Value> {
        self.push_undo();
        if let Some(id) = replace_layer_id {
            let index = self.layers.iter().position(|layer| layer.id == id)?;
            let polygon_count = polygons_world.len();
            let layer = &mut self.layers[index];
            layer.polygons_world = polygons_world;
            layer.source_geojson = Some(source_geojson.clone());
            layer.editable = editable;
            self.selection = self.selection.filter(|selection| selection.layer_id != id);
            self.commit();
            let mut response = self.layer_json(&self.layers[index]);
            response["reloaded"] = json!(true);
            response["path"] = json!(source_geojson.to_string_lossy());
            response["polygon_count"] = json!(polygon_count);
            return Some(response);
        }
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
        Some(json!({
            "imported": true,
            "path": source_geojson.to_string_lossy(),
            "layer_id": id,
            "polygon_count": polygon_count,
            "layer": self.layer_json(self.layers.last().expect("imported mask layer exists")),
        }))
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
}
