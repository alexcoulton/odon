use super::super::*;

impl OmeZarrViewerApp {
    pub fn control_mask_projection_snapshot(&self) -> serde_json::Value {
        let active_layer_id = match self.active_layer {
            LayerId::Mask(id) => Some(id),
            _ => None,
        };
        serde_json::json!({
            "generation": self.control_actor_mask_generation.max(1),
            "active_layer_id": active_layer_id,
            "layers": self.mask_layers.iter().map(MaskLayer::to_project).collect::<Vec<_>>(),
            "selection": self.control_get_mask_selection()["selection"].clone(),
            "dirty": self.mask_layers_project_dirty,
            "undo_available": if self.control_actor_mask_generation > 0 {
                self.control_actor_mask_undo_available
            } else {
                !self.undo_stack.is_empty()
            },
        })
    }

    pub fn control_get_mask_selection(&self) -> serde_json::Value {
        let selection = self.selected_mask_polygon.and_then(|selection| {
            let layer = self
                .mask_layers
                .iter()
                .find(|layer| layer.id == selection.layer_id)?;
            let polygon = layer.polygons_world.get(selection.polygon_idx)?;
            Some(serde_json::json!({
                "layer_id": selection.layer_id,
                "polygon_index": selection.polygon_idx,
                "vertex_index": self.selected_mask_vertex,
                "vertices_local": polygon.iter().map(|point| [point.x, point.y]).collect::<Vec<_>>(),
                "vertices_world": polygon.iter().map(|point| [point.x + layer.offset_world.x, point.y + layer.offset_world.y]).collect::<Vec<_>>(),
            }))
        });
        serde_json::json!({"selection": selection})
    }
}
