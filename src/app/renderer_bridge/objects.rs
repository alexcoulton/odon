use super::super::*;

impl OmeZarrViewerApp {
    pub fn control_object_selection_signature(&self) -> serde_json::Value {
        match self.control_object_selection_target(&serde_json::json!({})) {
            Ok(LayerId::SegmentationObjects) => serde_json::json!({
                "target": "segmentation_objects",
                "selection": self.seg_objects.selection_signature_json(),
            }),
            Ok(LayerId::SpatialShape(id)) => self
                .spatial_layers
                .shapes
                .iter()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer())
                .map(|objects| {
                    serde_json::json!({
                        "target": "spatial_shape", "layer_id": id,
                        "selection": objects.selection_signature_json(),
                    })
                })
                .unwrap_or(serde_json::Value::Null),
            _ => serde_json::Value::Null,
        }
    }
}
