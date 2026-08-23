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

#[cfg(test)]
impl OmeZarrViewerApp {
    pub fn control_get_object_overlay_visibility(
        &self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let target = params
            .get("target")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("objects");
        serde_json::json!({
            "target": target,
            "segmentation_labels": self.cells_outlines_visible,
            "segmentation_geojson": self.seg_geojson.visible,
            "segmentation_objects": self.seg_objects.visible,
            "object_count": self.seg_objects.object_count(),
        })
    }

    pub fn control_set_object_overlay_visibility(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let Some(visible) = params.get("visible").and_then(serde_json::Value::as_bool) else {
            return serde_json::json!({"error": "set_object_overlay_visibility requires visible"});
        };
        let target = params
            .get("target")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("objects");
        match target {
            "objects" => self.seg_objects.visible = visible,
            "labels" => self.cells_outlines_visible = visible,
            "geojson" => self.seg_geojson.visible = visible,
            "all" => {
                self.seg_objects.visible = visible;
                self.cells_outlines_visible = visible;
                self.seg_geojson.visible = visible;
            }
            other => {
                return serde_json::json!({"error": format!("unknown overlay target '{other}'")});
            }
        }
        self.bump_render_id();
        self.control_get_object_overlay_visibility(params)
    }

    pub fn control_select_object_ids(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(values) = params.get("ids").and_then(serde_json::Value::as_array) else {
            return serde_json::json!({"error": "ids is required"});
        };
        let ids = values
            .iter()
            .filter_map(serde_json::Value::as_str)
            .map(str::to_string)
            .collect::<HashSet<_>>();
        if ids.len() != values.len() {
            return serde_json::json!({"error": "ids must contain unique strings"});
        }
        let mode = params
            .get("mode")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("replace");
        let limit = control_object_debug_limit(params);
        let result = match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => self.seg_objects.control_select_ids_json(
                &ids,
                mode,
                self.seg_objects_offset_world,
                limit,
            ),
            Ok(LayerId::SpatialShape(id)) => {
                let Some(layer) = self
                    .spatial_layers
                    .shapes
                    .iter_mut()
                    .find(|layer| layer.id == id)
                else {
                    return serde_json::json!({"error": format!("spatial shape layer {id} not found")});
                };
                let offset = layer.offset_world;
                let Some(objects) = layer.object_layer_mut() else {
                    return serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")});
                };
                objects.control_select_ids_json(&ids, mode, offset, limit)
            }
            Ok(_) => serde_json::json!({"error": "active layer does not support object selection"}),
            Err(error) => serde_json::json!({"error": error}),
        };
        self.bump_render_id();
        result
    }

    pub fn control_select_filtered_objects(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        self.control_filter_sensitive_operation(
            params,
            OmeZarrViewerApp::control_select_filtered_objects_current,
        )
    }

    pub(in crate::app) fn control_select_filtered_objects_current(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let mode = params
            .get("mode")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("replace");
        let limit = control_object_debug_limit(params);
        let result = match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => self.seg_objects.control_select_filtered_json(
                mode,
                self.seg_objects_offset_world,
                limit,
            ),
            Ok(LayerId::SpatialShape(id)) => {
                let Some(layer) = self
                    .spatial_layers
                    .shapes
                    .iter_mut()
                    .find(|layer| layer.id == id)
                else {
                    return serde_json::json!({"error": format!("spatial shape layer {id} not found")});
                };
                let offset = layer.offset_world;
                let Some(objects) = layer.object_layer_mut() else {
                    return serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")});
                };
                objects.control_select_filtered_json(mode, offset, limit)
            }
            Ok(_) => serde_json::json!({"error": "active layer does not support object selection"}),
            Err(error) => serde_json::json!({"error": error}),
        };
        self.bump_render_id();
        result
    }

    pub fn control_set_object_filter_query(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let Some(query) = params
            .get("query")
            .or_else(|| params.get("expression"))
            .and_then(serde_json::Value::as_str)
        else {
            return serde_json::json!({"error": "set_object_filter_query requires query"});
        };
        let result = match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => {
                self.seg_objects.set_filter_query_from_text(query);
                serde_json::json!({
                    "target": "segmentation_objects",
                    "filter": self.seg_objects.filter_snapshot_json(),
                })
            }
            Ok(LayerId::SpatialShape(id)) => {
                let Some(layer) = self
                    .spatial_layers
                    .shapes
                    .iter_mut()
                    .find(|layer| layer.id == id)
                else {
                    return serde_json::json!({"error": format!("spatial shape layer {id} not found")});
                };
                let layer_name = layer.name.clone();
                let Some(objects) = layer.object_layer_mut() else {
                    return serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")});
                };
                objects.set_filter_query_from_text(query);
                serde_json::json!({
                    "target": "spatial_shape",
                    "layer_id": id,
                    "layer_name": layer_name,
                    "filter": objects.filter_snapshot_json(),
                })
            }
            Ok(_) => serde_json::json!({"error": "active layer does not support object filters"}),
            Err(error) => serde_json::json!({"error": error}),
        };
        self.bump_render_id();
        result
    }
}
