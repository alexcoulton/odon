use super::super::*;

impl OmeZarrViewerApp {
    pub fn control_get_object_analysis(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let channels = self.channels.clone();
        let selected_channel = self.selected_channel;
        match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => self
                .seg_objects
                .control_analysis_state_json(&channels, selected_channel),
            Ok(LayerId::SpatialShape(id)) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer_mut())
                .map(|objects| objects.control_analysis_state_json(&channels, selected_channel))
                .unwrap_or_else(|| serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")})),
            Ok(_) => serde_json::json!({"error": "analysis requires an object-backed layer"}),
            Err(error) => serde_json::json!({"error": error}),
        }
    }

    pub fn control_set_object_analysis(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let channels = self.channels.clone();
        let selected_channel = self.selected_channel;
        match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => self.seg_objects.control_set_analysis_state_json(
                params,
                &channels,
                selected_channel,
            ),
            Ok(LayerId::SpatialShape(id)) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer_mut())
                .map(|objects| {
                    objects.control_set_analysis_state_json(params, &channels, selected_channel)
                })
                .unwrap_or_else(|| serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")})),
            Ok(_) => serde_json::json!({"error": "analysis requires an object-backed layer"}),
            Err(error) => serde_json::json!({"error": error}),
        }
    }

    pub fn control_object_histogram(&mut self, params: &serde_json::Value) -> serde_json::Value {
        self.control_filter_sensitive_operation(
            params,
            OmeZarrViewerApp::control_object_histogram_current,
        )
    }

    pub(in crate::app) fn control_object_histogram_current(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => self.seg_objects.control_histogram_json(params),
            Ok(LayerId::SpatialShape(id)) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer_mut())
                .map(|objects| objects.control_histogram_json(params))
                .unwrap_or_else(|| serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")})),
            Ok(_) => serde_json::json!({"error": "analysis requires an object-backed layer"}),
            Err(error) => serde_json::json!({"error": error}),
        }
    }

    pub fn control_object_threshold_suggestions(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        self.control_filter_sensitive_operation(
            params,
            OmeZarrViewerApp::control_object_threshold_suggestions_current,
        )
    }

    pub(in crate::app) fn control_object_threshold_suggestions_current(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => self
                .seg_objects
                .control_threshold_suggestions_json(params),
            Ok(LayerId::SpatialShape(id)) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer_mut())
                .map(|objects| objects.control_threshold_suggestions_json(params))
                .unwrap_or_else(|| serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")})),
            Ok(_) => serde_json::json!({"error": "analysis requires an object-backed layer"}),
            Err(error) => serde_json::json!({"error": error}),
        }
    }

    pub fn control_get_analysis_warmup(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => self.seg_objects.control_analysis_warmup_json(),
            Ok(LayerId::SpatialShape(id)) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer_mut())
                .map(|objects| objects.control_analysis_warmup_json())
                .unwrap_or_else(|| serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")})),
            Ok(_) => serde_json::json!({"error": "analysis requires an object-backed layer"}),
            Err(error) => serde_json::json!({"error": error}),
        }
    }

    pub fn control_start_analysis_warmup(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let channels = self.channels.clone();
        let selected_channel = self.selected_channel;
        match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => self
                .seg_objects
                .control_start_analysis_warmup_json(&channels, selected_channel),
            Ok(LayerId::SpatialShape(id)) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer_mut())
                .map(|objects| {
                    objects.control_start_analysis_warmup_json(&channels, selected_channel)
                })
                .unwrap_or_else(|| serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")})),
            Ok(_) => serde_json::json!({"error": "analysis requires an object-backed layer"}),
            Err(error) => serde_json::json!({"error": error}),
        }
    }

    pub fn control_export_analysis_preset(
        &mut self,
        params: &serde_json::Value,
        path: &Path,
    ) -> serde_json::Value {
        let overwrite = params
            .get("overwrite")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => self
                .seg_objects
                .control_export_call_preset_json(path, overwrite),
            Ok(LayerId::SpatialShape(id)) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer_mut())
                .map(|objects| objects.control_export_call_preset_json(path, overwrite))
                .unwrap_or_else(|| serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")})),
            Ok(_) => serde_json::json!({"error": "analysis requires an object-backed layer"}),
            Err(error) => serde_json::json!({"error": error}),
        }
    }

    pub fn control_import_analysis_preset(
        &mut self,
        params: &serde_json::Value,
        path: &Path,
    ) -> serde_json::Value {
        let active_channel_name = self
            .channels
            .get(self.selected_channel)
            .map(|channel| channel.name.clone());
        match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => self.seg_objects.control_import_call_preset_json(
                path,
                active_channel_name.as_deref(),
            ),
            Ok(LayerId::SpatialShape(id)) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer_mut())
                .map(|objects| {
                    objects.control_import_call_preset_json(
                        path,
                        active_channel_name.as_deref(),
                    )
                })
                .unwrap_or_else(|| serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")})),
            Ok(_) => serde_json::json!({"error": "analysis requires an object-backed layer"}),
            Err(error) => serde_json::json!({"error": error}),
        }
    }

    pub fn control_get_measurement_state(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => self
                .seg_objects
                .control_measurement_state_json(&self.dataset),
            Ok(LayerId::SpatialShape(id)) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer_mut())
                .map(|objects| objects.control_measurement_state_json(&self.dataset))
                .unwrap_or_else(|| serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")})),
            Ok(_) => serde_json::json!({"error": "measurements require an object-backed layer"}),
            Err(error) => serde_json::json!({"error": error}),
        }
    }

    pub fn control_configure_measurement(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => self
                .seg_objects
                .control_configure_measurement_json(params, &self.dataset),
            Ok(LayerId::SpatialShape(id)) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer_mut())
                .map(|objects| objects.control_configure_measurement_json(params, &self.dataset))
                .unwrap_or_else(|| serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")})),
            Ok(_) => serde_json::json!({"error": "measurements require an object-backed layer"}),
            Err(error) => serde_json::json!({"error": error}),
        }
    }

    pub fn control_start_measurement(&mut self, params: &serde_json::Value) -> serde_json::Value {
        self.control_filter_sensitive_operation(
            params,
            OmeZarrViewerApp::control_start_measurement_current,
        )
    }

    pub(in crate::app) fn control_start_measurement_current(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let store = self.store.clone();
        let channels = self.channels.clone();
        match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => self.seg_objects.control_start_measurement_json(
                params,
                &self.dataset,
                store,
                &channels,
                self.seg_objects_offset_world,
            ),
            Ok(LayerId::SpatialShape(id)) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|layer| layer.id == id)
                .map(|layer| {
                    let offset = layer.offset_world;
                    layer
                        .object_layer_mut()
                        .map(|objects| {
                            objects.control_start_measurement_json(
                                params,
                                &self.dataset,
                                store,
                                &channels,
                                offset,
                            )
                        })
                        .unwrap_or_else(|| serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")}))
                })
                .unwrap_or_else(|| serde_json::json!({"error": format!("spatial shape layer {id} not found")})),
            Ok(_) => serde_json::json!({"error": "measurements require an object-backed layer"}),
            Err(error) => serde_json::json!({"error": error}),
        }
    }

    pub fn control_cancel_measurement(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => self.seg_objects.control_cancel_measurement_json(),
            Ok(LayerId::SpatialShape(id)) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer_mut())
                .map(|objects| objects.control_cancel_measurement_json())
                .unwrap_or_else(|| serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")})),
            Ok(_) => serde_json::json!({"error": "measurements require an object-backed layer"}),
            Err(error) => serde_json::json!({"error": error}),
        }
    }

    pub fn control_get_object_export_columns(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => self.seg_objects.control_export_columns_json(),
            Ok(LayerId::SpatialShape(id)) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer_mut())
                .map(|objects| objects.control_export_columns_json())
                .unwrap_or_else(|| serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")})),
            Ok(_) => serde_json::json!({"error": "object export requires an object-backed layer"}),
            Err(error) => serde_json::json!({"error": error}),
        }
    }

    pub fn control_get_object_export_state(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => self.seg_objects.control_export_state_json(),
            Ok(LayerId::SpatialShape(id)) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer_mut())
                .map(|objects| objects.control_export_state_json())
                .unwrap_or_else(|| serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")})),
            Ok(_) => serde_json::json!({"error": "object export requires an object-backed layer"}),
            Err(error) => serde_json::json!({"error": error}),
        }
    }

    pub fn control_start_object_export(
        &mut self,
        params: &serde_json::Value,
        path: PathBuf,
    ) -> serde_json::Value {
        let scope = params
            .get("scope")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("all");
        if scope == "filtered" {
            return self.control_filter_sensitive_operation(params, move |app, params| {
                app.control_start_object_export_current(params, path)
            });
        }
        self.control_start_object_export_current(params, path)
    }

    pub(in crate::app) fn control_start_object_export_current(
        &mut self,
        params: &serde_json::Value,
        path: PathBuf,
    ) -> serde_json::Value {
        match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => self
                .seg_objects
                .control_start_object_export_json(params, path),
            Ok(LayerId::SpatialShape(id)) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer_mut())
                .map(|objects| objects.control_start_object_export_json(params, path))
                .unwrap_or_else(|| serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")})),
            Ok(_) => serde_json::json!({"error": "object export requires an object-backed layer"}),
            Err(error) => serde_json::json!({"error": error}),
        }
    }
}
