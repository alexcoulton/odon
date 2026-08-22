use super::super::*;

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

    pub fn control_get_object_state(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match params
            .get("target")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("objects")
        {
            "objects" | "segmentation_objects" => serde_json::json!({
                "target": "segmentation_objects",
                "state": self.seg_objects.control_state_snapshot_json(),
            }),
            "spatial_shape" => {
                let Some(id) = params
                    .get("layer_id")
                    .or_else(|| params.get("id"))
                    .and_then(serde_json::Value::as_u64)
                else {
                    return serde_json::json!({"error": "target='spatial_shape' requires layer_id"});
                };
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
                serde_json::json!({
                    "target": "spatial_shape",
                    "layer_id": id,
                    "layer_name": layer_name,
                    "state": objects.control_state_snapshot_json(),
                })
            }
            target => serde_json::json!({"error": format!("unknown object target '{target}'")}),
        }
    }

    pub fn control_load_object_source(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(path) = params
            .get("path")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|path| !path.is_empty())
        else {
            return serde_json::json!({"error": "path is required"});
        };
        let path = PathBuf::from(path);
        if !path.exists() || !path.is_file() {
            return serde_json::json!({"error": format!("object source does not exist: {}", path.to_string_lossy())});
        }
        let downsample_factor = params
            .get("downsample_factor")
            .and_then(serde_json::Value::as_f64)
            .map(|value| value as f32)
            .unwrap_or(self.seg_objects.downsample_factor)
            .max(1e-6);
        self.seg_objects.load_path(path.clone(), downsample_factor);
        serde_json::json!({
            "queued": true,
            "path": path.to_string_lossy(),
            "downsample_factor": downsample_factor,
        })
    }

    pub fn control_reload_object_source(&mut self) -> serde_json::Value {
        let Some(path) = self.seg_objects.loaded_geojson.clone() else {
            return serde_json::json!({"error": "No object source is loaded."});
        };
        let downsample_factor = self.seg_objects.downsample_factor;
        self.seg_objects.load_path(path.clone(), downsample_factor);
        serde_json::json!({
            "queued": true,
            "path": path.to_string_lossy(),
            "downsample_factor": downsample_factor,
        })
    }

    pub fn control_clear_object_source(&mut self) -> serde_json::Value {
        let previous_path = self
            .seg_objects
            .loaded_geojson
            .as_ref()
            .map(|path| path.to_string_lossy().into_owned());
        let previous_count = self.seg_objects.object_count();
        self.seg_objects.clear();
        self.bump_render_id();
        serde_json::json!({
            "cleared": previous_path.is_some() || previous_count > 0,
            "previous_path": previous_path,
            "previous_count": previous_count,
        })
    }

    pub fn control_cancel_object_source_load(&mut self) -> serde_json::Value {
        serde_json::json!({
            "cancelled": self.seg_objects.cancel_load(),
            "state": self.seg_objects.control_state_snapshot_json(),
        })
    }

    pub fn control_get_object_style(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => self.seg_objects.control_style_snapshot_json(),
            Ok(LayerId::SpatialShape(id)) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer_mut())
                .map(ObjectsLayer::control_style_snapshot_json)
                .unwrap_or_else(|| serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")})),
            Ok(_) => serde_json::json!({"error": "active layer has no object style"}),
            Err(error) => serde_json::json!({"error": error}),
        }
    }

    pub fn control_set_object_style(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let result = match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => self.seg_objects.control_set_style_json(params),
            Ok(LayerId::SpatialShape(id)) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer_mut())
                .map(|objects| objects.control_set_style_json(params))
                .unwrap_or_else(|| serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")})),
            Ok(_) => serde_json::json!({"error": "active layer has no object style"}),
            Err(error) => serde_json::json!({"error": error}),
        };
        self.bump_render_id();
        result
    }

    pub fn control_set_object_legend(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let result = match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => self.seg_objects.control_set_legend_json(params),
            Ok(LayerId::SpatialShape(id)) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer_mut())
                .map(|objects| objects.control_set_legend_json(params))
                .unwrap_or_else(|| serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")})),
            Ok(_) => serde_json::json!({"error": "active layer has no object legend"}),
            Err(error) => serde_json::json!({"error": error}),
        };
        self.bump_render_id();
        result
    }

    pub fn control_get_fast_object_rendering(
        &self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => serde_json::json!({"enabled": self.seg_objects.fast_rendering}),
            Ok(LayerId::SpatialShape(id)) => self
                .spatial_layers
                .shapes
                .iter()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer())
                .map(|objects| serde_json::json!({"enabled": objects.fast_rendering}))
                .unwrap_or_else(|| serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")})),
            Ok(_) => serde_json::json!({"error": "active layer has no object rendering settings"}),
            Err(error) => serde_json::json!({"error": error}),
        }
    }

    pub fn control_set_fast_object_rendering(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let Some(enabled) = params.get("enabled").and_then(serde_json::Value::as_bool) else {
            return serde_json::json!({"error": "enabled is required"});
        };
        let result = match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => self.seg_objects.control_set_fast_rendering_json(enabled),
            Ok(LayerId::SpatialShape(id)) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer_mut())
                .map(|objects| objects.control_set_fast_rendering_json(enabled))
                .unwrap_or_else(|| serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")})),
            Ok(_) => serde_json::json!({"error": "active layer has no object rendering settings"}),
            Err(error) => serde_json::json!({"error": error}),
        };
        self.bump_render_id();
        result
    }

    pub fn control_list_object_properties(&self, params: &serde_json::Value) -> serde_json::Value {
        let offset = params
            .get("offset")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize;
        let limit = params
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(200) as usize;
        match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => self
                .seg_objects
                .control_property_schema_json(offset, limit),
            Ok(LayerId::SpatialShape(id)) => self
                .spatial_layers
                .shapes
                .iter()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer())
                .map(|objects| objects.control_property_schema_json(offset, limit))
                .unwrap_or_else(|| serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")})),
            Ok(_) => serde_json::json!({"error": "active layer has no object properties"}),
            Err(error) => serde_json::json!({"error": error}),
        }
    }

    pub fn control_load_object_property(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let Some(property) = params
            .get("property")
            .or_else(|| params.get("name"))
            .and_then(serde_json::Value::as_str)
        else {
            return serde_json::json!({"error": "property is required"});
        };
        match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => serde_json::json!({
                "target": "segmentation_objects",
                "result": self.seg_objects.control_request_property_load(property),
            }),
            Ok(LayerId::SpatialShape(id)) => {
                let result = self
                    .spatial_layers
                    .shapes
                    .iter_mut()
                    .find(|layer| layer.id == id)
                    .and_then(|layer| layer.object_layer_mut())
                    .map(|objects| objects.control_request_property_load(property))
                    .unwrap_or_else(|| serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")}));
                serde_json::json!({
                    "target": "spatial_shape",
                    "layer_id": id,
                    "result": result,
                })
            }
            Ok(_) => serde_json::json!({"error": "active layer has no object properties"}),
            Err(error) => serde_json::json!({"error": error}),
        }
    }

    pub fn control_get_object_property_values(
        &self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let Some(property) = params
            .get("property")
            .or_else(|| params.get("name"))
            .and_then(serde_json::Value::as_str)
        else {
            return serde_json::json!({"error": "property is required"});
        };
        let offset = params
            .get("offset")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize;
        let limit = params
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(200) as usize;
        match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => self
                .seg_objects
                .control_property_values_json(property, offset, limit),
            Ok(LayerId::SpatialShape(id)) => self
                .spatial_layers
                .shapes
                .iter()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer())
                .map(|objects| objects.control_property_values_json(property, offset, limit))
                .unwrap_or_else(|| serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")})),
            Ok(_) => serde_json::json!({"error": "active layer has no object properties"}),
            Err(error) => serde_json::json!({"error": error}),
        }
    }

    pub fn control_get_object_selection(&self, params: &serde_json::Value) -> serde_json::Value {
        let limit = control_object_debug_limit(params);
        match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => serde_json::json!({
                "target": "segmentation_objects",
                "selection": self
                    .seg_objects
                    .selection_snapshot_json(self.seg_objects_offset_world, limit),
            }),
            Ok(LayerId::SpatialShape(id)) => {
                let Some(layer) = self
                    .spatial_layers
                    .shapes
                    .iter()
                    .find(|layer| layer.id == id)
                else {
                    return serde_json::json!({"error": format!("spatial shape layer {id} not found")});
                };
                let Some(objects) = layer.object_layer() else {
                    return serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")});
                };
                serde_json::json!({
                    "target": "spatial_shape",
                    "layer_id": id,
                    "layer_name": layer.name.as_str(),
                    "selection": objects.selection_snapshot_json(layer.offset_world, limit),
                })
            }
            Ok(_) => serde_json::json!({"error": "active layer does not support object selection"}),
            Err(error) => serde_json::json!({"error": error}),
        }
    }

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

    pub fn control_query_object_ids_in_rect(
        &self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let limit = control_object_debug_limit(params);
        let world_rect = match self.control_world_rect_from_params(params) {
            Ok(rect) => rect,
            Err(error) => return serde_json::json!({"error": error}),
        };
        match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => serde_json::json!({
                "target": "segmentation_objects",
                "query": self.seg_objects.query_world_rect_snapshot_json(
                    world_rect,
                    self.seg_objects_offset_world,
                    limit,
                ),
                "selection": self
                    .seg_objects
                    .selection_snapshot_json(self.seg_objects_offset_world, limit),
            }),
            Ok(LayerId::SpatialShape(id)) => {
                let Some(layer) = self
                    .spatial_layers
                    .shapes
                    .iter()
                    .find(|layer| layer.id == id)
                else {
                    return serde_json::json!({"error": format!("spatial shape layer {id} not found")});
                };
                let Some(objects) = layer.object_layer() else {
                    return serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")});
                };
                serde_json::json!({
                    "target": "spatial_shape",
                    "layer_id": id,
                    "layer_name": layer.name.as_str(),
                    "query": objects.query_world_rect_snapshot_json(
                        world_rect,
                        layer.offset_world,
                        limit,
                    ),
                    "selection": objects.selection_snapshot_json(layer.offset_world, limit),
                })
            }
            Ok(_) => serde_json::json!({"error": "active layer does not support object selection"}),
            Err(error) => serde_json::json!({"error": error}),
        }
    }

    pub fn control_query_object_ids_in_view(
        &self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let Some(viewport) = self.last_canvas_rect else {
            return serde_json::json!({"error": "No canvas viewport is available yet."});
        };
        let world_min = self.camera.screen_to_world(viewport.left_top(), viewport);
        let world_max = self
            .camera
            .screen_to_world(viewport.right_bottom(), viewport);
        let params = match params.as_object() {
            Some(obj) => {
                let mut obj = obj.clone();
                obj.insert(
                    "world_rect".to_string(),
                    serde_json::json!([world_min.x, world_min.y, world_max.x, world_max.y]),
                );
                serde_json::Value::Object(obj)
            }
            None => serde_json::json!({
                "world_rect": [world_min.x, world_min.y, world_max.x, world_max.y],
            }),
        };
        self.control_query_object_ids_in_rect(&params)
    }

    pub fn control_select_object_ids_in_rect(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let limit = control_object_debug_limit(params);
        let mode = params
            .get("mode")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_else(|| {
                if params
                    .get("additive")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
                {
                    "add"
                } else {
                    "replace"
                }
            });
        let world_rect = match self.control_world_rect_from_params(params) {
            Ok(rect) => rect,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let result = match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => serde_json::json!({
                "target": "segmentation_objects",
                "result": self.seg_objects.select_in_world_rect_snapshot_json_mode(
                    world_rect,
                    self.seg_objects_offset_world,
                    mode,
                    limit,
                ),
            }),
            Ok(LayerId::SpatialShape(id)) => {
                let Some(layer) = self
                    .spatial_layers
                    .shapes
                    .iter_mut()
                    .find(|layer| layer.id == id)
                else {
                    return serde_json::json!({"error": format!("spatial shape layer {id} not found")});
                };
                let offset_world = layer.offset_world;
                let layer_name = layer.name.clone();
                let Some(objects) = layer.object_layer_mut() else {
                    return serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")});
                };
                serde_json::json!({
                    "target": "spatial_shape",
                    "layer_id": id,
                    "layer_name": layer_name,
                    "result": objects.select_in_world_rect_snapshot_json_mode(
                        world_rect,
                        offset_world,
                        mode,
                        limit,
                    ),
                })
            }
            Ok(_) => serde_json::json!({"error": "active layer does not support object selection"}),
            Err(error) => serde_json::json!({"error": error}),
        };
        self.bump_render_id();
        result
    }

    pub fn control_query_object_ids_in_lasso(
        &self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let points = match self.control_world_points_from_params(params) {
            Ok(points) => points,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let limit = control_object_debug_limit(params);
        match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => self.seg_objects.query_world_lasso_snapshot_json(
                &points,
                self.seg_objects_offset_world,
                limit,
            ),
            Ok(LayerId::SpatialShape(id)) => {
                let Some(layer) = self
                    .spatial_layers
                    .shapes
                    .iter()
                    .find(|layer| layer.id == id)
                else {
                    return serde_json::json!({"error": format!("spatial shape layer {id} not found")});
                };
                let Some(objects) = layer.object_layer() else {
                    return serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")});
                };
                objects.query_world_lasso_snapshot_json(&points, layer.offset_world, limit)
            }
            Ok(_) => serde_json::json!({"error": "active layer does not support object selection"}),
            Err(error) => serde_json::json!({"error": error}),
        }
    }

    pub fn control_select_object_ids_in_lasso(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let points = match self.control_world_points_from_params(params) {
            Ok(points) => points,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let limit = control_object_debug_limit(params);
        let mode = params
            .get("mode")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("replace");
        let result = match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => {
                self.seg_objects.select_in_world_lasso_snapshot_json_mode(
                    &points,
                    self.seg_objects_offset_world,
                    mode,
                    limit,
                )
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
                let offset = layer.offset_world;
                let Some(objects) = layer.object_layer_mut() else {
                    return serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")});
                };
                objects.select_in_world_lasso_snapshot_json_mode(&points, offset, mode, limit)
            }
            Ok(_) => serde_json::json!({"error": "active layer does not support object selection"}),
            Err(error) => serde_json::json!({"error": error}),
        };
        self.bump_render_id();
        result
    }

    pub fn control_clear_object_selection(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let limit = control_object_debug_limit(params);
        let result = match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => {
                self.seg_objects.clear_selection();
                serde_json::json!({
                    "target": "segmentation_objects",
                    "selection": self
                        .seg_objects
                        .selection_snapshot_json(self.seg_objects_offset_world, limit),
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
                let offset_world = layer.offset_world;
                let layer_name = layer.name.clone();
                let Some(objects) = layer.object_layer_mut() else {
                    return serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")});
                };
                objects.clear_selection();
                serde_json::json!({
                    "target": "spatial_shape",
                    "layer_id": id,
                    "layer_name": layer_name,
                    "selection": objects.selection_snapshot_json(offset_world, limit),
                })
            }
            Ok(_) => serde_json::json!({"error": "active layer does not support object selection"}),
            Err(error) => serde_json::json!({"error": error}),
        };
        self.bump_render_id();
        result
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

    pub fn control_focus_object(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let result = match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => self
                .seg_objects
                .control_focus_object_json(params, self.seg_objects_offset_world),
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
                objects.control_focus_object_json(params, offset)
            }
            Ok(_) => serde_json::json!({"error": "active layer does not support object focus"}),
            Err(error) => serde_json::json!({"error": error}),
        };
        self.bump_render_id();
        result
    }

    pub fn control_clear_object_focus(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let result = match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => self.seg_objects.control_clear_focus_json(),
            Ok(LayerId::SpatialShape(id)) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer_mut())
                .map(ObjectsLayer::control_clear_focus_json)
                .unwrap_or_else(|| serde_json::json!({"error": format!("spatial shape layer {id} has no object layer")})),
            Ok(_) => serde_json::json!({"error": "active layer does not support object focus"}),
            Err(error) => serde_json::json!({"error": error}),
        };
        self.bump_render_id();
        result
    }

    pub fn control_get_object_filter(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => serde_json::json!({
                "target": "segmentation_objects",
                "filter": self.seg_objects.filter_snapshot_json(),
            }),
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
                serde_json::json!({
                    "target": "spatial_shape",
                    "layer_id": id,
                    "layer_name": layer_name,
                    "filter": objects.filter_snapshot_json(),
                })
            }
            Ok(_) => serde_json::json!({"error": "active layer does not support object filters"}),
            Err(error) => serde_json::json!({"error": error}),
        }
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

    pub fn control_clear_object_filter(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let result = match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => {
                self.seg_objects.clear_filter();
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
                objects.clear_filter();
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

    pub fn control_set_object_filter_model(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let result = match self.control_object_selection_target(params) {
            Ok(LayerId::SegmentationObjects) => serde_json::json!({
                "target": "segmentation_objects",
                "filter": self.seg_objects.control_set_filter_model_json(params),
            }),
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
                serde_json::json!({
                    "target": "spatial_shape",
                    "layer_id": id,
                    "layer_name": layer_name,
                    "filter": objects.control_set_filter_model_json(params),
                })
            }
            Ok(_) => serde_json::json!({"error": "active layer does not support object filters"}),
            Err(error) => serde_json::json!({"error": error}),
        };
        self.bump_render_id();
        result
    }
}
