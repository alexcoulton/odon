use super::super::*;

impl OmeZarrViewerApp {
    pub(in crate::app) fn control_mask_layer_snapshot(
        &self,
        layer: &MaskLayer,
    ) -> serde_json::Value {
        serde_json::json!({
            "id": layer.id,
            "name": layer.name,
            "visible": layer.visible,
            "opacity": layer.opacity,
            "width_screen_px": layer.width_screen_px,
            "display_mode": layer.display_mode.storage_key(),
            "color_rgb": layer.color_rgb,
            "offset_world": [layer.offset_world.x, layer.offset_world.y],
            "editable": layer.editable,
            "polygon_count": layer.polygons_world.len(),
            "source_geojson": layer.source_geojson.as_ref().map(|path| path.to_string_lossy().into_owned()),
            "active": self.active_layer == LayerId::Mask(layer.id),
        })
    }

    pub fn control_list_mask_layers(&self) -> serde_json::Value {
        serde_json::json!({
            "total": self.mask_layers.len(),
            "layers": self.mask_layers.iter().map(|layer| self.control_mask_layer_snapshot(layer)).collect::<Vec<_>>(),
            "undo_available": !self.undo_stack.is_empty(),
        })
    }

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

    pub fn control_object_selection_projection_snapshot(&self) -> serde_json::Value {
        let mut projection = self.seg_objects.control_selection_projection_json();
        projection
            .as_object_mut()
            .expect("object selection projection is an object")
            .insert(
                "generation".to_string(),
                serde_json::json!(self.control_actor_object_selection_generation.max(1)),
            );
        projection
    }

    pub fn control_native_layers_projection_snapshot(&mut self) -> serde_json::Value {
        let workspace = self.control_viewport_workspace_snapshot();
        serde_json::json!({
            "viewports": workspace
                .get("viewports")
                .and_then(serde_json::Value::as_array)
                .into_iter()
                .flatten()
                .map(|viewport| serde_json::json!({
                    "viewport_id": viewport.get("viewport_id").cloned().unwrap_or(serde_json::Value::Null),
                    "presentation_revision": viewport.get("presentation_revision").cloned().unwrap_or(serde_json::json!(1)),
                    "layers": viewport.get("native_layers").cloned().unwrap_or_else(|| serde_json::json!([])),
                }))
                .collect::<Vec<_>>(),
        })
    }

    pub fn record_native_layers_intent(&mut self, before: &serde_json::Value) {
        let after = self.control_native_layers_projection_snapshot();
        let before_viewports = before
            .get("viewports")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        let after_viewports = after
            .get("viewports")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        for current in after_viewports {
            let Some(viewport_id) = current
                .get("viewport_id")
                .and_then(serde_json::Value::as_str)
            else {
                continue;
            };
            let Some(previous) = before_viewports.iter().find(|candidate| {
                candidate
                    .get("viewport_id")
                    .and_then(serde_json::Value::as_str)
                    == Some(viewport_id)
            }) else {
                continue;
            };
            if previous.get("layers") == current.get("layers") {
                continue;
            }

            // The atomic layer transaction subsumes the granular commands emitted by the
            // compatibility native-diff bridge. Keep channel sort/group and non-layer commands,
            // which are not represented by the native layer projection.
            self.native_control_intents.retain(|intent| {
                if intent
                    .params
                    .get("viewport_id")
                    .and_then(serde_json::Value::as_str)
                    != Some(viewport_id)
                {
                    return true;
                }
                match intent.method {
                    "viewer.viewports.channels.set_visible"
                    | "viewer.viewports.channels.set_active"
                    | "viewer.viewports.channels.set_color"
                    | "viewer.viewports.channels.set_contrast"
                    | "viewer.viewports.objects.style.set"
                    | "viewer.viewports.objects.legend.set"
                    | "viewer.viewports.layers.set"
                    | "viewer.viewports.layers.set_visibility"
                    | "viewer.viewports.layers.set_active" => false,
                    "viewer.viewports.channels.set_order" => intent.params.get("sort").is_some(),
                    _ => true,
                }
            });
            self.native_control_intents.push(NativeControlIntent {
                method: "viewer.viewports.layers.state.replace",
                params: serde_json::json!({
                    "viewport_id":viewport_id,
                    "if_presentation_revision":previous
                        .get("presentation_revision")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(1),
                    "state":current.get("layers").cloned().unwrap_or_else(|| serde_json::json!([])),
                }),
            });
        }
    }

    pub fn record_native_object_selection_intent(&mut self, before: &serde_json::Value) {
        if self.control_actor_object_selection_generation == 0 {
            return;
        }
        let after = self.control_object_selection_projection_snapshot();
        let semantic = |value: &serde_json::Value| {
            serde_json::json!({
                "selected_indices":value.get("selected_indices").cloned().unwrap_or_else(|| serde_json::json!([])),
                "primary_index":value.get("primary_index").cloned().unwrap_or(serde_json::Value::Null),
            })
        };
        if semantic(before) == semantic(&after) {
            return;
        }
        self.native_control_intents.push(NativeControlIntent {
            method: "viewer.objects.selection.state.replace",
            params: serde_json::json!({
                "expected_generation":before.get("generation").and_then(serde_json::Value::as_u64).unwrap_or(1),
                "state":semantic(&after),
                "target":"segmentation_objects",
            }),
        });
    }

    pub fn record_native_mask_intent(&mut self, before: &serde_json::Value) {
        if self.native_mask_actor_intent_emitted {
            self.native_mask_actor_intent_emitted = false;
            return;
        }
        let after = self.control_mask_projection_snapshot();
        let semantic = |value: &serde_json::Value| {
            serde_json::json!({
                "layers": value.get("layers").cloned().unwrap_or_else(|| serde_json::json!([])),
                "active_layer_id": value.get("active_layer_id").cloned().unwrap_or(serde_json::Value::Null),
                "selection": value.get("selection").cloned().unwrap_or(serde_json::Value::Null),
            })
        };
        if semantic(before) == semantic(&after) {
            return;
        }
        self.native_control_intents.push(NativeControlIntent {
            method: "viewer.masks.state.replace",
            params: serde_json::json!({
                "expected_generation": before.get("generation").and_then(serde_json::Value::as_u64).unwrap_or(1),
                "state": semantic(&after),
            }),
        });
    }

    pub fn control_get_mask_layer(&self, params: &serde_json::Value) -> serde_json::Value {
        let Some(id) = params.get("id").and_then(serde_json::Value::as_u64) else {
            return serde_json::json!({"error": "mask layer id is required"});
        };
        self.mask_layers
            .iter()
            .find(|layer| layer.id == id)
            .map(|layer| self.control_mask_layer_snapshot(layer))
            .unwrap_or_else(|| serde_json::json!({"error": format!("mask layer {id} not found")}))
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

    pub fn control_set_mask_selection(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(layer_id) = params.get("id").and_then(serde_json::Value::as_u64) else {
            return serde_json::json!({"error": "mask layer id is required"});
        };
        let Some(polygon_index) = params
            .get("index")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
        else {
            return serde_json::json!({"error": "polygon index is required"});
        };
        let Some(layer) = self.mask_layers.iter().find(|layer| layer.id == layer_id) else {
            return serde_json::json!({"error": format!("mask layer {layer_id} not found")});
        };
        let Some(polygon) = layer.polygons_world.get(polygon_index) else {
            return serde_json::json!({"error": format!("mask polygon index {polygon_index} is out of range")});
        };
        let vertex_index = match params.get("vertex_index") {
            Some(serde_json::Value::Null) | None => None,
            Some(value) => {
                let Some(index) = value.as_u64().and_then(|value| usize::try_from(value).ok())
                else {
                    return serde_json::json!({"error": "vertex_index must be a non-negative integer or null"});
                };
                if index >= Self::mask_polygon_unique_vertex_count(polygon) {
                    return serde_json::json!({"error": format!("mask vertex index {index} is out of range")});
                }
                Some(index)
            }
        };
        self.selected_mask_polygon = Some(MaskPolygonSelection {
            layer_id,
            polygon_idx: polygon_index,
        });
        self.selected_mask_vertex = vertex_index;
        self.active_layer = LayerId::Mask(layer_id);
        self.bump_render_id();
        self.control_get_mask_selection()
    }

    pub fn control_clear_mask_selection(&mut self) -> serde_json::Value {
        let cleared = self.selected_mask_polygon.is_some() || self.selected_mask_vertex.is_some();
        self.clear_mask_polygon_selection();
        if cleared {
            self.bump_render_id();
        }
        serde_json::json!({"cleared": cleared, "selection": null})
    }

    pub fn control_create_mask_layer(&mut self, params: &serde_json::Value) -> serde_json::Value {
        self.push_mask_undo_snapshot();
        let name = params
            .get("name")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
        let id = self.create_editable_mask_layer(name);
        self.active_layer = LayerId::Mask(id);
        if let Some(layer) = self.mask_layers.iter_mut().find(|layer| layer.id == id) {
            if let Some(editable) = params.get("editable").and_then(serde_json::Value::as_bool) {
                layer.editable = editable;
            }
        }
        self.rebuild_layer_orders();
        self.bump_render_id();
        self.control_get_mask_layer(&serde_json::json!({"id": id}))
    }

    pub fn control_update_mask_layer(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(id) = params.get("id").and_then(serde_json::Value::as_u64) else {
            return serde_json::json!({"error": "mask layer id is required"});
        };
        if !self.mask_layers.iter().any(|layer| layer.id == id) {
            return serde_json::json!({"error": format!("mask layer {id} not found")});
        }
        if params
            .get("name")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|name| name.trim().is_empty())
        {
            return serde_json::json!({"error": "mask layer name must not be empty"});
        }
        if let Some(value) = params.get("opacity").and_then(serde_json::Value::as_f64) {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return serde_json::json!({"error": "opacity must be finite and between 0 and 1"});
            }
        }
        if let Some(value) = params
            .get("width_screen_px")
            .and_then(serde_json::Value::as_f64)
        {
            if !value.is_finite() || value <= 0.0 {
                return serde_json::json!({"error": "width_screen_px must be finite and greater than zero"});
            }
        }
        let display_mode = match params
            .get("display_mode")
            .and_then(serde_json::Value::as_str)
        {
            Some(mode) => match MaskDisplayMode::from_storage_key(mode) {
                Some(mode) => Some(mode),
                None => return serde_json::json!({"error": "unknown mask display_mode"}),
            },
            None => None,
        };
        let color_rgb = match params.get("color_rgb") {
            Some(value) => {
                let Some(values) = value.as_array() else {
                    return serde_json::json!({"error": "color_rgb must contain three integers from 0 to 255"});
                };
                if values.len() != 3
                    || values
                        .iter()
                        .any(|value| value.as_u64().is_none_or(|value| value > 255))
                {
                    return serde_json::json!({"error": "color_rgb must contain three integers from 0 to 255"});
                }
                Some([
                    values[0].as_u64().unwrap() as u8,
                    values[1].as_u64().unwrap() as u8,
                    values[2].as_u64().unwrap() as u8,
                ])
            }
            None => None,
        };
        let offset_world = match params.get("offset_world") {
            Some(value) => {
                let Some(values) = value.as_array() else {
                    return serde_json::json!({"error": "offset_world must contain two finite numbers"});
                };
                if values.len() != 2
                    || values
                        .iter()
                        .any(|value| value.as_f64().is_none_or(|value| !value.is_finite()))
                {
                    return serde_json::json!({"error": "offset_world must contain two finite numbers"});
                }
                Some(egui::vec2(
                    values[0].as_f64().unwrap() as f32,
                    values[1].as_f64().unwrap() as f32,
                ))
            }
            None => None,
        };
        self.push_mask_undo_snapshot();
        let layer = self
            .mask_layers
            .iter_mut()
            .find(|layer| layer.id == id)
            .unwrap();
        if let Some(name) = params.get("name").and_then(serde_json::Value::as_str) {
            layer.name = name.trim().to_string();
        }
        if let Some(visible) = params.get("visible").and_then(serde_json::Value::as_bool) {
            layer.visible = visible;
        }
        if let Some(editable) = params.get("editable").and_then(serde_json::Value::as_bool) {
            layer.editable = editable;
        }
        if let Some(value) = params.get("opacity").and_then(serde_json::Value::as_f64) {
            layer.opacity = value as f32;
        }
        if let Some(value) = params
            .get("width_screen_px")
            .and_then(serde_json::Value::as_f64)
        {
            layer.width_screen_px = value as f32;
        }
        if let Some(mode) = display_mode {
            layer.display_mode = mode;
        }
        if let Some(color_rgb) = color_rgb {
            layer.color_rgb = color_rgb;
        }
        if let Some(offset_world) = offset_world {
            layer.offset_world = offset_world;
        }
        layer.raster_display = None;
        self.mark_mask_layers_project_dirty();
        self.bump_render_id();
        self.control_get_mask_layer(params)
    }

    pub fn control_delete_mask_layer(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(id) = params.get("id").and_then(serde_json::Value::as_u64) else {
            return serde_json::json!({"error": "mask layer id is required"});
        };
        if self.delete_mask_layer(id) {
            serde_json::json!({"deleted": true, "id": id})
        } else {
            serde_json::json!({"error": format!("mask layer {id} not found")})
        }
    }

    pub fn control_list_mask_polygons(&self, params: &serde_json::Value) -> serde_json::Value {
        let Some(id) = params.get("id").and_then(serde_json::Value::as_u64) else {
            return serde_json::json!({"error": "mask layer id is required"});
        };
        let Some(layer) = self.mask_layers.iter().find(|layer| layer.id == id) else {
            return serde_json::json!({"error": format!("mask layer {id} not found")});
        };
        let offset = params
            .get("offset")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize;
        let limit = params
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(200) as usize;
        let polygons = layer
            .polygons_world
            .iter()
            .enumerate()
            .skip(offset)
            .take(limit)
            .map(|(index, polygon)| serde_json::json!({
                "index": index,
                "vertices_local": polygon.iter().map(|point| [point.x, point.y]).collect::<Vec<_>>(),
                "vertices_world": polygon.iter().map(|point| [point.x + layer.offset_world.x, point.y + layer.offset_world.y]).collect::<Vec<_>>(),
            }))
            .collect::<Vec<_>>();
        serde_json::json!({
            "layer_id": id,
            "total": layer.polygons_world.len(),
            "offset": offset,
            "limit": limit,
            "has_more": offset.saturating_add(polygons.len()) < layer.polygons_world.len(),
            "polygons": polygons,
        })
    }

    pub(in crate::app) fn control_mask_vertices(
        params: &serde_json::Value,
        layer_offset: egui::Vec2,
    ) -> Result<Vec<egui::Pos2>, String> {
        let values = params
            .get("vertices")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| "vertices is required".to_string())?;
        if values.len() < 3 {
            return Err("vertices must contain at least three points".to_string());
        }
        let coordinate_space = params
            .get("coordinate_space")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("world");
        let world = match coordinate_space {
            "world" => true,
            "local" => false,
            _ => return Err("coordinate_space must be 'world' or 'local'".to_string()),
        };
        values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let pair = value
                    .as_array()
                    .filter(|pair| pair.len() == 2)
                    .ok_or_else(|| format!("vertices[{index}] must be [x, y]"))?;
                let x = pair[0]
                    .as_f64()
                    .filter(|v| v.is_finite())
                    .ok_or_else(|| format!("vertices[{index}][0] must be finite"))?
                    as f32;
                let y = pair[1]
                    .as_f64()
                    .filter(|v| v.is_finite())
                    .ok_or_else(|| format!("vertices[{index}][1] must be finite"))?
                    as f32;
                let point = egui::pos2(x, y);
                Ok(if world { point - layer_offset } else { point })
            })
            .collect()
    }

    pub fn control_add_mask_polygon(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(id) = params.get("id").and_then(serde_json::Value::as_u64) else {
            return serde_json::json!({"error": "mask layer id is required"});
        };
        let Some(layer) = self.mask_layers.iter().find(|layer| layer.id == id) else {
            return serde_json::json!({"error": format!("mask layer {id} not found")});
        };
        if !layer.editable {
            return serde_json::json!({"error": format!("mask layer {id} is read-only")});
        }
        let vertices = match Self::control_mask_vertices(params, layer.offset_world) {
            Ok(vertices) => vertices,
            Err(error) => return serde_json::json!({"error": error}),
        };
        self.push_mask_undo_snapshot();
        let layer = self
            .mask_layers
            .iter_mut()
            .find(|layer| layer.id == id)
            .unwrap();
        layer.add_closed_polygon(vertices);
        let index = layer.polygons_world.len() - 1;
        self.mark_mask_layers_project_dirty();
        self.bump_render_id();
        serde_json::json!({"added": true, "layer_id": id, "index": index})
    }

    pub fn control_update_mask_polygon(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(id) = params.get("id").and_then(serde_json::Value::as_u64) else {
            return serde_json::json!({"error": "mask layer id is required"});
        };
        let Some(index) = params
            .get("index")
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as usize)
        else {
            return serde_json::json!({"error": "polygon index is required"});
        };
        let Some(layer) = self.mask_layers.iter().find(|layer| layer.id == id) else {
            return serde_json::json!({"error": format!("mask layer {id} not found")});
        };
        if !layer.editable {
            return serde_json::json!({"error": format!("mask layer {id} is read-only")});
        }
        if index >= layer.polygons_world.len() {
            return serde_json::json!({"error": format!("mask polygon index {index} is out of range")});
        }
        let mut vertices = match Self::control_mask_vertices(params, layer.offset_world) {
            Ok(vertices) => vertices,
            Err(error) => return serde_json::json!({"error": error}),
        };
        if vertices.first() != vertices.last() {
            vertices.push(vertices[0]);
        }
        self.push_mask_undo_snapshot();
        let layer = self
            .mask_layers
            .iter_mut()
            .find(|layer| layer.id == id)
            .unwrap();
        layer.polygons_world[index] = vertices;
        layer.raster_display = None;
        self.mark_mask_layers_project_dirty();
        self.bump_render_id();
        serde_json::json!({"updated": true, "layer_id": id, "index": index})
    }

    pub fn control_remove_mask_polygon(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(id) = params.get("id").and_then(serde_json::Value::as_u64) else {
            return serde_json::json!({"error": "mask layer id is required"});
        };
        let Some(index) = params
            .get("index")
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as usize)
        else {
            return serde_json::json!({"error": "polygon index is required"});
        };
        let Some(layer) = self.mask_layers.iter().find(|layer| layer.id == id) else {
            return serde_json::json!({"error": format!("mask layer {id} not found")});
        };
        if !layer.editable {
            return serde_json::json!({"error": format!("mask layer {id} is read-only")});
        }
        if index >= layer.polygons_world.len() {
            return serde_json::json!({"error": format!("mask polygon index {index} is out of range")});
        }
        self.push_mask_undo_snapshot();
        let layer = self
            .mask_layers
            .iter_mut()
            .find(|layer| layer.id == id)
            .unwrap();
        layer.polygons_world.remove(index);
        layer.raster_display = None;
        self.mark_mask_layers_project_dirty();
        self.bump_render_id();
        serde_json::json!({"removed": true, "layer_id": id, "index": index})
    }

    pub fn control_undo_mask_edit(&mut self) -> serde_json::Value {
        let undone = self.undo_last_edit();
        if undone {
            self.bump_render_id();
        }
        serde_json::json!({"undone": undone, "undo_available": !self.undo_stack.is_empty()})
    }

    pub fn control_import_masks_geojson(
        &mut self,
        path: &Path,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let downsample_factor = params
            .get("downsample_factor")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1.0);
        if !downsample_factor.is_finite() || downsample_factor <= 0.0 {
            return serde_json::json!({"error": "downsample_factor must be finite and greater than zero"});
        }
        let editable = params
            .get("editable")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);
        let name = params
            .get("name")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(str::to_string)
            .or_else(|| {
                path.file_stem()
                    .and_then(|stem| stem.to_str())
                    .map(str::to_string)
            })
            .unwrap_or_else(|| "Imported masks".to_string());
        let polygons = match load_geojson_polylines_world(
            path,
            downsample_factor as f32,
            PolygonRingMode::AllRings,
        ) {
            Ok(polygons) => polygons,
            Err(error) => {
                return serde_json::json!({
                    "error": format!("failed to import mask GeoJSON: {error}"),
                    "path": path.to_string_lossy(),
                });
            }
        };
        if polygons.is_empty() {
            return serde_json::json!({
                "error": "mask GeoJSON contains no supported polygon or line geometry",
                "path": path.to_string_lossy(),
            });
        }

        self.push_mask_undo_snapshot();
        let id = self.next_mask_layer_id.max(1);
        self.next_mask_layer_id = id.saturating_add(1);
        let polygon_count = polygons.len();
        self.mask_layers.push(MaskLayer {
            id,
            name,
            visible: true,
            opacity: 0.85,
            width_screen_px: 1.5,
            display_mode: MaskDisplayMode::default_new_layer(),
            color_rgb: [50, 220, 255],
            offset_world: egui::Vec2::ZERO,
            editable,
            polygons_world: polygons,
            raster_display: None,
            source_geojson: Some(path.to_path_buf()),
        });
        self.active_layer = LayerId::Mask(id);
        self.mark_mask_layers_project_dirty();
        self.rebuild_layer_orders();
        self.bump_render_id();
        serde_json::json!({
            "imported": true,
            "path": path.to_string_lossy(),
            "layer_id": id,
            "polygon_count": polygon_count,
            "layer": self.control_get_mask_layer(&serde_json::json!({"id": id})),
        })
    }

    pub fn control_export_masks_geojson(
        &self,
        path: &Path,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let overwrite = params
            .get("overwrite")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        if path.exists() && !overwrite {
            return serde_json::json!({
                "error": "destination exists; pass overwrite=true to replace it",
                "path": path.to_string_lossy(),
            });
        }
        let layer_id = params.get("id").and_then(serde_json::Value::as_u64);
        let export = match layer_id {
            Some(id) => self.export_mask_layer_geojson(id, path),
            None => self.export_masks_geojson(path),
        };
        if let Err(error) = export {
            return serde_json::json!({
                "error": format!("failed to export mask GeoJSON: {error}"),
                "path": path.to_string_lossy(),
            });
        }
        let polygon_count = match layer_id {
            Some(id) => self
                .mask_layers
                .iter()
                .find(|layer| layer.id == id)
                .map(|layer| layer.polygons_world.len())
                .unwrap_or(0),
            None => self
                .mask_layers
                .iter()
                .map(|layer| layer.polygons_world.len())
                .sum(),
        };
        serde_json::json!({
            "exported": true,
            "path": path.to_string_lossy(),
            "layer_id": layer_id,
            "layer_count": layer_id.map(|_| 1).unwrap_or(self.mask_layers.len()),
            "polygon_count": polygon_count,
            "bytes": std::fs::metadata(path).ok().map(|metadata| metadata.len()),
        })
    }

    pub fn control_mask_persistence(&self) -> serde_json::Value {
        let local_root = self.dataset.source.local_path();
        let persisted_layer_count = local_root
            .and_then(|root| self.project_space.roi_mask_layers(root))
            .map(|layers| layers.len());
        serde_json::json!({
            "dirty": self.mask_layers_project_dirty,
            "dataset_local": local_root.is_some(),
            "dataset_path": local_root.map(|path| path.to_string_lossy().into_owned()),
            "project_path": self.project_space.current_project_path().map(|path| path.to_string_lossy().into_owned()),
            "live_layer_count": self.mask_layers.len(),
            "persisted_layer_count": persisted_layer_count,
        })
    }

    pub fn control_sync_masks_to_project(&mut self) -> serde_json::Value {
        if self.dataset.source.local_path().is_none() {
            return serde_json::json!({"error": "mask project persistence requires a local dataset"});
        }
        self.sync_mask_layers_into_project_space();
        serde_json::json!({
            "synced": true,
            "persistence": self.control_mask_persistence(),
        })
    }
}
