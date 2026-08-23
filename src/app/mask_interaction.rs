use super::*;

impl OmeZarrViewerApp {
    pub(super) fn mask_polygon_gesture_active(&self) -> bool {
        self.dragging_mask_vertex.is_some() || self.moving_mask_polygon.is_some()
    }

    pub(super) fn mask_semantic_state(&self) -> serde_json::Value {
        let active_layer_id = match self.active_layer {
            LayerId::Mask(id) => Some(id),
            _ => None,
        };
        serde_json::json!({
            "layers": self.mask_layers.iter().map(MaskLayer::to_project).collect::<Vec<_>>(),
            "active_layer_id": active_layer_id,
            "selection": self.mask_selection_value(),
        })
    }

    pub(super) fn mask_selection_value(&self) -> serde_json::Value {
        self.selected_mask_polygon
            .and_then(|selection| {
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
                    "vertices_world": polygon.iter().map(|point| [
                        point.x + layer.offset_world.x,
                        point.y + layer.offset_world.y,
                    ]).collect::<Vec<_>>(),
                }))
            })
            .unwrap_or(serde_json::Value::Null)
    }

    pub(super) fn submit_native_mask_command(
        &mut self,
        method: &'static str,
        mut params: serde_json::Map<String, serde_json::Value>,
    ) {
        params.insert(
            "expected_generation".to_string(),
            serde_json::json!(self.control_actor_mask_generation.max(1)),
        );
        if !matches!(
            method,
            "viewer.masks.selection.set" | "viewer.masks.selection.clear"
        ) {
            params.insert("sync_project".to_string(), serde_json::json!(true));
        }
        self.native_command_ingress.push(NativeControlIntent {
            method,
            params: serde_json::Value::Object(params),
        });
    }

    pub fn request_mask_export(&mut self, path: &Path, layer_id: Option<u64>) {
        let mut params = serde_json::Map::from_iter([
            (
                "path".to_string(),
                serde_json::json!(path.to_string_lossy()),
            ),
            ("overwrite".to_string(), serde_json::json!(true)),
        ]);
        if let Some(layer_id) = layer_id {
            params.insert("id".to_string(), serde_json::json!(layer_id));
        }
        self.native_command_ingress.push(NativeControlIntent {
            method: "viewer.masks.export_geojson",
            params: serde_json::Value::Object(params),
        });
        self.set_status(format!("Exporting masks -> {}", path.to_string_lossy()));
    }

    pub(super) fn submit_native_mask_layer_update(&mut self, layer: &MaskLayer) {
        let mut params = serde_json::Map::new();
        params.insert("id".to_string(), serde_json::json!(layer.id));
        params.insert("name".to_string(), serde_json::json!(layer.name));
        params.insert("visible".to_string(), serde_json::json!(layer.visible));
        params.insert("editable".to_string(), serde_json::json!(layer.editable));
        params.insert(
            "active".to_string(),
            serde_json::json!(self.active_layer == LayerId::Mask(layer.id)),
        );
        params.insert("opacity".to_string(), serde_json::json!(layer.opacity));
        params.insert(
            "width_screen_px".to_string(),
            serde_json::json!(layer.width_screen_px),
        );
        params.insert(
            "display_mode".to_string(),
            serde_json::json!(layer.display_mode.storage_key()),
        );
        params.insert("color_rgb".to_string(), serde_json::json!(layer.color_rgb));
        params.insert(
            "offset_world".to_string(),
            serde_json::json!([layer.offset_world.x, layer.offset_world.y]),
        );
        self.submit_native_mask_command("viewer.masks.layers.update", params);
    }

    pub(super) fn submit_native_mask_state_replace(
        &mut self,
        layers: &[MaskLayer],
        selection: serde_json::Value,
    ) {
        let active_layer_id = match self.active_layer {
            LayerId::Mask(id) if layers.iter().any(|layer| layer.id == id) => Some(id),
            _ => None,
        };
        self.submit_native_mask_state_replace_with_active(layers, active_layer_id, selection);
    }

    pub(super) fn submit_native_mask_state_replace_with_active(
        &mut self,
        layers: &[MaskLayer],
        active_layer_id: Option<u64>,
        selection: serde_json::Value,
    ) {
        self.native_command_ingress.push(NativeControlIntent {
            method: "viewer.masks.state.replace",
            params: serde_json::json!({
                "expected_generation":self.control_actor_mask_generation.max(1),
                "sync_project":true,
                "state":{
                    "layers":layers.iter().map(MaskLayer::to_project).collect::<Vec<_>>(),
                    "active_layer_id":active_layer_id,
                    "selection":selection,
                },
            }),
        });
    }

    pub(super) fn submit_native_mask_active_layer(&mut self, active_layer_id: Option<u64>) {
        self.submit_native_mask_state_replace_with_active(
            &self.mask_layers.clone(),
            active_layer_id,
            self.mask_selection_value(),
        );
    }

    pub(super) fn delete_mask_layer(&mut self, layer_id: u64) -> bool {
        if !self.mask_layers.iter().any(|layer| layer.id == layer_id) {
            return false;
        }
        self.submit_native_mask_command(
            "viewer.masks.layers.delete",
            serde_json::Map::from_iter([("id".to_string(), serde_json::json!(layer_id))]),
        );
        true
    }

    #[cfg(test)]
    pub(super) fn restore_mask_layers_from_project_space(&mut self) {
        let Some(local_root) = self.dataset.source.local_path() else {
            self.mask_layers.clear();
            self.clear_mask_polygon_selection();
            return;
        };
        let Some(layers) = self.project_space.roi_mask_layers(local_root) else {
            self.mask_layers.clear();
            self.clear_mask_polygon_selection();
            return;
        };

        self.mask_layers = layers.iter().map(MaskLayer::from_project).collect();

        // Reset any in-progress drawing if it no longer targets a valid layer.
        if let Some(id) = self.drawing_mask_layer {
            if !self.mask_layers.iter().any(|l| l.id == id) {
                self.drawing_mask_layer = None;
                self.drawing_mask_polygon.clear();
            }
        }

        self.rebuild_layer_orders();
        self.bump_render_id();
    }

    pub(super) fn ensure_editable_mask_layer(&mut self) -> Option<u64> {
        if let LayerId::Mask(id) = self.active_layer {
            if let Some(l) = self.mask_layers.iter().find(|l| l.id == id) {
                if l.editable {
                    return Some(id);
                }
            }
        }

        if let Some(l) = self.mask_layers.iter().rev().find(|l| l.editable) {
            return Some(l.id);
        }

        self.request_create_editable_mask_layer(None)
    }

    pub(super) fn request_create_editable_mask_layer(
        &mut self,
        name: Option<String>,
    ) -> Option<u64> {
        if !self
            .native_command_ingress
            .contains_pending("viewer.masks.layers.create")
        {
            let mut params = serde_json::Map::new();
            if let Some(name) = name {
                params.insert("name".to_string(), serde_json::json!(name));
            }
            self.submit_native_mask_command("viewer.masks.layers.create", params);
        }
        None
    }

    pub(super) fn finish_native_layer_move(&mut self, state: &LayerMoveState) {
        let mut desired_native = self.control_native_layer_snapshot_list();
        let mut desired = self.mask_layers.clone();
        let mut mask_changed = false;
        let mut native_changed = false;
        for start in &state.targets {
            let current = self.layer_offset_world(start.layer);
            if (current - start.offset_world).length_sq() <= 1e-12 {
                continue;
            }
            if let LayerId::Mask(id) = start.layer {
                if let Some(layer) = desired.iter_mut().find(|layer| layer.id == id) {
                    layer.offset_world = current;
                    mask_changed = true;
                }
                let layer_id = Self::layer_id_storage_key(start.layer);
                if let Some(layer) = desired_native.as_array_mut().and_then(|layers| {
                    layers.iter_mut().find(|layer| {
                        layer.get("layer_id").and_then(serde_json::Value::as_str)
                            == Some(layer_id.as_str())
                    })
                }) {
                    layer["offset_world"] =
                        serde_json::json!([start.offset_world.x, start.offset_world.y]);
                }
            } else {
                native_changed = true;
            }
            if let Some(local) = self.layer_offset_world_mut(start.layer) {
                *local = start.offset_world;
            }
        }
        if native_changed {
            if let Some((viewport_id, revision)) = state.actor_scope.as_ref() {
                self.submit_native_layer_state_replace_at(viewport_id, *revision, desired_native);
            }
        }
        if mask_changed {
            self.submit_native_mask_state_replace(&desired, self.mask_selection_value());
        }
    }

    pub(super) fn request_native_mask_undo(&mut self) -> bool {
        if !matches!(self.active_layer, LayerId::Mask(_)) || !self.control_actor_mask_undo_available
        {
            return false;
        }
        self.submit_native_mask_command("viewer.masks.undo", serde_json::Map::new());
        true
    }

    pub(super) fn mask_undo_available(&self) -> bool {
        matches!(self.active_layer, LayerId::Mask(_)) && self.control_actor_mask_undo_available
    }

    pub(super) fn finish_drawing_mask_polygon(&mut self) -> bool {
        if self.drawing_mask_polygon.len() < 3 {
            return false;
        }

        let vertices = std::mem::take(&mut self.drawing_mask_polygon);
        let Some(id) = self
            .drawing_mask_layer
            .or_else(|| self.ensure_editable_mask_layer())
        else {
            self.drawing_mask_polygon = vertices;
            return false;
        };
        self.drawing_mask_layer = Some(id);
        let Some(_layer) = self
            .mask_layers
            .iter()
            .find(|layer| layer.id == id && layer.editable)
        else {
            self.drawing_mask_polygon = vertices;
            return false;
        };
        let mut params = serde_json::Map::new();
        params.insert("id".to_string(), serde_json::json!(id));
        params.insert("coordinate_space".to_string(), serde_json::json!("local"));
        params.insert(
            "vertices".to_string(),
            serde_json::json!(
                vertices
                    .iter()
                    .map(|point| [point.x, point.y])
                    .collect::<Vec<_>>()
            ),
        );
        self.submit_native_mask_command("viewer.masks.polygons.add", params);
        self.drawing_mask_layer = Some(id);
        true
    }

    pub(super) fn clear_mask_polygon_selection(&mut self) {
        self.selected_mask_polygon = None;
        self.selected_mask_vertex = None;
        self.dragging_mask_vertex = None;
        self.moving_mask_polygon = None;
    }

    pub(super) fn mask_polygon_unique_vertex_count(poly: &[egui::Pos2]) -> usize {
        if poly.len() >= 2 && poly.first() == poly.last() {
            poly.len() - 1
        } else {
            poly.len()
        }
    }

    pub(super) fn validate_mask_polygon_selection(&mut self) {
        let Some(selection) = self.selected_mask_polygon else {
            self.selected_mask_vertex = None;
            self.dragging_mask_vertex = None;
            self.moving_mask_polygon = None;
            return;
        };
        let valid = self
            .mask_layers
            .iter()
            .find(|layer| layer.id == selection.layer_id)
            .and_then(|layer| layer.polygons_world.get(selection.polygon_idx))
            .is_some_and(|poly| Self::mask_polygon_unique_vertex_count(poly) >= 3);
        if !valid {
            self.clear_mask_polygon_selection();
        }
    }

    pub(super) fn hit_mask_polygon_at(
        &self,
        layer_id: u64,
        pointer_world: egui::Pos2,
        pointer_screen: egui::Pos2,
        rect: egui::Rect,
    ) -> Option<MaskPolygonHit> {
        let layer = self
            .mask_layers
            .iter()
            .find(|layer| layer.id == layer_id && layer.visible)?;
        let pointer_local = pointer_world - layer.offset_world;

        for (polygon_idx, poly) in layer.polygons_world.iter().enumerate().rev() {
            let n = Self::mask_polygon_unique_vertex_count(poly);
            if n < 3 {
                continue;
            }
            for (vertex_idx, vertex) in poly.iter().copied().take(n).enumerate() {
                let screen = self
                    .camera
                    .world_to_screen(vertex + layer.offset_world, rect);
                if pointer_screen.distance(screen) <= MASK_POLYGON_VERTEX_HIT_RADIUS_SCREEN_PX {
                    return Some(MaskPolygonHit {
                        polygon_idx,
                        vertex_idx: Some(vertex_idx),
                    });
                }
            }
        }

        for (polygon_idx, poly) in layer.polygons_world.iter().enumerate().rev() {
            let n = Self::mask_polygon_unique_vertex_count(poly);
            if n < 3 {
                continue;
            }
            for i in 0..n {
                let a = self
                    .camera
                    .world_to_screen(poly[i] + layer.offset_world, rect);
                let b = self
                    .camera
                    .world_to_screen(poly[(i + 1) % n] + layer.offset_world, rect);
                if distance_to_screen_segment(pointer_screen, a, b)
                    <= MASK_POLYGON_EDGE_HIT_RADIUS_SCREEN_PX
                {
                    return Some(MaskPolygonHit {
                        polygon_idx,
                        vertex_idx: None,
                    });
                }
            }
        }

        for (polygon_idx, poly) in layer.polygons_world.iter().enumerate().rev() {
            if point_in_mask_polygon(pointer_local, poly) {
                return Some(MaskPolygonHit {
                    polygon_idx,
                    vertex_idx: None,
                });
            }
        }

        None
    }

    pub(super) fn select_mask_polygon_at(
        &mut self,
        layer_id: u64,
        pointer_world: egui::Pos2,
        pointer_screen: egui::Pos2,
        rect: egui::Rect,
    ) -> bool {
        if let Some(hit) = self.hit_mask_polygon_at(layer_id, pointer_world, pointer_screen, rect) {
            let mut params = serde_json::Map::new();
            params.insert("id".to_string(), serde_json::json!(layer_id));
            params.insert("index".to_string(), serde_json::json!(hit.polygon_idx));
            params.insert(
                "vertex_index".to_string(),
                hit.vertex_idx
                    .map_or(serde_json::Value::Null, |index| serde_json::json!(index)),
            );
            self.submit_native_mask_command("viewer.masks.selection.set", params);
            true
        } else {
            self.commit_clear_mask_polygon_selection();
            false
        }
    }

    pub(super) fn commit_clear_mask_polygon_selection(&mut self) {
        self.submit_native_mask_command("viewer.masks.selection.clear", serde_json::Map::new());
    }

    pub(super) fn delete_selected_mask_polygon(&mut self) -> bool {
        self.validate_mask_polygon_selection();
        let Some(selection) = self.selected_mask_polygon else {
            return false;
        };
        let valid = self
            .mask_layers
            .iter()
            .find(|layer| layer.id == selection.layer_id)
            .is_some_and(|layer| selection.polygon_idx < layer.polygons_world.len());
        if !valid {
            self.clear_mask_polygon_selection();
            return false;
        }
        let mut params = serde_json::Map::new();
        params.insert("id".to_string(), serde_json::json!(selection.layer_id));
        params.insert(
            "index".to_string(),
            serde_json::json!(selection.polygon_idx),
        );
        self.submit_native_mask_command("viewer.masks.polygons.remove", params);
        true
    }

    pub(super) fn move_mask_polygon_vertex(
        &mut self,
        selection: MaskPolygonSelection,
        vertex_idx: usize,
        pointer_world: egui::Pos2,
    ) -> bool {
        let Some(layer) = self
            .mask_layers
            .iter_mut()
            .find(|layer| layer.id == selection.layer_id)
        else {
            self.clear_mask_polygon_selection();
            return false;
        };
        let Some(poly) = layer.polygons_world.get_mut(selection.polygon_idx) else {
            self.clear_mask_polygon_selection();
            return false;
        };
        let n = Self::mask_polygon_unique_vertex_count(poly);
        if n < 3 || vertex_idx >= n {
            self.clear_mask_polygon_selection();
            return false;
        }

        let local = pointer_world - layer.offset_world;
        poly[vertex_idx] = local;
        if vertex_idx == 0 && poly.len() > n {
            let last_idx = poly.len() - 1;
            poly[last_idx] = local;
        }
        layer.raster_display = None;
        true
    }

    pub(super) fn begin_mask_vertex_drag(
        &mut self,
        selection: MaskPolygonSelection,
        vertex_idx: usize,
    ) -> bool {
        let Some(start_polygon) = self
            .mask_layers
            .iter()
            .find(|layer| layer.id == selection.layer_id && layer.visible)
            .and_then(|layer| layer.polygons_world.get(selection.polygon_idx))
            .cloned()
        else {
            return false;
        };
        if vertex_idx >= Self::mask_polygon_unique_vertex_count(&start_polygon) {
            return false;
        }
        let start_selection = self.selected_mask_polygon;
        let start_selected_vertex = self.selected_mask_vertex;
        self.selected_mask_polygon = Some(selection);
        self.selected_mask_vertex = Some(vertex_idx);
        self.dragging_mask_vertex = Some(MaskVertexDrag {
            selection,
            vertex_idx,
            undo_recorded: false,
            start_polygon,
            start_selection,
            start_selected_vertex,
            actor_generation: self.control_actor_mask_generation,
        });
        true
    }

    pub(super) fn begin_mask_polygon_move(
        &mut self,
        selection: MaskPolygonSelection,
        pointer_world: egui::Pos2,
    ) -> bool {
        let Some(start_polygon) = self
            .mask_layers
            .iter()
            .find(|layer| layer.id == selection.layer_id && layer.visible)
            .and_then(|layer| layer.polygons_world.get(selection.polygon_idx))
            .cloned()
        else {
            self.clear_mask_polygon_selection();
            return false;
        };
        if Self::mask_polygon_unique_vertex_count(&start_polygon) < 3 {
            self.clear_mask_polygon_selection();
            return false;
        }

        let start_selection = self.selected_mask_polygon;
        let start_selected_vertex = self.selected_mask_vertex;
        self.selected_mask_polygon = Some(selection);
        self.selected_mask_vertex = None;
        self.dragging_mask_vertex = None;
        self.moving_mask_polygon = Some(MaskPolygonMoveState {
            selection,
            start_polygon,
            start_pointer_world: pointer_world,
            start_selection,
            start_selected_vertex,
            actor_generation: self.control_actor_mask_generation,
        });
        true
    }

    pub(super) fn move_mask_polygon_from_start(
        &mut self,
        state: &MaskPolygonMoveState,
        pointer_world: egui::Pos2,
    ) -> bool {
        let Some(layer) = self
            .mask_layers
            .iter_mut()
            .find(|layer| layer.id == state.selection.layer_id)
        else {
            self.clear_mask_polygon_selection();
            return false;
        };
        let Some(poly) = layer.polygons_world.get_mut(state.selection.polygon_idx) else {
            self.clear_mask_polygon_selection();
            return false;
        };

        let delta = pointer_world - state.start_pointer_world;
        *poly = state
            .start_polygon
            .iter()
            .copied()
            .map(|p| p + delta)
            .collect();
        layer.raster_display = None;
        true
    }

    pub(super) fn finish_mask_polygon_gesture(&mut self) -> bool {
        let baseline = if let Some(drag) = self.dragging_mask_vertex.take() {
            Some((
                drag.selection,
                drag.start_polygon,
                drag.start_selection,
                drag.start_selected_vertex,
                drag.actor_generation,
            ))
        } else {
            self.moving_mask_polygon.take().map(|drag| {
                (
                    drag.selection,
                    drag.start_polygon,
                    drag.start_selection,
                    drag.start_selected_vertex,
                    drag.actor_generation,
                )
            })
        };
        let Some((selection, start_polygon, start_selection, start_vertex, actor_generation)) =
            baseline
        else {
            return false;
        };

        let state = self.mask_semantic_state();
        self.native_command_ingress.push(NativeControlIntent {
            method: "viewer.masks.state.replace",
            params: serde_json::json!({
                "expected_generation": actor_generation.max(1),
                "sync_project": true,
                "state": state,
            }),
        });
        self.restore_mask_gesture_baseline(selection, start_polygon, start_selection, start_vertex);
        self.apply_pending_mask_projection_after_gesture();
        true
    }

    pub(super) fn cancel_mask_polygon_gesture(&mut self) -> bool {
        let baseline = if let Some(drag) = self.dragging_mask_vertex.take() {
            Some((
                drag.selection,
                drag.start_polygon,
                drag.start_selection,
                drag.start_selected_vertex,
            ))
        } else {
            self.moving_mask_polygon.take().map(|drag| {
                (
                    drag.selection,
                    drag.start_polygon,
                    drag.start_selection,
                    drag.start_selected_vertex,
                )
            })
        };
        let Some((selection, start_polygon, start_selection, start_vertex)) = baseline else {
            return false;
        };
        self.restore_mask_gesture_baseline(selection, start_polygon, start_selection, start_vertex);
        self.apply_pending_mask_projection_after_gesture();
        true
    }

    fn restore_mask_gesture_baseline(
        &mut self,
        selection: MaskPolygonSelection,
        start_polygon: Vec<egui::Pos2>,
        start_selection: Option<MaskPolygonSelection>,
        start_vertex: Option<usize>,
    ) {
        if let Some(layer) = self
            .mask_layers
            .iter_mut()
            .find(|layer| layer.id == selection.layer_id)
            && let Some(polygon) = layer.polygons_world.get_mut(selection.polygon_idx)
        {
            *polygon = start_polygon;
            layer.raster_display = None;
        }
        self.selected_mask_polygon = start_selection;
        self.selected_mask_vertex = start_vertex;
    }

    pub(super) fn apply_pending_mask_projection_after_gesture(&mut self) {
        if self.mask_polygon_gesture_active() {
            return;
        }
        if let Some(projection) = self.pending_control_actor_mask_projection.take() {
            let _ = self.apply_control_actor_masks_projection(&projection);
        }
    }
}
