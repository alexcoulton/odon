use super::*;

impl OmeZarrViewerApp {
    pub(super) fn delete_mask_layer(&mut self, layer_id: u64) -> bool {
        let Some(idx) = self.mask_layers.iter().position(|l| l.id == layer_id) else {
            return false;
        };

        self.push_mask_undo_snapshot();
        self.mask_layers.remove(idx);
        self.mark_mask_layers_project_dirty();
        if self
            .selected_mask_polygon
            .is_some_and(|selection| selection.layer_id == layer_id)
        {
            self.clear_mask_polygon_selection();
        }
        if self.drawing_mask_layer == Some(layer_id) {
            self.drawing_mask_layer = None;
            self.drawing_mask_polygon.clear();
        }
        self.layer_drag = None;
        if self.active_layer == LayerId::Mask(layer_id) {
            self.active_layer = if !self.channels.is_empty() {
                LayerId::Channel(self.selected_channel.min(self.channels.len() - 1))
            } else {
                LayerId::Points
            };
        }
        self.rebuild_layer_orders();
        self.bump_render_id();
        true
    }

    pub(super) fn sync_mask_layers_into_project_space(&mut self) {
        if !self.mask_layers_project_dirty {
            return;
        }
        let Some(local_root) = self.dataset.source.local_path() else {
            return;
        };
        let layers = self.mask_layers.iter().map(|l| l.to_project()).collect();
        self.project_space.set_roi_mask_layers(local_root, layers);
        self.mask_layers_project_dirty = false;
    }

    pub(super) fn restore_mask_layers_from_project_space(&mut self) {
        self.undo_stack.clear();
        let Some(local_root) = self.dataset.source.local_path() else {
            self.mask_layers.clear();
            self.next_mask_layer_id = 1;
            self.clear_mask_polygon_selection();
            self.mask_layers_project_dirty = false;
            return;
        };
        let Some(layers) = self.project_space.roi_mask_layers(local_root) else {
            self.mask_layers.clear();
            self.next_mask_layer_id = 1;
            self.clear_mask_polygon_selection();
            self.mask_layers_project_dirty = false;
            return;
        };

        self.mask_layers = layers.iter().map(MaskLayer::from_project).collect();
        self.next_mask_layer_id = self
            .mask_layers
            .iter()
            .map(|l| l.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
            .max(1);

        // Reset any in-progress drawing if it no longer targets a valid layer.
        if let Some(id) = self.drawing_mask_layer {
            if !self.mask_layers.iter().any(|l| l.id == id) {
                self.drawing_mask_layer = None;
                self.drawing_mask_polygon.clear();
            }
        }

        self.rebuild_layer_orders();
        self.bump_render_id();
        self.mask_layers_project_dirty = false;
    }

    pub(super) fn ensure_editable_mask_layer(&mut self) -> u64 {
        if let LayerId::Mask(id) = self.active_layer {
            if let Some(l) = self.mask_layers.iter().find(|l| l.id == id) {
                if l.editable {
                    return id;
                }
            }
        }

        if let Some(l) = self.mask_layers.iter().rev().find(|l| l.editable) {
            return l.id;
        }

        self.create_editable_mask_layer(None)
    }

    pub(super) fn push_undo_action(&mut self, action: UndoAction) {
        self.undo_stack.push(action);
        if self.undo_stack.len() > MAX_UNDO_ACTIONS {
            self.undo_stack.remove(0);
        }
    }

    pub(super) fn mark_mask_layers_project_dirty(&mut self) {
        self.mask_layers_project_dirty = true;
    }

    pub(super) fn push_mask_undo_snapshot(&mut self) {
        self.push_undo_action(UndoAction::Mask(MaskUndoSnapshot {
            layers: self.mask_layers.clone(),
            next_layer_id: self.next_mask_layer_id,
            active_layer: self.active_layer,
            selection: self.selected_mask_polygon,
            selected_vertex: self.selected_mask_vertex,
            drawing_layer: self.drawing_mask_layer,
            drawing_polygon: self.drawing_mask_polygon.clone(),
        }));
    }

    pub(super) fn push_layer_offsets_undo_snapshot(&mut self, layers: &[LayerId]) {
        let offsets = layers
            .iter()
            .copied()
            .filter(|&layer| self.layer_has_offset_world(layer))
            .map(|layer| LayerOffsetEntry {
                layer,
                offset_world: self.layer_offset_world(layer),
            })
            .collect::<Vec<_>>();
        if !offsets.is_empty() {
            self.push_undo_action(UndoAction::LayerOffsets(LayerOffsetUndoSnapshot {
                offsets,
            }));
        }
    }

    pub(super) fn undo_last_edit(&mut self) -> bool {
        let Some(action) = self.undo_stack.pop() else {
            return false;
        };
        match action {
            UndoAction::Mask(snapshot) => {
                self.mask_layers = snapshot.layers;
                self.next_mask_layer_id = snapshot.next_layer_id;
                self.active_layer = snapshot.active_layer;
                self.selected_mask_polygon = snapshot.selection;
                self.selected_mask_vertex = snapshot.selected_vertex;
                self.dragging_mask_vertex = None;
                self.moving_mask_polygon = None;
                self.drawing_mask_layer = snapshot.drawing_layer;
                self.drawing_mask_polygon = snapshot.drawing_polygon;
                self.validate_mask_polygon_selection();
                self.rebuild_layer_orders();
                self.mark_mask_layers_project_dirty();
            }
            UndoAction::LayerOffsets(snapshot) => {
                for entry in snapshot.offsets {
                    if let Some(offset) = self.layer_offset_world_mut(entry.layer) {
                        *offset = entry.offset_world;
                    }
                }
                self.hist_dirty = true;
                self.mark_mask_layers_project_dirty();
            }
        }
        true
    }

    pub(super) fn request_native_mask_undo(&mut self) -> bool {
        if self.control_actor_mask_generation > 0 && matches!(self.active_layer, LayerId::Mask(_)) {
            if !self.control_actor_mask_undo_available {
                return false;
            }
            self.native_control_intents.push(NativeControlIntent {
                method: "viewer.masks.undo",
                params: serde_json::json!({}),
            });
            self.native_mask_actor_intent_emitted = true;
            return true;
        }
        self.undo_last_edit()
    }

    pub(super) fn mask_undo_available(&self) -> bool {
        if self.control_actor_mask_generation > 0 && matches!(self.active_layer, LayerId::Mask(_)) {
            self.control_actor_mask_undo_available
        } else {
            !self.undo_stack.is_empty()
        }
    }

    pub(super) fn finish_drawing_mask_polygon(&mut self) -> bool {
        if self.drawing_mask_polygon.len() < 3 {
            return false;
        }

        self.push_mask_undo_snapshot();
        let vertices = std::mem::take(&mut self.drawing_mask_polygon);
        let id = self
            .drawing_mask_layer
            .unwrap_or_else(|| self.ensure_editable_mask_layer());
        self.drawing_mask_layer = Some(id);
        if let Some(layer) = self.mask_layers.iter_mut().find(|l| l.id == id) {
            layer.add_closed_polygon(vertices);
            layer.visible = true;
            self.mark_mask_layers_project_dirty();
            true
        } else {
            false
        }
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
            self.selected_mask_polygon = Some(MaskPolygonSelection {
                layer_id,
                polygon_idx: hit.polygon_idx,
            });
            self.selected_mask_vertex = hit.vertex_idx;
            true
        } else {
            self.clear_mask_polygon_selection();
            false
        }
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
        self.push_mask_undo_snapshot();
        let Some(layer) = self
            .mask_layers
            .iter_mut()
            .find(|layer| layer.id == selection.layer_id)
        else {
            self.clear_mask_polygon_selection();
            return false;
        };
        layer.polygons_world.remove(selection.polygon_idx);
        layer.raster_display = None;
        self.clear_mask_polygon_selection();
        self.mark_mask_layers_project_dirty();
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
        self.mark_mask_layers_project_dirty();
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

        self.push_mask_undo_snapshot();
        self.selected_mask_polygon = Some(selection);
        self.selected_mask_vertex = None;
        self.dragging_mask_vertex = None;
        self.moving_mask_polygon = Some(MaskPolygonMoveState {
            selection,
            start_polygon,
            start_pointer_world: pointer_world,
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
        self.mark_mask_layers_project_dirty();
        true
    }

    pub(super) fn create_editable_mask_layer(&mut self, name: Option<String>) -> u64 {
        let base = "Masks";
        let mut name = name.unwrap_or_else(|| base.to_string());
        if self
            .mask_layers
            .iter()
            .any(|l| l.name.eq_ignore_ascii_case(&name))
        {
            let mut i = 2;
            loop {
                let candidate = format!("{base} {i}");
                if !self
                    .mask_layers
                    .iter()
                    .any(|l| l.name.eq_ignore_ascii_case(&candidate))
                {
                    name = candidate;
                    break;
                }
                i += 1;
            }
        }

        let id = self.next_mask_layer_id.max(1);
        self.next_mask_layer_id = id.saturating_add(1);
        self.mask_layers.push(MaskLayer {
            id,
            name,
            visible: true,
            opacity: 0.9,
            width_screen_px: 2.0,
            display_mode: MaskDisplayMode::default_new_layer(),
            color_rgb: [255, 210, 60],
            offset_world: egui::Vec2::ZERO,
            editable: true,
            polygons_world: Vec::new(),
            raster_display: None,
            source_geojson: None,
        });
        self.mark_mask_layers_project_dirty();
        id
    }
}
