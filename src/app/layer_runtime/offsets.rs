use super::*;

impl OmeZarrViewerApp {
    pub(in crate::app) fn layer_offset_world(&self, id: LayerId) -> egui::Vec2 {
        match id {
            LayerId::Channel(idx) => self
                .channel_offsets_world
                .get(idx)
                .copied()
                .unwrap_or(egui::Vec2::ZERO),
            LayerId::SpatialImage(id) => self
                .spatial_image_layers
                .images
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.offset_world)
                .unwrap_or(egui::Vec2::ZERO),
            LayerId::Points => self.points_offset_world,
            LayerId::Annotation(id) => self
                .annotation_layers
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.offset_world)
                .unwrap_or(egui::Vec2::ZERO),
            LayerId::SpatialPoints => self.spatial_points_offset_world,
            LayerId::Mask(id) => self
                .mask_layers
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.offset_world)
                .unwrap_or(egui::Vec2::ZERO),
            LayerId::SegmentationLabels => self.seg_labels_offset_world,
            LayerId::SegmentationGeoJson => self.seg_geojson_offset_world,
            LayerId::SegmentationObjects => self.seg_objects_offset_world,
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter()
                .find(|s| s.id == id)
                .map(|s| s.offset_world)
                .unwrap_or(egui::Vec2::ZERO),
            LayerId::XeniumCells => self.xenium_cells_offset_world,
            LayerId::XeniumTranscripts => self.xenium_transcripts_offset_world,
        }
    }

    pub(in crate::app) fn layer_offset_world_mut(
        &mut self,
        id: LayerId,
    ) -> Option<&mut egui::Vec2> {
        match id {
            LayerId::Channel(idx) => self.channel_offsets_world.get_mut(idx),
            LayerId::SpatialImage(id) => self
                .spatial_image_layers
                .images
                .iter_mut()
                .find(|l| l.id == id)
                .map(|l| &mut l.offset_world),
            LayerId::Points => Some(&mut self.points_offset_world),
            LayerId::Annotation(id) => self
                .annotation_layers
                .iter_mut()
                .find(|l| l.id == id)
                .map(|l| &mut l.offset_world),
            LayerId::SpatialPoints => Some(&mut self.spatial_points_offset_world),
            LayerId::Mask(id) => self
                .mask_layers
                .iter_mut()
                .find(|l| l.id == id)
                .map(|l| &mut l.offset_world),
            LayerId::SegmentationLabels => Some(&mut self.seg_labels_offset_world),
            LayerId::SegmentationGeoJson => Some(&mut self.seg_geojson_offset_world),
            LayerId::SegmentationObjects => Some(&mut self.seg_objects_offset_world),
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|s| s.id == id)
                .map(|s| &mut s.offset_world),
            LayerId::XeniumCells => Some(&mut self.xenium_cells_offset_world),
            LayerId::XeniumTranscripts => Some(&mut self.xenium_transcripts_offset_world),
        }
    }

    pub(in crate::app) fn layer_has_offset_world(&self, id: LayerId) -> bool {
        match id {
            LayerId::Channel(idx) => idx < self.channel_offsets_world.len(),
            LayerId::SpatialImage(id) => {
                self.spatial_image_layers.images.iter().any(|l| l.id == id)
            }
            LayerId::Points => true,
            LayerId::Annotation(id) => self.annotation_layers.iter().any(|l| l.id == id),
            LayerId::SpatialPoints => true,
            LayerId::Mask(id) => self.mask_layers.iter().any(|l| l.id == id),
            LayerId::SegmentationLabels
            | LayerId::SegmentationGeoJson
            | LayerId::SegmentationObjects
            | LayerId::XeniumCells
            | LayerId::XeniumTranscripts => true,
            LayerId::SpatialShape(id) => self.spatial_layers.shapes.iter().any(|s| s.id == id),
        }
    }

    pub(in crate::app) fn current_offset_layer_ids(&self) -> Vec<LayerId> {
        let mut layers = (0..self.channel_offsets_world.len())
            .map(LayerId::Channel)
            .collect::<Vec<_>>();
        layers.extend(self.overlay_layer_order.iter().copied());
        Self::dedupe_layer_ids(layers)
    }

    pub(in crate::app) fn capture_loaded_layer_offsets(&mut self) {
        self.loaded_layer_offsets_world.clear();
        for layer in self.current_offset_layer_ids() {
            if self.layer_has_offset_world(layer) {
                self.loaded_layer_offsets_world
                    .insert(layer, self.layer_offset_world(layer));
            }
        }
    }

    pub(in crate::app) fn ensure_loaded_layer_offset_baselines(&mut self) {
        for layer in self.current_offset_layer_ids() {
            if self.layer_has_offset_world(layer) {
                let offset = self.layer_offset_world(layer);
                self.loaded_layer_offsets_world
                    .entry(layer)
                    .or_insert(offset);
            }
        }
    }

    pub(in crate::app) fn ensure_loaded_layer_offset_baselines_for(&mut self, layers: &[LayerId]) {
        self.ensure_loaded_layer_offset_baselines();
        for &layer in layers {
            if self.layer_has_offset_world(layer) {
                let offset = self.layer_offset_world(layer);
                self.loaded_layer_offsets_world
                    .entry(layer)
                    .or_insert(offset);
            }
        }
    }

    pub(in crate::app) fn restore_loaded_layer_offsets_from_project_view(
        &mut self,
        view: &ProjectRoiViewState,
    ) {
        self.loaded_layer_offsets_world.clear();
        for (idx, saved) in view.channels.iter().enumerate() {
            if let Some([x, y]) = saved.original_offset_world {
                self.loaded_layer_offsets_world
                    .insert(LayerId::Channel(idx), egui::vec2(x, y));
            }
        }
        for (id, [x, y]) in &view.overlay_original_offsets_world {
            if let Some(layer_id) = self.parse_layer_id_storage_key(id) {
                self.loaded_layer_offsets_world
                    .insert(layer_id, egui::vec2(*x, *y));
            }
        }
        self.ensure_loaded_layer_offset_baselines();
    }

    pub(in crate::app) fn restore_loaded_layer_offsets_from_current_project_view_or_capture(
        &mut self,
    ) {
        let saved_view = self
            .project_space
            .roi_view_state(&self.dataset.source)
            .cloned();
        if let Some(view) = saved_view.as_ref() {
            self.restore_loaded_layer_offsets_from_project_view(view);
        } else {
            self.capture_loaded_layer_offsets();
        }
    }

    pub(in crate::app) fn dedupe_layer_ids(layers: Vec<LayerId>) -> Vec<LayerId> {
        let mut seen = HashSet::new();
        layers
            .into_iter()
            .filter(|layer| seen.insert(*layer))
            .collect()
    }

    pub(in crate::app) fn filter_visible_movable_layers(
        &self,
        layers: Vec<LayerId>,
    ) -> Vec<LayerId> {
        Self::dedupe_layer_ids(layers)
            .into_iter()
            .filter(|&layer| {
                self.layer_has_offset_world(layer)
                    && self.layer_is_available(layer)
                    && self.layer_visible_value(layer).unwrap_or(false)
            })
            .collect()
    }

    pub(in crate::app) fn current_visible_move_target_layers(&self) -> Vec<LayerId> {
        let candidates = match self.active_layer {
            LayerId::Channel(idx) => {
                if let Some(group_id) = self.selected_channel_group_id {
                    self.channel_indices_in_group(group_id)
                        .into_iter()
                        .map(LayerId::Channel)
                        .collect()
                } else if self.selected_channel_layers.contains(&idx) {
                    self.selected_channel_layers
                        .iter()
                        .copied()
                        .map(LayerId::Channel)
                        .collect()
                } else {
                    vec![LayerId::Channel(idx)]
                }
            }
            layer if self.selected_overlay_layers.contains(&layer) => {
                self.selected_overlay_layers.iter().copied().collect()
            }
            layer => vec![layer],
        };
        self.filter_visible_movable_layers(candidates)
    }

    pub(in crate::app) fn current_visible_move_targets_have_moved(&mut self) -> bool {
        let targets = self.current_visible_move_target_layers();
        if targets.is_empty() {
            return false;
        }
        self.ensure_loaded_layer_offset_baselines_for(&targets);
        targets.iter().any(|&layer| {
            self.loaded_layer_offsets_world
                .get(&layer)
                .is_some_and(|baseline| {
                    (self.layer_offset_world(layer) - *baseline).length_sq() > 1e-12
                })
        })
    }

    pub(in crate::app) fn reset_current_visible_move_targets_to_loaded(&mut self) -> bool {
        let targets = self.current_visible_move_target_layers();
        if targets.is_empty() {
            self.set_status("No visible movable layers selected.");
            return false;
        }
        self.ensure_loaded_layer_offset_baselines_for(&targets);
        let reset_offsets = targets
            .iter()
            .filter_map(|&layer| {
                self.loaded_layer_offsets_world
                    .get(&layer)
                    .copied()
                    .map(|offset_world| LayerOffsetEntry {
                        layer,
                        offset_world,
                    })
            })
            .collect::<Vec<_>>();
        let will_change = reset_offsets.iter().any(|entry| {
            (self.layer_offset_world(entry.layer) - entry.offset_world).length_sq() > 1e-12
        });
        if !will_change {
            return false;
        }
        self.push_layer_offsets_undo_snapshot(&targets);
        if self.commit_layer_offsets(&reset_offsets) {
            self.bump_render_id();
            true
        } else {
            false
        }
    }

    pub(in crate::app) fn apply_layer_offsets(&mut self, offsets: &[LayerOffsetEntry]) -> bool {
        let mut changed = false;
        let mut mask_changed = false;
        for entry in offsets {
            if let Some(offset) = self.layer_offset_world_mut(entry.layer) {
                if (*offset - entry.offset_world).length_sq() > 1e-12 {
                    *offset = entry.offset_world;
                    changed = true;
                    mask_changed |= matches!(entry.layer, LayerId::Mask(_));
                }
            }
        }
        if changed {
            self.hist_dirty = true;
        }
        if mask_changed && !self.mask_actor_owned() {
            self.mark_mask_layers_project_dirty();
        }
        changed
    }

    pub(in crate::app) fn commit_layer_offsets(&mut self, offsets: &[LayerOffsetEntry]) -> bool {
        let mut state = self.control_native_layer_snapshot_list();
        let mut state_changed = false;
        let mut desired_masks = self.mask_actor_owned().then(|| self.mask_layers.clone());
        let mut mask_changed = false;
        for entry in offsets {
            if let (LayerId::Mask(id), Some(layers)) = (entry.layer, desired_masks.as_mut()) {
                if let Some(layer) = layers.iter_mut().find(|layer| layer.id == id)
                    && (layer.offset_world - entry.offset_world).length_sq() > 1e-12
                {
                    layer.offset_world = entry.offset_world;
                    mask_changed = true;
                }
                continue;
            }
            let layer_id = Self::layer_id_storage_key(entry.layer);
            if let Some(layer) = state.as_array_mut().and_then(|layers| {
                layers.iter_mut().find(|layer| {
                    layer.get("layer_id").and_then(serde_json::Value::as_str)
                        == Some(layer_id.as_str())
                })
            }) {
                let desired = serde_json::json!([entry.offset_world.x, entry.offset_world.y]);
                if layer.get("offset_world") != Some(&desired) {
                    layer["offset_world"] = desired;
                    state_changed = true;
                }
            }
        }
        if state_changed {
            self.submit_native_layer_state_replace(state);
        }
        if mask_changed {
            self.submit_native_mask_state_replace(
                desired_masks.as_deref().unwrap_or_default(),
                self.mask_selection_value(),
            );
        }
        state_changed || mask_changed
    }
}
