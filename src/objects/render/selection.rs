//! Hover, querying, object selection, and selection-render state.

use super::*;

impl ObjectsLayer {
    pub fn hover_tooltip(
        &self,
        pointer_world: egui::Pos2,
        local_to_world_offset: egui::Vec2,
        camera: &crate::camera::Camera,
    ) -> Option<Vec<String>> {
        let objects = self.objects.as_ref()?;
        let local = self.world_to_local_point(pointer_world, local_to_world_offset);
        let idx = self.hover_object_index(local, pointer_world, local_to_world_offset, camera)?;
        let obj = objects.get(idx)?;
        let centroid_world = self.local_to_world_point(obj.centroid_world, local_to_world_offset);

        let mut lines = Vec::new();
        lines.push(format!("id: {}", obj.id));
        lines.push(format!("area_px: {:.2}", obj.area_px));
        lines.push(format!("perimeter_px: {:.2}", obj.perimeter_px));
        lines.push(format!(
            "centroid: ({:.2}, {:.2})",
            centroid_world.x, centroid_world.y
        ));

        const MAX_TOOLTIP_PROPERTIES: usize = 11;
        for (key, text) in self
            .loaded_property_display_pairs(idx, obj)
            .into_iter()
            .take(MAX_TOOLTIP_PROPERTIES)
        {
            lines.push(format!("{key}: {text}"));
        }

        Some(lines)
    }

    pub fn select_at(
        &mut self,
        pointer_world: egui::Pos2,
        local_to_world_offset: egui::Vec2,
        camera: &crate::camera::Camera,
        additive: bool,
        toggle: bool,
    ) {
        let (selected, primary) = self.selection_state_after_click(
            pointer_world,
            local_to_world_offset,
            camera,
            additive,
            toggle,
        );
        if selected == self.selected_object_indices && primary == self.selected_object_index {
            return;
        }
        self.selected_object_indices = selected;
        self.selected_object_index = primary;
        self.rebuild_selection_render_lods();
        self.clear_measurements();
        self.invalidate_table_cache();
    }

    pub(crate) fn control_selection_state_after_click(
        &self,
        pointer_world: egui::Pos2,
        local_to_world_offset: egui::Vec2,
        camera: &crate::camera::Camera,
        additive: bool,
        toggle: bool,
    ) -> serde_json::Value {
        let (selected, primary) = self.selection_state_after_click(
            pointer_world,
            local_to_world_offset,
            camera,
            additive,
            toggle,
        );
        let mut selected_indices = selected.into_iter().collect::<Vec<_>>();
        selected_indices.sort_unstable();
        serde_json::json!({
            "selected_indices":selected_indices,
            "primary_index":primary,
        })
    }

    pub(super) fn selection_state_after_click(
        &self,
        pointer_world: egui::Pos2,
        local_to_world_offset: egui::Vec2,
        camera: &crate::camera::Camera,
        additive: bool,
        toggle: bool,
    ) -> (HashSet<usize>, Option<usize>) {
        let local = self.world_to_local_point(pointer_world, local_to_world_offset);
        let hit = self.hover_object_index(local, pointer_world, local_to_world_offset, camera);
        let mut selected = self.selected_object_indices.clone();
        let mut primary = self.selected_object_index;
        match (hit, additive, toggle) {
            (Some(idx), _, true) => {
                if !selected.insert(idx) {
                    selected.remove(&idx);
                    if primary == Some(idx) {
                        primary = selected.iter().next().copied();
                    }
                } else {
                    primary = Some(idx);
                }
            }
            (Some(idx), true, false) => {
                selected.insert(idx);
                primary = Some(idx);
            }
            (Some(idx), false, false) => {
                primary = Some(idx);
                if !self.has_live_analysis_selection() {
                    selected.clear();
                    selected.insert(idx);
                }
            }
            (None, false, false) => {
                if self.has_live_analysis_selection() {
                    primary = None;
                } else {
                    selected.clear();
                    primary = None;
                }
            }
            (None, _, _) => {}
        }
        (selected, primary)
    }

    pub fn selection_count(&self) -> usize {
        self.selected_object_indices.len()
    }

    #[cfg(test)]
    pub fn selection_snapshot_json(
        &self,
        local_to_world_offset: egui::Vec2,
        limit: usize,
    ) -> serde_json::Value {
        let Some(objects) = self.objects.as_ref() else {
            return serde_json::json!({
                "object_count": 0,
                "selection_count": 0,
                "primary": null,
                "selected": [],
                "truncated": false,
            });
        };
        let mut indices = self
            .selected_object_indices
            .iter()
            .copied()
            .collect::<Vec<_>>();
        indices.sort_unstable();
        let selected = self.object_entries_json(objects, &indices, local_to_world_offset, limit);
        serde_json::json!({
            "object_count": objects.len(),
            "selection_count": indices.len(),
            "primary": self.selected_object_index.and_then(|idx| {
                objects
                    .get(idx)
                    .map(|obj| self.object_entry_json(idx, obj, local_to_world_offset))
            }),
            "selected": selected,
            "truncated": indices.len() > limit,
        })
    }

    pub fn selection_signature_json(&self) -> serde_json::Value {
        let signature = self
            .selected_object_indices
            .iter()
            .fold(0u64, |hash, index| {
                hash ^ (*index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
            });
        serde_json::json!({
            "selection_count": self.selected_object_indices.len(),
            "primary_index": self.selected_object_index,
            "signature": signature,
        })
    }

    pub fn filtered_count(&self) -> usize {
        self.filtered_ordered_indices
            .as_ref()
            .map(|indices| indices.len())
            .unwrap_or_else(|| self.object_count())
    }

    pub fn clear_selection(&mut self) {
        self.selected_object_indices.clear();
        self.selected_object_index = None;
        self.rebuild_selection_render_lods();
        self.clear_measurements();
        self.invalidate_table_cache();
    }

    pub(super) fn rebuild_selection_fill_state(&mut self, object_count: usize) {
        if object_count == 0 {
            self.selection_fill_state = Arc::new(Vec::new());
            return;
        }

        let mut state = vec![0u8; object_count];
        for idx in &self.selected_object_indices {
            if let Some(slot) = state.get_mut(*idx) {
                *slot = 128;
            }
        }
        if let Some(primary_idx) = self.selected_object_index
            && let Some(slot) = state.get_mut(primary_idx)
        {
            *slot = 255;
        }
        self.selection_fill_state = Arc::new(state);
    }

    pub(super) fn ensure_cpu_selection_fill_mesh(&mut self) {
        if !self.selection_cpu_overlay_dirty {
            return;
        }
        let Some(objects) = self.objects.as_ref() else {
            self.selected_fill_mesh = None;
            self.selection_cpu_overlay_dirty = false;
            return;
        };

        let mut selected = Vec::with_capacity(self.selected_object_indices.len());
        for idx in &self.selected_object_indices {
            if let Some(obj) = objects.get(*idx) {
                selected.push(obj.clone());
            }
        }
        self.selected_fill_mesh = if selected.is_empty() {
            None
        } else {
            build_selection_fill_mesh(&selected).ok()
        };
        self.selection_cpu_overlay_dirty = false;
    }

    #[cfg(test)]
    pub fn query_world_rect_snapshot_json(
        &self,
        world_rect: egui::Rect,
        local_to_world_offset: egui::Vec2,
        limit: usize,
    ) -> serde_json::Value {
        let indices = self.query_indices_in_world_rect(world_rect, local_to_world_offset);
        self.rect_query_snapshot_json(world_rect, local_to_world_offset, &indices, limit)
    }

    #[cfg(test)]
    pub fn select_in_world_rect_snapshot_json(
        &mut self,
        world_rect: egui::Rect,
        local_to_world_offset: egui::Vec2,
        additive: bool,
        limit: usize,
    ) -> serde_json::Value {
        self.select_in_world_rect_snapshot_json_mode(
            world_rect,
            local_to_world_offset,
            if additive { "add" } else { "replace" },
            limit,
        )
    }

    #[cfg(test)]
    pub fn select_in_world_rect_snapshot_json_mode(
        &mut self,
        world_rect: egui::Rect,
        local_to_world_offset: egui::Vec2,
        mode: &str,
        limit: usize,
    ) -> serde_json::Value {
        if !matches!(mode, "replace" | "add" | "remove" | "toggle") {
            return serde_json::json!({"error": "selection mode must be replace, add, remove, or toggle"});
        }
        let indices = self.query_indices_in_world_rect(world_rect, local_to_world_offset);
        let changed = self.apply_object_selection_mode(&indices, mode);
        serde_json::json!({
            "changed": changed,
            "query": self.rect_query_snapshot_json(
                world_rect,
                local_to_world_offset,
                &indices,
                limit,
            ),
            "selection": self.selection_snapshot_json(local_to_world_offset, limit),
        })
    }

    #[cfg(test)]
    pub fn select_in_world_lasso(
        &mut self,
        world_points: &[egui::Pos2],
        local_to_world_offset: egui::Vec2,
        additive: bool,
    ) -> usize {
        let indices = self.query_indices_in_world_lasso(world_points, local_to_world_offset);
        self.apply_selection_indices(&indices, additive);
        indices.len()
    }

    #[cfg(test)]
    pub(super) fn query_indices_in_world_rect(
        &self,
        world_rect: egui::Rect,
        local_to_world_offset: egui::Vec2,
    ) -> Vec<usize> {
        let Some(objects) = self.objects.as_ref() else {
            return Vec::new();
        };
        let local_rect = self.world_to_local_rect(world_rect, local_to_world_offset);
        let mut out = Vec::new();
        for (idx, obj) in objects.iter().enumerate() {
            if !self.is_index_visible(idx) {
                continue;
            }
            if object_intersects_rect_for_selection(obj, local_rect) {
                out.push(idx);
            }
        }
        out
    }

    #[cfg(test)]
    pub(super) fn rect_query_snapshot_json(
        &self,
        world_rect: egui::Rect,
        local_to_world_offset: egui::Vec2,
        indices: &[usize],
        limit: usize,
    ) -> serde_json::Value {
        let local_rect = self.world_to_local_rect(world_rect, local_to_world_offset);
        let objects = self.objects.as_ref();
        let hits = objects
            .map(|objects| self.object_entries_json(objects, indices, local_to_world_offset, limit))
            .unwrap_or_default();
        serde_json::json!({
            "world_rect": rect_json(world_rect),
            "local_rect": rect_json(local_rect),
            "match_count": indices.len(),
            "matches": hits,
            "truncated": indices.len() > limit,
        })
    }

    #[cfg(test)]
    pub(super) fn object_entries_json(
        &self,
        objects: &[ObjectFeature],
        indices: &[usize],
        local_to_world_offset: egui::Vec2,
        limit: usize,
    ) -> Vec<serde_json::Value> {
        indices
            .iter()
            .take(limit)
            .filter_map(|idx| {
                objects
                    .get(*idx)
                    .map(|obj| self.object_entry_json(*idx, obj, local_to_world_offset))
            })
            .collect()
    }

    #[cfg(test)]
    pub(super) fn object_entry_json(
        &self,
        idx: usize,
        obj: &ObjectFeature,
        local_to_world_offset: egui::Vec2,
    ) -> serde_json::Value {
        let centroid_world = self.local_to_world_point(obj.centroid_world, local_to_world_offset);
        let bbox_min_world = self.local_to_world_point(obj.bbox_world.min, local_to_world_offset);
        let bbox_max_world = self.local_to_world_point(obj.bbox_world.max, local_to_world_offset);
        let bbox_world = egui::Rect::from_min_max(
            egui::pos2(
                bbox_min_world.x.min(bbox_max_world.x),
                bbox_min_world.y.min(bbox_max_world.y),
            ),
            egui::pos2(
                bbox_min_world.x.max(bbox_max_world.x),
                bbox_min_world.y.max(bbox_max_world.y),
            ),
        );
        serde_json::json!({
            "index": idx,
            "id": obj.id.as_str(),
            "centroid_world": [centroid_world.x, centroid_world.y],
            "centroid_local": [obj.centroid_world.x, obj.centroid_world.y],
            "bbox_world": rect_json(bbox_world),
            "bbox_local": rect_json(obj.bbox_world),
            "area_px": obj.area_px,
            "perimeter_px": obj.perimeter_px,
        })
    }

    #[cfg(test)]
    pub(super) fn query_indices_in_world_lasso(
        &self,
        world_points: &[egui::Pos2],
        local_to_world_offset: egui::Vec2,
    ) -> Vec<usize> {
        if world_points.len() < 3 {
            return Vec::new();
        }
        let Some(objects) = self.objects.as_ref() else {
            return Vec::new();
        };
        let Some(bins) = self.bins.as_ref() else {
            return Vec::new();
        };

        let local_points = world_points
            .iter()
            .copied()
            .map(|point| self.world_to_local_point(point, local_to_world_offset))
            .collect::<Vec<_>>();
        let mut min = local_points[0];
        let mut max = local_points[0];
        for point in local_points.iter().copied().skip(1) {
            min.x = min.x.min(point.x);
            min.y = min.y.min(point.y);
            max.x = max.x.max(point.x);
            max.y = max.y.max(point.y);
        }
        let local_bounds = egui::Rect::from_min_max(min, max);

        let (bx0, by0, bx1, by1) = bins.bin_range_for_world_rect(local_bounds);
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for by in by0..=by1 {
            for bx in bx0..=bx1 {
                let bi = by * bins.bins_w + bx;
                for &idx_u32 in bins.bin_slice(bi) {
                    let idx = idx_u32 as usize;
                    if !seen.insert(idx) || !self.is_index_visible(idx) {
                        continue;
                    }
                    let Some(obj) = objects.get(idx) else {
                        continue;
                    };
                    if !local_bounds.contains(obj.centroid_world) {
                        continue;
                    }
                    if point_in_polygon(obj.centroid_world, &local_points) {
                        out.push(idx);
                    }
                }
            }
        }
        out.sort_unstable();
        out
    }

    pub fn fit_bounds_world(&self, local_to_world_offset: egui::Vec2) -> Option<egui::Rect> {
        let objects = self.objects.as_ref()?;
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut any = false;

        for idx in &self.selected_object_indices {
            let Some(obj) = objects.get(*idx) else {
                continue;
            };
            any = true;
            let min = self.local_to_world_point(obj.bbox_world.min, local_to_world_offset);
            let max = self.local_to_world_point(obj.bbox_world.max, local_to_world_offset);
            min_x = min_x.min(min.x);
            min_y = min_y.min(min.y);
            max_x = max_x.max(max.x);
            max_y = max_y.max(max.y);
        }

        any.then(|| {
            let rect = egui::Rect::from_min_max(egui::pos2(min_x, min_y), egui::pos2(max_x, max_y));
            let pad = rect.size().max_elem().max(32.0) * 0.08;
            rect.expand(pad)
        })
    }

    pub fn fit_object_bounds_world(
        &self,
        object_index: usize,
        local_to_world_offset: egui::Vec2,
    ) -> Option<egui::Rect> {
        let obj = self.objects.as_ref()?.get(object_index)?;
        let min = self.local_to_world_point(obj.bbox_world.min, local_to_world_offset);
        let max = self.local_to_world_point(obj.bbox_world.max, local_to_world_offset);
        let rect = egui::Rect::from_min_max(min, max);
        let pad = rect.size().max_elem().max(32.0) * 0.08;
        Some(rect.expand(pad))
    }

    pub(in crate::objects) fn rebuild_selection_render_lods(&mut self) {
        let selected_count = self.selected_object_indices.len();
        if self.display_mode == ObjectDisplayMode::Points {
            self.selected_render_lods = None;
            self.primary_selected_render_lods = None;
            self.selected_fill_mesh = None;
            self.selection_fill_state = Arc::new(Vec::new());
            self.selection_cpu_overlay_dirty = false;
            let Some(objects) = self.objects.as_ref() else {
                self.selected_point_positions_world = None;
                self.selected_point_values = None;
                self.selected_point_lods = None;
                self.primary_selected_point_positions_world = None;
                self.primary_selected_point_values = None;
                self.selection_generation = self.selection_generation.wrapping_add(1).max(1);
                return;
            };
            if selected_count == 0 || selected_count > Self::SELECTED_RENDER_LOD_LIMIT {
                self.selected_point_positions_world = None;
                self.selected_point_values = None;
                self.selected_point_lods = None;
            } else {
                let selected = self
                    .selected_object_indices
                    .iter()
                    .filter_map(|idx| objects.get(*idx).cloned())
                    .collect::<Vec<_>>();
                let (positions, values, lods) =
                    build_object_point_payload(&selected, self.display_transform);
                self.selected_point_positions_world = Some(positions);
                self.selected_point_values = Some(values);
                self.selected_point_lods = Some(lods);
            }
            if let Some(primary) = self.selected_object_index.and_then(|idx| objects.get(idx)) {
                let (positions, values, _) = build_object_point_payload(
                    std::slice::from_ref(primary),
                    self.display_transform,
                );
                self.primary_selected_point_positions_world = Some(positions);
                self.primary_selected_point_values = Some(values);
            } else {
                self.primary_selected_point_positions_world = None;
                self.primary_selected_point_values = None;
            }
            self.selection_generation = self.selection_generation.wrapping_add(1).max(1);
            return;
        }
        let Some(objects) = self.objects.as_ref() else {
            self.selected_render_lods = None;
            self.primary_selected_render_lods = None;
            self.selected_fill_mesh = None;
            self.selected_point_positions_world = None;
            self.selected_point_values = None;
            self.selected_point_lods = None;
            self.selection_fill_state = Arc::new(Vec::new());
            self.selection_cpu_overlay_dirty = false;
            self.selection_generation = self.selection_generation.wrapping_add(1).max(1);
            return;
        };
        self.selected_point_positions_world = None;
        self.selected_point_values = None;
        self.selected_point_lods = None;
        let object_count = objects.len();
        self.selected_render_lods =
            if selected_count == 0 || selected_count > Self::SELECTED_RENDER_LOD_LIMIT {
                None
            } else {
                let selected = self
                    .selected_object_indices
                    .iter()
                    .filter_map(|idx| objects.get(*idx).cloned())
                    .collect::<Vec<_>>();
                build_render_lods(&selected).ok()
            };
        self.primary_selected_render_lods = self
            .selected_object_index
            .and_then(|idx| objects.get(idx))
            .and_then(|object| build_render_lods(std::slice::from_ref(object)).ok());
        self.selected_fill_mesh = None;
        self.rebuild_selection_fill_state(object_count);
        self.selection_cpu_overlay_dirty = true;

        self.selection_generation = self.selection_generation.wrapping_add(1).max(1);
    }

    pub(super) fn hover_object_index(
        &self,
        pointer_local: egui::Pos2,
        pointer_world: egui::Pos2,
        local_to_world_offset: egui::Vec2,
        camera: &crate::camera::Camera,
    ) -> Option<usize> {
        let objects = self.objects.as_ref()?;
        let bins = self.bins.as_ref()?;
        if self.display_mode == ObjectDisplayMode::Points {
            let radius_world = self.point_pick_radius_world(camera);
            let scale = self.display_scale();
            let rect = egui::Rect::from_min_max(
                egui::pos2(
                    pointer_local.x - radius_world / scale.x.max(1e-6),
                    pointer_local.y - radius_world / scale.y.max(1e-6),
                ),
                egui::pos2(
                    pointer_local.x + radius_world / scale.x.max(1e-6),
                    pointer_local.y + radius_world / scale.y.max(1e-6),
                ),
            );
            let (bx0, by0, bx1, by1) = bins.bin_range_for_world_rect(rect);
            let mut best_idx: Option<usize> = None;
            let mut best_dist_sq = f32::INFINITY;
            for by in by0..=by1 {
                for bx in bx0..=bx1 {
                    let bi = by * bins.bins_w + bx;
                    for &idx_u32 in bins.bin_slice(bi) {
                        let idx = idx_u32 as usize;
                        let Some(obj) = objects.get(idx) else {
                            continue;
                        };
                        if !self.is_index_visible(idx) {
                            continue;
                        }
                        let centroid_world =
                            self.local_to_world_point(obj.centroid_world, local_to_world_offset);
                        let dist_sq = centroid_world.distance_sq(pointer_world);
                        if dist_sq <= radius_world * radius_world && dist_sq < best_dist_sq {
                            best_dist_sq = dist_sq;
                            best_idx = Some(idx);
                        }
                    }
                }
            }
            return best_idx;
        }

        let rect = egui::Rect::from_center_size(pointer_local, egui::vec2(1.0, 1.0));
        let (bx0, by0, bx1, by1) = bins.bin_range_for_world_rect(rect);
        let mut best_idx: Option<usize> = None;
        let mut best_area = f32::INFINITY;

        for by in by0..=by1 {
            for bx in bx0..=bx1 {
                let bi = by * bins.bins_w + bx;
                for &idx_u32 in bins.bin_slice(bi) {
                    let idx = idx_u32 as usize;
                    let Some(obj) = objects.get(idx) else {
                        continue;
                    };
                    if !self.is_index_visible(idx) {
                        continue;
                    }
                    if !obj.bbox_world.contains(pointer_local) {
                        continue;
                    }
                    if !point_in_any_polygon(pointer_local, &obj.polygons_world) {
                        continue;
                    }
                    if obj.area_px < best_area {
                        best_area = obj.area_px;
                        best_idx = Some(idx);
                    }
                }
            }
        }

        best_idx
    }
}
