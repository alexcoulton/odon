use super::*;

impl OmeZarrViewerApp {
    pub(super) fn fit_to_last_canvas(&mut self) {
        let Some(viewport) = self.last_canvas_rect else {
            return;
        };
        self.fit_to_rect(viewport);
    }

    pub fn fit_to_viewport(&mut self, viewport: egui::Rect) {
        self.fit_to_rect(viewport);
    }

    pub(super) fn available_object_selection_targets(
        &self,
    ) -> Vec<(crate::spatialdata::PositiveCellSelectionTarget, String)> {
        let mut targets = Vec::new();
        if self.seg_objects.has_data() {
            targets.push((
                crate::spatialdata::PositiveCellSelectionTarget::SegmentationObjects,
                "Segmentation Objects".to_string(),
            ));
        }
        targets.extend(self.spatial_layers.positive_cell_selection_targets());
        targets
    }

    pub(super) fn select_objects_by_ids_target(
        &mut self,
        cell_ids: &[String],
        target: crate::spatialdata::PositiveCellSelectionTarget,
    ) -> Option<(usize, usize)> {
        let id_set = cell_ids.iter().cloned().collect::<HashSet<_>>();
        if id_set.is_empty() {
            return None;
        }

        let mut matched_layers = 0usize;
        let mut matched_objects = 0usize;
        match target {
            crate::spatialdata::PositiveCellSelectionTarget::SegmentationObjects => {
                let selected = self.seg_objects.select_objects_by_ids(&id_set);
                if selected > 0 {
                    matched_layers += 1;
                    matched_objects += selected;
                }
            }
            crate::spatialdata::PositiveCellSelectionTarget::AllObjectLayers => {
                let selected = self.seg_objects.select_objects_by_ids(&id_set);
                if selected > 0 {
                    matched_layers += 1;
                    matched_objects += selected;
                }
                if let Some((layers, objects)) = self
                    .spatial_layers
                    .select_positive_cells_by_ids(cell_ids, target)
                {
                    matched_layers += layers;
                    matched_objects += objects;
                }
            }
            crate::spatialdata::PositiveCellSelectionTarget::ShapeLayer(_) => {
                if let Some((layers, objects)) = self
                    .spatial_layers
                    .select_positive_cells_by_ids(cell_ids, target)
                {
                    matched_layers += layers;
                    matched_objects += objects;
                }
            }
        }

        (matched_layers > 0).then_some((matched_layers, matched_objects))
    }

    pub(super) fn fit_to_selected_seg_objects(&mut self) -> bool {
        let Some(viewport) = self.last_canvas_rect else {
            return false;
        };
        let off = self.layer_offset_world(LayerId::SegmentationObjects);
        let Some(world) = self.seg_objects.fit_bounds_world(off) else {
            return false;
        };
        self.camera.fit_to_world_rect(viewport, world);
        true
    }

    pub(super) fn fit_to_seg_object_index(&mut self, object_index: usize) -> bool {
        let Some(viewport) = self.last_canvas_rect else {
            return false;
        };
        let off = self.layer_offset_world(LayerId::SegmentationObjects);
        let Some(world) = self.seg_objects.fit_object_bounds_world(object_index, off) else {
            return false;
        };
        self.camera.fit_to_world_rect(viewport, world);
        true
    }

    pub(super) fn fit_to_rect(&mut self, viewport: egui::Rect) {
        let world = self.image_world_rect_lvl0();
        self.camera.fit_to_world_rect(viewport, world);
    }

    pub(super) fn choose_level(&self) -> usize {
        let Some(level0) = self.dataset.levels.first() else {
            return 0;
        };
        if !self.view_plane_is_xy() {
            let mut best = 0usize;
            let mut best_err = f32::INFINITY;
            for level in &self.dataset.levels {
                let Some((downsample_v, downsample_u)) =
                    display_downsample(&self.dataset.dims, level0, level, self.view_plane_mode)
                else {
                    continue;
                };
                let screen_per_level_px =
                    self.camera.zoom_screen_per_lvl0_px * (downsample_u * downsample_v).sqrt();
                let err = screen_per_level_px.ln().abs();
                if err < best_err {
                    best_err = err;
                    best = level.index;
                }
            }
            return best;
        }
        choose_level_auto(
            &self.dataset.levels,
            self.camera.zoom_screen_per_lvl0_px,
            1.0,
        )
    }

    pub(super) fn sort_tile_keys_near_center(
        &self,
        level_info: &crate::data::ome::LevelInfo,
        keys: &mut [TileKey],
    ) {
        // Request the tiles nearest the current viewport center first so zoom-in refines where the
        // user is looking before spending bandwidth on peripheral tiles.
        let Some(level0) = self.dataset.levels.first() else {
            return;
        };
        let Some(axes) = self.display_axes() else {
            return;
        };
        let y_dim = axes.vertical;
        let x_dim = axes.horizontal;
        let center_world = self.camera.center_world_lvl0;
        let center_local = self.primary_image_world_to_local(center_world);
        let (downsample_y, downsample_x) =
            display_downsample(&self.dataset.dims, level0, level_info, self.view_plane_mode)
                .unwrap_or((
                    level_info.downsample.max(1e-6),
                    level_info.downsample.max(1e-6),
                ));
        let center_lvl = egui::pos2(
            center_local.x / downsample_x.max(1e-6),
            center_local.y / downsample_y.max(1e-6),
        );
        let chunk_y = level_info.chunks[y_dim] as f32;
        let chunk_x = level_info.chunks[x_dim] as f32;

        keys.sort_by(|a, b| {
            let ay = (a.tile_y as f32 + 0.5) * chunk_y;
            let ax = (a.tile_x as f32 + 0.5) * chunk_x;
            let by = (b.tile_y as f32 + 0.5) * chunk_y;
            let bx = (b.tile_x as f32 + 0.5) * chunk_x;
            let da = (ax - center_lvl.x).powi(2) + (ay - center_lvl.y).powi(2);
            let db = (bx - center_lvl.x).powi(2) + (by - center_lvl.y).powi(2);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}
