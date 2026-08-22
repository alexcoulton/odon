use super::*;

impl OmeZarrViewerApp {
    pub(super) fn clear_spatial_selection_drag(&mut self) {
        self.selection_rect_start_world = None;
        self.selection_rect_current_world = None;
        self.selection_lasso_world.clear();
    }

    pub(super) fn cancel_viewport_transient_gestures(&mut self) {
        self.clear_spatial_selection_drag();
        self.dragging_mask_vertex = None;
        self.moving_mask_polygon = None;
        self.layer_drag = None;
        self.layer_move = None;
        self.layer_transform = None;
        self.drawing_mask_layer = None;
        self.drawing_mask_polygon.clear();
    }

    pub(super) fn switch_to_pan_if_analysis_interacted(&mut self, ui: &egui::Ui) {
        let interacted = ui.rect_contains_pointer(ui.max_rect())
            && ui.input(|i| i.pointer.any_pressed() || i.pointer.any_down());
        if interacted && matches!(self.tool_mode, ToolMode::Select | ToolMode::LassoSelect) {
            self.tool_mode = ToolMode::Pan;
            self.clear_spatial_selection_drag();
        }
    }

    pub(super) fn active_layer_supports_spatial_selection(&self) -> bool {
        self.spatial_selection_target_layer().is_some()
    }

    pub(super) fn spatial_selection_target_layer(&self) -> Option<LayerId> {
        if !self.view_plane_is_xy() {
            return None;
        }
        match self.active_layer {
            LayerId::SegmentationObjects => {
                (self.seg_objects.object_count() > 0).then_some(LayerId::SegmentationObjects)
            }
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter()
                .find(|layer| layer.id == id)
                .is_some_and(|layer| layer.has_object_layer())
                .then_some(LayerId::SpatialShape(id)),
            LayerId::Channel(_) | LayerId::SegmentationLabels | LayerId::SegmentationGeoJson => {
                (self.seg_objects.visible && self.seg_objects.object_count() > 0)
                    .then_some(LayerId::SegmentationObjects)
            }
            _ => None,
        }
    }

    pub(super) fn control_object_selection_target(
        &self,
        params: &serde_json::Value,
    ) -> Result<LayerId, String> {
        match params
            .get("target")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("active")
        {
            "active" => self.spatial_selection_target_layer().ok_or_else(|| {
                "active layer does not provide selectable objects in the current view".to_string()
            }),
            "objects" | "segmentation_objects" => {
                if self.seg_objects.object_count() > 0 {
                    Ok(LayerId::SegmentationObjects)
                } else {
                    Err("segmentation object layer is empty".to_string())
                }
            }
            "spatial_shape" => {
                let id = params
                    .get("layer_id")
                    .or_else(|| params.get("id"))
                    .and_then(serde_json::Value::as_u64)
                    .ok_or_else(|| "target='spatial_shape' requires layer_id".to_string())?;
                let id = id as u64;
                if self
                    .spatial_layers
                    .shapes
                    .iter()
                    .any(|layer| layer.id == id && layer.has_object_layer())
                {
                    Ok(LayerId::SpatialShape(id))
                } else {
                    Err(format!(
                        "spatial shape layer {id} was not found or has no objects"
                    ))
                }
            }
            other => Err(format!("unknown object selection target '{other}'")),
        }
    }

    pub(super) fn control_world_rect_from_params(
        &self,
        params: &serde_json::Value,
    ) -> Result<egui::Rect, String> {
        if let Some(values) = params
            .get("world_rect")
            .and_then(serde_json::Value::as_array)
        {
            return control_rect_from_array(values, "world_rect");
        }
        if let Some(values) = params
            .get("screen_rect")
            .and_then(serde_json::Value::as_array)
        {
            let screen_rect = control_rect_from_array(values, "screen_rect")?;
            let Some(viewport) = self.last_canvas_rect else {
                return Err("screen_rect requires an active canvas viewport".to_string());
            };
            let world_min = self.camera.screen_to_world(screen_rect.min, viewport);
            let world_max = self.camera.screen_to_world(screen_rect.max, viewport);
            return Ok(normalized_rect(world_min, world_max));
        }

        let x0 = params
            .get("min_x")
            .or_else(|| params.get("x0"))
            .and_then(serde_json::Value::as_f64)
            .map(|value| value as f32);
        let y0 = params
            .get("min_y")
            .or_else(|| params.get("y0"))
            .and_then(serde_json::Value::as_f64)
            .map(|value| value as f32);
        let x1 = params
            .get("max_x")
            .or_else(|| params.get("x1"))
            .and_then(serde_json::Value::as_f64)
            .map(|value| value as f32);
        let y1 = params
            .get("max_y")
            .or_else(|| params.get("y1"))
            .and_then(serde_json::Value::as_f64)
            .map(|value| value as f32);
        match (x0, y0, x1, y1) {
            (Some(x0), Some(y0), Some(x1), Some(y1)) => {
                Ok(normalized_rect(egui::pos2(x0, y0), egui::pos2(x1, y1)))
            }
            _ => Err("provide world_rect, screen_rect, or min_x/min_y/max_x/max_y".to_string()),
        }
    }

    pub(super) fn control_world_points_from_params(
        &self,
        params: &serde_json::Value,
    ) -> Result<Vec<egui::Pos2>, String> {
        let values = params
            .get("world_points")
            .or_else(|| params.get("points"))
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| "world_points is required".to_string())?;
        if values.len() < 3 {
            return Err("world_points must contain at least three points".to_string());
        }
        values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let point = value
                    .as_array()
                    .ok_or_else(|| format!("world_points[{index}] must be [x, y]"))?;
                if point.len() != 2 {
                    return Err(format!("world_points[{index}] must be [x, y]"));
                }
                let x = point[0]
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| format!("world_points[{index}][0] must be finite"))?;
                let y = point[1]
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| format!("world_points[{index}][1] must be finite"))?;
                Ok(egui::pos2(x as f32, y as f32))
            })
            .collect()
    }

    pub(super) fn apply_rect_selection_to_active_layer(
        &mut self,
        world_rect: egui::Rect,
        additive: bool,
    ) -> usize {
        match self.spatial_selection_target_layer() {
            Some(LayerId::SegmentationObjects) => self.seg_objects.select_in_world_rect(
                world_rect,
                self.seg_objects_offset_world,
                additive,
            ),
            Some(LayerId::SpatialShape(id)) => {
                let Some(layer) = self
                    .spatial_layers
                    .shapes
                    .iter_mut()
                    .find(|layer| layer.id == id)
                else {
                    return 0;
                };
                let offset_world = layer.offset_world;
                layer.object_layer_mut().map_or(0, |objects| {
                    objects.select_in_world_rect(world_rect, offset_world, additive)
                })
            }
            _ => 0,
        }
    }

    pub(super) fn apply_lasso_selection_to_active_layer(
        &mut self,
        world_points: &[egui::Pos2],
        additive: bool,
    ) -> usize {
        match self.spatial_selection_target_layer() {
            Some(LayerId::SegmentationObjects) => self.seg_objects.select_in_world_lasso(
                world_points,
                self.seg_objects_offset_world,
                additive,
            ),
            Some(LayerId::SpatialShape(id)) => {
                let Some(layer) = self
                    .spatial_layers
                    .shapes
                    .iter_mut()
                    .find(|layer| layer.id == id)
                else {
                    return 0;
                };
                let offset_world = layer.offset_world;
                layer.object_layer_mut().map_or(0, |objects| {
                    objects.select_in_world_lasso(world_points, offset_world, additive)
                })
            }
            _ => 0,
        }
    }

    pub(super) fn active_or_spatial_selection_layer(&self) -> LayerId {
        self.spatial_selection_target_layer()
            .unwrap_or(self.active_layer)
    }

    pub(super) fn active_object_selection_count(&self) -> usize {
        match self.active_or_spatial_selection_layer() {
            LayerId::SegmentationObjects => self.seg_objects.selection_count(),
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer())
                .map(|objects| objects.selection_count())
                .unwrap_or(0),
            _ => 0,
        }
    }

    pub(super) fn active_object_selection_elements_snapshot(&self) -> Vec<(usize, String, usize)> {
        match self.active_or_spatial_selection_layer() {
            LayerId::SegmentationObjects => self.seg_objects.selection_elements_snapshot(),
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer())
                .map(|objects| objects.selection_elements_snapshot())
                .unwrap_or_default(),
            _ => Vec::new(),
        }
    }

    pub(super) fn create_selection_element_from_active_selection(&mut self) -> usize {
        match self.active_or_spatial_selection_layer() {
            LayerId::SegmentationObjects => self
                .seg_objects
                .create_selection_element_from_current_selection_with_name(None),
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer_mut())
                .map(|objects| {
                    objects.create_selection_element_from_current_selection_with_name(None)
                })
                .unwrap_or(0),
            _ => 0,
        }
    }

    pub(super) fn add_active_selection_to_element(&mut self, element_idx: usize) -> usize {
        match self.active_or_spatial_selection_layer() {
            LayerId::SegmentationObjects => self
                .seg_objects
                .add_current_selection_to_element(element_idx),
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer_mut())
                .map(|objects| objects.add_current_selection_to_element(element_idx))
                .unwrap_or(0),
            _ => 0,
        }
    }

    pub(super) fn selected_channel_visible_data_rect_lvl0(
        &self,
        viewport: egui::Rect,
        ch_idx: usize,
    ) -> egui::Rect {
        let visible_world = self.visible_world_rect(viewport);
        let corners = [
            visible_world.left_top(),
            egui::pos2(visible_world.right(), visible_world.top()),
            visible_world.right_bottom(),
            egui::pos2(visible_world.left(), visible_world.bottom()),
        ];
        let pivot = self.image_world_rect_lvl0().center();
        let off = self
            .channel_offsets_world
            .get(ch_idx)
            .copied()
            .unwrap_or_default();
        let scale = self
            .channel_scales
            .get(ch_idx)
            .copied()
            .unwrap_or(egui::Vec2::splat(1.0));
        let rot = self
            .channel_rotations_rad
            .get(ch_idx)
            .copied()
            .unwrap_or(0.0);

        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for &corner in &corners {
            let local = inv_xform_world_point(corner, pivot, off, scale, rot);
            min_x = min_x.min(local.x);
            min_y = min_y.min(local.y);
            max_x = max_x.max(local.x);
            max_y = max_y.max(local.y);
        }

        egui::Rect::from_min_max(egui::pos2(min_x, min_y), egui::pos2(max_x, max_y))
            .intersect(self.image_local_rect_lvl0())
    }

    pub(super) fn selected_channel_local_to_world(
        &self,
        ch_idx: usize,
        local: egui::Pos2,
    ) -> egui::Pos2 {
        let pivot = self.image_world_rect_lvl0().center();
        let off = self
            .channel_offsets_world
            .get(ch_idx)
            .copied()
            .unwrap_or_default();
        let scale = self
            .channel_scales
            .get(ch_idx)
            .copied()
            .unwrap_or(egui::Vec2::splat(1.0));
        let rot = self
            .channel_rotations_rad
            .get(ch_idx)
            .copied()
            .unwrap_or(0.0);
        xform_screen_point(local, pivot, off, scale, rot)
    }
}
