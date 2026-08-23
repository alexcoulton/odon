use super::*;

impl OmeZarrViewerApp {
    pub(super) fn clear_spatial_selection_drag(&mut self) {
        self.selection_rect_start_world = None;
        self.selection_rect_current_world = None;
        self.selection_lasso_world.clear();
    }

    pub(super) fn cancel_viewport_transient_gestures(&mut self) {
        self.cancel_mask_polygon_gesture();
        self.cancel_native_layer_gestures();
        self.clear_spatial_selection_drag();
        self.layer_drag = None;
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

    pub(super) fn commit_rect_selection_to_active_layer(
        &mut self,
        world_rect: egui::Rect,
        additive: bool,
    ) -> usize {
        let target = self.spatial_selection_target_layer();
        if target.is_some() {
            let offset = self.object_selection_target_offset(target);
            let local = world_rect.translate(-offset);
            let mut params = self.object_selection_target_params(target);
            params.insert(
                "world_rect".to_string(),
                serde_json::json!([local.min.x, local.min.y, local.max.x, local.max.y]),
            );
            params.insert(
                "mode".to_string(),
                serde_json::json!(if additive { "add" } else { "replace" }),
            );
            self.native_control_intents.push(NativeControlIntent {
                method: "viewer.objects.select_rect",
                params: serde_json::Value::Object(params),
            });
            return 0;
        }
        0
    }

    pub(super) fn commit_lasso_selection_to_active_layer(
        &mut self,
        world_points: &[egui::Pos2],
        additive: bool,
    ) -> usize {
        let target = self.spatial_selection_target_layer();
        if target.is_some() {
            let offset = self.object_selection_target_offset(target);
            let points = world_points
                .iter()
                .map(|point| [point.x - offset.x, point.y - offset.y])
                .collect::<Vec<_>>();
            let mut params = self.object_selection_target_params(target);
            params.insert("points".to_string(), serde_json::json!(points));
            params.insert(
                "mode".to_string(),
                serde_json::json!(if additive { "add" } else { "replace" }),
            );
            self.native_control_intents.push(NativeControlIntent {
                method: "viewer.objects.select_lasso",
                params: serde_json::Value::Object(params),
            });
            return 0;
        }
        0
    }

    fn object_selection_target_generation(&self, target: LayerId) -> u64 {
        match target {
            LayerId::SegmentationObjects => self.control_actor_object_selection_generation,
            LayerId::SpatialShape(id) => self
                .control_actor_secondary_object_selection_generations
                .get(&id)
                .copied()
                .unwrap_or(0),
            _ => 0,
        }
        .max(1)
    }

    fn object_selection_target_offset(&self, target: Option<LayerId>) -> egui::Vec2 {
        match target {
            Some(LayerId::SegmentationObjects) => self.seg_objects_offset_world,
            Some(LayerId::SpatialShape(id)) => self
                .spatial_layers
                .shapes
                .iter()
                .find(|layer| layer.id == id)
                .map(|layer| layer.offset_world)
                .unwrap_or_default(),
            _ => egui::Vec2::ZERO,
        }
    }

    pub(super) fn object_selection_target_params(
        &self,
        target: Option<LayerId>,
    ) -> serde_json::Map<String, serde_json::Value> {
        let mut params = serde_json::Map::new();
        match target {
            Some(LayerId::SpatialShape(id)) => {
                params.insert("target".to_string(), serde_json::json!("spatial_shape"));
                params.insert("layer_id".to_string(), serde_json::json!(id));
            }
            _ => {
                params.insert(
                    "target".to_string(),
                    serde_json::json!("segmentation_objects"),
                );
            }
        }
        if let Some(workspace) = self.viewport_workspace.as_ref() {
            params.insert(
                "viewport_id".to_string(),
                serde_json::json!(workspace.active_id().as_str()),
            );
        }
        params
    }

    pub(super) fn commit_point_selection_to_layer(
        &mut self,
        target: LayerId,
        world: egui::Pos2,
        additive: bool,
        toggle: bool,
    ) -> bool {
        let state = match target {
            LayerId::SegmentationObjects => self.seg_objects.control_selection_state_after_click(
                world,
                self.seg_objects_offset_world,
                &self.camera,
                additive,
                toggle,
            ),
            LayerId::SpatialShape(id) => {
                let Some(layer) = self
                    .spatial_layers
                    .shapes
                    .iter()
                    .find(|layer| layer.id == id)
                else {
                    return false;
                };
                let Some(objects) = layer.object_layer() else {
                    return false;
                };
                objects.control_selection_state_after_click(
                    world,
                    layer.offset_world,
                    &self.camera,
                    additive,
                    toggle,
                )
            }
            _ => return false,
        };
        let mut params = self.object_selection_target_params(Some(target));
        params.insert(
            "expected_generation".to_string(),
            serde_json::json!(self.object_selection_target_generation(target)),
        );
        params.insert("state".to_string(), state);
        self.native_control_intents.push(NativeControlIntent {
            method: "viewer.objects.selection.state.replace",
            params: serde_json::Value::Object(params),
        });
        true
    }

    pub(super) fn commit_id_selection_to_layer(
        &mut self,
        target: LayerId,
        ids: &[String],
        id_set: &HashSet<String>,
    ) -> Option<usize> {
        let matched = match target {
            LayerId::SegmentationObjects => self
                .seg_objects
                .has_data()
                .then(|| self.seg_objects.object_indices_matching_ids(id_set).len()),
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter()
                .find(|layer| layer.id == id)
                .and_then(|layer| layer.object_layer())
                .map(|objects| objects.object_indices_matching_ids(id_set).len()),
            _ => None,
        }?;

        let mut params = self.object_selection_target_params(Some(target));
        params.insert("ids".to_string(), serde_json::json!(ids));
        params.insert("mode".to_string(), serde_json::json!("replace"));
        self.native_control_intents.push(NativeControlIntent {
            method: "viewer.objects.selection.select_ids",
            params: serde_json::Value::Object(params),
        });
        Some(matched)
    }

    pub(super) fn commit_clear_object_selection(&mut self, target: LayerId) {
        self.native_control_intents.push(NativeControlIntent {
            method: "viewer.objects.clear_selection",
            params: serde_json::Value::Object(self.object_selection_target_params(Some(target))),
        });
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
