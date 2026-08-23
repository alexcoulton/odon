use super::*;

impl OmeZarrViewerApp {
    pub(crate) fn control_projection_gesture_active(&self) -> bool {
        self.layer_move.is_some()
            || self.layer_transform.is_some()
            || self.mask_polygon_gesture_active()
            || self.draft_view_slice_level0.is_some()
    }

    pub(in crate::app) fn cancel_native_layer_gestures(&mut self) {
        if let Some(state) = self.layer_move.take() {
            self.apply_layer_offsets(&state.targets);
        }
        if let Some(state) = self.layer_transform.take()
            && let LayerId::Channel(channel) = state.layer
        {
            if let Some(value) = self.channel_offsets_world.get_mut(channel) {
                *value = state.start_offset_world;
            }
            if let Some(value) = self.channel_scales.get_mut(channel) {
                *value = state.start_scale;
            }
            if let Some(value) = self.channel_rotations_rad.get_mut(channel) {
                *value = state.start_rotation_rad;
            }
        }
    }

    pub(in crate::app) fn active_viewport_command_scope(&self) -> Option<(String, u64)> {
        if let Some(scope) = self.native_viewport_command_scope.as_ref() {
            return Some((
                scope.viewport_id.clone(),
                scope.presentation_revision.max(1),
            ));
        }
        let workspace = self.viewport_workspace.as_ref()?;
        let active = workspace.active();
        Some((
            active.id.as_str().to_string(),
            active.presentation_revision.max(1),
        ))
    }

    pub(in crate::app) fn submit_native_layer_visibility(
        &mut self,
        id: LayerId,
        visible: bool,
    ) -> bool {
        let Some((viewport_id, revision)) = self.active_viewport_command_scope() else {
            return false;
        };
        self.native_command_ingress.push(NativeControlIntent {
            method: "viewer.viewports.layers.set_visibility",
            params: serde_json::json!({
                "viewport_id":viewport_id,
                "layer_id":Self::layer_id_storage_key(id),
                "if_presentation_revision":revision,
                "visible":visible,
            }),
        });
        true
    }

    pub(in crate::app) fn submit_native_layer_visibilities(
        &mut self,
        ids: impl IntoIterator<Item = LayerId>,
        visible: bool,
    ) -> bool {
        let target_ids = ids
            .into_iter()
            .filter(|id| self.layer_is_available(*id))
            .map(Self::layer_id_storage_key)
            .collect::<HashSet<_>>();
        if target_ids.is_empty() {
            return false;
        }
        let mut state = self.control_native_layer_snapshot_list();
        let mut changed = false;
        if let Some(layers) = state.as_array_mut() {
            for layer in layers {
                if layer
                    .get("layer_id")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|id| target_ids.contains(id))
                {
                    changed |=
                        layer.get("visible").and_then(serde_json::Value::as_bool) != Some(visible);
                    layer["visible"] = serde_json::json!(visible);
                    layer["presentation"]["visible"] = serde_json::json!(visible);
                }
            }
        }
        changed && self.submit_native_layer_state_replace(state)
    }

    pub(in crate::app) fn submit_native_layer_active(&mut self, id: LayerId) -> bool {
        let Some((viewport_id, revision)) = self.active_viewport_command_scope() else {
            return false;
        };
        self.native_command_ingress.push(NativeControlIntent {
            method: "viewer.viewports.layers.set_active",
            params: serde_json::json!({
                "viewport_id":viewport_id,
                "layer_id":Self::layer_id_storage_key(id),
                "if_presentation_revision":revision,
            }),
        });
        true
    }

    pub(in crate::app) fn submit_native_layer_order(
        &mut self,
        stack: &'static str,
        layers: impl IntoIterator<Item = LayerId>,
    ) -> bool {
        let Some((viewport_id, revision)) = self.active_viewport_command_scope() else {
            return false;
        };
        let layers = layers
            .into_iter()
            .map(Self::layer_id_storage_key)
            .collect::<Vec<_>>();
        self.native_command_ingress.push(NativeControlIntent {
            method: "viewer.viewports.layers.set_order",
            params: serde_json::json!({
                "viewport_id":viewport_id,
                "stack":stack,
                "layers":layers,
                "if_presentation_revision":revision,
            }),
        });
        true
    }

    pub(in crate::app) fn submit_native_layer_state_replace(
        &mut self,
        state: serde_json::Value,
    ) -> bool {
        let Some((viewport_id, revision)) = self.active_viewport_command_scope() else {
            return false;
        };
        self.submit_native_layer_state_replace_at(&viewport_id, revision, state)
    }

    pub(in crate::app) fn submit_native_layer_state_replace_with_groups(
        &mut self,
        state: serde_json::Value,
        groups: &crate::data::project_config::ProjectLayerGroups,
    ) -> bool {
        let Some((viewport_id, revision)) = self.active_viewport_command_scope() else {
            return false;
        };
        self.native_command_ingress.push(NativeControlIntent {
            method: "viewer.viewports.layers.state.replace",
            params: serde_json::json!({
                "viewport_id":viewport_id,
                "if_presentation_revision":revision,
                "state":state,
                "channel_groups":channel_groups_snapshot(groups, &self.channels),
            }),
        });
        true
    }

    pub(in crate::app) fn submit_native_layer_state_replace_at(
        &mut self,
        viewport_id: &str,
        revision: u64,
        state: serde_json::Value,
    ) -> bool {
        self.native_command_ingress.push(NativeControlIntent {
            method: "viewer.viewports.layers.state.replace",
            params: serde_json::json!({
                "viewport_id":viewport_id,
                "if_presentation_revision":revision,
                "state":state,
            }),
        });
        true
    }

    pub(in crate::app) fn submit_native_channel_transform(
        &mut self,
        channel: usize,
        offset_world: Option<egui::Vec2>,
        scale: Option<egui::Vec2>,
        rotation_rad: Option<f32>,
    ) -> bool {
        let Some((viewport_id, revision)) = self.active_viewport_command_scope() else {
            return false;
        };
        self.submit_native_channel_transform_at(
            &viewport_id,
            revision,
            channel,
            offset_world,
            scale,
            rotation_rad,
        )
    }

    pub(in crate::app) fn submit_native_channel_transform_at(
        &mut self,
        viewport_id: &str,
        revision: u64,
        channel: usize,
        offset_world: Option<egui::Vec2>,
        scale: Option<egui::Vec2>,
        rotation_rad: Option<f32>,
    ) -> bool {
        let mut params = serde_json::json!({
            "viewport_id":viewport_id,
            "if_presentation_revision":revision.max(1),
            "channel":channel,
        });
        if let Some(offset) = offset_world {
            params["offset_world"] = serde_json::json!([offset.x, offset.y]);
        }
        if let Some(scale) = scale {
            params["scale"] = serde_json::json!([scale.x, scale.y]);
        }
        if let Some(rotation) = rotation_rad {
            params["rotation_rad"] = serde_json::json!(rotation);
        }
        self.native_command_ingress.push(NativeControlIntent {
            method: "viewer.channels.set_transform",
            params,
        });
        true
    }

    pub(in crate::app) fn commit_active_layer(&mut self, id: LayerId) {
        match id {
            LayerId::Mask(mask_id) if self.mask_layers.iter().any(|layer| layer.id == mask_id) => {
                self.submit_native_mask_active_layer(Some(mask_id));
            }
            _ if matches!(self.active_layer, LayerId::Mask(_)) => {
                self.submit_native_mask_active_layer(None);
            }
            _ => {}
        }
        self.submit_native_layer_active(id);
    }

    pub(in crate::app) fn set_active_layer(&mut self, id: LayerId) {
        self.set_active_layer_local(id);
    }

    pub(in crate::app) fn set_active_layer_local(&mut self, id: LayerId) {
        self.active_layer = id;
        if self
            .selected_mask_polygon
            .is_some_and(|selection| id != LayerId::Mask(selection.layer_id))
        {
            self.clear_mask_polygon_selection();
        }
        if let LayerId::Channel(idx) = id {
            self.selected_channel = idx.min(self.channels.len().saturating_sub(1));
            self.hist_dirty = true;
        } else {
            self.selected_channel_group_id = None;
        }
    }
}
