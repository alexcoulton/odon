use super::super::*;

impl OmeZarrViewerApp {
    pub fn control_channel_snapshot(&self) -> serde_json::Value {
        serde_json::Value::Array(
            self.channels
                .iter()
                .enumerate()
                .map(|(idx, ch)| {
                    let [r, g, b] = ch.color_rgb;
                    let window = ch
                        .window
                        .map(|(lo, hi)| serde_json::json!({"min": lo, "max": hi}))
                        .unwrap_or(serde_json::Value::Null);
                    serde_json::json!({
                        "index": idx,
                        "name": ch.name,
                        "visible": ch.visible,
                        "selected": idx == self.selected_channel,
                        "color_rgb": [r, g, b],
                        "window": window,
                        "note": ch.note,
                    })
                })
                .collect(),
        )
    }

    #[cfg(test)]
    pub fn control_visible_channel_snapshot(&self) -> serde_json::Value {
        serde_json::Value::Array(
            self.channels
                .iter()
                .enumerate()
                .filter(|(_, ch)| ch.visible)
                .map(|(idx, ch)| {
                    serde_json::json!({
                        "index": idx,
                        "name": ch.name,
                        "selected": idx == self.selected_channel,
                    })
                })
                .collect(),
        )
    }

    #[cfg(test)]
    pub fn control_plane_snapshot(&self) -> serde_json::Value {
        let selection = self.active_view_selection();
        let supported = self
            .view_plane_modes()
            .into_iter()
            .map(|mode| mode.label().to_ascii_lowercase())
            .collect::<Vec<_>>();
        serde_json::json!({
            "mode": selection.mode.label().to_ascii_lowercase(),
            "slice": selection.slice_level0,
            "slice_axis": selection.mode.slice_axis_label().to_ascii_lowercase(),
            "extent": self.view_slice_extent_level0().unwrap_or(1),
            "supported_modes": supported,
            "xy_only_operations_available": selection.mode == ViewPlaneMode::Xy,
        })
    }

    #[cfg(test)]
    pub fn control_set_plane(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let before = self.active_view_selection();
        if let Some(mode) = params.get("mode").and_then(serde_json::Value::as_str) {
            let mode = match mode.to_ascii_lowercase().as_str() {
                "xy" => ViewPlaneMode::Xy,
                "xz" => ViewPlaneMode::Xz,
                "yz" => ViewPlaneMode::Yz,
                _ => return serde_json::json!({"error": "mode must be 'xy', 'xz', or 'yz'"}),
            };
            if !self.view_plane_modes().contains(&mode) {
                return serde_json::json!({
                    "error": format!("{} view is not available for this dataset", mode.label()),
                });
            }
            self.set_view_plane_mode(mode);
        }
        if let Some(slice) = params.get("slice").and_then(serde_json::Value::as_u64) {
            self.set_active_view_slice_level0(slice);
        }
        let after = self.active_view_selection();
        serde_json::json!({
            "changed": before != after,
            "plane": self.control_plane_snapshot(),
        })
    }

    #[cfg(test)]
    pub fn control_step_plane(
        &mut self,
        params: &serde_json::Value,
        forward: bool,
    ) -> serde_json::Value {
        let step = params
            .get("step")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(1);
        let wrap = params
            .get("wrap")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let before = self.active_view_selection();
        let extent = self.view_slice_extent_level0().unwrap_or(1).max(1);
        let last = extent.saturating_sub(1);
        let next = if wrap {
            let offset = step % extent;
            if forward {
                (before.slice_level0 + offset) % extent
            } else {
                (before.slice_level0 + extent - offset) % extent
            }
        } else if forward {
            before.slice_level0.saturating_add(step).min(last)
        } else {
            before.slice_level0.saturating_sub(step)
        };
        self.set_active_view_slice_level0(next);
        serde_json::json!({
            "changed": before.slice_level0 != next,
            "plane": self.control_plane_snapshot(),
        })
    }

    #[cfg(test)]
    pub fn control_active_channel_snapshot(&self) -> serde_json::Value {
        self.channels
            .get(self.selected_channel)
            .map(|ch| {
                serde_json::json!({
                    "index": self.selected_channel,
                    "name": ch.name,
                    "visible": ch.visible,
                    "note": ch.note,
                })
            })
            .unwrap_or(serde_json::Value::Null)
    }

    pub fn control_side_panels_snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "left": self.show_left_panel,
            "right": self.show_right_panel,
        })
    }

    #[cfg(test)]
    pub fn control_set_side_panels(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let mut changed = false;
        let mut saw_panel = false;
        if let Some(left) = params.get("left").and_then(serde_json::Value::as_bool) {
            saw_panel = true;
            if self.show_left_panel != left {
                self.show_left_panel = left;
                changed = true;
            }
        }
        if let Some(right) = params.get("right").and_then(serde_json::Value::as_bool) {
            saw_panel = true;
            if self.show_right_panel != right {
                self.show_right_panel = right;
                changed = true;
            }
        }
        if !saw_panel {
            return serde_json::json!({"error": "set_side_panels requires left and/or right"});
        }
        serde_json::json!({
            "changed": changed,
            "panels": self.control_side_panels_snapshot(),
        })
    }

    pub fn control_smooth_pixels_snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "smooth": self.smooth_pixels,
        })
    }

    #[cfg(test)]
    pub fn control_rendering_snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "smooth_pixels": self.smooth_pixels,
            "show_scale_bar": self.show_scale_bar,
            "show_hud": self.show_hud,
            "show_tile_debug": self.show_tile_debug,
        })
    }

    #[cfg(test)]
    pub fn control_set_rendering(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let mut changed = false;
        let mut saw_field = false;
        let mut sampling_changed = false;

        macro_rules! set_bool {
            ($name:literal, $field:ident) => {
                if let Some(value) = params.get($name) {
                    saw_field = true;
                    let Some(value) = value.as_bool() else {
                        return serde_json::json!({
                            "error": format!("{} must be a boolean", $name),
                        });
                    };
                    if self.$field != value {
                        self.$field = value;
                        changed = true;
                    }
                }
            };
        }

        if let Some(value) = params.get("smooth_pixels").or_else(|| params.get("smooth")) {
            saw_field = true;
            let Some(value) = value.as_bool() else {
                return serde_json::json!({"error": "smooth_pixels must be a boolean"});
            };
            if self.smooth_pixels != value {
                self.smooth_pixels = value;
                changed = true;
                sampling_changed = true;
            }
        }
        set_bool!("show_scale_bar", show_scale_bar);
        set_bool!("show_hud", show_hud);
        set_bool!("show_tile_debug", show_tile_debug);

        if !saw_field {
            return serde_json::json!({
                "error": "provide smooth_pixels, show_scale_bar, show_hud, and/or show_tile_debug",
            });
        }
        if sampling_changed {
            // Sampling is selected when each viewport's draw callback runs.
            // A new presentation generation preserves already-composited tiles
            // for the other viewport while refreshing this one.
            self.bump_render_id();
        }
        serde_json::json!({
            "changed": changed,
            "rendering": self.control_rendering_snapshot(),
        })
    }

    #[cfg(test)]
    pub fn control_set_smooth_pixels(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(smooth) = params.get("smooth").and_then(serde_json::Value::as_bool) else {
            return serde_json::json!({"error": "set_smooth_pixels requires smooth"});
        };
        let changed = self.smooth_pixels != smooth;
        if changed {
            self.smooth_pixels = smooth;
            self.bump_render_id();
        }
        serde_json::json!({
            "changed": changed,
            "smooth_pixels": self.control_smooth_pixels_snapshot(),
        })
    }

    pub fn control_loading_state_snapshot(&self) -> serde_json::Value {
        let scene_reasons = self.scene_busy_debug_reasons();
        let async_reasons = self.async_ui_debug_reasons();
        let active_canvas_ready = self.control_active_canvas_ready();
        let workspace_canvas_ready = self.control_workspace_canvas_ready();
        let busy = self.is_loading_scene()
            || !async_reasons.is_empty()
            || !self.screenshot_pending.is_empty()
            || !self.screenshot_in_flight.is_empty();
        serde_json::json!({
            "busy": busy,
            "canvas_ready": workspace_canvas_ready,
            "canvas": {
                "active_ready": active_canvas_ready,
                "workspace_ready": workspace_canvas_ready,
            },
            "indicator_text": self.loading_indicator_text(),
            "scene_reasons": scene_reasons,
            "async_reasons": async_reasons,
            "image_tiles": {
                "in_flight_or_pending": self.image_tile_request_count(),
                "tiles_gl_busy": self.tiles_gl.as_ref().is_some_and(|tiles_gl| tiles_gl.is_busy()),
                "cpu_cache_busy": self.cache.is_busy(),
                "labels_gl_busy": self.labels_gl.as_ref().is_some_and(|labels_gl| labels_gl.is_busy()),
            },
            "segmentation": {
                "geojson_busy": self.seg_geojson.is_busy(),
                "objects_loading": self.seg_objects.is_loading(),
                "objects_busy": self.seg_objects.is_busy(),
                "objects_analyzing": self.seg_objects.is_analyzing(),
            },
            "spatial": {
                "image_layers_busy": self.spatial_image_layers.is_busy(),
                "shape_or_point_layers_busy": self.spatial_layers.is_busy(),
            },
            "pinned_levels_loading": self.pinned_levels.has_loading(),
            "screenshot": {
                "pending": !self.screenshot_pending.is_empty(),
                "pending_count": self.screenshot_pending.len(),
                "in_flight": !self.screenshot_in_flight.is_empty(),
                "in_flight_count": self.screenshot_in_flight.len(),
            },
        })
    }

    #[cfg(test)]
    pub fn control_set_active_channel(&mut self, params: &serde_json::Value) -> serde_json::Value {
        match self.control_channel_index_from_params(params) {
            Ok(idx) => {
                self.set_active_layer(LayerId::Channel(idx));
                self.bump_render_id();
                serde_json::json!({
                    "changed": true,
                    "active_channel": self.control_active_channel_snapshot(),
                })
            }
            Err(error) => serde_json::json!({"error": error}),
        }
    }

    #[cfg(test)]
    pub fn control_set_visible_channels(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let mode = params
            .get("mode")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("only");
        let Some(values) = params.get("channels").and_then(serde_json::Value::as_array) else {
            return serde_json::json!({"error": "set_visible_channels requires channels: [...]"});
        };
        let mut indices = Vec::new();
        let mut unresolved = Vec::new();
        for value in values {
            match self.control_channel_index_from_value(value) {
                Ok(idx) => {
                    if !indices.contains(&idx) {
                        indices.push(idx);
                    }
                }
                Err(err) => unresolved.push(err),
            }
        }
        if !unresolved.is_empty() {
            return serde_json::json!({"error": format!("unresolved channel(s): {}", unresolved.join("; "))});
        }

        match mode {
            "only" => {
                for (idx, ch) in self.channels.iter_mut().enumerate() {
                    ch.visible = indices.contains(&idx);
                }
            }
            "show" => {
                for idx in &indices {
                    if let Some(ch) = self.channels.get_mut(*idx) {
                        ch.visible = true;
                    }
                }
            }
            "hide" => {
                for idx in &indices {
                    if let Some(ch) = self.channels.get_mut(*idx) {
                        ch.visible = false;
                    }
                }
            }
            other => {
                return serde_json::json!({"error": format!("unknown visibility mode '{other}'")});
            }
        }
        if let Some(first) = indices.first().copied() {
            self.set_active_layer(LayerId::Channel(first));
        }
        self.bump_render_id();
        serde_json::json!({
            "changed": true,
            "mode": mode,
            "visible_channels": self.control_visible_channel_snapshot(),
        })
    }

    #[cfg(test)]
    pub fn control_get_channel_contrast(&self, params: &serde_json::Value) -> serde_json::Value {
        let idx = if params.is_object() && !params.as_object().is_some_and(|obj| obj.is_empty()) {
            match self.control_channel_index_from_params(params) {
                Ok(idx) => idx,
                Err(error) => return serde_json::json!({"error": error}),
            }
        } else {
            self.selected_channel
        };
        let Some(ch) = self.channels.get(idx) else {
            return serde_json::json!({"error": format!("channel index {idx} is out of range")});
        };
        let abs_max = self.dataset.abs_max.max(1.0);
        let (lo, hi) = ch.window.unwrap_or((0.0, abs_max));
        serde_json::json!({
            "index": idx,
            "name": ch.name,
            "min": lo,
            "max": hi,
            "abs_max": abs_max,
        })
    }

    #[cfg(test)]
    pub fn control_set_channel_contrast(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let idx = match self.control_channel_index_from_params(params) {
            Ok(idx) => idx,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let lo = params
            .get("min")
            .or_else(|| params.get("lo"))
            .and_then(serde_json::Value::as_f64)
            .map(|value| value as f32);
        let hi = params
            .get("max")
            .or_else(|| params.get("hi"))
            .and_then(serde_json::Value::as_f64)
            .map(|value| value as f32);
        let (Some(lo), Some(hi)) = (lo, hi) else {
            return serde_json::json!({"error": "set_channel_contrast requires min and max"});
        };
        if !self.set_channel_window_for_link(idx, lo, hi) {
            return serde_json::json!({"error": "invalid contrast limits"});
        }
        self.control_get_channel_contrast(&serde_json::json!({"index": idx}))
    }

    #[cfg(test)]
    pub fn control_set_channel_color(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let idx = match self.control_channel_index_from_params(params) {
            Ok(idx) => idx,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let Some(values) = params
            .get("color_rgb")
            .and_then(serde_json::Value::as_array)
        else {
            return serde_json::json!({"error": "set_channel_color requires color_rgb"});
        };
        let color = values
            .iter()
            .map(serde_json::Value::as_u64)
            .collect::<Option<Vec<_>>>()
            .and_then(|values| {
                (values.len() == 3 && values.iter().all(|value| *value <= 255))
                    .then(|| [values[0] as u8, values[1] as u8, values[2] as u8])
            });
        let Some(color) = color else {
            return serde_json::json!({"error": "color_rgb must contain three integers from 0 to 255"});
        };
        let Some(channel) = self.channels.get_mut(idx) else {
            return serde_json::json!({"error": format!("channel index {idx} is out of range")});
        };
        let color_changed = channel.color_rgb != color;
        channel.color_rgb = color;
        let channel_name = channel.name.clone();
        let mut groups = self.current_layer_groups();
        let inheritance_changed =
            groups
                .channel_members
                .get_mut(&channel_name)
                .is_some_and(|member| {
                    let changed = member.inherit_color;
                    member.inherit_color = false;
                    changed
                });
        if inheritance_changed {
            self.set_current_layer_groups(groups);
        }
        let changed = color_changed || inheritance_changed;
        if changed {
            self.bump_render_id();
        }
        serde_json::json!({
            "changed": changed,
            "channel": self.control_channel_snapshot()[idx].clone(),
        })
    }

    #[cfg(test)]
    pub fn control_set_channel_note(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let idx = match self.control_channel_index_from_params(params) {
            Ok(idx) => idx,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let Some(note) = params.get("note").and_then(serde_json::Value::as_str) else {
            return serde_json::json!({"error": "set_channel_note requires note"});
        };
        let Some(channel) = self.channels.get_mut(idx) else {
            return serde_json::json!({"error": format!("channel index {idx} is out of range")});
        };
        let changed = channel.note != note;
        channel.note = note.to_string();
        serde_json::json!({
            "changed": changed,
            "channel": self.control_channel_snapshot()[idx].clone(),
        })
    }

    pub fn control_get_channel_transform(&self, params: &serde_json::Value) -> serde_json::Value {
        let idx = match self.control_channel_index_from_params(params) {
            Ok(idx) => idx,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let channel = &self.channels[idx];
        let offset = self
            .channel_offsets_world
            .get(idx)
            .copied()
            .unwrap_or(egui::Vec2::ZERO);
        let scale = self
            .channel_scales
            .get(idx)
            .copied()
            .unwrap_or(egui::Vec2::splat(1.0));
        let rotation_rad = self.channel_rotations_rad.get(idx).copied().unwrap_or(0.0);
        serde_json::json!({
            "index": idx,
            "name": channel.name,
            "offset_world": [offset.x, offset.y],
            "scale": [scale.x, scale.y],
            "rotation_rad": rotation_rad,
            "rotation_degrees": rotation_rad.to_degrees(),
        })
    }

    #[cfg(test)]
    pub fn control_set_channel_transform(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let idx = match self.control_channel_index_from_params(params) {
            Ok(idx) => idx,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let before_offset = self.channel_offsets_world[idx];
        let before_scale = self.channel_scales[idx];
        let before_rotation = self.channel_rotations_rad[idx];
        let parse_pair = |name: &str| -> Result<Option<[f32; 2]>, String> {
            let Some(value) = params.get(name) else {
                return Ok(None);
            };
            let Some(values) = value.as_array().filter(|values| values.len() == 2) else {
                return Err(format!("{name} must contain exactly two numbers"));
            };
            let pair = [values[0].as_f64(), values[1].as_f64()];
            if pair
                .iter()
                .any(|value| value.is_none_or(|value| !value.is_finite()))
            {
                return Err(format!("{name} values must be finite numbers"));
            }
            Ok(Some([
                pair[0].unwrap_or_default() as f32,
                pair[1].unwrap_or_default() as f32,
            ]))
        };
        let offset = match parse_pair("offset_world") {
            Ok(value) => value,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let scale = match parse_pair("scale") {
            Ok(value) => value,
            Err(error) => return serde_json::json!({"error": error}),
        };
        if let Some([x, y]) = scale
            && (!(0.01..=100.0).contains(&x) || !(0.01..=100.0).contains(&y))
        {
            return serde_json::json!({"error": "scale values must be between 0.01 and 100"});
        }
        let rotation = match params.get("rotation_rad") {
            Some(value) => match value.as_f64().filter(|value| value.is_finite()) {
                Some(rotation) => Some(rotation as f32),
                None => {
                    return serde_json::json!({"error": "rotation_rad must be a finite number"});
                }
            },
            None => None,
        };
        if let Some([x, y]) = offset {
            self.channel_offsets_world[idx] = egui::vec2(x, y);
        }
        if let Some([x, y]) = scale {
            self.channel_scales[idx] = egui::vec2(x, y);
        }
        if let Some(rotation) = rotation {
            self.channel_rotations_rad[idx] = rotation;
        }
        let changed = before_offset != self.channel_offsets_world[idx]
            || before_scale != self.channel_scales[idx]
            || before_rotation != self.channel_rotations_rad[idx];
        if changed {
            self.hist_dirty = true;
            self.bump_render_id();
        }
        serde_json::json!({
            "changed": changed,
            "transform": self.control_get_channel_transform(&serde_json::json!({"index": idx})),
        })
    }

    #[cfg(test)]
    pub fn control_reset_channel_transform(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let idx = match self.control_channel_index_from_params(params) {
            Ok(idx) => idx,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let changed = self.channel_offsets_world[idx] != egui::Vec2::ZERO
            || self.channel_scales[idx] != egui::Vec2::splat(1.0)
            || self.channel_rotations_rad[idx] != 0.0;
        self.channel_offsets_world[idx] = egui::Vec2::ZERO;
        self.channel_scales[idx] = egui::Vec2::splat(1.0);
        self.channel_rotations_rad[idx] = 0.0;
        if changed {
            self.hist_dirty = true;
            self.bump_render_id();
        }
        serde_json::json!({
            "changed": changed,
            "transform": self.control_get_channel_transform(&serde_json::json!({"index": idx})),
        })
    }
}
