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
            "pinned_levels_loading": self.projected_memory_running(),
            "screenshot": {
                "pending": !self.screenshot_pending.is_empty(),
                "pending_count": self.screenshot_pending.len(),
                "in_flight": !self.screenshot_in_flight.is_empty(),
                "in_flight_count": self.screenshot_in_flight.len(),
            },
        })
    }

    pub(in crate::app) fn channel_transform_snapshot(&self, idx: usize) -> serde_json::Value {
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
}
