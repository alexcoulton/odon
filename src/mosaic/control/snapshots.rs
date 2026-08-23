use super::super::*;

impl MosaicViewerApp {
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

    pub fn control_smooth_pixels_snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "smooth": self.smooth_pixels,
        })
    }

    pub fn control_loading_state_snapshot(&self) -> serde_json::Value {
        let image_tiles_busy = self.tiles_gl.is_busy();
        let segmentation = self.seg_geojson.control_loading_snapshot();
        let segmentation_busy = self.seg_geojson.is_busy();
        let pinned_levels_loading = self.projected_memory_running();
        let screenshot_pending = self.screenshot_pending.is_some();
        let screenshot_in_flight = self.screenshot_in_flight.is_some();

        let mut reasons = Vec::new();
        if image_tiles_busy {
            reasons.push("image_tiles");
        }
        if segmentation_busy {
            reasons.push("segmentation_objects");
        }
        if self.seg_geojson_pending_visible {
            reasons.push("segmentation_pending_visible");
        }
        if pinned_levels_loading {
            reasons.push("pinned_levels");
        }
        if screenshot_pending || screenshot_in_flight {
            reasons.push("screenshot");
        }

        serde_json::json!({
            "busy": !reasons.is_empty(),
            "canvas_ready": self.control_canvas_ready(),
            "reasons": reasons,
            "top_right_spinner": {
                "visible": image_tiles_busy || segmentation_busy || self.seg_geojson_pending_visible,
                "note": "Mosaic tile debug labels only visible image-tile loading; segmentation/object loading can show the spinner with no tile count.",
            },
            "image_tiles": {
                "busy": image_tiles_busy,
                "in_flight": self.tiles_gl.in_flight_len(),
                "tile_debug_enabled": self.show_tile_debug,
                "request_generation": self.tile_request_generation,
            },
            "segmentation": segmentation,
            "segmentation_pending_visible": self.seg_geojson_pending_visible,
            "pinned_levels_loading": pinned_levels_loading,
            "screenshot": {
                "pending": screenshot_pending,
                "in_flight": screenshot_in_flight,
            },
        })
    }

    pub fn control_canvas_ready(&self) -> bool {
        self.last_canvas_rect.is_some_and(|rect| {
            rect.min.x.is_finite()
                && rect.min.y.is_finite()
                && rect.max.x.is_finite()
                && rect.max.y.is_finite()
                && rect.width() > 0.0
                && rect.height() > 0.0
        })
    }

    pub fn control_camera_snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "center_world_lvl0": [
                self.camera.center_world_lvl0.x,
                self.camera.center_world_lvl0.y,
            ],
            "zoom_screen_per_lvl0_px": self.camera.zoom_screen_per_lvl0_px,
        })
    }

    pub fn control_view_snapshot(&self) -> serde_json::Value {
        let active_channel = self.channels.get(self.selected_channel).map(|ch| {
            serde_json::json!({
                "index": self.selected_channel,
                "name": ch.name,
            })
        });
        serde_json::json!({
            "active_channel": active_channel,
            "channel_count": self.channels.len(),
            "roi_count": self.items.len(),
            "visible_channels": self.channels
                .iter()
                .filter(|ch| ch.visible)
                .map(|ch| ch.name.clone())
                .collect::<Vec<_>>(),
            "focused_roi": self.focused_core_id.and_then(|id| {
                self.items
                    .iter()
                    .find(|item| item.id == id)
                    .map(|item| item.sample_id.clone())
            }),
        })
    }
}
