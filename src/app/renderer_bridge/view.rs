use super::super::*;

impl OmeZarrViewerApp {
    pub fn control_channel_presentation_json(&self) -> serde_json::Value {
        serde_json::json!({
            "search": self.channel_list_search,
            "sort": self.channel_sort_mode.storage_key(),
            "order": self.control_channel_order_snapshot(),
        })
    }

    #[cfg(test)]
    pub fn control_channel_groups_snapshot(&self) -> serde_json::Value {
        channel_groups_snapshot(&self.current_layer_groups(), &self.channels)
    }

    pub(in crate::app) fn control_channel_order_snapshot(&self) -> serde_json::Value {
        serde_json::Value::Array(
            self.channel_layer_order
                .iter()
                .filter_map(|idx| {
                    self.channels.get(*idx).map(|ch| {
                        serde_json::json!({
                            "index": idx,
                            "name": ch.name,
                            "visible": ch.visible,
                        })
                    })
                })
                .collect(),
        )
    }

    pub fn control_camera_snapshot(&self) -> serde_json::Value {
        let viewport = self.last_canvas_rect.map(|rect| {
            let visible = self.visible_world_rect(rect);
            serde_json::json!({
                "screen_rect": [rect.min.x, rect.min.y, rect.max.x, rect.max.y],
                "visible_world_lvl0": [visible.min.x, visible.min.y, visible.max.x, visible.max.y],
            })
        });
        serde_json::json!({
            "center_world_lvl0": [
                self.camera.center_world_lvl0.x,
                self.camera.center_world_lvl0.y,
            ],
            "zoom_screen_per_lvl0_px": self.camera.zoom_screen_per_lvl0_px,
            "viewport": viewport,
        })
    }

    pub fn control_view_snapshot(&self) -> serde_json::Value {
        let active_channel = self.channels.get(self.selected_channel).map(|ch| {
            serde_json::json!({
                "index": self.selected_channel,
                "name": ch.name,
            })
        });
        let level0 = self.dataset.levels.first();
        let dataset = serde_json::json!({
            "source": self.dataset.source.source_key(),
            "axes": self.dataset.multiscale.axes.iter().map(|axis| serde_json::json!({
                "name": axis.name, "unit": axis.unit,
            })).collect::<Vec<_>>(),
            "shape": level0.map(|level| level.shape.clone()),
            "chunks": level0.map(|level| level.chunks.clone()),
            "dtype": level0.map(|level| level.dtype.clone()),
            "scale": level0.map(|level| level.scale.clone()),
            "translation": level0.map(|level| level.translation.clone()),
            "pyramid_levels": self.dataset.levels.len(),
            "render_kind": match self.dataset.render_kind {
                crate::data::ome::DatasetRenderKind::Image => "image",
                crate::data::ome::DatasetRenderKind::LabelMask => "labels",
            },
        });
        serde_json::json!({
            "dataset": self.dataset.source.source_key(),
            "dataset_descriptor": dataset,
            "active_channel": active_channel,
            "channel_count": self.channels.len(),
            "visible_channels": self.channels
                .iter()
                .filter(|ch| ch.visible)
                .map(|ch| ch.name.clone())
                .collect::<Vec<_>>(),
        })
    }
}
