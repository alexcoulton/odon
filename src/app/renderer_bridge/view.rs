use super::super::*;

impl OmeZarrViewerApp {
    #[cfg(test)]
    pub fn control_get_channel_intensity_stats(
        &self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
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
        let Some(level0) = self.dataset.levels.first() else {
            return serde_json::json!({"error": "dataset has no pyramid levels"});
        };
        let requested_level = params
            .get("level")
            .and_then(serde_json::Value::as_u64)
            .map(|value| value as usize);
        let level_idx = requested_level
            .unwrap_or_else(|| self.dataset.levels.len().saturating_sub(1))
            .min(self.dataset.levels.len().saturating_sub(1));
        let Some(level_info) = self.dataset.levels.get(level_idx) else {
            return serde_json::json!({"error": format!("level {level_idx} is out of range")});
        };
        let Some(axes) = display_axes_for_mode(&self.dataset.dims, self.view_plane_mode) else {
            return serde_json::json!({"error": "current view plane has no display axes"});
        };
        if axes.vertical >= level_info.shape.len() || axes.horizontal >= level_info.shape.len() {
            return serde_json::json!({"error": "display axes are outside image shape"});
        }
        let row_range = 0..level_info.shape[axes.vertical];
        let col_range = 0..level_info.shape[axes.horizontal];
        let Some(ranges) = image_subset_ranges_for_view(
            &self.dataset.dims,
            level0,
            level_info,
            Some(ch.index as u64),
            row_range,
            col_range,
            self.active_view_selection(),
        ) else {
            return serde_json::json!({"error": "failed to build image subset ranges"});
        };
        let zarr_path = format!("/{}", level_info.path.trim_start_matches('/'));
        let array = match Array::open(self.store.clone(), &zarr_path) {
            Ok(array) => array,
            Err(err) => {
                return serde_json::json!({"error": format!("failed to open level {level_idx}: {err}")});
            }
        };
        let subset = ArraySubset::new_with_ranges(&ranges);
        let data = match retrieve_image_subset_u16(&array, &subset, &level_info.dtype) {
            Ok(data) => data,
            Err(err) => {
                return serde_json::json!({"error": format!("failed to read level {level_idx}: {err}")});
            }
        };
        channel_intensity_stats_json(
            idx,
            &ch.name,
            level_info.index,
            level_info.downsample,
            &data,
        )
    }

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

    #[cfg(test)]
    pub(in crate::app) fn control_channel_index_from_params(
        &self,
        params: &serde_json::Value,
    ) -> Result<usize, String> {
        if let Some(value) = params.get("index").or_else(|| params.get("channel_index")) {
            return self.control_channel_index_from_value(value);
        }
        if let Some(value) = params
            .get("name")
            .or_else(|| params.get("channel"))
            .or_else(|| params.get("marker"))
        {
            return self.control_channel_index_from_value(value);
        }
        Err("provide index, name, channel, or marker".to_string())
    }

    #[cfg(test)]
    pub(in crate::app) fn control_channel_index_from_value(
        &self,
        value: &serde_json::Value,
    ) -> Result<usize, String> {
        if let Some(idx) = value.as_u64() {
            let idx = idx as usize;
            if idx < self.channels.len() {
                return Ok(idx);
            }
            return Err(format!("channel index {idx} is out of range"));
        }
        let Some(name) = value.as_str().map(str::trim).filter(|s| !s.is_empty()) else {
            return Err(format!("invalid channel selector: {value}"));
        };
        self.find_channel_index_for_link(name)
            .ok_or_else(|| format!("no channel matches '{name}'"))
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
