use super::super::*;

impl OmeZarrViewerApp {
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

    pub fn control_set_channel_order(&mut self, params: &serde_json::Value) -> serde_json::Value {
        if let Some(sort) = params.get("sort").and_then(serde_json::Value::as_str) {
            let Some(mode) = ChannelSortMode::from_storage_key(sort) else {
                return serde_json::json!({"error": format!("unknown channel sort mode '{sort}'")});
            };
            self.channel_sort_mode = mode;
            return serde_json::json!({
                "changed": true,
                "sort": mode.storage_key(),
                "order": self.control_channel_order_snapshot(),
            });
        }

        let Some(values) = params.get("channels").and_then(serde_json::Value::as_array) else {
            return serde_json::json!({"error": "set_channel_order requires channels or sort"});
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
        let mode = params
            .get("mode")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("listed_first");
        match mode {
            "listed_first" => self.move_channels_to_top_for_deep_link(&indices),
            "exact" => {
                if indices.len() != self.channels.len() {
                    return serde_json::json!({"error": "exact channel order must include every channel exactly once"});
                }
                self.channel_layer_order = indices;
                self.channel_sort_mode = ChannelSortMode::Manual;
                self.bump_render_id();
            }
            other => {
                return serde_json::json!({"error": format!("unknown channel order mode '{other}'")});
            }
        }
        serde_json::json!({
            "changed": true,
            "mode": mode,
            "sort": self.channel_sort_mode.storage_key(),
            "order": self.control_channel_order_snapshot(),
        })
    }

    pub fn control_channel_presentation_json(&self) -> serde_json::Value {
        serde_json::json!({
            "search": self.channel_list_search,
            "sort": self.channel_sort_mode.storage_key(),
            "order": self.control_channel_order_snapshot(),
        })
    }

    pub fn control_set_channel_presentation(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let search = match params.get("search") {
            Some(value) => match value.as_str() {
                Some(value) => Some(value.to_string()),
                None => return serde_json::json!({"error": "search must be a string"}),
            },
            None => None,
        };
        let sort = match params.get("sort") {
            Some(value) => match value.as_str().and_then(ChannelSortMode::from_storage_key) {
                Some(value) => Some(value),
                None => return serde_json::json!({"error": "unknown channel sort mode"}),
            },
            None => None,
        };
        if let Some(search) = search {
            self.channel_list_search = search;
        }
        if let Some(sort) = sort {
            self.channel_sort_mode = sort;
        }
        self.control_channel_presentation_json()
    }

    pub fn control_channel_groups_snapshot(&self) -> serde_json::Value {
        channel_groups_snapshot(&self.current_layer_groups(), &self.channels)
    }

    pub fn control_set_channel_group(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(values) = params.get("channels").and_then(serde_json::Value::as_array) else {
            return serde_json::json!({"error": "set_channel_group requires channels"});
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
        if indices.is_empty() {
            return serde_json::json!({"error": "no channels resolved"});
        }

        let mut groups = self.current_layer_groups();
        let requested_group_id = params.get("group_id").and_then(serde_json::Value::as_u64);
        let requested_name = params
            .get("group")
            .or_else(|| params.get("name"))
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let group_id = ensure_channel_group(
            &mut groups,
            requested_group_id,
            requested_name,
            mcp_color_from_params(params),
        );
        if params
            .get("replace_group_members")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            groups
                .channel_members
                .retain(|_, member| member.group_id != group_id);
        }
        let inherit_color = params
            .get("inherit_color")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);
        for idx in &indices {
            if let Some(ch) = self.channels.get(*idx) {
                groups.channel_members.insert(
                    ch.name.clone(),
                    ProjectChannelGroupMember {
                        group_id,
                        inherit_color,
                    },
                );
            }
        }
        self.selected_channel_group_id = Some(group_id);
        self.set_current_layer_groups(groups);
        serde_json::json!({
            "changed": true,
            "group_id": group_id,
            "groups": self.control_channel_groups_snapshot(),
        })
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

    pub fn control_set_camera(&mut self, params: &serde_json::Value) -> serde_json::Value {
        if let Some(center) = params
            .get("center_world_lvl0")
            .and_then(serde_json::Value::as_array)
            && center.len() == 2
        {
            let x = center[0].as_f64().map(|value| value as f32);
            let y = center[1].as_f64().map(|value| value as f32);
            if let (Some(x), Some(y)) = (x, y)
                && x.is_finite()
                && y.is_finite()
            {
                self.camera.center_world_lvl0 = egui::pos2(x, y);
            }
        }
        if let Some(x) = params
            .get("center_x")
            .and_then(serde_json::Value::as_f64)
            .map(|value| value as f32)
            && x.is_finite()
        {
            self.camera.center_world_lvl0.x = x;
        }
        if let Some(y) = params
            .get("center_y")
            .and_then(serde_json::Value::as_f64)
            .map(|value| value as f32)
            && y.is_finite()
        {
            self.camera.center_world_lvl0.y = y;
        }
        if let Some(zoom) = params
            .get("zoom_screen_per_lvl0_px")
            .or_else(|| params.get("zoom"))
            .and_then(serde_json::Value::as_f64)
            .map(|value| value as f32)
            && zoom.is_finite()
            && zoom > 0.0
        {
            self.camera.zoom_screen_per_lvl0_px = zoom.clamp(0.000_01, 5000.0);
        }
        self.bump_render_id();
        self.control_camera_snapshot()
    }

    pub fn control_zoom(&mut self, factor: f32) -> serde_json::Value {
        if !factor.is_finite() || factor <= 0.0 {
            return serde_json::json!({"error": "zoom factor must be finite and > 0"});
        }
        if let Some(viewport) = self.last_canvas_rect {
            self.camera
                .zoom_about_screen_point(viewport, viewport.center(), factor);
        } else {
            self.camera.zoom_screen_per_lvl0_px =
                (self.camera.zoom_screen_per_lvl0_px * factor).clamp(0.000_01, 5000.0);
        }
        self.bump_render_id();
        self.control_camera_snapshot()
    }

    pub fn control_fit_to_view(&mut self) -> serde_json::Value {
        let Some(viewport) = self.last_canvas_rect else {
            return serde_json::json!({"error": "No canvas viewport is available yet."});
        };
        self.fit_to_rect(viewport);
        self.bump_render_id();
        self.control_camera_snapshot()
    }

    pub fn control_capture_screenshot(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let target = match params
            .get("viewport_id")
            .and_then(serde_json::Value::as_str)
        {
            Some(value) => match ViewportId::new(value) {
                Ok(id)
                    if self
                        .viewport_workspace
                        .as_ref()
                        .is_some_and(|workspace| workspace.get(&id).is_some()) =>
                {
                    id
                }
                Ok(id) => {
                    return serde_json::json!({"error": format!("viewport '{id}' was not found")});
                }
                Err(error) => return serde_json::json!({"error": error.to_string()}),
            },
            None => match self
                .viewport_workspace
                .as_ref()
                .map(|workspace| workspace.active_id().clone())
            {
                Some(id) => id,
                None => {
                    return serde_json::json!({"error": "viewer workspace is not initialized"});
                }
            },
        };
        if let Some(path) = params
            .get("path")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            let path = PathBuf::from(path);
            self.request_screenshot_png_for_viewport(path.clone(), target.clone());
            return serde_json::json!({
                "queued": true,
                "path": path.to_string_lossy(),
                "viewport_id": target.as_str(),
            });
        }
        match self.request_quick_screenshot_png_for_viewport(target.clone()) {
            Ok(path) => serde_json::json!({
                "queued": true,
                "path": path.to_string_lossy(),
                "viewport_id": target.as_str(),
            }),
            Err(err) => serde_json::json!({"error": format!("{err}")}),
        }
    }

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
