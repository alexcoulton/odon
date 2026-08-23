use super::super::*;

#[cfg(test)]
fn normalize_mcp_channel_name(value: &str) -> String {
    value
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

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

    pub fn control_smooth_pixels_snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "smooth": self.smooth_pixels,
        })
    }

    pub fn control_loading_state_snapshot(&self) -> serde_json::Value {
        let image_tiles_busy = self.tiles_gl.is_busy();
        let segmentation = self.seg_geojson.control_loading_snapshot();
        let segmentation_busy = self.seg_geojson.is_busy();
        let pinned_levels_loading = self.pinned_levels.has_loading();
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
            self.set_active_layer(MosaicLayerId::Channel(first));
        }
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
        let abs_max = self.abs_max.max(1.0);
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
        if !lo.is_finite() || !hi.is_finite() || hi <= lo {
            return serde_json::json!({"error": "invalid contrast limits"});
        }
        let abs_max = self.abs_max.max(1.0);
        let lo = lo.clamp(0.0, abs_max);
        let hi = hi.clamp(0.0, abs_max);
        if hi <= lo {
            return serde_json::json!({"error": "invalid contrast limits"});
        }
        if let Some(ch) = self.channels.get_mut(idx) {
            ch.window = Some((lo, hi));
        }
        self.control_get_channel_contrast(&serde_json::json!({"index": idx}))
    }

    pub(in crate::mosaic) fn control_native_layer_kind(id: MosaicLayerId) -> &'static str {
        match id {
            MosaicLayerId::TextLabels => "text_labels",
            MosaicLayerId::SegmentationGeoJson => "segmentation_geojson",
            MosaicLayerId::Annotation(_) => "annotation",
            MosaicLayerId::Channel(_) => "channel",
        }
    }

    pub(in crate::mosaic) fn control_native_layer_snapshot(
        &self,
        id: MosaicLayerId,
        stack: &str,
        order: usize,
    ) -> serde_json::Value {
        serde_json::json!({
            "layer_id": Self::layer_id_storage_key(id),
            "kind": Self::control_native_layer_kind(id),
            "name": self.layer_display_name(id),
            "stack": stack,
            "order": order,
            "active": self.active_layer == id,
            "visible": self.layer_visible_value(id).unwrap_or(false),
            "available": self.layer_available(id),
            "offset_world": serde_json::Value::Null,
        })
    }

    pub fn control_native_layer_snapshot_list(&self) -> serde_json::Value {
        let mut layers = self
            .channel_layer_order
            .iter()
            .copied()
            .enumerate()
            .map(|(order, idx)| {
                self.control_native_layer_snapshot(MosaicLayerId::Channel(idx), "channels", order)
            })
            .collect::<Vec<_>>();
        layers.extend(
            self.overlay_layer_order
                .iter()
                .copied()
                .enumerate()
                .map(|(order, id)| self.control_native_layer_snapshot(id, "overlays", order)),
        );
        serde_json::Value::Array(layers)
    }

    pub fn control_channel_presentation_json(&self) -> serde_json::Value {
        serde_json::json!({
            "search": self.channel_list_search,
            "sort": self.channel_sort_mode.storage_key(),
            "order": self.control_channel_order_snapshot(),
        })
    }

    pub(in crate::mosaic) fn control_channel_order_snapshot(&self) -> serde_json::Value {
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
        serde_json::json!({
            "center_world_lvl0": [
                self.camera.center_world_lvl0.x,
                self.camera.center_world_lvl0.y,
            ],
            "zoom_screen_per_lvl0_px": self.camera.zoom_screen_per_lvl0_px,
        })
    }

    #[cfg(test)]
    pub fn control_configure_layout(&mut self, params: &serde_json::Value) -> serde_json::Value {
        if let Some(group_by) = params
            .get("group_by")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
        {
            self.group_by = group_by.to_string();
        }
        if let Some(sort_by) = params
            .get("sort_by")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            self.sort_by = sort_by.to_string();
        }
        if let Some(sort_by_secondary) = params
            .get("sort_by_secondary")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            self.sort_by_secondary = sort_by_secondary.to_string();
            self.sort_secondary_enabled = true;
        }
        if let Some(enabled) = params
            .get("sort_secondary_enabled")
            .and_then(serde_json::Value::as_bool)
        {
            self.sort_secondary_enabled = enabled;
        }
        if let Some(show) = params
            .get("show_group_labels")
            .and_then(serde_json::Value::as_bool)
        {
            self.show_group_labels = show;
        }
        if let Some(show) = params
            .get("show_text_labels")
            .and_then(serde_json::Value::as_bool)
        {
            self.show_text_labels = show;
        }
        if let Some(gap) = params.get("group_gap").and_then(serde_json::Value::as_f64)
            && gap.is_finite()
        {
            self.group_gap = gap.max(0.0) as f32;
        }
        if let Some(cols) = params.get("columns").and_then(serde_json::Value::as_u64) {
            self.grid_cols = (cols as usize).max(1);
        }
        if let Some(layout) = params
            .get("layout")
            .or_else(|| params.get("layout_mode"))
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            let Some(mode) = MosaicLayoutMode::from_storage_key(layout) else {
                return serde_json::json!({
                    "error": "unknown layout; expected fit_cells or native_pixels"
                });
            };
            self.layout_mode = mode;
        }
        if let Some(values) = params
            .get("label_columns")
            .and_then(serde_json::Value::as_array)
        {
            let columns = values
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>();
            self.label_columns = columns;
        }
        self.apply_sort_and_layout();
        if params
            .get("fit")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true)
            && self.last_canvas_rect.is_some()
        {
            self.fit_mosaic();
        }
        serde_json::json!({
            "right_tab": self.right_tab.storage_key(),
            "group_by": self.group_by,
            "sort_by": self.sort_by,
            "sort_secondary_enabled": self.sort_secondary_enabled,
            "sort_by_secondary": self.sort_by_secondary,
            "layout": self.layout_mode.storage_key(),
            "columns": self.grid_cols,
            "group_gap": self.group_gap,
            "show_group_labels": self.show_group_labels,
            "show_text_labels": self.show_text_labels,
            "label_columns": self.label_columns,
        })
    }

    #[cfg(test)]
    pub(in crate::mosaic) fn control_channel_index_from_params(
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
    pub(in crate::mosaic) fn control_channel_index_from_value(
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
        let needle = normalize_mcp_channel_name(name);
        let exact = self
            .channels
            .iter()
            .position(|ch| normalize_mcp_channel_name(&ch.name) == needle);
        if let Some(idx) = exact {
            return Ok(idx);
        }
        let matches = self
            .channels
            .iter()
            .enumerate()
            .filter_map(|(idx, ch)| {
                normalize_mcp_channel_name(&ch.name)
                    .contains(&needle)
                    .then_some(idx)
            })
            .collect::<Vec<_>>();
        if matches.len() == 1 {
            Ok(matches[0])
        } else if matches.is_empty() {
            Err(format!("no channel matches '{name}'"))
        } else {
            Err(format!("channel selector '{name}' is ambiguous"))
        }
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

    #[cfg(test)]
    pub fn control_focus_snapshot(&self) -> serde_json::Value {
        self.focused_core_id
            .and_then(|id| {
                self.items
                    .iter()
                    .position(|item| item.id == id)
                    .map(|index| {
                        let item = &self.items[index];
                        serde_json::json!({
                            "index": index,
                            "id": item.id,
                            "roi_id": item.sample_id,
                            "metadata": item.meta,
                        })
                    })
            })
            .unwrap_or(serde_json::Value::Null)
    }

    #[cfg(test)]
    pub fn control_mosaic_snapshot(&self) -> serde_json::Value {
        let unresolved_layout_fields = [self.group_by.as_str(), self.sort_by.as_str()]
            .into_iter()
            .chain(
                self.sort_secondary_enabled
                    .then_some(self.sort_by_secondary.as_str()),
            )
            .chain(self.label_columns.iter().map(String::as_str))
            .filter(|field| {
                !field.is_empty()
                    && *field != "id"
                    && !self.metadata_columns.iter().any(|column| column == field)
            })
            .map(str::to_string)
            .collect::<HashSet<_>>();
        serde_json::json!({
            "roi_count": self.items.len(),
            "focused": self.control_focus_snapshot(),
            "selection": self.control_selection_snapshot(),
            "metadata_columns": self.metadata_columns,
            "mosaic_bounds": {
                "min": [self.mosaic_bounds.min.x, self.mosaic_bounds.min.y],
                "max": [self.mosaic_bounds.max.x, self.mosaic_bounds.max.y],
            },
            "layout": {
                "group_by": self.group_by,
                "sort_by": self.sort_by,
                "sort_secondary_enabled": self.sort_secondary_enabled,
                "sort_by_secondary": self.sort_by_secondary,
                "layout": self.layout_mode.storage_key(),
                "columns": self.grid_cols,
                "group_gap": self.group_gap,
                "show_group_labels": self.show_group_labels,
                "show_text_labels": self.show_text_labels,
                "label_columns": self.label_columns,
                "unresolved_fields": unresolved_layout_fields,
            },
            "rois": self.items.iter().enumerate().map(|(index, item)| serde_json::json!({
                "index": index,
                "id": item.id,
                "roi_id": item.sample_id,
                "metadata": item.meta,
                "focused": self.focused_core_id == Some(item.id),
                "selected": self.selected_core_ids.contains(&item.id),
            })).collect::<Vec<_>>(),
        })
    }

    #[cfg(test)]
    pub fn control_selection_snapshot(&self) -> serde_json::Value {
        let selected = self
            .items
            .iter()
            .enumerate()
            .filter(|(_, item)| self.selected_core_ids.contains(&item.id))
            .map(|(index, item)| {
                serde_json::json!({"index": index, "id": item.id, "roi_id": item.sample_id})
            })
            .collect::<Vec<_>>();
        serde_json::json!({"count": selected.len(), "selected": selected})
    }

    #[cfg(test)]
    pub fn control_set_focused_roi(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let index = if let Some(index) = params
            .get("index")
            .and_then(serde_json::Value::as_u64)
            .map(|index| index as usize)
        {
            if index >= self.items.len() {
                return serde_json::json!({"error": format!("mosaic ROI index {index} is out of range")});
            }
            index
        } else if let Some(roi_id) = params
            .get("roi_id")
            .or_else(|| params.get("id"))
            .and_then(serde_json::Value::as_str)
        {
            let matches = self
                .items
                .iter()
                .enumerate()
                .filter(|(_, item)| item.sample_id == roi_id)
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            match matches.as_slice() {
                [index] => *index,
                [] => {
                    return serde_json::json!({"error": format!("mosaic ROI '{roi_id}' was not found")});
                }
                _ => {
                    return serde_json::json!({"error": format!("mosaic ROI '{roi_id}' is ambiguous")});
                }
            }
        } else {
            return serde_json::json!({"error": "provide index or roi_id"});
        };
        let before = self.focused_core_id;
        self.focused_core_id = Some(self.items[index].id);
        if params
            .get("fit")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true)
            && let Some(viewport) = self.last_canvas_rect
        {
            self.camera
                .fit_to_world_rect(viewport, item_rect(&self.items[index]));
        }
        serde_json::json!({
            "changed": before != self.focused_core_id,
            "focused": self.control_focus_snapshot(),
        })
    }

    #[cfg(test)]
    pub fn control_step_focused_roi(
        &mut self,
        params: &serde_json::Value,
        forward: bool,
    ) -> serde_json::Value {
        if self.items.is_empty() {
            return serde_json::json!({"error": "mosaic has no ROIs"});
        }
        let step = params
            .get("step")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(1) as usize;
        let wrap = params
            .get("wrap")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);
        let current = self
            .focused_core_id
            .and_then(|id| self.items.iter().position(|item| item.id == id))
            .unwrap_or_default();
        let index = if wrap {
            let offset = step % self.items.len();
            if forward {
                (current + offset) % self.items.len()
            } else {
                (current + self.items.len() - offset) % self.items.len()
            }
        } else if forward {
            current.saturating_add(step).min(self.items.len() - 1)
        } else {
            current.saturating_sub(step)
        };
        self.control_set_focused_roi(&serde_json::json!({
            "index": index,
            "fit": params.get("fit").and_then(serde_json::Value::as_bool).unwrap_or(true),
        }))
    }

    pub fn control_object_loading_snapshot(&self) -> serde_json::Value {
        let items = self
            .items
            .iter()
            .enumerate()
            .map(|(index, item)| {
                let mut snapshot = self.seg_geojson.control_item_snapshot(item.id);
                if let Some(object) = snapshot.as_object_mut() {
                    object.insert("index".to_string(), serde_json::json!(index));
                    object.insert("roi_id".to_string(), serde_json::json!(item.sample_id));
                    object.insert(
                        "selected".to_string(),
                        serde_json::json!(self.selected_core_ids.contains(&item.id)),
                    );
                    object.insert(
                        "requested".to_string(),
                        serde_json::json!(self.pending_object_load_ids.contains(&item.id)),
                    );
                }
                snapshot
            })
            .collect::<Vec<_>>();
        let requested_count = self.pending_object_load_ids.len();
        let requested_loading = self
            .pending_object_load_ids
            .iter()
            .filter(|item_id| {
                self.seg_geojson
                    .control_item_snapshot(**item_id)
                    .get("loading_data")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
            })
            .count();
        serde_json::json!({
            "overlay": self.seg_geojson.control_loading_snapshot(),
            "requested_count": requested_count,
            "requested_loading": requested_loading,
            "settled": requested_count == 0,
            "items": items,
        })
    }
}
