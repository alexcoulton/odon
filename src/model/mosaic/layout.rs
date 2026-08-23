//! Mosaic layout, item listing, selection, focus, and camera navigation.

use super::*;

impl MosaicModel {
    pub(super) fn set_left_tab(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let tab = params
            .get("tab")
            .or_else(|| params.get("left_tab"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|tab| !tab.is_empty())
            .ok_or_else(|| invalid("set_mosaic_left_tab requires tab"))?;
        if !matches!(tab, "layers" | "project") {
            return Err(invalid("unknown left tab; expected layers or project"));
        }
        self.left_tab = tab.to_string();
        Ok(json!({"left_tab":self.left_tab}))
    }

    pub(super) fn set_right_tab(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let tab = params
            .get("tab")
            .or_else(|| params.get("right_tab"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|tab| !tab.is_empty())
            .ok_or_else(|| invalid("set_mosaic_right_tab requires tab"))?;
        if !matches!(tab, "properties" | "views" | "layout" | "memory") {
            return Err(invalid(
                "unknown right tab; expected properties, views, layout, or memory",
            ));
        }
        self.right_tab = tab.to_string();
        Ok(json!({"right_tab":self.right_tab}))
    }

    pub(super) fn configure_layout(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        if let Some(group_by) = params.get("group_by").and_then(Value::as_str) {
            self.group_by = group_by.trim().to_string();
        }
        if let Some(sort_by) = params
            .get("sort_by")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            self.sort_by = sort_by.to_string();
        }
        if let Some(sort_by) = params
            .get("sort_by_secondary")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            self.sort_by_secondary = sort_by.to_string();
            self.sort_secondary_enabled = true;
        }
        if let Some(enabled) = params
            .get("sort_secondary_enabled")
            .and_then(Value::as_bool)
        {
            self.sort_secondary_enabled = enabled;
        }
        if let Some(show) = params.get("show_group_labels").and_then(Value::as_bool) {
            self.show_group_labels = show;
        }
        if let Some(show) = params.get("show_text_labels").and_then(Value::as_bool) {
            self.show_text_labels = show;
            let _ = self.native_layers.set_visibility("text_labels", show);
        }
        if let Some(gap) = params.get("group_gap").and_then(Value::as_f64) {
            if !gap.is_finite() {
                return Err(invalid("group_gap must be finite"));
            }
            self.group_gap = gap.max(0.0) as f32;
        }
        if let Some(columns) = params.get("columns").and_then(Value::as_u64) {
            self.columns = usize::try_from(columns).unwrap_or(usize::MAX).max(1);
        }
        if let Some(layout) = params
            .get("layout")
            .or_else(|| params.get("layout_mode"))
            .and_then(Value::as_str)
        {
            self.layout_mode = MosaicLayoutMode::parse(layout.trim())
                .ok_or_else(|| invalid("unknown layout; expected fit_cells or native_pixels"))?;
        }
        if let Some(columns) = params.get("label_columns").and_then(Value::as_array) {
            self.label_columns = columns
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .collect();
        }
        let preserve_center = self.camera_center;
        self.apply_layout();
        self.camera_center = clamp_point_to_bounds(preserve_center, self.bounds);
        if params.get("fit").and_then(Value::as_bool).unwrap_or(true) {
            self.fit_bounds(self.bounds);
        }
        Ok(self.layout_snapshot())
    }

    pub(super) fn layout_snapshot(&self) -> Value {
        json!({
            "left_tab":self.left_tab,
            "right_tab":self.right_tab,
            "group_by":self.group_by,
            "sort_by":self.sort_by,
            "sort_secondary_enabled":self.sort_secondary_enabled,
            "sort_by_secondary":self.sort_by_secondary,
            "layout":self.layout_mode.as_str(),
            "columns":self.columns,
            "group_gap":self.group_gap,
            "show_group_labels":self.show_group_labels,
            "show_text_labels":self.show_text_labels,
            "label_columns":self.label_columns,
        })
    }

    pub(super) fn list_items(&self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let offset = params.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
        let limit = params.get("limit").and_then(Value::as_u64).unwrap_or(200) as usize;
        let total = self.items.len();
        let items = self
            .items
            .iter()
            .enumerate()
            .skip(offset)
            .take(limit)
            .map(|(index, item)| {
                json!({
                    "index":index,
                    "id":item.id,
                    "roi_id":item.roi_id,
                    "metadata":item.metadata,
                    "source":item.source,
                    "offset_world":item.offset,
                    "scale":item.scale,
                    "placed_size":item.placed_size,
                    "bounds_world":{"min":item.bounds()[0],"max":item.bounds()[1]},
                    "focused":self.focused_id == Some(item.id),
                    "selected":self.selected_ids.contains(&item.id),
                })
            })
            .collect::<Vec<_>>();
        Ok(json!({
            "total":total,
            "offset":offset,
            "limit":limit,
            "has_more":offset.saturating_add(items.len()) < total,
            "items":items,
        }))
    }

    pub(super) fn selection_snapshot(&self) -> Value {
        let selected = self
            .items
            .iter()
            .enumerate()
            .filter(|(_, item)| self.selected_ids.contains(&item.id))
            .map(|(index, item)| json!({"index":index,"id":item.id,"roi_id":item.roi_id}))
            .collect::<Vec<_>>();
        json!({"count":selected.len(),"selected":selected})
    }

    pub(super) fn set_selection(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let mode = params
            .get("mode")
            .and_then(Value::as_str)
            .unwrap_or("replace");
        if mode == "all" {
            self.selected_ids = self.items.iter().map(|item| item.id).collect();
            return Ok(self.selection_snapshot());
        }
        if mode == "range" {
            let start = params
                .get("start")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("range selection requires start and end"))?;
            let end = params
                .get("end")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("range selection requires start and end"))?;
            let start = self.item_index_for_roi(start)?;
            let end = self.item_index_for_roi(end)?;
            let (lo, hi) = if start <= end {
                (start, end)
            } else {
                (end, start)
            };
            self.selected_ids = self.items[lo..=hi].iter().map(|item| item.id).collect();
            return Ok(self.selection_snapshot());
        }
        let ids = params
            .get("ids")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("ids is required"))?
            .iter()
            .map(|id| {
                id.as_str()
                    .ok_or_else(|| invalid("mosaic ROI IDs must be strings"))
                    .and_then(|id| self.item_index_for_roi(id))
                    .map(|index| self.items[index].id)
            })
            .collect::<Result<HashSet<_>, _>>()?;
        match mode {
            "replace" => self.selected_ids = ids,
            "add" => self.selected_ids.extend(ids),
            "remove" => self.selected_ids.retain(|id| !ids.contains(id)),
            "toggle" => {
                for id in ids {
                    if !self.selected_ids.insert(id) {
                        self.selected_ids.remove(&id);
                    }
                }
            }
            _ => return Err(invalid("unknown mosaic selection mode")),
        }
        Ok(self.selection_snapshot())
    }

    pub(super) fn clear_selection(&mut self) -> Result<Value, ControlError> {
        self.require_resource()?;
        self.selected_ids.clear();
        Ok(self.selection_snapshot())
    }

    pub(super) fn focus_snapshot(&self) -> Value {
        self.focused_id
            .and_then(|id| {
                self.items
                    .iter()
                    .position(|item| item.id == id)
                    .map(|index| {
                        let item = &self.items[index];
                        json!({
                            "index":index,
                            "id":item.id,
                            "roi_id":item.roi_id,
                            "metadata":item.metadata,
                        })
                    })
            })
            .unwrap_or(Value::Null)
    }

    pub(super) fn set_focus(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let index = if let Some(index) = params.get("index").and_then(Value::as_u64) {
            usize::try_from(index)
                .ok()
                .filter(|index| *index < self.items.len())
                .ok_or_else(|| invalid(format!("mosaic ROI index {index} is out of range")))?
        } else if let Some(roi_id) = params
            .get("roi_id")
            .or_else(|| params.get("id"))
            .and_then(Value::as_str)
        {
            self.item_index_for_roi(roi_id)?
        } else {
            return Err(invalid("provide index or roi_id"));
        };
        let before = self.focused_id;
        self.focused_id = Some(self.items[index].id);
        if params.get("fit").and_then(Value::as_bool).unwrap_or(true) {
            self.fit_bounds(self.items[index].bounds());
        }
        Ok(json!({
            "changed":before != self.focused_id,
            "focused":self.focus_snapshot(),
        }))
    }

    pub(super) fn step_focus(
        &mut self,
        params: &Value,
        forward: bool,
    ) -> Result<Value, ControlError> {
        self.require_resource()?;
        let step = params.get("step").and_then(Value::as_u64).unwrap_or(1) as usize;
        let wrap = params.get("wrap").and_then(Value::as_bool).unwrap_or(true);
        let current = self
            .focused_id
            .and_then(|id| self.items.iter().position(|item| item.id == id))
            .unwrap_or(0);
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
        let mut next =
            json!({"index":index,"fit":params.get("fit").and_then(Value::as_bool).unwrap_or(true)});
        if let Some(object) = next.as_object_mut() {
            object.insert("wrap".to_string(), Value::Bool(wrap));
        }
        self.set_focus(&next)
    }

    pub(super) fn fit_focus(&mut self) -> Result<Value, ControlError> {
        self.require_resource()?;
        let id = self
            .focused_id
            .ok_or_else(|| invalid("mosaic has no focused ROI"))?;
        let bounds = self
            .items
            .iter()
            .find(|item| item.id == id)
            .map(MosaicItemModel::bounds)
            .ok_or_else(|| invalid("focused mosaic ROI is not loaded"))?;
        self.fit_bounds(bounds);
        Ok(json!({"focused":self.focus_snapshot(),"camera":self.camera_snapshot()}))
    }

    pub(super) fn clear_focus(&mut self) -> Result<Value, ControlError> {
        self.require_resource()?;
        let changed = self.focused_id.take().is_some();
        Ok(json!({"changed":changed,"focused":null}))
    }

    pub(super) fn fit_all(&mut self) -> Result<Value, ControlError> {
        self.require_resource()?;
        self.fit_bounds(self.bounds);
        Ok(json!({"camera":self.camera_snapshot()}))
    }

    pub(crate) fn camera_snapshot(&self) -> Value {
        json!({
            "center_world_lvl0":self.camera_center,
            "zoom_screen_per_lvl0_px":self.camera_zoom,
        })
    }

    pub(super) fn fit_bounds(&mut self, bounds: [[f32; 2]; 2]) {
        let width = (bounds[1][0] - bounds[0][0]).max(1.0);
        let height = (bounds[1][1] - bounds[0][1]).max(1.0);
        self.camera_center = [
            (bounds[0][0] + bounds[1][0]) * 0.5,
            (bounds[0][1] + bounds[1][1]) * 0.5,
        ];
        self.camera_zoom = (self.logical_canvas[0] / width)
            .min(self.logical_canvas[1] / height)
            .max(0.000_01);
    }

    pub(super) fn apply_layout(&mut self) {
        let focused = self.focused_id;
        let group_by = self.group_by.clone();
        let sort_by = self.sort_by.clone();
        let secondary = self
            .sort_secondary_enabled
            .then(|| self.sort_by_secondary.clone());
        self.items.sort_by(|left, right| {
            if !group_by.is_empty() {
                let ordering = compare_sort_values(
                    &group_value(left, &group_by),
                    &group_value(right, &group_by),
                );
                if ordering != Ordering::Equal {
                    return ordering;
                }
            }
            let ordering =
                compare_sort_values(&left.sort_value(&sort_by), &right.sort_value(&sort_by));
            if ordering != Ordering::Equal {
                return ordering;
            }
            if let Some(secondary) = secondary.as_deref() {
                let ordering =
                    compare_sort_values(&left.sort_value(secondary), &right.sort_value(secondary));
                if ordering != Ordering::Equal {
                    return ordering;
                }
            }
            left.roi_id.cmp(&right.roi_id)
        });
        self.focused_id = focused
            .filter(|id| self.items.iter().any(|item| item.id == *id))
            .or_else(|| self.items.first().map(|item| item.id));

        let mut max_width = 1.0_f32;
        let mut y = 0.0_f32;
        if self.group_by.is_empty() {
            let [width, height] = layout_block(
                &mut self.items,
                0.0,
                self.columns,
                self.grid_cell_size,
                self.grid_pad,
                self.layout_mode,
            );
            self.bounds = [[0.0, 0.0], [width, height]];
            return;
        }

        let mut start = 0;
        while start < self.items.len() {
            let group = group_value(&self.items[start], &self.group_by).to_ascii_lowercase();
            let mut end = start + 1;
            while end < self.items.len()
                && group_value(&self.items[end], &self.group_by).to_ascii_lowercase() == group
            {
                end += 1;
            }
            let [width, height] = layout_block(
                &mut self.items[start..end],
                y + GROUP_HEADER_HEIGHT,
                self.columns,
                self.grid_cell_size,
                self.grid_pad,
                self.layout_mode,
            );
            max_width = max_width.max(width);
            y += GROUP_HEADER_HEIGHT + height;
            if end < self.items.len() {
                y += self.group_gap;
            }
            start = end;
        }
        self.bounds = [[0.0, 0.0], [max_width.max(1.0), y.max(1.0)]];
    }

    pub(super) fn item_index_for_roi(&self, roi_id: &str) -> Result<usize, ControlError> {
        let matches = self
            .items
            .iter()
            .enumerate()
            .filter(|(_, item)| item.roi_id == roi_id)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [index] => Ok(*index),
            [] => Err(ControlError::new(
                ControlErrorKind::ResourceNotFound,
                format!("mosaic ROI '{roi_id}' was not found"),
            )),
            _ => Err(invalid(format!("mosaic ROI '{roi_id}' is ambiguous"))),
        }
    }

    pub(super) fn metadata_columns(&self) -> &[String] {
        self.resource
            .as_ref()
            .map_or(&[], |resource| resource.metadata_columns.as_slice())
    }

    pub(super) fn require_resource(&self) -> Result<&Arc<ControlMosaicResource>, ControlError> {
        self.resource.as_ref().ok_or_else(|| {
            ControlError::new(
                ControlErrorKind::NotReady,
                "No mosaic resource is currently open",
            )
        })
    }
}
