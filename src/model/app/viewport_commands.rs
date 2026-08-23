//! Per-viewport workspace, navigation, channel, object, and rendering commands.

use super::*;

mod navigation;
mod presentation;
mod workspace;

impl AppModel {
    pub(super) fn dataset(&self) -> Result<&DatasetModel, ControlError> {
        self.dataset
            .as_ref()
            .ok_or_else(|| wrong_mode("No dataset viewer is currently open."))
    }

    pub(super) fn dataset_mut(&mut self) -> Result<&mut DatasetModel, ControlError> {
        self.dataset
            .as_mut()
            .ok_or_else(|| wrong_mode("No dataset viewer is currently open."))
    }

    pub(super) fn viewport_id(params: &Value) -> Result<ViewportId, ControlError> {
        let id = params
            .get("viewport_id")
            .or_else(|| params.get("id"))
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("viewport_id is required"))?;
        ViewportId::new(id).map_err(|error| invalid(error.to_string()))
    }

    pub(super) fn check_viewport_revision(&self, params: &Value) -> Result<(), ControlError> {
        let navigation = params.get("if_navigation_revision").and_then(Value::as_u64);
        let presentation = params
            .get("if_presentation_revision")
            .and_then(Value::as_u64);
        if navigation.is_none() && presentation.is_none() {
            return Ok(());
        }
        let id = Self::viewport_id(params)?;
        let workspace = &self.dataset()?.workspace;
        let viewport = workspace.get(&id).ok_or_else(|| not_found(&id))?;
        if let Some(expected) = navigation
            && expected != viewport.navigation_revision
        {
            return Err(ControlError::new(
                ControlErrorKind::Conflict,
                format!(
                    "viewport navigation revision conflict: expected {expected}, current {}",
                    viewport.navigation_revision
                ),
            )
            .with_data(json!({
                "viewport_id": id.as_str(),
                "expected_revision": expected,
                "current_revision": viewport.navigation_revision,
                "revision_domain": "navigation",
            })));
        }
        if let Some(expected) = presentation
            && expected != viewport.presentation_revision
        {
            return Err(ControlError::new(
                ControlErrorKind::Conflict,
                format!(
                    "viewport presentation revision conflict: expected {expected}, current {}",
                    viewport.presentation_revision
                ),
            )
            .with_data(json!({
                "viewport_id": id.as_str(),
                "expected_revision": expected,
                "current_revision": viewport.presentation_revision,
                "revision_domain": "presentation",
            })));
        }
        Ok(())
    }

    pub(super) fn channels_snapshot(&self) -> Result<Value, ControlError> {
        let dataset = self.dataset()?;
        let active = dataset.workspace.active();
        Ok(json!({
            "mode": "single",
            "channels": active.state.channels.iter().enumerate().map(|(index, channel)| {
                let mut value = channel_json(channel, index == active.state.active_channel);
                value.as_object_mut().expect("channel snapshot is an object").insert(
                    "note".to_string(),
                    Value::String(channel.note.clone()),
                );
                value
            }).collect::<Vec<_>>(),
        }))
    }

    pub(super) fn visible_channels_snapshot(&self) -> Result<Value, ControlError> {
        let active = self.dataset()?.workspace.active();
        Ok(json!({
            "mode": "single",
            "channels": visible_channels_json(&active.state),
        }))
    }

    pub(super) fn active_channel_snapshot(&self) -> Result<Value, ControlError> {
        let active = self.dataset()?.workspace.active();
        let channel = active
            .state
            .channels
            .get(active.state.active_channel)
            .map(active_channel_json)
            .unwrap_or(Value::Null);
        Ok(json!({"mode": "single", "active_channel": channel}))
    }

    pub(super) fn active_scoped_params(&self, params: &Value) -> Result<Value, ControlError> {
        let mut params = params.clone();
        params
            .as_object_mut()
            .ok_or_else(|| invalid("params must be an object"))?
            .insert(
                "viewport_id".to_string(),
                Value::String(self.dataset()?.workspace.active_id().as_str().to_string()),
            );
        Ok(params)
    }

    pub(super) fn get_object_style_global(&self, params: &Value) -> Result<Value, ControlError> {
        let params = self.active_scoped_params(params)?;
        Ok(self.get_object_style(&params)?["result"].clone())
    }

    pub(super) fn object_overlay_visibility_global(
        &self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let target = params
            .get("target")
            .and_then(Value::as_str)
            .unwrap_or("objects");
        let dataset = self.dataset()?;
        let viewport = &dataset.workspace.active().state;
        Ok(json!({
            "mode":"single",
            "overlay":{
                "target":target,
                "segmentation_labels":viewport.segmentation_labels_visible,
                "segmentation_geojson":viewport.segmentation_geojson_visible,
                "segmentation_objects":viewport.objects
                    .get("visible")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                "object_count":dataset.object_resource
                    .as_ref()
                    .map_or(0, |resource| resource.features.len()),
            },
        }))
    }

    pub(super) fn set_object_overlay_visibility_global(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let visible = params
            .get("visible")
            .and_then(Value::as_bool)
            .ok_or_else(|| invalid("set_object_overlay_visibility requires visible"))?;
        let target = params
            .get("target")
            .and_then(Value::as_str)
            .unwrap_or("objects");
        if !matches!(target, "objects" | "labels" | "geojson" | "all") {
            return Err(invalid(format!("unknown overlay target '{target}'")));
        }

        let dataset = self.dataset_mut()?;
        let viewport_id = dataset.workspace.active_id().clone();
        let viewport = &mut dataset.workspace.active_mut().state;
        let mut changed = false;
        if matches!(target, "objects" | "all") {
            let current = viewport
                .objects
                .get("visible")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            changed |= current != visible;
            viewport
                .objects
                .as_object_mut()
                .expect("object presentation is normalized")
                .insert("visible".to_string(), Value::Bool(visible));
            if viewport.native_layers.get("segmentation_objects").is_some() {
                changed |= viewport
                    .native_layers
                    .set_visibility("segmentation_objects", visible)?;
            }
        }
        if matches!(target, "labels" | "all") {
            changed |= viewport.segmentation_labels_visible != visible;
            viewport.segmentation_labels_visible = visible;
            if viewport.native_layers.get("segmentation_labels").is_some() {
                changed |= viewport
                    .native_layers
                    .set_visibility("segmentation_labels", visible)?;
            }
        }
        if matches!(target, "geojson" | "all") {
            changed |= viewport.segmentation_geojson_visible != visible;
            viewport.segmentation_geojson_visible = visible;
            if viewport.native_layers.get("segmentation_geojson").is_some() {
                changed |= viewport
                    .native_layers
                    .set_visibility("segmentation_geojson", visible)?;
            }
        }
        if changed {
            let _ = dataset.workspace.bump_presentation_revision(&viewport_id);
        }
        self.object_overlay_visibility_global(params)
    }

    pub(super) fn set_object_style_global(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let params = self.active_scoped_params(params)?;
        Ok(self.set_object_style(&params)?["result"].clone())
    }

    pub(super) fn set_object_legend_global(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let params = self.active_scoped_params(params)?;
        Ok(self.set_object_legend(&params)?["result"].clone())
    }

    pub(super) fn get_fast_object_rendering_global(
        &self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let target = self.resolve_object_target(params)?;
        let objects = self
            .dataset()?
            .workspace
            .active()
            .state
            .object_presentation(target)
            .ok_or_else(|| object_target_not_found(target))?;
        Ok(json!({
            "enabled":objects.get("fast_rendering").and_then(Value::as_bool).unwrap_or(true),
        }))
    }

    pub(super) fn set_fast_object_rendering_global(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        self.resolve_object_target(params)?;
        let enabled = params
            .get("enabled")
            .and_then(Value::as_bool)
            .ok_or_else(|| invalid("enabled is required"))?;
        let mut scoped = self.active_scoped_params(params)?;
        scoped
            .as_object_mut()
            .expect("active params are an object")
            .insert("fast_rendering".to_string(), Value::Bool(enabled));
        let response = self.set_object_style(&scoped)?;
        Ok(json!({
            "enabled":enabled,
            "changed":response["result"]["changed"],
        }))
    }

    pub(super) fn get_object_filter_global(&self, params: &Value) -> Result<Value, ControlError> {
        let target = self.resolve_object_target(params)?;
        let params = self.active_scoped_params(params)?;
        let mut response = json!({"filter":self.get_object_filter(&params)?["result"].clone()});
        Self::decorate_object_target(&mut response, target);
        Ok(response)
    }

    pub(super) fn clear_object_filter_global(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let target = self.resolve_object_target(params)?;
        let params = self.active_scoped_params(params)?;
        let mut response = json!({"filter":self.clear_object_filter(&params)?["result"].clone()});
        Self::decorate_object_target(&mut response, target);
        Ok(response)
    }

    pub(super) fn set_active_channel_global(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let params = self.active_scoped_params(params)?;
        let response = self.set_active_channel(&params)?;
        Ok(json!({"mode": "single", "result": response["result"]}))
    }

    pub(super) fn set_visible_channels_global(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let params = self.active_scoped_params(params)?;
        let response = self.set_visible_channels(&params)?;
        Ok(json!({"mode": "single", "result": response["result"]}))
    }

    pub(super) fn get_channel_contrast_global(
        &self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let dataset = self.dataset()?;
        let viewport = &dataset.workspace.active().state;
        let index = if params.as_object().is_some_and(|object| !object.is_empty()) {
            resolve_channel(&viewport.channels, channel_selector_from_params(params)?)?
        } else {
            viewport.active_channel
        };
        Ok(json!({
            "mode": "single",
            "contrast": contrast_json(
                &viewport.channels[index],
                index,
                dataset.descriptor.abs_max.max(1.0),
            ),
        }))
    }

    pub(super) fn set_channel_contrast_global(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let params = self.active_scoped_params(params)?;
        let response = self.set_channel_contrast(&params)?;
        Ok(json!({"mode": "single", "contrast": response["result"]}))
    }

    pub(super) fn set_channel_color_global(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let params = self.active_scoped_params(params)?;
        let response = self.set_channel_color(&params)?;
        Ok(json!({"mode": "single", "result": response["result"]}))
    }

    pub(super) fn set_channel_note_global(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let selector = channel_selector_from_params(params)?.clone();
        let note = params
            .get("note")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("set_channel_note requires note"))?
            .to_string();
        let dataset = self.dataset_mut()?;
        let index = resolve_channel(&dataset.workspace.active().state.channels, &selector)?;
        let changed = dataset
            .workspace
            .active()
            .state
            .channels
            .get(index)
            .is_some_and(|channel| channel.note != note);
        for slot in dataset.workspace.viewports_mut() {
            if let Some(channel) = slot.state.channels.get_mut(index) {
                channel.note.clone_from(&note);
            }
        }
        let channel = full_channel_json(
            &dataset.workspace.active().state.channels[index],
            index == dataset.workspace.active().state.active_channel,
        );
        Ok(json!({"changed": changed, "channel": channel}))
    }

    pub(super) fn get_channel_transform(&self, params: &Value) -> Result<Value, ControlError> {
        let viewport = &self.dataset()?.workspace.active().state;
        let index = resolve_channel(&viewport.channels, channel_selector_from_params(params)?)?;
        Ok(channel_transform_json(&viewport.channels[index], index))
    }

    pub(super) fn set_channel_transform(&mut self, params: &Value) -> Result<Value, ControlError> {
        let selector = channel_selector_from_params(params)?.clone();
        let offset = optional_finite_pair(params, "offset_world")?;
        let scale = optional_finite_pair(params, "scale")?;
        if let Some([x, y]) = scale
            && (!(0.01..=100.0).contains(&x) || !(0.01..=100.0).contains(&y))
        {
            return Err(invalid("scale values must be between 0.01 and 100"));
        }
        let rotation = match params.get("rotation_rad") {
            Some(value) => Some(
                value
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| invalid("rotation_rad must be a finite number"))?
                    as f32,
            ),
            None => None,
        };
        let dataset = self.dataset_mut()?;
        let index = resolve_channel(&dataset.workspace.active().state.channels, &selector)?;
        let before = dataset.workspace.active().state.channels[index].clone();
        for slot in dataset.workspace.viewports_mut() {
            let channel = &mut slot.state.channels[index];
            if let Some(offset) = offset {
                channel.offset_world = offset;
            }
            if let Some(scale) = scale {
                channel.scale = scale;
            }
            if let Some(rotation) = rotation {
                channel.rotation_rad = rotation;
            }
        }
        let channel = &dataset.workspace.active().state.channels[index];
        let changed = before.offset_world != channel.offset_world
            || before.scale != channel.scale
            || before.rotation_rad != channel.rotation_rad;
        let transform = channel_transform_json(channel, index);
        if changed {
            let viewport_ids = dataset
                .workspace
                .viewports()
                .iter()
                .map(|viewport| viewport.id.clone())
                .collect::<Vec<_>>();
            for viewport_id in viewport_ids {
                let _ = dataset.workspace.bump_presentation_revision(&viewport_id);
            }
        }
        Ok(json!({
            "changed": changed,
            "transform": transform,
        }))
    }

    pub(super) fn reset_channel_transform(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let mut reset = params.clone();
        let object = reset
            .as_object_mut()
            .ok_or_else(|| invalid("params must be an object"))?;
        object.insert("offset_world".to_string(), json!([0.0, 0.0]));
        object.insert("scale".to_string(), json!([1.0, 1.0]));
        object.insert("rotation_rad".to_string(), json!(0.0));
        self.set_channel_transform(&reset)
    }

    pub(super) fn set_channel_order_global(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let params = self.active_scoped_params(params)?;
        let response = self.set_channel_order(&params)?;
        Ok(response["result"].clone())
    }

    pub(super) fn channel_presentation_global(&self) -> Result<Value, ControlError> {
        Ok(channel_presentation_json(
            &self.dataset()?.workspace.active().state,
        ))
    }

    pub(super) fn set_channel_presentation_global(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let params = self.active_scoped_params(params)?;
        let response = self.set_channel_presentation(&params)?;
        Ok(response["result"].clone())
    }

    pub(super) fn channel_groups_global(&self) -> Result<Value, ControlError> {
        Ok(json!({
            "mode": "single",
            "groups": channel_groups_json(&self.dataset()?.workspace.active().state),
        }))
    }

    pub(super) fn set_channel_group_global(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let params = self.active_scoped_params(params)?;
        let response = self.set_channel_group(&params)?;
        Ok(json!({"mode": "single", "result": response["result"]}))
    }

    pub(super) fn get_camera_global(&self) -> Result<Value, ControlError> {
        let viewport = &self.dataset()?.workspace.active().state;
        Ok(json!({"mode": "single", "camera": control_camera_json(viewport)}))
    }

    pub(super) fn set_camera_global(&mut self, params: &Value) -> Result<Value, ControlError> {
        let params = self.active_scoped_params(params)?;
        let response = self.set_camera(&params)?;
        Ok(json!({"mode": "single", "camera": response["result"]}))
    }

    pub(super) fn zoom_camera_global(
        &mut self,
        params: &Value,
        zoom_in: bool,
    ) -> Result<Value, ControlError> {
        let raw_factor = params.get("factor").and_then(Value::as_f64).unwrap_or(1.5);
        let factor = if zoom_in {
            raw_factor
        } else if raw_factor > 0.0 {
            1.0 / raw_factor
        } else {
            raw_factor
        };
        if !factor.is_finite() || factor <= 0.0 {
            return Err(invalid("zoom factor must be finite and > 0"));
        }
        let current = self.dataset()?.workspace.active().state.zoom;
        self.set_camera_global(&json!({"zoom": current as f64 * factor}))
    }

    pub(super) fn fit_camera_global(&mut self) -> Result<Value, ControlError> {
        let params = self.active_scoped_params(&json!({}))?;
        let response = self.fit_viewport(&params)?;
        Ok(json!({"mode": "single", "camera": response["result"]}))
    }

    pub(super) fn get_plane_global(&self) -> Result<Value, ControlError> {
        let dataset = self.dataset()?;
        Ok(json!({
            "mode": "single",
            "plane": control_plane_json(
                &dataset.workspace.active().state,
                dataset.plane_extents,
                dataset.orthogonal_planes,
            ),
        }))
    }

    pub(super) fn set_plane_global(&mut self, params: &Value) -> Result<Value, ControlError> {
        let params = self.active_scoped_params(params)?;
        let response = self.set_plane(&params)?;
        Ok(json!({"mode": "single", "result": response["result"]}))
    }

    pub(super) fn step_plane_global(
        &mut self,
        params: &Value,
        forward: bool,
    ) -> Result<Value, ControlError> {
        let step = params.get("step").and_then(Value::as_u64).unwrap_or(1);
        let wrap = params.get("wrap").and_then(Value::as_bool).unwrap_or(false);
        let dataset = self.dataset()?;
        let viewport = &dataset.workspace.active().state;
        let current = current_plane_slice(viewport);
        let extent = dataset.plane_extents[plane_mode_index(&viewport.plane_mode)].max(1);
        let last = extent.saturating_sub(1);
        let next = if wrap {
            let offset = step % extent;
            if forward {
                (current + offset) % extent
            } else {
                (current + extent - offset) % extent
            }
        } else if forward {
            current.saturating_add(step).min(last)
        } else {
            current.saturating_sub(step)
        };
        self.set_plane_global(&json!({"slice": next}))
    }

    pub(super) fn plane_operation_availability(&self) -> Result<Value, ControlError> {
        let dataset = self.dataset()?;
        let plane = control_plane_json(
            &dataset.workspace.active().state,
            dataset.plane_extents,
            dataset.orthogonal_planes,
        );
        let xy = plane["mode"] == "xy";
        let operation = |requires_xy: bool| {
            json!({
                "available": !requires_xy || xy,
                "reason": (requires_xy && !xy).then_some("operation requires the XY view plane"),
            })
        };
        Ok(json!({
            "plane": plane,
            "operations": {
                "measurements": operation(true),
                "memory_pin": operation(true),
                "channel_max": operation(true),
                "threshold_preview": operation(true),
                "object_selection": operation(false),
            }
        }))
    }

    pub(super) fn get_smooth_pixels_global(&self) -> Result<Value, ControlError> {
        Ok(json!({
            "mode": "single",
            "smooth_pixels": {"smooth": self.dataset()?.workspace.active().state.smooth_pixels},
        }))
    }

    pub(super) fn set_smooth_pixels_global(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let smooth = params
            .get("smooth")
            .and_then(Value::as_bool)
            .ok_or_else(|| invalid("set_smooth_pixels requires smooth"))?;
        let params = self.active_scoped_params(&json!({"smooth_pixels": smooth}))?;
        let response = self.set_rendering(&params)?;
        Ok(json!({
            "mode": "single",
            "result": {
                "changed": response["result"]["changed"],
                "smooth_pixels": {"smooth": smooth},
            }
        }))
    }

    pub(super) fn get_scale_bar_global(&self) -> Result<Value, ControlError> {
        Ok(json!({
            "visible":self.dataset()?.workspace.active().state.show_scale_bar,
            "supported":true,
        }))
    }

    pub(super) fn set_scale_bar_global(&mut self, params: &Value) -> Result<Value, ControlError> {
        let visible = params
            .get("visible")
            .and_then(Value::as_bool)
            .ok_or_else(|| invalid("visible must be a boolean"))?;
        let params = self.active_scoped_params(&json!({"show_scale_bar":visible}))?;
        self.set_rendering(&params)?;
        Ok(json!({"visible":visible,"supported":true}))
    }

    pub(super) fn threshold_levels(&self) -> Result<Value, ControlError> {
        let dataset = self.dataset()?;
        let levels = dataset
            .descriptor
            .levels
            .iter()
            .map(|level| {
                let width = level.shape.get(dataset.descriptor.dims.x).copied();
                let height = level.shape.get(dataset.descriptor.dims.y).copied();
                let pixel_count = width.zip(height).and_then(|(width, height)| {
                    width.checked_mul(height)
                });
                json!({
                    "index":level.index,
                    "downsample":level.downsample,
                    "width":width,
                    "height":height,
                    "pixel_count":pixel_count,
                    "interactive":pixel_count.is_some_and(|pixels| pixels <= THRESHOLD_REGION_MAX_INTERACTIVE_PIXELS),
                })
            })
            .collect::<Vec<_>>();
        let default_full_level = levels.iter().find_map(|level| {
            level
                .get("interactive")
                .and_then(Value::as_bool)
                .filter(|interactive| *interactive)
                .and_then(|_| level.get("index"))
                .and_then(Value::as_u64)
        });
        Ok(json!({
            "max_interactive_pixels":THRESHOLD_REGION_MAX_INTERACTIVE_PIXELS,
            "default_full_level":default_full_level,
            "levels":levels,
        }))
    }

    pub(super) fn get_panels(&self) -> Result<Value, ControlError> {
        let dataset = self.dataset()?;
        Ok(json!({
            "mode": "single",
            "panels": {
                "left": dataset.show_left_panel,
                "right": dataset.show_right_panel,
            },
        }))
    }

    pub(super) fn set_panels(&mut self, params: &Value) -> Result<Value, ControlError> {
        let left = params
            .get("left")
            .map(|value| {
                value
                    .as_bool()
                    .ok_or_else(|| invalid("left must be a boolean"))
            })
            .transpose()?;
        let right = params
            .get("right")
            .map(|value| {
                value
                    .as_bool()
                    .ok_or_else(|| invalid("right must be a boolean"))
            })
            .transpose()?;
        if left.is_none() && right.is_none() {
            return Err(invalid("set_side_panels requires left and/or right"));
        }
        let dataset = self.dataset_mut()?;
        let before_left = dataset.show_left_panel;
        let before_right = dataset.show_right_panel;
        if let Some(left) = left {
            dataset.show_left_panel = left;
        }
        if let Some(right) = right {
            dataset.show_right_panel = right;
        }
        let changed =
            before_left != dataset.show_left_panel || before_right != dataset.show_right_panel;
        if changed {
            if before_left != dataset.show_left_panel {
                let delta = if dataset.show_left_panel {
                    -DEFAULT_LEFT_PANEL_WIDTH
                } else {
                    DEFAULT_LEFT_PANEL_WIDTH
                };
                dataset.logical_workspace_size[0] =
                    (dataset.logical_workspace_size[0] + delta).max(1.0);
            }
            if before_right != dataset.show_right_panel {
                let delta = if dataset.show_right_panel {
                    -DEFAULT_RIGHT_PANEL_WIDTH
                } else {
                    DEFAULT_RIGHT_PANEL_WIDTH
                };
                dataset.logical_workspace_size[0] =
                    (dataset.logical_workspace_size[0] + delta).max(1.0);
            }
            update_logical_geometry(dataset);
        }
        Ok(json!({
            "mode": "single",
            "result": {
                "changed": changed,
                "panels": {
                    "left": dataset.show_left_panel,
                    "right": dataset.show_right_panel,
                },
            },
        }))
    }

    pub(super) fn set_right_tab(&mut self, params: &Value) -> Result<Value, ControlError> {
        let tab = params
            .get("tab")
            .or_else(|| params.get("right_tab"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|tab| !tab.is_empty())
            .ok_or_else(|| invalid("set_right_tab requires tab"))?;
        if !matches!(
            tab,
            "properties" | "views" | "analysis" | "measurements" | "memory" | "roi_selector"
        ) {
            return Err(invalid(
                "unknown right tab; expected properties, views, analysis, measurements, memory, or roi_selector",
            ));
        }
        self.dataset_mut()?.right_tab = tab.to_string();
        Ok(json!({"mode":"single","tab":{"right_tab":tab}}))
    }
}
