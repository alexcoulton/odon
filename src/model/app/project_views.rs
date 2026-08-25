//! Project saved-view capture, normalization, and application.

use super::*;

impl AppModel {
    pub(super) fn project_view_spec(viewport: &ViewportModel, has_objects: bool) -> Value {
        let color_property = viewport
            .objects
            .get("color_property")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let color_mapping = viewport.objects.get("color_mapping").cloned();
        let overrides = viewport
            .objects
            .get("color_level_overrides")
            .and_then(Value::as_object);
        let hidden_cell_types = overrides
            .into_iter()
            .flatten()
            .filter(|(_, style)| style.get("visible").and_then(Value::as_bool) == Some(false))
            .map(|(value, _)| value.clone())
            .collect::<Vec<_>>();
        let visible_cell_types = if hidden_cell_types.is_empty() {
            Vec::new()
        } else {
            overrides
                .into_iter()
                .flatten()
                .filter(|(_, style)| style.get("visible").and_then(Value::as_bool) != Some(false))
                .map(|(value, _)| value.clone())
                .collect::<Vec<_>>()
        };
        let uses_objects = has_objects
            || color_property.is_some()
            || viewport
                .objects
                .get("fill_cells")
                .and_then(Value::as_bool)
                .unwrap_or(false);
        let active = viewport
            .channels
            .get(viewport.active_channel)
            .or_else(|| viewport.channels.first());
        let mut spec = json!({
            "channel_ref": active.map(|channel| json!({"label":channel.name,"alias":""})),
            "visible_channel_refs": viewport.channels.iter().filter(|channel| channel.visible).map(|channel| json!({"label":channel.name,"alias":""})).collect::<Vec<_>>(),
            "camera": {
                "center_world_lvl0": viewport.center,
                "zoom_screen_per_lvl0_px": viewport.zoom,
            },
        });
        if uses_objects {
            spec["segmentation_source"] = Value::String("geoparquet".to_string());
            spec["load_labels"] = Value::Bool(false);
            spec["cell_color_by"] =
                color_property.map_or(Value::Null, |value| Value::String(value.to_string()));
            if let Some(mapping) = color_mapping {
                spec["object_color_mapping"] = mapping;
            }
            spec["visible_cell_types"] = json!(visible_cell_types);
            spec["hidden_cell_types"] = json!(hidden_cell_types);
            spec["fill_cells"] = viewport
                .objects
                .get("fill_cells")
                .cloned()
                .unwrap_or(Value::Bool(false));
            spec["show_selection_overlay"] = viewport
                .objects
                .get("show_selection_overlay")
                .cloned()
                .unwrap_or(Value::Bool(true));
        }
        spec
    }

    pub(super) fn capture_project_view(&mut self, params: &Value) -> Result<Value, ControlError> {
        let name = required_nonempty_string(params, &["name"], "name")?;
        let viewport_id = params
            .get("viewport_id")
            .and_then(Value::as_str)
            .map(ViewportId::new)
            .transpose()
            .map_err(|error| invalid(error.to_string()))?
            .unwrap_or_else(|| self.dataset().unwrap().workspace.active_id().clone());
        let spec = {
            let dataset = self.dataset()?;
            let viewport = dataset
                .workspace
                .get(&viewport_id)
                .ok_or_else(|| not_found(&viewport_id))?;
            Self::project_view_spec(&viewport.state, dataset.object_resource.is_some())
        };
        let view = self
            .project
            .dispatch("project.views.create", &json!({"name":name,"spec":spec}))?;
        self.project_initialized = true;
        Ok(json!({
            "captured":true,
            "viewport_id":params.get("viewport_id").cloned().unwrap_or(Value::Null),
            "view":view,
        }))
    }

    pub(super) fn saved_view_channel_index(
        channels: &[ModelChannel],
        spec: &Value,
    ) -> Result<Option<usize>, ControlError> {
        let candidates = spec
            .get("channel_ref")
            .and_then(Value::as_object)
            .map(|reference| {
                ["alias", "label"]
                    .into_iter()
                    .filter_map(|name| reference.get(name).and_then(Value::as_str))
                    .chain(spec.get("channel").and_then(Value::as_str))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| {
                spec.get("channel")
                    .and_then(Value::as_str)
                    .into_iter()
                    .collect()
            });
        for candidate in candidates {
            if candidate.trim().is_empty() {
                continue;
            }
            if let Ok(index) = resolve_channel(channels, &Value::String(candidate.to_string())) {
                return Ok(Some(index));
            }
        }
        Ok(None)
    }

    pub(super) fn saved_view_visible_channel_indices(
        channels: &[ModelChannel],
        spec: &Value,
    ) -> Result<Vec<usize>, ControlError> {
        let mut indices = Vec::new();
        if let Some(references) = spec.get("visible_channel_refs").and_then(Value::as_array) {
            for reference in references {
                let mut found = None;
                for candidate in ["alias", "label"]
                    .into_iter()
                    .filter_map(|name| reference.get(name).and_then(Value::as_str))
                {
                    if let Ok(index) =
                        resolve_channel(channels, &Value::String(candidate.to_string()))
                    {
                        found = Some(index);
                        break;
                    }
                }
                if let Some(index) = found
                    && !indices.contains(&index)
                {
                    indices.push(index);
                }
            }
        }
        if let Some(names) = spec.get("visible_channels").and_then(Value::as_array) {
            for name in names {
                let index = resolve_channel(channels, name)?;
                if !indices.contains(&index) {
                    indices.push(index);
                }
            }
        }
        Ok(indices)
    }

    pub(super) fn apply_project_view(&mut self, params: &Value) -> Result<Value, ControlError> {
        let view = self.project.dispatch("project.views.get", params)?;
        let spec = view.get("spec").cloned().unwrap_or_else(|| json!({}));
        let dataset = self.dataset_mut()?;
        let viewport_id = dataset.workspace.active_id().clone();
        let before = dataset.workspace.active().state.clone();
        let viewport = &mut dataset.workspace.active_mut().state;

        if let Some(index) = Self::saved_view_channel_index(&viewport.channels, &spec)? {
            viewport.active_channel = index;
        }
        let visible = Self::saved_view_visible_channel_indices(&viewport.channels, &spec)?;
        if !visible.is_empty() {
            for (index, channel) in viewport.channels.iter_mut().enumerate() {
                channel.visible = visible.contains(&index);
            }
        }
        if let Some(hidden) = spec.get("hidden_channels").and_then(Value::as_array) {
            for selector in hidden {
                let index = resolve_channel(&viewport.channels, selector)?;
                viewport.channels[index].visible = false;
            }
        }
        if let Some(value) = spec.get("cell_color_by") {
            viewport
                .objects
                .as_object_mut()
                .expect("object presentation is normalized")
                .insert("color_property".to_string(), value.clone());
        }
        if let Some(value) = spec.get("object_color_mapping") {
            let mapping = normalize_object_color_mapping(value)?;
            let property = mapping
                .get("property")
                .and_then(Value::as_str)
                .map(str::to_string)
                .map(Value::String)
                .unwrap_or(Value::Null);
            let objects = viewport
                .objects
                .as_object_mut()
                .expect("object presentation is normalized");
            objects.insert("color_property".to_string(), property);
            objects.insert("color_mapping".to_string(), mapping);
        }
        for (name, visible) in [("visible_cell_types", true), ("hidden_cell_types", false)] {
            if let Some(values) = spec.get(name).and_then(Value::as_array) {
                let overrides = viewport
                    .objects
                    .as_object_mut()
                    .expect("object presentation is normalized")
                    .entry("color_level_overrides")
                    .or_insert_with(|| json!({}))
                    .as_object_mut()
                    .expect("object legend overrides are an object");
                for value in values.iter().filter_map(Value::as_str) {
                    overrides
                        .entry(value.to_string())
                        .or_insert_with(|| json!({}))["visible"] = Value::Bool(visible);
                }
            }
        }
        for name in ["fill_cells", "show_selection_overlay"] {
            if let Some(value) = spec.get(name).and_then(Value::as_bool) {
                viewport
                    .objects
                    .as_object_mut()
                    .expect("object presentation is normalized")
                    .insert(name.to_string(), Value::Bool(value));
            }
        }
        if let Some(camera) = spec.get("camera") {
            if let Some(center) = camera
                .get("center_world_lvl0")
                .and_then(Value::as_array)
                .filter(|values| values.len() == 2)
            {
                viewport.center = [
                    center[0]
                        .as_f64()
                        .ok_or_else(|| invalid("saved view camera x is invalid"))?
                        as f32,
                    center[1]
                        .as_f64()
                        .ok_or_else(|| invalid("saved view camera y is invalid"))?
                        as f32,
                ];
            }
            if let Some(zoom) = camera
                .get("zoom_screen_per_lvl0_px")
                .and_then(Value::as_f64)
            {
                if !zoom.is_finite() || zoom <= 0.0 {
                    return Err(invalid(
                        "saved view camera zoom must be positive and finite",
                    ));
                }
                viewport.zoom = zoom as f32;
            }
        }
        let after = viewport.clone();
        let navigation_changed = after.center != before.center || after.zoom != before.zoom;
        let presentation_changed = presentation_changed(&before, &after);
        if navigation_changed {
            let _ = dataset.workspace.bump_navigation_revision(&viewport_id);
            if dataset.workspace.links().camera {
                propagate_camera(&mut dataset.workspace, &viewport_id, &after);
            }
        }
        if presentation_changed {
            let _ = dataset.workspace.bump_presentation_revision(&viewport_id);
        }
        Ok(json!({"applied":true,"view":view}))
    }
}
