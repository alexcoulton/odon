//! Canonical native-layer queries and presentation mutations.

use super::*;

impl AppModel {
    pub(super) fn native_layer_id<'a>(params: &'a Value) -> Result<&'a str, ControlError> {
        params
            .get("layer_id")
            .or_else(|| params.get("id"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| invalid("layer_id is required"))
    }

    pub(super) fn effective_native_layers(viewport: &ViewportModel) -> Vec<Value> {
        let mut layers = viewport.native_layers.snapshots();
        for layer in &mut layers {
            let Some(layer_id) = layer.get("layer_id").and_then(Value::as_str) else {
                continue;
            };
            if let Some(index) = layer_id
                .strip_prefix("channel:")
                .and_then(|value| value.parse::<usize>().ok())
                && let Some(channel) = viewport.channels.get(index)
            {
                layer["visible"] = Value::Bool(channel.visible);
                layer["presentation"] = json!({
                    "visible": channel.visible,
                    "color_rgb": channel.color_rgb,
                    "window": channel.window.map(|(min,max)| json!({"min":min,"max":max})),
                });
                layer["offset_world"] = json!(channel.offset_world);
                layer["order"] = json!(
                    viewport
                        .channel_order
                        .iter()
                        .position(|candidate| *candidate == index)
                        .unwrap_or(index)
                );
            } else if layer_id == "segmentation_objects" {
                layer["visible"] = Value::Bool(
                    viewport
                        .objects
                        .get("visible")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                );
                layer["presentation"] = viewport.objects.clone();
            } else if let Some(id) = layer_id
                .strip_prefix("spatial_shape:")
                .and_then(|value| value.parse::<u64>().ok())
                && let Some(objects) = viewport.secondary_objects.get(&id)
            {
                let presentation = spatial_object_native_presentation(&objects.objects);
                layer["visible"] = presentation["visible"].clone();
                layer["presentation"] = presentation;
            } else if layer_id == "segmentation_labels" {
                layer["visible"] = Value::Bool(viewport.segmentation_labels_visible);
                if let Some(presentation) = layer["presentation"].as_object_mut() {
                    presentation.insert(
                        "visible".to_string(),
                        Value::Bool(viewport.segmentation_labels_visible),
                    );
                }
            } else if layer_id == "segmentation_geojson" {
                layer["visible"] = Value::Bool(viewport.segmentation_geojson_visible);
                if let Some(presentation) = layer["presentation"].as_object_mut() {
                    presentation.insert(
                        "visible".to_string(),
                        Value::Bool(viewport.segmentation_geojson_visible),
                    );
                }
            }
        }
        let mut channels = layers
            .iter()
            .filter(|layer| layer.get("stack").and_then(Value::as_str) == Some("channels"))
            .cloned()
            .collect::<Vec<_>>();
        channels.sort_by_key(|layer| {
            layer
                .get("order")
                .and_then(Value::as_u64)
                .unwrap_or(u64::MAX)
        });
        let overlays = layers
            .into_iter()
            .filter(|layer| layer.get("stack").and_then(Value::as_str) == Some("overlays"));
        channels.extend(overlays);
        channels
    }

    pub(super) fn native_layers_for(&self, params: &Value) -> Result<Value, ControlError> {
        let viewport_id = Self::viewport_id(params)?;
        let workspace = &self.dataset()?.workspace;
        let viewport = workspace
            .get(&viewport_id)
            .ok_or_else(|| not_found(&viewport_id))?;
        Ok(viewport_response(
            workspace,
            &viewport_id,
            Value::Array(Self::effective_native_layers(&viewport.state)),
            vec![viewport_id.clone()],
            false,
        ))
    }

    pub(super) fn native_layer_for(&self, params: &Value) -> Result<Value, ControlError> {
        let viewport_id = Self::viewport_id(params)?;
        let layer_id = Self::native_layer_id(params)?;
        let workspace = &self.dataset()?.workspace;
        let viewport = workspace
            .get(&viewport_id)
            .ok_or_else(|| not_found(&viewport_id))?;
        let layer = Self::effective_native_layers(&viewport.state)
            .into_iter()
            .find(|layer| layer.get("layer_id").and_then(Value::as_str) == Some(layer_id))
            .ok_or_else(|| invalid(format!("native layer '{layer_id}' is not loaded")))?;
        Ok(viewport_response(
            workspace,
            &viewport_id,
            layer,
            vec![viewport_id.clone()],
            false,
        ))
    }

    pub(super) fn apply_native_layer_visibility(
        state: &mut ViewportModel,
        layer_id: &str,
        visible: bool,
    ) -> Result<bool, ControlError> {
        let mut changed = state.native_layers.set_visibility(layer_id, visible)?;
        if let Some(index) = layer_id
            .strip_prefix("channel:")
            .and_then(|value| value.parse::<usize>().ok())
        {
            let channel = state
                .channels
                .get_mut(index)
                .ok_or_else(|| invalid(format!("native layer '{layer_id}' is not loaded")))?;
            changed |= channel.visible != visible;
            channel.visible = visible;
        } else if layer_id == "segmentation_objects" {
            let previous = state
                .objects
                .get("visible")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            changed |= previous != visible;
            state
                .objects
                .as_object_mut()
                .expect("object presentation is normalized")
                .insert("visible".to_string(), Value::Bool(visible));
        } else if let Some(id) = layer_id
            .strip_prefix("spatial_shape:")
            .and_then(|value| value.parse::<u64>().ok())
        {
            let objects = state
                .secondary_objects
                .get_mut(&id)
                .ok_or_else(|| invalid(format!("native layer '{layer_id}' is not loaded")))?;
            let previous = objects
                .objects
                .get("visible")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            changed |= previous != visible;
            objects
                .objects
                .as_object_mut()
                .expect("secondary object presentation is normalized")
                .insert("visible".to_string(), Value::Bool(visible));
        } else if layer_id == "segmentation_labels" {
            changed |= state.segmentation_labels_visible != visible;
            state.segmentation_labels_visible = visible;
        } else if layer_id == "segmentation_geojson" {
            changed |= state.segmentation_geojson_visible != visible;
            state.segmentation_geojson_visible = visible;
        }
        Ok(changed)
    }

    pub(super) fn apply_native_layer_presentation(
        state: &mut ViewportModel,
        layer_id: &str,
        params: &Value,
    ) -> Result<bool, ControlError> {
        let presentation = params.get("presentation").unwrap_or(params);
        let mut changed = state
            .native_layers
            .set_presentation(layer_id, presentation)?;
        if let Some(index) = layer_id
            .strip_prefix("channel:")
            .and_then(|value| value.parse::<usize>().ok())
        {
            let channel = state
                .channels
                .get_mut(index)
                .ok_or_else(|| invalid(format!("native layer '{layer_id}' is not loaded")))?;
            let before = channel.clone();
            if let Some(visible) = presentation.get("visible").and_then(Value::as_bool) {
                channel.visible = visible;
            }
            if let Some(color) = presentation.get("color_rgb") {
                let values = color
                    .as_array()
                    .filter(|values| values.len() == 3)
                    .ok_or_else(|| {
                        invalid("color_rgb must contain three integers from 0 to 255")
                    })?;
                channel.color_rgb = [to_u8(&values[0])?, to_u8(&values[1])?, to_u8(&values[2])?];
            }
            if let Some(window) = presentation.get("window").filter(|value| !value.is_null()) {
                let (min, max) = if let Some(values) = window.as_array().filter(|v| v.len() == 2) {
                    (values[0].as_f64(), values[1].as_f64())
                } else {
                    (
                        window.get("min").and_then(Value::as_f64),
                        window.get("max").and_then(Value::as_f64),
                    )
                };
                let (Some(min), Some(max)) = (min, max) else {
                    return Err(invalid(
                        "window must be [min, max] or an object containing min and max",
                    ));
                };
                if !min.is_finite() || !max.is_finite() || max <= min {
                    return Err(invalid(
                        "window values must be finite and max must be greater than min",
                    ));
                }
                channel.window = Some((min as f32, max as f32));
            }
            changed |= *channel != before;
        } else if layer_id == "segmentation_objects" {
            changed |= apply_native_object_layer_presentation(&mut state.objects, presentation)?;
        } else if let Some(id) = layer_id
            .strip_prefix("spatial_shape:")
            .and_then(|value| value.parse::<u64>().ok())
        {
            let objects = state
                .secondary_objects
                .get_mut(&id)
                .ok_or_else(|| invalid(format!("native layer '{layer_id}' is not loaded")))?;
            let patch = presentation.get("objects").unwrap_or(presentation);
            changed |= apply_object_style_patch(&mut objects.objects, patch)?;
            if let Some(filter) = patch.get("filter") {
                let previous = objects.objects.get("filter").cloned();
                set_object_filter_model(&mut objects.objects, filter.clone());
                changed |= previous.as_ref() != Some(filter);
                objects.filter_indices = Arc::new(Vec::new());
                objects.filter_active = false;
                objects.filter_revision = objects.filter_revision.wrapping_add(1).max(1);
            }
        } else if layer_id == "segmentation_labels" {
            if let Some(visible) = presentation.get("visible").and_then(Value::as_bool) {
                changed |= state.segmentation_labels_visible != visible;
                state.segmentation_labels_visible = visible;
            }
        } else if layer_id == "segmentation_geojson" {
            if let Some(visible) = presentation.get("visible").and_then(Value::as_bool) {
                changed |= state.segmentation_geojson_visible != visible;
                state.segmentation_geojson_visible = visible;
            }
        }
        Ok(changed)
    }

    pub(super) fn mutate_native_layer(
        &mut self,
        params: &Value,
        operation: impl FnOnce(&mut ViewportModel, &str) -> Result<bool, ControlError>,
    ) -> Result<Value, ControlError> {
        let viewport_id = Self::viewport_id(params)?;
        let layer_id = Self::native_layer_id(params)?.to_string();
        let workspace = &mut self.dataset_mut()?.workspace;
        let active_before = workspace.active().state.clone();
        let viewport = workspace
            .get_mut(&viewport_id)
            .ok_or_else(|| not_found(&viewport_id))?;
        let changed = operation(&mut viewport.state, &layer_id)?;
        if changed {
            let _ = workspace.bump_presentation_revision(&viewport_id);
        }
        let layer = Self::effective_native_layers(&workspace.get(&viewport_id).unwrap().state)
            .into_iter()
            .find(|layer| layer.get("layer_id").and_then(Value::as_str) == Some(layer_id.as_str()))
            .expect("mutated native layer remains present");
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        Ok(viewport_response(
            workspace,
            &viewport_id,
            json!({"changed":changed,"layer":layer}),
            vec![viewport_id.clone()],
            active_changed,
        ))
    }

    pub(super) fn set_native_layer_visibility(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let visible = params
            .get("visible")
            .and_then(Value::as_bool)
            .ok_or_else(|| invalid("visible is required"))?;
        self.mutate_native_layer(params, |state, layer_id| {
            Self::apply_native_layer_visibility(state, layer_id, visible)
        })
    }

    pub(super) fn set_native_layer_presentation(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        self.mutate_native_layer(params, |state, layer_id| {
            Self::apply_native_layer_presentation(state, layer_id, params)
        })
    }

    pub(super) fn set_native_layer_active(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        self.mutate_native_layer(params, |state, layer_id| {
            state.native_layers.set_active(layer_id)
        })
    }

    pub(super) fn set_native_layer_order(&mut self, params: &Value) -> Result<Value, ControlError> {
        let viewport_id = Self::viewport_id(params)?;
        let stack = params
            .get("stack")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("stack is required"))?;
        let layers = params
            .get("layers")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("layers is required"))?
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .ok_or_else(|| invalid("layer IDs must be strings"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let workspace = &mut self.dataset_mut()?.workspace;
        let active_before = workspace.active().state.clone();
        let viewport = workspace
            .get_mut(&viewport_id)
            .ok_or_else(|| not_found(&viewport_id))?;
        let changed = viewport.state.native_layers.set_order(stack, &layers)?;
        if stack == "channels" {
            viewport.state.channel_order = layers
                .iter()
                .map(|id| {
                    id.strip_prefix("channel:")
                        .and_then(|value| value.parse::<usize>().ok())
                        .ok_or_else(|| invalid("channels stack accepts only channel layers"))
                })
                .collect::<Result<Vec<_>, _>>()?;
        }
        if changed {
            let _ = workspace.bump_presentation_revision(&viewport_id);
        }
        let snapshots = Self::effective_native_layers(&workspace.get(&viewport_id).unwrap().state);
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        Ok(viewport_response(
            workspace,
            &viewport_id,
            json!({"changed":changed,"layers":snapshots}),
            vec![viewport_id.clone()],
            active_changed,
        ))
    }

    pub(super) fn replace_native_layers(&mut self, params: &Value) -> Result<Value, ControlError> {
        let viewport_id = Self::viewport_id(params)?;
        let state = params
            .get("state")
            .or_else(|| params.get("layers"))
            .ok_or_else(|| invalid("state is required"))?;
        let workspace = &mut self.dataset_mut()?.workspace;
        let active_before = workspace.active().state.clone();
        let viewport = workspace
            .get_mut(&viewport_id)
            .ok_or_else(|| not_found(&viewport_id))?;
        let replacement_groups = params
            .get("channel_groups")
            .map(|groups| {
                groups
                    .as_array()
                    .ok_or_else(|| invalid("channel_groups must be an array"))
                    .and_then(|groups| {
                        parse_channel_groups_snapshot(groups, &viewport.state.channels)
                    })
            })
            .transpose()?;
        let mut changed = viewport.state.native_layers.replace(state)?;
        let snapshots = viewport.state.native_layers.snapshots();
        for layer in &snapshots {
            let Some(layer_id) = layer.get("layer_id").and_then(Value::as_str) else {
                continue;
            };
            if let Some(presentation) = layer.get("presentation") {
                Self::apply_native_layer_presentation(&mut viewport.state, layer_id, presentation)?;
            }
            if let Some(index) = layer_id
                .strip_prefix("channel:")
                .and_then(|value| value.parse::<usize>().ok())
                && let Some(offset) = layer
                    .get("offset_world")
                    .and_then(Value::as_array)
                    .filter(|values| values.len() == 2)
                    .and_then(|values| {
                        Some([values[0].as_f64()? as f32, values[1].as_f64()? as f32])
                    })
                && let Some(channel) = viewport.state.channels.get_mut(index)
            {
                channel.offset_world = offset;
            }
        }
        let channel_order = snapshots
            .iter()
            .filter(|layer| layer.get("stack").and_then(Value::as_str) == Some("channels"))
            .filter_map(|layer| {
                Some((
                    layer.get("order").and_then(Value::as_u64)?,
                    layer
                        .get("layer_id")
                        .and_then(Value::as_str)?
                        .strip_prefix("channel:")?
                        .parse::<usize>()
                        .ok()?,
                ))
            })
            .collect::<Vec<_>>();
        if channel_order.len() == viewport.state.channels.len() {
            let mut channel_order = channel_order;
            channel_order.sort_by_key(|(order, _)| *order);
            viewport.state.channel_order =
                channel_order.into_iter().map(|(_, index)| index).collect();
        }
        if let Some(index) = viewport
            .state
            .native_layers
            .active_layer_id()
            .and_then(|layer_id| layer_id.strip_prefix("channel:"))
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|index| *index < viewport.state.channels.len())
        {
            viewport.state.active_channel = index;
        }
        if let Some(groups) = replacement_groups {
            changed |= groups != viewport.state.channel_groups;
            viewport.state.channel_groups = groups;
        }
        if changed {
            let _ = workspace.bump_presentation_revision(&viewport_id);
        }
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        Ok(viewport_response(
            workspace,
            &viewport_id,
            json!({"changed":changed,"layers":Self::effective_native_layers(&workspace.get(&viewport_id).unwrap().state)}),
            vec![viewport_id.clone()],
            active_changed,
        ))
    }

    pub(super) fn active_scoped_native_params(
        &self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        self.active_scoped_params(params)
    }

    pub(super) fn native_layers_global(&self) -> Result<Value, ControlError> {
        let viewport = &self.dataset()?.workspace.active().state;
        Ok(json!({"mode":"single","layers":Self::effective_native_layers(viewport)}))
    }

    pub(super) fn native_layer_global(&self, params: &Value) -> Result<Value, ControlError> {
        let layer_id = Self::native_layer_id(params)?;
        let layer = Self::effective_native_layers(&self.dataset()?.workspace.active().state)
            .into_iter()
            .find(|layer| layer.get("layer_id").and_then(Value::as_str) == Some(layer_id))
            .ok_or_else(|| invalid(format!("native layer '{layer_id}' is not loaded")))?;
        Ok(json!({"mode":"single","layer":layer}))
    }

    pub(super) fn unwrap_native_global_result(
        &mut self,
        method: &str,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let scoped = self.active_scoped_native_params(params)?;
        let response = match method {
            "viewer.native_layers.set_active" => self.set_native_layer_active(&scoped)?,
            "viewer.native_layers.set_visibility" => self.set_native_layer_visibility(&scoped)?,
            "viewer.native_layers.set_order" => self.set_native_layer_order(&scoped)?,
            _ => unreachable!("global native layer mutation was checked"),
        };
        Ok(json!({"mode":"single","result":response["result"].clone()}))
    }

    pub(super) fn set_native_layer_offset_global(
        &mut self,
        params: &Value,
        reset: bool,
    ) -> Result<Value, ControlError> {
        let layer_id = Self::native_layer_id(params)?.to_string();
        let offset = if reset {
            None
        } else {
            Some(
                optional_finite_pair(params, "offset_world")?
                    .ok_or_else(|| invalid("offset_world is required"))?,
            )
        };
        let dataset = self.dataset_mut()?;
        let mut changed = false;
        let viewport_ids = dataset
            .workspace
            .viewports()
            .iter()
            .map(|viewport| viewport.id.clone())
            .collect::<Vec<_>>();
        for viewport in dataset.workspace.viewports_mut() {
            let layer_changed = if let Some(offset) = offset {
                viewport.state.native_layers.set_offset(&layer_id, offset)?
            } else {
                viewport.state.native_layers.reset_offset(&layer_id)?
            };
            changed |= layer_changed;
            if let Some(index) = layer_id
                .strip_prefix("channel:")
                .and_then(|value| value.parse::<usize>().ok())
                && let Some(channel) = viewport.state.channels.get_mut(index)
            {
                let effective = viewport
                    .state
                    .native_layers
                    .get(&layer_id)
                    .expect("offset native layer remains present")
                    .offset_world;
                changed |= channel.offset_world != effective;
                channel.offset_world = effective;
            }
        }
        if changed {
            for viewport_id in &viewport_ids {
                let _ = dataset.workspace.bump_presentation_revision(viewport_id);
            }
        }
        let layer = Self::effective_native_layers(&dataset.workspace.active().state)
            .into_iter()
            .find(|layer| layer.get("layer_id").and_then(Value::as_str) == Some(layer_id.as_str()))
            .expect("offset native layer remains present");
        Ok(json!({"mode":"single","result":{"changed":changed,"layer":layer}}))
    }
}
