//! Mosaic channel, native-layer, camera, panel, and rendering presentation state.

use super::*;

impl MosaicModel {
    pub(crate) fn apply_fast_object_rendering_setting(&mut self, enabled: bool) {
        self.fast_object_rendering = enabled;
    }

    pub(super) fn channel_presentation_snapshot(&self) -> Value {
        json!({
            "search":self.channel_search,
            "sort":self.channel_sort,
            "order":self.channel_order.iter().map(|index| json!({
                "index":index,
                "name":self.channels[*index].name,
                "visible":self.channels[*index].visible,
            })).collect::<Vec<_>>(),
        })
    }

    pub(super) fn channel_presentation(&self) -> Result<Value, ControlError> {
        self.require_resource()?;
        Ok(json!({"mode":"mosaic","presentation":self.channel_presentation_snapshot()}))
    }

    pub(super) fn set_channel_presentation(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        self.require_resource()?;
        if let Some(search) = params.get("search") {
            self.channel_search = search
                .as_str()
                .ok_or_else(|| invalid("search must be a string"))?
                .to_string();
        }
        if let Some(sort) = params.get("sort") {
            self.channel_sort = canonical_channel_sort(
                sort.as_str()
                    .ok_or_else(|| invalid("sort must be a string"))?,
            )
            .ok_or_else(|| invalid("unknown channel sort mode"))?
            .to_string();
        }
        Ok(json!({"mode":"mosaic","presentation":self.channel_presentation_snapshot()}))
    }

    pub(super) fn channel_groups_snapshot(&self) -> Value {
        Value::Array(
            self.layer_groups
                .channel_groups
                .iter()
                .map(|group| {
                    let members = self
                        .channels
                        .iter()
                        .filter_map(|channel| {
                            let member = self.layer_groups.channel_members.get(&channel.name)?;
                            (member.group_id == group.id).then(|| {
                                json!({
                                    "index":channel.index,
                                    "name":channel.name,
                                    "inherit_color":member.inherit_color,
                                })
                            })
                        })
                        .collect::<Vec<_>>();
                    json!({
                        "id":group.id,
                        "name":group.name,
                        "expanded":group.expanded,
                        "color_rgb":group.color_rgb,
                        "members":members,
                    })
                })
                .collect(),
        )
    }

    pub(super) fn channel_groups(&self) -> Result<Value, ControlError> {
        self.require_resource()?;
        Ok(json!({"mode":"mosaic","groups":self.channel_groups_snapshot()}))
    }

    pub(super) fn set_channel_group(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        if let Some(state) = params.get("state") {
            let replacement: ProjectLayerGroups = serde_json::from_value(state.clone())
                .map_err(|error| invalid(format!("invalid channel-group state: {error}")))?;
            let changed = self.layer_groups != replacement;
            self.layer_groups = replacement;
            return Ok(json!({
                "mode":"mosaic",
                "result":{
                    "changed":changed,
                    "groups":self.channel_groups_snapshot(),
                },
            }));
        }
        let selectors = params
            .get("channels")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("set_channel_group requires channels"))?;
        let mut indices = selectors
            .iter()
            .map(|selector| self.channel_index(selector))
            .collect::<Result<Vec<_>, _>>()?;
        indices.sort_unstable();
        indices.dedup();
        if indices.is_empty() {
            return Err(invalid("no channels resolved"));
        }
        let requested_id = params.get("group_id").and_then(Value::as_u64);
        let requested_name = params
            .get("group")
            .or_else(|| params.get("name"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty());
        let color = color_from_params(params)?;
        let group_id =
            ensure_channel_group(&mut self.layer_groups, requested_id, requested_name, color);
        if params
            .get("replace_group_members")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            self.layer_groups
                .channel_members
                .retain(|_, member| member.group_id != group_id);
        }
        let inherit_color = params
            .get("inherit_color")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        for index in indices {
            self.layer_groups.channel_members.insert(
                self.channels[index].name.clone(),
                ProjectChannelGroupMember {
                    group_id,
                    inherit_color,
                },
            );
        }
        Ok(json!({
            "mode":"mosaic",
            "result":{
                "changed":true,
                "group_id":group_id,
                "groups":self.channel_groups_snapshot(),
            },
        }))
    }

    pub(super) fn native_layers_snapshot(&self) -> Result<Value, ControlError> {
        self.require_resource()?;
        Ok(json!({"mode":"mosaic","layers":self.native_layers.snapshots()}))
    }

    pub(super) fn native_layer_id<'a>(&self, params: &'a Value) -> Result<&'a str, ControlError> {
        params
            .get("layer_id")
            .or_else(|| params.get("id"))
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("layer_id is required"))
    }

    pub(super) fn native_layer_snapshot(&self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let layer_id = self.native_layer_id(params)?;
        let layer = self
            .native_layers
            .snapshots()
            .into_iter()
            .find(|layer| layer["layer_id"].as_str() == Some(layer_id))
            .ok_or_else(|| invalid(format!("native layer '{layer_id}' is not loaded")))?;
        Ok(json!({"mode":"mosaic","layer":layer}))
    }

    pub(super) fn set_native_layer_active(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        self.require_resource()?;
        let layer_id = self.native_layer_id(params)?.to_string();
        let changed = self.native_layers.set_active(&layer_id)?;
        self.sync_semantics_from_native_layers()?;
        let layer = self.native_layer_snapshot(&json!({"layer_id":layer_id}))?["layer"].clone();
        Ok(json!({"mode":"mosaic","result":{"changed":changed,"layer":layer}}))
    }

    pub(super) fn set_native_layer_visibility(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        self.require_resource()?;
        let layer_id = self.native_layer_id(params)?.to_string();
        let visible = params
            .get("visible")
            .and_then(Value::as_bool)
            .ok_or_else(|| invalid("visible is required"))?;
        let changed = self.native_layers.set_visibility(&layer_id, visible)?;
        self.sync_semantics_from_native_layers()?;
        let layer = self.native_layer_snapshot(&json!({"layer_id":layer_id}))?["layer"].clone();
        Ok(json!({"mode":"mosaic","result":{"changed":changed,"layer":layer}}))
    }

    pub(super) fn set_native_layer_order(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
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
        let changed = self.native_layers.set_order(stack, &layers)?;
        self.sync_semantics_from_native_layers()?;
        Ok(json!({
            "mode":"mosaic",
            "result":{"changed":changed,"layers":self.native_layers.snapshots()},
        }))
    }

    pub(super) fn sync_semantics_from_native_layers(&mut self) -> Result<(), ControlError> {
        let snapshots = self.native_layers.snapshots();
        let mut channel_order = Vec::new();
        for layer in &snapshots {
            let Some(layer_id) = layer["layer_id"].as_str() else {
                continue;
            };
            let visible = layer["visible"].as_bool().unwrap_or(false);
            if let Some(index) = layer_id
                .strip_prefix("channel:")
                .and_then(|value| value.parse::<usize>().ok())
            {
                let channel = self
                    .channels
                    .get_mut(index)
                    .ok_or_else(|| invalid(format!("native layer '{layer_id}' is not loaded")))?;
                channel.visible = visible;
                channel_order.push(index);
                if layer["active"].as_bool() == Some(true) {
                    self.active_channel = index;
                }
            } else if layer_id == "segmentation_geojson" {
                self.objects_visible = visible;
            } else if layer_id == "text_labels" {
                self.show_text_labels = visible;
            }
        }
        if channel_order.len() == self.channels.len() {
            self.channel_order = channel_order;
        }
        Ok(())
    }

    pub(super) fn sync_native_layers_from_semantics(&mut self) {
        for channel in &self.channels {
            let _ = self
                .native_layers
                .set_visibility(&format!("channel:{}", channel.index), channel.visible);
        }
        let _ = self
            .native_layers
            .set_visibility("segmentation_geojson", self.objects_visible);
        let _ = self
            .native_layers
            .set_visibility("text_labels", self.show_text_labels);
        let _ = self
            .native_layers
            .set_active(&format!("channel:{}", self.active_channel));
        let order = self
            .channel_order
            .iter()
            .map(|index| format!("channel:{index}"))
            .collect::<Vec<_>>();
        let _ = self.native_layers.set_order("channels", &order);
    }

    pub(super) fn channels_snapshot(&self) -> Result<Value, ControlError> {
        self.require_resource()?;
        Ok(json!({
            "mode":"mosaic",
            "channels":self.channels.iter().map(|channel| self.channel_json(channel)).collect::<Vec<_>>(),
        }))
    }

    pub(super) fn visible_channels_snapshot(&self) -> Result<Value, ControlError> {
        self.require_resource()?;
        Ok(json!({
            "mode":"mosaic",
            "channels":self.channels.iter().filter(|channel| channel.visible).map(|channel| json!({
                "index":channel.index,"name":channel.name
            })).collect::<Vec<_>>(),
        }))
    }

    pub(super) fn active_channel_snapshot(&self) -> Result<Value, ControlError> {
        self.require_resource()?;
        Ok(json!({
            "mode":"mosaic",
            "active_channel":self.channels.get(self.active_channel).map(|channel| self.channel_json(channel)),
        }))
    }

    pub(super) fn set_active_channel(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let index = self.channel_index_from_params(params)?;
        let changed = self.active_channel != index;
        self.active_channel = index;
        let _ = self.native_layers.set_active(&format!("channel:{index}"));
        Ok(json!({
            "mode":"mosaic",
            "result":{"changed":changed,"active_channel":self.channel_json(&self.channels[index])},
        }))
    }

    pub(super) fn set_visible_channels(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let selectors = params
            .get("channels")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("channels must be an array"))?;
        let indices = selectors
            .iter()
            .map(|selector| self.channel_index(selector))
            .collect::<Result<HashSet<_>, _>>()?;
        let mode = params.get("mode").and_then(Value::as_str).unwrap_or("only");
        if !matches!(mode, "only" | "show" | "hide" | "add" | "remove") {
            return Err(invalid(format!("unknown visibility mode '{mode}'")));
        }
        let before = self
            .channels
            .iter()
            .map(|channel| channel.visible)
            .collect::<Vec<_>>();
        for channel in &mut self.channels {
            channel.visible = match mode {
                "show" | "add" => channel.visible || indices.contains(&channel.index),
                "hide" | "remove" => channel.visible && !indices.contains(&channel.index),
                "only" => indices.contains(&channel.index),
                _ => unreachable!(),
            };
        }
        if let Some(first) = indices.iter().next() {
            self.active_channel = *first;
        }
        self.sync_native_layers_from_semantics();
        Ok(json!({
            "mode":"mosaic",
            "result":{
                "changed":before != self.channels.iter().map(|channel| channel.visible).collect::<Vec<_>>(),
                "mode":match mode { "show"=>"add", "hide"=>"remove", mode=>mode },
                "visible_channels":self.channels.iter().filter(|channel| channel.visible).map(|channel| json!({"index":channel.index,"name":channel.name})).collect::<Vec<_>>(),
            },
        }))
    }

    pub(super) fn get_channel_contrast(&self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let index = if params.as_object().is_some_and(|object| !object.is_empty()) {
            self.channel_index_from_params(params)?
        } else {
            self.active_channel
        };
        let channel = &self.channels[index];
        let (minimum, maximum) = channel.window.unwrap_or((0.0, self.abs_max()));
        Ok(json!({
            "mode":"mosaic",
            "contrast":{"index":index,"name":channel.name,"min":minimum,"max":maximum,"abs_max":self.abs_max()},
        }))
    }

    pub(super) fn set_channel_contrast(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let mut windows = if let Some(values) = params.get("windows") {
            values
                .as_array()
                .filter(|values| !values.is_empty())
                .ok_or_else(|| invalid("windows must be a non-empty array"))?
                .iter()
                .map(|value| {
                    let index = self.channel_index_from_params(value)?;
                    let minimum = value
                        .get("min")
                        .or_else(|| value.get("lo"))
                        .and_then(Value::as_f64)
                        .ok_or_else(|| invalid("each contrast window requires min"))?
                        as f32;
                    let maximum = value
                        .get("max")
                        .or_else(|| value.get("hi"))
                        .and_then(Value::as_f64)
                        .ok_or_else(|| invalid("each contrast window requires max"))?
                        as f32;
                    if !minimum.is_finite() || !maximum.is_finite() || maximum <= minimum {
                        return Err(invalid("contrast max must be greater than min"));
                    }
                    Ok((index, minimum, maximum))
                })
                .collect::<Result<Vec<_>, ControlError>>()?
        } else {
            let minimum = params
                .get("min")
                .or_else(|| params.get("lo"))
                .and_then(Value::as_f64)
                .ok_or_else(|| invalid("min is required"))? as f32;
            let maximum = params
                .get("max")
                .or_else(|| params.get("hi"))
                .and_then(Value::as_f64)
                .ok_or_else(|| invalid("max is required"))? as f32;
            if !minimum.is_finite() || !maximum.is_finite() || maximum <= minimum {
                return Err(invalid("contrast max must be greater than min"));
            }
            let indices = if let Some(selectors) = params.get("channels") {
                selectors
                    .as_array()
                    .filter(|selectors| !selectors.is_empty())
                    .ok_or_else(|| invalid("channels must be a non-empty array"))?
                    .iter()
                    .map(|selector| self.channel_index(selector))
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                vec![self.channel_index_from_params(params)?]
            };
            indices
                .into_iter()
                .map(|index| (index, minimum, maximum))
                .collect()
        };
        windows.sort_unstable_by_key(|(index, _, _)| *index);
        if windows.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(invalid("contrast channels must not contain duplicates"));
        }
        let mut changed = false;
        for &(index, minimum, maximum) in &windows {
            changed |= self.channels[index].window != Some((minimum, maximum));
            self.channels[index].window = Some((minimum, maximum));
        }
        let window_results = windows
            .iter()
            .map(|&(index, minimum, maximum)| {
                json!({"index":index,"name":self.channels[index].name,"min":minimum,"max":maximum})
            })
            .collect::<Vec<_>>();
        let mut contrast = json!({
            "changed":changed,
            "channels":window_results,
            "windows":window_results,
            "count":windows.len(),
            "abs_max":self.abs_max(),
        });
        let common_window = windows
            .first()
            .map(|(_, minimum, maximum)| (*minimum, *maximum));
        if common_window.is_some_and(|(minimum, maximum)| {
            windows.iter().all(|(_, other_minimum, other_maximum)| {
                *other_minimum == minimum && *other_maximum == maximum
            })
        }) {
            let (minimum, maximum) = common_window.expect("contrast windows are non-empty");
            contrast["min"] = json!(minimum);
            contrast["max"] = json!(maximum);
        }
        if windows.len() == 1 {
            let (index, minimum, maximum) = windows[0];
            contrast["index"] = json!(index);
            contrast["name"] = json!(self.channels[index].name);
            contrast["min"] = json!(minimum);
            contrast["max"] = json!(maximum);
        }
        Ok(json!({"mode":"mosaic","contrast":contrast}))
    }

    pub(super) fn set_channel_color(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let index = self.channel_index_from_params(params)?;
        let values = params
            .get("color_rgb")
            .or_else(|| params.get("color"))
            .and_then(Value::as_array)
            .filter(|values| values.len() == 3)
            .ok_or_else(|| invalid("color_rgb must contain three integers"))?;
        let color = [
            json_u8(&values[0])?,
            json_u8(&values[1])?,
            json_u8(&values[2])?,
        ];
        let changed = self.channels[index].color_rgb != color;
        self.channels[index].color_rgb = color;
        Ok(
            json!({"mode":"mosaic","result":{"changed":changed,"channel":self.channel_json(&self.channels[index])}}),
        )
    }

    pub(super) fn set_channel_note(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let index = self.channel_index_from_params(params)?;
        let note = params
            .get("note")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("set_channel_note requires note"))?
            .to_string();
        let changed = self.channels[index].note != note;
        self.channels[index].note = note;
        Ok(json!({"changed":changed,"channel":self.channel_json(&self.channels[index])}))
    }

    pub(super) fn set_channel_order(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let selectors = params
            .get("order")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("order must be an array"))?;
        let mut order = selectors
            .iter()
            .map(|selector| self.channel_index(selector))
            .collect::<Result<Vec<_>, _>>()?;
        let mut seen = HashSet::new();
        order.retain(|index| seen.insert(*index));
        if order.len() != self.channels.len() {
            return Err(invalid(
                "channel order must contain every channel exactly once",
            ));
        }
        let changed = self.channel_order != order;
        self.channel_order = order;
        self.channel_sort = "manual".to_string();
        self.sync_native_layers_from_semantics();
        Ok(json!({
            "changed":changed,
            "order":self.channel_order.iter().map(|index| json!({"index":index,"name":self.channels[*index].name})).collect::<Vec<_>>(),
        }))
    }

    pub(super) fn set_camera(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let before = (self.camera_center, self.camera_zoom);
        if let Some(center) = params.get("center_world_lvl0").and_then(json_pair) {
            self.camera_center = center;
        }
        if let Some(x) = params.get("center_x").and_then(Value::as_f64) {
            if !x.is_finite() {
                return Err(invalid("center_x must be finite"));
            }
            self.camera_center[0] = x as f32;
        }
        if let Some(y) = params.get("center_y").and_then(Value::as_f64) {
            if !y.is_finite() {
                return Err(invalid("center_y must be finite"));
            }
            self.camera_center[1] = y as f32;
        }
        if let Some(zoom) = params
            .get("zoom_screen_per_lvl0_px")
            .or_else(|| params.get("zoom"))
            .and_then(Value::as_f64)
        {
            if !zoom.is_finite() || zoom <= 0.0 {
                return Err(invalid("zoom must be finite and greater than zero"));
            }
            self.camera_zoom = (zoom as f32).clamp(0.000_01, 5000.0);
        }
        Ok(json!({
            "mode":"mosaic",
            "camera":self.camera_snapshot(),
            "changed":before != (self.camera_center,self.camera_zoom),
        }))
    }

    pub(super) fn zoom_camera(
        &mut self,
        params: &Value,
        zoom_in: bool,
    ) -> Result<Value, ControlError> {
        let factor = params.get("factor").and_then(Value::as_f64).unwrap_or(1.5) as f32;
        if !factor.is_finite() || factor <= 0.0 {
            return Err(invalid("zoom factor must be finite and > 0"));
        }
        let factor = if zoom_in { factor } else { 1.0 / factor };
        self.set_camera(&json!({"zoom":self.camera_zoom * factor}))
    }

    pub(super) fn fit_camera(&mut self) -> Result<Value, ControlError> {
        self.require_resource()?;
        self.fit_bounds(self.bounds);
        Ok(json!({"mode":"mosaic","camera":self.camera_snapshot()}))
    }

    pub(super) fn panels_snapshot(&self) -> Result<Value, ControlError> {
        self.require_resource()?;
        Ok(
            json!({"mode":"mosaic","panels":{"left":self.show_left_panel,"right":self.show_right_panel}}),
        )
    }

    pub(super) fn set_panels(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let before = (self.show_left_panel, self.show_right_panel);
        if let Some(value) = params.get("left").and_then(Value::as_bool) {
            self.show_left_panel = value;
        }
        if let Some(value) = params.get("right").and_then(Value::as_bool) {
            self.show_right_panel = value;
        }
        Ok(json!({
            "mode":"mosaic",
            "result":{"changed":before != (self.show_left_panel,self.show_right_panel),"panels":{"left":self.show_left_panel,"right":self.show_right_panel}},
        }))
    }

    pub(super) fn smooth_pixels_snapshot(&self) -> Result<Value, ControlError> {
        self.require_resource()?;
        Ok(json!({"mode":"mosaic","smooth_pixels":{"smooth":self.smooth_pixels}}))
    }

    pub(super) fn set_smooth_pixels(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let smooth = params
            .get("smooth")
            .and_then(Value::as_bool)
            .ok_or_else(|| invalid("set_smooth_pixels requires smooth"))?;
        let changed = self.smooth_pixels != smooth;
        self.smooth_pixels = smooth;
        Ok(json!({"mode":"mosaic","result":{"changed":changed,"smooth_pixels":{"smooth":smooth}}}))
    }

    pub(super) fn rendering_snapshot(&self) -> Result<Value, ControlError> {
        self.require_resource()?;
        Ok(json!({
            "mode":"mosaic",
            "gpu_available":true,
            "renderer":"opengl",
            "compositing":"additive",
            "smooth_pixels":self.smooth_pixels,
            "show_tile_debug":self.show_tile_debug,
        }))
    }

    pub(super) fn set_rendering(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let before = (self.smooth_pixels, self.show_tile_debug);
        let mut provided = false;
        if let Some(value) = params.get("smooth_pixels") {
            self.smooth_pixels = value
                .as_bool()
                .ok_or_else(|| invalid("smooth_pixels must be a boolean"))?;
            provided = true;
        }
        if let Some(value) = params.get("show_tile_debug") {
            self.show_tile_debug = value
                .as_bool()
                .ok_or_else(|| invalid("show_tile_debug must be a boolean"))?;
            provided = true;
        }
        if !provided {
            return Err(invalid(
                "mosaic.rendering.set requires smooth_pixels and/or show_tile_debug",
            ));
        }
        Ok(json!({
            "changed":before != (self.smooth_pixels,self.show_tile_debug),
            "rendering":{
                "smooth_pixels":self.smooth_pixels,
                "show_tile_debug":self.show_tile_debug,
            },
        }))
    }

    pub(super) fn object_visibility_snapshot(&self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let target = params
            .get("target")
            .and_then(Value::as_str)
            .unwrap_or("objects");
        Ok(json!({
            "mode":"mosaic",
            "overlay":{"target":target,"segmentation_objects":self.objects_visible,"object_count":self.object_resources.values().map(|resource| resource.features.len()).sum::<usize>()},
        }))
    }

    pub(super) fn set_object_visibility(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let visible = params
            .get("visible")
            .and_then(Value::as_bool)
            .ok_or_else(|| invalid("set_object_overlay_visibility requires visible"))?;
        let changed = self.objects_visible != visible;
        self.objects_visible = visible;
        let _ = self
            .native_layers
            .set_visibility("segmentation_geojson", visible);
        let mut response = self.object_visibility_snapshot(params)?;
        response
            .as_object_mut()
            .expect("object visibility response is an object")
            .insert("changed".to_string(), Value::Bool(changed));
        Ok(response)
    }

    pub(super) fn fast_object_rendering_snapshot(&self) -> Result<Value, ControlError> {
        self.require_resource()?;
        Ok(json!({"enabled":self.fast_object_rendering}))
    }

    pub(super) fn set_fast_object_rendering(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        self.require_resource()?;
        let enabled = params
            .get("enabled")
            .and_then(Value::as_bool)
            .ok_or_else(|| invalid("enabled is required"))?;
        let changed = self.fast_object_rendering != enabled;
        self.fast_object_rendering = enabled;
        Ok(json!({"enabled":enabled,"changed":changed}))
    }

    pub(super) fn channel_index_from_params(&self, params: &Value) -> Result<usize, ControlError> {
        let selector = params
            .get("index")
            .or_else(|| params.get("channel_index"))
            .or_else(|| params.get("name"))
            .or_else(|| params.get("channel"))
            .or_else(|| params.get("marker"))
            .ok_or_else(|| invalid("provide index, name, channel, or marker"))?;
        self.channel_index(selector)
    }

    pub(super) fn channel_index(&self, selector: &Value) -> Result<usize, ControlError> {
        if let Some(index) = selector.as_u64() {
            return usize::try_from(index)
                .ok()
                .filter(|index| *index < self.channels.len())
                .ok_or_else(|| invalid(format!("channel index {index} is out of range")));
        }
        let name = selector
            .as_str()
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| invalid(format!("invalid channel selector: {selector}")))?;
        let needle = normalize_name(name);
        if let Some(index) = self
            .channels
            .iter()
            .position(|channel| normalize_name(&channel.name) == needle)
        {
            return Ok(index);
        }
        let matches = self
            .channels
            .iter()
            .enumerate()
            .filter(|(_, channel)| normalize_name(&channel.name).contains(&needle))
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [index] => Ok(*index),
            [] => Err(ControlError::new(
                ControlErrorKind::ResourceNotFound,
                format!("no channel matches '{name}'"),
            )),
            _ => Err(invalid(format!("channel selector '{name}' is ambiguous"))),
        }
    }

    pub(super) fn channel_json(&self, channel: &MosaicChannelModel) -> Value {
        json!({
            "index":channel.index,
            "name":channel.name,
            "visible":channel.visible,
            "active":channel.index == self.active_channel,
            "color_rgb":channel.color_rgb,
            "window":channel.window.map(|(minimum,maximum)| [minimum,maximum]),
            "note":channel.note,
        })
    }

    pub(super) fn abs_max(&self) -> f32 {
        self.resource
            .as_ref()
            .into_iter()
            .flat_map(|resource| resource.items.iter())
            .map(|item| item.document.descriptor.abs_max)
            .fold(1.0_f32, f32::max)
    }
}
