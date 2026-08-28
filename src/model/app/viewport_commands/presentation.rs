use super::*;

impl AppModel {
    pub(in crate::model::app) fn get_viewport_channels(
        &self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let workspace = &self.dataset()?.workspace;
        let slot = workspace.get(&id).ok_or_else(|| not_found(&id))?;
        Ok(viewport_response(
            workspace,
            &id,
            Value::Array(full_channels_json(&slot.state)),
            vec![id.clone()],
            false,
        ))
    }

    pub(in crate::model::app) fn set_visible_channels(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let selectors = params
            .get("channels")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("channels must be an array"))?
            .clone();
        let workspace = &mut self.dataset_mut()?.workspace;
        let active_before = workspace.active().state.clone();
        let target = workspace.get_mut(&id).ok_or_else(|| not_found(&id))?;
        let indices = resolve_channels(&target.state.channels, &selectors)?;
        let mode = params.get("mode").and_then(Value::as_str).unwrap_or("only");
        if !matches!(mode, "only" | "show" | "hide" | "add" | "remove") {
            return Err(invalid(format!("unknown visibility mode '{mode}'")));
        }
        for channel in &mut target.state.channels {
            channel.visible = match mode {
                "show" | "add" => channel.visible || indices.contains(&channel.index),
                "hide" | "remove" => channel.visible && !indices.contains(&channel.index),
                "only" => indices.contains(&channel.index),
                _ => unreachable!("visibility mode validated above"),
            };
        }
        if let Some(first) = selectors.first() {
            target.state.active_channel = resolve_channel(&target.state.channels, first)?;
            target
                .state
                .native_layers
                .set_active(&format!("channel:{}", target.state.active_channel))?;
        }
        let visible_channels = visible_channels_json(&target.state);
        let _ = workspace.bump_presentation_revision(&id);
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        Ok(viewport_response(
            workspace,
            &id,
            json!({"changed": true, "mode": canonical_visibility_mode(mode), "visible_channels": visible_channels}),
            vec![id.clone()],
            active_changed,
        ))
    }

    pub(in crate::model::app) fn set_active_channel(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let selector = channel_selector_from_params(params)?.clone();
        let workspace = &mut self.dataset_mut()?.workspace;
        let active_before = workspace.active().state.clone();
        let target = workspace.get_mut(&id).ok_or_else(|| not_found(&id))?;
        target.state.active_channel = resolve_channel(&target.state.channels, &selector)?;
        target
            .state
            .native_layers
            .set_active(&format!("channel:{}", target.state.active_channel))?;
        let active_channel = target.state.active_channel;
        let active_channel = active_channel_json(&target.state.channels[active_channel]);
        let _ = workspace.bump_presentation_revision(&id);
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        Ok(viewport_response(
            workspace,
            &id,
            json!({"changed": true, "active_channel": active_channel}),
            vec![id.clone()],
            active_changed,
        ))
    }

    pub(in crate::model::app) fn set_channel_color(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let selector = channel_selector_from_params(params)?.clone();
        let color = params
            .get("color_rgb")
            .or_else(|| params.get("color"))
            .and_then(Value::as_array)
            .filter(|v| v.len() == 3)
            .ok_or_else(|| invalid("color_rgb must contain three integers"))?;
        let rgb = [to_u8(&color[0])?, to_u8(&color[1])?, to_u8(&color[2])?];
        let workspace = &mut self.dataset_mut()?.workspace;
        let active_before = workspace.active().state.clone();
        let target = workspace.get_mut(&id).ok_or_else(|| not_found(&id))?;
        let index = resolve_channel(&target.state.channels, &selector)?;
        let channel_name = target.state.channels[index].name.clone();
        let color_changed = target.state.channels[index].color_rgb != rgb;
        target.state.channels[index].color_rgb = rgb;
        let inheritance_changed = target
            .state
            .channel_groups
            .channel_members
            .get_mut(&channel_name)
            .is_some_and(|member| {
                let changed = member.inherit_color;
                member.inherit_color = false;
                changed
            });
        let changed = color_changed || inheritance_changed;
        let channel = full_channel_json(
            &target.state.channels[index],
            index == target.state.active_channel,
        );
        let _ = workspace.bump_presentation_revision(&id);
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        Ok(viewport_response(
            workspace,
            &id,
            json!({"changed": changed, "channel": channel}),
            vec![id.clone()],
            active_changed,
        ))
    }

    pub(in crate::model::app) fn set_channel_contrast(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let selectors = params
            .get("channels")
            .map(|value| {
                value
                    .as_array()
                    .filter(|values| !values.is_empty())
                    .cloned()
                    .ok_or_else(|| invalid("channels must be a non-empty array"))
            })
            .transpose()?;
        let selector = selectors
            .is_none()
            .then(|| channel_selector_from_params(params).cloned())
            .transpose()?;
        let min = params
            .get("min")
            .or_else(|| params.get("lo"))
            .and_then(Value::as_f64)
            .ok_or_else(|| invalid("min is required"))? as f32;
        let max = params
            .get("max")
            .or_else(|| params.get("hi"))
            .and_then(Value::as_f64)
            .ok_or_else(|| invalid("max is required"))? as f32;
        if !min.is_finite() || !max.is_finite() || max <= min {
            return Err(invalid("contrast max must be greater than min"));
        }
        let dataset = self.dataset_mut()?;
        let abs_max = dataset.descriptor.abs_max.max(1.0);
        let workspace = &mut dataset.workspace;
        let active_before = workspace.active().state.clone();
        let target = workspace.get_mut(&id).ok_or_else(|| not_found(&id))?;
        let mut indices = if let Some(selectors) = selectors {
            selectors
                .iter()
                .map(|selector| resolve_channel(&target.state.channels, selector))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            vec![resolve_channel(
                &target.state.channels,
                selector.as_ref().expect("single selector was validated"),
            )?]
        };
        indices.sort_unstable();
        indices.dedup();
        let mut changed = false;
        for &index in &indices {
            let channel = &mut target.state.channels[index];
            changed |= channel.window != Some((min, max));
            channel.window = Some((min, max));
            channel.contrast_manual = true;
        }
        let channels = indices
            .iter()
            .map(|&index| json!({"index":index,"name":target.state.channels[index].name}))
            .collect::<Vec<_>>();
        let mut result = json!({
            "changed": changed,
            "channels": channels,
            "count": indices.len(),
            "min": min,
            "max": max,
            "abs_max": abs_max,
        });
        if indices.len() == 1 {
            let index = indices[0];
            result["index"] = json!(index);
            result["name"] = json!(target.state.channels[index].name);
        }
        let _ = workspace.bump_presentation_revision(&id);
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        Ok(viewport_response(
            workspace,
            &id,
            result,
            vec![id.clone()],
            active_changed,
        ))
    }

    pub(in crate::model::app) fn set_channel_order(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let requested_sort = params
            .get("sort")
            .map(|value| {
                value
                    .as_str()
                    .and_then(canonical_channel_sort)
                    .ok_or_else(|| {
                        invalid(format!(
                            "unknown channel sort mode '{}'",
                            value.as_str().unwrap_or_default()
                        ))
                    })
            })
            .transpose()?;
        let selectors = params.get("channels").and_then(Value::as_array);
        if requested_sort.is_none() && selectors.is_none() {
            return Err(invalid("set_channel_order requires channels or sort"));
        }
        let workspace = &mut self.dataset_mut()?.workspace;
        let active_before = workspace.active().state.clone();
        let target = workspace.get_mut(&id).ok_or_else(|| not_found(&id))?;
        let response = if let Some(sort) = requested_sort {
            target.state.channel_sort = sort.to_string();
            json!({
                "changed": true,
                "sort": sort,
                "order": channel_order_json(&target.state),
            })
        } else {
            let selectors = selectors.expect("validated above");
            let indices = resolve_channel_list_ordered(&target.state.channels, selectors)?;
            let mode = params
                .get("mode")
                .and_then(Value::as_str)
                .unwrap_or("listed_first");
            match mode {
                "listed_first" => {
                    let pinned = indices.iter().copied().collect::<HashSet<_>>();
                    let mut next = indices;
                    next.extend(
                        target
                            .state
                            .channel_order
                            .iter()
                            .copied()
                            .filter(|index| !pinned.contains(index)),
                    );
                    for index in 0..target.state.channels.len() {
                        if !next.contains(&index) {
                            next.push(index);
                        }
                    }
                    target.state.channel_order = next;
                    target.state.channel_sort = "manual".to_string();
                }
                "exact" => {
                    if indices.len() != target.state.channels.len() {
                        return Err(invalid(
                            "exact channel order must include every channel exactly once",
                        ));
                    }
                    target.state.channel_order = indices;
                    target.state.channel_sort = "manual".to_string();
                }
                other => return Err(invalid(format!("unknown channel order mode '{other}'"))),
            }
            json!({
                "changed": true,
                "mode": mode,
                "sort": target.state.channel_sort,
                "order": channel_order_json(&target.state),
            })
        };
        let _ = workspace.bump_presentation_revision(&id);
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        Ok(viewport_response(
            workspace,
            &id,
            response,
            vec![id.clone()],
            active_changed,
        ))
    }

    pub(in crate::model::app) fn channel_groups(
        &self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let workspace = &self.dataset()?.workspace;
        let target = workspace.get(&id).ok_or_else(|| not_found(&id))?;
        Ok(viewport_response(
            workspace,
            &id,
            channel_groups_json(&target.state),
            vec![id.clone()],
            false,
        ))
    }

    pub(in crate::model::app) fn set_channel_group(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        if params.get("replace_all").and_then(Value::as_bool) == Some(true) {
            let groups = params
                .get("groups")
                .and_then(Value::as_array)
                .ok_or_else(|| invalid("replace_all requires groups"))?;
            let workspace = &mut self.dataset_mut()?.workspace;
            let active_before = workspace.active().state.clone();
            let target = workspace.get_mut(&id).ok_or_else(|| not_found(&id))?;
            let replacement = parse_channel_groups_snapshot(groups, &target.state.channels)?;
            let changed = replacement != target.state.channel_groups;
            target.state.channel_groups = replacement;
            let groups = channel_groups_json(&target.state);
            let _ = workspace.bump_presentation_revision(&id);
            let active_changed = presentation_changed(&active_before, &workspace.active().state);
            return Ok(viewport_response(
                workspace,
                &id,
                json!({"changed": changed, "group_id": Value::Null, "groups": groups}),
                vec![id.clone()],
                active_changed,
            ));
        }
        let selectors = params
            .get("channels")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("set_channel_group requires channels"))?;
        let workspace = &mut self.dataset_mut()?.workspace;
        let active_before = workspace.active().state.clone();
        let target = workspace.get_mut(&id).ok_or_else(|| not_found(&id))?;
        let indices = resolve_channel_list_ordered(&target.state.channels, selectors)?;
        if indices.is_empty() {
            return Err(invalid("no channels resolved"));
        }
        let requested_group_id = params.get("group_id").and_then(Value::as_u64);
        let requested_name = params
            .get("group")
            .or_else(|| params.get("name"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty());
        let color = optional_rgb(params, "color_rgb")?;
        let group_id = ensure_model_channel_group(
            &mut target.state.channel_groups,
            requested_group_id,
            requested_name,
            color,
        );
        if params
            .get("replace_group_members")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            target
                .state
                .channel_groups
                .channel_members
                .retain(|_, member| member.group_id != group_id);
        }
        let inherit_color = params
            .get("inherit_color")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        for index in indices {
            let name = target.state.channels[index].name.clone();
            target.state.channel_groups.channel_members.insert(
                name,
                ProjectChannelGroupMember {
                    group_id,
                    inherit_color,
                },
            );
        }
        let groups = channel_groups_json(&target.state);
        let _ = workspace.bump_presentation_revision(&id);
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        Ok(viewport_response(
            workspace,
            &id,
            json!({"changed": true, "group_id": group_id, "groups": groups}),
            vec![id.clone()],
            active_changed,
        ))
    }

    pub(in crate::model::app) fn get_object_style(
        &self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let object_target = self.resolve_object_target(params)?;
        let resource = self
            .object_resource_for_target(object_target, "viewer.objects.style.get")
            .ok();
        let id = Self::viewport_id(params)?;
        let workspace = &self.dataset()?.workspace;
        let target = workspace.get(&id).ok_or_else(|| not_found(&id))?;
        let objects = target
            .state
            .object_presentation(object_target)
            .ok_or_else(|| object_target_not_found(object_target))?;
        Ok(viewport_response(
            workspace,
            &id,
            object_style_json(objects, resource),
            vec![id.clone()],
            false,
        ))
    }

    pub(in crate::model::app) fn set_object_style(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let object_target = self.resolve_object_target(params)?;
        let resource = self
            .object_resource_for_target(object_target, "viewer.objects.style.set")
            .ok()
            .cloned();
        let id = Self::viewport_id(params)?;
        let workspace = &mut self.dataset_mut()?.workspace;
        let active_before = workspace.active().state.clone();
        let target = workspace.get_mut(&id).ok_or_else(|| not_found(&id))?;
        let objects = target
            .state
            .object_presentation_mut(object_target)
            .ok_or_else(|| object_target_not_found(object_target))?;
        let changed = apply_object_style_patch(objects, params)?;
        let style = object_style_json(objects, resource.as_ref());
        let native_presentation = objects.clone();
        if object_target != ObjectTarget::Primary {
            target
                .state
                .native_layers
                .set_presentation(&object_target.layer_id(), &native_presentation)?;
        }
        let _ = workspace.bump_presentation_revision(&id);
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        Ok(viewport_response(
            workspace,
            &id,
            json!({"changed": changed, "style": style}),
            vec![id.clone()],
            active_changed,
        ))
    }

    pub(in crate::model::app) fn set_object_legend(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let object_target = self.resolve_object_target(params)?;
        let resource = self
            .object_resource_for_target(object_target, "viewer.objects.legend.set")
            .ok()
            .cloned();
        let id = Self::viewport_id(params)?;
        let workspace = &mut self.dataset_mut()?.workspace;
        let active_before = workspace.active().state.clone();
        let target = workspace.get_mut(&id).ok_or_else(|| not_found(&id))?;
        let objects = target
            .state
            .object_presentation_mut(object_target)
            .ok_or_else(|| object_target_not_found(object_target))?;
        apply_object_legend_patch(objects, params)?;
        let style = object_style_json(objects, resource.as_ref());
        let native_presentation = objects.clone();
        if object_target != ObjectTarget::Primary {
            target
                .state
                .native_layers
                .set_presentation(&object_target.layer_id(), &native_presentation)?;
        }
        let _ = workspace.bump_presentation_revision(&id);
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        Ok(viewport_response(
            workspace,
            &id,
            json!({"changed": true, "style": style}),
            vec![id.clone()],
            active_changed,
        ))
    }

    pub(in crate::model::app) fn set_channel_presentation(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let search = params
            .get("search")
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .ok_or_else(|| invalid("search must be a string"))
            })
            .transpose()?;
        let sort = params
            .get("sort")
            .map(|value| {
                value
                    .as_str()
                    .and_then(canonical_channel_sort)
                    .map(str::to_string)
                    .ok_or_else(|| invalid("unknown channel sort mode"))
            })
            .transpose()?;
        let workspace = &mut self.dataset_mut()?.workspace;
        let active_before = workspace.active().state.clone();
        let target = workspace.get_mut(&id).ok_or_else(|| not_found(&id))?;
        if let Some(search) = search {
            target.state.channel_search = search;
        }
        if let Some(sort) = sort {
            target.state.channel_sort = sort;
        }
        let result = channel_presentation_json(&target.state);
        let _ = workspace.bump_presentation_revision(&id);
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        Ok(viewport_response(
            workspace,
            &id,
            result,
            vec![id.clone()],
            active_changed,
        ))
    }

    pub(in crate::model::app) fn get_rendering(
        &self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let workspace = &self.dataset()?.workspace;
        let slot = workspace.get(&id).ok_or_else(|| not_found(&id))?;
        Ok(viewport_response(
            workspace,
            &id,
            rendering_json(&slot.state),
            vec![id.clone()],
            false,
        ))
    }

    pub(in crate::model::app) fn set_rendering(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let workspace = &mut self.dataset_mut()?.workspace;
        let active_before = workspace.active().state.clone();
        let target = workspace.get_mut(&id).ok_or_else(|| not_found(&id))?;
        let before = rendering_json(&target.state);
        let mut saw_field = false;
        let mut smooth_pixels = target.state.smooth_pixels;
        let mut show_scale_bar = target.state.show_scale_bar;
        let mut show_hud = target.state.show_hud;
        let mut show_tile_debug = target.state.show_tile_debug;
        set_rendering_bool(
            params,
            &["smooth_pixels", "smooth"],
            "smooth_pixels",
            &mut smooth_pixels,
            &mut saw_field,
        )?;
        set_rendering_bool(
            params,
            &["show_scale_bar"],
            "show_scale_bar",
            &mut show_scale_bar,
            &mut saw_field,
        )?;
        set_rendering_bool(
            params,
            &["show_hud"],
            "show_hud",
            &mut show_hud,
            &mut saw_field,
        )?;
        set_rendering_bool(
            params,
            &["show_tile_debug"],
            "show_tile_debug",
            &mut show_tile_debug,
            &mut saw_field,
        )?;
        if !saw_field {
            return Err(invalid(
                "provide smooth_pixels, show_scale_bar, show_hud, and/or show_tile_debug",
            ));
        }
        target.state.smooth_pixels = smooth_pixels;
        target.state.show_scale_bar = show_scale_bar;
        target.state.show_hud = show_hud;
        target.state.show_tile_debug = show_tile_debug;
        let result = rendering_json(&target.state);
        let changed = before != result;
        let _ = workspace.bump_presentation_revision(&id);
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        Ok(viewport_response(
            workspace,
            &id,
            json!({"changed": changed, "rendering": result}),
            vec![id.clone()],
            active_changed,
        ))
    }
}
