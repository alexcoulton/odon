use super::*;

impl AppModel {
    pub(in crate::model::app) fn workspace_snapshot(&self) -> Result<Value, ControlError> {
        let dataset = self.dataset()?;
        let workspace = &dataset.workspace;
        let active = workspace.active_id();
        Ok(json!({
            "revision": workspace.revision(),
            "layout": workspace.layout().as_str(),
            "ratio": workspace.split_ratio(),
            "active_viewport_id": active.as_str(),
            "max_viewports": crate::viewports::MAX_VIEWPORTS,
            "shared_resources": dataset.shared_resources,
            "object_resource": dataset.object_resource.as_ref().map_or_else(
                || self.object_resource_state(),
                |resource| resource.descriptor_json(self.installed_object_resource_generation),
            ),
            "labels": self.labels_snapshot()?,
            "masks": dataset.masks.projection_json(),
            "object_selection": dataset.object_selection.projection_json(),
            "panels": {
                "left": dataset.show_left_panel,
                "right": dataset.show_right_panel,
            },
            "ui":{"right_tab":dataset.right_tab},
            "channel_metadata": workspace.active().state.channels.iter().map(|channel| json!({
                "index": channel.index,
                "name": channel.name,
                "note": channel.note,
            })).collect::<Vec<_>>(),
            "channel_transforms": workspace.active().state.channels.iter().enumerate().map(
                |(index, channel)| channel_transform_json(channel, index)
            ).collect::<Vec<_>>(),
            "channel_presentation": channel_presentation_json(&workspace.active().state),
            "performance": dataset.performance,
            "links": links_json(workspace.links()),
            "viewports": workspace.viewports().iter().map(|slot| viewport_json(slot, slot.id == *active)).collect::<Vec<_>>(),
        }))
    }

    pub(in crate::model::app) fn layout_snapshot(&self) -> Result<Value, ControlError> {
        let workspace = &self.dataset()?.workspace;
        Ok(json!({
            "revision": workspace.revision(),
            "layout": workspace.layout().as_str(),
            "ratio": workspace.split_ratio(),
            "viewport_ids": workspace.viewports().iter().map(|v| v.id.as_str()).collect::<Vec<_>>(),
        }))
    }

    pub(in crate::model::app) fn set_layout(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let value = params
            .get("layout")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("layout is required"))?;
        let layout = ViewportLayout::parse(value)
            .ok_or_else(|| invalid("layout must be 'single', 'horizontal', or 'vertical'"))?;
        let ratio = params
            .get("ratio")
            .map(|value| {
                value
                    .as_f64()
                    .ok_or_else(|| invalid("ratio must be a number"))
            })
            .transpose()?;
        if let Some(ratio) = ratio {
            if !ratio.is_finite() || !(0.1..=0.9).contains(&ratio) {
                return Err(invalid(
                    "split ratio must be finite and between 0.1 and 0.9",
                ));
            }
        }
        if let Some(requested) = params
            .get("viewports")
            .or_else(|| params.get("viewport_ids"))
        {
            validate_viewport_order(&self.dataset()?.workspace, requested)?;
        }
        let result = {
            let dataset = self.dataset_mut()?;
            let mut changed = dataset
                .workspace
                .set_layout(layout)
                .map_err(|e| invalid(e.to_string()))?;
            if let Some(ratio) = ratio {
                changed |= dataset
                    .workspace
                    .set_split_ratio(ratio as f32)
                    .map_err(|e| invalid(e.to_string()))?;
            }
            update_logical_geometry(dataset);
            json!({"changed": changed, "layout": layout.as_str(), "ratio": dataset.workspace.split_ratio()})
        };
        self.measured_viewports.clear();
        Ok(result)
    }

    pub(in crate::model::app) fn swap_viewports(&mut self) -> Result<Value, ControlError> {
        {
            let dataset = self.dataset_mut()?;
            if !dataset.workspace.swap_order() {
                return Err(invalid("swapping requires exactly two viewports"));
            }
            update_logical_geometry(dataset);
        }
        self.measured_viewports.clear();
        Ok(json!({"changed": true, "workspace": self.workspace_snapshot()?}))
    }

    pub(in crate::model::app) fn viewport_snapshot_for(
        &self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let workspace = &self.dataset()?.workspace;
        let slot = workspace.get(&id).ok_or_else(|| not_found(&id))?;
        Ok(viewport_json(slot, id == *workspace.active_id()))
    }

    pub(in crate::model::app) fn create_viewport(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let ratio = params
            .get("ratio")
            .map(|value| {
                value
                    .as_f64()
                    .ok_or_else(|| invalid("ratio must be a number"))
            })
            .transpose()?;
        if let Some(ratio) = ratio {
            if !ratio.is_finite() || !(0.1..=0.9).contains(&ratio) {
                return Err(invalid(
                    "split ratio must be finite and between 0.1 and 0.9",
                ));
            }
        }
        let id = {
            let dataset = self.dataset_mut()?;
            let workspace = &mut dataset.workspace;
            let source = params
                .get("source_viewport_id")
                .or_else(|| params.get("viewport_id"))
                .and_then(Value::as_str)
                .map(ViewportId::new)
                .transpose()
                .map_err(|e| invalid(e.to_string()))?
                .unwrap_or_else(|| workspace.active_id().clone());
            let layout = match params.get("layout").and_then(Value::as_str) {
                Some(value) => match ViewportLayout::parse(value) {
                    Some(layout @ (ViewportLayout::Horizontal | ViewportLayout::Vertical)) => {
                        layout
                    }
                    Some(ViewportLayout::Single) => {
                        return Err(invalid(
                            "creating a second viewport requires a split layout",
                        ));
                    }
                    None => return Err(invalid("layout must be 'horizontal' or 'vertical'")),
                },
                None => ViewportLayout::Horizontal,
            };
            let title = params
                .get("title")
                .and_then(Value::as_str)
                .map(str::to_string);
            let activate = params
                .get("activate")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            let previous = workspace.active_id().clone();
            let id = workspace
                .clone_viewport(&source, title, layout)
                .map_err(|e| invalid(e.to_string()))?;
            if let Some(ratio) = ratio {
                workspace
                    .set_split_ratio(ratio as f32)
                    .map_err(|e| invalid(e.to_string()))?;
            }
            if !activate {
                workspace
                    .set_active(&previous)
                    .map_err(|e| invalid(e.to_string()))?;
            }
            update_logical_geometry(dataset);
            id
        };
        self.measured_viewports.clear();
        Ok(
            json!({"created": true, "viewport_id": id.as_str(), "workspace": self.workspace_snapshot()?}),
        )
    }

    pub(in crate::model::app) fn rename_viewport(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let title = params
            .get("title")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("title is required"))?;
        let workspace = &mut self.dataset_mut()?.workspace;
        let changed = workspace
            .rename(&id, title.to_string())
            .map_err(|e| invalid(e.to_string()))?;
        if changed {
            let _ = workspace.bump_presentation_revision(&id);
        }
        let revision = workspace
            .get(&id)
            .map(|slot| slot.presentation_revision)
            .unwrap_or(0);
        Ok(
            json!({"changed": changed, "viewport_id": id.as_str(), "title": title.trim(), "presentation_revision": revision}),
        )
    }

    pub(in crate::model::app) fn remove_viewport(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        {
            let dataset = self.dataset_mut()?;
            dataset
                .workspace
                .remove(&id)
                .map_err(|e| invalid(e.to_string()))?;
            update_logical_geometry(dataset);
        }
        self.measured_viewports.clear();
        Ok(
            json!({"removed": true, "viewport_id": id.as_str(), "workspace": self.workspace_snapshot()?}),
        )
    }

    pub(in crate::model::app) fn set_active_viewport(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let workspace = &mut self.dataset_mut()?.workspace;
        let changed = workspace
            .set_active(&id)
            .map_err(|e| invalid(e.to_string()))?;
        Ok(
            json!({"changed": changed, "active_viewport_id": id.as_str(), "viewport": self.viewport_snapshot_for(params)?}),
        )
    }

    pub(in crate::model::app) fn links_snapshot(&self) -> Result<Value, ControlError> {
        let workspace = &self.dataset()?.workspace;
        Ok(
            json!({"links": links_json(workspace.links()), "viewport_ids": workspace.viewports().iter().map(|v| v.id.as_str()).collect::<Vec<_>>() }),
        )
    }

    pub(in crate::model::app) fn link_groups_snapshot(&self) -> Result<Value, ControlError> {
        Ok(json!({"link_groups": [self.link_group()?]}))
    }

    pub(in crate::model::app) fn link_group(&self) -> Result<Value, ControlError> {
        let workspace = &self.dataset()?.workspace;
        let links = workspace.links();
        let mut fields = Vec::new();
        if links.camera {
            fields.push("camera");
        }
        if links.plane {
            fields.push("plane");
        }
        fields.push("selection");
        Ok(
            json!({"link_group_id": "comparison-navigation", "viewport_ids": workspace.viewports().iter().map(|v| v.id.as_str()).collect::<Vec<_>>(), "fields": fields}),
        )
    }

    pub(in crate::model::app) fn set_links(
        &mut self,
        params: &Value,
        kind: LinkRequestKind,
    ) -> Result<Value, ControlError> {
        if params
            .get("link_group_id")
            .is_some_and(|v| v.as_str() != Some("comparison-navigation"))
        {
            return Err(invalid("link_group_id must be 'comparison-navigation'"));
        }
        let workspace = &self.dataset()?.workspace;
        if kind == LinkRequestKind::Create
            && params
                .get("viewports")
                .or_else(|| params.get("viewport_ids"))
                .is_none()
        {
            return Err(invalid("viewports must identify both workspace viewports"));
        }
        if let Some(requested) = params
            .get("viewports")
            .or_else(|| params.get("viewport_ids"))
        {
            validate_viewport_set(workspace, requested)?;
        }
        let current = workspace.links();
        let links = if kind != LinkRequestKind::Direct {
            let fields = params
                .get("fields")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    invalid("fields must be an array containing camera, plane, and/or selection")
                })?;
            let fields = fields
                .iter()
                .map(Value::as_str)
                .collect::<Option<HashSet<_>>>()
                .ok_or_else(|| invalid("fields must contain only strings"))?;
            for field in &fields {
                if !matches!(*field, "camera" | "plane" | "selection") {
                    return Err(invalid(format!("unknown viewport link field '{field}'")));
                }
            }
            ViewportLinks {
                camera: fields.contains("camera"),
                plane: fields.contains("plane"),
                selection: true,
            }
        } else {
            ViewportLinks {
                camera: params
                    .get("camera")
                    .and_then(Value::as_bool)
                    .unwrap_or(current.camera),
                plane: params
                    .get("plane")
                    .and_then(Value::as_bool)
                    .unwrap_or(current.plane),
                selection: params
                    .get("selection")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
            }
        };
        if !links.selection {
            return Err(invalid(
                "selection is document-shared in the two-viewport milestone",
            ));
        }
        let mut response = self.apply_links(links)?;
        if kind != LinkRequestKind::Direct {
            response["link_group"] = self.link_group()?;
        }
        Ok(response)
    }

    pub(in crate::model::app) fn apply_links(
        &mut self,
        links: ViewportLinks,
    ) -> Result<Value, ControlError> {
        let workspace = &mut self.dataset_mut()?.workspace;
        let before_revisions = workspace
            .viewports()
            .iter()
            .map(|viewport| (viewport.id.clone(), viewport.navigation_revision))
            .collect::<Vec<_>>();
        let active_id = workspace.active_id().clone();
        let active_state = workspace.active().state.clone();
        let other_ids = workspace
            .viewports()
            .iter()
            .filter(|viewport| viewport.id != active_id)
            .map(|viewport| viewport.id.clone())
            .collect::<Vec<_>>();
        for id in other_ids {
            let mut navigation_changed = false;
            if let Some(viewport) = workspace.get_mut(&id) {
                if links.camera
                    && (viewport.state.center != active_state.center
                        || viewport.state.zoom != active_state.zoom)
                {
                    viewport.state.center = active_state.center;
                    viewport.state.zoom = active_state.zoom;
                    navigation_changed = true;
                }
                if links.plane
                    && (viewport.state.plane_mode != active_state.plane_mode
                        || current_plane_slice(&viewport.state)
                            != current_plane_slice(&active_state))
                {
                    viewport.state.plane_mode = active_state.plane_mode.clone();
                    viewport.state.plane_slices = active_state.plane_slices;
                    navigation_changed = true;
                }
            }
            if navigation_changed {
                let _ = workspace.bump_navigation_revision(&id);
            }
        }
        let changed = workspace.set_links(links);
        let affected_viewport_ids = workspace
            .viewports()
            .iter()
            .filter(|viewport| {
                before_revisions
                    .iter()
                    .find(|(id, _)| *id == viewport.id)
                    .is_none_or(|(_, revision)| *revision != viewport.navigation_revision)
            })
            .map(|viewport| viewport.id.as_str().to_string())
            .collect::<Vec<_>>();
        Ok(json!({
            "changed": changed,
            "links": links_json(links),
            "affected_viewport_ids": affected_viewport_ids,
            "workspace": self.workspace_snapshot()?,
        }))
    }

    pub(in crate::model::app) fn remove_links(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        if params
            .get("link_group_id")
            .is_some_and(|v| v.as_str() != Some("comparison-navigation"))
        {
            return Err(invalid("link_group_id must be 'comparison-navigation'"));
        }
        let links = ViewportLinks {
            camera: false,
            plane: false,
            selection: true,
        };
        let mut response = self.apply_links(links)?;
        response["removed"] = Value::Bool(true);
        response["link_group"] = self.link_group()?;
        Ok(response)
    }
}
