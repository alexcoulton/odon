use super::super::*;

impl OmeZarrViewerApp {
    pub fn control_viewport_workspace_snapshot(&mut self) -> serde_json::Value {
        self.sync_runtime_to_active_viewport();
        let Some(workspace) = self.viewport_workspace.clone() else {
            return serde_json::json!({"error": "viewer workspace is not initialized"});
        };
        let active_id = workspace.active_id().clone();
        let runtime = ViewerViewportState::capture(self);
        let viewports = workspace
            .viewports()
            .iter()
            .map(|viewport| {
                viewport.state.apply(self);
                let mut snapshot = Self::viewport_state_snapshot(
                    &viewport.id,
                    &viewport.title,
                    viewport.id == active_id,
                    viewport.navigation_revision,
                    viewport.presentation_revision,
                    &viewport.state,
                );
                snapshot
                    .as_object_mut()
                    .expect("viewport snapshot is an object")
                    .insert(
                        "native_layers".to_string(),
                        self.control_native_layer_snapshot_list(),
                    );
                snapshot
            })
            .collect::<Vec<_>>();
        runtime.apply(self);
        let links = workspace.links();
        let cpu_loader_stats = self.loader.stats();
        serde_json::json!({
            "revision": workspace.revision(),
            "layout": workspace.layout().as_str(),
            "ratio": workspace.split_ratio(),
            "active_viewport_id": active_id.as_str(),
            "max_viewports": crate::viewports::MAX_VIEWPORTS,
            "shared_resources": {
                "document_instances": 1,
                "dataset_source": self.dataset.source.source_key(),
                "dataset_instances": 1,
                "cpu_tile_cache_instances": 1,
                "cpu_tile_cache_entries": self.cache.len(),
                "cpu_decoded_tile_cache_instances": 1,
                "cpu_decoded_tile_cache_entries": cpu_loader_stats.decoded_cache_entries,
                "cpu_decoded_tile_cache_bytes": cpu_loader_stats.decoded_cache_bytes,
                "cpu_decode_requests": cpu_loader_stats.decode_requests,
                "cpu_source_reads": cpu_loader_stats.source_reads,
                "cpu_decoded_cache_hits": cpu_loader_stats.cache_hits,
                "gpu_raw_tile_cache_instances": usize::from(self.tiles_gl.is_some()),
                "gpu_raw_tile_cache_entries": self.tiles_gl.as_ref().map(TilesGl::len).unwrap_or(0),
                "primary_object_geometry_instances": 1,
                "primary_object_count": self.seg_objects.object_count(),
            },
            "ui":{"right_tab":self.right_tab.storage_key()},
            "labels": self.control_labels_json(),
            "masks": self.control_mask_projection_snapshot(),
            "panels": self.control_side_panels_snapshot(),
            "channel_metadata": self.channels.iter().map(|channel| serde_json::json!({
                "index": channel.index,
                "name": channel.name,
                "note": channel.note,
            })).collect::<Vec<_>>(),
            "channel_transforms": self.channels.iter().enumerate().map(|(index, _)| {
                self.control_get_channel_transform(&serde_json::json!({"index": index}))
            }).collect::<Vec<_>>(),
            "channel_presentation": self.control_channel_presentation_json(),
            "performance": {
                "frame_plan_last_ms": self.viewport_frame_plan_ms,
                "frame_plan_ema_ms": self.viewport_frame_plan_ema_ms,
                "frame_plan_samples": self.viewport_frame_plan_samples,
            },
            "links": {
                "camera": links.camera,
                "plane": links.plane,
                "selection": links.selection,
            },
            "viewports": viewports,
        })
    }

    pub(in crate::app) fn control_canvas_rect_ready(rect: Option<egui::Rect>) -> bool {
        rect.is_some_and(|rect| {
            rect.min.x.is_finite()
                && rect.min.y.is_finite()
                && rect.max.x.is_finite()
                && rect.max.y.is_finite()
                && rect.width() > 0.0
                && rect.height() > 0.0
        })
    }

    pub fn control_active_canvas_ready(&self) -> bool {
        Self::control_canvas_rect_ready(self.last_canvas_rect)
    }

    pub fn control_viewport_canvas_ready(&self, viewport_id: &str) -> Option<bool> {
        let viewport_id = ViewportId::new(viewport_id).ok()?;
        let viewport = self.viewport_workspace.as_ref()?.get(&viewport_id)?;
        Some(Self::control_canvas_rect_ready(
            viewport.state.last_canvas_rect,
        ))
    }

    pub fn control_workspace_canvas_ready(&self) -> bool {
        self.viewport_workspace.as_ref().is_some_and(|workspace| {
            !workspace.viewports().is_empty()
                && workspace.viewports().iter().all(|viewport| {
                    Self::control_canvas_rect_ready(viewport.state.last_canvas_rect)
                })
        })
    }

    pub fn workspace_canvas_rect(&mut self) -> Option<egui::Rect> {
        self.sync_runtime_to_active_viewport();
        self.viewport_workspace
            .as_ref()?
            .viewports()
            .iter()
            .filter_map(|viewport| viewport.state.last_canvas_rect)
            .reduce(|union, rect| union.union(rect))
    }

    pub fn control_get_viewport(&mut self, params: &serde_json::Value) -> serde_json::Value {
        self.sync_runtime_to_active_viewport();
        let viewport_id = match Self::parse_viewport_id(params) {
            Ok(id) => id,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let Some(workspace) = self.viewport_workspace.as_ref() else {
            return serde_json::json!({"error": "viewer workspace is not initialized"});
        };
        let Some(viewport) = workspace.get(&viewport_id) else {
            return serde_json::json!({"error": format!("viewport '{viewport_id}' was not found")});
        };
        Self::viewport_state_snapshot(
            &viewport.id,
            &viewport.title,
            viewport.id == *workspace.active_id(),
            viewport.navigation_revision,
            viewport.presentation_revision,
            &viewport.state,
        )
    }

    pub fn control_create_viewport(&mut self, params: &serde_json::Value) -> serde_json::Value {
        self.sync_runtime_to_active_viewport();
        let split_ratio = match Self::parse_viewport_split_ratio(params) {
            Ok(value) => value,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let layout = match params
            .get("layout")
            .and_then(serde_json::Value::as_str)
            .map(ViewportLayout::parse)
        {
            Some(Some(layout @ (ViewportLayout::Horizontal | ViewportLayout::Vertical))) => layout,
            Some(Some(ViewportLayout::Single)) => {
                return serde_json::json!({"error": "creating a second viewport requires a split layout"});
            }
            Some(None) => {
                return serde_json::json!({"error": "layout must be 'horizontal' or 'vertical'"});
            }
            None => ViewportLayout::Horizontal,
        };
        let title = params
            .get("title")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
        let activate = params
            .get("activate")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);
        let Some(mut workspace) = self.viewport_workspace.take() else {
            return serde_json::json!({"error": "viewer workspace is not initialized"});
        };
        let previous_active = workspace.active_id().clone();
        let source_id = match params
            .get("source_viewport_id")
            .or_else(|| params.get("viewport_id"))
            .and_then(serde_json::Value::as_str)
        {
            Some(value) => match ViewportId::new(value) {
                Ok(id) => id,
                Err(error) => {
                    self.viewport_workspace = Some(workspace);
                    return serde_json::json!({"error": error.to_string()});
                }
            },
            None => previous_active.clone(),
        };
        let created = workspace.clone_viewport(&source_id, title, layout);
        let viewport_id = match created {
            Ok(id) => id,
            Err(error) => {
                self.viewport_workspace = Some(workspace);
                return serde_json::json!({"error": error.to_string()});
            }
        };
        if let Some(ratio) = split_ratio {
            if let Err(error) = workspace.set_split_ratio(ratio) {
                self.viewport_workspace = Some(workspace);
                return serde_json::json!({"error": error.to_string()});
            }
        }
        if !activate {
            let _ = workspace.set_active(&previous_active);
        } else {
            self.cancel_viewport_transient_gestures();
        }
        workspace.active().state.apply(self);
        self.bump_render_id();
        self.viewport_workspace = Some(workspace);
        serde_json::json!({
            "created": true,
            "viewport_id": viewport_id.as_str(),
            "workspace": self.control_viewport_workspace_snapshot(),
        })
    }

    pub fn control_remove_viewport(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let viewport_id = match Self::parse_viewport_id(params) {
            Ok(id) => id,
            Err(error) => return serde_json::json!({"error": error}),
        };
        match self.remove_viewport(&viewport_id) {
            Ok(()) => serde_json::json!({
                "removed": true,
                "viewport_id": viewport_id.as_str(),
                "workspace": self.control_viewport_workspace_snapshot(),
            }),
            Err(error) => serde_json::json!({"error": error}),
        }
    }

    pub fn control_set_active_viewport(&mut self, params: &serde_json::Value) -> serde_json::Value {
        self.sync_runtime_to_active_viewport();
        let viewport_id = match Self::parse_viewport_id(params) {
            Ok(id) => id,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let Some(mut workspace) = self.viewport_workspace.take() else {
            return serde_json::json!({"error": "viewer workspace is not initialized"});
        };
        let changed = match workspace.set_active(&viewport_id) {
            Ok(changed) => changed,
            Err(error) => {
                self.viewport_workspace = Some(workspace);
                return serde_json::json!({"error": error.to_string()});
            }
        };
        if changed {
            self.cancel_viewport_transient_gestures();
        }
        workspace.active().state.apply(self);
        self.bump_render_id();
        self.viewport_workspace = Some(workspace);
        serde_json::json!({
            "changed": changed,
            "active_viewport_id": viewport_id.as_str(),
            "viewport": self.control_get_viewport(&serde_json::json!({
                "viewport_id": viewport_id.as_str(),
            })),
        })
    }

    pub fn control_rename_viewport(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let viewport_id = match Self::parse_viewport_id(params) {
            Ok(id) => id,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let Some(title) = params.get("title").and_then(serde_json::Value::as_str) else {
            return serde_json::json!({"error": "title is required"});
        };
        let Some(workspace) = self.viewport_workspace.as_mut() else {
            return serde_json::json!({"error": "viewer workspace is not initialized"});
        };
        if let Some(expected) = params
            .get("if_presentation_revision")
            .and_then(serde_json::Value::as_u64)
        {
            let Some(current) = workspace
                .get(&viewport_id)
                .map(|viewport| viewport.presentation_revision)
            else {
                return serde_json::json!({"error": format!("viewport '{viewport_id}' was not found")});
            };
            if expected != current {
                return serde_json::json!({
                    "error": format!(
                        "viewport presentation revision conflict: expected {expected}, current {current}"
                    ),
                    "viewport_id": viewport_id.as_str(),
                    "expected_revision": expected,
                    "current_revision": current,
                    "revision_domain": "presentation",
                });
            }
        }
        match workspace.rename(&viewport_id, title.to_string()) {
            Ok(changed) => {
                let presentation_revision = if changed {
                    workspace
                        .bump_presentation_revision(&viewport_id)
                        .unwrap_or(1)
                } else {
                    workspace
                        .get(&viewport_id)
                        .map(|viewport| viewport.presentation_revision)
                        .unwrap_or(1)
                };
                serde_json::json!({
                    "changed": changed,
                    "viewport_id": viewport_id.as_str(),
                    "title": title.trim(),
                    "presentation_revision": presentation_revision,
                })
            }
            Err(error) => serde_json::json!({"error": error.to_string()}),
        }
    }

    pub fn control_set_viewport_layout(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(value) = params.get("layout").and_then(serde_json::Value::as_str) else {
            return serde_json::json!({"error": "layout is required"});
        };
        let Some(layout) = ViewportLayout::parse(value) else {
            return serde_json::json!({"error": "layout must be 'single', 'horizontal', or 'vertical'"});
        };
        if let Some(requested) = params
            .get("viewports")
            .or_else(|| params.get("viewport_ids"))
        {
            let Some(requested) = requested.as_array() else {
                return serde_json::json!({"error": "viewports must be an array of viewport IDs"});
            };
            let requested = requested
                .iter()
                .map(|value| value.as_str().map(str::to_string))
                .collect::<Option<Vec<_>>>();
            let Some(requested) = requested else {
                return serde_json::json!({"error": "viewports must contain only viewport ID strings"});
            };
            let current = self
                .viewport_workspace
                .as_ref()
                .map(|workspace| {
                    workspace
                        .viewports()
                        .iter()
                        .map(|viewport| viewport.id.as_str().to_string())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if requested != current {
                return serde_json::json!({
                    "error": "viewports must match the current workspace order; use viewer.workspace.swap to reorder",
                });
            }
        }
        let ratio = match Self::parse_viewport_split_ratio(params) {
            Ok(value) => value,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let mut changed = match self.set_viewport_layout(layout) {
            Ok(changed) => changed,
            Err(error) => return serde_json::json!({"error": error}),
        };
        if let Some(ratio) = ratio {
            let Some(workspace) = self.viewport_workspace.as_mut() else {
                return serde_json::json!({"error": "viewer workspace is not initialized"});
            };
            match workspace.set_split_ratio(ratio) {
                Ok(ratio_changed) => changed |= ratio_changed,
                Err(error) => return serde_json::json!({"error": error.to_string()}),
            }
        }
        serde_json::json!({
            "changed": changed,
            "layout": layout.as_str(),
            "ratio": self.viewport_workspace.as_ref().map(ViewportWorkspace::split_ratio),
        })
    }

    pub fn control_get_viewport_layout(&mut self) -> serde_json::Value {
        let workspace = self.control_viewport_workspace_snapshot();
        serde_json::json!({
            "revision": workspace["revision"],
            "layout": workspace["layout"],
            "ratio": workspace["ratio"],
            "viewport_ids": workspace["viewports"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|viewport| viewport["viewport_id"].as_str())
                .collect::<Vec<_>>(),
        })
    }

    pub fn control_swap_viewports(&mut self) -> serde_json::Value {
        let Some(workspace) = self.viewport_workspace.as_mut() else {
            return serde_json::json!({"error": "viewer workspace is not initialized"});
        };
        if !workspace.swap_order() {
            return serde_json::json!({"error": "swapping requires exactly two viewports"});
        }
        serde_json::json!({
            "changed": true,
            "workspace": self.control_viewport_workspace_snapshot(),
        })
    }

    pub fn control_get_viewport_links(&mut self) -> serde_json::Value {
        let workspace = self.control_viewport_workspace_snapshot();
        serde_json::json!({
            "links": workspace["links"],
            "viewport_ids": workspace["viewports"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|viewport| viewport["viewport_id"].as_str())
                .collect::<Vec<_>>(),
        })
    }

    pub(in crate::app) fn control_viewport_link_group_snapshot(&self) -> serde_json::Value {
        let links = self
            .viewport_workspace
            .as_ref()
            .map(ViewportWorkspace::links)
            .unwrap_or_default();
        let viewport_ids = self
            .viewport_workspace
            .as_ref()
            .map(|workspace| {
                workspace
                    .viewports()
                    .iter()
                    .map(|viewport| viewport.id.as_str().to_string())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let mut fields = Vec::new();
        if links.camera {
            fields.push("camera");
        }
        if links.plane {
            fields.push("plane");
        }
        // Selection identity belongs to the document in this milestone, so it
        // remains represented in the fixed comparison link group.
        fields.push("selection");
        serde_json::json!({
            "link_group_id": "comparison-navigation",
            "viewport_ids": viewport_ids,
            "fields": fields,
        })
    }

    pub fn control_list_viewport_link_groups(&self) -> serde_json::Value {
        serde_json::json!({
            "link_groups": [self.control_viewport_link_group_snapshot()],
        })
    }

    pub(in crate::app) fn validate_viewport_link_group_request(
        &self,
        params: &serde_json::Value,
        require_viewports: bool,
    ) -> Result<ViewportLinks, String> {
        if let Some(link_group_id) = params.get("link_group_id") {
            if link_group_id.as_str() != Some("comparison-navigation") {
                return Err("link_group_id must be 'comparison-navigation'".to_string());
            }
        }

        let requested_viewports = params
            .get("viewport_ids")
            .or_else(|| params.get("viewports"));
        if require_viewports && requested_viewports.is_none() {
            return Err("viewports must identify both workspace viewports".to_string());
        }
        if let Some(value) = requested_viewports {
            let Some(values) = value.as_array() else {
                return Err("viewports must be an array of viewport IDs".to_string());
            };
            let mut requested = HashSet::new();
            for value in values {
                let Some(id) = value.as_str() else {
                    return Err("viewports must contain only viewport ID strings".to_string());
                };
                if !requested.insert(id.to_string()) {
                    return Err("viewports must not contain duplicate IDs".to_string());
                }
            }
            let current = self
                .viewport_workspace
                .as_ref()
                .map(|workspace| {
                    workspace
                        .viewports()
                        .iter()
                        .map(|viewport| viewport.id.as_str().to_string())
                        .collect::<HashSet<_>>()
                })
                .unwrap_or_default();
            if current.len() != 2 || requested != current {
                return Err(
                    "viewports must identify exactly the two current workspace viewports"
                        .to_string(),
                );
            }
        }

        let Some(fields) = params.get("fields").and_then(serde_json::Value::as_array) else {
            return Err(
                "fields must be an array containing camera, plane, and/or selection".to_string(),
            );
        };
        let mut camera = false;
        let mut plane = false;
        for field in fields {
            match field.as_str() {
                Some("camera") => camera = true,
                Some("plane") => plane = true,
                Some("selection") => {}
                Some(other) => return Err(format!("unknown viewport link field '{other}'")),
                None => return Err("fields must contain only strings".to_string()),
            }
        }
        Ok(ViewportLinks {
            camera,
            plane,
            selection: true,
        })
    }

    pub(in crate::app) fn control_configure_viewport_link_group(
        &mut self,
        params: &serde_json::Value,
        require_viewports: bool,
    ) -> serde_json::Value {
        let links = match self.validate_viewport_link_group_request(params, require_viewports) {
            Ok(links) => links,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let mut response = self.control_set_viewport_links(&serde_json::json!({
            "camera": links.camera,
            "plane": links.plane,
            "selection": true,
        }));
        if response.get("error").is_none() {
            response["link_group"] = self.control_viewport_link_group_snapshot();
        }
        response
    }

    pub fn control_create_viewport_link_group(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        self.control_configure_viewport_link_group(params, true)
    }

    pub fn control_update_viewport_link_group(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        self.control_configure_viewport_link_group(params, false)
    }

    pub fn control_remove_viewport_link_group(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        if params
            .get("link_group_id")
            .is_some_and(|value| value.as_str() != Some("comparison-navigation"))
        {
            return serde_json::json!({
                "error": "link_group_id must be 'comparison-navigation'",
            });
        }
        let mut response = self.control_set_viewport_links(&serde_json::json!({
            "camera": false,
            "plane": false,
            "selection": true,
        }));
        if response.get("error").is_none() {
            response["removed"] = serde_json::Value::Bool(true);
            response["link_group"] = self.control_viewport_link_group_snapshot();
        }
        response
    }

    pub fn control_set_viewport_links(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let before = self.control_viewport_workspace_snapshot();
        let current = self
            .viewport_workspace
            .as_ref()
            .map(ViewportWorkspace::links)
            .unwrap_or_default();
        let links = ViewportLinks {
            camera: params
                .get("camera")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(current.camera),
            plane: params
                .get("plane")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(current.plane),
            selection: params
                .get("selection")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(true),
        };
        if !links.selection {
            return serde_json::json!({
                "error": "selection is document-shared in the two-viewport milestone",
            });
        }
        let changed = self.set_viewport_links(links);
        let workspace = self.control_viewport_workspace_snapshot();
        let before_revisions = before["viewports"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|viewport| {
                Some((
                    viewport["viewport_id"].as_str()?.to_string(),
                    viewport["navigation_revision"].as_u64()?,
                ))
            })
            .collect::<HashMap<_, _>>();
        let affected_viewport_ids = workspace["viewports"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|viewport| {
                let id = viewport["viewport_id"].as_str()?;
                let revision = viewport["navigation_revision"].as_u64()?;
                (before_revisions.get(id).copied() != Some(revision)).then(|| id.to_string())
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "changed": changed,
            "links": {
                "camera": links.camera,
                "plane": links.plane,
                "selection": links.selection,
            },
            "affected_viewport_ids": affected_viewport_ids,
            "workspace": workspace,
        })
    }

    pub(in crate::app) fn control_in_viewport(
        &mut self,
        params: &serde_json::Value,
        domain: ViewportControlDomain,
        operation: impl FnOnce(&mut Self, &serde_json::Value) -> serde_json::Value,
    ) -> serde_json::Value {
        self.sync_runtime_to_active_viewport();
        let viewport_id = match Self::parse_viewport_id(params) {
            Ok(id) => id,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let Some(mut workspace) = self.viewport_workspace.take() else {
            return serde_json::json!({"error": "viewer workspace is not initialized"});
        };
        let active_viewport_id = workspace.active_id().clone();
        let active_before = workspace.active().state.clone();
        let Some(target) = workspace.get(&viewport_id) else {
            self.viewport_workspace = Some(workspace);
            return serde_json::json!({"error": format!("viewport '{viewport_id}' was not found")});
        };
        let revision_guard = match domain {
            ViewportControlDomain::Read => None,
            ViewportControlDomain::Navigation => params
                .get("if_navigation_revision")
                .and_then(serde_json::Value::as_u64)
                .map(|expected| ("navigation", expected, target.navigation_revision)),
            ViewportControlDomain::Presentation => params
                .get("if_presentation_revision")
                .and_then(serde_json::Value::as_u64)
                .map(|expected| ("presentation", expected, target.presentation_revision)),
        };
        if let Some((kind, expected, current)) = revision_guard
            && expected != current
        {
            self.viewport_workspace = Some(workspace);
            return serde_json::json!({
                "error": format!(
                    "viewport {kind} revision conflict: expected {expected}, current {current}"
                ),
                "viewport_id": viewport_id.as_str(),
                "expected_revision": expected,
                "current_revision": current,
                "revision_domain": kind,
            });
        }
        let before = target.state.clone();
        before.apply(self);
        self.bump_render_id();
        let result = operation(self, params);
        let after = ViewerViewportState::capture(self);
        let succeeded = result.get("error").is_none();
        if let Some(target) = workspace.get_mut(&viewport_id) {
            target.state = after.clone();
        }
        let mut affected_viewport_ids = vec![viewport_id.as_str().to_string()];
        if succeeded {
            match domain {
                ViewportControlDomain::Read => {}
                ViewportControlDomain::Navigation => {
                    let _ = workspace.bump_navigation_revision(&viewport_id);
                }
                ViewportControlDomain::Presentation => {
                    let _ = workspace.bump_presentation_revision(&viewport_id);
                }
            }
        }
        let links = workspace.links();
        if succeeded
            && domain == ViewportControlDomain::Navigation
            && ((links.camera && after.camera_changed_from(&before))
                || (links.plane && after.plane_changed_from(&before)))
        {
            let other_ids = workspace
                .viewports()
                .iter()
                .filter(|viewport| viewport.id != viewport_id)
                .map(|viewport| viewport.id.clone())
                .collect::<Vec<_>>();
            for other_id in other_ids {
                if let Some(other) = workspace.get_mut(&other_id) {
                    other.state.copy_linked_navigation_from(&after, links);
                }
                let _ = workspace.bump_navigation_revision(&other_id);
                affected_viewport_ids.push(other_id.as_str().to_string());
            }
        }
        let target = workspace
            .get(&viewport_id)
            .expect("target viewport remains in workspace");
        let navigation_revision = target.navigation_revision;
        let presentation_revision = target.presentation_revision;
        let link_transaction_id = (affected_viewport_ids.len() > 1)
            .then(|| format!("{}-{navigation_revision}", viewport_id.as_str()));
        let active_after = &workspace.active().state;
        let active_viewport_changed = succeeded
            && match domain {
                ViewportControlDomain::Read => false,
                ViewportControlDomain::Navigation => {
                    active_after.camera_changed_from(&active_before)
                        || active_after.plane_changed_from(&active_before)
                }
                ViewportControlDomain::Presentation => {
                    active_after.presentation_changed_from(&active_before)
                }
            };
        workspace.active().state.apply(self);
        self.bump_render_id();
        self.viewport_workspace = Some(workspace);
        serde_json::json!({
            "viewport_id": viewport_id.as_str(),
            "navigation_revision": navigation_revision,
            "presentation_revision": presentation_revision,
            "affected_viewport_ids": affected_viewport_ids,
            "link_transaction_id": link_transaction_id,
            "active_viewport_id": active_viewport_id.as_str(),
            "active_viewport_changed": active_viewport_changed,
            "result": result,
        })
    }

    pub(in crate::app) fn control_filter_sensitive_operation(
        &mut self,
        params: &serde_json::Value,
        operation: impl FnOnce(&mut Self, &serde_json::Value) -> serde_json::Value,
    ) -> serde_json::Value {
        if params.get("viewport_id").is_some() {
            return self.control_in_viewport(params, ViewportControlDomain::Read, operation);
        }
        if params.get("filter_query").is_some()
            || params
                .get("use_all_objects")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
        {
            return self.control_with_temporary_object_filter(params, operation);
        }
        if params
            .get("use_active_viewport_filter")
            .and_then(serde_json::Value::as_bool)
            == Some(true)
            || self
                .viewport_workspace
                .as_ref()
                .is_none_or(|workspace| workspace.len() <= 1)
        {
            return operation(self, params);
        }
        serde_json::json!({
            "error": "multi-viewport filter-sensitive operations require viewport_id, filter_query, use_all_objects=true, or explicit use_active_viewport_filter=true",
        })
    }

    pub(in crate::app) fn control_with_temporary_object_filter(
        &mut self,
        params: &serde_json::Value,
        operation: impl FnOnce(&mut Self, &serde_json::Value) -> serde_json::Value,
    ) -> serde_json::Value {
        let mut effective = params.clone();
        if effective.get("target").is_none()
            && let Some(object) = effective.as_object_mut()
        {
            // A standalone query is a data input, so it defaults to the stable
            // primary object resource instead of whichever layer is active.
            object.insert("target".to_string(), serde_json::json!("objects"));
        }
        let target = match self.control_object_selection_target(&effective) {
            Ok(target @ (LayerId::SegmentationObjects | LayerId::SpatialShape(_))) => target,
            Ok(_) => {
                return serde_json::json!({
                    "error": "filter-sensitive operations require an object-backed target",
                });
            }
            Err(error) => return serde_json::json!({"error": error}),
        };
        let use_all = params
            .get("use_all_objects")
            .and_then(serde_json::Value::as_bool)
            == Some(true);
        let query = params
            .get("filter_query")
            .and_then(serde_json::Value::as_str);
        if !use_all && query.is_none() {
            return serde_json::json!({"error": "filter_query must be a string"});
        }

        let (saved_filter, saved_cache, query_error) = match target {
            LayerId::SegmentationObjects => {
                let filter = self.seg_objects.viewport_filter_state();
                let cache = self.seg_objects.viewport_filter_cache_state();
                if use_all {
                    self.seg_objects.clear_filter();
                } else if let Some(query) = query {
                    self.seg_objects.set_filter_query_from_text(query);
                }
                let error = self.seg_objects.filter_snapshot_json()["query"]["error"]
                    .as_str()
                    .map(str::to_string);
                (filter, cache, error)
            }
            LayerId::SpatialShape(id) => {
                let Some(objects) = self
                    .spatial_layers
                    .shapes
                    .iter_mut()
                    .find(|layer| layer.id == id)
                    .and_then(|layer| layer.object_layer_mut())
                else {
                    return serde_json::json!({
                        "error": format!("spatial shape layer {id} has no object layer"),
                    });
                };
                let filter = objects.viewport_filter_state();
                let cache = objects.viewport_filter_cache_state();
                if use_all {
                    objects.clear_filter();
                } else if let Some(query) = query {
                    objects.set_filter_query_from_text(query);
                }
                let error = objects.filter_snapshot_json()["query"]["error"]
                    .as_str()
                    .map(str::to_string);
                (filter, cache, error)
            }
            _ => unreachable!(),
        };
        if let Some(error) = query_error {
            match target {
                LayerId::SegmentationObjects => {
                    self.seg_objects.apply_viewport_filter_state(&saved_filter);
                    self.seg_objects
                        .apply_viewport_filter_cache_state(&saved_cache);
                }
                LayerId::SpatialShape(id) => {
                    if let Some(objects) = self
                        .spatial_layers
                        .shapes
                        .iter_mut()
                        .find(|layer| layer.id == id)
                        .and_then(|layer| layer.object_layer_mut())
                    {
                        objects.apply_viewport_filter_state(&saved_filter);
                        objects.apply_viewport_filter_cache_state(&saved_cache);
                    }
                }
                _ => unreachable!(),
            }
            return serde_json::json!({"error": format!("invalid filter_query: {error}")});
        }

        let result = operation(self, &effective);
        match target {
            LayerId::SegmentationObjects => {
                self.seg_objects.apply_viewport_filter_state(&saved_filter);
                self.seg_objects
                    .apply_viewport_filter_cache_state(&saved_cache);
            }
            LayerId::SpatialShape(id) => {
                if let Some(objects) = self
                    .spatial_layers
                    .shapes
                    .iter_mut()
                    .find(|layer| layer.id == id)
                    .and_then(|layer| layer.object_layer_mut())
                {
                    objects.apply_viewport_filter_state(&saved_filter);
                    objects.apply_viewport_filter_cache_state(&saved_cache);
                }
            }
            _ => unreachable!(),
        }
        result
    }

    pub fn control_get_viewport_camera(&mut self, params: &serde_json::Value) -> serde_json::Value {
        self.control_in_viewport(params, ViewportControlDomain::Read, |app, _| {
            app.control_camera_snapshot()
        })
    }

    pub fn control_set_viewport_camera(&mut self, params: &serde_json::Value) -> serde_json::Value {
        self.control_in_viewport(
            params,
            ViewportControlDomain::Navigation,
            OmeZarrViewerApp::control_set_camera,
        )
    }

    pub fn control_fit_viewport_camera(&mut self, params: &serde_json::Value) -> serde_json::Value {
        self.control_in_viewport(params, ViewportControlDomain::Navigation, |app, _| {
            app.control_fit_to_view()
        })
    }

    pub fn control_get_viewport_plane(&mut self, params: &serde_json::Value) -> serde_json::Value {
        self.control_in_viewport(params, ViewportControlDomain::Read, |app, _| {
            app.control_plane_snapshot()
        })
    }

    pub fn control_set_viewport_plane(&mut self, params: &serde_json::Value) -> serde_json::Value {
        self.control_in_viewport(
            params,
            ViewportControlDomain::Navigation,
            OmeZarrViewerApp::control_set_plane,
        )
    }

    pub fn control_get_viewport_object_style(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        self.control_in_viewport(params, ViewportControlDomain::Read, |app, _| {
            app.seg_objects.control_style_snapshot_json()
        })
    }

    pub fn control_set_viewport_object_style(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        self.control_in_viewport(
            params,
            ViewportControlDomain::Presentation,
            |app, params| {
                let result = app.seg_objects.control_set_style_json(params);
                app.bump_render_id();
                result
            },
        )
    }

    pub fn control_set_viewport_object_legend(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        self.control_in_viewport(
            params,
            ViewportControlDomain::Presentation,
            |app, params| {
                let result = app.seg_objects.control_set_legend_json(params);
                app.bump_render_id();
                result
            },
        )
    }

    pub fn control_get_viewport_object_filter(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        self.control_in_viewport(params, ViewportControlDomain::Read, |app, _| {
            app.seg_objects.filter_snapshot_json()
        })
    }

    pub fn control_set_viewport_object_filter(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        self.control_in_viewport(
            params,
            ViewportControlDomain::Presentation,
            |app, params| {
                let result = if params.get("model").is_some()
                    || params.get("clauses").is_some()
                    || params.get("logic").is_some()
                    || params.get("mode").is_some()
                {
                    app.seg_objects.control_set_filter_model_json(params)
                } else if let Some(query) = params
                    .get("query")
                    .or_else(|| params.get("expression"))
                    .and_then(serde_json::Value::as_str)
                {
                    app.seg_objects.set_filter_query_from_text(query);
                    app.seg_objects.filter_snapshot_json()
                } else {
                    serde_json::json!({"error": "provide query or a filter model"})
                };
                app.bump_render_id();
                result
            },
        )
    }

    pub fn control_clear_viewport_object_filter(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        self.control_in_viewport(params, ViewportControlDomain::Presentation, |app, _| {
            app.seg_objects.clear_filter();
            app.bump_render_id();
            app.seg_objects.filter_snapshot_json()
        })
    }

    pub fn control_get_viewport_channels(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        self.control_in_viewport(params, ViewportControlDomain::Read, |app, _| {
            app.control_channel_snapshot()
        })
    }

    pub fn control_set_viewport_channels(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        self.control_in_viewport(
            params,
            ViewportControlDomain::Presentation,
            OmeZarrViewerApp::control_set_visible_channels,
        )
    }

    pub fn control_set_viewport_active_channel(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        self.control_in_viewport(
            params,
            ViewportControlDomain::Presentation,
            OmeZarrViewerApp::control_set_active_channel,
        )
    }

    pub fn control_set_viewport_channel_color(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        self.control_in_viewport(
            params,
            ViewportControlDomain::Presentation,
            OmeZarrViewerApp::control_set_channel_color,
        )
    }

    pub fn control_set_viewport_channel_contrast(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        self.control_in_viewport(
            params,
            ViewportControlDomain::Presentation,
            OmeZarrViewerApp::control_set_channel_contrast,
        )
    }

    pub fn control_set_viewport_channel_order(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        self.control_in_viewport(
            params,
            ViewportControlDomain::Presentation,
            OmeZarrViewerApp::control_set_channel_order,
        )
    }

    pub fn control_get_viewport_channel_groups(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        self.control_in_viewport(params, ViewportControlDomain::Read, |app, _| {
            app.control_channel_groups_snapshot()
        })
    }

    pub fn control_set_viewport_channel_group(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        self.control_in_viewport(
            params,
            ViewportControlDomain::Presentation,
            OmeZarrViewerApp::control_set_channel_group,
        )
    }

    pub fn control_get_viewport_layers(&mut self, params: &serde_json::Value) -> serde_json::Value {
        self.control_in_viewport(params, ViewportControlDomain::Read, |app, _| {
            app.control_native_layer_snapshot_list()
        })
    }

    pub fn control_get_viewport_layer(&mut self, params: &serde_json::Value) -> serde_json::Value {
        self.control_in_viewport(params, ViewportControlDomain::Read, |app, params| {
            app.control_get_native_layer(params)
        })
    }

    pub fn control_set_viewport_layer(&mut self, params: &serde_json::Value) -> serde_json::Value {
        self.control_in_viewport(
            params,
            ViewportControlDomain::Presentation,
            OmeZarrViewerApp::control_set_native_layer_presentation,
        )
    }

    pub fn control_get_viewport_rendering(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        self.control_in_viewport(params, ViewportControlDomain::Read, |app, _| {
            app.control_rendering_snapshot()
        })
    }

    pub fn control_set_viewport_rendering(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        self.control_in_viewport(
            params,
            ViewportControlDomain::Presentation,
            OmeZarrViewerApp::control_set_rendering,
        )
    }

    pub fn control_set_viewport_layer_visibility(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        self.control_in_viewport(
            params,
            ViewportControlDomain::Presentation,
            OmeZarrViewerApp::control_set_native_layer_visibility,
        )
    }

    pub fn control_set_viewport_layer_order(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        self.control_in_viewport(
            params,
            ViewportControlDomain::Presentation,
            OmeZarrViewerApp::control_set_native_layer_order,
        )
    }

    pub fn control_set_viewport_active_layer(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        self.control_in_viewport(
            params,
            ViewportControlDomain::Presentation,
            OmeZarrViewerApp::control_set_active_native_layer,
        )
    }
}
