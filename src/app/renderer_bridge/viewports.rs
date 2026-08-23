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
            "tile_loading_observation": {
                "cache": {"loaded": self.cache.len(), "capacity": self.cache.capacity(), "in_flight": self.cache.in_flight_len()},
                "target_level": self.last_target_level,
                "realized_generation": self.control_actor_tile_policy_generation,
                "status": self.tile_loading_status,
            },
            "links": {
                "camera": links.camera,
                "plane": links.plane,
                "selection": links.selection,
            },
            "viewports": viewports,
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

    pub fn control_workspace_canvas_ready(&self) -> bool {
        self.viewport_workspace.as_ref().is_some_and(|workspace| {
            !workspace.viewports().is_empty()
                && workspace.viewports().iter().all(|viewport| {
                    Self::control_canvas_rect_ready(viewport.state.last_canvas_rect)
                })
        })
    }
}

#[cfg(test)]
impl OmeZarrViewerApp {
    pub fn control_viewport_canvas_ready(&self, viewport_id: &str) -> Option<bool> {
        let viewport_id = ViewportId::new(viewport_id).ok()?;
        let viewport = self.viewport_workspace.as_ref()?.get(&viewport_id)?;
        Some(Self::control_canvas_rect_ready(
            viewport.state.last_canvas_rect,
        ))
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

    #[cfg(test)]
    pub fn control_get_viewport_layer(&mut self, params: &serde_json::Value) -> serde_json::Value {
        self.control_in_viewport(params, ViewportControlDomain::Read, |app, params| {
            app.control_get_native_layer(params)
        })
    }

    #[cfg(test)]
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
}
