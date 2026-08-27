use super::super::*;

impl OmeZarrViewerApp {
    pub fn control_renderer_observation_snapshot(&mut self) -> serde_json::Value {
        let active_id = self
            .viewport_workspace
            .as_ref()
            .map(|workspace| workspace.active_id().clone());
        let runtime = ViewerViewportState::capture(self);
        let native_layer_observations = self
            .viewport_workspace
            .clone()
            .into_iter()
            .flat_map(|workspace| workspace.viewports().to_vec())
            .map(|viewport| {
                if active_id.as_ref() != Some(&viewport.id) {
                    viewport.state.apply(self);
                }
                serde_json::json!({
                    "viewport_id": viewport.id.as_str(),
                    "native_layers": self.control_native_layer_snapshot_list(),
                })
            })
            .collect::<Vec<_>>();
        runtime.apply(self);

        let cpu_loader_stats = self.loader.stats();
        serde_json::json!({
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
            "performance": {
                "frame_plan_last_ms": self.viewport_frame_plan_ms,
                "frame_plan_ema_ms": self.viewport_frame_plan_ema_ms,
                "frame_plan_samples": self.viewport_frame_plan_samples,
                "object_fill": self.seg_objects.render_diagnostics_json(),
            },
            "tile_loading_observation": {
                "cache": {
                    "loaded": self.cache.len(),
                    "capacity": self.cache.capacity(),
                    "in_flight": self.cache.in_flight_len(),
                },
                "target_level": self.last_target_level,
                "realized_generation": self.control_actor_tile_policy_generation,
                "status": self.tile_loading_status,
            },
            "native_layer_observations": native_layer_observations,
        })
    }

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
            "ui":{
                "left_tab":self.left_tab.storage_key(),
                "right_tab":self.right_tab.storage_key(),
            },
            "labels": self.control_labels_json(),
            "masks": self.control_mask_projection_snapshot(),
            "panels": self.control_side_panels_snapshot(),
            "channel_metadata": self.channels.iter().map(|channel| serde_json::json!({
                "index": channel.index,
                "name": channel.name,
                "note": channel.note,
            })).collect::<Vec<_>>(),
            "channel_transforms": self.channels.iter().enumerate().map(|(index, _)| {
                self.channel_transform_snapshot(index)
            }).collect::<Vec<_>>(),
            "channel_presentation": self.control_channel_presentation_json(),
            "performance": {
                "frame_plan_last_ms": self.viewport_frame_plan_ms,
                "frame_plan_ema_ms": self.viewport_frame_plan_ema_ms,
                "frame_plan_samples": self.viewport_frame_plan_samples,
                "object_fill": self.seg_objects.render_diagnostics_json(),
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
            .filter_map(|viewport| viewport.state.render.last_canvas_rect)
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
                    Self::control_canvas_rect_ready(viewport.state.render.last_canvas_rect)
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
            viewport.state.render.last_canvas_rect,
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

    pub(in crate::app) fn control_in_viewport(
        &mut self,
        params: &serde_json::Value,
        operation: impl FnOnce(&mut Self, &serde_json::Value) -> serde_json::Value,
    ) -> serde_json::Value {
        self.sync_runtime_to_active_viewport();
        let viewport_id = match Self::parse_viewport_id(params) {
            Ok(id) => id,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let Some(workspace) = self.viewport_workspace.take() else {
            return serde_json::json!({"error": "viewer workspace is not initialized"});
        };
        let active_viewport_id = workspace.active_id().clone();
        let Some(target) = workspace.get(&viewport_id) else {
            self.viewport_workspace = Some(workspace);
            return serde_json::json!({"error": format!("viewport '{viewport_id}' was not found")});
        };
        let target_state = target.state.clone();
        let navigation_revision = target.navigation_revision;
        let presentation_revision = target.presentation_revision;
        target_state.apply(self);
        self.bump_render_id();
        let result = operation(self, params);
        workspace.active().state.apply(self);
        self.bump_render_id();
        self.viewport_workspace = Some(workspace);
        serde_json::json!({
            "viewport_id": viewport_id.as_str(),
            "navigation_revision": navigation_revision,
            "presentation_revision": presentation_revision,
            "affected_viewport_ids": [viewport_id.as_str()],
            "link_transaction_id": null,
            "active_viewport_id": active_viewport_id.as_str(),
            "active_viewport_changed": false,
            "result": result,
        })
    }

    pub fn control_get_viewport_camera(&mut self, params: &serde_json::Value) -> serde_json::Value {
        self.control_in_viewport(params, |app, _| app.control_camera_snapshot())
    }

    pub fn control_get_viewport_plane(&mut self, params: &serde_json::Value) -> serde_json::Value {
        self.control_in_viewport(params, |app, _| app.control_plane_snapshot())
    }

    pub fn control_get_viewport_object_filter(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        self.control_in_viewport(params, |app, _| app.seg_objects.filter_snapshot_json())
    }

    pub fn control_get_viewport_channels(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        self.control_in_viewport(params, |app, _| app.control_channel_snapshot())
    }

    pub fn control_get_viewport_channel_groups(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        self.control_in_viewport(params, |app, _| app.control_channel_groups_snapshot())
    }

    #[cfg(test)]
    pub fn control_get_viewport_rendering(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        self.control_in_viewport(params, |app, _| app.control_rendering_snapshot())
    }
}
