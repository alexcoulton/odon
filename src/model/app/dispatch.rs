//! Registry-facing command dispatch into typed AppModel domain methods.

use super::*;

impl AppModel {
    pub fn dispatch(
        &mut self,
        method: &str,
        params: &Value,
    ) -> Option<Result<ModelDispatch, ControlError>> {
        let supported = matches!(
            method,
            "app.get_state"
                | "app.settings.get"
                | "app.recent_projects.list"
                | "app.lifecycle.get"
                | "deep_links.parse"
                | "deep_links.filters.get"
                | "deep_links.generate"
                | "app.get_loading_state"
                | "get_loading_state"
                | "app.get_method_availability"
                | "app.navigation.show_project"
                | "project.rois.list"
                | "project.get"
                | "project.create"
                | "project.update_metadata"
                | "project.rois.get"
                | "project.rois.add"
                | "project.rois.update"
                | "project.rois.remove"
                | "project.rois.reorder"
                | "project.rois.get_selection"
                | "project.rois.select"
                | "project.rois.focus"
                | "project.rois.next"
                | "project.rois.previous"
                | "project.views.list"
                | "project.views.get"
                | "project.views.create"
                | "project.views.rename"
                | "project.views.delete"
                | "project.views.capture"
                | "project.views.apply"
                | "viewer.channels.list"
                | "viewer.channels.list_visible"
                | "viewer.channels.get_active"
                | "viewer.channels.set_active"
                | "viewer.channels.set_visible"
                | "viewer.channels.get_contrast"
                | "viewer.channels.set_contrast"
                | "viewer.channels.set_color"
                | "viewer.channels.set_note"
                | "viewer.channels.get_transform"
                | "viewer.channels.set_transform"
                | "viewer.channels.reset_transform"
                | "viewer.channels.set_order"
                | "viewer.channels.presentation.get"
                | "viewer.channels.presentation.set"
                | "viewer.channels.list_groups"
                | "viewer.channels.set_group"
                | "viewer.camera.get"
                | "viewer.camera.set"
                | "viewer.camera.zoom_in"
                | "viewer.camera.zoom_out"
                | "viewer.camera.fit"
                | "viewer.planes.get"
                | "viewer.planes.set"
                | "viewer.planes.next"
                | "viewer.planes.previous"
                | "viewer.planes.operation_availability"
                | "viewer.rendering.get_smooth_pixels"
                | "viewer.rendering.set_smooth_pixels"
                | "viewer.rendering.get_state"
                | "viewer.scale_bar.get"
                | "viewer.scale_bar.set"
                | "viewer.screenshot.settings.get"
                | "viewer.screenshot.settings.set"
                | "memory.tiles.get"
                | "memory.tiles.set"
                | "memory.get"
                | "memory.pin"
                | "memory.unpin"
                | "memory.unpin_all"
                | "viewer.panels.get"
                | "viewer.panels.set"
                | "viewer.ui.set_left_tab"
                | "viewer.ui.set_right_tab"
                | "viewer.workspace.get"
                | "viewer.viewports.list"
                | "viewer.workspace.layout.get"
                | "viewer.workspace.layout.set"
                | "viewer.workspace.swap"
                | "viewer.viewports.get"
                | "viewer.viewports.create"
                | "viewer.viewports.clone"
                | "viewer.viewports.rename"
                | "viewer.viewports.remove"
                | "viewer.viewports.set_active"
                | "viewer.viewport_links.get"
                | "viewer.viewport_links.list"
                | "viewer.viewport_links.set"
                | "viewer.viewport_links.create"
                | "viewer.viewport_links.update"
                | "viewer.viewport_links.remove"
                | "viewer.viewports.camera.get"
                | "viewer.viewports.camera.set"
                | "viewer.viewports.camera.fit"
                | "viewer.viewports.planes.get"
                | "viewer.viewports.planes.set"
                | "viewer.viewports.channels.get"
                | "viewer.viewports.channels.set_visible"
                | "viewer.viewports.channels.set"
                | "viewer.viewports.channels.set_active"
                | "viewer.viewports.channels.set_color"
                | "viewer.viewports.channels.set_contrast"
                | "viewer.viewports.channels.set_order"
                | "viewer.viewports.channels.list_groups"
                | "viewer.viewports.channels.set_group"
                | "viewer.viewports.objects.style.get"
                | "viewer.viewports.objects.style.set"
                | "viewer.viewports.objects.legend.set"
                | "viewer.viewports.objects.filter.get"
                | "viewer.viewports.objects.filter.set"
                | "viewer.viewports.objects.filter.clear"
                | "viewer.viewports.layers.list"
                | "viewer.viewports.layers.get"
                | "viewer.viewports.layers.set"
                | "viewer.viewports.layers.set_visibility"
                | "viewer.viewports.layers.set_order"
                | "viewer.viewports.layers.set_active"
                | "viewer.viewports.layers.state.replace"
                | "viewer.objects.get_state"
                | "viewer.objects.get_visibility"
                | "viewer.objects.set_visibility"
                | "viewer.objects.style.get"
                | "viewer.objects.style.set"
                | "viewer.objects.legend.set"
                | "viewer.objects.rendering.get_fast"
                | "viewer.objects.rendering.set_fast"
                | "viewer.objects.source.clear"
                | "viewer.objects.source.cancel_load"
                | "viewer.objects.properties.list"
                | "viewer.objects.properties.load"
                | "viewer.objects.properties.values"
                | "viewer.objects.get_selection"
                | "viewer.objects.query_rect"
                | "viewer.objects.query_view"
                | "viewer.objects.query_lasso"
                | "viewer.objects.select_rect"
                | "viewer.objects.select_lasso"
                | "viewer.objects.clear_selection"
                | "viewer.objects.selection.select_ids"
                | "viewer.objects.selection.select_filtered"
                | "viewer.objects.focus.set"
                | "viewer.objects.focus.clear"
                | "viewer.objects.selection.state.replace"
                | "viewer.objects.get_filter"
                | "viewer.objects.set_filter"
                | "viewer.objects.clear_filter"
                | "viewer.objects.filters.set_model"
                | "viewer.objects.filters.get_revision"
                | "viewer.labels.list"
                | "viewer.labels.get"
                | "viewer.labels.load"
                | "viewer.labels.unload"
                | "viewer.labels.set_visibility"
                | "viewer.thresholds.levels.list"
                | "viewer.thresholds.preview.get"
                | "viewer.thresholds.preview.configure"
                | "viewer.thresholds.preview.start"
                | "viewer.thresholds.preview.refresh"
                | "viewer.thresholds.preview.apply"
                | "viewer.thresholds.preview.cancel"
                | "viewer.analysis.get"
                | "viewer.analysis.set"
                | "viewer.analysis.histogram"
                | "viewer.analysis.suggest_thresholds"
                | "viewer.analysis.warmup.get"
                | "viewer.analysis.warmup.start"
                | "viewer.analysis.presets.import"
                | "viewer.analysis.presets.export"
                | "viewer.measurements.get"
                | "viewer.measurements.configure"
                | "viewer.measurements.start"
                | "viewer.measurements.cancel"
                | "viewer.measurements.properties.list"
                | "exports.objects.columns"
                | "exports.objects.get_state"
                | "exports.objects.start"
                | "exports.objects.export_csv"
                | "exports.objects.export_geoparquet"
                | "viewer.native_layers.list"
                | "viewer.native_layers.get"
                | "viewer.native_layers.set_active"
                | "viewer.native_layers.set_visibility"
                | "viewer.native_layers.set_order"
                | "viewer.native_layers.set_offset"
                | "viewer.native_layers.reset_offset"
                | "viewer.masks.layers.list"
                | "viewer.masks.layers.get"
                | "viewer.masks.layers.create"
                | "viewer.masks.layers.update"
                | "viewer.masks.layers.delete"
                | "viewer.masks.polygons.list"
                | "viewer.masks.polygons.add"
                | "viewer.masks.polygons.update"
                | "viewer.masks.polygons.remove"
                | "viewer.masks.selection.get"
                | "viewer.masks.selection.set"
                | "viewer.masks.selection.clear"
                | "viewer.masks.undo"
                | "viewer.masks.state.replace"
                | "viewer.masks.persistence.get"
                | "viewer.masks.persistence.sync"
                | "viewer.viewports.rendering.get"
                | "viewer.viewports.rendering.set"
                | "mosaic.ui.set_left_tab"
                | "mosaic.ui.set_right_tab"
                | "mosaic.rendering.set"
                | "mosaic.layout.configure"
                | "mosaic.get_state"
                | "mosaic.items.list"
                | "mosaic.selection.get"
                | "mosaic.selection.set"
                | "mosaic.selection.clear"
                | "mosaic.focus.get"
                | "mosaic.focus.set"
                | "mosaic.focus.next"
                | "mosaic.focus.previous"
                | "mosaic.focus.fit"
                | "mosaic.focus.clear"
                | "mosaic.fit_all"
                | "mosaic.objects.get_state"
        );
        if !supported {
            return None;
        }
        if matches!(method, "app.get_loading_state" | "get_loading_state") {
            return Some(Ok(ModelDispatch {
                response: self.loading_state(),
                present: false,
            }));
        }
        if method == "app.get_state" {
            if self.mode == ModelMode::Mosaic {
                return Some(self.mosaic.snapshot().map(|mosaic| ModelDispatch {
                    response: json!({
                        "mode":"mosaic",
                        "view":{
                            "roi_count":mosaic["roi_count"],
                            "focused_roi":mosaic["focused"]["roi_id"],
                            "channel_count":self.mosaic.projection_state()["channels"]
                                .as_array()
                                .map_or(0, Vec::len),
                        },
                        "mosaic":mosaic,
                        "project":self.project.rois_json(),
                    }),
                    present: false,
                }));
            }
            return Some(Ok(ModelDispatch {
                response: self.application_state(),
                present: false,
            }));
        }
        if method == "app.settings.get" {
            return Some(Ok(ModelDispatch {
                response: self.settings_snapshot(),
                present: false,
            }));
        }
        if method == "app.recent_projects.list" {
            return Some(Ok(ModelDispatch {
                response: self.recent_projects_snapshot(),
                present: false,
            }));
        }
        if method == "app.lifecycle.get" {
            return Some(Ok(ModelDispatch {
                response: self.lifecycle_state(),
                present: false,
            }));
        }
        if method == "deep_links.parse" {
            return Some(Self::parse_deep_link(params).map(|response| ModelDispatch {
                response,
                present: false,
            }));
        }
        if method == "deep_links.filters.get" {
            return Some(
                Self::deep_link_filters(params).map(|response| ModelDispatch {
                    response,
                    present: false,
                }),
            );
        }
        if method == "deep_links.generate" {
            return Some(
                self.generate_deep_link(params)
                    .map(|response| ModelDispatch {
                        response,
                        present: false,
                    }),
            );
        }
        if method == "app.navigation.show_project" {
            return Some(self.show_project().map(|response| ModelDispatch {
                response,
                present: true,
            }));
        }
        if method == "app.get_method_availability" {
            let requested = params
                .get("methods")
                .and_then(Value::as_array)
                .map(|methods| {
                    methods
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect::<Vec<_>>()
                });
            return Some(Ok(ModelDispatch {
                response: crate::control::registry::availability_catalog(
                    self.mode.as_str(),
                    requested.as_deref(),
                ),
                present: false,
            }));
        }
        if method == "viewer.screenshot.settings.get" {
            return Some(
                self.screenshot_settings_snapshot()
                    .map(|response| ModelDispatch {
                        response,
                        present: false,
                    }),
            );
        }
        if method.starts_with("mosaic.") {
            if self.mode != ModelMode::Mosaic {
                return Some(Err(ControlError::new(
                    ControlErrorKind::WrongMode,
                    "No mosaic viewer is currently open",
                )));
            }
            let response = self.mosaic.dispatch(method, params).unwrap_or_else(|| {
                Err(ControlError::new(
                    ControlErrorKind::MethodNotFound,
                    format!("unsupported mosaic model method '{method}'"),
                ))
            });
            let response = response.map(|result| {
                let response = match method {
                    "mosaic.ui.set_left_tab" | "mosaic.ui.set_right_tab" => {
                        json!({"mode":"mosaic","tab":result})
                    }
                    "mosaic.rendering.set" => json!({"mode":"mosaic","result":result}),
                    "mosaic.layout.configure" => json!({"mode":"mosaic","layout":result}),
                    "mosaic.get_state" => json!({"mode":"mosaic","mosaic":result}),
                    "mosaic.items.list" => json!({"mode":"mosaic","result":result}),
                    "mosaic.selection.get" | "mosaic.selection.set" | "mosaic.selection.clear" => {
                        json!({"mode":"mosaic","selection":result})
                    }
                    "mosaic.focus.get" => json!({"mode":"mosaic","focused":result}),
                    "mosaic.objects.get_state" => json!({"mode":"mosaic","objects":result}),
                    _ => json!({"mode":"mosaic","result":result}),
                };
                ModelDispatch {
                    response,
                    present: !matches!(
                        method,
                        "mosaic.get_state"
                            | "mosaic.items.list"
                            | "mosaic.selection.get"
                            | "mosaic.focus.get"
                            | "mosaic.objects.get_state"
                    ),
                }
            });
            return Some(response);
        }
        if matches!(method, "project.views.capture" | "project.views.apply")
            && self.project_operation_pending
        {
            return Some(Err(ControlError::new(
                ControlErrorKind::NotReady,
                format!("{method} cannot run while a project persistence transaction is active"),
            )));
        }
        if is_project_model_method(method) {
            if self.project_operation_pending {
                return Some(Err(ControlError::new(
                    ControlErrorKind::NotReady,
                    format!(
                        "{method} cannot run while a project persistence transaction is active"
                    ),
                )
                .with_data(json!({
                    "method": method,
                    "required_readiness": ["project"],
                    "loading": self.loading_state()["loading"],
                }))));
            }
            if self.mode == ModelMode::Transition {
                return Some(Err(ControlError::new(
                    ControlErrorKind::NotReady,
                    format!("{method} requires the project model to leave transition state"),
                )
                .with_data(json!({
                    "method": method,
                    "required_readiness": ["model"],
                    "loading": self.loading_state()["loading"],
                }))));
            }
            let result = self.project.dispatch(method, params);
            if result.is_ok()
                && !matches!(
                    method,
                    "project.rois.list"
                        | "project.get"
                        | "project.rois.get"
                        | "project.rois.get_selection"
                        | "project.views.list"
                        | "project.views.get"
                )
            {
                self.project_initialized = true;
            }
            if result.is_ok() && method == "project.create" {
                self.set_mode(ModelMode::Project);
            }
            let present = !matches!(
                method,
                "project.rois.list"
                    | "project.get"
                    | "project.rois.get"
                    | "project.rois.get_selection"
                    | "project.views.list"
                    | "project.views.get"
            );
            return Some(result.map(|response| ModelDispatch { response, present }));
        }
        if self.mode == ModelMode::Mosaic {
            if let Some(result) = self.mosaic.dispatch_shared(method, params) {
                if result.is_ok() && matches!(method, "memory.unpin" | "memory.unpin_all") {
                    self.readiness.cancel_kind_pending(
                        OperationKind::MemoryPin,
                        "Mosaic pinned memory was unloaded",
                    );
                }
                return Some(result.map(|(response, present)| ModelDispatch { response, present }));
            }
        }
        if matches!(self.mode, ModelMode::Project | ModelMode::Mosaic) {
            return None;
        }
        if self.mode == ModelMode::Transition
            && !matches!(method, "app.get_loading_state" | "get_loading_state")
        {
            return Some(Err(ControlError::new(
                ControlErrorKind::NotReady,
                format!("{method} requires the dataset open to reach model/resource readiness"),
            )
            .with_data(json!({
                "method": method,
                "required_readiness": ["model", "resources"],
                "loading": self.loading_state()["loading"],
            }))));
        }
        if let Err(error) = self.check_viewport_revision(params) {
            return Some(Err(error));
        }
        let result = (|| -> Result<Value, ControlError> {
            Ok(match method {
                "app.get_loading_state" | "get_loading_state" | "app.get_method_availability" => {
                    unreachable!("mode-independent queries return before single-view dispatch")
                }
                "viewer.channels.list" => self.channels_snapshot()?,
                "viewer.channels.list_visible" => self.visible_channels_snapshot()?,
                "viewer.channels.get_active" => self.active_channel_snapshot()?,
                "viewer.channels.set_active" => self.set_active_channel_global(params)?,
                "viewer.channels.set_visible" => self.set_visible_channels_global(params)?,
                "viewer.channels.get_contrast" => self.get_channel_contrast_global(params)?,
                "viewer.channels.set_contrast" => self.set_channel_contrast_global(params)?,
                "viewer.channels.set_color" => self.set_channel_color_global(params)?,
                "viewer.channels.set_note" => self.set_channel_note_global(params)?,
                "viewer.channels.get_transform" => self.get_channel_transform(params)?,
                "viewer.channels.set_transform" => self.set_channel_transform(params)?,
                "viewer.channels.reset_transform" => self.reset_channel_transform(params)?,
                "viewer.channels.set_order" => self.set_channel_order_global(params)?,
                "viewer.channels.presentation.get" => self.channel_presentation_global()?,
                "viewer.channels.presentation.set" => {
                    self.set_channel_presentation_global(params)?
                }
                "viewer.channels.list_groups" => self.channel_groups_global()?,
                "viewer.channels.set_group" => self.set_channel_group_global(params)?,
                "viewer.camera.get" => self.get_camera_global()?,
                "viewer.camera.set" => self.set_camera_global(params)?,
                "viewer.camera.zoom_in" => self.zoom_camera_global(params, true)?,
                "viewer.camera.zoom_out" => self.zoom_camera_global(params, false)?,
                "viewer.camera.fit" => self.fit_camera_global()?,
                "viewer.planes.get" => self.get_plane_global()?,
                "viewer.planes.set" => self.set_plane_global(params)?,
                "viewer.planes.next" => self.step_plane_global(params, true)?,
                "viewer.planes.previous" => self.step_plane_global(params, false)?,
                "viewer.planes.operation_availability" => self.plane_operation_availability()?,
                "viewer.rendering.get_smooth_pixels" => self.get_smooth_pixels_global()?,
                "viewer.rendering.set_smooth_pixels" => self.set_smooth_pixels_global(params)?,
                "viewer.rendering.get_state" => self.rendering_state()?,
                "viewer.scale_bar.get" => self.get_scale_bar_global()?,
                "viewer.scale_bar.set" => self.set_scale_bar_global(params)?,
                "viewer.screenshot.settings.get" => self.screenshot_settings_snapshot()?,
                "viewer.screenshot.settings.set" => {
                    unreachable!("screenshot settings updates use the bounded worker dispatcher")
                }
                "memory.tiles.get" => self.tile_loading_snapshot()?,
                "memory.tiles.set" => self.set_tile_loading_policy(params)?,
                "memory.get" => self.memory_snapshot()?,
                "memory.pin" => {
                    unreachable!("memory pinning uses the bounded worker dispatcher")
                }
                "memory.unpin" => self.unpin_memory(params)?,
                "memory.unpin_all" => self.unpin_all_memory()?,
                "viewer.panels.get" => self.get_panels()?,
                "viewer.panels.set" => self.set_panels(params)?,
                "viewer.ui.set_left_tab" => self.set_left_tab(params)?,
                "viewer.ui.set_right_tab" => self.set_right_tab(params)?,
                "project.views.capture" => self.capture_project_view(params)?,
                "project.views.apply" => self.apply_project_view(params)?,
                "viewer.workspace.get" | "viewer.viewports.list" => self.workspace_snapshot()?,
                "viewer.workspace.layout.get" => self.layout_snapshot()?,
                "viewer.workspace.layout.set" => self.set_layout(params)?,
                "viewer.workspace.swap" => self.swap_viewports()?,
                "viewer.viewports.get" => self.viewport_snapshot_for(params)?,
                "viewer.viewports.create" | "viewer.viewports.clone" => {
                    self.create_viewport(params)?
                }
                "viewer.viewports.rename" => self.rename_viewport(params)?,
                "viewer.viewports.remove" => self.remove_viewport(params)?,
                "viewer.viewports.set_active" => self.set_active_viewport(params)?,
                "viewer.viewport_links.get" => self.links_snapshot()?,
                "viewer.viewport_links.list" => self.link_groups_snapshot()?,
                "viewer.viewport_links.set" => self.set_links(params, LinkRequestKind::Direct)?,
                "viewer.viewport_links.create" => {
                    self.set_links(params, LinkRequestKind::Create)?
                }
                "viewer.viewport_links.update" => {
                    self.set_links(params, LinkRequestKind::Update)?
                }
                "viewer.viewport_links.remove" => self.remove_links(params)?,
                "viewer.viewports.camera.get" => self.get_camera(params)?,
                "viewer.viewports.camera.set" => self.set_camera(params)?,
                "viewer.viewports.camera.fit" => self.fit_viewport(params)?,
                "viewer.viewports.planes.get" => self.get_plane(params)?,
                "viewer.viewports.planes.set" => self.set_plane(params)?,
                "viewer.viewports.channels.get" => self.get_viewport_channels(params)?,
                "viewer.viewports.channels.set_visible" | "viewer.viewports.channels.set" => {
                    self.set_visible_channels(params)?
                }
                "viewer.viewports.channels.set_active" => self.set_active_channel(params)?,
                "viewer.viewports.channels.set_color" => self.set_channel_color(params)?,
                "viewer.viewports.channels.set_contrast" => self.set_channel_contrast(params)?,
                "viewer.viewports.channels.set_order" => self.set_channel_order(params)?,
                "viewer.viewports.channels.list_groups" => self.channel_groups(params)?,
                "viewer.viewports.channels.set_group" => self.set_channel_group(params)?,
                "viewer.viewports.objects.style.get" => self.get_object_style(params)?,
                "viewer.viewports.objects.style.set" => self.set_object_style(params)?,
                "viewer.viewports.objects.legend.set" => self.set_object_legend(params)?,
                "viewer.viewports.objects.filter.get" => self.get_object_filter(params)?,
                "viewer.viewports.objects.filter.set" => {
                    unreachable!("filter evaluation is dispatched to a resource worker")
                }
                "viewer.viewports.objects.filter.clear" => self.clear_object_filter(params)?,
                "viewer.viewports.layers.list" => self.native_layers_for(params)?,
                "viewer.viewports.layers.get" => self.native_layer_for(params)?,
                "viewer.viewports.layers.set" => self.set_native_layer_presentation(params)?,
                "viewer.viewports.layers.set_visibility" => {
                    self.set_native_layer_visibility(params)?
                }
                "viewer.viewports.layers.set_order" => self.set_native_layer_order(params)?,
                "viewer.viewports.layers.set_active" => self.set_native_layer_active(params)?,
                "viewer.viewports.layers.state.replace" => self.replace_native_layers(params)?,
                "viewer.objects.get_state" => json!({
                    "target": "segmentation_objects",
                    "state": self.object_resource_state(),
                }),
                "viewer.objects.get_visibility" => self.object_overlay_visibility_global(params)?,
                "viewer.objects.set_visibility" => {
                    self.set_object_overlay_visibility_global(params)?
                }
                "viewer.objects.style.get" => self.get_object_style_global(params)?,
                "viewer.objects.style.set" => self.set_object_style_global(params)?,
                "viewer.objects.legend.set" => self.set_object_legend_global(params)?,
                "viewer.objects.rendering.get_fast" => {
                    self.get_fast_object_rendering_global(params)?
                }
                "viewer.objects.rendering.set_fast" => {
                    self.set_fast_object_rendering_global(params)?
                }
                "viewer.objects.source.clear" => self.clear_object_resource()?,
                "viewer.objects.source.cancel_load" => self.cancel_object_resource_load(),
                "viewer.objects.properties.list" => self.object_properties_list(params)?,
                "viewer.objects.properties.load" => self.object_property_load(params)?,
                "viewer.objects.properties.values" => self.object_property_values(params)?,
                "viewer.objects.get_selection" => json!({
                    "mode":"single",
                    "objects":self.object_selection_get(params)?,
                }),
                "viewer.objects.query_rect" => json!({
                    "mode":"single",
                    "objects":self.object_selection_query_rect(params)?,
                }),
                "viewer.objects.query_view" => json!({
                    "mode":"single",
                    "objects":self.object_selection_query_view(params)?,
                }),
                "viewer.objects.query_lasso" => self.object_selection_query_lasso(params)?,
                "viewer.objects.select_rect" => json!({
                    "mode":"single",
                    "objects":self.object_selection_select_rect(params)?,
                }),
                "viewer.objects.select_lasso" => self.object_selection_select_lasso(params)?,
                "viewer.objects.clear_selection" => json!({
                    "mode":"single",
                    "objects":self.object_selection_clear(params)?,
                }),
                "viewer.objects.selection.select_ids" => {
                    self.object_selection_select_ids(params)?
                }
                "viewer.objects.selection.select_filtered" => {
                    self.object_selection_select_filtered(params)?
                }
                "viewer.objects.focus.set" => self.object_selection_focus(params)?,
                "viewer.objects.focus.clear" => self.object_selection_clear_focus(params)?,
                "viewer.objects.selection.state.replace" => {
                    self.object_selection_replace(params)?
                }
                "viewer.objects.get_filter" | "viewer.objects.filters.get_revision" => {
                    self.get_object_filter_global(params)?
                }
                "viewer.objects.set_filter" | "viewer.objects.filters.set_model" => {
                    unreachable!("filter evaluation is dispatched to a resource worker")
                }
                "viewer.objects.clear_filter" => self.clear_object_filter_global(params)?,
                "viewer.labels.list" | "viewer.labels.get" => self.labels_snapshot()?,
                "viewer.labels.load" => {
                    unreachable!("label loading is dispatched to a resource worker")
                }
                "viewer.labels.unload" => self.unload_labels()?,
                "viewer.labels.set_visibility" => self.set_labels_visibility(params)?,
                "viewer.thresholds.levels.list" => self.threshold_levels()?,
                "viewer.thresholds.preview.get" => self.threshold_preview_snapshot()?,
                "viewer.thresholds.preview.configure"
                | "viewer.thresholds.preview.start"
                | "viewer.thresholds.preview.refresh"
                | "viewer.thresholds.preview.apply" => {
                    unreachable!("threshold work uses the bounded worker dispatcher")
                }
                "viewer.thresholds.preview.cancel" => self.cancel_threshold_preview()?,
                "viewer.analysis.get" => self.analysis_snapshot(params)?,
                "viewer.analysis.set" => self.set_analysis_state(params)?,
                "viewer.analysis.warmup.get" => self.analysis_warmup_snapshot(params)?,
                "viewer.analysis.histogram"
                | "viewer.analysis.suggest_thresholds"
                | "viewer.analysis.warmup.start"
                | "viewer.analysis.presets.import"
                | "viewer.analysis.presets.export" => {
                    unreachable!("analysis work uses the bounded worker dispatcher")
                }
                "viewer.measurements.get" | "viewer.measurements.properties.list" => {
                    self.measurement_snapshot(params)?
                }
                "viewer.measurements.configure" => self.configure_measurement(params)?,
                "viewer.measurements.start" => {
                    unreachable!("measurement work uses the bounded worker dispatcher")
                }
                "viewer.measurements.cancel" => self.cancel_measurement(params)?,
                "exports.objects.columns" => self.object_export_columns_snapshot(params)?,
                "exports.objects.get_state" => self.object_export_snapshot(params)?,
                "exports.objects.start"
                | "exports.objects.export_csv"
                | "exports.objects.export_geoparquet" => {
                    unreachable!("object export uses the bounded worker dispatcher")
                }
                "viewer.native_layers.list" => self.native_layers_global()?,
                "viewer.native_layers.get" => self.native_layer_global(params)?,
                "viewer.native_layers.set_active"
                | "viewer.native_layers.set_visibility"
                | "viewer.native_layers.set_order" => {
                    self.unwrap_native_global_result(method, params)?
                }
                "viewer.native_layers.set_offset" => {
                    self.set_native_layer_offset_global(params, false)?
                }
                "viewer.native_layers.reset_offset" => {
                    self.set_native_layer_offset_global(params, true)?
                }
                "viewer.masks.layers.list"
                | "viewer.masks.layers.get"
                | "viewer.masks.layers.create"
                | "viewer.masks.layers.update"
                | "viewer.masks.layers.delete"
                | "viewer.masks.polygons.list"
                | "viewer.masks.polygons.add"
                | "viewer.masks.polygons.update"
                | "viewer.masks.polygons.remove"
                | "viewer.masks.selection.get"
                | "viewer.masks.selection.set"
                | "viewer.masks.selection.clear"
                | "viewer.masks.undo"
                | "viewer.masks.state.replace" => {
                    let mut response = {
                        let dataset = self.dataset_mut()?;
                        let response = dataset.masks.dispatch(method, params)?;
                        if !matches!(
                            method,
                            "viewer.masks.layers.list"
                                | "viewer.masks.layers.get"
                                | "viewer.masks.polygons.list"
                                | "viewer.masks.selection.get"
                        ) {
                            Self::sync_mask_native_layers(dataset);
                        }
                        response
                    };
                    if params
                        .get("sync_project")
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
                        && !matches!(
                            method,
                            "viewer.masks.layers.list"
                                | "viewer.masks.layers.get"
                                | "viewer.masks.polygons.list"
                                | "viewer.masks.selection.get"
                                | "viewer.masks.selection.set"
                                | "viewer.masks.selection.clear"
                        )
                    {
                        let synced = self.sync_masks_to_project()?;
                        response["persistence"] = synced["persistence"].clone();
                    }
                    response
                }
                "viewer.masks.persistence.get" => self.mask_persistence_state()?,
                "viewer.masks.persistence.sync" => self.sync_masks_to_project()?,
                "viewer.viewports.rendering.get" => self.get_rendering(params)?,
                "viewer.viewports.rendering.set" => self.set_rendering(params)?,
                _ => unreachable!("supported method set and dispatch match diverged"),
            })
        })();
        let present = !matches!(
            method,
            "app.get_loading_state"
                | "get_loading_state"
                | "viewer.channels.list"
                | "viewer.channels.list_visible"
                | "viewer.channels.get_active"
                | "viewer.channels.get_contrast"
                | "viewer.channels.get_transform"
                | "viewer.channels.presentation.get"
                | "viewer.channels.list_groups"
                | "viewer.camera.get"
                | "viewer.planes.get"
                | "viewer.planes.operation_availability"
                | "viewer.rendering.get_smooth_pixels"
                | "viewer.rendering.get_state"
                | "viewer.scale_bar.get"
                | "viewer.screenshot.settings.get"
                | "memory.tiles.get"
                | "memory.get"
                | "viewer.panels.get"
                | "viewer.workspace.get"
                | "viewer.viewports.list"
                | "viewer.workspace.layout.get"
                | "viewer.viewports.get"
                | "viewer.viewport_links.get"
                | "viewer.viewport_links.list"
                | "viewer.viewports.camera.get"
                | "viewer.viewports.planes.get"
                | "viewer.viewports.channels.get"
                | "viewer.viewports.channels.list_groups"
                | "viewer.viewports.objects.style.get"
                | "viewer.viewports.objects.filter.get"
                | "viewer.viewports.layers.list"
                | "viewer.viewports.layers.get"
                | "viewer.objects.get_state"
                | "viewer.objects.get_visibility"
                | "viewer.objects.style.get"
                | "viewer.objects.rendering.get_fast"
                | "viewer.objects.properties.list"
                | "viewer.objects.properties.load"
                | "viewer.objects.properties.values"
                | "viewer.objects.get_selection"
                | "viewer.objects.query_rect"
                | "viewer.objects.query_view"
                | "viewer.objects.query_lasso"
                | "viewer.objects.get_filter"
                | "viewer.objects.filters.get_revision"
                | "viewer.labels.list"
                | "viewer.labels.get"
                | "viewer.thresholds.levels.list"
                | "viewer.thresholds.preview.get"
                | "viewer.analysis.get"
                | "viewer.analysis.warmup.get"
                | "viewer.measurements.get"
                | "viewer.measurements.properties.list"
                | "exports.objects.columns"
                | "exports.objects.get_state"
                | "viewer.native_layers.list"
                | "viewer.native_layers.get"
                | "viewer.masks.layers.list"
                | "viewer.masks.layers.get"
                | "viewer.masks.polygons.list"
                | "viewer.masks.selection.get"
                | "viewer.masks.persistence.get"
                | "viewer.viewports.rendering.get"
        );
        Some(result.map(|response| ModelDispatch { response, present }))
    }
}
