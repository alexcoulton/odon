//! Shared mount-based dispatcher for built-in single-view shell components.

use super::*;

impl OmeZarrViewerApp {
    pub(super) fn ui_viewer_top_bar(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        configuration: &serde_json::Value,
    ) {
        ui.horizontal(|ui| {
            let mut has_content = false;
            if shell_configuration_flag(configuration, "show_title") {
                let title = format!("OME-Zarr: {}", self.current_roi_compact_label());
                top_bar::ui_title(ui, title).on_hover_text(self.current_roi_hover_text());
                has_content = true;
            }
            let have_channels = !self.channels.is_empty();
            if shell_configuration_flag(configuration, "show_navigation") {
                top_bar_section(ui, &mut has_content);
                if top_bar::ui_fit(ui, "Fit (F)") {
                    self.fit_to_last_canvas();
                }
                let supported_view_planes = self.view_plane_modes();
                if supported_view_planes.len() > 1 {
                    ui.separator();
                    let mut mode = self.view_plane_mode;
                    if top_bar::ui_view_plane_mode(ui, &mut mode, &supported_view_planes) {
                        self.submit_native_active_viewport_plane(mode, None);
                    }
                }
                if let Some(slice_extent) =
                    self.view_slice_extent_level0().filter(|extent| *extent > 1)
                {
                    ui.separator();
                    let mut slice_level0 = self
                        .displayed_view_selection()
                        .slice_level0
                        .min(slice_extent.saturating_sub(1));
                    let slider = top_bar::ui_view_plane_slice(
                        ui,
                        self.view_plane_mode.slice_axis_label(),
                        &mut slice_level0,
                        slice_extent.saturating_sub(1),
                    );
                    if slider.changed && slider.dragging {
                        self.previous_displayed_view_selection =
                            Some(self.displayed_view_selection());
                        self.draft_view_slice_level0 = Some(slice_level0);
                        ctx.request_repaint();
                    } else if slider.changed {
                        self.previous_displayed_view_selection = None;
                        self.draft_view_slice_level0 = None;
                        self.submit_native_active_viewport_plane(
                            self.view_plane_mode,
                            Some(slice_level0),
                        );
                    } else if slider.released
                        && let Some(draft) = self.draft_view_slice_level0.take()
                    {
                        self.previous_displayed_view_selection = None;
                        self.submit_native_active_viewport_plane(self.view_plane_mode, Some(draft));
                    }
                }
                ui.separator();
                if let Some(step) = top_bar::ui_prev_next_channel(ui, have_channels) {
                    self.step_selected_channel_visibility(step);
                }
                if self.roi_selector_ui.has_multiple_rois() {
                    ui.separator();
                    if let Some(step) = top_bar::ui_prev_next_roi(ui, true)
                        && let Some(action) = self.roi_selector_ui.step_roi_action(step)
                    {
                        self.handle_roi_selector_action(ctx, action);
                    }
                }
            }
            if shell_configuration_flag(configuration, "show_panel_controls") {
                top_bar_section(ui, &mut has_content);
                let panels_before = (self.show_left_panel, self.show_right_panel);
                let (mut show_left_panel, mut show_right_panel) = panels_before;
                top_bar::ui_panel_toggles(ui, &mut show_left_panel, &mut show_right_panel);
                if panels_before != (show_left_panel, show_right_panel) {
                    self.native_command_ingress.push(NativeControlIntent {
                        method: "viewer.panels.set",
                        params: serde_json::json!({
                            "left": show_left_panel,
                            "right": show_right_panel,
                        }),
                    });
                }
            }
            if shell_configuration_flag(configuration, "show_viewport_controls") {
                top_bar_section(ui, &mut has_content);
                self.ui_viewport_controls(ui);
            }
            if shell_configuration_flag(configuration, "show_rendering_controls") {
                top_bar_section(ui, &mut has_content);
                let rendering_before = (
                    self.smooth_pixels,
                    self.show_tile_debug,
                    self.show_hud,
                    self.show_scale_bar,
                );
                let mut smooth_pixels = self.smooth_pixels;
                let mut show_tile_debug = self.show_tile_debug;
                let mut show_hud = self.show_hud;
                let mut show_scale_bar = self.show_scale_bar;
                top_bar::ui_smooth_toggle(ui, &mut smooth_pixels);
                ui.checkbox(&mut show_tile_debug, "Tile Debug");
                ui.checkbox(&mut show_hud, "HUD");
                ui.checkbox(&mut show_scale_bar, "Scale Bar");
                let rendering_after = (smooth_pixels, show_tile_debug, show_hud, show_scale_bar);
                if rendering_after != rendering_before {
                    self.submit_native_active_viewport_rendering(
                        smooth_pixels,
                        show_scale_bar,
                        show_hud,
                        show_tile_debug,
                    );
                }
            }
            if have_channels && shell_configuration_flag(configuration, "show_contrast_controls") {
                top_bar_section(ui, &mut has_content);
                self.ui_top_bar_quick_contrast(ui);
            }
        });
    }

    pub(super) fn ui_shell_builtin(&mut self, mount: &str, ui: &mut egui::Ui, ctx: &egui::Context) {
        match mount {
            "builtin:layers" => self.ui_layers(ui, ctx),
            "builtin:channels" => {
                let channel_search_before = self.channel_list_search.clone();
                channels_panel::show(self, ui, ctx);
                if self.channel_list_search != channel_search_before {
                    self.native_command_ingress.push(NativeControlIntent {
                        method: "viewer.channels.presentation.set",
                        params: serde_json::json!({"search":self.channel_list_search}),
                    });
                }
            }
            "builtin:project" => {
                let cur = self.dataset.source.local_path().map(Path::to_path_buf);
                if let Some(action) = self.project_space.ui(ui, cur.as_deref()) {
                    self.handle_project_space_action(action);
                }
            }
            "builtin:properties" => self.ui_layer_properties(ui, ctx),
            "builtin:views" => {
                self.ui_help_heading(ui, "Views", crate::ui::help::HelpTopic::ViewsPanel);
                ui.separator();
                let roi = self.current_project_roi().cloned();
                if let Some(action) = self.project_space.ui_views_panel(ui, roi, true) {
                    self.handle_project_space_action(action);
                }
            }
            "builtin:analysis" => self.ui_shell_analysis(ui),
            "builtin:measurements" => self.ui_shell_measurements(ui),
            "builtin:memory" => {
                self.ui_help_heading(ui, "Memory", crate::ui::help::HelpTopic::MemoryPanel);
                ui.separator();
                self.ui_memory(ui);
            }
            "builtin:roi-selector" => {
                self.ui_help_heading(
                    ui,
                    "ROI Selector",
                    crate::ui::help::HelpTopic::RoiSelectorPanel,
                );
                ui.separator();
                if let Some(action) = self.roi_selector_ui.ui(ui) {
                    self.handle_roi_selector_action(ctx, action);
                }
            }
            "builtin:recovery-controls" => {
                if crate::ui::shell_recovery::render(ui) {
                    self.native_command_ingress.push(NativeControlIntent {
                        method: "ui.shell.recover",
                        params: serde_json::json!({}),
                    });
                }
            }
            _ => {
                ui.colored_label(
                    ui.visuals().error_fg_color,
                    format!("Unavailable built-in shell mount: {mount}"),
                );
            }
        }
    }

    fn ui_shell_analysis(&mut self, ui: &mut egui::Ui) {
        self.switch_to_pan_if_analysis_interacted(ui);
        self.ui_help_heading(ui, "Analysis", crate::ui::help::HelpTopic::AnalysisPanel);
        ui.separator();
        let suspend_live_selection_sync =
            matches!(self.tool_mode, ToolMode::Select | ToolMode::LassoSelect);
        if self.seg_objects.object_count() > 0 {
            let analysis_before = self.seg_objects.project_analysis_state();
            let object_action = self.seg_objects.ui_analysis(
                ui,
                &self.dataset,
                self.store.clone(),
                &self.channels,
                self.selected_channel,
                suspend_live_selection_sync,
                self.seg_objects_offset_world,
                self.spatial_root.as_deref(),
                self.spatial_layers.table_elements(),
            );
            let analysis_after = self.seg_objects.project_analysis_state();
            if analysis_before != analysis_after {
                let active_channel = self
                    .channels
                    .get(self.selected_channel)
                    .map(|channel| channel.name.clone());
                self.seg_objects
                    .apply_project_analysis_state(&analysis_before, active_channel.as_deref());
                self.native_command_ingress.push(NativeControlIntent {
                    method: "viewer.analysis.set",
                    params: serde_json::json!({
                        "target":"segmentation_objects",
                        "state":analysis_after,
                    }),
                });
            }
            if let Some(action) = object_action {
                self.queue_object_ui_action(action);
            }
            if let Some(idx) = self.seg_objects.take_pending_zoom_object_index() {
                self.fit_to_seg_object_index(idx);
            }
        } else if let LayerId::SpatialShape(id) = self.active_layer {
            let spatial_tables = self.spatial_layers.table_elements().to_vec();
            let active_channel = self
                .channels
                .get(self.selected_channel)
                .map(|channel| channel.name.clone());
            let mut object_action = None;
            let mut analysis_after = None;
            let mut fit_world = None;
            if let Some(layer) = self.spatial_layers.shapes.iter_mut().find(|s| s.id == id) {
                let offset_world = layer.offset_world;
                if let Some(objects) = layer.object_layer_mut() {
                    let before = objects.project_analysis_state();
                    object_action = objects.ui_analysis(
                        ui,
                        &self.dataset,
                        self.store.clone(),
                        &self.channels,
                        self.selected_channel,
                        suspend_live_selection_sync,
                        offset_world,
                        self.spatial_root.as_deref(),
                        &spatial_tables,
                    );
                    let after = objects.project_analysis_state();
                    if before != after {
                        objects.apply_project_analysis_state(&before, active_channel.as_deref());
                        analysis_after = Some(after);
                    }
                    if let Some(idx) = objects.take_pending_zoom_object_index()
                        && let Some(viewport) = self.last_canvas_rect
                        && let Some(world) = objects.fit_object_bounds_world(idx, offset_world)
                    {
                        fit_world = Some((viewport, world));
                    }
                } else {
                    ui.heading("Analysis");
                    ui.label("Analysis is available for object-backed shape layers.");
                }
            } else {
                ui.heading("Analysis");
                ui.label("SpatialData shape layer not found.");
            }
            if let Some(after) = analysis_after {
                self.native_command_ingress.push(NativeControlIntent {
                    method: "viewer.analysis.set",
                    params: serde_json::json!({
                        "target":"spatial_shape",
                        "layer_id":id,
                        "state":after,
                    }),
                });
            }
            if let Some(action) = object_action {
                self.queue_object_ui_action(action);
            }
            if let Some((viewport, world)) = fit_world {
                self.fit_camera_to_world_rect(viewport, world);
            }
        } else {
            ui.heading("Analysis");
            ui.label(
                "Analysis is available for loaded Segmentation Objects and object-backed SpatialData shape layers.",
            );
        }
    }

    fn ui_shell_measurements(&mut self, ui: &mut egui::Ui) {
        self.ui_help_heading(
            ui,
            "Measurements",
            crate::ui::help::HelpTopic::MeasurementsPanel,
        );
        ui.separator();
        if self.seg_objects.object_count() > 0 {
            let actions = self.seg_objects.ui_measurements(
                ui,
                &self.dataset,
                self.store.clone(),
                &self.channels,
                self.seg_objects_offset_world,
                self.control_actor_measurement_generation > 0,
            );
            for action in actions {
                let (method, params) = match action {
                    crate::objects::MeasurementUiAction::Configure(params) => {
                        ("viewer.measurements.configure", params)
                    }
                    crate::objects::MeasurementUiAction::Start(params) => {
                        ("viewer.measurements.start", params)
                    }
                    crate::objects::MeasurementUiAction::Cancel => {
                        ("viewer.measurements.cancel", serde_json::json!({}))
                    }
                };
                self.native_command_ingress
                    .push(NativeControlIntent { method, params });
            }
        } else if let LayerId::SpatialShape(id) = self.active_layer {
            if let Some(layer) = self.spatial_layers.shapes.iter_mut().find(|s| s.id == id) {
                let offset_world = layer.offset_world;
                if let Some(objects) = layer.object_layer_mut() {
                    let _ = objects.ui_measurements(
                        ui,
                        &self.dataset,
                        self.store.clone(),
                        &self.channels,
                        offset_world,
                        false,
                    );
                } else {
                    ui.heading("Measurements");
                    ui.label("Measurements are available for object-backed shape layers.");
                }
            } else {
                ui.heading("Measurements");
                ui.label("SpatialData shape layer not found.");
            }
        } else {
            ui.heading("Measurements");
            ui.label(
                "Measurements are available for loaded Segmentation Objects and object-backed SpatialData shape layers.",
            );
        }
    }
}

fn shell_configuration_flag(configuration: &serde_json::Value, name: &str) -> bool {
    configuration
        .get(name)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true)
}

fn top_bar_section(ui: &mut egui::Ui, has_content: &mut bool) {
    if *has_content {
        ui.separator();
    }
    *has_content = true;
}
