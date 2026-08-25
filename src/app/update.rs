use super::*;

impl eframe::App for OmeZarrViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Per-frame flow:
        // 1. absorb external input (drops, close requests, project-config changes)
        // 2. drain worker output and tick async subsystems into a consistent snapshot
        // 3. build chrome/panels
        // 4. draw the central canvas and overlays
        // 5. schedule the next repaint based on outstanding async work
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        if !dropped.is_empty() {
            let mut other_paths = Vec::new();
            for path in dropped.into_iter().filter_map(|f| f.path) {
                if crate::objects::ObjectsLayer::supports_source_path(&path) {
                    if let Some(path) = self.seg_objects.prepare_source_path(path) {
                        self.native_command_ingress.push(NativeControlIntent {
                            method: "viewer.objects.source.load",
                            params: serde_json::json!({
                                "path": path,
                                "downsample_factor": self.seg_objects.downsample_factor,
                            }),
                        });
                    }
                } else {
                    other_paths.push(path);
                }
            }
            if !other_paths.is_empty() {
                self.project_space.handle_dropped_paths(other_paths);
            }
        }

        self.ui_seg_label_prompt(ctx);
        self.rebuild_layer_orders();

        // Push project config updates into custom panels that depend on it.
        let cfg_gen = self.project_space.config_generation();
        if cfg_gen != self.control_actor_project_config_generation {
            self.control_actor_project_config_generation = cfg_gen;
            let cfg = self.project_space.config().clone();
            self.roi_selector_ui
                .set_project_config(cfg.clone(), &self.dataset.source);
            self.legacy_cell_threshold_points.set_project_config(cfg);
        }

        // Napari-like "close window" prompt.
        // - Cmd/Ctrl+W opens confirmation
        // - Cmd/Ctrl+W again confirms close
        if top_bar::handle_cmd_w_close(ctx, &mut self.close_dialog_open) {
            self.native_command_ingress.push(NativeControlIntent {
                method: "app.lifecycle.request_close",
                params: serde_json::json!({"save":"discard"}),
            });
        }
        // Use literal Ctrl+M here because Cmd+M is reserved for window minimize on macOS.
        if !ctx.wants_keyboard_input()
            && ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::M))
        {
            self.open_mapping_settings();
        }

        self.drain_tiles(ctx);
        self.drain_raw_tiles();
        self.drain_label_tiles();
        let seg_objects_was_loading = self.seg_objects.is_loading();
        self.seg_objects.tick();
        if seg_objects_was_loading
            && !self.seg_objects.is_loading()
            && self.seg_objects.object_count() > 0
        {
            self.restore_project_object_state_after_segmentation_load();
        }
        self.spatial_image_layers.tick();
        self.spatial_layers.tick();
        self.legacy_cell_threshold_points
            .tick(&mut self.cell_points);
        self.roi_selector_ui.tick();
        refresh_system_memory_if_needed(
            &mut self.system_memory,
            &mut self.system_memory_last_refresh,
            Duration::from_secs(2),
        );
        self.sync_analysis_follow_active_channel_state();
        let actor_shell_layout = self.control_shell_projection.get("layout").is_some();
        if !actor_shell_layout {
            let top_bar_visible = self.shell_node_visible("builtin:single.top-bar", true);
            egui::TopBottomPanel::top("top").show_animated(ctx, top_bar_visible, |ui| {
                self.ui_viewer_top_bar(ui, ctx, &serde_json::Value::Null);
            });
        }
        if let Some(action) = self.seg_objects.ui_load_dialog(ctx) {
            self.queue_object_ui_action(action);
        }

        if !actor_shell_layout {
            let left_tabs = self.shell_left_tabs();
            if self.show_left_panel && !left_tabs.is_empty() {
                let previous_tab = self.left_tab;
                let mut tab = self.left_tab;
                left_panel::show(
                    ctx,
                    "viewer-left",
                    &mut tab,
                    &left_tabs,
                    |ui, tab| match tab {
                        LeftTab::Layers => self.ui_layers(ui, ctx),
                        LeftTab::Project => {
                            let cur = self.dataset.source.local_path().map(Path::to_path_buf);
                            if let Some(action) = self.project_space.ui(ui, cur.as_deref()) {
                                self.handle_project_space_action(action);
                            }
                        }
                    },
                );
                if tab != previous_tab {
                    self.native_command_ingress.push(NativeControlIntent {
                        method: "viewer.ui.set_left_tab",
                        params: serde_json::json!({"tab": tab.storage_key()}),
                    });
                }
            }
            let right_tabs = self.shell_right_tabs();
            if self.show_right_panel && !right_tabs.is_empty() {
                let previous_tab = self.right_tab;
                let mut tab = self.right_tab;
                right_panel::show(
                    ctx,
                    "right",
                    self.shell_right_panel_width(380.0),
                    &mut tab,
                    &right_tabs,
                    |ui, tab| match tab {
                        RightTab::Properties => self.ui_layer_properties(ui, ctx),
                        RightTab::Views => {
                            self.ui_help_heading(
                                ui,
                                "Views",
                                crate::ui::help::HelpTopic::ViewsPanel,
                            );
                            ui.separator();
                            let roi = self.current_project_roi().cloned();
                            if let Some(action) = self.project_space.ui_views_panel(ui, roi, true) {
                                self.handle_project_space_action(action);
                            }
                        }
                        RightTab::Analysis => {
                            self.switch_to_pan_if_analysis_interacted(ui);
                            self.ui_help_heading(
                                ui,
                                "Analysis",
                                crate::ui::help::HelpTopic::AnalysisPanel,
                            );
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
                                    self.seg_objects.apply_project_analysis_state(
                                        &analysis_before,
                                        active_channel.as_deref(),
                                    );
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
                                if let Some(idx) = self.seg_objects.take_pending_zoom_object_index()
                                {
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
                                if let Some(layer) =
                                    self.spatial_layers.shapes.iter_mut().find(|s| s.id == id)
                                {
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
                                            objects.apply_project_analysis_state(
                                                &before,
                                                active_channel.as_deref(),
                                            );
                                            analysis_after = Some(after);
                                        }
                                        if let Some(idx) = objects.take_pending_zoom_object_index()
                                        {
                                            if let Some(viewport) = self.last_canvas_rect
                                                && let Some(world) = objects
                                                    .fit_object_bounds_world(idx, offset_world)
                                            {
                                                fit_world = Some((viewport, world));
                                            }
                                        }
                                    } else {
                                        ui.heading("Analysis");
                                        ui.label(
                                            "Analysis is available for object-backed shape layers.",
                                        );
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
                        RightTab::Measurements => {
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
                                if let Some(layer) =
                                    self.spatial_layers.shapes.iter_mut().find(|s| s.id == id)
                                {
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
                                        ui.label(
                                    "Measurements are available for object-backed shape layers.",
                                );
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
                        RightTab::Memory => {
                            self.ui_help_heading(
                                ui,
                                "Memory",
                                crate::ui::help::HelpTopic::MemoryPanel,
                            );
                            ui.separator();
                            self.ui_memory(ui);
                        }
                        RightTab::RoiSelector => {
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
                    },
                );
                if tab != previous_tab {
                    self.native_command_ingress.push(NativeControlIntent {
                        method: "viewer.ui.set_right_tab",
                        params: serde_json::json!({"tab": tab.storage_key()}),
                    });
                }
            }
        }
        if let Some(action) = self.project_space.ui_floating_windows(ctx, true) {
            self.handle_project_space_action(action);
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            if !actor_shell_layout || !self.ui_actor_shell_tree(ui, ctx) {
                self.ui_viewport_workspace(ui, ctx);
            }
        });

        self.ui_group_layers_dialog(ctx);
        self.ui_memory_load_dialog(ctx);
        self.ui_object_export_dialogs(ctx);
        self.ui_mapping_settings_dialogs(ctx);
        self.ui_roi_info_window(ctx);
        self.ui_screenshot_settings_dialog(ctx);

        if top_bar::ui_close_dialog(ctx, &mut self.close_dialog_open) {
            self.native_command_ingress.push(NativeControlIntent {
                method: "app.lifecycle.request_close",
                params: serde_json::json!({"save":"discard"}),
            });
        }
        crate::ui::help::show_help_window(ctx, &mut self.active_help_topic);

        if ctx.input(|i| i.key_pressed(egui::Key::F)) {
            if self.active_layer == LayerId::SegmentationObjects
                && !self.fit_to_selected_seg_objects()
            {
                self.fit_to_last_canvas();
            } else if self.active_layer != LayerId::SegmentationObjects {
                self.fit_to_last_canvas();
            }
        }

        self.schedule_repaint(ctx);
    }
}
