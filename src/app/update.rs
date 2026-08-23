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
                        self.native_control_intents.push(NativeControlIntent {
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
        if cfg_gen != self.project_cfg_seen {
            self.project_cfg_seen = cfg_gen;
            let cfg = self.project_space.config().clone();
            self.roi_selector
                .set_project_config(cfg.clone(), &self.dataset.source);
            self.cell_thresholds.set_project_config(cfg);
        }

        // Napari-like "close window" prompt.
        // - Cmd/Ctrl+W opens confirmation
        // - Cmd/Ctrl+W again confirms close
        if top_bar::handle_cmd_w_close(ctx, &mut self.close_dialog_open) {
            self.native_control_intents.push(NativeControlIntent {
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
        self.drain_screenshots();
        self.drain_histogram();
        self.drain_channel_maxes();
        self.seg_geojson.tick();
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
        self.xenium_layers.tick();
        self.cell_thresholds.tick(&mut self.cell_points);
        self.roi_selector.tick();
        refresh_system_memory_if_needed(
            &mut self.system_memory,
            &mut self.system_memory_last_refresh,
            Duration::from_secs(2),
        );
        self.sync_analysis_follow_active_channel_state();
        let mut ann_changed = false;
        for layer in &mut self.annotation_layers {
            ann_changed |= layer.tick();
        }
        if ann_changed {
            self.bump_render_id();
            ctx.request_repaint();
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let title = format!("OME-Zarr: {}", self.current_roi_compact_label());
                top_bar::ui_title(ui, title).on_hover_text(self.current_roi_hover_text());
                ui.separator();
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
                let have_channels = !self.channels.is_empty();
                if let Some(step) = top_bar::ui_prev_next_channel(ui, have_channels) {
                    self.step_selected_channel_visibility(step);
                }
                if self.roi_selector.has_multiple_rois() {
                    ui.separator();
                    if let Some(step) = top_bar::ui_prev_next_roi(ui, true) {
                        if let Some(action) = self.roi_selector.step_roi_action(step) {
                            self.handle_roi_selector_action(ctx, action);
                        }
                    }
                }
                ui.separator();
                let panels_before = (self.show_left_panel, self.show_right_panel);
                let (mut show_left_panel, mut show_right_panel) = panels_before;
                top_bar::ui_panel_toggles(ui, &mut show_left_panel, &mut show_right_panel);
                if panels_before != (show_left_panel, show_right_panel) {
                    self.native_control_intents.push(NativeControlIntent {
                        method: "viewer.panels.set",
                        params: serde_json::json!({
                            "left": show_left_panel,
                            "right": show_right_panel,
                        }),
                    });
                }
                ui.separator();
                self.ui_viewport_controls(ui);

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

                if have_channels {
                    ui.separator();
                    self.ui_top_bar_quick_contrast(ui);
                }
            });
        });
        if let Some(action) = self.seg_objects.ui_load_dialog(ctx) {
            self.queue_object_ui_action(action);
        }

        if self.show_left_panel {
            let previous_tab = self.left_tab;
            let mut tab = self.left_tab;
            left_panel::show(
                ctx,
                "viewer-left",
                &mut tab,
                &[
                    left_panel::TabSpec {
                        tab: LeftTab::Layers,
                        label: "Layers",
                        panel_key: "layers",
                        default_width: 360.0,
                        scroll: false,
                    },
                    left_panel::TabSpec {
                        tab: LeftTab::Project,
                        label: "Project",
                        panel_key: "project",
                        default_width: 420.0,
                        scroll: false,
                    },
                ],
                |ui, tab| match tab {
                    LeftTab::Layers => self.ui_layers(ui, ctx),
                    LeftTab::Project => {
                        let cur = self.dataset.source.local_path().map(Path::to_path_buf);
                        self.sync_current_view_state_into_project_space();
                        if let Some(action) = self.project_space.ui(ui, cur.as_deref()) {
                            self.handle_project_space_action(action);
                        }
                    }
                },
            );
            if tab != previous_tab {
                self.native_control_intents.push(NativeControlIntent {
                    method: "viewer.ui.set_left_tab",
                    params: serde_json::json!({"tab": tab.storage_key()}),
                });
            }
        }
        if let Some(action) = self.project_space.ui_floating_windows(ctx, true) {
            self.handle_project_space_action(action);
        }
        if self.show_right_panel {
            let previous_tab = self.right_tab;
            let mut tab = self.right_tab;
            right_panel::show(
                ctx,
                "right",
                380.0,
                &mut tab,
                &[
                    right_panel::TabSpec {
                        tab: RightTab::Properties,
                        label: "Properties",
                        scroll: true,
                    },
                    right_panel::TabSpec {
                        tab: RightTab::Views,
                        label: "Views",
                        scroll: true,
                    },
                    right_panel::TabSpec {
                        tab: RightTab::Analysis,
                        label: "Analysis",
                        scroll: true,
                    },
                    right_panel::TabSpec {
                        tab: RightTab::Measurements,
                        label: "Measurements",
                        scroll: true,
                    },
                    right_panel::TabSpec {
                        tab: RightTab::Memory,
                        label: "Memory",
                        scroll: true,
                    },
                    right_panel::TabSpec {
                        tab: RightTab::RoiSelector,
                        label: "ROI Selector",
                        scroll: true,
                    },
                ],
                |ui, tab| match tab {
                    RightTab::Properties => self.ui_layer_properties(ui, ctx),
                    RightTab::Views => {
                        self.ui_help_heading(ui, "Views", crate::ui::help::HelpTopic::ViewsPanel);
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
                                self.native_control_intents.push(NativeControlIntent {
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
                                    if let Some(idx) = objects.take_pending_zoom_object_index() {
                                        if let Some(viewport) = self.last_canvas_rect
                                            && let Some(world) =
                                                objects.fit_object_bounds_world(idx, offset_world)
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
                                self.native_control_intents.push(NativeControlIntent {
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
                                self.native_control_intents
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
                        self.ui_help_heading(ui, "Memory", crate::ui::help::HelpTopic::MemoryPanel);
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
                        if let Some(action) = self.roi_selector.ui(ui) {
                            self.handle_roi_selector_action(ctx, action);
                        }
                    }
                },
            );
            if tab != previous_tab {
                self.native_control_intents.push(NativeControlIntent {
                    method: "viewer.ui.set_right_tab",
                    params: serde_json::json!({"tab": tab.storage_key()}),
                });
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            self.ui_viewport_workspace(ui, ctx);
        });

        if self.remote_dialog_open {
            self.ui_remote_dialog(ctx);
        }

        self.ui_group_layers_dialog(ctx);
        self.ui_memory_load_dialog(ctx);
        self.ui_object_export_dialogs(ctx);
        self.ui_mapping_settings_dialogs(ctx);
        self.ui_roi_info_window(ctx);
        self.ui_screenshot_settings_dialog(ctx);

        if top_bar::ui_close_dialog(ctx, &mut self.close_dialog_open) {
            self.native_control_intents.push(NativeControlIntent {
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

const DEFAULT_MAX_SCAN_PIXELS: u64 = 2_000_000;

pub(super) fn choose_default_max_level(dataset: &OmeZarrDataset) -> usize {
    let y_dim = dataset.dims.y;
    let x_dim = dataset.dims.x;
    let mut chosen = dataset.levels.len().saturating_sub(1);
    for (i, level) in dataset.levels.iter().enumerate().rev() {
        let y = *level.shape.get(y_dim).unwrap_or(&0);
        let x = *level.shape.get(x_dim).unwrap_or(&0);
        let pixels = y.saturating_mul(x);
        if pixels > 0 && pixels <= DEFAULT_MAX_SCAN_PIXELS {
            chosen = i;
        }
    }
    chosen
}
