//! Top-level eframe lifecycle and frame orchestration.

use super::*;

impl eframe::App for MosaicViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Per-frame flow mirrors the single-view app but with shared mosaic state:
        // refresh/tick async overlays, build chrome and side panels, then draw the current
        // viewport while progressively refining visible ROIs.
        self.refresh_system_memory_if_needed();
        self.seg_geojson.tick();
        self.drain_screenshots();
        // Napari-like "close window" prompt.
        // - Cmd/Ctrl+W opens confirmation
        // - Cmd/Ctrl+W again confirms close
        if top_bar::handle_cmd_w_close(ctx, &mut self.close_dialog_open) {
            self.pending_request = Some(MosaicRequest::CloseWindow);
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                top_bar::ui_title(ui, format!("Mosaic: {} ROIs", self.items.len()));
                ui.separator();
                if top_bar::ui_back(ui, self.allow_back) {
                    self.pending_request = Some(MosaicRequest::BackToSingle);
                }
                if top_bar::ui_fit(ui, "Fit Mosaic (F)") {
                    self.fit_mosaic();
                }
                ui.separator();
                top_bar::ui_status(ui, &self.status);
                ui.separator();
                let have_items = !self.items.is_empty();
                if let Some(step) = top_bar::ui_prev_next_core(ui, have_items) {
                    self.step_focused_core(ctx, step);
                }
                top_bar::ui_core_index(ui, self.focused_core_summary());
                ui.separator();
                let have_channels = !self.channels.is_empty();
                if let Some(step) = top_bar::ui_prev_next_channel(ui, have_channels) {
                    self.step_selected_channel_visibility(step);
                }
                ui.separator();
                let mut show_left_panel = self.show_left_panel;
                let mut show_right_panel = self.show_right_panel;
                top_bar::ui_panel_toggles(ui, &mut show_left_panel, &mut show_right_panel);
                if (show_left_panel, show_right_panel)
                    != (self.show_left_panel, self.show_right_panel)
                {
                    self.submit_native_control_intent(
                        "viewer.panels.set",
                        serde_json::json!({
                            "left":show_left_panel,
                            "right":show_right_panel,
                        }),
                    );
                }
                let mut smooth_pixels = self.smooth_pixels;
                if top_bar::ui_smooth_toggle(ui, &mut smooth_pixels) {
                    self.submit_native_control_intent(
                        "mosaic.rendering.set",
                        serde_json::json!({"smooth_pixels":smooth_pixels}),
                    );
                }
                let mut show_tile_debug = self.show_tile_debug;
                if ui.checkbox(&mut show_tile_debug, "Tile Debug").changed() {
                    self.submit_native_control_intent(
                        "mosaic.rendering.set",
                        serde_json::json!({"show_tile_debug":show_tile_debug}),
                    );
                }

                if have_channels {
                    ui.separator();
                    self.ui_top_bar_quick_contrast(ui);
                }
            });
        });

        if self.show_left_panel {
            let mut tab = self.left_tab;
            left_panel::show(
                ctx,
                "mosaic-left",
                &mut tab,
                &[
                    left_panel::TabSpec {
                        tab: LeftTab::Layers,
                        label: "Layers",
                        panel_key: "layers",
                        default_width: 360.0,
                        scroll: true,
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
                    LeftTab::Project => self.ui_project(ui),
                },
            );
            if tab != self.left_tab {
                self.submit_native_control_intent(
                    "mosaic.ui.set_left_tab",
                    serde_json::json!({"tab":tab.storage_key()}),
                );
            }
        }
        if let Some(action) = self.project_space.ui_floating_windows(ctx, false) {
            self.handle_project_space_action(action);
        }

        if self.show_right_panel {
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
                        tab: RightTab::Layout,
                        label: "Layout",
                        scroll: true,
                    },
                    right_panel::TabSpec {
                        tab: RightTab::Memory,
                        label: "Memory",
                        scroll: true,
                    },
                ],
                |ui, tab| match tab {
                    RightTab::Properties => self.ui_properties(ui),
                    RightTab::Views => {
                        if let Some(action) = self.project_space.ui_views_panel(ui, None, false) {
                            self.handle_project_space_action(action);
                        }
                    }
                    RightTab::Layout => self.ui_layout(ui, ctx),
                    RightTab::Memory => self.ui_memory(ui),
                },
            );
            if tab != self.right_tab {
                self.submit_native_control_intent(
                    "mosaic.ui.set_right_tab",
                    serde_json::json!({"tab":tab.storage_key()}),
                );
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            self.ui_canvas(ui, ctx);
        });

        self.ui_group_layers_dialog(ctx);
        self.ui_memory_load_dialog(ctx);
        self.ui_screenshot_settings_dialog(ctx);

        if top_bar::ui_close_dialog(ctx, &mut self.close_dialog_open) {
            self.pending_request = Some(MosaicRequest::CloseWindow);
        }
        crate::ui::help::show_help_window(ctx, &mut self.active_help_topic);

        if ctx.input(|i| i.key_pressed(egui::Key::F)) {
            self.fit_mosaic();
        }

        // Avoid a busy loop when idle. Only repaint while we are interacting or still streaming.
        if self.tiles_gl.is_busy()
            || self.seg_geojson.is_busy()
            || self.seg_geojson_pending_visible
            || self.projected_memory_running()
            || self.screenshot_pending.is_some()
            || self.screenshot_in_flight.is_some()
        {
            repaint_control::request_repaint_busy(ctx);
        }
    }
}
