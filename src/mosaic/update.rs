//! Top-level eframe lifecycle and frame orchestration.

use super::*;

impl eframe::App for MosaicViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Per-frame flow mirrors the single-view app but with shared mosaic state:
        // refresh/tick async overlays, build chrome and side panels, then draw the current
        // viewport while progressively refining visible ROIs.
        self.refresh_system_memory_if_needed();
        self.seg_geojson.tick();
        // Napari-like "close window" prompt.
        // - Cmd/Ctrl+W opens confirmation
        // - Cmd/Ctrl+W again confirms close
        if top_bar::handle_cmd_w_close(ctx, &mut self.close_dialog_open) {
            self.submit_native_control_intent(
                "app.lifecycle.request_close",
                serde_json::json!({"save":"discard"}),
            );
        }

        let actor_shell_layout = self.control_shell_projection.get("layout").is_some();
        if !actor_shell_layout {
            let top_bar_visible = self.shell_node_visible("builtin:mosaic.top-bar", true);
            egui::TopBottomPanel::top("top").show_animated(ctx, top_bar_visible, |ui| {
                self.ui_mosaic_top_bar(ui, ctx, &serde_json::Value::Null);
            });
        }
        if !actor_shell_layout {
            let left_tabs = self.shell_left_tabs();
            if self.show_left_panel && !left_tabs.is_empty() {
                let mut tab = self.left_tab;
                left_panel::show(
                    ctx,
                    "mosaic-left",
                    &mut tab,
                    &left_tabs,
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
            let right_tabs = self.shell_right_tabs();
            if self.show_right_panel && !right_tabs.is_empty() {
                let mut tab = self.right_tab;
                right_panel::show(
                    ctx,
                    "right",
                    self.shell_right_panel_width(380.0),
                    &mut tab,
                    &right_tabs,
                    |ui, tab| match tab {
                        RightTab::Properties => self.ui_properties(ui),
                        RightTab::Views => {
                            if let Some(action) = self.project_space.ui_views_panel(ui, None, false)
                            {
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
        }
        if let Some(action) = self.project_space.ui_floating_windows(ctx, false) {
            self.handle_project_space_action(action);
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            if !actor_shell_layout || !self.ui_actor_shell_tree(ui, ctx) {
                self.ui_canvas(ui, ctx);
            }
        });

        self.ui_group_layers_dialog(ctx);
        self.ui_memory_load_dialog(ctx);
        self.ui_screenshot_settings_dialog(ctx);

        if top_bar::ui_close_dialog(ctx, &mut self.close_dialog_open) {
            self.submit_native_control_intent(
                "app.lifecycle.request_close",
                serde_json::json!({"save":"discard"}),
            );
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
            || self.screenshot_capture.pending.is_some()
        {
            repaint_control::request_repaint_busy(ctx);
        }
    }
}
