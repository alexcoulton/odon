//! Shared mount-based dispatcher for built-in mosaic shell components.

use super::*;

impl MosaicViewerApp {
    pub(super) fn ui_mosaic_top_bar(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        configuration: &serde_json::Value,
    ) {
        ui.horizontal(|ui| {
            let mut has_content = false;
            if shell_configuration_flag(configuration, "show_title") {
                top_bar::ui_title(ui, format!("Mosaic: {} ROIs", self.items.len()));
                has_content = true;
            }
            if shell_configuration_flag(configuration, "show_navigation") {
                top_bar_section(ui, &mut has_content);
                if top_bar::ui_back(ui, self.show_return_navigation) {
                    if let Some(path) = self.return_dataset_root.clone() {
                        self.submit_native_control_intent(
                            "datasets.open_ome_zarr",
                            serde_json::json!({"path":path}),
                        );
                    } else {
                        self.submit_native_control_intent(
                            "app.navigation.show_project",
                            serde_json::json!({}),
                        );
                    }
                }
                if top_bar::ui_fit(ui, "Fit Mosaic (F)") {
                    self.fit_mosaic();
                }
            }
            if shell_configuration_flag(configuration, "show_status") {
                top_bar_section(ui, &mut has_content);
                top_bar::ui_status(ui, &self.renderer_status);
            }
            let have_channels = !self.channels.is_empty();
            if shell_configuration_flag(configuration, "show_navigation") {
                top_bar_section(ui, &mut has_content);
                let have_items = !self.items.is_empty();
                if let Some(step) = top_bar::ui_prev_next_core(ui, have_items) {
                    self.step_focused_core(ctx, step);
                }
                top_bar::ui_core_index(ui, self.focused_core_summary());
                ui.separator();
                if let Some(step) = top_bar::ui_prev_next_channel(ui, have_channels) {
                    self.step_selected_channel_visibility(step);
                }
            }
            if shell_configuration_flag(configuration, "show_panel_controls") {
                top_bar_section(ui, &mut has_content);
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
            }
            if shell_configuration_flag(configuration, "show_rendering_controls") {
                top_bar_section(ui, &mut has_content);
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
                    let desired = self.channel_list_search.clone();
                    self.channel_list_search = channel_search_before;
                    self.submit_native_control_intent(
                        "viewer.channels.presentation.set",
                        serde_json::json!({"search":desired}),
                    );
                }
            }
            "builtin:project" => self.ui_project(ui),
            "builtin:properties" => self.ui_properties(ui),
            "builtin:views" => {
                if let Some(action) = self.project_space.ui_views_panel(ui, None, false) {
                    self.handle_project_space_action(action);
                }
            }
            "builtin:mosaic-layout" => self.ui_layout(ui, ctx),
            "builtin:memory" => self.ui_memory(ui),
            "builtin:recovery-controls" => {
                if crate::ui::shell_recovery::render(ui) {
                    self.submit_native_control_intent("ui.shell.recover", serde_json::json!({}));
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
