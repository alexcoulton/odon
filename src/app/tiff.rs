use super::*;

impl OmeZarrViewerApp {
    pub(super) fn ui_screenshot_settings_dialog(&mut self, ctx: &egui::Context) {
        if !self.screenshot_dialog.open {
            return;
        }
        let before_output_dir = self.screenshot_dialog.output_dir.clone();
        let before_settings = self.screenshot_dialog.settings;
        let mut open = self.screenshot_dialog.open;
        egui::Window::new("Screenshot Settings")
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.label("These options affect canvas-only PNG screenshots.");
                ui.label(
                    "Quick Screenshot uses Cmd+Shift+S and saves directly to the folder below.",
                );
                ui.add_space(6.0);
                ui.label("Quick-save folder");
                ui.horizontal(|ui| {
                    let folder_text = self
                        .screenshot_dialog
                        .output_dir
                        .as_deref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "Not set".to_string());
                    ui.monospace(folder_text);
                    if ui.button("Choose...").clicked() {
                        let mut dialog = FileDialog::new().set_title("Select Screenshot Folder");
                        if let Some(dir) = self.screenshot_dialog.output_dir.as_deref() {
                            dialog = dialog.set_directory(dir);
                        }
                        if let Some(dir) = dialog.pick_folder() {
                            self.screenshot_dialog.output_dir = Some(dir);
                        }
                    }
                    if ui
                        .add_enabled(
                            self.screenshot_dialog.output_dir.is_some(),
                            egui::Button::new("Clear"),
                        )
                        .clicked()
                    {
                        self.screenshot_dialog.output_dir = None;
                    }
                });
                ui.add_space(6.0);
                ui.checkbox(
                    &mut self.screenshot_dialog.settings.include_scale_bar,
                    "Include scale bar",
                );
                ui.checkbox(
                    &mut self.screenshot_dialog.settings.include_legend,
                    "Include legend (visible markers)",
                );
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label("Scale bar size");
                    ui.add(
                        egui::Slider::new(
                            &mut self.screenshot_dialog.settings.scale_bar_scale,
                            0.5..=3.0,
                        )
                        .suffix("x"),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Legend size");
                    ui.add(
                        egui::Slider::new(
                            &mut self.screenshot_dialog.settings.legend_scale,
                            0.5..=3.0,
                        )
                        .suffix("x"),
                    );
                });
            });
        self.screenshot_dialog.open = open;
        if self.screenshot_dialog.output_dir != before_output_dir
            || self.screenshot_dialog.settings != before_settings
        {
            self.native_command_ingress.push(NativeControlIntent {
                method: "viewer.screenshot.settings.set",
                params: serde_json::json!({
                    "output_dir":self.screenshot_dialog.output_dir.as_ref().map(|path| path.to_string_lossy().into_owned()),
                    "include_scale_bar":self.screenshot_dialog.settings.include_scale_bar,
                    "include_legend":self.screenshot_dialog.settings.include_legend,
                    "scale_bar_scale":self.screenshot_dialog.settings.scale_bar_scale,
                    "legend_scale":self.screenshot_dialog.settings.legend_scale,
                }),
            });
        }
    }

    pub(super) fn ui_roi_info_window(&mut self, ctx: &egui::Context) {
        if !self.roi_info_open {
            return;
        }
        let mut open = self.roi_info_open;
        egui::Window::new("ROI Info")
            .default_width(560.0)
            .default_height(420.0)
            .open(&mut open)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        self.ui_current_roi_summary(ui);
                    });
            });
        self.roi_info_open = open;
    }

    pub(super) fn ui_tiff_plane_controls(&mut self, _ctx: &egui::Context, ui: &mut egui::Ui) {
        let Some(state) = self.tiff_plane_state.clone() else {
            return;
        };
        let mut draft = self
            .tiff_plane_draft
            .clone()
            .unwrap_or_else(|| TiffPlaneDraft::from_projection(&state));
        if state.size_z <= 1 && state.size_t <= 1 {
            return;
        }

        ui.heading("Plane");
        ui.label(format!(
            "Current: Z={}, T={}",
            state.current_z, state.current_t
        ));

        ui.horizontal(|ui| {
            ui.label("Z");
            ui.add_enabled(
                state.size_z > 1,
                egui::DragValue::new(&mut draft.draft_z).range(0..=state.size_z.saturating_sub(1)),
            );
            ui.label("T");
            ui.add_enabled(
                state.size_t > 1,
                egui::DragValue::new(&mut draft.draft_t).range(0..=state.size_t.saturating_sub(1)),
            );
        });

        let changed = draft.draft_z != state.current_z || draft.draft_t != state.current_t;
        ui.horizontal(|ui| {
            if ui
                .add_enabled(changed, egui::Button::new("Apply"))
                .clicked()
            {
                self.native_command_ingress.push(NativeControlIntent {
                    method: "datasets.open_tiff",
                    params: serde_json::json!({
                        "path":state.image_path.to_string_lossy(),
                        "z":draft.draft_z,
                        "t":draft.draft_t,
                    }),
                });
                draft.status = format!(
                    "Opening TIFF plane Z={}, T={}...",
                    draft.draft_z, draft.draft_t
                );
            }
            if ui
                .add_enabled(changed, egui::Button::new("Reset"))
                .clicked()
            {
                draft.draft_z = state.current_z;
                draft.draft_t = state.current_t;
                draft.status.clear();
            }
        });

        if !draft.status.is_empty() {
            ui.colored_label(ui.visuals().warn_fg_color, &draft.status);
        }
        self.tiff_plane_draft = Some(draft);
        ui.separator();
    }
}
