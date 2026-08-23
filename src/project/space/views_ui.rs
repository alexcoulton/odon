//! Floating windows, saved-view panels, object-cache controls, and view dialogs.

use super::*;

impl ProjectSpace {
    pub fn ui_floating_windows(
        &mut self,
        ctx: &egui::Context,
        can_capture_current_view: bool,
    ) -> Option<ProjectSpaceAction> {
        let mut action = None;
        self.ui_save_toast(ctx);
        self.ui_views_dialog(ctx, can_capture_current_view, &mut action);
        action
    }

    pub fn ui_views_panel(
        &mut self,
        ui: &mut egui::Ui,
        target_roi: Option<ProjectRoi>,
        can_capture_current_view: bool,
    ) -> Option<ProjectSpaceAction> {
        let mut action = None;
        ui.heading("Views");
        ui.horizontal_wrapped(|ui| {
            if ui.button("Manage...").clicked() {
                self.views_dialog_open = true;
            }
            if ui
                .add_enabled(can_capture_current_view, egui::Button::new("Capture"))
                .clicked()
            {
                action = Some(ProjectSpaceAction::CaptureCurrentView);
            }
        });

        if self.state.view_presets.is_empty() {
            ui.label("No saved views.");
            ui.small("Configure the viewer, then capture and save a view.");
            return action;
        }

        self.selected_view_preset = self
            .selected_view_preset
            .min(self.state.view_presets.len().saturating_sub(1));

        ui.add_space(6.0);
        ui.horizontal_wrapped(|ui| {
            if ui.button("Prev").clicked() {
                self.selected_view_preset = if self.selected_view_preset == 0 {
                    self.state.view_presets.len() - 1
                } else {
                    self.selected_view_preset - 1
                };
                if let (Some(roi), Some(preset)) = (
                    target_roi.clone(),
                    self.state
                        .view_presets
                        .get(self.selected_view_preset)
                        .cloned(),
                ) {
                    action = Some(ProjectSpaceAction::OpenView(roi, preset.spec));
                }
            }
            if ui.button("Next").clicked() {
                self.selected_view_preset =
                    (self.selected_view_preset + 1) % self.state.view_presets.len();
                if let (Some(roi), Some(preset)) = (
                    target_roi.clone(),
                    self.state
                        .view_presets
                        .get(self.selected_view_preset)
                        .cloned(),
                ) {
                    action = Some(ProjectSpaceAction::OpenView(roi, preset.spec));
                }
            }
            if ui
                .add_enabled(target_roi.is_some(), egui::Button::new("Apply"))
                .clicked()
                && let (Some(roi), Some(preset)) = (
                    target_roi.clone(),
                    self.state
                        .view_presets
                        .get(self.selected_view_preset)
                        .cloned(),
                )
            {
                action = Some(ProjectSpaceAction::OpenView(roi, preset.spec));
            }
        });

        ui.add_space(6.0);
        egui::ScrollArea::vertical()
            .id_salt("project-views-right-panel-list")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for (idx, preset) in self.state.view_presets.iter().enumerate() {
                    let selected = idx == self.selected_view_preset;
                    let resp = ui
                        .selectable_label(selected, preset.name.as_str())
                        .on_hover_text(view_preset_summary(preset));
                    if resp.clicked() {
                        self.selected_view_preset = idx;
                    }
                    if resp.double_clicked()
                        && let Some(roi) = target_roi.clone()
                    {
                        action = Some(ProjectSpaceAction::OpenView(roi, preset.spec.clone()));
                    }
                }
            });

        if let Some(preset) = self.state.view_presets.get(self.selected_view_preset) {
            ui.separator();
            let channel_names = visible_channel_display_names(&preset.spec);
            let channels = if channel_names.is_empty() {
                "(current channels)".to_string()
            } else {
                channel_names.join(", ")
            };
            ui.small(format!("Channels: {channels}"));
            if let Some(color_by) = preset.spec.cell_color_by.as_deref() {
                ui.small(format!("Color by: {color_by}"));
            }
            if !preset.spec.visible_cell_types.is_empty() {
                ui.small(format!(
                    "Visible values: {}",
                    preset.spec.visible_cell_types.join(", ")
                ));
            }
        }
        if target_roi.is_none() {
            ui.separator();
            ui.small("Open a project ROI to apply views from this tab.");
        }
        action
    }

    pub(super) fn ui_object_cache(
        &mut self,
        ui: &mut egui::Ui,
        action: &mut Option<ProjectSpaceAction>,
    ) {
        let cache = self.object_cache_ui;
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            ui.heading("Object Cache");
            if crate::ui::help::help_button(ui, HelpTopic::ObjectCache) {
                *action = Some(ProjectSpaceAction::ShowHelp(HelpTopic::ObjectCache));
            }
        });
        ui.label(format!(
            "{} GeoParquet/Parquet segmentation file(s), {} on disk.",
            cache.available_count,
            format_bytes(cache.on_disk_bytes)
        ));
        ui.horizontal_wrapped(|ui| {
            ui.label("Mode");
            egui::ComboBox::from_id_salt("project_object_cache_mode")
                .selected_text(self.object_cache_settings.mode.label())
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.object_cache_settings.mode,
                        ObjectPreloadMode::FullGeometry,
                        ObjectPreloadMode::FullGeometry.label(),
                    );
                    ui.selectable_value(
                        &mut self.object_cache_settings.mode,
                        ObjectPreloadMode::CentroidPoints,
                        ObjectPreloadMode::CentroidPoints.label(),
                    );
                });
        });
        ui.checkbox(
            &mut self.object_cache_settings.lazy_properties,
            "Load properties lazily",
        );
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(
                    !cache.loading
                        && cache.available_count > 0
                        && self.saved_project_path().is_some(),
                    egui::Button::new("Preload object segmentations"),
                )
                .clicked()
            {
                *action = Some(ProjectSpaceAction::PreloadObjectSegmentations(
                    self.object_cache_settings,
                ));
            }
            if ui
                .add_enabled(
                    cache.loading || cache.cached > 0,
                    egui::Button::new("Clear object cache"),
                )
                .clicked()
            {
                *action = Some(ProjectSpaceAction::ClearObjectCache);
            }
        });
        if cache.total > 0 {
            let done = cache.done.min(cache.total);
            let fraction = done as f32 / cache.total as f32;
            ui.add(
                egui::ProgressBar::new(fraction)
                    .show_percentage()
                    .text(format!("{done} / {}", cache.total)),
            );
        }
        ui.label(format!(
            "{} cached ({}, {}), {} failed{}.",
            cache.cached,
            cache.cached_settings.mode.label(),
            cache.cached_settings.property_label(),
            cache.failed,
            if cache.loading { ", loading" } else { "" }
        ));
    }

    pub(super) fn ui_views_launcher(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            let count = self.state.view_presets.len();
            if ui.button(format!("Views... ({count})")).clicked() {
                self.views_dialog_open = true;
            }
            if let Some(preset) = self.state.view_presets.get(self.selected_view_preset) {
                ui.small(format!("Selected: {}", preset.name));
            }
        });
    }

    pub(super) fn ui_views_dialog(
        &mut self,
        ctx: &egui::Context,
        can_capture_current_view: bool,
        action: &mut Option<ProjectSpaceAction>,
    ) {
        if !self.views_dialog_open {
            return;
        }
        let mut open = self.views_dialog_open;
        egui::Window::new("Project Views")
            .open(&mut open)
            .default_width(560.0)
            .default_height(420.0)
            .resizable(true)
            .show(ctx, |ui| {
                self.ui_views_dialog_contents(ui, can_capture_current_view, action);
            });
        self.views_dialog_open = open;
    }

    pub(super) fn ui_views_dialog_contents(
        &mut self,
        ui: &mut egui::Ui,
        can_capture_current_view: bool,
        action: &mut Option<ProjectSpaceAction>,
    ) {
        ui.heading("Save Current View");

        ui.horizontal(|ui| {
            ui.label("Name");
            ui.add_sized(
                [ui.available_width().max(180.0), 0.0],
                egui::TextEdit::singleline(&mut self.view_preset_name_input)
                    .hint_text("e.g. Tumour overview"),
            );
        });
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    can_capture_current_view,
                    egui::Button::new("Capture current view"),
                )
                .clicked()
            {
                *action = Some(ProjectSpaceAction::CaptureCurrentView);
            }
            let can_save =
                self.view_preset_draft.is_some() && !self.view_preset_name_input.trim().is_empty();
            if ui
                .add_enabled(can_save, egui::Button::new("Save view"))
                .clicked()
                && let Some(spec) = self.view_preset_draft.clone()
            {
                self.save_view_preset(self.view_preset_name_input.clone(), spec);
            }
        });
        if !can_capture_current_view {
            ui.small("Open an ROI to save a view from the live viewer.");
        }
        if let Some(draft) = self.view_preset_draft.as_mut() {
            ui.add_space(6.0);
            ui.label("Channel aliases");
            if draft.visible_channel_refs.is_empty() {
                ui.small("No visible channels were captured.");
            } else {
                egui::Grid::new("project-view-draft-channel-aliases")
                    .num_columns(2)
                    .spacing([12.0, 4.0])
                    .striped(true)
                    .show(ui, |ui| {
                        ui.small("Channel");
                        ui.small("Alias");
                        ui.end_row();
                        for channel in &mut draft.visible_channel_refs {
                            ui.label(channel.label.as_str());
                            ui.add(
                                egui::TextEdit::singleline(&mut channel.alias).desired_width(160.0),
                            );
                            ui.end_row();
                        }
                    });
            }
        }

        ui.add_space(10.0);
        ui.separator();
        ui.heading("Saved Views");

        if self.state.view_presets.is_empty() {
            ui.small("No saved views.");
            return;
        }

        self.selected_view_preset = self
            .selected_view_preset
            .min(self.state.view_presets.len().saturating_sub(1));

        let list_height = (ui.available_height() * 0.45).clamp(120.0, 220.0);
        egui::Frame::group(ui.style()).show(ui, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("project-view-preset-list")
                .max_height(list_height)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    for (idx, preset) in self.state.view_presets.iter().enumerate() {
                        let selected = idx == self.selected_view_preset;
                        if ui
                            .selectable_label(selected, preset.name.as_str())
                            .on_hover_text(view_preset_summary(preset))
                            .clicked()
                        {
                            self.selected_view_preset = idx;
                            if self.view_preset_name_input.trim().is_empty() {
                                self.view_preset_name_input = preset.name.clone();
                            }
                        }
                    }
                });
        });

        if let Some(preset) = self.state.view_presets.get(self.selected_view_preset) {
            if !preset.description.trim().is_empty() {
                ui.label(preset.description.clone());
            }
            let channel_names = visible_channel_display_names(&preset.spec);
            let channels = if channel_names.is_empty() {
                "(current channels)".to_string()
            } else {
                channel_names.join(", ")
            };
            let cell_types = if preset.spec.visible_cell_types.is_empty() {
                "(all cell types)".to_string()
            } else {
                preset.spec.visible_cell_types.join(", ")
            };
            ui.small(format!("Markers: {channels}"));
            ui.small(format!("Cell types: {cell_types}"));
        }

        let focused_roi = self.focused_roi().cloned();
        ui.horizontal(|ui| {
            if ui
                .add_enabled(focused_roi.is_some(), egui::Button::new("Open view"))
                .clicked()
                && let (Some(roi), Some(preset)) = (
                    focused_roi.clone(),
                    self.state
                        .view_presets
                        .get(self.selected_view_preset)
                        .cloned(),
                )
            {
                *action = Some(ProjectSpaceAction::OpenView(roi, preset.spec));
            }
            if ui.button("Prev preset").clicked() && !self.state.view_presets.is_empty() {
                self.selected_view_preset = if self.selected_view_preset == 0 {
                    self.state.view_presets.len() - 1
                } else {
                    self.selected_view_preset - 1
                };
            }
            if ui.button("Next preset").clicked() && !self.state.view_presets.is_empty() {
                self.selected_view_preset =
                    (self.selected_view_preset + 1) % self.state.view_presets.len();
            }
            if ui.button("Delete").clicked() && !self.state.view_presets.is_empty() {
                let idx = self
                    .selected_view_preset
                    .min(self.state.view_presets.len().saturating_sub(1));
                if let Err(error) = self.delete_view_preset(idx) {
                    self.status = error;
                }
            }
        });

        if let Some(roi) = focused_roi {
            ui.small(format!("Focused ROI: {}", roi.source_display()));
        } else {
            ui.small("Focus an ROI below to open a preset view.");
        }
    }
}
