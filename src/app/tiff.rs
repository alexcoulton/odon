use super::*;

impl OmeZarrViewerApp {
    pub(super) fn apply_tiff_plane_selection(
        &mut self,
        ctx: &egui::Context,
        target_z: usize,
        target_t: usize,
    ) -> anyhow::Result<()> {
        let Some(prev_state) = self.tiff_plane_state.clone() else {
            anyhow::bail!("TIFF plane selection is not active");
        };

        let prev_channels = self.channels.clone();
        let prev_selected_name = self
            .channels
            .get(self.selected_channel)
            .map(|c| c.name.clone());
        let old_world_w = self
            .dataset
            .levels
            .first()
            .map(|l| l.shape[self.dataset.dims.x] as f32)
            .unwrap_or(0.0);
        let old_world_h = self
            .dataset
            .levels
            .first()
            .map(|l| l.shape[self.dataset.dims.y] as f32)
            .unwrap_or(0.0);
        let fx = if old_world_w > 0.0 {
            (self.camera.center_world_lvl0.x / old_world_w).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let fy = if old_world_h > 0.0 {
            (self.camera.center_world_lvl0.y / old_world_h).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let old_zoom = self.camera.zoom_screen_per_lvl0_px;
        let preserve_transforms = self.channels.len();

        let mut assets = build_tiff_runtime_assets(
            self.tiles_gl.is_some(),
            prev_state.dataset_root,
            prev_state.image_path,
            prev_state.dataset_name,
            prev_state.channel_name,
            crate::xenium::TiffPlaneSelection {
                z: target_z,
                t: target_t,
            },
        )?;

        let mut new_channels = assets.dataset.channels.clone();
        apply_preserved_channel_settings(&prev_channels, &mut new_channels);
        for ch in &mut new_channels {
            if let Some(w) = self.channel_window_overrides.get(&ch.name).copied() {
                ch.window = Some(w);
            }
        }
        assets.dataset.channels = new_channels.clone();

        if let Some(tiles_gl) = self.tiles_gl.as_ref() {
            tiles_gl.reset();
        }
        if let Some(labels_gl) = self.labels_gl.as_ref() {
            labels_gl.reset();
        }

        self.dataset = assets.dataset;
        self.store = assets.store;
        self.loader = assets.loader;
        self.raw_loader = assets.raw_loader;
        self.hist_loader = assets.hist_loader;
        self.chanmax_loader = assets.chanmax_loader;
        self.chanmax_level = assets.chanmax_level;
        self.channels = new_channels;
        self.tiff_plane_state = assets.tiff_plane_state.take();
        if let Some(state) = self.tiff_plane_state.as_mut() {
            state.status.clear();
        }

        if self.channels.len() != preserve_transforms {
            self.channel_offsets_world = vec![egui::Vec2::ZERO; self.channels.len()];
            self.channel_scales = vec![egui::Vec2::splat(1.0); self.channels.len()];
            self.channel_rotations_rad = vec![0.0; self.channels.len()];
        }

        if let Some(name) = prev_selected_name {
            if let Some(ch) = self.channels.iter().find(|c| c.name == name) {
                self.selected_channel = ch.index;
            } else {
                self.selected_channel = self
                    .selected_channel
                    .min(self.channels.len().saturating_sub(1));
            }
        } else {
            self.selected_channel = self
                .selected_channel
                .min(self.channels.len().saturating_sub(1));
        }
        if matches!(self.active_layer, LayerId::Channel(_)) {
            self.active_layer = LayerId::Channel(self.selected_channel);
        }

        let new_world_w = self
            .dataset
            .levels
            .first()
            .map(|l| l.shape[self.dataset.dims.x] as f32)
            .unwrap_or(0.0);
        let new_world_h = self
            .dataset
            .levels
            .first()
            .map(|l| l.shape[self.dataset.dims.y] as f32)
            .unwrap_or(0.0);
        self.camera.center_world_lvl0 = egui::pos2(new_world_w * fx, new_world_h * fy);
        self.camera.zoom_screen_per_lvl0_px = old_zoom;

        self.cache = TileCache::new(256);
        self.pending.clear();
        self.previous_render_id = None;
        self.previous_view_selection = None;
        self.previous_displayed_view_selection = None;
        self.active_render_id = self.compute_render_id();
        self.last_render_view_selection = self.committed_view_selection();
        self.hist = None;
        self.hist_request_id = 0;
        self.hist_request_pending = false;
        self.hist_dirty = true;
        self.hist_navigation_dirty_since = None;
        self.hist_last_sent = Instant::now()
            .checked_sub(Duration::from_secs(3600))
            .unwrap_or_else(Instant::now);
        self.chanmax_request_id = self.chanmax_request_id.wrapping_add(1).max(1);
        self.chanmax_pending = vec![false; self.channels.len()];
        self.chanmax_snapshot = self.channels.iter().map(|c| c.window).collect();
        self.maybe_apply_auto_contrast_on_open();
        ctx.request_repaint();
        Ok(())
    }

    pub(super) fn ui_screenshot_settings_dialog(&mut self, ctx: &egui::Context) {
        if !self.screenshot_settings_open {
            return;
        }
        let mut open = self.screenshot_settings_open;
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
                        .screenshot_output_dir
                        .as_deref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "Not set".to_string());
                    ui.monospace(folder_text);
                    if ui.button("Choose...").clicked() {
                        let mut dialog = FileDialog::new().set_title("Select Screenshot Folder");
                        if let Some(dir) = self.screenshot_output_dir.as_deref() {
                            dialog = dialog.set_directory(dir);
                        }
                        if let Some(dir) = dialog.pick_folder() {
                            self.screenshot_output_dir = Some(dir);
                        }
                    }
                    if ui
                        .add_enabled(
                            self.screenshot_output_dir.is_some(),
                            egui::Button::new("Clear"),
                        )
                        .clicked()
                    {
                        self.screenshot_output_dir = None;
                    }
                });
                ui.add_space(6.0);
                ui.checkbox(
                    &mut self.screenshot_settings.include_scale_bar,
                    "Include scale bar",
                );
                ui.checkbox(
                    &mut self.screenshot_settings.include_legend,
                    "Include legend (visible markers)",
                );
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label("Scale bar size");
                    ui.add(
                        egui::Slider::new(&mut self.screenshot_settings.scale_bar_scale, 0.5..=3.0)
                            .suffix("x"),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Legend size");
                    ui.add(
                        egui::Slider::new(&mut self.screenshot_settings.legend_scale, 0.5..=3.0)
                            .suffix("x"),
                    );
                });
            });
        self.screenshot_settings_open = open;
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

    pub(super) fn ui_tiff_plane_controls(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let Some(mut state) = self.tiff_plane_state.clone() else {
            return;
        };
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
                egui::DragValue::new(&mut state.draft_z).range(0..=state.size_z.saturating_sub(1)),
            );
            ui.label("T");
            ui.add_enabled(
                state.size_t > 1,
                egui::DragValue::new(&mut state.draft_t).range(0..=state.size_t.saturating_sub(1)),
            );
        });

        let changed = state.draft_z != state.current_z || state.draft_t != state.current_t;
        ui.horizontal(|ui| {
            if ui
                .add_enabled(changed, egui::Button::new("Apply"))
                .clicked()
            {
                match self.apply_tiff_plane_selection(ctx, state.draft_z, state.draft_t) {
                    Ok(()) => return,
                    Err(err) => state.status = err.to_string(),
                }
            }
            if ui
                .add_enabled(changed, egui::Button::new("Reset"))
                .clicked()
            {
                state.draft_z = state.current_z;
                state.draft_t = state.current_t;
                state.status.clear();
            }
        });

        if !state.status.is_empty() {
            ui.colored_label(ui.visuals().warn_fg_color, &state.status);
        }
        self.tiff_plane_state = Some(state);
        ui.separator();
    }
}
