use super::*;

impl OmeZarrViewerApp {
    pub(super) fn ui_contrast(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        self.ui_tiff_plane_controls(ctx, ui);
        ui.heading("Contrast");

        let abs_max = self.dataset.abs_max.max(1.0);

        let selected_channel = self.selected_channel;
        let Some(selected_info) = self.channels.get(selected_channel).cloned() else {
            ui.label("No channel selected.");
            return;
        };
        let selected_name = selected_info.name.clone();
        ui.label(format!("Channel: {selected_name}"));

        // Optional group + inherit/override semantics (Napari-like).
        let mut groups_cfg = self.current_layer_groups();
        let mut groups_changed = false;
        let mut selected_group: Option<u64> = groups_cfg
            .channel_members
            .get(selected_name.as_str())
            .map(|m| m.group_id)
            .filter(|gid| groups_cfg.channel_groups.iter().any(|g| g.id == *gid));

        ui.horizontal(|ui| {
            ui.label("Group");
            egui::ComboBox::from_id_salt("channel-group-select")
                .selected_text(
                    selected_group
                        .and_then(|gid| groups_cfg.channel_groups.iter().find(|g| g.id == gid))
                        .map(|g| g.name.as_str())
                        .unwrap_or("(none)"),
                )
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut selected_group, None, "(none)");
                    for g in &groups_cfg.channel_groups {
                        ui.selectable_value(&mut selected_group, Some(g.id), g.name.clone());
                    }
                });

            if ui
                .button("+ Group")
                .on_hover_text("Create a new group")
                .clicked()
            {
                let existing = groups_cfg
                    .channel_groups
                    .iter()
                    .map(|g| g.id)
                    .collect::<Vec<_>>();
                let id = layer_groups::next_group_id(&existing);
                groups_cfg
                    .channel_groups
                    .push(crate::data::project_config::ProjectChannelGroup {
                        id,
                        name: format!("Group {id}"),
                        expanded: true,
                        color_rgb: [255, 255, 255],
                    });
                selected_group = Some(id);
                groups_changed = true;
            }
        });

        // Apply membership change.
        let have_member = groups_cfg
            .channel_members
            .get(selected_name.as_str())
            .is_some();
        if selected_group.is_none() && have_member {
            groups_cfg.channel_members.remove(selected_name.as_str());
            groups_changed = true;
        } else if let Some(gid) = selected_group {
            match groups_cfg.channel_members.get_mut(selected_name.as_str()) {
                Some(m) => {
                    if m.group_id != gid {
                        m.group_id = gid;
                        groups_changed = true;
                    }
                }
                None => {
                    groups_cfg.channel_members.insert(
                        selected_name.clone(),
                        crate::data::project_config::ProjectChannelGroupMember {
                            group_id: gid,
                            inherit_color: true,
                        },
                    );
                    groups_changed = true;
                }
            }
        }

        let mut inherit_group_color = true;
        let group_color_rgb: Option<[u8; 3]> = selected_group.and_then(|gid| {
            groups_cfg
                .channel_groups
                .iter()
                .find(|g| g.id == gid)
                .map(|g| g.color_rgb)
        });
        if let Some(m) = groups_cfg.channel_members.get(selected_name.as_str()) {
            inherit_group_color = m.inherit_color;
        }

        if let Some(gid) = selected_group {
            ui.horizontal(|ui| {
                ui.add_enabled_ui(
                    groups_cfg
                        .channel_members
                        .contains_key(selected_name.as_str()),
                    |ui| {
                        if ui
                            .checkbox(&mut inherit_group_color, "Inherit group color")
                            .changed()
                        {
                            if let Some(m) =
                                groups_cfg.channel_members.get_mut(selected_name.as_str())
                            {
                                m.inherit_color = inherit_group_color;
                                groups_changed = true;
                            }
                        }
                    },
                );
                if inherit_group_color {
                    if let Some(group) = groups_cfg.channel_groups.iter_mut().find(|g| g.id == gid)
                    {
                        ui.add_space(8.0);
                        ui.label("Group color");
                        let mut c = egui::Color32::from_rgb(
                            group.color_rgb[0],
                            group.color_rgb[1],
                            group.color_rgb[2],
                        );
                        if ui.color_edit_button_srgba(&mut c).changed() {
                            group.color_rgb = [c.r(), c.g(), c.b()];
                            groups_changed = true;
                        }
                    }
                }
            });
        }

        // Channel color (override or ungrouped).
        let allow_channel_color = selected_group.is_none() || !inherit_group_color;
        if let Some(ch) = self.channels.get(selected_channel) {
            let mut c = egui::Color32::from_rgb(ch.color_rgb[0], ch.color_rgb[1], ch.color_rgb[2]);
            let mut changed_color = false;
            ui.horizontal(|ui| {
                ui.label(if allow_channel_color {
                    "Color"
                } else {
                    "Color (override)"
                });
                ui.add_enabled_ui(allow_channel_color, |ui| {
                    changed_color = ui.color_edit_button_srgba(&mut c).changed();
                });
            });
            if changed_color {
                if let Some(dst) = self.channels.get_mut(selected_channel) {
                    dst.color_rgb = [c.r(), c.g(), c.b()];
                }
                self.bump_render_id();
            } else if !allow_channel_color {
                if let Some(rgb) = group_color_rgb {
                    ui.label(format!(
                        "Using group color: rgb({}, {}, {})",
                        rgb[0], rgb[1], rgb[2]
                    ));
                }
            }
        }

        if groups_changed {
            let new_groups = groups_cfg;
            self.commit_current_channel_groups(new_groups);
            self.bump_render_id();
        }

        let window = selected_info.window.unwrap_or((0.0, abs_max));
        let out = contrast::ui_contrast_window(
            ui,
            abs_max,
            window,
            contrast::ContrastUiOptions::standard("Set Max -> All"),
        );
        let (lo, hi) = out.window;

        if out.set_max_all_clicked {
            let new_hi = hi;
            for dst in &mut self.channels {
                let (mut dlo, _) = dst.window.unwrap_or((0.0, abs_max));
                dlo = dlo.clamp(0.0, abs_max);
                let dhi = new_hi.clamp(0.0, abs_max);
                let dlo = if dhi <= dlo {
                    (dhi - 1.0).clamp(0.0, abs_max)
                } else {
                    dlo
                };
                dst.window = Some((dlo, dhi));
                self.channel_window_overrides
                    .insert(dst.name.clone(), (dlo, dhi));
            }
            self.bump_render_id();
        }

        if out.limits_touched {
            if let Some(dst) = self.channels.get_mut(selected_channel) {
                dst.window = Some((lo, hi));
            }
            self.channel_window_overrides
                .insert(selected_name, (lo, hi));
            self.bump_render_id();
        }

        ui.separator();
        self.ui_histogram(ui, abs_max, (lo, hi));

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Threshold Regions");
            if crate::ui::help::help_button(ui, crate::ui::help::HelpTopic::Thresholding) {
                self.active_help_topic = Some(crate::ui::help::HelpTopic::Thresholding);
            }
        });
        ui.collapsing("Controls", |ui| {
            if !self.view_plane_is_xy() {
                ui.label("Threshold-region preview is only available in XY view.");
                return;
            }
            ui.label(
                "Capture pixels from the active channel, preview the thresholded raster on the canvas, then apply it as a mask layer.",
            );
            if self.threshold_region_preview.is_none() {
                ui.horizontal(|ui| {
                    ui.label("Scope");
                    ui.selectable_value(
                        &mut self.threshold_region_scope,
                        ThresholdRegionScope::VisibleRegion,
                        "Visible region",
                    );
                    ui.selectable_value(
                        &mut self.threshold_region_scope,
                        ThresholdRegionScope::EntireImage,
                        "Entire image",
                    );
                });

                let mut start_enabled = true;
                let mut start_label = match self.threshold_region_scope {
                    ThresholdRegionScope::VisibleRegion => "Start threshold preview from visible region".to_string(),
                    ThresholdRegionScope::EntireImage => "Start threshold preview from entire image".to_string(),
                };
                if self.threshold_region_scope == ThresholdRegionScope::EntireImage {
                    self.ensure_threshold_region_full_level_default();
                    let max_level = self.dataset.levels.len().saturating_sub(1);
                    self.threshold_region_full_level =
                        self.threshold_region_full_level.min(max_level);
                    ui.horizontal(|ui| {
                        ui.label("Level");
                        egui::ComboBox::from_id_salt("threshold-region-full-level")
                            .selected_text(format!("Level {}", self.threshold_region_full_level))
                            .show_ui(ui, |ui| {
                                for level in &self.dataset.levels {
                                    let label = self
                                        .threshold_region_full_level_summary(level.index)
                                        .map(|(width, height, pixels)| {
                                            let suffix = if pixels
                                                > THRESHOLD_REGION_MAX_INTERACTIVE_PIXELS
                                            {
                                                " - too large"
                                            } else {
                                                ""
                                            };
                                            format!(
                                                "Level {}: {} x {} ({} px){}",
                                                level.index, width, height, pixels, suffix
                                            )
                                        })
                                        .unwrap_or_else(|| format!("Level {}", level.index));
                                    ui.selectable_value(
                                        &mut self.threshold_region_full_level,
                                        level.index,
                                        label,
                                    );
                                }
                            });
                    });
                    if let Some((width, height, pixels)) =
                        self.threshold_region_full_level_summary(self.threshold_region_full_level)
                    {
                        ui.label(format!(
                            "Preview size: {width} x {height} ({pixels} pixels)."
                        ));
                        if pixels > THRESHOLD_REGION_MAX_INTERACTIVE_PIXELS {
                            start_enabled = false;
                            start_label = "Choose a coarser level".to_string();
                            ui.label(format!(
                                "Whole-image thresholding at this level would read {pixels} pixels; choose a coarser level."
                            ));
                        }
                    } else {
                        start_enabled = false;
                        start_label = "Invalid level".to_string();
                        ui.label("Whole-image thresholding is unavailable for this level.");
                    }
                }

                if ui
                    .add_enabled(start_enabled, egui::Button::new(start_label))
                    .clicked()
                {
                    self.threshold_region_status.clear();
                    if self.control_actor_threshold_generation > 0 {
                        let scope = match self.threshold_region_scope {
                            ThresholdRegionScope::VisibleRegion => "visible",
                            ThresholdRegionScope::EntireImage => "entire_image",
                        };
                        self.native_control_intents.push(NativeControlIntent {
                            method: "viewer.thresholds.preview.start",
                            params: serde_json::json!({
                                "scope":scope,
                                "level":self.threshold_region_full_level,
                                "min_component_pixels":self.threshold_region_min_pixels,
                                "channel":self.selected_channel,
                            }),
                        });
                    } else if let Err(err) = self.start_threshold_region_preview(ctx) {
                        self.threshold_region_status = format!("Threshold regions failed: {err}");
                    }
                }
            } else {
                let (
                    channel_name,
                    level_index,
                    width,
                    height,
                    plane_min,
                    plane_max,
                    mut threshold,
                    current_min_pixels,
                ) = {
                    let preview = self.threshold_region_preview.as_ref().expect("preview exists");
                    let plane_min = preview.plane.iter().copied().min().unwrap_or(0);
                    let plane_max = preview.plane.iter().copied().max().unwrap_or(0);
                    (
                        preview.channel_name.clone(),
                        preview.level_index,
                        preview.mask.width,
                        preview.mask.height,
                        plane_min,
                        plane_max.max(plane_min),
                        preview.threshold,
                        preview.min_component_pixels,
                    )
                };
                ui.label(format!(
                    "Previewing {channel_name} at level {level_index} ({width} x {height} px)."
                ));
                let threshold_changed = if plane_max > plane_min {
                    ui.add(
                        egui::Slider::new(&mut threshold, plane_min..=plane_max)
                            .text("Threshold")
                            .clamping(egui::SliderClamping::Always),
                    )
                    .changed()
                } else {
                    ui.label(format!("Threshold: {threshold}"));
                    false
                };
                let mut min_pixels = current_min_pixels;
                let min_pixels_changed = ui
                    .horizontal(|ui| {
                        ui.label("Min component pixels");
                        ui.add(
                            egui::DragValue::new(&mut min_pixels)
                                .range(1..=1_000_000)
                                .speed(1.0),
                        )
                        .changed()
                    })
                    .inner;
                let mut preview_changed = false;
                if threshold_changed {
                    if let Some(preview) = self.threshold_region_preview.as_mut() {
                        preview.threshold = threshold;
                    }
                    preview_changed = true;
                }
                if min_pixels_changed && min_pixels != self.threshold_region_min_pixels {
                    self.threshold_region_min_pixels = min_pixels;
                    preview_changed = true;
                }
                if preview_changed {
                    if self.control_actor_threshold_generation > 0 {
                        self.native_control_intents.push(NativeControlIntent {
                            method: "viewer.thresholds.preview.configure",
                            params: serde_json::json!({
                                "threshold":threshold,
                                "min_component_pixels":min_pixels,
                            }),
                        });
                    } else {
                        self.recompute_threshold_region_preview(ctx);
                    }
                }

                ui.horizontal(|ui| {
                    if ui.button("Refresh preview").clicked() {
                        self.threshold_region_status.clear();
                        if self.control_actor_threshold_generation > 0 {
                            self.native_control_intents.push(NativeControlIntent {
                                method: "viewer.thresholds.preview.refresh",
                                params: serde_json::json!({}),
                            });
                        } else if let Err(err) = self.start_threshold_region_preview(ctx) {
                            self.threshold_region_status =
                                format!("Threshold regions failed: {err}");
                        }
                    }
                    if ui.button("Apply mask from preview").clicked() {
                        self.threshold_region_status.clear();
                        if self.control_actor_threshold_generation > 0 {
                            self.native_control_intents.push(NativeControlIntent {
                                method: "viewer.thresholds.preview.apply",
                                params: serde_json::json!({"sync_project":true}),
                            });
                        } else if let Err(err) = self.create_threshold_mask_from_preview() {
                            self.threshold_region_status =
                                format!("Threshold regions failed: {err}");
                        }
                    }
                    if ui.button("Cancel preview").clicked() {
                        if self.control_actor_threshold_generation > 0 {
                            self.native_control_intents.push(NativeControlIntent {
                                method: "viewer.thresholds.preview.cancel",
                                params: serde_json::json!({}),
                            });
                        }
                        self.threshold_region_preview = None;
                        self.threshold_region_status.clear();
                    }
                });
                ui.small(
                    "The canvas overlay is a raster preview. The pixel grid appears automatically when you zoom in far enough.",
                );
            }
            if self.threshold_region_preview.is_none() {
                ui.horizontal(|ui| {
                    ui.label("Min component pixels");
                    ui.add(
                        egui::DragValue::new(&mut self.threshold_region_min_pixels)
                            .range(1..=1_000_000)
                            .speed(1.0),
                    );
                });
            }
            if !self.threshold_region_status.is_empty() {
                ui.label(self.threshold_region_status.clone());
            }
        });

        let changed_note = if let Some(ch) = self.channels.get_mut(selected_channel) {
            let channel_name = ch.name.clone();
            channel_notes::ui_channel_notes(ui, &channel_name, &mut ch.note)
                .then(|| (selected_channel, ch.note.clone()))
        } else {
            None
        };
        if let Some((channel, note)) = changed_note {
            self.native_control_intents.push(NativeControlIntent {
                method: "viewer.channels.set_note",
                params: serde_json::json!({"channel": channel, "note": note}),
            });
        }
    }

    pub(super) fn ui_histogram(&mut self, ui: &mut egui::Ui, abs_max: f32, limits: (f32, f32)) {
        let (rect, _response) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), 140.0),
            egui::Sense::hover(),
        );
        ui.painter()
            .rect_filled(rect, 0.0, egui::Color32::from_gray(18));

        let Some(hist) = self.hist.as_ref() else {
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Histogram: (loading...)".to_string(),
                egui::FontId::proportional(12.0),
                egui::Color32::from_gray(200),
            );
            return;
        };
        let histogram_stale =
            hist.request_id != self.hist_request_id || self.hist_dirty || self.hist_request_pending;

        let bins = &hist.bins;
        if bins.is_empty() {
            return;
        }
        let max_count = bins.iter().copied().max().unwrap_or(1).max(1) as f32;

        let w = rect.width().max(1.0);
        let h = rect.height().max(1.0);
        let bin_w = w / bins.len() as f32;
        for (i, &c) in bins.iter().enumerate() {
            let x0 = rect.left() + i as f32 * bin_w;
            let x1 = x0 + bin_w;
            let frac = (c as f32) / max_count;
            let y1 = rect.bottom();
            let y0 = y1 - frac * h;
            let r = egui::Rect::from_min_max(egui::pos2(x0, y0), egui::pos2(x1, y1));
            ui.painter()
                .rect_filled(r, 0.0, egui::Color32::from_gray(90));
        }

        let (lo, hi) = limits;
        let x_lo = rect.left() + (lo / abs_max.clamp(1.0, f32::MAX)) * w;
        let x_hi = rect.left() + (hi / abs_max.clamp(1.0, f32::MAX)) * w;
        ui.painter().line_segment(
            [
                egui::pos2(x_lo, rect.top()),
                egui::pos2(x_lo, rect.bottom()),
            ],
            egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 80, 80)),
        );
        ui.painter().line_segment(
            [
                egui::pos2(x_hi, rect.top()),
                egui::pos2(x_hi, rect.bottom()),
            ],
            egui::Stroke::new(2.0, egui::Color32::from_rgb(80, 255, 80)),
        );

        let stats_text = if let Some(s) = hist.stats.as_ref() {
            format!(
                "Min: {:.0} | Q1: {:.0} | Median: {:.0} | Q3: {:.0} | Max: {:.0} (n={})",
                s.min, s.q1, s.median, s.q3, s.max, s.n
            )
        } else {
            "Min: - | Q1: - | Median: - | Q3: - | Max: -".to_string()
        };
        ui.add_space(4.0);
        ui.label(stats_text);
        if histogram_stale {
            ui.painter().text(
                rect.right_top() + egui::vec2(-8.0, 8.0),
                egui::Align2::RIGHT_TOP,
                "updating...",
                egui::FontId::proportional(11.0),
                egui::Color32::from_gray(170),
            );
        }
    }
}
