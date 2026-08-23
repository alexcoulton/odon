use super::*;

impl OmeZarrViewerApp {
    pub(in crate::app) fn channel_indices_in_group(&self, group_id: u64) -> Vec<usize> {
        let groups = self.current_layer_groups();
        self.channel_layer_order
            .iter()
            .copied()
            .filter(|&idx| {
                self.channels.get(idx).is_some_and(|ch| {
                    groups
                        .channel_members
                        .get(ch.name.as_str())
                        .is_some_and(|m| m.group_id == group_id)
                })
            })
            .collect()
    }

    pub(in crate::app) fn group_contrast_window_for_indices(
        &self,
        indices: &[usize],
        abs_max: f32,
    ) -> Option<((f32, f32), bool)> {
        let mut first_window: Option<(f32, f32)> = None;
        let mut mixed = false;
        for &idx in indices {
            let Some(ch) = self.channels.get(idx) else {
                continue;
            };
            let window = ch.window.unwrap_or((0.0, abs_max));
            if let Some(prev) = first_window {
                if (prev.0 - window.0).abs() > 1e-6 || (prev.1 - window.1).abs() > 1e-6 {
                    mixed = true;
                }
            } else {
                first_window = Some(window);
            }
        }
        first_window.map(|window| (window, mixed))
    }

    pub(in crate::app) fn visible_channel_indices(&self) -> Vec<usize> {
        self.channel_layer_order
            .iter()
            .copied()
            .filter(|&idx| self.channels.get(idx).is_some_and(|ch| ch.visible))
            .collect()
    }

    pub(in crate::app) fn quick_contrast_target_options(
        &self,
    ) -> Vec<top_bar::QuickContrastTargetOption> {
        let visible_count = self.visible_channel_indices().len();
        let group_count = self
            .selected_channel_group_id
            .map(|group_id| self.channel_indices_in_group(group_id).len())
            .unwrap_or(0);
        let group_label = self
            .selected_channel_group_id
            .and_then(|group_id| {
                self.current_layer_groups()
                    .channel_groups
                    .iter()
                    .find(|group| group.id == group_id)
                    .map(|group| format!("Selected group ({})", group.name))
            })
            .unwrap_or_else(|| "Selected group".to_string());

        vec![
            top_bar::QuickContrastTargetOption {
                target: top_bar::QuickContrastTarget::Visible,
                label: format!("Visible channels ({visible_count})"),
                enabled: visible_count > 0,
            },
            top_bar::QuickContrastTargetOption {
                target: top_bar::QuickContrastTarget::Active,
                label: "Active channel".to_string(),
                enabled: !self.channels.is_empty(),
            },
            top_bar::QuickContrastTargetOption {
                target: top_bar::QuickContrastTarget::SelectedGroup,
                label: format!("{group_label} ({group_count})"),
                enabled: group_count > 0,
            },
        ]
    }

    pub(in crate::app) fn quick_contrast_indices_for_target(
        &self,
        target: top_bar::QuickContrastTarget,
    ) -> Vec<usize> {
        match target {
            top_bar::QuickContrastTarget::Active => {
                if self.channels.is_empty() {
                    Vec::new()
                } else {
                    vec![self.selected_channel.min(self.channels.len() - 1)]
                }
            }
            top_bar::QuickContrastTarget::Visible => {
                let visible = self.visible_channel_indices();
                if visible.is_empty() {
                    self.quick_contrast_indices_for_target(top_bar::QuickContrastTarget::Active)
                } else {
                    visible
                }
            }
            top_bar::QuickContrastTarget::SelectedGroup => self
                .selected_channel_group_id
                .map(|group_id| self.channel_indices_in_group(group_id))
                .filter(|indices| !indices.is_empty())
                .unwrap_or_else(|| {
                    self.quick_contrast_indices_for_target(top_bar::QuickContrastTarget::Visible)
                }),
        }
    }

    pub(in crate::app) fn apply_channel_window_to_indices(
        &mut self,
        indices: &[usize],
        lo: f32,
        hi: f32,
    ) {
        let abs_max = self.dataset.abs_max.max(1.0);
        let lo = lo.clamp(0.0, abs_max);
        let hi = hi.clamp(0.0, abs_max);
        let (lo, hi) = if hi <= lo {
            ((hi - 1.0).clamp(0.0, abs_max), hi)
        } else {
            (lo, hi)
        };
        if self.native_viewport_actor_owned() {
            let targets = indices.iter().copied().collect::<HashSet<_>>();
            let mut state = self.control_native_layer_snapshot_list();
            let mut changed = false;
            for layer in state.as_array_mut().into_iter().flatten() {
                let Some(_) = layer
                    .get("layer_id")
                    .and_then(serde_json::Value::as_str)
                    .and_then(|id| id.strip_prefix("channel:"))
                    .and_then(|index| index.parse::<usize>().ok())
                    .filter(|index| targets.contains(index))
                else {
                    continue;
                };
                let window = serde_json::json!({"min":lo,"max":hi});
                if layer["presentation"]["window"] != window {
                    layer["presentation"]["window"] = window;
                    changed = true;
                }
            }
            if changed {
                self.submit_native_layer_state_replace(state);
            }
            return;
        }
        let mut changed = false;
        for &idx in indices {
            if let Some(dst) = self.channels.get_mut(idx) {
                dst.window = Some((lo, hi));
                self.channel_window_overrides
                    .insert(dst.name.clone(), (lo, hi));
                changed = true;
            }
        }
        if changed {
            self.hist_dirty = true;
            self.bump_render_id();
        }
    }

    pub(in crate::app) fn apply_three_channel_rgb_preset(&mut self) -> bool {
        if self.channels.len() != 3 {
            return false;
        }
        let rgb = [[255, 0, 0], [0, 255, 0], [0, 0, 255]];
        let hi = self.dataset.abs_max.clamp(1.0, 255.0);
        if self.native_viewport_actor_owned() {
            let mut state = self.control_native_layer_snapshot_list();
            for layer in state.as_array_mut().into_iter().flatten() {
                let Some(index) = layer
                    .get("layer_id")
                    .and_then(serde_json::Value::as_str)
                    .and_then(|id| id.strip_prefix("channel:"))
                    .and_then(|index| index.parse::<usize>().ok())
                    .filter(|index| *index < 3)
                else {
                    layer["active"] = serde_json::json!(false);
                    continue;
                };
                layer["visible"] = serde_json::json!(true);
                layer["active"] = serde_json::json!(index == 0);
                layer["presentation"]["visible"] = serde_json::json!(true);
                layer["presentation"]["color_rgb"] = serde_json::json!(rgb[index]);
                layer["presentation"]["window"] = serde_json::json!({"min":0.0,"max":hi});
            }
            let mut groups = self.current_layer_groups();
            for channel in &self.channels {
                if let Some(member) = groups.channel_members.get_mut(&channel.name) {
                    member.inherit_color = false;
                }
            }
            self.persist_current_layer_groups(groups.clone());
            self.selected_channel_layers.extend(0..3);
            self.memory_selected_channels.extend(0..3);
            self.channel_select_anchor_idx = Some(0);
            self.selected_channel_group_id = None;
            self.set_status("Applying RGB preset to channels 0-2...");
            return self.submit_native_layer_state_replace_with_groups(state, &groups);
        }
        let mut changed = false;
        for (idx, color) in rgb.into_iter().enumerate() {
            let Some(channel) = self.channels.get_mut(idx) else {
                continue;
            };
            changed |= channel.color_rgb != color;
            channel.color_rgb = color;
            changed |= !channel.visible;
            channel.visible = true;
            let window = (0.0, hi);
            changed |= channel.window != Some(window);
            channel.window = Some(window);
            self.channel_window_overrides
                .insert(channel.name.clone(), window);
            self.selected_channel_layers.insert(idx);
            self.memory_selected_channels.insert(idx);
            self.set_channel_group_color_inheritance(idx, false);
        }
        if !changed {
            return false;
        }
        self.selected_channel = 0;
        self.active_layer = LayerId::Channel(0);
        self.selected_channel_group_id = None;
        self.channel_select_anchor_idx = Some(0);
        self.hist_dirty = true;
        self.bump_render_id();
        self.set_status("Applied RGB preset to channels 0-2.");
        true
    }

    pub(in crate::app) fn ui_top_bar_quick_contrast(&mut self, ui: &mut egui::Ui) {
        if self.channels.is_empty() {
            return;
        }
        if self.quick_contrast_target == top_bar::QuickContrastTarget::SelectedGroup
            && self
                .selected_channel_group_id
                .map(|group_id| self.channel_indices_in_group(group_id).is_empty())
                .unwrap_or(true)
        {
            self.quick_contrast_target = top_bar::QuickContrastTarget::Visible;
        }

        let options = self.quick_contrast_target_options();
        let indices = self.quick_contrast_indices_for_target(self.quick_contrast_target);
        if indices.is_empty() {
            return;
        }
        let abs_max = self.dataset.abs_max.max(1.0);
        let ((window, mixed), reference_idx) = (
            self.group_contrast_window_for_indices(&indices, abs_max)
                .unwrap_or(((0.0, abs_max), false)),
            self.selected_channel.min(self.channels.len() - 1),
        );
        let reference_name = self
            .channels
            .get(reference_idx)
            .map(|channel| channel.name.clone())
            .unwrap_or_else(|| "channel".to_string());
        let target_before = self.quick_contrast_target;
        let response = top_bar::ui_quick_contrast(
            ui,
            top_bar::QuickContrastParams {
                abs_max,
                target: &mut self.quick_contrast_target,
                target_options: &options,
                target_count: indices.len(),
                reference_channel_name: &reference_name,
                window,
                mixed,
                step: 1.0,
                id_salt: "top-quick-contrast",
            },
        );
        if response.changed && self.quick_contrast_target == target_before {
            let target_indices = self.quick_contrast_indices_for_target(self.quick_contrast_target);
            self.apply_channel_window_to_indices(
                &target_indices,
                response.window.0,
                response.window.1,
            );
        }
    }

    pub(in crate::app) fn ui_group_contrast(
        &mut self,
        _ctx: &egui::Context,
        ui: &mut egui::Ui,
        group_id: u64,
    ) {
        let abs_max = self.dataset.abs_max.max(1.0);
        let Some(group) = self
            .current_layer_groups()
            .channel_groups
            .iter()
            .find(|g| g.id == group_id)
            .cloned()
        else {
            self.selected_channel_group_id = None;
            ui.label("Selected channel group no longer exists.");
            return;
        };

        let members = self.channel_indices_in_group(group_id);
        ui.heading("Contrast");
        ui.label(format!("Group: {}", group.name));
        ui.label(format!("Applies to {} channel(s).", members.len()));

        if members.is_empty() {
            ui.label("This group has no channels.");
            return;
        }

        let Some((window, mixed)) = self.group_contrast_window_for_indices(&members, abs_max)
        else {
            ui.label("No channels available in this group.");
            return;
        };
        if mixed {
            ui.label("Group channels currently have mixed contrast limits. Applying changes here will overwrite them.");
        }

        let out = contrast::ui_contrast_window(
            ui,
            abs_max,
            window,
            contrast::ContrastUiOptions::standard("Set Max -> Group"),
        );
        let (lo, hi) = out.window;

        if out.set_max_all_clicked {
            for &idx in &members {
                if let Some(dst) = self.channels.get_mut(idx) {
                    let (mut dlo, _) = dst.window.unwrap_or((0.0, abs_max));
                    dlo = dlo.clamp(0.0, abs_max);
                    let dhi = hi.clamp(0.0, abs_max);
                    let dlo = if dhi <= dlo {
                        (dhi - 1.0).clamp(0.0, abs_max)
                    } else {
                        dlo
                    };
                    dst.window = Some((dlo, dhi));
                    self.channel_window_overrides
                        .insert(dst.name.clone(), (dlo, dhi));
                }
            }
            self.bump_render_id();
            return;
        }

        if out.limits_touched {
            for &idx in &members {
                if let Some(dst) = self.channels.get_mut(idx) {
                    dst.window = Some((lo, hi));
                    self.channel_window_overrides
                        .insert(dst.name.clone(), (lo, hi));
                }
            }
            self.bump_render_id();
        }
    }
}
