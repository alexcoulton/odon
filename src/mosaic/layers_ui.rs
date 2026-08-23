//! Layer state, quick contrast, grouping dialogs, annotations, and screenshot dialogs.

use super::*;

impl MosaicViewerApp {
    pub(super) fn set_active_layer(&mut self, id: MosaicLayerId) {
        match id {
            MosaicLayerId::Channel(index) => {
                self.submit_native_control_intent(
                    "viewer.channels.set_active",
                    serde_json::json!({"index":index}),
                );
            }
            _ => {
                self.submit_native_control_intent(
                    "viewer.native_layers.set_active",
                    serde_json::json!({"layer_id":Self::layer_id_storage_key(id)}),
                );
            }
        }
    }

    pub(super) fn apply_active_layer_projection(&mut self, id: MosaicLayerId) {
        self.active_layer = id;
        if let MosaicLayerId::Channel(idx) = id {
            self.selected_channel = idx.min(self.channels.len().saturating_sub(1));
        } else {
            self.selected_channel_group_id = None;
        }
    }

    pub(super) fn channel_indices_in_group(&self, group_id: u64) -> Vec<usize> {
        self.channel_layer_order
            .iter()
            .copied()
            .filter(|&idx| {
                self.channels.get(idx).is_some_and(|ch| {
                    self.layer_groups
                        .channel_members
                        .get(ch.name.as_str())
                        .is_some_and(|m| m.group_id == group_id)
                })
            })
            .collect()
    }

    pub(super) fn group_contrast_window_for_indices(
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

    pub(super) fn visible_channel_indices(&self) -> Vec<usize> {
        self.channel_layer_order
            .iter()
            .copied()
            .filter(|&idx| self.channels.get(idx).is_some_and(|ch| ch.visible))
            .collect()
    }

    pub(super) fn quick_contrast_target_options(&self) -> Vec<top_bar::QuickContrastTargetOption> {
        let visible_count = self.visible_channel_indices().len();
        let group_count = self
            .selected_channel_group_id
            .map(|group_id| self.channel_indices_in_group(group_id).len())
            .unwrap_or(0);
        let group_label = self
            .selected_channel_group_id
            .and_then(|group_id| {
                self.layer_groups
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

    pub(super) fn quick_contrast_indices_for_target(
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

    pub(super) fn apply_channel_window_to_indices(&mut self, indices: &[usize], lo: f32, hi: f32) {
        let abs_max = self.abs_max.max(1.0);
        let lo = lo.clamp(0.0, abs_max);
        let hi = hi.clamp(0.0, abs_max);
        let (lo, hi) = if hi <= lo {
            ((hi - 1.0).clamp(0.0, abs_max), hi)
        } else {
            (lo, hi)
        };
        for &index in indices {
            self.submit_native_control_intent(
                "viewer.channels.set_contrast",
                serde_json::json!({"index":index,"min":lo,"max":hi}),
            );
        }
    }

    pub(super) fn commit_channel_color(&mut self, index: usize, color_rgb: [u8; 3]) {
        self.submit_native_control_intent(
            "viewer.channels.set_color",
            serde_json::json!({"index":index,"color_rgb":color_rgb}),
        );
    }

    pub(super) fn commit_channel_note(&mut self, index: usize, note: String) {
        self.submit_native_control_intent(
            "viewer.channels.set_note",
            serde_json::json!({"index":index,"note":note}),
        );
    }

    pub(super) fn commit_layer_groups_preview(&mut self, before: ProjectLayerGroups) {
        let desired = self.layer_groups.clone();
        if serde_json::to_value(&desired).ok() == serde_json::to_value(&before).ok() {
            return;
        }
        self.layer_groups = before;
        self.submit_native_control_intent(
            "viewer.channels.set_group",
            serde_json::json!({"state":desired}),
        );
    }

    pub(super) fn ui_top_bar_quick_contrast(&mut self, ui: &mut egui::Ui) {
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
        let abs_max = self.abs_max.max(1.0);
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
                id_salt: "mosaic-top-quick-contrast",
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

    pub(super) fn ui_group_contrast(&mut self, ui: &mut egui::Ui, group_id: u64) {
        let abs_max = self.abs_max.max(1.0);
        let Some(group) = self
            .layer_groups
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
        ui.heading("Contrast (global)");
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
            let windows = members
                .iter()
                .filter_map(|&index| {
                    let (mut minimum, _) =
                        self.channels.get(index)?.window.unwrap_or((0.0, abs_max));
                    minimum = minimum.clamp(0.0, abs_max);
                    let maximum = hi.clamp(0.0, abs_max);
                    let minimum = if maximum <= minimum {
                        (maximum - 1.0).clamp(0.0, abs_max)
                    } else {
                        minimum
                    };
                    Some((index, minimum, maximum))
                })
                .collect::<Vec<_>>();
            for (index, minimum, maximum) in windows {
                self.apply_channel_window_to_indices(&[index], minimum, maximum);
            }
            ui.ctx().request_repaint();
            return;
        }

        if out.limits_touched {
            self.apply_channel_window_to_indices(&members, lo, hi);
            ui.ctx().request_repaint();
        }
    }

    pub(super) fn add_annotation_layer(&mut self) {
        self.submit_native_control_intent(
            "viewer.annotations.layers.create",
            serde_json::json!({}),
        );
    }

    pub(super) fn layer_display_name(&self, id: MosaicLayerId) -> String {
        match id {
            MosaicLayerId::TextLabels => "Text labels".to_string(),
            MosaicLayerId::SegmentationGeoJson => "Segmentation Objects".to_string(),
            MosaicLayerId::Annotation(id) => self
                .annotation_layers
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.name.clone())
                .unwrap_or_else(|| format!("Annotations {id}")),
            MosaicLayerId::Channel(idx) => self
                .channels
                .get(idx)
                .map(|c| c.name.clone())
                .unwrap_or_else(|| format!("Channel {idx}")),
        }
    }

    pub(super) fn layer_icon(&self, id: MosaicLayerId) -> Icon {
        match id {
            MosaicLayerId::Channel(_) => Icon::Image,
            MosaicLayerId::SegmentationGeoJson => Icon::Polygon,
            MosaicLayerId::TextLabels => Icon::Text,
            MosaicLayerId::Annotation(_) => Icon::Points,
        }
    }

    pub(super) fn layer_available(&self, id: MosaicLayerId) -> bool {
        match id {
            MosaicLayerId::TextLabels => true,
            MosaicLayerId::SegmentationGeoJson => self.seg_geojson.has_any_segpaths(),
            MosaicLayerId::Annotation(_) => true,
            MosaicLayerId::Channel(idx) => idx < self.channels.len(),
        }
    }

    pub(super) fn layer_visible_value(&self, id: MosaicLayerId) -> Option<bool> {
        match id {
            MosaicLayerId::TextLabels => Some(self.show_text_labels),
            MosaicLayerId::SegmentationGeoJson => Some(self.seg_geojson.visible),
            MosaicLayerId::Annotation(id) => self
                .annotation_layers
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.visible),
            MosaicLayerId::Channel(idx) => self.channels.get(idx).map(|c| c.visible),
        }
    }

    pub(super) fn set_layer_visible(&mut self, id: MosaicLayerId, visible: bool) {
        match id {
            MosaicLayerId::TextLabels => {
                self.submit_layout_value("show_text_labels", serde_json::json!(visible));
            }
            MosaicLayerId::SegmentationGeoJson => {
                self.submit_native_control_intent(
                    "viewer.objects.set_visibility",
                    serde_json::json!({"target":"objects","visible":visible}),
                );
            }
            MosaicLayerId::Annotation(_) => {
                self.submit_native_control_intent(
                    "viewer.native_layers.set_visibility",
                    serde_json::json!({
                        "layer_id":Self::layer_id_storage_key(id),
                        "visible":visible,
                    }),
                );
            }
            MosaicLayerId::Channel(index) => {
                self.submit_native_control_intent(
                    "viewer.channels.set_visible",
                    serde_json::json!({
                        "channels":[index],
                        "mode":if visible { "show" } else { "hide" },
                    }),
                );
            }
        }
    }

    pub(super) fn apply_layer_visibility_projection(&mut self, id: MosaicLayerId, visible: bool) {
        match id {
            MosaicLayerId::TextLabels => self.show_text_labels = visible,
            MosaicLayerId::SegmentationGeoJson => self.seg_geojson.visible = visible,
            MosaicLayerId::Annotation(id) => {
                if let Some(l) = self.annotation_layers.iter_mut().find(|l| l.id == id) {
                    l.visible = visible;
                }
            }
            MosaicLayerId::Channel(idx) => {
                if let Some(ch) = self.channels.get_mut(idx) {
                    ch.visible = visible;
                }
            }
        }
    }

    pub(super) fn ui_layers(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let groups_before = self.layer_groups.clone();
        let channel_search_before = self.channel_list_search.clone();
        ui.heading("Layers");
        ui.separator();

        // Mosaic layers are mostly shared overlays plus globally visible channels. The layer list
        // is therefore less about per-item state and more about shared visibility/group controls.
        // Overlays master visibility toggle.
        let overlay_ids = self.overlay_layer_order.clone();
        let mut overlays_all = true;
        let mut overlays_none = true;
        for id in overlay_ids.iter().copied() {
            if !self.layer_available(id) {
                continue;
            }
            if let Some(v) = self.layer_visible_value(id) {
                overlays_all &= v;
                overlays_none &= !v;
            }
        }
        let overlays_mixed = !overlays_all && !overlays_none;
        ui.horizontal(|ui| {
            ui.label("Overlays");
            ui.add_space(4.0);
            let mut all = overlays_all;
            if ui
                .add(egui::Checkbox::new(&mut all, "All").indeterminate(overlays_mixed))
                .changed()
            {
                for id in overlay_ids.iter().copied() {
                    if !self.layer_available(id) {
                        continue;
                    }
                    self.set_layer_visible(id, all);
                }
            }
            if ui.button("+ Annotations").clicked() {
                self.add_annotation_layer();
            }
        });

        let mut ann_members_by_group: HashMap<u64, Vec<u64>> = HashMap::new();
        for id in self.overlay_layer_order.iter().copied() {
            let MosaicLayerId::Annotation(aid) = id else {
                continue;
            };
            let Some(m) = self.layer_groups.annotation_members.get(&aid) else {
                continue;
            };
            if self
                .layer_groups
                .annotation_groups
                .iter()
                .any(|g| g.id == m.group_id)
            {
                ann_members_by_group
                    .entry(m.group_id)
                    .or_default()
                    .push(aid);
            }
        }
        let mut ann_headers_shown: HashSet<u64> = HashSet::new();
        let mut delete_ann_group: Option<u64> = None;

        for i in 0..self.overlay_layer_order.len() {
            let id = self.overlay_layer_order[i];

            if let MosaicLayerId::Annotation(aid) = id {
                if let Some(m) = self.layer_groups.annotation_members.get(&aid) {
                    let gid = m.group_id;
                    if self
                        .layer_groups
                        .annotation_groups
                        .iter()
                        .any(|g| g.id == gid)
                    {
                        if !ann_headers_shown.contains(&gid) {
                            ann_headers_shown.insert(gid);
                            let Some(group_idx) = self
                                .layer_groups
                                .annotation_groups
                                .iter()
                                .position(|g| g.id == gid)
                            else {
                                continue;
                            };
                            let members = ann_members_by_group
                                .get(&gid)
                                .map(|v| v.as_slice())
                                .unwrap_or(&[]);
                            let (
                                mut group_name,
                                mut group_expanded,
                                mut group_visible,
                                mut group_tint_rgb,
                                mut group_tint_strength,
                            ) = {
                                let g = &self.layer_groups.annotation_groups[group_idx];
                                (
                                    g.name.clone(),
                                    g.expanded,
                                    g.visible,
                                    g.tint_rgb,
                                    g.tint_strength,
                                )
                            };

                            let mut all = true;
                            let mut none = true;
                            for &mid in members {
                                let lid = MosaicLayerId::Annotation(mid);
                                if let Some(v) = self.layer_visible_value(lid) {
                                    all &= v;
                                    none &= !v;
                                }
                            }
                            let mixed = !members.is_empty() && !all && !none;

                            let header =
                                egui::collapsing_header::CollapsingState::load_with_default_open(
                                    ui.ctx(),
                                    ui.make_persistent_id(("mosaic-annotation-group", gid)),
                                    group_expanded,
                                )
                                .show_header(ui, |ui| {
                                    let mut set_all = all;
                                    ui.add_enabled_ui(!members.is_empty(), |ui| {
                                        if ui
                                            .add(
                                                egui::Checkbox::new(&mut set_all, "")
                                                    .indeterminate(mixed),
                                            )
                                            .on_hover_text("Toggle all annotation layers in group")
                                            .changed()
                                        {
                                            for &mid in members {
                                                self.set_layer_visible(
                                                    MosaicLayerId::Annotation(mid),
                                                    set_all,
                                                );
                                            }
                                            group_visible = set_all;
                                        }
                                    });
                                    ui.add_space(4.0);
                                    ui.label(group_name.clone());
                                });
                            let open = header.is_open();
                            let (_toggle, _hdr, _body) = header.body(|ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Name");
                                    ui.text_edit_singleline(&mut group_name);
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Visible");
                                    ui.checkbox(&mut group_visible, "");
                                    if ui.button("Delete group").clicked() {
                                        delete_ann_group = Some(gid);
                                    }
                                });
                                ui.horizontal(|ui| {
                                    let mut has_tint = group_tint_rgb.is_some();
                                    if ui.checkbox(&mut has_tint, "Tint").changed() {
                                        if has_tint && group_tint_rgb.is_none() {
                                            group_tint_rgb = Some([255, 255, 255]);
                                        }
                                        if !has_tint {
                                            group_tint_rgb = None;
                                        }
                                    }
                                    if let Some(rgb) = group_tint_rgb.as_mut() {
                                        let mut c = egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
                                        if ui.color_edit_button_srgba(&mut c).changed() {
                                            *rgb = [c.r(), c.g(), c.b()];
                                        }
                                    }
                                });
                                ui.add(
                                    egui::Slider::new(&mut group_tint_strength, 0.0..=1.0)
                                        .text("Tint strength")
                                        .clamping(egui::SliderClamping::Always),
                                );
                            });
                            group_expanded = open;

                            let g = &mut self.layer_groups.annotation_groups[group_idx];
                            g.name = group_name;
                            g.expanded = group_expanded;
                            g.visible = group_visible;
                            g.tint_rgb = group_tint_rgb;
                            g.tint_strength = group_tint_strength;
                        }

                        let expanded = self
                            .layer_groups
                            .annotation_groups
                            .iter()
                            .find(|g| g.id == gid)
                            .map(|g| g.expanded)
                            .unwrap_or(true);
                        if !expanded {
                            continue;
                        }
                    }
                }
            }

            let available = self.layer_available(id);
            let selected = self.active_layer == id || self.selected_overlay_layers.contains(&id);
            let icon = self.layer_icon(id);
            let name = self.layer_display_name(id);
            let visible = self.layer_visible_value(id);
            let resp = layer_list::ui_layer_row(
                ui,
                ctx,
                &mut self.layer_drag,
                layer_list::LayerGroup::Overlays,
                i,
                id,
                &name,
                layer_list::LayerRowOptions {
                    available,
                    selected,
                    icon,
                    visible,
                    color_rgb: None,
                    draggable: true,
                },
            );
            let mods = ctx.input(|i| i.modifiers);
            if resp.selected_clicked {
                if mods.shift && self.overlay_select_anchor_pos.is_some() {
                    let anchor = self.overlay_select_anchor_pos.unwrap_or(i);
                    let (a, b) = if anchor <= i {
                        (anchor, i)
                    } else {
                        (i, anchor)
                    };
                    self.selected_overlay_layers.clear();
                    for pos in a..=b {
                        if let Some(l) = self.overlay_layer_order.get(pos).copied() {
                            self.selected_overlay_layers.insert(l);
                        }
                    }
                } else if mods.command {
                    if !self.selected_overlay_layers.insert(id) {
                        self.selected_overlay_layers.remove(&id);
                    }
                    self.overlay_select_anchor_pos = Some(i);
                } else {
                    self.selected_overlay_layers.clear();
                    self.selected_overlay_layers.insert(id);
                    self.overlay_select_anchor_pos = Some(i);
                }
                self.set_active_layer(id);
            } else if resp.row_response.secondary_clicked() {
                if !self.selected_overlay_layers.contains(&id) {
                    self.selected_overlay_layers.clear();
                    self.selected_overlay_layers.insert(id);
                    self.overlay_select_anchor_pos = Some(i);
                    self.set_active_layer(id);
                }
            }
            if let Some(v) = resp.visible_changed {
                self.set_layer_visible(id, v);
            }

            resp.row_response.context_menu(|ui| {
                let selected_annotations: Vec<u64> = self
                    .selected_overlay_layers
                    .iter()
                    .filter_map(|l| match l {
                        MosaicLayerId::Annotation(a) => Some(*a),
                        _ => None,
                    })
                    .collect();
                let can_group = selected_annotations.len() >= 2
                    && selected_annotations.len() == self.selected_overlay_layers.len();
                if ui
                    .add_enabled(can_group, egui::Button::new("Group layers..."))
                    .clicked()
                {
                    self.open_group_layers_dialog_annotations(selected_annotations);
                    ui.close();
                }
            });
        }

        if let Some(group_id) = delete_ann_group {
            self.layer_groups
                .annotation_groups
                .retain(|g| g.id != group_id);
            self.layer_groups
                .annotation_members
                .retain(|_k, m| m.group_id != group_id);
        }

        ui.separator();
        channels_panel::show(self, ui, ctx);
        if self.channel_list_search != channel_search_before {
            let desired = self.channel_list_search.clone();
            self.channel_list_search = channel_search_before;
            self.submit_native_control_intent(
                "viewer.channels.presentation.set",
                serde_json::json!({"search":desired}),
            );
        }
        self.commit_layer_groups_preview(groups_before);

        layer_list::paint_drag_preview(ctx, self.layer_drag.as_ref(), |id| {
            self.layer_display_name(id)
        });
        let mut dropped: Option<(layer_list::LayerGroup, usize, usize)> = None;
        layer_list::finish_drag_if_released(ctx, &mut self.layer_drag, |group, from, to| {
            dropped = Some((group, from, to));
        });
        if let Some((group, from, to)) = dropped {
            let (stack, layers) = match group {
                layer_list::LayerGroup::Overlays => {
                    let mut order = self.overlay_layer_order.clone();
                    layer_list::reorder_vec(&mut order, from, to);
                    (
                        "overlays",
                        order
                            .into_iter()
                            .map(Self::layer_id_storage_key)
                            .collect::<Vec<_>>(),
                    )
                }
                layer_list::LayerGroup::Channels => {
                    let mut order = self.channel_layer_order.clone();
                    layer_list::reorder_vec(&mut order, from, to);
                    (
                        "channels",
                        order
                            .into_iter()
                            .map(|index| Self::layer_id_storage_key(MosaicLayerId::Channel(index)))
                            .collect::<Vec<_>>(),
                    )
                }
            };
            self.submit_native_control_intent(
                "viewer.native_layers.set_order",
                serde_json::json!({"stack":stack,"layers":layers}),
            );
        }
    }

    pub(super) fn open_group_layers_dialog_channels(&mut self, members: Vec<usize>) {
        let existing = self
            .layer_groups
            .channel_groups
            .iter()
            .map(|g| g.name.clone())
            .collect::<Vec<_>>();
        let default_name = default_group_name(existing);
        self.group_layers_dialog = Some(GroupLayersDialog::new(
            GroupLayersTarget::Channels(members),
            default_name,
        ));
    }

    pub(super) fn open_group_layers_dialog_annotations(&mut self, members: Vec<u64>) {
        let existing = self
            .layer_groups
            .annotation_groups
            .iter()
            .map(|g| g.name.clone())
            .collect::<Vec<_>>();
        let default_name = default_group_name(existing);
        self.group_layers_dialog = Some(GroupLayersDialog::new(
            GroupLayersTarget::Annotations(members),
            default_name,
        ));
    }

    pub(super) fn ui_group_layers_dialog(&mut self, ctx: &egui::Context) {
        let Some(dialog) = self.group_layers_dialog.as_mut() else {
            return;
        };

        let mut open = true;
        let mut accept = false;
        let mut cancel = false;

        egui::Window::new("Group layers")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .open(&mut open)
            .show(ctx, |ui| {
                ui.label("Group name");
                let mut name_output = egui::TextEdit::singleline(&mut dialog.name).show(ui);
                if dialog.focus_name_on_open {
                    name_output.response.request_focus();
                    name_output.state.cursor.set_char_range(Some(
                        egui::text::CCursorRange::select_all(&name_output.galley),
                    ));
                    name_output.state.store(ui.ctx(), name_output.response.id);
                    dialog.focus_name_on_open = false;
                }

                if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    accept = true;
                }
                if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    cancel = true;
                }

                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                    if ui.button("OK").clicked() {
                        accept = true;
                    }
                });
            });

        if !open || cancel {
            self.group_layers_dialog = None;
            return;
        }
        if accept {
            let name = dialog.resolved_name();
            let target = dialog.target.clone();
            self.group_layers_dialog = None;
            self.apply_new_group(name, target);
            ctx.request_repaint();
        }
    }

    pub(super) fn ui_memory_load_dialog(&mut self, ctx: &egui::Context) {
        if let Some((summary, requests)) =
            ui_pending_memory_action_dialog(ctx, &mut self.pending_memory_load)
        {
            self.execute_memory_load(summary, requests);
        }
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
                    &mut self.screenshot_settings.include_legend,
                    "Include legend (visible channels)",
                );
                ui.add_space(8.0);
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

    pub(super) fn drain_screenshots(&mut self) {
        while let Ok(resp) = self.screenshot_worker.rx.try_recv() {
            match resp {
                crate::app_support::screenshot::ScreenshotWorkerResp::Saved {
                    id,
                    path,
                    result,
                } => {
                    if self.screenshot_in_flight == Some(id) {
                        self.screenshot_in_flight = None;
                    }
                    self.status = match result {
                        Ok(()) => format!("Saved screenshot -> {}", path.to_string_lossy()),
                        Err(err) => format!("Save screenshot failed: {err}"),
                    };
                }
            }
        }
    }

    pub(super) fn apply_new_group(&mut self, name: String, target: GroupLayersTarget) {
        let groups_before = self.layer_groups.clone();
        match target {
            GroupLayersTarget::Channels(indices) => {
                let first_color = indices
                    .first()
                    .and_then(|idx| self.channels.get(*idx))
                    .map(|ch| {
                        layer_groups::effective_channel_color_rgb(
                            &self.layer_groups,
                            &ch.name,
                            ch.color_rgb,
                        )
                    })
                    .unwrap_or([255, 255, 255]);

                let existing_ids = self
                    .layer_groups
                    .channel_groups
                    .iter()
                    .map(|g| g.id)
                    .collect::<Vec<_>>();
                let gid = layer_groups::next_group_id(&existing_ids);
                self.layer_groups.channel_groups.push(
                    crate::data::project_config::ProjectChannelGroup {
                        id: gid,
                        name,
                        expanded: true,
                        color_rgb: first_color,
                    },
                );
                for idx in indices {
                    if let Some(ch) = self.channels.get(idx) {
                        self.layer_groups.channel_members.insert(
                            ch.name.clone(),
                            crate::data::project_config::ProjectChannelGroupMember {
                                group_id: gid,
                                inherit_color: true,
                            },
                        );
                    }
                }
            }
            GroupLayersTarget::Annotations(layer_ids) => {
                let existing_ids = self
                    .layer_groups
                    .annotation_groups
                    .iter()
                    .map(|g| g.id)
                    .collect::<Vec<_>>();
                let gid = layer_groups::next_group_id(&existing_ids);
                self.layer_groups.annotation_groups.push(
                    crate::data::project_config::ProjectAnnotationGroup {
                        id: gid,
                        name,
                        expanded: true,
                        visible: true,
                        tint_rgb: None,
                        tint_strength: 0.35,
                    },
                );
                for id in layer_ids {
                    self.layer_groups.annotation_members.insert(
                        id,
                        crate::data::project_config::ProjectAnnotationGroupMember {
                            group_id: gid,
                            inherit_tint: true,
                        },
                    );
                }
            }
        }
        self.commit_layer_groups_preview(groups_before);
    }
}
