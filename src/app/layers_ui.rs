use super::*;

impl OmeZarrViewerApp {
    pub(super) fn ui_layers(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        self.rebuild_layer_orders();

        ui.heading("Tools");
        // Tool availability depends on the current active layer. When a layer change invalidates
        // the current tool, fall back to pan and clear any partial selection gesture so we don't
        // leave stale drag state behind.
        let can_object_select = self.active_layer_supports_spatial_selection();
        let xy_tools_enabled = self.view_plane_is_xy();
        if !xy_tools_enabled
            && matches!(
                self.tool_mode,
                ToolMode::TransformLayer | ToolMode::DrawMaskPolygon
            )
        {
            self.clear_spatial_selection_drag();
            self.tool_mode = ToolMode::Pan;
        }
        if !can_object_select && matches!(self.tool_mode, ToolMode::LassoSelect) {
            self.clear_spatial_selection_drag();
            self.tool_mode = ToolMode::Pan;
        }
        ui.horizontal(|ui| {
            if icon_button(
                ui,
                Icon::Select,
                self.tool_mode == ToolMode::Select,
                egui::Sense::click(),
            )
            .on_hover_text("Select and edit mask polygons or objects")
            .clicked()
            {
                self.clear_spatial_selection_drag();
                self.tool_mode = ToolMode::Select;
            }
            if icon_button(
                ui,
                Icon::Pan,
                self.tool_mode == ToolMode::Pan,
                egui::Sense::click(),
            )
            .on_hover_text("Pan")
            .clicked()
            {
                self.clear_spatial_selection_drag();
                self.tool_mode = ToolMode::Pan;
            }
            if icon_button(
                ui,
                Icon::Move,
                self.tool_mode == ToolMode::MoveLayer,
                egui::Sense::click(),
            )
            .on_hover_text(
                "Move selected visible layer(s); on mask layers, drag a polygon to move it",
            )
            .clicked()
            {
                self.clear_spatial_selection_drag();
                self.tool_mode = ToolMode::MoveLayer;
            }
            let can_transform =
                xy_tools_enabled && matches!(self.active_layer, LayerId::Channel(_));
            let mut transform_clicked = false;
            ui.add_enabled_ui(can_transform, |ui| {
                if icon_button(
                    ui,
                    Icon::Transform,
                    self.tool_mode == ToolMode::TransformLayer,
                    egui::Sense::click(),
                )
                .on_hover_text("Transform active channel (scale/rotate)")
                .clicked()
                {
                    transform_clicked = true;
                }
            });
            if transform_clicked {
                self.clear_spatial_selection_drag();
                self.tool_mode = ToolMode::TransformLayer;
            }
            ui.add_enabled_ui(xy_tools_enabled, |ui| {
                if icon_button(
                    ui,
                    Icon::Polygon,
                    self.tool_mode == ToolMode::DrawMaskPolygon,
                    egui::Sense::click(),
                )
                .on_hover_text("Draw mask polygon")
                .clicked()
                {
                    self.clear_spatial_selection_drag();
                    self.tool_mode = ToolMode::DrawMaskPolygon;
                    if let Some(id) = self.ensure_editable_mask_layer() {
                        self.commit_active_layer(LayerId::Mask(id));
                        self.drawing_mask_layer = Some(id);
                    }
                }
            });
            ui.add_enabled_ui(can_object_select, |ui| {
                if icon_button(
                    ui,
                    Icon::LassoSelect,
                    self.tool_mode == ToolMode::LassoSelect,
                    egui::Sense::click(),
                )
                .on_hover_text("Draw a freehand lasso to select cells by centroid")
                .clicked()
                {
                    self.clear_spatial_selection_drag();
                    self.tool_mode = ToolMode::LassoSelect;
                }
            });
            if crate::ui::help::help_button(ui, crate::ui::help::HelpTopic::Tools) {
                self.active_help_topic = Some(crate::ui::help::HelpTopic::Tools);
            }
        });
        if !xy_tools_enabled {
            ui.small("Non-XY view is image-only. Overlays and editing tools stay disabled.");
        }

        ui.separator();
        ui.heading("Layers");

        egui::ScrollArea::vertical()
            .id_salt("layers-scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let mut groups_cfg = self.current_layer_groups();
                let mut groups_changed = false;

                // Overlays master visibility toggle.
                let overlay_ids = self.overlay_layer_order.clone();
                if overlay_ids.is_empty() {
                    ui.horizontal(|ui| {
                        ui.label("Overlays");
                        ui.add_space(4.0);
                        ui.label("(none)");
                    });
                } else {
                    let mut overlays_all = true;
                    let mut overlays_none = true;
                    for id in overlay_ids.iter().copied() {
                        if !self.layer_is_available(id) {
                            continue;
                        }
                        if let Some(v) = self.layer_visible_mut(id).map(|v| *v) {
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
                            if self.native_layers_actor_owned() {
                                self.submit_native_layer_visibilities(
                                    overlay_ids.iter().copied(),
                                    all,
                                );
                            } else {
                                let mut mask_visibility_changed = false;
                                for id in overlay_ids {
                                    if !self.layer_is_available(id) {
                                        continue;
                                    }
                                    if let Some(v) = self.layer_visible_mut(id) {
                                        mask_visibility_changed |=
                                            matches!(id, LayerId::Mask(_)) && *v != all;
                                        *v = all;
                                    }
                                }
                                if mask_visibility_changed {
                                    self.mark_mask_layers_project_dirty();
                                }
                            }
                            self.bump_render_id();
                        }
                    });

                    // Annotation groups: show a collapsible header at the first member, and hide
                    // members when collapsed.
                    let mut ann_members_by_group: HashMap<u64, Vec<u64>> = HashMap::new();
                    for id in self.overlay_layer_order.iter().copied() {
                        let LayerId::Annotation(aid) = id else { continue };
                        let Some(m) = groups_cfg.annotation_members.get(&aid) else { continue };
                        if groups_cfg.annotation_groups.iter().any(|g| g.id == m.group_id) {
                            ann_members_by_group.entry(m.group_id).or_default().push(aid);
                        }
                    }
                    let mut ann_headers_shown: HashSet<u64> = HashSet::new();
                    let mut delete_ann_group: Option<u64> = None;
                    let mut delete_mask_layer: Option<u64> = None;

                    for i in 0..self.overlay_layer_order.len() {
                        let id = self.overlay_layer_order[i];

                        if let LayerId::Annotation(aid) = id {
                            if let Some(m) = groups_cfg.annotation_members.get(&aid) {
                                let gid = m.group_id;
                                if groups_cfg.annotation_groups.iter().any(|g| g.id == gid) {
                                    if !ann_headers_shown.contains(&gid) {
                                        ann_headers_shown.insert(gid);
                                        let Some(group_idx) = groups_cfg
                                            .annotation_groups
                                            .iter()
                                            .position(|g| g.id == gid)
                                        else {
                                            // Shouldn't happen due to check above.
                                            continue;
                                        };
                                        let members = ann_members_by_group
                                            .get(&gid)
                                            .map(|v| v.as_slice())
                                            .unwrap_or(&[]);

                                        let group = &mut groups_cfg.annotation_groups[group_idx];
                                        let mut all = true;
                                        let mut none = true;
                                        for &mid in members {
                                            let lid = LayerId::Annotation(mid);
                                            if let Some(v) = self.layer_visible_mut(lid).map(|v| *v) {
                                                all &= v;
                                                none &= !v;
                                            }
                                        }
                                        let mixed = !members.is_empty() && !all && !none;

                                        let header = egui::collapsing_header::CollapsingState::load_with_default_open(
                                            ui.ctx(),
                                            ui.make_persistent_id(("annotation-group", group.id)),
                                            group.expanded,
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
                                                    if self.native_layers_actor_owned() {
                                                        self.submit_native_layer_visibilities(
                                                            members.iter().copied().map(LayerId::Annotation),
                                                            set_all,
                                                        );
                                                    } else {
                                                        for &mid in members {
                                                            if let Some(v) = self.layer_visible_mut(LayerId::Annotation(mid)) {
                                                                *v = set_all;
                                                            }
                                                        }
                                                    }
                                                    group.visible = set_all;
                                                    groups_changed = true;
                                                    self.bump_render_id();
                                                }
                                            });
                                            ui.add_space(4.0);
                                            ui.label(group.name.clone());
                                        });
                                        let open = header.is_open();
                                        let (_toggle, _hdr, _body) = header.body(|ui| {
                                            ui.horizontal(|ui| {
                                                ui.label("Name");
                                                groups_changed |= ui.text_edit_singleline(&mut group.name).changed();
                                            });
                                            ui.horizontal(|ui| {
                                                ui.label("Visible");
                                                if ui.checkbox(&mut group.visible, "").changed() {
                                                    groups_changed = true;
                                                }
                                                if ui.button("Delete group").clicked() {
                                                    delete_ann_group = Some(group.id);
                                                }
                                            });
                                            ui.horizontal(|ui| {
                                                let mut has_tint = group.tint_rgb.is_some();
                                                if ui.checkbox(&mut has_tint, "Tint").changed() {
                                                    if has_tint && group.tint_rgb.is_none() {
                                                        group.tint_rgb = Some([255, 255, 255]);
                                                    }
                                                    if !has_tint {
                                                        group.tint_rgb = None;
                                                    }
                                                    groups_changed = true;
                                                }
                                                if let Some(rgb) = group.tint_rgb.as_mut() {
                                                    let mut c = egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
                                                    if ui.color_edit_button_srgba(&mut c).changed() {
                                                        *rgb = [c.r(), c.g(), c.b()];
                                                        groups_changed = true;
                                                    }
                                                }
                                            });
                                            groups_changed |= ui
                                                .add(
                                                    egui::Slider::new(&mut group.tint_strength, 0.0..=1.0)
                                                        .text("Tint strength")
                                                        .clamping(egui::SliderClamping::Always),
                                                )
                                                .changed();
                                        });
                                        if open != group.expanded {
                                            group.expanded = open;
                                            groups_changed = true;
                                        }
                                    }

                                    // Hide grouped members when the group is collapsed.
                                    let expanded = groups_cfg
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

                        let available = self.layer_is_available(id);
                        let selected = self.active_layer == id || self.selected_overlay_layers.contains(&id);
                        let icon = self.layer_icon(id);
                        let name = self.layer_display_name(id);
                        let visible = self.layer_visible_mut(id).map(|v| *v);
                        let resp = layer_list::ui_layer_row(
                            ui,
                            ctx,
                            &mut self.layer_drag,
                            LayerGroup::Overlays,
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
                            // Primary click selection.
                            if mods.shift && self.overlay_select_anchor_pos.is_some() {
                                let anchor = self.overlay_select_anchor_pos.unwrap_or(i);
                                let (a, b) = if anchor <= i { (anchor, i) } else { (i, anchor) };
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
                            self.commit_active_layer(id);
                        } else if resp.row_response.secondary_clicked() {
                            // Right-click selects the row (single) if it wasn't already selected.
                            if !self.selected_overlay_layers.contains(&id) {
                                self.selected_overlay_layers.clear();
                                self.selected_overlay_layers.insert(id);
                                self.overlay_select_anchor_pos = Some(i);
                                self.commit_active_layer(id);
                            }
                        }
                        if let Some(v) = resp.visible_changed {
                            let mut mask_visibility_changed = false;
                            if self.native_layers_actor_owned() {
                                self.submit_native_layer_visibility(id, v);
                            } else if let Some(dst) = self.layer_visible_mut(id) {
                                mask_visibility_changed = matches!(id, LayerId::Mask(_)) && *dst != v;
                                *dst = v;
                            }
                            if mask_visibility_changed && !self.mask_actor_owned() {
                                self.mark_mask_layers_project_dirty();
                            }
                        }
                        if resp.changed {
                            self.bump_render_id();
                        }

                        // Context menu: group layers.
                        resp.row_response.context_menu(|ui| {
                            if self.current_visible_move_targets_have_moved() {
                                if ui.button("Reset position").clicked() {
                                    self.reset_current_visible_move_targets_to_loaded();
                                    ui.close();
                                }
                                ui.separator();
                            }
                            if let LayerId::Mask(mask_id) = id {
                                if ui.button("Export layer as GeoJSON...").clicked() {
                                    let default_name =
                                        self.default_mask_layer_export_filename(mask_id);
                                    if let Some(path) = FileDialog::new()
                                        .add_filter("GeoJSON", &["geojson", "json"])
                                        .set_file_name(&default_name)
                                        .set_title("Export Mask Layer GeoJSON")
                                        .save_file()
                                    {
                                        match self.export_mask_layer_geojson(mask_id, &path) {
                                            Ok(()) => self.set_status(format!(
                                                "Exported mask layer -> {}",
                                                path.to_string_lossy()
                                            )),
                                            Err(err) => self.set_status(format!(
                                                "Export mask layer failed: {err}"
                                            )),
                                        }
                                    }
                                    ui.close();
                                }
                                if ui.button("Delete layer").clicked() {
                                    delete_mask_layer = Some(mask_id);
                                    ui.close();
                                }
                                ui.separator();
                            }
                            let selected_annotations: Vec<u64> = self
                                .selected_overlay_layers
                                .iter()
                                .filter_map(|l| match l {
                                    LayerId::Annotation(a) => Some(*a),
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
                        groups_cfg.annotation_groups.retain(|g| g.id != group_id);
                        groups_cfg
                            .annotation_members
                            .retain(|_k, m| m.group_id != group_id);
                        groups_changed = true;
                    }
                    if let Some(mask_id) = delete_mask_layer
                        && self.delete_mask_layer(mask_id)
                    {
                        self.set_status("Deleted mask layer.");
                    }
                }
                if groups_changed {
                    self.set_current_layer_groups(groups_cfg);
                    self.bump_render_id();
                }
                ui.separator();
                let channel_search_before = self.channel_list_search.clone();
                channels_panel::show(self, ui, ctx);
                if self.channel_list_search != channel_search_before {
                    self.native_control_intents.push(NativeControlIntent {
                        method: "viewer.channels.presentation.set",
                        params: serde_json::json!({"search": self.channel_list_search}),
                    });
                }
            });

        layer_list::paint_drag_preview(ctx, self.layer_drag.as_ref(), |id| {
            self.layer_display_name(id)
        });

        let mut dropped: Option<(LayerGroup, usize, usize)> = None;
        layer_list::finish_drag_if_released(ctx, &mut self.layer_drag, |group, from, to| {
            dropped = Some((group, from, to));
        });
        if let Some((group, from, to)) = dropped {
            if self.native_layers_actor_owned() {
                match group {
                    LayerGroup::Overlays => {
                        let mut order = self.overlay_layer_order.clone();
                        layer_list::reorder_vec(&mut order, from, to);
                        self.submit_native_layer_order("overlays", order);
                    }
                    LayerGroup::Channels => {
                        let mut order = self
                            .channel_layer_order
                            .iter()
                            .copied()
                            .map(LayerId::Channel)
                            .collect::<Vec<_>>();
                        layer_list::reorder_vec(&mut order, from, to);
                        self.submit_native_layer_order("channels", order);
                    }
                }
            } else {
                match group {
                    LayerGroup::Overlays => {
                        layer_list::reorder_vec(&mut self.overlay_layer_order, from, to)
                    }
                    LayerGroup::Channels => {
                        layer_list::reorder_vec(&mut self.channel_layer_order, from, to)
                    }
                }
            }
            self.bump_render_id();
        }
    }

    pub(super) fn open_group_layers_dialog_channels(&mut self, members: Vec<usize>) {
        let existing = self
            .current_layer_groups()
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
            .current_layer_groups()
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
        }
    }

    pub(super) fn apply_new_group(&mut self, name: String, target: GroupLayersTarget) {
        match target {
            GroupLayersTarget::Channels(indices) => {
                let first_color = indices
                    .first()
                    .and_then(|idx| self.channels.get(*idx))
                    .map(|ch| {
                        layer_groups::effective_channel_color_rgb(
                            &self.current_layer_groups(),
                            &ch.name,
                            ch.color_rgb,
                        )
                    })
                    .unwrap_or([255, 255, 255]);

                let mut groups = self.current_layer_groups();
                {
                    let existing_ids = groups
                        .channel_groups
                        .iter()
                        .map(|g| g.id)
                        .collect::<Vec<_>>();
                    let gid = layer_groups::next_group_id(&existing_ids);
                    groups
                        .channel_groups
                        .push(crate::data::project_config::ProjectChannelGroup {
                            id: gid,
                            name,
                            expanded: true,
                            color_rgb: first_color,
                        });
                    for idx in indices {
                        if let Some(ch) = self.channels.get(idx) {
                            groups.channel_members.insert(
                                ch.name.clone(),
                                crate::data::project_config::ProjectChannelGroupMember {
                                    group_id: gid,
                                    inherit_color: true,
                                },
                            );
                        }
                    }
                }
                self.commit_current_channel_groups(groups);
                self.bump_render_id();
            }
            GroupLayersTarget::Annotations(layer_ids) => {
                let mut groups = self.current_layer_groups();
                {
                    let existing_ids = groups
                        .annotation_groups
                        .iter()
                        .map(|g| g.id)
                        .collect::<Vec<_>>();
                    let gid = layer_groups::next_group_id(&existing_ids);
                    groups.annotation_groups.push(
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
                        groups.annotation_members.insert(
                            id,
                            crate::data::project_config::ProjectAnnotationGroupMember {
                                group_id: gid,
                                inherit_tint: true,
                            },
                        );
                    }
                }
                self.set_current_layer_groups(groups);
                self.bump_render_id();
            }
        }
    }

    #[cfg(test)]
    pub(super) fn group_channel_indices_for_deep_link(
        &mut self,
        name: &str,
        indices: &[usize],
        color_rgb: Option<[u8; 3]>,
    ) {
        let first_color = indices
            .first()
            .and_then(|idx| self.channels.get(*idx))
            .map(|ch| {
                layer_groups::effective_channel_color_rgb(
                    &self.current_layer_groups(),
                    &ch.name,
                    ch.color_rgb,
                )
            })
            .unwrap_or([255, 255, 255]);
        let group_color = color_rgb.unwrap_or(first_color);

        let mut groups = self.current_layer_groups();
        let group_id = match groups
            .channel_groups
            .iter_mut()
            .find(|group| group.name.eq_ignore_ascii_case(name))
        {
            Some(group) => {
                group.expanded = true;
                group.color_rgb = group_color;
                group.id
            }
            None => {
                let existing_ids = groups
                    .channel_groups
                    .iter()
                    .map(|group| group.id)
                    .collect::<Vec<_>>();
                let id = layer_groups::next_group_id(&existing_ids);
                groups.channel_groups.push(ProjectChannelGroup {
                    id,
                    name: name.to_string(),
                    expanded: true,
                    color_rgb: group_color,
                });
                id
            }
        };

        groups
            .channel_members
            .retain(|_, member| member.group_id != group_id);
        for idx in indices {
            if let Some(channel) = self.channels.get(*idx) {
                groups.channel_members.insert(
                    channel.name.clone(),
                    ProjectChannelGroupMember {
                        group_id,
                        inherit_color: true,
                    },
                );
            }
        }
        self.selected_channel_group_id = Some(group_id);
        self.set_current_layer_groups(groups);
        self.bump_render_id();
    }

    pub(super) fn set_channel_group_color_inheritance(
        &mut self,
        channel_idx: usize,
        inherit_color: bool,
    ) {
        let Some(channel_name) = self
            .channels
            .get(channel_idx)
            .map(|channel| channel.name.clone())
        else {
            return;
        };
        let mut groups = self.current_layer_groups();
        let Some(member) = groups.channel_members.get_mut(&channel_name) else {
            return;
        };
        if member.inherit_color != inherit_color {
            member.inherit_color = inherit_color;
            self.set_current_layer_groups(groups);
        }
    }

    #[cfg(test)]
    pub(super) fn move_channels_to_top_for_deep_link(&mut self, channel_indices: &[usize]) {
        let channel_count = self.channels.len();
        if channel_count == 0 || channel_indices.is_empty() {
            return;
        }

        let mut pinned_seen = HashSet::new();
        let pinned = channel_indices
            .iter()
            .copied()
            .filter(|idx| *idx < channel_count && pinned_seen.insert(*idx))
            .collect::<Vec<_>>();
        if pinned.is_empty() {
            return;
        }
        let pinned_set = pinned.iter().copied().collect::<HashSet<_>>();

        let mut next_order = pinned;
        let mut seen = next_order.iter().copied().collect::<HashSet<_>>();
        next_order.extend(
            self.channel_layer_order
                .iter()
                .copied()
                .filter(|idx| *idx < channel_count)
                .filter(|idx| !pinned_set.contains(idx))
                .filter(|idx| seen.insert(*idx)),
        );
        for idx in 0..channel_count {
            if seen.insert(idx) {
                next_order.push(idx);
            }
        }

        if self.channel_layer_order != next_order {
            self.channel_layer_order = next_order;
            self.bump_render_id();
        }
        self.channel_sort_mode = ChannelSortMode::Manual;
    }
}
