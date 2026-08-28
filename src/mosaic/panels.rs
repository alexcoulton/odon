//! Project, properties, layout, memory, and contrast panels.

use super::*;

impl MosaicViewerApp {
    pub(super) fn ui_project(&mut self, ui: &mut egui::Ui) {
        if let Some(action) = self.project_space.ui(ui, None) {
            self.handle_project_space_action(action);
        }
    }

    pub(super) fn handle_project_space_action(
        &mut self,
        action: crate::project::ProjectSpaceAction,
    ) {
        if self.project_space.submit_action_control_intent(&action) {
            return;
        }
        match action {
            crate::project::ProjectSpaceAction::CaptureCurrentView => {}
            crate::project::ProjectSpaceAction::OpenRemoteDialog => {
                self.pending_platform_effect = Some(MosaicPlatformEffect::OpenRemoteDialog);
            }
            crate::project::ProjectSpaceAction::ShowHelp(topic) => {
                self.active_help_topic = Some(topic);
            }
            _ => unreachable!("actor-owned project action was not accepted by its command outbox"),
        }
    }

    pub(super) fn ui_properties(&mut self, ui: &mut egui::Ui) {
        if let Some(group_id) = self.selected_channel_group_id {
            self.ui_group_contrast(ui, group_id);
            return;
        }
        match self.active_layer {
            MosaicLayerId::Channel(idx) => {
                self.selected_channel = idx.min(self.channels.len().saturating_sub(1));
                self.ui_contrast(ui);
            }
            MosaicLayerId::TextLabels => {
                ui.heading("Text labels");
                ui.separator();
                let mut show_text_labels = self.show_text_labels;
                let mut label_columns = self.label_columns.clone();
                let mut changed = ui.checkbox(&mut show_text_labels, "Visible").changed();
                ui.add_enabled_ui(show_text_labels, |ui| {
                    let mut available_columns = vec!["id".to_string()];
                    available_columns.extend(self.metadata_columns.iter().cloned());

                    ui.horizontal(|ui| {
                        let mut add_clicked = false;
                        if ui.button("+ Add label line").clicked() {
                            add_clicked = true;
                        }
                        if ui
                            .add_enabled(
                                !label_columns.is_empty(),
                                egui::Button::new("Clear lines"),
                            )
                            .clicked()
                        {
                            label_columns.clear();
                            changed = true;
                        }
                        if add_clicked {
                            let next = available_columns
                                .iter()
                                .find(|col| !label_columns.contains(*col))
                                .cloned()
                                .or_else(|| available_columns.first().cloned());
                            if let Some(next) = next {
                                label_columns.push(next);
                                changed = true;
                            }
                        }
                    });

                    if label_columns.is_empty() {
                        ui.label("No label lines selected.");
                    }

                    let mut remove_idx = None;
                    for (idx, column) in label_columns.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(format!("Line {}", idx + 1));
                            egui::ComboBox::from_id_salt(("mosaic-label-line", idx))
                                .selected_text(column.clone())
                                .show_ui(ui, |ui| {
                                    for col in &available_columns {
                                        changed |=
                                            ui.selectable_value(column, col.clone(), col).changed();
                                    }
                                });
                            if ui.small_button("Remove").clicked() {
                                remove_idx = Some(idx);
                            }
                        });
                    }
                    if let Some(idx) = remove_idx {
                        label_columns.remove(idx);
                        changed = true;
                    }
                });
                if changed {
                    let mut params = self.layout_command_params();
                    params["show_text_labels"] = serde_json::json!(show_text_labels);
                    params["label_columns"] = serde_json::json!(label_columns);
                    self.submit_native_control_intent("mosaic.layout.configure", params);
                }
            }
            MosaicLayerId::SegmentationGeoJson => {
                let camera_before = self.control_camera_snapshot();
                let have_any_seg = self.seg_geojson.has_any_segpaths();
                let visible_before = self.seg_geojson.visible;
                let style_before = self.seg_geojson.control_style_json();
                let (zoom_selected, clear_selection) =
                    self.seg_geojson.ui_left_panel(ui, have_any_seg);
                let style_after = self.seg_geojson.control_style_json();
                if self.seg_geojson.visible != visible_before {
                    let visible = self.seg_geojson.visible;
                    self.seg_geojson.visible = visible_before;
                    self.submit_native_control_intent(
                        "viewer.objects.set_visibility",
                        serde_json::json!({"target":"objects","visible":visible}),
                    );
                }
                if style_after != style_before {
                    let _ = self.seg_geojson.apply_control_style(&style_before);
                    self.submit_native_control_intent(
                        "mosaic.objects.style.set",
                        serde_json::json!({"style":style_after}),
                    );
                }
                if clear_selection {
                    self.submit_native_control_intent(
                        "mosaic.objects.selection.clear",
                        serde_json::json!({}),
                    );
                }
                if zoom_selected
                    && let (Some(bounds), Some(viewport)) = (
                        self.seg_geojson.selected_bounds_world(),
                        self.last_canvas_rect,
                    )
                {
                    self.camera.fit_to_world_rect(viewport, bounds);
                }
                self.submit_camera_preview_if_changed(&camera_before);
                if have_any_seg {
                    let (loaded, loading, total) = self.seg_geojson.loaded_stats();
                    ui.label(format!(
                        "Objects: {loaded}/{total} loaded ({loading} loading)"
                    ));
                    let missing = self.seg_geojson.last_missing_bins();
                    if self.seg_geojson.visible && missing > 0 {
                        ui.label(format!("Object bins: {missing} pending GPU uploads"));
                    }
                }
            }
            MosaicLayerId::Annotation(id) => {
                let groups_before = self.layer_groups.clone();
                let Some(idx) = self.annotation_layers.iter().position(|l| l.id == id) else {
                    ui.label("Annotation layer not found.");
                    return;
                };
                let annotation_before = self.annotation_layers[idx].control_state_json();
                ui.heading(self.annotation_layers[idx].name.clone());
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Name");
                    ui.text_edit_singleline(&mut self.annotation_layers[idx].name);
                });
                ui.separator();

                let mut selected_group: Option<u64> = self
                    .layer_groups
                    .annotation_members
                    .get(&id)
                    .map(|m| m.group_id)
                    .filter(|gid| {
                        self.layer_groups
                            .annotation_groups
                            .iter()
                            .any(|g| g.id == *gid)
                    });
                let mut groups_changed = false;

                ui.horizontal(|ui| {
                    ui.label("Group");
                    egui::ComboBox::from_id_salt(("mosaic-annotation-group-select", id))
                        .selected_text(
                            selected_group
                                .and_then(|gid| {
                                    self.layer_groups
                                        .annotation_groups
                                        .iter()
                                        .find(|g| g.id == gid)
                                })
                                .map(|g| g.name.as_str())
                                .unwrap_or("(none)"),
                        )
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut selected_group, None, "(none)");
                            for g in &self.layer_groups.annotation_groups {
                                ui.selectable_value(
                                    &mut selected_group,
                                    Some(g.id),
                                    g.name.clone(),
                                );
                            }
                        });
                    if ui
                        .button("+ Group")
                        .on_hover_text("Create a new annotation group")
                        .clicked()
                    {
                        let existing = self
                            .layer_groups
                            .annotation_groups
                            .iter()
                            .map(|g| g.id)
                            .collect::<Vec<_>>();
                        let id2 = layer_groups::next_group_id(&existing);
                        self.layer_groups.annotation_groups.push(
                            crate::data::project_config::ProjectAnnotationGroup {
                                id: id2,
                                name: format!("Group {id2}"),
                                expanded: true,
                                visible: true,
                                tint_rgb: None,
                                tint_strength: 0.35,
                            },
                        );
                        selected_group = Some(id2);
                        groups_changed = true;
                    }
                });

                let have_member = self.layer_groups.annotation_members.get(&id).is_some();
                if selected_group.is_none() && have_member {
                    self.layer_groups.annotation_members.remove(&id);
                    groups_changed = true;
                } else if let Some(gid) = selected_group {
                    match self.layer_groups.annotation_members.get_mut(&id) {
                        Some(m) => {
                            if m.group_id != gid {
                                m.group_id = gid;
                                groups_changed = true;
                            }
                        }
                        None => {
                            self.layer_groups.annotation_members.insert(
                                id,
                                crate::data::project_config::ProjectAnnotationGroupMember {
                                    group_id: gid,
                                    inherit_tint: true,
                                },
                            );
                            groups_changed = true;
                        }
                    }
                }

                if let Some(gid) = selected_group {
                    let mut inherit_tint = self
                        .layer_groups
                        .annotation_members
                        .get(&id)
                        .map(|m| m.inherit_tint)
                        .unwrap_or(true);
                    ui.horizontal(|ui| {
                        if ui
                            .checkbox(&mut inherit_tint, "Inherit group tint")
                            .changed()
                        {
                            if let Some(m) = self.layer_groups.annotation_members.get_mut(&id) {
                                m.inherit_tint = inherit_tint;
                                groups_changed = true;
                            }
                        }
                    });

                    if let Some(group) = self
                        .layer_groups
                        .annotation_groups
                        .iter_mut()
                        .find(|g| g.id == gid)
                    {
                        ui.separator();
                        ui.label("Group settings");
                        ui.horizontal(|ui| {
                            ui.label("Name");
                            groups_changed |= ui.text_edit_singleline(&mut group.name).changed();
                        });
                        ui.horizontal(|ui| {
                            ui.label("Visible");
                            if ui.checkbox(&mut group.visible, "").changed() {
                                groups_changed = true;
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
                    }
                }
                ui.separator();
                self.annotation_layers[idx].ui_properties(ui);
                let annotation_after = self.annotation_layers[idx].control_state_json();
                let source_request = self.annotation_layers[idx].take_control_source_request();
                if annotation_after != annotation_before {
                    self.submit_native_control_intent(
                        "viewer.annotations.layers.update",
                        serde_json::json!({"layer_id":id,"state":annotation_after}),
                    );
                }
                if let Some((request, params)) = source_request {
                    let method = match request {
                        crate::annotations::AnnotationSourceRequest::Inspect => {
                            "viewer.annotations.source.inspect"
                        }
                        crate::annotations::AnnotationSourceRequest::Load => {
                            "viewer.annotations.source.load"
                        }
                        crate::annotations::AnnotationSourceRequest::Reload => {
                            "viewer.annotations.source.reload"
                        }
                    };
                    self.submit_native_control_intent(method, params);
                }
                ui.separator();
                if ui.button("Delete layer").clicked() {
                    self.submit_native_control_intent(
                        "viewer.annotations.layers.delete",
                        serde_json::json!({"layer_id":id}),
                    );
                }
                if groups_changed {
                    ui.ctx().request_repaint();
                }
                self.commit_layer_groups_preview(groups_before);
            }
        }
    }

    pub(super) fn ui_layout(&mut self, ui: &mut egui::Ui, _ctx: &egui::Context) {
        ui.heading("Arrange");
        ui.label("Sort and optionally group ROIs by samplesheet columns.");
        ui.add_space(6.0);

        let mut group_by = self.group_by.clone();
        let mut show_group_labels = self.show_group_labels;
        let mut group_gap = self.group_gap;
        let mut layout_mode = self.layout_mode;
        let mut sort_by = self.sort_by.clone();
        let mut sort_secondary_enabled = self.sort_secondary_enabled;
        let mut sort_by_secondary = self.sort_by_secondary.clone();
        let mut changed = false;
        egui::ComboBox::from_label("Group by")
            .selected_text(if group_by.is_empty() {
                "(none)".to_string()
            } else {
                group_by.clone()
            })
            .show_ui(ui, |ui| {
                changed |= ui
                    .selectable_value(&mut group_by, String::new(), "(none)")
                    .changed();
                for col in &self.metadata_columns {
                    changed |= ui
                        .selectable_value(&mut group_by, col.clone(), col)
                        .changed();
                }
            });
        ui.horizontal(|ui| {
            ui.add_enabled_ui(!group_by.is_empty(), |ui| {
                changed |= ui
                    .checkbox(&mut show_group_labels, "Show group labels")
                    .changed();
                ui.label("Gap");
                changed |= ui
                    .add(egui::DragValue::new(&mut group_gap).speed(5.0))
                    .changed();
            });
        });

        ui.add_space(8.0);
        egui::ComboBox::from_label("Layout")
            .selected_text(layout_mode.label())
            .show_ui(ui, |ui| {
                changed |= ui
                    .selectable_value(
                        &mut layout_mode,
                        MosaicLayoutMode::FitCells,
                        MosaicLayoutMode::FitCells.label(),
                    )
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut layout_mode,
                        MosaicLayoutMode::NativePixels,
                        MosaicLayoutMode::NativePixels.label(),
                    )
                    .changed();
            });

        ui.add_space(8.0);
        egui::ComboBox::from_label("Sort by")
            .selected_text(sort_by.clone())
            .show_ui(ui, |ui| {
                changed |= ui
                    .selectable_value(&mut sort_by, "id".to_string(), "id")
                    .changed();
                for col in &self.metadata_columns {
                    changed |= ui
                        .selectable_value(&mut sort_by, col.clone(), col)
                        .changed();
                }
            });

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            changed |= ui
                .checkbox(&mut sort_secondary_enabled, "Then by")
                .changed();
            ui.add_enabled_ui(sort_secondary_enabled, |ui| {
                egui::ComboBox::from_id_salt("sort-by-secondary")
                    .selected_text(sort_by_secondary.clone())
                    .show_ui(ui, |ui| {
                        changed |= ui
                            .selectable_value(&mut sort_by_secondary, "id".to_string(), "id")
                            .changed();
                        for col in &self.metadata_columns {
                            changed |= ui
                                .selectable_value(&mut sort_by_secondary, col.clone(), col)
                                .changed();
                        }
                    });
            });
        });

        if ui.button("Apply sort").clicked() || changed {
            let params = serde_json::json!({
                "group_by":group_by,
                "sort_by":sort_by,
                "sort_secondary_enabled":sort_secondary_enabled,
                "sort_by_secondary":sort_by_secondary,
                "layout":layout_mode.storage_key(),
                "columns":self.grid_cols,
                "group_gap":group_gap,
                "show_group_labels":show_group_labels,
                "show_text_labels":self.show_text_labels,
                "label_columns":self.label_columns,
                "fit":false,
            });
            self.submit_native_control_intent("mosaic.layout.configure", params);
        }

        ui.add_space(8.0);
        ui.label(format!(
            "Layout: {}, {} columns",
            self.layout_mode.label(),
            self.grid_cols
        ));
    }

    pub(super) fn ui_memory(&mut self, ui: &mut egui::Ui) {
        let tile_cache = self.tiles_gl.stats();
        ui.heading("Image tile cache");
        ui.label(format!(
            "{} tracked / {} budget ({} entries)",
            format_bytes(tile_cache.total_tracked_bytes),
            format_bytes(tile_cache.effective_budget_bytes),
            tile_cache.entries,
        ));
        ui.label(format!(
            "Pending CPU {} · uploaded texture {} · awaiting GL deletion {}",
            format_bytes(tile_cache.pending_cpu_bytes),
            format_bytes(tile_cache.uploaded_texture_bytes),
            format_bytes(tile_cache.queued_deletion_bytes),
        ));
        ui.label(format!(
            "Pressure: {} · policy: {} · previous channel group: {}",
            tile_cache.pressure_state,
            tile_cache.resolution_reason,
            if tile_cache.previous_channel_group.is_empty() {
                "not retained".to_string()
            } else {
                format!("{} channel(s)", tile_cache.previous_channel_group.len())
            },
        ));
        if tile_cache.over_budget_bytes > 0 {
            ui.colored_label(
                egui::Color32::YELLOW,
                format!(
                    "Visible working set is {} over budget; visible tiles are protected.",
                    format_bytes(tile_cache.over_budget_bytes)
                ),
            );
        }
        ui.label(format!(
            "Evictions: {} budget · {} channel change · peak {}",
            tile_cache.evictions_byte_budget,
            tile_cache.evictions_channel_change,
            format_bytes(tile_cache.peak_tracked_bytes),
        ));
        ui.separator();

        ui_memory_overview(
            ui,
            "Manually pin selected OME-Zarr channels and levels in CPU RAM.",
            Some(("Pinned total", self.pinned_levels.total_loaded_bytes())),
            self.system_memory.as_ref(),
        );
        ui.add_space(6.0);

        let rows = self
            .channel_layer_order
            .iter()
            .filter_map(|&gid| {
                self.channels.get(gid).map(|ch| MemoryChannelRow {
                    id: gid,
                    label: if ch.visible {
                        format!("{} (visible)", ch.name)
                    } else {
                        ch.name.clone()
                    },
                    visible: ch.visible,
                })
            })
            .collect::<Vec<_>>();
        ui_memory_channel_selector(
            ui,
            "mosaic-memory-channel-list",
            &rows,
            &mut self.memory_selected_channels,
        );
        if let Some(status) = self.control_actor_memory_state["status"].as_str()
            && !status.is_empty()
        {
            ui.label(status);
        }
        ui.separator();

        let selected_global_channels = self.selected_memory_global_channels();
        let selected_channel_count = selected_global_channels.len();

        let max_levels = self
            .sources
            .iter()
            .map(|src| src.levels.len())
            .max()
            .unwrap_or(0);
        if max_levels > 0 {
            ui.label("All ROIs");
            egui::Grid::new("mosaic-memory-all-grid")
                .num_columns(5)
                .striped(true)
                .show(ui, |ui| {
                    ui.strong("Level");
                    ui.strong("Eligible");
                    ui.strong("RAM");
                    ui.strong("State");
                    ui.strong("Action");
                    ui.end_row();

                    for level_idx in 0..max_levels {
                        let mut eligible = 0usize;
                        let mut loaded = 0usize;
                        let mut loading = 0usize;
                        let mut failed = 0usize;
                        let mut bytes = 0u64;

                        for item in &self.items {
                            let Some(source) = self.sources.get(item.id) else {
                                continue;
                            };
                            if source.levels.get(level_idx).is_none() {
                                continue;
                            }
                            let estimate = estimate_level_ram_bytes_for_channels(
                                source,
                                level_idx,
                                Some(&selected_global_channels),
                            )
                            .unwrap_or(0);
                            if estimate == 0 {
                                continue;
                            }
                            eligible += 1;
                            bytes = bytes.saturating_add(estimate);
                            match self.projected_memory_level_status(item.id, level_idx) {
                                MosaicPinnedLevelStatus::Unloaded => {}
                                MosaicPinnedLevelStatus::Loading => loading += 1,
                                MosaicPinnedLevelStatus::Loaded { .. } => loaded += 1,
                                MosaicPinnedLevelStatus::Failed(_) => failed += 1,
                            }
                        }

                        if eligible == 0 {
                            continue;
                        }

                        ui.label(level_idx.to_string());
                        ui.label(format!("{eligible} ROI(s)"));
                        let risk = self.memory_risk(bytes);
                        let risk_text = match risk.as_ref().map(|r| r.level) {
                            Some(MemoryRiskLevel::Danger) => " danger",
                            Some(MemoryRiskLevel::Warning) => " warning",
                            None => "",
                        };
                        ui.label(format!("{}{}", format_bytes(bytes), risk_text));
                        if loading > 0 {
                            ui.label(format!("Loading {loading}, loaded {loaded}/{eligible}"));
                        } else if loaded == eligible {
                            ui.label("Loaded for all");
                        } else if loaded > 0 || failed > 0 {
                            ui.label(format!("Loaded {loaded}/{eligible}, failed {failed}"));
                        } else {
                            ui.label("Not loaded");
                        }
                        ui.horizontal(|ui| {
                            if ui
                                .add_enabled(
                                    selected_channel_count > 0 && eligible > 0 && loading == 0,
                                    egui::Button::new("Load all"),
                                )
                                .clicked()
                            {
                                self.start_memory_load(
                                    format!(
                                        "Loading {} channel(s) from level {level_idx} into RAM for {eligible} ROI(s)",
                                        selected_channel_count
                                    ),
                                    serde_json::json!({
                                        "scope":"all",
                                        "level":level_idx,
                                        "channels":selected_global_channels,
                                    }),
                                    bytes,
                                );
                            }
                            if ui
                                .add_enabled(
                                    loaded > 0 || loading > 0 || failed > 0,
                                    egui::Button::new("Unload all"),
                                )
                                .clicked()
                            {
                                self.submit_native_control_intent(
                                    "memory.unpin",
                                    serde_json::json!({"scope":"all","level":level_idx}),
                                );
                            }
                        });
                        ui.end_row();
                    }
                });
            ui.separator();
        }

        let Some(item) = self.focused_item() else {
            ui.label("No focused ROI.");
            return;
        };
        let dataset_id = item.id;
        let item_dims = item.dataset.dims.clone();
        let sample_id = item.sample_id.clone();
        let levels = item.dataset.levels.clone();

        ui.label(format!("Focused ROI: {sample_id}"));
        ui.label("Loading is manual. The app estimates RAM usage but does not enforce a system-memory limit.");
        ui.separator();

        let Some(source) = self.sources.get(dataset_id).cloned() else {
            ui.label("Missing mosaic source metadata.");
            return;
        };

        egui::Grid::new(("mosaic-memory-grid", dataset_id))
            .num_columns(5)
            .striped(true)
            .show(ui, |ui| {
                ui.strong("Level");
                ui.strong("Shape");
                ui.strong("RAM");
                ui.strong("State");
                ui.strong("Action");
                ui.end_row();

                for (level_idx, level) in levels.iter().enumerate() {
                    let shape_y = level.shape.get(item_dims.y).copied().unwrap_or(0);
                    let shape_x = level.shape.get(item_dims.x).copied().unwrap_or(0);
                    let channels = item_dims
                        .c
                        .and_then(|c| level.shape.get(c).copied())
                        .unwrap_or(1);
                    let estimate = estimate_level_ram_bytes_for_channels(
                        &source,
                        level_idx,
                        Some(&selected_global_channels),
                    )
                    .unwrap_or(0);
                    let status = self.projected_memory_level_status(dataset_id, level_idx);

                    ui.label(level_idx.to_string());
                    ui.label(format!("{channels} x {shape_y} x {shape_x}"));
                    let risk = self.memory_risk(estimate);
                    let risk_text = match risk.as_ref().map(|r| r.level) {
                        Some(MemoryRiskLevel::Danger) => " danger",
                        Some(MemoryRiskLevel::Warning) => " warning",
                        None => "",
                    };
                    ui.label(format!("{}{}", format_bytes(estimate), risk_text));
                    if estimate == 0 {
                        ui.label("No selected channels");
                    } else {
                        match &status {
                            MosaicPinnedLevelStatus::Unloaded => {
                                ui.label("Not loaded");
                            }
                            MosaicPinnedLevelStatus::Loading => {
                                ui.label("Loading");
                            }
                            MosaicPinnedLevelStatus::Loaded {
                                bytes,
                                channels_loaded,
                            } => {
                                ui.label(format!(
                                    "Loaded ({}; {} ch)",
                                    format_bytes(*bytes),
                                    channels_loaded
                                ));
                            }
                            MosaicPinnedLevelStatus::Failed(err) => {
                                ui.colored_label(
                                    ui.visuals().warn_fg_color,
                                    format!("Failed: {err}"),
                                );
                            }
                        }
                    }

                    ui.horizontal(|ui| {
                        if ui
                            .add_enabled(
                                estimate > 0 && !matches!(status, MosaicPinnedLevelStatus::Loading),
                                egui::Button::new("Load"),
                            )
                            .clicked()
                        {
                            self.start_memory_load(
                                format!(
                                    "Loading {} channel(s) from ROI '{}' level {} into RAM",
                                    selected_channel_count, sample_id, level_idx
                                ),
                                serde_json::json!({
                                    "scope":"item",
                                    "item":dataset_id,
                                    "level":level_idx,
                                    "channels":selected_global_channels,
                                }),
                                estimate,
                            );
                        }
                        if ui
                            .add_enabled(
                                !matches!(status, MosaicPinnedLevelStatus::Unloaded),
                                egui::Button::new("Unload"),
                            )
                            .clicked()
                        {
                            self.submit_native_control_intent(
                                "memory.unpin",
                                serde_json::json!({
                                    "scope":"item",
                                    "item":dataset_id,
                                    "level":level_idx,
                                }),
                            );
                        }
                    });
                    ui.end_row();
                }
            });
    }

    pub(super) fn ui_contrast(&mut self, ui: &mut egui::Ui) {
        ui.heading("Contrast (global)");
        if self.channels.is_empty() {
            ui.label("No channels.");
            return;
        }

        let groups_before = self.layer_groups.clone();
        let mut selected_channel = self.selected_channel;
        let mut changed_channel = false;
        egui::ComboBox::from_label("Channel")
            .selected_text(
                self.channels
                    .get(selected_channel)
                    .map(|c| c.name.as_str())
                    .unwrap_or("-"),
            )
            .show_ui(ui, |ui| {
                let order = self.channel_layer_order.clone();
                for idx in order.into_iter() {
                    let Some(ch) = self.channels.get(idx) else {
                        continue;
                    };
                    changed_channel |= ui
                        .selectable_value(&mut selected_channel, idx, &ch.name)
                        .changed();
                }
            });
        if changed_channel {
            self.set_active_layer(MosaicLayerId::Channel(selected_channel));
        }

        let abs_max = self.abs_max.max(1.0);
        let Some(sel) = self.channels.get(selected_channel).cloned() else {
            return;
        };

        let selected_name = sel.name.clone();
        let mut selected_group: Option<u64> = self
            .layer_groups
            .channel_members
            .get(selected_name.as_str())
            .map(|m| m.group_id)
            .filter(|gid| {
                self.layer_groups
                    .channel_groups
                    .iter()
                    .any(|g| g.id == *gid)
            });
        let mut groups_changed = false;

        ui.horizontal(|ui| {
            ui.label("Group");
            egui::ComboBox::from_id_salt("mosaic-channel-group-select")
                .selected_text(
                    selected_group
                        .and_then(|gid| {
                            self.layer_groups
                                .channel_groups
                                .iter()
                                .find(|g| g.id == gid)
                        })
                        .map(|g| g.name.as_str())
                        .unwrap_or("(none)"),
                )
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut selected_group, None, "(none)");
                    for g in &self.layer_groups.channel_groups {
                        ui.selectable_value(&mut selected_group, Some(g.id), g.name.clone());
                    }
                });
            if ui
                .button("+ Group")
                .on_hover_text("Create a new group")
                .clicked()
            {
                let existing = self
                    .layer_groups
                    .channel_groups
                    .iter()
                    .map(|g| g.id)
                    .collect::<Vec<_>>();
                let id = layer_groups::next_group_id(&existing);
                self.layer_groups.channel_groups.push(
                    crate::data::project_config::ProjectChannelGroup {
                        id,
                        name: format!("Group {id}"),
                        expanded: true,
                        color_rgb: [255, 255, 255],
                    },
                );
                selected_group = Some(id);
                groups_changed = true;
            }
        });

        // Apply membership.
        let have_member = self
            .layer_groups
            .channel_members
            .get(selected_name.as_str())
            .is_some();
        if selected_group.is_none() && have_member {
            self.layer_groups
                .channel_members
                .remove(selected_name.as_str());
            groups_changed = true;
        } else if let Some(gid) = selected_group {
            match self
                .layer_groups
                .channel_members
                .get_mut(selected_name.as_str())
            {
                Some(m) => {
                    if m.group_id != gid {
                        m.group_id = gid;
                        groups_changed = true;
                    }
                }
                None => {
                    self.layer_groups.channel_members.insert(
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
        if let Some(m) = self
            .layer_groups
            .channel_members
            .get(selected_name.as_str())
        {
            inherit_group_color = m.inherit_color;
        }
        if let Some(gid) = selected_group {
            ui.horizontal(|ui| {
                if ui
                    .checkbox(&mut inherit_group_color, "Inherit group color")
                    .changed()
                {
                    if let Some(m) = self
                        .layer_groups
                        .channel_members
                        .get_mut(selected_name.as_str())
                    {
                        m.inherit_color = inherit_group_color;
                        groups_changed = true;
                    }
                }
                if inherit_group_color {
                    if let Some(group) = self
                        .layer_groups
                        .channel_groups
                        .iter_mut()
                        .find(|g| g.id == gid)
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

        let allow_channel_color = selected_group.is_none() || !inherit_group_color;
        let mut color_rgb = sel.color_rgb;
        ui.horizontal(|ui| {
            ui.label(if allow_channel_color {
                "Color"
            } else {
                "Color (override)"
            });
            ui.add_enabled_ui(allow_channel_color, |ui| {
                let mut color = egui::Color32::from_rgb(color_rgb[0], color_rgb[1], color_rgb[2]);
                if ui.color_edit_button_srgba(&mut color).changed() {
                    color_rgb = [color.r(), color.g(), color.b()];
                }
            });
        });
        if color_rgb != sel.color_rgb {
            self.commit_channel_color(selected_channel, color_rgb);
        }
        if groups_changed {
            ui.ctx().request_repaint();
        }

        let window = self
            .preview_channel_window(selected_channel)
            .or(sel.window)
            .unwrap_or((0.0, abs_max));
        let out = contrast::ui_contrast_window(
            ui,
            abs_max,
            window,
            contrast::ContrastUiOptions::standard("Set Max -> All"),
        );
        let (lo, hi) = out.window;

        if out.set_max_all_clicked {
            let windows = self
                .channels
                .iter()
                .enumerate()
                .map(|(index, channel)| {
                    let (mut minimum, _) = self
                        .preview_channel_window(index)
                        .or(channel.window)
                        .unwrap_or((0.0, abs_max));
                    minimum = minimum.clamp(0.0, abs_max);
                    let maximum = hi.clamp(0.0, abs_max);
                    let minimum = if maximum <= minimum {
                        (maximum - 1.0).clamp(0.0, abs_max)
                    } else {
                        minimum
                    };
                    (index, minimum, maximum)
                })
                .collect::<Vec<_>>();
            self.apply_channel_windows(&windows);
            self.commit_layer_groups_preview(groups_before);
            return;
        }

        if out.limits_touched || changed_channel {
            self.apply_channel_window_to_indices(&[selected_channel], lo, hi);
        }

        let mut note = sel.note.clone();
        if channel_notes::ui_channel_notes(ui, &sel.name, &mut note) {
            self.commit_channel_note(selected_channel, note);
        }
        self.commit_layer_groups_preview(groups_before);
    }
}
