use super::*;

impl OmeZarrViewerApp {
    pub(super) fn ui_layer_properties(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let active_layer_name = self.layer_display_name(self.active_layer);
        ui.horizontal(|ui| {
            ui.heading(active_layer_name);
            if crate::ui::help::help_button(ui, crate::ui::help::HelpTopic::PropertiesPanel) {
                self.active_help_topic = Some(crate::ui::help::HelpTopic::PropertiesPanel);
            }
        });
        ui.separator();
        self.ui_current_roi_summary(ui);
        ui.separator();

        let mut changed = false;
        ui.label("Transform");

        let active_layer = self.active_layer;
        let mut off = self.layer_offset_world(active_layer);

        let mut reset_clicked = false;
        ui.horizontal(|ui| {
            changed |= ui
                .add(egui::DragValue::new(&mut off.x).speed(5.0).prefix("x "))
                .changed();
            changed |= ui
                .add(egui::DragValue::new(&mut off.y).speed(5.0).prefix("y "))
                .changed();
            reset_clicked = ui
                .button("Reset")
                .on_hover_text("Reset selected visible layer translation to loaded position")
                .clicked();
        });

        if reset_clicked {
            changed |= self.reset_current_visible_move_targets_to_loaded();
        } else if changed {
            if let Some(dst) = self.layer_offset_world_mut(active_layer) {
                *dst = off;
            }
        }

        if let LayerId::Channel(idx0) = active_layer {
            let idx = idx0.min(self.channels.len().saturating_sub(1));

            let mut scale = self
                .channel_scales
                .get(idx)
                .copied()
                .unwrap_or(egui::Vec2::splat(1.0));
            let mut rot = self.channel_rotations_rad.get(idx).copied().unwrap_or(0.0);
            let mut deg = rot.to_degrees();

            ui.horizontal(|ui| {
                ui.label("Scale");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut scale.x)
                            .speed(0.02)
                            .range(0.01..=100.0)
                            .prefix("x "),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut scale.y)
                            .speed(0.02)
                            .range(0.01..=100.0)
                            .prefix("y "),
                    )
                    .changed();
                if ui.button("1x").clicked() {
                    scale = egui::Vec2::splat(1.0);
                    changed = true;
                }
            });

            ui.horizontal(|ui| {
                ui.label("Rotate");
                if ui
                    .add(
                        egui::DragValue::new(&mut deg)
                            .speed(1.0)
                            .range(-360.0..=360.0)
                            .suffix(" deg"),
                    )
                    .changed()
                {
                    changed = true;
                }
                if ui.button("0").clicked() {
                    deg = 0.0;
                    changed = true;
                }
            });

            rot = deg.to_radians();
            if let Some(dst) = self.channel_scales.get_mut(idx) {
                *dst = scale;
            }
            if let Some(dst) = self.channel_rotations_rad.get_mut(idx) {
                *dst = rot;
            }
        }

        if changed {
            self.hist_dirty = true;
            self.bump_render_id();
        }
        ui.separator();

        if let Some(group_id) = self.selected_channel_group_id {
            self.ui_group_contrast(ctx, ui, group_id);
            return;
        }

        match self.active_layer {
            LayerId::Channel(idx) => {
                self.selected_channel = idx.min(self.channels.len().saturating_sub(1));
                self.ui_contrast(ctx, ui);
                self.maybe_request_histogram(ctx);
            }
            LayerId::SpatialImage(id) => {
                if let Some(layer) = self
                    .spatial_image_layers
                    .images
                    .iter_mut()
                    .find(|l| l.id == id)
                {
                    if layer.ui_properties(ui) {
                        self.bump_render_id();
                    }
                } else {
                    ui.label("Not loaded.");
                }
            }
            LayerId::Points => {
                ui.checkbox(&mut self.cell_points.visible, "Visible");
                ui.add(
                    egui::Slider::new(&mut self.cell_points.style.radius_screen_px, 0.5..=20.0)
                        .text("Size")
                        .show_value(true)
                        .clamping(egui::SliderClamping::Always),
                );

                ui.separator();
                ui.label("Positive points");
                ui.horizontal(|ui| {
                    ui.label("Fill");
                    ui.color_edit_button_srgba(&mut self.cell_points.style.fill_positive);
                });
                ui.horizontal(|ui| {
                    ui.label("Stroke");
                    ui.add(
                        egui::DragValue::new(&mut self.cell_points.style.stroke_positive.width)
                            .speed(0.25)
                            .range(0.0..=10.0),
                    );
                    ui.color_edit_button_srgba(&mut self.cell_points.style.stroke_positive.color);
                });

                ui.separator();
                ui.label("Negative points");
                ui.horizontal(|ui| {
                    ui.label("Fill");
                    ui.color_edit_button_srgba(&mut self.cell_points.style.fill_negative);
                });
                ui.horizontal(|ui| {
                    ui.label("Stroke");
                    ui.add(
                        egui::DragValue::new(&mut self.cell_points.style.stroke_negative.width)
                            .speed(0.25)
                            .range(0.0..=10.0),
                    );
                    ui.color_edit_button_srgba(&mut self.cell_points.style.stroke_negative.color);
                });
            }
            LayerId::Annotation(id) => {
                let Some(idx) = self.annotation_layers.iter().position(|l| l.id == id) else {
                    ui.label("Annotation layer not found.");
                    return;
                };
                let mut groups_cfg = self.current_layer_groups();
                let mut groups_changed = false;
                ui.horizontal(|ui| {
                    ui.label("Name");
                    ui.text_edit_singleline(&mut self.annotation_layers[idx].name);
                });
                ui.separator();

                // Grouping (optional): visibility/tint can be controlled at group level.
                let mut selected_group: Option<u64> = groups_cfg
                    .annotation_members
                    .get(&id)
                    .map(|m| m.group_id)
                    .filter(|gid| groups_cfg.annotation_groups.iter().any(|g| g.id == *gid));
                ui.horizontal(|ui| {
                    ui.label("Group");
                    egui::ComboBox::from_id_salt(("annotation-group-select", id))
                        .selected_text(
                            selected_group
                                .and_then(|gid| {
                                    groups_cfg.annotation_groups.iter().find(|g| g.id == gid)
                                })
                                .map(|g| g.name.as_str())
                                .unwrap_or("(none)"),
                        )
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut selected_group, None, "(none)");
                            for g in &groups_cfg.annotation_groups {
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
                        let existing = groups_cfg
                            .annotation_groups
                            .iter()
                            .map(|g| g.id)
                            .collect::<Vec<_>>();
                        let id2 = layer_groups::next_group_id(&existing);
                        groups_cfg.annotation_groups.push(
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

                let have_member = groups_cfg.annotation_members.get(&id).is_some();
                if selected_group.is_none() && have_member {
                    groups_cfg.annotation_members.remove(&id);
                    groups_changed = true;
                } else if let Some(gid) = selected_group {
                    match groups_cfg.annotation_members.get_mut(&id) {
                        Some(m) => {
                            if m.group_id != gid {
                                m.group_id = gid;
                                groups_changed = true;
                            }
                        }
                        None => {
                            groups_cfg.annotation_members.insert(
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
                    let mut inherit_tint = groups_cfg
                        .annotation_members
                        .get(&id)
                        .map(|m| m.inherit_tint)
                        .unwrap_or(true);
                    ui.horizontal(|ui| {
                        if ui
                            .checkbox(&mut inherit_tint, "Inherit group tint")
                            .changed()
                        {
                            if let Some(m) = groups_cfg.annotation_members.get_mut(&id) {
                                m.inherit_tint = inherit_tint;
                                groups_changed = true;
                            }
                        }
                    });

                    if let Some(group) = groups_cfg
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
                if self.annotation_layers[idx].ui_properties(ui) {
                    self.bump_render_id();
                }
                ui.separator();
                if ui.button("Delete layer").clicked() {
                    self.annotation_layers.remove(idx);
                    if self.active_layer == LayerId::Annotation(id) {
                        self.active_layer = if !self.channels.is_empty() {
                            LayerId::Channel(self.selected_channel.min(self.channels.len() - 1))
                        } else {
                            LayerId::Points
                        };
                    }
                    self.rebuild_layer_orders();
                    self.bump_render_id();
                    return;
                }

                if groups_changed {
                    let new_groups = groups_cfg;
                    self.set_current_layer_groups(new_groups);
                    self.bump_render_id();
                }
            }
            LayerId::SegmentationLabels => {
                if !self.layer_is_available(LayerId::SegmentationLabels) {
                    ui.label("Segmentation labels require the GPU renderer.");
                    return;
                }

                ui.horizontal(|ui| {
                    ui.label("Label");

                    if self.dataset.is_root_label_mask() {
                        ui.label(self.seg_label_selected.clone());
                    } else if !self.seg_label_names.is_empty() {
                        let before = self.seg_label_selected.clone();
                        egui::ComboBox::from_id_salt("seg_label_select")
                            .selected_text(self.seg_label_selected.clone())
                            .show_ui(ui, |ui| {
                                for name in self.seg_label_names.clone() {
                                    ui.selectable_value(
                                        &mut self.seg_label_selected,
                                        name.clone(),
                                        name,
                                    );
                                }
                            });

                        if ui.button("Reload").clicked() || self.seg_label_selected != before {
                            let name = self.seg_label_selected.trim().to_string();
                            if name.is_empty() {
                                self.seg_label_status = "Label name is empty.".to_string();
                            } else {
                                match self.load_segmentation_labels(name.as_str()) {
                                    Ok(()) => {
                                        self.seg_label_status =
                                            format!("Loaded labels/{}.", name.as_str());
                                    }
                                    Err(err) => {
                                        self.seg_label_status =
                                            format!("Load labels/{} failed: {err}", name.as_str());
                                    }
                                }
                            }
                        }
                    } else {
                        ui.text_edit_singleline(&mut self.seg_label_input);
                        if ui.button("Load").clicked() {
                            let name = self.seg_label_input.trim().to_string();
                            if name.is_empty() {
                                self.seg_label_status = "Label name is empty.".to_string();
                            } else {
                                self.seg_label_selected = name.clone();
                                match self.load_segmentation_labels(name.as_str()) {
                                    Ok(()) => {
                                        self.seg_label_status =
                                            format!("Loaded labels/{}.", name.as_str());
                                    }
                                    Err(err) => {
                                        self.seg_label_status =
                                            format!("Load labels/{} failed: {err}", name.as_str());
                                    }
                                }
                            }
                        }
                    }

                    if !self.dataset.is_root_label_mask() && ui.button("Refresh").clicked() {
                        self.refresh_seg_label_names_for_current_roi();
                    }
                });

                if !self.seg_label_status.trim().is_empty() {
                    ui.label(self.seg_label_status.clone());
                }

                if self.label_cells.is_none() {
                    ui.label("Not loaded for this ROI.");
                    return;
                }

                ui.separator();
                ui.checkbox(&mut self.cells_outlines_visible, "Visible");
                ui.add(
                    egui::Slider::new(&mut self.cells_outlines_opacity, 0.0..=1.0)
                        .text("Opacity")
                        .show_value(true)
                        .clamping(egui::SliderClamping::Always),
                );
            }
            LayerId::SegmentationGeoJson => {
                let default_dir = self
                    .dataset
                    .source
                    .local_path()
                    .and_then(|p| p.parent())
                    .unwrap_or_else(|| Path::new("."));
                self.seg_geojson.ui_properties(ui, default_dir);
            }
            LayerId::SegmentationObjects => {
                let default_dir = self
                    .dataset
                    .source
                    .local_path()
                    .and_then(|p| p.parent())
                    .unwrap_or_else(|| Path::new("."));
                if let Some(action) = self.seg_objects.ui_properties(ui, default_dir) {
                    self.queue_object_source_action(action);
                }
            }
            LayerId::SpatialShape(id) => {
                if let Some(layer) = self.spatial_layers.shapes.iter_mut().find(|s| s.id == id) {
                    let default_dir = self
                        .dataset
                        .source
                        .local_path()
                        .and_then(|p| p.parent())
                        .unwrap_or_else(|| Path::new("."));
                    if layer.ui_properties(ui, default_dir) {
                        self.bump_render_id();
                    }
                } else {
                    ui.label("Not loaded.");
                }
            }
            LayerId::SpatialPoints => {
                if self.spatial_layers.points.is_some() {
                    let positive_targets = self.available_object_selection_targets();
                    let (mut changed, bounds, positive_cell_request) = {
                        let layer = self.spatial_layers.points.as_mut().expect("checked");
                        (
                            layer.ui_properties(ui, &positive_targets),
                            layer.bounds_world(),
                            layer.take_positive_cell_selection_request(),
                        )
                    };
                    if let Some((cell_ids, target)) = positive_cell_request {
                        let status = if let Some((matched_layers, matched_objects)) =
                            self.select_objects_by_ids_target(&cell_ids, target)
                        {
                            changed = true;
                            format!(
                                "Selected {matched_objects} object(s) across {matched_layers} layer(s)."
                            )
                        } else {
                            "No loaded object layers matched those cell IDs.".to_string()
                        };
                        if let Some(layer) = self.spatial_layers.points.as_mut() {
                            layer.set_cell_selection_status(status);
                        }
                    }
                    if changed {
                        self.bump_render_id();
                    }
                    if let Some(bounds) = bounds {
                        ui.separator();
                        ui.label(format!(
                            "Bounds: x [{:.0}, {:.0}]  y [{:.0}, {:.0}]",
                            bounds.min.x, bounds.max.x, bounds.min.y, bounds.max.y
                        ));
                        if ui.button("Fit to points").clicked() {
                            if let Some(viewport) = self.last_canvas_rect {
                                let off = self.layer_offset_world(LayerId::SpatialPoints);
                                self.camera
                                    .fit_to_world_rect(viewport, bounds.translate(off));
                                self.bump_render_id();
                            }
                        }
                    }
                } else {
                    ui.label("Not loaded.");
                }
            }
            LayerId::XeniumCells => {
                if let Some(layer) = self.xenium_layers.cells.as_mut() {
                    layer.ui_properties(ui);
                } else {
                    ui.label("Not loaded.");
                }
            }
            LayerId::XeniumTranscripts => {
                if let Some(layer) = self.xenium_layers.transcripts.as_mut() {
                    layer.ui_properties(ui);
                } else {
                    ui.label("Not loaded.");
                }
            }
            LayerId::Mask(id) => {
                let Some(idx) = self.mask_layers.iter().position(|l| l.id == id) else {
                    ui.label("Mask layer not found.");
                    return;
                };

                self.validate_mask_polygon_selection();
                let selected_polygon_idx = self
                    .selected_mask_polygon
                    .filter(|selection| selection.layer_id == id)
                    .map(|selection| selection.polygon_idx);
                let selected_vertex_idx = selected_polygon_idx.and(self.selected_mask_vertex);
                let selected_vertex_count = selected_polygon_idx
                    .and_then(|polygon_idx| {
                        self.mask_layers[idx]
                            .polygons_world
                            .get(polygon_idx)
                            .map(|poly| Self::mask_polygon_unique_vertex_count(poly))
                    })
                    .unwrap_or(0);

                let mut changed = false;
                let mut new_layer_clicked = false;
                let mut draw_tool_clicked = false;
                let mut clear_clicked = false;
                let mut delete_selected_polygon_clicked = false;
                let mut delete_clicked = false;
                let mut reload_from_roi_clicked = false;
                let mut reload_from_file: Option<PathBuf> = None;

                {
                    let layer = &mut self.mask_layers[idx];

                    ui.horizontal(|ui| {
                        ui.label("Name");
                        changed |= ui.text_edit_singleline(&mut layer.name).changed();
                    });

                    changed |= ui.checkbox(&mut layer.visible, "Visible").changed();
                    changed |= ui.checkbox(&mut layer.editable, "Editable").changed();

                    changed |= ui
                        .add(
                            egui::Slider::new(&mut layer.opacity, 0.0..=1.0)
                                .text("Opacity")
                                .show_value(true)
                                .clamping(egui::SliderClamping::Always),
                        )
                        .changed();
                    ui.horizontal(|ui| {
                        ui.label("Display");
                        for mode in [
                            MaskDisplayMode::OutlineOnly,
                            MaskDisplayMode::TranslucentFill,
                            MaskDisplayMode::FilledPreview,
                        ] {
                            changed |= ui
                                .selectable_value(&mut layer.display_mode, mode, mode.label())
                                .changed();
                        }
                    });
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut layer.width_screen_px, 0.25..=6.0)
                                .text("Width")
                                .show_value(true)
                                .clamping(egui::SliderClamping::Always),
                        )
                        .changed();

                    ui.horizontal(|ui| {
                        ui.label("Color");
                        let mut c = egui::Color32::from_rgb(
                            layer.color_rgb[0],
                            layer.color_rgb[1],
                            layer.color_rgb[2],
                        );
                        if ui.color_edit_button_srgba(&mut c).changed() {
                            layer.color_rgb = [c.r(), c.g(), c.b()];
                            changed = true;
                        }
                    });

                    if let Some(src) = layer.source_geojson.as_ref() {
                        ui.separator();
                        ui.label("Source (GeoJSON)");
                        ui.label(src.to_string_lossy());
                        ui.horizontal(|ui| {
                            if layer.name == "Exclusion masks" && !layer.editable {
                                reload_from_roi_clicked |= ui.button("Reload").clicked();
                            } else if ui.button("Reload").clicked() {
                                reload_from_file = Some(src.clone());
                            }
                        });
                    }

                    ui.separator();
                    if let Some(polygon_idx) = selected_polygon_idx {
                        ui.label(format!(
                            "Selected polygon {} of {}",
                            polygon_idx + 1,
                            layer.polygons_world.len()
                        ));
                        ui.label(format!("{selected_vertex_count} vertices"));
                        if let Some(vertex_idx) = selected_vertex_idx {
                            ui.label(format!("Vertex {} selected", vertex_idx + 1));
                        }
                        delete_selected_polygon_clicked |= ui.button("Delete polygon").clicked();
                    } else {
                        ui.label("No polygon selected.");
                    }

                    ui.separator();
                    ui.horizontal(|ui| {
                        new_layer_clicked |= ui.button("New layer").clicked();
                        draw_tool_clicked |= ui.button("Draw (tool)").clicked();
                        clear_clicked |= ui
                            .add_enabled(
                                !layer.polygons_world.is_empty(),
                                egui::Button::new("Clear"),
                            )
                            .clicked();
                    });

                    ui.horizontal(|ui| {
                        delete_clicked |= ui.button("Delete layer").clicked();
                    });
                }

                if reload_from_roi_clicked {
                    self.push_mask_undo_snapshot();
                    match self.ensure_exclusion_masks_loaded() {
                        Ok(_) => changed = true,
                        Err(err) => {
                            self.undo_stack.pop();
                            self.roi_selector
                                .set_status(format!("Reload masks failed: {err}"));
                        }
                    }
                } else if let Some(path) = reload_from_file {
                    match load_geojson_polylines_world(&path, 1.0, PolygonRingMode::AllRings) {
                        Ok(polys) => {
                            self.push_mask_undo_snapshot();
                            if let Some(layer) = self.mask_layers.get_mut(idx) {
                                layer.polygons_world = polys;
                                layer.raster_display = None;
                                changed = true;
                            }
                        }
                        Err(err) => self
                            .roi_selector
                            .set_status(format!("Reload masks failed: {err}")),
                    }
                }

                if new_layer_clicked {
                    self.push_mask_undo_snapshot();
                    let new_id = self.create_editable_mask_layer(None);
                    self.set_active_layer(LayerId::Mask(new_id));
                    changed = true;
                }
                if draw_tool_clicked {
                    self.tool_mode = ToolMode::DrawMaskPolygon;
                    self.drawing_mask_layer = Some(id);
                }
                if clear_clicked {
                    let should_record_undo = self
                        .mask_layers
                        .get(idx)
                        .is_some_and(|layer| !layer.polygons_world.is_empty());
                    if should_record_undo {
                        self.push_mask_undo_snapshot();
                    }
                    if let Some(layer) = self.mask_layers.get_mut(idx) {
                        layer.clear();
                        changed = true;
                    }
                    if self
                        .selected_mask_polygon
                        .is_some_and(|selection| selection.layer_id == id)
                    {
                        self.clear_mask_polygon_selection();
                    }
                    if self.drawing_mask_layer == Some(id) {
                        self.drawing_mask_polygon.clear();
                    }
                }
                if delete_selected_polygon_clicked && self.delete_selected_mask_polygon() {
                    changed = true;
                }
                if delete_clicked {
                    self.delete_mask_layer(id);
                    return;
                }

                if changed {
                    self.mark_mask_layers_project_dirty();
                    self.bump_render_id();
                }
            }
        }
    }
}
