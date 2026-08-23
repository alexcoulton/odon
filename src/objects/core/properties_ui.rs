//! Object properties UI, presentation, legends, color grouping, and filter configuration.

use super::*;

mod colors;
mod filters;
mod style;

impl ObjectsLayer {
    pub fn ui_properties(
        &mut self,
        ui: &mut egui::Ui,
        default_dir: &Path,
    ) -> Option<ObjectSourceUiAction> {
        self.ensure_filter_cache();
        self.ensure_color_groups();

        let mut source_action = None;
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.visible, "Visible");
            source_action =
                self.ui_topbar(ui, default_dir)
                    .map(|path| ObjectSourceUiAction::Load {
                        path,
                        options: None,
                    });
        });
        ui.add(
            egui::Slider::new(&mut self.opacity, 0.0..=1.0)
                .text("Opacity")
                .show_value(true)
                .clamping(egui::SliderClamping::Always),
        );
        ui.add(
            egui::Slider::new(&mut self.width_screen_px, 0.25..=6.0)
                .text("Width")
                .show_value(true)
                .clamping(egui::SliderClamping::Always),
        );
        ui.add_enabled_ui(self.display_mode == ObjectDisplayMode::Polygons, |ui| {
            ui.checkbox(&mut self.fast_rendering, "Fast rendering");
        });
        ui.add_enabled_ui(self.display_mode == ObjectDisplayMode::Polygons, |ui| {
            ui.checkbox(&mut self.fill_cells, "Fill cells");
            ui.add_enabled(
                self.fill_cells,
                egui::Slider::new(&mut self.fill_opacity, 0.0..=1.0)
                    .text("Fill opacity")
                    .show_value(true)
                    .clamping(egui::SliderClamping::Always),
            );
        });
        ui.add(
            egui::Slider::new(&mut self.selected_fill_opacity, 0.0..=1.0)
                .text("Selected fill")
                .show_value(true)
                .clamping(egui::SliderClamping::Always),
        );
        ui.horizontal(|ui| {
            ui.label("Color");
            let mut c =
                egui::Color32::from_rgb(self.color_rgb[0], self.color_rgb[1], self.color_rgb[2]);
            if ui.color_edit_button_srgba(&mut c).changed() {
                self.color_rgb = [c.r(), c.g(), c.b()];
            }
        });
        ui.horizontal(|ui| {
            ui.label("Color by");
            let mut next_color_property = match self.color_mode {
                ObjectColorMode::Single => String::new(),
                ObjectColorMode::ByProperty => self.color_property_key.clone(),
            };
            egui::ComboBox::from_id_salt("seg_objects_color_mode")
                .selected_text(if next_color_property.is_empty() {
                    "Single color".to_string()
                } else {
                    next_color_property.clone()
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut next_color_property, String::new(), "Single color");
                    for key in &self.color_property_keys {
                        ui.selectable_value(&mut next_color_property, key.clone(), key);
                    }
                    if let Some(source) = self.lazy_parquet_source.as_ref() {
                        let lazy_keys = source
                            .available_property_columns
                            .iter()
                            .filter(|key| !source.loaded_property_columns.contains(*key))
                            .filter(|key| !self.property_store.has_loaded(key))
                            .filter(|key| {
                                !self.color_property_keys.iter().any(|loaded| loaded == *key)
                            })
                            .cloned()
                            .collect::<Vec<_>>();
                        if !lazy_keys.is_empty() {
                            ui.separator();
                            for key in lazy_keys {
                                ui.selectable_value(
                                    &mut next_color_property,
                                    key.clone(),
                                    format!("{key} (load)"),
                                );
                            }
                        }
                    }
                });
            self.set_color_by_property(
                (!next_color_property.is_empty()).then_some(next_color_property),
            );
        });
        ui.horizontal(|ui| {
            ui.add(
                egui::DragValue::new(&mut self.downsample_factor)
                    .speed(0.1)
                    .prefix("Downsample "),
            );
            if ui
                .add_enabled(self.loaded_geojson.is_some(), egui::Button::new("Reload"))
                .clicked()
            {
                source_action = Some(ObjectSourceUiAction::Reload);
            }
            if ui
                .add_enabled(self.loaded_geojson.is_some(), egui::Button::new("Clear"))
                .clicked()
            {
                source_action = Some(ObjectSourceUiAction::Clear);
            }
        });
        ui.label(format!("Objects: {}", self.object_count()));
        if let Some(path) = self.loaded_geojson.as_ref() {
            ui.label(path.to_string_lossy().to_string());
        } else {
            ui.label("Not loaded");
        }
        if !self.status.is_empty() {
            ui.label(self.status.clone());
        }

        ui.separator();
        ui.label("Filter");
        let mut filter_changed = false;

        ui.horizontal(|ui| {
            for mode in [ObjectFilterMode::Simple, ObjectFilterMode::Query] {
                filter_changed |= ui
                    .selectable_value(&mut self.filter_mode, mode, mode.label())
                    .on_hover_text(match mode {
                        ObjectFilterMode::Simple => "Use row-based object filters",
                        ObjectFilterMode::Query => "Use a boolean expression query",
                    })
                    .changed();
            }
        });

        match self.filter_mode {
            ObjectFilterMode::Simple => {
                ui.horizontal(|ui| {
                    ui.label("Match");
                    for logic in [ObjectFilterLogic::All, ObjectFilterLogic::Any] {
                        filter_changed |= ui
                            .selectable_value(&mut self.filter_logic, logic, logic.label())
                            .on_hover_text(match logic {
                                ObjectFilterLogic::All => {
                                    "Show objects matching every enabled filter"
                                }
                                ObjectFilterLogic::Any => {
                                    "Show objects matching at least one enabled filter"
                                }
                            })
                            .changed();
                    }
                });
                self.ensure_filter_clause_row();
                let filter_candidates = self.filter_property_candidates();
                let filter_value_options = self.filter_value_options_by_key(64);
                let mut remove_filter_clause = None;
                let mut add_filter_clause = false;
                for idx in 0..self.filter_clauses.len() {
                    let is_last = idx == self.filter_clauses.len().saturating_sub(1);
                    ui.horizontal(|ui| {
                        let clause = &mut self.filter_clauses[idx];
                        filter_changed |= ui.checkbox(&mut clause.enabled, "").changed();
                        let previous_property_key = clause.property_key.clone();
                        egui::ComboBox::from_id_salt(format!("seg_objects_filter_key_{idx}"))
                            .selected_text(clause.property_key.clone())
                            .show_ui(ui, |ui| {
                                filter_changed |= ui
                                    .selectable_value(
                                        &mut clause.property_key,
                                        "id".to_string(),
                                        "id",
                                    )
                                    .changed();
                                for (key, needs_load) in &filter_candidates {
                                    let label = if *needs_load {
                                        format!("{key} (load)")
                                    } else {
                                        key.clone()
                                    };
                                    filter_changed |= ui
                                        .selectable_value(
                                            &mut clause.property_key,
                                            key.clone(),
                                            label,
                                        )
                                        .changed();
                                }
                            });
                        if clause.property_key != previous_property_key {
                            clause.query.clear();
                        }

                        if let Some(options) = filter_value_options.get(&clause.property_key) {
                            let selected_text = if clause.query.trim().is_empty() {
                                "(select value)".to_string()
                            } else {
                                clause.query.clone()
                            };
                            ui.add_enabled_ui(clause.enabled, |ui| {
                                egui::ComboBox::from_id_salt(format!(
                                    "seg_objects_filter_value_{idx}"
                                ))
                                .selected_text(selected_text)
                                .show_ui(ui, |ui| {
                                    for value in options {
                                        filter_changed |= ui
                                            .selectable_value(
                                                &mut clause.query,
                                                value.clone(),
                                                value.as_str(),
                                            )
                                            .changed();
                                    }
                                });
                            });
                        } else {
                            filter_changed |= ui
                                .add_enabled(
                                    clause.enabled,
                                    egui::TextEdit::singleline(&mut clause.query)
                                        .hint_text("contains...")
                                        .desired_width(180.0),
                                )
                                .changed();
                        }
                        if ui
                            .small_button("-")
                            .on_hover_text("Remove filter")
                            .clicked()
                        {
                            remove_filter_clause = Some(idx);
                        }
                        if is_last && ui.small_button("+").on_hover_text("Add filter").clicked() {
                            add_filter_clause = true;
                        }
                    });
                }
                if add_filter_clause {
                    self.filter_clauses.push(ObjectFilterClause::default());
                    filter_changed = true;
                }
                if let Some(idx) = remove_filter_clause {
                    if idx < self.filter_clauses.len() {
                        self.filter_clauses.remove(idx);
                        self.ensure_filter_clause_row();
                        filter_changed = true;
                    }
                }
                ui.horizontal(|ui| {
                    if ui.button("Clear").clicked() {
                        self.filter_clauses = vec![ObjectFilterClause::default()];
                        filter_changed = true;
                    }
                });
            }
            ObjectFilterMode::Query => {
                ui.add(
                    egui::TextEdit::multiline(&mut self.filter_query_text)
                        .hint_text(
                            "(broad_cell_type == \"immune_lymphoid\" and zz_mask_cd3)\n\
                             or (broad_cell_type == \"immune_myeloid\" and zz_mask_hla_dr)",
                        )
                        .desired_rows(4)
                        .desired_width(f32::INFINITY),
                );
                ui.horizontal(|ui| {
                    if ui.button("Apply").clicked() {
                        filter_changed |= self.apply_filter_query_text();
                    }
                    if ui.button("Clear").clicked() {
                        self.filter_query_text.clear();
                        self.filter_query_expr = None;
                        self.filter_query_error = None;
                        filter_changed = true;
                    }
                });
                if let Some(error) = self.filter_query_error.as_ref() {
                    ui.colored_label(egui::Color32::from_rgb(220, 80, 80), error);
                } else if self.filter_query_expr.is_some() {
                    ui.label("Query active.");
                } else if !self.filter_query_text.trim().is_empty() {
                    ui.label("Apply the query to update the filter.");
                }
            }
        }
        if filter_changed {
            self.reconcile_filter_clauses();
            self.ensure_active_filter_properties_loaded();
            self.invalidate_filter_cache();
            self.ensure_filter_cache();
            self.ensure_color_groups();
        }
        ui.label(format!(
            "Visible after filter: {} / {}",
            self.filtered_count(),
            self.object_count()
        ));

        if self.color_mode == ObjectColorMode::ByProperty {
            ui.separator();
            ui.label(format!("Legend: {}", self.color_property_key));
            if let Some(entries) = self.active_color_legend_entries() {
                self.color_level_overrides_property_key = self.color_property_key.clone();
                egui::ScrollArea::vertical()
                    .id_salt("seg_objects_legend_scroll")
                    .max_height(140.0)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        for entry in entries {
                            let default_color = entry.color_rgb;
                            let override_style = self
                                .color_level_overrides
                                .entry(entry.value_label.clone())
                                .or_default();
                            let mut visible = override_style.visible;
                            let color_rgb = override_style.color_rgb.unwrap_or(default_color);
                            let mut color =
                                egui::Color32::from_rgb(color_rgb[0], color_rgb[1], color_rgb[2]);
                            ui.horizontal(|ui| {
                                if ui.checkbox(&mut visible, "").changed() {
                                    override_style.visible = visible;
                                }
                                if ui.color_edit_button_srgba(&mut color).changed() {
                                    let next_rgb = [color.r(), color.g(), color.b()];
                                    override_style.color_rgb =
                                        (next_rgb != default_color).then_some(next_rgb);
                                }
                                ui.label(format!("{} ({})", entry.value_label, entry.count));
                            });
                        }
                    });
            }
        }

        ui.separator();
        ui.label("Selection");
        ui.checkbox(&mut self.show_selection_overlay, "Show selection overlay");
        ui.label(format!("Selected: {}", self.selection_count()));
        if self.selection_count() > 0 {
            let (_count, total_area, mean_area) = self.selection_summary();
            ui.label(format!("Selected area total: {:.2}", total_area));
            ui.label(format!("Selected area mean: {:.2}", mean_area));
            ui.horizontal(|ui| {
                if ui.button("Clear selection").clicked() {
                    self.clear_selection();
                }
            });
        }
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    self.filtered_count() > 0,
                    egui::Button::new("Select filtered"),
                )
                .clicked()
            {
                self.select_filtered_objects();
            }
            if ui
                .add_enabled(
                    self.selection_count() > 0,
                    egui::Button::new("Export selected..."),
                )
                .clicked()
            {
                let _ = self.export_selected_with_dialog(default_dir);
            }
            if ui
                .add_enabled(
                    self.filtered_count() > 0,
                    egui::Button::new("Export filtered..."),
                )
                .clicked()
            {
                let _ = self.export_filtered_with_dialog(default_dir);
            }
        });

        ui.separator();
        self.ui_selection_elements_editor(ui);
        ui.separator();
        ui.label("Primary object");
        if let Some(idx) = self.selected_object_index {
            let selected_details = self
                .objects
                .as_ref()
                .and_then(|objects| objects.get(idx))
                .map(|obj| {
                    (
                        obj.id.clone(),
                        obj.area_px,
                        obj.perimeter_px,
                        obj.centroid_world,
                        self.loaded_property_display_pairs(idx, obj),
                    )
                });

            if let Some((
                obj_id,
                obj_area_px,
                obj_perimeter_px,
                obj_centroid_world,
                obj_properties,
            )) = selected_details
            {
                ui.horizontal(|ui| {
                    ui.label(format!("id: {}", obj_id));
                    if ui.button("Clear").clicked() {
                        self.clear_selection();
                    }
                });
                ui.label(format!("area_px: {:.2}", obj_area_px));
                ui.label(format!("perimeter_px: {:.2}", obj_perimeter_px));
                ui.label(format!(
                    "centroid: ({:.2}, {:.2})",
                    obj_centroid_world.x, obj_centroid_world.y
                ));
                egui::ScrollArea::vertical()
                    .id_salt("seg_objects_properties_scroll")
                    .max_height(220.0)
                    .show(ui, |ui| {
                        for (key, value_text) in &obj_properties {
                            ui.horizontal(|ui| {
                                ui.monospace(format!("{key}:"));
                                ui.label(value_text);
                            });
                        }
                    });
            } else {
                ui.label("No object selected");
            }
        } else {
            ui.label("No object selected");
        }

        ui.separator();
        ui.label("Object table");
        let table_indices = self.table_indices();
        let table_len = table_indices.len();
        let table_preview = table_indices.iter().take(300).copied().collect::<Vec<_>>();
        if table_len == 0 {
            ui.label("No objects match the current filter");
        } else {
            ui.label(format!("Showing {} of {}", table_preview.len(), table_len));
            egui::ScrollArea::vertical()
                .id_salt("seg_objects_table_scroll")
                .max_height(260.0)
                .show(ui, |ui| {
                    for idx in &table_preview {
                        let Some(obj) = self.objects.as_ref().and_then(|objs| objs.get(*idx))
                        else {
                            continue;
                        };
                        let selected = self.selected_object_indices.contains(idx);
                        let focused = self.selected_object_index == Some(*idx);
                        let label = format!("{}  area {:.1}", obj.id, obj.area_px);
                        if ui.selectable_label(selected || focused, label).clicked() {
                            if !self.has_live_analysis_selection() {
                                self.selected_object_indices.clear();
                                self.selected_object_indices.insert(*idx);
                            }
                            self.selected_object_index = Some(*idx);
                            self.rebuild_selection_render_lods();
                            self.clear_measurements();
                            self.invalidate_table_cache();
                        }
                    }
                });
        }
        source_action
    }
}
