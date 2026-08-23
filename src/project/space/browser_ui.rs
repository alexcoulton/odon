//! Primary project browser UI and recent-project presentation.

use super::*;

impl ProjectSpace {
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        current_dataset_root: Option<&Path>,
    ) -> Option<ProjectSpaceAction> {
        self.set_current_dataset_root(current_dataset_root);
        let mut action = None;
        let block_page_wheel = ui.ctx().pointer_hover_pos().is_some_and(|pos| {
            self.roi_list_hover_rect
                .is_some_and(|rect| rect.contains(pos))
        });
        let page_scroll_source = egui::scroll_area::ScrollSource {
            mouse_wheel: !block_page_wheel,
            ..egui::scroll_area::ScrollSource::ALL
        };

        egui::ScrollArea::vertical()
            .id_salt("project-space-page")
            .scroll_source(page_scroll_source)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.heading("Project");
                ui.label(format!(
                    "{} ROI(s), {} selected",
                    self.config.rois.len(),
                    self.selected.len()
                ));
                ui.add_space(6.0);

                ui.separator();
                ui.horizontal_wrapped(|ui| {
                    if crate::ui::help::help_button(ui, HelpTopic::ImportOpen) {
                        action = Some(ProjectSpaceAction::ShowHelp(HelpTopic::ImportOpen));
                    }
                    if ui.button("Import Samplesheet CSV...").clicked() {
                        if let Some(path) = FileDialog::new()
                            .add_filter("CSV", &["csv"])
                            .set_title("Import Samplesheet CSV")
                            .pick_file()
                        {
                            if self.submit_control_intent(
                                "project.samplesheets.import",
                                serde_json::json!({"path":path}),
                            ) {
                                self.status = "Importing samplesheet...".to_string();
                            } else if let Err(err) = self.import_rois_from_csv(&path) {
                                self.status = format!("Import failed: {err}");
                            }
                        }
                    }
                    if ui
                        .add_enabled(
                            self.exportable_local_roi_count() > 0,
                            egui::Button::new("Export Samplesheet CSV..."),
                        )
                        .clicked()
                    {
                        let default_path = self.default_samplesheet_export_path();
                        let default_name = default_path
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("samplesheet.csv")
                            .to_string();
                        let mut dialog = FileDialog::new()
                            .add_filter("CSV", &["csv"])
                            .set_file_name(&default_name)
                            .set_title("Export Samplesheet CSV");
                        if let Some(parent) = default_path.parent() {
                            dialog = dialog.set_directory(parent);
                        }
                        if let Some(path) = dialog.save_file() {
                            if self.submit_control_intent(
                                "project.samplesheets.export",
                                serde_json::json!({"path":path}),
                            ) {
                                self.status = "Exporting samplesheet...".to_string();
                            } else if let Err(err) = self.export_samplesheet_csv(&path) {
                                self.status = format!("Export failed: {err}");
                            }
                        }
                    }
                    if ui.button("Add OME-Zarr Root...").clicked() {
                        if let Some(path) = FileDialog::new()
                            .set_title("Choose Folder Containing OME-Zarr ROIs")
                            .pick_folder()
                        {
                            if self.submit_control_intent(
                                "project.discovery.add_root",
                                serde_json::json!({"path":path}),
                            ) {
                                self.status = "Discovering project datasets...".to_string();
                            } else if let Err(err) = self.import_rois_from_root(&path) {
                                self.status = format!("OME-Zarr root import failed: {err}");
                            }
                        }
                    }
                    if ui.button("Open TIFF / OME-TIFF...").clicked() {
                        if let Some(path) = FileDialog::new()
                            .add_filter("TIFF / OME-TIFF", &["tif", "tiff"])
                            .set_title("Open TIFF / OME-TIFF Image")
                            .pick_file()
                        {
                            action = Some(ProjectSpaceAction::OpenLocalPath(path));
                        }
                    }
                    if ui.button("Open Remote...").clicked() {
                        action = Some(ProjectSpaceAction::OpenRemoteDialog);
                    }
                    if ui.button("Add Seg Search Root...").clicked() {
                        if let Some(path) = FileDialog::new()
                            .set_title("Add Segmentation Search Root")
                            .pick_folder()
                        {
                            self.add_segmentation_search_root(path);
                        }
                    }
                    if ui
                        .add_enabled(
                            !self.config.rois.is_empty(),
                            egui::Button::new("Auto-match Seg"),
                        )
                        .clicked()
                    {
                        self.auto_match_segmentations(false);
                    }
                    if ui.button("Edit config (JSON)").clicked() {
                        self.config_json_dirty = false;
                        self.config_json = serde_json::to_string_pretty(&self.config)
                            .unwrap_or_else(|_| "{}".to_string());
                        self.config_json_status.clear();
                    }
                });
                ui.add_space(6.0);
                self.ui_object_cache(ui, &mut action);
                ui.add_space(6.0);
                self.ui_views_launcher(ui);
                ui.add_space(6.0);
                ui.heading("ROIs");
                let browse = crate::ui::roi_browser::ui(
                    ui,
                    "project-space-roi-browser",
                    &self.config.rois,
                    None,
                    &mut self.roi_browse,
                );
                if browse.changed || self.roi_browse.has_filters() {
                    self.sync_filtered_selection(&browse.filtered_indices);
                }
                let has_rois = !self.config.rois.is_empty();
                let files_hovered = ui.ctx().input(|i| !i.raw.hovered_files.is_empty());
                ui.horizontal_wrapped(|ui| {
            ui.label(format!(
                "Showing {} / {} ROI(s).",
                browse.filtered_indices.len(),
                browse.total_count
            ));
            if has_rois {
                ui.separator();
                ui.label("Drop OME-Zarr folders, Xenium folders, or TIFF files here to add more.");
            }
        });
                let seg_roots = self.resolved_segmentation_search_roots();
                if !seg_roots.is_empty() {
                    ui.add_space(4.0);
                    ui.collapsing("Segmentation search roots", |ui| {
                        let mut remove_idx = None;
                        for (idx, root) in self
                            .config
                            .mosaic_segmentation_search_roots
                            .iter()
                            .enumerate()
                        {
                            ui.horizontal(|ui| {
                                ui.label(root.to_string_lossy());
                                if ui.small_button("Remove").clicked() {
                                    remove_idx = Some(idx);
                                }
                            });
                        }
                        if self.config.mosaic_segmentation_search_roots.is_empty() {
                            ui.label("Using project directory and ROI parent folders.");
                        }
                        if let Some(idx) = remove_idx {
                            self.remove_segmentation_search_root(idx);
                        }
                    });
                }
                if !self.config_json.is_empty() {
                    ui.add_space(6.0);
                    ui.label("Project config (JSON)");
                    let resp = ui.add(
                        egui::TextEdit::multiline(&mut self.config_json)
                            .desired_rows(10)
                            .font(egui::TextStyle::Monospace),
                    );
                    if resp.changed() {
                        self.config_json_dirty = true;
                    }
                    ui.horizontal(|ui| {
                        if ui.button("Apply").clicked() {
                            match serde_json::from_str::<ProjectConfig>(&self.config_json) {
                                Ok(cfg) => {
                                    if self.submit_control_intent(
                                        "project.create",
                                        serde_json::json!({"config":cfg}),
                                    ) {
                                        self.config_json_status = "Applying...".to_string();
                                    } else {
                                        self.config = cfg;
                                        self.config_generation =
                                            self.config_generation.wrapping_add(1);
                                        self.config_json_dirty = false;
                                        self.config_json_status = "Applied.".to_string();
                                    }
                                }
                                Err(err) => self.config_json_status = format!("Parse error: {err}"),
                            }
                        }
                        if ui.button("Refresh").clicked() {
                            self.config_json_dirty = false;
                            self.config_json = serde_json::to_string_pretty(&self.config)
                                .unwrap_or_else(|_| "{}".to_string());
                            self.config_json_status.clear();
                        }
                        if ui.button("Close").clicked() {
                            self.config_json.clear();
                            self.config_json_status.clear();
                            self.config_json_dirty = false;
                        }
                    });
                    if !self.config_json_status.is_empty() {
                        ui.label(self.config_json_status.clone());
                    }
                    ui.separator();
                }

                let mut clicked_index: Option<usize> = None;
                let mut clicked_key: Option<String> = None;
                let mut clicked_double: bool = false;

                let select_all_shortcut =
                    ui.input(|i| i.key_pressed(egui::Key::A) && i.modifiers.command);
                if select_all_shortcut
                    && !ui.ctx().wants_keyboard_input()
                    && !browse.filtered_indices.is_empty()
                {
                    let ids = browse
                        .filtered_indices
                        .iter()
                        .filter_map(|&index| self.config.rois.get(index))
                        .map(|roi| roi.id.clone())
                        .collect::<Vec<_>>();
                    if let Err(error) = self.select_roi_ids(&ids, "replace") {
                        self.status = error;
                    }
                }

                let visible_indices = browse.filtered_indices;

                ui.label(format!("ROI list ({})", visible_indices.len()));
                let mut roi_frame = egui::Frame::group(ui.style());
                if files_hovered {
                    roi_frame = roi_frame
                        .fill(ui.visuals().selection.bg_fill.gamma_multiply(0.18))
                        .stroke(egui::Stroke::new(2.0, ui.visuals().selection.stroke.color));
                }
                let roi_frame_response = roi_frame.show(ui, |ui| {
                    let list_height = if has_rois {
                        ui.available_height().clamp(180.0, 280.0)
                    } else {
                        ui.available_height().clamp(220.0, 360.0)
                    };
                    ui.set_min_height(list_height);
                    if !has_rois {
                        ui.allocate_ui_with_layout(
                            egui::vec2(ui.available_width(), list_height),
                            egui::Layout::top_down(egui::Align::Center),
                            |ui| {
                                let title_size = if ui.available_width() < 360.0 {
                                    16.0
                                } else if ui.available_width() < 560.0 {
                                    18.0
                                } else {
                                    22.0
                                };
                                ui.add_space((list_height * 0.22).max(18.0));
                                ui.add_sized(
                            [ui.available_width(), 0.0],
                            egui::Label::new(
                                egui::RichText::new(
                                    "Drop OME-Zarr folders, Xenium folders, or TIFF files here",
                                )
                                .size(title_size)
                                .strong(),
                            )
                            .wrap(),
                        );
                                ui.add_space(8.0);
                                ui.add_sized(
                                    [ui.available_width(), 0.0],
                                    egui::Label::new(
                                        "Imported datasets will appear as ROIs in this list.",
                                    )
                                    .wrap(),
                                );
                                ui.add_sized(
                            [ui.available_width(), 0.0],
                            egui::Label::new(
                                "You can also use Import Samplesheet CSV or Add OME-Zarr Root.",
                            )
                            .wrap(),
                        );
                            },
                        );
                        return;
                    }
                    egui::ScrollArea::vertical()
                        .id_salt("project-roi-list")
                        .max_height(list_height)
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            if visible_indices.is_empty() {
                                ui.label("No ROIs match the current browse filters.");
                            }
                            for &roi_idx in &visible_indices {
                                let Some(it) = self.config.rois.get(roi_idx) else {
                                    continue;
                                };
                                ui.push_id(roi_idx, |ui| {
                                    let name =
                                        it.display_name.clone().unwrap_or_else(|| it.id.clone());
                                    let source_key = it.source_key();
                                    let tooltip = it.source_display();
                                    let is_selected = source_key
                                        .as_ref()
                                        .is_some_and(|key| self.selected.contains(key));
                                    let is_focused = source_key
                                        .as_ref()
                                        .zip(self.focused.as_ref())
                                        .is_some_and(|(key, focused)| key == focused);
                                    let row_label = if it.segpath.is_some() {
                                        format!("{name}  [seg]")
                                    } else {
                                        name
                                    };

                                    let row_width = ui.available_width();
                                    let resp = ui.add_sized(
                                        [row_width, 0.0],
                                        egui::Button::new(row_label).selected(is_selected),
                                    );
                                    let resp = resp.on_hover_text(tooltip);
                                    if resp.clicked() || resp.double_clicked() {
                                        clicked_index = Some(roi_idx);
                                        clicked_key = source_key;
                                        clicked_double = resp.double_clicked();
                                    }

                                    if is_focused {
                                        let stroke = ui.visuals().selection.stroke;
                                        let stroke =
                                            egui::Stroke::new(stroke.width.max(2.0), stroke.color);
                                        ui.painter().rect_stroke(
                                            resp.rect.shrink(0.5),
                                            egui::CornerRadius::same(6),
                                            stroke,
                                            egui::StrokeKind::Middle,
                                        );
                                    }
                                    ui.separator();
                                });
                            }
                        });
                });
                self.roi_list_hover_rect = has_rois.then_some(roi_frame_response.response.rect);

                if let (Some(i), Some(_source_key)) = (clicked_index, clicked_key.clone()) {
                    let modifiers = ui.input(|inp| inp.modifiers);
                    let cmd = modifiers.command;
                    let shift = modifiers.shift;

                    let focused_idx = self.focused.as_ref().and_then(|p| {
                        visible_indices.iter().position(|&idx| {
                            self.config
                                .rois
                                .get(idx)
                                .and_then(ProjectRoi::source_key)
                                .as_deref()
                                == Some(p.as_str())
                        })
                    });
                    let clicked_visible_idx = visible_indices
                        .iter()
                        .position(|&idx| idx == i)
                        .unwrap_or_default();

                    if shift {
                        let mut ids = if let Some(fi) = focused_idx {
                            let a = fi.min(clicked_visible_idx);
                            let b = fi.max(clicked_visible_idx);
                            visible_indices[a..=b]
                                .iter()
                                .filter_map(|&index| self.config.rois.get(index))
                                .map(|roi| roi.id.clone())
                                .collect::<Vec<_>>()
                        } else {
                            self.config
                                .rois
                                .get(i)
                                .map(|roi| vec![roi.id.clone()])
                                .unwrap_or_default()
                        };
                        if let Some(clicked_id) = self.config.rois.get(i).map(|roi| roi.id.clone())
                            && let Some(position) = ids.iter().position(|id| id == &clicked_id)
                        {
                            ids.remove(position);
                            ids.push(clicked_id);
                        }
                        if let Err(error) = self.select_roi_ids(&ids, "replace") {
                            self.status = error;
                        }
                    } else if cmd {
                        if let Some(id) = self.config.rois.get(i).map(|roi| roi.id.clone())
                            && let Err(error) = self.select_roi_ids(&[id], "toggle")
                        {
                            self.status = error;
                        }
                    } else {
                        if let Some(id) = self.config.rois.get(i).map(|roi| roi.id.clone())
                            && let Err(error) = self.select_roi_ids(&[id], "replace")
                        {
                            self.status = error;
                        }
                    }

                    if clicked_double {
                        if let Some(roi) = self.config.rois.get(i).cloned() {
                            action = Some(ProjectSpaceAction::Open(roi));
                        }
                    }
                }

                ui.add_space(8.0);
                ui.horizontal_wrapped(|ui| {
                    let selected_count = self.selected.len();
                    let all_visible_selected = !visible_indices.is_empty()
                        && visible_indices.iter().all(|&idx| {
                            self.config
                                .rois
                                .get(idx)
                                .and_then(ProjectRoi::source_key)
                                .as_ref()
                                .is_some_and(|key| self.selected.contains(key))
                        });
                    let mosaic_enabled = selected_count >= 2
                        && self
                            .selected_rois()
                            .iter()
                            .all(|roi| roi.local_path().map(can_open_in_mosaic).unwrap_or(true));
                    let open_clicked = ui
                        .add_enabled(self.focused_roi().is_some(), egui::Button::new("Open"))
                        .clicked();
                    let open_mosaic_clicked = ui
                        .add_enabled(
                            mosaic_enabled,
                            egui::Button::new(format!("Open mosaic ({selected_count})")),
                        )
                        .clicked();
                    let select_all_clicked = ui
                        .add_enabled(
                            !visible_indices.is_empty(),
                            egui::Button::new(if self.roi_browse.has_filters() {
                                if all_visible_selected {
                                    "All visible selected"
                                } else {
                                    "Select visible"
                                }
                            } else {
                                "Select all"
                            }),
                        )
                        .clicked();
                    let remove_clicked = ui
                        .add_enabled(selected_count > 0, egui::Button::new("Remove"))
                        .clicked();
                    let clear_clicked = ui
                        .add_enabled(!self.config.rois.is_empty(), egui::Button::new("Clear"))
                        .clicked();

                    if open_clicked {
                        if let Some(roi) = self.focused_roi().cloned() {
                            action = Some(ProjectSpaceAction::Open(roi));
                        }
                    }
                    if open_mosaic_clicked {
                        action = Some(ProjectSpaceAction::OpenMosaic);
                    }
                    if select_all_clicked {
                        let ids = visible_indices
                            .iter()
                            .filter_map(|&index| self.config.rois.get(index))
                            .map(|roi| roi.id.clone())
                            .collect::<Vec<_>>();
                        if let Err(error) = self.select_roi_ids(&ids, "replace") {
                            self.status = error;
                        }
                    }
                    if remove_clicked {
                        let ids = self
                            .selected_rois()
                            .into_iter()
                            .map(|roi| roi.id)
                            .collect::<Vec<_>>();
                        for id in ids {
                            if let Err(error) = self.remove_roi_by_id(&id) {
                                self.status = error;
                                break;
                            }
                        }
                    }
                    if clear_clicked {
                        let ids = self
                            .config
                            .rois
                            .iter()
                            .map(|roi| roi.id.clone())
                            .collect::<Vec<_>>();
                        for id in ids {
                            if let Err(error) = self.remove_roi_by_id(&id) {
                                self.status = error;
                                break;
                            }
                        }
                    }
                });

                ui.add_space(8.0);
                ui.separator();
                ui.heading("ROI Details");
                let focused_idx = self.focused.as_ref().and_then(|focused| {
                    self.config
                        .rois
                        .iter()
                        .position(|roi| roi.source_key().as_deref() == Some(focused.as_str()))
                });
                if let Some(idx) = focused_idx {
                    let original_id = self.config.rois[idx].id.clone();
                    let mut roi_draft = self.config.rois[idx].clone();
                    let mut changed = false;
                    let mut choose_seg = false;
                    let mut clear_seg = false;
                    let mut remove_key = None::<String>;
                    let mut rename_meta = None::<(String, String, String)>;
                    let mut add_meta = None::<(String, String)>;
                    let mut new_meta_key = self.new_meta_key.clone();
                    let mut new_meta_value = self.new_meta_value.clone();

                    {
                        let details_wide = ui.available_width() >= 560.0;
                        let roi = &mut roi_draft;

                        if details_wide {
                            ui.horizontal(|ui| {
                                ui.label("ID");
                                changed |= ui
                                    .add_sized(
                                        [ui.available_width(), 0.0],
                                        egui::TextEdit::singleline(&mut roi.id),
                                    )
                                    .changed();
                            });
                        } else {
                            ui.label("ID");
                            changed |= ui
                                .add_sized(
                                    [ui.available_width(), 0.0],
                                    egui::TextEdit::singleline(&mut roi.id),
                                )
                                .changed();
                        }

                        if details_wide {
                            ui.horizontal(|ui| {
                                ui.label("Display");
                                let display =
                                    roi.display_name.get_or_insert_with(|| roi.id.clone());
                                changed |= ui
                                    .add_sized(
                                        [ui.available_width(), 0.0],
                                        egui::TextEdit::singleline(display),
                                    )
                                    .changed();
                            });
                        } else {
                            ui.label("Display");
                            let display = roi.display_name.get_or_insert_with(|| roi.id.clone());
                            changed |= ui
                                .add_sized(
                                    [ui.available_width(), 0.0],
                                    egui::TextEdit::singleline(display),
                                )
                                .changed();
                        }

                        ui.label("Image");
                        ui.horizontal_wrapped(|ui| {
                            let mut path_text = roi.source_display();
                            ui.add_sized(
                                [ui.available_width(), 0.0],
                                egui::TextEdit::singleline(&mut path_text),
                            );
                        });

                        ui.label("Segmentation");
                        ui.horizontal_wrapped(|ui| {
                            let mut seg_text = roi
                                .segpath
                                .as_ref()
                                .map(|p| p.to_string_lossy().to_string())
                                .unwrap_or_default();
                            let text_width = if ui.available_width() >= 520.0 {
                                (ui.available_width() - 150.0).max(220.0)
                            } else {
                                ui.available_width().max(180.0)
                            };
                            ui.add_sized(
                                [text_width, 0.0],
                                egui::TextEdit::singleline(&mut seg_text),
                            );
                            if ui.button("Choose...").clicked() {
                                choose_seg = true;
                            }
                            if ui
                                .add_enabled(roi.segpath.is_some(), egui::Button::new("Clear"))
                                .clicked()
                            {
                                clear_seg = true;
                            }
                        });

                        ui.add_space(6.0);
                        ui.label("Metadata");
                        let mut meta_keys = roi.meta.keys().cloned().collect::<Vec<_>>();
                        meta_keys.sort();
                        for key in meta_keys {
                            let existing = roi.meta.get(&key).cloned().unwrap_or_default();
                            let mut label = key.clone();
                            let mut value = existing.clone();
                            let mut remove_clicked = false;
                            ui.horizontal_wrapped(|ui| {
                                let field_width = if ui.available_width() >= 560.0 {
                                    ((ui.available_width() - 80.0) * 0.5).max(120.0)
                                } else {
                                    ui.available_width().max(160.0)
                                };
                                changed |= ui
                                    .add_sized(
                                        [field_width, 0.0],
                                        egui::TextEdit::singleline(&mut label),
                                    )
                                    .changed();
                                changed |= ui
                                    .add_sized(
                                        [field_width, 0.0],
                                        egui::TextEdit::singleline(&mut value),
                                    )
                                    .changed();
                                if ui.small_button("Remove").clicked() {
                                    remove_clicked = true;
                                }
                            });
                            if remove_clicked {
                                remove_key = Some(key.clone());
                            } else {
                                if value != existing {
                                    roi.meta.insert(key.clone(), value.clone());
                                }
                                if label != key
                                    && !label.trim().is_empty()
                                    && !roi.meta.contains_key(&label)
                                {
                                    rename_meta =
                                        Some((key.clone(), label.trim().to_string(), value));
                                }
                            }
                        }
                        ui.horizontal_wrapped(|ui| {
                            let field_width = if ui.available_width() >= 560.0 {
                                ((ui.available_width() - 130.0) * 0.5).max(120.0)
                            } else {
                                ui.available_width().max(160.0)
                            };
                            ui.add_sized(
                                [field_width, 0.0],
                                egui::TextEdit::singleline(&mut new_meta_key).hint_text("column"),
                            );
                            ui.add_sized(
                                [field_width, 0.0],
                                egui::TextEdit::singleline(&mut new_meta_value).hint_text("value"),
                            );
                            let can_add = !new_meta_key.trim().is_empty();
                            if ui
                                .add_enabled(can_add, egui::Button::new("Add metadata"))
                                .clicked()
                            {
                                add_meta = Some((
                                    new_meta_key.trim().to_string(),
                                    new_meta_value.trim().to_string(),
                                ));
                            }
                        });
                    }

                    self.new_meta_key = new_meta_key;
                    self.new_meta_value = new_meta_value;

                    if clear_seg {
                        roi_draft.segpath = None;
                        changed = true;
                    }
                    if choose_seg {
                        if let Some(path) = FileDialog::new()
                            .add_filter("GeoJSON", &["geojson", "json"])
                            .add_filter("GeoParquet", &["geoparquet", "parquet"])
                            .set_title("Choose ROI Segmentation")
                            .pick_file()
                        {
                            roi_draft.segpath = Some(path);
                            changed = true;
                        }
                    }
                    if let Some(key) = remove_key {
                        roi_draft.meta.remove(&key);
                        changed = true;
                    }
                    if let Some((old_key, new_key, value)) = rename_meta {
                        roi_draft.meta.remove(&old_key);
                        roi_draft.meta.insert(new_key, value);
                        changed = true;
                    }
                    if let Some((key, value)) = add_meta {
                        roi_draft.meta.insert(key, value);
                        self.new_meta_key.clear();
                        self.new_meta_value.clear();
                        changed = true;
                    }

                    if changed {
                        if let Err(error) = self.update_roi_record(&original_id, roi_draft) {
                            self.status = error;
                        }
                    }
                } else {
                    ui.label("Select an ROI to edit its segmentation path and metadata.");
                }

                ui.add_space(8.0);
                ui.separator();
                ui.label("Save / Load");

                ui.horizontal_wrapped(|ui| {
                    if ui.button("New Project").clicked() {
                        if self.submit_control_intent("project.create", serde_json::json!({})) {
                            self.status = "Creating new project...".to_string();
                        } else {
                            self.config = ProjectConfig::default();
                            self.focused = None;
                            self.selected.clear();
                            self.roi_browse.clear();
                            self.config_generation = self.config_generation.wrapping_add(1);
                            self.status = "New project.".to_string();
                        }
                    }
                    if ui.button("Save").clicked() {
                        let path = self
                            .saved_project_path()
                            .or_else(|| self.choose_project_save_path("Save Project"));
                        if let Some(path) = path {
                            action = Some(ProjectSpaceAction::SaveProject(path));
                        }
                    }
                    if ui.button("Save As...").clicked() {
                        if let Some(path) = self.choose_project_save_path("Save Project As") {
                            action = Some(ProjectSpaceAction::SaveProject(path));
                        }
                    }
                    if ui.button("Load Project...").clicked() {
                        if let Some(path) = FileDialog::new()
                            .add_filter("Project JSON", &["json"])
                            .set_title("Load Project")
                            .pick_file()
                        {
                            self.load_path = path.to_string_lossy().to_string();
                            action = Some(ProjectSpaceAction::OpenProject(path));
                        }
                    }
                });

                self.ui_recent_projects(ui, &mut action);

                if !self.status.is_empty() {
                    ui.add_space(6.0);
                    ui.label(self.status.clone());
                }
            });

        action
    }

    pub(super) fn ui_recent_projects(
        &mut self,
        ui: &mut egui::Ui,
        action: &mut Option<ProjectSpaceAction>,
    ) {
        ui.add_space(8.0);
        ui.label("Recent Projects");
        if self.recent_projects.is_empty() {
            ui.small("No recent projects yet.");
            return;
        }

        for recent in self.recent_projects.iter().take(10) {
            let path = recent.path.clone();
            let label = recent.display_name();
            ui.horizontal_wrapped(|ui| {
                let remove_width = 28.0;
                let button_width = if ui.available_width() > 260.0 {
                    (ui.available_width() - remove_width).max(120.0)
                } else {
                    ui.available_width().max(120.0)
                };
                let response = ui
                    .add_sized([button_width, 22.0], egui::Button::new(label).wrap())
                    .on_hover_text(path.to_string_lossy());
                if response.clicked() {
                    *action = Some(ProjectSpaceAction::OpenProject(path.clone()));
                }
                if ui
                    .small_button("x")
                    .on_hover_text("Remove from recent")
                    .clicked()
                {
                    *action = Some(ProjectSpaceAction::ForgetRecentProject(path));
                }
            });
        }

        if ui.small_button("Clear recent projects").clicked() {
            *action = Some(ProjectSpaceAction::ClearRecentProjects);
        }
    }
}
