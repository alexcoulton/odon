use super::*;

// Object-layer analysis UI and helpers.
//
// This file owns the viewer-side analysis surface for object layers: histogram/scatter UI,
// threshold-set editing, batch measurement requests, spatial-table joins, and the caches that
// support interactive brushing/selection. The key theme is that expensive analysis products are
// lazily derived from the currently visible/filter-selected object set and invalidated whenever
// that set changes.

impl ObjectsLayer {
    pub fn sync_analysis_follow_active_channel(
        &mut self,
        channels: &[ChannelInfo],
        selected_channel: usize,
    ) {
        if !self.has_data() || !self.analysis_follow_active_channel {
            return;
        }
        let numeric_columns = self.available_numeric_object_property_keys();
        if numeric_columns.is_empty() {
            return;
        }
        self.save_live_threshold_rules();
        self.sync_histogram_to_active_channel(channels, selected_channel, &numeric_columns);
    }

    pub fn ui_analysis(
        &mut self,
        ui: &mut egui::Ui,
        _dataset: &OmeZarrDataset,
        _store: Arc<dyn ReadableStorageTraits>,
        _channels: &[ChannelInfo],
        selected_channel: usize,
        suspend_live_selection_sync: bool,
        _local_to_world_offset: egui::Vec2,
        _spatial_root: Option<&Path>,
        _spatial_tables: &[SpatialDataElement],
    ) {
        // Analysis always operates on the currently active object snapshot: either the full loaded
        // set or the materialized filtered subset from `ensure_filter_cache`.
        self.ensure_filter_cache();

        ui.heading("Analysis");
        if !self.has_data() {
            ui.label("Load segmentation objects to run per-cell analysis.");
            return;
        }
        let all_count = self.object_count();
        let filtered_count = self.filtered_count();
        ui.label(format!("Cells available: {all_count}"));
        if self.has_active_filter() {
            ui.label(format!("Filtered cells: {filtered_count}"));
        }
        if self.analysis_warm_total_columns > 0
            && self.analysis_warm_completed_columns < self.analysis_warm_total_columns
        {
            ui.label(format!(
                "Warming analysis columns: {} / {}",
                self.analysis_warm_completed_columns, self.analysis_warm_total_columns
            ));
        }
        self.ui_selection_elements_editor(ui);
        self.analysis_source = AnalysisSource::ObjectProperties;
        ui.label("Source: Object properties");
        self.ui_threshold_set_editor(ui, _channels, selected_channel);
        self.ui_object_properties_analysis(
            ui,
            _channels,
            selected_channel,
            suspend_live_selection_sync,
        );
    }

    fn ui_object_properties_analysis(
        &mut self,
        ui: &mut egui::Ui,
        channels: &[ChannelInfo],
        selected_channel: usize,
        suspend_live_selection_sync: bool,
    ) {
        let numeric_columns = self.available_numeric_object_property_keys();
        if numeric_columns.is_empty() {
            ui.label("No numeric object properties are available for plotting.");
            return;
        }

        self.analysis_hist_channel = self
            .analysis_hist_channel
            .min(numeric_columns.len().saturating_sub(1));
        self.analysis_scatter_x_channel = self
            .analysis_scatter_x_channel
            .min(numeric_columns.len().saturating_sub(1));
        self.analysis_scatter_y_channel = self
            .analysis_scatter_y_channel
            .min(numeric_columns.len().saturating_sub(1));

        if self.has_active_filter() {
            ui.label("Plots use the currently filtered object set.");
        } else {
            ui.label("Plots use all loaded objects.");
        }

        ui.horizontal(|ui| {
            ui.label("Plot");
            ui.selectable_value(
                &mut self.analysis_plot_mode,
                AnalysisPlotMode::Histogram,
                "Histogram",
            );
            ui.selectable_value(
                &mut self.analysis_plot_mode,
                AnalysisPlotMode::Scatter,
                "Scatter",
            );
        });

        match self.analysis_plot_mode {
            AnalysisPlotMode::Histogram => {
                self.sync_histogram_to_active_channel(channels, selected_channel, &numeric_columns);
                let prev_hist_channel = self.analysis_hist_channel;
                Self::analysis_name_picker(
                    ui,
                    "Column",
                    "seg_objects_analysis_object_hist_column",
                    &numeric_columns,
                    &mut self.analysis_hist_channel,
                );
                let Some(column_name) = numeric_columns.get(self.analysis_hist_channel) else {
                    return;
                };
                if self.analysis_hist_channel != prev_hist_channel {
                    self.analysis_hist_drag_rule = None;
                    self.sync_histogram_editor_to_column(column_name);
                }
                ui.horizontal(|ui| {
                    ui.label("Transform");
                    let prev_transform = self.analysis_hist_value_transform;
                    ui.selectable_value(
                        &mut self.analysis_hist_value_transform,
                        HistogramValueTransform::None,
                        "Raw",
                    );
                    ui.selectable_value(
                        &mut self.analysis_hist_value_transform,
                        HistogramValueTransform::Arcsinh,
                        "arcsinh",
                    );
                    if self.analysis_hist_value_transform != prev_transform {
                        self.analysis_hist_drag_rule = None;
                        self.sync_current_column_rule_transform(column_name);
                        self.clear_histogram_snapped_level_for_column(column_name);
                    }
                });
                let Some(hist) = self.object_property_histogram(column_name) else {
                    ui.label("No finite values available for the selected column.");
                    return;
                };
                self.ensure_default_object_property_threshold(
                    column_name,
                    hist.median,
                    self.active_channel_name(channels, selected_channel),
                );
                self.ui_object_property_threshold_rules(ui, &numeric_columns, column_name);
                self.ui_object_property_histogram_levels(ui, column_name);
                let _drag_finished = self.ui_object_property_histogram(ui, column_name, &hist);
                let selected_ids = self.object_property_threshold_selected_indices();
                let has_matches = !selected_ids.is_empty();
                let mut ordered_passing = None;
                let mut current_focus_pos = None;
                if let Some(idx) = self.analysis_hist_focus_object_index {
                    let ordered = self.object_property_threshold_ordered_indices(column_name);
                    current_focus_pos = ordered.iter().position(|candidate| *candidate == idx);
                    if current_focus_pos.is_some() {
                        ordered_passing = Some(ordered);
                    } else {
                        self.analysis_hist_focus_object_index = None;
                    }
                }
                ui.horizontal(|ui| {
                    ui.label(format!("Matching cells: {}", selected_ids.len()));
                    if ui.button("Clear thresholds").clicked() {
                        self.analysis_property_thresholds.clear();
                        self.sync_active_threshold_element_from_live_rules();
                        self.analysis_hist_focus_object_index = None;
                        self.clear_histogram_snapped_level_for_column(column_name);
                        self.sync_live_analysis_selection(&[]);
                    }
                });
                if suspend_live_selection_sync && !self.analysis_property_thresholds.is_empty() {
                    ui.label("Live analysis selection is paused while Rect/Lasso is active.");
                }
                ui.horizontal(|ui| {
                    let low_idx = has_matches.then_some(());
                    let high_idx = has_matches.then_some(());
                    let prev_idx = current_focus_pos
                        .and_then(|pos| pos.checked_sub(1))
                        .and_then(|pos| ordered_passing.as_ref()?.get(pos).copied());
                    let next_idx = current_focus_pos
                        .and_then(|pos| ordered_passing.as_ref()?.get(pos + 1).copied());

                    if ui
                        .add_enabled(low_idx.is_some(), egui::Button::new("Zoom low"))
                        .clicked()
                        && let Some(idx) = self
                            .object_property_threshold_ordered_indices(column_name)
                            .first()
                            .copied()
                    {
                        self.request_zoom_to_object(idx);
                    }
                    if ui
                        .add_enabled(prev_idx.is_some(), egui::Button::new("Previous"))
                        .clicked()
                        && let Some(idx) = prev_idx
                    {
                        self.request_zoom_to_object(idx);
                    }
                    if ui
                        .add_enabled(next_idx.is_some(), egui::Button::new("Next"))
                        .clicked()
                        && let Some(idx) = next_idx
                    {
                        self.request_zoom_to_object(idx);
                    }
                    if ui
                        .add_enabled(high_idx.is_some(), egui::Button::new("Zoom high"))
                        .clicked()
                        && let Some(idx) = self
                            .object_property_threshold_ordered_indices(column_name)
                            .last()
                            .copied()
                    {
                        self.request_zoom_to_object(idx);
                    }
                });
                if self.analysis_property_thresholds.is_empty() {
                    if !suspend_live_selection_sync && self.consume_live_analysis_selection_dirty()
                    {
                        self.sync_live_analysis_selection(&[]);
                    }
                    ui.label("Add one or more thresholds to drive live selection.");
                } else if suspend_live_selection_sync {
                    ui.label("Switch back to Pan to resume live threshold selection.");
                } else if self.consume_live_analysis_selection_dirty() {
                    self.sync_live_analysis_selection(&selected_ids);
                }
            }
            AnalysisPlotMode::Scatter => {
                Self::analysis_name_picker(
                    ui,
                    "X",
                    "seg_objects_analysis_object_scatter_x",
                    &numeric_columns,
                    &mut self.analysis_scatter_x_channel,
                );
                Self::analysis_name_picker(
                    ui,
                    "Y",
                    "seg_objects_analysis_object_scatter_y",
                    &numeric_columns,
                    &mut self.analysis_scatter_y_channel,
                );
                let Some(x_column) = numeric_columns.get(self.analysis_scatter_x_channel) else {
                    return;
                };
                let Some(y_column) = numeric_columns.get(self.analysis_scatter_y_channel) else {
                    return;
                };
                let points = self.object_property_scatter_points(x_column, y_column);
                let prev_brush = self.analysis_scatter_brush;
                let prev_x_column = numeric_columns
                    .get(self.analysis_scatter_x_channel)
                    .cloned()
                    .unwrap_or_default();
                let prev_y_column = numeric_columns
                    .get(self.analysis_scatter_y_channel)
                    .cloned()
                    .unwrap_or_default();
                let current_selection = self.selected_object_indices.clone();
                let (selected_ids, drag_finished) = Self::ui_object_property_scatter(
                    ui,
                    prev_x_column.as_str(),
                    prev_y_column.as_str(),
                    points.as_slice(),
                    &current_selection,
                    &mut self.analysis_scatter_view_key,
                    &mut self.analysis_scatter_view_rect,
                    &mut self.analysis_scatter_brush,
                    &mut self.analysis_scatter_drag_anchor,
                );
                if self.analysis_scatter_brush != prev_brush {
                    self.mark_live_analysis_selection_dirty();
                }
                ui.horizontal(|ui| {
                    ui.label(format!("Matching cells: {}", selected_ids.len()));
                    if ui.button("Clear box").clicked() {
                        self.analysis_scatter_brush = None;
                        self.mark_live_analysis_selection_dirty();
                    }
                });
                if self.analysis_scatter_brush.is_some() {
                    if suspend_live_selection_sync {
                        ui.label("Live analysis selection is paused while Rect/Lasso is active.");
                    } else if (drag_finished || self.analysis_scatter_drag_anchor.is_none())
                        && self.consume_live_analysis_selection_dirty()
                    {
                        self.sync_live_analysis_selection(&selected_ids);
                    } else {
                        ui.label("Release to update selection.");
                    }
                } else {
                    if !suspend_live_selection_sync && self.consume_live_analysis_selection_dirty()
                    {
                        self.sync_live_analysis_selection(&[]);
                    }
                    ui.label("Drag a box to replace the current selection.");
                }
            }
        }
    }

    fn ui_image_measurement_analysis(
        &mut self,
        ui: &mut egui::Ui,
        dataset: &OmeZarrDataset,
        store: Arc<dyn ReadableStorageTraits>,
        channels: &[ChannelInfo],
        local_to_world_offset: egui::Vec2,
        all_count: usize,
        filtered_count: usize,
    ) {
        if dataset.is_root_label_mask() {
            ui.label("Image measurement analysis is unavailable for root label-mask datasets.");
            return;
        }
        if channels.is_empty() || dataset.channels.is_empty() {
            ui.label("No image channels available.");
            return;
        }

        self.analysis_level = self
            .analysis_level
            .min(dataset.levels.len().saturating_sub(1));
        ui.horizontal(|ui| {
            ui.label("Measurement level");
            let selected_text = dataset
                .levels
                .get(self.analysis_level)
                .map(|lvl| format!("L{} ({:.2}x)", lvl.index, lvl.downsample))
                .unwrap_or_else(|| "L0".to_string());
            egui::ComboBox::from_id_salt("seg_objects_analysis_level")
                .selected_text(selected_text)
                .show_ui(ui, |ui| {
                    for lvl in &dataset.levels {
                        ui.selectable_value(
                            &mut self.analysis_level,
                            lvl.index,
                            format!("L{} ({:.2}x)", lvl.index, lvl.downsample),
                        );
                    }
                });
        });
        ui.horizontal(|ui| {
            ui.label("Backend");
            ui.selectable_value(
                &mut self.analysis_backend,
                AnalysisBackend::Polygons,
                "Polygons",
            );
            ui.selectable_value(
                &mut self.analysis_backend,
                AnalysisBackend::ExternalLabelMask,
                "Label mask",
            );
        });
        if self.analysis_backend == AnalysisBackend::ExternalLabelMask {
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(!self.is_analyzing(), egui::Button::new("Load label mask"))
                    .clicked()
                {
                    if let Some(path) = FileDialog::new()
                        .add_filter("Label mask", &["tif", "tiff", "npy"])
                        .set_title("Load label mask")
                        .pick_file()
                    {
                        match load_external_label_mask(&path, dataset) {
                            Ok(mask) => {
                                let dims = format!("{}x{}", mask.width, mask.height);
                                self.analysis_external_label_mask = Some(Arc::new(mask));
                                self.analysis_status =
                                    format!("Loaded label mask {} ({dims}).", path.display());
                            }
                            Err(err) => {
                                self.analysis_status = format!("Load label mask failed: {err}");
                            }
                        }
                    }
                }
                if ui
                    .add_enabled(
                        !self.is_analyzing() && self.analysis_external_label_mask.is_some(),
                        egui::Button::new("Clear label mask"),
                    )
                    .clicked()
                {
                    self.analysis_external_label_mask = None;
                }
            });
            if let Some(mask) = self.analysis_external_label_mask.as_ref() {
                ui.label(format!(
                    "Label mask: {} ({}x{})",
                    mask.path.display(),
                    mask.width,
                    mask.height
                ));
            } else {
                ui.label(
                    "Load a `.tif`, `.tiff`, or `.npy` label mask with the same dimensions as the image.",
                );
            }
            if self.analysis_level > 0 {
                ui.label("Label-mask analysis at coarse image levels uses nearest-neighbour label lookup from the full-resolution mask.");
            }
        }

        let backend_ready = match self.analysis_backend {
            AnalysisBackend::Polygons => true,
            AnalysisBackend::ExternalLabelMask => self.analysis_external_label_mask.is_some(),
        };

        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    !self.is_analyzing() && filtered_count > 0 && backend_ready,
                    egui::Button::new("Measure filtered cells"),
                )
                .clicked()
            {
                self.request_batch_analysis(
                    dataset,
                    store.clone(),
                    channels,
                    local_to_world_offset,
                    true,
                );
            }
            if ui
                .add_enabled(
                    !self.is_analyzing() && all_count > 0 && backend_ready,
                    egui::Button::new("Measure all cells"),
                )
                .clicked()
            {
                self.request_batch_analysis(
                    dataset,
                    store.clone(),
                    channels,
                    local_to_world_offset,
                    false,
                );
            }
            if ui
                .add_enabled(self.is_analyzing(), egui::Button::new("Cancel"))
                .clicked()
            {
                if let Some(cancel) = self.analysis_cancel.as_ref() {
                    cancel.store(true, Ordering::Relaxed);
                    self.analysis_status = "Cancelling analysis...".to_string();
                }
            }
            if ui
                .add_enabled(
                    self.analysis_results.is_some(),
                    egui::Button::new("Clear results"),
                )
                .clicked()
            {
                self.clear_analysis();
            }
        });

        if self.analysis_progress_total > 0 {
            let frac = (self.analysis_progress_completed as f32
                / self.analysis_progress_total as f32)
                .clamp(0.0, 1.0);
            ui.add(
                egui::ProgressBar::new(frac)
                    .animate(self.is_analyzing())
                    .show_percentage()
                    .text(format!(
                        "{} / {}",
                        self.analysis_progress_completed, self.analysis_progress_total
                    )),
            );
        }
        if !self.analysis_status.is_empty() {
            ui.label(self.analysis_status.clone());
        }

        let Some(AnalysisResults::Measurements(table)) = self.analysis_results.as_ref() else {
            return;
        };
        if table.channels.is_empty() || table.rows.is_empty() {
            ui.label("No completed analysis rows available.");
            return;
        }

        ui.separator();
        ui.label(format!(
            "Results: {} measured cell(s) from {} at level {} ({:.2}x)",
            table.rows.len(),
            table.scope_label,
            table.level_index,
            table.level_downsample
        ));
        ui.label(format!("Failed cells: {}", table.failed_count));

        ui.horizontal(|ui| {
            ui.label("Plot");
            ui.selectable_value(
                &mut self.analysis_plot_mode,
                AnalysisPlotMode::Histogram,
                "Histogram",
            );
            ui.selectable_value(
                &mut self.analysis_plot_mode,
                AnalysisPlotMode::Scatter,
                "Scatter",
            );
        });
        ui.horizontal(|ui| {
            ui.label("Metric");
            ui.selectable_value(&mut self.analysis_metric, AnalysisMetric::Mean, "Mean");
            ui.selectable_value(&mut self.analysis_metric, AnalysisMetric::Max, "Max");
            ui.selectable_value(&mut self.analysis_metric, AnalysisMetric::Integrated, "Sum");
        });

        match self.analysis_plot_mode {
            AnalysisPlotMode::Histogram => {
                self.analysis_hist_channel = self
                    .analysis_hist_channel
                    .min(table.channels.len().saturating_sub(1));
                Self::analysis_channel_picker(
                    ui,
                    "Channel",
                    "seg_objects_analysis_hist_channel",
                    &table.channels,
                    &mut self.analysis_hist_channel,
                );
                let selected_ids = Self::ui_analysis_histogram(
                    ui,
                    table,
                    self.analysis_metric,
                    self.analysis_hist_channel,
                    &mut self.analysis_hist_brush,
                    &mut self.analysis_hist_drag_anchor,
                );
                self.ui_analysis_selection_actions(ui, &selected_ids, "Brush cells to select.");
            }
            AnalysisPlotMode::Scatter => {
                self.analysis_scatter_x_channel = self
                    .analysis_scatter_x_channel
                    .min(table.channels.len().saturating_sub(1));
                self.analysis_scatter_y_channel = self
                    .analysis_scatter_y_channel
                    .min(table.channels.len().saturating_sub(1));
                Self::analysis_channel_picker(
                    ui,
                    "X",
                    "seg_objects_analysis_scatter_x",
                    &table.channels,
                    &mut self.analysis_scatter_x_channel,
                );
                Self::analysis_channel_picker(
                    ui,
                    "Y",
                    "seg_objects_analysis_scatter_y",
                    &table.channels,
                    &mut self.analysis_scatter_y_channel,
                );
                let x_label = table
                    .channels
                    .get(self.analysis_scatter_x_channel)
                    .map(|channel| channel.channel_name.as_str())
                    .unwrap_or("X");
                let y_label = table
                    .channels
                    .get(self.analysis_scatter_y_channel)
                    .map(|channel| channel.channel_name.as_str())
                    .unwrap_or("Y");
                let current_selection = self.selected_object_indices.clone();
                let selected_ids = Self::ui_analysis_scatter(
                    ui,
                    table,
                    self.analysis_metric,
                    self.analysis_scatter_x_channel,
                    self.analysis_scatter_y_channel,
                    x_label,
                    y_label,
                    &current_selection,
                    &mut self.analysis_scatter_view_key,
                    &mut self.analysis_scatter_view_rect,
                    &mut self.analysis_scatter_brush,
                    &mut self.analysis_scatter_drag_anchor,
                );
                self.ui_analysis_selection_actions(
                    ui,
                    &selected_ids,
                    "Drag a box to select cells.",
                );
            }
        }
    }

    fn ui_analysis_histogram(
        ui: &mut egui::Ui,
        table: &AnalysisBatchTable,
        metric: AnalysisMetric,
        channel_pos: usize,
        brush: &mut Option<(f32, f32)>,
        drag_anchor: &mut Option<f32>,
    ) -> Vec<usize> {
        let values = analysis_metric_values(table, metric, channel_pos);
        let Some((_min_value, _max_value)) = finite_min_max_f32(&values) else {
            ui.label("No finite values available for the selected channel.");
            return Vec::new();
        };
        let hist = compute_histogram_f32(&values, 128);
        let (rect, response) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), 160.0),
            egui::Sense::click_and_drag(),
        );
        let painter = ui.painter();
        painter.rect_filled(rect, 4.0, ui.visuals().extreme_bg_color);
        painter.rect_stroke(
            rect,
            4.0,
            ui.visuals().widgets.noninteractive.bg_stroke,
            egui::StrokeKind::Middle,
        );

        let w = rect.width().max(1.0);
        let h = rect.height().max(1.0);
        let bin_w = w / hist.bins.len().max(1) as f32;
        for (i, &count) in hist.bins.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let frac = count as f32 / hist.max_count as f32;
            let x0 = rect.left() + i as f32 * bin_w;
            let x1 = x0 + bin_w;
            let y1 = rect.bottom() - 4.0;
            let y0 = y1 - frac * (h - 8.0);
            painter.rect_filled(
                egui::Rect::from_min_max(egui::pos2(x0, y0), egui::pos2(x1, y1)),
                0.0,
                ui.visuals()
                    .widgets
                    .noninteractive
                    .fg_stroke
                    .color
                    .linear_multiply(0.85),
            );
        }

        if response.drag_started() {
            if let Some(pos) = response.interact_pointer_pos() {
                *drag_anchor = Some(screen_x_to_value(pos.x, rect, hist.min, hist.max));
            }
        }
        if response.dragged() {
            if let (Some(anchor), Some(pos)) = (*drag_anchor, response.interact_pointer_pos()) {
                let current = screen_x_to_value(pos.x, rect, hist.min, hist.max);
                *brush = Some(order_pair(anchor, current));
            }
        }
        if response.drag_stopped() {
            *drag_anchor = None;
        }

        if let Some((lo, hi)) = *brush {
            let x0 = value_to_screen_x(lo, rect, hist.min, hist.max);
            let x1 = value_to_screen_x(hi, rect, hist.min, hist.max);
            let brush_rect = egui::Rect::from_min_max(
                egui::pos2(x0.min(x1), rect.top()),
                egui::pos2(x0.max(x1), rect.bottom()),
            );
            painter.rect_filled(
                brush_rect,
                0.0,
                egui::Color32::from_rgba_unmultiplied(120, 190, 255, 48),
            );
            painter.rect_stroke(
                brush_rect,
                0.0,
                egui::Stroke::new(1.5, egui::Color32::from_rgb(120, 190, 255)),
                egui::StrokeKind::Middle,
            );
            ui.label(format!("Range: {:.2} to {:.2}", lo, hi));
        }
        ui.label(format!(
            "Min {:.2}   Median {:.2}   Max {:.2}",
            hist.min, hist.median, hist.max
        ));

        analysis_hist_selected_indices(table, metric, channel_pos, *brush)
    }

    fn ui_spatial_table_analysis(
        &mut self,
        ui: &mut egui::Ui,
        spatial_root: Option<&Path>,
        spatial_tables: &[SpatialDataElement],
        all_count: usize,
        filtered_count: usize,
    ) {
        let Some(spatial_root) = spatial_root else {
            ui.label("No SpatialData container is attached to this object layer.");
            return;
        };
        if spatial_tables.is_empty() {
            ui.label("No SpatialData tables were discovered for this dataset.");
            return;
        }

        self.analysis_spatial_table_index = self
            .analysis_spatial_table_index
            .min(spatial_tables.len().saturating_sub(1));
        let selected_table = spatial_tables
            .get(self.analysis_spatial_table_index)
            .cloned()
            .unwrap_or_else(|| spatial_tables[0].clone());
        ui.horizontal(|ui| {
            ui.label("Table");
            egui::ComboBox::from_id_salt("seg_objects_analysis_spatial_table")
                .selected_text(selected_table.name.clone())
                .show_ui(ui, |ui| {
                    for (i, table) in spatial_tables.iter().enumerate() {
                        ui.selectable_value(
                            &mut self.analysis_spatial_table_index,
                            i,
                            table.name.clone(),
                        );
                    }
                });
        });

        let meta = match self.ensure_spatial_table_meta(spatial_root, &selected_table) {
            Ok(meta) => meta.clone(),
            Err(err) => {
                ui.label(format!("Table metadata unavailable: {err}"));
                return;
            }
        };
        if meta.numeric_columns.is_empty() {
            ui.label("The selected table has no numeric columns available for plotting.");
            return;
        }
        let available_join_keys = self.available_object_join_keys();
        let supported_join_keys = meta
            .join_keys
            .iter()
            .filter(|key| {
                available_join_keys
                    .iter()
                    .any(|candidate| candidate == *key)
            })
            .cloned()
            .collect::<Vec<_>>();
        if supported_join_keys.is_empty() {
            ui.label(
                "No compatible join key exists between this table and the active object layer.",
            );
            return;
        }
        if !supported_join_keys
            .iter()
            .any(|key| key == &self.analysis_spatial_join_key)
        {
            self.analysis_spatial_join_key = supported_join_keys[0].clone();
        }

        ui.label(format!("Table rows: {}", meta.row_count));
        ui.horizontal(|ui| {
            ui.label("Join key");
            egui::ComboBox::from_id_salt("seg_objects_analysis_spatial_join_key")
                .selected_text(self.analysis_spatial_join_key.clone())
                .show_ui(ui, |ui| {
                    for key in &supported_join_keys {
                        ui.selectable_value(&mut self.analysis_spatial_join_key, key.clone(), key);
                    }
                });
        });
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    !self.is_analyzing() && filtered_count > 0,
                    egui::Button::new("Load filtered rows"),
                )
                .clicked()
            {
                self.request_spatial_table_analysis(
                    spatial_root,
                    selected_table.clone(),
                    meta.clone(),
                    true,
                );
            }
            if ui
                .add_enabled(
                    !self.is_analyzing() && all_count > 0,
                    egui::Button::new("Load all rows"),
                )
                .clicked()
            {
                self.request_spatial_table_analysis(
                    spatial_root,
                    selected_table.clone(),
                    meta.clone(),
                    false,
                );
            }
            if ui
                .add_enabled(self.is_analyzing(), egui::Button::new("Cancel"))
                .clicked()
            {
                if let Some(cancel) = self.analysis_cancel.as_ref() {
                    cancel.store(true, Ordering::Relaxed);
                    self.analysis_status = "Cancelling analysis...".to_string();
                }
            }
            if ui
                .add_enabled(
                    self.analysis_results.is_some(),
                    egui::Button::new("Clear results"),
                )
                .clicked()
            {
                self.clear_analysis();
            }
        });

        if self.analysis_progress_total > 0 {
            let frac = (self.analysis_progress_completed as f32
                / self.analysis_progress_total as f32)
                .clamp(0.0, 1.0);
            ui.add(
                egui::ProgressBar::new(frac)
                    .animate(self.is_analyzing())
                    .show_percentage()
                    .text(format!(
                        "{} / {}",
                        self.analysis_progress_completed, self.analysis_progress_total
                    )),
            );
        }
        if !self.analysis_status.is_empty() {
            ui.label(self.analysis_status.clone());
        }

        let needed_columns = match self.analysis_plot_mode {
            AnalysisPlotMode::Histogram => vec![self.analysis_hist_channel],
            AnalysisPlotMode::Scatter => {
                vec![
                    self.analysis_scatter_x_channel,
                    self.analysis_scatter_y_channel,
                ]
            }
        };
        let load_result = {
            let Some(AnalysisResults::SpatialTable(table)) = self.analysis_results.as_mut() else {
                return;
            };
            if table.numeric_columns.is_empty() || table.rows.is_empty() {
                ui.label("No joined SpatialData table rows available.");
                return;
            }
            load_spatial_table_columns_for_indices(spatial_root, table, &needed_columns)
        };
        if let Err(err) = load_result {
            self.analysis_status = format!("SpatialData column load failed: {err}");
            ui.label(self.analysis_status.clone());
            return;
        }

        let Some(AnalysisResults::SpatialTable(table)) = self.analysis_results.as_ref() else {
            return;
        };
        if table.numeric_columns.is_empty() || table.rows.is_empty() {
            ui.label("No joined SpatialData table rows available.");
            return;
        }

        ui.separator();
        ui.label(format!(
            "Results: {} joined row(s) from '{}' via {}",
            table.rows.len(),
            table.table_name,
            table.join_key
        ));
        ui.label(format!(
            "Matched objects: {}   Unmatched objects: {}",
            table.matched_object_count, table.unmatched_object_count
        ));

        ui.horizontal(|ui| {
            ui.label("Plot");
            ui.selectable_value(
                &mut self.analysis_plot_mode,
                AnalysisPlotMode::Histogram,
                "Histogram",
            );
            ui.selectable_value(
                &mut self.analysis_plot_mode,
                AnalysisPlotMode::Scatter,
                "Scatter",
            );
        });

        match self.analysis_plot_mode {
            AnalysisPlotMode::Histogram => {
                self.analysis_hist_channel = self
                    .analysis_hist_channel
                    .min(table.numeric_columns.len().saturating_sub(1));
                Self::analysis_name_picker(
                    ui,
                    "Column",
                    "seg_objects_analysis_spatial_hist_column",
                    &table.numeric_columns,
                    &mut self.analysis_hist_channel,
                );
                let selected_ids = Self::ui_spatial_table_histogram(
                    ui,
                    table,
                    self.analysis_hist_channel,
                    &mut self.analysis_hist_brush,
                    &mut self.analysis_hist_drag_anchor,
                );
                self.ui_analysis_selection_actions(ui, &selected_ids, "Brush cells to select.");
            }
            AnalysisPlotMode::Scatter => {
                self.analysis_scatter_x_channel = self
                    .analysis_scatter_x_channel
                    .min(table.numeric_columns.len().saturating_sub(1));
                self.analysis_scatter_y_channel = self
                    .analysis_scatter_y_channel
                    .min(table.numeric_columns.len().saturating_sub(1));
                Self::analysis_name_picker(
                    ui,
                    "X",
                    "seg_objects_analysis_spatial_scatter_x",
                    &table.numeric_columns,
                    &mut self.analysis_scatter_x_channel,
                );
                Self::analysis_name_picker(
                    ui,
                    "Y",
                    "seg_objects_analysis_spatial_scatter_y",
                    &table.numeric_columns,
                    &mut self.analysis_scatter_y_channel,
                );
                let x_label = table
                    .numeric_columns
                    .get(self.analysis_scatter_x_channel)
                    .map(String::as_str)
                    .unwrap_or("X");
                let y_label = table
                    .numeric_columns
                    .get(self.analysis_scatter_y_channel)
                    .map(String::as_str)
                    .unwrap_or("Y");
                let current_selection = self.selected_object_indices.clone();
                let selected_ids = Self::ui_spatial_table_scatter(
                    ui,
                    table,
                    self.analysis_scatter_x_channel,
                    self.analysis_scatter_y_channel,
                    x_label,
                    y_label,
                    &current_selection,
                    &mut self.analysis_scatter_view_key,
                    &mut self.analysis_scatter_view_rect,
                    &mut self.analysis_scatter_brush,
                    &mut self.analysis_scatter_drag_anchor,
                );
                self.ui_analysis_selection_actions(
                    ui,
                    &selected_ids,
                    "Drag a box to select cells.",
                );
            }
        }
    }

    fn scatter_view_rect_for_key(
        scatter_view_key: &mut Option<String>,
        scatter_view_rect: &mut Option<egui::Rect>,
        brush: &mut Option<egui::Rect>,
        drag_anchor: &mut Option<egui::Pos2>,
        key: String,
        data_rect: egui::Rect,
    ) -> egui::Rect {
        let data_rect = normalize_scatter_view_rect(data_rect, data_rect);
        if scatter_view_key.as_deref() != Some(key.as_str()) {
            *scatter_view_key = Some(key);
            *scatter_view_rect = Some(data_rect);
            *brush = None;
            *drag_anchor = None;
            return data_rect;
        }

        let view_rect =
            normalize_scatter_view_rect(scatter_view_rect.unwrap_or(data_rect), data_rect);
        *scatter_view_rect = Some(view_rect);
        view_rect
    }

    fn ui_scatter_axis_limits(
        ui: &mut egui::Ui,
        scatter_view_key: &mut Option<String>,
        scatter_view_rect: &mut Option<egui::Rect>,
        brush: &mut Option<egui::Rect>,
        drag_anchor: &mut Option<egui::Pos2>,
        key: String,
        x_label: &str,
        y_label: &str,
        data_rect: egui::Rect,
    ) -> egui::Rect {
        let mut view_rect = Self::scatter_view_rect_for_key(
            scatter_view_key,
            scatter_view_rect,
            brush,
            drag_anchor,
            key,
            data_rect,
        );
        let x_speed = scatter_axis_drag_speed(data_rect.min.x, data_rect.max.x);
        let y_speed = scatter_axis_drag_speed(data_rect.min.y, data_rect.max.y);

        ui.horizontal(|ui| {
            ui.label("Axis limits");
            ui.label("X");
            ui.add(
                egui::DragValue::new(&mut view_rect.min.x)
                    .speed(x_speed)
                    .prefix("min "),
            );
            ui.add(
                egui::DragValue::new(&mut view_rect.max.x)
                    .speed(x_speed)
                    .prefix("max "),
            );
            ui.label("Y");
            ui.add(
                egui::DragValue::new(&mut view_rect.min.y)
                    .speed(y_speed)
                    .prefix("min "),
            );
            ui.add(
                egui::DragValue::new(&mut view_rect.max.y)
                    .speed(y_speed)
                    .prefix("max "),
            );
            if ui.button("Reset").clicked() {
                view_rect = data_rect;
            }
        });

        view_rect = normalize_scatter_view_rect(view_rect, data_rect);
        *scatter_view_rect = Some(view_rect);
        ui.label(format!(
            "Visible {x_label} [{:.2}, {:.2}]   {y_label} [{:.2}, {:.2}]",
            view_rect.min.x, view_rect.max.x, view_rect.min.y, view_rect.max.y
        ));
        view_rect
    }

    fn ui_analysis_scatter(
        ui: &mut egui::Ui,
        table: &AnalysisBatchTable,
        metric: AnalysisMetric,
        x_channel: usize,
        y_channel: usize,
        x_label: &str,
        y_label: &str,
        current_selection: &HashSet<usize>,
        scatter_view_key: &mut Option<String>,
        scatter_view_rect: &mut Option<egui::Rect>,
        brush: &mut Option<egui::Rect>,
        drag_anchor: &mut Option<egui::Pos2>,
    ) -> Vec<usize> {
        let x_values = analysis_metric_values(table, metric, x_channel);
        let y_values = analysis_metric_values(table, metric, y_channel);
        let Some((x_min, x_max)) = finite_min_max_f32(&x_values) else {
            ui.label("No finite X values available.");
            return Vec::new();
        };
        let Some((y_min, y_max)) = finite_min_max_f32(&y_values) else {
            ui.label("No finite Y values available.");
            return Vec::new();
        };
        let data_rect =
            egui::Rect::from_min_max(egui::pos2(x_min, y_min), egui::pos2(x_max, y_max));
        let view_rect = Self::ui_scatter_axis_limits(
            ui,
            scatter_view_key,
            scatter_view_rect,
            brush,
            drag_anchor,
            format!("analysis:measurements:{metric:?}:{x_label}:{y_label}"),
            x_label,
            y_label,
            data_rect,
        );

        let (rect, response) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), 220.0),
            egui::Sense::click_and_drag(),
        );
        let painter = ui.painter();
        painter.rect_filled(rect, 4.0, ui.visuals().extreme_bg_color);
        painter.rect_stroke(
            rect,
            4.0,
            ui.visuals().widgets.noninteractive.bg_stroke,
            egui::StrokeKind::Middle,
        );

        let max_points = 20_000usize;
        let step = (table.rows.len() / max_points).max(1);
        for (i, row) in table.rows.iter().enumerate().step_by(step) {
            let x = analysis_row_metric_value(row, metric, x_channel);
            let y = analysis_row_metric_value(row, metric, y_channel);
            if !(x.is_finite() && y.is_finite()) {
                continue;
            }
            if x < view_rect.min.x
                || x > view_rect.max.x
                || y < view_rect.min.y
                || y > view_rect.max.y
            {
                continue;
            }
            let pos = egui::pos2(
                value_to_screen_x(x, rect, view_rect.min.x, view_rect.max.x),
                value_to_screen_y(y, rect, view_rect.min.y, view_rect.max.y),
            );
            let color = if current_selection.contains(&table.rows[i].object_index) {
                egui::Color32::from_rgb(255, 245, 140)
            } else {
                egui::Color32::from_rgba_unmultiplied(180, 220, 255, 140)
            };
            painter.circle_filled(pos, 1.5, color);
        }

        if response.drag_started() {
            *drag_anchor = response.interact_pointer_pos();
        }
        if response.dragged() {
            if let (Some(anchor), Some(pos)) = (*drag_anchor, response.interact_pointer_pos()) {
                let screen_rect = egui::Rect::from_two_pos(anchor, pos).intersect(rect);
                *brush = screen_rect_to_value_rect(
                    screen_rect,
                    rect,
                    view_rect.min.x,
                    view_rect.max.x,
                    view_rect.min.y,
                    view_rect.max.y,
                );
            }
        }
        if response.drag_stopped() {
            *drag_anchor = None;
        }

        if let Some(value_rect) = *brush {
            let screen_rect = value_rect_to_screen_rect(
                value_rect,
                rect,
                view_rect.min.x,
                view_rect.max.x,
                view_rect.min.y,
                view_rect.max.y,
            );
            painter.rect_filled(
                screen_rect,
                0.0,
                egui::Color32::from_rgba_unmultiplied(120, 190, 255, 40),
            );
            painter.rect_stroke(
                screen_rect,
                0.0,
                egui::Stroke::new(1.5, egui::Color32::from_rgb(120, 190, 255)),
                egui::StrokeKind::Middle,
            );
            ui.label(format!(
                "X {:.2}..{:.2}   Y {:.2}..{:.2}",
                value_rect.min.x, value_rect.max.x, value_rect.min.y, value_rect.max.y
            ));
        }
        ui.label(format!(
            "Showing up to {} points. Data X [{:.2}, {:.2}]   Data Y [{:.2}, {:.2}]",
            max_points, x_min, x_max, y_min, y_max
        ));

        analysis_scatter_selected_indices(table, metric, x_channel, y_channel, *brush)
    }

    fn ui_object_property_histogram(
        &mut self,
        ui: &mut egui::Ui,
        column_name: &str,
        hist: &SimpleHistogram,
    ) -> bool {
        const THRESHOLD_GRAB_RADIUS_PX: f32 = 12.0;

        let (rect, response) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), 160.0),
            egui::Sense::click_and_drag(),
        );
        let painter = ui.painter();
        painter.rect_filled(rect, 4.0, ui.visuals().extreme_bg_color);
        painter.rect_stroke(
            rect,
            4.0,
            ui.visuals().widgets.noninteractive.bg_stroke,
            egui::StrokeKind::Middle,
        );

        let w = rect.width().max(1.0);
        let h = rect.height().max(1.0);
        let bin_w = w / hist.bins.len().max(1) as f32;
        for (i, &count) in hist.bins.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let frac = count as f32 / hist.max_count as f32;
            let x0 = rect.left() + i as f32 * bin_w;
            let x1 = x0 + bin_w;
            let y1 = rect.bottom() - 4.0;
            let y0 = y1 - frac * (h - 8.0);
            painter.rect_filled(
                egui::Rect::from_min_max(egui::pos2(x0, y0), egui::pos2(x1, y1)),
                0.0,
                ui.visuals()
                    .widgets
                    .noninteractive
                    .fg_stroke
                    .color
                    .linear_multiply(0.85),
            );
        }

        let threshold_indices = self
            .analysis_property_thresholds
            .iter()
            .enumerate()
            .filter_map(|(idx, rule)| (rule.column_key == column_name).then_some(idx))
            .collect::<Vec<_>>();
        let value_transform = self.analysis_hist_value_transform;
        if let Some(pointer_pos) = response.hover_pos() {
            let near_line = threshold_indices.iter().any(|idx| {
                let x = value_to_screen_x(
                    apply_histogram_value_transform(
                        self.analysis_property_thresholds[*idx].value,
                        value_transform,
                    ),
                    rect,
                    hist.min,
                    hist.max,
                );
                (pointer_pos.x - x).abs() <= THRESHOLD_GRAB_RADIUS_PX
            });
            if near_line {
                ui.output_mut(|o| o.cursor_icon = egui::CursorIcon::ResizeHorizontal);
            }
        }
        if response.is_pointer_button_down_on()
            && ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Primary))
        {
            self.analysis_hist_drag_rule =
                response.interact_pointer_pos().and_then(|pointer_pos| {
                    threshold_indices
                        .iter()
                        .copied()
                        .min_by(|a, b| {
                            let ax = value_to_screen_x(
                                apply_histogram_value_transform(
                                    self.analysis_property_thresholds[*a].value,
                                    value_transform,
                                ),
                                rect,
                                hist.min,
                                hist.max,
                            );
                            let bx = value_to_screen_x(
                                apply_histogram_value_transform(
                                    self.analysis_property_thresholds[*b].value,
                                    value_transform,
                                ),
                                rect,
                                hist.min,
                                hist.max,
                            );
                            (pointer_pos.x - ax)
                                .abs()
                                .partial_cmp(&(pointer_pos.x - bx).abs())
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .filter(|idx| {
                            let x = value_to_screen_x(
                                apply_histogram_value_transform(
                                    self.analysis_property_thresholds[*idx].value,
                                    value_transform,
                                ),
                                rect,
                                hist.min,
                                hist.max,
                            );
                            (pointer_pos.x - x).abs() <= THRESHOLD_GRAB_RADIUS_PX
                        })
                });
        }
        if response.is_pointer_button_down_on()
            && ui.input(|i| i.pointer.button_down(egui::PointerButton::Primary))
        {
            if let (Some(idx), Some(pointer_pos)) = (
                self.analysis_hist_drag_rule,
                response.interact_pointer_pos(),
            ) {
                if let Some(rule) = self.analysis_property_thresholds.get_mut(idx) {
                    let transformed = screen_x_to_value(pointer_pos.x, rect, hist.min, hist.max);
                    rule.value = invert_histogram_value_transform(transformed, value_transform);
                    self.clear_histogram_snapped_level_for_column(column_name);
                    self.sync_active_threshold_element_from_live_rules();
                }
            }
        }
        let drag_finished = response.drag_stopped();
        if drag_finished {
            self.analysis_hist_drag_rule = None;
        }

        for rule in self
            .analysis_property_thresholds
            .iter()
            .filter(|rule| rule.column_key == column_name)
        {
            let x = value_to_screen_x(
                apply_histogram_value_transform(rule.value, value_transform),
                rect,
                hist.min,
                hist.max,
            );
            let color = match rule.op {
                AnalysisThresholdOp::GreaterEqual => egui::Color32::from_rgb(255, 210, 80),
                AnalysisThresholdOp::LessEqual => egui::Color32::from_rgb(120, 190, 255),
            };
            painter.line_segment(
                [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                egui::Stroke::new(2.0, color),
            );
        }

        let transform_label = match value_transform {
            HistogramValueTransform::None => "Raw",
            HistogramValueTransform::Arcsinh => "arcsinh",
        };
        ui.label(format!(
            "{} ({transform_label}): Min {:.2}   Median {:.2}   Max {:.2}",
            column_name, hist.min, hist.median, hist.max
        ));
        drag_finished
    }

    fn ui_object_property_scatter(
        ui: &mut egui::Ui,
        x_label: &str,
        y_label: &str,
        points: &[(usize, f32, f32)],
        current_selection: &HashSet<usize>,
        scatter_view_key: &mut Option<String>,
        scatter_view_rect: &mut Option<egui::Rect>,
        brush: &mut Option<egui::Rect>,
        drag_anchor: &mut Option<egui::Pos2>,
    ) -> (Vec<usize>, bool) {
        let mut x_values = Vec::new();
        let mut y_values = Vec::new();
        for (_, x, y) in points {
            x_values.push(*x);
            y_values.push(*y);
        }

        let Some((x_min, x_max)) = finite_min_max_f32(&x_values) else {
            ui.label("No finite X values available.");
            return (Vec::new(), false);
        };
        let Some((y_min, y_max)) = finite_min_max_f32(&y_values) else {
            ui.label("No finite Y values available.");
            return (Vec::new(), false);
        };
        let data_rect =
            egui::Rect::from_min_max(egui::pos2(x_min, y_min), egui::pos2(x_max, y_max));
        let view_rect = Self::ui_scatter_axis_limits(
            ui,
            scatter_view_key,
            scatter_view_rect,
            brush,
            drag_anchor,
            format!("analysis:object-properties:{x_label}:{y_label}"),
            x_label,
            y_label,
            data_rect,
        );

        let (rect, response) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), 220.0),
            egui::Sense::click_and_drag(),
        );
        let painter = ui.painter();
        painter.rect_filled(rect, 4.0, ui.visuals().extreme_bg_color);
        painter.rect_stroke(
            rect,
            4.0,
            ui.visuals().widgets.noninteractive.bg_stroke,
            egui::StrokeKind::Middle,
        );

        let max_points = 20_000usize;
        let step = (points.len() / max_points).max(1);
        for (object_index, x, y) in points.iter().step_by(step) {
            if *x < view_rect.min.x
                || *x > view_rect.max.x
                || *y < view_rect.min.y
                || *y > view_rect.max.y
            {
                continue;
            }
            let pos = egui::pos2(
                value_to_screen_x(*x, rect, view_rect.min.x, view_rect.max.x),
                value_to_screen_y(*y, rect, view_rect.min.y, view_rect.max.y),
            );
            let color = if current_selection.contains(object_index) {
                egui::Color32::from_rgb(255, 245, 140)
            } else {
                egui::Color32::from_rgba_unmultiplied(180, 220, 255, 140)
            };
            painter.circle_filled(pos, 1.5, color);
        }

        if response.drag_started() {
            *drag_anchor = response.interact_pointer_pos();
        }
        if response.dragged() {
            if let (Some(anchor), Some(pos)) = (*drag_anchor, response.interact_pointer_pos()) {
                let screen_rect = egui::Rect::from_two_pos(anchor, pos).intersect(rect);
                *brush = screen_rect_to_value_rect(
                    screen_rect,
                    rect,
                    view_rect.min.x,
                    view_rect.max.x,
                    view_rect.min.y,
                    view_rect.max.y,
                );
            }
        }
        let drag_finished = response.drag_stopped();
        if drag_finished {
            *drag_anchor = None;
        }

        if let Some(value_rect) = *brush {
            let screen_rect = value_rect_to_screen_rect(
                value_rect,
                rect,
                view_rect.min.x,
                view_rect.max.x,
                view_rect.min.y,
                view_rect.max.y,
            );
            painter.rect_filled(
                screen_rect,
                0.0,
                egui::Color32::from_rgba_unmultiplied(120, 190, 255, 40),
            );
            painter.rect_stroke(
                screen_rect,
                0.0,
                egui::Stroke::new(1.5, egui::Color32::from_rgb(120, 190, 255)),
                egui::StrokeKind::Middle,
            );
            ui.label(format!(
                "X {:.2}..{:.2}   Y {:.2}..{:.2}",
                value_rect.min.x, value_rect.max.x, value_rect.min.y, value_rect.max.y
            ));
        }
        ui.label(format!(
            "Showing up to {} points. Data X [{:.2}, {:.2}]   Data Y [{:.2}, {:.2}]",
            max_points, x_min, x_max, y_min, y_max
        ));

        let Some(brush_rect) = *brush else {
            return (Vec::new(), drag_finished);
        };
        (
            points
                .iter()
                .filter_map(|(object_index, x, y)| {
                    brush_rect
                        .contains(egui::pos2(*x, *y))
                        .then_some(*object_index)
                })
                .collect(),
            drag_finished,
        )
    }

    fn ui_spatial_table_histogram(
        ui: &mut egui::Ui,
        table: &SpatialTableAnalysis,
        column_pos: usize,
        brush: &mut Option<(f32, f32)>,
        drag_anchor: &mut Option<f32>,
    ) -> Vec<usize> {
        let values = spatial_table_column_values(table, column_pos);
        let Some((_min_value, _max_value)) = finite_min_max_f32(&values) else {
            ui.label("No finite values available for the selected column.");
            return Vec::new();
        };
        let hist = compute_histogram_f32(&values, 128);
        let (rect, response) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), 160.0),
            egui::Sense::click_and_drag(),
        );
        let painter = ui.painter();
        painter.rect_filled(rect, 4.0, ui.visuals().extreme_bg_color);
        painter.rect_stroke(
            rect,
            4.0,
            ui.visuals().widgets.noninteractive.bg_stroke,
            egui::StrokeKind::Middle,
        );

        let w = rect.width().max(1.0);
        let h = rect.height().max(1.0);
        let bin_w = w / hist.bins.len().max(1) as f32;
        for (i, &count) in hist.bins.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let frac = count as f32 / hist.max_count as f32;
            let x0 = rect.left() + i as f32 * bin_w;
            let x1 = x0 + bin_w;
            let y1 = rect.bottom() - 4.0;
            let y0 = y1 - frac * (h - 8.0);
            painter.rect_filled(
                egui::Rect::from_min_max(egui::pos2(x0, y0), egui::pos2(x1, y1)),
                0.0,
                ui.visuals()
                    .widgets
                    .noninteractive
                    .fg_stroke
                    .color
                    .linear_multiply(0.85),
            );
        }

        if response.drag_started() {
            if let Some(pos) = response.interact_pointer_pos() {
                *drag_anchor = Some(screen_x_to_value(pos.x, rect, hist.min, hist.max));
            }
        }
        if response.dragged() {
            if let (Some(anchor), Some(pos)) = (*drag_anchor, response.interact_pointer_pos()) {
                let current = screen_x_to_value(pos.x, rect, hist.min, hist.max);
                *brush = Some(order_pair(anchor, current));
            }
        }
        if response.drag_stopped() {
            *drag_anchor = None;
        }

        if let Some((lo, hi)) = *brush {
            let x0 = value_to_screen_x(lo, rect, hist.min, hist.max);
            let x1 = value_to_screen_x(hi, rect, hist.min, hist.max);
            let brush_rect = egui::Rect::from_min_max(
                egui::pos2(x0.min(x1), rect.top()),
                egui::pos2(x0.max(x1), rect.bottom()),
            );
            painter.rect_filled(
                brush_rect,
                0.0,
                egui::Color32::from_rgba_unmultiplied(120, 190, 255, 48),
            );
            painter.rect_stroke(
                brush_rect,
                0.0,
                egui::Stroke::new(1.5, egui::Color32::from_rgb(120, 190, 255)),
                egui::StrokeKind::Middle,
            );
            ui.label(format!("Range: {:.2} to {:.2}", lo, hi));
        }
        ui.label(format!(
            "Min {:.2}   Median {:.2}   Max {:.2}",
            hist.min, hist.median, hist.max
        ));

        spatial_hist_selected_indices(table, column_pos, *brush)
    }

    fn ui_spatial_table_scatter(
        ui: &mut egui::Ui,
        table: &SpatialTableAnalysis,
        x_column: usize,
        y_column: usize,
        x_label: &str,
        y_label: &str,
        current_selection: &HashSet<usize>,
        scatter_view_key: &mut Option<String>,
        scatter_view_rect: &mut Option<egui::Rect>,
        brush: &mut Option<egui::Rect>,
        drag_anchor: &mut Option<egui::Pos2>,
    ) -> Vec<usize> {
        let x_values = spatial_table_column_values(table, x_column);
        let y_values = spatial_table_column_values(table, y_column);
        let Some((x_min, x_max)) = finite_min_max_f32(&x_values) else {
            ui.label("No finite X values available.");
            return Vec::new();
        };
        let Some((y_min, y_max)) = finite_min_max_f32(&y_values) else {
            ui.label("No finite Y values available.");
            return Vec::new();
        };
        let data_rect =
            egui::Rect::from_min_max(egui::pos2(x_min, y_min), egui::pos2(x_max, y_max));
        let view_rect = Self::ui_scatter_axis_limits(
            ui,
            scatter_view_key,
            scatter_view_rect,
            brush,
            drag_anchor,
            format!("analysis:spatial-table:{x_label}:{y_label}"),
            x_label,
            y_label,
            data_rect,
        );

        let (rect, response) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), 220.0),
            egui::Sense::click_and_drag(),
        );
        let painter = ui.painter();
        painter.rect_filled(rect, 4.0, ui.visuals().extreme_bg_color);
        painter.rect_stroke(
            rect,
            4.0,
            ui.visuals().widgets.noninteractive.bg_stroke,
            egui::StrokeKind::Middle,
        );

        let max_points = 20_000usize;
        let step = (table.rows.len() / max_points).max(1);
        let x_loaded = spatial_table_column_values(table, x_column);
        let y_loaded = spatial_table_column_values(table, y_column);
        for (i, row) in table.rows.iter().enumerate().step_by(step) {
            let x = x_loaded.get(i).copied().unwrap_or(f32::NAN);
            let y = y_loaded.get(i).copied().unwrap_or(f32::NAN);
            if !(x.is_finite() && y.is_finite()) {
                continue;
            }
            if x < view_rect.min.x
                || x > view_rect.max.x
                || y < view_rect.min.y
                || y > view_rect.max.y
            {
                continue;
            }
            let pos = egui::pos2(
                value_to_screen_x(x, rect, view_rect.min.x, view_rect.max.x),
                value_to_screen_y(y, rect, view_rect.min.y, view_rect.max.y),
            );
            let color = if current_selection.contains(&table.rows[i].object_index) {
                egui::Color32::from_rgb(255, 245, 140)
            } else {
                egui::Color32::from_rgba_unmultiplied(180, 220, 255, 140)
            };
            painter.circle_filled(pos, 1.5, color);
        }

        if response.drag_started() {
            *drag_anchor = response.interact_pointer_pos();
        }
        if response.dragged() {
            if let (Some(anchor), Some(pos)) = (*drag_anchor, response.interact_pointer_pos()) {
                let screen_rect = egui::Rect::from_two_pos(anchor, pos).intersect(rect);
                *brush = screen_rect_to_value_rect(
                    screen_rect,
                    rect,
                    view_rect.min.x,
                    view_rect.max.x,
                    view_rect.min.y,
                    view_rect.max.y,
                );
            }
        }
        if response.drag_stopped() {
            *drag_anchor = None;
        }

        if let Some(value_rect) = *brush {
            let screen_rect = value_rect_to_screen_rect(
                value_rect,
                rect,
                view_rect.min.x,
                view_rect.max.x,
                view_rect.min.y,
                view_rect.max.y,
            );
            painter.rect_filled(
                screen_rect,
                0.0,
                egui::Color32::from_rgba_unmultiplied(120, 190, 255, 40),
            );
            painter.rect_stroke(
                screen_rect,
                0.0,
                egui::Stroke::new(1.5, egui::Color32::from_rgb(120, 190, 255)),
                egui::StrokeKind::Middle,
            );
            ui.label(format!(
                "X {:.2}..{:.2}   Y {:.2}..{:.2}",
                value_rect.min.x, value_rect.max.x, value_rect.min.y, value_rect.max.y
            ));
        }
        ui.label(format!(
            "Showing up to {} points. Data X [{:.2}, {:.2}]   Data Y [{:.2}, {:.2}]",
            max_points, x_min, x_max, y_min, y_max
        ));

        spatial_scatter_selected_indices(table, x_column, y_column, *brush)
    }

    fn request_batch_analysis(
        &mut self,
        dataset: &OmeZarrDataset,
        store: Arc<dyn ReadableStorageTraits>,
        channels: &[ChannelInfo],
        local_to_world_offset: egui::Vec2,
        filtered_only: bool,
    ) {
        let Some(objects) = self.objects.as_ref().cloned() else {
            self.analysis_status = "No objects loaded.".to_string();
            return;
        };
        let target_indices = if filtered_only {
            if let Some(filtered) = self.filtered_indices.as_ref() {
                let mut out = filtered.iter().copied().collect::<Vec<_>>();
                out.sort_unstable();
                out
            } else {
                (0..objects.len()).collect::<Vec<_>>()
            }
        } else {
            (0..objects.len()).collect::<Vec<_>>()
        };
        if target_indices.is_empty() {
            self.analysis_status = "No cells available for analysis.".to_string();
            return;
        }

        // Only one long-running batch analysis is considered active at a time. Starting a new one
        // cancels the previous worker and bumps the request id so late messages can be ignored.
        if let Some(cancel) = self.analysis_cancel.take() {
            cancel.store(true, Ordering::Relaxed);
        }
        self.analysis_request_id = self.analysis_request_id.wrapping_add(1).max(1);
        let request_id = self.analysis_request_id;
        let total = target_indices.len();
        let scope_label = if filtered_only {
            format!("filtered cells ({total})")
        } else {
            format!("all cells ({total})")
        };
        let (tx, rx) = crossbeam_channel::unbounded::<AnalysisBatchEvent>();
        let cancel = Arc::new(AtomicBool::new(false));

        self.analysis_rx = Some(rx);
        self.analysis_cancel = Some(cancel.clone());
        self.analysis_progress_completed = 0;
        self.analysis_progress_total = 0;
        let level = self
            .analysis_level
            .min(dataset.levels.len().saturating_sub(1));
        self.analysis_status = match self.analysis_backend {
            AnalysisBackend::Polygons => {
                format!("Preparing chunk analysis for {scope_label} at level {level}...")
            }
            AnalysisBackend::ExternalLabelMask => {
                format!("Preparing label-mask analysis for {scope_label} at level {level}...")
            }
        };
        self.analysis_results = None;
        self.analysis_property_thresholds.clear();
        self.analysis_hist_brush = None;
        self.analysis_scatter_brush = None;
        self.analysis_hist_drag_anchor = None;
        self.analysis_scatter_drag_anchor = None;

        let dataset = dataset.clone();
        let channels = channels.to_vec();
        let analysis_backend = self.analysis_backend;
        let external_label_mask = self.analysis_external_label_mask.clone();
        std::thread::Builder::new()
            .name("seg-objects-analysis".to_string())
            .spawn(move || {
                let result = match analysis_backend {
                    AnalysisBackend::Polygons => measure_objects_batch_in_thread(
                        &dataset,
                        store,
                        &channels,
                        objects,
                        &target_indices,
                        local_to_world_offset,
                        level,
                        request_id,
                        &tx,
                        &cancel,
                        scope_label,
                    ),
                    AnalysisBackend::ExternalLabelMask => match external_label_mask {
                        Some(label_mask) => measure_objects_with_label_mask_in_thread(
                            &dataset,
                            store,
                            &channels,
                            objects,
                            &target_indices,
                            label_mask,
                            level,
                            request_id,
                            &tx,
                            &cancel,
                            scope_label,
                        ),
                        None => Err(anyhow!("no external label mask loaded")),
                    },
                };
                let cancelled = cancel.load(Ordering::Relaxed);
                let msg = match result {
                    Ok(table) => AnalysisBatchEvent::Finished {
                        request_id,
                        result: (!cancelled).then_some(AnalysisResults::Measurements(table)),
                        cancelled,
                        error: None,
                    },
                    Err(err) => AnalysisBatchEvent::Finished {
                        request_id,
                        result: None,
                        cancelled,
                        error: Some(err.to_string()),
                    },
                };
                let _ = tx.send(msg);
            })
            .ok();
    }

    fn request_spatial_table_analysis(
        &mut self,
        spatial_root: &Path,
        table_element: SpatialDataElement,
        meta: SpatialDataTableMeta,
        filtered_only: bool,
    ) {
        let object_lookup =
            self.object_lookup_for_join_key(self.analysis_spatial_join_key.as_str(), filtered_only);
        if object_lookup.is_empty() {
            self.analysis_status =
                "No objects with a matching join key are available for this table.".to_string();
            return;
        }

        // Spatial-table analysis is a separate async path from image measurement analysis, but it
        // uses the same request-id/cancellation protocol so the UI can treat them uniformly.
        if let Some(cancel) = self.analysis_cancel.take() {
            cancel.store(true, Ordering::Relaxed);
        }
        self.analysis_request_id = self.analysis_request_id.wrapping_add(1).max(1);
        let request_id = self.analysis_request_id;
        let (tx, rx) = crossbeam_channel::unbounded::<AnalysisBatchEvent>();
        let cancel = Arc::new(AtomicBool::new(false));
        let root = spatial_root.to_path_buf();
        let join_key = self.analysis_spatial_join_key.clone();
        let scope_count = object_lookup.len();
        let scope_label = if filtered_only {
            format!("filtered cells ({scope_count})")
        } else {
            format!("all cells ({scope_count})")
        };

        self.analysis_rx = Some(rx);
        self.analysis_cancel = Some(cancel.clone());
        self.analysis_progress_completed = 0;
        self.analysis_progress_total = 1;
        self.analysis_status = format!(
            "Loading SpatialData table '{}' for {scope_label} via {}...",
            table_element.name, join_key
        );
        self.analysis_results = None;
        self.analysis_hist_brush = None;
        self.analysis_scatter_brush = None;
        self.analysis_hist_drag_anchor = None;
        self.analysis_scatter_drag_anchor = None;

        std::thread::Builder::new()
            .name("seg-objects-spatial-table".to_string())
            .spawn(move || {
                let result = if cancel.load(Ordering::Relaxed) {
                    Ok(None)
                } else {
                    load_table_analysis(&root, &meta, &join_key, &object_lookup)
                        .map(|table| Some(AnalysisResults::SpatialTable(table)))
                };
                let msg = match result {
                    Ok(result) => AnalysisBatchEvent::Finished {
                        request_id,
                        result,
                        cancelled: cancel.load(Ordering::Relaxed),
                        error: None,
                    },
                    Err(err) => AnalysisBatchEvent::Finished {
                        request_id,
                        result: None,
                        cancelled: cancel.load(Ordering::Relaxed),
                        error: Some(err.to_string()),
                    },
                };
                let _ = tx.send(msg);
            })
            .ok();
    }

    pub(super) fn apply_selection_indices(&mut self, indices: &[usize], additive: bool) {
        if !additive {
            self.selected_object_indices.clear();
        }
        for idx in indices {
            self.selected_object_indices.insert(*idx);
        }
        self.selected_object_index = self.selected_object_indices.iter().next().copied();
        self.rebuild_selection_render_lods();
        self.clear_measurements();
        self.invalidate_table_cache();
    }

    pub(super) fn mark_live_analysis_selection_dirty(&mut self) {
        self.analysis_live_selection_generation = self
            .analysis_live_selection_generation
            .wrapping_add(1)
            .max(1);
    }

    fn consume_live_analysis_selection_dirty(&mut self) -> bool {
        if self.analysis_live_selection_applied_generation
            == self.analysis_live_selection_generation
        {
            return false;
        }
        self.analysis_live_selection_applied_generation = self.analysis_live_selection_generation;
        true
    }

    fn analysis_channel_picker(
        ui: &mut egui::Ui,
        label: &str,
        id_salt: &str,
        channels: &[AnalysisChannelSchema],
        selected: &mut usize,
    ) {
        ui.horizontal(|ui| {
            ui.label(label);
            let selected_label = channels
                .get(*selected)
                .map(|ch| ch.channel_name.clone())
                .unwrap_or_else(|| "-".to_string());
            egui::ComboBox::from_id_salt(id_salt)
                .selected_text(selected_label)
                .show_ui(ui, |ui| {
                    for (i, ch) in channels.iter().enumerate() {
                        ui.selectable_value(selected, i, &ch.channel_name);
                    }
                });
        });
    }

    fn analysis_name_picker(
        ui: &mut egui::Ui,
        label: &str,
        id_salt: &str,
        names: &[String],
        selected: &mut usize,
    ) {
        if names.is_empty() {
            return;
        }
        *selected = (*selected).min(names.len().saturating_sub(1));
        ui.horizontal(|ui| {
            ui.label(label);
            let selected_label = names
                .get(*selected)
                .cloned()
                .unwrap_or_else(|| names[0].clone());
            let popup_state_id = ui.make_persistent_id((id_salt, "popup_open"));
            let search_id = ui.make_persistent_id((id_salt, "search"));
            let search_preview = ui
                .data(|d| d.get_temp::<String>(search_id))
                .unwrap_or_default();
            let mut popup_open = ui
                .data(|d| d.get_temp::<bool>(popup_state_id))
                .unwrap_or(false);
            let was_popup_open = popup_open;
            let button_resp = ui.button(selected_label.clone());
            if button_resp.clicked() {
                popup_open = !popup_open;
            }
            let just_opened = popup_open && !was_popup_open;
            let popup_width = analysis_picker_popup_width(
                ui,
                button_resp.rect.width(),
                names,
                &search_preview,
                Some(selected_label.as_str()),
            );

            let popup_id = ui.make_persistent_id((id_salt, "popup"));
            egui::Popup::from_response(&button_resp)
                .id(popup_id)
                .open_bool(&mut popup_open)
                .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                .gap(2.0)
                .width(popup_width)
                .show(|ui| {
                    let mut request_close = false;
                    ui.set_min_width(popup_width);
                    let mut search = ui
                        .data(|d| d.get_temp::<String>(search_id))
                        .unwrap_or_default();
                    let response = ui.add(
                        egui::TextEdit::singleline(&mut search)
                            .hint_text("Search columns...")
                            .desired_width(ui.available_width()),
                    );
                    if just_opened {
                        response.request_focus();
                    }
                    if response.changed() {
                        ui.data_mut(|d| d.insert_temp(search_id, search.clone()));
                    }
                    ui.separator();

                    let mut matches = names
                        .iter()
                        .enumerate()
                        .filter_map(|(i, name)| {
                            fuzzy_name_score(&search, name).map(|score| (score, i, name))
                        })
                        .collect::<Vec<_>>();
                    matches.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

                    egui::ScrollArea::vertical()
                        .id_salt((id_salt, "scroll"))
                        .auto_shrink([false, false])
                        .max_height(320.0)
                        .show(ui, |ui| {
                            for (_, i, name) in matches.into_iter().take(100) {
                                if ui.selectable_label(*selected == i, name).clicked() {
                                    *selected = i;
                                    request_close = true;
                                }
                            }
                        });
                    if request_close {
                        ui.close();
                    }
                });
            ui.data_mut(|d| d.insert_temp(popup_state_id, popup_open));
        });
    }

    fn ui_threshold_set_editor(
        &mut self,
        ui: &mut egui::Ui,
        channels: &[ChannelInfo],
        selected_channel: usize,
    ) {
        ui.separator();
        ui.collapsing("Threshold Set", |ui| {
            ui.horizontal(|ui| {
                ui.label("Name");
                ui.add(
                    egui::TextEdit::singleline(&mut self.analysis_threshold_set_name)
                        .desired_width(220.0),
                );
                if ui.button("Save...").clicked() {
                    self.save_threshold_set_dialog();
                }
                if ui.button("Load...").clicked() {
                    self.load_threshold_set_dialog();
                }
            });
            ui.horizontal(|ui| {
                if ui.button("Export GeoParquet...").clicked()
                    && let Err(err) = self.export_objects_geoparquet_with_dialog()
                {
                    self.status = format!("Export GeoParquet failed: {err}");
                }
                if ui.button("Export CSV...").clicked()
                    && let Err(err) = self.export_objects_csv_with_dialog()
                {
                    self.status = format!("Export CSV failed: {err}");
                }
            });
            let prev_follow_active_channel = self.analysis_follow_active_channel;
            ui.horizontal(|ui| {
                ui.checkbox(
                    &mut self.analysis_follow_active_channel,
                    "Follow active channel",
                );
                if ui.button("Mapping settings...").clicked() {
                    self.analysis_channel_mapping_popup_open = true;
                }
            });
            if self.analysis_follow_active_channel != prev_follow_active_channel {
                self.save_live_threshold_rules();
                if self.analysis_follow_active_channel {
                    let active_channel = self.active_channel_name(channels, selected_channel);
                    self.load_live_threshold_rules(active_channel);
                } else if let Some(idx) = self
                    .analysis_threshold_selected_element
                    .filter(|idx| *idx < self.analysis_threshold_elements.len())
                    && let Some(element) = self.analysis_threshold_elements.get(idx).cloned()
                {
                    self.analysis_property_thresholds = element.rules;
                    self.analysis_live_threshold_channel_name = None;
                } else {
                    self.analysis_live_threshold_channel_name = None;
                }
            }
            if self.analysis_follow_active_channel
                && let Some(name) = self.active_channel_name(channels, selected_channel)
            {
                ui.label(format!("Active channel: {name}"));
            }
            self.ui_channel_mapping_popup(ui, channels, selected_channel);
            ui.horizontal(|ui| {
                let selected_idx = self
                    .analysis_threshold_selected_element
                    .filter(|idx| *idx < self.analysis_threshold_elements.len());
                if ui.button("New element").clicked() {
                    let next_idx = self.analysis_threshold_elements.len() + 1;
                    self.analysis_threshold_elements.push(ThresholdSetElement {
                        name: format!("Element {next_idx}"),
                        rules: Vec::new(),
                    });
                    self.load_threshold_element(
                        self.analysis_threshold_elements.len() - 1,
                        self.active_channel_name(channels, selected_channel),
                    );
                }
                if ui
                    .add_enabled(selected_idx.is_some(), egui::Button::new("Delete selected"))
                    .clicked()
                    && let Some(idx) = selected_idx
                {
                    self.analysis_threshold_elements.remove(idx);
                    if self.analysis_threshold_elements.is_empty() {
                        self.analysis_threshold_selected_element = None;
                        self.analysis_property_thresholds.clear();
                        self.analysis_live_threshold_channel_name = None;
                    } else {
                        self.load_threshold_element(
                            idx.min(self.analysis_threshold_elements.len() - 1),
                            self.active_channel_name(channels, selected_channel),
                        );
                    }
                }
            });
            if self.analysis_threshold_elements.is_empty() {
                ui.label("No elements yet. Threshold edits will populate the selected element.");
            } else {
                let mut clicked_idx = None;
                egui::ScrollArea::vertical()
                    .id_salt("seg_objects_threshold_elements")
                    .max_height(180.0)
                    .show(ui, |ui| {
                        for (idx, element) in
                            self.analysis_threshold_elements.iter_mut().enumerate()
                        {
                            ui.horizontal(|ui| {
                                let selected =
                                    self.analysis_threshold_selected_element == Some(idx);
                                if ui
                                    .selectable_label(
                                        selected,
                                        format!(
                                            "{} ({} rule{})",
                                            element.name,
                                            element.rules.len(),
                                            if element.rules.len() == 1 { "" } else { "s" }
                                        ),
                                    )
                                    .clicked()
                                {
                                    clicked_idx = Some(idx);
                                }
                                ui.add(
                                    egui::TextEdit::singleline(&mut element.name)
                                        .desired_width(140.0),
                                );
                            });
                        }
                    });
                if let Some(idx) = clicked_idx {
                    self.load_threshold_element(
                        idx,
                        self.active_channel_name(channels, selected_channel),
                    );
                }
            }
        });
    }

    fn sync_histogram_editor_to_column(&mut self, column_name: &str) {
        if let Some(rule) = self
            .analysis_property_thresholds
            .iter()
            .find(|rule| rule.column_key == column_name)
        {
            self.analysis_hist_value_transform = rule.value_transform;
        }
    }

    fn ui_channel_mapping_popup(
        &mut self,
        ui: &mut egui::Ui,
        channels: &[ChannelInfo],
        selected_channel: usize,
    ) {
        if !self.analysis_channel_mapping_popup_open {
            return;
        }

        let numeric_columns = self.available_numeric_object_property_keys();
        self.ensure_channel_mapping_suggestions_cache(channels, &numeric_columns);

        let filtered_channels = if self.analysis_channel_mapping_search.trim().is_empty() {
            (0..channels.len()).collect::<Vec<_>>()
        } else {
            channels
                .iter()
                .enumerate()
                .filter_map(|(idx, channel)| {
                    fuzzy_name_score(&self.analysis_channel_mapping_search, &channel.name)
                        .is_some()
                        .then_some(idx)
                })
                .collect::<Vec<_>>()
        };
        let mut open = self.analysis_channel_mapping_popup_open;
        egui::Window::new("Mapping settings")
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_width(760.0)
            .default_height(520.0)
            .show(ui.ctx(), |ui| {
                ui.label(
                    "Choose which numeric object-property column each image channel should drive in Analysis.",
                );
                ui.horizontal(|ui| {
                    if ui.button("Auto-suggest all").clicked() {
                        for channel in channels.iter() {
	                            let Some(suggestions) =
	                                self.analysis_channel_mapping_suggestions_cache.get(&channel.name)
	                            else {
	                                continue;
	                            };
	                            if suggestions.is_empty() {
	                                continue;
	                            }
                            let next = match self
                                .analysis_channel_mapping_overrides
                                .get(&channel.name)
                                .and_then(|current| {
                                    suggestions.iter().position(|candidate| candidate == current)
                                }) {
                                Some(idx) => suggestions[(idx + 1) % suggestions.len()].clone(),
                                None => suggestions[0].clone(),
                            };
                            if !next.is_empty() {
                                self.analysis_channel_mapping_overrides
                                    .insert(channel.name.clone(), next);
                            }
                        }
                    }
                    if ui.button("Clear overrides").clicked() {
                        self.analysis_channel_mapping_overrides.clear();
                    }
                    ui.label("Search");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.analysis_channel_mapping_search)
                            .hint_text("Filter channels...")
                            .desired_width(220.0),
                    );
                });
                ui.separator();
                let row_height = ui.text_style_height(&egui::TextStyle::Body).max(18.0) + 10.0;
                egui::ScrollArea::vertical()
                    .id_salt("seg_objects_channel_mapping_scroll")
                    .show_rows(ui, row_height, filtered_channels.len(), |ui, row_range| {
                        for row in row_range {
                            let idx = filtered_channels[row];
                            let channel = &channels[idx];

                            let suggestions = self
                                .analysis_channel_mapping_suggestions_cache
                                .get(&channel.name)
                                .map(|v| v.as_slice())
                                .unwrap_or(&[]);
                            let auto = suggestions.first().cloned();
                            let current = if let Some(column) =
                                self.analysis_channel_mapping_overrides.get(&channel.name)
                                && numeric_columns.iter().any(|candidate| candidate == column)
                            {
                                Some(column.clone())
                            } else {
                                auto.clone()
                            };

                            let mut selected = current.clone();
                            ui.horizontal(|ui| {
                                let prefix = if idx == selected_channel { "> " } else { "" };
                                ui.label(format!("{prefix}{}", channel.name));
                                Self::analysis_optional_value_name_picker(
                                    ui,
                                    "",
                                    ("seg_objects_channel_mapping", &channel.name),
                                    &numeric_columns,
                                    &mut selected,
                                    "Choose column...",
                                );
                                let status = if self
                                    .analysis_channel_mapping_overrides
                                    .contains_key(&channel.name)
                                {
                                    "Manual"
                                } else if auto.is_some() {
                                    "Auto"
                                } else {
                                    "No match"
                                };
                                ui.label(status);
                                if ui
                                    .add_enabled(
                                        self.analysis_channel_mapping_overrides
                                            .contains_key(&channel.name),
                                        egui::Button::new("Use auto"),
                                    )
                                    .clicked()
                                {
                                    self.analysis_channel_mapping_overrides.remove(&channel.name);
                                }
                            });
                            if selected != current {
                                if let Some(selected) = selected {
                                    self.analysis_channel_mapping_overrides
                                        .insert(channel.name.clone(), selected);
                                }
                            }
                        }
                    });
            });
        self.analysis_channel_mapping_popup_open = open;
    }

    fn active_channel_name<'a>(
        &self,
        channels: &'a [ChannelInfo],
        selected_channel: usize,
    ) -> Option<&'a str> {
        channels
            .get(selected_channel)
            .map(|channel| channel.name.as_str())
    }

    fn sync_histogram_to_active_channel(
        &mut self,
        channels: &[ChannelInfo],
        selected_channel: usize,
        numeric_columns: &[String],
    ) {
        if !self.analysis_follow_active_channel || numeric_columns.is_empty() {
            return;
        }
        let Some(channel_name) = self.active_channel_name(channels, selected_channel) else {
            return;
        };

        if self.analysis_live_threshold_channel_name.as_deref() != Some(channel_name) {
            self.load_live_threshold_rules(Some(channel_name));
        }

        let target_column = self
            .saved_threshold_rules()
            .iter()
            .find(|rule| rule.channel_name.as_deref() == Some(channel_name))
            .map(|rule| rule.column_key.clone())
            .or_else(|| self.mapped_column_for_channel(channel_name, channels, numeric_columns));

        let Some(target_column) = target_column else {
            return;
        };
        let Some(idx) = numeric_columns
            .iter()
            .position(|column| column == &target_column)
        else {
            return;
        };
        self.assign_channel_to_column_rule(&target_column, Some(channel_name));
        if self.analysis_hist_channel != idx {
            self.analysis_hist_channel = idx;
            self.analysis_hist_drag_rule = None;
            self.sync_histogram_editor_to_column(&target_column);
        }
    }

    fn auto_map_channel_to_numeric_column(
        &mut self,
        channel_name: &str,
        channels: &[ChannelInfo],
        numeric_columns: &[String],
    ) -> Option<String> {
        self.channel_mapping_suggestions_for(channel_name, channels, numeric_columns)
            .and_then(|suggestions| suggestions.first().cloned())
    }

    pub(super) fn mapped_column_for_channel(
        &mut self,
        channel_name: &str,
        channels: &[ChannelInfo],
        numeric_columns: &[String],
    ) -> Option<String> {
        if let Some(column) = self.analysis_channel_mapping_overrides.get(channel_name)
            && numeric_columns.iter().any(|candidate| candidate == column)
        {
            return Some(column.clone());
        }
        self.auto_map_channel_to_numeric_column(channel_name, channels, numeric_columns)
    }

    fn channel_mapping_suggestions_for<'a>(
        &'a mut self,
        channel_name: &str,
        channels: &[ChannelInfo],
        numeric_columns: &[String],
    ) -> Option<&'a Vec<String>> {
        self.ensure_channel_mapping_suggestions_cache(channels, numeric_columns);
        self.analysis_channel_mapping_suggestions_cache
            .get(channel_name)
    }

    fn ensure_channel_mapping_suggestions_cache(
        &mut self,
        channels: &[ChannelInfo],
        numeric_columns: &[String],
    ) {
        use std::hash::{Hash, Hasher};

        if self.analysis_channel_mapping_suggestions_cache_key != 0
            && self.analysis_channel_mapping_suggestions_cache_channels_len == channels.len()
            && self.analysis_channel_mapping_suggestions_cache_numeric_len == numeric_columns.len()
        {
            return;
        }

        // Build a cheap, deterministic key without allocating a giant String.
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        channels.len().hash(&mut hasher);
        for channel in channels {
            channel.name.hash(&mut hasher);
        }
        numeric_columns.len().hash(&mut hasher);
        for column in numeric_columns {
            column.hash(&mut hasher);
        }
        let cache_key = hasher.finish();
        if self.analysis_channel_mapping_suggestions_cache_key == cache_key {
            return;
        }

        let token_frequencies = analysis_token_frequencies(channels, numeric_columns);
        let column_tokens = numeric_columns
            .iter()
            .map(|column| (column.clone(), analysis_name_tokens(column)))
            .collect::<Vec<_>>();

        let mut cache = HashMap::new();
        for channel in channels {
            let channel_tokens = analysis_name_tokens(&channel.name);
            if channel_tokens.is_empty() {
                cache.insert(channel.name.clone(), Vec::new());
                continue;
            }

            let normalized_channel = normalize_analysis_name(&channel.name);
            let mut scored = column_tokens
                .iter()
                .filter_map(|(column, tokens)| {
                    let shared = channel_tokens
                        .intersection(tokens)
                        .cloned()
                        .collect::<Vec<_>>();
                    if shared.is_empty() {
                        return None;
                    }
                    let mut score = 0i32;
                    for token in shared {
                        let freq = token_frequencies.get(&token).copied().unwrap_or(1).max(1);
                        let rarity_weight = (200 / freq as i32).max(20);
                        let len_bonus = (token.len() as i32).min(12) * 4;
                        score += rarity_weight + len_bonus;
                    }
                    let normalized_column = normalize_analysis_name(column);
                    if normalized_column == normalized_channel {
                        score += 200;
                    }
                    if normalized_column.contains(&normalized_channel) {
                        score += 120;
                    }
                    score += fuzzy_name_score(&channel.name, column).unwrap_or_default() / 200;
                    Some((score, column.clone()))
                })
                .collect::<Vec<_>>();
            scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            scored.dedup_by(|a, b| a.1 == b.1);
            cache.insert(
                channel.name.clone(),
                scored.into_iter().map(|(_, column)| column).collect(),
            );
        }

        self.analysis_channel_mapping_suggestions_cache_key = cache_key;
        self.analysis_channel_mapping_suggestions_cache_channels_len = channels.len();
        self.analysis_channel_mapping_suggestions_cache_numeric_len = numeric_columns.len();
        self.analysis_channel_mapping_suggestions_cache = cache;
    }

    fn sync_current_column_rule_transform(&mut self, column_name: &str) {
        for rule in self
            .analysis_property_thresholds
            .iter_mut()
            .filter(|rule| rule.column_key == column_name)
        {
            rule.value_transform = self.analysis_hist_value_transform;
        }
        self.sync_active_threshold_element_from_live_rules();
    }

    fn assign_channel_to_column_rule(&mut self, column_name: &str, channel_name: Option<&str>) {
        let mut changed = false;
        for rule in self
            .analysis_property_thresholds
            .iter_mut()
            .filter(|rule| rule.column_key == column_name)
        {
            let next = channel_name.map(ToOwned::to_owned);
            if rule.channel_name != next {
                rule.channel_name = next;
                changed = true;
            }
        }
        if changed {
            self.sync_active_threshold_element_from_live_rules();
        }
    }

    fn saved_threshold_rules(&self) -> Vec<ObjectPropertyThresholdRule> {
        self.analysis_threshold_selected_element
            .filter(|idx| *idx < self.analysis_threshold_elements.len())
            .and_then(|idx| self.analysis_threshold_elements.get(idx))
            .map(|element| element.rules.clone())
            .unwrap_or_else(|| self.analysis_property_thresholds.clone())
    }

    fn extract_live_threshold_rules(
        &self,
        rules: &[ObjectPropertyThresholdRule],
        channel_name: Option<&str>,
    ) -> Vec<ObjectPropertyThresholdRule> {
        if !self.analysis_follow_active_channel {
            return rules.to_vec();
        }

        let Some(channel_name) = channel_name else {
            return Vec::new();
        };
        let tagged = rules
            .iter()
            .filter(|rule| rule.channel_name.as_deref() == Some(channel_name))
            .cloned()
            .collect::<Vec<_>>();
        if !tagged.is_empty() {
            return tagged;
        }

        // Backward compatibility for older threshold sets that predate per-channel tagging.
        if rules.iter().all(|rule| rule.channel_name.is_none()) {
            return rules.to_vec();
        }

        Vec::new()
    }

    fn load_live_threshold_rules(&mut self, channel_name: Option<&str>) {
        let rules = self.saved_threshold_rules();
        self.analysis_property_thresholds = self.extract_live_threshold_rules(&rules, channel_name);
        self.analysis_live_threshold_channel_name = if self.analysis_follow_active_channel {
            channel_name.map(ToOwned::to_owned)
        } else {
            None
        };
        self.analysis_hist_drag_rule = None;
        self.analysis_hist_focus_object_index = None;
        self.analysis_hist_snapped_level = None;
        if let Some(rule) = self.analysis_property_thresholds.first() {
            self.analysis_hist_value_transform = rule.value_transform;
        }
        self.mark_live_analysis_selection_dirty();
    }

    fn save_live_threshold_rules(&mut self) {
        if let Some(idx) = self
            .analysis_threshold_selected_element
            .filter(|idx| *idx < self.analysis_threshold_elements.len())
        {
            if self.analysis_follow_active_channel {
                let active_channel = self.analysis_live_threshold_channel_name.clone();
                let mut merged = self.analysis_threshold_elements[idx].rules.clone();
                if let Some(channel_name) = active_channel.as_deref() {
                    merged.retain(|rule| rule.channel_name.as_deref() != Some(channel_name));
                } else {
                    merged.clear();
                }
                merged.extend(self.analysis_property_thresholds.clone());
                self.analysis_threshold_elements[idx].rules = merged;
            } else {
                self.analysis_threshold_elements[idx].rules =
                    self.analysis_property_thresholds.clone();
            }
        } else if !self.analysis_property_thresholds.is_empty() {
            let next_idx = self.analysis_threshold_elements.len() + 1;
            self.analysis_threshold_elements.push(ThresholdSetElement {
                name: format!("Element {next_idx}"),
                rules: self.analysis_property_thresholds.clone(),
            });
            self.analysis_threshold_selected_element =
                Some(self.analysis_threshold_elements.len() - 1);
        }
    }

    fn load_threshold_element(&mut self, idx: usize, active_channel_name: Option<&str>) {
        let Some(element) = self.analysis_threshold_elements.get(idx).cloned() else {
            return;
        };
        self.analysis_threshold_selected_element = Some(idx);
        self.analysis_property_thresholds =
            self.extract_live_threshold_rules(&element.rules, active_channel_name);
        self.analysis_live_threshold_channel_name = if self.analysis_follow_active_channel {
            active_channel_name.map(ToOwned::to_owned)
        } else {
            None
        };
        self.analysis_hist_drag_rule = None;
        self.analysis_hist_focus_object_index = None;
        if let Some(rule) = self.analysis_property_thresholds.first() {
            self.analysis_hist_value_transform = rule.value_transform;
        }
        self.analysis_hist_snapped_level = None;
        self.mark_live_analysis_selection_dirty();
    }

    fn sync_active_threshold_element_from_live_rules(&mut self) {
        self.mark_live_analysis_selection_dirty();
        self.save_live_threshold_rules();
    }

    fn save_threshold_set_dialog(&self) {
        let Some(path) = FileDialog::new()
            .add_filter("Threshold set", &["json"])
            .set_title("Save threshold set")
            .set_file_name("threshold_set.json")
            .save_file()
        else {
            return;
        };
        let payload = ThresholdSetFile {
            name: self.analysis_threshold_set_name.clone(),
            elements: self.analysis_threshold_elements.clone(),
        };
        if let Ok(text) = serde_json::to_string_pretty(&payload) {
            let _ = std::fs::write(path, text);
        }
    }

    fn load_threshold_set_dialog(&mut self) {
        let Some(path) = FileDialog::new()
            .add_filter("Threshold set", &["json"])
            .set_title("Load threshold set")
            .pick_file()
        else {
            return;
        };
        let Ok(text) = std::fs::read_to_string(path) else {
            return;
        };
        let Ok(payload) = serde_json::from_str::<ThresholdSetFile>(&text) else {
            return;
        };
        self.analysis_threshold_set_name = payload.name;
        self.analysis_threshold_elements = payload.elements;
        self.analysis_threshold_selected_element = if self.analysis_threshold_elements.is_empty() {
            None
        } else {
            Some(0)
        };
        if self.analysis_threshold_selected_element.is_some() {
            let active_channel = self.analysis_live_threshold_channel_name.clone();
            self.load_threshold_element(0, active_channel.as_deref());
        } else {
            self.analysis_property_thresholds.clear();
            self.analysis_live_threshold_channel_name = None;
        }
    }

    fn analysis_value_name_picker(
        ui: &mut egui::Ui,
        label: &str,
        id_salt: impl std::hash::Hash,
        names: &[String],
        selected: &mut String,
    ) {
        if names.is_empty() {
            return;
        }
        if !names.iter().any(|name| name == selected) {
            *selected = names[0].clone();
        }
        ui.horizontal(|ui| {
            ui.label(label);
            let selected_label = selected.clone();
            let popup_state_id = ui.make_persistent_id((&id_salt, "popup_open"));
            let search_id = ui.make_persistent_id((&id_salt, "search"));
            let search_preview = ui
                .data(|d| d.get_temp::<String>(search_id))
                .unwrap_or_default();
            let mut popup_open = ui
                .data(|d| d.get_temp::<bool>(popup_state_id))
                .unwrap_or(false);
            let was_popup_open = popup_open;
            let button_resp = ui.button(selected_label.clone());
            if button_resp.clicked() {
                popup_open = !popup_open;
            }
            let just_opened = popup_open && !was_popup_open;
            let popup_width = analysis_picker_popup_width(
                ui,
                button_resp.rect.width(),
                names,
                &search_preview,
                Some(selected_label.as_str()),
            );

            let popup_id = ui.make_persistent_id((&id_salt, "popup"));
            egui::Popup::from_response(&button_resp)
                .id(popup_id)
                .open_bool(&mut popup_open)
                .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                .gap(2.0)
                .width(popup_width)
                .show(|ui| {
                    let mut request_close = false;
                    ui.set_min_width(popup_width);
                    let mut search = ui
                        .data(|d| d.get_temp::<String>(search_id))
                        .unwrap_or_default();
                    let response = ui.add(
                        egui::TextEdit::singleline(&mut search)
                            .hint_text("Search columns...")
                            .desired_width(ui.available_width()),
                    );
                    if just_opened {
                        response.request_focus();
                    }
                    if response.changed() {
                        ui.data_mut(|d| d.insert_temp(search_id, search.clone()));
                    }
                    ui.separator();

                    let mut matches = names
                        .iter()
                        .filter_map(|name| {
                            fuzzy_name_score(&search, name).map(|score| (score, name))
                        })
                        .collect::<Vec<_>>();
                    matches.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(b.1)));

                    egui::ScrollArea::vertical()
                        .id_salt((&id_salt, "scroll"))
                        .auto_shrink([false, false])
                        .max_height(320.0)
                        .show(ui, |ui| {
                            for (_, name) in matches.into_iter().take(100) {
                                if ui.selectable_label(*selected == *name, name).clicked() {
                                    *selected = name.clone();
                                    request_close = true;
                                }
                            }
                        });
                    if request_close {
                        ui.close();
                    }
                });
            ui.data_mut(|d| d.insert_temp(popup_state_id, popup_open));
        });
    }

    fn analysis_optional_value_name_picker(
        ui: &mut egui::Ui,
        label: &str,
        id_salt: impl std::hash::Hash,
        names: &[String],
        selected: &mut Option<String>,
        placeholder: &str,
    ) {
        if names.is_empty() {
            return;
        }
        ui.horizontal(|ui| {
            if !label.is_empty() {
                ui.label(label);
            }
            let selected_label = selected.as_deref().unwrap_or(placeholder).to_string();
            let popup_state_id = ui.make_persistent_id((&id_salt, "popup_open"));
            let search_id = ui.make_persistent_id((&id_salt, "search"));
            let search_preview = ui
                .data(|d| d.get_temp::<String>(search_id))
                .unwrap_or_default();
            let mut popup_open = ui
                .data(|d| d.get_temp::<bool>(popup_state_id))
                .unwrap_or(false);
            let was_popup_open = popup_open;
            let button_resp = ui.button(selected_label.clone());
            if button_resp.clicked() {
                popup_open = !popup_open;
            }
            let just_opened = popup_open && !was_popup_open;
            let popup_width = analysis_picker_popup_width(
                ui,
                button_resp.rect.width(),
                names,
                &search_preview,
                Some(selected_label.as_str()),
            );

            let popup_id = ui.make_persistent_id((&id_salt, "popup"));
            egui::Popup::from_response(&button_resp)
                .id(popup_id)
                .open_bool(&mut popup_open)
                .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                .gap(2.0)
                .width(popup_width)
                .show(|ui| {
                    let mut request_close = false;
                    ui.set_min_width(popup_width);
                    let mut search = ui
                        .data(|d| d.get_temp::<String>(search_id))
                        .unwrap_or_default();
                    let response = ui.add(
                        egui::TextEdit::singleline(&mut search)
                            .hint_text("Search columns...")
                            .desired_width(ui.available_width()),
                    );
                    if just_opened {
                        response.request_focus();
                    }
                    if response.changed() {
                        ui.data_mut(|d| d.insert_temp(search_id, search.clone()));
                    }
                    ui.separator();

                    let mut matches = names
                        .iter()
                        .filter_map(|name| {
                            fuzzy_name_score(&search, name).map(|score| (score, name))
                        })
                        .collect::<Vec<_>>();
                    matches.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(b.1)));

                    egui::ScrollArea::vertical()
                        .id_salt((&id_salt, "scroll"))
                        .auto_shrink([false, false])
                        .max_height(320.0)
                        .show(ui, |ui| {
                            for (_, name) in matches.into_iter().take(100) {
                                if ui
                                    .selectable_label(
                                        selected.as_deref() == Some(name.as_str()),
                                        name,
                                    )
                                    .clicked()
                                {
                                    *selected = Some(name.clone());
                                    request_close = true;
                                }
                            }
                        });
                    if request_close {
                        ui.close();
                    }
                });
            ui.data_mut(|d| d.insert_temp(popup_state_id, popup_open));
        });
    }

    fn ensure_spatial_table_meta(
        &mut self,
        spatial_root: &Path,
        table: &SpatialDataElement,
    ) -> anyhow::Result<&SpatialDataTableMeta> {
        if self.analysis_spatial_meta_table_name != table.name {
            self.analysis_spatial_meta = None;
        }
        if self.analysis_spatial_meta.is_none() {
            let meta = load_table_meta(spatial_root, table)?;
            if !meta
                .join_keys
                .iter()
                .any(|key| key == &self.analysis_spatial_join_key)
            {
                self.analysis_spatial_join_key =
                    meta.join_keys.first().cloned().unwrap_or_default();
            }
            self.analysis_spatial_meta = Some(meta);
            self.analysis_spatial_meta_table_name = table.name.clone();
        }
        self.analysis_spatial_meta
            .as_ref()
            .context("missing SpatialData table metadata")
    }

    fn available_object_join_keys(&self) -> Vec<String> {
        let Some(objects) = self.objects.as_ref() else {
            return Vec::new();
        };

        let mut out = Vec::new();
        if !objects.is_empty() {
            out.push("id".to_string());
        }
        for key in ["instance_id", "cell_id", "label", "object_id", "name"] {
            if objects.iter().any(|obj| obj.properties.contains_key(key)) {
                out.push(key.to_string());
            }
        }
        out
    }

    pub(super) fn available_numeric_object_property_keys(&mut self) -> Vec<String> {
        if let Some(cached) = self.object_property_numeric_keys_cache.as_ref() {
            return cached.clone();
        }
        let Some(objects) = self.objects.as_ref() else {
            return Vec::new();
        };
        let out = self
            .scalar_property_keys
            .iter()
            .filter(|key| key.as_str() != "id")
            .filter(|key| {
                objects.iter().any(|obj| {
                    obj.properties
                        .get(*key)
                        .and_then(numeric_json_value)
                        .is_some()
                })
            })
            .cloned()
            .collect::<Vec<_>>();
        self.object_property_numeric_keys_cache = Some(out.clone());
        out
    }

    fn object_property_column_pairs(&mut self, key: &str) -> Arc<Vec<(usize, f32)>> {
        if self.filtered_indices.is_none() {
            if let Some(cached) = self.object_property_base_pairs_cache.get(key) {
                return cached.clone();
            }
            let Some(objects) = self.objects.as_ref() else {
                return Arc::new(Vec::new());
            };
            let mut out = Vec::new();
            for (idx, obj) in objects.iter().enumerate() {
                let Some(value) = obj.properties.get(key).and_then(numeric_json_value) else {
                    continue;
                };
                if value.is_finite() {
                    out.push((idx, value));
                }
            }
            let out = Arc::new(out);
            self.object_property_base_pairs_cache
                .insert(key.to_string(), out.clone());
            return out;
        }

        if let Some(cached) = self.object_property_pairs_cache.get(key) {
            return cached.clone();
        }
        let Some(objects) = self.objects.as_ref() else {
            return Arc::new(Vec::new());
        };
        let filtered = self.filtered_indices.as_ref();
        let mut out = Vec::new();
        for (idx, obj) in objects.iter().enumerate() {
            if filtered.is_some_and(|set| !set.contains(&idx)) {
                continue;
            }
            let Some(value) = obj.properties.get(key).and_then(numeric_json_value) else {
                continue;
            };
            if value.is_finite() {
                out.push((idx, value));
            }
        }
        let out = Arc::new(out);
        self.object_property_pairs_cache
            .insert(key.to_string(), out.clone());
        out
    }

    fn object_property_sorted_pairs(&mut self, key: &str) -> Arc<Vec<(usize, f32)>> {
        if self.filtered_indices.is_none() {
            if let Some(cached) = self.object_property_base_sorted_pairs_cache.get(key) {
                return cached.clone();
            }
            let mut sorted = self.object_property_column_pairs(key).as_ref().clone();
            sorted.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            let sorted = Arc::new(sorted);
            self.object_property_base_sorted_pairs_cache
                .insert(key.to_string(), sorted.clone());
            return sorted;
        }

        let mut sorted = self.object_property_column_pairs(key).as_ref().clone();
        sorted.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        Arc::new(sorted)
    }

    fn object_property_histogram(&mut self, key: &str) -> Option<SimpleHistogram> {
        // Keep separate caches for base vs filtered histograms. Filter changes are much more
        // frequent than raw data changes, so this avoids rebuilding the unfiltered histogram when
        // users toggle subset views.
        if self.filtered_indices.is_none() {
            let cache_key = (key.to_string(), self.analysis_hist_value_transform);
            if let Some(cached) = self.object_property_base_hist_cache.get(&cache_key) {
                return Some(cached.clone());
            }
            let values = self.object_property_histogram_values(key);
            let hist = finite_min_max_f32(&values).map(|_| compute_histogram_f32(&values, 128))?;
            self.object_property_base_hist_cache
                .insert(cache_key, hist.clone());
            return Some(hist);
        }

        let cache_key = (key.to_string(), self.analysis_hist_value_transform);
        if let Some(cached) = self.object_property_hist_cache.get(&cache_key) {
            return Some::<SimpleHistogram>(cached.clone());
        }
        let values = self.object_property_histogram_values(key);
        let hist = finite_min_max_f32(&values).map(|_| compute_histogram_f32(&values, 128))?;
        self.object_property_hist_cache
            .insert(cache_key, hist.clone());
        Some(hist)
    }

    fn object_property_scatter_points(
        &mut self,
        x_key: &str,
        y_key: &str,
    ) -> Arc<Vec<(usize, f32, f32)>> {
        // Scatter plots are built by intersecting the per-column value maps on object index. The
        // cached output therefore preserves a stable object-id link for brushing back into the
        // layer selection model.
        let cache_key = (x_key.to_string(), y_key.to_string());
        if let Some(cached) = self.object_property_scatter_cache.get(&cache_key) {
            return cached.clone();
        }
        let x_pairs = self.object_property_column_pairs(x_key);
        let y_pairs = self.object_property_column_pairs(y_key);
        let x_map = x_pairs.iter().copied().collect::<HashMap<_, _>>();
        let mut out = Vec::<(usize, f32, f32)>::new();
        for (object_index, y) in y_pairs.iter() {
            let Some(x) = x_map.get(object_index).copied() else {
                continue;
            };
            if x.is_finite() && y.is_finite() {
                out.push((*object_index, x, *y));
            }
        }
        let out = Arc::new(out);
        self.object_property_scatter_cache
            .insert(cache_key, out.clone());
        out
    }

    pub(super) fn invalidate_object_property_analysis_cache(&mut self) {
        self.object_property_pairs_cache.clear();
        self.object_property_hist_cache.clear();
        self.object_property_scatter_cache.clear();
        self.object_property_hist_levels_cache.clear();
        self.object_property_threshold_selection_cache_key = None;
        self.object_property_threshold_selection_cache = Arc::new(Vec::new());
        self.object_property_threshold_order_cache_key = None;
        self.object_property_threshold_order_cache = Arc::new(Vec::new());
    }

    pub(super) fn reset_object_property_analysis_cache(&mut self) {
        self.object_property_numeric_keys_cache = None;
        self.object_property_base_pairs_cache.clear();
        self.object_property_base_sorted_pairs_cache.clear();
        self.object_property_base_hist_cache.clear();
        self.object_property_base_hist_levels_cache.clear();
        self.analysis_warm_started = false;
        self.analysis_warm_rx = None;
        self.analysis_warm_total_columns = 0;
        self.analysis_warm_completed_columns = 0;
        self.invalidate_object_property_analysis_cache();
    }

    fn object_lookup_for_join_key(
        &self,
        join_key: &str,
        filtered_only: bool,
    ) -> HashMap<String, (usize, String)> {
        let Some(objects) = self.objects.as_ref() else {
            return HashMap::new();
        };
        let filtered = self.filtered_indices.as_ref();
        let mut out = HashMap::new();
        for (idx, obj) in objects.iter().enumerate() {
            if filtered_only && filtered.is_some_and(|set| !set.contains(&idx)) {
                continue;
            }
            let value = if join_key == "id" {
                Some(obj.id.clone())
            } else {
                obj.properties
                    .get(join_key)
                    .and_then(json_value_short_string)
            };
            if let Some(value) = value {
                out.entry(value).or_insert_with(|| (idx, obj.id.clone()));
            }
        }
        out
    }

    fn sync_live_analysis_selection(&mut self, indices: &[usize]) {
        self.apply_selection_indices(indices, false);
    }

    pub(super) fn has_live_analysis_selection(&self) -> bool {
        !self.analysis_property_thresholds.is_empty() || self.analysis_scatter_brush.is_some()
    }

    fn ensure_default_object_property_threshold(
        &mut self,
        column_key: &str,
        default_value: f32,
        channel_name: Option<&str>,
    ) {
        let desired_channel = channel_name.map(ToOwned::to_owned);

        if let Some(idx) = desired_channel.as_ref().and_then(|channel_name| {
            self.analysis_property_thresholds
                .iter()
                .position(|rule| rule.channel_name.as_deref() == Some(channel_name.as_str()))
        }) {
            let rule = &mut self.analysis_property_thresholds[idx];
            let mut changed = false;
            if rule.column_key != column_key {
                rule.column_key = column_key.to_string();
                changed = true;
            }
            if rule.value_transform != self.analysis_hist_value_transform {
                rule.value_transform = self.analysis_hist_value_transform;
                changed = true;
            }
            if changed {
                self.sync_active_threshold_element_from_live_rules();
            }
            return;
        }

        if let Some(idx) = self
            .analysis_property_thresholds
            .iter()
            .position(|rule| rule.column_key == column_key)
        {
            let rule = &mut self.analysis_property_thresholds[idx];
            let mut changed = false;
            if rule.channel_name != desired_channel {
                rule.channel_name = desired_channel;
                changed = true;
            }
            if rule.value_transform != self.analysis_hist_value_transform {
                rule.value_transform = self.analysis_hist_value_transform;
                changed = true;
            }
            if changed {
                self.sync_active_threshold_element_from_live_rules();
            }
            return;
        }

        if self.analysis_follow_active_channel
            && desired_channel.is_some()
            && self.analysis_property_thresholds.len() == 1
        {
            let rule = &mut self.analysis_property_thresholds[0];
            let mut changed = false;
            if rule.column_key != column_key {
                rule.column_key = column_key.to_string();
                rule.value = default_value;
                changed = true;
            }
            if rule.channel_name != desired_channel {
                rule.channel_name = desired_channel;
                changed = true;
            }
            if rule.value_transform != self.analysis_hist_value_transform {
                rule.value_transform = self.analysis_hist_value_transform;
                changed = true;
            }
            if changed {
                self.sync_active_threshold_element_from_live_rules();
            }
            return;
        }

        if !self.analysis_property_thresholds.is_empty() {
            return;
        }
        self.analysis_property_thresholds
            .push(ObjectPropertyThresholdRule {
                column_key: column_key.to_string(),
                channel_name: channel_name.map(ToOwned::to_owned),
                op: AnalysisThresholdOp::GreaterEqual,
                value: default_value,
                value_transform: self.analysis_hist_value_transform,
            });
        self.sync_active_threshold_element_from_live_rules();
    }

    fn ui_object_property_threshold_rules(
        &mut self,
        ui: &mut egui::Ui,
        numeric_columns: &[String],
        default_column: &str,
    ) {
        ui.separator();
        ui.label("Thresholds");
        let mut remove_idx = None;
        let mut changed = false;
        for (idx, rule) in self.analysis_property_thresholds.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                let prev_column = rule.column_key.clone();
                Self::analysis_value_name_picker(
                    ui,
                    "Column",
                    ("seg_objects_threshold_column", idx),
                    numeric_columns,
                    &mut rule.column_key,
                );
                if rule.column_key != prev_column {
                    changed = true;
                }
                let prev_op = rule.op;
                egui::ComboBox::from_id_salt(("seg_objects_threshold_op", idx))
                    .selected_text(match rule.op {
                        AnalysisThresholdOp::GreaterEqual => ">=",
                        AnalysisThresholdOp::LessEqual => "<=",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut rule.op, AnalysisThresholdOp::GreaterEqual, ">=");
                        ui.selectable_value(&mut rule.op, AnalysisThresholdOp::LessEqual, "<=");
                    });
                if rule.op != prev_op {
                    changed = true;
                }
                let response = ui.add(egui::DragValue::new(&mut rule.value).speed(0.1));
                if response.changed() {
                    changed = true;
                }
                if ui.button("Remove").clicked() {
                    remove_idx = Some(idx);
                }
            });
        }
        if let Some(idx) = remove_idx {
            self.analysis_property_thresholds.remove(idx);
            changed = true;
        }
        if ui.button("Add threshold").clicked() {
            self.analysis_property_thresholds
                .push(ObjectPropertyThresholdRule {
                    column_key: default_column.to_string(),
                    channel_name: self.analysis_live_threshold_channel_name.clone(),
                    op: AnalysisThresholdOp::GreaterEqual,
                    value: 0.0,
                    value_transform: self.analysis_hist_value_transform,
                });
            changed = true;
        }
        if changed {
            self.sync_active_threshold_element_from_live_rules();
        }
    }

    fn ui_object_property_histogram_levels(&mut self, ui: &mut egui::Ui, column_name: &str) {
        ui.separator();
        ui.label("Levels");
        ui.horizontal(|ui| {
            ui.selectable_value(
                &mut self.analysis_hist_level_method,
                HistogramLevelMethod::Quantiles,
                "Quantiles",
            );
            ui.selectable_value(
                &mut self.analysis_hist_level_method,
                HistogramLevelMethod::KMeans,
                "K-means",
            );
            let label = match self.analysis_hist_level_method {
                HistogramLevelMethod::Quantiles => "Bins",
                HistogramLevelMethod::KMeans => "K",
            };
            ui.label(label);
            ui.add(
                egui::DragValue::new(&mut self.analysis_hist_level_count)
                    .range(2..=12)
                    .speed(0.05),
            );
        });

        let levels = self.object_property_histogram_levels(
            column_name,
            self.analysis_hist_value_transform,
            self.analysis_hist_level_method,
            self.analysis_hist_level_count.max(2),
        );
        if levels.is_empty() {
            ui.label("No level boundaries available for this column.");
            return;
        }

        ui.horizontal(|ui| {
            let current_value = self
                .analysis_property_thresholds
                .iter()
                .find(|rule| rule.column_key == column_name)
                .map(|rule| {
                    apply_histogram_value_transform(rule.value, self.analysis_hist_value_transform)
                });
            let prev = current_value.and_then(|current| {
                levels
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(_, v)| *v < current)
                    .next_back()
            });
            let next = current_value.and_then(|current| {
                levels
                    .iter()
                    .copied()
                    .enumerate()
                    .find(|(_, v)| *v > current)
            });

            if ui
                .add_enabled(prev.is_some(), egui::Button::new("Bin down"))
                .clicked()
                && let Some((level_index, value)) = prev
            {
                self.set_histogram_threshold_snap(column_name, value, level_index);
            }
            if ui
                .add_enabled(next.is_some(), egui::Button::new("Bin up"))
                .clicked()
                && let Some((level_index, value)) = next
            {
                self.set_histogram_threshold_snap(column_name, value, level_index);
            }
        });

        if let Some(selection) = self
            .analysis_hist_snapped_level
            .as_ref()
            .filter(|selection| {
                selection.column_key == column_name
                    && selection.method == self.analysis_hist_level_method
                    && selection.level_count == self.analysis_hist_level_count.max(2)
                    && selection.level_index < levels.len()
                    && selection.value_transform == self.analysis_hist_value_transform
            })
        {
            ui.label(format!(
                "Selected level: {} ({:.2})",
                histogram_level_label(
                    selection.method,
                    selection.level_count,
                    selection.level_index,
                    levels.len()
                ),
                levels[selection.level_index]
            ));
        }

        ui.horizontal_wrapped(|ui| {
            ui.label("Snap threshold:");
            for (level_index, &value) in levels.iter().enumerate() {
                let selected = self
                    .analysis_hist_snapped_level
                    .as_ref()
                    .is_some_and(|selection| {
                        selection.column_key == column_name
                            && selection.value_transform == self.analysis_hist_value_transform
                            && selection.method == self.analysis_hist_level_method
                            && selection.level_count == self.analysis_hist_level_count.max(2)
                            && selection.level_index == level_index
                    });
                let label = format!(
                    "{} ({value:.2})",
                    histogram_level_label(
                        self.analysis_hist_level_method,
                        self.analysis_hist_level_count.max(2),
                        level_index,
                        levels.len()
                    )
                );
                if ui.selectable_label(selected, label).clicked() {
                    self.set_histogram_threshold_snap(column_name, value, level_index);
                }
            }
        });
    }

    fn set_histogram_threshold_value(&mut self, column_name: &str, value: f32) {
        if let Some(rule) = self
            .analysis_property_thresholds
            .iter_mut()
            .find(|rule| rule.column_key == column_name)
        {
            rule.value = value;
            rule.value_transform = self.analysis_hist_value_transform;
            self.sync_active_threshold_element_from_live_rules();
            return;
        }
        self.analysis_property_thresholds
            .push(ObjectPropertyThresholdRule {
                column_key: column_name.to_string(),
                channel_name: self.analysis_live_threshold_channel_name.clone(),
                op: AnalysisThresholdOp::GreaterEqual,
                value,
                value_transform: self.analysis_hist_value_transform,
            });
        self.sync_active_threshold_element_from_live_rules();
    }

    fn set_histogram_threshold_snap(&mut self, column_name: &str, value: f32, level_index: usize) {
        self.set_histogram_threshold_value(
            column_name,
            invert_histogram_value_transform(value, self.analysis_hist_value_transform),
        );
        self.analysis_hist_snapped_level = Some(HistogramLevelSelection {
            column_key: column_name.to_string(),
            value_transform: self.analysis_hist_value_transform,
            method: self.analysis_hist_level_method,
            level_count: self.analysis_hist_level_count.max(2),
            level_index,
        });
    }

    fn clear_histogram_snapped_level_for_column(&mut self, column_name: &str) {
        if self
            .analysis_hist_snapped_level
            .as_ref()
            .is_some_and(|selection| selection.column_key == column_name)
        {
            self.analysis_hist_snapped_level = None;
        }
    }

    fn object_property_threshold_selected_indices(&mut self) -> Arc<Vec<usize>> {
        if self.analysis_property_thresholds.is_empty() {
            return Arc::new(Vec::new());
        }
        let cache_key = self
            .analysis_property_thresholds
            .iter()
            .map(|rule| {
                format!(
                    "{}|{}|{:.6}",
                    rule.column_key,
                    match rule.op {
                        AnalysisThresholdOp::GreaterEqual => "ge",
                        AnalysisThresholdOp::LessEqual => "le",
                    },
                    rule.value
                )
            })
            .collect::<Vec<_>>()
            .join("||");
        if self
            .object_property_threshold_selection_cache_key
            .as_deref()
            == Some(&cache_key)
        {
            return Arc::clone(&self.object_property_threshold_selection_cache);
        }
        if self.filtered_indices.is_none() && self.analysis_property_thresholds.len() == 1 {
            let rule = self.analysis_property_thresholds[0].clone();
            let sorted = self.object_property_sorted_pairs(&rule.column_key);
            let out = Arc::new(match rule.op {
                AnalysisThresholdOp::GreaterEqual => sorted
                    .iter()
                    .filter(|(_, value)| *value >= rule.value)
                    .map(|(object_index, _)| *object_index)
                    .collect::<Vec<_>>(),
                AnalysisThresholdOp::LessEqual => sorted
                    .iter()
                    .take_while(|(_, value)| *value <= rule.value)
                    .map(|(object_index, _)| *object_index)
                    .collect::<Vec<_>>(),
            });
            self.object_property_threshold_selection_cache_key = Some(cache_key);
            self.object_property_threshold_selection_cache = Arc::clone(&out);
            return out;
        }
        let rules = self.analysis_property_thresholds.clone();
        let mut selected: Option<HashSet<usize>> = None;
        for rule in &rules {
            let pairs = self.object_property_column_pairs(&rule.column_key);
            let rule_matches = pairs
                .iter()
                .filter_map(|(object_index, value)| {
                    let matches = match rule.op {
                        AnalysisThresholdOp::GreaterEqual => *value >= rule.value,
                        AnalysisThresholdOp::LessEqual => *value <= rule.value,
                    };
                    matches.then_some(*object_index)
                })
                .collect::<HashSet<_>>();
            selected = Some(match selected {
                Some(mut current) => {
                    current.retain(|idx| rule_matches.contains(idx));
                    current
                }
                None => rule_matches,
            });
        }
        let out = Arc::new(selected.unwrap_or_default().into_iter().collect::<Vec<_>>());
        self.object_property_threshold_selection_cache_key = Some(cache_key);
        self.object_property_threshold_selection_cache = Arc::clone(&out);
        out
    }

    fn object_property_histogram_levels(
        &mut self,
        column_name: &str,
        value_transform: HistogramValueTransform,
        method: HistogramLevelMethod,
        level_count: usize,
    ) -> Arc<Vec<f32>> {
        if self.filtered_indices.is_none() {
            let cache_key = (
                column_name.to_string(),
                value_transform,
                method,
                level_count,
            );
            if let Some(cached) = self.object_property_base_hist_levels_cache.get(&cache_key) {
                return Arc::clone(cached);
            }
            let values = self.object_property_histogram_values(column_name);
            let levels = match method {
                HistogramLevelMethod::Quantiles => quantile_threshold_levels(&values, level_count),
                HistogramLevelMethod::KMeans => kmeans_threshold_levels(&values, level_count, 24),
            };
            let levels = Arc::new(levels);
            self.object_property_base_hist_levels_cache
                .insert(cache_key, Arc::clone(&levels));
            return levels;
        }

        let cache_key = (
            column_name.to_string(),
            value_transform,
            method,
            level_count,
        );
        if let Some(cached) = self.object_property_hist_levels_cache.get(&cache_key) {
            return Arc::clone(cached);
        }
        let values = self.object_property_histogram_values(column_name);
        let levels = match method {
            HistogramLevelMethod::Quantiles => quantile_threshold_levels(&values, level_count),
            HistogramLevelMethod::KMeans => kmeans_threshold_levels(&values, level_count, 24),
        };
        let levels = Arc::new(levels);
        self.object_property_hist_levels_cache
            .insert(cache_key, Arc::clone(&levels));
        levels
    }

    fn object_property_threshold_ordered_indices(&mut self, column_name: &str) -> Arc<Vec<usize>> {
        let threshold_cache_key = self
            .analysis_property_thresholds
            .iter()
            .map(|rule| {
                format!(
                    "{}|{}|{:.6}",
                    rule.column_key,
                    match rule.op {
                        AnalysisThresholdOp::GreaterEqual => "ge",
                        AnalysisThresholdOp::LessEqual => "le",
                    },
                    rule.value
                )
            })
            .collect::<Vec<_>>()
            .join("||");
        let cache_key = format!(
            "{}|||{}|||{}",
            if self.filtered_indices.is_some() {
                "filtered"
            } else {
                "all"
            },
            column_name,
            threshold_cache_key
        );
        if self.object_property_threshold_order_cache_key.as_deref() == Some(&cache_key) {
            return Arc::clone(&self.object_property_threshold_order_cache);
        }

        let out = if self.filtered_indices.is_none()
            && self.analysis_property_thresholds.len() == 1
            && self.analysis_property_thresholds[0].column_key == column_name
        {
            let rule = self.analysis_property_thresholds[0].clone();
            let sorted = self.object_property_sorted_pairs(column_name);
            Arc::new(match rule.op {
                AnalysisThresholdOp::GreaterEqual => sorted
                    .iter()
                    .filter(|(_, value)| *value >= rule.value)
                    .map(|(object_index, _)| *object_index)
                    .collect(),
                AnalysisThresholdOp::LessEqual => sorted
                    .iter()
                    .take_while(|(_, value)| *value <= rule.value)
                    .map(|(object_index, _)| *object_index)
                    .collect(),
            })
        } else {
            let selected = self.object_property_threshold_selected_indices();
            if selected.is_empty() {
                Arc::new(Vec::new())
            } else {
                let selected_set = selected.iter().copied().collect::<HashSet<_>>();
                let mut pairs = self
                    .object_property_column_pairs(column_name)
                    .iter()
                    .copied()
                    .filter(|(object_index, value)| {
                        selected_set.contains(object_index) && value.is_finite()
                    })
                    .collect::<Vec<_>>();
                pairs.sort_by(|a, b| {
                    a.1.partial_cmp(&b.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.0.cmp(&b.0))
                });
                Arc::new(
                    pairs
                        .into_iter()
                        .map(|(object_index, _)| object_index)
                        .collect(),
                )
            }
        };

        self.object_property_threshold_order_cache_key = Some(cache_key);
        self.object_property_threshold_order_cache = Arc::clone(&out);
        out
    }

    fn object_property_histogram_values(&mut self, key: &str) -> Vec<f32> {
        let value_transform = self.analysis_hist_value_transform;
        self.object_property_column_pairs(key)
            .iter()
            .map(|(_, value)| apply_histogram_value_transform(*value, value_transform))
            .filter(|v| v.is_finite())
            .collect::<Vec<_>>()
    }

    fn ui_analysis_selection_actions(
        &mut self,
        ui: &mut egui::Ui,
        indices: &[usize],
        empty_hint: &str,
    ) {
        ui.horizontal(|ui| {
            ui.label(format!("Plot selection: {}", indices.len()));
            if ui.button("Clear brush").clicked() {
                self.analysis_hist_brush = None;
                self.analysis_scatter_brush = None;
            }
        });
        if indices.is_empty() {
            ui.label(empty_hint);
        }
    }

    pub fn sync_measurements(
        &mut self,
        dataset: &OmeZarrDataset,
        store: Arc<dyn ReadableStorageTraits>,
        channels: &[ChannelInfo],
        local_to_world_offset: egui::Vec2,
    ) {
        use crossbeam_channel::TryRecvError;

        loop {
            let Some(rx) = self.measurement_rx.as_ref() else {
                break;
            };
            match rx.try_recv() {
                Ok(msg) => {
                    if msg.request_id == self.measurement_request_id {
                        self.measurement_data = msg.data;
                        self.measurement_resolved_target = self.measurement_target.clone();
                        self.measurement_status = if let Some(err) = msg.error {
                            err
                        } else if let Some(data) = self.measurement_data.as_ref() {
                            format!(
                                "Measured {} channel(s) for {}.",
                                data.channels.len(),
                                data.object_id
                            )
                        } else {
                            String::new()
                        };
                    }
                    self.measurement_rx = None;
                    break;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.measurement_rx = None;
                    break;
                }
            }
        }

        let Some(object_index) = self.selected_object_index else {
            self.clear_measurements();
            return;
        };

        let target = MeasurementTarget {
            object_index,
            local_to_world_offset,
        };
        if self.measurement_target.as_ref() == Some(&target)
            && (self.measurement_rx.is_some()
                || self.measurement_resolved_target.as_ref() == Some(&target))
        {
            return;
        }

        if dataset.channels.is_empty() || channels.is_empty() {
            self.measurement_target = Some(target);
            self.measurement_resolved_target = self.measurement_target.clone();
            self.measurement_data = None;
            self.measurement_rx = None;
            self.measurement_status = "No image channels available.".to_string();
            return;
        }
        if dataset.is_root_label_mask() {
            self.measurement_target = Some(target);
            self.measurement_resolved_target = self.measurement_target.clone();
            self.measurement_data = None;
            self.measurement_rx = None;
            self.measurement_status =
                "Image measurements are unavailable for label-mask datasets.".to_string();
            return;
        }
        let Some(object) = self
            .objects
            .as_ref()
            .and_then(|objects| objects.get(object_index).cloned())
        else {
            self.clear_measurements();
            return;
        };

        self.measurement_request_id = self.measurement_request_id.wrapping_add(1).max(1);
        let request_id = self.measurement_request_id;
        let (tx, rx) = crossbeam_channel::bounded::<MeasurementResult>(1);
        let dataset = dataset.clone();
        let channels = channels.to_vec();

        self.measurement_target = Some(target);
        self.measurement_resolved_target = None;
        self.measurement_data = None;
        self.measurement_rx = Some(rx);
        self.measurement_status = format!("Measuring {} channel(s)...", channels.len());

        std::thread::Builder::new()
            .name("seg-objects-measure".to_string())
            .spawn(move || {
                let outcome = measure_object_in_thread(
                    &dataset,
                    store,
                    &channels,
                    object_index,
                    &object,
                    local_to_world_offset,
                );
                let msg = match outcome {
                    Ok(data) => MeasurementResult {
                        request_id,
                        data: Some(data),
                        error: None,
                    },
                    Err(err) => MeasurementResult {
                        request_id,
                        data: None,
                        error: Some(format!("Image measurement failed: {err}")),
                    },
                };
                let _ = tx.send(msg);
            })
            .ok();
    }

    pub(super) fn clear_measurements(&mut self) {
        self.measurement_target = None;
        self.measurement_resolved_target = None;
        self.measurement_data = None;
        self.measurement_rx = None;
        self.measurement_status.clear();
    }

    pub(super) fn clear_analysis(&mut self) {
        if let Some(cancel) = self.analysis_cancel.take() {
            cancel.store(true, Ordering::Relaxed);
        }
        self.analysis_rx = None;
        self.analysis_progress_completed = 0;
        self.analysis_progress_total = 0;
        self.analysis_status.clear();
        self.analysis_results = None;
        self.analysis_property_thresholds.clear();
        self.sync_active_threshold_element_from_live_rules();
        self.analysis_hist_drag_rule = None;
        self.analysis_hist_brush = None;
        self.analysis_scatter_brush = None;
        self.analysis_hist_drag_anchor = None;
        self.analysis_scatter_drag_anchor = None;
        self.analysis_scatter_view_key = None;
        self.analysis_scatter_view_rect = None;
    }
}

#[derive(Debug, Clone)]
pub(super) struct SimpleHistogram {
    min: f32,
    max: f32,
    median: f32,
    bins: Vec<u32>,
    max_count: u32,
}

fn measure_object_in_thread(
    dataset: &OmeZarrDataset,
    store: Arc<dyn ReadableStorageTraits>,
    channels: &[ChannelInfo],
    object_index: usize,
    object: &GeoJsonObjectFeature,
    local_to_world_offset: egui::Vec2,
) -> anyhow::Result<ObjectMeasurementSet> {
    let channel_schema = channels
        .iter()
        .map(|channel| AnalysisChannelSchema {
            channel_index: channel.index,
            channel_name: channel.name.clone(),
        })
        .collect::<Vec<_>>();
    let level0 = dataset
        .levels
        .first()
        .context("dataset has no level 0 for measurement")?;
    let zarr_path = format!("/{}", level0.path.trim_start_matches('/'));
    let array: Array<dyn ReadableStorageTraits> = Array::open(store, &zarr_path)
        .with_context(|| format!("open measurement array {zarr_path}"))?;
    let row = measure_object_row_from_array(
        dataset,
        &array,
        &channel_schema,
        object_index,
        object,
        local_to_world_offset,
    )?;
    Ok(batch_row_to_object_measurement_set(row, &channel_schema))
}

fn measure_objects_batch_in_thread(
    dataset: &OmeZarrDataset,
    store: Arc<dyn ReadableStorageTraits>,
    channels: &[ChannelInfo],
    objects: Arc<Vec<GeoJsonObjectFeature>>,
    target_indices: &[usize],
    local_to_world_offset: egui::Vec2,
    level: usize,
    request_id: u64,
    tx: &crossbeam_channel::Sender<AnalysisBatchEvent>,
    cancel: &AtomicBool,
    scope_label: String,
) -> anyhow::Result<AnalysisBatchTable> {
    let level_info = dataset
        .levels
        .get(level)
        .with_context(|| format!("dataset has no measurement level {level}"))?;
    let zarr_path = format!("/{}", level_info.path.trim_start_matches('/'));
    let array: Array<dyn ReadableStorageTraits> = Array::open(store, &zarr_path)
        .with_context(|| format!("open measurement array {zarr_path}"))?;
    let channel_schema = channels
        .iter()
        .map(|channel| AnalysisChannelSchema {
            channel_index: channel.index,
            channel_name: channel.name.clone(),
        })
        .collect::<Vec<_>>();
    let shape = &level_info.shape;
    let y_dim = dataset.dims.y;
    let x_dim = dataset.dims.x;
    let c_dim = dataset.dims.c;
    if y_dim >= shape.len() || x_dim >= shape.len() {
        anyhow::bail!("dataset dimensions do not match analysis level shape");
    }
    let y_chunk = *level_info
        .chunks
        .get(y_dim)
        .context("analysis level is missing Y chunk metadata")?;
    let x_chunk = *level_info
        .chunks
        .get(x_dim)
        .context("analysis level is missing X chunk metadata")?;
    if y_chunk == 0 || x_chunk == 0 {
        anyhow::bail!("analysis level has invalid zero-sized chunks");
    }

    let mut prepared_objects = Vec::with_capacity(target_indices.len());
    let mut failed_count = 0usize;
    let total_targets = target_indices.len();
    for (i, idx) in target_indices.iter().copied().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let Some(object) = objects.get(idx) else {
            failed_count += 1;
            continue;
        };
        if let Some(prepared) = prepare_analysis_object(
            object,
            idx,
            local_to_world_offset,
            level_info.downsample,
            shape,
            y_dim,
            x_dim,
        ) {
            prepared_objects.push(prepared);
        } else {
            failed_count += 1;
        }
        let completed = i + 1;
        if completed == 1 || completed == total_targets || completed % 256 == 0 {
            let _ = tx.send(AnalysisBatchEvent::Progress {
                request_id,
                phase: AnalysisProgressPhase::PreparingObjects,
                completed,
                total: total_targets,
            });
        }
    }

    let mut chunk_map: HashMap<(u64, u64), Vec<usize>> = HashMap::new();
    let total_prepared = prepared_objects.len();
    for (prepared_idx, object) in prepared_objects.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let [x0, y0, x1, y1] = object.bbox_level_px;
        let tile_x0 = x0 / x_chunk;
        let tile_x1 = (x1 - 1) / x_chunk;
        let tile_y0 = y0 / y_chunk;
        let tile_y1 = (y1 - 1) / y_chunk;
        for tile_y in tile_y0..=tile_y1 {
            for tile_x in tile_x0..=tile_x1 {
                chunk_map
                    .entry((tile_y, tile_x))
                    .or_default()
                    .push(prepared_idx);
            }
        }
        let completed = prepared_idx + 1;
        if completed == 1 || completed == total_prepared || completed % 256 == 0 {
            let _ = tx.send(AnalysisBatchEvent::Progress {
                request_id,
                phase: AnalysisProgressPhase::BuildingChunkIndex,
                completed,
                total: total_prepared,
            });
        }
    }

    let mut chunk_jobs = chunk_map
        .into_iter()
        .map(|((tile_y, tile_x), prepared_indices)| AnalysisChunkJob {
            tile_y,
            tile_x,
            prepared_indices,
        })
        .collect::<Vec<_>>();
    chunk_jobs.sort_unstable_by_key(|job| (job.tile_y, job.tile_x));

    let total_chunks = chunk_jobs.len();
    let _ = tx.send(AnalysisBatchEvent::Progress {
        request_id,
        phase: AnalysisProgressPhase::MeasuringChunks,
        completed: 0,
        total: total_chunks,
    });

    let mut accumulators = prepared_objects
        .iter()
        .map(|object| AnalysisAccumulator {
            object_index: object.object_index,
            object_id: object.object_id.clone(),
            pixel_count: 0,
            max_values: vec![0; channel_schema.len()],
            integrated_values: vec![0; channel_schema.len()],
        })
        .collect::<Vec<_>>();

    for (job_idx, job) in chunk_jobs.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }

        let y0 = job.tile_y * y_chunk;
        let x0 = job.tile_x * x_chunk;
        let y1 = (y0 + y_chunk).min(shape[y_dim]);
        let x1 = (x0 + x_chunk).min(shape[x_dim]);
        let chunk_height = (y1 - y0) as usize;
        let chunk_width = (x1 - x0) as usize;

        let mut planes = Vec::with_capacity(channel_schema.len());
        for channel in &channel_schema {
            let mut ranges: Vec<std::ops::Range<u64>> = Vec::with_capacity(shape.len());
            for dim in 0..shape.len() {
                if Some(dim) == c_dim {
                    let ch = channel.channel_index as u64;
                    ranges.push(ch..(ch + 1));
                } else if dim == y_dim {
                    ranges.push(y0..y1);
                } else if dim == x_dim {
                    ranges.push(x0..x1);
                } else {
                    ranges.push(0..1);
                }
            }
            let subset = ArraySubset::new_with_ranges(&ranges);
            let data = retrieve_image_subset_u16(&array, &subset, &level_info.dtype)
                .with_context(|| format!("read channel {} chunk", channel.channel_index))?;
            planes.push(plane_from_channel_data(data, c_dim)?);
        }

        for &prepared_idx in &job.prepared_indices {
            let object = &prepared_objects[prepared_idx];
            let [obj_x0, obj_y0, obj_x1, obj_y1] = object.bbox_level_px;
            let ix0 = obj_x0.max(x0);
            let iy0 = obj_y0.max(y0);
            let ix1 = obj_x1.min(x1);
            let iy1 = obj_y1.min(y1);
            if ix1 <= ix0 || iy1 <= iy0 {
                continue;
            }

            let width = (ix1 - ix0) as usize;
            let height = (iy1 - iy0) as usize;
            let mask = build_polygon_mask(&object.polygons_level, ix0, iy0, width, height);
            let pixel_count = mask.iter().copied().filter(|inside| *inside).count();
            if pixel_count == 0 {
                continue;
            }

            let x_off = (ix0 - x0) as usize;
            let y_off = (iy0 - y0) as usize;
            let accumulator = &mut accumulators[prepared_idx];
            accumulator.pixel_count = accumulator.pixel_count.saturating_add(pixel_count);

            for (channel_i, plane) in planes.iter().enumerate() {
                let mut integrated = 0u64;
                let mut max_value = accumulator.max_values[channel_i];
                for yy in 0..height {
                    let plane_y = y_off + yy;
                    for xx in 0..width {
                        if !mask[yy * width + xx] {
                            continue;
                        }
                        let plane_x = x_off + xx;
                        if plane_y >= chunk_height || plane_x >= chunk_width {
                            continue;
                        }
                        let value = plane[(plane_y, plane_x)];
                        integrated = integrated.saturating_add(value as u64);
                        if value > max_value {
                            max_value = value;
                        }
                    }
                }
                accumulator.integrated_values[channel_i] =
                    accumulator.integrated_values[channel_i].saturating_add(integrated);
                accumulator.max_values[channel_i] = max_value;
            }
        }

        let completed = job_idx + 1;
        if completed == 1 || completed == total_chunks || completed % 8 == 0 {
            let _ = tx.send(AnalysisBatchEvent::Progress {
                request_id,
                phase: AnalysisProgressPhase::MeasuringChunks,
                completed,
                total: total_chunks,
            });
        }
    }

    let mut rows = Vec::with_capacity(accumulators.len());
    for acc in accumulators {
        if acc.pixel_count == 0 {
            failed_count += 1;
            continue;
        }
        let mean_values = acc
            .integrated_values
            .iter()
            .map(|integrated| *integrated as f32 / acc.pixel_count as f32)
            .collect::<Vec<_>>();
        rows.push(AnalysisMeasurementRow {
            object_index: acc.object_index,
            object_id: acc.object_id,
            pixel_count: acc.pixel_count,
            mean_values,
            max_values: acc.max_values,
            integrated_values: acc.integrated_values,
        });
    }

    Ok(AnalysisBatchTable {
        scope_label,
        level_index: level_info.index,
        level_downsample: level_info.downsample,
        channels: channel_schema,
        rows,
        failed_count,
    })
}

fn measure_objects_with_label_mask_in_thread(
    dataset: &OmeZarrDataset,
    store: Arc<dyn ReadableStorageTraits>,
    channels: &[ChannelInfo],
    objects: Arc<Vec<GeoJsonObjectFeature>>,
    target_indices: &[usize],
    label_mask: Arc<ExternalLabelMask>,
    level: usize,
    request_id: u64,
    tx: &crossbeam_channel::Sender<AnalysisBatchEvent>,
    cancel: &AtomicBool,
    scope_label: String,
) -> anyhow::Result<AnalysisBatchTable> {
    let level_info = dataset
        .levels
        .get(level)
        .with_context(|| format!("dataset has no measurement level {level}"))?;
    let level0 = dataset
        .levels
        .first()
        .context("dataset has no level 0 for label-mask measurement")?;
    let zarr_path = format!("/{}", level_info.path.trim_start_matches('/'));
    let array: Array<dyn ReadableStorageTraits> = Array::open(store, &zarr_path)
        .with_context(|| format!("open measurement array {zarr_path}"))?;
    let channel_schema = channels
        .iter()
        .map(|channel| AnalysisChannelSchema {
            channel_index: channel.index,
            channel_name: channel.name.clone(),
        })
        .collect::<Vec<_>>();
    let shape = &level_info.shape;
    let y_dim = dataset.dims.y;
    let x_dim = dataset.dims.x;
    let c_dim = dataset.dims.c;
    if y_dim >= shape.len() || x_dim >= shape.len() {
        anyhow::bail!("dataset dimensions do not match analysis level shape");
    }
    let level0_w = level0.shape.get(x_dim).copied().unwrap_or(0) as usize;
    let level0_h = level0.shape.get(y_dim).copied().unwrap_or(0) as usize;
    if label_mask.width != level0_w || label_mask.height != level0_h {
        anyhow::bail!(
            "label mask dimensions {}x{} do not match image level-0 dimensions {}x{}",
            label_mask.width,
            label_mask.height,
            level0_w,
            level0_h
        );
    }
    let y_chunk = *level_info
        .chunks
        .get(y_dim)
        .context("analysis level is missing Y chunk metadata")?;
    let x_chunk = *level_info
        .chunks
        .get(x_dim)
        .context("analysis level is missing X chunk metadata")?;
    if y_chunk == 0 || x_chunk == 0 {
        anyhow::bail!("analysis level has invalid zero-sized chunks");
    }

    let mut failed_count = 0usize;
    let mut label_to_acc = HashMap::<u32, usize>::new();
    let mut accumulators = Vec::<AnalysisAccumulator>::with_capacity(target_indices.len());
    let total_targets = target_indices.len();
    for (i, idx) in target_indices.iter().copied().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let Some(object) = objects.get(idx) else {
            failed_count += 1;
            continue;
        };
        let Some(label_id) = analysis_object_label_id(object) else {
            failed_count += 1;
            continue;
        };
        match label_to_acc.entry(label_id) {
            std::collections::hash_map::Entry::Occupied(_) => {
                failed_count += 1;
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                let acc_idx = accumulators.len();
                entry.insert(acc_idx);
                accumulators.push(AnalysisAccumulator {
                    object_index: idx,
                    object_id: object.id.clone(),
                    pixel_count: 0,
                    max_values: vec![0; channel_schema.len()],
                    integrated_values: vec![0; channel_schema.len()],
                });
            }
        }
        let completed = i + 1;
        if completed == 1 || completed == total_targets || completed % 256 == 0 {
            let _ = tx.send(AnalysisBatchEvent::Progress {
                request_id,
                phase: AnalysisProgressPhase::PreparingLabelTargets,
                completed,
                total: total_targets,
            });
        }
    }

    let tiles_y = shape[y_dim].div_ceil(y_chunk);
    let tiles_x = shape[x_dim].div_ceil(x_chunk);
    let total_chunks = (tiles_y as usize).saturating_mul(tiles_x as usize);
    let _ = tx.send(AnalysisBatchEvent::Progress {
        request_id,
        phase: AnalysisProgressPhase::MeasuringChunks,
        completed: 0,
        total: total_chunks,
    });

    let ds = level_info.downsample.max(1e-6);
    let mut completed_chunks = 0usize;
    for tile_y in 0..tiles_y {
        for tile_x in 0..tiles_x {
            if cancel.load(Ordering::Relaxed) {
                break;
            }

            let y0 = tile_y * y_chunk;
            let x0 = tile_x * x_chunk;
            let y1 = (y0 + y_chunk).min(shape[y_dim]);
            let x1 = (x0 + x_chunk).min(shape[x_dim]);
            let chunk_height = (y1 - y0) as usize;
            let chunk_width = (x1 - x0) as usize;
            if chunk_width == 0 || chunk_height == 0 {
                completed_chunks += 1;
                continue;
            }

            let mut label_lookup = vec![None; chunk_width.saturating_mul(chunk_height)];
            for yy in 0..chunk_height {
                let src_y = ((((y0 as f32 + yy as f32) + 0.5) * ds).floor() as usize)
                    .min(label_mask.height.saturating_sub(1));
                let src_row = src_y.saturating_mul(label_mask.width);
                let dst_row = yy.saturating_mul(chunk_width);
                for xx in 0..chunk_width {
                    let src_x = ((((x0 as f32 + xx as f32) + 0.5) * ds).floor() as usize)
                        .min(label_mask.width.saturating_sub(1));
                    let label_id = label_mask.labels[src_row + src_x];
                    if label_id == 0 {
                        continue;
                    }
                    label_lookup[dst_row + xx] = label_to_acc.get(&label_id).copied();
                }
            }

            for (channel_i, channel) in channel_schema.iter().enumerate() {
                let mut ranges: Vec<std::ops::Range<u64>> = Vec::with_capacity(shape.len());
                for dim in 0..shape.len() {
                    if Some(dim) == c_dim {
                        let ch = channel.channel_index as u64;
                        ranges.push(ch..(ch + 1));
                    } else if dim == y_dim {
                        ranges.push(y0..y1);
                    } else if dim == x_dim {
                        ranges.push(x0..x1);
                    } else {
                        ranges.push(0..1);
                    }
                }
                let subset = ArraySubset::new_with_ranges(&ranges);
                let data = retrieve_image_subset_u16(&array, &subset, &level_info.dtype)
                    .with_context(|| format!("read channel {} chunk", channel.channel_index))?;
                let plane = plane_from_channel_data(data, c_dim)?;

                for yy in 0..chunk_height {
                    for xx in 0..chunk_width {
                        let idx = yy * chunk_width + xx;
                        let Some(acc_idx) = label_lookup[idx] else {
                            continue;
                        };
                        let value = plane[(yy, xx)];
                        let acc = &mut accumulators[acc_idx];
                        if channel_i == 0 {
                            acc.pixel_count = acc.pixel_count.saturating_add(1);
                        }
                        acc.integrated_values[channel_i] =
                            acc.integrated_values[channel_i].saturating_add(value as u64);
                        if value > acc.max_values[channel_i] {
                            acc.max_values[channel_i] = value;
                        }
                    }
                }
            }

            completed_chunks += 1;
            if completed_chunks == 1
                || completed_chunks == total_chunks
                || completed_chunks % 8 == 0
            {
                let _ = tx.send(AnalysisBatchEvent::Progress {
                    request_id,
                    phase: AnalysisProgressPhase::MeasuringChunks,
                    completed: completed_chunks,
                    total: total_chunks,
                });
            }
        }
        if cancel.load(Ordering::Relaxed) {
            break;
        }
    }

    let mut rows = Vec::with_capacity(accumulators.len());
    for acc in accumulators {
        if acc.pixel_count == 0 {
            failed_count += 1;
            continue;
        }
        let mean_values = acc
            .integrated_values
            .iter()
            .map(|integrated| *integrated as f32 / acc.pixel_count as f32)
            .collect::<Vec<_>>();
        rows.push(AnalysisMeasurementRow {
            object_index: acc.object_index,
            object_id: acc.object_id,
            pixel_count: acc.pixel_count,
            mean_values,
            max_values: acc.max_values,
            integrated_values: acc.integrated_values,
        });
    }

    Ok(AnalysisBatchTable {
        scope_label,
        level_index: level_info.index,
        level_downsample: level_info.downsample,
        channels: channel_schema,
        rows,
        failed_count,
    })
}

fn prepare_analysis_object(
    object: &GeoJsonObjectFeature,
    object_index: usize,
    local_to_world_offset: egui::Vec2,
    level_downsample: f32,
    shape: &[u64],
    y_dim: usize,
    x_dim: usize,
) -> Option<PreparedAnalysisObject> {
    let level_downsample = level_downsample.max(1e-6);
    let polygons_level = object
        .polygons_world
        .iter()
        .map(|poly| {
            poly.iter()
                .copied()
                .map(|p| {
                    let world = p + local_to_world_offset;
                    egui::pos2(world.x / level_downsample, world.y / level_downsample)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let bbox_world = object.bbox_world.translate(local_to_world_offset);
    let x0 = (bbox_world.min.x / level_downsample).floor().max(0.0) as u64;
    let y0 = (bbox_world.min.y / level_downsample).floor().max(0.0) as u64;
    let x1 = (bbox_world.max.x / level_downsample).ceil().max(0.0) as u64;
    let y1 = (bbox_world.max.y / level_downsample).ceil().max(0.0) as u64;
    let x1 = x1.min(shape.get(x_dim).copied().unwrap_or(0));
    let y1 = y1.min(shape.get(y_dim).copied().unwrap_or(0));
    if x1 <= x0 || y1 <= y0 {
        return None;
    }

    Some(PreparedAnalysisObject {
        object_index,
        object_id: object.id.clone(),
        polygons_level,
        bbox_level_px: [x0, y0, x1, y1],
    })
}

fn analysis_object_label_id(object: &GeoJsonObjectFeature) -> Option<u32> {
    object
        .properties
        .get("label")
        .and_then(json_value_to_u32)
        .or_else(|| object.properties.get("id").and_then(json_value_to_u32))
        .or_else(|| object.id.parse::<u32>().ok())
}

fn json_value_to_u32(value: &serde_json::Value) -> Option<u32> {
    match value {
        serde_json::Value::Number(n) => {
            n.as_u64().and_then(|v| u32::try_from(v).ok()).or_else(|| {
                n.as_i64()
                    .filter(|v| *v >= 0)
                    .and_then(|v| u32::try_from(v as u64).ok())
            })
        }
        serde_json::Value::String(s) => s.parse::<u32>().ok(),
        _ => None,
    }
}

fn load_external_label_mask(
    path: &Path,
    dataset: &OmeZarrDataset,
) -> anyhow::Result<ExternalLabelMask> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    let mask = match ext.as_str() {
        "tif" | "tiff" => load_external_label_mask_tiff(path)?,
        "npy" => load_external_label_mask_npy(path)?,
        _ => anyhow::bail!("unsupported label-mask format: {}", path.display()),
    };

    let level0 = dataset
        .levels
        .first()
        .context("dataset has no level 0 for label-mask validation")?;
    let y_dim = dataset.dims.y;
    let x_dim = dataset.dims.x;
    let expected_w = level0.shape.get(x_dim).copied().unwrap_or(0) as usize;
    let expected_h = level0.shape.get(y_dim).copied().unwrap_or(0) as usize;
    if mask.width != expected_w || mask.height != expected_h {
        anyhow::bail!(
            "label mask dimensions {}x{} do not match image level-0 dimensions {}x{}",
            mask.width,
            mask.height,
            expected_w,
            expected_h
        );
    }

    Ok(mask)
}

fn load_external_label_mask_npy(path: &Path) -> anyhow::Result<ExternalLabelMask> {
    let file =
        File::open(path).with_context(|| format!("open NPY label mask: {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut magic = [0u8; 6];
    reader.read_exact(&mut magic).context("read NPY magic")?;
    if magic != *b"\x93NUMPY" {
        anyhow::bail!("invalid `.npy` file header");
    }

    let mut version = [0u8; 2];
    reader
        .read_exact(&mut version)
        .context("read NPY version")?;
    let header_len = match version[0] {
        1 => {
            let mut buf = [0u8; 2];
            reader
                .read_exact(&mut buf)
                .context("read NPY v1 header length")?;
            u16::from_le_bytes(buf) as usize
        }
        2 | 3 => {
            let mut buf = [0u8; 4];
            reader
                .read_exact(&mut buf)
                .context("read NPY v2/v3 header length")?;
            u32::from_le_bytes(buf) as usize
        }
        other => anyhow::bail!("unsupported `.npy` version: {other}"),
    };

    let mut header_bytes = vec![0u8; header_len];
    reader
        .read_exact(&mut header_bytes)
        .context("read NPY header payload")?;
    let header = String::from_utf8(header_bytes).context("NPY header is not valid UTF-8")?;
    let descr = npy_header_field(&header, "descr").context("NPY header missing `descr`")?;
    let fortran =
        npy_header_field(&header, "fortran_order").context("NPY header missing `fortran_order`")?;
    if fortran.trim() != "False" {
        anyhow::bail!("Fortran-order `.npy` label masks are not supported");
    }
    let (height, width) = parse_npy_shape_2d(&header)?;

    let elem_count = width.saturating_mul(height);
    let labels = match descr.trim_matches('\'').trim_matches('"') {
        "|u1" => {
            let mut buf = vec![0u8; elem_count];
            reader
                .read_exact(&mut buf)
                .context("read `.npy` u8 label data")?;
            buf.into_iter().map(|v| v as u32).collect::<Vec<_>>()
        }
        "<u2" => read_npy_u16_labels(&mut reader, elem_count)?,
        "<u4" => read_npy_u32_labels(&mut reader, elem_count)?,
        "<i1" => convert_signed_labels(read_npy_i8_labels(&mut reader, elem_count)?.into_iter())?,
        "<i2" => convert_signed_labels(read_npy_i16_labels(&mut reader, elem_count)?.into_iter())?,
        "<i4" => convert_signed_labels(read_npy_i32_labels(&mut reader, elem_count)?.into_iter())?,
        "<i8" => convert_signed_labels(read_npy_i64_labels(&mut reader, elem_count)?.into_iter())?,
        other => anyhow::bail!("unsupported `.npy` label dtype `{other}`"),
    };

    external_label_mask_from_u32(path, width, height, labels)
}

fn load_external_label_mask_tiff(path: &Path) -> anyhow::Result<ExternalLabelMask> {
    let file =
        File::open(path).with_context(|| format!("open TIFF label mask: {}", path.display()))?;
    let mut decoder = Decoder::new(BufReader::new(file)).context("create TIFF decoder")?;
    let (width, height) = decoder.dimensions().context("read TIFF dimensions")?;
    let decoded = decoder.read_image().context("decode TIFF label mask")?;
    let labels = match decoded {
        DecodingResult::U8(v) => v.into_iter().map(|v| v as u32).collect::<Vec<_>>(),
        DecodingResult::U16(v) => v.into_iter().map(|v| v as u32).collect::<Vec<_>>(),
        DecodingResult::U32(v) => v,
        DecodingResult::I8(v) => convert_signed_labels(v.into_iter().map(|v| v as i64))?,
        DecodingResult::I16(v) => convert_signed_labels(v.into_iter().map(|v| v as i64))?,
        DecodingResult::I32(v) => convert_signed_labels(v.into_iter().map(|v| v as i64))?,
        DecodingResult::U64(v) => v
            .into_iter()
            .map(|v| u32::try_from(v).map_err(|_| anyhow!("TIFF label value exceeds u32 range")))
            .collect::<anyhow::Result<Vec<_>>>()?,
        other => anyhow::bail!("unsupported TIFF label-mask dtype: {:?}", other),
    };
    external_label_mask_from_u32(path, width as usize, height as usize, labels)
}

fn npy_header_field<'a>(header: &'a str, field: &str) -> Option<&'a str> {
    let needle = format!("'{field}':");
    let start = header.find(&needle)? + needle.len();
    let tail = header.get(start..)?.trim_start();
    let end = tail.find(',').unwrap_or(tail.len());
    Some(tail[..end].trim())
}

fn parse_npy_shape_2d(header: &str) -> anyhow::Result<(usize, usize)> {
    let needle = "'shape':";
    let start = header
        .find(needle)
        .map(|idx| idx + needle.len())
        .context("NPY header missing `shape`")?;
    let tail = header.get(start..).unwrap_or("").trim_start();
    let open = tail.find('(').context("NPY shape is missing `(`")?;
    let close = tail.find(')').context("NPY shape is missing `)`")?;
    let inner = &tail[(open + 1)..close];
    let dims = inner
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .with_context(|| format!("invalid NPY shape dimension `{s}`"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    if dims.len() != 2 {
        anyhow::bail!("expected a 2D `.npy` label mask, found shape {:?}", dims);
    }
    Ok((dims[0], dims[1]))
}

fn read_npy_u16_labels(reader: &mut dyn Read, elem_count: usize) -> anyhow::Result<Vec<u32>> {
    let mut raw = vec![0u8; elem_count.saturating_mul(2)];
    reader
        .read_exact(&mut raw)
        .context("read `.npy` u16 label data")?;
    Ok(raw
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]) as u32)
        .collect())
}

fn read_npy_u32_labels(reader: &mut dyn Read, elem_count: usize) -> anyhow::Result<Vec<u32>> {
    let mut raw = vec![0u8; elem_count.saturating_mul(4)];
    reader
        .read_exact(&mut raw)
        .context("read `.npy` u32 label data")?;
    Ok(raw
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn read_npy_i8_labels(reader: &mut dyn Read, elem_count: usize) -> anyhow::Result<Vec<i64>> {
    let mut raw = vec![0u8; elem_count];
    reader
        .read_exact(&mut raw)
        .context("read `.npy` i8 label data")?;
    Ok(raw.into_iter().map(|v| (v as i8) as i64).collect())
}

fn read_npy_i16_labels(reader: &mut dyn Read, elem_count: usize) -> anyhow::Result<Vec<i64>> {
    let mut raw = vec![0u8; elem_count.saturating_mul(2)];
    reader
        .read_exact(&mut raw)
        .context("read `.npy` i16 label data")?;
    Ok(raw
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as i64)
        .collect())
}

fn read_npy_i32_labels(reader: &mut dyn Read, elem_count: usize) -> anyhow::Result<Vec<i64>> {
    let mut raw = vec![0u8; elem_count.saturating_mul(4)];
    reader
        .read_exact(&mut raw)
        .context("read `.npy` i32 label data")?;
    Ok(raw
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as i64)
        .collect())
}

fn read_npy_i64_labels(reader: &mut dyn Read, elem_count: usize) -> anyhow::Result<Vec<i64>> {
    let mut raw = vec![0u8; elem_count.saturating_mul(8)];
    reader
        .read_exact(&mut raw)
        .context("read `.npy` i64 label data")?;
    Ok(raw
        .chunks_exact(8)
        .map(|chunk| {
            i64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect())
}

fn convert_signed_labels<I>(values: I) -> anyhow::Result<Vec<u32>>
where
    I: IntoIterator,
    I::Item: Into<i64>,
{
    values
        .into_iter()
        .map(|value| {
            let value = value.into();
            if value < 0 {
                Err(anyhow!("label masks cannot contain negative IDs"))
            } else {
                u32::try_from(value as u64).map_err(|_| anyhow!("label value exceeds u32 range"))
            }
        })
        .collect()
}

fn external_label_mask_from_u32(
    path: &Path,
    width: usize,
    height: usize,
    labels: Vec<u32>,
) -> anyhow::Result<ExternalLabelMask> {
    if labels.len() != width.saturating_mul(height) {
        anyhow::bail!(
            "label mask pixel count {} does not match dimensions {}x{}",
            labels.len(),
            width,
            height
        );
    }
    Ok(ExternalLabelMask {
        path: path.to_path_buf(),
        width,
        height,
        labels: Arc::new(labels),
    })
}

fn measure_object_row_from_array(
    dataset: &OmeZarrDataset,
    array: &Array<dyn ReadableStorageTraits>,
    channels: &[AnalysisChannelSchema],
    object_index: usize,
    object: &GeoJsonObjectFeature,
    local_to_world_offset: egui::Vec2,
) -> anyhow::Result<AnalysisMeasurementRow> {
    let level0 = dataset
        .levels
        .first()
        .context("dataset has no level 0 for measurement")?;
    let shape = &level0.shape;
    let y_dim = dataset.dims.y;
    let x_dim = dataset.dims.x;
    let c_dim = dataset.dims.c;
    if y_dim >= shape.len() || x_dim >= shape.len() {
        anyhow::bail!("dataset dimensions do not match level-0 shape");
    }

    let polygons_world = object
        .polygons_world
        .iter()
        .map(|poly| {
            poly.iter()
                .copied()
                .map(|p| p + local_to_world_offset)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let bbox_world = object.bbox_world.translate(local_to_world_offset);
    let x0 = bbox_world.min.x.floor().max(0.0) as u64;
    let y0 = bbox_world.min.y.floor().max(0.0) as u64;
    let x1 = bbox_world.max.x.ceil().max(0.0) as u64;
    let y1 = bbox_world.max.y.ceil().max(0.0) as u64;
    let x1 = x1.min(shape[x_dim]);
    let y1 = y1.min(shape[y_dim]);
    if x1 <= x0 || y1 <= y0 {
        anyhow::bail!("selected object is outside the image bounds");
    }

    let width = (x1 - x0) as usize;
    let height = (y1 - y0) as usize;
    let mask = build_polygon_mask(&polygons_world, x0, y0, width, height);
    let pixel_count = mask.iter().copied().filter(|inside| *inside).count();
    if pixel_count == 0 {
        anyhow::bail!("selected object contains no image pixels");
    }

    let mut mean_values = Vec::with_capacity(channels.len());
    let mut max_values = Vec::with_capacity(channels.len());
    let mut integrated_values = Vec::with_capacity(channels.len());
    for channel in channels {
        let mut ranges: Vec<std::ops::Range<u64>> = Vec::with_capacity(shape.len());
        for dim in 0..shape.len() {
            if Some(dim) == c_dim {
                let ch = channel.channel_index as u64;
                ranges.push(ch..(ch + 1));
            } else if dim == y_dim {
                ranges.push(y0..y1);
            } else if dim == x_dim {
                ranges.push(x0..x1);
            } else {
                ranges.push(0..1);
            }
        }
        let subset = ArraySubset::new_with_ranges(&ranges);
        let data = retrieve_image_subset_u16(array, &subset, &level0.dtype)
            .with_context(|| format!("read channel {} subset", channel.channel_index))?;
        let plane = plane_from_channel_data(data, c_dim)?;

        let mut integrated = 0u64;
        let mut max_value = 0u16;
        for (inside, value) in mask.iter().zip(plane.iter()) {
            if !inside {
                continue;
            }
            integrated = integrated.saturating_add(*value as u64);
            if *value > max_value {
                max_value = *value;
            }
        }
        mean_values.push(integrated as f32 / pixel_count as f32);
        max_values.push(max_value);
        integrated_values.push(integrated);
    }

    Ok(AnalysisMeasurementRow {
        object_index,
        object_id: object.id.clone(),
        pixel_count,
        mean_values,
        max_values,
        integrated_values,
    })
}

fn batch_row_to_object_measurement_set(
    row: AnalysisMeasurementRow,
    channels: &[AnalysisChannelSchema],
) -> ObjectMeasurementSet {
    let mut channel_rows = Vec::with_capacity(channels.len());
    for (i, channel) in channels.iter().enumerate() {
        channel_rows.push(ObjectChannelMeasurement {
            channel_index: channel.channel_index,
            channel_name: channel.channel_name.clone(),
            mean: row.mean_values.get(i).copied().unwrap_or(0.0),
            max: row.max_values.get(i).copied().unwrap_or(0),
            integrated: row.integrated_values.get(i).copied().unwrap_or(0),
        });
    }
    ObjectMeasurementSet {
        object_index: row.object_index,
        object_id: row.object_id,
        pixel_count: row.pixel_count,
        channels: channel_rows,
    }
}

fn plane_from_channel_data(
    data: ndarray::ArrayD<u16>,
    c_dim: Option<usize>,
) -> anyhow::Result<ndarray::Array2<u16>> {
    if c_dim.is_some() {
        data.into_dimensionality::<ndarray::Ix3>()
            .ok()
            .map(|a| a.index_axis(ndarray::Axis(0), 0).to_owned())
    } else {
        data.into_dimensionality::<ndarray::Ix2>().ok()
    }
    .context("unexpected array dimensionality for object measurements")
}

pub(super) fn build_polygon_mask(
    polygons_world: &[Vec<egui::Pos2>],
    x0: u64,
    y0: u64,
    width: usize,
    height: usize,
) -> Vec<bool> {
    let mut mask = vec![false; width.saturating_mul(height)];
    for yy in 0..height {
        for xx in 0..width {
            let world = egui::pos2(x0 as f32 + xx as f32 + 0.5, y0 as f32 + yy as f32 + 0.5);
            mask[yy * width + xx] = point_in_any_polygon(world, polygons_world);
        }
    }
    mask
}

fn point_in_any_polygon(p: egui::Pos2, polygons: &[Vec<egui::Pos2>]) -> bool {
    polygons.iter().any(|poly| point_in_polygon(p, poly))
}

fn point_in_polygon(p: egui::Pos2, poly: &[egui::Pos2]) -> bool {
    if poly.len() < 4 {
        return false;
    }
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let pi = poly[i];
        let pj = poly[j];
        let dy = pj.y - pi.y;
        let intersects = ((pi.y > p.y) != (pj.y > p.y))
            && dy.abs() > 1e-12
            && (p.x < (pj.x - pi.x) * (p.y - pi.y) / dy + pi.x);
        if intersects {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn analysis_row_metric_value(
    row: &AnalysisMeasurementRow,
    metric: AnalysisMetric,
    channel_pos: usize,
) -> f32 {
    match metric {
        AnalysisMetric::Mean => row.mean_values.get(channel_pos).copied().unwrap_or(0.0),
        AnalysisMetric::Max => row.max_values.get(channel_pos).copied().unwrap_or(0) as f32,
        AnalysisMetric::Integrated => {
            row.integrated_values.get(channel_pos).copied().unwrap_or(0) as f32
        }
    }
}

fn analysis_metric_values(
    table: &AnalysisBatchTable,
    metric: AnalysisMetric,
    channel_pos: usize,
) -> Vec<f32> {
    table
        .rows
        .iter()
        .map(|row| analysis_row_metric_value(row, metric, channel_pos))
        .filter(|v| v.is_finite())
        .collect()
}

fn analysis_hist_selected_indices(
    table: &AnalysisBatchTable,
    metric: AnalysisMetric,
    channel_pos: usize,
    brush: Option<(f32, f32)>,
) -> Vec<usize> {
    let Some((lo, hi)) = brush else {
        return Vec::new();
    };
    table
        .rows
        .iter()
        .filter_map(|row| {
            let value = analysis_row_metric_value(row, metric, channel_pos);
            (value.is_finite() && value >= lo && value <= hi).then_some(row.object_index)
        })
        .collect()
}

fn analysis_scatter_selected_indices(
    table: &AnalysisBatchTable,
    metric: AnalysisMetric,
    x_channel: usize,
    y_channel: usize,
    brush: Option<egui::Rect>,
) -> Vec<usize> {
    let Some(brush) = brush else {
        return Vec::new();
    };
    table
        .rows
        .iter()
        .filter_map(|row| {
            let x = analysis_row_metric_value(row, metric, x_channel);
            let y = analysis_row_metric_value(row, metric, y_channel);
            (x.is_finite() && y.is_finite() && brush.contains(egui::pos2(x, y)))
                .then_some(row.object_index)
        })
        .collect()
}

fn spatial_table_column_values(table: &SpatialTableAnalysis, column_pos: usize) -> Vec<f32> {
    let Some(column_name) = table.numeric_columns.get(column_pos) else {
        return Vec::new();
    };
    table
        .loaded_numeric_columns
        .get(column_name)
        .cloned()
        .unwrap_or_default()
}

fn spatial_hist_selected_indices(
    table: &SpatialTableAnalysis,
    column_pos: usize,
    brush: Option<(f32, f32)>,
) -> Vec<usize> {
    let Some((lo, hi)) = brush else {
        return Vec::new();
    };
    table
        .rows
        .iter()
        .zip(spatial_table_column_values(table, column_pos))
        .filter_map(|(row, value)| {
            (value.is_finite() && value >= lo && value <= hi).then_some(row.object_index)
        })
        .collect()
}

fn spatial_scatter_selected_indices(
    table: &SpatialTableAnalysis,
    x_column: usize,
    y_column: usize,
    brush: Option<egui::Rect>,
) -> Vec<usize> {
    let Some(brush) = brush else {
        return Vec::new();
    };
    table
        .rows
        .iter()
        .zip(spatial_table_column_values(table, x_column))
        .zip(spatial_table_column_values(table, y_column))
        .filter_map(|((row, x), y)| {
            (x.is_finite() && y.is_finite() && brush.contains(egui::pos2(x, y)))
                .then_some(row.object_index)
        })
        .collect()
}

fn json_value_short_string(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(v) => Some(v.clone()),
        serde_json::Value::Number(v) => Some(v.to_string()),
        serde_json::Value::Bool(v) => Some(v.to_string()),
        _ => None,
    }
}

pub(super) fn numeric_json_value(value: &serde_json::Value) -> Option<f32> {
    match value {
        serde_json::Value::Number(v) => v.as_f64().map(|v| v as f32),
        serde_json::Value::String(v) => v.parse::<f32>().ok(),
        _ => None,
    }
}

fn load_spatial_table_columns_for_indices(
    spatial_root: &Path,
    table: &mut SpatialTableAnalysis,
    column_indices: &[usize],
) -> anyhow::Result<()> {
    let mut requested = column_indices
        .iter()
        .copied()
        .filter_map(|idx| table.numeric_columns.get(idx).cloned())
        .collect::<Vec<_>>();
    requested.sort();
    requested.dedup();
    for column_name in requested {
        if table.loaded_numeric_columns.contains_key(&column_name) {
            continue;
        }
        let values = load_numeric_column_for_rows(spatial_root, table, &column_name)?;
        table.loaded_numeric_columns.insert(column_name, values);
    }
    Ok(())
}

fn normalize_analysis_name(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(|ch| ch.to_lowercase())
        .collect()
}

fn analysis_name_tokens(value: &str) -> HashSet<String> {
    let raw_tokens = value
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_ascii_lowercase())
        .collect::<Vec<_>>();

    let mut tokens = HashSet::new();
    for token in &raw_tokens {
        tokens.insert(token.clone());
    }
    for window in raw_tokens.windows(2) {
        let joined = format!("{}{}", window[0], window[1]);
        if joined.len() >= 3 {
            tokens.insert(joined);
        }
    }
    let collapsed = normalize_analysis_name(value);
    if collapsed.len() >= 3 {
        tokens.insert(collapsed);
    }
    tokens
}

fn analysis_token_frequencies(
    channels: &[ChannelInfo],
    numeric_columns: &[String],
) -> HashMap<String, usize> {
    let mut frequencies = HashMap::new();
    for token_set in channels
        .iter()
        .map(|channel| analysis_name_tokens(&channel.name))
        .chain(
            numeric_columns
                .iter()
                .map(|column| analysis_name_tokens(column)),
        )
    {
        for token in token_set {
            *frequencies.entry(token).or_insert(0) += 1;
        }
    }
    frequencies
}

fn fuzzy_name_score(query: &str, candidate: &str) -> Option<i32> {
    let query = query.trim().to_ascii_lowercase();
    if query.is_empty() {
        return Some(0);
    }
    let candidate_lower = candidate.to_ascii_lowercase();
    if candidate_lower == query {
        return Some(100_000);
    }
    if let Some(rest) = candidate_lower.strip_prefix(&query) {
        return Some(90_000 - rest.len() as i32);
    }
    if let Some(pos) = candidate_lower.find(&query) {
        return Some(80_000 - pos as i32);
    }

    let mut score = 50_000i32;
    let mut search_from = 0usize;
    for ch in query.chars() {
        let hay = &candidate_lower[search_from..];
        let rel = hay.find(ch)?;
        score -= rel as i32;
        search_from += rel + ch.len_utf8();
    }
    Some(score)
}

fn finite_min_max_f32(values: &[f32]) -> Option<(f32, f32)> {
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    let mut any = false;
    for &v in values {
        if !v.is_finite() {
            continue;
        }
        any = true;
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    any.then(|| {
        if (max_v - min_v).abs() <= 1e-12 {
            (min_v, min_v + 1.0)
        } else {
            (min_v, max_v)
        }
    })
}

pub(super) fn compute_histogram_f32(values: &[f32], bin_count: usize) -> SimpleHistogram {
    // Histogram statistics are computed over finite values only. The median is derived from the
    // sorted finite sample so UI annotations stay consistent with the plotted bins.
    let (min_v, max_v) = finite_min_max_f32(values).unwrap_or((0.0, 1.0));
    let mut sorted = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if sorted.is_empty() {
        0.0
    } else {
        sorted[sorted.len() / 2]
    };
    let bin_count = bin_count.max(8);
    let mut bins = vec![0u32; bin_count];
    let inv = (bin_count as f32 - 1.0) / (max_v - min_v).max(1e-6);
    for &v in &sorted {
        let idx = (((v - min_v) * inv).floor() as usize).min(bin_count - 1);
        bins[idx] = bins[idx].saturating_add(1);
    }
    let max_count = bins.iter().copied().max().unwrap_or(1).max(1);
    SimpleHistogram {
        min: min_v,
        max: max_v,
        median,
        bins,
        max_count,
    }
}

pub(super) fn quantile_threshold_levels(values: &[f32], level_count: usize) -> Vec<f32> {
    if values.len() < 2 || level_count < 2 {
        return Vec::new();
    }
    let mut sorted = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    if sorted.len() < 2 {
        return Vec::new();
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let bins = level_count.max(2);
    let mut out = Vec::with_capacity(bins.saturating_sub(1));
    for q in 1..bins {
        let pos = ((sorted.len() - 1) as f32 * (q as f32 / bins as f32)).round() as usize;
        let value = sorted[pos.min(sorted.len() - 1)];
        if out.last().copied() != Some(value) {
            out.push(value);
        }
    }
    out
}

pub(super) fn kmeans_threshold_levels(values: &[f32], k: usize, iterations: usize) -> Vec<f32> {
    // Use 1D k-means centroids to propose threshold boundaries halfway between neighboring
    // clusters. This is heuristic UI assistance, not a persisted analysis model.
    let mut samples = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    if samples.len() < 2 {
        return Vec::new();
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let k = k.clamp(2, samples.len().min(12));

    let mut centroids = (0..k)
        .map(|i| {
            let pos = ((samples.len() - 1) as f32 * (i as f32 / (k - 1) as f32)).round() as usize;
            samples[pos.min(samples.len() - 1)]
        })
        .collect::<Vec<_>>();
    let mut assignments = vec![0usize; samples.len()];

    for _ in 0..iterations.max(1) {
        let mut changed = false;
        for (i, value) in samples.iter().enumerate() {
            let mut best_idx = 0usize;
            let mut best_dist = f32::INFINITY;
            for (idx, centroid) in centroids.iter().enumerate() {
                let dist = (*value - *centroid).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = idx;
                }
            }
            if assignments[i] != best_idx {
                assignments[i] = best_idx;
                changed = true;
            }
        }

        let mut sums = vec![0.0f32; k];
        let mut counts = vec![0usize; k];
        for (value, &cluster) in samples.iter().zip(assignments.iter()) {
            sums[cluster] += *value;
            counts[cluster] += 1;
        }
        for i in 0..k {
            if counts[i] > 0 {
                centroids[i] = sums[i] / counts[i] as f32;
            }
        }
        if !changed {
            break;
        }
    }

    centroids.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = Vec::with_capacity(k.saturating_sub(1));
    for pair in centroids.windows(2) {
        let value = 0.5 * (pair[0] + pair[1]);
        if out.last().copied() != Some(value) {
            out.push(value);
        }
    }
    out
}

fn histogram_level_label(
    method: HistogramLevelMethod,
    level_count: usize,
    level_index: usize,
    total_levels: usize,
) -> String {
    match method {
        HistogramLevelMethod::Quantiles => {
            let denom = level_count.max(2) as f32;
            let pct = (((level_index + 1) as f32 / denom) * 100.0).round() as i32;
            format!("Quantile {pct}%")
        }
        HistogramLevelMethod::KMeans => {
            let left_cluster = (level_index + 1).min(total_levels.max(1));
            format!("K={} boundary {}", level_count.max(2), left_cluster)
        }
    }
}

pub(super) fn apply_histogram_value_transform(
    value: f32,
    transform: HistogramValueTransform,
) -> f32 {
    // The transform is applied symmetrically anywhere histogram values are compared or displayed
    // so brushing and threshold handles remain in the same value space as the plotted axis.
    match transform {
        HistogramValueTransform::None => value,
        HistogramValueTransform::Arcsinh => value.asinh(),
    }
}

fn invert_histogram_value_transform(value: f32, transform: HistogramValueTransform) -> f32 {
    match transform {
        HistogramValueTransform::None => value,
        HistogramValueTransform::Arcsinh => value.sinh(),
    }
}

pub(super) fn analysis_progress_status(
    phase: AnalysisProgressPhase,
    completed: usize,
    total: usize,
) -> String {
    match phase {
        AnalysisProgressPhase::PreparingObjects => {
            format!("Preparing objects: {completed} / {total} cells")
        }
        AnalysisProgressPhase::BuildingChunkIndex => {
            format!("Building chunk index: {completed} / {total} cells")
        }
        AnalysisProgressPhase::PreparingLabelTargets => {
            format!("Preparing label targets: {completed} / {total} cells")
        }
        AnalysisProgressPhase::MeasuringChunks => {
            format!("Measuring intensities: {completed} / {total} chunks")
        }
    }
}

fn analysis_picker_popup_width(
    ui: &egui::Ui,
    button_width: f32,
    names: &[String],
    search: &str,
    selected_label: Option<&str>,
) -> f32 {
    let viewport_width = ui.ctx().content_rect().width().max(button_width);
    let max_width = (viewport_width - 32.0).max(button_width);
    let mut candidates = names
        .iter()
        .filter_map(|name| fuzzy_name_score(search, name).map(|score| (score, name.as_str())))
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(b.1)));

    let font_id = egui::TextStyle::Button.resolve(ui.style());
    let mut target_width = button_width.max(240.0);
    for label in selected_label
        .into_iter()
        .chain(candidates.into_iter().take(12).map(|(_, name)| name))
    {
        let galley = ui.ctx().fonts_mut(|fonts| {
            fonts.layout_no_wrap(label.to_owned(), font_id.clone(), egui::Color32::WHITE)
        });
        target_width = target_width.max(galley.size().x + 72.0);
    }

    target_width.clamp(button_width.max(320.0), max_width)
}

fn order_pair(a: f32, b: f32) -> (f32, f32) {
    if a <= b { (a, b) } else { (b, a) }
}

fn scatter_axis_drag_speed(min_v: f32, max_v: f32) -> f64 {
    let span = (max_v - min_v).abs().max(1.0);
    (span / 200.0).max(0.01) as f64
}

fn normalize_scatter_axis_pair(min_v: f32, max_v: f32, fallback: (f32, f32)) -> (f32, f32) {
    let (fallback_min, fallback_max) = order_pair(fallback.0, fallback.1);
    if !(min_v.is_finite() && max_v.is_finite()) {
        return (fallback_min, fallback_max.max(fallback_min + 1.0));
    }

    let (mut lo, mut hi) = order_pair(min_v, max_v);
    if (hi - lo).abs() <= 1e-6 {
        let span = (fallback_max - fallback_min).abs().max(1.0);
        let pad = (span * 0.005).max(0.5);
        lo -= pad;
        hi += pad;
    }
    (lo, hi)
}

fn normalize_scatter_view_rect(view_rect: egui::Rect, fallback_rect: egui::Rect) -> egui::Rect {
    let (x_min, x_max) = normalize_scatter_axis_pair(
        view_rect.min.x,
        view_rect.max.x,
        (fallback_rect.min.x, fallback_rect.max.x),
    );
    let (y_min, y_max) = normalize_scatter_axis_pair(
        view_rect.min.y,
        view_rect.max.y,
        (fallback_rect.min.y, fallback_rect.max.y),
    );
    egui::Rect::from_min_max(egui::pos2(x_min, y_min), egui::pos2(x_max, y_max))
}

fn value_to_screen_x(value: f32, rect: egui::Rect, min_v: f32, max_v: f32) -> f32 {
    let t = ((value - min_v) / (max_v - min_v).max(1e-6)).clamp(0.0, 1.0);
    rect.left() + t * rect.width()
}

fn screen_x_to_value(x: f32, rect: egui::Rect, min_v: f32, max_v: f32) -> f32 {
    let t = ((x - rect.left()) / rect.width().max(1e-6)).clamp(0.0, 1.0);
    min_v + t * (max_v - min_v)
}

fn value_to_screen_y(value: f32, rect: egui::Rect, min_v: f32, max_v: f32) -> f32 {
    let t = ((value - min_v) / (max_v - min_v).max(1e-6)).clamp(0.0, 1.0);
    rect.bottom() - t * rect.height()
}

fn screen_y_to_value(y: f32, rect: egui::Rect, min_v: f32, max_v: f32) -> f32 {
    let t = ((rect.bottom() - y) / rect.height().max(1e-6)).clamp(0.0, 1.0);
    min_v + t * (max_v - min_v)
}

fn screen_rect_to_value_rect(
    screen_rect: egui::Rect,
    plot_rect: egui::Rect,
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
) -> Option<egui::Rect> {
    if !screen_rect.is_positive() {
        return None;
    }
    let min_x = screen_x_to_value(screen_rect.min.x, plot_rect, x_min, x_max);
    let max_x = screen_x_to_value(screen_rect.max.x, plot_rect, x_min, x_max);
    let min_y = screen_y_to_value(screen_rect.max.y, plot_rect, y_min, y_max);
    let max_y = screen_y_to_value(screen_rect.min.y, plot_rect, y_min, y_max);
    Some(egui::Rect::from_min_max(
        egui::pos2(min_x.min(max_x), min_y.min(max_y)),
        egui::pos2(min_x.max(max_x), min_y.max(max_y)),
    ))
}

fn value_rect_to_screen_rect(
    value_rect: egui::Rect,
    plot_rect: egui::Rect,
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
) -> egui::Rect {
    egui::Rect::from_min_max(
        egui::pos2(
            value_to_screen_x(value_rect.min.x, plot_rect, x_min, x_max),
            value_to_screen_y(value_rect.max.y, plot_rect, y_min, y_max),
        ),
        egui::pos2(
            value_to_screen_x(value_rect.max.x, plot_rect, x_min, x_max),
            value_to_screen_y(value_rect.min.y, plot_rect, y_min, y_max),
        ),
    )
}
