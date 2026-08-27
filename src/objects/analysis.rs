use super::core::object_proxy_position_world;
use super::*;

mod algorithms;
mod data_selection;

pub(in crate::objects) use algorithms::*;

// Object-layer analysis UI and helpers.
//
// This file owns the viewer-side analysis surface for object layers: histogram/scatter UI,
// threshold-set editing, batch measurement requests, spatial-table joins, and the caches that
// support interactive brushing/selection. The key theme is that expensive analysis products are
// lazily derived from the currently visible/filter-selected object set and invalidated whenever
// that set changes.

const LIVE_ANALYSIS_SELECTION_OBJECT_LIMIT: usize = 200_000;

impl ObjectsLayer {
    pub(crate) fn project_analysis_state(&self) -> ObjectProjectAnalysisState {
        ObjectProjectAnalysisState {
            threshold_set_name: self.analysis_threshold_set_name.clone(),
            threshold_elements: self.analysis_threshold_elements.clone(),
            threshold_selected_element: self.analysis_threshold_selected_element,
            follow_active_channel: self.analysis_follow_active_channel,
            live_threshold_channel_name: self.analysis_live_threshold_channel_name.clone(),
            channel_mapping_overrides: self.analysis_channel_mapping_overrides.clone(),
            selection_elements: self.selection_elements.clone(),
            selection_element_selected: self.selection_element_selected,
            show_selection_overlay: self.show_selection_overlay,
        }
    }

    pub(crate) fn apply_project_analysis_state(
        &mut self,
        state: &ObjectProjectAnalysisState,
        active_channel_name: Option<&str>,
    ) {
        self.analysis_threshold_set_name = if state.threshold_set_name.is_empty() {
            "Threshold Set".to_string()
        } else {
            state.threshold_set_name.clone()
        };
        self.analysis_threshold_elements = state.threshold_elements.clone();
        self.normalize_threshold_call_elements();
        self.analysis_follow_active_channel = state.follow_active_channel;
        self.analysis_channel_mapping_overrides = state.channel_mapping_overrides.clone();
        self.selection_elements = state.selection_elements.clone();
        self.show_selection_overlay = state.show_selection_overlay;
        self.selection_element_selected = state
            .selection_element_selected
            .filter(|idx| *idx < self.selection_elements.len());

        self.analysis_threshold_selected_element = state
            .threshold_selected_element
            .filter(|idx| *idx < self.analysis_threshold_elements.len());

        if let Some(idx) = self.analysis_threshold_selected_element {
            self.analysis_live_threshold_channel_name = state.live_threshold_channel_name.clone();
            self.load_threshold_element(idx, active_channel_name);
        } else {
            self.analysis_property_thresholds.clear();
            self.analysis_live_threshold_channel_name = if self.analysis_follow_active_channel {
                active_channel_name.map(ToOwned::to_owned)
            } else {
                state.live_threshold_channel_name.clone()
            };
            self.analysis_hist_drag_rule = None;
            self.analysis_hist_focus_object_index = None;
            self.analysis_hist_snapped_level = None;
            self.mark_live_analysis_selection_dirty();
        }
    }

    pub fn open_analysis_channel_mapping_popup(&mut self) {
        if !self.has_data() {
            return;
        }
        self.analysis_channel_mapping_popup_open = true;
    }

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
        self.sync_marker_call_selection_to_active_channel(channels, selected_channel);
        self.sync_histogram_to_active_channel(channels, selected_channel, &numeric_columns);
    }

    fn selected_threshold_call(&self) -> Option<&ThresholdSetElement> {
        self.analysis_threshold_selected_element
            .filter(|idx| *idx < self.analysis_threshold_elements.len())
            .and_then(|idx| self.analysis_threshold_elements.get(idx))
    }

    fn selected_threshold_call_marker_name(&self) -> Option<&str> {
        self.selected_threshold_call()
            .and_then(threshold_call_bound_channel_name)
    }

    fn effective_threshold_channel_name(
        &self,
        active_channel_name: Option<&str>,
    ) -> Option<String> {
        if let Some(bound) = self.selected_threshold_call_marker_name() {
            return Some(bound.to_string());
        }
        if self.analysis_follow_active_channel {
            active_channel_name.map(ToOwned::to_owned)
        } else {
            self.analysis_live_threshold_channel_name
                .clone()
                .or_else(|| active_channel_name.map(ToOwned::to_owned))
        }
    }

    fn selected_call_marker_mismatch<'a>(
        &self,
        active_channel_name: Option<&'a str>,
    ) -> Option<(&'a str, String)> {
        let Some(active_channel_name) = active_channel_name else {
            return None;
        };
        let Some(bound_marker) = self.selected_threshold_call_marker_name() else {
            return None;
        };
        (bound_marker != active_channel_name)
            .then(|| (active_channel_name, bound_marker.to_string()))
    }

    fn create_threshold_call_for_channel(&mut self, channel_name: &str) {
        self.analysis_threshold_elements.push(ThresholdSetElement {
            name: self.unique_threshold_call_name(&format!("{channel_name} positive")),
            scope: Some(ThresholdCallScope::Marker {
                channel_name: channel_name.to_string(),
            }),
            mark_failed: false,
            rules: Vec::new(),
        });
        let idx = self.analysis_threshold_elements.len() - 1;
        self.load_threshold_element(idx, Some(channel_name));
    }

    fn marker_call_index_for_channel(&self, channel_name: &str) -> Option<usize> {
        self.analysis_threshold_elements.iter().position(|element| {
            matches!(
                threshold_call_scope(element),
                ThresholdCallScope::Marker { channel_name: ref bound }
                if bound == channel_name
            )
        })
    }

    fn ensure_marker_call_for_channel(&mut self, channel_name: &str) -> usize {
        if let Some(idx) = self.marker_call_index_for_channel(channel_name) {
            return idx;
        }
        self.analysis_threshold_elements.push(ThresholdSetElement {
            name: self.unique_threshold_call_name(&format!("{channel_name} positive")),
            scope: Some(ThresholdCallScope::Marker {
                channel_name: channel_name.to_string(),
            }),
            mark_failed: false,
            rules: Vec::new(),
        });
        self.analysis_threshold_elements.len() - 1
    }

    fn sync_marker_call_selection_to_active_channel(
        &mut self,
        channels: &[ChannelInfo],
        selected_channel: usize,
    ) {
        if !self.analysis_follow_active_channel {
            return;
        }
        let Some(active_channel_name) = self.active_channel_name(channels, selected_channel) else {
            return;
        };
        if matches!(
            self.selected_threshold_call().map(threshold_call_scope),
            Some(ThresholdCallScope::Composite)
        ) {
            return;
        }
        let idx = self.ensure_marker_call_for_channel(active_channel_name);
        if self.analysis_threshold_selected_element != Some(idx) {
            self.load_threshold_element(idx, Some(active_channel_name));
        }
    }

    fn convert_selected_threshold_call_to_composite(&mut self) {
        let Some(idx) = self
            .analysis_threshold_selected_element
            .filter(|idx| *idx < self.analysis_threshold_elements.len())
        else {
            return;
        };
        let Some(element) = self.analysis_threshold_elements.get_mut(idx) else {
            return;
        };
        element.scope = Some(ThresholdCallScope::Composite);
        element.mark_failed = false;
    }

    fn normalize_threshold_call_elements(&mut self) {
        for element in &mut self.analysis_threshold_elements {
            if element.scope.is_none() {
                element.scope = Some(infer_threshold_call_scope(&element.rules));
            }
        }
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
    ) -> Option<ObjectUiAction> {
        let selection_generation_before = self.selection_generation;
        // Analysis always operates on the currently active object snapshot: either the full loaded
        // set or the materialized filtered subset from `ensure_filter_cache`.
        self.ensure_filter_cache();

        ui.heading("Analysis");
        if !self.has_data() {
            ui.label("Load segmentation objects to run per-cell analysis.");
            return None;
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
        let action = self.ui_selection_elements_editor(ui);
        self.normalize_threshold_call_elements();
        ui.label("Source: Object columns");
        self.ui_threshold_set_editor(ui, _channels, selected_channel);
        self.ui_object_properties_analysis(
            ui,
            _channels,
            selected_channel,
            suspend_live_selection_sync,
        );
        // Actor-owned shell layouts may place the canvas before the Analysis mount. In that case
        // this frame's canvas callback already captured the previous selection state. Schedule one
        // follow-up frame so the new state texture is presented even when pointer motion stops.
        if self.selection_generation != selection_generation_before {
            ui.ctx().request_repaint();
        }
        action
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
            ui.label("No numeric object columns are available for plotting.");
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
        ui.checkbox(&mut self.show_selection_overlay, "Show selection overlay");
        self.ui_live_analysis_selection_toggle(ui);

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
                let histogram_column_changed = self.analysis_hist_channel != prev_hist_channel;
                if histogram_column_changed {
                    self.analysis_hist_drag_rule = None;
                    self.sync_histogram_editor_to_column(column_name);
                }
                if self.property_column_available_but_unloaded(column_name) {
                    self.ensure_property_loaded(column_name);
                    ui.label(format!("Loading analysis column '{column_name}'..."));
                    return;
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
                if histogram_column_changed {
                    self.retarget_object_property_threshold(
                        column_name,
                        hist.median,
                        self.active_channel_name(channels, selected_channel),
                    );
                }
                self.ensure_default_object_property_threshold(
                    column_name,
                    hist.median,
                    self.active_channel_name(channels, selected_channel),
                );
                let threshold_rules_finished = self.ui_object_property_threshold_rules(
                    ui,
                    channels,
                    selected_channel,
                    &numeric_columns,
                    column_name,
                );
                self.ui_object_property_histogram_levels(ui, column_name);
                let histogram_drag_finished =
                    self.ui_object_property_histogram(ui, column_name, &hist);
                if !self.analysis_live_selection_enabled
                    && !suspend_live_selection_sync
                    && (threshold_rules_finished || histogram_drag_finished)
                {
                    self.request_threshold_selection_apply();
                }
                let selected_ids = if self.analysis_live_selection_enabled
                    || self.threshold_selection_cache_is_current()
                {
                    Some(self.object_property_threshold_selected_indices())
                } else {
                    None
                };
                let has_matches = selected_ids.as_ref().is_some_and(|ids| !ids.is_empty());
                let mut ordered_passing = None;
                let mut current_focus_pos = None;
                if selected_ids.is_some()
                    && let Some(idx) = self.analysis_hist_focus_object_index
                {
                    let ordered = self.object_property_threshold_ordered_indices(column_name);
                    current_focus_pos = ordered.iter().position(|candidate| *candidate == idx);
                    if current_focus_pos.is_some() {
                        ordered_passing = Some(ordered);
                    } else {
                        self.analysis_hist_focus_object_index = None;
                    }
                }
                ui.horizontal(|ui| {
                    let matching_label = selected_ids
                        .as_ref()
                        .map(|ids| ids.len().to_string())
                        .unwrap_or_else(|| "not calculated".to_string());
                    ui.label(format!("Matching cells: {matching_label}"));
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
                } else if !self.analysis_live_selection_enabled {
                    ui.label("Live threshold selection is off.");
                } else if suspend_live_selection_sync {
                    ui.label("Switch back to Pan to resume live threshold selection.");
                } else if self.consume_live_analysis_selection_dirty() {
                    if let Some(selected_ids) = selected_ids.as_ref() {
                        self.sync_live_analysis_selection(selected_ids);
                    }
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
                if self.property_column_available_but_unloaded(x_column) {
                    self.ensure_property_loaded(x_column);
                    ui.label(format!("Loading analysis column '{x_column}'..."));
                    return;
                }
                if self.property_column_available_but_unloaded(y_column) {
                    self.ensure_property_loaded(y_column);
                    ui.label(format!("Loading analysis column '{y_column}'..."));
                    return;
                }
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
                    if !self.analysis_live_selection_enabled {
                        if (drag_finished || self.analysis_scatter_drag_anchor.is_none())
                            && self.consume_live_analysis_selection_dirty()
                        {
                            self.sync_live_analysis_selection(&selected_ids);
                        } else {
                            ui.label("Release to update selection.");
                        }
                    } else if suspend_live_selection_sync {
                        ui.label("Live analysis selection is paused while Rect/Lasso is active.");
                    } else if (drag_finished || self.analysis_scatter_drag_anchor.is_none())
                        && self.consume_live_analysis_selection_dirty()
                    {
                        self.sync_live_analysis_selection(&selected_ids);
                    } else {
                        ui.label("Release to update selection.");
                    }
                } else {
                    if self.analysis_live_selection_enabled
                        && !suspend_live_selection_sync
                        && self.consume_live_analysis_selection_dirty()
                    {
                        self.sync_live_analysis_selection(&[]);
                    }
                    ui.label("Drag a box to replace the current selection.");
                }
            }
        }
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
            && let (Some(idx), Some(pointer_pos)) = (
                self.analysis_hist_drag_rule,
                response.interact_pointer_pos(),
            )
        {
            let transformed = screen_x_to_value(pointer_pos.x, rect, hist.min, hist.max);
            let value = invert_histogram_value_transform(transformed, value_transform);
            self.preview_histogram_threshold_drag(idx, column_name, value);
        }
        let pointer_released = self.analysis_hist_drag_rule.is_some()
            && !ui.input(|i| i.pointer.button_down(egui::PointerButton::Primary));
        let drag_finished = response.drag_stopped() || pointer_released;
        if drag_finished {
            self.commit_histogram_threshold_drag();
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
        if response.dragged()
            && let (Some(anchor), Some(pos)) = (*drag_anchor, response.interact_pointer_pos())
        {
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

        let selected = points
            .iter()
            .filter_map(|(object_index, x, y)| {
                let value = egui::pos2(*x, *y);
                brush
                    .is_some_and(|rect| rect.contains(value))
                    .then_some(*object_index)
            })
            .collect::<Vec<_>>();
        (selected, drag_finished)
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
        ui.collapsing("Calls", |ui| {
            ui.small("Turn measured marker intensities into reusable boolean calls.");
            ui.horizontal(|ui| {
                ui.label("Preset name");
                ui.add(
                    egui::TextEdit::singleline(&mut self.analysis_threshold_set_name)
                        .desired_width(220.0),
                );
                if ui.button("Save Preset...").clicked() {
                    self.save_threshold_set_dialog();
                }
                if ui.button("Load Preset...").clicked() {
                    self.load_threshold_set_dialog();
                }
            });
            ui.small(
                "Call presets save call definitions only. Measurement results are exported from the Measurements tab.",
            );
            let prev_follow_active_channel = self.analysis_follow_active_channel;
            ui.horizontal(|ui| {
                ui.checkbox(
                    &mut self.analysis_follow_active_channel,
                    "Bind edits to active marker",
                );
                if ui.button("Mapping settings...").clicked() {
                    self.open_analysis_channel_mapping_popup();
                }
            });
            if self.analysis_follow_active_channel != prev_follow_active_channel {
                self.save_live_threshold_rules();
                if self.analysis_follow_active_channel {
                    self.sync_marker_call_selection_to_active_channel(channels, selected_channel);
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
                ui.label(format!("Current marker: {name}"));
            }
            let active_marker_name = self.active_channel_name(channels, selected_channel);
            ui.horizontal(|ui| {
                let selected_idx = self
                    .analysis_threshold_selected_element
                    .filter(|idx| *idx < self.analysis_threshold_elements.len());
                let new_call_label = if self.analysis_follow_active_channel && active_marker_name.is_some() {
                    "New call for active marker"
                } else {
                    "New call"
                };
                if ui.button(new_call_label).clicked() {
                    if self.analysis_follow_active_channel
                        && let Some(channel_name) = active_marker_name
                    {
                        self.create_threshold_call_for_channel(channel_name);
                    } else {
                        self.analysis_threshold_elements.push(ThresholdSetElement {
                            name: self.new_threshold_call_name(channels, selected_channel),
                            scope: Some(ThresholdCallScope::Composite),
                            mark_failed: false,
                            rules: Vec::new(),
                        });
                        self.load_threshold_element(
                            self.analysis_threshold_elements.len() - 1,
                            active_marker_name,
                        );
                    }
                }
                if ui.button("New composite call").clicked() {
                    self.analysis_threshold_elements.push(ThresholdSetElement {
                        name: self.unique_threshold_call_name("Composite call"),
                        scope: Some(ThresholdCallScope::Composite),
                        mark_failed: false,
                        rules: Vec::new(),
                    });
                    self.load_threshold_element(
                        self.analysis_threshold_elements.len() - 1,
                        active_marker_name,
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
                ui.label("No calls yet. Threshold edits will populate the selected call.");
            } else {
                let mut clicked_idx = None;
                egui::ScrollArea::vertical()
                    .id_salt("seg_objects_threshold_elements")
                    .auto_shrink([false, false])
                    .max_height(180.0)
                    .show(ui, |ui| {
                        for (idx, element) in
                            self.analysis_threshold_elements.iter_mut().enumerate()
                        {
                            ui.allocate_ui_with_layout(
                                egui::vec2(ui.available_width(), 0.0),
                                egui::Layout::left_to_right(egui::Align::Min),
                                |ui| {
                                    let selected =
                                        self.analysis_threshold_selected_element == Some(idx);
                                    ui.vertical(|ui| {
                                        if ui
                                            .selectable_label(
                                                selected,
                                                threshold_call_display_name(&element.name),
                                            )
                                            .clicked()
                                        {
                                            clicked_idx = Some(idx);
                                        }
                                        ui.small(format!(
                            "{} • {} rule{}",
                            threshold_call_scope_text(element),
                            element.rules.len(),
                            if element.rules.len() == 1 { "" } else { "s" }
                        ));
                                    });
                                    ui.add_sized(
                                        [140.0, ui.spacing().interact_size.y],
                                        egui::TextEdit::singleline(&mut element.name),
                                    );
                                },
                            );
                        }
                    });
                if let Some(idx) = clicked_idx {
                    if self.analysis_follow_active_channel
                        && self.analysis_threshold_selected_element != Some(idx)
                    {
                        self.save_live_threshold_rules();
                        self.analysis_follow_active_channel = false;
                    }
                    self.load_threshold_element(
                        idx,
                        self.active_channel_name(channels, selected_channel),
                    );
                }
                if let Some(idx) = self
                    .analysis_threshold_selected_element
                    .filter(|idx| *idx < self.analysis_threshold_elements.len())
                {
                    let Some((
                        is_marker_call,
                        type_label,
                        scope_text,
                        export_column,
                        mut mark_failed,
                    )) = self.analysis_threshold_elements.get(idx).map(|element| {
                        (
                            matches!(threshold_call_scope(element), ThresholdCallScope::Marker { .. }),
                            threshold_call_type_label(element),
                            threshold_call_scope_text(element),
                            threshold_call_export_column_name(element),
                            element.mark_failed,
                        )
                    }) else {
                        return;
                    };
                    ui.small(format!("Type: {type_label}"));
                    ui.small(format!("Scope: {scope_text}"));
                    ui.small(format!("Export column: {export_column}"));
                    if ui
                        .add_enabled(
                            is_marker_call,
                            egui::Checkbox::new(&mut mark_failed, "Mark failed"),
                        )
                        .changed()
                        && let Some(element) = self.analysis_threshold_elements.get_mut(idx)
                    {
                        element.mark_failed = mark_failed;
                    }
                    if is_marker_call && mark_failed {
                        ui.small("This call exports `FAIL` for every row.");
                    } else if !is_marker_call {
                        ui.small("`Mark failed` is only available for marker calls.");
                    }
                    if let Some((active_marker_name, bound_marker)) =
                        self.selected_call_marker_mismatch(active_marker_name)
                    {
                        ui.label(format!(
                            "Selected call is bound to {bound_marker}. Threshold edits still apply to {bound_marker}, not {active_marker_name}."
                        ));
                        ui.horizontal(|ui| {
                            if ui.button(format!("Create call for {active_marker_name}")).clicked() {
                                self.create_threshold_call_for_channel(active_marker_name);
                            }
                            if ui.button("Convert to composite").clicked() {
                                self.convert_selected_threshold_call_to_composite();
                            }
                        });
                    } else if is_marker_call && ui.button("Convert to composite").clicked()
                    {
                        self.convert_selected_threshold_call_to_composite();
                    }
                }
            }
        });
    }

    fn new_threshold_call_name(&self, channels: &[ChannelInfo], selected_channel: usize) -> String {
        let fallback = format!("Call {}", self.analysis_threshold_elements.len() + 1);
        let base = if self.analysis_follow_active_channel {
            self.active_channel_name(channels, selected_channel)
                .map(|name| format!("{name} positive"))
                .unwrap_or(fallback)
        } else {
            self.unique_threshold_call_name("Composite call")
        };
        self.unique_threshold_call_name(&base)
    }

    fn unique_threshold_call_name(&self, base: &str) -> String {
        let trimmed = base.trim();
        let base = if trimmed.is_empty() { "Call" } else { trimmed };
        let names = self
            .analysis_threshold_elements
            .iter()
            .map(|element| element.name.trim().to_ascii_lowercase())
            .collect::<std::collections::HashSet<_>>();
        if !names.contains(&base.to_ascii_lowercase()) {
            return base.to_string();
        }
        let mut idx = 2usize;
        loop {
            let candidate = format!("{base} {idx}");
            if !names.contains(&candidate.to_ascii_lowercase()) {
                return candidate;
            }
            idx += 1;
        }
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

    pub fn ui_analysis_channel_mapping_popup(
        &mut self,
        ctx: &egui::Context,
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
        let dialog_id = egui::Id::new((
            "analysis_channel_mapping_popup",
            self as *const ObjectsLayer as usize,
        ));
        egui::Window::new("Mapping settings")
            .id(dialog_id)
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_width(760.0)
            .default_height(520.0)
            .show(ctx, |ui| {
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
                            let mut ordered_columns = Vec::with_capacity(numeric_columns.len());
                            for column in suggestions.iter().chain(numeric_columns.iter()) {
                                if !ordered_columns.iter().any(|candidate| candidate == column) {
                                    ordered_columns.push(column.clone());
                                }
                            }
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
                                    ("seg_objects_channel_mapping", idx),
                                    &ordered_columns,
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
        if numeric_columns.is_empty() {
            return;
        }
        let Some(channel_name) = self
            .effective_threshold_channel_name(self.active_channel_name(channels, selected_channel))
        else {
            return;
        };

        if self.analysis_live_threshold_channel_name.as_deref() != Some(channel_name.as_str()) {
            self.load_live_threshold_rules(Some(channel_name.as_str()));
        }

        let target_column = self
            .saved_threshold_rules()
            .iter()
            .find(|rule| rule.channel_name.as_deref() == Some(channel_name.as_str()))
            .map(|rule| rule.column_key.clone())
            .or_else(|| self.mapped_column_for_channel(&channel_name, channels, numeric_columns));

        let Some(target_column) = target_column else {
            return;
        };
        let Some(idx) = numeric_columns
            .iter()
            .position(|column| column == &target_column)
        else {
            return;
        };
        self.assign_channel_to_column_rule(&target_column, Some(channel_name.as_str()));
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

    fn ordered_threshold_picker_columns(
        &self,
        numeric_columns: &[String],
        channel_name: Option<&str>,
        current_column: &str,
    ) -> Vec<String> {
        let mut ordered = Vec::with_capacity(numeric_columns.len());
        let mut push_column = |column: &str| {
            if !ordered.iter().any(|candidate| candidate == column) {
                ordered.push(column.to_string());
            }
        };

        if let Some(channel_name) = channel_name
            && let Some(suggestions) = self
                .analysis_channel_mapping_suggestions_cache
                .get(channel_name)
        {
            for column in suggestions {
                push_column(column);
            }
        }

        push_column(current_column);

        for column in numeric_columns {
            push_column(column);
        }

        ordered
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
        element: &ThresholdSetElement,
        channel_name: Option<&str>,
    ) -> Vec<ObjectPropertyThresholdRule> {
        match threshold_call_scope(element) {
            ThresholdCallScope::Marker { channel_name } => {
                let mut rules = element.rules.clone();
                for rule in &mut rules {
                    rule.channel_name = Some(channel_name.clone());
                }
                rules
            }
            ThresholdCallScope::Composite => {
                if !self.analysis_follow_active_channel {
                    return element.rules.clone();
                }

                let Some(channel_name) = channel_name else {
                    return Vec::new();
                };
                let tagged = element
                    .rules
                    .iter()
                    .filter(|rule| rule.channel_name.as_deref() == Some(channel_name))
                    .cloned()
                    .collect::<Vec<_>>();
                if !tagged.is_empty() {
                    return tagged;
                }

                // Backward compatibility for older threshold sets that predate per-channel tagging.
                if element.rules.iter().all(|rule| rule.channel_name.is_none()) {
                    return element.rules.to_vec();
                }

                Vec::new()
            }
        }
    }

    fn load_live_threshold_rules(&mut self, channel_name: Option<&str>) {
        if let Some(element) = self.selected_threshold_call().cloned() {
            let live_channel_name = match threshold_call_scope(&element) {
                ThresholdCallScope::Marker { channel_name } => Some(channel_name),
                ThresholdCallScope::Composite => {
                    if self.analysis_follow_active_channel {
                        channel_name.map(ToOwned::to_owned)
                    } else {
                        None
                    }
                }
            };
            self.analysis_property_thresholds =
                self.extract_live_threshold_rules(&element, live_channel_name.as_deref());
            self.analysis_live_threshold_channel_name = live_channel_name;
        } else {
            let live_channel_name = if self.analysis_follow_active_channel {
                channel_name.map(ToOwned::to_owned)
            } else {
                None
            };
            self.analysis_property_thresholds = self.analysis_property_thresholds.clone();
            self.analysis_live_threshold_channel_name = live_channel_name;
        }
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
            let Some(element) = self.analysis_threshold_elements.get(idx).cloned() else {
                return;
            };
            match threshold_call_scope(&element) {
                ThresholdCallScope::Marker { channel_name } => {
                    let mut rules = self.analysis_property_thresholds.clone();
                    for rule in &mut rules {
                        rule.channel_name = Some(channel_name.clone());
                    }
                    self.analysis_threshold_elements[idx].scope =
                        Some(ThresholdCallScope::Marker { channel_name });
                    self.analysis_threshold_elements[idx].rules = rules;
                }
                ThresholdCallScope::Composite => {
                    if self.analysis_follow_active_channel {
                        let active_channel = self.analysis_live_threshold_channel_name.clone();
                        let mut merged = self.analysis_threshold_elements[idx].rules.clone();
                        if let Some(channel_name) = active_channel.as_deref() {
                            merged
                                .retain(|rule| rule.channel_name.as_deref() != Some(channel_name));
                        } else {
                            merged.clear();
                        }
                        merged.extend(self.analysis_property_thresholds.clone());
                        self.analysis_threshold_elements[idx].rules = merged;
                    } else {
                        self.analysis_threshold_elements[idx].rules =
                            self.analysis_property_thresholds.clone();
                    }
                    self.analysis_threshold_elements[idx].scope =
                        Some(ThresholdCallScope::Composite);
                }
            }
        } else if !self.analysis_property_thresholds.is_empty() {
            let scope = self
                .analysis_live_threshold_channel_name
                .as_ref()
                .map(|channel_name| ThresholdCallScope::Marker {
                    channel_name: channel_name.clone(),
                })
                .unwrap_or(ThresholdCallScope::Composite);
            self.analysis_threshold_elements.push(ThresholdSetElement {
                name: self.unique_threshold_call_name("Call"),
                scope: Some(scope.clone()),
                mark_failed: false,
                rules: self.analysis_property_thresholds.clone(),
            });
            if let Some(element) = self.analysis_threshold_elements.last_mut()
                && let ThresholdCallScope::Marker { channel_name } = scope
            {
                for rule in &mut element.rules {
                    rule.channel_name = Some(channel_name.clone());
                }
            }
            self.analysis_threshold_selected_element =
                Some(self.analysis_threshold_elements.len() - 1);
        }
    }

    fn load_threshold_element(&mut self, idx: usize, active_channel_name: Option<&str>) {
        let Some(element) = self.analysis_threshold_elements.get(idx).cloned() else {
            return;
        };
        self.analysis_threshold_selected_element = Some(idx);
        let live_channel_name = match threshold_call_scope(&element) {
            ThresholdCallScope::Marker { channel_name } => Some(channel_name),
            ThresholdCallScope::Composite => {
                if self.analysis_follow_active_channel {
                    active_channel_name.map(ToOwned::to_owned)
                } else {
                    None
                }
            }
        };
        self.analysis_property_thresholds =
            self.extract_live_threshold_rules(&element, live_channel_name.as_deref());
        self.analysis_live_threshold_channel_name = live_channel_name;
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
            .add_filter("Call preset", &["json"])
            .set_title("Save call preset")
            .set_file_name("call_preset.json")
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
            .add_filter("Call preset", &["json"])
            .set_title("Load call preset")
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
        self.normalize_threshold_call_elements();
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

                    let matches = if search.trim().is_empty() {
                        names.iter().collect::<Vec<_>>()
                    } else {
                        let mut matches = names
                            .iter()
                            .filter_map(|name| {
                                fuzzy_name_score(&search, name).map(|score| (score, name))
                            })
                            .collect::<Vec<_>>();
                        matches.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(b.1)));
                        matches
                            .into_iter()
                            .map(|(_, name)| name)
                            .collect::<Vec<_>>()
                    };

                    egui::ScrollArea::vertical()
                        .id_salt((&id_salt, "scroll"))
                        .auto_shrink([false, false])
                        .max_height(320.0)
                        .show(ui, |ui| {
                            for name in matches.into_iter().take(100) {
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

                    let matches = if search.trim().is_empty() {
                        names.iter().enumerate().collect::<Vec<_>>()
                    } else {
                        let mut matches = names
                            .iter()
                            .enumerate()
                            .filter_map(|(idx, name)| {
                                fuzzy_name_score(&search, name).map(|score| (score, idx, name))
                            })
                            .collect::<Vec<_>>();
                        matches.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
                        matches
                            .into_iter()
                            .map(|(_, idx, name)| (idx, name))
                            .collect::<Vec<_>>()
                    };

                    egui::ScrollArea::vertical()
                        .id_salt((&id_salt, "scroll"))
                        .auto_shrink([false, false])
                        .max_height(320.0)
                        .show(ui, |ui| {
                            for (match_idx, (_, name)) in matches.into_iter().take(100).enumerate()
                            {
                                ui.push_id((&id_salt, "option", match_idx), |ui| {
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
                                });
                            }
                        });
                    if request_close {
                        ui.close();
                    }
                });
            ui.data_mut(|d| d.insert_temp(popup_state_id, popup_open));
        });
    }
}
