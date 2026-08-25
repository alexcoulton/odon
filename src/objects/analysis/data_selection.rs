//! Object-property data derivation, caching, and live-selection state.

use super::*;

impl ObjectsLayer {
    pub(crate) fn available_numeric_object_property_keys(&mut self) -> Vec<String> {
        if let Some(cached) = self.object_property_numeric_keys_cache.as_ref() {
            return cached.clone();
        }
        let Some(objects) = self.objects.as_ref() else {
            return Vec::new();
        };
        let mut out = self.property_store.numeric_keys();
        let mut existing = out.iter().cloned().collect::<HashSet<_>>();
        out.extend(
            self.scalar_property_keys
                .iter()
                .filter(|key| key.as_str() != "id")
                .filter(|key| !existing.contains(*key))
                .filter(|key| {
                    objects.iter().any(|obj| {
                        obj.inline_properties
                            .get(*key)
                            .and_then(numeric_json_value)
                            .is_some()
                    })
                })
                .cloned(),
        );
        existing.extend(out.iter().cloned());
        if let Some(source) = self.lazy_parquet_source.as_ref() {
            out.extend(
                source
                    .numeric_property_columns
                    .iter()
                    .filter(|key| key.as_str() != "id")
                    .filter(|key| !existing.contains(*key))
                    .cloned(),
            );
        }
        out.sort();
        self.object_property_numeric_keys_cache = Some(out.clone());
        out
    }

    pub(in crate::objects) fn object_property_column_pairs(
        &mut self,
        key: &str,
    ) -> Arc<Vec<(usize, f32)>> {
        if self.property_column_available_but_unloaded(key) {
            self.ensure_property_loaded(key);
            return Arc::new(Vec::new());
        }
        if self.filtered_mask.is_none() {
            if let Some(cached) = self.object_property_base_pairs_cache.get(key) {
                return cached.clone();
            }
            if let Some(out) = self.property_store.numeric_pairs(key) {
                let out = Arc::new(out);
                self.object_property_base_pairs_cache
                    .insert(key.to_string(), out.clone());
                return out;
            }
            let Some(objects) = self.objects.as_ref() else {
                return Arc::new(Vec::new());
            };
            let mut out = Vec::new();
            for (idx, obj) in objects.iter().enumerate() {
                let Some(value) = obj.inline_properties.get(key).and_then(numeric_json_value)
                else {
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
        if let Some(mut out) = self.property_store.numeric_pairs(key) {
            if let Some(filtered_mask) = self.filtered_mask.as_ref() {
                out.retain(|(idx, _)| filtered_mask.get(*idx).copied().unwrap_or(false));
            }
            let out = Arc::new(out);
            self.object_property_pairs_cache
                .insert(key.to_string(), out.clone());
            return out;
        }
        let Some(objects) = self.objects.as_ref() else {
            return Arc::new(Vec::new());
        };
        let filtered_mask = self.filtered_mask.as_ref();
        let mut out = Vec::new();
        for (idx, obj) in objects.iter().enumerate() {
            if filtered_mask.is_some_and(|mask| !mask.get(idx).copied().unwrap_or(false)) {
                continue;
            }
            let Some(value) = obj.inline_properties.get(key).and_then(numeric_json_value) else {
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

    pub(in crate::objects) fn object_property_sorted_pairs(
        &mut self,
        key: &str,
    ) -> Arc<Vec<(usize, f32)>> {
        if self.filtered_mask.is_none() {
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

    pub(in crate::objects) fn object_property_histogram(
        &mut self,
        key: &str,
    ) -> Option<SimpleHistogram> {
        // Keep separate caches for base vs filtered histograms. Filter changes are much more
        // frequent than raw data changes, so this avoids rebuilding the unfiltered histogram when
        // users toggle subset views.
        if self.filtered_mask.is_none() {
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

    pub(in crate::objects) fn object_property_scatter_points(
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

    pub(in crate::objects) fn invalidate_object_property_analysis_cache(&mut self) {
        self.object_property_pairs_cache.clear();
        self.object_property_hist_cache.clear();
        self.object_property_scatter_cache.clear();
        self.object_property_hist_levels_cache.clear();
        self.object_property_threshold_selection_cache_key = None;
        self.object_property_threshold_selection_cache = Arc::new(Vec::new());
        self.object_property_threshold_order_cache_key = None;
        self.object_property_threshold_order_cache = Arc::new(Vec::new());
    }

    pub(in crate::objects) fn reset_object_property_analysis_cache(&mut self) {
        self.object_property_numeric_keys_cache = None;
        self.object_property_base_pairs_cache.clear();
        self.object_property_base_sorted_pairs_cache.clear();
        self.object_property_base_hist_cache.clear();
        self.object_property_base_hist_levels_cache.clear();
        self.analysis_warm_started = false;
        self.analysis_warm_rx = None;
        self.analysis_selection_rx = None;
        self.analysis_warm_total_columns = 0;
        self.analysis_warm_completed_columns = 0;
        self.invalidate_object_property_analysis_cache();
    }

    pub(in crate::objects) fn sync_live_analysis_selection(&mut self, indices: &[usize]) {
        self.apply_selection_indices(indices, false);
    }

    pub(in crate::objects) fn request_threshold_selection_apply(&mut self) {
        if self.analysis_property_thresholds.is_empty() {
            self.sync_live_analysis_selection(&[]);
            self.mark_live_analysis_selection_applied();
            self.analysis_selection_rx = None;
            return;
        }

        let cache_key = self.threshold_selection_cache_key();
        if self.threshold_selection_cache_is_current() {
            let selected_ids = Arc::clone(&self.object_property_threshold_selection_cache);
            self.sync_live_analysis_selection(&selected_ids);
            self.mark_live_analysis_selection_applied();
            self.analysis_selection_rx = None;
            return;
        }

        let Some(objects) = self.objects.as_ref().cloned() else {
            self.analysis_selection_rx = None;
            return;
        };
        let job_rules = self
            .analysis_property_thresholds
            .iter()
            .cloned()
            .map(|rule| AnalysisSelectionJobRule { rule })
            .collect::<Vec<_>>();
        let property_store = self.property_store.clone();
        let filtered_mask = self.filtered_mask.clone();

        self.analysis_selection_request_id =
            self.analysis_selection_request_id.wrapping_add(1).max(1);
        let request_id = self.analysis_selection_request_id;
        let (tx, rx) = crossbeam_channel::bounded::<AnalysisSelectionResult>(1);
        self.analysis_selection_rx = Some(rx);
        self.mark_live_analysis_selection_applied();

        std::thread::Builder::new()
            .name("seg-objects-analysis-selection".to_string())
            .spawn(move || {
                let (indices, proxy_positions_world, proxy_values) =
                    compute_threshold_selection_indices(
                        &job_rules,
                        objects.as_slice(),
                        &property_store,
                        filtered_mask.as_ref().map(|mask| mask.as_slice()),
                    );
                let _ = tx.send(AnalysisSelectionResult {
                    request_id,
                    cache_key,
                    indices,
                    proxy_positions_world,
                    proxy_values,
                });
            })
            .ok();
    }

    pub(in crate::objects) fn reset_live_analysis_selection_default(&mut self) {
        self.analysis_live_selection_enabled =
            self.object_count() <= LIVE_ANALYSIS_SELECTION_OBJECT_LIMIT;
        self.mark_live_analysis_selection_applied();
    }

    pub(in crate::objects) fn ui_live_analysis_selection_toggle(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui
                .checkbox(&mut self.analysis_live_selection_enabled, "Live")
                .changed()
                && self.analysis_live_selection_enabled
            {
                self.mark_live_analysis_selection_dirty();
            }
            let _ = ui.small_button("?").on_hover_text(format!(
                "Updates the object selection automatically as analysis thresholds or brushes change. Disabled by default above {LIVE_ANALYSIS_SELECTION_OBJECT_LIMIT} cells."
            ));
        });
    }

    pub(in crate::objects) fn apply_selection_indices(
        &mut self,
        indices: &[usize],
        additive: bool,
    ) {
        if !additive {
            self.selected_object_indices = indices.iter().copied().collect();
        } else {
            for idx in indices {
                self.selected_object_indices.insert(*idx);
            }
        }
        self.selected_object_index = self.selected_object_indices.iter().next().copied();
        self.rebuild_selection_render_lods();
        self.clear_measurements();
        self.invalidate_table_cache();
    }

    pub(in crate::objects) fn mark_live_analysis_selection_dirty(&mut self) {
        self.analysis_live_selection_generation = self
            .analysis_live_selection_generation
            .wrapping_add(1)
            .max(1);
    }

    pub(in crate::objects) fn mark_live_analysis_selection_applied(&mut self) {
        self.analysis_live_selection_applied_generation = self.analysis_live_selection_generation;
    }

    pub(in crate::objects) fn consume_live_analysis_selection_dirty(&mut self) -> bool {
        if self.analysis_live_selection_applied_generation
            == self.analysis_live_selection_generation
        {
            return false;
        }
        self.analysis_live_selection_applied_generation = self.analysis_live_selection_generation;
        true
    }

    pub(in crate::objects) fn has_live_analysis_selection(&self) -> bool {
        !self.analysis_property_thresholds.is_empty() || self.analysis_scatter_brush.is_some()
    }

    pub(in crate::objects) fn ensure_default_object_property_threshold(
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

    pub(in crate::objects) fn ui_object_property_threshold_rules(
        &mut self,
        ui: &mut egui::Ui,
        channels: &[ChannelInfo],
        selected_channel: usize,
        numeric_columns: &[String],
        default_column: &str,
    ) -> bool {
        ui.separator();
        ui.label("Thresholds");
        self.ensure_channel_mapping_suggestions_cache(channels, numeric_columns);
        let active_channel_name = self.active_channel_name(channels, selected_channel);
        let effective_channel_name = self.effective_threshold_channel_name(active_channel_name);
        let ordered_columns_per_rule = self
            .analysis_property_thresholds
            .iter()
            .map(|rule| {
                self.ordered_threshold_picker_columns(
                    numeric_columns,
                    rule.channel_name
                        .as_deref()
                        .or(effective_channel_name.as_deref()),
                    &rule.column_key,
                )
            })
            .collect::<Vec<_>>();
        let mut remove_idx = None;
        let mut changed = false;
        let mut finished = false;
        for (idx, rule) in self.analysis_property_thresholds.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                let prev_column = rule.column_key.clone();
                Self::analysis_value_name_picker(
                    ui,
                    "Column",
                    ("seg_objects_threshold_column", idx),
                    ordered_columns_per_rule
                        .get(idx)
                        .map(Vec::as_slice)
                        .unwrap_or(numeric_columns),
                    &mut rule.column_key,
                );
                if rule.column_key != prev_column {
                    changed = true;
                    finished = true;
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
                    finished = true;
                }
                let response = ui.add(egui::DragValue::new(&mut rule.value).speed(0.1));
                if response.changed() {
                    changed = true;
                }
                if response.drag_stopped() {
                    finished = true;
                }
                if ui.button("Remove").clicked() {
                    remove_idx = Some(idx);
                    finished = true;
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
            finished = true;
        }
        if changed {
            self.sync_active_threshold_element_from_live_rules();
        }
        finished
    }

    pub(in crate::objects) fn ui_object_property_histogram_levels(
        &mut self,
        ui: &mut egui::Ui,
        column_name: &str,
    ) {
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

    pub(in crate::objects) fn set_histogram_threshold_value(
        &mut self,
        column_name: &str,
        value: f32,
    ) {
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

    pub(in crate::objects) fn set_histogram_threshold_snap(
        &mut self,
        column_name: &str,
        value: f32,
        level_index: usize,
    ) {
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

    pub(in crate::objects) fn clear_histogram_snapped_level_for_column(
        &mut self,
        column_name: &str,
    ) {
        if self
            .analysis_hist_snapped_level
            .as_ref()
            .is_some_and(|selection| selection.column_key == column_name)
        {
            self.analysis_hist_snapped_level = None;
        }
    }

    pub(in crate::objects) fn threshold_selection_cache_key(&self) -> String {
        self.analysis_property_thresholds
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
            .join("||")
    }

    pub(in crate::objects) fn threshold_selection_cache_is_current(&self) -> bool {
        if self.analysis_property_thresholds.is_empty() {
            return false;
        }
        let cache_key = self.threshold_selection_cache_key();
        self.object_property_threshold_selection_cache_key
            .as_deref()
            == Some(cache_key.as_str())
    }

    pub(in crate::objects) fn object_property_threshold_selected_indices(
        &mut self,
    ) -> Arc<Vec<usize>> {
        if self.analysis_property_thresholds.is_empty() {
            return Arc::new(Vec::new());
        }
        let cache_key = self.threshold_selection_cache_key();
        if self
            .object_property_threshold_selection_cache_key
            .as_deref()
            == Some(&cache_key)
        {
            return Arc::clone(&self.object_property_threshold_selection_cache);
        }
        if self.filtered_mask.is_none() && self.analysis_property_thresholds.len() == 1 {
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

    pub(in crate::objects) fn object_property_histogram_levels(
        &mut self,
        column_name: &str,
        value_transform: HistogramValueTransform,
        method: HistogramLevelMethod,
        level_count: usize,
    ) -> Arc<Vec<f32>> {
        if self.filtered_mask.is_none() {
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

    pub(in crate::objects) fn object_property_threshold_ordered_indices(
        &mut self,
        column_name: &str,
    ) -> Arc<Vec<usize>> {
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
            if self.filtered_mask.is_some() {
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

        let out = if self.filtered_mask.is_none()
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

    pub(in crate::objects) fn object_property_histogram_values(&mut self, key: &str) -> Vec<f32> {
        let value_transform = self.analysis_hist_value_transform;
        self.object_property_column_pairs(key)
            .iter()
            .map(|(_, value)| apply_histogram_value_transform(*value, value_transform))
            .filter(|v| v.is_finite())
            .collect::<Vec<_>>()
    }

    pub(in crate::objects) fn clear_measurements(&mut self) {}

    pub(in crate::objects) fn clear_analysis(&mut self) {
        self.analysis_property_thresholds.clear();
        self.sync_active_threshold_element_from_live_rules();
        self.analysis_selection_rx = None;
        self.analysis_hist_drag_rule = None;
        self.analysis_hist_brush = None;
        self.analysis_scatter_brush = None;
        self.analysis_hist_drag_anchor = None;
        self.analysis_scatter_drag_anchor = None;
        self.analysis_scatter_view_key = None;
        self.analysis_scatter_view_rect = None;
    }
}
