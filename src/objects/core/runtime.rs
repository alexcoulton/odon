//! Object-layer loading, async completion, source dialogs, and runtime installation.

use super::*;

impl ObjectsLayer {
    pub(super) fn cancel_current_load(&mut self) {
        if let Some(cancel) = self.object_load_cancel.take() {
            cancel.store(true, Ordering::Relaxed);
        }
        self.load_rx = None;
    }

    pub fn tick(&mut self) {
        use crossbeam_channel::TryRecvError;

        if let Some(rx) = self.load_rx.clone() {
            loop {
                match rx.try_recv() {
                    Ok(Ok(msg)) => {
                        if msg.request_id != self.object_load_request_id {
                            continue;
                        }
                        self.load_rx = None;
                        self.object_load_cancel = None;
                        self.install_load_result(msg);
                    }
                    Ok(Err(error)) => {
                        self.load_rx = None;
                        self.object_load_cancel = None;
                        self.status = format!("Object load failed: {error}");
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        self.load_rx = None;
                        self.object_load_cancel = None;
                        break;
                    }
                }
            }
        }

        if let Some(rx) = self.property_load_rx.clone() {
            loop {
                match rx.try_recv() {
                    Ok(msg) => {
                        self.apply_loaded_property_values(
                            msg.property_key.as_str(),
                            &msg.values_by_row,
                        );
                        if self.property_load_key.as_deref() == Some(msg.property_key.as_str()) {
                            self.property_load_rx = None;
                            self.property_load_key = None;
                        }
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        self.property_load_rx = None;
                        self.property_load_key = None;
                        break;
                    }
                }
            }
        }

        if let Some(rx) = self.analysis_warm_rx.clone() {
            loop {
                match rx.try_recv() {
                    Ok(AnalysisWarmupEvent::Started {
                        request_id,
                        numeric_columns,
                        total,
                    }) => {
                        if request_id == self.analysis_warm_request_id {
                            self.object_property_numeric_keys_cache = Some(numeric_columns);
                            self.analysis_warm_total_columns = total;
                            self.analysis_warm_completed_columns = 0;
                        }
                    }
                    Ok(AnalysisWarmupEvent::ColumnReady {
                        request_id,
                        key,
                        pairs,
                        sorted_pairs,
                        histograms,
                        levels,
                        completed,
                        total,
                    }) => {
                        if request_id == self.analysis_warm_request_id {
                            self.object_property_base_pairs_cache
                                .insert(key.clone(), pairs);
                            self.object_property_base_sorted_pairs_cache
                                .insert(key, sorted_pairs);
                            for (cache_key, hist) in histograms {
                                self.object_property_base_hist_cache.insert(cache_key, hist);
                            }
                            for (cache_key, level_values) in levels {
                                self.object_property_base_hist_levels_cache
                                    .insert(cache_key, level_values);
                            }
                            self.analysis_warm_completed_columns = completed.min(total);
                            self.analysis_warm_total_columns = total;
                        }
                    }
                    Ok(AnalysisWarmupEvent::Finished { request_id }) => {
                        if request_id == self.analysis_warm_request_id {
                            self.analysis_warm_completed_columns = self.analysis_warm_total_columns;
                            self.analysis_warm_rx = None;
                        }
                        break;
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        self.analysis_warm_rx = None;
                        break;
                    }
                }
            }
        }

        loop {
            let Some(rx) = self.bulk_measurement_rx.as_ref() else {
                break;
            };
            match rx.try_recv() {
                Ok(BulkMeasurementEvent::Progress {
                    request_id,
                    phase,
                    completed,
                    total,
                }) => {
                    if request_id == self.bulk_measurement_request_id {
                        self.bulk_measurement_progress_completed = completed.min(total);
                        self.bulk_measurement_progress_total = total;
                        self.bulk_measurement_status =
                            measurements::bulk_measurement_progress_status(phase, completed, total);
                    }
                }
                Ok(BulkMeasurementEvent::Finished {
                    request_id,
                    result,
                    cancelled,
                    error,
                }) => {
                    if request_id == self.bulk_measurement_request_id {
                        self.bulk_measurement_rx = None;
                        self.bulk_measurement_cancel = None;
                        if let Some(err) = error {
                            self.bulk_measurement_status = format!("Measurements failed: {err}");
                        } else if cancelled {
                            self.bulk_measurement_status = format!(
                                "Measurements cancelled at {} / {} steps.",
                                self.bulk_measurement_progress_completed,
                                self.bulk_measurement_progress_total
                            );
                        } else if let Some(result) = result {
                            self.apply_bulk_measurement_result(result);
                        }
                    }
                    break;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.bulk_measurement_rx = None;
                    self.bulk_measurement_cancel = None;
                    break;
                }
            }
        }

        if let Some(rx) = self.object_export_rx.as_ref() {
            match rx.try_recv() {
                Ok(ObjectExportEvent::Finished {
                    request_id,
                    path,
                    object_count,
                    error,
                }) => {
                    if request_id == self.object_export_request_id {
                        self.object_export_rx = None;
                        if let Some(err) = error {
                            self.status = format!("Export failed: {err}");
                        } else {
                            self.status = format!(
                                "Exported {} object(s) to {}",
                                object_count,
                                path.to_string_lossy()
                            );
                        }
                    }
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => {
                    self.object_export_rx = None;
                }
            }
        }

        if let Some(rx) = self.analysis_selection_rx.clone() {
            loop {
                match rx.try_recv() {
                    Ok(result) => {
                        if result.request_id != self.analysis_selection_request_id {
                            continue;
                        }
                        if result.cache_key != self.threshold_selection_cache_key() {
                            self.analysis_selection_rx = None;
                            continue;
                        }
                        self.analysis_selection_rx = None;
                        self.object_property_threshold_selection_cache_key = Some(result.cache_key);
                        self.object_property_threshold_selection_cache =
                            Arc::clone(&result.indices);
                        self.apply_selection_indices(&result.indices, false);
                        if self.display_mode == ObjectDisplayMode::Polygons
                            && result.indices.len() > ObjectsLayer::SELECTED_RENDER_LOD_LIMIT
                        {
                            self.selected_point_positions_world =
                                Some(Arc::clone(&result.proxy_positions_world));
                            self.selected_point_values = Some(Arc::clone(&result.proxy_values));
                            self.selected_point_lods = Some(Arc::new(Vec::new()));
                        }
                        self.status = format!(
                            "Applied analysis selection to {} object(s).",
                            result.indices.len()
                        );
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        self.analysis_selection_rx = None;
                        break;
                    }
                }
            }
        }
    }

    pub(super) fn install_load_result(&mut self, msg: LoadResult) {
        // Resource projection may resend an immutable payload on consecutive frames. Treat that
        // as a complete no-op; for a real update, only discard selection and Analysis state when
        // the underlying geometry changed and those indices could refer to a different object set.
        let payload_unchanged = self
            .objects
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, &msg.objects))
            && self.loaded_geojson.as_ref() == Some(&msg.path)
            && self.downsample_factor == msg.downsample_factor.max(1e-6)
            && self.display_transform == msg.display_transform
            && self.display_mode == msg.display_mode
            && self.object_property_keys == msg.object_property_keys
            && self.scalar_property_keys == msg.scalar_property_keys
            && self.color_property_keys == msg.color_property_keys
            && self.property_store.shares_storage_with(&msg.property_store)
            && self.lazy_parquet_source == msg.lazy_parquet_source;
        if payload_unchanged {
            return;
        }
        let geometry_changed = match (&self.object_fill_mesh, &msg.object_fill_mesh) {
            (Some(current), Some(incoming)) => {
                !Arc::ptr_eq(&current.vertices_local, &incoming.vertices_local)
            }
            (None, None) => self
                .objects
                .as_ref()
                .is_none_or(|current| !Arc::ptr_eq(current, &msg.objects)),
            _ => true,
        };
        self.control_renderer_payload_identity = None;
        self.property_load_rx = None;
        self.property_load_key = None;
        self.display_transform = msg.display_transform;
        self.display_mode = msg.display_mode;
        self.objects = Some(msg.objects);
        self.bins = Some(msg.bins);
        self.render_lods = Some(msg.render_lods);
        self.object_fill_mesh = msg.object_fill_mesh;
        self.object_selection_lods = msg.object_selection_lods;
        self.point_positions_world = Some(msg.point_positions_world);
        self.point_values = Some(msg.point_values);
        self.point_lods = Some(msg.point_lods);
        self.gl_proxy_group_points.clear();
        self.object_property_keys = msg.object_property_keys;
        self.scalar_property_keys = msg.scalar_property_keys;
        self.color_property_keys = msg.color_property_keys;
        self.lazy_parquet_source = msg.lazy_parquet_source;
        self.property_store = msg.property_store;
        self.color_legend_cache = None;
        self.continuous_color_payload = None;
        self.color_groups_cache.clear();
        let has_active_color_key = (self.color_mode == ObjectColorMode::Continuous
            && (self
                .scalar_property_keys
                .iter()
                .any(|key| key == &self.color_property_key)
                || self.property_store.has_loaded(&self.color_property_key)))
            || self
                .color_property_keys
                .iter()
                .any(|k| k == &self.color_property_key)
            || self.lazy_parquet_source.as_ref().is_some_and(|source| {
                source
                    .available_property_columns
                    .iter()
                    .any(|k| k == &self.color_property_key)
            });
        if !has_active_color_key {
            self.color_property_key.clear();
            self.color_mode = ObjectColorMode::Single;
            self.color_mapping = ObjectColorMapping::Single;
            self.resolved_continuous_domain = None;
            self.color_level_overrides_property_key.clear();
            self.color_level_overrides.clear();
        }
        self.reconcile_filter_clauses();
        self.color_groups = None;
        self.filtered_ordered_indices = None;
        self.filtered_mask = None;
        self.filtered_render_lods = None;
        self.filtered_point_positions_world = None;
        self.filtered_point_values = None;
        self.filtered_point_lods = None;
        self.filtered_color_groups = None;
        if geometry_changed {
            self.selected_object_indices.clear();
            self.selected_object_index = None;
            self.selection_elements.clear();
            self.selection_element_selected = None;
            self.selection_element_name_draft = "Selection Element 1".to_string();
            self.selected_render_lods = None;
            self.primary_selected_render_lods = None;
            self.selected_fill_mesh = None;
            self.selection_fill_state = Arc::new(Vec::new());
            self.selection_cpu_overlay_dirty = false;
            self.selected_point_positions_world = None;
            self.selected_point_values = None;
            self.selected_point_lods = None;
            self.primary_selected_point_positions_world = None;
            self.primary_selected_point_values = None;
            self.visible_selected_render_cache = None;
            self.selection_generation = self.selection_generation.wrapping_add(1).max(1);
        }
        self.clear_measurements();
        self.clear_bulk_measurements();
        if geometry_changed {
            self.clear_analysis();
            self.analysis_threshold_set_name = "Threshold Set".to_string();
            self.analysis_threshold_elements.clear();
            self.analysis_threshold_selected_element = None;
            self.analysis_live_threshold_channel_name = None;
            self.analysis_channel_mapping_overrides.clear();
            self.analysis_channel_mapping_popup_open = false;
            self.analysis_channel_mapping_search.clear();
        }
        self.analysis_channel_mapping_suggestions_cache_key = 0;
        self.analysis_channel_mapping_suggestions_cache_channels_len = 0;
        self.analysis_channel_mapping_suggestions_cache_numeric_len = 0;
        self.analysis_channel_mapping_suggestions_cache.clear();
        self.reset_object_property_analysis_cache();
        self.invalidate_table_cache();
        self.bounds_local = Some(msg.bounds_local);
        self.loaded_geojson = Some(msg.path);
        self.downsample_factor = msg.downsample_factor.max(1e-6);
        if self.color_mode == ObjectColorMode::ByProperty && !self.color_property_key.is_empty() {
            crate::log_warn!(
                "objects: applying deferred Color by '{}' after object load",
                self.color_property_key
            );
            self.set_color_by_property(Some(self.color_property_key.clone()));
        } else if self.color_mode == ObjectColorMode::Continuous {
            let mapping = self.color_mapping.clone();
            if let Err(error) = self.set_color_mapping(mapping) {
                self.status = error;
            }
        }
        self.apply_pending_color_value_colors();
        self.apply_pending_color_value_visibility();
        if geometry_changed {
            self.analysis_hist_focus_object_index = None;
        }
        self.pending_zoom_object_index = None;
        self.visible = true;
        if geometry_changed {
            self.geometry_generation = self.geometry_generation.wrapping_add(1).max(1);
            self.gl_object_fill.clear_id_tiles();
        }
        self.generation = self.generation.wrapping_add(1).max(1);
        let n = self.object_count();
        if geometry_changed {
            self.reset_live_analysis_selection_default();
        } else {
            self.mark_live_analysis_selection_dirty();
        }
        self.status = format!("Loaded {n} object(s).");
    }

    pub fn clear(&mut self) {
        self.control_renderer_payload_identity = None;
        self.objects = None;
        self.bins = None;
        self.render_lods = None;
        self.object_fill_mesh = None;
        self.object_selection_lods = None;
        self.point_positions_world = None;
        self.point_values = None;
        self.point_lods = None;
        self.gl_proxy_group_points.clear();
        self.object_property_keys.clear();
        self.scalar_property_keys.clear();
        self.color_property_keys.clear();
        self.property_store = ObjectPropertyStore::default();
        self.lazy_parquet_source = None;
        self.color_legend_cache = None;
        self.color_groups = None;
        self.color_groups_cache.clear();
        self.color_property_key.clear();
        self.color_mapping = ObjectColorMapping::Single;
        self.resolved_continuous_domain = None;
        self.continuous_color_payload = None;
        self.color_level_overrides_property_key.clear();
        self.color_level_overrides.clear();
        self.pending_color_value_visibility = None;
        self.pending_color_value_colors = None;
        self.color_mode = ObjectColorMode::Single;
        self.filter_clauses = vec![ObjectFilterClause::default()];
        self.filtered_ordered_indices = None;
        self.filtered_mask = None;
        self.filtered_render_lods = None;
        self.filtered_point_positions_world = None;
        self.filtered_point_values = None;
        self.filtered_point_lods = None;
        self.filtered_color_groups = None;
        self.selected_object_indices.clear();
        self.selected_object_index = None;
        self.selected_render_lods = None;
        self.primary_selected_render_lods = None;
        self.selected_fill_mesh = None;
        self.selection_fill_state = Arc::new(Vec::new());
        self.selection_cpu_overlay_dirty = false;
        self.selected_point_positions_world = None;
        self.selected_point_values = None;
        self.selected_point_lods = None;
        self.primary_selected_point_positions_world = None;
        self.primary_selected_point_values = None;
        self.visible_selected_render_cache = None;
        self.selection_generation = self.selection_generation.wrapping_add(1).max(1);
        self.clear_measurements();
        self.clear_bulk_measurements();
        self.clear_analysis();
        self.reset_object_property_analysis_cache();
        self.table_indices_cache.clear();
        self.table_cache_dirty = true;
        self.bounds_local = None;
        self.loaded_geojson = None;
        self.visible = false;
        self.display_transform = SpatialDataTransform2::default();
        self.display_mode = ObjectDisplayMode::Polygons;
        self.object_export_dialog = None;
        self.object_export_rx = None;
        self.cancel_current_load();
        self.property_load_rx = None;
        self.property_load_key = None;
        self.analysis_warm_rx = None;
        self.analysis_warm_started = false;
        self.analysis_selection_rx = None;
        self.geometry_generation = self.geometry_generation.wrapping_add(1).max(1);
        self.gl_object_fill.clear_id_tiles();
        self.generation = self.generation.wrapping_add(1).max(1);
        self.status.clear();
    }

    pub(super) fn clear_bulk_measurements(&mut self) {
        if let Some(cancel) = self.bulk_measurement_cancel.take() {
            cancel.store(true, Ordering::Relaxed);
        }
        self.bulk_measurement_rx = None;
        self.bulk_measurement_progress_completed = 0;
        self.bulk_measurement_progress_total = 0;
        self.bulk_measurement_status.clear();
    }

    pub(super) fn apply_bulk_measurement_result(&mut self, result: BulkMeasurementResult) {
        if self.objects.is_none() {
            self.bulk_measurement_status =
                "Measurements finished, but no objects are loaded.".to_string();
            return;
        }
        let metric_label = match result.metric {
            BulkMeasurementMetric::Mean => "mean-intensity",
            BulkMeasurementMetric::Median => "median-intensity",
        };
        for (column_key, values) in &result.column_values {
            self.property_store.insert_column(
                column_key.clone(),
                ObjectPropertyColumn::F64(Arc::new(
                    values.iter().map(|value| value.map(f64::from)).collect(),
                )),
            );
        }
        self.extend_object_property_keys(
            result
                .column_values
                .iter()
                .map(|(column_key, _)| column_key.as_str()),
        );
        for (column_key, _) in &result.column_values {
            match self
                .scalar_property_keys
                .binary_search_by(|existing| existing.as_str().cmp(column_key))
            {
                Ok(_) => {}
                Err(idx) => self.scalar_property_keys.insert(idx, column_key.clone()),
            }
        }
        self.reconcile_filter_clauses();
        self.analysis_channel_mapping_suggestions_cache_key = 0;
        self.analysis_channel_mapping_suggestions_cache_channels_len = 0;
        self.analysis_channel_mapping_suggestions_cache_numeric_len = 0;
        self.analysis_channel_mapping_suggestions_cache.clear();
        self.reset_object_property_analysis_cache();
        self.invalidate_table_cache();
        self.bulk_measurement_progress_completed = result.measured_count + result.failed_count;
        self.bulk_measurement_progress_total = result.object_count;
        self.bulk_measurement_status = format!(
            "Attached {} {} column(s) to {} object(s) from {} at level {} (downsample {:.2}x). Failed: {}.",
            result.column_values.len(),
            metric_label,
            result.measured_count,
            result.scope_label,
            result.level_index,
            result.level_downsample,
            result.failed_count
        );
    }

    pub fn ensure_object_property_analysis_warmup_started(
        &mut self,
        channels: &[ChannelInfo],
        selected_channel: usize,
    ) {
        if self.analysis_warm_started || self.objects.is_none() {
            return;
        }
        let numeric_columns = self.available_numeric_object_property_keys();
        if numeric_columns.is_empty() {
            self.analysis_warm_started = true;
            return;
        }

        let mut ordered_columns = Vec::with_capacity(numeric_columns.len());
        let mut seen = HashSet::new();

        let mut push_priority = |column: String| {
            if seen.insert(column.clone()) {
                ordered_columns.push(column);
            }
        };

        if let Some(channel_name) = channels
            .get(selected_channel)
            .map(|channel| channel.name.as_str())
            && let Some(column) =
                self.mapped_column_for_channel(channel_name, channels, &numeric_columns)
        {
            push_priority(column);
        }

        for channel in channels.iter().filter(|channel| channel.visible) {
            if let Some(column) =
                self.mapped_column_for_channel(channel.name.as_str(), channels, &numeric_columns)
            {
                push_priority(column);
            }
        }

        for channel in channels {
            if let Some(column) =
                self.mapped_column_for_channel(channel.name.as_str(), channels, &numeric_columns)
            {
                push_priority(column);
            }
        }

        for rule in &self.analysis_property_thresholds {
            if numeric_columns
                .iter()
                .any(|column| column == &rule.column_key)
            {
                push_priority(rule.column_key.clone());
            }
        }

        if let Some(column) = numeric_columns.get(self.analysis_hist_channel) {
            push_priority(column.to_string());
        }
        if let Some(column) = numeric_columns.get(self.analysis_scatter_x_channel) {
            push_priority(column.to_string());
        }
        if let Some(column) = numeric_columns.get(self.analysis_scatter_y_channel) {
            push_priority(column.to_string());
        }

        for column in numeric_columns {
            push_priority(column);
        }

        ordered_columns.retain(|column| !self.property_column_available_but_unloaded(column));
        self.start_object_property_analysis_warmup(ordered_columns);
    }

    pub(super) fn start_object_property_analysis_warmup(&mut self, numeric_columns: Vec<String>) {
        let Some(objects) = self.objects.as_ref().cloned() else {
            return;
        };
        let property_store = self.property_store.clone();
        self.analysis_warm_started = true;
        self.analysis_warm_request_id = self.analysis_warm_request_id.wrapping_add(1).max(1);
        let request_id = self.analysis_warm_request_id;
        let (tx, rx) = crossbeam_channel::unbounded::<AnalysisWarmupEvent>();
        self.analysis_warm_rx = Some(rx);
        self.analysis_warm_total_columns = 0;
        self.analysis_warm_completed_columns = 0;

        std::thread::Builder::new()
            .name("seg-objects-analysis-warmup".to_string())
            .spawn(move || {
                let total = numeric_columns.len();
                let _ = tx.send(AnalysisWarmupEvent::Started {
                    request_id,
                    numeric_columns: numeric_columns.clone(),
                    total,
                });

                for (column_index, key) in numeric_columns.into_iter().enumerate() {
                    let pairs_vec = if let Some(pairs) = property_store.numeric_pairs(&key) {
                        pairs
                    } else {
                        let mut pairs = Vec::new();
                        for (object_index, obj) in objects.iter().enumerate() {
                            let Some(value) =
                                obj.inline_properties.get(&key).and_then(numeric_json_value)
                            else {
                                continue;
                            };
                            if value.is_finite() {
                                pairs.push((object_index, value));
                            }
                        }
                        pairs
                    };
                    let mut pairs_vec = pairs_vec;
                    pairs_vec.retain(|(_, value)| value.is_finite());
                    let pairs = Arc::new(pairs_vec);
                    let mut sorted_pairs = pairs.as_ref().clone();
                    sorted_pairs.sort_by(|a, b| {
                        a.1.partial_cmp(&b.1)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| a.0.cmp(&b.0))
                    });
                    let sorted_pairs = Arc::new(sorted_pairs);

                    let mut histograms = Vec::new();
                    let mut levels = Vec::new();
                    for transform in [
                        HistogramValueTransform::None,
                        HistogramValueTransform::Arcsinh,
                    ] {
                        let values = pairs
                            .iter()
                            .map(|(_, value)| apply_histogram_value_transform(*value, transform))
                            .filter(|value| value.is_finite())
                            .collect::<Vec<_>>();
                        if !values.is_empty() {
                            histograms.push((
                                (key.clone(), transform),
                                compute_histogram_f32(&values, 128),
                            ));
                            for level_count in 2..=12 {
                                levels.push((
                                    (
                                        key.clone(),
                                        transform,
                                        HistogramLevelMethod::Quantiles,
                                        level_count,
                                    ),
                                    Arc::new(quantile_threshold_levels(&values, level_count)),
                                ));
                                levels.push((
                                    (
                                        key.clone(),
                                        transform,
                                        HistogramLevelMethod::KMeans,
                                        level_count,
                                    ),
                                    Arc::new(kmeans_threshold_levels(&values, level_count, 24)),
                                ));
                            }
                        }
                    }

                    let _ = tx.send(AnalysisWarmupEvent::ColumnReady {
                        request_id,
                        key,
                        pairs,
                        sorted_pairs,
                        histograms,
                        levels,
                        completed: column_index + 1,
                        total,
                    });
                }

                let _ = tx.send(AnalysisWarmupEvent::Finished { request_id });
            })
            .ok();
    }

    pub fn object_count(&self) -> usize {
        self.objects.as_ref().map(|v| v.len()).unwrap_or(0)
    }

    pub fn prepare_source_path(&mut self, path: PathBuf) -> Option<PathBuf> {
        if is_parquet_objects_path(&path) {
            self.open_geoparquet_dialog(path);
            None
        } else if is_csv_objects_path(&path) {
            self.open_csv_dialog(path);
            None
        } else {
            Some(path)
        }
    }

    pub fn choose_source_dialog(&self, default_dir: &Path) -> Option<PathBuf> {
        let start_dir = self
            .loaded_geojson
            .as_ref()
            .and_then(|p| p.parent())
            .unwrap_or(default_dir);
        FileDialog::new()
            .add_filter("GeoJSON", &["geojson", "json"])
            .add_filter("GeoParquet", &["parquet", "geoparquet"])
            .add_filter("CSV", &["csv"])
            .set_title("Open Segmentation Objects")
            .set_directory(start_dir)
            .pick_file()
    }

    pub fn supports_source_path(path: &Path) -> bool {
        is_parquet_objects_path(path)
            || is_csv_objects_path(path)
            || path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| {
                    matches!(extension.to_ascii_lowercase().as_str(), "geojson" | "json")
                })
    }

    pub(super) fn open_geoparquet_dialog(&mut self, path: PathBuf) {
        match inspect_shapes_object_schema(&path) {
            Ok(schema) => {
                if schema.geometry_candidates.is_empty() {
                    self.status =
                        "No supported binary geometry columns found in GeoParquet.".to_string();
                    return;
                }
                let x_column = preferred_xy_column(
                    &schema.numeric_property_columns,
                    &["x_centroid", "x", "x_centroid_image", "centroid_x"],
                )
                .unwrap_or_default();
                let y_column = preferred_xy_column(
                    &schema.numeric_property_columns,
                    &["y_centroid", "y", "y_centroid_image", "centroid_y"],
                )
                .unwrap_or_default();
                let selected_property_columns = schema
                    .property_columns
                    .iter()
                    .cloned()
                    .collect::<HashSet<_>>();
                self.object_load_dialog = Some(ObjectTableLoadDialog {
                    source_kind: ObjectTableSourceKind::GeoParquet,
                    path,
                    display_mode: ObjectDisplayMode::Polygons,
                    point_source: GeoParquetPointSource::Geometry,
                    geometry_column: schema.geometry_candidates[0].clone(),
                    geometry_candidates: schema.geometry_candidates,
                    geometry_search: String::new(),
                    numeric_columns: schema.numeric_property_columns,
                    x_column,
                    y_column,
                    x_search: String::new(),
                    y_search: String::new(),
                    property_columns: schema.property_columns,
                    property_search: String::new(),
                    selected_property_columns,
                });
            }
            Err(err) => {
                self.status = format!("GeoParquet schema read failed: {err}");
            }
        }
    }

    pub(super) fn open_csv_dialog(&mut self, path: PathBuf) {
        match inspect_csv_object_schema(&path) {
            Ok(schema) => {
                if schema.numeric_columns.len() < 2 {
                    self.status =
                        "CSV needs at least two numeric columns for X and Y point import."
                            .to_string();
                    return;
                }
                let x_column = preferred_xy_column(
                    &schema.numeric_columns,
                    &[
                        "x_centroid",
                        "x",
                        "x_centroid_image",
                        "centroid_x",
                        "xcoord",
                    ],
                )
                .unwrap_or_default();
                let y_column = preferred_xy_column(
                    &schema.numeric_columns,
                    &[
                        "y_centroid",
                        "y",
                        "y_centroid_image",
                        "centroid_y",
                        "ycoord",
                    ],
                )
                .unwrap_or_else(|| {
                    schema
                        .numeric_columns
                        .iter()
                        .find(|name| *name != &x_column)
                        .cloned()
                        .unwrap_or_default()
                });
                let selected_property_columns = schema
                    .property_columns
                    .iter()
                    .cloned()
                    .collect::<HashSet<_>>();
                self.object_load_dialog = Some(ObjectTableLoadDialog {
                    source_kind: ObjectTableSourceKind::Csv,
                    path,
                    display_mode: ObjectDisplayMode::Points,
                    point_source: GeoParquetPointSource::XYColumns,
                    geometry_candidates: Vec::new(),
                    geometry_column: String::new(),
                    geometry_search: String::new(),
                    numeric_columns: schema.numeric_columns,
                    x_column,
                    y_column,
                    x_search: String::new(),
                    y_search: String::new(),
                    property_columns: schema.property_columns,
                    property_search: String::new(),
                    selected_property_columns,
                });
            }
            Err(err) => {
                self.status = format!("CSV schema read failed: {err}");
            }
        }
    }

    pub fn ui_load_dialog(&mut self, ctx: &egui::Context) -> Option<ObjectUiAction> {
        let Some(mut dialog) = self.object_load_dialog.clone() else {
            return None;
        };
        let mut source_action = None;
        let mut keep_open = true;
        let mut close_requested = false;
        let mut do_load = false;
        let title = match dialog.source_kind {
            ObjectTableSourceKind::GeoParquet => "Load GeoParquet objects",
            ObjectTableSourceKind::Csv => "Load CSV objects",
        };
        egui::Window::new(title)
            .collapsible(false)
            .resizable(true)
            .default_width(520.0)
            .open(&mut keep_open)
            .show(ctx, |ui| {
                ui.label(dialog.path.file_name().and_then(|s| s.to_str()).unwrap_or(
                    match dialog.source_kind {
                        ObjectTableSourceKind::GeoParquet => "geoparquet",
                        ObjectTableSourceKind::Csv => "csv",
                    },
                ));
                ui.separator();

                if dialog.source_kind == ObjectTableSourceKind::GeoParquet {
                    ui.label("Display as");
                    ui.horizontal(|ui| {
                        ui.radio_value(
                            &mut dialog.display_mode,
                            ObjectDisplayMode::Polygons,
                            "Polygons",
                        );
                        ui.radio_value(
                            &mut dialog.display_mode,
                            ObjectDisplayMode::Points,
                            "Points",
                        );
                    });
                } else {
                    ui.label("Display as: Points");
                }

                if dialog.display_mode == ObjectDisplayMode::Points
                    && dialog.source_kind == ObjectTableSourceKind::GeoParquet
                {
                    ui.separator();
                    ui.label("Point source");
                    ui.horizontal(|ui| {
                        ui.radio_value(
                            &mut dialog.point_source,
                            GeoParquetPointSource::Geometry,
                            "Geometry column",
                        );
                        ui.radio_value(
                            &mut dialog.point_source,
                            GeoParquetPointSource::XYColumns,
                            "X/Y columns",
                        );
                    });
                }

                match dialog.point_source {
                    GeoParquetPointSource::Geometry
                        if dialog.display_mode == ObjectDisplayMode::Points
                            || dialog.display_mode == ObjectDisplayMode::Polygons =>
                    {
                        ui.separator();
                        ui.label("Geometry column");
                        ui.add(
                            egui::TextEdit::singleline(&mut dialog.geometry_search)
                                .hint_text("Search geometry columns"),
                        );
                        let geometry_candidates = fuzzy_filter_names(
                            &dialog.geometry_search,
                            &dialog.geometry_candidates,
                        );
                        egui::ScrollArea::vertical()
                            .id_salt("seg_objects_geoparquet_geometry_columns")
                            .max_height(110.0)
                            .show(ui, |ui| {
                                for name in geometry_candidates {
                                    ui.radio_value(&mut dialog.geometry_column, name.clone(), name);
                                }
                            });
                    }
                    GeoParquetPointSource::XYColumns
                        if dialog.display_mode == ObjectDisplayMode::Points =>
                    {
                        ui.separator();
                        ui.label("X column");
                        ui.add(
                            egui::TextEdit::singleline(&mut dialog.x_search)
                                .hint_text("Search numeric columns"),
                        );
                        let x_candidates =
                            fuzzy_filter_names(&dialog.x_search, &dialog.numeric_columns);
                        egui::ScrollArea::vertical()
                            .id_salt("seg_objects_geoparquet_x_columns")
                            .max_height(96.0)
                            .show(ui, |ui| {
                                for name in x_candidates {
                                    ui.radio_value(&mut dialog.x_column, name.clone(), name);
                                }
                            });

                        ui.label("Y column");
                        ui.add(
                            egui::TextEdit::singleline(&mut dialog.y_search)
                                .hint_text("Search numeric columns"),
                        );
                        let y_candidates =
                            fuzzy_filter_names(&dialog.y_search, &dialog.numeric_columns);
                        egui::ScrollArea::vertical()
                            .id_salt("seg_objects_geoparquet_y_columns")
                            .max_height(96.0)
                            .show(ui, |ui| {
                                for name in y_candidates {
                                    ui.radio_value(&mut dialog.y_column, name.clone(), name);
                                }
                            });
                    }
                    _ => {}
                }

                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Columns to load");
                    ui.label(format!(
                        "{} selected",
                        dialog.selected_property_columns.len()
                    ));
                });
                ui.add(
                    egui::TextEdit::singleline(&mut dialog.property_search)
                        .hint_text("Fuzzy search columns"),
                );
                let visible_columns =
                    fuzzy_filter_names(&dialog.property_search, &dialog.property_columns);
                ui.horizontal(|ui| {
                    if ui.button("Select visible").clicked() {
                        for name in &visible_columns {
                            dialog.selected_property_columns.insert(name.clone());
                        }
                    }
                    if ui.button("Clear visible").clicked() {
                        for name in &visible_columns {
                            dialog.selected_property_columns.remove(name);
                        }
                    }
                    if ui.button("Select all").clicked() {
                        dialog.selected_property_columns =
                            dialog.property_columns.iter().cloned().collect();
                    }
                    if ui.button("Clear all").clicked() {
                        dialog.selected_property_columns.clear();
                    }
                });
                egui::ScrollArea::vertical()
                    .id_salt("seg_objects_geoparquet_property_columns")
                    .max_height(280.0)
                    .show(ui, |ui| {
                        for name in visible_columns {
                            let mut selected = dialog.selected_property_columns.contains(&name);
                            if ui.checkbox(&mut selected, name.as_str()).changed() {
                                if selected {
                                    dialog.selected_property_columns.insert(name);
                                } else {
                                    dialog.selected_property_columns.remove(&name);
                                }
                            }
                        }
                    });

                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        close_requested = true;
                    }
                    if ui.button("Load").clicked() {
                        do_load = true;
                        close_requested = true;
                    }
                });
            });

        if close_requested {
            keep_open = false;
        }

        if do_load {
            let property_columns = dialog
                .selected_property_columns
                .iter()
                .cloned()
                .collect::<Vec<_>>();
            let loader_options = match dialog.source_kind {
                ObjectTableSourceKind::GeoParquet => {
                    match (dialog.display_mode, dialog.point_source) {
                        (ObjectDisplayMode::Polygons, _) | (_, GeoParquetPointSource::Geometry) => {
                            Some(serde_json::json!({
                                "format":"geoparquet",
                                "display_mode": match dialog.display_mode {
                                    ObjectDisplayMode::Polygons => "polygons",
                                    ObjectDisplayMode::Points => "points",
                                },
                                "source":"geometry",
                                "geometry_column":dialog.geometry_column,
                                "property_columns":property_columns,
                            }))
                        }
                        (ObjectDisplayMode::Points, GeoParquetPointSource::XYColumns) => {
                            if dialog.x_column.is_empty() || dialog.y_column.is_empty() {
                                self.status =
                                    "Choose both X and Y columns before loading point objects."
                                        .to_string();
                                None
                            } else {
                                Some(serde_json::json!({
                                    "format":"geoparquet",
                                    "display_mode":"points",
                                    "source":"xy_columns",
                                    "x_column":dialog.x_column,
                                    "y_column":dialog.y_column,
                                    "property_columns":property_columns,
                                }))
                            }
                        }
                    }
                }
                ObjectTableSourceKind::Csv => {
                    if dialog.x_column.is_empty() || dialog.y_column.is_empty() {
                        self.status =
                            "Choose both X and Y columns before loading point objects.".to_string();
                        None
                    } else {
                        Some(serde_json::json!({
                            "format":"csv",
                            "x_column":dialog.x_column,
                            "y_column":dialog.y_column,
                            "property_columns":property_columns,
                        }))
                    }
                }
            };
            if let Some(options) = loader_options {
                source_action = Some(ObjectUiAction::Load {
                    path: dialog.path.clone(),
                    options: Some(options),
                });
            }
        }

        if keep_open {
            self.object_load_dialog = Some(dialog);
        } else {
            self.object_load_dialog = None;
        }
        source_action
    }

    pub fn ui_topbar(&mut self, ui: &mut egui::Ui, default_dir: &Path) -> Option<PathBuf> {
        if ui.button("Load Seg Objects...").clicked() {
            return self.choose_source_dialog(default_dir);
        }
        None
    }

    #[cfg(test)]
    pub fn load_path(&mut self, path: PathBuf, downsample_factor: f32) {
        self.request_load(path, downsample_factor, None);
    }

    pub fn install_preloaded(&mut self, preloaded: &PreloadedObjectLayer) {
        self.cancel_current_load();
        self.load_rx = None;
        self.object_load_cancel = None;
        self.install_load_result(preloaded.result.clone());
    }

    pub fn install_control_resource(
        &mut self,
        resource: &odon::model::ControlObjectResource,
    ) -> bool {
        let payload_identity = resource.renderer_payload_identity();
        if payload_identity.is_some() && payload_identity == self.control_renderer_payload_identity
        {
            return true;
        }
        let Some(preloaded) = resource.renderer_payload::<PreloadedObjectLayer>() else {
            return false;
        };
        self.install_preloaded(preloaded);
        self.control_renderer_payload_identity = payload_identity;
        true
    }

    pub fn load_spatialdata_shapes(
        &mut self,
        path: PathBuf,
        transform: SpatialDataTransform2,
        element_name: &str,
    ) {
        self.cancel_current_load();
        self.object_load_request_id = self.object_load_request_id.wrapping_add(1).max(1);
        let request_id = self.object_load_request_id;
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_worker = cancel.clone();
        let (tx, rx) = crossbeam_channel::bounded::<Result<LoadResult, String>>(1);
        self.object_load_cancel = Some(cancel);
        self.load_rx = Some(rx);
        self.property_load_rx = None;
        self.property_load_key = None;
        self.status = format!("Loading SpatialData objects: {element_name}");

        std::thread::Builder::new()
            .name("seg-objects-spatialdata-loader".to_string())
            .spawn(move || {
                let msg = load_spatialdata_in_thread(path, transform, request_id, &cancel_worker)
                    .map_err(|error| error.to_string());
                let _ = tx.send(msg);
            })
            .ok();
    }
}
