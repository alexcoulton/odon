use super::analysis::{
    apply_histogram_value_transform, compute_histogram_f32, kmeans_threshold_levels,
    numeric_json_value, quantile_threshold_levels,
};
use super::render::{
    build_color_groups_for_property, build_object_fill_mesh, build_object_selection_render_lods,
    build_render_lods, discover_categorical_color_keys, discover_property_keys,
    discover_scalar_property_keys, hashed_color_rgb, property_scalar_value, summarize_geometry,
};
use super::*;
use arrow_array::RecordBatch;
use arrow_array::builder::{
    BinaryBuilder, BooleanBuilder, Float64Builder, Int64Builder, StringBuilder, UInt64Builder,
};
use arrow_schema::{Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use std::collections::{BTreeSet, HashMap};

// Object-layer runtime and data-loading shell.
//
// This file owns the non-render analysis-adjacent state for object layers: background loading,
// lazy property hydration, filtering/color grouping caches, selection/export state, and format
// adapters for GeoJSON, GeoParquet, CSV, and SpatialData shapes. Rendering and analysis helpers
// live in sibling modules; this file decides when those derived products are invalidated or
// rebuilt.

impl ObjectsLayer {
    fn cancel_current_load(&mut self) {
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
                    Ok(msg) => {
                        if msg.request_id != self.object_load_request_id {
                            continue;
                        }
                        self.load_rx = None;
                        self.object_load_cancel = None;
                        // A successful object reload replaces nearly every derived cache. Keep the
                        // raw loaded payload, then aggressively clear filter/selection/analysis
                        // state so no view survives that still points at the previous object set.
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
                        self.color_legend_cache = None;
                        let has_active_color_key = self
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
                            self.color_level_overrides_property_key.clear();
                            self.color_level_overrides.clear();
                        }
                        if self.filter_property_key != "id"
                            && !self
                                .scalar_property_keys
                                .iter()
                                .any(|k| k == &self.filter_property_key)
                            && !self.lazy_parquet_source.as_ref().is_some_and(|source| {
                                source
                                    .available_property_columns
                                    .iter()
                                    .any(|k| k == &self.filter_property_key)
                            })
                        {
                            self.filter_property_key = "id".to_string();
                        }
                        self.color_groups = None;
                        self.filtered_indices = None;
                        self.filtered_render_lods = None;
                        self.filtered_point_positions_world = None;
                        self.filtered_point_values = None;
                        self.filtered_point_lods = None;
                        self.filtered_color_groups = None;
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
                        self.selection_generation =
                            self.selection_generation.wrapping_add(1).max(1);
                        self.clear_measurements();
                        self.clear_bulk_measurements();
                        self.clear_analysis();
                        self.analysis_threshold_set_name = "Threshold Set".to_string();
                        self.analysis_threshold_elements.clear();
                        self.analysis_threshold_selected_element = None;
                        self.analysis_live_threshold_channel_name = None;
                        self.analysis_channel_mapping_overrides.clear();
                        self.analysis_channel_mapping_popup_open = false;
                        self.analysis_channel_mapping_search.clear();
                        self.analysis_channel_mapping_suggestions_cache_key = 0;
                        self.analysis_channel_mapping_suggestions_cache_channels_len = 0;
                        self.analysis_channel_mapping_suggestions_cache_numeric_len = 0;
                        self.analysis_channel_mapping_suggestions_cache.clear();
                        self.reset_object_property_analysis_cache();
                        self.invalidate_table_cache();
                        self.bounds_local = Some(msg.bounds_local);
                        self.loaded_geojson = Some(msg.path);
                        self.downsample_factor = msg.downsample_factor.max(1e-6);
                        if self.color_mode == ObjectColorMode::ByProperty
                            && !self.color_property_key.is_empty()
                        {
                            self.set_color_by_property(Some(self.color_property_key.clone()));
                        }
                        self.analysis_hist_focus_object_index = None;
                        self.pending_zoom_object_index = None;
                        self.visible = true;
                        self.generation = self.generation.wrapping_add(1).max(1);
                        let n = self.object_count();
                        self.status = format!("Loaded {n} object(s).");
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

        loop {
            let Some(rx) = self.object_export_rx.as_ref() else {
                break;
            };
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
                    break;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.object_export_rx = None;
                    break;
                }
            }
        }
    }

    pub fn clear(&mut self) {
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
        self.lazy_parquet_source = None;
        self.color_legend_cache = None;
        self.color_groups = None;
        self.color_property_key.clear();
        self.color_level_overrides_property_key.clear();
        self.color_level_overrides.clear();
        self.color_mode = ObjectColorMode::Single;
        self.filter_property_key = "id".to_string();
        self.filter_query.clear();
        self.filtered_indices = None;
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

    fn apply_bulk_measurement_result(&mut self, result: BulkMeasurementResult) {
        let Some(objects_arc) = self.objects.as_ref() else {
            self.bulk_measurement_status =
                "Measurements finished, but no objects are loaded.".to_string();
            return;
        };
        let metric_label = match result.metric {
            BulkMeasurementMetric::Mean => "mean-intensity",
            BulkMeasurementMetric::Median => "median-intensity",
        };
        let mut objects = objects_arc.as_ref().clone();
        for (column_key, values) in &result.column_values {
            for (idx, value) in values.iter().enumerate() {
                let Some(obj) = objects.get_mut(idx) else {
                    continue;
                };
                if let Some(value) = value
                    && let Some(number) = serde_json::Number::from_f64(*value as f64)
                {
                    obj.properties
                        .insert(column_key.clone(), serde_json::Value::Number(number));
                }
            }
        }
        self.objects = Some(Arc::new(objects));
        self.extend_object_property_keys(
            result
                .column_values
                .iter()
                .map(|(column_key, _)| column_key.as_str()),
        );
        if let Some(objects) = self.objects.as_ref() {
            self.scalar_property_keys = discover_scalar_property_keys(objects);
            self.color_property_keys = discover_categorical_color_keys(objects);
            if self.filter_property_key != "id"
                && !self
                    .scalar_property_keys
                    .iter()
                    .any(|key| key == &self.filter_property_key)
            {
                self.filter_property_key = "id".to_string();
            }
        }
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

        self.start_object_property_analysis_warmup(ordered_columns);
    }

    fn start_object_property_analysis_warmup(&mut self, numeric_columns: Vec<String>) {
        let Some(objects) = self.objects.as_ref().cloned() else {
            return;
        };
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
                    let mut pairs = Vec::new();
                    for (object_index, obj) in objects.iter().enumerate() {
                        let Some(value) = obj.properties.get(&key).and_then(numeric_json_value)
                        else {
                            continue;
                        };
                        if value.is_finite() {
                            pairs.push((object_index, value));
                        }
                    }

                    let pairs = Arc::new(pairs);
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

    pub fn open_dialog(&mut self, default_dir: &Path) {
        let start_dir = self
            .loaded_geojson
            .as_ref()
            .and_then(|p| p.parent())
            .unwrap_or(default_dir);
        if let Some(path) = FileDialog::new()
            .add_filter("GeoJSON", &["geojson", "json"])
            .add_filter("GeoParquet", &["parquet", "geoparquet"])
            .add_filter("CSV", &["csv"])
            .set_title("Open Segmentation Objects")
            .set_directory(start_dir)
            .pick_file()
        {
            if is_parquet_objects_path(&path) {
                self.open_geoparquet_dialog(path);
            } else if is_csv_objects_path(&path) {
                self.open_csv_dialog(path);
            } else {
                self.request_load(path, self.downsample_factor, None);
            }
        }
    }

    pub fn handle_dropped_path(&mut self, path: PathBuf) -> bool {
        if is_parquet_objects_path(&path) {
            self.open_geoparquet_dialog(path);
            return true;
        }
        if is_csv_objects_path(&path) {
            self.open_csv_dialog(path);
            return true;
        }
        let Some(ext) = path.extension().and_then(|s| s.to_str()) else {
            return false;
        };
        if matches!(ext.to_ascii_lowercase().as_str(), "geojson" | "json") {
            self.request_load(path, self.downsample_factor, None);
            return true;
        }
        false
    }

    fn open_geoparquet_dialog(&mut self, path: PathBuf) {
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

    fn open_csv_dialog(&mut self, path: PathBuf) {
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

    pub fn ui_load_dialog(&mut self, ctx: &egui::Context) {
        let Some(mut dialog) = self.object_load_dialog.clone() else {
            return;
        };
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
            let property_columns = Some(
                dialog
                    .selected_property_columns
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>(),
            );
            let load_options = match dialog.source_kind {
                ObjectTableSourceKind::GeoParquet => {
                    match (dialog.display_mode, dialog.point_source) {
                        (ObjectDisplayMode::Polygons, _) | (_, GeoParquetPointSource::Geometry) => {
                            Some(ObjectLoadOptions::Parquet(ObjectParquetLoadOptions {
                                display_mode: dialog.display_mode,
                                source: ObjectParquetSource::Geometry(ShapesLoadOptions {
                                    transform: SpatialDataTransform2::default(),
                                    geometry_column: dialog.geometry_column.clone(),
                                    property_columns,
                                }),
                            }))
                        }
                        (ObjectDisplayMode::Points, GeoParquetPointSource::XYColumns) => {
                            if dialog.x_column.is_empty() || dialog.y_column.is_empty() {
                                self.status =
                                    "Choose both X and Y columns before loading point objects."
                                        .to_string();
                                None
                            } else {
                                Some(ObjectLoadOptions::Parquet(ObjectParquetLoadOptions {
                                    display_mode: ObjectDisplayMode::Points,
                                    source: ObjectParquetSource::XYColumns {
                                        x_column: dialog.x_column.clone(),
                                        y_column: dialog.y_column.clone(),
                                        property_columns,
                                    },
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
                        Some(ObjectLoadOptions::Csv(ObjectCsvLoadOptions {
                            x_column: dialog.x_column.clone(),
                            y_column: dialog.y_column.clone(),
                            property_columns,
                        }))
                    }
                }
            };
            if let Some(load_options) = load_options {
                self.request_load(
                    dialog.path.clone(),
                    self.downsample_factor,
                    Some(load_options),
                );
            }
        }

        if keep_open {
            self.object_load_dialog = Some(dialog);
        } else {
            self.object_load_dialog = None;
        }
    }

    pub fn ui_topbar(&mut self, ui: &mut egui::Ui, default_dir: &Path) {
        if ui.button("Load Seg Objects...").clicked() {
            self.open_dialog(default_dir);
        }
    }

    pub fn load_path(&mut self, path: PathBuf, downsample_factor: f32) {
        self.request_load(path, downsample_factor, None);
    }

    pub fn load_objects_with_transform(
        &mut self,
        path: PathBuf,
        downsample_factor: f32,
        display_transform: SpatialDataTransform2,
    ) {
        self.cancel_current_load();
        self.object_load_request_id = self.object_load_request_id.wrapping_add(1).max(1);
        let request_id = self.object_load_request_id;
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_worker = cancel.clone();
        let (tx, rx) = crossbeam_channel::bounded::<LoadResult>(1);
        self.object_load_cancel = Some(cancel);
        self.load_rx = Some(rx);
        self.property_load_rx = None;
        self.property_load_key = None;
        self.status = format!("Loading objects: {}", path.to_string_lossy());

        std::thread::Builder::new()
            .name("seg-objects-transform-loader".to_string())
            .spawn(move || {
                let load_options = if is_parquet_objects_path(&path) {
                    minimal_parquet_load_options(&path)
                        .ok()
                        .map(ObjectLoadOptions::Parquet)
                } else {
                    None
                };
                let msg = load_in_thread(
                    path,
                    downsample_factor,
                    load_options,
                    request_id,
                    &cancel_worker,
                )
                .map(|mut msg| {
                    msg.display_transform = display_transform;
                    msg
                });
                if let Ok(msg) = msg {
                    let _ = tx.send(msg);
                }
            })
            .ok();
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
        let (tx, rx) = crossbeam_channel::bounded::<LoadResult>(1);
        self.object_load_cancel = Some(cancel);
        self.load_rx = Some(rx);
        self.property_load_rx = None;
        self.property_load_key = None;
        self.status = format!("Loading SpatialData objects: {element_name}");

        std::thread::Builder::new()
            .name("seg-objects-spatialdata-loader".to_string())
            .spawn(move || {
                if let Ok(msg) =
                    load_spatialdata_in_thread(path, transform, request_id, &cancel_worker)
                {
                    let _ = tx.send(msg);
                }
            })
            .ok();
    }

    pub fn ui_properties(&mut self, ui: &mut egui::Ui, default_dir: &Path) {
        self.ensure_filter_cache();
        self.ensure_color_groups();

        ui.horizontal(|ui| {
            ui.checkbox(&mut self.visible, "Visible");
            self.ui_topbar(ui, default_dir);
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
                if let Some(path) = self.loaded_geojson.clone() {
                    self.request_load(path, self.downsample_factor, None);
                }
            }
            if ui
                .add_enabled(self.loaded_geojson.is_some(), egui::Button::new("Clear"))
                .clicked()
            {
                self.clear();
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
            egui::ComboBox::from_id_salt("seg_objects_filter_key")
                .selected_text(self.filter_property_key.clone())
                .show_ui(ui, |ui| {
                    filter_changed |= ui
                        .selectable_value(&mut self.filter_property_key, "id".to_string(), "id")
                        .changed();
                    for key in &self.scalar_property_keys {
                        filter_changed |= ui
                            .selectable_value(&mut self.filter_property_key, key.clone(), key)
                            .changed();
                    }
                });
            filter_changed |= ui
                .add(
                    egui::TextEdit::singleline(&mut self.filter_query)
                        .hint_text("contains...")
                        .desired_width(180.0),
                )
                .changed();
            if ui.button("Clear").clicked() {
                self.filter_property_key = "id".to_string();
                self.filter_query.clear();
                filter_changed = true;
            }
        });
        if filter_changed {
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
                    let mut props = obj
                        .properties
                        .iter()
                        .map(|(key, value)| (key.clone(), value_to_display_text(value)))
                        .collect::<Vec<_>>();
                    props.sort_by(|a, b| a.0.cmp(&b.0));
                    (
                        obj.id.clone(),
                        obj.area_px,
                        obj.perimeter_px,
                        obj.centroid_world,
                        props,
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
    }

    pub fn has_data(&self) -> bool {
        self.objects.as_ref().is_some_and(|v| !v.is_empty())
    }

    pub fn set_display_transform(&mut self, display_transform: SpatialDataTransform2) {
        self.display_transform = display_transform;
    }

    pub fn is_busy(&self) -> bool {
        self.is_loading()
            || self.is_property_loading()
            || self.is_analyzing()
            || self.is_bulk_measuring()
    }

    pub fn is_loading(&self) -> bool {
        self.load_rx.is_some()
    }

    pub fn is_property_loading(&self) -> bool {
        self.property_load_rx.is_some()
    }

    pub fn status(&self) -> &str {
        &self.status
    }

    pub fn selected_object_index(&self) -> Option<usize> {
        self.selected_object_index
    }

    pub fn available_property_columns(&self) -> &[String] {
        self.lazy_parquet_source
            .as_ref()
            .map(|source| source.available_property_columns.as_slice())
            .unwrap_or(self.color_property_keys.as_slice())
    }

    pub fn active_color_legend_entries(&mut self) -> Option<Vec<ObjectColorLegendEntry>> {
        if self.color_mode != ObjectColorMode::ByProperty || self.color_property_key.is_empty() {
            return None;
        }
        if let Some(cache) = self.color_legend_cache.as_ref()
            && cache.property_key == self.color_property_key
            && cache.generation == self.generation
        {
            return Some(cache.entries.clone());
        }

        use std::collections::BTreeMap;

        let objects = self.objects.as_ref()?;
        let mut counts = BTreeMap::<String, usize>::new();
        if let Some(filtered_indices) = self.filtered_indices.as_ref() {
            for idx in filtered_indices {
                let Some(obj) = objects.get(*idx) else {
                    continue;
                };
                let Some(value_label) = obj
                    .properties
                    .get(&self.color_property_key)
                    .and_then(property_scalar_value)
                else {
                    continue;
                };
                *counts.entry(value_label).or_default() += 1;
            }
        } else {
            for obj in objects.iter() {
                let Some(value_label) = obj
                    .properties
                    .get(&self.color_property_key)
                    .and_then(property_scalar_value)
                else {
                    continue;
                };
                *counts.entry(value_label).or_default() += 1;
            }
        }

        if counts.is_empty() {
            return None;
        }

        let entries = counts
            .into_iter()
            .map(|(value_label, count)| ObjectColorLegendEntry {
                color_rgb: hashed_color_rgb(&self.color_property_key, &value_label),
                count,
                value_label,
            })
            .collect::<Vec<_>>();
        self.color_legend_cache = Some(ObjectColorLegendCache {
            property_key: self.color_property_key.clone(),
            generation: self.generation,
            entries: entries.clone(),
        });
        Some(entries)
    }

    pub fn set_color_by_property(&mut self, property_key: Option<String>) {
        let (next_mode, next_key) = match property_key {
            Some(property_key) if !property_key.is_empty() => {
                (ObjectColorMode::ByProperty, property_key)
            }
            _ => (ObjectColorMode::Single, String::new()),
        };
        let needs_property_load = next_mode == ObjectColorMode::ByProperty
            && self.property_column_available_but_unloaded(next_key.as_str());
        if self.color_mode == next_mode
            && self.color_property_key == next_key
            && (!needs_property_load
                || self.property_load_key.as_deref() == Some(next_key.as_str()))
        {
            return;
        }
        self.color_mode = next_mode;
        self.color_property_key = next_key;
        if self.color_mode == ObjectColorMode::ByProperty {
            let key = self.color_property_key.clone();
            self.ensure_property_loaded(key.as_str());
            self.reconcile_active_color_property();
        }
        self.color_groups = None;
        self.filtered_color_groups = None;
        self.color_legend_cache = None;
    }

    pub(crate) fn project_display_state(&self) -> ObjectProjectDisplayState {
        let color_property_key = (self.color_mode == ObjectColorMode::ByProperty)
            .then(|| self.color_property_key.clone())
            .filter(|key| !key.is_empty());
        let color_level_overrides = if color_property_key.as_deref()
            == Some(self.color_level_overrides_property_key.as_str())
        {
            self.color_level_overrides.clone()
        } else {
            BTreeMap::new()
        };
        ObjectProjectDisplayState {
            color_property_key,
            color_level_overrides,
            fill_cells: self.fill_cells,
            fill_opacity: self.fill_opacity,
            selected_fill_opacity: self.selected_fill_opacity,
        }
    }

    pub(crate) fn apply_project_display_state(&mut self, state: &ObjectProjectDisplayState) {
        self.set_color_by_property(state.color_property_key.clone());
        self.color_level_overrides_property_key =
            state.color_property_key.clone().unwrap_or_default();
        self.color_level_overrides = state.color_level_overrides.clone();
        self.fill_cells = state.fill_cells;
        self.fill_opacity = state.fill_opacity.clamp(0.0, 1.0);
        self.selected_fill_opacity = state.selected_fill_opacity.clamp(0.0, 1.0);
        self.color_groups = None;
        self.filtered_color_groups = None;
        self.color_legend_cache = None;
    }

    pub(crate) fn clear_project_display_state(&mut self) {
        self.set_color_by_property(None);
        self.color_level_overrides_property_key.clear();
        self.color_level_overrides.clear();
        self.fill_cells = false;
        self.fill_opacity = 0.30;
        self.selected_fill_opacity = 0.70;
        self.color_groups = None;
        self.filtered_color_groups = None;
        self.color_legend_cache = None;
    }

    pub fn set_color_level_overrides(
        &mut self,
        property_key: Option<&str>,
        overrides: &std::collections::HashMap<String, ObjectColorLevelOverride>,
    ) {
        let property_key = property_key.unwrap_or_default();
        self.color_level_overrides_property_key = property_key.to_string();
        self.color_level_overrides.clear();
        if !property_key.is_empty() {
            self.color_level_overrides
                .extend(overrides.iter().map(|(key, value)| (key.clone(), *value)));
        }
    }

    fn property_column_available_but_unloaded(&self, property_key: &str) -> bool {
        self.lazy_parquet_source.as_ref().is_some_and(|source| {
            source
                .available_property_columns
                .iter()
                .any(|key| key == property_key)
                && !source.loaded_property_columns.contains(property_key)
        })
    }

    fn ensure_property_loaded(&mut self, property_key: &str) {
        let Some(source) = self.lazy_parquet_source.as_ref() else {
            return;
        };
        if source.loaded_property_columns.contains(property_key) {
            return;
        }
        if !source
            .available_property_columns
            .iter()
            .any(|key| key == property_key)
        {
            return;
        }
        let Some(path) = self.loaded_geojson.as_ref() else {
            return;
        };
        if self.property_load_key.as_deref() == Some(property_key)
            && self.property_load_rx.is_some()
        {
            return;
        };

        let geometry_column = source.geometry_column.clone();
        let property_key_owned = property_key.to_string();
        let path = path.clone();
        let (tx, rx) = crossbeam_channel::bounded::<PropertyLoadResult>(1);
        self.property_load_rx = Some(rx);
        self.property_load_key = Some(property_key_owned.clone());
        self.status = format!("Loading property '{property_key}'...");

        std::thread::Builder::new()
            .name(format!("seg-objects-property-loader-{property_key}"))
            .spawn(move || {
                if let Ok(values_by_row) = load_parquet_property_values_for_loaded_objects(
                    &path,
                    geometry_column.as_str(),
                    property_key_owned.as_str(),
                ) {
                    let _ = tx.send(PropertyLoadResult {
                        property_key: property_key_owned,
                        values_by_row,
                    });
                }
            })
            .ok();
    }

    fn apply_loaded_property_values(
        &mut self,
        property_key: &str,
        property_values: &HashMap<usize, serde_json::Value>,
    ) {
        {
            let Some(objects_arc) = self.objects.as_mut() else {
                return;
            };
            let objects = Arc::make_mut(objects_arc);
            for obj in objects.iter_mut() {
                let Some(row_index) = obj.source_row_index else {
                    continue;
                };
                if let Some(value) = property_values.get(&row_index) {
                    obj.properties
                        .insert(property_key.to_string(), value.clone());
                } else {
                    obj.properties.remove(property_key);
                }
            }
        };
        if let Some(source) = self.lazy_parquet_source.as_mut() {
            source
                .loaded_property_columns
                .insert(property_key.to_string());
        }
        self.extend_object_property_keys(std::iter::once(property_key));
        let Some(objects) = self.objects.as_ref() else {
            return;
        };
        self.scalar_property_keys = discover_scalar_property_keys(objects);
        self.color_property_keys = discover_categorical_color_keys(objects);
        self.color_legend_cache = None;
        self.color_groups = None;
        self.invalidate_filter_cache();
        self.reset_object_property_analysis_cache();
        self.generation = self.generation.wrapping_add(1).max(1);
        self.reconcile_active_color_property();
        let n = self.object_count();
        self.status = format!("Loaded {n} object(s).");
    }

    fn reconcile_active_color_property(&mut self) {
        if self.color_mode != ObjectColorMode::ByProperty || self.color_property_key.is_empty() {
            return;
        }
        if self
            .color_property_keys
            .iter()
            .any(|loaded| loaded == &self.color_property_key)
        {
            return;
        }
        if self.property_column_available_but_unloaded(self.color_property_key.as_str())
            || self.property_load_key.as_deref() == Some(self.color_property_key.as_str())
        {
            return;
        }
        self.status = format!(
            "Property '{}' has too many distinct values for Color by.",
            self.color_property_key
        );
        self.color_mode = ObjectColorMode::Single;
        self.color_property_key.clear();
    }

    pub fn selected_object_details(
        &self,
        local_to_world_offset: egui::Vec2,
    ) -> Option<SelectedObjectDetails> {
        let idx = self.selected_object_index?;
        let obj = self.objects.as_ref()?.get(idx)?;
        let mut properties = obj
            .properties
            .iter()
            .map(|(key, value)| (key.clone(), value_to_display_text(value)))
            .collect::<Vec<_>>();
        properties.sort_by(|a, b| a.0.cmp(&b.0));
        let scale = egui::vec2(
            self.display_transform.scale[0].max(1e-6),
            self.display_transform.scale[1].max(1e-6),
        );
        let offset = egui::vec2(
            local_to_world_offset.x + self.display_transform.translation[0],
            local_to_world_offset.y + self.display_transform.translation[1],
        );
        Some(SelectedObjectDetails {
            id: obj.id.clone(),
            area_px: obj.area_px,
            perimeter_px: obj.perimeter_px,
            centroid_world: egui::pos2(
                obj.centroid_world.x * scale.x + offset.x,
                obj.centroid_world.y * scale.y + offset.y,
            ),
            properties,
        })
    }

    pub fn select_objects_by_ids(&mut self, ids: &std::collections::HashSet<String>) -> usize {
        let Some(objects) = self.objects.as_ref() else {
            self.clear_selection();
            return 0;
        };

        self.selected_object_indices.clear();
        for (idx, obj) in objects.iter().enumerate() {
            let mut matched = ids.contains(&obj.id);
            if !matched {
                for key in ["cell_id", "id", "object_id", "label", "name"] {
                    if let Some(value) = obj.properties.get(key).and_then(value_to_short_string) {
                        if ids.contains(&value) {
                            matched = true;
                            break;
                        }
                    }
                }
            }
            if matched {
                self.selected_object_indices.insert(idx);
            }
        }
        self.selected_object_index = self.selected_object_indices.iter().next().copied();
        self.rebuild_selection_render_lods();
        self.clear_measurements();
        self.invalidate_table_cache();
        self.selected_object_indices.len()
    }

    fn current_selection_object_ids(&self) -> Vec<String> {
        let Some(objects) = self.objects.as_ref() else {
            return Vec::new();
        };
        let mut indices = self
            .selected_object_indices
            .iter()
            .copied()
            .collect::<Vec<_>>();
        indices.sort_unstable();
        indices
            .into_iter()
            .filter_map(|idx| objects.get(idx).map(|obj| obj.id.clone()))
            .collect()
    }

    fn create_selection_element_from_current_selection(&mut self) -> usize {
        let object_ids = self.current_selection_object_ids();
        if object_ids.is_empty() {
            return 0;
        }
        let next_idx = self.selection_elements.len() + 1;
        let name = if self.selection_element_name_draft.trim().is_empty() {
            format!("Selection Element {next_idx}")
        } else {
            self.selection_element_name_draft.trim().to_string()
        };
        self.selection_elements
            .push(SelectionElement { name, object_ids });
        self.selection_element_selected = Some(self.selection_elements.len() - 1);
        self.selection_element_name_draft =
            format!("Selection Element {}", self.selection_elements.len() + 1);
        self.selection_elements
            .last()
            .map(|element| element.object_ids.len())
            .unwrap_or(0)
    }

    fn create_selection_element_from_ids(
        &mut self,
        name: Option<String>,
        object_ids: Vec<String>,
    ) -> usize {
        if object_ids.is_empty() {
            return 0;
        }
        let next_idx = self.selection_elements.len() + 1;
        let name = name
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| format!("Selection Element {next_idx}"));
        self.selection_elements
            .push(SelectionElement { name, object_ids });
        self.selection_element_selected = Some(self.selection_elements.len() - 1);
        self.selection_element_name_draft =
            format!("Selection Element {}", self.selection_elements.len() + 1);
        self.selection_elements
            .last()
            .map(|element| element.object_ids.len())
            .unwrap_or(0)
    }

    fn select_selection_element(&mut self, idx: usize) -> usize {
        let Some(element) = self.selection_elements.get(idx) else {
            return 0;
        };
        let ids = element.object_ids.iter().cloned().collect::<HashSet<_>>();
        self.selection_element_selected = Some(idx);
        self.select_objects_by_ids(&ids)
    }

    pub fn selection_elements_snapshot(&self) -> Vec<(usize, String, usize)> {
        self.selection_elements
            .iter()
            .enumerate()
            .map(|(idx, element)| (idx, element.name.clone(), element.object_ids.len()))
            .collect()
    }

    pub fn create_selection_element_from_current_selection_with_name(
        &mut self,
        name: Option<String>,
    ) -> usize {
        let object_ids = self.current_selection_object_ids();
        let count = self.create_selection_element_from_ids(name, object_ids);
        if count > 0 {
            self.status = format!("Saved selection element with {count} object(s).");
        }
        count
    }

    pub fn add_current_selection_to_element(&mut self, idx: usize) -> usize {
        let selected_ids = self.current_selection_object_ids();
        let Some(element) = self.selection_elements.get_mut(idx) else {
            return 0;
        };
        let mut existing = element.object_ids.iter().cloned().collect::<HashSet<_>>();
        let mut added = 0usize;
        for id in selected_ids {
            if existing.insert(id.clone()) {
                element.object_ids.push(id);
                added += 1;
            }
        }
        self.selection_element_selected = Some(idx);
        self.status = format!("Added {added} object(s) to selection element.");
        added
    }

    pub(super) fn ui_selection_elements_editor(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("Selection Elements", |ui| {
            ui.horizontal(|ui| {
                ui.label("Name");
                ui.add(
                    egui::TextEdit::singleline(&mut self.selection_element_name_draft)
                        .desired_width(180.0),
                );
                if ui
                    .add_enabled(
                        self.selection_count() > 0,
                        egui::Button::new("New from selection"),
                    )
                    .clicked()
                {
                    let count = self.create_selection_element_from_current_selection();
                    if count > 0 {
                        self.status = format!("Saved selection element with {count} object(s).");
                    }
                }
            });
            if self.selection_elements.is_empty() {
                ui.label("No saved selection elements.");
            } else {
                let mut clicked_idx = None;
                let mut delete_idx = None;
                egui::ScrollArea::vertical()
                    .id_salt("seg_objects_selection_elements")
                    .max_height(180.0)
                    .show(ui, |ui| {
                        for (idx, element) in self.selection_elements.iter_mut().enumerate() {
                            ui.horizontal(|ui| {
                                let selected = self.selection_element_selected == Some(idx);
                                let label =
                                    format!("{} ({})", element.name, element.object_ids.len());
                                if ui.selectable_label(selected, label).clicked() {
                                    clicked_idx = Some(idx);
                                }
                                ui.add(
                                    egui::TextEdit::singleline(&mut element.name)
                                        .desired_width(140.0),
                                );
                                if ui.button("Delete").clicked() {
                                    delete_idx = Some(idx);
                                }
                            });
                        }
                    });
                if let Some(idx) = clicked_idx {
                    let count = self.select_selection_element(idx);
                    self.status = format!("Selected {count} object(s) from saved element.");
                }
                if let Some(idx) = delete_idx {
                    self.selection_elements.remove(idx);
                    self.selection_element_selected = match self.selection_elements.is_empty() {
                        true => None,
                        false => self
                            .selection_element_selected
                            .map(|selected| selected.min(self.selection_elements.len() - 1)),
                    };
                }
            }
        });
    }

    pub fn is_analyzing(&self) -> bool {
        false
    }

    fn request_load(
        &mut self,
        path: PathBuf,
        downsample_factor: f32,
        load_options: Option<ObjectLoadOptions>,
    ) {
        // The worker thread produces a single fully prepared `LoadResult` that already contains
        // the expensive derived structures (bins, LODs, point payloads). `tick` then installs it
        // atomically on the UI thread.
        self.cancel_current_load();
        self.object_load_request_id = self.object_load_request_id.wrapping_add(1).max(1);
        let request_id = self.object_load_request_id;
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_worker = cancel.clone();
        let (tx, rx) = crossbeam_channel::bounded::<LoadResult>(1);
        self.object_load_cancel = Some(cancel);
        self.load_rx = Some(rx);
        self.property_load_rx = None;
        self.property_load_key = None;
        self.status = format!("Loading objects: {}", path.to_string_lossy());

        std::thread::Builder::new()
            .name("seg-objects-loader".to_string())
            .spawn(move || {
                if let Ok(msg) = load_in_thread(
                    path,
                    downsample_factor,
                    load_options,
                    request_id,
                    &cancel_worker,
                ) {
                    let _ = tx.send(msg);
                }
            })
            .ok();
    }

    fn invalidate_filter_cache(&mut self) {
        self.filtered_indices = None;
        self.filtered_render_lods = None;
        self.filtered_point_positions_world = None;
        self.filtered_point_values = None;
        self.filtered_point_lods = None;
        self.filtered_color_groups = None;
        self.color_legend_cache = None;
        self.mark_live_analysis_selection_dirty();
        self.invalidate_object_property_analysis_cache();
        self.invalidate_table_cache();
    }

    pub(super) fn ensure_filter_cache(&mut self) {
        if self.filter_query.trim().is_empty() {
            if self.filtered_indices.is_some()
                || self.filtered_render_lods.is_some()
                || self.filtered_color_groups.is_some()
            {
                self.invalidate_filter_cache();
            }
            return;
        }
        if self.filtered_indices.is_some() {
            return;
        }
        let Some(objects) = self.objects.as_ref() else {
            return;
        };

        // Filtering materializes a subset snapshot plus the derived render/point/color products
        // for that subset. The rest of the layer reads from these caches instead of re-evaluating
        // the filter predicate on every paint or analysis pass.
        let mut indices = HashSet::new();
        let mut subset = Vec::new();
        for (idx, obj) in objects.iter().enumerate() {
            if !self.object_matches_filter(obj) {
                continue;
            }
            indices.insert(idx);
            subset.push(obj.clone());
        }

        self.filtered_indices = Some(indices.clone());
        if subset.is_empty() {
            self.filtered_render_lods = None;
            self.filtered_point_positions_world = None;
            self.filtered_point_values = None;
            self.filtered_point_lods = None;
            self.filtered_color_groups = None;
        } else {
            self.filtered_render_lods = build_render_lods(&subset).ok();
            let (positions, values, lods) =
                build_object_point_payload(&subset, self.display_transform);
            self.filtered_point_positions_world = Some(positions);
            self.filtered_point_values = Some(values);
            self.filtered_point_lods = Some(lods);
            self.filtered_color_groups = if self.color_mode == ObjectColorMode::ByProperty
                && !self.color_property_key.is_empty()
            {
                build_color_groups_for_property(
                    objects
                        .iter()
                        .enumerate()
                        .filter(|(idx, _)| indices.contains(idx)),
                    &self.color_property_key,
                )
                .ok()
            } else {
                None
            };
        }

        self.selected_object_indices
            .retain(|idx| indices.contains(idx));
        self.rebuild_selection_render_lods();
        self.mark_live_analysis_selection_dirty();
        self.invalidate_table_cache();
    }

    pub(super) fn ensure_color_groups(&mut self) {
        // Color grouping is lazily built against either the full set or the filtered subset,
        // depending on which view is currently active. This keeps legend/group generation aligned
        // with what the user actually sees.
        if self.filtered_indices.is_some() {
            if self.color_mode != ObjectColorMode::ByProperty || self.color_property_key.is_empty()
            {
                self.filtered_color_groups = None;
                return;
            }
            if self
                .filtered_color_groups
                .as_ref()
                .is_some_and(|g| g.property_key == self.color_property_key)
            {
                return;
            }
            self.ensure_filter_cache();
            let (Some(objects), Some(filtered_indices)) =
                (self.objects.as_ref(), self.filtered_indices.as_ref())
            else {
                self.filtered_color_groups = None;
                return;
            };
            self.filtered_color_groups = build_color_groups_for_property(
                objects
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| filtered_indices.contains(idx)),
                &self.color_property_key,
            )
            .ok();
            return;
        }
        if self.color_mode != ObjectColorMode::ByProperty || self.color_property_key.is_empty() {
            return;
        }
        if self
            .color_groups
            .as_ref()
            .is_some_and(|g| g.property_key == self.color_property_key)
        {
            return;
        }
        let Some(objects) = self.objects.as_ref() else {
            return;
        };
        self.color_groups =
            build_color_groups_for_property(objects.iter().enumerate(), &self.color_property_key)
                .ok();
    }

    pub(super) fn active_color_groups(&self) -> Option<&ObjectColorGroups> {
        self.filtered_color_groups
            .as_ref()
            .or(self.color_groups.as_ref())
    }

    pub(super) fn has_active_filter(&self) -> bool {
        !self.filter_query.trim().is_empty()
    }

    fn object_matches_filter(&self, obj: &GeoJsonObjectFeature) -> bool {
        // Filtering is intentionally simple substring matching over a single projected display
        // field. The expensive part is not the predicate itself, but rebuilding the derived subset
        // state once the predicate changes.
        let needle = self.filter_query.trim();
        if needle.is_empty() {
            return true;
        }
        let haystack = if self.filter_property_key == "id" {
            obj.id.clone()
        } else {
            obj.properties
                .get(&self.filter_property_key)
                .map(value_to_display_text)
                .unwrap_or_default()
        };
        haystack
            .to_ascii_lowercase()
            .contains(&needle.to_ascii_lowercase())
    }

    pub(super) fn is_index_visible(&self, idx: usize) -> bool {
        self.filtered_indices
            .as_ref()
            .map(|set| set.contains(&idx))
            .unwrap_or(true)
    }

    fn selection_summary(&self) -> (usize, f32, f32) {
        let Some(objects) = self.objects.as_ref() else {
            return (0, 0.0, 0.0);
        };
        let mut count = 0usize;
        let mut total = 0.0f32;
        for idx in &self.selected_object_indices {
            if let Some(obj) = objects.get(*idx) {
                count += 1;
                total += obj.area_px;
            }
        }
        let mean = if count > 0 { total / count as f32 } else { 0.0 };
        (count, total, mean)
    }

    fn table_indices(&mut self) -> &[usize] {
        if !self.table_cache_dirty {
            return &self.table_indices_cache;
        }
        let mut out = if let Some(filtered) = self.filtered_indices.as_ref() {
            filtered.iter().copied().collect::<Vec<_>>()
        } else if let Some(objects) = self.objects.as_ref() {
            (0..objects.len()).collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        out.sort_by(|a, b| {
            let sel_a = self.selected_object_indices.contains(a);
            let sel_b = self.selected_object_indices.contains(b);
            sel_b.cmp(&sel_a).then_with(|| a.cmp(b))
        });
        self.table_indices_cache = out;
        self.table_cache_dirty = false;
        &self.table_indices_cache
    }

    fn export_selected_with_dialog(&mut self, default_dir: &Path) -> anyhow::Result<()> {
        let indices = self
            .selected_object_indices
            .iter()
            .copied()
            .collect::<Vec<_>>();
        self.export_indices_with_dialog(default_dir, "seg_objects_selected.geojson", &indices)
    }

    fn export_filtered_with_dialog(&mut self, default_dir: &Path) -> anyhow::Result<()> {
        let indices = if let Some(filtered) = self.filtered_indices.as_ref() {
            filtered.iter().copied().collect::<Vec<_>>()
        } else if let Some(objects) = self.objects.as_ref() {
            (0..objects.len()).collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        self.export_indices_with_dialog(default_dir, "seg_objects_filtered.geojson", &indices)
    }

    fn export_indices_with_dialog(
        &mut self,
        default_dir: &Path,
        default_name: &str,
        indices: &[usize],
    ) -> anyhow::Result<()> {
        if indices.is_empty() {
            anyhow::bail!("no objects to export");
        }
        let start_dir = self
            .loaded_geojson
            .as_ref()
            .and_then(|p| p.parent())
            .unwrap_or(default_dir);
        let Some(path) = FileDialog::new()
            .add_filter("GeoJSON", &["geojson", "json"])
            .set_title("Export Segmentation Objects")
            .set_directory(start_dir)
            .set_file_name(default_name)
            .save_file()
        else {
            return Ok(());
        };
        save_geojson_objects(
            path.as_path(),
            self.objects.as_deref().map_or(&[], |v| v),
            indices,
        )?;
        self.status = format!(
            "Exported {} object(s) to {}",
            indices.len(),
            path.to_string_lossy()
        );
        Ok(())
    }

    pub fn export_objects_geoparquet_with_dialog(&mut self) -> anyhow::Result<()> {
        self.open_object_export_dialog(ObjectExportFormat::GeoParquet)
    }

    pub fn export_objects_csv_with_dialog(&mut self) -> anyhow::Result<()> {
        let Some(objects) = self.objects.as_ref() else {
            anyhow::bail!("no objects loaded");
        };
        if objects.is_empty() {
            anyhow::bail!("no objects loaded");
        }
        self.open_object_export_dialog(ObjectExportFormat::Csv)
    }

    fn default_object_export_dir(&self) -> &Path {
        self.loaded_geojson
            .as_deref()
            .and_then(Path::parent)
            .unwrap_or_else(|| Path::new("."))
    }

    fn default_object_export_stem(&self) -> String {
        let fallback = "seg_objects".to_string();
        let Some(path) = self.loaded_geojson.as_ref() else {
            return fallback;
        };
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            return fallback;
        };
        let trimmed = name
            .strip_suffix(".geoparquet")
            .or_else(|| name.strip_suffix(".parquet"))
            .or_else(|| name.strip_suffix(".geojson"))
            .or_else(|| name.strip_suffix(".json"))
            .or_else(|| name.strip_suffix(".csv"))
            .unwrap_or(name);
        let trimmed = trimmed.trim_matches('.');
        if trimmed.is_empty() {
            fallback
        } else {
            trimmed.to_string()
        }
    }

    fn export_objects_csv(
        snapshot: &ObjectExportSnapshot,
        path: &Path,
        selected_columns: &HashSet<String>,
    ) -> anyhow::Result<()> {
        // CSV export writes the normalized export table directly: one logical object row and one
        // scalar column per exported property. Geometry is omitted here in favor of tabular tools.
        let table =
            Self::build_object_export_table_from_snapshot(snapshot, selected_columns, false)?;
        let mut writer = csv::Writer::from_path(path)
            .with_context(|| format!("failed to create CSV: {}", path.to_string_lossy()))?;
        let headers = table
            .columns
            .iter()
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>();
        writer.write_record(headers)?;
        for row_idx in 0..table.row_count {
            let row = table
                .columns
                .iter()
                .map(|column| {
                    export_scalar_to_csv(
                        column.values.get(row_idx).and_then(|value| value.as_ref()),
                    )
                })
                .collect::<Vec<_>>();
            writer.write_record(row)?;
        }
        writer.flush()?;
        Ok(())
    }

    fn export_objects_geoparquet(
        snapshot: &ObjectExportSnapshot,
        path: &Path,
        selected_columns: &HashSet<String>,
    ) -> anyhow::Result<()> {
        // GeoParquet export shares the same normalized property table as CSV, but prefixes it with
        // a WKB geometry column so downstream spatial tools can preserve shape information.
        let table =
            Self::build_object_export_table_from_snapshot(snapshot, selected_columns, true)?;
        let mut fields = Vec::with_capacity(table.columns.len() + 1);
        let mut arrays = Vec::with_capacity(table.columns.len() + 1);

        fields.push(Field::new(
            "geometry",
            arrow_schema::DataType::Binary,
            false,
        ));
        let mut geometry_builder = BinaryBuilder::new();
        for geom in &table.geometry_wkb {
            geometry_builder.append_value(geom);
        }
        arrays.push(Arc::new(geometry_builder.finish()) as arrow_array::ArrayRef);

        for column in &table.columns {
            let (field, array) = export_column_to_arrow_array(column)?;
            fields.push(field);
            arrays.push(array);
        }

        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;

        let geometry_types_json = table
            .geometry_types
            .iter()
            .map(|name| format!("\"{name}\""))
            .collect::<Vec<_>>()
            .join(",");
        let geo_metadata = format!(
            "{{\"version\":\"1.0.0\",\"primary_column\":\"geometry\",\"columns\":{{\"geometry\":{{\"encoding\":\"WKB\",\"geometry_types\":[{}],\"crs\":null}}}}}}",
            geometry_types_json
        );
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_key_value_metadata(Some(vec![KeyValue {
                key: "geo".to_string(),
                value: Some(geo_metadata),
            }]))
            .build();

        let file = std::fs::File::create(path)
            .with_context(|| format!("failed to create parquet: {}", path.to_string_lossy()))?;
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }

    fn build_object_export_table_from_snapshot(
        snapshot: &ObjectExportSnapshot,
        selected_columns: &HashSet<String>,
        include_geometry: bool,
    ) -> anyhow::Result<ObjectExportTable> {
        let objects = snapshot.objects.as_ref();
        if objects.is_empty() {
            anyhow::bail!("no objects loaded");
        }

        let property_keys = snapshot.property_keys.clone();
        let mut used_names = property_keys.iter().cloned().collect::<HashSet<_>>();

        let mut columns = Vec::new();
        for key in &property_keys {
            if selected_columns.contains(key) {
                columns.push(ExportColumn {
                    name: key.clone(),
                    values: objects
                        .iter()
                        .map(|obj| obj.properties.get(key).map(export_scalar_from_json))
                        .collect(),
                });
            }
        }

        let mut geometry_types_cache: Option<Vec<String>> = None;

        let geometry_type_column_name = unique_export_name("_odon_geometry_type", &mut used_names);
        if selected_columns.contains(&geometry_type_column_name) {
            if geometry_types_cache.is_none() {
                geometry_types_cache = Some(
                    objects
                        .iter()
                        .map(|obj| export_geometry_type_label(obj).to_string())
                        .collect::<Vec<_>>(),
                );
            }
            let geometry_types = geometry_types_cache
                .as_ref()
                .expect("geometry types cached");
            columns.push(ExportColumn {
                name: geometry_type_column_name,
                values: geometry_types
                    .iter()
                    .map(|name| Some(ExportScalar::String(name.clone())))
                    .collect(),
            });
        }

        let centroid_x_name = unique_export_name("_odon_centroid_x", &mut used_names);
        if selected_columns.contains(&centroid_x_name) {
            columns.push(ExportColumn {
                name: centroid_x_name,
                values: objects
                    .iter()
                    .map(|obj| Some(ExportScalar::Float64(obj.centroid_world.x as f64)))
                    .collect(),
            });
        }

        let centroid_y_name = unique_export_name("_odon_centroid_y", &mut used_names);
        if selected_columns.contains(&centroid_y_name) {
            columns.push(ExportColumn {
                name: centroid_y_name,
                values: objects
                    .iter()
                    .map(|obj| Some(ExportScalar::Float64(obj.centroid_world.y as f64)))
                    .collect(),
            });
        }

        if objects.iter().any(|obj| obj.point_position_world.is_some()) {
            let point_x_name = unique_export_name("_odon_point_x", &mut used_names);
            if selected_columns.contains(&point_x_name) {
                columns.push(ExportColumn {
                    name: point_x_name,
                    values: objects
                        .iter()
                        .map(|obj| {
                            obj.point_position_world
                                .map(|pos| ExportScalar::Float64(pos.x as f64))
                        })
                        .collect(),
                });
            }
            let point_y_name = unique_export_name("_odon_point_y", &mut used_names);
            if selected_columns.contains(&point_y_name) {
                columns.push(ExportColumn {
                    name: point_y_name,
                    values: objects
                        .iter()
                        .map(|obj| {
                            obj.point_position_world
                                .map(|pos| ExportScalar::Float64(pos.y as f64))
                        })
                        .collect(),
                });
            }
        }

        let area_name = unique_export_name("_odon_area_px", &mut used_names);
        if selected_columns.contains(&area_name) {
            columns.push(ExportColumn {
                name: area_name,
                values: objects
                    .iter()
                    .map(|obj| Some(ExportScalar::Float64(obj.area_px as f64)))
                    .collect(),
            });
        }

        let perimeter_name = unique_export_name("_odon_perimeter_px", &mut used_names);
        if selected_columns.contains(&perimeter_name) {
            columns.push(ExportColumn {
                name: perimeter_name,
                values: objects
                    .iter()
                    .map(|obj| Some(ExportScalar::Float64(obj.perimeter_px as f64)))
                    .collect(),
            });
        }

        let selected_name = unique_export_name("_odon_selected", &mut used_names);
        if selected_columns.contains(&selected_name) {
            columns.push(ExportColumn {
                name: selected_name,
                values: objects
                    .iter()
                    .enumerate()
                    .map(|(idx, _)| {
                        Some(ExportScalar::Bool(
                            snapshot.selected_object_indices.contains(&idx),
                        ))
                    })
                    .collect(),
            });
        }

        if !snapshot.analysis_property_thresholds.is_empty() {
            let live_name = unique_export_name(
                &live_threshold_call_export_column_name(
                    &snapshot.analysis_property_thresholds,
                    snapshot.analysis_live_threshold_channel_name.as_deref(),
                ),
                &mut used_names,
            );
            if selected_columns.contains(&live_name) {
                columns.push(ExportColumn {
                    name: live_name,
                    values: objects
                        .iter()
                        .map(|obj| {
                            Some(ExportScalar::Bool(object_passes_threshold_rules(
                                obj,
                                &snapshot.analysis_property_thresholds,
                            )))
                        })
                        .collect(),
                });
            }
        }

        for element in &snapshot.analysis_threshold_elements {
            let column_name =
                unique_export_name(&threshold_call_export_column_name(element), &mut used_names);
            if selected_columns.contains(&column_name) {
                columns.push(ExportColumn {
                    name: column_name,
                    values: objects
                        .iter()
                        .map(|obj| {
                            if threshold_call_marks_failed(element) {
                                Some(ExportScalar::String("FAIL".to_string()))
                            } else {
                                Some(ExportScalar::Bool(object_passes_threshold_rules(
                                    obj,
                                    &element.rules,
                                )))
                            }
                        })
                        .collect(),
                });
            }
        }

        for element in &snapshot.selection_elements {
            let column_name = unique_export_name(
                &format!("_odon_selection_{}", sanitize_export_key(&element.name)),
                &mut used_names,
            );
            if selected_columns.contains(&column_name) {
                let selected_ids = element.object_ids.iter().cloned().collect::<HashSet<_>>();
                columns.push(ExportColumn {
                    name: column_name,
                    values: objects
                        .iter()
                        .map(|obj| Some(ExportScalar::Bool(selected_ids.contains(&obj.id))))
                        .collect(),
                });
            }
        }

        let geometry_wkb = if include_geometry {
            objects.iter().map(encode_object_wkb).collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let geometry_types = if include_geometry {
            if geometry_types_cache.is_none() {
                geometry_types_cache = Some(
                    objects
                        .iter()
                        .map(|obj| export_geometry_type_label(obj).to_string())
                        .collect::<Vec<_>>(),
                );
            }
            geometry_types_cache
                .as_ref()
                .expect("geometry types cached")
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect()
        } else {
            Vec::new()
        };

        Ok(ObjectExportTable {
            row_count: objects.len(),
            columns,
            geometry_wkb,
            geometry_types,
        })
    }

    fn build_object_export_column_names(&self) -> anyhow::Result<Vec<String>> {
        let snapshot = self.object_export_snapshot()?;
        let objects = snapshot.objects.as_ref();
        let property_keys = snapshot.property_keys.clone();
        let mut used_names = property_keys.iter().cloned().collect::<HashSet<_>>();
        let mut columns = property_keys.clone();
        let mut push_name = |base_name: &str| {
            let name = unique_export_name(base_name, &mut used_names);
            columns.push(name);
        };

        push_name("_odon_geometry_type");
        push_name("_odon_centroid_x");
        push_name("_odon_centroid_y");
        if objects.iter().any(|obj| obj.point_position_world.is_some()) {
            push_name("_odon_point_x");
            push_name("_odon_point_y");
        }
        push_name("_odon_area_px");
        push_name("_odon_perimeter_px");
        push_name("_odon_selected");

        if !snapshot.analysis_property_thresholds.is_empty() {
            push_name(&live_threshold_call_export_column_name(
                &snapshot.analysis_property_thresholds,
                snapshot.analysis_live_threshold_channel_name.as_deref(),
            ));
        }

        for element in &snapshot.analysis_threshold_elements {
            push_name(&threshold_call_export_column_name(element));
        }
        for element in &snapshot.selection_elements {
            push_name(&format!(
                "_odon_selection_{}",
                sanitize_export_key(&element.name)
            ));
        }

        Ok(columns)
    }

    fn open_object_export_dialog(&mut self, format: ObjectExportFormat) -> anyhow::Result<()> {
        if self.is_exporting() {
            anyhow::bail!("an export is already in progress");
        }
        let columns = self
            .build_object_export_column_names()?
            .into_iter()
            .map(|name| ObjectExportColumnSelection {
                name,
                selected: true,
            })
            .collect();
        self.object_export_dialog = Some(ObjectExportDialog { format, columns });
        Ok(())
    }

    pub fn ui_export_dialog(&mut self, ctx: &egui::Context) {
        let Some(mut dialog) = self.object_export_dialog.clone() else {
            return;
        };

        let mut keep_open = true;
        let mut close_requested = false;
        let mut export_requested = false;
        let title = match dialog.format {
            ObjectExportFormat::GeoParquet => "Export Enriched GeoParquet",
            ObjectExportFormat::Csv => "Export Enriched CSV",
        };

        egui::Window::new(title)
            .collapsible(false)
            .resizable(true)
            .default_width(560.0)
            .default_height(520.0)
            .open(&mut keep_open)
            .show(ctx, |ui| {
                ui.label("Choose which enriched columns to include in the export.");
                ui.small(
                    "GeoParquet always includes geometry. CSV exports only the selected tabular columns.",
                );
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Select all").clicked() {
                        for column in &mut dialog.columns {
                            column.selected = true;
                        }
                    }
                    if ui.button("Select none").clicked() {
                        for column in &mut dialog.columns {
                            column.selected = false;
                        }
                    }
                });
                let selected_count = dialog.columns.iter().filter(|column| column.selected).count();
                ui.small(format!(
                    "{selected_count} of {} columns selected",
                    dialog.columns.len()
                ));
                ui.separator();
                let list_height = ui.available_height().clamp(200.0, 480.0);
                ui.set_min_height(list_height);
                egui::ScrollArea::vertical()
                    .id_salt(("object_export_columns", title))
                    .auto_shrink([false, false])
                    .max_height(list_height)
                    .show(ui, |ui| {
                        for column in &mut dialog.columns {
                            ui.checkbox(&mut column.selected, &column.name);
                        }
                    });
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        close_requested = true;
                    }
                    let export_label = match dialog.format {
                        ObjectExportFormat::GeoParquet => "Choose file and export",
                        ObjectExportFormat::Csv => "Choose file and export",
                    };
                    if ui
                        .add_enabled(selected_count > 0, egui::Button::new(export_label))
                        .clicked()
                    {
                        export_requested = true;
                    }
                });
            });

        if export_requested {
            let selected_columns = dialog
                .columns
                .iter()
                .filter(|column| column.selected)
                .map(|column| column.name.clone())
                .collect::<HashSet<_>>();
            let default_name = match dialog.format {
                ObjectExportFormat::GeoParquet => {
                    format!("{}.enriched.geoparquet", self.default_object_export_stem())
                }
                ObjectExportFormat::Csv => {
                    format!("{}.enriched.csv", self.default_object_export_stem())
                }
            };
            let mut file_dialog = FileDialog::new().set_directory(self.default_object_export_dir());
            file_dialog = match dialog.format {
                ObjectExportFormat::GeoParquet => file_dialog
                    .add_filter("GeoParquet", &["geoparquet", "parquet"])
                    .set_title("Export Objects GeoParquet"),
                ObjectExportFormat::Csv => file_dialog
                    .add_filter("CSV", &["csv"])
                    .set_title("Export Objects CSV"),
            };
            let path = file_dialog.set_file_name(&default_name).save_file();
            if let Some(path) = path {
                match self.start_object_export(dialog.format, path.clone(), selected_columns) {
                    Ok(()) => {
                        close_requested = true;
                    }
                    Err(err) => {
                        self.status = format!("Export failed: {err}");
                    }
                }
            }
        }

        if !keep_open || close_requested {
            self.object_export_dialog = None;
        } else {
            self.object_export_dialog = Some(dialog);
        }
    }

    fn start_object_export(
        &mut self,
        format: ObjectExportFormat,
        path: PathBuf,
        selected_columns: HashSet<String>,
    ) -> anyhow::Result<()> {
        if self.is_exporting() {
            anyhow::bail!("an export is already in progress");
        }
        if selected_columns.is_empty() {
            anyhow::bail!("no export columns selected");
        }
        let snapshot = self.object_export_snapshot()?;
        let object_count = snapshot.objects.len();
        let request_id = self.object_export_request_id.wrapping_add(1).max(1);
        self.object_export_request_id = request_id;
        let (tx, rx) = crossbeam_channel::unbounded();
        self.object_export_rx = Some(rx);
        self.status = format!("Exporting {} object(s)...", object_count);
        std::thread::spawn(move || {
            let error = match format {
                ObjectExportFormat::GeoParquet => {
                    Self::export_objects_geoparquet(&snapshot, path.as_path(), &selected_columns)
                }
                ObjectExportFormat::Csv => {
                    Self::export_objects_csv(&snapshot, path.as_path(), &selected_columns)
                }
            }
            .err()
            .map(|err| err.to_string());
            let _ = tx.send(ObjectExportEvent::Finished {
                request_id,
                path,
                object_count,
                error,
            });
        });
        Ok(())
    }

    fn object_export_snapshot(&self) -> anyhow::Result<ObjectExportSnapshot> {
        let Some(objects) = self.objects.as_ref() else {
            anyhow::bail!("no objects loaded");
        };
        if objects.is_empty() {
            anyhow::bail!("no objects loaded");
        }
        Ok(ObjectExportSnapshot {
            objects: Arc::clone(objects),
            property_keys: self.object_property_keys.clone(),
            selected_object_indices: self.selected_object_indices.clone(),
            analysis_property_thresholds: self.analysis_property_thresholds.clone(),
            analysis_live_threshold_channel_name: self.analysis_live_threshold_channel_name.clone(),
            analysis_threshold_elements: self.analysis_threshold_elements.clone(),
            selection_elements: self.selection_elements.clone(),
        })
    }

    pub fn is_exporting(&self) -> bool {
        self.object_export_rx.is_some()
    }

    fn extend_object_property_keys<'a, I>(&mut self, keys: I)
    where
        I: IntoIterator<Item = &'a str>,
    {
        for key in keys {
            match self
                .object_property_keys
                .binary_search_by(|existing| existing.as_str().cmp(key))
            {
                Ok(_) => {}
                Err(idx) => self.object_property_keys.insert(idx, key.to_string()),
            }
        }
    }

    pub(super) fn invalidate_table_cache(&mut self) {
        self.table_cache_dirty = true;
    }

    pub fn request_zoom_to_object(&mut self, idx: usize) {
        self.analysis_hist_focus_object_index = Some(idx);
        self.pending_zoom_object_index = Some(idx);
    }

    pub fn take_pending_zoom_object_index(&mut self) -> Option<usize> {
        self.pending_zoom_object_index.take()
    }
}

impl ObjectIndexBins {
    fn build(bounds: &[egui::Rect], bin_size: f32) -> Option<Self> {
        let bin_size = bin_size.max(1.0);
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut any = false;

        for rect in bounds {
            if rect.is_positive() {
                any = true;
                min_x = min_x.min(rect.min.x);
                min_y = min_y.min(rect.min.y);
                max_x = max_x.max(rect.max.x);
                max_y = max_y.max(rect.max.y);
            }
        }
        if !any {
            return None;
        }

        let w = (max_x - min_x).max(1.0);
        let h = (max_y - min_y).max(1.0);
        let bins_w = ((w / bin_size).ceil() as usize).max(1);
        let bins_h = ((h / bin_size).ceil() as usize).max(1);
        let origin = egui::pos2(min_x, min_y);
        let bins_len = bins_w.saturating_mul(bins_h);
        let mut counts = vec![0u32; bins_len];

        for rect in bounds {
            let (x0, y0, x1, y1) = rect_bins(*rect, origin, bin_size, bins_w, bins_h);
            for by in y0..=y1 {
                for bx in x0..=x1 {
                    counts[by * bins_w + bx] = counts[by * bins_w + bx].saturating_add(1);
                }
            }
        }

        let mut offsets = vec![0u32; bins_len];
        let mut total = 0u32;
        for (i, c) in counts.iter().copied().enumerate() {
            offsets[i] = total;
            total = total.saturating_add(c);
        }
        let mut indices = vec![0u32; total as usize];
        let mut cursor = offsets.clone();

        for (idx, rect) in bounds.iter().copied().enumerate() {
            let (x0, y0, x1, y1) = rect_bins(rect, origin, bin_size, bins_w, bins_h);
            for by in y0..=y1 {
                for bx in x0..=x1 {
                    let bi = by * bins_w + bx;
                    let write = cursor[bi] as usize;
                    if write < indices.len() {
                        indices[write] = idx as u32;
                    }
                    cursor[bi] = cursor[bi].saturating_add(1);
                }
            }
        }

        Some(Self {
            origin,
            bin_size,
            bins_w,
            bins_h,
            indices,
            offsets,
            counts,
        })
    }

    pub(super) fn bin_range_for_world_rect(
        &self,
        rect: egui::Rect,
    ) -> (usize, usize, usize, usize) {
        rect_bins(rect, self.origin, self.bin_size, self.bins_w, self.bins_h)
    }

    pub(super) fn bin_slice(&self, bin_index: usize) -> &[u32] {
        let start = self.offsets.get(bin_index).copied().unwrap_or(0) as usize;
        let count = self.counts.get(bin_index).copied().unwrap_or(0) as usize;
        let end = start.saturating_add(count).min(self.indices.len());
        &self.indices[start..end]
    }
}

fn check_cancel(cancel: &AtomicBool) -> anyhow::Result<()> {
    if cancel.load(Ordering::Relaxed) {
        anyhow::bail!("object load cancelled");
    }
    Ok(())
}

fn load_in_thread(
    path: PathBuf,
    downsample_factor: f32,
    load_options: Option<ObjectLoadOptions>,
    request_id: u64,
    cancel: &AtomicBool,
) -> anyhow::Result<LoadResult> {
    // Format dispatch is based on the path. Regardless of source, each branch is normalized into
    // `GeoJsonObjectFeature` records plus enough metadata to rebuild display/analysis state.
    check_cancel(cancel)?;
    let (display_mode, objects, lazy_parquet_source) = if is_parquet_objects_path(&path) {
        let parquet_options = match load_options.as_ref() {
            Some(ObjectLoadOptions::Parquet(options)) => Some(options),
            _ => None,
        };
        let display_mode = parquet_options
            .map(|opts| opts.display_mode)
            .unwrap_or(ObjectDisplayMode::Polygons);
        let schema = inspect_shapes_object_schema(&path)?;
        check_cancel(cancel)?;
        let loaded_property_columns = parquet_loaded_property_columns(parquet_options, &schema);
        (
            display_mode,
            parse_geoparquet_objects(&path, parquet_options, cancel)?,
            Some(LazyParquetSource {
                geometry_column: parquet_options
                    .and_then(|opts| match &opts.source {
                        ObjectParquetSource::Geometry(shape_options) => {
                            Some(shape_options.geometry_column.clone())
                        }
                        ObjectParquetSource::XYColumns { .. } => None,
                    })
                    .unwrap_or_else(|| preferred_geometry_column(&schema)),
                available_property_columns: schema.property_columns,
                loaded_property_columns,
            }),
        )
    } else if is_csv_objects_path(&path) {
        let csv_options = match load_options.as_ref() {
            Some(ObjectLoadOptions::Csv(options)) => Some(options),
            _ => None,
        };
        (
            ObjectDisplayMode::Points,
            parse_csv_objects(&path, csv_options, cancel)?,
            None,
        )
    } else {
        check_cancel(cancel)?;
        (
            ObjectDisplayMode::Polygons,
            parse_geojson_objects(&path, downsample_factor)?,
            None,
        )
    };
    load_result_from_objects(
        request_id,
        path,
        downsample_factor,
        SpatialDataTransform2::default(),
        display_mode,
        objects,
        lazy_parquet_source,
        cancel,
    )
}

fn load_spatialdata_in_thread(
    path: PathBuf,
    transform: SpatialDataTransform2,
    request_id: u64,
    cancel: &AtomicBool,
) -> anyhow::Result<LoadResult> {
    let objects = parse_spatialdata_objects(&path, cancel)?;
    load_result_from_objects(
        request_id,
        path,
        1.0,
        transform,
        ObjectDisplayMode::Polygons,
        objects,
        None,
        cancel,
    )
}

fn load_result_from_objects(
    request_id: u64,
    path: PathBuf,
    downsample_factor: f32,
    display_transform: SpatialDataTransform2,
    display_mode: ObjectDisplayMode,
    objects: Vec<GeoJsonObjectFeature>,
    lazy_parquet_source: Option<LazyParquetSource>,
    cancel: &AtomicBool,
) -> anyhow::Result<LoadResult> {
    check_cancel(cancel)?;
    let bounds = objects.iter().map(|o| o.bbox_world).collect::<Vec<_>>();
    let bounds_local =
        union_rects(&bounds).ok_or_else(|| anyhow!("no valid object bounds after parsing"))?;
    check_cancel(cancel)?;
    let bins = ObjectIndexBins::build(&bounds, 512.0)
        .ok_or_else(|| anyhow!("no valid object bounds after parsing"))?;
    check_cancel(cancel)?;
    let render_lods = build_render_lods(&objects)?;
    check_cancel(cancel)?;
    let object_fill_mesh = if display_mode == ObjectDisplayMode::Polygons {
        build_object_fill_mesh(&objects).ok()
    } else {
        None
    };
    check_cancel(cancel)?;
    let object_selection_lods = if display_mode == ObjectDisplayMode::Polygons {
        build_object_selection_render_lods(&objects).ok()
    } else {
        None
    };
    check_cancel(cancel)?;
    let (point_positions_world, point_values, point_lods) =
        build_object_point_payload(&objects, display_transform);
    check_cancel(cancel)?;
    let object_property_keys = discover_property_keys(&objects);
    let scalar_property_keys = discover_scalar_property_keys(&objects);
    let color_property_keys = discover_categorical_color_keys(&objects);
    Ok(LoadResult {
        request_id,
        path,
        downsample_factor,
        display_transform,
        display_mode,
        objects: Arc::new(objects),
        bins: Arc::new(bins),
        render_lods,
        object_fill_mesh,
        object_selection_lods,
        point_positions_world,
        point_values,
        point_lods,
        object_property_keys,
        scalar_property_keys,
        color_property_keys,
        lazy_parquet_source,
        bounds_local,
    })
}

fn preferred_geometry_column(schema: &ShapesObjectSchema) -> String {
    schema
        .geometry_candidates
        .iter()
        .find(|name| name.as_str() == "geometry")
        .cloned()
        .or_else(|| schema.geometry_candidates.first().cloned())
        .unwrap_or_else(|| "geometry".to_string())
}

fn preferred_object_id_property_columns(property_columns: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for key in [
        "id",
        "instance_id",
        "instance_id_polygon",
        "cell_id",
        "label",
        "name",
        "polygon_name",
    ] {
        if property_columns.iter().any(|name| name == key) {
            out.push(key.to_string());
        }
    }
    out
}

fn minimal_parquet_load_options(path: &Path) -> anyhow::Result<ObjectParquetLoadOptions> {
    let schema = inspect_shapes_object_schema(path)?;
    if schema.geometry_candidates.is_empty() {
        anyhow::bail!("No supported binary geometry columns found in GeoParquet.");
    }
    let geometry_column = preferred_geometry_column(&schema);
    let property_columns = preferred_object_id_property_columns(&schema.property_columns);
    Ok(ObjectParquetLoadOptions {
        display_mode: ObjectDisplayMode::Polygons,
        source: ObjectParquetSource::Geometry(ShapesLoadOptions {
            transform: SpatialDataTransform2::default(),
            geometry_column,
            property_columns: Some(property_columns),
        }),
    })
}

fn parquet_loaded_property_columns(
    options: Option<&ObjectParquetLoadOptions>,
    schema: &ShapesObjectSchema,
) -> HashSet<String> {
    match options {
        Some(ObjectParquetLoadOptions {
            source: ObjectParquetSource::Geometry(shape_options),
            ..
        }) => shape_options
            .property_columns
            .clone()
            .unwrap_or_else(|| schema.property_columns.clone())
            .into_iter()
            .collect(),
        Some(ObjectParquetLoadOptions {
            source:
                ObjectParquetSource::XYColumns {
                    property_columns, ..
                },
            ..
        }) => property_columns
            .clone()
            .unwrap_or_else(|| schema.property_columns.clone())
            .into_iter()
            .collect(),
        None => schema.property_columns.iter().cloned().collect(),
    }
}

fn load_parquet_property_values_for_loaded_objects(
    path: &Path,
    geometry_column: &str,
    property_key: &str,
) -> anyhow::Result<HashMap<usize, serde_json::Value>> {
    let cancel = AtomicBool::new(false);
    let loaded = load_shapes_objects(
        path,
        &ShapesLoadOptions {
            transform: SpatialDataTransform2::default(),
            geometry_column: geometry_column.to_string(),
            property_columns: Some(vec![property_key.to_string()]),
        },
        &cancel,
    )?;

    let mut out = HashMap::with_capacity(loaded.len());
    for obj in loaded {
        let Some(row_index) = obj.source_row_index else {
            continue;
        };
        let Some(value) = obj.properties.get(property_key).cloned() else {
            continue;
        };
        out.insert(row_index, value);
    }
    Ok(out)
}

fn parse_geojson_objects(
    path: &Path,
    downsample_factor: f32,
) -> anyhow::Result<Vec<GeoJsonObjectFeature>> {
    if !path.exists() {
        anyhow::bail!("missing GeoJSON file: {}", path.to_string_lossy());
    }
    let text = std::fs::read_to_string(path)
        .map_err(anyhow::Error::from)
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).map_err(anyhow::Error::from))?;
    let feats = text
        .get("features")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let scale = downsample_factor.max(1e-6);

    let mut out = Vec::new();
    for (feature_index, feat) in feats.iter().enumerate() {
        let Some(geom) = feat.get("geometry") else {
            continue;
        };
        let polygons_world = parse_feature_polygons(geom, scale);
        if polygons_world.is_empty() {
            continue;
        }
        let Some((bbox_world, area_px, perimeter_px, centroid_world)) =
            summarize_geometry(&polygons_world)
        else {
            continue;
        };

        let properties = feat
            .get("properties")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();
        let id = feature_id(feat, &properties, feature_index);

        out.push(GeoJsonObjectFeature {
            id,
            polygons_world,
            point_position_world: None,
            bbox_world,
            area_px,
            perimeter_px,
            centroid_world,
            properties,
            source_row_index: None,
        });
    }

    if out.is_empty() {
        anyhow::bail!("no supported polygon objects in GeoJSON");
    }
    Ok(out)
}

fn parse_spatialdata_objects(
    path: &Path,
    cancel: &AtomicBool,
) -> anyhow::Result<Vec<GeoJsonObjectFeature>> {
    if !path.exists() {
        anyhow::bail!(
            "missing SpatialData shapes parquet: {}",
            path.to_string_lossy()
        );
    }
    let loaded = load_shapes_objects(path, &ShapesLoadOptions::default(), cancel)?;
    let mut out = Vec::with_capacity(loaded.len());
    for obj in loaded {
        check_cancel(cancel)?;
        let Some((bbox_world, area_px, perimeter_px, centroid_world)) =
            summarize_geometry(&obj.polygons_world)
        else {
            continue;
        };
        out.push(GeoJsonObjectFeature {
            id: obj.id,
            polygons_world: obj.polygons_world,
            point_position_world: obj.point_position_world,
            bbox_world,
            area_px,
            perimeter_px,
            centroid_world,
            properties: obj.properties,
            source_row_index: None,
        });
    }
    if out.is_empty() {
        anyhow::bail!("no polygon objects found in SpatialData shapes parquet");
    }
    Ok(out)
}

fn parse_geoparquet_objects(
    path: &Path,
    options: Option<&ObjectParquetLoadOptions>,
    cancel: &AtomicBool,
) -> anyhow::Result<Vec<GeoJsonObjectFeature>> {
    // GeoParquet can contribute either polygon geometries or point-like objects built from XY
    // columns. Both paths are normalized into the same object feature struct so the rest of the
    // layer does not care which representation produced them.
    if !path.exists() {
        anyhow::bail!("missing GeoParquet file: {}", path.to_string_lossy());
    }
    let default_options = ObjectParquetLoadOptions {
        display_mode: ObjectDisplayMode::Polygons,
        source: ObjectParquetSource::Geometry(ShapesLoadOptions::default()),
    };
    let loaded = match &options.unwrap_or(&default_options).source {
        ObjectParquetSource::Geometry(shape_options) => {
            load_shapes_objects(path, shape_options, cancel)?
        }
        ObjectParquetSource::XYColumns {
            x_column,
            y_column,
            property_columns,
        } => load_shapes_xy_point_objects(
            path,
            x_column,
            y_column,
            property_columns.as_deref(),
            cancel,
        )?,
    };
    let mut out = Vec::with_capacity(loaded.len());
    for obj in loaded {
        check_cancel(cancel)?;
        let Some((bbox_world, area_px, perimeter_px, centroid_world)) =
            summarize_geometry(&obj.polygons_world)
        else {
            continue;
        };
        out.push(GeoJsonObjectFeature {
            id: obj.id,
            polygons_world: obj.polygons_world,
            point_position_world: obj.point_position_world,
            bbox_world,
            area_px,
            perimeter_px,
            centroid_world,
            properties: obj.properties,
            source_row_index: obj.source_row_index,
        });
    }
    if out.is_empty() {
        anyhow::bail!("no polygon objects found in GeoParquet");
    }
    Ok(out)
}

#[derive(Debug)]
struct CsvObjectSchema {
    property_columns: Vec<String>,
    numeric_columns: Vec<String>,
}

fn inspect_csv_object_schema(path: &Path) -> anyhow::Result<CsvObjectSchema> {
    if !path.exists() {
        anyhow::bail!("missing CSV file: {}", path.to_string_lossy());
    }
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .with_context(|| format!("failed to open CSV: {}", path.to_string_lossy()))?;
    let headers = reader
        .headers()
        .with_context(|| format!("failed to read CSV headers: {}", path.to_string_lossy()))?
        .iter()
        .map(|s| s.trim().to_string())
        .collect::<Vec<_>>();
    let mut numeric_ok = vec![true; headers.len()];
    let mut numeric_seen = vec![false; headers.len()];
    for record in reader.records().take(1024) {
        let record = record.with_context(|| {
            format!(
                "failed to read CSV records while inspecting: {}",
                path.to_string_lossy()
            )
        })?;
        for (idx, value) in record.iter().enumerate() {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                continue;
            }
            if trimmed.parse::<f64>().is_ok() {
                if let Some(seen) = numeric_seen.get_mut(idx) {
                    *seen = true;
                }
            } else if let Some(ok) = numeric_ok.get_mut(idx) {
                *ok = false;
            }
        }
    }
    let numeric_columns = headers
        .iter()
        .enumerate()
        .filter(|(idx, _)| {
            numeric_ok.get(*idx).copied().unwrap_or(false)
                && numeric_seen.get(*idx).copied().unwrap_or(false)
        })
        .map(|(_, name)| name.clone())
        .collect::<Vec<_>>();
    Ok(CsvObjectSchema {
        property_columns: headers,
        numeric_columns,
    })
}

fn parse_csv_objects(
    path: &Path,
    options: Option<&ObjectCsvLoadOptions>,
    cancel: &AtomicBool,
) -> anyhow::Result<Vec<GeoJsonObjectFeature>> {
    // CSV import is point-oriented: infer X/Y columns, then lift each row into an object feature
    // whose geometry is represented primarily by a point position rather than polygon rings.
    if !path.exists() {
        anyhow::bail!("missing CSV file: {}", path.to_string_lossy());
    }
    let schema = inspect_csv_object_schema(path)?;
    let x_column = options
        .map(|opts| opts.x_column.clone())
        .or_else(|| {
            preferred_xy_column(
                &schema.numeric_columns,
                &[
                    "x_centroid",
                    "x",
                    "x_centroid_image",
                    "centroid_x",
                    "xcoord",
                ],
            )
        })
        .ok_or_else(|| anyhow!("CSV is missing a usable X column"))?;
    let y_column = options
        .map(|opts| opts.y_column.clone())
        .or_else(|| {
            preferred_xy_column(
                &schema.numeric_columns,
                &[
                    "y_centroid",
                    "y",
                    "y_centroid_image",
                    "centroid_y",
                    "ycoord",
                ],
            )
        })
        .ok_or_else(|| anyhow!("CSV is missing a usable Y column"))?;
    let selected_property_columns = options
        .and_then(|opts| opts.property_columns.as_ref())
        .cloned()
        .unwrap_or_else(|| schema.property_columns.clone());

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .with_context(|| format!("failed to open CSV: {}", path.to_string_lossy()))?;
    let headers = reader
        .headers()
        .with_context(|| format!("failed to read CSV headers: {}", path.to_string_lossy()))?
        .iter()
        .map(|s| s.trim().to_string())
        .collect::<Vec<_>>();
    let x_idx = headers
        .iter()
        .position(|name| name == &x_column)
        .with_context(|| format!("missing x column '{x_column}'"))?;
    let y_idx = headers
        .iter()
        .position(|name| name == &y_column)
        .with_context(|| format!("missing y column '{y_column}'"))?;

    let selected = selected_property_columns
        .iter()
        .cloned()
        .collect::<HashSet<_>>();
    let mut out = Vec::new();
    for (row_index, record) in reader.records().enumerate() {
        check_cancel(cancel)?;
        let record = record.with_context(|| {
            format!(
                "failed reading CSV row {}: {}",
                row_index + 1,
                path.to_string_lossy()
            )
        })?;
        let Some(x) = record.get(x_idx).and_then(|s| s.trim().parse::<f32>().ok()) else {
            continue;
        };
        let Some(y) = record.get(y_idx).and_then(|s| s.trim().parse::<f32>().ok()) else {
            continue;
        };
        if !x.is_finite() || !y.is_finite() {
            continue;
        }

        let mut properties = serde_json::Map::new();
        properties.insert(x_column.clone(), serde_json::Value::from(x as f64));
        properties.insert(y_column.clone(), serde_json::Value::from(y as f64));
        for (idx, name) in headers.iter().enumerate() {
            if idx == x_idx || idx == y_idx {
                continue;
            }
            if !selected.contains(name) {
                continue;
            }
            let Some(raw) = record.get(idx) else {
                continue;
            };
            if raw.trim().is_empty() {
                continue;
            }
            properties.insert(name.clone(), csv_cell_to_json(raw));
        }

        let id = object_id_from_properties_local(&properties)
            .unwrap_or_else(|| (row_index + 1).to_string());
        properties.insert("id".to_string(), serde_json::Value::String(id.clone()));
        let center = egui::pos2(x, y);
        let polygons_world = vec![circle_polyline_local(center, 4.0, 24)];
        let Some((bbox_world, area_px, perimeter_px, centroid_world)) =
            summarize_geometry(&polygons_world)
        else {
            continue;
        };
        out.push(GeoJsonObjectFeature {
            id,
            polygons_world,
            point_position_world: Some(center),
            bbox_world,
            area_px,
            perimeter_px,
            centroid_world,
            properties,
            source_row_index: Some(row_index),
        });
    }

    if out.is_empty() {
        anyhow::bail!("no point objects found in CSV");
    }
    Ok(out)
}

fn is_parquet_objects_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "parquet" | "geoparquet"))
        .unwrap_or(false)
}

fn is_csv_objects_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("csv"))
        .unwrap_or(false)
}

fn csv_cell_to_json(raw: &str) -> serde_json::Value {
    let trimmed = raw.trim();
    if let Ok(value) = trimmed.parse::<i64>() {
        return serde_json::Value::from(value);
    }
    if let Ok(value) = trimmed.parse::<u64>() {
        return serde_json::Value::from(value);
    }
    if let Ok(value) = trimmed.parse::<f64>() {
        if value.is_finite() {
            return serde_json::Value::from(value);
        }
    }
    if trimmed.eq_ignore_ascii_case("true") {
        return serde_json::Value::Bool(true);
    }
    if trimmed.eq_ignore_ascii_case("false") {
        return serde_json::Value::Bool(false);
    }
    serde_json::Value::String(trimmed.to_string())
}

fn object_id_from_properties_local(
    properties: &serde_json::Map<String, serde_json::Value>,
) -> Option<String> {
    for key in [
        "id",
        "instance_id",
        "instance_id_polygon",
        "cell_id",
        "label",
        "name",
        "polygon_name",
    ] {
        if let Some(value) = properties.get(key) {
            match value {
                serde_json::Value::String(v) => return Some(v.clone()),
                other => return Some(other.to_string()),
            }
        }
    }
    None
}

fn circle_polyline_local(
    center: egui::Pos2,
    radius_world: f32,
    segments: usize,
) -> Vec<egui::Pos2> {
    let n = segments.max(8);
    let mut pts = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let t = (i as f32) * std::f32::consts::TAU / (n as f32);
        pts.push(egui::pos2(
            center.x + radius_world * t.cos(),
            center.y + radius_world * t.sin(),
        ));
    }
    pts
}

fn fuzzy_filter_names(query: &str, names: &[String]) -> Vec<String> {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return names.to_vec();
    }
    let mut ranked = names
        .iter()
        .filter_map(|name| fuzzy_name_score_local(trimmed, name).map(|score| (score, name)))
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));
    ranked.into_iter().map(|(_, name)| name.clone()).collect()
}

fn fuzzy_name_score_local(query: &str, candidate: &str) -> Option<i32> {
    let q = query.trim().to_ascii_lowercase();
    let c = candidate.trim().to_ascii_lowercase();
    if q.is_empty() {
        return Some(0);
    }
    if c == q {
        return Some(0);
    }
    if c.starts_with(&q) {
        return Some(10 + (c.len() as i32 - q.len() as i32).max(0));
    }
    if let Some(idx) = c.find(&q) {
        return Some(100 + idx as i32 + (c.len() as i32 - q.len() as i32).max(0));
    }
    let mut pos = 0usize;
    let mut score = 300i32;
    for ch in q.chars() {
        let rest = &c[pos..];
        let found = rest.find(ch)?;
        score += found as i32;
        pos += found + ch.len_utf8();
    }
    Some(score + (c.len() as i32 - q.len() as i32).max(0))
}

fn preferred_xy_column(columns: &[String], preferred_names: &[&str]) -> Option<String> {
    for preferred in preferred_names {
        if let Some(found) = columns
            .iter()
            .find(|name| name.eq_ignore_ascii_case(preferred))
        {
            return Some(found.clone());
        }
    }
    columns.first().cloned()
}

pub(super) fn build_object_point_payload(
    objects: &[GeoJsonObjectFeature],
    _display_transform: SpatialDataTransform2,
) -> (
    Arc<Vec<egui::Pos2>>,
    Arc<Vec<f32>>,
    Arc<Vec<FeaturePointLod>>,
) {
    let positions = objects
        .iter()
        .map(|obj| obj.point_position_world.unwrap_or(obj.centroid_world))
        .collect::<Vec<_>>();
    let values = vec![1.0f32; positions.len()];
    (Arc::new(positions), Arc::new(values), Arc::new(Vec::new()))
}

fn parse_feature_polygons(geom: &serde_json::Value, scale: f32) -> Vec<Vec<egui::Pos2>> {
    let gtype = geom
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    let coords = geom.get("coordinates");
    let mut out = Vec::new();

    match gtype.as_str() {
        "polygon" => {
            if let Some(rings) = coords.and_then(|v| v.as_array()) {
                if let Some(ring) = rings.first() {
                    if let Some(points) = parse_ring_points(ring, scale) {
                        out.push(points);
                    }
                }
            }
        }
        "multipolygon" => {
            if let Some(polys) = coords.and_then(|v| v.as_array()) {
                for poly in polys {
                    let Some(rings) = poly.as_array() else {
                        continue;
                    };
                    if let Some(ring) = rings.first() {
                        if let Some(points) = parse_ring_points(ring, scale) {
                            out.push(points);
                        }
                    }
                }
            }
        }
        _ => {}
    }

    out
}

fn parse_ring_points(node: &serde_json::Value, scale: f32) -> Option<Vec<egui::Pos2>> {
    let arr = node.as_array()?;
    let mut pts = Vec::with_capacity(arr.len().saturating_add(1));
    for p in arr {
        let Some(xy) = p.as_array() else {
            continue;
        };
        if xy.len() < 2 {
            continue;
        }
        let Some(x0) = xy.first().and_then(|v| v.as_f64()) else {
            continue;
        };
        let Some(y0) = xy.get(1).and_then(|v| v.as_f64()) else {
            continue;
        };
        let x = x0 as f32 * scale;
        let y = y0 as f32 * scale;
        if x.is_finite() && y.is_finite() {
            pts.push(egui::pos2(x, y));
        }
    }
    if pts.len() < 3 {
        return None;
    }
    if pts.first() != pts.last() {
        if let Some(first) = pts.first().copied() {
            pts.push(first);
        }
    }
    Some(pts)
}

fn feature_id(
    feat: &serde_json::Value,
    properties: &serde_json::Map<String, serde_json::Value>,
    feature_index: usize,
) -> String {
    if let Some(v) = feat.get("id").and_then(value_to_short_string) {
        return v;
    }
    for key in ["id", "cell_id", "object_id", "label", "name"] {
        if let Some(v) = properties.get(key).and_then(value_to_short_string) {
            return v;
        }
    }
    format!("feature-{}", feature_index + 1)
}

fn value_to_short_string(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(v) => Some(v.clone()),
        serde_json::Value::Number(v) => Some(v.to_string()),
        serde_json::Value::Bool(v) => Some(v.to_string()),
        _ => None,
    }
}

fn value_to_display_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(v) => v.to_string(),
        serde_json::Value::Number(v) => v.to_string(),
        serde_json::Value::String(v) => v.clone(),
        _ => value.to_string(),
    }
}

fn save_geojson_objects(
    path: &Path,
    objects: &[GeoJsonObjectFeature],
    indices: &[usize],
) -> anyhow::Result<()> {
    let mut features = Vec::with_capacity(indices.len());
    for idx in indices {
        let Some(obj) = objects.get(*idx) else {
            continue;
        };
        features.push(export_object_feature_value(obj));
    }
    let root = serde_json::json!({
        "type": "FeatureCollection",
        "features": features,
    });
    let text = serde_json::to_string_pretty(&root)?;
    std::fs::write(path, text)?;
    Ok(())
}

fn export_object_feature_value(obj: &GeoJsonObjectFeature) -> serde_json::Value {
    let geometry = if obj.polygons_world.len() <= 1 {
        let coords = serde_json::Value::Array(vec![ring_coords_value(
            obj.polygons_world
                .first()
                .map(|p| p.as_slice())
                .unwrap_or(&[]),
        )]);
        serde_json::json!({
            "type": "Polygon",
            "coordinates": coords,
        })
    } else {
        let coords = serde_json::Value::Array(
            obj.polygons_world
                .iter()
                .map(|poly| serde_json::Value::Array(vec![ring_coords_value(poly)]))
                .collect(),
        );
        serde_json::json!({
            "type": "MultiPolygon",
            "coordinates": coords,
        })
    };
    serde_json::json!({
        "type": "Feature",
        "id": obj.id,
        "properties": obj.properties,
        "geometry": geometry,
    })
}

fn ring_coords_value(poly: &[egui::Pos2]) -> serde_json::Value {
    serde_json::Value::Array(
        poly.iter()
            .map(|p| serde_json::Value::Array(vec![serde_json::json!(p.x), serde_json::json!(p.y)]))
            .collect(),
    )
}

#[derive(Debug, Clone)]
struct ObjectExportTable {
    row_count: usize,
    columns: Vec<ExportColumn>,
    geometry_wkb: Vec<Vec<u8>>,
    geometry_types: Vec<String>,
}

#[derive(Debug, Clone)]
struct ExportColumn {
    name: String,
    values: Vec<Option<ExportScalar>>,
}

#[derive(Debug, Clone)]
enum ExportScalar {
    Bool(bool),
    Int64(i64),
    UInt64(u64),
    Float64(f64),
    String(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExportScalarType {
    Bool,
    Int64,
    UInt64,
    Float64,
    String,
}

fn export_scalar_from_json(value: &serde_json::Value) -> ExportScalar {
    match value {
        serde_json::Value::Null => ExportScalar::String(String::new()),
        serde_json::Value::Bool(v) => ExportScalar::Bool(*v),
        serde_json::Value::Number(v) => {
            if let Some(v) = v.as_i64() {
                ExportScalar::Int64(v)
            } else if let Some(v) = v.as_u64() {
                ExportScalar::UInt64(v)
            } else {
                ExportScalar::Float64(v.as_f64().unwrap_or_default())
            }
        }
        serde_json::Value::String(v) => ExportScalar::String(v.clone()),
        _ => ExportScalar::String(value.to_string()),
    }
}

fn export_scalar_to_csv(value: Option<&ExportScalar>) -> String {
    match value {
        None => String::new(),
        Some(ExportScalar::Bool(v)) => v.to_string(),
        Some(ExportScalar::Int64(v)) => v.to_string(),
        Some(ExportScalar::UInt64(v)) => v.to_string(),
        Some(ExportScalar::Float64(v)) => v.to_string(),
        Some(ExportScalar::String(v)) => v.clone(),
    }
}

fn infer_export_scalar_type(values: &[Option<ExportScalar>]) -> ExportScalarType {
    let mut saw_value = false;
    let mut saw_bool = false;
    let mut saw_int = false;
    let mut saw_uint = false;
    let mut saw_float = false;
    let mut saw_string = false;

    for value in values.iter().flatten() {
        saw_value = true;
        match value {
            ExportScalar::Bool(_) => saw_bool = true,
            ExportScalar::Int64(_) => saw_int = true,
            ExportScalar::UInt64(_) => saw_uint = true,
            ExportScalar::Float64(_) => saw_float = true,
            ExportScalar::String(_) => saw_string = true,
        }
    }

    if !saw_value {
        return ExportScalarType::String;
    }
    if saw_string {
        return ExportScalarType::String;
    }
    if saw_bool && !(saw_int || saw_uint || saw_float) {
        return ExportScalarType::Bool;
    }
    if (saw_int || saw_uint || saw_float) && !saw_bool {
        if saw_float || (saw_int && saw_uint) {
            return ExportScalarType::Float64;
        }
        if saw_uint {
            return ExportScalarType::UInt64;
        }
        return ExportScalarType::Int64;
    }
    ExportScalarType::String
}

fn export_column_to_arrow_array(
    column: &ExportColumn,
) -> anyhow::Result<(Field, arrow_array::ArrayRef)> {
    let ty = infer_export_scalar_type(&column.values);
    match ty {
        ExportScalarType::Bool => {
            let mut builder = BooleanBuilder::new();
            for value in &column.values {
                match value {
                    Some(ExportScalar::Bool(v)) => builder.append_value(*v),
                    None => builder.append_null(),
                    Some(other) => builder
                        .append_value(matches!(other, ExportScalar::String(v) if v == "true")),
                }
            }
            Ok((
                Field::new(&column.name, arrow_schema::DataType::Boolean, true),
                Arc::new(builder.finish()) as arrow_array::ArrayRef,
            ))
        }
        ExportScalarType::Int64 => {
            let mut builder = Int64Builder::new();
            for value in &column.values {
                match value {
                    Some(ExportScalar::Int64(v)) => builder.append_value(*v),
                    Some(ExportScalar::UInt64(v)) => builder.append_value(*v as i64),
                    Some(ExportScalar::Float64(v)) => builder.append_value(*v as i64),
                    None => builder.append_null(),
                    Some(ExportScalar::Bool(v)) => builder.append_value(i64::from(*v)),
                    Some(ExportScalar::String(v)) => {
                        builder.append_value(v.parse::<i64>().unwrap_or_default())
                    }
                }
            }
            Ok((
                Field::new(&column.name, arrow_schema::DataType::Int64, true),
                Arc::new(builder.finish()) as arrow_array::ArrayRef,
            ))
        }
        ExportScalarType::UInt64 => {
            let mut builder = UInt64Builder::new();
            for value in &column.values {
                match value {
                    Some(ExportScalar::UInt64(v)) => builder.append_value(*v),
                    Some(ExportScalar::Int64(v)) if *v >= 0 => builder.append_value(*v as u64),
                    Some(ExportScalar::Float64(v)) if *v >= 0.0 => builder.append_value(*v as u64),
                    None => builder.append_null(),
                    Some(ExportScalar::Bool(v)) => builder.append_value(u64::from(*v)),
                    Some(ExportScalar::String(v)) => {
                        builder.append_value(v.parse::<u64>().unwrap_or_default())
                    }
                    _ => builder.append_null(),
                }
            }
            Ok((
                Field::new(&column.name, arrow_schema::DataType::UInt64, true),
                Arc::new(builder.finish()) as arrow_array::ArrayRef,
            ))
        }
        ExportScalarType::Float64 => {
            let mut builder = Float64Builder::new();
            for value in &column.values {
                match value {
                    Some(ExportScalar::Float64(v)) => builder.append_value(*v),
                    Some(ExportScalar::Int64(v)) => builder.append_value(*v as f64),
                    Some(ExportScalar::UInt64(v)) => builder.append_value(*v as f64),
                    None => builder.append_null(),
                    Some(ExportScalar::Bool(v)) => builder.append_value(if *v { 1.0 } else { 0.0 }),
                    Some(ExportScalar::String(v)) => {
                        builder.append_value(v.parse::<f64>().unwrap_or_default())
                    }
                }
            }
            Ok((
                Field::new(&column.name, arrow_schema::DataType::Float64, true),
                Arc::new(builder.finish()) as arrow_array::ArrayRef,
            ))
        }
        ExportScalarType::String => {
            let mut builder = StringBuilder::new();
            for value in &column.values {
                match value {
                    Some(value) => builder.append_value(export_scalar_to_csv(Some(value))),
                    None => builder.append_null(),
                }
            }
            Ok((
                Field::new(&column.name, arrow_schema::DataType::Utf8, true),
                Arc::new(builder.finish()) as arrow_array::ArrayRef,
            ))
        }
    }
}

fn unique_export_name(base: &str, used_names: &mut HashSet<String>) -> String {
    if used_names.insert(base.to_string()) {
        return base.to_string();
    }
    let mut counter = 2usize;
    loop {
        let candidate = format!("{base}_{counter}");
        if used_names.insert(candidate.clone()) {
            return candidate;
        }
        counter += 1;
    }
}

fn object_passes_threshold_rules(
    obj: &ObjectFeature,
    rules: &[ObjectPropertyThresholdRule],
) -> bool {
    !rules.is_empty()
        && rules.iter().all(|rule| {
            let Some(value) = obj
                .properties
                .get(&rule.column_key)
                .and_then(numeric_json_value)
            else {
                return false;
            };
            match rule.op {
                AnalysisThresholdOp::GreaterEqual => value >= rule.value,
                AnalysisThresholdOp::LessEqual => value <= rule.value,
            }
        })
}

fn export_geometry_type_label(obj: &ObjectFeature) -> &'static str {
    if obj.polygons_world.is_empty() {
        "Point"
    } else if obj.polygons_world.len() == 1 {
        "Polygon"
    } else {
        "MultiPolygon"
    }
}

fn encode_object_wkb(obj: &ObjectFeature) -> Vec<u8> {
    if obj.polygons_world.is_empty() {
        return encode_wkb_point(obj.point_position_world.unwrap_or(obj.centroid_world));
    }
    if obj.polygons_world.len() == 1 {
        return encode_wkb_polygon(std::slice::from_ref(&obj.polygons_world[0]));
    }
    encode_wkb_multipolygon(&obj.polygons_world)
}

fn encode_wkb_point(pos: egui::Pos2) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + 4 + 16);
    out.push(1);
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&(pos.x as f64).to_le_bytes());
    out.extend_from_slice(&(pos.y as f64).to_le_bytes());
    out
}

fn encode_wkb_polygon(rings: &[Vec<egui::Pos2>]) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(1);
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(rings.len() as u32).to_le_bytes());
    for ring in rings {
        append_wkb_ring(&mut out, ring);
    }
    out
}

fn encode_wkb_multipolygon(polygons: &[Vec<egui::Pos2>]) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(1);
    out.extend_from_slice(&6u32.to_le_bytes());
    out.extend_from_slice(&(polygons.len() as u32).to_le_bytes());
    for polygon in polygons {
        out.extend_from_slice(&encode_wkb_polygon(std::slice::from_ref(polygon)));
    }
    out
}

fn append_wkb_ring(out: &mut Vec<u8>, ring: &[egui::Pos2]) {
    let mut coords = ring.iter().copied().collect::<Vec<_>>();
    if let (Some(first), Some(last)) = (coords.first().copied(), coords.last().copied())
        && ((first.x - last.x).abs() > f32::EPSILON || (first.y - last.y).abs() > f32::EPSILON)
    {
        coords.push(first);
    }
    out.extend_from_slice(&(coords.len() as u32).to_le_bytes());
    for p in coords {
        out.extend_from_slice(&(p.x as f64).to_le_bytes());
        out.extend_from_slice(&(p.y as f64).to_le_bytes());
    }
}

pub(super) fn union_rects(rects: &[egui::Rect]) -> Option<egui::Rect> {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut any = false;

    for rect in rects {
        if !rect.is_positive() {
            continue;
        }
        any = true;
        min_x = min_x.min(rect.min.x);
        min_y = min_y.min(rect.min.y);
        max_x = max_x.max(rect.max.x);
        max_y = max_y.max(rect.max.y);
    }

    any.then(|| egui::Rect::from_min_max(egui::pos2(min_x, min_y), egui::pos2(max_x, max_y)))
}

pub(super) fn rect_bins(
    rect: egui::Rect,
    origin: egui::Pos2,
    bin_size: f32,
    bins_w: usize,
    bins_h: usize,
) -> (usize, usize, usize, usize) {
    let x0 = ((rect.min.x - origin.x) / bin_size)
        .floor()
        .clamp(0.0, (bins_w.saturating_sub(1)) as f32) as usize;
    let y0 = ((rect.min.y - origin.y) / bin_size)
        .floor()
        .clamp(0.0, (bins_h.saturating_sub(1)) as f32) as usize;
    let x1 = ((rect.max.x - origin.x) / bin_size)
        .floor()
        .clamp(0.0, (bins_w.saturating_sub(1)) as f32) as usize;
    let y1 = ((rect.max.y - origin.y) / bin_size)
        .floor()
        .clamp(0.0, (bins_h.saturating_sub(1)) as f32) as usize;
    (x0, y0, x1, y1)
}
