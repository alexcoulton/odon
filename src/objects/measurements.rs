use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Context;
use eframe::egui;
use zarrs::array::{Array, ArraySubset};
use zarrs::storage::ReadableStorageTraits;

use crate::data::ome::{ChannelInfo, OmeZarrDataset, retrieve_image_subset_u16};

use super::analysis::build_polygon_mask;
use super::*;

pub(super) fn bulk_measurement_progress_status(
    phase: BulkMeasurementPhase,
    completed: usize,
    total: usize,
) -> String {
    match phase {
        BulkMeasurementPhase::RasterizingLabels => {
            format!("Rasterizing polygons into label image... {completed} / {total}")
        }
        BulkMeasurementPhase::MeasuringChannels => {
            format!("Measuring channel intensities... {completed} / {total}")
        }
    }
}

fn bulk_measurement_metric_label(metric: BulkMeasurementMetric) -> &'static str {
    match metric {
        BulkMeasurementMetric::Mean => "mean",
        BulkMeasurementMetric::Median => "median",
    }
}

fn default_bulk_measurement_prefix(metric: BulkMeasurementMetric) -> &'static str {
    match metric {
        BulkMeasurementMetric::Mean => "mean_intensity_",
        BulkMeasurementMetric::Median => "median_intensity_",
    }
}

impl ObjectsLayer {
    pub fn ui_measurements(
        &mut self,
        ui: &mut egui::Ui,
        dataset: &OmeZarrDataset,
        store: Arc<dyn ReadableStorageTraits>,
        channels: &[ChannelInfo],
        local_to_world_offset: egui::Vec2,
    ) {
        ui.heading("Measurements");
        if !self.has_data() {
            ui.label("Measurements are available for loaded polygon object layers.");
            return;
        }
        if self.display_mode != ObjectDisplayMode::Polygons {
            ui.label(
                "This first measurement implementation is available for polygon object layers only.",
            );
            return;
        }

        let Some(level_info) = dataset.levels.get(
            self.bulk_measurement_level
                .min(dataset.levels.len().saturating_sub(1)),
        ) else {
            ui.label("No image levels available.");
            return;
        };
        let level_w = level_info.shape.get(dataset.dims.x).copied().unwrap_or(0);
        let level_h = level_info.shape.get(dataset.dims.y).copied().unwrap_or(0);
        let raster_bytes = level_w.saturating_mul(level_h).saturating_mul(4);

        ui.label(
            "Bulk measurements currently use an internal rasterized label image built from the loaded polygons.",
        );
        ui.horizontal(|ui| {
            ui.label("Metric");
            let before = self.bulk_measurement_metric;
            ui.selectable_value(
                &mut self.bulk_measurement_metric,
                BulkMeasurementMetric::Mean,
                "Mean",
            );
            ui.selectable_value(
                &mut self.bulk_measurement_metric,
                BulkMeasurementMetric::Median,
                "Median (exact)",
            );
            if self.bulk_measurement_metric != before {
                let old_default = default_bulk_measurement_prefix(before);
                if self.bulk_measurement_prefix.is_empty()
                    || self.bulk_measurement_prefix == old_default
                {
                    self.bulk_measurement_prefix =
                        default_bulk_measurement_prefix(self.bulk_measurement_metric).to_string();
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("Level");
            let selected_text = format!("L{} ({:.2}x)", level_info.index, level_info.downsample);
            egui::ComboBox::from_id_salt("seg_objects_bulk_measurement_level")
                .selected_text(selected_text)
                .show_ui(ui, |ui| {
                    for lvl in &dataset.levels {
                        ui.selectable_value(
                            &mut self.bulk_measurement_level,
                            lvl.index,
                            format!("L{} ({:.2}x)", lvl.index, lvl.downsample),
                        );
                    }
                });
        });
        ui.horizontal(|ui| {
            ui.label("Concurrency");
            ui.add(
                egui::DragValue::new(&mut self.bulk_measurement_concurrency)
                    .range(1..=64)
                    .speed(1.0),
            );
            let recommended = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);
            ui.small(format!("Recommended: {recommended}"));
        });
        ui.horizontal(|ui| {
            ui.checkbox(
                &mut self.bulk_measurement_filtered_only,
                "Filtered cells only",
            );
            ui.label(format!(
                "Raster estimate: {} x {} px, {:.1} MB",
                level_w,
                level_h,
                raster_bytes as f64 / (1024.0 * 1024.0)
            ));
        });
        ui.horizontal(|ui| {
            ui.label("Column prefix");
            ui.add(
                egui::TextEdit::singleline(&mut self.bulk_measurement_prefix).desired_width(180.0),
            );
        });

        let target_count = self.bulk_measurement_target_indices().len();
        ui.label(format!(
            "Target cells: {}",
            if self.bulk_measurement_filtered_only {
                format!("{target_count} filtered")
            } else {
                format!("{target_count} total")
            }
        ));

        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    !self.is_bulk_measuring() && !self.is_analyzing() && target_count > 0,
                    egui::Button::new(format!(
                        "Measure {} intensities",
                        bulk_measurement_metric_label(self.bulk_measurement_metric)
                    )),
                )
                .clicked()
            {
                self.request_bulk_measurement(
                    dataset,
                    store.clone(),
                    channels,
                    local_to_world_offset,
                );
            }
            if ui
                .add_enabled(self.is_bulk_measuring(), egui::Button::new("Cancel"))
                .clicked()
            {
                if let Some(cancel) = self.bulk_measurement_cancel.as_ref() {
                    cancel.store(true, Ordering::Relaxed);
                    self.bulk_measurement_status = "Cancelling measurements...".to_string();
                }
            }
        });

        if self.is_bulk_measuring() {
            let denom = self.bulk_measurement_progress_total.max(1) as f32;
            let progress = self.bulk_measurement_progress_completed as f32 / denom;
            ui.add(
                egui::ProgressBar::new(progress)
                    .show_percentage()
                    .animate(true),
            );
        }
        if !self.bulk_measurement_status.is_empty() {
            ui.label(self.bulk_measurement_status.clone());
        }
        ui.separator();
        ui.label("Persist results");
        ui.horizontal(|ui| {
            if ui.button("Export Enriched GeoParquet...").clicked()
                && let Err(err) = self.export_objects_geoparquet_with_dialog()
            {
                self.status = format!("Export GeoParquet failed: {err}");
            }
            if ui.button("Export Enriched CSV...").clicked()
                && let Err(err) = self.export_objects_csv_with_dialog()
            {
                self.status = format!("Export CSV failed: {err}");
            }
        });
        ui.small(
            "Results are attached back onto objects as numeric properties using the chosen prefix, so they become available in Analysis immediately.",
        );
        ui.small(
            "Use export to save those enriched object properties. CSV and GeoParquet exports also include derived call and selection columns.",
        );
        if self.bulk_measurement_metric == BulkMeasurementMetric::Median {
            ui.small(
                "Exact median uses a flat per-channel pixel buffer in RAM, so runtime and memory can increase substantially on large datasets.",
            );
        }
    }

    fn bulk_measurement_target_indices(&self) -> Vec<usize> {
        let Some(objects) = self.objects.as_ref() else {
            return Vec::new();
        };
        if self.bulk_measurement_filtered_only {
            if let Some(filtered) = self.filtered_indices.as_ref() {
                let mut out = filtered.iter().copied().collect::<Vec<_>>();
                out.sort_unstable();
                return out;
            }
        }
        (0..objects.len()).collect()
    }

    fn request_bulk_measurement(
        &mut self,
        dataset: &OmeZarrDataset,
        store: Arc<dyn ReadableStorageTraits>,
        channels: &[ChannelInfo],
        local_to_world_offset: egui::Vec2,
    ) {
        let Some(objects) = self.objects.as_ref().cloned() else {
            self.bulk_measurement_status = "No objects loaded.".to_string();
            return;
        };
        let target_indices = self.bulk_measurement_target_indices();
        if target_indices.is_empty() {
            self.bulk_measurement_status = "No target cells available for measurement.".to_string();
            return;
        }
        if channels.is_empty() || dataset.channels.is_empty() {
            self.bulk_measurement_status = "No image channels available.".to_string();
            return;
        }
        if self.display_mode != ObjectDisplayMode::Polygons {
            self.bulk_measurement_status =
                "Bulk polygon raster measurements are unavailable for point-only object layers."
                    .to_string();
            return;
        }
        if dataset.is_root_label_mask() {
            self.bulk_measurement_status =
                "Image measurements are unavailable for label-mask root datasets.".to_string();
            return;
        }

        self.bulk_measurement_request_id = self.bulk_measurement_request_id.wrapping_add(1).max(1);
        let request_id = self.bulk_measurement_request_id;
        let (tx, rx) = crossbeam_channel::unbounded::<BulkMeasurementEvent>();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_worker = cancel.clone();
        let dataset = dataset.clone();
        let channels = channels.to_vec();
        let level = self
            .bulk_measurement_level
            .min(dataset.levels.len().saturating_sub(1));
        let concurrency = self.bulk_measurement_concurrency.max(1);
        let metric = self.bulk_measurement_metric;
        let prefix = self.bulk_measurement_prefix.clone();
        let filtered_only = self.bulk_measurement_filtered_only;

        self.bulk_measurement_rx = Some(rx);
        self.bulk_measurement_cancel = Some(cancel);
        self.bulk_measurement_progress_completed = 0;
        self.bulk_measurement_progress_total = target_indices.len();
        self.bulk_measurement_status = format!(
            "Preparing {}-intensity measurement for {} {} with concurrency {}...",
            bulk_measurement_metric_label(metric),
            target_indices.len(),
            if filtered_only {
                "filtered cell(s)"
            } else {
                "cell(s)"
            },
            concurrency
        );

        std::thread::Builder::new()
            .name("seg-objects-bulk-measure".to_string())
            .spawn(move || {
                let outcome = measure_objects_rasterized_in_thread(
                    &dataset,
                    store,
                    &channels,
                    objects,
                    &target_indices,
                    local_to_world_offset,
                    level,
                    metric,
                    &prefix,
                    concurrency,
                    request_id,
                    &tx,
                    &cancel_worker,
                    if filtered_only {
                        "filtered cells".to_string()
                    } else {
                        "all cells".to_string()
                    },
                );
                let cancelled = cancel_worker.load(Ordering::Relaxed);
                let msg = match outcome {
                    Ok(result) => BulkMeasurementEvent::Finished {
                        request_id,
                        result: (!cancelled).then_some(result),
                        cancelled,
                        error: None,
                    },
                    Err(err) => BulkMeasurementEvent::Finished {
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

    pub fn is_bulk_measuring(&self) -> bool {
        self.bulk_measurement_rx.is_some()
    }
}

fn measure_objects_rasterized_in_thread(
    dataset: &OmeZarrDataset,
    store: Arc<dyn ReadableStorageTraits>,
    channels: &[ChannelInfo],
    objects: Arc<Vec<GeoJsonObjectFeature>>,
    target_indices: &[usize],
    local_to_world_offset: egui::Vec2,
    level: usize,
    metric: BulkMeasurementMetric,
    prefix: &str,
    concurrency: usize,
    request_id: u64,
    tx: &crossbeam_channel::Sender<BulkMeasurementEvent>,
    cancel: &AtomicBool,
    scope_label: String,
) -> anyhow::Result<BulkMeasurementResult> {
    let level_info = dataset
        .levels
        .get(level)
        .with_context(|| format!("dataset has no measurement level {level}"))?;
    let y_dim = dataset.dims.y;
    let x_dim = dataset.dims.x;
    let level_w = level_info.shape.get(x_dim).copied().unwrap_or(0) as usize;
    let level_h = level_info.shape.get(y_dim).copied().unwrap_or(0) as usize;
    if level_w == 0 || level_h == 0 {
        anyhow::bail!("measurement level has invalid dimensions");
    }

    let mut label_image = vec![0u32; level_w.saturating_mul(level_h)];
    let ds = level_info.downsample.max(1e-6);
    let total_targets = target_indices.len();
    for (i, object_index) in target_indices.iter().copied().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let Some(object) = objects.get(object_index) else {
            continue;
        };
        let polygons_level = object
            .polygons_world
            .iter()
            .map(|poly| {
                poly.iter()
                    .copied()
                    .map(|p| {
                        let world = p + local_to_world_offset;
                        egui::pos2(world.x / ds, world.y / ds)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let bbox_world = object.bbox_world.translate(local_to_world_offset);
        let x0 = (bbox_world.min.x / ds).floor().max(0.0) as u64;
        let y0 = (bbox_world.min.y / ds).floor().max(0.0) as u64;
        let x1 = (bbox_world.max.x / ds).ceil().min(level_w as f32).max(0.0) as u64;
        let y1 = (bbox_world.max.y / ds).ceil().min(level_h as f32).max(0.0) as u64;
        if x1 > x0 && y1 > y0 {
            let width = (x1 - x0) as usize;
            let height = (y1 - y0) as usize;
            let mask = build_polygon_mask(&polygons_level, x0, y0, width, height);
            let label_id = u32::try_from(object_index + 1)
                .context("object count exceeds internal label-mask capacity")?;
            for yy in 0..height {
                let row_off = (y0 as usize + yy).saturating_mul(level_w);
                for xx in 0..width {
                    if mask[yy * width + xx] {
                        label_image[row_off + x0 as usize + xx] = label_id;
                    }
                }
            }
        }
        let completed = i + 1;
        if completed == 1 || completed == total_targets || completed % 128 == 0 {
            let _ = tx.send(BulkMeasurementEvent::Progress {
                request_id,
                phase: BulkMeasurementPhase::RasterizingLabels,
                completed,
                total: total_targets,
            });
        }
    }

    let mut counts = vec![0u32; objects.len()];
    for &label in &label_image {
        if label == 0 {
            continue;
        }
        let idx = label as usize - 1;
        if let Some(count) = counts.get_mut(idx) {
            *count = count.saturating_add(1);
        }
    }

    let mut seen_keys = HashSet::<String>::new();
    let resolved_prefix = if prefix.trim().is_empty() {
        default_bulk_measurement_prefix(metric)
    } else {
        prefix.trim()
    };
    let mut channel_keys = Vec::with_capacity(channels.len());
    for channel in channels {
        channel_keys.push(unique_measurement_key(
            resolved_prefix,
            &channel.name,
            channel.index as u32,
            &mut seen_keys,
        ));
    }
    let mut column_values = vec![(String::new(), Vec::new()); channels.len()];
    let total_channels = channels.len();
    let _ = tx.send(BulkMeasurementEvent::Progress {
        request_id,
        phase: BulkMeasurementPhase::MeasuringChannels,
        completed: 0,
        total: total_channels,
    });

    let dataset_arc = Arc::new(dataset.clone());
    let label_image = Arc::new(label_image);
    let counts = Arc::new(counts);
    let target_indices = Arc::new(target_indices.to_vec());
    let worker_count = concurrency.max(1).min(channels.len().max(1));
    let mut measured_completed = 0usize;

    for batch_start in (0..channels.len()).step_by(worker_count) {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let batch_end = (batch_start + worker_count).min(channels.len());
        let mut batch_results = Vec::with_capacity(batch_end - batch_start);
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(batch_end - batch_start);
            for channel_pos in batch_start..batch_end {
                let dataset = dataset_arc.clone();
                let store = store.clone();
                let label_image = label_image.clone();
                let counts = counts.clone();
                let target_indices = target_indices.clone();
                let channel = channels[channel_pos].clone();
                handles.push(scope.spawn(move || {
                    match metric {
                        BulkMeasurementMetric::Mean => measure_mean_for_channel(
                            &dataset,
                            store,
                            &channel,
                            level,
                            label_image,
                            counts,
                            target_indices,
                            cancel,
                        ),
                        BulkMeasurementMetric::Median => measure_median_for_channel(
                            &dataset,
                            store,
                            &channel,
                            level,
                            label_image,
                            counts,
                            target_indices,
                            cancel,
                        ),
                    }
                    .map(|values| (channel_pos, values))
                }));
            }
            for handle in handles {
                match handle.join() {
                    Ok(result) => batch_results.push(result),
                    Err(_) => batch_results.push(Err(anyhow::anyhow!(
                        "channel worker panicked during bulk measurement"
                    ))),
                }
            }
        });

        for result in batch_results {
            let (channel_pos, values) = result?;
            column_values[channel_pos] = (channel_keys[channel_pos].clone(), values);
            measured_completed += 1;
            let _ = tx.send(BulkMeasurementEvent::Progress {
                request_id,
                phase: BulkMeasurementPhase::MeasuringChannels,
                completed: measured_completed,
                total: total_channels,
            });
        }
    }

    let measured_count = target_indices
        .iter()
        .copied()
        .filter(|idx| counts.get(*idx).copied().unwrap_or(0) > 0)
        .count();
    let failed_count = target_indices.len().saturating_sub(measured_count);

    Ok(BulkMeasurementResult {
        metric,
        scope_label,
        level_index: level_info.index,
        level_downsample: level_info.downsample,
        object_count: target_indices.len(),
        measured_count,
        failed_count,
        column_values,
    })
}

fn measure_mean_for_channel(
    dataset: &OmeZarrDataset,
    store: Arc<dyn ReadableStorageTraits>,
    channel: &ChannelInfo,
    level: usize,
    label_image: Arc<Vec<u32>>,
    counts: Arc<Vec<u32>>,
    target_indices: Arc<Vec<usize>>,
    cancel: &AtomicBool,
) -> anyhow::Result<Vec<Option<f32>>> {
    let level_info = dataset
        .levels
        .get(level)
        .with_context(|| format!("dataset has no measurement level {level}"))?;
    let zarr_path = format!("/{}", level_info.path.trim_start_matches('/'));
    let array: Array<dyn ReadableStorageTraits> = Array::open(store, &zarr_path)
        .with_context(|| format!("open measurement array {zarr_path}"))?;
    let shape = &level_info.shape;
    let y_dim = dataset.dims.y;
    let x_dim = dataset.dims.x;
    let c_dim = dataset.dims.c;
    if y_dim >= shape.len() || x_dim >= shape.len() {
        anyhow::bail!("dataset dimensions do not match measurement level shape");
    }
    let level_w = shape.get(x_dim).copied().unwrap_or(0) as usize;
    let level_h = shape.get(y_dim).copied().unwrap_or(0) as usize;
    if level_w == 0 || level_h == 0 {
        anyhow::bail!("measurement level has invalid dimensions");
    }
    if label_image.len() != level_w.saturating_mul(level_h) {
        anyhow::bail!("internal label image dimensions do not match measurement level");
    }

    let y_chunk = *level_info
        .chunks
        .get(y_dim)
        .context("measurement level is missing Y chunk metadata")?;
    let x_chunk = *level_info
        .chunks
        .get(x_dim)
        .context("measurement level is missing X chunk metadata")?;
    if y_chunk == 0 || x_chunk == 0 {
        anyhow::bail!("measurement level has invalid zero-sized chunks");
    }

    let tiles_y = shape[y_dim].div_ceil(y_chunk);
    let tiles_x = shape[x_dim].div_ceil(x_chunk);
    let mut sums = vec![0u64; counts.len()];

    'tiles: for tile_y in 0..tiles_y {
        for tile_x in 0..tiles_x {
            if cancel.load(Ordering::Relaxed) {
                break 'tiles;
            }

            let y0 = tile_y * y_chunk;
            let x0 = tile_x * x_chunk;
            let y1 = (y0 + y_chunk).min(shape[y_dim]);
            let x1 = (x0 + x_chunk).min(shape[x_dim]);
            let chunk_height = (y1 - y0) as usize;
            let chunk_width = (x1 - x0) as usize;
            if chunk_width == 0 || chunk_height == 0 {
                continue;
            }

            let mut ranges: Vec<std::ops::Range<u64>> = Vec::with_capacity(shape.len());
            for dim in 0..shape.len() {
                if Some(dim) == c_dim {
                    let ch = channel.index as u64;
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
                .with_context(|| format!("read channel {} chunk", channel.index))?;
            let plane = plane_from_channel_data(data, c_dim)?;

            for yy in 0..chunk_height {
                let global_row = (y0 as usize + yy).saturating_mul(level_w);
                for xx in 0..chunk_width {
                    let label_id = label_image[global_row + x0 as usize + xx];
                    if label_id == 0 {
                        continue;
                    }
                    let object_index = label_id as usize - 1;
                    if let Some(sum) = sums.get_mut(object_index) {
                        *sum = sum.saturating_add(plane[(yy, xx)] as u64);
                    }
                }
            }
        }
    }

    let mut values = vec![None; counts.len()];
    for object_index in target_indices.iter().copied() {
        let Some(&count) = counts.get(object_index) else {
            continue;
        };
        if count == 0 {
            continue;
        }
        let mean = sums[object_index] as f32 / count as f32;
        values[object_index] = Some(mean);
    }
    Ok(values)
}

fn measure_median_for_channel(
    dataset: &OmeZarrDataset,
    store: Arc<dyn ReadableStorageTraits>,
    channel: &ChannelInfo,
    level: usize,
    label_image: Arc<Vec<u32>>,
    counts: Arc<Vec<u32>>,
    target_indices: Arc<Vec<usize>>,
    cancel: &AtomicBool,
) -> anyhow::Result<Vec<Option<f32>>> {
    let level_info = dataset
        .levels
        .get(level)
        .with_context(|| format!("dataset has no measurement level {level}"))?;
    let zarr_path = format!("/{}", level_info.path.trim_start_matches('/'));
    let array: Array<dyn ReadableStorageTraits> = Array::open(store, &zarr_path)
        .with_context(|| format!("open measurement array {zarr_path}"))?;
    let shape = &level_info.shape;
    let y_dim = dataset.dims.y;
    let x_dim = dataset.dims.x;
    let c_dim = dataset.dims.c;
    if y_dim >= shape.len() || x_dim >= shape.len() {
        anyhow::bail!("dataset dimensions do not match measurement level shape");
    }
    let level_w = shape.get(x_dim).copied().unwrap_or(0) as usize;
    let level_h = shape.get(y_dim).copied().unwrap_or(0) as usize;
    if level_w == 0 || level_h == 0 {
        anyhow::bail!("measurement level has invalid dimensions");
    }
    if label_image.len() != level_w.saturating_mul(level_h) {
        anyhow::bail!("internal label image dimensions do not match measurement level");
    }

    let y_chunk = *level_info
        .chunks
        .get(y_dim)
        .context("measurement level is missing Y chunk metadata")?;
    let x_chunk = *level_info
        .chunks
        .get(x_dim)
        .context("measurement level is missing X chunk metadata")?;
    if y_chunk == 0 || x_chunk == 0 {
        anyhow::bail!("measurement level has invalid zero-sized chunks");
    }

    let mut offsets = vec![0usize; counts.len() + 1];
    for (i, count) in counts.iter().copied().enumerate() {
        offsets[i + 1] = offsets[i].saturating_add(count as usize);
    }
    let total_pixels = *offsets.last().unwrap_or(&0);
    let mut flat_values = vec![0u16; total_pixels];
    let mut write_positions = offsets[..counts.len()].to_vec();

    let tiles_y = shape[y_dim].div_ceil(y_chunk);
    let tiles_x = shape[x_dim].div_ceil(x_chunk);
    'tiles: for tile_y in 0..tiles_y {
        for tile_x in 0..tiles_x {
            if cancel.load(Ordering::Relaxed) {
                break 'tiles;
            }

            let y0 = tile_y * y_chunk;
            let x0 = tile_x * x_chunk;
            let y1 = (y0 + y_chunk).min(shape[y_dim]);
            let x1 = (x0 + x_chunk).min(shape[x_dim]);
            let chunk_height = (y1 - y0) as usize;
            let chunk_width = (x1 - x0) as usize;
            if chunk_width == 0 || chunk_height == 0 {
                continue;
            }

            let mut ranges: Vec<std::ops::Range<u64>> = Vec::with_capacity(shape.len());
            for dim in 0..shape.len() {
                if Some(dim) == c_dim {
                    let ch = channel.index as u64;
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
                .with_context(|| format!("read channel {} chunk", channel.index))?;
            let plane = plane_from_channel_data(data, c_dim)?;

            for yy in 0..chunk_height {
                let global_row = (y0 as usize + yy).saturating_mul(level_w);
                for xx in 0..chunk_width {
                    let label_id = label_image[global_row + x0 as usize + xx];
                    if label_id == 0 {
                        continue;
                    }
                    let object_index = label_id as usize - 1;
                    if let Some(pos) = write_positions.get_mut(object_index) {
                        if *pos < flat_values.len() {
                            flat_values[*pos] = plane[(yy, xx)];
                            *pos += 1;
                        }
                    }
                }
            }
        }
    }

    let mut values = vec![None; counts.len()];
    for object_index in target_indices.iter().copied() {
        let Some(&count_u32) = counts.get(object_index) else {
            continue;
        };
        let count = count_u32 as usize;
        if count == 0 {
            continue;
        }
        let start = offsets[object_index];
        let end = offsets[object_index + 1];
        let slice = &mut flat_values[start..end];
        let mid = count / 2;
        let upper_value = {
            let (_, upper, _) = slice.select_nth_unstable(mid);
            *upper
        };
        let median = if count % 2 == 1 {
            upper_value as f32
        } else {
            let lower = slice[..mid].iter().copied().max().unwrap_or(upper_value);
            (lower as f32 + upper_value as f32) * 0.5
        };
        values[object_index] = Some(median);
    }
    Ok(values)
}

fn unique_measurement_key(
    prefix: &str,
    channel_name: &str,
    channel_index: u32,
    seen: &mut HashSet<String>,
) -> String {
    let prefix = prefix.trim();
    let token = sanitize_property_token(channel_name);
    let mut base = if token.is_empty() {
        format!("{prefix}ch{channel_index}")
    } else {
        format!("{prefix}{token}")
    };
    if seen.insert(base.clone()) {
        return base;
    }
    base.push_str(&format!("_ch{channel_index}"));
    let mut candidate = base.clone();
    let mut suffix = 2usize;
    while !seen.insert(candidate.clone()) {
        candidate = format!("{base}_{suffix}");
        suffix += 1;
    }
    candidate
}

fn sanitize_property_token(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut prev_underscore = false;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            prev_underscore = false;
        } else if !prev_underscore {
            out.push('_');
            prev_underscore = true;
        }
    }
    out.trim_matches('_').to_string()
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
    .context("unexpected array dimensionality for bulk mean measurements")
}
