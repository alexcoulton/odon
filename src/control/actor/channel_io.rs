use super::*;

pub fn read_channel_intensity_stats(
    document: &RenderDocument,
    spec: &ChannelIntensitySpec,
) -> anyhow::Result<Value> {
    let (mut values, shape) = read_channel_values(document, spec)?;
    if values.is_empty() {
        let mut result = json!({
            "index": spec.channel_index,
            "name": spec.channel_name,
            "level": spec.level_number,
            "downsample": spec.downsample,
            "shape": shape,
            "n": 0,
            "error": "empty image subset",
        });
        if let Some(request_id) = spec.client_request_id {
            result["request_id"] = json!(request_id);
        }
        if let Some(bins) = spec.bins {
            result["histogram"] = json!({
                "bins":vec![0_u32; bins.clamp(8, 4096)],
                "abs_max":spec.abs_max,
                "stats":{"n":0},
            });
        }
        return Ok(result);
    }
    values.sort_unstable();
    let n = values.len();
    let sum = values
        .iter()
        .fold(0_u64, |sum, value| sum.saturating_add(u64::from(*value)));
    let nonzero = values.iter().filter(|value| **value != 0).count();
    let percentile = |quantile: f64| {
        let index = ((n.saturating_sub(1)) as f64 * quantile.clamp(0.0, 1.0)).round() as usize;
        values[index.min(n - 1)]
    };
    let histogram = spec.bins.map(|bins| {
        let bins = bins.clamp(8, 4096);
        let mut counts = vec![0_u32; bins];
        let scale = (bins as f32 - 1.0) / spec.abs_max.max(1.0);
        for value in &values {
            let index = ((*value as f32).clamp(0.0, spec.abs_max) * scale).floor() as usize;
            counts[index.min(bins - 1)] = counts[index.min(bins - 1)].saturating_add(1);
        }
        json!({
            "bins":counts,
            "abs_max":spec.abs_max,
            "stats":{
                "min":values[0] as f32,
                "q1":percentile(0.25) as f32,
                "median":percentile(0.50) as f32,
                "q3":percentile(0.75) as f32,
                "max":values[n - 1] as f32,
                "n":n,
            },
        })
    });
    let mut result = json!({
        "index": spec.channel_index,
        "name": spec.channel_name,
        "level": spec.level_number,
        "downsample": spec.downsample,
        "shape": shape,
        "n": n,
        "nonzero": nonzero,
        "nonzero_fraction": nonzero as f64 / n as f64,
        "min": values[0],
        "q1": percentile(0.25),
        "median": percentile(0.50),
        "q3": percentile(0.75),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": values[n - 1],
        "mean": sum as f64 / n as f64,
    });
    if let Some(request_id) = spec.client_request_id {
        result["request_id"] = json!(request_id);
    }
    if let Some(histogram) = histogram {
        result["histogram"] = histogram;
    }
    Ok(result)
}

pub(crate) fn read_auto_contrast(
    document: &RenderDocument,
    spec: &AutoContrastSpec,
) -> anyhow::Result<Vec<AutoContrastChannelResult>> {
    spec.channels
        .iter()
        .map(|channel| {
            let intensity = &channel.intensity;
            let (values, _) = read_channel_values(document, intensity)?;
            let mut histogram = vec![0_u64; 65_536];
            let mut sample_count = 0_u64;
            let mut observed_max = 0_u16;
            for value in values {
                histogram[value as usize] = histogram[value as usize].saturating_add(1);
                sample_count = sample_count.saturating_add(1);
                observed_max = observed_max.max(value);
            }
            let (min, max) = crate::settings::auto_contrast_window_from_histogram(
                spec.settings,
                &histogram,
                sample_count,
                observed_max,
            );
            Ok(AutoContrastChannelResult {
                channel_index: intensity.channel_index,
                channel_name: intensity.channel_name.clone(),
                min,
                max,
                sample_count,
            })
        })
        .collect()
}

/// One histogram per channel across every ROI, because a mosaic has one contrast per channel. An
/// unreadable ROI is reported rather than failing the others; only a total failure is an error.
pub(crate) fn read_mosaic_auto_contrast(
    spec: &MosaicAutoContrastSpec,
    cancelled: &impl Fn() -> bool,
) -> anyhow::Result<MosaicAutoContrastResult> {
    let mut histograms = vec![vec![0_u64; 65_536]; spec.channels.len()];
    let mut totals = vec![(0_u64, 0_u16); spec.channels.len()];
    let mut failed_items = Vec::new();
    for item in &spec.items {
        let outcome = spec.channels.iter().enumerate().try_for_each(
            |(position, channel)| -> anyhow::Result<()> {
                anyhow::ensure!(!cancelled(), "automatic contrast was cancelled");
                let Some(local_channel) = item.channel_map.get(channel.index).copied().flatten()
                else {
                    return Ok(());
                };
                let read = mosaic_channel_read(item, channel, local_channel)?;
                let (values, _) = read_opened_channel_values(&item.document, &read)?;
                let (samples, observed_max) = &mut totals[position];
                add_to_histogram(&mut histograms[position], samples, observed_max, values);
                Ok(())
            },
        );
        if let Err(error) = outcome {
            anyhow::ensure!(!cancelled(), "automatic contrast was cancelled");
            failed_items.push((item.item_id, format!("{error:#}")));
        }
    }
    anyhow::ensure!(
        failed_items.len() < spec.items.len(),
        "no mosaic image could be read: {}",
        failed_items
            .first()
            .map_or("no images", |(_, error)| error.as_str())
    );
    let channels = spec
        .channels
        .iter()
        .zip(histograms.iter().zip(&totals))
        .filter(|(_, (_, (samples, _)))| *samples > 0)
        .map(|(channel, (histogram, (samples, observed_max)))| {
            let (min, max) = crate::settings::auto_contrast_window_from_histogram(
                spec.settings,
                histogram,
                *samples,
                *observed_max,
            );
            MosaicAutoContrastChannelResult {
                index: channel.index,
                min,
                max,
                sample_count: *samples,
            }
        })
        .collect();
    Ok(MosaicAutoContrastResult {
        channels,
        failed_items,
    })
}

fn add_to_histogram(
    histogram: &mut [u64],
    samples: &mut u64,
    observed_max: &mut u16,
    values: impl IntoIterator<Item = u16>,
) {
    for value in values {
        histogram[value as usize] = histogram[value as usize].saturating_add(1);
        *samples = samples.saturating_add(1);
        *observed_max = (*observed_max).max(value);
    }
}

fn mosaic_channel_read(
    item: &MosaicAutoContrastItemSpec,
    channel: &MosaicAutoContrastChannelSpec,
    local_channel: u64,
) -> anyhow::Result<ChannelIntensitySpec> {
    let descriptor = &item.document.descriptor;
    let level = descriptor
        .levels
        .get(item.level)
        .ok_or_else(|| anyhow::anyhow!("missing level {}", item.level))?;
    let (y, x) = (descriptor.dims.y, descriptor.dims.x);
    let height = level.shape.get(y).copied().unwrap_or(0);
    let width = level.shape.get(x).copied().unwrap_or(0);
    let ranges = (0..level.shape.len())
        .map(|dimension| {
            if Some(dimension) == descriptor.dims.c {
                local_channel..local_channel.saturating_add(1)
            } else if dimension == y || dimension == x {
                0..level.shape[dimension]
            } else {
                0..1
            }
        })
        .collect();
    Ok(ChannelIntensitySpec {
        channel_index: channel.index,
        channel_name: channel.name.clone(),
        level_number: level.index,
        downsample: level.downsample,
        zarr_path: format!("/{}", level.path.trim_start_matches('/')),
        dtype: level.dtype.clone(),
        ranges,
        bins: None,
        abs_max: descriptor.abs_max.max(1.0),
        client_request_id: None,
        source_channel: local_channel as usize,
        region: [0, height, 0, width],
    })
}

fn read_channel_values(
    document: &RenderDocument,
    spec: &ChannelIntensitySpec,
) -> anyhow::Result<(Vec<u16>, Vec<usize>)> {
    read_opened_channel_values(&document.opened, spec)
}

fn read_opened_channel_values(
    document: &crate::data::document::ControlOpenedDocument,
    spec: &ChannelIntensitySpec,
) -> anyhow::Result<(Vec<u16>, Vec<usize>)> {
    if let crate::data::document::DocumentResource::Alternate(resource) = &document.resource
        && let Some(reader) = resource.intensity_reader()
    {
        let [y0, y1, x0, x1] = spec.region;
        let data =
            reader.read_channel_region(&crate::data::document::AlternateIntensityRequest {
                level: spec.level_number,
                channel: spec.source_channel,
                y0,
                y1,
                x0,
                x1,
            })?;
        return Ok((data.values, data.shape));
    }
    let array = Array::open(Arc::clone(document.resource.store()), &spec.zarr_path)?;
    let subset = ArraySubset::new_with_ranges(&spec.ranges);
    let data = retrieve_image_subset_u16(&array, &subset, &spec.dtype)?;
    let shape = data.shape().to_vec();
    Ok((data.iter().copied().collect(), shape))
}
