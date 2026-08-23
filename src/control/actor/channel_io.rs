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

fn read_channel_values(
    document: &RenderDocument,
    spec: &ChannelIntensitySpec,
) -> anyhow::Result<(Vec<u16>, Vec<usize>)> {
    if let crate::data::document::DocumentResource::Alternate(resource) = &document.opened.resource
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
    let array = Array::open(Arc::clone(document.store()), &spec.zarr_path)?;
    let subset = ArraySubset::new_with_ranges(&spec.ranges);
    let data = retrieve_image_subset_u16(&array, &subset, &spec.dtype)?;
    let shape = data.shape().to_vec();
    Ok((data.iter().copied().collect(), shape))
}
