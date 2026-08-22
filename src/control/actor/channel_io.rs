use super::*;

pub fn read_channel_intensity_stats(
    document: &RenderDocument,
    spec: &ChannelIntensitySpec,
) -> anyhow::Result<Value> {
    let array = Array::open(Arc::clone(document.store()), &spec.zarr_path)?;
    let subset = ArraySubset::new_with_ranges(&spec.ranges);
    let data = retrieve_image_subset_u16(&array, &subset, &spec.dtype)?;
    let mut values = data.iter().copied().collect::<Vec<_>>();
    if values.is_empty() {
        return Ok(json!({
            "index": spec.channel_index,
            "name": spec.channel_name,
            "level": spec.level_number,
            "downsample": spec.downsample,
            "n": 0,
            "error": "empty image subset",
        }));
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
    Ok(json!({
        "index": spec.channel_index,
        "name": spec.channel_name,
        "level": spec.level_number,
        "downsample": spec.downsample,
        "shape": data.shape(),
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
    }))
}
