//! Analysis computation, quantiles, clustering, and preset I/O.

use super::*;

pub(in crate::control::actor) fn compute_analysis_on_worker(
    spec: &AnalysisResourceSpec,
    kind: AnalysisComputeKind,
    params: &Value,
    request: &OdonControlRequest,
) -> anyhow::Result<Value> {
    anyhow::ensure!(!worker_request_cancelled(request), "analysis was cancelled");
    if matches!(kind, AnalysisComputeKind::Warmup) {
        let mut completed = 0usize;
        for property in spec.resource.property_names.iter() {
            if property == "id" {
                continue;
            }
            if analysis_values(spec, property, "none").next().is_some() {
                completed += 1;
            }
            anyhow::ensure!(!worker_request_cancelled(request), "analysis was cancelled");
        }
        return Ok(json!({"completed":completed}));
    }
    let property = params
        .get("property")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|property| !property.is_empty())
        .ok_or_else(|| anyhow::anyhow!("property is required"))?;
    let transform = params
        .get("transform")
        .and_then(Value::as_str)
        .unwrap_or("none");
    anyhow::ensure!(
        matches!(transform, "none" | "arcsinh"),
        "transform must be 'none' or 'arcsinh'"
    );
    let mut values = analysis_values(spec, property, transform).collect::<Vec<_>>();
    anyhow::ensure!(
        !values.is_empty(),
        "numeric property '{property}' has no finite values in the active object set"
    );
    anyhow::ensure!(!worker_request_cancelled(request), "analysis was cancelled");
    values.sort_by(f32::total_cmp);
    match kind {
        AnalysisComputeKind::Histogram => {
            let bins = params
                .get("bins")
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .unwrap_or(128);
            anyhow::ensure!(
                (8..=4096).contains(&bins),
                "bins must be an integer from 8 to 4096"
            );
            let minimum = values[0];
            let maximum = values[values.len() - 1];
            let median = quantile(&values, 0.5);
            let mut counts = vec![0u64; bins];
            if maximum <= minimum {
                counts[0] = values.len() as u64;
            } else {
                let scale = bins as f32 / (maximum - minimum);
                for value in &values {
                    let index = (((*value - minimum) * scale).floor() as usize).min(bins - 1);
                    counts[index] += 1;
                }
            }
            Ok(json!({
                "property":property,
                "transform":transform,
                "filtered":spec.filtered,
                "count":values.len(),
                "min":minimum,
                "max":maximum,
                "median":median,
                "max_bin_count":counts.iter().copied().max().unwrap_or(0),
                "bins":counts,
            }))
        }
        AnalysisComputeKind::ThresholdSuggestions => {
            let method = params
                .get("method")
                .and_then(Value::as_str)
                .unwrap_or("quantiles");
            anyhow::ensure!(
                matches!(method, "quantiles" | "kmeans"),
                "method must be 'quantiles' or 'kmeans'"
            );
            let count = params
                .get("count")
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .unwrap_or(3);
            anyhow::ensure!(
                (2..=12).contains(&count),
                "count must be an integer from 2 to 12"
            );
            let levels = if method == "quantiles" {
                (1..count)
                    .map(|index| quantile(&values, index as f32 / count as f32))
                    .collect::<Vec<_>>()
            } else {
                kmeans_thresholds(&values, count)
            };
            Ok(json!({
                "property":property,
                "method":method,
                "transform":transform,
                "filtered":spec.filtered,
                "sample_count":values.len(),
                "levels":levels,
            }))
        }
        AnalysisComputeKind::Warmup => unreachable!(),
    }
}

pub(in crate::control::actor) fn analysis_values<'a>(
    spec: &'a AnalysisResourceSpec,
    property: &'a str,
    transform: &'a str,
) -> impl Iterator<Item = f32> + 'a {
    let indices: Box<dyn Iterator<Item = usize> + 'a> = match spec.indices.as_ref() {
        Some(indices) => Box::new(indices.iter().copied()),
        None => Box::new(0..spec.resource.features.len()),
    };
    indices.filter_map(move |index| {
        let value = spec.resource.property_f64(index, property)? as f32;
        let value = if transform == "arcsinh" {
            value.asinh()
        } else {
            value
        };
        value.is_finite().then_some(value)
    })
}

pub(in crate::control::actor) fn quantile(values: &[f32], fraction: f32) -> f32 {
    if values.len() == 1 {
        return values[0];
    }
    let position = fraction.clamp(0.0, 1.0) * (values.len() - 1) as f32;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    values[lower] + (values[upper] - values[lower]) * (position - lower as f32)
}

pub(in crate::control::actor) fn kmeans_thresholds(
    values: &[f32],
    cluster_count: usize,
) -> Vec<f32> {
    let clusters = cluster_count.min(values.len()).max(1);
    let mut centers = (0..clusters)
        .map(|index| quantile(values, (index as f32 + 0.5) / clusters as f32))
        .collect::<Vec<_>>();
    for _ in 0..24 {
        let mut sums = vec![0.0f64; clusters];
        let mut counts = vec![0usize; clusters];
        for value in values {
            let closest = centers
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| {
                    (*value - **left).abs().total_cmp(&(*value - **right).abs())
                })
                .map(|(index, _)| index)
                .unwrap_or(0);
            sums[closest] += *value as f64;
            counts[closest] += 1;
        }
        for index in 0..clusters {
            if counts[index] > 0 {
                centers[index] = (sums[index] / counts[index] as f64) as f32;
            }
        }
        centers.sort_by(f32::total_cmp);
    }
    centers
        .windows(2)
        .map(|pair| (pair[0] + pair[1]) * 0.5)
        .collect()
}

pub(in crate::control::actor) fn read_analysis_preset_on_worker(
    path: &Path,
) -> anyhow::Result<Value> {
    let payload: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    let name = payload
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let elements = payload
        .get("elements")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("invalid call preset: elements must be an array"))?;
    Ok(json!({
        "threshold_set_name":name,
        "threshold_elements":elements,
        "threshold_selected_element":if elements.is_empty() { Value::Null } else { json!(0) },
    }))
}

pub(in crate::control::actor) fn write_analysis_preset_on_worker(
    path: &Path,
    overwrite: bool,
    state: &Value,
) -> anyhow::Result<usize> {
    use std::io::Write;
    let elements = state
        .get("threshold_elements")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let payload = json!({
        "name":state.get("threshold_set_name").and_then(Value::as_str).unwrap_or_default(),
        "elements":elements,
    });
    let mut options = fs::OpenOptions::new();
    options.write(true);
    if overwrite {
        options.create(true).truncate(true);
    } else {
        options.create_new(true);
    }
    let mut file = options.open(path)?;
    file.write_all(serde_json::to_string_pretty(&payload)?.as_bytes())?;
    file.sync_all()?;
    Ok(elements.len())
}
