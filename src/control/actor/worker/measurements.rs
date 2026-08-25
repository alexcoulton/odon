//! Object measurement computation and geometry helpers.

use super::*;

pub(in crate::control::actor) fn measure_objects_on_worker(
    document: &RenderDocument,
    spec: &MeasurementSpec,
    request: &OdonControlRequest,
) -> anyhow::Result<(ControlObjectResource, usize)> {
    anyhow::ensure!(
        !worker_request_cancelled(request),
        "measurement was cancelled"
    );
    let dataset = document.dataset();
    let level = dataset
        .levels
        .get(spec.level)
        .ok_or_else(|| anyhow::anyhow!("measurement level is out of range"))?;
    let width = level.shape[dataset.dims.x] as usize;
    let height = level.shape[dataset.dims.y] as usize;
    anyhow::ensure!(
        width > 0 && height > 0,
        "measurement level has invalid dimensions"
    );
    let mut features = spec.resource.features.as_ref().clone();
    let mut property_names = spec.resource.property_names.as_ref().clone();
    let mut measured_objects = std::collections::HashSet::new();
    for channel in &dataset.channels {
        anyhow::ensure!(
            !worker_request_cancelled(request),
            "measurement was cancelled"
        );
        let mut ranges = Vec::with_capacity(level.shape.len());
        for dimension in 0..level.shape.len() {
            if Some(dimension) == dataset.dims.c {
                let selected = (channel.index as u64).min(level.shape[dimension].saturating_sub(1));
                ranges.push(selected..selected.saturating_add(1));
            } else if dimension == dataset.dims.y || dimension == dataset.dims.x {
                ranges.push(0..level.shape[dimension]);
            } else {
                ranges.push(0..level.shape[dimension].min(1));
            }
        }
        let array: Array<dyn ReadableStorageTraits> = Array::open(
            Arc::clone(document.store()),
            &format!("/{}", level.path.trim_start_matches('/')),
        )?;
        let plane = squeeze_pinned_plane(
            retrieve_image_subset_u16(
                &array,
                &ArraySubset::new_with_ranges(&ranges),
                &level.dtype,
            )?,
            dataset.dims.y,
            dataset.dims.x,
        )
        .ok_or_else(|| anyhow::anyhow!("unexpected measurement dimensionality"))?;
        let key =
            measurement_property_key(&spec.prefix, &channel.name, channel.index, &property_names);
        property_names.push(key.clone());
        for &index in spec.target_indices.iter() {
            anyhow::ensure!(
                !worker_request_cancelled(request),
                "measurement was cancelled"
            );
            let Some(feature) = features.get_mut(index) else {
                continue;
            };
            let mut values = Vec::new();
            let downsample = level.downsample.max(1e-6);
            let x0 = (feature.bbox_world[0] / downsample).floor().max(0.0) as usize;
            let y0 = (feature.bbox_world[1] / downsample).floor().max(0.0) as usize;
            let x1 = (feature.bbox_world[2] / downsample).ceil().max(0.0) as usize;
            let y1 = (feature.bbox_world[3] / downsample).ceil().max(0.0) as usize;
            for y in y0.min(height)..y1.min(height) {
                for x in x0.min(width)..x1.min(width) {
                    let world = [(x as f32 + 0.5) * downsample, (y as f32 + 0.5) * downsample];
                    if feature
                        .polygons_world
                        .iter()
                        .any(|polygon| point_in_polygon(world, polygon))
                    {
                        values.push(plane[(y, x)] as f32);
                    }
                }
            }
            if !values.is_empty() {
                let value = match spec.metric {
                    MeasurementMetric::Mean => {
                        values.iter().map(|value| *value as f64).sum::<f64>() / values.len() as f64
                    }
                    MeasurementMetric::Median => {
                        values.sort_by(f32::total_cmp);
                        quantile(&values, 0.5) as f64
                    }
                };
                feature.properties.insert(key.clone(), json!(value));
                measured_objects.insert(index);
            }
        }
    }
    property_names.sort();
    property_names.dedup();
    let numeric_summaries =
        ControlObjectResource::build_numeric_summaries(&features, &property_names);
    Ok((
        ControlObjectResource {
            source: spec.resource.source.clone(),
            downsample_factor: spec.resource.downsample_factor,
            features: Arc::new(features),
            property_names: Arc::new(property_names),
            numeric_summaries,
            renderer_payload: None,
        },
        measured_objects.len(),
    ))
}

pub(in crate::control::actor) fn measurement_property_key(
    prefix: &str,
    name: &str,
    index: usize,
    existing: &[String],
) -> String {
    let token = name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_string();
    let base = format!("{}{token}", prefix.trim());
    if !existing.contains(&base) {
        base
    } else {
        format!("{base}_{index}")
    }
}

pub(in crate::control::actor) fn point_in_polygon(point: [f32; 2], polygon: &[[f32; 2]]) -> bool {
    if polygon.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut previous = polygon.len() - 1;
    for current in 0..polygon.len() {
        let a = polygon[current];
        let b = polygon[previous];
        if ((a[1] > point[1]) != (b[1] > point[1]))
            && point[0] < (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[0]
        {
            inside = !inside;
        }
        previous = current;
    }
    inside
}
