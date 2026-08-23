//! Threshold preview worker computations and coordinate transforms.

use super::*;

pub(in crate::control::actor) fn load_threshold_preview_on_worker(
    document: &RenderDocument,
    spec: &ThresholdPreviewLoadSpec,
    request: &OdonControlRequest,
) -> anyhow::Result<ControlThresholdPreviewResource> {
    anyhow::ensure!(
        !worker_request_cancelled(request),
        "threshold preview was cancelled"
    );
    let array: Array<dyn ReadableStorageTraits> =
        Array::open(Arc::clone(document.store()), &spec.zarr_path)?;
    let subset = ArraySubset::new_with_ranges(&spec.ranges);
    let data = retrieve_image_subset_u16(&array, &subset, &spec.dtype)?;
    let dataset = document.dataset();
    let plane = squeeze_pinned_plane(data, dataset.dims.y, dataset.dims.x)
        .ok_or_else(|| anyhow::anyhow!("unexpected threshold preview dimensionality"))?;
    anyhow::ensure!(
        plane.dim() == (spec.height, spec.width),
        "threshold preview dimensions changed during loading"
    );
    anyhow::ensure!(
        !worker_request_cancelled(request),
        "threshold preview was cancelled"
    );
    let mask = extract_threshold_mask(&plane, spec.threshold, spec.min_component_pixels);
    let values = Arc::new(plane.iter().copied().collect());
    Ok(ControlThresholdPreviewResource {
        generation: spec.operation_generation,
        channel_index: spec.channel_index,
        channel_name: spec.channel_name.clone(),
        scope: spec.scope,
        level: spec.level,
        downsample: spec.downsample,
        x0: spec.x0,
        y0: spec.y0,
        width: spec.width,
        height: spec.height,
        values,
        included: Arc::new(mask.included),
        threshold: spec.threshold,
        min_component_pixels: spec.min_component_pixels,
    })
}

pub(in crate::control::actor) fn recompute_threshold_preview_on_worker(
    spec: &ThresholdPreviewRecomputeSpec,
    request: &OdonControlRequest,
) -> anyhow::Result<ControlThresholdPreviewResource> {
    anyhow::ensure!(
        !worker_request_cancelled(request),
        "threshold preview was cancelled"
    );
    let mut preview = (*spec.preview).clone();
    let plane = ndarray::Array2::from_shape_vec(
        (preview.height, preview.width),
        preview.values.as_ref().clone(),
    )?;
    let mask = extract_threshold_mask(&plane, preview.threshold, preview.min_component_pixels);
    anyhow::ensure!(
        !worker_request_cancelled(request),
        "threshold preview was cancelled"
    );
    preview.included = Arc::new(mask.included);
    Ok(preview)
}

pub(in crate::control::actor) fn apply_threshold_preview_on_worker(
    spec: &ThresholdPreviewApplySpec,
    request: &OdonControlRequest,
) -> anyhow::Result<Vec<Vec<[f32; 2]>>> {
    anyhow::ensure!(
        !worker_request_cancelled(request),
        "threshold apply was cancelled"
    );
    let mask = ThresholdMask {
        width: spec.preview.width,
        height: spec.preview.height,
        included: spec.preview.included.as_ref().clone(),
    };
    let polygons = threshold_mask_polygons(&mask);
    anyhow::ensure!(
        !polygons.is_empty(),
        "no visible regions found above the current threshold"
    );
    let transformed = polygons
        .into_iter()
        .map(|polygon| {
            polygon
                .into_iter()
                .map(|point| {
                    let local = [
                        (spec.preview.x0 as f32 + point[0]) * spec.preview.downsample,
                        (spec.preview.y0 as f32 + point[1]) * spec.preview.downsample,
                    ];
                    threshold_local_to_world(
                        local,
                        spec.pivot,
                        spec.offset,
                        spec.scale,
                        spec.rotation_rad,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    anyhow::ensure!(
        !worker_request_cancelled(request),
        "threshold apply was cancelled"
    );
    Ok(transformed)
}

pub(in crate::control::actor) fn worker_request_cancelled(request: &OdonControlRequest) -> bool {
    request
        .task_id
        .as_deref()
        .and_then(|task_id| request.task_registry.get(task_id).ok())
        .is_some_and(|task| task.state == TaskState::Cancelled)
}

pub(in crate::control::actor) fn threshold_local_to_world(
    local: [f32; 2],
    pivot: [f32; 2],
    offset: [f32; 2],
    scale: [f32; 2],
    rotation_rad: f32,
) -> [f32; 2] {
    let scaled = [
        (local[0] - pivot[0]) * scale[0],
        (local[1] - pivot[1]) * scale[1],
    ];
    let (sin, cos) = rotation_rad.sin_cos();
    [
        pivot[0] + scaled[0] * cos - scaled[1] * sin + offset[0],
        pivot[1] + scaled[0] * sin + scaled[1] * cos + offset[1],
    ]
}
