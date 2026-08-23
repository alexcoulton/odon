//! Pinned-memory and mosaic-memory worker computations.

use super::*;

pub(in crate::control::actor) fn load_pinned_memory_on_worker(
    document: &RenderDocument,
    spec: &MemoryPinSpec,
    request: &OdonControlRequest,
) -> anyhow::Result<MemoryPinWorkerResult> {
    let cancelled = || {
        request
            .task_id
            .as_deref()
            .and_then(|task_id| request.task_registry.get(task_id).ok())
            .is_some_and(|task| task.state == TaskState::Cancelled)
    };
    anyhow::ensure!(!cancelled(), "memory pinning was cancelled");
    let system = {
        use sysinfo::System;
        let mut system = System::new();
        system.refresh_memory();
        (system.total_memory() > 0).then_some(SystemMemorySnapshot {
            total_bytes: system.total_memory(),
            available_bytes: system.available_memory(),
        })
    };
    let projected_bytes = spec.pinned_bytes.saturating_add(spec.estimated_bytes);
    let risk = system.and_then(|system| {
        if projected_bytes > system.available_bytes {
            Some("danger")
        } else if projected_bytes.saturating_mul(100)
            >= system.available_bytes.max(1).saturating_mul(75)
        {
            Some("warning")
        } else {
            None
        }
    });
    if let (false, Some(risk), Some(system)) = (spec.force, risk, system) {
        return Ok(MemoryPinWorkerResult {
            system: Some(system),
            outcome: MemoryPinWorkerOutcome::Confirmation {
                risk,
                projected_bytes,
                available_bytes: system.available_bytes,
            },
        });
    }

    let dataset = document.dataset();
    let info = dataset
        .levels
        .get(spec.level)
        .ok_or_else(|| anyhow::anyhow!("missing level {}", spec.level))?;
    let zarr_path = format!("/{}", info.path.trim_start_matches('/'));
    let array: Array<dyn ReadableStorageTraits> =
        Array::open(Arc::clone(document.store()), &zarr_path)?;
    let height = *info.shape.get(dataset.dims.y).unwrap_or(&0) as usize;
    let width = *info.shape.get(dataset.dims.x).unwrap_or(&0) as usize;
    let plane_len = height.saturating_mul(width);
    let mut raw = Vec::new();
    let mut channel_offsets = HashMap::new();

    if let Some(channel_dimension) = dataset.dims.c {
        for &channel in &spec.channel_ids {
            anyhow::ensure!(!cancelled(), "memory pinning was cancelled");
            let mut ranges = Vec::with_capacity(info.shape.len());
            for dimension in 0..info.shape.len() {
                if dimension == channel_dimension {
                    ranges.push(channel..channel.saturating_add(1));
                } else if dimension == dataset.dims.y || dimension == dataset.dims.x {
                    ranges.push(0..info.shape[dimension]);
                } else {
                    ranges.push(0..1);
                }
            }
            let subset = ArraySubset::new_with_ranges(&ranges);
            let data = retrieve_image_subset_u16(&array, &subset, &info.dtype)?;
            let plane = squeeze_pinned_plane(data, dataset.dims.y, dataset.dims.x)
                .ok_or_else(|| anyhow::anyhow!("unexpected pinned level dimensionality"))?;
            let (plane_raw, offset) = plane.into_raw_vec_and_offset();
            anyhow::ensure!(
                offset.unwrap_or(0) == 0,
                "non-zero pinned level buffer offset"
            );
            anyhow::ensure!(
                plane_raw.len() == plane_len,
                "unexpected pinned plane length"
            );
            channel_offsets.insert(channel, raw.len() / plane_len.max(1));
            raw.extend_from_slice(&plane_raw);
        }
    } else {
        anyhow::ensure!(!cancelled(), "memory pinning was cancelled");
        let mut ranges = Vec::with_capacity(info.shape.len());
        for dimension in 0..info.shape.len() {
            if dimension == dataset.dims.y || dimension == dataset.dims.x {
                ranges.push(0..info.shape[dimension]);
            } else {
                ranges.push(0..1);
            }
        }
        let subset = ArraySubset::new_with_ranges(&ranges);
        let data = retrieve_image_subset_u16(&array, &subset, &info.dtype)?;
        let plane = squeeze_pinned_plane(data, dataset.dims.y, dataset.dims.x)
            .ok_or_else(|| anyhow::anyhow!("unexpected pinned level dimensionality"))?;
        let (plane_raw, offset) = plane.into_raw_vec_and_offset();
        anyhow::ensure!(
            offset.unwrap_or(0) == 0,
            "non-zero pinned level buffer offset"
        );
        raw = plane_raw;
        for &channel in &spec.channel_ids {
            channel_offsets.insert(channel, 0);
        }
    }
    anyhow::ensure!(
        !channel_offsets.is_empty(),
        "none of the selected channels were pinned"
    );
    Ok(MemoryPinWorkerResult {
        system,
        outcome: MemoryPinWorkerOutcome::Loaded(ControlPinnedLevelResource::new(
            spec.level,
            width,
            height,
            channel_offsets,
            raw,
        )),
    })
}

pub(in crate::control::actor) fn load_mosaic_pinned_memory_on_worker(
    spec: &MosaicMemoryPinSpec,
    request: &OdonControlRequest,
) -> anyhow::Result<MosaicMemoryPinWorkerResult> {
    let cancelled = || {
        request
            .task_id
            .as_deref()
            .and_then(|task_id| request.task_registry.get(task_id).ok())
            .is_some_and(|task| task.state == TaskState::Cancelled)
    };
    anyhow::ensure!(!cancelled(), "mosaic memory pinning was cancelled");
    let system = {
        use sysinfo::System;
        let mut system = System::new();
        system.refresh_memory();
        (system.total_memory() > 0).then_some(SystemMemorySnapshot {
            total_bytes: system.total_memory(),
            available_bytes: system.available_memory(),
        })
    };
    let projected_bytes = spec.pinned_bytes.saturating_add(spec.estimated_bytes);
    let risk = system.and_then(|system| {
        if projected_bytes > system.available_bytes {
            Some("danger")
        } else if projected_bytes.saturating_mul(100)
            >= system.available_bytes.max(1).saturating_mul(75)
        {
            Some("warning")
        } else {
            None
        }
    });
    if let (false, Some(risk), Some(system)) = (spec.force, risk, system) {
        return Ok(MosaicMemoryPinWorkerResult {
            system: Some(system),
            outcome: MosaicMemoryPinWorkerOutcome::Confirmation {
                risk,
                projected_bytes,
                available_bytes: system.available_bytes,
            },
        });
    }

    let mut loaded = Vec::new();
    let mut failures = Vec::new();
    for item in &spec.items {
        anyhow::ensure!(!cancelled(), "mosaic memory pinning was cancelled");
        match load_mosaic_pinned_item(item, spec.level, &spec.channel_ids, &cancelled) {
            Ok(resource) => loaded.push((item.item_id, resource)),
            Err(error) => failures.push((item.item_id, error.to_string())),
        }
    }
    anyhow::ensure!(
        !loaded.is_empty(),
        "failed to pin the requested level for every selected mosaic ROI{}",
        failures
            .first()
            .map(|(_, error)| format!("; first failure: {error}"))
            .unwrap_or_default()
    );
    Ok(MosaicMemoryPinWorkerResult {
        system,
        outcome: MosaicMemoryPinWorkerOutcome::Loaded(MosaicMemoryPinResult { loaded, failures }),
    })
}

pub(in crate::control::actor) fn load_mosaic_pinned_item(
    item: &crate::model::MosaicMemoryPinItemSpec,
    level: usize,
    selected_global_channels: &[u64],
    cancelled: &impl Fn() -> bool,
) -> anyhow::Result<ControlPinnedLevelResource> {
    let descriptor = &item.document.descriptor;
    let info = descriptor
        .levels
        .get(level)
        .ok_or_else(|| anyhow::anyhow!("missing level {level}"))?;
    let zarr_path = format!("/{}", info.path.trim_start_matches('/'));
    let array: Array<dyn ReadableStorageTraits> =
        Array::open(Arc::clone(item.document.resource.store()), &zarr_path)?;
    let height = *info.shape.get(descriptor.dims.y).unwrap_or(&0) as usize;
    let width = *info.shape.get(descriptor.dims.x).unwrap_or(&0) as usize;
    let plane_len = height.saturating_mul(width);
    let mut raw = Vec::new();
    let mut channel_offsets = HashMap::new();

    if let Some(channel_dimension) = descriptor.dims.c {
        for &global_channel in selected_global_channels {
            anyhow::ensure!(!cancelled(), "mosaic memory pinning was cancelled");
            let Some(local_channel) = item
                .channel_map
                .get(global_channel as usize)
                .copied()
                .flatten()
            else {
                continue;
            };
            let mut ranges = Vec::with_capacity(info.shape.len());
            for dimension in 0..info.shape.len() {
                if dimension == channel_dimension {
                    ranges.push(local_channel..local_channel.saturating_add(1));
                } else if dimension == descriptor.dims.y || dimension == descriptor.dims.x {
                    ranges.push(0..info.shape[dimension]);
                } else {
                    ranges.push(0..1);
                }
            }
            let subset = ArraySubset::new_with_ranges(&ranges);
            let data = retrieve_image_subset_u16(&array, &subset, &info.dtype)?;
            let plane = squeeze_pinned_plane(data, descriptor.dims.y, descriptor.dims.x)
                .ok_or_else(|| anyhow::anyhow!("unexpected pinned mosaic level dimensionality"))?;
            let (plane_raw, offset) = plane.into_raw_vec_and_offset();
            anyhow::ensure!(
                offset.unwrap_or(0) == 0,
                "non-zero pinned mosaic buffer offset"
            );
            anyhow::ensure!(
                plane_raw.len() == plane_len,
                "unexpected pinned mosaic plane length"
            );
            channel_offsets.insert(global_channel, raw.len() / plane_len.max(1));
            raw.extend_from_slice(&plane_raw);
        }
    } else {
        anyhow::ensure!(!cancelled(), "mosaic memory pinning was cancelled");
        let matched = selected_global_channels
            .iter()
            .copied()
            .filter(|channel| {
                item.channel_map
                    .get(*channel as usize)
                    .copied()
                    .flatten()
                    .is_some()
            })
            .collect::<Vec<_>>();
        anyhow::ensure!(
            !matched.is_empty(),
            "none of the selected channels are present"
        );
        let mut ranges = Vec::with_capacity(info.shape.len());
        for dimension in 0..info.shape.len() {
            if dimension == descriptor.dims.y || dimension == descriptor.dims.x {
                ranges.push(0..info.shape[dimension]);
            } else {
                ranges.push(0..1);
            }
        }
        let subset = ArraySubset::new_with_ranges(&ranges);
        let data = retrieve_image_subset_u16(&array, &subset, &info.dtype)?;
        let plane = squeeze_pinned_plane(data, descriptor.dims.y, descriptor.dims.x)
            .ok_or_else(|| anyhow::anyhow!("unexpected pinned mosaic level dimensionality"))?;
        let (plane_raw, offset) = plane.into_raw_vec_and_offset();
        anyhow::ensure!(
            offset.unwrap_or(0) == 0,
            "non-zero pinned mosaic buffer offset"
        );
        raw = plane_raw;
        for channel in matched {
            channel_offsets.insert(channel, 0);
        }
    }
    anyhow::ensure!(
        !channel_offsets.is_empty(),
        "none of the selected channels were pinned"
    );
    Ok(ControlPinnedLevelResource::new(
        level,
        width,
        height,
        channel_offsets,
        raw,
    ))
}

pub(in crate::control::actor) fn squeeze_pinned_plane(
    mut data: ndarray::ArrayD<u16>,
    mut vertical_dimension: usize,
    mut horizontal_dimension: usize,
) -> Option<ndarray::Array2<u16>> {
    use ndarray::Axis;
    for dimension in (0..data.ndim()).rev() {
        if dimension == vertical_dimension || dimension == horizontal_dimension {
            continue;
        }
        if data.shape().get(dimension).copied()? != 1 {
            return None;
        }
        data = data.index_axis_move(Axis(dimension), 0);
        if dimension < vertical_dimension {
            vertical_dimension = vertical_dimension.saturating_sub(1);
        }
        if dimension < horizontal_dimension {
            horizontal_dimension = horizontal_dimension.saturating_sub(1);
        }
    }
    let mut plane = data.into_dimensionality::<ndarray::Ix2>().ok()?;
    match (vertical_dimension, horizontal_dimension) {
        (0, 1) => {}
        (1, 0) => plane.swap_axes(0, 1),
        _ => return None,
    }
    Some(plane)
}
