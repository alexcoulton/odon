//! Threshold-preview configuration, worker generations, and projection state.

use super::*;

impl AppModel {
    pub fn threshold_preview_resource(&self) -> Option<Arc<ControlThresholdPreviewResource>> {
        self.threshold_preview.preview.clone()
    }

    pub fn threshold_preview_generation(&self) -> u64 {
        self.threshold_preview.operation_generation
    }

    pub fn threshold_preview_pending(&self) -> bool {
        self.readiness.is_pending(
            OperationKind::ThresholdPreview,
            self.threshold_preview.operation_generation,
        )
    }

    pub(crate) fn threshold_preview_snapshot(&self) -> Result<Value, ControlError> {
        self.dataset()?;
        Ok(self.threshold_preview.snapshot())
    }

    pub(crate) fn configure_threshold_preview(
        &mut self,
        params: &Value,
    ) -> Result<Option<ThresholdPreviewRecomputeSpec>, ControlError> {
        let dataset = self.dataset()?;
        let scope = optional_threshold_scope(params)?;
        let level = optional_threshold_level(params, dataset.descriptor.levels.len())?;
        let min_component_pixels = optional_threshold_min_pixels(params)?;
        let threshold = optional_threshold_value(params)?;
        let has_channel = has_channel_selector(params);
        let channel = has_channel
            .then(|| {
                resolve_channel(
                    &dataset.workspace.active().state.channels,
                    channel_selector_from_params(params)?,
                )
            })
            .transpose()?;
        let active = self.threshold_preview.preview.clone();
        if threshold.is_some() && active.is_none() {
            return Err(invalid(
                "threshold can only be set after a preview has started",
            ));
        }
        if let Some(preview) = active.as_ref()
            && (scope.is_some_and(|scope| scope != preview.scope)
                || level.is_some_and(|level| level != preview.level)
                || channel.is_some_and(|channel| channel != preview.channel_index))
        {
            return Err(invalid(
                "cancel the active threshold preview before changing its scope, level, or channel",
            ));
        }
        if let Some(scope) = scope {
            self.threshold_preview.scope = scope;
        }
        if let Some(level) = level {
            self.threshold_preview.full_level = level;
        }
        if let Some(channel) = channel {
            self.dataset_mut()?
                .workspace
                .active_mut()
                .state
                .active_channel = channel;
        }
        if let Some(value) = min_component_pixels {
            self.threshold_preview.min_component_pixels = value;
        }
        let Some(active) = active else {
            return Ok(None);
        };
        let next_threshold = threshold.unwrap_or(active.threshold);
        let next_min_pixels = min_component_pixels.unwrap_or(active.min_component_pixels);
        if next_threshold == active.threshold && next_min_pixels == active.min_component_pixels {
            return Ok(None);
        }
        let generation = self.threshold_preview.next_generation();
        let mut candidate = (*active).clone();
        candidate.generation = generation;
        candidate.threshold = next_threshold;
        candidate.min_component_pixels = next_min_pixels;
        self.threshold_preview.status = "Recomputing threshold preview".to_string();
        self.readiness.begin(
            OperationKind::ThresholdPreview,
            generation,
            "Recomputing threshold preview",
        );
        Ok(Some(ThresholdPreviewRecomputeSpec {
            document_generation: self.document_generation,
            operation_generation: generation,
            preview: Arc::new(candidate),
        }))
    }

    pub(crate) fn prepare_threshold_preview_load(
        &mut self,
        params: &Value,
        refresh: bool,
    ) -> Result<ThresholdPreviewLoadSpec, ControlError> {
        let existing = self.threshold_preview.preview.clone();
        if refresh && existing.is_none() {
            return Err(invalid("no threshold preview is active"));
        }
        if !refresh && existing.is_some() {
            return Err(invalid(
                "a threshold preview is already active; refresh or cancel it first",
            ));
        }
        let dataset = self.dataset()?;
        let viewport = &dataset.workspace.active().state;
        if viewport.plane_mode != "xy" {
            return Err(invalid("threshold preview is only available in XY view"));
        }
        let requested_scope = optional_threshold_scope(params)?;
        let requested_level = optional_threshold_level(params, dataset.descriptor.levels.len())?;
        let requested_min = optional_threshold_min_pixels(params)?;
        let requested_threshold = optional_threshold_value(params)?;
        let requested_channel = if has_channel_selector(params) {
            Some(resolve_channel(
                &viewport.channels,
                channel_selector_from_params(params)?,
            )?)
        } else {
            None
        };
        let scope = existing
            .as_ref()
            .map(|preview| preview.scope)
            .or(requested_scope)
            .unwrap_or(self.threshold_preview.scope);
        let channel_index = existing
            .as_ref()
            .map(|preview| preview.channel_index)
            .or(requested_channel)
            .unwrap_or(viewport.active_channel)
            .min(viewport.channels.len().saturating_sub(1));
        let min_component_pixels = existing
            .as_ref()
            .map(|preview| preview.min_component_pixels)
            .or(requested_min)
            .unwrap_or(self.threshold_preview.min_component_pixels)
            .max(1);
        let threshold = existing
            .as_ref()
            .map(|preview| preview.threshold)
            .or(requested_threshold)
            .unwrap_or_else(|| {
                viewport
                    .channels
                    .get(channel_index)
                    .and_then(|channel| channel.window)
                    .map(|(minimum, _)| minimum.round().clamp(0.0, u16::MAX as f32) as u16)
                    .unwrap_or(0)
            });
        let level_position = if scope == ThresholdScope::EntireImage {
            existing
                .as_ref()
                .map(|preview| preview.level)
                .or(requested_level)
                .unwrap_or(self.threshold_preview.full_level)
        } else {
            dataset
                .descriptor
                .levels
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| {
                    let left_error = (viewport.zoom * left.downsample.max(1e-6)).ln().abs();
                    let right_error = (viewport.zoom * right.downsample.max(1e-6)).ln().abs();
                    left_error.total_cmp(&right_error)
                })
                .map(|(position, _)| position)
                .unwrap_or(0)
        };
        let level = dataset
            .descriptor
            .levels
            .get(level_position)
            .ok_or_else(|| invalid(format!("threshold level {level_position} is out of range")))?;
        let (x0, y0, x1, y1) = threshold_extent(dataset, viewport, channel_index, scope, level)?;
        let width = usize::try_from(x1.saturating_sub(x0))
            .map_err(|_| invalid("threshold width is too large"))?;
        let height = usize::try_from(y1.saturating_sub(y0))
            .map_err(|_| invalid("threshold height is too large"))?;
        let pixels = (width as u64)
            .checked_mul(height as u64)
            .ok_or_else(|| invalid("threshold region is too large"))?;
        if pixels == 0 {
            return Err(invalid("threshold region is empty"));
        }
        if pixels > THRESHOLD_REGION_MAX_INTERACTIVE_PIXELS {
            return Err(invalid(format!(
                "thresholding at this level would read {pixels} pixels; choose a coarser level"
            )));
        }
        let level0 = dataset
            .descriptor
            .levels
            .first()
            .ok_or_else(|| invalid("dataset has no image levels"))?;
        let channel = viewport
            .channels
            .get(channel_index)
            .ok_or_else(|| invalid("threshold channel is out of range"))?;
        let z_index = dataset.descriptor.dims.z.and_then(|dimension| {
            map_level0_axis_index(level0, level, dimension, viewport.plane_slices[0])
        });
        let ranges = (0..level.shape.len())
            .map(|dimension| {
                let length = level.shape[dimension];
                if Some(dimension) == dataset.descriptor.dims.c {
                    let selected = (channel.index as u64).min(length.saturating_sub(1));
                    selected..selected.saturating_add(1)
                } else if Some(dimension) == dataset.descriptor.dims.z {
                    let selected = z_index.unwrap_or(0).min(length.saturating_sub(1));
                    selected..selected.saturating_add(1)
                } else if dimension == dataset.descriptor.dims.y {
                    y0.min(length)..y1.min(length)
                } else if dimension == dataset.descriptor.dims.x {
                    x0.min(length)..x1.min(length)
                } else {
                    0..length.min(1)
                }
            })
            .collect::<Vec<_>>();
        let channel_name = channel.name.clone();
        let document_generation = self.document_generation;
        let zarr_path = format!("/{}", level.path.trim_start_matches('/'));
        let dtype = level.dtype.clone();
        let downsample = level.downsample.max(1e-6);
        if !refresh {
            self.threshold_preview.scope = scope;
            self.threshold_preview.full_level = level_position;
            self.threshold_preview.min_component_pixels = min_component_pixels;
            self.dataset_mut()?
                .workspace
                .active_mut()
                .state
                .active_channel = channel_index;
        }
        let operation_generation = self.threshold_preview.next_generation();
        self.threshold_preview.status = if refresh {
            "Refreshing threshold preview".to_string()
        } else {
            "Loading threshold preview".to_string()
        };
        self.readiness.begin(
            OperationKind::ThresholdPreview,
            operation_generation,
            self.threshold_preview.status.clone(),
        );
        Ok(ThresholdPreviewLoadSpec {
            document_generation,
            operation_generation,
            channel_index,
            channel_name,
            scope,
            level: level_position,
            downsample,
            x0,
            y0,
            width,
            height,
            zarr_path,
            dtype,
            ranges,
            threshold,
            min_component_pixels,
        })
    }

    pub(crate) fn threshold_operation_is_current(
        &self,
        document_generation: u64,
        operation_generation: u64,
    ) -> bool {
        self.mode == ModelMode::Single
            && document_generation == self.document_generation
            && operation_generation == self.threshold_preview.operation_generation
    }

    pub(crate) fn install_threshold_preview(
        &mut self,
        document_generation: u64,
        operation_generation: u64,
        preview: Arc<ControlThresholdPreviewResource>,
    ) -> Option<Value> {
        if !self.threshold_operation_is_current(document_generation, operation_generation) {
            return None;
        }
        self.threshold_preview.status = format!(
            "Preview: {} pixels selected in {} {} at level {}.",
            preview
                .included
                .iter()
                .filter(|included| **included)
                .count(),
            preview.channel_name,
            preview.scope.as_str(),
            preview.level,
        );
        self.threshold_preview.preview = Some(preview);
        self.readiness.finish(
            OperationKind::ThresholdPreview,
            operation_generation,
            "Threshold preview ready",
        );
        self.threshold_preview_snapshot().ok()
    }

    pub(crate) fn prepare_threshold_preview_apply(
        &mut self,
    ) -> Result<ThresholdPreviewApplySpec, ControlError> {
        let preview = self
            .threshold_preview
            .preview
            .clone()
            .ok_or_else(|| invalid("no threshold preview is active"))?;
        let dataset = self.dataset()?;
        let viewport = &dataset.workspace.active().state;
        let channel = viewport
            .channels
            .get(preview.channel_index)
            .ok_or_else(|| invalid("threshold channel is no longer available"))?;
        let spec = ThresholdPreviewApplySpec {
            document_generation: self.document_generation,
            operation_generation: self
                .threshold_preview
                .operation_generation
                .wrapping_add(1)
                .max(1),
            preview,
            pivot: [dataset.world_size[0] * 0.5, dataset.world_size[1] * 0.5],
            offset: channel.offset_world,
            scale: channel.scale,
            rotation_rad: channel.rotation_rad,
        };
        self.threshold_preview.operation_generation = spec.operation_generation;
        self.threshold_preview.status = "Creating threshold mask".to_string();
        self.readiness.begin(
            OperationKind::ThresholdPreview,
            spec.operation_generation,
            "Creating threshold mask",
        );
        Ok(spec)
    }

    pub(crate) fn install_threshold_mask(
        &mut self,
        spec: &ThresholdPreviewApplySpec,
        polygons_world: Vec<Vec<[f32; 2]>>,
    ) -> Option<Value> {
        if !self.threshold_operation_is_current(spec.document_generation, spec.operation_generation)
        {
            return None;
        }
        let dataset = self.dataset.as_mut()?;
        let response = dataset.masks.install_generated_threshold_layer(
            format!(
                "Threshold {} {} level {}",
                spec.preview.channel_name,
                spec.preview.scope.layer_label(),
                spec.preview.level,
            ),
            polygons_world,
        );
        Self::sync_mask_native_layers(dataset);
        self.threshold_preview.preview = None;
        self.threshold_preview.status = format!(
            "Created {} threshold region(s) from {}.",
            response["polygon_count"].as_u64().unwrap_or(0),
            spec.preview.channel_name,
        );
        self.readiness.finish(
            OperationKind::ThresholdPreview,
            spec.operation_generation,
            "Threshold mask ready",
        );
        Some(json!({
            "applied":true,
            "layer_id":response["id"],
            "polygon_count":response["polygon_count"],
            "mask_layer":response,
        }))
    }

    pub(crate) fn fail_threshold_operation(
        &mut self,
        document_generation: u64,
        operation_generation: u64,
        message: impl Into<String>,
    ) -> bool {
        if !self.threshold_operation_is_current(document_generation, operation_generation) {
            return false;
        }
        let message = message.into();
        self.threshold_preview.status = message.clone();
        self.readiness.fail(
            OperationKind::ThresholdPreview,
            operation_generation,
            message,
        );
        true
    }

    pub(crate) fn cancel_threshold_preview(&mut self) -> Result<Value, ControlError> {
        self.dataset()?;
        let cancelled = self.threshold_preview.preview.take().is_some()
            || self.readiness.is_pending(
                OperationKind::ThresholdPreview,
                self.threshold_preview.operation_generation,
            );
        self.threshold_preview.next_generation();
        self.threshold_preview.status.clear();
        self.readiness.cancel_kind_pending(
            OperationKind::ThresholdPreview,
            "Threshold preview was cancelled",
        );
        Ok(json!({"cancelled":cancelled,"active":false}))
    }
}
