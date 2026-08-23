//! Actor-owned channel histogram and automatic-contrast operation state.

use super::*;

const DEFAULT_MAX_SCAN_PIXELS: u64 = 2_000_000;

#[derive(Debug, Clone)]
pub(super) struct ChannelComputeModel {
    projection_generation: u64,
    intensity_operation_generation: u64,
    intensity_pending: bool,
    intensity_state: Value,
    auto_contrast_operation_generation: u64,
    auto_contrast_pending: bool,
    auto_contrast_document_generation: u64,
    auto_contrast_on_open_queued: bool,
    auto_contrast_state: Value,
}

impl Default for ChannelComputeModel {
    fn default() -> Self {
        Self {
            projection_generation: 0,
            intensity_operation_generation: 0,
            intensity_pending: false,
            intensity_state: json!({"pending":false}),
            auto_contrast_operation_generation: 0,
            auto_contrast_pending: false,
            auto_contrast_document_generation: 0,
            auto_contrast_on_open_queued: false,
            auto_contrast_state: json!({"pending":false}),
        }
    }
}

impl ChannelComputeModel {
    fn touch(&mut self) {
        self.projection_generation = self.projection_generation.wrapping_add(1).max(1);
    }

    fn reset_document(&mut self, document_generation: u64, auto_contrast_on_open: bool) {
        self.intensity_operation_generation =
            self.intensity_operation_generation.wrapping_add(1).max(1);
        self.intensity_pending = false;
        self.intensity_state = json!({"pending":false,"document_generation":document_generation});
        self.auto_contrast_operation_generation = self
            .auto_contrast_operation_generation
            .wrapping_add(1)
            .max(1);
        self.auto_contrast_pending = false;
        self.auto_contrast_document_generation = document_generation;
        self.auto_contrast_on_open_queued = auto_contrast_on_open;
        self.auto_contrast_state = json!({
            "pending":false,
            "queued":auto_contrast_on_open,
            "document_generation":document_generation,
        });
        self.touch();
    }
}

impl AppModel {
    pub(super) fn channel_compute_viewport_id(
        &self,
        params: &Value,
    ) -> Result<ViewportId, ControlError> {
        match params
            .get("viewport_id")
            .or_else(|| params.get("id"))
            .and_then(Value::as_str)
        {
            Some(id) => ViewportId::new(id).map_err(|error| invalid(error.to_string())),
            None => Ok(self.dataset()?.workspace.active_id().clone()),
        }
    }

    pub fn channel_compute_generation(&self) -> u64 {
        self.channel_compute.projection_generation
    }

    pub fn channel_compute_state(&self) -> Value {
        json!({
            "generation":self.channel_compute.projection_generation,
            "histogram":self.channel_compute.intensity_state,
            "auto_contrast":self.channel_compute.auto_contrast_state,
        })
    }

    pub(crate) fn reset_channel_compute_for_document(&mut self) {
        self.channel_compute.reset_document(
            self.document_generation,
            self.settings.auto_contrast.enabled_on_open,
        );
        self.readiness.cancel_kind_pending(
            OperationKind::ChannelCompute,
            "Channel compute superseded by dataset replacement",
        );
    }

    pub(crate) fn begin_channel_intensity_operation(
        &mut self,
        client_request_id: Option<u64>,
    ) -> u64 {
        let generation = self
            .channel_compute
            .intensity_operation_generation
            .wrapping_add(1)
            .max(1);
        self.channel_compute.intensity_operation_generation = generation;
        self.channel_compute.intensity_pending = true;
        self.channel_compute.intensity_state = json!({
            "pending":true,
            "operation_generation":generation,
            "document_generation":self.document_generation,
            "request_id":client_request_id,
        });
        self.channel_compute.touch();
        self.readiness.begin_scoped(
            OperationKind::ChannelCompute,
            "histogram",
            generation,
            "Computing channel histogram",
        );
        generation
    }

    pub(crate) fn finish_channel_intensity_operation(
        &mut self,
        document_generation: u64,
        operation_generation: u64,
        value: &Value,
    ) -> bool {
        if document_generation != self.document_generation
            || operation_generation != self.channel_compute.intensity_operation_generation
        {
            return false;
        }
        self.channel_compute.intensity_pending = false;
        self.channel_compute.intensity_state = value.clone();
        if let Some(state) = self.channel_compute.intensity_state.as_object_mut() {
            state.insert("pending".to_string(), Value::Bool(false));
            state.insert(
                "operation_generation".to_string(),
                json!(operation_generation),
            );
            state.insert(
                "document_generation".to_string(),
                json!(document_generation),
            );
        }
        self.channel_compute.touch();
        self.readiness.finish_scoped(
            OperationKind::ChannelCompute,
            "histogram",
            operation_generation,
            "Channel histogram ready",
        );
        true
    }

    pub(crate) fn fail_channel_intensity_operation(
        &mut self,
        document_generation: u64,
        operation_generation: u64,
        message: impl Into<String>,
    ) -> bool {
        if document_generation != self.document_generation
            || operation_generation != self.channel_compute.intensity_operation_generation
        {
            return false;
        }
        let message = message.into();
        self.channel_compute.intensity_pending = false;
        self.channel_compute.intensity_state = json!({
            "pending":false,
            "operation_generation":operation_generation,
            "document_generation":document_generation,
            "error":message,
        });
        self.channel_compute.touch();
        self.readiness.fail_scoped(
            OperationKind::ChannelCompute,
            "histogram",
            operation_generation,
            message,
        );
        true
    }

    pub(crate) fn auto_contrast_on_open_spec(
        &self,
        dataset: &OmeZarrDataset,
    ) -> Result<Option<AutoContrastSpec>, ControlError> {
        if !self.channel_compute.auto_contrast_on_open_queued
            || self.channel_compute.auto_contrast_pending
            || self.mode != ModelMode::Single
        {
            return Ok(None);
        }
        self.build_auto_contrast_spec(dataset, &json!({"overwrite_manual":false}), false)
            .map(Some)
    }

    pub(crate) fn prepare_auto_contrast(
        &mut self,
        dataset: &OmeZarrDataset,
        params: &Value,
    ) -> Result<AutoContrastSpec, ControlError> {
        let overwrite_manual = match params.get("overwrite_manual") {
            Some(value) => value
                .as_bool()
                .ok_or_else(|| invalid("overwrite_manual must be a boolean"))?,
            None => true,
        };
        if overwrite_manual {
            let viewport_id = self.channel_compute_viewport_id(params)?;
            let requested = self.auto_contrast_channel_indices(&viewport_id, params, true)?;
            let workspace = &mut self.dataset_mut()?.workspace;
            let viewport = workspace
                .get_mut(&viewport_id)
                .ok_or_else(|| not_found(&viewport_id))?;
            for index in requested {
                if let Some(channel) = viewport.state.channels.get_mut(index) {
                    channel.contrast_manual = false;
                }
            }
        }
        self.build_auto_contrast_spec(dataset, params, true)
    }

    fn build_auto_contrast_spec(
        &self,
        dataset: &OmeZarrDataset,
        params: &Value,
        explicit: bool,
    ) -> Result<AutoContrastSpec, ControlError> {
        let viewport_id = self.channel_compute_viewport_id(params)?;
        let channel_indices = self.auto_contrast_channel_indices(&viewport_id, params, explicit)?;
        let viewport = &self
            .dataset()?
            .workspace
            .get(&viewport_id)
            .ok_or_else(|| not_found(&viewport_id))?
            .state;
        let level = choose_auto_contrast_level(dataset, viewport);
        let operation_generation = self
            .channel_compute
            .auto_contrast_operation_generation
            .wrapping_add(1)
            .max(1);
        let channels = channel_indices
            .into_iter()
            .map(|index| {
                let channel = viewport
                    .channels
                    .get(index)
                    .ok_or_else(|| invalid(format!("channel index {index} is out of range")))?;
                let mut spec_params = json!({
                    "viewport_id":viewport_id.as_str(),
                    "channel":index,
                    "level":level,
                });
                spec_params["bins"] = Value::Null;
                Ok(AutoContrastChannelSpec {
                    intensity: self.channel_intensity_spec(dataset, &spec_params)?,
                    baseline_window: channel.window,
                })
            })
            .collect::<Result<Vec<_>, ControlError>>()?;
        if channels.is_empty() {
            return Err(invalid("no channels are eligible for automatic contrast"));
        }
        let mut settings = self.settings.auto_contrast;
        if let Some(value) = params.get("method") {
            let method = value
                .as_str()
                .ok_or_else(|| invalid("method must be a string"))?;
            settings.method = match method {
                "zero_to_p97" => AutoContrastMethod::ZeroToP97,
                "p1_to_p99" => AutoContrastMethod::P1ToP99,
                "zero_to_max" => AutoContrastMethod::ZeroToMax,
                _ => return Err(invalid(format!("unknown auto-contrast method '{method}'"))),
            };
        }
        if let Some(value) = params.get("lower_percentile") {
            let lower = value
                .as_u64()
                .ok_or_else(|| invalid("lower_percentile must be an integer from 0 to 99"))?;
            settings.lower_percentile = u8::try_from(lower)
                .map_err(|_| invalid("lower_percentile must be from 0 to 99"))?;
        }
        if let Some(value) = params.get("upper_percentile") {
            let upper = value
                .as_u64()
                .ok_or_else(|| invalid("upper_percentile must be an integer from 1 to 100"))?;
            settings.upper_percentile = u8::try_from(upper)
                .map_err(|_| invalid("upper_percentile must be from 1 to 100"))?;
        }
        if settings.lower_percentile >= settings.upper_percentile
            || settings.lower_percentile > 99
            || settings.upper_percentile == 0
            || settings.upper_percentile > 100
        {
            return Err(invalid("automatic-contrast percentiles are invalid"));
        }
        Ok(AutoContrastSpec {
            document_generation: self.document_generation,
            operation_generation,
            viewport_id,
            settings: settings.normalized(),
            channels,
        })
    }

    fn auto_contrast_channel_indices(
        &self,
        viewport_id: &ViewportId,
        params: &Value,
        explicit: bool,
    ) -> Result<Vec<usize>, ControlError> {
        let viewport = &self
            .dataset()?
            .workspace
            .get(viewport_id)
            .ok_or_else(|| not_found(viewport_id))?
            .state;
        let requested = match params.get("channels") {
            Some(value) => resolve_channel_list_ordered(
                &viewport.channels,
                value
                    .as_array()
                    .ok_or_else(|| invalid("channels must be an array"))?,
            )?,
            None => (0..viewport.channels.len()).collect(),
        };
        Ok(requested
            .into_iter()
            .filter(|index| {
                explicit
                    || !viewport
                        .channels
                        .get(*index)
                        .is_some_and(|channel| channel.contrast_manual)
            })
            .collect())
    }

    pub(crate) fn mark_auto_contrast_started(&mut self, spec: &AutoContrastSpec) -> bool {
        if spec.document_generation != self.document_generation
            || spec.operation_generation <= self.channel_compute.auto_contrast_operation_generation
        {
            return false;
        }
        self.channel_compute.auto_contrast_operation_generation = spec.operation_generation;
        self.channel_compute.auto_contrast_pending = true;
        self.channel_compute.auto_contrast_on_open_queued = false;
        self.channel_compute.auto_contrast_state = json!({
            "pending":true,
            "operation_generation":spec.operation_generation,
            "document_generation":spec.document_generation,
            "viewport_id":spec.viewport_id.as_str(),
            "channels":spec.channels.len(),
            "method":spec.settings.method,
        });
        self.channel_compute.touch();
        self.readiness.begin_scoped(
            OperationKind::ChannelCompute,
            "auto_contrast",
            spec.operation_generation,
            "Computing automatic contrast",
        );
        true
    }

    pub(crate) fn fail_auto_contrast_on_open_preparation(
        &mut self,
        message: impl Into<String>,
    ) -> bool {
        if !self.channel_compute.auto_contrast_on_open_queued
            || self.channel_compute.auto_contrast_pending
        {
            return false;
        }
        self.channel_compute.auto_contrast_on_open_queued = false;
        self.channel_compute.auto_contrast_state = json!({
            "pending":false,
            "queued":false,
            "document_generation":self.document_generation,
            "error":message.into(),
        });
        self.channel_compute.touch();
        true
    }

    pub(crate) fn install_auto_contrast(
        &mut self,
        spec: &AutoContrastSpec,
        results: &[AutoContrastChannelResult],
    ) -> Option<Value> {
        if spec.document_generation != self.document_generation
            || spec.operation_generation != self.channel_compute.auto_contrast_operation_generation
            || !self.channel_compute.auto_contrast_pending
        {
            return None;
        }
        let abs_max = self.dataset().ok()?.descriptor.abs_max.max(1.0);
        let workspace = &mut self.dataset_mut().ok()?.workspace;
        let active_before = workspace.active().state.clone();
        let viewport = workspace.get_mut(&spec.viewport_id)?;
        let mut applied = Vec::new();
        let mut skipped = Vec::new();
        for (channel_spec, result) in spec.channels.iter().zip(results) {
            let index = channel_spec.intensity.channel_index;
            if result.channel_index != index {
                skipped.push(json!({"index":index,"reason":"worker_channel_mismatch"}));
                continue;
            }
            let Some(channel) = viewport.state.channels.get_mut(index) else {
                skipped.push(json!({"index":index,"reason":"channel_removed"}));
                continue;
            };
            if channel.window != channel_spec.baseline_window || channel.contrast_manual {
                skipped.push(json!({
                    "index":index,
                    "name":channel.name,
                    "reason":"contrast_changed_after_request",
                }));
                continue;
            }
            let mut min = (result.min as f32).clamp(0.0, abs_max);
            let mut max = (result.max as f32).clamp(0.0, abs_max);
            if min >= abs_max {
                min = (abs_max - 1.0).max(0.0);
            }
            if max <= min {
                max = (min + 1.0).min(abs_max);
            }
            channel.window = Some((min, max));
            channel.contrast_manual = false;
            applied.push(json!({
                "index":index,
                "name":result.channel_name,
                "min":min,
                "max":max,
                "n":result.sample_count,
            }));
        }
        if !applied.is_empty() {
            let _ = workspace.bump_presentation_revision(&spec.viewport_id);
        }
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        let response = viewport_response(
            workspace,
            &spec.viewport_id,
            json!({
                "applied":applied,
                "skipped":skipped,
                "method":spec.settings.method,
                "completed":true,
            }),
            vec![spec.viewport_id.clone()],
            active_changed,
        );
        self.channel_compute.auto_contrast_pending = false;
        self.channel_compute.auto_contrast_document_generation = spec.document_generation;
        self.channel_compute.auto_contrast_state = response.clone();
        if let Some(state) = self.channel_compute.auto_contrast_state.as_object_mut() {
            state.insert("pending".to_string(), Value::Bool(false));
            state.insert("completed".to_string(), Value::Bool(true));
            state.insert(
                "operation_generation".to_string(),
                json!(spec.operation_generation),
            );
            state.insert(
                "document_generation".to_string(),
                json!(spec.document_generation),
            );
        }
        self.channel_compute.touch();
        self.readiness.finish_scoped(
            OperationKind::ChannelCompute,
            "auto_contrast",
            spec.operation_generation,
            "Automatic contrast ready",
        );
        Some(response)
    }

    pub(crate) fn fail_auto_contrast(
        &mut self,
        spec: &AutoContrastSpec,
        message: impl Into<String>,
    ) -> bool {
        if spec.document_generation != self.document_generation
            || spec.operation_generation != self.channel_compute.auto_contrast_operation_generation
        {
            return false;
        }
        let message = message.into();
        self.channel_compute.auto_contrast_pending = false;
        self.channel_compute.auto_contrast_on_open_queued = false;
        self.channel_compute.auto_contrast_state = json!({
            "pending":false,
            "operation_generation":spec.operation_generation,
            "document_generation":spec.document_generation,
            "error":message,
        });
        self.channel_compute.touch();
        self.readiness.fail_scoped(
            OperationKind::ChannelCompute,
            "auto_contrast",
            spec.operation_generation,
            message,
        );
        true
    }
}

fn choose_auto_contrast_level(dataset: &OmeZarrDataset, viewport: &ViewportModel) -> usize {
    let (vertical, horizontal) = match viewport.plane_mode.as_str() {
        "xz" => (dataset.dims.z.unwrap_or(dataset.dims.y), dataset.dims.x),
        "yz" => (dataset.dims.z.unwrap_or(dataset.dims.y), dataset.dims.y),
        _ => (dataset.dims.y, dataset.dims.x),
    };
    let mut chosen = dataset.levels.len().saturating_sub(1);
    for (index, level) in dataset.levels.iter().enumerate().rev() {
        let pixels = level
            .shape
            .get(vertical)
            .copied()
            .unwrap_or(0)
            .saturating_mul(level.shape.get(horizontal).copied().unwrap_or(0));
        if pixels > 0 && pixels <= DEFAULT_MAX_SCAN_PIXELS {
            chosen = index;
        }
    }
    chosen
}
