//! Application preferences, tile-loading policy, and pinned-memory state.

use super::*;

impl AppModel {
    pub fn bootstrap_settings(
        &mut self,
        settings: AppSettings,
        path: Option<PathBuf>,
        recent_project_exists: Vec<(PathBuf, bool)>,
    ) {
        if self.settings_operation_pending {
            return;
        }
        self.settings = settings.normalized();
        self.recent_project_exists = recent_project_exists.into_iter().collect();
        self.settings_path = path;
        self.settings_status.clear();
        self.settings_bootstrapped = true;
        self.apply_startup_shell_layout_if_needed();
    }

    pub fn settings(&self) -> &AppSettings {
        &self.settings
    }

    pub fn screenshot_preferences(&self) -> &ScreenshotPreferences {
        &self.screenshot_preferences
    }

    pub fn tile_loading_policy(&self) -> &TileLoadingPolicy {
        self.tile_loading.policy()
    }

    pub fn tile_loading_snapshot(&self) -> Result<Value, ControlError> {
        let supported =
            self.dataset()?.descriptor.kind != crate::data::document::DocumentKind::Tiff;
        Ok(self.tile_loading.snapshot(supported))
    }

    pub(super) fn set_tile_loading_policy(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let supported =
            self.dataset()?.descriptor.kind != crate::data::document::DocumentKind::Tiff;
        self.tile_loading.set(params, supported)
    }

    pub(crate) fn prepare_memory_pin(
        &mut self,
        params: &Value,
    ) -> Result<MemoryPinSpec, ControlError> {
        let dataset = self.dataset()?;
        if dataset.plane_extents[0] > 1 {
            return Err(invalid(
                "RAM pinning is currently unavailable for OME-Zarr z-stacks",
            ));
        }
        let level = params
            .get("level")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|level| *level < dataset.descriptor.levels.len())
            .ok_or_else(|| invalid("memory level is required and must be in range"))?;
        let selected_channel_indices = match params.get("channels") {
            Some(value) => {
                let values = value
                    .as_array()
                    .ok_or_else(|| invalid("channels must be an array"))?;
                resolve_channel_list_ordered(&dataset.workspace.active().state.channels, values)?
            }
            None => {
                let selected = self.pinned_memory.selected_channels().to_vec();
                if selected.is_empty() {
                    (0..dataset.workspace.active().state.channels.len()).collect()
                } else {
                    selected
                }
            }
        };
        if selected_channel_indices.is_empty() {
            return Err(invalid("select at least one channel to pin"));
        }
        let channel_ids = selected_channel_indices
            .iter()
            .map(|index| dataset.workspace.active().state.channels[*index].index as u64)
            .collect::<Vec<_>>();
        let estimated_bytes =
            estimate_pinned_level_bytes(&dataset.descriptor, level, selected_channel_indices.len());
        let status = format!(
            "Loading {} channel(s) from level {level} into RAM",
            channel_ids.len()
        );
        let operation_generation =
            self.pinned_memory
                .begin(level, selected_channel_indices.clone(), status.clone());
        self.readiness.begin_scoped(
            OperationKind::MemoryPin,
            level.to_string(),
            operation_generation,
            status,
        );
        Ok(MemoryPinSpec {
            document_generation: self.document_generation,
            operation_generation,
            level,
            channel_ids,
            estimated_bytes,
            pinned_bytes: self.pinned_memory.total_loaded_bytes(),
            force: params
                .get("force")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        })
    }

    pub(crate) fn memory_pin_is_current(&self, spec: &MemoryPinSpec) -> bool {
        self.mode == ModelMode::Single
            && spec.document_generation == self.document_generation
            && self
                .pinned_memory
                .is_current(spec.level, spec.operation_generation)
    }

    pub(crate) fn install_memory_pin(
        &mut self,
        spec: &MemoryPinSpec,
        resource: Arc<ControlPinnedLevelResource>,
        system: Option<SystemMemorySnapshot>,
    ) -> Option<Value> {
        if !self.memory_pin_is_current(spec)
            || !self
                .pinned_memory
                .install(spec.level, spec.operation_generation, resource, system)
        {
            return None;
        }
        self.readiness.finish_scoped(
            OperationKind::MemoryPin,
            &spec.level.to_string(),
            spec.operation_generation,
            "Pinned level ready",
        );
        Some(json!({
            "started":true,
            "completed":true,
            "level":spec.level,
            "estimated_bytes":spec.estimated_bytes,
            "memory":self.memory_snapshot().ok()?,
        }))
    }

    pub(crate) fn finish_memory_pin_confirmation(
        &mut self,
        spec: &MemoryPinSpec,
        system: Option<SystemMemorySnapshot>,
        risk: &str,
        projected_bytes: u64,
        available_bytes: u64,
    ) -> Option<Value> {
        if !self.memory_pin_is_current(spec)
            || !self
                .pinned_memory
                .confirmation(spec.level, spec.operation_generation, system)
        {
            return None;
        }
        self.readiness.finish_scoped(
            OperationKind::MemoryPin,
            &spec.level.to_string(),
            spec.operation_generation,
            "RAM pinning requires confirmation",
        );
        Some(json!({
            "confirmation_required":true,
            "level":spec.level,
            "requested_bytes":spec.estimated_bytes,
            "projected_bytes":projected_bytes,
            "available_bytes":available_bytes,
            "risk":risk,
        }))
    }

    pub(crate) fn fail_memory_pin(
        &mut self,
        spec: &MemoryPinSpec,
        message: impl Into<String>,
    ) -> bool {
        if !self.memory_pin_is_current(spec) {
            return false;
        }
        let message = message.into();
        self.pinned_memory
            .fail(spec.level, spec.operation_generation, message.clone());
        self.readiness.fail_scoped(
            OperationKind::MemoryPin,
            &spec.level.to_string(),
            spec.operation_generation,
            message,
        );
        true
    }

    pub(crate) fn cancel_memory_pin(
        &mut self,
        spec: &MemoryPinSpec,
        message: impl Into<String>,
    ) -> bool {
        if !self.memory_pin_is_current(spec) {
            return false;
        }
        let message = message.into();
        self.pinned_memory
            .cancel(spec.level, spec.operation_generation, message.clone());
        self.readiness.cancel_scoped(
            OperationKind::MemoryPin,
            &spec.level.to_string(),
            spec.operation_generation,
            message,
        );
        true
    }

    pub(crate) fn pinned_level_resources(&self) -> Vec<Arc<ControlPinnedLevelResource>> {
        self.pinned_memory.resources()
    }

    pub(crate) fn memory_projection_state(&mut self) -> Arc<Value> {
        let generation = match self.mode {
            ModelMode::Single => self.pinned_memory.projection_generation(),
            ModelMode::Mosaic => self.mosaic.memory_projection_generation(),
            ModelMode::Project | ModelMode::Transition => 0,
        };
        if let Some((mode, cached_generation, state)) = &self.memory_projection_cache
            && *mode == self.mode
            && *cached_generation == generation
        {
            return Arc::clone(state);
        }
        let state = Arc::new(match self.mode {
            ModelMode::Single => self.memory_snapshot().unwrap_or_else(|_| json!({})),
            ModelMode::Mosaic => self.mosaic.memory_snapshot().unwrap_or_else(|_| json!({})),
            ModelMode::Project | ModelMode::Transition => json!({}),
        });
        self.memory_projection_cache = Some((self.mode, generation, Arc::clone(&state)));
        state
    }

    pub(super) fn memory_snapshot(&self) -> Result<Value, ControlError> {
        let dataset = self.dataset()?;
        let selected = if self.pinned_memory.selected_channels().is_empty() {
            (0..dataset.workspace.active().state.channels.len()).collect::<Vec<_>>()
        } else {
            self.pinned_memory.selected_channels().to_vec()
        };
        let levels = dataset
            .descriptor
            .levels
            .iter()
            .enumerate()
            .map(|(level_index, level)| {
                let (status, bytes, channels_loaded, error) =
                    self.pinned_memory.status(level_index);
                json!({
                    "level":level_index,
                    "downsample":level.downsample,
                    "shape_y":level.shape.get(dataset.descriptor.dims.y),
                    "shape_x":level.shape.get(dataset.descriptor.dims.x),
                    "selected_channel_estimate_bytes":estimate_pinned_level_bytes(&dataset.descriptor, level_index, selected.len()),
                    "status":status,
                    "loaded_bytes":bytes,
                    "channels_loaded":channels_loaded,
                    "error":error,
                })
            })
            .collect::<Vec<_>>();
        let system = self.pinned_memory.system();
        Ok(json!({
            "running":self.pinned_memory.running(),
            "status":self.pinned_memory.status_message(),
            "pinned_bytes":self.pinned_memory.total_loaded_bytes(),
            "system":system.map(|memory| json!({"total_bytes":memory.total_bytes,"available_bytes":memory.available_bytes})),
            "selected_channels":selected,
            "z_stack_supported":dataset.plane_extents[0] <= 1,
            "levels":levels,
        }))
    }

    pub(super) fn unpin_memory(&mut self, params: &Value) -> Result<Value, ControlError> {
        let level_count = self.dataset()?.descriptor.levels.len();
        let level = params
            .get("level")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|level| *level < level_count)
            .ok_or_else(|| invalid("memory level is required and must be in range"))?;
        let pending_generation = self.pinned_memory.pending_generation(level);
        let unloaded = self.pinned_memory.unpin(level);
        if let Some(generation) = pending_generation {
            self.readiness.cancel_scoped(
                OperationKind::MemoryPin,
                &level.to_string(),
                generation,
                "Pinned level was unloaded",
            );
        }
        Ok(json!({"unloaded":unloaded,"level":level}))
    }

    pub(super) fn unpin_all_memory(&mut self) -> Result<Value, ControlError> {
        self.dataset()?;
        let count = self.pinned_memory.unpin_all();
        self.readiness
            .cancel_kind_pending(OperationKind::MemoryPin, "All pinned levels were unloaded");
        Ok(json!({"unloaded_levels":count}))
    }
}
