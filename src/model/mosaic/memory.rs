//! Mosaic memory planning, pin lifecycle, and memory projections.

use super::*;

impl MosaicModel {
    fn touch_memory_projection(&mut self) {
        self.memory_projection_generation =
            self.memory_projection_generation.wrapping_add(1).max(1);
    }

    pub(crate) fn memory_projection_generation(&self) -> u64 {
        self.memory_projection_generation
    }

    pub(crate) fn pinned_level_resources(&self) -> Vec<(usize, Arc<ControlPinnedLevelResource>)> {
        self.pinned_levels
            .iter()
            .filter_map(|((item_id, _), state)| match state {
                MosaicPinnedLevelState::Loaded(resource) => Some((*item_id, Arc::clone(resource))),
                MosaicPinnedLevelState::Failed(_) => None,
            })
            .collect()
    }

    pub(crate) fn prepare_memory_pin(
        &mut self,
        params: &Value,
    ) -> Result<MosaicMemoryPinSpec, ControlError> {
        let resource = Arc::clone(self.require_resource()?);
        let level = params
            .get("level")
            .and_then(Value::as_u64)
            .and_then(|level| usize::try_from(level).ok())
            .ok_or_else(|| invalid("memory level is required"))?;
        let selected_channels = match params.get("channels") {
            Some(value) => value
                .as_array()
                .ok_or_else(|| invalid("channels must be an array"))?
                .iter()
                .map(|selector| self.channel_index(selector))
                .collect::<Result<Vec<_>, _>>()?,
            None if self.memory_selected_channels.is_empty() => (0..self.channels.len()).collect(),
            None => self.memory_selected_channels.clone(),
        };
        let mut selected_channels = selected_channels;
        selected_channels.sort_unstable();
        selected_channels.dedup();
        if selected_channels.is_empty() {
            return Err(invalid("select at least one channel to pin"));
        }
        let item_ids = self.memory_item_ids(params)?;
        let mut items = Vec::new();
        let mut estimated_bytes = 0_u64;
        for item_id in item_ids {
            let Some(item) = resource.items.iter().find(|item| item.id == item_id) else {
                continue;
            };
            let z_extent = item
                .document
                .descriptor
                .dims
                .z
                .and_then(|dimension| {
                    item.document
                        .descriptor
                        .levels
                        .first()
                        .and_then(|level| level.shape.get(dimension))
                })
                .copied()
                .unwrap_or(1);
            if z_extent > 1 {
                return Err(invalid(format!(
                    "RAM pinning is unavailable for z-stack mosaic ROI '{}'",
                    item.roi_id
                )));
            }
            if level >= item.document.descriptor.levels.len() {
                continue;
            }
            let channel_map = self
                .channels
                .iter()
                .map(|global| {
                    item.document
                        .descriptor
                        .channels
                        .iter()
                        .find(|local| local.name == global.name)
                        .map(|local| local.index as u64)
                })
                .collect::<Vec<_>>();
            let present_channels = selected_channels
                .iter()
                .filter(|index| channel_map.get(**index).copied().flatten().is_some())
                .count();
            if present_channels == 0 {
                continue;
            }
            estimated_bytes = estimated_bytes.saturating_add(estimate_level_bytes(
                &item.document,
                level,
                present_channels,
            ));
            items.push(MosaicMemoryPinItemSpec {
                item_id,
                document: item.document.clone(),
                channel_map,
            });
        }
        if items.is_empty() {
            return Err(invalid(
                "the requested level and channels are unavailable for the selected mosaic scope",
            ));
        }
        self.memory_operation_generation = self.memory_operation_generation.wrapping_add(1).max(1);
        self.memory_selected_channels = selected_channels.clone();
        for item in &items {
            self.memory_pending
                .insert((item.item_id, level), self.memory_operation_generation);
        }
        self.memory_status = format!(
            "Loading {} channel(s) from level {level} into RAM for {} ROI(s)",
            selected_channels.len(),
            items.len()
        );
        self.touch_memory_projection();
        Ok(MosaicMemoryPinSpec {
            resource_generation: self.resource_generation(),
            operation_generation: self.memory_operation_generation,
            level,
            channel_ids: selected_channels
                .iter()
                .map(|index| *index as u64)
                .collect(),
            items,
            estimated_bytes,
            pinned_bytes: self.pinned_bytes(),
            force: params
                .get("force")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        })
    }

    pub(crate) fn memory_pin_is_current(&self, spec: &MosaicMemoryPinSpec) -> bool {
        self.resource_generation() == spec.resource_generation
            && spec.items.iter().all(|item| {
                self.memory_pending
                    .get(&(item.item_id, spec.level))
                    .copied()
                    == Some(spec.operation_generation)
            })
    }

    pub(crate) fn install_memory_pin(
        &mut self,
        spec: &MosaicMemoryPinSpec,
        result: MosaicMemoryPinResult,
        system: Option<SystemMemorySnapshot>,
    ) -> Option<Value> {
        if !self.memory_pin_is_current(spec) {
            return None;
        }
        for item in &spec.items {
            self.memory_pending.remove(&(item.item_id, spec.level));
        }
        for (item_id, resource) in result.loaded {
            self.pinned_levels.insert(
                (item_id, spec.level),
                MosaicPinnedLevelState::Loaded(Arc::new(resource)),
            );
        }
        for (item_id, error) in result.failures {
            self.pinned_levels
                .insert((item_id, spec.level), MosaicPinnedLevelState::Failed(error));
        }
        self.system_memory = system;
        self.memory_status = format!("Pinned mosaic level {} into RAM", spec.level);
        self.touch_memory_projection();
        Some(json!({
            "started":true,
            "completed":true,
            "level":spec.level,
            "items":spec.items.len(),
            "estimated_bytes":spec.estimated_bytes,
            "memory":self.memory_snapshot().ok()?,
        }))
    }

    pub(crate) fn finish_memory_confirmation(
        &mut self,
        spec: &MosaicMemoryPinSpec,
        system: Option<SystemMemorySnapshot>,
        risk: &str,
        projected_bytes: u64,
        available_bytes: u64,
    ) -> Option<Value> {
        if !self.memory_pin_is_current(spec) {
            return None;
        }
        for item in &spec.items {
            self.memory_pending.remove(&(item.item_id, spec.level));
        }
        self.system_memory = system;
        self.memory_status = format!(
            "RAM pinning mosaic level {} requires confirmation",
            spec.level
        );
        self.touch_memory_projection();
        Some(json!({
            "confirmation_required":true,
            "level":spec.level,
            "items":spec.items.len(),
            "requested_bytes":spec.estimated_bytes,
            "projected_bytes":projected_bytes,
            "available_bytes":available_bytes,
            "risk":risk,
        }))
    }

    pub(crate) fn fail_memory_pin(
        &mut self,
        spec: &MosaicMemoryPinSpec,
        message: impl Into<String>,
    ) -> bool {
        if !self.memory_pin_is_current(spec) {
            return false;
        }
        let message = message.into();
        for item in &spec.items {
            self.memory_pending.remove(&(item.item_id, spec.level));
            self.pinned_levels.insert(
                (item.item_id, spec.level),
                MosaicPinnedLevelState::Failed(message.clone()),
            );
        }
        self.memory_status = message;
        self.touch_memory_projection();
        true
    }

    pub(crate) fn cancel_memory_pin(
        &mut self,
        spec: &MosaicMemoryPinSpec,
        message: impl Into<String>,
    ) -> bool {
        if !self.memory_pin_is_current(spec) {
            return false;
        }
        for item in &spec.items {
            self.memory_pending.remove(&(item.item_id, spec.level));
        }
        self.memory_status = message.into();
        self.touch_memory_projection();
        true
    }

    pub(super) fn memory_item_ids(&self, params: &Value) -> Result<Vec<usize>, ControlError> {
        match params
            .get("scope")
            .and_then(Value::as_str)
            .unwrap_or("focused")
        {
            "all" => Ok(self.items.iter().map(|item| item.id).collect()),
            "focused" => self
                .focused_id
                .map(|id| vec![id])
                .ok_or_else(|| invalid("mosaic has no focused ROI")),
            "item" => {
                let selector = params
                    .get("item")
                    .ok_or_else(|| invalid("item is required when scope is item"))?;
                let id = if let Some(id) = selector.as_u64() {
                    usize::try_from(id)
                        .ok()
                        .filter(|id| self.items.iter().any(|item| item.id == *id))
                } else if let Some(roi_id) = selector.as_str() {
                    self.items
                        .iter()
                        .find(|item| item.roi_id == roi_id)
                        .map(|item| item.id)
                } else {
                    None
                };
                id.map(|id| vec![id])
                    .ok_or_else(|| invalid(format!("unknown mosaic item selector: {selector}")))
            }
            scope => Err(invalid(format!(
                "unknown memory scope '{scope}'; use focused, item, or all"
            ))),
        }
    }

    pub(crate) fn memory_snapshot(&self) -> Result<Value, ControlError> {
        let resource = self.require_resource()?;
        let selected = if self.memory_selected_channels.is_empty() {
            (0..self.channels.len()).collect::<Vec<_>>()
        } else {
            self.memory_selected_channels.clone()
        };
        let items = resource
            .items
            .iter()
            .map(|item| {
                let levels = item
                    .document
                    .descriptor
                    .levels
                    .iter()
                    .enumerate()
                    .map(|(level, descriptor)| {
                        let key = (item.id, level);
                        let (status, bytes, channels_loaded, error) =
                            if self.memory_pending.contains_key(&key) {
                                ("loading", None, None, None)
                            } else {
                                match self.pinned_levels.get(&key) {
                                    None => ("unloaded", None, None, None),
                                    Some(MosaicPinnedLevelState::Loaded(resource)) => (
                                        "loaded",
                                        Some(resource.bytes()),
                                        Some(resource.channels_loaded()),
                                        None,
                                    ),
                                    Some(MosaicPinnedLevelState::Failed(error)) => {
                                        ("failed", None, None, Some(error.as_str()))
                                    }
                                }
                            };
                        let present = selected
                            .iter()
                            .filter(|index| {
                                self.channels.get(**index).is_some_and(|global| {
                                    item.document
                                        .descriptor
                                        .channels
                                        .iter()
                                        .any(|local| local.name == global.name)
                                })
                            })
                            .count();
                        json!({
                            "level":level,
                            "shape":descriptor.shape,
                            "selected_channel_estimate_bytes":estimate_level_bytes(&item.document, level, present),
                            "status":status,
                            "loaded_bytes":bytes,
                            "channels_loaded":channels_loaded,
                            "error":error,
                        })
                    })
                    .collect::<Vec<_>>();
                json!({
                    "id":item.id,
                    "sample_id":item.roi_id,
                    "focused":self.focused_id == Some(item.id),
                    "levels":levels,
                })
            })
            .collect::<Vec<_>>();
        Ok(json!({
            "mode":"mosaic",
            "running":!self.memory_pending.is_empty(),
            "status":self.memory_status,
            "pinned_bytes":self.pinned_bytes(),
            "system":self.system_memory.map(|memory| json!({
                "total_bytes":memory.total_bytes,
                "available_bytes":memory.available_bytes,
            })),
            "selected_channels":selected,
            "items":items,
        }))
    }

    pub(super) fn unpin_memory(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let level = params
            .get("level")
            .and_then(Value::as_u64)
            .and_then(|level| usize::try_from(level).ok())
            .ok_or_else(|| invalid("memory level is required"))?;
        let item_ids = self.memory_item_ids(params)?;
        self.memory_operation_generation = self.memory_operation_generation.wrapping_add(1).max(1);
        let mut unloaded = 0;
        for item_id in item_ids {
            self.memory_pending.remove(&(item_id, level));
            if self.pinned_levels.remove(&(item_id, level)).is_some() {
                unloaded += 1;
            }
        }
        self.memory_status = format!("Unloaded level {level} from {unloaded} ROI(s)");
        self.touch_memory_projection();
        Ok(json!({"unloaded_items":unloaded,"level":level}))
    }

    pub(super) fn unpin_all_memory(&mut self) -> Result<Value, ControlError> {
        self.require_resource()?;
        self.memory_operation_generation = self.memory_operation_generation.wrapping_add(1).max(1);
        self.memory_pending.clear();
        let unloaded = self.pinned_levels.len();
        self.pinned_levels.clear();
        self.memory_status = format!("Unloaded {unloaded} pinned mosaic level(s) from RAM");
        self.touch_memory_projection();
        Ok(json!({"unloaded_item_levels":unloaded}))
    }

    pub(super) fn pinned_bytes(&self) -> u64 {
        self.pinned_levels
            .values()
            .filter_map(|state| match state {
                MosaicPinnedLevelState::Loaded(resource) => Some(resource.bytes()),
                MosaicPinnedLevelState::Failed(_) => None,
            })
            .sum()
    }
}
