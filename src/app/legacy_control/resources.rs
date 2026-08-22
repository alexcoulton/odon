use super::super::*;

impl OmeZarrViewerApp {
    pub fn control_tile_loading_json(&self) -> serde_json::Value {
        serde_json::json!({
            "workers": self.tile_loader_threads,
            "runtime_tuning_supported": self.supports_runtime_tile_loader_tuning(),
            "prefetch_mode": match self.tile_prefetch_mode {
                TilePrefetchMode::Off => "off",
                TilePrefetchMode::TargetHalo => "target_halo",
                TilePrefetchMode::TargetAndFinerHalo => "target_and_finer_halo",
            },
            "prefetch_aggressiveness": match self.tile_prefetch_aggressiveness {
                TilePrefetchAggressiveness::Conservative => "conservative",
                TilePrefetchAggressiveness::Balanced => "balanced",
                TilePrefetchAggressiveness::Aggressive => "aggressive",
            },
            "prefer_pinned_finer_levels": self.prefer_pinned_finer_levels,
            "status": self.tile_loading_status,
            "cache": {"loaded": self.cache.len(), "capacity": self.cache.capacity(), "in_flight": self.cache.in_flight_len()},
            "target_level": self.last_target_level,
        })
    }

    pub fn control_labels_json(&self) -> serde_json::Value {
        let mut available = self.seg_label_names.clone();
        if self.dataset.is_root_label_mask() {
            let name = LabelZarrDataset::root_label_name(&self.dataset);
            if !available.contains(&name) {
                available.push(name);
            }
        }
        serde_json::json!({
            "available": available,
            "selected": self.seg_label_selected,
            "loaded": self.label_cells.as_ref().map(|labels| labels.label_name.clone()),
            "visible": self.cells_outlines_visible,
            "busy": self.labels_gl.as_ref().is_some_and(|labels| labels.is_busy()),
            "gpu_available": self.tiles_gl.is_some(),
            "status": self.seg_label_status,
            "offset_world": [self.seg_labels_offset_world.x, self.seg_labels_offset_world.y],
            "generation": self.control_actor_label_generation.max(1),
            "actor_owned": self.control_actor_label_generation > 0,
        })
    }

    pub fn control_load_labels(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let name = params
            .get("name")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .unwrap_or(self.seg_label_selected.as_str())
            .to_string();
        if name.is_empty() {
            return serde_json::json!({"error": "label name is required because this dataset has no default label group"});
        }
        match self.load_segmentation_labels(&name) {
            Ok(()) => {
                self.cells_outlines_visible = true;
                self.seg_label_status = format!("Loaded labels/{name}.");
                self.control_labels_json()
            }
            Err(error) => {
                serde_json::json!({"error": format!("load labels/{name} failed: {error}")})
            }
        }
    }

    pub fn control_unload_labels(&mut self) -> serde_json::Value {
        let unloaded = self.label_cells.take().map(|labels| labels.label_name);
        self.label_loader = None;
        self.label_cells_xform = None;
        self.cells_outlines_visible = false;
        if let Some(labels) = self.labels_gl.as_ref() {
            labels.reset();
        }
        self.seg_label_status = "Unloaded segmentation labels.".to_string();
        serde_json::json!({"unloaded": unloaded, "labels": self.control_labels_json()})
    }

    pub fn control_set_labels_visibility(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let Some(visible) = params.get("visible").and_then(serde_json::Value::as_bool) else {
            return serde_json::json!({"error": "visible must be a boolean"});
        };
        if visible && self.label_cells.is_none() {
            let loaded = self.control_load_labels(params);
            if loaded.get("error").is_some() {
                return loaded;
            }
        }
        self.cells_outlines_visible = visible;
        self.control_labels_json()
    }

    pub fn control_set_tile_loading_json(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let workers = match params.get("workers") {
            Some(value) => {
                let Some(value) = value
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .filter(|value| (1..=12).contains(value))
                else {
                    return serde_json::json!({"error": "workers must be an integer from 1 to 12"});
                };
                Some(value)
            }
            None => None,
        };
        let prefetch_mode = match params
            .get("prefetch_mode")
            .and_then(serde_json::Value::as_str)
        {
            Some("off") => Some(TilePrefetchMode::Off),
            Some("target_halo") => Some(TilePrefetchMode::TargetHalo),
            Some("target_and_finer_halo") => Some(TilePrefetchMode::TargetAndFinerHalo),
            Some(_) => return serde_json::json!({"error": "unknown prefetch_mode"}),
            None => None,
        };
        let aggressiveness = match params
            .get("prefetch_aggressiveness")
            .and_then(serde_json::Value::as_str)
        {
            Some("conservative") => Some(TilePrefetchAggressiveness::Conservative),
            Some("balanced") => Some(TilePrefetchAggressiveness::Balanced),
            Some("aggressive") => Some(TilePrefetchAggressiveness::Aggressive),
            Some(_) => return serde_json::json!({"error": "unknown prefetch_aggressiveness"}),
            None => None,
        };
        if let Some(workers) = workers
            && workers != self.tile_loader_threads
        {
            if !self.supports_runtime_tile_loader_tuning() {
                return serde_json::json!({"error": "runtime tile-loader tuning is unavailable for this dataset backend"});
            }
            self.tile_loader_threads = workers;
            if let Err(error) = self.respawn_tile_loaders() {
                return serde_json::json!({"error": format!("tile loader reconfigure failed: {error}")});
            }
            self.tile_loading_status = format!("Respawned tile loaders with {workers} worker(s).");
        }
        if let Some(mode) = prefetch_mode {
            self.tile_prefetch_mode = mode;
        }
        if let Some(value) = aggressiveness {
            self.tile_prefetch_aggressiveness = value;
        }
        if let Some(value) = params
            .get("prefer_pinned_finer_levels")
            .and_then(serde_json::Value::as_bool)
        {
            self.prefer_pinned_finer_levels = value;
        }
        self.control_tile_loading_json()
    }

    pub fn control_memory_json(&mut self) -> serde_json::Value {
        self.system_memory = crate::app_support::memory::read_system_memory_snapshot();
        let selected_channels = self.selected_memory_channel_indices();
        let levels = self
            .dataset
            .levels
            .iter()
            .enumerate()
            .map(|(level_index, level)| {
                let estimate = self.estimate_level_ram_bytes_for_selected_channels(
                    level_index,
                    &selected_channels,
                );
                let (status, bytes, channels_loaded, error) =
                    match self.pinned_levels.status(level_index) {
                        PinnedLevelStatus::Unloaded => ("unloaded", None, None, None),
                        PinnedLevelStatus::Loading => ("loading", None, None, None),
                        PinnedLevelStatus::Loaded {
                            bytes,
                            channels_loaded,
                        } => ("loaded", Some(bytes), Some(channels_loaded), None),
                        PinnedLevelStatus::Failed(error) => ("failed", None, None, Some(error)),
                    };
                serde_json::json!({
                    "level": level_index,
                    "downsample": level.downsample,
                    "shape_y": level.shape.get(self.dataset.dims.y),
                    "shape_x": level.shape.get(self.dataset.dims.x),
                    "selected_channel_estimate_bytes": estimate,
                    "status": status,
                    "loaded_bytes": bytes,
                    "channels_loaded": channels_loaded,
                    "error": error,
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "running": self.pinned_levels.has_loading(),
            "status": self.memory_status,
            "pinned_bytes": self.pinned_levels.total_loaded_bytes(),
            "system": self.system_memory.as_ref().map(|memory| serde_json::json!({"total_bytes": memory.total_bytes, "available_bytes": memory.available_bytes})),
            "selected_channels": selected_channels,
            "z_stack_supported": !self.z_extent_level0().is_some_and(|extent| extent > 1),
            "levels": levels,
        })
    }

    pub fn control_pin_memory_level(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(level) = params
            .get("level")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|level| *level < self.dataset.levels.len())
        else {
            return serde_json::json!({"error": "memory level is required and must be in range"});
        };
        let selected = match params.get("channels") {
            Some(value) => {
                let Some(values) = value.as_array() else {
                    return serde_json::json!({"error": "channels must be an array"});
                };
                let mut selected = Vec::new();
                for value in values {
                    match self.control_channel_index_from_value(value) {
                        Ok(index) if !selected.contains(&index) => selected.push(index),
                        Ok(_) => {}
                        Err(error) => return serde_json::json!({"error": error}),
                    }
                }
                selected
            }
            None => self.selected_memory_channel_indices(),
        };
        if selected.is_empty() {
            return serde_json::json!({"error": "select at least one channel to pin"});
        }
        if self.z_extent_level0().is_some_and(|extent| extent > 1) {
            return serde_json::json!({"error": "RAM pinning is currently unavailable for OME-Zarr z-stacks"});
        }
        let estimate = self.estimate_level_ram_bytes_for_selected_channels(level, &selected);
        let force = params
            .get("force")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        if let Some(risk) = self.memory_risk(estimate)
            && !force
        {
            return serde_json::json!({
                "confirmation_required": true,
                "level": level,
                "requested_bytes": risk.requested_bytes,
                "projected_bytes": risk.projected_bytes,
                "available_bytes": risk.available_bytes,
                "risk": match risk.level { crate::app_support::memory::MemoryRiskLevel::Warning => "warning", crate::app_support::memory::MemoryRiskLevel::Danger => "danger" },
            });
        }
        let channel_ids = selected
            .iter()
            .filter_map(|index| {
                self.channels
                    .get(*index)
                    .map(|channel| channel.index as u64)
            })
            .collect::<Vec<_>>();
        self.memory_selected_channels = selected.into_iter().collect();
        self.execute_memory_load(
            format!(
                "Loading {} channel(s) from level {level} into RAM",
                channel_ids.len()
            ),
            vec![PendingPinnedLevelLoadRequest {
                level,
                selected_channels: channel_ids,
            }],
        );
        serde_json::json!({"started": true, "level": level, "estimated_bytes": estimate})
    }

    pub fn control_unpin_memory_level(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let Some(level) = params
            .get("level")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|level| *level < self.dataset.levels.len())
        else {
            return serde_json::json!({"error": "memory level is required and must be in range"});
        };
        let was_loaded = !matches!(
            self.pinned_levels.status(level),
            PinnedLevelStatus::Unloaded
        );
        self.pinned_levels.unload(level);
        self.memory_status = format!("Unloaded pinned level {level} from RAM.");
        serde_json::json!({"unloaded": was_loaded, "level": level})
    }

    pub fn control_unpin_all_memory(&mut self) -> serde_json::Value {
        let mut count = 0;
        for level in 0..self.dataset.levels.len() {
            if !matches!(
                self.pinned_levels.status(level),
                PinnedLevelStatus::Unloaded
            ) {
                count += 1;
                self.pinned_levels.unload(level);
            }
        }
        self.memory_status = format!("Unloaded {count} pinned level(s) from RAM.");
        serde_json::json!({"unloaded_levels": count})
    }
}
