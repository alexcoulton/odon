use super::*;

impl OmeZarrViewerApp {
    pub(super) fn selected_memory_channel_indices(&self) -> Vec<usize> {
        self.channel_layer_order
            .iter()
            .copied()
            .filter(|idx| self.memory_selected_channels.contains(idx))
            .collect()
    }

    pub(super) fn selected_memory_channel_ids(&self) -> Vec<u64> {
        self.selected_memory_channel_indices()
            .into_iter()
            .filter_map(|idx| self.channels.get(idx).map(|channel| channel.index as u64))
            .collect()
    }

    pub(super) fn memory_channel_rows(&self) -> Vec<MemoryChannelRow> {
        self.channel_layer_order
            .iter()
            .filter_map(|&idx| {
                self.channels.get(idx).map(|channel| MemoryChannelRow {
                    id: idx,
                    label: if channel.visible {
                        format!("{} (visible)", channel.name)
                    } else {
                        channel.name.clone()
                    },
                    visible: channel.visible,
                })
            })
            .collect()
    }

    pub(super) fn estimate_level_ram_bytes_for_selected_channels(
        &self,
        level: usize,
        selected: &[usize],
    ) -> u64 {
        let Some(info) = self.dataset.levels.get(level) else {
            return 0;
        };
        if selected.is_empty() {
            return 0;
        }
        let Some(&shape_y) = info.shape.get(self.dataset.dims.y) else {
            return 0;
        };
        let Some(&shape_x) = info.shape.get(self.dataset.dims.x) else {
            return 0;
        };
        let channel_count = if self.dataset.dims.c.is_some() {
            selected.len() as u64
        } else {
            1
        };
        let bytes_per_sample = match info.dtype.as_str() {
            "|u1" | "|i1" => 1u64,
            "<u2" | ">u2" | "<i2" | ">i2" => 2u64,
            "<f4" | ">f4" | "<u4" | ">u4" | "<i4" | ">i4" => 4u64,
            _ => 2u64,
        };
        channel_count
            .checked_mul(shape_y)
            .and_then(|v| v.checked_mul(shape_x))
            .and_then(|v| v.checked_mul(bytes_per_sample))
            .unwrap_or(0)
    }

    pub(super) fn memory_risk(
        &self,
        requested_bytes: u64,
    ) -> Option<crate::app_support::memory::MemoryRisk> {
        memory_risk(
            self.system_memory.as_ref(),
            self.pinned_levels.total_loaded_bytes(),
            requested_bytes,
        )
    }

    pub(super) fn start_memory_load(
        &mut self,
        summary: String,
        requests: Vec<PendingPinnedLevelLoadRequest>,
        requested_bytes: u64,
    ) {
        if requests.is_empty() {
            self.memory_status = "No eligible channels selected for RAM pinning.".to_string();
            return;
        }
        if let Some(risk) = self.memory_risk(requested_bytes) {
            self.pending_memory_load = Some(PendingMemoryAction {
                summary,
                payload: requests,
                risk,
            });
        } else {
            self.execute_memory_load(summary, requests);
        }
    }

    pub(super) fn execute_memory_load(
        &mut self,
        summary: String,
        requests: Vec<PendingPinnedLevelLoadRequest>,
    ) {
        if requests.is_empty() {
            self.memory_status = "No eligible channels selected for RAM pinning.".to_string();
            return;
        }
        if self.z_extent_level0().is_some_and(|extent| extent > 1) {
            self.memory_status =
                "RAM pinning is currently unavailable for OME-Zarr z-stacks.".to_string();
            return;
        }
        for request in requests {
            if self.control_actor_tile_policy_generation > 0 {
                self.native_control_intents.push(NativeControlIntent {
                    method: "memory.pin",
                    params: serde_json::json!({
                        "level":request.level,
                        "channels":request.selected_channels,
                        "force":true,
                    }),
                });
            } else {
                self.pinned_levels.request_load(
                    self.store.clone(),
                    self.dataset.dims.clone(),
                    self.dataset.levels.clone(),
                    request.level,
                    request.selected_channels,
                );
            }
        }
        self.memory_status = summary;
    }

    pub(super) fn ui_memory_load_dialog(&mut self, ctx: &egui::Context) {
        if let Some((summary, requests)) =
            ui_pending_memory_action_dialog(ctx, &mut self.pending_memory_load)
        {
            self.execute_memory_load(summary, requests);
        }
    }

    pub(super) fn ui_memory(&mut self, ui: &mut egui::Ui) {
        let z_stacks_unsupported = self.z_extent_level0().is_some_and(|extent| extent > 1);
        ui_memory_overview(
            ui,
            "Manually pin selected OME-Zarr channels and levels in CPU RAM for the current image. Pinned levels feed the existing tile renderer instead of replacing it.",
            Some(("Pinned total", self.pinned_levels.total_loaded_bytes())),
            self.system_memory.as_ref(),
        );
        ui.add_space(6.0);

        let mut tile_loader_threads = self.tile_loader_threads;
        let mut tile_prefetch_mode = self.tile_prefetch_mode;
        let mut tile_prefetch_aggressiveness = self.tile_prefetch_aggressiveness;
        let mut prefer_pinned_finer_levels = self.prefer_pinned_finer_levels;
        ui.collapsing("Tile Loading", |ui| {
            if self.supports_runtime_tile_loader_tuning() {
                let mut threads = tile_loader_threads as u32;
                ui.horizontal(|ui| {
                    ui.label("Workers");
                    let changed = ui
                        .add(
                            egui::DragValue::new(&mut threads)
                                .range(1..=12)
                                .speed(0.2),
                        )
                        .changed();
                    if ui.button("Auto").clicked() {
                        threads = Self::default_tile_loader_threads() as u32;
                    }
                    if changed || threads as usize != tile_loader_threads {
                        tile_loader_threads = threads.max(1) as usize;
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Prefetch");
                    egui::ComboBox::from_id_salt("tile-prefetch-mode")
                        .selected_text(match tile_prefetch_mode {
                            TilePrefetchMode::Off => "Off",
                            TilePrefetchMode::TargetHalo => "Target halo",
                            TilePrefetchMode::TargetAndFinerHalo => "Target + finer halo",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut tile_prefetch_mode,
                                TilePrefetchMode::Off,
                                "Off",
                            );
                            ui.selectable_value(
                                &mut tile_prefetch_mode,
                                TilePrefetchMode::TargetHalo,
                                "Target halo",
                            );
                            ui.selectable_value(
                                &mut tile_prefetch_mode,
                                TilePrefetchMode::TargetAndFinerHalo,
                                "Target + finer halo",
                            );
                        });
                });
                ui.horizontal(|ui| {
                    ui.label("Aggressiveness");
                    egui::ComboBox::from_id_salt("tile-prefetch-aggressiveness")
                        .selected_text(match tile_prefetch_aggressiveness {
                            TilePrefetchAggressiveness::Conservative => "Conservative",
                            TilePrefetchAggressiveness::Balanced => "Balanced",
                            TilePrefetchAggressiveness::Aggressive => "Aggressive",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut tile_prefetch_aggressiveness,
                                TilePrefetchAggressiveness::Conservative,
                                "Conservative",
                            );
                            ui.selectable_value(
                                &mut tile_prefetch_aggressiveness,
                                TilePrefetchAggressiveness::Balanced,
                                "Balanced",
                            );
                            ui.selectable_value(
                                &mut tile_prefetch_aggressiveness,
                                TilePrefetchAggressiveness::Aggressive,
                                "Aggressive",
                            );
                        });
                });
                ui.checkbox(
                    &mut prefer_pinned_finer_levels,
                    "Use pinned finer levels for missing coarser levels",
                )
                .on_hover_text(
                    "If the current zoom level is not pinned, render it from a finer pinned level before falling back to disk or network reads.",
                );
                ui.label(
                    "Target halo prefetches nearby tiles at the current level. Target + finer halo also warms the next finer level to reduce zoom-in stalls.",
                );
            } else {
                ui.label("Runtime tile loading controls are unavailable for this dataset backend.");
            }
            if !self.tile_loading_status.is_empty() {
                ui.label(self.tile_loading_status.clone());
            }
            ui.separator();
        });
        if (
            tile_loader_threads,
            tile_prefetch_mode,
            tile_prefetch_aggressiveness,
            prefer_pinned_finer_levels,
        ) != (
            self.tile_loader_threads,
            self.tile_prefetch_mode,
            self.tile_prefetch_aggressiveness,
            self.prefer_pinned_finer_levels,
        ) {
            self.native_control_intents.push(NativeControlIntent {
                method: "memory.tiles.set",
                params: serde_json::json!({
                    "workers":tile_loader_threads,
                    "prefetch_mode":match tile_prefetch_mode {
                        TilePrefetchMode::Off => "off",
                        TilePrefetchMode::TargetHalo => "target_halo",
                        TilePrefetchMode::TargetAndFinerHalo => "target_and_finer_halo",
                    },
                    "prefetch_aggressiveness":match tile_prefetch_aggressiveness {
                        TilePrefetchAggressiveness::Conservative => "conservative",
                        TilePrefetchAggressiveness::Balanced => "balanced",
                        TilePrefetchAggressiveness::Aggressive => "aggressive",
                    },
                    "prefer_pinned_finer_levels":prefer_pinned_finer_levels,
                }),
            });
        }

        let rows = self.memory_channel_rows();
        ui_memory_channel_selector(
            ui,
            "viewer-memory-channel-list",
            &rows,
            &mut self.memory_selected_channels,
        );
        ui.separator();

        ui.label(format!(
            "Texture tile cache: {} / {} tiles, {} in flight",
            self.cache.len(),
            self.cache.capacity(),
            self.cache.in_flight_len()
        ));
        if let Some(level) = self.last_target_level {
            ui.label(format!("Current draw level: {level}"));
        }
        ui.label("Loading is manual. The app estimates RAM usage but does not enforce a system-memory limit.");
        if !self.memory_status.is_empty() {
            ui.label(self.memory_status.clone());
        }
        if z_stacks_unsupported {
            ui.colored_label(
                ui.visuals().warn_fg_color,
                "RAM pinning is disabled for OME-Zarr z-stacks in this viewer build.",
            );
        }
        ui.separator();

        let selected_channels = self.selected_memory_channel_indices();
        let selected_channel_ids = self.selected_memory_channel_ids();
        egui::Grid::new("viewer-memory-grid")
            .num_columns(5)
            .striped(true)
            .show(ui, |ui| {
                ui.strong("Level");
                ui.strong("Shape");
                ui.strong("RAM");
                ui.strong("Status");
                ui.strong("Action");
                ui.end_row();

                for level_idx in 0..self.dataset.levels.len() {
                    let (shape_y, shape_x) = self
                        .dataset
                        .levels
                        .get(level_idx)
                        .map(|level| {
                            (
                                level.shape.get(self.dataset.dims.y).copied().unwrap_or(0),
                                level.shape.get(self.dataset.dims.x).copied().unwrap_or(0),
                            )
                        })
                        .unwrap_or((0, 0));
                    let selected_count = if self.dataset.dims.c.is_some() {
                        selected_channels.len()
                    } else if selected_channels.is_empty() {
                        0
                    } else {
                        1
                    };
                    let estimate = self.estimate_level_ram_bytes_for_selected_channels(
                        level_idx,
                        &selected_channels,
                    );

                    ui.label(level_idx.to_string());
                    ui.label(format!("{selected_count} x {shape_y} x {shape_x}"));
                    ui.label(if estimate == 0 {
                        "No selected channels".to_string()
                    } else {
                        format_bytes(estimate)
                    });
                    match self.pinned_levels.status(level_idx) {
                        PinnedLevelStatus::Unloaded => {
                            if self.last_target_level == Some(level_idx) {
                                ui.label("Streaming (current)");
                            } else {
                                ui.label("Streaming");
                            }
                        }
                        PinnedLevelStatus::Loading => {
                            ui.label("Loading");
                        }
                        PinnedLevelStatus::Loaded {
                            bytes,
                            channels_loaded,
                        } => {
                            ui.label(format!(
                                "Pinned ({}; {} ch)",
                                format_bytes(bytes),
                                channels_loaded
                            ));
                        }
                        PinnedLevelStatus::Failed(err) => {
                            ui.colored_label(ui.visuals().warn_fg_color, format!("Failed: {err}"));
                        }
                    }
                    ui.horizontal(|ui| {
                        let risk = self.memory_risk(estimate);
                        let load_label = match risk.as_ref().map(|risk| risk.level) {
                            Some(crate::app_support::memory::MemoryRiskLevel::Danger) => {
                                "Load danger"
                            }
                            Some(crate::app_support::memory::MemoryRiskLevel::Warning) => {
                                "Load warning"
                            }
                            None => "Load",
                        };
                        if ui
                            .add_enabled(
                                estimate > 0 && !z_stacks_unsupported,
                                egui::Button::new(load_label),
                            )
                            .clicked()
                        {
                            self.start_memory_load(
                                format!(
                                    "Loading {} channel(s) from level {level_idx} into RAM",
                                    selected_channel_ids.len()
                                ),
                                vec![PendingPinnedLevelLoadRequest {
                                    level: level_idx,
                                    selected_channels: selected_channel_ids.clone(),
                                }],
                                estimate,
                            );
                        }
                        let can_unload = !matches!(
                            self.pinned_levels.status(level_idx),
                            PinnedLevelStatus::Unloaded
                        );
                        if ui
                            .add_enabled(can_unload, egui::Button::new("Unload"))
                            .clicked()
                        {
                            if self.control_actor_tile_policy_generation > 0 {
                                self.native_control_intents.push(NativeControlIntent {
                                    method: "memory.unpin",
                                    params: serde_json::json!({"level":level_idx}),
                                });
                            }
                            self.pinned_levels.unload(level_idx);
                            self.memory_status =
                                format!("Unloaded pinned level {level_idx} from RAM.");
                        }
                    });
                    ui.end_row();
                }
            });
    }
}
