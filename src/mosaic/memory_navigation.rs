//! Mosaic memory operations, refinement ordering, and focus/navigation helpers.

use super::*;

impl MosaicViewerApp {
    pub(super) fn focused_item(&self) -> Option<&MosaicItem> {
        let id = self.focused_core_id?;
        self.items.iter().find(|it| it.id == id)
    }

    pub(super) fn selected_memory_global_channels(&self) -> Vec<u64> {
        self.channel_layer_order
            .iter()
            .copied()
            .filter(|gid| self.memory_selected_channels.contains(gid))
            .map(|gid| gid as u64)
            .collect()
    }

    pub(super) fn refresh_system_memory_if_needed(&mut self) {
        refresh_system_memory_if_needed(
            &mut self.system_memory,
            &mut self.system_memory_last_refresh,
            Duration::from_secs(2),
        );
    }

    pub(super) fn memory_risk(&self, requested_bytes: u64) -> Option<MemoryRisk> {
        memory_risk(
            self.system_memory.as_ref(),
            self.pinned_levels.total_loaded_bytes(),
            requested_bytes,
        )
    }

    pub(super) fn start_memory_load(
        &mut self,
        summary: String,
        requests: Vec<PendingMemoryLoadRequest>,
        requested_bytes: u64,
    ) {
        if requests.is_empty() {
            self.status = "No eligible channels selected for RAM pinning.".to_string();
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
        requests: Vec<PendingMemoryLoadRequest>,
    ) {
        let count = requests.len();
        for request in requests {
            self.pinned_levels.request_load(
                request.dataset_id,
                request.source,
                request.level,
                request.selected_global_channels,
            );
        }
        self.status = if count == 0 {
            "No eligible channels selected for RAM pinning.".to_string()
        } else {
            summary
        };
    }

    pub(super) fn memory_load_requests_for_all_rois(
        &self,
        level: usize,
        selected_global_channels: &[u64],
    ) -> (Vec<PendingMemoryLoadRequest>, u64) {
        let mut requests = Vec::new();
        let mut total_bytes = 0u64;
        for item in &self.items {
            let Some(source) = self.sources.get(item.id).cloned() else {
                continue;
            };
            if source.levels.get(level).is_none() {
                continue;
            }
            let estimate = estimate_level_ram_bytes_for_channels(
                &source,
                level,
                Some(selected_global_channels),
            )
            .unwrap_or(0);
            if estimate == 0 {
                continue;
            }
            total_bytes = total_bytes.saturating_add(estimate);
            requests.push(PendingMemoryLoadRequest {
                dataset_id: item.id,
                source,
                level,
                selected_global_channels: selected_global_channels.to_vec(),
            });
        }
        (requests, total_bytes)
    }

    pub(super) fn memory_load_request_for_dataset(
        &self,
        dataset_id: usize,
        source: MosaicSource,
        level: usize,
        selected_global_channels: &[u64],
    ) -> Option<(PendingMemoryLoadRequest, u64)> {
        let requested_bytes =
            estimate_level_ram_bytes_for_channels(&source, level, Some(selected_global_channels))
                .unwrap_or(0);
        if requested_bytes == 0 {
            return None;
        }
        Some((
            PendingMemoryLoadRequest {
                dataset_id,
                source,
                level,
                selected_global_channels: selected_global_channels.to_vec(),
            },
            requested_bytes,
        ))
    }

    pub(super) fn unload_level_for_all_rois(&mut self, level: usize) -> usize {
        let mut count = 0usize;
        for item in &self.items {
            if self
                .sources
                .get(item.id)
                .and_then(|s| s.levels.get(level))
                .is_none()
            {
                continue;
            }
            self.pinned_levels.unload(item.id, level);
            count += 1;
        }
        count
    }

    pub(super) fn refine_item_order(&self, visible_world: egui::Rect) -> Vec<usize> {
        if self.items.is_empty() {
            return Vec::new();
        }

        let center_world = self.camera.center_world_lvl0;
        let center_item_id = self
            .items
            .iter()
            .find(|it| item_rect(it).contains(center_world))
            .map(|it| it.id);
        let focused_id = self.focused_core_id;

        let mut out: Vec<(u8, f32, usize)> = Vec::new();
        out.reserve(self.items.len().min(256));
        for (idx, it) in self.items.iter().enumerate() {
            let r = item_rect(it);
            if !r.intersects(visible_world) {
                continue;
            }
            let pri = if Some(it.id) == center_item_id {
                0u8
            } else if Some(it.id) == focused_id {
                1u8
            } else {
                2u8
            };
            let d = r.center() - center_world;
            let dist2 = d.x * d.x + d.y * d.y;
            out.push((pri, dist2, idx));
        }

        out.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));
        out.into_iter().map(|t| t.2).collect()
    }

    pub(super) fn step_selected_channel_visibility(&mut self, step: i32) {
        if self.channels.is_empty() || self.channel_layer_order.is_empty() {
            return;
        }
        let cur_gid = self
            .selected_channel
            .min(self.channels.len().saturating_sub(1));
        let cur_pos = self
            .channel_layer_order
            .iter()
            .position(|&g| g == cur_gid)
            .unwrap_or(0);
        let n = self.channel_layer_order.len() as i32;
        let next_pos = ((cur_pos as i32) + step).rem_euclid(n) as usize;
        let next_gid =
            self.channel_layer_order[next_pos].min(self.channels.len().saturating_sub(1));

        if self.control_actor_owned {
            self.submit_native_control_intent(
                "viewer.channels.set_visible",
                serde_json::json!({"channels":[cur_gid],"mode":"hide"}),
            );
            self.submit_native_control_intent(
                "viewer.channels.set_visible",
                serde_json::json!({"channels":[next_gid],"mode":"show"}),
            );
            self.submit_native_control_intent(
                "viewer.channels.set_active",
                serde_json::json!({"index":next_gid}),
            );
            return;
        }

        if let Some(cur) = self.channels.get_mut(cur_gid) {
            cur.visible = false;
        }
        if let Some(next) = self.channels.get_mut(next_gid) {
            next.visible = true;
        }
        self.selected_channel = next_gid;
        self.active_layer = MosaicLayerId::Channel(next_gid);
    }

    pub(super) fn drain_raw_tiles(&mut self) {
        while let Ok(msg) = self.loader.rx.try_recv() {
            match msg {
                MosaicRawTileWorkerResponse::Tile(msg) => {
                    if msg.generation == self.tile_request_generation {
                        self.tiles_gl.insert_pending(msg);
                    } else {
                        self.tiles_gl.cancel_in_flight(&msg.key);
                    }
                }
                MosaicRawTileWorkerResponse::Dropped { key, .. } => {
                    self.tiles_gl.cancel_in_flight(&key);
                }
                MosaicRawTileWorkerResponse::Failed { key, error } => {
                    self.tiles_gl.cancel_in_flight(&key);
                    crate::log_warn!("mosaic raw tile load failed for {:?}: {}", key, error);
                }
            }
        }
    }

    pub(super) fn sync_tile_request_generation(
        &mut self,
        visible_world: egui::Rect,
        viewport: egui::Rect,
        channels_draw: &[ChannelDraw],
    ) {
        let signature = TileRequestSignature {
            viewport_width_bits: viewport.width().to_bits(),
            viewport_height_bits: viewport.height().to_bits(),
            visible_world_min_x_bits: visible_world.min.x.to_bits(),
            visible_world_min_y_bits: visible_world.min.y.to_bits(),
            visible_world_max_x_bits: visible_world.max.x.to_bits(),
            visible_world_max_y_bits: visible_world.max.y.to_bits(),
            visible_channels: channels_draw.iter().map(|ch| ch.index).collect(),
        };

        if self.last_tile_request_signature.as_ref() != Some(&signature) {
            self.tile_request_generation = self.tile_request_generation.wrapping_add(1).max(1);
            self.last_tile_request_signature = Some(signature);
            self.loader
                .set_latest_generation(self.tile_request_generation);
        }
    }

    pub(super) fn fit_mosaic(&mut self) {
        if self.submit_native_control_intent("mosaic.fit_all", serde_json::json!({})) {
            return;
        }
        if let Some(viewport) = self.last_canvas_rect {
            self.camera.fit_to_world_rect(viewport, self.mosaic_bounds);
        }
    }

    pub(super) fn focused_core_summary(&self) -> Option<(usize, usize, String)> {
        let n = self.items.len();
        if n == 0 {
            return None;
        }
        let Some(id) = self.focused_core_id else {
            return None;
        };
        let idx = self.items.iter().position(|it| it.id == id)?;
        let name = self
            .items
            .get(idx)
            .map(|it| it.sample_id.clone())
            .unwrap_or_default();
        Some((idx + 1, n, name))
    }

    pub(super) fn fit_focused_core(&mut self, ctx: &egui::Context) {
        if self.items.is_empty() {
            return;
        }
        let id = self
            .focused_core_id
            .filter(|id| self.items.iter().any(|it| it.id == *id))
            .unwrap_or(self.items[0].id);
        if self.control_actor_owned {
            if self.focused_core_id == Some(id) {
                self.submit_native_control_intent("mosaic.focus.fit", serde_json::json!({}));
            } else if let Some(item) = self.items.iter().find(|item| item.id == id) {
                self.submit_native_control_intent(
                    "mosaic.focus.set",
                    serde_json::json!({"roi_id":item.sample_id,"fit":true}),
                );
            }
            return;
        }
        self.focused_core_id = Some(id);

        let Some(it) = self.items.iter().find(|it| it.id == id) else {
            return;
        };
        let world = item_rect(it);
        let viewport = self
            .last_canvas_rect
            .or_else(|| ctx.input(|i| i.viewport().inner_rect));
        if let Some(viewport) = viewport {
            self.camera.fit_to_world_rect(viewport, world);
        }
    }

    pub(super) fn step_focused_core(&mut self, ctx: &egui::Context, step: i32) {
        let n = self.items.len();
        if n == 0 {
            return;
        }
        if self.control_actor_owned {
            let method = if step >= 0 {
                "mosaic.focus.next"
            } else {
                "mosaic.focus.previous"
            };
            self.submit_native_control_intent(
                method,
                serde_json::json!({"step":step.unsigned_abs(),"wrap":true}),
            );
            return;
        }
        let cur_id = self.focused_core_id;
        let cur_idx = cur_id
            .and_then(|id| self.items.iter().position(|it| it.id == id))
            .unwrap_or(0);
        let next_idx = ((cur_idx as i32) + step).rem_euclid(n as i32) as usize;
        self.focused_core_id = Some(self.items[next_idx].id);
        self.fit_focused_core(ctx);
    }
}
