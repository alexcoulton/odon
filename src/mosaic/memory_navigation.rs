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

    pub(super) fn projected_memory_running(&self) -> bool {
        self.control_actor_memory_state["running"]
            .as_bool()
            .unwrap_or(false)
    }

    pub(super) fn projected_memory_level_status(
        &self,
        dataset_id: usize,
        level: usize,
    ) -> MosaicPinnedLevelStatus {
        let projected = self.control_actor_memory_state["items"]
            .as_array()
            .and_then(|items| {
                items
                    .iter()
                    .find(|item| item["id"].as_u64() == Some(dataset_id as u64))
            })
            .and_then(|item| item["levels"].as_array())
            .and_then(|levels| {
                levels
                    .iter()
                    .find(|entry| entry["level"].as_u64() == Some(level as u64))
            });
        match projected
            .and_then(|entry| entry["status"].as_str())
            .unwrap_or("unloaded")
        {
            "loading" => MosaicPinnedLevelStatus::Loading,
            "loaded" => MosaicPinnedLevelStatus::Loaded {
                bytes: projected
                    .and_then(|entry| entry["loaded_bytes"].as_u64())
                    .unwrap_or(0),
                channels_loaded: projected
                    .and_then(|entry| entry["channels_loaded"].as_u64())
                    .unwrap_or(0) as usize,
            },
            "failed" => MosaicPinnedLevelStatus::Failed(
                projected
                    .and_then(|entry| entry["error"].as_str())
                    .unwrap_or("memory pin failed")
                    .to_string(),
            ),
            _ => self.pinned_levels.status(dataset_id, level),
        }
    }

    pub(super) fn start_memory_load(
        &mut self,
        summary: String,
        params: serde_json::Value,
        requested_bytes: u64,
    ) {
        if let Some(risk) = self.memory_risk(requested_bytes) {
            self.pending_memory_load = Some(PendingMemoryAction {
                summary,
                payload: params,
                risk,
            });
        } else {
            self.execute_memory_load(params);
        }
    }

    pub(super) fn execute_memory_load(&mut self, mut params: serde_json::Value) {
        params["force"] = serde_json::Value::Bool(true);
        self.submit_native_control_intent("memory.pin", params);
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
    }

    pub(super) fn drain_raw_tiles(&mut self) {
        const MAX_RESPONSES_PER_FRAME: usize = 128;
        for _ in 0..MAX_RESPONSES_PER_FRAME {
            let Ok(msg) = self.loader.try_recv() else {
                break;
            };
            match msg {
                MosaicRawTileWorkerResponse::Tile(msg) => {
                    if msg.generation == self.tile_request_generation {
                        self.tiles_gl.insert_pending(msg);
                    } else {
                        self.tiles_gl.cancel_in_flight(&msg.key);
                        self.tiles_gl.record_stale_drop_before_install();
                    }
                }
                MosaicRawTileWorkerResponse::Dropped { key, .. } => {
                    self.tiles_gl.cancel_in_flight(&key);
                    self.tiles_gl.record_worker_drop();
                }
                MosaicRawTileWorkerResponse::Failed { key, error } => {
                    self.tiles_gl.cancel_in_flight(&key);
                    self.tiles_gl.record_failed_load();
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
        self.submit_native_control_intent("mosaic.fit_all", serde_json::json!({}));
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

    pub(super) fn step_focused_core(&mut self, _ctx: &egui::Context, step: i32) {
        if self.items.is_empty() {
            return;
        }
        let method = if step >= 0 {
            "mosaic.focus.next"
        } else {
            "mosaic.focus.previous"
        };
        self.submit_native_control_intent(
            method,
            serde_json::json!({"step":step.unsigned_abs(),"wrap":true}),
        );
    }
}
