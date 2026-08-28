use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui;
use glow::HasContext;
use lru::LruCache;
use parking_lot::Mutex;

use super::io::{MosaicRawTileKey, MosaicRawTileResponse, MosaicSource};
use odon::settings::{ImageTileCacheMode, ImageTileCacheSettings, ImageTileChannelHistory};

const CACHE_METADATA_ENTRY_LIMIT: usize = 32_768;
const DELETE_TEXTURES_PER_FRAME: usize = 256;
const UPLOAD_TEXTURES_PER_FRAME: usize = 128;
const POLICY_REFRESH_INTERVAL: Duration = Duration::from_secs(2);
const LOW_WATERMARK_PERCENT: u64 = 82;

#[derive(Debug, Clone, Default)]
pub struct MosaicTileCacheStats {
    pub entries: usize,
    pub in_flight: usize,
    pub pending_cpu_bytes: u64,
    pub uploaded_texture_bytes: u64,
    pub queued_deletion_bytes: u64,
    pub total_tracked_bytes: u64,
    pub peak_tracked_bytes: u64,
    pub protected_visible_bytes: u64,
    pub over_budget_bytes: u64,
    pub effective_budget_bytes: u64,
    pub low_watermark_bytes: u64,
    pub hits: u64,
    pub misses: u64,
    pub inserts: u64,
    pub evictions_byte_budget: u64,
    pub evictions_channel_change: u64,
    pub evictions_metadata_limit: u64,
    pub stale_drops_before_install: u64,
    pub worker_drops: u64,
    pub failed_loads: u64,
    pub gl_deletions: u64,
    pub current_channel_group: Vec<u64>,
    pub previous_channel_group: Vec<u64>,
    pub pressure_state: &'static str,
    pub resolution_reason: &'static str,
    pub realized_generation: u64,
}

impl MosaicTileCacheStats {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "loaded": self.entries,
            "capacity": CACHE_METADATA_ENTRY_LIMIT,
            "in_flight": self.in_flight,
            "pending_cpu_bytes": self.pending_cpu_bytes,
            "uploaded_texture_bytes": self.uploaded_texture_bytes,
            "queued_deletion_bytes": self.queued_deletion_bytes,
            "total_tracked_bytes": self.total_tracked_bytes,
            "peak_tracked_bytes": self.peak_tracked_bytes,
            "protected_visible_working_set_bytes": self.protected_visible_bytes,
            "over_budget_bytes": self.over_budget_bytes,
            "effective_budget_bytes": self.effective_budget_bytes,
            "low_watermark_bytes": self.low_watermark_bytes,
            "pressure_state": self.pressure_state,
            "resolution_reason": self.resolution_reason,
            "hits": self.hits,
            "misses": self.misses,
            "inserts": self.inserts,
            "evictions": {
                "byte_budget": self.evictions_byte_budget,
                "channel_change": self.evictions_channel_change,
                "metadata_limit": self.evictions_metadata_limit,
            },
            "stale_drops": {
                "before_install": self.stale_drops_before_install,
                "worker": self.worker_drops,
            },
            "failed_loads": self.failed_loads,
            "gl_deletions": self.gl_deletions,
            "current_channel_group": self.current_channel_group,
            "previous_channel_group": self.previous_channel_group,
            "realized_generation": self.realized_generation,
            "over_budget_reason": if self.over_budget_bytes > 0 {
                "protected_visible_working_set"
            } else {
                ""
            },
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextureFilter {
    Linear,
    Nearest,
}

impl TextureFilter {
    fn as_gl(self) -> i32 {
        match self {
            Self::Linear => glow::LINEAR as i32,
            Self::Nearest => glow::NEAREST as i32,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MosaicTileDraw {
    pub dataset_id: usize,
    pub level: usize,
    pub tile_y: u64,
    pub tile_x: u64,
    pub screen_rect: egui::Rect,
}

#[derive(Debug, Clone, Copy)]
pub struct ChannelDraw {
    pub index: u64,
    pub color_rgb: [f32; 3],
    pub window: (f32, f32),
}

#[derive(Clone)]
pub struct MosaicTilesGl {
    inner: Arc<Mutex<Inner>>,
}

impl std::fmt::Debug for MosaicTilesGl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MosaicTilesGl").finish_non_exhaustive()
    }
}

impl MosaicTilesGl {
    pub fn new(settings: ImageTileCacheSettings) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner::new(settings))),
        }
    }

    pub fn set_policy(&self, settings: ImageTileCacheSettings, generation: u64) {
        self.inner.lock().set_policy(settings, generation);
    }

    pub fn refresh_automatic_policy(&self) {
        let should_refresh = {
            let inner = self.inner.lock();
            inner.settings.mode == ImageTileCacheMode::Automatic
                && inner
                    .last_policy_refresh
                    .is_none_or(|last| last.elapsed() >= POLICY_REFRESH_INTERVAL)
        };
        if !should_refresh {
            return;
        }
        use sysinfo::System;
        let mut system = System::new();
        system.refresh_memory();
        let total = system.total_memory();
        let available = system.available_memory();
        self.inner.lock().resolve_policy(
            (total > 0).then_some(total),
            (available > 0).then_some(available),
        );
    }

    pub fn update_working_set(
        &self,
        protected: HashSet<MosaicRawTileKey>,
        current_channel_group: Vec<u64>,
    ) {
        self.inner
            .lock()
            .update_working_set(protected, current_channel_group);
    }

    pub fn stats(&self) -> MosaicTileCacheStats {
        self.inner.lock().stats()
    }

    pub fn has_queued_deletions(&self) -> bool {
        !self.inner.lock().textures_to_delete.is_empty()
    }

    pub fn record_stale_drop_before_install(&self) {
        let mut inner = self.inner.lock();
        inner.stale_drops_before_install = inner.stale_drops_before_install.saturating_add(1);
    }

    pub fn record_worker_drop(&self) {
        let mut inner = self.inner.lock();
        inner.worker_drops = inner.worker_drops.saturating_add(1);
    }

    pub fn record_failed_load(&self) {
        let mut inner = self.inner.lock();
        inner.failed_loads = inner.failed_loads.saturating_add(1);
    }

    pub fn set_smooth_pixels(&self, smooth: bool) {
        let mut inner = self.inner.lock();
        inner.desired_filter = if smooth {
            TextureFilter::Linear
        } else {
            TextureFilter::Nearest
        };
    }

    pub fn mark_in_flight(&self, key: MosaicRawTileKey) -> bool {
        self.inner.lock().mark_in_flight(key)
    }

    pub fn contains(&self, key: &MosaicRawTileKey) -> bool {
        self.inner.lock().cache.contains(key)
    }

    pub fn cancel_in_flight(&self, key: &MosaicRawTileKey) {
        self.inner.lock().in_flight.remove(key);
    }

    pub fn insert_pending(&self, resp: MosaicRawTileResponse) {
        self.inner.lock().insert_pending(resp);
    }

    pub fn prune_in_flight(&self, keep: &HashSet<MosaicRawTileKey>) {
        let mut inner = self.inner.lock();
        inner.in_flight.retain(|k| keep.contains(k));
    }

    pub fn is_busy(&self) -> bool {
        !self.inner.lock().in_flight.is_empty()
    }

    pub fn loading_tile_count_for(&self, keep: &HashSet<MosaicRawTileKey>) -> usize {
        self.inner.lock().loading_count_for(keep)
    }

    pub fn in_flight_len(&self) -> usize {
        self.inner.lock().in_flight.len()
    }

    pub fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        painter: &egui_glow::Painter,
        sources: &[MosaicSource],
        tiles: &[MosaicTileDraw],
        channels: &[ChannelDraw],
    ) {
        let gl = painter.gl();
        let mut inner = self.inner.lock();
        inner.ensure_gl(gl);
        inner.delete_queued_textures(gl);
        inner.uploads_remaining = UPLOAD_TEXTURES_PER_FRAME;
        if tiles.is_empty() || channels.is_empty() {
            return;
        }

        let Some(bindings) = inner.bindings() else {
            return;
        };

        let viewport = info.viewport;
        let w = viewport.width().max(1.0);
        let h = viewport.height().max(1.0);
        let ppp = info.pixels_per_point.max(1e-6);

        unsafe {
            let gl = gl.as_ref();
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::CULL_FACE);
            gl.use_program(Some(bindings.program));
            gl.bind_vertex_array(Some(bindings.vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(bindings.vbo));
            gl.active_texture(glow::TEXTURE0);
            gl.uniform_1_i32(bindings.u_tex.as_ref(), 0);
        }

        // Only draw a tile at this level when all (dataset-available) visible channels have the
        // tile, so that we don't replace a coarse composite with a partial high-res composite.
        //
        // In mosaic mode, some ROIs may not contain every globally-visible channel. Those missing
        // channels should not blank the ROI, so we skip them per-dataset.
        let mut complete_tiles: Vec<(MosaicTileDraw, Vec<(ChannelDraw, glow::Texture)>)> =
            Vec::new();
        complete_tiles.reserve(tiles.len().min(1024));
        for td in tiles {
            if !td.screen_rect.intersects(viewport) {
                continue;
            }
            let Some(src) = sources.get(td.dataset_id) else {
                continue;
            };
            let mut texs: Vec<(ChannelDraw, glow::Texture)> = Vec::with_capacity(channels.len());
            let mut all_present = true;
            for ch in channels.iter() {
                let gid = ch.index as usize;
                if src.channel_map.get(gid).copied().flatten().is_none() {
                    continue;
                }
                let key = MosaicRawTileKey {
                    dataset_id: td.dataset_id,
                    level: td.level,
                    tile_y: td.tile_y,
                    tile_x: td.tile_x,
                    channel: ch.index,
                };
                if let Some(tex) = inner.ensure_uploaded(gl, &key) {
                    texs.push((*ch, tex));
                } else {
                    all_present = false;
                    break;
                }
            }
            if all_present && !texs.is_empty() {
                complete_tiles.push((*td, texs));
            }
        }

        for (td, texs) in complete_tiles {
            let verts = tile_vertices_ndc(td.screen_rect, viewport, w, h, ppp);
            unsafe {
                let gl = gl.as_ref();
                gl.buffer_data_u8_slice(
                    glow::ARRAY_BUFFER,
                    bytemuck::cast_slice(&verts),
                    glow::STREAM_DRAW,
                );
            }

            let (base, base_tex) = texs[0];
            unsafe {
                let gl = gl.as_ref();
                gl.disable(glow::BLEND);
                set_channel_uniforms(gl, &bindings, base.window, base.color_rgb);
                gl.bind_texture(glow::TEXTURE_2D, Some(base_tex));
                gl.draw_arrays(glow::TRIANGLES, 0, 6);
            }

            if texs.len() > 1 {
                unsafe {
                    let gl = gl.as_ref();
                    gl.enable(glow::BLEND);
                    gl.blend_func(glow::ONE, glow::ONE);
                }
                for (ch, tex) in texs.into_iter().skip(1) {
                    unsafe {
                        let gl = gl.as_ref();
                        set_channel_uniforms(gl, &bindings, ch.window, ch.color_rgb);
                        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                        gl.draw_arrays(glow::TRIANGLES, 0, 6);
                    }
                }
            }
        }

        unsafe {
            let gl = gl.as_ref();
            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.use_program(None);
        }
    }
}

enum TileState {
    Pending {
        width: usize,
        height: usize,
        data: Vec<u16>,
    },
    Uploaded {
        tex: glow::Texture,
        filter: TextureFilter,
        width: usize,
        height: usize,
    },
}

impl TileState {
    fn bytes(&self) -> u64 {
        match self {
            Self::Pending { data, .. } => (data.len() as u64).saturating_mul(2),
            Self::Uploaded { width, height, .. } => (*width as u64)
                .saturating_mul(*height as u64)
                .saturating_mul(2),
        }
    }
}

struct QueuedTexture {
    tex: glow::Texture,
    bytes: u64,
}

#[derive(Clone)]
struct GlBindings {
    program: glow::Program,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    u_tex: Option<glow::UniformLocation>,
    u_window: Option<glow::UniformLocation>,
    u_color: Option<glow::UniformLocation>,
}

struct Inner {
    cache: LruCache<MosaicRawTileKey, TileState>,
    in_flight: HashSet<MosaicRawTileKey>,
    pending_count: usize,
    pending_cpu_bytes: u64,
    uploaded_texture_bytes: u64,
    queued_deletion_bytes: u64,
    peak_tracked_bytes: u64,
    protected_visible_bytes: u64,
    protected_keys: HashSet<MosaicRawTileKey>,
    current_channel_group: Vec<u64>,
    previous_channel_group: Vec<u64>,
    settings: ImageTileCacheSettings,
    effective_budget_bytes: u64,
    low_watermark_bytes: u64,
    pressure_state: &'static str,
    resolution_reason: &'static str,
    realized_generation: u64,
    last_policy_refresh: Option<Instant>,
    pressure_recovery_samples: u8,
    hits: u64,
    misses: u64,
    inserts: u64,
    evictions_byte_budget: u64,
    evictions_channel_change: u64,
    evictions_metadata_limit: u64,
    stale_drops_before_install: u64,
    worker_drops: u64,
    failed_loads: u64,
    gl_deletions: u64,
    textures_to_delete: Vec<QueuedTexture>,
    globj: Option<GlObjects>,
    desired_filter: TextureFilter,
    uploads_remaining: usize,
}

impl Inner {
    fn new(settings: ImageTileCacheSettings) -> Self {
        let settings = settings.normalized();
        let (effective_budget_bytes, resolution_reason) = settings.resolved_budget_bytes(None);
        let low_watermark_bytes =
            effective_budget_bytes.saturating_mul(LOW_WATERMARK_PERCENT) / 100;
        Self {
            cache: LruCache::new(NonZeroUsize::new(CACHE_METADATA_ENTRY_LIMIT).unwrap()),
            in_flight: HashSet::new(),
            pending_count: 0,
            pending_cpu_bytes: 0,
            uploaded_texture_bytes: 0,
            queued_deletion_bytes: 0,
            peak_tracked_bytes: 0,
            protected_visible_bytes: 0,
            protected_keys: HashSet::new(),
            current_channel_group: Vec::new(),
            previous_channel_group: Vec::new(),
            settings,
            effective_budget_bytes,
            low_watermark_bytes,
            pressure_state: "normal",
            resolution_reason,
            realized_generation: 0,
            last_policy_refresh: None,
            pressure_recovery_samples: 0,
            hits: 0,
            misses: 0,
            inserts: 0,
            evictions_byte_budget: 0,
            evictions_channel_change: 0,
            evictions_metadata_limit: 0,
            stale_drops_before_install: 0,
            worker_drops: 0,
            failed_loads: 0,
            gl_deletions: 0,
            textures_to_delete: Vec::new(),
            globj: None,
            desired_filter: TextureFilter::Linear,
            uploads_remaining: UPLOAD_TEXTURES_PER_FRAME,
        }
    }

    fn mark_in_flight(&mut self, key: MosaicRawTileKey) -> bool {
        if self.cache.contains(&key) || self.in_flight.contains(&key) {
            self.hits = self.hits.saturating_add(1);
            return false;
        }
        self.misses = self.misses.saturating_add(1);
        self.in_flight.insert(key);
        true
    }

    fn insert_pending(&mut self, resp: MosaicRawTileResponse) {
        if let Some(previous) = self.cache.pop(&resp.key) {
            self.remove_state(previous);
        }
        let bytes = (resp.data_u16.len() as u64).saturating_mul(2);
        let evicted = self.cache.push(
            resp.key,
            TileState::Pending {
                width: resp.width,
                height: resp.height,
                data: resp.data_u16,
            },
        );
        self.in_flight.remove(&resp.key);
        self.pending_count = self.pending_count.saturating_add(1);
        self.pending_cpu_bytes = self.pending_cpu_bytes.saturating_add(bytes);
        self.inserts = self.inserts.saturating_add(1);
        if let Some((_key, state)) = evicted {
            self.evictions_metadata_limit = self.evictions_metadata_limit.saturating_add(1);
            self.remove_state(state);
        }
        self.update_peak();
        self.enforce_budget();
    }

    fn set_policy(&mut self, settings: ImageTileCacheSettings, generation: u64) {
        let settings = settings.normalized();
        if generation <= self.realized_generation && settings == self.settings {
            return;
        }
        self.settings = settings;
        self.realized_generation = generation;
        self.last_policy_refresh = None;
        self.resolve_policy(None, None);
    }

    fn resolve_policy(&mut self, total: Option<u64>, available: Option<u64>) {
        let (ceiling, base_reason) = self.settings.resolved_budget_bytes(total);
        let (pressure, target, reason) = if self.settings.mode != ImageTileCacheMode::Automatic {
            ("normal", ceiling, base_reason)
        } else if let (Some(total), Some(available)) = (total, available) {
            if available.saturating_mul(100) < total.saturating_mul(10) {
                self.pressure_recovery_samples = 0;
                (
                    "critical",
                    (ceiling / 4).max(128 * 1024 * 1024),
                    "automatic_critical_pressure",
                )
            } else if available.saturating_mul(100) < total.saturating_mul(20) {
                self.pressure_recovery_samples = 0;
                (
                    "warning",
                    (ceiling / 2).max(128 * 1024 * 1024),
                    "automatic_warning_pressure",
                )
            } else {
                self.pressure_recovery_samples = self.pressure_recovery_samples.saturating_add(1);
                if self.pressure_state != "normal" && self.pressure_recovery_samples < 3 {
                    (
                        self.pressure_state,
                        self.effective_budget_bytes.min(ceiling),
                        "automatic_pressure_recovery_hysteresis",
                    )
                } else {
                    ("normal", ceiling, base_reason)
                }
            }
        } else {
            ("normal", ceiling, base_reason)
        };
        self.pressure_state = pressure;
        self.effective_budget_bytes = target;
        self.low_watermark_bytes = target.saturating_mul(LOW_WATERMARK_PERCENT) / 100;
        self.resolution_reason = reason;
        self.last_policy_refresh = Some(Instant::now());
        if pressure != "normal" {
            self.previous_channel_group.clear();
        }
        self.prune_disallowed_channels();
        self.enforce_budget();
    }

    fn update_working_set(
        &mut self,
        protected: HashSet<MosaicRawTileKey>,
        mut current_channel_group: Vec<u64>,
    ) {
        current_channel_group.sort_unstable();
        current_channel_group.dedup();
        if current_channel_group != self.current_channel_group {
            let retain_previous = match self.settings.channel_history {
                ImageTileChannelHistory::CurrentOnly => false,
                ImageTileChannelHistory::CurrentAndPrevious => self.pressure_state == "normal",
                ImageTileChannelHistory::Automatic => {
                    self.pressure_state == "normal"
                        && self.effective_budget_bytes >= 512 * 1024 * 1024
                }
            };
            self.previous_channel_group = if retain_previous {
                std::mem::take(&mut self.current_channel_group)
            } else {
                Vec::new()
            };
            self.current_channel_group = current_channel_group;
            self.protected_keys = protected;
            self.prune_disallowed_channels();
        } else {
            self.protected_keys = protected;
        }
        self.recalculate_protected_bytes();
        self.enforce_budget();
    }

    fn allowed_channel(&self, channel: u64) -> bool {
        self.current_channel_group.contains(&channel)
            || self.previous_channel_group.contains(&channel)
    }

    fn prune_disallowed_channels(&mut self) {
        let keys = self
            .cache
            .iter()
            .filter_map(|(key, _)| {
                (!self.protected_keys.contains(key) && !self.allowed_channel(key.channel))
                    .then_some(*key)
            })
            .collect::<Vec<_>>();
        for key in keys {
            if let Some(state) = self.cache.pop(&key) {
                self.evictions_channel_change = self.evictions_channel_change.saturating_add(1);
                self.remove_state(state);
            }
        }
        self.update_peak();
    }

    fn resident_bytes(&self) -> u64 {
        self.pending_cpu_bytes
            .saturating_add(self.uploaded_texture_bytes)
    }

    fn tracked_bytes(&self) -> u64 {
        self.resident_bytes()
            .saturating_add(self.queued_deletion_bytes)
    }

    fn update_peak(&mut self) {
        self.peak_tracked_bytes = self.peak_tracked_bytes.max(self.tracked_bytes());
    }

    fn recalculate_protected_bytes(&mut self) {
        self.protected_visible_bytes = self
            .protected_keys
            .iter()
            .filter_map(|key| self.cache.peek(key))
            .map(TileState::bytes)
            .sum();
    }

    fn eviction_candidates(&self) -> Vec<MosaicRawTileKey> {
        let mut current = Vec::new();
        let mut previous = Vec::new();
        let mut disallowed = Vec::new();
        for (key, _) in self.cache.iter() {
            if self.protected_keys.contains(key) {
                continue;
            }
            if !self.allowed_channel(key.channel) {
                disallowed.push(*key);
            } else if self.previous_channel_group.contains(&key.channel) {
                previous.push(*key);
            } else {
                current.push(*key);
            }
        }
        // LruCache::iter is most-recent to least-recent. Reverse each priority bucket so budget
        // eviction remains LRU within the channel-aware ordering without repeatedly scanning the
        // complete cache for every removed tile.
        let mut out = Vec::with_capacity(disallowed.len() + previous.len() + current.len());
        out.extend(disallowed.into_iter().rev());
        out.extend(previous.into_iter().rev());
        out.extend(current.into_iter().rev());
        out
    }

    fn enforce_budget(&mut self) {
        if self.resident_bytes() <= self.effective_budget_bytes {
            return;
        }
        for key in self.eviction_candidates() {
            if self.resident_bytes() <= self.low_watermark_bytes {
                break;
            }
            let Some(state) = self.cache.pop(&key) else {
                continue;
            };
            self.evictions_byte_budget = self.evictions_byte_budget.saturating_add(1);
            self.remove_state(state);
        }
        self.recalculate_protected_bytes();
        self.update_peak();
    }

    fn remove_state(&mut self, state: TileState) {
        match state {
            TileState::Pending { data, .. } => {
                self.pending_count = self.pending_count.saturating_sub(1);
                self.pending_cpu_bytes = self
                    .pending_cpu_bytes
                    .saturating_sub((data.len() as u64).saturating_mul(2));
            }
            TileState::Uploaded {
                tex, width, height, ..
            } => {
                let bytes = (width as u64)
                    .saturating_mul(height as u64)
                    .saturating_mul(2);
                self.uploaded_texture_bytes = self.uploaded_texture_bytes.saturating_sub(bytes);
                self.queued_deletion_bytes = self.queued_deletion_bytes.saturating_add(bytes);
                self.textures_to_delete.push(QueuedTexture { tex, bytes });
            }
        }
    }

    fn stats(&self) -> MosaicTileCacheStats {
        MosaicTileCacheStats {
            entries: self.cache.len(),
            in_flight: self.in_flight.len(),
            pending_cpu_bytes: self.pending_cpu_bytes,
            uploaded_texture_bytes: self.uploaded_texture_bytes,
            queued_deletion_bytes: self.queued_deletion_bytes,
            total_tracked_bytes: self.tracked_bytes(),
            peak_tracked_bytes: self.peak_tracked_bytes,
            protected_visible_bytes: self.protected_visible_bytes,
            over_budget_bytes: self
                .resident_bytes()
                .saturating_sub(self.effective_budget_bytes),
            effective_budget_bytes: self.effective_budget_bytes,
            low_watermark_bytes: self.low_watermark_bytes,
            hits: self.hits,
            misses: self.misses,
            inserts: self.inserts,
            evictions_byte_budget: self.evictions_byte_budget,
            evictions_channel_change: self.evictions_channel_change,
            evictions_metadata_limit: self.evictions_metadata_limit,
            stale_drops_before_install: self.stale_drops_before_install,
            worker_drops: self.worker_drops,
            failed_loads: self.failed_loads,
            gl_deletions: self.gl_deletions,
            current_channel_group: self.current_channel_group.clone(),
            previous_channel_group: self.previous_channel_group.clone(),
            pressure_state: self.pressure_state,
            resolution_reason: self.resolution_reason,
            realized_generation: self.realized_generation,
        }
    }

    fn ensure_gl(&mut self, gl: &Arc<glow::Context>) {
        if self.globj.is_some() {
            return;
        }
        self.globj = GlObjects::new(gl).ok();
    }

    fn bindings(&self) -> Option<GlBindings> {
        let g = self.globj.as_ref()?;
        Some(GlBindings {
            program: g.program,
            vao: g.vao,
            vbo: g.vbo,
            u_tex: g.u_tex.clone(),
            u_window: g.u_window.clone(),
            u_color: g.u_color.clone(),
        })
    }

    fn delete_queued_textures(&mut self, gl: &Arc<glow::Context>) {
        if self.textures_to_delete.is_empty() {
            return;
        }
        let gl = gl.as_ref();
        let delete_count = self.textures_to_delete.len().min(DELETE_TEXTURES_PER_FRAME);
        unsafe {
            for queued in self.textures_to_delete.drain(..delete_count) {
                gl.delete_texture(queued.tex);
                self.queued_deletion_bytes =
                    self.queued_deletion_bytes.saturating_sub(queued.bytes);
                self.gl_deletions = self.gl_deletions.saturating_add(1);
            }
        }
    }

    fn ensure_uploaded(
        &mut self,
        gl: &Arc<glow::Context>,
        key: &MosaicRawTileKey,
    ) -> Option<glow::Texture> {
        let state = self.cache.pop(key)?;
        match state {
            TileState::Uploaded {
                tex,
                mut filter,
                width,
                height,
            } => {
                if filter != self.desired_filter {
                    set_texture_filter(gl, tex, self.desired_filter);
                    filter = self.desired_filter;
                }
                self.cache.put(
                    *key,
                    TileState::Uploaded {
                        tex,
                        filter,
                        width,
                        height,
                    },
                );
                Some(tex)
            }
            TileState::Pending {
                width,
                height,
                data,
            } => {
                if self.uploads_remaining == 0 {
                    self.cache.put(
                        *key,
                        TileState::Pending {
                            width,
                            height,
                            data,
                        },
                    );
                    return None;
                }
                let bytes = (data.len() as u64).saturating_mul(2);
                let Some(tex) = upload_r16_texture(gl, width, height, &data, self.desired_filter)
                else {
                    self.cache.put(
                        *key,
                        TileState::Pending {
                            width,
                            height,
                            data,
                        },
                    );
                    return None;
                };
                self.uploads_remaining = self.uploads_remaining.saturating_sub(1);
                self.cache.put(
                    *key,
                    TileState::Uploaded {
                        tex,
                        filter: self.desired_filter,
                        width,
                        height,
                    },
                );
                self.pending_count = self.pending_count.saturating_sub(1);
                self.pending_cpu_bytes = self.pending_cpu_bytes.saturating_sub(bytes);
                self.uploaded_texture_bytes = self.uploaded_texture_bytes.saturating_add(bytes);
                self.update_peak();
                Some(tex)
            }
        }
    }

    fn loading_count_for(&self, keep: &HashSet<MosaicRawTileKey>) -> usize {
        keep.iter()
            .filter(|key| {
                self.in_flight.contains(*key)
                    || matches!(self.cache.peek(*key), Some(TileState::Pending { .. }))
            })
            .count()
    }
}

struct GlObjects {
    program: glow::Program,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    u_tex: Option<glow::UniformLocation>,
    u_window: Option<glow::UniformLocation>,
    u_color: Option<glow::UniformLocation>,
}

impl GlObjects {
    fn new(gl: &Arc<glow::Context>) -> anyhow::Result<Self> {
        let gl = gl.as_ref();
        let (vs, fs) = shader_sources(gl.version().major);
        let program = compile_program(gl, vs, fs)?;

        let (vao, vbo, uniforms) = unsafe {
            let vao = gl
                .create_vertex_array()
                .map_err(|e| anyhow::anyhow!("create_vertex_array failed: {e}"))?;
            let vbo = gl
                .create_buffer()
                .map_err(|e| anyhow::anyhow!("create_buffer failed: {e}"))?;
            gl.bind_vertex_array(Some(vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));

            let stride = (4 * std::mem::size_of::<f32>()) as i32;
            let Some(loc_pos) = gl.get_attrib_location(program, "a_pos_ndc") else {
                return Err(anyhow::anyhow!("missing attribute a_pos_ndc"));
            };
            let Some(loc_uv) = gl.get_attrib_location(program, "a_uv") else {
                return Err(anyhow::anyhow!("missing attribute a_uv"));
            };
            gl.enable_vertex_attrib_array(loc_pos);
            gl.vertex_attrib_pointer_f32(loc_pos, 2, glow::FLOAT, false, stride, 0);
            gl.enable_vertex_attrib_array(loc_uv);
            gl.vertex_attrib_pointer_f32(
                loc_uv,
                2,
                glow::FLOAT,
                false,
                stride,
                (2 * std::mem::size_of::<f32>()) as i32,
            );

            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);

            let u_tex = gl.get_uniform_location(program, "u_tex");
            let u_window = gl.get_uniform_location(program, "u_window");
            let u_color = gl.get_uniform_location(program, "u_color");
            Ok::<_, anyhow::Error>((vao, vbo, (u_tex, u_window, u_color)))?
        };

        Ok(Self {
            program,
            vao,
            vbo,
            u_tex: uniforms.0,
            u_window: uniforms.1,
            u_color: uniforms.2,
        })
    }
}

fn set_channel_uniforms(
    gl: &glow::Context,
    bindings: &GlBindings,
    window: (f32, f32),
    color: [f32; 3],
) {
    let (w0, w1) = window;
    unsafe {
        gl.uniform_2_f32(bindings.u_window.as_ref(), w0, w1);
        gl.uniform_3_f32(bindings.u_color.as_ref(), color[0], color[1], color[2]);
    }
}

fn tile_vertices_ndc(
    screen_rect: egui::Rect,
    viewport: egui::Rect,
    viewport_w: f32,
    viewport_h: f32,
    pixels_per_point: f32,
) -> [f32; 6 * 4] {
    let snap = |v: f32| (v * pixels_per_point).round() / pixels_per_point;
    let min_x = snap(screen_rect.min.x);
    let max_x = snap(screen_rect.max.x);
    let min_y = snap(screen_rect.min.y);
    let max_y = snap(screen_rect.max.y);

    let x0 = ((min_x - viewport.min.x) / viewport_w) * 2.0 - 1.0;
    let x1 = ((max_x - viewport.min.x) / viewport_w) * 2.0 - 1.0;
    let y0 = 1.0 - ((min_y - viewport.min.y) / viewport_h) * 2.0;
    let y1 = 1.0 - ((max_y - viewport.min.y) / viewport_h) * 2.0;

    let u0 = 0.0f32;
    let u1 = 1.0f32;
    let v0 = 0.0f32;
    let v1 = 1.0f32;

    [
        x0, y0, u0, v0, //
        x1, y0, u1, v0, //
        x1, y1, u1, v1, //
        x0, y0, u0, v0, //
        x1, y1, u1, v1, //
        x0, y1, u0, v1, //
    ]
}

fn upload_r16_texture(
    gl: &Arc<glow::Context>,
    width: usize,
    height: usize,
    data: &[u16],
    filter: TextureFilter,
) -> Option<glow::Texture> {
    if width == 0 || height == 0 || data.len() != width * height {
        return None;
    }
    let gl = gl.as_ref();
    unsafe {
        let tex = gl.create_texture().ok()?;
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, filter.as_gl());
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, filter.as_gl());
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_S,
            glow::CLAMP_TO_EDGE as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_T,
            glow::CLAMP_TO_EDGE as i32,
        );

        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::R16 as i32,
            width as i32,
            height as i32,
            0,
            glow::RED,
            glow::UNSIGNED_SHORT,
            glow::PixelUnpackData::Slice(Some(bytemuck::cast_slice(data))),
        );
        gl.bind_texture(glow::TEXTURE_2D, None);
        Some(tex)
    }
}

fn set_texture_filter(gl: &Arc<glow::Context>, tex: glow::Texture, filter: TextureFilter) {
    let gl = gl.as_ref();
    unsafe {
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, filter.as_gl());
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, filter.as_gl());
        gl.bind_texture(glow::TEXTURE_2D, None);
    }
}

fn shader_sources(gl_major: u32) -> (&'static str, &'static str) {
    if gl_major >= 3 {
        (VERT_330, FRAG_330)
    } else {
        (VERT_120, FRAG_120)
    }
}

fn compile_program(
    gl: &glow::Context,
    vs_src: &str,
    fs_src: &str,
) -> anyhow::Result<glow::Program> {
    unsafe {
        let vs = gl
            .create_shader(glow::VERTEX_SHADER)
            .map_err(|e| anyhow::anyhow!("create vertex shader failed: {e}"))?;
        gl.shader_source(vs, vs_src);
        gl.compile_shader(vs);
        if !gl.get_shader_compile_status(vs) {
            let log = gl.get_shader_info_log(vs);
            gl.delete_shader(vs);
            return Err(anyhow::anyhow!("vertex shader compile failed: {log}"));
        }

        let fs = gl
            .create_shader(glow::FRAGMENT_SHADER)
            .map_err(|e| anyhow::anyhow!("create fragment shader failed: {e}"))?;
        gl.shader_source(fs, fs_src);
        gl.compile_shader(fs);
        if !gl.get_shader_compile_status(fs) {
            let log = gl.get_shader_info_log(fs);
            gl.delete_shader(vs);
            gl.delete_shader(fs);
            return Err(anyhow::anyhow!("fragment shader compile failed: {log}"));
        }

        let program = gl
            .create_program()
            .map_err(|e| anyhow::anyhow!("create_program failed: {e}"))?;
        gl.attach_shader(program, vs);
        gl.attach_shader(program, fs);
        gl.bind_attrib_location(program, 0, "a_pos_ndc");
        gl.bind_attrib_location(program, 1, "a_uv");
        gl.link_program(program);
        gl.detach_shader(program, vs);
        gl.detach_shader(program, fs);
        gl.delete_shader(vs);
        gl.delete_shader(fs);
        if !gl.get_program_link_status(program) {
            let log = gl.get_program_info_log(program);
            gl.delete_program(program);
            return Err(anyhow::anyhow!("program link failed: {log}"));
        }
        Ok(program)
    }
}

const VERT_330: &str = r#"#version 330 core
layout(location = 0) in vec2 a_pos_ndc;
layout(location = 1) in vec2 a_uv;

out vec2 v_uv;

void main() {
    gl_Position = vec4(a_pos_ndc, 0.0, 1.0);
    v_uv = a_uv;
}
"#;

const FRAG_330: &str = r#"#version 330 core
in vec2 v_uv;

uniform sampler2D u_tex;
uniform vec2 u_window;
uniform vec3 u_color;

out vec4 out_color;

void main() {
    float raw = texture(u_tex, v_uv).r * 65535.0;
    float denom = max(u_window.y - u_window.x, 1.0);
    float t = clamp((raw - u_window.x) / denom, 0.0, 1.0);
    vec3 rgb = t * u_color;
    out_color = vec4(rgb, 1.0);
}
"#;

const VERT_120: &str = r#"#version 120
attribute vec2 a_pos_ndc;
attribute vec2 a_uv;

varying vec2 v_uv;

void main() {
    gl_Position = vec4(a_pos_ndc, 0.0, 1.0);
    v_uv = a_uv;
}
"#;

const FRAG_120: &str = r#"#version 120
varying vec2 v_uv;

uniform sampler2D u_tex;
uniform vec2 u_window;
uniform vec3 u_color;

void main() {
    float raw = texture2D(u_tex, v_uv).r * 65535.0;
    float denom = max(u_window.y - u_window.x, 1.0);
    float t = clamp((raw - u_window.x) / denom, 0.0, 1.0);
    vec3 rgb = t * u_color;
    gl_FragColor = vec4(rgb, 1.0);
}
"#;

#[cfg(test)]
mod cache_tests {
    use super::*;

    fn key(channel: u64, tile_x: u64) -> MosaicRawTileKey {
        MosaicRawTileKey {
            dataset_id: 0,
            level: 0,
            tile_y: 0,
            tile_x,
            channel,
        }
    }

    fn response(key: MosaicRawTileKey, width: usize, height: usize) -> MosaicRawTileResponse {
        MosaicRawTileResponse {
            key,
            generation: 1,
            width,
            height,
            data_u16: vec![7; width * height],
        }
    }

    fn tiny_budget(inner: &mut Inner, high: u64, low: u64) {
        inner.effective_budget_bytes = high;
        inner.low_watermark_bytes = low;
    }

    #[test]
    fn pending_tiles_are_accounted_by_actual_edge_dimensions() {
        let mut inner = Inner::new(ImageTileCacheSettings::default());
        inner.insert_pending(response(key(0, 0), 512, 512));
        inner.insert_pending(response(key(0, 1), 17, 9));
        let expected = ((512 * 512 + 17 * 9) * 2) as u64;
        assert_eq!(inner.pending_cpu_bytes, expected);
        assert_eq!(inner.uploaded_texture_bytes, 0);
        assert_eq!(inner.tracked_bytes(), expected);
        assert_eq!(inner.cache.len(), 2);
    }

    #[test]
    fn byte_budget_evicts_to_low_watermark_but_protects_visible_tiles() {
        let mut inner = Inner::new(ImageTileCacheSettings::default());
        tiny_budget(&mut inner, 48, 32);
        let visible = key(0, 0);
        inner.update_working_set(HashSet::from([visible]), vec![0]);
        inner.insert_pending(response(visible, 4, 4)); // 32 bytes, protected
        inner.insert_pending(response(key(0, 1), 4, 4));
        assert!(inner.cache.contains(&visible));
        assert!(!inner.cache.contains(&key(0, 1)));
        assert_eq!(inner.pending_cpu_bytes, 32);
        assert_eq!(inner.evictions_byte_budget, 1);
    }

    #[test]
    fn oversized_visible_working_set_is_reported_instead_of_evicted() {
        let mut inner = Inner::new(ImageTileCacheSettings::default());
        tiny_budget(&mut inner, 16, 12);
        let visible = key(0, 0);
        inner.update_working_set(HashSet::from([visible]), vec![0]);
        inner.insert_pending(response(visible, 4, 4));
        let stats = inner.stats();
        assert!(inner.cache.contains(&visible));
        assert_eq!(stats.over_budget_bytes, 16);
        assert_eq!(stats.protected_visible_bytes, 32);
    }

    #[test]
    fn current_and_previous_history_discards_older_channel_groups() {
        let mut settings = ImageTileCacheSettings::default();
        settings.channel_history = ImageTileChannelHistory::CurrentAndPrevious;
        let mut inner = Inner::new(settings);
        inner.update_working_set(HashSet::new(), vec![1, 2]);
        inner.insert_pending(response(key(1, 0), 2, 2));
        inner.insert_pending(response(key(2, 0), 2, 2));
        inner.update_working_set(HashSet::new(), vec![3]);
        assert!(inner.cache.contains(&key(1, 0)));
        assert!(inner.cache.contains(&key(2, 0)));
        inner.insert_pending(response(key(3, 0), 2, 2));
        inner.update_working_set(HashSet::new(), vec![4, 5]);
        assert!(!inner.cache.contains(&key(1, 0)));
        assert!(!inner.cache.contains(&key(2, 0)));
        assert!(inner.cache.contains(&key(3, 0)));
        assert_eq!(inner.current_channel_group, vec![4, 5]);
        assert_eq!(inner.previous_channel_group, vec![3]);
    }

    #[test]
    fn current_only_drops_old_channel_immediately() {
        let mut settings = ImageTileCacheSettings::default();
        settings.channel_history = ImageTileChannelHistory::CurrentOnly;
        let mut inner = Inner::new(settings);
        inner.update_working_set(HashSet::new(), vec![7]);
        inner.insert_pending(response(key(7, 0), 2, 2));
        inner.update_working_set(HashSet::new(), vec![8]);
        assert!(!inner.cache.contains(&key(7, 0)));
        assert_eq!(inner.evictions_channel_change, 1);
    }
}
