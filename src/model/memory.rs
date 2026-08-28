use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use serde_json::{Value, json};

use crate::control::ControlError;
use crate::settings::{
    ImageTileCacheMode, ImageTileCacheSettings, ImageTileChannelHistory,
    MAX_CUSTOM_IMAGE_TILE_CACHE_BYTES, MIN_CUSTOM_IMAGE_TILE_CACHE_BYTES,
};

#[derive(Debug, Clone)]
pub struct ControlPinnedLevelResource {
    level: usize,
    width: usize,
    height: usize,
    channel_offsets: Arc<HashMap<u64, usize>>,
    data: Arc<Vec<u16>>,
    bytes: u64,
}

impl ControlPinnedLevelResource {
    pub(crate) fn new(
        level: usize,
        width: usize,
        height: usize,
        channel_offsets: HashMap<u64, usize>,
        data: Vec<u16>,
    ) -> Self {
        let bytes = (data.len() as u64).saturating_mul(2);
        Self {
            level,
            width,
            height,
            channel_offsets: Arc::new(channel_offsets),
            data: Arc::new(data),
            bytes,
        }
    }

    pub fn level(&self) -> usize {
        self.level
    }
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
    pub fn channel_offsets(&self) -> &Arc<HashMap<u64, usize>> {
        &self.channel_offsets
    }
    pub fn data(&self) -> &Arc<Vec<u16>> {
        &self.data
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn channels_loaded(&self) -> usize {
        self.channel_offsets.len()
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SystemMemorySnapshot {
    pub(crate) total_bytes: u64,
    pub(crate) available_bytes: u64,
}

#[derive(Debug, Clone)]
enum PinnedLevelModelState {
    Loaded(Arc<ControlPinnedLevelResource>),
    Failed(String),
}

#[derive(Debug, Clone, Default)]
pub(crate) struct PinnedMemoryModel {
    levels: BTreeMap<usize, PinnedLevelModelState>,
    selected_channels: Vec<usize>,
    status: String,
    system: Option<SystemMemorySnapshot>,
    operation_generation: u64,
    projection_generation: u64,
    pending: HashMap<usize, u64>,
}

impl PinnedMemoryModel {
    fn touch_projection(&mut self) {
        self.projection_generation = self.projection_generation.wrapping_add(1).max(1);
    }

    pub(crate) fn projection_generation(&self) -> u64 {
        self.projection_generation
    }

    pub(crate) fn selected_channels(&self) -> &[usize] {
        &self.selected_channels
    }

    pub(crate) fn begin(
        &mut self,
        level: usize,
        selected_channels: Vec<usize>,
        status: String,
    ) -> u64 {
        self.operation_generation = self.operation_generation.wrapping_add(1).max(1);
        self.selected_channels = selected_channels;
        self.status = status;
        self.pending.insert(level, self.operation_generation);
        self.touch_projection();
        self.operation_generation
    }

    pub(crate) fn is_current(&self, level: usize, generation: u64) -> bool {
        self.pending.get(&level).copied() == Some(generation)
    }

    pub(crate) fn pending_generation(&self, level: usize) -> Option<u64> {
        self.pending.get(&level).copied()
    }

    pub(crate) fn install(
        &mut self,
        level: usize,
        generation: u64,
        resource: Arc<ControlPinnedLevelResource>,
        system: Option<SystemMemorySnapshot>,
    ) -> bool {
        if !self.is_current(level, generation) {
            return false;
        }
        self.pending.remove(&level);
        self.system = system;
        self.status = format!("Loaded pinned level {level} into RAM.");
        self.levels
            .insert(level, PinnedLevelModelState::Loaded(resource));
        self.touch_projection();
        true
    }

    pub(crate) fn confirmation(
        &mut self,
        level: usize,
        generation: u64,
        system: Option<SystemMemorySnapshot>,
    ) -> bool {
        if !self.is_current(level, generation) {
            return false;
        }
        self.pending.remove(&level);
        self.system = system;
        self.status = format!("RAM pinning level {level} requires confirmation.");
        self.touch_projection();
        true
    }

    pub(crate) fn fail(&mut self, level: usize, generation: u64, message: String) -> bool {
        if !self.is_current(level, generation) {
            return false;
        }
        self.pending.remove(&level);
        self.status = message.clone();
        self.levels
            .insert(level, PinnedLevelModelState::Failed(message));
        self.touch_projection();
        true
    }

    pub(crate) fn cancel(&mut self, level: usize, generation: u64, message: String) -> bool {
        if !self.is_current(level, generation) {
            return false;
        }
        self.pending.remove(&level);
        self.status = message;
        self.touch_projection();
        true
    }

    pub(crate) fn unpin(&mut self, level: usize) -> bool {
        self.operation_generation = self.operation_generation.wrapping_add(1).max(1);
        self.pending.remove(&level);
        let removed = self.levels.remove(&level).is_some();
        self.status = format!("Unloaded pinned level {level} from RAM.");
        self.touch_projection();
        removed
    }

    pub(crate) fn unpin_all(&mut self) -> usize {
        self.operation_generation = self.operation_generation.wrapping_add(1).max(1);
        self.pending.clear();
        let count = self.levels.len();
        self.levels.clear();
        self.status = format!("Unloaded {count} pinned level(s) from RAM.");
        self.touch_projection();
        count
    }

    pub(crate) fn clear_for_document(&mut self, channel_count: usize) {
        self.operation_generation = self.operation_generation.wrapping_add(1).max(1);
        self.levels.clear();
        self.pending.clear();
        self.selected_channels = (0..channel_count).collect();
        self.status.clear();
        self.touch_projection();
    }

    pub(crate) fn resources(&self) -> Vec<Arc<ControlPinnedLevelResource>> {
        self.levels
            .values()
            .filter_map(|state| match state {
                PinnedLevelModelState::Loaded(resource) => Some(Arc::clone(resource)),
                _ => None,
            })
            .collect()
    }

    pub(crate) fn total_loaded_bytes(&self) -> u64 {
        self.resources()
            .iter()
            .map(|resource| resource.bytes())
            .sum()
    }

    pub(crate) fn status(&self, level: usize) -> (&str, Option<u64>, Option<usize>, Option<&str>) {
        if self.pending.contains_key(&level) {
            return ("loading", None, None, None);
        }
        match self.levels.get(&level) {
            None => ("unloaded", None, None, None),
            Some(PinnedLevelModelState::Loaded(resource)) => (
                "loaded",
                Some(resource.bytes()),
                Some(resource.channels_loaded()),
                None,
            ),
            Some(PinnedLevelModelState::Failed(error)) => {
                ("failed", None, None, Some(error.as_str()))
            }
        }
    }

    pub(crate) fn running(&self) -> bool {
        !self.pending.is_empty()
    }
    pub(crate) fn status_message(&self) -> &str {
        &self.status
    }
    pub(crate) fn system(&self) -> Option<SystemMemorySnapshot> {
        self.system
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TilePrefetchMode {
    Off,
    TargetHalo,
    TargetAndFinerHalo,
}

impl TilePrefetchMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::TargetHalo => "target_halo",
            Self::TargetAndFinerHalo => "target_and_finer_halo",
        }
    }

    fn parse(value: &str) -> Result<Self, ControlError> {
        match value {
            "off" => Ok(Self::Off),
            "target_halo" => Ok(Self::TargetHalo),
            "target_and_finer_halo" => Ok(Self::TargetAndFinerHalo),
            _ => Err(ControlError::invalid_params(
                "memory.tiles.set",
                "unknown prefetch_mode",
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TilePrefetchAggressiveness {
    Conservative,
    Balanced,
    Aggressive,
}

impl TilePrefetchAggressiveness {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Conservative => "conservative",
            Self::Balanced => "balanced",
            Self::Aggressive => "aggressive",
        }
    }

    fn parse(value: &str) -> Result<Self, ControlError> {
        match value {
            "conservative" => Ok(Self::Conservative),
            "balanced" => Ok(Self::Balanced),
            "aggressive" => Ok(Self::Aggressive),
            _ => Err(ControlError::invalid_params(
                "memory.tiles.set",
                "unknown prefetch_aggressiveness",
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileLoadingPolicy {
    workers: usize,
    prefetch_mode: TilePrefetchMode,
    prefetch_aggressiveness: TilePrefetchAggressiveness,
    prefer_pinned_finer_levels: bool,
    image_tile_cache: ImageTileCacheSettings,
    generation: u64,
}

impl Default for TileLoadingPolicy {
    fn default() -> Self {
        Self {
            workers: std::thread::available_parallelism()
                .map(|count| count.get().min(6))
                .unwrap_or(4)
                .max(2),
            prefetch_mode: TilePrefetchMode::TargetHalo,
            prefetch_aggressiveness: TilePrefetchAggressiveness::Balanced,
            prefer_pinned_finer_levels: false,
            image_tile_cache: ImageTileCacheSettings::default(),
            generation: 1,
        }
    }
}

impl TileLoadingPolicy {
    pub fn workers(&self) -> usize {
        self.workers
    }

    pub fn prefetch_mode(&self) -> TilePrefetchMode {
        self.prefetch_mode
    }

    pub fn prefetch_aggressiveness(&self) -> TilePrefetchAggressiveness {
        self.prefetch_aggressiveness
    }

    pub fn prefer_pinned_finer_levels(&self) -> bool {
        self.prefer_pinned_finer_levels
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn image_tile_cache(&self) -> ImageTileCacheSettings {
        self.image_tile_cache
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TileLoadingModel {
    policy: TileLoadingPolicy,
    status: String,
    cache_loaded: usize,
    cache_capacity: usize,
    cache_in_flight: usize,
    target_level: Option<usize>,
    realized_generation: u64,
    cache_observation: Value,
}

impl Default for TileLoadingModel {
    fn default() -> Self {
        Self {
            policy: TileLoadingPolicy::default(),
            status: String::new(),
            cache_loaded: 0,
            cache_capacity: 256,
            cache_in_flight: 0,
            target_level: None,
            realized_generation: 0,
            cache_observation: json!({}),
        }
    }
}

impl TileLoadingModel {
    pub(crate) fn policy(&self) -> &TileLoadingPolicy {
        &self.policy
    }

    pub(crate) fn apply_image_tile_cache_settings(&mut self, settings: ImageTileCacheSettings) {
        let settings = settings.normalized();
        if self.policy.image_tile_cache == settings {
            return;
        }
        self.policy.image_tile_cache = settings;
        self.policy.generation = self.policy.generation.wrapping_add(1).max(1);
        self.status =
            "Updated mosaic image tile cache policy; renderer realization pending.".to_string();
    }

    pub(crate) fn set(
        &mut self,
        params: &Value,
        runtime_tuning_supported: bool,
    ) -> Result<Value, ControlError> {
        let mut candidate = self.policy.clone();
        let old_workers = candidate.workers;
        if let Some(value) = params.get("workers") {
            candidate.workers = value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| (1..=12).contains(value))
                .ok_or_else(|| {
                    ControlError::invalid_params(
                        "memory.tiles.set",
                        "workers must be an integer from 1 to 12",
                    )
                })?;
        }
        if candidate.workers != old_workers && !runtime_tuning_supported {
            return Err(ControlError::invalid_params(
                "memory.tiles.set",
                "runtime tile-loader tuning is unavailable for this dataset backend",
            ));
        }
        if let Some(value) = params.get("prefetch_mode") {
            candidate.prefetch_mode =
                TilePrefetchMode::parse(value.as_str().ok_or_else(|| {
                    ControlError::invalid_params(
                        "memory.tiles.set",
                        "prefetch_mode must be a string",
                    )
                })?)?;
        }
        if let Some(value) = params.get("prefetch_aggressiveness") {
            candidate.prefetch_aggressiveness =
                TilePrefetchAggressiveness::parse(value.as_str().ok_or_else(|| {
                    ControlError::invalid_params(
                        "memory.tiles.set",
                        "prefetch_aggressiveness must be a string",
                    )
                })?)?;
        }
        if let Some(value) = params.get("prefer_pinned_finer_levels") {
            candidate.prefer_pinned_finer_levels = value.as_bool().ok_or_else(|| {
                ControlError::invalid_params(
                    "memory.tiles.set",
                    "prefer_pinned_finer_levels must be a boolean",
                )
            })?;
        }
        if let Some(value) = params.get("cache_mode") {
            candidate.image_tile_cache.mode = ImageTileCacheMode::parse(
                value.as_str().ok_or_else(|| {
                    ControlError::invalid_params("memory.tiles.set", "cache_mode must be a string")
                })?,
            )
            .ok_or_else(|| {
                ControlError::invalid_params(
                    "memory.tiles.set",
                    "cache_mode must be automatic, conservative, balanced, performance, or custom",
                )
            })?;
        }
        if let Some(value) = params.get("cache_budget_bytes") {
            candidate.image_tile_cache.custom_budget_bytes = value
                .as_u64()
                .filter(|value| {
                    (MIN_CUSTOM_IMAGE_TILE_CACHE_BYTES..=MAX_CUSTOM_IMAGE_TILE_CACHE_BYTES)
                        .contains(value)
                })
                .ok_or_else(|| {
                    ControlError::invalid_params(
                        "memory.tiles.set",
                        "cache_budget_bytes must be between 128 MiB and 4 GiB",
                    )
                })?;
        }
        if let Some(value) = params.get("channel_history") {
            candidate.image_tile_cache.channel_history =
                ImageTileChannelHistory::parse(value.as_str().ok_or_else(|| {
                    ControlError::invalid_params(
                        "memory.tiles.set",
                        "channel_history must be a string",
                    )
                })?)
                .ok_or_else(|| {
                    ControlError::invalid_params(
                        "memory.tiles.set",
                        "channel_history must be automatic, current_only, or current_and_previous",
                    )
                })?;
        }
        candidate.image_tile_cache = candidate.image_tile_cache.normalized();
        if candidate != self.policy {
            candidate.generation = self.policy.generation.wrapping_add(1).max(1);
            self.status = if candidate.workers != old_workers {
                format!(
                    "Tile-loader policy set to {} worker(s); renderer realization pending.",
                    candidate.workers
                )
            } else {
                "Updated tile loading policy.".to_string()
            };
            self.policy = candidate;
        }
        Ok(self.snapshot(runtime_tuning_supported))
    }

    pub(crate) fn observe(&mut self, value: &Value) {
        if let Some(cache) = value.get("cache").filter(|cache| cache.is_object()) {
            self.cache_observation = cache.clone();
        }
        self.cache_loaded = value
            .get("cache")
            .and_then(|cache| cache.get("loaded"))
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(self.cache_loaded);
        self.cache_capacity = value
            .get("cache")
            .and_then(|cache| cache.get("capacity"))
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(self.cache_capacity);
        self.cache_in_flight = value
            .get("cache")
            .and_then(|cache| cache.get("in_flight"))
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(self.cache_in_flight);
        self.target_level = value
            .get("target_level")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok());
        self.realized_generation = value
            .get("realized_generation")
            .and_then(Value::as_u64)
            .unwrap_or(self.realized_generation)
            .min(self.policy.generation);
        if self.realized_generation == self.policy.generation {
            self.status = value
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("Tile loading policy realized by renderer.")
                .to_string();
        }
    }

    pub(crate) fn reset_observation(&mut self) {
        self.cache_loaded = 0;
        self.cache_capacity = 256;
        self.cache_in_flight = 0;
        self.target_level = None;
        self.realized_generation = 0;
        self.cache_observation = json!({});
    }

    pub(crate) fn snapshot(&self, runtime_tuning_supported: bool) -> Value {
        let mut cache = self.cache_observation.clone();
        if !cache.is_object() {
            cache = json!({});
        }
        let cache_object = cache.as_object_mut().expect("cache observation object");
        cache_object
            .entry("loaded".to_string())
            .or_insert(json!(self.cache_loaded));
        cache_object
            .entry("capacity".to_string())
            .or_insert(json!(self.cache_capacity));
        cache_object
            .entry("in_flight".to_string())
            .or_insert(json!(self.cache_in_flight));
        json!({
            "workers":self.policy.workers,
            "runtime_tuning_supported":runtime_tuning_supported,
            "prefetch_mode":self.policy.prefetch_mode.as_str(),
            "prefetch_aggressiveness":self.policy.prefetch_aggressiveness.as_str(),
            "prefer_pinned_finer_levels":self.policy.prefer_pinned_finer_levels,
            "cache_mode":self.policy.image_tile_cache.mode.as_str(),
            "cache_budget_bytes":if self.policy.image_tile_cache.mode == ImageTileCacheMode::Custom {
                Some(self.policy.image_tile_cache.custom_budget_bytes)
            } else {
                None
            },
            "channel_history":self.policy.image_tile_cache.channel_history.as_str(),
            "status":self.status,
            "cache":cache,
            "target_level":self.target_level,
            "generation":self.policy.generation,
            "realized_generation":self.realized_generation,
            "presentation_pending":self.realized_generation < self.policy.generation,
        })
    }
}
