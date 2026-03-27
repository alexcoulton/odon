use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Context;
use parking_lot::Mutex;
use zarrs::array::{Array, ArraySubset};
use zarrs::storage::ReadableStorageTraits;

use crate::data::ome::{Dims, LevelInfo, retrieve_image_subset_u16};
use crate::render::tiles::{RenderChannel, TileKey, TileResponse};
use crate::render::tiles_raw::{RawTileKey, RawTileResponse};

#[derive(Clone)]
pub struct PinnedLevels {
    inner: Arc<Mutex<PinnedLevelsInner>>,
}

#[derive(Debug, Clone)]
pub enum PinnedLevelStatus {
    Unloaded,
    Loading,
    Loaded { bytes: u64, channels_loaded: usize },
    Failed(String),
}

#[derive(Debug, Clone)]
struct PinnedLevelData {
    width: usize,
    height: usize,
    channel_offsets: HashMap<u64, usize>,
    data: Arc<Vec<u16>>,
    bytes: u64,
}

#[derive(Debug, Clone)]
enum PinnedLevelState {
    Loading { request_id: u64 },
    Loaded(PinnedLevelData),
    Failed(String),
}

#[derive(Default)]
struct PinnedLevelsInner {
    levels: HashMap<usize, PinnedLevelState>,
    next_request_id: u64,
}

impl PinnedLevels {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(PinnedLevelsInner::default())),
        }
    }

    pub fn status(&self, level: usize) -> PinnedLevelStatus {
        match self.inner.lock().levels.get(&level) {
            None => PinnedLevelStatus::Unloaded,
            Some(PinnedLevelState::Loading { .. }) => PinnedLevelStatus::Loading,
            Some(PinnedLevelState::Loaded(data)) => PinnedLevelStatus::Loaded {
                bytes: data.bytes,
                channels_loaded: data.channel_offsets.len(),
            },
            Some(PinnedLevelState::Failed(err)) => PinnedLevelStatus::Failed(err.clone()),
        }
    }

    pub fn unload(&self, level: usize) {
        self.inner.lock().levels.remove(&level);
    }

    pub fn total_loaded_bytes(&self) -> u64 {
        self.inner
            .lock()
            .levels
            .values()
            .filter_map(|state| match state {
                PinnedLevelState::Loaded(data) => Some(data.bytes),
                _ => None,
            })
            .sum()
    }

    pub fn has_loading(&self) -> bool {
        self.inner
            .lock()
            .levels
            .values()
            .any(|state| matches!(state, PinnedLevelState::Loading { .. }))
    }

    pub fn request_load(
        &self,
        store: Arc<dyn ReadableStorageTraits>,
        dims: Dims,
        levels: Vec<LevelInfo>,
        level: usize,
        selected_channels: Vec<u64>,
    ) {
        let request_id = {
            let mut inner = self.inner.lock();
            inner.next_request_id = inner.next_request_id.wrapping_add(1).max(1);
            let request_id = inner.next_request_id;
            inner
                .levels
                .insert(level, PinnedLevelState::Loading { request_id });
            request_id
        };

        let this = self.clone();
        std::thread::Builder::new()
            .name(format!("pin-level-{level}"))
            .spawn(move || {
                let result = load_full_level(store, dims, levels, level, &selected_channels);
                let mut inner = this.inner.lock();
                let is_current = matches!(
                    inner.levels.get(&level),
                    Some(PinnedLevelState::Loading { request_id: current }) if *current == request_id
                );
                if !is_current {
                    return;
                }
                match result {
                    Ok(data) => {
                        inner.levels.insert(level, PinnedLevelState::Loaded(data));
                    }
                    Err(err) => {
                        inner
                            .levels
                            .insert(level, PinnedLevelState::Failed(err.to_string()));
                    }
                }
            })
            .ok();
    }

    pub fn try_get_raw_tile(
        &self,
        key: RawTileKey,
        dims: &Dims,
        level: &LevelInfo,
    ) -> Option<RawTileResponse> {
        let data = match self.inner.lock().levels.get(&key.level) {
            Some(PinnedLevelState::Loaded(data)) => data.clone(),
            _ => return None,
        };
        let channel_offset = data.channel_offsets.get(&key.channel).copied()?;
        let (width, height, tile_data) =
            slice_tile_u16(&data, channel_offset, level, dims, key.tile_y, key.tile_x)?;
        Some(RawTileResponse {
            key,
            width,
            height,
            data_u16: tile_data,
        })
    }

    pub fn try_get_raw_tile_resampled_from_level(
        &self,
        source_level: usize,
        key: RawTileKey,
        dims: &Dims,
        target_level: &LevelInfo,
        source_level_info: &LevelInfo,
    ) -> Option<RawTileResponse> {
        let data = match self.inner.lock().levels.get(&source_level) {
            Some(PinnedLevelState::Loaded(data)) => data.clone(),
            _ => return None,
        };
        let channel_offset = data.channel_offsets.get(&key.channel).copied()?;
        let (width, height, tile_data) = resample_tile_u16_from_level(
            &data,
            channel_offset,
            source_level_info,
            target_level,
            dims,
            key.tile_y,
            key.tile_x,
        )?;
        Some(RawTileResponse {
            key,
            width,
            height,
            data_u16: tile_data,
        })
    }

    pub fn try_get_composited_tile(
        &self,
        key: TileKey,
        channels: &[RenderChannel],
        dims: &Dims,
        level: &LevelInfo,
    ) -> Option<TileResponse> {
        if channels.is_empty() {
            return None;
        }
        let data = match self.inner.lock().levels.get(&key.level) {
            Some(PinnedLevelState::Loaded(data)) => data.clone(),
            _ => return None,
        };

        let Some((width, height, first_channel)) = channels.first().and_then(|ch| {
            let channel_offset = data.channel_offsets.get(&ch.index).copied()?;
            slice_tile_u16(&data, channel_offset, level, dims, key.tile_y, key.tile_x)
        }) else {
            return None;
        };

        let mut acc = vec![0.0f32; width.saturating_mul(height).saturating_mul(3)];
        accumulate_channel(&first_channel, width, height, &channels[0], &mut acc);

        for ch in channels.iter().skip(1) {
            let Some(channel_offset) = data.channel_offsets.get(&ch.index).copied() else {
                return None;
            };
            let Some((tile_width, tile_height, tile_data)) =
                slice_tile_u16(&data, channel_offset, level, dims, key.tile_y, key.tile_x)
            else {
                return None;
            };
            if tile_width != width || tile_height != height {
                return None;
            }
            accumulate_channel(&tile_data, width, height, ch, &mut acc);
        }

        let mut rgba = vec![0u8; width.saturating_mul(height).saturating_mul(4)];
        for i in 0..(width * height) {
            rgba[i * 4] = (acc[i * 3].clamp(0.0, 1.0) * 255.0).round() as u8;
            rgba[i * 4 + 1] = (acc[i * 3 + 1].clamp(0.0, 1.0) * 255.0).round() as u8;
            rgba[i * 4 + 2] = (acc[i * 3 + 2].clamp(0.0, 1.0) * 255.0).round() as u8;
            rgba[i * 4 + 3] = 255;
        }

        Some(TileResponse {
            key,
            width,
            height,
            rgba,
        })
    }

    pub fn try_get_composited_tile_resampled_from_level(
        &self,
        source_level: usize,
        key: TileKey,
        channels: &[RenderChannel],
        dims: &Dims,
        target_level: &LevelInfo,
        source_level_info: &LevelInfo,
    ) -> Option<TileResponse> {
        if channels.is_empty() {
            return None;
        }
        let data = match self.inner.lock().levels.get(&source_level) {
            Some(PinnedLevelState::Loaded(data)) => data.clone(),
            _ => return None,
        };

        let Some((width, height, first_channel)) = channels.first().and_then(|ch| {
            let channel_offset = data.channel_offsets.get(&ch.index).copied()?;
            resample_tile_u16_from_level(
                &data,
                channel_offset,
                source_level_info,
                target_level,
                dims,
                key.tile_y,
                key.tile_x,
            )
        }) else {
            return None;
        };

        let mut acc = vec![0.0f32; width.saturating_mul(height).saturating_mul(3)];
        accumulate_channel(&first_channel, width, height, &channels[0], &mut acc);

        for ch in channels.iter().skip(1) {
            let Some(channel_offset) = data.channel_offsets.get(&ch.index).copied() else {
                return None;
            };
            let Some((tile_width, tile_height, tile_data)) = resample_tile_u16_from_level(
                &data,
                channel_offset,
                source_level_info,
                target_level,
                dims,
                key.tile_y,
                key.tile_x,
            ) else {
                return None;
            };
            if tile_width != width || tile_height != height {
                return None;
            }
            accumulate_channel(&tile_data, width, height, ch, &mut acc);
        }

        let mut rgba = vec![0u8; width.saturating_mul(height).saturating_mul(4)];
        for i in 0..(width * height) {
            rgba[i * 4] = (acc[i * 3].clamp(0.0, 1.0) * 255.0).round() as u8;
            rgba[i * 4 + 1] = (acc[i * 3 + 1].clamp(0.0, 1.0) * 255.0).round() as u8;
            rgba[i * 4 + 2] = (acc[i * 3 + 2].clamp(0.0, 1.0) * 255.0).round() as u8;
            rgba[i * 4 + 3] = 255;
        }

        Some(TileResponse {
            key,
            width,
            height,
            rgba,
        })
    }
}

impl Default for PinnedLevels {
    fn default() -> Self {
        Self::new()
    }
}

fn load_full_level(
    store: Arc<dyn ReadableStorageTraits>,
    dims: Dims,
    levels: Vec<LevelInfo>,
    level: usize,
    selected_channels: &[u64],
) -> anyhow::Result<PinnedLevelData> {
    let info = levels
        .get(level)
        .cloned()
        .with_context(|| format!("missing level {level}"))?;
    let zarr_path = format!("/{}", info.path.trim_start_matches('/'));
    let array: Array<dyn ReadableStorageTraits> = Array::open(store, &zarr_path)?;

    if let Some(c_dim) = dims.c {
        let height = *info.shape.get(dims.y).unwrap_or(&0) as usize;
        let width = *info.shape.get(dims.x).unwrap_or(&0) as usize;
        let plane_len = height.saturating_mul(width);
        let mut raw = Vec::<u16>::new();
        let mut channel_offsets = HashMap::<u64, usize>::new();

        for &channel in selected_channels {
            let mut ranges: Vec<std::ops::Range<u64>> = Vec::with_capacity(info.shape.len());
            for dim in 0..info.shape.len() {
                if dim == c_dim {
                    ranges.push(channel..(channel + 1));
                } else if dim == dims.y || dim == dims.x {
                    ranges.push(0..info.shape[dim]);
                } else {
                    ranges.push(0..1);
                }
            }
            let subset = ArraySubset::new_with_ranges(&ranges);
            let data = retrieve_image_subset_u16(&array, &subset, &info.dtype)?;
            let data = data
                .into_dimensionality::<ndarray::Ix3>()
                .context("unexpected dimensionality for pinned level channel")?;
            let plane = data.index_axis(ndarray::Axis(0), 0).to_owned();
            let (plane_raw, offset) = plane.into_raw_vec_and_offset();
            if offset.unwrap_or(0) != 0 {
                anyhow::bail!("unexpected non-zero offset in pinned level channel buffer");
            }
            if plane_raw.len() != plane_len {
                anyhow::bail!("unexpected plane length while pinning level");
            }
            channel_offsets.insert(channel, raw.len() / plane_len.max(1));
            raw.extend_from_slice(&plane_raw);
        }

        if channel_offsets.is_empty() {
            anyhow::bail!("none of the selected channels were pinned");
        }

        Ok(PinnedLevelData {
            width,
            height,
            channel_offsets,
            bytes: (raw.len() as u64).saturating_mul(2),
            data: Arc::new(raw),
        })
    } else {
        if selected_channels.is_empty() {
            anyhow::bail!("no channels selected for pinning");
        }

        let mut ranges: Vec<std::ops::Range<u64>> = Vec::with_capacity(info.shape.len());
        for dim in 0..info.shape.len() {
            if dim == dims.y || dim == dims.x {
                ranges.push(0..info.shape[dim]);
            } else {
                ranges.push(0..1);
            }
        }
        let subset = ArraySubset::new_with_ranges(&ranges);
        let data = retrieve_image_subset_u16(&array, &subset, &info.dtype)?;
        let data = data
            .into_dimensionality::<ndarray::Ix2>()
            .context("unexpected dimensionality for pinned level")?;
        let height = data.shape()[0];
        let width = data.shape()[1];
        let (raw, offset) = data.into_raw_vec_and_offset();
        if offset.unwrap_or(0) != 0 {
            anyhow::bail!("unexpected non-zero offset in pinned level buffer");
        }
        let mut channel_offsets = HashMap::new();
        for &channel in selected_channels {
            channel_offsets.insert(channel, 0);
        }
        Ok(PinnedLevelData {
            width,
            height,
            channel_offsets,
            bytes: (raw.len() as u64).saturating_mul(2),
            data: Arc::new(raw),
        })
    }
}

fn slice_tile_u16(
    data: &PinnedLevelData,
    channel_offset: usize,
    level: &LevelInfo,
    dims: &Dims,
    tile_y: u64,
    tile_x: u64,
) -> Option<(usize, usize, Vec<u16>)> {
    let y_chunk = *level.chunks.get(dims.y)?;
    let x_chunk = *level.chunks.get(dims.x)?;
    let shape_y = *level.shape.get(dims.y)?;
    let shape_x = *level.shape.get(dims.x)?;

    let y0 = tile_y.saturating_mul(y_chunk).min(shape_y) as usize;
    let x0 = tile_x.saturating_mul(x_chunk).min(shape_x) as usize;
    let y1 = (y0 as u64 + y_chunk).min(shape_y) as usize;
    let x1 = (x0 as u64 + x_chunk).min(shape_x) as usize;
    if y1 <= y0 || x1 <= x0 {
        return None;
    }

    let width = x1 - x0;
    let height = y1 - y0;
    let plane_stride = data.width.saturating_mul(data.height);
    if plane_stride == 0 {
        return None;
    }
    let base = channel_offset.saturating_mul(plane_stride);
    let mut out = vec![0u16; width.saturating_mul(height)];
    for row in 0..height {
        let src_start = base + (y0 + row).saturating_mul(data.width) + x0;
        let src_end = src_start + width;
        let dst_start = row.saturating_mul(width);
        out[dst_start..dst_start + width].copy_from_slice(&data.data[src_start..src_end]);
    }
    Some((width, height, out))
}

fn resample_tile_u16_from_level(
    data: &PinnedLevelData,
    channel_offset: usize,
    source_level: &LevelInfo,
    target_level: &LevelInfo,
    dims: &Dims,
    tile_y: u64,
    tile_x: u64,
) -> Option<(usize, usize, Vec<u16>)> {
    let target_y_chunk = *target_level.chunks.get(dims.y)?;
    let target_x_chunk = *target_level.chunks.get(dims.x)?;
    let target_shape_y = *target_level.shape.get(dims.y)?;
    let target_shape_x = *target_level.shape.get(dims.x)?;

    let target_y0 = tile_y.saturating_mul(target_y_chunk).min(target_shape_y) as usize;
    let target_x0 = tile_x.saturating_mul(target_x_chunk).min(target_shape_x) as usize;
    let target_y1 = (target_y0 as u64 + target_y_chunk).min(target_shape_y) as usize;
    let target_x1 = (target_x0 as u64 + target_x_chunk).min(target_shape_x) as usize;
    if target_y1 <= target_y0 || target_x1 <= target_x0 {
        return None;
    }

    let width = target_x1 - target_x0;
    let height = target_y1 - target_y0;
    let plane_stride = data.width.saturating_mul(data.height);
    if plane_stride == 0 {
        return None;
    }
    let base = channel_offset.saturating_mul(plane_stride);
    let ratio_x = (target_level.downsample / source_level.downsample).max(1.0);
    let ratio_y = (target_level.downsample / source_level.downsample).max(1.0);
    let source_shape_y = *source_level.shape.get(dims.y)? as isize;
    let source_shape_x = *source_level.shape.get(dims.x)? as isize;

    let mut out = vec![0u16; width.saturating_mul(height)];
    for oy in 0..height {
        let src_y = (((target_y0 + oy) as f32 + 0.5) * ratio_y).floor() as isize;
        let src_y = src_y.clamp(0, source_shape_y.saturating_sub(1)) as usize;
        for ox in 0..width {
            let src_x = (((target_x0 + ox) as f32 + 0.5) * ratio_x).floor() as isize;
            let src_x = src_x.clamp(0, source_shape_x.saturating_sub(1)) as usize;
            let src_idx = base + src_y.saturating_mul(data.width) + src_x;
            out[oy * width + ox] = *data.data.get(src_idx)?;
        }
    }
    Some((width, height, out))
}

fn accumulate_channel(
    data: &[u16],
    width: usize,
    height: usize,
    channel: &RenderChannel,
    acc: &mut [f32],
) {
    let _ = (width, height);
    let (w0, w1) = channel.window;
    let denom = (w1 - w0).max(1.0);
    for (idx, val) in data.iter().enumerate() {
        let t = ((*val as f32 - w0) / denom).clamp(0.0, 1.0);
        acc[idx * 3] += t * channel.color_rgb[0];
        acc[idx * 3 + 1] += t * channel.color_rgb[1];
        acc[idx * 3 + 2] += t * channel.color_rgb[2];
    }
}
