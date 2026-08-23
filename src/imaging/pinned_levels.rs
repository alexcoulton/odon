use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::data::ome::{Dims, LevelInfo};
#[cfg(test)]
use crate::render::array_dims::squeeze_to_2d;
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

#[derive(Default)]
struct PinnedLevelsInner {
    levels: HashMap<usize, PinnedLevelData>,
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
            Some(data) => PinnedLevelStatus::Loaded {
                bytes: data.bytes,
                channels_loaded: data.channel_offsets.len(),
            },
        }
    }

    pub fn total_loaded_bytes(&self) -> u64 {
        self.inner
            .lock()
            .levels
            .values()
            .map(|data| data.bytes)
            .sum()
    }

    pub fn replace_control_actor_resources(
        &self,
        resources: &[Arc<odon::model::ControlPinnedLevelResource>],
    ) {
        let levels = resources
            .iter()
            .map(|resource| {
                (
                    resource.level(),
                    PinnedLevelData {
                        width: resource.width(),
                        height: resource.height(),
                        channel_offsets: resource.channel_offsets().as_ref().clone(),
                        data: Arc::clone(resource.data()),
                        bytes: resource.bytes(),
                    },
                )
            })
            .collect();
        self.inner.lock().levels = levels;
    }

    pub fn try_get_raw_tile(
        &self,
        key: RawTileKey,
        dims: &Dims,
        level: &LevelInfo,
    ) -> Option<RawTileResponse> {
        let data = match self.inner.lock().levels.get(&key.level) {
            Some(data) => data.clone(),
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
            Some(data) => data.clone(),
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
            Some(data) => data.clone(),
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
            Some(data) => data.clone(),
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

#[cfg(test)]
fn pinned_level_plane(
    data: ndarray::ArrayD<u16>,
    y_dim: usize,
    x_dim: usize,
) -> Option<ndarray::Array2<u16>> {
    squeeze_to_2d(data, y_dim, x_dim)
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

#[cfg(test)]
mod tests {
    use super::pinned_level_plane;
    use ndarray::{Array, IxDyn};

    #[test]
    fn pinned_level_plane_squeezes_singleton_tczyx() {
        let data = Array::from_iter(0u16..12)
            .into_shape_with_order(IxDyn(&[1, 1, 1, 3, 4]))
            .expect("shape");

        let plane = pinned_level_plane(data, 3, 4).expect("plane");

        assert_eq!(plane.shape(), &[3, 4]);
        assert_eq!(plane[(0, 0)], 0);
        assert_eq!(plane[(2, 3)], 11);
    }

    #[test]
    fn pinned_level_plane_rejects_non_singleton_non_spatial_axis() {
        let data = Array::from_iter(0u16..24)
            .into_shape_with_order(IxDyn(&[2, 1, 3, 4]))
            .expect("shape");

        assert!(pinned_level_plane(data, 2, 3).is_none());
    }
}
