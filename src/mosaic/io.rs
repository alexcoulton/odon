use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use anyhow::Context;
use crossbeam_channel::{Receiver, Sender};
use parking_lot::Mutex;
use zarrs::array::{Array, ArraySubset};
use zarrs::storage::ReadableStorageTraits;

use crate::data::dataset_source::DatasetSource;
use crate::data::ome::{Dims, LevelInfo};

#[derive(Clone)]
pub struct MosaicSource {
    pub source: DatasetSource,
    pub store: Arc<dyn ReadableStorageTraits>,
    pub levels: Vec<LevelInfo>,
    pub dims: Dims,
    /// Mapping from global channel id -> dataset channel index.
    /// `None` means this dataset does not have that channel.
    pub channel_map: Vec<Option<u64>>,
}

impl std::fmt::Debug for MosaicSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MosaicSource")
            .field("source", &self.source)
            .field("levels", &self.levels.len())
            .field("dims", &self.dims)
            .field("channel_map_len", &self.channel_map.len())
            .finish()
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct MosaicRawTileKey {
    pub dataset_id: usize,
    pub level: usize,
    pub tile_y: u64,
    pub tile_x: u64,
    pub channel: u64,
}

#[derive(Debug, Clone)]
pub struct MosaicRawTileRequest {
    pub key: MosaicRawTileKey,
    pub generation: u64,
}

#[derive(Debug, Clone)]
pub struct MosaicRawTileResponse {
    pub key: MosaicRawTileKey,
    pub generation: u64,
    pub width: usize,
    pub height: usize,
    pub data_u16: Vec<u16>,
}

#[derive(Debug, Clone)]
pub enum MosaicRawTileWorkerResponse {
    Tile(MosaicRawTileResponse),
    Dropped { key: MosaicRawTileKey },
}

#[derive(Debug)]
pub struct MosaicRawTileLoaderHandle {
    pub tx: Sender<MosaicRawTileRequest>,
    pub rx: Receiver<MosaicRawTileWorkerResponse>,
    latest_generation: Arc<AtomicU64>,
}

impl MosaicRawTileLoaderHandle {
    pub fn set_latest_generation(&self, generation: u64) {
        self.latest_generation
            .store(generation.max(1), Ordering::Relaxed);
    }
}

#[derive(Clone)]
pub struct MosaicPinnedLevels {
    inner: Arc<Mutex<MosaicPinnedLevelsInner>>,
}

#[derive(Debug, Clone)]
pub enum MosaicPinnedLevelStatus {
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
struct MosaicPinnedLevelsInner {
    levels: HashMap<(usize, usize), PinnedLevelState>,
    next_request_id: u64,
}

impl MosaicPinnedLevels {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(MosaicPinnedLevelsInner::default())),
        }
    }

    pub fn status(&self, dataset_id: usize, level: usize) -> MosaicPinnedLevelStatus {
        match self.inner.lock().levels.get(&(dataset_id, level)) {
            None => MosaicPinnedLevelStatus::Unloaded,
            Some(PinnedLevelState::Loading { .. }) => MosaicPinnedLevelStatus::Loading,
            Some(PinnedLevelState::Loaded(data)) => MosaicPinnedLevelStatus::Loaded {
                bytes: data.bytes,
                channels_loaded: data.channel_offsets.len(),
            },
            Some(PinnedLevelState::Failed(err)) => MosaicPinnedLevelStatus::Failed(err.clone()),
        }
    }

    pub fn unload(&self, dataset_id: usize, level: usize) {
        self.inner.lock().levels.remove(&(dataset_id, level));
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
        dataset_id: usize,
        source: MosaicSource,
        level: usize,
        selected_global_channels: Vec<u64>,
    ) {
        let request_id = {
            let mut inner = self.inner.lock();
            inner.next_request_id = inner.next_request_id.wrapping_add(1).max(1);
            let request_id = inner.next_request_id;
            inner.levels.insert(
                (dataset_id, level),
                PinnedLevelState::Loading { request_id },
            );
            request_id
        };

        let this = self.clone();
        std::thread::Builder::new()
            .name(format!("mosaic-pin-level-{dataset_id}-{level}"))
            .spawn(move || {
                let result = load_full_level(&source, level, &selected_global_channels);
                let mut inner = this.inner.lock();
                let is_current = matches!(
                    inner.levels.get(&(dataset_id, level)),
                    Some(PinnedLevelState::Loading { request_id: current }) if *current == request_id
                );
                if !is_current {
                    return;
                }
                match result {
                    Ok(data) => {
                        inner
                            .levels
                            .insert((dataset_id, level), PinnedLevelState::Loaded(data));
                    }
                    Err(err) => {
                        inner.levels.insert(
                            (dataset_id, level),
                            PinnedLevelState::Failed(err.to_string()),
                        );
                    }
                }
            })
            .ok();
    }

    fn try_get_tile(
        &self,
        key: MosaicRawTileKey,
        generation: u64,
        src: &MosaicSource,
        level: &LevelInfo,
    ) -> Option<MosaicRawTileResponse> {
        let data = match self.inner.lock().levels.get(&(key.dataset_id, key.level)) {
            Some(PinnedLevelState::Loaded(data)) => data.clone(),
            _ => return None,
        };

        let Some(channel_offset) = data.channel_offsets.get(&key.channel).copied() else {
            return None;
        };

        let y_chunk = *level.chunks.get(src.dims.y)?;
        let x_chunk = *level.chunks.get(src.dims.x)?;
        let shape_y = *level.shape.get(src.dims.y)?;
        let shape_x = *level.shape.get(src.dims.x)?;

        let y0 = key.tile_y.saturating_mul(y_chunk).min(shape_y) as usize;
        let x0 = key.tile_x.saturating_mul(x_chunk).min(shape_x) as usize;
        let y1 = (y0 as u64 + y_chunk).min(shape_y) as usize;
        let x1 = (x0 as u64 + x_chunk).min(shape_x) as usize;
        if y1 <= y0 || x1 <= x0 {
            return None;
        }

        let width = x1 - x0;
        let height = y1 - y0;
        let mut out = vec![0u16; width.saturating_mul(height)];
        let plane_stride = data.width.saturating_mul(data.height);
        if plane_stride == 0 {
            return None;
        }
        let base = channel_offset.saturating_mul(plane_stride);
        for row in 0..height {
            let src_start = base + (y0 + row).saturating_mul(data.width) + x0;
            let src_end = src_start + width;
            let dst_start = row.saturating_mul(width);
            out[dst_start..dst_start + width].copy_from_slice(&data.data[src_start..src_end]);
        }

        Some(MosaicRawTileResponse {
            key,
            generation,
            width,
            height,
            data_u16: out,
        })
    }
}

impl Default for MosaicPinnedLevels {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
pub fn estimate_level_ram_bytes(source: &MosaicSource, level: usize) -> Option<u64> {
    estimate_level_ram_bytes_for_channels(source, level, None)
}

pub fn estimate_level_ram_bytes_for_channels(
    source: &MosaicSource,
    level: usize,
    selected_global_channels: Option<&[u64]>,
) -> Option<u64> {
    let info = source.levels.get(level)?;
    let shape_y = *info.shape.get(source.dims.y)?;
    let shape_x = *info.shape.get(source.dims.x)?;
    let channels = if let Some(selected) = selected_global_channels {
        selected
            .iter()
            .filter(|gid| {
                source
                    .channel_map
                    .get(**gid as usize)
                    .copied()
                    .flatten()
                    .is_some()
            })
            .count() as u64
    } else {
        source
            .dims
            .c
            .and_then(|c| info.shape.get(c).copied())
            .unwrap_or(1)
    };
    let bytes_per_sample = match info.dtype.as_str() {
        "|u1" | "|i1" => 1u64,
        "<u2" | ">u2" | "<i2" | ">i2" => 2u64,
        "<f4" | ">f4" | "<u4" | ">u4" | "<i4" | ">i4" => 4u64,
        _ => 2u64,
    };
    channels
        .checked_mul(shape_y)
        .and_then(|v| v.checked_mul(shape_x))
        .and_then(|v| v.checked_mul(bytes_per_sample))
}

pub fn spawn_mosaic_raw_tile_loader(
    sources: Arc<Vec<MosaicSource>>,
    pinned_levels: MosaicPinnedLevels,
    worker_threads: usize,
    queue_capacity: usize,
) -> anyhow::Result<MosaicRawTileLoaderHandle> {
    let threads = worker_threads.max(1);
    let cap = queue_capacity.max(256);

    let (tx_req, rx_req) = crossbeam_channel::bounded::<MosaicRawTileRequest>(cap);
    let (tx_rsp, rx_rsp) = crossbeam_channel::unbounded::<MosaicRawTileWorkerResponse>();
    let latest_generation = Arc::new(AtomicU64::new(1));

    for t in 0..threads {
        let rx_req = rx_req.clone();
        let tx_rsp = tx_rsp.clone();
        let sources = Arc::clone(&sources);
        let pinned_levels = pinned_levels.clone();
        let latest_generation = Arc::clone(&latest_generation);
        std::thread::Builder::new()
            .name(format!("mosaic-raw-loader-{t}"))
            .spawn(move || {
                if let Err(err) = mosaic_raw_tile_worker(
                    sources,
                    pinned_levels,
                    rx_req,
                    tx_rsp,
                    latest_generation,
                ) {
                    eprintln!("mosaic raw tile worker exited: {err:?}");
                }
            })
            .context("failed to spawn mosaic raw tile worker")?;
    }

    Ok(MosaicRawTileLoaderHandle {
        tx: tx_req,
        rx: rx_rsp,
        latest_generation,
    })
}

struct WorkerDataset {
    arrays: Vec<Array<dyn ReadableStorageTraits>>,
}

fn mosaic_raw_tile_worker(
    sources: Arc<Vec<MosaicSource>>,
    pinned_levels: MosaicPinnedLevels,
    rx_req: Receiver<MosaicRawTileRequest>,
    tx_rsp: Sender<MosaicRawTileWorkerResponse>,
    latest_generation: Arc<AtomicU64>,
) -> anyhow::Result<()> {
    let mut opened: HashMap<usize, WorkerDataset> = HashMap::new();

    for req in rx_req.iter() {
        let key = req.key;
        if req.generation != latest_generation.load(Ordering::Relaxed) {
            let _ = tx_rsp.send(MosaicRawTileWorkerResponse::Dropped { key });
            continue;
        }
        let Some(src) = sources.get(key.dataset_id) else {
            continue;
        };
        let Some(level) = src.levels.get(key.level) else {
            continue;
        };
        if let Some(resp) = pinned_levels.try_get_tile(key, req.generation, src, level) {
            let _ = tx_rsp.send(MosaicRawTileWorkerResponse::Tile(resp));
            continue;
        }

        let ds = if opened.contains_key(&key.dataset_id) {
            opened.get_mut(&key.dataset_id).unwrap()
        } else {
            let store = src.store.clone();
            let mut arrays: Vec<Array<dyn ReadableStorageTraits>> =
                Vec::with_capacity(src.levels.len());
            for lvl in &src.levels {
                let zarr_path = format!("/{}", lvl.path.trim_start_matches('/'));
                arrays.push(Array::open(store.clone(), &zarr_path)?);
            }
            opened.insert(key.dataset_id, WorkerDataset { arrays });
            opened.get_mut(&key.dataset_id).unwrap()
        };

        let array = ds
            .arrays
            .get(key.level)
            .context("missing array for level")?;
        let shape = &level.shape;
        let chunks = &level.chunks;
        let (c_dim, y_dim, x_dim) = (src.dims.c, src.dims.y, src.dims.x);
        let y_chunk = chunks[y_dim];
        let x_chunk = chunks[x_dim];

        let y0 = key.tile_y * y_chunk;
        let x0 = key.tile_x * x_chunk;
        let y1 = (y0 + y_chunk).min(shape[y_dim]);
        let x1 = (x0 + x_chunk).min(shape[x_dim]);

        let height = (y1 - y0) as usize;
        let width = (x1 - x0) as usize;

        let mut ranges: Vec<std::ops::Range<u64>> = Vec::with_capacity(shape.len());
        let mapped_channel = src.channel_map.get(key.channel as usize).copied().flatten();
        for dim in 0..shape.len() {
            if Some(dim) == c_dim {
                let Some(ch) = mapped_channel else {
                    ranges.clear();
                    break;
                };
                ranges.push(ch..(ch + 1));
            } else if dim == y_dim {
                ranges.push(y0..y1);
            } else if dim == x_dim {
                ranges.push(x0..x1);
            } else {
                ranges.push(0..1);
            }
        }
        if ranges.is_empty() {
            continue;
        }

        let subset = ArraySubset::new_with_ranges(&ranges);
        let data: ndarray::ArrayD<u16> = array.retrieve_array_subset(&subset)?;

        let data = if c_dim.is_some() {
            data.into_dimensionality::<ndarray::Ix3>()
                .ok()
                .map(|a| a.index_axis(ndarray::Axis(0), 0).to_owned())
        } else {
            data.into_dimensionality::<ndarray::Ix2>().ok()
        }
        .context("unexpected array dimensionality for raw tile")?;

        let data_u16 = data.iter().copied().collect::<Vec<_>>();
        if data_u16.len() != width * height {
            continue;
        }
        if req.generation != latest_generation.load(Ordering::Relaxed) {
            let _ = tx_rsp.send(MosaicRawTileWorkerResponse::Dropped { key });
            continue;
        }

        let _ = tx_rsp.send(MosaicRawTileWorkerResponse::Tile(MosaicRawTileResponse {
            key,
            generation: req.generation,
            width,
            height,
            data_u16,
        }));
    }

    Ok(())
}

fn load_full_level(
    source: &MosaicSource,
    level: usize,
    selected_global_channels: &[u64],
) -> anyhow::Result<PinnedLevelData> {
    let info = source
        .levels
        .get(level)
        .with_context(|| format!("missing level {level}"))?;
    let store = source.store.clone();
    let zarr_path = format!("/{}", info.path.trim_start_matches('/'));
    let array: Array<dyn ReadableStorageTraits> = Array::open(store, &zarr_path)?;

    if let Some(c_dim) = source.dims.c {
        let height = *info.shape.get(source.dims.y).unwrap_or(&0) as usize;
        let width = *info.shape.get(source.dims.x).unwrap_or(&0) as usize;
        let plane_len = height.saturating_mul(width);
        let mut raw = Vec::<u16>::new();
        let mut channel_offsets = HashMap::<u64, usize>::new();

        for &gid in selected_global_channels {
            let Some(dataset_channel) = source.channel_map.get(gid as usize).copied().flatten()
            else {
                continue;
            };
            let mut ranges: Vec<std::ops::Range<u64>> = Vec::with_capacity(info.shape.len());
            for dim in 0..info.shape.len() {
                if dim == c_dim {
                    ranges.push(dataset_channel..(dataset_channel + 1));
                } else if dim == source.dims.y || dim == source.dims.x {
                    ranges.push(0..info.shape[dim]);
                } else {
                    ranges.push(0..1);
                }
            }
            let subset = ArraySubset::new_with_ranges(&ranges);
            let data: ndarray::ArrayD<u16> = array.retrieve_array_subset(&subset)?;
            let data = data
                .into_dimensionality::<ndarray::Ix3>()
                .context("unexpected dimensionality for pinned mosaic level")?;
            let plane = data.index_axis(ndarray::Axis(0), 0).to_owned();
            let (plane_raw, offset) = plane.into_raw_vec_and_offset();
            if offset.unwrap_or(0) != 0 {
                anyhow::bail!("unexpected non-zero offset in pinned mosaic level channel buffer");
            }
            if plane_raw.len() != plane_len {
                anyhow::bail!("unexpected plane length while pinning mosaic level");
            }
            channel_offsets.insert(gid, raw.len() / plane_len.max(1));
            raw.extend_from_slice(&plane_raw);
        }

        if channel_offsets.is_empty() {
            anyhow::bail!("none of the selected channels are present in this ROI");
        }

        Ok(PinnedLevelData {
            height,
            width,
            channel_offsets,
            bytes: (raw.len() as u64).saturating_mul(2),
            data: Arc::new(raw),
        })
    } else {
        let matched_global_channels = if selected_global_channels.is_empty() {
            all_global_channels(source)
        } else {
            selected_global_channels
                .iter()
                .copied()
                .filter(|gid| {
                    source
                        .channel_map
                        .get(*gid as usize)
                        .copied()
                        .flatten()
                        .is_some()
                })
                .collect::<Vec<_>>()
        };
        if matched_global_channels.is_empty() {
            anyhow::bail!("none of the selected channels are present in this ROI");
        }

        let mut ranges: Vec<std::ops::Range<u64>> = Vec::with_capacity(info.shape.len());
        for dim in 0..info.shape.len() {
            if dim == source.dims.y || dim == source.dims.x {
                ranges.push(0..info.shape[dim]);
            } else {
                ranges.push(0..1);
            }
        }
        let subset = ArraySubset::new_with_ranges(&ranges);
        let data: ndarray::ArrayD<u16> = array.retrieve_array_subset(&subset)?;
        let data = data
            .into_dimensionality::<ndarray::Ix2>()
            .context("unexpected dimensionality for pinned mosaic level")?;
        let height = data.shape()[0];
        let width = data.shape()[1];
        let (raw, offset) = data.into_raw_vec_and_offset();
        if offset.unwrap_or(0) != 0 {
            anyhow::bail!("unexpected non-zero offset in pinned mosaic level buffer");
        }
        let mut channel_offsets = HashMap::new();
        for gid in matched_global_channels {
            channel_offsets.insert(gid, 0);
        }
        Ok(PinnedLevelData {
            height,
            width,
            channel_offsets,
            bytes: (raw.len() as u64).saturating_mul(2),
            data: Arc::new(raw),
        })
    }
}

fn all_global_channels(source: &MosaicSource) -> Vec<u64> {
    source
        .channel_map
        .iter()
        .enumerate()
        .filter_map(|(gid, mapped)| mapped.map(|_| gid as u64))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::estimate_level_ram_bytes;
    use crate::data::ome::{Dims, LevelInfo};

    #[test]
    fn estimates_ram_for_cyx_level() {
        let source = super::MosaicSource {
            source: crate::data::dataset_source::DatasetSource::Local(std::path::PathBuf::from(
                "dummy",
            )),
            store: Arc::new(zarrs::filesystem::FilesystemStore::new(".").unwrap()),
            levels: vec![LevelInfo {
                index: 2,
                path: "2".to_string(),
                shape: vec![4, 100, 50],
                chunks: vec![1, 32, 32],
                downsample: 4.0,
                dtype: "<u2".to_string(),
            }],
            dims: Dims {
                c: Some(0),
                y: 1,
                x: 2,
                ndim: 3,
            },
            channel_map: vec![Some(0), Some(1), Some(2), Some(3)],
        };

        assert_eq!(estimate_level_ram_bytes(&source, 0), Some(4 * 100 * 50 * 2));
    }
}
