use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use crossbeam_channel::{Receiver, Sender};
use lru::LruCache;

use crate::data::ome::retrieve_image_subset_u16;
use crate::data::ome::{Dims, LevelInfo};
use crate::imaging::view_plane::{ViewPlaneSelection, display_axes, image_subset_ranges_for_view};
use crate::render::array_dims::squeeze_to_2d;
use crate::{log_debug, log_warn};
use zarrs::array::{Array, ArraySubset};
use zarrs::storage::ReadableStorageTraits;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct RawTileKey {
    pub view: ViewPlaneSelection,
    pub level: usize,
    pub tile_y: u64,
    pub tile_x: u64,
    pub channel: u64,
}

#[derive(Debug, Clone)]
pub struct RawTileRequest {
    pub key: RawTileKey,
}

#[derive(Debug, Clone)]
pub struct RawTileResponse {
    pub key: RawTileKey,
    pub width: usize,
    pub height: usize,
    pub data_u16: Vec<u16>,
}

#[derive(Debug, Clone)]
pub enum RawTileWorkerResponse {
    Tile(RawTileResponse),
    Failed { key: RawTileKey, error: String },
}

#[derive(Debug)]
pub struct RawTileLoaderHandle {
    pub tx: Sender<RawTileRequest>,
    pub rx: Receiver<RawTileWorkerResponse>,
}

pub struct RawTileCache<T> {
    cache: LruCache<RawTileKey, T>,
    in_flight: HashSet<RawTileKey>,
}

impl<T> RawTileCache<T> {
    pub fn new(capacity_tiles: usize) -> Self {
        Self {
            cache: LruCache::new(NonZeroUsize::new(capacity_tiles.max(1)).unwrap()),
            in_flight: HashSet::new(),
        }
    }

    pub fn contains(&self, key: &RawTileKey) -> bool {
        self.cache.contains(key)
    }

    pub fn get_mut(&mut self, key: &RawTileKey) -> Option<&mut T> {
        self.cache.get_mut(key)
    }

    pub fn push(&mut self, key: RawTileKey, value: T) -> Option<(RawTileKey, T)> {
        self.in_flight.remove(&key);
        self.cache.push(key, value)
    }

    pub fn mark_in_flight(&mut self, key: RawTileKey) -> bool {
        if self.cache.contains(&key) || self.in_flight.contains(&key) {
            return false;
        }
        self.in_flight.insert(key);
        true
    }

    pub fn cancel_in_flight(&mut self, key: &RawTileKey) {
        self.in_flight.remove(key);
    }

    pub fn drain(&mut self) -> Vec<(RawTileKey, T)> {
        self.in_flight.clear();
        let mut out = Vec::with_capacity(self.cache.len());
        while let Some((k, v)) = self.cache.pop_lru() {
            out.push((k, v));
        }
        out
    }

    pub fn prune_in_flight(&mut self, keep: &HashSet<RawTileKey>) {
        self.in_flight.retain(|k| keep.contains(k));
    }

    pub fn is_busy(&self) -> bool {
        !self.in_flight.is_empty()
    }

    pub fn in_flight_len(&self) -> usize {
        self.in_flight.len()
    }

    pub fn capacity(&self) -> usize {
        self.cache.cap().get()
    }

    pub fn grow_capacity(&mut self, capacity_tiles: usize) {
        let capacity_tiles = capacity_tiles.max(1);
        if capacity_tiles <= self.capacity() {
            return;
        }
        self.cache
            .resize(NonZeroUsize::new(capacity_tiles).unwrap());
    }
}

pub fn spawn_raw_tile_loader(
    store: Arc<dyn ReadableStorageTraits>,
    levels: Vec<LevelInfo>,
    dims: Dims,
    worker_threads: usize,
) -> anyhow::Result<RawTileLoaderHandle> {
    let (tx_req, rx_req) = crossbeam_channel::unbounded::<RawTileRequest>();
    let (tx_rsp, rx_rsp) = crossbeam_channel::unbounded::<RawTileWorkerResponse>();
    let threads = worker_threads.max(1);
    let levels = Arc::new(levels);

    for worker_idx in 0..threads {
        let rx_req = rx_req.clone();
        let tx_rsp = tx_rsp.clone();
        let store = store.clone();
        let levels = Arc::clone(&levels);
        let dims = dims.clone();
        std::thread::Builder::new()
            .name(format!("raw-tile-loader-{worker_idx}"))
            .spawn(move || {
                if let Err(err) = raw_tile_loader_thread(store, levels, dims, rx_req, tx_rsp) {
                    eprintln!("raw tile loader worker exited: {err:?}");
                }
            })
            .context("failed to spawn raw tile loader thread")?;
    }

    Ok(RawTileLoaderHandle {
        tx: tx_req,
        rx: rx_rsp,
    })
}

fn raw_tile_loader_thread(
    store: Arc<dyn ReadableStorageTraits>,
    levels: Arc<Vec<LevelInfo>>,
    dims: Dims,
    rx_req: Receiver<RawTileRequest>,
    tx_rsp: Sender<RawTileWorkerResponse>,
) -> anyhow::Result<()> {
    let mut arrays: Vec<Array<dyn ReadableStorageTraits>> = Vec::with_capacity(levels.len());
    for info in levels.iter() {
        let path = &info.path;
        let zarr_path = format!("/{}", path.trim_start_matches('/'));
        arrays.push(Array::open(store.clone(), &zarr_path)?);
    }
    let Some(level0) = levels.first() else {
        return Ok(());
    };

    for req in rx_req.iter() {
        let debug_io = crate::debug_log::debug_io_enabled();
        let start = Instant::now();
        let worker_name = std::thread::current()
            .name()
            .unwrap_or("raw-tile-loader")
            .to_string();
        if debug_io {
            log_debug!(
                "{worker_name}: start {:?} lvl={} tile=({}, {}) ch={}",
                req.key.view,
                req.key.level,
                req.key.tile_y,
                req.key.tile_x,
                req.key.channel
            );
        }
        let level = req.key.level;
        if level >= arrays.len() {
            let error = format!("invalid raw tile level {}", req.key.level);
            if debug_io {
                log_warn!(
                    "{worker_name}: fail lvl={} tile=({}, {}) ch={} after {:?}: {}",
                    req.key.level,
                    req.key.tile_y,
                    req.key.tile_x,
                    req.key.channel,
                    start.elapsed(),
                    error
                );
            }
            let _ = tx_rsp.send(RawTileWorkerResponse::Failed {
                key: req.key,
                error,
            });
            continue;
        }
        let array = &arrays[level];
        let level_info = &levels[level];
        let shape = &level_info.shape;
        let chunks = &level_info.chunks;
        let dtype = &level_info.dtype;

        let Some(display_axes) = display_axes(&dims, req.key.view.mode) else {
            let error = "unsupported view plane for this dataset".to_string();
            let _ = tx_rsp.send(RawTileWorkerResponse::Failed {
                key: req.key,
                error,
            });
            continue;
        };
        let y_dim = display_axes.vertical;
        let x_dim = display_axes.horizontal;
        let y_chunk = chunks[y_dim];
        let x_chunk = chunks[x_dim];

        let y0 = req.key.tile_y * y_chunk;
        let x0 = req.key.tile_x * x_chunk;
        let y1 = (y0 + y_chunk).min(shape[y_dim]);
        let x1 = (x0 + x_chunk).min(shape[x_dim]);

        let height = (y1 - y0) as usize;
        let width = (x1 - x0) as usize;

        let Some(ranges) = image_subset_ranges_for_view(
            &dims,
            level0,
            level_info,
            Some(req.key.channel),
            y0..y1,
            x0..x1,
            req.key.view,
        ) else {
            let error = "unsupported view plane for this dataset".to_string();
            let _ = tx_rsp.send(RawTileWorkerResponse::Failed {
                key: req.key,
                error,
            });
            continue;
        };

        let subset = ArraySubset::new_with_ranges(&ranges);
        let data = match retrieve_image_subset_u16(array, &subset, dtype) {
            Ok(data) => data,
            Err(err) => {
                if debug_io {
                    log_warn!(
                        "{worker_name}: fail {:?} lvl={} tile=({}, {}) ch={} after {:?}: {}",
                        req.key.view,
                        req.key.level,
                        req.key.tile_y,
                        req.key.tile_x,
                        req.key.channel,
                        start.elapsed(),
                        err
                    );
                }
                let _ = tx_rsp.send(RawTileWorkerResponse::Failed {
                    key: req.key,
                    error: err.to_string(),
                });
                continue;
            }
        };

        let data = match squeeze_to_2d(data, y_dim, x_dim) {
            Some(data) => data,
            None => {
                let error =
                    "unexpected array dimensionality for raw tile (expected displayed axes plus singleton dims)".to_string();
                if debug_io {
                    log_warn!(
                        "{worker_name}: fail {:?} lvl={} tile=({}, {}) ch={} after {:?}: {}",
                        req.key.view,
                        req.key.level,
                        req.key.tile_y,
                        req.key.tile_x,
                        req.key.channel,
                        start.elapsed(),
                        error
                    );
                }
                let _ = tx_rsp.send(RawTileWorkerResponse::Failed {
                    key: req.key,
                    error,
                });
                continue;
            }
        };

        let data_u16 = data.iter().copied().collect::<Vec<_>>();
        if data_u16.len() != width * height {
            let error = format!(
                "raw tile size mismatch: got {} samples for {}x{} tile",
                data_u16.len(),
                width,
                height
            );
            if debug_io {
                log_warn!(
                    "{worker_name}: fail {:?} lvl={} tile=({}, {}) ch={} after {:?}: {}",
                    req.key.view,
                    req.key.level,
                    req.key.tile_y,
                    req.key.tile_x,
                    req.key.channel,
                    start.elapsed(),
                    error
                );
            }
            let _ = tx_rsp.send(RawTileWorkerResponse::Failed {
                key: req.key,
                error,
            });
            continue;
        }

        if debug_io {
            log_debug!(
                "{worker_name}: done {:?} lvl={} tile=({}, {}) ch={} {}x{} after {:?}",
                req.key.view,
                req.key.level,
                req.key.tile_y,
                req.key.tile_x,
                req.key.channel,
                width,
                height,
                start.elapsed()
            );
        }
        let _ = tx_rsp.send(RawTileWorkerResponse::Tile(RawTileResponse {
            key: req.key,
            width,
            height,
            data_u16,
        }));
    }

    Ok(())
}
