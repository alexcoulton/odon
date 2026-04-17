use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::Arc;

use anyhow::Context;
use crossbeam_channel::{Receiver, Sender};
use lru::LruCache;

use crate::data::ome::retrieve_image_subset_u16;
use crate::data::ome::{Dims, LevelInfo};
use crate::imaging::view_plane::{
    ViewPlaneSelection, display_axes, image_subset_ranges_for_view,
};
use crate::render::array_dims::squeeze_to_2d;
use zarrs::array::{Array, ArraySubset};
use zarrs::storage::ReadableStorageTraits;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct TileKey {
    pub render_id: u64,
    pub view: ViewPlaneSelection,
    pub level: usize,
    pub tile_y: u64,
    pub tile_x: u64,
}

#[derive(Debug, Clone)]
pub struct RenderChannel {
    pub index: u64,
    pub color_rgb: [f32; 3],
    pub window: (f32, f32),
}

#[derive(Debug, Clone)]
pub struct TileRequest {
    pub key: TileKey,
    pub channels: Vec<RenderChannel>,
}

#[derive(Debug, Clone)]
pub struct TileResponse {
    pub key: TileKey,
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}

#[derive(Debug, Clone)]
pub enum TileWorkerResponse {
    Tile(TileResponse),
    Failed { key: TileKey, error: String },
}

#[derive(Debug)]
pub struct TileLoaderHandle {
    pub tx: Sender<TileRequest>,
    pub rx: Receiver<TileWorkerResponse>,
}

pub struct TileCache<T> {
    cache: LruCache<TileKey, T>,
    in_flight: HashSet<TileKey>,
}

impl<T> TileCache<T> {
    pub fn new(capacity_tiles: usize) -> Self {
        Self {
            cache: LruCache::new(NonZeroUsize::new(capacity_tiles.max(1)).unwrap()),
            in_flight: HashSet::new(),
        }
    }

    pub fn get(&mut self, key: &TileKey) -> Option<&T> {
        self.cache.get(key)
    }

    pub fn put(&mut self, key: TileKey, value: T) {
        self.cache.put(key, value);
        self.in_flight.remove(&key);
    }

    pub fn mark_in_flight(&mut self, key: TileKey) -> bool {
        if self.cache.contains(&key) || self.in_flight.contains(&key) {
            return false;
        }
        self.in_flight.insert(key);
        true
    }

    pub fn cancel_in_flight(&mut self, key: &TileKey) {
        self.in_flight.remove(key);
    }

    pub fn prune_in_flight(&mut self, keep: &HashSet<TileKey>) {
        self.in_flight.retain(|k| keep.contains(k));
    }

    pub fn is_busy(&self) -> bool {
        !self.in_flight.is_empty()
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn capacity(&self) -> usize {
        self.cache.cap().get()
    }

    pub fn in_flight_len(&self) -> usize {
        self.in_flight.len()
    }
}

pub fn recommended_tile_loader_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(6))
        .unwrap_or(4)
        .max(2)
}

pub fn spawn_tile_loader(
    store: Arc<dyn ReadableStorageTraits>,
    levels: Vec<LevelInfo>,
    dims: Dims,
    worker_threads: usize,
) -> anyhow::Result<TileLoaderHandle> {
    let (tx_req, rx_req) = crossbeam_channel::unbounded::<TileRequest>();
    let (tx_rsp, rx_rsp) = crossbeam_channel::unbounded::<TileWorkerResponse>();
    let threads = worker_threads.max(1);
    let levels = Arc::new(levels);

    for worker_idx in 0..threads {
        let rx_req = rx_req.clone();
        let tx_rsp = tx_rsp.clone();
        let store = store.clone();
        let levels = Arc::clone(&levels);
        let dims = dims.clone();
        std::thread::Builder::new()
            .name(format!("tile-loader-{worker_idx}"))
            .spawn(move || {
                if let Err(err) = tile_loader_thread(store, levels, dims, rx_req, tx_rsp) {
                    eprintln!("tile loader worker exited: {err:?}");
                }
            })
            .context("failed to spawn tile loader thread")?;
    }

    Ok(TileLoaderHandle {
        tx: tx_req,
        rx: rx_rsp,
    })
}

fn tile_loader_thread(
    store: Arc<dyn ReadableStorageTraits>,
    levels: Arc<Vec<LevelInfo>>,
    dims: Dims,
    rx_req: Receiver<TileRequest>,
    tx_rsp: Sender<TileWorkerResponse>,
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
        let level = req.key.level;
        if level >= arrays.len() {
            continue;
        }
        let array = &arrays[level];
        let level_info = &levels[level];
        let shape = &level_info.shape;
        let chunks = &level_info.chunks;
        let dtype = &level_info.dtype;

        let Some(display_axes) = display_axes(&dims, req.key.view.mode) else {
            let _ = tx_rsp.send(TileWorkerResponse::Failed {
                key: req.key,
                error: "unsupported view plane for this dataset".to_string(),
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

        let mut acc = vec![0.0f32; width * height * 3];

        for ch in &req.channels {
            let (w0, w1) = ch.window;
            let denom = (w1 - w0).max(1.0);
            let Some(ranges) = image_subset_ranges_for_view(
                &dims,
                level0,
                level_info,
                Some(ch.index),
                y0..y1,
                x0..x1,
                req.key.view,
            ) else {
                let _ = tx_rsp.send(TileWorkerResponse::Failed {
                    key: req.key,
                    error: "unsupported view plane for this dataset".to_string(),
                });
                continue;
            };

            let subset = ArraySubset::new_with_ranges(&ranges);
            let data = match retrieve_image_subset_u16(array, &subset, dtype) {
                Ok(data) => data,
                Err(err) => {
                    let _ = tx_rsp.send(TileWorkerResponse::Failed {
                        key: req.key,
                        error: err.to_string(),
                    });
                    continue;
                }
            };

            let data = squeeze_to_2d(data, y_dim, x_dim).context(
                "unexpected array dimensionality for tile (expected displayed axes plus singleton dims)",
            )?;

            for (idx, val) in data.iter().enumerate() {
                let t = ((*val as f32 - w0) / denom).clamp(0.0, 1.0);
                acc[idx * 3 + 0] += t * ch.color_rgb[0];
                acc[idx * 3 + 1] += t * ch.color_rgb[1];
                acc[idx * 3 + 2] += t * ch.color_rgb[2];
            }
        }

        let mut rgba = vec![0u8; width * height * 4];
        for i in 0..(width * height) {
            let r = (acc[i * 3 + 0].clamp(0.0, 1.0) * 255.0).round() as u8;
            let g = (acc[i * 3 + 1].clamp(0.0, 1.0) * 255.0).round() as u8;
            let b = (acc[i * 3 + 2].clamp(0.0, 1.0) * 255.0).round() as u8;
            rgba[i * 4 + 0] = r;
            rgba[i * 4 + 1] = g;
            rgba[i * 4 + 2] = b;
            rgba[i * 4 + 3] = 255;
        }

        let _ = tx_rsp.send(TileWorkerResponse::Tile(TileResponse {
            key: req.key,
            width,
            height,
            rgba,
        }));
    }

    Ok(())
}
