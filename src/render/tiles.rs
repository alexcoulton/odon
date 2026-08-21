use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Context;
use crossbeam_channel::{Receiver, Sender};
use lru::LruCache;

use crate::data::ome::retrieve_image_subset_u16;
use crate::data::ome::{Dims, LevelInfo};
use crate::imaging::view_plane::{ViewPlaneSelection, display_axes, image_subset_ranges_for_view};
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TileLoaderStatsSnapshot {
    pub decode_requests: u64,
    pub source_reads: u64,
    pub cache_hits: u64,
    pub decoded_cache_entries: usize,
    pub decoded_cache_bytes: usize,
}

#[derive(Debug, Default)]
struct TileLoaderStats {
    decode_requests: AtomicU64,
    source_reads: AtomicU64,
    cache_hits: AtomicU64,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub(crate) struct DecodedTileKey {
    pub(crate) view: ViewPlaneSelection,
    pub(crate) level: usize,
    pub(crate) tile_y: u64,
    pub(crate) tile_x: u64,
    pub(crate) channel: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct DecodedTile {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) values: Arc<Vec<u16>>,
}

type DecodedTileCell = Arc<OnceLock<Result<DecodedTile, String>>>;

/// Shared, presentation-independent CPU decode cache.
///
/// The key deliberately excludes render generation, colour and contrast so
/// multiple viewports can composite the same source samples differently while
/// paying for source IO and decoding only once.
#[derive(Debug)]
pub(crate) struct CpuDecodedTileCache {
    cache: Mutex<LruCache<DecodedTileKey, DecodedTileCell>>,
    stats: TileLoaderStats,
}

impl CpuDecodedTileCache {
    pub(crate) fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            cache: Mutex::new(LruCache::new(
                NonZeroUsize::new(capacity.max(1)).expect("non-zero decoded tile cache capacity"),
            )),
            stats: TileLoaderStats::default(),
        })
    }

    pub(crate) fn get_or_decode<F>(
        &self,
        key: DecodedTileKey,
        decode: F,
    ) -> Result<DecodedTile, String>
    where
        F: FnOnce() -> Result<DecodedTile, String>,
    {
        self.stats.decode_requests.fetch_add(1, Ordering::Relaxed);
        let (cell, cache_hit) = {
            let mut cache = self
                .cache
                .lock()
                .map_err(|_| "decoded tile cache lock poisoned".to_string())?;
            match cache.get(&key) {
                Some(cell) => (Arc::clone(cell), true),
                None => {
                    let cell = Arc::new(OnceLock::new());
                    cache.put(key, Arc::clone(&cell));
                    (cell, false)
                }
            }
        };
        if cache_hit {
            self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
        }
        cell.get_or_init(|| {
            self.stats.source_reads.fetch_add(1, Ordering::Relaxed);
            decode()
        })
        .clone()
    }

    fn stats(&self) -> TileLoaderStatsSnapshot {
        let (decoded_cache_entries, decoded_cache_bytes) = self
            .cache
            .lock()
            .map(|cache| {
                let bytes = cache
                    .iter()
                    .filter_map(|(_, cell)| cell.get())
                    .filter_map(|result| result.as_ref().ok())
                    .map(|tile| tile.values.len().saturating_mul(std::mem::size_of::<u16>()))
                    .sum();
                (cache.len(), bytes)
            })
            .unwrap_or_default();
        TileLoaderStatsSnapshot {
            decode_requests: self.stats.decode_requests.load(Ordering::Relaxed),
            source_reads: self.stats.source_reads.load(Ordering::Relaxed),
            cache_hits: self.stats.cache_hits.load(Ordering::Relaxed),
            decoded_cache_entries,
            decoded_cache_bytes,
        }
    }
}

#[derive(Debug)]
pub struct TileLoaderHandle {
    pub tx: Sender<TileRequest>,
    pub rx: Receiver<TileWorkerResponse>,
    pub(crate) active_render_ids: Arc<Mutex<HashSet<u64>>>,
    pub(crate) active_keys: Arc<Mutex<HashSet<TileKey>>>,
    decoded_tiles: Arc<CpuDecodedTileCache>,
}

impl TileLoaderHandle {
    pub(crate) fn with_decoded_tiles(
        tx: Sender<TileRequest>,
        rx: Receiver<TileWorkerResponse>,
        active_render_ids: Arc<Mutex<HashSet<u64>>>,
        active_keys: Arc<Mutex<HashSet<TileKey>>>,
        decoded_tiles: Arc<CpuDecodedTileCache>,
    ) -> Self {
        Self {
            tx,
            rx,
            active_render_ids,
            active_keys,
            decoded_tiles,
        }
    }

    /// Replace the accepted render generations with one generation.
    ///
    /// This remains the efficient single-viewport compatibility path. A zero
    /// render ID clears generation filtering.
    pub fn set_latest_render_id(&self, render_id: u64) {
        if let Ok(mut active) = self.active_render_ids.lock() {
            active.clear();
            if render_id != 0 {
                active.insert(render_id);
            }
        }
    }

    /// Replace the set of render generations that may complete concurrently.
    pub fn set_active_render_ids(&self, render_ids: HashSet<u64>) {
        if let Ok(mut active) = self.active_render_ids.lock() {
            *active = render_ids;
        }
    }

    /// Admit a newly-created presentation generation without invalidating
    /// generations used by another viewport.
    pub fn activate_render_id(&self, render_id: u64) {
        if render_id == 0 {
            return;
        }
        if let Ok(mut active) = self.active_render_ids.lock() {
            active.insert(render_id);
        }
    }

    pub fn set_active_keys(&self, keys: HashSet<TileKey>) {
        if let Ok(mut active) = self.active_keys.lock() {
            *active = keys;
        }
    }

    pub fn stats(&self) -> TileLoaderStatsSnapshot {
        self.decoded_tiles.stats()
    }
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
    let active_render_ids = Arc::new(Mutex::new(HashSet::new()));
    let active_keys = Arc::new(Mutex::new(HashSet::new()));
    let decoded_tiles = CpuDecodedTileCache::new(8192);

    for worker_idx in 0..threads {
        let rx_req = rx_req.clone();
        let tx_rsp = tx_rsp.clone();
        let store = store.clone();
        let levels = Arc::clone(&levels);
        let active_render_ids = Arc::clone(&active_render_ids);
        let active_keys = Arc::clone(&active_keys);
        let decoded_tiles = Arc::clone(&decoded_tiles);
        let dims = dims.clone();
        std::thread::Builder::new()
            .name(format!("tile-loader-{worker_idx}"))
            .spawn(move || {
                if let Err(err) = tile_loader_thread(
                    store,
                    levels,
                    dims,
                    rx_req,
                    tx_rsp,
                    active_render_ids,
                    active_keys,
                    decoded_tiles,
                ) {
                    eprintln!("tile loader worker exited: {err:?}");
                }
            })
            .context("failed to spawn tile loader thread")?;
    }

    Ok(TileLoaderHandle::with_decoded_tiles(
        tx_req,
        rx_rsp,
        active_render_ids,
        active_keys,
        decoded_tiles,
    ))
}

fn tile_loader_thread(
    store: Arc<dyn ReadableStorageTraits>,
    levels: Arc<Vec<LevelInfo>>,
    dims: Dims,
    rx_req: Receiver<TileRequest>,
    tx_rsp: Sender<TileWorkerResponse>,
    active_render_ids: Arc<Mutex<HashSet<u64>>>,
    active_keys: Arc<Mutex<HashSet<TileKey>>>,
    decoded_tiles: Arc<CpuDecodedTileCache>,
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

    'requests: for req in rx_req.iter() {
        if let Ok(active) = active_render_ids.lock()
            && !active.is_empty()
            && !active.contains(&req.key.render_id)
        {
            continue;
        }
        if let Ok(active) = active_keys.lock()
            && !active.is_empty()
            && !active.contains(&req.key)
        {
            continue;
        }
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
            let decoded_key = DecodedTileKey {
                view: req.key.view,
                level,
                tile_y: req.key.tile_y,
                tile_x: req.key.tile_x,
                channel: ch.index,
            };
            let decoded = decoded_tiles.get_or_decode(decoded_key, || {
                let ranges = image_subset_ranges_for_view(
                    &dims,
                    level0,
                    level_info,
                    Some(ch.index),
                    y0..y1,
                    x0..x1,
                    req.key.view,
                )
                .ok_or_else(|| "unsupported view plane for this dataset".to_string())?;
                let subset = ArraySubset::new_with_ranges(&ranges);
                let data = retrieve_image_subset_u16(array, &subset, dtype)
                    .map_err(|error| error.to_string())?;
                let data = squeeze_to_2d(data, y_dim, x_dim).ok_or_else(|| {
                    "unexpected array dimensionality for tile (expected displayed axes plus singleton dims)"
                        .to_string()
                })?;
                Ok(DecodedTile {
                    width,
                    height,
                    values: Arc::new(data.iter().copied().collect()),
                })
            });
            let decoded = match decoded {
                Ok(decoded) => decoded,
                Err(error) => {
                    let _ = tx_rsp.send(TileWorkerResponse::Failed {
                        key: req.key,
                        error,
                    });
                    continue 'requests;
                }
            };
            if decoded.width != width || decoded.height != height {
                let _ = tx_rsp.send(TileWorkerResponse::Failed {
                    key: req.key,
                    error: "decoded tile dimensions changed unexpectedly".to_string(),
                });
                continue 'requests;
            }

            for (idx, val) in decoded.values.iter().enumerate() {
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

        // A viewport can disappear while a slow source read is in progress.
        // Re-check ownership before publishing so removing one presentation
        // drops only its stale completion and leaves peer generations alive.
        if let Ok(active) = active_render_ids.lock()
            && !active.is_empty()
            && !active.contains(&req.key.render_id)
        {
            continue;
        }
        if let Ok(active) = active_keys.lock()
            && !active.is_empty()
            && !active.contains(&req.key)
        {
            continue;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::ome::OmeZarrDataset;
    use crate::imaging::view_plane::ViewPlaneMode;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::time::Duration;

    #[test]
    fn tile_cache_deduplicates_requests_and_prunes_stale_work() {
        let mut cache = TileCache::new(2);
        let key = TileKey {
            render_id: 1,
            view: ViewPlaneSelection {
                mode: ViewPlaneMode::Xy,
                slice_level0: 0,
            },
            level: 0,
            tile_y: 0,
            tile_x: 0,
        };
        assert!(cache.mark_in_flight(key));
        assert!(!cache.mark_in_flight(key));
        assert!(cache.is_busy());
        cache.prune_in_flight(&HashSet::new());
        assert!(!cache.is_busy());
        cache.put(key, 7u8);
        assert_eq!(cache.get(&key), Some(&7));
        assert!(!cache.mark_in_flight(key));
    }

    #[test]
    fn tile_worker_composites_real_channels_and_honours_render_generation() {
        let fixture =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
        let (dataset, store) = OmeZarrDataset::open_local(&fixture).expect("open fixture");
        let loader = spawn_tile_loader(store, dataset.levels.clone(), dataset.dims.clone(), 1)
            .expect("spawn tile loader");
        let level = dataset.levels.last().expect("pyramid level");
        let view = ViewPlaneSelection {
            mode: ViewPlaneMode::Xy,
            slice_level0: 0,
        };
        let stale = TileKey {
            render_id: 9,
            view,
            level: level.index,
            tile_y: 0,
            tile_x: 0,
        };
        let current = TileKey {
            render_id: 10,
            ..stale
        };
        loader.set_latest_render_id(10);
        loader.set_active_keys(HashSet::from([current]));
        let channels = vec![
            RenderChannel {
                index: 0,
                color_rgb: [1.0, 0.0, 0.0],
                window: (0.0, dataset.abs_max),
            },
            RenderChannel {
                index: 1,
                color_rgb: [0.0, 1.0, 0.0],
                window: (0.0, dataset.abs_max),
            },
        ];
        loader
            .tx
            .send(TileRequest {
                key: stale,
                channels: channels.clone(),
            })
            .expect("queue stale request");
        loader
            .tx
            .send(TileRequest {
                key: current,
                channels,
            })
            .expect("queue current request");

        let response = loader
            .rx
            .recv_timeout(Duration::from_secs(5))
            .expect("tile completion");
        let TileWorkerResponse::Tile(tile) = response else {
            panic!("expected composited tile");
        };
        assert_eq!(tile.key, current, "stale generation must be discarded");
        assert!(tile.width > 0 && tile.height > 0);
        assert_eq!(tile.rgba.len(), tile.width * tile.height * 4);
        assert!(
            tile.rgba
                .chunks_exact(4)
                .all(|pixel| pixel[2] == 0 && pixel[3] == 255)
        );
        assert!(
            tile.rgba
                .chunks_exact(4)
                .any(|pixel| pixel[0] > 0 || pixel[1] > 0)
        );
    }

    #[test]
    fn tile_worker_accepts_two_live_viewport_generations() {
        let fixture =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
        let (dataset, store) = OmeZarrDataset::open_local(&fixture).expect("open fixture");
        let loader = spawn_tile_loader(store, dataset.levels.clone(), dataset.dims.clone(), 1)
            .expect("spawn tile loader");
        let level = dataset.levels.last().expect("pyramid level");
        let left = TileKey {
            render_id: 21,
            view: ViewPlaneSelection {
                mode: ViewPlaneMode::Xy,
                slice_level0: 0,
            },
            level: level.index,
            tile_y: 0,
            tile_x: 0,
        };
        let right = TileKey {
            render_id: 22,
            ..left
        };
        loader.set_active_render_ids(HashSet::from([left.render_id, right.render_id]));
        loader.set_active_keys(HashSet::from([left, right]));
        loader
            .tx
            .send(TileRequest {
                key: left,
                channels: vec![RenderChannel {
                    index: 0,
                    color_rgb: [1.0, 0.0, 0.0],
                    window: (0.0, dataset.abs_max),
                }],
            })
            .expect("queue left viewport");
        loader
            .tx
            .send(TileRequest {
                key: right,
                channels: vec![RenderChannel {
                    // Same raw channel, different viewport presentation. The
                    // decoder must read it once and compose twice.
                    index: 0,
                    color_rgb: [0.0, 1.0, 0.0],
                    window: (0.0, dataset.abs_max),
                }],
            })
            .expect("queue right viewport");

        let mut responses = HashMap::new();
        for _ in 0..2 {
            let response = loader
                .rx
                .recv_timeout(Duration::from_secs(5))
                .expect("tile completion");
            let TileWorkerResponse::Tile(tile) = response else {
                panic!("expected composited tile");
            };
            responses.insert(tile.key.render_id, tile.rgba);
        }
        assert_eq!(responses.len(), 2);
        assert_ne!(responses[&left.render_id], responses[&right.render_id]);
        let stats = loader.stats();
        assert_eq!(stats.decode_requests, 2);
        assert_eq!(stats.source_reads, 1);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.decoded_cache_entries, 1);
        assert!(stats.decoded_cache_bytes > 0);
    }

    #[test]
    fn decoded_cache_shares_slow_work_and_reuses_failures() {
        use std::sync::Barrier;

        let cache = CpuDecodedTileCache::new(4);
        let key = DecodedTileKey {
            view: ViewPlaneSelection {
                mode: ViewPlaneMode::Xy,
                slice_level0: 0,
            },
            level: 0,
            tile_y: 0,
            tile_x: 0,
            channel: 0,
        };
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let worker_cache = Arc::clone(&cache);
        let worker_entered = Arc::clone(&entered);
        let worker_release = Arc::clone(&release);
        let worker = std::thread::spawn(move || {
            worker_cache.get_or_decode(key, || {
                worker_entered.wait();
                worker_release.wait();
                Ok(DecodedTile {
                    width: 2,
                    height: 1,
                    values: Arc::new(vec![10, 20]),
                })
            })
        });
        entered.wait();
        let peer_cache = Arc::clone(&cache);
        let peer = std::thread::spawn(move || {
            peer_cache.get_or_decode(key, || panic!("peer must reuse in-flight decode"))
        });
        release.wait();
        assert_eq!(worker.join().unwrap().unwrap().values.as_slice(), &[10, 20]);
        assert_eq!(peer.join().unwrap().unwrap().values.as_slice(), &[10, 20]);
        let stats = cache.stats();
        assert_eq!(stats.decode_requests, 2);
        assert_eq!(stats.source_reads, 1);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.decoded_cache_bytes, 4);

        let failed_key = DecodedTileKey { channel: 1, ..key };
        assert_eq!(
            cache
                .get_or_decode(failed_key, || Err("remote read failed".to_string()))
                .unwrap_err(),
            "remote read failed"
        );
        assert_eq!(
            cache
                .get_or_decode(failed_key, || panic!("cached failure must be reused"))
                .unwrap_err(),
            "remote read failed"
        );
        let stats = cache.stats();
        assert_eq!(stats.source_reads, 2);
        assert_eq!(stats.cache_hits, 2);
    }
}
