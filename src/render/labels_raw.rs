use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::Arc;

use anyhow::Context;
use crossbeam_channel::{Receiver, Sender};
use lru::LruCache;

use crate::data::ome::{Dims, LevelInfo};
use crate::imaging::plane_selection::{image_subset_ranges, plane_selection_for_z};
use zarrs::array::{Array, ArraySubset};
use zarrs::storage::ReadableStorageTraits;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct LabelTileKey {
    pub z_level0: u64,
    pub level: usize,
    pub tile_y: u64,
    pub tile_x: u64,
}

#[derive(Debug, Clone)]
pub struct LabelTileRequest {
    pub key: LabelTileKey,
}

#[derive(Debug, Clone)]
pub struct LabelTileResponse {
    pub key: LabelTileKey,
    pub tex_width: usize,
    pub tex_height: usize,
    pub data_u32: Vec<u32>,
}

#[derive(Debug)]
pub struct LabelTileLoaderHandle {
    pub tx: Sender<LabelTileRequest>,
    pub rx: Receiver<LabelTileResponse>,
}

pub struct LabelTileCache<T> {
    cache: LruCache<LabelTileKey, T>,
    in_flight: HashSet<LabelTileKey>,
}

impl<T> LabelTileCache<T> {
    pub fn new(capacity_tiles: usize) -> Self {
        Self {
            cache: LruCache::new(NonZeroUsize::new(capacity_tiles.max(1)).unwrap()),
            in_flight: HashSet::new(),
        }
    }

    pub fn get_mut(&mut self, key: &LabelTileKey) -> Option<&mut T> {
        self.cache.get_mut(key)
    }

    pub fn push(&mut self, key: LabelTileKey, value: T) -> Option<(LabelTileKey, T)> {
        self.in_flight.remove(&key);
        self.cache.push(key, value)
    }

    pub fn mark_in_flight(&mut self, key: LabelTileKey) -> bool {
        if self.cache.contains(&key) || self.in_flight.contains(&key) {
            return false;
        }
        self.in_flight.insert(key);
        true
    }

    pub fn drain(&mut self) -> Vec<(LabelTileKey, T)> {
        self.in_flight.clear();
        let mut out = Vec::with_capacity(self.cache.len());
        while let Some((k, v)) = self.cache.pop_lru() {
            out.push((k, v));
        }
        out
    }

    pub fn prune_in_flight(&mut self, keep: &HashSet<LabelTileKey>) {
        self.in_flight.retain(|k| keep.contains(k));
    }

    pub fn is_busy(&self) -> bool {
        !self.in_flight.is_empty()
    }
}

pub fn spawn_label_tile_loader(
    store: Arc<dyn ReadableStorageTraits>,
    levels: Vec<LevelInfo>,
    dims: Dims,
) -> anyhow::Result<LabelTileLoaderHandle> {
    let (tx_req, rx_req) = crossbeam_channel::unbounded::<LabelTileRequest>();
    let (tx_rsp, rx_rsp) = crossbeam_channel::unbounded::<LabelTileResponse>();

    std::thread::Builder::new()
        .name("label-tile-loader".to_string())
        .spawn(move || {
            if let Err(err) = label_tile_loader_thread(store, levels, dims, rx_req, tx_rsp) {
                eprintln!("label tile loader thread exited: {err:?}");
            }
        })
        .context("failed to spawn label tile loader thread")?;

    Ok(LabelTileLoaderHandle {
        tx: tx_req,
        rx: rx_rsp,
    })
}

fn label_tile_loader_thread(
    store: Arc<dyn ReadableStorageTraits>,
    levels: Vec<LevelInfo>,
    dims: Dims,
    rx_req: Receiver<LabelTileRequest>,
    tx_rsp: Sender<LabelTileResponse>,
) -> anyhow::Result<()> {
    let mut arrays: Vec<Array<dyn ReadableStorageTraits>> = Vec::with_capacity(levels.len());
    for info in &levels {
        let path = &info.path;
        let zarr_path = format!("/{}", path.trim_start_matches('/'));
        arrays.push(Array::open(store.clone(), &zarr_path)?);
    }
    let Some(level0) = levels.first() else {
        return Ok(());
    };
    let y_dim = dims.y;
    let x_dim = dims.x;

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
        if y_dim >= shape.len() || x_dim >= shape.len() {
            continue;
        }

        let y_chunk = chunks[y_dim];
        let x_chunk = chunks[x_dim];

        let y0 = req.key.tile_y * y_chunk;
        let x0 = req.key.tile_x * x_chunk;
        let y1 = (y0 + y_chunk).min(shape[y_dim]);
        let x1 = (x0 + x_chunk).min(shape[x_dim]);

        let height = (y1 - y0) as usize;
        let width = (x1 - x0) as usize;
        if width == 0 || height == 0 {
            continue;
        }

        let y0_halo = y0 as i64 - 1;
        let x0_halo = x0 as i64 - 1;
        let read_y0 = (y0_halo.max(0) as u64).min(shape[y_dim]);
        let read_x0 = (x0_halo.max(0) as u64).min(shape[x_dim]);
        let read_y1 = y1.saturating_add(1).min(shape[y_dim]).max(read_y0);
        let read_x1 = x1.saturating_add(1).min(shape[x_dim]).max(read_x0);

        let read_h = (read_y1 - read_y0) as usize;
        let read_w = (read_x1 - read_x0) as usize;
        if read_w == 0 || read_h == 0 {
            continue;
        }

        let plane = plane_selection_for_z(&dims, level0, req.key.z_level0);
        let ranges = image_subset_ranges(
            &dims,
            level0,
            level_info,
            None,
            read_y0..read_y1,
            read_x0..read_x1,
            plane,
        );

        let subset = ArraySubset::new_with_ranges(&ranges);
        let data = retrieve_label_subset_u32(array, &subset, dtype)?;
        let read = squeeze_to_yx(data, y_dim, x_dim).context(
            "unexpected dimensionality for label tile (expected y/x plus singleton dims)",
        )?;

        if read.shape()[0] != read_h || read.shape()[1] != read_w {
            continue;
        }

        let tex_width = width + 2;
        let tex_height = height + 2;
        let mut halo = vec![0u32; tex_width * tex_height];

        let offset_y = (read_y0 as i64 - y0_halo) as usize;
        let offset_x = (read_x0 as i64 - x0_halo) as usize;

        // Place read block into halo buffer.
        for ry in 0..read_h {
            let dst_y = ry + offset_y;
            let dst_row = dst_y * tex_width;
            for rx in 0..read_w {
                halo[dst_row + (rx + offset_x)] = read[(ry, rx)];
            }
        }

        // Replicate edges if read block does not cover full halo (dataset boundary).
        let start_y = offset_y;
        let end_y = offset_y + read_h;
        let start_x = offset_x;
        let end_x = offset_x + read_w;

        // Top/bottom rows.
        if start_y > 0 {
            let src = halo[start_y * tex_width..(start_y + 1) * tex_width].to_vec();
            for y in 0..start_y {
                halo[y * tex_width..(y + 1) * tex_width].copy_from_slice(&src);
            }
        }
        if end_y < tex_height {
            let src_y = end_y.saturating_sub(1).min(tex_height.saturating_sub(1));
            let src = halo[src_y * tex_width..(src_y + 1) * tex_width].to_vec();
            for y in end_y..tex_height {
                halo[y * tex_width..(y + 1) * tex_width].copy_from_slice(&src);
            }
        }

        // Left/right columns.
        if start_x > 0 || end_x < tex_width {
            let src_left = start_x.min(tex_width.saturating_sub(1));
            let src_right = end_x.saturating_sub(1).min(tex_width.saturating_sub(1));
            for y in 0..tex_height {
                let row = y * tex_width;
                if start_x > 0 {
                    let v = halo[row + src_left];
                    for x in 0..start_x {
                        halo[row + x] = v;
                    }
                }
                if end_x < tex_width {
                    let v = halo[row + src_right];
                    for x in end_x..tex_width {
                        halo[row + x] = v;
                    }
                }
            }
        }

        let _ = tx_rsp.send(LabelTileResponse {
            key: req.key,
            tex_width,
            tex_height,
            data_u32: halo,
        });
    }

    Ok(())
}

fn retrieve_label_subset_u32(
    array: &Array<dyn ReadableStorageTraits>,
    subset: &ArraySubset,
    dtype: &str,
) -> anyhow::Result<ndarray::ArrayD<u32>> {
    for part in dtype.split('/').map(|s| s.trim().to_ascii_lowercase()) {
        match part.as_str() {
            "uint8" | "|u1" | "u1" => {
                let data: ndarray::ArrayD<u8> = array.retrieve_array_subset(subset)?;
                return Ok(data.mapv(u32::from));
            }
            "uint16" | "<u2" | ">u2" | "u2" => {
                let data: ndarray::ArrayD<u16> = array.retrieve_array_subset(subset)?;
                return Ok(data.mapv(u32::from));
            }
            "uint32" | "<u4" | ">u4" | "u4" => {
                let data: ndarray::ArrayD<u32> = array.retrieve_array_subset(subset)?;
                return Ok(data);
            }
            _ => {}
        }
    }
    anyhow::bail!("unsupported label dtype: {dtype}")
}

fn squeeze_to_yx(
    data: ndarray::ArrayD<u32>,
    y_dim_orig: usize,
    x_dim_orig: usize,
) -> Option<ndarray::Array2<u32>> {
    crate::render::array_dims::squeeze_to_yx(data, y_dim_orig, x_dim_orig)
}
