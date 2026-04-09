use std::sync::Arc;

use anyhow::Context;
use crossbeam_channel::{Receiver, Sender};

use crate::data::ome::retrieve_image_subset_u16;
use crate::render::array_dims::squeeze_to_yx;
use zarrs::array::{Array, ArraySubset};
use zarrs::storage::ReadableStorageTraits;

#[derive(Debug, Clone)]
pub struct HistogramRequest {
    pub request_id: u64,
    pub level: usize,
    pub channel: u64,
    pub y0: u64,
    pub y1: u64,
    pub x0: u64,
    pub x1: u64,
    pub bins: usize,
    pub abs_max: f32,
}

#[derive(Debug, Clone)]
pub struct HistogramStats {
    pub min: f32,
    pub q1: f32,
    pub median: f32,
    pub q3: f32,
    pub max: f32,
    pub n: usize,
}

#[derive(Debug, Clone)]
pub struct HistogramResponse {
    pub request_id: u64,
    pub bins: Vec<u32>,
    pub stats: Option<HistogramStats>,
}

#[derive(Debug)]
pub struct HistogramLoaderHandle {
    pub tx: Sender<HistogramRequest>,
    pub rx: Receiver<HistogramResponse>,
}

pub fn spawn_histogram_loader(
    store: Arc<dyn ReadableStorageTraits>,
    level_paths: Vec<String>,
    level_shapes: Vec<Vec<u64>>,
    level_dtypes: Vec<String>,
    dims_cyx: (Option<usize>, usize, usize),
) -> anyhow::Result<HistogramLoaderHandle> {
    let (tx_req, rx_req) = crossbeam_channel::unbounded::<HistogramRequest>();
    let (tx_rsp, rx_rsp) = crossbeam_channel::unbounded::<HistogramResponse>();

    std::thread::Builder::new()
        .name("hist-loader".to_string())
        .spawn(move || {
            if let Err(err) = histogram_loader_thread(
                store,
                level_paths,
                level_shapes,
                level_dtypes,
                dims_cyx,
                rx_req,
                tx_rsp,
            ) {
                eprintln!("hist loader thread exited: {err:?}");
            }
        })
        .context("failed to spawn histogram loader thread")?;

    Ok(HistogramLoaderHandle {
        tx: tx_req,
        rx: rx_rsp,
    })
}

fn histogram_loader_thread(
    store: Arc<dyn ReadableStorageTraits>,
    level_paths: Vec<String>,
    level_shapes: Vec<Vec<u64>>,
    level_dtypes: Vec<String>,
    dims_cyx: (Option<usize>, usize, usize),
    rx_req: Receiver<HistogramRequest>,
    tx_rsp: Sender<HistogramResponse>,
) -> anyhow::Result<()> {
    let mut arrays: Vec<Array<dyn ReadableStorageTraits>> = Vec::with_capacity(level_paths.len());
    for path in &level_paths {
        let zarr_path = format!("/{}", path.trim_start_matches('/'));
        arrays.push(Array::open(store.clone(), &zarr_path)?);
    }

    for req in rx_req.iter() {
        let Some(array) = arrays.get(req.level) else {
            continue;
        };
        let Some(shape) = level_shapes.get(req.level) else {
            continue;
        };
        let Some(dtype) = level_dtypes.get(req.level) else {
            continue;
        };

        let bins = req.bins.clamp(8, 4096);
        let abs_max = if req.abs_max.is_finite() && req.abs_max > 0.0 {
            req.abs_max
        } else {
            1.0
        };

        let (c_dim, y_dim, x_dim) = dims_cyx;
        if y_dim >= shape.len() || x_dim >= shape.len() {
            continue;
        }

        let y0 = req.y0.min(shape[y_dim]);
        let y1 = req.y1.min(shape[y_dim]).max(y0);
        let x0 = req.x0.min(shape[x_dim]);
        let x1 = req.x1.min(shape[x_dim]).max(x0);
        if y1 <= y0 || x1 <= x0 {
            let _ = tx_rsp.send(HistogramResponse {
                request_id: req.request_id,
                bins: vec![0u32; bins],
                stats: None,
            });
            continue;
        }

        let mut ranges: Vec<std::ops::Range<u64>> = Vec::with_capacity(shape.len());
        for dim in 0..shape.len() {
            if Some(dim) == c_dim {
                ranges.push(req.channel..(req.channel + 1));
            } else if dim == y_dim {
                ranges.push(y0..y1);
            } else if dim == x_dim {
                ranges.push(x0..x1);
            } else {
                ranges.push(0..1);
            }
        }
        let subset = ArraySubset::new_with_ranges(&ranges);

        let data = match retrieve_image_subset_u16(array, &subset, dtype) {
            Ok(v) => v,
            Err(_) => {
                continue;
            }
        };

        let plane: ndarray::Array2<u16> = squeeze_to_yx(data, y_dim, x_dim)
            .context("unexpected array dimensionality for histogram (expected y/x plus singleton dims)")?;

        let mut values: Vec<u16> = plane.iter().copied().collect();
        values.retain(|v| (*v as f32).is_finite());

        let stats = compute_stats_u16(&values);
        let hist = compute_hist_u16(&values, bins, abs_max);

        let _ = tx_rsp.send(HistogramResponse {
            request_id: req.request_id,
            bins: hist,
            stats,
        });
    }

    Ok(())
}

fn compute_hist_u16(values: &[u16], bins: usize, abs_max: f32) -> Vec<u32> {
    let bins = bins.max(8);
    let mut out = vec![0u32; bins];
    if values.is_empty() {
        return out;
    }
    let inv = (bins as f32 - 1.0) / abs_max.max(1.0);
    for &v in values {
        let vf = (v as f32).clamp(0.0, abs_max);
        let idx = (vf * inv).floor() as usize;
        let idx = idx.min(bins - 1);
        out[idx] = out[idx].saturating_add(1);
    }
    out
}

fn compute_stats_u16(values: &Vec<u16>) -> Option<HistogramStats> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.clone();
    sorted.sort_unstable();
    let n = sorted.len();
    let min = sorted[0] as f32;
    let max = sorted[n - 1] as f32;

    let q1 = sorted[(n * 25) / 100] as f32;
    let median = sorted[(n * 50) / 100] as f32;
    let q3 = sorted[(n * 75) / 100] as f32;

    Some(HistogramStats {
        min,
        q1,
        median,
        q3,
        max,
        n,
    })
}
