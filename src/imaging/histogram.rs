use std::sync::Arc;

use anyhow::Context;
use crossbeam_channel::{Receiver, Sender};

use crate::data::ome::{Dims, LevelInfo, retrieve_image_subset_u16};
use crate::imaging::view_plane::{ViewPlaneSelection, display_axes, image_subset_ranges_for_view};
use crate::render::array_dims::squeeze_to_2d;
use zarrs::array::{Array, ArraySubset};
use zarrs::storage::ReadableStorageTraits;

#[derive(Debug, Clone)]
pub struct HistogramRequest {
    pub request_id: u64,
    pub view: ViewPlaneSelection,
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
    levels: Vec<LevelInfo>,
    dims: Dims,
) -> anyhow::Result<HistogramLoaderHandle> {
    let (tx_req, rx_req) = crossbeam_channel::unbounded::<HistogramRequest>();
    let (tx_rsp, rx_rsp) = crossbeam_channel::unbounded::<HistogramResponse>();

    std::thread::Builder::new()
        .name("hist-loader".to_string())
        .spawn(move || {
            if let Err(err) = histogram_loader_thread(store, levels, dims, rx_req, tx_rsp) {
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
    levels: Vec<LevelInfo>,
    dims: Dims,
    rx_req: Receiver<HistogramRequest>,
    tx_rsp: Sender<HistogramResponse>,
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

    for req in rx_req.iter() {
        let Some(array) = arrays.get(req.level) else {
            continue;
        };
        let Some(level_info) = levels.get(req.level) else {
            continue;
        };
        let shape = &level_info.shape;
        let dtype = &level_info.dtype;

        let bins = req.bins.clamp(8, 4096);
        let abs_max = if req.abs_max.is_finite() && req.abs_max > 0.0 {
            req.abs_max
        } else {
            1.0
        };

        let Some(display_axes) = display_axes(&dims, req.view.mode) else {
            continue;
        };
        let y_dim = display_axes.vertical;
        let x_dim = display_axes.horizontal;
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

        let Some(ranges) = image_subset_ranges_for_view(
            &dims,
            level0,
            level_info,
            Some(req.channel),
            y0..y1,
            x0..x1,
            req.view,
        ) else {
            continue;
        };
        let subset = ArraySubset::new_with_ranges(&ranges);

        let data = match retrieve_image_subset_u16(array, &subset, dtype) {
            Ok(v) => v,
            Err(_) => {
                continue;
            }
        };

        let plane: ndarray::Array2<u16> = squeeze_to_2d(data, y_dim, x_dim).context(
            "unexpected array dimensionality for histogram (expected displayed axes plus singleton dims)",
        )?;

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
