use std::sync::Arc;

use anyhow::Context;
use crossbeam_channel::{Receiver, Sender};

use crate::data::ome::retrieve_image_subset_u16;
use zarrs::array::{Array, ArraySubset};
use zarrs::storage::ReadableStorageTraits;

#[derive(Debug, Clone)]
pub struct ChannelMaxRequest {
    pub request_id: u64,
    pub level: usize,
    pub channel: u64,
}

#[derive(Debug, Clone)]
pub struct ChannelMaxResponse {
    pub request_id: u64,
    pub channel: u64,
    pub p97: u16,
    pub max: u16,
}

#[derive(Debug)]
pub struct ChannelMaxLoaderHandle {
    pub tx: Sender<ChannelMaxRequest>,
    pub rx: Receiver<ChannelMaxResponse>,
}

pub fn spawn_channel_max_loader(
    store: Arc<dyn ReadableStorageTraits>,
    level_paths: Vec<String>,
    level_shapes: Vec<Vec<u64>>,
    level_chunks: Vec<Vec<u64>>,
    level_dtypes: Vec<String>,
    dims_cyx: (Option<usize>, usize, usize),
) -> anyhow::Result<ChannelMaxLoaderHandle> {
    let (tx_req, rx_req) = crossbeam_channel::unbounded::<ChannelMaxRequest>();
    let (tx_rsp, rx_rsp) = crossbeam_channel::unbounded::<ChannelMaxResponse>();

    std::thread::Builder::new()
        .name("chan-max-loader".to_string())
        .spawn(move || {
            if let Err(err) = channel_max_loader_thread(
                store,
                level_paths,
                level_shapes,
                level_chunks,
                level_dtypes,
                dims_cyx,
                rx_req,
                tx_rsp,
            ) {
                eprintln!("channel max loader thread exited: {err:?}");
            }
        })
        .context("failed to spawn channel max loader thread")?;

    Ok(ChannelMaxLoaderHandle {
        tx: tx_req,
        rx: rx_rsp,
    })
}

fn channel_max_loader_thread(
    store: Arc<dyn ReadableStorageTraits>,
    level_paths: Vec<String>,
    level_shapes: Vec<Vec<u64>>,
    level_chunks: Vec<Vec<u64>>,
    level_dtypes: Vec<String>,
    dims_cyx: (Option<usize>, usize, usize),
    rx_req: Receiver<ChannelMaxRequest>,
    tx_rsp: Sender<ChannelMaxResponse>,
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
        let Some(chunks) = level_chunks.get(req.level) else {
            continue;
        };
        let Some(dtype) = level_dtypes.get(req.level) else {
            continue;
        };

        let (c_dim, y_dim, x_dim) = dims_cyx;
        if y_dim >= shape.len()
            || x_dim >= shape.len()
            || y_dim >= chunks.len()
            || x_dim >= chunks.len()
        {
            continue;
        }

        let y_chunk = chunks[y_dim].max(1);
        let x_chunk = chunks[x_dim].max(1);
        let shape_y = shape[y_dim];
        let shape_x = shape[x_dim];

        let tiles_y = (shape_y + y_chunk - 1) / y_chunk;
        let tiles_x = (shape_x + x_chunk - 1) / x_chunk;

        let mut hist = vec![0u64; 65536];
        let mut n: u64 = 0;
        let mut max_v: u16 = 0;
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let y0 = ty * y_chunk;
                let x0 = tx * x_chunk;
                let y1 = (y0 + y_chunk).min(shape_y);
                let x1 = (x0 + x_chunk).min(shape_x);

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
                    Err(_) => continue,
                };
                for &v in data.iter() {
                    hist[v as usize] = hist[v as usize].saturating_add(1);
                    n = n.saturating_add(1);
                    if v > max_v {
                        max_v = v;
                    }
                }
            }
        }

        let p97 = if n == 0 {
            0
        } else {
            // Nearest-rank percentile (ceil(p*n)).
            let target = (n.saturating_mul(97).saturating_add(99)) / 100;
            let mut acc: u64 = 0;
            let mut out: u16 = 0;
            for (i, c) in hist.iter().enumerate() {
                acc = acc.saturating_add(*c);
                if acc >= target {
                    out = i as u16;
                    break;
                }
            }
            out
        };

        let _ = tx_rsp.send(ChannelMaxResponse {
            request_id: req.request_id,
            channel: req.channel,
            p97,
            max: max_v,
        });
    }

    Ok(())
}
