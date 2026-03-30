use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, anyhow};
use serde_json::Value;
use zarrs::array::{Array, ArraySubset};
use zarrs::storage::ReadableStorageTraits;

use crate::xenium::ZipStore;

#[derive(Debug, Clone, Default)]
pub struct XeniumTranscriptsMeta {
    pub gene_names: Vec<String>,
    pub gene_indices: HashMap<String, u16>,
    pub grid_keys_lvl0: Vec<String>,
}

fn parse_json_stream_last_object(bytes: &[u8]) -> anyhow::Result<serde_json::Map<String, Value>> {
    let mut de = serde_json::Deserializer::from_slice(bytes).into_iter::<Value>();
    let mut last: Option<serde_json::Map<String, Value>> = None;
    while let Some(v) = de.next() {
        let v = v?;
        if let Value::Object(map) = v {
            last = Some(map);
        }
    }
    last.ok_or_else(|| anyhow!("no JSON object found"))
}

pub fn load_transcripts_meta(transcripts_zarr_zip: &Path) -> anyhow::Result<XeniumTranscriptsMeta> {
    let store = ZipStore::open(transcripts_zarr_zip).context("open transcripts.zarr.zip")?;
    let store: Arc<dyn ReadableStorageTraits> = store;

    let attrs = crate::data::zarr_attrs::read_node_attributes_store(store.as_ref(), "")?
        .ok_or_else(|| anyhow!("missing root .zattrs in transcripts.zarr.zip"))?;
    let attrs_v = Value::Object(attrs);

    let gene_names = attrs_v
        .get("gene_names")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let mut gene_indices: HashMap<String, u16> = HashMap::new();
    if let Some(obj) = attrs_v.get("gene_indices").and_then(|v| v.as_object()) {
        for (k, v) in obj {
            if let Some(i) = v.as_u64() {
                if i <= u16::MAX as u64 {
                    gene_indices.insert(k.to_ascii_uppercase(), i as u16);
                }
            }
        }
    }

    // grids/.zattrs is sometimes multiple JSON objects concatenated; take the last.
    let grids_attrs_bytes = store
        .get(&zarrs::storage::StoreKey::new("grids/.zattrs")?)
        .map_err(|e| anyhow!("read grids/.zattrs failed: {e}"))?
        .ok_or_else(|| anyhow!("missing grids/.zattrs"))?;
    let grids_map = parse_json_stream_last_object(grids_attrs_bytes.as_ref())?;

    let mut grid_keys_lvl0: Vec<String> = Vec::new();
    if let Some(keys_levels) = grids_map.get("grid_keys").and_then(|v| v.as_array()) {
        if let Some(keys0) = keys_levels.first().and_then(|v| v.as_array()) {
            for k in keys0 {
                if let Some(s) = k.as_str() {
                    grid_keys_lvl0.push(s.to_string());
                }
            }
        }
    }

    Ok(XeniumTranscriptsMeta {
        gene_names,
        gene_indices,
        grid_keys_lvl0,
    })
}

#[derive(Debug, Clone)]
pub struct XeniumTranscriptsAllPayload {
    pub positions_by_gene: Vec<Vec<eframe::egui::Pos2>>,
    pub qv_by_gene: Vec<Vec<f32>>,
    /// Transcript ids, if present. If missing, the vector will be empty for that gene.
    pub id_by_gene: Vec<Vec<u64>>,
    pub total_points: usize,
}

pub fn load_transcripts_all_points(
    transcripts_zarr_zip: &Path,
    meta: &XeniumTranscriptsMeta,
    pixel_size_um: f32,
    max_points_total: usize,
) -> anyhow::Result<XeniumTranscriptsAllPayload> {
    let store = ZipStore::open(transcripts_zarr_zip).context("open transcripts.zarr.zip")?;
    let store: Arc<dyn ReadableStorageTraits> = store;

    let genes_n = meta.gene_names.len().max(1);
    let mut positions_by_gene: Vec<Vec<eframe::egui::Pos2>> =
        (0..genes_n).map(|_| Vec::new()).collect();
    let mut qv_by_gene: Vec<Vec<f32>> = (0..genes_n).map(|_| Vec::new()).collect();
    let mut id_by_gene: Vec<Vec<u64>> = (0..genes_n).map(|_| Vec::new()).collect();
    let mut any_ids = false;

    let inv_px = 1.0 / pixel_size_um.max(1e-6);
    let mut total = 0usize;

    'grids: for key in &meta.grid_keys_lvl0 {
        if max_points_total > 0 && total >= max_points_total {
            break;
        }

        let base = format!("/grids/0/{key}");
        let loc_arr: Array<dyn ReadableStorageTraits> =
            match Array::open(store.clone(), &(base.clone() + "/location")) {
                Ok(a) => a,
                Err(_) => continue,
            };
        let gene_arr: Array<dyn ReadableStorageTraits> =
            match Array::open(store.clone(), &(base.clone() + "/gene_identity")) {
                Ok(a) => a,
                Err(_) => continue,
            };
        let qv_arr: Option<Array<dyn ReadableStorageTraits>> =
            Array::open(store.clone(), &(base.clone() + "/quality_score")).ok();
        let id_arr: Option<Array<dyn ReadableStorageTraits>> =
            Array::open(store.clone(), &(base.clone() + "/id")).ok();

        let loc_shape = loc_arr.shape().to_vec();
        if loc_shape.len() != 2 || loc_shape.get(1).copied().unwrap_or(0) < 2 {
            continue;
        }
        let n = loc_shape[0];
        if n == 0 {
            continue;
        }

        let subset_all = ArraySubset::new_with_ranges(&[0..n, 0..loc_shape[1]]);
        let loc: ndarray::ArrayD<f32> = loc_arr.retrieve_array_subset(&subset_all)?;
        let loc = loc
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| anyhow!("location dimensionality"))?;

        let gshape = gene_arr.shape().to_vec();
        let gsubset = if gshape.len() == 2 {
            ArraySubset::new_with_ranges(&[0..n, 0..1])
        } else {
            ArraySubset::new_with_ranges(&[0..n])
        };
        let gids: ndarray::ArrayD<u16> = gene_arr.retrieve_array_subset(&gsubset)?;
        let gids: Vec<u16> = gids.iter().copied().collect();

        let qvs: Option<Vec<f32>> = if let Some(qv_arr) = qv_arr {
            let qshape = qv_arr.shape().to_vec();
            let qsubset = if qshape.len() == 2 {
                ArraySubset::new_with_ranges(&[0..n, 0..1])
            } else {
                ArraySubset::new_with_ranges(&[0..n])
            };
            match (|| -> anyhow::Result<ndarray::ArrayD<f32>> {
                Ok(qv_arr.retrieve_array_subset(&qsubset)?)
            })() {
                Ok(q) => Some(q.iter().copied().collect()),
                Err(_) => None,
            }
        } else {
            None
        };

        let ids: Option<Vec<u64>> = if let Some(id_arr) = id_arr {
            let ishape = id_arr.shape().to_vec();
            let isubset = if ishape.len() == 2 {
                ArraySubset::new_with_ranges(&[0..n, 0..1])
            } else {
                ArraySubset::new_with_ranges(&[0..n])
            };
            // id appears to be an integer-like array, but can vary.
            if let Ok(ids_u64) = (|| -> anyhow::Result<ndarray::ArrayD<u64>> {
                Ok(id_arr.retrieve_array_subset(&isubset)?)
            })() {
                Some(ids_u64.iter().copied().collect())
            } else if let Ok(ids_u32) = (|| -> anyhow::Result<ndarray::ArrayD<u32>> {
                Ok(id_arr.retrieve_array_subset(&isubset)?)
            })() {
                Some(ids_u32.iter().map(|&v| v as u64).collect())
            } else if let Ok(ids_u16) = (|| -> anyhow::Result<ndarray::ArrayD<u16>> {
                Ok(id_arr.retrieve_array_subset(&isubset)?)
            })() {
                Some(ids_u16.iter().map(|&v| v as u64).collect())
            } else {
                None
            }
        } else {
            None
        };

        any_ids |= ids.is_some();

        for i in 0..(n as usize) {
            if max_points_total > 0 && total >= max_points_total {
                break 'grids;
            }
            let gid = *gids.get(i).unwrap_or(&u16::MAX) as usize;
            if gid >= positions_by_gene.len() {
                continue;
            }
            let x_um = loc[(i, 0)];
            let y_um = loc[(i, 1)];
            if !x_um.is_finite() || !y_um.is_finite() {
                continue;
            }
            let qv = qvs.as_ref().and_then(|v| v.get(i)).copied().unwrap_or(0.0);
            positions_by_gene[gid].push(eframe::egui::pos2(x_um * inv_px, y_um * inv_px));
            qv_by_gene[gid].push(qv);
            let tid = ids
                .as_ref()
                .and_then(|v| v.get(i))
                .copied()
                .unwrap_or(u64::MAX);
            id_by_gene[gid].push(tid);
            total += 1;
        }
    }

    if !any_ids {
        // Drop ids to save memory if none were present anywhere.
        id_by_gene = (0..genes_n).map(|_| Vec::new()).collect();
    }

    Ok(XeniumTranscriptsAllPayload {
        positions_by_gene,
        qv_by_gene,
        id_by_gene,
        total_points: total,
    })
}
