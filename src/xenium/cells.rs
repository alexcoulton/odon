use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, anyhow};
use zarrs::array::{Array, ArraySubset};
use zarrs::storage::ReadableStorageTraits;

use crate::render::line_bins::LineSegmentsBins;
use crate::xenium::ZipStore;

#[derive(Debug, Clone, Copy)]
pub enum XeniumPolygonSet {
    Cell = 1,
}

pub fn load_cells_outline_bins(
    cells_zarr_zip: &Path,
    polygon_set: XeniumPolygonSet,
    pixel_size_um: f32,
) -> anyhow::Result<Arc<LineSegmentsBins>> {
    let store = ZipStore::open(cells_zarr_zip).context("open cells.zarr.zip")?;
    let store: Arc<dyn ReadableStorageTraits> = store;

    let verts: Array<dyn ReadableStorageTraits> =
        Array::open(store.clone(), "/polygon_vertices").context("open polygon_vertices")?;
    let nverts: Array<dyn ReadableStorageTraits> =
        Array::open(store.clone(), "/polygon_num_vertices").context("open polygon_num_vertices")?;

    // polygon_vertices: [set, cell, 2*max_vertices]
    // polygon_num_vertices: [set, cell]
    let verts_shape = verts.shape().to_vec();
    let nverts_shape = nverts.shape().to_vec();
    if verts_shape.len() != 3 || nverts_shape.len() != 2 {
        return Err(anyhow!(
            "unexpected cells zarr shapes: polygon_vertices={verts_shape:?}, polygon_num_vertices={nverts_shape:?}"
        ));
    }

    let set = polygon_set as u64;
    let cells = verts_shape[1] as u64;
    let floats_per_cell = verts_shape[2] as u64;
    if floats_per_cell < 4 {
        anyhow::bail!("polygon_vertices last dim too small: {floats_per_cell}");
    }

    let verts_subset =
        ArraySubset::new_with_ranges(&[set..(set + 1), 0..cells, 0..floats_per_cell]);
    let nverts_subset = ArraySubset::new_with_ranges(&[set..(set + 1), 0..cells]);

    let verts_data: ndarray::ArrayD<f32> = verts
        .retrieve_array_subset(&verts_subset)
        .context("read polygon_vertices subset")?;
    let nverts_data: ndarray::ArrayD<i32> = nverts
        .retrieve_array_subset(&nverts_subset)
        .context("read polygon_num_vertices subset")?;

    let verts_data = verts_data
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|_| anyhow!("polygon_vertices dimensionality mismatch"))?;
    let nverts_data = nverts_data
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|_| anyhow!("polygon_num_vertices dimensionality mismatch"))?;

    let inv_px = 1.0 / pixel_size_um.max(1e-6);

    let mut polylines: Vec<Vec<eframe::egui::Pos2>> = Vec::with_capacity(cells as usize);
    for cell_i in 0..(cells as usize) {
        let nv = nverts_data[(0, cell_i)].max(0) as usize;
        if nv < 3 {
            continue;
        }
        // Stored as interleaved x,y in microns.
        let need = (nv * 2).min(verts_data.shape()[2]);
        if need < 6 {
            continue;
        }
        let mut pts: Vec<eframe::egui::Pos2> = Vec::with_capacity(nv + 1);
        for vi in 0..nv {
            let xi = vi * 2;
            let yi = vi * 2 + 1;
            if yi >= need {
                break;
            }
            let x_um = verts_data[(0, cell_i, xi)];
            let y_um = verts_data[(0, cell_i, yi)];
            if !x_um.is_finite() || !y_um.is_finite() {
                continue;
            }
            pts.push(eframe::egui::pos2(x_um * inv_px, y_um * inv_px));
        }
        if pts.len() < 3 {
            continue;
        }
        // Close ring.
        if let Some(first) = pts.first().copied() {
            pts.push(first);
        }
        polylines.push(pts);
    }

    let Some(bins) = LineSegmentsBins::build_from_polylines(&polylines, 2048.0) else {
        anyhow::bail!("no valid segments after parsing cells polygons");
    };
    Ok(Arc::new(bins))
}
