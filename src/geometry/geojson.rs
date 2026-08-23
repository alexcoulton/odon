use std::path::Path;

use eframe::egui;

pub use odon::data::segmentation_geojson::PolygonRingMode;

pub fn load_geojson_polylines_world(
    path: &Path,
    downsample_factor: f32,
    ring_mode: PolygonRingMode,
) -> anyhow::Result<Vec<Vec<egui::Pos2>>> {
    Ok(
        odon::data::segmentation_geojson::load_geojson_polyline_coordinates_world(
            path,
            downsample_factor,
            ring_mode,
        )?
        .into_iter()
        .map(|line| {
            line.into_iter()
                .map(|point| egui::pos2(point[0], point[1]))
                .collect()
        })
        .collect(),
    )
}
