use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Context;
use eframe::egui;

use crate::data::project_config::ProjectMaskLayer;

#[derive(Debug, Clone)]
pub struct MaskLayer {
    pub id: u64,
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub width_screen_px: f32,
    pub color_rgb: [u8; 3],
    pub offset_world: egui::Vec2,
    pub editable: bool,
    pub polygons_world: Vec<Vec<egui::Pos2>>,
    pub source_geojson: Option<PathBuf>,
}

impl MaskLayer {
    pub fn clear(&mut self) {
        self.polygons_world.clear();
    }

    pub fn add_closed_polygon(&mut self, mut vertices_world: Vec<egui::Pos2>) {
        if vertices_world.len() < 3 {
            return;
        }
        if vertices_world.first() != vertices_world.last() {
            if let Some(first) = vertices_world.first().copied() {
                vertices_world.push(first);
            }
        }
        self.polygons_world.push(vertices_world);
    }

    pub fn to_project(&self) -> ProjectMaskLayer {
        ProjectMaskLayer {
            id: self.id,
            name: self.name.clone(),
            visible: self.visible,
            opacity: self.opacity,
            width_screen_px: self.width_screen_px,
            color_rgb: self.color_rgb,
            offset_world: [self.offset_world.x, self.offset_world.y],
            editable: self.editable,
            polygons_world: self
                .polygons_world
                .iter()
                .map(|poly| poly.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>())
                .collect(),
            source_geojson: self.source_geojson.clone(),
        }
    }

    pub fn from_project(p: &ProjectMaskLayer) -> Self {
        Self {
            id: p.id,
            name: p.name.clone(),
            visible: p.visible,
            opacity: if p.opacity <= 0.0 { 0.9 } else { p.opacity },
            width_screen_px: if p.width_screen_px <= 0.0 {
                2.0
            } else {
                p.width_screen_px
            },
            color_rgb: if p.color_rgb == [0, 0, 0] {
                [255, 210, 60]
            } else {
                p.color_rgb
            },
            offset_world: egui::vec2(p.offset_world[0], p.offset_world[1]),
            editable: p.editable,
            polygons_world: p
                .polygons_world
                .iter()
                .map(|poly| poly.iter().map(|xy| egui::pos2(xy[0], xy[1])).collect())
                .collect(),
            source_geojson: p.source_geojson.clone(),
        }
    }
}

pub fn export_mask_layers_geojson_value(layers: &[MaskLayer]) -> serde_json::Value {
    let features = layers
        .iter()
        .flat_map(|layer| {
            layer
                .polygons_world
                .iter()
                .enumerate()
                .filter_map(move |(shape_index, poly)| {
                    if poly.len() < 3 {
                        return None;
                    }
                    let mut ring: Vec<Vec<f64>> = poly
                        .iter()
                        .map(|p| {
                            vec![
                                (p.x + layer.offset_world.x) as f64,
                                (p.y + layer.offset_world.y) as f64,
                            ]
                        })
                        .collect();
                    if ring.first() != ring.last() {
                        if let Some(first) = ring.first().cloned() {
                            ring.push(first);
                        }
                    }
                    Some(serde_json::json!({
                        "type": "Feature",
                        "geometry": { "type": "Polygon", "coordinates": [ ring ] },
                        "properties": {
                            "layer_id": layer.id,
                            "layer_name": layer.name,
                            "layer_color_rgb": layer.color_rgb,
                            "layer_opacity": layer.opacity,
                            "layer_width_screen_px": layer.width_screen_px,
                            "layer_visible": layer.visible,
                            "layer_editable": layer.editable,
                            "shape_index": shape_index as i64,
                        }
                    }))
                })
        })
        .collect::<Vec<_>>();

    serde_json::json!({
        "type": "FeatureCollection",
        "odon_masks_version": 1,
        "features": features,
    })
}

pub fn save_mask_layers_geojson(path: &Path, layers: &[MaskLayer]) -> anyhow::Result<()> {
    if layers.iter().all(|l| l.polygons_world.is_empty()) {
        anyhow::bail!("no mask shapes to save");
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.to_string_lossy()))?;
    }
    let root = export_mask_layers_geojson_value(layers);
    let text = serde_json::to_string_pretty(&root).context("failed to encode GeoJSON")?;
    fs::write(path, text).with_context(|| format!("failed to write {}", path.to_string_lossy()))?;
    Ok(())
}
