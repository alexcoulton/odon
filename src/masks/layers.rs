use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use eframe::egui;

use crate::data::project_config::ProjectMaskLayer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskDisplayMode {
    OutlineOnly,
    TranslucentFill,
    FilledPreview,
}

impl MaskDisplayMode {
    pub fn storage_key(self) -> &'static str {
        match self {
            Self::OutlineOnly => "outline_only",
            Self::TranslucentFill => "translucent_fill",
            Self::FilledPreview => "filled_preview",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::OutlineOnly => "Outline",
            Self::TranslucentFill => "Translucent fill",
            Self::FilledPreview => "Mask preview",
        }
    }

    pub fn from_storage_key(value: &str) -> Option<Self> {
        match value {
            "outline_only" | "outline" => Some(Self::OutlineOnly),
            "translucent_fill" | "fill_outline" | "semi_transparent_fill" => {
                Some(Self::TranslucentFill)
            }
            "filled_preview" | "mask_preview" => Some(Self::FilledPreview),
            _ => None,
        }
    }

    pub fn default_new_layer() -> Self {
        Self::TranslucentFill
    }

    pub fn default_legacy_project() -> Self {
        Self::OutlineOnly
    }
}

#[derive(Debug, Clone)]
pub struct MaskRasterDisplayCache {
    pub generation: u64,
    pub width: usize,
    pub height: usize,
    pub values: Arc<Vec<u16>>,
    pub corners_world: [egui::Pos2; 4],
}

#[derive(Debug, Clone)]
pub struct MaskLayer {
    pub id: u64,
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub width_screen_px: f32,
    pub display_mode: MaskDisplayMode,
    pub color_rgb: [u8; 3],
    pub offset_world: egui::Vec2,
    pub editable: bool,
    pub polygons_world: Vec<Vec<egui::Pos2>>,
    pub raster_display: Option<MaskRasterDisplayCache>,
    pub source_geojson: Option<PathBuf>,
}

impl MaskLayer {
    pub fn clear(&mut self) {
        self.polygons_world.clear();
        self.raster_display = None;
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
        self.raster_display = None;
    }

    pub fn to_project(&self) -> ProjectMaskLayer {
        ProjectMaskLayer {
            id: self.id,
            name: self.name.clone(),
            visible: self.visible,
            opacity: self.opacity,
            width_screen_px: self.width_screen_px,
            display_mode: Some(self.display_mode.storage_key().to_string()),
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
            display_mode: p
                .display_mode
                .as_deref()
                .and_then(MaskDisplayMode::from_storage_key)
                .unwrap_or_else(MaskDisplayMode::default_legacy_project),
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
            raster_display: None,
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
                            "layer_display_mode": layer.display_mode.storage_key(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::geojson::{PolygonRingMode, load_geojson_polylines_world};

    fn layer(id: u64, offset: egui::Vec2) -> MaskLayer {
        MaskLayer {
            id,
            name: format!("Mask {id}"),
            visible: id % 2 == 1,
            opacity: 0.4,
            width_screen_px: 3.0,
            display_mode: MaskDisplayMode::TranslucentFill,
            color_rgb: [10, 20, 30],
            offset_world: offset,
            editable: true,
            polygons_world: Vec::new(),
            raster_display: None,
            source_geojson: Some(PathBuf::from(format!("mask-{id}.geojson"))),
        }
    }

    #[test]
    fn mask_layer_closes_polygons_and_round_trips_project_state() {
        let mut mask = layer(7, egui::vec2(2.5, -4.0));
        mask.add_closed_polygon(vec![
            egui::pos2(0.0, 0.0),
            egui::pos2(10.0, 0.0),
            egui::pos2(10.0, 10.0),
        ]);
        mask.add_closed_polygon(vec![egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)]);

        assert_eq!(mask.polygons_world.len(), 1);
        assert_eq!(
            mask.polygons_world[0].first(),
            mask.polygons_world[0].last()
        );
        let project = mask.to_project();
        let restored = MaskLayer::from_project(&project);
        assert_eq!(restored.id, mask.id);
        assert_eq!(restored.name, mask.name);
        assert_eq!(restored.visible, mask.visible);
        assert_eq!(restored.opacity, mask.opacity);
        assert_eq!(restored.width_screen_px, mask.width_screen_px);
        assert_eq!(restored.display_mode, mask.display_mode);
        assert_eq!(restored.color_rgb, mask.color_rgb);
        assert_eq!(restored.offset_world, mask.offset_world);
        assert_eq!(restored.editable, mask.editable);
        assert_eq!(restored.polygons_world, mask.polygons_world);
        assert!(restored.raster_display.is_none());

        let mut legacy = project;
        legacy.opacity = 0.0;
        legacy.width_screen_px = 0.0;
        legacy.display_mode = None;
        legacy.color_rgb = [0, 0, 0];
        let restored = MaskLayer::from_project(&legacy);
        assert_eq!(restored.opacity, 0.9);
        assert_eq!(restored.width_screen_px, 2.0);
        assert_eq!(restored.display_mode, MaskDisplayMode::OutlineOnly);
        assert_eq!(restored.color_rgb, [255, 210, 60]);
    }

    #[test]
    fn mask_geojson_export_preserves_geometry_and_layer_metadata() {
        let mut first = layer(1, egui::vec2(100.0, 200.0));
        first.add_closed_polygon(vec![
            egui::pos2(0.0, 0.0),
            egui::pos2(10.0, 0.0),
            egui::pos2(10.0, 5.0),
        ]);
        let mut second = layer(2, egui::vec2(-5.0, 3.0));
        second.display_mode = MaskDisplayMode::FilledPreview;
        second.add_closed_polygon(vec![
            egui::pos2(20.0, 20.0),
            egui::pos2(30.0, 20.0),
            egui::pos2(30.0, 30.0),
        ]);
        let layers = vec![first, second];

        let value = export_mask_layers_geojson_value(&layers);
        assert_eq!(value["type"], "FeatureCollection");
        assert_eq!(value["odon_masks_version"], 1);
        assert_eq!(value["features"].as_array().map(Vec::len), Some(2));
        assert_eq!(value["features"][0]["properties"]["layer_id"], 1);
        assert_eq!(
            value["features"][1]["properties"]["layer_display_mode"],
            "filled_preview"
        );
        assert_eq!(
            value["features"][0]["geometry"]["coordinates"][0][0],
            serde_json::json!([100.0, 200.0])
        );

        let path =
            std::env::temp_dir().join(format!("odon-mask-export-{}.geojson", std::process::id()));
        save_mask_layers_geojson(&path, &layers).expect("save mask GeoJSON");
        let loaded = load_geojson_polylines_world(&path, 1.0, PolygonRingMode::AllRings)
            .expect("reload mask geometry");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0][0], egui::pos2(100.0, 200.0));
        assert_eq!(loaded[1][0], egui::pos2(15.0, 23.0));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn saving_empty_masks_is_rejected() {
        let path = std::env::temp_dir().join(format!(
            "odon-empty-mask-export-{}.geojson",
            std::process::id()
        ));
        let error = save_mask_layers_geojson(&path, &[layer(1, egui::Vec2::ZERO)])
            .expect_err("empty mask export must fail");
        assert!(error.to_string().contains("no mask shapes"));
        assert!(!path.exists());
    }
}
