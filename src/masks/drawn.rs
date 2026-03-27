use std::fs;
use std::path::Path;

use anyhow::Context;
use eframe::egui;

use crate::custom::roi_selector::{MasksConfig, RoiEntry};
use super::ResolvedMasksPath;

#[derive(Debug, Clone)]
pub struct DrawnMasksLayer {
    pub visible: bool,
    pub opacity: f32,
    pub width_screen_px: f32,
    pub color_rgb: [u8; 3],
    pub polygons_world: Vec<Vec<egui::Pos2>>,
}

impl Default for DrawnMasksLayer {
    fn default() -> Self {
        Self {
            visible: true,
            opacity: 0.9,
            width_screen_px: 2.0,
            color_rgb: [255, 210, 60],
            polygons_world: Vec::new(),
        }
    }
}

impl DrawnMasksLayer {
    pub fn clear(&mut self) {
        self.polygons_world.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.polygons_world.is_empty()
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

    pub fn save_append_to_geojson(
        &self,
        resolved: &ResolvedMasksPath,
        roi_root: &Path,
        _cfg: &MasksConfig,
        _entry: Option<&RoiEntry>,
    ) -> anyhow::Result<()> {
        if self.polygons_world.is_empty() {
            anyhow::bail!("no drawn masks to save");
        }

        let path = &resolved.geojson_path;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.to_string_lossy()))?;
        }

        let mut root: serde_json::Value = if path.exists() {
            let text = fs::read_to_string(path)
                .with_context(|| format!("failed to read {}", path.to_string_lossy()))?;
            serde_json::from_str(&text).context("failed to parse existing GeoJSON")?
        } else {
            serde_json::json!({"type":"FeatureCollection","features":[]})
        };

        let feats = root
            .get_mut("features")
            .and_then(|v| v.as_array_mut())
            .ok_or_else(|| anyhow::anyhow!("GeoJSON missing 'features' array"))?;

        let ds = resolved.downsample_factor.max(1e-6);
        for (idx, poly) in self.polygons_world.iter().enumerate() {
            if poly.len() < 4 {
                continue;
            }
            let mut ring: Vec<Vec<f64>> = Vec::with_capacity(poly.len());
            for &p in poly {
                ring.push(vec![(p.x as f64) * (ds as f64), (p.y as f64) * (ds as f64)]);
            }
            if ring.first() != ring.last() {
                if let Some(first) = ring.first().cloned() {
                    ring.push(first);
                }
            }

            let feature = serde_json::json!({
                "type": "Feature",
                "geometry": { "type": "Polygon", "coordinates": [ ring ] },
                "properties": {
                    "layer": "odon_drawn_masks",
                    "shape_index": idx as i64,
                    "roi_root": roi_root.to_string_lossy(),
                }
            });
            feats.push(feature);
        }

        let text = serde_json::to_string_pretty(&root).context("failed to encode GeoJSON")?;
        fs::write(path, text)
            .with_context(|| format!("failed to write {}", path.to_string_lossy()))?;
        Ok(())
    }
}
