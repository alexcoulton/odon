use std::path::{Path, PathBuf};

use anyhow::Context;
use eframe::egui;

use crate::custom::roi_selector::{MasksConfig, RoiEntry};
use crate::geometry::geojson::{PolygonRingMode, load_geojson_polylines_world};

#[derive(Debug, Clone)]
pub struct ResolvedMasksPath {
    pub geojson_path: PathBuf,
    pub downsample_factor: f32,
}

#[derive(Debug, Clone)]
pub struct ExclusionMasksLayer {
    pub visible: bool,
    pub opacity: f32,
    pub width_screen_px: f32,
    pub color_rgb: [u8; 3],
    pub polylines_world: Vec<Vec<egui::Pos2>>,
    pub loaded_geojson: Option<PathBuf>,
}

impl Default for ExclusionMasksLayer {
    fn default() -> Self {
        Self {
            visible: false,
            opacity: 0.85,
            width_screen_px: 1.5,
            color_rgb: [50, 220, 255],
            polylines_world: Vec::new(),
            loaded_geojson: None,
        }
    }
}

impl ExclusionMasksLayer {
    pub fn clear(&mut self) {
        self.polylines_world.clear();
        self.loaded_geojson = None;
    }

    pub fn load_for_roi(
        &mut self,
        roi_root: &Path,
        cfg: &MasksConfig,
        entry: Option<&RoiEntry>,
    ) -> anyhow::Result<usize> {
        let resolved = resolve_masks_geojson_path_and_downsample(roi_root, cfg, entry)?;
        let polylines = load_geojson_polylines_world(
            &resolved.geojson_path,
            resolved.downsample_factor,
            PolygonRingMode::AllRings,
        )
        .with_context(|| {
            format!(
                "failed to load masks: {}",
                resolved.geojson_path.to_string_lossy()
            )
        })?;

        self.polylines_world = polylines;
        self.loaded_geojson = Some(resolved.geojson_path);
        self.visible = true;
        Ok(self.polylines_world.len())
    }
}

pub fn resolve_masks_geojson_path_and_downsample(
    roi_root: &Path,
    cfg: &MasksConfig,
    entry: Option<&RoiEntry>,
) -> anyhow::Result<ResolvedMasksPath> {
    let Some(masks_dir) = cfg.masks_dir.as_ref() else {
        anyhow::bail!("project config missing `masks_dir` for this dataset");
    };

    let (sample, roi_short) = infer_sample_and_roi_short(roi_root, cfg, entry);
    let rel_dir = sample.as_ref().filter(|s| !s.is_empty());
    let dir = match rel_dir {
        Some(sample) => masks_dir.join(sample),
        None => masks_dir.to_path_buf(),
    };
    let filename = format!("{roi_short}_artefact_masks_fullres_Shapes.geojson");
    let geojson_path = dir.join(filename);
    let downsample_factor = downsample_factor_for_roi(roi_root, cfg).max(1e-6);

    Ok(ResolvedMasksPath {
        geojson_path,
        downsample_factor,
    })
}

fn downsample_factor_for_roi(roi_root: &Path, cfg: &MasksConfig) -> f32 {
    let mut best: Option<(usize, f32)> = None;
    if let Some(base) = cfg.base_dir_full_res.as_ref() {
        if roi_root.starts_with(base) {
            let len = base.as_os_str().len();
            best = Some((len, cfg.masks_downsample_full_res));
        }
    }
    if let Some(base) = cfg.base_dir_downsampled.as_ref() {
        if roi_root.starts_with(base) {
            let len = base.as_os_str().len();
            if best.as_ref().map(|b| len > b.0).unwrap_or(true) {
                best = Some((len, cfg.masks_downsample_downsampled));
            }
        }
    }

    if let Some((_, f)) = best {
        return f;
    }
    if cfg.base_dir_full_res.is_some() {
        return cfg.masks_downsample_full_res;
    }
    cfg.masks_downsample_downsampled
}

fn infer_sample_and_roi_short(
    roi_root: &Path,
    cfg: &MasksConfig,
    entry: Option<&RoiEntry>,
) -> (Option<String>, String) {
    if let Some(e) = entry {
        let sample = e.sample_name.clone();
        let roi_short = if e.roi_short.is_empty() {
            infer_roi_short_from_path(roi_root)
        } else {
            e.roi_short.clone()
        };
        return (sample, roi_short);
    }

    let roi_short = infer_roi_short_from_path(roi_root);
    if cfg.layout != "sample_roi" {
        return (None, roi_short);
    }

    // Prefer relative-to-base inference to match napari helpers.
    for base in [
        cfg.base_dir_full_res.as_ref(),
        cfg.base_dir_downsampled.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        if let Ok(rel) = roi_root.strip_prefix(base) {
            let mut comps = rel.components().filter_map(|c| c.as_os_str().to_str());
            let first = comps.next().unwrap_or("");
            if !first.is_empty() && !first.to_ascii_lowercase().starts_with("roi") {
                return (Some(first.to_string()), roi_short);
            }
            return (None, roi_short);
        }
    }

    // Fallback: .../<sample>/<ROI>/<ROI>.ome.zarr
    let sample = roi_root
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .map(|s| s.to_string());
    (sample, roi_short)
}

fn infer_roi_short_from_path(roi_root: &Path) -> String {
    // Typical: .../ROI1/ROI1.ome.zarr
    if let Some(parent) = roi_root.parent() {
        if let Some(name) = parent.file_name().and_then(|s| s.to_str()) {
            if !name.is_empty() {
                return name.to_string();
            }
        }
    }
    let name = roi_root
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("roi");
    name.trim_end_matches(".ome.zarr").to_string()
}

// GeoJSON parsing lives in `src/geojson_polylines.rs` so other layers can reuse it.
