use super::*;
use anyhow::Context;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

pub(super) fn export_mask_layers_geojson(
    path: &std::path::Path,
    layers: &[ProjectMaskLayer],
    overwrite: bool,
) -> anyhow::Result<u64> {
    if path.exists() && !overwrite {
        anyhow::bail!("destination exists; pass overwrite=true to replace it");
    }
    if layers.iter().all(|layer| layer.polygons_world.is_empty()) {
        anyhow::bail!("no mask shapes to save");
    }
    let features = layers
        .iter()
        .flat_map(|layer| {
            layer
                .polygons_world
                .iter()
                .enumerate()
                .filter_map(move |(shape_index, polygon)| {
                    if polygon.len() < 3 {
                        return None;
                    }
                    let mut ring = polygon
                        .iter()
                        .map(|point| {
                            vec![
                                f64::from(point[0] + layer.offset_world[0]),
                                f64::from(point[1] + layer.offset_world[1]),
                            ]
                        })
                        .collect::<Vec<_>>();
                    if ring.first() != ring.last()
                        && let Some(first) = ring.first().cloned()
                    {
                        ring.push(first);
                    }
                    Some(json!({
                        "type": "Feature",
                        "geometry": {"type":"Polygon","coordinates":[ring]},
                        "properties": {
                            "layer_id": layer.id,
                            "layer_name": layer.name,
                            "layer_color_rgb": layer.color_rgb,
                            "layer_opacity": layer.opacity,
                            "layer_width_screen_px": layer.width_screen_px,
                            "layer_display_mode": layer.display_mode.as_deref().unwrap_or("outline_only"),
                            "layer_visible": layer.visible,
                            "layer_editable": layer.editable,
                            "shape_index": shape_index,
                        },
                    }))
                })
        })
        .collect::<Vec<_>>();
    if features.is_empty() {
        anyhow::bail!("no mask shapes to save");
    }
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let text = serde_json::to_string_pretty(&json!({
        "type":"FeatureCollection",
        "odon_masks_version":1,
        "features":features,
    }))?;
    fs::write(path, text.as_bytes())?;
    Ok(text.len() as u64)
}

pub(super) fn append_mask_layers_geojson(
    path: &Path,
    layers: &[ProjectMaskLayer],
    downsample_factor: f32,
    roi_root: &str,
    cancelled: impl Fn() -> bool,
) -> anyhow::Result<MaskAppendWorkerResult> {
    anyhow::ensure!(!cancelled(), "mask save was cancelled");
    let scale = downsample_factor.max(1.0e-6);
    let appended_polygon_count = layers
        .iter()
        .map(|layer| {
            layer
                .polygons_world
                .iter()
                .filter(|polygon| polygon.len() >= 3)
                .count()
        })
        .sum::<usize>();
    anyhow::ensure!(appended_polygon_count > 0, "no drawn masks to save");

    let mut root = if path.exists() {
        let text = fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        serde_json::from_str::<Value>(&text).context("failed to parse existing GeoJSON")?
    } else {
        json!({"type":"FeatureCollection","features":[]})
    };
    anyhow::ensure!(
        root.get("type").and_then(Value::as_str) == Some("FeatureCollection"),
        "GeoJSON root must be a FeatureCollection"
    );
    let features = root
        .get_mut("features")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| anyhow::anyhow!("GeoJSON missing 'features' array"))?;

    let mut appended_index = 0usize;
    for layer in layers {
        for polygon in &layer.polygons_world {
            if polygon.len() < 3 {
                continue;
            }
            let mut ring = polygon
                .iter()
                .map(|point| {
                    vec![
                        f64::from(point[0] + layer.offset_world[0]) / f64::from(scale),
                        f64::from(point[1] + layer.offset_world[1]) / f64::from(scale),
                    ]
                })
                .collect::<Vec<_>>();
            if ring.first() != ring.last()
                && let Some(first) = ring.first().cloned()
            {
                ring.push(first);
            }
            features.push(json!({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "layer": "odon_masks",
                    "layer_id": layer.id,
                    "layer_name": layer.name,
                    "shape_index": appended_index,
                    "roi_root": roi_root,
                },
            }));
            appended_index = appended_index.saturating_add(1);
        }
    }

    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).with_context(|| format!("failed to create {}", parent.display()))?;
    let temporary = mask_temporary_sibling(path);
    let result = (|| -> anyhow::Result<u64> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .with_context(|| format!("failed to create {}", temporary.display()))?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, &root).context("failed to encode GeoJSON")?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        anyhow::ensure!(!cancelled(), "mask save was cancelled");
        replace_file(&temporary, path)?;
        Ok(fs::metadata(path)?.len())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    let bytes = result?;
    let polygons_world = crate::model::load_geojson_mask_polylines(path, scale)?;
    Ok(MaskAppendWorkerResult {
        bytes,
        appended_polygon_count,
        polygons_world,
    })
}

fn mask_temporary_sibling(path: &Path) -> PathBuf {
    static NEXT_TEMP: AtomicU64 = AtomicU64::new(1);
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("masks.geojson");
    let sequence = NEXT_TEMP.fetch_add(1, AtomicOrdering::Relaxed);
    parent.join(format!(
        ".{name}.odon-{}-{sequence}.tmp",
        std::process::id()
    ))
}

fn replace_file(temporary: &Path, destination: &Path) -> anyhow::Result<()> {
    match fs::rename(temporary, destination) {
        Ok(()) => Ok(()),
        Err(error) if destination.exists() => {
            fs::remove_file(destination)?;
            fs::rename(temporary, destination).map_err(Into::into)
        }
        Err(error) => Err(error.into()),
    }
}
