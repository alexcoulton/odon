use super::*;

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
