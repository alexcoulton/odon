use std::fs;
use std::path::Path;

use anyhow::Context;
use serde_json::Value;

pub(crate) fn load_geojson_mask_polylines(
    path: &Path,
    downsample_factor: f32,
) -> anyhow::Result<Vec<Vec<[f32; 2]>>> {
    if !path.exists() {
        anyhow::bail!("missing GeoJSON file: {}", path.to_string_lossy());
    }
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.to_string_lossy()))?;
    let root: Value = serde_json::from_str(&text).context("failed to parse GeoJSON")?;
    let features = root
        .get("features")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let scale = downsample_factor.max(1.0e-6);
    let mut polygons = Vec::new();
    for feature in features {
        let Some(geometry) = feature.get("geometry") else {
            continue;
        };
        let geometry_type = geometry
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        let coordinates = geometry.get("coordinates");
        match geometry_type.as_str() {
            "polygon" => {
                if let Some(rings) = coordinates.and_then(Value::as_array) {
                    polygons.extend(
                        rings
                            .iter()
                            .filter_map(|ring| parse_geojson_points(ring, scale, true)),
                    );
                }
            }
            "multipolygon" => {
                if let Some(values) = coordinates.and_then(Value::as_array) {
                    for polygon in values {
                        if let Some(rings) = polygon.as_array() {
                            polygons.extend(
                                rings
                                    .iter()
                                    .filter_map(|ring| parse_geojson_points(ring, scale, true)),
                            );
                        }
                    }
                }
            }
            "linestring" => {
                if let Some(points) =
                    coordinates.and_then(|value| parse_geojson_points(value, scale, false))
                {
                    polygons.push(points);
                }
            }
            "multilinestring" => {
                if let Some(lines) = coordinates.and_then(Value::as_array) {
                    polygons.extend(
                        lines
                            .iter()
                            .filter_map(|line| parse_geojson_points(line, scale, false)),
                    );
                }
            }
            _ => {}
        }
    }
    if polygons.is_empty() {
        anyhow::bail!("no supported shapes in GeoJSON");
    }
    Ok(polygons)
}

fn parse_geojson_points(node: &Value, scale: f32, close: bool) -> Option<Vec<[f32; 2]>> {
    let coordinates = node.as_array()?;
    let mut points = coordinates
        .iter()
        .filter_map(|coordinate| {
            let pair = coordinate.as_array()?;
            let x = pair.first()?.as_f64()? as f32 * scale;
            let y = pair.get(1)?.as_f64()? as f32 * scale;
            (x.is_finite() && y.is_finite()).then_some([x, y])
        })
        .collect::<Vec<_>>();
    if points.len() < 2 {
        return None;
    }
    if points.first() == points.last() {
        points.pop();
    }
    if close && let Some(first) = points.first().copied() {
        points.push(first);
    }
    Some(points)
}
