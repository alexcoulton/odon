use std::fs;
use std::path::Path;

use anyhow::Context;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolygonRingMode {
    AllRings,
    ExteriorOnly,
}

pub fn load_geojson_polyline_coordinates_world(
    path: &Path,
    downsample_factor: f32,
    ring_mode: PolygonRingMode,
) -> anyhow::Result<Vec<Vec<[f32; 2]>>> {
    if !path.exists() {
        anyhow::bail!("missing GeoJSON file: {}", path.to_string_lossy());
    }
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.to_string_lossy()))?;
    let root: serde_json::Value = serde_json::from_str(&text).context("failed to parse GeoJSON")?;
    let features = root
        .get("features")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    let downsample_factor = downsample_factor.max(1e-6);
    let mut output = Vec::new();
    for feature in features {
        let Some(geometry) = feature.get("geometry") else {
            continue;
        };
        let kind = geometry
            .get("type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        let coordinates = geometry.get("coordinates");
        match kind.as_str() {
            "polygon" => {
                if let Some(rings) = coordinates.and_then(serde_json::Value::as_array) {
                    append_polygon_rings(&mut output, rings, downsample_factor, ring_mode);
                }
            }
            "multipolygon" => {
                if let Some(polygons) = coordinates.and_then(serde_json::Value::as_array) {
                    for polygon in polygons {
                        if let Some(rings) = polygon.as_array() {
                            append_polygon_rings(&mut output, rings, downsample_factor, ring_mode);
                        }
                    }
                }
            }
            "linestring" => {
                if let Some(points) =
                    coordinates.and_then(|value| parse_line_points(value, downsample_factor, false))
                {
                    output.push(points);
                }
            }
            "multilinestring" => {
                if let Some(lines) = coordinates.and_then(serde_json::Value::as_array) {
                    for line in lines {
                        if let Some(points) = parse_line_points(line, downsample_factor, false) {
                            output.push(points);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    if output.is_empty() {
        anyhow::bail!("no supported shapes in GeoJSON");
    }
    Ok(output)
}

fn append_polygon_rings(
    output: &mut Vec<Vec<[f32; 2]>>,
    rings: &[serde_json::Value],
    downsample_factor: f32,
    ring_mode: PolygonRingMode,
) {
    let take = match ring_mode {
        PolygonRingMode::AllRings => rings.len(),
        PolygonRingMode::ExteriorOnly => rings.len().min(1),
    };
    for ring in rings.iter().take(take) {
        if let Some(points) = parse_line_points(ring, downsample_factor, true) {
            output.push(points);
        }
    }
}

fn parse_line_points(
    node: &serde_json::Value,
    downsample_factor: f32,
    close: bool,
) -> Option<Vec<[f32; 2]>> {
    let coordinates = node.as_array()?;
    let mut points = Vec::with_capacity(coordinates.len().saturating_add(1));
    for coordinate in coordinates {
        let Some(xy) = coordinate.as_array() else {
            continue;
        };
        let Some(x) = xy.first().and_then(serde_json::Value::as_f64) else {
            continue;
        };
        let Some(y) = xy.get(1).and_then(serde_json::Value::as_f64) else {
            continue;
        };
        let x = x as f32 * downsample_factor;
        let y = y as f32 * downsample_factor;
        if x.is_finite() && y.is_finite() {
            points.push([x, y]);
        }
    }
    if points.len() < 2 {
        return None;
    }
    if points.first() == points.last() {
        points.pop();
    }
    if close {
        points.push(points[0]);
    }
    Some(points)
}
