use std::fs;
use std::path::Path;

use anyhow::Context;
use eframe::egui;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolygonRingMode {
    /// Load exterior + interior rings (holes).
    AllRings,
    /// Load only the exterior ring (first ring).
    ExteriorOnly,
}

pub fn load_geojson_polylines_world(
    path: &Path,
    downsample_factor: f32,
    ring_mode: PolygonRingMode,
) -> anyhow::Result<Vec<Vec<egui::Pos2>>> {
    if !path.exists() {
        anyhow::bail!("missing GeoJSON file: {}", path.to_string_lossy());
    }
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.to_string_lossy()))?;
    let root: serde_json::Value = serde_json::from_str(&text).context("failed to parse GeoJSON")?;

    let feats = root
        .get("features")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let downsample_factor = downsample_factor.max(1e-6);

    let mut out: Vec<Vec<egui::Pos2>> = Vec::new();
    for feat in feats {
        let Some(geom) = feat.get("geometry") else {
            continue;
        };
        let gtype = geom
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        let coords = geom.get("coordinates");
        match gtype.as_str() {
            "polygon" => {
                if let Some(rings) = coords.and_then(|v| v.as_array()) {
                    match ring_mode {
                        PolygonRingMode::AllRings => {
                            for ring in rings {
                                if let Some(points) =
                                    parse_line_points(ring, downsample_factor, true)
                                {
                                    out.push(points);
                                }
                            }
                        }
                        PolygonRingMode::ExteriorOnly => {
                            if let Some(ring) = rings.first() {
                                if let Some(points) =
                                    parse_line_points(ring, downsample_factor, true)
                                {
                                    out.push(points);
                                }
                            }
                        }
                    }
                }
            }
            "multipolygon" => {
                if let Some(polys) = coords.and_then(|v| v.as_array()) {
                    for poly in polys {
                        let Some(rings) = poly.as_array() else {
                            continue;
                        };
                        match ring_mode {
                            PolygonRingMode::AllRings => {
                                for ring in rings {
                                    if let Some(points) =
                                        parse_line_points(ring, downsample_factor, true)
                                    {
                                        out.push(points);
                                    }
                                }
                            }
                            PolygonRingMode::ExteriorOnly => {
                                if let Some(ring) = rings.first() {
                                    if let Some(points) =
                                        parse_line_points(ring, downsample_factor, true)
                                    {
                                        out.push(points);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            "linestring" => {
                if let Some(points) =
                    coords.and_then(|c| parse_line_points(c, downsample_factor, false))
                {
                    out.push(points);
                }
            }
            "multilinestring" => {
                if let Some(lines) = coords.and_then(|v| v.as_array()) {
                    for line in lines {
                        if let Some(points) = parse_line_points(line, downsample_factor, false) {
                            out.push(points);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if out.is_empty() {
        anyhow::bail!("no supported shapes in GeoJSON");
    }
    Ok(out)
}

fn parse_line_points(
    node: &serde_json::Value,
    downsample_factor: f32,
    close: bool,
) -> Option<Vec<egui::Pos2>> {
    let arr = node.as_array()?;
    let mut pts: Vec<egui::Pos2> = Vec::with_capacity(arr.len().saturating_add(1));
    for p in arr {
        let Some(xy) = p.as_array() else {
            continue;
        };
        if xy.len() < 2 {
            continue;
        }
        let Some(x0) = xy.get(0).and_then(|v| v.as_f64()) else {
            continue;
        };
        let Some(y0) = xy.get(1).and_then(|v| v.as_f64()) else {
            continue;
        };
        // `downsample_factor` is interpreted as "the GeoJSON coordinates were generated on an
        // image downsampled by N", so we scale coordinates up by N to match the viewer's
        // world-pixel coordinate system (typically level-0 pixels).
        let x = x0 as f32 * downsample_factor;
        let y = y0 as f32 * downsample_factor;
        if x.is_finite() && y.is_finite() {
            pts.push(egui::pos2(x, y));
        }
    }
    if pts.len() < 2 {
        return None;
    }

    // Drop duplicate closing vertex if present.
    if pts.len() >= 2 && pts.first() == pts.last() {
        pts.pop();
    }
    if close && pts.len() >= 2 {
        pts.push(*pts.first().unwrap());
    }
    Some(pts)
}
