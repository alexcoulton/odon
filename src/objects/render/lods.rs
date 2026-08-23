//! Render LOD construction, property discovery, and color grouping.

use super::*;

pub(in crate::objects) fn build_render_lods(
    objects: &[GeoJsonObjectFeature],
) -> anyhow::Result<Vec<ObjectRenderLod>> {
    let polylines_world = flatten_object_polylines(objects);
    build_render_lods_from_polylines(&polylines_world)
}

pub(in crate::objects) fn build_object_selection_render_lods(
    objects: &[GeoJsonObjectFeature],
) -> anyhow::Result<Vec<ObjectSelectionRenderLod>> {
    let polylines_world = flatten_indexed_object_polylines(objects);
    build_object_selection_render_lods_from_polylines(&polylines_world)
}

pub(in crate::objects) fn build_render_lods_from_polylines(
    polylines_world: &[Vec<egui::Pos2>],
) -> anyhow::Result<Vec<ObjectRenderLod>> {
    if polylines_world.is_empty() {
        anyhow::bail!("no valid object outlines available for rendering");
    }

    let lod_specs: &[(u8, f32, f32)] = &[
        (0, 1.0, 2048.0),
        (1, 4.0, 8192.0),
        (2, 16.0, 32768.0),
        (3, 64.0, 1_000_000.0),
    ];

    let mut out = Vec::new();
    for (lod, step, bin_size) in lod_specs {
        let step = step.max(1.0);
        let bin_size = bin_size.max(256.0);
        let lines = if *lod == 0 {
            polylines_world.to_vec()
        } else if *lod >= 3 {
            let sampled = sample_polylines(polylines_world, 4_000);
            quantize_polylines(&sampled, step)
        } else {
            quantize_polylines(polylines_world, step)
        };
        let Some(bins) = LineSegmentsBins::build_from_polylines(&lines, bin_size) else {
            continue;
        };
        out.push(ObjectRenderLod {
            lod: *lod,
            bins: Arc::new(bins),
        });
    }

    if out.is_empty() {
        anyhow::bail!("no valid renderable object outlines after parsing");
    }
    Ok(out)
}

pub(in crate::objects) fn build_object_selection_render_lods_from_polylines(
    polylines_world: &[(usize, Vec<egui::Pos2>)],
) -> anyhow::Result<Vec<ObjectSelectionRenderLod>> {
    if polylines_world.is_empty() {
        anyhow::bail!("no valid object outlines available for selection rendering");
    }

    let lod_specs: &[(u8, f32, f32)] = &[
        (0, 1.0, 2048.0),
        (1, 4.0, 8192.0),
        (2, 16.0, 32768.0),
        (3, 64.0, 1_000_000.0),
    ];

    let mut out = Vec::new();
    for (lod, step, bin_size) in lod_specs {
        let step = step.max(1.0);
        let bin_size = bin_size.max(1024.0);
        let lines = if *lod == 0 {
            polylines_world.to_vec()
        } else {
            quantize_indexed_polylines(polylines_world, step)
        };
        let Some(bins) = ObjectLineSegmentsBins::build_from_indexed_polylines(&lines, bin_size)
        else {
            continue;
        };
        out.push(ObjectSelectionRenderLod {
            lod: *lod,
            bins: Arc::new(bins),
        });
    }

    if out.is_empty() {
        anyhow::bail!("no valid renderable object selection outlines after parsing");
    }
    Ok(out)
}

pub(in crate::objects) fn discover_categorical_color_keys(
    objects: &[GeoJsonObjectFeature],
) -> Vec<String> {
    let mut distinct: HashMap<String, HashSet<String>> = HashMap::new();
    let mut overflow = HashSet::new();

    for obj in objects {
        for (key, value) in &obj.inline_properties {
            let Some(text) = property_scalar_value(value) else {
                continue;
            };
            if overflow.contains(key) {
                continue;
            }
            let set = distinct.entry(key.clone()).or_default();
            set.insert(text);
            if set.len() > 24 {
                distinct.remove(key);
                overflow.insert(key.clone());
            }
        }
    }

    let mut keys = distinct.into_keys().collect::<Vec<_>>();
    keys.sort();
    keys
}

pub(in crate::objects) fn discover_property_keys(objects: &[GeoJsonObjectFeature]) -> Vec<String> {
    let mut keys = HashSet::new();
    for obj in objects {
        keys.extend(obj.inline_properties.keys().cloned());
    }
    let mut out = keys.into_iter().collect::<Vec<_>>();
    out.sort();
    out
}

pub(in crate::objects) fn discover_scalar_property_keys(
    objects: &[GeoJsonObjectFeature],
) -> Vec<String> {
    let mut keys = HashSet::new();
    for obj in objects {
        for (key, value) in &obj.inline_properties {
            if property_scalar_value(value).is_some() {
                keys.insert(key.clone());
            }
        }
    }
    let mut out = keys.into_iter().collect::<Vec<_>>();
    out.sort();
    out
}

pub(in crate::objects) fn build_color_groups_for_property_labels<'a, I>(
    objects: I,
    property_key: &str,
) -> anyhow::Result<ObjectColorGroups>
where
    I: IntoIterator<Item = (usize, &'a GeoJsonObjectFeature, String)>,
{
    use std::collections::BTreeMap;
    use std::hash::{Hash, Hasher};

    let mut grouped: BTreeMap<String, Vec<Vec<egui::Pos2>>> = BTreeMap::new();
    let mut grouped_points: BTreeMap<String, Vec<egui::Pos2>> = BTreeMap::new();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut grouped_indices: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut object_count = 0usize;

    for (object_index, obj, value) in objects {
        object_count = object_count.max(object_index.saturating_add(1));
        counts
            .entry(value.clone())
            .and_modify(|count| *count += 1)
            .or_insert(1);
        grouped
            .entry(value.clone())
            .or_default()
            .extend(obj.polygons_world.iter().cloned());
        grouped_points
            .entry(value.clone())
            .or_default()
            .push(object_proxy_position_world(obj));
        grouped_indices.entry(value).or_default().push(object_index);
    }

    if counts.is_empty() {
        anyhow::bail!("no scalar values found for property '{property_key}'");
    }

    let mut groups = Vec::new();
    for (value_label, _) in counts {
        let polylines = grouped.remove(&value_label).unwrap_or_default();
        let lods = if polylines.is_empty() {
            Vec::new()
        } else {
            build_render_lods_from_polylines(&polylines)?
        };
        let point_positions = grouped_points.remove(&value_label).unwrap_or_default();
        let mut object_indices = grouped_indices.remove(&value_label).unwrap_or_default();
        object_indices.sort_unstable();
        object_indices.dedup();

        let mut fill_state = vec![0u8; object_count];
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        property_key.hash(&mut hasher);
        value_label.hash(&mut hasher);
        for object_index in object_indices {
            if let Some(slot) = fill_state.get_mut(object_index) {
                *slot = 255;
                object_index.hash(&mut hasher);
            }
        }

        groups.push(ObjectColorGroup {
            color_rgb: hashed_color_rgb(property_key, &value_label),
            value_label,
            lods,
            point_values: Arc::new(vec![1.0; point_positions.len()]),
            point_positions_world: Arc::new(point_positions),
            fill_state: Arc::new(fill_state),
            fill_generation: hasher.finish(),
        });
    }

    Ok(ObjectColorGroups {
        property_key: property_key.to_string(),
        groups,
    })
}

pub(in crate::objects) fn property_scalar_value(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::Null => None,
        serde_json::Value::Bool(v) => Some(v.to_string()),
        serde_json::Value::Number(v) => Some(v.to_string()),
        serde_json::Value::String(v) => {
            let trimmed = v.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        }
        _ => None,
    }
}

pub(in crate::objects) fn hashed_color_rgb(property_key: &str, value_label: &str) -> [u8; 3] {
    use std::hash::{Hash, Hasher};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    property_key.hash(&mut hasher);
    value_label.hash(&mut hasher);
    let hash = hasher.finish();
    let hue = (hash % 360) as f32;
    hsv_to_rgb(hue, 0.6, 0.95)
}

pub(in crate::objects) fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [u8; 3] {
    let c = v * s;
    let hh = (h / 60.0) % 6.0;
    let x = c * (1.0 - ((hh % 2.0) - 1.0).abs());
    let (r1, g1, b1) = match hh {
        h if (0.0..1.0).contains(&h) => (c, x, 0.0),
        h if (1.0..2.0).contains(&h) => (x, c, 0.0),
        h if (2.0..3.0).contains(&h) => (0.0, c, x),
        h if (3.0..4.0).contains(&h) => (0.0, x, c),
        h if (4.0..5.0).contains(&h) => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [
        ((r1 + m) * 255.0).round() as u8,
        ((g1 + m) * 255.0).round() as u8,
        ((b1 + m) * 255.0).round() as u8,
    ]
}

pub(in crate::objects) fn flatten_object_polylines(
    objects: &[GeoJsonObjectFeature],
) -> Vec<Vec<egui::Pos2>> {
    let mut out = Vec::new();
    for obj in objects {
        for poly in &obj.polygons_world {
            if poly.len() >= 2 {
                out.push(poly.clone());
            }
        }
    }
    out
}

pub(in crate::objects) fn flatten_indexed_object_polylines(
    objects: &[GeoJsonObjectFeature],
) -> Vec<(usize, Vec<egui::Pos2>)> {
    let mut out = Vec::new();
    for (object_index, obj) in objects.iter().enumerate() {
        for poly in &obj.polygons_world {
            if poly.len() >= 2 {
                out.push((object_index, poly.clone()));
            }
        }
    }
    out
}

pub(in crate::objects) fn sample_polylines(
    polys: &[Vec<egui::Pos2>],
    max_polylines: usize,
) -> Vec<Vec<egui::Pos2>> {
    if polys.len() <= max_polylines {
        return polys.to_vec();
    }

    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for pts in polys {
        for p in pts {
            if p.x.is_finite() && p.y.is_finite() {
                min_x = min_x.min(p.x);
                min_y = min_y.min(p.y);
                max_x = max_x.max(p.x);
                max_y = max_y.max(p.y);
            }
        }
    }
    let w = (max_x - min_x).max(1.0);
    let h = (max_y - min_y).max(1.0);

    let grid_w = 32usize;
    let grid_h = 32usize;
    let cell_w = (w / grid_w as f32).max(1.0);
    let cell_h = (h / grid_h as f32).max(1.0);
    let cells = grid_w * grid_h;
    let per_cell_cap = max_polylines.div_ceil(cells).max(1);

    let mut chosen = vec![false; polys.len()];
    let mut buckets = Vec::with_capacity(max_polylines.min(polys.len()));
    let mut bucket_counts = vec![0usize; cells];

    for (i, pts) in polys.iter().enumerate() {
        if buckets.len() >= max_polylines {
            break;
        }
        if pts.len() < 2 {
            continue;
        }
        let Some(bounds) = polyline_bounds(pts) else {
            continue;
        };
        let cx = 0.5 * (bounds.min.x + bounds.max.x);
        let cy = 0.5 * (bounds.min.y + bounds.max.y);
        let gx = ((cx - min_x) / cell_w)
            .floor()
            .clamp(0.0, (grid_w - 1) as f32) as usize;
        let gy = ((cy - min_y) / cell_h)
            .floor()
            .clamp(0.0, (grid_h - 1) as f32) as usize;
        let bi = gy * grid_w + gx;
        if bucket_counts[bi] >= per_cell_cap {
            continue;
        }
        bucket_counts[bi] += 1;
        chosen[i] = true;
        buckets.push(i);
    }

    if buckets.len() < max_polylines {
        let remaining = max_polylines - buckets.len();
        let step = (polys.len() / remaining.max(1)).max(1);
        for i in (0..polys.len()).step_by(step) {
            if buckets.len() >= max_polylines {
                break;
            }
            if chosen[i] {
                continue;
            }
            chosen[i] = true;
            buckets.push(i);
        }
    }

    buckets
        .into_iter()
        .take(max_polylines)
        .map(|i| polys[i].clone())
        .collect()
}

pub(in crate::objects) fn polyline_bounds(poly: &[egui::Pos2]) -> Option<egui::Rect> {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut any = false;
    for p in poly {
        if !(p.x.is_finite() && p.y.is_finite()) {
            continue;
        }
        any = true;
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
    }
    any.then(|| egui::Rect::from_min_max(egui::pos2(min_x, min_y), egui::pos2(max_x, max_y)))
}

pub(in crate::objects) fn quantize_polylines(
    polys: &[Vec<egui::Pos2>],
    step_world: f32,
) -> Vec<Vec<egui::Pos2>> {
    let s = step_world.max(1e-6);
    let inv = 1.0 / s;
    let mut out = Vec::with_capacity(polys.len());

    for pts in polys {
        if pts.len() < 2 {
            continue;
        }
        let is_closed = pts.first() == pts.last();
        let mut q = Vec::with_capacity(pts.len());
        for p in pts {
            if !(p.x.is_finite() && p.y.is_finite()) {
                continue;
            }
            let qp = egui::pos2((p.x * inv).round() * s, (p.y * inv).round() * s);
            if q.last().copied() == Some(qp) {
                continue;
            }
            q.push(qp);
        }
        if q.len() < 2 {
            continue;
        }
        if is_closed && q.first() != q.last() {
            if let Some(first) = q.first().copied() {
                q.push(first);
            }
        }
        if q.len() >= 2 {
            out.push(q);
        }
    }

    out
}

pub(in crate::objects) fn quantize_indexed_polylines(
    polys: &[(usize, Vec<egui::Pos2>)],
    step_world: f32,
) -> Vec<(usize, Vec<egui::Pos2>)> {
    let s = step_world.max(1e-6);
    let inv = 1.0 / s;
    let mut out = Vec::with_capacity(polys.len());

    for (object_index, pts) in polys {
        if pts.len() < 2 {
            continue;
        }
        let is_closed = pts.first() == pts.last();
        let mut q = Vec::with_capacity(pts.len());
        for p in pts {
            if !(p.x.is_finite() && p.y.is_finite()) {
                continue;
            }
            let qp = egui::pos2((p.x * inv).round() * s, (p.y * inv).round() * s);
            if q.last().copied() == Some(qp) {
                continue;
            }
            q.push(qp);
        }
        if q.len() < 2 {
            continue;
        }
        if is_closed && q.first() != q.last() {
            if let Some(first) = q.first().copied() {
                q.push(first);
            }
        }
        if q.len() >= 2 {
            out.push((*object_index, q));
        }
    }

    out
}

pub(in crate::objects) fn choose_lod_index(
    lods: &[ObjectRenderLod],
    dataset_long_side_screen_px: f32,
) -> usize {
    if lods.len() <= 1 {
        return 0;
    }
    let s = dataset_long_side_screen_px.max(1e-3);
    let desired_lod = if s < 160.0 {
        3u8
    } else if s < 420.0 {
        2u8
    } else if s < 1000.0 {
        1u8
    } else {
        0u8
    };

    let mut best_i = 0usize;
    let mut best_err = i32::MAX;
    for (i, lod) in lods.iter().enumerate() {
        let err = (lod.lod as i32 - desired_lod as i32).abs();
        if err < best_err {
            best_err = err;
            best_i = i;
        }
    }
    best_i
}

#[cfg(test)]
pub(in crate::objects) fn rect_json(rect: egui::Rect) -> serde_json::Value {
    serde_json::json!({
        "min_x": rect.min.x,
        "min_y": rect.min.y,
        "max_x": rect.max.x,
        "max_y": rect.max.y,
        "width": rect.width(),
        "height": rect.height(),
    })
}

pub(in crate::objects) fn choose_object_selection_lod_index(
    lods: &[ObjectSelectionRenderLod],
    dataset_long_side_screen_px: f32,
) -> usize {
    if lods.len() <= 1 {
        return 0;
    }
    let s = dataset_long_side_screen_px.max(1e-3);
    let desired_lod = if s < 160.0 {
        3u8
    } else if s < 420.0 {
        2u8
    } else if s < 1000.0 {
        1u8
    } else {
        0u8
    };

    let mut best_i = 0usize;
    let mut best_err = i32::MAX;
    for (i, lod) in lods.iter().enumerate() {
        let err = (lod.lod as i32 - desired_lod as i32).abs();
        if err < best_err {
            best_err = err;
            best_i = i;
        }
    }
    best_i
}
