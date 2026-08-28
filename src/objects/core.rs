use super::analysis::{
    apply_histogram_value_transform, compute_histogram_f32, kmeans_threshold_levels,
    numeric_json_value, quantile_threshold_levels,
};
use super::render::{
    build_color_groups_for_property_labels, build_object_fill_mesh,
    build_object_selection_render_lods, build_render_lods, discover_categorical_color_keys,
    discover_property_keys, discover_scalar_property_keys, hashed_color_rgb, property_scalar_value,
    summarize_geometry,
};
use super::*;
use arrow_array::RecordBatch;
use arrow_array::builder::{
    BinaryBuilder, BooleanBuilder, Float64Builder, Int64Builder, StringBuilder, UInt64Builder,
};
use arrow_schema::{Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use std::collections::{BTreeSet, HashMap};

mod export;
mod loading;
mod properties_ui;
mod runtime;
mod selection;

use loading::*;
pub(in crate::objects) use loading::{build_object_point_payload, object_proxy_position_world};
pub use loading::{
    load_control_object_resource, load_control_object_resource_with_options,
    load_control_spatialdata_object_resource,
};

// Object-layer runtime and data-loading shell.
//
// This file owns the non-render analysis-adjacent state for object layers: background loading,
// lazy property hydration, filtering/color grouping caches, selection/export state, and format
// adapters for GeoJSON, GeoParquet, CSV, and SpatialData shapes. Rendering and analysis helpers
// live in sibling modules; this file decides when those derived products are invalidated or
// rebuilt.

struct PreparedObjectFilterClause<'a> {
    property_key: &'a str,
    needle: String,
    column: Option<&'a ObjectPropertyColumn>,
    column_matcher: Option<ObjectPropertyContainsMatcher>,
}

#[derive(Debug, Clone)]
pub(crate) struct NativeObjectExportIntent {
    pub(crate) method: &'static str,
    pub(crate) params: serde_json::Value,
}

impl ObjectIndexBins {
    fn build(bounds: &[egui::Rect], bin_size: f32) -> Option<Self> {
        let bin_size = bin_size.max(1.0);
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut any = false;

        for rect in bounds {
            if rect.is_positive() {
                any = true;
                min_x = min_x.min(rect.min.x);
                min_y = min_y.min(rect.min.y);
                max_x = max_x.max(rect.max.x);
                max_y = max_y.max(rect.max.y);
            }
        }
        if !any {
            return None;
        }

        let w = (max_x - min_x).max(1.0);
        let h = (max_y - min_y).max(1.0);
        let bins_w = ((w / bin_size).ceil() as usize).max(1);
        let bins_h = ((h / bin_size).ceil() as usize).max(1);
        let origin = egui::pos2(min_x, min_y);
        let bins_len = bins_w.saturating_mul(bins_h);
        let mut counts = vec![0u32; bins_len];

        for rect in bounds {
            let (x0, y0, x1, y1) = rect_bins(*rect, origin, bin_size, bins_w, bins_h);
            for by in y0..=y1 {
                for bx in x0..=x1 {
                    counts[by * bins_w + bx] = counts[by * bins_w + bx].saturating_add(1);
                }
            }
        }

        let mut offsets = vec![0u32; bins_len];
        let mut total = 0u32;
        for (i, c) in counts.iter().copied().enumerate() {
            offsets[i] = total;
            total = total.saturating_add(c);
        }
        let mut indices = vec![0u32; total as usize];
        let mut cursor = offsets.clone();

        for (idx, rect) in bounds.iter().copied().enumerate() {
            let (x0, y0, x1, y1) = rect_bins(rect, origin, bin_size, bins_w, bins_h);
            for by in y0..=y1 {
                for bx in x0..=x1 {
                    let bi = by * bins_w + bx;
                    let write = cursor[bi] as usize;
                    if write < indices.len() {
                        indices[write] = idx as u32;
                    }
                    cursor[bi] = cursor[bi].saturating_add(1);
                }
            }
        }

        Some(Self {
            origin,
            bin_size,
            bins_w,
            bins_h,
            indices,
            offsets,
            counts,
        })
    }

    pub(super) fn bin_range_for_world_rect(
        &self,
        rect: egui::Rect,
    ) -> (usize, usize, usize, usize) {
        rect_bins(rect, self.origin, self.bin_size, self.bins_w, self.bins_h)
    }

    pub(super) fn bin_slice(&self, bin_index: usize) -> &[u32] {
        let start = self.offsets.get(bin_index).copied().unwrap_or(0) as usize;
        let count = self.counts.get(bin_index).copied().unwrap_or(0) as usize;
        let end = start.saturating_add(count).min(self.indices.len());
        &self.indices[start..end]
    }
}

#[cfg(test)]
#[path = "core/tests.rs"]
mod point_payload_tests;

fn parse_feature_polygons(geom: &serde_json::Value, scale: f32) -> Vec<Vec<egui::Pos2>> {
    let gtype = geom
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    let coords = geom.get("coordinates");
    let mut out = Vec::new();

    match gtype.as_str() {
        "polygon" => {
            if let Some(rings) = coords.and_then(|v| v.as_array()) {
                if let Some(ring) = rings.first() {
                    if let Some(points) = parse_ring_points(ring, scale) {
                        out.push(points);
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
                    if let Some(ring) = rings.first() {
                        if let Some(points) = parse_ring_points(ring, scale) {
                            out.push(points);
                        }
                    }
                }
            }
        }
        _ => {}
    }

    out
}

fn parse_ring_points(node: &serde_json::Value, scale: f32) -> Option<Vec<egui::Pos2>> {
    let arr = node.as_array()?;
    let mut pts = Vec::with_capacity(arr.len().saturating_add(1));
    for p in arr {
        let Some(xy) = p.as_array() else {
            continue;
        };
        if xy.len() < 2 {
            continue;
        }
        let Some(x0) = xy.first().and_then(|v| v.as_f64()) else {
            continue;
        };
        let Some(y0) = xy.get(1).and_then(|v| v.as_f64()) else {
            continue;
        };
        let x = x0 as f32 * scale;
        let y = y0 as f32 * scale;
        if x.is_finite() && y.is_finite() {
            pts.push(egui::pos2(x, y));
        }
    }
    if pts.len() < 3 {
        return None;
    }
    if pts.first() != pts.last() {
        if let Some(first) = pts.first().copied() {
            pts.push(first);
        }
    }
    Some(pts)
}

fn feature_id(
    feat: &serde_json::Value,
    properties: &serde_json::Map<String, serde_json::Value>,
    feature_index: usize,
) -> String {
    if let Some(v) = feat.get("id").and_then(value_to_short_string) {
        return v;
    }
    for key in ["id", "cell_id", "object_id", "label", "name"] {
        if let Some(v) = properties.get(key).and_then(value_to_short_string) {
            return v;
        }
    }
    format!("feature-{}", feature_index + 1)
}

fn value_to_short_string(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(v) => Some(v.clone()),
        serde_json::Value::Number(v) => Some(v.to_string()),
        serde_json::Value::Bool(v) => Some(v.to_string()),
        _ => None,
    }
}

fn value_to_display_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(v) => v.to_string(),
        serde_json::Value::Number(v) => v.to_string(),
        serde_json::Value::String(v) => v.clone(),
        _ => value.to_string(),
    }
}

pub(super) fn normalize_color_value_label(value: &str) -> String {
    value.trim().trim_matches('"').to_ascii_lowercase()
}

fn save_geojson_objects(
    path: &Path,
    objects: &[GeoJsonObjectFeature],
    indices: &[usize],
) -> anyhow::Result<()> {
    let mut features = Vec::with_capacity(indices.len());
    for idx in indices {
        let Some(obj) = objects.get(*idx) else {
            continue;
        };
        features.push(export_object_feature_value(obj));
    }
    let root = serde_json::json!({
        "type": "FeatureCollection",
        "features": features,
    });
    let text = serde_json::to_string_pretty(&root)?;
    std::fs::write(path, text)?;
    Ok(())
}

fn export_object_feature_value(obj: &GeoJsonObjectFeature) -> serde_json::Value {
    let geometry = if obj.polygons_world.len() <= 1 {
        let coords = serde_json::Value::Array(vec![ring_coords_value(
            obj.polygons_world
                .first()
                .map(|p| p.as_slice())
                .unwrap_or(&[]),
        )]);
        serde_json::json!({
            "type": "Polygon",
            "coordinates": coords,
        })
    } else {
        let coords = serde_json::Value::Array(
            obj.polygons_world
                .iter()
                .map(|poly| serde_json::Value::Array(vec![ring_coords_value(poly)]))
                .collect(),
        );
        serde_json::json!({
            "type": "MultiPolygon",
            "coordinates": coords,
        })
    };
    serde_json::json!({
        "type": "Feature",
        "id": obj.id,
        "properties": obj.inline_properties,
        "geometry": geometry,
    })
}

fn ring_coords_value(poly: &[egui::Pos2]) -> serde_json::Value {
    serde_json::Value::Array(
        poly.iter()
            .map(|p| serde_json::Value::Array(vec![serde_json::json!(p.x), serde_json::json!(p.y)]))
            .collect(),
    )
}

#[derive(Debug, Clone)]
struct ObjectExportTable {
    row_count: usize,
    columns: Vec<ExportColumn>,
    geometry_wkb: Vec<Vec<u8>>,
    geometry_types: Vec<String>,
}

#[derive(Debug, Clone)]
struct ExportColumn {
    name: String,
    values: Vec<Option<ExportScalar>>,
}

#[derive(Debug, Clone)]
enum ExportScalar {
    Bool(bool),
    Int64(i64),
    UInt64(u64),
    Float64(f64),
    String(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExportScalarType {
    Bool,
    Int64,
    UInt64,
    Float64,
    String,
}

fn export_scalar_from_json(value: &serde_json::Value) -> ExportScalar {
    match value {
        serde_json::Value::Null => ExportScalar::String(String::new()),
        serde_json::Value::Bool(v) => ExportScalar::Bool(*v),
        serde_json::Value::Number(v) => {
            if let Some(v) = v.as_i64() {
                ExportScalar::Int64(v)
            } else if let Some(v) = v.as_u64() {
                ExportScalar::UInt64(v)
            } else {
                ExportScalar::Float64(v.as_f64().unwrap_or_default())
            }
        }
        serde_json::Value::String(v) => ExportScalar::String(v.clone()),
        _ => ExportScalar::String(value.to_string()),
    }
}

fn export_scalar_from_property_store(
    store: &ObjectPropertyStore,
    key: &str,
    object_index: usize,
) -> Option<ExportScalar> {
    let column = store.loaded_columns.get(key)?;
    match column {
        ObjectPropertyColumn::Bool(values) => values
            .get(object_index)
            .and_then(|value| value.map(ExportScalar::Bool)),
        ObjectPropertyColumn::I64(values) => values
            .get(object_index)
            .and_then(|value| value.map(ExportScalar::Int64)),
        ObjectPropertyColumn::F32(values) => values
            .get(object_index)
            .map(|value| ExportScalar::Float64(f64::from(value))),
        ObjectPropertyColumn::Dictionary { dictionary, values } => values
            .get(object_index)
            .and_then(|code| code.and_then(|code| dictionary.get(code as usize)))
            .map(|value| ExportScalar::String(value.clone())),
        ObjectPropertyColumn::Json(values) => values
            .get(object_index)
            .and_then(|value| value.as_ref())
            .map(export_scalar_from_json),
    }
}

fn numeric_property_value(
    store: &ObjectPropertyStore,
    object_index: usize,
    obj: &ObjectFeature,
    key: &str,
) -> Option<f32> {
    if let Some(column) = store.loaded_columns.get(key) {
        match column {
            ObjectPropertyColumn::I64(values) => {
                return values
                    .get(object_index)
                    .and_then(|value| value.map(|value| value as f32))
                    .filter(|value| value.is_finite());
            }
            ObjectPropertyColumn::F32(values) => {
                return values.get(object_index).filter(|value| value.is_finite());
            }
            ObjectPropertyColumn::Json(values) => {
                if let Some(value) = values
                    .get(object_index)
                    .and_then(|value| value.as_ref())
                    .and_then(numeric_json_value)
                    .filter(|value| value.is_finite())
                {
                    return Some(value);
                }
            }
            ObjectPropertyColumn::Bool(_) | ObjectPropertyColumn::Dictionary { .. } => {}
        }
    }
    obj.inline_properties.get(key).and_then(numeric_json_value)
}

fn export_scalar_to_csv(value: Option<&ExportScalar>) -> String {
    match value {
        None => String::new(),
        Some(ExportScalar::Bool(v)) => v.to_string(),
        Some(ExportScalar::Int64(v)) => v.to_string(),
        Some(ExportScalar::UInt64(v)) => v.to_string(),
        Some(ExportScalar::Float64(v)) => v.to_string(),
        Some(ExportScalar::String(v)) => v.clone(),
    }
}

fn infer_export_scalar_type(values: &[Option<ExportScalar>]) -> ExportScalarType {
    let mut saw_value = false;
    let mut saw_bool = false;
    let mut saw_int = false;
    let mut saw_uint = false;
    let mut saw_float = false;
    let mut saw_string = false;

    for value in values.iter().flatten() {
        saw_value = true;
        match value {
            ExportScalar::Bool(_) => saw_bool = true,
            ExportScalar::Int64(_) => saw_int = true,
            ExportScalar::UInt64(_) => saw_uint = true,
            ExportScalar::Float64(_) => saw_float = true,
            ExportScalar::String(_) => saw_string = true,
        }
    }

    if !saw_value {
        return ExportScalarType::String;
    }
    if saw_string {
        return ExportScalarType::String;
    }
    if saw_bool && !(saw_int || saw_uint || saw_float) {
        return ExportScalarType::Bool;
    }
    if (saw_int || saw_uint || saw_float) && !saw_bool {
        if saw_float || (saw_int && saw_uint) {
            return ExportScalarType::Float64;
        }
        if saw_uint {
            return ExportScalarType::UInt64;
        }
        return ExportScalarType::Int64;
    }
    ExportScalarType::String
}

fn export_column_to_arrow_array(
    column: &ExportColumn,
) -> anyhow::Result<(Field, arrow_array::ArrayRef)> {
    let ty = infer_export_scalar_type(&column.values);
    match ty {
        ExportScalarType::Bool => {
            let mut builder = BooleanBuilder::new();
            for value in &column.values {
                match value {
                    Some(ExportScalar::Bool(v)) => builder.append_value(*v),
                    None => builder.append_null(),
                    Some(other) => builder
                        .append_value(matches!(other, ExportScalar::String(v) if v == "true")),
                }
            }
            Ok((
                Field::new(&column.name, arrow_schema::DataType::Boolean, true),
                Arc::new(builder.finish()) as arrow_array::ArrayRef,
            ))
        }
        ExportScalarType::Int64 => {
            let mut builder = Int64Builder::new();
            for value in &column.values {
                match value {
                    Some(ExportScalar::Int64(v)) => builder.append_value(*v),
                    Some(ExportScalar::UInt64(v)) => builder.append_value(*v as i64),
                    Some(ExportScalar::Float64(v)) => builder.append_value(*v as i64),
                    None => builder.append_null(),
                    Some(ExportScalar::Bool(v)) => builder.append_value(i64::from(*v)),
                    Some(ExportScalar::String(v)) => {
                        builder.append_value(v.parse::<i64>().unwrap_or_default())
                    }
                }
            }
            Ok((
                Field::new(&column.name, arrow_schema::DataType::Int64, true),
                Arc::new(builder.finish()) as arrow_array::ArrayRef,
            ))
        }
        ExportScalarType::UInt64 => {
            let mut builder = UInt64Builder::new();
            for value in &column.values {
                match value {
                    Some(ExportScalar::UInt64(v)) => builder.append_value(*v),
                    Some(ExportScalar::Int64(v)) if *v >= 0 => builder.append_value(*v as u64),
                    Some(ExportScalar::Float64(v)) if *v >= 0.0 => builder.append_value(*v as u64),
                    None => builder.append_null(),
                    Some(ExportScalar::Bool(v)) => builder.append_value(u64::from(*v)),
                    Some(ExportScalar::String(v)) => {
                        builder.append_value(v.parse::<u64>().unwrap_or_default())
                    }
                    _ => builder.append_null(),
                }
            }
            Ok((
                Field::new(&column.name, arrow_schema::DataType::UInt64, true),
                Arc::new(builder.finish()) as arrow_array::ArrayRef,
            ))
        }
        ExportScalarType::Float64 => {
            let mut builder = Float64Builder::new();
            for value in &column.values {
                match value {
                    Some(ExportScalar::Float64(v)) => builder.append_value(*v),
                    Some(ExportScalar::Int64(v)) => builder.append_value(*v as f64),
                    Some(ExportScalar::UInt64(v)) => builder.append_value(*v as f64),
                    None => builder.append_null(),
                    Some(ExportScalar::Bool(v)) => builder.append_value(if *v { 1.0 } else { 0.0 }),
                    Some(ExportScalar::String(v)) => {
                        builder.append_value(v.parse::<f64>().unwrap_or_default())
                    }
                }
            }
            Ok((
                Field::new(&column.name, arrow_schema::DataType::Float64, true),
                Arc::new(builder.finish()) as arrow_array::ArrayRef,
            ))
        }
        ExportScalarType::String => {
            let mut builder = StringBuilder::new();
            for value in &column.values {
                match value {
                    Some(value) => builder.append_value(export_scalar_to_csv(Some(value))),
                    None => builder.append_null(),
                }
            }
            Ok((
                Field::new(&column.name, arrow_schema::DataType::Utf8, true),
                Arc::new(builder.finish()) as arrow_array::ArrayRef,
            ))
        }
    }
}

fn unique_export_name(base: &str, used_names: &mut HashSet<String>) -> String {
    if used_names.insert(base.to_string()) {
        return base.to_string();
    }
    let mut counter = 2usize;
    loop {
        let candidate = format!("{base}_{counter}");
        if used_names.insert(candidate.clone()) {
            return candidate;
        }
        counter += 1;
    }
}

fn object_passes_threshold_rules(
    store: &ObjectPropertyStore,
    object_index: usize,
    obj: &ObjectFeature,
    rules: &[ObjectPropertyThresholdRule],
) -> bool {
    !rules.is_empty()
        && rules.iter().all(|rule| {
            let Some(value) = numeric_property_value(store, object_index, obj, &rule.column_key)
            else {
                return false;
            };
            match rule.op {
                AnalysisThresholdOp::GreaterEqual => value >= rule.value,
                AnalysisThresholdOp::LessEqual => value <= rule.value,
            }
        })
}

fn export_geometry_type_label(obj: &ObjectFeature) -> &'static str {
    if obj.polygons_world.is_empty() {
        "Point"
    } else if obj.polygons_world.len() == 1 {
        "Polygon"
    } else {
        "MultiPolygon"
    }
}

fn encode_object_wkb(obj: &ObjectFeature) -> Vec<u8> {
    if obj.polygons_world.is_empty() {
        return encode_wkb_point(obj.point_position_world.unwrap_or(obj.centroid_world));
    }
    if obj.polygons_world.len() == 1 {
        return encode_wkb_polygon(std::slice::from_ref(&obj.polygons_world[0]));
    }
    encode_wkb_multipolygon(&obj.polygons_world)
}

fn encode_wkb_point(pos: egui::Pos2) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + 4 + 16);
    out.push(1);
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&(pos.x as f64).to_le_bytes());
    out.extend_from_slice(&(pos.y as f64).to_le_bytes());
    out
}

fn encode_wkb_polygon(rings: &[Vec<egui::Pos2>]) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(1);
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(rings.len() as u32).to_le_bytes());
    for ring in rings {
        append_wkb_ring(&mut out, ring);
    }
    out
}

fn encode_wkb_multipolygon(polygons: &[Vec<egui::Pos2>]) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(1);
    out.extend_from_slice(&6u32.to_le_bytes());
    out.extend_from_slice(&(polygons.len() as u32).to_le_bytes());
    for polygon in polygons {
        out.extend_from_slice(&encode_wkb_polygon(std::slice::from_ref(polygon)));
    }
    out
}

fn append_wkb_ring(out: &mut Vec<u8>, ring: &[egui::Pos2]) {
    let mut coords = ring.iter().copied().collect::<Vec<_>>();
    if let (Some(first), Some(last)) = (coords.first().copied(), coords.last().copied())
        && ((first.x - last.x).abs() > f32::EPSILON || (first.y - last.y).abs() > f32::EPSILON)
    {
        coords.push(first);
    }
    out.extend_from_slice(&(coords.len() as u32).to_le_bytes());
    for p in coords {
        out.extend_from_slice(&(p.x as f64).to_le_bytes());
        out.extend_from_slice(&(p.y as f64).to_le_bytes());
    }
}

pub(super) fn union_rects(rects: &[egui::Rect]) -> Option<egui::Rect> {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut any = false;

    for rect in rects {
        if !rect.is_positive() {
            continue;
        }
        any = true;
        min_x = min_x.min(rect.min.x);
        min_y = min_y.min(rect.min.y);
        max_x = max_x.max(rect.max.x);
        max_y = max_y.max(rect.max.y);
    }

    any.then(|| egui::Rect::from_min_max(egui::pos2(min_x, min_y), egui::pos2(max_x, max_y)))
}

pub(super) fn rect_bins(
    rect: egui::Rect,
    origin: egui::Pos2,
    bin_size: f32,
    bins_w: usize,
    bins_h: usize,
) -> (usize, usize, usize, usize) {
    let x0 = ((rect.min.x - origin.x) / bin_size)
        .floor()
        .clamp(0.0, (bins_w.saturating_sub(1)) as f32) as usize;
    let y0 = ((rect.min.y - origin.y) / bin_size)
        .floor()
        .clamp(0.0, (bins_h.saturating_sub(1)) as f32) as usize;
    let x1 = ((rect.max.x - origin.x) / bin_size)
        .floor()
        .clamp(0.0, (bins_w.saturating_sub(1)) as f32) as usize;
    let y1 = ((rect.max.y - origin.y) / bin_size)
        .floor()
        .clamp(0.0, (bins_h.saturating_sub(1)) as f32) as usize;
    (x0, y0, x1, y1)
}
