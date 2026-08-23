mod geometry;
#[cfg(test)]
#[path = "parquet_shapes/tests.rs"]
mod tests;

use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Context;
use arrow_array::Array;
use arrow_schema::DataType;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::Value;

use geometry::{
    append_geom_points, append_geoms, array_value_to_f64, array_value_to_json,
    centroid_summary_from_wkb, circle_polyline, circle_polylines_from_wkb, geometry_array_len,
    geometry_bytes_at, parse_wkb_object_polygons, render_kind_from_wkb,
};

use crate::spatialdata::SpatialDataTransform2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapesRenderKind {
    Lines,
    Points,
    Circles,
}

#[derive(Debug, Clone)]
pub struct ShapesLoadOptions {
    pub transform: SpatialDataTransform2,
    pub geometry_column: String,
    pub property_columns: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct ShapesObjectSchema {
    pub geometry_candidates: Vec<String>,
    pub property_columns: Vec<String>,
    pub numeric_property_columns: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct LoadedShapeObject {
    pub id: String,
    pub polygons_world: Vec<Vec<eframe::egui::Pos2>>,
    pub point_position_world: Option<eframe::egui::Pos2>,
    pub properties: serde_json::Map<String, serde_json::Value>,
    pub source_row_index: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct LoadedPointObject {
    pub id: String,
    pub point_world: eframe::egui::Pos2,
    pub bbox_world: eframe::egui::Rect,
    pub area_px: f32,
    pub perimeter_px: f32,
    pub properties: serde_json::Map<String, serde_json::Value>,
    pub source_row_index: Option<usize>,
}

impl Default for ShapesLoadOptions {
    fn default() -> Self {
        Self {
            transform: SpatialDataTransform2::default(),
            geometry_column: "geometry".to_string(),
            property_columns: None,
        }
    }
}

pub fn inspect_shapes_object_schema(
    shapes_parquet_file: &Path,
) -> anyhow::Result<ShapesObjectSchema> {
    let file = std::fs::File::open(shapes_parquet_file)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let mut geometry_candidates = Vec::new();
    let mut property_columns = Vec::new();
    let mut numeric_property_columns = Vec::new();
    for field in builder.schema().fields() {
        let name = field.name().clone();
        if matches!(field.data_type(), DataType::Binary | DataType::LargeBinary) {
            geometry_candidates.push(name.clone());
        }
        if supports_object_property_type(field.data_type()) {
            property_columns.push(name);
        }
        if supports_numeric_property_type(field.data_type()) {
            numeric_property_columns.push(field.name().clone());
        }
    }
    Ok(ShapesObjectSchema {
        geometry_candidates,
        property_columns,
        numeric_property_columns,
    })
}

pub fn load_shapes_polylines_exterior(
    shapes_parquet_file: &Path,
    options: &ShapesLoadOptions,
) -> anyhow::Result<Vec<Vec<eframe::egui::Pos2>>> {
    let file = std::fs::File::open(shapes_parquet_file)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let projection = ProjectionMask::columns(builder.parquet_schema(), ["geometry"]);
    let mut reader = builder
        .with_projection(projection)
        .with_batch_size(32_768)
        .build()?;

    let mut polylines: Vec<Vec<eframe::egui::Pos2>> = Vec::new();

    while let Some(batch) = reader.next() {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        let schema = batch.schema();
        let geom_i = schema
            .index_of("geometry")
            .context("missing required column 'geometry'")?;
        let geom = batch.column(geom_i).as_ref();
        append_geoms(geom, options, &mut polylines).context("decode geometry column")?;
    }

    Ok(polylines)
}

pub fn detect_shapes_render_kind(shapes_parquet_file: &Path) -> anyhow::Result<ShapesRenderKind> {
    let file = std::fs::File::open(shapes_parquet_file)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let has_radius = builder
        .schema()
        .fields()
        .iter()
        .any(|field| field.name() == "radius");
    let projection = ProjectionMask::columns(builder.parquet_schema(), ["geometry"]);
    let mut reader = builder
        .with_projection(projection)
        .with_batch_size(1024)
        .build()?;

    while let Some(batch) = reader.next() {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        let schema = batch.schema();
        let geom_i = schema
            .index_of("geometry")
            .context("missing required column 'geometry'")?;
        let geom = batch.column(geom_i).as_ref();
        let rows = geometry_array_len(geom)?;
        for row in 0..rows {
            let Some(bytes) = geometry_bytes_at(geom, row) else {
                continue;
            };
            if let Some(kind) = render_kind_from_wkb(bytes, has_radius)? {
                return Ok(kind);
            }
        }
    }

    anyhow::bail!("no supported geometry found in shapes parquet")
}

pub fn shapes_support_object_layer(shapes_parquet_file: &Path) -> anyhow::Result<bool> {
    let file = std::fs::File::open(shapes_parquet_file)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema();
    Ok(schema.fields().iter().any(|field| {
        matches!(
            field.name().as_str(),
            "cell_id" | "instance_id" | "instance_id_polygon" | "label" | "id" | "name"
        )
    }))
}

pub fn load_shapes_points(
    shapes_parquet_file: &Path,
    options: &ShapesLoadOptions,
) -> anyhow::Result<Vec<eframe::egui::Pos2>> {
    let file = std::fs::File::open(shapes_parquet_file)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let projection = ProjectionMask::columns(builder.parquet_schema(), ["geometry"]);
    let mut reader = builder
        .with_projection(projection)
        .with_batch_size(32_768)
        .build()?;

    let mut points = Vec::new();
    while let Some(batch) = reader.next() {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        let schema = batch.schema();
        let geom_i = schema
            .index_of("geometry")
            .context("missing required column 'geometry'")?;
        let geom = batch.column(geom_i).as_ref();
        append_geom_points(geom, options, &mut points).context("decode point geometry column")?;
    }
    Ok(points)
}

pub fn load_shapes_circle_polylines(
    shapes_parquet_file: &Path,
    options: &ShapesLoadOptions,
    segments: usize,
) -> anyhow::Result<Vec<Vec<eframe::egui::Pos2>>> {
    let file = std::fs::File::open(shapes_parquet_file)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let projection = ProjectionMask::columns(builder.parquet_schema(), ["geometry", "radius"]);
    let mut reader = builder
        .with_projection(projection)
        .with_batch_size(16_384)
        .build()?;

    let segs = segments.max(8);
    let mut polylines = Vec::new();
    while let Some(batch) = reader.next() {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        let schema = batch.schema();
        let geom_i = schema
            .index_of("geometry")
            .context("missing required column 'geometry'")?;
        let radius_i = schema
            .index_of("radius")
            .context("missing required column 'radius'")?;
        let geom = batch.column(geom_i).as_ref();
        let radius = batch.column(radius_i).as_ref();
        let rows = geometry_array_len(geom)?;
        for row in 0..rows {
            let Some(bytes) = geometry_bytes_at(geom, row) else {
                continue;
            };
            let Some(radius_world) = array_value_to_f64(radius, row).map(|r| r as f32) else {
                continue;
            };
            if !radius_world.is_finite() || radius_world <= 0.0 {
                continue;
            }
            polylines.extend(circle_polylines_from_wkb(
                bytes,
                radius_world,
                &options.transform,
                segs,
            )?);
        }
    }
    Ok(polylines)
}

pub fn load_shapes_objects(
    shapes_parquet_file: &Path,
    options: &ShapesLoadOptions,
    cancel: &AtomicBool,
) -> anyhow::Result<Vec<LoadedShapeObject>> {
    let file = std::fs::File::open(shapes_parquet_file)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let geometry_column = options.geometry_column.as_str();
    let selected_property_columns = options
        .property_columns
        .as_ref()
        .map(|cols| cols.iter().cloned().collect::<HashSet<_>>());
    let mut projection_cols = vec![geometry_column];
    for field in builder.schema().fields() {
        if field.name() == geometry_column {
            continue;
        }
        if selected_property_columns
            .as_ref()
            .is_some_and(|cols| !cols.contains(field.name().as_str()))
        {
            continue;
        }
        if supports_object_property_type(field.data_type()) {
            projection_cols.push(field.name().as_str());
        }
    }
    let projection = ProjectionMask::columns(builder.parquet_schema(), projection_cols);
    let mut reader = builder
        .with_projection(projection)
        .with_batch_size(16_384)
        .build()?;

    let mut out = Vec::new();
    let mut fallback_index = 0usize;

    while let Some(batch) = reader.next() {
        if cancel.load(Ordering::Relaxed) {
            anyhow::bail!("object load cancelled");
        }
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        let schema = batch.schema();
        let geom_i = schema
            .index_of(geometry_column)
            .with_context(|| format!("missing required geometry column '{geometry_column}'"))?;
        let geom = batch.column(geom_i).as_ref();
        let property_columns = schema
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, field)| field.name() != geometry_column)
            .map(|(idx, field)| (field.name().clone(), batch.column(idx).as_ref()))
            .collect::<Vec<_>>();

        let rows = if let Some(col) = geom.as_any().downcast_ref::<arrow_array::BinaryArray>() {
            col.len()
        } else if let Some(col) = geom
            .as_any()
            .downcast_ref::<arrow_array::LargeBinaryArray>()
        {
            col.len()
        } else {
            anyhow::bail!("unsupported geometry column type (expected binary/largebinary)");
        };

        for row in 0..rows {
            if cancel.load(Ordering::Relaxed) {
                anyhow::bail!("object load cancelled");
            }
            let Some(bytes) = geometry_bytes_at(geom, row) else {
                fallback_index += 1;
                continue;
            };
            let radius_world = property_columns
                .iter()
                .find_map(|(name, col)| (name == "radius").then(|| array_value_to_f64(*col, row)))
                .flatten()
                .map(|v| v as f32);
            let polygons_world =
                parse_wkb_object_polygons(bytes, &options.transform, radius_world)?;
            if polygons_world.is_empty() {
                fallback_index += 1;
                continue;
            }

            let mut properties = serde_json::Map::new();
            for (name, col) in &property_columns {
                if let Some(value) = array_value_to_json(*col, row) {
                    properties.insert(name.clone(), value);
                }
            }
            let id = object_id_from_properties(&properties)
                .unwrap_or_else(|| (fallback_index + 1).to_string());
            properties.insert("id".to_string(), Value::String(id.clone()));
            out.push(LoadedShapeObject {
                id,
                polygons_world,
                point_position_world: None,
                properties,
                source_row_index: Some(fallback_index),
            });
            fallback_index += 1;
        }
    }

    Ok(out)
}

pub fn load_shapes_xy_point_objects(
    shapes_parquet_file: &Path,
    x_column: &str,
    y_column: &str,
    property_columns: Option<&[String]>,
    cancel: &AtomicBool,
) -> anyhow::Result<Vec<LoadedShapeObject>> {
    let file = std::fs::File::open(shapes_parquet_file)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let selected_property_columns =
        property_columns.map(|cols| cols.iter().cloned().collect::<HashSet<_>>());
    let mut projection_cols = vec![x_column, y_column];
    for field in builder.schema().fields() {
        if field.name() == x_column || field.name() == y_column {
            continue;
        }
        if selected_property_columns
            .as_ref()
            .is_some_and(|cols| !cols.contains(field.name().as_str()))
        {
            continue;
        }
        if supports_object_property_type(field.data_type()) {
            projection_cols.push(field.name().as_str());
        }
    }
    let projection = ProjectionMask::columns(builder.parquet_schema(), projection_cols);
    let mut reader = builder
        .with_projection(projection)
        .with_batch_size(16_384)
        .build()?;

    let mut out = Vec::new();
    let mut fallback_index = 0usize;
    while let Some(batch) = reader.next() {
        if cancel.load(Ordering::Relaxed) {
            anyhow::bail!("object load cancelled");
        }
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        let schema = batch.schema();
        let x_i = schema
            .index_of(x_column)
            .with_context(|| format!("missing x column '{x_column}'"))?;
        let y_i = schema
            .index_of(y_column)
            .with_context(|| format!("missing y column '{y_column}'"))?;
        let x_arr = batch.column(x_i).as_ref();
        let y_arr = batch.column(y_i).as_ref();
        let property_columns = schema
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, field)| field.name() != x_column && field.name() != y_column)
            .map(|(idx, field)| (field.name().clone(), batch.column(idx).as_ref()))
            .collect::<Vec<_>>();
        let rows = batch.num_rows();
        for row in 0..rows {
            if cancel.load(Ordering::Relaxed) {
                anyhow::bail!("object load cancelled");
            }
            let Some(x) = array_value_to_f64(x_arr, row).map(|v| v as f32) else {
                fallback_index += 1;
                continue;
            };
            let Some(y) = array_value_to_f64(y_arr, row).map(|v| v as f32) else {
                fallback_index += 1;
                continue;
            };
            if !x.is_finite() || !y.is_finite() {
                fallback_index += 1;
                continue;
            }

            let center = eframe::egui::pos2(x, y);
            let polygons_world = vec![circle_polyline(center, 4.0, 8)];

            let mut properties = serde_json::Map::new();
            properties.insert(x_column.to_string(), Value::from(x));
            properties.insert(y_column.to_string(), Value::from(y));
            for (name, col) in &property_columns {
                if let Some(value) = array_value_to_json(*col, row) {
                    properties.insert(name.clone(), value);
                }
            }
            let id = object_id_from_properties(&properties)
                .unwrap_or_else(|| (fallback_index + 1).to_string());
            properties.insert("id".to_string(), Value::String(id.clone()));
            out.push(LoadedShapeObject {
                id,
                polygons_world,
                point_position_world: Some(center),
                properties,
                source_row_index: Some(fallback_index),
            });
            fallback_index += 1;
        }
    }

    Ok(out)
}

pub fn load_shapes_xy_point_features(
    shapes_parquet_file: &Path,
    x_column: &str,
    y_column: &str,
    property_columns: Option<&[String]>,
    cancel: &AtomicBool,
) -> anyhow::Result<Vec<LoadedPointObject>> {
    let file = std::fs::File::open(shapes_parquet_file)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let selected_property_columns =
        property_columns.map(|cols| cols.iter().cloned().collect::<HashSet<_>>());
    let mut projection_cols = vec![x_column, y_column];
    for field in builder.schema().fields() {
        if field.name() == x_column || field.name() == y_column {
            continue;
        }
        if selected_property_columns
            .as_ref()
            .is_some_and(|cols| !cols.contains(field.name().as_str()))
        {
            continue;
        }
        if supports_object_property_type(field.data_type()) {
            projection_cols.push(field.name().as_str());
        }
    }
    let projection = ProjectionMask::columns(builder.parquet_schema(), projection_cols);
    let mut reader = builder
        .with_projection(projection)
        .with_batch_size(16_384)
        .build()?;

    let mut out = Vec::new();
    let mut fallback_index = 0usize;
    while let Some(batch) = reader.next() {
        if cancel.load(Ordering::Relaxed) {
            anyhow::bail!("object load cancelled");
        }
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        let schema = batch.schema();
        let x_i = schema
            .index_of(x_column)
            .with_context(|| format!("missing x column '{x_column}'"))?;
        let y_i = schema
            .index_of(y_column)
            .with_context(|| format!("missing y column '{y_column}'"))?;
        let x_arr = batch.column(x_i).as_ref();
        let y_arr = batch.column(y_i).as_ref();
        let property_columns = schema
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, field)| field.name() != x_column && field.name() != y_column)
            .map(|(idx, field)| (field.name().clone(), batch.column(idx).as_ref()))
            .collect::<Vec<_>>();
        let rows = batch.num_rows();
        for row in 0..rows {
            if cancel.load(Ordering::Relaxed) {
                anyhow::bail!("object load cancelled");
            }
            let Some(x) = array_value_to_f64(x_arr, row).map(|v| v as f32) else {
                fallback_index += 1;
                continue;
            };
            let Some(y) = array_value_to_f64(y_arr, row).map(|v| v as f32) else {
                fallback_index += 1;
                continue;
            };
            if !x.is_finite() || !y.is_finite() {
                fallback_index += 1;
                continue;
            }

            let center = eframe::egui::pos2(x, y);
            let mut properties = serde_json::Map::new();
            properties.insert(x_column.to_string(), Value::from(x));
            properties.insert(y_column.to_string(), Value::from(y));
            for (name, col) in &property_columns {
                if let Some(value) = array_value_to_json(*col, row) {
                    properties.insert(name.clone(), value);
                }
            }
            let radius_world = properties
                .get("radius")
                .and_then(|value| value.as_f64())
                .map(|value| value as f32)
                .filter(|value| value.is_finite() && *value > 0.0)
                .unwrap_or(1.0);
            let area_px = std::f32::consts::PI * radius_world * radius_world;
            let perimeter_px = std::f32::consts::TAU * radius_world;
            let bbox_world = eframe::egui::Rect::from_center_size(
                center,
                eframe::egui::Vec2::splat(radius_world),
            );
            let id = object_id_from_properties(&properties)
                .unwrap_or_else(|| (fallback_index + 1).to_string());
            properties.insert("id".to_string(), Value::String(id.clone()));
            out.push(LoadedPointObject {
                id,
                point_world: center,
                bbox_world,
                area_px,
                perimeter_px,
                properties,
                source_row_index: Some(fallback_index),
            });
            fallback_index += 1;
        }
    }

    Ok(out)
}

pub fn load_shapes_centroid_point_objects(
    shapes_parquet_file: &Path,
    options: &ShapesLoadOptions,
    cancel: &AtomicBool,
) -> anyhow::Result<Vec<LoadedPointObject>> {
    let file = std::fs::File::open(shapes_parquet_file)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let geometry_column = options.geometry_column.as_str();
    let selected_property_columns = options
        .property_columns
        .as_ref()
        .map(|cols| cols.iter().cloned().collect::<HashSet<_>>());
    let mut projection_cols = vec![geometry_column];
    for field in builder.schema().fields() {
        if field.name() == geometry_column {
            continue;
        }
        if selected_property_columns
            .as_ref()
            .is_some_and(|cols| !cols.contains(field.name().as_str()))
        {
            continue;
        }
        if supports_object_property_type(field.data_type()) {
            projection_cols.push(field.name().as_str());
        }
    }
    let projection = ProjectionMask::columns(builder.parquet_schema(), projection_cols);
    let mut reader = builder
        .with_projection(projection)
        .with_batch_size(16_384)
        .build()?;

    let mut out = Vec::new();
    let mut fallback_index = 0usize;

    while let Some(batch) = reader.next() {
        if cancel.load(Ordering::Relaxed) {
            anyhow::bail!("object load cancelled");
        }
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        let schema = batch.schema();
        let geom_i = schema
            .index_of(geometry_column)
            .with_context(|| format!("missing required geometry column '{geometry_column}'"))?;
        let geom = batch.column(geom_i).as_ref();
        let property_columns = schema
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, field)| field.name() != geometry_column)
            .map(|(idx, field)| (field.name().clone(), batch.column(idx).as_ref()))
            .collect::<Vec<_>>();
        let rows = geometry_array_len(geom)?;
        for row in 0..rows {
            if cancel.load(Ordering::Relaxed) {
                anyhow::bail!("object load cancelled");
            }
            let Some(bytes) = geometry_bytes_at(geom, row) else {
                fallback_index += 1;
                continue;
            };
            let radius_world = property_columns
                .iter()
                .find_map(|(name, col)| (name == "radius").then(|| array_value_to_f64(*col, row)))
                .flatten()
                .map(|v| v as f32);
            let Some(summary) = centroid_summary_from_wkb(bytes, &options.transform, radius_world)?
            else {
                fallback_index += 1;
                continue;
            };

            let mut properties = serde_json::Map::new();
            for (name, col) in &property_columns {
                if let Some(value) = array_value_to_json(*col, row) {
                    properties.insert(name.clone(), value);
                }
            }
            let id = object_id_from_properties(&properties)
                .unwrap_or_else(|| (fallback_index + 1).to_string());
            properties.insert("id".to_string(), Value::String(id.clone()));
            out.push(LoadedPointObject {
                id,
                point_world: summary.centroid_world,
                bbox_world: summary.bbox_world,
                area_px: summary.area_px,
                perimeter_px: summary.perimeter_px,
                properties,
                source_row_index: Some(fallback_index),
            });
            fallback_index += 1;
        }
    }

    Ok(out)
}

pub fn load_shapes_property_values_by_row(
    shapes_parquet_file: &Path,
    property_key: &str,
    cancel: &AtomicBool,
) -> anyhow::Result<std::collections::HashMap<usize, serde_json::Value>> {
    let file = std::fs::File::open(shapes_parquet_file)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema();
    let field = schema
        .fields()
        .iter()
        .find(|field| field.name() == property_key)
        .with_context(|| format!("missing property column '{property_key}'"))?;
    if !supports_object_property_type(field.data_type()) {
        anyhow::bail!("unsupported property column type for '{property_key}'");
    }
    let projection = ProjectionMask::columns(builder.parquet_schema(), [property_key]);
    let mut reader = builder
        .with_projection(projection)
        .with_batch_size(16_384)
        .build()?;

    let mut out = std::collections::HashMap::new();
    let mut row_index = 0usize;
    while let Some(batch) = reader.next() {
        if cancel.load(Ordering::Relaxed) {
            anyhow::bail!("property load cancelled");
        }
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        let schema = batch.schema();
        let prop_i = schema
            .index_of(property_key)
            .with_context(|| format!("missing property column '{property_key}'"))?;
        let prop = batch.column(prop_i).as_ref();
        for row in 0..batch.num_rows() {
            if cancel.load(Ordering::Relaxed) {
                anyhow::bail!("property load cancelled");
            }
            if let Some(value) = array_value_to_json(prop, row) {
                out.insert(row_index, value);
            }
            row_index += 1;
        }
    }
    Ok(out)
}

fn supports_object_property_type(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Utf8
            | DataType::LargeUtf8
    )
}

fn supports_numeric_property_type(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
    )
}

fn object_id_from_properties(properties: &serde_json::Map<String, Value>) -> Option<String> {
    for key in [
        "id",
        "instance_id",
        "instance_id_polygon",
        "cell_id",
        "label",
        "name",
        "polygon_name",
    ] {
        if let Some(value) = properties.get(key) {
            match value {
                Value::String(v) => return Some(v.clone()),
                other => return Some(other.to_string()),
            }
        }
    }
    None
}
