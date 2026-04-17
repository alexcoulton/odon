use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Context;
use arrow_array::Array;
use arrow_array::{
    BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, LargeStringArray,
    StringArray, UInt32Array, UInt64Array,
};
use arrow_schema::DataType;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::Value;

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
            let mut cur = Cursor::new(bytes);
            let geom = read_geom(&mut cur)?;
            let kind = match classify_geom_kind(&geom) {
                GeomKind::Pointish => {
                    if has_radius {
                        ShapesRenderKind::Circles
                    } else {
                        ShapesRenderKind::Points
                    }
                }
                GeomKind::Linear | GeomKind::Polygonal => ShapesRenderKind::Lines,
                GeomKind::Unsupported => continue,
            };
            return Ok(kind);
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
            let mut cur = Cursor::new(bytes);
            let geom = read_geom(&mut cur)?;
            let mut centers = Vec::new();
            flatten_geom_points(&geom, &options.transform, &mut centers);
            for center in centers {
                polylines.push(circle_polyline(center, radius_world, segs));
            }
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
            let polygons_world = vec![circle_polyline(center, 4.0, 24)];

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

fn parse_wkb_object_polygons(
    bytes: &[u8],
    xform: &SpatialDataTransform2,
    radius_world: Option<f32>,
) -> anyhow::Result<Vec<Vec<eframe::egui::Pos2>>> {
    let mut cur = Cursor::new(bytes);
    let geom = read_geom(&mut cur)?;
    let mut out = Vec::new();
    flatten_geom_object_polygons(&geom, xform, radius_world, &mut out);
    Ok(out)
}

fn geometry_bytes_at<'a>(geom: &'a dyn Array, row: usize) -> Option<&'a [u8]> {
    if let Some(col) = geom.as_any().downcast_ref::<arrow_array::BinaryArray>() {
        return (!col.is_null(row)).then(|| col.value(row));
    }
    if let Some(col) = geom
        .as_any()
        .downcast_ref::<arrow_array::LargeBinaryArray>()
    {
        return (!col.is_null(row)).then(|| col.value(row));
    }
    None
}

fn geometry_array_len(geom: &dyn Array) -> anyhow::Result<usize> {
    if let Some(col) = geom.as_any().downcast_ref::<arrow_array::BinaryArray>() {
        return Ok(col.len());
    }
    if let Some(col) = geom
        .as_any()
        .downcast_ref::<arrow_array::LargeBinaryArray>()
    {
        return Ok(col.len());
    }
    anyhow::bail!("unsupported geometry column type (expected binary/largebinary)");
}

fn array_value_to_json(arr: &dyn Array, row: usize) -> Option<Value> {
    if row >= arr.len() || arr.is_null(row) {
        return None;
    }
    if let Some(col) = arr.as_any().downcast_ref::<Int32Array>() {
        return Some(Value::from(col.value(row)));
    }
    if let Some(col) = arr.as_any().downcast_ref::<Int64Array>() {
        return Some(Value::from(col.value(row)));
    }
    if let Some(col) = arr.as_any().downcast_ref::<arrow_array::Int8Array>() {
        return Some(Value::from(col.value(row) as i64));
    }
    if let Some(col) = arr.as_any().downcast_ref::<arrow_array::Int16Array>() {
        return Some(Value::from(col.value(row) as i64));
    }
    if let Some(col) = arr.as_any().downcast_ref::<UInt32Array>() {
        return Some(Value::from(col.value(row)));
    }
    if let Some(col) = arr.as_any().downcast_ref::<UInt64Array>() {
        return Some(Value::from(col.value(row)));
    }
    if let Some(col) = arr.as_any().downcast_ref::<arrow_array::UInt8Array>() {
        return Some(Value::from(col.value(row) as u64));
    }
    if let Some(col) = arr.as_any().downcast_ref::<arrow_array::UInt16Array>() {
        return Some(Value::from(col.value(row) as u64));
    }
    if let Some(col) = arr.as_any().downcast_ref::<Float32Array>() {
        return serde_json::Number::from_f64(col.value(row) as f64).map(Value::Number);
    }
    if let Some(col) = arr.as_any().downcast_ref::<Float64Array>() {
        return serde_json::Number::from_f64(col.value(row)).map(Value::Number);
    }
    if let Some(col) = arr.as_any().downcast_ref::<BooleanArray>() {
        return Some(Value::Bool(col.value(row)));
    }
    if let Some(col) = arr.as_any().downcast_ref::<StringArray>() {
        return Some(Value::String(col.value(row).to_string()));
    }
    if let Some(col) = arr.as_any().downcast_ref::<LargeStringArray>() {
        return Some(Value::String(col.value(row).to_string()));
    }
    None
}

fn array_value_to_f64(arr: &dyn Array, row: usize) -> Option<f64> {
    if row >= arr.len() || arr.is_null(row) {
        return None;
    }
    if let Some(col) = arr.as_any().downcast_ref::<Int32Array>() {
        return Some(col.value(row) as f64);
    }
    if let Some(col) = arr.as_any().downcast_ref::<Int64Array>() {
        return Some(col.value(row) as f64);
    }
    if let Some(col) = arr.as_any().downcast_ref::<arrow_array::Int8Array>() {
        return Some(col.value(row) as f64);
    }
    if let Some(col) = arr.as_any().downcast_ref::<arrow_array::Int16Array>() {
        return Some(col.value(row) as f64);
    }
    if let Some(col) = arr.as_any().downcast_ref::<UInt32Array>() {
        return Some(col.value(row) as f64);
    }
    if let Some(col) = arr.as_any().downcast_ref::<UInt64Array>() {
        return Some(col.value(row) as f64);
    }
    if let Some(col) = arr.as_any().downcast_ref::<arrow_array::UInt8Array>() {
        return Some(col.value(row) as f64);
    }
    if let Some(col) = arr.as_any().downcast_ref::<arrow_array::UInt16Array>() {
        return Some(col.value(row) as f64);
    }
    if let Some(col) = arr.as_any().downcast_ref::<Float32Array>() {
        return Some(col.value(row) as f64);
    }
    if let Some(col) = arr.as_any().downcast_ref::<Float64Array>() {
        return Some(col.value(row));
    }
    None
}

fn append_geoms(
    geom: &dyn Array,
    options: &ShapesLoadOptions,
    out: &mut Vec<Vec<eframe::egui::Pos2>>,
) -> anyhow::Result<()> {
    if let Some(col) = geom.as_any().downcast_ref::<arrow_array::BinaryArray>() {
        for i in 0..col.len() {
            if col.is_null(i) {
                continue;
            }
            let bytes = col.value(i);
            out.extend(parse_wkb_polylines_exterior(bytes, &options.transform)?);
        }
        return Ok(());
    }
    if let Some(col) = geom
        .as_any()
        .downcast_ref::<arrow_array::LargeBinaryArray>()
    {
        for i in 0..col.len() {
            if col.is_null(i) {
                continue;
            }
            let bytes = col.value(i);
            out.extend(parse_wkb_polylines_exterior(bytes, &options.transform)?);
        }
        return Ok(());
    }

    anyhow::bail!("unsupported geometry column type (expected binary/largebinary)")
}

fn append_geom_points(
    geom: &dyn Array,
    options: &ShapesLoadOptions,
    out: &mut Vec<eframe::egui::Pos2>,
) -> anyhow::Result<()> {
    if let Some(col) = geom.as_any().downcast_ref::<arrow_array::BinaryArray>() {
        for i in 0..col.len() {
            if col.is_null(i) {
                continue;
            }
            let bytes = col.value(i);
            let mut cur = Cursor::new(bytes);
            let geom = read_geom(&mut cur)?;
            flatten_geom_points(&geom, &options.transform, out);
        }
        return Ok(());
    }
    if let Some(col) = geom
        .as_any()
        .downcast_ref::<arrow_array::LargeBinaryArray>()
    {
        for i in 0..col.len() {
            if col.is_null(i) {
                continue;
            }
            let bytes = col.value(i);
            let mut cur = Cursor::new(bytes);
            let geom = read_geom(&mut cur)?;
            flatten_geom_points(&geom, &options.transform, out);
        }
        return Ok(());
    }
    anyhow::bail!("unsupported geometry column type (expected binary/largebinary)")
}

fn parse_wkb_polylines_exterior(
    bytes: &[u8],
    xform: &SpatialDataTransform2,
) -> anyhow::Result<Vec<Vec<eframe::egui::Pos2>>> {
    let mut cur = Cursor::new(bytes);
    let geom = read_geom(&mut cur)?;
    let mut out = Vec::new();
    flatten_geom_exterior(&geom, xform, &mut out);
    Ok(out)
}

#[derive(Debug)]
enum Geom {
    Point { pt: [f64; 2] },
    MultiPoint { pts: Vec<[f64; 2]> },
    Polygon { rings: Vec<Vec<[f64; 2]>> },
    MultiPolygon { polys: Vec<Geom> },
    LineString { pts: Vec<[f64; 2]> },
    MultiLineString { lines: Vec<Vec<[f64; 2]>> },
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GeomKind {
    Pointish,
    Linear,
    Polygonal,
    Unsupported,
}

fn flatten_geom_exterior(
    geom: &Geom,
    xform: &SpatialDataTransform2,
    out: &mut Vec<Vec<eframe::egui::Pos2>>,
) {
    match geom {
        Geom::Point { .. } | Geom::MultiPoint { .. } => {}
        Geom::Polygon { rings } => {
            if let Some(r0) = rings.first() {
                let pts = r0
                    .iter()
                    .map(|p| {
                        let q = xform.apply([p[0] as f32, p[1] as f32]);
                        eframe::egui::pos2(q[0], q[1])
                    })
                    .collect::<Vec<_>>();
                if pts.len() >= 2 {
                    out.push(pts);
                }
            }
        }
        Geom::MultiPolygon { polys } => {
            for g in polys {
                flatten_geom_exterior(g, xform, out);
            }
        }
        Geom::LineString { pts } => {
            let pts = pts
                .iter()
                .map(|p| {
                    let q = xform.apply([p[0] as f32, p[1] as f32]);
                    eframe::egui::pos2(q[0], q[1])
                })
                .collect::<Vec<_>>();
            if pts.len() >= 2 {
                out.push(pts);
            }
        }
        Geom::MultiLineString { lines } => {
            for l in lines {
                let pts = l
                    .iter()
                    .map(|p| {
                        let q = xform.apply([p[0] as f32, p[1] as f32]);
                        eframe::egui::pos2(q[0], q[1])
                    })
                    .collect::<Vec<_>>();
                if pts.len() >= 2 {
                    out.push(pts);
                }
            }
        }
        Geom::Unsupported => {}
    }
}

fn flatten_geom_object_polygons(
    geom: &Geom,
    xform: &SpatialDataTransform2,
    radius_world: Option<f32>,
    out: &mut Vec<Vec<eframe::egui::Pos2>>,
) {
    match geom {
        Geom::Point { pt } => {
            let r = radius_world.unwrap_or(4.0).max(1e-3);
            out.push(circle_polyline_transformed(*pt, r, xform, 24));
        }
        Geom::MultiPoint { pts } => {
            let r = radius_world.unwrap_or(4.0).max(1e-3);
            for &pt in pts {
                out.push(circle_polyline_transformed(pt, r, xform, 24));
            }
        }
        Geom::Polygon { .. } | Geom::MultiPolygon { .. } => {
            flatten_geom_exterior(geom, xform, out);
        }
        Geom::LineString { .. } | Geom::MultiLineString { .. } | Geom::Unsupported => {}
    }
}

fn flatten_geom_points(
    geom: &Geom,
    xform: &SpatialDataTransform2,
    out: &mut Vec<eframe::egui::Pos2>,
) {
    match geom {
        Geom::Point { pt } => {
            let q = xform.apply([pt[0] as f32, pt[1] as f32]);
            out.push(eframe::egui::pos2(q[0], q[1]));
        }
        Geom::MultiPoint { pts } => {
            for pt in pts {
                let q = xform.apply([pt[0] as f32, pt[1] as f32]);
                out.push(eframe::egui::pos2(q[0], q[1]));
            }
        }
        Geom::MultiPolygon { polys } => {
            for g in polys {
                flatten_geom_points(g, xform, out);
            }
        }
        _ => {}
    }
}

fn classify_geom_kind(geom: &Geom) -> GeomKind {
    match geom {
        Geom::Point { .. } | Geom::MultiPoint { .. } => GeomKind::Pointish,
        Geom::Polygon { .. } | Geom::MultiPolygon { .. } => GeomKind::Polygonal,
        Geom::LineString { .. } | Geom::MultiLineString { .. } => GeomKind::Linear,
        Geom::Unsupported => GeomKind::Unsupported,
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    i: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, i: 0 }
    }

    fn take(&mut self, n: usize) -> anyhow::Result<&'a [u8]> {
        if self.i + n > self.bytes.len() {
            anyhow::bail!("unexpected end of WKB");
        }
        let out = &self.bytes[self.i..self.i + n];
        self.i += n;
        Ok(out)
    }

    fn u8(&mut self) -> anyhow::Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self, le: bool) -> anyhow::Result<u32> {
        let b = self.take(4)?;
        Ok(if le {
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        } else {
            u32::from_be_bytes([b[0], b[1], b[2], b[3]])
        })
    }

    fn f64(&mut self, le: bool) -> anyhow::Result<f64> {
        let b = self.take(8)?;
        Ok(if le {
            f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
        } else {
            f64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
        })
    }
}

fn read_geom(cur: &mut Cursor<'_>) -> anyhow::Result<Geom> {
    let endian = cur.u8()?;
    let le = match endian {
        0 => false,
        1 => true,
        _ => anyhow::bail!("invalid WKB endian byte: {endian}"),
    };

    let raw_ty = cur.u32(le)?;
    // EWKB flags: strip Z/M/SRID bits if present.
    let ty = raw_ty & 0x1FFF_FFFF;
    // OGC WKB uses +1000/+2000/+3000 for Z/M/ZM.
    let base = ty % 1000;
    let has_z = (ty >= 1000) || (raw_ty & 0x8000_0000) != 0;
    let coords = if has_z { 3 } else { 2 };

    match base {
        1 => {
            let x = cur.f64(le)?;
            let y = cur.f64(le)?;
            if coords == 3 {
                let _ = cur.f64(le)?;
            }
            Ok(Geom::Point { pt: [x, y] })
        }
        2 => {
            // LineString
            let n = cur.u32(le)? as usize;
            let mut pts = Vec::with_capacity(n);
            for _ in 0..n {
                let x = cur.f64(le)?;
                let y = cur.f64(le)?;
                if coords == 3 {
                    let _z = cur.f64(le)?;
                    let _ = _z;
                }
                pts.push([x, y]);
            }
            Ok(Geom::LineString { pts })
        }
        3 => {
            // Polygon
            let rings_n = cur.u32(le)? as usize;
            let mut rings: Vec<Vec<[f64; 2]>> = Vec::with_capacity(rings_n);
            for _ in 0..rings_n {
                let n = cur.u32(le)? as usize;
                let mut pts = Vec::with_capacity(n);
                for _ in 0..n {
                    let x = cur.f64(le)?;
                    let y = cur.f64(le)?;
                    if coords == 3 {
                        let _z = cur.f64(le)?;
                        let _ = _z;
                    }
                    pts.push([x, y]);
                }
                rings.push(pts);
            }
            Ok(Geom::Polygon { rings })
        }
        5 => {
            // MultiLineString
            let n = cur.u32(le)? as usize;
            let mut lines = Vec::with_capacity(n);
            for _ in 0..n {
                let g = read_geom(cur)?;
                if let Geom::LineString { pts } = g {
                    lines.push(pts);
                }
            }
            Ok(Geom::MultiLineString { lines })
        }
        4 => {
            // MultiPoint
            let n = cur.u32(le)? as usize;
            let mut pts = Vec::with_capacity(n);
            for _ in 0..n {
                match read_geom(cur)? {
                    Geom::Point { pt } => pts.push(pt),
                    _ => {}
                }
            }
            Ok(Geom::MultiPoint { pts })
        }
        6 => {
            // MultiPolygon
            let n = cur.u32(le)? as usize;
            let mut polys = Vec::with_capacity(n);
            for _ in 0..n {
                polys.push(read_geom(cur)?);
            }
            Ok(Geom::MultiPolygon { polys })
        }
        _ => Ok(Geom::Unsupported),
    }
}

fn circle_polyline(
    center: eframe::egui::Pos2,
    radius_world: f32,
    segments: usize,
) -> Vec<eframe::egui::Pos2> {
    let n = segments.max(8);
    let mut pts = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let t = (i as f32) * std::f32::consts::TAU / (n as f32);
        pts.push(eframe::egui::pos2(
            center.x + radius_world * t.cos(),
            center.y + radius_world * t.sin(),
        ));
    }
    pts
}

fn circle_polyline_transformed(
    center: [f64; 2],
    radius: f32,
    xform: &SpatialDataTransform2,
    segments: usize,
) -> Vec<eframe::egui::Pos2> {
    let n = segments.max(8);
    let mut pts = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let t = (i as f32) * std::f32::consts::TAU / (n as f32);
        let src = [
            center[0] as f32 + radius * t.cos(),
            center[1] as f32 + radius * t.sin(),
        ];
        let q = xform.apply(src);
        pts.push(eframe::egui::pos2(q[0], q[1]));
    }
    pts
}
