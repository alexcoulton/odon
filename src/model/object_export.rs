use std::collections::{BTreeSet, HashSet};
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow_array::RecordBatch;
use arrow_array::builder::{
    BinaryBuilder, BooleanBuilder, Float64Builder, Int64Builder, StringBuilder, UInt64Builder,
};
use arrow_schema::{Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use serde_json::{Value, json};

use crate::control::{ControlError, ControlErrorKind};

use super::{ControlObjectFeature, ControlObjectResource, ObjectTarget};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ObjectExportFormat {
    Csv,
    GeoParquet,
}

impl ObjectExportFormat {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Csv => "csv",
            Self::GeoParquet => "geoparquet",
        }
    }
}

#[derive(Clone)]
pub(crate) struct ObjectExportSpec {
    pub(crate) document_generation: u64,
    pub(crate) resource_generation: u64,
    pub(crate) operation_generation: u64,
    pub(crate) target: ObjectTarget,
    pub(crate) path: PathBuf,
    pub(crate) overwrite: bool,
    pub(crate) format: ObjectExportFormat,
    pub(crate) scope: String,
    pub(crate) resource: Arc<ControlObjectResource>,
    pub(crate) row_indices: Arc<Vec<usize>>,
    pub(crate) columns: Arc<Vec<String>>,
    pub(crate) selected_indices: Arc<HashSet<usize>>,
    pub(crate) analysis_state: Value,
}

#[derive(Debug, Clone)]
pub(crate) struct ObjectExportResult {
    pub(crate) bytes: u64,
    pub(crate) object_count: usize,
    pub(crate) column_count: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct ObjectExportModel {
    generation: u64,
    running: bool,
    status: String,
    last_output: Option<Value>,
}

impl Default for ObjectExportModel {
    fn default() -> Self {
        Self {
            generation: 1,
            running: false,
            status: String::new(),
            last_output: None,
        }
    }
}

impl ObjectExportModel {
    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn begin(&mut self, path: &Path, object_count: usize) -> Result<u64, ControlError> {
        if self.running {
            return Err(invalid("an object export is already in progress"));
        }
        self.generation = self.generation.wrapping_add(1).max(1);
        self.running = true;
        self.status = format!(
            "Exporting {object_count} object(s) to {}",
            path.to_string_lossy()
        );
        Ok(self.generation)
    }

    pub(crate) fn finish(
        &mut self,
        generation: u64,
        path: &Path,
        format: ObjectExportFormat,
        result: &ObjectExportResult,
    ) -> Option<Value> {
        if !self.running || self.generation != generation {
            return None;
        }
        self.running = false;
        self.status = format!(
            "Exported {} object(s) to {}",
            result.object_count,
            path.to_string_lossy()
        );
        self.generation = self.generation.wrapping_add(1).max(1);
        let output = json!({
            "path":path.to_string_lossy(),
            "format":format.as_str(),
            "object_count":result.object_count,
            "column_count":result.column_count,
            "bytes":result.bytes,
        });
        self.last_output = Some(output.clone());
        Some(output)
    }

    pub(crate) fn fail(&mut self, generation: u64, message: impl Into<String>) -> bool {
        if !self.running || self.generation != generation {
            return false;
        }
        self.running = false;
        self.status = message.into();
        self.generation = self.generation.wrapping_add(1).max(1);
        true
    }

    pub(crate) fn snapshot(&self) -> Value {
        json!({
            "running":self.running,
            "status":self.status,
            "request_id":self.generation,
            "generation":self.generation,
            "last_output":self.last_output,
        })
    }
}

pub(crate) fn object_export_columns(
    resource: &ControlObjectResource,
    analysis_state: &Value,
) -> Vec<String> {
    let mut columns = resource.property_names.as_ref().clone();
    if !columns.iter().any(|column| column == "id") {
        columns.push("id".to_string());
    }
    columns.sort();
    columns.dedup();
    let mut used = columns.iter().cloned().collect::<HashSet<_>>();
    push_unique(&mut columns, &mut used, "_odon_geometry_type");
    push_unique(&mut columns, &mut used, "_odon_centroid_x");
    push_unique(&mut columns, &mut used, "_odon_centroid_y");
    if resource
        .features
        .iter()
        .any(|feature| feature.point_position_world.is_some())
    {
        push_unique(&mut columns, &mut used, "_odon_point_x");
        push_unique(&mut columns, &mut used, "_odon_point_y");
    }
    push_unique(&mut columns, &mut used, "_odon_area_px");
    push_unique(&mut columns, &mut used, "_odon_perimeter_px");
    push_unique(&mut columns, &mut used, "_odon_selected");

    let calls = analysis_state
        .get("threshold_elements")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    if let Some(selected) = analysis_state
        .get("threshold_selected_element")
        .and_then(Value::as_u64)
        .and_then(|index| usize::try_from(index).ok())
        .and_then(|index| calls.get(index))
    {
        let name = live_call_column_name(selected, analysis_state);
        push_unique(&mut columns, &mut used, &name);
    }
    for call in calls {
        let name = threshold_call_column_name(call);
        push_unique(&mut columns, &mut used, &name);
    }
    for selection in analysis_state
        .get("selection_elements")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
    {
        let name = format!(
            "_odon_selection_{}",
            sanitize_export_key(
                selection
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("unnamed")
            )
        );
        push_unique(&mut columns, &mut used, &name);
    }
    columns
}

pub(crate) fn write_object_export(
    spec: &ObjectExportSpec,
    cancelled: impl Fn() -> bool,
) -> anyhow::Result<ObjectExportResult> {
    anyhow::ensure!(!cancelled(), "object export was cancelled");
    let parent = spec.path.parent().unwrap_or_else(|| Path::new("."));
    anyhow::ensure!(parent.is_dir(), "export directory does not exist");
    let temp_path = temporary_sibling(&spec.path);
    let result = (|| {
        let table = build_table(spec, &cancelled)?;
        match spec.format {
            ObjectExportFormat::Csv => write_csv(&temp_path, &table, &cancelled)?,
            ObjectExportFormat::GeoParquet => write_geoparquet(&temp_path, &table, &cancelled)?,
        }
        anyhow::ensure!(!cancelled(), "object export was cancelled");
        commit_temp(&temp_path, &spec.path, spec.overwrite)?;
        let bytes = fs::metadata(&spec.path)?.len();
        Ok(ObjectExportResult {
            bytes,
            object_count: table.row_count,
            column_count: table.columns.len(),
        })
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    result
}

#[derive(Debug, Clone)]
struct ExportTable {
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

fn build_table(
    spec: &ObjectExportSpec,
    cancelled: &impl Fn() -> bool,
) -> anyhow::Result<ExportTable> {
    let available = object_export_columns(&spec.resource, &spec.analysis_state);
    let selected = spec.columns.iter().cloned().collect::<HashSet<_>>();
    let mut columns = spec
        .columns
        .iter()
        .map(|name| ExportColumn {
            name: name.clone(),
            values: Vec::with_capacity(spec.row_indices.len()),
        })
        .collect::<Vec<_>>();
    anyhow::ensure!(
        selected.iter().all(|name| available.contains(name)),
        "export column set changed before execution"
    );
    let derived = derived_column_map(&spec.resource, &spec.analysis_state);
    let mut geometry_wkb = Vec::with_capacity(spec.row_indices.len());
    let mut geometry_types = BTreeSet::new();
    for (position, index) in spec.row_indices.iter().copied().enumerate() {
        if position % 256 == 0 {
            anyhow::ensure!(!cancelled(), "object export was cancelled");
        }
        let feature = spec
            .resource
            .features
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("object export row is out of range"))?;
        let geometry_type = geometry_type(feature);
        geometry_types.insert(geometry_type.to_string());
        geometry_wkb.push(encode_wkb(feature));
        for column in &mut columns {
            column.values.push(export_value(
                &column.name,
                index,
                feature,
                &spec.resource,
                &spec.selected_indices,
                &derived,
            ));
        }
    }
    Ok(ExportTable {
        row_count: spec.row_indices.len(),
        columns,
        geometry_wkb,
        geometry_types: geometry_types.into_iter().collect(),
    })
}

#[derive(Debug, Clone)]
enum DerivedColumn {
    GeometryType,
    CentroidX,
    CentroidY,
    PointX,
    PointY,
    Area,
    Perimeter,
    Selected,
    Call(Value),
    FailedCall,
    NamedSelection(HashSet<String>),
}

fn derived_column_map(
    resource: &ControlObjectResource,
    analysis_state: &Value,
) -> std::collections::HashMap<String, DerivedColumn> {
    let columns = object_export_columns(resource, analysis_state);
    let properties = resource
        .property_names
        .iter()
        .cloned()
        .collect::<HashSet<_>>();
    let mut map = std::collections::HashMap::new();
    for column in columns {
        if properties.contains(&column) || column == "id" {
            continue;
        }
        let kind = match column.as_str() {
            name if name.starts_with("_odon_geometry_type") => DerivedColumn::GeometryType,
            name if name.starts_with("_odon_centroid_x") => DerivedColumn::CentroidX,
            name if name.starts_with("_odon_centroid_y") => DerivedColumn::CentroidY,
            name if name.starts_with("_odon_point_x") => DerivedColumn::PointX,
            name if name.starts_with("_odon_point_y") => DerivedColumn::PointY,
            name if name.starts_with("_odon_area_px") => DerivedColumn::Area,
            name if name.starts_with("_odon_perimeter_px") => DerivedColumn::Perimeter,
            name if name.starts_with("_odon_selected") => DerivedColumn::Selected,
            _ => continue,
        };
        map.insert(column, kind);
    }
    let mut used = resource
        .property_names
        .iter()
        .cloned()
        .collect::<HashSet<_>>();
    for base in [
        "_odon_geometry_type",
        "_odon_centroid_x",
        "_odon_centroid_y",
    ] {
        let _ = unique_export_name(base, &mut used);
    }
    if resource
        .features
        .iter()
        .any(|feature| feature.point_position_world.is_some())
    {
        let _ = unique_export_name("_odon_point_x", &mut used);
        let _ = unique_export_name("_odon_point_y", &mut used);
    }
    for base in ["_odon_area_px", "_odon_perimeter_px", "_odon_selected"] {
        let _ = unique_export_name(base, &mut used);
    }
    let calls = analysis_state
        .get("threshold_elements")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    if let Some(call) = analysis_state
        .get("threshold_selected_element")
        .and_then(Value::as_u64)
        .and_then(|index| usize::try_from(index).ok())
        .and_then(|index| calls.get(index))
    {
        let name = unique_export_name(&live_call_column_name(call, analysis_state), &mut used);
        map.insert(name, DerivedColumn::Call(call.clone()));
    }
    for call in calls {
        let name = unique_export_name(&threshold_call_column_name(call), &mut used);
        let failed = call
            .get("mark_failed")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            && call.pointer("/scope/kind").and_then(Value::as_str) == Some("marker");
        map.insert(
            name,
            if failed {
                DerivedColumn::FailedCall
            } else {
                DerivedColumn::Call(call.clone())
            },
        );
    }
    for selection in analysis_state
        .get("selection_elements")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
    {
        let base = format!(
            "_odon_selection_{}",
            sanitize_export_key(
                selection
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("unnamed")
            )
        );
        let name = unique_export_name(&base, &mut used);
        let ids = selection
            .get("object_ids")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect();
        map.insert(name, DerivedColumn::NamedSelection(ids));
    }
    map
}

fn export_value(
    column: &str,
    index: usize,
    feature: &ControlObjectFeature,
    resource: &ControlObjectResource,
    selected: &HashSet<usize>,
    derived: &std::collections::HashMap<String, DerivedColumn>,
) -> Option<ExportScalar> {
    if column == "id" {
        return Some(ExportScalar::String(feature.id.clone()));
    }
    if let Some(value) = resource.property_value(index, column) {
        return Some(scalar_from_json(&value));
    }
    match derived.get(column)? {
        DerivedColumn::GeometryType => {
            Some(ExportScalar::String(geometry_type(feature).to_string()))
        }
        DerivedColumn::CentroidX => Some(ExportScalar::Float64(feature.centroid_world.x as f64)),
        DerivedColumn::CentroidY => Some(ExportScalar::Float64(feature.centroid_world.y as f64)),
        DerivedColumn::PointX => feature
            .point_position_world
            .map(|point| ExportScalar::Float64(point.x as f64)),
        DerivedColumn::PointY => feature
            .point_position_world
            .map(|point| ExportScalar::Float64(point.y as f64)),
        DerivedColumn::Area => Some(ExportScalar::Float64(feature.area_px as f64)),
        DerivedColumn::Perimeter => Some(ExportScalar::Float64(feature.perimeter_px as f64)),
        DerivedColumn::Selected => Some(ExportScalar::Bool(selected.contains(&index))),
        DerivedColumn::Call(call) => Some(ExportScalar::Bool(passes_call(resource, index, call))),
        DerivedColumn::FailedCall => Some(ExportScalar::String("FAIL".to_string())),
        DerivedColumn::NamedSelection(ids) => Some(ExportScalar::Bool(ids.contains(&feature.id))),
    }
}

fn passes_call(resource: &ControlObjectResource, feature_index: usize, call: &Value) -> bool {
    let Some(rules) = call.get("rules").and_then(Value::as_array) else {
        return false;
    };
    !rules.is_empty()
        && rules.iter().all(|rule| {
            let Some(property) = rule.get("column_key").and_then(Value::as_str) else {
                return false;
            };
            let Some(value) = resource.property_f64(feature_index, property) else {
                return false;
            };
            let threshold = rule
                .get("value")
                .and_then(Value::as_f64)
                .unwrap_or(f64::NAN);
            match rule.get("op").and_then(Value::as_str) {
                Some("greater_equal") => value >= threshold,
                Some("less_equal") => value <= threshold,
                _ => false,
            }
        })
}

fn write_csv(
    path: &Path,
    table: &ExportTable,
    cancelled: &impl Fn() -> bool,
) -> anyhow::Result<()> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut writer = csv::Writer::from_writer(file);
    writer.write_record(table.columns.iter().map(|column| column.name.as_str()))?;
    for row in 0..table.row_count {
        if row % 256 == 0 {
            anyhow::ensure!(!cancelled(), "object export was cancelled");
        }
        writer.write_record(
            table
                .columns
                .iter()
                .map(|column| scalar_to_csv(column.values.get(row).and_then(Option::as_ref))),
        )?;
    }
    let file = writer.into_inner()?;
    file.sync_all()?;
    Ok(())
}

fn write_geoparquet(
    path: &Path,
    table: &ExportTable,
    cancelled: &impl Fn() -> bool,
) -> anyhow::Result<()> {
    anyhow::ensure!(!cancelled(), "object export was cancelled");
    let mut fields = Vec::with_capacity(table.columns.len() + 1);
    let mut arrays = Vec::with_capacity(table.columns.len() + 1);
    fields.push(Field::new(
        "geometry",
        arrow_schema::DataType::Binary,
        false,
    ));
    let mut geometry = BinaryBuilder::new();
    for value in &table.geometry_wkb {
        geometry.append_value(value);
    }
    arrays.push(Arc::new(geometry.finish()) as arrow_array::ArrayRef);
    for column in &table.columns {
        let (field, array) = column_to_arrow(column)?;
        fields.push(field);
        arrays.push(array);
    }
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;
    let geometry_types = table
        .geometry_types
        .iter()
        .map(|name| format!("\"{name}\""))
        .collect::<Vec<_>>()
        .join(",");
    let geo = format!(
        "{{\"version\":\"1.0.0\",\"primary_column\":\"geometry\",\"columns\":{{\"geometry\":{{\"encoding\":\"WKB\",\"geometry_types\":[{geometry_types}],\"crs\":null}}}}}}"
    );
    let properties = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_key_value_metadata(Some(vec![KeyValue {
            key: "geo".to_string(),
            value: Some(geo),
        }]))
        .build();
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let sync_file = file.try_clone()?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(properties))?;
    writer.write(&batch)?;
    writer.close()?;
    sync_file.sync_all()?;
    anyhow::ensure!(!cancelled(), "object export was cancelled");
    Ok(())
}

fn column_to_arrow(column: &ExportColumn) -> anyhow::Result<(Field, arrow_array::ArrayRef)> {
    match infer_type(&column.values) {
        ScalarType::Bool => {
            let mut builder = BooleanBuilder::new();
            for value in &column.values {
                match value {
                    Some(ExportScalar::Bool(value)) => builder.append_value(*value),
                    None => builder.append_null(),
                    Some(value) => builder.append_value(scalar_to_csv(Some(value)) == "true"),
                }
            }
            Ok((
                Field::new(&column.name, arrow_schema::DataType::Boolean, true),
                Arc::new(builder.finish()),
            ))
        }
        ScalarType::Int64 => {
            let mut builder = Int64Builder::new();
            for value in &column.values {
                match value {
                    Some(ExportScalar::Int64(value)) => builder.append_value(*value),
                    Some(ExportScalar::UInt64(value)) => builder.append_value(*value as i64),
                    Some(ExportScalar::Float64(value)) => builder.append_value(*value as i64),
                    None => builder.append_null(),
                    Some(value) => {
                        builder.append_value(scalar_to_csv(Some(value)).parse().unwrap_or_default())
                    }
                }
            }
            Ok((
                Field::new(&column.name, arrow_schema::DataType::Int64, true),
                Arc::new(builder.finish()),
            ))
        }
        ScalarType::UInt64 => {
            let mut builder = UInt64Builder::new();
            for value in &column.values {
                match value {
                    Some(ExportScalar::UInt64(value)) => builder.append_value(*value),
                    Some(ExportScalar::Int64(value)) if *value >= 0 => {
                        builder.append_value(*value as u64)
                    }
                    Some(ExportScalar::Float64(value)) if *value >= 0.0 => {
                        builder.append_value(*value as u64)
                    }
                    None => builder.append_null(),
                    Some(value) => match scalar_to_csv(Some(value)).parse() {
                        Ok(value) => builder.append_value(value),
                        Err(_) => builder.append_null(),
                    },
                }
            }
            Ok((
                Field::new(&column.name, arrow_schema::DataType::UInt64, true),
                Arc::new(builder.finish()),
            ))
        }
        ScalarType::Float64 => {
            let mut builder = Float64Builder::new();
            for value in &column.values {
                match value {
                    Some(ExportScalar::Float64(value)) => builder.append_value(*value),
                    Some(ExportScalar::Int64(value)) => builder.append_value(*value as f64),
                    Some(ExportScalar::UInt64(value)) => builder.append_value(*value as f64),
                    None => builder.append_null(),
                    Some(value) => match scalar_to_csv(Some(value)).parse() {
                        Ok(value) => builder.append_value(value),
                        Err(_) => builder.append_null(),
                    },
                }
            }
            Ok((
                Field::new(&column.name, arrow_schema::DataType::Float64, true),
                Arc::new(builder.finish()),
            ))
        }
        ScalarType::String => {
            let mut builder = StringBuilder::new();
            for value in &column.values {
                match value {
                    Some(value) => builder.append_value(scalar_to_csv(Some(value))),
                    None => builder.append_null(),
                }
            }
            Ok((
                Field::new(&column.name, arrow_schema::DataType::Utf8, true),
                Arc::new(builder.finish()),
            ))
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ScalarType {
    Bool,
    Int64,
    UInt64,
    Float64,
    String,
}

fn infer_type(values: &[Option<ExportScalar>]) -> ScalarType {
    let mut kinds = HashSet::new();
    for value in values.iter().flatten() {
        kinds.insert(match value {
            ExportScalar::Bool(_) => 0,
            ExportScalar::Int64(_) => 1,
            ExportScalar::UInt64(_) => 2,
            ExportScalar::Float64(_) => 3,
            ExportScalar::String(_) => 4,
        });
    }
    if kinds == HashSet::from([0]) {
        ScalarType::Bool
    } else if kinds == HashSet::from([1]) {
        ScalarType::Int64
    } else if kinds == HashSet::from([2]) {
        ScalarType::UInt64
    } else if !kinds.contains(&0) && !kinds.contains(&4) && !kinds.is_empty() {
        ScalarType::Float64
    } else {
        ScalarType::String
    }
}

fn scalar_from_json(value: &Value) -> ExportScalar {
    match value {
        Value::Bool(value) => ExportScalar::Bool(*value),
        Value::Number(value) if value.as_i64().is_some() => {
            ExportScalar::Int64(value.as_i64().unwrap())
        }
        Value::Number(value) if value.as_u64().is_some() => {
            ExportScalar::UInt64(value.as_u64().unwrap())
        }
        Value::Number(value) => ExportScalar::Float64(value.as_f64().unwrap_or_default()),
        Value::String(value) => ExportScalar::String(value.clone()),
        Value::Null => ExportScalar::String(String::new()),
        value => ExportScalar::String(value.to_string()),
    }
}

fn scalar_to_csv(value: Option<&ExportScalar>) -> String {
    match value {
        None => String::new(),
        Some(ExportScalar::Bool(value)) => value.to_string(),
        Some(ExportScalar::Int64(value)) => value.to_string(),
        Some(ExportScalar::UInt64(value)) => value.to_string(),
        Some(ExportScalar::Float64(value)) => value.to_string(),
        Some(ExportScalar::String(value)) => value.clone(),
    }
}

fn geometry_type(feature: &ControlObjectFeature) -> &'static str {
    if feature.polygons_world.is_empty() {
        "Point"
    } else if feature.polygons_world.len() == 1 {
        "Polygon"
    } else {
        "MultiPolygon"
    }
}

fn encode_wkb(feature: &ControlObjectFeature) -> Vec<u8> {
    if feature.polygons_world.is_empty() {
        let point = feature
            .point_position_world
            .unwrap_or(feature.centroid_world);
        let mut output = Vec::with_capacity(21);
        output.push(1);
        output.extend_from_slice(&1u32.to_le_bytes());
        output.extend_from_slice(&(point.x as f64).to_le_bytes());
        output.extend_from_slice(&(point.y as f64).to_le_bytes());
        return output;
    }
    if feature.polygons_world.len() == 1 {
        return encode_polygon(std::slice::from_ref(&feature.polygons_world[0]));
    }
    let mut output = Vec::new();
    output.push(1);
    output.extend_from_slice(&6u32.to_le_bytes());
    output.extend_from_slice(&(feature.polygons_world.len() as u32).to_le_bytes());
    for polygon in feature.polygons_world.iter() {
        output.extend_from_slice(&encode_polygon(std::slice::from_ref(polygon)));
    }
    output
}

fn encode_polygon(rings: &[Vec<eframe::egui::Pos2>]) -> Vec<u8> {
    let mut output = Vec::new();
    output.push(1);
    output.extend_from_slice(&3u32.to_le_bytes());
    output.extend_from_slice(&(rings.len() as u32).to_le_bytes());
    for ring in rings {
        let close = ring.first() != ring.last();
        output.extend_from_slice(&((ring.len() + usize::from(close)) as u32).to_le_bytes());
        for point in ring.iter().chain(close.then(|| ring.first()).flatten()) {
            output.extend_from_slice(&(point.x as f64).to_le_bytes());
            output.extend_from_slice(&(point.y as f64).to_le_bytes());
        }
    }
    output
}

fn threshold_call_column_name(call: &Value) -> String {
    let label = sanitize_export_key(
        call.get("name")
            .and_then(Value::as_str)
            .unwrap_or("unnamed"),
    );
    let channel = call
        .pointer("/scope/channel_name")
        .and_then(Value::as_str)
        .map(sanitize_export_key);
    match channel {
        Some(channel) if label == channel || label.starts_with(&format!("{channel}_")) => {
            format!("_odon_call_{label}")
        }
        Some(channel) => format!("_odon_call_{channel}_{label}"),
        None => format!("_odon_call_{label}"),
    }
}

fn live_call_column_name(call: &Value, state: &Value) -> String {
    let channel = call
        .pointer("/scope/channel_name")
        .and_then(Value::as_str)
        .or_else(|| {
            state
                .get("live_threshold_channel_name")
                .and_then(Value::as_str)
        });
    channel
        .map(|name| format!("_odon_live_call_{}", sanitize_export_key(name)))
        .unwrap_or_else(|| "_odon_live_call".to_string())
}

fn sanitize_export_key(name: &str) -> String {
    let mut output = String::new();
    let mut underscore = false;
    for character in name.chars() {
        if character.is_ascii_alphanumeric() {
            output.push(character.to_ascii_lowercase());
            underscore = false;
        } else if !underscore {
            output.push('_');
            underscore = true;
        }
    }
    let output = output.trim_matches('_');
    if output.is_empty() {
        "unnamed".to_string()
    } else {
        output.to_string()
    }
}

fn push_unique(columns: &mut Vec<String>, used: &mut HashSet<String>, base: &str) {
    columns.push(unique_export_name(base, used));
}

fn unique_export_name(base: &str, used: &mut HashSet<String>) -> String {
    if used.insert(base.to_string()) {
        return base.to_string();
    }
    for suffix in 2usize.. {
        let candidate = format!("{base}_{suffix}");
        if used.insert(candidate.clone()) {
            return candidate;
        }
    }
    unreachable!()
}

fn temporary_sibling(path: &Path) -> PathBuf {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("objects");
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    parent.join(format!(".{name}.odon-{}-{nonce}.tmp", std::process::id()))
}

fn commit_temp(temp: &Path, destination: &Path, overwrite: bool) -> anyhow::Result<()> {
    if !overwrite {
        fs::hard_link(temp, destination).map_err(|error| {
            if error.kind() == std::io::ErrorKind::AlreadyExists {
                anyhow::anyhow!("destination exists; pass overwrite=true to replace it")
            } else {
                error.into()
            }
        })?;
        fs::remove_file(temp)?;
        return Ok(());
    }
    match fs::rename(temp, destination) {
        Ok(()) => Ok(()),
        Err(error) if destination.exists() => {
            fs::remove_file(destination)?;
            fs::rename(temp, destination).map_err(Into::into)
        }
        Err(error) => Err(error.into()),
    }
}

fn invalid(message: impl Into<String>) -> ControlError {
    ControlError::new(ControlErrorKind::InvalidParams, message)
}
