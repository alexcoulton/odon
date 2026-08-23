use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use arrow_array::Array;
use eframe::egui;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use super::{AnnotationDataset, AnnotationRoiData, AnnotationValueMode, ColumnInfo};
use crate::render::point_bins::PointIndexBins;

pub(super) fn read_parquet_columns(path: &Path) -> anyhow::Result<Vec<ColumnInfo>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open parquet: {}", path.to_string_lossy()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("failed to create parquet reader builder")?;
    let schema = builder.schema();
    let mut out = Vec::new();
    for f in schema.fields() {
        out.push(ColumnInfo {
            name: f.name().to_string(),
        });
    }
    Ok(out)
}

pub(super) fn load_annotations_parquet(
    path: &Path,
    roi_id_column: &str,
    x_column: &str,
    y_column: &str,
    value_column: &str,
) -> anyhow::Result<AnnotationDataset> {
    let file = File::open(path)
        .with_context(|| format!("failed to open parquet: {}", path.to_string_lossy()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("failed to create parquet reader builder")?;

    // Projection: id, x, y, value.
    let projection = ProjectionMask::columns(
        builder.parquet_schema(),
        [roi_id_column, x_column, y_column, value_column],
    );
    let mut reader = builder
        .with_batch_size(65_536)
        .with_projection(projection)
        .build()
        .context("failed to build parquet record batch reader")?;

    let mut roi_map: HashMap<String, usize> = HashMap::new();
    let mut rois: Vec<(String, Vec<egui::Pos2>, Vec<f32>)> = Vec::new();

    let mut categories: Vec<String> = Vec::new();
    let mut cat_index: HashMap<String, u32> = HashMap::new();

    let mut mode: Option<AnnotationValueMode> = None;
    let mut vmin = f32::INFINITY;
    let mut vmax = f32::NEG_INFINITY;

    while let Some(batch) = reader.next() {
        let batch = batch.context("failed to read parquet batch")?;
        let n = batch.num_rows();
        if n == 0 {
            continue;
        }

        let schema = batch.schema();
        let id_i = schema
            .index_of(roi_id_column)
            .with_context(|| format!("missing required column '{roi_id_column}'"))?;
        let x_i = schema
            .index_of(x_column)
            .with_context(|| format!("missing required column '{x_column}'"))?;
        let y_i = schema
            .index_of(y_column)
            .with_context(|| format!("missing required column '{y_column}'"))?;
        let v_i = schema
            .index_of(value_column)
            .with_context(|| format!("missing required column '{value_column}'"))?;

        let id = batch.column(id_i).as_ref();
        let x = batch.column(x_i).as_ref();
        let y = batch.column(y_i).as_ref();
        let v = batch.column(v_i).as_ref();

        let id_col = StrCol::try_new(id).context("ROI id column")?;
        let x_col = NumAnyF32Col::try_new(x).context("x column")?;
        let y_col = NumAnyF32Col::try_new(y).context("y column")?;

        // Determine mode on first batch.
        if mode.is_none() {
            mode = Some(match v.data_type() {
                arrow_schema::DataType::Utf8 | arrow_schema::DataType::LargeUtf8 => {
                    AnnotationValueMode::Categorical
                }
                arrow_schema::DataType::Float16
                | arrow_schema::DataType::Float32
                | arrow_schema::DataType::Float64
                | arrow_schema::DataType::Int8
                | arrow_schema::DataType::Int16
                | arrow_schema::DataType::Int32
                | arrow_schema::DataType::Int64
                | arrow_schema::DataType::UInt8
                | arrow_schema::DataType::UInt16
                | arrow_schema::DataType::UInt32
                | arrow_schema::DataType::UInt64 => AnnotationValueMode::Continuous,
                _ => {
                    anyhow::bail!(
                        "unsupported value column type for '{value_column}': {:?}",
                        v.data_type()
                    );
                }
            });
        }

        match mode.unwrap_or(AnnotationValueMode::Categorical) {
            AnnotationValueMode::Categorical => {
                let v_col = StrCol::try_new(v).context("value column")?;
                for row in 0..n {
                    let Some(roi_id) = id_col.get(row) else {
                        continue;
                    };
                    let Some(xv) = x_col.get(row) else { continue };
                    let Some(yv) = y_col.get(row) else { continue };
                    let label = v_col.get(row).unwrap_or("(missing)");
                    let code = if let Some(&c) = cat_index.get(label) {
                        c
                    } else {
                        let c = categories.len() as u32;
                        categories.push(label.to_string());
                        cat_index.insert(label.to_string(), c);
                        c
                    };

                    let idx = *roi_map.entry(roi_id.to_string()).or_insert_with(|| {
                        let idx = rois.len();
                        rois.push((roi_id.to_string(), Vec::new(), Vec::new()));
                        idx
                    });
                    rois[idx].1.push(egui::pos2(xv, yv));
                    rois[idx].2.push(code as f32);
                }
            }
            AnnotationValueMode::Continuous => {
                let v_col = NumAnyF32Col::try_new(v).context("value column")?;
                for row in 0..n {
                    let Some(roi_id) = id_col.get(row) else {
                        continue;
                    };
                    let Some(xv) = x_col.get(row) else { continue };
                    let Some(yv) = y_col.get(row) else { continue };
                    let Some(vv) = v_col.get(row) else { continue };
                    vmin = vmin.min(vv);
                    vmax = vmax.max(vv);
                    let idx = *roi_map.entry(roi_id.to_string()).or_insert_with(|| {
                        let idx = rois.len();
                        rois.push((roi_id.to_string(), Vec::new(), Vec::new()));
                        idx
                    });
                    rois[idx].1.push(egui::pos2(xv, yv));
                    rois[idx].2.push(vv);
                }
            }
        }
    }

    if vmin == f32::INFINITY {
        vmin = 0.0;
    }
    if vmax == f32::NEG_INFINITY {
        vmax = 1.0;
    }

    let mut roi: HashMap<String, AnnotationRoiData> = HashMap::new();
    roi.reserve(rois.len());
    let mut total_points = 0usize;
    for (id, pos, vals) in rois.into_iter() {
        let n = pos.len().min(vals.len());
        let bins_local = PointIndexBins::build(&pos, 64.0).map(Arc::new);
        total_points += n;
        roi.insert(
            id,
            AnnotationRoiData {
                positions_local: Arc::new(pos),
                values: Arc::new(vals),
                count: n,
                bins_local,
            },
        );
    }

    let total_rois = roi.len();
    Ok(AnnotationDataset {
        mode: mode.unwrap_or(AnnotationValueMode::Categorical),
        categories,
        roi,
        value_min: vmin,
        value_max: vmax,
        total_points,
        total_rois,
    })
}

#[derive(Clone)]
enum StrCol<'a> {
    Utf8(&'a arrow_array::StringArray),
    LargeUtf8(&'a arrow_array::LargeStringArray),
}

impl<'a> StrCol<'a> {
    fn try_new(array: &'a dyn arrow_array::Array) -> anyhow::Result<Self> {
        if let Some(col) = array.as_any().downcast_ref::<arrow_array::StringArray>() {
            return Ok(Self::Utf8(col));
        }
        if let Some(col) = array
            .as_any()
            .downcast_ref::<arrow_array::LargeStringArray>()
        {
            return Ok(Self::LargeUtf8(col));
        }
        anyhow::bail!("unsupported string type")
    }

    fn get(&self, row: usize) -> Option<&'a str> {
        match self {
            Self::Utf8(col) => (!col.is_null(row)).then(|| col.value(row)),
            Self::LargeUtf8(col) => (!col.is_null(row)).then(|| col.value(row)),
        }
    }
}

#[derive(Clone)]
enum NumAnyF32Col<'a> {
    F32(&'a arrow_array::Float32Array),
    F64(&'a arrow_array::Float64Array),
    I8(&'a arrow_array::Int8Array),
    I16(&'a arrow_array::Int16Array),
    I32(&'a arrow_array::Int32Array),
    I64(&'a arrow_array::Int64Array),
    U8(&'a arrow_array::UInt8Array),
    U16(&'a arrow_array::UInt16Array),
    U32(&'a arrow_array::UInt32Array),
    U64(&'a arrow_array::UInt64Array),
}

impl<'a> NumAnyF32Col<'a> {
    fn try_new(array: &'a dyn arrow_array::Array) -> anyhow::Result<Self> {
        if let Some(col) = array.as_any().downcast_ref::<arrow_array::Float32Array>() {
            return Ok(Self::F32(col));
        }
        if let Some(col) = array.as_any().downcast_ref::<arrow_array::Float64Array>() {
            return Ok(Self::F64(col));
        }
        if let Some(col) = array.as_any().downcast_ref::<arrow_array::Int8Array>() {
            return Ok(Self::I8(col));
        }
        if let Some(col) = array.as_any().downcast_ref::<arrow_array::Int16Array>() {
            return Ok(Self::I16(col));
        }
        if let Some(col) = array.as_any().downcast_ref::<arrow_array::Int32Array>() {
            return Ok(Self::I32(col));
        }
        if let Some(col) = array.as_any().downcast_ref::<arrow_array::Int64Array>() {
            return Ok(Self::I64(col));
        }
        if let Some(col) = array.as_any().downcast_ref::<arrow_array::UInt8Array>() {
            return Ok(Self::U8(col));
        }
        if let Some(col) = array.as_any().downcast_ref::<arrow_array::UInt16Array>() {
            return Ok(Self::U16(col));
        }
        if let Some(col) = array.as_any().downcast_ref::<arrow_array::UInt32Array>() {
            return Ok(Self::U32(col));
        }
        if let Some(col) = array.as_any().downcast_ref::<arrow_array::UInt64Array>() {
            return Ok(Self::U64(col));
        }
        anyhow::bail!("unsupported numeric type for f32 conversion")
    }

    fn get(&self, row: usize) -> Option<f32> {
        match self {
            Self::F32(col) => (!col.is_null(row)).then(|| col.value(row)),
            Self::F64(col) => (!col.is_null(row)).then(|| col.value(row) as f32),
            Self::I8(col) => (!col.is_null(row)).then(|| col.value(row) as f32),
            Self::I16(col) => (!col.is_null(row)).then(|| col.value(row) as f32),
            Self::I32(col) => (!col.is_null(row)).then(|| col.value(row) as f32),
            Self::I64(col) => (!col.is_null(row)).then(|| col.value(row) as f32),
            Self::U8(col) => (!col.is_null(row)).then(|| col.value(row) as f32),
            Self::U16(col) => (!col.is_null(row)).then(|| col.value(row) as f32),
            Self::U32(col) => (!col.is_null(row)).then(|| col.value(row) as f32),
            Self::U64(col) => (!col.is_null(row)).then(|| col.value(row) as f32),
        }
    }
}
