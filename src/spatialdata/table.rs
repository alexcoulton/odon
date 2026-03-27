use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use zarrs::array::{Array, ArraySubset};
use zarrs::storage::ReadableStorageTraits;

use crate::spatialdata::SpatialDataElement;
use crate::data::zarr_attrs::read_node_attributes;

#[derive(Debug, Clone)]
pub struct SpatialDataTableMeta {
    pub table_name: String,
    pub rel_group: std::path::PathBuf,
    pub row_count: usize,
    pub join_keys: Vec<String>,
    pub numeric_columns: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SpatialTableAnalysis {
    pub table_name: String,
    pub rel_group: std::path::PathBuf,
    pub join_key: String,
    pub numeric_columns: Vec<String>,
    pub rows: Vec<SpatialTableRow>,
    pub loaded_numeric_columns: HashMap<String, Vec<f32>>,
    pub matched_object_count: usize,
    pub unmatched_object_count: usize,
}

#[derive(Debug, Clone)]
pub struct SpatialTableRow {
    pub row_index: usize,
    pub object_index: usize,
    pub object_id: String,
}

pub fn load_table_meta(
    root: &Path,
    element: &SpatialDataElement,
) -> anyhow::Result<SpatialDataTableMeta> {
    let store: Arc<dyn ReadableStorageTraits> =
        Arc::new(zarrs::filesystem::FilesystemStore::new(root)?);
    let obs_dir = root.join(&element.rel_group).join("obs");
    let attrs = read_node_attributes(&obs_dir)
        .with_context(|| format!("read SpatialData table attrs: {}", obs_dir.display()))?
        .unwrap_or_default();
    let mut column_names = attrs
        .get("column-order")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(ToOwned::to_owned))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    if column_names.is_empty() {
        for entry in std::fs::read_dir(&obs_dir)? {
            let entry = entry?;
            if entry.path().is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    column_names.push(name.to_string());
                }
            }
        }
        column_names.sort();
    }

    let mut row_count = 0usize;
    let mut numeric_columns = Vec::new();
    let mut available_columns = HashSet::new();
    for column in &column_names {
        let Ok(array) = open_obs_array(store.clone(), element, column) else {
            continue;
        };
        row_count = row_count.max(array.shape().first().copied().unwrap_or(0) as usize);
        let dtype = format!("{}", array.data_type()).to_ascii_lowercase();
        available_columns.insert(column.clone());
        if is_numeric_dtype(&dtype) {
            numeric_columns.push(column.clone());
        }
    }

    let index_name = attrs
        .get("_index")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let mut join_keys = Vec::new();
    if let Some(index_name) = index_name.filter(|name| available_columns.contains(name)) {
        join_keys.push(index_name);
    }
    for name in ["instance_id", "cell_id", "label", "id", "object_id", "name"] {
        if available_columns.contains(name) && !join_keys.iter().any(|k| k == name) {
            join_keys.push(name.to_string());
        }
    }

    Ok(SpatialDataTableMeta {
        table_name: element.name.clone(),
        rel_group: element.rel_group.clone(),
        row_count,
        join_keys,
        numeric_columns,
    })
}

pub fn load_table_analysis(
    root: &Path,
    meta: &SpatialDataTableMeta,
    join_key: &str,
    object_lookup: &HashMap<String, (usize, String)>,
) -> anyhow::Result<SpatialTableAnalysis> {
    let store: Arc<dyn ReadableStorageTraits> =
        Arc::new(zarrs::filesystem::FilesystemStore::new(root)?);
    let join_array = open_obs_array_by_group(store.clone(), &meta.rel_group, join_key)?;
    let join_values = retrieve_column_strings(&join_array, &format!("{}", join_array.data_type()))
        .with_context(|| format!("read join key column '{join_key}'"))?;

    let mut matched_rows = Vec::<(usize, usize, String)>::new();
    let mut matched_object_indices = HashSet::new();
    for (row_idx, join_value) in join_values.iter().enumerate() {
        if let Some((object_index, object_id)) = object_lookup.get(join_value) {
            matched_rows.push((row_idx, *object_index, object_id.clone()));
            matched_object_indices.insert(*object_index);
        }
    }

    let mut rows = matched_rows
        .iter()
        .map(|(row_index, object_index, object_id)| SpatialTableRow {
            row_index: *row_index,
            object_index: *object_index,
            object_id: object_id.clone(),
        })
        .collect::<Vec<_>>();
    rows.shrink_to_fit();

    Ok(SpatialTableAnalysis {
        table_name: meta.table_name.clone(),
        rel_group: meta.rel_group.clone(),
        join_key: join_key.to_string(),
        numeric_columns: meta.numeric_columns.clone(),
        rows,
        loaded_numeric_columns: HashMap::new(),
        matched_object_count: matched_object_indices.len(),
        unmatched_object_count: object_lookup
            .len()
            .saturating_sub(matched_object_indices.len()),
    })
}

pub fn load_numeric_column_for_rows(
    root: &Path,
    analysis: &SpatialTableAnalysis,
    column: &str,
) -> anyhow::Result<Vec<f32>> {
    let store: Arc<dyn ReadableStorageTraits> =
        Arc::new(zarrs::filesystem::FilesystemStore::new(root)?);
    let array = open_obs_array_by_group(store, &analysis.rel_group, column)?;
    let dtype = format!("{}", array.data_type());
    let values = retrieve_numeric_column_f32(&array, &dtype)
        .with_context(|| format!("read numeric table column '{column}'"))?;
    Ok(analysis
        .rows
        .iter()
        .map(|row| values.get(row.row_index).copied().unwrap_or(f32::NAN))
        .collect())
}

fn open_obs_array(
    store: Arc<dyn ReadableStorageTraits>,
    element: &SpatialDataElement,
    column: &str,
) -> anyhow::Result<Array<dyn ReadableStorageTraits>> {
    open_obs_array_by_group(store, &element.rel_group, column)
}

fn open_obs_array_by_group(
    store: Arc<dyn ReadableStorageTraits>,
    rel_group: &std::path::Path,
    column: &str,
) -> anyhow::Result<Array<dyn ReadableStorageTraits>> {
    let fs_path = rel_group.join("obs").join(column);
    let attrs = read_node_attributes(&fs_path)
        .ok()
        .flatten()
        .unwrap_or_default();
    let node_type = attrs
        .get("encoding-type")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    if node_type == "categorical" {
        anyhow::bail!("categorical columns are not direct arrays");
    }
    let zarr_path = format!(
        "/{}/obs/{}",
        rel_group.to_string_lossy().trim_matches('/'),
        column
    );
    Array::open(store, &zarr_path).with_context(|| format!("open table array {zarr_path}"))
}

fn retrieve_column_strings(
    array: &Array<dyn ReadableStorageTraits>,
    dtype: &str,
) -> anyhow::Result<Vec<String>> {
    let subset = subset_all_1d(array)?;
    let dtype = dtype.trim().to_ascii_lowercase();
    let is_string_like = dtype
        .split('/')
        .map(|part| part.trim())
        .any(|part| part == "string" || part == "|o" || part == "object");
    if is_string_like {
        let data: ndarray::ArrayD<String> = array.retrieve_array_subset(&subset)?;
        return Ok(data.into_iter().collect());
    }

    retrieve_numeric_column_f32(array, dtype.as_str()).map(|values| {
        values
            .into_iter()
            .map(|value| {
                if value.fract() == 0.0 {
                    format!("{}", value as i64)
                } else {
                    value.to_string()
                }
            })
            .collect()
    })
}

fn retrieve_numeric_column_f32(
    array: &Array<dyn ReadableStorageTraits>,
    dtype: &str,
) -> anyhow::Result<Vec<f32>> {
    let subset = subset_all_1d(array)?;
    match parse_numeric_dtype(dtype) {
        Some(NumericDtype::Unsigned(8)) => {
            let data: ndarray::ArrayD<u8> = array.retrieve_array_subset(&subset)?;
            Ok(data.into_iter().map(|v| v as f32).collect())
        }
        Some(NumericDtype::Unsigned(16)) => {
            let data: ndarray::ArrayD<u16> = array.retrieve_array_subset(&subset)?;
            Ok(data.into_iter().map(|v| v as f32).collect())
        }
        Some(NumericDtype::Unsigned(32)) => {
            let data: ndarray::ArrayD<u32> = array.retrieve_array_subset(&subset)?;
            Ok(data.into_iter().map(|v| v as f32).collect())
        }
        Some(NumericDtype::Unsigned(64)) => {
            let data: ndarray::ArrayD<u64> = array.retrieve_array_subset(&subset)?;
            Ok(data.into_iter().map(|v| v as f32).collect())
        }
        Some(NumericDtype::Signed(8)) => {
            let data: ndarray::ArrayD<i8> = array.retrieve_array_subset(&subset)?;
            Ok(data.into_iter().map(|v| v as f32).collect())
        }
        Some(NumericDtype::Signed(16)) => {
            let data: ndarray::ArrayD<i16> = array.retrieve_array_subset(&subset)?;
            Ok(data.into_iter().map(|v| v as f32).collect())
        }
        Some(NumericDtype::Signed(32)) => {
            let data: ndarray::ArrayD<i32> = array.retrieve_array_subset(&subset)?;
            Ok(data.into_iter().map(|v| v as f32).collect())
        }
        Some(NumericDtype::Signed(64)) => {
            let data: ndarray::ArrayD<i64> = array.retrieve_array_subset(&subset)?;
            Ok(data.into_iter().map(|v| v as f32).collect())
        }
        Some(NumericDtype::Float(32)) => {
            let data: ndarray::ArrayD<f32> = array.retrieve_array_subset(&subset)?;
            Ok(data.into_iter().collect())
        }
        Some(NumericDtype::Float(64)) => {
            let data: ndarray::ArrayD<f64> = array.retrieve_array_subset(&subset)?;
            Ok(data.into_iter().map(|v| v as f32).collect())
        }
        _ => anyhow::bail!("unsupported numeric table dtype: {dtype}"),
    }
}

fn subset_all_1d(array: &Array<dyn ReadableStorageTraits>) -> anyhow::Result<ArraySubset> {
    let shape = array.shape().to_vec();
    if shape.len() != 1 {
        anyhow::bail!("expected 1D table column, got shape {:?}", shape);
    }
    Ok(ArraySubset::new_with_ranges(&[0..shape[0]]))
}

fn is_numeric_dtype(dtype: &str) -> bool {
    parse_numeric_dtype(dtype).is_some()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumericDtype {
    Unsigned(u32),
    Signed(u32),
    Float(u32),
}

fn parse_numeric_dtype(dtype: &str) -> Option<NumericDtype> {
    for part in dtype.split('/') {
        let d = part.trim().to_ascii_lowercase();
        if d.is_empty() {
            continue;
        }

        if let Some(bits) = d.strip_prefix("uint").and_then(|s| s.parse::<u32>().ok()) {
            return Some(NumericDtype::Unsigned(bits));
        }
        if let Some(bits) = d.strip_prefix("int").and_then(|s| s.parse::<u32>().ok()) {
            return Some(NumericDtype::Signed(bits));
        }
        if let Some(bits) = d.strip_prefix("float").and_then(|s| s.parse::<u32>().ok()) {
            return Some(NumericDtype::Float(bits));
        }
        if let Some(bits) = d.strip_prefix('u').and_then(|s| s.parse::<u32>().ok()) {
            return Some(NumericDtype::Unsigned(bits));
        }
        if let Some(bits) = d.strip_prefix('i').and_then(|s| s.parse::<u32>().ok()) {
            return Some(NumericDtype::Signed(bits));
        }
        if let Some(bits) = d.strip_prefix('f').and_then(|s| s.parse::<u32>().ok()) {
            return Some(NumericDtype::Float(bits));
        }

        let np = d
            .strip_prefix('<')
            .or_else(|| d.strip_prefix('>'))
            .or_else(|| d.strip_prefix('|'))
            .unwrap_or(d.as_str());
        if np.len() >= 2 {
            let kind = np.chars().next()?;
            let bytes = np[1..].parse::<u32>().ok()?;
            let bits = bytes.checked_mul(8)?;
            match kind {
                'u' => return Some(NumericDtype::Unsigned(bits)),
                'i' => return Some(NumericDtype::Signed(bits)),
                'f' => return Some(NumericDtype::Float(bits)),
                _ => {}
            }
        }
    }

    None
}
