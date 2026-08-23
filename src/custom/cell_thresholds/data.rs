use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, anyhow};
use arrow_array::{Array, RecordBatch, RecordBatchReader};
use eframe::egui;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use super::{LoadRequest, LoadResponse, MarkerChoice};
use crate::data::project_config::{ProjectConfig, ProjectDatasetConfig};

pub(super) fn project_dataset_key_for_root(project: &ProjectConfig, dataset_root: &Path) -> String {
    // Explicit ROI list is the source of truth: use the dataset key attached to the ROI entry
    // (fall back to the project default, then "default").
    let mut key = None;
    for roi in &project.rois {
        if roi.local_path().is_some_and(|path| {
            path == dataset_root || path.to_string_lossy() == dataset_root.to_string_lossy()
        }) {
            key = roi.dataset.clone();
            break;
        }
    }
    key.or_else(|| project.default_dataset.clone())
        .unwrap_or_else(|| "default".to_string())
}

pub(super) fn best_base_dir_for_root(
    ds_cfg: &ProjectDatasetConfig,
    dataset_root: &Path,
) -> (Option<PathBuf>, bool) {
    let mut best: Option<(usize, PathBuf, bool)> = None;
    for (raw, downsampled) in [
        (ds_cfg.base_dir_full_res.as_deref(), false),
        (ds_cfg.base_dir_downsampled.as_deref(), true),
    ] {
        let Some(raw) = raw else {
            continue;
        };
        let p = PathBuf::from(expand_tilde(raw));
        if dataset_root.starts_with(&p) {
            let len = p.as_os_str().len();
            if best.as_ref().map(|b| len > b.0).unwrap_or(true) {
                best = Some((len, p, downsampled));
            }
        }
    }
    best.map(|(_, p, d)| (Some(p), d)).unwrap_or((None, false))
}

pub(super) fn parquet_path_for_zarr_root(
    parquet_dir: &Path,
    dataset_root: &Path,
) -> Option<PathBuf> {
    if !parquet_dir.is_dir() {
        return None;
    }
    let roi_dir = dataset_root.parent()?;
    let sample_dir = roi_dir.parent()?;
    let sample = sample_dir.file_name()?.to_str()?.trim();
    let roi_short = roi_dir.file_name()?.to_str()?.trim();
    let roi_short = normalize_roi_label(roi_short);
    let roi_n = parse_roi_number(&roi_short)?;
    let roi_dash = format!("ROI-{roi_n:02}");
    let fname = format!("{sample}.{roi_dash}.cells.parquet");
    Some(parquet_dir.join(fname))
}

fn parse_roi_number(label: &str) -> Option<u64> {
    let tail = label.rsplit_once('/').map(|(_, t)| t).unwrap_or(label);
    let tail = tail.trim();
    let digits = tail.strip_prefix("ROI")?;
    digits.parse::<u64>().ok()
}

pub(super) fn spawn_loader_thread() -> (
    crossbeam_channel::Sender<LoadRequest>,
    crossbeam_channel::Receiver<LoadResponse>,
) {
    let (tx_req, rx_req) = crossbeam_channel::unbounded::<LoadRequest>();
    let (tx_rsp, rx_rsp) = crossbeam_channel::unbounded::<LoadResponse>();

    std::thread::Builder::new()
        .name("cells-parquet-loader".to_string())
        .spawn(move || {
            if let Err(err) = loader_thread(rx_req, tx_rsp) {
                eprintln!("cells parquet loader thread exited: {err:?}");
            }
        })
        .expect("failed to spawn cells parquet loader thread");

    (tx_req, rx_rsp)
}

fn loader_thread(
    rx_req: crossbeam_channel::Receiver<LoadRequest>,
    tx_rsp: crossbeam_channel::Sender<LoadResponse>,
) -> anyhow::Result<()> {
    for req in rx_req.iter() {
        let resp = load_points_for_marker(&req)?;
        let _ = tx_rsp.send(resp);
    }
    Ok(())
}

fn load_points_for_marker(req: &LoadRequest) -> anyhow::Result<LoadResponse> {
    let file = fs::File::open(&req.parquet_path).with_context(|| {
        format!(
            "failed to open parquet: {}",
            req.parquet_path.to_string_lossy()
        )
    })?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("failed to create parquet record batch reader builder")?;
    let projection = ProjectionMask::columns(
        builder.parquet_schema(),
        [
            "roi_id",
            "x_centroid",
            "y_centroid",
            req.key.marker_column.as_str(),
        ],
    );
    let mut reader = builder
        .with_projection(projection)
        .with_batch_size(65_536)
        .build()
        .context("failed to build parquet record batch reader")?;

    let out_schema = reader.schema();
    let roi_i = out_schema
        .index_of("roi_id")
        .context("missing required column 'roi_id'")?;
    let x_i = out_schema
        .index_of("x_centroid")
        .context("missing required column 'x_centroid'")?;
    let y_i = out_schema
        .index_of("y_centroid")
        .context("missing required column 'y_centroid'")?;
    let m_i = out_schema
        .index_of(req.key.marker_column.as_str())
        .with_context(|| format!("missing marker column '{}'", req.key.marker_column))?;

    let mut positions: Vec<egui::Pos2> = Vec::new();
    let mut values: Vec<f32> = Vec::new();
    let roi_norm = normalize_roi_label(&req.key.roi_label);
    let inv_down = 1.0 / f32::from_bits(req.key.coord_downsample_bits).max(1e-6);

    while let Some(batch) = reader.next() {
        let batch = batch.context("failed to read parquet batch")?;
        extract_batch(
            &batch,
            roi_i,
            x_i,
            y_i,
            m_i,
            &roi_norm,
            inv_down as f64,
            &mut positions,
            &mut values,
        )?;
    }

    let (min, max) = finite_min_max(&values).unwrap_or((0.0, 1.0));

    Ok(LoadResponse {
        request_id: req.request_id,
        key: req.key.clone(),
        positions,
        values,
        min,
        max,
    })
}

fn extract_batch(
    batch: &RecordBatch,
    roi_i: usize,
    x_i: usize,
    y_i: usize,
    m_i: usize,
    roi_norm: &str,
    inv_downsample: f64,
    out_positions: &mut Vec<egui::Pos2>,
    out_values: &mut Vec<f32>,
) -> anyhow::Result<()> {
    let len = batch.num_rows();
    for i in 0..len {
        let Some(roi_raw) = get_utf8(batch.column(roi_i).as_ref(), i)? else {
            continue;
        };
        if roi_raw != roi_norm && normalize_roi_label(roi_raw) != roi_norm {
            continue;
        }

        let Some(x0) = get_f64(batch.column(x_i).as_ref(), i)? else {
            continue;
        };
        let Some(y0) = get_f64(batch.column(y_i).as_ref(), i)? else {
            continue;
        };
        let Some(mut v) = get_f64(batch.column(m_i).as_ref(), i)? else {
            continue;
        };

        let x = x0 * inv_downsample;
        let y = y0 * inv_downsample;
        if !v.is_finite() {
            continue;
        }
        v = v.asinh();

        out_positions.push(egui::pos2(x as f32, y as f32));
        out_values.push(v as f32);
    }

    Ok(())
}

pub(super) fn list_marker_choices(
    parquet_path: &Path,
    channel_labels: &[String],
    marker_stat: &str,
) -> anyhow::Result<Vec<MarkerChoice>> {
    let file = fs::File::open(parquet_path)
        .with_context(|| format!("failed to open parquet: {}", parquet_path.to_string_lossy()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("failed to create parquet record batch reader builder")?;
    let schema = builder.schema();

    let names = schema
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect::<Vec<_>>();
    let mut desired = Vec::new();
    let mut median = Vec::new();
    let mut mean = Vec::new();
    let stat = marker_stat.trim().to_ascii_lowercase();
    let desired_suffix = format!("_{stat}_intensity");
    for n in names {
        if n.starts_with("marker_") && n.ends_with(&desired_suffix) {
            desired.push(n);
        } else if n.starts_with("marker_") && n.ends_with("_median_intensity") {
            median.push(n);
        } else if n.starts_with("marker_") && n.ends_with("_mean_intensity") {
            mean.push(n);
        }
    }

    let columns = if !desired.is_empty() {
        desired
    } else if !median.is_empty() {
        median
    } else {
        mean
    };
    let suffix = if !columns.is_empty() {
        if columns[0].ends_with("_median_intensity") {
            "_median_intensity"
        } else if columns[0].ends_with("_mean_intensity") {
            "_mean_intensity"
        } else if columns[0].ends_with(&desired_suffix) {
            desired_suffix.as_str()
        } else {
            "_median_intensity"
        }
    } else {
        "_median_intensity"
    };

    // Build canonical marker -> channel labels (in order) for nice display names.
    let mut available: HashMap<String, Vec<String>> = HashMap::new();
    for lbl in channel_labels {
        let base = base_marker_label(lbl);
        let canon = canonical_marker_token(&base);
        if canon.is_empty() {
            continue;
        }
        available.entry(canon).or_default().push(lbl.clone());
    }
    let channel_order: HashMap<String, usize> = channel_labels
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i))
        .collect();

    let mut out = Vec::new();
    let mut used_display: std::collections::HashSet<String> = std::collections::HashSet::new();
    for col in columns {
        let token = col
            .strip_prefix("marker_")
            .unwrap_or(&col)
            .strip_suffix(suffix)
            .unwrap_or(&col);
        let (marker_token, clone) = token.split_once("_C_").unwrap_or((token, ""));
        let marker_name = marker_token.replace('_', " ").trim().to_string();
        let canon = canonical_marker_token(&marker_name);

        let mut display = available
            .get_mut(&canon)
            .and_then(|v| (!v.is_empty()).then_some(v.remove(0)))
            .unwrap_or_else(|| marker_name.clone());
        if used_display.contains(&display) {
            let suffix = if !clone.is_empty() { clone } else { token };
            display = format!("{display} ({suffix})");
        }
        used_display.insert(display.clone());

        let marker_key = canonical_marker_token(token);
        out.push(MarkerChoice {
            display: display.clone(),
            column: col,
            marker_key,
        });
    }
    out.sort_by_key(|m| channel_order.get(&m.display).copied().unwrap_or(usize::MAX));
    Ok(out)
}

pub(super) fn read_channel_labels(
    dataset_root: &Path,
    channels_index_path: Option<&Path>,
) -> Vec<String> {
    let mut candidate: Option<PathBuf> = None;

    let (sample, roi_short) = infer_sample_and_roi_short(dataset_root).unwrap_or_default();
    if !sample.is_empty() && !roi_short.is_empty() {
        if let Some(index_path) = channels_index_path {
            if let Ok(text) = fs::read_to_string(index_path) {
                for line in text.lines() {
                    let raw = line.trim();
                    if raw.is_empty() {
                        continue;
                    }
                    let p = PathBuf::from(expand_tilde(raw));
                    let roi_dir = p.parent().unwrap_or_else(|| Path::new(""));
                    let sample_dir = roi_dir.parent().unwrap_or_else(|| Path::new(""));
                    let sample_i = sample_dir
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("");
                    let roi_i = roi_dir.file_name().and_then(|s| s.to_str()).unwrap_or("");
                    let roi_i = normalize_roi_label(roi_i);
                    if sample_i == sample && roi_i == roi_short {
                        candidate = Some(p);
                        break;
                    }
                }
            }
        }
    }

    if candidate.is_none() {
        if let Some(roi_dir) = dataset_root.parent() {
            let roi_name = roi_dir.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if !roi_name.is_empty() {
                let local = roi_dir.join(format!("{roi_name}.channels.txt"));
                if local.exists() {
                    candidate = Some(local);
                }
            }
        }
    }

    let Some(path) = candidate else {
        return Vec::new();
    };
    let Ok(text) = fs::read_to_string(&path) else {
        return Vec::new();
    };
    text.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect()
}

fn infer_sample_and_roi_short(dataset_root: &Path) -> Option<(String, String)> {
    let roi_dir = dataset_root.parent()?;
    let sample_dir = roi_dir.parent()?;
    let sample = sample_dir.file_name()?.to_str()?.trim().to_string();
    let roi_short = roi_dir
        .file_name()
        .and_then(|s| s.to_str())
        .map(normalize_roi_label)?;
    Some((sample, roi_short))
}

pub(super) fn infer_roi_label(dataset_root: &Path, ome_multiscale_name: Option<&str>) -> String {
    if let Some(name) = ome_multiscale_name {
        let n = normalize_roi_label(name);
        if !n.is_empty() {
            return n;
        }
    }
    let base = dataset_root
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();
    let stripped = base.strip_suffix(".ome.zarr").unwrap_or(&base);
    let candidate = if stripped.is_empty() {
        dataset_root
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string()
    } else {
        stripped.to_string()
    };
    normalize_roi_label(&candidate)
}

pub(super) fn infer_roi_label_with_layout(
    dataset_root: &Path,
    base_dir: &Path,
    layout: &str,
) -> String {
    let roi_short = infer_roi_label(dataset_root, None);
    if !layout.contains("sample") {
        return roi_short;
    }
    let rel = dataset_root.strip_prefix(base_dir).ok();
    if let Some(rel) = rel {
        let parts: Vec<String> = rel
            .components()
            .filter_map(|c| match c {
                std::path::Component::Normal(os) => os.to_str().map(|s| s.to_string()),
                _ => None,
            })
            .collect();
        if parts.len() >= 2 {
            let sample = parts[0].clone();
            let roi = normalize_roi_label(&parts[1]);
            if !sample.is_empty() && !roi.is_empty() {
                return format!("{sample}/{roi}");
            }
        }
    }
    if let Some((sample, roi)) = infer_sample_and_roi_short(dataset_root) {
        if !sample.is_empty() && !roi.is_empty() {
            return format!("{sample}/{roi}");
        }
    }
    roi_short
}

pub(super) fn normalize_roi_label(value: &str) -> String {
    // Minimal port of napari.gui/helpers.py normalize_roi_label.
    // Matches ROI, roi_001, roi-1 -> ROI1, etc.
    let text = value.trim();
    if text.is_empty() {
        return "".to_string();
    }
    if let Some((head, tail)) = text.rsplit_once('/') {
        let t = normalize_roi_label(tail);
        if t.is_empty() {
            return text.to_string();
        }
        return format!("{head}/{t}");
    }
    let lower = text.to_ascii_lowercase();
    if let Some(idx) = lower.find("roi") {
        let mut digits = String::new();
        for ch in lower[idx + 3..].chars() {
            if ch.is_ascii_digit() {
                digits.push(ch);
            }
        }
        if let Ok(n) = digits.parse::<u64>() {
            return format!("ROI{n}");
        }
    }
    text.to_string()
}

fn get_utf8<'a>(array: &'a dyn arrow_array::Array, row: usize) -> anyhow::Result<Option<&'a str>> {
    if array.is_null(row) {
        return Ok(None);
    }
    if let Some(col) = array.as_any().downcast_ref::<arrow_array::StringArray>() {
        return Ok(Some(col.value(row)));
    }
    if let Some(col) = array
        .as_any()
        .downcast_ref::<arrow_array::LargeStringArray>()
    {
        return Ok(Some(col.value(row)));
    }
    macro_rules! dict_utf8 {
        ($key:ty) => {
            if let Some(col) = array
                .as_any()
                .downcast_ref::<arrow_array::DictionaryArray<$key>>()
            {
                if col.is_null(row) {
                    return Ok(None);
                }
                let keys = col.keys();
                let key_i64 = keys.value(row) as i64;
                if key_i64 < 0 {
                    return Err(anyhow!("invalid dictionary key"));
                }
                return get_utf8(col.values().as_ref(), key_i64 as usize);
            }
        };
    }
    dict_utf8!(arrow_array::types::Int8Type);
    dict_utf8!(arrow_array::types::Int16Type);
    dict_utf8!(arrow_array::types::Int32Type);
    dict_utf8!(arrow_array::types::Int64Type);
    dict_utf8!(arrow_array::types::UInt8Type);
    dict_utf8!(arrow_array::types::UInt16Type);
    dict_utf8!(arrow_array::types::UInt32Type);
    dict_utf8!(arrow_array::types::UInt64Type);

    Err(anyhow!(
        "unsupported utf8-like column type: {}",
        array.data_type()
    ))
}

fn get_f64(array: &dyn arrow_array::Array, row: usize) -> anyhow::Result<Option<f64>> {
    if array.is_null(row) {
        return Ok(None);
    }
    macro_rules! prim {
        ($ty:ty) => {
            if let Some(col) = array.as_any().downcast_ref::<$ty>() {
                if col.is_null(row) {
                    return Ok(None);
                }
                return Ok(Some(col.value(row) as f64));
            }
        };
    }
    prim!(arrow_array::Float64Array);
    prim!(arrow_array::Float32Array);
    prim!(arrow_array::Int64Array);
    prim!(arrow_array::Int32Array);
    prim!(arrow_array::Int16Array);
    prim!(arrow_array::Int8Array);
    prim!(arrow_array::UInt64Array);
    prim!(arrow_array::UInt32Array);
    prim!(arrow_array::UInt16Array);
    prim!(arrow_array::UInt8Array);

    macro_rules! dict_num {
        ($key:ty) => {
            if let Some(col) = array
                .as_any()
                .downcast_ref::<arrow_array::DictionaryArray<$key>>()
            {
                if col.is_null(row) {
                    return Ok(None);
                }
                let keys = col.keys();
                let key_i64 = keys.value(row) as i64;
                if key_i64 < 0 {
                    return Err(anyhow!("invalid dictionary key"));
                }
                return get_f64(col.values().as_ref(), key_i64 as usize);
            }
        };
    }
    dict_num!(arrow_array::types::Int8Type);
    dict_num!(arrow_array::types::Int16Type);
    dict_num!(arrow_array::types::Int32Type);
    dict_num!(arrow_array::types::Int64Type);
    dict_num!(arrow_array::types::UInt8Type);
    dict_num!(arrow_array::types::UInt16Type);
    dict_num!(arrow_array::types::UInt32Type);
    dict_num!(arrow_array::types::UInt64Type);

    Err(anyhow!(
        "unsupported numeric column type for f64 conversion: {}",
        array.data_type()
    ))
}

pub(super) fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest).to_string_lossy().to_string();
        }
    }
    path.to_string()
}

pub(super) fn loosely_matches(a: &str, b: &str) -> bool {
    let norm = |s: &str| {
        s.chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect::<String>()
    };
    norm(a).contains(&norm(b)) || norm(b).contains(&norm(a))
}

fn sanitize_marker_token(label: &str) -> String {
    let mut out = String::with_capacity(label.len());
    let mut prev_us = false;
    for ch in label.chars() {
        let ok = ch.is_ascii_alphanumeric() || ch == '_';
        let c = if ok { ch } else { '_' };
        if c == '_' {
            if prev_us {
                continue;
            }
            prev_us = true;
        } else {
            prev_us = false;
        }
        out.push(c);
    }
    out
}

pub(super) fn canonical_marker_token(label: &str) -> String {
    sanitize_marker_token(label)
        .trim_matches('_')
        .to_ascii_lowercase()
}

pub(super) fn base_marker_label(label: &str) -> String {
    // Port of napari.gui/channels.py base_marker_label.
    let text = label.trim();
    if text.is_empty() {
        return String::new();
    }
    // Split on " C " / " c " (clone separator).
    let mut head = text.to_string();
    for sep in [" c ", " C "] {
        if let Some((h, _)) = format!(" {text} ").split_once(sep) {
            head = h.trim().to_string();
            break;
        }
    }
    // Strip "C008 - " prefix style.
    if let Some(rest) = head.strip_prefix('C') {
        let rest = rest.trim_start_matches(|c: char| c.is_ascii_digit() || c.is_whitespace());
        if let Some(rest) = rest.strip_prefix('-') {
            head = rest.trim().to_string();
        }
    }
    // Drop trailing metadata "(...)" or "[...]".
    if let Some((h, _)) = head.split_once('(') {
        head = h.trim().to_string();
    }
    if let Some((h, _)) = head.split_once('[') {
        head = h.trim().to_string();
    }
    head
}

fn finite_min_max(values: &[f32]) -> Option<(f32, f32)> {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in values {
        if !v.is_finite() {
            continue;
        }
        min = min.min(v);
        max = max.max(v);
    }
    if min.is_finite() && max.is_finite() {
        Some((min, max))
    } else {
        None
    }
}
