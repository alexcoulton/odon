use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{Context, anyhow};

#[derive(Debug, Clone)]
pub struct SampleRow {
    pub id: String,
    pub path: PathBuf,
    pub meta: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct SampleSheet {
    /// Metadata column headers (excludes the first 2 required columns: id, path).
    pub meta_columns: Vec<String>,
    pub rows: Vec<SampleRow>,
}

pub fn load_samplesheet_csv(path: &Path) -> anyhow::Result<SampleSheet> {
    let base_dir = path.parent().map(|p| p.to_path_buf());
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open samplesheet CSV: {}", path.to_string_lossy()))?;

    let headers = reader
        .headers()
        .context("samplesheet missing CSV header row")?
        .clone();
    if headers.len() < 2 {
        anyhow::bail!("samplesheet must have at least 2 columns: id, path");
    }

    let meta_columns: Vec<String> = headers
        .iter()
        .skip(2)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let mut rows = Vec::new();
    for (i, rec) in reader.records().enumerate() {
        let rec = rec.with_context(|| format!("failed to read samplesheet row {}", i + 1))?;
        if rec.len() < 2 {
            return Err(anyhow!("samplesheet row {} has <2 columns", i + 1));
        }
        let id = rec.get(0).unwrap_or("").trim().to_string();
        let path_str = rec.get(1).unwrap_or("").trim().to_string();
        if id.is_empty() || path_str.is_empty() {
            continue;
        }

        let mut meta = HashMap::new();
        for (col_idx, col_name) in headers.iter().enumerate().skip(2) {
            let key = col_name.trim();
            if key.is_empty() {
                continue;
            }
            let value = rec.get(col_idx).unwrap_or("").trim().to_string();
            meta.insert(key.to_string(), value);
        }

        let row_path = PathBuf::from(path_str);
        let row_path = if row_path.is_relative() {
            base_dir
                .as_ref()
                .map(|dir| dir.join(&row_path))
                .unwrap_or(row_path)
        } else {
            row_path
        };

        rows.push(SampleRow {
            id,
            path: row_path,
            meta,
        });
    }

    if rows.is_empty() {
        anyhow::bail!("samplesheet contained no usable rows (need non-empty id and path)");
    }

    Ok(SampleSheet { meta_columns, rows })
}

pub fn write_samplesheet_csv(path: &Path, sheet: &SampleSheet) -> anyhow::Result<()> {
    if sheet.rows.is_empty() {
        anyhow::bail!("samplesheet has no rows to export");
    }

    let mut writer = csv::Writer::from_path(path).with_context(|| {
        format!(
            "failed to create samplesheet CSV: {}",
            path.to_string_lossy()
        )
    })?;

    let mut meta_columns = if sheet.meta_columns.is_empty() {
        let mut cols = BTreeSet::new();
        for row in &sheet.rows {
            for key in row.meta.keys() {
                let key = key.trim();
                if !key.is_empty() {
                    cols.insert(key.to_string());
                }
            }
        }
        cols.into_iter().collect::<Vec<_>>()
    } else {
        sheet.meta_columns.clone()
    };
    meta_columns.sort();
    meta_columns.dedup();

    let mut header = vec!["id".to_string(), "path".to_string()];
    header.extend(meta_columns.iter().cloned());
    writer
        .write_record(&header)
        .with_context(|| format!("failed to write header to {}", path.to_string_lossy()))?;

    for row in &sheet.rows {
        let mut record = vec![row.id.clone(), row.path.to_string_lossy().to_string()];
        for key in &meta_columns {
            record.push(row.meta.get(key).cloned().unwrap_or_default());
        }
        writer.write_record(&record).with_context(|| {
            format!(
                "failed to write row '{}' to {}",
                row.id,
                path.to_string_lossy()
            )
        })?;
    }

    writer.flush().with_context(|| {
        format!(
            "failed to finalize samplesheet CSV: {}",
            path.to_string_lossy()
        )
    })
}
