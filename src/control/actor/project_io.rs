use super::*;

pub(super) fn inspect_samplesheet(path: &std::path::Path, offset: usize, limit: usize) -> Value {
    let sheet = match load_samplesheet_csv(path) {
        Ok(sheet) => sheet,
        Err(error) => {
            return json!({
                "valid": false,
                "path": path.to_string_lossy(),
                "error": format!("failed to parse samplesheet: {error}"),
            });
        }
    };
    let mut seen = HashSet::new();
    let duplicate_ids = sheet
        .rows
        .iter()
        .filter_map(|row| (!seen.insert(row.id.clone())).then(|| row.id.clone()))
        .collect::<Vec<_>>();
    let missing_source_count = sheet.rows.iter().filter(|row| !row.path.exists()).count();
    let total = sheet.rows.len();
    let rows = sheet
        .rows
        .iter()
        .skip(offset)
        .take(limit)
        .map(|row| {
            json!({
                "id": row.id,
                "path": row.path.to_string_lossy(),
                "resolved_path": row.path.to_string_lossy(),
                "exists": row.path.exists(),
                "kind": classify_local_dataset_path(&row.path).map(|kind| match kind {
                    LocalDatasetKind::OmeZarr => "ome_zarr",
                    LocalDatasetKind::Tiff => "tiff",
                    LocalDatasetKind::Xenium => "xenium",
                }),
                "metadata": row.meta,
            })
        })
        .collect::<Vec<_>>();
    json!({
        "valid": duplicate_ids.is_empty(),
        "path": path.to_string_lossy(),
        "metadata_columns": sheet.meta_columns,
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": offset.saturating_add(rows.len()) < total,
        "missing_source_count": missing_source_count,
        "duplicate_ids": duplicate_ids,
        "rows": rows,
    })
}

pub(super) fn import_samplesheet_rois(
    path: &std::path::Path,
    default_dataset: &str,
) -> anyhow::Result<Vec<ProjectRoi>> {
    let sheet = load_samplesheet_csv(path)?;
    let base_dir = path.parent();
    let mut rois = Vec::with_capacity(sheet.rows.len());
    for row in sheet.rows {
        let meta = row.meta;
        let resolved_path = if row.path.is_relative() {
            base_dir.map_or(row.path.clone(), |dir| dir.join(&row.path))
        } else {
            row.path
        };
        let resolved_path = resolved_path.canonicalize().unwrap_or(resolved_path);
        let segpath = meta
            .get("segpath")
            .filter(|value| !value.trim().is_empty())
            .map(PathBuf::from)
            .map(|segmentation| {
                if segmentation.is_relative() {
                    base_dir.map_or(segmentation.clone(), |dir| dir.join(&segmentation))
                } else {
                    segmentation
                }
            })
            .map(|segmentation| segmentation.canonicalize().unwrap_or(segmentation));
        let dataset = meta
            .get("dataset")
            .filter(|value| !value.trim().is_empty())
            .cloned()
            .or_else(|| Some(default_dataset.to_string()));
        let mut roi = ProjectRoi {
            id: row.id.clone(),
            display_name: Some(row.id),
            dataset,
            segpath,
            meta,
            ..ProjectRoi::default()
        };
        roi.set_dataset_source(DatasetSource::Local(resolved_path));
        rois.push(roi);
    }
    Ok(rois)
}

pub(super) fn export_samplesheet_rois(
    path: &std::path::Path,
    rois: &[ProjectRoi],
    overwrite: bool,
) -> anyhow::Result<u64> {
    if path.exists() && !overwrite {
        anyhow::bail!("destination exists; pass overwrite=true to replace it");
    }
    let mut meta_columns = BTreeSet::new();
    let mut rows = Vec::new();
    for roi in rois {
        let Some(local_path) = roi.local_path() else {
            continue;
        };
        let mut meta = roi
            .meta
            .iter()
            .filter_map(|(key, value)| {
                let key = key.trim();
                (!key.is_empty()).then(|| (key.to_string(), value.clone()))
            })
            .collect::<std::collections::HashMap<_, _>>();
        if let Some(dataset) = roi
            .dataset
            .as_ref()
            .filter(|value| !value.trim().is_empty())
        {
            meta.insert("dataset".to_string(), dataset.clone());
        }
        if let Some(segpath) = roi.segpath.as_ref() {
            let segpath = segpath
                .canonicalize()
                .unwrap_or_else(|_| segpath.to_path_buf());
            meta.insert("segpath".to_string(), segpath.to_string_lossy().to_string());
        }
        meta_columns.extend(meta.keys().cloned());
        let id = if roi.id.trim().is_empty() {
            roi.display_name
                .clone()
                .filter(|value| !value.trim().is_empty())
                .or_else(|| {
                    local_path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .map(str::to_string)
                })
                .unwrap_or_else(|| "ROI".to_string())
        } else {
            roi.id.trim().to_string()
        };
        rows.push(SampleRow {
            id,
            path: local_path
                .canonicalize()
                .unwrap_or_else(|_| local_path.to_path_buf()),
            meta,
        });
    }
    if rows.is_empty() {
        anyhow::bail!("project has no local ROIs to export");
    }
    write_samplesheet_csv(
        path,
        &SampleSheet {
            meta_columns: meta_columns.into_iter().collect(),
            rows,
        },
    )?;
    Ok(fs::metadata(path)?.len())
}

pub(super) fn discover_omezarr_roots_under(root: &std::path::Path) -> anyhow::Result<Vec<PathBuf>> {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    if !root.is_dir() {
        anyhow::bail!("not a directory: {}", root.display());
    }
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        let Ok(read_dir) = fs::read_dir(&dir) else {
            continue;
        };
        let mut is_omezarr_root = false;
        let mut child_dirs = Vec::new();
        for entry in read_dir.flatten() {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_file()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name == ".zattrs" || name == "zarr.json")
            {
                is_omezarr_root = true;
            } else if file_type.is_dir() {
                child_dirs.push(path);
            }
        }
        if is_omezarr_root {
            let canonical = dir.canonicalize().unwrap_or(dir);
            if seen.insert(canonical.clone()) {
                out.push(canonical);
            }
        } else {
            stack.extend(child_dirs);
        }
    }
    out.sort();
    if out.is_empty() {
        anyhow::bail!("no OME-Zarr datasets found under {}", root.display());
    }
    Ok(out)
}

pub(super) fn read_project_file(path: &std::path::Path) -> anyhow::Result<(ProjectConfig, Value)> {
    let text = fs::read_to_string(path)?;
    let file: Value = serde_json::from_str(&text)?;
    let version = file.get("version").and_then(Value::as_u64).unwrap_or(1);
    let browser = |focused: Value, selected: Value| {
        json!({
            "browser": {
                "focused": focused,
                "selected": selected,
            }
        })
    };
    match version {
        1 => {
            let config = legacy_project_config(&file)?;
            let selected_index = file
                .get("selected")
                .and_then(Value::as_u64)
                .map(|index| index as usize);
            let focused = selected_index
                .and_then(|index| config.rois.get(index))
                .and_then(ProjectRoi::source_key)
                .map(Value::String)
                .unwrap_or(Value::Null);
            Ok((config, browser(focused, json!([]))))
        }
        2 | 3 => {
            let config = legacy_project_config(&file)?;
            let focused = file
                .get("focused")
                .and_then(Value::as_str)
                .map(|path| Value::String(local_source_key(path)))
                .unwrap_or(Value::Null);
            let selected = file
                .get("selected")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(local_source_key)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            Ok((config, browser(focused, json!(selected))))
        }
        4 => {
            let config = serde_json::from_value::<ProjectConfig>(
                file.get("config").cloned().unwrap_or_else(|| json!({})),
            )?;
            let focused = file
                .get("focused")
                .and_then(Value::as_str)
                .map(|path| Value::String(local_source_key(path)))
                .unwrap_or(Value::Null);
            let selected = file
                .get("selected")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(local_source_key)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            Ok((config, browser(focused, json!(selected))))
        }
        5 => {
            let config = serde_json::from_value::<ProjectConfig>(
                file.get("config").cloned().unwrap_or_else(|| json!({})),
            )?;
            Ok((
                config,
                browser(
                    file.get("focused").cloned().unwrap_or(Value::Null),
                    file.get("selected").cloned().unwrap_or_else(|| json!([])),
                ),
            ))
        }
        6 => Ok((
            serde_json::from_value::<ProjectConfig>(
                file.get("config").cloned().unwrap_or_else(|| json!({})),
            )?,
            file.get("state").cloned().unwrap_or_else(|| json!({})),
        )),
        version => anyhow::bail!("unsupported project version: {version}"),
    }
}

fn legacy_project_config(file: &Value) -> anyhow::Result<ProjectConfig> {
    let items = file
        .get("items")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("legacy project has no items array"))?;
    let rois = items
        .iter()
        .map(|item| {
            let path = item
                .get("path")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("legacy project item has no path"))?;
            let display_name = item
                .get("display_name")
                .and_then(Value::as_str)
                .map(str::to_string);
            let mut roi = ProjectRoi {
                id: display_name.clone().unwrap_or_else(|| path.to_string()),
                display_name,
                ..ProjectRoi::default()
            };
            roi.set_dataset_source(crate::data::dataset_source::DatasetSource::Local(
                PathBuf::from(path),
            ));
            Ok(roi)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(ProjectConfig {
        rois,
        ..ProjectConfig::default()
    })
}

fn local_source_key(path: &str) -> String {
    crate::data::dataset_source::DatasetSource::Local(PathBuf::from(path)).source_key()
}
