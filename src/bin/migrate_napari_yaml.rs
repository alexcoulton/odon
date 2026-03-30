use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, anyhow};
use serde::Deserialize;

use odon::dataset_source::DatasetSource;
use odon::project_config::{ProjectConfig, ProjectDatasetConfig, ProjectRoi};

#[derive(Debug, Clone, Deserialize)]
struct NapariGuiYaml {
    #[serde(default)]
    use_full_res_ome_zarrs: bool,
    #[serde(default)]
    tumour_antigen_only_mode: bool,
    #[serde(default)]
    default_dataset: Option<String>,
    #[serde(default)]
    secondary_dataset: Option<String>,
    #[serde(default)]
    clinical_annotations_parquet: Option<String>,
    #[serde(default)]
    default_threshold_marker: Option<String>,
    #[serde(default)]
    cluster_column: Option<String>,
    #[serde(default)]
    unlabeled_cluster_value: Option<String>,
    #[serde(default)]
    all_clusters_label: Option<String>,
    #[serde(default)]
    tma1_orig_cells_parquet: Option<String>,
    #[serde(default)]
    datasets: HashMap<String, NapariDatasetYaml>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct NapariDatasetYaml {
    #[serde(default)]
    layout: Option<String>,
    #[serde(default)]
    samplesheet_path: Option<String>,
    #[serde(default)]
    channel_mode: Option<String>,
    #[serde(default)]
    subset_channel_labels: Option<Vec<String>>,

    #[serde(default)]
    base_dir_full_res: Option<String>,
    #[serde(default)]
    base_dir_downsampled: Option<String>,
    #[serde(default)]
    roi_start: Option<u64>,
    #[serde(default)]
    roi_end: Option<u64>,
    #[serde(default)]
    sample_id: Option<String>,
    #[serde(default)]
    include_all_samples: Option<bool>,

    #[serde(default)]
    cells_backend: Option<String>,
    #[serde(default)]
    cells_parquet: Option<String>,
    #[serde(default)]
    cells_parquet_dir: Option<String>,
    #[serde(default)]
    cells_parquet_dir_flatfield: Option<String>,
    #[serde(default)]
    channels_index_path: Option<String>,
    #[serde(default)]
    needs_correction_csv: Option<String>,

    #[serde(default)]
    thresholds_csv: Option<String>,
    #[serde(default)]
    auto_thresholds_json: Option<String>,

    #[serde(default)]
    masks_dir: Option<String>,
    #[serde(default)]
    coord_downsample_full_res: Option<f32>,
    #[serde(default)]
    coord_downsample_downsampled: Option<f32>,
    #[serde(default)]
    masks_downsample_full_res: Option<f32>,
    #[serde(default)]
    masks_downsample_downsampled: Option<f32>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct ProjectFileV5 {
    version: u32,
    #[serde(default)]
    config: ProjectConfig,
    focused: Option<String>,
    selected: Vec<String>,
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut yaml_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--yaml" => {
                yaml_path = args.next().map(PathBuf::from);
            }
            "--out" => {
                out_path = args.next().map(PathBuf::from);
            }
            "-h" | "--help" => {
                eprintln!(
                    r#"migrate_napari_yaml

Usage:
  cargo run --bin migrate_napari_yaml -- --yaml "/path/to/config.yaml" --out "/path/to/project.json"

Notes:
  - Generates a Project JSON (v4) containing:
    - `config.datasets[...]` mapped from the YAML
    - an explicit `config.rois[]` list by scanning base dirs / ROI ranges in the YAML
"#
                );
                return Ok(());
            }
            other => {
                if other.starts_with("--yaml=") {
                    yaml_path = other.strip_prefix("--yaml=").map(PathBuf::from);
                } else if other.starts_with("--out=") {
                    out_path = other.strip_prefix("--out=").map(PathBuf::from);
                }
            }
        }
    }

    let yaml_path = yaml_path.context("missing --yaml /path/to/config.yaml")?;
    let yaml_text = fs::read_to_string(&yaml_path)
        .with_context(|| format!("failed to read YAML: {}", yaml_path.to_string_lossy()))?;

    let napari: NapariGuiYaml =
        yaml_serde::from_str(&yaml_text).context("failed to parse napari config.yaml")?;

    let mut cfg = ProjectConfig::default();
    cfg.use_full_res_ome_zarrs = napari.use_full_res_ome_zarrs;
    cfg.tumour_antigen_only_mode = napari.tumour_antigen_only_mode;
    cfg.default_dataset = napari.default_dataset.clone();
    cfg.secondary_dataset = napari.secondary_dataset.clone();
    cfg.clinical_annotations_parquet = napari.clinical_annotations_parquet.clone();
    cfg.default_threshold_marker = napari.default_threshold_marker.clone();
    cfg.cluster_column = napari.cluster_column.clone();
    cfg.unlabeled_cluster_value = napari.unlabeled_cluster_value.clone();
    cfg.all_clusters_label = napari.all_clusters_label.clone();
    cfg.tma1_orig_cells_parquet = napari.tma1_orig_cells_parquet.clone();

    // Map dataset configs.
    for (name, ds) in &napari.datasets {
        cfg.datasets.insert(name.clone(), map_dataset_config(ds));
    }

    // Build explicit ROI list (runtime uses explicit list; YAML had discovery hints).
    let project_dir = yaml_path.parent().map(|p| p.to_path_buf());
    let mut rois: Vec<ProjectRoi> = Vec::new();
    for (name, ds) in &napari.datasets {
        let ds_rois = discover_rois_for_dataset(
            name,
            ds,
            napari.use_full_res_ome_zarrs,
            project_dir.as_deref(),
        )
        .with_context(|| format!("failed to discover ROIs for dataset '{name}'"))?;
        rois.extend(ds_rois);
    }

    rois.sort_by(|a, b| a.id.cmp(&b.id));
    rois.dedup_by(|a, b| a.source_key() == b.source_key());
    cfg.rois = rois;

    let focused = cfg.rois.first().and_then(ProjectRoi::source_key);
    let selected = focused.clone().into_iter().collect::<Vec<_>>();

    let out_path = out_path.unwrap_or_else(|| {
        yaml_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("odon.project.json")
    });

    let file = ProjectFileV5 {
        version: 5,
        config: cfg,
        focused,
        selected,
    };

    let json = serde_json::to_string_pretty(&file).context("failed to serialize Project JSON")?;
    fs::write(&out_path, json)
        .with_context(|| format!("failed to write: {}", out_path.to_string_lossy()))?;

    eprintln!("Wrote: {}", out_path.to_string_lossy());
    Ok(())
}

fn map_dataset_config(ds: &NapariDatasetYaml) -> ProjectDatasetConfig {
    ProjectDatasetConfig {
        layout: ds.layout.clone(),
        samplesheet_path: ds.samplesheet_path.clone(),
        channel_mode: ds.channel_mode.clone(),
        base_dir_full_res: ds.base_dir_full_res.clone(),
        base_dir_downsampled: ds.base_dir_downsampled.clone(),
        roi_start: ds.roi_start,
        roi_end: ds.roi_end,
        sample_id: ds.sample_id.clone(),
        include_all_samples: ds.include_all_samples,
        cells_backend: ds.cells_backend.clone(),
        cells_parquet: ds.cells_parquet.clone(),
        cells_parquet_dir: ds.cells_parquet_dir.clone(),
        cells_parquet_dir_flatfield: ds.cells_parquet_dir_flatfield.clone(),
        channels_index_path: ds.channels_index_path.clone(),
        needs_correction_csv: ds.needs_correction_csv.clone(),
        coord_downsample_full_res: ds.coord_downsample_full_res,
        coord_downsample_downsampled: ds.coord_downsample_downsampled,
        subset_channel_labels: ds.subset_channel_labels.clone(),
        thresholds_csv: ds.thresholds_csv.clone(),
        auto_thresholds_json: ds.auto_thresholds_json.clone(),
        masks_dir: ds.masks_dir.clone(),
        masks_downsample_full_res: ds.masks_downsample_full_res,
        masks_downsample_downsampled: ds.masks_downsample_downsampled,
    }
}

fn discover_rois_for_dataset(
    dataset_name: &str,
    ds: &NapariDatasetYaml,
    prefer_full_res: bool,
    project_dir: Option<&Path>,
) -> anyhow::Result<Vec<ProjectRoi>> {
    let base = pick_base_dir(ds, prefer_full_res, project_dir)
        .with_context(|| format!("dataset '{dataset_name}' missing base_dir_*"))?;

    let layout = ds
        .layout
        .as_deref()
        .unwrap_or("flat_roi")
        .trim()
        .to_ascii_lowercase();

    let mut out = Vec::new();
    match layout.as_str() {
        "flat_roi" => {
            let start = ds.roi_start.unwrap_or(1);
            let end = ds.roi_end.unwrap_or(start);
            let lo = start.min(end);
            let hi = start.max(end);
            for i in lo..=hi {
                let roi_id = format!("ROI{i}");
                let root = base.join(&roi_id).join(format!("{roi_id}.ome.zarr"));
                if looks_like_omezarr_root(&root) {
                    let mut roi = ProjectRoi {
                        id: roi_id.clone(),
                        source: None,
                        path: None,
                        dataset: Some(dataset_name.to_string()),
                        display_name: Some(roi_id),
                        segpath: None,
                        mask_layers: Vec::new(),
                        channel_order: Vec::new(),
                        meta: Default::default(),
                    };
                    roi.set_dataset_source(DatasetSource::Local(canonical_or(root)));
                    out.push(roi);
                }
            }
        }
        "sample_roi" => {
            let include_all = ds.include_all_samples.unwrap_or(true);
            let mut sample_dirs: Vec<PathBuf> = Vec::new();
            if let Some(sample_id) = ds.sample_id.as_deref().filter(|s| !s.trim().is_empty()) {
                let sample_dir = resolve_maybe_relative(&base, sample_id, project_dir);
                if sample_dir.is_dir() {
                    sample_dirs.push(sample_dir);
                }
            } else if include_all {
                if let Ok(rd) = fs::read_dir(&base) {
                    for ent in rd.flatten() {
                        let p = ent.path();
                        if p.is_dir() {
                            sample_dirs.push(p);
                        }
                    }
                }
            } else {
                // include_all_samples=false but no sample_id: keep first sample dir (sorted).
                if let Ok(rd) = fs::read_dir(&base) {
                    for ent in rd.flatten() {
                        let p = ent.path();
                        if p.is_dir() {
                            sample_dirs.push(p);
                        }
                    }
                }
                sample_dirs.sort();
                if sample_dirs.len() > 1 {
                    sample_dirs.truncate(1);
                }
            }

            sample_dirs.sort();
            for sample_dir in sample_dirs {
                let sample = sample_dir
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();
                if let Ok(rd) = fs::read_dir(&sample_dir) {
                    for ent in rd.flatten() {
                        let roi_dir = ent.path();
                        if !roi_dir.is_dir() {
                            continue;
                        }
                        let roi_short = roi_dir
                            .file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or("")
                            .to_string();
                        if !roi_short.to_ascii_lowercase().starts_with("roi") {
                            continue;
                        }
                        let root = roi_dir.join(format!("{roi_short}.ome.zarr"));
                        if !looks_like_omezarr_root(&root) {
                            continue;
                        }
                        let roi_norm = normalize_roi_label(&roi_short);
                        let id = if !sample.is_empty() {
                            format!("{sample}/{roi_norm}")
                        } else {
                            roi_norm.clone()
                        };
                        let mut roi = ProjectRoi {
                            id: id.clone(),
                            source: None,
                            path: None,
                            dataset: Some(dataset_name.to_string()),
                            display_name: Some(id),
                            segpath: None,
                            mask_layers: Vec::new(),
                            channel_order: Vec::new(),
                            meta: Default::default(),
                        };
                        roi.set_dataset_source(DatasetSource::Local(canonical_or(root)));
                        out.push(roi);
                    }
                }
            }
        }
        other => return Err(anyhow!("unsupported layout '{other}'")),
    }

    Ok(out)
}

fn pick_base_dir(
    ds: &NapariDatasetYaml,
    prefer_full_res: bool,
    project_dir: Option<&Path>,
) -> Option<PathBuf> {
    let mut candidates: Vec<&str> = Vec::new();
    if prefer_full_res {
        if let Some(p) = ds.base_dir_full_res.as_deref() {
            candidates.push(p);
        }
        if let Some(p) = ds.base_dir_downsampled.as_deref() {
            candidates.push(p);
        }
    } else {
        if let Some(p) = ds.base_dir_downsampled.as_deref() {
            candidates.push(p);
        }
        if let Some(p) = ds.base_dir_full_res.as_deref() {
            candidates.push(p);
        }
    }

    for raw in candidates {
        let expanded = expand_tilde(raw);
        let p = if Path::new(&expanded).is_absolute() {
            PathBuf::from(expanded)
        } else {
            project_dir
                .map(|d| d.join(&expanded))
                .unwrap_or_else(|| PathBuf::from(expanded))
        };
        if p.is_dir() {
            return Some(p);
        }
    }
    None
}

fn resolve_maybe_relative(base_dir: &Path, raw: &str, project_dir: Option<&Path>) -> PathBuf {
    let expanded = expand_tilde(raw);
    let p = PathBuf::from(&expanded);
    if p.is_absolute() {
        return p;
    }
    // Prefer resolving relative to the dataset base directory.
    let p2 = base_dir.join(&expanded);
    if p2.is_dir() {
        return p2;
    }
    project_dir
        .map(|d| d.join(&expanded))
        .unwrap_or_else(|| PathBuf::from(expanded))
}

fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest).to_string_lossy().to_string();
        }
    }
    path.to_string()
}

fn canonical_or(p: PathBuf) -> PathBuf {
    p.canonicalize().unwrap_or(p)
}

fn looks_like_omezarr_root(root: &Path) -> bool {
    root.is_dir() && (root.join(".zattrs").is_file() || root.join("zarr.json").is_file())
}

fn normalize_roi_label(value: &str) -> String {
    let text = value.trim();
    if text.is_empty() {
        return "".to_string();
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
