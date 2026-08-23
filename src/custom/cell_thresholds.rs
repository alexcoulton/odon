mod data;
mod threshold_files;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use eframe::egui;

use data::{
    base_marker_label, best_base_dir_for_root, canonical_marker_token, expand_tilde,
    infer_roi_label, infer_roi_label_with_layout, list_marker_choices, loosely_matches,
    normalize_roi_label, parquet_path_for_zarr_root, project_dataset_key_for_root,
    read_channel_labels, spawn_loader_thread,
};
use threshold_files::{load_auto_thresholds_json, load_thresholds_csv};

use crate::data::project_config::ProjectConfig;
use crate::render::points::{Point, PointsLayer};

#[derive(Debug, Clone)]
pub struct CellThresholdsPanel {
    enabled: bool,
    status: String,

    dataset_root: PathBuf,
    project: ProjectConfig,
    dataset_name: Option<String>,
    channels_index_path: Option<PathBuf>,
    parquet_path: Option<PathBuf>,
    coord_downsample: f32,

    roi_label: String,

    // Multi-source parquet backends can provide multiple metric sources (standard vs flatfield).
    parquet_dir_standard: Option<PathBuf>,
    parquet_dir_flatfield: Option<PathBuf>,
    cells_source: CellsSource,
    cells_source_available: Vec<CellsSource>,

    marker_choices: Vec<MarkerChoice>,
    marker_base_lookup: HashMap<String, usize>,
    selected_marker: usize,
    threshold: f32,
    values_min: f32,
    values_max: f32,
    positions_world: Arc<Vec<egui::Pos2>>,
    values: Arc<Vec<f32>>,
    values_generation: u64,
    last_loaded_key: Option<LoadKey>,
    positive_count: usize,
    total_count: usize,
    points_visible: bool,

    thresholds_csv_path: Option<PathBuf>,
    thresholds_loaded: HashMap<(String, String, String), ThresholdCsvRow>,

    auto_thresholds_path: Option<PathBuf>,
    auto_thresholds: HashMap<(String, String), AutoThresholdRecord>,
    auto_method: AutoMethod,
    auto_positive_ge: u8,
    auto_kmeans_k: u8,
    marker_stat: String,

    load_request_id: u64,
    tx: crossbeam_channel::Sender<LoadRequest>,
    rx: crossbeam_channel::Receiver<LoadResponse>,
    last_loaded_request_id: u64,
}

#[derive(Debug, Clone)]
struct MarkerChoice {
    display: String,
    column: String,
    marker_key: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AutoMethod {
    Manual,
    KMeans,
    Otsu,
}

#[derive(Debug, Clone)]
struct AutoThreshold {
    kmeans_cutoffs_arcsinh: Vec<f32>,
    otsu_arcsinh: Option<f32>,
    kmeans_k: u8,
}

#[derive(Debug, Clone)]
struct AutoThresholdRecord {
    preferred_source: Option<String>,
    sources: HashMap<String, AutoThreshold>,
}

#[derive(Debug, Clone)]
struct ThresholdCsvRow {
    arcsinh_threshold: f32,
    method: String,
    kmeans_k: Option<u8>,
    positive_ge: Option<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CellsSource {
    Standard,
    Flatfield,
}

impl CellsSource {
    fn as_str(self) -> &'static str {
        match self {
            CellsSource::Standard => "standard",
            CellsSource::Flatfield => "flatfield",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LoadKey {
    roi_label: String,
    marker_column: String,
    coord_downsample_bits: u32,
}

#[derive(Debug, Clone)]
struct LoadRequest {
    request_id: u64,
    parquet_path: PathBuf,
    key: LoadKey,
}

#[derive(Debug, Clone)]
struct LoadResponse {
    request_id: u64,
    key: LoadKey,
    positions: Vec<egui::Pos2>,
    values: Vec<f32>,
    min: f32,
    max: f32,
}

impl CellThresholdsPanel {
    pub fn set_project_config(&mut self, project: ProjectConfig) {
        self.project = project;
        let root = self.dataset_root.clone();
        self.reload_config(&root);
    }

    pub fn new(dataset_root: &Path, ome_multiscale_name: Option<&str>) -> Self {
        let roi_label = infer_roi_label(dataset_root, ome_multiscale_name);

        let (tx, rx) = spawn_loader_thread();

        let mut panel = Self {
            enabled: false,
            status: "Not configured.".to_string(),
            dataset_root: dataset_root.to_path_buf(),
            project: ProjectConfig::default(),
            dataset_name: None,
            channels_index_path: None,
            parquet_path: None,
            coord_downsample: 1.0,
            roi_label,
            parquet_dir_standard: None,
            parquet_dir_flatfield: None,
            cells_source: CellsSource::Standard,
            cells_source_available: Vec::new(),
            marker_choices: Vec::new(),
            marker_base_lookup: HashMap::new(),
            selected_marker: 0,
            threshold: 0.0,
            values_min: 0.0,
            values_max: 1.0,
            positions_world: Arc::new(Vec::new()),
            values: Arc::new(Vec::new()),
            values_generation: 0,
            last_loaded_key: None,
            positive_count: 0,
            total_count: 0,
            points_visible: true,
            thresholds_csv_path: None,
            thresholds_loaded: HashMap::new(),
            auto_thresholds_path: None,
            auto_thresholds: HashMap::new(),
            auto_method: AutoMethod::Manual,
            auto_positive_ge: 5,
            auto_kmeans_k: 6,
            marker_stat: "median".to_string(),
            load_request_id: 0,
            tx,
            rx,
            last_loaded_request_id: 0,
        };

        panel.reload_config(dataset_root);
        panel
    }

    pub fn tick(&mut self, points_layer: &mut PointsLayer) {
        // Keep panel state in sync with the global layer visibility toggle.
        self.points_visible = points_layer.visible;
        self.drain_loader(points_layer);
    }

    pub fn gpu_points(&self) -> Option<(u64, Arc<Vec<egui::Pos2>>, Arc<Vec<f32>>)> {
        if !self.enabled || !self.points_visible {
            return None;
        }
        if self.positions_world.is_empty() || self.values.is_empty() {
            return None;
        }
        Some((
            self.values_generation,
            Arc::clone(&self.positions_world),
            Arc::clone(&self.values),
        ))
    }

    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    pub fn request_load(&mut self) {
        let Some(path) = self.parquet_path.clone() else {
            return;
        };
        let Some(marker) = self.marker_choices.get(self.selected_marker).cloned() else {
            return;
        };
        self.status = "Loading...".to_string();
        self.load_request_id = self.load_request_id.wrapping_add(1);
        let key = LoadKey {
            roi_label: self.roi_label.clone(),
            marker_column: marker.column,
            coord_downsample_bits: self.coord_downsample.to_bits(),
        };
        let req = LoadRequest {
            request_id: self.load_request_id,
            parquet_path: path,
            key,
        };
        let _ = self.tx.send(req);
    }

    fn parquet_path_for_source(&self, dataset_root: &Path) -> Option<PathBuf> {
        let dir = match self.cells_source {
            CellsSource::Standard => self.parquet_dir_standard.as_ref(),
            CellsSource::Flatfield => self.parquet_dir_flatfield.as_ref(),
        }?;
        parquet_path_for_zarr_root(dir, dataset_root)
    }

    fn drain_loader(&mut self, points_layer: &mut PointsLayer) {
        while let Ok(msg) = self.rx.try_recv() {
            if msg.request_id < self.last_loaded_request_id {
                continue;
            }
            self.last_loaded_request_id = msg.request_id;
            let key = msg.key.clone();
            self.positions_world = Arc::new(msg.positions);
            self.values = Arc::new(msg.values);
            self.values_generation = self.values_generation.wrapping_add(1);
            self.values_min = msg.min;
            self.values_max = msg.max.max(self.values_min + 1e-6);
            self.status = format!("Loaded {} points.", self.positions_world.len());
            self.total_count = self.values.len();

            points_layer.points = self
                .positions_world
                .iter()
                .copied()
                .map(|world_lvl0| Point {
                    world_lvl0,
                    positive: false,
                })
                .collect();
            points_layer.visible = self.points_visible;

            let key_changed = self.last_loaded_key.as_ref().is_none_or(|k| *k != key);
            self.last_loaded_key = Some(key);

            if key_changed {
                self.restore_persisted_state_for_current();
                if self.auto_method != AutoMethod::Manual {
                    self.apply_auto_threshold_if_available();
                }
            }
            let lo = self.values_min;
            let hi = self.values_max.max(lo + 1e-6);
            self.threshold = self.threshold.clamp(lo, hi);
            if !self.threshold.is_finite() {
                self.threshold = (lo + hi) * 0.5;
            }
            self.apply_threshold(points_layer);
        }
    }

    fn apply_threshold(&mut self, points_layer: &mut PointsLayer) {
        let t = self.threshold;
        let n = points_layer.points.len().min(self.values.len());
        let mut positive = 0usize;
        for i in 0..n {
            let is_pos = self.values[i] >= t;
            points_layer.points[i].positive = is_pos;
            positive += is_pos as usize;
        }
        self.positive_count = positive;
        self.total_count = n;
    }

    fn normalized_roi_label(&self) -> String {
        normalize_roi_label(&self.roi_label)
    }

    fn rebuild_marker_base_lookup(&mut self) {
        self.marker_base_lookup.clear();
        for (i, m) in self.marker_choices.iter().enumerate() {
            let base = base_marker_label(&m.display);
            let key = canonical_marker_token(&base);
            if key.is_empty() {
                continue;
            }
            self.marker_base_lookup.entry(key).or_insert(i);
        }
    }

    fn current_marker_display(&self) -> Option<String> {
        self.marker_choices
            .get(self.selected_marker)
            .map(|m| m.display.clone())
    }

    fn current_source_tag(&self) -> String {
        if self.cells_source_available.len() >= 2 {
            self.cells_source.as_str().to_string()
        } else {
            String::new()
        }
    }

    fn current_key(&self) -> Option<(String, String, String)> {
        let roi = self.normalized_roi_label();
        let marker = self.current_marker_display()?;
        let source = self.current_source_tag();
        Some((roi, marker, source))
    }

    fn current_auto_marker_key(&self) -> Option<String> {
        let display = self.current_marker_display()?;
        let base = base_marker_label(&display);
        let key = canonical_marker_token(&base);
        if key.is_empty() {
            return None;
        }
        Some(key)
    }

    fn auto_threshold_for_current(&self) -> Option<&AutoThreshold> {
        let roi = self.normalized_roi_label();
        let desired = self.current_source_tag();
        let mut candidates = Vec::new();
        if let Some(ch) = self.marker_choices.get(self.selected_marker) {
            if !ch.marker_key.is_empty() {
                candidates.push(ch.marker_key.clone());
            }
        }
        if let Some(k) = self.current_auto_marker_key() {
            if !k.is_empty() {
                candidates.push(k);
            }
        }
        candidates.dedup();
        for marker_key in candidates {
            let Some(rec) = self.auto_thresholds.get(&(roi.clone(), marker_key)) else {
                continue;
            };
            // Choose source with napari-like fallback order.
            if !desired.is_empty() {
                if let Some(v) = rec.sources.get(&desired) {
                    return Some(v);
                }
                if let Some(pref) = rec.preferred_source.as_deref() {
                    if let Some(v) = rec.sources.get(pref) {
                        return Some(v);
                    }
                }
            }
            if let Some(v) = rec.sources.get("standard") {
                return Some(v);
            }
            if let Some((_k, v)) = rec.sources.iter().next() {
                return Some(v);
            }
        }
        None
    }

    fn restore_persisted_state_for_current(&mut self) {
        let Some((roi, marker, source)) = self.current_key() else {
            return;
        };

        let mut keys = vec![(roi.clone(), marker.clone(), source.clone())];
        if !source.is_empty() {
            keys.push((roi.clone(), marker.clone(), String::new()));
            if source != "standard" {
                keys.push((roi.clone(), marker.clone(), "standard".to_string()));
            }
        }
        let mut row = None;
        for k in &keys {
            row = self.thresholds_loaded.get(k).cloned();
            if row.is_some() {
                break;
            }
        }

        if let Some(row) = row {
            self.threshold = row.arcsinh_threshold;
            let method = row.method.trim().to_ascii_lowercase();
            if method == "kmeans" {
                self.auto_method = AutoMethod::KMeans;
                if let Some(k) = row.kmeans_k {
                    self.auto_kmeans_k = k.max(2);
                }
                if let Some(ge) = row.positive_ge {
                    self.auto_positive_ge = ge.max(2);
                }
            } else if method == "otsu" {
                self.auto_method = AutoMethod::Otsu;
            } else {
                self.auto_method = AutoMethod::Manual;
            }
            return;
        }

        if self.auto_threshold_for_current().is_some() {
            self.auto_method = AutoMethod::KMeans;
            self.apply_auto_threshold_if_available();
        } else {
            self.auto_method = AutoMethod::Manual;
        }
    }

    fn apply_auto_threshold_if_available(&mut self) {
        let Some(auto) = self.auto_threshold_for_current().cloned() else {
            return;
        };

        let thr_arcsinh = match self.auto_method {
            AutoMethod::Manual => return,
            AutoMethod::Otsu => auto.otsu_arcsinh,
            AutoMethod::KMeans => {
                let k = auto.kmeans_k.max(2) as i32;
                let ge = self.auto_positive_ge.clamp(2, auto.kmeans_k.max(2));
                let idx = (ge as i32 - 2).clamp(0, k.saturating_sub(2)) as usize;
                auto.kmeans_cutoffs_arcsinh.get(idx).copied()
            }
        };
        let Some(thr_arcsinh) = thr_arcsinh else {
            return;
        };

        self.threshold = thr_arcsinh;
    }

    pub fn sync_marker_from_channel_name(&mut self, channel_name: &str) -> bool {
        let base = base_marker_label(channel_name);
        let key = canonical_marker_token(&base);
        if key.is_empty() {
            return false;
        }
        if let Some(&idx) = self.marker_base_lookup.get(&key) {
            if idx != self.selected_marker {
                self.selected_marker = idx;
                self.restore_persisted_state_for_current();
                self.request_load();
                return true;
            }
        }
        false
    }

    fn reload_config(&mut self, dataset_root: &Path) {
        let dataset_key = project_dataset_key_for_root(&self.project, dataset_root);
        let Some(ds_cfg) = self.project.datasets.get(&dataset_key) else {
            self.enabled = false;
            self.status = format!(
                "No dataset config found for dataset '{dataset_key}'.\nConfigure it in the Project JSON under `datasets`."
            );
            return;
        };

        self.dataset_name = Some(dataset_key.clone());
        self.channels_index_path = ds_cfg
            .channels_index_path
            .as_deref()
            .map(expand_tilde)
            .map(PathBuf::from);

        let layout = ds_cfg
            .layout
            .as_deref()
            .unwrap_or("flat_roi")
            .trim()
            .to_ascii_lowercase();

        let (base_dir, uses_downsampled) = best_base_dir_for_root(ds_cfg, dataset_root);
        if let Some(base_dir) = base_dir.as_deref() {
            self.roi_label = infer_roi_label_with_layout(dataset_root, base_dir, &layout);
        } else {
            self.roi_label = infer_roi_label(dataset_root, None);
        }

        self.thresholds_csv_path = ds_cfg
            .thresholds_csv
            .as_deref()
            .map(expand_tilde)
            .map(PathBuf::from);
        self.thresholds_loaded.clear();

        if let Some(p) = self.thresholds_csv_path.as_ref() {
            match load_thresholds_csv(p) {
                Ok(rows) => {
                    self.thresholds_loaded = rows;
                }
                Err(err) => {
                    // Keep the UI usable even if the CSV can't be read yet.
                    self.status = format!("Thresholds CSV load failed: {err}");
                }
            }
        }

        self.auto_thresholds_path = ds_cfg
            .auto_thresholds_json
            .as_deref()
            .map(expand_tilde)
            .map(PathBuf::from);
        self.auto_thresholds.clear();
        if let Some(p) = self.auto_thresholds_path.as_ref() {
            match load_auto_thresholds_json(p) {
                Ok((k, marker_stat, map)) => {
                    self.auto_kmeans_k = k.max(2);
                    if let Some(stat) = marker_stat {
                        if !stat.trim().is_empty() {
                            self.marker_stat = stat.trim().to_ascii_lowercase();
                        }
                    }
                    self.auto_thresholds = map;
                }
                Err(err) => {
                    self.status = format!("Auto-thresholds JSON load failed: {err}");
                }
            }
        }

        let backend = ds_cfg
            .cells_backend
            .as_deref()
            .unwrap_or("single_parquet")
            .trim()
            .to_ascii_lowercase();
        self.parquet_dir_standard = None;
        self.parquet_dir_flatfield = None;
        self.cells_source_available.clear();
        let parquet_path: Option<PathBuf> = match backend.as_str() {
            "single_parquet" => ds_cfg
                .cells_parquet
                .as_deref()
                .map(expand_tilde)
                .map(PathBuf::from),
            "per_roi_parquet" => ds_cfg
                .cells_parquet_dir
                .as_deref()
                .map(expand_tilde)
                .map(PathBuf::from)
                .and_then(|dir| parquet_path_for_zarr_root(&dir, dataset_root)),
            "multi_source_parquet" => {
                self.parquet_dir_standard = ds_cfg
                    .cells_parquet_dir
                    .as_deref()
                    .map(expand_tilde)
                    .map(PathBuf::from);
                self.parquet_dir_flatfield = ds_cfg
                    .cells_parquet_dir_flatfield
                    .as_deref()
                    .map(expand_tilde)
                    .map(PathBuf::from);

                if self
                    .parquet_dir_standard
                    .as_ref()
                    .is_some_and(|p| p.is_dir())
                {
                    self.cells_source_available.push(CellsSource::Standard);
                }
                if self
                    .parquet_dir_flatfield
                    .as_ref()
                    .is_some_and(|p| p.is_dir())
                {
                    self.cells_source_available.push(CellsSource::Flatfield);
                }
                if self.cells_source_available.is_empty() {
                    None
                } else {
                    if !self.cells_source_available.contains(&self.cells_source) {
                        self.cells_source = self
                            .cells_source_available
                            .first()
                            .copied()
                            .unwrap_or(CellsSource::Standard);
                    }
                    self.parquet_path_for_source(dataset_root)
                }
            }
            other => {
                self.enabled = false;
                self.status =
                    format!("Unsupported cells_backend '{other}' for dataset {dataset_key}");
                return;
            }
        };
        let Some(parquet_path) = parquet_path else {
            self.enabled = false;
            self.status = format!(
                "No cells parquet configured for dataset {dataset_key} (backend={backend})"
            );
            return;
        };
        if !parquet_path.exists() {
            self.enabled = false;
            self.status = format!(
                "Cells parquet not found: {}",
                parquet_path.to_string_lossy()
            );
            return;
        }
        self.parquet_path = Some(parquet_path.clone());

        self.coord_downsample = if uses_downsampled {
            ds_cfg.coord_downsample_downsampled.unwrap_or(1.0)
        } else {
            ds_cfg.coord_downsample_full_res.unwrap_or(1.0)
        }
        .max(1e-6);

        let channel_labels = read_channel_labels(dataset_root, self.channels_index_path.as_deref());

        match list_marker_choices(&parquet_path, &channel_labels, &self.marker_stat) {
            Ok(mut choices) => {
                if let Some(order) = ds_cfg.subset_channel_labels.as_ref() {
                    // Put configured channel labels first if we can find a fuzzy match.
                    choices.sort_by_key(|m| {
                        let idx = order
                            .iter()
                            .position(|lbl| loosely_matches(lbl, &m.display));
                        idx.unwrap_or(usize::MAX)
                    });
                }
                self.marker_choices = choices;
                self.rebuild_marker_base_lookup();
            }
            Err(err) => {
                self.enabled = false;
                self.status = format!("Failed to read parquet schema: {err}");
                return;
            }
        }

        if self.marker_choices.is_empty() {
            self.enabled = false;
            self.status = "No marker intensity columns found in parquet".to_string();
            return;
        }

        if let Some(default_marker) = self.project.default_threshold_marker.as_deref() {
            if let Some(i) = self
                .marker_choices
                .iter()
                .position(|m| loosely_matches(default_marker, &m.display))
            {
                self.selected_marker = i;
            }
        }

        self.enabled = true;
        self.status = "Ready.".to_string();
        self.restore_persisted_state_for_current();
        self.request_load();
    }
}
