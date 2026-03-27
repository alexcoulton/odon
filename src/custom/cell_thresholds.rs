use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, anyhow};
use arrow_array::{Array, RecordBatch, RecordBatchReader};
use eframe::egui;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::render::points::{Point, PointsLayer, PointsStyle};
use crate::data::project_config::{ProjectConfig, ProjectDatasetConfig};

#[derive(Debug, Clone)]
pub enum CellThresholdsAction {
    CycleImageChannel(i32),
    NudgeImageContrastMax(f32),
}

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
    scale_mode: ScaleMode,
    threshold: f32,
    threshold_touched: bool,
    values_min: f32,
    values_max: f32,
    positions_world: Arc<Vec<egui::Pos2>>,
    values: Arc<Vec<f32>>,
    values_generation: u64,
    last_loaded_key: Option<LoadKey>,
    positive_count: usize,
    total_count: usize,
    hist_values_generation: u64,
    hist: Option<ValueHistogram>,

    points_visible: bool,
    style: PointsStyle,

    thresholds_csv_path: Option<PathBuf>,
    thresholds_loaded: HashMap<(String, String, String), ThresholdCsvRow>,
    threshold_meta_loaded: HashMap<(String, String, String), ThresholdMeta>,
    threshold_dirty: HashMap<(String, String, String), ThresholdCsvRow>,
    notes: String,
    working: bool,
    autosave_csv: bool,
    autosave_pending: bool,
    autosave_last_edit: Instant,

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

    write_request_id: u64,
    tx_write: crossbeam_channel::Sender<WriteRequest>,
    rx_write: crossbeam_channel::Receiver<WriteResponse>,
    last_write_response_id: u64,
}

#[derive(Debug, Clone)]
struct MarkerChoice {
    display: String,
    column: String,
    marker_key: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScaleMode {
    Raw,
    Arcsinh,
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
struct ThresholdMeta {
    method: AutoMethod,
    kmeans_k: Option<u8>,
    positive_ge: Option<u8>,
}

#[derive(Debug, Clone)]
struct ThresholdCsvRow {
    roi: String,
    marker: String,
    source: String,
    raw_threshold: f32,
    arcsinh_threshold: f32,
    method: String,
    kmeans_k: Option<u8>,
    positive_ge: Option<u8>,
    notes: String,
    working: bool,
}

#[derive(Debug, Clone)]
struct ValueHistogram {
    min: f32,
    max: f32,
    bins: Vec<u32>,
    max_count: u32,
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
    scale_mode: ScaleMode,
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

#[derive(Debug, Clone)]
struct WriteRequest {
    request_id: u64,
    csv_path: PathBuf,
    rows: Vec<ThresholdCsvRow>,
}

#[derive(Debug, Clone)]
struct WriteResponse {
    request_id: u64,
    ok: bool,
    status: String,
}

impl CellThresholdsPanel {
    pub fn set_project_config(&mut self, project: ProjectConfig) {
        self.project = project;
        let root = self.dataset_root.clone();
        self.reload_config(&root);
    }

    fn recompute_histogram(&mut self) {
        let values = self.values.as_ref();
        if values.is_empty() {
            self.hist = None;
            self.hist_values_generation = self.values_generation;
            return;
        }
        let (min, max) = finite_min_max(values).unwrap_or((0.0, 1.0));
        let min = min.min(max);
        let max = max.max(min + 1e-6);

        const BINS: usize = 256;
        let mut bins = vec![0u32; BINS];
        let inv = (BINS as f32) / (max - min);

        // Cap work for very large point clouds.
        let max_samples = 2_000_000usize;
        let step = (values.len() / max_samples).max(1);
        for &v in values.iter().step_by(step) {
            if !v.is_finite() {
                continue;
            }
            let mut idx = ((v - min) * inv).floor() as i32;
            if idx < 0 {
                idx = 0;
            }
            if idx as usize >= BINS {
                idx = (BINS as i32) - 1;
            }
            bins[idx as usize] = bins[idx as usize].saturating_add(1);
        }
        let max_count = bins.iter().copied().max().unwrap_or(1).max(1);

        self.hist = Some(ValueHistogram {
            min,
            max,
            bins,
            max_count,
        });
        self.hist_values_generation = self.values_generation;
    }

    fn ui_histogram(&mut self, ui: &mut egui::Ui) -> bool {
        let Some(hist) = self.hist.as_ref() else {
            return false;
        };
        let height = 110.0;
        let (rect, resp) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), height),
            egui::Sense::click_and_drag(),
        );
        let painter = ui.painter();

        painter.rect_filled(
            rect,
            egui::CornerRadius::same(4),
            ui.visuals().extreme_bg_color,
        );
        painter.rect_stroke(
            rect,
            egui::CornerRadius::same(4),
            ui.visuals().widgets.noninteractive.bg_stroke,
            egui::StrokeKind::Middle,
        );

        let bins = &hist.bins;
        if bins.is_empty() {
            return false;
        }

        let bar_color = ui.visuals().widgets.noninteractive.fg_stroke.color;
        let stroke = egui::Stroke::new(1.0, bar_color.linear_multiply(0.75));

        let w = rect.width().max(1.0);
        let h = rect.height().max(1.0);
        let n = bins.len().max(1) as f32;
        for (i, &c) in bins.iter().enumerate() {
            if c == 0 {
                continue;
            }
            let x0 = rect.left() + (i as f32 + 0.5) * (w / n);
            let frac = (c as f32) / (hist.max_count as f32);
            let bh = (frac * (h - 6.0)).clamp(0.0, h);
            let y0 = rect.bottom() - 3.0;
            let y1 = (y0 - bh).max(rect.top() + 3.0);
            painter.line_segment([egui::pos2(x0, y0), egui::pos2(x0, y1)], stroke);
        }

        // Threshold line.
        let t = self.threshold.clamp(hist.min, hist.max);
        let tx = if (hist.max - hist.min) > 0.0 {
            rect.left() + ((t - hist.min) / (hist.max - hist.min)) * w
        } else {
            rect.left()
        };
        let sel = ui.visuals().selection.stroke;
        painter.line_segment(
            [egui::pos2(tx, rect.top()), egui::pos2(tx, rect.bottom())],
            egui::Stroke::new(sel.width.max(2.0), sel.color),
        );

        // Hover tooltip with value.
        if resp.hovered() {
            if let Some(pos) = resp.hover_pos() {
                let frac = ((pos.x - rect.left()) / w).clamp(0.0, 1.0);
                let v = hist.min + frac * (hist.max - hist.min);
                let layer_id =
                    egui::LayerId::new(egui::Order::Tooltip, ui.id().with("ct_hist_tip_layer"));
                egui::show_tooltip_at_pointer(
                    ui.ctx(),
                    layer_id,
                    ui.id().with("ct_hist_tip"),
                    |ui: &mut egui::Ui| {
                        ui.label(format!("Value: {:.4}", v));
                        ui.label(format!("Threshold: {:.4}", self.threshold));
                    },
                );
            }
        }

        // Click/drag to set threshold.
        let mut changed = false;
        if resp.clicked() || resp.dragged() {
            if let Some(pos) = resp.interact_pointer_pos() {
                let frac = ((pos.x - rect.left()) / w).clamp(0.0, 1.0);
                let v = hist.min + frac * (hist.max - hist.min);
                if v.is_finite() {
                    self.threshold = v.clamp(hist.min, hist.max);
                    changed = true;
                }
            }
        }

        changed
    }

    pub fn new(dataset_root: &Path, ome_multiscale_name: Option<&str>) -> Self {
        let roi_label = infer_roi_label(dataset_root, ome_multiscale_name);

        let (tx, rx) = spawn_loader_thread();
        let (tx_write, rx_write) = spawn_writer_thread();

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
            scale_mode: ScaleMode::Arcsinh,
            threshold: 0.0,
            threshold_touched: false,
            values_min: 0.0,
            values_max: 1.0,
            positions_world: Arc::new(Vec::new()),
            values: Arc::new(Vec::new()),
            values_generation: 0,
            last_loaded_key: None,
            positive_count: 0,
            total_count: 0,
            hist_values_generation: 0,
            hist: None,
            points_visible: true,
            style: PointsStyle::default(),
            thresholds_csv_path: None,
            thresholds_loaded: HashMap::new(),
            threshold_meta_loaded: HashMap::new(),
            threshold_dirty: HashMap::new(),
            notes: String::new(),
            working: true,
            autosave_csv: false,
            autosave_pending: false,
            autosave_last_edit: Instant::now()
                .checked_sub(Duration::from_secs(3600))
                .unwrap_or_else(Instant::now),
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
            write_request_id: 0,
            tx_write,
            rx_write,
            last_write_response_id: 0,
        };

        panel.reload_config(dataset_root);
        panel
    }

    pub fn set_dataset_root(
        &mut self,
        dataset_root: &Path,
        ome_multiscale_name: Option<&str>,
        points_layer: &mut PointsLayer,
    ) {
        self.dataset_root = dataset_root.to_path_buf();
        self.roi_label = infer_roi_label(dataset_root, ome_multiscale_name);

        self.status = "Loading...".to_string();
        self.last_loaded_key = None;
        self.values_generation = self.values_generation.wrapping_add(1);
        self.positions_world = Arc::new(Vec::new());
        self.values = Arc::new(Vec::new());
        self.hist = None;
        self.values_min = 0.0;
        self.values_max = 1.0;
        self.positive_count = 0;
        self.total_count = 0;
        points_layer.points.clear();
        points_layer.visible = false;
        self.points_visible = points_layer.visible;

        self.reload_config(dataset_root);
    }

    pub fn tick(&mut self, points_layer: &mut PointsLayer) {
        // Keep panel state in sync with the global layer visibility toggle.
        self.points_visible = points_layer.visible;
        self.drain_loader(points_layer);
        self.drain_writer();
        self.maybe_flush_autosave();
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

    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        points_layer: &mut PointsLayer,
    ) -> Option<CellThresholdsAction> {
        ui.heading("Cell Thresholds");

        if !self.enabled {
            ui.label(&self.status);
            points_layer.visible = false;
            self.points_visible = points_layer.visible;
            ui.add_space(8.0);
            if ui.button("Reload config").clicked() {
                let root = self.dataset_root.clone();
                self.reload_config(&root);
            }
            return None;
        }

        self.drain_loader(points_layer);
        self.drain_writer();
        self.maybe_flush_autosave();

        let mut action: Option<CellThresholdsAction> = None;

        ui.horizontal(|ui| {
            // `points_layer.visible` is the source of truth (toggled from the left panel "Layers").
            self.points_visible = points_layer.visible;
            let mut v = self.points_visible;
            if ui.checkbox(&mut v, "Show points").changed() {
                points_layer.visible = v;
                self.points_visible = v;
            }
        });

        ui.horizontal(|ui| {
            if ui.button("Previous Channel").clicked() {
                action = Some(CellThresholdsAction::CycleImageChannel(-1));
            }
            if ui.button("Next Channel").clicked() {
                action = Some(CellThresholdsAction::CycleImageChannel(1));
            }
        });
        ui.horizontal(|ui| {
            for (delta, label) in [
                (-1000.0f32, "Max -1000"),
                (-100.0f32, "Max -100"),
                (100.0f32, "Max +100"),
                (1000.0f32, "Max +1000"),
            ] {
                if ui.button(label).clicked() {
                    action = Some(CellThresholdsAction::NudgeImageContrastMax(delta));
                }
            }
        });

        let mut roi_changed = false;
        ui.horizontal(|ui| {
            ui.label("ROI");
            let edit = ui.text_edit_singleline(&mut self.roi_label);
            if edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                roi_changed = true;
            }
            if ui.button("Reload").clicked() {
                roi_changed = true;
            }
        });

        ui.separator();
        ui.label(format!(
            "Dataset: {}",
            self.dataset_name.as_deref().unwrap_or("-")
        ));
        ui.label(format!(
            "Parquet: {}",
            self.parquet_path
                .as_ref()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|| "-".to_string())
        ));
        ui.label(format!(
            "Project: {} ROIs, {} datasets",
            self.project.rois.len(),
            self.project.datasets.len()
        ));
        ui.label(format!("Status: {}", self.status));

        ui.separator();
        let old_scale = self.scale_mode;
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.scale_mode, ScaleMode::Raw, "raw");
            ui.selectable_value(&mut self.scale_mode, ScaleMode::Arcsinh, "arcsinh");
        });
        let scale_changed = old_scale != self.scale_mode;

        let mut source_changed = false;
        if self.cells_source_available.len() >= 2 {
            let before = self.cells_source;
            ui.horizontal(|ui| {
                ui.label("Source");
                egui::ComboBox::from_id_salt("cells-source")
                    .selected_text(self.cells_source.as_str())
                    .show_ui(ui, |ui| {
                        for src in self.cells_source_available.clone() {
                            ui.selectable_value(&mut self.cells_source, src, src.as_str());
                        }
                    });
            });
            source_changed = before != self.cells_source;
            if source_changed {
                let prev_display = self
                    .marker_choices
                    .get(self.selected_marker)
                    .map(|m| m.display.clone());
                self.parquet_path = self.parquet_path_for_source(&self.dataset_root);
                if let Some(parquet_path) = self.parquet_path.clone() {
                    let channel_labels = read_channel_labels(
                        &self.dataset_root,
                        self.channels_index_path.as_deref(),
                    );
                    if let Ok(choices) =
                        list_marker_choices(&parquet_path, &channel_labels, &self.marker_stat)
                    {
                        self.marker_choices = choices;
                        self.rebuild_marker_base_lookup();
                        if let Some(want) = prev_display {
                            if let Some(i) =
                                self.marker_choices.iter().position(|m| m.display == want)
                            {
                                self.selected_marker = i;
                            } else {
                                self.selected_marker = 0;
                            }
                        } else {
                            self.selected_marker = 0;
                        }
                    }
                }
            }
        }

        let mut marker_changed = false;
        egui::ComboBox::from_label("Marker")
            .selected_text(
                self.marker_choices
                    .get(self.selected_marker)
                    .map(|m| m.display.as_str())
                    .unwrap_or("-"),
            )
            .show_ui(ui, |ui| {
                for (i, m) in self.marker_choices.iter().enumerate() {
                    marker_changed |= ui
                        .selectable_value(&mut self.selected_marker, i, &m.display)
                        .changed();
                }
            });

        let mut method_changed = false;
        let mut kcutoff_changed = false;
        let has_auto = self.auto_threshold_for_current().is_some();
        ui.horizontal(|ui| {
            ui.label("Method");
            let selected_text = match self.auto_method {
                AutoMethod::Manual => "Manual".to_string(),
                AutoMethod::KMeans => format!("KMeans (k={})", self.auto_kmeans_k.max(2)),
                AutoMethod::Otsu => "Otsu".to_string(),
            };
            egui::ComboBox::from_id_salt("auto-method")
                .selected_text(selected_text)
                .show_ui(ui, |ui| {
                    method_changed |= ui
                        .selectable_value(&mut self.auto_method, AutoMethod::Manual, "Manual")
                        .changed();
                    ui.add_enabled_ui(has_auto, |ui| {
                        method_changed |= ui
                            .selectable_value(
                                &mut self.auto_method,
                                AutoMethod::KMeans,
                                format!("KMeans (k={})", self.auto_kmeans_k.max(2)),
                            )
                            .changed();
                        method_changed |= ui
                            .selectable_value(&mut self.auto_method, AutoMethod::Otsu, "Otsu")
                            .changed();
                    });
                });
        });

        ui.horizontal(|ui| {
            let max_ge = self
                .auto_threshold_for_current()
                .map(|a| a.kmeans_k)
                .unwrap_or(self.auto_kmeans_k)
                .max(2);
            let mut ge = self.auto_positive_ge.clamp(2, max_ge);
            ui.add_enabled_ui(self.auto_method == AutoMethod::KMeans && has_auto, |ui| {
                egui::ComboBox::from_label("Positive >=")
                    .selected_text(format!("{ge}"))
                    .show_ui(ui, |ui| {
                        for v in 2..=max_ge {
                            kcutoff_changed |=
                                ui.selectable_value(&mut ge, v, format!("{v}")).changed();
                        }
                    });
            });
            if ui.add_enabled(has_auto, egui::Button::new("K -")).clicked() {
                ge = ge.saturating_sub(1).max(2);
                kcutoff_changed = true;
                self.auto_method = AutoMethod::KMeans;
            }
            if ui.add_enabled(has_auto, egui::Button::new("K +")).clicked() {
                ge = (ge + 1).min(max_ge);
                kcutoff_changed = true;
                self.auto_method = AutoMethod::KMeans;
            }
            self.auto_positive_ge = ge;
        });

        let mut write_clicked = false;
        let mut autosave_toggled = false;
        ui.horizontal(|ui| {
            write_clicked |= ui
                .add_enabled(
                    self.thresholds_csv_path.is_some(),
                    egui::Button::new("Write CSV"),
                )
                .clicked();
            autosave_toggled |= ui
                .add_enabled(
                    self.thresholds_csv_path.is_some(),
                    egui::Checkbox::new(&mut self.autosave_csv, "Auto-save CSV"),
                )
                .changed();
        });

        ui.label(self.threshold_loaded_label_text());

        let abs_min = self.values_min;
        let abs_max = self.values_max.max(abs_min + 1e-6);
        if self.hist_values_generation != self.values_generation {
            self.recompute_histogram();
        }
        let mut threshold_changed = self.ui_histogram(ui);
        threshold_changed |= ui
            .add(egui::Slider::new(&mut self.threshold, abs_min..=abs_max).text("Threshold"))
            .changed();
        ui.horizontal(|ui| {
            threshold_changed |= ui
                .add(
                    egui::DragValue::new(&mut self.threshold)
                        .speed(((abs_max - abs_min) / 500.0).max(0.1)),
                )
                .changed();
            for (delta, label) in [
                (-1000.0, "-1000"),
                (-100.0, "-100"),
                (100.0, "+100"),
                (1000.0, "+1000"),
            ] {
                if ui.button(label).clicked() {
                    self.threshold = (self.threshold + delta).clamp(abs_min, abs_max);
                    threshold_changed = true;
                }
            }
        });
        if self.scale_mode == ScaleMode::Arcsinh {
            ui.label(format!(
                "Raw ~ {:.3}",
                (self.threshold as f64).sinh() as f32
            ));
        }
        ui.label(format!(
            "{} / {} cells >= {:.3}",
            self.positive_count, self.total_count, self.threshold
        ));

        ui.separator();
        let mut not_working = !self.working;
        let working_changed = ui
            .add_enabled(
                self.thresholds_csv_path.is_some(),
                egui::Checkbox::new(&mut not_working, "Not working"),
            )
            .changed();
        if working_changed {
            self.working = !not_working;
        }
        ui.label("Notes");
        let notes_changed = ui
            .add_enabled(
                self.thresholds_csv_path.is_some(),
                egui::TextEdit::multiline(&mut self.notes).desired_rows(3),
            )
            .changed();
        let save_note_clicked = ui
            .add_enabled(
                self.thresholds_csv_path.is_some(),
                egui::Button::new("Save note"),
            )
            .clicked();

        ui.separator();
        ui.label("Point style");
        ui.add(egui::Slider::new(&mut self.style.radius_screen_px, 1.0..=20.0).text("Size"));
        ui.horizontal(|ui| {
            ui.label("Pos");
            ui.color_edit_button_srgba(&mut self.style.fill_positive);
            ui.label("Neg");
            ui.color_edit_button_srgba(&mut self.style.fill_negative);
        });
        ui.horizontal(|ui| {
            ui.label("Stroke");
            let mut stroke_color = self.style.stroke_positive.color;
            ui.color_edit_button_srgba(&mut stroke_color);
            self.style.stroke_positive.color = stroke_color;
            ui.add(egui::Slider::new(&mut self.style.stroke_positive.width, 0.0..=4.0).text("W"));
        });
        ui.horizontal(|ui| {
            ui.label("Neg stroke");
            let mut stroke_color = self.style.stroke_negative.color;
            ui.color_edit_button_srgba(&mut stroke_color);
            self.style.stroke_negative.color = stroke_color;
            ui.add(egui::Slider::new(&mut self.style.stroke_negative.width, 0.0..=4.0).text("W"));
        });
        points_layer.style = self.style.clone();

        if roi_changed || source_changed {
            // Avoid showing stale ROI points while a new ROI/source load is in flight.
            points_layer.points.clear();
            self.positions_world = Arc::new(Vec::new());
            self.values = Arc::new(Vec::new());
            self.values_generation = self.values_generation.wrapping_add(1);
            self.hist = None;
            self.positive_count = 0;
            self.total_count = 0;
        }

        if marker_changed || roi_changed || scale_changed || source_changed {
            self.restore_persisted_state_for_current();
            self.request_load();
        }

        if autosave_toggled {
            // Toggle doesn't mark the data dirty, but if we have pending changes and the user
            // just enabled autosave, schedule a flush.
            self.autosave_pending = self.autosave_csv && !self.threshold_dirty.is_empty();
            self.autosave_last_edit = Instant::now();
        }

        if method_changed || kcutoff_changed {
            if self.auto_method != AutoMethod::Manual {
                self.apply_auto_threshold_if_available();
                self.threshold_touched = false;
                self.apply_threshold(points_layer);
            }
            self.mark_dirty_current();
            self.schedule_autosave();
        }

        // Apply threshold to the already-loaded points (fast).
        if threshold_changed {
            if self.auto_method != AutoMethod::Manual {
                self.auto_method = AutoMethod::Manual;
            }
            self.threshold_touched = true;
            self.apply_threshold(points_layer);
            self.mark_dirty_current();
            self.schedule_autosave();
        }

        if working_changed || notes_changed {
            self.mark_dirty_current();
            self.schedule_autosave();
        }

        if write_clicked || save_note_clicked {
            let _ = self.enqueue_write_current();
        }

        action
    }

    pub fn request_load(&mut self) {
        let Some(path) = self.parquet_path.clone() else {
            return;
        };
        let Some(marker) = self.marker_choices.get(self.selected_marker).cloned() else {
            return;
        };
        self.status = "Loading...".to_string();
        self.threshold_touched = false;
        self.load_request_id = self.load_request_id.wrapping_add(1);
        let key = LoadKey {
            roi_label: self.roi_label.clone(),
            marker_column: marker.column,
            scale_mode: self.scale_mode,
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
            self.recompute_histogram();

            points_layer.points = self
                .positions_world
                .iter()
                .copied()
                .map(|world_lvl0| Point {
                    world_lvl0,
                    positive: false,
                })
                .collect();
            points_layer.style = self.style.clone();
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

    fn threshold_loaded_label_text(&self) -> String {
        let Some((roi, marker, source)) = self.current_key() else {
            return "Threshold: -".to_string();
        };
        let mut loaded =
            self.thresholds_loaded
                .contains_key(&(roi.clone(), marker.clone(), source.clone()));
        let mut modified =
            self.threshold_dirty
                .contains_key(&(roi.clone(), marker.clone(), source.clone()));
        if !loaded && !source.is_empty() {
            loaded =
                self.thresholds_loaded
                    .contains_key(&(roi.clone(), marker.clone(), String::new()));
        }
        if !modified && !source.is_empty() {
            modified =
                self.threshold_dirty
                    .contains_key(&(roi.clone(), marker.clone(), String::new()));
        }

        let method_text = match self.auto_method {
            AutoMethod::Manual => "manual".to_string(),
            AutoMethod::Otsu => "otsu".to_string(),
            AutoMethod::KMeans => format!(
                "kmeans(k={}) >={}",
                self.auto_kmeans_k.max(2),
                self.auto_positive_ge.max(2)
            ),
        };
        let state = if loaded { "loaded" } else { "not loaded" };
        let suffix = if modified { " (modified)" } else { "" };
        format!("Threshold: {state} | method={method_text}{suffix}")
    }

    fn restore_persisted_state_for_current(&mut self) {
        let Some((roi, marker, source)) = self.current_key() else {
            return;
        };

        // Prefer dirty (in-memory edits), then loaded CSV.
        let mut keys = vec![(roi.clone(), marker.clone(), source.clone())];
        if !source.is_empty() {
            keys.push((roi.clone(), marker.clone(), String::new()));
            if source != "standard" {
                keys.push((roi.clone(), marker.clone(), "standard".to_string()));
            }
        }
        let mut row = None;
        for k in &keys {
            row = self.threshold_dirty.get(k).cloned();
            if row.is_some() {
                break;
            }
        }
        if row.is_none() {
            for k in &keys {
                row = self.thresholds_loaded.get(k).cloned();
                if row.is_some() {
                    break;
                }
            }
        }

        if let Some(row) = row {
            self.notes = row.notes.clone();
            self.working = row.working;
            self.threshold = match self.scale_mode {
                ScaleMode::Raw => row.raw_threshold,
                ScaleMode::Arcsinh => row.arcsinh_threshold,
            };
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
            self.threshold_touched = false;
            return;
        }

        // No CSV state; default to auto mode if available.
        self.notes.clear();
        self.working = true;
        if self.auto_threshold_for_current().is_some() {
            self.auto_method = AutoMethod::KMeans;
            self.apply_auto_threshold_if_available();
            self.threshold_touched = false;
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

        self.threshold = match self.scale_mode {
            ScaleMode::Arcsinh => thr_arcsinh,
            ScaleMode::Raw => (thr_arcsinh as f64).sinh() as f32,
        };
    }

    fn mark_dirty_current(&mut self) {
        let Some((roi, marker, source)) = self.current_key() else {
            return;
        };
        let (raw, arcsinh) = match self.scale_mode {
            ScaleMode::Raw => {
                let raw = self.threshold;
                let arcsinh = (raw as f64).asinh() as f32;
                (raw, arcsinh)
            }
            ScaleMode::Arcsinh => {
                let arcsinh = self.threshold;
                let raw = (arcsinh as f64).sinh() as f32;
                (raw, arcsinh)
            }
        };

        let (method, kmeans_k, positive_ge) = match self.auto_method {
            AutoMethod::Manual => ("manual".to_string(), None, None),
            AutoMethod::Otsu => ("otsu".to_string(), None, None),
            AutoMethod::KMeans => (
                "kmeans".to_string(),
                Some(self.auto_kmeans_k.max(2)),
                Some(self.auto_positive_ge.max(2)),
            ),
        };

        self.threshold_dirty.insert(
            (roi.clone(), marker.clone(), source.clone()),
            ThresholdCsvRow {
                roi,
                marker,
                source,
                raw_threshold: raw,
                arcsinh_threshold: arcsinh,
                method,
                kmeans_k,
                positive_ge,
                notes: self.notes.clone(),
                working: self.working,
            },
        );
    }

    fn schedule_autosave(&mut self) {
        if !self.autosave_csv {
            return;
        }
        if self.thresholds_csv_path.is_none() {
            return;
        }
        self.autosave_pending = true;
        self.autosave_last_edit = Instant::now();
    }

    fn maybe_flush_autosave(&mut self) {
        if !self.autosave_pending || !self.autosave_csv {
            return;
        }
        if self.thresholds_csv_path.is_none() {
            self.autosave_pending = false;
            return;
        }
        if self.autosave_last_edit.elapsed() < Duration::from_millis(650) {
            return;
        }
        let _ = self.enqueue_write_all_dirty();
        self.autosave_pending = false;
    }

    fn enqueue_write_current(&mut self) -> anyhow::Result<()> {
        self.mark_dirty_current();
        self.enqueue_write_all_dirty()
    }

    fn enqueue_write_all_dirty(&mut self) -> anyhow::Result<()> {
        let Some(path) = self.thresholds_csv_path.clone() else {
            anyhow::bail!("no thresholds_csv configured");
        };
        if self.threshold_dirty.is_empty() {
            return Ok(());
        }

        self.write_request_id = self.write_request_id.wrapping_add(1);
        let rows = self.threshold_dirty.values().cloned().collect::<Vec<_>>();
        let req = WriteRequest {
            request_id: self.write_request_id,
            csv_path: path,
            rows,
        };
        let _ = self.tx_write.send(req);
        Ok(())
    }

    fn drain_writer(&mut self) {
        while let Ok(msg) = self.rx_write.try_recv() {
            if msg.request_id < self.last_write_response_id {
                continue;
            }
            self.last_write_response_id = msg.request_id;
            self.status = msg.status;
            if msg.ok {
                for (k, v) in self.threshold_dirty.drain() {
                    self.thresholds_loaded.insert(k, v);
                }
            }
        }
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
        self.threshold_meta_loaded.clear();
        self.threshold_dirty.clear();
        self.autosave_pending = false;
        self.notes.clear();
        self.working = true;

        if let Some(p) = self.thresholds_csv_path.as_ref() {
            match load_thresholds_csv(p) {
                Ok((rows, meta)) => {
                    self.thresholds_loaded = rows;
                    self.threshold_meta_loaded = meta;
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

fn project_dataset_key_for_root(project: &ProjectConfig, dataset_root: &Path) -> String {
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

fn best_base_dir_for_root(
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

fn parquet_path_for_zarr_root(parquet_dir: &Path, dataset_root: &Path) -> Option<PathBuf> {
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

fn spawn_loader_thread() -> (
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

fn spawn_writer_thread() -> (
    crossbeam_channel::Sender<WriteRequest>,
    crossbeam_channel::Receiver<WriteResponse>,
) {
    let (tx_req, rx_req) = crossbeam_channel::unbounded::<WriteRequest>();
    let (tx_rsp, rx_rsp) = crossbeam_channel::unbounded::<WriteResponse>();

    std::thread::Builder::new()
        .name("thresholds-csv-writer".to_string())
        .spawn(move || {
            if let Err(err) = writer_thread(rx_req, tx_rsp) {
                eprintln!("thresholds csv writer thread exited: {err:?}");
            }
        })
        .expect("failed to spawn thresholds csv writer thread");

    (tx_req, rx_rsp)
}

fn writer_thread(
    rx_req: crossbeam_channel::Receiver<WriteRequest>,
    tx_rsp: crossbeam_channel::Sender<WriteResponse>,
) -> anyhow::Result<()> {
    for req in rx_req.iter() {
        let result = write_thresholds_csv(&req.csv_path, &req.rows);
        let (ok, status) = match result {
            Ok(()) => (
                true,
                format!("Wrote thresholds: {}", req.csv_path.to_string_lossy()),
            ),
            Err(err) => (false, format!("Write CSV failed: {err}")),
        };
        let _ = tx_rsp.send(WriteResponse {
            request_id: req.request_id,
            ok,
            status,
        });
    }
    Ok(())
}

fn write_thresholds_csv(path: &Path, updates: &[ThresholdCsvRow]) -> anyhow::Result<()> {
    if updates.is_empty() {
        return Ok(());
    }

    let (mut headers, mut rows) = if path.exists() {
        let text = fs::read_to_string(path).with_context(|| {
            format!("failed to read thresholds csv: {}", path.to_string_lossy())
        })?;
        let mut recs = parse_csv(&text);
        if recs.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            let hdr = recs.remove(0);
            (hdr, recs)
        }
    } else {
        (Vec::new(), Vec::new())
    };

    if headers.is_empty() {
        headers = vec![
            "roi".to_string(),
            "marker".to_string(),
            "raw_threshold".to_string(),
            "arcsinh_threshold".to_string(),
            "method".to_string(),
            "kmeans_k".to_string(),
            "positive_ge".to_string(),
            "source".to_string(),
            "notes".to_string(),
            "working".to_string(),
        ];
    }

    let mut col_index: HashMap<String, usize> = HashMap::new();
    for (i, h) in headers.iter().enumerate() {
        col_index.insert(h.to_ascii_lowercase(), i);
    }

    let mut ensure_col = |name: &str| -> usize {
        let key = name.to_ascii_lowercase();
        if let Some(&i) = col_index.get(&key) {
            return i;
        }
        let i = headers.len();
        headers.push(name.to_string());
        col_index.insert(key, i);
        for r in rows.iter_mut() {
            while r.len() < headers.len() {
                r.push(String::new());
            }
        }
        i
    };

    let i_roi = ensure_col("roi");
    let i_marker = ensure_col("marker");
    let i_raw = ensure_col("raw_threshold");
    let i_arc = ensure_col("arcsinh_threshold");
    let i_method = ensure_col("method");
    let i_k = ensure_col("kmeans_k");
    let i_ge = ensure_col("positive_ge");
    let i_source = ensure_col("source");
    let i_notes = ensure_col("notes");
    let i_work = ensure_col("working");

    let mut row_lookup: HashMap<(String, String, String), usize> = HashMap::new();
    for (ri, r) in rows.iter().enumerate() {
        let roi = r.get(i_roi).cloned().unwrap_or_default();
        let marker = r.get(i_marker).cloned().unwrap_or_default();
        let source = r
            .get(i_source)
            .map(|s| s.trim().to_ascii_lowercase())
            .unwrap_or_default();
        let key = (normalize_roi_label(&roi), marker, source);
        row_lookup.entry(key).or_insert(ri);
    }

    for up in updates {
        let key = (
            normalize_roi_label(&up.roi),
            up.marker.clone(),
            up.source.trim().to_ascii_lowercase(),
        );
        let ri = if let Some(&idx) = row_lookup.get(&key) {
            idx
        } else {
            let idx = rows.len();
            rows.push(vec![String::new(); headers.len()]);
            row_lookup.insert(key.clone(), idx);
            idx
        };
        let r = &mut rows[ri];
        while r.len() < headers.len() {
            r.push(String::new());
        }
        r[i_roi] = key.0.clone();
        r[i_marker] = key.1.clone();
        r[i_raw] = format!("{:.15}", up.raw_threshold as f64);
        r[i_arc] = format!("{:.6}", up.arcsinh_threshold as f64);
        r[i_method] = up.method.clone();
        r[i_k] = up.kmeans_k.map(|v| v.to_string()).unwrap_or_default();
        r[i_ge] = up.positive_ge.map(|v| v.to_string()).unwrap_or_default();
        r[i_source] = up.source.trim().to_ascii_lowercase();
        r[i_notes] = up.notes.clone();
        r[i_work] = if up.working {
            "TRUE".to_string()
        } else {
            "FALSE".to_string()
        };
    }

    if let Some(parent) = path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create thresholds csv parent dir: {}",
                    parent.to_string_lossy()
                )
            })?;
        }
    }

    let mut out = String::new();
    out.push_str(&csv_join_record(&headers));
    out.push('\n');
    for mut r in rows {
        while r.len() < headers.len() {
            r.push(String::new());
        }
        out.push_str(&csv_join_record(&r));
        out.push('\n');
    }

    let tmp = path.with_extension("tmp");
    fs::write(&tmp, out)
        .with_context(|| format!("failed to write temp csv: {}", tmp.to_string_lossy()))?;

    if path.exists() {
        let _ = fs::remove_file(path);
    }
    fs::rename(&tmp, path).with_context(|| {
        format!(
            "failed to move temp csv into place: {} -> {}",
            tmp.to_string_lossy(),
            path.to_string_lossy()
        )
    })?;

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
            req.key.scale_mode,
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
    scale_mode: ScaleMode,
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
        if scale_mode == ScaleMode::Arcsinh {
            v = v.asinh();
        }

        out_positions.push(egui::pos2(x as f32, y as f32));
        out_values.push(v as f32);
    }

    Ok(())
}

fn list_marker_choices(
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

fn read_channel_labels(dataset_root: &Path, channels_index_path: Option<&Path>) -> Vec<String> {
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

fn infer_roi_label(dataset_root: &Path, ome_multiscale_name: Option<&str>) -> String {
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

fn infer_roi_label_with_layout(dataset_root: &Path, base_dir: &Path, layout: &str) -> String {
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

fn normalize_roi_label(value: &str) -> String {
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

fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest).to_string_lossy().to_string();
        }
    }
    path.to_string()
}

fn loosely_matches(a: &str, b: &str) -> bool {
    let norm = |s: &str| {
        s.chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect::<String>()
    };
    norm(a).contains(&norm(b)) || norm(b).contains(&norm(a))
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

fn parse_csv(text: &str) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;

    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' => {
                if in_quotes {
                    if chars.peek() == Some(&'"') {
                        field.push('"');
                        let _ = chars.next();
                    } else {
                        in_quotes = false;
                    }
                } else {
                    in_quotes = true;
                }
            }
            ',' if !in_quotes => {
                row.push(std::mem::take(&mut field));
            }
            '\n' if !in_quotes => {
                row.push(std::mem::take(&mut field));
                // Skip fully empty trailing line.
                if !(row.len() == 1 && row[0].is_empty() && out.is_empty()) {
                    out.push(std::mem::take(&mut row));
                } else {
                    row.clear();
                }
            }
            '\r' if !in_quotes => {
                if chars.peek() == Some(&'\n') {
                    let _ = chars.next();
                }
                row.push(std::mem::take(&mut field));
                out.push(std::mem::take(&mut row));
            }
            other => field.push(other),
        }
    }

    // Best-effort: accept unterminated quotes as literal.
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        out.push(row);
    }
    out
}

fn csv_escape(field: &str) -> String {
    let needs_quotes = field.contains(['"', ',', '\n', '\r']);
    if !needs_quotes {
        return field.to_string();
    }
    let mut out = String::with_capacity(field.len() + 2);
    out.push('"');
    for ch in field.chars() {
        if ch == '"' {
            out.push('"');
            out.push('"');
        } else {
            out.push(ch);
        }
    }
    out.push('"');
    out
}

fn csv_join_record(fields: &[String]) -> String {
    let mut out = String::new();
    for (i, f) in fields.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&csv_escape(f));
    }
    out
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

fn canonical_marker_token(label: &str) -> String {
    sanitize_marker_token(label)
        .trim_matches('_')
        .to_ascii_lowercase()
}

fn base_marker_label(label: &str) -> String {
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

fn load_thresholds_csv(
    path: &Path,
) -> anyhow::Result<(
    HashMap<(String, String, String), ThresholdCsvRow>,
    HashMap<(String, String, String), ThresholdMeta>,
)> {
    if !path.exists() {
        return Ok((HashMap::new(), HashMap::new()));
    }

    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read thresholds csv: {}", path.to_string_lossy()))?;
    let mut recs = parse_csv(&text);
    if recs.is_empty() {
        return Ok((HashMap::new(), HashMap::new()));
    }
    let header_row = recs.remove(0);
    let headers = header_row
        .iter()
        .enumerate()
        .map(|(i, h)| (h.to_ascii_lowercase(), i))
        .collect::<HashMap<_, _>>();

    let idx = |name: &str| headers.get(&name.to_ascii_lowercase()).copied();
    let i_roi = idx("roi").context("thresholds.csv missing 'roi' column")?;
    let i_marker = idx("marker").context("thresholds.csv missing 'marker' column")?;

    let i_raw = idx("raw_threshold");
    let i_arc = idx("arcsinh_threshold");
    let i_method = idx("method");
    let i_k = idx("kmeans_k");
    let i_ge = idx("positive_ge");
    let i_source = idx("source");
    let i_notes = idx("notes");
    let i_work = idx("working");

    let mut rows = HashMap::new();
    let mut meta = HashMap::new();

    for rec in recs {
        let roi = normalize_roi_label(rec.get(i_roi).map(|s| s.as_str()).unwrap_or("").trim());
        let marker = rec
            .get(i_marker)
            .map(|s| s.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if roi.is_empty() || marker.is_empty() {
            continue;
        }

        let raw_threshold = i_raw
            .and_then(|i| rec.get(i))
            .and_then(|s| s.trim().parse::<f32>().ok())
            .unwrap_or(0.0);
        let arcsinh_threshold = i_arc
            .and_then(|i| rec.get(i))
            .and_then(|s| s.trim().parse::<f32>().ok())
            .unwrap_or_else(|| (raw_threshold as f64).asinh() as f32);
        let method = i_method
            .and_then(|i| rec.get(i))
            .map_or("manual", |v| v.as_str())
            .trim()
            .to_string();
        let source = i_source
            .and_then(|i| rec.get(i))
            .map(|s| s.trim().to_ascii_lowercase())
            .unwrap_or_default();
        let kmeans_k = i_k
            .and_then(|i| rec.get(i))
            .and_then(|s| s.trim().parse::<u8>().ok());
        let positive_ge = i_ge
            .and_then(|i| rec.get(i))
            .and_then(|s| s.trim().parse::<u8>().ok());
        let notes = i_notes
            .and_then(|i| rec.get(i))
            .map_or("", |v| v.as_str())
            .to_string();
        let working = i_work
            .and_then(|i| rec.get(i))
            .map(|s| s.trim().eq_ignore_ascii_case("true") || s.trim() == "1")
            .unwrap_or(true);

        let row = ThresholdCsvRow {
            roi: roi.clone(),
            marker: marker.clone(),
            source: source.clone(),
            raw_threshold,
            arcsinh_threshold,
            method: method.clone(),
            kmeans_k,
            positive_ge,
            notes,
            working,
        };
        rows.insert((roi.clone(), marker.clone(), source.clone()), row);

        let m = method.to_ascii_lowercase();
        let method = if m == "kmeans" {
            AutoMethod::KMeans
        } else if m == "otsu" {
            AutoMethod::Otsu
        } else {
            AutoMethod::Manual
        };
        meta.insert(
            (roi, marker, source),
            ThresholdMeta {
                method,
                kmeans_k,
                positive_ge,
            },
        );
    }

    Ok((rows, meta))
}

fn load_auto_thresholds_json(
    path: &Path,
) -> anyhow::Result<(
    u8,
    Option<String>,
    HashMap<(String, String), AutoThresholdRecord>,
)> {
    if !path.exists() {
        return Ok((6, None, HashMap::new()));
    }
    let text = fs::read_to_string(path).with_context(|| {
        format!(
            "failed to read auto thresholds json: {}",
            path.to_string_lossy()
        )
    })?;

    let root: serde_json::Value =
        serde_json::from_str(&text).context("failed to parse auto thresholds JSON")?;
    let marker_stat = root
        .get("marker_stat")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let mut out: HashMap<(String, String), AutoThresholdRecord> = HashMap::new();
    let mut global_kmeans_k = root.get("kmeans_k").and_then(|v| v.as_u64()).unwrap_or(6) as u8;

    let thresholds = root
        .get("thresholds")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let parse_auto = |node: &serde_json::Value, fallback_k: u8| -> Option<AutoThreshold> {
        let km = node.get("kmeans").and_then(|v| v.as_object());
        let otsu = node.get("otsu").and_then(|v| v.as_object());
        let km_k = km
            .and_then(|o| o.get("k"))
            .and_then(|v| v.as_u64())
            .unwrap_or(fallback_k as u64) as u8;
        let cutoffs = km
            .and_then(|o| o.get("cutoffs_arcsinh"))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_f64())
                    .map(|v| v as f32)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let thr = otsu
            .and_then(|o| o.get("threshold_arcsinh"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32);
        Some(AutoThreshold {
            kmeans_cutoffs_arcsinh: cutoffs,
            otsu_arcsinh: thr,
            kmeans_k: km_k.max(2),
        })
    };

    for rec in thresholds {
        let roi = rec
            .get("roi")
            .and_then(|v| v.as_str())
            .map(normalize_roi_label)
            .unwrap_or_default();
        let marker_key = rec
            .get("marker_key")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_ascii_lowercase())
            .unwrap_or_default();
        if roi.is_empty() || marker_key.is_empty() {
            continue;
        }

        let preferred_source = rec
            .get("preferred_source")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| !s.is_empty());

        let mut sources: HashMap<String, AutoThreshold> = HashMap::new();
        if let Some(obj) = rec.get("sources").and_then(|v| v.as_object()) {
            for (k, v) in obj {
                if let Some(thr) = parse_auto(v, global_kmeans_k) {
                    global_kmeans_k = global_kmeans_k.max(thr.kmeans_k);
                    sources.insert(k.trim().to_ascii_lowercase(), thr);
                }
            }
        }
        // Legacy / fallback: treat record itself as a "standard" threshold set.
        if sources.is_empty() {
            if let Some(thr) = parse_auto(&rec, global_kmeans_k) {
                global_kmeans_k = global_kmeans_k.max(thr.kmeans_k);
                sources.insert("standard".to_string(), thr);
            }
        }

        out.insert(
            (roi, marker_key),
            AutoThresholdRecord {
                preferred_source,
                sources,
            },
        );
    }

    Ok((global_kmeans_k.max(2), marker_stat, out))
}
