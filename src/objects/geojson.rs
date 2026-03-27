use std::path::{Path, PathBuf};
use std::sync::Arc;

use crossbeam_channel::Receiver;
use eframe::egui;
use rfd::FileDialog;

use crate::geometry::geojson::{PolygonRingMode, load_geojson_polylines_world};
use crate::render::line_bins::LineSegmentsBins;
use crate::render::line_bins_gl::{LineBinsGlDrawData, LineBinsGlDrawParams, LineBinsGlRenderer};

#[derive(Debug, Clone)]
pub struct GeoJsonSegmentationLayer {
    pub visible: bool,
    pub opacity: f32,
    pub width_screen_px: f32,
    pub color_rgb: [u8; 3],

    pub loaded_geojson: Option<PathBuf>,
    pub downsample_factor: f32,

    bins: Option<Arc<LineSegmentsBins>>,
    generation: u64,
    gl: LineBinsGlRenderer,

    load_rx: Option<Receiver<LoadResult>>,
    status: String,
}

#[derive(Debug)]
struct LoadResult {
    path: PathBuf,
    downsample_factor: f32,
    bins: Arc<LineSegmentsBins>,
}

impl Default for GeoJsonSegmentationLayer {
    fn default() -> Self {
        Self {
            visible: false,
            opacity: 0.75,
            width_screen_px: 1.0,
            color_rgb: [0, 255, 120],
            loaded_geojson: None,
            downsample_factor: 1.0,
            bins: None,
            generation: 1,
            gl: LineBinsGlRenderer::new(512),
            load_rx: None,
            status: String::new(),
        }
    }
}

impl GeoJsonSegmentationLayer {
    pub fn open_dialog(&mut self, default_dir: &Path) {
        let start_dir = self
            .loaded_geojson
            .as_ref()
            .and_then(|p| p.parent())
            .unwrap_or(default_dir);
        if let Some(path) = FileDialog::new()
            .add_filter("GeoJSON", &["geojson", "json"])
            .set_title("Open Segmentation GeoJSON")
            .set_directory(start_dir)
            .pick_file()
        {
            self.request_load(path, self.downsample_factor);
        }
    }

    pub fn tick(&mut self) {
        use crossbeam_channel::TryRecvError;

        let Some(rx) = self.load_rx.as_ref() else {
            return;
        };

        loop {
            match rx.try_recv() {
                Ok(msg) => {
                    self.bins = Some(msg.bins);
                    self.loaded_geojson = Some(msg.path);
                    self.downsample_factor = msg.downsample_factor.max(1e-6);
                    self.visible = true;
                    self.generation = self.generation.wrapping_add(1).max(1);
                    let segs = self.bins.as_ref().map(|b| b.segments.len()).unwrap_or(0);
                    self.status = format!("Loaded {segs} segments.");
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.load_rx = None;
                    break;
                }
            }
        }
    }

    pub fn clear(&mut self) {
        self.bins = None;
        self.loaded_geojson = None;
        self.status.clear();
        self.visible = false;
        self.load_rx = None;
        self.generation = self.generation.wrapping_add(1).max(1);
    }

    pub fn ui_topbar(&mut self, ui: &mut egui::Ui, default_dir: &Path) {
        if ui.button("Load Seg GeoJSON...").clicked() {
            self.open_dialog(default_dir);
        }
    }

    pub fn ui_left_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.visible, "");
            ui.label("Segmentation (GeoJSON)");
        });
        ui.horizontal(|ui| {
            ui.add_enabled(
                self.visible,
                egui::Slider::new(&mut self.opacity, 0.0..=1.0)
                    .text("Opacity")
                    .show_value(true)
                    .clamping(egui::SliderClamping::Always),
            );
        });
        ui.horizontal(|ui| {
            ui.add_enabled(
                self.visible,
                egui::Slider::new(&mut self.width_screen_px, 0.25..=4.0)
                    .text("Width")
                    .show_value(true)
                    .clamping(egui::SliderClamping::Always),
            );
        });
        ui.horizontal(|ui| {
            ui.add_enabled(
                self.visible,
                egui::DragValue::new(&mut self.downsample_factor)
                    .speed(0.1)
                    .prefix("Downsample "),
            );
            if ui
                .add_enabled(self.loaded_geojson.is_some(), egui::Button::new("Reload"))
                .clicked()
            {
                if let Some(path) = self.loaded_geojson.clone() {
                    self.request_load(path, self.downsample_factor);
                }
            }
            if ui
                .add_enabled(self.loaded_geojson.is_some(), egui::Button::new("Clear"))
                .clicked()
            {
                self.clear();
            }
        });
        if let Some(path) = self.loaded_geojson.as_ref() {
            ui.label(
                path.file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("geojson")
                    .to_string(),
            );
        } else {
            ui.label("Not loaded");
        }
        if !self.status.is_empty() {
            ui.label(self.status.clone());
        }
    }

    pub fn ui_properties(&mut self, ui: &mut egui::Ui, default_dir: &Path) {
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.visible, "Visible");
            self.ui_topbar(ui, default_dir);
        });
        ui.add(
            egui::Slider::new(&mut self.opacity, 0.0..=1.0)
                .text("Opacity")
                .show_value(true)
                .clamping(egui::SliderClamping::Always),
        );
        ui.add(
            egui::Slider::new(&mut self.width_screen_px, 0.25..=4.0)
                .text("Width")
                .show_value(true)
                .clamping(egui::SliderClamping::Always),
        );
        ui.horizontal(|ui| {
            ui.add(
                egui::DragValue::new(&mut self.downsample_factor)
                    .speed(0.1)
                    .prefix("Downsample "),
            );
            if ui
                .add_enabled(self.loaded_geojson.is_some(), egui::Button::new("Reload"))
                .clicked()
            {
                if let Some(path) = self.loaded_geojson.clone() {
                    self.request_load(path, self.downsample_factor);
                }
            }
            if ui
                .add_enabled(self.loaded_geojson.is_some(), egui::Button::new("Clear"))
                .clicked()
            {
                self.clear();
            }
        });
        if let Some(path) = self.loaded_geojson.as_ref() {
            ui.label(path.to_string_lossy().to_string());
        } else {
            ui.label("Not loaded");
        }
        if !self.status.is_empty() {
            ui.label(self.status.clone());
        }
    }

    pub fn load_from_path(&mut self, path: &Path, downsample_factor: f32) -> anyhow::Result<usize> {
        let polylines =
            load_geojson_polylines_world(path, downsample_factor, PolygonRingMode::ExteriorOnly)?;
        let Some(bins) = LineSegmentsBins::build_from_polylines(&polylines, 2048.0) else {
            anyhow::bail!("no valid segments after parsing");
        };
        self.bins = Some(Arc::new(bins));
        self.loaded_geojson = Some(path.to_path_buf());
        self.visible = true;
        self.generation = self.generation.wrapping_add(1).max(1);
        Ok(self.bins.as_ref().map(|b| b.segments.len()).unwrap_or(0))
    }

    pub fn is_busy(&self) -> bool {
        self.load_rx.is_some()
    }

    fn request_load(&mut self, path: PathBuf, downsample_factor: f32) {
        let (tx, rx) = crossbeam_channel::bounded::<LoadResult>(1);
        self.load_rx = Some(rx);
        self.status = format!("Loading: {}", path.to_string_lossy());

        std::thread::Builder::new()
            .name("seg-geojson-loader".to_string())
            .spawn(move || {
                if let Ok(msg) = load_in_thread(path, downsample_factor) {
                    let _ = tx.send(msg);
                }
            })
            .ok();
    }

    pub fn draw(
        &self,
        ui: &mut egui::Ui,
        camera: &crate::camera::Camera,
        viewport: egui::Rect,
        visible_world: egui::Rect,
        gpu_available: bool,
    ) {
        let Some(bins) = self.bins.as_ref() else {
            return;
        };
        if !self.visible {
            return;
        }
        if bins.segments.is_empty() {
            return;
        }

        // Prefer GPU when available.
        if gpu_available {
            let a = (self.opacity.clamp(0.0, 1.0) * 255.0).round() as u8;
            let c = self.color_rgb;
            let color = egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], a);
            let data = LineBinsGlDrawData {
                cache_id: 0,
                generation: self.generation,
                bins: Arc::clone(bins),
            };
            let params = LineBinsGlDrawParams {
                center_world: camera.center_world_lvl0,
                zoom_screen_per_world: camera.zoom_screen_per_lvl0_px,
                width_points: self.width_screen_px.max(0.0),
                color,
                visible: self.visible,
                local_to_world_offset: egui::Vec2::ZERO,
                local_to_world_scale: egui::vec2(1.0, 1.0),
            };
            let renderer = self.gl.clone();
            let cb = egui_glow::CallbackFn::new(move |info, painter| {
                renderer.paint(info, painter, &data, &params, visible_world);
            });
            ui.painter().add(egui::PaintCallback {
                rect: viewport,
                callback: Arc::new(cb),
            });
            return;
        }

        // CPU fallback (only used if no GL context is available).
        let a = (self.opacity.clamp(0.0, 1.0) * 255.0).round() as u8;
        let c = self.color_rgb;
        let color = egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], a);
        let stroke = egui::Stroke::new(self.width_screen_px.max(0.0), color);

        let (bx0, by0, bx1, by1) = bins.bin_range_for_world_rect(visible_world);
        for by in by0..=by1 {
            for bx in bx0..=bx1 {
                let bin_index = by * bins.bins_w + bx;
                for seg in bins.bin_slice(bin_index) {
                    let a = camera.world_to_screen(egui::pos2(seg[0], seg[1]), viewport);
                    let b = camera.world_to_screen(egui::pos2(seg[2], seg[3]), viewport);
                    ui.painter().line_segment([a, b], stroke);
                }
            }
        }
    }
}

fn load_in_thread(path: PathBuf, downsample_factor: f32) -> anyhow::Result<LoadResult> {
    let polylines_world =
        load_geojson_polylines_world(&path, downsample_factor, PolygonRingMode::ExteriorOnly)?;
    let Some(bins) = LineSegmentsBins::build_from_polylines(&polylines_world, 2048.0) else {
        anyhow::bail!("no valid segments after parsing");
    };
    Ok(LoadResult {
        path,
        downsample_factor,
        bins: Arc::new(bins),
    })
}
