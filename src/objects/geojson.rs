use std::path::{Path, PathBuf};
use std::sync::Arc;

use eframe::egui;
use rfd::FileDialog;

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

    resource_generation: u64,
    pending: bool,
    status: String,
}

#[derive(Debug, Clone)]
pub enum GeoJsonSourceAction {
    Load(PathBuf, f32),
    Clear,
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
            resource_generation: 0,
            pending: false,
            status: String::new(),
        }
    }
}

impl GeoJsonSegmentationLayer {
    pub fn open_dialog(&self, default_dir: &Path) -> Option<PathBuf> {
        let start_dir = self
            .loaded_geojson
            .as_ref()
            .and_then(|p| p.parent())
            .unwrap_or(default_dir);
        FileDialog::new()
            .add_filter("GeoJSON", &["geojson", "json"])
            .set_title("Open Segmentation GeoJSON")
            .set_directory(start_dir)
            .pick_file()
    }

    pub fn install_control_resource(
        &mut self,
        generation: u64,
        resource: Option<&odon::model::ControlSegmentationGeoJsonResource>,
        state: &serde_json::Value,
    ) -> Result<(), String> {
        self.pending = state
            .get("pending")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        self.status = state
            .get("status")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string();
        self.downsample_factor = state
            .get("downsample_factor")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1.0) as f32;
        self.loaded_geojson = state
            .get("path")
            .and_then(serde_json::Value::as_str)
            .map(PathBuf::from);
        if generation <= self.resource_generation {
            return Ok(());
        }
        self.resource_generation = generation;
        self.generation = self.generation.wrapping_add(1).max(1);
        self.bins = resource
            .map(|resource| {
                let polylines = resource
                    .polylines
                    .iter()
                    .map(|line| {
                        line.iter()
                            .map(|point| egui::pos2(point[0], point[1]))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                LineSegmentsBins::build_from_polylines(&polylines, 2048.0)
                    .map(Arc::new)
                    .ok_or_else(|| "segmentation GeoJSON has no valid segments".to_string())
            })
            .transpose()?;
        Ok(())
    }

    pub fn ui_topbar(&self, ui: &mut egui::Ui, default_dir: &Path) -> Option<GeoJsonSourceAction> {
        if ui.button("Load Seg GeoJSON...").clicked() {
            return self
                .open_dialog(default_dir)
                .map(|path| GeoJsonSourceAction::Load(path, self.downsample_factor));
        }
        None
    }

    pub fn ui_properties(
        &mut self,
        ui: &mut egui::Ui,
        default_dir: &Path,
    ) -> Option<GeoJsonSourceAction> {
        let mut action = None;
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.visible, "Visible");
            action = self.ui_topbar(ui, default_dir);
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
                    action = Some(GeoJsonSourceAction::Load(path, self.downsample_factor));
                }
            }
            if ui
                .add_enabled(self.loaded_geojson.is_some(), egui::Button::new("Clear"))
                .clicked()
            {
                action = Some(GeoJsonSourceAction::Clear);
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
        action
    }

    pub fn is_busy(&self) -> bool {
        self.pending
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
