mod colors;
mod gl;
mod selection;

pub use colors::build_category_luts;

use std::path::PathBuf;
use std::sync::Arc;

use eframe::egui;

use crate::ui::tooltip;

use self::colors::{tint_color32, turbo_rgb_u8};
use self::gl::{AnnotationGlDraw, AnnotationGlDrawParams, AnnotationGlRenderer};
use self::selection::{PointsRadius, pick_nearest_in_roi};
use odon::data::annotations::{
    AnnotationColumnInfo, AnnotationDataset, AnnotationRoiData, AnnotationValueMode,
};
use odon::model::ControlAnnotationLayerProjection;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnotationShape {
    Circle = 0,
    Square = 1,
    Diamond = 2,
    Cross = 3,
}

impl AnnotationShape {
    pub const ALL: [AnnotationShape; 4] = [
        AnnotationShape::Circle,
        AnnotationShape::Square,
        AnnotationShape::Diamond,
        AnnotationShape::Cross,
    ];

    pub fn label(self) -> &'static str {
        match self {
            AnnotationShape::Circle => "Circle",
            AnnotationShape::Square => "Square",
            AnnotationShape::Diamond => "Diamond",
            AnnotationShape::Cross => "Cross",
        }
    }

    pub fn storage_key(self) -> &'static str {
        match self {
            AnnotationShape::Circle => "circle",
            AnnotationShape::Square => "square",
            AnnotationShape::Diamond => "diamond",
            AnnotationShape::Cross => "cross",
        }
    }

    pub fn from_storage_key(value: &str) -> Option<Self> {
        match value {
            "circle" => Some(Self::Circle),
            "square" => Some(Self::Square),
            "diamond" => Some(Self::Diamond),
            "cross" => Some(Self::Cross),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnnotationCategoryStyle {
    pub name: String,
    pub visible: bool,
    pub color: egui::Color32,
    pub shape: AnnotationShape,
}

#[derive(Debug, Clone)]
pub struct AnnotationLayerStyle {
    pub radius_screen_px: f32,
    pub opacity: f32,
    pub stroke: egui::Stroke,
}

impl Default for AnnotationLayerStyle {
    fn default() -> Self {
        Self {
            radius_screen_px: 4.0,
            opacity: 0.9,
            stroke: egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(0, 0, 0, 140)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnnotationParquetConfig {
    pub path: Option<PathBuf>,
    pub roi_id_column: String,
    pub x_column: String,
    pub y_column: String,
    pub value_column: String,
}

impl Default for AnnotationParquetConfig {
    fn default() -> Self {
        Self {
            path: None,
            roi_id_column: "id".to_string(),
            x_column: "x_centroid".to_string(),
            y_column: "y_centroid".to_string(),
            value_column: "cluster_label".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnnotationPointsLayer {
    pub id: u64,
    pub name: String,
    pub visible: bool,
    pub style: AnnotationLayerStyle,
    pub offset_world: egui::Vec2,

    pub parquet: AnnotationParquetConfig,

    // UI state
    pub selected_value_column: String,
    pub status: String,

    // Loaded data
    pub dataset: Option<Arc<AnnotationDataset>>,
    pub category_styles: Vec<AnnotationCategoryStyle>,
    pub continuous_shape: AnnotationShape,
    pub continuous_range: Option<(f32, f32)>,

    // GL
    gl: AnnotationGlRenderer,
    generation: u64,
    schema: Option<Vec<AnnotationColumnInfo>>,
    schema_status: String,
    pending_source_request: Option<AnnotationSourceRequest>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnotationSourceRequest {
    Inspect,
    Load,
    Reload,
}

impl AnnotationPointsLayer {
    pub fn new(id: u64, name: impl Into<String>) -> Self {
        let parquet = AnnotationParquetConfig::default();
        let selected_value_column = parquet.value_column.clone();
        Self {
            id,
            name: name.into(),
            visible: true,
            style: AnnotationLayerStyle::default(),
            offset_world: egui::Vec2::ZERO,
            parquet,
            selected_value_column,
            status: String::new(),
            dataset: None,
            category_styles: Vec::new(),
            continuous_shape: AnnotationShape::Circle,
            continuous_range: None,
            gl: AnnotationGlRenderer::default(),
            generation: 1,
            schema: None,
            schema_status: String::new(),
            pending_source_request: None,
        }
    }

    pub fn apply_control_projection(&mut self, projection: &ControlAnnotationLayerProjection) {
        let state = &projection.state;
        self.id = state.id;
        self.name = state.name.clone();
        self.visible = state.visible;
        self.style.radius_screen_px = state.radius_screen_px;
        self.style.opacity = state.opacity;
        self.style.stroke.width = state.stroke_width;
        self.style.stroke.color = egui::Color32::from_rgba_unmultiplied(
            state.stroke_color_rgb[0],
            state.stroke_color_rgb[1],
            state.stroke_color_rgb[2],
            state.stroke_color_alpha,
        );
        self.offset_world = egui::vec2(state.offset_world[0], state.offset_world[1]);
        self.parquet.path = state.parquet_path.as_deref().map(PathBuf::from);
        self.parquet.roi_id_column = state.roi_id_column.clone();
        self.parquet.x_column = state.x_column.clone();
        self.parquet.y_column = state.y_column.clone();
        self.parquet.value_column = state.value_column.clone();
        self.selected_value_column = state.selected_value_column.clone();
        self.category_styles = state
            .category_styles
            .iter()
            .map(|style| AnnotationCategoryStyle {
                name: style.name.clone(),
                visible: style.visible,
                color: egui::Color32::from_rgb(
                    style.color_rgb[0],
                    style.color_rgb[1],
                    style.color_rgb[2],
                ),
                shape: AnnotationShape::from_storage_key(&style.shape)
                    .unwrap_or(AnnotationShape::Circle),
            })
            .collect();
        self.continuous_shape = state
            .continuous_shape
            .as_deref()
            .and_then(AnnotationShape::from_storage_key)
            .unwrap_or(AnnotationShape::Circle);
        self.continuous_range = state.continuous_range.map(|[low, high]| (low, high));
        self.schema = Some(projection.schema.as_ref().clone());
        self.schema_status = if projection.pending {
            projection.status.clone()
        } else {
            String::new()
        };
        self.status = projection.status.clone();
        self.dataset = projection
            .resource
            .as_ref()
            .map(|resource| Arc::clone(&resource.dataset));
        self.generation = projection.resource_generation.max(1);
    }

    pub fn control_state_json(&self) -> serde_json::Value {
        serde_json::json!({
            "name":self.name,
            "visible":self.visible,
            "radius_screen_px":self.style.radius_screen_px,
            "opacity":self.style.opacity,
            "stroke_width":self.style.stroke.width,
            "stroke_color_rgb":[self.style.stroke.color.r(),self.style.stroke.color.g(),self.style.stroke.color.b()],
            "stroke_color_alpha":self.style.stroke.color.a(),
            "offset_world":[self.offset_world.x,self.offset_world.y],
            "roi_id_column":self.parquet.roi_id_column,
            "x_column":self.parquet.x_column,
            "y_column":self.parquet.y_column,
            "value_column":self.parquet.value_column,
            "selected_value_column":self.selected_value_column,
            "category_styles":self.category_styles.iter().map(|style| serde_json::json!({
                "name":style.name,
                "visible":style.visible,
                "color_rgb":[style.color.r(),style.color.g(),style.color.b()],
                "shape":style.shape.storage_key(),
            })).collect::<Vec<_>>(),
            "continuous_shape":self.continuous_shape.storage_key(),
            "continuous_range":self.continuous_range.map(|(low, high)| [low, high]),
        })
    }

    pub fn take_control_source_request(
        &mut self,
    ) -> Option<(AnnotationSourceRequest, serde_json::Value)> {
        let request = self.pending_source_request.take()?;
        Some((
            request,
            serde_json::json!({
                "layer_id":self.id,
                "path":self.parquet.path.as_ref().map(|path| path.to_string_lossy().into_owned()),
                "roi_id_column":self.parquet.roi_id_column,
                "x_column":self.parquet.x_column,
                "y_column":self.parquet.y_column,
                "value_column":self.parquet.value_column,
            }),
        ))
    }

    pub fn draw_single(
        &mut self,
        ui: &mut egui::Ui,
        viewport: egui::Rect,
        camera_center_world: egui::Pos2,
        zoom_screen_per_world: f32,
        roi_id: &str,
        group_tint: Option<([u8; 3], f32)>,
        use_gpu: bool,
    ) {
        if !self.visible {
            return;
        }
        let Some(ds) = self.dataset.clone() else {
            return;
        };
        let Some(roi) = ds.roi.get(roi_id).cloned() else {
            return;
        };
        self.draw_roi(
            ui,
            viewport,
            camera_center_world,
            zoom_screen_per_world,
            egui::Vec2::ZERO,
            1.0,
            &roi,
            group_tint,
            use_gpu,
        );
    }

    pub fn draw_mosaic_roi(
        &mut self,
        ui: &mut egui::Ui,
        viewport: egui::Rect,
        camera_center_world: egui::Pos2,
        zoom_screen_per_world: f32,
        roi_id: &str,
        roi_offset_world: egui::Vec2,
        roi_scale: f32,
        group_tint: Option<([u8; 3], f32)>,
        use_gpu: bool,
    ) {
        if !self.visible {
            return;
        }
        let Some(ds) = self.dataset.clone() else {
            return;
        };
        let Some(roi) = ds.roi.get(roi_id).cloned() else {
            return;
        };
        self.draw_roi(
            ui,
            viewport,
            camera_center_world,
            zoom_screen_per_world,
            roi_offset_world,
            roi_scale,
            &roi,
            group_tint,
            use_gpu,
        );
    }

    pub fn draw_mosaic(
        &mut self,
        ui: &mut egui::Ui,
        viewport: egui::Rect,
        camera_center_world: egui::Pos2,
        zoom_screen_per_world: f32,
        visible_rois: &[(String, egui::Vec2, f32)],
        group_tint: Option<([u8; 3], f32)>,
        use_gpu: bool,
    ) {
        if !self.visible {
            return;
        }
        let Some(ds) = self.dataset.as_ref() else {
            return;
        };
        if visible_rois.is_empty() {
            return;
        }

        if !use_gpu {
            for (roi_id, off, scale) in visible_rois {
                self.draw_mosaic_roi(
                    ui,
                    viewport,
                    camera_center_world,
                    zoom_screen_per_world,
                    roi_id.as_str(),
                    *off,
                    *scale,
                    group_tint,
                    false,
                );
            }
            return;
        }

        let mut draws: Vec<(AnnotationGlDraw, egui::Vec2, f32)> = Vec::new();
        draws.reserve(visible_rois.len().min(256));
        for (roi_id, off, scale) in visible_rois {
            let Some(roi) = ds.roi.get(roi_id.as_str()) else {
                continue;
            };
            if roi.count == 0 {
                continue;
            }
            draws.push((
                AnnotationGlDraw {
                    generation: self.generation,
                    positions_local: Arc::clone(&roi.positions_local),
                    values: Arc::clone(&roi.values),
                },
                *off,
                *scale,
            ));
        }
        if draws.is_empty() {
            return;
        }

        let mut params = self.gl_params(
            camera_center_world,
            zoom_screen_per_world,
            egui::Vec2::ZERO,
            1.0,
            group_tint,
        );
        let gl = self.gl.clone();
        let layer_off = self.offset_world;
        params.layer_offset_world = layer_off;

        ui.painter().add(egui::PaintCallback {
            rect: viewport,
            callback: Arc::new(egui_glow::CallbackFn::new(move |info, painter| {
                for (draw, roi_off, roi_scale) in &draws {
                    let mut p = params.clone();
                    p.roi_offset_world = *roi_off;
                    p.roi_scale = *roi_scale;
                    gl.paint(&info, painter, draw, &p);
                }
            })),
        });
    }

    fn draw_roi(
        &mut self,
        ui: &mut egui::Ui,
        viewport: egui::Rect,
        camera_center_world: egui::Pos2,
        zoom_screen_per_world: f32,
        roi_offset_world: egui::Vec2,
        roi_scale: f32,
        roi: &AnnotationRoiData,
        group_tint: Option<([u8; 3], f32)>,
        use_gpu: bool,
    ) {
        if roi.count == 0 {
            return;
        }
        let style = self.style.clone();

        if use_gpu {
            let params = self.gl_params(
                camera_center_world,
                zoom_screen_per_world,
                roi_offset_world,
                roi_scale,
                group_tint,
            );
            let draw = AnnotationGlDraw {
                generation: self.generation,
                positions_local: Arc::clone(&roi.positions_local),
                values: Arc::clone(&roi.values),
            };
            let gl = self.gl.clone();
            ui.painter().add(egui::PaintCallback {
                rect: viewport,
                callback: Arc::new(egui_glow::CallbackFn::new(move |info, painter| {
                    gl.paint(&info, painter, &draw, &params);
                })),
            });
        } else {
            // CPU fallback: draw a subset (keeps it responsive without GPU).
            let mut shapes = Vec::new();
            let max = roi.count.min(50_000);
            let radius = style.radius_screen_px.max(0.5);
            let mut n = 0usize;
            for i in 0..roi.count {
                if n >= max {
                    break;
                }
                let local = roi.positions_local[i];
                let world =
                    (roi_offset_world + local.to_vec2() * roi_scale + self.offset_world).to_pos2();
                let screen = {
                    let viewport_center = viewport.min + 0.5 * viewport.size();
                    viewport_center + (world - camera_center_world) * zoom_screen_per_world
                };
                if !viewport.expand(radius + 2.0).contains(screen) {
                    continue;
                }
                let col = self.cpu_point_color(roi.values[i], group_tint);
                shapes.push(egui::Shape::circle_filled(screen, radius, col));
                n += 1;
            }
            ui.painter().extend(shapes);
        }
    }

    fn gl_params(
        &self,
        center_world: egui::Pos2,
        zoom_screen_per_world: f32,
        roi_offset_world: egui::Vec2,
        roi_scale: f32,
        group_tint: Option<([u8; 3], f32)>,
    ) -> AnnotationGlDrawParams {
        let (mode, cat_colors, cat_shapes, cat_visible, value_min, value_max, cont_shape) =
            if let Some(ds) = self.dataset.as_ref() {
                match ds.mode {
                    AnnotationValueMode::Categorical => {
                        let (colors, shapes, visible) =
                            build_category_luts(&self.category_styles, group_tint);
                        (
                            AnnotationValueMode::Categorical,
                            Arc::new(colors),
                            Arc::new(shapes),
                            Arc::new(visible),
                            0.0,
                            1.0,
                            AnnotationShape::Circle,
                        )
                    }
                    AnnotationValueMode::Continuous => {
                        let (lo, hi) = self
                            .continuous_range
                            .unwrap_or((ds.value_min, ds.value_max));
                        (
                            AnnotationValueMode::Continuous,
                            Arc::new(Vec::new()),
                            Arc::new(Vec::new()),
                            Arc::new(Vec::new()),
                            lo,
                            hi,
                            self.continuous_shape,
                        )
                    }
                }
            } else {
                (
                    AnnotationValueMode::Categorical,
                    Arc::new(Vec::new()),
                    Arc::new(Vec::new()),
                    Arc::new(Vec::new()),
                    0.0,
                    1.0,
                    AnnotationShape::Circle,
                )
            };

        AnnotationGlDrawParams {
            center_world,
            zoom_screen_per_world,
            roi_offset_world,
            roi_scale,
            layer_offset_world: self.offset_world,
            radius_screen_px: self.style.radius_screen_px,
            opacity: self.style.opacity,
            stroke: if let Some((rgb, strength)) =
                group_tint.filter(|_| mode == AnnotationValueMode::Continuous)
            {
                let c = tint_color32(self.style.stroke.color, rgb, strength);
                egui::Stroke {
                    color: c,
                    ..self.style.stroke
                }
            } else {
                self.style.stroke
            },
            mode,
            cat_colors,
            cat_shapes,
            cat_visible,
            value_min,
            value_max,
            continuous_shape: cont_shape,
        }
    }

    fn cpu_point_color(&self, value: f32, group_tint: Option<([u8; 3], f32)>) -> egui::Color32 {
        let a = (self.style.opacity.clamp(0.0, 1.0) * 255.0).round() as u8;
        if let Some(ds) = self.dataset.as_ref() {
            match ds.mode {
                AnnotationValueMode::Categorical => {
                    let n = self.category_styles.len().max(1);
                    let mut idx = value.round() as i32;
                    idx %= n as i32;
                    if idx < 0 {
                        idx += n as i32;
                    }
                    let c = self
                        .category_styles
                        .get(idx as usize)
                        .map(|s| s.color)
                        .unwrap_or(egui::Color32::from_rgb(255, 255, 255));
                    let mut out = egui::Color32::from_rgba_unmultiplied(
                        c.r(),
                        c.g(),
                        c.b(),
                        (c.a() as u16 * a as u16 / 255) as u8,
                    );
                    if let Some((rgb, strength)) = group_tint {
                        out = tint_color32(out, rgb, strength);
                    }
                    out
                }
                AnnotationValueMode::Continuous => {
                    let (lo, hi) = self
                        .continuous_range
                        .unwrap_or((ds.value_min, ds.value_max));
                    let t = if hi > lo {
                        ((value - lo) / (hi - lo)).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let (r, g, b) = turbo_rgb_u8(t);
                    let mut out = egui::Color32::from_rgba_unmultiplied(r, g, b, a);
                    if let Some((rgb, strength)) = group_tint {
                        out = tint_color32(out, rgb, strength);
                    }
                    out
                }
            }
        } else {
            let mut out = egui::Color32::from_rgba_unmultiplied(255, 255, 255, a);
            if let Some((rgb, strength)) = group_tint {
                out = tint_color32(out, rgb, strength);
            }
            out
        }
    }

    pub fn ui_properties(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;

        changed |= ui.checkbox(&mut self.visible, "Visible").changed();

        ui.separator();
        ui.label("Style");
        changed |= ui
            .add(
                egui::Slider::new(&mut self.style.radius_screen_px, 0.5..=20.0)
                    .text("Size")
                    .show_value(true)
                    .clamping(egui::SliderClamping::Always),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.style.opacity, 0.0..=1.0)
                    .text("Opacity")
                    .show_value(true)
                    .clamping(egui::SliderClamping::Always),
            )
            .changed();
        ui.horizontal(|ui| {
            ui.label("Stroke");
            changed |= ui
                .add(egui::DragValue::new(&mut self.style.stroke.width).speed(0.25))
                .changed();
            changed |= ui
                .color_edit_button_srgba(&mut self.style.stroke.color)
                .changed();
        });

        ui.separator();
        ui.label("Source (Parquet)");
        ui.horizontal(|ui| {
            let path_txt = self
                .parquet
                .path
                .as_ref()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|| "(none)".to_string());
            ui.monospace(path_txt);
        });

        ui.horizontal(|ui| {
            if ui.button("Choose…").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Parquet", &["parquet"])
                    .pick_file()
                {
                    self.parquet.path = Some(path);
                    self.schema = None;
                    self.dataset = None;
                    self.status.clear();
                    self.pending_source_request = Some(AnnotationSourceRequest::Inspect);
                    changed = true;
                }
            }
            if ui.button("Reload").clicked() {
                self.pending_source_request = Some(AnnotationSourceRequest::Reload);
            }
        });
        if !self.schema_status.is_empty() {
            ui.label(self.schema_status.clone());
        }

        ui.separator();
        ui.label("Columns");
        if let Some(cols) = self.schema.as_ref() {
            let all_names: Vec<String> = cols.iter().map(|c| c.name.clone()).collect();
            ui.horizontal(|ui| {
                ui.label("ROI id");
                egui::ComboBox::from_id_salt(("ann-roi-id-col", self.id))
                    .selected_text(self.parquet.roi_id_column.clone())
                    .show_ui(ui, |ui| {
                        for name in &all_names {
                            changed |= ui
                                .selectable_value(
                                    &mut self.parquet.roi_id_column,
                                    name.clone(),
                                    name,
                                )
                                .changed();
                        }
                    });
            });
            ui.horizontal(|ui| {
                ui.label("x");
                egui::ComboBox::from_id_salt(("ann-x-col", self.id))
                    .selected_text(self.parquet.x_column.clone())
                    .show_ui(ui, |ui| {
                        for name in &all_names {
                            changed |= ui
                                .selectable_value(&mut self.parquet.x_column, name.clone(), name)
                                .changed();
                        }
                    });
            });
            ui.horizontal(|ui| {
                ui.label("y");
                egui::ComboBox::from_id_salt(("ann-y-col", self.id))
                    .selected_text(self.parquet.y_column.clone())
                    .show_ui(ui, |ui| {
                        for name in &all_names {
                            changed |= ui
                                .selectable_value(&mut self.parquet.y_column, name.clone(), name)
                                .changed();
                        }
                    });
            });
            ui.horizontal(|ui| {
                ui.label("Value");
                let cur = self.parquet.value_column.clone();
                egui::ComboBox::from_id_salt(("ann-val-col", self.id))
                    .selected_text(cur)
                    .show_ui(ui, |ui| {
                        for name in &all_names {
                            changed |= ui
                                .selectable_value(
                                    &mut self.parquet.value_column,
                                    name.clone(),
                                    name,
                                )
                                .changed();
                        }
                    });
            });
        } else {
            ui.label("Schema not loaded yet.");
        }
        if ui.button("Load").clicked() {
            self.selected_value_column = self.parquet.value_column.clone();
            self.pending_source_request = Some(AnnotationSourceRequest::Load);
            changed = true;
        }

        if let Some(ds) = self.dataset.as_ref() {
            ui.separator();
            ui.label(format!(
                "Loaded: {} points across {} ROIs",
                ds.total_points, ds.total_rois
            ));

            match ds.mode {
                AnnotationValueMode::Categorical => {
                    ui.separator();
                    ui.label("Categories");
                    egui::ScrollArea::vertical()
                        .id_salt(("annotation-categories", self.id))
                        .max_height(260.0)
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            for i in 0..self.category_styles.len() {
                                let Some(s) = self.category_styles.get_mut(i) else {
                                    continue;
                                };
                                ui.horizontal(|ui| {
                                    changed |= ui.checkbox(&mut s.visible, "").changed();
                                    changed |= ui.color_edit_button_srgba(&mut s.color).changed();
                                    egui::ComboBox::from_id_salt(("ann-shape", self.id, i))
                                        .selected_text(s.shape.label())
                                        .show_ui(ui, |ui| {
                                            for sh in AnnotationShape::ALL {
                                                changed |= ui
                                                    .selectable_value(&mut s.shape, sh, sh.label())
                                                    .changed();
                                            }
                                        });
                                    ui.label(s.name.clone());
                                });
                            }
                        });
                }
                AnnotationValueMode::Continuous => {
                    ui.separator();
                    ui.label("Continuous");
                    let (mut lo, mut hi) = self
                        .continuous_range
                        .unwrap_or((ds.value_min, ds.value_max));
                    ui.horizontal(|ui| {
                        ui.label("Min");
                        changed |= ui.add(egui::DragValue::new(&mut lo).speed(1.0)).changed();
                        ui.label("Max");
                        changed |= ui.add(egui::DragValue::new(&mut hi).speed(1.0)).changed();
                    });
                    if hi < lo {
                        std::mem::swap(&mut lo, &mut hi);
                    }
                    self.continuous_range = Some((lo, hi));
                    egui::ComboBox::from_id_salt(("ann-cont-shape", self.id))
                        .selected_text(self.continuous_shape.label())
                        .show_ui(ui, |ui| {
                            for sh in AnnotationShape::ALL {
                                changed |= ui
                                    .selectable_value(&mut self.continuous_shape, sh, sh.label())
                                    .changed();
                            }
                        });
                }
            }
        }

        ui.separator();
        ui.label(self.status.clone());

        changed
    }

    pub fn maybe_hover_tooltip(
        &self,
        ctx: &egui::Context,
        _viewport: egui::Rect,
        pointer_world: egui::Pos2,
        zoom_screen_per_world: f32,
        roi_id: &str,
        roi_offset_world: egui::Vec2,
        roi_scale: f32,
    ) {
        if !self.visible {
            return;
        }
        let Some(ds) = self.dataset.as_ref() else {
            return;
        };
        let Some(roi) = ds.roi.get(roi_id) else {
            return;
        };
        if roi.count == 0 {
            return;
        }

        let radius_points =
            PointsRadius::effective(self.style.radius_screen_px, zoom_screen_per_world);
        let radius_world = (radius_points * 1.25) / zoom_screen_per_world.max(1e-6);
        let radius_local = radius_world / roi_scale.max(1e-6);

        // Convert pointer to local coordinates for this ROI.
        let local = ((pointer_world.to_vec2() - roi_offset_world - self.offset_world) / roi_scale)
            .to_pos2();

        let Some(picked) = pick_nearest_in_roi(roi, local, radius_local) else {
            return;
        };
        let value = roi.values.get(picked).copied().unwrap_or(0.0);
        let value_text = self.format_value(ds.as_ref(), value);
        let x = roi
            .positions_local
            .get(picked)
            .map(|p| p.x)
            .unwrap_or(local.x);
        let y = roi
            .positions_local
            .get(picked)
            .map(|p| p.y)
            .unwrap_or(local.y);
        let col_name = self.parquet.value_column.clone();
        let roi_id = roi_id.to_string();

        tooltip::show_tooltip_at_pointer(
            ctx,
            egui::Id::new(("annotations-tooltip", self.id)),
            move |ui| {
                ui.label(format!("ROI: {roi_id}"));
                ui.label(format!("{col_name}: {value_text}"));
                ui.separator();
                ui.monospace(format!("x={x:.1}  y={y:.1}"));
            },
        );
    }

    fn format_value(&self, ds: &AnnotationDataset, value: f32) -> String {
        match ds.mode {
            AnnotationValueMode::Categorical => {
                let idx = value.round() as i32;
                let idx = idx.max(0) as usize;
                ds.categories
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| format!("#{idx}"))
            }
            AnnotationValueMode::Continuous => format!("{value:.4}"),
        }
    }
}
