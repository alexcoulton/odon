use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crossbeam_channel::Receiver;
use eframe::egui;

use crate::objects::ObjectsLayer;
use crate::render::line_bins::LineSegmentsBins;
use crate::render::line_bins_gl::{LineBinsGlDrawData, LineBinsGlDrawParams, LineBinsGlRenderer};
use crate::render::points::PointsStyle;
use crate::render::points_gl::{PointsGlDrawData, PointsGlDrawParams, PointsGlRenderer};
use crate::spatialdata::{
    ShapesLoadOptions, ShapesRenderKind, SpatialDataTransform2, detect_shapes_render_kind,
    load_shapes_circle_polylines, load_shapes_points, load_shapes_polylines_exterior,
    shapes_support_object_layer,
};

#[derive(Debug)]
pub struct SpatialShapesLayer {
    pub id: u64,
    pub external_id: Option<String>,
    pub external_resource_id: Option<String>,
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub width_screen_px: f32,
    pub color_rgb: [u8; 3],
    pub offset_world: egui::Vec2,

    parquet_path: PathBuf,
    pub transform: SpatialDataTransform2,
    object_layer: Option<ObjectsLayer>,

    data: Option<SpatialShapesData>,
    generation: u64,
    gl_lines: LineBinsGlRenderer,
    gl_points: PointsGlRenderer,
    load_rx: Option<Receiver<anyhow::Result<SpatialShapesData>>>,
    status: String,
}

#[derive(Debug, Clone)]
enum SpatialShapesData {
    Lines(Arc<LineSegmentsBins>),
    Points {
        positions_world: Arc<Vec<egui::Pos2>>,
        values: Arc<Vec<f32>>,
    },
}

impl SpatialShapesLayer {
    pub fn new(
        id: u64,
        external_id: Option<String>,
        external_resource_id: Option<String>,
        name: String,
        parquet_path: PathBuf,
        transform: SpatialDataTransform2,
    ) -> Self {
        let render_kind = detect_shapes_render_kind(&parquet_path).ok();
        let mut object_layer = None;
        let supports_objects = shapes_support_object_layer(&parquet_path).unwrap_or(false);
        if matches!(
            render_kind,
            Some(ShapesRenderKind::Points | ShapesRenderKind::Circles)
        ) || (matches!(render_kind, Some(ShapesRenderKind::Lines)) && supports_objects)
        {
            let mut objects = ObjectsLayer::default();
            objects.visible = true;
            objects.opacity = 0.75;
            objects.width_screen_px = 1.0;
            objects.color_rgb = [0, 255, 120];
            objects.load_spatialdata_shapes(parquet_path.clone(), transform, &name);
            object_layer = Some(objects);
        }
        let mut s = Self {
            id,
            external_id,
            external_resource_id,
            name,
            visible: true,
            opacity: 0.75,
            width_screen_px: 1.0,
            color_rgb: [0, 255, 120],
            offset_world: egui::Vec2::ZERO,
            parquet_path,
            transform,
            object_layer,
            data: None,
            generation: 1,
            gl_lines: LineBinsGlRenderer::new(1024),
            gl_points: PointsGlRenderer::default(),
            load_rx: None,
            status: String::new(),
        };
        if s.object_layer.is_none() {
            s.request_load();
        }
        s
    }

    fn request_load(&mut self) {
        let (tx, rx) = crossbeam_channel::bounded::<anyhow::Result<SpatialShapesData>>(1);
        self.load_rx = Some(rx);
        self.status = "Loading shapes...".to_string();

        let parquet_path = self.parquet_path.clone();
        let options = ShapesLoadOptions {
            transform: self.transform,
            ..Default::default()
        };

        std::thread::Builder::new()
            .name("spatialdata-shapes-loader".to_string())
            .spawn(move || {
                let msg = (|| -> anyhow::Result<SpatialShapesData> {
                    match detect_shapes_render_kind(&parquet_path)? {
                        ShapesRenderKind::Lines => {
                            let polylines =
                                load_shapes_polylines_exterior(&parquet_path, &options)?;
                            let Some(bins) =
                                LineSegmentsBins::build_from_polylines(&polylines, 2048.0)
                            else {
                                anyhow::bail!("no valid segments after parsing");
                            };
                            Ok(SpatialShapesData::Lines(Arc::new(bins)))
                        }
                        ShapesRenderKind::Circles => {
                            let polylines =
                                load_shapes_circle_polylines(&parquet_path, &options, 16)?;
                            let Some(bins) =
                                LineSegmentsBins::build_from_polylines(&polylines, 2048.0)
                            else {
                                anyhow::bail!("no valid circle segments after parsing");
                            };
                            Ok(SpatialShapesData::Lines(Arc::new(bins)))
                        }
                        ShapesRenderKind::Points => {
                            let positions = load_shapes_points(&parquet_path, &options)?;
                            if positions.is_empty() {
                                anyhow::bail!("no valid points after parsing");
                            }
                            Ok(SpatialShapesData::Points {
                                values: Arc::new(vec![1.0f32; positions.len()]),
                                positions_world: Arc::new(positions),
                            })
                        }
                    }
                })();
                let _ = tx.send(msg);
            })
            .ok();
    }

    pub fn tick(&mut self) {
        if let Some(layer) = self.object_layer.as_mut() {
            layer.tick();
            return;
        }
        use crossbeam_channel::TryRecvError;

        let Some(rx) = self.load_rx.as_ref().cloned() else {
            return;
        };
        loop {
            match rx.try_recv() {
                Ok(msg) => {
                    self.load_rx = None;
                    match msg {
                        Ok(data) => {
                            let status = match &data {
                                SpatialShapesData::Lines(bins) => {
                                    format!("Loaded {} segments.", bins.segments.len())
                                }
                                SpatialShapesData::Points {
                                    positions_world, ..
                                } => {
                                    format!("Loaded {} points.", positions_world.len())
                                }
                            };
                            self.data = Some(data);
                            self.generation = self.generation.wrapping_add(1).max(1);
                            self.status = status;
                        }
                        Err(err) => {
                            self.status = format!("Load failed: {err}");
                        }
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.load_rx = None;
                    break;
                }
            }
        }
    }

    pub fn is_loading(&self) -> bool {
        self.load_rx.is_some()
            || self
                .object_layer
                .as_ref()
                .is_some_and(|layer| layer.is_loading())
    }

    pub fn is_busy(&self) -> bool {
        self.is_loading()
            || self
                .object_layer
                .as_ref()
                .is_some_and(|layer| layer.is_busy())
    }

    pub fn draw(
        &mut self,
        ui: &mut egui::Ui,
        camera: &crate::camera::Camera,
        viewport: egui::Rect,
        visible_world: egui::Rect,
        gpu_available: bool,
        local_to_world_offset: egui::Vec2,
    ) {
        // Object-backed shape layers delegate the full draw path so interaction,
        // coloring, and visibility rules stay identical to other object overlays.
        // The fallback path handles the simpler raw line/point representation.
        if let Some(layer) = self.object_layer.as_ref() {
            if !layer.visible {
                return;
            }
        } else if !self.visible {
            return;
        }
        if let Some(layer) = self.object_layer.as_mut() {
            layer.draw(
                ui,
                camera,
                viewport,
                visible_world,
                local_to_world_offset,
                gpu_available,
            );
            return;
        }
        let Some(data) = self.data.as_ref() else {
            return;
        };

        match data {
            SpatialShapesData::Lines(bins) => {
                if bins.segments.is_empty() {
                    return;
                }
                if gpu_available {
                    let a = (self.opacity.clamp(0.0, 1.0) * 255.0).round() as u8;
                    let c = self.color_rgb;
                    let color = egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], a);
                    let data = LineBinsGlDrawData {
                        cache_id: self.id,
                        generation: self.generation,
                        bins: Arc::clone(bins),
                    };
                    let params = LineBinsGlDrawParams {
                        center_world: camera.center_world_lvl0,
                        zoom_screen_per_world: camera.zoom_screen_per_lvl0_px,
                        width_points: self.width_screen_px.max(0.0),
                        color,
                        visible: self.visible,
                        local_to_world_offset,
                        local_to_world_scale: egui::vec2(1.0, 1.0),
                    };
                    let visible_local = visible_world.translate(-local_to_world_offset);
                    let renderer = self.gl_lines.clone();
                    let cb = egui_glow::CallbackFn::new(move |info, painter| {
                        renderer.paint(info, painter, &data, &params, visible_local);
                    });
                    ui.painter().add(egui::PaintCallback {
                        rect: viewport,
                        callback: Arc::new(cb),
                    });
                } else {
                    let a = (self.opacity.clamp(0.0, 1.0) * 255.0).round() as u8;
                    let c = self.color_rgb;
                    let color = egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], a);
                    let stroke = egui::Stroke::new(self.width_screen_px.max(0.25), color);
                    let visible_world = visible_world.translate(-local_to_world_offset);
                    let (x0, y0, x1, y1) = bins.bin_range_for_world_rect(visible_world);
                    for by in y0..=y1 {
                        for bx in x0..=x1 {
                            let idx = by * bins.bins_w + bx;
                            for seg in bins.bin_slice(idx) {
                                let a = egui::pos2(seg[0], seg[1]) + local_to_world_offset;
                                let b = egui::pos2(seg[2], seg[3]) + local_to_world_offset;
                                let a = camera.world_to_screen(a, viewport);
                                let b = camera.world_to_screen(b, viewport);
                                ui.painter().line_segment([a, b], stroke);
                            }
                        }
                    }
                }
            }
            SpatialShapesData::Points {
                positions_world,
                values,
            } => {
                if positions_world.is_empty() {
                    return;
                }
                if gpu_available {
                    let a = (self.opacity.clamp(0.0, 1.0) * 255.0).round() as u8;
                    let c = self.color_rgb;
                    let mut style = PointsStyle::default();
                    style.radius_screen_px = self.width_screen_px.max(0.75) * 2.0;
                    style.fill_positive =
                        egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], a);
                    style.stroke_positive = egui::Stroke::new(0.0, egui::Color32::TRANSPARENT);
                    let data = PointsGlDrawData {
                        generation: self.generation,
                        positions_world: Arc::clone(positions_world),
                        values: Arc::clone(values),
                    };
                    let params = PointsGlDrawParams {
                        center_world: camera.center_world_lvl0,
                        zoom_screen_per_world: camera.zoom_screen_per_lvl0_px,
                        threshold: 0.0,
                        style,
                        visible: self.visible,
                        local_to_world_offset,
                        local_to_world_scale: egui::vec2(1.0, 1.0),
                    };
                    let renderer = self.gl_points.clone();
                    let cb = egui_glow::CallbackFn::new(move |info, painter| {
                        renderer.paint(info, painter, &data, &params);
                    });
                    ui.painter().add(egui::PaintCallback {
                        rect: viewport,
                        callback: Arc::new(cb),
                    });
                } else {
                    let a = (self.opacity.clamp(0.0, 1.0) * 255.0).round() as u8;
                    let c = self.color_rgb;
                    let color = egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], a);
                    let zoom = camera.zoom_screen_per_lvl0_px;
                    let radius_px =
                        (self.width_screen_px.max(0.75) * 2.0 * zoom.sqrt()).clamp(0.75, 40.0);
                    let visible_world = visible_world.translate(-local_to_world_offset);
                    for &p in positions_world.iter() {
                        if !visible_world.contains(p) {
                            continue;
                        }
                        let s = camera.world_to_screen(p + local_to_world_offset, viewport);
                        ui.painter().circle_filled(s, radius_px, color);
                    }
                }
            }
        }
    }

    pub fn ui_properties(&mut self, ui: &mut egui::Ui, default_dir: &Path) -> bool {
        if let Some(layer) = self.object_layer.as_mut() {
            layer.ui_properties(ui, default_dir);
            return false;
        }

        let mut changed = false;
        changed |= ui.checkbox(&mut self.visible, "Visible").changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.opacity, 0.0..=1.0)
                    .text("Opacity")
                    .show_value(true)
                    .clamping(egui::SliderClamping::Always),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.width_screen_px, 0.25..=6.0)
                    .text("Width")
                    .show_value(true)
                    .clamping(egui::SliderClamping::Always),
            )
            .changed();
        ui.horizontal(|ui| {
            ui.label("Color");
            let mut c =
                egui::Color32::from_rgb(self.color_rgb[0], self.color_rgb[1], self.color_rgb[2]);
            if ui.color_edit_button_srgba(&mut c).changed() {
                self.color_rgb = [c.r(), c.g(), c.b()];
                changed = true;
            }
        });
        if !self.status.is_empty() {
            ui.label(self.status.clone());
        }
        changed
    }

    pub fn visible_mut(&mut self) -> &mut bool {
        if let Some(layer) = self.object_layer.as_mut() {
            &mut layer.visible
        } else {
            &mut self.visible
        }
    }

    pub fn hover_tooltip(
        &self,
        pointer_world: egui::Pos2,
        camera: &crate::camera::Camera,
    ) -> Option<Vec<String>> {
        let layer = self.object_layer.as_ref()?;
        layer.hover_tooltip(pointer_world, self.offset_world, camera)
    }

    pub fn select_at(
        &mut self,
        pointer_world: egui::Pos2,
        additive: bool,
        toggle: bool,
        camera: &crate::camera::Camera,
    ) -> bool {
        let Some(layer) = self.object_layer.as_mut() else {
            return false;
        };
        layer.select_at(pointer_world, self.offset_world, camera, additive, toggle);
        true
    }

    pub fn clear_selection(&mut self) {
        if let Some(layer) = self.object_layer.as_mut() {
            layer.clear_selection();
        }
    }

    pub fn has_object_layer(&self) -> bool {
        self.object_layer.is_some()
    }

    pub fn select_objects_by_ids(&mut self, ids: &HashSet<String>) -> usize {
        self.object_layer
            .as_mut()
            .map(|layer| layer.select_objects_by_ids(ids))
            .unwrap_or(0)
    }

    pub fn object_layer_mut(&mut self) -> Option<&mut ObjectsLayer> {
        self.object_layer.as_mut()
    }

    pub fn object_layer(&self) -> Option<&ObjectsLayer> {
        self.object_layer.as_ref()
    }
}
