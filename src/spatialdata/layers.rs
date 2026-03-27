use std::collections::HashMap;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crossbeam_channel::Receiver;
use eframe::egui;

use crate::features::points::{
    FeaturePickerItem, FeaturePointLod, FeaturePointSeries, color_for_feature,
    normalize_feature_key, select_draw_payload, show_feature_picker,
};
use crate::render::line_bins::LineSegmentsBins;
use crate::render::line_bins_gl::{LineBinsGlDrawData, LineBinsGlDrawParams, LineBinsGlRenderer};
use crate::render::point_bins::PointIndexBins;
use crate::render::points::PointsStyle;
use crate::render::points_gl::{PointsGlDrawData, PointsGlDrawParams, PointsGlRenderer};
use crate::objects::ObjectsLayer;
use crate::spatialdata::{
    PointsLoadOptions, PointsMeta, ShapesLoadOptions, ShapesRenderKind, detect_shapes_render_kind,
    load_points_sample, load_shapes_circle_polylines, load_shapes_points,
    load_shapes_polylines_exterior, shapes_support_object_layer,
};
use crate::spatialdata::{SpatialDataElement, SpatialDataTransform2};

// SpatialData elements are discovered from format-specific metadata, then adapted
// into the viewer's native overlay types. The rest of the app should not need to
// care whether a shape/point layer came from SpatialData or from another source.

#[derive(Debug, Default)]
pub struct SpatialDataLayers {
    pub root: Option<PathBuf>,
    pub tables: Vec<SpatialDataElement>,
    pub shapes: Vec<SpatialShapesLayer>,
    pub points: Option<SpatialPointsLayer>,
    next_shape_layer_id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositiveCellSelectionTarget {
    SegmentationObjects,
    AllObjectLayers,
    ShapeLayer(u64),
}

impl SpatialDataLayers {
    pub fn clear(&mut self) {
        self.root = None;
        self.tables.clear();
        self.shapes.clear();
        self.points = None;
        self.next_shape_layer_id = 1;
    }

    pub fn set_root(&mut self, root: PathBuf) {
        self.root = Some(root);
    }

    pub fn set_tables(&mut self, tables: Vec<SpatialDataElement>) {
        self.tables = tables;
    }

    fn root(&self) -> Option<&Path> {
        self.root.as_deref()
    }

    pub fn load_shapes(&mut self, element: &SpatialDataElement) -> u64 {
        // Shape elements always start as lightweight SpatialShapesLayer wrappers.
        // The actual load step decides whether they stay as raw polylines/points or
        // are promoted into an ObjectsLayer for shared selection/filtering behavior.
        let Some(root) = self.root().map(|p| p.to_path_buf()) else {
            return 0;
        };
        let Some(rel) = element.rel_parquet.clone() else {
            return 0;
        };
        let path = root.join(rel);
        let id = self.next_shape_layer_id.max(1);
        self.next_shape_layer_id = id.wrapping_add(1).max(1);
        let layer = SpatialShapesLayer::new(
            id,
            format!("Shapes: {}", element.name),
            path,
            element.transform,
        );
        self.shapes.push(layer);
        id
    }

    pub fn load_points(&mut self, element: &SpatialDataElement, max_points: usize) {
        self.load_points_with_image_size(element, max_points, None);
    }

    pub fn load_points_with_image_size(
        &mut self,
        element: &SpatialDataElement,
        max_points: usize,
        image_size_world: Option<[f32; 2]>,
    ) {
        // Points are rebuilt from the parquet source each time because preparation
        // derives world-space bounds, feature caches, and optional image scaling
        // from the current metadata instead of storing a second normalized copy.
        let Some(root) = self.root().map(|p| p.to_path_buf()) else {
            return;
        };
        let Some(rel) = element.rel_parquet.clone() else {
            return;
        };
        let path = root.join(rel);
        let layer = SpatialPointsLayer::new(
            format!("Points: {}", element.name),
            path,
            element.transform,
            element.feature_key.clone(),
            max_points,
            image_size_world,
        );
        self.points = Some(layer);
    }

    pub fn tick(&mut self) {
        for s in &mut self.shapes {
            s.tick();
        }
        if let Some(p) = self.points.as_mut() {
            p.tick();
        }
    }

    pub fn is_loading_shapes(&self) -> bool {
        self.shapes.iter().any(|s| s.is_loading())
    }

    pub fn is_loading_points(&self) -> bool {
        self.points.as_ref().is_some_and(|p| p.is_loading())
    }

    pub fn is_busy(&self) -> bool {
        self.is_loading_shapes() || self.is_loading_points()
    }

    pub fn select_positive_cells_by_ids(
        &mut self,
        cell_ids: &[String],
        target: PositiveCellSelectionTarget,
    ) -> Option<(usize, usize)> {
        if target == PositiveCellSelectionTarget::SegmentationObjects {
            return None;
        }
        let id_set = cell_ids.iter().cloned().collect::<HashSet<_>>();
        if id_set.is_empty() {
            return None;
        }

        let mut matched_layers = 0usize;
        let mut matched_objects = 0usize;
        for layer in &mut self.shapes {
            match target {
                PositiveCellSelectionTarget::AllObjectLayers => {}
                PositiveCellSelectionTarget::ShapeLayer(id) if id == layer.id => {}
                PositiveCellSelectionTarget::ShapeLayer(_) => continue,
                PositiveCellSelectionTarget::SegmentationObjects => continue,
            }
            let selected = layer.select_objects_by_ids(&id_set);
            if selected > 0 {
                matched_layers += 1;
                matched_objects += selected;
            }
        }

        (matched_layers > 0).then_some((matched_layers, matched_objects))
    }

    pub fn positive_cell_selection_targets(&self) -> Vec<(PositiveCellSelectionTarget, String)> {
        self.shapes
            .iter()
            .filter(|layer| layer.has_object_layer())
            .map(|layer| {
                (
                    PositiveCellSelectionTarget::ShapeLayer(layer.id),
                    layer.name.clone(),
                )
            })
            .collect()
    }

    pub fn table_elements(&self) -> &[SpatialDataElement] {
        &self.tables
    }
}

#[derive(Debug)]
pub struct SpatialShapesLayer {
    pub id: u64,
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub width_screen_px: f32,
    pub color_rgb: [u8; 3],
    pub offset_world: egui::Vec2,

    parquet_path: PathBuf,
    transform: SpatialDataTransform2,
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

    pub fn is_loaded(&self) -> bool {
        if let Some(layer) = self.object_layer.as_ref() {
            return layer.has_data();
        }
        match self.data.as_ref() {
            Some(SpatialShapesData::Lines(b)) => !b.segments.is_empty(),
            Some(SpatialShapesData::Points {
                positions_world, ..
            }) => !positions_world.is_empty(),
            None => false,
        }
    }

    pub fn is_loading(&self) -> bool {
        self.load_rx.is_some()
            || self
                .object_layer
                .as_ref()
                .is_some_and(|layer| layer.is_loading())
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

    pub fn status_text(&self) -> String {
        self.object_layer
            .as_ref()
            .map(|l| l.status().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| self.status.clone())
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

#[derive(Debug)]
pub struct SpatialPointsLayer {
    pub name: String,
    pub visible: bool,
    pub style: PointsStyle,
    pub threshold: f32,
    pub max_render_points_total: usize,

    points_parquet_dir: PathBuf,
    options: PointsLoadOptions,
    base_transform: SpatialDataTransform2,
    image_size_world: Option<[f32; 2]>,
    scale_mode: SpatialScaleMode,
    axis_mode: SpatialAxisMode,
    scale_mul: f32,
    feature_query: String,
    feature_popup_open: bool,
    positive_cell_min_count: usize,
    positive_cell_target: PositiveCellSelectionTarget,
    positive_cell_target_initialized: bool,
    pending_positive_cell_selection: Option<PositiveCellSelectionRequest>,
    cell_selection_status: String,
    last_match_count: usize,
    last_loaded_count: usize,
    last_auto_choice: String,

    generation: u64,
    raw_xy: Option<Arc<Vec<[f32; 2]>>>,
    meta: Option<Arc<PointsMeta>>,
    positions_world: Option<Arc<Vec<egui::Pos2>>>,
    visible_raw_indices: Option<Arc<Vec<u32>>>,
    values: Option<Arc<Vec<f32>>>,
    lod_levels: Option<Arc<Vec<FeaturePointLod>>>,
    feature_points: HashMap<String, SpatialFeaturePoints>,
    feature_cache: Vec<Option<Arc<SpatialFeatureCache>>>,
    hover_positions_world: Option<Arc<Vec<egui::Pos2>>>,
    hover_raw_indices: Option<Arc<Vec<u32>>>,
    bounds_world: Option<egui::Rect>,
    bins: Option<Arc<PointIndexBins>>,
    load_rx: Option<Receiver<anyhow::Result<PreparedSpatialPoints>>>,
    status: String,
    gl: PointsGlRenderer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpatialScaleMode {
    Auto,
    UseScale,
    InvertScale,
    Identity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpatialAxisMode {
    Auto,
    XY,
    YX,
}

#[derive(Debug)]
struct SpatialFeaturePoints {
    feature_id: u32,
    raw_indices: Option<Arc<Vec<u32>>>,
    series: FeaturePointSeries,
}

#[derive(Debug)]
struct SpatialFeatureCache {
    positions_world: Arc<Vec<egui::Pos2>>,
    raw_indices: Arc<Vec<u32>>,
    values: Arc<Vec<f32>>,
    lod_levels: Arc<Vec<FeaturePointLod>>,
}

#[derive(Debug)]
struct PositiveCellSelectionRequest {
    cell_ids: Vec<String>,
    target: PositiveCellSelectionTarget,
}

#[derive(Debug, Clone, Copy)]
struct SpatialPointsPrepareConfig {
    base_transform: SpatialDataTransform2,
    image_size_world: Option<[f32; 2]>,
    scale_mode: SpatialScaleMode,
    axis_mode: SpatialAxisMode,
    scale_mul: f32,
}

#[derive(Debug)]
struct PreparedSpatialPoints {
    raw_xy: Arc<Vec<[f32; 2]>>,
    meta: Arc<PointsMeta>,
    positions_world: Arc<Vec<egui::Pos2>>,
    values: Arc<Vec<f32>>,
    lod_levels: Arc<Vec<FeaturePointLod>>,
    feature_counts: Vec<usize>,
    feature_cache: Vec<Option<Arc<SpatialFeatureCache>>>,
    bounds_world: Option<egui::Rect>,
    bins: Option<Arc<PointIndexBins>>,
    last_auto_choice: String,
    loaded_count: usize,
}

impl SpatialPointsLayer {
    pub fn new(
        name: String,
        points_parquet_dir: PathBuf,
        transform: SpatialDataTransform2,
        feature_key: Option<String>,
        max_points: usize,
        image_size_world: Option<[f32; 2]>,
    ) -> Self {
        let mut options = PointsLoadOptions::default();
        options.max_points = max_points;
        if let Some(k) = feature_key {
            if !k.trim().is_empty() {
                options.feature_column = Some(k);
            }
        }

        let mut s = Self {
            name,
            visible: false,
            style: PointsStyle::default(),
            threshold: 0.5,
            max_render_points_total: 200_000,
            points_parquet_dir,
            options,
            base_transform: transform,
            image_size_world,
            scale_mode: SpatialScaleMode::Auto,
            axis_mode: SpatialAxisMode::Auto,
            scale_mul: 1.0,
            feature_query: String::new(),
            feature_popup_open: false,
            positive_cell_min_count: 1,
            positive_cell_target: PositiveCellSelectionTarget::AllObjectLayers,
            positive_cell_target_initialized: false,
            pending_positive_cell_selection: None,
            cell_selection_status: String::new(),
            last_match_count: 0,
            last_loaded_count: 0,
            last_auto_choice: String::new(),
            generation: 1,
            raw_xy: None,
            meta: None,
            positions_world: None,
            visible_raw_indices: None,
            values: None,
            lod_levels: None,
            feature_points: HashMap::new(),
            feature_cache: Vec::new(),
            hover_positions_world: None,
            hover_raw_indices: None,
            bounds_world: None,
            bins: None,
            load_rx: None,
            status: String::new(),
            gl: PointsGlRenderer::default(),
        };
        s.request_load();
        s
    }

    fn request_load(&mut self) {
        let (tx, rx) = crossbeam_channel::bounded::<anyhow::Result<PreparedSpatialPoints>>(1);
        self.load_rx = Some(rx);
        self.status = "Loading points...".to_string();
        self.visible = true;

        let dir = self.points_parquet_dir.clone();
        let options = self.options.clone();
        let config = self.prepare_config();

        std::thread::Builder::new()
            .name("spatialdata-points-loader".to_string())
            .spawn(move || {
                let msg = load_points_sample(&dir, &options)
                    .and_then(|payload| prepare_spatial_points_payload(payload, config));
                let _ = tx.send(msg);
            })
            .ok();
    }

    pub fn tick(&mut self) {
        use crossbeam_channel::TryRecvError;

        let Some(rx) = self.load_rx.as_ref().cloned() else {
            return;
        };
        loop {
            match rx.try_recv() {
                Ok(msg) => {
                    self.load_rx = None;
                    match msg {
                        Ok(prepared) => {
                            let n = prepared.loaded_count;
                            self.apply_prepared_snapshot(prepared);
                            self.generation = self.generation.wrapping_add(1).max(1);
                            self.last_loaded_count = n;
                            self.status = format!("Loaded {n} points (sample).");
                        }
                        Err(err) => {
                            self.status = format!("Load failed: {err}");
                            self.visible = false;
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

    pub fn is_loaded(&self) -> bool {
        self.hover_positions_world
            .as_ref()
            .is_some_and(|p| !p.is_empty())
    }

    pub fn is_loading(&self) -> bool {
        self.load_rx.is_some()
    }

    pub fn bounds_world(&self) -> Option<egui::Rect> {
        self.bounds_world
    }

    pub fn take_positive_cell_selection_request(
        &mut self,
    ) -> Option<(Vec<String>, PositiveCellSelectionTarget)> {
        self.pending_positive_cell_selection
            .take()
            .map(|req| (req.cell_ids, req.target))
    }

    pub fn set_cell_selection_status(&mut self, status: String) {
        self.cell_selection_status = status;
    }

    fn ensure_positive_cell_target_initialized(
        &mut self,
        targets: &[(PositiveCellSelectionTarget, String)],
    ) {
        let current_valid = match self.positive_cell_target {
            PositiveCellSelectionTarget::SegmentationObjects => targets
                .iter()
                .any(|(target, _)| *target == PositiveCellSelectionTarget::SegmentationObjects),
            PositiveCellSelectionTarget::AllObjectLayers => true,
            PositiveCellSelectionTarget::ShapeLayer(id) => targets
                .iter()
                .any(|(target, _)| *target == PositiveCellSelectionTarget::ShapeLayer(id)),
        };
        if self.positive_cell_target_initialized && current_valid {
            return;
        }

        if let Some((target, _)) = targets
            .iter()
            .find(|(_, name)| name.to_ascii_lowercase().contains("cell_boundaries"))
        {
            self.positive_cell_target = *target;
        } else if let Some((target, _)) = targets.first() {
            self.positive_cell_target = *target;
        } else {
            self.positive_cell_target = PositiveCellSelectionTarget::AllObjectLayers;
        }
        self.positive_cell_target_initialized = true;
    }

    fn prepare_config(&self) -> SpatialPointsPrepareConfig {
        SpatialPointsPrepareConfig {
            base_transform: self.base_transform,
            image_size_world: self.image_size_world,
            scale_mode: self.scale_mode,
            axis_mode: self.axis_mode,
            scale_mul: self.scale_mul,
        }
    }

    fn has_enabled_features(&self) -> bool {
        self.feature_points.values().any(|f| f.series.enabled)
    }

    fn clear_feature_draw_data(&mut self) {
        for feature in self.feature_points.values_mut() {
            feature.series.clear_payload();
            feature.raw_indices = None;
        }
    }

    fn sync_feature_points_with_counts(&mut self, counts_by_id: &[usize]) {
        let Some(feature) = self.meta.as_ref().and_then(|m| m.feature.as_ref()) else {
            self.feature_points.clear();
            self.feature_cache.clear();
            return;
        };

        let mut prior = std::mem::take(&mut self.feature_points);
        let mut next = HashMap::new();
        for (feature_id, feature_name) in feature.dict.iter().enumerate() {
            let key = normalize_feature_key(feature_name);
            let mut entry = prior.remove(&key).unwrap_or_else(|| SpatialFeaturePoints {
                feature_id: feature_id as u32,
                raw_indices: None,
                series: FeaturePointSeries::new(
                    feature_name.clone(),
                    color_for_feature(feature_name),
                ),
            });
            entry.series.feature_name = feature_name.clone();
            entry.feature_id = feature_id as u32;
            entry.series.point_count = counts_by_id.get(feature_id).copied().unwrap_or(0);
            entry.raw_indices = None;
            entry.series.positions_world = None;
            entry.series.values = None;
            entry.series.lod_levels = None;
            next.insert(key, entry);
        }
        self.feature_points = next;
    }

    fn apply_prepared_snapshot(&mut self, prepared: PreparedSpatialPoints) {
        self.raw_xy = Some(Arc::clone(&prepared.raw_xy));
        self.meta = Some(Arc::clone(&prepared.meta));
        self.positions_world = Some(Arc::clone(&prepared.positions_world));
        self.visible_raw_indices = None;
        self.values = Some(Arc::clone(&prepared.values));
        self.lod_levels = Some(Arc::clone(&prepared.lod_levels));
        self.last_auto_choice = prepared.last_auto_choice;
        self.feature_cache = prepared.feature_cache;
        self.sync_feature_points_with_counts(&prepared.feature_counts);

        if self.has_enabled_features() {
            self.apply_feature_selection();
        } else {
            self.last_match_count = prepared.loaded_count;
            self.hover_positions_world = Some(Arc::clone(&prepared.positions_world));
            self.hover_raw_indices = None;
            self.bounds_world = prepared.bounds_world;
            self.bins = prepared.bins;
        }
    }

    fn set_feature_enabled(&mut self, feature_name: &str, enabled: bool) {
        let key = normalize_feature_key(feature_name);
        if let Some(feature) = self.feature_points.get_mut(&key) {
            feature.series.enabled = enabled;
        }
    }

    fn set_hover_data(
        &mut self,
        positions_world: Option<Arc<Vec<egui::Pos2>>>,
        raw_indices: Option<Arc<Vec<u32>>>,
    ) {
        self.hover_positions_world = positions_world;
        self.hover_raw_indices = raw_indices;
        if let Some(points) = self.hover_positions_world.as_ref() {
            self.bounds_world = bounds_of_points(points.as_ref());
            self.bins = PointIndexBins::build(points.as_ref(), 256.0).map(Arc::new);
        } else {
            self.bounds_world = None;
            self.bins = None;
        }
    }

    fn rebuild_feature_cache(&mut self) {
        let Some(feature_meta) = self.meta.as_ref().and_then(|m| m.feature.as_ref()) else {
            self.feature_cache.clear();
            self.clear_feature_draw_data();
            return;
        };
        let Some(all_positions) = self.positions_world.as_ref() else {
            self.feature_cache.clear();
            self.clear_feature_draw_data();
            return;
        };

        let mut positions_by_id = vec![Vec::<egui::Pos2>::new(); feature_meta.dict.len()];
        let mut raw_indices_by_id = vec![Vec::<u32>::new(); feature_meta.dict.len()];
        for (raw_i, &feature_id) in feature_meta.ids.iter().enumerate() {
            let feature_i = feature_id as usize;
            let Some(bucket) = positions_by_id.get_mut(feature_i) else {
                continue;
            };
            let Some(raw_bucket) = raw_indices_by_id.get_mut(feature_i) else {
                continue;
            };
            let Some(&p) = all_positions.get(raw_i) else {
                continue;
            };
            bucket.push(p);
            raw_bucket.push(raw_i as u32);
        }

        self.feature_cache = positions_by_id
            .into_iter()
            .zip(raw_indices_by_id)
            .map(|(positions_world, raw_indices)| {
                if positions_world.is_empty() {
                    return None;
                }
                let values = vec![1.0f32; positions_world.len()];
                Some(Arc::new(SpatialFeatureCache {
                    positions_world: Arc::new(positions_world),
                    raw_indices: Arc::new(raw_indices),
                    values: Arc::new(values),
                    lod_levels: Arc::new(Vec::new()),
                }))
            })
            .collect();
    }

    fn apply_feature_selection(&mut self) {
        self.clear_feature_draw_data();

        if self.has_enabled_features() {
            let mut hover_positions = Vec::new();
            let mut hover_raw_indices = Vec::new();
            for feature in self.feature_points.values_mut() {
                if !feature.series.enabled {
                    continue;
                }
                let Some(cache) = self
                    .feature_cache
                    .get(feature.feature_id as usize)
                    .and_then(|cache| cache.as_ref())
                else {
                    continue;
                };
                feature.series.set_payload(
                    Arc::clone(&cache.positions_world),
                    Arc::clone(&cache.values),
                    None,
                );
                feature.raw_indices = Some(Arc::clone(&cache.raw_indices));
                hover_positions.extend(cache.positions_world.iter().copied());
                hover_raw_indices.extend(cache.raw_indices.iter().copied());
            }
            self.last_match_count = hover_positions.len();
            if hover_positions.is_empty() {
                self.set_hover_data(None, None);
            } else {
                self.set_hover_data(
                    Some(Arc::new(hover_positions)),
                    Some(Arc::new(hover_raw_indices)),
                );
            }
            return;
        }

        self.last_match_count = self.positions_world.as_ref().map_or(0, |p| p.len());
        self.set_hover_data(self.positions_world.clone(), None);
    }

    fn rebuild_draw_data(&mut self) {
        let (Some(raw), Some(meta)) = (self.raw_xy.as_ref(), self.meta.as_ref()) else {
            self.positions_world = None;
            self.values = None;
            self.lod_levels = None;
            self.bounds_world = None;
            self.last_match_count = 0;
            self.last_auto_choice.clear();
            self.bins = None;
            self.visible_raw_indices = None;
            self.feature_cache.clear();
            self.clear_feature_draw_data();
            self.hover_positions_world = None;
            self.hover_raw_indices = None;
            return;
        };
        match prepare_spatial_points_from_parts(
            Arc::clone(raw),
            Arc::clone(meta),
            self.prepare_config(),
        ) {
            Ok(prepared) => self.apply_prepared_snapshot(prepared),
            Err(err) => {
                self.status = format!("Prepare failed: {err}");
                self.positions_world = None;
                self.values = None;
                self.lod_levels = None;
                self.bounds_world = None;
                self.last_match_count = 0;
                self.last_auto_choice.clear();
                self.bins = None;
                self.visible_raw_indices = None;
                self.feature_cache.clear();
                self.clear_feature_draw_data();
                self.hover_positions_world = None;
                self.hover_raw_indices = None;
            }
        }
    }

    pub fn hover_point_index(
        &self,
        pointer_world: egui::Pos2,
        local_to_world_offset: egui::Vec2,
        camera: &crate::camera::Camera,
    ) -> Option<usize> {
        let (Some(points), Some(bins)) = (self.hover_positions_world.as_ref(), self.bins.as_ref())
        else {
            return None;
        };
        if points.is_empty() {
            return None;
        }

        // Convert to local coords (same space as `positions_world`).
        let pointer = pointer_world - local_to_world_offset;

        // Pick radius in world coords.
        let zoom = camera.zoom_screen_per_lvl0_px.max(1e-6);
        let radius_px = (self.style.radius_screen_px.max(1.0) * zoom.sqrt()).clamp(2.0, 30.0);
        let radius_world = (radius_px / zoom).max(1.0);

        let query = egui::Rect::from_center_size(
            pointer,
            egui::vec2(radius_world * 2.0, radius_world * 2.0),
        );
        let (x0, y0, x1, y1) = bins.bin_range_for_world_rect(query);

        let mut best_i: Option<usize> = None;
        let mut best_d2 = radius_world * radius_world;

        // Search bins that intersect the query rect.
        for by in y0..=y1 {
            for bx in x0..=x1 {
                let bi = by * bins.bins_w + bx;
                for &pi_u32 in bins.bin_slice(bi) {
                    let pi = pi_u32 as usize;
                    if pi >= points.len() {
                        continue;
                    }
                    let p = points[pi];
                    let dx = p.x - pointer.x;
                    let dy = p.y - pointer.y;
                    let d2 = dx * dx + dy * dy;
                    if d2 <= best_d2 {
                        best_d2 = d2;
                        best_i = Some(pi);
                    }
                }
            }
        }
        best_i
    }

    pub fn hover_tooltip(
        &self,
        pointer_world: egui::Pos2,
        local_to_world_offset: egui::Vec2,
        camera: &crate::camera::Camera,
    ) -> Option<Vec<String>> {
        let idx = self.hover_point_index(pointer_world, local_to_world_offset, camera)?;
        let points = self.hover_positions_world.as_ref()?;
        let p = points.get(idx).copied()?;
        let raw_i = self
            .hover_raw_indices
            .as_ref()
            .and_then(|m| m.get(idx))
            .copied()
            .unwrap_or(idx as u32) as usize;
        let meta = self.meta.as_ref();
        let world = p + local_to_world_offset;

        let mut lines = Vec::new();
        if let Some(f) = meta.as_ref().and_then(|m| m.feature.as_ref()) {
            if let Some(id) = f.ids.get(raw_i).copied() {
                if let Some(name) = f.dict.get(id as usize) {
                    if !name.trim().is_empty() {
                        lines.push(format!("feature: {name}"));
                    }
                }
            }
        }
        if let Some(v) = meta
            .as_ref()
            .and_then(|m| m.cell_id.as_ref())
            .and_then(|v| v.get(raw_i))
        {
            if *v >= 0 {
                lines.push(format!("cell_id: {v}"));
            }
        }
        if let Some(v) = meta
            .as_ref()
            .and_then(|m| m.qv.as_ref())
            .and_then(|v| v.get(raw_i))
        {
            if v.is_finite() {
                lines.push(format!("qv: {:.3}", v));
            }
        }
        if let Some(v) = meta
            .as_ref()
            .and_then(|m| m.transcript_id.as_ref())
            .and_then(|v| v.get(raw_i))
        {
            if *v != 0 {
                lines.push(format!("transcript_id: {v}"));
            }
        }
        if let Some(v) = meta
            .as_ref()
            .and_then(|m| m.overlaps_nucleus.as_ref())
            .and_then(|v| v.get(raw_i))
        {
            lines.push(format!(
                "overlaps_nucleus: {}",
                if *v != 0 { "yes" } else { "no" }
            ));
        }
        if let Some(v) = meta
            .as_ref()
            .and_then(|m| m.z.as_ref())
            .and_then(|v| v.get(raw_i))
        {
            if v.is_finite() {
                lines.push(format!("z: {:.2}", v));
            }
        }
        lines.push(format!("x: {:.2}", world.x));
        lines.push(format!("y: {:.2}", world.y));
        if let Some(raw) = self.raw_xy.as_ref().and_then(|r| r.get(raw_i)) {
            lines.push(format!("raw_x: {:.2}", raw[0]));
            lines.push(format!("raw_y: {:.2}", raw[1]));
        }
        Some(lines)
    }

    pub fn draw(
        &self,
        ui: &mut egui::Ui,
        viewport: egui::Rect,
        camera: &crate::camera::Camera,
        local_to_world_offset: egui::Vec2,
        gpu_available: bool,
    ) {
        if !self.visible {
            return;
        }

        // When feature metadata is present, points are rendered as named feature
        // series so they can share LOD selection, color assignment, and picker
        // behavior with Xenium and other feature overlay sources.
        let enabled_features = self
            .feature_points
            .values()
            .filter(|feature| feature.series.enabled)
            .count()
            .max(1);
        let per_feature_render_budget = if self.max_render_points_total == 0 {
            None
        } else {
            Some((self.max_render_points_total / enabled_features).max(1))
        };

        if gpu_available {
            if self.has_enabled_features() {
                for feature in self.feature_points.values() {
                    feature.series.draw(
                        ui,
                        viewport,
                        camera,
                        local_to_world_offset,
                        self.visible,
                        gpu_available,
                        self.threshold,
                        &self.style,
                        per_feature_render_budget,
                    );
                }
            } else {
                let (Some(positions_world), Some(values)) = (&self.positions_world, &self.values)
                else {
                    return;
                };
                if positions_world.is_empty() || values.is_empty() {
                    return;
                }
                let (generation, draw_positions, draw_values) = select_draw_payload(
                    self.generation,
                    positions_world,
                    values,
                    self.lod_levels.as_ref().map(|v| v.as_slice()),
                    camera.zoom_screen_per_lvl0_px,
                    if self.max_render_points_total == 0 {
                        None
                    } else {
                        Some(self.max_render_points_total)
                    },
                );
                let data = PointsGlDrawData {
                    generation,
                    positions_world: draw_positions,
                    values: draw_values,
                };
                let params = PointsGlDrawParams {
                    center_world: camera.center_world_lvl0,
                    zoom_screen_per_world: camera.zoom_screen_per_lvl0_px,
                    threshold: self.threshold,
                    style: self.style.clone(),
                    visible: self.visible,
                    local_to_world_offset,
                    local_to_world_scale: egui::vec2(1.0, 1.0),
                };
                let renderer = self.gl.clone();
                let cb = egui_glow::CallbackFn::new(move |info, painter| {
                    renderer.paint(info, painter, &data, &params);
                });
                ui.painter().add(egui::PaintCallback {
                    rect: viewport,
                    callback: Arc::new(cb),
                });
            }
        } else {
            let world_to_screen =
                |p: egui::Pos2| camera.world_to_screen(p + local_to_world_offset, viewport);
            let visible_world =
                screen_rect_to_world(camera, viewport).translate(-local_to_world_offset);
            let zoom = camera.zoom_screen_per_lvl0_px;
            let radius_px = (self.style.radius_screen_px.max(1.0) * zoom.sqrt()).clamp(0.75, 40.0);
            if self.has_enabled_features() {
                for feature in self.feature_points.values() {
                    if !feature.series.enabled {
                        continue;
                    }
                    let Some(positions_world) = feature.series.positions_world.as_ref() else {
                        continue;
                    };
                    let color = egui::Color32::from_rgba_unmultiplied(
                        feature.series.color_rgb[0],
                        feature.series.color_rgb[1],
                        feature.series.color_rgb[2],
                        230,
                    );
                    for p in positions_world.iter() {
                        if !visible_world.contains(*p) {
                            continue;
                        }
                        let s = world_to_screen(*p);
                        ui.painter().circle_filled(s, radius_px, color);
                    }
                }
            } else if let Some(positions_world) = self.positions_world.as_ref() {
                for p in positions_world.iter() {
                    if !visible_world.contains(*p) {
                        continue;
                    }
                    let s = world_to_screen(*p);
                    ui.painter()
                        .circle_filled(s, radius_px, self.style.fill_positive);
                }
            }
        }
    }

    pub fn ui_properties(
        &mut self,
        ui: &mut egui::Ui,
        positive_targets: &[(PositiveCellSelectionTarget, String)],
    ) -> bool {
        let mut changed = false;
        changed |= ui.checkbox(&mut self.visible, "Visible").changed();

        ui.separator();
        if !self.feature_points.is_empty() {
            let mut items: Vec<FeaturePickerItem> = self
                .feature_points
                .values()
                .map(|feature| FeaturePickerItem {
                    name: feature.series.feature_name.clone(),
                    enabled: feature.series.enabled,
                    color_rgb: feature.series.color_rgb,
                    status: Some(format!("{} points", feature.series.point_count)),
                })
                .collect();
            items.sort_by(|a, b| a.name.cmp(&b.name));
            let picker = show_feature_picker(
                ui,
                "spatialdata_feature_picker",
                "Features",
                "SpatialData features",
                &mut self.feature_query,
                &mut self.feature_popup_open,
                &items,
            );
            if !picker.toggles.is_empty() {
                for (feature, on) in picker.toggles {
                    self.set_feature_enabled(feature.as_str(), on);
                }
                self.apply_feature_selection();
                changed = true;
            }
        }
        let loaded = self.last_loaded_count;
        let showing = self.last_match_count;
        if loaded > 0 {
            ui.label(format!(
                "Showing {showing} / {loaded}  render cap: inactive"
            ));
        } else {
            ui.label(format!("Showing {showing}  render cap: inactive"));
        }
        ui.separator();
        ui.label("SpatialData point LOD is disabled.");
        ui.separator();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.style.radius_screen_px, 0.5..=20.0)
                    .text("Size")
                    .show_value(true)
                    .clamping(egui::SliderClamping::Always),
            )
            .changed();

        ui.horizontal(|ui| {
            ui.label("Fill");
            changed |= ui
                .color_edit_button_srgba(&mut self.style.fill_positive)
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("Stroke");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.style.stroke_positive.width)
                        .speed(0.25)
                        .clamp_range(0.0..=10.0),
                )
                .changed();
            changed |= ui
                .color_edit_button_srgba(&mut self.style.stroke_positive.color)
                .changed();
        });

        ui.separator();
        ui.label("Transform");
        let mut transform_changed = false;
        egui::ComboBox::from_id_salt("spatial_points_axis_mode")
            .selected_text(match self.axis_mode {
                SpatialAxisMode::Auto => "Axes: Auto",
                SpatialAxisMode::XY => "Axes: x,y",
                SpatialAxisMode::YX => "Axes: y,x",
            })
            .show_ui(ui, |ui| {
                transform_changed |= ui
                    .selectable_value(&mut self.axis_mode, SpatialAxisMode::Auto, "Axes: Auto")
                    .changed();
                transform_changed |= ui
                    .selectable_value(&mut self.axis_mode, SpatialAxisMode::XY, "Axes: x,y")
                    .changed();
                transform_changed |= ui
                    .selectable_value(&mut self.axis_mode, SpatialAxisMode::YX, "Axes: y,x")
                    .changed();
            });
        egui::ComboBox::from_id_salt("spatial_points_scale_mode")
            .selected_text(match self.scale_mode {
                SpatialScaleMode::Auto => "Auto",
                SpatialScaleMode::UseScale => "Use scale",
                SpatialScaleMode::InvertScale => "Invert scale",
                SpatialScaleMode::Identity => "Identity",
            })
            .show_ui(ui, |ui| {
                transform_changed |= ui
                    .selectable_value(&mut self.scale_mode, SpatialScaleMode::Auto, "Auto")
                    .changed();
                transform_changed |= ui
                    .selectable_value(
                        &mut self.scale_mode,
                        SpatialScaleMode::UseScale,
                        "Use scale",
                    )
                    .changed();
                transform_changed |= ui
                    .selectable_value(
                        &mut self.scale_mode,
                        SpatialScaleMode::InvertScale,
                        "Invert scale",
                    )
                    .changed();
                transform_changed |= ui
                    .selectable_value(&mut self.scale_mode, SpatialScaleMode::Identity, "Identity")
                    .changed();
            });
        if transform_changed {
            self.rebuild_draw_data();
            self.generation = self.generation.wrapping_add(1).max(1);
            changed = true;
        }
        if ui
            .add(
                egui::DragValue::new(&mut self.scale_mul)
                    .speed(0.01)
                    .clamp_range(0.0001..=10_000.0)
                    .prefix("Scale × "),
            )
            .changed()
        {
            self.rebuild_draw_data();
            self.generation = self.generation.wrapping_add(1).max(1);
            changed = true;
        }
        if (self.scale_mode == SpatialScaleMode::Auto || self.axis_mode == SpatialAxisMode::Auto)
            && !self.last_auto_choice.is_empty()
        {
            ui.label(format!("Auto: {}", self.last_auto_choice));
        }

        ui.separator();
        ui.label("Cell selection");
        self.ensure_positive_cell_target_initialized(positive_targets);
        egui::ComboBox::from_id_salt("spatial_points_positive_cell_target")
            .selected_text(match self.positive_cell_target {
                PositiveCellSelectionTarget::SegmentationObjects => positive_targets
                    .iter()
                    .find(|(target, _)| *target == PositiveCellSelectionTarget::SegmentationObjects)
                    .map(|(_, name)| name.clone())
                    .unwrap_or_else(|| "Segmentation Objects".to_string()),
                PositiveCellSelectionTarget::AllObjectLayers => "All object layers".to_string(),
                PositiveCellSelectionTarget::ShapeLayer(id) => positive_targets
                    .iter()
                    .find(|(target, _)| *target == PositiveCellSelectionTarget::ShapeLayer(id))
                    .map(|(_, name)| name.clone())
                    .unwrap_or_else(|| format!("Layer {id}")),
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut self.positive_cell_target,
                    PositiveCellSelectionTarget::AllObjectLayers,
                    "All object layers",
                );
                for (target, name) in positive_targets {
                    ui.selectable_value(&mut self.positive_cell_target, *target, name);
                }
            });
        ui.horizontal(|ui| {
            ui.label("Min transcripts / cell");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.positive_cell_min_count)
                        .speed(1)
                        .range(1..=1_000_000),
                )
                .changed();
        });
        let can_select_positive = self
            .meta
            .as_ref()
            .and_then(|m| m.cell_id.as_ref())
            .is_some()
            && self
                .meta
                .as_ref()
                .and_then(|m| m.feature.as_ref())
                .is_some()
            && self
                .feature_points
                .values()
                .any(|feature| feature.series.enabled);
        if ui
            .add_enabled(
                can_select_positive,
                egui::Button::new("Select positive cells"),
            )
            .clicked()
        {
            match self.compute_positive_cell_ids() {
                Ok(cell_ids) => {
                    let cell_count = cell_ids.len();
                    self.pending_positive_cell_selection = Some(PositiveCellSelectionRequest {
                        cell_ids,
                        target: self.positive_cell_target,
                    });
                    self.cell_selection_status =
                        format!("Queued selection for {cell_count} positive cell(s).");
                }
                Err(err) => {
                    self.pending_positive_cell_selection = None;
                    self.cell_selection_status = err;
                }
            }
        }
        if !can_select_positive {
            ui.label("Enable one or more transcript features to select positive cells.");
        }
        if !self.cell_selection_status.is_empty() {
            ui.label(self.cell_selection_status.clone());
        }
        ui.separator();
        ui.horizontal(|ui| {
            let mut all = self.options.max_points == 0;
            if ui
                .checkbox(&mut all, "All")
                .on_hover_text("Load all points (may be slow / memory-heavy).")
                .changed()
            {
                self.options.max_points = if all { 0 } else { 200_000 };
                changed = true;
            }
            if !all {
                ui.add(
                    egui::DragValue::new(&mut self.options.max_points)
                        .speed(1)
                        .clamp_range(1..=200_000_000)
                        .prefix("Max points "),
                )
                .on_hover_text("Reload to apply.");
            } else {
                ui.label("Max points: ∞");
            }
            if ui.button("Reload points").clicked() {
                self.raw_xy = None;
                self.meta = None;
                self.positions_world = None;
                self.visible_raw_indices = None;
                self.values = None;
                self.feature_points.clear();
                self.feature_cache.clear();
                self.hover_positions_world = None;
                self.hover_raw_indices = None;
                self.bounds_world = None;
                self.bins = None;
                self.request_load();
                changed = true;
            }
        });

        if !self.status.is_empty() {
            ui.label(self.status.clone());
        }
        changed
    }

    fn compute_positive_cell_ids(&self) -> Result<Vec<String>, String> {
        // "Positive" is defined against the currently enabled features only. Hidden
        // features are ignored so the exported selection mirrors what the user sees.
        let meta = self
            .meta
            .as_ref()
            .ok_or_else(|| "Points metadata is not loaded.".to_string())?;
        let cell_ids = meta
            .cell_id
            .as_ref()
            .ok_or_else(|| "This points layer has no cell_id column.".to_string())?;
        let feature = meta
            .feature
            .as_ref()
            .ok_or_else(|| "This points layer has no feature metadata.".to_string())?;

        let enabled_feature_ids: HashSet<u32> = self
            .feature_points
            .values()
            .filter(|entry| entry.series.enabled)
            .map(|entry| entry.feature_id)
            .collect();
        if enabled_feature_ids.is_empty() {
            return Err("No transcript features are enabled.".to_string());
        }

        let mut counts_by_cell: HashMap<i32, usize> = HashMap::new();
        for (row_i, &feature_id) in feature.ids.iter().enumerate() {
            if !enabled_feature_ids.contains(&feature_id) {
                continue;
            }
            let Some(&cell_id) = cell_ids.get(row_i) else {
                continue;
            };
            if cell_id <= 0 {
                continue;
            }
            *counts_by_cell.entry(cell_id).or_default() += 1;
        }

        let min_count = self.positive_cell_min_count.max(1);
        let mut positive = counts_by_cell
            .into_iter()
            .filter_map(|(cell_id, count)| (count >= min_count).then(|| cell_id.to_string()))
            .collect::<Vec<_>>();
        positive.sort();
        positive.dedup();
        if positive.is_empty() {
            return Err("No positive cells found in the currently loaded points.".to_string());
        }
        Ok(positive)
    }
}

fn prepare_spatial_points_payload(
    payload: crate::spatialdata::PointsPayload,
    config: SpatialPointsPrepareConfig,
) -> anyhow::Result<PreparedSpatialPoints> {
    let raw_xy = Arc::new(payload.xy);
    let meta = Arc::new(payload.meta);
    prepare_spatial_points_from_parts(raw_xy, meta, config)
}

fn prepare_spatial_points_from_parts(
    raw_xy: Arc<Vec<[f32; 2]>>,
    meta: Arc<PointsMeta>,
    config: SpatialPointsPrepareConfig,
) -> anyhow::Result<PreparedSpatialPoints> {
    // Normalize the raw parquet payload into viewer-native structures once so the
    // draw path can stay format-agnostic: world coordinates, feature counts, bins,
    // and LOD payloads are all derived here rather than during every frame.
    let raw_len = raw_xy.len();
    let mut feature_counts = meta
        .feature
        .as_ref()
        .map(|feature| vec![0usize; feature.dict.len()])
        .unwrap_or_default();
    if let Some(feature) = meta.feature.as_ref() {
        for &id in &feature.ids {
            if let Some(count) = feature_counts.get_mut(id as usize) {
                *count += 1;
            }
        }
    }

    if raw_len == 0 {
        return Ok(PreparedSpatialPoints {
            raw_xy,
            meta,
            positions_world: Arc::new(Vec::new()),
            values: Arc::new(Vec::new()),
            lod_levels: Arc::new(Vec::new()),
            feature_counts,
            feature_cache: Vec::new(),
            bounds_world: None,
            bins: None,
            last_auto_choice: String::new(),
            loaded_count: 0,
        });
    }

    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for p in raw_xy.iter() {
        min_x = min_x.min(p[0]);
        min_y = min_y.min(p[1]);
        max_x = max_x.max(p[0]);
        max_y = max_y.max(p[1]);
    }
    let raw_w = (max_x - min_x).abs().max(1e-6);
    let raw_h = (max_y - min_y).abs().max(1e-6);

    let base = config.base_transform.scale;
    let inv = [
        if base[0].abs() > 1e-12 {
            1.0 / base[0]
        } else {
            1.0
        },
        if base[1].abs() > 1e-12 {
            1.0 / base[1]
        } else {
            1.0
        },
    ];
    let identity = [1.0f32, 1.0f32];
    let tr = config.base_transform.translation;

    let scale_candidates: [(&str, [f32; 2]); 3] = [("scale", base), ("inv", inv), ("id", identity)];
    let axis_candidates: [(&str, SpatialAxisMode); 2] =
        [("xy", SpatialAxisMode::XY), ("yx", SpatialAxisMode::YX)];

    let want_scales: Vec<(&str, [f32; 2])> = match config.scale_mode {
        SpatialScaleMode::Identity => vec![("id", identity)],
        SpatialScaleMode::UseScale => vec![("scale", base)],
        SpatialScaleMode::InvertScale => vec![("inv", inv)],
        SpatialScaleMode::Auto => scale_candidates.to_vec(),
    };
    let want_axes: Vec<(&str, SpatialAxisMode)> = match config.axis_mode {
        SpatialAxisMode::XY => vec![("xy", SpatialAxisMode::XY)],
        SpatialAxisMode::YX => vec![("yx", SpatialAxisMode::YX)],
        SpatialAxisMode::Auto => axis_candidates.to_vec(),
    };

    let (mut pick_scale_name, mut pick_scale) = want_scales[0];
    let (mut pick_axis_name, mut pick_axis) = want_axes[0];
    let mut best_score = f32::INFINITY;

    if let Some(img) = config.image_size_world {
        let img_w = img[0].max(1.0);
        let img_h = img[1].max(1.0);
        for (sname, s0) in &want_scales {
            for (aname, a0) in &want_axes {
                let sx = s0[0].abs().max(1e-12);
                let sy = s0[1].abs().max(1e-12);
                let (w, h, min_mapped_x, min_mapped_y, max_mapped_x, max_mapped_y) = match a0 {
                    SpatialAxisMode::XY => {
                        let w = raw_w * sx;
                        let h = raw_h * sy;
                        let minx = min_x * s0[0] + tr[0];
                        let miny = min_y * s0[1] + tr[1];
                        let maxx = max_x * s0[0] + tr[0];
                        let maxy = max_y * s0[1] + tr[1];
                        (
                            w,
                            h,
                            minx.min(maxx),
                            miny.min(maxy),
                            minx.max(maxx),
                            miny.max(maxy),
                        )
                    }
                    SpatialAxisMode::YX => {
                        let w = raw_h * sx;
                        let h = raw_w * sy;
                        let minx = min_y * s0[0] + tr[0];
                        let miny = min_x * s0[1] + tr[1];
                        let maxx = max_y * s0[0] + tr[0];
                        let maxy = max_x * s0[1] + tr[1];
                        (
                            w,
                            h,
                            minx.min(maxx),
                            miny.min(maxy),
                            minx.max(maxx),
                            miny.max(maxy),
                        )
                    }
                    SpatialAxisMode::Auto => unreachable!("auto resolved above"),
                };

                let size_score = (w / img_w).ln().abs() + (h / img_h).ln().abs();
                let off_left = (-min_mapped_x).max(0.0);
                let off_top = (-min_mapped_y).max(0.0);
                let off_right = (max_mapped_x - img_w).max(0.0);
                let off_bottom = (max_mapped_y - img_h).max(0.0);
                let outside_score = (off_left + off_right) / img_w + (off_top + off_bottom) / img_h;
                let origin_score = (min_mapped_x / img_w).abs() + (min_mapped_y / img_h).abs();
                let score = size_score + 0.35 * outside_score + 0.05 * origin_score;
                if score < best_score {
                    best_score = score;
                    pick_scale_name = sname;
                    pick_scale = *s0;
                    pick_axis_name = aname;
                    pick_axis = *a0;
                }
            }
        }
    }

    let last_auto_choice = if config.scale_mode == SpatialScaleMode::Auto
        || config.axis_mode == SpatialAxisMode::Auto
    {
        format!("{pick_scale_name} + {pick_axis_name}")
    } else {
        String::new()
    };

    let s = [
        pick_scale[0] * config.scale_mul,
        pick_scale[1] * config.scale_mul,
    ];

    let mut pos: Vec<egui::Pos2> = Vec::with_capacity(raw_len);
    for p in raw_xy.iter() {
        let (in_x, in_y) = match pick_axis {
            SpatialAxisMode::XY => (p[0], p[1]),
            SpatialAxisMode::YX => (p[1], p[0]),
            SpatialAxisMode::Auto => (p[0], p[1]),
        };
        let x = in_x * s[0] + tr[0];
        let y = in_y * s[1] + tr[1];
        pos.push(egui::pos2(x, y));
    }

    let positions_world = Arc::new(pos);
    let values = Arc::new(vec![1.0f32; raw_len]);
    let lod_levels = Arc::new(Vec::new());

    let feature_cache = if let Some(feature_meta) = meta.feature.as_ref() {
        let mut positions_by_id = vec![Vec::<egui::Pos2>::new(); feature_meta.dict.len()];
        let mut raw_indices_by_id = vec![Vec::<u32>::new(); feature_meta.dict.len()];
        for (raw_i, &feature_id) in feature_meta.ids.iter().enumerate() {
            let feature_i = feature_id as usize;
            let Some(bucket) = positions_by_id.get_mut(feature_i) else {
                continue;
            };
            let Some(raw_bucket) = raw_indices_by_id.get_mut(feature_i) else {
                continue;
            };
            let Some(&p) = positions_world.get(raw_i) else {
                continue;
            };
            bucket.push(p);
            raw_bucket.push(raw_i as u32);
        }

        positions_by_id
            .into_iter()
            .zip(raw_indices_by_id)
            .map(|(positions_world, raw_indices)| {
                if positions_world.is_empty() {
                    return None;
                }
                let values = vec![1.0f32; positions_world.len()];
                Some(Arc::new(SpatialFeatureCache {
                    positions_world: Arc::new(positions_world),
                    raw_indices: Arc::new(raw_indices),
                    values: Arc::new(values),
                    lod_levels: Arc::new(Vec::new()),
                }))
            })
            .collect()
    } else {
        Vec::new()
    };

    let bounds_world = bounds_of_points(positions_world.as_ref());
    let bins = PointIndexBins::build(positions_world.as_ref(), 256.0).map(Arc::new);

    Ok(PreparedSpatialPoints {
        raw_xy,
        meta,
        positions_world,
        values,
        lod_levels,
        feature_counts,
        feature_cache,
        bounds_world,
        bins,
        last_auto_choice,
        loaded_count: raw_len,
    })
}

fn screen_rect_to_world(camera: &crate::camera::Camera, viewport: egui::Rect) -> egui::Rect {
    let p0 = camera.screen_to_world(viewport.left_top(), viewport);
    let p1 = camera.screen_to_world(viewport.right_bottom(), viewport);
    egui::Rect::from_min_max(
        egui::pos2(p0.x.min(p1.x), p0.y.min(p1.y)),
        egui::pos2(p0.x.max(p1.x), p0.y.max(p1.y)),
    )
}

fn bounds_of_points(points: &[egui::Pos2]) -> Option<egui::Rect> {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut any = false;
    for p in points {
        if !(p.x.is_finite() && p.y.is_finite()) {
            continue;
        }
        any = true;
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
    }
    if !any {
        return None;
    }
    Some(egui::Rect::from_min_max(
        egui::pos2(min_x, min_y),
        egui::pos2(max_x, max_y),
    ))
}
