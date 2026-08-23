use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use eframe::egui;

use crate::features::points::{
    FeaturePickerItem, FeaturePointLod, FeaturePointSeries, color_for_feature,
    normalize_feature_key, select_draw_payload, show_feature_picker,
};
use crate::render::points::PointsStyle;
use crate::render::points_gl::{PointsGlDrawData, PointsGlDrawParams, PointsGlRenderer};
use crate::spatialdata::{
    PointsLoadOptions, PointsMeta, SpatialDataTransform2, load_points_sample,
};
use odon::data::point_bins::PointIndexBins;

mod prepare;

use super::PositiveCellSelectionTarget;
use prepare::{
    bounds_of_points, prepare_spatial_points_from_parts, prepare_spatial_points_payload,
};

#[derive(Debug)]
pub struct SpatialPointsLayer {
    pub name: String,
    pub visible: bool,
    pub style: PointsStyle,
    pub threshold: f32,
    pub max_render_points_total: usize,

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

#[derive(Debug, Clone)]
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

#[derive(Clone)]
pub(crate) struct PreparedSpatialPointsLayer {
    name: String,
    base_transform: SpatialDataTransform2,
    image_size_world: Option<[f32; 2]>,
    prepared: PreparedSpatialPoints,
}

pub(crate) fn prepare_spatial_points_layer(
    name: String,
    points_parquet_dir: PathBuf,
    transform: SpatialDataTransform2,
    feature_key: Option<String>,
    max_points: usize,
    image_size_world: Option<[f32; 2]>,
) -> anyhow::Result<PreparedSpatialPointsLayer> {
    let mut options = PointsLoadOptions {
        max_points,
        ..Default::default()
    };
    if let Some(key) = feature_key.filter(|key| !key.trim().is_empty()) {
        options.feature_column = Some(key);
    }
    let config = SpatialPointsPrepareConfig {
        base_transform: transform,
        image_size_world,
        scale_mode: SpatialScaleMode::Auto,
        axis_mode: SpatialAxisMode::Auto,
        scale_mul: 1.0,
    };
    let prepared = load_points_sample(&points_parquet_dir, &options)
        .and_then(|payload| prepare_spatial_points_payload(payload, config))?;
    Ok(PreparedSpatialPointsLayer {
        name,
        base_transform: transform,
        image_size_world,
        prepared,
    })
}

impl SpatialPointsLayer {
    pub(crate) fn from_prepared(prepared: PreparedSpatialPointsLayer) -> Self {
        let PreparedSpatialPointsLayer {
            name,
            base_transform,
            image_size_world,
            prepared,
        } = prepared;
        let loaded_count = prepared.loaded_count;
        let mut layer = Self {
            name,
            visible: true,
            style: PointsStyle::default(),
            threshold: 0.5,
            max_render_points_total: 200_000,
            base_transform,
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
            last_loaded_count: loaded_count,
            last_auto_choice: String::new(),
            generation: 2,
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
            status: format!("Loaded {loaded_count} points (sample)."),
            gl: PointsGlRenderer::default(),
        };
        layer.apply_prepared_snapshot(prepared);
        layer
    }

    pub fn is_loading(&self) -> bool {
        false
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
                        .range(0.0..=10.0),
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
                    .range(0.0001..=10_000.0)
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
        ui.label(format!("Actor-prepared points: {}", self.last_loaded_count));

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

fn screen_rect_to_world(camera: &crate::camera::Camera, viewport: egui::Rect) -> egui::Rect {
    let p0 = camera.screen_to_world(viewport.left_top(), viewport);
    let p1 = camera.screen_to_world(viewport.right_bottom(), viewport);
    egui::Rect::from_min_max(
        egui::pos2(p0.x.min(p1.x), p0.y.min(p1.y)),
        egui::pos2(p0.x.max(p1.x), p0.y.max(p1.y)),
    )
}
