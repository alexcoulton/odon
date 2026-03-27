use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crossbeam_channel::Receiver;
use eframe::egui;

use crate::features::points::{
    FeaturePickerItem, FeaturePointLod, FeaturePointSeries, build_point_lods, color_for_feature,
    normalize_feature_key, show_feature_picker,
};
use crate::render::line_bins::LineSegmentsBins;
use crate::render::line_bins_gl::{LineBinsGlDrawData, LineBinsGlDrawParams, LineBinsGlRenderer};
use crate::render::point_bins::PointIndexBins;
use crate::render::points::PointsStyle;
use crate::xenium::cells::{XeniumPolygonSet, load_cells_outline_bins};
use crate::xenium::transcripts::{
    XeniumTranscriptsAllPayload, XeniumTranscriptsMeta, load_transcripts_all_points,
    load_transcripts_meta,
};

#[derive(Debug, Default)]
pub struct XeniumLayers {
    pub dataset_root: Option<PathBuf>,
    pub cells: Option<XeniumCellsLayer>,
    pub transcripts: Option<XeniumTranscriptsLayer>,
}

impl XeniumLayers {
    pub fn clear(&mut self) {
        self.dataset_root = None;
        self.cells = None;
        self.transcripts = None;
    }

    pub fn attach(
        &mut self,
        dataset_root: PathBuf,
        cells_zip: Option<PathBuf>,
        transcripts_zip: Option<PathBuf>,
        pixel_size_um: f32,
    ) {
        self.dataset_root = Some(dataset_root);
        self.cells = cells_zip.map(|p| XeniumCellsLayer::new(p, pixel_size_um));
        self.transcripts = transcripts_zip.map(|p| XeniumTranscriptsLayer::new(p, pixel_size_um));
    }

    pub fn tick(&mut self) {
        if let Some(c) = self.cells.as_mut() {
            c.tick();
        }
        if let Some(t) = self.transcripts.as_mut() {
            t.tick();
        }
    }
}

#[derive(Debug)]
pub struct XeniumCellsLayer {
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub width_screen_px: f32,
    pub color_rgb: [u8; 3],

    cells_zip: PathBuf,
    pixel_size_um: f32,

    bins: Option<Arc<LineSegmentsBins>>,
    generation: u64,
    gl: LineBinsGlRenderer,
    load_rx: Option<Receiver<anyhow::Result<Arc<LineSegmentsBins>>>>,
    status: String,
}

impl XeniumCellsLayer {
    pub fn new(cells_zip: PathBuf, pixel_size_um: f32) -> Self {
        let mut s = Self {
            name: "Cells (Xenium)".to_string(),
            visible: true,
            opacity: 0.75,
            width_screen_px: 1.0,
            color_rgb: [0, 255, 120],
            cells_zip,
            pixel_size_um,
            bins: None,
            generation: 1,
            gl: LineBinsGlRenderer::new(1024),
            load_rx: None,
            status: String::new(),
        };
        s.request_load();
        s
    }

    fn request_load(&mut self) {
        let (tx, rx) = crossbeam_channel::bounded::<anyhow::Result<Arc<LineSegmentsBins>>>(1);
        self.load_rx = Some(rx);
        self.status = "Loading Xenium cells...".to_string();

        let zip = self.cells_zip.clone();
        let pixel_size_um = self.pixel_size_um;
        std::thread::Builder::new()
            .name("xenium-cells-loader".to_string())
            .spawn(move || {
                let msg = load_cells_outline_bins(&zip, XeniumPolygonSet::Cell, pixel_size_um);
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
                        Ok(bins) => {
                            let segs = bins.segments.len();
                            self.bins = Some(bins);
                            self.generation = self.generation.wrapping_add(1).max(1);
                            self.status = format!("Loaded {segs} segments.");
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

    pub fn draw(
        &self,
        ui: &mut egui::Ui,
        camera: &crate::camera::Camera,
        viewport: egui::Rect,
        visible_world: egui::Rect,
        gpu_available: bool,
        local_to_world_offset: egui::Vec2,
    ) {
        let Some(bins) = self.bins.as_ref() else {
            return;
        };
        if !self.visible || bins.segments.is_empty() {
            return;
        }
        if gpu_available {
            let a = (self.opacity.clamp(0.0, 1.0) * 255.0).round() as u8;
            let c = self.color_rgb;
            let color = egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], a);
            let data = LineBinsGlDrawData {
                cache_id: 9001,
                generation: self.generation,
                bins: Arc::clone(bins),
            };
            let params = LineBinsGlDrawParams {
                center_world: camera.center_world_lvl0,
                zoom_screen_per_world: camera.zoom_screen_per_lvl0_px,
                color,
                width_points: self.width_screen_px.max(0.0),
                visible: self.visible,
                local_to_world_offset,
                local_to_world_scale: egui::vec2(1.0, 1.0),
            };
            let visible_local = visible_world.translate(-local_to_world_offset);
            let renderer = self.gl.clone();
            let cb = egui_glow::CallbackFn::new(move |info, painter| {
                renderer.paint(info, painter, &data, &params, visible_local);
            });
            ui.painter().add(egui::PaintCallback {
                rect: viewport,
                callback: Arc::new(cb),
            });
        }
    }

    pub fn ui_properties(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        changed |= ui.checkbox(&mut self.visible, "Visible").changed();
        ui.separator();
        changed |= ui
            .add(egui::Slider::new(&mut self.opacity, 0.0..=1.0).text("Opacity"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.width_screen_px, 0.25..=3.0).text("Width"))
            .changed();
        ui.separator();
        ui.label(self.status.clone());
        changed
    }
}

#[derive(Debug)]
pub struct XeniumTranscriptsLayer {
    pub name: String,
    pub visible: bool,
    pub style: PointsStyle,
    pub gene_query: String,
    pub max_points_total: usize,
    pub max_render_points_total: usize,

    gene_popup_open: bool,

    zip: PathBuf,
    pixel_size_um: f32,

    meta: Option<Arc<XeniumTranscriptsMeta>>,
    genes: HashMap<String, XeniumGenePoints>,

    preloaded: Option<Vec<Option<PreloadedGeneData>>>,
    preload_rx: Option<Receiver<anyhow::Result<XeniumTranscriptsAllPayload>>>,

    hover_refs: Vec<HoverRef>,
    hover_positions_world: Option<Arc<Vec<egui::Pos2>>>,
    hover_bins: Option<Arc<PointIndexBins>>,

    status: String,
}

#[derive(Debug, Clone)]
struct PreloadedGeneData {
    positions_world: Arc<Vec<egui::Pos2>>,
    values: Arc<Vec<f32>>,
    ids: Option<Arc<Vec<u64>>>,
    lod_levels: Arc<Vec<FeaturePointLod>>,
}

#[derive(Debug, Clone, Copy)]
struct HoverRef {
    gene_id: u16,
    point_index: u32,
}

#[derive(Debug)]
struct XeniumGenePoints {
    gene_id: u16,
    ids: Option<Arc<Vec<u64>>>,
    series: FeaturePointSeries,
}

impl XeniumTranscriptsLayer {
    pub fn new(zip: PathBuf, pixel_size_um: f32) -> Self {
        let meta = load_transcripts_meta(&zip).ok().map(Arc::new);
        let mut s = Self {
            name: "Transcripts (Xenium)".to_string(),
            visible: false,
            style: PointsStyle {
                // XeniumExplorer-like: small points, no stroke by default.
                radius_screen_px: 2.0,
                fill_positive: egui::Color32::from_rgba_unmultiplied(255, 255, 255, 230),
                fill_negative: egui::Color32::TRANSPARENT,
                stroke_positive: egui::Stroke::new(0.0, egui::Color32::TRANSPARENT),
                stroke_negative: egui::Stroke::new(0.0, egui::Color32::TRANSPARENT),
            },
            gene_query: String::new(),
            // 0 = no limit (load everything).
            max_points_total: 0,
            // Cap the number of points rendered at once by forcing a coarser LOD.
            max_render_points_total: 200_000,
            gene_popup_open: false,
            zip,
            pixel_size_um,
            meta,
            genes: HashMap::new(),
            preloaded: None,
            preload_rx: None,
            hover_refs: Vec::new(),
            hover_positions_world: None,
            hover_bins: None,
            status: String::new(),
        };
        s.request_preload_all();
        s
    }

    fn ensure_gene(&mut self, gene_name: &str) -> &mut XeniumGenePoints {
        let key = normalize_feature_key(gene_name);
        let gene_id = self.gene_id_for_name(gene_name).unwrap_or(u16::MAX);
        self.genes
            .entry(key.clone())
            .or_insert_with(|| XeniumGenePoints {
                gene_id,
                ids: None,
                series: FeaturePointSeries::new(
                    gene_name.to_string(),
                    color_for_feature(gene_name),
                ),
            })
    }

    fn gene_id_for_name(&self, gene_name: &str) -> Option<u16> {
        let meta = self.meta.as_ref()?;
        let key = normalize_feature_key(gene_name);
        if let Some(id) = meta.gene_indices.get(&key).copied() {
            return Some(id);
        }
        meta.gene_names
            .iter()
            .position(|g| normalize_feature_key(g) == key)
            .and_then(|i| u16::try_from(i).ok())
    }

    fn request_preload_all(&mut self) {
        if self.preloaded.is_some() || self.preload_rx.is_some() {
            return;
        }
        let Some(meta) = self.meta.as_ref().cloned() else {
            self.status = "No transcripts metadata.".to_string();
            return;
        };
        let zip = self.zip.clone();
        let pixel_size_um = self.pixel_size_um;
        let max_points_total = self.max_points_total;
        let (tx, rx) = crossbeam_channel::bounded::<anyhow::Result<XeniumTranscriptsAllPayload>>(1);
        self.preload_rx = Some(rx);
        self.status = "Preloading transcripts (all genes)...".to_string();
        std::thread::Builder::new()
            .name("xenium-transcripts-preload".to_string())
            .spawn(move || {
                let msg = load_transcripts_all_points(&zip, &meta, pixel_size_um, max_points_total);
                let _ = tx.send(msg);
            })
            .ok();
    }

    fn attach_preloaded_for_gene(
        preloaded: Option<&[Option<PreloadedGeneData>]>,
        g: &mut XeniumGenePoints,
    ) {
        let Some(preloaded) = preloaded else {
            g.series.status = "Waiting for preload...".to_string();
            return;
        };
        let Some(d) = preloaded.get(g.gene_id as usize).and_then(|d| d.as_ref()) else {
            g.series.status = "0 points.".to_string();
            return;
        };
        g.ids = d.ids.as_ref().map(Arc::clone);
        g.series.set_payload(
            Arc::clone(&d.positions_world),
            Arc::clone(&d.values),
            Some(Arc::clone(&d.lod_levels)),
        );
        g.series.status = format!("{} points.", d.positions_world.len());
    }

    fn set_gene_enabled(&mut self, gene_name: &str, enabled: bool) {
        let key = normalize_feature_key(gene_name);
        let already_loaded = {
            let g = self.ensure_gene(gene_name);
            g.series.enabled = enabled;
            g.series.positions_world.is_some()
        };
        if !enabled || already_loaded {
            return;
        }
        if self.preloaded.is_none() {
            self.request_preload_all();
        }
        let preloaded = self.preloaded.as_deref();
        if let Some(g) = self.genes.get_mut(&key) {
            Self::attach_preloaded_for_gene(preloaded, g);
        }
    }

    fn rebuild_hover_index(&mut self) {
        self.hover_refs.clear();
        let mut positions: Vec<egui::Pos2> = Vec::new();
        for g in self.genes.values() {
            if !g.series.enabled {
                continue;
            }
            let Some(pos) = g.series.positions_world.as_ref() else {
                continue;
            };
            for (i, p) in pos.iter().copied().enumerate() {
                positions.push(p);
                self.hover_refs.push(HoverRef {
                    gene_id: g.gene_id,
                    point_index: i as u32,
                });
            }
        }
        if positions.is_empty() {
            self.hover_positions_world = None;
            self.hover_bins = None;
            return;
        }
        self.hover_bins = PointIndexBins::build(&positions, 2048.0).map(Arc::new);
        self.hover_positions_world = Some(Arc::new(positions));
    }

    pub fn tick(&mut self) {
        use crossbeam_channel::TryRecvError;
        let Some(rx) = self.preload_rx.as_ref().cloned() else {
            return;
        };
        loop {
            match rx.try_recv() {
                Ok(msg) => {
                    self.preload_rx = None;
                    match msg {
                        Ok(payload) => {
                            let XeniumTranscriptsAllPayload {
                                positions_by_gene,
                                qv_by_gene,
                                id_by_gene,
                                total_points,
                            } = payload;
                            let mut preloaded: Vec<Option<PreloadedGeneData>> = Vec::new();
                            for ((pos, qv), ids) in positions_by_gene
                                .into_iter()
                                .zip(qv_by_gene.into_iter())
                                .zip(id_by_gene.into_iter())
                            {
                                if pos.is_empty() {
                                    preloaded.push(None);
                                    continue;
                                }
                                let ids = if ids.is_empty() {
                                    None
                                } else {
                                    Some(Arc::new(ids))
                                };
                                preloaded.push(Some(PreloadedGeneData {
                                    lod_levels: Arc::new(build_point_lods(&pos, &qv)),
                                    positions_world: Arc::new(pos),
                                    values: Arc::new(qv),
                                    ids,
                                }));
                            }
                            self.preloaded = Some(preloaded);
                            self.status = format!("Preloaded {total_points} transcripts.");

                            // Attach data to any already-enabled genes.
                            let preloaded = self.preloaded.as_deref();
                            for g in self.genes.values_mut() {
                                if g.series.enabled && g.series.positions_world.is_none() {
                                    Self::attach_preloaded_for_gene(preloaded, g);
                                }
                            }
                            self.rebuild_hover_index();
                        }
                        Err(err) => {
                            self.status = format!("Preload failed: {err}");
                        }
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.preload_rx = None;
                    break;
                }
            }
        }
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
        if !gpu_available {
            return;
        }
        let enabled_genes = self
            .genes
            .values()
            .filter(|g| g.series.enabled)
            .count()
            .max(1);
        let per_gene_render_budget = if self.max_render_points_total == 0 {
            None
        } else {
            Some((self.max_render_points_total / enabled_genes).max(1))
        };
        for g in self.genes.values() {
            if !g.series.enabled {
                continue;
            }
            g.series.draw(
                ui,
                viewport,
                camera,
                local_to_world_offset,
                self.visible,
                gpu_available,
                0.0,
                &self.style,
                per_gene_render_budget,
            );
        }
    }

    pub fn ui_properties(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        changed |= ui.checkbox(&mut self.visible, "Visible").changed();
        ui.separator();
        let picker_items = self
            .meta
            .as_ref()
            .map(|meta| {
                meta.gene_names
                    .iter()
                    .map(|gene_name| {
                        let key = normalize_feature_key(gene_name);
                        let entry = self.genes.get(&key);
                        FeaturePickerItem {
                            name: gene_name.clone(),
                            enabled: entry.map(|g| g.series.enabled).unwrap_or(false),
                            color_rgb: entry
                                .map(|g| g.series.color_rgb)
                                .unwrap_or_else(|| color_for_feature(gene_name)),
                            status: entry.map(|g| g.series.status.clone()),
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let picker = show_feature_picker(
            ui,
            "xenium_gene_picker",
            "Genes",
            "Transcripts genes",
            &mut self.gene_query,
            &mut self.gene_popup_open,
            &picker_items,
        );
        if !picker.toggles.is_empty() {
            for (gene, on) in picker.toggles {
                self.set_gene_enabled(gene.as_str(), on);
            }
            self.rebuild_hover_index();
            changed = true;
        }
        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Preload limit (0 = no limit)");
            ui.add(egui::DragValue::new(&mut self.max_points_total).speed(1000));
            if ui.button("Reload").clicked() {
                // Re-run the preload with the new limit.
                self.preloaded = None;
                self.preload_rx = None;
                for g in self.genes.values_mut() {
                    g.ids = None;
                    g.series.clear_payload();
                    g.series.status.clear();
                }
                self.request_preload_all();
                self.rebuild_hover_index();
                changed = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label("Render cap (0 = no limit)");
            changed |= ui
                .add(egui::DragValue::new(&mut self.max_render_points_total).speed(1000))
                .changed();
        });
        ui.separator();
        changed |= ui
            .add(egui::Slider::new(&mut self.style.radius_screen_px, 0.5..=20.0).text("Size"))
            .changed();
        ui.separator();
        let selected = self.genes.values().filter(|g| g.series.enabled).count();
        let total_points: usize = self
            .genes
            .values()
            .filter(|g| g.series.enabled)
            .map(|g| {
                g.series
                    .positions_world
                    .as_ref()
                    .map(|p| p.len())
                    .unwrap_or(0)
            })
            .sum();
        if selected > 0 {
            ui.label(format!(
                "Selected genes: {selected}  points: {total_points}  render cap: {}",
                if self.max_render_points_total == 0 {
                    "unlimited".to_string()
                } else {
                    self.max_render_points_total.to_string()
                }
            ));
        }
        if !self.status.is_empty() {
            ui.label(self.status.clone());
        }
        changed
    }

    fn hover_point_index(
        &self,
        pointer_world: egui::Pos2,
        local_to_world_offset: egui::Vec2,
        camera: &crate::camera::Camera,
    ) -> Option<usize> {
        let bins = self.hover_bins.as_ref()?;
        let points = self.hover_positions_world.as_ref()?;
        if points.is_empty() {
            return None;
        }

        let pointer = pointer_world - local_to_world_offset;
        let zoom = camera.zoom_screen_per_lvl0_px.max(1e-6);
        let radius_screen = (self.style.radius_screen_px.max(0.5) * zoom.sqrt()).clamp(0.75, 40.0);
        let radius_world = radius_screen / zoom;
        let query = egui::Rect::from_center_size(
            pointer,
            egui::vec2(radius_world * 2.0, radius_world * 2.0),
        );
        let (x0, y0, x1, y1) = bins.bin_range_for_world_rect(query);

        let mut best_i: Option<usize> = None;
        let mut best_d2 = radius_world * radius_world;
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
        let href = self.hover_refs.get(idx).copied()?;
        let pos = self.hover_positions_world.as_ref()?.get(idx).copied()?;
        let mut lines: Vec<String> = Vec::new();
        let gene = self
            .meta
            .as_ref()
            .and_then(|m| m.gene_names.get(href.gene_id as usize))
            .cloned()
            .unwrap_or_else(|| format!("gene_id={}", href.gene_id));
        lines.push(gene);
        let world = pos + local_to_world_offset;
        lines.push(format!("x: {:.2}", world.x));
        lines.push(format!("y: {:.2}", world.y));
        let g = self.genes.values().find(|g| g.gene_id == href.gene_id);
        let qv = g
            .and_then(|g| g.series.values.as_ref())
            .and_then(|v| v.get(href.point_index as usize))
            .copied()
            .unwrap_or(0.0);
        lines.push(format!("qv: {:.3}", qv));
        if let Some(id) = g
            .and_then(|g| g.ids.as_ref())
            .and_then(|v| v.get(href.point_index as usize))
            .copied()
        {
            if id != u64::MAX {
                lines.push(format!("id: {id}"));
            }
        }
        Some(lines)
    }
}
