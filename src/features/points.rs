use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use eframe::egui;

use crate::camera::Camera;
use crate::render::points::PointsStyle;
use crate::render::points_gl::{PointsGlDrawData, PointsGlDrawParams, PointsGlRenderer};

#[derive(Debug, Clone)]
pub struct FeaturePickerItem {
    pub name: String,
    pub enabled: bool,
    pub color_rgb: [u8; 3],
    pub status: Option<String>,
}

#[derive(Debug, Default)]
pub struct FeaturePickerResult {
    pub toggles: Vec<(String, bool)>,
}

#[derive(Debug, Clone)]
pub struct FeaturePointLod {
    pub bin_world: f32,
    pub positions_world: Arc<Vec<egui::Pos2>>,
    pub values: Arc<Vec<f32>>,
}

#[derive(Debug)]
pub struct FeaturePointSeries {
    pub feature_name: String,
    pub enabled: bool,
    pub color_rgb: [u8; 3],
    pub point_count: usize,
    pub positions_world: Option<Arc<Vec<egui::Pos2>>>,
    pub values: Option<Arc<Vec<f32>>>,
    pub lod_levels: Option<Arc<Vec<FeaturePointLod>>>,
    pub generation: u64,
    pub gl: PointsGlRenderer,
    pub status: String,
}

impl FeaturePointSeries {
    pub fn new(feature_name: String, color_rgb: [u8; 3]) -> Self {
        Self {
            feature_name,
            enabled: false,
            color_rgb,
            point_count: 0,
            positions_world: None,
            values: None,
            lod_levels: None,
            generation: 1,
            gl: PointsGlRenderer::default(),
            status: String::new(),
        }
    }

    pub fn clear_payload(&mut self) {
        self.positions_world = None;
        self.values = None;
        self.lod_levels = None;
        self.generation = self.generation.wrapping_add(1).max(1);
    }

    pub fn set_payload(
        &mut self,
        positions_world: Arc<Vec<egui::Pos2>>,
        values: Arc<Vec<f32>>,
        lod_levels: Option<Arc<Vec<FeaturePointLod>>>,
    ) {
        self.point_count = positions_world.len().min(values.len());
        self.positions_world = Some(positions_world);
        self.values = Some(values);
        self.lod_levels = lod_levels;
        self.generation = self.generation.wrapping_add(1).max(1);
    }

    pub fn draw_payload_for_zoom(
        &self,
        zoom_screen_per_world: f32,
        max_points: Option<usize>,
    ) -> Option<(u64, Arc<Vec<egui::Pos2>>, Arc<Vec<f32>>)> {
        let full_positions = self.positions_world.as_ref()?;
        let full_values = self.values.as_ref()?;
        Some(select_draw_payload(
            self.generation,
            full_positions,
            full_values,
            self.lod_levels.as_ref().map(|v| v.as_slice()),
            zoom_screen_per_world,
            max_points,
        ))
    }

    pub fn draw(
        &self,
        ui: &mut egui::Ui,
        viewport: egui::Rect,
        camera: &Camera,
        local_to_world_offset: egui::Vec2,
        visible: bool,
        gpu_available: bool,
        threshold: f32,
        style: &PointsStyle,
        max_points: Option<usize>,
    ) {
        if !visible || !gpu_available || !self.enabled {
            return;
        }
        let Some((generation, positions_world, values)) =
            self.draw_payload_for_zoom(camera.zoom_screen_per_lvl0_px, max_points)
        else {
            return;
        };
        if positions_world.is_empty() || values.is_empty() {
            return;
        }
        let data = PointsGlDrawData {
            generation,
            positions_world,
            values,
        };
        let params = PointsGlDrawParams {
            center_world: camera.center_world_lvl0,
            zoom_screen_per_world: camera.zoom_screen_per_lvl0_px,
            threshold,
            style: feature_style(style, self.color_rgb),
            visible,
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
}

pub fn normalize_feature_key(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>()
        .to_ascii_uppercase()
}

pub fn color_for_feature(name: &str) -> [u8; 3] {
    let mut h = DefaultHasher::new();
    normalize_feature_key(name).hash(&mut h);
    let idx = (h.finish() as usize) % feature_palette().len();
    feature_palette()[idx]
}

pub fn show_feature_picker(
    ui: &mut egui::Ui,
    id_salt: &str,
    label: &str,
    title: &str,
    query: &mut String,
    popup_open: &mut bool,
    items: &[FeaturePickerItem],
) -> FeaturePickerResult {
    let mut result = FeaturePickerResult::default();
    let mut picker_resp: Option<egui::Response> = None;

    ui.horizontal(|ui| {
        ui.label(label);
        let resp =
            ui.add(egui::TextEdit::singleline(query).hint_text("click to select / type to filter"));
        picker_resp = Some(resp.clone());
        if resp.clicked() {
            *popup_open = !*popup_open;
        }
        ui.label(format!(
            "{} selected",
            items.iter().filter(|item| item.enabled).count()
        ));
    });

    if !*popup_open {
        return result;
    }

    let Some(anchor_resp) = picker_resp.as_ref() else {
        *popup_open = false;
        return result;
    };

    let popup_id = ui.make_persistent_id(id_salt);
    let needle = normalize_feature_key(query);
    let filtered: Option<Vec<usize>> = if needle.is_empty() {
        None
    } else {
        let mut out = Vec::new();
        for (i, item) in items.iter().enumerate() {
            if normalize_feature_key(&item.name).contains(&needle) {
                out.push(i);
            }
        }
        Some(out)
    };
    let rows = filtered.as_ref().map(|v| v.len()).unwrap_or(items.len());

    let mut request_enable_all = false;
    let mut request_disable_all = false;
    let mut request_close = false;

    let popup = egui::Popup::from_response(anchor_resp)
        .id(popup_id)
        .open_bool(popup_open)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .gap(2.0)
        .width(anchor_resp.rect.width().max(520.0))
        .show(|ui| {
            ui.set_min_width(anchor_resp.rect.width().max(520.0));
            ui.set_max_width(680.0);

            ui.horizontal(|ui| {
                ui.heading(title);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Close").clicked() {
                        request_close = true;
                    }
                });
            });
            ui.add_space(6.0);

            ui.horizontal(|ui| {
                if ui.button("Enable all (filtered)").clicked() {
                    request_enable_all = true;
                }
                if ui.button("Disable all (filtered)").clicked() {
                    request_disable_all = true;
                }
                if ui.button("Clear filter").clicked() {
                    query.clear();
                }
            });
            ui.add_space(6.0);

            ui.label(format!("Showing {rows} entries"));
            ui.separator();

            let row_h = ui.text_style_height(&egui::TextStyle::Body) + 6.0;
            egui::ScrollArea::vertical()
                .id_salt(format!("{id_salt}_scroll"))
                .max_height(420.0)
                .auto_shrink([false, false])
                .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysVisible)
                .show_rows(ui, row_h, rows, |ui, row_range| {
                    for row_i in row_range {
                        let item_i = filtered.as_ref().map(|v| v[row_i]).unwrap_or(row_i);
                        let item = &items[item_i];
                        ui.horizontal(|ui| {
                            let mut on = item.enabled;
                            let label =
                                egui::RichText::new(&item.name).color(egui::Color32::from_rgb(
                                    item.color_rgb[0],
                                    item.color_rgb[1],
                                    item.color_rgb[2],
                                ));
                            let r = ui.checkbox(&mut on, label);
                            if r.changed() {
                                result.toggles.push((item.name.clone(), on));
                            }

                            if let Some(status) = &item.status {
                                if !status.is_empty() && (item.enabled || on) {
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            ui.add(egui::Label::new(status.clone()).truncate());
                                        },
                                    );
                                }
                            }
                        });
                    }
                });
        });

    if request_close {
        *popup_open = false;
    }
    if request_enable_all || request_disable_all {
        let enabled = request_enable_all && !request_disable_all;
        let indices = filtered.unwrap_or_else(|| (0..items.len()).collect());
        for item_i in indices {
            result.toggles.push((items[item_i].name.clone(), enabled));
        }
    }

    let _ = popup;
    result
}

pub fn feature_style(base: &PointsStyle, rgb: [u8; 3]) -> PointsStyle {
    let mut style = base.clone();
    style.fill_positive = egui::Color32::from_rgba_unmultiplied(rgb[0], rgb[1], rgb[2], 230);
    style
}

pub fn build_point_lods(positions: &[egui::Pos2], values: &[f32]) -> Vec<FeaturePointLod> {
    const LOD_BIN_WORLD: [f32; 11] = [
        2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0,
    ];
    LOD_BIN_WORLD
        .iter()
        .filter_map(|&bin_world| build_point_lod(positions, values, bin_world))
        .collect()
}

pub fn select_draw_payload(
    generation: u64,
    full_positions: &Arc<Vec<egui::Pos2>>,
    full_values: &Arc<Vec<f32>>,
    lod_levels: Option<&[FeaturePointLod]>,
    zoom_screen_per_world: f32,
    max_points: Option<usize>,
) -> (u64, Arc<Vec<egui::Pos2>>, Arc<Vec<f32>>) {
    let full_count = full_positions.len().min(full_values.len());
    let zoom_choice = lod_levels.and_then(|lods| choose_lod_level(lods, zoom_screen_per_world));
    let count_choice = match (lod_levels, max_points) {
        (Some(lods), Some(limit)) => choose_lod_level_by_count(lods, full_count, limit),
        _ => None,
    };
    let chosen = match (zoom_choice, count_choice) {
        (Some((zi, zl)), Some((ci, cl))) => {
            if ci > zi {
                Some((ci, cl))
            } else {
                Some((zi, zl))
            }
        }
        (Some(choice), None) | (None, Some(choice)) => Some(choice),
        (None, None) => None,
    };
    if let Some((lod_idx, lod)) = chosen {
        (
            generation
                .wrapping_mul(257)
                .wrapping_add((lod_idx as u64) + 1),
            Arc::clone(&lod.positions_world),
            Arc::clone(&lod.values),
        )
    } else {
        (
            generation,
            Arc::clone(full_positions),
            Arc::clone(full_values),
        )
    }
}

fn choose_lod_level(
    lod_levels: &[FeaturePointLod],
    zoom_screen_per_world: f32,
) -> Option<(usize, &FeaturePointLod)> {
    const MAX_BIN_SCREEN_PX: f32 = 8.0;
    let mut chosen: Option<(usize, &FeaturePointLod)> = None;
    for (idx, lod) in lod_levels.iter().enumerate() {
        if lod.bin_world * zoom_screen_per_world <= MAX_BIN_SCREEN_PX {
            chosen = Some((idx, lod));
        }
    }
    chosen
}

fn choose_lod_level_by_count(
    lod_levels: &[FeaturePointLod],
    full_count: usize,
    max_points: usize,
) -> Option<(usize, &FeaturePointLod)> {
    if max_points == 0 || full_count <= max_points {
        return None;
    }
    for (idx, lod) in lod_levels.iter().enumerate() {
        if lod.positions_world.len() <= max_points {
            return Some((idx, lod));
        }
    }
    lod_levels.iter().enumerate().next_back()
}

fn build_point_lod(
    positions: &[egui::Pos2],
    values: &[f32],
    bin_world: f32,
) -> Option<FeaturePointLod> {
    #[derive(Default)]
    struct Bucket {
        sum_x: f32,
        sum_y: f32,
        sum_v: f32,
        count: u32,
    }

    let mut bins: std::collections::HashMap<(i32, i32), Bucket> = std::collections::HashMap::new();
    for (i, &p) in positions.iter().enumerate() {
        if !(p.x.is_finite() && p.y.is_finite()) {
            continue;
        }
        let bx = (p.x / bin_world).floor() as i32;
        let by = (p.y / bin_world).floor() as i32;
        let bucket = bins.entry((bx, by)).or_default();
        bucket.sum_x += p.x;
        bucket.sum_y += p.y;
        bucket.sum_v += values.get(i).copied().unwrap_or(0.0);
        bucket.count += 1;
    }

    if bins.is_empty() || bins.len() >= positions.len() {
        return None;
    }

    let mut lod_positions = Vec::with_capacity(bins.len());
    let mut lod_values = Vec::with_capacity(bins.len());
    for bucket in bins.into_values() {
        if bucket.count == 0 {
            continue;
        }
        let denom = bucket.count as f32;
        lod_positions.push(egui::pos2(bucket.sum_x / denom, bucket.sum_y / denom));
        lod_values.push(bucket.sum_v / denom);
    }

    if lod_positions.is_empty() || lod_positions.len() >= positions.len() {
        return None;
    }

    Some(FeaturePointLod {
        bin_world,
        positions_world: Arc::new(lod_positions),
        values: Arc::new(lod_values),
    })
}

fn feature_palette() -> [[u8; 3]; 20] {
    [
        [31, 119, 180],
        [255, 127, 14],
        [44, 160, 44],
        [214, 39, 40],
        [148, 103, 189],
        [140, 86, 75],
        [227, 119, 194],
        [127, 127, 127],
        [188, 189, 34],
        [23, 190, 207],
        [174, 199, 232],
        [255, 187, 120],
        [152, 223, 138],
        [255, 152, 150],
        [197, 176, 213],
        [196, 156, 148],
        [247, 182, 210],
        [199, 199, 199],
        [219, 219, 141],
        [158, 218, 229],
    ]
}
