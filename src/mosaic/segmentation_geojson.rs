use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;

use eframe::egui;

use crate::objects::{
    ObjectColorLegendEntry, ObjectColorLevelOverride, ObjectsLayer, SelectedObjectDetails,
};
use crate::spatialdata::SpatialDataTransform2;

#[derive(Debug, Default)]
struct ItemState {
    seg_path: Option<PathBuf>,
    layer: Option<ObjectsLayer>,
    status: String,
}

#[derive(Debug, Clone, Copy)]
struct SharedStyle {
    visible: bool,
    opacity: f32,
    width_screen_px: f32,
    color_rgb: [u8; 3],
    fill_cells: bool,
    fill_opacity: f32,
    selected_fill_opacity: f32,
    downsample_factor: f32,
}

#[derive(Debug)]
pub struct MosaicGeoJsonSegmentationOverlay {
    pub visible: bool,
    pub opacity: f32,
    pub width_screen_px: f32,
    pub color_rgb: [u8; 3],
    pub fill_cells: bool,
    pub fill_opacity: f32,
    pub selected_fill_opacity: f32,
    pub downsample_factor: f32,

    samplesheet_dir: Option<PathBuf>,
    items: HashMap<usize, ItemState>,
    color_property_key: String,
    color_level_overrides: HashMap<String, HashMap<String, ObjectColorLevelOverride>>,
    max_concurrent_loads: usize,
    force_repaint_frames: u32,
    primary_selected_item_id: Option<usize>,
}

impl Default for MosaicGeoJsonSegmentationOverlay {
    fn default() -> Self {
        Self {
            visible: false,
            opacity: 0.75,
            width_screen_px: 1.0,
            color_rgb: [0, 255, 120],
            fill_cells: false,
            fill_opacity: 0.30,
            selected_fill_opacity: 0.70,
            downsample_factor: 1.0,
            samplesheet_dir: None,
            items: HashMap::new(),
            color_property_key: String::new(),
            color_level_overrides: HashMap::new(),
            max_concurrent_loads: 1,
            force_repaint_frames: 0,
            primary_selected_item_id: None,
        }
    }
}

impl MosaicGeoJsonSegmentationOverlay {
    pub fn set_samplesheet_dir(&mut self, dir: Option<PathBuf>) {
        self.samplesheet_dir = dir;
    }

    pub fn discover_from_meta(&mut self, item_id: usize, meta: &HashMap<String, String>) {
        let Some(raw) = meta.get("segpath") else {
            return;
        };
        let raw = raw.trim();
        if raw.is_empty() {
            return;
        }
        let p = PathBuf::from(raw);
        let resolved = if p.is_relative() {
            self.samplesheet_dir
                .as_ref()
                .map(|d| d.join(p))
                .unwrap_or_else(|| PathBuf::from(raw))
        } else {
            p
        };
        let st = self.items.entry(item_id).or_default();
        st.seg_path = Some(resolved);
    }

    pub fn is_busy(&self) -> bool {
        self.force_repaint_frames > 0
            || self.items.values().any(|s| {
                s.layer
                    .as_ref()
                    .is_some_and(|layer| layer.is_loading() || layer.is_busy())
            })
    }

    pub fn ui_left_panel(&mut self, ui: &mut egui::Ui, have_any: bool) -> bool {
        if !have_any {
            return false;
        }
        let mut zoom_requested = false;

        ui.separator();
        ui.heading("Segmentation");
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.visible, "");
            ui.label("Selectable GeoJSON objects");
        });
        ui.add_enabled(
            self.visible,
            egui::Slider::new(&mut self.opacity, 0.0..=1.0)
                .text("Opacity")
                .show_value(true)
                .clamping(egui::SliderClamping::Always),
        );
        ui.add_enabled(
            self.visible,
            egui::Slider::new(&mut self.width_screen_px, 0.25..=4.0)
                .text("Width")
                .show_value(true)
                .clamping(egui::SliderClamping::Always),
        );
        ui.add_enabled_ui(self.visible, |ui| {
            ui.checkbox(&mut self.fill_cells, "Fill cells");
            ui.add_enabled(
                self.fill_cells,
                egui::Slider::new(&mut self.fill_opacity, 0.0..=1.0)
                    .text("Fill opacity")
                    .show_value(true)
                    .clamping(egui::SliderClamping::Always),
            );
            ui.add(
                egui::Slider::new(&mut self.selected_fill_opacity, 0.0..=1.0)
                    .text("Selected fill")
                    .show_value(true)
                    .clamping(egui::SliderClamping::Always),
            );
        });
        ui.horizontal(|ui| {
            ui.label("Color");
            let mut c =
                egui::Color32::from_rgb(self.color_rgb[0], self.color_rgb[1], self.color_rgb[2]);
            if ui.color_edit_button_srgba(&mut c).changed() {
                self.color_rgb = [c.r(), c.g(), c.b()];
            }
        });
        let available_color_properties = self.available_color_properties();
        ui.horizontal(|ui| {
            ui.label("Color by");
            egui::ComboBox::from_id_salt("mosaic_seg_objects_color_mode")
                .selected_text(if self.color_property_key.is_empty() {
                    "Single color".to_string()
                } else {
                    self.color_property_key.clone()
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.color_property_key,
                        String::new(),
                        "Single color",
                    );
                    for key in &available_color_properties {
                        ui.selectable_value(&mut self.color_property_key, key.clone(), key);
                    }
                });
        });
        ui.horizontal(|ui| {
            ui.label("Load");
            ui.add(
                egui::DragValue::new(&mut self.max_concurrent_loads)
                    .range(1..=8)
                    .speed(1),
            )
            .on_hover_text("Maximum number of GeoJSON files to load concurrently.");
        });
        ui.horizontal(|ui| {
            ui.add(
                egui::DragValue::new(&mut self.downsample_factor)
                    .speed(0.1)
                    .prefix("Downsample "),
            )
            .on_hover_text("Scales GeoJSON coordinates by this factor (use if GeoJSON was generated on downsampled imagery).");
        });
        if !self.color_property_key.is_empty() {
            let legend = self.active_color_legend_entries();
            if !legend.is_empty() {
                ui.separator();
                ui.label(format!("Legend: {}", self.color_property_key));
                egui::ScrollArea::vertical()
                    .id_salt("mosaic_seg_objects_legend_scroll")
                    .max_height(140.0)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        let property_key = self.color_property_key.clone();
                        for entry in legend {
                            let default_color = entry.color_rgb;
                            let override_style = self
                                .color_level_overrides
                                .entry(property_key.clone())
                                .or_default()
                                .entry(entry.value_label.clone())
                                .or_default();
                            let mut visible = override_style.visible;
                            let color_rgb = override_style.color_rgb.unwrap_or(default_color);
                            let mut color =
                                egui::Color32::from_rgb(color_rgb[0], color_rgb[1], color_rgb[2]);
                            ui.horizontal(|ui| {
                                if ui.checkbox(&mut visible, "").changed() {
                                    override_style.visible = visible;
                                }
                                if ui.color_edit_button_srgba(&mut color).changed() {
                                    let next_rgb = [color.r(), color.g(), color.b()];
                                    override_style.color_rgb =
                                        (next_rgb != default_color).then_some(next_rgb);
                                }
                                ui.label(format!("{} ({})", entry.value_label, entry.count));
                            });
                        }
                    });
            }
        }

        ui.separator();
        ui.label(format!("Selected: {}", self.selection_count()));
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    self.selection_count() > 0,
                    egui::Button::new("Clear selection"),
                )
                .clicked()
            {
                self.clear_selection();
            }
            if ui
                .add_enabled(
                    self.selected_bounds_world().is_some(),
                    egui::Button::new("Zoom to selected"),
                )
                .clicked()
            {
                zoom_requested = true;
            }
        });

        ui.separator();
        ui.label("Primary object");
        if let Some((item_id, details)) = self.selected_object_details() {
            ui.label(format!("ROI item: {item_id}"));
            ui.label(format!("id: {}", details.id));
            ui.label(format!("area_px: {:.2}", details.area_px));
            ui.label(format!("perimeter_px: {:.2}", details.perimeter_px));
            ui.label(format!(
                "centroid: ({:.2}, {:.2})",
                details.centroid_world.x, details.centroid_world.y
            ));
            egui::ScrollArea::vertical()
                .id_salt("mosaic_seg_objects_properties_scroll")
                .max_height(220.0)
                .show(ui, |ui| {
                    for (key, value) in &details.properties {
                        ui.horizontal(|ui| {
                            ui.monospace(format!("{key}:"));
                            ui.label(value);
                        });
                    }
                });
        } else {
            ui.label("No object selected");
        }

        zoom_requested
    }

    pub fn tick(&mut self) {
        let mut loaded_any = false;
        for st in self.items.values_mut() {
            if let Some(layer) = st.layer.as_mut() {
                let was_loading = layer.is_loading();
                layer.tick();
                st.status = layer.status().to_string();
                if was_loading && !layer.is_loading() && layer.has_data() {
                    loaded_any = true;
                }
            }
        }
        if loaded_any {
            self.force_repaint_frames = self.force_repaint_frames.max(4);
        }
        self.update_primary_selection();
    }

    pub fn ensure_visible_items_loading(
        &mut self,
        items: &[(usize, egui::Rect, egui::Vec2, f32)],
        visible_world: egui::Rect,
    ) -> bool {
        if !self.visible {
            return false;
        }

        let mut pending_any = false;
        let style = self.shared_style();
        let color_level_overrides = self.current_color_level_overrides().clone();
        let mut in_flight = 0usize;
        for st in self.items.values() {
            if st.layer.as_ref().is_some_and(ObjectsLayer::is_loading) {
                in_flight += 1;
            }
        }
        let mut budget = self.max_concurrent_loads.saturating_sub(in_flight);

        for (id, world_rect, offset, scale) in items {
            if !world_rect.intersects(visible_world) {
                continue;
            }
            let Some(st) = self.items.get_mut(id) else {
                continue;
            };
            let transform = mosaic_transform(*offset, *scale);
            if let Some(layer) = st.layer.as_mut() {
                layer.set_display_transform(transform);
                apply_style(
                    layer,
                    style,
                    Some(self.color_property_key.as_str()),
                    &color_level_overrides,
                );
                if layer.is_loading() {
                    pending_any = true;
                }
                continue;
            }

            let Some(path) = st.seg_path.clone() else {
                continue;
            };
            if !path.exists() {
                st.status = "Missing GeoJSON".to_string();
                st.seg_path = None;
                continue;
            }
            if budget == 0 {
                pending_any = true;
                continue;
            }

            let mut layer = ObjectsLayer::default();
            apply_style(
                &mut layer,
                style,
                Some(self.color_property_key.as_str()),
                &color_level_overrides,
            );
            layer.load_objects_with_transform(path.clone(), self.downsample_factor, transform);
            st.status = format!("Loading objects: {}", path.to_string_lossy());
            st.layer = Some(layer);
            budget = budget.saturating_sub(1);
            pending_any = true;
        }

        pending_any
    }

    pub fn paint(
        &mut self,
        ui: &mut egui::Ui,
        camera: &crate::camera::Camera,
        viewport: egui::Rect,
        visible_world: egui::Rect,
        visible_items: &[(usize, egui::Rect, egui::Vec2, f32)],
    ) -> bool {
        if !self.visible {
            return false;
        }

        let mut pending_any = false;
        let style = self.shared_style();
        let color_level_overrides = self.current_color_level_overrides().clone();
        for (item_id, world_rect, offset, scale) in visible_items {
            if !world_rect.intersects(visible_world) {
                continue;
            }
            let Some(st) = self.items.get_mut(item_id) else {
                continue;
            };
            let Some(layer) = st.layer.as_mut() else {
                continue;
            };
            layer.set_display_transform(mosaic_transform(*offset, *scale));
            apply_style(
                layer,
                style,
                Some(self.color_property_key.as_str()),
                &color_level_overrides,
            );
            layer.draw(ui, camera, viewport, visible_world, egui::Vec2::ZERO, true);
            pending_any |= layer.is_loading();
        }

        if self.force_repaint_frames > 0 {
            self.force_repaint_frames = self.force_repaint_frames.saturating_sub(1);
            pending_any = true;
        }
        pending_any
    }

    pub fn hover_tooltip(
        &mut self,
        item_id: usize,
        pointer_world: egui::Pos2,
        camera: &crate::camera::Camera,
    ) -> Option<Vec<String>> {
        let style = self.shared_style();
        let color_level_overrides = self.current_color_level_overrides().clone();
        let st = self.items.get_mut(&item_id)?;
        let layer = st.layer.as_mut()?;
        apply_style(
            layer,
            style,
            Some(self.color_property_key.as_str()),
            &color_level_overrides,
        );
        layer.hover_tooltip(pointer_world, egui::Vec2::ZERO, camera)
    }

    pub fn select_at(
        &mut self,
        item_id: usize,
        pointer_world: egui::Pos2,
        camera: &crate::camera::Camera,
        additive: bool,
        toggle: bool,
    ) {
        let style = self.shared_style();
        let color_level_overrides = self.current_color_level_overrides().clone();
        if !additive && !toggle {
            self.clear_selection();
        }

        if let Some(st) = self.items.get_mut(&item_id)
            && let Some(layer) = st.layer.as_mut()
        {
            apply_style(
                layer,
                style,
                Some(self.color_property_key.as_str()),
                &color_level_overrides,
            );
            layer.select_at(pointer_world, egui::Vec2::ZERO, camera, additive, toggle);
        }
        self.update_primary_selection();
    }

    pub fn clear_selection(&mut self) {
        for st in self.items.values_mut() {
            if let Some(layer) = st.layer.as_mut() {
                layer.clear_selection();
            }
        }
        self.primary_selected_item_id = None;
    }

    pub fn selection_count(&self) -> usize {
        self.items
            .values()
            .filter_map(|st| st.layer.as_ref())
            .map(ObjectsLayer::selection_count)
            .sum()
    }

    pub fn selected_object_details(&self) -> Option<(usize, SelectedObjectDetails)> {
        let item_id = self.primary_selected_item_id?;
        let st = self.items.get(&item_id)?;
        let layer = st.layer.as_ref()?;
        Some((item_id, layer.selected_object_details(egui::Vec2::ZERO)?))
    }

    pub fn selected_bounds_world(&self) -> Option<egui::Rect> {
        let item_id = self.primary_selected_item_id?;
        let st = self.items.get(&item_id)?;
        let layer = st.layer.as_ref()?;
        let idx = layer.selected_object_index()?;
        layer.fit_object_bounds_world(idx, egui::Vec2::ZERO)
    }

    pub fn loaded_stats(&self) -> (usize, usize, usize) {
        let mut total = 0usize;
        let mut loaded = 0usize;
        let mut loading = 0usize;
        for st in self.items.values() {
            if st.seg_path.is_some() {
                total += 1;
                if let Some(layer) = st.layer.as_ref() {
                    loaded += usize::from(layer.has_data());
                    loading += usize::from(layer.is_loading());
                }
            }
        }
        (loaded, loading, total)
    }

    pub fn last_missing_bins(&self) -> usize {
        0
    }

    pub fn has_any_segpaths(&self) -> bool {
        self.items.values().any(|s| s.seg_path.is_some())
    }

    fn available_color_properties(&self) -> Vec<String> {
        let mut keys = BTreeSet::new();
        if !self.color_property_key.is_empty() {
            keys.insert(self.color_property_key.clone());
        }
        for st in self.items.values() {
            let Some(layer) = st.layer.as_ref() else {
                continue;
            };
            for key in layer.available_property_columns() {
                keys.insert(key.clone());
            }
        }
        keys.into_iter().collect()
    }

    fn active_color_legend_entries(&mut self) -> Vec<ObjectColorLegendEntry> {
        let mut merged = std::collections::BTreeMap::<String, ([u8; 3], usize)>::new();
        for st in self.items.values_mut() {
            let Some(layer) = st.layer.as_mut() else {
                continue;
            };
            let Some(entries) = layer.active_color_legend_entries() else {
                continue;
            };
            for entry in entries {
                let slot = merged
                    .entry(entry.value_label)
                    .or_insert((entry.color_rgb, 0usize));
                slot.1 += entry.count;
            }
        }
        merged
            .into_iter()
            .map(|(value_label, (color_rgb, count))| ObjectColorLegendEntry {
                value_label,
                count,
                color_rgb,
            })
            .collect()
    }

    fn shared_style(&self) -> SharedStyle {
        SharedStyle {
            visible: self.visible,
            opacity: self.opacity,
            width_screen_px: self.width_screen_px,
            color_rgb: self.color_rgb,
            fill_cells: self.fill_cells,
            fill_opacity: self.fill_opacity,
            selected_fill_opacity: self.selected_fill_opacity,
            downsample_factor: self.downsample_factor,
        }
    }

    fn update_primary_selection(&mut self) {
        if self.primary_selected_item_id.is_some_and(|item_id| {
            self.items
                .get(&item_id)
                .and_then(|st| st.layer.as_ref())
                .is_some_and(|layer| layer.selected_object_index().is_some())
        }) {
            return;
        }

        self.primary_selected_item_id = self.items.iter().find_map(|(item_id, st)| {
            st.layer
                .as_ref()
                .and_then(|layer| layer.selected_object_index().map(|_| *item_id))
        });
    }

    fn current_color_level_overrides(&self) -> &HashMap<String, ObjectColorLevelOverride> {
        self.color_level_overrides
            .get(&self.color_property_key)
            .unwrap_or(&EMPTY_COLOR_LEVEL_OVERRIDES)
    }
}

static EMPTY_COLOR_LEVEL_OVERRIDES: std::sync::LazyLock<HashMap<String, ObjectColorLevelOverride>> =
    std::sync::LazyLock::new(HashMap::new);

fn mosaic_transform(offset: egui::Vec2, scale: f32) -> SpatialDataTransform2 {
    SpatialDataTransform2 {
        scale: [scale, scale],
        translation: [offset.x, offset.y],
    }
}

fn apply_style(
    layer: &mut ObjectsLayer,
    style: SharedStyle,
    color_property_key: Option<&str>,
    color_level_overrides: &HashMap<String, ObjectColorLevelOverride>,
) {
    layer.visible = style.visible;
    layer.opacity = style.opacity;
    layer.width_screen_px = style.width_screen_px;
    layer.color_rgb = style.color_rgb;
    layer.fill_cells = style.fill_cells;
    layer.fill_opacity = style.fill_opacity;
    layer.selected_fill_opacity = style.selected_fill_opacity;
    layer.downsample_factor = style.downsample_factor;
    layer.set_color_by_property(color_property_key.map(str::to_owned));
    layer.set_color_level_overrides(color_property_key, color_level_overrides);
}
