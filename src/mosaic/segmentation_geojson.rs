use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::Arc;

use eframe::egui;

use crate::objects::legend_ui::{
    CategoryColors, LegendAction, LegendUiState, category_colors_params, legend_rows, ui_legend,
};
use crate::objects::{
    ObjectColorLegendEntry, ObjectColorLevelOverride, ObjectsLayer, PreloadedObjectLayer,
    SelectedObjectDetails,
};
use crate::render::polygon_fill_gl::ObjectFillGlRenderer;
use crate::spatialdata::SpatialDataTransform2;
use odon::model::{
    ContinuousDomain, ContinuousPalette, ContinuousScale, ObjectColorMapping, OutOfRangeMode,
};

#[derive(Debug, Default)]
struct ItemState {
    roi_id: String,
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
    fast_rendering: bool,
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
    pub fast_rendering: bool,
    pub downsample_factor: f32,

    samplesheet_dir: Option<PathBuf>,
    items: HashMap<usize, ItemState>,
    color_property_key: String,
    color_mapping: ObjectColorMapping,
    continuous_domains_by_roi: HashMap<String, [f64; 2]>,
    color_level_overrides: HashMap<String, HashMap<String, ObjectColorLevelOverride>>,
    category_colors: CategoryColors,
    legend_ui: LegendUiState,
    property_cache_capacity: Option<usize>,
    canonical_paths: HashMap<PathBuf, PathBuf>,
    actor_load_requested: BTreeSet<usize>,
    force_repaint_frames: u32,
    primary_selected_item_id: Option<usize>,
    object_fill_renderer: ObjectFillGlRenderer,
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
            fast_rendering: true,
            downsample_factor: 1.0,
            samplesheet_dir: None,
            items: HashMap::new(),
            color_property_key: String::new(),
            color_mapping: ObjectColorMapping::Single,
            continuous_domains_by_roi: HashMap::new(),
            color_level_overrides: HashMap::new(),
            category_colors: CategoryColors::default(),
            legend_ui: LegendUiState::default(),
            property_cache_capacity: None,
            canonical_paths: HashMap::new(),
            actor_load_requested: BTreeSet::new(),
            force_repaint_frames: 0,
            primary_selected_item_id: None,
            object_fill_renderer: ObjectFillGlRenderer::application_pool(),
        }
    }
}

impl MosaicGeoJsonSegmentationOverlay {
    pub(crate) fn set_object_fill_renderer(&mut self, renderer: ObjectFillGlRenderer) {
        self.object_fill_renderer = renderer.clone();
        for state in self.items.values_mut() {
            if let Some(layer) = state.layer.as_mut() {
                layer.set_object_fill_renderer(renderer.clone());
            }
        }
    }

    pub(crate) fn set_property_cache_capacity(&mut self, capacity: Option<usize>) {
        if self.property_cache_capacity == capacity {
            return;
        }
        self.property_cache_capacity = capacity;
        for state in self.items.values_mut() {
            if let Some(layer) = state.layer.as_mut() {
                layer.set_lazy_property_cache_capacity(capacity);
            }
        }
        self.force_repaint_frames = self.force_repaint_frames.max(2);
    }

    pub fn control_style_json(&self) -> serde_json::Value {
        serde_json::json!({
            "opacity":self.opacity,
            "width_screen_px":self.width_screen_px,
            "color_rgb":self.color_rgb,
            "fill_cells":self.fill_cells,
            "fill_opacity":self.fill_opacity,
            "selected_fill_opacity":self.selected_fill_opacity,
            "color_property_key":self.color_property_key,
            "color_mapping":self.color_mapping,
            "continuous_domains_by_roi":self.continuous_domains_by_roi,
            "color_level_overrides":self.color_level_overrides,
            "downsample_factor":self.downsample_factor,
        })
    }

    pub fn apply_control_style(&mut self, style: &serde_json::Value) -> Result<(), String> {
        if let Some(value) = style.get("opacity").and_then(serde_json::Value::as_f64) {
            self.opacity = value as f32;
        }
        if let Some(value) = style
            .get("width_screen_px")
            .and_then(serde_json::Value::as_f64)
        {
            self.width_screen_px = value as f32;
        }
        if let Some(values) = style
            .get("color_rgb")
            .and_then(serde_json::Value::as_array)
            .filter(|values| values.len() == 3)
        {
            self.color_rgb = [
                values[0].as_u64().unwrap_or(0) as u8,
                values[1].as_u64().unwrap_or(0) as u8,
                values[2].as_u64().unwrap_or(0) as u8,
            ];
        }
        if let Some(value) = style.get("fill_cells").and_then(serde_json::Value::as_bool) {
            self.fill_cells = value;
        }
        if let Some(value) = style
            .get("fill_opacity")
            .and_then(serde_json::Value::as_f64)
        {
            self.fill_opacity = value as f32;
        }
        if let Some(value) = style
            .get("selected_fill_opacity")
            .and_then(serde_json::Value::as_f64)
        {
            self.selected_fill_opacity = value as f32;
        }
        if let Some(value) = style
            .get("color_property_key")
            .and_then(serde_json::Value::as_str)
        {
            self.color_property_key = value.to_string();
            self.color_mapping = if self.color_property_key.is_empty() {
                ObjectColorMapping::Single
            } else {
                ObjectColorMapping::categorical(self.color_property_key.clone())
            };
        }
        if let Some(value) = style.get("color_mapping") {
            let mapping: ObjectColorMapping = serde_json::from_value(value.clone())
                .map_err(|error| format!("invalid mosaic object color mapping: {error}"))?;
            mapping.validate()?;
            self.color_property_key = mapping.property().unwrap_or_default().to_string();
            self.color_mapping = mapping;
        }
        if let Some(value) = style.get("continuous_domains_by_roi") {
            self.continuous_domains_by_roi = serde_json::from_value(value.clone())
                .map_err(|error| format!("invalid per-ROI continuous domains: {error}"))?;
        }
        if let Some(value) = style.get("color_level_overrides") {
            self.color_level_overrides = serde_json::from_value(value.clone())
                .map_err(|error| format!("invalid mosaic object legend projection: {error}"))?;
        }
        if let Some(value) = style
            .get("downsample_factor")
            .and_then(serde_json::Value::as_f64)
        {
            self.downsample_factor = value as f32;
        }
        Ok(())
    }

    pub fn control_selection_state_after_click(
        &self,
        item_id: usize,
        pointer_world: egui::Pos2,
        camera: &crate::camera::Camera,
        additive: bool,
        toggle: bool,
    ) -> Option<serde_json::Value> {
        let layer = self.items.get(&item_id)?.layer.as_ref()?;
        Some(layer.control_selection_state_after_click(
            pointer_world,
            egui::Vec2::ZERO,
            camera,
            additive,
            toggle,
        ))
    }

    pub fn apply_control_selections(
        &mut self,
        selections: &serde_json::Value,
    ) -> Result<(), String> {
        let selections = selections
            .as_object()
            .ok_or_else(|| "mosaic object selections projection must be an object".to_string())?;
        for (item_id, state) in selections {
            let item_id = item_id
                .parse::<usize>()
                .map_err(|_| "mosaic object selection key is invalid".to_string())?;
            let Some(layer) = self
                .items
                .get_mut(&item_id)
                .and_then(|state| state.layer.as_mut())
            else {
                continue;
            };
            let selected = state
                .get("selected_indices")
                .and_then(serde_json::Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(serde_json::Value::as_u64)
                .map(|index| index as usize)
                .collect::<Vec<_>>();
            let primary = state
                .get("primary_index")
                .and_then(serde_json::Value::as_u64)
                .map(|index| index as usize);
            layer.install_control_selection(&selected, primary)?;
        }
        self.update_primary_selection();
        Ok(())
    }

    pub fn set_samplesheet_dir(&mut self, dir: Option<PathBuf>) {
        self.samplesheet_dir = dir;
    }

    pub fn samplesheet_dir(&self) -> Option<PathBuf> {
        self.samplesheet_dir.clone()
    }

    pub fn segmentation_path(&self, item_id: usize) -> Option<PathBuf> {
        self.items
            .get(&item_id)
            .and_then(|state| state.seg_path.clone())
    }

    pub fn discover_from_meta(
        &mut self,
        item_id: usize,
        roi_id: &str,
        meta: &HashMap<String, String>,
    ) {
        self.items.entry(item_id).or_default().roi_id = roi_id.to_string();
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

    /// Called for every projection, so an unchanged preload must return before doing any work.
    pub fn install_preloaded(
        &mut self,
        seg_path: &std::path::Path,
        preloaded: &Arc<PreloadedObjectLayer>,
    ) -> usize {
        let targets = self
            .items
            .iter()
            .filter(|(_, st)| {
                st.seg_path.as_ref().is_some_and(|path| {
                    paths_match(path, seg_path, &mut self.canonical_paths)
                        && st
                            .layer
                            .as_ref()
                            .is_none_or(|layer| !layer.has_shared_preload(preloaded))
                })
            })
            .map(|(item_id, _)| *item_id)
            .collect::<Vec<_>>();
        if targets.is_empty() {
            return 0;
        }

        let style = self.shared_style();
        let color_mapping = self.render_color_mapping();
        let domains_by_roi = self.continuous_domains_by_roi.clone();
        let color_level_overrides = self.current_color_level_overrides().clone();
        let mut installed = 0usize;
        for item_id in targets {
            let Some(st) = self.items.get_mut(&item_id) else {
                continue;
            };
            let Some(path) = st.seg_path.as_ref() else {
                continue;
            };
            let mut layer = ObjectsLayer::default();
            layer.set_object_fill_renderer(self.object_fill_renderer.clone());
            layer.set_lazy_property_cache_capacity(self.property_cache_capacity);
            let item_color_mapping = color_mapping_with_domain(
                &color_mapping,
                domains_by_roi.get(st.roi_id.as_str()).copied(),
            );
            apply_style(
                &mut layer,
                style,
                &item_color_mapping,
                &color_level_overrides,
            );
            layer.install_shared_preload(preloaded);
            st.status = format!("Using cached objects: {}", path.to_string_lossy());
            st.layer = Some(layer);
            installed += 1;
        }
        if installed > 0 {
            self.force_repaint_frames = self.force_repaint_frames.max(4);
        }
        installed
    }

    pub fn install_control_resource(
        &mut self,
        item_id: usize,
        resource: &odon::model::ControlObjectResource,
    ) -> bool {
        let Some(preloaded) = resource.renderer_payload::<PreloadedObjectLayer>() else {
            return false;
        };
        let style = self.shared_style();
        let color_mapping = self.render_color_mapping();
        let domains_by_roi = self.continuous_domains_by_roi.clone();
        let color_level_overrides = self.current_color_level_overrides().clone();
        let Some(state) = self.items.get_mut(&item_id) else {
            return false;
        };
        let mut layer = ObjectsLayer::default();
        layer.set_object_fill_renderer(self.object_fill_renderer.clone());
        layer.set_lazy_property_cache_capacity(self.property_cache_capacity);
        let item_color_mapping = color_mapping_with_domain(
            &color_mapping,
            domains_by_roi.get(state.roi_id.as_str()).copied(),
        );
        apply_style(
            &mut layer,
            style,
            &item_color_mapping,
            &color_level_overrides,
        );
        layer.install_preloaded(preloaded);
        state.status = format!("Using actor-loaded objects: {}", resource.source.display());
        state.layer = Some(layer);
        self.actor_load_requested.remove(&item_id);
        self.visible = true;
        self.force_repaint_frames = self.force_repaint_frames.max(4);
        true
    }

    pub fn is_busy(&self) -> bool {
        self.visible
            && (self.force_repaint_frames > 0
                || self.items.values().any(|s| {
                    s.layer
                        .as_ref()
                        .is_some_and(|layer| layer.is_loading() || layer.is_busy())
                }))
    }

    pub fn set_fast_object_rendering(&mut self, enabled: bool) {
        self.fast_rendering = enabled;
        for st in self.items.values_mut() {
            if let Some(layer) = st.layer.as_mut() {
                layer.fast_rendering = enabled;
            }
        }
        self.force_repaint_frames = self.force_repaint_frames.max(2);
    }

    pub fn ui_left_panel(&mut self, ui: &mut egui::Ui, have_any: bool) -> (bool, bool) {
        if !have_any {
            return (false, false);
        }
        let mut zoom_requested = false;
        let mut clear_selection_requested = false;

        ui.separator();
        ui.heading("Segmentation");
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.visible, "");
            ui.label("Object segmentations");
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
        let mut color_mode = match self.color_mapping {
            ObjectColorMapping::Single => 0u8,
            ObjectColorMapping::Categorical { .. } => 1u8,
            ObjectColorMapping::Continuous { .. } => 2u8,
        };
        ui.horizontal(|ui| {
            ui.label("Color mode");
            egui::ComboBox::from_id_salt("mosaic_seg_objects_color_mode")
                .selected_text(["Single", "Categorical", "Continuous"][color_mode as usize])
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut color_mode, 0, "Single");
                    ui.selectable_value(&mut color_mode, 1, "Categorical");
                    ui.selectable_value(&mut color_mode, 2, "Continuous");
                });
        });
        let current_mode = match self.color_mapping {
            ObjectColorMapping::Single => 0,
            ObjectColorMapping::Categorical { .. } => 1,
            ObjectColorMapping::Continuous { .. } => 2,
        };
        if color_mode != current_mode {
            self.color_mapping = match color_mode {
                1 => self
                    .available_color_properties()
                    .into_iter()
                    .next()
                    .map(ObjectColorMapping::categorical)
                    .unwrap_or(ObjectColorMapping::Single),
                2 => self
                    .available_numeric_properties()
                    .into_iter()
                    .next()
                    .map(|property| ObjectColorMapping::Continuous {
                        property,
                        palette: ContinuousPalette::default(),
                        domain: ContinuousDomain::default(),
                        scale: ContinuousScale::default(),
                        reverse: false,
                        out_of_range: OutOfRangeMode::default(),
                        missing_color_rgb: None,
                    })
                    .unwrap_or(ObjectColorMapping::Single),
                _ => ObjectColorMapping::Single,
            };
            self.color_property_key = self
                .color_mapping
                .property()
                .unwrap_or_default()
                .to_string();
        }
        let available_color_properties = self.available_color_properties();
        let available_numeric_properties = self.available_numeric_properties();
        match self.color_mapping.clone() {
            ObjectColorMapping::Single => {}
            ObjectColorMapping::Categorical { mut property } => {
                ui.horizontal(|ui| {
                    ui.label("Property");
                    egui::ComboBox::from_id_salt("mosaic_seg_objects_categorical_property")
                        .selected_text(property.clone())
                        .show_ui(ui, |ui| {
                            for key in available_color_properties {
                                ui.selectable_value(&mut property, key.clone(), key);
                            }
                        });
                });
                self.color_property_key = property.clone();
                self.color_mapping = ObjectColorMapping::categorical(property);
            }
            ObjectColorMapping::Continuous { .. } => {
                let mut mapping = self.color_mapping.clone();
                if let ObjectColorMapping::Continuous {
                    property,
                    palette,
                    domain,
                    scale,
                    reverse,
                    out_of_range,
                    ..
                } = &mut mapping
                {
                    ui.horizontal(|ui| {
                        ui.label("Numeric property");
                        egui::ComboBox::from_id_salt("mosaic_seg_objects_continuous_property")
                            .selected_text(property.clone())
                            .show_ui(ui, |ui| {
                                for key in &available_numeric_properties {
                                    ui.selectable_value(property, key.clone(), key);
                                }
                            });
                    });
                    ui.horizontal(|ui| {
                        ui.label("Palette");
                        let label = match palette {
                            ContinuousPalette::Named(name) => name.as_str(),
                            ContinuousPalette::Custom(_) => "Custom",
                        };
                        egui::ComboBox::from_id_salt("mosaic_seg_objects_continuous_palette")
                            .selected_text(label)
                            .show_ui(ui, |ui| {
                                for name in ContinuousPalette::NAMED {
                                    if ui.selectable_label(
                                        matches!(palette, ContinuousPalette::Named(current) if current == name),
                                        name,
                                    ).clicked() {
                                        *palette = ContinuousPalette::Named(name.to_string());
                                    }
                                }
                            });
                        ui.checkbox(reverse, "Reverse");
                    });
                    ui.horizontal(|ui| {
                        ui.selectable_value(scale, ContinuousScale::Linear, "Linear");
                        ui.selectable_value(scale, ContinuousScale::Log10, "Log10");
                        ui.selectable_value(out_of_range, OutOfRangeMode::Clamp, "Clamp");
                        ui.selectable_value(out_of_range, OutOfRangeMode::Hide, "Hide");
                    });
                    let mut automatic = matches!(domain, ContinuousDomain::Automatic(_));
                    ui.horizontal(|ui| {
                        if ui.checkbox(&mut automatic, "Automatic range").changed() {
                            *domain = if automatic {
                                ContinuousDomain::default()
                            } else {
                                ContinuousDomain::Fixed([0.0, 1.0])
                            };
                        }
                        if let ContinuousDomain::Fixed(range) = domain {
                            ui.add(egui::DragValue::new(&mut range[0]).speed(0.1));
                            ui.label("to");
                            ui.add(egui::DragValue::new(&mut range[1]).speed(0.1));
                        }
                    });
                }
                if mapping.validate().is_ok() {
                    self.color_property_key = mapping.property().unwrap_or_default().to_string();
                    self.color_mapping = mapping;
                }
            }
        }
        ui.horizontal(|ui| {
            ui.add(
                egui::DragValue::new(&mut self.downsample_factor)
                    .speed(0.1)
                    .prefix("Downsample "),
            )
            .on_hover_text("Scales object coordinates by this factor (use if segmentations were generated on downsampled imagery).");
        });
        if matches!(self.color_mapping, ObjectColorMapping::Categorical { .. }) {
            let legend = self.active_color_legend_entries();
            if !legend.is_empty() {
                ui.separator();
                let property_key = self.color_property_key.clone();
                let overrides = self.color_level_overrides.entry(property_key.clone());
                let overrides = overrides.or_default();
                let rows = legend_rows(legend, &self.category_colors, |label| {
                    let style = overrides.get(label).copied().unwrap_or_default();
                    (style.visible, style.color_rgb)
                });
                let actions = ui_legend(
                    ui,
                    "mosaic_seg_objects_legend",
                    &property_key,
                    &rows,
                    &mut self.legend_ui,
                );
                for action in actions {
                    match action {
                        LegendAction::SetVisible { label, visible } => {
                            overrides.entry(label).or_default().visible = visible;
                        }
                        LegendAction::SetColor { label, rgb } => {
                            overrides.entry(label.clone()).or_default().color_rgb = None;
                            self.category_colors.set(&label, rgb);
                        }
                        LegendAction::ResetColor { label } => {
                            overrides.entry(label.clone()).or_default().color_rgb = None;
                            self.category_colors.remove(&label);
                        }
                    }
                }
            }
        } else if let ObjectColorMapping::Continuous {
            property,
            palette,
            reverse,
            ..
        } = self.render_color_mapping()
        {
            ui.separator();
            ui.label(format!("Legend: {property}"));
            let width = ui.available_width().max(80.0);
            let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 16.0), egui::Sense::hover());
            for index in 0..64 {
                let start = index as f64 / 64.0;
                let end = (index + 1) as f64 / 64.0;
                let position = if reverse { 1.0 - start } else { start };
                let rgb = palette.color_rgb(position);
                ui.painter().rect_filled(
                    egui::Rect::from_min_max(
                        egui::pos2(rect.left() + rect.width() * start as f32, rect.top()),
                        egui::pos2(rect.left() + rect.width() * end as f32, rect.bottom()),
                    ),
                    0.0,
                    egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]),
                );
            }
            if let ObjectColorMapping::Continuous {
                domain: ContinuousDomain::Fixed([minimum, maximum]),
                ..
            } = self.render_color_mapping()
            {
                ui.columns(3, |columns| {
                    columns[0].label(format!("{minimum:.4}"));
                    columns[1].vertical_centered(|ui| {
                        ui.label(format!("{:.4}", 0.5 * (minimum + maximum)));
                    });
                    columns[2].with_layout(
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            ui.label(format!("{maximum:.4}"));
                        },
                    );
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
                clear_selection_requested = true;
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

        (zoom_requested, clear_selection_requested)
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

    pub fn request_actor_load_for_visible_items(
        &mut self,
        items: &[(usize, egui::Rect, egui::Vec2, f32)],
        visible_world: egui::Rect,
    ) -> Vec<usize> {
        if !self.visible || !self.actor_load_requested.is_empty() {
            return Vec::new();
        }
        let requested = items
            .iter()
            .filter_map(|(id, world_rect, _, _)| {
                if !world_rect.intersects(visible_world) {
                    return None;
                }
                let state = self.items.get(id)?;
                (state.seg_path.is_some() && state.layer.is_none()).then_some(*id)
            })
            .collect::<Vec<_>>();
        self.actor_load_requested.extend(requested.iter().copied());
        requested
    }

    pub fn reconcile_actor_load_state(&mut self, state: &serde_json::Value) {
        let Some(items) = state.get("items").and_then(serde_json::Value::as_array) else {
            return;
        };
        for item in items {
            let Some(id) = item
                .get("item_id")
                .and_then(serde_json::Value::as_u64)
                .map(|id| id as usize)
            else {
                continue;
            };
            if item.get("loaded").and_then(serde_json::Value::as_bool) == Some(true) {
                self.actor_load_requested.remove(&id);
            } else if let Some(error) = item.get("error").and_then(serde_json::Value::as_str) {
                if let Some(local) = self.items.get_mut(&id) {
                    local.status = error.to_string();
                }
            }
        }
        if state.get("settled").and_then(serde_json::Value::as_bool) == Some(true) {
            self.actor_load_requested.clear();
        }
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
        let color_mapping = self.render_color_mapping();
        let domains_by_roi = self.continuous_domains_by_roi.clone();
        let color_level_overrides = self.current_color_level_overrides().clone();
        // Taken from the previous frame, because this frame's work is only known after drawing.
        let scene_outline_records = visible_items
            .iter()
            .filter(|(_, world_rect, _, _)| world_rect.intersects(visible_world))
            .filter_map(|(item_id, _, _, _)| self.items.get(item_id)?.layer.as_ref())
            .map(|layer| layer.outline_frame_stats().visible_records)
            .sum::<usize>();
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
            layer.set_outline_scene_records(scene_outline_records);
            layer.set_display_transform(mosaic_transform(*offset, *scale));
            let item_color_mapping = color_mapping_with_domain(
                &color_mapping,
                domains_by_roi.get(st.roi_id.as_str()).copied(),
            );
            apply_style(layer, style, &item_color_mapping, &color_level_overrides);
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
        let color_mapping = self.render_color_mapping();
        let domains_by_roi = self.continuous_domains_by_roi.clone();
        let color_level_overrides = self.current_color_level_overrides().clone();
        let st = self.items.get_mut(&item_id)?;
        let layer = st.layer.as_mut()?;
        let item_color_mapping = color_mapping_with_domain(
            &color_mapping,
            domains_by_roi.get(st.roi_id.as_str()).copied(),
        );
        apply_style(layer, style, &item_color_mapping, &color_level_overrides);
        layer.hover_tooltip(pointer_world, egui::Vec2::ZERO, camera)
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

    pub fn control_loading_snapshot(&self) -> serde_json::Value {
        let gpu = self.object_fill_renderer.stats();
        let mut total = 0usize;
        let mut loaded = 0usize;
        let mut layer_allocated = 0usize;
        let mut missing_layer = 0usize;
        let mut loading_data = 0usize;
        let mut loading_properties = 0usize;
        let mut analyzing = 0usize;
        let mut bulk_measuring = 0usize;
        let mut busy_statuses = Vec::new();
        let mut resident_lazy_columns = 0usize;
        let mut resident_lazy_column_bytes = 0u64;
        let mut total_loaded_column_bytes = 0u64;
        let mut property_cache_evictions = 0u64;
        let mut property_loads_in_flight = 0usize;
        let mut property_loading_estimated_bytes = 0u64;
        let mut property_peak_loading_estimated_bytes = 0u64;
        let mut property_loads_started = 0u64;
        let mut property_loads_completed = 0u64;
        let mut property_loads_cancelled = 0u64;
        let mut property_stale_results_dropped = 0u64;
        let mut property_cache_items = Vec::new();
        let mut outline_gpu = crate::render::line_bins_gl::ObjectLineBinsGlStats::default();
        let mut outline_frame = crate::objects::ObjectOutlineFrameStats::default();
        let mut presentation_state = crate::objects::ObjectPresentationStateStats::default();
        for (item_id, st) in &self.items {
            if st.seg_path.is_none() {
                continue;
            }
            total += 1;
            let Some(layer) = st.layer.as_ref() else {
                missing_layer += 1;
                continue;
            };
            layer_allocated += 1;
            let property_cache = layer.lazy_property_cache_snapshot();
            resident_lazy_columns += property_cache["resident_lazy_columns"]
                .as_u64()
                .unwrap_or(0) as usize;
            resident_lazy_column_bytes = resident_lazy_column_bytes.saturating_add(
                property_cache["resident_lazy_column_bytes"]
                    .as_u64()
                    .unwrap_or(0),
            );
            total_loaded_column_bytes = total_loaded_column_bytes.saturating_add(
                property_cache["total_loaded_column_bytes"]
                    .as_u64()
                    .unwrap_or(0),
            );
            property_cache_evictions = property_cache_evictions
                .saturating_add(property_cache["evictions"].as_u64().unwrap_or(0));
            property_loads_in_flight +=
                usize::from(property_cache["loading"].as_bool().unwrap_or(false));
            property_loading_estimated_bytes = property_loading_estimated_bytes.saturating_add(
                property_cache["loading_estimated_bytes"]
                    .as_u64()
                    .unwrap_or(0),
            );
            property_peak_loading_estimated_bytes = property_peak_loading_estimated_bytes
                .saturating_add(
                    property_cache["peak_loading_estimated_bytes"]
                        .as_u64()
                        .unwrap_or(0),
                );
            property_loads_started = property_loads_started
                .saturating_add(property_cache["loads_started"].as_u64().unwrap_or(0));
            property_loads_completed = property_loads_completed
                .saturating_add(property_cache["loads_completed"].as_u64().unwrap_or(0));
            property_loads_cancelled = property_loads_cancelled
                .saturating_add(property_cache["loads_cancelled"].as_u64().unwrap_or(0));
            property_stale_results_dropped = property_stale_results_dropped.saturating_add(
                property_cache["stale_results_dropped"]
                    .as_u64()
                    .unwrap_or(0),
            );
            property_cache_items.push(serde_json::json!({
                "item_id": item_id,
                "cache": property_cache,
            }));
            outline_gpu.merge(layer.outline_gpu_stats());
            outline_frame.merge(layer.outline_frame_stats());
            presentation_state.merge(layer.object_presentation_state_stats());
            loaded += usize::from(layer.has_data());
            loading_data += usize::from(layer.is_loading());
            loading_properties += usize::from(layer.is_property_loading());
            analyzing += usize::from(layer.is_analyzing());
            bulk_measuring += usize::from(layer.is_bulk_measuring());
            if layer.is_busy() && busy_statuses.len() < 12 {
                busy_statuses.push(serde_json::json!({
                    "item_id": item_id,
                    "status": layer.status(),
                    "loading_data": layer.is_loading(),
                    "loading_properties": layer.is_property_loading(),
                    "analyzing": layer.is_analyzing(),
                    "bulk_measuring": layer.is_bulk_measuring(),
                }));
            }
        }
        serde_json::json!({
            "visible": self.visible,
            "busy": self.is_busy(),
            "force_repaint_frames": self.force_repaint_frames,
            "total_with_segmentation_path": total,
            "layers_loaded": loaded,
            "layers_allocated": layer_allocated,
            "layers_without_loaded_instance": missing_layer,
            "layers_loading_data": loading_data,
            "layers_loading_properties": loading_properties,
            "layers_analyzing": analyzing,
            "layers_bulk_measuring": bulk_measuring,
            "sample_busy_statuses": busy_statuses,
            "property_cache": {
                "policy": if self.property_cache_capacity.is_some() { "lru" } else { "unbounded" },
                "capacity_per_roi": self.property_cache_capacity,
                "resident_lazy_columns": resident_lazy_columns,
                "resident_lazy_column_bytes": resident_lazy_column_bytes,
                "total_loaded_column_bytes": total_loaded_column_bytes,
                "evictions": property_cache_evictions,
                "loads_in_flight": property_loads_in_flight,
                "loading_estimated_bytes": property_loading_estimated_bytes,
                "peak_loading_estimated_bytes": property_peak_loading_estimated_bytes,
                "loads_started": property_loads_started,
                "loads_completed": property_loads_completed,
                "loads_cancelled": property_loads_cancelled,
                "stale_results_dropped": property_stale_results_dropped,
                "max_concurrent_decodes": 4,
                "items": property_cache_items,
            },
            "gpu_object_fill_pool": {
                "shared": true,
                "mesh_entries": gpu.mesh_entries,
                "mesh_bytes": gpu.mesh_bytes,
                "mesh_budget_bytes": gpu.mesh_budget_bytes,
                "state_bytes": gpu.state_bytes,
                "color_bytes": gpu.color_bytes,
                "texture_budget_bytes": gpu.texture_budget_bytes,
                "tile_entries": gpu.tile_entries,
                "tile_bytes": gpu.tile_bytes,
                "tile_pending_bytes": gpu.tile_pending_bytes,
                "tile_budget_bytes": gpu.tile_budget_bytes,
                "tile_pending": gpu.tile_pending,
                "tile_peak_pending": gpu.tile_peak_pending,
                "tile_frame_generation": gpu.tile_frame_generation,
                "tile_frame_generated": gpu.tile_frame_generated,
                "tile_frame_raster_vertices": gpu.tile_frame_raster_vertices,
                "tile_requests": gpu.tile_requests,
                "tile_generations": gpu.tile_generations,
                "tile_evictions": gpu.tile_evictions,
                "last_tile_raster_draw_calls": gpu.last_tile_raster_draw_calls,
                "last_tile_compose_draw_calls": gpu.last_tile_compose_draw_calls,
                "last_tile_multisample_compose_draw_calls": gpu.last_tile_multisample_compose_draw_calls,
                "last_tile_border_compose_draw_calls": gpu.last_tile_border_compose_draw_calls,
                "last_tile_raster_ms": gpu.last_tile_raster_ms,
                "last_tile_compose_ms": gpu.last_tile_compose_ms,
            },
            "gpu_object_outline_cache": {
                "shared": false,
                "layer_count": layer_allocated,
                "stats": outline_gpu,
            },
            "object_outline_frame": outline_frame,
            "object_presentation_state": presentation_state,
        })
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

    fn available_numeric_properties(&mut self) -> Vec<String> {
        let mut keys = BTreeSet::new();
        if let ObjectColorMapping::Continuous { property, .. } = &self.color_mapping
            && !property.is_empty()
        {
            keys.insert(property.clone());
        }
        for state in self.items.values_mut() {
            if let Some(layer) = state.layer.as_mut() {
                keys.extend(layer.available_numeric_object_property_keys());
            }
        }
        keys.into_iter().collect()
    }

    fn render_color_mapping(&mut self) -> ObjectColorMapping {
        let ObjectColorMapping::Continuous {
            property,
            domain: ContinuousDomain::Automatic(_),
            palette,
            scale,
            reverse,
            out_of_range,
            missing_color_rgb,
        } = &self.color_mapping
        else {
            return self.color_mapping.clone();
        };
        let property = property.clone();
        let mut minimum = f64::INFINITY;
        let mut maximum = f64::NEG_INFINITY;
        for state in self.items.values_mut() {
            if let Some(domain) = state
                .layer
                .as_mut()
                .and_then(|layer| layer.numeric_property_domain(&property))
            {
                minimum = minimum.min(domain[0]);
                maximum = maximum.max(domain[1]);
            }
        }
        if !minimum.is_finite() || !maximum.is_finite() {
            return self.color_mapping.clone();
        }
        if minimum >= maximum {
            let epsilon = minimum.abs().max(1.0) * 1.0e-9;
            minimum -= epsilon;
            maximum += epsilon;
        }
        ObjectColorMapping::Continuous {
            property,
            palette: palette.clone(),
            domain: ContinuousDomain::Fixed([minimum, maximum]),
            scale: *scale,
            reverse: *reverse,
            out_of_range: *out_of_range,
            missing_color_rgb: *missing_color_rgb,
        }
    }

    /// Keeps every ROI layer on the same project-wide colours, including edits not yet committed.
    pub(in crate::mosaic) fn sync_category_colors(
        &mut self,
        project: &std::collections::BTreeMap<String, [u8; 3]>,
    ) {
        self.category_colors.sync(project);
        for st in self.items.values_mut() {
            if let Some(layer) = st.layer.as_mut() {
                layer.sync_category_colors(self.category_colors.colors());
            }
        }
    }

    pub(in crate::mosaic) fn take_category_color_commit(
        &mut self,
        pointer_down: bool,
    ) -> Option<serde_json::Value> {
        self.category_colors
            .take_commit(pointer_down)
            .map(|colors| category_colors_params(&colors))
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
            fast_rendering: self.fast_rendering,
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

/// Compares paths by what they resolve to. Resolving touches the filesystem, which may be a slow
/// external or network drive on the UI thread, so each successful result is remembered.
fn paths_match(
    a: &std::path::Path,
    b: &std::path::Path,
    canonical_paths: &mut HashMap<PathBuf, PathBuf>,
) -> bool {
    if a == b {
        return true;
    }
    match (
        canonical_path(a, canonical_paths),
        canonical_path(b, canonical_paths),
    ) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

fn canonical_path(
    path: &std::path::Path,
    canonical_paths: &mut HashMap<PathBuf, PathBuf>,
) -> Option<PathBuf> {
    if let Some(canonical) = canonical_paths.get(path) {
        return Some(canonical.clone());
    }
    let canonical = path.canonicalize().ok()?;
    canonical_paths.insert(path.to_path_buf(), canonical.clone());
    Some(canonical)
}

static EMPTY_COLOR_LEVEL_OVERRIDES: std::sync::LazyLock<HashMap<String, ObjectColorLevelOverride>> =
    std::sync::LazyLock::new(HashMap::new);

fn mosaic_transform(offset: egui::Vec2, scale: f32) -> SpatialDataTransform2 {
    SpatialDataTransform2 {
        scale: [scale, scale],
        translation: [offset.x, offset.y],
        affine: None,
    }
}

fn color_mapping_with_domain(
    mapping: &ObjectColorMapping,
    domain: Option<[f64; 2]>,
) -> ObjectColorMapping {
    let Some(domain) = domain else {
        return mapping.clone();
    };
    let ObjectColorMapping::Continuous {
        property,
        palette,
        scale,
        reverse,
        out_of_range,
        missing_color_rgb,
        ..
    } = mapping
    else {
        return mapping.clone();
    };
    ObjectColorMapping::Continuous {
        property: property.clone(),
        palette: palette.clone(),
        domain: ContinuousDomain::Fixed(domain),
        scale: *scale,
        reverse: *reverse,
        out_of_range: *out_of_range,
        missing_color_rgb: *missing_color_rgb,
    }
}

fn apply_style(
    layer: &mut ObjectsLayer,
    style: SharedStyle,
    color_mapping: &ObjectColorMapping,
    color_level_overrides: &HashMap<String, ObjectColorLevelOverride>,
) {
    layer.visible = style.visible;
    layer.opacity = style.opacity;
    layer.width_screen_px = style.width_screen_px;
    layer.color_rgb = style.color_rgb;
    layer.fill_cells = style.fill_cells;
    layer.fill_opacity = style.fill_opacity;
    layer.selected_fill_opacity = style.selected_fill_opacity;
    layer.fast_rendering = style.fast_rendering;
    layer.downsample_factor = style.downsample_factor;
    let _ = layer.set_color_mapping(color_mapping.clone());
    layer.set_color_level_overrides(color_mapping.property(), color_level_overrides);
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir()
                .join(format!("odon-seg-overlay-{name}-{}", std::process::id()));
            std::fs::create_dir_all(&path).expect("create test directory");
            Self(path)
        }

        fn preload(&self, file: &str) -> (PathBuf, Arc<PreloadedObjectLayer>) {
            let path = self.0.join(file);
            let fixture = serde_json::json!({
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "id": "cell",
                    "properties": {"class": "tumour"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]
                    }
                }]
            });
            std::fs::write(&path, fixture.to_string()).expect("write object fixture");
            let resource = crate::objects::load_control_object_resource(path.clone(), 1.0)
                .expect("load object fixture");
            let preloaded = resource
                .renderer_payload::<PreloadedObjectLayer>()
                .expect("object resource carries a renderer payload")
                .clone();
            (path, Arc::new(preloaded))
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn overlay_with_items(paths: &[&PathBuf]) -> MosaicGeoJsonSegmentationOverlay {
        let mut overlay = MosaicGeoJsonSegmentationOverlay::default();
        for (item_id, path) in paths.iter().enumerate() {
            overlay.items.insert(
                item_id,
                ItemState {
                    roi_id: format!("ROI{item_id}"),
                    seg_path: Some((*path).clone()),
                    ..ItemState::default()
                },
            );
        }
        overlay
    }

    /// A value `install_preloaded` resets, so surviving proves the layer was not replaced.
    const UNTOUCHED_MARKER: f32 = 0.123;

    fn mark_layer(overlay: &mut MosaicGeoJsonSegmentationOverlay, item_id: usize) {
        overlay
            .items
            .get_mut(&item_id)
            .unwrap()
            .layer
            .as_mut()
            .unwrap()
            .opacity = UNTOUCHED_MARKER;
    }

    fn layer_is_marked(overlay: &MosaicGeoJsonSegmentationOverlay, item_id: usize) -> bool {
        overlay.items[&item_id].layer.as_ref().unwrap().opacity == UNTOUCHED_MARKER
    }

    #[test]
    fn repeated_projections_keep_the_installed_layer() {
        let dir = TestDir::new("repeat");
        let (path, preloaded) = dir.preload("a.geojson");
        let mut overlay = overlay_with_items(&[&path]);

        assert_eq!(overlay.install_preloaded(&path, &preloaded), 1);
        mark_layer(&mut overlay, 0);

        assert_eq!(overlay.install_preloaded(&path, &preloaded), 0);
        assert!(layer_is_marked(&overlay, 0));
    }

    #[test]
    fn a_changed_preload_replaces_only_the_matching_layer() {
        let dir = TestDir::new("changed");
        let (path_a, preloaded_a) = dir.preload("a.geojson");
        let (path_b, preloaded_b) = dir.preload("b.geojson");
        let mut overlay = overlay_with_items(&[&path_a, &path_b]);
        overlay.install_preloaded(&path_a, &preloaded_a);
        overlay.install_preloaded(&path_b, &preloaded_b);
        mark_layer(&mut overlay, 0);
        mark_layer(&mut overlay, 1);

        let (_, reloaded_a) = dir.preload("a.geojson");

        assert_eq!(overlay.install_preloaded(&path_a, &reloaded_a), 1);
        assert!(!layer_is_marked(&overlay, 0));
        assert!(layer_is_marked(&overlay, 1));
    }

    #[test]
    fn items_without_a_layer_still_receive_an_already_seen_preload() {
        let dir = TestDir::new("late-item");
        let (path, preloaded) = dir.preload("a.geojson");
        let mut overlay = overlay_with_items(&[&path]);
        overlay.install_preloaded(&path, &preloaded);
        overlay.items.insert(
            1,
            ItemState {
                roi_id: "ROI1".to_string(),
                seg_path: Some(path.clone()),
                ..ItemState::default()
            },
        );

        assert_eq!(overlay.install_preloaded(&path, &preloaded), 1);
        assert!(overlay.items[&1].layer.is_some());
    }

    #[test]
    fn loading_other_data_forgets_the_installed_preload() {
        let dir = TestDir::new("forget");
        let (_, preloaded) = dir.preload("a.geojson");
        let (_, other) = dir.preload("b.geojson");
        let mut layer = ObjectsLayer::default();

        assert!(layer.install_shared_preload(&preloaded));
        assert!(layer.has_shared_preload(&preloaded));
        assert!(!layer.has_shared_preload(&other));

        layer.install_preloaded(other.as_ref());
        assert!(!layer.has_shared_preload(&preloaded));
        assert!(layer.install_shared_preload(&preloaded));
    }

    #[test]
    fn differently_spelled_paths_match_and_resolved_paths_are_remembered() {
        let dir = TestDir::new("paths");
        let (path, _) = dir.preload("a.geojson");
        std::fs::create_dir_all(dir.0.join("sub")).expect("create subdirectory");
        let spelled_differently = dir.0.join("sub").join("..").join("a.geojson");
        let (other, _) = dir.preload("b.geojson");
        let mut canonical_paths = HashMap::new();

        assert!(paths_match(&path, &path, &mut canonical_paths));
        assert!(
            canonical_paths.is_empty(),
            "identical paths need no filesystem access"
        );
        assert!(paths_match(
            &spelled_differently,
            &path,
            &mut canonical_paths
        ));
        assert!(!paths_match(&other, &path, &mut canonical_paths));
        assert_eq!(canonical_paths.len(), 3);

        std::fs::remove_file(&path).expect("remove fixture");
        assert!(
            paths_match(&spelled_differently, &path, &mut canonical_paths),
            "a remembered result must not touch the filesystem again"
        );
    }

    #[test]
    fn missing_files_do_not_match_a_different_path() {
        let mut canonical_paths = HashMap::new();
        assert!(!paths_match(
            std::path::Path::new("/odon-test-missing/a"),
            std::path::Path::new("/odon-test-missing/b"),
            &mut canonical_paths,
        ));
    }

    #[test]
    fn per_roi_domain_replaces_only_the_continuous_domain() {
        let mapping = ObjectColorMapping::Continuous {
            property: "median_raw__cd3".to_string(),
            palette: ContinuousPalette::Named("viridis".to_string()),
            domain: ContinuousDomain::Fixed([0.0, 1_000.0]),
            scale: ContinuousScale::Linear,
            reverse: true,
            out_of_range: OutOfRangeMode::Clamp,
            missing_color_rgb: Some([40, 40, 40]),
        };

        let overridden = color_mapping_with_domain(&mapping, Some([10.0, 20.0]));

        let ObjectColorMapping::Continuous {
            property,
            domain,
            reverse,
            missing_color_rgb,
            ..
        } = overridden
        else {
            panic!("continuous mapping should remain continuous");
        };
        assert_eq!(property, "median_raw__cd3");
        assert_eq!(domain, ContinuousDomain::Fixed([10.0, 20.0]));
        assert!(reverse);
        assert_eq!(missing_color_rgb, Some([40, 40, 40]));
        assert_eq!(color_mapping_with_domain(&mapping, None), mapping);
    }
}
