use super::*;

impl OmeZarrViewerApp {
    pub(super) fn set_active_layer(&mut self, id: LayerId) {
        self.active_layer = id;
        if self
            .selected_mask_polygon
            .is_some_and(|selection| id != LayerId::Mask(selection.layer_id))
        {
            self.clear_mask_polygon_selection();
        }
        if let LayerId::Channel(idx) = id {
            self.selected_channel = idx.min(self.channels.len().saturating_sub(1));
            self.hist_dirty = true;
        } else {
            self.selected_channel_group_id = None;
        }
    }

    pub(super) fn channel_indices_in_group(&self, group_id: u64) -> Vec<usize> {
        let groups = self.current_layer_groups();
        self.channel_layer_order
            .iter()
            .copied()
            .filter(|&idx| {
                self.channels.get(idx).is_some_and(|ch| {
                    groups
                        .channel_members
                        .get(ch.name.as_str())
                        .is_some_and(|m| m.group_id == group_id)
                })
            })
            .collect()
    }

    pub(super) fn group_contrast_window_for_indices(
        &self,
        indices: &[usize],
        abs_max: f32,
    ) -> Option<((f32, f32), bool)> {
        let mut first_window: Option<(f32, f32)> = None;
        let mut mixed = false;
        for &idx in indices {
            let Some(ch) = self.channels.get(idx) else {
                continue;
            };
            let window = ch.window.unwrap_or((0.0, abs_max));
            if let Some(prev) = first_window {
                if (prev.0 - window.0).abs() > 1e-6 || (prev.1 - window.1).abs() > 1e-6 {
                    mixed = true;
                }
            } else {
                first_window = Some(window);
            }
        }
        first_window.map(|window| (window, mixed))
    }

    pub(super) fn visible_channel_indices(&self) -> Vec<usize> {
        self.channel_layer_order
            .iter()
            .copied()
            .filter(|&idx| self.channels.get(idx).is_some_and(|ch| ch.visible))
            .collect()
    }

    pub(super) fn quick_contrast_target_options(&self) -> Vec<top_bar::QuickContrastTargetOption> {
        let visible_count = self.visible_channel_indices().len();
        let group_count = self
            .selected_channel_group_id
            .map(|group_id| self.channel_indices_in_group(group_id).len())
            .unwrap_or(0);
        let group_label = self
            .selected_channel_group_id
            .and_then(|group_id| {
                self.current_layer_groups()
                    .channel_groups
                    .iter()
                    .find(|group| group.id == group_id)
                    .map(|group| format!("Selected group ({})", group.name))
            })
            .unwrap_or_else(|| "Selected group".to_string());

        vec![
            top_bar::QuickContrastTargetOption {
                target: top_bar::QuickContrastTarget::Visible,
                label: format!("Visible channels ({visible_count})"),
                enabled: visible_count > 0,
            },
            top_bar::QuickContrastTargetOption {
                target: top_bar::QuickContrastTarget::Active,
                label: "Active channel".to_string(),
                enabled: !self.channels.is_empty(),
            },
            top_bar::QuickContrastTargetOption {
                target: top_bar::QuickContrastTarget::SelectedGroup,
                label: format!("{group_label} ({group_count})"),
                enabled: group_count > 0,
            },
        ]
    }

    pub(super) fn quick_contrast_indices_for_target(
        &self,
        target: top_bar::QuickContrastTarget,
    ) -> Vec<usize> {
        match target {
            top_bar::QuickContrastTarget::Active => {
                if self.channels.is_empty() {
                    Vec::new()
                } else {
                    vec![self.selected_channel.min(self.channels.len() - 1)]
                }
            }
            top_bar::QuickContrastTarget::Visible => {
                let visible = self.visible_channel_indices();
                if visible.is_empty() {
                    self.quick_contrast_indices_for_target(top_bar::QuickContrastTarget::Active)
                } else {
                    visible
                }
            }
            top_bar::QuickContrastTarget::SelectedGroup => self
                .selected_channel_group_id
                .map(|group_id| self.channel_indices_in_group(group_id))
                .filter(|indices| !indices.is_empty())
                .unwrap_or_else(|| {
                    self.quick_contrast_indices_for_target(top_bar::QuickContrastTarget::Visible)
                }),
        }
    }

    pub(super) fn apply_channel_window_to_indices(&mut self, indices: &[usize], lo: f32, hi: f32) {
        let abs_max = self.dataset.abs_max.max(1.0);
        let lo = lo.clamp(0.0, abs_max);
        let hi = hi.clamp(0.0, abs_max);
        let (lo, hi) = if hi <= lo {
            ((hi - 1.0).clamp(0.0, abs_max), hi)
        } else {
            (lo, hi)
        };
        let mut changed = false;
        for &idx in indices {
            if let Some(dst) = self.channels.get_mut(idx) {
                dst.window = Some((lo, hi));
                self.channel_window_overrides
                    .insert(dst.name.clone(), (lo, hi));
                changed = true;
            }
        }
        if changed {
            self.hist_dirty = true;
            self.bump_render_id();
        }
    }

    pub(super) fn apply_three_channel_rgb_preset(&mut self) -> bool {
        if self.channels.len() != 3 {
            return false;
        }
        let rgb = [[255, 0, 0], [0, 255, 0], [0, 0, 255]];
        let hi = self.dataset.abs_max.clamp(1.0, 255.0);
        let mut changed = false;
        for (idx, color) in rgb.into_iter().enumerate() {
            let Some(channel) = self.channels.get_mut(idx) else {
                continue;
            };
            changed |= channel.color_rgb != color;
            channel.color_rgb = color;
            changed |= !channel.visible;
            channel.visible = true;
            let window = (0.0, hi);
            changed |= channel.window != Some(window);
            channel.window = Some(window);
            self.channel_window_overrides
                .insert(channel.name.clone(), window);
            self.selected_channel_layers.insert(idx);
            self.memory_selected_channels.insert(idx);
            self.set_channel_group_color_inheritance(idx, false);
        }
        if !changed {
            return false;
        }
        self.selected_channel = 0;
        self.active_layer = LayerId::Channel(0);
        self.selected_channel_group_id = None;
        self.channel_select_anchor_idx = Some(0);
        self.hist_dirty = true;
        self.bump_render_id();
        self.set_status("Applied RGB preset to channels 0-2.");
        true
    }

    pub(super) fn ui_top_bar_quick_contrast(&mut self, ui: &mut egui::Ui) {
        if self.channels.is_empty() {
            return;
        }
        if self.quick_contrast_target == top_bar::QuickContrastTarget::SelectedGroup
            && self
                .selected_channel_group_id
                .map(|group_id| self.channel_indices_in_group(group_id).is_empty())
                .unwrap_or(true)
        {
            self.quick_contrast_target = top_bar::QuickContrastTarget::Visible;
        }

        let options = self.quick_contrast_target_options();
        let indices = self.quick_contrast_indices_for_target(self.quick_contrast_target);
        if indices.is_empty() {
            return;
        }
        let abs_max = self.dataset.abs_max.max(1.0);
        let ((window, mixed), reference_idx) = (
            self.group_contrast_window_for_indices(&indices, abs_max)
                .unwrap_or(((0.0, abs_max), false)),
            self.selected_channel.min(self.channels.len() - 1),
        );
        let reference_name = self
            .channels
            .get(reference_idx)
            .map(|channel| channel.name.clone())
            .unwrap_or_else(|| "channel".to_string());
        let target_before = self.quick_contrast_target;
        let response = top_bar::ui_quick_contrast(
            ui,
            top_bar::QuickContrastParams {
                abs_max,
                target: &mut self.quick_contrast_target,
                target_options: &options,
                target_count: indices.len(),
                reference_channel_name: &reference_name,
                window,
                mixed,
                step: 1.0,
                id_salt: "top-quick-contrast",
            },
        );
        if response.changed && self.quick_contrast_target == target_before {
            let target_indices = self.quick_contrast_indices_for_target(self.quick_contrast_target);
            self.apply_channel_window_to_indices(
                &target_indices,
                response.window.0,
                response.window.1,
            );
        }
    }

    pub(super) fn ui_group_contrast(
        &mut self,
        _ctx: &egui::Context,
        ui: &mut egui::Ui,
        group_id: u64,
    ) {
        let abs_max = self.dataset.abs_max.max(1.0);
        let Some(group) = self
            .current_layer_groups()
            .channel_groups
            .iter()
            .find(|g| g.id == group_id)
            .cloned()
        else {
            self.selected_channel_group_id = None;
            ui.label("Selected channel group no longer exists.");
            return;
        };

        let members = self.channel_indices_in_group(group_id);
        ui.heading("Contrast");
        ui.label(format!("Group: {}", group.name));
        ui.label(format!("Applies to {} channel(s).", members.len()));

        if members.is_empty() {
            ui.label("This group has no channels.");
            return;
        }

        let Some((window, mixed)) = self.group_contrast_window_for_indices(&members, abs_max)
        else {
            ui.label("No channels available in this group.");
            return;
        };
        if mixed {
            ui.label("Group channels currently have mixed contrast limits. Applying changes here will overwrite them.");
        }

        let out = contrast::ui_contrast_window(
            ui,
            abs_max,
            window,
            contrast::ContrastUiOptions::standard("Set Max -> Group"),
        );
        let (lo, hi) = out.window;

        if out.set_max_all_clicked {
            for &idx in &members {
                if let Some(dst) = self.channels.get_mut(idx) {
                    let (mut dlo, _) = dst.window.unwrap_or((0.0, abs_max));
                    dlo = dlo.clamp(0.0, abs_max);
                    let dhi = hi.clamp(0.0, abs_max);
                    let dlo = if dhi <= dlo {
                        (dhi - 1.0).clamp(0.0, abs_max)
                    } else {
                        dlo
                    };
                    dst.window = Some((dlo, dhi));
                    self.channel_window_overrides
                        .insert(dst.name.clone(), (dlo, dhi));
                }
            }
            self.bump_render_id();
            return;
        }

        if out.limits_touched {
            for &idx in &members {
                if let Some(dst) = self.channels.get_mut(idx) {
                    dst.window = Some((lo, hi));
                    self.channel_window_overrides
                        .insert(dst.name.clone(), (lo, hi));
                }
            }
            self.bump_render_id();
        }
    }

    pub(super) fn add_annotation_layer(&mut self) {
        let id = self.next_annotation_layer_id.max(1);
        self.next_annotation_layer_id = id.wrapping_add(1).max(1);
        let name = format!("Annotations {id}");
        self.annotation_layers
            .push(AnnotationPointsLayer::new(id, name));
        self.set_active_layer(LayerId::Annotation(id));
        self.rebuild_layer_orders();
    }

    pub fn add_annotation_layer_from_menu(&mut self) {
        self.add_annotation_layer();
    }

    pub(super) fn queue_object_source_action(
        &mut self,
        action: crate::objects::ObjectSourceUiAction,
    ) {
        let (method, params) = match action {
            crate::objects::ObjectSourceUiAction::Load { path, options } => {
                let path = if options.is_some() {
                    path
                } else {
                    let Some(path) = self.seg_objects.prepare_source_path(path) else {
                        return;
                    };
                    path
                };
                let mut params = serde_json::json!({
                    "path": path,
                    "downsample_factor": self.seg_objects.downsample_factor,
                });
                if let Some(options) = options {
                    params
                        .as_object_mut()
                        .expect("object source params are an object")
                        .insert("loader_options".to_string(), options);
                }
                ("viewer.objects.source.load", params)
            }
            crate::objects::ObjectSourceUiAction::Reload => {
                ("viewer.objects.source.reload", serde_json::json!({}))
            }
            crate::objects::ObjectSourceUiAction::Clear => {
                ("viewer.objects.source.clear", serde_json::json!({}))
            }
        };
        self.native_control_intents
            .push(NativeControlIntent { method, params });
    }

    pub fn open_seg_geojson_dialog(&mut self) {
        let default_dir = self
            .dataset
            .source
            .local_path()
            .and_then(|p| p.parent())
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        self.seg_geojson.open_dialog(&default_dir);
    }

    pub fn open_seg_objects_dialog(&mut self) {
        let default_dir = self
            .dataset
            .source
            .local_path()
            .and_then(|p| p.parent())
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        if let Some(path) = self.seg_objects.choose_source_dialog(&default_dir) {
            self.queue_object_source_action(crate::objects::ObjectSourceUiAction::Load {
                path,
                options: None,
            });
        }
    }

    pub(super) fn layer_is_available(&self, id: LayerId) -> bool {
        match id {
            LayerId::SegmentationLabels => self.tiles_gl.is_some(),
            _ => true,
        }
    }

    pub(super) fn layer_display_name(&self, id: LayerId) -> String {
        match id {
            LayerId::Channel(idx) => self
                .channels
                .get(idx)
                .map(|c| c.name.clone())
                .unwrap_or_else(|| format!("Channel {idx}")),
            LayerId::SpatialImage(id) => self
                .spatial_image_layers
                .images
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.name.clone())
                .unwrap_or_else(|| format!("Image {id}")),
            LayerId::SegmentationLabels => {
                let name = self.seg_label_selected.trim();
                if name.is_empty() {
                    "Segmentation labels".to_string()
                } else {
                    format!("Segmentation ({name})")
                }
            }
            LayerId::SegmentationGeoJson => "Segmentation (GeoJSON)".to_string(),
            LayerId::SegmentationObjects => "Segmentation (Objects)".to_string(),
            LayerId::Mask(id) => self
                .mask_layers
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.name.clone())
                .unwrap_or_else(|| format!("Mask {id}")),
            LayerId::Points => "Points".to_string(),
            LayerId::Annotation(id) => self
                .annotation_layers
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.name.clone())
                .unwrap_or_else(|| format!("Annotations {id}")),
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter()
                .find(|s| s.id == id)
                .map(|s| s.name.clone())
                .unwrap_or_else(|| format!("Shapes {id}")),
            LayerId::SpatialPoints => self
                .spatial_layers
                .points
                .as_ref()
                .map(|p| p.name.clone())
                .unwrap_or_else(|| "Points (SpatialData)".to_string()),
            LayerId::XeniumCells => self
                .xenium_layers
                .cells
                .as_ref()
                .map(|c| c.name.clone())
                .unwrap_or_else(|| "Cells (Xenium)".to_string()),
            LayerId::XeniumTranscripts => self
                .xenium_layers
                .transcripts
                .as_ref()
                .map(|t| t.name.clone())
                .unwrap_or_else(|| "Transcripts (Xenium)".to_string()),
        }
    }

    pub(super) fn layer_icon(&self, id: LayerId) -> Icon {
        match id {
            LayerId::Channel(_) => Icon::Image,
            LayerId::SpatialImage(_) => Icon::Image,
            LayerId::Points => Icon::Points,
            LayerId::Annotation(_) => Icon::Points,
            LayerId::SpatialPoints => Icon::Points,
            LayerId::XeniumTranscripts => Icon::Points,
            LayerId::SegmentationLabels
            | LayerId::SegmentationGeoJson
            | LayerId::SegmentationObjects
            | LayerId::Mask(_)
            | LayerId::SpatialShape(_)
            | LayerId::XeniumCells => Icon::Polygon,
        }
    }

    pub(super) fn layer_visible_mut(&mut self, id: LayerId) -> Option<&mut bool> {
        match id {
            LayerId::Channel(idx) => self.channels.get_mut(idx).map(|c| &mut c.visible),
            LayerId::SpatialImage(id) => self
                .spatial_image_layers
                .images
                .iter_mut()
                .find(|l| l.id == id)
                .map(|l| &mut l.visible),
            LayerId::SegmentationLabels => Some(&mut self.cells_outlines_visible),
            LayerId::SegmentationGeoJson => Some(&mut self.seg_geojson.visible),
            LayerId::SegmentationObjects => Some(&mut self.seg_objects.visible),
            LayerId::Mask(id) => self
                .mask_layers
                .iter_mut()
                .find(|l| l.id == id)
                .map(|l| &mut l.visible),
            LayerId::Points => Some(&mut self.cell_points.visible),
            LayerId::Annotation(id) => self
                .annotation_layers
                .iter_mut()
                .find(|l| l.id == id)
                .map(|l| &mut l.visible),
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|s| s.id == id)
                .map(|s| s.visible_mut()),
            LayerId::SpatialPoints => self.spatial_layers.points.as_mut().map(|p| &mut p.visible),
            LayerId::XeniumCells => self.xenium_layers.cells.as_mut().map(|c| &mut c.visible),
            LayerId::XeniumTranscripts => self
                .xenium_layers
                .transcripts
                .as_mut()
                .map(|t| &mut t.visible),
        }
    }

    pub(super) fn layer_visible_value(&self, id: LayerId) -> Option<bool> {
        match id {
            LayerId::Channel(idx) => self.channels.get(idx).map(|c| c.visible),
            LayerId::SpatialImage(id) => self
                .spatial_image_layers
                .images
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.visible),
            LayerId::SegmentationLabels => Some(self.cells_outlines_visible),
            LayerId::SegmentationGeoJson => Some(self.seg_geojson.visible),
            LayerId::SegmentationObjects => Some(self.seg_objects.visible),
            LayerId::Mask(id) => self
                .mask_layers
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.visible),
            LayerId::Points => Some(self.cell_points.visible),
            LayerId::Annotation(id) => self
                .annotation_layers
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.visible),
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter()
                .find(|s| s.id == id)
                .map(|s| s.visible),
            LayerId::SpatialPoints => self.spatial_layers.points.as_ref().map(|p| p.visible),
            LayerId::XeniumCells => self.xenium_layers.cells.as_ref().map(|c| c.visible),
            LayerId::XeniumTranscripts => {
                self.xenium_layers.transcripts.as_ref().map(|t| t.visible)
            }
        }
    }

    pub(super) fn layer_offset_world(&self, id: LayerId) -> egui::Vec2 {
        match id {
            LayerId::Channel(idx) => self
                .channel_offsets_world
                .get(idx)
                .copied()
                .unwrap_or(egui::Vec2::ZERO),
            LayerId::SpatialImage(id) => self
                .spatial_image_layers
                .images
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.offset_world)
                .unwrap_or(egui::Vec2::ZERO),
            LayerId::Points => self.points_offset_world,
            LayerId::Annotation(id) => self
                .annotation_layers
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.offset_world)
                .unwrap_or(egui::Vec2::ZERO),
            LayerId::SpatialPoints => self.spatial_points_offset_world,
            LayerId::Mask(id) => self
                .mask_layers
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.offset_world)
                .unwrap_or(egui::Vec2::ZERO),
            LayerId::SegmentationLabels => self.seg_labels_offset_world,
            LayerId::SegmentationGeoJson => self.seg_geojson_offset_world,
            LayerId::SegmentationObjects => self.seg_objects_offset_world,
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter()
                .find(|s| s.id == id)
                .map(|s| s.offset_world)
                .unwrap_or(egui::Vec2::ZERO),
            LayerId::XeniumCells => self.xenium_cells_offset_world,
            LayerId::XeniumTranscripts => self.xenium_transcripts_offset_world,
        }
    }

    pub(super) fn layer_offset_world_mut(&mut self, id: LayerId) -> Option<&mut egui::Vec2> {
        match id {
            LayerId::Channel(idx) => self.channel_offsets_world.get_mut(idx),
            LayerId::SpatialImage(id) => self
                .spatial_image_layers
                .images
                .iter_mut()
                .find(|l| l.id == id)
                .map(|l| &mut l.offset_world),
            LayerId::Points => Some(&mut self.points_offset_world),
            LayerId::Annotation(id) => self
                .annotation_layers
                .iter_mut()
                .find(|l| l.id == id)
                .map(|l| &mut l.offset_world),
            LayerId::SpatialPoints => Some(&mut self.spatial_points_offset_world),
            LayerId::Mask(id) => self
                .mask_layers
                .iter_mut()
                .find(|l| l.id == id)
                .map(|l| &mut l.offset_world),
            LayerId::SegmentationLabels => Some(&mut self.seg_labels_offset_world),
            LayerId::SegmentationGeoJson => Some(&mut self.seg_geojson_offset_world),
            LayerId::SegmentationObjects => Some(&mut self.seg_objects_offset_world),
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|s| s.id == id)
                .map(|s| &mut s.offset_world),
            LayerId::XeniumCells => Some(&mut self.xenium_cells_offset_world),
            LayerId::XeniumTranscripts => Some(&mut self.xenium_transcripts_offset_world),
        }
    }

    pub(super) fn layer_has_offset_world(&self, id: LayerId) -> bool {
        match id {
            LayerId::Channel(idx) => idx < self.channel_offsets_world.len(),
            LayerId::SpatialImage(id) => {
                self.spatial_image_layers.images.iter().any(|l| l.id == id)
            }
            LayerId::Points => true,
            LayerId::Annotation(id) => self.annotation_layers.iter().any(|l| l.id == id),
            LayerId::SpatialPoints => true,
            LayerId::Mask(id) => self.mask_layers.iter().any(|l| l.id == id),
            LayerId::SegmentationLabels
            | LayerId::SegmentationGeoJson
            | LayerId::SegmentationObjects
            | LayerId::XeniumCells
            | LayerId::XeniumTranscripts => true,
            LayerId::SpatialShape(id) => self.spatial_layers.shapes.iter().any(|s| s.id == id),
        }
    }

    pub(super) fn current_offset_layer_ids(&self) -> Vec<LayerId> {
        let mut layers = (0..self.channel_offsets_world.len())
            .map(LayerId::Channel)
            .collect::<Vec<_>>();
        layers.extend(self.overlay_layer_order.iter().copied());
        Self::dedupe_layer_ids(layers)
    }

    pub(super) fn capture_loaded_layer_offsets(&mut self) {
        self.loaded_layer_offsets_world.clear();
        for layer in self.current_offset_layer_ids() {
            if self.layer_has_offset_world(layer) {
                self.loaded_layer_offsets_world
                    .insert(layer, self.layer_offset_world(layer));
            }
        }
    }

    pub(super) fn ensure_loaded_layer_offset_baselines(&mut self) {
        for layer in self.current_offset_layer_ids() {
            if self.layer_has_offset_world(layer) {
                let offset = self.layer_offset_world(layer);
                self.loaded_layer_offsets_world
                    .entry(layer)
                    .or_insert(offset);
            }
        }
    }

    pub(super) fn ensure_loaded_layer_offset_baselines_for(&mut self, layers: &[LayerId]) {
        self.ensure_loaded_layer_offset_baselines();
        for &layer in layers {
            if self.layer_has_offset_world(layer) {
                let offset = self.layer_offset_world(layer);
                self.loaded_layer_offsets_world
                    .entry(layer)
                    .or_insert(offset);
            }
        }
    }

    pub(super) fn restore_loaded_layer_offsets_from_project_view(
        &mut self,
        view: &ProjectRoiViewState,
    ) {
        self.loaded_layer_offsets_world.clear();
        for (idx, saved) in view.channels.iter().enumerate() {
            if let Some([x, y]) = saved.original_offset_world {
                self.loaded_layer_offsets_world
                    .insert(LayerId::Channel(idx), egui::vec2(x, y));
            }
        }
        for (id, [x, y]) in &view.overlay_original_offsets_world {
            if let Some(layer_id) = self.parse_layer_id_storage_key(id) {
                self.loaded_layer_offsets_world
                    .insert(layer_id, egui::vec2(*x, *y));
            }
        }
        self.ensure_loaded_layer_offset_baselines();
    }

    pub(super) fn restore_loaded_layer_offsets_from_current_project_view_or_capture(&mut self) {
        let saved_view = self
            .project_space
            .roi_view_state(&self.dataset.source)
            .cloned();
        if let Some(view) = saved_view.as_ref() {
            self.restore_loaded_layer_offsets_from_project_view(view);
        } else {
            self.capture_loaded_layer_offsets();
        }
    }

    pub(super) fn dedupe_layer_ids(layers: Vec<LayerId>) -> Vec<LayerId> {
        let mut seen = HashSet::new();
        layers
            .into_iter()
            .filter(|layer| seen.insert(*layer))
            .collect()
    }

    pub(super) fn filter_visible_movable_layers(&self, layers: Vec<LayerId>) -> Vec<LayerId> {
        Self::dedupe_layer_ids(layers)
            .into_iter()
            .filter(|&layer| {
                self.layer_has_offset_world(layer)
                    && self.layer_is_available(layer)
                    && self.layer_visible_value(layer).unwrap_or(false)
            })
            .collect()
    }

    pub(super) fn current_visible_move_target_layers(&self) -> Vec<LayerId> {
        let candidates = match self.active_layer {
            LayerId::Channel(idx) => {
                if let Some(group_id) = self.selected_channel_group_id {
                    self.channel_indices_in_group(group_id)
                        .into_iter()
                        .map(LayerId::Channel)
                        .collect()
                } else if self.selected_channel_layers.contains(&idx) {
                    self.selected_channel_layers
                        .iter()
                        .copied()
                        .map(LayerId::Channel)
                        .collect()
                } else {
                    vec![LayerId::Channel(idx)]
                }
            }
            layer if self.selected_overlay_layers.contains(&layer) => {
                self.selected_overlay_layers.iter().copied().collect()
            }
            layer => vec![layer],
        };
        self.filter_visible_movable_layers(candidates)
    }

    pub(super) fn current_visible_move_targets_have_moved(&mut self) -> bool {
        let targets = self.current_visible_move_target_layers();
        if targets.is_empty() {
            return false;
        }
        self.ensure_loaded_layer_offset_baselines_for(&targets);
        targets.iter().any(|&layer| {
            self.loaded_layer_offsets_world
                .get(&layer)
                .is_some_and(|baseline| {
                    (self.layer_offset_world(layer) - *baseline).length_sq() > 1e-12
                })
        })
    }

    pub(super) fn reset_current_visible_move_targets_to_loaded(&mut self) -> bool {
        let targets = self.current_visible_move_target_layers();
        if targets.is_empty() {
            self.set_status("No visible movable layers selected.");
            return false;
        }
        self.ensure_loaded_layer_offset_baselines_for(&targets);
        let reset_offsets = targets
            .iter()
            .filter_map(|&layer| {
                self.loaded_layer_offsets_world
                    .get(&layer)
                    .copied()
                    .map(|offset_world| LayerOffsetEntry {
                        layer,
                        offset_world,
                    })
            })
            .collect::<Vec<_>>();
        let will_change = reset_offsets.iter().any(|entry| {
            (self.layer_offset_world(entry.layer) - entry.offset_world).length_sq() > 1e-12
        });
        if !will_change {
            return false;
        }
        self.push_layer_offsets_undo_snapshot(&targets);
        if self.apply_layer_offsets(&reset_offsets) {
            self.bump_render_id();
            true
        } else {
            false
        }
    }

    pub(super) fn apply_layer_offsets(&mut self, offsets: &[LayerOffsetEntry]) -> bool {
        let mut changed = false;
        let mut mask_changed = false;
        for entry in offsets {
            if let Some(offset) = self.layer_offset_world_mut(entry.layer) {
                if (*offset - entry.offset_world).length_sq() > 1e-12 {
                    *offset = entry.offset_world;
                    changed = true;
                    mask_changed |= matches!(entry.layer, LayerId::Mask(_));
                }
            }
        }
        if changed {
            self.hist_dirty = true;
        }
        if mask_changed {
            self.mark_mask_layers_project_dirty();
        }
        changed
    }

    pub(super) fn any_visible_channel_offset(&self) -> bool {
        for (i, ch) in self.channels.iter().enumerate() {
            if !ch.visible {
                continue;
            }
            let off = self
                .channel_offsets_world
                .get(i)
                .copied()
                .unwrap_or_default();
            if off.x.abs() > 1e-6 || off.y.abs() > 1e-6 {
                return true;
            }
        }
        false
    }

    pub(super) fn any_visible_channel_affine(&self) -> bool {
        for (i, ch) in self.channels.iter().enumerate() {
            if !ch.visible {
                continue;
            }
            let scale = self
                .channel_scales
                .get(i)
                .copied()
                .unwrap_or(egui::Vec2::splat(1.0));
            let rot = self.channel_rotations_rad.get(i).copied().unwrap_or(0.0);
            if (scale.x - 1.0).abs() > 1e-6 || (scale.y - 1.0).abs() > 1e-6 || rot.abs() > 1e-6 {
                return true;
            }
        }
        false
    }

    pub(super) fn union_visible_world_for_visible_channels(
        &self,
        visible_world: egui::Rect,
    ) -> egui::Rect {
        let mut min_off_x = 0.0f32;
        let mut max_off_x = 0.0f32;
        let mut min_off_y = 0.0f32;
        let mut max_off_y = 0.0f32;
        let mut any = false;
        for (i, ch) in self.channels.iter().enumerate() {
            if !ch.visible {
                continue;
            }
            let off = self
                .channel_offsets_world
                .get(i)
                .copied()
                .unwrap_or_default();
            if !any {
                min_off_x = off.x;
                max_off_x = off.x;
                min_off_y = off.y;
                max_off_y = off.y;
                any = true;
            } else {
                min_off_x = min_off_x.min(off.x);
                max_off_x = max_off_x.max(off.x);
                min_off_y = min_off_y.min(off.y);
                max_off_y = max_off_y.max(off.y);
            }
        }
        if !any {
            return visible_world;
        }

        // For a channel with offset `off`, the region of *data* that must be fetched is
        // `visible_world - off`. Union all of those to avoid missing tiles.
        egui::Rect::from_min_max(
            egui::pos2(
                visible_world.min.x - max_off_x,
                visible_world.min.y - max_off_y,
            ),
            egui::pos2(
                visible_world.max.x - min_off_x,
                visible_world.max.y - min_off_y,
            ),
        )
    }

    pub(super) fn union_visible_world_for_visible_channels_xform(
        &self,
        visible_world: egui::Rect,
    ) -> egui::Rect {
        let img_world = self.image_local_rect_lvl0();
        let pivot = img_world.center();

        let corners = [
            visible_world.left_top(),
            egui::pos2(visible_world.right(), visible_world.top()),
            visible_world.right_bottom(),
            egui::pos2(visible_world.left(), visible_world.bottom()),
        ];

        let mut acc: Option<egui::Rect> = None;
        for (i, ch) in self.channels.iter().enumerate() {
            if !ch.visible {
                continue;
            }
            let off = self
                .channel_offsets_world
                .get(i)
                .copied()
                .unwrap_or_default();
            let scale = self
                .channel_scales
                .get(i)
                .copied()
                .unwrap_or(egui::Vec2::splat(1.0));
            let rot = self.channel_rotations_rad.get(i).copied().unwrap_or(0.0);

            let mut min_x = f32::INFINITY;
            let mut max_x = f32::NEG_INFINITY;
            let mut min_y = f32::INFINITY;
            let mut max_y = f32::NEG_INFINITY;
            for &c in &corners {
                let p = inv_xform_world_point(c, pivot, off, scale, rot);
                min_x = min_x.min(p.x);
                max_x = max_x.max(p.x);
                min_y = min_y.min(p.y);
                max_y = max_y.max(p.y);
            }
            let r = egui::Rect::from_min_max(egui::pos2(min_x, min_y), egui::pos2(max_x, max_y));
            acc = Some(match acc {
                None => r,
                Some(prev) => prev.union(r),
            });
        }

        acc.unwrap_or(visible_world)
    }

    pub(super) fn rebuild_layer_orders(&mut self) {
        // Channels: retain valid indices, then append missing.
        let n = self.channels.len();
        self.channel_layer_order.retain(|&i| i < n);
        let mut seen = HashSet::new();
        self.channel_layer_order.retain(|i| seen.insert(*i));
        if self.channel_layer_order.len() != n {
            self.channel_layer_order = (0..n).collect();
        }

        let mut want: Vec<LayerId> = Vec::new();
        for layer in &self.spatial_image_layers.images {
            want.push(LayerId::SpatialImage(layer.id));
        }
        for l in &self.mask_layers {
            want.push(LayerId::Mask(l.id));
        }
        for l in &self.annotation_layers {
            want.push(LayerId::Annotation(l.id));
        }
        if self.seg_geojson.loaded_geojson.is_some() {
            want.push(LayerId::SegmentationGeoJson);
        }
        if self.seg_objects.has_data() {
            want.push(LayerId::SegmentationObjects);
        }
        if self.label_cells.is_some() {
            want.push(LayerId::SegmentationLabels);
        }
        if !self.cell_points.points.is_empty() {
            want.push(LayerId::Points);
        }
        for layer in &self.spatial_layers.shapes {
            want.push(LayerId::SpatialShape(layer.id));
        }
        if self.spatial_layers.points.is_some() {
            want.push(LayerId::SpatialPoints);
        }
        if self.xenium_layers.cells.is_some() {
            want.push(LayerId::XeniumCells);
        }
        if self.xenium_layers.transcripts.is_some() {
            want.push(LayerId::XeniumTranscripts);
        }

        let mut seen2 = HashSet::new();
        self.overlay_layer_order
            .retain(|id| want.contains(id) && seen2.insert(*id));
        for id in want {
            if !self.overlay_layer_order.contains(&id) {
                self.overlay_layer_order.push(id);
            }
        }

        if let LayerId::Channel(idx) = self.active_layer {
            if idx >= n {
                self.active_layer = if n > 0 {
                    LayerId::Channel(0)
                } else {
                    LayerId::Points
                };
            }
        }
        if matches!(
            self.active_layer,
            LayerId::SpatialImage(_)
                | LayerId::Mask(_)
                | LayerId::SegmentationGeoJson
                | LayerId::SegmentationObjects
                | LayerId::SegmentationLabels
                | LayerId::Points
                | LayerId::Annotation(_)
                | LayerId::SpatialShape(_)
                | LayerId::SpatialPoints
                | LayerId::XeniumCells
                | LayerId::XeniumTranscripts
        ) && !self.overlay_layer_order.contains(&self.active_layer)
        {
            self.active_layer = if n > 0 {
                LayerId::Channel(self.selected_channel.min(n - 1))
            } else {
                LayerId::Points
            };
        }
    }
}
