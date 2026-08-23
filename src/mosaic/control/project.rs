use super::super::*;

impl MosaicViewerApp {
    pub(in crate::mosaic) fn layer_id_storage_key(id: MosaicLayerId) -> String {
        match id {
            MosaicLayerId::TextLabels => "text_labels".to_string(),
            MosaicLayerId::SegmentationGeoJson => "segmentation_geojson".to_string(),
            MosaicLayerId::Annotation(id) => format!("annotation:{id}"),
            MosaicLayerId::Channel(idx) => format!("channel:{idx}"),
        }
    }

    pub(in crate::mosaic) fn parse_layer_id_storage_key(
        &self,
        value: &str,
    ) -> Option<MosaicLayerId> {
        if let Some(raw) = value.strip_prefix("annotation:") {
            return raw.parse::<u64>().ok().map(MosaicLayerId::Annotation);
        }
        if let Some(raw) = value.strip_prefix("channel:") {
            return raw.parse::<usize>().ok().map(MosaicLayerId::Channel);
        }
        match value {
            "text_labels" => Some(MosaicLayerId::TextLabels),
            "segmentation_geojson" => Some(MosaicLayerId::SegmentationGeoJson),
            _ => None,
        }
    }

    pub(in crate::mosaic) fn project_camera_state(&self) -> ProjectCameraState {
        ProjectCameraState {
            center_world_lvl0: [
                self.camera.center_world_lvl0.x,
                self.camera.center_world_lvl0.y,
            ],
            zoom_screen_per_lvl0_px: self.camera.zoom_screen_per_lvl0_px,
        }
    }

    pub(in crate::mosaic) fn apply_project_camera_state(&mut self, state: &ProjectCameraState) {
        self.camera.center_world_lvl0 =
            egui::pos2(state.center_world_lvl0[0], state.center_world_lvl0[1]);
        if state.zoom_screen_per_lvl0_px.is_finite() && state.zoom_screen_per_lvl0_px > 0.0 {
            self.camera.zoom_screen_per_lvl0_px = state.zoom_screen_per_lvl0_px;
        }
    }

    pub(in crate::mosaic) fn project_ui_state(&self) -> ProjectUiState {
        ProjectUiState {
            show_left_panel: Some(self.show_left_panel),
            show_right_panel: Some(self.show_right_panel),
            left_tab: Some(self.left_tab.storage_key().to_string()),
            right_tab: Some(self.right_tab.storage_key().to_string()),
            channel_sort: Some(self.channel_sort_mode.storage_key().to_string()),
            smooth_pixels: Some(self.smooth_pixels),
            show_tile_debug: Some(self.show_tile_debug),
            show_scale_bar: None,
            show_hud: None,
            auto_level: None,
            manual_level: None,
        }
    }

    pub(in crate::mosaic) fn project_path_string(&self, path: &Path) -> String {
        if let Some(project_dir) = self.project_space.project_dir()
            && let Ok(relative) = path.strip_prefix(&project_dir)
        {
            return relative.to_string_lossy().to_string();
        }
        path.to_string_lossy().to_string()
    }

    pub(in crate::mosaic) fn resolve_project_path(&self, path: &str) -> PathBuf {
        let path_buf = PathBuf::from(path);
        if path_buf.is_absolute() {
            path_buf
        } else {
            self.project_space
                .project_dir()
                .map(|dir| dir.join(&path_buf))
                .unwrap_or(path_buf)
        }
    }

    pub(in crate::mosaic) fn project_annotation_layer_state(
        &self,
        layer: &AnnotationPointsLayer,
    ) -> ProjectAnnotationLayerState {
        ProjectAnnotationLayerState {
            id: layer.id,
            name: layer.name.clone(),
            visible: layer.visible,
            radius_screen_px: layer.style.radius_screen_px,
            opacity: layer.style.opacity,
            stroke_width: layer.style.stroke.width,
            stroke_color_rgb: [
                layer.style.stroke.color.r(),
                layer.style.stroke.color.g(),
                layer.style.stroke.color.b(),
            ],
            stroke_color_alpha: layer.style.stroke.color.a(),
            offset_world: [layer.offset_world.x, layer.offset_world.y],
            parquet_path: layer
                .parquet
                .path
                .as_deref()
                .map(|path| self.project_path_string(path)),
            roi_id_column: layer.parquet.roi_id_column.clone(),
            x_column: layer.parquet.x_column.clone(),
            y_column: layer.parquet.y_column.clone(),
            value_column: layer.parquet.value_column.clone(),
            selected_value_column: layer.selected_value_column.clone(),
            category_styles: layer
                .category_styles
                .iter()
                .map(|style| ProjectAnnotationCategoryStyleState {
                    name: style.name.clone(),
                    visible: style.visible,
                    color_rgb: [style.color.r(), style.color.g(), style.color.b()],
                    shape: style.shape.storage_key().to_string(),
                })
                .collect(),
            continuous_shape: Some(layer.continuous_shape.storage_key().to_string()),
            continuous_range: layer.continuous_range.map(|(lo, hi)| [lo, hi]),
        }
    }

    pub(in crate::mosaic) fn restore_annotation_layers(
        &mut self,
        layers: &[ProjectAnnotationLayerState],
    ) {
        self.annotation_layers.clear();
        self.next_annotation_layer_id = 1;
        for saved in layers {
            let mut layer = AnnotationPointsLayer::new(saved.id, saved.name.clone());
            layer.visible = saved.visible;
            layer.style.radius_screen_px = saved.radius_screen_px;
            layer.style.opacity = saved.opacity;
            layer.style.stroke.width = saved.stroke_width;
            layer.style.stroke.color = egui::Color32::from_rgba_unmultiplied(
                saved.stroke_color_rgb[0],
                saved.stroke_color_rgb[1],
                saved.stroke_color_rgb[2],
                saved.stroke_color_alpha,
            );
            layer.offset_world = egui::vec2(saved.offset_world[0], saved.offset_world[1]);
            layer.parquet.path = saved
                .parquet_path
                .as_deref()
                .map(|path| self.resolve_project_path(path));
            layer.parquet.roi_id_column = saved.roi_id_column.clone();
            layer.parquet.x_column = saved.x_column.clone();
            layer.parquet.y_column = saved.y_column.clone();
            layer.parquet.value_column = saved.value_column.clone();
            layer.selected_value_column = saved.selected_value_column.clone();
            layer.category_styles = saved
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
            if let Some(shape) = saved
                .continuous_shape
                .as_deref()
                .and_then(AnnotationShape::from_storage_key)
            {
                layer.continuous_shape = shape;
            }
            layer.continuous_range = saved.continuous_range.map(|[lo, hi]| (lo, hi));
            if layer.parquet.path.is_some() {
                layer.request_schema_load();
                layer.request_load();
            }
            self.next_annotation_layer_id = self.next_annotation_layer_id.max(saved.id + 1);
            self.annotation_layers.push(layer);
        }
    }

    pub(in crate::mosaic) fn apply_project_ui_state(&mut self, state: &ProjectUiState) {
        if let Some(show_left_panel) = state.show_left_panel {
            self.show_left_panel = show_left_panel;
        }
        if let Some(show_right_panel) = state.show_right_panel {
            self.show_right_panel = show_right_panel;
        }
        if let Some(left_tab) = state
            .left_tab
            .as_deref()
            .and_then(LeftTab::from_storage_key)
        {
            self.left_tab = left_tab;
        }
        if let Some(right_tab) = state
            .right_tab
            .as_deref()
            .and_then(RightTab::from_storage_key)
        {
            self.right_tab = right_tab;
        }
        if let Some(channel_sort) = state
            .channel_sort
            .as_deref()
            .and_then(ChannelSortMode::from_storage_key)
        {
            self.channel_sort_mode = channel_sort;
        }
        if let Some(smooth_pixels) = state.smooth_pixels {
            self.smooth_pixels = smooth_pixels;
            self.tiles_gl.set_smooth_pixels(self.smooth_pixels);
        }
        if let Some(show_tile_debug) = state.show_tile_debug {
            self.show_tile_debug = show_tile_debug;
        }
    }

    pub fn set_layer_groups(&mut self, groups: ProjectLayerGroups) {
        self.layer_groups = groups;
    }

    pub fn take_project_space(&mut self) -> ProjectSpace {
        self.project_space
            .set_mosaic_view_state(ProjectMosaicViewState {
                channel_order: self.channel_layer_order.clone(),
                channels: self
                    .channels
                    .iter()
                    .map(|ch| ProjectChannelViewState {
                        name: Some(ch.name.clone()),
                        visible: Some(ch.visible),
                        color_rgb: Some(ch.color_rgb),
                        window: ch.window.map(|(lo, hi)| [lo, hi]),
                        offset_world: None,
                        original_offset_world: None,
                        scale: None,
                        rotation_rad: None,
                        note: (!ch.note.is_empty()).then(|| ch.note.clone()),
                    })
                    .collect(),
                active_channel: Some(self.selected_channel),
                active_layer: Some(Self::layer_id_storage_key(self.active_layer)),
                overlay_order: self
                    .overlay_layer_order
                    .iter()
                    .copied()
                    .map(Self::layer_id_storage_key)
                    .collect(),
                overlay_visibility: self
                    .overlay_layer_order
                    .iter()
                    .copied()
                    .filter_map(|id| {
                        self.layer_visible_value(id)
                            .map(|visible| (Self::layer_id_storage_key(id), visible))
                    })
                    .collect::<BTreeMap<_, _>>(),
                sort_by: Some(self.sort_by.clone()),
                sort_secondary_enabled: Some(self.sort_secondary_enabled),
                sort_by_secondary: Some(self.sort_by_secondary.clone()),
                group_by: Some(self.group_by.clone()),
                show_group_labels: Some(self.show_group_labels),
                group_gap: Some(self.group_gap),
                layout_mode: Some(self.layout_mode.storage_key().to_string()),
                show_text_labels: Some(self.show_text_labels),
                label_columns: self.label_columns.clone(),
                camera: Some(self.project_camera_state()),
                ui: Some(self.project_ui_state()),
                annotation_layers: self
                    .annotation_layers
                    .iter()
                    .map(|layer| self.project_annotation_layer_state(layer))
                    .collect(),
            });
        self.project_space.update_layer_groups(|g| {
            *g = self.layer_groups.clone();
        });
        std::mem::take(&mut self.project_space)
    }

    pub fn project_space_mut(&mut self) -> &mut ProjectSpace {
        &mut self.project_space
    }

    pub fn project_space(&self) -> &ProjectSpace {
        &self.project_space
    }

    pub fn set_project_object_cache_ui_state(&mut self, state: ProjectObjectCacheUiState) {
        self.project_space.set_object_cache_ui_state(state);
    }

    pub fn install_preloaded_project_segmentations(
        &mut self,
        preloaded: &[(PathBuf, Arc<PreloadedObjectLayer>)],
    ) -> usize {
        let mut installed = 0usize;
        for (path, layer) in preloaded {
            installed += self.seg_geojson.install_preloaded(path, layer.as_ref());
        }
        if installed > 0 {
            if !self.control_actor_owned {
                self.seg_geojson.visible = true;
                self.active_layer = MosaicLayerId::SegmentationGeoJson;
            }
            self.status = format!("Using cached object segmentations for {installed} ROI(s).");
        }
        installed
    }

    pub fn set_project_space(&mut self, mut project_space: ProjectSpace) {
        if self.control_actor_owned {
            project_space.set_control_actor_owned(true);
            self.project_space = project_space;
            return;
        }
        self.layer_groups = project_space.layer_groups().clone();
        if let Some(view) = project_space.mosaic_view_state() {
            if let Some(ui) = view.ui.as_ref() {
                self.apply_project_ui_state(ui);
            }
            if !view.channel_order.is_empty() {
                let mut channel_order = view
                    .channel_order
                    .iter()
                    .copied()
                    .filter(|&idx| idx < self.channels.len())
                    .collect::<Vec<_>>();
                for idx in 0..self.channels.len() {
                    if !channel_order.contains(&idx) {
                        channel_order.push(idx);
                    }
                }
                self.channel_layer_order = channel_order;
            }
            let channel_notes_by_name = view
                .channels
                .iter()
                .filter_map(|saved| {
                    let name = saved.name.as_deref()?;
                    let note = saved.note.as_ref()?;
                    Some((name, note))
                })
                .collect::<HashMap<_, _>>();
            for (idx, saved) in view.channels.iter().enumerate() {
                let Some(ch) = self.channels.get_mut(idx) else {
                    continue;
                };
                if let Some(note) = channel_notes_by_name.get(ch.name.as_str()) {
                    ch.note = (*note).clone();
                }
                if let Some(visible) = saved.visible {
                    ch.visible = visible;
                }
                if let Some(color_rgb) = saved.color_rgb {
                    ch.color_rgb = color_rgb;
                }
                if let Some([lo, hi]) = saved.window {
                    ch.window = Some((lo, hi));
                }
            }
            if let Some(active_channel) = view.active_channel {
                self.selected_channel = active_channel.min(self.channels.len().saturating_sub(1));
            }
            if !view.overlay_order.is_empty() {
                let mut overlay_order = view
                    .overlay_order
                    .iter()
                    .filter_map(|id| self.parse_layer_id_storage_key(id))
                    .collect::<Vec<_>>();
                for id in self.overlay_layer_order.iter().copied() {
                    if !overlay_order.contains(&id) {
                        overlay_order.push(id);
                    }
                }
                self.overlay_layer_order = overlay_order;
            }
            if let Some(sort_by) = view.sort_by.as_ref() {
                self.sort_by = sort_by.clone();
            }
            if let Some(sort_secondary_enabled) = view.sort_secondary_enabled {
                self.sort_secondary_enabled = sort_secondary_enabled;
            }
            if let Some(sort_by_secondary) = view.sort_by_secondary.as_ref() {
                self.sort_by_secondary = sort_by_secondary.clone();
            }
            if let Some(group_by) = view.group_by.as_ref() {
                self.group_by = group_by.clone();
            }
            if let Some(show_group_labels) = view.show_group_labels {
                self.show_group_labels = show_group_labels;
            }
            if let Some(group_gap) = view.group_gap {
                self.group_gap = group_gap;
            }
            if let Some(layout_mode) = view.layout_mode.as_deref()
                && let Some(parsed) = MosaicLayoutMode::from_storage_key(layout_mode)
            {
                self.layout_mode = parsed;
            }
            if let Some(show_text_labels) = view.show_text_labels {
                self.show_text_labels = show_text_labels;
            }
            self.label_columns = view.label_columns.clone();
            self.apply_sort_and_layout();
            for (id, visible) in &view.overlay_visibility {
                if let Some(layer_id) = self.parse_layer_id_storage_key(id) {
                    self.set_layer_visible(layer_id, *visible);
                }
            }
            if let Some(camera) = view.camera.as_ref() {
                self.apply_project_camera_state(camera);
            }
            self.restore_annotation_layers(&view.annotation_layers);
            if let Some(active_layer) = view
                .active_layer
                .as_deref()
                .and_then(|id| self.parse_layer_id_storage_key(id))
            {
                self.set_active_layer(active_layer);
            } else if !self.channels.is_empty() {
                self.set_active_layer(MosaicLayerId::Channel(
                    self.selected_channel
                        .min(self.channels.len().saturating_sub(1)),
                ));
            }
        }
        project_space.update_layer_groups(|g| {
            *g = self.layer_groups.clone();
        });
        self.project_space = project_space;
    }
}
