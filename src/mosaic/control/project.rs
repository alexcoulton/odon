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
            self.status = format!("Using cached object segmentations for {installed} ROI(s).");
        }
        installed
    }

    pub fn set_project_space(&mut self, mut project_space: ProjectSpace) {
        if let Some(view) = project_space.mosaic_view_state() {
            // Annotation geometry remains a renderer-installed shared resource until the
            // annotation ownership milestone. Mosaic UI/layout/channel semantics are restored by
            // the actor from the same project snapshot and arrive here only through projection.
            self.restore_annotation_layers(&view.annotation_layers);
        }
        project_space.set_control_actor_owned(true);
        self.project_space = project_space;
    }
}
