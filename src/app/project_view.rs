use super::*;

impl OmeZarrViewerApp {
    pub(super) fn layer_id_storage_key(id: LayerId) -> String {
        match id {
            LayerId::Channel(idx) => format!("channel:{idx}"),
            LayerId::SpatialImage(id) => format!("spatial_image:{id}"),
            LayerId::SegmentationLabels => "segmentation_labels".to_string(),
            LayerId::SegmentationGeoJson => "segmentation_geojson".to_string(),
            LayerId::SegmentationObjects => "segmentation_objects".to_string(),
            LayerId::Mask(id) => format!("mask:{id}"),
            LayerId::Points => "points".to_string(),
            LayerId::Annotation(id) => format!("annotation:{id}"),
            LayerId::SpatialShape(id) => format!("spatial_shape:{id}"),
            LayerId::SpatialPoints => "spatial_points".to_string(),
            LayerId::XeniumCells => "xenium_cells".to_string(),
            LayerId::XeniumTranscripts => "xenium_transcripts".to_string(),
        }
    }

    pub(super) fn parse_layer_id_storage_key(&self, value: &str) -> Option<LayerId> {
        if let Some(raw) = value.strip_prefix("channel:") {
            return raw.parse::<usize>().ok().map(LayerId::Channel);
        }
        if let Some(raw) = value.strip_prefix("spatial_image:") {
            return raw.parse::<u64>().ok().map(LayerId::SpatialImage);
        }
        if let Some(raw) = value.strip_prefix("mask:") {
            return raw.parse::<u64>().ok().map(LayerId::Mask);
        }
        if let Some(raw) = value.strip_prefix("annotation:") {
            return raw.parse::<u64>().ok().map(LayerId::Annotation);
        }
        if let Some(raw) = value.strip_prefix("spatial_shape:") {
            return raw.parse::<u64>().ok().map(LayerId::SpatialShape);
        }
        match value {
            "segmentation_labels" => Some(LayerId::SegmentationLabels),
            "segmentation_geojson" => Some(LayerId::SegmentationGeoJson),
            "segmentation_objects" => Some(LayerId::SegmentationObjects),
            "points" => Some(LayerId::Points),
            "spatial_points" => Some(LayerId::SpatialPoints),
            "xenium_cells" => Some(LayerId::XeniumCells),
            "xenium_transcripts" => Some(LayerId::XeniumTranscripts),
            _ => None,
        }
    }

    pub(super) fn project_camera_state(&self) -> ProjectCameraState {
        ProjectCameraState {
            center_world_lvl0: [
                self.camera.center_world_lvl0.x,
                self.camera.center_world_lvl0.y,
            ],
            zoom_screen_per_lvl0_px: self.camera.zoom_screen_per_lvl0_px,
        }
    }

    #[cfg(test)]
    pub(super) fn apply_project_camera_state(&mut self, state: &ProjectCameraState) {
        self.camera.center_world_lvl0 =
            egui::pos2(state.center_world_lvl0[0], state.center_world_lvl0[1]);
        if state.zoom_screen_per_lvl0_px.is_finite() && state.zoom_screen_per_lvl0_px > 0.0 {
            self.camera.zoom_screen_per_lvl0_px = state.zoom_screen_per_lvl0_px;
        }
    }

    #[cfg(test)]
    pub(super) fn project_ui_state(&self) -> ProjectUiState {
        ProjectUiState {
            show_left_panel: Some(self.show_left_panel),
            show_right_panel: Some(self.show_right_panel),
            left_tab: Some(self.left_tab.storage_key().to_string()),
            right_tab: Some(self.right_tab.storage_key().to_string()),
            channel_sort: Some(self.channel_sort_mode.storage_key().to_string()),
            smooth_pixels: Some(self.smooth_pixels),
            show_tile_debug: Some(self.show_tile_debug),
            show_scale_bar: Some(self.show_scale_bar),
            show_hud: Some(self.show_hud),
            auto_level: None,
            manual_level: None,
        }
    }

    #[cfg(test)]
    pub(super) fn resolve_project_path(&self, path: &str) -> PathBuf {
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

    #[cfg(test)]
    pub(super) fn restore_annotation_layers(&mut self, layers: &[ProjectAnnotationLayerState]) {
        self.annotation_layers.clear();
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
                .map(|style| crate::annotations::AnnotationCategoryStyle {
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
            self.annotation_layers.push(layer);
        }
    }

    #[cfg(test)]
    pub(super) fn apply_project_ui_state(&mut self, state: &ProjectUiState) {
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
            if let Some(tiles_gl) = self.tiles_gl.as_ref() {
                tiles_gl.set_smooth_pixels(self.smooth_pixels);
            }
            self.spatial_image_layers
                .set_smooth_pixels(self.smooth_pixels);
        }
        if let Some(show_tile_debug) = state.show_tile_debug {
            self.show_tile_debug = show_tile_debug;
        }
        if let Some(show_scale_bar) = state.show_scale_bar {
            self.show_scale_bar = show_scale_bar;
        }
        if let Some(show_hud) = state.show_hud {
            self.show_hud = show_hud;
        }
        let _ = (state.auto_level, state.manual_level);
    }
}
