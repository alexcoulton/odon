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

    pub fn set_layer_groups(&mut self, groups: ProjectLayerGroups) {
        self.layer_groups = groups;
    }

    pub fn take_project_space(&mut self) -> ProjectSpace {
        // Annotation state is actor-owned. Preserve the actor projection already stored in the
        // project instead of reconstructing it from renderer adapters.
        let annotation_layers = self
            .project_space
            .mosaic_view_state()
            .map(|view| view.annotation_layers.clone())
            .unwrap_or_default();
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
                annotation_layers,
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
            self.renderer_status =
                format!("Using cached object segmentations for {installed} ROI(s).");
        }
        installed
    }

    pub fn set_project_space(&mut self, mut project_space: ProjectSpace) {
        project_space.set_control_actor_owned(true);
        self.project_space = project_space;
    }
}
