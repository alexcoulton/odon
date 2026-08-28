use super::*;

pub(super) struct PreparedMosaicConstruction {
    pub(super) items: Vec<MosaicItem>,
    pub(super) sources: Arc<Vec<MosaicSource>>,
    pub(super) pinned_levels: MosaicPinnedLevels,
    pub(super) loader: MosaicRawTileLoaderHandle,
    pub(super) remote_runtimes: Vec<Arc<tokio::runtime::Runtime>>,
    pub(super) camera: Camera,
    pub(super) mosaic_bounds: egui::Rect,
    pub(super) abs_max: f32,
    pub(super) channels: Vec<GlobalChannel>,
    pub(super) metadata_columns: Vec<String>,
    pub(super) group_blocks: Vec<GroupBlock>,
    pub(super) grid_cols: usize,
    pub(super) renderer_status: String,
    pub(super) show_return_navigation: bool,
    pub(super) seg_geojson: MosaicGeoJsonSegmentationOverlay,
    pub(super) consumed_mosaic_resource_generation: u64,
}

impl MosaicViewerApp {
    pub(super) fn from_prepared_construction(prepared: PreparedMosaicConstruction) -> Self {
        let PreparedMosaicConstruction {
            items,
            sources,
            pinned_levels,
            loader,
            remote_runtimes,
            camera,
            mosaic_bounds,
            abs_max,
            channels,
            metadata_columns,
            group_blocks,
            grid_cols,
            renderer_status,
            show_return_navigation,
            seg_geojson,
            consumed_mosaic_resource_generation,
        } = prepared;

        let focused_core_id = items.first().map(|item| item.id);
        let channel_layer_order = (0..channels.len()).collect::<Vec<_>>();
        let active_layer = channel_layer_order
            .first()
            .copied()
            .map(MosaicLayerId::Channel)
            .unwrap_or(MosaicLayerId::TextLabels);
        let selected_channel_layers = if channels.is_empty() {
            HashSet::new()
        } else {
            HashSet::from([0])
        };
        let memory_selected_channels = (0..channels.len()).collect();
        let sources_len = sources.len();

        Self {
            items,
            sources,
            pinned_levels,
            loader,
            tiles_gl: MosaicTilesGl::new(odon::settings::ImageTileCacheSettings::default()),
            _remote_runtimes: remote_runtimes,
            camera,
            last_canvas_rect: None,
            mosaic_bounds,
            focused_core_id,
            selected_core_ids: focused_core_id.into_iter().collect(),
            abs_max,
            channels,
            selected_channel: 0,
            channel_list_search: String::new(),
            active_layer,
            selected_channel_layers,
            memory_selected_channels,
            channel_select_anchor_idx: None,
            selected_channel_group_id: None,
            quick_contrast_target: top_bar::QuickContrastTarget::Visible,
            selected_overlay_layers: HashSet::new(),
            overlay_select_anchor_pos: None,
            overlay_layer_order: vec![
                MosaicLayerId::SegmentationGeoJson,
                MosaicLayerId::TextLabels,
            ],
            channel_layer_order,
            channel_sort_mode: ChannelSortMode::Manual,
            annotation_layers: Vec::new(),
            last_target_level_by_dataset_id: vec![None; sources_len],
            fallback_ceiling_by_dataset_id: vec![None; sources_len],
            zoom_out_floor_by_dataset_id: vec![None; sources_len],
            zoom_out_floor_until_by_dataset_id: vec![None; sources_len],
            zoom_out_floor_world_by_dataset_id: vec![None; sources_len],
            last_visible_world: None,
            layer_groups: ProjectLayerGroups::default(),
            layer_drag: None,
            left_tab: LeftTab::Layers,
            right_tab: RightTab::Properties,
            control_shell_projection: serde_json::json!({}),
            control_shell_layout: Default::default(),
            extension_ui_registry: None,
            metadata_columns,
            sort_by: "id".to_string(),
            sort_secondary_enabled: false,
            sort_by_secondary: "id".to_string(),
            group_by: String::new(),
            show_group_labels: true,
            group_gap: 96.0,
            layout_mode: MosaicLayoutMode::FitCells,
            group_blocks,
            show_text_labels: true,
            label_columns: vec!["id".to_string()],
            grid_cols,
            show_left_panel: true,
            show_right_panel: true,
            close_dialog_open: false,
            system_memory: None,
            system_memory_last_refresh: None,
            pending_memory_load: None,
            control_actor_memory_state: serde_json::json!({}),
            tile_request_generation: 1,
            last_tile_request_signature: None,
            renderer_status,
            show_return_navigation,
            return_dataset_root: None,
            pending_platform_effect: None,
            group_layers_dialog: None,
            smooth_pixels: true,
            show_tile_debug: false,
            screenshot_dialog: ScreenshotDialogState::default(),
            screenshot_capture: MosaicScreenshotCaptureAdapter::default(),
            seg_geojson,
            seg_geojson_pending_visible: false,
            project_space: ProjectSpace::default(),
            active_help_topic: None,
            consumed_mosaic_resource_generation,
            consumed_mosaic_object_generation: 0,
            native_command_ingress: NativeControlIngress::detached(),
        }
    }
}
