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
    pub(super) status: String,
    pub(super) allow_back: bool,
    pub(super) seg_geojson: MosaicGeoJsonSegmentationOverlay,
    pub(super) control_actor_generation: u64,
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
            status,
            allow_back,
            seg_geojson,
            control_actor_generation,
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
            tiles_gl: MosaicTilesGl::new(12_000),
            _remote_runtimes: remote_runtimes,
            camera,
            last_canvas_rect: None,
            mosaic_bounds,
            focused_core_id,
            selected_core_ids: focused_core_id.into_iter().collect(),
            pending_object_load_ids: HashSet::new(),
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
            next_annotation_layer_id: 1,
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
            tile_request_generation: 1,
            last_tile_request_signature: None,
            status,
            allow_back,
            pending_request: None,
            group_layers_dialog: None,
            smooth_pixels: true,
            show_tile_debug: false,
            screenshot_settings: ScreenshotSettings::default(),
            screenshot_settings_open: false,
            screenshot_worker: ScreenshotWorkerHandle::spawn(),
            screenshot_next_id: 1,
            screenshot_pending: None,
            screenshot_in_flight: None,
            screenshot_output_dir: None,
            seg_geojson,
            seg_geojson_pending_visible: false,
            project_space: ProjectSpace::default(),
            active_help_topic: None,
            control_actor_generation,
            control_actor_object_generation: 0,
            native_control_intents: Vec::new(),
        }
    }
}
