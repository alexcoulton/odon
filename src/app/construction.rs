use super::*;

impl OmeZarrViewerApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        dataset: OmeZarrDataset,
        store: Arc<dyn zarrs::storage::ReadableStorageTraits>,
        auto_contrast_settings: AutoContrastSettings,
    ) -> Self {
        apply_napari_like_dark(&cc.egui_ctx);

        let tile_loader_threads = Self::default_tile_loader_threads();

        let loader = spawn_tile_loader(
            store.clone(),
            dataset.levels.clone(),
            dataset.dims.clone(),
            tile_loader_threads,
        )
        .expect("failed to spawn tile loader");

        let (raw_loader, tiles_gl) = if cc.gl.is_some() {
            let raw = spawn_raw_tile_loader(
                store.clone(),
                dataset.levels.clone(),
                dataset.dims.clone(),
                tile_loader_threads,
            )
            .ok();
            (raw, Some(TilesGl::new(RAW_TILE_CACHE_CAPACITY_TILES)))
        } else {
            (None, None)
        };

        let seg_label_names = dataset
            .source
            .local_path()
            .map(discover_label_names_local)
            .unwrap_or_default();
        let seg_label_selected = if seg_label_names.iter().any(|n| n == "cells") {
            "cells".to_string()
        } else if let Some(first) = seg_label_names.first() {
            first.clone()
        } else {
            "cells".to_string()
        };
        let seg_label_input = seg_label_selected.clone();
        let seg_label_status = String::new();
        let seg_label_prompt_open = cc.gl.is_some() && !seg_label_names.is_empty();

        let (label_cells, label_loader, label_cells_xform, labels_gl) = if cc.gl.is_some() {
            // Labels are not auto-loaded; we prompt on open if any are present.
            (None, None, None, Some(LabelsGl::new(1024)))
        } else {
            (None, None, None, None)
        };

        let hist_loader =
            spawn_histogram_loader(store.clone(), dataset.levels.clone(), dataset.dims.clone())
                .expect("failed to spawn histogram loader");

        let chanmax_level = update::choose_default_max_level(&dataset);
        let chanmax_loader =
            spawn_channel_max_loader(store.clone(), dataset.levels.clone(), dataset.dims.clone())
                .expect("failed to spawn channel max loader");

        let mut camera = Camera::default();
        camera.center_world_lvl0 = egui::pos2(0.0, 0.0);
        camera.zoom_screen_per_lvl0_px = 0.1;

        let mut app = Self {
            dataset: dataset.clone(),
            store: store.clone(),
            remote_runtime: None,
            loader,
            raw_loader,
            label_cells,
            label_loader,
            label_cells_xform,
            seg_label_names,
            seg_label_selected,
            seg_label_input,
            seg_label_status,
            seg_label_prompt_open,
            seg_label_prompt_always: false,
            seg_label_prompt_preference: LabelPromptSessionPreference::Ask,
            hist_loader,
            chanmax_loader,
            chanmax_request_id: 1,
            chanmax_level,
            chanmax_pending: vec![true; dataset.channels.len()],
            chanmax_snapshot: dataset.channels.iter().map(|c| c.window).collect(),
            cache: TileCache::new(256),
            pending: Vec::new(),
            hist: None,
            hist_request_id: 0,
            hist_request_pending: false,
            hist_dirty: true,
            hist_navigation_dirty_since: None,
            hist_last_sent: Instant::now()
                .checked_sub(Duration::from_secs(3600))
                .unwrap_or_else(Instant::now),
            camera,
            active_render_id: 1,
            previous_render_id: None,
            active_render_smooth_pixels: true,
            previous_render_smooth_pixels: None,
            previous_view_selection: None,
            previous_displayed_view_selection: None,
            last_render_view_selection: ViewPlaneSelection {
                mode: ViewPlaneMode::Xy,
                slice_level0: 0,
            },
            last_canvas_rect: None,
            last_target_level: None,
            fallback_ceiling_level: None,
            last_visible_world_tiles: None,
            zoom_out_floor_level: None,
            zoom_out_floor_until: None,
            zoom_out_floor_visible_world_tiles: None,
            selected_channel: 0,
            view_plane_mode: ViewPlaneMode::Xy,
            draft_view_slice_level0: None,
            current_x_level0: 0,
            current_y_level0: 0,
            current_z_level0: 0,
            channels: dataset.channels.clone(),
            channel_window_overrides: HashMap::new(),
            auto_contrast_settings,
            fast_object_rendering: true,
            channel_list_search: String::new(),

            active_layer: if dataset.channels.is_empty() {
                LayerId::Points
            } else {
                LayerId::Channel(0)
            },
            selected_channel_layers: if dataset.channels.is_empty() {
                HashSet::new()
            } else {
                HashSet::from([0usize])
            },
            memory_selected_channels: (0..dataset.channels.len()).collect(),
            channel_select_anchor_idx: None,
            selected_channel_group_id: None,
            quick_contrast_target: top_bar::QuickContrastTarget::Visible,
            selected_overlay_layers: HashSet::new(),
            overlay_select_anchor_pos: None,
            show_left_panel: true,
            show_right_panel: true,
            close_dialog_open: false,
            pinned_levels: PinnedLevels::new(),
            pending_memory_load: None,
            memory_status: String::new(),
            system_memory: None,
            system_memory_last_refresh: None,
            left_tab: LeftTab::Layers,
            right_tab: RightTab::Properties,
            project_space: ProjectSpace::default(),
            project_cfg_seen: 0,
            roi_selector: RoiSelectorPanel::new(&dataset.source),
            cell_thresholds: CellThresholdsPanel::new(
                dataset
                    .source
                    .local_path()
                    .unwrap_or_else(|| std::path::Path::new("")),
                dataset.multiscale.name.as_deref(),
            ),
            cell_points: PointsLayer::new("cell_centroids"),
            annotation_layers: Vec::new(),
            next_annotation_layer_id: 1,
            mask_layers: Vec::new(),
            next_mask_layer_id: 1,
            mask_layers_project_dirty: true,
            tool_mode: ToolMode::Pan,
            drawing_mask_layer: None,
            drawing_mask_polygon: Vec::new(),
            selected_mask_polygon: None,
            selected_mask_vertex: None,
            dragging_mask_vertex: None,
            moving_mask_polygon: None,
            undo_stack: Vec::new(),
            selection_rect_start_world: None,
            selection_rect_current_world: None,
            selection_lasso_world: Vec::new(),
            threshold_region_min_pixels: 32,
            threshold_region_scope: ThresholdRegionScope::VisibleRegion,
            threshold_region_full_level: 0,
            threshold_region_status: String::new(),
            threshold_region_preview: None,
            cells_outlines_visible: true,
            cells_outlines_color_rgb: [0, 255, 0],
            cells_outlines_opacity: 0.75,
            cells_outlines_width_px: 0.0,
            points_gl: cc.gl.as_ref().map(|_| PointsGlRenderer::default()),
            threshold_preview_gl: cc
                .gl
                .as_ref()
                .map(|_| ThresholdPreviewGlRenderer::default()),
            tiles_gl,
            labels_gl,

            remote_dialog_open: false,
            remote_mode: RemoteMode::Http,
            remote_http_url: String::new(),
            remote_s3_endpoint: String::new(),
            remote_s3_region: "auto".to_string(),
            remote_s3_bucket: String::new(),
            remote_s3_prefix: String::new(),
            remote_s3_access_key: String::new(),
            remote_s3_secret_key: String::new(),
            remote_status: String::new(),
            remote_s3_browser: None,

            pending_request: None,
            native_control_intents: Vec::new(),
            control_actor_object_generation: 0,
            control_actor_label_generation: 0,
            control_actor_object_selection_generation: 0,
            control_actor_mask_generation: 0,
            control_actor_mask_undo_available: false,
            native_mask_actor_intent_emitted: false,
            group_layers_dialog: None,
            hover_tooltip_state: None,
            active_help_topic: None,
            roi_info_open: false,
            smooth_pixels: true,
            show_tile_debug: false,
            mask_draw_debug_stats: MaskDrawDebugStats::default(),
            show_scale_bar: true,
            show_hud: true,
            tile_loader_threads,
            tile_prefetch_mode: TilePrefetchMode::TargetHalo,
            tile_prefetch_aggressiveness: TilePrefetchAggressiveness::Balanced,
            tile_loading_status: String::new(),
            prefer_pinned_finer_levels: false,

            seg_geojson: GeoJsonSegmentationLayer::default(),
            seg_objects: ObjectsLayer::default(),
            spatial_image_layers: SpatialImageLayers::default(),
            spatial_layers: SpatialDataLayers::default(),
            spatial_image_transform: SpatialDataTransform2::default(),
            spatial_label_transform: SpatialDataTransform2::default(),
            spatial_root: None,
            spatial_label_store: None,
            xenium_layers: XeniumLayers::default(),

            channel_offsets_world: vec![egui::Vec2::ZERO; dataset.channels.len()],
            channel_scales: vec![egui::Vec2::splat(1.0); dataset.channels.len()],
            channel_rotations_rad: vec![0.0; dataset.channels.len()],
            loaded_layer_offsets_world: HashMap::new(),
            points_offset_world: egui::Vec2::ZERO,
            spatial_points_offset_world: egui::Vec2::ZERO,
            seg_labels_offset_world: egui::Vec2::ZERO,
            seg_geojson_offset_world: egui::Vec2::ZERO,
            seg_objects_offset_world: egui::Vec2::ZERO,
            xenium_cells_offset_world: egui::Vec2::ZERO,
            xenium_transcripts_offset_world: egui::Vec2::ZERO,

            overlay_layer_order: Vec::new(),
            channel_layer_order: (0..dataset.channels.len()).collect(),
            channel_sort_mode: ChannelSortMode::Manual,
            layer_drag: None,
            layer_move: None,
            layer_transform: None,
            tiff_plane_state: None,
            threshold_region_preview_generation: 1,
            screenshot_settings: ScreenshotSettings::default(),
            screenshot_settings_open: false,
            screenshot_worker: ScreenshotWorkerHandle::spawn(),
            screenshot_next_id: 1,
            screenshot_pending: VecDeque::new(),
            screenshot_in_flight: HashMap::new(),
            screenshot_output_dir: None,
            viewport_workspace: None,
            viewport_layer_groups: ProjectLayerGroups::default(),
            viewport_raw_active_keys: None,
            viewport_cpu_active_keys: None,
            viewport_label_active_keys: None,
            viewport_spatial_image_active_keys: None,
            viewport_frame_plan_ms: 0.0,
            viewport_frame_plan_ema_ms: 0.0,
            viewport_frame_plan_samples: 0,
        };

        app.configure_root_label_dataset_if_needed();
        app.rebuild_layer_orders();
        app.capture_loaded_layer_offsets();
        app.maybe_apply_auto_contrast_on_open();
        app.active_render_id = app.compute_render_id();

        // Initial fit (best effort).
        let world = app.image_world_rect_lvl0();
        if let Some(viewport) = cc.egui_ctx.input(|i| i.viewport().inner_rect) {
            app.camera.fit_to_world_rect(viewport, world);
        }
        app.viewport_workspace = Some(ViewportWorkspace::new(ViewerViewportState::capture(&app)));

        app
    }

    pub fn new_runtime(
        ctx: &egui::Context,
        gpu_available: bool,
        dataset: OmeZarrDataset,
        store: Arc<dyn zarrs::storage::ReadableStorageTraits>,
        auto_contrast_settings: AutoContrastSettings,
    ) -> Self {
        apply_napari_like_dark(ctx);

        let tile_loader_threads = Self::default_tile_loader_threads();

        let loader = spawn_tile_loader(
            store.clone(),
            dataset.levels.clone(),
            dataset.dims.clone(),
            tile_loader_threads,
        )
        .expect("failed to spawn tile loader");

        let (raw_loader, tiles_gl) = if gpu_available {
            let raw = spawn_raw_tile_loader(
                store.clone(),
                dataset.levels.clone(),
                dataset.dims.clone(),
                tile_loader_threads,
            )
            .ok();
            (raw, Some(TilesGl::new(RAW_TILE_CACHE_CAPACITY_TILES)))
        } else {
            (None, None)
        };

        let seg_label_names = dataset
            .source
            .local_path()
            .map(discover_label_names_local)
            .unwrap_or_default();
        let seg_label_selected = if seg_label_names.iter().any(|n| n == "cells") {
            "cells".to_string()
        } else if let Some(first) = seg_label_names.first() {
            first.clone()
        } else {
            "cells".to_string()
        };
        let seg_label_input = seg_label_selected.clone();
        let seg_label_status = String::new();
        let seg_label_prompt_open = gpu_available && !seg_label_names.is_empty();

        let (label_cells, label_loader, label_cells_xform, labels_gl) = if gpu_available {
            // Labels are not auto-loaded; we prompt on open if any are present.
            (None, None, None, Some(LabelsGl::new(1024)))
        } else {
            (None, None, None, None)
        };

        let hist_loader =
            spawn_histogram_loader(store.clone(), dataset.levels.clone(), dataset.dims.clone())
                .expect("failed to spawn histogram loader");

        let chanmax_level = update::choose_default_max_level(&dataset);
        let chanmax_loader =
            spawn_channel_max_loader(store.clone(), dataset.levels.clone(), dataset.dims.clone())
                .expect("failed to spawn channel max loader");

        let mut camera = Camera::default();
        camera.center_world_lvl0 = egui::pos2(0.0, 0.0);
        camera.zoom_screen_per_lvl0_px = 0.1;

        let mut app = Self {
            dataset: dataset.clone(),
            store: store.clone(),
            remote_runtime: None,
            loader,
            raw_loader,
            label_cells,
            label_loader,
            label_cells_xform,
            seg_label_names,
            seg_label_selected,
            seg_label_input,
            seg_label_status,
            seg_label_prompt_open,
            seg_label_prompt_always: false,
            seg_label_prompt_preference: LabelPromptSessionPreference::Ask,
            hist_loader,
            chanmax_loader,
            chanmax_request_id: 1,
            chanmax_level,
            chanmax_pending: vec![true; dataset.channels.len()],
            chanmax_snapshot: dataset.channels.iter().map(|c| c.window).collect(),
            cache: TileCache::new(256),
            pending: Vec::new(),
            hist: None,
            hist_request_id: 0,
            hist_request_pending: false,
            hist_dirty: true,
            hist_navigation_dirty_since: None,
            hist_last_sent: Instant::now()
                .checked_sub(Duration::from_secs(3600))
                .unwrap_or_else(Instant::now),
            camera,
            active_render_id: 1,
            previous_render_id: None,
            active_render_smooth_pixels: true,
            previous_render_smooth_pixels: None,
            previous_view_selection: None,
            previous_displayed_view_selection: None,
            last_render_view_selection: ViewPlaneSelection {
                mode: ViewPlaneMode::Xy,
                slice_level0: 0,
            },
            last_canvas_rect: None,
            last_target_level: None,
            fallback_ceiling_level: None,
            last_visible_world_tiles: None,
            zoom_out_floor_level: None,
            zoom_out_floor_until: None,
            zoom_out_floor_visible_world_tiles: None,
            selected_channel: 0,
            view_plane_mode: ViewPlaneMode::Xy,
            draft_view_slice_level0: None,
            current_x_level0: 0,
            current_y_level0: 0,
            current_z_level0: 0,
            channels: dataset.channels.clone(),
            channel_window_overrides: HashMap::new(),
            auto_contrast_settings,
            fast_object_rendering: true,
            channel_list_search: String::new(),

            active_layer: if dataset.channels.is_empty() {
                LayerId::Points
            } else {
                LayerId::Channel(0)
            },
            selected_channel_layers: if dataset.channels.is_empty() {
                HashSet::new()
            } else {
                HashSet::from([0usize])
            },
            memory_selected_channels: (0..dataset.channels.len()).collect(),
            channel_select_anchor_idx: None,
            selected_channel_group_id: None,
            quick_contrast_target: top_bar::QuickContrastTarget::Visible,
            selected_overlay_layers: HashSet::new(),
            overlay_select_anchor_pos: None,
            show_left_panel: true,
            show_right_panel: true,
            close_dialog_open: false,
            pinned_levels: PinnedLevels::new(),
            pending_memory_load: None,
            memory_status: String::new(),
            system_memory: None,
            system_memory_last_refresh: None,
            left_tab: LeftTab::Layers,
            right_tab: RightTab::Properties,
            project_space: ProjectSpace::default(),
            project_cfg_seen: 0,
            roi_selector: RoiSelectorPanel::new(&dataset.source),
            cell_thresholds: CellThresholdsPanel::new(
                dataset
                    .source
                    .local_path()
                    .unwrap_or_else(|| std::path::Path::new("")),
                dataset.multiscale.name.as_deref(),
            ),
            cell_points: PointsLayer::new("cell_centroids"),
            annotation_layers: Vec::new(),
            next_annotation_layer_id: 1,
            mask_layers: Vec::new(),
            next_mask_layer_id: 1,
            mask_layers_project_dirty: true,
            tool_mode: ToolMode::Pan,
            drawing_mask_layer: None,
            drawing_mask_polygon: Vec::new(),
            selected_mask_polygon: None,
            selected_mask_vertex: None,
            dragging_mask_vertex: None,
            moving_mask_polygon: None,
            undo_stack: Vec::new(),
            selection_rect_start_world: None,
            selection_rect_current_world: None,
            selection_lasso_world: Vec::new(),
            threshold_region_min_pixels: 32,
            threshold_region_scope: ThresholdRegionScope::VisibleRegion,
            threshold_region_full_level: 0,
            threshold_region_status: String::new(),
            threshold_region_preview: None,
            cells_outlines_visible: true,
            cells_outlines_color_rgb: [0, 255, 0],
            cells_outlines_opacity: 0.75,
            cells_outlines_width_px: 0.0,
            points_gl: gpu_available.then_some(PointsGlRenderer::default()),
            threshold_preview_gl: gpu_available.then_some(ThresholdPreviewGlRenderer::default()),
            tiles_gl,
            labels_gl,

            remote_dialog_open: false,
            remote_mode: RemoteMode::Http,
            remote_http_url: String::new(),
            remote_s3_endpoint: String::new(),
            remote_s3_region: "auto".to_string(),
            remote_s3_bucket: String::new(),
            remote_s3_prefix: String::new(),
            remote_s3_access_key: String::new(),
            remote_s3_secret_key: String::new(),
            remote_status: String::new(),
            remote_s3_browser: None,

            pending_request: None,
            native_control_intents: Vec::new(),
            control_actor_object_generation: 0,
            control_actor_label_generation: 0,
            control_actor_object_selection_generation: 0,
            control_actor_mask_generation: 0,
            control_actor_mask_undo_available: false,
            native_mask_actor_intent_emitted: false,
            group_layers_dialog: None,
            hover_tooltip_state: None,
            active_help_topic: None,
            roi_info_open: false,
            smooth_pixels: true,
            show_tile_debug: false,
            mask_draw_debug_stats: MaskDrawDebugStats::default(),
            show_scale_bar: true,
            show_hud: true,
            tile_loader_threads: Self::default_tile_loader_threads(),
            tile_prefetch_mode: TilePrefetchMode::TargetHalo,
            tile_prefetch_aggressiveness: TilePrefetchAggressiveness::Balanced,
            tile_loading_status: String::new(),
            prefer_pinned_finer_levels: false,

            seg_geojson: GeoJsonSegmentationLayer::default(),
            seg_objects: ObjectsLayer::default(),
            spatial_image_layers: SpatialImageLayers::default(),
            spatial_layers: SpatialDataLayers::default(),
            spatial_image_transform: SpatialDataTransform2::default(),
            spatial_label_transform: SpatialDataTransform2::default(),
            spatial_root: None,
            spatial_label_store: None,
            xenium_layers: XeniumLayers::default(),

            channel_offsets_world: vec![egui::Vec2::ZERO; dataset.channels.len()],
            channel_scales: vec![egui::Vec2::splat(1.0); dataset.channels.len()],
            channel_rotations_rad: vec![0.0; dataset.channels.len()],
            loaded_layer_offsets_world: HashMap::new(),
            points_offset_world: egui::Vec2::ZERO,
            spatial_points_offset_world: egui::Vec2::ZERO,
            seg_labels_offset_world: egui::Vec2::ZERO,
            seg_geojson_offset_world: egui::Vec2::ZERO,
            seg_objects_offset_world: egui::Vec2::ZERO,
            xenium_cells_offset_world: egui::Vec2::ZERO,
            xenium_transcripts_offset_world: egui::Vec2::ZERO,

            overlay_layer_order: Vec::new(),
            channel_layer_order: (0..dataset.channels.len()).collect(),
            channel_sort_mode: ChannelSortMode::Manual,
            layer_drag: None,
            layer_move: None,
            layer_transform: None,
            tiff_plane_state: None,
            threshold_region_preview_generation: 1,
            screenshot_settings: ScreenshotSettings::default(),
            screenshot_settings_open: false,
            screenshot_worker: ScreenshotWorkerHandle::spawn(),
            screenshot_next_id: 1,
            screenshot_pending: VecDeque::new(),
            screenshot_in_flight: HashMap::new(),
            screenshot_output_dir: None,
            viewport_workspace: None,
            viewport_layer_groups: ProjectLayerGroups::default(),
            viewport_raw_active_keys: None,
            viewport_cpu_active_keys: None,
            viewport_label_active_keys: None,
            viewport_spatial_image_active_keys: None,
            viewport_frame_plan_ms: 0.0,
            viewport_frame_plan_ema_ms: 0.0,
            viewport_frame_plan_samples: 0,
        };

        app.configure_root_label_dataset_if_needed();
        app.rebuild_layer_orders();
        app.capture_loaded_layer_offsets();
        app.maybe_apply_auto_contrast_on_open();
        app.active_render_id = app.compute_render_id();

        // Initial fit (best effort).
        let world = app.image_world_rect_lvl0();
        if let Some(viewport) = ctx.input(|i| i.viewport().inner_rect) {
            app.camera.fit_to_world_rect(viewport, world);
        }
        app.viewport_workspace = Some(ViewportWorkspace::new(ViewerViewportState::capture(&app)));

        app
    }

    pub fn new_xenium_runtime(
        ctx: &egui::Context,
        gpu_available: bool,
        dataset_root: PathBuf,
        morphology_mip_tiff: PathBuf,
        cells_zarr_zip: Option<PathBuf>,
        transcripts_zarr_zip: Option<PathBuf>,
        pixel_size_um: f32,
        auto_contrast_settings: AutoContrastSettings,
    ) -> anyhow::Result<Self> {
        apply_napari_like_dark(ctx);

        let mut app = Self::new_tiff_runtime_named(
            ctx,
            gpu_available,
            dataset_root.clone(),
            morphology_mip_tiff,
            "xenium".to_string(),
            "morphology".to_string(),
            auto_contrast_settings,
        )?;
        app.attach_xenium_layers(
            dataset_root,
            cells_zarr_zip,
            transcripts_zarr_zip,
            pixel_size_um,
        );
        // Xenium default: cells ON, transcripts layer present but OFF until a gene is typed.
        if let Some(c) = app.xenium_layers.cells.as_mut() {
            c.visible = true;
        }
        if let Some(t) = app.xenium_layers.transcripts.as_mut() {
            t.visible = false;
        }
        Ok(app)
    }

    pub fn new_tiff_runtime(
        ctx: &egui::Context,
        gpu_available: bool,
        image_path: PathBuf,
        auto_contrast_settings: AutoContrastSettings,
    ) -> anyhow::Result<Self> {
        apply_napari_like_dark(ctx);

        let dataset_name = image_path
            .file_stem()
            .and_then(|s| s.to_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("tiff")
            .to_string();

        Self::new_tiff_runtime_named(
            ctx,
            gpu_available,
            image_path.clone(),
            image_path,
            dataset_name,
            "image".to_string(),
            auto_contrast_settings,
        )
    }

    /// Realize renderer-only TIFF loaders from metadata and an immutable pyramid prepared by the
    /// control actor's native worker adapter. No TIFF metadata is reopened on the UI thread.
    pub fn new_tiff_runtime_from_resource(
        ctx: &egui::Context,
        gpu_available: bool,
        resource: &crate::data::document::AlternateDocumentResource,
        auto_contrast_settings: AutoContrastSettings,
    ) -> anyhow::Result<Self> {
        apply_napari_like_dark(ctx);
        let assets = build_tiff_runtime_assets_from_resource(gpu_available, resource)?;
        let tiles_gl = gpu_available.then(|| TilesGl::new(RAW_TILE_CACHE_CAPACITY_TILES));
        let mut app = Self::new_runtime_with_handles(
            ctx,
            gpu_available,
            assets.dataset,
            assets.store,
            assets.loader,
            assets.raw_loader,
            tiles_gl,
            assets.hist_loader,
            assets.chanmax_loader,
            assets.chanmax_level,
            auto_contrast_settings,
        );
        app.tiff_plane_state = assets.tiff_plane_state;
        Ok(app)
    }

    pub fn new_tiff_runtime_from_prepared_resource(
        ctx: &egui::Context,
        gpu_available: bool,
        resource: &crate::data::document::AlternateDocumentResource,
        pyramid: Arc<crate::xenium::TiffPyramid>,
        auto_contrast_settings: AutoContrastSettings,
    ) -> anyhow::Result<Self> {
        apply_napari_like_dark(ctx);
        let assets =
            build_tiff_runtime_assets_from_prepared_resource(gpu_available, resource, pyramid)?;
        let tiles_gl = gpu_available.then(|| TilesGl::new(RAW_TILE_CACHE_CAPACITY_TILES));
        let mut app = Self::new_runtime_with_handles(
            ctx,
            gpu_available,
            assets.dataset,
            assets.store,
            assets.loader,
            assets.raw_loader,
            tiles_gl,
            assets.hist_loader,
            assets.chanmax_loader,
            assets.chanmax_level,
            auto_contrast_settings,
        );
        app.tiff_plane_state = assets.tiff_plane_state;
        Ok(app)
    }

    pub(super) fn new_tiff_runtime_named(
        ctx: &egui::Context,
        gpu_available: bool,
        dataset_root: PathBuf,
        image_path: PathBuf,
        dataset_name: String,
        channel_name: String,
        auto_contrast_settings: AutoContrastSettings,
    ) -> anyhow::Result<Self> {
        Self::new_tiff_runtime_named_with_plane(
            ctx,
            gpu_available,
            dataset_root,
            image_path,
            dataset_name,
            channel_name,
            crate::xenium::TiffPlaneSelection { z: 0, t: 0 },
            auto_contrast_settings,
        )
    }

    pub(super) fn new_tiff_runtime_named_with_plane(
        ctx: &egui::Context,
        gpu_available: bool,
        dataset_root: PathBuf,
        image_path: PathBuf,
        dataset_name: String,
        channel_name: String,
        plane_selection: crate::xenium::TiffPlaneSelection,
        auto_contrast_settings: AutoContrastSettings,
    ) -> anyhow::Result<Self> {
        crate::log_info!(
            "open tiff: root={} image={}",
            dataset_root.to_string_lossy(),
            image_path.to_string_lossy()
        );
        let assets = build_tiff_runtime_assets(
            gpu_available,
            dataset_root,
            image_path,
            dataset_name,
            channel_name,
            plane_selection,
        )?;
        let tiles_gl = gpu_available.then(|| TilesGl::new(RAW_TILE_CACHE_CAPACITY_TILES));

        let mut app = Self::new_runtime_with_handles(
            ctx,
            gpu_available,
            assets.dataset,
            assets.store,
            assets.loader,
            assets.raw_loader,
            tiles_gl,
            assets.hist_loader,
            assets.chanmax_loader,
            assets.chanmax_level,
            auto_contrast_settings,
        );
        app.tiff_plane_state = assets.tiff_plane_state;
        Ok(app)
    }

    pub(super) fn new_runtime_with_handles(
        ctx: &egui::Context,
        gpu_available: bool,
        dataset: OmeZarrDataset,
        store: Arc<dyn zarrs::storage::ReadableStorageTraits>,
        loader: crate::render::tiles::TileLoaderHandle,
        raw_loader: Option<RawTileLoaderHandle>,
        tiles_gl: Option<TilesGl>,
        hist_loader: HistogramLoaderHandle,
        chanmax_loader: ChannelMaxLoaderHandle,
        chanmax_level: usize,
        auto_contrast_settings: AutoContrastSettings,
    ) -> Self {
        let mut camera = Camera::default();
        camera.center_world_lvl0 = egui::pos2(0.0, 0.0);
        camera.zoom_screen_per_lvl0_px = 0.1;

        let seg_label_names = dataset
            .source
            .local_path()
            .map(discover_label_names_local)
            .unwrap_or_default();
        let seg_label_selected = if seg_label_names.iter().any(|n| n == "cells") {
            "cells".to_string()
        } else if let Some(first) = seg_label_names.first() {
            first.clone()
        } else {
            "cells".to_string()
        };
        let seg_label_input = seg_label_selected.clone();
        let seg_label_status = String::new();
        let seg_label_prompt_open = gpu_available && !seg_label_names.is_empty();

        let (label_cells, label_loader, label_cells_xform, labels_gl) = if gpu_available {
            (None, None, None, Some(LabelsGl::new(1024)))
        } else {
            (None, None, None, None)
        };

        let mut app = Self {
            dataset: dataset.clone(),
            store: store.clone(),
            remote_runtime: None,
            loader,
            raw_loader,
            label_cells,
            label_loader,
            label_cells_xform,
            seg_label_names,
            seg_label_selected,
            seg_label_input,
            seg_label_status,
            seg_label_prompt_open,
            seg_label_prompt_always: false,
            seg_label_prompt_preference: LabelPromptSessionPreference::Ask,
            hist_loader,
            chanmax_loader,
            chanmax_request_id: 1,
            chanmax_level,
            chanmax_pending: vec![false; dataset.channels.len()],
            chanmax_snapshot: dataset.channels.iter().map(|c| c.window).collect(),
            cache: TileCache::new(256),
            pending: Vec::new(),
            hist: None,
            hist_request_id: 0,
            hist_request_pending: false,
            hist_dirty: true,
            hist_navigation_dirty_since: None,
            hist_last_sent: Instant::now()
                .checked_sub(Duration::from_secs(3600))
                .unwrap_or_else(Instant::now),
            camera,
            active_render_id: 1,
            previous_render_id: None,
            active_render_smooth_pixels: true,
            previous_render_smooth_pixels: None,
            previous_view_selection: None,
            previous_displayed_view_selection: None,
            last_render_view_selection: ViewPlaneSelection {
                mode: ViewPlaneMode::Xy,
                slice_level0: 0,
            },
            last_canvas_rect: None,
            last_target_level: None,
            fallback_ceiling_level: None,
            last_visible_world_tiles: None,
            zoom_out_floor_level: None,
            zoom_out_floor_until: None,
            zoom_out_floor_visible_world_tiles: None,
            selected_channel: 0,
            view_plane_mode: ViewPlaneMode::Xy,
            draft_view_slice_level0: None,
            current_x_level0: 0,
            current_y_level0: 0,
            current_z_level0: 0,
            channels: dataset.channels.clone(),
            channel_window_overrides: HashMap::new(),
            auto_contrast_settings,
            fast_object_rendering: true,
            channel_list_search: String::new(),
            active_layer: if dataset.channels.is_empty() {
                LayerId::Points
            } else {
                LayerId::Channel(0)
            },
            selected_channel_layers: if dataset.channels.is_empty() {
                HashSet::new()
            } else {
                HashSet::from([0usize])
            },
            memory_selected_channels: (0..dataset.channels.len()).collect(),
            channel_select_anchor_idx: None,
            selected_channel_group_id: None,
            quick_contrast_target: top_bar::QuickContrastTarget::Visible,
            selected_overlay_layers: HashSet::new(),
            overlay_select_anchor_pos: None,
            show_left_panel: true,
            show_right_panel: true,
            close_dialog_open: false,
            pinned_levels: PinnedLevels::new(),
            pending_memory_load: None,
            memory_status: String::new(),
            system_memory: None,
            system_memory_last_refresh: None,
            left_tab: LeftTab::Layers,
            right_tab: RightTab::Properties,
            project_space: ProjectSpace::default(),
            project_cfg_seen: 0,
            roi_selector: RoiSelectorPanel::new(&dataset.source),
            cell_thresholds: CellThresholdsPanel::new(
                dataset
                    .source
                    .local_path()
                    .unwrap_or_else(|| std::path::Path::new("")),
                dataset.multiscale.name.as_deref(),
            ),
            cell_points: PointsLayer::new("cell_centroids"),
            annotation_layers: Vec::new(),
            next_annotation_layer_id: 1,
            mask_layers: Vec::new(),
            next_mask_layer_id: 1,
            mask_layers_project_dirty: true,
            tool_mode: ToolMode::Pan,
            drawing_mask_layer: None,
            drawing_mask_polygon: Vec::new(),
            selected_mask_polygon: None,
            selected_mask_vertex: None,
            dragging_mask_vertex: None,
            moving_mask_polygon: None,
            undo_stack: Vec::new(),
            selection_rect_start_world: None,
            selection_rect_current_world: None,
            selection_lasso_world: Vec::new(),
            threshold_region_min_pixels: 32,
            threshold_region_scope: ThresholdRegionScope::VisibleRegion,
            threshold_region_full_level: 0,
            threshold_region_status: String::new(),
            threshold_region_preview: None,
            cells_outlines_visible: true,
            cells_outlines_color_rgb: [0, 255, 0],
            cells_outlines_opacity: 0.75,
            cells_outlines_width_px: 0.0,
            points_gl: gpu_available.then(|| PointsGlRenderer::default()),
            threshold_preview_gl: gpu_available.then(|| ThresholdPreviewGlRenderer::default()),
            tiles_gl,
            labels_gl,
            remote_dialog_open: false,
            remote_mode: RemoteMode::Http,
            remote_http_url: String::new(),
            remote_s3_endpoint: String::new(),
            remote_s3_region: "auto".to_string(),
            remote_s3_bucket: String::new(),
            remote_s3_prefix: String::new(),
            remote_s3_access_key: String::new(),
            remote_s3_secret_key: String::new(),
            remote_status: String::new(),
            remote_s3_browser: None,
            pending_request: None,
            native_control_intents: Vec::new(),
            control_actor_object_generation: 0,
            control_actor_label_generation: 0,
            control_actor_object_selection_generation: 0,
            control_actor_mask_generation: 0,
            control_actor_mask_undo_available: false,
            native_mask_actor_intent_emitted: false,
            group_layers_dialog: None,
            hover_tooltip_state: None,
            active_help_topic: None,
            roi_info_open: false,
            smooth_pixels: true,
            show_tile_debug: false,
            mask_draw_debug_stats: MaskDrawDebugStats::default(),
            show_scale_bar: true,
            show_hud: true,
            tile_loader_threads: Self::default_tile_loader_threads(),
            tile_prefetch_mode: TilePrefetchMode::TargetHalo,
            tile_prefetch_aggressiveness: TilePrefetchAggressiveness::Balanced,
            tile_loading_status: String::new(),
            prefer_pinned_finer_levels: false,
            seg_geojson: GeoJsonSegmentationLayer::default(),
            seg_objects: ObjectsLayer::default(),
            spatial_image_layers: SpatialImageLayers::default(),
            spatial_layers: SpatialDataLayers::default(),
            spatial_image_transform: SpatialDataTransform2::default(),
            spatial_label_transform: SpatialDataTransform2::default(),
            spatial_root: None,
            spatial_label_store: None,
            xenium_layers: XeniumLayers::default(),
            channel_offsets_world: vec![egui::Vec2::ZERO; dataset.channels.len()],
            channel_scales: vec![egui::Vec2::splat(1.0); dataset.channels.len()],
            channel_rotations_rad: vec![0.0; dataset.channels.len()],
            loaded_layer_offsets_world: HashMap::new(),
            points_offset_world: egui::Vec2::ZERO,
            spatial_points_offset_world: egui::Vec2::ZERO,
            seg_labels_offset_world: egui::Vec2::ZERO,
            seg_geojson_offset_world: egui::Vec2::ZERO,
            seg_objects_offset_world: egui::Vec2::ZERO,
            xenium_cells_offset_world: egui::Vec2::ZERO,
            xenium_transcripts_offset_world: egui::Vec2::ZERO,
            overlay_layer_order: Vec::new(),
            channel_layer_order: (0..dataset.channels.len()).collect(),
            channel_sort_mode: ChannelSortMode::Manual,
            layer_drag: None,
            layer_move: None,
            layer_transform: None,
            tiff_plane_state: None,
            threshold_region_preview_generation: 1,
            screenshot_settings: ScreenshotSettings::default(),
            screenshot_settings_open: false,
            screenshot_worker: ScreenshotWorkerHandle::spawn(),
            screenshot_next_id: 1,
            screenshot_pending: VecDeque::new(),
            screenshot_in_flight: HashMap::new(),
            screenshot_output_dir: None,
            viewport_workspace: None,
            viewport_layer_groups: ProjectLayerGroups::default(),
            viewport_raw_active_keys: None,
            viewport_cpu_active_keys: None,
            viewport_label_active_keys: None,
            viewport_spatial_image_active_keys: None,
            viewport_frame_plan_ms: 0.0,
            viewport_frame_plan_ema_ms: 0.0,
            viewport_frame_plan_samples: 0,
        };

        app.configure_root_label_dataset_if_needed();
        app.rebuild_layer_orders();
        app.capture_loaded_layer_offsets();
        app.active_render_id = app.compute_render_id();

        // Initial fit.
        let world = app.image_world_rect_lvl0();
        if let Some(viewport) = ctx.input(|i| i.viewport().inner_rect) {
            app.camera.fit_to_world_rect(viewport, world);
        }
        app.viewport_workspace = Some(ViewportWorkspace::new(ViewerViewportState::capture(&app)));

        app
    }
}
