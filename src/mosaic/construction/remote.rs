use super::*;

impl MosaicViewerApp {
    pub fn from_remote_s3_sources(
        ctx: &egui::Context,
        gpu_available: bool,
        datasets: Vec<S3DatasetSelection>,
        columns: Option<usize>,
    ) -> anyhow::Result<Self> {
        apply_napari_like_dark(ctx);
        if !gpu_available {
            anyhow::bail!("mosaic mode requires GPU (OpenGL) backend");
        }

        let mut items: Vec<MosaicItem> = Vec::new();
        let mut stores: Vec<Arc<dyn zarrs::storage::ReadableStorageTraits>> = Vec::new();
        let mut remote_runtimes: Vec<Arc<tokio::runtime::Runtime>> = Vec::new();

        for spec in datasets {
            let endpoint = spec.endpoint.trim().to_string();
            let region = spec.region.trim().to_string();
            let bucket = spec.bucket.trim().to_string();
            let prefix = spec.prefix.trim().trim_matches('/').to_string();
            let access_key = spec.access_key.trim().to_string();
            let secret_key = spec.secret_key.trim().to_string();

            let crate::data::remote_store::S3Store { store, runtime } = build_s3_store(
                &endpoint,
                &region,
                &bucket,
                &prefix,
                &access_key,
                &secret_key,
            )?;
            let source = DatasetSource::S3 {
                endpoint,
                region: if region.is_empty() {
                    "auto".to_string()
                } else {
                    region
                },
                bucket,
                prefix,
            };
            let ds = OmeZarrDataset::open_with_store(source, store.clone())?;
            let id = items.len();
            let sample_id = ds.source.display_name();
            items.push(MosaicItem {
                id,
                sample_id,
                meta: Default::default(),
                dataset: ds,
                offset: egui::vec2(0.0, 0.0),
                scale: 1.0,
                placed_size: egui::vec2(1.0, 1.0),
            });
            stores.push(store);
            remote_runtimes.push(runtime);
        }

        if items.len() < 2 {
            anyhow::bail!("need at least 2 valid S3 OME-Zarr roots to open mosaic");
        }

        let abs_max = items
            .iter()
            .map(|it| it.dataset.abs_max)
            .fold(0.0f32, f32::max)
            .max(1.0);

        let channels: Vec<GlobalChannel> =
            build_global_channels(items.iter().map(|it| &it.dataset));

        let n = items.len();
        let cols = columns
            .filter(|&c| c > 0)
            .unwrap_or_else(|| ((n as f32).sqrt().ceil() as usize).max(1));

        let pad = 64.0f32;
        let (cell_w, cell_h) = max_level0_size_items(&items);
        let cell_w = cell_w.max(1.0);
        let cell_h = cell_h.max(1.0);

        let (mosaic_bounds, group_blocks) = layout_items_grouped(
            &mut items,
            cols,
            cell_w,
            cell_h,
            pad,
            None,
            0.0,
            MosaicLayoutMode::FitCells,
        );

        let sources = Arc::new(
            items
                .iter()
                .zip(stores.iter())
                .map(|(it, store)| MosaicSource {
                    source: it.dataset.source.clone(),
                    store: store.clone(),
                    levels: it.dataset.levels.clone(),
                    dims: it.dataset.dims.clone(),
                    channel_map: build_channel_map(&channels, &it.dataset),
                })
                .collect::<Vec<_>>(),
        );

        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8)
            .clamp(2, 16);
        let pinned_levels = MosaicPinnedLevels::new();
        let loader = spawn_mosaic_raw_tile_loader(
            Arc::clone(&sources),
            pinned_levels.clone(),
            threads,
            8192,
        )?;

        let mut camera = Camera::default();
        if let Some(viewport) = ctx.input(|i| i.viewport().inner_rect) {
            camera.fit_to_world_rect(viewport, mosaic_bounds);
        } else {
            camera.center_world_lvl0 = mosaic_bounds.center();
            camera.zoom_screen_per_lvl0_px = 0.01;
        }

        Ok(Self::from_prepared_construction(
            PreparedMosaicConstruction {
                items,
                sources,
                pinned_levels,
                loader,
                remote_runtimes,
                camera,
                mosaic_bounds,
                abs_max,
                channels,
                metadata_columns: Vec::new(),
                group_blocks,
                grid_cols: cols,
                grid_cell_w: cell_w,
                grid_cell_h: cell_h,
                grid_pad: pad,
                status: "Ready.".to_string(),
                allow_back: true,
                seg_geojson: MosaicGeoJsonSegmentationOverlay::default(),
                control_actor_generation: 0,
            },
        ))
    }
}
