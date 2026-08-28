use super::*;

impl MosaicViewerApp {
    pub(super) fn from_samplesheet_context(
        ctx: &egui::Context,
        samplesheet_csv: &Path,
        columns: Option<usize>,
    ) -> anyhow::Result<Self> {
        let sheet = load_samplesheet_csv(samplesheet_csv)?;
        let samplesheet_dir = samplesheet_csv.parent().map(|p| p.to_path_buf());
        let mut items: Vec<MosaicItem> = Vec::with_capacity(sheet.rows.len());
        let mut stores: Vec<Arc<dyn zarrs::storage::ReadableStorageTraits>> =
            Vec::with_capacity(sheet.rows.len());
        for row in &sheet.rows {
            match OmeZarrDataset::open_local(&row.path) {
                Ok((ds, store)) => {
                    let id = items.len();
                    items.push(MosaicItem {
                        id,
                        sample_id: row.id.clone(),
                        meta: row.meta.clone(),
                        dataset: ds,
                        offset: egui::vec2(0.0, 0.0),
                        scale: 1.0,
                        placed_size: egui::vec2(1.0, 1.0),
                    });
                    stores.push(store);
                }
                Err(err) => eprintln!(
                    "skipping samplesheet row id='{}' path='{}': {err}",
                    row.id,
                    row.path.to_string_lossy()
                ),
            }
        }
        if items.is_empty() {
            anyhow::bail!(
                "failed to open any ROIs from samplesheet: {}",
                samplesheet_csv.to_string_lossy()
            );
        }

        let mut seg_geojson = MosaicGeoJsonSegmentationOverlay::default();
        seg_geojson.set_samplesheet_dir(samplesheet_dir);
        for it in &items {
            seg_geojson.discover_from_meta(it.id, &it.sample_id, &it.meta);
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
        for it in &mut items {
            if it.sample_id.trim().is_empty() {
                it.sample_id = it.dataset.source.display_name();
            }
        }

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
                remote_runtimes: Vec::new(),
                camera,
                mosaic_bounds,
                abs_max,
                channels,
                metadata_columns: sheet.meta_columns,
                group_blocks,
                grid_cols: cols,
                renderer_status: format!(
                    "Loaded samplesheet: {}",
                    samplesheet_csv
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("<samplesheet>")
                ),
                show_return_navigation: false,
                seg_geojson,
                consumed_mosaic_resource_generation: 0,
            },
        ))
    }
}
