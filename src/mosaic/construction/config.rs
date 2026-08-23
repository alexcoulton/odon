use super::*;

impl MosaicViewerApp {
    pub(super) fn from_config(
        cc: &eframe::CreationContext<'_>,
        args: MosaicCliArgs,
    ) -> anyhow::Result<Self> {
        apply_napari_like_dark(&cc.egui_ctx);

        let _gl = cc
            .gl
            .as_ref()
            .context("mosaic mode requires GPU (OpenGL) backend")?;

        let project_path = args
            .project_path
            .as_deref()
            .context("mosaic config mode requires --project")?;
        let mut ps = ProjectSpace::default();
        ps.load_from_file(project_path).with_context(|| {
            format!("failed to load project: {}", project_path.to_string_lossy())
        })?;
        let cfg = ps.config().clone();
        let project_dir = project_path.parent().map(|p| p.to_path_buf());
        let default_dataset = cfg
            .default_dataset
            .as_deref()
            .unwrap_or("default")
            .to_string();

        let dataset_names = args.dataset_names;
        let want_all = dataset_names.is_empty();

        let mut meta_keys: HashSet<String> = HashSet::new();
        let mut items: Vec<MosaicItem> = Vec::new();
        let mut stores: Vec<Arc<dyn zarrs::storage::ReadableStorageTraits>> = Vec::new();
        let mut remote_runtimes: Vec<Arc<tokio::runtime::Runtime>> = Vec::new();
        for roi in &cfg.rois {
            let ds_key = roi.dataset.as_deref().unwrap_or(default_dataset.as_str());
            if !want_all && !dataset_names.iter().any(|n| n == ds_key) {
                continue;
            }

            let Some(source) = roi.dataset_source().map(|source| match source {
                DatasetSource::Local(path) if path.is_relative() => DatasetSource::Local(
                    project_dir.as_ref().map(|d| d.join(&path)).unwrap_or(path),
                ),
                other => other,
            }) else {
                continue;
            };

            let opened = match &source {
                DatasetSource::Local(path) => {
                    OmeZarrDataset::open_local(path).map(|(ds, store)| (ds, store, None))
                }
                DatasetSource::Http { base_url } => build_http_store(base_url).and_then(|store| {
                    OmeZarrDataset::open_with_store(source.clone(), store.clone())
                        .map(|ds| (ds, store, None))
                }),
                DatasetSource::S3 { .. } => Err(anyhow::anyhow!(
                    "project-backed S3 mosaic requires credentials via the S3 browser path"
                )),
            };

            match opened {
                Ok((ds, store, runtime)) => {
                    let id = items.len();
                    let mut meta = roi.meta.clone();
                    if let Some(seg) = roi.segpath.as_ref() {
                        meta.insert("segpath".to_string(), seg.to_string_lossy().to_string());
                    }
                    for k in meta.keys() {
                        if !k.trim().is_empty() {
                            meta_keys.insert(k.clone());
                        }
                    }
                    items.push(MosaicItem {
                        id,
                        sample_id: roi
                            .display_name
                            .as_deref()
                            .unwrap_or(roi.id.as_str())
                            .to_string(),
                        meta,
                        dataset: ds,
                        offset: egui::vec2(0.0, 0.0),
                        scale: 1.0,
                        placed_size: egui::vec2(1.0, 1.0),
                    });
                    stores.push(store);
                    if let Some(runtime) = runtime {
                        remote_runtimes.push(runtime);
                    }
                }
                Err(err) => eprintln!(
                    "skipping ROI id='{}' source='{}': {err}",
                    roi.id,
                    roi.source_display()
                ),
            }
        }
        if items.len() < 2 {
            anyhow::bail!(
                "need at least 2 valid ROIs to open mosaic (filtered by datasets={:?})",
                dataset_names
            );
        }

        let mut meta_columns = meta_keys.into_iter().collect::<Vec<_>>();
        meta_columns.sort();

        let mut seg_geojson = MosaicGeoJsonSegmentationOverlay::default();
        seg_geojson.set_samplesheet_dir(project_dir);
        for it in &items {
            seg_geojson.discover_from_meta(it.id, &it.meta);
        }

        let abs_max = items
            .iter()
            .map(|it| it.dataset.abs_max)
            .fold(0.0f32, f32::max)
            .max(1.0);

        let channels: Vec<GlobalChannel> =
            build_global_channels(items.iter().map(|it| &it.dataset));

        let n = items.len();
        let cols = args
            .columns
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
        if let Some(viewport) = cc.egui_ctx.input(|i| i.viewport().inner_rect) {
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
                metadata_columns: meta_columns,
                group_blocks,
                grid_cols: cols,
                grid_cell_w: cell_w,
                grid_cell_h: cell_h,
                grid_pad: pad,
                status: "Ready.".to_string(),
                allow_back: false,
                seg_geojson,
                control_actor_generation: 0,
            },
        ))
    }
}
