use super::*;
use std::collections::HashMap;
use std::path::Path;

pub(super) fn spawn_resource_workers(
    load_job_rx: Receiver<LoadJob>,
    load_tx: Sender<LoadCompletion>,
    object_loader: Option<Arc<dyn ObjectResourceLoader>>,
    dataset_inspector: Arc<dyn DatasetInspector>,
    remote_backend: Arc<dyn RemoteDatasetBackend>,
    alternate_backend: Arc<dyn AlternateDatasetBackend>,
) -> anyhow::Result<()> {
    for index in 0..LOAD_WORKERS {
        let jobs = load_job_rx.clone();
        let completions = load_tx.clone();
        let object_loader = object_loader.clone();
        let dataset_inspector = Arc::clone(&dataset_inspector);
        let remote_backend = Arc::clone(&remote_backend);
        let alternate_backend = Arc::clone(&alternate_backend);
        thread::Builder::new()
            .name(format!("odon-resource-worker-{index}"))
            .spawn(move || {
                while let Ok(job) = jobs.recv() {
                    match job {
                        LoadJob::DatasetInspect {
                            operation_generation,
                            operation_scope,
                            request,
                            path,
                        } => {
                            let result = dataset_inspector.inspect(&path);
                            if completions
                                .send(LoadCompletion::DatasetInspect {
                                    operation_generation,
                                    operation_scope,
                                    request,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::DeepLinkResolve {
                            operation_generation,
                            operation_scope,
                            request,
                            deep_link,
                            current_project,
                        } => {
                            let result = resolve_deep_link_on_worker(deep_link, current_project);
                            if completions
                                .send(LoadCompletion::DeepLinkResolve {
                                    operation_generation,
                                    operation_scope,
                                    request,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::DeepLinkApply {
                            operation_generation,
                            guard,
                            request,
                            spec,
                        } => {
                            let result = apply_deep_link_on_worker(
                                spec,
                                object_loader.as_deref(),
                                dataset_inspector.as_ref(),
                                remote_backend.as_ref(),
                                alternate_backend.as_ref(),
                            );
                            if completions
                                .send(LoadCompletion::DeepLinkApply {
                                    operation_generation,
                                    guard,
                                    request,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::OmeZarr {
                            generation,
                            request,
                            path,
                        } => {
                            let result = open_local_ome_zarr(&path).map(|opened| {
                                complete_ome_zarr_resource(
                                    opened,
                                    discover_label_names_local(&path),
                                )
                            });
                            if completions
                                .send(LoadCompletion::OmeZarr {
                                    generation,
                                    request,
                                    path,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::Tiff {
                            generation,
                            request,
                            path,
                            z,
                            t,
                        } => {
                            let result = alternate_backend.open_tiff(&path, z, t);
                            if completions
                                .send(LoadCompletion::Tiff {
                                    generation,
                                    request,
                                    path,
                                    z,
                                    t,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::SpatialData {
                            generation,
                            request,
                            path,
                            options,
                        } => {
                            let result = alternate_backend.open_spatialdata(&path, &options);
                            if completions
                                .send(LoadCompletion::SpatialData {
                                    generation,
                                    request,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::Xenium {
                            generation,
                            request,
                            path,
                            options,
                        } => {
                            let result = alternate_backend.open_xenium(&path, &options);
                            if completions
                                .send(LoadCompletion::Xenium {
                                    generation,
                                    request,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ProjectRoiOpen {
                            operation_generation,
                            scope,
                            request,
                            spec,
                        } => {
                            let result = open_project_roi_on_worker(
                                spec,
                                object_loader.as_deref(),
                                dataset_inspector.as_ref(),
                                remote_backend.as_ref(),
                                alternate_backend.as_ref(),
                            );
                            if completions
                                .send(LoadCompletion::ProjectRoiOpen {
                                    operation_generation,
                                    scope,
                                    request,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::RemoteList {
                            session_generation,
                            operation_generation,
                            operation_scope,
                            request,
                            credentials,
                            prefix,
                        } => {
                            let result =
                                remote_backend
                                    .list_s3(&credentials, &prefix)
                                    .map_err(|error| {
                                        anyhow::anyhow!(
                                            credentials.redact_message(&error.to_string())
                                        )
                                    });
                            if completions
                                .send(LoadCompletion::RemoteList {
                                    session_generation,
                                    operation_generation,
                                    operation_scope,
                                    request,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::RemoteOpen {
                            generation,
                            session_generation,
                            request,
                            spec,
                        } => {
                            let (identity, result) = match spec {
                                RemoteOpenSpec::Http { url } => {
                                    let result = remote_backend.open_http(&url).map(|opened| {
                                        complete_ome_zarr_resource(opened, Vec::new())
                                    });
                                    (RemoteOpenIdentity::Http { url }, result)
                                }
                                RemoteOpenSpec::S3 {
                                    credentials,
                                    prefix,
                                } => {
                                    let identity = RemoteOpenIdentity::S3 {
                                        endpoint: credentials.endpoint.clone(),
                                        region: credentials.region.clone(),
                                        bucket: credentials.bucket.clone(),
                                        prefix: prefix.clone(),
                                    };
                                    let result = remote_backend
                                        .open_s3(&credentials, &prefix)
                                        .map(|opened| {
                                            complete_ome_zarr_resource(opened, Vec::new())
                                        })
                                        .map_err(|error| {
                                            anyhow::anyhow!(
                                                credentials.redact_message(&error.to_string())
                                            )
                                        });
                                    (identity, result)
                                }
                            };
                            if completions
                                .send(LoadCompletion::RemoteOpen {
                                    generation,
                                    session_generation,
                                    request,
                                    identity,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ChannelIntensity {
                            generation,
                            request,
                            document,
                            spec,
                        } => {
                            let result = read_channel_intensity_stats(&document, &spec);
                            if completions
                                .send(LoadCompletion::ChannelIntensity {
                                    generation,
                                    request,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ProjectOpen {
                            generation,
                            request,
                            path,
                        } => {
                            let result = read_project_file(&path);
                            if completions
                                .send(LoadCompletion::ProjectOpen {
                                    generation,
                                    request,
                                    path,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ProjectSave {
                            generation,
                            request,
                            path,
                            payload,
                            saved_config_generation,
                            platform_effect,
                        } => {
                            let result = serde_json::to_string_pretty(&payload)
                                .map_err(anyhow::Error::from)
                                .and_then(|text| {
                                    fs::write(&path, text).map_err(anyhow::Error::from)
                                });
                            if completions
                                .send(LoadCompletion::ProjectSave {
                                    generation,
                                    request,
                                    path,
                                    saved_config_generation,
                                    platform_effect,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::SettingsSave {
                            generation,
                            request,
                            path,
                            settings,
                            response,
                        } => {
                            let result = settings.save_to(&path).map(|()| path);
                            if completions
                                .send(LoadCompletion::SettingsSave {
                                    generation,
                                    request,
                                    settings,
                                    response,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::SettingsPersist {
                            generation,
                            path,
                            settings,
                            response,
                        } => {
                            let result = settings.save_to(&path).map(|()| path);
                            if completions
                                .send(LoadCompletion::SettingsPersist {
                                    generation,
                                    settings,
                                    response,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ScreenshotSettingsValidate {
                            generation,
                            request,
                            preferences,
                        } => {
                            let result = preferences
                                .validate_output_dir()
                                .map_err(|error| anyhow::anyhow!(error.message));
                            if completions
                                .send(LoadCompletion::ScreenshotSettingsValidate {
                                    generation,
                                    request,
                                    preferences,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::MemoryPin {
                            request,
                            document,
                            spec,
                        } => {
                            let result = load_pinned_memory_on_worker(&document, &spec, &request);
                            if completions
                                .send(LoadCompletion::MemoryPin {
                                    request,
                                    spec,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::MosaicMemoryPin { request, spec } => {
                            let result = load_mosaic_pinned_memory_on_worker(&spec, &request);
                            if completions
                                .send(LoadCompletion::MosaicMemoryPin {
                                    request,
                                    spec,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ThresholdLoad {
                            request,
                            document,
                            spec,
                        } => {
                            let result =
                                load_threshold_preview_on_worker(&document, &spec, &request);
                            if completions
                                .send(LoadCompletion::ThresholdLoad {
                                    request,
                                    spec,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ThresholdRecompute { request, spec } => {
                            let result = recompute_threshold_preview_on_worker(&spec, &request);
                            if completions
                                .send(LoadCompletion::ThresholdRecompute {
                                    request,
                                    spec,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ThresholdApply { request, spec } => {
                            let result = apply_threshold_preview_on_worker(&spec, &request);
                            if completions
                                .send(LoadCompletion::ThresholdApply {
                                    request,
                                    spec,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::AnalysisCompute {
                            request,
                            spec,
                            kind,
                            params,
                        } => {
                            let result = compute_analysis_on_worker(&spec, kind, &params, &request);
                            if completions
                                .send(LoadCompletion::AnalysisCompute {
                                    request,
                                    spec,
                                    kind,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::AnalysisPresetImport {
                            request,
                            spec,
                            path,
                        } => {
                            let result = read_analysis_preset_on_worker(&path);
                            if completions
                                .send(LoadCompletion::AnalysisPresetImport {
                                    request,
                                    spec,
                                    path,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::AnalysisPresetExport {
                            request,
                            spec,
                            path,
                            overwrite,
                            state,
                        } => {
                            let result = write_analysis_preset_on_worker(&path, overwrite, &state);
                            if completions
                                .send(LoadCompletion::AnalysisPresetExport {
                                    request,
                                    spec,
                                    path,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::Measurement {
                            request,
                            document,
                            spec,
                        } => {
                            let result = measure_objects_on_worker(&document, &spec, &request);
                            if completions
                                .send(LoadCompletion::Measurement {
                                    request,
                                    spec,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ObjectExport { request, spec } => {
                            let result =
                                write_object_export(&spec, || worker_request_cancelled(&request));
                            if completions
                                .send(LoadCompletion::ObjectExport {
                                    request,
                                    spec,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::MosaicSamplesheet {
                            generation,
                            request,
                            path,
                        } => {
                            let columns = request
                                .command
                                .params()
                                .get("columns")
                                .or_else(|| request.command.params().get("cols"))
                                .and_then(Value::as_u64)
                                .and_then(|columns| usize::try_from(columns).ok())
                                .filter(|columns| *columns > 0);
                            let result =
                                open_mosaic_samplesheet_on_worker(generation, &path, columns);
                            if completions
                                .send(LoadCompletion::MosaicOpen {
                                    generation,
                                    request,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::MosaicProject {
                            generation,
                            request,
                            rois,
                            project_dir,
                            s3_session,
                        } => {
                            let result = open_mosaic_project_on_worker(
                                generation,
                                rois,
                                project_dir,
                                s3_session,
                                remote_backend.as_ref(),
                            );
                            if completions
                                .send(LoadCompletion::MosaicOpen {
                                    generation,
                                    request,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::MosaicObjects { request, spec } => {
                            let result =
                                load_mosaic_objects_on_worker(&spec, object_loader.as_deref());
                            if completions
                                .send(LoadCompletion::MosaicObjects {
                                    request,
                                    spec,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::SamplesheetInspect {
                            request,
                            path,
                            offset,
                            limit,
                        } => {
                            let result = inspect_samplesheet(&path, offset, limit);
                            if completions
                                .send(LoadCompletion::SamplesheetInspect { request, result })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::SamplesheetImport {
                            generation,
                            request,
                            path,
                            default_dataset,
                        } => {
                            let result = import_samplesheet_rois(&path, &default_dataset);
                            if completions
                                .send(LoadCompletion::SamplesheetImport {
                                    generation,
                                    request,
                                    path,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::SamplesheetExport {
                            generation,
                            request,
                            path,
                            rois,
                            overwrite,
                        } => {
                            let result = export_samplesheet_rois(&path, &rois, overwrite);
                            if completions
                                .send(LoadCompletion::SamplesheetExport {
                                    generation,
                                    request,
                                    path,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ProjectDiscovery {
                            generation,
                            request,
                            root,
                        } => {
                            let result = discover_omezarr_roots_under(&root);
                            if completions
                                .send(LoadCompletion::ProjectDiscovery {
                                    generation,
                                    request,
                                    root,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ProjectObjectSourceScan {
                            scope,
                            request,
                            candidates,
                        } => {
                            let result = scan_project_object_sources(candidates);
                            if completions
                                .send(LoadCompletion::ProjectObjectSourceScan {
                                    scope,
                                    request,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ProjectObjectPreload {
                            generation,
                            scope,
                            request,
                            settings,
                            candidates,
                        } => {
                            let result = preload_project_objects(
                                object_loader.as_deref(),
                                &request,
                                settings,
                                candidates,
                            );
                            if completions
                                .send(LoadCompletion::ProjectObjectPreload {
                                    generation,
                                    scope,
                                    request,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ObjectResource {
                            document_generation,
                            resource_generation,
                            request,
                            path,
                            downsample_factor,
                            options,
                        } => {
                            let result = object_loader.as_ref().map_or_else(
                                || anyhow::bail!("object resource loader is unavailable"),
                                |loader| {
                                    loader.load_with_options(
                                        path.clone(),
                                        downsample_factor,
                                        options,
                                    )
                                },
                            );
                            if completions
                                .send(LoadCompletion::ObjectResource {
                                    document_generation,
                                    resource_generation,
                                    request,
                                    path,
                                    downsample_factor,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::Labels {
                            document_generation,
                            label_generation,
                            request,
                            document,
                            name,
                        } => {
                            let result = if document.dataset().is_root_label_mask() {
                                let labels =
                                    LabelZarrDataset::from_root_dataset(document.dataset());
                                if labels.label_name == name {
                                    Ok(ControlLabelResource {
                                        dataset: labels,
                                        store: Arc::clone(document.store()),
                                    })
                                } else {
                                    Err(anyhow::anyhow!(
                                        "top-level label mask is named '{}', not '{}'",
                                        labels.label_name,
                                        name
                                    ))
                                }
                            } else {
                                match LabelZarrDataset::try_open(
                                    Arc::clone(document.store()),
                                    &name,
                                ) {
                                    Ok(Some(dataset)) => Ok(ControlLabelResource {
                                        dataset,
                                        store: Arc::clone(document.store()),
                                    }),
                                    Ok(None) => {
                                        Err(anyhow::anyhow!("no labels/{name} found in this ROI"))
                                    }
                                    Err(error) => Err(error),
                                }
                            };
                            if completions
                                .send(LoadCompletion::Labels {
                                    document_generation,
                                    label_generation,
                                    request,
                                    name,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ObjectFilter {
                            document_generation,
                            resource_generation,
                            operation_generation,
                            viewport_id,
                            expected_presentation_revision,
                            request,
                            resource,
                            model,
                        } => {
                            let result = object_loader.as_ref().map_or_else(
                                || anyhow::bail!("object filter evaluator is unavailable"),
                                |loader| loader.evaluate_filter(resource, model),
                            );
                            if completions
                                .send(LoadCompletion::ObjectFilter {
                                    document_generation,
                                    resource_generation,
                                    operation_generation,
                                    viewport_id,
                                    expected_presentation_revision,
                                    request,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ObjectSelectionFilter {
                            document_generation,
                            resource_generation,
                            selection_generation,
                            operation_generation,
                            request,
                            resource,
                            model,
                            mode,
                            limit,
                        } => {
                            let result = object_loader.as_ref().map_or_else(
                                || anyhow::bail!("object filter evaluator is unavailable"),
                                |loader| loader.evaluate_filter(resource, model),
                            );
                            if completions
                                .send(LoadCompletion::ObjectSelectionFilter {
                                    document_generation,
                                    resource_generation,
                                    selection_generation,
                                    operation_generation,
                                    request,
                                    mode,
                                    limit,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::MaskImport {
                            document_generation,
                            mask_generation,
                            operation_generation,
                            operation_scope,
                            request,
                            path,
                            name,
                            editable,
                            downsample_factor,
                        } => {
                            let result =
                                crate::model::load_geojson_mask_polylines(&path, downsample_factor);
                            if completions
                                .send(LoadCompletion::MaskImport {
                                    document_generation,
                                    mask_generation,
                                    operation_generation,
                                    operation_scope,
                                    request,
                                    path,
                                    name,
                                    editable,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::MaskExport {
                            operation_generation,
                            operation_scope,
                            request,
                            path,
                            layer_id,
                            layers,
                            overwrite,
                        } => {
                            let layer_count = layers.len();
                            let polygon_count =
                                layers.iter().map(|layer| layer.polygons_world.len()).sum();
                            let result = export_mask_layers_geojson(&path, &layers, overwrite);
                            if completions
                                .send(LoadCompletion::MaskExport {
                                    operation_generation,
                                    operation_scope,
                                    request,
                                    path,
                                    layer_id,
                                    layer_count,
                                    polygon_count,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                    }
                }
            })?;
    }

    Ok(())
}

fn load_pinned_memory_on_worker(
    document: &RenderDocument,
    spec: &MemoryPinSpec,
    request: &OdonControlRequest,
) -> anyhow::Result<MemoryPinWorkerResult> {
    let cancelled = || {
        request
            .task_id
            .as_deref()
            .and_then(|task_id| request.task_registry.get(task_id).ok())
            .is_some_and(|task| task.state == TaskState::Cancelled)
    };
    anyhow::ensure!(!cancelled(), "memory pinning was cancelled");
    let system = {
        use sysinfo::System;
        let mut system = System::new();
        system.refresh_memory();
        (system.total_memory() > 0).then_some(SystemMemorySnapshot {
            total_bytes: system.total_memory(),
            available_bytes: system.available_memory(),
        })
    };
    let projected_bytes = spec.pinned_bytes.saturating_add(spec.estimated_bytes);
    let risk = system.and_then(|system| {
        if projected_bytes > system.available_bytes {
            Some("danger")
        } else if projected_bytes.saturating_mul(100)
            >= system.available_bytes.max(1).saturating_mul(75)
        {
            Some("warning")
        } else {
            None
        }
    });
    if let (false, Some(risk), Some(system)) = (spec.force, risk, system) {
        return Ok(MemoryPinWorkerResult {
            system: Some(system),
            outcome: MemoryPinWorkerOutcome::Confirmation {
                risk,
                projected_bytes,
                available_bytes: system.available_bytes,
            },
        });
    }

    let dataset = document.dataset();
    let info = dataset
        .levels
        .get(spec.level)
        .ok_or_else(|| anyhow::anyhow!("missing level {}", spec.level))?;
    let zarr_path = format!("/{}", info.path.trim_start_matches('/'));
    let array: Array<dyn ReadableStorageTraits> =
        Array::open(Arc::clone(document.store()), &zarr_path)?;
    let height = *info.shape.get(dataset.dims.y).unwrap_or(&0) as usize;
    let width = *info.shape.get(dataset.dims.x).unwrap_or(&0) as usize;
    let plane_len = height.saturating_mul(width);
    let mut raw = Vec::new();
    let mut channel_offsets = HashMap::new();

    if let Some(channel_dimension) = dataset.dims.c {
        for &channel in &spec.channel_ids {
            anyhow::ensure!(!cancelled(), "memory pinning was cancelled");
            let mut ranges = Vec::with_capacity(info.shape.len());
            for dimension in 0..info.shape.len() {
                if dimension == channel_dimension {
                    ranges.push(channel..channel.saturating_add(1));
                } else if dimension == dataset.dims.y || dimension == dataset.dims.x {
                    ranges.push(0..info.shape[dimension]);
                } else {
                    ranges.push(0..1);
                }
            }
            let subset = ArraySubset::new_with_ranges(&ranges);
            let data = retrieve_image_subset_u16(&array, &subset, &info.dtype)?;
            let plane = squeeze_pinned_plane(data, dataset.dims.y, dataset.dims.x)
                .ok_or_else(|| anyhow::anyhow!("unexpected pinned level dimensionality"))?;
            let (plane_raw, offset) = plane.into_raw_vec_and_offset();
            anyhow::ensure!(
                offset.unwrap_or(0) == 0,
                "non-zero pinned level buffer offset"
            );
            anyhow::ensure!(
                plane_raw.len() == plane_len,
                "unexpected pinned plane length"
            );
            channel_offsets.insert(channel, raw.len() / plane_len.max(1));
            raw.extend_from_slice(&plane_raw);
        }
    } else {
        anyhow::ensure!(!cancelled(), "memory pinning was cancelled");
        let mut ranges = Vec::with_capacity(info.shape.len());
        for dimension in 0..info.shape.len() {
            if dimension == dataset.dims.y || dimension == dataset.dims.x {
                ranges.push(0..info.shape[dimension]);
            } else {
                ranges.push(0..1);
            }
        }
        let subset = ArraySubset::new_with_ranges(&ranges);
        let data = retrieve_image_subset_u16(&array, &subset, &info.dtype)?;
        let plane = squeeze_pinned_plane(data, dataset.dims.y, dataset.dims.x)
            .ok_or_else(|| anyhow::anyhow!("unexpected pinned level dimensionality"))?;
        let (plane_raw, offset) = plane.into_raw_vec_and_offset();
        anyhow::ensure!(
            offset.unwrap_or(0) == 0,
            "non-zero pinned level buffer offset"
        );
        raw = plane_raw;
        for &channel in &spec.channel_ids {
            channel_offsets.insert(channel, 0);
        }
    }
    anyhow::ensure!(
        !channel_offsets.is_empty(),
        "none of the selected channels were pinned"
    );
    Ok(MemoryPinWorkerResult {
        system,
        outcome: MemoryPinWorkerOutcome::Loaded(ControlPinnedLevelResource::new(
            spec.level,
            width,
            height,
            channel_offsets,
            raw,
        )),
    })
}

fn load_mosaic_pinned_memory_on_worker(
    spec: &MosaicMemoryPinSpec,
    request: &OdonControlRequest,
) -> anyhow::Result<MosaicMemoryPinWorkerResult> {
    let cancelled = || {
        request
            .task_id
            .as_deref()
            .and_then(|task_id| request.task_registry.get(task_id).ok())
            .is_some_and(|task| task.state == TaskState::Cancelled)
    };
    anyhow::ensure!(!cancelled(), "mosaic memory pinning was cancelled");
    let system = {
        use sysinfo::System;
        let mut system = System::new();
        system.refresh_memory();
        (system.total_memory() > 0).then_some(SystemMemorySnapshot {
            total_bytes: system.total_memory(),
            available_bytes: system.available_memory(),
        })
    };
    let projected_bytes = spec.pinned_bytes.saturating_add(spec.estimated_bytes);
    let risk = system.and_then(|system| {
        if projected_bytes > system.available_bytes {
            Some("danger")
        } else if projected_bytes.saturating_mul(100)
            >= system.available_bytes.max(1).saturating_mul(75)
        {
            Some("warning")
        } else {
            None
        }
    });
    if let (false, Some(risk), Some(system)) = (spec.force, risk, system) {
        return Ok(MosaicMemoryPinWorkerResult {
            system: Some(system),
            outcome: MosaicMemoryPinWorkerOutcome::Confirmation {
                risk,
                projected_bytes,
                available_bytes: system.available_bytes,
            },
        });
    }

    let mut loaded = Vec::new();
    let mut failures = Vec::new();
    for item in &spec.items {
        anyhow::ensure!(!cancelled(), "mosaic memory pinning was cancelled");
        match load_mosaic_pinned_item(item, spec.level, &spec.channel_ids, &cancelled) {
            Ok(resource) => loaded.push((item.item_id, resource)),
            Err(error) => failures.push((item.item_id, error.to_string())),
        }
    }
    anyhow::ensure!(
        !loaded.is_empty(),
        "failed to pin the requested level for every selected mosaic ROI{}",
        failures
            .first()
            .map(|(_, error)| format!("; first failure: {error}"))
            .unwrap_or_default()
    );
    Ok(MosaicMemoryPinWorkerResult {
        system,
        outcome: MosaicMemoryPinWorkerOutcome::Loaded(MosaicMemoryPinResult { loaded, failures }),
    })
}

fn load_mosaic_pinned_item(
    item: &crate::model::MosaicMemoryPinItemSpec,
    level: usize,
    selected_global_channels: &[u64],
    cancelled: &impl Fn() -> bool,
) -> anyhow::Result<ControlPinnedLevelResource> {
    let descriptor = &item.document.descriptor;
    let info = descriptor
        .levels
        .get(level)
        .ok_or_else(|| anyhow::anyhow!("missing level {level}"))?;
    let zarr_path = format!("/{}", info.path.trim_start_matches('/'));
    let array: Array<dyn ReadableStorageTraits> =
        Array::open(Arc::clone(item.document.resource.store()), &zarr_path)?;
    let height = *info.shape.get(descriptor.dims.y).unwrap_or(&0) as usize;
    let width = *info.shape.get(descriptor.dims.x).unwrap_or(&0) as usize;
    let plane_len = height.saturating_mul(width);
    let mut raw = Vec::new();
    let mut channel_offsets = HashMap::new();

    if let Some(channel_dimension) = descriptor.dims.c {
        for &global_channel in selected_global_channels {
            anyhow::ensure!(!cancelled(), "mosaic memory pinning was cancelled");
            let Some(local_channel) = item
                .channel_map
                .get(global_channel as usize)
                .copied()
                .flatten()
            else {
                continue;
            };
            let mut ranges = Vec::with_capacity(info.shape.len());
            for dimension in 0..info.shape.len() {
                if dimension == channel_dimension {
                    ranges.push(local_channel..local_channel.saturating_add(1));
                } else if dimension == descriptor.dims.y || dimension == descriptor.dims.x {
                    ranges.push(0..info.shape[dimension]);
                } else {
                    ranges.push(0..1);
                }
            }
            let subset = ArraySubset::new_with_ranges(&ranges);
            let data = retrieve_image_subset_u16(&array, &subset, &info.dtype)?;
            let plane = squeeze_pinned_plane(data, descriptor.dims.y, descriptor.dims.x)
                .ok_or_else(|| anyhow::anyhow!("unexpected pinned mosaic level dimensionality"))?;
            let (plane_raw, offset) = plane.into_raw_vec_and_offset();
            anyhow::ensure!(
                offset.unwrap_or(0) == 0,
                "non-zero pinned mosaic buffer offset"
            );
            anyhow::ensure!(
                plane_raw.len() == plane_len,
                "unexpected pinned mosaic plane length"
            );
            channel_offsets.insert(global_channel, raw.len() / plane_len.max(1));
            raw.extend_from_slice(&plane_raw);
        }
    } else {
        anyhow::ensure!(!cancelled(), "mosaic memory pinning was cancelled");
        let matched = selected_global_channels
            .iter()
            .copied()
            .filter(|channel| {
                item.channel_map
                    .get(*channel as usize)
                    .copied()
                    .flatten()
                    .is_some()
            })
            .collect::<Vec<_>>();
        anyhow::ensure!(
            !matched.is_empty(),
            "none of the selected channels are present"
        );
        let mut ranges = Vec::with_capacity(info.shape.len());
        for dimension in 0..info.shape.len() {
            if dimension == descriptor.dims.y || dimension == descriptor.dims.x {
                ranges.push(0..info.shape[dimension]);
            } else {
                ranges.push(0..1);
            }
        }
        let subset = ArraySubset::new_with_ranges(&ranges);
        let data = retrieve_image_subset_u16(&array, &subset, &info.dtype)?;
        let plane = squeeze_pinned_plane(data, descriptor.dims.y, descriptor.dims.x)
            .ok_or_else(|| anyhow::anyhow!("unexpected pinned mosaic level dimensionality"))?;
        let (plane_raw, offset) = plane.into_raw_vec_and_offset();
        anyhow::ensure!(
            offset.unwrap_or(0) == 0,
            "non-zero pinned mosaic buffer offset"
        );
        raw = plane_raw;
        for channel in matched {
            channel_offsets.insert(channel, 0);
        }
    }
    anyhow::ensure!(
        !channel_offsets.is_empty(),
        "none of the selected channels were pinned"
    );
    Ok(ControlPinnedLevelResource::new(
        level,
        width,
        height,
        channel_offsets,
        raw,
    ))
}

fn squeeze_pinned_plane(
    mut data: ndarray::ArrayD<u16>,
    mut vertical_dimension: usize,
    mut horizontal_dimension: usize,
) -> Option<ndarray::Array2<u16>> {
    use ndarray::Axis;
    for dimension in (0..data.ndim()).rev() {
        if dimension == vertical_dimension || dimension == horizontal_dimension {
            continue;
        }
        if data.shape().get(dimension).copied()? != 1 {
            return None;
        }
        data = data.index_axis_move(Axis(dimension), 0);
        if dimension < vertical_dimension {
            vertical_dimension = vertical_dimension.saturating_sub(1);
        }
        if dimension < horizontal_dimension {
            horizontal_dimension = horizontal_dimension.saturating_sub(1);
        }
    }
    let mut plane = data.into_dimensionality::<ndarray::Ix2>().ok()?;
    match (vertical_dimension, horizontal_dimension) {
        (0, 1) => {}
        (1, 0) => plane.swap_axes(0, 1),
        _ => return None,
    }
    Some(plane)
}

fn load_threshold_preview_on_worker(
    document: &RenderDocument,
    spec: &ThresholdPreviewLoadSpec,
    request: &OdonControlRequest,
) -> anyhow::Result<ControlThresholdPreviewResource> {
    anyhow::ensure!(
        !worker_request_cancelled(request),
        "threshold preview was cancelled"
    );
    let array: Array<dyn ReadableStorageTraits> =
        Array::open(Arc::clone(document.store()), &spec.zarr_path)?;
    let subset = ArraySubset::new_with_ranges(&spec.ranges);
    let data = retrieve_image_subset_u16(&array, &subset, &spec.dtype)?;
    let dataset = document.dataset();
    let plane = squeeze_pinned_plane(data, dataset.dims.y, dataset.dims.x)
        .ok_or_else(|| anyhow::anyhow!("unexpected threshold preview dimensionality"))?;
    anyhow::ensure!(
        plane.dim() == (spec.height, spec.width),
        "threshold preview dimensions changed during loading"
    );
    anyhow::ensure!(
        !worker_request_cancelled(request),
        "threshold preview was cancelled"
    );
    let mask = extract_threshold_mask(&plane, spec.threshold, spec.min_component_pixels);
    let values = Arc::new(plane.iter().copied().collect());
    Ok(ControlThresholdPreviewResource {
        generation: spec.operation_generation,
        channel_index: spec.channel_index,
        channel_name: spec.channel_name.clone(),
        scope: spec.scope,
        level: spec.level,
        downsample: spec.downsample,
        x0: spec.x0,
        y0: spec.y0,
        width: spec.width,
        height: spec.height,
        values,
        included: Arc::new(mask.included),
        threshold: spec.threshold,
        min_component_pixels: spec.min_component_pixels,
    })
}

fn recompute_threshold_preview_on_worker(
    spec: &ThresholdPreviewRecomputeSpec,
    request: &OdonControlRequest,
) -> anyhow::Result<ControlThresholdPreviewResource> {
    anyhow::ensure!(
        !worker_request_cancelled(request),
        "threshold preview was cancelled"
    );
    let mut preview = (*spec.preview).clone();
    let plane = ndarray::Array2::from_shape_vec(
        (preview.height, preview.width),
        preview.values.as_ref().clone(),
    )?;
    let mask = extract_threshold_mask(&plane, preview.threshold, preview.min_component_pixels);
    anyhow::ensure!(
        !worker_request_cancelled(request),
        "threshold preview was cancelled"
    );
    preview.included = Arc::new(mask.included);
    Ok(preview)
}

fn apply_threshold_preview_on_worker(
    spec: &ThresholdPreviewApplySpec,
    request: &OdonControlRequest,
) -> anyhow::Result<Vec<Vec<[f32; 2]>>> {
    anyhow::ensure!(
        !worker_request_cancelled(request),
        "threshold apply was cancelled"
    );
    let mask = ThresholdMask {
        width: spec.preview.width,
        height: spec.preview.height,
        included: spec.preview.included.as_ref().clone(),
    };
    let polygons = threshold_mask_polygons(&mask);
    anyhow::ensure!(
        !polygons.is_empty(),
        "no visible regions found above the current threshold"
    );
    let transformed = polygons
        .into_iter()
        .map(|polygon| {
            polygon
                .into_iter()
                .map(|point| {
                    let local = [
                        (spec.preview.x0 as f32 + point[0]) * spec.preview.downsample,
                        (spec.preview.y0 as f32 + point[1]) * spec.preview.downsample,
                    ];
                    threshold_local_to_world(
                        local,
                        spec.pivot,
                        spec.offset,
                        spec.scale,
                        spec.rotation_rad,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    anyhow::ensure!(
        !worker_request_cancelled(request),
        "threshold apply was cancelled"
    );
    Ok(transformed)
}

fn worker_request_cancelled(request: &OdonControlRequest) -> bool {
    request
        .task_id
        .as_deref()
        .and_then(|task_id| request.task_registry.get(task_id).ok())
        .is_some_and(|task| task.state == TaskState::Cancelled)
}

fn threshold_local_to_world(
    local: [f32; 2],
    pivot: [f32; 2],
    offset: [f32; 2],
    scale: [f32; 2],
    rotation_rad: f32,
) -> [f32; 2] {
    let scaled = [
        (local[0] - pivot[0]) * scale[0],
        (local[1] - pivot[1]) * scale[1],
    ];
    let (sin, cos) = rotation_rad.sin_cos();
    [
        pivot[0] + scaled[0] * cos - scaled[1] * sin + offset[0],
        pivot[1] + scaled[0] * sin + scaled[1] * cos + offset[1],
    ]
}

fn compute_analysis_on_worker(
    spec: &AnalysisResourceSpec,
    kind: AnalysisComputeKind,
    params: &Value,
    request: &OdonControlRequest,
) -> anyhow::Result<Value> {
    anyhow::ensure!(!worker_request_cancelled(request), "analysis was cancelled");
    if matches!(kind, AnalysisComputeKind::Warmup) {
        let mut completed = 0usize;
        for property in spec.resource.property_names.iter() {
            if property == "id" {
                continue;
            }
            if analysis_values(spec, property, "none").next().is_some() {
                completed += 1;
            }
            anyhow::ensure!(!worker_request_cancelled(request), "analysis was cancelled");
        }
        return Ok(json!({"completed":completed}));
    }
    let property = params
        .get("property")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|property| !property.is_empty())
        .ok_or_else(|| anyhow::anyhow!("property is required"))?;
    let transform = params
        .get("transform")
        .and_then(Value::as_str)
        .unwrap_or("none");
    anyhow::ensure!(
        matches!(transform, "none" | "arcsinh"),
        "transform must be 'none' or 'arcsinh'"
    );
    let mut values = analysis_values(spec, property, transform).collect::<Vec<_>>();
    anyhow::ensure!(
        !values.is_empty(),
        "numeric property '{property}' has no finite values in the active object set"
    );
    anyhow::ensure!(!worker_request_cancelled(request), "analysis was cancelled");
    values.sort_by(f32::total_cmp);
    match kind {
        AnalysisComputeKind::Histogram => {
            let bins = params
                .get("bins")
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .unwrap_or(128);
            anyhow::ensure!(
                (8..=4096).contains(&bins),
                "bins must be an integer from 8 to 4096"
            );
            let minimum = values[0];
            let maximum = values[values.len() - 1];
            let median = quantile(&values, 0.5);
            let mut counts = vec![0u64; bins];
            if maximum <= minimum {
                counts[0] = values.len() as u64;
            } else {
                let scale = bins as f32 / (maximum - minimum);
                for value in &values {
                    let index = (((*value - minimum) * scale).floor() as usize).min(bins - 1);
                    counts[index] += 1;
                }
            }
            Ok(json!({
                "property":property,
                "transform":transform,
                "filtered":spec.filtered,
                "count":values.len(),
                "min":minimum,
                "max":maximum,
                "median":median,
                "max_bin_count":counts.iter().copied().max().unwrap_or(0),
                "bins":counts,
            }))
        }
        AnalysisComputeKind::ThresholdSuggestions => {
            let method = params
                .get("method")
                .and_then(Value::as_str)
                .unwrap_or("quantiles");
            anyhow::ensure!(
                matches!(method, "quantiles" | "kmeans"),
                "method must be 'quantiles' or 'kmeans'"
            );
            let count = params
                .get("count")
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .unwrap_or(3);
            anyhow::ensure!(
                (2..=12).contains(&count),
                "count must be an integer from 2 to 12"
            );
            let levels = if method == "quantiles" {
                (1..count)
                    .map(|index| quantile(&values, index as f32 / count as f32))
                    .collect::<Vec<_>>()
            } else {
                kmeans_thresholds(&values, count)
            };
            Ok(json!({
                "property":property,
                "method":method,
                "transform":transform,
                "filtered":spec.filtered,
                "sample_count":values.len(),
                "levels":levels,
            }))
        }
        AnalysisComputeKind::Warmup => unreachable!(),
    }
}

fn analysis_values<'a>(
    spec: &'a AnalysisResourceSpec,
    property: &'a str,
    transform: &'a str,
) -> impl Iterator<Item = f32> + 'a {
    let indices: Box<dyn Iterator<Item = usize> + 'a> = match spec.indices.as_ref() {
        Some(indices) => Box::new(indices.iter().copied()),
        None => Box::new(0..spec.resource.features.len()),
    };
    indices.filter_map(move |index| {
        let value = spec
            .resource
            .features
            .get(index)?
            .properties
            .get(property)?
            .as_f64()? as f32;
        let value = if transform == "arcsinh" {
            value.asinh()
        } else {
            value
        };
        value.is_finite().then_some(value)
    })
}

fn quantile(values: &[f32], fraction: f32) -> f32 {
    if values.len() == 1 {
        return values[0];
    }
    let position = fraction.clamp(0.0, 1.0) * (values.len() - 1) as f32;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    values[lower] + (values[upper] - values[lower]) * (position - lower as f32)
}

fn kmeans_thresholds(values: &[f32], cluster_count: usize) -> Vec<f32> {
    let clusters = cluster_count.min(values.len()).max(1);
    let mut centers = (0..clusters)
        .map(|index| quantile(values, (index as f32 + 0.5) / clusters as f32))
        .collect::<Vec<_>>();
    for _ in 0..24 {
        let mut sums = vec![0.0f64; clusters];
        let mut counts = vec![0usize; clusters];
        for value in values {
            let closest = centers
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| {
                    (*value - **left).abs().total_cmp(&(*value - **right).abs())
                })
                .map(|(index, _)| index)
                .unwrap_or(0);
            sums[closest] += *value as f64;
            counts[closest] += 1;
        }
        for index in 0..clusters {
            if counts[index] > 0 {
                centers[index] = (sums[index] / counts[index] as f64) as f32;
            }
        }
        centers.sort_by(f32::total_cmp);
    }
    centers
        .windows(2)
        .map(|pair| (pair[0] + pair[1]) * 0.5)
        .collect()
}

fn read_analysis_preset_on_worker(path: &Path) -> anyhow::Result<Value> {
    let payload: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    let name = payload
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let elements = payload
        .get("elements")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("invalid call preset: elements must be an array"))?;
    Ok(json!({
        "threshold_set_name":name,
        "threshold_elements":elements,
        "threshold_selected_element":if elements.is_empty() { Value::Null } else { json!(0) },
    }))
}

fn write_analysis_preset_on_worker(
    path: &Path,
    overwrite: bool,
    state: &Value,
) -> anyhow::Result<usize> {
    use std::io::Write;
    let elements = state
        .get("threshold_elements")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let payload = json!({
        "name":state.get("threshold_set_name").and_then(Value::as_str).unwrap_or_default(),
        "elements":elements,
    });
    let mut options = fs::OpenOptions::new();
    options.write(true);
    if overwrite {
        options.create(true).truncate(true);
    } else {
        options.create_new(true);
    }
    let mut file = options.open(path)?;
    file.write_all(serde_json::to_string_pretty(&payload)?.as_bytes())?;
    file.sync_all()?;
    Ok(elements.len())
}

fn measure_objects_on_worker(
    document: &RenderDocument,
    spec: &MeasurementSpec,
    request: &OdonControlRequest,
) -> anyhow::Result<(ControlObjectResource, usize)> {
    anyhow::ensure!(
        !worker_request_cancelled(request),
        "measurement was cancelled"
    );
    let dataset = document.dataset();
    let level = dataset
        .levels
        .get(spec.level)
        .ok_or_else(|| anyhow::anyhow!("measurement level is out of range"))?;
    let width = level.shape[dataset.dims.x] as usize;
    let height = level.shape[dataset.dims.y] as usize;
    anyhow::ensure!(
        width > 0 && height > 0,
        "measurement level has invalid dimensions"
    );
    let mut features = spec.resource.features.as_ref().clone();
    let mut property_names = spec.resource.property_names.as_ref().clone();
    let mut measured_objects = std::collections::HashSet::new();
    for channel in &dataset.channels {
        anyhow::ensure!(
            !worker_request_cancelled(request),
            "measurement was cancelled"
        );
        let mut ranges = Vec::with_capacity(level.shape.len());
        for dimension in 0..level.shape.len() {
            if Some(dimension) == dataset.dims.c {
                let selected = (channel.index as u64).min(level.shape[dimension].saturating_sub(1));
                ranges.push(selected..selected.saturating_add(1));
            } else if dimension == dataset.dims.y || dimension == dataset.dims.x {
                ranges.push(0..level.shape[dimension]);
            } else {
                ranges.push(0..level.shape[dimension].min(1));
            }
        }
        let array: Array<dyn ReadableStorageTraits> = Array::open(
            Arc::clone(document.store()),
            &format!("/{}", level.path.trim_start_matches('/')),
        )?;
        let plane = squeeze_pinned_plane(
            retrieve_image_subset_u16(
                &array,
                &ArraySubset::new_with_ranges(&ranges),
                &level.dtype,
            )?,
            dataset.dims.y,
            dataset.dims.x,
        )
        .ok_or_else(|| anyhow::anyhow!("unexpected measurement dimensionality"))?;
        let key =
            measurement_property_key(&spec.prefix, &channel.name, channel.index, &property_names);
        property_names.push(key.clone());
        for &index in spec.target_indices.iter() {
            anyhow::ensure!(
                !worker_request_cancelled(request),
                "measurement was cancelled"
            );
            let Some(feature) = features.get_mut(index) else {
                continue;
            };
            let mut values = Vec::new();
            let downsample = level.downsample.max(1e-6);
            let x0 = (feature.bbox_world[0] / downsample).floor().max(0.0) as usize;
            let y0 = (feature.bbox_world[1] / downsample).floor().max(0.0) as usize;
            let x1 = (feature.bbox_world[2] / downsample).ceil().max(0.0) as usize;
            let y1 = (feature.bbox_world[3] / downsample).ceil().max(0.0) as usize;
            for y in y0.min(height)..y1.min(height) {
                for x in x0.min(width)..x1.min(width) {
                    let world = [(x as f32 + 0.5) * downsample, (y as f32 + 0.5) * downsample];
                    if feature
                        .polygons_world
                        .iter()
                        .any(|polygon| point_in_polygon(world, polygon))
                    {
                        values.push(plane[(y, x)] as f32);
                    }
                }
            }
            if !values.is_empty() {
                let value = match spec.metric {
                    MeasurementMetric::Mean => {
                        values.iter().map(|value| *value as f64).sum::<f64>() / values.len() as f64
                    }
                    MeasurementMetric::Median => {
                        values.sort_by(f32::total_cmp);
                        quantile(&values, 0.5) as f64
                    }
                };
                feature.properties.insert(key.clone(), json!(value));
                measured_objects.insert(index);
            }
        }
    }
    property_names.sort();
    property_names.dedup();
    Ok((
        ControlObjectResource {
            source: spec.resource.source.clone(),
            downsample_factor: spec.resource.downsample_factor,
            features: Arc::new(features),
            property_names: Arc::new(property_names),
            renderer_payload: None,
        },
        measured_objects.len(),
    ))
}

fn measurement_property_key(prefix: &str, name: &str, index: usize, existing: &[String]) -> String {
    let token = name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_string();
    let base = format!("{}{token}", prefix.trim());
    if !existing.contains(&base) {
        base
    } else {
        format!("{base}_{index}")
    }
}

fn point_in_polygon(point: [f32; 2], polygon: &[[f32; 2]]) -> bool {
    if polygon.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut previous = polygon.len() - 1;
    for current in 0..polygon.len() {
        let a = polygon[current];
        let b = polygon[previous];
        if ((a[1] > point[1]) != (b[1] > point[1]))
            && point[0] < (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[0]
        {
            inside = !inside;
        }
        previous = current;
    }
    inside
}

fn scan_project_object_sources(
    candidates: Vec<PathBuf>,
) -> anyhow::Result<Vec<ProjectObjectPreloadSource>> {
    let mut sources = Vec::new();
    for path in candidates {
        match fs::metadata(&path) {
            Ok(metadata) => sources.push(ProjectObjectPreloadSource {
                path,
                bytes: metadata.len(),
            }),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(anyhow::anyhow!(
                    "could not inspect object source {}: {error}",
                    path.display()
                ));
            }
        }
    }
    Ok(sources)
}

fn preload_project_objects(
    object_loader: Option<&dyn ObjectResourceLoader>,
    request: &OdonControlRequest,
    settings: ProjectObjectPreloadSettings,
    candidates: Vec<PathBuf>,
) -> anyhow::Result<ProjectObjectPreloadWorkerResult> {
    let loader =
        object_loader.ok_or_else(|| anyhow::anyhow!("object resource loader is unavailable"))?;
    let sources = scan_project_object_sources(candidates)?;
    if sources.is_empty() {
        anyhow::bail!(
            "project has no preload-eligible Parquet or GeoParquet segmentation paths on disk"
        );
    }
    let mut resources = Vec::new();
    let mut failures = Vec::new();
    let total = sources.len();
    for (index, source) in sources.iter().enumerate() {
        if request
            .task_id
            .as_deref()
            .and_then(|task_id| request.task_registry.get(task_id).ok())
            .is_some_and(|task| task.state == TaskState::Cancelled)
        {
            break;
        }
        match loader.load_with_options(source.path.clone(), 1.0, Some(settings.worker_options())) {
            Ok(resource) => resources.push((source.path.clone(), resource)),
            Err(error) => failures.push((source.path.clone(), error.to_string())),
        }
        if let Some(task_id) = request.task_id.as_deref() {
            let _ = request.task_registry.progress(
                task_id,
                Some((index + 1) as f64 / total as f64),
                format!("preloaded {} of {total} project object sources", index + 1),
            );
        }
    }
    Ok(ProjectObjectPreloadWorkerResult {
        sources,
        resources,
        failures,
    })
}

pub(super) fn open_project_roi_on_worker(
    spec: ProjectRoiOpenSpec,
    object_loader: Option<&dyn ObjectResourceLoader>,
    dataset_inspector: &dyn DatasetInspector,
    remote_backend: &dyn RemoteDatasetBackend,
    alternate_backend: &dyn AlternateDatasetBackend,
) -> anyhow::Result<ProjectRoiOpenWorkerResult> {
    let ProjectRoiOpenSpec {
        roi,
        source,
        saved_view,
        object_path,
        cached_object,
        s3_session,
        requested_label,
    } = spec;
    let mut label_available = Vec::new();
    let mut label_resource = None;
    let mut s3_session_generation = None;
    let opened = match &source {
        DatasetSource::Local(requested_path) => {
            let path = normalize_local_dataset_path(requested_path)
                .unwrap_or_else(|| requested_path.clone());
            match classify_local_dataset_path(&path) {
                Some(LocalDatasetKind::OmeZarr) => {
                    let (opened, available, root) =
                        complete_ome_zarr_resource(open_local_ome_zarr(&path)?, Vec::new());
                    label_available = available;
                    label_resource = root;
                    opened.into_control()
                }
                Some(LocalDatasetKind::Tiff) => {
                    alternate_backend.open_tiff(&path, 0, 0)?.into_control()
                }
                Some(LocalDatasetKind::Xenium) => {
                    let options = crate::data::document::XeniumOpenOptions {
                        imagery: "auto".to_string(),
                        load_cells: true,
                        load_transcripts: true,
                    };
                    alternate_backend
                        .open_xenium(&path, &options)?
                        .0
                        .into_control()
                }
                None => {
                    let inspection = dataset_inspector.inspect(&path);
                    if inspection.kind
                        != Some(crate::data::document::DatasetInspectionKind::SpatialData)
                    {
                        anyhow::bail!(
                            "project ROI source {} is not a supported dataset",
                            path.display()
                        );
                    }
                    let image = inspection
                        .elements
                        .as_ref()
                        .into_iter()
                        .flatten()
                        .find(|element| element.kind == "image")
                        .map(|element| element.name.clone())
                        .ok_or_else(|| {
                            anyhow::anyhow!("SpatialData source has no image element")
                        })?;
                    let options = crate::data::document::SpatialDataOpenOptions {
                        image,
                        extra_images: Vec::new(),
                        labels: None,
                        shapes: Vec::new(),
                        points: None,
                        points_max: 200_000,
                    };
                    alternate_backend
                        .open_spatialdata(&path, &options)?
                        .0
                        .into_control()
                }
            }
        }
        DatasetSource::Http { base_url } => {
            let (opened, available, root) =
                complete_ome_zarr_resource(remote_backend.open_http(base_url)?, Vec::new());
            label_available = available;
            label_resource = root;
            opened.into_control()
        }
        DatasetSource::S3 { prefix, .. } => {
            let (generation, credentials) = s3_session
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("S3 session credentials are unavailable"))?;
            let (opened, available, root) = complete_ome_zarr_resource(
                remote_backend.open_s3(credentials, prefix)?,
                Vec::new(),
            );
            s3_session_generation = Some(*generation);
            label_available = available;
            label_resource = root;
            opened.into_control()
        }
    };

    if opened.descriptor.kind == crate::data::document::DocumentKind::OmeZarr
        && let Some(selected) =
            requested_label.or_else(|| saved_project_label_name(saved_view.as_ref()))
    {
        if !label_available.contains(&selected) {
            label_available.push(selected.clone());
        }
        let root_matches = label_resource
            .as_ref()
            .is_some_and(|resource| resource.dataset.label_name == selected);
        if !root_matches {
            label_resource =
                match LabelZarrDataset::try_open(Arc::clone(opened.resource.store()), &selected)? {
                    Some(dataset) => Some(ControlLabelResource {
                        dataset,
                        store: Arc::clone(opened.resource.store()),
                    }),
                    None => anyhow::bail!("saved project label '{selected}' was not found"),
                };
        }
    }

    let object_resource = match (cached_object, object_path) {
        (Some(resource), _) => Some(resource),
        (None, Some(path)) => {
            let loader = object_loader
                .ok_or_else(|| anyhow::anyhow!("object resource loader is unavailable"))?;
            let settings = ProjectObjectPreloadSettings::default();
            Some(Arc::new(loader.load_with_options(
                path,
                1.0,
                Some(settings.worker_options()),
            )?))
        }
        (None, None) => None,
    };

    Ok(ProjectRoiOpenWorkerResult {
        opened,
        roi,
        saved_view,
        label_available,
        label_resource,
        object_resource,
        s3_session_generation,
        reuse_current: false,
    })
}

fn saved_project_label_name(view: Option<&Value>) -> Option<String> {
    let view = view?;
    view.get("segmentation")
        .and_then(|segmentation| segmentation.get("label_name"))
        .and_then(Value::as_str)
        .or_else(|| {
            let workspace = view.get("workspace")?;
            let active = workspace.get("active_viewport_id")?.as_str()?;
            workspace
                .get("viewports")?
                .as_array()?
                .iter()
                .find(|viewport| viewport.get("id").and_then(Value::as_str) == Some(active))?
                .get("segmentation")?
                .get("label_name")?
                .as_str()
        })
        .map(str::to_string)
}

fn complete_ome_zarr_resource(
    opened: OpenedDocument<OmeZarrDocumentResource>,
    mut available: Vec<String>,
) -> (
    OpenedDocument<OmeZarrDocumentResource>,
    Vec<String>,
    Option<ControlLabelResource>,
) {
    let dataset = &opened.resource.dataset;
    let root_resource = dataset.is_root_label_mask().then(|| {
        let labels = LabelZarrDataset::from_root_dataset(dataset);
        if !available.contains(&labels.label_name) {
            available.push(labels.label_name.clone());
        }
        ControlLabelResource {
            dataset: labels,
            store: Arc::clone(&opened.resource.store),
        }
    });
    (opened, available, root_resource)
}
