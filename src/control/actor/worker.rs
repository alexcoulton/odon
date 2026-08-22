use super::*;

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
