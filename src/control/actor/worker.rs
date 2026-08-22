use super::*;

pub(super) fn spawn_resource_workers(
    load_job_rx: Receiver<LoadJob>,
    load_tx: Sender<LoadCompletion>,
    object_loader: Option<Arc<dyn ObjectResourceLoader>>,
    dataset_inspector: Arc<dyn DatasetInspector>,
    remote_backend: Arc<dyn RemoteDatasetBackend>,
) -> anyhow::Result<()> {
    for index in 0..LOAD_WORKERS {
        let jobs = load_job_rx.clone();
        let completions = load_tx.clone();
        let object_loader = object_loader.clone();
        let dataset_inspector = Arc::clone(&dataset_inspector);
        let remote_backend = Arc::clone(&remote_backend);
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
