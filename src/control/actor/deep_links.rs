use super::*;

pub(super) fn begin_deep_link_application(
    model: &mut AppModel,
    remote_session: &RemoteSessionState,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    render_document: &Option<Arc<RenderDocument>>,
    diagnostics: &ActorDiagnostics,
) {
    let mut deep_link = match AppModel::deep_link_request_from_params(request.command.params()) {
        Ok(request) => request,
        Err(error) => {
            reject_actor_request(request, diagnostics, error);
            return;
        }
    };
    if let Some(example) = deep_link.example.clone() {
        apply_example_defaults(&mut deep_link, &example);
    }
    if let Some(path) = deep_link.project_path.as_deref() {
        deep_link.project_path = Some(expand_path(&path.to_string_lossy()));
    }
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return;
    }

    let current_project = model.project_snapshot();
    let cached_object = deep_link
        .project_path
        .as_ref()
        .is_none_or(|path| current_project.saved_path.as_ref() == Some(path))
        .then(|| {
            resolve_roi_target(
                &current_project.rois,
                deep_link.roi.as_deref(),
                deep_link.sample.as_deref(),
            )
            .ok()
        })
        .flatten()
        .and_then(|roi| project_roi_segmentation_path(&current_project, &roi))
        .and_then(|path| {
            model
                .cached_project_object_resource(&path)
                .map(|resource| (path, resource))
        });
    let s3_session = remote_session.credentials().ok();
    let url = deep_link.to_url();
    let (operation_generation, guard) =
        model.begin_deep_link_apply(format!("Applying deep link {url}"));
    let spec = DeepLinkApplySpec {
        deep_link,
        current_project,
        cached_object,
        s3_session,
        current_document: render_document
            .as_ref()
            .map(|document| document.opened.clone()),
        current_resources: model.deep_link_current_resources(),
    };
    match load_job_tx.try_send(LoadJob::DeepLinkApply {
        operation_generation,
        guard,
        request,
        spec,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let LoadJob::DeepLinkApply {
                request,
                operation_generation,
                guard,
                ..
            } = error.into_inner()
            else {
                unreachable!("deep-link application submission returns its own job")
            };
            model.fail_deep_link_apply(
                operation_generation,
                guard,
                "Deep-link worker queue is unavailable",
            );
            reject_worker_submission(request, diagnostics);
        }
    }
}

pub(super) fn begin_deep_link_resolution(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    let mut deep_link = match AppModel::deep_link_request_from_params(request.command.params()) {
        Ok(request) => request,
        Err(error) => {
            finish_request(request, json!({"error":error.message}), diagnostics);
            return;
        }
    };
    if let Some(example) = deep_link.example.clone() {
        apply_example_defaults(&mut deep_link, &example);
    }
    if let Some(path) = deep_link.project_path.as_deref() {
        deep_link.project_path = Some(expand_path(&path.to_string_lossy()));
    }
    let current_project = model.project_snapshot();
    let needs_example_lookup = deep_link.example.is_some() && deep_link.project_path.is_none();
    let use_current = deep_link
        .project_path
        .as_ref()
        .is_none_or(|path| current_project.saved_path.as_ref() == Some(path));
    if !needs_example_lookup && use_current {
        let resolution = resolve_roi_target(
            &current_project.rois,
            deep_link.roi.as_deref(),
            deep_link.sample.as_deref(),
        )
        .map(|roi| DeepLinkResolution {
            project_source: "current".to_string(),
            project_path: deep_link.project_path.clone(),
            roi,
        });
        finish_request(
            request,
            deep_link_resolution_response(deep_link, resolution),
            diagnostics,
        );
        return;
    }
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return;
    }
    let operation_scope = deep_link.to_url();
    let operation_generation = model.begin_deep_link_resolution(operation_scope.clone());
    match load_job_tx.try_send(LoadJob::DeepLinkResolve {
        operation_generation,
        operation_scope: operation_scope.clone(),
        request,
        deep_link,
        current_project,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let LoadJob::DeepLinkResolve { request, .. } = error.into_inner() else {
                unreachable!("deep-link resolution submission returns its own job")
            };
            model.cancel_deep_link_resolution(
                &operation_scope,
                operation_generation,
                "Deep-link resolver worker queue is unavailable",
            );
            reject_worker_submission(request, diagnostics);
        }
    }
}

pub(super) fn resolve_deep_link_on_worker(
    mut deep_link: DeepLinkRequest,
    current_project: ProjectModelSnapshot,
) -> DeepLinkResolveWorkerResult {
    if let Some(example) = deep_link.example.clone() {
        apply_example_defaults(&mut deep_link, &example);
        if deep_link.project_path.is_none() {
            deep_link.project_path = resolve_example_project_path(&example);
        }
    }
    if let Some(path) = deep_link.project_path.as_deref() {
        deep_link.project_path = Some(expand_path(&path.to_string_lossy()));
    }
    let use_current = deep_link
        .project_path
        .as_ref()
        .is_none_or(|path| current_project.saved_path.as_ref() == Some(path));
    let resolution = if use_current {
        resolve_roi_target(
            &current_project.rois,
            deep_link.roi.as_deref(),
            deep_link.sample.as_deref(),
        )
        .map(|roi| DeepLinkResolution {
            project_source: "current".to_string(),
            project_path: deep_link.project_path.clone(),
            roi,
        })
    } else {
        let path = deep_link
            .project_path
            .as_ref()
            .expect("non-current deep link has a project path");
        if !path.exists() {
            Err(format!(
                "Deep-link project does not exist: {}",
                path.to_string_lossy()
            ))
        } else {
            read_project_file(path)
                .map_err(|error| format!("Deep-link project could not be loaded: {error}"))
                .and_then(|(config, state)| {
                    crate::model::normalized_loaded_project_snapshot(path.clone(), config, state)
                        .map_err(|error| {
                            format!("Deep-link project could not be loaded: {}", error.message)
                        })
                })
                .and_then(|project| {
                    resolve_roi_target(
                        &project.rois,
                        deep_link.roi.as_deref(),
                        deep_link.sample.as_deref(),
                    )
                })
                .map(|roi| DeepLinkResolution {
                    project_source: "project_file".to_string(),
                    project_path: deep_link.project_path.clone(),
                    roi,
                })
        }
    };
    DeepLinkResolveWorkerResult {
        request: deep_link,
        resolution,
    }
}

pub(super) fn apply_deep_link_on_worker(
    spec: DeepLinkApplySpec,
    object_loader: Option<&dyn ObjectResourceLoader>,
    dataset_inspector: &dyn DatasetInspector,
    remote_backend: &dyn crate::data::remote_store::RemoteDatasetBackend,
    alternate_backend: &dyn AlternateDatasetBackend,
) -> anyhow::Result<DeepLinkApplyWorkerResult> {
    let DeepLinkApplySpec {
        mut deep_link,
        current_project,
        cached_object,
        s3_session,
        current_document,
        current_resources,
    } = spec;
    if let Some(example) = deep_link.example.clone() {
        apply_example_defaults(&mut deep_link, &example);
        if deep_link.project_path.is_none() {
            deep_link.project_path = resolve_example_project_path(&example);
        }
    }
    if let Some(path) = deep_link.project_path.as_deref() {
        deep_link.project_path = Some(expand_path(&path.to_string_lossy()));
    }

    let use_current = deep_link
        .project_path
        .as_ref()
        .is_none_or(|path| current_project.saved_path.as_ref() == Some(path));
    let (project, project_source) = if use_current {
        (current_project, "current".to_string())
    } else {
        let path = deep_link
            .project_path
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Deep link does not identify a project"))?;
        if !path.exists() {
            anyhow::bail!("Deep-link project does not exist: {}", path.display());
        }
        let (config, state) = read_project_file(path)
            .map_err(|error| anyhow::anyhow!("Deep-link project could not be loaded: {error}"))?;
        let project = crate::model::normalized_loaded_project_snapshot(path.clone(), config, state)
            .map_err(|error| {
                anyhow::anyhow!("Deep-link project could not be loaded: {}", error.message)
            })?;
        (project, "project_file".to_string())
    };

    let roi = resolve_roi_target(
        &project.rois,
        deep_link.roi.as_deref(),
        deep_link.sample.as_deref(),
    )
    .map_err(anyhow::Error::msg)?;
    let source = roi
        .dataset_source()
        .ok_or_else(|| anyhow::anyhow!("project ROI has no dataset source configured"))?;
    let source_key = source.source_key();
    let saved_view = project
        .state
        .get("roi_views")
        .and_then(|views| views.get(&source_key))
        .cloned();
    let object_path = project_roi_segmentation_path(&project, &roi);
    let cached_object = cached_object.and_then(|(cached_path, resource)| {
        object_path
            .as_ref()
            .is_some_and(|path| path == &cached_path)
            .then_some(resource)
    });
    let s3_session = match &source {
        DatasetSource::S3 {
            endpoint,
            region,
            bucket,
            ..
        } => {
            let (generation, credentials) = s3_session
                .ok_or_else(|| anyhow::anyhow!("S3 session credentials are not configured"))?;
            Some((
                generation,
                crate::data::remote_store::S3SessionCredentials::normalized(
                    endpoint,
                    region,
                    bucket,
                    &credentials.access_key,
                    &credentials.secret_key,
                ),
            ))
        }
        _ => None,
    };
    let requested_label = requested_bundled_label(&deep_link);
    let reuse_current = use_current
        && current_document
            .as_ref()
            .is_some_and(|document| document.descriptor.source.source_key() == source_key)
        && current_resources
            .as_ref()
            .is_some_and(|resources| resources.source_key == source_key);
    let opened = if reuse_current {
        reuse_current_project_roi_on_worker(
            roi,
            current_document.expect("reuse requires a current document"),
            current_resources.expect("reuse requires current resources"),
            object_path,
            cached_object,
            requested_label,
            object_loader,
        )?
    } else {
        open_project_roi_on_worker(
            ProjectRoiOpenSpec {
                roi,
                source,
                saved_view,
                object_path,
                cached_object,
                s3_session,
                requested_label,
            },
            object_loader,
            dataset_inspector,
            remote_backend,
            alternate_backend,
        )?
    };
    let object_filter = match (
        object_filter_model(&deep_link),
        opened.object_resource.as_ref(),
    ) {
        (Some(model), Some(resource)) => Some(
            object_loader
                .ok_or_else(|| anyhow::anyhow!("object filter evaluator is unavailable"))?
                .evaluate_filter(Arc::clone(resource), model)?,
        ),
        _ => None,
    };
    Ok(DeepLinkApplyWorkerResult {
        deep_link,
        project,
        project_source,
        opened,
        object_filter,
    })
}

fn reuse_current_project_roi_on_worker(
    roi: ProjectRoi,
    opened: ControlOpenedDocument,
    current: DeepLinkCurrentResources,
    object_path: Option<PathBuf>,
    cached_object: Option<Arc<ControlObjectResource>>,
    requested_label: Option<String>,
    object_loader: Option<&dyn ObjectResourceLoader>,
) -> anyhow::Result<ProjectRoiOpenWorkerResult> {
    let mut label_available = current.label_available;
    let label_resource = if let Some(label) = requested_label {
        if !label_available.contains(&label) {
            label_available.push(label.clone());
        }
        if current.label_loaded.as_deref() == Some(label.as_str()) {
            current.label.map(|resource| resource.as_ref().clone())
        } else if opened.descriptor.kind == crate::data::document::DocumentKind::OmeZarr {
            LabelZarrDataset::try_open(Arc::clone(opened.resource.store()), &label)?.map(
                |dataset| ControlLabelResource {
                    dataset,
                    store: Arc::clone(opened.resource.store()),
                },
            )
        } else {
            anyhow::bail!("labels/{label} cannot be loaded from this document kind")
        }
    } else {
        None
    };
    let object_resource = match (current.object, cached_object, object_path) {
        (Some(resource), _, _) => Some(resource),
        (None, Some(resource), _) => Some(resource),
        (None, None, Some(path)) => {
            let loader = object_loader
                .ok_or_else(|| anyhow::anyhow!("object resource loader is unavailable"))?;
            Some(Arc::new(loader.load_with_options(
                path,
                1.0,
                Some(ProjectObjectPreloadSettings::default().worker_options()),
            )?))
        }
        (None, None, None) => None,
    };
    Ok(ProjectRoiOpenWorkerResult {
        opened,
        roi,
        saved_view: None,
        label_available,
        label_resource,
        object_resource,
        s3_session_generation: None,
        reuse_current: true,
    })
}

pub(super) fn deep_link_resolution_response(
    request: DeepLinkRequest,
    resolution: Result<DeepLinkResolution, String>,
) -> Value {
    match resolution {
        Ok(resolution) => json!({
            "resolved":true,
            "url":request.to_url(),
            "request":request,
            "resolution":resolution,
        }),
        Err(error) => json!({"resolved":false,"error":error}),
    }
}
