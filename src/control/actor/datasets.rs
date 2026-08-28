use super::*;

pub(super) fn begin_ome_zarr_load(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    let Some(raw_path) = request.command.params().get("path").and_then(Value::as_str) else {
        diagnostics
            .rejected_requests
            .fetch_add(1, Ordering::Relaxed);
        diagnostics.record_reply_time(request.command.queue_age());
        let _ = request.reply.send(Err(ControlError::invalid_params(
            "datasets.open_ome_zarr",
            "path is required",
        )));
        return;
    };
    let identity = OmeZarrOpenIdentity::local_path(expand_path(raw_path));
    // The actor is the only producer for this mailbox. If it is not full now, a worker can only
    // make more space before the subsequent try_send; no competing producer can consume it.
    if load_job_tx.is_full() {
        diagnostics
            .rejected_requests
            .fetch_add(1, Ordering::Relaxed);
        diagnostics.record_reply_time(request.command.queue_age());
        let _ = request.reply.send(Err(ControlError::new(
            ControlErrorKind::NotReady,
            "Odon's dataset loader queue is full; retry later",
        )));
        return;
    }
    let generation = model.begin_dataset_open(identity.path.to_string_lossy());
    match load_job_tx.try_send(LoadJob::OmeZarr {
        generation,
        request,
        identity,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(crossbeam_channel::TrySendError::Full(LoadJob::OmeZarr { request, .. })) => {
            model.fail_dataset_open("dataset loader queue became full");
            diagnostics
                .rejected_requests
                .fetch_add(1, Ordering::Relaxed);
            diagnostics.record_reply_time(request.command.queue_age());
            let _ = request.reply.send(Err(ControlError::new(
                ControlErrorKind::NotReady,
                "Odon's dataset loader queue is full; retry later",
            )));
        }
        Err(crossbeam_channel::TrySendError::Disconnected(LoadJob::OmeZarr {
            request, ..
        })) => {
            model.fail_dataset_open("dataset loader workers are unavailable");
            diagnostics
                .rejected_requests
                .fetch_add(1, Ordering::Relaxed);
            diagnostics.record_reply_time(request.command.queue_age());
            let _ = request.reply.send(Err(ControlError::new(
                ControlErrorKind::NotReady,
                "Odon's dataset loader workers are unavailable",
            )));
        }
        Err(_) => unreachable!("submitted OME-Zarr job changed variant"),
    }
}

pub(super) fn begin_data_resource_load(
    model: &mut AppModel,
    request: OdonControlRequest,
    resource_registry: &ResourceRegistry,
    remote_session: &mut RemoteSessionState,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    let Some(resource_id) = request
        .command
        .params()
        .get("resource_id")
        .and_then(Value::as_str)
    else {
        reject_actor_request(
            request,
            diagnostics,
            ControlError::invalid_params("datasets.open_resource", "resource_id is required"),
        );
        return;
    };
    let resource =
        match resource_registry.get_resource_for_session(resource_id, &request.session_id) {
            Ok(resource) => resource,
            Err(error) => {
                reject_actor_request(request, diagnostics, error);
                return;
            }
        };
    let mut layers = request
        .command
        .params()
        .get("layers")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    assign_registered_object_layer_ids(&mut layers);
    if let Err(error) = resource_registry.validate_layers(&layers, &request.session_id) {
        reject_actor_request(
            request,
            diagnostics,
            error.with_data(json!({
                "resource_id": resource.resource_id,
                "phase": "validate_layers",
            })),
        );
        return;
    }
    let object_layers =
        match registered_object_layer_specs(resource_registry, &layers, &request.session_id) {
            Ok(layers) => layers,
            Err(error) => {
                reject_actor_request(
                    request,
                    diagnostics,
                    error.with_data(json!({
                        "resource_id": resource.resource_id,
                        "phase": "prepare_object_layers",
                    })),
                );
                return;
            }
        };
    if !matches!(resource.format.as_str(), "ome-zarr" | "zarr") {
        reject_actor_request(
            request,
            diagnostics,
            ControlError::invalid_params(
                "datasets.open_resource",
                "the primary resource must use the ome-zarr or zarr format",
            )
            .with_data(json!({
                "resource_id": resource.resource_id,
                "format": resource.format,
                "phase": "validate_resource",
            })),
        );
        return;
    }
    let url = match url::Url::parse(&resource.uri) {
        Ok(url) => url,
        Err(error) => {
            reject_actor_request(
                request,
                diagnostics,
                ControlError::invalid_params(
                    "datasets.open_resource",
                    format!("resource URI is invalid: {error}"),
                )
                .with_data(json!({
                    "resource_id": resource.resource_id,
                    "source_uri": resource.uri,
                    "phase": "resolve_source",
                })),
            );
            return;
        }
    };
    if matches!(url.scheme(), "http" | "https" | "s3") {
        if load_job_tx.is_full() {
            reject_worker_submission(request, diagnostics);
            return;
        }
        let registered = RegisteredResourceOpenIdentity {
            resource_id: resource.resource_id.clone(),
            source_uri: resource.uri.clone(),
            group_path: resource.group_path.clone(),
            layers,
            object_layers,
        };
        let source_display = resource
            .group_path
            .as_deref()
            .map(|group_path| format!("{}#{group_path}", resource.uri))
            .unwrap_or_else(|| resource.uri.clone());
        let generation = model.begin_dataset_open(source_display);
        match url.scheme() {
            "http" | "https" => {
                let mut element_url = url;
                if let Some(group_path) = resource.group_path.as_deref() {
                    let Ok(mut segments) = element_url.path_segments_mut() else {
                        model.fail_dataset_open_for_generation(
                            generation,
                            "registered HTTP resource URL cannot address a nested group",
                        );
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::invalid_params(
                                "datasets.open_resource",
                                "HTTP resource URL cannot address a nested group",
                            ),
                        );
                        return;
                    };
                    segments.pop_if_empty();
                    for component in group_path.split('/') {
                        segments.push(component);
                    }
                }
                submit_remote_open(
                    model,
                    request,
                    load_job_tx,
                    diagnostics,
                    generation,
                    None,
                    RemoteOpenSpec::Http {
                        url: element_url.to_string().trim_end_matches('/').to_string(),
                    },
                    Some(registered),
                );
            }
            "s3" => {
                let (session_generation, credentials) = match remote_session.credentials() {
                    Ok(session) => session,
                    Err(error) => {
                        model.fail_dataset_open_for_generation(
                            generation,
                            "S3 session credentials are unavailable",
                        );
                        reject_actor_request(request, diagnostics, error);
                        return;
                    }
                };
                let bucket = url.host_str().unwrap_or_default();
                if bucket != credentials.bucket {
                    model.fail_dataset_open_for_generation(
                        generation,
                        "registered S3 resource does not match the configured session bucket",
                    );
                    reject_actor_request(
                        request,
                        diagnostics,
                        ControlError::new(
                            ControlErrorKind::PermissionDenied,
                            "registered S3 resource bucket does not match the configured session",
                        )
                        .with_data(json!({
                            "resource_id": resource.resource_id,
                            "bucket": bucket,
                            "phase": "resolve_credentials",
                        })),
                    );
                    return;
                }
                let mut prefix = normalize_s3_prefix(url.path());
                if let Some(group_path) = resource.group_path.as_deref() {
                    if !prefix.is_empty() {
                        prefix.push('/');
                    }
                    prefix.push_str(group_path);
                }
                remote_session.mark_s3_open_pending(generation);
                submit_remote_open(
                    model,
                    request,
                    load_job_tx,
                    diagnostics,
                    generation,
                    Some(session_generation),
                    RemoteOpenSpec::S3 {
                        credentials,
                        prefix,
                    },
                    Some(registered),
                );
            }
            _ => unreachable!("validated remote resource scheme"),
        }
        return;
    }
    let Ok(root_path) = url.to_file_path() else {
        reject_actor_request(
            request,
            diagnostics,
            ControlError::invalid_params(
                "datasets.open_resource",
                "resource URI must use file, http, https, or s3",
            )
            .with_data(json!({
                "resource_id": resource.resource_id,
                "source_uri": resource.uri,
                "phase": "resolve_source",
            })),
        );
        return;
    };
    let path = resource
        .group_path
        .as_deref()
        .map(|group_path| root_path.join(group_path))
        .unwrap_or(root_path);
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return;
    }
    let identity = OmeZarrOpenIdentity {
        path,
        resource_id: Some(resource.resource_id.clone()),
        source_uri: Some(resource.uri.clone()),
        group_path: resource.group_path.clone(),
        layers,
        object_layers,
    };
    let source_display = match resource.group_path.as_deref() {
        Some(group_path) => format!("{}#{group_path}", resource.uri),
        None => resource.uri,
    };
    let generation = model.begin_dataset_open(source_display);
    match load_job_tx.try_send(LoadJob::OmeZarr {
        generation,
        request,
        identity,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let LoadJob::OmeZarr { request, .. } = error.into_inner() else {
                unreachable!("data-resource submission returns its own job")
            };
            model.fail_dataset_open_for_generation(
                generation,
                "data resource loader workers are unavailable",
            );
            reject_worker_submission(request, diagnostics);
        }
    }
}

fn assign_registered_object_layer_ids(layers: &mut [Value]) {
    static NEXT_ID: AtomicU64 = AtomicU64::new(1_000_000_000);
    for layer in layers {
        let is_object_layer = layer
            .get("kind")
            .and_then(Value::as_str)
            .is_some_and(|kind| matches!(kind, "objects" | "points" | "shapes"));
        if !is_object_layer || layer.get("layer_id").is_some() {
            continue;
        }
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed).max(1);
        if let Some(object) = layer.as_object_mut() {
            object.insert(
                "layer_id".to_string(),
                Value::String(format!("spatial_shape:{id}")),
            );
        }
    }
}

fn registered_object_layer_specs(
    registry: &ResourceRegistry,
    layers: &[Value],
    session_id: &str,
) -> Result<Vec<RegisteredObjectLayerSpec>, ControlError> {
    let mut specs = Vec::new();
    for layer in layers {
        let Some(kind) = layer.get("kind").and_then(Value::as_str) else {
            continue;
        };
        if !matches!(kind, "objects" | "points" | "shapes") {
            continue;
        }
        let Some(layer_id) = layer.get("layer_id").and_then(Value::as_str) else {
            continue;
        };
        let Some(numeric_id) = layer_id
            .strip_prefix("spatial_shape:")
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|id| *id > 0)
        else {
            continue;
        };
        let resource_id = layer
            .get("data_resource_id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ControlError::invalid_params(
                    "datasets.open_resource",
                    "object layer data_resource_id is required",
                )
            })?;
        let resource = registry.get_resource_for_session(resource_id, session_id)?;
        if !matches!(resource.format.as_str(), "parquet" | "geoparquet") {
            continue;
        }
        let url = url::Url::parse(&resource.uri).map_err(|error| {
            ControlError::invalid_params(
                "datasets.open_resource",
                format!("object resource URI is invalid: {error}"),
            )
        })?;
        let path = url.to_file_path().map_err(|_| {
            ControlError::invalid_params(
                "datasets.open_resource",
                "selectable object resources currently require a local file URI",
            )
        })?;
        let coordinate = &resource.coordinate_space;
        let axis_value = |values: &[f64], axis: &str, default: f64| {
            coordinate
                .axes
                .iter()
                .position(|name| name.eq_ignore_ascii_case(axis))
                .and_then(|index| values.get(index))
                .copied()
                .unwrap_or(default)
        };
        let matrix = coordinate.affine.as_ref().map_or_else(
            || {
                vec![
                    vec![
                        axis_value(&coordinate.scale, "x", 1.0),
                        0.0,
                        axis_value(&coordinate.translation, "x", 0.0),
                    ],
                    vec![
                        0.0,
                        axis_value(&coordinate.scale, "y", 1.0),
                        axis_value(&coordinate.translation, "y", 0.0),
                    ],
                    vec![0.0, 0.0, 1.0],
                ]
            },
            |affine| affine.matrix.clone(),
        );
        specs.push(RegisteredObjectLayerSpec {
            layer_id: format!("spatial_shape:{numeric_id}"),
            name: layer
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("Objects")
                .to_string(),
            path,
            loader_options: json!({
                "format": "geoparquet",
                "display_mode": if kind == "points" { "points" } else { "polygons" },
                "source": "geometry",
                "geometry_column": resource.metadata
                    .get("geometry_column")
                    .and_then(Value::as_str)
                    .unwrap_or("geometry"),
                "transform": {"matrix": matrix},
            }),
        });
    }
    Ok(specs)
}

pub(super) fn begin_tiff_load(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    let Some(raw_path) = request.command.params().get("path").and_then(Value::as_str) else {
        reject_actor_request(
            request,
            diagnostics,
            ControlError::invalid_params("datasets.open_tiff", "path is required"),
        );
        return;
    };
    let requested_path = expand_path(raw_path);
    let Some(path) = normalize_local_dataset_path(&requested_path) else {
        reject_actor_request(
            request,
            diagnostics,
            ControlError::invalid_params(
                "datasets.open_tiff",
                "path is not a local TIFF / OME-TIFF file",
            ),
        );
        return;
    };
    if classify_local_dataset_path(&path) != Some(LocalDatasetKind::Tiff) {
        reject_actor_request(
            request,
            diagnostics,
            ControlError::invalid_params(
                "datasets.open_tiff",
                "path is not a TIFF / OME-TIFF dataset",
            ),
        );
        return;
    }
    let z = request
        .command
        .params()
        .get("z")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    let t = request
        .command
        .params()
        .get("t")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return;
    }
    let generation = model.begin_dataset_open(path.to_string_lossy());
    match load_job_tx.try_send(LoadJob::Tiff {
        generation,
        request,
        path,
        z,
        t,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let LoadJob::Tiff { request, .. } = error.into_inner() else {
                unreachable!("TIFF submission returns its own job")
            };
            model.fail_dataset_open_for_generation(
                generation,
                "TIFF dataset loader workers are unavailable",
            );
            reject_worker_submission(request, diagnostics);
        }
    }
}

pub(super) fn begin_spatialdata_load(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    let Some(raw_path) = request.command.params().get("path").and_then(Value::as_str) else {
        reject_actor_request(
            request,
            diagnostics,
            ControlError::invalid_params("datasets.open_spatialdata", "path is required"),
        );
        return;
    };
    let path = expand_path(raw_path);
    let mut options = request.command.params().clone();
    options.as_object_mut().map(|object| object.remove("path"));
    let options =
        match serde_json::from_value::<crate::data::document::SpatialDataOpenOptions>(options) {
            Ok(options) => options,
            Err(error) => {
                reject_actor_request(
                    request,
                    diagnostics,
                    ControlError::invalid_params(
                        "datasets.open_spatialdata",
                        format!("invalid SpatialData options: {error}"),
                    ),
                );
                return;
            }
        };
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return;
    }
    let generation = model.begin_dataset_open(path.to_string_lossy());
    match load_job_tx.try_send(LoadJob::SpatialData {
        generation,
        request,
        path,
        options,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let LoadJob::SpatialData { request, .. } = error.into_inner() else {
                unreachable!("SpatialData submission returns its own job")
            };
            model.fail_dataset_open_for_generation(
                generation,
                "SpatialData loader workers are unavailable",
            );
            reject_worker_submission(request, diagnostics);
        }
    }
}

pub(super) fn begin_xenium_load(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    let Some(raw_path) = request.command.params().get("path").and_then(Value::as_str) else {
        reject_actor_request(
            request,
            diagnostics,
            ControlError::invalid_params("datasets.open_xenium", "path is required"),
        );
        return;
    };
    let requested_path = expand_path(raw_path);
    let path = normalize_local_dataset_path(&requested_path).unwrap_or(requested_path);
    let mut options = request.command.params().clone();
    options.as_object_mut().map(|object| object.remove("path"));
    let options = match serde_json::from_value::<crate::data::document::XeniumOpenOptions>(options)
    {
        Ok(options) => options,
        Err(error) => {
            reject_actor_request(
                request,
                diagnostics,
                ControlError::invalid_params(
                    "datasets.open_xenium",
                    format!("invalid Xenium options: {error}"),
                ),
            );
            return;
        }
    };
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return;
    }
    let generation = model.begin_dataset_open(path.to_string_lossy());
    match load_job_tx.try_send(LoadJob::Xenium {
        generation,
        request,
        path,
        options,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let LoadJob::Xenium { request, .. } = error.into_inner() else {
                unreachable!("Xenium submission returns its own job")
            };
            model.fail_dataset_open_for_generation(
                generation,
                "Xenium loader workers are unavailable",
            );
            reject_worker_submission(request, diagnostics);
        }
    }
}

pub(super) fn begin_dataset_inspection(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    let Some(raw_path) = request.command.params().get("path").and_then(Value::as_str) else {
        finish_request(request, json!({"error":"path is required"}), diagnostics);
        return;
    };
    let path = expand_path(raw_path);
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return;
    }
    let (operation_generation, operation_scope) = model.begin_dataset_inspection(&path);
    match load_job_tx.try_send(LoadJob::DatasetInspect {
        operation_generation,
        operation_scope: operation_scope.clone(),
        request,
        path,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let LoadJob::DatasetInspect { request, .. } = error.into_inner() else {
                unreachable!("dataset inspection submission returns its own job")
            };
            model.cancel_dataset_inspection(
                &operation_scope,
                operation_generation,
                "Dataset inspection worker queue is unavailable",
            );
            reject_worker_submission(request, diagnostics);
        }
    }
}
