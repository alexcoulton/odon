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
    let path = expand_path(raw_path);
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
    let generation = model.begin_dataset_open(path.to_string_lossy());
    match load_job_tx.try_send(LoadJob::OmeZarr {
        generation,
        request,
        path,
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
