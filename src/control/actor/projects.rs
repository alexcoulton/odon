use super::*;

pub(super) fn begin_project_open(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    let Some(raw_path) = request.command.params().get("path").and_then(Value::as_str) else {
        diagnostics
            .rejected_requests
            .fetch_add(1, Ordering::Relaxed);
        let _ = request.reply.send(Err(ControlError::invalid_params(
            "project.open",
            "path is required",
        )));
        return;
    };
    if load_job_tx.is_full() {
        diagnostics
            .rejected_requests
            .fetch_add(1, Ordering::Relaxed);
        let _ = request.reply.send(Err(ControlError::new(
            ControlErrorKind::NotReady,
            "Odon's resource worker queue is full; retry later",
        )));
        return;
    }
    let path = expand_path(raw_path);
    let generation = model.begin_project_operation(format!("Opening {}", path.display()));
    match load_job_tx.try_send(LoadJob::ProjectOpen {
        generation,
        request,
        path,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let job = error.into_inner();
            let LoadJob::ProjectOpen { request, .. } = job else {
                unreachable!("project-open submission returns its own job")
            };
            model.fail_project_operation(generation, "project worker queue is unavailable");
            diagnostics
                .rejected_requests
                .fetch_add(1, Ordering::Relaxed);
            let _ = request.reply.send(Err(ControlError::new(
                ControlErrorKind::NotReady,
                "Odon's resource worker queue is unavailable; retry later",
            )));
        }
    }
}

pub(super) fn begin_project_save(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    let path = request
        .command
        .params()
        .get("path")
        .and_then(Value::as_str)
        .map(expand_path)
        .or_else(|| model.project_snapshot().saved_path);
    let Some(path) = path else {
        diagnostics
            .rejected_requests
            .fetch_add(1, Ordering::Relaxed);
        let _ = request.reply.send(Err(ControlError::new(
            ControlErrorKind::InvalidParams,
            "project.save requires an existing project path; use project.save_as first",
        )));
        return;
    };
    let (payload, saved_config_generation) = match model.prepare_lifecycle_project_save() {
        Ok(payload) => payload,
        Err(error) => {
            diagnostics
                .rejected_requests
                .fetch_add(1, Ordering::Relaxed);
            let _ = request.reply.send(Err(error));
            return;
        }
    };
    if load_job_tx.is_full() {
        diagnostics
            .rejected_requests
            .fetch_add(1, Ordering::Relaxed);
        let _ = request.reply.send(Err(ControlError::new(
            ControlErrorKind::NotReady,
            "Odon's resource worker queue is full; retry later",
        )));
        return;
    }
    let generation = model.begin_project_operation(format!("Saving {}", path.display()));
    match load_job_tx.try_send(LoadJob::ProjectSave {
        generation,
        request,
        path,
        payload,
        saved_config_generation,
        platform_effect: None,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let job = error.into_inner();
            let LoadJob::ProjectSave { request, .. } = job else {
                unreachable!("project-save submission returns its own job")
            };
            model.fail_project_operation(generation, "project worker queue is unavailable");
            diagnostics
                .rejected_requests
                .fetch_add(1, Ordering::Relaxed);
            let _ = request.reply.send(Err(ControlError::new(
                ControlErrorKind::NotReady,
                "Odon's resource worker queue is unavailable; retry later",
            )));
        }
    }
}

pub(super) fn begin_samplesheet_inspect(
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    let params = request.command.params();
    let path = params
        .get("path")
        .and_then(Value::as_str)
        .map(expand_path)
        .expect("typed samplesheet inspection requires a path");
    let offset = params
        .get("offset")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(0);
    let limit = params
        .get("limit")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(200);
    match load_job_tx.try_send(LoadJob::SamplesheetInspect {
        request,
        path,
        offset,
        limit,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let LoadJob::SamplesheetInspect { request, .. } = error.into_inner() else {
                unreachable!("samplesheet-inspect submission returns its own job")
            };
            reject_worker_submission(request, diagnostics);
        }
    }
}

pub(super) fn begin_samplesheet_import(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    let path = request
        .command
        .params()
        .get("path")
        .and_then(Value::as_str)
        .map(expand_path)
        .expect("typed samplesheet import requires a path");
    let default_dataset = model
        .project_snapshot()
        .default_dataset
        .unwrap_or_else(|| "default".to_string());
    let generation = model.begin_project_operation(format!("Importing {}", path.display()));
    match load_job_tx.try_send(LoadJob::SamplesheetImport {
        generation,
        request,
        path,
        default_dataset,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let LoadJob::SamplesheetImport { request, .. } = error.into_inner() else {
                unreachable!("samplesheet-import submission returns its own job")
            };
            model.fail_project_operation(generation, "project worker queue is unavailable");
            reject_worker_submission(request, diagnostics);
        }
    }
}

pub(super) fn begin_samplesheet_export(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    let path = request
        .command
        .params()
        .get("path")
        .and_then(Value::as_str)
        .map(expand_path)
        .expect("typed samplesheet export requires a path");
    let overwrite = request
        .command
        .params()
        .get("overwrite")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let rois = model.project_snapshot().rois;
    let generation = model.begin_project_operation(format!("Exporting {}", path.display()));
    match load_job_tx.try_send(LoadJob::SamplesheetExport {
        generation,
        request,
        path,
        rois,
        overwrite,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let LoadJob::SamplesheetExport { request, .. } = error.into_inner() else {
                unreachable!("samplesheet-export submission returns its own job")
            };
            model.fail_project_operation(generation, "project worker queue is unavailable");
            reject_worker_submission(request, diagnostics);
        }
    }
}

pub(super) fn begin_project_discovery(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    let root = request
        .command
        .params()
        .get("path")
        .and_then(Value::as_str)
        .map(expand_path)
        .expect("typed project discovery requires a path");
    let generation = model.begin_project_operation(format!("Discovering {}", root.display()));
    match load_job_tx.try_send(LoadJob::ProjectDiscovery {
        generation,
        request,
        root,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let LoadJob::ProjectDiscovery { request, .. } = error.into_inner() else {
                unreachable!("project-discovery submission returns its own job")
            };
            model.fail_project_operation(generation, "project worker queue is unavailable");
            reject_worker_submission(request, diagnostics);
        }
    }
}
