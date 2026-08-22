use super::*;

pub(super) fn begin_analysis_compute(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    kind: AnalysisComputeKind,
    diagnostics: &ActorDiagnostics,
) -> bool {
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return false;
    }
    let property = request
        .command
        .params()
        .get("property")
        .and_then(Value::as_str)
        .unwrap_or("all");
    let scope = match kind {
        AnalysisComputeKind::Histogram => format!("analysis_histogram:{property}"),
        AnalysisComputeKind::ThresholdSuggestions => format!("analysis_suggestions:{property}"),
        AnalysisComputeKind::Warmup => "analysis_warmup".to_string(),
    };
    let spec = match kind {
        AnalysisComputeKind::Warmup => model.begin_analysis_warmup(request.command.params()),
        _ => model.prepare_analysis_resource_operation(request.command.params(), &scope),
    };
    let spec = match spec {
        Ok(spec) => spec,
        Err(error) => {
            reject_actor_request(request, diagnostics, error);
            return false;
        }
    };
    let params = request.command.params().clone();
    match load_job_tx.try_send(LoadJob::AnalysisCompute {
        request,
        spec,
        kind,
        params,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
            true
        }
        Err(error) => {
            let LoadJob::AnalysisCompute { request, spec, .. } = error.into_inner() else {
                unreachable!("analysis submission returns its own job")
            };
            model.fail_analysis_operation(&spec, "Analysis worker queue is unavailable");
            reject_worker_submission(request, diagnostics);
            false
        }
    }
}

pub(super) fn begin_analysis_preset_import(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) -> bool {
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return false;
    }
    let path = match request
        .command
        .params()
        .get("path")
        .and_then(Value::as_str)
        .map(expand_path)
    {
        Some(path) => path,
        None => {
            reject_actor_request(
                request,
                diagnostics,
                ControlError::new(ControlErrorKind::InvalidParams, "path is required"),
            );
            return false;
        }
    };
    let spec = match model
        .prepare_analysis_resource_operation(request.command.params(), "analysis_preset_import")
    {
        Ok(spec) => spec,
        Err(error) => {
            reject_actor_request(request, diagnostics, error);
            return false;
        }
    };
    match load_job_tx.try_send(LoadJob::AnalysisPresetImport {
        request,
        spec,
        path,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
            true
        }
        Err(error) => {
            let LoadJob::AnalysisPresetImport { request, spec, .. } = error.into_inner() else {
                unreachable!("analysis preset import returns its own job")
            };
            model.fail_analysis_operation(&spec, "Analysis worker queue is unavailable");
            reject_worker_submission(request, diagnostics);
            false
        }
    }
}

pub(super) fn begin_analysis_preset_export(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) -> bool {
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return false;
    }
    let path = match request
        .command
        .params()
        .get("path")
        .and_then(Value::as_str)
        .map(expand_path)
    {
        Some(path) => path,
        None => {
            reject_actor_request(
                request,
                diagnostics,
                ControlError::new(ControlErrorKind::InvalidParams, "path is required"),
            );
            return false;
        }
    };
    let overwrite = request
        .command
        .params()
        .get("overwrite")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let spec = match model
        .prepare_analysis_resource_operation(request.command.params(), "analysis_preset_export")
    {
        Ok(spec) => spec,
        Err(error) => {
            reject_actor_request(request, diagnostics, error);
            return false;
        }
    };
    let state = model.analysis_state().clone();
    match load_job_tx.try_send(LoadJob::AnalysisPresetExport {
        request,
        spec,
        path,
        overwrite,
        state,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
            true
        }
        Err(error) => {
            let LoadJob::AnalysisPresetExport { request, spec, .. } = error.into_inner() else {
                unreachable!("analysis preset export returns its own job")
            };
            model.fail_analysis_operation(&spec, "Analysis worker queue is unavailable");
            reject_worker_submission(request, diagnostics);
            false
        }
    }
}
