use super::*;

pub(super) fn begin_threshold_configure(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) -> bool {
    if model.threshold_preview_resource().is_some() && load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return false;
    }
    match model.configure_threshold_preview(request.command.params()) {
        Ok(None) => {
            let response = model
                .threshold_preview_snapshot()
                .expect("validated threshold configuration has a dataset");
            finish_request(request, response, diagnostics);
            true
        }
        Ok(Some(spec)) => match load_job_tx.try_send(LoadJob::ThresholdRecompute { request, spec })
        {
            Ok(()) => {
                diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
                true
            }
            Err(error) => {
                let LoadJob::ThresholdRecompute { request, spec } = error.into_inner() else {
                    unreachable!("threshold recompute submission returns its own job")
                };
                model.fail_threshold_operation(
                    spec.document_generation,
                    spec.operation_generation,
                    "Threshold worker queue is unavailable",
                );
                reject_worker_submission(request, diagnostics);
                false
            }
        },
        Err(error) => {
            reject_actor_request(request, diagnostics, error);
            false
        }
    }
}

pub(super) fn begin_threshold_load(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    render_document: &Option<Arc<RenderDocument>>,
    refresh: bool,
    diagnostics: &ActorDiagnostics,
) -> bool {
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return false;
    }
    let Some(document) = render_document.as_ref().cloned() else {
        reject_actor_request(
            request,
            diagnostics,
            ControlError::new(
                ControlErrorKind::NotReady,
                "threshold preview requires an installed document resource",
            ),
        );
        return false;
    };
    let spec = match model.prepare_threshold_preview_load(request.command.params(), refresh) {
        Ok(spec) => spec,
        Err(error) => {
            reject_actor_request(request, diagnostics, error);
            return false;
        }
    };
    match load_job_tx.try_send(LoadJob::ThresholdLoad {
        request,
        document,
        spec,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
            true
        }
        Err(error) => {
            let LoadJob::ThresholdLoad { request, spec, .. } = error.into_inner() else {
                unreachable!("threshold load submission returns its own job")
            };
            model.fail_threshold_operation(
                spec.document_generation,
                spec.operation_generation,
                "Threshold worker queue is unavailable",
            );
            reject_worker_submission(request, diagnostics);
            false
        }
    }
}

pub(super) fn begin_threshold_apply(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) -> bool {
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return false;
    }
    let spec = match model.prepare_threshold_preview_apply() {
        Ok(spec) => spec,
        Err(error) => {
            reject_actor_request(request, diagnostics, error);
            return false;
        }
    };
    match load_job_tx.try_send(LoadJob::ThresholdApply { request, spec }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
            true
        }
        Err(error) => {
            let LoadJob::ThresholdApply { request, spec } = error.into_inner() else {
                unreachable!("threshold apply submission returns its own job")
            };
            model.fail_threshold_operation(
                spec.document_generation,
                spec.operation_generation,
                "Threshold worker queue is unavailable",
            );
            reject_worker_submission(request, diagnostics);
            false
        }
    }
}
