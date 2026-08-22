use super::*;

pub(super) fn begin_measurement(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    render_document: &Option<Arc<RenderDocument>>,
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
                "measurements require an installed document resource",
            ),
        );
        return false;
    };
    let spec = match model.prepare_measurement(request.command.params()) {
        Ok(spec) => spec,
        Err(error) => {
            reject_actor_request(request, diagnostics, error);
            return false;
        }
    };
    match load_job_tx.try_send(LoadJob::Measurement {
        request,
        document,
        spec,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
            true
        }
        Err(error) => {
            let LoadJob::Measurement { request, spec, .. } = error.into_inner() else {
                unreachable!()
            };
            model.fail_measurement(&spec, "Measurement worker queue is unavailable");
            reject_worker_submission(request, diagnostics);
            false
        }
    }
}
