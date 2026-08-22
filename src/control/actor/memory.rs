use super::*;

pub(super) fn begin_memory_pin(
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
                "memory pinning requires an installed document resource",
            ),
        );
        return false;
    };
    let spec = match model.prepare_memory_pin(request.command.params()) {
        Ok(spec) => spec,
        Err(error) => {
            reject_actor_request(request, diagnostics, error);
            return false;
        }
    };
    if document.generation != spec.document_generation {
        model.fail_memory_pin(&spec, "Document changed before memory pinning started");
        reject_actor_request(
            request,
            diagnostics,
            ControlError::new(
                ControlErrorKind::Conflict,
                "document changed before memory pinning started",
            ),
        );
        return false;
    }
    match load_job_tx.try_send(LoadJob::MemoryPin {
        request,
        document,
        spec,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
            true
        }
        Err(error) => {
            let LoadJob::MemoryPin { request, spec, .. } = error.into_inner() else {
                unreachable!("memory pin submission returns its own job")
            };
            model.fail_memory_pin(&spec, "Memory pin worker queue is unavailable");
            reject_worker_submission(request, diagnostics);
            false
        }
    }
}
