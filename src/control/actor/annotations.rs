use super::*;

pub(super) fn begin_annotation_load(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) -> bool {
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return false;
    }
    let id = match AppModel::annotation_id(request.command.params()) {
        Ok(id) => id,
        Err(error) => {
            reject_actor_request(request, diagnostics, error);
            return false;
        }
    };
    let load_dataset = request.command.method() != "viewer.annotations.source.inspect";
    let spec = match model.prepare_annotation_load(id, request.command.params(), load_dataset) {
        Ok(spec) => spec,
        Err(error) => {
            reject_actor_request(request, diagnostics, error);
            return false;
        }
    };
    match load_job_tx.try_send(LoadJob::Annotations {
        request: Some(request),
        spec,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
            true
        }
        Err(error) => {
            let LoadJob::Annotations { request, spec } = error.into_inner() else {
                unreachable!("annotation submission returns its own job")
            };
            model.fail_annotation_load(&spec, "Annotation worker queue is unavailable".to_string());
            reject_worker_submission(
                request.expect("API annotation jobs retain their request"),
                diagnostics,
            );
            false
        }
    }
}
