use super::*;

pub(super) fn begin_project_view_apply(
    model: &mut AppModel,
    request: OdonControlRequest,
    spec: ProjectViewApplySpec,
    load_job_tx: &Sender<LoadJob>,
    render_document: &Option<Arc<RenderDocument>>,
    diagnostics: &ActorDiagnostics,
) {
    let Some(document) = render_document.as_ref().cloned() else {
        model.fail_project_view_apply(&spec, "saved-view resources require an opened document");
        reject_actor_request(
            request,
            diagnostics,
            ControlError::new(
                ControlErrorKind::NotReady,
                "saved-view resources require an actor-owned opened document",
            ),
        );
        return;
    };
    match load_job_tx.try_send(LoadJob::ProjectViewApply {
        request,
        spec,
        document,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let LoadJob::ProjectViewApply { request, spec, .. } = error.into_inner() else {
                unreachable!("saved-view submission returns its own job")
            };
            model.fail_project_view_apply(&spec, "saved-view resource worker queue is unavailable");
            reject_worker_submission(request, diagnostics);
        }
    }
}
