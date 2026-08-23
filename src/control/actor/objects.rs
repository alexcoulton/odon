use super::*;

pub(super) fn begin_object_filter_evaluation(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return;
    }
    let (
        document_generation,
        resource_generation,
        operation_generation,
        viewport_id,
        target,
        expected_presentation_revision,
        resource,
        filter_model,
    ) = match model.begin_object_filter_evaluation(request.command.params()) {
        Ok(work) => work,
        Err(error) => {
            reject_actor_request(request, diagnostics, error);
            return;
        }
    };
    match load_job_tx.try_send(LoadJob::ObjectFilter {
        document_generation,
        resource_generation,
        operation_generation,
        viewport_id: viewport_id.clone(),
        target,
        expected_presentation_revision,
        request,
        resource,
        model: filter_model,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let LoadJob::ObjectFilter { request, .. } = error.into_inner() else {
                unreachable!("object-filter submission returns its own job")
            };
            model.fail_object_filter_for_generation(
                &viewport_id,
                target,
                operation_generation,
                "Object filter worker queue is unavailable",
            );
            reject_worker_submission(request, diagnostics);
        }
    }
}

pub(super) fn begin_object_selection_filter_evaluation(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return;
    }
    let (
        document_generation,
        resource_generation,
        selection_generation,
        operation_generation,
        target,
        resource,
        filter_model,
        mode,
        limit,
    ) = match model.begin_object_selection_filter_evaluation(request.command.params()) {
        Ok(work) => work,
        Err(error) => {
            reject_actor_request(request, diagnostics, error);
            return;
        }
    };
    match load_job_tx.try_send(LoadJob::ObjectSelectionFilter {
        document_generation,
        resource_generation,
        selection_generation,
        operation_generation,
        target,
        request,
        resource,
        model: filter_model,
        mode,
        limit,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let LoadJob::ObjectSelectionFilter { request, .. } = error.into_inner() else {
                unreachable!("object selection-filter submission returns its own job")
            };
            model.fail_object_selection_filter_for_generation(
                target,
                operation_generation,
                "Object selection-filter worker queue is unavailable",
            );
            reject_worker_submission(request, diagnostics);
        }
    }
}
