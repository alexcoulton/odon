use super::*;

pub(super) struct CompletionContext<'a> {
    pub(super) model: &'a mut AppModel,
    pub(super) render_document: &'a mut Option<Arc<RenderDocument>>,
    pub(super) remote_session: &'a mut RemoteSessionState,
    pub(super) resource_registry: &'a ResourceRegistry,
    pub(super) presentation_tx: &'a Sender<RenderProjection>,
    pub(super) presentation_coalesce_rx: &'a Receiver<RenderProjection>,
    pub(super) platform_effect_tx: &'a Sender<PlatformEffect>,
    pub(super) load_job_tx: &'a Sender<LoadJob>,
    pub(super) wake_ui: &'a UiWake,
    pub(super) diagnostics: &'a ActorDiagnostics,
}

pub(super) fn finish_load(
    model: &mut AppModel,
    render_document: &mut Option<Arc<RenderDocument>>,
    completion: LoadCompletion,
    remote_session: &mut RemoteSessionState,
    resource_registry: &ResourceRegistry,
    presentation_tx: &Sender<RenderProjection>,
    presentation_coalesce_rx: &Receiver<RenderProjection>,
    platform_effect_tx: &Sender<PlatformEffect>,
    load_job_tx: &Sender<LoadJob>,
    wake_ui: &UiWake,
    diagnostics: &ActorDiagnostics,
) {
    diagnostics
        .workers_completed
        .fetch_add(1, Ordering::Relaxed);
    let domain = completion.domain();
    let context = CompletionContext {
        model,
        render_document,
        remote_session,
        resource_registry,
        presentation_tx,
        presentation_coalesce_rx,
        platform_effect_tx,
        load_job_tx,
        wake_ui,
        diagnostics,
    };
    match domain {
        CompletionDomain::Opening => completion_opening::finish(completion, context),
        CompletionDomain::Project => completion_project::finish(completion, context),
        CompletionDomain::Resources => completion_resources::finish(completion, context),
        CompletionDomain::Objects => completion_objects::finish(completion, context),
        CompletionDomain::Masks => completion_masks::finish(completion, context),
        CompletionDomain::Mosaic => completion_mosaic::finish(completion, context),
    }
}

pub(super) fn request_is_cancelled(request: &OdonControlRequest) -> bool {
    request
        .task_id
        .as_deref()
        .and_then(|task_id| request.task_registry.get(task_id).ok())
        .is_some_and(|task| task.state == TaskState::Cancelled)
}

pub(super) fn reject_cancelled_request(
    request: OdonControlRequest,
    diagnostics: &ActorDiagnostics,
    operation: &str,
) {
    diagnostics.record_reply_time(request.command.queue_age());
    let _ = request.reply.send(Err(ControlError::new(
        ControlErrorKind::Cancelled,
        format!("{operation} was cancelled"),
    )));
}

pub(super) fn reject_stale_project_worker(
    request: OdonControlRequest,
    diagnostics: &ActorDiagnostics,
    operation: &str,
) {
    diagnostics
        .stale_worker_completions
        .fetch_add(1, Ordering::Relaxed);
    diagnostics.record_reply_time(request.command.queue_age());
    let _ = request.reply.send(Err(ControlError::new(
        ControlErrorKind::Conflict,
        format!("{operation} was superseded by a newer project transaction"),
    )));
}

pub(super) fn reject_actor_request(
    request: OdonControlRequest,
    diagnostics: &ActorDiagnostics,
    error: ControlError,
) {
    diagnostics
        .rejected_requests
        .fetch_add(1, Ordering::Relaxed);
    diagnostics.record_reply_time(request.command.queue_age());
    let _ = request.reply.send(Err(error));
}

pub(super) fn fail_project_worker(
    model: &mut AppModel,
    generation: u64,
    request: OdonControlRequest,
    diagnostics: &ActorDiagnostics,
    operation: &str,
    path: &std::path::Path,
    error: anyhow::Error,
) {
    let message = format!("failed to {operation} {}: {error}", path.display());
    if model.fail_project_operation(generation, &message) {
        reject_actor_request(
            request,
            diagnostics,
            ControlError::new(ControlErrorKind::Application, message)
                .with_data(json!({"path": path.to_string_lossy()})),
        );
    } else {
        reject_stale_project_worker(request, diagnostics, operation);
    }
}
