use super::*;

pub(super) fn begin_channel_intensity(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    render_document: &Option<Arc<RenderDocument>>,
    diagnostics: &ActorDiagnostics,
) -> bool {
    let Some(document) = render_document.as_ref().cloned() else {
        diagnostics
            .rejected_requests
            .fetch_add(1, Ordering::Relaxed);
        diagnostics.record_reply_time(request.command.queue_age());
        let _ = request.reply.send(Err(ControlError::new(
            ControlErrorKind::NotReady,
            "channel intensity statistics require an opened dataset resource",
        )));
        return false;
    };
    let spec = match model.channel_intensity_spec(document.dataset(), request.command.params()) {
        Ok(spec) => spec,
        Err(error) => {
            diagnostics
                .rejected_requests
                .fetch_add(1, Ordering::Relaxed);
            diagnostics.record_reply_time(request.command.queue_age());
            let _ = request.reply.send(Err(error));
            return false;
        }
    };
    let generation = model.document_generation();
    let operation_generation = model.begin_channel_intensity_operation(spec.client_request_id);
    match load_job_tx.try_send(LoadJob::ChannelIntensity {
        generation,
        operation_generation,
        request,
        document,
        spec,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
            true
        }
        Err(crossbeam_channel::TrySendError::Full(LoadJob::ChannelIntensity {
            request, ..
        })) => {
            let _ = model.fail_channel_intensity_operation(
                generation,
                operation_generation,
                "Odon's resource worker queue is full; retry later",
            );
            diagnostics
                .rejected_requests
                .fetch_add(1, Ordering::Relaxed);
            diagnostics.record_reply_time(request.command.queue_age());
            let _ = request.reply.send(Err(ControlError::new(
                ControlErrorKind::NotReady,
                "Odon's resource worker queue is full; retry later",
            )));
            true
        }
        Err(crossbeam_channel::TrySendError::Disconnected(LoadJob::ChannelIntensity {
            request,
            ..
        })) => {
            let _ = model.fail_channel_intensity_operation(
                generation,
                operation_generation,
                "Odon's resource workers are unavailable",
            );
            diagnostics
                .rejected_requests
                .fetch_add(1, Ordering::Relaxed);
            diagnostics.record_reply_time(request.command.queue_age());
            let _ = request.reply.send(Err(ControlError::new(
                ControlErrorKind::NotReady,
                "Odon's resource workers are unavailable",
            )));
            true
        }
        Err(_) => unreachable!("submitted channel intensity job changed variant"),
    }
}

pub(super) fn begin_auto_contrast(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    render_document: &Option<Arc<RenderDocument>>,
    diagnostics: &ActorDiagnostics,
) -> bool {
    let Some(document) = render_document.as_ref().cloned() else {
        reject_actor_request(
            request,
            diagnostics,
            ControlError::new(
                ControlErrorKind::NotReady,
                "automatic contrast requires an opened dataset resource",
            ),
        );
        return false;
    };
    let mut candidate = model.clone();
    let spec = match candidate.prepare_auto_contrast(document.dataset(), request.command.params()) {
        Ok(spec) => spec,
        Err(error) => {
            reject_actor_request(request, diagnostics, error);
            return false;
        }
    };
    if !candidate.mark_auto_contrast_started(&spec) {
        reject_actor_request(
            request,
            diagnostics,
            ControlError::new(
                ControlErrorKind::Conflict,
                "automatic contrast was superseded before it started",
            ),
        );
        return false;
    }
    match load_job_tx.try_send(LoadJob::AutoContrast {
        spec: spec.clone(),
        request: Some(request),
        document,
    }) {
        Ok(()) => {
            *model = candidate;
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
            true
        }
        Err(crossbeam_channel::TrySendError::Full(LoadJob::AutoContrast {
            request: Some(request),
            ..
        }))
        | Err(crossbeam_channel::TrySendError::Disconnected(LoadJob::AutoContrast {
            request: Some(request),
            ..
        })) => {
            let message = "Odon's resource workers could not start automatic contrast";
            reject_actor_request(
                request,
                diagnostics,
                ControlError::new(ControlErrorKind::NotReady, message),
            );
            false
        }
        Err(_) => unreachable!("submitted automatic contrast job changed variant"),
    }
}

pub(super) fn enqueue_auto_contrast_on_open(
    model: &mut AppModel,
    render_document: &Option<Arc<RenderDocument>>,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) -> bool {
    let Some(document) = render_document.as_ref().cloned() else {
        return false;
    };
    let spec = match model.auto_contrast_on_open_spec(document.dataset()) {
        Ok(Some(spec)) => spec,
        Ok(None) => return false,
        Err(error) => {
            eprintln!(
                "could not prepare automatic contrast on open: {}",
                error.message
            );
            return model.fail_auto_contrast_on_open_preparation(error.message);
        }
    };
    match load_job_tx.try_send(LoadJob::AutoContrast {
        spec: spec.clone(),
        request: None,
        document,
    }) {
        Ok(()) => {
            if model.mark_auto_contrast_started(&spec) {
                diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
                true
            } else {
                false
            }
        }
        Err(crossbeam_channel::TrySendError::Full(_)) => false,
        Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
            if model.mark_auto_contrast_started(&spec) {
                let _ = model.fail_auto_contrast(&spec, "Odon's resource workers are unavailable");
                true
            } else {
                false
            }
        }
    }
}
